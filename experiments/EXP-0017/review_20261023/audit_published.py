"""Read and audit the immutable published ABBA snapshot; no SSH or GPU access."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import csv
import datetime
import hashlib
import json
import statistics
import subprocess
import sys

BASE=Path(__file__).resolve().parent
COMMIT='8bcfbb98dececb5b6f593c1504f1c67d1defd088'
REPO='yyq0210/MiniCPM-Infer-Sys-Opt'
PREFIX='experiments/EXP-0017/'
DEST=BASE/'snapshot'
def sha(data):return hashlib.sha256(data).hexdigest()
def read(p):return json.loads(p.read_text())
def download(url):
    return subprocess.check_output(['curl','--silent','--show-error','--fail','--location','--connect-timeout','15','--max-time','40',url],timeout=45)
def csvrows(p):
    with p.open() as f:return list(csv.DictReader(f))
def main():
    # The public tree endpoint returned HTTP403. Use the already-public raw
    # manifests at the independently resolved immutable commit, without auth.
    def get(rel):
        assert '..' not in Path(rel).parts and not rel.startswith('/')
        target=DEST/rel
        data=target.read_bytes() if '--offline' in sys.argv else download(f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/{PREFIX}{rel}')
        assert len(data)<8*1024*1024
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        return rel,sha(data)
    hashes=dict(get(rel) for rel in ['RESULTS.json','STATUS.json','arms.json','script_hashes.json'])
    result=read(DEST/'RESULTS.json');status=read(DEST/'STATUS.json');arms=read(DEST/'arms.json')
    chosen=list(read(DEST/'script_hashes.json'))
    for item in result['sessions']:
        prefix='artifacts/'+item['run_name']+'/'
        rel=prefix+'source_hashes.json';key,digest=get(rel);hashes[key]=digest
        chosen.extend(prefix+path for path in read(DEST/rel))
    with ThreadPoolExecutor(max_workers=4) as pool:hashes.update(dict(pool.map(get,chosen)))
    assert result['status']=='complete_ABBA' and status['stage']=='complete'
    assert result['order']==arms['order']==['A','B','B','A']
    assert result['sessions']==status['completed_sessions']
    for name,digest in read(DEST/'script_hashes.json').items():assert sha((DEST/name).read_bytes())==digest,name
    reviewed=[]
    for i,item in enumerate(result['sessions']):
        arm=item['arm'];assert arm==arms['order'][i] and item['index']==i+1
        assert item['commits']==arms['commits'][arm]
        r=DEST/'artifacts'/item['run_name'];folder=r/'candidate_benchmark'
        assert read(r/'result.json')==item
        state=read(r/'status.json');assert state['stage']=='performance_case_complete' and state['commits']==item['commits']
        assert read(r/'environment_check.json')['candidate_commits']==item['commits']
        raw_path,=(folder/'benchmark_results').glob('raw_runs_*.csv')
        summary_path,=(folder/'benchmark_results').glob('summary_*.csv')
        raw,summary=csvrows(raw_path),csvrows(summary_path)
        assert len(raw)==4 and len(summary)==1 and summary[0]==item['summary']
        for row in raw:
            assert row['Run Status']=='SUCCESS'
            assert [int(float(row[k])) for k in ['Prefill','Decode','Conc','Num Prompts']]==[4096,1024,64,256]
            assert int(float(row['Successful Requests']))==256
            assert int(float(row['Total Input Tokens']))==1048576
            assert int(float(row['Total Output Tokens']))==262144
        for key in result['metrics']:
            assert abs(float(summary[0][key])-statistics.mean(float(x[key]) for x in raw[1:]))<=.011
        assert item['retained_total_tokens_s']==[float(x['Total tok/s']) for x in raw[1:]]
        assert abs(item['cv_percent']-statistics.stdev(item['retained_total_tokens_s'])/statistics.mean(item['retained_total_tokens_s'])*100)<1e-8
        log=(folder/'benchmark.log').read_text();assert log.count('Failed requests:                         0')==4
        assert read(folder/'smoke.json')['usage']['completion_tokens']>0
        server=(folder/'server.log').read_text()
        assert server.count('LeanScale GQA active:')==42 and server.count('LeanScale fused RoPE+KV active:')==42
        assert 'Capturing CUDA graphs (decode, FULL)' in server and '35/35' in server
        assert 'KVReservationAsyncScheduler' not in server and 'KV growth admission enabled=True' not in server
        route=read(folder/'route_evidence.json')
        assert route['status']=='passed' and route['gqa_layers']==route['fused_runtime_layers']==42
        assert route['old_cache_updates_in_fused_graph']==0 and route['kernel_counts']['decode'] and route['kernel_counts']['prefill']
        for k in route['kernels']:
            assert k['num_warps']==4 and k['num_stages']==(2 if arm=='A' and k['kind']=='prefill' else 1)
            assert (k['logical_query_rows'],k['shared'],k['page_table_load_elements'])==((8,18432,2) if k['kind']=='decode' else (64,32768,32))
        # One compiled computation_graph.py can back all 35 captured sizes.
        assert route['graphs']
        assert all(g['counts']==dict(fl_rope_kv_dispatch=42,fl_rotary_embedding_dispatch=0,unified_kv_cache_update=0) for g in route['graphs'])
        source_hashes=read(r/'source_hashes.json');verified={}
        for p in [raw_path,summary_path,folder/'route_evidence.json',r/'result.json',r/'status.json']:
            rel=str(p.relative_to(r));data=p.read_bytes();expected=source_hashes[rel]
            candidates=[data,data.replace(b'\n',b'\r\n')] if p.suffix=='.csv' else [data]
            assert expected in [sha(x) for x in candidates],rel
            verified[rel]=expected
        reviewed.append(dict(index=i+1,arm=arm,run_name=item['run_name'],metrics=item['summary'],
            retained=item['retained_total_tokens_s'],cv_percent=item['cv_percent'],verified_raw_source_hashes=verified,
            kernel_counts=route['kernel_counts'],compiled_graph_source_files=len(route['graphs']),
            decode_graph_capture_sizes=35,actual_kernel_files_rehashed=False))
    for metric,v in result['metrics'].items():
        vals=[float(s['summary'][metric]) for s in result['sessions']]
        a,b=statistics.mean([vals[0],vals[3]]),statistics.mean(vals[1:3])
        expected=dict(A=a,B=b,change_percent=(b/a-1)*100,pair1_change_percent=(vals[1]/vals[0]-1)*100,
            pair2_change_percent=(vals[2]/vals[3]-1)*100,A_endpoint_change_percent=(vals[3]/vals[0]-1)*100,
            B_repeat_change_percent=(vals[2]/vals[1]-1)*100)
        assert all(abs(v[k]-x)<1e-8 for k,x in expected.items()),metric
    audit=dict(status='published_snapshot_audited',checked_at=datetime.datetime.now().astimezone().isoformat(),
        repository=REPO,commit=COMMIT,completed_at=status['updated_at'],metrics=result['metrics'],sessions=reviewed,
        downloaded_files_at_fixed_commit=len(hashes),downloaded_sha256=hashes,
        limitations=['SSH public-key authentication failed on 2026-10-23; current server worktrees and compiled JSON/TTIR/Graph files could not be independently rehashed.',
            'Published source CSV hashes and official round/token/summary consistency verified; route evidence assertions verified, not their unavailable raw cache contents.',
            'A1 retained-round CV14.14%; A1 to A2 mean shift5.78%; paired gain7.90% and1.27%. Do not assert stable4.49% causal gain or statistical significance.',
            'Only two sessions per arm; no removal of A1 slow round; original EXP16 remains independent.'],
        selected_for_serving=False,full_task_accuracy='deferred_by_user')
    (BASE/'published_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
    lines=['# 4K ABBA 复测：发布结果离线审核','','审核日期：2026-10-23。GitHub固定提交：`'+COMMIT+'`。',
        '全部4个独立服务会话、16个轮次完成；各会话严格弃首轮，剩余12轮保留，零失败、请求与token计数正确。','',
        '| 指标 | 旧版A | 新版B | 变化 |','|---|---:|---:|---:|']
    for metric,v in result['metrics'].items():lines.append(f"| {metric} | {v['A']:.3f} | {v['B']:.3f} | {v['change_percent']:+.3f}% |")
    lines+=['','两个配对吞吐增幅为+7.90%和+1.27%；旧版A首尾均值变化+5.78%，新版B两次均值变化-0.72%。',
        'A1保留轮为2688.62/3454.37/3494.74，CV14.14%，该慢轮原样计入。整体有收益方向的证据，但不能把均值+4.49%表述为稳定因果提升或统计显著。',
        '已从固定Git提交下载并记录文件SHA256，核对冻结脚本清单、原始CSV源哈希（兼容归档换行标准化）、五项指标均值、42层与35图的归档接入证据。',
        '服务器SSH公钥认证失败，无法重新读取运行工作树及真实编译JSON/TTIR/Graph文件；这一部分独立审核待恢复后补齐。暂未更新性能版本选择，也未启动新GPU任务。',
        '后续：恢复访问并确认当天运行窗口→补齐实际缓存/源码审核→记录性能版本选择→当前版本4K阶段/热点诊断。',
        '完整机器可读审核：[published_audit.json](published_audit.json)。']
    (BASE/'REVIEW.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=audit['status'],completed_at=audit['completed_at'],files=len(hashes),metrics=result['metrics']),ensure_ascii=False))
if __name__=='__main__':main()
