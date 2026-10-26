"""Independently audit all paired evidence, then select a working performance baseline."""
from pathlib import Path
from datetime import datetime
import csv
import fcntl
import hashlib
import importlib.util
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import tempfile

ROOT=Path('/workspace/LeanScale');REPO=ROOT/'MiniCPM-Infer-Sys-Opt';EXP=REPO/'experiments/EXP-0021';ART=ROOT/'artifacts/EXP-0021'
DEST=EXP/'review_20261026'
KEYS=['Total tok/s','Output tok/s','Mean TTFT (ms)','Mean TPOT (ms)','Mean ITL (ms)']
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open() as f:return list(csv.DictReader(f))
def git(p,*args):return subprocess.check_output(['git','-C',str(p),*args],text=True,stderr=subprocess.STDOUT,timeout=120).strip()
def close(a,b):assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-9),(a,b)
def sanitize(t):
    t=re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b','[IP_REDACTED]',t)
    return re.sub(r'uuid=[^,)]+','uuid=[REDACTED]',t).replace('\r\n','\n')

def check_case(folder,prefill,count):
    rawp,=(folder/'benchmark_results').glob('raw_runs_*.csv');sump,=(folder/'benchmark_results').glob('summary_*.csv')
    raw,summary=rows(rawp),rows(sump);assert len(raw)==4 and len(summary)==1
    assert (folder/'benchmark.log').read_text().count('Failed requests:                         0')==4
    for r in raw:
        assert r['Run Status']=='SUCCESS'
        assert tuple(int(float(r[k])) for k in ['Prefill','Decode','Conc','Num Prompts'])==(prefill,1024,64,count)
        assert int(float(r['Successful Requests']))==count
        assert int(float(r['Total Input Tokens']))==prefill*count and int(float(r['Total Output Tokens']))==1024*count
    for k in KEYS:assert abs(float(summary[0][k])-statistics.mean(float(r[k]) for r in raw[1:]))<=.011
    return raw,summary[0],dict(raw_sha256=sha(rawp),summary_sha256=sha(sump))

def main():
    assert not DEST.exists(),'Do not overwrite an audit'
    state=read(ART/'status.json');result=read(EXP/'RESULTS.json');publication=read(ART/'publication.json')
    assert state['stage']=='complete' and result['status']=='complete_ABBA'
    assert publication['remote_verified'] and publication['status']=='pushed'
    git(REPO,'merge-base','--is-ancestor',publication['commit'],'HEAD')
    assert result['sessions']==state['completed_sessions'] and result['order']==['A','B','B','A']
    sys.path.insert(0,str(EXP));from common import COMMITS,verify_sources
    from route_checks import verify_route
    for arm in COMMITS:verify_sources(arm)
    for name,h in read(EXP/'source_provenance.json').items():assert sha(EXP/name)==h
    proc=Path('/proc')/str(state['pid'])
    if proc.exists():
        f=(proc/'stat').read_text().rsplit(') ',1)[1].split();assert f[0]=='Z' or f[19]!=str(state['process_start_ticks'])
    checked=[];previous_end=None;recomputed=[]
    with tempfile.TemporaryDirectory(prefix='exp21_audit_') as temp:
        for i,item in enumerate(result['sessions']):
            arm=result['order'][i];run=ART/item['run_name'];folder=run/'candidate_benchmark'
            assert item['index']==i+1 and item['arm']==arm and item['commits']==COMMITS[arm]
            s=read(run/'status.json');pub=read(run/'publication.json')
            assert s['stage']=='performance_case_complete' and s['commits']==COMMITS[arm] and not (run/'error.json').exists()
            assert pub['remote_verified'] and pub['status']=='pushed';git(REPO,'merge-base','--is-ancestor',pub['commit'],'HEAD')
            history=[json.loads(t) for t in (run/'stage_history.jsonl').read_text().splitlines()]
            times=[datetime.fromisoformat(x['updated_at']) for x in history]
            assert times==sorted(times) and times[-1]<=datetime.fromisoformat(s['planned_finish_before'])
            assert previous_end is None or previous_end<times[0];previous_end=times[-1]
            output=Path(temp)/(str(i)+'.json');verify_route(run,arm,output)
            actual=read(output);saved=read(folder/'route_evidence.json');assert actual==saved
            assert saved['kernel_counts']=={'decode':64,'prefill':64}
            for entry in saved['kernels']:
                p=run/entry['path'];assert sha(p)==entry['sha256'] and sha(p.with_suffix('.ttir'))==entry['ttir_sha256']
            for entry in saved['graphs']:assert sha(run/entry['path'])==entry['sha256']
            archive=EXP/'artifacts'/run.name;manifest=read(archive/'source_hashes.json')
            for name,digest in manifest.items():
                src=run/name;assert sha(src)==digest,name
                assert (archive/name).read_text()==sanitize(src.read_text(errors='replace')),name
            raw,summary,hashes=check_case(folder,4096,256)
            assert summary==item['summary'] and summary==read(run/'result.json')['summary']
            retained=[float(x['Total tok/s']) for x in raw[1:]];assert retained==item['retained_total_tokens_s']
            close(statistics.stdev(retained)/statistics.mean(retained)*100,item['cv_percent'])
            recomputed.append(summary)
            checked.append(dict(index=i+1,arm=arm,run_name=run.name,raw_rounds=4,successes=1024,
                retained_total_tokens_s=retained,cv_percent=item['cv_percent'],kernel_counts=saved['kernel_counts'],
                route_sha256=sha(folder/'route_evidence.json'),archive_files=len(manifest),**hashes))
        metrics={}
        for key in KEYS:
            v=[float(x[key]) for x in recomputed];a=statistics.mean([v[0],v[3]]);b=statistics.mean(v[1:3])
            m=dict(A=a,B=b,change_percent=(b/a-1)*100,pair1_change_percent=(v[1]/v[0]-1)*100,
              pair2_change_percent=(v[2]/v[3]-1)*100,A_endpoint_change_percent=(v[3]/v[0]-1)*100,B_repeat_change_percent=(v[2]/v[1]-1)*100)
            for k,value in m.items():close(value,result['metrics'][key][k]);close(value,state['metrics'][key][k])
            metrics[key]=m
        # Rehash both formal candidate cases again before making a selection.
        formal=[]
        for case,prefill,count in [('4k',4096,256),('16k',16384,128)]:
            audit=read(REPO/f'experiments/EXP-0020/review_{case}.json')
            assert audit['status']=='complete_case_audited' and audit['candidate_commits']==COMMITS['B']
            run=ROOT/'artifacts/EXP-0020'/audit['run_name'];folder=run/'candidate_benchmark'
            output=Path(temp)/(case+'.json');verify_route(run,'B',output);saved=read(folder/'route_evidence.json');assert read(output)==saved
            for entry in saved['kernels']:
                p=run/entry['path'];assert sha(p)==entry['sha256'] and sha(p.with_suffix('.ttir'))==entry['ttir_sha256']
            for entry in saved['graphs']:assert sha(run/entry['path'])==entry['sha256']
            _,summary,hashes=check_case(folder,prefill,count)
            assert hashes['raw_sha256']==audit['raw_sha256'] and hashes['summary_sha256']==audit['summary_sha256']
            assert sha(folder/'route_evidence.json')==audit['route_sha256']
            for key in KEYS:close(float(summary[key]),audit['metrics'][key]['prefill_page_broadcast'])
            audit.update(selected_for_serving=True,compiled_files_rehashed_on_selection=dict(kernels=len(saved['kernels']),graph_source_files=len(saved['graphs'])))
            formal.append(audit)
    assert all(x['metrics']['Total tok/s']['change_percent']>0 for x in formal)
    assert all(metrics['Total tok/s'][p]>0 for p in ['pair1_change_percent','pair2_change_percent'])
    assert all(metrics[k][p]<0 for k in ['Mean TTFT (ms)','Mean TPOT (ms)'] for p in ['pair1_change_percent','pair2_change_percent'])
    active_path=ROOT/'artifacts/active_inference_plan.json';previous=read(active_path)
    assert previous['selected_serving_commits']==COMMITS['A']
    report=dict(status='complete_paired_audit',checked_at=datetime.now().astimezone().isoformat(),publication=publication,
      session_count=4,raw_rounds=16,successful_requests=4096,sessions=checked,metrics=metrics,
      limitations='A1 retained CV14.499%, A endpoint+11.367%; retain all slow rounds. Mean+9.008% is drift-sensitive, not stable causal gain. Two sessions per arm, no significance claim.',
      decision='Select EXP20 as working performance baseline from positive formal4K/16K and both paired throughput/latency directions; full task accuracy deferred')
    selection=dict(selected_experiment='EXP-0020',selected_serving_commits=COMMITS['B'],previous_selection=previous,
      selected_at=report['checked_at'],scope='Working performance baseline; full-model task accuracy deferred',
      basis=report['decision'],limitations=report['limitations'],case_results=formal,paired_review='../EXP-0021/review_20261026/audit.json',
      paired_metrics=metrics,scheduler_candidate='rejected_by_user',next_direction='4K small-M native projection path diagnosis')
    with (ROOT/'artifacts/docs-publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);assert not git(REPO,'diff','--cached','--name-only')
        DEST.mkdir();write(DEST/'audit.json',report);shutil.copy2(Path(__file__),DEST/'audit_select.py')
        write(REPO/'experiments/EXP-0020/selection.json',selection)
        lines=['# 页号广播配对审核与工作基线选择','','2026-10-26。四个独立服务会话共16轮、4096请求全部成功；每会话弃首，完整保留所有慢轮与暖机轮。','',
          '| 会话 | 版本 | 保留轮 total tok/s | 会话均值 | CV |','|---|---|---|---:|---:|']
        for x,raw in zip(checked,recomputed):lines.append(f"| {x['index']} | {x['arm']} | {x['retained_total_tokens_s']} | {float(raw['Total tok/s']):.2f} | {x['cv_percent']:.3f}% |")
        lines+=['','A=EXP16，B=EXP20。两配对吞吐分别+14.718%与+3.881%；A首尾+11.367%，B重复+0.847%。A1存在明显慢轮，保留轮CV14.499%。',
          'A会话均值3268.265、B3562.680，观测+9.008%；这个汇总受漂移和慢轮影响，不能报告为稳定的9%因果收益。只有每arm两个独立会话，ABBA仅平衡近似线性漂移，不声明统计显著。',
          'TTFT两配对下降11.429%/8.937%，TPOT下降13.476%/3.872%，方向均有利于B。正式4K历史观测+2.951%，不与配对轮次混合计算。',
          '独立审核重新计算全部16轮、5个核心指标、CV及两配对/端点变化；四个会话各128个实际attention编译核与融合Graph逐一核验，归档内容与原文件哈希复核。',
          '另对正式4K128个、16K67个编译核重新核验，原始CSV和版本通过。两组正式性能及两配对方向支持将EXP20选为后续实验工作基线。选择不等于完整任务精度达标，MATH500仍按用户要求暂缓。','',
          '| 正式负载 | 新工作基线 total tok/s | 相对EXP16 | 相对原始 |','|---|---:|---:|---:|']
        for x in formal:
            m=x['metrics']['Total tok/s'];lines.append(f"| {x['case']} | {m['prefill_page_broadcast']:.2f} | {m['change_percent']:+.3f}% | {x['comparison_original']['throughput_change_percent']:+.3f}% |")
        lines+=['','16K队列和抢占/KV容量问题未解决；下一阶段重点4K小M投影，先核对真实库GEMM路径和完整子图开销。',
          '[审核数据](../experiments/EXP-0021/review_20261026/audit.json) / [工作基线选择](../experiments/EXP-0020/selection.json)。']
        (REPO/'docs/gqa_page_broadcast_selection_20261026.md').write_text('\n'.join(lines)+'\n')
        with (REPO/'docs/optimization_journal.md').open('a') as f:f.write('\n\n### 2026-10-26：EXP21固定ABBA审核完成，工作基线选择EXP20\n\n全部16轮/4096请求成功，独立复核真实编译与数据。两配对吞吐+14.72%/+3.88%，但A1慢轮CV14.50%、A端点漂移11.37%，不把均值+9.01%说成固定因果收益。正式4K3579.12、16K2541.43及两配对延迟方向支持选择EXP20为下一阶段性能基线。原始数据全部保留，任务精度仍未测，随后重点小M投影。\n')
        with (REPO/'experiments/EXP-0020/PR_DRAFT.md').open('a') as f:f.write('\n\n配对审核更新：固定ABBA16轮完整，两配对吞吐+14.72%/+3.88%，但A1慢轮和A端点漂移较大，不宣称稳定+9.01%。综合正式两组及配对吞吐/延迟方向选择EXP20为工作性能基线；完整MATH500未测，保持Draft。详见docs/gqa_page_broadcast_selection_20261026.md。\n')
        paths=['experiments/EXP-0021/review_20261026','experiments/EXP-0020/selection.json','docs/gqa_page_broadcast_selection_20261026.md','docs/optimization_journal.md','experiments/EXP-0020/PR_DRAFT.md']
        git(REPO,'add','--',*paths);git(REPO,'diff','--cached','--check');git(REPO,'commit','--only','-m','experiments: audit complete page-broadcast ABBA and select performance baseline','--',*paths)
        head=git(REPO,'rev-parse','HEAD');git(REPO,'push','origin','main')
        assert git(REPO,'ls-remote',git(REPO,'remote','get-url','--push','origin'),'refs/heads/main').split()[0]==head
    assert read(active_path)==previous
    write(active_path,dict(selected_experiment='EXP-0020',selected_serving_commits=COMMITS['B'],previous_serving_commits=COMMITS['A'],
      current_experiment='EXP-0021',status='Paired evidence audited; EXP20 selected as working performance baseline',selection_document_commit=head,
      scheduler_candidate='rejected_by_user',next_direction=selection['next_direction']))
    write(ART/'selection_publication.json',dict(commit=head,remote_verified=True,selected_experiment='EXP-0020'))
    print(json.dumps(dict(status='audited_selected_pushed',commit=head,metrics=metrics,formal=[x['metrics'] for x in formal])),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        write(ART/'selection_audit_error.json',dict(error=traceback.format_exc()));raise
