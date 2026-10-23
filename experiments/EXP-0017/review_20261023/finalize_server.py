"""CPU-only independent audit and performance-baseline selection after ABBA."""
from pathlib import Path
import csv
import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import re
import shutil
import statistics
import subprocess
import sys
import tempfile

ROOT=Path('/workspace/LeanScale');REPO=ROOT/'MiniCPM-Infer-Sys-Opt'
EXP=REPO/'experiments/EXP-0017';PRE=REPO/'experiments/EXP-0016'
STAGING=Path(__file__).resolve().parent
DEST=EXP/'review_20261023'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def git(p,*a):return subprocess.check_output(['git','-C',str(p),*a],text=True,stderr=subprocess.STDOUT,timeout=120).strip()
def rows(p):
    with p.open() as f:return list(csv.DictReader(f))
def route_files(run,route):
    for k in route['kernels']:
        p=run/k['path'];assert p.resolve().is_relative_to(run)
        assert sha(p)==k['sha256'] and sha(p.with_suffix('.ttir'))==k['ttir_sha256']
        m=read(p);assert m['target']['backend']=='corex'
        assert m['num_warps']==k['num_warps'] and m['num_stages']==k['num_stages'] and m['shared']==k['shared']
    for g in route['graphs']:
        p=run/g['path'];assert p.resolve().is_relative_to(run) and sha(p)==g['sha256']
    return dict(kernels=len(route['kernels']),graph_source_files=len(route['graphs']))
def main():
    assert not DEST.exists() and not (PRE/'selection.json').exists(),'Do not overwrite completed review/selection'
    assert not git(REPO,'status','--porcelain')
    offline=read(STAGING/'published_audit.json');assert offline['status']=='published_snapshot_audited'
    pub=read(ROOT/'artifacts/EXP-0017/publication.json')
    assert pub['status']=='pushed' and pub['remote_verified'] and pub['commit']==offline['commit']
    git(REPO,'merge-base','--is-ancestor',pub['commit'],'HEAD')
    for rel,digest in offline['downloaded_sha256'].items():assert sha(EXP/rel)==digest,rel
    sys.path.insert(0,str(EXP))
    import common
    from route_checks import verify_route
    for arm in ['A','B']:common.verify_sources(arm)
    assert sha(Path('/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py'))=='0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565'
    assert importlib.metadata.version('torch')=='2.10.0+corex.4.5.0'
    assert importlib.metadata.version('vllm')=='0.24.0+empty'
    result=read(EXP/'RESULTS.json');state=read(ROOT/'artifacts/EXP-0017/status.json')
    assert state['stage']=='complete' and result['sessions']==state['completed_sessions']
    reviews=[]
    with tempfile.TemporaryDirectory(prefix='exp17_final_audit_') as t:
        for item in result['sessions']:
            run=ROOT/'artifacts/EXP-0017'/item['run_name'];folder=run/'candidate_benchmark'
            assert read(run/'result.json')==item
            p=read(run/'publication.json');assert p['status']=='pushed' and p['remote_verified'];git(REPO,'merge-base','--is-ancestor',p['commit'],'HEAD')
            rawp,=(folder/'benchmark_results').glob('raw_runs_*.csv');sump,=(folder/'benchmark_results').glob('summary_*.csv')
            raw,summary=rows(rawp),rows(sump);assert len(raw)==4 and summary==[item['summary']]
            for row in raw:
                assert row['Run Status']=='SUCCESS' and int(float(row['Successful Requests']))==256
                assert [int(float(row[k])) for k in ['Prefill','Decode','Conc','Num Prompts','Total Input Tokens','Total Output Tokens']]==[4096,1024,64,256,1048576,262144]
            for key in result['metrics']:assert abs(float(summary[0][key])-statistics.mean(float(x[key]) for x in raw[1:]))<=.011
            assert (folder/'benchmark.log').read_text().count('Failed requests:                         0')==4
            source=read(EXP/'artifacts'/run.name/'source_hashes.json')
            for rel,digest in source.items():assert sha(run/rel)==digest,rel
            env=read(run/'environment_check.json');assert env['candidate_commits']==item['commits']
            for rel,digest in env['installed_source_hashes'].items():assert sha(Path('/usr/local/lib/python3.12/site-packages')/rel)==digest
            output=Path(t)/(run.name+'.json');verify_route(run,item['arm'],output)
            assert read(output)==read(folder/'route_evidence.json')
            count=route_files(run,read(output))
            reviews.append(dict(index=item['index'],arm=item['arm'],run_name=run.name,
                raw_sha256=sha(rawp),summary_sha256=sha(sump),route_sha256=sha(folder/'route_evidence.json'),
                archive_commit=p['commit'],compiled_files_rehashed=count,raw_source_hash_count=len(source),checks='passed'))
    gate=read(PRE/'candidate_validation.json')
    for rel,digest in gate['diagnostic_hashes'].items():assert sha(Path(rel))==digest
    for name,digest in gate['script_hashes'].items():assert sha(PRE/name)==digest
    for name,commit in common.COMMITS['B'].items():
        tree=ROOT/'worktrees/EXP-0016'/name
        assert git(tree,'rev-parse','HEAD')==commit and not git(tree,'status','--porcelain')
        for rel,digest in gate['code_hashes'].get(name,{}).items():assert sha(tree/rel)==digest
    cases=[]
    for case in ['4k','16k']:
        a=read(PRE/f'audited_{case}.json');assert a['status']=='complete_case_audited'
        run=ROOT/'artifacts/EXP-0016'/a['run_name'];folder=run/'candidate_benchmark'
        rawp,=(folder/'benchmark_results').glob('raw_runs_*.csv');sump,=(folder/'benchmark_results').glob('summary_*.csv')
        assert sha(rawp)==a['raw_sha256'] and sha(sump)==a['summary_sha256'] and sha(folder/'route_evidence.json')==a['route_sha256']
        route=read(folder/'route_evidence.json');counts=route_files(run,route)
        a['compiled_files_rehashed_on_selection']=counts;a['selected_for_serving']=True;cases.append(a)
    stamp=datetime.datetime.now().astimezone().isoformat()
    audit=dict(status='complete_server_audit',checked_at=stamp,source_publication=pub,
        published_snapshot_sha256=sha(STAGING/'published_audit.json'),sessions=reviews,
        current_source_commits_verified=common.COMMITS,installed_vllm_source_hashes_verified=True,
        original_official_cases_revalidated=[dict(case=a['case'],run_name=a['run_name'],counts=a['compiled_files_rehashed_on_selection']) for a in cases],
        metrics=result['metrics'],full_task_accuracy='deferred_by_user',gpu_work_performed=False)
    previous=read(ROOT/'artifacts/active_inference_plan.json');assert previous['selected_serving_commits']==common.COMMITS['A']
    selection=dict(selected_experiment='EXP-0016',selected_serving_commits=common.COMMITS['B'],previous_selection=previous,
        selected_at=stamp,scope='Working performance baseline for subsequent optimization; full model task accuracy deferred.',
        basis='Complete official 4K and16K improve throughput and latency; both ABBA 4K pairs have positive throughput and latency direction, and prefill operator tests support the mechanism. Select for further development without claiming a fixed causal gain.',
        limitations='ABBA A1 CV14.14%, A endpoints+5.78%, paired gains+7.90%/+1.27%. Only two sessions per arm; do not call mean+4.49% stable or statistically significant. Keep all slow rounds.',
        case_results=cases,paired_review='../EXP-0017/review_20261023/server_audit.json',paired_metrics=result['metrics'],
        scheduler_candidate='rejected_by_user',next_direction='Dedicated4K phase and hotspot profiling; new dated run window required before GPU work')
    with (ROOT/'artifacts/docs-publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert not git(REPO,'diff','--cached','--name-only')
        DEST.mkdir()
        for name in ['audit_published.py','published_audit.json','REVIEW.md','FETCH_NOTES.md','finalize_server.py']:shutil.copy2(STAGING/name,DEST/name)
        write(DEST/'server_audit.json',audit);write(PRE/'selection.json',selection)
        text='''# GQA prefill性能选择与4K配对复核

2026-10-23。选择EXP-0016 stage1 prefill作为后续开发的性能基线；该选择不等于完整模型精度或赛事验收。

EXP16原始完整官方结果保持：4K3476.54、16K2314.37 total tok/s，相对EXP14分别+4.33%/+6.08%。两组各四轮弃首轮。
独立EXP17 ABBA复核中，A均值3305.385、新B3453.87 tok/s，观测+4.49%；TTFT5949.805→5580.35ms（-6.21%），TPOT90.63→86.08ms（-5.02%）。

| 会话 | 版本 | 保留三轮均值tok/s | CV |
|---|---|---:|---:|
| A1 | stage2 |3212.58|14.14%|
| B1 | stage1 |3466.30|3.15%|
| B2 | stage1 |3441.44|4.85%|
| A2 | stage2 |3398.19|3.03%|

两配对吞吐增幅+7.90%与+1.27%，方向一致但幅度不同。A首尾变化+5.78%，A1慢轮2688.62原样纳入，不剔除、不混用EXP16历史结果。样本只含每版本两会话，不能宣称统计显著或稳定4.49%因果增益。
选择依据是完整16K结果、4K多个会话吞吐/延迟方向和单算子数值/性能证据的共同支持，用于继续优化；后续仍需关注4K轮间变化。

服务器恢复后完成四个会话原始文件哈希、实际JSON/TTIR、计算图源码、当前A/B工作树和已装vLLM源码的独立复核，并再次核对EXP16两组原始数据/编译缓存。35指CUDA Graph捕获尺寸，每会话计算图源码文件为1；两者分别验证。
此前公开结果离线审核与两次审核工具失败均保留，服务器侧补核通过后消除访问受限导致的证据缺口。整模MATH500精度仍暂缓，KV压缩未启用。

下一步创建新实验，按选定版本对4K的initial、mixed、大batch decode、尾部小batch做有界profiling；FULL和NONE口径分开。今晚关机时间待确认，新GPU任务须使用当日运行窗口。

[最终审核](../experiments/EXP-0017/review_20261023/server_audit.json) / [性能选择](../experiments/EXP-0016/selection.json) / [逐会话完整结果](../experiments/EXP-0017/RESULTS.json)。
'''
        (REPO/'docs/gqa_prefill_selection_20261023.md').write_text(text)
        with (EXP/'steps.jsonl').open('a') as f:f.write(json.dumps(dict(time=stamp,action='Rehash actual artifacts after restart; select EXP16 as development performance baseline with ABBA limitations',review='review_20261023/server_audit.json',gpu_work=False))+'\n')
        with (REPO/'docs/optimization_journal.md').open('a') as f:f.write('\n\n### 2026-10-23：4K ABBA复核完成，选定prefill stage1性能基线\n\n四组完整数据与实际编译缓存独立复核通过，配对均值+4.49%、两对+7.90%/+1.27%，保留A1慢轮与漂移限制。结合完整16K+6.08%及算子证据，选择EXP16继续开发，不声明固定因果收益或精度验收。详见[gqa_prefill_selection_20261023.md](gqa_prefill_selection_20261023.md)。\n')
        paths=['experiments/EXP-0017/review_20261023','experiments/EXP-0017/steps.jsonl','experiments/EXP-0016/selection.json','docs/gqa_prefill_selection_20261023.md','docs/optimization_journal.md']
        bad=re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|lab_inst_[A-Za-z0-9]+|ssh\.flagos\.io')
        for rel in paths:
            p=REPO/rel
            for f in p.rglob('*') if p.is_dir() else [p]:
                if f.is_file():assert not bad.search(f.read_bytes()),str(f)
        git(REPO,'add','--',*paths);git(REPO,'diff','--cached','--check')
        git(REPO,'commit','--only','-m','experiments: audit ABBA caches and select GQA prefill performance baseline','--',*paths)
        head=git(REPO,'rev-parse','HEAD');git(REPO,'push','origin','main')
        assert git(REPO,'ls-remote',git(REPO,'remote','get-url','--push','origin'),'refs/heads/main').split()[0]==head
    previous.update(previous_serving_commits=common.COMMITS['A'],selected_serving_commits=common.COMMITS['B'],selected_experiment='EXP-0016',
        status='Official4K/16K andABBA audited; stage1 selected for development; awaiting fresh-window4K profiling',selection_document_commit=head,
        next_direction='Dedicated4K phase/hotspot profiling on selected stage1 version')
    write(ROOT/'artifacts/active_inference_plan.json',previous)
    write(ROOT/'artifacts/EXP-0017/final_review_publication.json',dict(status='pushed',commit=head,remote_verified=True))
    print(json.dumps(dict(status='audited_selected_pushed',commit=head,selected=common.COMMITS['B'],actual_sessions=len(reviews))),flush=True)
if __name__=='__main__':main()
