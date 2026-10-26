"""Audit one completed official case without GPU work or changing the serving selection."""
from pathlib import Path
import argparse
import csv
import datetime
import fcntl
import hashlib
import json
import re
import shutil
import statistics
import subprocess

ROOT=Path('/workspace/LeanScale');REPO=ROOT/'MiniCPM-Infer-Sys-Opt'
EXP=REPO/'experiments/EXP-0020';WT=ROOT/'worktrees/EXP-0020'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open() as f:return list(csv.DictReader(f))
def git(repo,*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True,stderr=subprocess.STDOUT,timeout=120).strip()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',choices=('4k','16k'),required=True);args=ap.parse_args()
    dest=EXP/f'review_{args.case}.json';assert not dest.exists(),'Review the existing audit instead of replacing it'
    prefill,count=(4096,256) if args.case=='4k' else (16384,128)
    commits=read(EXP/'candidate_commits.json');gate=read(EXP/'candidate_validation.json')
    assert gate['status']=='passed' and gate['candidate_commits']==commits
    for name,expected in commits.items():
        assert git(WT/name,'rev-parse','HEAD')==expected
        assert not git(WT/name,'status','--porcelain')
    for name,expected in gate['diagnostic_hashes'].items():assert sha(Path(name))==expected
    for repo,files in gate['code_hashes'].items():
        for name,expected in files.items():assert sha(WT/repo/name)==expected
    for name,expected in gate['script_hashes'].items():assert sha(EXP/name)==expected
    run=Path(read(ROOT/f'artifacts/EXP-0020-latest-{args.case}.json')['run_directory'])
    state=read(run/'status.json');handler=read(run/'completion_handler_status.json')
    assert state['stage']=='performance_case_complete' and state['commits']==commits
    assert handler['status']=='pushed' and handler['complete_results'] and handler['remote_verified']
    git(REPO,'merge-base','--is-ancestor',handler['commit'],'HEAD')
    folder=run/'candidate_benchmark';route=read(folder/'route_evidence.json')
    assert route['status']=='passed' and route['gqa_layers']==route['fused_runtime_layers']==42
    assert route['gqa_launch']==gate['launch'] and route['old_cache_updates_in_fused_graph']==0
    assert route['kernel_counts']['decode'] and route['kernel_counts']['prefill'] and route['graphs']
    for item in route['kernels']:
        p=run/item['path'];assert sha(p)==item['sha256'] and sha(p.with_suffix('.ttir'))==item['ttir_sha256']
        metadata=read(p);assert metadata['num_warps']==4 and metadata['num_stages']==1 and metadata['target']['backend']=='corex'
        assert metadata['shared']==item['shared']
        assert (item['shared'],item['logical_query_rows'],item['page_table_load_elements']) == (
            (18432,8,2) if item['kind']=='decode' else (32768,64,2))
        ir=p.with_suffix('.ttir').read_text()
        q=8 if item['kind']=='decode' else 64
        assert re.search(r'tt\.load[^\n]*tensor<'+str(q)+r'x128x!tt\.ptr<(?:bf16|f16)>>',ir)
        assert re.search(r'tt\.load[^\n]*tensor<2x!tt\.ptr<i32>>',ir)
        assert re.search(r'tt\.broadcast[^\n]*tensor<2x1xi64> -> tensor<2x16xi64>',ir)
        assert re.search(r'tt\.reshape[^\n]*tensor<2x16xi64> -> tensor<32xi64>',ir)
        if item['kind']=='prefill':
            assert re.search(r'tt\.load[^\n]*tensor<128x32x!tt\.ptr<(?:bf16|f16)>>',ir)
            assert re.search(r'tt\.dot[^\n]*tensor<64x32xf32>',ir)
    for item in route['graphs']:
        assert sha(run/item['path'])==item['sha256']
        assert item['counts']==dict(fl_rope_kv_dispatch=42,fl_rotary_embedding_dispatch=0,unified_kv_cache_update=0)
    assert read(folder/'smoke.json')['usage']['completion_tokens']>0
    raw_path,=(folder/'benchmark_results').glob('raw_runs_*.csv')
    summary_path,=(folder/'benchmark_results').glob('summary_*.csv')
    raw,summary=rows(raw_path),rows(summary_path);assert len(raw)==4 and len(summary)==1
    assert (folder/'benchmark.log').read_text().count('Failed requests:                         0')==4
    for row in raw:
        assert row['Run Status']=='SUCCESS'
        assert tuple(int(float(row[k])) for k in ['Prefill','Decode','Conc','Num Prompts'])==(prefill,1024,64,count)
        assert int(float(row['Successful Requests']))==count
        assert int(float(row['Total Input Tokens']))==prefill*count
        assert int(float(row['Total Output Tokens']))==1024*count
    parent=read(EXP.parent/'EXP-0016/selection.json')
    assert parent['selected_serving_commits']==read(EXP/'parent_commits.json')
    parent_item=next(x for x in parent['case_results'] if x['case']==args.case)
    parent_folder=ROOT/'artifacts/EXP-0016'/parent_item['run_name']/'candidate_benchmark'
    parent_raw,=(parent_folder/'benchmark_results').glob('raw_runs_*.csv')
    parent_summary,=(parent_folder/'benchmark_results').glob('summary_*.csv')
    assert sha(parent_raw)==parent_item['raw_sha256'] and sha(parent_summary)==parent_item['summary_sha256']
    assert sha(parent_folder/'route_evidence.json')==parent_item['route_sha256']
    before,after=rows(parent_summary)[0],summary[0];metrics={}
    for key in ['Total tok/s','Output tok/s','Mean TTFT (ms)','Mean TPOT (ms)','Mean ITL (ms)']:
        a,b=float(before[key]),float(after[key])
        assert abs(b-statistics.mean(float(x[key]) for x in raw[1:]))<=.011
        metrics[key]=dict(exp16=a,prefill_page_broadcast=b,change_percent=(b/a-1)*100)
    comparison=read(run/'comparison_exp16.json')[0]
    assert comparison['candidate_total_tokens_s']==float(after['Total tok/s'])
    assert comparison['candidate_retained_runs']==[float(x['Total tok/s']) for x in raw[1:]]
    assert abs(comparison['throughput_change_percent']-metrics['Total tok/s']['change_percent'])<1e-9
    assert abs(comparison['candidate_cv_percent']-statistics.stdev(float(x['Total tok/s']) for x in raw[1:])/statistics.mean(float(x['Total tok/s']) for x in raw[1:])*100)<1e-9
    allow_16k=(args.case=='4k' and metrics['Total tok/s']['change_percent']>=-3
               and metrics['Mean TTFT (ms)']['change_percent']<=5 and metrics['Mean TPOT (ms)']['change_percent']<=5)
    report=dict(allow_16k=allow_16k,checked_at=datetime.datetime.now().astimezone().isoformat(),case=args.case,run_name=run.name,
        status='complete_case_audited',metrics=metrics,comparison_exp16=comparison,comparison_original=state['performance'][0],
        accounting=state['scheduler_accounting'],candidate_commits=commits,archive_commit=handler['commit'],
        raw_sha256=sha(raw_path),summary_sha256=sha(summary_path),route_sha256=sha(folder/'route_evidence.json'),
        kernel_counts=route['kernel_counts'],selected_for_serving=False,
        checks=dict(four_successful_rounds=True,discard_first=True,partial_runs_combined=False,route_hashes_revalidated=True),
        interpretation='Historical sessions; CV describes three retained rounds, not confidence intervals. Assess both official cases before selection; 4K attribution needs a bounded paired recheck when gains are similar to observed run variation.',
        full_task_accuracy='deferred_by_user')
    with (ROOT/'artifacts/docs-publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert not git(REPO,'diff','--cached','--name-only')
        dest.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        source=Path(__file__).resolve()
        (EXP/'reviews').mkdir(exist_ok=True)
        if source!=EXP/'reviews/review_case.py':shutil.copy2(source,EXP/'reviews/review_case.py')
        with (EXP/'steps.jsonl').open('a') as f:
            f.write(json.dumps(dict(time=report['checked_at'],action='CPU audit complete official case; no serving selection change',
                case=args.case,run_name=run.name,metrics=metrics,archive_commit=handler['commit']))+'\n')
        paths=[str(dest.relative_to(REPO)),'experiments/EXP-0020/reviews/review_case.py','experiments/EXP-0020/steps.jsonl']
        forbidden=re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|lab_inst_[A-Za-z0-9]+|ssh\.flagos\.io')
        for p in paths:assert not forbidden.search((REPO/p).read_bytes()),p
        git(REPO,'add','--',*paths);git(REPO,'diff','--cached','--check')
        git(REPO,'commit','--only','-m','experiments: audit complete prefill '+args.case+' E2E result','--',*paths)
        head=git(REPO,'rev-parse','HEAD');git(REPO,'push','origin','main')
        assert git(REPO,'ls-remote',git(REPO,'remote','get-url','--push','origin'),'refs/heads/main').split()[0]==head
    print(json.dumps(dict(status='audited_and_pushed',case=args.case,commit=head,metrics=metrics)),flush=True)


if __name__=='__main__':main()
