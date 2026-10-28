"""Freeze the documented complete-MLP diagnostic and start one finite run."""
from pathlib import Path
from datetime import datetime,timedelta
import ast,fcntl,hashlib,json,shutil,subprocess,time,traceback
ROOT=Path('/workspace/LeanScale');REPO=ROOT/'MiniCPM-Infer-Sys-Opt';EXP=REPO/'experiments/EXP-0023/expanded';ART=ROOT/'artifacts/EXP-0023/expanded'
STAGING=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a],text=True,stderr=subprocess.STDOUT,timeout=120).strip()
def step(action,**kw):
    with (EXP/'steps.jsonl').open('a') as f:f.write(json.dumps(dict(time=datetime.now().astimezone().isoformat(),action=action,**kw),ensure_ascii=False)+'\n')
def main():
    assert (EXP/'README.md').exists() and not (EXP/'trial.py').exists() and not ART.exists() and not (ROOT/'artifacts/EXP-0023-expanded-request.json').exists()
    assert read(ROOT/'artifacts/EXP-0023/expanded_hypothesis_publication.json')['remote_verified']
    prior=read(EXP.parent/'plan.json');assert read(ROOT/'artifacts/active_inference_plan.json')['selected_serving_commits']==prior['selected_commits']
    now=datetime.now().astimezone();assert now.strftime('%Y-%m-%d')=='2026-10-28';ART.mkdir()
    for p,h in prior['source_hashes'].items():assert sha(Path(p))==h
    route=read(EXP/'route_evidence.json');assert sha(Path(route['graph_path']))==route['graph_sha256'] and sha(Path(route['module_path']))==route['module_sha256']
    source=read(EXP/'production_silu_source.json')['source'];assert hashlib.sha256(source.encode()).hexdigest()==route['kernel_source_sha256']
    for n in ['trial.py','run_screen.py','prepare_and_start.py']:
        shutil.copy2(STAGING/n,EXP/n);ast.parse((EXP/n).read_text(),filename=n)
    sources=dict(prior['source_hashes'])
    for p in [Path(route['graph_path']),Path(route['module_path']),EXP/'production_silu_source.json',EXP/'README.md',EXP/'route_evidence.json',EXP/'daily_window.json']:
        sources[str(p)]=sha(p)
    plan=dict(selected_commits=prior['selected_commits'],source_hashes=sources,script_hashes={p.name:sha(p) for p in EXP.glob('*.py')},
      expected_checks=117,timing_points=180,production_kernel_source_sha256=route['kernel_source_sha256'],
      candidate_gate='all3seeds M8/16 in both modes<=-5percent; other points<=+3percent; abs drift<=3percent; no automatic integration')
    write(EXP/'plan.json',plan);write(EXP/'STATUS.json',dict(stage='prepared',GPU_started=False))
    step('Install and syntax-check independent full MLP scripts after published revised protocol',hypothesis_commit=read(ROOT/'artifacts/EXP-0023/expanded_hypothesis_publication.json')['commit'])
    with (ROOT/'artifacts/docs-publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);assert not git('diff','--cached','--name-only')
        paths=['experiments/EXP-0023/expanded'];git('add','--',*paths);git('diff','--cached','--check');assert '__pycache__' not in git('diff','--cached','--name-only')
        git('commit','--only','-m','experiments: prepare independent complete MLP layout refinement','--',*paths)
        head=git('rev-parse','HEAD');git('push','origin','main');assert git('ls-remote',git('remote','get-url','--push','origin'),'refs/heads/main').split()[0]==head
    write(ART/'preparation.json',dict(status='prepared',commit=head,remote_verified=True))
    now=datetime.now().astimezone();assert now.strftime('%Y-%m-%d')=='2026-10-28'
    daily=read(EXP/'daily_window.json');end=min(now+timedelta(minutes=40),datetime.fromisoformat(daily['compute_stop_at']))
    assert (end-now).total_seconds()>35*60,'Prepared only; insufficient fresh window'
    policy=dict(experiment='EXP-0023-expanded',created_at=now.isoformat(),compute_stop_at=end.isoformat(),archive_target_at=(end+timedelta(minutes=10)).isoformat(),daily_window=daily)
    write(ART/'run_window.json',policy);step('Launch separate bounded complete MLP refinement',policy=policy)
    with (ART/'controller.log').open('w') as log:
        child=subprocess.Popen(['python3',str(EXP/'run_screen.py')],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    latest=ROOT/'artifacts/EXP-0023-expanded-request.json'
    for _ in range(100):
        if latest.exists():break
        assert child.poll() is None;time.sleep(.2)
    else:raise RuntimeError('No launch receipt')
    receipt=read(latest);assert receipt['pid']==child.pid
    write(ART/'launch.json',dict(preparation_commit=head,receipt=receipt,policy=policy));print(json.dumps(dict(commit=head,receipt=receipt,policy=policy)),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        if EXP.exists():step('Preparation/start failed; preserve files and do not blindly rerun',error=traceback.format_exc())
        write(ROOT/'artifacts/EXP-0023-expanded-preparation-error.json',dict(error=traceback.format_exc()));raise
