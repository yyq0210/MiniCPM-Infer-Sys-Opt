"""Install a previously documented weight-layout hypothesis and launch bounded screen."""
from pathlib import Path
from datetime import datetime,timedelta
import ast,fcntl,hashlib,json,shutil,subprocess,time,traceback
ROOT=Path('/workspace/LeanScale');REPO=ROOT/'MiniCPM-Infer-Sys-Opt';EXP=REPO/'experiments/EXP-0023';ART=ROOT/'artifacts/EXP-0023'
STAGING=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a],text=True,stderr=subprocess.STDOUT,timeout=120).strip()
def step(action,**kw):
    with (EXP/'steps.jsonl').open('a') as f:f.write(json.dumps(dict(time=datetime.now().astimezone().isoformat(),action=action,**kw),ensure_ascii=False)+'\n')
def main():
    assert (EXP/'README.md').exists() and not (EXP/'trial.py').exists() and not ART.exists() and not (ROOT/'artifacts/EXP-0023-request.json').exists()
    assert read(ROOT/'artifacts/EXP-0022/review_publication.json')['remote_verified']
    prior=read(REPO/'experiments/EXP-0022/plan.json');assert read(ROOT/'artifacts/active_inference_plan.json')['selected_serving_commits']==prior['selected_commits']
    now=datetime.now().astimezone();assert now.strftime('%Y-%m-%d')=='2026-10-26';ART.mkdir()
    for p,h in prior['source_hashes'].items():assert sha(Path(p))==h
    for n in ['trial.py','run_screen.py','prepare_and_start.py']:
        shutil.copy2(STAGING/n,EXP/n);ast.parse((EXP/n).read_text(),filename=n)
    plan=dict(selected_commits=prior['selected_commits'],source_hashes=prior['source_hashes'],script_hashes={p.name:sha(p) for p in EXP.glob('*.py')},
      expected_checks=56,timing_points=80,packing_records=4,event_records=16,
      candidate_gate='per projection: M8/16 in both modes <=-5percent; every other M <=+3percent; all drift<=3percent; no observed candidate aten copy; no automatic integration')
    write(EXP/'plan.json',plan);write(EXP/'STATUS.json',dict(stage='prepared',GPU_started=False))
    step('Install new experimental code after published hypothesis; freeze sources and coverage',hypothesis_commit='30e2f7f292b9a7d4ebbfc519a716edb14341426c')
    with (ROOT/'artifacts/docs-publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);assert not git('diff','--cached','--name-only')
        paths=['experiments/EXP-0023'];git('add','--',*paths);git('diff','--cached','--check');assert '__pycache__' not in git('diff','--cached','--name-only')
        git('commit','--only','-m','experiments: prepare native linear weight layout screen','--',*paths)
        head=git('rev-parse','HEAD');git('push','origin','main');assert git('ls-remote',git('remote','get-url','--push','origin'),'refs/heads/main').split()[0]==head
    write(ART/'preparation.json',dict(status='prepared',commit=head,remote_verified=True))
    now=datetime.now().astimezone();assert now.strftime('%Y-%m-%d')=='2026-10-26'
    daily=read(REPO/'experiments/EXP-0020/launches/daily_20261026.json');end=min(now+timedelta(minutes=40),datetime.fromisoformat(daily['compute_stop_at']))
    assert (end-now).total_seconds()>35*60,'Prepared only; insufficient fresh window'
    policy=dict(experiment='EXP-0023',created_at=now.isoformat(),compute_stop_at=end.isoformat(),archive_target_at=(end+timedelta(minutes=10)).isoformat(),daily_window=daily)
    write(ART/'run_window.json',policy);step('Launch separate finite layout screen after published plan',policy=policy)
    with (ART/'controller.log').open('w') as log:
        child=subprocess.Popen(['python3',str(EXP/'run_screen.py')],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    latest=ROOT/'artifacts/EXP-0023-request.json'
    for _ in range(100):
        if latest.exists():break
        assert child.poll() is None;time.sleep(.2)
    else:raise RuntimeError('No launch receipt')
    receipt=read(latest);assert receipt['pid']==child.pid
    write(ART/'launch.json',dict(preparation_commit=head,receipt=receipt,policy=policy));print(json.dumps(dict(commit=head,receipt=receipt,policy=policy)),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        if EXP.exists():step('Preparation/start failed; retain files and do not blindly rerun',error=traceback.format_exc())
        write(ROOT/'artifacts/EXP-0023-preparation-error.json',dict(error=traceback.format_exc()));raise
