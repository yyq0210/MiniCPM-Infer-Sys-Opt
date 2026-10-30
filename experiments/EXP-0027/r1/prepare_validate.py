"""Isolated production launcher implementation and bounded device validation."""
from pathlib import Path
from datetime import datetime,timedelta
import argparse,ast,fcntl,hashlib,json,os,re,shutil,signal,socket,subprocess,sys,time,traceback
ROOT=Path('/workspace/LeanScale');DOC=ROOT/'MiniCPM-Infer-Sys-Opt';EXP=DOC/'experiments/EXP-0027/r1'
WT=ROOT/'worktrees/EXP-0027-r1';ART=ROOT/'artifacts/EXP-0027-r1';SRC=Path(__file__).resolve().parent
BASE=ROOT/'worktrees/EXP-0027';COMMITS={'vllm-plugin-FL':'2a4cf99f5a765dc7de3a3337303e91183bc9decd','FlagGems':'de958ddfc4deeb43d6c94bb8652a3d4d3bc905fe'}
def now():return datetime.now().astimezone()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');t.replace(p)
def git(repo,*a):return subprocess.check_output(['git','-C',str(repo),*a],text=True,stderr=subprocess.STDOUT,timeout=120).strip()
def step(action,**kw):
    with (EXP/'steps.jsonl').open('a') as f:f.write(json.dumps(dict(time=now().isoformat(),action=action,**kw))+'\n')
def publish(paths,message):
    sys.path.insert(0,str(DOC/'experiments/EXP-0025'));from common import publish as p
    return p(paths,message)
FILES=['vllm_fl/patches/topp_row_parallel.py','tests/standalone_topp_shadow_budget.py']
def prepare():
    assert (EXP/'README.md').exists() and not WT.exists() and not ART.exists()
    assert read(ROOT/'artifacts/EXP-0027/review_20261030_publication.json')['remote_verified']
    assert read(ROOT/'artifacts/active_inference_plan.json')['selected_experiment']=='EXP-0020'
    assert sha(SRC/'topp_row_parallel.py')=='897b3981551c2ac5ec5b5b865fd3167f4851734bec2e9526079a4d3a08c4486b'
    current=now()
    assert current.date().isoformat()=='2026-10-30' and current.hour<22, 'Refresh available compute window'
    step('Begin r1 separate worktree after archived hypothesis and 11 CPU checks',baseline=COMMITS)
    WT.mkdir();ART.mkdir()
    for name,commit in COMMITS.items():
        assert git(BASE/name,'rev-parse','HEAD')==commit and not git(BASE/name,'status','--porcelain')
        assert not (BASE/name/'AGENTS.md').exists()
        args=['worktree','add']+(['-b','leanscale/iluvatar-topp-shadow-coverage'] if name=='vllm-plugin-FL' else ['--detach'])
        git(BASE/name,*args,str(WT/name),commit)
    plugin=WT/'vllm-plugin-FL'
    source=Path('/usr/local/lib/python3.12/site-packages/vllm/v1/sample/ops/topk_topp_triton.py')
    assert sha(source)=='64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8'
    shutil.copy2(SRC/'topp_row_parallel.py',plugin/FILES[0])
    shutil.copy2(SRC/'validate_shadow_gpu.py',plugin/FILES[1])
    for name in FILES:ast.parse((plugin/name).read_text())
    git(plugin,'diff','--check')
    shutil.copy2(__file__,EXP/'prepare_validate.py')
    shutil.copy2(SRC/'validate_all.py',EXP/'validate_all.py')
    write(EXP/'implementation_sources.json',{n:sha(plugin/n) for n in FILES+['vllm_fl/__init__.py','tests/standalone_topp_row_parallel.py']})
    write(EXP/'STATUS.json',dict(stage='implemented_not_validated',selected_experiment='EXP-0020'))
    step('Apply per-batch armed shadow policy; original launcher and numerical algorithm retained',files=FILES,expected_checks=39)
    pub=publish(['experiments/EXP-0027/r1'],'experiments: prepare isolated top-p shadow coverage GPU validation')
    current=now()
    assert current.date().isoformat()=='2026-10-30' and current.hour<22
    end=current+timedelta(minutes=15)
    run=ART/current.strftime('%Y%m%d_%H%M%S_validation')
    write(ART/'validation_window.json',dict(started_at=current.isoformat(),compute_stop_at=end.isoformat(),save_target_at=(end+timedelta(minutes=10)).isoformat(),scope='New short validation window; does not reuse Oct29 window'))
    with (ART/'validation_controller.log').open('w') as log:
        child=subprocess.Popen(['python3',str(EXP/'prepare_validate.py'),'--run',str(run)],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    receipt=dict(pid=child.pid,process_start_ticks=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(') ',1)[1].split()[19],run_directory=str(run),compute_stop_at=end.isoformat(),preparation_publication=pub)
    write(ART/'validation_launch.json',receipt)
    active=read(ROOT/'artifacts/active_inference_plan.json');active.update(current_experiment='EXP-0027-r1',status='R1 bounded GPU diagnostic validation running',next_direction='Real-service shadow and official4K after validation')
    write(ROOT/'artifacts/active_inference_plan.json',active)
    print(json.dumps(receipt),flush=True)


def run_validation(run):
    assert run.parent==ART;run.mkdir(exist_ok=False);plugin=WT/'vllm-plugin-FL'
    end=datetime.fromisoformat(read(ART/'validation_window.json')['compute_stop_at'])
    assert now()<end
    lock=Path('/workspace/experiments/codex_20261019/runner.lock').open('a');proc=None
    status=dict(stage='preparing',started_at=now().isoformat(),deadline=end.isoformat())
    def abort(*a):raise TimeoutError('Validation terminated')
    signal.signal(signal.SIGTERM,abort)
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for port in [9031,9032]:
            with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',port))!=0
        for _ in range(2):
            out=subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True,timeout=20)
            m=re.search(r'(\d+)MiB\s*/\s*(\d+)MiB\s*\|\s*(\d+)%',out)
            assert m and int(m[1])<=1024 and int(m[3])<=5;time.sleep(3)
        for n,h in read(EXP/'implementation_sources.json').items():assert sha(plugin/n)==h
        env=dict(os.environ,PYTHONPATH=str(plugin)+':'+str(WT/'FlagGems/src'),VLLM_PLUGINS='fl',PYTHONUNBUFFERED='1',
            VLLM_FL_TOPP_ROW_PARALLEL='0',VLLM_FL_TOPP_ROW_PARALLEL_SHADOW='0',
            LEAN_SCALE_VALIDATION_DIR=str(run),TRITON_CACHE_DIR=str(run/'triton_cache'))
        status['stage']='gpu_checks';write(run/'status.json',status)
        with (run/'validation.log').open('w') as log:
            proc=subprocess.Popen(['python3',str(EXP/'validate_all.py')],cwd=run,env=env,
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        rc=proc.wait(timeout=max(1,(end-now()).total_seconds()));assert rc==0,('validation exit',rc)
        assert read(run/'result.json')['status']=='passed'
        for n,h in read(EXP/'implementation_sources.json').items():assert sha(plugin/n)==h
        git(plugin,'diff','--check');git(plugin,'add','--',*FILES)
        git(plugin,'commit','--only','-m','fix: arm diagnostic checks after warmup and reserve per-batch coverage','--',*FILES)
        commit=git(plugin,'rev-parse','HEAD');branch=git(plugin,'branch','--show-current')
        assert branch=='leanscale/iluvatar-topp-shadow-coverage'
        assert git(plugin,'remote','get-url','--push','leanscale')=='git@github.com:yyq0210/vllm-plugin-FL.git'
        git(plugin,'push','leanscale',branch)
        assert git(plugin,'ls-remote',git(plugin,'remote','get-url','--push','leanscale'),'refs/heads/'+branch).split()[0]==commit
        commits=dict(COMMITS,**{'vllm-plugin-FL':commit});write(EXP/'candidate_commits.json',commits)
        write(run/'code_publication.json',dict(commits=commits,remote_verified=True,branch=branch))
        status.update(stage='passed',checks=read(run/'result.json')['checks'],candidate_commits=commits)
        step('Integration checks passed and default-off candidate pushed to personal fork',checks=status['checks'],commits=commits)
    except BaseException:
        status.update(stage='failed',error=traceback.format_exc());step('Integration attempt failed; preserve source and output',error=traceback.format_exc());traceback.print_exc()
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        status.update(finished_at=now().isoformat(),owned_children_reaped=True);write(run/'status.json',status);lock.close()
    dest=EXP/'validation'/run.name;dest.mkdir(parents=True,exist_ok=False);manifest={}
    for f in list(run.iterdir())+list((run/'standard').glob('*'))+list((run/'shadow').glob('*')):
        if f.is_file() and f.suffix in ['.json','.jsonl','.log']:
            t=dest/f.relative_to(run);t.parent.mkdir(parents=True,exist_ok=True);t.write_text('\n'.join(l.rstrip() for l in f.read_text(errors='replace').splitlines())+'\n')
            manifest[str(f.relative_to(run))]=dict(raw_sha256=sha(f),archived_sha256=sha(t))
    write(dest/'artifact_manifest.json',manifest);write(EXP/'STATUS.json',status)
    paths=['experiments/EXP-0027/r1/validation/'+run.name,'experiments/EXP-0027/r1/STATUS.json','experiments/EXP-0027/r1/steps.jsonl']
    if (EXP/'candidate_commits.json').exists():paths.append('experiments/EXP-0027/r1/candidate_commits.json')
    try:pub=publish(paths,'experiments: archive top-p launcher integration checks')
    except Exception as exc:pub=dict(status='push_failed_local_saved',error=repr(exc))
    write(run/'publication.json',pub);write(ART/'validation_publication.json',pub)
    print(json.dumps(dict(status=status,publication=pub)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path);args=parser.parse_args()
    if args.run:run_validation(args.run)
    else:prepare()
