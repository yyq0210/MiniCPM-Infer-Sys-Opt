"""Isolated production launcher implementation and bounded device validation."""
from pathlib import Path
from datetime import datetime,timedelta
import argparse,ast,fcntl,hashlib,json,os,re,shutil,signal,socket,subprocess,sys,time,traceback
ROOT=Path('/workspace/LeanScale');DOC=ROOT/'MiniCPM-Infer-Sys-Opt';EXP=DOC/'experiments/EXP-0027'
WT=ROOT/'worktrees/EXP-0027';ART=ROOT/'artifacts/EXP-0027';SRC=Path(__file__).resolve().parent
BASE=ROOT/'worktrees/EXP-0020';COMMITS={'vllm-plugin-FL':'ce19f93eee4c73f5a26911554700457e879f891d','FlagGems':'de958ddfc4deeb43d6c94bb8652a3d4d3bc905fe'}
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
FILES=['vllm_fl/__init__.py','vllm_fl/patches/topp_row_parallel.py','tests/standalone_topp_row_parallel.py']
def prepare():
    assert (EXP/'README.md').exists() and not WT.exists() and not ART.exists()
    for name in COMMITS:
        p=BASE/name/'AGENTS.md'
        assert not p.exists(), 'Read additional repository instructions before implementation: '+str(p)
    assert read(ROOT/'artifacts/EXP-0026/review_publication.json')['remote_verified']
    assert read(ROOT/'artifacts/active_inference_plan.json')['selected_serving_commits']==COMMITS
    step('Begin bounded top-p launcher implementation after published hypothesis',baseline=COMMITS)
    WT.mkdir();ART.mkdir()
    for name,commit in COMMITS.items():
        assert git(BASE/name,'rev-parse','HEAD')==commit and not git(BASE/name,'status','--porcelain')
        args=['worktree','add']+(['-b','leanscale/iluvatar-topp-row-parallel'] if name=='vllm-plugin-FL' else ['--detach'])
        git(BASE/name,*args,str(WT/name),commit)
    plugin=WT/'vllm-plugin-FL';source=Path('/usr/local/lib/python3.12/site-packages/vllm/v1/sample/ops/topk_topp_triton.py')
    code=(SRC/'topp_row_parallel.py').read_text();assert code.count('@TOPP_SOURCE_SHA256@')==1
    (plugin/'vllm_fl/patches/topp_row_parallel.py').write_text(code.replace('@TOPP_SOURCE_SHA256@',sha(source)))
    shutil.copy2(SRC/'validate_topp.py',plugin/'tests/standalone_topp_row_parallel.py')
    p=plugin/'vllm_fl/__init__.py';code=p.read_text();needle='    _patch_custom_ops()\n';assert code.count(needle)==1
    code=code.replace(needle,'    from vllm_fl.patches.topp_row_parallel import register_topp_row_parallel_env\n    register_topp_row_parallel_env()\n\n'+needle)
    needle='    install_rope_kv_fusion()\n';assert code.count(needle)==1
    code=code.replace(needle,needle+'\n    from vllm_fl.patches.topp_row_parallel import install_topp_row_parallel\n    install_topp_row_parallel()\n')
    p.write_text(code)
    for name in FILES:ast.parse((plugin/name).read_text())
    git(plugin,'diff','--check')
    shutil.copy2(__file__,EXP/'prepare_validate.py')
    write(EXP/'implementation_sources.json',{n:sha(plugin/n) for n in FILES})
    write(EXP/'STATUS.json',dict(stage='implemented_not_validated',selected_experiment='EXP-0020'))
    step('Implement default-off bounded top-p launcher; original kernel and shared buffer cache',files=FILES,
        source_sha256=sha(source),shadow='diagnostic only, at most32 exact/RNG comparisons')
    pub=publish(['experiments/EXP-0027'],'experiments: record top-p row concurrency integration before tests')
    current=now();daily=datetime.fromisoformat('2026-10-29T23:35:00+08:00')
    assert current.date()==daily.date() and current+timedelta(minutes=20)<=daily
    run=ART/current.strftime('%Y%m%d_%H%M%S_validation');end=current+timedelta(minutes=20)
    write(ART/'validation_window.json',dict(started_at=current.isoformat(),compute_stop_at=end.isoformat(),daily_compute_stop=daily.isoformat(),save_target_at=(end+timedelta(minutes=10)).isoformat()))
    env=dict(os.environ,PYTHONUNBUFFERED='1')
    with (ART/'validation_controller.log').open('w') as log:
        child=subprocess.Popen(['python3',str(EXP/'prepare_validate.py'),'--run',str(run)],env=env,
            stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    receipt=dict(pid=child.pid,process_start_ticks=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(') ',1)[1].split()[19],
        run_directory=str(run),compute_stop_at=end.isoformat(),preparation_publication=pub)
    write(ART/'validation_launch.json',receipt)
    active=read(ROOT/'artifacts/active_inference_plan.json');active.update(current_experiment='EXP-0027',
        status='Top-p launcher implemented; bounded GPU integration validation running',next_direction='Real service shadow route and official4K only after integration checks')
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
            proc=subprocess.Popen(['python3',str(plugin/'tests/standalone_topp_row_parallel.py')],cwd=run,env=env,
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        rc=proc.wait(timeout=max(1,(end-now()).total_seconds()));assert rc==0,('validation exit',rc)
        assert read(run/'result.json')['status']=='passed'
        for n,h in read(EXP/'implementation_sources.json').items():assert sha(plugin/n)==h
        git(plugin,'diff','--check');git(plugin,'add','--',*FILES)
        git(plugin,'commit','--only','-m','feat: opt-in Iluvatar top-p row parallel launcher','--',*FILES)
        commit=git(plugin,'rev-parse','HEAD');branch=git(plugin,'branch','--show-current')
        assert branch=='leanscale/iluvatar-topp-row-parallel'
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
    for f in run.iterdir():
        if f.is_file() and f.suffix in ['.json','.jsonl','.log']:
            t=dest/f.name;t.write_text('\n'.join(l.rstrip() for l in f.read_text(errors='replace').splitlines())+'\n')
            manifest[f.name]=dict(raw_sha256=sha(f),archived_sha256=sha(t))
    write(dest/'artifact_manifest.json',manifest);write(EXP/'STATUS.json',status)
    paths=['experiments/EXP-0027/validation/'+run.name,'experiments/EXP-0027/STATUS.json','experiments/EXP-0027/steps.jsonl']
    if (EXP/'candidate_commits.json').exists():paths.append('experiments/EXP-0027/candidate_commits.json')
    try:pub=publish(paths,'experiments: archive top-p launcher integration checks')
    except Exception as exc:pub=dict(status='push_failed_local_saved',error=repr(exc))
    write(run/'publication.json',pub);write(ART/'validation_publication.json',pub)
    print(json.dumps(dict(status=status,publication=pub)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path);args=parser.parse_args()
    if args.run:run_validation(args.run)
    else:prepare()
