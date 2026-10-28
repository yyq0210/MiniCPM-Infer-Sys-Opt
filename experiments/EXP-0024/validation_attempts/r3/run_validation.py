"""Bounded diagnostic validation; save failures and publish exact source snapshots."""
from pathlib import Path
from datetime import datetime, timedelta
import fcntl, hashlib, json, os, re, shutil, signal, socket, subprocess, time, traceback

ROOT=Path('/workspace/LeanScale'); DOC=ROOT/'MiniCPM-Infer-Sys-Opt'
EXP=DOC/'experiments/EXP-0024'; WORK=ROOT/'worktrees/EXP-0024'
ART=ROOT/'artifacts/EXP-0024'; ATTEMPT='r3'
PLUGIN=WORK/'vllm-plugin-FL'
FILES=['vllm_fl/__init__.py','vllm_fl/patches/mlp_weight_layout.py','tests/test_mlp_weight_layout.py']
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(repo,*a):return subprocess.check_output(['git','-C',str(repo),*a],text=True,stderr=subprocess.STDOUT,timeout=120).strip()
def step(action,**kw):
    with (EXP/'steps.jsonl').open('a') as f:f.write(json.dumps(dict(time=datetime.now().astimezone().isoformat(),action=action,**kw))+'\n')
def stop(p):
    if p is not None and p.poll() is None:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
def main():
    ART.mkdir(exist_ok=True)
    now=datetime.now().astimezone();assert now.strftime('%Y-%m-%d')=='2026-10-28'
    run=ART/('validation_'+ATTEMPT);run.mkdir(exist_ok=False)
    end=min(now+timedelta(minutes=15),now.replace(hour=23,minute=35,second=0,microsecond=0))
    assert (end-now).total_seconds()>10*60
    status=dict(status='preparing',started_at=now.isoformat(),compute_stop_at=end.isoformat(),
                pid=os.getpid(),process_start_ticks=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()[19])
    write(run/'status.json',status)
    expected={f:sha(PLUGIN/f) for f in FILES}
    write(run/'source_hashes.json',expected)
    for f in FILES:
        target=run/'source'/f;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(PLUGIN/f,target)
    shutil.copy2(Path(__file__),run/'run_validation.py')
    lock=(ROOT/'../experiments/codex_20261019/runner.lock').resolve().open('a')
    child=None;codes={};observations=[]
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for port in (9031,9032):
            with socket.socket() as s:assert s.connect_ex(('127.0.0.1',port))!=0
        for _ in range(2):
            out=subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True,timeout=20)
            m=re.search(r'(\d+)MiB\s*/\s*(\d+)MiB\s*\|\s*(\d+)%',out)
            assert m and int(m[1])<=1024 and int(m[3])<=5
            observations.append(dict(memory_mib=int(m[1]),util_percent=int(m[3])));time.sleep(3)
        write(run/'idle_evidence.json',observations)
        env=dict(os.environ,PYTHONPATH=f'{PLUGIN}:{WORK}/FlagGems/src',VLLM_PLUGINS='fl',
                 TRITON_CACHE_DIR=str(run/'triton_cache'),TORCHINDUCTOR_CACHE_DIR=str(run/'inductor_cache'),
                 VLLM_CACHE_ROOT=str(run/'vllm_cache'),PYTHONUNBUFFERED='1')
        # Run the archived test file outside repository conftest side effects.
        test=run/'source/tests/test_mlp_weight_layout.py'
        for label,enabled,extra in [('cpu','0',[]),('gpu','1',['-k','real_gpu'])]:
            status.update(status=label+'_running');write(run/'status.json',status)
            command=['python3','-m','pytest','-q','-s','-p','no:cacheprovider',str(test),*extra]
            write(run/(label+'.command.json'),command)
            with (run/(label+'.log')).open('w') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=run,
                                       env=dict(env,LEANSCALE_TEST_GPU=enabled),start_new_session=True)
                timeout=(end-datetime.now().astimezone()).total_seconds()
                codes[label]=child.wait(timeout=max(1,timeout))
            assert codes[label]==0, label+' checks failed'
        assert all(sha(PLUGIN/f)==h for f,h in expected.items()), 'Source changed during checks'
        status.update(status='passed',exit_codes=codes)
        assert re.search(r'2 passed', (run/'gpu.log').read_text())
    except BaseException:
        status.update(status='failed',exit_codes=codes,error=traceback.format_exc())
    finally:
        stop(child);lock.close()
        status.update(finished_at=datetime.now().astimezone().isoformat(),owned_children_reaped=True)
        write(run/'status.json',status)
    if status['status']=='passed':
        assert not git(PLUGIN,'diff','--cached','--name-only')
        git(PLUGIN,'add','--',*FILES);git(PLUGIN,'diff','--cached','--check')
        git(PLUGIN,'commit','--only','-m','feat: opt-in MiniCPM MLP weight layout on Iluvatar','--',*FILES)
        commit=git(PLUGIN,'rev-parse','HEAD');branch=git(PLUGIN,'branch','--show-current')
        git(PLUGIN,'push','-u','origin',branch)
        assert git(PLUGIN,'ls-remote',git(PLUGIN,'remote','get-url','--push','origin'),'refs/heads/'+branch).split()[0]==commit
        commits={'vllm-plugin-FL':commit,'FlagGems':git(WORK/'FlagGems','rev-parse','HEAD')}
        write(EXP/'candidate_commits.json',commits)
        write(run/'code_publication.json',dict(commits=commits,remote_verified=True,branch=branch))
    step('Complete bounded CPU/GPU load-hook validation',attempt=ATTEMPT,result=status)
    dest=EXP/'validation_attempts'/ATTEMPT;dest.mkdir(parents=True,exist_ok=False)
    for f in run.rglob('*'):
        if not f.is_file() or any(x in f.parts for x in ['__pycache__','triton_cache','inductor_cache','vllm_cache']):continue
        target=dest/f.relative_to(run);target.parent.mkdir(parents=True,exist_ok=True)
        if f.suffix=='.log':
            target.write_text('\n'.join(line.rstrip() for line in f.read_text().splitlines())+'\n')
        else:shutil.copy2(f,target)
    write(dest/'raw_artifact_hashes.json',{
        str(p.relative_to(run)):sha(p) for p in run.rglob('*')
        if p.is_file() and not any(x in p.parts for x in ['__pycache__','triton_cache','inductor_cache','vllm_cache'])})
    write(EXP/'STATUS.json',dict(stage='load_hook_validation_'+status['status'],serving_selected='EXP-0020',attempt=ATTEMPT))
    with (ROOT/'artifacts/docs-publish.lock').open('a') as pub:
        fcntl.flock(pub,fcntl.LOCK_EX);assert not git(DOC,'diff','--cached','--name-only')
        paths=['experiments/EXP-0024']
        forbidden=re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|lab_inst_[A-Za-z0-9]+|ssh\.flagos\.io|GPU-[0-9a-f]{8}')
        for p in EXP.rglob('*'):
            if p.is_file():assert not forbidden.search(p.read_bytes()),str(p)
        git(DOC,'add','--',*paths);git(DOC,'diff','--cached','--check')
        git(DOC,'commit','--only','-m','experiments: validate MLP load-time layout integration','--',*paths)
        commit=git(DOC,'rev-parse','HEAD');git(DOC,'push','origin','main')
        assert git(DOC,'ls-remote',git(DOC,'remote','get-url','--push','origin'),'refs/heads/main').split()[0]==commit
        write(run/'publication.json',dict(commit=commit,remote_verified=True))
    print(json.dumps(status),flush=True)

if __name__=='__main__':main()
