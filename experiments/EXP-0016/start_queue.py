"""Start the reviewed official queue once; never resume or overwrite an old run."""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import shutil
import subprocess
import time

ROOT=Path('/workspace/LeanScale');REPO=ROOT/'MiniCPM-Infer-Sys-Opt';EXP=REPO/'experiments/EXP-0016'


def read(p):return json.loads(p.read_text())
def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args],text=True,stderr=subprocess.STDOUT,timeout=120).strip()


def main():
    receipt=ROOT/'artifacts/EXP-0016-queue.json'
    assert not receipt.exists(), 'Inspect existing queue; do not start twice'
    assert not (ROOT/'artifacts/EXP-0016-latest-official.json').exists()
    setup=read(ROOT/'artifacts/EXP-0016-setup.json');assert setup['status']=='prepared'
    gate=read(EXP/'candidate_validation.json')
    assert gate['candidate_commits']==setup['candidate_commits']
    for name,wanted in gate['script_hashes'].items():
        assert hashlib.sha256((EXP/name).read_bytes()).hexdigest()==wanted
    assert read(EXP/'route_cpu_checks.json')['status']=='passed'
    policy=read(ROOT/'artifacts/run_window.json')
    deadline=datetime.datetime.fromisoformat(policy['compute_stop_at'])
    assert (deadline-datetime.datetime.now().astimezone()).total_seconds()>145*60
    with (ROOT/'artifacts/docs-publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert not git('status','--porcelain')
        shutil.copy2(Path(__file__).resolve(),EXP/'start_queue.py')
        with (EXP/'steps.jsonl').open('a') as f:
            f.write(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),
                action='Start serialized official 4K then 16K queue with fixed candidates after 19 CPU gate checks',
                candidate_commits=setup['candidate_commits'],compute_stop_at=deadline.isoformat()))+'\n')
        paths=['experiments/EXP-0016/start_queue.py','experiments/EXP-0016/steps.jsonl']
        git('add','--',*paths);git('diff','--cached','--check')
        git('commit','--only','-m','experiments: start official GQA prefill E2E queue','--',*paths)
        head=git('rev-parse','HEAD');git('push','origin','main')
        assert git('ls-remote',git('remote','get-url','--push','origin'),'refs/heads/main').split()[0]==head
    with (ROOT/'artifacts/EXP-0016-queue.log').open('a') as log:
        p=subprocess.Popen(['python3',str(EXP/'queue_official.py')],stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(100):
        if receipt.exists():break
        if p.poll() is not None:raise RuntimeError('Queue exited before recording receipt')
        time.sleep(.2)
    else:raise RuntimeError('No queue receipt')
    state=read(receipt);assert state['pid']==p.pid
    launch=dict(documentation_commit=head,queue=state)
    (ROOT/'artifacts/EXP-0016-queue-launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    print(json.dumps(launch),flush=True)


if __name__=='__main__':main()
