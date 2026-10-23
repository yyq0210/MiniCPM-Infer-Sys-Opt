"""Issue a fresh EXP19-only microbenchmark budget; never inherit EXP18 policy."""
from datetime import datetime,timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/workspace/LeanScale');EXP=ROOT/'MiniCPM-Infer-Sys-Opt/experiments/EXP-0019';ART=ROOT/'artifacts/EXP-0019'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def main():
    lock=(ART/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'artifacts/EXP-0019-request.json').exists() and not (ART/'launch.json').exists()
    prep=read(ART/'preparation.json');assert prep['remote_verified']
    assert read(ROOT/'artifacts/EXP-0018-request.json')['stage']=='complete'
    assert read(ROOT/'artifacts/EXP-0018/review_publication.json')['remote_verified']
    manifest=read(EXP/'script_manifest.json')
    assert {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in EXP.glob('*.py')}==manifest
    now=datetime.now().astimezone();assert now.strftime('%Y-%m-%d')=='2026-10-23'
    end=min(now+timedelta(minutes=40),now.replace(hour=16,minute=25,second=0,microsecond=0))
    assert (end-now).total_seconds()>35*60,'New screen needs at least35minutes before daytime cap'
    policy=dict(experiment='EXP-0019',timezone='Asia/Shanghai',created_at=now.isoformat(),compute_stop_at=end.isoformat(),
                archive_target_at=(end+timedelta(minutes=10)).isoformat(),machine_shutdown_at=None,user_confirmed_shutdown=False,
                user_notice='Today noon startup confirmed; actual shutdown time remains unknown',
                scope='New independent budget: four prefill-memory microbenchmark configurations only, at most40minutes and before16:25',
                resume_policy='No automatic restart, extension or subsequent GPU experiment')
    write(ART/'run_window.json',policy);write(EXP/'screen_window.json',policy)
    with (EXP/'steps.jsonl').open('a') as f:f.write(json.dumps(dict(time=now.isoformat(),action='Issue independent bounded EXP19 microbenchmark budget after CPU freeze; do not inherit or extend EXP18 window',policy=policy),ensure_ascii=False)+'\n')
    run=ART/now.strftime('%Y%m%d_%H%M%S_screen')
    with (ART/'controller.log').open('w') as log:
        p=subprocess.Popen(['python3',str(EXP/'run_screen.py'),'--run-dir',str(run)],cwd=EXP,
                           stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                           env=dict(os.environ,PYTHONUNBUFFERED='1'))
    for _ in range(100):
        if (run/'status.json').exists():break
        assert p.poll() is None,'Controller exited before status';time.sleep(.2)
    else:raise RuntimeError('Missing controller receipt')
    receipt=dict(pid=p.pid,run_directory=str(run),state=read(run/'status.json'),window=policy)
    write(ART/'launch.json',receipt);print(json.dumps(receipt))

if __name__=='__main__':main()
