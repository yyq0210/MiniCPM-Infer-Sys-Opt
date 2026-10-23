"""Launch the authorized diagnostic in a fresh, conservative three-hour window.

No shutdown time is invented: only today's noon startup is user-confirmed.
No further GPU job may inherit this experiment-specific window.
"""
from datetime import datetime, timedelta
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from run_profile import ART, EXP, REQUEST, ROOT, atomic, read, verify_sources

def main():
    lock=(ART/'EXP-0018-start.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not REQUEST.exists(),'An EXP18 attempt already exists; inspect it instead of relaunching'
    prep=read(ART/'EXP-0018/preparation.json')
    assert prep['publication']['remote_verified'] and prep['checks']>0
    verify_sources()
    now=datetime.now().astimezone()
    assert now.strftime('%Y-%m-%d')=='2026-10-23' and now.hour>=12,'This launcher is dated to the confirmed restart'
    # Finish before the earliest previously mentioned 18:00 shutdown as well
    # as bounding the entire diagnostic to three hours. This is our runtime
    # cap, not a claim that today's physical shutdown has been confirmed.
    cutoff=min(now+timedelta(hours=3),now.replace(hour=17,minute=35,second=0,microsecond=0))
    assert (cutoff-now).total_seconds()>2*4800+120,'Not enough time for both bounded passes'
    old=read(ART/'run_window.json')
    atomic(ART/'EXP-0018/previous_run_window.json',old)
    policy=dict(timezone='Asia/Shanghai',user_notice='2026-10-23 noon startup confirmed; today shutdown time remains unknown',
                compute_stop_at=cutoff.isoformat(),archive_target_at=(cutoff+timedelta(minutes=15)).isoformat(),
                machine_shutdown_at=None,user_confirmed_shutdown=False,experiment='EXP-0018',
                scope='Two selected-version 4K profiling passes only; conservative agent-chosen runtime cap',
                resume_policy='Do not automatically extend this window or apply it to subsequent GPU experiments')
    atomic(ART/'run_window.json',policy)
    atomic(EXP/'launch_window.json',policy)
    with (EXP/'steps.jsonl').open('a') as f:
        f.write(json.dumps(dict(time=now.isoformat(),step='Use fresh bounded daytime window for already-authorized diagnostics; shutdown time still unknown',policy=policy),ensure_ascii=False)+'\n')
    with (ART/'EXP-0018/controller.log').open('w') as log:
        p=subprocess.Popen(['python3',str(EXP/'run_profile.py')],cwd=EXP,stdin=subprocess.DEVNULL,
                           stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                           env=dict(os.environ,PYTHONUNBUFFERED='1'))
    time.sleep(3)
    assert p.poll() is None or p.returncode==0, 'Controller failed at launch; inspect saved controller.log'
    receipt=dict(pid=p.pid,launched_at=now.isoformat(),window=policy,request_state=read(REQUEST))
    atomic(ART/'EXP-0018/launch.json',receipt)
    active=read(ART/'active_inference_plan.json')
    active.update(current_experiment='EXP-0018',status='Selected-version 4K diagnostics launched in bounded daytime window')
    atomic(ART/'active_inference_plan.json',active)
    print(json.dumps(receipt,ensure_ascii=False))

if __name__=='__main__':main()
