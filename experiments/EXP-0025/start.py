"""One-shot launch with a fresh complete ABBA budget and an absolute daily cap."""
import datetime
import fcntl
import subprocess
import time
from common import *

def main():
    lock=(ART/'launch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ART/'status.json').exists() and not (ART/'launch.json').exists()
    prep=read(ART/'preparation.json')
    assert prep['status']=='prepared' and prep['publication']['remote_verified']
    assert read(EXP/'cpu_route_checks.json')['status']=='passed'
    for arm in COMMITS: verify_sources(arm)
    prerequisite=read(EXP/'prerequisites.json')
    assert sha(REPO/'experiments/EXP-0024/review_20261029/audit.json')==prerequisite['exp24_audit_sha256']
    assert sha(REPO/'experiments/EXP-0020/selection.json')==prerequisite['exp20_selection_sha256']
    assert sha(ROOT/'artifacts/EXP-0024/validation_r3/status.json')==prerequisite['exp24_validation_sha256']
    current=now()
    daily=read(EXP/'daily_window.json')
    assert daily['user_confirmed_shutdown'] and current.strftime('%Y-%m-%d')==daily['date']
    deadline=min(current+datetime.timedelta(minutes=235), datetime.datetime.fromisoformat(daily['compute_stop_at']))
    assert (deadline-current).total_seconds()>=235*60, 'Not enough time for full predetermined ABBA'
    policy=dict(experiment='EXP-0025', created_at=current.isoformat(), compute_stop_at=deadline.isoformat(),
        archive_target_at=min(deadline+datetime.timedelta(minutes=10), datetime.datetime.fromisoformat(daily['archive_target_at'])).isoformat(),
        daily_window=daily, scope='Fixed four sessions, 55min each; no automatic retry or selection')
    write(ART/'run_window.json', policy)
    write(EXP/'run_window_at_launch.json', policy)
    step('Allocate fresh 235min ABBA window after verified CPU preparation', policy=policy)
    pub=publish(['experiments/EXP-0025/run_window_at_launch.json', 'experiments/EXP-0025/steps.jsonl'],
        'experiments: register fresh MLP layout ABBA launch window')
    assert (deadline-now()).total_seconds()>=230*60
    with (ART/'controller.log').open('w') as log:
        child=subprocess.Popen(['python3', str(EXP/'run_abba.py')], stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
    ticks=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(') ',1)[1].split()[19]
    receipt=dict(pid=child.pid, process_start_ticks=ticks, started_at=now().isoformat(),
        deadline=deadline.isoformat(), order=ORDER, launch_publication=pub)
    write(ART/'launch.json', receipt)
    active=read(ROOT/'artifacts/active_inference_plan.json')
    assert active['selected_serving_commits']==COMMITS['A']
    active.update(current_experiment='EXP-0025', status='Same-day MLP layout official4K ABBA running',
        next_direction='Review paired 4K results before any selection change', launch_receipt=str(ART/'launch.json'))
    write(ROOT/'artifacts/active_inference_plan.json', active)
    for _ in range(100):
        if (ART/'status.json').exists(): break
        if child.poll() is not None: raise RuntimeError('Controller exited; inspect before any retry')
        time.sleep(.2)
    else: raise RuntimeError('No status yet; do not blindly restart')
    print(json.dumps(dict(launch=receipt, status=read(ART/'status.json'))), flush=True)

if __name__=='__main__': main()
