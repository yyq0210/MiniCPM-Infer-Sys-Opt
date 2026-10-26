"""One-shot ABBA start after both official cases pass independent review."""
import datetime
import fcntl
import subprocess
import time
from common import *

def main():
    lock=(ART/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ART/'status.json').exists() and not (ART/'launch.json').exists()
    prep=read(ART/'preparation.json');assert prep['status']=='prepared' and prep['publication']['remote_verified']
    for arm in COMMITS:verify_sources(arm)
    official=[]
    for case in ['4k','16k']:
        a=read(REPO/f'experiments/EXP-0020/review_{case}.json')
        assert a['status']=='complete_case_audited' and a['candidate_commits']==COMMITS['B']
        m=a['metrics'];assert m['Total tok/s']['change_percent']>=-3 and m['Mean TTFT (ms)']['change_percent']<=5 and m['Mean TPOT (ms)']['change_percent']<=5
        official.append(a)
    current=now();assert current.strftime('%Y-%m-%d')=='2026-10-26'
    daily=read(REPO/'experiments/EXP-0020/launches/daily_20261026.json');assert daily['user_confirmed_shutdown']
    deadline=min(current+datetime.timedelta(minutes=235),datetime.datetime.fromisoformat(daily['compute_stop_at']))
    assert (deadline-current).total_seconds()>=235*60,'Need full predetermined ABBA budget; do not shorten protocol'
    policy=dict(experiment='EXP-0021',created_at=current.isoformat(),compute_stop_at=deadline.isoformat(),
      archive_target_at=min(deadline+datetime.timedelta(minutes=10),datetime.datetime.fromisoformat(daily['archive_target_at'])).isoformat(),
      daily_window=daily,scope='Four fixed ABBA sessions only, up to55min each; no extension or automatic retry')
    write(ART/'run_window.json',policy);write(EXP/'run_window_at_launch.json',policy)
    write(EXP/'official_prerequisites.json',official)
    step('Pass both official case gates and allocate fresh235min ABBA budget',policy=policy)
    pub=publish(['experiments/EXP-0021/run_window_at_launch.json','experiments/EXP-0021/official_prerequisites.json','experiments/EXP-0021/steps.jsonl'],
      'experiments: record complete prerequisites and fresh ABBA window')
    assert (deadline-now()).total_seconds()>=230*60
    with (ART/'controller.log').open('w') as log:
        child=subprocess.Popen(['python3',str(EXP/'run_abba.py')],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    ticks=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(') ',1)[1].split()[19]
    receipt=dict(pid=child.pid,process_start_ticks=ticks,started_at=now().isoformat(),deadline=deadline.isoformat(),order=ORDER,launch_publication=pub)
    write(ART/'launch.json',receipt)
    for _ in range(100):
        if (ART/'status.json').exists():break
        if child.poll() is not None:raise RuntimeError('Controller exited; inspect receipt before any action')
        time.sleep(.2)
    else:raise RuntimeError('No status; do not blindly restart')
    print(json.dumps(dict(launch=receipt,status=read(ART/'status.json'))),flush=True)

if __name__=='__main__':main()
