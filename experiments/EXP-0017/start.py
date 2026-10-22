"""One-shot detached launcher; does not resume incomplete ABBA sessions."""
import datetime
import fcntl
import subprocess
import time
from common import *

def main():
    lock=(ART/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ART/'status.json').exists() and not (ART/'launch.json').exists()
    assert read(ART/'preparation.json')['status']=='prepared'
    for arm in COMMITS:verify_sources(arm)
    deadline=datetime.datetime.fromisoformat(read(ROOT/'artifacts/run_window.json')['compute_stop_at'])
    assert (deadline-now()).total_seconds()>=4*55*60+10*60
    with (ART/'controller.log').open('w') as log:
        p=subprocess.Popen(['python3',str(EXP/'run_abba.py')],stdout=log,stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL,start_new_session=True)
    ticks=(Path('/proc')/str(p.pid)/'stat').read_text().rsplit(') ',1)[1].split()[19]
    receipt=dict(pid=p.pid,process_start_ticks=ticks,started_at=now().isoformat(),deadline=deadline.isoformat(),order=ORDER)
    write(ART/'launch.json',receipt)
    for _ in range(100):
        if (ART/'status.json').exists():break
        if p.poll() is not None:raise RuntimeError('Controller exited; inspect launch receipt and controller log')
        time.sleep(.2)
    else:raise RuntimeError('Controller did not record status; do not blindly restart')
    print(json.dumps(dict(launch=receipt,status=read(ART/'status.json'))),flush=True)
if __name__=='__main__':main()
