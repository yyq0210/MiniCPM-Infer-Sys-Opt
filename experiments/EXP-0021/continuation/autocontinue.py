"""Wait for completed 16K publication, audit once, then launch fixed ABBA."""
from pathlib import Path
import datetime
import os
import signal
import subprocess
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *

DEST=EXP/'continuation';STATE={}
def update(stage,**kw):
    STATE.update(stage=stage,updated_at=now().isoformat(),**kw)
    write(ART/'continuation_status.json',STATE)
    write(DEST/'status.json',STATE)

def command(args,name):
    with (DEST/(name+'.log')).open('w') as log:
        child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        try:
            code=child.wait(timeout=360)
            assert code==0,(name,code)
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise

def main():
    assert not (ART/'continuation_status.json').exists(),'Inspect prior continuation instead of restarting'
    assert now().strftime('%Y-%m-%d')=='2026-10-26'
    assert read(ART/'preparation.json')['publication']['remote_verified']
    limit=now().replace(hour=16,minute=35,second=0,microsecond=0)
    assert now()<limit
    STATE.update(pid=os.getpid(),process_start_ticks=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()[19],
        started_at=now().isoformat(),waiting_deadline=limit.isoformat(),script_sha256=sha(Path(__file__)),
        scope='CPU wait/audit, then one new235min ABBA allocation only after all gates; no repeat benchmarks')
    update('waiting_for_16k_publication')
    try:
        run=Path(read(ROOT/'artifacts/EXP-0020-latest-16k.json')['run_directory'])
        while now()<limit:
            status=read(run/'status.json');handler=read(run/'completion_handler_status.json')
            if status['stage']=='failed':raise RuntimeError('16K failed; no ABBA launch')
            if status['stage']=='performance_case_complete' and handler.get('complete_results') and handler.get('remote_verified') and handler.get('status')=='pushed':break
            time.sleep(15)
        else:raise TimeoutError('16K publication not ready before CPU continuation deadline')
        assert not (REPO/'experiments/EXP-0020/review_16k.json').exists(),'An audit exists; inspect it rather than duplicate'
        assert not (ART/'status.json').exists() and not (ART/'launch.json').exists()
        update('auditing_16k',source_run=run.name,official_publication=handler)
        command(['python3',str(REPO/'experiments/EXP-0020/reviews/review_case.py'),'--case','16k'],'review_16k')
        review=read(REPO/'experiments/EXP-0020/review_16k.json')
        assert review['status']=='complete_case_audited' and review['candidate_commits']==COMMITS['B']
        m=review['metrics']
        if not (m['Total tok/s']['change_percent']>=-3 and m['Mean TTFT (ms)']['change_percent']<=5 and m['Mean TPOT (ms)']['change_percent']<=5):
            update('review_requires_analysis_no_abba',metrics=m);return
        for arm in COMMITS:verify_sources(arm)
        update('launching_abba_after_complete_audits',metrics=m)
        command(['python3',str(EXP/'start.py')],'start_abba')
        update('abba_started',metrics=m,launch=read(ART/'launch.json'))
    except BaseException:
        update('failed_no_automatic_retry',error=traceback.format_exc());traceback.print_exc()
    finally:
        try:
            publication=publish(['experiments/EXP-0021/continuation'],'experiments: archive gated 16K-to-ABBA continuation')
            write(ART/'continuation_publication.json',publication)
        except BaseException:
            write(ART/'continuation_publication.json',dict(status='local_saved_push_failed',error=traceback.format_exc()))

if __name__=='__main__':main()
