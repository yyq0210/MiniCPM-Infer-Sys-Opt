"""Run 4K then 16K once, with frozen commits and shutdown-aware whole-case budgets."""
from pathlib import Path
import datetime
import fcntl
import json
import os
import subprocess
import time
import traceback

ROOT=Path('/workspace/LeanScale');EXP=ROOT/'MiniCPM-Infer-Sys-Opt/experiments/EXP-0016'
state={'pid':os.getpid(),'process_start_ticks':Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()[19],
       'started_at':datetime.datetime.now().astimezone().isoformat(),'completed_cases':[]}
def update(stage,**kw):
    state.update(stage=stage,updated_at=datetime.datetime.now().astimezone().isoformat(),**kw)
    p=ROOT/'artifacts/EXP-0016-queue.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(p)
lock=(ROOT/'artifacts/EXP-0016-queue.lock').open('a')
try:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    policy=json.loads((ROOT/'artifacts/run_window.json').read_text())
    finish=datetime.datetime.fromisoformat(policy['compute_stop_at'])
    expected=json.loads((EXP/'candidate_commits.json').read_text())
    for case in ('4k','16k'):
        receipt=ROOT/('artifacts/EXP-0016-latest-'+case+'.json')
        if not receipt.exists():
            needed=55*60 if case=='4k' else 90*60
            if (finish-datetime.datetime.now().astimezone()).total_seconds()<needed:
                update('deferred_for_shutdown',deferred_case=case);break
            update('launching',case=case)
            result=json.loads(subprocess.check_output(['python3',str(EXP/'launch_split.py'),'--case',case,
                '--finish-before',finish.isoformat()],text=True,timeout=120))
            run=Path(result['run_directory'])
            log=ROOT/'artifacts/EXP-0016-split-controller.log'
            if log.exists() and not (run/'coordinator.log').exists():log.rename(run/'coordinator.log')
        run=Path(json.loads(receipt.read_text())['run_directory'])
        while True:
            s=json.loads((run/'status.json').read_text());assert s.get('commits',expected)==expected
            if s['stage']=='failed':raise RuntimeError(f'{case} failed; inspect {run}')
            p=run/'completion_handler_status.json';h=json.loads(p.read_text()) if p.exists() else {}
            if h.get('status')=='handler_failed_raw_data_retained':raise RuntimeError(f'Publisher failed: {run}')
            if h.get('status') in ('pushed','push_failed_local_saved','publish_failed_local_saved','local_saved'):
                assert h['complete_results'] and s['stage']=='performance_case_complete';break
            update('waiting_for_case',case=case,run_directory=str(run),observed_stage=s['stage'])
            time.sleep(30)
        state['completed_cases'].append(case)
        gain=s['incremental_vs_exp14'][0]['throughput_change_percent']
        if gain < -3:
            update('needs_review',reason='GQA E2E regression exceeds 3% gate',case=case,gain_percent=gain);break
    else:update('complete',reason='Both full official cases archived; assess before selecting serving version')
except BaseException as e:update('failed',error=repr(e));traceback.print_exc();raise
finally:lock.close()
