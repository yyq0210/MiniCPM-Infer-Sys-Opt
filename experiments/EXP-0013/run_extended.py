"""Deadline-bounded expanded validation; no change to the serving selection."""
from run_screen import ROOT,REPO,EXP,WT,write,stop,publish
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import os
import subprocess
import traceback

def main():
    now=datetime.datetime.now().astimezone();run=ROOT/'artifacts/EXP-0013'/now.strftime('%Y%m%d_%H%M%S_extended')
    run.mkdir(parents=True,exist_ok=False)
    prior=json.loads((ROOT/'artifacts/EXP-0013-request.json').read_text())
    assert prior['stage']=='screen_complete' and prior['safe_to_resume_gpu']
    screening=Path(prior['run_directory']);summary=json.loads((screening/'screen_summary.json').read_text())
    assert next(x for x in summary['rows'] if x['configuration']=='layout1_rows8')['global_candidate_gate_passed']
    deadline=datetime.datetime.fromisoformat(json.loads((ROOT/'artifacts/run_window.json').read_text())['compute_stop_at'])
    state={'pid':os.getpid(),'process_start_ticks':Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()[19],
       'run_directory':str(run),'screening_source':str(screening),'started_at':now.isoformat(),
       'compute_stop_at':deadline.isoformat(),'stage':'preparing','config':{'decode_warps':4,'decode_stages':1,'segments':16,'decode_layout':1,'decode_rows':8},
       'commits':json.loads((EXP/'candidate_commits.json').read_text()),'selected_for_serving':False,
       'kind':'expanded_operator_validation_not_official_e2e'}
    def update(stage,**kw):
        state.update(stage=stage,updated_at=datetime.datetime.now().astimezone().isoformat(),**kw)
        write(run/'status.json',state);write(ROOT/'artifacts/EXP-0013-expanded-request.json',state)
    lock=Path('/workspace/experiments/codex_20261019/runner.lock').open('a');child=None;locked=False
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locked=True
        remaining=(deadline-datetime.datetime.now().astimezone()).total_seconds();assert remaining>600
        for n,sha in state['commits'].items():
            assert subprocess.check_output(['git','-C',str(WT/n),'rev-parse','HEAD'],text=True).strip()==sha
            assert not subprocess.check_output(['git','-C',str(WT/n),'status','--porcelain'],text=True)
        state['validation_sha256']=hashlib.sha256((EXP/'validate_extended.py').read_bytes()).hexdigest()
        env=dict(os.environ,PYTHONPATH=f'{WT}/vllm-plugin-FL:{WT}/FlagGems/src',PYTHONUNBUFFERED='1',
                 VLLM_PLUGINS='fl',TRITON_CACHE_DIR=str(run/'triton_cache'))
        with (run/'validation.log').open('w') as log:
            child=subprocess.Popen(['python3',str(EXP/'validate_extended.py'),str(run/'validation.json')],
                env=env,cwd=run,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            update('validating',child_pid=child.pid)
            code=child.wait(timeout=min(1800,remaining))
        assert code==0,f'Validation exit {code}'
        data=json.loads((run/'validation.json').read_text());assert data['status']=='passed'
        assert len(data['checks'])==26 and len(data['timings'])==34
        regressions=[x for x in data['timings'] if x['time_change_percent']>3]
        drift=[x for x in data['timings'] if abs(x['baseline_drift_percent'])>3]
        update('validation_complete',checks=26,timing_points=34,
               regression_names=[x['name'] for x in regressions],drift_names=[x['name'] for x in drift],
               requires_review_before_integration=True)
    except BaseException as e:update('failed',error=repr(e));traceback.print_exc()
    finally:
        stop(child);update(state['stage'],owned_children_reaped=True,safe_to_resume_gpu=locked);lock.close()
        try:publish(run,state)
        except BaseException as e:write(run/'publication.json',{'status':'local_saved_push_failed','error':repr(e)});traceback.print_exc()

if __name__=='__main__':main()
