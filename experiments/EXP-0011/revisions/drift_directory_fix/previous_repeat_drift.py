"""Three predeclared bracketed repetitions of the one unstable extended point."""
from pathlib import Path
import datetime
import fcntl
import json
import sys
import time
import torch
from trial import make,timed,oracle,baseline,candidate

root=Path('/workspace/LeanScale');output=Path(sys.argv[1]);output.parent.mkdir(parents=True,exist_ok=False)
policy=json.loads((root/'artifacts/run_window.json').read_text())
deadline=datetime.datetime.fromisoformat(policy['compute_stop_at'])
r={'status':'running','reason':'B8/L17408 baseline drift 3.54097% in extended sweep; retain original and all three repeats',
   'config':{'decode_warps':4,'decode_stages':1,'segments':16},'timings':[],
   'kind':'diagnostic_not_official_e2e','started_at':datetime.datetime.now().astimezone().isoformat()}
def save():
 p=output.with_suffix('.tmp');p.write_text(json.dumps(r,indent=2)+'\n');p.replace(output)
lock=Path('/workspace/experiments/codex_20261019/runner.lock').open('a')
try:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert (deadline-datetime.datetime.now().astimezone()).total_seconds()>600
 torch.manual_seed(20261023);qs=[1]*8;ls=[17408-i for i in range(8)];x=make(qs,ls)
 old=lambda:baseline.paged_gqa(**x);new=lambda:candidate.paged_gqa(**x,**r['config'])
 old();ref=x['out'].clone();new();torch.testing.assert_close(x['out'],ref,atol=3e-3,rtol=3e-2)
 r['fp32_error']=oracle(x,qs,ls);save()
 for _ in range(30):old();new()
 torch.cuda.synchronize()
 for i in range(3):
  a=timed(old);b=timed(new);c=timed(old)
  r['timings'].append({'repeat':i+1,'before':a,'candidate':b,'after':c,
      'baseline_drift_percent':(c['median_us']/a['median_us']-1)*100,
      'time_change_percent':(b['median_us']/((a['median_us']+c['median_us'])/2)-1)*100});save()
 assert all(abs(x['baseline_drift_percent'])<=3 and x['time_change_percent']<=-3 for x in r['timings'])
 r['status']='passed'
except BaseException as e:r.update(status='failed',error=repr(e));raise
finally:r['finished_at']=datetime.datetime.now().astimezone().isoformat();save();lock.close()
