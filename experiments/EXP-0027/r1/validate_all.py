"""Run existing numerical integration checks and new diagnostic checks in fresh processes."""
from pathlib import Path
import json,os,subprocess,sys
run=Path(os.environ['LEAN_SCALE_VALIDATION_DIR'])
plugin=Path('/workspace/LeanScale/worktrees/EXP-0027-r1/vllm-plugin-FL')
checks=[]
for label,name,expected in [('standard','standalone_topp_row_parallel.py',30),('shadow','standalone_topp_shadow_budget.py',9)]:
    folder=run/label;folder.mkdir(exist_ok=False)
    env=dict(os.environ,LEAN_SCALE_VALIDATION_DIR=str(folder),VLLM_FL_TOPP_ROW_PARALLEL='0',VLLM_FL_TOPP_ROW_PARALLEL_SHADOW='0')
    with (folder/'validation.log').open('w') as log:
        subprocess.run([sys.executable,str(plugin/'tests'/name)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads((folder/'result.json').read_text());assert result['status']=='passed' and result['checks']==expected,result
    rows=[json.loads(l) for l in (folder/'checks.jsonl').read_text().splitlines()]
    assert len(rows)==expected and all(r['passed'] for r in rows)
    checks.extend([dict(r,suite=label) for r in rows])
(run/'checks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in checks))
(run/'result.json').write_text(json.dumps(dict(status='passed',checks=len(checks),suites={'standard':30,'shadow':9}),indent=2)+'\n')
print(json.dumps(dict(status='passed',checks=len(checks))),flush=True)
