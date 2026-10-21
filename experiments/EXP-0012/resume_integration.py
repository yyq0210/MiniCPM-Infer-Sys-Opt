"""Resume setup after its source-uniqueness assertion stopped before any edit."""
from pathlib import Path
import json
import shutil
import subprocess
import datetime

root=Path('/workspace/LeanScale');exp=root/'MiniCPM-Infer-Sys-Opt/experiments/EXP-0012'
stage=root/'staging/exp12';wt=root/'worktrees/EXP-0012'
for name,sha in [('vllm-plugin-FL','07d9fa0cfdc5f09db178e29508e3c0bcb5c1e908'),('FlagGems','19aa059af4ce456ef2e34e3bb83eb949d7e7b702')]:
    assert subprocess.check_output(['git','-C',str(wt/name),'rev-parse','HEAD'],text=True).strip()==sha
    assert not subprocess.check_output(['git','-C',str(wt/name),'status','--porcelain'],text=True)
failure={'time':datetime.datetime.now().astimezone().isoformat(),'action':'Initial source-match assertion caught two identical suffixes (scratch tuple and call); stopped before edit. Narrowed exact match to indented call; original worktrees verified clean before resuming.'}
with (exp/'steps.jsonl').open('a') as f:f.write(json.dumps(failure)+'\n')
shutil.copy2(stage/'resume_integration.py',exp/'resume_integration.py')
src=(stage/'prepare_integration.py').read_text()
prefix=src[:src.index('assert not EXP.exists() and not WT.exists()')]
tail=src[src.index("rel='vllm_fl/dispatch/backends/flaggems/impl/iluvatar_gqa.py'"):]
exec(compile(prefix+tail,str(stage/'prepare_integration.py'),'exec'),{'__name__':'__main__'})
