from pathlib import Path
from datetime import datetime
import ast, hashlib, json, shutil, subprocess

ROOT = Path('/workspace/LeanScale')
DOC = ROOT/'MiniCPM-Infer-Sys-Opt'
EXP = DOC/'experiments/EXP-0024'
WORK = ROOT/'worktrees/EXP-0024'
SRC = Path(__file__).resolve().parent
PARENT = {'vllm-plugin-FL':'ce19f93eee4c73f5a26911554700457e879f891d',
          'FlagGems':'de958ddfc4deeb43d6c94bb8652a3d4d3bc905fe'}

def git(repo, *args):
    return subprocess.check_output(['git','-C',str(repo),*args], text=True, stderr=subprocess.STDOUT).strip()
def step(action, **kw):
    with (EXP/'steps.jsonl').open('a') as f:
        f.write(json.dumps(dict(time=datetime.now().astimezone().isoformat(),action=action,**kw))+'\n')

def main():
    assert (EXP/'README.md').exists() and not WORK.exists()
    assert json.loads((ROOT/'artifacts/EXP-0023/expanded-r1/review_publication.json').read_text())['remote_verified']
    assert json.loads((ROOT/'artifacts/active_inference_plan.json').read_text())['selected_serving_commits'] == PARENT
    step('Begin isolated implementation after published EXP24 hypothesis',
         change='post-load parameter storage only; register env in compile factors before config hashing')
    WORK.mkdir()
    for name, commit in PARENT.items():
        base=ROOT/'worktrees/EXP-0020'/name
        assert git(base,'rev-parse','HEAD') == commit and not git(base,'status','--porcelain')
        args = ['worktree','add']
        args += ['-b','leanscale/iluvatar-mlp-weight-layout'] if name=='vllm-plugin-FL' else ['--detach']
        git(base,*args,str(WORK/name),commit)
    plugin=WORK/'vllm-plugin-FL'
    shutil.copy2(SRC/'mlp_weight_layout.py',plugin/'vllm_fl/patches/mlp_weight_layout.py')
    shutil.copy2(SRC/'test_mlp_weight_layout.py',plugin/'tests/test_mlp_weight_layout.py')
    path=plugin/'vllm_fl/__init__.py';old=path.read_text()
    needle='    _patch_custom_ops()\n'
    assert old.count(needle)==1
    new=old.replace(needle, '    from vllm_fl.patches.mlp_weight_layout import register_mlp_weight_layout_env\n    register_mlp_weight_layout_env()\n\n'+needle)
    needle='    install_rope_kv_fusion()\n'
    assert new.count(needle)==1
    new=new.replace(needle,needle+'\n    from vllm_fl.patches.mlp_weight_layout import install_mlp_weight_layout\n    install_mlp_weight_layout()\n')
    path.write_text(new)
    for p in [path, plugin/'vllm_fl/patches/mlp_weight_layout.py', plugin/'tests/test_mlp_weight_layout.py']:
        ast.parse(p.read_text())
    git(plugin,'diff','--check')
    shutil.copy2(Path(__file__),EXP/Path(__file__).name)
    step('Implement opt-in bounded MLP layout and original-first idempotent load hook',
         parent_commits=PARENT, plugin_branch=git(plugin,'branch','--show-current'),
         cache_key='Register VLLM_FL_MLP_WEIGHT_LAYOUT in vllm.envs.environment_variables before config hashing',
         status='unvalidated; serving selection unchanged')
    print(json.dumps(dict(worktree=str(WORK),status=git(plugin,'status','--short'))),flush=True)

if __name__=='__main__':main()
