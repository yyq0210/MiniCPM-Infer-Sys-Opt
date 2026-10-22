"""Prepare and freeze EXP15, then launch one serialized prefill sweep."""
from pathlib import Path
import ast
import datetime
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import time

ROOT=Path('/workspace/LeanScale');REPO=ROOT/'MiniCPM-Infer-Sys-Opt'
EXP=REPO/'experiments/EXP-0015';WT=ROOT/'worktrees/EXP-0015'
STAGING=Path(__file__).resolve().parent
REL='src/flag_gems/runtime/backend/_iluvatar/fused/paged_gqa.py'
PARENTS={'vllm-plugin-FL':'82c6a143f6984f946df42125f1a8699799e8045f',
         'FlagGems':'1c526ca99d412e203ed3caa8b1663566976ea38c'}


def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args],text=True,stderr=subprocess.STDOUT,timeout=120).strip()


def write(p,x):
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')


def step(action,**kw):
    with (EXP/'steps.jsonl').open('a') as f:
        f.write(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),action=action,**kw),ensure_ascii=False)+'\n')


def main():
    assert not EXP.exists() and not WT.exists(), 'Inspect existing experiment instead of overwriting'
    active=json.loads((ROOT/'artifacts/active_inference_plan.json').read_text())
    assert active['selected_experiment']=='EXP-0014' and active['selected_serving_commits']==PARENTS
    assert json.loads((ROOT/'artifacts/EXP-0014-queue.json').read_text())['stage']=='complete'
    selection=json.loads((REPO/'experiments/EXP-0014/selection.json').read_text())
    assert len(selection['case_results'])==2
    policy=json.loads((ROOT/'artifacts/run_window.json').read_text())
    assert (datetime.datetime.fromisoformat(policy['compute_stop_at'])-datetime.datetime.now().astimezone()).total_seconds()>600
    assert not git(REPO,'status','--porcelain')
    for name,sha in PARENTS.items():
        source=ROOT/'worktrees/EXP-0014'/name
        assert git(source,'rev-parse','HEAD')==sha and not git(source,'status','--porcelain')
    EXP.mkdir();WT.mkdir(parents=True)
    shutil.copy2(STAGING/'README.md',EXP/'README.md')
    write(EXP/'parent_commits.json',PARENTS);write(EXP/'run_window_at_setup.json',policy)
    step('record_hypothesis',parents=PARENTS,configurations=12,selected_for_serving=False)
    for name,sha in PARENTS.items():
        base=ROOT/'worktrees/EXP-0014'/name
        opts=['-b','leanscale/iluvatar-gqa-prefill-tuning'] if name=='FlagGems' else ['--detach']
        git(base,'worktree','add',*opts,str(WT/name),sha)
    path=WT/'FlagGems'/REL;old=path.read_text();source=old
    edits=[('decode_warps=4, decode_stages=2, decode_layout=0, decode_rows=16):',
            'decode_warps=4, decode_stages=2, decode_layout=0, decode_rows=16,\n              prefill_warps=4, prefill_stages=2):'),
           ('*args, prefill_tile, prefill_rows, False, 0, num_warps=4, num_stages=2)',
            '*args, prefill_tile, prefill_rows, False, 0,\n            num_warps=prefill_warps, num_stages=prefill_stages)')]
    for before,after in edits:
        assert source.count(before)==1
        source=source.replace(before,after)
    tree=ast.parse(source);parent_tree=ast.parse(old)
    reversed_source=source
    for before,after in reversed(edits):reversed_source=reversed_source.replace(after,before)
    assert reversed_source==old
    old_nodes={n.name:n for n in parent_tree.body if isinstance(n,ast.FunctionDef)}
    new_nodes={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    for name in ('_attention','_reduce'):
        assert ast.dump(old_nodes[name])==ast.dump(new_nodes[name])
    defaults=lambda n:{a.arg:ast.literal_eval(v) for a,v in zip(n.args.kwonlyargs,n.args.kw_defaults)}
    a,b=defaults(old_nodes['paged_gqa']),defaults(new_nodes['paged_gqa'])
    assert b.pop('prefill_warps')==4 and b.pop('prefill_stages')==2 and b==a
    path.write_text(source)
    step('add_prefill_launch_parameters',defaults_unchanged=True,attention_and_reduce_unchanged=True)
    write(EXP/'source_provenance.json',dict(parent_source_sha256=hashlib.sha256(old.encode()).hexdigest(),
        candidate_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        inverse_patch_exact=True,jit_kernel_ast_unchanged=True,original_defaults_unchanged=True))
    parent_dir=EXP/'parent_exp13';parent_dir.mkdir()
    for name in ('trial.py','run_screen.py'):
        shutil.copy2(REPO/'experiments/EXP-0013'/name,parent_dir/name)
    for name in ('build_scripts.py','trial.py','run_screen.py','prepare_and_launch.py'):
        ast.parse((STAGING/name).read_text());shutil.copy2(STAGING/name,EXP/name)
    frozen_scripts={n:hashlib.sha256((EXP/n).read_bytes()).hexdigest() for n in ('trial.py','run_screen.py')}
    # Regeneration uses the archived parent scripts; prove the deployed files match the documented derivation.
    subprocess.run(['python3',str(EXP/'build_scripts.py')],check=True)
    assert frozen_scripts=={n:hashlib.sha256((EXP/n).read_bytes()).hexdigest() for n in frozen_scripts}
    write(EXP/'script_manifest.json',frozen_scripts)
    (EXP/'PR_DRAFT_FlagGems.md').write_text('''# Draft

标题：【FlagOS开放计算全球挑战赛S2.赛题2】【LeanScale】探索天数GQA prefill launch参数

在已验证的EXP-0014上暴露独立prefill warps/stages，默认4/2不变；内核公式、decode布局、归约、完整KV及原调度保持。对比rows32/64/128、warps2/4、stages1/2；当前仅实验接口，参数筛选、扩大验证、真实模型接入及4K/16K端到端待完成。

保留Draft；不宣称完整模型任务精度验收完成。原始基线和全部历史结果保留。
''')
    g=WT/'FlagGems';git(g,'diff','--check');git(g,'add','--',REL)
    git(g,'commit','--only','-m','feat(iluvatar): expose independent GQA prefill launch parameters','--',REL)
    commits=dict(PARENTS,FlagGems=git(g,'rev-parse','HEAD'))
    git(g,'push','leanscale','HEAD:refs/heads/leanscale/iluvatar-gqa-prefill-tuning')
    assert git(g,'ls-remote',git(g,'remote','get-url','--push','leanscale'),'refs/heads/leanscale/iluvatar-gqa-prefill-tuning').split()[0]==commits['FlagGems']
    write(EXP/'candidate_commits.json',commits)
    step('freeze_and_push_candidate',commits=commits,script_manifest=frozen_scripts)
    with (ROOT/'artifacts/docs-publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert not git(REPO,'diff','--cached','--name-only')
        forbidden=re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|lab_inst_[A-Za-z0-9]+|ssh\.flagos\.io')
        for p in EXP.rglob('*'):
            if p.is_file():assert not forbidden.search(p.read_bytes()),str(p)
        rel='experiments/EXP-0015';git(REPO,'add','--',rel);git(REPO,'diff','--cached','--check')
        git(REPO,'commit','--only','-m','experiments: prepare GQA prefill launch parameter screening','--',rel)
        head=git(REPO,'rev-parse','HEAD');git(REPO,'push','origin','main')
        assert git(REPO,'ls-remote',git(REPO,'remote','get-url','--push','origin'),'refs/heads/main').split()[0]==head
    active.update(current_experiment='EXP-0015',candidate_commits=commits,
                  status='GQA prefill screening; EXP-0014 remains selected performance baseline')
    write(ROOT/'artifacts/active_inference_plan.json',active)
    run=ROOT/'artifacts/EXP-0015'/datetime.datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_screen')
    with (ROOT/'artifacts/EXP-0015-controller.log').open('a') as log:
        child=subprocess.Popen(['python3',str(EXP/'run_screen.py'),'--run-dir',str(run)],
            stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    for _ in range(100):
        if (run/'status.json').exists():break
        if child.poll() is not None:raise RuntimeError('Controller exited before receipt')
        time.sleep(.2)
    else:raise RuntimeError('No controller receipt')
    receipt=json.loads((run/'status.json').read_text())
    write(ROOT/'artifacts/EXP-0015-launch.json',dict(documentation_commit=head,candidate_commits=commits,**receipt))
    print(json.dumps(dict(documentation_commit=head,candidate_commits=commits,run_directory=str(run),
                         pid=receipt['pid'],process_start_ticks=receipt['process_start_ticks'],stage=receipt['stage'])),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if EXP.exists():step('preparation_failed',error=repr(exc))
        raise
