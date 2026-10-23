"""Implement an opt-in prefill load layout and freeze it; CPU preparation only."""
from pathlib import Path
import ast
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import types

ROOT=Path('/workspace/LeanScale');REPO=ROOT/'MiniCPM-Infer-Sys-Opt'
EXP=REPO/'experiments/EXP-0019';WT=ROOT/'worktrees/EXP-0019';ART=ROOT/'artifacts/EXP-0019'
STAGING=Path(__file__).resolve().parent
REL='src/flag_gems/runtime/backend/_iluvatar/fused/paged_gqa.py'
BRANCH='leanscale/iluvatar-gqa-prefill-memory'
def git(repo,*args):return subprocess.check_output(['git',*args],cwd=repo,text=True,stderr=subprocess.STDOUT,timeout=120).strip()
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def step(action,**kw):
    with (EXP/'steps.jsonl').open('a') as f:f.write(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),action=action,**kw),ensure_ascii=False)+'\n')

def check_dispatch(parent,candidate):
    records=[]
    class Jit:
        def __init__(self,fn):self.name=fn.__name__
        def __getitem__(self,grid):
            def call(*args,**kwargs):records.append((self.name,grid,args,kwargs))
            return call
    language=types.SimpleNamespace(constexpr=object)
    sys.modules['triton']=types.SimpleNamespace(jit=Jit,next_power_of_2=lambda n:1<<(n-1).bit_length(),language=language)
    sys.modules['triton.language']=language
    def module(path,name):
        spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    old=module(parent,'old_dispatch');new=module(candidate,'new_dispatch')
    class Tensor:
        def __init__(self,shape,strides=None):
            self.shape=shape;self.strides=strides or tuple(math.prod(shape[i+1:]) for i in range(len(shape)))
        def stride(self,index=None):return self.strides if index is None else self.strides[index]
        def numel(self):return math.prod(self.shape)
    cfg=dict(decode_warps=4,decode_stages=1,segments=16,decode_layout=1,decode_rows=8,
             prefill_rows=64,prefill_warps=4,prefill_stages=1,prefill_tile=32)
    checks=[]
    for qn in [1,2048]:
        args=(Tensor((2048,16,128),(2560,128,1)),Tensor((1024,16,2,128)),Tensor((1024,16,2,128)),
              Tensor((2048,16,128)),Tensor((65,)),Tensor((64,)),Tensor((64,256)),128**-.5,qn,
              Tensor((64,16,16,128)),Tensor((64,16,16)),Tensor((64,16,16)))
        records.clear();expected=old.paged_gqa(*args,**cfg);before=records.copy()
        records.clear();actual=new.paged_gqa(*args,**cfg)
        assert actual is expected is args[3] and records==before
        checks.append(f'default_dispatch_exact_maxquery{qn}')
        for layout in range(4):
            records.clear();new.paged_gqa(*args,**cfg,prefill_layout=layout)
            assert records[-2:]==before[-2:],'Decode/reduce dispatch changed'
            if qn>1:
                assert records[0][2][-2:]==(False,layout)
                assert records[0][3]==dict(num_warps=4,num_stages=1)
            checks.append(f'layout{layout}_preserves_decode_maxquery{qn}')
    # Validate effective page addresses, including partial last pages and
    # offsets beyond signed int32. Compare only lanes whose K/V loads are valid.
    for page in [8,16,32,64]:
        for length in [0,1,7,15,16,17,31,32,33,65,129]:
            for base in range(0,160,32):
                if page<=32 and 32%page==0:
                    for lane in range(32):
                        token=base+lane
                        if token<length:
                            original=(token//page+1000000)*4096+(token%page)*256
                            broadcast=(base//page+lane//page+1000000)*4096+(token%page)*256
                            assert original==broadcast and original>2**31
    checks.append('effective_page_broadcast_addresses_partial_pages_int64_CPU_model')
    return checks

def main():
    assert (EXP/'README.md').exists() and not WT.exists() and not ART.exists()
    assert read(ROOT/'artifacts/EXP-0018/review_publication.json')['remote_verified']
    parents=read(EXP/'parent_commits.json')
    active=read(ROOT/'artifacts/active_inference_plan.json');assert active['selected_experiment']=='EXP-0016' and active['selected_serving_commits']==parents
    assert not git(REPO,'status','--porcelain')
    WT.mkdir();ART.mkdir()
    try:
        step('Begin isolated CPU implementation after published hypothesis',parents=parents)
        for name,commit in parents.items():
            opts=['-b',BRANCH] if name=='FlagGems' else ['--detach']
            git(ROOT/'worktrees/EXP-0016'/name,'worktree','add',*opts,str(WT/name),commit)
        path=WT/'FlagGems'/REL;old=path.read_text();source=old
        edits=[('DECODE_LAYOUT: tl.constexpr','LOAD_LAYOUT: tl.constexpr'),
               ('if (DECODE and (DECODE_LAYOUT == 1 or DECODE_LAYOUT == 3)','if ((LOAD_LAYOUT == 1 or LOAD_LAYOUT == 3)'),
               ('if DECODE and (DECODE_LAYOUT == 2 or DECODE_LAYOUT == 3):','if LOAD_LAYOUT == 2 or LOAD_LAYOUT == 3:'),
               ('prefill_warps=4, prefill_stages=2):','prefill_warps=4, prefill_stages=2, prefill_layout=0):'),
               ('    nseq = lengths.numel()','    assert prefill_layout in (0, 1, 2, 3)\n    nseq = lengths.numel()'),
               ('*args, prefill_tile, prefill_rows, False, 0,','*args, prefill_tile, prefill_rows, False, prefill_layout,')]
        for a,b in edits:assert source.count(a)==1;source=source.replace(a,b)
        restored=source
        for a,b in reversed(edits):assert restored.count(b)==1;restored=restored.replace(b,a)
        assert restored==old
        a={n.name:n for n in ast.parse(old).body if isinstance(n,ast.FunctionDef)}
        b={n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
        assert ast.dump(a['_reduce'])==ast.dump(b['_reduce'])
        defaults=lambda n:{k.arg:ast.literal_eval(v) for k,v in zip(n.args.kwonlyargs,n.args.kw_defaults)}
        newdefaults=defaults(b['paged_gqa']);assert newdefaults.pop('prefill_layout')==0 and newdefaults==defaults(a['paged_gqa'])
        path.write_text(source)
        checks=check_dispatch(ROOT/'worktrees/EXP-0016/FlagGems'/REL,path)
        checks+=['inverse_patch_exact','reduce_AST_unchanged','old_defaults_unchanged_new_default_zero']
        for p in STAGING.iterdir():
            if p.is_file() and (p.suffix=='.py' or p.name=='source_reuse.json') and p.resolve()!=(EXP/p.name).resolve():shutil.copy2(p,EXP/p.name)
        frozen={n:sha(EXP/n) for n in ['trial.py','run_screen.py']}
        subprocess.run(['python3',str(EXP/'build_scripts.py')],check=True)
        assert frozen=={n:sha(EXP/n) for n in frozen}
        subprocess.run(['python3','-m','py_compile',str(path),*[str(p) for p in EXP.glob('*.py')]],check=True)
        checks.append('deployed_microbenchmark_regeneration_exact')
        write(EXP/'cpu_checks.json',dict(status='passed',checks=checks,
            limitation='CPU address model and dispatch tests are not GPU numeric/Graph/int64 validation'))
        write(EXP/'source_provenance.json',dict(parent_source_sha256=hashlib.sha256(old.encode()).hexdigest(),candidate_source_sha256=sha(path),
            inverse_patch_exact=True,old_defaults_preserved=True,observed_workloads_sha256=sha(EXP/'observed_workloads.json')))
        write(EXP/'script_manifest.json',{p.name:sha(p) for p in EXP.glob('*.py')})
        (EXP/'PR_DRAFT_FlagGems.md').write_text('''# Draft\n\n标题：【FlagOS开放计算全球挑战赛S2.赛题2】【LeanScale】探索天数GQA prefill分页读取与K加载布局\n\n增加prefill_layout实验接口：0保持现有路径，1页号广播，2按token/dim加载K再转置，3组合；通用默认0，decode分派及reduce保持。基于当前4K热点验证有限候选，尚未通过GPU数值/Graph/性能或接入检查，不宣称收益。服务继续使用已选EXP16。CPU派发与地址模型检查不代替GPU验证；完整模型任务精度暂缓，保持Draft。\n''')
        gems=WT/'FlagGems';git(gems,'diff','--check');git(gems,'add','--',REL)
        git(gems,'commit','--only','-m','feat(iluvatar): expose opt-in GQA prefill load layouts','--',REL)
        commits=dict(parents,FlagGems=git(gems,'rev-parse','HEAD'))
        git(gems,'push','leanscale','HEAD:refs/heads/'+BRANCH)
        assert git(gems,'ls-remote',git(gems,'remote','get-url','--push','leanscale'),'refs/heads/'+BRANCH).split()[0]==commits['FlagGems']
        write(EXP/'candidate_commits.json',commits)
        step('Freeze optional candidate and derived diagnostic scripts; CPU tests passed; no GPU work yet',checks=len(checks),commits=commits)
        write(EXP/'STATUS.json',dict(stage='cpu_prepared_gpu_unvalidated',checks=len(checks),selected_for_serving=False,candidate_commits=commits))
        with (ROOT/'artifacts/docs-publish.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            git(REPO,'add','--','experiments/EXP-0019');git(REPO,'diff','--cached','--check')
            paths=git(REPO,'diff','--cached','--name-only','--','experiments/EXP-0019').splitlines()
            assert paths and all('__pycache__' not in p for p in paths)
            git(REPO,'commit','--only','-m','experiments: freeze prefill memory candidates and profile-derived tests','--','experiments/EXP-0019')
            commit=git(REPO,'rev-parse','HEAD');git(REPO,'push','origin','main')
            assert git(REPO,'ls-remote','origin','refs/heads/main').split()[0]==commit
        write(ART/'preparation.json',dict(status='prepared',commits=commits,checks=len(checks),documentation_commit=commit,remote_verified=True))
        active.update(current_experiment='EXP-0019',candidate_commits=commits,status='Prefill memory candidates CPU-prepared; serving remains EXP16')
        write(ROOT/'artifacts/active_inference_plan.json',active)
        print(json.dumps(dict(status='prepared',commits=commits,checks=len(checks),documentation_commit=commit)))
    except BaseException:
        import traceback
        write(ART/'preparation_error.json',dict(error=traceback.format_exc()));step('CPU preparation failed',error=traceback.format_exc());raise

if __name__=='__main__':main()
