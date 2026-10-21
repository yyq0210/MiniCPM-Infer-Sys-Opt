"""Record the hypothesis, create isolated worktrees, freeze and launch EXP-0013."""
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
EXP=REPO/'experiments/EXP-0013';WT=ROOT/'worktrees/EXP-0013'
STAGING=Path(__file__).resolve().parent
REL='src/flag_gems/runtime/backend/_iluvatar/fused/paged_gqa.py'
PARENTS={'vllm-plugin-FL':'08d1711e58fdd0f4cd4f746f9e21cb59800ec1d6',
         'FlagGems':'19aa059af4ce456ef2e34e3bb83eb949d7e7b702'}

def git(repo,*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True,stderr=subprocess.STDOUT,timeout=120).strip()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def replace(s,a,b):
    assert s.count(a)==1,(a,s.count(a))
    return s.replace(a,b)
def step(action,**details):
    with (EXP/'steps.jsonl').open('a') as f:f.write(json.dumps({'time':datetime.datetime.now().astimezone().isoformat(),'action':action,**details},ensure_ascii=False)+'\n')

def main():
    assert not EXP.exists() and not WT.exists(),'An existing experiment must be inspected, not overwritten'
    active=json.loads((ROOT/'artifacts/active_inference_plan.json').read_text())
    assert active['selected_serving_commits']==PARENTS
    policy=json.loads((ROOT/'artifacts/run_window.json').read_text())
    deadline=datetime.datetime.fromisoformat(policy['compute_stop_at'])
    assert (deadline-datetime.datetime.now().astimezone()).total_seconds()>600
    assert not git(REPO,'status','--porcelain')
    # Read-only compiler evidence: CoreX accepts small M/N and may pad internally.
    compiler=Path('/usr/local/lib/python3.12/site-packages/triton/backends/iluvatar/compiler.py')
    c=compiler.read_text()
    node=next(n for n in ast.parse(c).body if isinstance(n,ast.FunctionDef) and n.name=='min_dot_size')
    compiler_excerpt='\n'.join(c.splitlines()[node.lineno-1:node.end_lineno])
    EXP.mkdir();WT.mkdir(parents=True)
    (EXP/'README.md').write_text('''# EXP-0013：GQA 分页 KV 读取与查询行布局

假设先于代码修改记录。基线为 EXP-0012 已通过官方 4K/16K 的 GQA stage1，保留 4 warps / 1 stage / 16 segments、tile32、全部 KV 与原调度。

1. 生产 page16 / tile32 的页表索引在 token 维重复。显式加载两个页号再广播，可能减少页表加载和地址生成；编译器也可能已自动完成等价优化，不能预先宣称收益。
2. 将 K 按 [token,dimension] 读取后转置为 dot 操作数，测试后端是否产生更好的读取布局；可能生成完全相同的代码或反而变慢。
3. 将 decode 逻辑查询行从 16 改为 8。CoreX min_dot_size 对 BF16/FP16 返回 M/N>=1、K>=16，但注释说明可能内部补齐，合法不代表消除硬件填充。

预声明 8 组：layout 0原样 / 1页号广播 / 2K转置读取 / 3两者结合，分别搭配 rows16/rows8；首组 layout0/rows16 为等价对照。prefill 固定原路径。
所有页号在乘 stride 前转换 int64。页广播只用于 PAGE<=TILE 且整除的 decode，其他形状走原读取路径。

每配置覆盖 BF16/FP16、NHD/HND、page8/16/32/64、空上下文与查询、边界长度、ragged/mixed、非连续Q、FP32 oracle、Graph更换Q/K/V/length/block_table。
计时为完整 GQA（主核+归约）的 Graph event，工作点 B64/L4700、B30/L16384、B1/L4096，前后基线夹测，7个样本，每次Graph含8次调用；保留全部采样和失败。
门槛沿用：基线漂移<=3%，两个主点均不退化>3%且至少一项耗时下降>=3%，小batch不退化>5%。仅筛选；获选后须扩大shape、Graph query布局、长地址验证，再决定引擎接入与官方测评。

23:35计算绝对截止，配置最长10分钟、子进程独立进程组；结果本地保留并推送。原最优服务选择保持EXP-0012，完整任务精度按用户要求暂缓。没有重启劣化调度，没有KV压缩，没有新增推理引擎接线。
''')
    write(EXP/'parent_commits.json',PARENTS)
    write(EXP/'compiler_evidence.json',{'compiler_sha256':hashlib.sha256(compiler.read_bytes()).hexdigest(),'min_dot_size_source':compiler_excerpt})
    step('record_hypothesis',parents=PARENTS,variants=8,selected_for_serving=False)
    for name,parent in PARENTS.items():
        base=ROOT/'worktrees/EXP-0012'/name
        if name=='FlagGems':git(base,'worktree','add','-b','leanscale/iluvatar-gqa-kv-layout',str(WT/name),parent)
        else:git(base,'worktree','add','--detach',str(WT/name),parent)
    source=(WT/'FlagGems'/REL).read_text()
    old=source
    source=replace(source,'TILE: tl.constexpr, BM: tl.constexpr, DECODE: tl.constexpr,',
                          'TILE: tl.constexpr, BM: tl.constexpr, DECODE: tl.constexpr,\n    DECODE_LAYOUT: tl.constexpr,')
    source=replace(source,'''        blocks = tl.load(BT + seq * BTS + tokens // PAGE,
                         tokens < length, 0).to(tl.int64)
        k = tl.load(K + blocks[None, :] * K0 + (tokens % PAGE)[None, :] * K1
                    + group * K2 + d[:, None], tokens[None, :] < length, 0)''',
'''        if (DECODE and (DECODE_LAYOUT == 1 or DECODE_LAYOUT == 3)
                and PAGE <= TILE and TILE % PAGE == 0):
            # base is TILE-aligned. Load one physical page id per logical page,
            # then expand to token lanes, preserving int64 address arithmetic.
            page_ids = base // PAGE + tl.arange(0, TILE // PAGE)
            page_blocks = tl.load(BT + seq * BTS + page_ids,
                                  page_ids * PAGE < length, 0).to(tl.int64)
            blocks = tl.reshape(tl.broadcast_to(page_blocks[:, None],
                                                (TILE // PAGE, PAGE)), (TILE,))
        else:
            blocks = tl.load(BT + seq * BTS + tokens // PAGE,
                             tokens < length, 0).to(tl.int64)
        if DECODE and (DECODE_LAYOUT == 2 or DECODE_LAYOUT == 3):
            kt = tl.load(K + blocks[:, None] * K0 + (tokens % PAGE)[:, None] * K1
                         + group * K2 + d[None, :], tokens[:, None] < length, 0)
            k = tl.trans(kt)
        else:
            k = tl.load(K + blocks[None, :] * K0 + (tokens % PAGE)[None, :] * K1
                        + group * K2 + d[:, None], tokens[None, :] < length, 0)''')
    source=replace(source,'decode_warps=4, decode_stages=2):',
                          'decode_warps=4, decode_stages=2, decode_layout=0, decode_rows=16):')
    source=replace(source,'    nseq = lengths.numel()',
                          '    assert decode_layout in (0, 1, 2, 3) and decode_rows in (8, 16)\n    nseq = lengths.numel()')
    source=replace(source,'*args, prefill_tile, prefill_rows, False, num_warps=4, num_stages=2)',
                          '*args, prefill_tile, prefill_rows, False, 0, num_warps=4, num_stages=2)')
    source=replace(source,'*args, decode_tile, 16, True,',
                          '*args, decode_tile, decode_rows, True, decode_layout,')
    ast.parse(source);(WT/'FlagGems'/REL).write_text(source)
    for n in ['build_scripts.py','trial.py','run_screen.py','prepare_and_launch.py']:
        ast.parse((STAGING/n).read_text());shutil.copy2(STAGING/n,EXP/n)
    (EXP/'PR_DRAFT_FlagGems.md').write_text('''# Draft

标题：【FlagOS开放计算全球挑战赛S2.赛题2】【LeanScale】探索天数 GQA 分页读取与查询行布局

在 EXP-0012 stage1 基线上增加实验性 decode_layout/decode_rows 参数，默认保持原路径；prefill、归约、完整KV、int64页地址及原调度保持。比较按页广播索引、K转置读取、8行逻辑查询及其组合。
单算子正确性与性能待测，扩大验证及整模测评待定；无新引擎接入，维持Draft，不宣称整模收益或完整任务精度达标。
''')
    g=WT/'FlagGems';git(g,'diff','--check');git(g,'add','--',REL)
    git(g,'commit','--only','-m','feat(iluvatar): add experimental paged GQA load and query row layouts','--',REL)
    commits=dict(PARENTS,FlagGems=git(g,'rev-parse','HEAD'))
    git(g,'push','leanscale','HEAD:refs/heads/leanscale/iluvatar-gqa-kv-layout')
    assert git(g,'ls-remote',git(g,'remote','get-url','--push','leanscale'),'refs/heads/leanscale/iluvatar-gqa-kv-layout').split()[0]==commits['FlagGems']
    write(EXP/'candidate_commits.json',commits)
    write(EXP/'source_provenance.json',{'parent_source_sha256':hashlib.sha256(old.encode()).hexdigest(),
        'candidate_source_sha256':hashlib.sha256(source.encode()).hexdigest(),'defaults_unchanged':True,
        'timing_reference':'EXP-0012 with explicit decode_warps=4, decode_stages=1, segments=16'})
    step('freeze_candidate_and_scripts',commits=commits,fork_verified=True)
    with (ROOT/'artifacts/docs-publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert not git(REPO,'diff','--cached','--name-only')
        forbidden=re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|lab_inst_[A-Za-z0-9]+|ssh\.flagos\.io')
        for p in EXP.rglob('*'):
            if p.is_file():assert not forbidden.search(p.read_bytes()),str(p)
        path='experiments/EXP-0013';git(REPO,'add','--',path);git(REPO,'diff','--cached','--check')
        git(REPO,'commit','--only','-m','experiments: prepare GQA paged KV and query row layout screening','--',path)
        head=git(REPO,'rev-parse','HEAD');git(REPO,'push','origin','main')
        assert git(REPO,'ls-remote',git(REPO,'remote','get-url','--push','origin'),'refs/heads/main').split()[0]==head
    active.update(current_experiment='EXP-0013',status='GQA paged KV/query layout screen; EXP-0012 remains selected',
                  candidate_commits=commits)
    write(ROOT/'artifacts/active_inference_plan.json',active)
    run=ROOT/'artifacts/EXP-0013'/datetime.datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_screen')
    log=(ROOT/'artifacts/EXP-0013-controller.log').open('a')
    child=subprocess.Popen(['python3',str(EXP/'run_screen.py'),'--run-dir',str(run)],
        stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    for _ in range(100):
        if (run/'status.json').exists():break
        if child.poll() is not None:raise RuntimeError('Screen controller exited before writing receipt')
        time.sleep(.2)
    else:raise RuntimeError('No screen controller receipt')
    receipt=json.loads((run/'status.json').read_text())
    write(ROOT/'artifacts/EXP-0013-launch.json',{'documentation_commit':head,'candidate_commits':commits,**receipt})
    print(json.dumps({'documentation_commit':head,'candidate_commits':commits,'run_directory':str(run),
                      'pid':receipt['pid'],'process_start_ticks':receipt['process_start_ticks'],'stage':receipt['stage']}),flush=True)

if __name__=='__main__':main()
