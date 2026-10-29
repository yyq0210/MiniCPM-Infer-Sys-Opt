"""CPU-only preparation. Record the hypothesis before adding immutable worktrees."""
from pathlib import Path
import ast
import datetime
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import traceback

ROOT = Path('/workspace/LeanScale')
REPO = ROOT/'MiniCPM-Infer-Sys-Opt'
EXP = REPO/'experiments/EXP-0025'
ART = ROOT/'artifacts/EXP-0025'
WT = ROOT/'worktrees/EXP-0025'
STAGING = Path(__file__).resolve().parent
REFS = {'A':ROOT/'artifacts/EXP-0020/20261026_125756_official_4k',
        'B':ROOT/'artifacts/EXP-0024/e2e-r1/20261028_192854_official_4k_r1'}
COMMITS = {
    'A':{'vllm-plugin-FL':'ce19f93eee4c73f5a26911554700457e879f891d', 'FlagGems':'de958ddfc4deeb43d6c94bb8652a3d4d3bc905fe'},
    'B':{'vllm-plugin-FL':'41fd7a47e3f39badb9e455fb04b2fe7c888d3db1', 'FlagGems':'de958ddfc4deeb43d6c94bb8652a3d4d3bc905fe'}}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def write(p, x): p.write_text(json.dumps(x, indent=2, ensure_ascii=False)+'\n')
def git(repo, *args): return subprocess.check_output(['git', '-C', str(repo), *args], text=True, stderr=subprocess.STDOUT, timeout=120).strip()
def once(text, old, new):
    assert text.count(old)==1, (old, text.count(old))
    return text.replace(old, new)

def main():
    assert datetime.datetime.now().astimezone().strftime('%Y-%m-%d')=='2026-10-29'
    assert not any(p.exists() for p in [EXP, ART, WT]), 'Do not overwrite partial preparation'
    assert not git(REPO, 'diff', '--cached', '--name-only')
    audit=read(REPO/'experiments/EXP-0024/review_20261029/audit.json')
    assert audit['status']=='passed' and audit['candidate_commits']==COMMITS['B']
    assert read(ROOT/'artifacts/active_inference_plan.json')['selected_serving_commits']==COMMITS['A']
    EXP.mkdir(); ART.mkdir(); WT.mkdir()
    (EXP/'README.md').write_text('''# EXP-0025：MLP 权重布局同日官方 4K ABBA

实验假设（建工作树前记录）：EXP23 完整 MLP 小 M 微基准收益约 12.63%，但 EXP24 官方 4K 历史会话比较为 -1.07%，保留轮 CV 为 5.38%。需要区分布局效果与服务会话漂移，不能从任一数字直接升级或断言稳定劣化。

A=当前 EXP20，B=EXP24 的加载时 gate/up + down 权重重排。两组 GQA、RoPE+KV、原调度、完整 KV、服务参数及官方 benchmark 不变。生产源码不修改；独立 detached worktree 固定提交。A 明确设置布局开关 0，B 设置 1；A 提交未包含该开关，原生连续权重路径由实际编译图再次验证。

固定 A1→B1→B2→A2，每会话新服务及独立 vLLM、Triton、Inductor 缓存，官方 [4096,1024,64,256] 四轮，分别弃首轮。保留全部首轮、慢轮和失败；不混入历史会话，不按中间成绩中止或挑选。每组只有两个独立服务会话，轮次不作为独立样本，不宣称统计显著。

先 CPU 检查已完成的两版真实缓存：42 个编译 MLP 分图，每个 gu/down mm 直接读原模型参数；A 转置视图 stride=(1,K)，B=(N,1)，完整 FX 各 84 权重节点。B 要求全部 84 次精确值重排，A 为 0；另核对原 42 GQA/RoPEKV、35 FULL 图及 64 decode+64 prefill 核。错误 arm 必须被拒绝。新服务正式计时前后重复实际路径检查。

共享 GPU 锁覆盖全组，服务前两次空闲及端口检查。启动分配新的 235 分钟预算，每会话 55 分钟硬上限；整组必须留足完整预算，不压缩官方轮数。今日用户确认 2026-10-29 23:35 停计算、23:50 保存、次日 00:00 关机。每次服务退出后先保存 CSV、日志、路径证据、哈希及精确 KV 容量再自动推送个人文档 fork，失败也保留。禁止上游 PR。

结果报告两组会话均值、两对相邻比较、A 首尾漂移、B 重复差异、每轮数据/CV、TTFT/TPOT、排队/抢占及 KV 容量。ABBA 只平衡近似线性时间漂移，不能消除非线性变化。KV 容量允许范围沿用启动 gate >= EXP20 历史容量的 99%，所有实际差异仍如实披露，不把容量近似相同当作完全相同。

本实验不自动切换基线。完整结果独立审核后，只有正向收益在两对比较中方向一致、延迟和容量无明显损失且漂移不足以解释结果，才考虑采用；证据模糊则保留 EXP20，继续下一项 4K 热点。整模任务精度按用户暂缓，不宣称通过；这里没有 KV 压缩。
''')
    # Derive only the small common helper; preserve source provenance.
    parent21=REPO/'experiments/EXP-0021'
    common=(parent21/'common.py').read_text().replace('EXP-0021', 'EXP-0025')
    start=common.index('COMMITS = {'); end=common.index('\nORDER = ', start)
    common=common[:start]+'COMMITS = '+repr(COMMITS)+common[end:]
    common+='\nREFS = '+repr({k:str(v) for k,v in REFS.items()})+'\nREFS = {k:Path(v) for k,v in REFS.items()}\n'
    # Avoid scanner matching its own literal private endpoint regex.
    common=common.replace("|lab_inst_[A-Za-z0-9]+|ssh\\.flagos\\.io", "|lab_inst_[A-Za-z0-9]+|ssh[.]flagos[.]io")
    (EXP/'common.py').write_text(common)
    sys.path.insert(0, str(EXP))
    import common as c
    c.step('Record same-day ABBA hypothesis before worktrees', commits=COMMITS, order=c.ORDER, GPU_started=False)
    for arm, commits in COMMITS.items():
        (WT/arm).mkdir()
        for name, commit in commits.items():
            git(ROOT/'worktrees/EXP-0024'/name, 'worktree', 'add', '--detach', str(WT/arm/name), commit)
    p20=REPO/'experiments/EXP-0020'
    assert sha(p20/'run_split_e2e.py')=='2e599679dfe365de907b5071cee469b08489b66ff371588bf78c0f1782281ab1'
    assert sha(p20/'analyze_schedule.py')=='574042b1d7d97454691b3608f2fe0073cc760a6c0314a1cd1315bc95a769ea9f'
    shutil.copy2(p20/'run_split_e2e.py', EXP/'frozen_runtime.py')
    shutil.copy2(p20/'analyze_schedule.py', EXP/'analyze_schedule.py')
    for name in ['run_abba.py', 'start.py', 'prepare.py']:
        shutil.copy2(STAGING/name, EXP/name)
    source=REPO/'experiments/EXP-0024/e2e-r1/run_e2e.py'
    code=source.read_text()
    fn=next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name=='layout_route')
    layout=ast.get_source_segment(code, fn)
    layout=once(layout, 'def layout_route(run,folder,reference):', 'def layout_route(run,folder,reference,arm):\n    assert arm in ("A","B")')
    layout=once(layout, "    assert len(matches)==84, ('Expected 84 one-time layout conversions',len(matches))", "    assert len(matches)==(84 if arm=='B' else 0), ('Unexpected layout conversion count',arm,len(matches))\n    if arm=='A': assert 'LeanScale MLP layout active:' not in log")
    layout=once(layout, "    assert {(r[0],r[1]) for r in matches}==expected", "    assert {(r[0],r[1]) for r in matches}==(expected if arm=='B' else set())")
    layout=once(layout, "            assert stride==(shape[1],1),(p.name,name,stride)", "            expected_stride=(shape[1],1) if arm=='B' else (1,shape[0])\n            assert stride==expected_stride,(p.name,name,stride)")
    layout=once(layout, "evidence=dict(status='passed',converted_gate_up=42,converted_down=42,", "evidence=dict(status='passed',arm=arm,converted_gate_up=42 if arm=='B' else 0,converted_down=42 if arm=='B' else 0,")
    layout=once(layout, 'one_time_conversions=84,', "one_time_conversions=84 if arm=='B' else 0,")
    (EXP/'layout_checks.py').write_text('import ast, re\nfrom common import sha, write\n\n'+layout+'\n')
    write(EXP/'candidate_commits.json', COMMITS['B'])
    write(EXP/'arms.json', dict(commits=COMMITS, order=c.ORDER))
    write(EXP/'source_provenance.json', {str(p.relative_to(REPO)):sha(p) for p in [parent21/'common.py', p20/'run_split_e2e.py', p20/'analyze_schedule.py', source]})
    write(EXP/'prerequisites.json', dict(exp24_audit_sha256=sha(REPO/'experiments/EXP-0024/review_20261029/audit.json'),
        exp24_audit_commit='5fa3b380cdaa50e64baef5b122d08ab09f033d98', exp20_selection_sha256=sha(p20/'selection.json'),
        exp24_validation_sha256=sha(ROOT/'artifacts/EXP-0024/validation_r3/status.json')))
    daily=dict(date='2026-10-29', user_confirmed_shutdown=True, compute_stop_at='2026-10-29T23:35:00+08:00',
        archive_target_at='2026-10-29T23:50:00+08:00', shutdown_at='2026-10-30T00:00:00+08:00')
    write(EXP/'daily_window.json', daily)
    for p in EXP.glob('*.py'): ast.parse(p.read_text(), filename=str(p))
    import frozen_runtime as rt
    assert callable(rt.compare_schedule)
    from layout_checks import layout_route
    checks=[]
    # Read actual historical caches; write evidence only into a temporary fixture.
    with tempfile.TemporaryDirectory(prefix='exp25_cpu_route_') as tmp:
        tmp=Path(tmp)
        for arm, ref in REFS.items():
            folder=tmp/arm; folder.mkdir()
            shutil.copy2(ref/'candidate_benchmark/server.log', folder/'server.log')
            rt.RUN=ref
            rt.verify_route(folder)
            gqa=read(folder/'route_evidence.json')
            assert gqa['kernel_counts']=={'decode':64, 'prefill':64}
            layout=layout_route(ref, folder, REFS['A']/'candidate_benchmark', arm)
            assert layout['compiled_mlp_blocks']==42
            write(EXP/(arm+'_historical_layout_evidence.json'), layout)
            checks.append(dict(name=arm+'_actual128_GQA_and42_MLP_blocks', passed=True,
                gqa_sha256=sha(folder/'route_evidence.json'), layout_sha256=sha(folder/'layout_route_evidence.json')))
            try:
                layout_route(ref, folder, REFS['A']/'candidate_benchmark', 'B' if arm=='A' else 'A')
            except AssertionError:
                checks.append(dict(name=arm+'_reject_opposite_layout', passed=True))
            else:
                raise AssertionError('Opposite arm accepted')
    write(EXP/'cpu_route_checks.json', dict(status='passed', checks=checks))
    write(EXP/'script_hashes.json', {p.name:sha(p) for p in EXP.glob('*.py')})
    for arm in COMMITS: c.verify_sources(arm)
    c.step('CPU preparation passed; actual historical paths verified, no GPU work', checks=len(checks))
    write(EXP/'STATUS.json', dict(stage='cpu_prepared', GPU_started=False, checks=len(checks)))
    pub=c.publish(['experiments/EXP-0025'], 'experiments: prepare same-day official 4K MLP layout ABBA')
    write(ART/'preparation.json', dict(status='prepared', checks=len(checks), publication=pub, GPU_started=False))
    print(json.dumps(dict(status='prepared', checks=checks, publication=pub)), flush=True)

if __name__=='__main__':
    try: main()
    except BaseException:
        if ART.exists(): write(ART/'preparation_error.json', dict(error=traceback.format_exc()))
        raise
