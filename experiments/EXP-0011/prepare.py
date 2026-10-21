"""Create a new GQA workspace from the proven fusion commits, without rewinding history."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import subprocess

ROOT = Path('/workspace/LeanScale')
REPO = ROOT/'MiniCPM-Infer-Sys-Opt'
EXP = REPO/'experiments/EXP-0011'
WT = ROOT/'worktrees/EXP-0011'
PARENT = {'vllm-plugin-FL':'07d9fa0cfdc5f09db178e29508e3c0bcb5c1e908',
          'FlagGems':'484413ea0db18221dd9e63e742361d837df9ff13'}

def git(path, *args):
    return subprocess.check_output(['git','-C',str(path),*args],text=True).strip()

assert not EXP.exists() and not WT.exists(), 'Do not overwrite an existing experiment'
old = json.loads((ROOT/'artifacts/EXP-0010-latest-official.json').read_text())
run = Path(old['run_directory'])
assert (run/'user_stop_notice.json').exists()
assert json.loads((run/'completion_handler_status.json').read_text())['status']=='pushed'
EXP.mkdir(parents=True)
(EXP/'README.md').write_text('''# EXP-0011：GQA decode 参数优化

用户于2026-10-21要求停止劣化调度方案，恢复原最优融合方案，转入GQA优化。
本实验直接基于EXP-0006的插件07d9fa0c与FlagGems484413ea，不包含EXP-0010的KV准入调度器。
官方原始基线保持不动，失败及中断实验保留。

## 假设与路径（修改代码前记录）

实际入口为IluvatarGQAImpl.forward → FlagGems paged_gqa → _attention(decode) → _reduce。
Q16/KV2/D128，8个Q heads共享KV，decode BM16只有8行有效。
已有诊断显示主kernel占NONE/late可见GPU kernel时间78.34%，但该占比不代表生产Graph延迟。
先固定tile32与prefill64行，独立扫描decode warps={2,4}、stages={1,2,3}、segments={8,16}。
不改KV内容、采样、官方负载或服务参数；不同时引入NSEQ/分页布局修改。

## 筛选与验证

首轮为单算子诊断，不是官方端到端成绩。12种配置分别用独立进程，失败配置完整保留。
数值检查包括BF16/NHD、FP16/HND、非连续Q、ragged/空query、mixed，独立FP32 causal参考。
Graph重放更改Q、K/V、length和block table内容，同时保持地址稳定，核对eager与Graph。
计时工作点为B64/L4700、B30/L16384、B1/L4096，包含主kernel及reduce。
每种配置以原最优版本前后夹测，7次Graph event采样；另列主kernel、reduce计时，不混同总调用计时。
预设筛选：主要两个工作点均不退化超过3%，至少一个改善3%；小batch若退化超过5%则只能考虑明确的shape分支，不能全局替换。
之后才做扩大工作点、FP16/BF16全布局、int64大偏移、prefill2048/mixed和完整Graph检查；通过后接入42层真实模型，再跑官方四轮。
完整模型精度按用户要求暂缓；KV压缩不在本实验范围。

## 截止与保存

遵守artifacts/run_window.json的23:35计算截止。控制器独占原GPU锁，子进程超时不超过截止时间。
每个trial结束即原子保存；控制器结束后发布紧凑结果和日志，不上传缓存、模型或数据集。
''')
(EXP/'parent_commits.json').write_text(json.dumps(PARENT,indent=2)+'\n')
with (EXP/'steps.jsonl').open('a') as f:
    f.write(json.dumps({'time':datetime.datetime.now().astimezone().isoformat(),
        'action':'hypothesis recorded before source change','parent_commits':PARENT})+'\n')
WT.mkdir(parents=True)
for name, sha in PARENT.items():
    source = ROOT/'worktrees/EXP-0006'/name
    assert git(source,'rev-parse','HEAD') == sha and not git(source,'status','--porcelain')
    args = ['worktree','add']
    args += ['-b','leanscale/iluvatar-gqa-tuning'] if name=='FlagGems' else ['--detach']
    print(git(source,*args,str(WT/name),sha),flush=True)
p = WT/'FlagGems/src/flag_gems/runtime/backend/_iluvatar/fused/paged_gqa.py'
t=p.read_text()
old='decode_tile=32, prefill_tile=32, prefill_rows=64):'
assert t.count(old)==1
t=t.replace(old,'decode_tile=32, prefill_tile=32, prefill_rows=64,\n              decode_warps=4, decode_stages=2):')
old='*args, decode_tile, 16, True, num_warps=4, num_stages=2)'
assert t.count(old)==1
t=t.replace(old,'*args, decode_tile, 16, True,\n        num_warps=decode_warps, num_stages=decode_stages)')
p.write_text(t)
assert 'kv_reservation_scheduler' not in (WT/'vllm-plugin-FL/vllm_fl/platform.py').read_text() if (WT/'vllm-plugin-FL/vllm_fl/platform.py').exists() else True
git(WT/'FlagGems','add','src/flag_gems/runtime/backend/_iluvatar/fused/paged_gqa.py')
git(WT/'FlagGems','diff','--cached','--check')
print(git(WT/'FlagGems','commit','-m','feat(iluvatar): expose decode launch parameters for GQA tuning'),flush=True)
commits={n:git(WT/n,'rev-parse','HEAD') for n in PARENT}
(EXP/'candidate_commits.json').write_text(json.dumps(commits,indent=2)+'\n')
for n in ('trial.py','run_screen.py'):
    shutil.copy2(ROOT/'staging/exp11'/n,EXP/n)
(ROOT/'artifacts/active_inference_plan.json').write_text(json.dumps({
    'selected_serving_commits':PARENT,'scheduler_candidate':'rejected_by_user',
    'current_experiment':'EXP-0011','candidate_commits':commits,
    'status':'diagnostic_candidate_not_selected_for_serving'},indent=2)+'\n')
with (REPO/'docs/optimization_journal.md').open('a') as f:
    f.write('\n\n### 2026-10-21：按用户指示终止调度候选，恢复融合版本并转入GQA\n\n'
      '15:45停止EXP-0010控制器和接续队列，GPU释放，三轮完整数据保留、第四轮中断。'
      '两个可保留样本均值1832.755 tok/s，比EXP-0006低4.112%，TTFT增加6.050%；这不是完整四轮成绩。'
      '中断归档313ca685已推送。禁止接续调度4K或重跑该候选。'
      'EXP-0011以原最优融合提交建立独立工作树，先记录假设，再增加仅供诊断的decode launch参数；'
      '默认4 warps/2 stages保持原行为，获胜前不改变服务配置。\n')
print(json.dumps({'commits':commits,'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}),flush=True)
