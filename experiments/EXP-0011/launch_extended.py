from pathlib import Path
import datetime
import fcntl
import json
import shutil
import subprocess
import time

ROOT=Path('/workspace/LeanScale');REPO=ROOT/'MiniCPM-Infer-Sys-Opt';EXP=REPO/'experiments/EXP-0011'
s=json.loads((ROOT/'artifacts/EXP-0011-request.json').read_text());run=Path(s['run_directory'])
assert s['stage']=='screen_complete' and s['safe_to_resume_gpu']
assert json.loads((run/'publication.json').read_text())['status']=='pushed'
latest=ROOT/'artifacts/EXP-0011-expanded-request.json'
assert not latest.exists(),'Do not duplicate an expanded run'
data=json.loads((run/'w4_t1_s16.json').read_text());assert data['status']=='passed'
for n in ('validate_extended.py','run_extended.py','launch_extended.py'):
    shutil.copy2(ROOT/'staging/exp11'/n,EXP/n)
lines=['# GQA 单级流水候选：首轮筛选与验证计划','','2026-10-21，EXP-0011。',
       '用户要求中止劣化调度后，恢复EXP-0006融合基线。本轮12种配置全部完成，72项数值/Graph检查通过。',
       '唯一通过预设全局候选门槛的是4 warps、1 stage、16 segments；原配置为4/2/16。',
       '只改变decode流水级数，保持Q16/KV2/D128、tile32、BM16、分段softmax定义、prefill与RoPE融合不变。','',
       '| 工作点 | 原版夹测均值 μs | 候选中位数 μs | 完整GQA调用时间变化 |',
       '|---|---:|---:|---:|']
for x in data['timings']:
    old=(x['baseline_before']['median_us']+x['baseline_after']['median_us'])/2
    lines.append(f"| B{x['B']}/L{x['L']} | {old:.3f} | {x['candidate_total']['median_us']:.3f} | {x['time_change_percent']:+.3f}% |")
lines += ['','以上是Graph event单算子微基准（主kernel＋reduce），不是完整模型收益。',
 '原参数对照与自身差异约-0.10%～+0.46%；各配置夹测漂移均低于3%。',
 '编译资源字段显示主kernel寄存器由111降至96，shared仍20480字节，n_spills均为0。'
 '这支持进一步检查寄存器压力，但没有硬件计数器证据证明动态occupancy的具体变化。',
 '8分段在部分大batch稍快，但小batch明显退化；2-warps并未胜过当前4-warps/1-stage。不能只取最好单点。','',
 '下一步：同一冻结候选扩大到26项数值/Graph检查和34个计时工作点，包含FP16/BF16、NHD/HND、'
 '大于2^31元素的真实KV地址、query/length/block table动态内容、prefill2048及mixed。'
 '通过后才修改默认参数或显式接入引擎，并跑官方四轮。当前服务选择仍为原最优融合版本。','',
 '[全部12组数据](../experiments/EXP-0011/artifacts/'+run.name+'/RESULTS.md)。']
(REPO/'docs/gqa_stage1_screen_20261021.md').write_text('\n'.join(lines)+'\n')
(EXP/'PR_DRAFT_FlagGems.md').write_text('''# Draft：赛事PR文案

标题：【FlagOS开放计算全球挑战赛S2.赛题2】【LeanScale】优化天数GQA decode流水参数

当前阶段为诊断原型，未创建上游PR，未改变默认服务参数。
在Q16/KV2/D128 decode主kernel暴露线程/流水参数，保留prefill和fallback路径。12组参数扫描发现4-warps/1-stage/16-segments可降低三个代表工作点的单算子时间约13.8%～20.8%。
此前实现固定4-warps/2-stage；候选寄存器资源字段从111降为96，shared不变。未测动态occupancy，不能据此宣称确切硬件瓶颈。
已有72项数值/Graph检查通过，扩大验证和整模端到端结果待补。完整任务精度暂缓；不能声称满足最终赛事精度门槛。
具体协议、全部配置、失败记录和原始测量见EXP-0011。最终PR须按目标仓库模板整理并重写为最终实现。
''')
with (EXP/'steps.jsonl').open('a') as f:f.write(json.dumps({'time':datetime.datetime.now().astimezone().isoformat(),
    'action':'screen complete: all 12 configurations passed; select w4_t1_s16 for expanded validation only; no serving change',
    'source_screen':run.name,'expanded_checks':26,'timing_points':34})+'\n')
with (ROOT/'artifacts/docs-publish.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    paths=['experiments/EXP-0011/validate_extended.py','experiments/EXP-0011/run_extended.py',
      'experiments/EXP-0011/launch_extended.py','experiments/EXP-0011/PR_DRAFT_FlagGems.md',
      'experiments/EXP-0011/steps.jsonl','docs/gqa_stage1_screen_20261021.md']
    def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a],text=True,timeout=120).strip()
    git('add','--',*paths);git('diff','--cached','--check')
    print(git('commit','--only','-m','experiments: select GQA stage-one candidate for expanded validation','--',*paths),flush=True)
    git('push','origin','main');head=git('rev-parse','HEAD')
    assert git('ls-remote',git('remote','get-url','--push','origin'),'refs/heads/main').split()[0]==head
with (ROOT/'artifacts/EXP-0011-extended-controller.log').open('a') as log:
    p=subprocess.Popen(['python3',str(EXP/'run_extended.py')],stdin=subprocess.DEVNULL,
       stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(50):
    if latest.exists():break
    if p.poll() is not None:raise RuntimeError('Expanded controller failed before receipt')
    time.sleep(.2)
receipt=json.loads(latest.read_text());assert receipt['pid']==p.pid and receipt['stage']=='validating'
print(json.dumps({'docs_commit':head,'receipt':receipt}),flush=True)
