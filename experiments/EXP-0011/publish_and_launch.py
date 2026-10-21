from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import shutil
import subprocess
import time

ROOT=Path('/workspace/LeanScale');REPO=ROOT/'MiniCPM-Infer-Sys-Opt'
EXP=REPO/'experiments/EXP-0011';WT=ROOT/'worktrees/EXP-0011'
for name in ('prepare.py','trial.py','run_screen.py','publish_and_launch.py'):
    shutil.copy2(ROOT/'staging/exp11'/name,EXP/name)
source=WT/'FlagGems/src/flag_gems/runtime/backend/_iluvatar/fused/paged_gqa.py'
original=ROOT/'worktrees/EXP-0006/FlagGems/src/flag_gems/runtime/backend/_iluvatar/fused/paged_gqa.py'
t=source.read_text()
t=t.replace('decode_tile=32, prefill_tile=32, prefill_rows=64,\n              decode_warps=4, decode_stages=2):',
            'decode_tile=32, prefill_tile=32, prefill_rows=64):')
t=t.replace('*args, decode_tile, 16, True,\n        num_warps=decode_warps, num_stages=decode_stages)',
            '*args, decode_tile, 16, True, num_warps=4, num_stages=2)')
assert t==original.read_text(),'Unexpected changes beyond diagnostic launch arguments'
subprocess.run(['python3','-m','py_compile',str(EXP/'trial.py'),str(EXP/'run_screen.py')],check=True)
res=subprocess.run(['git','-C',str(WT/'vllm-plugin-FL'),'grep','-n','KVReservationAsyncScheduler'],capture_output=True,text=True)
assert res.returncode==1,'Scheduler candidate leaked into restored plugin'
(EXP/'source_equivalence_check.json').write_text(json.dumps({
    'status':'passed','check':'Reversing the two decode-parameter substitutions exactly restores frozen source bytes; scheduler class absent from plugin',
    'parent_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
    'candidate_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'scope':'CPU source/compile checks, not a numerical or performance result'},indent=2)+'\n')
old=Path(json.loads((ROOT/'artifacts/EXP-0010-latest-official.json').read_text())['run_directory'])
dest=REPO/'experiments/EXP-0010'
shutil.copy2(old/'user_stop_notice.json',dest/'user_stop_notice.json')
(dest/'USER_STOP.md').write_text('''# 用户终止调度候选

2026-10-21 15:45:19，应用户明确指示停止EXP-0010控制器和自动接续队列，服务已释放GPU。
三轮完整数据和中断第四轮现场已归档并推送313ca6853736b7c7746319f3afb6e79125a10fc4。
status.json中的failed/KeyboardInterrupt表示用户主动中断，不能解释成内核运行崩溃。
首轮按协议丢弃，第二/第三轮吞吐1833.66/1831.85 tok/s，均值1832.755，相对EXP-0006低4.112%。
两轮平均TTFT283587.52ms，相对EXP-0006增加6.050%。仅为提前终止的诊断依据，不是完整官方成绩。
不接续此调度候选4K，不重新启动。原最优融合提交保持07d9fa0c/484413ea，后续GQA实验为EXP-0011。
''')
def git(path,*args):return subprocess.check_output(['git','-C',str(path),*args],text=True,timeout=120).strip()
fg=WT/'FlagGems'
url=git(fg,'remote','get-url','--push','leanscale')
subprocess.run(['git','-C',str(fg),'push','leanscale','HEAD:refs/heads/leanscale/iluvatar-gqa-tuning'],check=True,timeout=120)
remote=subprocess.check_output(['git','-C',str(fg),'ls-remote',url,'refs/heads/leanscale/iluvatar-gqa-tuning'],text=True,timeout=60).split()[0]
assert remote==git(fg,'rev-parse','HEAD')
with (ROOT/'artifacts/docs-publish.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    paths=['experiments/EXP-0011','experiments/EXP-0010/steps.jsonl',
           'experiments/EXP-0010/user_stop_notice.json','experiments/EXP-0010/USER_STOP.md','docs/optimization_journal.md']
    git(REPO,'add','--',*paths);git(REPO,'diff','--cached','--check')
    print(git(REPO,'commit','--only','-m','experiments: stop scheduler and start GQA launch-parameter study','--',*paths),flush=True)
    git(REPO,'push','origin','main')
    url=git(REPO,'remote','get-url','--push','origin');head=git(REPO,'rev-parse','HEAD')
    assert subprocess.check_output(['git','-C',str(REPO),'ls-remote',url,'refs/heads/main'],text=True,timeout=60).split()[0]==head
now=datetime.datetime.now().astimezone();run=ROOT/'artifacts/EXP-0011'/now.strftime('%Y%m%d_%H%M%S_screen')
with (ROOT/'artifacts/EXP-0011-controller.log').open('a') as log:
    p=subprocess.Popen(['python3',str(EXP/'run_screen.py'),'--run-dir',str(run)],
        stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
for _ in range(50):
    if (run/'status.json').exists():break
    if p.poll() is not None:raise RuntimeError('Controller exited before receipt')
    time.sleep(.2)
state=json.loads((run/'status.json').read_text())
assert state['pid']==p.pid and state['stage']!='failed'
print(json.dumps({'docs_commit':head,'code_commit':remote,'run_directory':str(run),'controller':state}),flush=True)
