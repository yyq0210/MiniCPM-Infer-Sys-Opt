"""Publish a frozen measurement protocol, then launch one official 4K case."""
from pathlib import Path
from datetime import datetime, timedelta
import ast, fcntl, hashlib, json, shutil, subprocess, time

ROOT=Path('/workspace/LeanScale');DOC=ROOT/'MiniCPM-Infer-Sys-Opt'
EXP=DOC/'experiments/EXP-0024';ART=ROOT/'artifacts/EXP-0024';SRC=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*a):return subprocess.check_output(['git','-C',str(DOC),*a],text=True,stderr=subprocess.STDOUT,timeout=120).strip()
def step(action,**kw):
    with (EXP/'steps.jsonl').open('a') as f:f.write(json.dumps(dict(time=datetime.now().astimezone().isoformat(),action=action,**kw))+'\n')
def main():
    assert not (EXP/'e2e_plan.json').exists() and not (ART/'official_4k_request.json').exists()
    validation=ART/'validation_r3'
    assert read(validation/'status.json')['status']=='passed'
    assert read(validation/'status.json')['owned_children_reaped']
    assert read(validation/'publication.json')['remote_verified']
    publication=read(validation/'code_publication.json');assert publication['remote_verified']
    commits=read(EXP/'candidate_commits.json');assert commits==publication['commits']
    now=datetime.now().astimezone();assert now.strftime('%Y-%m-%d')=='2026-10-28'
    daily=now.replace(hour=23,minute=35,second=0,microsecond=0)
    assert (daily-now).total_seconds()>75*60,'Insufficient budget; keep validated code saved'
    protocol='''\n\n## 官方4K预注册执行协议\n\n接入验证通过后冻结候选提交。保持官方 [4096,1024,64,256]、random/default sampling/ignore EOS、四轮弃首、FULL_DECODE_ONLY、gpu_memory_utilization=.85、max_model_len=131072。对照已审核EXP20历史四轮；不将会话间差异称为稳定因果收益。\n服务先过接线门：42层gate/up与42层down各一次重排、逐元素值一致、原Parameter保留；编译模块84个MLP matmul直接读取模型参数，gate/up转置视图[2048,12288] stride[12288,1]、down转置视图[6144,2048] stride[2048,1]，无forward权重重排缓冲；原42GQA/42RoPEKV与35 FULL capture尺寸；原调度；新KV token容量至少为历史值99%。容量差异完整记录，1%为启动检查阈值，不是允许隐藏容量下降。\n只有全部接线检查通过才开始四轮。任意失败保留独立run，不改运行源、不合并未完成轮次、不复跑挑最好。每阶段刷新窗口，最多110分钟并受23:35总截止限制，结果先本地保存再push，保留所有原始逐轮数据。4K完成后独立审核，再决定配对或16K；本次自动流程不切换selected基线。\n完整模型任务精度按用户暂缓；operator/Graph校验通过不等于赛事精度达标。\n'''
    with (EXP/'README.md').open('a') as f:f.write(protocol)
    for name in ['run_e2e.py','prepare_e2e.py']:
        shutil.copy2(SRC/name,EXP/name);ast.parse((EXP/name).read_text())
    plan=dict(experiment='EXP-0024',candidate_commits=commits,selected_baseline='EXP-0020',
        script_hashes={name:sha(EXP/name) for name in ['run_e2e.py','prepare_e2e.py']},
        validation_attempt='r3',validation_status_sha256=sha(validation/'status.json'),
        protocol='official4K four rounds discard first; historical EXP20 comparator; actual route gates before timing',
        model_accuracy='deferred_by_user',kv_capacity_startup_min_ratio=.99)
    write(EXP/'e2e_plan.json',plan)
    step('Freeze validated source and preregister actual-route gates before official4K',candidate_commits=commits)
    with (ROOT/'artifacts/docs-publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);assert not git('diff','--cached','--name-only')
        paths=['experiments/EXP-0024/README.md','experiments/EXP-0024/run_e2e.py','experiments/EXP-0024/prepare_e2e.py','experiments/EXP-0024/e2e_plan.json','experiments/EXP-0024/steps.jsonl']
        git('add','--',*paths);git('diff','--cached','--check')
        git('commit','--only','-m','experiments: preregister official MLP layout route and performance gates','--',*paths)
        head=git('rev-parse','HEAD');git('push','origin','main');assert git('ls-remote',git('remote','get-url','--push','origin'),'refs/heads/main').split()[0]==head
    write(ART/'e2e_preparation.json',dict(commit=head,remote_verified=True))
    now=datetime.now().astimezone();assert now.strftime('%Y-%m-%d')=='2026-10-28'
    end=min(now+timedelta(minutes=110),daily);assert (end-now).total_seconds()>70*60
    window=dict(created_at=now.isoformat(),compute_stop_at=end.isoformat(),
                archive_target_at=min(end+timedelta(minutes=10),daily.replace(minute=50)).isoformat(),
                daily_compute_stop_at=daily.isoformat(),daily_save_by=daily.replace(minute=50).isoformat())
    write(ART/'e2e_window_4k.json',window)
    run=ART/(now.strftime('%Y%m%d_%H%M%S')+'_official_4k')
    with (ART/'official_4k.controller.log').open('w') as log:
        p=subprocess.Popen(['python3',str(EXP/'run_e2e.py'),'--case','4k','--run-dir',str(run)],
                           stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    for _ in range(100):
        if (ART/'official_4k_request.json').exists():break
        assert p.poll() is None;time.sleep(.2)
    else:raise RuntimeError('No official run receipt')
    receipt=read(ART/'official_4k_request.json');assert receipt['pid']==p.pid
    write(ART/'e2e_launch.json',dict(preparation_commit=head,receipt=receipt,window=window))
    print(json.dumps(dict(preparation=head,receipt=receipt,window=window)),flush=True)
if __name__=='__main__':main()
