"""Preserve failed metadata collection and retry the unchanged MLP protocol."""
from pathlib import Path
from datetime import datetime,timedelta
import ast,fcntl,hashlib,json,subprocess,time,traceback
ROOT=Path('/workspace/LeanScale');REPO=ROOT/'MiniCPM-Infer-Sys-Opt';OLD=REPO/'experiments/EXP-0023/expanded';EXP=REPO/'experiments/EXP-0023/expanded-r1';ART=ROOT/'artifacts/EXP-0023/expanded-r1'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a],text=True,stderr=subprocess.STDOUT,timeout=120).strip()
def step(p,action,**kw):
    with (p/'steps.jsonl').open('a') as f:f.write(json.dumps(dict(time=datetime.now().astimezone().isoformat(),action=action,**kw),ensure_ascii=False)+'\n')
def main():
    assert not EXP.exists() and not ART.exists() and not (ROOT/'artifacts/EXP-0023-expanded-r1-request.json').exists()
    state=read(ROOT/'artifacts/EXP-0023-expanded-request.json');run=Path(state['run_directory']);result=read(run/'result.json')
    assert state['stage']=='failed' and state['owned_children_reaped'] and read(run/'publication.json')['remote_verified']
    assert result['error']=='''TypeError("'NoneType' object is not iterable")''' and result['checks']==[] and result['timings']==[]
    prior=read(OLD/'plan.json')
    for n,h in prior['script_hashes'].items():assert sha(OLD/n)==h
    for p,h in prior['source_hashes'].items():assert sha(Path(p))==h
    assert read(ROOT/'artifacts/active_inference_plan.json')['selected_serving_commits']==prior['selected_commits']
    now=datetime.now().astimezone();assert now.strftime('%Y-%m-%d')=='2026-10-28'
    EXP.mkdir();ART.mkdir()
    note='''\n\n## 第一次重启：仅修复元数据为空值的兼容问题\n\n原expanded于2026-10-28 16:15:25失败并完整归档，0数值检查/0计时点；生产激活核预热后的kernel.configs为None，记录元数据时迭代该字段触发TypeError。失败发生在任何候选比较前，不能判定布局效果。\n本次单独目录expanded-r1，仅将元数据列表读取改为None安全，并记录configs_available。候选、内核、shape、种子、精度门槛、计时、筛选逻辑全部不变，不改原失败源码/manifest。使用新窗口、新缓存和request。\n'''
    (EXP/'README.md').write_text((OLD/'README.md').read_text()+note)
    step(EXP,'Record compatibility-only retry rationale before modifying copied code',failed_run=str(run),failed_result_sha256=sha(run/'result.json'))
    old=(OLD/'trial.py').read_text();needle="configs=[str(v) for v in getattr(kernel,'configs',[])]"
    replacement="configs_available=getattr(kernel,'configs',None) is not None,configs=[str(v) for v in (getattr(kernel,'configs',None) or [])]"
    assert old.count(needle)==1
    new=old.replace(needle,replacement);assert new.replace(replacement,needle)==old
    (EXP/'trial.py').write_text(new)
    controller=(OLD/'run_screen.py').read_text().replace("/EXP-0023/expanded'","/EXP-0023/expanded-r1'").replace('EXP-0023-expanded-request.json','EXP-0023-expanded-r1-request.json')
    assert "EXP=REPO/'experiments/EXP-0023/expanded-r1'" in controller and "ART=ROOT/'artifacts/EXP-0023/expanded-r1'" in controller
    (EXP/'run_screen.py').write_text(controller);(EXP/'prepare_retry1.py').write_text(Path(__file__).read_text())
    for n in ['production_silu_source.json','route_evidence.json','daily_window.json']:(EXP/n).write_bytes((OLD/n).read_bytes())
    for p in EXP.glob('*.py'):ast.parse(p.read_text(),filename=p.name)
    sources=dict(prior['source_hashes']);sources[str(OLD/'trial.py')]=sha(OLD/'trial.py')
    for n in ['README.md','production_silu_source.json','route_evidence.json','daily_window.json']:sources[str(EXP/n)]=sha(EXP/n)
    plan=dict(prior,source_hashes=sources,script_hashes={p.name:sha(p) for p in EXP.glob('*.py')},retry_reason='metadata-only None-safe configs collection',parent_trial_sha256=sha(OLD/'trial.py'),parent_failed_result_sha256=sha(run/'result.json'))
    write(EXP/'plan.json',plan);write(EXP/'STATUS.json',dict(stage='prepared',GPU_started=False))
    write(OLD/'STATUS.json',dict(stage='failed_before_measurement',checks=0,timings=0,reason='Optional compiler configs metadata was None',publication=read(run/'publication.json'),next='isolated expanded-r1 compatibility fix'))
    step(OLD,'Preserve failure before all candidate checks/timings; start separate compatibility-only retry')
    with (REPO/'docs/optimization_journal.md').open('a') as f:f.write('\n\n### 2026-10-28：EXP23完整MLP元数据兼容修复\n\n首轮expanded因预热后kernel.configs=None在记录阶段失败，0检查/0计时，自动归档353d3feb4c5070f735f77538f17d08610c32e5cf。创建expanded-r1，只做None安全元数据读取；精度/计时/种子/候选/门槛不变，原源码和失败保留，新窗口重启。\n')
    with (ROOT/'artifacts/docs-publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);assert not git('diff','--cached','--name-only')
        paths=['experiments/EXP-0023/expanded/STATUS.json','experiments/EXP-0023/expanded/steps.jsonl','experiments/EXP-0023/expanded-r1','docs/optimization_journal.md']
        git('add','--',*paths);git('diff','--cached','--check');git('commit','--only','-m','experiments: preserve MLP preflight failure and fix optional metadata','--',*paths)
        head=git('rev-parse','HEAD');git('push','origin','main');assert git('ls-remote',git('remote','get-url','--push','origin'),'refs/heads/main').split()[0]==head
    write(ART/'preparation.json',dict(commit=head,remote_verified=True,compatibility_fix_only=True))
    now=datetime.now().astimezone();assert now.strftime('%Y-%m-%d')=='2026-10-28';daily=read(EXP/'daily_window.json')
    end=min(now+timedelta(minutes=40),datetime.fromisoformat(daily['compute_stop_at']));assert (end-now).total_seconds()>35*60
    policy=dict(experiment='EXP-0023-expanded-r1',created_at=now.isoformat(),compute_stop_at=end.isoformat(),archive_target_at=(end+timedelta(minutes=10)).isoformat(),daily_window=daily)
    write(ART/'run_window.json',policy);step(EXP,'Launch new independent bounded run after preserving failure and publishing compatibility fix',policy=policy)
    with (ART/'controller.log').open('w') as log:
        child=subprocess.Popen(['python3',str(EXP/'run_screen.py')],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    latest=ROOT/'artifacts/EXP-0023-expanded-r1-request.json'
    for _ in range(100):
        if latest.exists():break
        assert child.poll() is None;time.sleep(.2)
    else:raise RuntimeError('No retry receipt')
    receipt=read(latest);assert receipt['pid']==child.pid
    write(ART/'launch.json',dict(preparation_commit=head,receipt=receipt,policy=policy));print(json.dumps(dict(commit=head,receipt=receipt,policy=policy)),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        if EXP.exists():step(EXP,'Retry preparation/start failed; preserve and inspect before recovery',error=traceback.format_exc())
        write(ROOT/'artifacts/EXP-0023-expanded-r1-preparation-error.json',dict(error=traceback.format_exc()));raise
