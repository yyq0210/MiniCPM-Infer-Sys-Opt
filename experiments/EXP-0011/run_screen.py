"""Serialized, deadline-bounded GQA sweep. Publish every outcome, including failures."""
from pathlib import Path
import argparse
import datetime
import fcntl
import hashlib
import itertools
import json
import os
import shutil
import signal
import subprocess
import time
import traceback

ROOT=Path('/workspace/LeanScale');REPO=ROOT/'MiniCPM-Infer-Sys-Opt'
EXP=REPO/'experiments/EXP-0011';WT=ROOT/'worktrees/EXP-0011'
REQUEST=ROOT/'artifacts/EXP-0011-request.json'

def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)

def stop(p):
    if p and p.poll() is None:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()

def git(*args):
    return subprocess.check_output(['git','-C',str(REPO),*args],text=True,timeout=120).strip()

def publish(run,state):
    dest=EXP/'artifacts'/run.name;dest.mkdir(parents=True,exist_ok=True)
    for p in run.iterdir():
        if p.is_file() and p.suffix in ('.json','.log','.md','.csv'):
            shutil.copy2(p,dest/p.name)
    with (ROOT/'artifacts/docs-publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        paths=[str(dest.relative_to(REPO))]
        git('add','--',*paths);git('diff','--cached','--check')
        git('commit','--only','-m','experiments: archive EXP-0011 GQA parameter screening','--',*paths)
        git('push','origin','main')
        head=git('rev-parse','HEAD');url=git('remote','get-url','--push','origin')
        remote=subprocess.check_output(['git','-C',str(REPO),'ls-remote',url,'refs/heads/main'],text=True,timeout=60).split()[0]
        assert head==remote
        write(run/'publication.json',{'status':'pushed','commit':head,'remote_verified':True})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);a=ap.parse_args()
    run=a.run_dir.resolve();assert run.parent==ROOT/'artifacts/EXP-0011';run.mkdir(parents=True,exist_ok=False)
    policy=json.loads((ROOT/'artifacts/run_window.json').read_text())
    deadline=datetime.datetime.fromisoformat(policy['compute_stop_at'])
    def remaining():return (deadline-datetime.datetime.now().astimezone()).total_seconds()
    state={'pid':os.getpid(),'process_start_ticks':Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()[19],
      'run_directory':str(run),'started_at':datetime.datetime.now().astimezone().isoformat(),
      'compute_stop_at':deadline.isoformat(),'stage':'preparing','trials':[],
      'kind':'single_operator_diagnostic_not_official_e2e','selected_for_serving':False,
      'commits':json.loads((EXP/'candidate_commits.json').read_text())}
    def update(stage,**kw):
        state.update(stage=stage,updated_at=datetime.datetime.now().astimezone().isoformat(),**kw)
        write(run/'status.json',state);write(REQUEST,state)
    child=None;locked=False
    lock=(Path('/workspace/experiments/codex_20261019/runner.lock')).open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locked=True
        assert remaining()>600,'Insufficient time to start a fresh sweep'
        for name,sha in state['commits'].items():
            actual=subprocess.check_output(['git','-C',str(WT/name),'rev-parse','HEAD'],text=True).strip()
            assert actual==sha
            assert not subprocess.check_output(['git','-C',str(WT/name),'status','--porcelain'],text=True)
        for n in ('trial.py','run_screen.py'):
            state[n+'_sha256']=hashlib.sha256((EXP/n).read_bytes()).hexdigest()
        env=dict(os.environ,PYTHONPATH=f'{WT}/vllm-plugin-FL:{WT}/FlagGems/src',
                 PYTHONUNBUFFERED='1',VLLM_PLUGINS='fl',TRITON_CACHE_DIR=str(run/'triton_cache'))
        configs=[(4,2,16)]+[x for x in itertools.product((2,4),(1,2,3),(8,16)) if x!=(4,2,16)]
        for w,t,s in configs:
            if remaining()<120:
                update('deferred_for_shutdown');break
            name=f'w{w}_t{t}_s{s}'
            update('screening',current_configuration=name,completed_configurations=len(state['trials']))
            command=['python3',str(EXP/'trial.py'),'--warps',str(w),'--stages',str(t),'--segments',str(s),
                     '--output',str(run/(name+'.json'))]
            write(run/(name+'.command.json'),command)
            start=time.monotonic()
            with (run/(name+'.log')).open('w') as log:
                child=subprocess.Popen(command,env=env,cwd=run,stdin=subprocess.DEVNULL,
                       stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                update('screening',child_pid=child.pid)
                try:code=child.wait(timeout=min(600,remaining()))
                except subprocess.TimeoutExpired:
                    stop(child);code='timeout'
                finally:stop(child)
            record={'configuration':name,'returncode':code,'elapsed_seconds':time.monotonic()-start}
            p=run/(name+'.json')
            if p.exists():
                data=json.loads(p.read_text());record['status']=data['status'];record['checks']=len(data.get('checks',[]))
            else:record['status']='failed_before_result'
            if code!=0:record['status']='failed'
            state['trials'].append(record);update('screening')
            if name=='w4_t2_s16' and record['status']!='passed':
                raise RuntimeError('Baseline-equivalent configuration failed; stop sweep for diagnosis')
        else:update('screen_complete')
        rows=[]
        for rec in state['trials']:
            if rec['status']!='passed':continue
            data=json.loads((run/(rec['configuration']+'.json')).read_text())
            changes=[v['time_change_percent'] for v in data['timings']]
            stable=all(abs(v['baseline_drift_percent'])<=3 for v in data['timings'])
            primary=stable and max(changes[:2])<=3 and min(changes[:2])<=-3
            rows.append({'configuration':rec['configuration'],'time_change_percent':changes,
                         'stable_baseline':stable,'primary_gate_passed':primary,
                         'global_candidate_gate_passed':primary and changes[2]<=5})
        write(run/'screen_summary.json',{'kind':state['kind'],'order':['B64/L4700','B30/L16384','B1/L4096'],
              'rows':rows,'selected_for_serving':False,'next_gate':'Expanded numerical/Graph/shape validation before engine integration'})
        lines=['# EXP-0011 GQA 参数筛选','','单算子诊断；未选择服务版本，未测官方E2E。',
               '| 配置 | B64/4700 时间变化 | B30/16384 时间变化 | B1/4096 时间变化 | 基线稳定 | 全局候选门槛 |',
               '|---|---:|---:|---:|---|---|']
        for r in rows:
            c=r['time_change_percent'];lines.append(f"| {r['configuration']} | {c[0]:+.3f}% | {c[1]:+.3f}% | {c[2]:+.3f}% | {r['stable_baseline']} | {r['global_candidate_gate_passed']} |")
        lines += ['','负数表示时间减少。门槛仅用于筛选，不能替代扩大验证或整模测评。','',
                  '失败/超时配置及全部原始采样见status.json与各配置JSON/log。']
        (run/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    except BaseException as e:
        update('failed',error=repr(e));traceback.print_exc()
    finally:
        stop(child)
        update(state['stage'],owned_children_reaped=True,safe_to_resume_gpu=locked)
        lock.close()
        try:publish(run,state)
        except BaseException as e:
            write(run/'publication.json',{'status':'local_saved_push_failed','error':repr(e)});traceback.print_exc()

if __name__=='__main__':main()
