"""Serialized, deadline-bounded GQA sweep. Publish every outcome, including failures."""
from pathlib import Path
import argparse
import datetime
import fcntl
import hashlib
import itertools
import json
import os
import re
import shutil
import signal
import subprocess
import time
import traceback

ROOT=Path('/workspace/LeanScale');REPO=ROOT/'MiniCPM-Infer-Sys-Opt'
EXP=REPO/'experiments/EXP-0015';WT=ROOT/'worktrees/EXP-0015'
REQUEST=ROOT/'artifacts/EXP-0015-request.json'

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
            text=p.read_text(errors='replace').replace('\r\n','\n')
            text=re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b','[IP_REDACTED]',text)
            text=re.sub(r'uuid=[^,)]+','uuid=[REDACTED]',text)
            (dest/p.name).write_text(text)
    with (ROOT/'artifacts/docs-publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        paths=[str(dest.relative_to(REPO))]
        assert not git('diff','--cached','--name-only'),'Existing staged changes'
        forbidden=re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|lab_inst_[A-Za-z0-9]+|ssh\.flagos\.io')
        for item in dest.rglob('*'):
            if item.is_file():assert not forbidden.search(item.read_bytes()),str(item)
        git('add','--',*paths);git('diff','--cached','--check')
        git('commit','--only','-m','experiments: archive EXP-0015 prefill parameter screening','--',*paths)
        git('push','origin','main')
        head=git('rev-parse','HEAD');url=git('remote','get-url','--push','origin')
        remote=subprocess.check_output(['git','-C',str(REPO),'ls-remote',url,'refs/heads/main'],text=True,timeout=60).split()[0]
        assert head==remote
        write(run/'publication.json',{'status':'pushed','commit':head,'remote_verified':True})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);a=ap.parse_args()
    run=a.run_dir.resolve();assert run.parent==ROOT/'artifacts/EXP-0015';run.mkdir(parents=True,exist_ok=False)
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
        configs=[(64,4,2)]+[x for x in itertools.product((32,64,128),(2,4),(1,2)) if x!=(64,4,2)]
        for rows,warps,stages in configs:
            if remaining()<120:
                update('deferred_for_shutdown');break
            name=f'rows{rows}_w{warps}_s{stages}'
            update('screening',current_configuration=name,completed_configurations=len(state['trials']))
            command=['python3',str(EXP/'trial.py'),'--rows',str(rows),'--warps',str(warps),'--stages',str(stages),
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
            if name=='rows64_w4_s2' and record['status']!='passed':
                raise RuntimeError('Baseline-equivalent configuration failed; stop sweep for diagnosis')
        else:update('screen_complete')
        rows=[]
        for rec in state['trials']:
            if rec['status']!='passed':continue
            data=json.loads((run/(rec['configuration']+'.json')).read_text())
            changes=[v['time_change_percent'] for v in data['timings']]
            stable=all(abs(v['baseline_drift_percent'])<=3 for v in data['timings'])
            assert len(changes)==9
            primary=stable and max(changes[:6])<=3 and min(changes[:6])<=-5
            rows.append({'configuration':rec['configuration'],'time_change_percent':changes,
                         'stable_baseline':stable,'primary_gate_passed':primary,
                         'global_candidate_gate_passed':primary and max(changes[6:])<=5})
        write(run/'screen_summary.json',{'kind':state['kind'],'order':['prefill4k','prefill16k','mixed4k_decode_first','mixed4k_prefill_first','mixed16k_decode_first','mixed16k_prefill_first','decode64_4k','decode30_16k','decode1_4k'],
              'rows':rows,'selected_for_serving':False,'next_gate':'Expanded numerical/Graph/shape validation before engine integration'})
        lines=['# EXP-0015 GQA prefill 参数筛选','','单算子诊断；未改变服务版本，未测官方E2E。',
               '全部9个工作点及原始采样见各配置JSON；完整GQA总耗时为筛选依据，prefill-only单列解释。','',
               '| 配置 | prefill4K | prefill16K | mixed4K两顺序 | mixed16K两顺序 | decode三个回归点 | 基线稳定 | 全局候选门槛 |',
               '|---|---:|---:|---|---|---|---|---|']
        for row in rows:
            c=row['time_change_percent']
            fmt=lambda xs:' / '.join(f'{x:+.3f}%' for x in xs)
            lines.append(f"| {row['configuration']} | {fmt(c[:1])} | {fmt(c[1:2])} | {fmt(c[2:4])} | {fmt(c[4:6])} | {fmt(c[6:])} | {row['stable_baseline']} | {row['global_candidate_gate_passed']} |")
        lines += ['','负数表示时间减少。基线漂移≤3%；六个prefill/mixed点均不退化>3%、至少一点减少≥5%，decode回归≤5%。',
                  '失败/超时、资源超限以及全部采样保留；获选后还须扩大数值/Graph/地址检查及官方4K/16K。']
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
