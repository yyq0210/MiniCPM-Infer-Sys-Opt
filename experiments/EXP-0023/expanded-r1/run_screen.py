"""Bounded native weight-layout screen; preserve selected inference sources."""
from pathlib import Path
from datetime import datetime
import fcntl
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import time
import traceback

ROOT=Path('/workspace/LeanScale');REPO=ROOT/'MiniCPM-Infer-Sys-Opt';EXP=REPO/'experiments/EXP-0023/expanded-r1';ART=ROOT/'artifacts/EXP-0023/expanded-r1'
def read(p):return json.loads(p.read_text())
def write(p,x):
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(p,*args):return subprocess.check_output(['git','-C',str(p),*args],text=True,stderr=subprocess.STDOUT,timeout=120).strip()
def verify():
    plan=read(EXP/'plan.json')
    assert read(ROOT/'artifacts/active_inference_plan.json')['selected_serving_commits']==plan['selected_commits']
    for n,h in plan['script_hashes'].items():assert sha(EXP/n)==h
    for p,h in plan['source_hashes'].items():assert sha(Path(p))==h
    for n,c in plan['selected_commits'].items():
        p=ROOT/'worktrees/EXP-0020'/n;assert git(p,'rev-parse','HEAD')==c and not git(p,'status','--porcelain')

def main():
    latest=ROOT/'artifacts/EXP-0023-expanded-r1-request.json';assert not latest.exists()
    now=datetime.now().astimezone();run=ART/now.strftime('%Y%m%d_%H%M%S_expanded');run.mkdir()
    deadline=datetime.fromisoformat(read(ART/'run_window.json')['compute_stop_at'])
    state=dict(pid=os.getpid(),process_start_ticks=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()[19],
      run_directory=str(run),started_at=now.isoformat(),compute_stop_at=deadline.isoformat(),selected_for_serving=False)
    def update(stage,**kw):
        state.update(stage=stage,updated_at=datetime.now().astimezone().isoformat(),**kw);write(run/'status.json',state);write(latest,state)
    lock=Path('/workspace/experiments/codex_20261019/runner.lock').open('a');child=None;locked=False
    update('preparing')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locked=True
        assert deadline.date()==now.date() and (deadline-datetime.now().astimezone()).total_seconds()>1800
        for port in (9031,9032):
            with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',port))!=0
        idle=[]
        for _ in range(2):
            text=subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True,timeout=20)
            m=re.search(r'(\d+)MiB\s*/\s*(\d+)MiB\s*\|\s*(\d+)%',text)
            assert m and int(m[1])<=1024 and int(m[3])<=5
            idle.append(dict(memory_mib=int(m[1]),util_percent=int(m[3])));time.sleep(3)
        write(run/'idle_evidence.json',idle);verify()
        env=dict(os.environ,PYTHONPATH=str(ROOT/'worktrees/EXP-0020/vllm-plugin-FL')+':'+str(ROOT/'worktrees/EXP-0020/FlagGems/src'),
                 VLLM_PLUGINS='fl',PYTHONUNBUFFERED='1',TRITON_CACHE_DIR=str(run/'triton_cache'),VLLM_CACHE_ROOT=str(run/'vllm_cache'),TORCHINDUCTOR_CACHE_DIR=str(run/'inductor_cache'))
        with (run/'trial.log').open('w') as log:
            child=subprocess.Popen(['python3',str(EXP/'trial.py'),'--output',str(run/'result.json')],env=env,cwd=run,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            update('screening',child_pid=child.pid)
            code=child.wait(timeout=min(1800,(deadline-datetime.now().astimezone()).total_seconds()))
        assert code==0,code;verify();d=read(run/'result.json')
        assert d['status']=='passed' and (len(d['checks']),len(d['timings']))==(117,180)
        candidates=[];scores={}
        for variant in d['variants']:
            rows=[x for x in d['timings'] if x['variant']==variant]
            primary=[x for x in rows if x['M'] in (8,16)]
            other=[x for x in rows if x['M'] not in (8,16)]
            if len(rows)==60 and len(primary)==12 and len(other)==48 and all(x['time_change_percent']<=-5 for x in primary) and all(x['time_change_percent']<=3 for x in other) and all(abs(x['baseline_drift_percent'])<=3 for x in rows):
                candidates.append(variant);scores[variant]=sum(x['time_change_percent'] for x in primary)/len(primary)
        selected=None
        if candidates:
            best=min(scores.values());near=[v for v in candidates if scores[v]-best<1]
            selected=min(near,key=lambda v:(2 if v=='both' else 1,scores[v]))
        write(run/'screen_summary.json',dict(status='complete',candidates=candidates,primary_mean_changes=scores,selected_for_refinement=selected,selection_rule='all3seeds: M8/16 <=-5percent in both modes; others<=+3percent; all drift<=3percent; within1percentagepoint prefer fewer repacked weights; independent audit required'))
        lines=['# EXP23：独立完整MLP布局复测','','117项数值/Graph检查、180个完整gate/up→生产生成SwiGLU→down计时点。',
          '三个独立种子，随机权重与激活；hot与8权重轮换是缓存敏感性代理，不是模型请求重放。沿用生产内核源，在独立缓存重新编译/调优。','',
          '| 种子 | M | 候选 | 模式 | 完整MLP耗时变化 | 夹测漂移 |','|---:|---:|---|---|---:|---:|']
        lines += [f"| {x['seed']} | {x['M']} | {x['variant']} | {x['mode']} | {x['time_change_percent']:+.3f}% | {x['baseline_drift_percent']:+.3f}% |" for x in d['timings']]
        lines += ['', '候选：'+json.dumps(candidates,ensure_ascii=False),'独立审核后待集成候选：'+str(selected),'一次性packing不计入逐次MLP，所有退化点保留；没有服务切换。']
        (run/'RESULTS.md').write_text('\n'.join(lines)+'\n');update('expanded_complete',checks=117,timing_points=180,candidates=candidates,selected_for_refinement=selected)
    except BaseException:
        update('failed',error=traceback.format_exc());traceback.print_exc()
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        update(state['stage'],owned_children_reaped=True,safe_to_resume_gpu=locked);lock.close()
        try:
            with (ROOT/'artifacts/docs-publish.lock').open('a') as pub:
                fcntl.flock(pub,fcntl.LOCK_EX);dest=EXP/'artifacts'/run.name;dest.mkdir(parents=True)
                for p in run.iterdir():
                    if p.is_file() and p.suffix in ('.json','.md','.log'):
                        text=p.read_text(errors='replace').replace('\r\n','\n');text=re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b','[IP_REDACTED]',text)
                        text=re.sub(r'uuid=[^,)]+','uuid=[REDACTED]',text);(dest/p.name).write_text(text)
                path=str(dest.relative_to(REPO));assert not git(REPO,'diff','--cached','--name-only')
                forbidden=re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|lab_inst_[A-Za-z0-9]+|ssh\.flagos\.io')
                for p in dest.iterdir():assert not forbidden.search(p.read_bytes()),str(p)
                git(REPO,'add','--',path);git(REPO,'diff','--cached','--check');git(REPO,'commit','--only','-m','experiments: archive complete MLP weight layout refinement','--',path)
                head=git(REPO,'rev-parse','HEAD');git(REPO,'push','origin','main')
                assert git(REPO,'ls-remote',git(REPO,'remote','get-url','--push','origin'),'refs/heads/main').split()[0]==head
            write(run/'publication.json',dict(status='pushed',commit=head,remote_verified=True))
        except BaseException:write(run/'publication.json',dict(status='local_saved_push_failed',error=traceback.format_exc()))

if __name__=='__main__':main()
