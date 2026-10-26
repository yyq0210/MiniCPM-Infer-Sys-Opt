"""Four fresh official 4K service sessions in predetermined ABBA order."""
import datetime
import fcntl
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import time
import traceback
from common import *
import frozen_runtime as rt
from route_checks import verify_route

STATE={}
def state(stage,**kw):
    STATE.update(stage=stage,updated_at=now().isoformat(),**kw)
    write(ART/'status.json',STATE)
    print(json.dumps(STATE),flush=True)
def safe_text(s):
    s=re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b','[IP_REDACTED]',s)
    s=re.sub(r'uuid=[^,)]+','uuid=[REDACTED]',s)
    return s.replace('\r\n','\n')
def archive(run):
    dest=EXP/'artifacts'/run.name;hashes={}
    files=[run/x for x in ['status.json','environment_check.json','result.json','error.json','stage_history.jsonl','idle_evidence.json']]
    folder=run/'candidate_benchmark'
    files += [folder/x for x in ['server.log','server.command.json','benchmark.log','benchmark.command.json','smoke.json','route_evidence.json','startup_route_evidence.json','metrics_before.prom','metrics_after.prom']]
    files += list((folder/'benchmark_results').glob('*.csv'))
    for src in files:
        if src.is_file():
            rel=src.relative_to(run);target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(safe_text(src.read_text(errors='replace')))
            hashes[str(rel)]=sha(src)
    write(dest/'source_hashes.json',hashes)
    write(EXP/'STATUS.json',STATE)
    try:
        publication=publish(['experiments/EXP-0021/artifacts/'+run.name,'experiments/EXP-0021/STATUS.json','experiments/EXP-0021/steps.jsonl'],
            'experiments: archive ABBA 4K session '+run.name)
    except Exception as exc:
        publication=dict(status='push_failed_local_saved',error=repr(exc));traceback.print_exc()
    write(run/'publication.json',publication)
    return publication
def remaining(deadline):
    seconds=(deadline-now()).total_seconds()
    if seconds<=0: raise TimeoutError('Absolute compute deadline reached')
    return seconds
def one(index,arm,deadline):
    verify_sources(arm)
    assert remaining(deadline)>=55*60,'Insufficient complete-case budget'
    run=ART/(now().strftime('%Y%m%d_%H%M%S_')+f'{index+1}_{arm}_official_4k')
    run.mkdir(exist_ok=False)
    rt.RUN=run;rt.WORKTREES=WT/arm;rt.EXP=EXP;rt.CANDIDATE_COMMITS=COMMITS[arm]
    # Each service is bounded to 55 minutes, within the common shutdown window.
    end=min(deadline,now()+datetime.timedelta(minutes=55))
    rt.STATE=dict(pid=os.getpid(),process_start_ticks=STATE['process_start_ticks'],commits=COMMITS[arm],
                  planned_finish_before=end.isoformat(),arm=arm,session_index=index+1)
    proc=None;result=None
    state('starting_session',session_index=index+1,arm=arm,run_directory=str(run))
    step('Start fresh official 4K session',index=index+1,arm=arm,commits=COMMITS[arm],run_name=run.name)
    try:
        import socket
        idle=[]
        for port in (9031,9032):
            with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',port))!=0
        for _ in range(2):
            out=subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True,timeout=20)
            m=re.search(r'(\d+)MiB\s*/\s*(\d+)MiB\s*\|\s*(\d+)%',out)
            assert m and int(m[1])<=1024 and int(m[3])<=5,'GPU not idle'
            idle.append(dict(memory_mib=int(m[1]),util_percent=int(m[3])));time.sleep(3)
        write(run/'idle_evidence.json',idle)
        assert sha(rt.BENCH_FILE)==rt.BENCH_SHA
        rt.verify_installed_environment()
        rt.state('service_starting')
        proc=rt.launch(run/'candidate_benchmark',True)
        rt.smoke(run/'candidate_benchmark')
        verify_route(run,arm,run/'candidate_benchmark/startup_route_evidence.json')
        rt.metrics(run/'candidate_benchmark','metrics_before.prom')
        rt.state('candidate_throughput');state('candidate_throughput',server_pid=proc.pid)
        command=rt.BENCH[:-1]+['[[4096,1024,64,256]]']
        rt.run_command(command,run/'candidate_benchmark','benchmark',timeout=remaining(end))
        rt.metrics(run/'candidate_benchmark','metrics_after.prom')
        verify_sources(arm);assert sha(rt.BENCH_FILE)==rt.BENCH_SHA
        verify_route(run,arm)
        raw,summary=rt.validate_benchmark(run/'candidate_benchmark',{(4096,1024,64,256)})
        assert (run/'candidate_benchmark/benchmark.log').read_text().count('Failed requests:                         0')==4
        for key in ['Total tok/s','Output tok/s','Mean TTFT (ms)','Mean TPOT (ms)','Mean ITL (ms)']:
            assert abs(float(summary[0][key])-statistics.mean(float(r[key]) for r in raw[1:]))<=.011
        retained=[float(r['Total tok/s']) for r in raw[1:]]
        result=dict(index=index+1,arm=arm,run_name=run.name,commits=COMMITS[arm],summary=summary[0],
            retained_total_tokens_s=retained,cv_percent=statistics.stdev(retained)/statistics.mean(retained)*100,
            status='complete_case_validated',full_task_accuracy='deferred_by_user',partial_runs_combined=False)
        write(run/'result.json',result);rt.state('performance_case_complete',result=result)
        step('Complete official 4K session, zero failed requests, discard first',**result)
    except BaseException as exc:
        write(run/'error.json',dict(error=repr(exc),time=now().isoformat()))
        rt.state('failed',error=repr(exc));step('Session failed; preserve raw data',arm=arm,index=index+1,error=repr(exc))
        raise
    finally:
        rt.stop(proc)
        # Do not expose the GPU to another session until this service has stopped.
        assert not rt.ready(),'Own server did not stop'
        publication=archive(run)
    return result,publication
def summarize(results):
    assert [r['arm'] for r in results]==ORDER and len(results)==4
    metrics={}
    for key in ['Total tok/s','Output tok/s','Mean TTFT (ms)','Mean TPOT (ms)','Mean ITL (ms)']:
        vals=[float(r['summary'][key]) for r in results]
        a,b=statistics.mean([vals[0],vals[3]]),statistics.mean(vals[1:3])
        metrics[key]=dict(A=a,B=b,change_percent=(b/a-1)*100,
            pair1_change_percent=(vals[1]/vals[0]-1)*100,pair2_change_percent=(vals[2]/vals[3]-1)*100,
            A_endpoint_change_percent=(vals[3]/vals[0]-1)*100,B_repeat_change_percent=(vals[2]/vals[1]-1)*100)
    out=dict(status='complete_ABBA',order=ORDER,metrics=metrics,sessions=results,
        interpretation='Two fresh sessions per arm; full official case each, discard first separately. ABBA balances linear time drift, not nonlinear drift; rounds are correlated, no significance claim. No combination with EXP20 official result; no automatic serving selection.')
    write(EXP/'RESULTS.json',out)
    lines=['# EXP-0021：4K ABBA配对复测','','每次独立启动服务、缓存隔离，官方完整四轮弃首轮。原始EXP-0020结果保持独立。','',
           '| 指标 | A：原页表读取 | B：页号广播 | 变化 |','|---|---:|---:|---:|']
    for k,v in metrics.items():lines.append(f"| {k} | {v['A']:.2f} | {v['B']:.2f} | {v['change_percent']:+.3f}% |")
    lines += ['','ABBA可平衡近似线性时间漂移；只有每版本两个独立会话，不声明统计显著。轮次并非独立实验。',
              '同时查看两组相邻对照及A首尾、B重复变化。保留所有轮次，不挑最快结果。全模型精度暂缓，性能版本选择需另行评审。']
    (EXP/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return out
def main():
    assert not (ART/'status.json').exists(),'One-shot controller; never silently restart partial ABBA'
    deadline=datetime.datetime.fromisoformat(read(ART/'run_window.json')['compute_stop_at'])
    assert deadline.date()==now().date()
    assert remaining(deadline)>=4*55*60+10*60,'Insufficient budget for predetermined ABBA'
    STATE.update(pid=os.getpid(),process_start_ticks=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()[19],
                 started_at=now().isoformat(),order=ORDER,completed_sessions=[],deadline=deadline.isoformat())
    lock=(Path('/workspace/experiments/codex_20261019')/'runner.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not rt.ready(),'Service port occupied'
        state('running_ABBA')
        for i,arm in enumerate(ORDER):
            result,publication=one(i,arm,deadline)
            STATE['completed_sessions'].append(result)
            state('session_archived',publication=publication)
            # Stop on incomplete results; do not stop on a performance outcome or select favorable trials.
        summary=summarize(STATE['completed_sessions']);step('Complete predetermined ABBA',metrics=summary['metrics'])
        state('complete',metrics=summary['metrics']);write(EXP/'STATUS.json',STATE)
        try: pub=publish(['experiments/EXP-0021/RESULTS.json','experiments/EXP-0021/RESULTS.md','experiments/EXP-0021/STATUS.json','experiments/EXP-0021/steps.jsonl'], 'experiments: report complete 4K ABBA prefill comparison')
        except Exception as exc:pub=dict(status='push_failed_local_saved',error=repr(exc))
        write(ART/'publication.json',pub)
    except BaseException as exc:
        state('failed',error=repr(exc));step('ABBA controller failed; no automatic restart',error=repr(exc));traceback.print_exc()
        write(EXP/'STATUS.json',STATE)
        try:write(ART/'publication.json',publish(['experiments/EXP-0021/STATUS.json','experiments/EXP-0021/steps.jsonl'],'experiments: retain interrupted ABBA status'))
        except Exception:traceback.print_exc()
        raise
    finally:lock.close()
if __name__=='__main__':main()
