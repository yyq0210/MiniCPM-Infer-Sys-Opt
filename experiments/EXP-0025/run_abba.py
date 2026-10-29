"""EXP25: four bounded, fresh official 4K sessions; no automatic selection."""
import datetime
import fcntl
import os
import re
import signal
import socket
import statistics
import subprocess
import time
import traceback
from common import *
import frozen_runtime as rt
from layout_checks import layout_route

STATE = {}

def state(stage, **kw):
    STATE.update(stage=stage, updated_at=now().isoformat(), **kw)
    write(ART/'status.json', STATE)
    print(json.dumps(STATE), flush=True)

def sanitize(text):
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[IP_REDACTED]', text)
    text = re.sub(r'uuid=[^,)]+', 'uuid=[REDACTED]', text)
    text = re.sub(r'GPU-[0-9a-fA-F-]{16,}', '[GPU_ID_REDACTED]', text)
    return '\n'.join(line.rstrip() for line in text.splitlines())+'\n'

def archive(run):
    dest = EXP/'artifacts'/run.name
    dest.mkdir(parents=True, exist_ok=False)
    files = [p for p in run.iterdir() if p.is_file() and p.suffix in ('.json', '.jsonl')]
    folder = run/'candidate_benchmark'
    files += [folder/x for x in ['server.log', 'server.command.json', 'benchmark.log', 'benchmark.command.json',
        'smoke.json', 'route_evidence.json', 'startup_route_evidence.json', 'layout_route_evidence.json',
        'startup_layout_route_evidence.json', 'metrics_before.prom', 'metrics_after.prom']]
    files += list((folder/'benchmark_results').glob('*.csv'))
    manifest = {}
    for src in files:
        if not src.is_file():
            continue
        rel = src.relative_to(run)
        target = dest/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(sanitize(src.read_text(errors='replace')))
        manifest[str(rel)] = dict(raw_sha256=sha(src), archived_sha256=sha(target))
    write(dest/'artifact_manifest.json', manifest)
    write(EXP/'STATUS.json', STATE)
    try:
        publication = publish(['experiments/EXP-0025/artifacts/'+run.name,
            'experiments/EXP-0025/STATUS.json', 'experiments/EXP-0025/steps.jsonl'],
            'experiments: archive MLP layout ABBA session '+run.name)
    except Exception as exc:
        publication = dict(status='push_failed_local_saved', error=repr(exc))
        traceback.print_exc()
    write(run/'publication.json', publication)
    return publication

def remaining(deadline):
    seconds = (deadline-now()).total_seconds()
    if seconds <= 0:
        raise TimeoutError('Absolute compute deadline reached')
    return seconds

def abort(*unused):
    raise TimeoutError('Session deadline or termination signal')

BASE_ENV = rt.env

def one(index, arm, deadline):
    verify_sources(arm)
    assert remaining(deadline) >= 55*60, 'Insufficient complete-session budget'
    run = ART/(now().strftime('%Y%m%d_%H%M%S_')+f'{index+1}_{arm}_official_4k')
    run.mkdir(exist_ok=False)
    rt.RUN, rt.WORKTREES, rt.EXP, rt.CANDIDATE_COMMITS = run, WT/arm, EXP, COMMITS[arm]
    def environment(candidate):
        env = BASE_ENV(candidate)
        env.update(VLLM_FL_MLP_WEIGHT_LAYOUT='0' if arm=='A' else '1',
                   TORCHINDUCTOR_CACHE_DIR=str(run/'inductor_cache'))
        return env
    rt.env = environment
    end = min(deadline, now()+datetime.timedelta(minutes=55))
    rt.STATE = dict(pid=os.getpid(), process_start_ticks=STATE['process_start_ticks'], commits=COMMITS[arm],
        planned_finish_before=end.isoformat(), arm=arm, session_index=index+1, started_at=now().isoformat())
    state('starting_session', session_index=index+1, arm=arm, run_directory=str(run), server_pid=None)
    step('Start fresh official 4K session', index=index+1, arm=arm, commits=COMMITS[arm], run_name=run.name)
    proc = None
    result = None
    signal.setitimer(signal.ITIMER_REAL, remaining(end))
    try:
        for port in (9031, 9032):
            with socket.socket() as sock:
                assert sock.connect_ex(('127.0.0.1', port)) != 0, 'Existing port owner; do not interfere'
        idle = []
        for _ in range(2):
            out = subprocess.check_output(['/usr/local/corex/bin/ixsmi'], text=True, timeout=20)
            m = re.search(r'(\d+)MiB\s*/\s*(\d+)MiB\s*\|\s*(\d+)%', out)
            assert m and int(m[1]) <= 1024 and int(m[3]) <= 5, 'GPU not idle'
            clocks = re.search(r'(\d+)MHz\s+(\d+)MHz', out)
            power = re.search(r'(\d+)C\s+P\d+\s+(\d+)W\s*/\s*(\d+)W', out)
            idle.append(dict(memory_mib=int(m[1]), util_percent=int(m[3]),
                clocks_mhz=list(map(int, clocks.groups())) if clocks else None,
                temperature_power=list(map(int, power.groups())) if power else None))
            time.sleep(3)
        write(run/'idle_evidence.json', idle)
        env = rt.env(True)
        write(run/'explicit_environment.json', {k:env[k] for k in ['VLLM_PLUGINS', 'PYTHONPATH',
            'VLLM_CACHE_ROOT', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'VLLM_FL_MLP_WEIGHT_LAYOUT']})
        assert sha(rt.BENCH_FILE) == rt.BENCH_SHA
        rt.verify_installed_environment()
        rt.state('service_starting')
        folder = run/'candidate_benchmark'
        proc = rt.launch(folder, True)
        rt.smoke(folder)
        rt.verify_route(folder)
        layout_route(run, folder, REFS['A']/'candidate_benchmark', arm)
        for name in ['route_evidence.json', 'layout_route_evidence.json']:
            (folder/('startup_'+name)).write_bytes((folder/name).read_bytes())
        rt.metrics(folder, 'metrics_before.prom')
        assert (folder/'metrics_before.prom').is_file()
        rt.state('candidate_throughput')
        state('candidate_throughput', server_pid=proc.pid)
        command = rt.BENCH[:-1]+['[[4096,1024,64,256]]']
        rt.run_command(command, folder, 'benchmark', timeout=remaining(end))
        rt.metrics(folder, 'metrics_after.prom')
        assert (folder/'metrics_after.prom').is_file()
        verify_sources(arm)
        assert sha(rt.BENCH_FILE) == rt.BENCH_SHA
        rt.verify_installed_environment()
        rt.verify_route(folder)
        layout = layout_route(run, folder, REFS['A']/'candidate_benchmark', arm)
        raw, summary = rt.validate_benchmark(folder, {(4096, 1024, 64, 256)})
        assert (folder/'benchmark.log').read_text().count('Failed requests:                         0') == 4
        for key in ['Total tok/s', 'Output tok/s', 'Mean TTFT (ms)', 'Mean TPOT (ms)', 'Mean ITL (ms)']:
            assert abs(float(summary[0][key])-statistics.mean(float(r[key]) for r in raw[1:])) <= .011
        retained = [float(r['Total tok/s']) for r in raw[1:]]
        result = dict(index=index+1, arm=arm, run_name=run.name, commits=COMMITS[arm], summary=summary[0],
            retained_total_tokens_s=retained, cv_percent=statistics.stdev(retained)/statistics.mean(retained)*100,
            kv_tokens=layout['candidate_kv_tokens'], status='complete_case_validated',
            full_task_accuracy='deferred_by_user', partial_runs_combined=False)
        write(run/'result.json', result)
        rt.state('performance_case_complete', result=result)
        step('Complete official 4K session; all four retained in raw, first discarded by protocol', **result)
    except BaseException as exc:
        write(run/'error.json', dict(error=repr(exc), traceback=traceback.format_exc(), time=now().isoformat()))
        rt.state('failed', error=repr(exc))
        step('Session failed; preserve raw data', arm=arm, index=index+1, error=repr(exc))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            rt.stop(proc)
            # rt.launch cleans its own process if readiness failed before returning.
            assert not rt.ready(), 'Own service failed to stop'
            rt.state(rt.STATE['stage'], owned_children_reaped=True, finished_at=now().isoformat())
        except BaseException as exc:
            rt.state('failed', owned_children_reaped=False, cleanup_error=repr(exc), finished_at=now().isoformat())
            raise
        finally:
            publication = archive(run)
    return result, publication

def summarize(results):
    assert [r['arm'] for r in results] == ORDER and len(results) == 4
    metrics = {}
    for key in ['Total tok/s', 'Output tok/s', 'Mean TTFT (ms)', 'Mean TPOT (ms)', 'Mean ITL (ms)']:
        vals = [float(r['summary'][key]) for r in results]
        a, b = statistics.mean([vals[0], vals[3]]), statistics.mean(vals[1:3])
        metrics[key] = dict(A=a, B=b, change_percent=(b/a-1)*100,
            pair1_change_percent=(vals[1]/vals[0]-1)*100, pair2_change_percent=(vals[2]/vals[3]-1)*100,
            A_endpoint_change_percent=(vals[3]/vals[0]-1)*100, B_repeat_change_percent=(vals[2]/vals[1]-1)*100)
    accounting = {}
    for label, ai, bi in [('pair1', 0, 1), ('pair2', 3, 2)]:
        accounting[label] = rt.compare_schedule(ART/results[ai]['run_name']/'candidate_benchmark',
            ART/results[bi]['run_name']/'candidate_benchmark', EXP/(label+'_scheduler.json'))
    out = dict(status='complete_ABBA', order=ORDER, metrics=metrics, sessions=results, scheduler=accounting,
        selected_experiment='EXP-0020', selection_changed=False,
        interpretation='Two sessions per arm; official four rounds, discard first separately. ABBA balances approximately linear drift only. Rounds are correlated; no statistical significance claim. Historical results excluded. Independent review required before selection.')
    write(EXP/'RESULTS.json', out)
    lines = ['# EXP-0025：MLP 权重布局官方 4K ABBA', '',
        'A 为当前 EXP20；B 为 EXP24，加载时同时重排 gate/up 和 down 权重。每会话独立服务及缓存，官方四轮弃首。', '',
        '| 指标 | A：原权重布局 | B：新权重布局 | 变化 |', '|---|---:|---:|---:|']
    for key, v in metrics.items():
        lines.append(f"| {key} | {v['A']:.2f} | {v['B']:.2f} | {v['change_percent']:+.3f}% |")
    lines += ['', '| 会话 | 保留轮均值 tok/s | 保留三轮 CV | KV 容量 tokens |', '|---|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['index']}/{r['arm']} | {float(r['summary']['Total tok/s']):.2f} | {r['cv_percent']:.3f}% | {r['kv_tokens']} |")
    v = metrics['Total tok/s']
    lines += ['', f"相邻配对变化：{v['pair1_change_percent']:+.3f}% / {v['pair2_change_percent']:+.3f}%；A 首尾漂移 {v['A_endpoint_change_percent']:+.3f}%；B 重复变化 {v['B_repeat_change_percent']:+.3f}%。",
        '', '每组只有两个独立服务会话；ABBA 只平衡近似线性漂移。慢轮全部保留，历史结果不混入，不声称统计显著。',
        '自动归档不切换推理基线；当前继续选择 EXP20，待独立审核。整模任务精度按用户要求暂缓。']
    (EXP/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return out

def main():
    assert not (ART/'status.json').exists(), 'One-shot; inspect incomplete group, never silently restart'
    deadline = datetime.datetime.fromisoformat(read(ART/'run_window.json')['compute_stop_at'])
    assert deadline.date() == now().date()
    assert remaining(deadline) >= 230*60
    STATE.update(pid=os.getpid(), process_start_ticks=Path('/proc/self/stat').read_text().rsplit(') ', 1)[1].split()[19],
        started_at=now().isoformat(), order=ORDER, completed_sessions=[], deadline=deadline.isoformat())
    signal.signal(signal.SIGTERM, abort)
    signal.signal(signal.SIGALRM, abort)
    lock = Path('/workspace/experiments/codex_20261019/runner.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        state('running_ABBA')
        for i, arm in enumerate(ORDER):
            result, publication = one(i, arm, deadline)
            STATE['completed_sessions'].append(result)
            state('session_archived', publication=publication)
        summary = summarize(STATE['completed_sessions'])
        step('Complete fixed MLP-layout ABBA; independent review pending', metrics=summary['metrics'])
        state('complete', metrics=summary['metrics'])
        write(EXP/'STATUS.json', STATE)
        paths = ['experiments/EXP-0025/'+p for p in ['RESULTS.json', 'RESULTS.md', 'pair1_scheduler.json',
            'pair2_scheduler.json', 'STATUS.json', 'steps.jsonl']]
        try:
            pub = publish(paths, 'experiments: report complete MLP layout official 4K ABBA')
        except Exception as exc:
            pub = dict(status='push_failed_local_saved', error=repr(exc))
        write(ART/'publication.json', pub)
    except BaseException as exc:
        state('failed', error=repr(exc))
        step('ABBA stopped; no automatic restart or selection', error=traceback.format_exc())
        traceback.print_exc()
        write(EXP/'STATUS.json', STATE)
        try:
            write(ART/'publication.json', publish(['experiments/EXP-0025/STATUS.json', 'experiments/EXP-0025/steps.jsonl'],
                'experiments: preserve interrupted MLP layout ABBA'))
        except Exception:
            traceback.print_exc()
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        lock.close()

if __name__ == '__main__':
    main()
