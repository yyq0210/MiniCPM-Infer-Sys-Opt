"""EXP-0010 official KV admission experiment, preserving the EXP-0006 kernels."""
import argparse
import csv
from collections import Counter
from datetime import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import shutil
import statistics
import subprocess
import time
import traceback
import urllib.request
from analyze_schedule import compare as compare_schedule

EXP = Path(__file__).resolve().parent
ROOT = Path('/workspace/LeanScale')
WORKTREES = ROOT/'worktrees/EXP-0010'
BASE = Path('/workspace/experiments/codex_20261019')
RUN = None  # Initialized in main; importing this module must not start a run.
SERVER = ['vllm', 'serve', '/workspace/MiniCPM5-2B', '--port', '9031',
          '--compilation-config', '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
          '--served-model-name', 'minicpm', '--gpu-memory-utilization', '0.85',
          '--max-model-len', '131072']
BENCH_FILE = Path('/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py')
BENCH_SHA = '0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565'
BENCH = ['python3', str(BENCH_FILE), '--model', '/workspace/MiniCPM5-2B',
         '--served-model-name', 'minicpm', '--port', '9031',
         '--test-cases', '[[4096,1024,64,256],[16384,1024,64,128]]']
CASES = {(4096, 1024, 64, 256), (16384, 1024, 64, 128)}
STATE = {}
BASELINE_COMMITS = {'vllm-plugin-FL': '13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0',
                    'FlagGems': 'a7620cc191a0b42e040194622c5758b22a7a25dc'}
CANDIDATE_COMMITS = json.loads((EXP/'candidate_commits.json').read_text())


def state(stage, **kw):
    STATE.update(stage=stage, updated_at=datetime.now().astimezone().isoformat(timespec='seconds'), **kw)
    text = json.dumps(STATE, indent=2, ensure_ascii=False)
    run_tmp = RUN / 'status.json.tmp'
    run_tmp.write_text(text)
    run_tmp.replace(RUN / 'status.json')
    tmp = EXP / 'STATUS.json.tmp'
    tmp.write_text(text)
    tmp.replace(EXP / 'STATUS.json')
    with (RUN / 'stage_history.jsonl').open('a') as history:
        history.write(json.dumps(STATE, ensure_ascii=False)+'\n')
    print(json.dumps({'stage': stage, **kw}, ensure_ascii=False), flush=True)


def env(candidate):
    base = str(WORKTREES) if candidate else '/workspace'
    result = dict(os.environ, VLLM_PLUGINS='fl', PYTHONUNBUFFERED='1',
                  PYTHONPATH=f'{base}/vllm-plugin-FL:{base}/FlagGems/src')
    if candidate:
        # Force code tracing for this variant instead of reusing a baseline AOT
        # graph. This changes cache location, not the fixed serving arguments.
        result['VLLM_CACHE_ROOT'] = str(RUN/'candidate_cache')
    return result


def verify_route(folder):
    log = (folder/'server.log').read_text(errors='replace')
    assert "Op 'rotary_embedding' using 'default.flagos'" in log
    assert "Op 'rotary_embedding' switched" not in log
    assert log.count('LeanScale GQA active:') == 42, 'Expected 42 GQA layers'
    assert log.count('LeanScale fused RoPE+KV active:') == 42, 'Expected 42 fused runtime layers'
    assert 'LeanScale RoPE+KV graph fusion: replaced=42' in log
    graphs = []
    for path in (RUN/'candidate_cache').rglob('computation_graph.py'):
        lines = [line.strip() for line in path.read_text().splitlines()
                 if not line.lstrip().startswith('#')]
        counts = {name: sum('torch.ops.vllm.'+name in line for line in lines)
                  for name in ('fl_rope_kv_dispatch', 'fl_rotary_embedding_dispatch',
                               'unified_kv_cache_update')}
        if counts['fl_rope_kv_dispatch']:
            assert counts == {'fl_rope_kv_dispatch':42,
                              'fl_rotary_embedding_dispatch':0,
                              'unified_kv_cache_update':0}, counts
            graphs.append(dict(path=str(path.relative_to(RUN)), counts=counts,
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert graphs, 'No complete fused model graph found'
    assert ('Capturing CUDA graphs (decode, FULL)' in log and '35/35' in log), (
        'Complete FULL_DECODE_ONLY graph capture not observed')
    assert 'LeanScale KV growth admission enabled=True policy=full_max_tokens' in log
    assert 'Selecting LeanScale async scheduler with KV growth admission.' in log
    (folder/'route_evidence.json').write_text(json.dumps(
        dict(status='passed', scheduler='KVReservationAsyncScheduler', kv_growth_admission=True, fused_runtime_layers=42, gqa_layers=42,
             old_cache_updates_in_fused_graph=0, graphs=graphs), indent=2)+'\n')


def run_command(argv, folder, name, candidate=True, timeout=1800):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / (name + '.command.json')).write_text(json.dumps(argv, ensure_ascii=False, indent=2))
    with (folder / (name + '.log')).open('w') as log:
        proc = subprocess.Popen(argv, cwd=folder, env=env(candidate), stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = proc.wait(timeout=timeout)
            if returncode:
                raise subprocess.CalledProcessError(returncode, argv)
        finally:
            stop(proc)


def ready():
    try:
        with urllib.request.urlopen('http://127.0.0.1:9031/health', timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def launch(folder, candidate):
    if ready():
        raise RuntimeError('Port 9031 is occupied; refusing to touch the existing service')
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'server.command.json').write_text(json.dumps(SERVER, indent=2))
    with (folder / 'server.log').open('w') as log:
        proc = subprocess.Popen(SERVER, cwd=folder, env=env(candidate), stdout=log,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
    STATE['server_pid'] = proc.pid
    deadline = time.monotonic() + 1800
    try:
        while not ready():
            planned = STATE.get("planned_finish_before")
            if planned and datetime.now().astimezone() >= datetime.fromisoformat(planned):
                raise TimeoutError("Planned shutdown checkpoint reached during startup")
            if proc.poll() is not None:
                raise RuntimeError(f'Server exited with {proc.returncode}')
            if time.monotonic() > deadline:
                raise TimeoutError('Server did not become ready')
            time.sleep(5)
    except BaseException:
        stop(proc)
        raise
    return proc


def stop(proc):
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()



def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def validate_benchmark(folder, cases=CASES):
    raw = rows(sorted((folder / 'benchmark_results').glob('raw_runs_*.csv'))[-1])
    summary = rows(sorted((folder / 'benchmark_results').glob('summary_*.csv'))[-1])
    def key(row):
        return tuple(int(float(row[k])) for k in ('Prefill', 'Decode', 'Conc', 'Num Prompts'))
    assert Counter(key(r) for r in raw) == Counter({c: 4 for c in cases})
    assert len(summary) == len(cases) and {key(r) for r in summary} == cases
    for r in raw:
        prefill, decode, _, n = key(r)
        assert r['Run Status'] == 'SUCCESS' and int(float(r['Successful Requests'])) == n
        assert int(float(r['Total Input Tokens'])) == prefill * n
        assert int(float(r['Total Output Tokens'])) == decode * n
    for item in summary:
        trials = [r for r in raw if key(r) == key(item)][1:]
        for metric in ('Total tok/s', 'Mean TTFT (ms)', 'Output tok/s'):
            assert abs(float(item[metric]) - statistics.mean(float(r[metric]) for r in trials)) <= .011
    return raw, summary


def retain_baseline(source):
    """Reuse only complete baseline cases; never combine partial candidate rounds."""
    source = source.resolve()
    assert source.parent == ROOT/'artifacts/EXP-0004'
    original = source/'baseline_combined.json'
    data = json.loads(original.read_text())
    assert data['baseline_commits'] == BASELINE_COMMITS
    old4k = Path(data['retained_4k'])
    old16k = Path(data['rerun_16k'])
    assert old4k == BASE/'baseline_official'
    assert old16k == source/'baseline_16k'
    four = validate_benchmark(old4k, {(4096,1024,64,256)})
    sixteen = validate_benchmark(old16k, {(16384,1024,64,128)})
    baseline = (four[0]+sixteen[0], four[1]+sixteen[1])
    assert data['raw'] == baseline[0] and data['summary'] == baseline[1]
    fingerprints = {'baseline_combined.json': hashlib.sha256(original.read_bytes()).hexdigest()}
    for label, folder in [('4k', old4k), ('16k', old16k)]:
        for path in (folder/'benchmark_results').glob('*.csv'):
            target = RUN/'retained_baseline'/label/path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            fingerprints[str(target.relative_to(RUN))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (RUN/'baseline_combined.json').write_bytes(original.read_bytes())
    (RUN/'resume_provenance.json').write_text(json.dumps({
        'source_run': str(source), 'retained_baseline_hashes': fingerprints,
        'candidate_policy': 'Fresh service, one complete official case, four rounds, no reuse of partial candidate rounds.',
        'baseline_sessions': 2, 'candidate_is_new_session': True,
    }, indent=2)+'\n')
    return baseline


def verify_installed_environment():
    """Check the installed vLLM sources against the analysis snapshot, without loading a model."""
    index = EXP.parent/'EXP-0004/artifacts/analysis_20261020/source_index.json'
    sources = json.loads(index.read_text())
    checked = {}
    for name, metadata in sources.items():
        if not name.startswith('vllm/'):
            continue
        path = Path('/usr/local/lib/python3.12/site-packages')/name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == metadata['sha256'], 'Installed source changed: '+name
        checked[name] = digest
    # The vendor image installs Triton's module without a standard triton
    # distribution metadata record. Check the actual import used by kernels.
    import triton
    versions = {name: importlib.metadata.version(name) for name in ('torch','vllm')}
    versions['triton'] = triton.__version__
    assert versions == {'torch':'2.10.0+corex.4.5.0','triton':'3.6.0','vllm':'0.24.0+empty'}, versions
    (RUN/'environment_check.json').write_text(json.dumps({
        'versions': versions, 'installed_source_hashes': checked,
        'triton_module': triton.__file__,
        'triton_init_sha256': hashlib.sha256(Path(triton.__file__).read_bytes()).hexdigest(),
        'candidate_commits': CANDIDATE_COMMITS,
        'benchmark_sha256': BENCH_SHA,
    }, indent=2)+'\n')


def comparison(before, after):
    old_raw, old_summary = before
    new_raw, new_summary = after
    report = []
    for old in old_summary:
        new = next(r for r in new_summary if r['Prefill'] == old['Prefill'])
        a, b = float(old['Total tok/s']), float(new['Total tok/s'])
        t1, t2 = float(old['Mean TTFT (ms)']), float(new['Mean TTFT (ms)'])
        item = {'prefill': int(old['Prefill']), 'baseline_total_tokens_s': a,
                'candidate_total_tokens_s': b, 'throughput_change_percent': (b/a-1)*100,
                'baseline_ttft_ms': t1, 'candidate_ttft_ms': t2,
                'ttft_change_percent': (t2/t1-1)*100}
        for label, raw in [('baseline', old_raw), ('candidate', new_raw)]:
            runs = [float(r['Total tok/s']) for r in raw if r['Prefill'] == old['Prefill']][1:]
            item[label+'_retained_runs'] = runs
            item[label+'_cv_percent'] = statistics.stdev(runs)/statistics.mean(runs)*100
        report.append(item)
    (RUN / 'comparison.json').write_text(json.dumps(report, indent=2))
    return report


def smoke(folder):
    body = json.dumps({'model': 'minicpm', 'prompt': 'The capital of France is',
                       'max_tokens': 32, 'temperature': 0}).encode()
    req = urllib.request.Request('http://127.0.0.1:9031/v1/completions', body,
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=180) as response:
        sample = json.load(response)
    (folder/'smoke.json').write_text(json.dumps(sample, ensure_ascii=False, indent=2))
    assert sample['usage']['completion_tokens'] > 0


def metrics(folder, name):
    try:
        with urllib.request.urlopen('http://127.0.0.1:9031/metrics', timeout=10) as r:
            body = r.read().decode()
        selected = [line for line in body.splitlines() if line.startswith('vllm:')]
        (folder/name).write_text('\n'.join(selected)+'\n')
    except Exception as exc:
        (folder/(name+'.error')).write_text(repr(exc))


def verify_candidate_gate():
    gate = json.loads((EXP/'candidate_validation.json').read_text())
    assert gate['status'] == 'passed'
    assert gate['candidate_commits'] == CANDIDATE_COMMITS
    for filename, expected in gate['diagnostic_hashes'].items():
        path = Path(filename)
        assert path.is_relative_to(ROOT/'artifacts/EXP-0010')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        assert json.loads(path.read_text())['status'] == 'passed'
    for repo, files in gate['code_hashes'].items():
        for relative, expected in files.items():
            path = (WORKTREES/repo/relative).resolve()
            assert path.is_relative_to(WORKTREES/repo)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    (RUN/'candidate_validation.json').write_text(json.dumps(gate, indent=2)+'\n')


def main():
    global RUN
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=('4k','16k'), required=True)
    parser.add_argument('--finish-before', type=datetime.fromisoformat)
    parser.add_argument('--baseline-from', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    requested_prefill = 4096 if args.case == '4k' else 16384
    selected_cases = {c for c in CASES if c[0] == requested_prefill}
    bench_command = BENCH[:-1]+[json.dumps([list(c) for c in sorted(selected_cases)], separators=(',',':'))]
    def remaining_seconds():
        if args.finish_before is None:
            return 86400
        remaining = (args.finish_before-datetime.now().astimezone()).total_seconds()
        if remaining <= 0:
            raise TimeoutError('Planned shutdown checkpoint deadline reached')
        return remaining
    def filter_cases(data):
        return tuple([r for r in rows if int(r['Prefill']) == requested_prefill] for rows in data)
    RUN = args.run_dir.resolve()
    assert RUN.parent == ROOT/'artifacts/EXP-0010'
    RUN.mkdir(parents=True, exist_ok=False)
    start_ticks = Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()[19]
    STATE.update(pid=os.getpid(), process_start_ticks=start_ticks,
                 run_directory=str(RUN), started_at=datetime.now().astimezone().isoformat(),
                 selected_cases=[list(c) for c in selected_cases],
                 planned_finish_before=args.finish_before.isoformat() if args.finish_before else None)
    proc = None
    lock = (BASE/'runner.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not ready(), 'Port 9031 is occupied'
        assert hashlib.sha256(BENCH_FILE.read_bytes()).hexdigest() == BENCH_SHA
        commits = {}
        for candidate in (False, True):
            for name in ('vllm-plugin-FL', 'FlagGems'):
                path = (WORKTREES if candidate else Path('/workspace'))/name
                sha = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
                assert not subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True)
                assert sha == (CANDIDATE_COMMITS if candidate else BASELINE_COMMITS)[name]
                if candidate:
                    commits[name] = sha
        state('environment_verifying', commits=commits, benchmark_sha256=BENCH_SHA,
              accuracy_status='deferred_by_user; math_500_not_run',
              protocol='Full max_tokens KV growth admission on EXP-0006 fusion; one complete official case; historical original and EXP-0006 references; no full-model accuracy claim.')
        verify_candidate_gate()
        verify_installed_environment()
        baseline = filter_cases(retain_baseline(args.baseline_from))
        parent_receipt = json.loads((ROOT/('artifacts/EXP-0006-latest-'+args.case+'.json')).read_text())
        parent_run = Path(parent_receipt['run_directory'])
        parent_state = json.loads((parent_run/'status.json').read_text())
        assert parent_state['stage'] == 'performance_case_complete'
        assert parent_state['commits'] == json.loads((EXP/'parent_commits.json').read_text())
        assert json.loads((parent_run/'completion_handler_status.json').read_text())['complete_results']
        previous_path = parent_run/'candidate_benchmark'
        previous = filter_cases(validate_benchmark(previous_path, selected_cases))
        (RUN/'exp6_reference.json').write_text(json.dumps(
            dict(source=str(previous_path), raw=previous[0], summary=previous[1]), indent=2)+'\n')
        remaining_seconds()
        state('candidate_official_starting')
        proc = launch(RUN/'candidate_benchmark', True)
        smoke(RUN/'candidate_benchmark')
        verify_route(RUN/'candidate_benchmark')
        metrics(RUN/'candidate_benchmark', 'metrics_before.prom')
        state('candidate_throughput')
        run_command(bench_command, RUN/'candidate_benchmark', 'benchmark', timeout=remaining_seconds())
        metrics(RUN/'candidate_benchmark', 'metrics_after.prom')
        assert hashlib.sha256(BENCH_FILE.read_bytes()).hexdigest() == BENCH_SHA
        verify_route(RUN/'candidate_benchmark')
        optimized = validate_benchmark(RUN/'candidate_benchmark', selected_cases)
        incremental = comparison(previous, optimized)
        (RUN/'comparison.json').rename(RUN/'comparison_exp6.json')
        report = comparison(baseline, optimized)
        accounting = compare_schedule(
            previous_path, RUN/'candidate_benchmark', RUN/'scheduler_comparison.json')
        state('performance_case_complete', performance=report, incremental_vs_exp6=incremental,
              scheduler_accounting=accounting,
              accuracy_status='deferred_by_user; math_500_not_run')
    except BaseException as exc:
        state('failed', error=repr(exc))
        traceback.print_exc()
        raise
    finally:
        stop(proc)
        lock.close()


if __name__ == '__main__':
    main()
