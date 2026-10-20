"""Resume official-case A/B with pinned code and unchanged workload parameters."""
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

EXP = Path(__file__).resolve().parent
ROOT = Path('/workspace/LeanScale')
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
CANDIDATE_COMMITS = {'vllm-plugin-FL': '0ad2b82abdd8563e6a27d20b67341a109f2c2e7c',
                     'FlagGems': '646668a82b64c9ddb2c79e50ecf25a5c229b8560'}


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
    base = str(ROOT) if candidate else '/workspace'
    result = dict(os.environ, VLLM_PLUGINS='fl', PYTHONUNBUFFERED='1',
                  PYTHONPATH=f'{base}/vllm-plugin-FL:{base}/FlagGems/src')
    if candidate:
        # Force code tracing for this variant instead of reusing a baseline AOT
        # graph. This changes cache location, not the fixed serving arguments.
        result['VLLM_CACHE_ROOT'] = str(RUN/'candidate_cache')
    return result


def verify_route(folder):
    log = (folder/'server.log').read_text(errors='replace')
    assert "Op 'rotary_embedding' using 'default.flagos'" in log, 'RoPE dispatch not observed'
    assert "Op 'rotary_embedding' switched" not in log, 'RoPE fell back during startup'
    graphs = []
    for path in (RUN/'candidate_cache').rglob('computation_graph.py'):
        code = path.read_text()
        calls = [line.strip() for line in code.splitlines()
                 if 'torch.ops.vllm.fl_rotary_embedding_dispatch' in line
                 and not line.lstrip().startswith('#')]
        if calls:
            graphs.append({'path': str(path.relative_to(RUN)), 'calls': len(calls),
                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                           'example': calls[0]})
    assert graphs, 'No optimized RoPE node found in the actual model compilation graph'
    (folder/'route_evidence.json').write_text(json.dumps(graphs, indent=2))


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
        'candidate_policy': 'Fresh service, both official cases, four rounds each; no reuse of partial candidate rounds.',
        'baseline_sessions': 2, 'candidate_is_new_session': True,
    }, indent=2)+'\n')
    return baseline


def verify_installed_environment():
    """Check the installed vLLM sources against the analysis snapshot, without loading a model."""
    index = EXP/'artifacts/analysis_20261020/source_index.json'
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


def main():
    global RUN
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-from', type=Path,
                        help='Reuse complete 4K and 16K baselines from this EXP-0004 run.')
    parser.add_argument('--run-dir', type=Path, help='New directory under artifacts/EXP-0004.')
    args = parser.parse_args()
    stamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    RUN = (args.run_dir or ROOT/'artifacts/EXP-0004'/stamp).resolve()
    assert RUN.parent == ROOT/'artifacts/EXP-0004'
    RUN.mkdir(parents=True, exist_ok=False)
    start_ticks = Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()[19]
    STATE.update(pid=os.getpid(), process_start_ticks=start_ticks,
                 run_directory=str(RUN), started_at=stamp)
    proc = None
    lock = (BASE/'runner.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not ready(), 'Port 9031 is occupied'
        assert hashlib.sha256(BENCH_FILE.read_bytes()).hexdigest() == BENCH_SHA
        commits = {}
        baseline_commits = BASELINE_COMMITS
        for candidate in (False, True):
            for repo in ('vllm-plugin-FL', 'FlagGems'):
                path = (ROOT if candidate else Path('/workspace'))/repo
                sha = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
                assert not subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True)
                if candidate:
                    assert sha == CANDIDATE_COMMITS[repo], (repo, sha)
                    commits[repo] = sha
                else:
                    assert sha == baseline_commits[repo]
        state('environment_verifying', commits=commits, benchmark_sha256=BENCH_SHA,
              accuracy_status='deferred_by_user; math_500_not_run',
              protocol=('Retain both complete baseline cases; rerun both official candidate cases in a fresh service.'
                        if args.baseline_from else
                        'official cases only; retain completed baseline 4K, rerun interrupted baseline 16K'))
        verify_installed_environment()
        baseline = retain_baseline(args.baseline_from) if args.baseline_from else None
        state('validation_running')
        run_command(['python', str(EXP/'validate_rope_e2e.py'), str(RUN/'validation')], RUN, 'validation')
        assert json.loads((RUN/'validation/results.json').read_text())['status'] == 'passed'
        # Check model compilation and the actual route before the long official
        # run. This service is stopped; benchmark services get fresh KV caches.
        if baseline is None:
            state('candidate_preflight_starting')
            proc = launch(RUN/'candidate_preflight', True)
            verify_route(RUN/'candidate_preflight')
            smoke(RUN/'candidate_preflight')
            stop(proc)
            proc = None
            baseline_4k = validate_benchmark(BASE/'baseline_official', {(4096,1024,64,256)})
            state('baseline_16k_starting')
            proc = launch(RUN/'baseline_16k', False)
            smoke(RUN/'baseline_16k')
            state('baseline_16k_throughput')
            bench_16k = BENCH[:-1] + ['[[16384,1024,64,128]]']
            run_command(bench_16k, RUN/'baseline_16k', 'benchmark', candidate=False, timeout=86400)
            baseline_16k = validate_benchmark(RUN/'baseline_16k', {(16384,1024,64,128)})
            baseline = (baseline_4k[0]+baseline_16k[0], baseline_4k[1]+baseline_16k[1])
            provenance = {'retained_4k': str(BASE/'baseline_official'),
                          'rerun_16k': str(RUN/'baseline_16k'),
                          'note': 'Baseline cases were measured in two server sessions after an intentional interruption; each case has four complete runs.',
                          'baseline_commits': baseline_commits, 'raw': baseline[0], 'summary': baseline[1]}
            (RUN/'baseline_combined.json').write_text(json.dumps(provenance, indent=2))
            stop(proc)
            proc = None
        state('candidate_official_starting')
        proc = launch(RUN/'candidate_benchmark', True)
        verify_route(RUN/'candidate_benchmark')
        smoke(RUN/'candidate_benchmark')
        state('candidate_throughput')
        run_command(BENCH, RUN/'candidate_benchmark', 'benchmark', timeout=86400)
        assert hashlib.sha256(BENCH_FILE.read_bytes()).hexdigest() == BENCH_SHA
        verify_route(RUN/'candidate_benchmark')
        optimized = validate_benchmark(RUN/'candidate_benchmark')
        report = comparison(baseline, optimized)
        state('performance_complete', performance=report,
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
