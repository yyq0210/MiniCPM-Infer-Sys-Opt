"""One finite serial experiment: wait for baseline, validate, compare and evaluate."""
import csv
from collections import Counter
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time
import traceback
import urllib.request

EXP = Path(__file__).resolve().parent
ROOT = Path('/workspace/LeanScale')
BASE = Path('/workspace/experiments/codex_20261019')
STAMP = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
RUN = ROOT / 'artifacts/EXP-0004' / STAMP
RUN.mkdir(parents=True)
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
STATE = {'pid': os.getpid(), 'run_directory': str(RUN), 'started_at': STAMP}


def state(stage, **kw):
    STATE.update(stage=stage, updated_at=datetime.now().astimezone().isoformat(timespec='seconds'), **kw)
    text = json.dumps(STATE, indent=2, ensure_ascii=False)
    (RUN / 'status.json').write_text(text)
    tmp = EXP / 'STATUS.json.tmp'
    tmp.write_text(text)
    tmp.replace(EXP / 'STATUS.json')
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
                 if 'torch.ops.vllm.fl_rotary_embedding_dispatch' in line]
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


PROMPTS = [
    'The capital of France is',
    'Explain in one paragraph why the sky appears blue.',
    'Write a Python function that reverses a list without changing the input.',
    'Solve 3*x + 7 = 34. Show your reasoning.',
    'A rectangle has sides 7 and 11. Calculate its area and perimeter.',
    '用三句话解释什么是旋转位置编码。',
    'List the prime numbers smaller than 30 and explain how to find them.',
    'Summarize this sequence and give its next two numbers: 2, 4, 8, 16, 32.',
]


def generation_samples(folder):
    results = []
    for prompt in PROMPTS:
        body = {'model': 'minicpm', 'prompt': prompt, 'max_tokens': 128,
                'temperature': 0, 'seed': 0, 'logprobs': 1}
        request = urllib.request.Request('http://127.0.0.1:9031/v1/completions',
                                         json.dumps(body).encode(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=300) as response:
            answer = json.load(response)
        assert answer['usage']['completion_tokens'] > 0
        results.append({'request': body, 'response': answer})
    (folder / 'generation_samples.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    return results


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def validate_benchmark(folder):
    raw = rows(sorted((folder / 'benchmark_results').glob('raw_runs_*.csv'))[-1])
    summary = rows(sorted((folder / 'benchmark_results').glob('summary_*.csv'))[-1])
    def key(row):
        return tuple(int(float(row[k])) for k in ('Prefill', 'Decode', 'Conc', 'Num Prompts'))
    assert Counter(key(r) for r in raw) == Counter({c: 4 for c in CASES})
    assert len(summary) == 2 and {key(r) for r in summary} == CASES
    for r in raw:
        prefill, decode, _, n = key(r)
        assert r['Run Status'] == 'SUCCESS' and int(float(r['Successful Requests'])) == n
        assert int(float(r['Total Input Tokens'])) == prefill * n
        assert int(float(r['Total Output Tokens'])) == decode * n
    return raw, summary


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


def main():
    proc = None
    lock = (BASE / 'runner.lock').open('a')
    try:
        state('waiting_baseline')
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(15)
        baseline_state = json.loads((BASE / 'status.json').read_text())
        assert baseline_state['stage'] == 'throughput_complete', baseline_state
        baseline = validate_benchmark(BASE / 'baseline_official')
        assert hashlib.sha256(BENCH_FILE.read_bytes()).hexdigest() == BENCH_SHA
        commits = {}
        for repo in ('vllm-plugin-FL', 'FlagGems'):
            path = ROOT / repo
            commits[repo] = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
            dirty = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True)
            assert not dirty, (repo, dirty)
        state('validation_running', commits=commits, benchmark_sha256=BENCH_SHA)
        run_command(['python', str(EXP/'validate_rope_e2e.py'), str(RUN/'validation')], RUN, 'validation')
        assert json.loads((RUN/'validation/results.json').read_text())['status'] == 'passed'
        state('baseline_generation')
        proc = launch(RUN/'baseline_generation', False)
        before = generation_samples(RUN/'baseline_generation')
        stop(proc)
        proc = None
        state('candidate_starting')
        proc = launch(RUN/'candidate_official', True)
        verify_route(RUN/'candidate_official')
        state('candidate_generation')
        after = generation_samples(RUN/'candidate_official')
        generated = []
        for a, b in zip(before, after):
            ca, cb = a['response']['choices'][0], b['response']['choices'][0]
            generated.append({'prompt': a['request']['prompt'], 'exact_text': ca['text'] == cb['text'],
                              'baseline_tokens': a['response']['usage']['completion_tokens'],
                              'candidate_tokens': b['response']['usage']['completion_tokens']})
        (RUN/'generation_comparison.json').write_text(json.dumps(generated, ensure_ascii=False, indent=2))
        # Restart after diagnostic prompts so the official candidate has the same
        # pre-benchmark workload as EXP-0003: only the 32-token smoke request.
        stop(proc)
        proc = None
        state('candidate_official_starting')
        proc = launch(RUN/'candidate_benchmark', True)
        verify_route(RUN/'candidate_benchmark')
        body = json.dumps({'model':'minicpm','prompt':'The capital of France is','max_tokens':32,'temperature':0}).encode()
        req = urllib.request.Request('http://127.0.0.1:9031/v1/completions', body, {'Content-Type':'application/json'})
        with urllib.request.urlopen(req, timeout=180) as response:
            smoke = json.load(response)
        (RUN/'candidate_benchmark/smoke.json').write_text(json.dumps(smoke, ensure_ascii=False, indent=2))
        assert smoke['usage']['completion_tokens'] > 0
        state('candidate_throughput')
        run_command(BENCH, RUN/'candidate_benchmark', 'benchmark', timeout=14400)
        assert hashlib.sha256(BENCH_FILE.read_bytes()).hexdigest() == BENCH_SHA
        optimized = validate_benchmark(RUN/'candidate_benchmark')
        report = comparison(baseline, optimized)
        state('candidate_accuracy', performance=report)
        accuracy = ['evalscope', 'eval', '--model', 'minicpm', '--api-url',
                    'http://127.0.0.1:9031/v1/chat/completions', '--api-key', 'EMPTY',
                    '--eval-type', 'openai_api', '--datasets', 'math_500', '--dataset-args',
                    '{"math_500":{"dataset_id":"/workspace/evalscope-datasets/math_500","subset_list":["Level 3"]}}',
                    '--eval-batch-size', '8', '--timeout', '3600', '--generation-config',
                    '{"temperature":1.0,"top_p":0.95,"max_tokens":32768}',
                    '--work-dir', str(RUN/'accuracy'), '--ignore-errors']
        run_command(accuracy, RUN, 'accuracy', timeout=21600)
        state('evaluation_finished', accuracy_status='requires_report_and_error_count_review')
    except BaseException as exc:
        state('failed', error=repr(exc))
        traceback.print_exc()
        raise
    finally:
        stop(proc)
        lock.close()


if __name__ == '__main__':
    main()
