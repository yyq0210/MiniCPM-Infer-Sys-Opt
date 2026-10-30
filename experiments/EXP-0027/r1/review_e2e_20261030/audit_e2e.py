"""Independent CPU-only audit of EXP27 diagnostic + official four-round 4K."""
from pathlib import Path
from collections import Counter
from datetime import datetime
import ast, csv, hashlib, json, re, statistics, subprocess, sys

ROOT = Path('/workspace/LeanScale')
DOC = ROOT / 'MiniCPM-Infer-Sys-Opt'
EXP = DOC / 'experiments/EXP-0027/r1'
ART = ROOT / 'artifacts/EXP-0027-r1/service_r5'
COMMITS = {'vllm-plugin-FL': 'c50b5f0526e8112c72f1de918487300cf7376378',
           'FlagGems': 'de958ddfc4deeb43d6c94bb8652a3d4d3bc905fe'}
BENCH_SHA = '0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565'

def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open() as f: return list(csv.DictReader(f))
def git(p, *args): return subprocess.check_output(['git', '-C', str(p), *args], text=True, timeout=120).strip()

def route_verifier():
    # Reuse the independently reviewed AST/TTIR checks, without running its main.
    p = DOC / 'experiments/EXP-0025/review_20261029/audit_abba.py'
    assert sha(p) == '001bd5e695817fa1d545e6af507e27efa758462e14f60e1f93b1451739260d3c'
    tree = ast.parse(p.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'inspect_route')
    scope = dict(globals())
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(p), 'exec'), scope)
    return scope['inspect_route'], sha(p)

def inspect(run, phase, inspect_route):
    status = read(run / 'status.json')
    assert status['stage'] == 'complete_' + phase and status['owned_children_reaped']
    assert status['commits'] == COMMITS
    assert datetime.fromisoformat(status['finished_at']) < datetime.fromisoformat(status['planned_finish_before'])
    publication = read(run / 'publication.json'); assert publication['remote_verified']
    archive = EXP / 'service_artifacts_r5' / run.name
    manifest = read(archive / 'artifact_manifest.json')
    for name, value in manifest.items():
        assert sha(run / name) == value['raw_sha256'], name
        assert sha(archive / name) == value['archived_sha256'], name
    folder = run / 'candidate_benchmark'
    server = read(folder / 'server.command.json')
    assert json.loads(server[server.index('--compilation-config') + 1]) == {'cudagraph_mode': 'FULL_DECODE_ONLY'}
    assert server[server.index('--gpu-memory-utilization') + 1] == '0.85'
    assert server[server.index('--max-model-len') + 1] == '131072'
    environment = read(run / 'explicit_environment.json')
    assert environment['VLLM_FL_TOPP_ROW_PARALLEL'] == '1'
    assert environment['VLLM_FL_TOPP_ROW_PARALLEL_SHADOW'] == ('1' if phase == 'shadow' else '0')
    assert environment['VLLM_FL_TOPP_ROW_PARALLEL_SHADOW_ARM_FILE'] == (str(run / 'shadow.arm') if phase == 'shadow' else '')
    for key, name in [('VLLM_CACHE_ROOT', 'candidate_cache'), ('TRITON_CACHE_DIR', 'triton_cache'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor_cache')]:
        assert environment[key] == str(run / name)
    client = read(folder / 'client_environment.json')
    assert client['VLLM_FL_TOPP_ROW_PARALLEL_SHADOW'] == '0' and client['VLLM_FL_TOPP_ROW_PARALLEL_SHADOW_ARM_FILE'] == ''
    installed = read(run / 'environment_check.json')
    assert installed['benchmark_sha256'] == BENCH_SHA and installed['candidate_commits'] == COMMITS
    assert installed['versions'] == {'torch': '2.10.0+corex.4.5.0', 'triton': '3.6.0', 'vllm': '0.24.0+empty'}
    for name, value in installed['installed_source_hashes'].items():
        assert sha(Path('/usr/local/lib/python3.12/site-packages') / name) == value
    original_route = inspect_route(run, 'A')
    evidence = read(folder / 'topp_route_evidence.json')
    assert evidence['status'] == 'passed' and evidence['phase'] == phase
    assert evidence['unchanged_original_kernel'] and evidence['maximum_scratch_payload_bytes'] == 33423360
    log = (folder / 'server.log').read_text()
    start = read(run / 'workload_start.json')
    tail = log[start['server_log_character_offset']:]
    events = re.findall(r'LeanScale top-p grid active: B=(\d+) V=(\d+) programs=(\d+) scratch_rows=(\d+) scratch_bytes=(\d+) k=None calls=(\d+) shadow=(True|False)', tail)
    assert [list(x) for x in events] == evidence['grid_events']
    assert any(int(x[0]) == 64 for x in events)
    for b, v, programs, count, size, calls, shadow in events:
        assert 17 <= int(b) <= 64 and int(b) == int(programs) and int(v) == 130560
        assert int(count) == 64 and int(size) == 33423360
        assert shadow == ('True' if phase == 'shadow' else 'False')
    assert max(int(x[5]) for x in events) >= 1024
    assert evidence['kernels']
    for k in evidence['kernels']:
        p = run / k['path']; assert p.resolve().is_relative_to(run)
        assert sha(p) == k['sha256'] and sha(p.with_suffix('.ttir')) == k['ttir_sha256']
        data = read(p)
        assert data['target']['backend'] == 'corex'
        assert data['num_warps'] == k['num_warps'] and data['num_stages'] == k['num_stages'] and data.get('shared') == k['shared']
    result = dict(run_name=run.name, started_at=status['started_at'], finished_at=status['finished_at'],
                  archive_files_checked=len(manifest), publication=publication, top_p_variants=len(evidence['kernels']),
                  top_p_grid_events=len(events), max_logged_candidate_calls=max(int(x[5]) for x in events), **original_route)
    if phase == 'shadow':
        arm = read(run / 'arm_evidence.json')
        assert arm['warmup_shadow_checks'] == 0 and arm['before_workload_dispatch']
        assert datetime.fromisoformat(arm['armed_at']) >= datetime.fromisoformat(start['started_at'])
        assert 'LeanScale top-p shadow passed:' not in log[:start['server_log_character_offset']]
        checks = re.findall(r'LeanScale top-p shadow passed: B=(\d+) V=(\d+) stride=(.*?) p_min=([^ ]+) p_max=([^ ]+) exact_mask=True rng_unchanged=True check=(\d+)', tail)
        stored = read(run / 'shadow_evidence.json')
        assert [list(x) for x in checks] == stored['checks_during_workload']
        assert stored['status'] == 'passed' and stored['successful_requests'] == 256 and stored['failed_requests'] == 0
        assert len(checks) == 96 and Counter(int(x[0]) for x in checks) == Counter({b: 2 for b in range(17, 65)})
        assert [int(x[5]) for x in checks] == list(range(1, 97))
        assert all(int(x[1]) == 130560 and abs(float(x[3]) - .95) < 1e-6 and abs(float(x[4]) - .95) < 1e-6 for x in checks)
        text = (folder / 'diagnostic.log').read_text()
        assert re.findall(r'Successful requests:\s+(\d+)', text) == ['256'] and re.findall(r'Failed requests:\s+(\d+)', text) == ['0']
        assert re.findall(r'Total input tokens:\s+(\d+)', text) == ['1048576'] and re.findall(r'Total generated tokens:\s+(\d+)', text) == ['262144']
        result.update(successful_requests=256, failed_requests=0, exact_mask_rng_checks=96, diagnostics_excluded_from_performance=True)
    else:
        assert 'LeanScale top-p shadow passed:' not in log
        result['shadow_checks'] = 0
    return result

def counter(p, key):
    values = [float(line.rsplit(' ', 1)[1]) for line in p.read_text().splitlines() if line.startswith('vllm:' + key + '{') or line.startswith('vllm:' + key + ' ')]
    assert values, key
    return sum(values)

def audit():
    flow = read(ART / 'status.json'); assert flow['stage'] == 'complete' and not flow['selection_changed']
    assert [x['phase'] for x in flow['completed_stages']] == ['shadow', 'official4k']
    assert read(ART / 'publication.json')['remote_verified']
    assert read(EXP / 'candidate_commits.json') == COMMITS
    for name, commit in COMMITS.items():
        p = ROOT / 'worktrees/EXP-0027-r1' / name
        assert git(p, 'rev-parse', 'HEAD') == commit and not git(p, 'status', '--porcelain')
    for name, value in read(EXP / 'implementation_sources.json').items():
        assert sha(ROOT / 'worktrees/EXP-0027-r1/vllm-plugin-FL' / name) == value
    for path, value in read(EXP / 'service_plan_r5.json')['source_hashes'].items():
        assert sha(Path(path)) == value
    assert sha(Path('/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py')) == BENCH_SHA
    verifier, verifier_sha = route_verifier()
    shadow_run, official_run = [Path(x['run_directory']) for x in flow['completed_stages']]
    diagnostic = inspect(shadow_run, 'shadow', verifier)
    official = inspect(official_run, 'official4k', verifier)
    folder = official_run / 'candidate_benchmark'
    raw_path = next((folder / 'benchmark_results').glob('raw_runs_*.csv'))
    summary_path = next((folder / 'benchmark_results').glob('summary_*.csv'))
    raw, summary = rows(raw_path), rows(summary_path)
    assert len(raw) == 4 and len(summary) == 1
    for r in raw:
        assert tuple(int(r[k]) for k in ['Prefill', 'Decode', 'Conc', 'Num Prompts']) == (4096, 1024, 64, 256)
        assert r['Run Status'] == 'SUCCESS' and float(r['Successful Requests']) == 256
        assert float(r['Total Input Tokens']) == 1048576 and float(r['Total Output Tokens']) == 262144
    for k, value in summary[0].items():
        if k not in ['Prefill', 'Decode', 'Conc', 'Num Prompts']:
            assert abs(float(value) - statistics.mean(float(r[k]) for r in raw[1:])) <= .011, (k, value)
    command = read(folder / 'benchmark.command.json')
    assert json.loads(command[command.index('--test-cases') + 1]) == [[4096, 1024, 64, 256]]
    log = (folder / 'benchmark.log').read_text()
    assert re.findall(r'Successful requests:\s+(\d+)', log) == ['256'] * 4
    assert re.findall(r'Failed requests:\s+(\d+)', log) == ['0'] * 4
    retained = [float(r['Total tok/s']) for r in raw[1:]]
    cv = statistics.stdev(retained) / statistics.mean(retained) * 100
    reference = read(official_run / 'reference.json'); ref = Path(reference['source'])
    ref_raw = next((ref / 'benchmark_results').glob('raw_runs_*.csv'))
    ref_summary = next((ref / 'benchmark_results').glob('summary_*.csv'))
    assert sha(ref_raw) == reference['raw_sha256'] and sha(ref_summary) == reference['summary_sha256']
    old = rows(ref_summary)[0]
    metrics = {k: dict(reference=float(old[k]), candidate=float(summary[0][k]), change_percent=(float(summary[0][k]) / float(old[k]) - 1) * 100) for k in ['Total tok/s', 'Output tok/s', 'Mean TTFT (ms)', 'Mean TPOT (ms)', 'Mean ITL (ms)']}
    comparison = read(official_run / 'comparison.json')[0]
    assert comparison['candidate_retained_runs'] == retained and abs(comparison['candidate_cv_percent'] - cv) < 1e-9
    assert abs(comparison['throughput_change_percent'] - metrics['Total tok/s']['change_percent']) < 1e-9
    assert abs(comparison['ttft_change_percent'] - metrics['Mean TTFT (ms)']['change_percent']) < 1e-9
    delta = {key: counter(folder / 'metrics_after.prom', key) - counter(folder / 'metrics_before.prom', key) for key in ['num_preemptions_total', 'request_queue_time_seconds_count', 'request_queue_time_seconds_sum']}
    assert delta['request_queue_time_seconds_count'] == 1024
    schedule = read(official_run / 'scheduler_comparison.json')['candidate']
    assert schedule['counter_delta'] == delta
    queue = delta['request_queue_time_seconds_sum'] / 1024
    assert abs(schedule['mean_queue_seconds_all_four_rounds'] - queue) < 1e-9
    peaks = [float(x) for x in re.findall(r'GPU KV cache usage: ([\d.]+)%', (folder / 'server.log').read_text())]
    assert peaks
    return dict(status='passed', checked_at=datetime.now().astimezone().isoformat(), selected_experiment='EXP-0020',
                source_commits=COMMITS, inherited_ast_ttir_verifier_sha256=verifier_sha,
                diagnostic=diagnostic, official=official, raw_sha256=sha(raw_path), summary_sha256=sha(summary_path),
                all_four_successful_requests=1024, all_four_failed_requests=0,
                all_four_total_tokens_s=[float(r['Total tok/s']) for r in raw], retained_total_tokens_s=retained,
                retained_cv_percent=cv, summary=summary[0], metrics_against_previous_day_A2=metrics,
                reference_scope='Previous-day EXP25 A2; no fresh same-day baseline in this flow',
                change_vs_historical_exp20_percent=(float(summary[0]['Total tok/s']) / 3579.12 - 1) * 100,
                scheduling=dict(counter_delta=delta, mean_queue_seconds_all_four_rounds=queue, sampled_peak_kv_percent=max(peaks)),
                selection_changed=False, full_model_task_accuracy='deferred_by_user')

if __name__ == '__main__':
    result = audit()
    out = Path(sys.argv[1]); assert not out.exists()
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, ensure_ascii=False))
