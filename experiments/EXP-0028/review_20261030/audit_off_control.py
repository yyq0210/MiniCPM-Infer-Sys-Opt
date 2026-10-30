"""CPU-only independent audit of the fixed-code top-p off control and fixed prior B."""
from pathlib import Path
from collections import Counter
from datetime import datetime
import ast, csv, hashlib, json, re, statistics, subprocess, sys

ROOT = Path('/workspace/LeanScale')
DOC = ROOT / 'MiniCPM-Infer-Sys-Opt'
EXP = DOC / 'experiments/EXP-0028'
ART = ROOT / 'artifacts/EXP-0028/r1'
COMMITS = {'vllm-plugin-FL': 'c50b5f0526e8112c72f1de918487300cf7376378',
           'FlagGems': 'de958ddfc4deeb43d6c94bb8652a3d4d3bc905fe'}
BENCH_SHA = '0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565'

def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open() as f: return list(csv.DictReader(f))
def git(p, *args):
    return subprocess.check_output(['git', '-C', str(p), *args], text=True, timeout=120).strip()
def counter(p, key):
    values = [float(line.rsplit(' ', 1)[1]) for line in p.read_text().splitlines()
              if line.startswith('vllm:' + key + '{') or line.startswith('vllm:' + key + ' ')]
    assert values, key
    return sum(values)

def route_verifier():
    p = DOC / 'experiments/EXP-0025/review_20261029/audit_abba.py'
    assert sha(p) == '001bd5e695817fa1d545e6af507e27efa758462e14f60e1f93b1451739260d3c'
    node = next(n for n in ast.parse(p.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'inspect_route')
    scope = dict(globals())
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(p), 'exec'), scope)
    return scope['inspect_route'], sha(p)

def benchmark(run):
    folder = run / 'candidate_benchmark'
    raw_paths = list((folder / 'benchmark_results').glob('raw_runs_*.csv'))
    summary_paths = list((folder / 'benchmark_results').glob('summary_*.csv'))
    assert len(raw_paths) == len(summary_paths) == 1
    raw, summary = rows(raw_paths[0]), rows(summary_paths[0])
    assert len(raw) == 4 and len(summary) == 1
    for r in raw:
        assert tuple(int(r[k]) for k in ['Prefill', 'Decode', 'Conc', 'Num Prompts']) == (4096,1024,64,256)
        assert r['Run Status'] == 'SUCCESS' and float(r['Successful Requests']) == 256
        assert float(r['Total Input Tokens']) == 1048576 and float(r['Total Output Tokens']) == 262144
    for k, value in summary[0].items():
        if k not in ['Prefill', 'Decode', 'Conc', 'Num Prompts']:
            assert abs(float(value) - statistics.mean(float(r[k]) for r in raw[1:])) <= .011, (k, value)
    command = read(folder / 'benchmark.command.json')
    assert command[1] == '/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py'
    assert json.loads(command[command.index('--test-cases') + 1]) == [[4096,1024,64,256]]
    log = (folder / 'benchmark.log').read_text()
    for label, expected in [('Successful requests',256),('Failed requests',0),
                            ('Total input tokens',1048576),('Total generated tokens',262144)]:
        assert re.findall(label + r':\s+(\d+)', log) == [str(expected)] * 4, label
    retained = [float(r['Total tok/s']) for r in raw[1:]]
    return dict(raw_sha256=sha(raw_paths[0]), summary_sha256=sha(summary_paths[0]), summary=summary[0],
                all_four_total_tokens_s=[float(r['Total tok/s']) for r in raw],
                retained_total_tokens_s=retained,
                retained_cv_percent=statistics.stdev(retained)/statistics.mean(retained)*100,
                retained_first_to_last_change_percent=(retained[-1]/retained[0]-1)*100,
                successful_requests=1024, failed_requests=0)

def schedule(run):
    folder = run / 'candidate_benchmark'
    delta = {key: counter(folder/'metrics_after.prom',key)-counter(folder/'metrics_before.prom',key)
             for key in ['num_preemptions_total','request_queue_time_seconds_count','request_queue_time_seconds_sum']}
    assert delta['request_queue_time_seconds_count'] == 1024
    peaks = [float(x) for x in re.findall(r'GPU KV cache usage: ([\d.]+)%', (folder/'server.log').read_text())]
    assert peaks
    return dict(counter_delta=delta, mean_queue_seconds_all_four_rounds=delta['request_queue_time_seconds_sum']/1024,
                sampled_peak_kv_percent=max(peaks))

def audit():
    launch = read(ART / 'launch.json'); run = Path(launch['run_directory'])
    assert run.parent == ART
    status = read(run/'status.json')
    assert status == read(ART/'status.json')
    assert status['stage'] == 'complete_official4k_off' and status['owned_children_reaped']
    assert status['commits'] == COMMITS and not status['selection_changed']
    assert datetime.fromisoformat(status['finished_at']) < datetime.fromisoformat(launch['window']['compute_stop_at'])
    pub = read(run/'publication.json')
    assert pub == read(ART/'publication.json') and pub['remote_verified']
    assert read(EXP/'imports_preflight_r1.json')['status'] == 'passed'
    assert sha(EXP/'run_off_control_r1.py') == '40c430bf140336cb8edc703e6a9dd3cd3d49e98a6ddbabcc0b0788bd9c3742f2'
    plan = read(EXP/'plan_r1.json')
    for name, value in plan['source_hashes'].items(): assert sha(Path(name)) == value, name
    for name, commit in COMMITS.items():
        p = ROOT/'worktrees/EXP-0027-r1'/name
        assert git(p,'rev-parse','HEAD') == commit and not git(p,'status','--porcelain')
    assert sha(Path('/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py')) == BENCH_SHA
    archive = EXP/'artifacts'/run.name
    manifest = read(archive/'artifact_manifest.json')
    required = ['status.json','explicit_environment.json','environment_check.json',
                'candidate_benchmark/disabled_gate.log','candidate_benchmark/server.log',
                'candidate_benchmark/benchmark.log','candidate_benchmark/original_topp_route_evidence.json']
    assert set(required) <= set(manifest)
    for name, value in manifest.items():
        assert sha(run/name) == value['raw_sha256'], name
        assert sha(archive/name) == value['archived_sha256'], name
    folder = run/'candidate_benchmark'
    server = read(folder/'server.command.json')
    assert json.loads(server[server.index('--compilation-config')+1]) == {'cudagraph_mode':'FULL_DECODE_ONLY'}
    assert server[server.index('--gpu-memory-utilization')+1] == '0.85'
    assert server[server.index('--max-model-len')+1] == '131072'
    env = read(run/'explicit_environment.json')
    for key in ['VLLM_FL_TOPP_ROW_PARALLEL','VLLM_FL_TOPP_ROW_PARALLEL_SHADOW','VLLM_FL_MLP_WEIGHT_LAYOUT']:
        assert env[key] == '0', key
    assert env['VLLM_FL_TOPP_ROW_PARALLEL_SHADOW_ARM_FILE'] == ''
    for key, name in [('VLLM_CACHE_ROOT','candidate_cache'),('TRITON_CACHE_DIR','triton_cache'),('TORCHINDUCTOR_CACHE_DIR','inductor_cache')]:
        assert env[key] == str(run/name)
    gate = (folder/'disabled_gate.log').read_text()
    assert 'ORIGINAL_TOPP_ALIASES_CONFIRMED' in gate
    gate_command = read(folder/'disabled_gate.command.json')
    assert gate_command[:2] == ['python3','-c']
    assert "assert not p.install_topp_row_parallel()" in gate_command[2]
    assert "s.apply_top_k_top_p_triton is original" in gate_command[2]
    installed = read(run/'environment_check.json')
    assert installed['benchmark_sha256'] == BENCH_SHA and installed['candidate_commits'] == COMMITS
    assert installed['versions'] == {'torch':'2.10.0+corex.4.5.0','triton':'3.6.0','vllm':'0.24.0+empty'}
    for name, value in installed['installed_source_hashes'].items():
        assert sha(Path('/usr/local/lib/python3.12/site-packages')/name) == value
    verifier, verifier_sha = route_verifier()
    route = verifier(run, 'A')
    evidence = read(folder/'original_topp_route_evidence.json')
    assert evidence['status'] == 'passed' and evidence['aliases_gate_passed'] and evidence['candidate_hook_absent']
    original_sha = '64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8'
    assert evidence['original_launcher_source_sha256'] == original_sha
    assert sha(Path('/usr/local/lib/python3.12/site-packages/vllm/v1/sample/ops/topk_topp_triton.py')) == original_sha
    log = (folder/'server.log').read_text()
    for forbidden in ['LeanScale top-p grid hook installed:', 'LeanScale top-p grid active:', 'LeanScale top-p shadow passed:']:
        assert forbidden not in log
    assert evidence['kernels']
    for k in evidence['kernels']:
        p = run/k['path']; assert p.resolve().is_relative_to(run)
        assert sha(p) == k['sha256'] and sha(p.with_suffix('.ttir')) == k['ttir_sha256']
        data = read(p); assert data['target']['backend'] == 'corex'
        assert all(data[key] == k[key] for key in ['num_warps','num_stages','shared'])
    A = benchmark(run); A['schedule'] = schedule(run)
    A.update(run_name=run.name, started_at=status['started_at'],finished_at=status['finished_at'],
             archive_files_checked=len(manifest), publication=pub, route=route,
             top_p_variants=len(evidence['kernels']), candidate_hook=False, shadow_checks=0)
    reference = read(EXP/'reference_B.json')
    previous_audit_path = DOC/'experiments/EXP-0027/r1/review_e2e_20261030/audit.json'
    assert sha(previous_audit_path) == reference['audit_sha256']
    previous_audit = read(previous_audit_path); assert previous_audit['status'] == 'passed'
    b_run = Path(reference['run_directory']); B = benchmark(b_run); B['schedule'] = schedule(b_run)
    assert B['raw_sha256'] == reference['raw_sha256'] and B['summary_sha256'] == reference['summary_sha256']
    assert B['summary'] == reference['summary'] == previous_audit['summary']
    assert B['all_four_total_tokens_s'] == previous_audit['all_four_total_tokens_s']
    assert B['schedule'] == previous_audit['scheduling']
    B.update(run_name=b_run.name, started_at=previous_audit['official']['started_at'],
             finished_at=previous_audit['official']['finished_at'], prior_audit_sha256=reference['audit_sha256'],
             kv_tokens=previous_audit['official']['kv_tokens'])
    metrics = {k:dict(A_off=float(A['summary'][k]), B_on=float(B['summary'][k]),
                     B_on_vs_A_off_percent=(float(B['summary'][k])/float(A['summary'][k])-1)*100)
               for k in ['Total tok/s','Output tok/s','Mean TTFT (ms)','Mean TPOT (ms)','Mean ITL (ms)']}
    comp = read(run/'comparison.json')[0]
    for arm, stat in [('baseline',A),('candidate',B)]:
        assert comp[arm+'_retained_runs'] == stat['retained_total_tokens_s']
        assert abs(comp[arm+'_cv_percent'] - stat['retained_cv_percent']) < 1e-9
        assert comp[arm+'_total_tokens_s'] == float(stat['summary']['Total tok/s'])
    assert abs(comp['throughput_change_percent']-metrics['Total tok/s']['B_on_vs_A_off_percent']) < 1e-9
    assert abs(comp['ttft_change_percent']-metrics['Mean TTFT (ms)']['B_on_vs_A_off_percent']) < 1e-9
    schedules = read(run/'scheduler_comparison.json')
    for key, stat in [('reference',A),('candidate',B)]:
        saved = schedules[key]; actual = stat['schedule']
        assert saved['counter_delta'] == actual['counter_delta']
        assert abs(saved['mean_queue_seconds_all_four_rounds']-actual['mean_queue_seconds_all_four_rounds']) < 1e-9
        assert saved['peak_cache_usage_percent_sampled'] == actual['sampled_peak_kv_percent']
    assert datetime.fromisoformat(B['finished_at']) < datetime.fromisoformat(A['started_at'])
    return dict(status='passed',checked_at=datetime.now().astimezone().isoformat(),source_commits=COMMITS,
                route_verifier_sha256=verifier_sha,A_off=A,B_on=B,metrics=metrics,
                retained_round_position_comparisons_percent=[(b/a-1)*100 for a,b in zip(A['retained_total_tokens_s'],B['retained_total_tokens_s'])],
                order='B then A, post hoc same-day consistency check; one session per arm; not randomized or ABBA',
                selection_changed=False,selected_experiment='EXP-0020',
                full_model_task_accuracy='deferred_by_user',
                interpretation='Positive same-day directional evidence; stable causal gain remains unconfirmed.')

if __name__ == '__main__':
    out = Path(sys.argv[1]); assert not out.exists()
    result = audit()
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,ensure_ascii=False))
