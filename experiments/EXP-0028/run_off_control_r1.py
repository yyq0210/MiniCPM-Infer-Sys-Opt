"""EXP28: one complete same-day top-p-off control; preserve the completed B."""
from pathlib import Path
from datetime import datetime, timedelta
import argparse, fcntl, hashlib, importlib.util, json, os, re, shutil, signal, socket, subprocess, sys, time, traceback

ROOT = Path('/workspace/LeanScale'); DOC = ROOT / 'MiniCPM-Infer-Sys-Opt'
EXP = DOC / 'experiments/EXP-0028'; ART = ROOT / 'artifacts/EXP-0028/r1'
PARENT = DOC / 'experiments/EXP-0020'; PREVIOUS = DOC / 'experiments/EXP-0027/r1'
WT = ROOT / 'worktrees/EXP-0027-r1'
COMMITS = {'vllm-plugin-FL': 'c50b5f0526e8112c72f1de918487300cf7376378', 'FlagGems': 'de958ddfc4deeb43d6c94bb8652a3d4d3bc905fe'}
BOUND = datetime.fromisoformat('2026-10-30T17:30:00+08:00')
def now(): return datetime.now().astimezone()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp'); tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n'); tmp.replace(p)
def git(p, *args): return subprocess.check_output(['git', '-C', str(p), *args], text=True, stderr=subprocess.STDOUT, timeout=120).strip()
def load(name, p):
    spec = importlib.util.spec_from_file_location(name, p); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
def step(action, **kw):
    with (EXP / 'steps.jsonl').open('a') as f: f.write(json.dumps(dict(time=now().isoformat(), action=action, **kw)) + '\n')
def publish(paths, message):
    sys.path.insert(0, str(DOC / 'experiments/EXP-0025')); from common import publish as fn
    return fn(paths, message)
def verify():
    for p, value in read(EXP / 'plan_r1.json')['source_hashes'].items(): assert sha(Path(p)) == value, p
    for name, commit in COMMITS.items():
        assert git(WT / name, 'rev-parse', 'HEAD') == commit and not git(WT / name, 'status', '--porcelain')
    for name, value in read(PREVIOUS / 'implementation_sources.json').items(): assert sha(WT / 'vllm-plugin-FL' / name) == value
    reference = read(EXP / 'reference_B.json'); folder = Path(reference['run_directory']) / 'candidate_benchmark'
    assert sha(next((folder / 'benchmark_results').glob('raw_runs_*.csv'))) == reference['raw_sha256']
    assert sha(next((folder / 'benchmark_results').glob('summary_*.csv'))) == reference['summary_sha256']
    assert sha(PREVIOUS / 'review_e2e_20261030/audit.json') == reference['audit_sha256']

def prepare():
    assert now().date() == BOUND.date() and now() + timedelta(minutes=85) <= BOUND, 'No full fresh control window'
    assert (EXP / 'README.md').exists() and not ART.exists() and not (EXP / 'run_off_control_r1.py').exists()
    assert read(ROOT / 'artifacts/EXP-0027-r1/e2e_review_publication.json')['remote_verified']
    assert read(PREVIOUS / 'review_e2e_20261030/audit.json')['status'] == 'passed'
    sys.path.insert(0, str(PARENT)); sys.path.insert(0, str(DOC / 'experiments/EXP-0025'))
    imported_core = load('exp28_r1_preflight_core', PARENT / 'run_split_e2e.py')
    imported_layout = load('exp28_r1_preflight_layout', DOC / 'experiments/EXP-0025/layout_checks.py')
    assert callable(imported_core.launch) and callable(imported_layout.layout_route)
    write(EXP / 'imports_preflight_r1.json', dict(status='passed', checked_at=now().isoformat(), gpu_work_started=False,
        modules=[str(PARENT / 'run_split_e2e.py'),str(DOC / 'experiments/EXP-0025/layout_checks.py'),str(DOC / 'experiments/EXP-0025/common.py')]))
    ART.mkdir(); shutil.copy2(__file__, EXP / 'run_off_control_r1.py')
    prior_sources = read(PREVIOUS / 'service_plan_r5.json')['source_hashes']
    for p, value in prior_sources.items(): assert sha(Path(p)) == value
    sources = dict(prior_sources); sources[str(EXP / 'run_off_control_r1.py')] = sha(EXP / 'run_off_control_r1.py')
    write(EXP / 'plan_r1.json', dict(source_hashes=sources, commits=COMMITS, case=[4096, 1024, 64, 256],
        arm='A: same code, top-p flag off', order='B already completed; append A after observing B',
        max_compute_minutes=70, save_margin_minutes=10, hard_save_bound=BOUND.isoformat(), automatic_selection=False))
    verify(); step('Prepare full same-day off control after registered protocol', commits=COMMITS)
    pub = publish(['experiments/EXP-0028'], 'experiments: prepare fixed-code same-day top-p off control')
    current = now(); assert current.date() == BOUND.date() and current + timedelta(minutes=80) <= BOUND
    end = current + timedelta(minutes=70); run = ART / current.strftime('%Y%m%d_%H%M%S_official4k_off')
    window = dict(created_at=current.isoformat(), compute_stop_at=end.isoformat(), save_target_at=(end + timedelta(minutes=10)).isoformat(), hard_save_bound=BOUND.isoformat())
    write(ART / 'run_window.json', window); write(EXP / 'run_window_r1_at_launch.json', window)
    pub = publish(['experiments/EXP-0028/run_window_r1_at_launch.json'], 'experiments: allocate fresh same-day off-control window')
    assert (end - now()).total_seconds() > 65 * 60
    with (ART / 'controller.log').open('w') as log:
        proc = subprocess.Popen(['python3', str(EXP / 'run_off_control_r1.py'), '--run', str(run)], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    receipt = dict(pid=proc.pid, process_start_ticks=(Path('/proc') / str(proc.pid) / 'stat').read_text().rsplit(') ', 1)[1].split()[19], run_directory=str(run), window=window, preparation_publication=pub)
    write(ART / 'launch.json', receipt)
    active = read(ROOT / 'artifacts/active_inference_plan.json'); active.update(current_experiment='EXP-0028', status='Same-day fixed-code top-p off control running', launch_receipt=str(ART / 'launch.json'))
    write(ROOT / 'artifacts/active_inference_plan.json', active)
    print(json.dumps(receipt), flush=True)

def execute(run):
    assert run.parent == ART; run.mkdir(exist_ok=False)
    end = datetime.fromisoformat(read(ART / 'run_window.json')['compute_stop_at'])
    assert now().date() == end.date() and now() < end
    sys.path.insert(0, str(PARENT)); core = load('exp28_core', PARENT / 'run_split_e2e.py')
    sys.path.insert(0, str(DOC / 'experiments/EXP-0025'))
    layout = load('exp28_layout', DOC / 'experiments/EXP-0025/layout_checks.py')
    core.EXP = EXP; core.RUN = run; core.WORKTREES = WT; core.CANDIDATE_COMMITS = COMMITS
    original_env = core.env
    def environment(candidate):
        result = original_env(candidate)
        result.update(VLLM_FL_TOPP_ROW_PARALLEL='0', VLLM_FL_TOPP_ROW_PARALLEL_SHADOW='0',
                      VLLM_FL_TOPP_ROW_PARALLEL_SHADOW_ARM_FILE='', VLLM_FL_MLP_WEIGHT_LAYOUT='0', TORCHINDUCTOR_CACHE_DIR=str(run / 'inductor_cache'))
        return result
    core.env = environment
    core.STATE = dict(started_at=now().isoformat(), pid=os.getpid(), commits=COMMITS, arm='A_off', planned_finish_before=end.isoformat())
    reference = Path(read(EXP / 'reference_B.json')['run_directory']) / 'candidate_benchmark'
    folder = run / 'candidate_benchmark'; server = None
    lock = Path('/workspace/experiments/codex_20261019/runner.lock').open('a')
    def abort(*args): raise TimeoutError('EXP28 bounded window or termination signal')
    signal.signal(signal.SIGTERM, abort); signal.signal(signal.SIGALRM, abort)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.setitimer(signal.ITIMER_REAL, (end - now()).total_seconds())
        verify()
        for port in [9031, 9032]:
            with socket.socket() as sock: assert sock.connect_ex(('127.0.0.1', port)) != 0
        idle = []
        for _ in range(2):
            text = subprocess.check_output(['/usr/local/corex/bin/ixsmi'], text=True, timeout=20)
            m = re.search(r'(\d+)MiB\s*/\s*(\d+)MiB\s*\|\s*(\d+)%', text)
            assert m and int(m[1]) <= 1024 and int(m[3]) <= 5
            idle.append(dict(memory_mib=int(m[1]), util_percent=int(m[3]))); time.sleep(3)
        write(run / 'idle.json', idle); core.verify_installed_environment()
        environment_values = environment(True)
        write(run / 'explicit_environment.json', {k: environment_values[k] for k in ['PYTHONPATH', 'VLLM_PLUGINS', 'VLLM_CACHE_ROOT', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'VLLM_FL_TOPP_ROW_PARALLEL', 'VLLM_FL_TOPP_ROW_PARALLEL_SHADOW', 'VLLM_FL_TOPP_ROW_PARALLEL_SHADOW_ARM_FILE', 'VLLM_FL_MLP_WEIGHT_LAYOUT']})
        gate = "import vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar; import vllm.v1.sample.ops.topk_topp_triton as b; import vllm.v1.sample.ops.topk_topp_sampler as s; from vllm_fl.patches import topp_row_parallel as p; original=b.apply_top_k_top_p_triton; assert not p.install_topp_row_parallel(); assert b.apply_top_k_top_p_triton is original and s.apply_top_k_top_p_triton is original; assert not getattr(original, '_leanscale_topp_row_parallel', False); print('ORIGINAL_TOPP_ALIASES_CONFIRMED')"
        core.run_command(['python3', '-c', gate], folder, 'disabled_gate', timeout=180)
        assert 'ORIGINAL_TOPP_ALIASES_CONFIRMED' in (folder / 'disabled_gate.log').read_text()
        core.state('starting_off_control'); step('Start original top-p off control', run_name=run.name)
        server = core.launch(folder, True); core.smoke(folder)
        core.verify_route(folder); layout.layout_route(run, folder, reference, 'A')
        log = (folder / 'server.log').read_text()
        assert 'LeanScale top-p grid hook installed:' not in log
        write(run / 'workload_start.json', dict(started_at=now().isoformat(), server_log_character_offset=len(log)))
        core.metrics(folder, 'metrics_before.prom'); core.state('running_official4k_off')
        core.run_command(core.BENCH[:-1] + ['[[4096,1024,64,256]]'], folder, 'benchmark', timeout=(end - now()).total_seconds())
        before = core.validate_benchmark(folder, {(4096, 1024, 64, 256)})
        after = core.validate_benchmark(reference, {(4096, 1024, 64, 256)})
        assert re.findall(r'Failed requests:\s+(\d+)', (folder / 'benchmark.log').read_text()) == ['0'] * 4
        comparison = core.comparison(before, after)  # A(off) baseline, B(on) candidate.
        core.metrics(folder, 'metrics_after.prom'); core.compare_schedule(folder, reference, run / 'scheduler_comparison.json')
        core.verify_route(folder); layout.layout_route(run, folder, reference, 'A')
        verify(); core.verify_installed_environment()
        log = (folder / 'server.log').read_text()
        for forbidden in ['LeanScale top-p grid hook installed:', 'LeanScale top-p grid active:', 'LeanScale top-p shadow passed:']: assert forbidden not in log
        kernels = []
        for p in (run / 'triton_cache').glob('*/_topk_topp_kernel.json'):
            data = read(p); ir = p.with_suffix('.ttir'); assert ir.exists() and data['target']['backend'] == 'corex'
            kernels.append(dict(path=str(p.relative_to(run)), sha256=sha(p), ttir_sha256=sha(ir), num_warps=data['num_warps'], num_stages=data['num_stages'], shared=data.get('shared')))
        assert kernels
        write(folder / 'original_topp_route_evidence.json', dict(status='passed', aliases_gate_passed=True, candidate_hook_absent=True, kernels=kernels, original_launcher_source_sha256=sha(Path('/usr/local/lib/python3.12/site-packages/vllm/v1/sample/ops/topk_topp_triton.py'))))
        core.state('complete_official4k_off', comparison_B_on_vs_A_off=comparison, selection_changed=False)
        step('Complete same-day off control; independent audit pending', run_name=run.name)
    except BaseException:
        core.state('failed', error=traceback.format_exc()); step('Off-control failure retained', error=traceback.format_exc()); traceback.print_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try: core.stop(server)
        finally:
            core.state(core.STATE['stage'], owned_children_reaped=not core.ready(), finished_at=now().isoformat()); lock.close()
    archive = EXP / 'artifacts' / run.name; archive.mkdir(parents=True, exist_ok=False); manifest = {}
    files = [p for p in run.iterdir() if p.is_file() and p.suffix in ['.json', '.jsonl']]
    files += [p for p in folder.iterdir() if p.is_file() and p.suffix in ['.json', '.log', '.prom']] if folder.exists() else []
    files += list((folder / 'benchmark_results').glob('*.csv'))
    for p in files:
        rel = p.relative_to(run); target = archive / rel; target.parent.mkdir(parents=True, exist_ok=True)
        text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[IP_REDACTED]', p.read_text(errors='replace'))
        text = re.sub(r'uuid=[^,)]+', 'uuid=[REDACTED]', text)
        target.write_text('\n'.join(line.rstrip() for line in text.splitlines()) + '\n')
        manifest[str(rel)] = dict(raw_sha256=sha(p), archived_sha256=sha(target))
    write(archive / 'artifact_manifest.json', manifest); write(ART / 'status.json', core.STATE)
    try: pub = publish(['experiments/EXP-0028/artifacts/' + run.name, 'experiments/EXP-0028/STATUS.json', 'experiments/EXP-0028/steps.jsonl'], 'experiments: archive same-day top-p off control')
    except Exception as exc: pub = dict(status='push_failed_local_saved', error=repr(exc))
    write(run / 'publication.json', pub); write(ART / 'publication.json', pub)
    print(json.dumps(dict(status=core.STATE, publication=pub)), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path); args = parser.parse_args()
    if args.run:
        try:
            execute(args.run)
        except BaseException:
            failure = dict(stage='controller_exception', time=now().isoformat(), error=traceback.format_exc())
            write(ART / 'controller_failure.json', failure)
            if not (args.run / 'status.json').exists():
                write(args.run / 'status.json', failure)
            dest = EXP / 'controller_failures' / args.run.name
            dest.mkdir(parents=True, exist_ok=True)
            write(dest / 'failure.json', failure)
            step('Unhandled controller exception preserved', failure=failure)
            try:
                result = publish(['experiments/EXP-0028/controller_failures/' + args.run.name, 'experiments/EXP-0028/steps.jsonl'],
                                 'experiments: preserve early off-control exception')
                write(ART / 'controller_failure_publication.json', result)
            except Exception:
                traceback.print_exc()
            raise
    else:
        prepare()
