"""EXP18: isolated, bounded diagnostics; only terminate owned child groups."""
from datetime import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request

EXP = Path(__file__).resolve().parent
ROOT = Path('/workspace/LeanScale')
REPO = ROOT/'MiniCPM-Infer-Sys-Opt'
ART = ROOT/'artifacts'
WORK = ROOT/'worktrees/EXP-0018'
REQUEST = ART/'EXP-0018-request.json'
BENCH = Path('/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py')
BENCH_SHA = '0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565'
COMMITS = {'vllm-plugin-FL': '588a6147cd291e272324ab6c812024cd587c6287',
           'FlagGems': 'fd209df717648002ebf5ba4ec351424dcc997a53'}
INSTALLED = {'profiler/wrapper.py': '31f8a33afd23e67a2ade5a95d0d837a03b0a9164d5b18acf561d948c9ae6c06c',
             'config/profiler.py': '981161215276d72564567d368d05d086621c831b2d1da79c8a8cff6963b5a305',
             'benchmarks/serve.py': 'b09e85eed6f5f7a9075b40a19fc1dd8e94a1c449665c2c6510f349e0409e77b7'}
PORT = 9032
WINDOWS = ('initial', 'mixed', 'large_decode', 'tail')
MODES = ('FULL_DECODE_ONLY', 'NONE')
STATE, CHILDREN = {}, []
RUN = GPU_LOCK = None
DEADLINE = 0
PASS_BUDGET = 4800

def read(path):
    try: return json.loads(path.read_text())
    except FileNotFoundError: return None

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def atomic(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(path)

def state(stage, **kw):
    STATE.update(stage=stage, updated_at=datetime.now().astimezone().isoformat(), **kw)
    atomic(RUN/'status.json', STATE)
    atomic(REQUEST, STATE)
    with (RUN/'history.jsonl').open('a') as f: f.write(json.dumps(STATE, ensure_ascii=False)+'\n')
    print(json.dumps(STATE, ensure_ascii=False), flush=True)

def check_deadline():
    if time.monotonic() >= DEADLINE: raise TimeoutError('Dated computation cutoff reached')

def validate_window(policy, now):
    stop = datetime.fromisoformat(policy['compute_stop_at'])
    assert stop.tzinfo is not None
    assert stop.date() == now.date(), 'Require a fresh dated computation window'
    assert stop > now, 'Computation window expired'
    return (stop-now).total_seconds()

def choose_window(latest, captured, current, epoch):
    # No time-based fallbacks: retain missing coverage rather than mislabel it.
    if not latest.get('main_workload_started') or latest.get('window') != current:
        return None
    if epoch-latest.get('epoch', 0) > 3 or latest.get('profiling', True): return None
    if latest.get('window_profiled_steps', 0) < 16: return None
    if 'tail' not in captured and latest.get('new_requests_total') == 256 and 0 < latest.get('B', 0) <= 16:
        return 'tail'
    if 'mixed' not in captured and latest.get('phase') == 'mixed': return 'mixed'
    if 'large_decode' not in captured and latest.get('phase') == 'decode' and latest.get('B', 0) >= 48:
        return 'large_decode'
    return None

def gpu_snapshot():
    output = subprocess.check_output(['/usr/local/corex/bin/ixsmi'], text=True, timeout=20)
    m = re.search(r'(\d+)MiB\s*/\s*(\d+)MiB\s*\|\s*(\d+)%', output)
    assert m, 'Cannot verify GPU idle status'
    return dict(memory_used_mib=int(m[1]), memory_total_mib=int(m[2]), gpu_util_percent=int(m[3]),
                checked_at=datetime.now().astimezone().isoformat())

def port_free(port):
    with socket.socket() as s: return s.connect_ex(('127.0.0.1', port)) != 0

def acquire_idle():
    global GPU_LOCK
    # This job was explicitly queued only for an idle GPU. Do not silently
    # wait for hours and then launch outside the validated per-pass budget.
    handle = Path('/workspace/experiments/codex_20261019/runner.lock').open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        observations = [gpu_snapshot()]
        time.sleep(5)
        observations.append(gpu_snapshot())
        assert all(s['memory_used_mib'] <= 1024 and s['gpu_util_percent'] <= 5 for s in observations)
        assert port_free(9031) and port_free(PORT), 'An inference port is in use'
        atomic(RUN/'idle_evidence.json', observations)
        GPU_LOCK = handle
    except BaseException:
        handle.close()
        raise

def environment(folder):
    return dict(os.environ, VLLM_PLUGINS='fl', PYTHONUNBUFFERED='1',
                LEANSCALE_PROFILE_PASS=str(folder),
                PYTHONPATH=f'{EXP}:{WORK}/vllm-plugin-FL:{WORK}/FlagGems/src',
                VLLM_CACHE_ROOT=str(folder/'cache'), TRITON_CACHE_DIR=str(folder/'triton_cache'))

def start(argv, folder, name):
    check_deadline()
    folder.mkdir(parents=True, exist_ok=True)
    atomic(folder/(name+'.command.json'), argv)
    with (folder/(name+'.log')).open('w') as log:
        proc = subprocess.Popen(argv, cwd=folder, env=environment(folder), stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    CHILDREN.append(proc)
    return proc

def stop(proc):
    if proc.poll() is not None: return
    try: os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError: return
    try: proc.wait(timeout=40)
    except subprocess.TimeoutExpired:
        try: os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        proc.wait(timeout=20)

def wait_release():
    end = time.monotonic()+120
    while gpu_snapshot()['memory_used_mib'] > 1024:
        if time.monotonic() >= end: raise RuntimeError('GPU memory did not release after owned children stopped')
        time.sleep(5)

def http(path, post=False):
    check_deadline()
    request = urllib.request.Request(f'http://127.0.0.1:{PORT}/{path}', data=b'' if post else None)
    with urllib.request.urlopen(request, timeout=min(90 if post else 3, max(1, DEADLINE-time.monotonic()))) as r:
        assert r.status == 200

def verify_sources():
    assert sha(BENCH) == BENCH_SHA
    active = read(ART/'active_inference_plan.json')
    assert active['selected_experiment'] == 'EXP-0016' and active['selected_serving_commits'] == COMMITS
    for name, commit in COMMITS.items():
        tree = WORK/name
        assert subprocess.check_output(['git','rev-parse','HEAD'], cwd=tree, text=True).strip() == commit
        assert not subprocess.check_output(['git','status','--porcelain'], cwd=tree, text=True).strip()
    manifest = read(EXP/'script_manifest.json')
    actual = {p.name:sha(p) for p in EXP.glob('*.py')}
    assert actual == manifest, 'Profiling scripts changed after validation/freeze'
    installed = Path('/usr/local/lib/python3.12/site-packages/vllm')
    for rel, digest in INSTALLED.items(): assert sha(installed/rel) == digest, rel
    return dict(commits=COMMITS, benchmark_sha256=BENCH_SHA, script_hashes=actual,
                installed_hashes=INSTALLED, case=[4096,1024,64,256], passes=list(MODES),
                diagnostic_only=True, performance_comparison_allowed=False,
                protocol='One full official-shaped request set per diagnostic pass, not official four-round performance',
                accuracy='No full model accuracy evaluation; inference implementation unchanged')

def startup_route(folder, mode):
    log = (folder/'server.log').read_text(errors='replace')
    assert log.count('LeanScale GQA active:') == 42
    assert log.count('LeanScale fused RoPE+KV active:') == 42
    assert 'LeanScale RoPE+KV graph fusion: replaced=42' in log
    assert 'KVReservationAsyncScheduler' not in log
    if mode == 'FULL_DECODE_ONLY': assert 'Capturing CUDA graphs (decode, FULL)' in log and '35/35' in log

def one_pass(mode):
    folder = RUN/mode
    folder.mkdir()
    atomic(folder/'window_control.json', {'window':'initial'})
    cfg = dict(profiler='torch', torch_profiler_dir=str(folder/'traces'),
               torch_profiler_with_stack=False, torch_profiler_record_shapes=True,
               torch_profiler_with_memory=False, torch_profiler_use_gzip=True,
               torch_profiler_dump_cuda_time_total=False, ignore_frontend=True, max_iterations=16)
    server = start(['vllm','serve','/workspace/MiniCPM5-2B','--port',str(PORT),
                    '--compilation-config',json.dumps({'cudagraph_mode':mode}),
                    '--served-model-name','minicpm','--gpu-memory-utilization','0.85',
                    '--max-model-len','131072','--profiler-config',json.dumps(cfg),
                    '--worker-extension-cls','profile_hooks.ProfileExtension'], folder, 'server')
    state('starting_profile_server', mode=mode, server_pid=server.pid)
    try:
        startup_end = time.monotonic()+1800
        while True:
            check_deadline()
            assert server.poll() is None, f'Server exited: {server.returncode}'
            try: http('health'); break
            except Exception:
                if time.monotonic() >= startup_end: raise TimeoutError('Service startup exceeded 30 minutes')
                time.sleep(3)
        payload = json.dumps(dict(model='minicpm',prompt='Explain matrix multiplication briefly.',
                                  max_tokens=16,ignore_eos=True)).encode()
        req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/completions',data=payload,
                                     headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=min(180,max(1,DEADLINE-time.monotonic()))) as r: smoke=json.load(r)
        assert smoke['usage']['completion_tokens'] == 16
        # Keep only token counts, not arbitrary model output or request IDs.
        atomic(folder/'smoke.json',dict(usage=smoke['usage'],passed=True))
        startup_route(folder,mode)
        spec=importlib.util.spec_from_file_location('official_benchmark',BENCH)
        official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
        command=official.build_common_args('/workspace/MiniCPM5-2B',PORT,'minicpm')
        command += ['--random-input-len','4096','--random-output-len','1024',
                    '--max-concurrency','64','--num-prompts','256','--profile']
        client=start(command,folder,'client')
        state('profiling_4k',mode=mode,client_pid=client.pid)
        captured,current,events={'initial'},'initial',[]
        workload_end=time.monotonic()+2700
        while client.poll() is None:
            check_deadline()
            assert server.poll() is None, f'Server exited: {server.returncode}'
            if time.monotonic() >= workload_end: raise TimeoutError('Diagnostic request set exceeded 45 minutes')
            latest=read(folder/'latest_batch.json') or {}
            window=choose_window(latest,captured,current,time.time())
            if window:
                http('stop_profile',post=True)
                atomic(folder/'window_control.json',{'window':window})
                http('start_profile',post=True)
                current=window;captured.add(window)
                events.append(dict(window=window,epoch=time.time(),trigger_batch=latest))
                atomic(folder/'capture_events.json',events)
                state('profiling_4k',mode=mode,capture_window=window,batch=latest)
            time.sleep(.5)
        assert client.returncode == 0, f'Diagnostic client failed: {client.returncode}'
        http('stop_profile',post=True)
        metrics=official.extract_metrics((folder/'client.log').read_text(errors='replace'))
        assert metrics['successful_requests']==256 and metrics['total_input_tokens']==1048576 and metrics['total_output_tokens']==262144,metrics
        assert read(folder/'latest_batch.json')['new_requests_total']==256, 'Main request counter did not match completed workload'
        metadata=[json.loads(s) for s in (folder/'step_metadata.jsonl').read_text().splitlines()]
        batch=[m for m in metadata if m['kind']=='batch']
        assert batch and all(m['main_requests_entered']<=256 for m in batch)
        assert all(m['B']==len(m['query_lengths'])==len(m['context_lengths_cpu_upper_bound'])==len(m['request_phases']) for m in batch)
        atomic(folder/'workload_validation.json',dict(complete=True,requests=256,input_tokens=1048576,
               output_tokens=262144,windows_requested=sorted(captured),missing_windows=sorted(set(WINDOWS)-captured),
               performance_comparison_allowed=False,actual_profiled_batch_steps=len(batch)))
        from route_checks import verify_route
        verify_route(folder,mode)
        assert verify_sources() == read(RUN/'provenance.json')
        for window in captured:
            assert list((folder/'traces'/window).glob('*.pt.trace.json*')), f'Missing trace export: {mode}/{window}'
    finally:
        for proc in reversed(CHILDREN): stop(proc)
        wait_release()

def publish():
    """Curated derived evidence only; raw traces, caches and logs remain local."""
    dest=EXP/'artifacts'/RUN.name
    dest.mkdir(parents=True,exist_ok=True)
    for name in ('status.json','provenance.json','analysis.json','idle_evidence.json','run_window.json','RESULTS.md','error.txt'):
        if (RUN/name).exists(): shutil.copy2(RUN/name,dest/name)
    for source in (RUN/'summary').rglob('*'):
        if source.is_file() and source.name in ('kernel_times.csv','operator_times.csv','cpu_operator_times.csv','summary.json','step_metadata.json'):
            target=dest/'summary'/source.relative_to(RUN/'summary');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    if (RUN/'probe/probe_result.json').exists():shutil.copy2(RUN/'probe/probe_result.json',dest/'probe_result.json')
    for mode in MODES:
        for name in ('route_evidence.json','workload_validation.json','capture_events.json','server.command.json','client.command.json','smoke.json'):
            source=RUN/mode/name
            if source.exists():
                target=dest/mode/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    # Hash all raw evidence without publishing process/host identifiers in trace filenames.
    raw=[]
    for mode in MODES:
        for p in sorted((RUN/mode).rglob('*')):
            if p.is_file() and ('traces' in p.parts or p.name in ('server.log','client.log','step_metadata.jsonl','batch_timeline.jsonl')):
                raw.append(dict(mode=mode,kind='trace' if 'traces' in p.parts else p.name,
                                window=p.parent.name if 'traces' in p.parts else None,sha256=sha(p),bytes=p.stat().st_size))
    atomic(dest/'raw_evidence_hashes.json',raw)
    if (RUN/'RESULTS.md').exists():shutil.copy2(RUN/'RESULTS.md',EXP/'RESULTS.md')
    atomic(EXP/'STATUS.json',STATE)
    for p in dest.rglob('*'):
        if p.is_file():
            assert p.stat().st_size<8*1024*1024,p.name
            assert not any(m in p.read_text(errors='replace') for m in ('PRIVATE'+' KEY','ssh.'+'flagos.io','lab'+'_inst_'))
    lock=(ART/'docs-publish.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    try:
        paths=['experiments/EXP-0018']
        subprocess.run(['git','add','--',*paths],cwd=REPO,check=True)
        diff=subprocess.run(['git','diff','--cached','--quiet','--',*paths],cwd=REPO)
        assert diff.returncode in (0,1)
        if diff.returncode: subprocess.run(['git','commit','-m','experiments: save EXP18 selected-version 4K phase profiling','--only','--',*paths],cwd=REPO,check=True)
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
        subprocess.run(['git','push','origin','main'],cwd=REPO,check=True,timeout=120)
        remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=REPO,text=True,timeout=60).split()[0]
        assert remote==head
        atomic(RUN/'publication.json',dict(status='pushed',commit=head,remote_verified=True))
    finally:lock.close()

def main():
    global RUN, GPU_LOCK, DEADLINE
    singleton=(ART/'EXP-0018-controller.lock').open('a');fcntl.flock(singleton,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not REQUEST.exists(), 'Do not silently restart a completed or failed diagnostic'
    policy=read(ART/'run_window.json')
    seconds=validate_window(policy,datetime.now().astimezone())
    assert seconds>2*PASS_BUDGET+120,'Need 162 minutes before cutoff for both bounded passes'
    DEADLINE=time.monotonic()+seconds
    RUN=ART/'EXP-0018'/datetime.now().strftime('%Y%m%d_%H%M%S_profile_4k');RUN.mkdir(parents=True)
    STATE.update(pid=os.getpid(),process_start_ticks=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()[19],
                 run_directory=str(RUN),started_at=datetime.now().astimezone().isoformat(),commits=COMMITS)
    atomic(RUN/'run_window.json',policy)
    state('queued')
    success=False
    try:
        atomic(RUN/'provenance.json',verify_sources())
        acquire_idle()
        state('checking_device_profiler')
        probe=start(['python3',str(EXP/'probe_profiler.py'),str(RUN/'probe')],RUN,'probe')
        assert probe.wait(timeout=120)==0,'Device profiler capability probe failed'
        for mode in MODES:
            assert DEADLINE-time.monotonic()>PASS_BUDGET,'Insufficient remaining pass budget'
            one_pass(mode)
            state('pass_complete',mode=mode)
        success=True
    except BaseException:
        (RUN/'error.txt').write_text(traceback.format_exc());traceback.print_exc()
    finally:
        clean=True
        for proc in reversed(CHILDREN):
            try:stop(proc)
            except BaseException:clean=False;traceback.print_exc()
        if GPU_LOCK is not None:
            try:wait_release()
            except BaseException:clean=False;traceback.print_exc()
            GPU_LOCK.close();GPU_LOCK=None
        state('analyzing' if success else 'failed',owned_children_reaped=clean)
        try:
            import analyze_profile
            report=analyze_profile.main(RUN)
            if success:
                for mode in MODES:
                    traces=[r for r in report['traces'] if r['mode']==mode]
                    assert traces and any(r['kernel_events']>0 for r in traces),mode
                none=[r for r in report['traces'] if r['mode']=='NONE']
                assert all(r['model_kernel_visibility'] for r in none),'NONE model visibility incomplete'
                STATE['coverage']=report['coverage']
                STATE['graph_visibility_limitation']=not bool(report['profiler_capability'].get('graph_device_kernel_events'))
        except BaseException:
            success=False
            with (RUN/'error.txt').open('a') as f:f.write(traceback.format_exc())
        state('complete' if success and clean else 'failed',raw_data_saved=True,owned_children_reaped=clean)
        try:publish()
        except BaseException:atomic(RUN/'publication.json',dict(status='push_failed_local_saved',error=traceback.format_exc()))
        singleton.close()

if __name__=='__main__':main()
