"""Deferred, serialized 4K profiling. Only terminates processes it launches."""
import contextlib
from datetime import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request

EXP = Path(__file__).resolve().parent
ROOT = Path("/workspace/LeanScale")
REPO = ROOT / "MiniCPM-Infer-Sys-Opt"
ART = ROOT / "artifacts"
WORK = ROOT / "worktrees/EXP-0008"
REQUEST = ART / "EXP-0008-request.json"
BENCH = Path("/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py")
BENCH_SHA = "0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565"
COMMITS = {"vllm-plugin-FL": "07d9fa0cfdc5f09db178e29508e3c0bcb5c1e908",
           "FlagGems": "484413ea0db18221dd9e63e742361d837df9ff13"}
PORT = 9032
STATE = {}
RUN = None
CHILDREN = []
GPU_LOCK = None
DEADLINE = time.monotonic() + 16 * 3600


def read(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def atomic(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def state(stage, **kwargs):
    STATE.update(stage=stage, updated_at=datetime.now().astimezone().isoformat(), **kwargs)
    atomic(RUN / "status.json", STATE)
    atomic(REQUEST, STATE)
    with (RUN / "history.jsonl").open("a") as f:
        f.write(json.dumps(STATE, ensure_ascii=False) + "\n")
    print(json.dumps(STATE, ensure_ascii=False), flush=True)


def wait_previous():
    while time.monotonic() < DEADLINE:
        receipt = read(ART / "EXP-0006-latest-16k.json")
        if receipt:
            folder = Path(receipt["run_directory"])
            status = read(folder / "status.json") or {}
            handler = read(folder / "completion_handler_status.json") or {}
            terminal = handler.get("status") in (
                "pushed", "push_failed_local_saved", "publish_failed_local_saved", "local_saved",
                "handler_failed_raw_data_retained")
            if status.get("stage") in ("performance_case_complete", "failed") and terminal:
                state("waiting_for_idle_gpu", previous_run=str(folder),
                      previous_terminal_stage=status["stage"])
                return
        state("waiting_for_EXP0006_16k")
        time.sleep(30)
    raise TimeoutError("Previous experiment did not finish before the 16-hour deadline")


def gpu_snapshot():
    text = subprocess.check_output(["ixsmi"], text=True, timeout=20)
    match = re.search(r"(\d+)MiB\s*/\s*(\d+)MiB\s*\|\s*(\d+)%", text)
    if not match:
        raise RuntimeError("Cannot parse GPU memory/utilization; refusing an unverified idle launch")
    return {"memory_used_mib": int(match[1]), "memory_total_mib": int(match[2]),
            "gpu_util_percent": int(match[3]), "checked_at": datetime.now().astimezone().isoformat()}


def port_free(port):
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def acquire_idle():
    global GPU_LOCK
    while time.monotonic() < DEADLINE:
        handle = Path("/workspace/experiments/codex_20261019/runner.lock").open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            time.sleep(15)
            continue
        snapshots = []
        for _ in range(2):
            snapshots.append(gpu_snapshot())
            if snapshots[-1]["memory_used_mib"] > 1024 or snapshots[-1]["gpu_util_percent"] > 5:
                break
            time.sleep(5)
        idle = (len(snapshots) == 2 and all(
            x["memory_used_mib"] <= 1024 and x["gpu_util_percent"] <= 5 for x in snapshots))
        if idle and port_free(9031) and port_free(PORT):
            GPU_LOCK = handle
            atomic(RUN / "idle_evidence.json", snapshots)
            return
        handle.close()
        state("waiting_for_idle_gpu", observed=snapshots[-1])
        time.sleep(30)
    raise TimeoutError("GPU did not become safely idle before the deadline")


def env(folder):
    return dict(os.environ, VLLM_PLUGINS="fl", PYTHONUNBUFFERED="1",
                LEANSCALE_PROFILE_PASS=str(folder),
                PYTHONPATH=f"{EXP}:{WORK}/vllm-plugin-FL:{WORK}/FlagGems/src",
                VLLM_CACHE_ROOT=str(folder / "cache"))


def start(argv, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    atomic(folder / f"{name}.command.json", argv)
    with (folder / f"{name}.log").open("w") as log:
        proc = subprocess.Popen(argv, cwd=folder, env=env(folder), stdout=log,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
    CHILDREN.append(proc)
    return proc


def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=40)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=20)


def http(path, post=False):
    request = urllib.request.Request(f"http://127.0.0.1:{PORT}/{path}",
                                     data=b"" if post else None)
    with urllib.request.urlopen(request, timeout=90 if post else 3) as response:
        assert response.status == 200


def verify_sources():
    assert hashlib.sha256(BENCH.read_bytes()).hexdigest() == BENCH_SHA
    hashes = {}
    for repo, commit in COMMITS.items():
        actual = subprocess.check_output(["git", "-C", str(WORK / repo), "rev-parse", "HEAD"],
                                         text=True).strip()
        assert actual == commit, (repo, actual)
        assert not subprocess.check_output(["git", "-C", str(WORK / repo), "status", "--porcelain"],
                                            text=True).strip(), f"Dirty profiling worktree: {repo}"
    for p in EXP.glob("*.py"):
        hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = read(EXP / "script_manifest.json")
    assert manifest and all(hashes[name] == sha for name, sha in manifest.items()), (
        "Profiling scripts changed after CPU validation/freeze")
    for rel in ["profiler/wrapper.py", "config/profiler.py", "benchmarks/serve.py"]:
        p = Path("/usr/local/lib/python3.12/site-packages/vllm") / rel
        hashes["vllm/" + rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    atomic(RUN / "provenance.json", {
        "commits": COMMITS, "benchmark_sha256": BENCH_SHA, "script_hashes": hashes,
        "case": [4096, 1024, 64, 256], "passes": ["FULL_DECODE_ONLY", "NONE"],
        "accuracy": "No model accuracy evaluation in this diagnostic.",
        "protocol": "One complete official-shaped request set per diagnostic pass; not official performance.",
    })


def verify_route(folder, mode):
    log = (folder / "server.log").read_text(errors="replace")
    evidence = {
        "gqa_layers": log.count("LeanScale GQA active:"),
        "fused_runtime_layers": log.count("LeanScale fused RoPE+KV active:"),
        "fusion_pass_42": "LeanScale RoPE+KV graph fusion: replaced=42" in log,
        "graph_mode": mode,
        "graph_capture_35": "Capturing CUDA graphs (decode, FULL)" in log and "35/35" in log,
    }
    atomic(folder / "route_evidence.json", evidence)
    assert evidence["gqa_layers"] == 42 and evidence["fused_runtime_layers"] == 42, evidence
    assert evidence["fusion_pass_42"], evidence
    if mode == "FULL_DECODE_ONLY":
        assert evidence["graph_capture_35"], evidence


def one_pass(mode):
    folder = RUN / mode
    folder.mkdir()
    atomic(folder / "window_control.json", {"window": "initial"})
    cfg = {
        "profiler": "torch", "torch_profiler_dir": str(folder / "traces"),
        "torch_profiler_with_stack": False, "torch_profiler_record_shapes": True,
        "torch_profiler_with_memory": False, "torch_profiler_use_gzip": True,
        "torch_profiler_dump_cuda_time_total": False,
        "ignore_frontend": True, "max_iterations": 16,
    }
    server = start([
        "vllm", "serve", "/workspace/MiniCPM5-2B", "--port", str(PORT),
        "--compilation-config", json.dumps({"cudagraph_mode": mode}),
        "--served-model-name", "minicpm", "--gpu-memory-utilization", "0.85",
        "--max-model-len", "131072", "--profiler-config", json.dumps(cfg),
        "--worker-extension-cls", "profile_hooks.ProfileExtension",
    ], folder, "server")
    state("starting_profile_server", mode=mode, server_pid=server.pid)
    try:
        timeout = time.monotonic() + 1800
        while True:
            assert server.poll() is None, f"Profile server exited: {server.returncode}"
            try:
                http("health")
                break
            except Exception:
                if time.monotonic() > timeout:
                    raise TimeoutError("Profile server startup exceeded 30 minutes")
                time.sleep(3)
        verify_route(folder, mode)
        # Import the unchanged official command builder, not a reimplemented dataset.
        spec = importlib.util.spec_from_file_location("official_benchmark", BENCH)
        official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official)
        command = official.build_common_args("/workspace/MiniCPM5-2B", PORT, "minicpm")
        command += ["--random-input-len", "4096", "--random-output-len", "1024",
                    "--max-concurrency", "64", "--num-prompts", "256", "--profile"]
        client = start(command, folder, "client")
        state("profiling_4k", mode=mode, client_pid=client.pid)
        main_start = None
        captured = {"initial"}
        capture_events = []
        timeout = time.monotonic() + 2700
        while client.poll() is None:
            assert server.poll() is None, f"Profile server exited: {server.returncode}"
            now = time.monotonic()
            if now > timeout:
                raise TimeoutError("Diagnostic request set exceeded 45 minutes")
            log = (folder / "client.log").read_text(errors="replace")
            if main_start is None and "Starting main benchmark run..." in log:
                main_start = now
            latest = read(folder / "latest_batch.json") or {}
            elapsed = now - main_start if main_start is not None else 0
            window = None
            if "steady" not in captured and elapsed > 40:
                if (latest.get("phase") == "decode" and latest.get("B", 0) >= 48) or elapsed > 100:
                    window = "steady"
            elif "late" not in captured and "steady" in captured and elapsed > 120:
                if (latest.get("new_requests_total", 0) > 65 and latest.get("phase") != "decode") or elapsed > 240:
                    window = "late"
            if window:
                http("stop_profile", post=True)
                atomic(folder / "window_control.json", {"window": window})
                http("start_profile", post=True)
                captured.add(window)
                capture_events.append(dict(window=window, elapsed_s=elapsed, batch=latest))
                atomic(folder / "capture_events.json", capture_events)
                state("profiling_4k", mode=mode, capture_window=window, batch=latest)
            time.sleep(1)
        assert client.returncode == 0, f"Diagnostic client failed: {client.returncode}"
        http("stop_profile", post=True)
        metrics = official.extract_metrics((folder / "client.log").read_text(errors="replace"))
        assert metrics["successful_requests"] == 256, metrics
        assert metrics["total_input_tokens"] == 4096 * 256, metrics
        assert metrics["total_output_tokens"] == 1024 * 256, metrics
        atomic(folder / "workload_validation.json", {
            "complete": True, "requests": 256, "input_tokens": 4096 * 256,
            "output_tokens": 1024 * 256, "windows_requested": sorted(captured),
            "performance_comparison_allowed": False,
        })
        verify_route(folder, mode)
        assert captured == {"initial", "steady", "late"}, captured
        for window in captured:
            assert list((folder / "traces" / window).glob("*.pt.trace.json*")), (
                f"Profiler silently failed to export {mode}/{window}")
    finally:
        for proc in reversed(CHILDREN):
            stop(proc)
        # Wait for actual device release before the next service.
        timeout = time.monotonic() + 120
        while gpu_snapshot()["memory_used_mib"] > 1024:
            if time.monotonic() > timeout:
                raise RuntimeError("GPU memory did not release after diagnostic service stopped")
            time.sleep(5)


def publish():
    """Copy small curated outputs only; raw traces/logs stay on the server."""
    import shutil
    dest = EXP / "artifacts" / RUN.name
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("status.json", "provenance.json", "analysis.json", "idle_evidence.json",
                 "RESULTS.md", "error.txt"):
        source = RUN / name
        if source.exists():
            shutil.copy2(source, dest / name)
    if (RUN / "summary").exists():
        # Kernel mapping can be large; retain compact per-operator/kernel CSV
        # and summary JSON. Raw detailed records remain in RUN/summary.
        for source in (RUN / "summary").rglob("*"):
            if source.is_file() and source.name in (
                    "kernel_times.csv", "operator_times.csv", "cpu_operator_times.csv",
                    "summary.json", "step_metadata.json"):
                target = dest / "summary" / source.relative_to(RUN / "summary")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    if (RUN / "probe" / "probe_result.json").exists():
        shutil.copy2(RUN / "probe" / "probe_result.json", dest / "probe_result.json")
    for mode in ("FULL_DECODE_ONLY", "NONE"):
        for name in ("route_evidence.json", "workload_validation.json", "capture_events.json",
                     "server.command.json", "client.command.json"):
            source = RUN / mode / name
            if source.exists():
                target = dest / mode / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    if (RUN / "RESULTS.md").exists():
        shutil.copy2(RUN / "RESULTS.md", EXP / "RESULTS.md")
    atomic(EXP / "STATUS.json", STATE)
    # Only this experiment's files; concurrent finishers may touch other paths.
    for source in dest.rglob("*"):
        if source.is_file():
            assert source.stat().st_size < 8 * 1024 * 1024, f"Unexpected large export: {source.name}"
            text = source.read_text(errors="replace")
            assert not any(x in text for x in ("PRIVATE KEY", "ssh.flagos.io", "lab_inst_")), (
                f"Sensitive marker in export: {source.name}")
    lock = (ART / "docs-publish.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        subprocess.run(["git", "add", "experiments/EXP-0008"], cwd=REPO, check=True)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet", "--", "experiments/EXP-0008"],
                              cwd=REPO)
        if diff.returncode == 1:
            subprocess.run(["git", "commit", "-m", "docs: record EXP-0008 4K operator profiling",
                            "--only", "--", "experiments/EXP-0008"], cwd=REPO, check=True)
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        subprocess.run(["git", "push", "origin", "main"], cwd=REPO, check=True, timeout=120)
        url = subprocess.check_output(["git", "remote", "get-url", "--push", "origin"],
                                      cwd=REPO, text=True).strip()
        remote = subprocess.check_output(["git", "ls-remote", url, "refs/heads/main"],
                                         cwd=REPO, text=True, timeout=60).split()[0]
        assert remote == head
        atomic(RUN / "publication.json", {"status": "pushed", "commit": head, "verified": True})
    finally:
        lock.close()


def main():
    global RUN, GPU_LOCK
    singleton = (ART / "EXP-0008-controller.lock").open("a")
    fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
    RUN = ART / "EXP-0008" / datetime.now().strftime("%Y%m%d_%H%M%S_profile_4k")
    RUN.mkdir(parents=True)
    STATE.update(pid=os.getpid(), run_directory=str(RUN), safe_to_resume_scheduler=False,
                 process_start_ticks=Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()[19],
                 started_at=datetime.now().astimezone().isoformat(), commits=COMMITS)
    state("queued")
    success = False
    try:
        wait_previous()
        verify_sources()
        acquire_idle()
        state("checking_device_profiler")
        probe = start(["python3", str(EXP / "probe_profiler.py"), str(RUN / "probe")], RUN, "probe")
        assert probe.wait(timeout=120) == 0, "Device profiler capability probe failed"
        for mode in ("FULL_DECODE_ONLY", "NONE"):
            one_pass(mode)
        success = True
    except BaseException:
        (RUN / "error.txt").write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        clean = True
        for proc in reversed(CHILDREN):
            try:
                stop(proc)
            except Exception:
                clean = False
                traceback.print_exc()
        if GPU_LOCK is not None:
            try:
                release_deadline = time.monotonic() + 90
                while gpu_snapshot()["memory_used_mib"] > 1024:
                    if time.monotonic() > release_deadline:
                        raise RuntimeError("GPU memory remains occupied after child cleanup")
                    time.sleep(5)
            except Exception:
                clean = False
                traceback.print_exc()
            GPU_LOCK.close()
            GPU_LOCK = None
        state("analyzing" if success else "failed", owned_children_reaped=clean)
        try:
            import analyze_profile
            report = analyze_profile.main(RUN)
            if success:
                assert report["device_trace_count"] >= 6, "Missing GPU events in one or more trace windows"
                assert all(x["model_kernel_visibility"] for x in report["traces"]), (
                    "Incomplete model-kernel visibility; retain partial diagnostics, no complete-profile claim")
        except Exception:
            success = False
            with (RUN / "error.txt").open("a") as f:
                f.write(traceback.format_exc())
        # No GPU work remains. Scheduler may resume even if profiling failed;
        # its normal source/performance gates remain in effect.
        state("complete" if success else "failed", safe_to_resume_scheduler=clean,
              raw_data_saved=True)
        try:
            publish()
        except Exception:
            atomic(RUN / "publication.json", {
                "status": "push_failed_local_saved", "error": traceback.format_exc()})
        singleton.close()


if __name__ == "__main__":
    main()
