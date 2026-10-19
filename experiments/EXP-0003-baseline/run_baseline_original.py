"""Run the unchanged competition throughput script with the required server args.

Logs and status are persistent; completion/failure shuts down only this runner's server.
"""
import fcntl
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "baseline_official"
RUN.mkdir(exist_ok=True)
SERVER = ["vllm", "serve", "/workspace/MiniCPM5-2B", "--port", "9031",
          "--compilation-config", '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
          "--served-model-name", "minicpm", "--gpu-memory-utilization", "0.85",
          "--max-model-len", "131072"]
BENCH = ["python3", "/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py",
         "--model", "/workspace/MiniCPM5-2B", "--served-model-name", "minicpm",
         "--port", "9031", "--test-cases", "[[4096,1024,64,256],[16384,1024,64,128]]"]
STATE = {"runner_pid": os.getpid(), "started_at": time.strftime("%FT%T%z"),
         "server_argv": SERVER, "benchmark_argv": BENCH, "run_directory": str(RUN)}


def status(stage, **kwargs):
    STATE.update(stage=stage, updated_at=time.strftime("%FT%T%z"), **kwargs)
    tmp=ROOT / "status.json.tmp"
    tmp.write_text(json.dumps(STATE, indent=2))
    tmp.replace(ROOT / "status.json")
    print(json.dumps({"stage":stage, **kwargs}), flush=True)


def main():
    lock = (ROOT / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    expected={"vllm-plugin-FL":"13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0",
              "FlagGems":"a7620cc191a0b42e040194622c5758b22a7a25dc"}
    for repo, sha in expected.items():
        head=subprocess.check_output(["git","-C","/workspace/"+repo,"rev-parse","HEAD"],text=True).strip()
        dirty=subprocess.check_output(["git","-C","/workspace/"+repo,"status","--porcelain"],text=True)
        if head != sha or dirty:
            raise RuntimeError(f"Baseline checkout is not clean at pinned commit: {repo}")
    STATE["commits"]=expected
    bench_path=Path(BENCH[1])
    STATE["benchmark_sha256"]=hashlib.sha256(bench_path.read_bytes()).hexdigest()
    (RUN / "commands.json").write_text(json.dumps(STATE,indent=2))
    try:
        urllib.request.urlopen("http://127.0.0.1:9031/health",timeout=3)
    except Exception:
        pass
    else:
        raise RuntimeError("Port 9031 already has a healthy server; refusing a duplicate run")
    env=dict(os.environ,VLLM_PLUGINS="fl",PYTHONUNBUFFERED="1")
    proc = None
    try:
        with (RUN / "server.log").open("w") as log:
            proc=subprocess.Popen(SERVER,cwd=RUN,env=env,stdout=log,stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL,start_new_session=True)
        (ROOT / "official_server.pid").write_text(str(proc.pid))
        status("server_starting",server_pid=proc.pid)
        deadline=time.monotonic()+1200
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"Server exited: {proc.returncode}; inspect server.log")
            try:
                with urllib.request.urlopen("http://127.0.0.1:9031/health",timeout=3) as resp:
                    if resp.status==200:
                        break
            except Exception:
                if time.monotonic()>deadline:
                    raise TimeoutError("Server did not become ready within 20 minutes")
                time.sleep(5)
        status("smoke_test")
        body=json.dumps({"model":"minicpm","prompt":"The capital of France is",
                         "max_tokens":32,"temperature":0}).encode()
        req=urllib.request.Request("http://127.0.0.1:9031/v1/completions",body,
                                   {"Content-Type":"application/json"})
        with urllib.request.urlopen(req,timeout=180) as response:
            sample=json.load(response)
        assert sample["choices"] and sample["usage"]["completion_tokens"]>0, sample
        (RUN / "smoke.json").write_text(json.dumps(sample,ensure_ascii=False,indent=2))
        status("throughput_running")
        with (RUN / "benchmark.log").open("w") as log:
            result=subprocess.run(BENCH,cwd=RUN,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=10800)
        if result.returncode:
            raise RuntimeError(f"Benchmark exited {result.returncode}")
        if hashlib.sha256(bench_path.read_bytes()).hexdigest()!=STATE["benchmark_sha256"]:
            raise RuntimeError("Official benchmark file changed during measurement")
        summaries=list(RUN.glob("benchmark_results/summary_*.csv"))
        if not summaries:
            raise RuntimeError("Benchmark did not produce a summary CSV")
        with summaries[-1].open() as f:
            rows=list(csv.DictReader(f))
        if len(rows)!=2 or {int(r["Prefill"]) for r in rows}!={4096,16384}:
            raise RuntimeError("Official benchmark omitted a failed case; inspect raw CSV and benchmark.log")
        status("throughput_complete",summaries=[str(p) for p in summaries],
               accuracy_status="not_run",optimization_status="isolated_microbench_only")
    except BaseException as exc:
        status("failed",error=str(exc))
        traceback.print_exc()
        raise
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                proc.wait()


if __name__ == "__main__":
    main()
