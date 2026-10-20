"""Wait for the shared GPU lock, then run frozen fusion diagnostics serially."""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import os
import signal
import shutil
import subprocess
import time
import traceback
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--kinds", nargs="+", choices=("rope", "mlp", "norm"),
                    default=["rope", "mlp", "norm"])
args = parser.parse_args()

root = Path("/workspace/LeanScale")
exp = root / "MiniCPM-Infer-Sys-Opt/experiments/EXP-0006"
wt = root / "worktrees/EXP-0006"
run = root / "artifacts/EXP-0006" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S_diagnostics")
run.mkdir(parents=True)
state = dict(pid=os.getpid(), run_directory=str(run), stage="waiting_for_gpu",
             kind="isolated_diagnostics_not_official_benchmark")
signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))


def update(stage, **extra):
    state.update(stage=stage, updated_at=datetime.datetime.now().astimezone().isoformat(),
                 **extra)
    tmp = run / "status.tmp"
    tmp.write_text(json.dumps(state, indent=2)+"\n")
    tmp.replace(run / "status.json")
    (root / "artifacts/EXP-0006-latest-diagnostics.json").write_text(
        json.dumps(state, indent=2)+"\n")
    print(json.dumps(state), flush=True)


files = [exp/"validate_fusion.py"]
files += list((wt/"FlagGems/src/flag_gems/runtime/backend/_iluvatar/fused").glob("*kv*.py"))
files += [wt/"FlagGems/src/flag_gems/runtime/backend/_iluvatar/fused/gate_up_swiglu.py",
          wt/"FlagGems/src/flag_gems/runtime/backend/_iluvatar/fused/rotary_embedding_parallel.py",
          wt/"FlagGems/src/flag_gems/runtime/backend/_iluvatar/fused/rms_qkv.py"]
files += [wt/"vllm-plugin-FL"/p for p in (
    "vllm_fl/__init__.py", "vllm_fl/ops/rope_kv_dispatch.py",
    "vllm_fl/compilation/rope_kv_fusion.py")]
manifest = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(run/"source_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
for p in files:
    shutil.copy2(p, run/p.name)
update("waiting_for_gpu", source_manifest_sha256=hashlib.sha256(
    (run/"source_manifest.json").read_bytes()).hexdigest())
lock = open("/workspace/experiments/codex_20261019/runner.lock", "a+")
try:
    while True:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            time.sleep(15)
    for path, expected in manifest.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    env = dict(os.environ, VLLM_PLUGINS="fl", PYTHONUNBUFFERED="1",
               PYTHONPATH=str(wt/"vllm-plugin-FL")+":"+str(wt/"FlagGems/src"),
               TORCHINDUCTOR_CACHE_DIR=str(run/"inductor_cache"))
    outcomes = {}
    for kind in args.kinds:
        update(kind+"_diagnostics", outcomes=outcomes)
        command = ["python3", str(run/"validate_fusion.py"), kind, str(run/(kind+".json"))]
        (run/(kind+".command.json")).write_text(json.dumps(command, indent=2))
        with (run/(kind+".log")).open("w") as log:
            proc = subprocess.Popen(command, env=env, cwd=run, stdout=log,
                                    stderr=subprocess.STDOUT)
            state["child_pid"] = proc.pid
            update(kind+"_diagnostics")
            try:
                outcomes[kind] = proc.wait(timeout=3600)
            except BaseException:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                raise
    update("diagnostics_complete" if all(v == 0 for v in outcomes.values())
           else "diagnostics_complete_with_failures", outcomes=outcomes)
except BaseException as exc:
    (run/"error.log").write_text(traceback.format_exc())
    update("failed", error=repr(exc))
    raise
finally:
    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    lock.close()
