"""Launch one complete official case, with a shutdown deadline if supplied."""
from pathlib import Path
import argparse
import datetime
import fcntl
import json
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--case", choices=("4k", "16k"), required=True)
parser.add_argument("--finish-before", type=datetime.datetime.fromisoformat)
args = parser.parse_args()
root = Path("/workspace/LeanScale")
exp = root/"MiniCPM-Infer-Sys-Opt/experiments/EXP-0016"
now = datetime.datetime.now().astimezone()
policy = json.loads((root/'artifacts/run_window.json').read_text())
assert args.finish_before is not None
assert args.finish_before == datetime.datetime.fromisoformat(policy['compute_stop_at'])
if args.finish_before:
    minimum_seconds = 55*60 if args.case == "4k" else 90*60
    assert (args.finish_before-now).total_seconds() >= minimum_seconds, (
        "Insufficient time for a complete case; save it for after restart")
lock = open(root/"artifacts/EXP-0016-launch.lock", "a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
latest = root/"artifacts/EXP-0016-latest-official.json"
if latest.exists():
    prior = json.loads(latest.read_text())
    state = json.loads((Path(prior["run_directory"])/"status.json").read_text())
    proc = Path("/proc")/str(state["pid"])
    if proc.exists():
        fields = (proc/"stat").read_text().rsplit(") ", 1)[1].split()
        assert fields[0] == "Z" or fields[19] != state["process_start_ticks"], (
            "An official coordinator is still active")
    handler = Path(prior["run_directory"])/"completion_handler_status.json"
    if handler.exists():
        assert json.loads(handler.read_text())["status"] != "waiting", (
            "Previous result publisher still active")
run = root/"artifacts/EXP-0016"/now.strftime("%Y%m%d_%H%M%S_official_"+args.case)
command = ["python3", str(exp/"run_split_e2e.py"), "--case", args.case,
           "--baseline-from", str(root/"artifacts/EXP-0004/20261019_202342"),
           "--run-dir", str(run)]
if args.finish_before:
    command += ["--finish-before", args.finish_before.isoformat()]
log = open(root/"artifacts/EXP-0016-split-controller.log", "a")
p = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
for _ in range(100):
    if (run/"status.json").exists():
        break
    if p.poll() is not None:
        raise RuntimeError("Coordinator failed before recording status")
    time.sleep(.2)
else:
    raise RuntimeError("No fresh coordinator receipt")
receipt = json.loads((run/"status.json").read_text())
latest.write_text(json.dumps(receipt, indent=2)+"\n")
(root/("artifacts/EXP-0016-latest-"+args.case+".json")).write_text(
    json.dumps(receipt, indent=2)+"\n")
log2 = open(root/"artifacts/EXP-0016-split-completion.log", "a")
f = subprocess.Popen(["python3", str(exp/"finish_split.py"), "--run-dir", str(run), "--publish"],
                     stdout=log2, stderr=subprocess.STDOUT, start_new_session=True)
print(json.dumps(dict(run_directory=str(run), controller_pid=p.pid, finisher_pid=f.pid,
                      case=args.case, deadline=receipt.get("planned_finish_before"))), flush=True)
lock.close()
