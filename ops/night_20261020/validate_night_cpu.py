"""CPU-only validation of process identity, cleanup signaling and launch budgets."""
from datetime import datetime, timedelta
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = load("guard", Path(__file__).with_name("night_guard.py"))
queue = load("queue_test", Path(sys.argv[1]))
with tempfile.TemporaryDirectory() as d:
    root = Path(d)
    script = root / "safe_child.py"
    script.write_text(
        "from pathlib import Path\nimport time\n"
        "try:\n Path('ready').touch()\n time.sleep(30)\n"
        "finally:\n Path('cleaned').touch()\n")
    proc = subprocess.Popen([sys.executable, str(script)], cwd=root,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)
    try:
        for _ in range(50):
            if (root / "ready").exists():
                break
            time.sleep(.1)
        assert (root / "ready").exists()
        ticks = Path(f"/proc/{proc.pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]
        receipt = dict(pid=proc.pid, process_start_ticks=ticks, run_directory=str(root))
        guard.REPO = root
        assert guard.process_identity(receipt, script)
        bad = dict(receipt, process_start_ticks=str(int(ticks) + 1))
        assert guard.stop_known(bad, script, "must_not_signal") is None
        assert proc.poll() is None
        assert not guard.process_identity(receipt, root / "other.py")
        assert guard.stop_known(receipt, script, "validation_only")
        proc.wait(timeout=5)
        assert (root / "cleaned").exists(), "SIGINT must enter finally cleanup"
        assert not guard.process_identity(receipt, script)
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)

    (root / "artifacts").mkdir()
    queue.ROOT = root
    now = datetime.now().astimezone()
    parent = {"stage": "performance_case_complete",
              "started_at": (now - timedelta(hours=2)).isoformat(),
              "updated_at": now.isoformat()}
    queue.inspect = lambda exp, case: (root, parent, None)
    p = root / "artifacts/run_window.json"
    p.write_text(json.dumps({"compute_stop_at": (now + timedelta(hours=4)).isoformat(),
                            "machine_shutdown_at": (now + timedelta(hours=5)).isoformat()}))
    args, budget = queue.launch_window("16k")
    assert args[0] == "--finish-before"
    assert abs(budget["estimated_required_seconds"] - (7200 * 1.15 + 300)) < .01
    p.write_text(json.dumps({"compute_stop_at": (now + timedelta(minutes=30)).isoformat(),
                            "machine_shutdown_at": (now + timedelta(hours=1)).isoformat()}))
    assert queue.launch_window("16k")[0] is None
    p.write_text(json.dumps({"compute_stop_at": (now - timedelta(minutes=1)).isoformat(),
                            "machine_shutdown_at": now.isoformat()}))
    assert queue.launch_window("4k")[0] is None
print(json.dumps({"status": "passed", "checks": [
    "exact_PID_start_tick_script_match", "PID_reuse_not_signaled", "wrong_script_not_signaled",
    "SIGINT_runs_finally", "dead_process_not_signaled", "deadline_passed_to_launcher",
    "observed_duration_plus_margin", "insufficient_time_deferred", "expired_window_deferred"]}))
