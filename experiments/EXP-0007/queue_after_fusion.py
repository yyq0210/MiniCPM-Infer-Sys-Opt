"""Serial continuation: complete fusion reference -> scheduler 16K -> scheduler 4K.

No partial benchmark reuse, no concurrent GPU runs, no automatic retry of failed
code. All launches still acquire the shared GPU lock in run_split_e2e.py.
"""
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path("/workspace/LeanScale")
EXP = ROOT / "MiniCPM-Infer-Sys-Opt/experiments/EXP-0007"
STATE_FILE = ROOT / "artifacts/EXP-0007-queue.json"
STATE = {}


def read(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def state(stage, **kwargs):
    STATE.update(stage=stage, updated_at=datetime.datetime.now().astimezone().isoformat(),
                 **kwargs)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(STATE, indent=2) + "\n")
    tmp.replace(STATE_FILE)
    print(json.dumps(STATE), flush=True)


def inspect(exp, case):
    receipt = read(ROOT / f"artifacts/{exp}-latest-{case}.json")
    if receipt is None:
        return None
    run = Path(receipt["run_directory"])
    return run, read(run / "status.json"), read(run / "completion_handler_status.json")


def launch_window(case):
    """Respect the user's dated shutdown window without changing the benchmark."""
    policy = read(ROOT / "artifacts/run_window.json")
    if not policy:
        return [], None
    finish = datetime.datetime.fromisoformat(policy["compute_stop_at"])
    remaining = (finish - datetime.datetime.now().astimezone()).total_seconds()
    required = 55 * 60 if case == "4k" else 90 * 60
    parent = inspect("EXP-0006", case)
    if parent and parent[1] and parent[1].get("stage") == "performance_case_complete":
        status = parent[1]
        duration = (datetime.datetime.fromisoformat(status["updated_at"]) -
                    datetime.datetime.fromisoformat(status["started_at"])).total_seconds()
        required = max(required, duration * 1.15 + 5 * 60)
    budget = dict(case=case, remaining_seconds=remaining,
                  estimated_required_seconds=required,
                  compute_stop_at=policy["compute_stop_at"],
                  machine_shutdown_at=policy["machine_shutdown_at"])
    if remaining < required:
        return None, budget
    return ["--finish-before", finish.isoformat()], budget


def wait_complete(exp, case, deadline):
    last_update = 0
    while time.monotonic() < deadline:
        data = inspect(exp, case)
        if data:
            run, status, handler = data
            if status and status["stage"] == "failed":
                raise RuntimeError(f"{exp} {case} failed; inspect {run}/status.json")
            if handler and handler["status"] == "handler_failed_raw_data_retained":
                raise RuntimeError(f"{exp} {case} archiving failed; inspect {run}")
            if handler and handler["status"] in ("pushed", "push_failed_local_saved",
                                                "publish_failed_local_saved",
                                                "local_saved"):
                assert handler.get("complete_results"), handler
                assert status["stage"] == "performance_case_complete", status
                return data
        if time.monotonic() - last_update > 60:
            state("waiting_for_case", waiting_experiment=exp, waiting_case=case,
                  observed_stage=data[1].get("stage") if data and data[1] else None)
            last_update = time.monotonic()
        time.sleep(15)
    raise TimeoutError(f"Wait for {exp} {case} exceeded the continuation deadline")


def main():
    lock = (ROOT / "artifacts/EXP-0007-queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    STATE.update(pid=os.getpid(),
                 process_start_ticks=Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()[19],
                 started_at=datetime.datetime.now().astimezone().isoformat(),
                 completed_cases=[])
    deadline = time.monotonic() + 16 * 3600
    try:
        candidate = read(EXP / "candidate_commits.json")
        parent = read(EXP / "parent_commits.json")
        assert candidate and parent
        for case in ("4k", "16k"):
            _, status, _ = wait_complete("EXP-0006", case, deadline)
            assert status["commits"] == parent
            if case == "16k":
                gain = status["incremental_vs_exp5"][0]["throughput_change_percent"]
                if gain < -3:
                    state("needs_review", reason="Fusion 16K regression exceeds predeclared 3% gate",
                          fusion_change_percent=gain)
                    return
        # User-requested 4K operator profiling has priority in the idle slot
        # following EXP-0006. It owns the same GPU lock; never race its launch.
        while time.monotonic() < deadline:
            profile = read(ROOT / "artifacts/EXP-0008-request.json")
            if profile is None:
                break
            if profile.get("stage") in ("complete", "failed"):
                if profile.get("safe_to_resume_scheduler"):
                    break
                raise RuntimeError("EXP-0008 did not confirm safe GPU release")
            state("waiting_for_operator_profile",
                  profiling_stage=profile.get("stage"),
                  profiling_run=profile.get("run_directory"))
            time.sleep(30)
        else:
            raise TimeoutError("Waiting for EXP-0008 profiling exceeded deadline")
        # Long-context policy first; 4K is the regression check.
        for case in ("16k", "4k"):
            prior = inspect("EXP-0007", case)
            if prior:
                assert prior[1]["commits"] == candidate
                run, status, handler = wait_complete("EXP-0007", case, deadline)
            else:
                deadline_args, budget = launch_window(case)
                if deadline_args is None:
                    state("deferred_for_shutdown", reason="Insufficient time for a complete official case",
                          launch_budget=budget, deferred_case=case)
                    return
                state("launching", launching_case=case, launch_budget=budget)
                result = subprocess.check_output(
                    ["python3", str(EXP / "launch_split.py"), "--case", case] + deadline_args,
                    text=True, timeout=120)
                receipt = json.loads(result)
                run = Path(receipt["run_directory"])
                log = ROOT / "artifacts/EXP-0007-split-controller.log"
                if log.exists() and not (run / "coordinator.log").exists():
                    log.rename(run / "coordinator.log")
                run, status, handler = wait_complete("EXP-0007", case, deadline)
            assert status["commits"] == candidate
            STATE["completed_cases"].append(case)
            if case == "16k":
                gain = status["incremental_vs_exp6"][0]["throughput_change_percent"]
                if gain < -3:
                    state("needs_review", reason="Scheduler 16K regression exceeds predeclared 3% gate",
                          scheduler_change_percent=gain)
                    return
        state("complete", reason="Both official scheduler cases saved; inspect results before selection")
    except BaseException as exc:
        state("failed", error=repr(exc))
        traceback.print_exc()
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
