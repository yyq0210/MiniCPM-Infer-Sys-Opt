"""Dated stop/checkpoint guard for known LeanScale controllers only.

Runs independently of the desktop. SIGINT enters each Python controller's
existing try/finally cleanup and archive path. No system shutdown, broad pkill,
or signaling of unidentified GPU processes.
"""
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

ROOT = Path("/workspace/LeanScale")
ART = ROOT / "artifacts"
REPO = ROOT / "MiniCPM-Infer-Sys-Opt"
POLICY = ART / "run_window.json"
DEST = REPO / "docs/evidence/night_20261020"
STATE_FILE = ART / "night_guard_20261020.json"
STATE = {}


def read(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def state(stage, **kwargs):
    STATE.update(stage=stage, updated_at=datetime.now().astimezone().isoformat(), **kwargs)
    write(STATE_FILE, STATE)
    print(json.dumps(STATE, ensure_ascii=False), flush=True)


def process_identity(receipt, script):
    """Require PID, process start tick and exact known script argument."""
    if not receipt or "pid" not in receipt or "process_start_ticks" not in receipt:
        return False
    path = Path(f"/proc/{receipt['pid']}")
    try:
        fields = (path / "stat").read_text().rsplit(") ", 1)[1].split()
        args = (path / "cmdline").read_bytes().split(b"\0")
        return (fields[0] != "Z" and fields[19] == str(receipt["process_start_ticks"])
                and str(script).encode() in args)
    except FileNotFoundError:
        return False


def stop_known(receipt, script, reason):
    if not process_identity(receipt, script):
        return None
    # Python was launched detached, but SIGINT still needs a live handler.
    # Do not substitute an unscoped kill if a different process inherited it.
    status = Path(f"/proc/{receipt['pid']}/status").read_text().splitlines()
    ignored = int(next(x.split()[1] for x in status if x.startswith("SigIgn:")), 16)
    if ignored & (1 << (signal.SIGINT - 1)):
        raise RuntimeError(f"Known controller {receipt['pid']} ignores SIGINT; review required")
    record = dict(pid=receipt["pid"], process_start_ticks=receipt["process_start_ticks"],
                  script=str(script.relative_to(REPO)), reason=reason,
                  requested_at=datetime.now().astimezone().isoformat())
    run = receipt.get("run_directory")
    if run:
        write(Path(run) / "planned_shutdown_notice.json", record)
    os.kill(receipt["pid"], signal.SIGINT)
    return record


def snapshot(label):
    """Small audit records, CSV/trace hashes, and completed artifact locations."""
    folder = DEST / label
    folder.mkdir(parents=True, exist_ok=True)
    report = {"at": datetime.now().astimezone().isoformat(),
              "policy": read(POLICY), "guard": dict(STATE), "experiments": {}}
    for experiment, receipt_name in [
        ("EXP-0006", "EXP-0006-latest-16k.json"),
        ("EXP-0007", "EXP-0007-latest-official.json"),
        ("EXP-0008", "EXP-0008-request.json"),
    ]:
        receipt = read(ART / receipt_name)
        if not receipt:
            continue
        run = Path(receipt["run_directory"])
        entry = {"run_directory": str(run)}
        for name in ["status.json", "progress.json", "completion_handler_status.json",
                     "publication.json", "planned_shutdown_notice.json"]:
            data = read(run / name)
            if data:
                entry[name] = data
        files = []
        for pattern in ("**/raw_runs_*.csv", "**/summary_*.csv", "**/*.pt.trace.json.gz",
                        "**/workload_validation.json", "RESULTS.md", "analysis.json"):
            for path in run.glob(pattern):
                # Raw traces remain on server; publish their locations/hashes.
                digest = hashlib.sha256()
                with path.open("rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        digest.update(chunk)
                files.append({"path": str(path.relative_to(run)), "bytes": path.stat().st_size,
                              "sha256": digest.hexdigest()})
        entry["saved_files"] = files
        report["experiments"][experiment] = entry
    report["scheduler_queue"] = read(ART / "EXP-0007-queue.json")
    write(folder / "checkpoint.json", report)
    lines = ["# 2026-10-20 午夜关机前检查点", "",
             f"保存时间：{report['at']}。", "",
             "计算截止 23:35，归档目标 23:50，机器预计于 2026-10-21 00:00 关闭（北京时间）。",
             "只采纳完整完成的官方四轮数据；中断轮次保留为现场记录。profiling 结果独立于正式性能成绩。", ""]
    for exp, data in report["experiments"].items():
        s = data.get("status.json", {})
        lines += [f"- {exp}：`{s.get('stage', 'unknown')}`；"
                  f"原始目录 `{data['run_directory']}`；已存文件 {len(data['saved_files'])} 个。"]
    lines += ["", "各实验自动归档器负责结果与 CSV 的提交；本检查点保留状态及原始文件哈希。",
              "若上传失败，服务器原始文件与已生成的仓库文件仍保留，重启后核验补推。",
              "重启后需先确认机器已恢复、更新过期运行窗口，再继续未完成组；不拼接中断轮次。", ""]
    (DEST / "README.md").write_text("\n".join(lines))
    return report


def publish_checkpoint():
    relative = str(DEST.relative_to(REPO))
    # Finishers own other paths. If their index is busy, retry rather than
    # absorbing any staged result into this checkpoint commit.
    if subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=REPO, text=True).strip():
        return False
    for path in DEST.rglob("*"):
        if path.is_file():
            body = path.read_text(errors="replace")
            assert "PRIVATE KEY" not in body and "lab_inst_" not in body
    subprocess.run(["git", "add", "--", relative], cwd=REPO, check=True)
    diff = subprocess.run(["git", "diff", "--cached", "--quiet", "--", relative], cwd=REPO)
    if diff.returncode == 1:
        subprocess.run(["git", "commit", "--only", "-m", "docs: save midnight shutdown checkpoint",
                        "--", relative], cwd=REPO, check=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    subprocess.run(["git", "push", "origin", "main"], cwd=REPO, check=True, timeout=120)
    url = subprocess.check_output(["git", "remote", "get-url", "--push", "origin"],
                                  cwd=REPO, text=True).strip()
    remote = subprocess.check_output(["git", "ls-remote", url, "refs/heads/main"],
                                     cwd=REPO, text=True, timeout=60).split()[0]
    assert remote == head
    state("checkpoint_pushed", checkpoint_commit=head, verified=True)
    return True


def main():
    lock = (ART / "night_guard_20261020.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = read(POLICY)
    stop_at = datetime.fromisoformat(cfg["compute_stop_at"])
    archive_at = datetime.fromisoformat(cfg["archive_target_at"])
    latest_profile = datetime.fromisoformat(cfg["profile_latest_start_at"])
    STATE.update(pid=os.getpid(),
                 process_start_ticks=Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()[19],
                 policy=cfg, signal_requests=[])
    state("armed")
    profile_deferred = False
    while datetime.now().astimezone() < stop_at:
        if datetime.now().astimezone() >= latest_profile and not profile_deferred:
            p = read(ART / "EXP-0008-request.json")
            if p and p.get("stage") in ("queued", "waiting_for_EXP0006_16k", "waiting_for_idle_gpu"):
                try:
                    event = stop_known(p, REPO / "experiments/EXP-0008/run_profile.py",
                                       "Less than one hour remains for the two profiling passes")
                except Exception:
                    event = None
                    state("profile_stop_needs_review", error=traceback.format_exc())
                if event:
                    STATE["signal_requests"].append(event)
                    state("profile_deferred_for_shutdown")
                profile_deferred = True
        time.sleep(20)
    # Stop the waiting launch queue before asking running controllers to exit.
    targets = [
        ("EXP-0007-queue.json", "experiments/EXP-0007/queue_after_fusion.py"),
        ("EXP-0008-request.json", "experiments/EXP-0008/run_profile.py"),
        ("EXP-0006-latest-16k.json", "experiments/EXP-0006/run_split_e2e.py"),
        ("EXP-0007-latest-official.json", "experiments/EXP-0007/run_split_e2e.py"),
    ]
    stopped = []
    for receipt_name, script in targets:
        receipt = read(ART / receipt_name)
        if receipt and receipt.get("run_directory"):
            current = read(Path(receipt["run_directory"]) / "status.json") or receipt
            if current.get("stage") in ("performance_case_complete", "failed", "complete", "analyzing"):
                # Let an already completed controller finish cleanup/export.
                continue
        try:
            event = stop_known(receipt, REPO / script, "23:35 computation cutoff before midnight power-off")
        except Exception:
            state("controller_stop_needs_review", stop_error=traceback.format_exc())
            continue
        if event:
            stopped.append((receipt, REPO / script))
            STATE["signal_requests"].append(event)
    state("cleanup_requested")
    # Existing controller finally blocks stop only their own services; existing
    # finishers remain running so they can archive partial or complete results.
    until = time.monotonic() + 180
    while any(process_identity(p, script) for p, script in stopped) and time.monotonic() < until:
        time.sleep(5)
    alive = [p["pid"] for p, script in stopped if process_identity(p, script)]
    state("awaiting_archive", controllers_still_alive=alive)
    # Allow normal finishers to publish before the final checkpoint. Keep raw
    # files untouched even if a controller fails to respond.
    while datetime.now().astimezone() < archive_at:
        time.sleep(20)
    snapshot("final")
    for attempt in range(3):
        try:
            if publish_checkpoint():
                return
        except Exception:
            state("checkpoint_push_failed_local_saved", error=traceback.format_exc())
        time.sleep(20)
    state("local_checkpoint_saved", checkpoint_directory=str(DEST))


if __name__ == "__main__":
    main()
