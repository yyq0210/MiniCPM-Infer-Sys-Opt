"""One-shot local archive and authorized fork push after fusion diagnostics."""
from pathlib import Path
import datetime
import fcntl
import json
import os
import re
import subprocess
import sys
import time

root = Path("/workspace/LeanScale")
repo = root/"MiniCPM-Infer-Sys-Opt"
exp = repo/"experiments/EXP-0006"
run = Path(sys.argv[1])
lock = open(exp/".finish.lock", "a")
fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
receipt = run/"completion_handler_status.json"
state = json.loads((run/"status.json").read_text())
pid = state["pid"]
proc = Path("/proc")/str(pid)
start_ticks = (proc/"stat").read_text().split()[21] if proc.exists() else None


def save(value):
    value["updated_at"] = datetime.datetime.now().astimezone().isoformat()
    tmp = receipt.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2)+"\n")
    tmp.replace(receipt)


def alive():
    try:
        fields = (proc/"stat").read_text().split()
        return fields[2] != "Z" and fields[21] == start_ticks
    except FileNotFoundError:
        return False


def git(*args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True,
                                   stderr=subprocess.STDOUT, timeout=180).strip()


try:
    save(dict(status="waiting", pid=os.getpid(), run_directory=str(run), publish_on_finish=True))
    while True:
        state = json.loads((run/"status.json").read_text())
        (exp/"STATUS.md").write_text(
            "# EXP-0006 当前状态\n\n更新："+datetime.datetime.now().astimezone().isoformat()+
            "\n\n阶段：`"+state["stage"]+"`。\n\n"
            "GPU 诊断与已有官方测评串行；结果写入服务器独立 artifacts 目录。"
            "算子微基准不代表整模收益，尚未宣布最终融合版本。\n")
        if state["stage"] in ("diagnostics_complete", "diagnostics_complete_with_failures", "failed"):
            break
        if not alive():
            state.update(stage="failed", error="diagnostics coordinator exited without terminal status")
            (run/"status.json").write_text(json.dumps(state, indent=2)+"\n")
            break
        time.sleep(30)
    subprocess.run(["python3", str(exp/"summarize_fusion.py"), str(run)], check=True)
    # EXP-0005 owns the same main checkout. Wait for its one-shot publication
    # handler to finish before staging anything in that checkout.
    r5 = json.loads((root/"artifacts/EXP-0005-latest-official.json").read_text())
    handler5 = Path(r5["run_directory"])/"completion_handler_status.json"
    deadline = time.monotonic()+600
    while handler5.exists() and json.loads(handler5.read_text()).get("status") == "waiting":
        if time.monotonic() > deadline:
            raise RuntimeError("EXP-0005 publisher still pending; all EXP-0006 data saved locally")
        time.sleep(15)
    assert git("branch", "--show-current") == "main"
    assert not git("diff", "--cached", "--name-only"), "Git index belongs to another operation"
    candidates = [p for p in exp.rglob("*") if p.is_file() and p.name != ".finish.lock"]
    for p in candidates:
        if p.stat().st_size > 20_000_000:
            raise RuntimeError("Refusing unexpected large artifact: "+str(p))
        if p.suffix in (".json", ".md", ".py", ".log"):
            content = p.read_text(errors="replace")
            patterns = (r"-----BEGIN (?:OPENSSH|RSA) PRIVATE KEY-----",
                        r"\b(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")
            assert not any(re.search(pattern, content) for pattern in patterns), (
                "Credential-shaped content in "+str(p))
    paths = [str(p.relative_to(repo)) for p in candidates]
    git("add", "--", *paths)
    git("diff", "--cached", "--check")
    if git("diff", "--cached", "--name-only"):
        git("commit", "-m", "experiments: archive fusion diagnostics and candidate comparison")
    commit = git("rev-parse", "HEAD")
    failures = []
    for attempt in range(3):
        try:
            git("push", "origin", "main")
            remote = git("remote", "get-url", "--push", "origin")
            assert git("ls-remote", remote, "refs/heads/main").split()[0] == commit
            save(dict(status="pushed", commit=commit, remote_verified=True,
                      experiment_stage=state["stage"]))
            break
        except Exception as exc:
            failures.append(str(exc))
            if attempt < 2:
                time.sleep(30)
    else:
        save(dict(status="push_failed_local_saved", commit=commit, errors=failures))
except BaseException as exc:
    save(dict(status="handler_failed_raw_data_retained", error=repr(exc)))
    raise
finally:
    lock.close()
