"""Freeze the validated scheduling candidate, archive evidence, and push forks."""
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path("/workspace/LeanScale")
DOCS = ROOT / "MiniCPM-Infer-Sys-Opt"
EXP = DOCS / "experiments/EXP-0007"
PLUGIN = ROOT / "worktrees/EXP-0007/vllm-plugin-FL"
GEMS = ROOT / "worktrees/EXP-0007/FlagGems"
PARENT = {
    "vllm-plugin-FL": "07d9fa0cfdc5f09db178e29508e3c0bcb5c1e908",
    "FlagGems": "484413ea0db18221dd9e63e742361d837df9ff13",
}
CODE = [
    "vllm_fl/platform.py", "vllm_fl/worker/kv_reservation_config.py",
    "vllm_fl/worker/kv_reservation_scheduler.py",
    "tests/standalone/validate_kv_reservation_cpu.py",
]
VALIDATIONS = [
    ROOT / "artifacts/EXP-0007/20261020_190653_cpu",
    ROOT / "artifacts/EXP-0007/20261020_190948_cpu_lifecycle",
]
FORBIDDEN = re.compile(
    rb"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|"
    rb"github_pat_[A-Za-z0-9_]{20,}|lab_inst_[A-Za-z0-9]+|ssh\.flagos\.io")


def git(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True,
        stderr=subprocess.STDOUT, timeout=120).strip()


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def publish(repo, remote, branch):
    sha = git(repo, "rev-parse", "HEAD")
    git(repo, "push", "-u", remote, branch)
    url = git(repo, "remote", "get-url", "--push", remote)
    actual = git(repo, "ls-remote", url, "refs/heads/" + branch).split()[0]
    assert actual == sha
    return sha


def main():
    assert git(PLUGIN, "branch", "--show-current") == "leanscale/iluvatar-kv-reservation"
    assert git(PLUGIN, "rev-parse", "HEAD") == PARENT["vllm-plugin-FL"]
    assert git(GEMS, "rev-parse", "HEAD") == PARENT["FlagGems"]
    assert not git(GEMS, "status", "--porcelain")
    assert not git(PLUGIN, "diff", "--cached", "--name-only")
    target = PLUGIN / CODE[-1]
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(EXP / "validate_admission_cpu.py", target)
    for p in [*(PLUGIN / name for name in CODE), *EXP.glob("*.py")]:
        assert not FORBIDDEN.search(p.read_bytes()), str(p)
        ast.parse(p.read_text(), filename=str(p))
    final = json.loads((VALIDATIONS[-1] / "results.json").read_text())
    assert final["status"] == "passed" and not final["cuda_initialized"]
    assert len(final["checks"]) == 13
    assert all(item["passed"] for item in final["checks"])
    gate_hashes = {}
    for source in VALIDATIONS:
        dest = EXP / "artifacts" / source.name
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("results.json", "validation.log"):
            raw = (source / name).read_text()
            raw = re.sub(r"uuid=[^,)]+", "uuid=[REDACTED]", raw)
            raw = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[IP_REDACTED]", raw)
            (dest / name).write_text(raw)
        gate_hashes[str(source / "results.json")] = digest(source / "results.json")
    git(PLUGIN, "add", "--", *CODE)
    git(PLUGIN, "diff", "--cached", "--check")
    changed = set(git(PLUGIN, "diff", "--cached", "--name-only").splitlines())
    assert changed == set(CODE), changed
    git(PLUGIN, "commit", "-m", "perf(iluvatar): reserve KV growth before request admission")
    candidate = dict(PARENT, **{"vllm-plugin-FL": publish(
        PLUGIN, "leanscale", "leanscale/iluvatar-kv-reservation")})
    write(EXP / "parent_commits.json", PARENT)
    write(EXP / "candidate_commits.json", candidate)
    write(EXP / "candidate_validation.json", {
        "status": "passed", "scope": "CPU allocator/lifecycle; actual service gate pending",
        "candidate_commits": candidate, "parent_commits": PARENT,
        "diagnostic_hashes": gate_hashes,
        "code_hashes": {
            "vllm-plugin-FL": {name: digest(PLUGIN / name) for name in CODE},
            "FlagGems": {},
        },
        "cuda_initialized": False, "full_model_accuracy": "deferred_by_user",
        "parent_numerics": "EXP-0006 candidate_validation.json; kernels unchanged",
        "source_index": {
            str(p.relative_to(Path("/usr/local/lib/python3.12/site-packages"))): digest(p)
            for p in [
                Path("/usr/local/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py"),
                Path("/usr/local/lib/python3.12/site-packages/vllm/v1/core/kv_cache_manager.py"),
                Path("/usr/local/lib/python3.12/site-packages/vllm/v1/core/single_type_kv_cache_manager.py"),
                Path("/usr/local/lib/python3.12/site-packages/vllm/config/scheduler.py"),
            ]
        },
    })
    (EXP / "STATUS.md").write_text(
        "# EXP-0007 状态\n\nCPU 验证 13 项通过，代码已冻结并推送。"
        "等待 EXP-0006 16K 完整结束及归档后，串行执行官方调度测评。"
        "尚无真实服务接入或性能结论，完整任务精度按用户要求暂缓。\n")
    # Stage only this experiment, never the live EXP-0006 publisher's files.
    files = [
        p for p in EXP.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and not p.name.startswith(".")
    ]
    for p in files:
        assert p.stat().st_size < 5_000_000
        assert not FORBIDDEN.search(p.read_bytes()), str(p)
    assert not git(DOCS, "diff", "--cached", "--name-only")
    git(DOCS, "add", "--", *(str(p.relative_to(DOCS)) for p in files))
    git(DOCS, "diff", "--cached", "--check")
    git(DOCS, "commit", "-m", "experiments: prepare validated KV admission candidate and serial official runs")
    docs_sha = publish(DOCS, "origin", "main")
    print(json.dumps({"candidate_commits": candidate, "docs_commit": docs_sha,
                      "remote_verified": True}), flush=True)


if __name__ == "__main__":
    main()
