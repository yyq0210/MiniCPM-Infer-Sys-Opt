"""Freeze the graph-side-effect fix and checkpoint-aware official runner."""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import subprocess

root = Path("/workspace/LeanScale")
repo = root/"MiniCPM-Infer-Sys-Opt"
exp = repo/"experiments/EXP-0006"
wt = root/"worktrees/EXP-0006"
st = root/"staging/EXP-0006"
lock = open("/workspace/experiments/codex_20261019/runner.lock", "a+")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def git(folder, *args):
    return subprocess.check_output(["git", "-C", str(folder), *args], text=True,
                                   stderr=subprocess.STDOUT, timeout=180).strip()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n")


prior = root/"artifacts/EXP-0006/20261020_163052_official"
assert json.loads((prior/"completion_handler_status.json").read_text())["status"] == "pushed"
regression = root/"artifacts/EXP-0006/20261020_164258_dce_regression"
assert json.loads((regression/"before.json").read_text())["status"] == "failed"
assert json.loads((regression/"after.json").read_text())["status"] == "passed"
for name, dest in (
    ("rope_kv_fusion.py", "vllm_fl/compilation/rope_kv_fusion.py"),
    ("rope_kv_dispatch.py", "vllm_fl/ops/rope_kv_dispatch.py"),
):
    (wt/"vllm-plugin-FL"/dest).write_bytes((st/name).read_bytes())
run = root/"artifacts/EXP-0006"/datetime.datetime.now().strftime("%Y%m%d_%H%M%S_cpu_final")
run.mkdir()
env = dict(os.environ, VLLM_PLUGINS="fl", PYTHONUNBUFFERED="1",
           PYTHONPATH=str(wt/"vllm-plugin-FL")+":"+str(wt/"FlagGems/src"))
shutil.copy2(st/"validate_graph_cpu.py", run/"validate_graph_cpu.py")
with (run/"validation.log").open("w") as log:
    subprocess.run(["python3", str(run/"validate_graph_cpu.py"), str(run/"validation.json")],
                   env=env, cwd=run, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
assert len(json.loads((run/"validation.json").read_text())["checks"]) == 10
plugin = wt/"vllm-plugin-FL"
assert set(git(plugin, "diff", "--name-only").splitlines()) == {
    "vllm_fl/compilation/rope_kv_fusion.py", "vllm_fl/ops/rope_kv_dispatch.py"}
git(plugin, "diff", "--check")
git(plugin, "add", "--", "vllm_fl/compilation/rope_kv_fusion.py", "vllm_fl/ops/rope_kv_dispatch.py")
git(plugin, "commit", "-m", "fix(iluvatar): preserve attention writes in prepartition fusion")
git(plugin, "push", "leanscale", "leanscale/iluvatar-fusion")
sha = git(plugin, "rev-parse", "HEAD")
assert git(plugin, "ls-remote", git(plugin, "remote", "get-url", "--push", "leanscale"),
           "refs/heads/leanscale/iluvatar-fusion").split()[0] == sha
commits = json.loads((exp/"candidate_commits.json").read_text())
commits["vllm-plugin-FL"] = sha
write(exp/"candidate_commits.json", commits)
gate = json.loads((exp/"candidate_validation.json").read_text())
gate["candidate_commits"] = commits
gate["diagnostic_hashes"][str(run/"validation.json")] = hashlib.sha256(
    (run/"validation.json").read_bytes()).hexdigest()
for name, files in gate["code_hashes"].items():
    for p in files:
        files[p] = hashlib.sha256((wt/name/p).read_bytes()).hexdigest()
write(exp/"candidate_validation.json", gate)
for source in (regression, run):
    dest = exp/"artifacts/cpu"/source.name
    dest.mkdir(parents=True, exist_ok=True)
    for p in source.iterdir():
        if p.is_file() and p.suffix in (".json", ".log", ".py"):
            shutil.copy2(p, dest/p.name)
for name in ("validate_graph_cpu.py", "run_split_e2e.py", "finish_split.py",
             "launch_split.py", "prepare_split.py"):
    shutil.copy2(st/name, exp/name)
stamp = datetime.datetime.now().astimezone().isoformat()
with (exp/"STEPS.md").open("a") as f:
    f.write("\n- "+stamp+"：packet + 仅输出 out buffer 的 CPU 回归确实复现 DCE 缺陷：原版失败，"
            "删除全局 DCE 后 10 项通过；注意力写入节点必须保留。日志使用已配置 vLLM namespace，"
            "重新冻结插件 "+sha+"。首次真机启动不计为测评；失败日志已归档。"
            "后续使用相同官方参数的独立 4K/16K 完整组，17:50 硬截止、18:30 后接续。\n")
with (exp/"FUSION_DECISION.md").open("a") as f:
    f.write("\n## 实际图接入发现的副作用缺陷\n\n"
            "首次完整模型启动触发接入门槛失败，没有进行 benchmark。"
            "预分区的 FX 图仍是 functionalization 之前的图；通用 eliminate_dead_code "
            "不能可靠识别 OpOverloadPacket 形式注意力的 output 缓冲区写入，"
            "导致注意力及其 RoPE/GQA 输入链被删除。"
            "修复仅保留显式匹配节点替换，不做全局 DCE；后续标准编译流程负责清理视图。"
            "新增 packet + model-output-only 图能让原版失败、修复版通过，"
            "最终 10 项 CPU 检查通过；仍需重新完成真实 42 层接入验证。\n")
write(exp/"shutdown_resume.json", dict(
    shutdown_at="2026-10-20T18:00:00+08:00", restart_expected_at="2026-10-20T18:30:00+08:00",
    gpu_finish_deadline="2026-10-20T17:50:00+08:00", candidate_commits=commits,
    before_shutdown_case="4k", after_restart_case="16k",
    partial_case_policy="discard_from_official_summary_and_rerun_all_four",
    server_data_persistence="verify_on_reconnect", completed_cases=[]))
(exp/"SHUTDOWN_RESUME.md").write_text(
    "# 停机与接续说明\n\n"
    "用户计划北京时间 2026-10-20 18:00 关闭机器，18:30 重启。17:50 前结束 GPU 工作，留十分钟归档推送。\n\n"
    "本次先跑完整 4K 组，16K 在重启后另启服务跑完整四轮。所有负载参数、原 benchmark 脚本和 serve 参数保持不变；"
    "每组舍弃首轮。两个服务会话必须分别记录，整合前校验相同代码提交与配置。"
    "部分轮次只能作为诊断；重启后重新跑该组四轮，不能和前半段拼接。\n\n"
    "启动脚本会拒绝停机窗口内运行、拒绝重复协调进程和未完成发布；"
    "当前 deadline 会在启动等待或 benchmark 超时后停止子进程及服务，结果先落盘再尝试推送。"
    "重启后先确认 GPU、模型文件、仓库和提交仍在，检查最新 receipt 与发布状态。\n\n"
    "重启后先读取 `/workspace/LeanScale/artifacts/EXP-0006-latest-4k.json`；"
    "只有其运行 `status.json` 为 `performance_case_complete` 且归档 `completion_validation.json` 通过，才保留该组。\n\n"
    "在 18:30 之后确认服务恢复后运行：\n\n"
    "```bash\npython3 /workspace/LeanScale/MiniCPM-Infer-Sys-Opt/experiments/EXP-0006/launch_split.py --case 16k\n```\n\n"
    "若 4K 未完成则先将 case 改成 `4k`；当前 launcher 在停机前还要求 `--finish-before 2026-10-20T17:50:00+08:00`。"
    "每个运行的独立 finisher 会保存 CSV、日志、命令、哈希、接入证据并推送。"
    "4K/16K 分别写入 RESULTS_4096.md / RESULTS_16384.md；两组完成后再汇总官方整体结果。"
    "不要把单组完成、接入失败或局部微基准称为整个实验完成。\n\n"
    f"当前冻结提交：`{commits}`。\n")
assert not git(repo, "diff", "--cached", "--name-only")
paths = [p for p in exp.rglob("*") if p.is_file() and "__pycache__" not in p.parts
         and p.name != ".finish.lock"]
for p in paths:
    assert p.stat().st_size < 20_000_000
git(repo, "add", "--", *[str(p.relative_to(repo)) for p in paths])
git(repo, "diff", "--cached", "--check")
git(repo, "commit", "-m", "experiments: fix graph integration and checkpoint official cases around shutdown")
git(repo, "push", "origin", "main")
head = git(repo, "rev-parse", "HEAD")
assert git(repo, "ls-remote", git(repo, "remote", "get-url", "--push", "origin"),
           "refs/heads/main").split()[0] == head
lock.close()
print(json.dumps(dict(plugin_commit=sha, docs_commit=head, cpu_run=str(run))), flush=True)
subprocess.run(["python3", str(exp/"launch_split.py"), "--case", "4k",
                "--finish-before", "2026-10-20T17:50:00+08:00"], check=True)
