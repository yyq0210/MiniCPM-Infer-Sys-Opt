"""Record confirmed selection, freeze fork commits, then start engine/official gates."""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import re
import shutil
import statistics
import subprocess
import time

root = Path("/workspace/LeanScale")
repo = root/"MiniCPM-Infer-Sys-Opt"
exp = repo/"experiments/EXP-0006"
staging = root/"staging/EXP-0006"
wt = root/"worktrees/EXP-0006"
first = root/"artifacts/EXP-0006/20261020_145132_diagnostics"
corrected = root/"artifacts/EXP-0006/20261020_160859_diagnostics"
confirmation = root/"artifacts/EXP-0006/20261020_161133_confirmation"
selected = sorted((root/"artifacts/EXP-0006").glob("*_selected"))[-1]


def git(folder, *args):
    return subprocess.check_output(["git", "-C", str(folder), *args], text=True,
                                   stderr=subprocess.STDOUT, timeout=180).strip()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False)+"\n")


lock = open("/workspace/experiments/codex_20261019/runner.lock", "a+")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert json.loads((corrected/"rope.json").read_text())["status"] == "passed"
assert len(json.loads((corrected/"rope.json").read_text())["checks"]) == 49
conf = json.loads((confirmation/"confirmation.json").read_text())
assert conf["status"] == "passed" and len(conf["rounds"]) == 36
unstable = {r["n"] for r in conf["rounds"]
            if abs(r["baseline"]["median_us"]-r["baseline_after"]["median_us"])/
            ((r["baseline"]["median_us"]+r["baseline_after"]["median_us"])/2) >= .05}
recheck = None
if unstable:
    recheck = sorted((root/"artifacts/EXP-0006").glob("*_confirmation_recheck"))[-1]
    fresh = json.loads((recheck/"confirmation.json").read_text())
    assert fresh["status"] == "passed" and len(fresh["rounds"]) == 3*len(unstable)
    assert {r["n"] for r in fresh["rounds"]} == unstable
    conf["rounds"] = [r for r in conf["rounds"] if r["n"] not in unstable]+fresh["rounds"]
sel = json.loads((selected/"selected.json").read_text())
assert sel["status"] == "passed" and len(sel["checks"]) == 28
kernel = wt/"FlagGems/src/flag_gems/runtime/backend/_iluvatar/fused/rope_kv_cache.py"
assert sel["source_sha256"] == hashlib.sha256(kernel.read_bytes()).hexdigest()
assert json.loads((corrected/"completion_handler_status.json").read_text())["status"] == "pushed"

stamp = datetime.datetime.now().astimezone().isoformat()
with (exp/"STEPS.md").open("a") as f:
    f.write("\n- "+stamp+"：49 项内核检查、48 组确认数值检查、36 轮交错计时、28 个默认策略边界/Graph 检查完成；"
            "冻结 RoPE+KV 候选。gate/up+SwiGLU 与 RMSNorm+QKV 的所有已测配置均慢于现有 vendor GEMM，不接入。"
            "官方测评前必须实际观察 42 层图融合、42 层运行调用、35 个 FULL decode 图和 API 请求成功；失败自动停止并归档。\n")

commits = {}
for name in ("FlagGems", "vllm-plugin-FL"):
    path = wt/name
    assert git(path, "branch", "--show-current") == "leanscale/iluvatar-fusion"
    assert not git(path, "diff", "--cached", "--name-only")
    changed = git(path, "diff", "--name-only").splitlines()
    if name == "FlagGems":
        if changed:
            assert changed == ["src/flag_gems/runtime/backend/_iluvatar/fused/rope_kv_cache.py"], changed
            git(path, "diff", "--check")
            git(path, "add", "--", changed[0])
            git(path, "commit", "-m", "perf(iluvatar): select confirmed RoPE cache fusion launch shapes")
        assert not git(path, "status", "--porcelain")
    else:
        assert not git(path, "status", "--porcelain")
    git(path, "push", "leanscale", "leanscale/iluvatar-fusion")
    commits[name] = git(path, "rev-parse", "HEAD")
    remote = git(path, "remote", "get-url", "--push", "leanscale")
    assert git(path, "ls-remote", remote, "refs/heads/leanscale/iluvatar-fusion").split()[0] == commits[name]

write(exp/"candidate_commits.json", commits)
sources = {
    "FlagGems": ["src/flag_gems/runtime/backend/_iluvatar/fused/"+n for n in (
        "rope_kv_cache.py", "rotary_embedding_parallel.py", "paged_gqa.py")],
    "vllm-plugin-FL": ["vllm_fl/__init__.py", "vllm_fl/compilation/rope_kv_fusion.py",
                     "vllm_fl/ops/rope_kv_dispatch.py",
                     "vllm_fl/dispatch/backends/flaggems/impl/iluvatar_gqa.py"],
}
gate = dict(status="passed", candidate_commits=commits,
            scope="kernel and graph-transform diagnostics; actual model route checked before benchmark",
            diagnostic_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (corrected/"rope.json", confirmation/"confirmation.json",
                                         selected/"selected.json")},
            code_hashes={name: {p: hashlib.sha256((wt/name/p).read_bytes()).hexdigest()
                               for p in paths} for name, paths in sources.items()})
write(exp/"candidate_validation.json", gate)
if recheck:
    p = recheck/"confirmation.json"
    gate["diagnostic_hashes"][str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    write(exp/"candidate_validation.json", gate)
for source in (confirmation, selected, *([recheck] if recheck else [])):
    dest = exp/"artifacts/confirmation"/source.name
    dest.mkdir(parents=True, exist_ok=True)
    for p in source.iterdir():
        if p.is_file() and p.suffix in (".json", ".py", ".log"):
            shutil.copy2(p, dest/p.name)
for name in ("confirm_rope.py", "validate_selected.py", "run_optimized_e2e.py", "finish_official.py",
             "freeze_fusion.py"):
    shutil.copy2(staging/name, exp/name)

selection = dict(status="microbenchmark_confirmed_engine_gate_pending",
                 original_confirmation=str(confirmation),
                 drift_rechecks={str(n): str(recheck) for n in unstable},
                 rule=[dict(max_tokens=32, head_tile=4, token_tile=1),
                       dict(max_tokens=128, head_tile=16, token_tile=1),
                       dict(max_tokens=None, head_tile=4, token_tile=4)],
                 num_warps=4, rows=[], rejected=[])
lines = ["# EXP-0006 融合选择与前后思路", "", f"更新：{stamp}。", "",
         "当前选择是已测候选中的 RoPE+KV 写入融合，不代表所有可能融合中的全局最优。"
         "以下数字是独立算子微基准；整模收益须等待完整官方测评。", "",
         "## 为什么选择这个边界", "",
         "原来：QKV 投影后，RoPE 产生 Q/K，再启动独立 cache update 读取 K/V 并写入分页缓存。"
         "现在：同一个 Triton kernel 旋转 Q/K，同时将旋转后的 K 和原 V 写入分页缓存。"
         "保留 Q/K 独立输出与未改写的 QKV 投影存储，保留原有注意力依赖；少一次 kernel 启动和一次 K 读取。"
         "K 输出仍然物化，不能宣称完全消除了中间张量。GQA 沿用 EXP-0005。", "",
         "在 vLLM 图拆分前匹配 RoPE→KV update，以函数式 Q/K 和第三个依赖输出替换；"
         "不支持的 shape、类型、缓存方式或算子策略执行原始 RoPE 与一次 KV update。"
         "真实 42 层接入和 FULL_DECODE_ONLY 图捕获在正式性能前强制验证。", "",
         "## 配置确认", "",
         "每个 N 用四个候选乱序执行三轮；每轮基线在候选前后各测一次，CUDA graph 内计时。"
         "N=32 有一轮基线漂移 5.33%，冻结门槛拒绝采用；原数据保留，单独重测该点三轮后用于表格。"
         "表中取各轮中位数。N≤32 使用 BH4/BT1，33–128 使用 BH16/BT1，N>128 使用 BH4/BT4，warps=4。"
         "N=129 的 BH8/BT1 比所选方案快约 0.5%，采用简单分段，避免对单点微小差异拟合。", "",
         "| N | 原 RoPE+写缓存 µs | 所选融合 µs | 局部加速 | 最大基线漂移 |",
         "| ---: | ---: | ---: | ---: | ---: |"]
for n in sorted({r["n"] for r in conf["rounds"]}):
    rows = [r for r in conf["rounds"] if r["n"] == n]
    bh, bt = (4, 1) if n <= 32 else ((16, 1) if n <= 128 else (4, 4))
    config = dict(head_tile=bh, token_tile=bt, num_warps=4)
    a = statistics.median((r["baseline"]["median_us"]+r["baseline_after"]["median_us"])/2 for r in rows)
    b = statistics.median(next(c["median_us"] for c in r["candidates"] if c["config"] == config) for r in rows)
    drift = max(abs(r["baseline"]["median_us"]-r["baseline_after"]["median_us"])/
                ((r["baseline"]["median_us"]+r["baseline_after"]["median_us"])/2) for r in rows)
    assert drift < .05, (n, drift)
    selection["rows"].append(dict(n=n, config=config, baseline_us=a, fused_us=b, speedup=a/b,
                                  max_baseline_drift=drift))
    lines.append(f"| {n} | {a:.3f} | {b:.3f} | {a/b:.3f}× | {100*drift:.2f}% |")
lines += ["", "## 淘汰的大粒度融合", "",
          "两个原型的固定数值门槛均通过，但所有已测配置都慢于现有 vendor GEMM 加独立前后处理。"
          "融合减少启动和访存，并不能补偿矩阵乘法实现效率的损失。"
          "RMSNorm 融合在每个输出 tile 重算归约、gate/up 融合维护两套累加器；"
          "这些是代码结构上的开销解释，尚未用硬件 profiler 分解归因。", "",
          "| 融合 | N | 现有实现 µs | 最快原型 µs | 比值（>1 才加速） |",
          "| --- | ---: | ---: | ---: | ---: |"]
for kind, name in (("mlp", "gate/up GEMM + SwiGLU"), ("norm", "residual + RMSNorm + QKV GEMM")):
    result = json.loads((first/(kind+".json")).read_text())
    assert result["status"] == "passed"
    for row in result["timings"]:
        a = (row["baseline"]["median_us"]+row["baseline_after"]["median_us"])/2
        c = min(row["candidates"], key=lambda c: c["median_us"])
        assert a/c["median_us"] < 1
        lines.append(f"| {name} | {row['n']} | {a:.2f} | {c['median_us']:.2f} | {a/c['median_us']:.3f}× |")
    selection["rejected"].append(dict(kind=kind, reason="all_measured_configurations_slower",
                                       source_run=str(first)))
lines += ["", "## 失败与验证边界", "",
          "首轮 RoPE+KV 在 stride=2 slot 映射下报缓存差异。定位发现诊断给旧 vLLM 写入内核传了非连续映射，"
          "而它只按 SLOT+token_idx 读取；这是参考测试缺陷。修正参考输入并加入独立 torch 索引写入核验后，"
          "融合计算内核未变即通过 49 项检查。计时双方都使用连续、全部有效且不重复的真实 slot 映射，复制不进入计时。",
          "随后 48 组候选逐位检查、36 轮交错计时和 28 个默认策略边界检查通过。"
          "覆盖 BF16/FP16、NeoX/交错 RoPE、NHD/HND、负 slot、短映射、原投影不变、Graph 输入变化、"
          "超过 2^31 元素偏移；边界检查另覆盖空输入和 N=32/33、128/129。",
          "按用户要求暂缓完整 MATH500 精度；以上算子数值检查不等于模型质量达标。KV 压缩仍必须做精度对照。",
          "", "## 下一步与计量方式", "",
          "使用未修改的官方 serve 和两组 benchmark 参数，各四轮舍弃第一轮；"
          "同时对照原始基线和 EXP-0005 完整结果。后者用于观察融合增量，"
          "但这是不同时段服务会话；EXP-0005 的 4K 波动 CV=7.73%，小幅变化不能据此归因。"
          "引擎接入失败、部分轮次或未通过完整性检查的结果不报告为官方成功。",
          "融合定版后才开始系统调度和 KV 压缩实验。压缩需要真正回收物理缓存、"
          "官方 MATH500 Level 3 与实际触发压缩的长文质量对照。", ""]
(exp/"FUSION_DECISION.md").write_text("\n".join(lines))
write(exp/"selected_candidate.json", selection)
assert not git(repo, "diff", "--cached", "--name-only")
paths = [p for p in exp.rglob("*") if p.is_file() and p.name != ".finish.lock"
         and "__pycache__" not in p.parts]
for p in paths:
    assert p.stat().st_size < 20_000_000, p
    assert not re.search(rb"-----BEGIN (?:OPENSSH|RSA) PRIVATE KEY-----|\b(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b", p.read_bytes()), p
git(repo, "add", "--", *[str(p.relative_to(repo)) for p in paths])
git(repo, "diff", "--cached", "--check")
git(repo, "commit", "-m", "experiments: confirm fusion choice and record rejected GEMM candidates")
git(repo, "push", "origin", "main")
head = git(repo, "rev-parse", "HEAD")
assert git(repo, "ls-remote", git(repo, "remote", "get-url", "--push", "origin"),
           "refs/heads/main").split()[0] == head
fcntl.flock(lock, fcntl.LOCK_UN)
lock.close()

run = root/"artifacts/EXP-0006"/datetime.datetime.now().strftime("%Y%m%d_%H%M%S_official")
log = (root/"artifacts/EXP-0006-official-controller.log").open("a")
command = ["python3", str(exp/"run_optimized_e2e.py"), "--baseline-from",
           str(root/"artifacts/EXP-0004/20261019_202342"), "--run-dir", str(run)]
p = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
for _ in range(100):
    if (run/"status.json").exists():
        break
    if p.poll() is not None:
        raise RuntimeError("Official coordinator exited before receipt")
    time.sleep(.2)
else:
    raise RuntimeError("No official launch receipt")
receipt = json.loads((run/"status.json").read_text())
write(root/"artifacts/EXP-0006-latest-official.json", receipt)
log2 = (root/"artifacts/EXP-0006-official-completion.log").open("a")
finisher = subprocess.Popen(["python3", str(exp/"finish_official.py"), "--run-dir", str(run),
                             "--publish"], stdout=log2, stderr=subprocess.STDOUT,
                            start_new_session=True)
print(json.dumps(dict(run_directory=str(run), controller_pid=p.pid,
                      finisher_pid=finisher.pid, commits=commits, docs_commit=head)), flush=True)
