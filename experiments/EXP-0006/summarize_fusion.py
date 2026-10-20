"""Archive every diagnostic and propose candidates for a separate confirmation."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import statistics
import sys

run = Path(sys.argv[1])
exp = Path("/workspace/LeanScale/MiniCPM-Infer-Sys-Opt/experiments/EXP-0006")
dst = exp / "artifacts/diagnostics" / run.name
dst.mkdir(parents=True, exist_ok=True)
for p in run.iterdir():
    if p.is_file() and p.suffix in (".json", ".log", ".py"):
        shutil.copy2(p, dst/p.name)

lines = ["# EXP-0006 融合诊断结果", "",
         "以下为独立算子诊断，尚不是官方端到端收益。所有配置及失败保留在原始 JSON。"
         "候选还需复测和真实模型接入，不能把有限参数搜索称为全局最优。", ""]
proposals = {}
for kind, title in [("rope", "RoPE + KV cache 写入"),
                    ("mlp", "gate/up GEMM + SwiGLU"),
                    ("norm", "residual + RMSNorm + QKV GEMM")]:
    p = run / (kind + ".json")
    if not p.exists():
        previous = sorted((exp/"artifacts/diagnostics").glob("*/"+kind+".json"))
        if not previous:
            lines += [f"## {title}", "", "尚无结果。", ""]
            continue
        p = previous[-1]
    result = json.loads(p.read_text())
    lines += [f"## {title}", "", f"脚本状态：{result['status']}。数据来源：`{p.parent.name}`。"]
    if "error" in result:
        lines += ["", f"失败记录：`{result['error']}`。"]
    lines += ["", "| tokens | 基线前/后 µs | 配置 | 候选 µs | 局部比值 | 数值 |",
              "| --- | --- | --- | --- | --- | --- |"]
    rows = result.get("timings", [])
    for row in rows:
        before = row["baseline"]["median_us"]
        after = row["baseline_after"]["median_us"]
        for c in row["candidates"]:
            elapsed = c.get("median_us")
            ratio = ((before+after)/2/elapsed) if elapsed else None
            lines.append(f"| {row['n']} | {before:.2f}/{after:.2f} | `{c['config']}` | "
                         f"{f'{elapsed:.2f}' if elapsed else '未计时'} | "
                         f"{f'{ratio:.3f}×' if ratio else '—'} | "
                         f"{'通过' if c.get('numerical_passed', True) else '失败'} |")
    if kind == "rope" and result["status"] == "passed":
        groups = {}
        for group, subset in [("decode", [r for r in rows if r["n"] <= 128]),
                              ("prefill", [r for r in rows if r["n"] > 128])]:
            candidates = []
            for config in result["configs"]:
                speeds, drifts = [], []
                for row in subset:
                    c = next(c for c in row["candidates"] if c["config"] == config)
                    a, b = row["baseline"]["median_us"], row["baseline_after"]["median_us"]
                    speeds.append((a+b)/2/c["median_us"])
                    drifts.append(abs(a-b)/((a+b)/2))
                if speeds:
                    candidates.append(dict(config=config, geometric_speedup=statistics.geometric_mean(speeds),
                                           minimum_speedup=min(speeds), max_baseline_drift=max(drifts)))
            candidates.sort(key=lambda c: c["geometric_speedup"], reverse=True)
            groups[group] = candidates
        proposals[kind] = groups
    lines += [""]
lines += ["## 后续门槛", "",
          "对候选复测；基线前后漂移大于 5% 的结果不能直接用于定版。"
          "无收益或数值失败的融合不启用。RoPE+KV 必须在真实模型观察到 42 层融合、"
          "原 KV update 不重复执行、Graph 与 fallback 正常，然后才进行完整官方两组测评。",
          "系统调度及 KV 压缩在融合定版之后开展；KV 压缩需要完整上下文精度对照、"
          "官方 MATH500 Level 3 和实际触发压缩的长文任务。", ""]
(exp/"DIAGNOSTICS.md").write_text("\n".join(lines))
(exp/"selection_candidates.json").write_text(json.dumps(
    dict(status="requires_confirmation_not_final", run=str(run), proposals=proposals), indent=2)+"\n")
(exp/"artifact_manifest.json").write_text(json.dumps({
    str(p.relative_to(exp)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted((exp/"artifacts").rglob("*")) if p.is_file()
}, indent=2)+"\n")
with (exp/"STEPS.md").open("a") as f:
    f.write("\n- "+datetime.datetime.now().astimezone().isoformat()+
            "：归档融合微基准全部配置、失败和候选排序，尚待复测与整模接入。\n")
print(json.dumps({"archived": str(dst), "selection": str(exp/"selection_candidates.json")}))
