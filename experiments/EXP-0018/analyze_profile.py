"""Aggregate Chrome trace events without treating nested CPU time as GPU time."""
from collections import Counter, defaultdict
import base64
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys


def read_trace(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return json.load(f)["traceEvents"]


def union_us(intervals):
    total, end = 0.0, None
    for a, b in sorted(intervals):
        if end is None or a > end:
            total += b - a
        elif b > end:
            total += b - end
        end = b if end is None else max(end, b)
    return total


def write_csv(path, rows, fields):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def category(name, shape_chain):
    """Conservative attribution: ambiguous compiled/Graph kernels stay grouped."""
    low = name.lower()
    for shapes in shape_chain:
        if isinstance(shapes, list) and len(shapes) >= 2:
            a, b = shapes[:2]
            if isinstance(a, list) and isinstance(b, list) and len(a) == len(b) == 2:
                k, n = a[-1], b[-1]
                if b[0] == k:
                    op = {(2048, 2560): "OP03_QKV", (2048, 2048): "OP08_O",
                          (2048, 12288): "OP09_gate_up", (6144, 2048): "OP11_down",
                          (2048, 130560): "OP12_LM_head"}.get((k, n))
                    if op:
                        return op, "linked_aten_mm_shape"
    if "_rope_kv" in low:
        return "OP04_RoPE_KV", "kernel_name"
    if re.match(r"^_attention(?:_|$)", name):
        return "OP05_06_GQA", "kernel_name_prefill_decode_not_separated"
    if re.match(r"^_reduce(?:_|$)", name):
        return "OP07_GQA_reduce", "kernel_name_verify_against_source"
    if any(x in low for x in ("rms_norm", "rmsnorm")):
        return "OP02_norm_or_fused_norm", "kernel_name"
    if "silu" in low or "swiglu" in low:
        return "OP10_SwiGLU_or_compiler_fusion", "kernel_name"
    if any(x in low for x in ("gemm", "matmul", "gemv")):
        return "GEMM_unresolved_projection", "kernel_name_no_shape_correlation"
    if any(x in low for x in ("softmax", "topk", "top_k", "multinomial", "exponential")):
        return "sampling_candidate", "kernel_name_requires_path_confirmation"
    return "unattributed", "no_reliable_semantic_mapping"


def analyze_trace(path, out):
    events = [e for e in read_trace(path) if e.get("ph") == "X" and e.get("dur", 0) >= 0]
    kernels = [e for e in events if "kernel" in e.get("cat", "").lower()]
    copies = [e for e in events if any(x in e.get("cat", "").lower()
                                     for x in ("gpu_memcpy", "gpu_memset"))]
    cpu = [e for e in events if any(x in e.get("cat", "").lower()
                   for x in ("cpu_op", "user_annotation", "cuda_runtime", "cuda_driver"))]
    # Build a per-thread interval tree. Only direct child spans are subtracted.
    threads = defaultdict(list)
    for i, e in enumerate(cpu):
        e["_i"] = i
        threads[(e.get("pid"), e.get("tid"))].append(e)
    parents, children = {}, defaultdict(list)
    for group in threads.values():
        stack = []
        for e in sorted(group, key=lambda e: (e["ts"], -e["dur"])):
            while stack and e["ts"] + e["dur"] > stack[-1]["ts"] + stack[-1]["dur"] + 0.001:
                stack.pop()
            if stack:
                parents[e["_i"]] = stack[-1]
                children[stack[-1]["_i"]].append((e["ts"], e["ts"] + e["dur"]))
            stack.append(e)

    def chain(event):
        while event is not None:
            yield event
            event = parents.get(event["_i"])

    step_meta = defaultdict(dict)
    for e in cpu:
        if e["name"].startswith(("LS_META:", "LS_META64:")):
            payload = e["name"].split(":", 1)[1]
            if e["name"].startswith("LS_META64:"):
                payload = base64.urlsafe_b64decode(payload).decode()
            meta = json.loads(payload)
            step_meta[int(meta["step"])].update(meta)
    runtime = {}
    external = {}
    for e in cpu:
        args = e.get("args", {})
        if "correlation" in args and ("runtime" in e.get("cat", "") or "driver" in e.get("cat", "")):
            runtime[args["correlation"]] = e
        if "External id" in args:
            external.setdefault(args["External id"], e)
    groups, semantic = defaultdict(list), defaultdict(list)
    correlations = Counter()
    per_kernel = []
    for e in kernels:
        args = e.get("args", {})
        ancestor = runtime.get(args.get("correlation"))
        if ancestor is None:
            ancestor = external.get(args.get("External id"))
        ancestors = list(chain(ancestor)) if ancestor is not None else []
        step = None
        for a in ancestors:
            if a["name"].startswith(("LS_STEP:", "LS_SAMPLE:")):
                step = int(a["name"].split(":", 1)[1])
                break
        phase = step_meta.get(step, {}).get("phase", "unlinked")
        shapes = []
        for a in ancestors:
            dims = a.get("args", {}).get("Input Dims")
            if a["name"] == "aten::linear" and isinstance(dims, list) and len(dims) >= 2:
                x, weight = dims[:2]
                if (isinstance(x, list) and isinstance(weight, list)
                        and len(x) == len(weight) == 2 and x[1] == weight[1]):
                    # F.linear stores weights [out_features, in_features].
                    shapes.append([x, [weight[1], weight[0]]])
            elif a["name"] in ("aten::mm", "aten::addmm", "aten::bmm"):
                shapes.append(dims)
        op, method = category(e["name"], shapes)
        if op == "OP12_LM_head" and any(a["name"] == "aten::linear" for a in ancestors):
            method = "linked_aten_linear_shape_transposed_weight"
        if any(a["name"].startswith("LS_SAMPLE:") for a in ancestors):
            op, method = "OP13_sampling", "linked_worker_sample_tokens_scope"
        elif e["name"] == "_attention":
            grid = args.get("grid")
            if isinstance(grid, list) and len(grid) == 3:
                if grid[2] == 16:
                    op, method = "OP06_GQA_decode_branch", "verified_fixed_S16_grid"
                elif grid[2] == 1:
                    op, method = "OP05_GQA_prefill_branch", "verified_fixed_S16_grid"

        groups[(phase, e["name"])].append(e["dur"])
        semantic[(phase, op, method)].append(e["dur"])
        correlations["linked_to_step" if step is not None else "unlinked_to_step"] += 1
        per_kernel.append({"phase": phase, "step": step, "name": e["name"], "op": op,
                           "mapping": method, "duration_us": e["dur"],
                           "grid": args.get("grid"), "block": args.get("block")})
    total_gpu = sum(e["dur"] for e in kernels)

    def rows_for(source, keys):
        rows = []
        for key, values in source.items():
            row = dict(zip(keys, key))
            row.update(calls=len(values), total_us=sum(values),
                       mean_us=statistics.mean(values), median_us=statistics.median(values),
                       p95_us=sorted(values)[max(0, int(len(values) * .95) - 1)],
                       percent_kernel_sum=sum(values) / total_gpu * 100 if total_gpu else 0)
            rows.append(row)
        return sorted(rows, key=lambda r: r["total_us"], reverse=True)
    kernel_rows = rows_for(groups, ["phase", "kernel"])
    semantic_rows = rows_for(semantic, ["phase", "operator", "mapping"])
    metrics = ["calls", "total_us", "mean_us", "median_us", "p95_us", "percent_kernel_sum"]
    write_csv(out / "kernel_times.csv", kernel_rows, ["phase", "kernel"] + metrics)
    write_csv(out / "operator_times.csv", semantic_rows, ["phase", "operator", "mapping"] + metrics)
    cpu_groups = defaultdict(lambda: [0, 0.0, 0.0])
    for e in cpu:
        if e.get("cat") != "cpu_op":
            continue
        key = (e["name"], json.dumps(e.get("args", {}).get("Input Dims", [])))
        row = cpu_groups[key]
        row[0] += 1
        row[1] += max(0, e["dur"] - union_us(children[e["_i"]]))
        row[2] += e["dur"]
    cpu_rows = [dict(operator=k[0], shapes=k[1], calls=v[0], self_cpu_us=v[1],
                     inclusive_cpu_us=v[2]) for k, v in cpu_groups.items()]
    cpu_rows.sort(key=lambda x: x["self_cpu_us"], reverse=True)
    write_csv(out / "cpu_operator_times.csv", cpu_rows,
              ["operator", "shapes", "calls", "self_cpu_us", "inclusive_cpu_us"])
    (out / "step_metadata.json").write_text(json.dumps(dict(step_meta), indent=2) + "\n")
    (out / "kernel_mapping.json").write_text(json.dumps(per_kernel, indent=2) + "\n")
    report = {
        "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "trace_bytes": path.stat().st_size, "kernel_events": len(kernels),
        "kernel_sum_us": total_gpu,
        "kernel_interval_union_us": union_us([(e["ts"], e["ts"] + e["dur"]) for e in kernels]),
        "copy_events": len(copies), "copy_sum_us": sum(e["dur"] for e in copies),
        "correlations": dict(correlations),
        "profiled_step_phases": dict(Counter(x.get("phase", "unknown") for x in step_meta.values())),
        "model_kernel_visibility": any("_rope_kv" in e["name"].lower() for e in kernels),
        "top_kernels": kernel_rows[:20], "top_operators": semantic_rows[:20],
        "notes": [
            "Kernel sum is aggregate GPU work; concurrent streams can overlap. It is not wall latency.",
            "CPU inclusive times overlap; use self_cpu_us and do not sum inclusive_cpu_us.",
            "Unlinked kernels are retained, never assigned a phase by GPU/CPU timestamp proximity.",
            "Graph replay may hide individual CPU operator shapes; unresolved GEMMs remain grouped.",
            "NONE pass helps attribution only and must not replace FULL_DECODE_ONLY timings.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(run):
    provenance = json.loads((run / "provenance.json").read_text())
    assert provenance["commits"] == {
        "vllm-plugin-FL": "588a6147cd291e272324ab6c812024cd587c6287",
        "FlagGems": "fd209df717648002ebf5ba4ec351424dcc997a53"}
    results, coverage = [], []
    for mode in ("FULL_DECODE_ONLY", "NONE"):
        for window in ("initial", "mixed", "large_decode", "tail"):
            folder = run / mode / "traces" / window
            traces = sorted(folder.glob("*.pt.trace.json*"))
            count = 0
            for i, trace in enumerate(traces):
                out = run / "summary" / mode / window / str(i)
                out.mkdir(parents=True, exist_ok=True)
                result = analyze_trace(trace, out)
                meta = json.loads((out / "step_metadata.json").read_text())
                qualifying = 0
                for step in meta.values():
                    if window == "initial": ok = "phase" in step
                    elif window == "mixed": ok = step.get("phase") == "mixed"
                    elif window == "large_decode": ok = step.get("phase") == "decode" and step.get("B", 0) >= 48
                    else: ok = step.get("main_requests_entered") == 256 and 0 < step.get("B", 0) <= 16
                    qualifying += bool(ok)
                count += qualifying
                result.update(mode=mode, window=window, index=i,
                              captured_steps=len(meta), qualifying_steps=qualifying,
                              batch_sizes=sorted({m["B"] for m in meta.values() if "B" in m}),
                              trace_slot=f"{mode}/{window}/{i}")
                (out / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
                results.append(result)
            coverage.append(dict(mode=mode, window=window, trace_count=len(traces),
                                 qualifying_steps=count,
                                 coverage="absent" if not traces else "no_qualifying_steps" if count == 0 else "partial" if count < 16 else "at_least_16_steps"))
    report = dict(diagnostic_only=True, performance_comparison_allowed=False,
                  rules=["aten::linear transposed weight", "sampling RPC ancestry", "fixed S16 GQA branch grid"],
                  traces=results, coverage=coverage,
                  device_trace_count=sum(x["kernel_events"] > 0 for x in results))
    probe = run / "probe" / "probe_result.json"
    report["profiler_capability"] = json.loads(probe.read_text()) if probe.exists() else None
    (run / "analysis.json").write_text(json.dumps(report, indent=2)+"\n")
    lines = ["# EXP-0018：选定版本 4K 分阶段热点诊断", "",
             "每个 pass 使用 4096 / 1024 / 64 / 256 一次完整请求集；带 profiler，不能作为官方吞吐成绩。",
             "窗口名仅表示触发目标，实际阶段和 batch size 来自逐步 metadata。缺失或不足 16 步的覆盖如实保留。",
             "CPU context 长度是异步 CPU 上界。GPU kernel 总时间可能跨 stream 重叠，不等于墙钟延迟。",
             "FULL Graph 内部可能不可见；NONE 仅辅助归属。不同窗口/模式不合并成整程占比。", "",
             "| 模式 | 窗口 | trace 数 | 满足目标的步数 | 覆盖 |",
             "|---|---|---:|---:|---|"]
    for c in coverage:
        lines.append(f"| {c['mode']} | {c['window']} | {c['trace_count']} | {c['qualifying_steps']} | {c['coverage']} |")
    for r in results:
        lines += ["", f"## {r['mode']} / {r['window']} / {r['index']}", "",
                  f"实际阶段 {r['profiled_step_phases']}；batch sizes {r['batch_sizes']}；满足目标 {r['qualifying_steps']}/{r['captured_steps']} 步。",
                  f"GPU 事件 {r['kernel_events']}，kernel 总和 {r['kernel_sum_us']/1000:.3f} ms，区间并集 {r['kernel_interval_union_us']/1000:.3f} ms。",
                  f"关联情况 {r['correlations']}；可见 RoPEKV 模型 kernel：{r['model_kernel_visibility']}。", "",
                  "| 实际阶段 | 算子 | 调用数 | GPU 总 ms | 此窗口可见 kernel 时间占比 |",
                  "|---|---|---:|---:|---:|"]
        for row in r['top_operators']:
            lines.append(f"| {row['phase']} | {row['operator']} | {row['calls']} | {row['total_us']/1000:.3f} | {row['percent_kernel_sum']:.2f}% |")
    if not results:
        lines += ["", "尚无可解析 trace，不能报告算子耗时。检查 status 和失败日志。"]
    (run / "RESULTS.md").write_text("\n".join(lines)+"\n")
    return report

if __name__ == "__main__":
    main(Path(sys.argv[1]))
