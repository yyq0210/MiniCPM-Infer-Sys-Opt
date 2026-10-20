"""Aggregate Chrome trace events without treating nested CPU time as GPU time."""
from collections import Counter, defaultdict
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
        writer = csv.DictWriter(f, fieldnames=fields)
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
        if e["name"].startswith("LS_META:"):
            meta = json.loads(e["name"].split(":", 1)[1])
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
        shapes = [a.get("args", {}).get("Input Dims") for a in ancestors
                  if a["name"] in ("aten::mm", "aten::addmm", "aten::bmm")]
        op, method = category(e["name"], shapes)
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
    results = []
    for mode in ("FULL_DECODE_ONLY", "NONE"):
        for window in ("initial", "steady", "late"):
            folder = run / mode / "traces" / window
            for i, trace in enumerate(sorted(folder.glob("*.pt.trace.json*"))):
                out = run / "summary" / mode / window / str(i)
                out.mkdir(parents=True, exist_ok=True)
                result = analyze_trace(trace, out)
                result.update(mode=mode, window=window, index=i,
                              trace_relative=str(trace.relative_to(run)))
                results.append(result)
    report = {"traces": results, "device_trace_count": sum(x["kernel_events"] > 0 for x in results)}
    probe = run / "probe" / "probe_result.json"
    report["profiler_capability"] = json.loads(probe.read_text()) if probe.exists() else None
    (run / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# EXP-0008：4K 算子 profiling 结果", "",
             "输入 4096 / 输出 1024 / 并发上限 64 / 请求 256；每个 pass 一次完整负载。",
             "这是带 profiler 的诊断实验，不是官方吞吐成绩；各窗口与 Graph 模式单独汇总。",
             "kernel 总时间是设备工作量之和，可能跨 stream 重叠，不等于请求延迟。",
             "CPU inclusive 时间不可相加。未关联 kernel 保留 unlinked，不按相近时间戳猜测阶段。", ""]
    if report["profiler_capability"]:
        lines += [f"设备采集能力检查：`{json.dumps(report['profiler_capability'], ensure_ascii=False)}`。", ""]
    for r in results:
        lines += [f"## {r['mode']} / {r['window']} / trace {r['index']}", "",
                  f"设备事件 {r['kernel_events']}；kernel 总计 {r['kernel_sum_us']/1000:.3f} ms；"
                  f"设备区间并集 {r['kernel_interval_union_us']/1000:.3f} ms。",
                  f"关联情况：`{json.dumps(r['correlations'])}`；"
                  f"实际阶段：`{json.dumps(r['profiled_step_phases'])}`。", "",
                  f"观察到模型 RoPE+KV kernel：`{r['model_kernel_visibility']}`。"
                  "若为 false，本窗口不得用于声称完整模型算子覆盖。", "",
                  "| 阶段 | 算子/候选归属 | 调用数 | GPU 总耗时 ms | kernel 时间占比 |",
                  "|---|---|---:|---:|---:|"]
        for row in r["top_operators"]:
            lines.append(f"| {row['phase']} | {row['operator']} | {row['calls']} | "
                         f"{row['total_us']/1000:.3f} | {row['percent_kernel_sum']:.2f}% |")
        lines += ["", "归属方法、均值、中位数、P95 和原始 kernel 名见对应 CSV。", ""]
    if not results:
        lines += ["尚无可解析 trace；检查 STATUS.json 与本地日志，不能据此报告算子耗时。"]
    (run / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return report


if __name__ == "__main__":
    main(Path(sys.argv[1]))
