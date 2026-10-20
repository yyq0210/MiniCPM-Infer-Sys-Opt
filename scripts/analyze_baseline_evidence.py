#!/usr/bin/env python3
"""Recompute baseline statistics from archived CSVs; no GPU or network access."""
from pathlib import Path
import csv
import hashlib
import json
import re
import statistics
import sys


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main(root):
    result = {"aggregation": "four rounds; discard first; arithmetic mean of last three",
              "baseline_sessions": 2, "candidate_complete": False}
    for case, path in [
        ("4k", root / "results/original_4k/raw_runs_20261019_190747.csv"),
        ("16k", root / "results/baseline_16k/benchmark_results/raw_runs_20261019_203029.csv"),
    ]:
        raw = rows(path)
        assert len(raw) == 4
        for row in raw:
            assert row["Run Status"] == "SUCCESS"
            n, length, output = (int(row[k]) for k in ("Num Prompts", "Prefill", "Decode"))
            assert float(row["Successful Requests"]) == n
            assert float(row["Total Input Tokens"]) == n * length
            assert float(row["Total Output Tokens"]) == n * output
        kept = raw[1:]
        keys = ["Benchmark Duration (s)", "Total tok/s", "Output tok/s",
                "Mean TTFT (ms)", "P99 TTFT (ms)", "Mean TPOT (ms)", "P99 ITL (ms)"]
        means = {k: statistics.mean(float(r[k]) for r in kept) for k in keys}
        summary_path = path.with_name(path.name.replace("raw_runs_", "summary_"))
        summary = rows(summary_path)
        assert len(summary) == 1
        for key, value in means.items():
            assert abs(float(summary[0][key]) - value) <= 0.011, (case, key, value)
        throughput = [float(r["Total tok/s"]) for r in kept]
        result[case] = {
            "raw_file": str(path.relative_to(root)),
            "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "complete_rounds": 4, "means": means,
            "throughput_sample_cv_percent": statistics.stdev(throughput) / statistics.mean(throughput) * 100,
            "rounds": raw,
        }
    candidate = rows(root / "results/candidate_benchmark/benchmark_results/raw_runs_20261019_231826.csv")
    result["candidate_completed_rounds"] = len(candidate)
    assert len(candidate) == 3
    result["candidate_retained_rounds_provisional_mean_total_tok_s"] = statistics.mean(
        float(r["Total tok/s"]) for r in candidate[1:])
    logs = json.loads((root / "runtime_log_evidence.json").read_text())["baseline16k"]["selected_lines"]
    pattern = r"Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%, Prefix cache hit rate: ([\d.]+)%"
    samples = [tuple(map(float, m.groups())) for line in logs if (m := re.search(pattern, line))]
    result["log_samples"] = {}
    for name, group in [
        ("active", [s for s in samples if s[0] + s[1] > 0]),
        ("waiting", [s for s in samples if s[1] > 0]),
        ("waiting_and_kv_at_least_90_percent", [s for s in samples if s[1] > 0 and s[2] >= 90]),
    ]:
        result["log_samples"][name] = {
            "count": len(group),
            "median_running_waiting_kv_percent_prefix_percent":
                [statistics.median(s[i] for s in group) for i in range(4)],
            "min_running": min(s[0] for s in group),
            "max_running": max(s[0] for s in group),
        }
    result["log_samples_note"] = "Medians over periodic log samples, not per-request or exact time-weighted statistics."
    result["capacity"] = {
        "bytes_per_token": 42 * 2 * 2 * 128 * 2,
        "baseline_token_slots": 521936,
        "baseline_raw_kv_gib": 521936 * 43008 / 2**30,
        "16k_plus_1k_request_mib": 17408 * 43008 / 2**20,
        "64_requests_raw_kv_gib": 64 * 17408 * 43008 / 2**30,
        "ideal_token_budget_per_request_at_concurrency64": 521936 / 64,
        "max_prompt_retention_if_full_1k_output_retained": (521936 / 64 - 1024) / 16384,
    }
    (root / "analysis_statistics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["candidate_completed_rounds", "log_samples", "capacity"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent)
