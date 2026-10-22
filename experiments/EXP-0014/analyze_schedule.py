"""Summarize observed scheduler counters; preserve the all-four-round scope."""
import json
import re
from pathlib import Path

COUNTERS = (
    "num_preemptions_total",
    "request_queue_time_seconds_count",
    "request_queue_time_seconds_sum",
)


def counters(path):
    values = {}
    for line in path.read_text().splitlines():
        match = re.match(r'^vllm:(\w+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$', line)
        if match and match[1] in COUNTERS:
            values[match[1]] = values.get(match[1], 0) + float(match[2])
    missing = set(COUNTERS) - values.keys()
    if missing:
        raise ValueError(f"Missing counters in {path.name}: {sorted(missing)}")
    return values


def summarize(folder):
    folder = Path(folder)
    before = counters(folder / "metrics_before.prom")
    after = counters(folder / "metrics_after.prom")
    delta = {k: after[k] - before[k] for k in COUNTERS}
    if any(v < 0 for v in delta.values()):
        raise ValueError("Counter reset within the same service")
    count = delta["request_queue_time_seconds_count"]
    log = (folder / "server.log").read_text(errors="replace")
    samples = [
        (int(a), int(b), float(c)) for a, b, c in re.findall(
            r"Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%",
            log)
    ]
    admissions = [
        json.loads(item) for item in re.findall(
            r"LeanScale KV admission stats (\{[^\n]*\})", log)
    ]
    return {
        "counter_delta": delta,
        "mean_queue_seconds_all_four_rounds": (
            delta["request_queue_time_seconds_sum"] / count if count else None),
        "server_log_samples": len(samples),
        "peak_running_sampled": max((r[0] for r in samples), default=None),
        "peak_waiting_sampled": max((r[1] for r in samples), default=None),
        "peak_cache_usage_percent_sampled": max((r[2] for r in samples), default=None),
        "admission_last_snapshot": admissions[-1] if admissions else None,
        "scope": (
            "Metric deltas cover all four official rounds, including discarded warmup. "
            "Server concurrency/cache peaks are sampled, not exact maxima. Admission "
            "counters are cumulative from service start and logged at most once/minute."
        ),
    }


def compare(parent, candidate, output):
    report = {
        "reference": summarize(parent),
        "candidate": summarize(candidate),
        "limitations": (
            "Historical service sessions, not interleaved A/B. Lower preemption alone "
            "does not establish an optimization; assess throughput and latency together."
        ),
    }
    # EXP-0012 uses the original scheduler. The withdrawn EXP-0010 admission
    # telemetry must be absent; ordinary vLLM counters remain mandatory.
    log = (Path(candidate) / "server.log").read_text(errors="replace")
    if (report["candidate"]["admission_last_snapshot"] is not None
            or "KVReservationAsyncScheduler" in log
            or "KV growth admission enabled=True" in log):
        raise ValueError("Withdrawn scheduler unexpectedly present")
    if not report["candidate"]["server_log_samples"]:
        raise ValueError("No ordinary scheduler samples")
    report["scheduler_policy"] = "original; admission telemetry must be absent"
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report
