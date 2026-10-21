"""Meaningful CPU-only checks: nested timing, phase correlation, shape mapping."""
import base64
import json
from pathlib import Path
import tempfile

from analyze_profile import analyze_trace, category, union_us

assert union_us([(0, 5), (3, 8), (9, 12)]) == 11
assert category("vendor_gemm", [[[64, 2048], [2048, 2560]]])[0] == "OP03_QKV"
assert category("vendor_gemm", [])[0] == "GEMM_unresolved_projection"
assert category("_rope_kv", [])[0] == "OP04_RoPE_KV"


def event(name, cat, ts, dur, args=None, tid=1):
    return dict(name=name, cat=cat, ph="X", ts=ts, dur=dur,
                pid=1, tid=tid, args=args or {})


events = [
    event("LS_STEP:7", "user_annotation", 0, 100),
    event('LS_META64:' + base64.urlsafe_b64encode(b'{"step":7,"kind":"batch","phase":"decode","B":64}').decode(), "user_annotation", 1, 1),
    event("aten::mm", "cpu_op", 5, 80, {"External id": 5, "Input Dims": [[64, 2048], [2048, 2560]]}),
    event("cudaLaunchKernel", "cuda_runtime", 10, 3, {"correlation": 99, "External id": 5}),
    event("vendor_gemm", "kernel", 110, 20, {"correlation": 99}, tid=2),
    # An unlinked GPU kernel must not be assigned the step by timestamp proximity.
    event("unknown", "kernel", 40, 5, {"correlation": 777}, tid=2),
]
with tempfile.TemporaryDirectory() as d:
    root = Path(d)
    trace = root / "trace.json"
    trace.write_text(json.dumps({"traceEvents": events}))
    out = root / "out"
    out.mkdir()
    report = analyze_trace(trace, out)
    assert report["kernel_events"] == 2
    assert report["kernel_sum_us"] == 25
    assert report["correlations"] == {"linked_to_step": 1, "unlinked_to_step": 1}
    qkv = next(x for x in report["top_operators"] if x["operator"] == "OP03_QKV")
    assert qkv["phase"] == "decode" and qkv["total_us"] == 20
    import csv
    rows = list(csv.DictReader((out / "cpu_operator_times.csv").open()))
    assert float(rows[0]["self_cpu_us"]) == 77
    assert float(rows[0]["inclusive_cpu_us"]) == 80
print(json.dumps({"status": "passed", "checks": [
    "overlap_union", "shape_QKV", "ambiguous_GEMM_retained", "rope_mapping",
    "GPU_CPU_correlation", "unlinked_GPU_retained", "CPU_nested_self_time",
    "GPU_not_CPU_timing"]}))

# Exercise the diagnostic metadata hooks with CPU arrays and a torch stub that
# has no CUDA API. The wrapper must return exactly the underlying outputs.
import contextlib
import importlib.util
import os
import sys
import types
import numpy as np

with tempfile.TemporaryDirectory() as d:
    root = Path(d)
    os.environ["LEANSCALE_PROFILE_PASS"] = str(root)
    sys.modules["torch"] = types.SimpleNamespace(
        profiler=types.SimpleNamespace(record_function=lambda name: contextlib.nullcontext()))

    class FakeWorker:
        def profile(self, is_start=True, profile_prefix=None):
            return is_start

        def sample_tokens(self, *args, **kwargs):
            return "sample_output"

    sys.modules["vllm_fl.worker.worker"] = types.SimpleNamespace(WorkerFL=FakeWorker)
    spec = importlib.util.spec_from_file_location("hooks_test", Path(__file__).with_name("profile_hooks.py"))
    hooks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hooks)
    output = object()
    batch = types.SimpleNamespace(num_reqs=3,
                                  num_computed_tokens_cpu=np.array([4096, 0, 4097]),
                                  num_prompt_tokens=np.array([4096, 4096, 4096]))
    padding_result = ("FULL", types.SimpleNamespace(num_tokens=272, num_reqs=3))
    runner = types.SimpleNamespace(input_batch=batch,
                                   _prepare_inputs=lambda scheduled, counts: output,
                                   _determine_batch_execution_and_padding=lambda **kwargs: padding_result)
    worker = FakeWorker()
    worker.model_runner = runner
    worker.profiler = types.SimpleNamespace(_running=True, step=lambda: None)
    scheduled = types.SimpleNamespace(scheduled_new_reqs=[object()])
    with worker.annotate_profile(scheduled):
        assert runner._prepare_inputs(scheduled, np.array([1, 256, 1])) is output
        assert runner._determine_batch_execution_and_padding() is padding_result
    meta = json.loads((root / "latest_batch.json").read_text())
    assert (meta["phase"], meta["M_unpadded"], meta["B"], meta["prefill_requests"]) == ("mixed", 258, 3, 1)
    assert worker.sample_tokens() == "sample_output"
    assert not hooks.CURRENT["active"]
    # Pure decode is determined from computed versus prompt counts, not the
    # fact that requests are cached (cached requests can be chunked prefill).
    batch.num_computed_tokens_cpu = np.array([4096, 4100, 4097])
    with worker.annotate_profile(scheduled):
        runner._prepare_inputs(scheduled, np.array([1, 1, 1]))
        assert hooks.CURRENT["metadata"]["phase"] == "decode"
    print(json.dumps({"status": "passed", "checks": [
        "hook_no_CUDA_access", "prepare_output_identity", "padding_output_identity",
        "mixed_phase_from_prompt_boundary", "decode_phase_from_prompt_boundary",
        "sample_output_identity", "profile_scope_cleanup"]}))
