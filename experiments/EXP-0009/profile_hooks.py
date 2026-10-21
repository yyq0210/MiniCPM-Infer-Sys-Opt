"""Diagnostic-only worker extension; no changes to tensors, kernels or outputs.

Imported explicitly by --worker-extension-cls. All metadata is CPU-resident.
"""
import base64
import contextlib
import copy
import json
import os
from pathlib import Path
import time

import torch
from vllm_fl.worker.worker import WorkerFL

ROOT = Path(os.environ["LEANSCALE_PROFILE_PASS"])
CURRENT = {"step": 0, "active": False, "metadata": {}}
ORIGINAL_PROFILE = WorkerFL.profile


def emit_meta(kind, data):
    if not CURRENT["active"]:
        return
    data = dict(kind=kind, step=CURRENT["step"], **data)
    with torch.profiler.record_function("LS_META64:" + base64.urlsafe_b64encode(json.dumps(data, separators=(",", ":")).encode()).decode()):
        pass
    with (ROOT / "step_metadata.jsonl").open("a") as f:
        f.write(json.dumps(data) + "\n")


def profile(self, is_start=True, profile_prefix=None):
    if is_start:
        # A fresh wrapper per window avoids schedule/table overwrite and stale
        # iteration counters. All windows are started by this task's controller.
        if self.profiler is not None:
            self.profiler.stop()
        control = json.loads((ROOT / "window_control.json").read_text())
        window = control["window"]
        assert window in ("initial", "steady", "late")
        config = copy.deepcopy(self.profiler_config)
        config.torch_profiler_dir = str(ROOT / "traces" / window)
        Path(config.torch_profiler_dir).mkdir(parents=True, exist_ok=True)
        self.profiler_config = config
        self.profiler = None
        CURRENT["window"] = window
    return ORIGINAL_PROFILE(self, is_start, profile_prefix)


@contextlib.contextmanager
def annotate(self, scheduler_output):
    CURRENT["step"] += 1
    CURRENT["active"] = False
    if self.profiler is not None:
        self.profiler.step()
        CURRENT["active"] = bool(self.profiler._running)
    # Install instance wrappers only after the model runner exists. Wrappers
    # execute outside compiled model code and do not change any return value.
    runner = self.model_runner
    if not getattr(runner, "_leanscale_profile_wrapped", False):
        prepare = runner._prepare_inputs
        padding = runner._determine_batch_execution_and_padding

        def prepare_inputs(scheduled, counts):
            out = prepare(scheduled, counts)
            batch = runner.input_batch
            n = int(batch.num_reqs)
            computed = batch.num_computed_tokens_cpu[:n]
            prompt = batch.num_prompt_tokens[:n]
            prefill = int((computed < prompt).sum())
            q = [int(x) for x in counts]
            lengths = [int(x) + q[i] for i, x in enumerate(computed)]
            meta = {
                "phase": "prefill" if prefill == n else "decode" if prefill == 0 else "mixed",
                "B": n, "M_unpadded": int(sum(q)), "prefill_requests": prefill,
                "decode_requests": n - prefill, "q_min": min(q), "q_max": max(q),
                "L_min_cpu_upper_bound": min(lengths),
                "L_max_cpu_upper_bound": max(lengths),
                "new_requests": len(scheduled.scheduled_new_reqs),
            }
            CURRENT["metadata"] = meta
            CURRENT["new_requests_total"] = (
                CURRENT.get("new_requests_total", 0) + meta["new_requests"])
            emit_meta("batch", meta)
            # Low-rate status for the controller, also between capture windows.
            now = time.monotonic()
            if now - CURRENT.get("last_status", 0) > 0.5:
                status = dict(meta, step=CURRENT["step"], epoch=time.time(),
                              new_requests_total=CURRENT["new_requests_total"],
                              profiling=CURRENT["active"])
                tmp = ROOT / "latest_batch.tmp"
                tmp.write_text(json.dumps(status))
                tmp.replace(ROOT / "latest_batch.json")
                CURRENT["last_status"] = now
            return out

        def batch_padding(*args, **kwargs):
            out = padding(*args, **kwargs)
            mode, desc = out[:2]
            emit_meta("padding", {
                "graph_mode": str(mode), "M_padded": int(desc.num_tokens),
                "B_padded": desc.num_reqs,
            })
            return out

        runner._prepare_inputs = prepare_inputs
        runner._determine_batch_execution_and_padding = batch_padding
        runner._leanscale_profile_wrapped = True
    cm = (torch.profiler.record_function(f"LS_STEP:{CURRENT['step']}")
          if CURRENT["active"] else contextlib.nullcontext())
    try:
        with cm:
            yield
    finally:
        # Sampling is a separate worker RPC. Its annotation uses the same step.
        CURRENT["last_step_profiled"] = CURRENT["active"]
        CURRENT["active"] = False


ORIGINAL_SAMPLE = WorkerFL.sample_tokens


def sample(self, *args, **kwargs):
    active = CURRENT.get("last_step_profiled", False)
    cm = (torch.profiler.record_function(f"LS_SAMPLE:{CURRENT['step']}")
          if active else contextlib.nullcontext())
    with cm:
        return ORIGINAL_SAMPLE(self, *args, **kwargs)


WorkerFL.profile = profile
WorkerFL.annotate_profile = annotate
WorkerFL.sample_tokens = sample


class ProfileExtension:
    """Explicit opt-in import hook; no conflicting worker methods."""
