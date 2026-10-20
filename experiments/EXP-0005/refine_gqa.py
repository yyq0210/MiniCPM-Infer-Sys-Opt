"""Uncouple decode and prefill tiles after the complete first diagnostic sweep."""
import json
from pathlib import Path
import sys
import time
import torch
from validate_gqa import make_case, baseline, paged_gqa, fp32_check, timed


def main():
    previous, output = map(Path, sys.argv[1:3])
    prior = json.loads(previous.read_text())
    assert prior["status"] == "passed" and len(prior["timings"]) == 12
    result = {"status": "running", "kind": "diagnostic_not_official_e2e",
              "prior": str(previous), "cases": [], "started_at": time.strftime("%FT%T%z")}
    def save():
        output.write_text(json.dumps(result, indent=2)+"\n")
    configs = [
        dict(segments=16, decode_tile=32, prefill_tile=64, prefill_rows=32),
        dict(segments=16, decode_tile=32, prefill_tile=32, prefill_rows=32),
        dict(segments=16, decode_tile=32, prefill_tile=32, prefill_rows=64),
        dict(segments=16, decode_tile=32, prefill_tile=64, prefill_rows=64),
    ]
    cases = [
        ("mixed_decode30_prefill2018_16k", [1]*30+[2018], [16384]*31),
        ("mixed_prefill2018_decode30_16k", [2018]+[1]*30, [16384]*31),
        ("prefill2048_context16k", [2048], [16384]),
        ("prefill2048_context4k", [2048], [4096]),
        ("prefill2048_context2k", [2048], [2048]),
    ]
    torch.manual_seed(20261021)
    save()
    try:
        for name, qlens, lengths in cases:
            x, _, _ = make_case(qlens, lengths)
            original_q = x["q"].clone()
            base = timed(lambda: baseline(x, lengths))
            entry = {"name": name, "baseline": base, "candidates": []}
            for config in configs:
                paged_gqa(**x, **config)
                error = fp32_check(x, qlens, lengths)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    paged_gqa(**x, **config)
                x["q"].mul_(0.99)
                paged_gqa(**x, **config)
                eager = x["out"].clone()
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(eager, x["out"], atol=0, rtol=0)
                x["q"].copy_(original_q)
                measurement = timed(lambda: paged_gqa(**x, **config))
                entry["candidates"].append(dict(config=config, fp32_max_abs_error=error,
                    graph_passed=True, **measurement, speedup=base["median_us"]/measurement["median_us"]))
            entry["baseline_after"] = timed(lambda: baseline(x, lengths))
            result["cases"].append(entry)
            save()
            print(json.dumps(entry), flush=True)
            del x, eager, graph, original_q
            torch.cuda.empty_cache()
        result["status"] = "passed"
    except BaseException as exc:
        result.update(status="failed", error=repr(exc))
        raise
    finally:
        result["finished_at"] = time.strftime("%FT%T%z")
        save()


if __name__ == "__main__":
    main()
