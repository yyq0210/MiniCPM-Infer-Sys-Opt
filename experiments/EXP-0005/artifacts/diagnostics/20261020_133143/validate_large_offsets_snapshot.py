"""Exercise >int32 KV offsets and recheck final-kernel diagnostic workpoints."""
import json
from pathlib import Path
import sys
import torch
from validate_gqa import make_case, paged_gqa, fp32_check, baseline, timed


def main():
    path = Path(sys.argv[1])
    result = {"status": "running", "timings": []}
    torch.manual_seed(20261022)
    try:
        x, qlens, lengths = make_case([1], [17])
        stride = 2**30 + 4096
        elements = 2 * stride + 4096
        # Only three small pages are initialized, with large holes between
        # them. Both offset operands fit int32, but page 2's offset does not.
        kb = torch.empty(elements, device="cuda", dtype=torch.bfloat16)
        vb = torch.empty_like(kb)
        x["k"] = kb.as_strided((3, 16, 2, 128), (stride, 256, 128, 1))
        x["v"] = vb.as_strided((3, 16, 2, 128), (stride, 256, 128, 1))
        x["k"].normal_(std=0.3)
        x["v"].normal_()
        x["block_table"] = torch.tensor([[2, 1]], device="cuda", dtype=torch.int32)
        paged_gqa(**x)
        error = fp32_check(x, qlens, lengths)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            paged_gqa(**x)
        graph.replay()
        torch.cuda.synchronize()
        fp32_check(x, qlens, lengths)
        result["large_offsets"] = {"page_stride_elements": stride,
                                   "page2_offset_elements": 2*stride,
                                   "fp32_max_abs_error": error, "graph_passed": True}
        del graph, x, kb, vb
        torch.cuda.empty_cache()
        for name, qs, ls in [
            ("decode_b30_k16384", [1]*30, [16384-i for i in range(30)]),
            ("decode_b64_k4096", [1]*64, [4096-i for i in range(64)]),
            ("mixed_decode30_prefill2018_16k", [1]*30+[2018], [16384]*31),
        ]:
            x, _, _ = make_case(qs, ls)
            before = timed(lambda: baseline(x, ls))
            paged_gqa(**x)
            fp32_check(x, qs, ls)
            after = timed(lambda: paged_gqa(**x))
            result["timings"].append({"name": name, "baseline": before, "candidate": after,
                                      "speedup": before["median_us"]/after["median_us"]})
            del x
            torch.cuda.empty_cache()
        result["status"] = "passed"
    except BaseException as exc:
        result.update(status="failed", error=repr(exc))
        raise
    finally:
        path.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
