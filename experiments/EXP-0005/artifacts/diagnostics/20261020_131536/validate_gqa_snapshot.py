"""EXP-0005: correctness and isolated diagnostic timings, never official E2E."""
import argparse
import json
import math
from pathlib import Path
import statistics
import time

import torch
from flag_gems.runtime.backend._iluvatar.fused.paged_gqa import paged_gqa
from vllm.v1.attention.ops.triton_unified_attention import unified_attention


def make_case(qlens, lengths, dtype=torch.bfloat16, page=16, hnd=False):
    n = len(qlens)
    width = math.ceil(max(lengths) / page)
    pages = width * n
    raw = torch.randn((pages, 2, 2, page, 128) if hnd else
                      (pages, 2, page, 2, 128), device="cuda", dtype=dtype)
    cache = raw.permute(0, 1, 3, 2, 4) if hnd else raw
    k, v = cache.unbind(1)
    table = torch.randperm(pages, device="cuda").int().reshape(n, width)
    qkv = torch.randn(sum(qlens), 20, 128, device="cuda", dtype=dtype)
    q = qkv[:, :16]
    out = torch.empty_like(q)
    lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    cu = torch.tensor([0] + list(__import__("itertools").accumulate(qlens)),
                      device="cuda", dtype=torch.int32)
    po = torch.empty(max(n, 64), 16, 16, 128, device="cuda", dtype=torch.float32)
    pm = torch.empty(max(n, 64), 16, 16, device="cuda", dtype=torch.float32)
    pl = torch.empty_like(pm)
    return dict(q=q, k=k, v=v, out=out, cu=cu, lengths=lens, block_table=table,
                scale=128 ** -0.5, max_query_len=max(qlens),
                partial_out=po, partial_max=pm, partial_sum=pl), qlens, lengths


def baseline(x, lengths):
    return unified_attention(
        q=x["q"], k=x["k"], v=x["v"], out=x["out"], cu_seqlens_q=x["cu"],
        max_seqlen_q=x["max_query_len"], seqused_k=x["lengths"],
        max_seqlen_k=max(lengths), softmax_scale=x["scale"], causal=True,
        window_size=(-1, -1), block_table=x["block_table"], softcap=0,
        q_descale=None, k_descale=None, v_descale=None, seq_threshold_3D=64,
        num_par_softmax_segments=16, softmax_segm_output=x["partial_out"],
        softmax_segm_max=x["partial_max"], softmax_segm_expsum=x["partial_sum"])


def fp32_check(x, qlens, lengths, *, check=True):
    start = 0
    maximum_error = 0
    for seq, (qlen, length) in enumerate(zip(qlens, lengths)):
        # Independent FP32 reference at beginning/middle/end of each sequence.
        if qlen:
            indices = list(range(qlen)) if qlen <= 67 else sorted({0, qlen // 2, qlen - 1})
            token_ids = torch.arange(length, device="cuda")
            blocks = x["block_table"][seq, token_ids // x["k"].shape[1]]
            k = x["k"][blocks, token_ids % x["k"].shape[1]].float().repeat_interleave(8, 1)
            v = x["v"][blocks, token_ids % x["v"].shape[1]].float().repeat_interleave(8, 1)
            for i in indices:
                scores = torch.einsum("hd,lhd->hl", x["q"][start+i].float(), k) * x["scale"]
                scores[:, length-qlen+i+1:] = -torch.inf
                ref = torch.einsum("hl,lhd->hd", scores.softmax(-1), v)
                got = x["out"][start+i].float()
                if check:
                    torch.testing.assert_close(got, ref, atol=3e-3, rtol=3e-2)
                maximum_error = max(maximum_error, (got-ref).abs().max().item())
        start += qlen
    return maximum_error


def timed(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(8):
            fn()
    values = []
    for _ in range(5):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        values.append(begin.elapsed_time(end) * 1000 / 8)
    return {"median_us": statistics.median(values), "trials_us": values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--correctness-only", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"status": "running", "kind": "diagnostic_not_official_e2e",
              "checks": [], "timings": [], "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    def save():
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2))
        tmp.replace(args.output)
    torch.manual_seed(20261020)
    save()
    try:
        for dtype in (torch.bfloat16, torch.float16):
            for page, hnd in ((16, False), (32, True)):
                for qlens, lengths in [
                    ([1], [1]), ([1, 0, 1, 1], [17, 0, 33, 4097]),
                    ([1]*30, [16384-i*7 for i in range(30)]),
                    ([1, 37, 0, 1, 7], [16385, 201, 0, 4096, 71]),
                    ([3, 17, 67], [3, 33, 1025]),
                ]:
                    x, _, _ = make_case(qlens, lengths, dtype, page, hnd)
                    baseline(x, lengths)
                    expected = x["out"].clone()
                    baseline_error = fp32_check(x, qlens, lengths, check=False)
                    for segments, tile, rows in ((8, 32, 16), (16, 64, 32)):
                        x["out"].fill_(float("nan"))
                        paged_gqa(**x, segments=segments, decode_tile=tile,
                                  prefill_tile=tile, prefill_rows=rows)
                        baseline_difference = (x["out"] - expected).abs().max().item()
                        if max(qlens) <= 1:
                            torch.testing.assert_close(x["out"], expected, atol=3e-3, rtol=3e-2)
                        error = fp32_check(x, qlens, lengths)
                        # Graph replay also changes tensor contents at stable addresses.
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            paged_gqa(**x, segments=segments, decode_tile=tile,
                                      prefill_tile=tile, prefill_rows=rows)
                        x["q"].mul_(0.8)
                        baseline(x, lengths)
                        expected.copy_(x["out"])
                        paged_gqa(**x, segments=segments, decode_tile=tile,
                                  prefill_tile=tile, prefill_rows=rows)
                        eager = x["out"].clone()
                        fp32_check(x, qlens, lengths)
                        graph.replay()
                        torch.cuda.synchronize()
                        torch.testing.assert_close(x["out"], eager, atol=0, rtol=0)
                        result["checks"].append(dict(dtype=str(dtype), page=page, hnd=hnd,
                            qlens=qlens, lengths=lengths, segments=segments, tile=tile,
                            prefill_rows=rows, fp32_max_abs_error=error, graph_passed=True,
                            baseline_fp32_max_abs_error=baseline_error,
                            baseline_candidate_max_abs_difference=baseline_difference,
                            reference="independent_fp32_causal_attention"))
                        # Reference for the next config follows updated Q.
                    save()
                    del x, graph, expected, eager
                    torch.cuda.empty_cache()
        result["correctness_passed"] = True
        save()
        if not args.correctness_only:
            cases = []
            for b in (1, 8, 30, 64):
                for length in (4096, 16384):
                    cases.append((f"decode_b{b}_k{length}", [1]*b, [length-i for i in range(b)]))
            cases += [
                ("mixed_decode30_prefill2018_16k", [1]*30+[2018], [16384]*30+[16384]),
                ("mixed_prefill2018_decode30_16k", [2018]+[1]*30, [16384]*31),
                ("prefill2048_context16k", [2048], [16384]),
                ("prefill2048_context4k", [2048], [4096]),
            ]
            configs = [
                dict(segments=s, decode_tile=t, prefill_tile=p, prefill_rows=m)
                for s, t, p, m in ((8,32,32,16), (16,32,32,16),
                                   (8,64,64,32), (16,64,64,32), (16,128,64,32))
            ]
            for name, qlens, lengths in cases:
                x, _, _ = make_case(qlens, lengths)
                base_time = timed(lambda: baseline(x, lengths))
                expected = x["out"].clone()
                entry = {"name": name, "baseline": base_time, "candidates": []}
                for config in configs:
                    paged_gqa(**x, **config)
                    fp32_check(x, qlens, lengths)
                    if max(qlens) <= 1:
                        torch.testing.assert_close(x["out"], expected, atol=3e-3, rtol=3e-2)
                    timing = timed(lambda: paged_gqa(**x, **config))
                    entry["candidates"].append(dict(config=config, **timing,
                        speedup=base_time["median_us"]/timing["median_us"]))
                # Second baseline exposes drift/order effects.
                entry["baseline_after"] = timed(lambda: baseline(x, lengths))
                result["timings"].append(entry)
                print(json.dumps(entry), flush=True)
                save()
                del x, expected
                torch.cuda.empty_cache()
        result["status"] = "passed"
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = repr(exc)
        raise
    finally:
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        save()


if __name__ == "__main__":
    main()
