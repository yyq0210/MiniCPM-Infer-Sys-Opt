"""EXP-0006 isolated diagnostics. Requires the shared GPU experiment lock."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F
from flag_gems.runtime.backend._iluvatar.fused.rope_kv_cache import rope_kv_cache
from flag_gems.runtime.backend._iluvatar.fused.rotary_embedding_parallel import rope_parallel
from flag_gems.runtime.backend._iluvatar.fused.gate_up_swiglu import gate_up_swiglu
from flag_gems.runtime.backend._iluvatar.fused.rms_qkv import rms_qkv
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import triton_reshape_and_cache_flash


def timed(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(16):
            fn()
    values = []
    for _ in range(7):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        values.append(begin.elapsed_time(end) * 1000 / 16)
    return dict(median_us=statistics.median(values), trials_us=values)


def make_rope(n, dtype=torch.bfloat16, hnd=False, page=16, short_slots=False):
    qkv = torch.randn(n, 20, 128, dtype=dtype, device="cuda")
    q, k, v = qkv[:, :16], qkv[:, 16:18], qkv[:, 18:]
    angles = torch.randn(4096, 64, device="cuda")
    cs = torch.cat((angles.cos(), angles.sin()), -1).to(dtype)
    cos, sin = cs.chunk(2, -1)
    positions = torch.randperm(4096, device="cuda")[:n]
    pages = max(8, (n * 2 + page - 1) // page)
    raw = torch.full((pages, 2, 2, page, 128) if hnd else (pages, 2, page, 2, 128),
                     0.125, dtype=dtype, device="cuda")
    cache = raw.permute(0, 1, 3, 2, 4) if hnd else raw
    kc, vc = cache.unbind(1)
    slot_raw = torch.randperm(pages * page, device="cuda")[:n * 2]
    slots = slot_raw[::2][:n]
    slots[::7] = -1
    if short_slots and n > 1:
        slots = slots[:-1]
    return dict(q=q, k=k, v=v, cos=cos, sin=sin, positions=positions,
                k_cache=kc, v_cache=vc, slots=slots), qkv, cache


def old_rope(x, interleaved=False):
    q, k = rope_parallel(x["q"], x["k"], x["cos"], x["sin"], x["positions"],
                         interleaved=interleaved, inplace=False)
    scale = torch.ones((), device="cuda")
    triton_reshape_and_cache_flash(k, x["v"], x["k_cache"], x["v_cache"],
                                   x["slots"], "auto", scale, scale)
    return q, k


def reference_rope(x, tensor, interleaved):
    a = tensor[..., ::2] if interleaved else tensor[..., :64]
    b = tensor[..., 1::2] if interleaved else tensor[..., 64:]
    cos = x["cos"][x["positions"]].float().unsqueeze(1)
    sin = x["sin"][x["positions"]].float().unsqueeze(1)
    lo = a.float() * cos - b.float() * sin
    hi = b.float() * cos + a.float() * sin
    return (torch.stack((lo, hi), -1).flatten(-2) if interleaved
            else torch.cat((lo, hi), -1))


def run_rope(result, save):
    configs = [dict(head_tile=bh, token_tile=bt, num_warps=w)
               for bh, bt, w in [(2, 1, 4), (4, 1, 4), (8, 1, 4),
                                 (16, 1, 4), (4, 4, 4), (8, 4, 4)]]
    result["configs"] = configs
    for dtype in (torch.bfloat16, torch.float16):
        for n, hnd, page, interleaved, short in [
            (1, False, 16, False, False), (7, True, 32, True, True),
            (64, True, 16, False, True), (257, False, 32, True, False),
        ]:
            x, qkv, cache = make_rope(n, dtype, hnd, page, short)
            original = qkv.clone()
            qr, kr = reference_rope(x, x["q"], interleaved), reference_rope(x, x["k"], interleaved)
            old_q, old_k = old_rope(x, interleaved)
            old_cache = cache.clone()
            for config in configs:
                cache.fill_(0.125)
                q, k = rope_kv_cache(**x, interleaved=interleaved, **config)
                tol = dict(atol=0.016, rtol=0.02) if dtype == torch.bfloat16 else dict(atol=0.002, rtol=0.003)
                torch.testing.assert_close(q.float(), qr, **tol)
                torch.testing.assert_close(k.float(), kr, **tol)
                # Same operation order as the existing specialized RoPE.
                torch.testing.assert_close(q, old_q, atol=0, rtol=0)
                torch.testing.assert_close(k, old_k, atol=0, rtol=0)
                torch.testing.assert_close(cache, old_cache, atol=0, rtol=0)
                torch.testing.assert_close(qkv, original, atol=0, rtol=0)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    gq, gk = rope_kv_cache(**x, interleaved=interleaved, **config)
                qkv.mul_(0.9)
                x["positions"].copy_(x["positions"].roll(1))
                x["slots"].copy_(x["slots"].roll(1))
                cache.fill_(0.125)
                eq, ek = rope_kv_cache(**x, interleaved=interleaved, **config)
                expected_cache = cache.clone()
                cache.fill_(0.125)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(gq, eq, atol=0, rtol=0)
                torch.testing.assert_close(gk, ek, atol=0, rtol=0)
                torch.testing.assert_close(cache, expected_cache, atol=0, rtol=0)
                result["checks"].append(dict(n=n, dtype=str(dtype), hnd=hnd, page=page,
                    interleaved=interleaved, short_slots=short, config=config,
                    reference="independent_fp32_and_exact_previous_rope_cache",
                    graph_changed_inputs=True, passed=True))
                qkv.copy_(original)
                x["positions"].copy_(x["positions"].roll(-1))
                x["slots"].copy_(x["slots"].roll(-1))
                save()
            del x, cache, qkv, graph, old_cache, expected_cache
            torch.cuda.empty_cache()
    # Exercise physical addresses above int32 without allocating a large model.
    x, qkv, cache = make_rope(1)
    stride = 1_073_745_920
    kc = torch.empty_strided((3, 16, 2, 128), (stride, 256, 128, 1),
                             device="cuda", dtype=torch.bfloat16)
    vc = torch.empty_strided((3, 16, 2, 128), (stride, 256, 128, 1),
                             device="cuda", dtype=torch.bfloat16)
    kc[2, :2].fill_(0.125)
    vc[2, :2].fill_(0.125)
    x.update(k_cache=kc, v_cache=vc, slots=torch.tensor([32], device="cuda"))
    q, k = rope_kv_cache(**x)
    torch.testing.assert_close(kc[2, 0], k[0], atol=0, rtol=0)
    torch.testing.assert_close(vc[2, 0], x["v"][0], atol=0, rtol=0)
    assert bool((kc[2, 1] == 0.125).all()) and bool((vc[2, 1] == 0.125).all())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gq, gk = rope_kv_cache(**x)
    qkv.mul_(0.7)
    eq, ek = rope_kv_cache(**x)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(gk, ek, atol=0, rtol=0)
    torch.testing.assert_close(kc[2, 0], ek[0], atol=0, rtol=0)
    result["checks"].append(dict(kind="large_physical_offset", element_offset=2*stride,
                                  graph_changed_inputs=True, passed=True))
    save()
    del x, kc, vc, graph, qkv, cache, gq, gk, eq, ek
    torch.cuda.empty_cache()
    for n in (1, 8, 30, 64, 128, 512, 2048):
        x, qkv, cache = make_rope(n)
        # Timed cases use all-valid unique slots; negative/padded slots belong
        # to the correctness matrix, not a reduced-work performance shortcut.
        x["slots"].copy_(torch.randperm(x["k_cache"].shape[0] * 16, device="cuda")[:n])
        # Baseline scale is preallocated; exclude allocation kernels in timings.
        scale = torch.ones((), device="cuda")
        def baseline():
            q, k = rope_parallel(x["q"], x["k"], x["cos"], x["sin"], x["positions"], inplace=False)
            triton_reshape_and_cache_flash(k, x["v"], x["k_cache"], x["v_cache"],
                                           x["slots"], "auto", scale, scale)
            return q, k
        before = timed(baseline)
        row = dict(n=n, baseline=before, candidates=[])
        for config in configs:
            timing = timed(lambda: rope_kv_cache(**x, **config))
            row["candidates"].append(dict(config=config, **timing,
                                         speedup=before["median_us"]/timing["median_us"]))
        row["baseline_after"] = timed(baseline)
        result["timings"].append(row)
        print(json.dumps(row), flush=True)
        save()
        del x, qkv, cache
        torch.cuda.empty_cache()


def run_mlp(result, save):
    configs = [dict(bm=bm, bn=bn, bk=bk, warps=w)
               for bm, bn, bk, w in [(16, 64, 32, 4), (16, 64, 64, 4),
                                      (32, 64, 32, 4), (32, 128, 32, 8)]]
    # Compile exactly the unfused activation expression used by LlamaMLP.
    @torch.compile
    def activation(y):
        g, u = y.chunk(2, -1)
        return F.silu(g) * u
    weight = torch.randn(12288, 2048, device="cuda", dtype=torch.bfloat16) / (2048 ** .5)
    result["configs"] = configs
    for n in (1, 8, 30, 64, 512, 2048):
        x = torch.randn(n, 2048, device="cuda", dtype=torch.bfloat16)
        def baseline():
            return activation(F.linear(x, weight))
        expected = baseline()
        before = timed(baseline)
        row = dict(n=n, baseline=before, candidates=[])
        for config in configs:
            output = gate_up_swiglu(x, weight, **config)
            max_error = (output.float()-expected.float()).abs().max().item()
            normalized_rmse = ((output.float()-expected.float()).square().mean()
                               / expected.float().square().mean()).sqrt().item()
            # Fixed BF16 tolerances, never increased after a failure.
            passed = torch.allclose(output, expected, atol=0.02, rtol=0.03) and normalized_rmse < 0.01
            entry = dict(config=config, max_abs_error=max_error,
                         normalized_rmse=normalized_rmse, numerical_passed=passed)
            if passed:
                timing = timed(lambda: gate_up_swiglu(x, weight, **config))
                entry.update(**timing, speedup=before["median_us"]/timing["median_us"])
            row["candidates"].append(entry)
        row["baseline_after"] = timed(baseline)
        result["timings"].append(row)
        print(json.dumps(row), flush=True)
        save()


def run_norm(result, save):
    configs = [dict(bm=bm, bn=bn, bk=bk, warps=w)
               for bm, bn, bk, w in [(16, 32, 32, 4), (16, 64, 32, 4),
                                      (16, 64, 64, 4)]]
    @torch.compile
    def norm(x, residual, gamma):
        summed = x.float() + residual.float()
        normalized = summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + 1e-6)
        return (normalized.to(gamma.dtype) * gamma).to(x.dtype), summed.to(x.dtype)
    weight = torch.randn(2560, 2048, device="cuda", dtype=torch.bfloat16) / (2048 ** .5)
    gamma = (torch.randn(2048, device="cuda") * .1 + 1).bfloat16()
    result["configs"] = configs
    for n in (1, 8, 30, 64, 512):
        x = torch.randn(n, 2048, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        def baseline():
            y, r = norm(x, residual, gamma)
            return F.linear(y, weight), r
        expected, expected_r = baseline()
        before = timed(baseline)
        row = dict(n=n, baseline=before, candidates=[])
        for config in configs:
            y, r = rms_qkv(x, gamma, weight, residual, **config)
            torch.testing.assert_close(r, expected_r, atol=0, rtol=0)
            max_error = (y.float()-expected.float()).abs().max().item()
            normalized_rmse = ((y.float()-expected.float()).square().mean()
                               / expected.float().square().mean()).sqrt().item()
            passed = torch.allclose(y, expected, atol=0.02, rtol=0.03) and normalized_rmse < .01
            entry = dict(config=config, max_abs_error=max_error,
                         normalized_rmse=normalized_rmse, numerical_passed=passed,
                         residual_bitwise_equal=True)
            if passed:
                timing = timed(lambda: rms_qkv(x, gamma, weight, residual, **config))
                entry.update(**timing, speedup=before["median_us"]/timing["median_us"])
            row["candidates"].append(entry)
        row["baseline_after"] = timed(baseline)
        result["timings"].append(row)
        print(json.dumps(row), flush=True)
        save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["rope", "mlp", "norm"])
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    result = dict(status="running", kind=args.kind, scope="diagnostic_not_official_e2e",
                  started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), checks=[], timings=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2))
        tmp.replace(args.output)
    torch.manual_seed(20261020)
    save()
    try:
        {"rope": run_rope, "mlp": run_mlp, "norm": run_norm}[args.kind](result, save)
        result["status"] = "passed"
    except BaseException as exc:
        result.update(status="failed", error=repr(exc))
        raise
    finally:
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        save()


if __name__ == "__main__":
    main()
