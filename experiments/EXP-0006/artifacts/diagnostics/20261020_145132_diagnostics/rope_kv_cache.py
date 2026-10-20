# SPDX-License-Identifier: Apache-2.0
"""Functional RoPE outputs and one paged K/V write in the same launch."""
import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kv(
    Q, K, V, OQ, OK, C, S, P, KC, VC, SLOT,
    N: tl.constexpr, NSLOT: tl.constexpr,
    QS: tl.constexpr, QH: tl.constexpr, KS: tl.constexpr, KH: tl.constexpr,
    VS: tl.constexpr, VH: tl.constexpr, CS: tl.constexpr, SS: tl.constexpr,
    PS: tl.constexpr, SL: tl.constexpr,
    K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
    V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr,
    PAGE: tl.constexpr, BH: tl.constexpr, BT: tl.constexpr,
    INTERLEAVED: tl.constexpr,
):
    group = tl.program_id(1)
    r = tl.arange(0, BT * BH)
    token = tl.program_id(0) * BT + r // BH
    head = group * BH + r % BH
    d = tl.arange(0, 64)
    position = tl.load(P + token * PS, token < N, 0)
    c = tl.load(C + position[:, None] * CS + d[None, :]).to(tl.float32)
    s = tl.load(S + position[:, None] * SS + d[None, :]).to(tl.float32)
    if INTERLEAVED:
        lo, hi = d * 2, d * 2 + 1
    else:
        lo, hi = d, d + 64
    if group < tl.cdiv(16, BH):
        mask = (token[:, None] < N) & (head[:, None] < 16)
        base = Q + token[:, None] * QS + head[:, None] * QH
        out = OQ + (token[:, None] * 16 + head[:, None]) * 128
    else:
        head = (group - tl.cdiv(16, BH)) * BH + r % BH
        mask = (token[:, None] < N) & (head[:, None] < 2)
        base = K + token[:, None] * KS + head[:, None] * KH
        out = OK + (token[:, None] * 2 + head[:, None]) * 128
    a = tl.load(base + lo[None, :], mask, 0).to(tl.float32)
    b = tl.load(base + hi[None, :], mask, 0).to(tl.float32)
    lower, upper = a * c + b * (-s), b * c + a * s
    tl.store(out + lo[None, :], lower, mask)
    tl.store(out + hi[None, :], upper, mask)
    if group >= tl.cdiv(16, BH):
        slot = tl.load(SLOT + token * SL, (token < N) & (token < NSLOT), -1)
        block = (slot // PAGE).to(tl.int64)
        offset = slot % PAGE
        cache_mask = mask & (slot[:, None] >= 0)
        kb = KC + block[:, None] * K0 + offset[:, None] * K1 + head[:, None] * K2
        vb = VC + block[:, None] * V0 + offset[:, None] * V1 + head[:, None] * V2
        tl.store(kb + lo[None, :], lower, cache_mask)
        tl.store(kb + hi[None, :], upper, cache_mask)
        src = V + token[:, None] * VS + head[:, None] * VH
        va = tl.load(src + lo[None, :], cache_mask, 0)
        vb_value = tl.load(src + hi[None, :], cache_mask, 0)
        tl.store(vb + lo[None, :], va, cache_mask)
        tl.store(vb + hi[None, :], vb_value, cache_mask)


def rope_kv_cache(q, k, v, cos, sin, positions, k_cache, v_cache, slots,
                  *, interleaved=False, head_tile=8, token_tile=1,
                  num_warps=4):
    """Caller validates Q16/KV2/D128, unquantized cache and unique valid slots.

    Q/K/V and projection storage are never mutated. Negative slots and graph
    padding beyond the slot tensor never write the cache. Cache strides are
    explicit; page addresses are widened before multiplication.
    """
    oq = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    ok = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    if q.shape[0]:
        _rope_kv[(triton.cdiv(q.shape[0], token_tile),
                  triton.cdiv(16, head_tile) + triton.cdiv(2, head_tile))](
            q, k, v, oq, ok, cos, sin, positions, k_cache, v_cache, slots,
            q.shape[0], slots.numel(), q.stride(0), q.stride(1),
            k.stride(0), k.stride(1), v.stride(0), v.stride(1),
            cos.stride(0), sin.stride(0), positions.stride(0), slots.stride(0),
            *k_cache.stride()[:3], *v_cache.stride()[:3], k_cache.shape[1],
            head_tile, token_tile, interleaved, num_warps=num_warps,
        )
    return oq, ok
