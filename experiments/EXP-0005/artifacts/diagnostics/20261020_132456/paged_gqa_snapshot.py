# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 LeanScale.
"""Paged Q16/KV2/D128 attention for Iluvatar.

Decode sequences retain split-K parallelism even in a mixed prefill batch.
The caller supplies persistent scratch buffers; no CPU reads or allocations
are performed here. Unsupported features must be handled by the caller.
"""
import triton
import triton.language as tl


@triton.jit
def _attention(
    Q, K, V, O, CU, LEN, BT, PO, PM, PL,
    SCALE: tl.constexpr, NSEQ: tl.constexpr,
    QS: tl.constexpr, QH: tl.constexpr,
    OS: tl.constexpr, OH: tl.constexpr,
    K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
    V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr,
    BTS: tl.constexpr, PAGE: tl.constexpr,
    CAP_SEG: tl.constexpr, SEGMENTS: tl.constexpr,
    TILE: tl.constexpr, BM: tl.constexpr, DECODE: tl.constexpr,
):
    group = tl.program_id(1)
    if DECODE:
        seq = tl.program_id(0)
        local_block = 0
        seg = tl.program_id(2)
    else:
        # Upper-bound q-block grid. The monotone prefix includes one extra
        # block per sequence, so empty sequences and odd tails are safe.
        target = tl.program_id(0)
        left, right = 0, NSEQ
        while left < right:
            mid = (left + right) // 2
            prefix = tl.load(CU + mid) // (BM // 8) + mid
            if prefix <= target:
                left = mid + 1
            else:
                right = mid
        seq = left - 1
        local_block = target - (tl.load(CU + seq) // (BM // 8) + seq)
        seg = 0
    start = tl.load(CU + seq)
    qlen = tl.load(CU + seq + 1) - start
    if DECODE:
        if qlen != 1:
            return
    else:
        if qlen <= 1 or local_block * (BM // 8) >= qlen:
            return
    length = tl.load(LEN + seq)
    r = tl.arange(0, BM)
    d = tl.arange(0, 128)
    t = tl.arange(0, TILE)
    if DECODE:
        pos = tl.full((BM,), 0, tl.int32)
        valid = r < 8
        tiles_per_segment = tl.cdiv(length, SEGMENTS * TILE)
        lo = seg * tiles_per_segment * TILE
        hi = tl.minimum(lo + tiles_per_segment * TILE, length)
    else:
        pos = local_block * (BM // 8) + r // 8
        valid = pos < qlen
        lo = 0
        hi = tl.minimum(length, length - qlen + (local_block + 1) * (BM // 8))
    head = group * 8 + r % 8
    q = tl.load(Q + (start + pos[:, None]) * QS + head[:, None] * QH
                + d[None, :], valid[:, None], 0)
    m = tl.full((BM,), float("-inf"), tl.float32)
    l = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, 128), tl.float32)
    for base in range(lo, hi, TILE):
        tokens = base + t
        blocks = tl.load(BT + seq * BTS + tokens // PAGE,
                         tokens < length, 0)
        k = tl.load(K + blocks[None, :] * K0 + (tokens % PAGE)[None, :] * K1
                    + group * K2 + d[:, None], tokens[None, :] < length, 0)
        v = tl.load(V + blocks[:, None] * V0 + (tokens % PAGE)[:, None] * V1
                    + group * V2 + d[None, :], tokens[:, None] < length, 0)
        scores = tl.dot(q, k) * SCALE
        allowed = valid[:, None] & (tokens[None, :] < length)
        if not DECODE:
            allowed = allowed & (tokens[None, :] <= length - qlen + pos[:, None])
        scores = tl.where(allowed, scores, float("-inf"))
        new_m = tl.maximum(m, tl.max(scores, 1))
        # Invalid query lanes and empty contexts must never produce NaNs.
        safe_m = tl.where(new_m == float("-inf"), 0.0, new_m)
        alpha = tl.exp(m - safe_m)
        p = tl.exp(scores - safe_m[:, None])
        l = l * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        m = new_m
    if DECODE:
        off = (seq * 16 + head) * CAP_SEG + seg
        tl.store(PO + off[:, None] * 128 + d[None, :], acc, valid[:, None])
        tl.store(PM + off, m, valid)
        tl.store(PL + off, l, valid)
    else:
        result = acc / tl.where(l > 0, l, 1.0)[:, None]
        tl.store(O + (start + pos[:, None]) * OS + head[:, None] * OH
                 + d[None, :], result, valid[:, None])


@triton.jit
def _reduce(PO, PM, PL, O, CU, OS: tl.constexpr, OH: tl.constexpr,
            CAP_SEG: tl.constexpr, SEGMENTS: tl.constexpr, BS: tl.constexpr):
    seq, head = tl.program_id(0), tl.program_id(1)
    start = tl.load(CU + seq)
    if tl.load(CU + seq + 1) - start != 1:
        return
    s = tl.arange(0, BS)
    d = tl.arange(0, 128)
    off = (seq * 16 + head) * CAP_SEG + s
    m = tl.load(PM + off, s < SEGMENTS, float("-inf"))
    l = tl.load(PL + off, s < SEGMENTS, 0)
    maximum = tl.max(m, 0)
    maximum = tl.where(maximum == float("-inf"), 0.0, maximum)
    weight = tl.exp(m - maximum)
    acc = tl.load(PO + off[:, None] * 128 + d[None, :],
                  s[:, None] < SEGMENTS, 0)
    denominator = tl.sum(l * weight, 0)
    result = tl.sum(acc * weight[:, None], 0) / tl.where(denominator > 0, denominator, 1.0)
    tl.store(O + start * OS + head * OH + d, result)


def paged_gqa(q, k, v, out, cu, lengths, block_table, scale, max_query_len,
              partial_out, partial_max, partial_sum, *, segments=16,
              decode_tile=32, prefill_tile=32, prefill_rows=64):
    """Write exact causal attention (up to floating-point roundoff) to out.

    Q/O: [tokens,16,128]; K/V: [pages,page_size,2,128].
    CU, lengths and block_table are contiguous along their final dimensions.
    Scratch: contiguous FP32 [capacity,16,capacity_segments,128] / [...,segments].
    Python arguments describe shapes only, never GPU tensor contents.
    """
    nseq = lengths.numel()
    capacity_segments = partial_max.shape[-1]
    assert segments <= capacity_segments and partial_max.shape[0] >= nseq
    if nseq == 0 or q.shape[0] == 0:
        return out
    args = (q, k, v, out, cu, lengths, block_table,
            partial_out, partial_max, partial_sum, scale, nseq,
            q.stride(0), q.stride(1), out.stride(0), out.stride(1),
            *k.stride()[:3], *v.stride()[:3], block_table.stride(0), k.shape[1],
            capacity_segments, segments)
    # EXP-0005 on BI-V150: KV tiles above 32 regress decode. A 64-row
    # prefill tile reuses each KV load across eight query tokens and avoids
    # the register pressure of increasing the KV tile at the same time.
    if max_query_len > 1:
        _attention[(q.shape[0] // (prefill_rows // 8) + nseq, 2)](
            *args, prefill_tile, prefill_rows, False, num_warps=4, num_stages=2)
    _attention[(nseq, 2, segments)](
        *args, decode_tile, 16, True, num_warps=4, num_stages=2)
    _reduce[(nseq, 16)](
        partial_out, partial_max, partial_sum, out, cu, out.stride(0), out.stride(1),
        capacity_segments, segments, triton.next_power_of_2(segments), num_warps=4)
    return out
