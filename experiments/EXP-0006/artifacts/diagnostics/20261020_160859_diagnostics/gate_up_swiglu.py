# SPDX-License-Identifier: Apache-2.0
"""Experimental paired gate/up GEMM epilogue; not enabled in serving."""
import torch
import triton
import triton.language as tl


@triton.jit
def _gate_up(
    X, W, Y, M: tl.constexpr, H: tl.constexpr, I: tl.constexpr,
    XS: tl.constexpr, WN: tl.constexpr, WK: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ROUND_SILU: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    gate = tl.zeros((BM, BN), tl.float32)
    up = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(H, BK)):
        idx = block * BK + kk
        x = tl.load(X + rows[:, None] * XS + idx[None, :],
                    (rows[:, None] < M) & (idx[None, :] < H), 0)
        wg = tl.load(W + cols[None, :] * WN + idx[:, None] * WK,
                     (cols[None, :] < I) & (idx[:, None] < H), 0)
        wu = tl.load(W + (cols[None, :] + I) * WN + idx[:, None] * WK,
                     (cols[None, :] < I) & (idx[:, None] < H), 0)
        gate += tl.dot(x, wg)
        up += tl.dot(x, wu)
    # Preserve the projection's output dtype rounding before activation.
    g = gate.to(Y.dtype.element_ty).to(tl.float32)
    u = up.to(Y.dtype.element_ty).to(tl.float32)
    z = g / (1 + tl.exp(-g))
    if ROUND_SILU:
        z = z.to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y + rows[:, None] * I + cols[None, :], z * u,
             (rows[:, None] < M) & (cols[None, :] < I))


def gate_up_swiglu(x, weight, *, bm=16, bn=64, bk=32, warps=4,
                   round_silu=False):
    m, h = x.shape
    intermediate = weight.shape[0] // 2
    out = torch.empty((m, intermediate), dtype=x.dtype, device=x.device)
    _gate_up[(triton.cdiv(m, bm), triton.cdiv(intermediate, bn))](
        x, weight, out, m, h, intermediate, x.stride(0),
        weight.stride(0), weight.stride(1), bm, bn, bk, round_silu,
        num_warps=warps, num_stages=2,
    )
    return out
