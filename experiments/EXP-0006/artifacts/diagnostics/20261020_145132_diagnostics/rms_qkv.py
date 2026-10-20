# SPDX-License-Identifier: Apache-2.0
"""Experimental residual + RMSNorm + QKV GEMM; not enabled in serving."""
import torch
import triton
import triton.language as tl


@triton.jit
def _rms_qkv(X, R, G, W, Y, RY,
             M: tl.constexpr, H: tl.constexpr, N: tl.constexpr,
             XS: tl.constexpr, RS: tl.constexpr,
             WN: tl.constexpr, WK: tl.constexpr, EPS: tl.constexpr,
             RESIDUAL: tl.constexpr, BM: tl.constexpr,
             BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    all_h = tl.arange(0, H)
    xall = tl.load(X + rows[:, None] * XS + all_h[None, :],
                   rows[:, None] < M, 0).to(tl.float32)
    if RESIDUAL:
        xall += tl.load(R + rows[:, None] * RS + all_h[None, :],
                        rows[:, None] < M, 0).to(tl.float32)
        if tl.program_id(1) == 0:
            tl.store(RY + rows[:, None] * H + all_h[None, :],
                     xall, rows[:, None] < M)
    inv = tl.rsqrt(tl.sum(xall * xall, 1) / H + EPS)
    acc = tl.zeros((BM, BN), tl.float32)
    kk = tl.arange(0, BK)
    for block in range(tl.cdiv(H, BK)):
        idx = block * BK + kk
        x = tl.load(X + rows[:, None] * XS + idx[None, :],
                    (rows[:, None] < M) & (idx[None, :] < H), 0).to(tl.float32)
        if RESIDUAL:
            x += tl.load(R + rows[:, None] * RS + idx[None, :],
                         (rows[:, None] < M) & (idx[None, :] < H), 0).to(tl.float32)
        gamma = tl.load(G + idx, idx < H, 0).to(tl.float32)
        # Installed vLLM IR rounds normalized x to gamma.dtype BEFORE the
        # gamma multiply, then rounds again to the projection input dtype.
        normalized = (x * inv[:, None]).to(G.dtype.element_ty).to(tl.float32)
        normalized = (normalized * gamma[None, :]).to(X.dtype.element_ty)
        weight = tl.load(W + cols[None, :] * WN + idx[:, None] * WK,
                         (cols[None, :] < N) & (idx[:, None] < H), 0)
        acc += tl.dot(normalized, weight)
    tl.store(Y + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))


def rms_qkv(x, gamma, weight, residual=None, *, epsilon=1e-6,
            bm=16, bn=64, bk=32, warps=4):
    m, h = x.shape
    n = weight.shape[0]
    assert h == 2048
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    r_out = torch.empty_like(x) if residual is not None else None
    _rms_qkv[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        x, residual if residual is not None else x, gamma, weight, out,
        r_out if r_out is not None else x, m, h, n, x.stride(0),
        residual.stride(0) if residual is not None else x.stride(0),
        weight.stride(0), weight.stride(1), epsilon, residual is not None,
        bm, bn, bk, num_warps=warps, num_stages=2,
    )
    return out, r_out
