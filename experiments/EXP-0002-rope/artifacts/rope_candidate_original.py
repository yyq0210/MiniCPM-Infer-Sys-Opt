"""Experimental RoPE kernel; independent of serving until correctness/timing pass."""
import triton
import triton.language as tl


@triton.jit
def rope_parallel_kernel(
    Q, K, C, S, P,
    QS: tl.constexpr, QH: tl.constexpr, QD: tl.constexpr,
    KS: tl.constexpr, KH: tl.constexpr, KD: tl.constexpr,
    CS: tl.constexpr, SS: tl.constexpr, PS: tl.constexpr,
    NQ: tl.constexpr, NK: tl.constexpr, D: tl.constexpr,
    BH: tl.constexpr, BD: tl.constexpr, INTERLEAVED: tl.constexpr,
):
    token = tl.program_id(0)
    group = tl.program_id(1)
    h = tl.arange(0, BH)
    d = tl.arange(0, BD)
    position = tl.load(P + token * PS)
    c = tl.load(C + position * CS + d, d < D // 2, 0).to(tl.float32)
    s = tl.load(S + position * SS + d, d < D // 2, 0).to(tl.float32)
    if INTERLEAVED:
        lo, hi = 2 * d, 2 * d + 1
    else:
        lo, hi = d, d + D // 2
    if group < tl.cdiv(NQ, BH):
        heads = group * BH + h
        base = Q + token * QS + heads[:, None] * QH
        a_ptr = base + lo[None, :] * QD
        b_ptr = base + hi[None, :] * QD
        mask = (heads[:, None] < NQ) & (d[None, :] < D // 2)
    else:
        heads = (group - tl.cdiv(NQ, BH)) * BH + h
        base = K + token * KS + heads[:, None] * KH
        a_ptr = base + lo[None, :] * KD
        b_ptr = base + hi[None, :] * KD
        mask = (heads[:, None] < NK) & (d[None, :] < D // 2)
    # Each pair belongs to exactly one lane/program: both reads precede stores.
    a = tl.load(a_ptr, mask, 0).to(tl.float32)
    b = tl.load(b_ptr, mask, 0).to(tl.float32)
    lower = a * c[None, :] + b * (-s[None, :])
    upper = b * c[None, :] + a * s[None, :]
    tl.store(a_ptr, lower, mask)
    tl.store(b_ptr, upper, mask)


def rope_parallel(q, k, cos, sin, positions, *, bh=4, warps=4, interleaved=False):
    assert q.ndim == k.ndim == 3 and q.shape[0] == k.shape[0]
    assert q.shape[-1] == k.shape[-1] and q.shape[-1] % 2 == 0
    d = q.shape[-1]
    rope_parallel_kernel[(q.shape[0], triton.cdiv(q.shape[1], bh) + triton.cdiv(k.shape[1], bh))](
        q, k, cos, sin, positions,
        *q.stride(), *k.stride(), cos.stride(0), sin.stride(0), positions.stride(0),
        q.shape[1], k.shape[1], d, bh, triton.next_power_of_2(d // 2), interleaved,
        num_warps=warps,
    )
    return q, k
