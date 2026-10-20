# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Grouped-head RoPE for Iluvatar, with paired reads and separate outputs."""
import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


@triton.jit
def rope_parallel_kernel(
    Q, K, OQ, OK, C, S, P,
    QS: tl.constexpr, QH: tl.constexpr, QD: tl.constexpr,
    KS: tl.constexpr, KH: tl.constexpr, KD: tl.constexpr,
    OQS: tl.constexpr, OQH: tl.constexpr, OQD: tl.constexpr,
    OKS: tl.constexpr, OKH: tl.constexpr, OKD: tl.constexpr,
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
        out = OQ + token * OQS + heads[:, None] * OQH
        out_a, out_b = out + lo[None, :] * OQD, out + hi[None, :] * OQD
        mask = (heads[:, None] < NQ) & (d[None, :] < D // 2)
    else:
        heads = (group - tl.cdiv(NQ, BH)) * BH + h
        base = K + token * KS + heads[:, None] * KH
        a_ptr = base + lo[None, :] * KD
        b_ptr = base + hi[None, :] * KD
        out = OK + token * OKS + heads[:, None] * OKH
        out_a, out_b = out + lo[None, :] * OKD, out + hi[None, :] * OKD
        mask = (heads[:, None] < NK) & (d[None, :] < D // 2)
    # Each pair belongs to exactly one lane/program: both reads precede stores.
    a = tl.load(a_ptr, mask, 0).to(tl.float32)
    b = tl.load(b_ptr, mask, 0).to(tl.float32)
    lower = a * c[None, :] + b * (-s[None, :])
    upper = b * c[None, :] + a * s[None, :]
    tl.store(out_a, lower, mask)
    tl.store(out_b, upper, mask)


def rope_parallel(q, k, cos, sin, positions, *, bh=None, warps=4,
                  interleaved=False, inplace=True):
    assert q.ndim == k.ndim == 3 and q.shape[0] == k.shape[0]
    assert q.shape[-1] == k.shape[-1] and q.shape[-1] % 2 == 0
    d = q.shape[-1]
    oq = q if inplace else torch.empty(q.shape, dtype=q.dtype, device=q.device)
    ok = k if inplace else torch.empty(k.shape, dtype=k.dtype, device=k.device)
    if not q.shape[0]:
        return oq, ok
    if bh is None:
        # Fixed rule selected from EXP-0002; no serving-time autotuning.
        bh = 2 if q.shape[0] == 1 else (4 if q.shape[0] <= 8 else (8 if q.shape[0] <= 128 else 16))
    with torch_device_fn.device(q.device):
        rope_parallel_kernel[(q.shape[0], triton.cdiv(q.shape[1], bh) + triton.cdiv(k.shape[1], bh))](
            q, k, oq, ok, cos, sin, positions,
            *q.stride(), *k.stride(), *oq.stride(), *ok.stride(),
            cos.stride(0), sin.stride(0), positions.stride(0),
            q.shape[1], k.shape[1], d, bh, triton.next_power_of_2(d // 2), interleaved,
            num_warps=warps,
        )
    return oq, ok
