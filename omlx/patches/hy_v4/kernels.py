# SPDX-License-Identifier: Apache-2.0
"""Fused Metal kernels for Hy4's identity hyper-connections (iHC).

``hc_pre`` replaces the fp32 cast, RMS, ``[rows, HC*D] x [HC*D, MIX]`` mix
product, gate sigmoids and the weighted stream collapse with one kernel that
reads the streams twice: once for the sum of squares and the mixes, once for
the collapse. One threadgroup owns one row and reduces in a fixed order, so a
row's result does not depend on the chunk length or the batch it runs in.

``MIX`` is ``2 * HC`` for a layer connection (pre and post gates) and ``HC``
for the final head (pre gates only).
"""

from __future__ import annotations

from functools import cache

import mlx.core as mx

_THREADS = 256

_SOURCE = r"""
    constexpr int K = HC * D;
    constexpr int NSG = NT / 32;
    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;

    threadgroup float red[NSG][MIX + 1];
    threadgroup float gate[MIX];

    const device T* xr = x + size_t(row) * K;
    float sq = 0.0f;
    float acc[MIX];
    for (int j = 0; j < MIX; ++j) acc[j] = 0.0f;
    for (int k = int(tid); k < K; k += NT) {
        const float v = float(xr[k]);
        sq += v * v;
        const device float* w = fnT + size_t(k) * MIX;
        for (int j = 0; j < MIX; ++j) acc[j] += v * w[j];
    }
    sq = simd_sum(sq);
    for (int j = 0; j < MIX; ++j) acc[j] = simd_sum(acc[j]);
    if (lane == 0) {
        red[sg][MIX] = sq;
        for (int j = 0; j < MIX; ++j) red[sg][j] = acc[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < MIX) {
        float total = 0.0f;
        float total_sq = 0.0f;
        for (int s = 0; s < NSG; ++s) {
            total += red[s][tid];
            total_sq += red[s][MIX];
        }
        const float inv = metal::precise::rsqrt(total_sq / float(K) + params[0]);
        const float mix = total * inv;
        const int half_idx = int(tid) / HC;
        const float logit = mix * scale[half_idx] + base[tid];
        const float sig = 1.0f / (1.0f + metal::precise::exp(-logit));
        // params: norm eps, gate eps, post magnitude.
        gate[tid] = half_idx == 0 ? sig + params[1] : params[2] * sig + params[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (MIX > HC && tid < HC) {
        post[size_t(row) * HC + tid] = gate[HC + tid];
    }
    device T* out = collapsed + size_t(row) * D;
    for (int d = int(tid); d < D; d += NT) {
        float sum = 0.0f;
        for (int h = 0; h < HC; ++h) sum += gate[h] * float(xr[h * D + d]);
        out[d] = T(sum);
    }
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="omlx_hy4_hc_pre",
        input_names=["x", "fnT", "base", "scale", "params"],
        output_names=["collapsed", "post"],
        source=_SOURCE,
    )


def available() -> bool:
    return mx.metal.is_available()


def hc_pre(
    streams: mx.array,
    fn_t: mx.array,
    base: mx.array,
    scale: mx.array,
    norm_eps: float,
    gate_eps: float,
    magnitude: float | None,
):
    """Fused iHC gates and collapse.

    ``streams`` is [..., HC, D]; ``fn_t`` is the fp32 mix weight transposed to
    [HC*D, MIX]. Returns (collapsed [..., D], post [..., HC] or None).
    """
    *lead, hc, d = streams.shape
    rows = 1
    for n in lead:
        rows *= n
    mix = fn_t.shape[1]
    with_post = magnitude is not None
    params = mx.array(
        [norm_eps, gate_eps, magnitude if with_post else 0.0], dtype=mx.float32
    )
    # The head's kernel (MIX == HC) never writes ``post``.
    outs = _kernel()(
        inputs=[streams.reshape(rows, hc * d), fn_t, base, scale, params],
        template=[
            ("T", streams.dtype),
            ("HC", hc),
            ("D", d),
            ("MIX", mix),
            ("NT", _THREADS),
        ],
        grid=(rows * _THREADS, 1, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(rows, d), (rows, hc)],
        output_dtypes=[streams.dtype, mx.float32],
    )
    collapsed = outs[0].reshape(*lead, d)
    post = outs[1].reshape(*lead, hc) if with_post else None
    return collapsed, post
