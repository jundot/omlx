# SPDX-License-Identifier: MIT
"""mHC normalization and four-stream mixing kernels with FP32 accumulation."""

from functools import cache

import mlx.core as mx

_SOURCE = r"""
    const uint row = thread_position_in_grid.x;
    if (row >= ROWS) return;
    float values[16];
    for (int i = 0; i < 16; ++i) values[i] = x[row * 16 + i];
    for (int iteration = 0; iteration < ITERS; ++iteration) {
        if (iteration > 0) {
            for (int r = 0; r < 4; ++r) {
                float total = 0.0f;
                for (int c = 0; c < 4; ++c) total = values[r * 4 + c] + total;
                total = total + eps[0];
                for (int c = 0; c < 4; ++c) values[r * 4 + c] /= total;
            }
        }
        for (int c = 0; c < 4; ++c) {
            float total = 0.0f;
            for (int r = 0; r < 4; ++r) total = values[r * 4 + c] + total;
            total = total + eps[0];
            for (int r = 0; r < 4; ++r) values[r * 4 + c] /= total;
        }
    }
    for (int i = 0; i < 16; ++i) y[row * 16 + i] = values[i];
"""


@cache
def _sinkhorn_kernel():
    return mx.fast.metal_kernel(
        name="deepseek_v41_sinkhorn",
        input_names=["x", "eps"],
        output_names=["y"],
        source=_SOURCE,
        header=("#pragma clang fp reassociate(off)\n#pragma clang fp contract(off)\n"),
    )


def sinkhorn_reference(comb, eps, iters):
    comb /= mx.sum(comb, -2, keepdims=True) + eps
    for _ in range(iters - 1):
        comb /= mx.sum(comb, -1, keepdims=True) + eps
        comb /= mx.sum(comb, -2, keepdims=True) + eps
    return comb


def sinkhorn(comb, eps, iters):
    """Fuse only normalization; preserve softmax and mHC projection arithmetic."""
    if (
        comb.shape[-2:] != (4, 4)
        or comb.dtype != mx.float32
        or mx.default_device() != mx.gpu
        or not comb.size
    ):
        return sinkhorn_reference(comb, eps, iters)
    rows = comb.size // 16
    return _sinkhorn_kernel()(
        inputs=[comb, mx.array([eps], mx.float32)],
        template=[("ROWS", rows), ("ITERS", max(1, iters))],
        grid=(rows, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[comb.shape],
        output_dtypes=[mx.float32],
    )[0]


_POST_SOURCE = r"""
    const uint z = thread_position_in_grid.x;
    if (z >= ROWS * D) return;
    const uint row = z / D, d = z % D;
    float values[4];
    for (uint i = 0; i < 4; ++i)
        values[i] = float(residual[(row * 4 + i) * D + d]);
    const float value = float(x[z]);
    for (uint j = 0; j < 4; ++j) {
        float sum = 0.0f;
        for (uint i = 0; i < 4; ++i)
            sum = fma(comb[row * 16 + i * 4 + j], values[i], sum);
        y[(row * 4 + j) * D + d] = T(post[row * 4 + j] * value + sum);
    }
"""


@cache
def _post_kernel():
    return mx.fast.metal_kernel(
        name="deepseek_v41_hc_post",
        input_names=["x", "residual", "post", "comb"],
        output_names=["y"],
        source=_POST_SOURCE,
        header="#pragma clang fp contract(off)\n",
    )


def fused_hc_post(x, residual, post, comb):
    """Mix four residual streams without expanding the residual to FP32."""
    return _post_kernel()(
        inputs=[x, residual, post, comb],
        template=[("T", x.dtype), ("ROWS", x.size // x.shape[-1]), ("D", x.shape[-1])],
        grid=(x.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[residual.shape],
        output_dtypes=[x.dtype],
    )[0]


_PRE_NORM_SOURCE = r"""
    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint simd = simdgroup_index_in_threadgroup;
    threadgroup float sums[8];
    float values[(D + 255) / 256];
    float total = 0.0f;
    for (uint t = 0; t < (D + 255) / 256; ++t) {
        const uint d = tid + 256 * t;
        float value = 0.0f;
        if (d < D) {
            for (uint i = 0; i < 4; ++i)
                value = float(x[(row * 4 + i) * D + d]) * pre[row * 4 + i] + value;
            // The separate hc_pre output is rounded before normalization.
            value = float(T(value));
        }
        values[t] = value;
        total = total + value * value;
    }
    total = simd_sum(total);
    if (lane == 0) sums[simd] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        float sum = lane < 8 ? sums[lane] : 0.0f;
        sum = simd_sum(sum);
        if (lane == 0) sums[0] = rsqrt(sum / float(D) + eps[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = 0; t < (D + 255) / 256; ++t) {
        const uint d = tid + 256 * t;
        if (d < D) y[row * D + d] = T((values[t] * sums[0]) * float(weight[d]));
    }
"""


@cache
def _pre_norm_kernel():
    return mx.fast.metal_kernel(
        name="deepseek_v41_hc_pre_norm",
        input_names=["x", "pre", "weight", "eps"],
        output_names=["y"],
        source=_PRE_NORM_SOURCE,
        header="#pragma clang fp contract(off)\n",
    )


def fused_hc_pre_norm(x, pre, weight, eps):
    """Preserve the intermediate dtype cast before RMS normalization."""
    width = x.shape[-1]
    rows = x.size // (4 * width)
    return _pre_norm_kernel()(
        inputs=[x, pre, weight, mx.array([eps], mx.float32)],
        template=[("T", x.dtype), ("D", width)],
        grid=(rows * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(*x.shape[:-2], width)],
        output_dtypes=[x.dtype],
    )[0]


_PROJECTION_WARP_SOURCE = r"""
    const uint row = threadgroup_position_in_grid.x;
    const uint block = threadgroup_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    float sums[12];
    for (uint j = 0; j < 12; ++j) sums[j] = 0.0f;
    float square = 0.0f;
    for (uint k = lane; k < D; k += 32) {
        const float value = float(x[row * D + k]);
        square = fma(value, value, square);
        for (uint j = 0; j < 12; ++j)
            sums[j] = fma(value, fn[(block * 12 + j) * D + k], sums[j]);
    }
    const float inv = rsqrt(simd_sum(square) / float(D) + eps[0]);
    for (uint j = 0; j < 12; ++j) {
        const float value = simd_sum(sums[j]);
        if (lane == 0) y[row * 24 + block * 12 + j] = value * inv;
    }
"""

_PROJECTION_SOURCE = r"""
    const uint row = threadgroup_position_in_grid.x;
    const uint block = threadgroup_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    threadgroup float scratch[13 * 8];
    float sums[12];
    for (uint j = 0; j < 12; ++j) sums[j] = 0.0f;
    float square = 0.0f;
    for (uint k = tid; k < D; k += 256) {
        const float value = float(x[row * D + k]);
        square = fma(value, value, square);
        for (uint j = 0; j < 12; ++j)
            sums[j] = fma(value, fn[(block * 12 + j) * D + k], sums[j]);
    }
    for (uint j = 0; j < 12; ++j) {
        const float value = simd_sum(sums[j]);
        if (lane == 0) scratch[j * 8 + sg] = value;
    }
    const float sq = simd_sum(square);
    if (lane == 0) scratch[96 + sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float q = lane < 8 ? scratch[96 + lane] : 0.0f;
        q = simd_sum(q);
        const float inv = rsqrt(q / float(D) + eps[0]);
        for (uint j = 0; j < 12; ++j) {
            float value = lane < 8 ? scratch[j * 8 + lane] : 0.0f;
            value = simd_sum(value);
            if (lane == 0) y[row * 24 + block * 12 + j] = value * inv;
        }
    }
"""


@cache
def _projection_kernel(warp=False):
    return mx.fast.metal_kernel(
        name=(
            "deepseek_v41_hc_projection_warp" if warp else "deepseek_v41_hc_projection"
        ),
        input_names=["x", "fn", "eps"],
        output_names=["y"],
        source=_PROJECTION_WARP_SOURCE if warp else _PROJECTION_SOURCE,
    )


def fused_hc_projection(x, fn, eps):
    """Project BF16 residual streams with FP32 weights and accumulators."""
    width = x.shape[-1] * 4
    rows = x.size // width
    # Small batches need more warps to occupy the GPU.
    threads = 32 if rows >= 1024 else 256
    return _projection_kernel(warp=threads == 32)(
        inputs=[x, fn, mx.array([eps], mx.float32)],
        template=[("D", width)],
        grid=(rows * threads, 2, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(*x.shape[:-2], 24)],
        output_dtypes=[mx.float32],
    )[0]


def _make_hc_sinkhorn_only_kernel():
    """Fused sigmoid/softmax + sinkhorn for deferred mHC decode (from our V4.1 fork)."""
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None

    source = """
        uint tid  = thread_position_in_threadgroup.x;
        uint row  = threadgroup_position_in_grid.x;
        uint lane = tid % 32;
        uint sg   = tid / 32;

        constexpr int MIX      = (2 + HC) * HC;
        constexpr int BASE_OFF = 2 * HC;
        constexpr float EPS = EPS_INT * 1e-9;

        const device float* mix      = (const device float*)mixes + row * MIX;
        device float*       pre_out  = (device float*)pre + row * HC;
        device float*       post_out = (device float*)post + row * HC;
        device float*       comb_out = (device float*)comb + row * HC * HC;

        if (sg == 0) {
            const float pre_scale  = scale[0];
            const float post_scale = scale[1];
            const float comb_scale = scale[2];

            const float active = (lane < (uint)HC) ? 1.0f : 0.0f;
            const uint  llane  = metal::min(lane, (uint)(HC - 1));

            float pre_z  = mix[llane]      * pre_scale  + base[llane];
            float post_z = mix[HC + llane] * post_scale + base[HC + llane];
            float pre_v  = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + EPS;
            float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));

            if (lane < (uint)HC) {
                pre_out[lane]  = pre_v;
                post_out[lane] = post_v;
            }

            float4 v = (*(const device float4*)(mix  + BASE_OFF + llane * HC)
                            * comb_scale
                      + *(const device float4*)(base + BASE_OFF + llane * HC))
                     * active;

            float row_max = metal::max(metal::max(v.x, v.y),
                                       metal::max(v.z, v.w));
            float4 e = metal::fast::exp(v - row_max) * active;
            float4 r = e * (1.0f / (e.x + e.y + e.z + e.w + EPS))
                     + EPS * active;

            float4 col_inv = 1.0f / (float4(
                simd_sum(r.x), simd_sum(r.y),
                simd_sum(r.z), simd_sum(r.w)
            ) + EPS);
            r *= col_inv;

            for (int iter = 1; iter < ITERS; ++iter) {
                r *= (1.0f / (r.x + r.y + r.z + r.w + EPS)) * active;
                col_inv = 1.0f / (float4(
                    simd_sum(r.x), simd_sum(r.y),
                    simd_sum(r.z), simd_sum(r.w)
                ) + EPS);
                r *= col_inv;
            }

            if (lane < (uint)HC) {
                *(device float4*)(comb_out + lane * HC) = r;
            }
        }
    """

    return mx.fast.metal_kernel(
        name="hc_sinkhorn_only",
        input_names=["mixes", "scale", "base"],
        output_names=["pre", "post", "comb"],
        source=source,
        ensure_row_contiguous=True,
    )


_hc_sinkhorn_only_kernel = _make_hc_sinkhorn_only_kernel()


def hc_sinkhorn_only(mixes, scale, base, hc_mult, sinkhorn_iters, eps, batch_shape):
    """Metal fused mix→(pre,post,comb); falls back to None if unavailable."""
    if (
        _hc_sinkhorn_only_kernel is None
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or hc_mult != 4
        or mixes.dtype not in (mx.float32, mx.bfloat16, mx.float16)
    ):
        return None

    mixes_f = mixes.astype(mx.float32)
    mix_flat = mixes_f.reshape(-1, mixes_f.shape[-1])
    rows = mix_flat.shape[0]
    pre, post, comb = _hc_sinkhorn_only_kernel(
        inputs=[mix_flat, scale.astype(mx.float32), base.astype(mx.float32)],
        template=[
            ("HC", hc_mult),
            ("ITERS", sinkhorn_iters),
            ("EPS_INT", round(eps / 1e-9)),
        ],
        grid=(rows * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[
            (rows, hc_mult),
            (rows, hc_mult),
            (rows, hc_mult, hc_mult),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    pre = pre.reshape(*batch_shape, hc_mult)
    post = post.reshape(*batch_shape, hc_mult)
    comb = comb.reshape(*batch_shape, hc_mult, hc_mult)
    return pre, post, comb

