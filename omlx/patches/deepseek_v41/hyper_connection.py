# SPDX-License-Identifier: MIT
"""mHC normalization and four-stream mixing kernels with FP32 accumulation."""

from functools import cache

import mlx.core as mx

_SOURCE = r"""
    const uint row = thread_position_in_grid.x;
    uint rows = 1;
    for (int dim = 0; dim + 2 < x_ndim; ++dim) rows *= x_shape[dim];
    if (row >= rows) return;
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

_LANE_SOURCE = r"""
    // One lane per matrix entry, 16 lanes per row. Each lane repeats the
    // serial per-row and per-column sum order, so results match exactly.
    const uint lane = thread_index_in_simdgroup;
    const uint first = lane & ~15u;
    const uint row = thread_position_in_grid.x / 16;
    const uint entry = lane % 16, r = entry / 4, c = entry % 4;
    uint rows = 1;
    for (int dim = 0; dim + 2 < x_ndim; ++dim) rows *= x_shape[dim];
    float value = x[(row < rows ? row : 0) * 16 + entry];
    for (int iteration = 0; iteration < ITERS; ++iteration) {
        if (iteration > 0) {
            float total = 0.0f;
            for (uint i = 0; i < 4; ++i)
                total = simd_shuffle(value, first + r * 4 + i) + total;
            total = total + eps[0];
            value /= total;
        }
        float total = 0.0f;
        for (uint i = 0; i < 4; ++i)
            total = simd_shuffle(value, first + i * 4 + c) + total;
        total = total + eps[0];
        value /= total;
    }
    if (row < rows) y[row * 16 + entry] = value;
"""


@cache
def _sinkhorn_kernel(lanes=False):
    return mx.fast.metal_kernel(
        name="deepseek_v41_sinkhorn_lanes" if lanes else "deepseek_v41_sinkhorn",
        input_names=["x", "eps"],
        output_names=["y"],
        source=_LANE_SOURCE if lanes else _SOURCE,
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
    # Decode and verify rows are latency bound; long prefill chunks have
    # enough rows to fill the GPU with one thread per row.
    lanes = rows <= 64
    return _sinkhorn_kernel(lanes)(
        inputs=[comb, mx.array([eps], mx.float32)],
        template=[("ITERS", max(1, iters))],
        grid=((rows + 1) // 2 * 32 if lanes else rows, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[comb.shape],
        output_dtypes=[mx.float32],
    )[0]


_POST_SOURCE = r"""
    const uint z = thread_position_in_grid.x;
    uint size = D;
    for (int dim = 0; dim + 1 < x_ndim; ++dim) size *= x_shape[dim];
    if (z >= size) return;
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
        template=[("T", x.dtype), ("D", x.shape[-1])],
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


# RM rows per threadgroup share each fn load, and each of the NT / 32
# simdgroups multiplies one contiguous K slice on 8x8 fp32 simdgroup matrices.
# The slice partials and squares are summed in threadgroup memory.
_PROJECTION_SOURCE = r"""
    constexpr int NSG = NT / 32;
    constexpr int RT = RM / 8;
    constexpr int KS = K / NSG;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_position_in_threadgroup.x;
    const int row0 = int(threadgroup_position_in_grid.x) * RM;
    int rows = 1;
    for (int dim = 0; dim + 2 < x_ndim; ++dim) rows *= x_shape[dim];
    threadgroup float part[NSG][RM][24];
    threadgroup float square_part[NSG][RM];

    // This lane's two elements of an 8x8 matrix: (fm, fc) and (fm, fc + 1).
    const short qid = short(lane / 4);
    const short fm = (qid & 4) + short((lane / 2) % 4);
    const short fc = (qid & 2) * 2 + short(lane % 2) * 2;
    bool valid[RT];
    const device T* xa[RT];
    for (int t = 0; t < RT; ++t) {
        const int row = row0 + t * 8 + fm;
        valid[t] = row < rows;
        xa[t] = x + (size_t)(valid[t] ? row : row0) * K + fc;
    }
    // B[k][n] = fn[n][k].
    const device float* fb = fn + fc * K + fm;
    simdgroup_matrix<float, 8, 8> a, b0, b1, b2;
    simdgroup_matrix<float, 8, 8> c[RT][3];
    float square[RT];
    for (int t = 0; t < RT; ++t) {
        square[t] = 0.0f;
        for (int j = 0; j < 3; ++j) c[t][j] = simdgroup_matrix<float, 8, 8>(0.0f);
    }
    const int k_begin = int(sg) * KS;
    for (int k0 = k_begin; k0 < k_begin + KS; k0 += 8) {
        reinterpret_cast<thread float2&>(b0.thread_elements()) =
            float2(fb[k0], fb[K + k0]);
        reinterpret_cast<thread float2&>(b1.thread_elements()) =
            float2(fb[8 * K + k0], fb[9 * K + k0]);
        reinterpret_cast<thread float2&>(b2.thread_elements()) =
            float2(fb[16 * K + k0], fb[17 * K + k0]);
        for (int t = 0; t < RT; ++t) {
            const float2 value = valid[t]
                ? float2(*(const device vec<T, 2>*)(xa[t] + k0)) : float2(0.0f);
            square[t] = fma(value.x, value.x, square[t]);
            square[t] = fma(value.y, value.y, square[t]);
            reinterpret_cast<thread float2&>(a.thread_elements()) = value;
            simdgroup_multiply_accumulate(c[t][0], a, b0, c[t][0]);
            simdgroup_multiply_accumulate(c[t][1], a, b1, c[t][1]);
            simdgroup_multiply_accumulate(c[t][2], a, b2, c[t][2]);
        }
    }
    for (int t = 0; t < RT; ++t) {
        // Lane bits 0 and 3 select the column, so these lanes share row fm.
        float total = square[t];
        total += simd_shuffle_xor(total, 1);
        total += simd_shuffle_xor(total, 8);
        if ((lane & 9) == 0) square_part[sg][t * 8 + fm] = total;
        for (int j = 0; j < 3; ++j)
            simdgroup_store(c[t][j], &part[sg][t * 8][j * 8], 24);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int o = int(tid); o < RM * 24; o += NT) {
        const int m = o / 24, j = o % 24;
        if (row0 + m >= rows) continue;
        float value = 0.0f, total = 0.0f;
        for (int s = 0; s < NSG; ++s) {
            value += part[s][m][j];
            total += square_part[s][m];
        }
        y[(row0 + m) * 24 + j] = value * rsqrt(total / float(K) + eps[0]);
    }
"""

# Rows per threadgroup and threads (simdgroups x 32) of the projection.
_ROWS, _THREADS = 16, 512
# Widths the K slices split evenly (8 columns per simdgroup step).
PROJECTION_WIDTH_MULTIPLE = 8 * _THREADS // 32


@cache
def _projection_kernel():
    return mx.fast.metal_kernel(
        name="deepseek_v41_hc_projection_tiled",
        input_names=["x", "fn", "eps"],
        output_names=["y"],
        source=_PROJECTION_SOURCE,
        header="#include <metal_simdgroup_matrix>\n",
    )


def fused_hc_projection(x, fn, eps):
    """Project BF16 residual streams with FP32 weights and accumulators.

    The stream width (4 x dim) must be a multiple of PROJECTION_WIDTH_MULTIPLE.
    """
    width = x.shape[-1] * 4
    rows = x.size // width
    return _projection_kernel()(
        inputs=[x, fn, mx.array([eps], mx.float32)],
        template=[("T", x.dtype), ("K", width), ("RM", _ROWS), ("NT", _THREADS)],
        grid=((rows + _ROWS - 1) // _ROWS * _THREADS, 1, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(*x.shape[:-2], 24)],
        output_dtypes=[mx.float32],
    )[0]
