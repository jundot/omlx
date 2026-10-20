# SPDX-License-Identifier: MIT
"""MXFP8 GEMV for decode rows and short verify blocks.

Both kernels repeat the arithmetic of MLX's fp_qmv_fast (one row) and
fp_qmv_wide (four or five rows), so results match mx.quantized_matmul bit for
bit. Only the mapping of output rows to SIMD groups changes.
"""

from functools import cache

import mlx.core as mx

_HEADER = r"""
inline float v41_fp8_e4m3(uchar bits) {
    ushort v = bits & 127;
    ushort sign_bit = ((ushort)((bits >> 7) & 1)) << 15;
    ushort u = (v << 7) | (((v + 1) >> 7) << 14) | sign_bit;
    half converted = as_type<half>(u);
    half scaled = converted * 256.0;
    return static_cast<float>(scaled);
}
inline float v41_e8m0(uchar bits) {
    uint32_t out = (bits == 0 ? 0x400000 : (static_cast<uint16_t>(bits) << 23));
    return as_type<float>(out);
}
// MLX qdot for 8-bit values, kept as a separate function like the original.
inline float v41_qdot8(
    const device uint8_t* w, const thread float* x_thread, float scale) {
    float accum = 0;
    for (int i = 0; i < 8; i++) {
        accum += x_thread[i] * v41_fp8_e4m3(w[i]);
    }
    return scale * accum;
}
"""

# fp_qmv_fast with one output row per SIMD group instead of four, so thin
# projections run four times as many SIMD groups. Each lane keeps its 8-value
# slice of every 256-value block and the same accumulation order.
_ONE_ROW_SOURCE = r"""
    const uint3 tid = threadgroup_position_in_grid;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    constexpr int G = K / 32;
    const int out_row = tid.y * 2 + simd_gid;
    const device uint8_t* ws =
        (const device uint8_t*)w + size_t(out_row) * K + simd_lid * 8;
    const device uint8_t* sl = scales + size_t(out_row) * G + simd_lid / 4;
    const device T* xp = x + simd_lid * 8;
    float result = 0;
    for (int k = 0; k < K; k += 256) {
        float x_thread[8];
        for (int i = 0; i < 8; i++) x_thread[i] = xp[i];
        const float s = v41_e8m0(sl[0]);
        result += v41_qdot8(ws, x_thread, s);
        ws += 256;
        sl += 8;
        xp += 256;
    }
    result = simd_sum(result);
    if (simd_lid == 0) y[out_row] = static_cast<T>(result);
"""

# fp_qmv_wide (16 K lanes, one tile of M vectors) where each lane serves two
# consecutive rows, so every activation chunk it loads feeds both rows. Group
# order, expressions and the shuffle-down reduction are unchanged.
_ROWS_SOURCE = r"""
    const uint3 tid = threadgroup_position_in_grid;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    constexpr int G = K / 32;
    constexpr int R = 2;
    const short k_lane = simd_lid % 16;
    const short half_id = simd_lid / 16;
    const int row0 = tid.y * (4 * R) + simd_gid * (2 * R) + half_id * R;
    const device uint8_t* wrow[R];
    const device uint8_t* srow[R];
    for (int r = 0; r < R; r++) {
        const int row = min(row0 + r, N - 1);
        wrow[r] = (const device uint8_t*)w + size_t(row) * K;
        srow[r] = scales + size_t(row) * G;
    }
    const device T* xv[M];
    for (int v = 0; v < M; v++) xv[v] = x + v * K;
    float result[R][M];
    for (int r = 0; r < R; r++) for (int v = 0; v < M; v++) result[r][v] = 0;
    for (int g = k_lane; g < G; g += 16) {
        const int k0 = g * 32;
        float s[R];
        for (int r = 0; r < R; r++) s[r] = v41_e8m0(srow[r][g]);
        float acc[R][M];
        for (int r = 0; r < R; r++) for (int v = 0; v < M; v++) acc[r][v] = 0;
        for (int j = 0; j < 8; j++) {
            float4 xq[M];
            for (int v = 0; v < M; v++)
                xq[v] = float4(((const device vec<T, 4>*)(xv[v] + k0))[j]);
            for (int r = 0; r < R; r++) {
                const device uint8_t* wg = wrow[r] + k0 + 4 * j;
                const float4 w4 = float4(
                    v41_fp8_e4m3(wg[0]), v41_fp8_e4m3(wg[1]),
                    v41_fp8_e4m3(wg[2]), v41_fp8_e4m3(wg[3]));
                for (int v = 0; v < M; v++) acc[r][v] += dot(w4, xq[v]);
            }
        }
        for (int r = 0; r < R; r++)
            for (int v = 0; v < M; v++) result[r][v] += s[r] * acc[r][v];
    }
    for (int r = 0; r < R; r++) {
        for (int v = 0; v < M; v++) {
            result[r][v] += simd_shuffle_down(result[r][v], 8);
            result[r][v] += simd_shuffle_down(result[r][v], 4);
            result[r][v] += simd_shuffle_down(result[r][v], 2);
            result[r][v] += simd_shuffle_down(result[r][v], 1);
        }
    }
    if (k_lane == 0)
        for (int r = 0; r < R; r++)
            if (row0 + r < N)
                for (int v = 0; v < M; v++)
                    y[v * N + row0 + r] = static_cast<T>(result[r][v]);
"""


@cache
def _kernel(rows):
    return mx.fast.metal_kernel(
        name="v41_mxfp8_gemv_rows" if rows else "v41_mxfp8_gemv_one_row",
        input_names=["w", "scales", "x"],
        output_names=["y"],
        source=_ROWS_SOURCE if rows else _ONE_ROW_SOURCE,
        header=_HEADER,
    )


def mxfp8_gemv(x, weight, scales):
    """Return ``mx.quantized_matmul(x, weight, scales, mode="mxfp8")`` for one,
    four or five rows, or None when MLX would pick another kernel."""
    k = x.shape[-1]
    n = weight.shape[0]
    rows = x.shape[-2] if x.ndim > 1 else 1
    if (
        x.dtype not in (mx.bfloat16, mx.float16)
        or x.size != rows * k
        or weight.ndim != 2
        or weight.dtype != mx.uint32
        or weight.shape[1] * 4 != k
        or scales.shape != (n, k // 32)
        or k % 256
        or mx.default_device() != mx.gpu
    ):
        return None
    if rows == 1:
        # One row per SIMD group pays off for long rows and few outputs; MLX
        # already fills the GPU when the output is wide.
        if n % 8 or n > 8192 or k < 4096:
            return None
        return _kernel(False)(
            inputs=[weight, scales, x],
            template=[("T", x.dtype), ("K", k)],
            grid=(32, n, 1),
            threadgroup=(32, 2, 1),
            output_shapes=[(*x.shape[:-1], n)],
            output_dtypes=[x.dtype],
        )[0]
    # Two and three rows run fewer threadgroups than MLX and measure slower.
    if rows not in (4, 5):
        return None
    return _kernel(True)(
        inputs=[weight, scales, x],
        template=[("T", x.dtype), ("K", k), ("N", n), ("M", rows)],
        grid=(32, (n + 7) // 8 * 2, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(*x.shape[:-1], n)],
        output_dtypes=[x.dtype],
    )[0]
