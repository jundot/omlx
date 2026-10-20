# SPDX-License-Identifier: Apache-2.0
"""Packed attention with 64-key online softmax and BF16 PV probabilities."""

from functools import cache

import mlx.core as mx

_READ = r"""
#include <metal_stdlib>
using namespace metal;
inline float rounded_fp4(uchar code) {
    const float table[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return (code & 8 ? -1.0f : 1.0f) * table[code & 7];
}
inline float rounded_fp8(uchar code) {
    const uint a = code & 127, exponent = a >> 3, mantissa = a & 7;
    const float value = exponent == 0 ? float(mantissa) * 0x1p-9f
        : as_type<float>(((exponent + 120u) << 23) | (mantissa << 20));
    return code & 128 ? -value : value;
}
template <typename W, typename P>
inline float read_kv(W window, P pooled,
                     int row, int d, int dim, bool compressed, float pooled_scale, int window_exp, bool reuse) {
    if (compressed) {
        const size_t base = size_t(row) * (dim / 2 + dim / 16);
        const uchar code = (pooled[base + d / 2] >> ((d % 2) * 4)) & 15;
        return rounded_fp4(code) * (reuse ? pooled_scale : rounded_fp8(pooled[base + dim / 2 + d / 16]));
    }
    const size_t base = size_t(row) * (dim + dim / 32);
    return ldexp(rounded_fp8(window[base + d]), reuse ? window_exp : int(window[base + dim + d / 32]) - 127);
}
"""

_FUSED = r"""
    const uint lane = thread_index_in_simdgroup, block = simdgroup_index_in_threadgroup;
    const uint head = threadgroup_position_in_grid.x, query = threadgroup_position_in_grid.y;
    const int H = meta[0], W = meta[1], C = meta[2], NW = meta[3], NC = meta[4];
    threadgroup float scores[NB * CHUNK], maxima[NB], probabilities[NB * CHUNK];
    float qv[D / 32];
    for (int v = 0; v < D / 32; ++v) qv[v] = float(q[(query * H + head) * D + lane * (D / 32) + v]);
    float maximum = -1e30f;
    for (int j = 0; j < CHUNK; ++j) {
        const int slot = int(block) * CHUNK + j;
        const bool compressed = slot >= W;
        const int row = slot >= W + C ? -1 : compressed ? ci[query * C + slot - W] : wi[query * W + slot];
        float score = -INFINITY;
        if (row >= 0 && row < (compressed ? NC : NW)) {
            constexpr bool REUSE = D == 32 || D == 64 || D == 128 || D == 256 || D == 512;
            const int first = lane * (D / 32);
            const size_t row_base = size_t(row) * (compressed ? (D / 2 + D / 16) : (D + D / 32));
            const float pooled_scale = REUSE && compressed ? rounded_fp8(pooled[row_base + D / 2 + first / 16]) : 0.0f;
            const int window_exp = REUSE && !compressed ? int(window[row_base + D + first / 32]) - 127 : 0;
            float dot = 0.0f;
            for (int v = 0; v < D / 32; ++v) dot = fma(qv[v], read_kv(window, pooled, row, first + v, D, compressed, pooled_scale, window_exp, REUSE), dot);
            score = simd_sum(dot) * scalep[0];
        }
        maximum = max(maximum, score);
        if (lane == 0) scores[block * CHUNK + j] = score;
    }
    if (lane == 0) maxima[block] = maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum = -1e30f;
    for (uint b = lane; b <= min(block | uint(64 / CHUNK - 1), uint(NB - 1)); b += 32) maximum = max(maximum, maxima[b]);
    maximum = simd_max(maximum);
    float denominator = 0.0f;
    for (uint j = lane; j < CHUNK; j += 32) {
        const float p = exp(scores[block * CHUNK + j] - maximum);
        denominator += p;
        probabilities[block * CHUNK + j] = float(bfloat16_t(p));
    }
    denominator = simd_sum(denominator);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float acc[D / 32];
    for (int v = 0; v < D / 32; ++v) acc[v] = 0.0f;
    for (int j = 0; j < CHUNK; ++j) {
        const int slot = int(block) * CHUNK + j;
        const bool compressed = slot >= W;
        const int row = slot >= W + C ? -1 : compressed ? ci[query * C + slot - W] : wi[query * W + slot];
        if (row < 0 || row >= (compressed ? NC : NW)) continue;
        constexpr bool REUSE = D == 32 || D == 64 || D == 128 || D == 256 || D == 512;
        const int first = lane * (D / 32);
        const size_t row_base = size_t(row) * (compressed ? (D / 2 + D / 16) : (D + D / 32));
        const float pooled_scale = REUSE && compressed ? rounded_fp8(pooled[row_base + D / 2 + first / 16]) : 0.0f;
        const int window_exp = REUSE && !compressed ? int(window[row_base + D + first / 32]) - 127 : 0;
        const float p = probabilities[block * CHUNK + j];
        for (int v = 0; v < D / 32; ++v) acc[v] = fma(p, read_kv(window, pooled, row, first + v, D, compressed, pooled_scale, window_exp, REUSE), acc[v]);
    }
    const size_t out_base = ((size_t(query) * H + head) * NB + block) * (D + 2);
    for (int v = 0; v < D / 32; ++v) partial[out_base + lane * (D / 32) + v] = acc[v];
    if (lane == 0) { partial[out_base + D] = maximum; partial[out_base + D + 1] = denominator; }
"""

_SCORES = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint head = threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup;
    const uint query = threadgroup_position_in_grid.y, block = threadgroup_position_in_grid.z;
    const int H = meta[0], W = meta[1], C = meta[2], NW = meta[3], NC = meta[4], NB = meta[5];
    if (head >= H) return;
    float qv[D / 32];
    for (int v = 0; v < D / 32; ++v)
        qv[v] = float(q[(query * H + head) * D + lane * (D / 32) + v]);
    float maximum = -1e30f;
    const size_t base = (size_t(query) * H + head) * NB + block;
    for (int j = 0; j < 16; ++j) {
        const int slot = int(block) * 16 + j;
        const bool compressed = slot >= W;
        const int row = slot >= W + C ? -1 : compressed
            ? ci[query * C + slot - W] : wi[query * W + slot];
        float score = -INFINITY;
        if (row >= 0 && row < (compressed ? NC : NW)) {

            constexpr bool REUSE = D == 32 || D == 64 || D == 128 || D == 256 || D == 512;
            const int first = lane * (D / 32);
            const size_t row_base = size_t(row) * (compressed ? (D / 2 + D / 16) : (D + D / 32));
            const float pooled_scale = REUSE && compressed ? rounded_fp8(pooled[row_base + D / 2 + first / 16]) : 0.0f;
            const int window_exp = REUSE && !compressed ? int(window[row_base + D + first / 32]) - 127 : 0;
            float dot = 0.0f;
            for (int v = 0; v < D / 32; ++v) {
                const int d = lane * (D / 32) + v;
                const float value = read_kv(window, pooled, row, d, D, compressed, pooled_scale, window_exp, REUSE);
                dot = fma(qv[v], value, dot);
            }
            score = simd_sum(dot) * scalep[0];
        }
        maximum = max(maximum, score);
        if (lane == 0) scores[base * 16 + j] = score;
    }
    if (lane == 0) maxima[base] = maximum;
"""

_VALUES = r"""
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const uint head = threadgroup_position_in_grid.x * 4 + sg;
    const uint query = threadgroup_position_in_grid.y, block = threadgroup_position_in_grid.z;
    const int H = meta[0], W = meta[1], C = meta[2], NW = meta[3], NC = meta[4], NB = meta[5];
    if (head >= H) return;
    const size_t base = (size_t(query) * H + head) * NB;
    float maximum = -1e30f;
    // Four 16-key subparts share the maximum through their official 64-key tile.
    for (uint b = lane; b <= min(block | 3u, uint(NB - 1)); b += 32) maximum = max(maximum, maxima[base + b]);
    maximum = simd_max(maximum);
    threadgroup float probabilities[4 * 16];
    float denominator = 0.0f;
    for (uint j = lane; j < 16; j += 32) {
        const float p = exp(scores[(base + block) * 16 + j] - maximum);
        denominator += p;
        probabilities[sg * 16 + j] = float(bfloat16_t(p));
    }
    denominator = simd_sum(denominator);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float acc[D / 32];
    for (int v = 0; v < D / 32; ++v) acc[v] = 0.0f;
    for (int j = 0; j < 16; ++j) {
        const int slot = int(block) * 16 + j;
        const bool compressed = slot >= W;
        const int row = slot >= W + C ? -1 : compressed
            ? ci[query * C + slot - W] : wi[query * W + slot];
        if (row < 0 || row >= (compressed ? NC : NW)) continue;

            constexpr bool REUSE = D == 32 || D == 64 || D == 128 || D == 256 || D == 512;
            const int first = lane * (D / 32);
            const size_t row_base = size_t(row) * (compressed ? (D / 2 + D / 16) : (D + D / 32));
            const float pooled_scale = REUSE && compressed ? rounded_fp8(pooled[row_base + D / 2 + first / 16]) : 0.0f;
            const int window_exp = REUSE && !compressed ? int(window[row_base + D + first / 32]) - 127 : 0;
        const float p = probabilities[sg * 16 + j];
        for (int v = 0; v < D / 32; ++v) {
            const int d = lane * (D / 32) + v;
            acc[v] = fma(p, read_kv(window, pooled, row, d, D, compressed, pooled_scale, window_exp, REUSE), acc[v]);
        }
    }
    const size_t out = (base + block) * (D + 2);
    for (int v = 0; v < D / 32; ++v) partial[out + lane * (D / 32) + v] = acc[v];
    if (lane == 0) {
        partial[out + D] = maximum;
        partial[out + D + 1] = denominator;
    }
"""

_MERGE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint part = threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup;
    const uint head = part / (D / 32), d = (part % (D / 32)) * 32 + lane;
    const uint query = threadgroup_position_in_grid.y;
    const int H = meta[0], NB = meta[1];
    if (head >= H) return;
    const size_t base = (size_t(query) * H + head) * NB * (D + 2);
    float maximum = -1e30f, denominator = 0.0f, acc = 0.0f;
    for (int b = 0; b < NB; ++b) {
        const size_t pos = base + b * (D + 2);
        const float next_maximum = partial[pos + D];
        const float correction = (b % (64 / CHUNK) == 0) ? exp(maximum - next_maximum) : 1.0f;
        denominator = denominator * correction + partial[pos + D + 1];
        acc = acc * correction + partial[pos + d];
        maximum = next_maximum;
    }
    denominator += exp(float(sink[head]) - maximum);
    out[(query * H + head) * D + d] = T(acc / denominator);
"""


@cache
def _kernel(stage):
    inputs, outputs, source = {
        "fused": (
            ["q", "window", "pooled", "wi", "ci", "meta", "scalep"],
            ["partial"],
            _FUSED,
        ),
        "scores": (
            ["q", "window", "pooled", "wi", "ci", "meta", "scalep"],
            ["scores", "maxima"],
            _SCORES,
        ),
        "values": (
            ["scores", "maxima", "window", "pooled", "wi", "ci", "meta"],
            ["partial"],
            _VALUES,
        ),
        "merge": (["partial", "sink", "meta"], ["out"], _MERGE),
    }[stage]
    return mx.fast.metal_kernel(
        name="deepseek_v41_online_" + stage,
        input_names=inputs,
        output_names=outputs,
        source=source,
        header=_READ,
    )


def rounded_packed_attention(q, window, pooled, wi, ci, sink, scale):
    """Read packed KV directly with the official 64-key BF16 PV boundary.

    Matrix kernels share decoded tiles across heads for large candidate lists.
    Short lists fit QK and PV in one threadgroup per head. Shapes exceeding
    that capacity use separate dispatches with the same prefix maxima.
    """
    _, length, heads, dim = q.shape
    window_slots, pooled_slots = wi.shape[-1], ci.shape[-1]
    from . import fused_attention

    large = heads >= 8 and dim <= 512 and window_slots + pooled_slots >= 512
    if large and fused_attention.supported(q):
        # Matrix kernels in one dispatch (fused_attention.py).
        return fused_attention.fused_mma_attention(q, window, pooled, wi, ci, sink, scale)
    if window_slots + pooled_slots <= 2048:
        return _fused_attention(q, window, pooled, wi, ci, sink, scale)
    blocks = max(1, (wi.shape[-1] + ci.shape[-1] + 15) // 16)
    meta = mx.array(
        [heads, wi.shape[-1], ci.shape[-1], window.shape[1], pooled.shape[1], blocks],
        mx.int32,
    )
    window = window if window.size else mx.zeros((1,), mx.uint8)
    pooled = pooled if pooled.size else mx.zeros((1,), mx.uint8)
    wi = wi.astype(mx.int32) if wi.size else mx.zeros((1,), mx.int32)
    ci = ci.astype(mx.int32) if ci.size else mx.zeros((1,), mx.int32)
    scale = mx.array([scale])
    merge_meta = mx.array([heads, blocks], mx.int32)
    # Share a single contiguous input between the score and value stages.
    window, pooled = mx.contiguous(window), mx.contiguous(pooled)
    wi, ci = mx.contiguous(wi), mx.contiguous(ci)
    grid = ((heads + 3) // 4 * 128, length, blocks)
    scores, maxima = _kernel("scores")(
        inputs=[q, window, pooled, wi, ci, meta, scale],
        template=[("D", dim)],
        grid=grid,
        threadgroup=(128, 1, 1),
        output_shapes=[(1, length, heads, blocks, 16), (1, length, heads, blocks)],
        output_dtypes=[mx.float32, mx.float32],
    )
    partial = _kernel("values")(
        inputs=[scores, maxima, window, pooled, wi, ci, meta],
        template=[("D", dim)],
        grid=grid,
        threadgroup=(128, 1, 1),
        output_shapes=[(1, length, heads, blocks, dim + 2)],
        output_dtypes=[mx.float32],
    )[0]
    result = _kernel("merge")(
        inputs=[partial, sink, merge_meta],
        template=[("D", dim), ("T", q.dtype), ("CHUNK", 16)],
        grid=((heads * (dim // 32) + 3) // 4 * 128, length, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, length, heads, dim)],
        output_dtypes=[q.dtype],
    )[0]

    return result


def _fused_attention(q, window, pooled, wi, ci, sink, scale):
    _, length, heads, dim = q.shape
    window_slots, pooled_slots = wi.shape[-1], ci.shape[-1]
    count = window_slots + pooled_slots
    chunk = 32 if count <= 1024 else 64
    blocks = max(1, (count + chunk - 1) // chunk)
    meta = mx.array(
        [heads, window_slots, pooled_slots, window.shape[1], pooled.shape[1]], mx.int32
    )
    window = window if window.size else mx.zeros((1,), mx.uint8)
    pooled = pooled if pooled.size else mx.zeros((1,), mx.uint8)
    wi = wi.astype(mx.int32) if wi.size else mx.zeros((1,), mx.int32)
    ci = ci.astype(mx.int32) if ci.size else mx.zeros((1,), mx.int32)
    scale = mx.array([scale])
    merge_meta = mx.array([heads, blocks], mx.int32)
    partial = _kernel("fused")(
        inputs=[q, window, pooled, wi, ci, meta, scale],
        template=[("D", dim), ("NB", blocks), ("CHUNK", chunk)],
        grid=(heads * blocks * 32, length, 1),
        threadgroup=(blocks * 32, 1, 1),
        output_shapes=[(1, length, heads, blocks, dim + 2)],
        output_dtypes=[mx.float32],
    )[0]
    result = _kernel("merge")(
        inputs=[partial, sink, merge_meta],
        template=[("D", dim), ("T", q.dtype), ("CHUNK", chunk)],
        grid=((heads * (dim // 32) + 3) // 4 * 128, length, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]
    return result
