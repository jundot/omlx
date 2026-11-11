# SPDX-License-Identifier: MIT
"""Fused packed sparse attention for DeepSeek V4.1, bitwise identical to the three-kernel
matrix path it replaced (kept as the oracle in tests/test_deepseek_v41_fused_attention.py).

Exactness contract (C0, bitwise). For every (query, head, dim) the arithmetic is the
reference's, in the reference's order:
  * packed KV values come from read_kv's expressions, kept in fp32
  * scores: 8x8 simdgroup MMA over D/4-wide quarters, k ascending in steps of 8, each
    quarter from a zero accumulator; quarters summed left to right, then * scale;
    invalid slots are -inf
  * value block b (64 keys): max is the running prefix max (from -1e30) through the
    block end; p = exp(s - max); PV uses float(bfloat16_t(p)) and the block denominator
    sums unrounded p in key order; PV is 8x8 MMA over the block's eight 8-key tiles in
    ascending order from a zero accumulator
  * merge: blocks ascending, correction = exp(prev_max - max_b), den = den*corr + den_b,
    acc = acc*corr + pv_b; den += exp(sink - max); out = T(acc / den)

Where the time goes instead:
  * each packed row is decoded once per call into fp32 (all cached rows for long query
    blocks; only each query's selected rows when that is smaller, e.g. decode), not once
    per (query, 8 heads, pass) as in the three-kernel reference
  * one threadgroup per (query, 8 heads) with 32 dimension slices; all score chains of a
    64-key block run without intermediate barriers, MMA operands come straight from the
    decoded rows, probabilities are computed in parallel
Measured on M3 Ultra, real prefill inputs (2048 queries, 64 heads, 640 slots): 177 -> 28 ms;
decode (1 query): 0.109 -> 0.041 ms per layer, flat from 2k to 64k cached rows.
"""

from functools import cache

import mlx.core as mx

from .packed_attention import _READ

_ATTENTION = r"""
    constexpr int TPG = HT / 8 * SL * 32;
    constexpr int NSG = TPG / 32;
    constexpr int CHAINS = HT / 8 * 8 * 4;                 // head groups x 8 tiles x 4 quarters
    const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint query = threadgroup_position_in_grid.y;
    const uint head0 = threadgroup_position_in_grid.x * HT;
    const int H = meta[0], W = meta[1], C = meta[2], NW = meta[3], NC = meta[4];
    const int count = W + C, NB = (count + 63) / 64;
    const uint fm = ((lane >> 2) & 4) + ((lane >> 1) & 3);
    const uint fn = (((lane >> 2) & 2) << 1) + ((lane & 1) << 1);
    threadgroup float dots[CHAINS * 64];
    threadgroup float sblock[HT * 64];
    threadgroup float probability[HT * 64], praw[HT * 64];
    threadgroup int rowidx[64];
    threadgroup float running_max[HT], block_max[HT], correction[HT], den_block[HT], den_run[HT];
    if (tid < uint(HT)) { running_max[tid] = -1e30f; den_run[tid] = 0.0f; }
    const uint hg = sg / SL, slice = sg % SL;              // PV role
    const uint head = head0 + hg * 8 + fm;
    simdgroup_matrix<float, 8, 8> acc[D / (8 * SL)], pv[D / (8 * SL)];
    _Pragma("unroll") for (int v = 0; v < D / (8 * SL); ++v) {
        acc[v].thread_elements()[0] = 0.0f; acc[v].thread_elements()[1] = 0.0f;
    }
    for (int b = 0; b < NB; ++b) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 64) {
            const int j = b * 64 + int(tid);
            const bool compressed = j >= W;
            const int row = j >= count ? -1 : compressed ? ci[query * C + j - W] : wi[query * W + j];
            const bool ok = row >= 0 && row < (compressed ? NC : NW);
            rowidx[tid] = ok ? (compressed ? NW + row : row) : -1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // ---- score chains: chain = (head group, tile, quarter) ----
        for (uint chain = sg; chain < uint(CHAINS); chain += NSG) {
            const uint cg = chain / 32, tile = (chain / 4) % 8, quarter = chain % 4;
            const uint chead = head0 + cg * 8 + fm;
            const int r0 = rowidx[tile * 8 + fn], r1 = rowidx[tile * 8 + fn + 1];
            simdgroup_matrix<float, 8, 8> score, a, bm;
            score.thread_elements()[0] = 0.0f; score.thread_elements()[1] = 0.0f;
            for (int k = quarter * (D / 4); k < int(quarter + 1) * (D / 4); k += 8) {
                a.thread_elements()[0] = chead < uint(H) ? float(q[(query * H + chead) * D + k + fn]) : 0.0f;
                a.thread_elements()[1] = chead < uint(H) ? float(q[(query * H + chead) * D + k + fn + 1]) : 0.0f;
                bm.thread_elements()[0] = r0 >= 0 ? float(kv[size_t(r0) * D + k + fm]) : 0.0f;
                bm.thread_elements()[1] = r1 >= 0 ? float(kv[size_t(r1) * D + k + fm]) : 0.0f;
                simdgroup_multiply_accumulate(score, a, bm, score);
            }
            dots[chain * 64 + fm * 8 + fn] = score.thread_elements()[0];
            dots[chain * 64 + fm * 8 + fn + 1] = score.thread_elements()[1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < uint(HT * 64); i += TPG) {
            const uint hh = i / 64, key = i % 64, tile = key / 8, r = key % 8;
            const uint base = (((hh / 8) * 8 + tile) * 4) * 64 + (hh % 8) * 8 + r;
            float sum = dots[base] + dots[base + 64] + dots[base + 128] + dots[base + 192];
            sblock[hh * 64 + key] = (rowidx[key] >= 0 && head0 + hh < uint(H)) ? sum * scalep[0] : -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < uint(HT)) {
            float m = running_max[tid];
            for (int s = 0; s < 64; ++s) m = max(m, sblock[tid * 64 + s]);
            block_max[tid] = m;
            correction[tid] = exp(running_max[tid] - m);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < uint(HT * 64); i += TPG) {
            const uint hh = i / 64, key = i % 64;
            const float p = head0 + hh < uint(H) && rowidx[key] >= 0 ? exp(sblock[hh * 64 + key] - block_max[hh]) : 0.0f;
            praw[hh * 64 + key] = p;
            probability[hh * 64 + key] = float(bfloat16_t(p));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < uint(HT)) {
            float total = 0.0f;
            for (uint key = 0; key < 64; ++key) total += praw[tid * 64 + key];
            den_block[tid] = total;
        }
        _Pragma("unroll") for (int v = 0; v < D / (8 * SL); ++v) {
            pv[v].thread_elements()[0] = 0.0f; pv[v].thread_elements()[1] = 0.0f;
        }
        simdgroup_matrix<float, 8, 8> a, bm;
        for (uint tile = 0; tile < 8; ++tile) {
            a.thread_elements()[0] = probability[(hg * 8 + fm) * 64 + tile * 8 + fn];
            a.thread_elements()[1] = probability[(hg * 8 + fm) * 64 + tile * 8 + fn + 1];
            const int r = rowidx[tile * 8 + fm];
            _Pragma("unroll") for (int v = 0; v < D / (8 * SL); ++v) {
                const uint d = slice * (D / SL) + v * 8 + fn;
                bm.thread_elements()[0] = r >= 0 ? float(kv[size_t(r) * D + d]) : 0.0f;
                bm.thread_elements()[1] = r >= 0 ? float(kv[size_t(r) * D + d + 1]) : 0.0f;
                simdgroup_multiply_accumulate(pv[v], a, bm, pv[v]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float corr = correction[hg * 8 + fm];
        _Pragma("unroll") for (int v = 0; v < D / (8 * SL); ++v) {
            acc[v].thread_elements()[0] = acc[v].thread_elements()[0] * corr + pv[v].thread_elements()[0];
            acc[v].thread_elements()[1] = acc[v].thread_elements()[1] * corr + pv[v].thread_elements()[1];
        }
        if (tid < uint(HT)) {
            den_run[tid] = den_run[tid] * correction[tid] + den_block[tid];
            running_max[tid] = block_max[tid];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (head < uint(H)) {
        const float denominator = den_run[hg * 8 + fm] + exp(float(sink[head]) - running_max[hg * 8 + fm]);
        _Pragma("unroll") for (int v = 0; v < D / (8 * SL); ++v) {
            const uint d = slice * (D / SL) + v * 8 + fn;
            out[(query * H + head) * D + d] = T(acc[v].thread_elements()[0] / denominator);
            out[(query * H + head) * D + d + 1] = T(acc[v].thread_elements()[1] / denominator);
        }
    }
"""

_ATTENTION_GATHERED = _ATTENTION.replace(
    "rowidx[tid] = ok ? (compressed ? NW + row : row) : -1;",
    "rowidx[tid] = ok ? int(query) * count + j : -1;")

_DECODE_ROWS = r"""
    const uint e = thread_position_in_grid.x;               // one thread per (row, 8 dims)
    const int NW = meta[0], NC = meta[1];
    const uint row = e / (D / 8), d0 = (e % (D / 8)) * 8;
    if (row >= uint(NW + NC)) return;
    const bool compressed = row >= uint(NW);
    const int r = compressed ? int(row) - NW : int(row);
    for (uint d = d0; d < d0 + 8; ++d)
        out[size_t(row) * D + d] = read_kv(window, pooled, r, d, D, compressed, 0.0f, 0, false);
"""

_DECODE_GATHER = r"""
    const uint e = thread_position_in_grid.x;               // (query, slot, 8 dims)
    const int W = meta[0], C = meta[1], NW = meta[2], NC = meta[3], L = meta[4];
    const int count = W + C;
    const uint per_query = uint(count) * (D / 8);
    const uint query = e / per_query, rem = e % per_query, slot = rem / (D / 8), d0 = (rem % (D / 8)) * 8;
    if (query >= uint(L)) return;
    const int j = int(slot);
    const bool compressed = j >= W;
    const int row = compressed ? ci[query * C + j - W] : wi[query * W + j];
    const bool ok = row >= 0 && row < (compressed ? NC : NW);
    for (uint d = d0; d < d0 + 8; ++d)
        out[(size_t(query) * count + slot) * D + d] = ok ? read_kv(window, pooled, row, d, D, compressed, 0.0f, 0, false) : 0.0f;
"""

@cache
def _kernel(name, source, inputs):
    header = _READ + ("\n#include <metal_simdgroup_matrix>\n" if "simdgroup" in source else "")
    return mx.fast.metal_kernel(
        name="deepseek_v41_" + name, input_names=list(inputs), output_names=["out"],
        source=source, header=header)


def supported(q):
    _, _, heads, dim = q.shape
    return q.dtype == mx.bfloat16 and dim % 32 == 0 and dim <= 512 and heads % 8 == 0


def fused_mma_attention(q, window, pooled, wi, ci, sink, scale):
    _, length, heads, dim = q.shape
    nw, nc, w, c = window.shape[1], pooled.shape[1], wi.shape[-1], ci.shape[-1]
    wi = wi.astype(mx.int32) if wi.size else mx.zeros((1,), mx.int32)
    ci = ci.astype(mx.int32) if ci.size else mx.zeros((1,), mx.int32)
    window = window if window.size else mx.zeros((1,), mx.uint8)
    pooled = pooled if pooled.size else mx.zeros((1,), mx.uint8)
    if length * (w + c) < nw + nc:
        kv = _kernel("decode_gather_f32", _DECODE_GATHER, ("window", "pooled", "wi", "ci", "meta"))(
            inputs=[window, pooled, wi, ci, mx.array([w, c, nw, nc, length], mx.int32)],
            template=[("D", dim)], grid=(max(1, length * (w + c) * (dim // 8)), 1, 1),
            threadgroup=(256, 1, 1), output_shapes=[(max(1, length * (w + c)), dim)],
            output_dtypes=[mx.float32])[0]
        attention = _kernel("fused_attention_gathered", _ATTENTION_GATHERED,
                            ("q", "kv", "wi", "ci", "sink", "meta", "scalep"))
    else:
        kv = _kernel("decode_rows_f32", _DECODE_ROWS, ("window", "pooled", "meta"))(
            inputs=[window, pooled, mx.array([nw, nc], mx.int32)],
            template=[("D", dim)], grid=(max(1, (nw + nc) * (dim // 8)), 1, 1),
            threadgroup=(256, 1, 1), output_shapes=[(max(1, nw + nc), dim)],
            output_dtypes=[mx.float32])[0]
        attention = _kernel("fused_attention", _ATTENTION,
                            ("q", "kv", "wi", "ci", "sink", "meta", "scalep"))
    # 32 dimension slices of D / 32 (D = 512: two 8-wide MMA tiles per simdgroup); smaller
    # heads use fewer slices so every slice still owns a whole number of 8-wide tiles.
    heads_per_tg, slices = 8, 32
    while (dim // 8) % slices:
        slices //= 2
    return attention(
        inputs=[q, kv, wi, ci, sink, mx.array([heads, w, c, nw, nc], mx.int32), mx.array([scale])],
        template=[("D", dim), ("HT", heads_per_tg), ("SL", slices), ("T", q.dtype)],
        grid=(heads // heads_per_tg * slices * 32, length, 1),
        threadgroup=(heads_per_tg // 8 * slices * 32, 1, 1),
        output_shapes=[q.shape], output_dtypes=[q.dtype])[0]
