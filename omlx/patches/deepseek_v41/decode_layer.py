# SPDX-License-Identifier: MIT
"""Fused single-token decode kernels for DeepSeek V4.1 (hyper-connections + attention).

Each kernel reproduces the DeepSeek-V4.1-Flash-1017 reference (inference/model.py,
kernel.py) at every rounding point it states: fp32 tensors, bf16 tensors, act_quant
(FP8 E4M3, 32-element groups, ue8m0 scales), the packed window row, RoPE in fp32 on bf16
inputs. Free, i.e. unspecified by the reference (tensor-core / reduce_sum / all_reduce
orders and CUDA transcendentals): the fp32 accumulation order of every dot product and
sum, and exp / rsqrt / sigmoid / sqrt. For a K-term fp32 sum the reordering error is at
most gamma_K * sum|terms| (gamma_K = K 2^-24 / (1 - K 2^-24)) before the next rounding
point; products of a bf16 activation with an fp8/fp4 weight times a power-of-two scale
are exact in fp32.

Why this differs from upstream oMLX's decode (the unfused language.Block path): identical
rounding points, but a decoder layer runs in ~15 dispatches instead of ~150. On the M3 Ultra each dependent
dispatch costs a few microseconds of pipeline drain, which was ~20 ms per token.
"""

from functools import cache

import mlx.core as mx

from .decode_moe import _HEADER as _MOE_HEADER
from .decode_moe import format_macros, format_tag

_HEADER = _MOE_HEADER + r"""
// E4M3FN code of a value already on the E4M3 grid (|v| <= 448).
inline uchar e4m3_code(float q) {
    const uint b = as_type<uint>(q);
    const uchar s = uchar((b >> 31) << 7);
    const uint e = (b >> 23) & 255u;
    if (e <= 120u) return uchar(s | uchar(int(rint(abs(q) * 512.0f))));
    return uchar(s | uchar(((e - 120u) << 3) | ((b >> 20) & 7u)));
}
// Reference Block.hc_post: y = post * x + sum_j(comb[j][i] * res_j) as separate torch ops,
// so the product is rounded before the add (no FMA); the sum over j is order free.
template <typename P, typename C, typename X, typename R>
inline bfloat16_t hc_post_value(P post, C comb, X sub, R res, uint i, uint d, uint hc, uint dim) {
    #pragma clang fp contract(off)
    float s = 0.0f;
    for (uint j = 0; j < hc; ++j) s += comb[j * hc + i] * float(res[j * dim + d]);
    const float px = post[i] * float(sub[d]);
    return bfloat16_t(px + s);
}
inline float rsqrt_mean_eps(float sumsq, float n, float eps) {
    return 1.0f / metal::precise::sqrt(sumsq / n + eps);
}
"""

# K1: hyper-connection mixing GEMV partials and hc_pre. x = flatten(h) as fp32 (HC*DIM
# values); a threadgroup of 64 threads takes dims [64 tg, 64 tg + 64) of all HC streams
# (4 consecutive dims per thread) and writes, column-major over the NPART threadgroups:
# 24 dot partials, the sum of squares of x, and the sum of squares of hc_pre(h, pre_in),
# whose bf16 values go to xpre. With POST, h is first produced from the previous sublayer:
#   h'_i = bf16(post_i * x_sub + sum_j comb[j][i] * res_j)   (reference Block.hc_post)
_MIX_PARTIAL = r"""
    const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup, tg = threadgroup_position_in_grid.x;
    static_assert(HC == 4, "64 threads = 4 streams x 16 threads");
    const uint i = tid / 16, dl = (tid % 16) * 4, d = tg * 64 + dl, col = i * DIM + d;
    threadgroup float hv[HC][64];
    threadgroup float red[2][MIX + 2];
    float4 v;
    if (POST) {
        // hc_post_value for 4 dims: the product is rounded before the add (no FMA)
        #pragma clang fp contract(off)
        float4 s = 0.0f;
        for (uint j = 0; j < uint(HC); ++j) s += comb[j * HC + i] * float4(*(const device packed_bfloat4*)(res + j * DIM + d));
        const float4 px = post[i] * float4(*(const device packed_bfloat4*)(sub + d));
        const packed_bfloat4 hb = packed_bfloat4(px + s);
        *(device packed_bfloat4*)(hout + col) = hb;
        v = float4(hb);
    } else {
        v = float4(*(const device packed_bfloat4*)(res + col));
    }
    for (uint k = 0; k < 4; ++k) hv[i][dl + k] = v[k];
    float part[MIX];
    for (uint r = 0; r < uint(MIX); ++r) {
        const float4 w = *(const device float4*)(fn + r * HC * DIM + col);
        part[r] = v.x * w.x + v.y * w.y + v.z * w.z + v.w * w.w;
    }
    float sq = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    for (uint r = 0; r < uint(MIX); ++r) part[r] = simd_sum(part[r]);
    sq = simd_sum(sq);
    if (lane == 0) { for (uint r = 0; r < uint(MIX); ++r) red[sg][r] = part[r]; red[sg][MIX] = sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // hc_pre(h, pre_in) -> bf16 for dim 64 tg + tid
    float y = 0.0f;
    for (uint j = 0; j < uint(HC); ++j) y += pre_in[j] * hv[j][tid];
    const float yb = float(bfloat16_t(y));
    xpre[tg * 64 + tid] = bfloat16_t(yb);
    const float s2 = simd_sum(yb * yb);
    if (lane == 0) red[sg][MIX + 1] = s2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < uint(MIX + 2)) partial[tid * NPART + tg] = red[0][tid] + red[1][tid];
"""

# K2: threadgroup 0 finalizes the mixes -> pre/post/comb (hc_split_sinkhorn); threadgroup
# 1 + t applies RMSNorm and act_quant to dims [TPG t, TPG t + TPG) of hc_pre(h, pre_in).
_MIX_FINAL = r"""
    const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup, tg = threadgroup_position_in_grid.x;
    threadgroup float mixes[MIX + 1];
    threadgroup float coef[MIX];
    threadgroup float rs[1];
    if (tg == 0) {
        // Load the constants first so their latency overlaps the partial sums.
        const float e0 = eps[0], he = hc_eps[0], sc0 = scale[0], sc1 = scale[1], sc2 = scale[2];
        const uint l = lane & 15u, j = l >> 2, k = l & 3u;
        const float b0 = base[lane & 3u], b1 = base[HC + (lane & 3u)], b2 = base[2 * HC + l];
        // partial dot products -> mixes: simdgroup r sums column r over the NPART threadgroup
        // partials (fp32, order free: F.linear's accumulation in the reference)
        if (sg <= uint(MIX)) {
            float t = 0.0f;
            for (uint p = lane; p < uint(NPART); p += 32) t += partial[sg * NPART + p];
            t = simd_sum(t);
            if (lane == 0) mixes[sg] = t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            // mixes = F.linear(x, hc_fn) * rsqrt(mean(x^2) + norm_eps)
            const float r = rsqrt_mean_eps(mixes[MIX], float(HC * DIM), e0);
            // kernel.py hc_split_sinkhorn with one lane per comb element (lane & 15 = j * HC + k;
            // lanes 16..31 mirror 0..15). The 4-term row / column sums are T.reduce_sum in the
            // reference (order free); here they are xor-butterfly sums, |error| <= gamma_4 * sum.
            // Every other operation is the reference's, in its order.
            static_assert(HC == 4, "lane layout assumes hc_mult == 4");
            if (lane < uint(HC)) {
                const float z = mixes[lane] * r * sc0 + b0;
                coef[lane] = 1.0f / (1.0f + metal::precise::exp(-z)) + he;
                const float z2 = mixes[lane + HC] * r * sc1 + b1;
                coef[HC + lane] = 2.0f * (1.0f / (1.0f + metal::precise::exp(-z2)));
            }
            float c = mixes[2 * HC + l] * r * sc2 + b2;
            float mrow = max(c, simd_shuffle_xor(c, 1u));
            mrow = max(mrow, simd_shuffle_xor(mrow, 2u));
            c = metal::precise::exp(c - mrow);
            float srow = c + simd_shuffle_xor(c, 1u);
            srow = srow + simd_shuffle_xor(srow, 2u);
            c = c / srow + he;
            float scol = c + simd_shuffle_xor(c, 4u);
            scol = scol + simd_shuffle_xor(scol, 8u);
            c = c / (scol + he);
            for (int it = 0; it < ITERS - 1; ++it) {
                srow = c + simd_shuffle_xor(c, 1u);
                srow = srow + simd_shuffle_xor(srow, 2u);
                c = c / (srow + he);
                scol = c + simd_shuffle_xor(c, 4u);
                scol = scol + simd_shuffle_xor(scol, 8u);
                c = c / (scol + he);
            }
            if (lane < 16u) coef[2 * HC + j * HC + k] = c;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < uint(MIX)) mix_out[tid] = coef[tid];
        return;
    }
    // RMSNorm(norm_w) of hc_pre -> bf16, then act_quant
    const uint d = (tg - 1) * TPG + tid;
    const float e0 = eps[0], w = float(norm_w[d]), yv = float(y[d]);
    if (sg == 0) {
        float t = 0.0f;
        for (uint p = lane; p < uint(NPART); p += 32) t += partial[(MIX + 1) * NPART + p];
        t = simd_sum(t);
        if (lane == 0) rs[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float r = rsqrt_mean_eps(rs[0], float(DIM), e0);
    const float xb = float(bfloat16_t(w * (yv * r)));
    xn[d] = bfloat16_t(xb);
    // act_quant groups of 32 = one simdgroup
    const float s = act_scale(max(simd_max(abs(xb)), 1e-4f));
    xq[d] = act_quant_value(xb, s);
"""

# K3/K9: MXFP8 GEMV for one or two matrices sharing an input, one simdgroup per row.
_GEMV_FP8 = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint row0 = (threadgroup_position_in_grid.x * 2 + simdgroup_index_in_threadgroup) * 4;
    if (row0 >= uint(NA + NB)) return;
    float acc[4];
    if (row0 >= uint(NA)) {
        B_ROWS(K, 4, wb, sb, bb, row0 - NA, x, lane, acc);
        if (lane < 4) yb[row0 - NA + lane] = bfloat16_t(acc[lane]);
    } else {
        A_ROWS(K, 4, wa, sa, ba, row0, x, lane, acc);
        if (lane < 4) ya[row0 + lane] = bfloat16_t(acc[lane]);
    }
"""

# K4: q_norm -> bf16 qr, act_quant(qr); kv_norm -> bf16, RoPE tail (fp32, bf16), act_quant,
# packed window row (fp8 codes + exponent bytes). One threadgroup.
_ATTN_MID = r"""
    const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    threadgroup float red[32];
    threadgroup float kvn[HD];
    // ---- q_norm ----
    float sq = 0.0f;
    for (uint d = tid; d < uint(QR); d += 1024) { const float v = float(qa[d]); sq += v * v; }
    sq = simd_sum(sq);
    if (lane == 0) red[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float t = 0.0f; for (uint s = 0; s < 32; ++s) t += red[s]; red[0] = t; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rq = rsqrt_mean_eps(red[0], float(QR), eps[0]);
    for (uint d = tid; d < uint(QR); d += 1024) {
        const float qb = float(bfloat16_t(float(qnw[d]) * (float(qa[d]) * rq)));
        qr[d] = bfloat16_t(qb);
        const float s = act_scale(max(simd_max(abs(qb)), 1e-4f));
        qrq[d] = act_quant_value(qb, s);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // ---- kv_norm ----
    float sk = 0.0f;
    for (uint d = tid; d < uint(HD); d += 1024) { const float v = float(kv[d]); sk += v * v; }
    sk = simd_sum(sk);
    if (lane == 0) red[sg] = sk;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float t = 0.0f; for (uint s = 0; s < 32; ++s) t += red[s]; red[0] = t; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rk = rsqrt_mean_eps(red[0], float(HD), eps[0]);
    for (uint d = tid; d < uint(HD); d += 1024) kvn[d] = float(bfloat16_t(float(kvnw[d]) * (float(kv[d]) * rk)));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // ---- RoPE on the last RD dims (adjacent pairs as complex numbers), bf16 ----
    if (tid < uint(HD)) {
        float v = kvn[tid];
        if (tid >= uint(HD - RD)) {
            const uint p = (tid - (HD - RD)) / 2;
            const float a = kvn[tid & ~1u], b = kvn[tid | 1u];
            const float cs = cosv[p], sn = sinv[p];
            v = float(bfloat16_t((tid & 1u) ? a * sn + b * cs : a * cs - b * sn));
        }
        // ---- act_quant (32-groups) + packed window row ----
        const float s = act_scale(max(simd_max(abs(v)), 1e-4f));
        const float q = round_e4m3(clamp(v / s, -448.0f, 448.0f));
        packed[tid] = e4m3_code(q);
        if (lane == 0) packed[HD + tid / 32] = uchar(int((as_type<uint>(s) >> 23) & 255u));
    }
"""

# K5: wq_b MXFP8 GEMV with the RoPE epilogue: a simdgroup computes rows (2p, 2p+1); the
# GEMM output is bf16, then the tail pairs of each head are rotated in fp32 and stored bf16.
_WQB_ROPE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint row0 = (threadgroup_position_in_grid.x * 2 + simdgroup_index_in_threadgroup) * 4;
    if (row0 >= uint(N)) return;
    float acc[4];
    W_ROWS(K, 4, w, sc, bi, row0, x, lane, acc);
    if (lane < 2) {
        // rows (row0 + 2*lane, +1) form one RoPE pair; GEMM output is bf16 first
        float a = float(bfloat16_t(acc[2 * lane])), b = float(bfloat16_t(acc[2 * lane + 1]));
        const uint d = (row0 + 2 * lane) % HD;
        if (d >= uint(HD - RD)) {
            const uint p = (d - (HD - RD)) / 2;
            const float cs = cosv[p], sn = sinv[p];
            const float ra = a * cs - b * sn, rb = a * sn + b * cs;
            a = ra; b = rb;
        }
        y[row0 + 2 * lane] = bfloat16_t(a);
        y[row0 + 2 * lane + 1] = bfloat16_t(b);
    }
"""

# K8: wo_a as a grouped MXFP8 GEMV (the bf16 checkpoint tensor is exactly FP8 * 2^k per
# 32-group, so the products are the same exact values), bf16 einsum output, then act_quant
# over each 32-row block (= one threadgroup) as the wo_b input.
_WOA = r"""
    threadgroup float yb[32];
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const uint row0 = threadgroup_position_in_grid.x * 32 + sg * 4;    // 8 simdgroups x 4 rows
    const uint g = row0 / R;
    float acc[4];
    fp8_rows<K, 4>(w, sc, row0, x + g * K, lane, acc);
    if (lane < 4) yb[sg * 4 + lane] = float(bfloat16_t(acc[lane]));    // bf16 einsum output
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float v = yb[lane];
        const float s = act_scale(max(simd_max(abs(v)), 1e-4f));
        yq[threadgroup_position_in_grid.x * 32 + lane] = act_quant_value(v, s);
    }
"""


_KV_RUN = r"""
// N consecutive decoded values of one packed KV row from dim d0 (d0 % N == 0; N = 2, 4 or a
// multiple of 8): the read_kv values (exact products), read with vector loads and decoded through
// fp16 bit patterns (fp8_decode4 / fp4_decode8 give the code value times 2^-8 / 2^-14).
template <int N, typename W, typename P>
inline void read_kv_run(W window, P pooled, int row, int d0, int dim, bool compressed, thread float* out) {
    if (N < 4) {
        for (int j = 0; j < N; ++j) out[j] = read_kv(window, pooled, row, d0 + j, dim, compressed, 0.0f, 0, false);
    } else if (compressed) {
        const size_t base = size_t(row) * (dim / 2 + dim / 16);
        const device uint* c = (const device uint*)(pooled + base + d0 / 2);
        for (int g = 0; g < N / 8; ++g) {
            const float sc = rounded_fp8(pooled[base + dim / 2 + (d0 + 8 * g) / 16]) * 0x1p14f;
            float v[8];
            fp4_decode8(c[g], v);
            for (int j = 0; j < 8; ++j) out[8 * g + j] = v[j] * sc;
        }
        if (N == 4) {
            const float sc = rounded_fp8(pooled[base + dim / 2 + d0 / 16]) * 0x1p14f;
            float v[8];
            fp4_decode8(uint(*(const device ushort*)(pooled + base + d0 / 2)), v);
            for (int j = 0; j < 4; ++j) out[j] = v[j] * sc;
        }
    } else {
        const size_t base = size_t(row) * (dim + dim / 32);
        const float sc = as_type<float>(uint(window[base + dim + d0 / 32] + 8) << 23);
        const device uint* c = (const device uint*)(window + base + d0);
        for (int g = 0; g < N / 4; ++g) {
            float v[4];
            fp8_decode4(c[g], v);
            for (int j = 0; j < 4; ++j) out[4 * g + j] = v[j] * sc;
        }
    }
}
"""


# Split single-query sparse attention (kernel.py sparse_attn). The reference processes the
# selected keys in 64-key blocks per head: s = (q . k) * scale (fp32 accumulator, -inf for
# invalid slots); M_b = max(M_{b-1}, rowmax(s_b)) from M_{-1} = -1e30; p = exp(s - M_b);
# den = den * exp(M_{b-1} - M_b) + sum(p); acc = acc * exp(M_{b-1} - M_b) + bf16(p) @ V_b;
# out = bf16(acc / (den + exp(sink - M_last))). Scores do not depend on the running state and
# M_b is a prefix max of per-block maxima (exact), so the work splits into
#   attn_scores: s and rowmax(s_b) for every (8-head group, block) in parallel,
#   attn_pv:     M_b, p, bf16(p) @ V_b and sum(p) for every (8-head group, block) in parallel,
#   o_merge:     the reference's sequential merge over blocks (in the o epilogue),
# with every rounding point above unchanged (same values, same op sequence per block). Only
# the fp32 accumulation order inside the q.k dot, the PV GEMM and sum(p) differs from
# upstream oMLX's fused kernel (one threadgroup per 8 heads walking the blocks serially, ~8
# of 80 GPU cores busy, ~150 us/layer); those orders are free in the reference (tensor-core
# GEMM, T.reduce_sum), error <= gamma_K * sum|terms| per sum (K = 512 / 64 / 64).
# KV rows are decoded on the fly to the read_kv values (fp8 * 2^e window rows, fp4 * fp8 pooled
# rows, exact in fp32), each thread reading a run of consecutive dims of a row (read_kv_run).
_ATTN_COMMON = r"""
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const uint hg = threadgroup_position_in_grid.x, b = threadgroup_position_in_grid.y;
    const int W = meta[0], C = meta[1], NW = meta[2], NC = meta[3], NB = meta[4];
    const int count = W + C;
    const uint fm = ((lane >> 2) & 4) + ((lane >> 1) & 3);
    const uint fn = (((lane >> 2) & 2) << 1) + ((lane & 1) << 1);
    threadgroup int rowidx[64];
    threadgroup uchar iscomp[64];
    if (sg == 0 || sg == 1) {
        const uint t = sg * 32 + lane;
        const int j = int(b) * 64 + int(t);
        const bool compressed = j >= W;
        const int row = j >= count ? -1 : compressed ? ci[j - W] : wi[j];
        const bool ok = row >= 0 && row < (compressed ? NC : NW);
        rowidx[t] = ok ? row : -1;
        iscomp[t] = compressed ? 1 : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""
_ATTN_SCORES = _ATTN_COMMON + r"""
    // S^T = K Q^T on 8x8 tiles. 32 simdgroups: key tile = sg % 8, D quarter = sg / 8. Thread (fm, fn)
    // holds key fm of the tile and the CH = D / 16 consecutive dims of chunk fn / 2 of the quarter
    // (one vector read of the row); MMA step s takes dims 2 s, 2 s + 1 of every chunk.
    constexpr int CH = D / 16;
    threadgroup float dots[4 * 8 * 64];
    threadgroup float sblock[8 * 64];
    threadgroup bfloat16_t qs[8 * D];
    const uint tid = thread_index_in_threadgroup;
    for (uint i = tid * 4; i < 8 * D; i += 4096) *(threadgroup packed_bfloat4*)(qs + i) = *(const device packed_bfloat4*)(q + hg * 8 * D + i);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint tile = sg % 8, quarter = sg / 8;
    const int r = rowidx[tile * 8 + fm];
    const bool cmp = iscomp[tile * 8 + fm];
    float kv[CH];
    if (r >= 0) read_kv_run<CH>(window, pooled, r, int(quarter * (D / 4) + (fn / 2) * CH), D, cmp, kv);
    else for (int j = 0; j < CH; ++j) kv[j] = 0.0f;
    simdgroup_matrix<float, 8, 8> score, a, bm;
    score.thread_elements()[0] = 0.0f; score.thread_elements()[1] = 0.0f;
    const uint qd = quarter * (D / 4) + (fm / 2) * CH + (fm & 1);
    for (int s = 0; s < CH / 2; ++s) {
        a.thread_elements()[0] = kv[2 * s];
        a.thread_elements()[1] = kv[2 * s + 1];
        bm.thread_elements()[0] = float(qs[fn * D + qd + 2 * s]);
        bm.thread_elements()[1] = float(qs[(fn + 1) * D + qd + 2 * s]);
        simdgroup_multiply_accumulate(score, a, bm, score);
    }
    dots[(quarter * 8 + fn) * 64 + tile * 8 + fm] = score.thread_elements()[0];
    dots[(quarter * 8 + fn + 1) * 64 + tile * 8 + fm] = score.thread_elements()[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 512) {   // (head, key): q.k = sum of the 4 quarter partials; * scale; -inf if invalid
        const uint hh = tid / 64, key = tid % 64;
        const float dot = dots[hh * 64 + key] + dots[(8 + hh) * 64 + key] + dots[(16 + hh) * 64 + key] + dots[(24 + hh) * 64 + key];
        const float sv = rowidx[key] >= 0 ? dot * scalep[0] : -INFINITY;
        S[size_t(hg * 8 + hh) * (NB * 64) + b * 64 + key] = sv;
        sblock[hh * 64 + key] = sv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg < 8) {   // rowmax over the block's 64 keys for head hg*8+sg
        const float m = simd_max(max(sblock[sg * 64 + lane], sblock[sg * 64 + 32 + lane]));
        if (lane == 0) BMAX[(hg * 8 + sg) * NB + b] = m;
    }
"""
_ATTN_PV = _ATTN_COMMON + r"""
    threadgroup float mb[8];
    threadgroup float prob[8 * 64];
    if (sg == 0 && lane < 8) {   // M_b: prefix max of block maxima from -1e30 (exact)
        float m = -1e30f;
        for (uint j = 0; j <= b; ++j) m = max(m, BMAX[(hg * 8 + lane) * NB + j]);
        mb[lane] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg < 8) {   // head hg*8+sg: p = exp(s - M_b); den_b = sum(p) unrounded; bf16(p) for PV
        const uint head = hg * 8 + sg;
        float psum = 0.0f;
        for (uint t = lane; t < 64; t += 32) {
            const float p = rowidx[t] >= 0 ? exp(S[size_t(head) * (NB * 64) + b * 64 + t] - mb[sg]) : 0.0f;
            psum += p;
            prob[sg * 64 + t] = float(bfloat16_t(p));
        }
        psum = simd_sum(psum);
        if (lane == 0) DEN[head * NB + b] = psum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // PV: simdgroup sg takes dims [sg D/32, (sg + 1) D/32), K = 64 keys (8 tiles of 8). In output
    // tile v, MMA column n is dim sg D/32 + (n / 2) 2 TILES + 2 v + (n & 1), so a thread's
    // 2 TILES values of a key row are consecutive (one read per tile).
    constexpr int TILES = D / 256;
    simdgroup_matrix<float, 8, 8> pv[TILES], a, bm;
    for (int v = 0; v < TILES; ++v) { pv[v].thread_elements()[0] = 0.0f; pv[v].thread_elements()[1] = 0.0f; }
    const int d0 = int(sg) * (D / 32) + int(fn / 2) * 2 * TILES;
    for (uint tile = 0; tile < 8; ++tile) {
        a.thread_elements()[0] = prob[fm * 64 + tile * 8 + fn];
        a.thread_elements()[1] = prob[fm * 64 + tile * 8 + fn + 1];
        const int r = rowidx[tile * 8 + fm];
        float vv[2 * TILES];
        if (r >= 0) read_kv_run<2 * TILES>(window, pooled, r, d0, D, iscomp[tile * 8 + fm], vv);
        else for (int j = 0; j < 2 * TILES; ++j) vv[j] = 0.0f;
        for (int v = 0; v < TILES; ++v) {
            bm.thread_elements()[0] = vv[2 * v];
            bm.thread_elements()[1] = vv[2 * v + 1];
            simdgroup_multiply_accumulate(pv[v], a, bm, pv[v]);
        }
    }
    for (int v = 0; v < TILES; ++v) {
        const size_t o = (size_t(hg * 8 + fm) * NB + b) * D + d0 + 2 * v;
        PV[o] = pv[v].thread_elements()[0];
        PV[o + 1] = pv[v].thread_elements()[1];
    }
"""
# o epilogue with the attention merge: one simdgroup per 32 output dims of one head.
_O_MERGE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint grp = threadgroup_position_in_grid.x * (TPG / 32) + simdgroup_index_in_threadgroup;
    if (grp * 32 >= uint(NH * HD)) return;
    const int NB = meta[4];
    const uint idx = grp * 32 + lane, head = idx / HD, d = idx % HD;
    float m = -1e30f, den = 0.0f, acc = 0.0f;
    for (int b = 0; b < NB; ++b) {
        const float mb = max(m, BMAX[head * NB + b]);
        const float corr = exp(m - mb);
        den = den * corr + DEN[head * NB + b];
        acc = acc * corr + PV[(size_t(head) * NB + b) * HD + d];
        m = mb;
    }
    den += exp(float(sink[head]) - m);
    float v = float(bfloat16_t(acc / den));                // attention output o (bf16)
    const float partner = simd_shuffle_xor(v, 1u);
    if (d >= uint(HD - RD)) {
        const uint p = (d - (HD - RD)) / 2;
        const float cs = cosv[p], sn = -sinv[p];          // conj(freqs_cis)
        const float a = (lane & 1u) ? partner : v, b = (lane & 1u) ? v : partner;
        v = float(bfloat16_t((lane & 1u) ? a * sn + b * cs : a * cs - b * sn));
    }
    const float s = act_scale(max(simd_max(abs(v)), 1e-4f));
    oq[idx] = act_quant_value(v, s);
"""

# ---- batched decode (B requests, one token each) ------------------------------------------
# Shared-weight GEMVs: each simdgroup decodes its 4 weight rows once and applies them to all
# NB activation vectors (fp8_rows_nb), so a step reads the weights once instead of B times.
_GEMV_FP8_NB = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint row0 = (threadgroup_position_in_grid.x * 2 + simdgroup_index_in_threadgroup) * 4;
    if (row0 >= uint(NA + NB_ROWS)) return;
    float acc[4 * NB];
    if (row0 >= uint(NA)) {
        B_ROWS_NB(K, 4, NB, wb, sb, bb, row0 - NA, x, uint(K), lane, acc);
        if (lane < 4) for (int b = 0; b < NB; ++b) yb[b * NB_ROWS + row0 - NA + lane] = bfloat16_t(acc[lane * NB + b]);
    } else {
        A_ROWS_NB(K, 4, NB, wa, sa, ba, row0, x, uint(K), lane, acc);
        if (lane < 4) for (int b = 0; b < NB; ++b) ya[b * NA + row0 + lane] = bfloat16_t(acc[lane * NB + b]);
    }
"""
_WQB_ROPE_NB = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint row0 = (threadgroup_position_in_grid.x * 2 + simdgroup_index_in_threadgroup) * 4;
    if (row0 >= uint(N)) return;
    float acc[4 * NB];
    W_ROWS_NB(K, 4, NB, w, sc, bi, row0, x, uint(K), lane, acc);
    if (lane < 2 * NB) {
        const uint b = lane / 2, pr = lane % 2;      // request b, RoPE pair (row0 + 2 pr, +1)
        float a = float(bfloat16_t(acc[(2 * pr) * NB + b])), c = float(bfloat16_t(acc[(2 * pr + 1) * NB + b]));
        const uint d = (row0 + 2 * pr) % HD;
        if (d >= uint(HD - RD)) {
            const uint p = (d - (HD - RD)) / 2;
            const float cs = cosv[b * (RD / 2) + p], sn = sinv[b * (RD / 2) + p];
            const float ra = a * cs - c * sn, rc = a * sn + c * cs;
            a = ra; c = rc;
        }
        y[b * N + row0 + 2 * pr] = bfloat16_t(a);
        y[b * N + row0 + 2 * pr + 1] = bfloat16_t(c);
    }
"""
_WOA_NB = r"""
    threadgroup float yb[NB][32];
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const uint row0 = threadgroup_position_in_grid.x * 32 + sg * 4;    // 8 simdgroups x 4 rows
    const uint g = row0 / R;
    float acc[4 * NB];
    fp8_rows_nb<K, 4, NB>(w, sc, row0, x + g * K, uint(G * K), lane, acc);
    if (lane < 4) for (int b = 0; b < NB; ++b) yb[b][sg * 4 + lane] = float(bfloat16_t(acc[lane * NB + b]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg < uint(NB)) {
        const float v = yb[sg][lane];
        const float s = act_scale(max(simd_max(abs(v)), 1e-4f));
        yq[sg * (G * R) + threadgroup_position_in_grid.x * 32 + lane] = act_quant_value(v, s);
    }
"""

# Batched projections on simdgroup matrices (fp8_mma8; used from decode_moe.MMA_MIN_BATCH
# vectors). Epilogues are the fp8_rows_nb kernels' own.
_GEMV_FP8_MMA = r"""
    threadgroup float red[KS * 64], tile[64];
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup, tid = thread_index_in_threadgroup;
    const uint r0 = threadgroup_position_in_grid.x * 8;
    const bool in_b = r0 >= uint(NA);                      // uniform per threadgroup
    if (in_b) B_MMA8(K, KS, NB, wb, sb, bb, r0 - NA, x, uint(K), lane, sg, red, tile);
    else A_MMA8(K, KS, NB, wa, sa, ba, r0, x, uint(K), lane, sg, red, tile);
    for (uint i = tid; i < uint(8 * NB); i += KS * 32) {
        const uint r = i % 8, b = i / 8;
        if (in_b) yb[b * NB_ROWS + r0 - NA + r] = bfloat16_t(tile[r * 8 + b]);
        else ya[b * NA + r0 + r] = bfloat16_t(tile[r * 8 + b]);
    }
"""
_WQB_ROPE_MMA = r"""
    threadgroup float red[KS * 64], tile[64];
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup, tid = thread_index_in_threadgroup;
    const uint r0 = threadgroup_position_in_grid.x * 8;
    W_MMA8(K, KS, NB, w, sc, bi, r0, x, uint(K), lane, sg, red, tile);
    for (uint i = tid; i < uint(4 * NB); i += KS * 32) {
        const uint pr = i % 4, b = i / 4;                 // request b, RoPE pair (r0 + 2 pr, +1)
        float a = float(bfloat16_t(tile[(2 * pr) * 8 + b])), c = float(bfloat16_t(tile[(2 * pr + 1) * 8 + b]));
        const uint d = (r0 + 2 * pr) % HD;
        if (d >= uint(HD - RD)) {
            const uint p = (d - (HD - RD)) / 2;
            const float cs = cosv[b * (RD / 2) + p], sn = sinv[b * (RD / 2) + p];
            const float ra = a * cs - c * sn, rc = a * sn + c * cs;
            a = ra; c = rc;
        }
        y[b * N + r0 + 2 * pr] = bfloat16_t(a);
        y[b * N + r0 + 2 * pr + 1] = bfloat16_t(c);
    }
"""
_WOA_MMA = r"""
    // 32 rows (one act_quant group) per threadgroup: 4 row tiles x KS K-splits
    threadgroup float red[4][KS * 64], tl[4][64], yb[NB][32];
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup, tid = thread_index_in_threadgroup;
    const uint t = sg / KS, ks = sg % KS, row32 = threadgroup_position_in_grid.x * 32;
    const uint g = row32 / R;
    fp8_mma8<K, KS, NB>(w, sc, row32 + t * 8, x + g * K, uint(G * K), lane, ks, red[t], tl[t]);
    for (uint i = tid; i < uint(32 * NB); i += 4 * KS * 32) {
        const uint r = i % 32, b = i / 32;
        yb[b][r] = float(bfloat16_t(tl[r / 8][(r % 8) * 8 + b]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg < uint(NB)) {
        const float v = yb[sg][lane];
        const float s = act_scale(max(simd_max(abs(v)), 1e-4f));
        yq[sg * (G * R) + row32 + lane] = act_quant_value(v, s);
    }
"""

# Standalone hc_post (last step of a layer).
_HC_POST = r"""
    const uint col = thread_position_in_grid.x;
    if (col >= uint(HC * DIM)) return;
    out[col] = hc_post_value(post, comb, sub, res, col / DIM, col % DIM, HC, DIM);
"""


# Batched single-query attention over REQ requests: the request index selects that request's
# window rows (WSTRIDE bytes each), its compressed rows (offset meta[5] into the concatenated
# pooled buffer), its window / compressed index lists and its slice of every output; blocks
# beyond the request's own count exit. The body is the single-request source.
_ATTN_PRELUDE_B = r"""
    const uint REQ = threadgroup_position_in_grid.z;
    const device int* meta = meta_all + REQ * 8;
    if (int(threadgroup_position_in_grid.y) >= meta[4]) return;
    auto window = window_all + size_t(REQ) * size_t(WSTRIDE);
    auto pooled = pooled_all + size_t(meta[5]) * size_t(D / 2 + D / 16);
    auto wi = wi_all + size_t(REQ) * size_t(WIS);
    auto ci = ci_all + size_t(REQ) * size_t(CMAX);
"""
_ATTN_SCORES_B = _ATTN_PRELUDE_B + r"""
    auto q = q_all + size_t(REQ) * size_t(H * D);
    auto S = S_all + size_t(REQ) * size_t(H * NBMAX * 64);
    auto BMAX = BMAX_all + size_t(REQ) * size_t(H * NBMAX);
""" + _ATTN_SCORES
_ATTN_PV_B = _ATTN_PRELUDE_B + r"""
    auto S = S_all + size_t(REQ) * size_t(H * NBMAX * 64);
    auto BMAX = BMAX_all + size_t(REQ) * size_t(H * NBMAX);
    auto PV = PV_all + size_t(REQ) * size_t(H * NBMAX * D);
    auto DEN = DEN_all + size_t(REQ) * size_t(H * NBMAX);
""" + _ATTN_PV
_O_MERGE_B = r"""
    const uint REQ = threadgroup_position_in_grid.y;
    const device int* meta = meta_all + REQ * 8;
    auto PV = PV_all + size_t(REQ) * size_t(NH * NBMAX * HD);
    auto DEN = DEN_all + size_t(REQ) * size_t(NH * NBMAX);
    auto BMAX = BMAX_all + size_t(REQ) * size_t(NH * NBMAX);
    auto cosv = cosv_all + REQ * (RD / 2);
    auto sinv = sinv_all + REQ * (RD / 2);
    auto oq = oq_all + size_t(REQ) * size_t(NH * HD);
""" + _O_MERGE

# Per-request kernels in a batch: request b = threadgroup_position_in_grid.y reads and writes
# its own slice (stride per buffer) and otherwise runs the single-request source unchanged.
_BATCHED_STRIDES = {
    "mix_partial": {"res": "HC * DIM", "sub": "(POST ? DIM : 0)", "post": "(POST ? HC : 0)", "comb": "(POST ? HC * HC : 0)",
                    "pre_in": "HC", "partial": "NPART * (MIX + 2)", "hout": "(POST ? HC * DIM : 0)", "xpre": "DIM"},
    "mix_final": {"partial": "NPART * (MIX + 2)", "y": "DIM", "mix_out": "MIX", "xn": "DIM", "xq": "DIM"},
    "attn_mid": {"qa": "QR", "kv": "HD", "cosv": "RD / 2", "sinv": "RD / 2",
                 "qr": "QR", "qrq": "QR", "packed": "HD + HD / 32"},
    "hc_post": {"sub": "DIM", "res": "HC * DIM", "post": "HC", "comb": "HC * HC", "out": "HC * DIM"},
}


@cache
def kernel(name, formats=()):
    """The decode kernel ``name``; kernels that read attention projections take their
    weight formats (decode_moe.weight_format) for the roles A, B (gemv pair) or W (wq_b)."""
    specs = {
        "mix_partial": (["res", "fn", "sub", "post", "comb", "pre_in"], ["partial", "hout", "xpre"], _MIX_PARTIAL),
        "mix_final": (["partial", "y", "norm_w", "scale", "base", "eps", "hc_eps"], ["mix_out", "xn", "xq"], _MIX_FINAL),
        "gemv_fp8": (["x", "wa", "sa", "ba", "wb", "sb", "bb"], ["ya", "yb"], _GEMV_FP8),
        "attn_mid": (["qa", "kv", "qnw", "kvnw", "cosv", "sinv", "eps"], ["qr", "qrq", "packed"], _ATTN_MID),
        "wqb_rope": (["x", "w", "sc", "bi", "cosv", "sinv"], ["y"], _WQB_ROPE),
        "woa": (["x", "w", "sc"], ["yq"], _WOA),
        "hc_post": (["sub", "res", "post", "comb"], ["out"], _HC_POST),
        "attn_scores": (["q", "window", "pooled", "wi", "ci", "meta", "scalep"], ["S", "BMAX"], _ATTN_SCORES),
        "attn_pv": (["window", "pooled", "wi", "ci", "meta", "S", "BMAX"], ["PV", "DEN"], _ATTN_PV),
        "o_merge": (["PV", "DEN", "BMAX", "sink", "meta", "cosv", "sinv"], ["oq"], _O_MERGE),
        "gemv_fp8_nb": (["x", "wa", "sa", "ba", "wb", "sb", "bb"], ["ya", "yb"], _GEMV_FP8_NB),
        "wqb_rope_nb": (["x", "w", "sc", "bi", "cosv", "sinv"], ["y"], _WQB_ROPE_NB),
        "woa_nb": (["x", "w", "sc"], ["yq"], _WOA_NB),
        "gemv_fp8_mma": (["x", "wa", "sa", "ba", "wb", "sb", "bb"], ["ya", "yb"], _GEMV_FP8_MMA),
        "wqb_rope_mma": (["x", "w", "sc", "bi", "cosv", "sinv"], ["y"], _WQB_ROPE_MMA),
        "woa_mma": (["x", "w", "sc"], ["yq"], _WOA_MMA),
        "attn_scores_bq": (["q_all", "window_all", "pooled_all", "wi_all", "ci_all", "meta_all", "scalep"],
                           ["S_all", "BMAX_all"], _ATTN_SCORES_B),
        "attn_pv_bq": (["window_all", "pooled_all", "wi_all", "ci_all", "meta_all", "S_all", "BMAX_all"],
                       ["PV_all", "DEN_all"], _ATTN_PV_B),
        "o_merge_bq": (["PV_all", "DEN_all", "BMAX_all", "sink", "meta_all", "cosv_all", "sinv_all"],
                       ["oq_all"], _O_MERGE_B),
    }
    batched = name.endswith("_b") and name[:-2] in _BATCHED_STRIDES
    base = name[:-2] if batched else name
    inputs, outputs, source = specs[base]
    if batched:
        strides = _BATCHED_STRIDES[base]
        prelude = "    const uint BROW = threadgroup_position_in_grid.y;\n" + "".join(
            f"    auto {n} = {n}_all + size_t(BROW) * size_t({strides[n]});\n" for n in inputs + outputs if n in strides)
        inputs = [n + "_all" if n in strides else n for n in inputs]
        outputs = [n + "_all" if n in strides else n for n in outputs]
        source = prelude + source
    header = _HEADER
    if base.startswith("attn_"):
        from .packed_attention import _READ

        header = _READ + "\n#include <metal_simdgroup_matrix>\n" + _HEADER + _KV_RUN
    full = "v41_decode_" + name
    if formats:
        roles = ("W",) if base.startswith("wqb") else ("A", "B")
        header = format_macros(zip(roles, formats)) + header
        full += "_" + format_tag(formats)
    return mx.fast.metal_kernel(name=full, input_names=inputs,
                                output_names=outputs, source=source, header=header)
