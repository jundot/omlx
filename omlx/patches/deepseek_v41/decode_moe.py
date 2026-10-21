# SPDX-License-Identifier: MIT
"""Fused single-token MoE for DeepSeek V4.1 decode: 3 dispatches instead of ~30.

Follows DeepSeek-V4.1-Flash-1017 inference/model.py (Gate, Expert, MoE) and kernel.py:
  gate:    scores = sqrt(softplus(linear(x.float(), W.float()) / gate_temp)), softplus with
           torch's threshold 20; top-k of (scores + bias) picks experts, weights are the
           unbiased scores, w / (sum(w) + 1e-20) * route_scale
  expert:  act_quant(x) -> fp4/fp8/affine GEMM -> bf16 gate, up; clamp(up, +-limit),
           min(gate, limit); silu(gate) * up * weight in fp32 -> bf16 -> act_quant ->
           w2 GEMM -> bf16
  combine: fp32 sum of the routed bf16 outputs plus the shared expert's, stored as bf16
Every bf16/fp8 rounding point above is reproduced exactly. Free (unspecified in the
reference): fp32 accumulation order of each GEMM row and of the expert sum, the order of
tied top-k candidates, and the transcendentals (exp, log1p, sqrt). Weight times activation
products are exact in fp32 (fp4 or fp8 times a power-of-two scale times bf16), so each
GEMM row differs from any other fp32 order by at most gamma_K * sum|x_i w_i| before its
bf16 rounding (K = 5120 or 2304, gamma_K = K 2^-24 / (1 - K 2^-24)). Affine weights (oQ
exports) add scale * sum(q x) + bias * sum(x) per group, the order MLX's qdot uses.

Why this differs from upstream oMLX's decode MoE (language.MoE): same math, but gate, top-k,
quantization, the seven expert GEMMs, SwiGLU and the combine run in three kernels with the
selected experts' packed MXFP4 / MXFP8 / affine weights read in place (weight_format).
"""

from functools import cache

import mlx.core as mx

_HEADER = r"""
#include <metal_simdgroup_matrix>
constant float FP4_VALUES[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                 -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
inline float e8m0(uchar s) { return s == 0 ? 0x1p-127f : as_type<float>(uint(s) << 23); }
inline float e4m3(uint code) {
    const uint a = code & 127u, e = a >> 3, m = a & 7u;
    const float v = e == 0 ? float(m) * 0x1p-9f : as_type<float>(((e + 120u) << 23) | (m << 20));
    return (code & 128u) ? -v : v;
}
// act_quant scale for an amax already floored at 1e-4 (kernel.py fast_round_scale).
inline float act_scale(float amax) {
    const uint b = as_type<uint>(amax * (1.0f / 448.0f));
    const int e = int((b >> 23) & 0xFFu) - 127 + int((b & 0x7FFFFFu) != 0u);
    return as_type<float>(uint(e + 127) << 23);
}
// E4M3FN round-to-nearest-even of a value already clamped to [-448, 448].
inline float round_e4m3(float v) {
    const float a = abs(v);
    const int step_e = max(int((as_type<uint>(a) >> 23) & 0xFFu) - 127 - 3, -9);
    const float step = as_type<float>(uint(step_e + 127) << 23);
    const float q = min(rint(a / step) * step, 448.0f);
    return v < 0.0f ? -q : q;
}
// One act_quant(inplace) element: x / s clamped, cast to fp8, times s, cast to bf16.
inline float act_quant_value(float x, float s) {
    return float(bfloat16_t(round_e4m3(clamp(x / s, -448.0f, 448.0f)) * s));
}
// Exact FP8 E4M3 / FP4 E2M1 decode via fp16 bit patterns: shifting the code's exponent and
// mantissa bits into an fp16 encoding gives value * 2^-8 (E4M3) or value * 2^-14 (E2M1) for
// normal and subnormal codes alike. Products with act_quant-ed inputs (<= 4 significant bits,
// magnitude >= 2^-31) are exact in fp32 either way, and multiplying the group partial by 2^8 /
// 2^14 before the e8m0 scale is exact (no overflow or underflow at these magnitudes), so the
// per-group partial equals the reference's partial * scale for the same summation order.
// Codes are decoded two at a time: two codes 16 bits apart in a packed word become the two
// fp16 halves of one half2 with the same 32-bit masks and shifts, half the integer work of
// decoding them one by one (decode issue bounds these GEMVs: B=1..8 decode +4..9%).
// v[j] = E2M1 code j (bits 4j..4j+3) of p, times 2^-14.
inline void fp4_decode8(uint p, thread float* v) {
    for (int j = 0; j < 4; ++j) {
        const uint q = p >> (4 * j);
        const float2 f = float2(as_type<half2>(((q & 0x00080008u) << 12) | ((q & 0x00070007u) << 9)));
        v[j] = f.x; v[j + 4] = f.y;
    }
}
// v[j] = E4M3 code j (bits 8j..8j+7) of p, times 2^-8.
inline void fp8_decode4(uint p, thread float* v) {
    const float2 f02 = float2(as_type<half2>(((p & 0x00800080u) << 8) | ((p & 0x007F007Fu) << 7)));
    const uint q = p >> 8;
    const float2 f13 = float2(as_type<half2>(((q & 0x00800080u) << 8) | ((q & 0x007F007Fu) << 7)));
    v[0] = f02.x; v[1] = f13.x; v[2] = f02.y; v[3] = f13.y;
}
// ROWS rows of an MXFP8 matrix ([*, K] bytes as uint32, one e8m0 scale per 32) dotted with an
// fp32 vector; each lane covers 8 contiguous values per step (MLX qmv_fast layout), x is
// loaded once per step and reused for all ROWS rows. fp32 accumulation order is free.
template <int K, int ROWS, typename W, typename S, typename X>
inline void fp8_rows(W w, S s, uint row0, X x, uint lane, thread float* acc) {
    static_assert(K % 256 == 0, "K must be a multiple of 32 lanes x 8 values");
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;
    for (uint k = lane * 8; k < uint(K); k += 256) {
        float xv[8];
        for (int i = 0; i < 8; ++i) xv[i] = x[k + i];
        for (int r = 0; r < ROWS; ++r) {
            const auto wr = w + size_t(row0 + r) * (K / 4) + k / 4;
            const uint p0 = wr[0], p1 = wr[1];
            float a = 0.0f;
            float wv[8];
            fp8_decode4(p0, wv); fp8_decode4(p1, wv + 4);
            for (int j = 0; j < 8; ++j) a += wv[j] * xv[j];
            acc[r] += (a * 256.0f) * e8m0(s[size_t(row0 + r) * (K / 32) + k / 32]);
        }
    }
    for (int r = 0; r < ROWS; ++r) acc[r] = simd_sum(acc[r]);
}
// fp8_rows for NB activation vectors (x + b * XS) sharing each weight load: the per-vector
// arithmetic is fp8_rows' (same lane split, same in-group order), so each vector's result
// is the single-vector result. acc[r * NB + b]. The ROWS weight groups are decoded once per
// step and each vector's 8 values are loaded once and applied to all ROWS rows: loading x
// per (row, vector) made the activation loads 4x the needed ones, which bound the kernel
// from NB = 4 up (wo_b at NB = 8: 35 GB/s).
template <int K, int ROWS, int NB, typename W, typename S, typename X>
inline void fp8_rows_nb(W w, S s, uint row0, X x, uint XS, uint lane, thread float* acc) {
    static_assert(K % 256 == 0, "K must be a multiple of 32 lanes x 8 values");
    for (int i = 0; i < ROWS * NB; ++i) acc[i] = 0.0f;
    for (uint k = lane * 8; k < uint(K); k += 256) {
        float wv[ROWS][8], sc[ROWS];
        for (int r = 0; r < ROWS; ++r) {
            const auto wr = w + size_t(row0 + r) * (K / 4) + k / 4;
            const uint p0 = wr[0], p1 = wr[1];
            fp8_decode4(p0, wv[r]); fp8_decode4(p1, wv[r] + 4);
            sc[r] = e8m0(s[size_t(row0 + r) * (K / 32) + k / 32]);
        }
        for (int b = 0; b < NB; ++b) {
            const auto xb = x + size_t(b) * XS + k;
            float xv[8];
            for (int j = 0; j < 8; ++j) xv[j] = float(xb[j]);
            for (int r = 0; r < ROWS; ++r) {
                float a = 0.0f;
                for (int j = 0; j < 8; ++j) a += wv[r][j] * xv[j];
                acc[r * NB + b] += (a * 256.0f) * sc[r];
            }
        }
    }
    for (int i = 0; i < ROWS * NB; ++i) acc[i] = simd_sum(acc[i]);
}
// fp8_rows_nb on simdgroup matrices, for batches (NB >= 2): rows r0..r0+7 of an MXFP8 matrix
// against NB <= 8 activation vectors (x + b * XS; vectors >= NB read as 0), written to
// tile[row * 8 + vector] in threadgroup memory. Per 32-wide scale group a lane loads 8
// contiguous weight bytes and float4 slices of two vectors feeding 4 MMAs (k-columns
// base + 4c + j); the group sum is scaled by the row's e8m0 scale and accumulated, and the KS
// simdgroups splitting K are summed in order. Why this is correct: every product is the
// same exact fp8 x act_quant product as in fp8_rows; only the fp32 summation order differs,
// which the reference leaves free (tensor-core order). Versus fp8_rows_nb it is not bitwise
// equal: on real 1017 weights both match bf16(exact float64 dot) on >= 99.9% of outputs and
// stay far inside the fp32 accumulation bound K * 2^-24 * sum|w x| (mma_accuracy.py).
// It has threadgroup barriers: every simdgroup of the threadgroup must call it.
template <int K, int KS, int NB, typename W, typename S, typename X>
inline void fp8_mma8(W w, S s, uint r0, X x, uint XS, uint lane, uint ks,
                     threadgroup float* red, threadgroup float* tile) {
    constexpr int G = K / 32;
    static_assert(K % 32 == 0 && G % KS == 0, "K split must cover whole scale groups");
    constexpr int GPS = G / KS;
    const uint fm = ((lane >> 2) & 4) + ((lane >> 1) & 3);
    const uint fn = (((lane >> 2) & 2) << 1) + ((lane & 1) << 1);
    simdgroup_matrix<float, 8, 8> acc, part, a, b;
    acc.thread_elements()[0] = 0.0f; acc.thread_elements()[1] = 0.0f;
    const device uchar* wrow = reinterpret_cast<const device uchar*>(w + size_t(r0 + fm) * (K / 4));
    const auto srow = s + size_t(r0 + fm) * G;
    const bool v0 = int(fn) < NB, v1 = int(fn + 1) < NB;
    const auto x0 = x + size_t(v0 ? fn : 0) * XS;
    const auto x1 = x + size_t(v1 ? fn + 1 : 0) * XS;
    for (int g = int(ks) * GPS; g < int(ks + 1) * GPS; ++g) {
        const uint base = uint(g) * 32;
        const uint2 wb = *reinterpret_cast<const device uint2*>(wrow + base + 4 * fn);
        const float4 xa = v0 ? float4(x0[base + 4 * fm], x0[base + 4 * fm + 1], x0[base + 4 * fm + 2], x0[base + 4 * fm + 3]) : float4(0.0f);
        const float4 xb = v1 ? float4(x1[base + 4 * fm], x1[base + 4 * fm + 1], x1[base + 4 * fm + 2], x1[base + 4 * fm + 3]) : float4(0.0f);
        part.thread_elements()[0] = 0.0f; part.thread_elements()[1] = 0.0f;
        float wx[4], wy[4];
        fp8_decode4(wb.x, wx); fp8_decode4(wb.y, wy);
        for (int j = 0; j < 4; ++j) {
            a.thread_elements()[0] = wx[j];
            a.thread_elements()[1] = wy[j];
            b.thread_elements()[0] = xa[j];
            b.thread_elements()[1] = xb[j];
            simdgroup_multiply_accumulate(part, a, b, part);
        }
        const float sc = e8m0(srow[g]) * 256.0f;
        acc.thread_elements()[0] += part.thread_elements()[0] * sc;
        acc.thread_elements()[1] += part.thread_elements()[1] * sc;
    }
    red[ks * 64 + fm * 8 + fn] = acc.thread_elements()[0];
    red[ks * 64 + fm * 8 + fn + 1] = acc.thread_elements()[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ks == 0) {
        for (int e = 0; e < 2; ++e) {
            float t = 0.0f;
            for (int q = 0; q < KS; ++q) t += red[q * 64 + fm * 8 + fn + e];
            tile[fm * 8 + fn + e] = t;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
// Same for MXFP4 (8 values per uint32): 16 contiguous values per lane per step when K allows
// (K % 512 == 0), else 8 (e.g. the 2304-wide expert down projection).
template <int K, int ROWS, typename W, typename S, typename X>
inline void fp4_rows(W w, S s, uint row0, X x, uint lane, thread float* acc) {
    constexpr int NW = (K % 512 == 0) ? 2 : 1;      // uint32 words per lane per step
    static_assert(K % (256 * NW) == 0, "K must be a multiple of 32 lanes x 8 values");
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;
    for (uint k = lane * 8 * NW; k < uint(K); k += 256 * NW) {
        float xv[8 * NW];
        for (int i = 0; i < 8 * NW; ++i) xv[i] = x[k + i];
        for (int r = 0; r < ROWS; ++r) {
            const auto wr = w + size_t(row0 + r) * (K / 8) + k / 8;
            float a = 0.0f;
            for (int q = 0; q < NW; ++q) {
                float wv[8];
                fp4_decode8(wr[q], wv);
                for (int j = 0; j < 8; ++j) a += wv[j] * xv[8 * q + j];
            }
            acc[r] += (a * 16384.0f) * e8m0(s[size_t(row0 + r) * (K / 32) + k / 32]);
        }
    }
    for (int r = 0; r < ROWS; ++r) acc[r] = simd_sum(acc[r]);
}
// Affine weights (MLX layout, as oQ exports store them) at 3 or 5 bits, read like MLX's qdot: x is
// pre-scaled per term so each code is multiplied masked in place; a code that straddles a byte
// gives two terms. A lane covers 16 contiguous values: aff_x16 gives their terms' x and their sum,
// aff_w16 one row's masked codes. Their dot is order-fixed (explicit fma), so the one- and
// multi-vector helpers give a vector the same value.
template <int BITS>
constexpr int aff_terms16() {
    static_assert(BITS == 3 || BITS == 5, "other widths are read with shifts (aff8_rows)");
    return BITS == 3 ? 20 : 24;
}
template <int BITS, typename X>
inline float aff_x16(X x, thread float* t) {
    float xv[16], sum = 0.0f;
    for (int i = 0; i < 16; ++i) { xv[i] = float(x[i]); sum += xv[i]; }
    if (BITS == 3) {
        for (int i = 0, o = 0; i < 16; i += 8, o += 10) {
            t[o] = xv[i]; t[o + 1] = xv[i + 1] / 8.0f; t[o + 2] = xv[i + 2] / 64.0f; t[o + 3] = xv[i + 2] * 4.0f;
            t[o + 4] = xv[i + 3] / 2.0f; t[o + 5] = xv[i + 4] / 16.0f; t[o + 6] = xv[i + 5] / 128.0f; t[o + 7] = xv[i + 5] * 2.0f;
            t[o + 8] = xv[i + 6] / 4.0f; t[o + 9] = xv[i + 7] / 32.0f;
        }
    } else {
        for (int i = 0, o = 0; i < 16; i += 8, o += 12) {
            t[o] = xv[i]; t[o + 1] = xv[i + 1] / 32.0f; t[o + 2] = xv[i + 1] * 8.0f; t[o + 3] = xv[i + 2] / 4.0f;
            t[o + 4] = xv[i + 3] / 128.0f; t[o + 5] = xv[i + 3] * 2.0f; t[o + 6] = xv[i + 4] / 16.0f; t[o + 7] = xv[i + 4] * 16.0f;
            t[o + 8] = xv[i + 5] / 2.0f; t[o + 9] = xv[i + 6] / 64.0f; t[o + 10] = xv[i + 6] * 4.0f; t[o + 11] = xv[i + 7] / 8.0f;
        }
    }
    return sum;
}
template <int BITS>
inline void aff_w16(const device uchar* p, thread float* m) {
    if (BITS == 3) {
        for (int i = 0; i < 2; ++i) {
            const uint b0 = p[3 * i], b1 = p[3 * i + 1], b2 = p[3 * i + 2];
            thread float* o = m + 10 * i;
            o[0] = float(b0 & 0x07u); o[1] = float(b0 & 0x38u); o[2] = float(b0 & 0xc0u); o[3] = float(b1 & 0x01u);
            o[4] = float(b1 & 0x0eu); o[5] = float(b1 & 0x70u); o[6] = float(b1 & 0x80u); o[7] = float(b2 & 0x03u);
            o[8] = float(b2 & 0x1cu); o[9] = float(b2 & 0xe0u);
        }
    } else {
        for (int i = 0; i < 2; ++i) {
            const device uchar* q = p + 5 * i;
            thread float* o = m + 12 * i;
            o[0] = float(q[0] & 0x1fu); o[1] = float(q[0] & 0xe0u); o[2] = float(q[1] & 0x03u); o[3] = float(q[1] & 0x7cu);
            o[4] = float(q[1] & 0x80u); o[5] = float(q[2] & 0x0fu); o[6] = float(q[2] & 0xf0u); o[7] = float(q[3] & 0x01u);
            o[8] = float(q[3] & 0x3eu); o[9] = float(q[3] & 0xc0u); o[10] = float(q[4] & 0x07u); o[11] = float(q[4] & 0xf8u);
        }
    }
}
// ROWS rows of an affine matrix (MLX layout: K * BITS / 32 uint32 words per row, a scale and
// bias per GS values) dotted with an fp32 vector: scale * sum(q x) + bias * sum(x) per lane slice.
template <int K, int ROWS, int BITS, int GS, typename W, typename S, typename B, typename X>
inline void aff16_rows(W w, S s, B b, uint row0, X x, uint lane, thread float* acc) {
    static_assert(K % 256 == 0 && GS % 16 == 0, "K must be a multiple of 16 lanes x 16 values");
    constexpr int NT = aff_terms16<BITS>();
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;
    for (uint k = lane * 16; k < uint(K); k += 512) {
        float t[NT];
        const float xs = aff_x16<BITS>(x + k, t);
        for (int r = 0; r < ROWS; ++r) {
            float m[NT];
            aff_w16<BITS>(reinterpret_cast<const device uchar*>(w + size_t(row0 + r) * (K * BITS / 32)) + k * BITS / 8, m);
            float a = 0.0f;
            for (int i = 0; i < NT; ++i) a = fma(m[i], t[i], a);
            const size_t g = size_t(row0 + r) * (K / GS) + k / GS;
            acc[r] = fma(float(s[g]), a, fma(float(b[g]), xs, acc[r]));
        }
    }
    for (int r = 0; r < ROWS; ++r) acc[r] = simd_sum(acc[r]);
}
// aff16_rows for NB vectors (x + v * XS) sharing each row's masked codes.
template <int K, int ROWS, int NB, int BITS, int GS, typename W, typename S, typename B, typename X>
inline void aff16_rows_nb(W w, S s, B b, uint row0, X x, uint XS, uint lane, thread float* acc) {
    static_assert(K % 256 == 0 && GS % 16 == 0, "K must be a multiple of 16 lanes x 16 values");
    constexpr int NT = aff_terms16<BITS>();
    for (int i = 0; i < ROWS * NB; ++i) acc[i] = 0.0f;
    for (uint k = lane * 16; k < uint(K); k += 512) {
        float m[ROWS][NT], sc[ROWS], bi[ROWS];
        for (int r = 0; r < ROWS; ++r) {
            aff_w16<BITS>(reinterpret_cast<const device uchar*>(w + size_t(row0 + r) * (K * BITS / 32)) + k * BITS / 8, m[r]);
            const size_t g = size_t(row0 + r) * (K / GS) + k / GS;
            sc[r] = float(s[g]);
            bi[r] = float(b[g]);
        }
        for (int v = 0; v < NB; ++v) {
            float t[NT];
            const float xs = aff_x16<BITS>(x + size_t(v) * XS + k, t);
            for (int r = 0; r < ROWS; ++r) {
                float a = 0.0f;
                for (int i = 0; i < NT; ++i) a = fma(m[r][i], t[i], a);
                acc[r * NB + v] = fma(sc[r], a, fma(bi[r], xs, acc[r * NB + v]));
            }
        }
    }
    for (int i = 0; i < ROWS * NB; ++i) acc[i] = simd_sum(acc[i]);
}
// Affine weights read with shifts, 8 contiguous values per lane per step (rows at 2, 4, 6 or 8
// bits, matrix tiles at every width). q[j] = code of value j of the 8 values starting at byte p.
template <int BITS>
inline void affine_codes8(const device uchar* p, thread float* q) {
    constexpr uint MASK = (1u << BITS) - 1u;
    if (BITS == 8) {
        const uint2 v = *reinterpret_cast<const device uint2*>(p);
        for (int j = 0; j < 4; ++j) { q[j] = float((v.x >> (8 * j)) & MASK); q[j + 4] = float((v.y >> (8 * j)) & MASK); }
    } else if (BITS == 4) {
        const uint v = *reinterpret_cast<const device uint*>(p);
        for (int j = 0; j < 8; ++j) q[j] = float((v >> (4 * j)) & MASK);
    } else if (BITS == 2) {
        const uint v = *reinterpret_cast<const device ushort*>(p);
        for (int j = 0; j < 8; ++j) q[j] = float((v >> (2 * j)) & MASK);
    } else if (BITS == 6) {   // 6 bytes
        const device ushort* h = reinterpret_cast<const device ushort*>(p);
        const uint lo = uint(h[0]) | (uint(h[1]) << 16), hi = h[2];
        for (int j = 0; j < 5; ++j) q[j] = float((lo >> (6 * j)) & MASK);
        q[5] = float(((lo >> 30) | (hi << 2)) & MASK);
        q[6] = float((hi >> 4) & MASK);
        q[7] = float((hi >> 10) & MASK);
    } else {   // 3 or 5 bits: 3 or 5 bytes, byte aligned
        uint lo = 0, hi = 0;
        for (int i = 0; i < min(BITS, 4); ++i) lo |= uint(p[i]) << (8 * i);
        if (BITS > 4) hi = p[4];
        for (int j = 0; j < 8; ++j) {
            const int bit = BITS * j;
            const uint v = bit + BITS <= 32 ? lo >> bit : bit >= 32 ? hi >> (bit - 32) : (lo >> bit) | (hi << (32 - bit));
            q[j] = float(v & MASK);
        }
    }
}
template <int K, int ROWS, int BITS, int GS, typename W, typename S, typename B, typename X>
inline void aff8_rows(W w, S s, B b, uint row0, X x, uint lane, thread float* acc) {
    static_assert(K % 256 == 0 && GS % 8 == 0, "K must be a multiple of 32 lanes x 8 values");
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;
    for (uint k = lane * 8; k < uint(K); k += 256) {
        float xv[8], xs = 0.0f;
        for (int i = 0; i < 8; ++i) { xv[i] = x[k + i]; xs += xv[i]; }
        for (int r = 0; r < ROWS; ++r) {
            const auto wr = reinterpret_cast<const device uchar*>(w + size_t(row0 + r) * (K * BITS / 32)) + k * BITS / 8;
            float q[8];
            affine_codes8<BITS>(wr, q);
            float a = 0.0f;
            for (int j = 0; j < 8; ++j) a = fma(q[j], xv[j], a);
            const size_t g = size_t(row0 + r) * (K / GS) + k / GS;
            acc[r] = fma(float(s[g]), a, fma(float(b[g]), xs, acc[r]));
        }
    }
    for (int r = 0; r < ROWS; ++r) acc[r] = simd_sum(acc[r]);
}
// aff8_rows for NB vectors sharing each weight decode; per vector the arithmetic is aff8_rows'.
template <int K, int ROWS, int NB, int BITS, int GS, typename W, typename S, typename B, typename X>
inline void aff8_rows_nb(W w, S s, B b, uint row0, X x, uint XS, uint lane, thread float* acc) {
    static_assert(K % 256 == 0 && GS % 8 == 0, "K must be a multiple of 32 lanes x 8 values");
    for (int i = 0; i < ROWS * NB; ++i) acc[i] = 0.0f;
    for (uint k = lane * 8; k < uint(K); k += 256) {
        float q[ROWS][8], sc[ROWS], bi[ROWS];
        for (int r = 0; r < ROWS; ++r) {
            const auto wr = reinterpret_cast<const device uchar*>(w + size_t(row0 + r) * (K * BITS / 32)) + k * BITS / 8;
            affine_codes8<BITS>(wr, q[r]);
            const size_t g = size_t(row0 + r) * (K / GS) + k / GS;
            sc[r] = float(s[g]);
            bi[r] = float(b[g]);
        }
        for (int v = 0; v < NB; ++v) {
            const auto xb = x + size_t(v) * XS + k;
            float xv[8], xs = 0.0f;
            for (int i = 0; i < 8; ++i) { xv[i] = float(xb[i]); xs += xv[i]; }
            for (int r = 0; r < ROWS; ++r) {
                float a = 0.0f;
                for (int j = 0; j < 8; ++j) a = fma(q[r][j], xv[j], a);
                acc[r * NB + v] = fma(sc[r], a, fma(bi[r], xs, acc[r * NB + v]));
            }
        }
    }
    for (int i = 0; i < ROWS * NB; ++i) acc[i] = simd_sum(acc[i]);
}
// fp8_mma8 for affine weights: rows r0..r0+7 against NB <= 8 vectors on simdgroup matrices, with
// fp8_mma8's k mapping (column c of a 32-value sub-block is k = base + 4c + j for MMA j), so a lane
// decodes 8 contiguous codes of its row. Per GS-group: scale * sum(q x) (MMA) + bias * sum(x). A
// vector's result does not depend on NB, so one-token decode and verify blocks agree bitwise.
template <int K, int KS, int NB, int BITS, int GS, typename W, typename S, typename B, typename X>
inline void aff_mma8(W w, S s, B b, uint r0, X x, uint XS, uint lane, uint ks,
                     threadgroup float* red, threadgroup float* tile) {
    constexpr int G = K / GS;
    static_assert(GS % 32 == 0 && G % KS == 0, "K split must cover whole groups");
    constexpr int GPS = G / KS;
    const uint fm = ((lane >> 2) & 4) + ((lane >> 1) & 3);
    const uint fn = (((lane >> 2) & 2) << 1) + ((lane & 1) << 1);
    simdgroup_matrix<float, 8, 8> part, a, bm;
    float acc0 = 0.0f, acc1 = 0.0f;
    const device uchar* wrow = reinterpret_cast<const device uchar*>(w + size_t(r0 + fm) * (K * BITS / 32));
    const bool v0 = int(fn) < NB, v1 = int(fn + 1) < NB;
    const auto x0 = x + size_t(v0 ? fn : 0) * XS;
    const auto x1 = x + size_t(v1 ? fn + 1 : 0) * XS;
    for (int g = int(ks) * GPS; g < int(ks + 1) * GPS; ++g) {
        part.thread_elements()[0] = 0.0f; part.thread_elements()[1] = 0.0f;
        float xs0 = 0.0f, xs1 = 0.0f;
        for (int sb = 0; sb < GS / 32; ++sb) {
            const uint base = uint(g) * GS + uint(sb) * 32;
            float q[8], xa[4], xb[4];
            affine_codes8<BITS>(wrow + (base + 4 * fn) * BITS / 8, q);
            for (int j = 0; j < 4; ++j) {
                xa[j] = v0 ? float(x0[base + 4 * fm + j]) : 0.0f;
                xb[j] = v1 ? float(x1[base + 4 * fm + j]) : 0.0f;
                xs0 += xa[j]; xs1 += xb[j];
            }
            for (int j = 0; j < 4; ++j) {
                a.thread_elements()[0] = q[j];
                a.thread_elements()[1] = q[j + 4];
                bm.thread_elements()[0] = xa[j];
                bm.thread_elements()[1] = xb[j];
                simdgroup_multiply_accumulate(part, a, bm, part);
            }
        }
        xs0 += simd_shuffle_xor(xs0, 2u); xs0 += simd_shuffle_xor(xs0, 4u); xs0 += simd_shuffle_xor(xs0, 16u);
        xs1 += simd_shuffle_xor(xs1, 2u); xs1 += simd_shuffle_xor(xs1, 4u); xs1 += simd_shuffle_xor(xs1, 16u);
        const size_t gi = size_t(r0 + fm) * G + g;
        const float sc = float(s[gi]), bi = float(b[gi]);
        acc0 = fma(sc, part.thread_elements()[0], fma(bi, xs0, acc0));
        acc1 = fma(sc, part.thread_elements()[1], fma(bi, xs1, acc1));
    }
    red[ks * 64 + fm * 8 + fn] = acc0;
    red[ks * 64 + fm * 8 + fn + 1] = acc1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ks == 0) {
        for (int e = 0; e < 2; ++e) {
            float t = 0.0f;
            for (int q = 0; q < KS; ++q) t += red[q * 64 + fm * 8 + fn + e];
            tile[fm * 8 + fn + e] = t;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
// Top-k of scores + bias with ties to the lower index (any tie order is spec-conformant).
template <int NE, int K>
inline void topk_select(const device float* scores, const device float* bias, threadgroup int* chosen, uint tid) {
    // Simdgroup 0 keeps the biased scores in registers (lane l holds experts l + 32 i) and takes
    // the K maxima in turn, ties to the lower index; taken experts read as -inf.
    constexpr int PER = (NE + 31) / 32;
    if (tid < 32) {
        float v[PER];
        for (int i = 0; i < PER; ++i) {
            const uint e = tid + 32 * i;
            v[i] = e < uint(NE) ? scores[e] + bias[e] : -INFINITY;
        }
        for (int k = 0; k < K; ++k) {
            float best = -INFINITY; int bi = 0x7FFFFFFF;
            for (int i = 0; i < PER; ++i) {
                const int e = int(tid) + 32 * i;
                if (e < NE && (v[i] > best || (v[i] == best && e < bi))) { best = v[i]; bi = e; }
            }
            for (uint off = 16; off > 0; off >>= 1) {
                const float ov = simd_shuffle_xor(best, off); const int oi = simd_shuffle_xor(bi, off);
                if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
            }
            if (tid == 0) chosen[k] = bi;
            for (int i = 0; i < PER; ++i) if (int(tid) + 32 * i == bi) v[i] = -INFINITY;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
"""

# M1: gate scores of NB tokens + act_quant(x) shared by all experts. KS simdgroups split one
# expert row (8 consecutive dims per lane and step, one 16-byte weight load for all NB
# tokens); the KS partial dots are added in a fixed order.
_GATE = r"""
    const uint lane = thread_index_in_simdgroup, sgi = simdgroup_index_in_threadgroup, part = sgi % KS;
    const uint row = threadgroup_position_in_grid.x * (TPG / 32 / KS) + sgi / KS;
    threadgroup float pdot[TPG / 32][NB];
    if (row < uint(NE)) {
        float dot[NB];
        for (int b = 0; b < NB; ++b) dot[b] = 0.0f;
        const device bfloat16_t* wr = w + row * DIM;
        for (uint k = part * (DIM / KS) + lane * 8; k < (part + 1) * (DIM / KS); k += 256) {
            const float4 w0 = float4(*(const device packed_bfloat4*)(wr + k));
            const float4 w1 = float4(*(const device packed_bfloat4*)(wr + k + 4));
            for (int b = 0; b < NB; ++b) {
                const auto xb = x + b * DIM;
                const float4 a0 = float4(*(const device packed_bfloat4*)(xb + k));
                const float4 a1 = float4(*(const device packed_bfloat4*)(xb + k + 4));
                dot[b] += a0.x * w0.x; dot[b] += a0.y * w0.y; dot[b] += a0.z * w0.z; dot[b] += a0.w * w0.w;
                dot[b] += a1.x * w1.x; dot[b] += a1.y * w1.y; dot[b] += a1.z * w1.z; dot[b] += a1.w * w1.w;
            }
        }
        for (int b = 0; b < NB; ++b) {
            const float dp = simd_sum(dot[b]);
            if (lane == 0) pdot[sgi][b] = dp;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row < uint(NE) && part == 0 && lane == 0) {
        for (int b = 0; b < NB; ++b) {
            float d = 0.0f;
            for (uint p = 0; p < KS; ++p) d += pdot[sgi + p][b];
            const float raw = d / temp[0];
            // torch F.softplus: x above threshold 20, else log1p(exp(x)) (accurate for tiny exp(x)).
            const float y = metal::precise::exp(raw);
            const float sp = raw > 20.0f ? raw
                : (y < 1e-4f ? y * (1.0f - y * (0.5f - y * (1.0f / 3.0f))) : metal::precise::log(1.0f + y));
            scores[b * NE + row] = metal::precise::sqrt(sp);
        }
    }
    // act_quant(x) for the expert GEMMs: groups of 32, spread over the gate grid.
    const uint group = threadgroup_position_in_grid.x * (TPG / 32) + sgi;
    for (uint gb = group; gb < uint(NB * (DIM / 32)); gb += NGROUPS_STRIDE) {
        const uint b = gb / (DIM / 32), g = gb % (DIM / 32);
        const float v = float(x[b * DIM + g * 32 + lane]);
        const float s = act_scale(max(simd_max(abs(v)), 1e-4f));
        xq[b * DIM + g * 32 + lane] = act_quant_value(v, s);
    }
"""

# M2: up projections of the K routed experts (MXFP4) + shared expert (MXFP8), SwiGLU,
# routing weight, bf16, and act_quant over each 32-row block (= one threadgroup).
_UP = r"""
    threadgroup int chosen[8];
    threadgroup float hblock[32];
    const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    topk_select<NE, TOPK>(scores, bias, chosen, tid);
    const uint slot = threadgroup_position_in_grid.y;           // 0..TOPK-1 routed, TOPK shared
    // routing weights (unbiased scores, normalized, scaled) -- identical in every threadgroup
    float wsum = 0.0f;
    for (int j = 0; j < TOPK; ++j) wsum += scores[chosen[j]];
    const float rw = slot < uint(TOPK) ? scores[chosen[slot]] / (wsum + 1e-20f) * route_scale[0] : 1.0f;
    if (tid == 0 && threadgroup_position_in_grid.x == 0 && slot == 0)
        for (int j = 0; j < TOPK; ++j) {
            out_idx[j] = chosen[j];
            out_w[j] = scores[chosen[j]] / (wsum + 1e-20f) * route_scale[0];
        }
    float g4[4], u4[4];
    const uint row0 = threadgroup_position_in_grid.x * 32 + sg * 4;
    if (slot < uint(TOPK)) {
        const uint e = uint(chosen[slot]);
        R1_ROWS(DIM, 4, w1 + size_t(e) * INTER * R1_WPR(DIM), s1p + size_t(e) * INTER * R1_SPR(DIM), b1p + size_t(e) * INTER * R1_SPR(DIM), row0, xq, lane, g4);
        R3_ROWS(DIM, 4, w3 + size_t(e) * INTER * R3_WPR(DIM), s3p + size_t(e) * INTER * R3_SPR(DIM), b3p + size_t(e) * INTER * R3_SPR(DIM), row0, xq, lane, u4);
    } else {
        S1_ROWS(DIM, 4, sw1, ss1, sb1, row0, xq, lane, g4);
        S3_ROWS(DIM, 4, sw3, ss3, sb3, row0, xq, lane, u4);
    }
    if (lane < 4) {
        float gate = float(bfloat16_t(g4[lane])), up = float(bfloat16_t(u4[lane]));   // bf16 GEMM outputs
        if (limit[0] > 0.0f) { up = clamp(up, -limit[0], limit[0]); gate = min(gate, limit[0]); }
        float h = gate / (1.0f + metal::precise::exp(-gate)) * up;               // F.silu(gate) * up
        if (slot < uint(TOPK)) h = rw * h;                                       // weights * x
        hblock[sg * 4 + lane] = float(bfloat16_t(h));                            // x.to(dtype)
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {   // act_quant over this 32-row block (the w2 input's quantization group)
        const float v = hblock[lane];
        const float s = act_scale(max(simd_max(abs(v)), 1e-4f));
        hq[slot * INTER + threadgroup_position_in_grid.x * 32 + lane] = act_quant_value(v, s);
    }
"""

# M3: down projections of all K+1 experts per output row, bf16 per expert, fp32 sum, bf16.
_DOWN = r"""
    // simdgroup s computes expert slot s (s == TOPK: shared expert) for 4 output rows; the
    // per-expert bf16 outputs are then summed in slot order, as in the serial formulation.
    threadgroup float part[(TOPK + 1) * 4];
    const uint lane = thread_index_in_simdgroup, slot = simdgroup_index_in_threadgroup;
    const uint row0 = threadgroup_position_in_grid.x * 4;
    if (slot <= uint(TOPK)) {
        float acc[4];
        const device float* hrow = hq + slot * INTER;
        if (slot < uint(TOPK)) {
            const uint e = uint(idx[slot]);
            R2_ROWS(INTER, 4, w2 + size_t(e) * DIM * R2_WPR(INTER), s2p + size_t(e) * DIM * R2_SPR(INTER), b2p + size_t(e) * DIM * R2_SPR(INTER), row0, hrow, lane, acc);
        } else {
            S2_ROWS(INTER, 4, sw2, ss2, sb2, row0, hrow, lane, acc);
        }
        if (lane < 4) part[slot * 4 + lane] = float(bfloat16_t(acc[lane]));   // each expert's w2 output is bf16
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (slot == 0 && lane < 4) {
        float total = 0.0f;
        for (int s = 0; s <= TOPK; ++s) total += part[s * 4 + lane];
        out[row0 + lane] = bfloat16_t(total);
    }
"""


# ---- batched decode MoE (NB tokens) --------------------------------------------------------
# Up: threadgroup (x, y) = (column, 32-row block). Column x < NB * TOPK is (token x / TOPK,
# routed slot x % TOPK) and runs _UP's routed path on that token; the last column runs the
# shared expert for all NB tokens with one weight read (fp8_rows_nb). Columns vary fastest in
# dispatch order, so tokens routed to the same expert read the same 32 rows at about the same
# time and the repeats hit the GPU caches instead of DRAM (real decode at B = 8 routes 48 slots
# to ~34 distinct experts; row-block-major order: B = 8 +9%, B = 4 +4%).
_UP_NB = r"""
    threadgroup int chosen[8];
    threadgroup float hblock[NB][32];
    const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint ys = threadgroup_position_in_grid.x, rb = threadgroup_position_in_grid.y;
    const uint row0 = rb * 32 + sg * 4;
    if (ys < uint(NB * TOPK)) {
        const uint b = ys / TOPK, slot = ys % TOPK;
        const device float* sc_b = scores + b * NE;
        topk_select<NE, TOPK>(sc_b, bias, chosen, tid);
        float wsum = 0.0f;
        for (int j = 0; j < TOPK; ++j) wsum += sc_b[chosen[j]];
        const float rw = sc_b[chosen[slot]] / (wsum + 1e-20f) * route_scale[0];
        if (tid == 0 && rb == 0 && slot == 0)
            for (int j = 0; j < TOPK; ++j) {
                out_idx[b * TOPK + j] = chosen[j];
                out_w[b * TOPK + j] = sc_b[chosen[j]] / (wsum + 1e-20f) * route_scale[0];
            }
        const uint e = uint(chosen[slot]);
        float g4[4], u4[4];
        R1_ROWS(DIM, 4, w1 + size_t(e) * INTER * R1_WPR(DIM), s1p + size_t(e) * INTER * R1_SPR(DIM), b1p + size_t(e) * INTER * R1_SPR(DIM), row0, xq + b * DIM, lane, g4);
        R3_ROWS(DIM, 4, w3 + size_t(e) * INTER * R3_WPR(DIM), s3p + size_t(e) * INTER * R3_SPR(DIM), b3p + size_t(e) * INTER * R3_SPR(DIM), row0, xq + b * DIM, lane, u4);
        if (lane < 4) {
            float gate = float(bfloat16_t(g4[lane])), up = float(bfloat16_t(u4[lane]));
            if (limit[0] > 0.0f) { up = clamp(up, -limit[0], limit[0]); gate = min(gate, limit[0]); }
            float h = gate / (1.0f + metal::precise::exp(-gate)) * up;
            h = rw * h;
            hblock[0][sg * 4 + lane] = float(bfloat16_t(h));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            const float v = hblock[0][lane];
            const float s = act_scale(max(simd_max(abs(v)), 1e-4f));
            hq[(b * TOPK + slot) * INTER + rb * 32 + lane] = act_quant_value(v, s);
        }
    } else {
        float g4[4 * NB], u4[4 * NB];
        S1_ROWS_NB(DIM, 4, NB, sw1, ss1, sb1, row0, xq, uint(DIM), lane, g4);
        S3_ROWS_NB(DIM, 4, NB, sw3, ss3, sb3, row0, xq, uint(DIM), lane, u4);
        if (lane < 4)
            for (int b = 0; b < NB; ++b) {
                float gate = float(bfloat16_t(g4[lane * NB + b])), up = float(bfloat16_t(u4[lane * NB + b]));
                if (limit[0] > 0.0f) { up = clamp(up, -limit[0], limit[0]); gate = min(gate, limit[0]); }
                hblock[b][sg * 4 + lane] = float(bfloat16_t(gate / (1.0f + metal::precise::exp(-gate)) * up));
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg < uint(NB)) {
            const float v = hblock[sg][lane];
            const float s = act_scale(max(simd_max(abs(v)), 1e-4f));
            hqs[sg * INTER + rb * 32 + lane] = act_quant_value(v, s);
        }
    }
"""
# _UP_NB with the shared expert on simdgroup matrices (fp8_mma8); routed path unchanged.
_UP_NB_MMA = r"""
    threadgroup int chosen[8];
    threadgroup float hblock[NB][32];
    const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint ys = threadgroup_position_in_grid.x, rb = threadgroup_position_in_grid.y;
    const uint row0 = rb * 32 + sg * 4;
    if (ys < uint(NB * TOPK)) {
        const uint b = ys / TOPK, slot = ys % TOPK;
        const device float* sc_b = scores + b * NE;
        topk_select<NE, TOPK>(sc_b, bias, chosen, tid);
        float wsum = 0.0f;
        for (int j = 0; j < TOPK; ++j) wsum += sc_b[chosen[j]];
        const float rw = sc_b[chosen[slot]] / (wsum + 1e-20f) * route_scale[0];
        if (tid == 0 && rb == 0 && slot == 0)
            for (int j = 0; j < TOPK; ++j) {
                out_idx[b * TOPK + j] = chosen[j];
                out_w[b * TOPK + j] = sc_b[chosen[j]] / (wsum + 1e-20f) * route_scale[0];
            }
        const uint e = uint(chosen[slot]);
        float g4[4], u4[4];
        R1_ROWS(DIM, 4, w1 + size_t(e) * INTER * R1_WPR(DIM), s1p + size_t(e) * INTER * R1_SPR(DIM), b1p + size_t(e) * INTER * R1_SPR(DIM), row0, xq + b * DIM, lane, g4);
        R3_ROWS(DIM, 4, w3 + size_t(e) * INTER * R3_WPR(DIM), s3p + size_t(e) * INTER * R3_SPR(DIM), b3p + size_t(e) * INTER * R3_SPR(DIM), row0, xq + b * DIM, lane, u4);
        if (lane < 4) {
            float gate = float(bfloat16_t(g4[lane])), up = float(bfloat16_t(u4[lane]));
            if (limit[0] > 0.0f) { up = clamp(up, -limit[0], limit[0]); gate = min(gate, limit[0]); }
            float h = gate / (1.0f + metal::precise::exp(-gate)) * up;
            h = rw * h;
            hblock[0][sg * 4 + lane] = float(bfloat16_t(h));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            const float v = hblock[0][lane];
            const float s = act_scale(max(simd_max(abs(v)), 1e-4f));
            hq[(b * TOPK + slot) * INTER + rb * 32 + lane] = act_quant_value(v, s);
        }
    } else {
        // shared expert, 32 rows x NB tokens: 4 row tiles x 2 K-halves on the 8 simdgroups
        threadgroup float red[4][2 * 64], tg_g[4][64], tg_u[4][64];
        const uint t = sg / 2, ks = sg % 2, r0 = rb * 32 + t * 8;
        S1_MMA8(DIM, 2, NB, sw1, ss1, sb1, r0, xq, uint(DIM), lane, ks, red[t], tg_g[t]);
        S3_MMA8(DIM, 2, NB, sw3, ss3, sb3, r0, xq, uint(DIM), lane, ks, red[t], tg_u[t]);
        for (uint i = tid; i < uint(32 * NB); i += 256) {
            const uint r = i % 32, b = i / 32;
            float gate = float(bfloat16_t(tg_g[r / 8][(r % 8) * 8 + b])), up = float(bfloat16_t(tg_u[r / 8][(r % 8) * 8 + b]));
            if (limit[0] > 0.0f) { up = clamp(up, -limit[0], limit[0]); gate = min(gate, limit[0]); }
            hblock[b][r] = float(bfloat16_t(gate / (1.0f + metal::precise::exp(-gate)) * up));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg < uint(NB)) {
            const float v = hblock[sg][lane];
            const float s = act_scale(max(simd_max(abs(v)), 1e-4f));
            hqs[sg * INTER + rb * 32 + lane] = act_quant_value(v, s);
        }
    }
"""

# Down, routed: one threadgroup per (token x, 4 output rows y), simdgroup = routed slot; the
# fp32 sum over slots in slot order is kept unrounded for the shared expert's addition.
# Tokens vary fastest in dispatch order (shared experts' rows read together, as in up).
_DOWN_R_NB = r"""
    threadgroup float part[TOPK * 4];
    const uint lane = thread_index_in_simdgroup, slot = simdgroup_index_in_threadgroup;
    const uint b = threadgroup_position_in_grid.x, row0 = threadgroup_position_in_grid.y * 4;
    float acc[4];
    const uint e = uint(idx[b * TOPK + slot]);
    R2_ROWS(INTER, 4, w2 + size_t(e) * DIM * R2_WPR(INTER), s2p + size_t(e) * DIM * R2_SPR(INTER),
            b2p + size_t(e) * DIM * R2_SPR(INTER), row0, hq + (b * TOPK + slot) * INTER, lane, acc);
    if (lane < 4) part[slot * 4 + lane] = float(bfloat16_t(acc[lane]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (slot == 0 && lane < 4) {
        float total = 0.0f;
        for (int s = 0; s < TOPK; ++s) total += part[s * 4 + lane];
        partial[b * DIM + row0 + lane] = total;
    }
"""
# Down, shared: all NB tokens per weight read, then out = bf16(routed partial + bf16(shared)),
# i.e. _DOWN's slot-order sum with the shared expert last.
_DOWN_S_NB = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint row0 = (threadgroup_position_in_grid.x * 2 + simdgroup_index_in_threadgroup) * 4;
    float acc[4 * NB];
    S2_ROWS_NB(INTER, 4, NB, sw2, ss2, sb2, row0, hqs, uint(INTER), lane, acc);
    if (lane < 4)
        for (int b = 0; b < NB; ++b)
            out[b * DIM + row0 + lane] = bfloat16_t(partial[b * DIM + row0 + lane] + float(bfloat16_t(acc[lane * NB + b])));
"""

# _DOWN_S_NB on simdgroup matrices: one 8-row tile per threadgroup, K split over 2 simdgroups.
_DOWN_S_NB_MMA = r"""
    threadgroup float red[2 * 64], tile[64];
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup, tid = thread_index_in_threadgroup;
    const uint r0 = threadgroup_position_in_grid.x * 8;
    S2_MMA8(INTER, 2, NB, sw2, ss2, sb2, r0, hqs, uint(INTER), lane, sg, red, tile);
    for (uint i = tid; i < uint(8 * NB); i += 64) {
        const uint r = i % 8, b = i / 8;
        out[b * DIM + r0 + r] = bfloat16_t(partial[b * DIM + r0 + r] + float(bfloat16_t(tile[r * 8 + b])));
    }
"""



# Batched FP8 GEMVs run on simdgroup matrices from MMA_MIN_BATCH vectors: a request's result
# then agrees with decoding it alone within fp32 summation rounding (see fp8_mma8), an order
# the reference leaves free. Fewer vectors run the fp8_rows_nb kernels, bit for bit per
# request: below 4 most of each 8-wide matrix tile is padding and they are faster
# (BatchGenerator decode, M3 Ultra, 1017 weights: B=2 -8%, B=4 +4%, B=8 +31% on matrices).
MMA_MIN_BATCH = 4

MXFP8 = ("mxfp8", 8, 32)
MXFP4 = ("mxfp4", 4, 32)


def use_mma(nb):
    return nb >= MMA_MIN_BATCH


def weight_format(p):
    """(mode, bits, group_size) of a projection the fused kernels read, or None: MXFP8 /
    MXFP4 with group 32, or affine (oQ exports) with bf16 / fp16 scales and biases."""
    mode = getattr(p, "mode", None)
    if p is None or not getattr(p, "quantize_input", True):
        return None
    if mode in ("mxfp8", "mxfp4"):
        fmt = MXFP8 if mode == "mxfp8" else MXFP4
        return fmt if p.group_size == 32 and "biases" not in p else None
    if (
        mode == "affine"
        and p.bits in (2, 3, 4, 5, 6, 8)
        and p.group_size in (32, 64, 128)
        and "biases" in p
        and p.scales.dtype in (mx.bfloat16, mx.float16)
        and p.biases.dtype == p.scales.dtype
    ):
        return ("affine", p.bits, p.group_size)
    return None


def format_macros(roles):
    """Per-role weight access for the kernel sources: <role>_WPR(K) uint32 words and
    <role>_SPR(K) scales per row, <role>_ROWS / <role>_ROWS_NB / <role>_MMA8 the row dot
    products (one vector, NB vectors, NB vectors on simdgroup matrices)."""
    lines = []
    for role, (mode, bits, gs) in roles:
        if mode == "affine":
            # Masked terms for 3 and 5 bits (codes straddle bytes), shifts otherwise.
            helper = "aff16" if bits in (3, 5) else "aff8"
            wpr, spr = f"((K_) * {bits} / 32)", f"((K_) / {gs})"
            rows = f"{helper}_rows<K_, R_, {bits}, {gs}>(w_, s_, b_, r0_, x_, l_, acc_)"
            nb = f"{helper}_rows_nb<K_, R_, NB_, {bits}, {gs}>(w_, s_, b_, r0_, x_, xs_, l_, acc_)"
            mma = f"aff_mma8<K_, KS_, NB_, {bits}, {gs}>(w_, s_, b_, r0_, x_, xs_, l_, ks_, red_, tile_)"
        elif mode == "mxfp8":
            wpr, spr = "((K_) / 4)", "((K_) / 32)"
            rows = "fp8_rows<K_, R_>(w_, s_, r0_, x_, l_, acc_)"
            nb = "fp8_rows_nb<K_, R_, NB_>(w_, s_, r0_, x_, xs_, l_, acc_)"
            mma = "fp8_mma8<K_, KS_, NB_>(w_, s_, r0_, x_, xs_, l_, ks_, red_, tile_)"
        else:
            wpr, spr = "((K_) / 8)", "((K_) / 32)"
            rows = "fp4_rows<K_, R_>(w_, s_, r0_, x_, l_, acc_)"
            nb = "static_assert(false, \"no MXFP4 multi-vector rows\")"
            mma = "static_assert(false, \"no MXFP4 matrix rows\")"
        lines += [
            f"#define {role}_WPR(K_) {wpr}",
            f"#define {role}_SPR(K_) {spr}",
            f"#define {role}_ROWS(K_, R_, w_, s_, b_, r0_, x_, l_, acc_) {rows}",
            f"#define {role}_ROWS_NB(K_, R_, NB_, w_, s_, b_, r0_, x_, xs_, l_, acc_) {nb}",
            f"#define {role}_MMA8(K_, KS_, NB_, w_, s_, b_, r0_, x_, xs_, l_, ks_, red_, tile_) {mma}",
        ]
    return "\n".join(lines) + "\n"


def format_tag(formats):
    return "_".join(f"{m[0]}{b}g{g}" for m, b, g in formats)


def _biases(p):
    return p.biases if "biases" in p else _const((0,) * 8)


def _formats(moe):
    e, s = moe.experts, moe.shared_experts
    return tuple(weight_format(getattr(m, n, None)) for m in (e, s) for n in ("w1", "w3", "w2"))


_ROLES = ("R1", "R3", "R2", "S1", "S3", "S2")


@cache
def _gate_kernel():
    return mx.fast.metal_kernel(name="v41_decode_moe_gate", input_names=["x", "w", "temp"],
                                output_names=["scores", "xq"], source=_GATE, header=_HEADER)


def _gate(moe, x, config):
    """Gate scores [NB * n_experts] and act_quant(x) [NB * dim] of x: [NB, dim] bf16."""
    nb, dim, ne = x.shape[0], config.dim, config.n_routed_experts
    tpg = 256
    ks = next(k for k in (4, 2, 1) if dim % (256 * k) == 0)
    rows = tpg // 32 // ks
    n_tg = (ne + rows - 1) // rows
    return _gate_kernel()(
        inputs=[x, moe.gate.weight, _const((getattr(config, "gate_temp", 1.0),))],
        template=[("NE", ne), ("DIM", dim), ("TPG", tpg), ("KS", ks), ("NGROUPS_STRIDE", n_tg * (tpg // 32)),
                  ("NB", nb)],
        grid=(n_tg * tpg, 1, 1), threadgroup=(tpg, 1, 1),
        output_shapes=[(nb * ne,), (nb * dim,)], output_dtypes=[mx.float32, mx.float32])


@cache
def _kernels_nb(mma, formats):
    header = format_macros(zip(_ROLES, formats)) + _HEADER
    tag = format_tag(formats)
    up_inputs = ["scores", "bias", "route_scale", "limit", "xq", "w1", "s1p", "b1p", "w3", "s3p", "b3p",
                 "sw1", "ss1", "sb1", "sw3", "ss3", "sb3"]
    up = mx.fast.metal_kernel(
        name=f"v41_decode_moe_up_nb_{tag}", input_names=up_inputs,
        output_names=["hq", "hqs", "out_idx", "out_w"], source=_UP_NB, header=header)
    down_r = mx.fast.metal_kernel(name=f"v41_decode_moe_down_routed_nb_{tag}",
                                  input_names=["hq", "idx", "w2", "s2p", "b2p"],
                                  output_names=["partial"], source=_DOWN_R_NB, header=header)
    down_s = mx.fast.metal_kernel(name=f"v41_decode_moe_down_shared_nb_{tag}",
                                  input_names=["hqs", "partial", "sw2", "ss2", "sb2"],
                                  output_names=["out"], source=_DOWN_S_NB, header=header)
    if mma:
        up = mx.fast.metal_kernel(
            name=f"v41_decode_moe_up_nb_mma_{tag}", input_names=up_inputs,
            output_names=["hq", "hqs", "out_idx", "out_w"], source=_UP_NB_MMA, header=header)
        down_s = mx.fast.metal_kernel(name=f"v41_decode_moe_down_shared_nb_mma_{tag}",
                                      input_names=["hqs", "partial", "sw2", "ss2", "sb2"],
                                      output_names=["out"], source=_DOWN_S_NB_MMA, header=header)
    return up, down_r, down_s


def fused_decode_moe_nb(moe, x, config, mma=None):
    """NB tokens at once. x: [NB, dim] bf16 (ffn_norm outputs). Returns [NB, dim] bf16; each
    token's result equals fused_decode_moe on that token alone, bit for bit without matrix
    tiles (mma=False) and within fp32 summation rounding with them (default: use_mma(NB)).
    Matrix tiles (MXFP8 or affine shared experts) give a token the same result for any NB."""
    nb = x.shape[0]
    formats = _formats(moe)
    mma = (use_mma(nb) if mma is None else mma) and all(f[0] in ("mxfp8", "affine") for f in formats[3:])
    up_k, down_r, down_s = _kernels_nb(mma, formats)
    dim, inter, ne, topk = config.dim, config.moe_inter_dim, config.n_routed_experts, config.n_activated_experts
    scores, xq = _gate(moe, x, config)
    e, s = moe.experts, moe.shared_experts
    hq, hqs, idx, w = up_k(
        inputs=[scores, moe.gate.bias, _const((config.route_scale,)), _const((config.swiglu_limit,)), xq,
                e.w1.weight, e.w1.scales, _biases(e.w1), e.w3.weight, e.w3.scales, _biases(e.w3),
                s.w1.weight, s.w1.scales, _biases(s.w1), s.w3.weight, s.w3.scales, _biases(s.w3)],
        template=[("NE", ne), ("TOPK", topk), ("DIM", dim), ("INTER", inter), ("NB", nb)],
        grid=((nb * topk + 1) * 256, inter // 32, 1), threadgroup=(256, 1, 1),
        output_shapes=[(nb * topk * inter,), (nb * inter,), (nb * topk,), (nb * topk,)],
        output_dtypes=[mx.float32, mx.float32, mx.int32, mx.float32])
    partial = down_r(
        inputs=[hq, idx, e.w2.weight, e.w2.scales, _biases(e.w2)],
        template=[("TOPK", topk), ("DIM", dim), ("INTER", inter)],
        grid=(nb * topk * 32, dim // 4, 1), threadgroup=(topk * 32, 1, 1),
        output_shapes=[(nb * dim,)], output_dtypes=[mx.float32])[0]
    out = down_s(
        inputs=[hqs, partial, s.w2.weight, s.w2.scales, _biases(s.w2)],
        template=[("DIM", dim), ("INTER", inter), ("NB", nb)],
        grid=(dim // 8 * 64, 1, 1), threadgroup=(64, 1, 1),
        output_shapes=[(nb * dim,)], output_dtypes=[mx.bfloat16])[0]
    return out.reshape(nb, dim), idx.reshape(nb, topk), w.reshape(nb, topk)


@cache
def _const(values):
    """Materialized scalar constant (cached; no per-step array construction)."""
    a = mx.array(list(values), mx.float32)
    mx.eval(a)
    return a


@cache
def _kernels(formats):
    header = format_macros(zip(_ROLES, formats)) + _HEADER
    tag = format_tag(formats)
    up = mx.fast.metal_kernel(
        name=f"v41_decode_moe_up_{tag}",
        input_names=["scores", "bias", "route_scale", "limit", "xq", "w1", "s1p", "b1p", "w3", "s3p", "b3p",
                     "sw1", "ss1", "sb1", "sw3", "ss3", "sb3"],
        output_names=["hq", "out_idx", "out_w"], source=_UP, header=header)
    down = mx.fast.metal_kernel(
        name=f"v41_decode_moe_down_{tag}",
        input_names=["hq", "idx", "w2", "s2p", "b2p", "sw2", "ss2", "sb2"],
        output_names=["out"], source=_DOWN, header=header)
    return up, down


def supported(moe, x, config):
    # Offloaded experts (OffloadedExpert) hold resident slots, not w1 / w2 / w3.
    formats = _formats(moe)
    return (
        x.size == config.dim
        and x.dtype == mx.bfloat16
        and getattr(config, "score_func", "sqrtsoftplus") == "sqrtsoftplus"
        and all(f is not None for f in formats)
        and formats[3] != MXFP4 and formats[4] != MXFP4 and formats[5] != MXFP4
        and config.n_activated_experts <= 31
        and config.dim % 256 == 0 and config.moe_inter_dim % 256 == 0
    )


def fused_decode_moe(moe, x, config):
    """x: [1, 1, dim] bf16 (ffn_norm output). Returns ([1, 1, dim] bf16, idx, weights)."""
    up_k, down_k = _kernels(_formats(moe))
    dim, inter, ne, topk = config.dim, config.moe_inter_dim, config.n_routed_experts, config.n_activated_experts
    scores, xq = _gate(moe, x.reshape(1, -1), config)
    e, s = moe.experts, moe.shared_experts
    hq, idx, w = up_k(
        inputs=[scores, moe.gate.bias, _const((config.route_scale,)),
                _const((config.swiglu_limit,)), xq,
                e.w1.weight, e.w1.scales, _biases(e.w1), e.w3.weight, e.w3.scales, _biases(e.w3),
                s.w1.weight, s.w1.scales, _biases(s.w1), s.w3.weight, s.w3.scales, _biases(s.w3)],
        template=[("NE", ne), ("TOPK", topk), ("DIM", dim), ("INTER", inter)],
        grid=(inter // 32 * 256, topk + 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[((topk + 1) * inter,), (topk,), (topk,)],
        output_dtypes=[mx.float32, mx.int32, mx.float32])
    out = down_k(
        inputs=[hq, idx, e.w2.weight, e.w2.scales, _biases(e.w2), s.w2.weight, s.w2.scales, _biases(s.w2)],
        template=[("TOPK", topk), ("DIM", dim), ("INTER", inter)],
        grid=(dim // 4 * (topk + 1) * 32, 1, 1), threadgroup=((topk + 1) * 32, 1, 1),
        output_shapes=[(dim,)], output_dtypes=[mx.bfloat16])[0]
    return out.reshape(x.shape), idx, w
