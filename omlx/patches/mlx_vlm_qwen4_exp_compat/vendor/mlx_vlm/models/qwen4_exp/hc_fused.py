# SPDX-License-Identifier: Apache-2.0
"""Metal kernels for Qwen4 hyper-connection inputs.

Decode (at most 16 BF16 rows) fuses per-stream RMS norm, down/inject
projections with activation, and the up projection with stream mixing.
Prefill fuses the stream norm, silu(down / HC), and the mixing/inject
epilogue around the down/up projections; on GPUs with tensor units (M5) those
projections run in hc_prefill_nax's kernels with the epilogues folded in
(bit-identical to the MLX matmul path), elsewhere on MLX's quantized matmul.
Both need four streams and affine group-size-32 or -64 projections with 4/5/6/8-bit
weights. FP32 epilogues can round differently from the canonical BF16
operations.

A pending residual write (previous block's branch times its injection gate)
can be applied inside the stream norm, which also stores the written
residual; it rounds exactly like the eager multiply and add.

Each kernel specialization is evaluated once to catch lazy compilation errors.
A failure falls back for the current call only; the NAX prefill projections
instead fall back to MLX for the process after a failure or a bitwise
mismatch on first use. Later evaluation errors propagate.
Disable with OMLX_QWEN4_HC_FUSED=0. OMLX_QWEN4_HC_FUSED_WRITE=0 keeps the
decode residual writes eager and stops deferring each layer's tail write into
the next hyper-connection norm. OMLX_QWEN4_HC_DECODE_V2=0 restores the
three-launch decode kernels (norm, down/inject, up/mix) in place of the
two-launch ones; both produce the same bits.
"""

from __future__ import annotations

import logging
from itertools import chain
from operator import is_

import mlx.core as mx
import mlx.nn as nn

from . import hc_prefill_nax
from .hc_projection import _HEADER, env_enabled

logger = logging.getLogger(__name__)

MAX_ROWS = 16
# Each projection's group size is a template parameter (GS_D, GS_I, GS_U): the
# traversal is the same for every group size, only the scale/bias index moves.
_GROUP_SIZES = (32, 64)
_SUPPORTED_BITS = (4, 5, 6, 8)
_DISABLED = not env_enabled("OMLX_QWEN4_HC_FUSED")
_WRITE_DISABLED = not env_enabled("OMLX_QWEN4_HC_FUSED_WRITE")
# Prefill rows on the tensor units (hc_prefill_nax): three dispatches instead of
# six, bit-identical. Disable with OMLX_QWEN4_HC_NAX_PREFILL=0.
_NAX_DISABLED = not env_enabled("OMLX_QWEN4_HC_NAX_PREFILL")
_NAX_AVAILABLE: bool | None = None
_NAX_BROKEN = False
_DECODE_V2 = env_enabled("OMLX_QWEN4_HC_DECODE_V2")
_KERNELS: dict[str, object] = {}
_VALIDATED: set[tuple] = set()
_FAILURE_LOGGED = False
_INELIGIBLE_LOGGED = False

_N_SOURCE = r"""
    const uint row = threadgroup_position_in_grid.z;
    const uint s = threadgroup_position_in_grid.y;
    const uint t = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    threadgroup float part[8];
    constexpr int PER = (H + 255) / 256;
    const device T* xp = x + (size_t)row * K + (size_t)s * H;
    const device T* wp = w + (size_t)s * H;
    device T* op = xn + (size_t)row * K + (size_t)s * H;
    float v[PER];
    float ss = 0.0f;
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        v[i] = (k < H) ? float(xp[k]) : 0.0f;
        ss += v[i] * v[i];
    }
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += part[i];
    const float inv = metal::rsqrt(tot / float(H) + eps[0]);
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        if (k < H) op[k] = T(v[i] * inv * (1.0f + float(wp[k])));
    }
"""

_D_SOURCE = r"""
    const uint tg = threadgroup_position_in_grid.y;
    const uint row = threadgroup_position_in_grid.z;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    constexpr int KS = 4;
    constexpr int SLICE = K / KS;
    constexpr int GROUPS = K / GS_D;
    threadgroup float part[KS][8];
    const device T* x = xn + (size_t)row * K;
    const int rg = int(sg & 1);
    const int ks = int(sg >> 1);
    if (tg < R / 8) {
        constexpr int PF = hc_pack_factor<BITS_D>();
        constexpr int BP = hc_bytes_per_pack<BITS_D>();
        constexpr int ROW_BYTES = K * BP / PF;
        constexpr int PPT = 2;
        constexpr int VPT = PF * PPT;
        constexpr int BLOCK = VPT * 32;
        constexpr int SCALE_STEP = GS_D / VPT;
        const int out_row = int(tg) * 8 + rg * 4;
        const device uint8_t* wp = (const device uint8_t*)down_w
            + out_row * ROW_BYTES + ks * (SLICE * BP / PF) + int(lane) * PPT * BP;
        const device T* sp = down_s + out_row * GROUPS + ks * (SLICE / GS_D)
            + int(lane) / SCALE_STEP;
        const device T* bp = down_b + out_row * GROUPS + ks * (SLICE / GS_D)
            + int(lane) / SCALE_STEP;
        const device T* xp = x + ks * SLICE + int(lane) * VPT;
        float result[4] = {0.0f};
        float xv[VPT];
        constexpr int TAIL = SLICE % BLOCK;
        for (int k = 0; k < SLICE - TAIL; k += BLOCK) {
            float sum = hc_load_vector<T, VPT, BITS_D>(xp, xv);
            for (int r = 0; r < 4; ++r) {
                result[r] += hc_qdot<VPT, BITS_D>(
                    wp + r * ROW_BYTES, xv,
                    float(sp[r * GROUPS]), float(bp[r * GROUPS]), sum);
            }
            wp += BLOCK * BP / PF;
            sp += BLOCK / GS_D;
            bp += BLOCK / GS_D;
            xp += BLOCK;
        }
        // Load only lanes inside the tail; all lanes must join simd_sum.
        if (TAIL > 0 && int(lane) * VPT < TAIL) {
            float sum = hc_load_vector<T, VPT, BITS_D>(xp, xv);
            for (int r = 0; r < 4; ++r) {
                result[r] += hc_qdot<VPT, BITS_D>(
                    wp + r * ROW_BYTES, xv,
                    float(sp[r * GROUPS]), float(bp[r * GROUPS]), sum);
            }
        }
        for (int r = 0; r < 4; ++r) {
            float v = simd_sum(result[r]);
            if (lane == 0) part[ks][rg * 4 + r] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg < 2 && lane < 4) {
            const int r = int(lane);
            float v = part[0][sg * 4 + r] + part[1][sg * 4 + r]
                + part[2][sg * 4 + r] + part[3][sg * 4 + r];
            v = v / float(HC);
            act[(size_t)row * R + int(tg) * 8 + int(sg) * 4 + r]
                = T(v / (1.0f + metal::exp(-v)));
        }
        return;
    }
    if (INJ == 0 || rg != 0) return;
    {
        constexpr int PF = hc_pack_factor<BITS_I>();
        constexpr int BP = hc_bytes_per_pack<BITS_I>();
        constexpr int ROW_BYTES = K * BP / PF;
        constexpr int PPT = 1;
        constexpr int VPT = PF;
        constexpr int BLOCK = VPT * 32;
        constexpr int GROUPS_I = K / GS_I;
        constexpr int SCALE_STEP = GS_I / VPT;
        const device uint8_t* wp = (const device uint8_t*)inject_w
            + ks * (SLICE * BP / PF) + int(lane) * BP;
        const device T* sp = inject_s + ks * (SLICE / GS_I) + int(lane) / SCALE_STEP;
        const device T* bp = inject_b + ks * (SLICE / GS_I) + int(lane) / SCALE_STEP;
        const device T* xp = x + ks * SLICE + int(lane) * VPT;
        float result[HC] = {0.0f};
        float xv[VPT];
        constexpr int TAIL = SLICE % BLOCK;
        for (int k = 0; k < SLICE - TAIL; k += BLOCK) {
            float sum = hc_load_vector<T, VPT, BITS_I>(xp, xv);
            for (int r = 0; r < HC; ++r) {
                result[r] += hc_qdot<VPT, BITS_I>(
                    wp + r * ROW_BYTES, xv,
                    float(sp[r * GROUPS_I]), float(bp[r * GROUPS_I]), sum);
            }
            wp += BLOCK * BP / PF;
            sp += BLOCK / GS_I;
            bp += BLOCK / GS_I;
            xp += BLOCK;
        }
        if (TAIL > 0 && int(lane) * VPT < TAIL) {
            float sum = hc_load_vector<T, VPT, BITS_I>(xp, xv);
            for (int r = 0; r < HC; ++r) {
                result[r] += hc_qdot<VPT, BITS_I>(
                    wp + r * ROW_BYTES, xv,
                    float(sp[r * GROUPS_I]), float(bp[r * GROUPS_I]), sum);
            }
        }
        for (int r = 0; r < HC; ++r) {
            float v = simd_sum(result[r]);
            if (lane == 0) part[ks][r] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0 && lane < HC) {
            const int r = int(lane);
            float v = part[0][r] + part[1][r] + part[2][r] + part[3][r];
            v = v / float(HC);
            inj[(size_t)row * HC + r] = T(2.0f / (1.0f + metal::exp(-v)));
        }
    }
"""

_U_SOURCE = r"""
    const uint g = threadgroup_position_in_grid.y;
    const uint t = thread_index_in_threadgroup;
    const int h = int(g) * 64 + int(t >> 2);
    const int s = int(t & 3);
    const int n = s * H + h;
    constexpr int PF = hc_pack_factor<BITS_U>();
    constexpr int BP = hc_bytes_per_pack<BITS_U>();
    constexpr int ROW_BYTES = R * BP / PF;
    constexpr int GROUPS_R = R / GS_U;
    constexpr int CH = 2 * PF;
    constexpr int CPG = GS_U / CH;
    const device uint8_t* w = (const device uint8_t*)up_w + (size_t)n * ROW_BYTES;
    const device T* sp = up_s + (size_t)n * GROUPS_R;
    const device T* bp = up_b + (size_t)n * GROUPS_R;
    float xv[CH];
    const int r = threadgroup_position_in_grid.z;
    const device T* a = act + (size_t)r * R;
    float acc = 0.0f;
    for (int gq = 0; gq < GROUPS_R; ++gq) {
        const float sc = float(sp[gq]);
        const float bi = float(bp[gq]);
        for (int c = 0; c < CPG; ++c) {
            const int e = gq * GS_U + c * CH;
            float sum = hc_load_vector<T, CH, BITS_U>(a + e, xv);
            acc += hc_qdot<CH, BITS_U>(w + e * BP / PF, xv, sc, bi, sum);
        }
    }
    float gate = 1.0f / (1.0f + metal::exp(-acc));
    float v = gate * float(xn[(size_t)r * K + n]);
    v += simd_shuffle_down(v, 1);
    v += simd_shuffle_down(v, 2);
    if (s == 0) mixed[(size_t)r * H + h] = T(v / float(HC));
"""

# Two-launch decode, bit-identical to _N_SOURCE/_WND_SOURCE -> _D_SOURCE -> _U_SOURCE.
# A dependent launch costs ~2 us of GPU time on its own, and the three-launch
# kernels ran on 4, 41 and 40 threadgroups of an 80-core GPU. With four streams
# the down projection's K slice ks is stream ks, so the stream norm folds into
# the down launch: one 256-thread threadgroup per (row group g, slice ks)
# normalizes only its own stream. Device stores come last in both kernels: a
# device load after a store the compiler cannot prove disjoint waits for it.
#
# _NDN_SOURCE: threadgroup (g, ks) normalizes stream ks (and applies the pending
# write) exactly like the norm kernels (element k = t + i * 256, simd_sum, then
# the eight partials in order) into threadgroup memory. Simdgroup sg then runs the
# _D_SOURCE lanes, block order and simd_sum of down rows g * 8 * RPS + sg * RPS + r
# over slice ks (RPS only sets how many rows share one pass over x); the extra row
# group g = R / (8 * RPS) runs the inject rows. Each simd_sum goes to
# parts[row][ks][out], and threadgroup (g, ks) stores strip i = g of stream ks.
_NDN_SOURCE_TEMPLATE = r"""
    const uint gy = threadgroup_position_in_grid.y;
    const uint row = threadgroup_position_in_grid.z;
    const uint t = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint s = gy & 3;
    const uint g = gy >> 2;
    constexpr int PER = (H + 255) / 256;
    constexpr int KS = 4;
    constexpr int SLICE = K / KS;
    constexpr int GROUPS = K / GS_D;
    constexpr int GROUPS_I = K / GS_I;
    constexpr int PR = R + HC;
    constexpr int ROWS_TG = 8 * RPS;
    constexpr int NDG = R / ROWS_TG;
    static_assert(HC == 4 && K == 4 * H && R % ROWS_TG == 0, "down K slice ks is stream ks");
    static_assert(PER <= NDG + INJ, "one stored strip per threadgroup");
    threadgroup T xs[H];
    threadgroup float npart[8];
    float v[PER];
    T y_mine = T(0.0f);
    T n_mine = T(0.0f);
    const size_t base = (size_t)row * K + (size_t)s * H;
    // The norm weights do not depend on the sum of squares: load them with x.
    const device T* wp = w + (size_t)s * H;
    float wv[PER];
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        wv[i] = (k < H) ? float(wp[k]) : 0.0f;
    }
    const float e = eps[0];
    float ss = 0.0f;
    // LOAD
    ss = simd_sum(ss);
    if (lane == 0) npart[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += npart[i];
    const float inv = metal::rsqrt(tot / float(H) + e);
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        if (k < H) {
            const T y = T(v[i] * inv * (1.0f + wv[i]));
            xs[k] = y;
            if (i == int(g)) n_mine = y;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool down_tg = g < NDG;
    const int ks = int(s);
    device float* pp = parts + ((size_t)row * KS + ks) * PR;
    if (down_tg) {
        constexpr int PF = hc_pack_factor<BITS_D>();
        constexpr int BP = hc_bytes_per_pack<BITS_D>();
        constexpr int ROW_BYTES = K * BP / PF;
        constexpr int PPT = 2;
        constexpr int VPT = PF * PPT;
        constexpr int BLOCK = VPT * 32;
        constexpr int SCALE_STEP = GS_D / VPT;
        const int out_row = int(g) * ROWS_TG + int(sg) * RPS;
        const device uint8_t* wq = (const device uint8_t*)down_w
            + out_row * ROW_BYTES + ks * (SLICE * BP / PF) + int(lane) * PPT * BP;
        const device T* sp = down_s + out_row * GROUPS + ks * (SLICE / GS_D)
            + int(lane) / SCALE_STEP;
        const device T* bp = down_b + out_row * GROUPS + ks * (SLICE / GS_D)
            + int(lane) / SCALE_STEP;
        const threadgroup T* xp = xs + int(lane) * VPT;
        float result[RPS] = {0.0f};
        float xv[VPT];
        constexpr int TAIL = SLICE % BLOCK;
        for (int k = 0; k < SLICE - TAIL; k += BLOCK) {
            float sum = hc_load_vector_tg<T, VPT, BITS_D>(xp, xv);
            for (int r = 0; r < RPS; ++r) {
                result[r] += hc_qdot<VPT, BITS_D>(
                    wq + r * ROW_BYTES, xv,
                    float(sp[r * GROUPS]), float(bp[r * GROUPS]), sum);
            }
            wq += BLOCK * BP / PF;
            sp += BLOCK / GS_D;
            bp += BLOCK / GS_D;
            xp += BLOCK;
        }
        // Load only lanes inside the tail; all lanes must join simd_sum.
        if (TAIL > 0 && int(lane) * VPT < TAIL) {
            float sum = hc_load_vector_tg<T, VPT, BITS_D>(xp, xv);
            for (int r = 0; r < RPS; ++r) {
                result[r] += hc_qdot<VPT, BITS_D>(
                    wq + r * ROW_BYTES, xv,
                    float(sp[r * GROUPS]), float(bp[r * GROUPS]), sum);
            }
        }
        for (int r = 0; r < RPS; ++r) {
            float v = simd_sum(result[r]);
            if (lane == 0) pp[out_row + r] = v;
        }
    } else if (INJ != 0 && int(sg) < HC) {
        constexpr int PF = hc_pack_factor<BITS_I>();
        constexpr int BP = hc_bytes_per_pack<BITS_I>();
        constexpr int ROW_BYTES = K * BP / PF;
        constexpr int VPT = PF;
        constexpr int BLOCK = VPT * 32;
        constexpr int SCALE_STEP = GS_I / VPT;
        const int r = int(sg);
        const device uint8_t* wq = (const device uint8_t*)inject_w
            + r * ROW_BYTES + ks * (SLICE * BP / PF) + int(lane) * BP;
        const device T* sp = inject_s + r * GROUPS_I + ks * (SLICE / GS_I)
            + int(lane) / SCALE_STEP;
        const device T* bp = inject_b + r * GROUPS_I + ks * (SLICE / GS_I)
            + int(lane) / SCALE_STEP;
        const threadgroup T* xp = xs + int(lane) * VPT;
        float result = 0.0f;
        float xv[VPT];
        constexpr int TAIL = SLICE % BLOCK;
        for (int k = 0; k < SLICE - TAIL; k += BLOCK) {
            float sum = hc_load_vector_tg<T, VPT, BITS_I>(xp, xv);
            result += hc_qdot<VPT, BITS_I>(wq, xv, float(sp[0]), float(bp[0]), sum);
            wq += BLOCK * BP / PF;
            sp += BLOCK / GS_I;
            bp += BLOCK / GS_I;
            xp += BLOCK;
        }
        if (TAIL > 0 && int(lane) * VPT < TAIL) {
            float sum = hc_load_vector_tg<T, VPT, BITS_I>(xp, xv);
            result += hc_qdot<VPT, BITS_I>(wq, xv, float(sp[0]), float(bp[0]), sum);
        }
        float v = simd_sum(result);
        if (lane == 0) pp[R + r] = v;
    }
    if (int(g) < PER && int(t) + int(g) * 256 < H) {
        const size_t at = base + int(t) + int(g) * 256;
        xn[at] = n_mine;
        // WRITE
    }
"""

_NDN_LOAD = r"""
    const device T* xp = x + base;
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        v[i] = (k < H) ? float(xp[k]) : 0.0f;
        ss += v[i] * v[i];
    }
"""

_NDN_WRITE_LOAD = r"""
    const device T* bq = branch + (size_t)row * H;
    const float g_s = float(gate[(size_t)row * HC + s]);
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        v[i] = 0.0f;
        if (k < H) {
            const T y = T(float(x[base + k]) + float(T(float(bq[k]) * g_s)));
            if (i == int(g)) y_mine = y;
            v[i] = float(y);
        }
        ss += v[i] * v[i];
    }
"""


def _ndn_source(load: str, write: bool) -> str:
    return _NDN_SOURCE_TEMPLATE.replace("    // LOAD\n", load).replace(
        "        // WRITE\n", "        y_out[at] = y_mine;\n" if write else ""
    )


_NDN_SOURCE = _ndn_source(_NDN_LOAD, write=False)
_NDNW_SOURCE = _ndn_source(_NDN_WRITE_LOAD, write=True)

# _U2_SOURCE: threadgroup y owns up rows j = s * HB + hh (n = s * H + y * HB + hh).
# Every threadgroup combines the slice sums as _D_SOURCE does (part[0] + part[1]
# + part[2] + part[3], then / HC and the BF16-rounded silu; 2 * sigmoid for the
# inject rows in threadgroup 0) into threadgroup memory. Thread t then owns chunk
# c = t % NC of rows j = t / NC, t / NC + TGS / NC, ... and stages each hc_qdot;
# one thread per output adds its NC chunks in chunk order from 0.0f, and the
# sigmoid gate / shuffle-down stream mix is _U_SOURCE's.
_U2_SOURCE = r"""
    constexpr int PF = hc_pack_factor<BITS_U>();
    constexpr int BP = hc_bytes_per_pack<BITS_U>();
    constexpr int ROW_BYTES = R * BP / PF;
    constexpr int GROUPS_R = R / GS_U;
    constexpr int CH = 2 * PF;
    constexpr int CPG = GS_U / CH;
    constexpr int NC = R / CH;
    constexpr int NCP = NC + 1;
    constexpr int HB = NO / HC;
    constexpr int JS = TGS / NC;
    constexpr int KS = 4;
    constexpr int PR = R + HC;
    static_assert(HC == 4 && NO % 32 == 0 && TGS % NC == 0 && JS > 0,
                  "up tiles: four streams, whole simdgroups, whole chunk rows");
    threadgroup T acts[R];
    threadgroup float q[NO * NCP];
    const uint t = thread_index_in_threadgroup;
    const int r = threadgroup_position_in_grid.z;
    const int h0 = int(threadgroup_position_in_grid.y) * HB;
    const int s = int(t & 3);
    const int h = h0 + int(t >> 2);
    const int n = s * H + h;
    // Epilogue threads fetch their normed element before the weights.
    const float xnv = int(t) < NO ? float(xn[(size_t)r * K + n]) : 0.0f;
    const device float* pp = parts + (size_t)r * KS * PR;
    for (int i = int(t); i < R; i += TGS) {
        float v = pp[i] + pp[PR + i] + pp[2 * PR + i] + pp[3 * PR + i];
        v = v / float(HC);
        acts[i] = T(v / (1.0f + metal::exp(-v)));
    }
    const bool inj_thread = INJ != 0 && threadgroup_position_in_grid.y == 0 && int(t) < HC;
    T injv = T(0.0f);
    if (inj_thread) {
        const int i = R + int(t);
        float v = pp[i] + pp[PR + i] + pp[2 * PR + i] + pp[3 * PR + i];
        v = v / float(HC);
        injv = T(2.0f / (1.0f + metal::exp(-v)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        const int c = int(t) % NC;
        const int gq = c / CPG;
        const int e = c * CH;
        float xv[CH];
        float sum = hc_load_vector_tg<T, CH, BITS_U>(acts + e, xv);
        for (int j = int(t) / NC; j < NO; j += JS) {
            const int nj = (j / HB) * H + h0 + j % HB;
            const device uint8_t* w = (const device uint8_t*)up_w + (size_t)nj * ROW_BYTES;
            const float sc = float(up_s[(size_t)nj * GROUPS_R + gq]);
            const float bi = float(up_b[(size_t)nj * GROUPS_R + gq]);
            q[j * NCP + c] = hc_qdot<CH, BITS_U>(w + e * BP / PF, xv, sc, bi, sum);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(t) >= NO) return;
    const threadgroup float* qp = q + (s * HB + int(t >> 2)) * NCP;
    float acc = 0.0f;
    for (int c = 0; c < NC; ++c) {
        acc += qp[c];
    }
    float gate = 1.0f / (1.0f + metal::exp(-acc));
    float v = gate * xnv;
    v += simd_shuffle_down(v, 1);
    v += simd_shuffle_down(v, 2);
    if (s == 0) mixed[(size_t)r * H + h] = T(v / float(HC));
    if (inj_thread) inj[(size_t)r * HC + int(t)] = injv;
"""


# Prefill epilogue for one row per threadgroup:
# - mixed = mean over streams of sigmoid(up) * normed, with BF16 rounding after
#   each gate, product and partial sum.
# - inj = 2 * sigmoid(normed . inject / HC) from the quantized four-row bank.
# The normed row is read once: each stream is staged in threadgroup memory and
# its inject dot runs from there. The inject sum keeps one fixed order: virtual
# thread v = s * VT + t (t < VT) sums elements [v * PER, (v + 1) * PER) of the
# row, then simd_sum per 32 virtual threads, then the 8 partials in order.
_TI_SOURCE = r"""
    const uint row = threadgroup_position_in_grid.z;
    const uint t = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    constexpr int PER_H = (H + 255) / 256;
    constexpr int PF = hc_pack_factor<BITS_I>();
    constexpr int BP = hc_bytes_per_pack<BITS_I>();
    constexpr int ROW_BYTES = K * BP / PF;
    constexpr int GROUPS = K / GS_I;
    constexpr int PER = K / 256;
    constexpr int VT = H / PER;
    static_assert(K == HC * H && H % PER == 0 && VT % 32 == 0 && HC * VT == 256,
                  "inject chunks must tile each stream");
    threadgroup T xs[H];
    threadgroup float part[HC][8];
    const device T* up_r = up + (size_t)row * K;
    const device T* xn_r = xn + (size_t)row * K;
    float acc[PER_H];
    for (int s = 0; s < HC; ++s) {
        for (int i = 0; i < PER_H; ++i) {
            const int h = int(t) + i * 256;
            if (h < H) {
                const int n = s * H + h;
                const T x = xn_r[n];
                xs[h] = x;
                const T g = T(1.0f / (1.0f + metal::exp(-float(up_r[n]))));
                const T p = T(float(g) * float(x));
                acc[i] = s == 0 ? float(p) : float(T(acc[i] + float(p)));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (int(t) < VT) {
            float res[HC] = {0.0f};
            float xv[PF];
            const int e0 = int(t) * PER;
            for (int e = e0; e < e0 + PER; e += PF) {
                const float sum = hc_load_vector_tg<T, PF, BITS_I>(xs + e, xv);
                const int ge = s * H + e;
                const int g = ge / GS_I;
                for (int r = 0; r < HC; ++r) {
                    res[r] += hc_qdot<PF, BITS_I>(
                        (const device uint8_t*)inject_w + r * ROW_BYTES + ge * BP / PF,
                        xv, float(inject_s[r * GROUPS + g]), float(inject_b[r * GROUPS + g]),
                        sum);
                }
            }
            for (int r = 0; r < HC; ++r) {
                const float v = simd_sum(res[r]);
                if (lane == 0) part[r][s * (VT / 32) + int(sg)] = v;
            }
        }
        // Stream s + 1 overwrites xs only after every inject read of stream s.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int i = 0; i < PER_H; ++i) {
        const int h = int(t) + i * 256;
        if (h < H) mixed[(size_t)row * H + h] = T(acc[i] * (1.0f / float(HC)));
    }
    if (t < HC) {
        float v = 0.0f;
        for (int i = 0; i < 8; ++i) v += part[t][i];
        const float q = float(T(float(T(v)) / float(HC)));
        const float gate = float(T(1.0f / (1.0f + metal::exp(-q))));
        inj[(size_t)row * HC + t] = T(2.0f * gate);
    }
"""

# hc_load_vector for a threadgroup-staged vector; same arithmetic as the device form.
_TG_HEADER = r"""
template <typename T, int N, int bits>
inline float hc_load_vector_tg(const threadgroup T* x, thread float* xt) {
    float sum = 0.0f;
    if (bits == 4) {
        for (int i = 0; i < N; i += 4) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            xt[i] = x[i];
            xt[i + 1] = x[i + 1] / 16.0f;
            xt[i + 2] = x[i + 2] / 256.0f;
            xt[i + 3] = x[i + 3] / 4096.0f;
        }
    } else if (bits == 5) {
        for (int i = 0; i < N; i += 8) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3]
                + x[i + 4] + x[i + 5] + x[i + 6] + x[i + 7];
            xt[i] = x[i];
            xt[i + 1] = x[i + 1] / 32.0f;
            xt[i + 2] = x[i + 2] / 4.0f;
            xt[i + 3] = x[i + 3] / 128.0f;
            xt[i + 4] = x[i + 4] / 16.0f;
            xt[i + 5] = x[i + 5] / 2.0f;
            xt[i + 6] = x[i + 6] / 64.0f;
            xt[i + 7] = x[i + 7] / 8.0f;
        }
    } else if (bits == 6) {
        for (int i = 0; i < N; i += 4) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            xt[i] = x[i];
            xt[i + 1] = x[i + 1] / 64.0f;
            xt[i + 2] = x[i + 2] / 16.0f;
            xt[i + 3] = x[i + 3] / 4.0f;
        }
    } else if (bits == 8) {
        for (int i = 0; i < N; ++i) {
            sum += x[i];
            xt[i] = x[i];
        }
    }
    return sum;
}
"""

_V2_HEADER = _HEADER + _TG_HEADER


def _kernel(
    name: str,
    input_names: list[str],
    output_names: list[str],
    source: str,
    header: str = "",
):
    kernel = _KERNELS.get(name)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=name,
            input_names=input_names,
            output_names=output_names,
            header=header,
            source=source,
            ensure_row_contiguous=True,
        )
        _KERNELS[name] = kernel
    return kernel


def enabled() -> bool:
    return not _DISABLED


def write_enabled() -> bool:
    """Whether residual writes may be deferred into the next fused stream norm."""
    return not (_DISABLED or _WRITE_DISABLED)


def _quantized_ok(projection) -> bool:
    return (
        type(projection) is nn.QuantizedLinear
        and getattr(projection, "group_size", None) in _GROUP_SIZES
        and getattr(projection, "bits", None) in _SUPPORTED_BITS
        and getattr(projection, "mode", "affine") == "affine"
        and "bias" not in projection
        and isinstance(getattr(projection, "weight", None), mx.array)
        and projection.weight.dtype == mx.uint32
        and isinstance(getattr(projection, "scales", None), mx.array)
        and isinstance(getattr(projection, "biases", None), mx.array)
        and projection.scales.dtype == mx.bfloat16
        and projection.biases.dtype == mx.bfloat16
    )


def _ineligible(reason: str) -> bool:
    """Log the first unsupported model layout per process."""
    global _INELIGIBLE_LOGGED
    if not _INELIGIBLE_LOGGED:
        _INELIGIBLE_LOGGED = True
        logger.info(
            "Qwen4 fused hyper-connection kernels not used for this model (%s); "
            "the canonical path stays in effect",
            reason,
        )
    return False


def _rows_of(hyper_input) -> int | None:
    if not (
        isinstance(hyper_input, mx.array)
        and hyper_input.ndim == 3
        and hyper_input.dtype == mx.bfloat16
    ):
        return None
    return hyper_input.shape[0] * hyper_input.shape[1]


def compatible(module, hyper_input) -> bool:
    """Whether ``module`` (a Qwen4ExpGatedResidual) can take the fused path for ``hyper_input``."""
    # Runs for every hyper-connection of every decode step: the row gate
    # first (prefill rows must not reach the layout checks), then the
    # cached layout verdict.
    if _DISABLED or not isinstance(hyper_input, mx.array):
        return False
    shape = hyper_input.shape
    return (
        len(shape) == 3
        and hyper_input.dtype == mx.bfloat16
        and 1 <= shape[0] * shape[1] <= MAX_ROWS
        and _layout_compatible(module, hyper_input)
    )


def prefill_compatible(module, hyper_input) -> bool:
    """Whether rows above MAX_ROWS can use the compiled prefill mean."""
    rows = _rows_of(hyper_input)
    return (
        enabled()
        and rows is not None
        and rows > MAX_ROWS
        and _layout_compatible(module, hyper_input)
    )


def _layout_compatible(module, hyper_input) -> bool:
    if not _static_layout_compatible(module):
        return False
    if hyper_input.shape[2] != module.hc_count * module.hidden_size:
        return _ineligible(f"input width {hyper_input.shape[2]}")
    return mx.default_device() == mx.gpu and mx.metal.is_available()


def _static_layout_compatible(module) -> bool:
    """Cached per module; any reassigned child, weight, scale or bias re-checks.

    The key is the identity of every value of the module and of its child
    modules, compared with ``is`` (``id()`` raises an audit event per call).
    The cache keeps those objects alive so a match cannot come from a new
    object reusing a freed one's address.
    """
    cached = module.__dict__.get("_omlx_hc_layout")
    if cached is not None:
        children, refs, ok = cached
        current = tuple(chain(dict.values(module), *map(dict.values, children)))
        if len(current) == len(refs) and all(map(is_, current, refs)):
            return ok
    ok = _check_static_layout(module)
    children = [value for value in dict.values(module) if isinstance(value, nn.Module)]
    refs = tuple(chain(dict.values(module), *map(dict.values, children)))
    module.__dict__["_omlx_hc_layout"] = (children, refs, ok)
    return ok


def _check_static_layout(module) -> bool:
    if hasattr(module, "input_inject_weight"):
        return _ineligible("combined input projection layout")
    hc_count = getattr(module, "hc_count", None)
    hidden = getattr(module, "hidden_size", None)
    lowrank = getattr(module, "hc_lowrank", None)
    if not (
        isinstance(hc_count, int)
        and isinstance(hidden, int)
        and isinstance(lowrank, int)
        and hc_count == 4
        # The up grid and quantization groups require 64-element alignment.
        and hidden % 64 == 0
        and lowrank % 64 == 0
    ):
        return _ineligible(
            f"geometry hidden_size={hidden} hc_lowrank={lowrank} hc_count={hc_count}"
        )
    norm = getattr(module, "hc_norm", None)
    if not (
        norm is not None
        and getattr(norm, "group_size", None) == hidden
        and isinstance(getattr(norm, "weight", None), mx.array)
        and norm.weight.dtype == mx.bfloat16
        and norm.weight.shape == (hc_count * hidden,)
    ):
        return _ineligible("hc_norm layout")
    if not (
        _quantized_ok(getattr(module, "input_mix_weight_down", None))
        and _quantized_ok(getattr(module, "input_mix_weight_up", None))
    ):
        return _ineligible(
            "projection quantisation (need affine group-size-32/64 4/5/6/8-bit with bf16 scales)"
        )
    if "block_inject_weight" in module and not _quantized_ok(
        module.block_inject_weight
    ):
        return _ineligible("block_inject_weight quantisation")
    return True


def _eps_array(module) -> mx.array:
    eps = getattr(module, "_omlx_hc_fused_eps", None)
    if eps is None:
        eps = mx.array([float(module.hc_norm.eps)], dtype=mx.float32)
        mx.eval(eps)
        module._omlx_hc_fused_eps = eps
    return eps


# Prefill keeps Qwen4ExpRMSNorm's precise rsqrt; only the f32 sum order differs.
_NP_SOURCE = _N_SOURCE.replace("metal::rsqrt", "metal::precise::rsqrt")

# The previous block's residual write fused into the stream norm:
# - y = T(x + T(branch * gate[s])) rounds like the eager multiply and add.
# - y is stored as the new residual stream and normalized as in _NP_SOURCE
#   (prefill) or _N_SOURCE (decode, _WND_SOURCE below).
_WN_SOURCE = r"""
    const uint row = threadgroup_position_in_grid.z;
    const uint s = threadgroup_position_in_grid.y;
    const uint t = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    threadgroup float part[8];
    constexpr int PER = (H + 255) / 256;
    const size_t base = (size_t)row * K + (size_t)s * H;
    const device T* bp = branch + (size_t)row * H;
    const float g = float(gate[(size_t)row * HC + s]);
    const device T* wp = w + (size_t)s * H;
    float v[PER];
    float ss = 0.0f;
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        v[i] = 0.0f;
        if (k < H) {
            const T y = T(float(x[base + k]) + float(T(float(bp[k]) * g)));
            y_out[base + k] = y;
            v[i] = float(y);
        }
        ss += v[i] * v[i];
    }
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += part[i];
    const float inv = metal::precise::rsqrt(tot / float(H) + eps[0]);
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        if (k < H) xn[base + k] = T(v[i] * inv * (1.0f + float(wp[k])));
    }
"""

# Decode keeps _N_SOURCE's fast rsqrt so rows match the unfused decode norm.
_WND_SOURCE = _WN_SOURCE.replace("metal::precise::rsqrt", "metal::rsqrt")


def _kernel_norm(module, flat, rows, hc, hidden, dtype, precise=False):
    width = hc * hidden
    name = "omlx_qwen4_hc_prefill_norm" if precise else "omlx_qwen4_hc_fused_norm"
    return _kernel(
        name, ["x", "w", "eps"], ["xn"], _NP_SOURCE if precise else _N_SOURCE
    )(
        inputs=[flat, module.hc_norm.weight, _eps_array(module)],
        template=[("T", dtype), ("K", width), ("H", hidden)],
        grid=(256, hc, rows),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, width)],
        output_dtypes=[dtype],
    )[0]


def _write_operands(hyper_input, write, hidden, hc):
    """Flattened (branch, gate) for the write-norm kernels, or None if they cannot take it.

    Mixed dtypes would promote the eager write, so only same-dtype operands qualify.
    """
    branch, gate = write
    lead = hyper_input.shape[:-1]
    dtype = hyper_input.dtype
    if not (
        isinstance(branch, mx.array)
        and isinstance(gate, mx.array)
        and branch.shape == (*lead, hidden)
        and gate.shape == (*lead, hc)
        and branch.dtype == dtype
        and gate.dtype == dtype
    ):
        return None
    rows = _rows_of(hyper_input)
    return branch.reshape(rows, hidden), gate.reshape(rows, hc)


def _kernel_write_norm(
    module, flat, branch, gate, rows, hc, hidden, dtype, precise=True
):
    width = hc * hidden
    return _kernel(
        "omlx_qwen4_hc_prefill_write_norm"
        if precise
        else "omlx_qwen4_hc_fused_write_norm",
        ["x", "branch", "gate", "w", "eps"],
        ["y_out", "xn"],
        _WN_SOURCE if precise else _WND_SOURCE,
    )(
        inputs=[flat, branch, gate, module.hc_norm.weight, _eps_array(module)],
        template=[("T", dtype), ("K", width), ("H", hidden), ("HC", hc)],
        grid=(256, hc, rows),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, width), (rows, width)],
        output_dtypes=[dtype, dtype],
    )


def _pack_factor(bits: int) -> int:
    return 8 if bits == 5 else (4 if bits == 6 else 32 // bits)


# Two-launch decode geometry, tuned on an 80-core M5 Ultra; none of it changes an
# output bit. Down rows per simdgroup: 1 for one or two rows (more threadgroups),
# 2 above. From 7 rows on, the per-row threadgroups re-read the weights more
# often than the three-launch kernels, which are then as fast or faster.
_V2_MAX_ROWS = 6
_V2_UP_OUTPUTS = 32
_V2_UP_THREADS = 320
_TG_MEMORY_BYTES = 32768


def _up_chunks(bits: int, lowrank: int) -> int:
    return lowrank // (2 * _pack_factor(bits))


def _down_rps(rows: int) -> int:
    return 1 if rows <= 2 else 2


def _down_row_groups(lowrank: int, inject: bool, rps: int) -> int:
    return lowrank // (8 * rps) + int(inject)


def _decode_v2(rows: int, hc: int, hidden: int, lowrank: int, up_bits: int, inject: bool) -> bool:
    """Whether the two-launch decode kernels apply: four streams, one stored
    256-element strip per norm/down threadgroup, and each kernel's threadgroup
    arrays within threadgroup memory."""
    rps = _down_rps(rows)
    chunks = _up_chunks(up_bits, lowrank)
    return (
        _DECODE_V2
        and rows <= _V2_MAX_ROWS
        and hc == 4
        and lowrank % (8 * rps) == 0
        and -(-hidden // 256) <= _down_row_groups(lowrank, inject, rps)
        and hidden * 2 + 32 <= _TG_MEMORY_BYTES
        and lowrank * 2 + _V2_UP_OUTPUTS * (chunks + 1) * 4 <= _TG_MEMORY_BYTES
        and chunks <= _V2_UP_THREADS
    )


def _kernel_norm_down(module, flat, operands, rows, hc, hidden, lowrank, dtype, down, inject):
    """(written or None, normed, parts): the stream norm (with the pending write)
    and the per-slice down/inject simd sums, parts[row][slice][R + HC], in one launch."""
    width = hc * hidden
    tensors = inject if inject is not None else down
    inputs = [
        module["hc_norm"]["weight"],
        _eps_array(module),
        down["weight"],
        down["scales"],
        down["biases"],
        tensors["weight"],
        tensors["scales"],
        tensors["biases"],
    ]
    names = ["w", "eps", "down_w", "down_s", "down_b", "inject_w", "inject_s", "inject_b"]
    rps = _down_rps(rows)
    launch = dict(
        template=[
            ("T", dtype),
            ("BITS_D", down.bits),
            ("BITS_I", tensors.bits),
            ("GS_D", down.group_size),
            ("GS_I", tensors.group_size),
            ("K", width),
            ("H", hidden),
            ("R", lowrank),
            ("HC", hc),
            ("INJ", 1 if inject is not None else 0),
            ("RPS", rps),
        ],
        grid=(256, hc * _down_row_groups(lowrank, inject is not None, rps), rows),
        threadgroup=(256, 1, 1),
    )
    shapes = [(rows, width), (rows, hc, lowrank + hc)]
    types = [dtype, mx.float32]
    if operands is None:
        normed, parts = _kernel(
            "omlx_qwen4_hc_decode_norm_down",
            ["x", *names],
            ["xn", "parts"],
            _NDN_SOURCE,
            header=_V2_HEADER,
        )(inputs=[flat, *inputs], output_shapes=shapes, output_dtypes=types, **launch)
        return None, normed, parts
    written, normed, parts = _kernel(
        "omlx_qwen4_hc_decode_write_norm_down",
        ["x", "branch", "gate", *names],
        ["y_out", "xn", "parts"],
        _NDNW_SOURCE,
        header=_V2_HEADER,
    )(
        inputs=[flat, *operands, *inputs],
        output_shapes=[(rows, width), *shapes],
        output_dtypes=[dtype, *types],
        **launch,
    )
    return written, normed, parts


def _kernel_up_mix(normed, parts, rows, hc, hidden, lowrank, dtype, up, inject: bool):
    """(mixed, inj): slice-sum combine and activations, then the up projection and stream mix."""
    chunks = _up_chunks(up.bits, lowrank)
    threads = chunks * (_V2_UP_THREADS // chunks)
    return _kernel(
        "omlx_qwen4_hc_decode_up",
        ["xn", "parts", "up_w", "up_s", "up_b"],
        ["mixed", "inj"],
        _U2_SOURCE,
        header=_V2_HEADER,
    )(
        inputs=[normed, parts, up["weight"], up["scales"], up["biases"]],
        template=[
            ("T", dtype),
            ("BITS_U", up.bits),
            ("GS_U", up.group_size),
            ("K", hc * hidden),
            ("R", lowrank),
            ("HC", hc),
            ("H", hidden),
            ("INJ", int(inject)),
            ("NO", _V2_UP_OUTPUTS),
            ("TGS", threads),
        ],
        grid=(threads, hc * hidden // _V2_UP_OUTPUTS, rows),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows, hidden), (rows, hc)],
        output_dtypes=[dtype, dtype],
    )


_TAILS: dict[tuple[int, int], object] = {}
_ACTS: dict[int, object] = {}


def _act(hc: int):
    """Compiled silu(down / hc): one launch that rounds like the two eager ops."""
    fn = _ACTS.get(hc)
    if fn is None:
        fn = mx.compile(lambda y: nn.silu(y / hc), shapeless=True)
        _ACTS[hc] = fn
    return fn


def _tail(hc: int, hidden: int):
    """Compiled mean over the streams as slice products: one fused pass instead of a strided reduce."""
    fn = _TAILS.get((hc, hidden))
    if fn is None:

        def tail(up, normed):
            gate = mx.sigmoid(up)
            acc = gate[..., :hidden] * normed[..., :hidden]
            for g in range(1, hc):
                lo = g * hidden
                acc = acc + gate[..., lo : lo + hidden] * normed[..., lo : lo + hidden]
            return acc * (1.0 / hc)

        fn = mx.compile(tail)
        _TAILS[(hc, hidden)] = fn
    return fn


def _nax_available() -> bool:
    global _NAX_AVAILABLE
    if _NAX_AVAILABLE is None:
        try:
            from omlx.custom_kernels.nax import is_nax_available

            _NAX_AVAILABLE = bool(is_nax_available())
        except Exception:  # noqa: BLE001 - no tensor units: keep MLX's kernels
            _NAX_AVAILABLE = False
    return _NAX_AVAILABLE


def _nax_prefill_ok(module, rows: int, width: int, inject) -> bool:
    """Whether the tensor-unit kernels reproduce this call bit for bit.

    They repeat MLX's unsplit NAX quantized matmul, so the up projection must
    take that kernel in MLX too, every projection must use the tile loop's
    group size, and the inject partials need the tail kernel's whole-pack
    thread slices.
    """
    return (
        not _NAX_DISABLED
        and not _NAX_BROKEN
        and inject is not None
        and width % (256 * _pack_factor(inject.bits)) == 0
        and all(
            projection.group_size == hc_prefill_nax.GROUP_SIZE
            for projection in (
                module.input_mix_weight_down,
                module.input_mix_weight_up,
                inject,
            )
        )
        and module.hidden_size % hc_prefill_nax.UP_COLS == 0
        and hc_prefill_nax.plain_qmm_nax(rows, width)
        and _nax_available()
    )


def _nax_prefill(module, hyper_input, flat, write, rows, hc, hidden, dtype, inject):
    """Three-dispatch prefill on the tensor units; None to use the MLX path."""
    global _NAX_BROKEN
    lead = hyper_input.shape[:-1]
    pending = None
    if write is not None:
        pending = _write_operands(hyper_input, write, hidden, hc)
        if pending is None:
            return None
    down, up = module.input_mix_weight_down, module.input_mix_weight_up
    try:
        written, normed, inj_part = hc_prefill_nax.norm_inject(
            module, flat, pending, rows, hc, hidden, dtype, _eps_array(module), inject
        )
        fused_down = hc_prefill_nax.plain_qmm_nax(rows, module.hc_lowrank)
        if fused_down:
            act = hc_prefill_nax.down_silu(normed, down, rows, hc)
        else:
            # MLX splits K for this few row tiles: keep its kernel.
            act = nn.silu(down(normed) / hc)
        mixed, injection = hc_prefill_nax.up_tail(
            act, up, normed, inj_part, rows, hc, hidden
        )
        passthrough = (
            hyper_input if written is None else written.reshape(hyper_input.shape)
        )
        out = (
            mixed.reshape(*lead, hidden),
            passthrough,
            injection.reshape(*lead, hc),
        )
        signature = (
            "prefill_nax",
            dtype,
            hc,
            hidden,
            module.hc_lowrank,
            down.bits,
            up.bits,
            inject.bits,
            write is not None,
            fused_down,
        )
        if signature not in _VALIDATED:
            # The kernels repeat MLX's quantized matmul arithmetic: check the
            # first call of each specialization against the MLX path bit for
            # bit (an MLX with a different qmm would fail here) and keep the
            # MLX path for good if they differ.
            reference = prefill_forward(module, hyper_input, write, nax=False)
            mx.eval(out, reference)
            if not all(
                mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item()
                for a, b in zip(out, reference)
            ):
                _NAX_BROKEN = True
                logger.warning(
                    "Qwen4 tensor-unit hyper-connection prefill kernels differ "
                    "from this MLX's quantized matmul; using the MLX matmul path"
                )
                return reference
            _VALIDATED.add(signature)
    except Exception as exc:  # noqa: BLE001 - optional native path
        _NAX_BROKEN = True
        logger.warning(
            "Qwen4 tensor-unit hyper-connection prefill kernels failed; using "
            "the MLX matmul path: %s",
            exc,
        )
        return None
    return out


def prefill_forward(module, hyper_input, write=None, nax=True):
    """Prefill with the fused stream norm and tail/inject epilogue; None on failure.

    ``write`` is a pending ``(branch, gate)`` residual write onto ``hyper_input``.
    The norm kernel applies it and returns the written stream as the passthrough;
    a module without inject weights then returns ``(mixed, written)``. With
    ``nax`` the projections run in hc_prefill_nax's kernels where they
    reproduce this path bit for bit.
    """
    global _FAILURE_LOGGED
    try:
        hc, hidden = module.hc_count, module.hidden_size
        width = hc * hidden
        dtype = hyper_input.dtype
        rows = _rows_of(hyper_input)
        flat = hyper_input.reshape(rows, width)
        inject = module.block_inject_weight if "block_inject_weight" in module else None
        if nax and _nax_prefill_ok(module, rows, width, inject):
            fused = _nax_prefill(
                module, hyper_input, flat, write, rows, hc, hidden, dtype, inject
            )
            if fused is not None:
                return fused
        if write is None:
            normed = _kernel_norm(module, flat, rows, hc, hidden, dtype, precise=True)
        else:
            operands = _write_operands(hyper_input, write, hidden, hc)
            if operands is None:
                return None
            written, normed = _kernel_write_norm(
                module, flat, *operands, rows, hc, hidden, dtype
            )
            hyper_input = written.reshape(hyper_input.shape)
        normed = normed.reshape(hyper_input.shape)
        mix = _act(hc)(module.input_mix_weight_down(normed))
        up = module.input_mix_weight_up(mix)
        if inject is None or width % (256 * _pack_factor(inject.bits)):
            # Each of the 256 threads dots whole packs of the inject row.
            mixed = _tail(hc, hidden)(up, normed)
            injection = None if inject is None else 2 * mx.sigmoid(inject(normed) / hc)
        else:
            mixed, injection = _kernel(
                "omlx_qwen4_hc_prefill_tail_inject",
                ["up", "xn", "inject_w", "inject_s", "inject_b"],
                ["mixed", "inj"],
                _TI_SOURCE,
                header=_HEADER + _TG_HEADER,
            )(
                inputs=[
                    up.reshape(rows, width),
                    normed.reshape(rows, width),
                    inject.weight,
                    inject.scales,
                    inject.biases,
                ],
                template=[
                    ("T", dtype),
                    ("BITS_I", inject.bits),
                    ("GS_I", inject.group_size),
                    ("K", width),
                    ("H", hidden),
                    ("HC", hc),
                ],
                grid=(256, 1, rows),
                threadgroup=(256, 1, 1),
                output_shapes=[(rows, hidden), (rows, hc)],
                output_dtypes=[dtype, dtype],
            )
            mixed = mixed.reshape(*hyper_input.shape[:-1], hidden)
            injection = injection.reshape(*hyper_input.shape[:-1], hc)
        signature = (
            "prefill",
            dtype,
            hc,
            hidden,
            module.hc_lowrank,
            module.input_mix_weight_down.bits,
            inject.bits if inject is not None else None,
            inject.group_size if inject is not None else None,
            write is not None,
        )
        if signature not in _VALIDATED:
            mx.eval(mixed) if injection is None else mx.eval(mixed, injection)
            _VALIDATED.add(signature)
        if injection is None:
            return mixed if write is None else (mixed, hyper_input)
        return mixed, hyper_input, injection
    except Exception as exc:  # noqa: BLE001 - optional native path
        if not _FAILURE_LOGGED:
            _FAILURE_LOGGED = True
            logger.warning(
                "Qwen4 fused hyper-connection kernels failed closed; using the "
                "canonical path: %s",
                exc,
            )
        return None


def fused_forward(module, hyper_input, write=None):
    """Return fused outputs, or None on construction or first-evaluation failure.

    ``write`` is a pending ``(branch, gate)`` residual write onto ``hyper_input``,
    applied inside the norm kernel as in :func:`prefill_forward`; the written
    stream is the passthrough, and a module without inject weights returns
    ``(mixed, written)``.
    """
    global _FAILURE_LOGGED
    try:
        hc, hidden, lowrank = module.hc_count, module.hidden_size, module.hc_lowrank
        width = hc * hidden
        batch, seq, _ = hyper_input.shape
        rows = batch * seq
        down, up = module["input_mix_weight_down"], module["input_mix_weight_up"]
        inject = module.get("block_inject_weight")
        flat = hyper_input.reshape(rows, width)
        dtype = hyper_input.dtype
        operands = None
        if write is not None:
            operands = _write_operands(hyper_input, write, hidden, hc)
            if operands is None:
                return None
        v2 = _decode_v2(rows, hc, hidden, lowrank, up.bits, inject is not None)
        if v2:
            written, normed, parts = _kernel_norm_down(
                module, flat, operands, rows, hc, hidden, lowrank, dtype, down, inject
            )
            mixed, injection = _kernel_up_mix(
                normed, parts, rows, hc, hidden, lowrank, dtype, up, inject is not None
            )
        else:
            written, normed = (
                (None, _kernel_norm(module, flat, rows, hc, hidden, dtype))
                if operands is None
                else _kernel_write_norm(
                    module, flat, *operands, rows, hc, hidden, dtype, precise=False
                )
            )
            if inject is not None:
                inject_tensors = (inject.weight, inject.scales, inject.biases)
            else:
                inject_tensors = (down.weight, down.scales, down.biases)
            act, injection = _kernel(
                "omlx_qwen4_hc_fused_down",
                ["xn", "down_w", "down_s", "down_b", "inject_w", "inject_s", "inject_b"],
                ["act", "inj"],
                _D_SOURCE,
                header=_HEADER,
            )(
                inputs=[normed, down.weight, down.scales, down.biases, *inject_tensors],
                template=[
                    ("T", dtype),
                    ("BITS_D", down.bits),
                    ("BITS_I", inject.bits if inject is not None else down.bits),
                    ("GS_D", down.group_size),
                    ("GS_I", inject.group_size if inject is not None else down.group_size),
                    ("K", width),
                    ("R", lowrank),
                    ("HC", hc),
                    ("INJ", 1 if inject is not None else 0),
                ],
                grid=(32, 8 * (lowrank // 8 + 1), rows),
                threadgroup=(32, 8, 1),
                output_shapes=[(rows, lowrank), (rows, hc)],
                output_dtypes=[dtype, dtype],
            )
            mixed = _kernel(
                "omlx_qwen4_hc_fused_up",
                ["xn", "act", "up_w", "up_s", "up_b"],
                ["mixed"],
                _U_SOURCE,
                header=_HEADER,
            )(
                inputs=[normed, act, up.weight, up.scales, up.biases],
                template=[
                    ("T", dtype),
                    ("BITS_U", up.bits),
                    ("GS_U", up.group_size),
                    ("K", width),
                    ("R", lowrank),
                    ("HC", hc),
                    ("H", hidden),
                ],
                grid=(256, hidden // 64, rows),
                threadgroup=(256, 1, 1),
                output_shapes=[(rows, hidden)],
                output_dtypes=[dtype],
            )[0]
        if written is not None:
            hyper_input = written.reshape(hyper_input.shape)
        signature = (
            dtype,
            hc,
            hidden,
            lowrank,
            rows,
            down.bits,
            up.bits,
            inject.bits if inject is not None else None,
            down.group_size,
            up.group_size,
            inject.group_size if inject is not None else None,
            write is not None,
            v2,
        )
        if signature not in _VALIDATED:
            # Metal compilation is lazy. Validate once, without synchronizing
            # subsequent layers or decode steps using the same specialization.
            if inject is None:
                mx.eval(mixed)
            else:
                mx.eval(mixed, injection)
            _VALIDATED.add(signature)
        mixed = mixed.reshape(batch, seq, hidden)
        if inject is None:
            return mixed if write is None else (mixed, hyper_input)
        return mixed, hyper_input, injection.reshape(batch, seq, hc)
    except Exception as exc:  # noqa: BLE001 - optional native path
        if not _FAILURE_LOGGED:
            _FAILURE_LOGGED = True
            logger.warning(
                "Qwen4 fused hyper-connection kernels failed closed; using the "
                "canonical path: %s",
                exc,
            )
        return None


__all__ = [
    "MAX_ROWS",
    "compatible",
    "enabled",
    "fused_forward",
    "prefill_compatible",
    "prefill_forward",
    "write_enabled",
]
