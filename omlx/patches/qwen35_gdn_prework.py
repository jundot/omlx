# SPDX-License-Identifier: Apache-2.0
#
# Kernel adapted from mlx-serve (src/transformer.zig, GDN_PREWORK_SOURCE),
# itself a port of the mlxfast-challenge qwen35_packed_gdn_prework kernel.
"""Fused GDN prework for Qwen3.5/3.6 verify and selected decode paths.

The composed target-verify prework in mlx-vlm's ``Qwen3_5GatedDeltaNet`` —
conv-state concat + depthwise conv1d + SiLU + q/k/v split + reshapes + two
ones-weight RMS norms + two scalar scales + the next conv-state slice — is
~10 small dispatches per GDN layer per verify cycle. On the 27B that is 48
layers x ~0.29 ms (measured sync-mode at S=4 on M3 Ultra), the largest
remaining verify-cycle cost after the attention split.

One Metal launch replaces the whole chain. Numerics notes carried from the
donor kernel: the in-kernel sigmoid uses MLX's own unary formula
(exp-of-abs), which the challenge swept bit-exact over all finite bf16
inputs; the RMS applies the ones-weight rounding then the separate scalar
multiply's rounding — the composed chain's two casts. The RMS eps follows
mlx-lm ``normalize_qk`` (see ``apply_qwen35_vlm_qk_norm_patch``).

For S=2, the next conv state retains one row from the old conv state.
Longer verify windows fill the entire next state from the new qkv rows.
This kernel also runs automatically for compatible FP16/BF16 B1/T1 decode
through Qwen3_5GatedDeltaNet on Metal. Gates,
recurrence, final norm and projections remain unchanged. Other Qwen3.5
decode shapes and prefill keep the stock path. Qwen4 has its separate
BF16 decode prework and norm-gate kernels below, and its planned decode runs
them with the recurrence as one launch between the two projections. Its B1
speculative verify runs the same step for every row in one launch between
multi-row projections with one-row arithmetic per row, and emits the conv
window and per-step recurrent states the speculative cache records.
"""

from __future__ import annotations

import functools
import logging
import os
import sys

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gated_delta import normalize_qk

from . import qwen35_gdn_verify_fused
from .module_cache import cached_per_module
from .qwen35_verify_qmm import is_row_exact_armed
from .row_exact_qmv import one_row_qmv, rows_qmv

logger = logging.getLogger(__name__)

_PATCHED = False
_ENGAGED_LOGGED = False
_KERNEL = None
_QWEN4_DECODE_KERNEL = None
_QWEN4_NORM_GATE_KERNEL = None
_QWEN4_DECODE_STEP_KERNEL = None
_QWEN4_DECODE_ENGAGED_LOGGED = False
_QWEN35_DECODE_ENGAGED_LOGGED = False
_QWEN4_PREFILL_KERNELS = None
_QWEN4_PREFILL_ENGAGED_LOGGED = False
_QWEN4_PREFILL_ENABLED = os.environ.get("OMLX_QWEN4_GDN_PREFILL_FUSED", "1") != "0"
_QWEN4_PREFILL_MIN_ROWS = 64
# The B1/T1 decode resolves its static eligibility and operands once per layer
# (rebuilt when a weight or child module is replaced); =0 re-derives them per call.
_QWEN4_DECODE_PLAN_ENABLED = os.environ.get("OMLX_QWEN4_GDN_DECODE_PLAN", "1") != "0"
# On the planned decode: the prework, recurrence and norm-gate as one launch
# (=0 keeps the three launches), and the in-/out-projections as one-row
# qmv_fast with a narrower column tile (=0 keeps stock quantized_matmul).
_QWEN4_DECODE_STEP_FUSED = os.environ.get("OMLX_QWEN4_GDN_DECODE_STEP_FUSED", "1") != "0"
_QWEN4_DECODE_QMV = os.environ.get("OMLX_QWEN4_GDN_DECODE_QMV", "1") != "0"
# B1 speculative verify rows (Qwen4 L2 arm): the stacked in-projection, one
# launch for every row's prework, recurrence (with the per-step rollback
# states) and norm-gate, and the out-projection (=0 keeps the per-op path).
_QWEN4_VERIFY_FUSED = os.environ.get("OMLX_QWEN4_GDN_VERIFY_FUSED", "1") != "0"
_QWEN4_BATCH_DECODE = os.environ.get("OMLX_QWEN4_GDN_BATCH_DECODE", "1") != "0"
# Rows of a batched one-token decode per step launch, and the widest batch
# whose projections run per row (bit-identical to each row decoded alone).
_QWEN4_BATCH_DECODE_MAX_ROWS = 16
_QWEN4_BATCH_ROW_EXACT_ROWS = 3
# Its 2..8-row projections on the fully unrolled row-exact tile with the
# per-row-count tiles below (=0 keeps the rolled tiles of the first geometry).
_QWEN4_VERIFY_TILES = os.environ.get("OMLX_QWEN4_GDN_VERIFY_TILES", "1") != "0"
# Its rollback records skip the per-step recurrent states (S-1 x 3 MB per
# layer, written on every verify and read only on a partial accept): a commit
# keeping m of S rows reruns the fused step on the first m rows (=0 writes
# them per verify).
_QWEN4_VERIFY_DEFERRED_STATES = (
    os.environ.get("OMLX_QWEN4_GDN_VERIFY_DEFERRED_STATES", "1") != "0"
)
_QWEN4_VERIFY_STEP_KERNELS: dict = {}
_QWEN4_VERIFY_ENGAGED_LOGGED = False
# The fused verify's norm-gate stage runs step t on simdgroup t of its
# head_v_dim // 8 = 16 simdgroups, so a window holds at most 16 rows.
_QWEN4_VERIFY_MAX_ROWS = 16
_VERIFY_REJECT_DIAG = 0

_SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    uint batch_idx = threadgroup_position_in_grid.y / uint(S);
    uint row = threadgroup_position_in_grid.y % uint(S);
    uint logical_head = threadgroup_position_in_grid.z;
    constexpr uint q_heads = uint(HK);
    constexpr uint k_head_base = uint(HK);
    constexpr uint v_head_base = 2 * uint(HK);
    bool is_q = logical_head < q_heads;
    bool is_k = logical_head >= k_head_base && logical_head < v_head_base;
    uint head = is_q ? logical_head
               : (is_k ? logical_head - k_head_base : logical_head - v_head_base);
    uint channel_base = is_q ? head * uint(DK)
                       : (is_k ? uint(HK) * uint(DK) + head * uint(DK)
                               : 2 * uint(HK) * uint(DK) + head * uint(DV));
    T activated[4];
    float sumsq = 0.0f;
    T l2acc = T(0);
    for (uint i = 0; i < 4; ++i) {
        uint channel = channel_base + lane * 4 + i;
        float acc = 0.0f;
        for (uint tap = 0; tap < 4; ++tap) {
            uint input_row = row + tap;
            const T xv = input_row < uint(NKEEP)
                ? conv_state[(batch_idx * uint(NKEEP) + input_row) * uint(C) + channel]
                : qkv[(batch_idx * uint(S) + input_row - uint(NKEEP)) * uint(C) + channel];
            acc += float(xv) * float(conv_w[channel * 4 + tap]);
        }
        const T conv = T(acc);
        const auto sy = 1 / (1 + metal::precise::exp(metal::abs(conv)));
        const T act = conv * T((conv < T(0)) ? sy : 1 - sy);
        activated[i] = act;
        if (L2) {
            const T sqv = T(float(act) * float(act));
            l2acc = T(float(l2acc) + float(sqv));
        } else {
            float value = float(act);
            sumsq += value * value;
        }
    }
    if (is_q || is_k) {
        uint out_base = ((batch_idx * uint(S) + row) * uint(HK) + head) * uint(DK) + lane * 4;
        if (L2) {
            // Stock Qwen4 L2 chain: x * rsqrt(sum(square(x), -1) + 1e-6),
            // with dk^-0.5 applied to q only.  mx.square rounds per
            // element; mx.sum accumulates each lane's four contiguous
            // bf16 values sequentially, reduces with an fp32 xor tree and
            // rounds once.  Mirror every rounding site exactly.
            float tv = float(l2acc);
            tv += simd_shuffle_xor(tv, short(16));
            tv += simd_shuffle_xor(tv, short(8));
            tv += simd_shuffle_xor(tv, short(4));
            tv += simd_shuffle_xor(tv, short(2));
            tv += simd_shuffle_xor(tv, short(1));
            const T eps = T(float(T(tv)) + float(T(1e-6f)));
            const T inv = T(metal::precise::rsqrt(float(eps)));
            for (uint i = 0; i < 4; ++i) {
                const T l2 = T(float(activated[i]) * float(inv));
                const T value = is_q ? T(float(l2) * float(q_scale)) : l2;
                if (is_q) {
                    q_out[out_base + i] = value;
                } else {
                    k_out[out_base + i] = value;
                }
            }
        } else {
            sumsq = simd_sum(sumsq);
            // normalize_qk: the l2norm eps 1e-6 lands on sum(x^2), so the
            // RMS form uses 1e-6 / DK.
            float inv = metal::precise::rsqrt(sumsq / float(DK) + 1e-6f / float(DK));
            const T scale = is_q ? q_scale : k_scale;
            for (uint i = 0; i < 4; ++i) {
                const T rms = T(1) * T(float(activated[i]) * inv);
                const T value = scale * rms;
                if (is_q) {
                    q_out[out_base + i] = value;
                } else {
                    k_out[out_base + i] = value;
                }
            }
        }
    } else {
        uint out_base = ((batch_idx * uint(S) + row) * uint(HV) + head) * uint(DV) + lane * 4;
        for (uint i = 0; i < 4; ++i) {
            v_out[out_base + i] = activated[i];
        }
    }
    if (S < NKEEP && row == 0) {
        for (uint old_row = 0; old_row < uint(NKEEP - S); ++old_row) {
            uint dst = (batch_idx * uint(NKEEP) + old_row) * uint(C)
                       + channel_base + lane * 4;
            uint src = (batch_idx * uint(NKEEP) + old_row + uint(S)) * uint(C)
                       + channel_base + lane * 4;
            for (uint i = 0; i < 4; ++i) {
                conv_out[dst + i] = conv_state[src + i];
            }
        }
    }
    if (row + uint(NKEEP) >= uint(S)) {
        uint state_row = row + uint(NKEEP) - uint(S);
        uint raw_base = (batch_idx * uint(S) + row) * uint(C) + channel_base + lane * 4;
        uint state_base = (batch_idx * uint(NKEEP) + state_row) * uint(C) + channel_base + lane * 4;
        for (uint i = 0; i < 4; ++i) {
            conv_out[state_base + i] = qkv[raw_base + i];
        }
    }
"""


# Copyright (c) 2026 David Dalcu.  The Qwen4 decode prework and norm-gate
# kernels below, and their prefill variants, are adapted from ddalcu/mlx-serve's
# MIT-licensed ``src/transformer.zig`` at tag ``v26.8.11-pre-release.1``.
# Preserve this scoped notice with those kernels.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Qwen4 decode-only continuation of the donor kernel.  This is deliberately a
# separate ABI from the verify kernel above: the production Qwen4 recurrence
# consumes an FP32 forget gate, while the older donor stores that gate as BF16.
# Keeping distinct outputs lets this path match mlx-vlm's current arithmetic
# exactly instead of silently changing the recurrent state update.  The q/k
# normalization is the verify kernel's L2 arm (stock ``_normalize_qk``), so a
# decode step and a speculative verify row produce the same bits.
_QWEN4_DECODE_HEADER = """
    inline float omlx_log1p(float x) {
        float xp1 = 1.0f + x;
        if (xp1 == metal::numeric_limits<float>::max()) {
            return metal::numeric_limits<float>::max();
        }
        if (xp1 == 1.0f) {
            return x;
        }
        return x * (metal::log(xp1) / (xp1 - 1.0f));
    }
"""


_QWEN4_DECODE_SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    uint logical_head = threadgroup_position_in_grid.z;
    constexpr uint q_heads = uint(HK);
    constexpr uint k_head_base = uint(HK);
    constexpr uint v_head_base = 2 * uint(HK);
    bool is_q = logical_head < q_heads;
    bool is_k = logical_head >= k_head_base && logical_head < v_head_base;
    uint head = is_q ? logical_head
               : (is_k ? logical_head - k_head_base
                       : logical_head - v_head_base);
    uint channel_base = is_q ? head * uint(DK)
                       : (is_k ? uint(HK) * uint(DK) + head * uint(DK)
                               : 2 * uint(HK) * uint(DK) + head * uint(DV));

    T activated[4];
    T l2acc = T(0);
    for (uint i = 0; i < 4; ++i) {
        uint channel = channel_base + lane * 4 + i;
        float acc = 0.0f;
        for (uint tap = 0; tap < 3; ++tap) {
            acc += float(conv_state[tap * uint(C) + channel])
                 * float(conv_w[channel * 4 + tap]);
        }
        acc += float(qkv[channel]) * float(conv_w[channel * 4 + 3]);
        const T conv = T(acc);
        const auto sy = 1 / (1 + metal::precise::exp(metal::abs(conv)));
        const T act = conv * T((conv < T(0)) ? sy : 1 - sy);
        activated[i] = act;
        const T sqv = T(float(act) * float(act));
        l2acc = T(float(l2acc) + float(sqv));

        // T=1: [old0, old1, old2, qkv] -> [old1, old2, qkv].  Each
        // channel is owned by exactly one thread, so no synchronization is
        // needed between these three stores.
        conv_out[channel] = conv_state[uint(C) + channel];
        conv_out[uint(C) + channel] = conv_state[2 * uint(C) + channel];
        conv_out[2 * uint(C) + channel] = qkv[channel];
    }

    if (is_q || is_k) {
        // Stock Qwen4 L2 chain, rounding for rounding as the verify kernel's
        // L2 arm: x * rsqrt(sum(square(x), -1) + 1e-6), dk^-0.5 on q only.
        float tv = float(l2acc);
        tv += simd_shuffle_xor(tv, short(16));
        tv += simd_shuffle_xor(tv, short(8));
        tv += simd_shuffle_xor(tv, short(4));
        tv += simd_shuffle_xor(tv, short(2));
        tv += simd_shuffle_xor(tv, short(1));
        const T eps = T(float(T(tv)) + float(T(1e-6f)));
        const T inv = T(metal::precise::rsqrt(float(eps)));
        uint out_base = head * uint(DK) + lane * 4;
        for (uint i = 0; i < 4; ++i) {
            const T l2 = T(float(activated[i]) * float(inv));
            if (is_q) {
                q_out[out_base + i] = T(float(l2) * float(q_scale));
            } else {
                k_out[out_base + i] = l2;
            }
        }
    } else {
        uint out_base = head * uint(DV) + lane * 4;
        for (uint i = 0; i < 4; ++i) {
            v_out[out_base + i] = activated[i];
        }
        if (lane == 0) {
            const T bv = b_in[head];
            // MLX's Sigmoid functor, rounded to T once.
            const auto by = 1 / (1 + metal::precise::exp(metal::abs(bv)));
            beta_out[head] = T((bv < T(0)) ? by : 1 - by);

            // compute_g casts A_log to FP32 but keeps softplus(a+dt_bias)
            // in BF16 before the FP32 multiply and outer exp.
            const T apd = T(float(a_in[head]) + float(dt_bias[head]));
            const T neg_abs = -metal::abs(apd);
            const T exp_term = T(metal::precise::exp(float(neg_abs)));
            const T log_term = T(omlx_log1p(float(exp_term)));
            const T positive = metal::max(apd, T(0));
            const T sp = T(float(positive) + float(log_term));
            float ea = metal::precise::exp(float(A_log[head]));
            g_out[head] = metal::precise::exp(-(ea * float(sp)));
        }
    }
"""


_QWEN4_NORM_GATE_SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    uint head = threadgroup_position_in_grid.z;
    uint base = head * uint(DV) + lane * 4;
    float xs[4];
    float sumsq = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        xs[i] = float(y[base + i]);
        sumsq += xs[i] * xs[i];
    }
    sumsq = simd_sum(sumsq);
    float inv = metal::precise::rsqrt(sumsq / float(DV) + float(eps));
    for (uint i = 0; i < 4; ++i) {
        // mx.fast.rms_norm materializes BF16 before Qwen4 casts it back to
        // FP32 for the sigmoid product.
        const T normed = norm_w[lane * 4 + i] * T(xs[i] * inv);
        float zv = float(z[base + i]);
        float sy = 1.0f / (1.0f + metal::precise::exp(metal::abs(zv)));
        float sig = zv < 0.0f ? sy : 1.0f - sy;
        out[base + i] = T(float(normed) * sig);
    }
"""


# The whole B1/T1 GDN step between the projections in one launch, one
# threadgroup per value head (16 simdgroups, the head's 128 value rows):
#   1. simdgroups 0/1/2 run the decode prework above for this head's q, k
#      (key head hv / (HV / HK)) and v channels, lane for lane (same channels
#      per lane, same L2 butterfly), one lane of simdgroup 3 its g and beta;
#      the results go to threadgroup memory in the prework's output types.
#      The next conv state is written once per channel: q/k channels by the
#      first value head of their key head, v channels by their own head.
#   2. mlx-lm's ``gated_delta_step_packed_btree`` recurrence (T=1; four lanes
#      and 32 FP32 state values per value row, the same partial sums and
#      xor-1/xor-2 shuffles) reading q/k/v/g/beta from threadgroup memory;
#      each row's BF16 output goes to threadgroup memory.
#   3. simdgroup 0 runs the norm-gate above on the head's 128 outputs (same
#      four values per lane and simd_sum).
# Every value takes the three-launch path's arithmetic in the same order;
# only where the intermediates live changes (threadgroup memory instead of
# q/k/v/g/beta/y device buffers).
_QWEN4_DECODE_STEP_SOURCE = """
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint hv = threadgroup_position_in_grid.z;
    const uint hk = hv / uint(HV / HK);
    threadgroup T tq[DK];
    threadgroup T tk[DK];
    threadgroup T tv[DV];
    threadgroup T ty[DV];
    threadgroup float tg_g[1];
    threadgroup T tg_beta[1];

    // Recurrence geometry (packed kernel): four lanes per value row.
    constexpr int lanes_per_row = 4;
    constexpr int rows_per_simdgroup = 32 / lanes_per_row;
    constexpr int values_per_lane = DK / lanes_per_row;
    constexpr int partials_per_lane = values_per_lane / 4;
    const int lane_in_row = int(lane) & (lanes_per_row - 1);
    const int dv_idx = int(sg) * rows_per_simdgroup + int(lane) / lanes_per_row;
    auto i_state = state_in + (int(hv) * DV + dv_idx) * DK + lane_in_row * values_per_lane;
    auto o_state = state_out + (int(hv) * DV + dv_idx) * DK + lane_in_row * values_per_lane;

    float state[values_per_lane];
    for (int i = 0; i < values_per_lane; ++i) {
      state[i] = static_cast<float>(i_state[i]);
    }

    if (sg < 3) {
        const bool is_q = sg == 0;
        const bool is_k = sg == 1;
        const uint head = (is_q || is_k) ? hk : hv;
        const uint channel_base = is_q ? head * uint(DK)
                           : (is_k ? uint(HK) * uint(DK) + head * uint(DK)
                                   : 2 * uint(HK) * uint(DK) + head * uint(DV));
        const bool write_conv = !(is_q || is_k) || (hv % uint(HV / HK)) == 0;

        T activated[4];
        T l2acc = T(0);
        for (uint i = 0; i < 4; ++i) {
            uint channel = channel_base + lane * 4 + i;
            float acc = 0.0f;
            for (uint tap = 0; tap < 3; ++tap) {
                acc += float(conv_state[tap * uint(C) + channel])
                     * float(conv_w[channel * 4 + tap]);
            }
            acc += float(qkv[channel]) * float(conv_w[channel * 4 + 3]);
            const T conv = T(acc);
            const auto sy = 1 / (1 + metal::precise::exp(metal::abs(conv)));
            const T act = conv * T((conv < T(0)) ? sy : 1 - sy);
            activated[i] = act;
            const T sqv = T(float(act) * float(act));
            l2acc = T(float(l2acc) + float(sqv));
            if (write_conv) {
                conv_out[channel] = conv_state[uint(C) + channel];
                conv_out[uint(C) + channel] = conv_state[2 * uint(C) + channel];
                conv_out[2 * uint(C) + channel] = qkv[channel];
            }
        }

        if (is_q || is_k) {
            float tv_sum = float(l2acc);
            tv_sum += simd_shuffle_xor(tv_sum, short(16));
            tv_sum += simd_shuffle_xor(tv_sum, short(8));
            tv_sum += simd_shuffle_xor(tv_sum, short(4));
            tv_sum += simd_shuffle_xor(tv_sum, short(2));
            tv_sum += simd_shuffle_xor(tv_sum, short(1));
            const T eps_l2 = T(float(T(tv_sum)) + float(T(1e-6f)));
            const T inv = T(metal::precise::rsqrt(float(eps_l2)));
            for (uint i = 0; i < 4; ++i) {
                const T l2 = T(float(activated[i]) * float(inv));
                if (is_q) {
                    tq[lane * 4 + i] = T(float(l2) * float(q_scale));
                } else {
                    tk[lane * 4 + i] = l2;
                }
            }
        } else {
            for (uint i = 0; i < 4; ++i) {
                tv[lane * 4 + i] = activated[i];
            }
        }
    } else if (sg == 3 && lane == 0) {
        const uint head = hv;
        const T bv = b_in[head];
        // MLX's Sigmoid functor, rounded to T once.
        const auto by = 1 / (1 + metal::precise::exp(metal::abs(bv)));
        tg_beta[0] = T((bv < T(0)) ? by : 1 - by);

        const T apd = T(float(a_in[head]) + float(dt_bias[head]));
        const T neg_abs = -metal::abs(apd);
        const T exp_term = T(metal::precise::exp(float(neg_abs)));
        const T log_term = T(omlx_log1p(float(exp_term)));
        const T positive = metal::max(apd, T(0));
        const T sp = T(float(positive) + float(log_term));
        float ea = metal::precise::exp(float(A_log[head]));
        tg_g[0] = metal::precise::exp(-(ea * float(sp)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        float gt = static_cast<float>(tg_g[0]);
        const threadgroup T* q_ = tq + lane_in_row * values_per_lane;
        const threadgroup T* k_ = tk + lane_in_row * values_per_lane;

        float part[partials_per_lane];
        for (int pb = 0; pb < partials_per_lane; ++pb) {
          float acc = 0.0f;
          for (int i = 0; i < 4; ++i) {
            int e = pb * 4 + i;
            state[e] = state[e] * gt;
            acc += state[e] * static_cast<float>(k_[e]);
          }
          part[pb] = acc;
        }
        float kv_mem =
            ((part[0] + part[1]) + (part[2] + part[3])) +
            ((part[4] + part[5]) + (part[6] + part[7]));
        kv_mem += simd_shuffle_xor(kv_mem, 1);
        kv_mem += simd_shuffle_xor(kv_mem, 2);

        auto delta =
            (static_cast<float>(tv[dv_idx]) - kv_mem) *
            static_cast<float>(tg_beta[0]);

        for (int pb = 0; pb < partials_per_lane; ++pb) {
          float acc = 0.0f;
          for (int i = 0; i < 4; ++i) {
            int e = pb * 4 + i;
            state[e] = state[e] + static_cast<float>(k_[e]) * delta;
            acc += state[e] * static_cast<float>(q_[e]);
          }
          part[pb] = acc;
        }
        float row_out =
            ((part[0] + part[1]) + (part[2] + part[3])) +
            ((part[4] + part[5]) + (part[6] + part[7]));
        row_out += simd_shuffle_xor(row_out, 1);
        row_out += simd_shuffle_xor(row_out, 2);
        if (lane_in_row == 0) {
          ty[dv_idx] = static_cast<T>(row_out);
        }

        for (int i = 0; i < values_per_lane; ++i) {
          o_state[i] = static_cast<float>(state[i]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        uint base = hv * uint(DV) + lane * 4;
        float xs[4];
        float sumsq = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            xs[i] = float(ty[lane * 4 + i]);
            sumsq += xs[i] * xs[i];
        }
        sumsq = simd_sum(sumsq);
        float inv = metal::precise::rsqrt(sumsq / float(DV) + float(eps));
        for (uint i = 0; i < 4; ++i) {
            const T normed = norm_w[lane * 4 + i] * T(xs[i] * inv);
            float zv = float(z[base + i]);
            float sy = 1.0f / (1.0f + metal::precise::exp(metal::abs(zv)));
            float sig = zv < 0.0f ? sy : 1.0f - sy;
            out[base + i] = T(float(normed) * sig);
        }
    }
"""


# The Qwen4 speculative verify of S rows between the projections in one
# launch: the decode step above for t = 0..S-1 in order from the committed
# state, one threadgroup per value head (16 simdgroups, 128 value rows).
# ``proj`` holds each row's stacked in-projection [qkv (C) | z (HV*DV) |
# b (HV) | a (HV)]; step t's convolution window is rows t..t+2 of
# [conv_state; qkv rows] and its input qkv row t.
#   1. The decode step's prework for every (step, q/k/v) pair, one simdgroup
#      per pair (same channels per lane, same L2 butterfly), and one lane of
#      the last simdgroup per step for g and beta; all S steps' q/k/v/g/beta go
#      to threadgroup memory (they do not depend on the recurrence). The
#      rollback window [conv_state; qkv rows] and the next conv state (its
#      last three rows) are written once per channel.
#   2. The decode step's recurrence for t = 0..S-1 on the state registers;
#      the state after every step but the last goes to ``states`` (the
#      per-step history the speculative cache records), the last to
#      ``state_out``.
#   3. The decode step's norm-gate of step t on simdgroup t.
# Every value takes the t-th serial decode step's arithmetic in the same
# order; only where the intermediates live and which simdgroup computes a
# (step, q/k/v) pair change.
_QWEN4_VERIFY_STEP_SOURCE = """
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint hv = threadgroup_position_in_grid.z;
    const uint hk = hv / uint(HV / HK);
    constexpr uint simdgroups = uint(DV) / 8;
    constexpr uint z_off = uint(C);
    constexpr uint b_off = z_off + uint(HV) * uint(DV);
    constexpr uint a_off = b_off + uint(HV);
    constexpr uint P = a_off + uint(HV);
    threadgroup T tq[S * DK];
    threadgroup T tk[S * DK];
    threadgroup T tv[S * DV];
    threadgroup T ty[S * DV];
    threadgroup float tg_g[S];
    threadgroup T tg_beta[S];

    constexpr int lanes_per_row = 4;
    constexpr int rows_per_simdgroup = 32 / lanes_per_row;
    constexpr int values_per_lane = DK / lanes_per_row;
    constexpr int partials_per_lane = values_per_lane / 4;
    const int lane_in_row = int(lane) & (lanes_per_row - 1);
    const int dv_idx = int(sg) * rows_per_simdgroup + int(lane) / lanes_per_row;
    const int state_offset =
        (int(hv) * DV + dv_idx) * DK + lane_in_row * values_per_lane;

    float state[values_per_lane];
    for (int i = 0; i < values_per_lane; ++i) {
      state[i] = static_cast<float>(state_in[state_offset + i]);
    }

    for (uint task = sg; task < 3 * uint(S); task += simdgroups) {
        const uint t = task / 3;
        const bool is_q = task % 3 == 0;
        const bool is_k = task % 3 == 1;
        const uint head = (is_q || is_k) ? hk : hv;
        const uint channel_base = is_q ? head * uint(DK)
                           : (is_k ? uint(HK) * uint(DK) + head * uint(DK)
                                   : 2 * uint(HK) * uint(DK) + head * uint(DV));
        const bool write_conv = !(is_q || is_k) || (hv % uint(HV / HK)) == 0;

        T activated[4];
        T l2acc = T(0);
        for (uint i = 0; i < 4; ++i) {
            uint channel = channel_base + lane * 4 + i;
            float acc = 0.0f;
            for (uint tap = 0; tap < 3; ++tap) {
                const uint src = t + tap;
                const T xv = src < 3 ? conv_state[src * uint(C) + channel]
                                     : proj[(src - 3) * P + channel];
                acc += float(xv) * float(conv_w[channel * 4 + tap]);
            }
            const T x_t = proj[t * P + channel];
            acc += float(x_t) * float(conv_w[channel * 4 + 3]);
            const T conv = T(acc);
            const auto sy = 1 / (1 + metal::precise::exp(metal::abs(conv)));
            const T act = conv * T((conv < T(0)) ? sy : 1 - sy);
            activated[i] = act;
            const T sqv = T(float(act) * float(act));
            l2acc = T(float(l2acc) + float(sqv));
            if (write_conv) {
                window[(3 + t) * uint(C) + channel] = x_t;
                if (t == 0) {
                    for (uint row = 0; row < 3; ++row) {
                        window[row * uint(C) + channel] = conv_state[row * uint(C) + channel];
                        const uint src = uint(S) + row;
                        conv_out[row * uint(C) + channel] = src < 3
                            ? conv_state[src * uint(C) + channel]
                            : proj[(src - 3) * P + channel];
                    }
                }
            }
        }

        if (is_q || is_k) {
            float tv_sum = float(l2acc);
            tv_sum += simd_shuffle_xor(tv_sum, short(16));
            tv_sum += simd_shuffle_xor(tv_sum, short(8));
            tv_sum += simd_shuffle_xor(tv_sum, short(4));
            tv_sum += simd_shuffle_xor(tv_sum, short(2));
            tv_sum += simd_shuffle_xor(tv_sum, short(1));
            const T eps_l2 = T(float(T(tv_sum)) + float(T(1e-6f)));
            const T inv = T(metal::precise::rsqrt(float(eps_l2)));
            for (uint i = 0; i < 4; ++i) {
                const T l2 = T(float(activated[i]) * float(inv));
                if (is_q) {
                    tq[t * DK + lane * 4 + i] = T(float(l2) * float(q_scale));
                } else {
                    tk[t * DK + lane * 4 + i] = l2;
                }
            }
        } else {
            for (uint i = 0; i < 4; ++i) {
                tv[t * DV + lane * 4 + i] = activated[i];
            }
        }
    }
    if (sg == simdgroups - 1 && lane < uint(S)) {
        const uint t = lane;
        const uint head = hv;
        const T bv = proj[t * P + b_off + head];
        // MLX's Sigmoid functor, rounded to T once.
        const auto by = 1 / (1 + metal::precise::exp(metal::abs(bv)));
        tg_beta[t] = T((bv < T(0)) ? by : 1 - by);

        const T apd = T(float(proj[t * P + a_off + head]) + float(dt_bias[head]));
        const T neg_abs = -metal::abs(apd);
        const T exp_term = T(metal::precise::exp(float(neg_abs)));
        const T log_term = T(omlx_log1p(float(exp_term)));
        const T positive = metal::max(apd, T(0));
        const T sp = T(float(positive) + float(log_term));
        float ea = metal::precise::exp(float(A_log[head]));
        tg_g[t] = metal::precise::exp(-(ea * float(sp)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = 0; t < uint(S); ++t) {
        float gt = static_cast<float>(tg_g[t]);
        const threadgroup T* q_ = tq + t * DK + lane_in_row * values_per_lane;
        const threadgroup T* k_ = tk + t * DK + lane_in_row * values_per_lane;

        float part[partials_per_lane];
        for (int pb = 0; pb < partials_per_lane; ++pb) {
          float acc = 0.0f;
          for (int i = 0; i < 4; ++i) {
            int e = pb * 4 + i;
            state[e] = state[e] * gt;
            acc += state[e] * static_cast<float>(k_[e]);
          }
          part[pb] = acc;
        }
        float kv_mem =
            ((part[0] + part[1]) + (part[2] + part[3])) +
            ((part[4] + part[5]) + (part[6] + part[7]));
        kv_mem += simd_shuffle_xor(kv_mem, 1);
        kv_mem += simd_shuffle_xor(kv_mem, 2);

        auto delta =
            (static_cast<float>(tv[t * DV + dv_idx]) - kv_mem) *
            static_cast<float>(tg_beta[t]);

        for (int pb = 0; pb < partials_per_lane; ++pb) {
          float acc = 0.0f;
          for (int i = 0; i < 4; ++i) {
            int e = pb * 4 + i;
            state[e] = state[e] + static_cast<float>(k_[e]) * delta;
            acc += state[e] * static_cast<float>(q_[e]);
          }
          part[pb] = acc;
        }
        float row_out =
            ((part[0] + part[1]) + (part[2] + part[3])) +
            ((part[4] + part[5]) + (part[6] + part[7]));
        row_out += simd_shuffle_xor(row_out, 1);
        row_out += simd_shuffle_xor(row_out, 2);
        if (lane_in_row == 0) {
          ty[t * DV + dv_idx] = static_cast<T>(row_out);
        }
__STATES__
    }
    for (int i = 0; i < values_per_lane; ++i) {
      state_out[state_offset + i] = static_cast<float>(state[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg < uint(S)) {
        const uint t = sg;
        uint base = (t * uint(HV) + hv) * uint(DV) + lane * 4;
        uint z_base = t * P + z_off + hv * uint(DV) + lane * 4;
        float xs[4];
        float sumsq = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            xs[i] = float(ty[t * DV + lane * 4 + i]);
            sumsq += xs[i] * xs[i];
        }
        sumsq = simd_sum(sumsq);
        float inv = metal::precise::rsqrt(sumsq / float(DV) + float(eps));
        for (uint i = 0; i < 4; ++i) {
            const T normed = norm_w[lane * 4 + i] * T(xs[i] * inv);
            float zv = float(proj[z_base + i]);
            float sy = 1.0f / (1.0f + metal::precise::exp(metal::abs(zv)));
            float sig = zv < 0.0f ? sy : 1.0f - sy;
            out[base + i] = T(float(normed) * sig);
        }
    }
"""

# The state after step t < S - 1, [1, S - 1, HV, DV, DK] (S > 1 only).
_QWEN4_VERIFY_STATES_WRITE = """
        if (t + 1 < uint(S)) {
          auto t_state = states + int(t) * (HV * DV * DK) + state_offset;
          for (int i = 0; i < values_per_lane; ++i) {
            t_state[i] = static_cast<float>(state[i]);
          }
        }
"""


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="omlx_qwen35_gdn_prework",
            input_names=["qkv", "conv_state", "conv_w", "q_scale", "k_scale"],
            output_names=["q_out", "k_out", "v_out", "conv_out"],
            source=_SOURCE,
        )
    return _KERNEL


def gdn_prework_fused(
    qkv, conv_state, conv_w, q_scale, k_scale, hk, hv, dk, dv, l2=False
):
    """One fused dispatch. qkv [B,S,C], conv_state [B,3,C], conv_w [C,4,1].

    l2=True selects the Qwen4 L2 q/k normalization (q_scale carries the
    dk^-0.5 query scale); otherwise the Qwen3.5 RMS scaling is used.
    """
    batch_size = qkv.shape[0]
    s_len = qkv.shape[1]
    c_dim = qkv.shape[2]
    outs = _kernel()(
        inputs=[qkv, conv_state, conv_w, q_scale, k_scale],
        template=[
            ("T", qkv.dtype),
            ("HK", hk),
            ("HV", hv),
            ("DK", dk),
            ("DV", dv),
            ("NKEEP", 3),
            ("C", c_dim),
            ("S", s_len),
            ("L2", 1 if l2 else 0),
        ],
        grid=(32, batch_size * s_len, 2 * hk + hv),
        threadgroup=(32, 1, 1),
        output_shapes=[
            (batch_size, s_len, hk, dk),
            (batch_size, s_len, hk, dk),
            (batch_size, s_len, hv, dv),
            (batch_size, 3, c_dim),
        ],
        output_dtypes=[qkv.dtype] * 4,
    )
    return outs


def _qwen4_decode_kernel():
    global _QWEN4_DECODE_KERNEL
    if _QWEN4_DECODE_KERNEL is None:
        _QWEN4_DECODE_KERNEL = mx.fast.metal_kernel(
            name="omlx_qwen4_gdn_decode_prework",
            input_names=[
                "qkv",
                "conv_state",
                "conv_w",
                "q_scale",
                "b_in",
                "a_in",
                "A_log",
                "dt_bias",
            ],
            output_names=[
                "q_out",
                "k_out",
                "v_out",
                "conv_out",
                "g_out",
                "beta_out",
            ],
            header=_QWEN4_DECODE_HEADER,
            source=_QWEN4_DECODE_SOURCE,
        )
    return _QWEN4_DECODE_KERNEL


def qwen4_decode_prework_fused(
    qkv,
    conv_state,
    conv_w,
    q_scale,
    b,
    a,
    A_log,
    dt_bias,
    hk,
    hv,
    dk,
    dv,
):
    """Fuse the exact Qwen4 B1/T1 GDN prework into one Metal dispatch.

    ``q_scale`` is the dk^-0.5 query scale applied after the L2 norm.
    """

    c_dim = int(qkv.shape[-1])
    return _qwen4_decode_kernel()(
        inputs=[
            qkv,
            conv_state,
            conv_w,
            q_scale,
            b,
            a,
            A_log,
            dt_bias,
        ],
        template=[
            ("T", qkv.dtype),
            ("HK", hk),
            ("HV", hv),
            ("DK", dk),
            ("DV", dv),
            ("C", c_dim),
        ],
        grid=(32, 1, 2 * hk + hv),
        threadgroup=(32, 1, 1),
        output_shapes=[
            (1, 1, hk, dk),
            (1, 1, hk, dk),
            (1, 1, hv, dv),
            (1, 3, c_dim),
            (1, 1, hv),
            (1, 1, hv),
        ],
        output_dtypes=[
            qkv.dtype,
            qkv.dtype,
            qkv.dtype,
            qkv.dtype,
            mx.float32,
            qkv.dtype,
        ],
    )


def _qwen4_prefill_prework_source() -> str:
    # The verify L2 prework with the row count read at run time, so every
    # prefill width shares one pipeline.
    source = _SOURCE
    for old, new in (
        ("uint(NKEEP - S)", "uint(NKEEP) - S_rt"),
        ("S < NKEEP", "S_rt < uint(NKEEP)"),
        ("uint(S)", "S_rt"),
    ):
        if old not in source:
            raise RuntimeError(f"GDN prework source changed: {old!r} missing")
        source = source.replace(old, new)
    return "    const uint S_rt = uint(s_len);\n" + source


def _qwen4_prefill_norm_gate_source() -> str:
    old = "uint base = head * uint(DV) + lane * 4;"
    if old not in _QWEN4_NORM_GATE_SOURCE:
        raise RuntimeError("GDN norm-gate source changed")
    return _QWEN4_NORM_GATE_SOURCE.replace(
        old,
        "uint base = (threadgroup_position_in_grid.y * uint(HV) + head) * uint(DV)"
        " + lane * 4;",
    )


def _qwen4_prefill_kernels():
    global _QWEN4_PREFILL_KERNELS
    if _QWEN4_PREFILL_KERNELS is None:
        _QWEN4_PREFILL_KERNELS = (
            mx.fast.metal_kernel(
                name="omlx_qwen4_gdn_prefill_prework",
                input_names=[
                    "qkv",
                    "conv_state",
                    "conv_w",
                    "q_scale",
                    "k_scale",
                    "s_len",
                ],
                output_names=["q_out", "k_out", "v_out", "conv_out"],
                source=_qwen4_prefill_prework_source(),
            ),
            mx.fast.metal_kernel(
                name="omlx_qwen4_gdn_prefill_norm_gate",
                input_names=["y", "z", "norm_w", "eps"],
                output_names=["out"],
                source=_qwen4_prefill_norm_gate_source(),
            ),
        )
    return _QWEN4_PREFILL_KERNELS


def _qwen4_prefill_eligible(module, inputs, mask, cache) -> bool:
    if (
        not _QWEN4_PREFILL_ENABLED
        or mask is not None
        or cache is None
        or not isinstance(inputs, mx.array)
        or inputs.ndim != 3
        or inputs.shape[0] != 1
        or inputs.shape[1] < _QWEN4_PREFILL_MIN_ROWS
        or inputs.shape[2] != _QWEN4_HIDDEN_SIZE
        or inputs.dtype != mx.bfloat16
        or mx.default_device() != mx.gpu
        or len(getattr(cache, "cache", ())) != 2
        or getattr(cache, "is_speculating", True)
        or getattr(cache, "history_capacity", 0)
        or getattr(cache, "lengths", None) is not None
        or getattr(cache, "left_padding", None) is not None
    ):
        return False
    conv_state, recurrent_state = cache[0], cache[1]
    if conv_state is not None and not (
        isinstance(conv_state, mx.array)
        and conv_state.shape == (1, 3, 10240)
        and conv_state.dtype == mx.bfloat16
    ):
        return False
    if recurrent_state is not None and not (
        isinstance(recurrent_state, mx.array)
        and recurrent_state.shape == (1, 48, 128, 128)
        and recurrent_state.dtype == mx.float32
    ):
        return False
    return _qwen4_gdn_geometry_ok(module)


def _qwen4_prefill(module, inputs, cache):
    """Stock Qwen4 GDN prefill with the conv/L2 prework and norm-gate fused."""
    from mlx_vlm.models.qwen3_5 import language as q35

    length = inputs.shape[1]
    mixed_qkv = module.in_proj_qkv(inputs)
    z = module.in_proj_z(inputs)
    b, a = module._project_gates(inputs)
    conv_state = cache[0]
    if conv_state is None:
        conv_state = mx.zeros((1, 3, module.conv_dim), dtype=inputs.dtype)
    prework, norm_gate = _qwen4_prefill_kernels()
    hk, hv = module.num_k_heads, module.num_v_heads
    dk, dv = module.head_k_dim, module.head_v_dim
    q, k, v, next_conv = prework(
        inputs=[
            mixed_qkv,
            conv_state,
            module.conv1d.weight,
            mx.array(dk**-0.5, dtype=mx.bfloat16),
            mx.array(1.0, dtype=mx.bfloat16),
            mx.array(length, dtype=mx.int32),
        ],
        template=[
            ("T", inputs.dtype),
            ("HK", hk),
            ("HV", hv),
            ("DK", dk),
            ("DV", dv),
            ("NKEEP", 3),
            ("C", module.conv_dim),
            ("L2", 1),
        ],
        grid=(32, length, 2 * hk + hv),
        threadgroup=(32, 1, 1),
        output_shapes=[
            (1, length, hk, dk),
            (1, length, hk, dk),
            (1, length, hv, dv),
            (1, 3, module.conv_dim),
        ],
        output_dtypes=[inputs.dtype] * 4,
    )
    cache[0] = next_conv
    out, _ = q35.gated_delta_update(
        q, k, v, a, b, module.A_log, module.dt_bias, cache=cache, use_kernel=True
    )
    if hasattr(cache, "advance"):
        cache.advance(length)
        q35._qwen3_5_advance_lengths_info(cache, length)
    flat = norm_gate(
        inputs=[
            out,
            z,
            module.norm.weight,
            mx.array(module.norm.eps, dtype=mx.float32),
        ],
        template=[("T", out.dtype), ("HV", hv), ("DV", dv)],
        grid=(32, length, hv),
        threadgroup=(32, 1, 1),
        output_shapes=[(1, length, hv * dv)],
        output_dtypes=[out.dtype],
    )[0]
    global _QWEN4_PREFILL_ENGAGED_LOGGED
    if not _QWEN4_PREFILL_ENGAGED_LOGGED:
        _QWEN4_PREFILL_ENGAGED_LOGGED = True
        logger.info("Qwen4 fused GDN prefill prework and norm-gate engaged")
    return module.out_proj(flat)


def _qwen4_norm_gate_kernel():
    global _QWEN4_NORM_GATE_KERNEL
    if _QWEN4_NORM_GATE_KERNEL is None:
        _QWEN4_NORM_GATE_KERNEL = mx.fast.metal_kernel(
            name="omlx_qwen4_gdn_decode_norm_gate",
            input_names=["y", "z", "norm_w", "eps"],
            output_names=["out"],
            source=_QWEN4_NORM_GATE_SOURCE,
        )
    return _QWEN4_NORM_GATE_KERNEL


def qwen4_decode_norm_gate_fused(y, z, norm_w, *, hv, dv, eps):
    """Fuse Qwen4's BF16 RMSNorm + FP32 sigmoid-gate at B1/T1."""

    return _qwen4_norm_gate(y, z, norm_w, mx.array(eps, dtype=mx.float32), hv, dv)


def _qwen4_norm_gate(y, z, norm_w, eps, hv, dv):
    """``qwen4_decode_norm_gate_fused`` with ``eps`` as a float32 scalar array."""

    return _qwen4_norm_gate_kernel()(
        inputs=[y, z, norm_w, eps],
        template=[
            ("T", y.dtype),
            ("HV", hv),
            ("DV", dv),
        ],
        grid=(32, 1, hv),
        threadgroup=(32, 1, 1),
        output_shapes=[(1, 1, hv * dv)],
        output_dtypes=[y.dtype],
    )[0]


def _qwen4_decode_recurrence(q, k, v, g, beta, state):
    from mlx_lm.models.gated_delta import gated_delta_kernel

    return gated_delta_kernel(q, k, v, g, beta, state, None)


def _qwen4_decode_step_kernel():
    global _QWEN4_DECODE_STEP_KERNEL
    if _QWEN4_DECODE_STEP_KERNEL is None:
        _QWEN4_DECODE_STEP_KERNEL = mx.fast.metal_kernel(
            name="omlx_qwen4_gdn_decode_step",
            input_names=[
                "qkv",
                "z",
                "b_in",
                "a_in",
                "conv_state",
                "conv_w",
                "q_scale",
                "A_log",
                "dt_bias",
                "state_in",
                "norm_w",
                "eps",
            ],
            output_names=["conv_out", "state_out", "out"],
            header=_QWEN4_DECODE_HEADER,
            source=_QWEN4_DECODE_STEP_SOURCE,
        )
    return _QWEN4_DECODE_STEP_KERNEL


def qwen4_decode_step_fused(
    qkv, z, b, a, conv_state, conv_w, q_scale, A_log, dt_bias, state, norm_w, eps,
    hk, hv, dk, dv,
):
    """The B1/T1 prework, recurrence and norm-gate in one launch (dk = dv = 128).

    Returns (next conv state, next recurrent state, gated output [1, 1, hv*dv]),
    bit-identical to ``qwen4_decode_prework_fused`` -> ``gated_delta_kernel``
    -> ``_qwen4_norm_gate``. ``eps`` is the norm epsilon as a float32 array.
    """
    rows = dv // 8
    return _qwen4_decode_step_kernel()(
        inputs=[qkv, z, b, a, conv_state, conv_w, q_scale, A_log, dt_bias, state, norm_w, eps],
        template=[
            ("T", qkv.dtype),
            ("HK", hk),
            ("HV", hv),
            ("DK", dk),
            ("DV", dv),
            ("C", int(qkv.shape[-1])),
        ],
        grid=(32, rows, hv),
        threadgroup=(32, rows, 1),
        output_shapes=[conv_state.shape, state.shape, (1, 1, hv * dv)],
        output_dtypes=[qkv.dtype, state.dtype, qkv.dtype],
    )


# The decode step above for every row of a batched one-token decode: grid z
# is (row, value head); each threadgroup rebinds the kernel's buffers to its
# row and runs the one-row source verbatim, so row r's outputs and next
# states are bit-identical to a B1 decode step of that row alone.
_QWEN4_BATCH_DECODE_STEP_SOURCE = (
    """
    const uint batch_row = threadgroup_position_in_grid.z / uint(HV);
    const uint head_in_row = threadgroup_position_in_grid.z % uint(HV);
    const auto qkv_row = qkv + batch_row * uint(C);
    const auto z_row = z + batch_row * uint(HV) * uint(DV);
    const auto b_row = b_in + batch_row * uint(HV);
    const auto a_row = a_in + batch_row * uint(HV);
    const auto conv_state_row = conv_state + batch_row * 3 * uint(C);
    const auto state_in_row = state_in + batch_row * uint(HV) * uint(DV) * uint(DK);
    const auto conv_out_row = conv_out + batch_row * 3 * uint(C);
    const auto state_out_row = state_out + batch_row * uint(HV) * uint(DV) * uint(DK);
    const auto out_row = out + batch_row * uint(HV) * uint(DV);
    {
    const auto qkv = qkv_row;
    const auto z = z_row;
    const auto b_in = b_row;
    const auto a_in = a_row;
    const auto conv_state = conv_state_row;
    const auto state_in = state_in_row;
    const auto conv_out = conv_out_row;
    const auto state_out = state_out_row;
    const auto out = out_row;
"""
    + _QWEN4_DECODE_STEP_SOURCE.replace("threadgroup_position_in_grid.z", "head_in_row")
    + """
    }
"""
)
_QWEN4_BATCH_DECODE_STEP_KERNEL = None


def qwen4_batch_decode_step_fused(
    qkv, z, b, a, conv_state, conv_w, q_scale, A_log, dt_bias, state, norm_w, eps,
    hk, hv, dk, dv,
):
    """``qwen4_decode_step_fused`` for each of B rows ([B, 1, ...] inputs and
    [B, ...] states) in one launch."""
    global _QWEN4_BATCH_DECODE_STEP_KERNEL
    if _QWEN4_BATCH_DECODE_STEP_KERNEL is None:
        _QWEN4_BATCH_DECODE_STEP_KERNEL = mx.fast.metal_kernel(
            name="omlx_qwen4_gdn_batch_decode_step",
            input_names=[
                "qkv", "z", "b_in", "a_in", "conv_state", "conv_w", "q_scale",
                "A_log", "dt_bias", "state_in", "norm_w", "eps",
            ],
            output_names=["conv_out", "state_out", "out"],
            header=_QWEN4_DECODE_HEADER,
            source=_QWEN4_BATCH_DECODE_STEP_SOURCE,
        )
    batch = qkv.shape[0]
    rows = dv // 8
    return _QWEN4_BATCH_DECODE_STEP_KERNEL(
        inputs=[qkv, z, b, a, conv_state, conv_w, q_scale, A_log, dt_bias, state, norm_w, eps],
        template=[
            ("T", qkv.dtype),
            ("HK", hk),
            ("HV", hv),
            ("DK", dk),
            ("DV", dv),
            ("C", int(qkv.shape[-1])),
        ],
        grid=(32, rows, hv * batch),
        threadgroup=(32, rows, 1),
        output_shapes=[conv_state.shape, state.shape, (batch, 1, hv * dv)],
        output_dtypes=[qkv.dtype, state.dtype, qkv.dtype],
    )


def _qwen4_verify_step_kernel(states: bool):
    kernel = _QWEN4_VERIFY_STEP_KERNELS.get(states)
    if kernel is None:
        kernel = _QWEN4_VERIFY_STEP_KERNELS[states] = mx.fast.metal_kernel(
            name="omlx_qwen4_gdn_verify_step" + ("_states" if states else ""),
            input_names=[
                "proj",
                "conv_state",
                "conv_w",
                "q_scale",
                "A_log",
                "dt_bias",
                "state_in",
                "norm_w",
                "eps",
            ],
            output_names=["conv_out", "window", "state_out"]
            + (["states"] if states else [])
            + ["out"],
            header=_QWEN4_DECODE_HEADER,
            source=_QWEN4_VERIFY_STEP_SOURCE.replace(
                "__STATES__", _QWEN4_VERIFY_STATES_WRITE if states else ""
            ),
        )
    return kernel


@functools.cache
def _qwen4_verify_step_launch(steps, dtype, state_dtype, c_dim, hk, hv, dk, dv, history=True):
    """Launch parameters of ``qwen4_verify_step_fused`` for one block shape."""
    history = history and steps > 1
    rows = dv // 8
    return _qwen4_verify_step_kernel(history), {
        "template": [
            ("T", dtype),
            ("HK", hk),
            ("HV", hv),
            ("DK", dk),
            ("DV", dv),
            ("C", c_dim),
            ("S", steps),
        ],
        "grid": (32, rows, hv),
        "threadgroup": (32, rows, 1),
        "output_shapes": [(1, 3, c_dim), (1, 3 + steps, c_dim), (1, hv, dv, dk)]
        + ([(1, steps - 1, hv, dv, dk)] if history else [])
        + [(1, steps, hv * dv)],
        "output_dtypes": [dtype, dtype, state_dtype]
        + ([state_dtype] if history else [])
        + [dtype],
    }


def qwen4_verify_step_fused(
    proj, conv_state, conv_w, q_scale, A_log, dt_bias, state, norm_w, eps, hk, hv, dk, dv,
    history=True,
):
    """``S = proj.shape[1]`` decode steps from (conv_state, state) in one launch
    (B = 1, dk = dv = 128, a 4-tap convolution: conv_state [1, 3, C]).

    Row t of ``proj`` is the stacked in-projection [qkv | z | b | a] of step t.
    Returns (next conv state, rollback window [1, 3 + S, C] = [conv_state; qkv
    rows], the states after steps 0..S-2 [1, S-1, hv, dv, dk] or None at S = 1
    or without ``history``, the state after step S-1, gated output [1, S,
    hv*dv]); step t's values are bit-identical to the (t+1)-th of S chained
    ``qwen4_decode_step_fused`` calls.
    """
    steps = proj.shape[1]
    kernel, launch = _qwen4_verify_step_launch(
        steps, proj.dtype, state.dtype, conv_state.shape[-1], hk, hv, dk, dv, history
    )
    outputs = kernel(
        inputs=[proj, conv_state, conv_w, q_scale, A_log, dt_bias, state, norm_w, eps],
        **launch,
    )
    if len(outputs) == 5:
        conv_out, window, state_out, states, out = outputs
        return conv_out, window, states, state_out, out
    conv_out, window, state_out, out = outputs
    return conv_out, window, None, state_out, out


def configure_qwen4_decode(model, *, wide_projections: bool) -> None:
    """Capture the decode setting per layer when the model is loaded."""
    for module in model.modules():
        if (
            type(module).__name__ == "Qwen4ExpGatedDeltaNet"
            and type(module).__module__ == "mlx_vlm.models.qwen4_exp.language"
        ):
            module._omlx_qwen4_wide_projections = wide_projections


_ALLOWED_BITS = frozenset({2, 3, 4, 5, 6, 8})
_ALLOWED_GROUPS = frozenset({32, 64, 128})
_QWEN4_HIDDEN_SIZE = 2560


def _qwen4_gdn_geometry_ok(module) -> bool:
    """Require the supported Qwen4 GDN geometry, convolution and gate weights."""

    conv_dim = 2 * 16 * 128 + 48 * 128
    if (
        type(module).__name__ != "Qwen4ExpGatedDeltaNet"
        or type(module).__module__ != "mlx_vlm.models.qwen4_exp.language"
        or module.training
        or module.num_k_heads != 16
        or module.num_v_heads != 48
        or module.head_k_dim != 128
        or module.head_v_dim != 128
        or module.conv_kernel_size != 4
    ):
        return False

    conv = getattr(module, "conv1d", None)
    norm = getattr(module, "norm", None)
    return not (
        conv is None
        or getattr(conv, "bias", None) is not None
        or conv.weight.shape != (conv_dim, 4, 1)
        or conv.weight.dtype != mx.bfloat16
        or norm is None
        or getattr(norm, "activation", None) != "sigmoid"
        or norm.weight.shape != (128,)
        or norm.weight.dtype != mx.bfloat16
        or module.A_log.shape != (48,)
        or module.A_log.dtype != mx.bfloat16
        or module.dt_bias.shape != (48,)
        or module.dt_bias.dtype != mx.bfloat16
    )


def _canonical_projection(linear, rows, signatures, in_dim=_QWEN4_HIDDEN_SIZE):
    # The q4 prefill routing reclasses these projections to a QuantizedLinear
    # subclass; the fused decode reads their packed storage, not their forward.
    if not isinstance(linear, nn.QuantizedLinear) or linear.mode != "affine":
        return False
    signature = (linear.bits, linear.group_size)
    if signatures is not None and signature not in signatures:
        return False
    if signature[0] not in _ALLOWED_BITS or signature[1] not in _ALLOWED_GROUPS:
        return False
    bits, group_size = signature
    if in_dim % group_size:
        return False
    packed_cols = in_dim * bits // 32
    scale_cols = in_dim // group_size
    return (
        linear.weight.shape == (rows, packed_cols)
        and linear.weight.dtype == mx.uint32
        and linear.scales.shape == (rows, scale_cols)
        and linear.scales.dtype == mx.bfloat16
        and linear.biases is not None
        and linear.biases.shape == (rows, scale_cols)
        and linear.biases.dtype == mx.bfloat16
        and "bias" not in linear
    )


def _qwen4_decode_static_check(module) -> bool:
    """Require the supported Qwen4 geometry and canonical affine storage."""

    if not _qwen4_gdn_geometry_ok(module):
        return False
    conv_dim = 2 * 16 * 128 + 48 * 128

    # The shipped oQe allocation is intentionally mixed per tensor.  This
    # kernel begins after those projections, so accept only the exact
    # canonical layouts emitted by the converter rather than demanding that
    # all four happen to share layer 0's q6/g64 allocation.  ``wide`` lifts the
    # recipe allow-list only; every shape/dtype/bias check above still applies.
    wide = getattr(module, "_omlx_qwen4_wide_projections", False)
    qkv_signatures = None if wide else {(4, 64), (5, 64), (6, 64)}
    aux_signatures = None if wide else {(5, 128), (6, 64)}
    if not _canonical_projection(
        module.in_proj_qkv,
        conv_dim,
        qkv_signatures,
    ):
        return False
    for linear, rows in (
        (module.in_proj_z, 48 * 128),
        (module.in_proj_b, 48),
        (module.in_proj_a, 48),
    ):
        if not _canonical_projection(linear, rows, aux_signatures):
            return False

    out = module.out_proj
    # out_proj sits after the fused norm/gate, so its allocation cannot reach
    # the fused kernels at all; in the opt-in arm only the canonical-layout
    # checks remain. Its input is the concatenated value stream, not the
    # residual stream.
    out_signatures = None if wide else {(5, 128)}
    return _canonical_projection(
        out,
        _QWEN4_HIDDEN_SIZE,
        out_signatures,
        in_dim=module.num_v_heads * module.head_v_dim,
    )


def _one_row_fused_projections(linears):
    """Operands of mlx-vlm's one-row fused in-projection, or None when it keeps four.

    ``_target_verify_linears`` runs one-row bf16 decode through
    ``_decode_quantized_linears_fused``: one ``quantized_matmul`` over the
    row-concatenated projections, split back, when all four share bits, group
    size and mode. This mirrors its gate (the input is always one bf16 row
    here) and shares its concatenation cache on the first projection.
    """
    first = linears[0]
    if not all(
        isinstance(linear, nn.QuantizedLinear)
        and linear.bits == first.bits
        and linear.group_size == first.group_size
        and linear.mode == first.mode
        and linear.biases is not None
        and linear.scales.dtype == mx.bfloat16
        and linear.biases.dtype == mx.bfloat16
        and "bias" not in linear
        for linear in linears
    ):
        return None
    cache_key = tuple(
        (id(linear.weight), id(linear.scales), id(linear.biases)) for linear in linears
    )
    cached = getattr(first, "_fused_decode_linears", None)
    if cached is None or cached[0] != cache_key:
        weights = mx.concatenate([linear.weight for linear in linears], axis=0)
        scales = mx.concatenate([linear.scales for linear in linears], axis=0)
        biases = mx.concatenate([linear.biases for linear in linears], axis=0)
        split_indices = []
        offset = 0
        for linear in linears[:-1]:
            offset += linear.weight.shape[0]
            split_indices.append(offset)
        mx.eval(weights, scales, biases)
        cached = (cache_key, weights, scales, biases, split_indices)
        first._fused_decode_linears = cached
    _, weights, scales, biases, split_indices = cached
    return (
        weights,
        scales,
        biases,
        split_indices,
        first.group_size,
        first.bits,
        first.mode,
    )


# Output columns per simdgroup of the planned one-row projections (stock
# qmv_fast owns 4): the tiles that measured fastest in a chain of 36 decode
# layers on M5 Ultra.
_QWEN4_IN_QMV_RPS = 2
_QWEN4_OUT_QMV_RPS = 1


def _build_qwen4_decode_plan(module):
    """Resolved operands of the fused B1/T1 decode, or None when ineligible."""
    if not _qwen4_decode_static_check(module):
        return None
    projections = (
        module.in_proj_qkv,
        module.in_proj_z,
        module.in_proj_b,
        module.in_proj_a,
    )
    fused = _one_row_fused_projections(projections)
    in_qmv = None
    if fused is not None:
        weights, scales, biases, _, group_size, bits, mode = fused
        in_qmv = one_row_qmv(
            weights, scales, biases, bits, group_size, mode, mx.bfloat16, _QWEN4_IN_QMV_RPS
        )
    out = module.out_proj
    out_qmv = one_row_qmv(
        out.weight,
        out.scales,
        out.biases,
        out.bits,
        out.group_size,
        out.mode,
        mx.bfloat16,
        _QWEN4_OUT_QMV_RPS,
    )
    return (
        projections,
        fused,
        module.conv1d.weight,
        mx.array(module.head_k_dim**-0.5, dtype=mx.bfloat16),
        module.A_log,
        module.dt_bias,
        module.norm.weight,
        mx.array(module.norm.eps, dtype=mx.float32),
        out,
        module.num_k_heads,
        module.num_v_heads,
        module.head_k_dim,
        module.head_v_dim,
        in_qmv,
        out_qmv,
    )


_QWEN4_GDN_CLASSES: dict = {}


def _is_qwen4_gdn(module) -> bool:
    """Whether ``module`` is an ``nn.Module`` of the loaded Qwen4 GDN class."""
    cls = type(module)
    qwen4 = _QWEN4_GDN_CLASSES.get(cls)
    if qwen4 is None:
        qwen4 = _QWEN4_GDN_CLASSES[cls] = (
            issubclass(cls, nn.Module)
            and cls.__name__ == "Qwen4ExpGatedDeltaNet"
            and cls.__module__ == "mlx_vlm.models.qwen4_exp.language"
        )
    return qwen4


def _qwen4_decode_plan(module):
    """The cached decode plan of an ``nn.Module`` Qwen4 GDN layer, else None."""
    if not _is_qwen4_gdn(module):
        return None
    return cached_per_module(
        module,
        "_omlx_qwen4_decode_plan",
        _build_qwen4_decode_plan,
        flags=(module.__dict__.get("_omlx_qwen4_wide_projections", False),),
    )


def _qwen4_decode_static_eligible(module) -> bool:
    """Require the supported Qwen4 geometry and canonical affine storage."""

    if _QWEN4_DECODE_PLAN_ENABLED and isinstance(module, nn.Module):
        return _qwen4_decode_plan(module) is not None
    return _qwen4_decode_static_check(module)


def _qwen4_decode_state_eligible(inputs, cache) -> bool:
    """One bf16 row against a canonical, unpadded conv and recurrent state."""
    if (
        not isinstance(inputs, mx.array)
        or inputs.shape != (1, 1, 2560)
        or inputs.dtype != mx.bfloat16
        or getattr(cache, "lengths", None) is not None
        or getattr(cache, "left_padding", None) is not None
    ):
        return False
    conv_state = cache[0]
    recurrent_state = cache[1]
    return (
        isinstance(conv_state, mx.array)
        and conv_state.shape == (1, 3, 10240)
        and conv_state.dtype == mx.bfloat16
        and isinstance(recurrent_state, mx.array)
        and recurrent_state.shape == (1, 48, 128, 128)
        and recurrent_state.dtype == mx.float32
    )


def _qwen4_verify_in_geometry(rows):
    # (Columns per simdgroup, rows per threadgroup) of the stacked
    # in-projection: the one-row decode tile, and for verify blocks rows share
    # each decoded weight tile three or two at a time (fastest over a chain of
    # 36 layers on M5 Ultra at 2..4 rows; two rows prefer one column).
    if rows == 2:
        return 1, 2
    return _QWEN4_IN_QMV_RPS, next(d for d in (3, 2, 1) if rows % d == 0)


def _qwen4_verify_out_geometry(rows):
    # The one-row decode tile, and stock qmv_fast's four columns per
    # simdgroup (one row per threadgroup) for verify blocks.
    return (_QWEN4_OUT_QMV_RPS, 1) if rows == 1 else (4, 1)


# The same (columns per simdgroup, rows per threadgroup) pairs for the
# unrolled tile, per block size: the fastest over a chain of 36 layers on M5
# Ultra (oQ5e: 6-bit in-, 5-bit out-projection), each inside the unrolled
# tile's register envelope at every bit width the kernel serves (at most 16
# values per lane). Other block sizes keep the pairs above.
_QWEN4_VERIFY_IN_TILES = {2: (1, 2), 3: (2, 3), 4: (4, 2), 5: (2, 5), 6: (4, 3), 7: (8, 1), 8: (4, 2)}
_QWEN4_VERIFY_OUT_TILES = {2: (4, 1), 3: (4, 3), 4: (2, 4), 5: (2, 5), 6: (4, 3), 7: (4, 1), 8: (2, 4)}


def _qwen4_verify_in_tile(rows):
    return _QWEN4_VERIFY_IN_TILES.get(rows) or _qwen4_verify_in_geometry(rows)


def _qwen4_verify_out_tile(rows):
    return _QWEN4_VERIFY_OUT_TILES.get(rows) or _qwen4_verify_out_geometry(rows)


def _build_qwen4_verify_plan(module):
    """Resolved operands of the fused speculative verify, or None when ineligible.

    The verify serves every canonical affine layout whose four in-projections
    share one allocation (its projections run per row with one-row qmv_fast
    bits, whatever the decode allow-list admits)."""
    if not _qwen4_gdn_geometry_ok(module):
        return None
    projections = (
        module.in_proj_qkv,
        module.in_proj_z,
        module.in_proj_b,
        module.in_proj_a,
    )
    rows = (2 * 16 * 128 + 48 * 128, 48 * 128, 48, 48)
    out = module.out_proj
    if not (
        all(_canonical_projection(linear, n, None) for linear, n in zip(projections, rows))
        and _canonical_projection(out, _QWEN4_HIDDEN_SIZE, None, in_dim=48 * 128)
    ):
        return None
    fused = _one_row_fused_projections(projections)
    if fused is None:
        return None
    weights, scales, biases, _, group_size, bits, mode = fused
    tiles = _QWEN4_VERIFY_TILES
    in_rows = rows_qmv(
        weights, scales, biases, bits, group_size, mode, mx.bfloat16,
        _qwen4_verify_in_tile if tiles else _qwen4_verify_in_geometry,
        unrolled=tiles,
    )
    out_rows = rows_qmv(
        out.weight, out.scales, out.biases, out.bits, out.group_size, out.mode,
        mx.bfloat16, _qwen4_verify_out_tile if tiles else _qwen4_verify_out_geometry,
        unrolled=tiles,
    )
    if in_rows is None or out_rows is None:
        return None
    return (
        in_rows,
        out_rows,
        module.conv1d.weight,
        mx.array(module.head_k_dim**-0.5, dtype=mx.bfloat16),
        module.A_log,
        module.dt_bias,
        module.norm.weight,
        mx.array(module.norm.eps, dtype=mx.float32),
        module.num_k_heads,
        module.num_v_heads,
        module.head_k_dim,
        module.head_v_dim,
    )


def _qwen4_verify_plan(module):
    """The cached verify plan of an ``nn.Module`` Qwen4 GDN layer, else None."""
    if not _is_qwen4_gdn(module):
        return None
    return cached_per_module(module, "_omlx_qwen4_verify_plan", _build_qwen4_verify_plan)


def _qwen4_verify_state_eligible(inputs, cache) -> bool:
    """B1 bf16 rows against a canonical conv and recurrent state."""
    if not (
        isinstance(inputs, mx.array)
        and inputs.ndim == 3
        and inputs.shape[0] == 1
        and inputs.shape[2] == _QWEN4_HIDDEN_SIZE
        and inputs.dtype == mx.bfloat16
    ):
        return False
    conv_state = cache[0]
    recurrent_state = cache[1]
    return (
        isinstance(conv_state, mx.array)
        and conv_state.shape == (1, 3, 10240)
        and conv_state.dtype == mx.bfloat16
        and isinstance(recurrent_state, mx.array)
        and recurrent_state.shape == (1, 48, 128, 128)
        and recurrent_state.dtype == mx.float32
    )


def _qwen4_verify(plan, inputs, cache):
    """Fused verify rows on a cached plan: bit-identical outputs, next states
    and committed rollback states (the conv window, and the per-step
    recurrent states or their deferred recomputation) to the per-op verify
    path below."""
    (
        in_rows,
        out_rows,
        conv_w,
        q_scale,
        A_log,
        dt_bias,
        norm_w,
        eps,
        hk,
        hv,
        dk,
        dv,
    ) = plan
    proj = in_rows(inputs)
    steps = proj.shape[1]
    conv_start, start = cache[0], cache[1]
    deferred = (
        _QWEN4_VERIFY_DEFERRED_STATES
        and steps > 1
        and qwen35_gdn_verify_fused.deferred_states_ready(cache, 1, steps)
    )
    conv_state, window, states, state, gated = qwen4_verify_step_fused(
        proj, conv_start, conv_w, q_scale, A_log, dt_bias, start, norm_w,
        eps, hk, hv, dk, dv, not deferred,
    )
    cache.record_speculative_window(0, window, 3)
    cache[0] = conv_state
    if deferred:
        qwen35_gdn_verify_fused.record_deferred_states(
            cache, 1, start, state, steps,
            functools.partial(
                _qwen4_state_after, proj, conv_start, conv_w, q_scale, A_log, dt_bias,
                start, norm_w, eps, hk, hv, dk, dv,
            ),
        )
    else:
        cache.record_speculative_states(1, states, state)
    cache[1] = state
    return out_rows(gated)


def _qwen4_state_after(
    proj, conv_state, conv_w, q_scale, A_log, dt_bias, state, norm_w, eps, hk, hv, dk, dv,
    keep,
):
    """The recurrent state after the first ``keep`` rows of a verify block:
    the fused step on those rows, whose last state is the block's state
    after row ``keep - 1`` bit for bit (row t's arithmetic does not depend
    on the block length)."""
    return qwen4_verify_step_fused(
        proj[:, :keep], conv_state, conv_w, q_scale, A_log, dt_bias, state, norm_w,
        eps, hk, hv, dk, dv, False,
    )[3]


def _qwen4_batch_decode_state_eligible(inputs, cache) -> bool:
    """2..16 bf16 one-token rows against canonical per-row conv and
    recurrent states, outside a speculative transaction."""
    if not (
        isinstance(inputs, mx.array)
        and inputs.ndim == 3
        and 2 <= inputs.shape[0] <= _QWEN4_BATCH_DECODE_MAX_ROWS
        and inputs.shape[1] == 1
        and inputs.shape[2] == _QWEN4_HIDDEN_SIZE
        and inputs.dtype == mx.bfloat16
        and getattr(cache, "lengths", None) is None
        and not getattr(cache, "is_speculating", False)
    ):
        return False
    rows = inputs.shape[0]
    conv_state = cache[0]
    recurrent_state = cache[1]
    return (
        isinstance(conv_state, mx.array)
        and conv_state.shape == (rows, 3, 10240)
        and conv_state.dtype == mx.bfloat16
        and isinstance(recurrent_state, mx.array)
        and recurrent_state.shape == (rows, 48, 128, 128)
        and recurrent_state.dtype == mx.float32
    )


def _qwen4_batch_decode(module, plan, inputs, cache):
    """A batched one-token decode: the in-projection, one launch of the
    fused decode step for every row, and the out-projection.

    Up to ``_QWEN4_BATCH_ROW_EXACT_ROWS`` rows take the verify plan's per-row
    projections (one-row qmv bits), so each row is bit-identical to that row
    decoded alone; wider batches take one stock quantized matmul per
    projection, which reads the weights once for all rows (faster from four
    rows on, M5 Ultra)."""
    (
        projections, fused, conv_w, q_scale, A_log, dt_bias, norm_w, eps, out_proj,
        hk, hv, dk, dv, _, _,
    ) = plan
    rows = inputs.shape[0]
    verify = _qwen4_verify_plan(module) if rows <= _QWEN4_BATCH_ROW_EXACT_ROWS else None
    if verify is not None:
        c_dim = 2 * hk * dk + hv * dv
        proj = verify[0](inputs.reshape(1, rows, -1)).reshape(rows, 1, -1)
        split = [c_dim, c_dim + hv * dv, c_dim + hv * dv + hv]
        mixed_qkv, z, b, a = mx.split(proj, split, axis=-1)
    elif fused is not None:
        weights, w_scales, w_biases, split, group_size, bits, mode = fused
        proj = mx.quantized_matmul(
            inputs, weights, scales=w_scales, biases=w_biases, transpose=True,
            group_size=group_size, bits=bits, mode=mode,
        )
        mixed_qkv, z, b, a = mx.split(proj, split, axis=-1)
    else:
        mixed_qkv, z, b, a = (linear(inputs) for linear in projections)
    conv_state, state, gated = qwen4_batch_decode_step_fused(
        mixed_qkv, z, b, a, cache[0], conv_w, q_scale, A_log, dt_bias, cache[1], norm_w,
        eps, hk, hv, dk, dv,
    )
    if verify is not None:
        result = verify[1](gated.reshape(1, rows, -1)).reshape(rows, 1, -1)
    else:
        result = out_proj(gated)
    cache[0], cache[1] = conv_state, state
    if hasattr(cache, "advance"):
        from mlx_vlm.models.qwen3_5 import language as q35

        cache.advance(1)
        q35._qwen3_5_advance_lengths_info(cache, 1)
    return result


def _qwen4_decode_dynamic_eligible(
    module,
    inputs,
    mask,
    cache,
    gdn_sink,
    target_verify,
) -> bool:
    return (
        not target_verify
        and gdn_sink is None
        and mask is None
        and cache is not None
        and _qwen4_decode_state_eligible(inputs, cache)
        and _qwen4_decode_static_eligible(module)
    )


def _qwen35_decode_eligible(module, inputs, mask, cache) -> bool:
    """Check the fused kernel shape, precision and cache requirements."""
    from mlx_vlm.models.cache import ArraysCache
    from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet

    if (
        type(module) is not Qwen3_5GatedDeltaNet
        or module.training
        or not isinstance(inputs, mx.array)
        or inputs.shape != (1, 1, module.hidden_size)
        or inputs.dtype not in (mx.float16, mx.bfloat16)
        or mx.default_device() != mx.gpu
        or mask is not None
        or type(cache) is not ArraysCache
        or len(cache.cache) != 2
        or cache.is_speculating
        or cache.lengths is not None
        or cache.left_padding is not None
        or module.head_k_dim != 128
        or module.head_v_dim != 128
        or module.conv_kernel_size != 4
        or module.conv1d.weight.shape != (module.conv_dim, 4, 1)
        or module.conv1d.weight.dtype != inputs.dtype
        or getattr(module.conv1d, "bias", None) is not None
    ):
        return False
    return (
        isinstance(cache[0], mx.array)
        and cache[0].shape == (1, 3, module.conv_dim)
        and cache[0].dtype == inputs.dtype
        and isinstance(cache[1], mx.array)
        and cache[1].shape == (1, module.num_v_heads, 128, 128)
        and cache[1].dtype == mx.float32
    )


def _qwen4_l2_norm_sites():
    """Return the (verifier, layer) normalize functions of the loaded qwen4_exp.

    Looked up in sys.modules only: importing qwen4_exp here would pin the
    upstream copy before the compat vendor registers its own.
    """
    q4_lang = sys.modules.get("mlx_vlm.models.qwen4_exp.language")
    if q4_lang is None:
        return None
    verifier_cls = getattr(q4_lang, "_Qwen4Verifier", None) or getattr(
        q4_lang, "Qwen4ExpBatchInvariantForward", None
    )
    gdn_cls = getattr(q4_lang, "Qwen4ExpGatedDeltaNet", None)
    if verifier_cls is None or gdn_cls is None:
        return None
    return (
        verifier_cls._normalize_gated_delta_qk,
        getattr(gdn_cls, "_normalize_qk", None),
    )


def _qwen35_normalize_qk(self, q, k):
    return normalize_qk(q, k, inv_scale=k.shape[-1] ** -0.5, eps=1e-6)


def _qwen35_verify_normalize_qk(layer, q, k):
    del layer
    return normalize_qk(q, k, inv_scale=k.shape[-1] ** -0.5, eps=1e-6)


def apply_qwen35_vlm_qk_norm_patch() -> bool:
    """Normalize mlx-vlm Qwen3.5 GDN q/k like mlx-lm ``normalize_qk``.

    mlx-vlm adds the l2norm eps to mean(x^2) instead of sum(x^2), which
    differs from the reference model and from the mlx-lm path. The fused
    prework kernel above implements the patched form.
    """
    from mlx_vlm.models.qwen3_5 import language as q35
    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward

    gdn_cls = q35.Qwen3_5GatedDeltaNet
    if gdn_cls.__dict__.get("_normalize_qk") is _qwen35_normalize_qk:
        return False
    gdn_cls._normalize_qk = _qwen35_normalize_qk
    Qwen3_5BatchInvariantForward._normalize_gated_delta_qk = staticmethod(
        _qwen35_verify_normalize_qk
    )
    logger.info("mlx-vlm Qwen3.5 GDN q/k normalization follows mlx-lm normalize_qk")
    return True


def apply_qwen35_gdn_prework_patch() -> bool:
    """Install fused prework at ordinary decode and speculative entry points."""
    global _PATCHED
    # The fused kernel assumes the patched q/k normalization.
    apply_qwen35_vlm_qk_norm_patch()
    if _PATCHED:
        return True
    if not mx.metal.is_available():
        return False

    from mlx_vlm.models.qwen3_5 import language as q35
    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward
    from mlx_vlm.speculative.ops.linear import _target_verify_linears

    cls = q35.Qwen3_5GatedDeltaNet
    original = cls.__call__
    original_verify = Qwen3_5BatchInvariantForward._gated_delta
    scales = {
        dtype: (mx.array(128**-1, dtype=dtype), mx.array(128**-0.5, dtype=dtype))
        for dtype in (mx.float16, mx.bfloat16)
    }

    def qwen4_decode(self, plan, inputs, cache):
        """The fused B1/T1 decode on a cached plan, bit-identical to the per-call
        path below: in-projection, one step launch (prework, recurrence and
        norm-gate) and out-projection, each switchable back to its stock form."""
        (
            projections,
            fused,
            conv_w,
            q_scale,
            A_log,
            dt_bias,
            norm_w,
            eps,
            out_proj,
            hk,
            hv,
            dk,
            dv,
            in_qmv,
            out_qmv,
        ) = plan
        if fused is None:
            mixed_qkv, z, b, a = (linear(inputs) for linear in projections)
        else:
            weights, w_scales, w_biases, split_indices, group_size, bits, mode = fused
            if in_qmv is not None and _QWEN4_DECODE_QMV:
                projected = in_qmv(inputs)
            else:
                projected = mx.quantized_matmul(
                    inputs,
                    weights,
                    scales=w_scales,
                    biases=w_biases,
                    transpose=True,
                    group_size=group_size,
                    bits=bits,
                    mode=mode,
                )
            mixed_qkv, z, b, a = mx.split(projected, split_indices, axis=-1)
        if _QWEN4_DECODE_STEP_FUSED:
            conv_state, state, gated = qwen4_decode_step_fused(
                mixed_qkv, z, b, a, cache[0], conv_w, q_scale, A_log, dt_bias,
                cache[1], norm_w, eps, hk, hv, dk, dv,
            )
        else:
            q, k, v, conv_state, g, beta = qwen4_decode_prework_fused(
                mixed_qkv, cache[0], conv_w, q_scale, b, a, A_log, dt_bias, hk, hv, dk, dv
            )
            out, state = _qwen4_decode_recurrence(q, k, v, g, beta, cache[1])
            gated = _qwen4_norm_gate(out, z, norm_w, eps, hv, dv)
        if out_qmv is not None and _QWEN4_DECODE_QMV:
            result = out_qmv(gated)
        else:
            result = out_proj(gated)
        cache[0], cache[1] = conv_state, state
        if hasattr(cache, "advance"):
            cache.advance(1)
            q35._qwen3_5_advance_lengths_info(cache, 1)
        global _QWEN4_DECODE_ENGAGED_LOGGED
        if not _QWEN4_DECODE_ENGAGED_LOGGED:
            _QWEN4_DECODE_ENGAGED_LOGGED = True
            logger.info("Qwen4 fused B1/T1 GDN decode prework and norm-gate engaged")
        return result

    def decode(self, inputs, mask=None, cache=None, **kwargs):
        # Runtime extensions (for example capture/verification keywords)
        # must keep their original implementation and cache semantics.
        if kwargs:
            return original(self, inputs, mask=mask, cache=cache, **kwargs)
        # Qwen4 B1/T1 decode first: its one-row input is below the prefill
        # floor and its class is not Qwen3_5GatedDeltaNet, so the two gates
        # below could not claim it.
        if _QWEN4_DECODE_PLAN_ENABLED and mask is None and cache is not None:
            plan = _qwen4_decode_plan(self)
            if plan is not None and _qwen4_decode_state_eligible(inputs, cache):
                return qwen4_decode(self, plan, inputs, cache)
            if (
                _QWEN4_BATCH_DECODE
                and plan is not None
                and _qwen4_batch_decode_state_eligible(inputs, cache)
            ):
                return _qwen4_batch_decode(self, plan, inputs, cache)
        if _qwen4_prefill_eligible(self, inputs, mask, cache):
            return _qwen4_prefill(self, inputs, cache)
        if _qwen35_decode_eligible(self, inputs, mask, cache):
            mixed_qkv = self.in_proj_qkv(inputs)
            # The kernel reads projections, convolution weights and state as one dtype.
            if mixed_qkv.dtype != inputs.dtype or mixed_qkv.shape != (
                1,
                1,
                self.conv_dim,
            ):
                return original(self, inputs, mask=mask, cache=cache)
            z = self.in_proj_z(inputs).reshape(1, 1, self.num_v_heads, self.head_v_dim)
            b, a = self._project_gates(inputs)
            q_scale, k_scale = scales[inputs.dtype]
            q, k, v, conv_state = gdn_prework_fused(
                mixed_qkv,
                cache[0],
                self.conv1d.weight,
                q_scale,
                k_scale,
                self.num_k_heads,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
            )
            # An ordinary ArraysCache has no speculative history to record.
            # Compute both next states first, then commit them together;
            # never retry stock code against a partially advanced cache.
            out, state = q35.gated_delta_update(
                q,
                k,
                v,
                a,
                b,
                self.A_log,
                self.dt_bias,
                state=cache[1],
                use_kernel=True,
            )
            out = self.norm(out, z)
            result = self.out_proj(out.reshape(1, 1, -1))
            cache[0], cache[1] = conv_state, state
            cache.advance(1)
            q35._qwen3_5_advance_lengths_info(cache, 1)
            global _QWEN35_DECODE_ENGAGED_LOGGED
            if not _QWEN35_DECODE_ENGAGED_LOGGED:
                _QWEN35_DECODE_ENGAGED_LOGGED = True
                logger.info("Qwen B1/T1 fused GDN prework engaged")
            return result

        if not _qwen4_decode_dynamic_eligible(self, inputs, mask, cache, None, False):
            return original(self, inputs, mask=mask, cache=cache)
        mixed_qkv, z, b, a = _target_verify_linears(
            (self.in_proj_qkv, self.in_proj_z, self.in_proj_b, self.in_proj_a), inputs
        )
        q, k, v, conv_state, g, beta = qwen4_decode_prework_fused(
            mixed_qkv,
            cache[0],
            self.conv1d.weight,
            mx.array(self.head_k_dim**-0.5, dtype=mx.bfloat16),
            b,
            a,
            self.A_log,
            self.dt_bias,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
        )
        out, state = _qwen4_decode_recurrence(q, k, v, g, beta, cache[1])
        flat = qwen4_decode_norm_gate_fused(
            out,
            z,
            self.norm.weight,
            hv=self.num_v_heads,
            dv=self.head_v_dim,
            eps=self.norm.eps,
        )
        result = self.out_proj(flat)
        cache[0], cache[1] = conv_state, state
        if hasattr(cache, "advance"):
            cache.advance(1)
            q35._qwen3_5_advance_lengths_info(cache, 1)
        global _QWEN4_DECODE_ENGAGED_LOGGED
        if not _QWEN4_DECODE_ENGAGED_LOGGED:
            _QWEN4_DECODE_ENGAGED_LOGGED = True
            logger.info("Qwen4 fused B1/T1 GDN decode prework and norm-gate engaged")
        return result

    def verify(verifier, layer, inputs, mask, cache):
        length = inputs.shape[1]
        # The fused prework emits either the stock Qwen3.5 RMS scaling or
        # the stock Qwen4 L2 scaling (L2 kernel variant), bit-exact to the
        # normalize implementation each verifier class installs.
        compatible_norm = (
            type(verifier)._normalize_gated_delta_qk
            is Qwen3_5BatchInvariantForward._normalize_gated_delta_qk
        )
        l2_norm = False
        if not compatible_norm:
            sites = _qwen4_l2_norm_sites()
            l2_norm = (
                sites is not None
                and type(verifier)._normalize_gated_delta_qk is sites[0]
                and sites[1] is not None
                and getattr(type(layer), "_normalize_qk", None) is sites[1]
            )
        # Qwen4 B1 rows: the fused verify reproduces the per-op path below bit
        # for bit. Its projections run one-row qmv_fast arithmetic per row, as
        # the verifier's do at one row and in the armed row-exact mode. The
        # plan pins the layer geometry and weights, the state gate the inputs
        # and both cache states.
        if (
            l2_norm
            and _QWEN4_VERIFY_FUSED
            and mask is None
            and cache is not None
            and cache.is_speculating
            and cache.lengths is None
            and 1 <= length <= _QWEN4_VERIFY_MAX_ROWS
            and (length == 1 or is_row_exact_armed())
            and _qwen4_verify_state_eligible(inputs, cache)
        ):
            plan = _qwen4_verify_plan(layer)
            if plan is not None:
                result = _qwen4_verify(plan, inputs, cache)
                if hasattr(cache, "advance"):
                    cache.advance(length)
                    q35._qwen3_5_advance_lengths_info(cache, length)
                global _QWEN4_VERIFY_ENGAGED_LOGGED
                if not _QWEN4_VERIFY_ENGAGED_LOGGED:
                    _QWEN4_VERIFY_ENGAGED_LOGGED = True
                    logger.info("[gdn-prework] Qwen4 fused verify engaged (S=%d)", length)
                return result
        # Qwen4 one-row MTP steps take the fused prework too, so their q/k/v
        # match the fused B1/T1 decode kernel (same L2 arithmetic).
        min_length = 1 if l2_norm else 2
        if not (
            (compatible_norm or l2_norm)
            and cache is not None
            and cache.is_speculating
            and min_length <= length <= 9
            and mask is None
            and inputs.dtype in (mx.bfloat16, mx.float16)
            and layer.conv_kernel_size == 4
            and layer.head_k_dim == 128
            and layer.head_v_dim == 128
            and cache.lengths is None
            and cache[0] is not None
            and cache[0].shape[0] == inputs.shape[0]
            and cache[0].dtype == inputs.dtype
            and layer.conv1d.weight.dtype == inputs.dtype
            and getattr(layer.conv1d, "bias", None) is None
        ):
            global _VERIFY_REJECT_DIAG
            # Only verify-width calls can engage; skip S=1 decode probes.
            if cache is not None and length >= min_length and _VERIFY_REJECT_DIAG < 3:
                _VERIFY_REJECT_DIAG += 1
                failed = [
                    name
                    for name, ok in (
                        ("norm", compatible_norm or l2_norm),
                        ("speculating", cache.is_speculating),
                        ("length", min_length <= length <= 9),
                        ("mask", mask is None),
                        ("inputs_dtype", inputs.dtype in (mx.bfloat16, mx.float16)),
                        ("conv_kernel", layer.conv_kernel_size == 4),
                        ("dk128", layer.head_k_dim == 128),
                        ("dv128", layer.head_v_dim == 128),
                        ("lengths", cache.lengths is None),
                        (
                            "c0",
                            cache[0] is not None
                            and cache[0].shape[0] == inputs.shape[0]
                            and cache[0].dtype == inputs.dtype,
                        ),
                        (
                            "conv_w",
                            layer.conv1d.weight.dtype == inputs.dtype
                            and getattr(layer.conv1d, "bias", None) is None,
                        ),
                    )
                    if not ok
                ]
                logger.info(
                    "[gdn-prework] verify gate reject: %s (verifier=%s S=%d l2=%s)",
                    failed,
                    type(verifier).__name__,
                    length,
                    l2_norm,
                )
            return original_verify(verifier, layer, inputs, mask, cache)
        mixed_qkv, z, b, a = verifier._linears(
            (layer.in_proj_qkv, layer.in_proj_z, layer.in_proj_b, layer.in_proj_a),
            inputs,
        )
        inv = layer.head_k_dim**-0.5
        if l2_norm:
            q_scale = mx.array(inv, dtype=inputs.dtype)
            k_scale = mx.array(1.0, dtype=inputs.dtype)
        else:
            q_scale = mx.array(inv * inv, dtype=inputs.dtype)
            k_scale = mx.array(inv, dtype=inputs.dtype)
        q, k, v, conv_state = gdn_prework_fused(
            mixed_qkv,
            cache[0],
            layer.conv1d.weight,
            q_scale,
            k_scale,
            layer.num_k_heads,
            layer.num_v_heads,
            layer.head_k_dim,
            layer.head_v_dim,
            l2=l2_norm,
        )
        fused = not l2_norm and qwen35_gdn_verify_fused.fused_eligible(
            layer, q, cache, length
        )
        if fused and 0 not in cache._speculation["records"]:
            qwen35_gdn_verify_fused.record_window(
                cache, 0, cache[0], mixed_qkv, layer.conv_kernel_size - 1
            )
        else:
            conv_input = mx.concatenate([cache[0], mixed_qkv], axis=1)
            cache.record_speculative_window(0, conv_input, layer.conv_kernel_size - 1)
        cache[0] = conv_state
        if fused:
            out = qwen35_gdn_verify_fused.verify_block(layer, cache, q, k, v, a, b, z)
        else:
            out, _ = q35.gated_delta_update(
                q,
                k,
                v,
                a,
                b,
                layer.A_log,
                layer.dt_bias,
                cache=cache,
                use_kernel=not layer.training,
            )
        if hasattr(cache, "advance"):
            cache.advance(length)
            q35._qwen3_5_advance_lengths_info(cache, length)
        if not fused:
            out = layer.norm(
                out, z.reshape(inputs.shape[0], length, -1, layer.head_v_dim)
            ).reshape(inputs.shape[0], length, -1)
        result = verifier._linear(layer.out_proj, out)
        global _ENGAGED_LOGGED
        if not _ENGAGED_LOGGED:
            _ENGAGED_LOGGED = True
            logger.info(
                "[gdn-prework] fused verify prework engaged (S=%d, l2=%s)",
                length,
                l2_norm,
            )
        return result

    cls.__call__ = decode
    cls._omlx_gdn_prework_patched = True
    Qwen3_5BatchInvariantForward._gated_delta = verify
    qwen35_gdn_verify_fused.apply_arrays_cache_replay_patch()
    _PATCHED = True
    logger.info("Qwen fused GDN prework patch applied")
    return True
