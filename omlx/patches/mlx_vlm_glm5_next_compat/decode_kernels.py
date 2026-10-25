# SPDX-License-Identifier: Apache-2.0
"""Exact fused decode/verify kernels for GLM-5.3-Flash (glm5_next).

Single-token decode and short verify blocks (L <= 8) are dominated by
thousands of tiny dependent dispatches, which cost both GPU time and host
encode time. The kernels here fuse chains of them while reproducing the
stock MLX arithmetic bit for bit:

* hyper-connections: ``hc_mix`` (fp32 RMS + mix GEMV), ``hc_expand_one``
  (the one-token comb product, NAX matmul2d or simdgroup MMA, + epilogue),
  and for one token ``hc_pre_fused`` (mix + collapse + RMSNorm in one
  dispatch, with the previous expand's epilogue folded in) plus
  ``hc_post_mm`` (post, sinkhorn comb and the comb product, off the
  dependent chain);
* MoE: ``moe_router`` (logits GEMV + sigmoid/bias, top-k select with the
  stable-sort tie order), ``moe_gate_up_swiglu`` and ``moe_down_combine``
  (routed + shared experts, clamped SwiGLU, routing-weighted sum);
* KDA linear attention: ``kda_decode_step`` (short conv, SiLU, l2norm,
  gate projections, vector-gated delta rule, RMSNormGated);
* DSA indexer: ``pool_compress``, ``dsa_decode_scores`` and ``dsa_select_topk``;
* row-exact verify: ``rows_gemv`` (float projections of a verify block,
  every row with the one-token gemv arithmetic).

Exactness rules: every reduction replays the order of the MLX kernel it
replaces (qmv/qmv_quad lane mapping, gemv shuffle ladders, row_reduce
orders, Steel/NAX MMA fragments); every intermediate is rounded where the
reference materializes it; and a product is never contracted into an add
that consumed it in a different reference kernel (separate statements or
``volatile``). ``tests/test_mlx_vlm_glm5_next_compat.py`` checks bitwise
equality against the reference op graphs, per kernel and end to end.
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from functools import lru_cache
from typing import Optional

import mlx.core as mx

from omlx.custom_kernels.nax import is_nax_available

logger = logging.getLogger(__name__)

# Successful fused dispatches by kernel family (graph-build time counts; used
# by tests and profilers to confirm the fused paths engage).
STATS: Counter = Counter()

_QMV_HEADER = r"""
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

template <int bits>
constexpr int glm_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : 32 / bits);
}

template <int bits>
constexpr int glm_bytes_per_pack() {
  return ((bits & (bits - 1)) == 0) ? 4 : (bits == 5 ? 5 : 3);
}

// Verbatim copy of MLX quantized.h load_vector (U = float).
template <typename T, int values_per_thread, int bits>
inline float glm_load_vector(const device T* x, thread float* x_thread) {
  float sum = 0;
  if (bits == 3) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]) +
          float(x[i + 4]) + float(x[i + 5]) + float(x[i + 6]) + float(x[i + 7]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  } else if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  } else if (bits == 5) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]) +
          float(x[i + 4]) + float(x[i + 5]) + float(x[i + 6]) + float(x[i + 7]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  } else if (bits == 6) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  } else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }
  return sum;
}

// Verbatim copy of MLX quantized.h qdot (U = float).
template <int values_per_thread, int bits>
inline float glm_qdot(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum) {
  float accum = 0;
  if (bits == 3) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  } else if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  } else if (bits == 5) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;
      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  } else if (bits == 6) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;
      accum += (w[0] & 0x3f) * x_thread[0];
      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);
      accum += (w[2] & 0xfc) * x_thread[3];
    }
  } else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * w[i];
    }
  }
  return scale * accum + sum * bias;
}

// qmv_fast_impl for RPS consecutive rows of one [N, K] affine matrix
// (row pointers already offset to the first row), one simdgroup.  Leaves the
// per-lane partial sums in `result`; the caller simd_sums them.
template <typename T, int K, int group_size, int bits, int RPS>
inline void glm_qmv_rows(
    const device uint8_t* ws,
    const device T* scales,
    const device T* biases,
    const device T* x,
    uint simd_lid,
    thread float* result) {
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int pack_factor = glm_pack_factor<bits>();
  constexpr int bytes_per_pack = glm_bytes_per_pack<bits>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * 32;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  constexpr int in_vec_size_w = K * bytes_per_pack / pack_factor;
  constexpr int in_vec_size_g = K / group_size;

  thread float x_thread[values_per_thread];
  ws += simd_lid * packs_per_thread * bytes_per_pack;
  scales += simd_lid / scale_step_per_thread;
  biases += simd_lid / scale_step_per_thread;
  x += simd_lid * values_per_thread;

  for (int k = 0; k < K; k += block_size) {
    float sum = glm_load_vector<T, values_per_thread, bits>(x, x_thread);
    for (int row = 0; row < RPS; row++) {
      const device uint8_t* wl = ws + row * in_vec_size_w;
      const device T* sl = scales + row * in_vec_size_g;
      const device T* bl = biases + row * in_vec_size_g;
      float s = sl[0];
      float b = bl[0];
      result[row] += glm_qdot<values_per_thread, bits>(wl, x_thread, s, b, sum);
    }
    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }
}

// Verbatim copy of MLX quantized.h dequantize (U = float) for 4/5/6/8 bits.
template <int N, int bits>
inline void glm_dequantize(const device uint8_t* w, float scale, float bias, thread float* w_local) {
  const float s = float(scale);
  const float b = float(bias);
  if (bits == 4) {
    float sc[2] = {s, s / 16.0f};
    for (int i = 0; i < (N / 2); i++) {
      w_local[2 * i] = static_cast<float>(sc[0] * (w[i] & 0x0f) + b);
      w_local[2 * i + 1] = static_cast<float>(sc[1] * (w[i] & 0xf0) + b);
    }
  } else if (bits == 5) {
    for (int i = 0; i < (N / 8); i++) {
      w_local += 8 * i;
      w += 5 * i;
      w_local[0] = static_cast<float>((w[0] & 0x1f) * s + b);
      w_local[1] =
          static_cast<float>((((w[0] & 0xe0) >> 5) + ((w[1] & 0x3) << 3)) * s + b);
      w_local[2] = static_cast<float>(((w[1] & 0x7c) >> 2) * s + b);
      w_local[3] =
          static_cast<float>((((w[1] & 0x80) >> 7) + ((w[2] & 0xf) << 1)) * s + b);
      w_local[4] =
          static_cast<float>((((w[2] & 0xf0) >> 4) + ((w[3] & 0x1) << 4)) * s + b);
      w_local[5] = static_cast<float>(((w[3] & 0x3e) >> 1) * s + b);
      w_local[6] =
          static_cast<float>((((w[3] & 0xc0) >> 6) + ((w[4] & 0x7) << 2)) * s + b);
      w_local[7] = static_cast<float>(((w[4] & 0xf8) >> 3) * s + b);
    }
  } else if (bits == 6) {
    for (int i = 0; i < (N / 4); i++) {
      w_local += 4 * i;
      w += 3 * i;
      w_local[0] = static_cast<float>((w[0] & 0x3f) * s + b);
      w_local[1] =
          static_cast<float>((((w[0] >> 6) & 0x03) + ((w[1] & 0x0f) << 2)) * s + b);
      w_local[2] =
          static_cast<float>((((w[1] >> 4) & 0x0f) + ((w[2] & 0x03) << 4)) * s + b);
      w_local[3] = static_cast<float>(((w[2] >> 2) & 0x3f) * s + b);
    }
  } else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      w_local[i] = static_cast<float>(s * w[i] + b);
    }
  }
}

// MLX qmv_wide_impl (affine, k_lanes = 8) for one weight row and NV input
// vectors: each lane reduces groups k_lane, k_lane + 8, ... in 8-value
// sub-chunks; the caller applies the 4/2/1 shuffle-down ladder.
template <typename T, int K, int GS, int BITS, int NV>
inline void glm_qmv_wide_row(
    const device uint8_t* wrow,
    const device T* srow,
    const device T* brow,
    const device T* x,
    int nv,
    int k_lane,
    thread float* result) {
  constexpr int sub = 8;
  constexpr int G = K / GS;
  for (int g = k_lane; g < G; g += 8) {
    float scale = srow[g];
    float bias = brow[g];
    for (int sc = 0; sc < GS / sub; sc++) {
      const int k0 = g * GS + sc * sub;
      const device uint8_t* wc = wrow + k0 * BITS / 8;
      float w_dq[sub];
      glm_dequantize<sub, BITS>(wc, scale, bias, w_dq);
      for (int v = 0; v < NV; v++) {
        if (v < nv) {
          const device T* xc = x + v * K + k0;
          float acc = 0;
          for (int i = 0; i < sub; i++) {
            acc += static_cast<float>(xc[i]) * w_dq[i];
          }
          result[v] += acc;
        }
      }
    }
  }
}

// Same expressions as MLX's Sigmoid / Minimum / Maximum functors.
template <typename T>
inline T glm_sigmoid(T x) {
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
template <typename T>
inline T glm_minimum(T x, T y) {
  if (metal::isnan(x)) {
    return x;
  }
  return x < y ? x : y;
}
template <typename T>
inline T glm_maximum(T x, T y) {
  if (metal::isnan(x)) {
    return x;
  }
  return x > y ? x : y;
}

// The router's top-k selection (the select kernel's loop, one simdgroup):
// argpartition order of the biased scores, i.e. descending values with ties
// to the lower expert index and NaNs last (lowest index first). Every lane
// ends with the same picked[].
template <int E, int TOPK, typename P>
inline void glm_router_topk(P bz, uint lane, thread int* picked) {
  constexpr int PER = (E + 31) / 32;
  float vals[PER];
  bool taken[PER];
  for (int j = 0; j < PER; j++) {
    const int e = j * 32 + int(lane);
    vals[j] = e < E ? bz[e] : -INFINITY;
    taken[j] = e >= E;
  }
  for (int r = 0; r < TOPK; r++) {
    float best = -INFINITY;
    int best_e = 0x7fffffff;
    for (int j = 0; j < PER; j++) {
      const int e = j * 32 + int(lane);
      if (!taken[j] && !isnan(vals[j]) &&
          (best_e == 0x7fffffff || vals[j] > best || (vals[j] == best && e < best_e))) {
        best = vals[j];
        best_e = e;
      }
    }
    for (ushort off = 16; off >= 1; off >>= 1) {
      float ob = simd_shuffle_xor(best, off);
      int oe = simd_shuffle_xor(best_e, off);
      const bool other_better = oe != 0x7fffffff &&
          (best_e == 0x7fffffff || ob > best || (ob == best && oe < best_e));
      if (other_better) {
        best = ob;
        best_e = oe;
      }
    }
    if (best_e == 0x7fffffff) {
      for (int j = 0; j < PER; j++) {
        const int e = j * 32 + int(lane);
        if (!taken[j] && e < best_e) {
          best_e = e;
        }
      }
      for (ushort off = 16; off >= 1; off >>= 1) {
        best_e = min(best_e, simd_shuffle_xor(best_e, off));
      }
    }
    picked[r] = best_e;
    if ((best_e % 32) == int(lane)) {
      taken[best_e / 32] = true;
    }
  }
}

// Glm5NextClampedSwiGLU / Glm5NextMLP epilogue on bfloat16 projections:
//   silu(minimum(gate, limit)) * minimum(maximum(up, -limit), limit)
template <typename T>
inline T glm_clamped_swiglu(T gate, T up, T limit, T neg_limit) {
  T g = glm_minimum(gate, limit);
  T s = g * glm_sigmoid(g);
  T u = glm_minimum(glm_maximum(up, neg_limit), limit);
  return s * u;
}
"""


# Fused routed-expert (+ optional shared-expert) gate/up projection with the
# clamped SwiGLU epilogue.  One threadgroup z-slice per (token, route); route
# TOPK (when HAS_SHARED) is the shared expert.
_GATE_UP_SOURCE = r"""
  const uint simd_lid = thread_index_in_simdgroup;
  const uint simd_gid = simdgroup_index_in_threadgroup;
#if SLOT_MAJOR
  // Route slots vary fastest, so the slots of an expert that several
  // tokens share read its row block back to back (cache hits).
  const int tile = int(threadgroup_position_in_grid.z);
  const int z = int(threadgroup_position_in_grid.y);
#else
  const int tile = int(threadgroup_position_in_grid.y);
  const int z = int(threadgroup_position_in_grid.z);
#endif
  const T lim = T(limit[0]);
  const T neg_lim = T(-limit[0]);
#if SHARED_WIDE
  // Last z slice: the shared expert for all NTOK tokens with MLX's
  // multi-row qmv_wide arithmetic (8 lanes per row, 4 rows per simdgroup).
  if (z == NTOK * TOPK) {
    const int k_lane = int(simd_lid) % 8;
    const int row = (tile * NSG + int(simd_gid)) * 4 + int(simd_lid) / 8;
    constexpr int WB = K * SBITS / 8;
    constexpr int G = K / SGS;
    float g_res[NTOK];
    float u_res[NTOK];
    for (int v = 0; v < NTOK; v++) {
      g_res[v] = 0.0f;
      u_res[v] = 0.0f;
    }
    glm_qmv_wide_row<T, K, SGS, SBITS, NTOK>(
        (const device uint8_t*)sh_gate_w + size_t(row) * WB, sh_gate_s + row * G,
        sh_gate_b + row * G, x, NTOK, k_lane, g_res);
    glm_qmv_wide_row<T, K, SGS, SBITS, NTOK>(
        (const device uint8_t*)sh_up_w + size_t(row) * WB, sh_up_s + row * G,
        sh_up_b + row * G, x, NTOK, k_lane, u_res);
    for (int v = 0; v < NTOK; v++) {
      g_res[v] += simd_shuffle_down(g_res[v], 4);
      g_res[v] += simd_shuffle_down(g_res[v], 2);
      g_res[v] += simd_shuffle_down(g_res[v], 1);
      u_res[v] += simd_shuffle_down(u_res[v], 4);
      u_res[v] += simd_shuffle_down(u_res[v], 2);
      u_res[v] += simd_shuffle_down(u_res[v], 1);
    }
    if (k_lane == 0) {
      for (int v = 0; v < NTOK; v++) {
        shared_out[size_t(v) * N + row] = glm_clamped_swiglu<T>(
            static_cast<T>(g_res[v]), static_cast<T>(u_res[v]), lim, neg_lim);
      }
    }
    return;
  }
  constexpr int RT = TOPK;
#else
  constexpr int RT = TOPK + HAS_SHARED;
#endif
  const int token = z / RT;
  const int r = z - token * RT;
  const int out_row = (tile * NSG + int(simd_gid)) * RPS;
  const device T* xr = x + token * K;

  float g_res[RPS] = {0};
  float u_res[RPS] = {0};
  if (r < TOPK) {
#if SELECT
    // One token: this simdgroup replays the router's selection on the
    // biased sigmoid scores (no separate select dispatch); slot 0 / tile 0
    // publishes the routes and routing weights for the down kernel.
    int picked[TOPK];
    glm_router_topk<NE, TOPK>(sel_biased, simd_lid, picked);
    const int expert = picked[r];
    if (z == 0 && tile == 0 && simd_gid == 0 && simd_lid == 0) {
      float total = 0.0f;
      float gathered[TOPK];
      for (int q = 0; q < TOPK; q++) {
        gathered[q] = sel_sig[picked[q]];
        total = gathered[q] + total;
      }
      for (int q = 0; q < TOPK; q++) {
        float qv = SEL_NORM ? gathered[q] / total : gathered[q];
        float sv = qv * sel_scaling[0];
        sel_indices[q] = uint(picked[q]);
        sel_scores[q] = sv;
      }
    }
#else
    const int expert = int(indices[token * TOPK + r]);
#endif
    constexpr int WB = K * RBITS / 8;   // bytes per weight row
    constexpr int G = K / RGS;          // groups per row
    // ESTRIDE rows per expert; a fused [gate; up] tensor (ESTRIDE = 2N) is
    // passed as both gate and up with the up rows UP_OFF = N further on.
    const size_t row0 = size_t(expert) * ESTRIDE + out_row;
    const size_t urow0 = row0 + UP_OFF;
    glm_qmv_rows<T, K, RGS, RBITS, RPS>(
        (const device uint8_t*)gate_w + row0 * WB, gate_s + row0 * G,
        gate_b + row0 * G, xr, simd_lid, g_res);
    glm_qmv_rows<T, K, RGS, RBITS, RPS>(
        (const device uint8_t*)up_w + urow0 * WB, up_s + urow0 * G,
        up_b + urow0 * G, xr, simd_lid, u_res);
  } else {
#if HAS_SHARED
    constexpr int WB = K * SBITS / 8;
    constexpr int G = K / SGS;
    const size_t row0 = size_t(out_row);
    glm_qmv_rows<T, K, SGS, SBITS, RPS>(
        (const device uint8_t*)sh_gate_w + row0 * WB, sh_gate_s + row0 * G,
        sh_gate_b + row0 * G, xr, simd_lid, g_res);
    glm_qmv_rows<T, K, SGS, SBITS, RPS>(
        (const device uint8_t*)sh_up_w + row0 * WB, sh_up_s + row0 * G,
        sh_up_b + row0 * G, xr, simd_lid, u_res);
#endif
  }
  device T* o = out + size_t(z) * N + out_row;
  for (int row = 0; row < RPS; row++) {
    float gv = simd_sum(g_res[row]);
    float uv = simd_sum(u_res[row]);
    if (simd_lid == 0) {
      o[row] = glm_clamped_swiglu<T>(static_cast<T>(gv), static_cast<T>(uv), lim, neg_lim);
    }
  }
"""


# One token's shared-expert gate/up with the clamped SwiGLU: the shared slot
# of the fused gate/up kernel as its own dispatch. It does not depend on the
# router, so it runs concurrently with the router logits kernel (no barrier
# between them) and hides that kernel's latency under its weight stream.
_SHARED_GATE_UP_SOURCE = r"""
  const uint simd_lid = thread_index_in_simdgroup;
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const int tile = int(threadgroup_position_in_grid.y);
  const T lim = T(limit[0]);
  const T neg_lim = T(-limit[0]);
  const int out_row = (tile * NSG + int(simd_gid)) * RPS;
  float g_res[RPS] = {0};
  float u_res[RPS] = {0};
  {
    constexpr int WB = K * SBITS / 8;
    constexpr int G = K / SGS;
    const size_t row0 = size_t(out_row);
    glm_qmv_rows<T, K, SGS, SBITS, RPS>(
        (const device uint8_t*)sh_gate_w + row0 * WB, sh_gate_s + row0 * G,
        sh_gate_b + row0 * G, x, simd_lid, g_res);
    glm_qmv_rows<T, K, SGS, SBITS, RPS>(
        (const device uint8_t*)sh_up_w + row0 * WB, sh_up_s + row0 * G,
        sh_up_b + row0 * G, x, simd_lid, u_res);
  }
  device T* o = out + out_row;
  for (int row = 0; row < RPS; row++) {
    float gv = simd_sum(g_res[row]);
    float uv = simd_sum(u_res[row]);
    if (simd_lid == 0) {
      o[row] = glm_clamped_swiglu<T>(static_cast<T>(gv), static_cast<T>(uv), lim, neg_lim);
    }
  }
"""


@lru_cache(maxsize=None)
def _shared_gate_up_kernel():
    return mx.fast.metal_kernel(
        name="glm5_moe_shared_gate_up_swiglu",
        input_names=["x", "limit", "sh_gate_w", "sh_gate_s", "sh_gate_b", "sh_up_w", "sh_up_s", "sh_up_b"],
        output_names=["out"],
        header=_QMV_HEADER,
        source=_SHARED_GATE_UP_SOURCE,
    )


def mlp_gate_up_swiglu(x: mx.array, gate, up, limit: float) -> Optional[mx.array]:
    """One token of Glm5NextMLP's gate/up projections and clamped SwiGLU in
    one dispatch (the shared-expert gate/up kernel): ``silu(min(gate(x),
    limit)) * clip(up(x), -limit, limit)`` for ``x`` [1, K], each projection
    row with the one-row qmv_fast arithmetic. Returns [1, N] or None."""
    if x.ndim != 2 or x.shape[0] != 1:
        return None
    if x.dtype not in (mx.bfloat16, mx.float16):
        return None
    parts = [_affine_parts(m) for m in (gate, up)]
    if any(p is None for p in parts):
        return None
    (gw, gs, gb, bits, gsz), (uw, us, ub, ubits, usz) = parts
    K = x.shape[1]
    N = gw.shape[0]
    if (bits, gsz) != (ubits, usz) or gw.ndim != 2 or gw.shape != uw.shape:
        return None
    if gw.shape[1] * 32 // bits != K or gs.dtype != x.dtype or us.dtype != x.dtype:
        return None
    rps, nsg = 4, 2
    if not _qmv_fast_ok(bits, gsz, N, K) or N % (rps * nsg):
        return None
    STATS["mlp_gate_up"] += 1
    return _shared_gate_up_kernel()(
        inputs=[x, mx.array([limit], dtype=mx.float32), gw, gs, gb, uw, us, ub],
        template=[("T", x.dtype), ("K", K), ("N", N), ("SBITS", bits), ("SGS", gsz),
                  ("RPS", rps), ("NSG", nsg)],
        grid=(32, (N // (rps * nsg)) * nsg, 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(1, N)],
        output_dtypes=[x.dtype],
    )[0]


# Fused routed down projection + routing-weighted sum (+ shared expert down
# projection and residual-free add), reproducing
#   y = (down(act) * scores[..., None]).sum(-2).astype(T) + shared_down(act_s)
_DOWN_SOURCE = r"""
  const uint simd_lid = thread_index_in_simdgroup;
  const uint simd_gid = simdgroup_index_in_threadgroup;
#if SLOT_MAJOR
  // Tokens vary fastest: experts they share are read back to back.
  const int tile = int(threadgroup_position_in_grid.z);
  const int token = int(threadgroup_position_in_grid.y);
#else
  const int tile = int(threadgroup_position_in_grid.y);
  const int token = int(threadgroup_position_in_grid.z);
#endif
  // Activation slots per token: the routed ones, then the shared expert's.
  constexpr int RT = TOPK + HAS_SHARED;
  const int out_row = (tile * NSG + int(simd_gid)) * RPS;

  float acc[RPS] = {0};
  constexpr int WB = K * RBITS / 8;
  constexpr int G = K / RGS;
  for (int r = 0; r < TOPK; r++) {
    const int expert = int(indices[token * TOPK + r]);
    const size_t row0 = size_t(expert) * N + out_row;
    float res[RPS] = {0};
    glm_qmv_rows<T, K, RGS, RBITS, RPS>(
        (const device uint8_t*)down_w + row0 * WB, down_s + row0 * G,
        down_b + row0 * G, act + (size_t(token) * RT + r) * K, simd_lid, res);
    const float score = scores[token * TOPK + r];
    for (int row = 0; row < RPS; row++) {
      float v = simd_sum(res[row]);
      // The reference rounds the fp32 product in its own Multiply kernel
      // before the Sum; keep the compiler from contracting it into an FMA.
      volatile float weighted = static_cast<float>(static_cast<T>(v)) * score;
      acc[row] += weighted;
    }
  }
#if HAS_SHARED
  float sres[RPS] = {0};
  {
    constexpr int SWB = K * SBITS / 8;
    constexpr int SG = K / SGS;
    const size_t row0 = size_t(out_row);
    const device T* sh_x = act + (size_t(token) * RT + TOPK) * K;
    glm_qmv_rows<T, K, SGS, SBITS, RPS>(
        (const device uint8_t*)sh_down_w + row0 * SWB, sh_down_s + row0 * SG,
        sh_down_b + row0 * SG, sh_x, simd_lid, sres);
  }
#endif
#if SHARED_WIDE_DOWN
  // Shared expert down projection of this token with the multi-row qmv_wide
  // arithmetic its own [T, K] matmul uses (8 lanes per row; each token's
  // accumulation is independent of the others).
  {
    static_assert(RPS == 4, "qmv_wide rows per simdgroup");
    constexpr int SWB = K * SBITS / 8;
    constexpr int SG = K / SGS;
    const int k_lane = int(simd_lid) % 8;
    const int srow = out_row + int(simd_lid) / 8;
    float sv[1] = {0.0f};
    glm_qmv_wide_row<T, K, SGS, SBITS, 1>(
        (const device uint8_t*)sh_down_w + size_t(srow) * SWB, sh_down_s + srow * SG,
        sh_down_b + srow * SG, sh_act + size_t(token) * K, 1, k_lane, sv);
    sv[0] += simd_shuffle_down(sv[0], 4);
    sv[0] += simd_shuffle_down(sv[0], 2);
    sv[0] += simd_shuffle_down(sv[0], 1);
    if (k_lane == 0) {
      const int r = int(simd_lid) / 8;
      float a = acc[0];
      for (int row = 1; row < RPS; row++) {
        a = row == r ? acc[row] : a;
      }
      out[size_t(token) * N + srow] = static_cast<T>(a) + static_cast<T>(sv[0]);
    }
  }
  return;
#endif
  device T* o = out + size_t(token) * N + out_row;
  for (int row = 0; row < RPS; row++) {
#if HAS_SHARED
    float sv = simd_sum(sres[row]);
#endif
    if (simd_lid == 0) {
#if HAS_SHARED
      o[row] = static_cast<T>(acc[row]) + static_cast<T>(sv);
#elif ADD_SHARED_Y
      o[row] = static_cast<T>(acc[row]) + shared_y[size_t(token) * N + out_row + row];
#else
      o[row] = static_cast<T>(acc[row]);
#endif
    }
  }
"""


def _source(body: str, **defines) -> str:
    lines = [f"#define {k} {int(v)}" for k, v in defines.items()]
    undef = [f"#undef {k}" for k in defines]
    return "\n".join(lines) + "\n" + body + "\n" + "\n".join(undef) + "\n"


@lru_cache(maxsize=None)
def _gate_up_kernel(
    has_shared: bool,
    shared_wide: bool = False,
    slot_major: bool = False,
    select: bool = False,
):
    routes = ["sel_sig", "sel_biased", "sel_scaling"] if select else ["indices"]
    inputs = ["x"] + routes + ["limit", "gate_w", "gate_s", "gate_b", "up_w", "up_s", "up_b"]
    if has_shared or shared_wide:
        inputs += ["sh_gate_w", "sh_gate_s", "sh_gate_b", "sh_up_w", "sh_up_s", "sh_up_b"]
    suffix = "_widesh" if shared_wide else ("_shared" if has_shared else "")
    suffix += "_sm" if slot_major else ""
    suffix += "_select" if select else ""
    outputs = ["out", "shared_out"] if shared_wide else ["out"]
    if select:
        outputs += ["sel_indices", "sel_scores"]
    return mx.fast.metal_kernel(
        name=f"glm5_moe_gate_up_swiglu{suffix}",
        input_names=inputs,
        output_names=outputs,
        header=_QMV_HEADER,
        source=_source(
            _GATE_UP_SOURCE,
            HAS_SHARED=int(has_shared and not shared_wide),
            SHARED_WIDE=int(shared_wide),
            SLOT_MAJOR=int(slot_major),
            SELECT=int(select),
        ),
    )


@lru_cache(maxsize=None)
def _down_kernel(
    has_shared: bool,
    add_shared_y: bool,
    slot_major: bool = False,
    shared_wide: bool = False,
):
    inputs = ["act", "indices", "scores", "down_w", "down_s", "down_b"]
    if has_shared:
        inputs += ["sh_down_w", "sh_down_s", "sh_down_b"]
        if shared_wide:
            inputs += ["sh_act"]
    elif add_shared_y:
        inputs += ["shared_y"]
    suffix = "_widesh" if shared_wide else (
        "_shared" if has_shared else ("_add" if add_shared_y else "")
    )
    suffix += "_sm" if slot_major else ""
    return mx.fast.metal_kernel(
        name=f"glm5_moe_down_combine{suffix}",
        input_names=inputs,
        output_names=["out"],
        header=_QMV_HEADER,
        source=_source(
            _DOWN_SOURCE,
            HAS_SHARED=int(has_shared and not shared_wide),
            ADD_SHARED_Y=int(add_shared_y and not has_shared),
            SLOT_MAJOR=int(slot_major),
            SHARED_WIDE_DOWN=int(shared_wide),
        ),
    )


# MLX uses the same qmv_fast alignment rule for 3-bit and 4/5/6/8-bit weights.
def _qmv_fast_ok(bits: int, group_size: int, n: int, k: int) -> bool:
    """MLX qmv_fast shapes with full output tiles for the fused kernels."""
    if bits not in (3, 4, 5, 6, 8) or group_size not in (32, 64, 128):
        return False
    pack_factor = 8 if bits in (3, 5) else (4 if bits == 6 else 32 // bits)
    values_per_thread = pack_factor * 2
    if group_size % values_per_thread:
        return False
    return n % 8 == 0 and k % (values_per_thread * 32) == 0


def _affine_parts(layer):
    """(weight, scales, biases, bits, group_size) of an affine quantized layer."""
    if getattr(layer, "mode", "affine") != "affine":
        return None
    biases = layer.get("biases") if hasattr(layer, "get") else getattr(layer, "biases", None)
    if biases is None or "bias" in layer:
        return None
    return layer["weight"], layer["scales"], biases, int(layer.bits), int(layer.group_size)


def moe_gate_up_swiglu(
    x: mx.array,
    indices: mx.array,
    limit: float,
    routed_gate,
    routed_up,
    shared_gate=None,
    shared_up=None,
    *,
    rps: int = 4,
    nsg: int = 2,
    shared_wide: bool = False,
    select=None,
):
    """Clamped-SwiGLU activations for every (token, routed expert[, shared]).

    ``x`` is [T, K] (one row per token), ``indices`` [T, TOPK].  Returns
    [T, TOPK (+1), N] in ``x.dtype`` or None when the shapes are not covered.
    With ``shared_wide`` (2 <= T <= 8) the shared expert uses the multi-row
    qmv_wide arithmetic the reference applies to T > 1 rows and the call
    returns ``(routed [T, TOPK, N], shared [T, N])``. ``routed_up=None``
    means ``routed_gate`` is a fused ``gate_up_proj`` ([E, 2N, *]: gate rows
    then up rows per expert, as the MoE gate/up fusion lays them out).

    ``select = (sig, biased, top_k, scaling, norm_topk_prob)`` (one token,
    the ``moe_router_logits`` outputs) replaces ``indices``: every routed
    threadgroup replays the router's top-k selection, and the call returns
    ``(act, indices [1, top_k] uint32, scores [1, top_k] fp32)`` like
    ``moe_router`` + this kernel, or None when not covered.
    """
    if select is not None:
        if shared_wide or x.ndim != 2 or x.shape[0] != 1:
            return None
        sig, biased, sel_topk, sel_scaling, sel_norm = select
        E_r = sig.shape[-1]
        if sig.shape != (1, E_r) or biased.shape != (1, E_r) or not 1 <= sel_topk <= 32:
            return None
        if sig.dtype != mx.float32 or biased.dtype != mx.float32 or E_r > 1024:
            return None
        indices = mx.zeros((1, sel_topk), dtype=mx.uint32)  # shape only
    fused_gu = routed_up is None
    parts = [_affine_parts(m) for m in ((routed_gate,) if fused_gu else (routed_gate, routed_up))]
    if any(p is None for p in parts) or x.ndim != 2 or indices.ndim != 2:
        return None
    if fused_gu:
        parts = parts * 2
    (gw, gs, gb, rbits, rgs), (uw, us, ub, ubits, ugs) = parts
    if (rbits, rgs) != (ubits, ugs) or gw.shape != uw.shape or gw.ndim != 3:
        return None
    T, K = x.shape
    E, N, _ = gw.shape
    estride, up_off = N, 0
    if fused_gu:
        if N % 2:
            return None
        N //= 2
        estride, up_off = 2 * N, N
    topk = indices.shape[1]
    if x.dtype not in (mx.bfloat16, mx.float16) or gs.dtype != x.dtype or us.dtype != x.dtype:
        return None
    if not _qmv_fast_ok(rbits, rgs, N, K) or N % (rps * nsg):
        return None
    has_shared = shared_gate is not None
    if shared_wide and (not has_shared or not 2 <= T <= 8 or rps != 4):
        return None
    if select is not None:
        routes = [sig, biased, mx.array([sel_scaling], dtype=mx.float32)]
    else:
        routes = [indices]
    inputs = [x] + routes + [mx.array([limit], dtype=mx.float32), gw, gs, gb, uw, us, ub]
    template = [
        ("T", x.dtype), ("K", K), ("N", N), ("TOPK", topk), ("RBITS", rbits),
        ("RGS", rgs), ("RPS", rps), ("NSG", nsg), ("ESTRIDE", estride), ("UP_OFF", up_off),
    ]
    if select is not None:
        template += [("NE", E_r), ("SEL_NORM", int(bool(sel_norm) and sel_topk > 1))]
    if has_shared:
        sparts = [_affine_parts(m) for m in (shared_gate, shared_up)]
        if any(p is None for p in sparts):
            return None
        (sgw, sgs, sgb, sbits, sgsz), (suw, sus, sub, subits, susz) = sparts
        if (sbits, sgsz) != (subits, susz) or sgw.shape[0] != N or suw.shape[0] != N:
            return None
        if sgs.dtype != x.dtype or sus.dtype != x.dtype:
            return None
        if shared_wide:
            # qmv_wide: groups decoded in 8-value sub-chunks, 8 lanes per row.
            if sbits not in (4, 5, 6, 8) or sgsz % 8 or K % sgsz or (K // sgsz) < 1:
                return None
        elif not _qmv_fast_ok(sbits, sgsz, N, K):
            return None
        inputs += [sgw, sgs, sgb, suw, sus, sub]
        template += [("SBITS", sbits), ("SGS", sgsz)]
    slot_major = _slot_major(T)
    kernel = _gate_up_kernel(has_shared, shared_wide, slot_major, select is not None)
    STATS["moe_gate_up"] += 1
    tiles = N // (rps * nsg)
    slots = T * topk + 1 if shared_wide else T * (topk + int(has_shared))
    grid = (32, slots * nsg, tiles) if slot_major else (32, tiles * nsg, slots)
    if shared_wide:
        template += [("NTOK", T)]
        routed, shared = kernel(
            inputs=inputs,
            template=template,
            grid=grid,
            threadgroup=(32, nsg, 1),
            output_shapes=[(T, topk, N), (T, N)],
            output_dtypes=[x.dtype, x.dtype],
        )
        STATS["moe_shared_wide"] += 1
        return routed, shared
    rt = topk + int(has_shared)
    if select is not None:
        STATS["router_select_fused"] += 1
        act, sel_indices, sel_scores = kernel(
            inputs=inputs,
            template=template,
            grid=grid,
            threadgroup=(32, nsg, 1),
            output_shapes=[(T, rt, N), (1, topk), (1, topk)],
            output_dtypes=[x.dtype, mx.uint32, mx.float32],
        )
        return act, sel_indices, sel_scores
    return kernel(
        inputs=inputs,
        template=template,
        grid=grid,
        threadgroup=(32, nsg, 1),
        output_shapes=[(T, rt, N)],
        output_dtypes=[x.dtype],
    )[0]


def _slot_major(tokens: int) -> bool:
    """Route-slot-major grids for blocks of 2+ tokens (they share experts)."""
    return tokens > 1


def moe_down_combine(
    act: mx.array,
    indices: mx.array,
    scores: mx.array,
    routed_down,
    shared_down=None,
    shared_y: Optional[mx.array] = None,
    *,
    shared_act: Optional[mx.array] = None,
    rps: int = 4,
    nsg: int = 2,
) -> Optional[mx.array]:
    """Routed down projections combined with the routing weights (+ shared).

    ``act`` is [T, TOPK (+1), K] from :func:`moe_gate_up_swiglu`, ``scores``
    [T, TOPK] float32.  The shared expert is either projected here from the
    last activation slot (``shared_down``), from ``shared_act`` [T, K] (the
    ``shared_wide`` gate/up output, with the multi-row qmv_wide arithmetic)
    or added from a precomputed ``shared_y`` [T, N].  Returns [T, N] in
    ``act.dtype``.
    """
    p = _affine_parts(routed_down)
    if p is None or act.ndim != 3 or scores.dtype != mx.float32:
        return None
    dw, ds, db, rbits, rgs = p
    T, rt, K = act.shape
    E, N, _ = dw.shape
    topk = indices.shape[1]
    has_shared = shared_down is not None
    shared_wide = shared_act is not None
    if shared_wide and (not has_shared or shared_y is not None or rps != 4):
        return None
    if shared_wide and not 2 <= T <= 8:
        return None
    if rt != topk + int(has_shared and not shared_wide) or ds.dtype != act.dtype:
        return None
    if not _qmv_fast_ok(rbits, rgs, N, K) or N % (rps * nsg):
        return None
    inputs = [act, indices, scores, dw, ds, db]
    template = [
        ("T", act.dtype), ("K", K), ("N", N), ("TOPK", topk), ("RBITS", rbits),
        ("RGS", rgs), ("RPS", rps), ("NSG", nsg),
    ]
    if has_shared:
        sp = _affine_parts(shared_down)
        if sp is None:
            return None
        sdw, sds, sdb, sbits, sgsz = sp
        if sdw.shape[0] != N or sds.dtype != act.dtype:
            return None
        if shared_wide:
            if shared_act.shape != (T, K) or shared_act.dtype != act.dtype:
                return None
            if sbits not in (4, 5, 6, 8) or sgsz % 8 or K % sgsz or K in (64, 128):
                return None
        elif not _qmv_fast_ok(sbits, sgsz, N, K):
            return None
        inputs += [sdw, sds, sdb] + ([shared_act] if shared_wide else [])
        template += [("SBITS", sbits), ("SGS", sgsz)]
    elif shared_y is not None:
        if shared_y.shape != (T, N) or shared_y.dtype != act.dtype:
            return None
        inputs.append(shared_y)
    slot_major = _slot_major(T)
    kernel = _down_kernel(has_shared, shared_y is not None, slot_major, shared_wide)
    STATS["moe_down"] += 1
    if shared_wide:
        STATS["moe_down_shared_wide"] += 1
    tiles = N // (rps * nsg)
    return kernel(
        inputs=inputs,
        template=template,
        grid=(32, T * nsg, tiles) if slot_major else (32, tiles * nsg, T),
        threadgroup=(32, nsg, 1),
        output_shapes=[(T, N)],
        output_dtypes=[act.dtype],
    )[0]


# ---------------------------------------------------------------------------
# Hyper-connection mix: x.astype(f32) -> rms_norm (no weight) -> @ fn.T
# ---------------------------------------------------------------------------
#
# Reproduces MLX's ``rms_looped`` (1024 threads, 4 reads per thread) for the
# inverse RMS and the non-transposed ``gemv`` kernel that MLX selects for a
# [1, HC*D] x [HC*D, MIX] product with MIX < 4096 and K >= 16 * MIX
# (BM=1, BN=8, SM=1, SN=32, TN=4): every output row is reduced by eight
# simdgroups, each lane accumulating 4 contiguous products per 1024-wide K
# block, a shuffle-down ladder inside the simdgroup and a sequential sum over
# the eight simdgroups.  One mix row per 256-thread threadgroup: each real
# simdgroup plays four of the 32 virtual 1024-thread simdgroups of the RMS
# reduction, and each threadgroup recomputes the (cheap) RMS.
_HC_MIX1_SOURCE = r"""
  const uint lid = thread_position_in_threadgroup.x;
  const uint simd_lid = thread_index_in_simdgroup;
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const int tok = int(threadgroup_position_in_grid.y);
  const int row = int(threadgroup_position_in_grid.x);
  constexpr int KSZ = HCD;
  const device T* xr = x + size_t(tok) * KSZ;

  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[32];
  for (int v = 0; v < 4; v++) {
    const uint vt = uint(v) * 256 + lid;
    float acc = 0;
    for (uint r = 0; r < uint(KSZ); r += 1024 * 4) {
      for (int i = 0; i < 4; i++) {
        float xi = static_cast<float>(xr[r + vt * 4 + i]);
        acc += xi * xi;
      }
    }
    acc = simd_sum(acc);
    if (simd_lid == 0) {
      local_sums[v * 8 + int(simd_gid)] = acc;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    float acc = simd_sum(local_sums[simd_lid]);
    if (simd_lid == 0) {
      local_inv_mean[0] = metal::precise::rsqrt(acc / KSZ + eps[0]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = local_inv_mean[0];

  threadgroup float partial[8];
  const int sgN = int(simd_gid);
  float result = 0;
  const device auto* mrow = fn + size_t(row) * KSZ;
  int bn = (32 * sgN + int(simd_lid)) * 4;
  for (int i = 0; i < KSZ / 1024; ++i) {
    float v_coeff[4];
    float inter[4];
    for (int tn = 0; tn < 4; tn++) {
      v_coeff[tn] = static_cast<float>(xr[bn + tn]) * inv;
    }
    for (int tn = 0; tn < 4; tn++) {
      inter[tn] = static_cast<float>(mrow[bn + tn]);
    }
    for (int tn = 0; tn < 4; tn++) {
      result += inter[tn] * v_coeff[tn];
    }
    bn += 1024;
  }
  for (ushort sn = 16; sn >= 1; sn >>= 1) {
    result += simd_shuffle_down(result, sn);
  }
  if (simd_lid == 0) {
    partial[sgN] = result;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgN == 0 && simd_lid == 0) {
    float total = partial[0];
    for (int s = 1; s < 8; s++) {
      total += partial[s];
    }
    mixes[size_t(tok) * MIX + row] = total;
  }
"""


@lru_cache(maxsize=None)
def _hc_mix1_kernel():
    return mx.fast.metal_kernel(
        name="glm5_hc_mix_rms_gemv_row",
        input_names=["x", "fn", "eps"],
        output_names=["mixes"],
        source=_HC_MIX1_SOURCE,
    )


def _exact_bf16(owner, name: str) -> mx.array:
    """``getattr(owner, name)`` as BF16 when every fp32 value is a BF16
    value, else the fp32 array (decided once per array, cached on ``owner``)."""
    key = f"_omlx_{name}_bf16"
    wide = getattr(owner, name)
    cached = getattr(owner, key, None)
    if cached is None or cached[0] is not wide:
        narrow = wide.astype(mx.bfloat16)
        if mx.array_equal(narrow.astype(mx.float32), wide).item():
            mx.eval(narrow)
        else:
            narrow = None
        cached = (wide, narrow)
        # A plain attribute, not a module parameter.
        object.__setattr__(owner, key, cached)
    return wide if cached[1] is None else cached[1]


def mix_weights(connection) -> mx.array:
    """``connection.fn`` [MIX, HC * D] as BF16 when every fp32 weight is a
    BF16 value, else the fp32 array. GLM-5.3 checkpoints store these weights
    in BF16 (the sanitize keeps them in fp32): the kernels read the same
    values, as floats, from half the bytes."""
    return _exact_bf16(connection, "fn")


def router_weights(gate) -> mx.array:
    """The MoE router's ``gate.weight`` [E, K] as BF16 when every fp32 weight
    is a BF16 value, else the fp32 array: the FP8 checkpoint stores it in
    BF16 and oQ in fp32 from those BF16 values, and the router runs in fp32
    (``moe_router_dtype``), so ``moe_router`` reads the same values, as
    floats, from half the bytes."""
    return _exact_bf16(gate, "weight")


def hc_mix(x: mx.array, fn: mx.array, eps: float) -> Optional[mx.array]:
    """``(rms_norm(x.astype(f32).flatten(-2)) @ fn.T)`` per token, M=1 exact.

    ``x`` is [B, L, HC, D] bf16/fp16, ``fn`` [MIX, HC*D] float32 (or its
    exact BF16 copy, ``mix_weights``).  Returns [B, L, MIX] float32 or None
    when the shape is outside the replicated kernel configuration.
    """
    if x.ndim != 4 or fn.ndim != 2 or fn.dtype not in (mx.float32, mx.bfloat16):
        return None
    B, L, hc, d = x.shape
    K = hc * d
    mix = fn.shape[0]
    if fn.shape[1] != K or K % 4096 or mix >= 4096 or K < 16 * mix:
        return None
    if x.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        return None
    STATS["hc_mix"] += 1
    return _hc_mix1_kernel()(
        inputs=[x, fn, mx.array([eps], dtype=mx.float32)],
        template=[("T", x.dtype), ("HCD", K), ("MIX", mix)],
        grid=(256 * mix, B * L, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, L, mix)],
        output_dtypes=[mx.float32],
    )[0]


# ---------------------------------------------------------------------------
# DSA indexer: decode/verify scores and top-k index expansion
# ---------------------------------------------------------------------------
#
# The prefill score kernel (Steel GEMM tile, BM=64) is launched with the
# query rows zero-padded to 64 and only P/64 threadgroups, which makes it the
# single most expensive decode kernel once the context passes 2k tokens.  The
# kernel below computes the same values for up to eight query rows: per head
# it accumulates the 8x8 simdgroup MMAs over D in the same 8-wide K order as
# the Steel tile (float fragments, zero rows for missing queries), and it
# adds max(score, 0) * weight over the heads in the same sequential order.
# Invalid pooled positions receive the same -1e30 sentinel the Python path
# writes with ``mx.where``.
_DSA_SCORES_SOURCE = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint tid = thread_position_in_threadgroup.x + 32 * sg;
  const int key0 = int(threadgroup_position_in_grid.x) * 8;
  const int P = int(pool_len_cap[1]);
  const int pool_len = int(pool_len_cap[0]);
  const int qpos0 = int(qpos[0]);

  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;

  threadgroup float hs[HEADS][8][8];

  // B fragments (K x 8 keys) for this key block, kept in registers.
  simdgroup_matrix<float, 8, 8> bfrag[DIM / 8];
  for (int kb = 0; kb < DIM / 8; kb++) {
    float2 bv = float2(0.0f);
    for (short e = 0; e < 2; e++) {
      int key = key0 + fn + e;
      if (key < P) {
        bv[e] = static_cast<float>(keys[size_t(key) * DIM + kb * 8 + fm]);
      }
    }
    reinterpret_cast<thread float2&>(bfrag[kb].thread_elements()) = bv;
  }

  for (int hh = 0; hh < HEADS / NSG; hh++) {
    const int h = int(sg) * (HEADS / NSG) + hh;
    // This head's query fragments, loaded ahead of the (accumulator
    // dependent) MMA chain; the chain itself is unchanged.
    float2 av[DIM / 8];
    for (int kb = 0; kb < DIM / 8; kb++) {
      av[kb] = float2(0.0f);
      if (fm < L) {
        const device T* qr = q + (size_t(fm) * HEADS + h) * DIM + kb * 8 + fn;
        av[kb][0] = static_cast<float>(qr[0]);
        av[kb][1] = static_cast<float>(qr[1]);
      }
    }
    simdgroup_matrix<float, 8, 8> c = simdgroup_matrix<float, 8, 8>(0.0f);
    for (int kb = 0; kb < DIM / 8; kb++) {
      simdgroup_matrix<float, 8, 8> a;
      reinterpret_cast<thread float2&>(a.thread_elements()) = av[kb];
      simdgroup_multiply_accumulate(c, a, bfrag[kb], c);
    }
    float2 cv = reinterpret_cast<thread float2&>(c.thread_elements());
    hs[h][fm][fn] = cv[0];
    hs[h][fm][fn + 1] = cv[1];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < uint(L * 8)) {
    const int row = int(tid) / 8;
    const int j = int(tid) % 8;
    const int key = key0 + j;
    if (key < P) {
      float accum = 0.0f;
      for (int h = 0; h < HEADS; h++) {
        const float weight = static_cast<float>(w[row * HEADS + h]);
        accum += max(hs[h][row][j], 0.0f) * weight;
      }
      const bool valid = key < pool_len && (key + 1) * KPOOL - 1 <= qpos0 + row;
      scores[size_t(row) * P + key] = valid ? static_cast<T>(accum) : static_cast<T>(-1e30f);
    }
  }
"""


@lru_cache(maxsize=None)
def _dsa_scores_kernel():
    return mx.fast.metal_kernel(
        name="glm5_dsa_decode_scores",
        input_names=["q", "keys", "w", "qpos", "pool_len_cap"],
        output_names=["scores"],
        header="#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n",
        source=_DSA_SCORES_SOURCE,
    )


def dsa_decode_scores(
    q: mx.array,
    pool_keys: mx.array,
    weights: mx.array,
    query_pos0: int,
    pool_len: int,
    kpool: int,
    *,
    nsg: int = 8,
) -> Optional[mx.array]:
    """Masked indexer scores for L <= 8 query rows of one sequence.

    ``q`` [1, L, H, D], ``pool_keys`` [1, P, D],
    ``weights`` [1, L, H] (already scaled, q dtype).  Returns [1, L, P]
    scores equal to the padded Steel kernel followed by the validity
    ``mx.where``.
    """
    if q.ndim != 4 or q.shape[0] != 1 or pool_keys.ndim != 3 or pool_keys.shape[0] != 1:
        return None
    _, L, H, D = q.shape
    P = pool_keys.shape[1]
    if not (1 <= L <= 8) or D % 8 or H % nsg or P == 0:
        return None
    if q.dtype not in (mx.bfloat16, mx.float16) or pool_keys.dtype != q.dtype or weights.dtype != q.dtype:
        return None
    # Inputs are made row contiguous by the kernel launch (a no-op for the
    # pooled cache view, whose rows are contiguous for one sequence).
    STATS["dsa_scores"] += 1
    return _dsa_scores_kernel()(
        inputs=[
            q,
            pool_keys,
            weights,
            mx.array([query_pos0], dtype=mx.int32),
            mx.array([pool_len, P], dtype=mx.int32),
        ],
        template=[("T", q.dtype), ("L", L), ("HEADS", H), ("DIM", D), ("NSG", nsg), ("KPOOL", kpool)],
        grid=(32 * ((P + 7) // 8), nsg, 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(1, L, P)],
        output_dtypes=[q.dtype],
    )[0]


# Top-k of the indexer's pooled-block scores for decode/verify rows, expanded
# into token indices: the native radix-select kernel's output (a function of
# the scores alone: keys of the 16-bit ordered score bits; strictly greater
# keys in index order, then threshold ties in index order) passed through
# Glm5NextIndexer.__call__'s expansion (validity, kpool expansion, left
# padding, the always-selected tail window and the -1 padding up to the
# output width), in one dispatch. The threshold key comes from a 4-bit radix
# select (four passes of simdgroup digit counts, two barriers each).
_DSA_SELECT_SOURCE = r"""
  // One threadgroup (1024 threads) per score row.
  const uint tid = thread_position_in_threadgroup.x;
  const uint lane = tid % 32;
  const uint sg = tid / 32;
  const int row = int(threadgroup_position_in_grid.y);
  const int P = int(meta[0]);
  const int qp = int(meta[1]) + row;
  const int pool_len = int(meta[2]);
  const int lp = int(left_padding[0]);
  const device T* rs = scores + size_t(row) * P;
  device int* ro = out + size_t(row) * OUT_W;
  threadgroup uint part[32][16];
  threadgroup uint bcast[2];
  threadgroup uint part_g[32];
  threadgroup uint part_t[32];

  // The TOPK-th largest key: digit by digit from the top, keep the digit
  // whose keys (among those matching the digits so far) reach rank `need`.
  uint prefix = 0u;
  uint need = uint(TOPK);
  for (int shift = 12; shift >= 0; shift -= 4) {
    const uint hi_mask = (0xffffu << uint(shift + 4)) & 0xffffu;
    uint cnt[16];
    for (int b = 0; b < 16; b++) {
      cnt[b] = 0u;
    }
    for (int i = int(tid); i < P; i += 1024) {
      const ushort bits = as_type<ushort>(rs[i]);
      const uint key = (bits & 0x8000) ? uint((~bits) & 0xffff) : uint(bits | 0x8000);
      if ((key & hi_mask) == prefix) {
        cnt[(key >> uint(shift)) & 15u] += 1u;
      }
    }
    for (int b = 0; b < 16; b++) {
      const uint c = metal::simd_sum(cnt[b]);
      if (lane == uint(b)) {
        part[sg][b] = c;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      // Lane r counts digit 15 - r; its inclusive prefix counts digits >= 15 - r.
      uint total = 0u;
      if (lane < 16) {
        for (int s = 0; s < 32; s++) {
          total += part[s][15 - lane];
        }
      }
      const uint incl = metal::simd_prefix_inclusive_sum(total);
      const uint r = metal::simd_min(lane < 16 && incl >= need ? lane : 31u);
      const uint above = metal::simd_shuffle(incl - total, ushort(r));
      if (lane == 0) {
        bcast[0] = 15u - r;
        bcast[1] = need - above;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix |= bcast[0] << uint(shift);
    need = bcast[1];
  }
  const uint threshold_key = prefix;

  // Deterministic order (as the native kernel): strictly greater keys in
  // index order fill [0, n_greater), threshold ties in index order the rest.
  const int seg = (P + 1023) / 1024;
  const int s0 = int(tid) * seg;
  const int s1 = metal::min(s0 + seg, P);
  uint local_g = 0, local_t = 0;
  for (int i = s0; i < s1; ++i) {
    const ushort bits = as_type<ushort>(rs[i]);
    const uint key = (bits & 0x8000) ? uint((~bits) & 0xffff) : uint(bits | 0x8000);
    local_g += key > threshold_key ? 1u : 0u;
    local_t += key == threshold_key ? 1u : 0u;
  }
  const uint pre_g = metal::simd_prefix_exclusive_sum(local_g);
  const uint pre_t = metal::simd_prefix_exclusive_sum(local_t);
  if (lane == 31) {
    part_g[sg] = pre_g + local_g;
    part_t[sg] = pre_t + local_t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const uint pg = part_g[lane];
    const uint pt = part_t[lane];
    const uint eg = metal::simd_prefix_exclusive_sum(pg);
    const uint et = metal::simd_prefix_exclusive_sum(pt);
    part_g[lane] = eg;
    part_t[lane] = et;
    if (lane == 31) {
      bcast[0] = eg + pg;   // total strictly greater
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint n_greater = bcast[0];
  uint pos_g = part_g[sg] + pre_g;
  uint pos_t = n_greater + part_t[sg] + pre_t;
  if (local_g > 0 || (local_t > 0 && pos_t < uint(TOPK))) {
    for (int j = s0; j < s1; ++j) {
      const ushort bits = as_type<ushort>(rs[j]);
      const uint key = (bits & 0x8000) ? uint((~bits) & 0xffff) : uint(bits | 0x8000);
      uint pos = uint(TOPK);
      if (key > threshold_key) {
        pos = pos_g++;
      } else if (key == threshold_key && pos_t < uint(TOPK)) {
        pos = pos_t++;
      }
      if (pos < uint(TOPK)) {
        // Selected pool row j -> its KPOOL token indices, or -1 when the
        // row is not a complete window before this query.
        const bool valid = j < pool_len && (j + 1) * KPOOL - 1 <= qp;
        for (int c = 0; c < KPOOL; c++) {
          ro[pos * KPOOL + c] = valid ? j * KPOOL + c + lp : -1;
        }
      }
    }
  }
  // The always-selected tail window, then -1 up to OUT_W.
  for (int col = TOPK * KPOOL + int(tid); col < OUT_W; col += 1024) {
    const int t = col - TOPK * KPOOL;
    int v = -1;
    if (t < TAIL_W) {
      const int tail_count = (qp + 1) % KPOOL;
      if (t < tail_count) {
        v = qp + 1 - tail_count + t + lp;
      }
    }
    ro[col] = v;
  }
"""


@lru_cache(maxsize=None)
def _dsa_select_kernel():
    return mx.fast.metal_kernel(
        name="glm5_dsa_select_topk",
        input_names=["scores", "meta", "left_padding"],
        output_names=["out"],
        source=_DSA_SELECT_SOURCE,
    )


def dsa_select_topk(
    scores: mx.array,
    topk: int,
    query_pos0: int,
    pool_len: int,
    left_padding: mx.array,
    kpool: int,
    tail_width: int,
    output_width: int,
) -> Optional[mx.array]:
    """Selected token indices of ``scores`` [1, L, P] (L <= 8): the native
    ``dsa_topk_indices(scores[:, None], topk)`` pool rows, expanded as
    Glm5NextIndexer.__call__ expands them. ``left_padding`` is the KV
    cache's [1] padding array (kept on device). Returns [1, 1, L,
    output_width] int32, or None when not covered."""
    if scores.ndim != 3 or scores.shape[0] != 1 or not 1 <= scores.shape[1] <= 8:
        return None
    _, L, P = scores.shape
    if not 1 <= topk <= P or scores.dtype not in (mx.bfloat16, mx.float16):
        return None
    if output_width < topk * kpool + tail_width:
        return None
    STATS["dsa_topk"] += 1
    return _dsa_select_kernel()(
        inputs=[
            scores,
            mx.array([P, query_pos0, pool_len], dtype=mx.int32),
            left_padding.astype(mx.int32) if left_padding.dtype != mx.int32 else left_padding,
        ],
        template=[
            ("T", scores.dtype),
            ("TOPK", topk),
            ("KPOOL", kpool),
            ("TAIL_W", tail_width),
            ("OUT_W", output_width),
        ],
        grid=(1024, L, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(1, 1, L, output_width)],
        output_dtypes=[mx.int32],
    )[0]


# The indexer's pool compress (Glm5NextIndexer._compress_windows): logits =
# gate + ape, a softmax over each window's KPOOL rows, then the probability-
# weighted sum of the window's keys. The softmax is not over the last axis,
# so MLX composes it (max, subtract, exp, sum, divide) and the weighted sum
# is a multiply and a sum: eight dispatches, every one rounded to the
# activation dtype, the sums accumulated in it (row by row from 0). One
# thread per (window, column) replays them; on a decode step that completes
# a window they precede the indexer scores.
_POOL_COMPRESS_SOURCE = r"""
  const int d = int(thread_position_in_grid.x);
  const int w = int(thread_position_in_grid.y);
  if (d >= HD) {
    return;
  }
  const int b = w / NWIN;
  const size_t base = (size_t(b) * ROWS + size_t(w - b * NWIN) * KP) * HD + d;
  T logits[KP];
  for (int j = 0; j < KP; j++) {
    logits[j] = static_cast<T>(
        static_cast<float>(gate[base + j * HD]) + static_cast<float>(ape[j * HD + d]));
  }
  T top = logits[0];
  for (int j = 1; j < KP; j++) {
    top = logits[j] > top ? logits[j] : top;
  }
  T ex[KP];
  for (int j = 0; j < KP; j++) {
    const T shifted = static_cast<T>(static_cast<float>(logits[j]) - static_cast<float>(top));
    ex[j] = static_cast<T>(metal::precise::exp(static_cast<float>(shifted)));
  }
  T total = static_cast<T>(0.0f);
  for (int j = 0; j < KP; j++) {
    total = static_cast<T>(static_cast<float>(ex[j]) + static_cast<float>(total));
  }
  T acc = static_cast<T>(0.0f);
  for (int j = 0; j < KP; j++) {
    const T prob = static_cast<T>(static_cast<float>(ex[j]) / static_cast<float>(total));
    const T prod = static_cast<T>(static_cast<float>(prob) * static_cast<float>(keys[base + j * HD]));
    acc = static_cast<T>(static_cast<float>(prod) + static_cast<float>(acc));
  }
  out[size_t(w) * HD + d] = acc;
"""


@lru_cache(maxsize=None)
def _pool_compress_kernel():
    return mx.fast.metal_kernel(
        name="glm5_pool_compress",
        input_names=["keys", "gate", "ape"],
        output_names=["out"],
        source=_POOL_COMPRESS_SOURCE,
    )


def pool_compress(keys: mx.array, gate: mx.array, ape: mx.array) -> Optional[mx.array]:
    """``Glm5NextIndexer._compress_windows(keys, gate)``: ``keys`` and
    ``gate`` [B, S, hd], ``ape`` [kpool, hd], all of one 16-bit float dtype.
    Returns [B, S // kpool, hd], or None when not covered."""
    if keys.ndim != 3 or gate.shape != keys.shape or ape.ndim != 2:
        return None
    B, S, hd = keys.shape
    kp = ape.shape[0]
    if ape.shape[1] != hd or not 1 <= hd <= 1024 or S < kp:
        return None
    if (
        keys.dtype not in (mx.bfloat16, mx.float16)
        or gate.dtype != keys.dtype
        or ape.dtype != keys.dtype
    ):
        return None
    n = S // kp
    STATS["pool_compress"] += 1
    return _pool_compress_kernel()(
        inputs=[keys, gate, ape],
        template=[("T", keys.dtype), ("KP", kp), ("HD", hd), ("NWIN", n), ("ROWS", S)],
        grid=(hd, B * n, 1),
        threadgroup=(hd, 1, 1),
        output_shapes=[(B, n, hd)],
        output_dtypes=[keys.dtype],
    )[0]


# One-token sparse attention: the selected latent rows and their validity in
# one dispatch instead of clip (maximum, minimum), take_along_axis and >= 0.
_DSA_GATHER_SOURCE = r"""
  const int j = int(threadgroup_position_in_grid.y) * ROWS + int(simdgroup_index_in_threadgroup);
  if (j >= W) {
    return;
  }
  const uint lane = thread_index_in_simdgroup;
  const int kv_len = int(kv_shape[2]);
  const int raw = idx[j];
  const int r = metal::min(metal::max(raw, 0), kv_len - 1);
  // 16-byte copies of the row.
  constexpr int NV = D * int(sizeof(T)) / 16;
  const device uint4* src = (const device uint4*)(kv + size_t(r) * D);
  device uint4* dst = (device uint4*)(out + size_t(j) * D);
  for (int c = int(lane); c < NV; c += 32) {
    dst[c] = src[c];
  }
  if (lane == 0) {
    valid[j] = raw >= 0;
  }
"""


@lru_cache(maxsize=None)
def _dsa_gather_kernel():
    return mx.fast.metal_kernel(
        name="glm5_dsa_gather_selected",
        input_names=["kv", "idx"],
        output_names=["out", "valid"],
        source=_DSA_GATHER_SOURCE,
    )


def dsa_gather_selected(kv_latent: mx.array, indices: mx.array):
    """``kv_latent`` [1, 1, Kv, D] rows at ``indices`` [1, 1, W] (int32, -1 =
    unused) clamped to [0, Kv - 1], and ``indices >= 0``: the one-token
    sparse attention's ``take_along_axis(kv_latent, clip(indices))`` and
    selection mask ([1, 1, 1, W] bool). None when not covered."""
    if kv_latent.ndim != 4 or kv_latent.shape[:2] != (1, 1) or indices.ndim != 3:
        return None
    if indices.shape[:2] != (1, 1) or indices.dtype != mx.int32:
        return None
    D = kv_latent.shape[3]
    W = indices.shape[2]
    if kv_latent.shape[2] < 1 or D % 8 or kv_latent.dtype not in (mx.bfloat16, mx.float16):
        return None
    rows = 8
    STATS["dsa_gather"] += 1
    out, valid = _dsa_gather_kernel()(
        inputs=[kv_latent, indices],
        template=[("T", kv_latent.dtype), ("D", D), ("W", W), ("ROWS", rows)],
        grid=(32, rows * ((W + rows - 1) // rows), 1),
        threadgroup=(32, rows, 1),
        output_shapes=[(1, 1, W, D), (1, 1, 1, W)],
        output_dtypes=[kv_latent.dtype, mx.bool_],
    )
    return out, valid


# One-token sparse attention without NAX: dsa_gather_selected feeds the
# unfused SDPA fallback (64 query heads share the latent head, so GQA x L >
# 32): qs = bf16(q * bf16(scale)); scores = qs @ g^T, a steel GEMM collapsed
# to M = heads (bf16 out, 65 threadgroups each running 16 staged K blocks);
# where(valid, scores, bf16 lowest); precise softmax; p @ g. Each score is a
# chain of fp32 8 x 8 simdgroup MMAs over D in order whatever the tiling, so
# one kernel gathers the selected rows by index (staging each threadgroup's
# TK rows once, and writing them out for the P x V GEMM as the gather did),
# scales the queries and computes the masked scores: gather, GEMM and where
# become one dispatch with one staging round trip per threadgroup.
_DSA_DECODE_ATTN_SOURCE = r"""
  constexpr int NSG = H / 8;          // one simdgroup per 8 heads
  constexpr int LDK = D + 8;
  constexpr int PER = TK * (D / 8) / (NSG * 32);
  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int tid = int(thread_index_in_threadgroup);
  const int k0 = int(threadgroup_position_in_grid.x) * TK;
  const int kv_len = int(kv_shape[2]);
  threadgroup T ks[TK * LDK];
  // This threadgroup's TK selected rows (clamped as dsa_gather_selected).
  int slot[PER];
  int src[PER];
  for (int u = 0; u < PER; u++) {
    slot[u] = k0 + (tid + u * NSG * 32) / (D / 8);
    const int raw = slot[u] < W ? idx[slot[u]] : 0;
    src[u] = metal::min(metal::max(raw, 0), kv_len - 1);
  }
  uint4 rows[PER];
  for (int u = 0; u < PER; u++) {
    const int d = ((tid + u * NSG * 32) % (D / 8)) * 8;
    rows[u] = *reinterpret_cast<const device uint4*>(kv + size_t(src[u]) * D + d);
  }
  for (int u = 0; u < PER; u++) {
    const int c = tid + u * NSG * 32;
    const int d = (c % (D / 8)) * 8;
    *reinterpret_cast<threadgroup uint4*>(ks + (c / (D / 8)) * LDK + d) = rows[u];
    if (slot[u] < W) {
      *reinterpret_cast<device uint4*>(gathered + size_t(slot[u]) * D + d) = rows[u];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float sbf = static_cast<float>(scale[0]);
  const short qid = short(lane / 4);
  const short fm = (qid & 4) + short((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + short(lane % 2) * 2;
  const device T* qh = q + size_t(sg * 8 + fm) * D + fn;
  simdgroup_matrix<float, 8, 8> acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  #pragma unroll(8)
  for (int kk = 0; kk < D; kk += 8) {
    simdgroup_matrix<float, 8, 8> a;
    simdgroup_matrix<float, 8, 8> b;
    // qs = bf16(q * bf16(scale)), the fallback's scaled queries.
    a.thread_elements()[0] =
        static_cast<float>(static_cast<T>(static_cast<float>(qh[kk]) * sbf));
    a.thread_elements()[1] =
        static_cast<float>(static_cast<T>(static_cast<float>(qh[kk + 1]) * sbf));
    b.thread_elements()[0] = static_cast<float>(ks[fn * LDK + kk + fm]);
    b.thread_elements()[1] = static_cast<float>(ks[(fn + 1) * LDK + kk + fm]);
    simdgroup_multiply_accumulate(acc, a, b, acc);
  }
  const T lowest = lowest_in[0];
  const int h = sg * 8 + fm;
  for (int e = 0; e < 2; e++) {
    const int k = k0 + fn + e;
    if (k < W) {
      const T v = static_cast<T>(acc.thread_elements()[e]);
      scores[size_t(h) * W + k] = idx[k] >= 0 ? v : lowest;
    }
  }
"""

_DSA_DECODE_ATTN_ROWS = 8  # TK: selected rows per threadgroup


@lru_cache(maxsize=None)
def _dsa_decode_attn_kernel():
    return mx.fast.metal_kernel(
        name="glm5_dsa_decode_attention_scores",
        input_names=["q", "kv", "idx", "scale", "lowest_in"],
        output_names=["scores", "gathered"],
        header=_QMV_HEADER,
        source=_DSA_DECODE_ATTN_SOURCE,
    )


def dsa_decode_attention_supported(heads: int, dim: int, dtype) -> bool:
    """Whether ``dsa_decode_attention`` covers these shapes on this GPU."""
    return (
        heads % 8 == 0
        and dim % (4 * heads) == 0
        and dtype in (mx.bfloat16, mx.float16)
        and not is_nax_available()
    )


def dsa_decode_attention(
    q_lat: mx.array, kv_latent: mx.array, indices: mx.array, scale: float
) -> Optional[mx.array]:
    """``dsa_gather_selected`` + ``mx.fast.scaled_dot_product_attention`` of
    one token: ``q_lat`` [1, H, 1, D] embedded queries, ``kv_latent`` [1, 1,
    Kv, D], ``indices`` [1, 1, W] int32 (-1 = unused). Returns [1, H, 1, D],
    or None when not covered."""
    if q_lat.ndim != 4 or q_lat.shape[0] != 1 or q_lat.shape[2] != 1:
        return None
    if kv_latent.ndim != 4 or kv_latent.shape[:2] != (1, 1) or kv_latent.shape[2] < 1:
        return None
    if indices.ndim != 3 or indices.shape[:2] != (1, 1) or indices.dtype != mx.int32:
        return None
    _, H, _, D = q_lat.shape
    W = indices.shape[2]
    # One selected row makes the reference scores product a GEMV.
    if kv_latent.shape[3] != D or kv_latent.dtype != q_lat.dtype or W < 2:
        return None
    if not dsa_decode_attention_supported(H, D, q_lat.dtype):
        return None
    tk = _DSA_DECODE_ATTN_ROWS
    STATS["dsa_decode_attention"] += 1
    scores, gathered = _dsa_decode_attn_kernel()(
        inputs=[
            q_lat,
            kv_latent,
            indices,
            mx.array([scale], dtype=q_lat.dtype),
            mx.array([mx.finfo(q_lat.dtype).min], dtype=q_lat.dtype),
        ],
        template=[("T", q_lat.dtype), ("H", H), ("D", D), ("W", W), ("TK", tk)],
        grid=(4 * H * ((W + tk - 1) // tk), 1, 1),
        threadgroup=(4 * H, 1, 1),
        output_shapes=[(1, 1, H, W), (1, 1, W, D)],
        output_dtypes=[q_lat.dtype, q_lat.dtype],
    )
    probs = mx.softmax(scores, axis=-1, precise=True)
    return (probs @ gathered).reshape(1, H, 1, D)


# ---------------------------------------------------------------------------
# DSA attention over the selected rows of a row-exact verify block
# ---------------------------------------------------------------------------
#
# Without NAX, a decode step's sparse attention is dsa_gather_selected plus
# the unfused SDPA fallback (64 query heads share the latent head, so
# GQA x L > 32): qs = bf16(q * bf16(scale)); scores = qs @ g^T, a steel GEMM
# collapsed to M = 64 (bf16 out); where(valid, scores, bf16 lowest);
# precise softmax; out = p @ g, a steel split-K GEMM (bk 16; fp32 partials
# of K partitions, the last one also taking the K tail as a zero-padded
# 16-wide tile; partials summed in order from 0). Every GEMM output is a
# chain of fp32 8 x 8 simdgroup MMAs over K in order, independent of the
# tiling, so two kernels with their own tiling replay the two GEMMs (the
# gather fused in, by index) around MLX's softmax: three dispatches for all
# rows of a verify block instead of six per row. (For one row they are no
# faster than the stock kernels; decode keeps those.)
_DSA_ROWS_SCORES_SOURCE = r"""
  constexpr int TH = 16;   // heads per threadgroup (two 8-row fragments)
  constexpr int TK = 32;   // selected rows per threadgroup
  constexpr int BD = 32;   // latent dims staged per step
  constexpr int LDS = BD + 8;
  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int tid = int(thread_index_in_threadgroup);
  const int k0 = int(threadgroup_position_in_grid.x) * TK;
  const int h0 = int(threadgroup_position_in_grid.y) * TH;
  const int row = int(threadgroup_position_in_grid.z);
  const int kv_len = int(kv_lens[row]);
  const device int* sel = idx + size_t(row) * W;
  const device T* qr = q + size_t(row) * H * D;
  threadgroup T qs[TH * LDS];
  threadgroup T ks[TK * LDS];
  const float sbf = static_cast<float>(scale[0]);
  const short qid = short(lane / 4);
  const short fm = (qid & 4) + short((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + short(lane % 2) * 2;
  // Four simdgroups: heads (sg / 2) x rows (sg % 2), 8 heads x 16 rows each.
  const int sh = sg / 2;
  const int sk = sg % 2;
  simdgroup_matrix<float, 8, 8> acc[2];
  acc[0] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  acc[1] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  for (int d0 = 0; d0 < D; d0 += BD) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int c = tid; c < TH * BD / 4; c += 128) {
      const int h = c / (BD / 4);
      const int d = (c % (BD / 4)) * 4;
      const device T* src = qr + size_t(h0 + h) * D + d0 + d;
      for (int e = 0; e < 4; e++) {
        qs[h * LDS + d + e] = static_cast<T>(static_cast<float>(src[e]) * sbf);
      }
    }
    for (int c = tid; c < TK * BD / 8; c += 128) {
      const int k = c / (BD / 8);
      const int d = (c % (BD / 8)) * 8;
      const int slot = k0 + k;
      const int raw = slot < W ? sel[slot] : 0;
      const int r = metal::min(metal::max(raw, 0), kv_len - 1);
      *reinterpret_cast<threadgroup uint4*>(ks + k * LDS + d) =
          *reinterpret_cast<const device uint4*>(kv + size_t(r) * D + d0 + d);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int kk = 0; kk < BD; kk += 8) {
      simdgroup_matrix<float, 8, 8> a;
      simdgroup_matrix<float, 8, 8> b[2];
      const int h = sh * 8 + fm;
      a.thread_elements()[0] = static_cast<float>(qs[h * LDS + kk + fn]);
      a.thread_elements()[1] = static_cast<float>(qs[h * LDS + kk + fn + 1]);
      for (int j = 0; j < 2; j++) {
        const int k = (sk * 2 + j) * 8 + fn;
        b[j].thread_elements()[0] = static_cast<float>(ks[k * LDS + kk + fm]);
        b[j].thread_elements()[1] = static_cast<float>(ks[(k + 1) * LDS + kk + fm]);
      }
      simdgroup_multiply_accumulate(acc[0], a, b[0], acc[0]);
      simdgroup_multiply_accumulate(acc[1], a, b[1], acc[1]);
    }
  }
  const T lowest = lowest_in[0];
  const int h = h0 + sh * 8 + fm;
  for (int j = 0; j < 2; j++) {
    const int k = k0 + (sk * 2 + j) * 8 + fn;
    for (int e = 0; e < 2; e++) {
      if (k + e < W) {
#if MASKED
        const bool ok = bool(valid[size_t(row) * W + k + e]);
#else
        const bool ok = sel[k + e] >= 0;
#endif
        const T v = static_cast<T>(acc[j].thread_elements()[e]);
        scores[(size_t(row) * H + h) * W + k + e] = ok ? v : lowest;
      }
    }
  }
"""

_DSA_ROWS_PV_SOURCE = r"""
  constexpr int TH = 16;   // heads per threadgroup
  constexpr int TD = 16;   // output dims per threadgroup
  const uint lane = thread_index_in_simdgroup;
  const int p = int(simdgroup_index_in_threadgroup);  // this simdgroup's K partition
  const int d0 = int(threadgroup_position_in_grid.x) * TD;
  const int h0 = int(threadgroup_position_in_grid.y) * TH;
  const int row = int(threadgroup_position_in_grid.z);
  const int kv_len = int(kv_lens[row]);
  const device int* sel = idx + size_t(row) * W;
  const device T* pr = probs + size_t(row) * H * W;
  const short qid = short(lane / 4);
  const short fm = (qid & 4) + short((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + short(lane % 2) * 2;
  simdgroup_matrix<float, 8, 8> acc[2][2];
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
  }
  const int kb = p * PSZ;
  const int kend = p == NP - 1 ? (W / 16) * 16 + (W % 16 ? 16 : 0) : kb + PSZ;
  // The next 8 rows' fragment values load while this step's MMAs run.
  float na[2][2];
  float nb[2][2];
  auto fetch = [&](int k) {
    const int ka = k + fn;
    for (int i = 0; i < 2; i++) {
      const device T* prow = pr + size_t(h0 + i * 8 + fm) * W;
      na[i][0] = ka < W ? static_cast<float>(prow[ka]) : 0.0f;
      na[i][1] = ka + 1 < W ? static_cast<float>(prow[ka + 1]) : 0.0f;
    }
    const int kr = k + fm;
    const int raw = kr < W ? sel[kr] : 0;
    const device T* vrow = kv + size_t(metal::min(metal::max(raw, 0), kv_len - 1)) * D;
    for (int j = 0; j < 2; j++) {
      const int d = d0 + j * 8 + fn;
      nb[j][0] = kr < W ? static_cast<float>(vrow[d]) : 0.0f;
      nb[j][1] = kr < W ? static_cast<float>(vrow[d + 1]) : 0.0f;
    }
  };
  fetch(kb);
  for (int k = kb; k < kend; k += 8) {
    simdgroup_matrix<float, 8, 8> a[2];
    simdgroup_matrix<float, 8, 8> b[2];
    for (int i = 0; i < 2; i++) {
      a[i].thread_elements()[0] = na[i][0];
      a[i].thread_elements()[1] = na[i][1];
      b[i].thread_elements()[0] = nb[i][0];
      b[i].thread_elements()[1] = nb[i][1];
    }
    if (k + 8 < kend) {
      fetch(k + 8);
    }
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
      }
    }
  }
  threadgroup float part[NP][TH][TD];
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      part[p][i * 8 + fm][j * 8 + fn] = acc[i][j].thread_elements()[0];
      part[p][i * 8 + fm][j * 8 + fn + 1] = acc[i][j].thread_elements()[1];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int e = p * 32 + int(lane); e < TH * TD; e += NP * 32) {
    const int hh = e / TD;
    const int dd = e % TD;
    float total = 0.0f;
    for (int q = 0; q < NP; q++) {
      total += part[q][hh][dd];
    }
    out[(size_t(row) * H + h0 + hh) * D + d0 + dd] = static_cast<T>(total);
  }
"""


@lru_cache(maxsize=None)
def _dsa_rows_scores_kernel(masked: bool):
    return mx.fast.metal_kernel(
        name="glm5_dsa_rows_scores" + ("_masked" if masked else ""),
        input_names=["q", "kv", "idx", "kv_lens", "scale", "lowest_in"]
        + (["valid"] if masked else []),
        output_names=["scores"],
        header=_QMV_HEADER,
        source=_source(_DSA_ROWS_SCORES_SOURCE, MASKED=int(masked)),
    )


@lru_cache(maxsize=None)
def _dsa_rows_pv_kernel():
    return mx.fast.metal_kernel(
        name="glm5_dsa_rows_pv",
        input_names=["probs", "kv", "idx", "kv_lens"],
        output_names=["out"],
        header=_QMV_HEADER,
        source=_DSA_ROWS_PV_SOURCE,
    )


def _splitk_partitions(M: int, N: int, K: int):
    """steel_gemm_splitk_axpby's K partitions (count, size) for bk = 16."""
    tm, tn, tk = (M + 31) // 32, (N + 31) // 32, K // 16
    want = max(1, tk // (tm * tn))
    parts = min(max(2, 1 << (want - 1).bit_length()), 32)
    return parts, ((K // 16) // parts) * 16


def dsa_attention_rows_supported(heads: int, dim: int, width: int, dtype) -> bool:
    """Whether ``dsa_attention_rows`` covers these shapes on this GPU."""
    return (
        heads % 16 == 0
        and dim % 32 == 0
        and width > 512
        and dtype in (mx.bfloat16, mx.float16)
        and not is_nax_available()
    )


def dsa_attention_rows(
    q_lat: mx.array,
    kv: mx.array,
    idx: mx.array,
    kv_lens: mx.array,
    scale: float,
    valid: Optional[mx.array] = None,
) -> Optional[mx.array]:
    """Sparse attention of R one-token rows over their selected latent rows,
    each row as its decode step computes it (gather + SDPA fallback).

    ``q_lat`` [R, H, D] (the embedded queries), ``kv`` [Kv, D] latent keys,
    ``idx`` [R, W] int32 selected rows (-1 = unused), ``kv_lens`` [R] int32
    (each row's keys, for the index clamp), ``valid`` [R, W] bool for a
    row's mask beyond ``idx >= 0``. Returns [R, H, D] or None when not
    covered (NAX GPUs run other GEMMs; W <= 512 changes their kernels)."""
    if q_lat.ndim != 3 or kv.ndim != 2 or idx.ndim != 2 or idx.dtype != mx.int32:
        return None
    R, H, D = q_lat.shape
    W = idx.shape[1]
    if kv.shape[1] != D or idx.shape[0] != R or kv_lens.shape != (R,):
        return None
    if kv.dtype != q_lat.dtype or not dsa_attention_rows_supported(
        H, D, W, q_lat.dtype
    ):
        return None
    masked = valid is not None
    inputs = [
        q_lat,
        kv,
        idx,
        kv_lens.astype(mx.int32),
        mx.array([scale], dtype=q_lat.dtype),
        mx.array([mx.finfo(q_lat.dtype).min], dtype=q_lat.dtype),
    ]
    if masked:
        inputs.append(valid)
    STATS["dsa_rows_attention"] += 1
    scores = _dsa_rows_scores_kernel(masked)(
        inputs=inputs,
        template=[("T", q_lat.dtype), ("H", H), ("D", D), ("W", W)],
        grid=(128 * ((W + 31) // 32), H // 16, R),
        threadgroup=(128, 1, 1),
        output_shapes=[(R, H, W)],
        output_dtypes=[q_lat.dtype],
    )[0]
    probs = mx.softmax(scores, axis=-1, precise=True)
    parts, size = _splitk_partitions(H, D, W)
    return _dsa_rows_pv_kernel()(
        inputs=[probs, kv, idx, kv_lens.astype(mx.int32)],
        template=[
            ("T", q_lat.dtype),
            ("H", H),
            ("D", D),
            ("W", W),
            ("NP", parts),
            ("PSZ", size),
        ],
        grid=(32 * parts * (D // 16), H // 16, R),
        threadgroup=(32 * parts, 1, 1),
        output_shapes=[(R, H, D)],
        output_dtypes=[q_lat.dtype],
    )[0]


# ---------------------------------------------------------------------------
# KDA (linear attention) decode/verify step
# ---------------------------------------------------------------------------
#
# Everything between the fused input projection and o_proj of a
# Glm5NextLinearAttention layer, for one sequence and T <= 8 tokens, in one
# dispatch with one 1024-thread threadgroup per head:
#
#   1. depthwise short conv over [conv_state, q|k|v] + SiLU (bf16), and the
#      new conv state (MLX depthwise_conv_1d + the compiled nn.silu);
#   2. l2-normalization of q (with the 1/sqrt(Dk) scale) and k (fp32, the
#      row_reduce_simple order of MLX's Sum);
#   3. the forget-gate / output-gate low-rank projections (MLX qmv_quad for
#      K == 128), the safe gate g (compiled compute_g_safe) and beta =
#      sigmoid(b);
#   4. the vector-gated delta rule (the vendored gated_delta_step_vec
#      kernel, statement for statement);
#   5. Glm5NextRMSNormGated (fp32, row_reduce_simple order, separate
#      product/sum roundings).
#
# Each reference op is its own kernel there, so every intermediate is rounded
# to its dtype here too and products are never contracted into the adds that
# consumed them in a different kernel.
_KDA_SOURCE = r"""
  constexpr int CK = 4;
  constexpr int NROW = CK - 1 + TOK;
  constexpr int NP = 3 * QKV;
  const uint tid = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const int h = int(threadgroup_position_in_grid.x);
  const float q_scale = consts[0];
  const float l2_eps = consts[1];
  const float norm_eps = consts[2];
  const float lower = consts[3];
  const float inv_n = consts[4];

  threadgroup T qs[TOK][DK];
  threadgroup T ks[TOK][DK];
  threadgroup T vs[TOK][DK];
  threadgroup T as_[TOK][DK];
  threadgroup T gates[TOK][DK];
  threadgroup T ys[TOK][DK];
  threadgroup float gs[TOK][DK];
  threadgroup T betas[TOK];

  // ---- 1. short conv + SiLU -------------------------------------------------
  if (tid < uint(3 * DK)) {
    const int part = int(tid) / DK;
    const int i = int(tid) % DK;
    const int gc = part * QKV + h * DK + i;
    T win[NROW];
    for (int r = 0; r < CK - 1; r++) {
#if HAS_CONV_STATE
      win[r] = conv_state[r * NP + gc];
#else
      win[r] = static_cast<T>(0);
#endif
    }
    for (int t = 0; t < TOK; t++) {
      win[CK - 1 + t] = proj[t * PROJ_W + gc];
    }
    const device T* w = conv_w + gc * CK;
    for (int t = 0; t < TOK; t++) {
      float acc = 0.0;
      for (int j = 0; j < CK; ++j) {
        acc += static_cast<float>(win[t + j]) * w[j];
      }
      T co = static_cast<T>(acc);
      T sgm = glm_sigmoid<T>(co);
      T sv = co * sgm;
      if (part == 0) {
        qs[t][i] = sv;
      } else if (part == 1) {
        ks[t][i] = sv;
      } else {
        vs[t][i] = sv;
      }
    }
    for (int r = 0; r < CK - 1; r++) {
      conv_state_out[r * NP + gc] = win[TOK + r];
    }
  }

  // ---- 3a. low-rank gate projections (qmv_quad rows of this head) ----------
#if PRE_AG
  for (int e = int(tid); e < TOK * DK; e += 1024) {
    const int t = e / DK;
    const int i = e % DK;
    as_[t][i] = a_pre[t * QKV + h * DK + i];
    gates[t][i] = gate_pre[t * QKV + h * DK + i];
  }
#elif GATE5
  // One token, 5-bit K = 128 rows: MLX's qmv (qmv_impl): lanes 0..15 load
  // 8 values each (load_vector_safe / qdot_safe with N = 8, i.e. load_vector
  // / qdot), lanes 16..31 add nothing, one simd_sum per row.
  {
    static_assert(TOK == 1 && DK == 128, "5-bit gate rows: one token");
    constexpr int WBYTES = 128 * 5 / 8;           // 80 bytes per weight row
    constexpr int G = 128 / GS;                   // groups per row
    for (int rr = 0; rr < (2 * DK) / 32; rr++) {
      const int q = int(sg) * ((2 * DK) / 32) + rr;
      const int which = q / DK;
      const int i = q % DK;
      const int row = h * DK + i;
      float result = 0;
      if (lane < 16u) {
        const device T* xin = proj + (which == 0 ? OFF_FA : OFF_GA) + int(lane) * 8;
        float x_thread[8];
        float sum = glm_load_vector<T, 8, 5>(xin, x_thread);
        const device uint8_t* wl = (const device uint8_t*)(which == 0 ? fb_w : gb_w)
            + size_t(row) * WBYTES + int(lane) * 5;
        const device T* sl = (which == 0 ? fb_s : gb_s) + row * G + int(lane) / (GS / 8);
        const device T* bl = (which == 0 ? fb_b : gb_b) + row * G + int(lane) / (GS / 8);
        const float s = sl[0];
        const float b = bl[0];
        result += glm_qdot<8, 5>(wl, x_thread, s, b, sum);
      }
      result = simd_sum(result);
      if (lane == 0) {
        if (which == 0) {
          as_[0][i] = static_cast<T>(result);
        } else {
          gates[0][i] = static_cast<T>(result);
        }
      }
    }
  }
#else
  {
    constexpr int VPT = 32;                       // values per thread (K = 128)
    constexpr int WBYTES = 128 * BITS / 8;        // bytes per weight row
    constexpr int G = 128 / GS;                   // groups per row
    const int quad = int(tid) / 4;
    const int ql = int(tid) % 4;
    const int which = quad / DK;
    const int i = quad % DK;
    const int row = h * DK + i;
    const device uint8_t* wl = (const device uint8_t*)(which == 0 ? fb_w : gb_w)
        + size_t(row) * WBYTES + ql * (VPT * BITS / 8);
    const device T* sl = (which == 0 ? fb_s : gb_s) + row * G + ql / (GS / VPT);
    const device T* bl = (which == 0 ? fb_b : gb_b) + row * G + ql / (GS / VPT);
    const float s = sl[0];
    const float b = bl[0];
    for (int t = 0; t < TOK; t++) {
      const device T* xin = proj + t * PROJ_W + (which == 0 ? OFF_FA : OFF_GA) + ql * VPT;
      float x_thread[VPT];
      float sum = glm_load_vector<T, VPT, BITS>(xin, x_thread);
      float result = 0;
      result += glm_qdot<VPT, BITS>(wl, x_thread, s, b, sum);
      result = quad_sum(result);
      if (ql == 0) {
        if (which == 0) {
          as_[t][i] = static_cast<T>(result);
        } else {
          gates[t][i] = static_cast<T>(result);
        }
      }
    }
  }
#endif
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- 2. l2norm(q) * scale, l2norm(k) --------------------------------------
  if (sg < uint(2 * TOK)) {
    const int t = int(sg) / 2;
    const bool is_q = (sg % 2) == 0;
    threadgroup T* row = is_q ? qs[t] : ks[t];
    float x[4];
    float tot = 0.0f;
    for (int e = 0; e < 4; e++) {
      x[e] = static_cast<float>(row[4 * lane + e]);
      float sq = x[e] * x[e];
      tot = sq + tot;
    }
    tot = simd_sum(tot);
    float u = tot + l2_eps;
    float r = metal::precise::rsqrt(u);
    for (int e = 0; e < 4; e++) {
      float xn = x[e] * r;
      if (is_q) {
        float xs = xn * q_scale;
        row[4 * lane + e] = static_cast<T>(xs);
      } else {
        row[4 * lane + e] = static_cast<T>(xn);
      }
    }
  }

  // ---- 3b. g = exp(lower * sigmoid(exp(A_log) * (a + dt_bias))), beta -------
  if (tid < uint(TOK * DK)) {
    const int t = int(tid) / DK;
    const int i = int(tid) % DK;
    float ea = metal::precise::exp(a_log[h]);
    float af = static_cast<float>(as_[t][i]);
    float s1 = af + dt_bias[h * DK + i];
    float s2 = ea * s1;
    float s3 = glm_sigmoid<float>(s2);
    float s4 = lower * s3;
    gs[t][i] = metal::precise::exp(s4);
  }
  if (tid < uint(TOK)) {
    betas[tid] = glm_sigmoid<T>(proj[tid * PROJ_W + OFF_B + h]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- 4. vector-gated delta rule --------------------------------------------
  for (int j = 0; j < DK / 32; j++) {
    const int dv_idx = int(sg) + 32 * j;
    constexpr int n_per_t = DK / 32;
    const int dk_idx = int(lane);
    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
#if HAS_STATE
      state[i] = static_cast<float>(state_in[(size_t(h) * DK + dv_idx) * DK + s_idx]);
#else
      state[i] = 0.0f;
#endif
    }
    for (int t = 0; t < TOK; ++t) {
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] * gs[t][s_idx];
        kv_mem += state[i] * ks[t][s_idx];
      }
      kv_mem = simd_sum(kv_mem);

      auto delta = (vs[t][dv_idx] - kv_mem) * betas[t];

      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] + ks[t][s_idx] * delta;
        out += state[i] * qs[t][s_idx];
      }
      out = simd_sum(out);
      if (thread_index_in_simdgroup == 0) {
        ys[t][dv_idx] = static_cast<T>(out);
      }
    }
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state_out[(size_t(h) * DK + dv_idx) * DK + s_idx] = static_cast<float>(state[i]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- 5. RMSNormGated ---------------------------------------------------------
  if (sg < uint(TOK)) {
    const int t = int(sg);
    float x[4];
    float tot = 0.0f;
    for (int e = 0; e < 4; e++) {
      x[e] = static_cast<float>(ys[t][4 * lane + e]);
      float sq = x[e] * x[e];
      tot = sq + tot;
    }
    tot = simd_sum(tot);
    float var = tot * inv_n;
    float u = var + norm_eps;
    float r = metal::precise::rsqrt(u);
    for (int e = 0; e < 4; e++) {
      const int c = 4 * lane + e;
      float xn = x[e] * r;
      float wf = static_cast<float>(norm_w[c]);
      float wx = wf * xn;
      float gf = static_cast<float>(gates[t][c]);
      float gsg = glm_sigmoid<float>(gf);
      float o = wx * gsg;
      y[t * QKV + h * DK + c] = static_cast<T>(o);
    }
  }
"""


@lru_cache(maxsize=None)
def _kda_kernel(
    has_conv_state: bool,
    has_state: bool,
    pre_ag: bool,
    gate5: bool = False,
):
    inputs = ["proj", "conv_w", "a_log", "dt_bias", "norm_w", "consts"]
    if has_conv_state:
        inputs.append("conv_state")
    if has_state:
        inputs.append("state_in")
    if pre_ag:
        inputs += ["a_pre", "gate_pre"]
    else:
        inputs += ["fb_w", "fb_s", "fb_b", "gb_w", "gb_s", "gb_b"]
    return mx.fast.metal_kernel(
        name=(
            f"glm5_kda_decode_c{int(has_conv_state)}_s{int(has_state)}_p{int(pre_ag)}"
            f"{'_q5' if gate5 else ''}"
        ),
        input_names=inputs,
        output_names=["y", "conv_state_out", "state_out"],
        header=_QMV_HEADER,
        source=_source(
            _KDA_SOURCE,
            HAS_CONV_STATE=int(has_conv_state),
            HAS_STATE=int(has_state),
            PRE_AG=int(pre_ag),
            GATE5=int(gate5),
        ),
    )


def kda_decode_step(
    proj: mx.array,
    conv_state: Optional[mx.array],
    conv_w: mx.array,
    a_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array],
    norm_w: mx.array,
    *,
    heads: int,
    head_dim: int,
    off_fa: int,
    off_ga: int,
    off_b: int,
    q_scale: float,
    l2_eps: float,
    norm_eps: float,
    lower_bound: float,
    f_b=None,
    g_b=None,
    a_pre: Optional[mx.array] = None,
    gate_pre: Optional[mx.array] = None,
):
    """Fused KDA layer body for one sequence and T <= 8 tokens.

    ``proj`` is the fused q|k|v|f_a|g_a|b projection [1, T, W] (bf16/fp16).
    The forget/output gate projections are either the affine quantized
    ``f_b``/``g_b`` layers (K == 128, 4- or 8-bit: MLX's qmv_quad path) or
    precomputed ``a_pre``/``gate_pre`` [1, T, H * Dk]. Returns
    ``(y [1, T, H * Dk], conv_state [1, 3, 3 * H * Dk], state [1, H, Dk, Dk])``
    or None when the shapes are not covered.
    """
    if proj.ndim != 3 or proj.shape[0] != 1 or proj.dtype not in (mx.bfloat16, mx.float16):
        return None
    _, T, width = proj.shape
    qkv = heads * head_dim
    if not 1 <= T <= 8 or head_dim != 128 or heads < 1:
        return None
    if conv_w.shape != (3 * qkv, 4, 1) or conv_w.dtype != proj.dtype:
        return None
    if norm_w.shape != (head_dim,) or norm_w.dtype != proj.dtype:
        return None
    if a_log.size != heads or a_log.dtype != mx.float32:
        return None
    if dt_bias.size != qkv or dt_bias.dtype != mx.float32:
        return None
    if conv_state is not None and (
        conv_state.shape != (1, 3, 3 * qkv) or conv_state.dtype != proj.dtype
    ):
        return None
    if state is not None and (
        state.shape != (1, heads, head_dim, head_dim) or state.dtype != mx.float32
    ):
        return None
    pre = a_pre is not None
    inputs_extra = []
    template = [("T", proj.dtype), ("TOK", T), ("DK", head_dim), ("QKV", qkv),
                ("PROJ_W", width), ("OFF_FA", off_fa), ("OFF_GA", off_ga), ("OFF_B", off_b)]
    if pre:
        if gate_pre is None or a_pre.shape != (1, T, qkv) or gate_pre.shape != (1, T, qkv):
            return None
        if a_pre.dtype != proj.dtype or gate_pre.dtype != proj.dtype:
            return None
        inputs_extra = [a_pre, gate_pre]
        template += [("BITS", 8), ("GS", 64)]
    else:
        parts = [_affine_parts(m) for m in (f_b, g_b)]
        if any(p is None for p in parts):
            return None
        (fw, fs, fbias, fbits, fgs), (gw, gs_, gbias, gbits, ggs) = parts
        if (fbits, fgs) != (gbits, ggs) or fgs not in (32, 64, 128):
            return None
        # 4/8 bits: MLX's qmv_quad (any token count). 5 bits: the one-row
        # qmv (more rows take qmv_wide, which is not replayed).
        if fbits not in (4, 8) and not (fbits == 5 and T == 1):
            return None
        if fw.shape != (qkv, 128 * fbits // 32) or gw.shape != fw.shape:
            return None
        if fs.dtype != proj.dtype or gs_.dtype != proj.dtype:
            return None
        inputs_extra = [fw, fs, fbias, gw, gs_, gbias]
        template += [("BITS", fbits), ("GS", fgs)]
    consts = mx.array(
        [q_scale, l2_eps, norm_eps, lower_bound, 1.0 / head_dim], dtype=mx.float32
    )
    inputs = [proj, conv_w, a_log.reshape(-1), dt_bias.reshape(-1), norm_w, consts]
    if conv_state is not None:
        inputs.append(conv_state)
    if state is not None:
        inputs.append(state)
    inputs += inputs_extra
    gate5 = not pre and dict(template)["BITS"] == 5
    kernel = _kda_kernel(conv_state is not None, state is not None, pre, gate5)
    STATS["kda"] += 1
    if gate5:
        STATS["kda_gate5"] += 1
    y, conv_out, state_out = kernel(
        inputs=inputs,
        template=template,
        grid=(1024 * heads, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(1, T, qkv), (1, 3, 3 * qkv), (1, heads, head_dim, head_dim)],
        output_dtypes=[proj.dtype, proj.dtype, mx.float32],
    )
    return y, conv_out, state_out


# ---------------------------------------------------------------------------
# MoE router (one token): logits GEMV + sigmoid + bias, then top-k selection
# ---------------------------------------------------------------------------
#
# The logits reproduce MLX's non-transposed fp32 gemv for x @ W.T with
# 16 <= E < 4096 outputs and K < 16 * E (BM=4, BN=1, SM=1, SN=32, TM=4, TN=4):
# each simdgroup owns 4 rows, every lane accumulates 4 contiguous products
# per 128-wide K block, then a shuffle-down ladder. The epilogue applies the
# Sigmoid functor and the correction bias as separate roundings (they are
# separate kernels in the reference). The select kernel reproduces
# argpartition (a stable ascending merge sort of -(sigmoid + bias), i.e.
# descending scores with ties to the lower expert index), the gathered
# sigmoid scores, their sequential sum (row_reduce_small), the division and
# the routed scaling factor.
_ROUTER_LOGITS_SOURCE = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const int tok = int(threadgroup_position_in_grid.y);
  // One row per simdgroup: MLX's gemv gives each thread TM = 4 rows, but
  // every row's per-lane products and shuffle ladder are independent of TM.
  constexpr int RPS = ROWS_PER_SIMD;
  const int out_row = (int(threadgroup_position_in_grid.x) * 4 + int(sg)) * RPS;
  if (out_row >= E) {
    return;
  }
  const device auto* mat = w + size_t(out_row) * K;
  const device T* xv = x + size_t(tok) * K;
  float result[RPS];
  for (int tm = 0; tm < RPS; tm++) {
    result[tm] = 0.0f;
  }
  int bn = int(lane) * 4;
  for (int i = 0; i < K / 128; ++i) {
    float v_coeff[4];
    for (int tn = 0; tn < 4; tn++) {
      v_coeff[tn] = static_cast<float>(xv[bn + tn]);
    }
    int mat_offset = 0;
    for (int tm = 0; tm < RPS; tm++) {
      float inter[4];
      for (int tn = 0; tn < 4; tn++) {
        inter[tn] = static_cast<float>(mat[mat_offset + bn + tn]);
      }
      for (int tn = 0; tn < 4; tn++) {
        result[tm] += inter[tn] * v_coeff[tn];
      }
      mat_offset += K;
    }
    bn += 128;
  }
  for (int tm = 0; tm < RPS; tm++) {
    for (ushort sn = 16; sn >= 1; sn >>= 1) {
      result[tm] += simd_shuffle_down(result[tm], sn);
    }
  }
  if (lane == 0) {
    for (int tm = 0; tm < RPS; tm++) {
      const int e = out_row + tm;
      float sgm = glm_sigmoid<float>(result[tm]);
      float biased = sgm + bias[e];
      sig[size_t(tok) * E + e] = sgm;
      biased_out[size_t(tok) * E + e] = biased;
    }
  }
"""

_ROUTER_SELECT_SOURCE = r"""
  const uint lane = thread_index_in_simdgroup;
  const int tok = int(threadgroup_position_in_grid.x);
  constexpr int PER = (E + 31) / 32;
  const device float* bz = biased + size_t(tok) * E;
  const device float* sz = sig + size_t(tok) * E;
  float vals[PER];
  bool taken[PER];
  for (int j = 0; j < PER; j++) {
    const int e = j * 32 + int(lane);
    vals[j] = e < E ? bz[e] : -INFINITY;
    taken[j] = e >= E;
  }
  int picked[TOPK];
  for (int r = 0; r < TOPK; r++) {
    // Best remaining candidate of this lane: highest value, lowest index.
    float best = -INFINITY;
    int best_e = 0x7fffffff;
    for (int j = 0; j < PER; j++) {
      const int e = j * 32 + int(lane);
      if (!taken[j] && !isnan(vals[j]) &&
          (best_e == 0x7fffffff || vals[j] > best || (vals[j] == best && e < best_e))) {
        best = vals[j];
        best_e = e;
      }
    }
    for (ushort off = 16; off >= 1; off >>= 1) {
      float ob = simd_shuffle_xor(best, off);
      int oe = simd_shuffle_xor(best_e, off);
      const bool other_better = oe != 0x7fffffff &&
          (best_e == 0x7fffffff || ob > best || (ob == best && oe < best_e));
      if (other_better) {
        best = ob;
        best_e = oe;
      }
    }
    if (best_e == 0x7fffffff) {
      // Only NaNs remain (uniform branch): argpartition's sort places them
      // after every number, lowest index first.
      for (int j = 0; j < PER; j++) {
        const int e = j * 32 + int(lane);
        if (!taken[j] && e < best_e) {
          best_e = e;
        }
      }
      for (ushort off = 16; off >= 1; off >>= 1) {
        best_e = min(best_e, simd_shuffle_xor(best_e, off));
      }
    }
    picked[r] = best_e;
    if ((best_e % 32) == int(lane)) {
      taken[best_e / 32] = true;
    }
  }
  if (lane == 0) {
    float total = 0.0f;
    float gathered[TOPK];
    for (int r = 0; r < TOPK; r++) {
      gathered[r] = sz[picked[r]];
      total = gathered[r] + total;
    }
    for (int r = 0; r < TOPK; r++) {
      float q = NORM ? gathered[r] / total : gathered[r];
      float s = q * scaling[0];
      indices[tok * TOPK + r] = uint(picked[r]);
      scores[tok * TOPK + r] = s;
    }
  }
"""


@lru_cache(maxsize=None)
def _router_logits_kernel():
    return mx.fast.metal_kernel(
        name="glm5_router_logits_sigmoid",
        input_names=["x", "w", "bias"],
        output_names=["sig", "biased_out"],
        header=_QMV_HEADER,
        source=_ROUTER_LOGITS_SOURCE,
    )


@lru_cache(maxsize=None)
def _router_select_kernel():
    return mx.fast.metal_kernel(
        name="glm5_router_select",
        input_names=["sig", "biased", "scaling"],
        output_names=["indices", "scores"],
        source=_ROUTER_SELECT_SOURCE,
    )


def moe_router_logits(x: mx.array, weight: mx.array, bias: mx.array):
    """The router logits kernel alone: ``(sigmoid(x @ W.T), sigmoid + bias)``
    [T, E] fp32 with ``moe_router``'s arithmetic, or None when not covered.
    ``weight`` may be ``router_weights``' BF16 copy."""
    if x.ndim != 2 or weight.ndim != 2 or bias.ndim != 1:
        return None
    T, K = x.shape
    E = weight.shape[0]
    if weight.shape[1] != K or bias.shape[0] != E:
        return None
    if weight.dtype not in (mx.float32, mx.bfloat16) or bias.dtype != mx.float32:
        return None
    if x.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        return None
    if E < 16 or E >= 4096 or K >= 16 * E or K <= 64 or K % 128 or E % 16 or E > 1024:
        return None
    sig, biased = _router_logits_kernel()(
        inputs=[x, weight, bias],
        template=[("T", x.dtype), ("K", K), ("E", E), ("ROWS_PER_SIMD", 1)],
        grid=(128 * (E // 4), T, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(T, E), (T, E)],
        output_dtypes=[mx.float32, mx.float32],
    )
    STATS["router"] += 1
    return sig, biased


def moe_router(
    x: mx.array,
    weight: mx.array,
    bias: mx.array,
    top_k: int,
    scaling: float,
    norm_topk_prob: bool,
):
    """``group_expert_select(x.astype(f32) @ weight.T, bias, ...)`` for n_group == 1.

    ``x`` [T, K] (one-token rows; bf16/fp16/fp32), ``weight`` [E, K] fp32
    (or ``router_weights``' BF16 copy of it), ``bias`` [E] fp32. Returns
    ``(indices uint32 [T, top_k], scores fp32 [T, top_k])`` bit-identical to
    the reference for rows that the reference computes with the one-token
    gemv, or None when not covered.
    """
    if x.ndim != 2 or weight.ndim != 2 or bias.ndim != 1:
        return None
    T, K = x.shape
    E = weight.shape[0]
    if weight.shape[1] != K or bias.shape[0] != E:
        return None
    if weight.dtype not in (mx.float32, mx.bfloat16) or bias.dtype != mx.float32:
        return None
    if x.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        return None
    # Config of the reference gemv (see gemv_axbpy): bm=4, bn=1 needs
    # E < 4096 and K < 16 * E; full 128-wide blocks and whole 16-row tiles.
    if E < 16 or E >= 4096 or K >= 16 * E or K <= 64 or K % 128 or E % 16:
        return None
    if not 1 <= top_k <= min(32, E) or E > 1024:
        return None
    rows_per_simd = 1
    sig, biased = _router_logits_kernel()(
        inputs=[x, weight, bias],
        template=[("T", x.dtype), ("K", K), ("E", E), ("ROWS_PER_SIMD", rows_per_simd)],
        grid=(128 * (E // (4 * rows_per_simd)), T, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(T, E), (T, E)],
        output_dtypes=[mx.float32, mx.float32],
    )
    threads = 32  # one simdgroup per token (eight selection rounds)
    indices, scores = _router_select_kernel()(
        inputs=[sig, biased, mx.array([scaling], dtype=mx.float32)],
        template=[("E", E), ("TOPK", top_k), ("NORM", int(bool(norm_topk_prob) and top_k > 1))],
        grid=(threads * T, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(T, top_k), (T, top_k)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    STATS["router"] += 1
    return indices, scores


# ---------------------------------------------------------------------------
# Hyper-connection expand for one token (L == 1)
# ---------------------------------------------------------------------------
#
# The one-token reference (``hyper_connection._hc_expand_op``) computes
#   bf16(post * float(x) + comb^T @ float(residual))
# where the [HC, HC] x [HC, D] fp32 product runs on MLX's NAX steel GEMM,
# i.e. an MPP matmul2d with relaxed precision, and on GPUs without NAX on
# the steel GEMM's 8x8 simdgroup MMAs. These kernels issue the same matmul2d
# (16x32x16, relaxed) or simdgroup MMA on the same zero-padded fragments
# (bit-identical to mx.matmul for this shape) and apply the compiled epilogue
# with its separate multiply/add roundings: one dispatch instead of cast +
# GEMM + elementwise.
_NAX_HEADER = r"""
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
"""

_HC_EXPAND1_SOURCE = r"""
  const ushort lane = thread_index_in_simdgroup;
  const int tile = int(threadgroup_position_in_grid.x) * SIMDS + int(simdgroup_index_in_threadgroup);
  if (tile * 32 >= D) {
    return;
  }
  const short qid = lane >> 2;
  const short fm = ((qid & 4) | ((lane >> 1) & 3));
  const short fn = ((qid & 2) | (lane & 1)) * 4;
  constexpr auto desc = matmul2d_descriptor(
      16, 32, 16, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto ct_a = op.template get_left_input_cooperative_tensor<float, float, float>();
  auto ct_b = op.template get_right_input_cooperative_tensor<float, float, float>();
  auto ct_c = op.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(ct_a)>,
      metal::remove_addrspace_t<decltype(ct_b)>,
      float>();
  for (short i = 0; i < 8; i++) {
    const short r = fm + (i >> 2) * 8;
    const short c = fn + (i & 3);
    // A = comb^T (rows: output stream, cols: source stream), zero padded.
    ct_a[i] = (r < HC && c < HC) ? comb[c * HC + r] : 0.0f;
    ct_b[i] = (r < HC) ? static_cast<float>(residual[r * D + tile * 32 + c]) : 0.0f;
    ct_b[8 + i] = (r < HC) ? static_cast<float>(residual[r * D + tile * 32 + 16 + c]) : 0.0f;
    ct_c[i] = 0.0f;
    ct_c[8 + i] = 0.0f;
  }
  op.run(ct_a, ct_b, ct_c);
  for (short i = 0; i < 8; i++) {
    const short r = fm + (i >> 2) * 8;
    const short c = fn + (i & 3);
    if (r < HC) {
      for (short hh = 0; hh < 2; hh++) {
        const int col = tile * 32 + hh * 16 + c;
        const float mm = ct_c[hh * 8 + i];
        // Separate roundings, as in the compiled reference epilogue (the
        // MPP headers enable FP contraction for the whole kernel).
        volatile float prod = post[r] * static_cast<float>(x[col]);
        float sum = prod + mm;
        out[r * D + col] = static_cast<T>(sum);
      }
    }
  }
"""

# The comb product of one 32-column tile on 8x8 simdgroup MMAs, as the steel
# GEMM computes it without NAX: A = comb^T and B = residual rows zero padded
# to 8x8 fragments, one multiply-accumulate into zero per 8 columns. A lane
# holds steel's BaseMMAFrag elements: row ``fm``, columns ``fn`` and ``fn +
# 1`` of each fragment, i.e. ``mm[t][j]`` is output row fm, column
# tile * 32 + t * 8 + fn + j (a real element when fm < HC).
_SIMD_COMB_HEADER = r"""
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

template <int HC, int D, typename CombPtr, typename T>
inline void glm_hc_comb_simd_mma(
    CombPtr comb, const device T* residual, int tile, ushort lane,
    thread short& fm, thread short& fn, thread float (&mm)[4][2]) {
  const short qid = lane / 4;
  fm = (qid & 4) + ((lane / 2) % 4);
  fn = (qid & 2) * 2 + (lane % 2) * 2;
  simdgroup_matrix<float, 8, 8> a;
  for (short j = 0; j < 2; j++) {
    const short c = fn + j;
    a.thread_elements()[j] = (fm < HC && c < HC) ? float(comb[c * HC + fm]) : 0.0f;
  }
  for (short t = 0; t < 4; t++) {
    const int col0 = tile * 32 + t * 8;
    simdgroup_matrix<float, 8, 8> b;
    simdgroup_matrix<float, 8, 8> c = simdgroup_matrix<float, 8, 8>(0.0f);
    for (short j = 0; j < 2; j++) {
      b.thread_elements()[j] =
          fm < HC ? static_cast<float>(residual[fm * D + col0 + fn + j]) : 0.0f;
    }
    simdgroup_multiply_accumulate(c, a, b, c);
    for (short j = 0; j < 2; j++) {
      mm[t][j] = c.thread_elements()[j];
    }
  }
}
"""

_HC_EXPAND1_SIMD_SOURCE = r"""
  const ushort lane = thread_index_in_simdgroup;
  const int tile = int(threadgroup_position_in_grid.x) * SIMDS + int(simdgroup_index_in_threadgroup);
  if (tile * 32 >= D) {
    return;
  }
  short fm, fn;
  float mm[4][2];
  glm_hc_comb_simd_mma<HC, D>(comb, residual, tile, lane, fm, fn, mm);
  if (fm < HC) {
    for (short t = 0; t < 4; t++) {
      for (short j = 0; j < 2; j++) {
        const int col = tile * 32 + t * 8 + fn + j;
        // Separate roundings, as in the compiled reference epilogue.
        volatile float prod = post[fm] * static_cast<float>(x[col]);
        float sum = prod + mm[t][j];
        out[fm * D + col] = static_cast<T>(sum);
      }
    }
  }
"""


def _atoi(text: str) -> int:
    """C ``atoi`` (how MLX parses its integer environment switches)."""
    m = re.match(r"\s*([+-]?\d+)", text)
    return int(m.group(1)) if m else 0


@lru_cache(maxsize=None)
def fused_decode_supported() -> bool:
    """True on gen-15+ GPUs (M3 and newer). From gen 15 MLX runs multi-row
    affine qmv as qmv_wide, which these kernels replay, and the 1024-thread
    kernels fit the pipeline limit; on older GPUs neither holds."""
    if not mx.metal.is_available():
        return False
    arch = str(mx.device_info().get("architecture", ""))
    match = re.match(r"applegpu_g(\d+)", arch)
    return match is not None and int(match.group(1)) >= 15


@lru_cache(maxsize=None)
def nax_relaxed_fp32_matmul() -> bool:
    """True when MLX runs fp32 GEMMs on NAX with relaxed (TF32) precision:
    NAX available and ``env::enable_tf32()`` (MLX_ENABLE_TF32, default 1).
    Kernels that reproduce the fp32 NAX product are only valid then."""
    value = os.environ.get("MLX_ENABLE_TF32")
    if value is not None and _atoi(value) == 0:
        return False
    return is_nax_available()


@lru_cache(maxsize=None)
def hc_comb_product() -> Optional[str]:
    """How MLX runs the one-token fp32 comb product of the HC expand
    ([HC, HC] x [HC, D]): ``"nax"`` (relaxed matmul2d), ``"simd"`` (8x8
    simdgroup MMAs, GPUs without NAX) or None when no kernel here replays it
    (NAX without TF32)."""
    if nax_relaxed_fp32_matmul():
        return "nax"
    if not is_nax_available():
        return "simd"
    return None


@lru_cache(maxsize=None)
def _hc_expand1_kernel(product: str):
    simd = product == "simd"
    return mx.fast.metal_kernel(
        name="glm5_hc_expand_one_token" + ("_simd" if simd else ""),
        input_names=["x", "residual", "post", "comb"],
        output_names=["out"],
        header=_SIMD_COMB_HEADER if simd else _NAX_HEADER,
        source=_HC_EXPAND1_SIMD_SOURCE if simd else _HC_EXPAND1_SOURCE,
    )


def hc_expand_one(
    x: mx.array, residual: mx.array, post: mx.array, comb: mx.array
) -> Optional[mx.array]:
    """``_hc_expand_op(x, residual, post, comb)`` for a single token.

    ``x`` [1, 1, D], ``residual`` [1, 1, HC, D] (bf16/fp16), ``post``
    [1, 1, HC] and ``comb`` [1, 1, HC, HC] fp32. Returns [1, 1, HC, D] or
    None when not covered.
    """
    if x.ndim != 3 or x.shape[:2] != (1, 1) or residual.ndim != 4:
        return None
    D = x.shape[2]
    hc = residual.shape[2]
    if residual.shape != (1, 1, hc, D) or not 1 <= hc <= 16 or D % 32:
        return None
    if x.dtype not in (mx.bfloat16, mx.float16) or residual.dtype != x.dtype:
        return None
    if post.shape != (1, 1, hc) or comb.shape != (1, 1, hc, hc):
        return None
    if post.dtype != mx.float32 or comb.dtype != mx.float32:
        return None
    product = hc_comb_product()
    if product is None:
        return None
    simds = 8
    tiles = D // 32
    STATS["hc_expand"] += 1
    return _hc_expand1_kernel(product)(
        inputs=[x, residual, post, comb],
        template=[("T", x.dtype), ("HC", hc), ("D", D), ("SIMDS", simds)],
        grid=(32 * simds * ((tiles + simds - 1) // simds), 1, 1),
        threadgroup=(32 * simds, 1, 1),
        output_shapes=[residual.shape],
        output_dtypes=[x.dtype],
    )[0]


# ---------------------------------------------------------------------------
# One-token HC pre in one dispatch, with the previous expand folded in
# ---------------------------------------------------------------------------
#
# A half-layer's HC chain is hc_expand (previous branch) -> hc_mix ->
# exact_hc_norm -> branch: three dependent dispatches, all latency bound.
# The branch input only needs mix rows 0..HC-1 (the pre weights), so
# ``hc_pre_fused`` computes the mix rows HC at a time per 1024-thread
# threadgroup (hc_mix's multi-row layout) and threadgroup 0, which owns the
# pre rows, finishes exact_hc_norm's collapse and RMSNorm itself. The post
# and comb rows only feed the next expand: ``hc_post_mm`` (sinkhorn plus the
# comb product of hc_expand_one) runs beside the branch, and the next
# ``hc_pre_fused`` applies hc_expand_one's epilogue to the branch output as it
# loads h (threadgroup 0 also stores h, the next residual). Every value is
# computed with the reference kernels' arithmetic and order.
_HC_PRE_HEADER = r"""
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

// hc_expand_one's epilogue for element e = r * D + col of h. The product is
// its own statement, so it is rounded before the add (the reference runs
// them as two kernels): MLX compiles kernels in MathModeSafe, which never
// contracts across statements. No volatile: it serialized the many h loads
// of the mix GEMV loop below.
template <typename T, int D, typename YPtr, typename MPtr, typename PPtr>
inline T glm_hc_expand_value(YPtr y, MPtr mm, PPtr post, int e) {
  const int r = e / D;
  const int col = e - r * D;
  float prod = post[r] * static_cast<float>(y[col]);
  float sum = prod + mm[e];
  return static_cast<T>(sum);
}
"""

_HC_PRE_SOURCE = r"""
  const uint lid = thread_position_in_threadgroup.x;
  const uint simd_lid = thread_index_in_simdgroup;
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const int tile = int(threadgroup_position_in_grid.x);
  constexpr int KSZ = HC * D;
  constexpr int CH = KSZ / 4096;  // rms_looped reads per thread, 4 values each
  constexpr int D4 = D / 4;
  static_assert(KSZ % 4096 == 0 && D4 <= 1024 && D % 4 == 0, "hc_pre_fused shape");
  constexpr float HC_EPS = HC_EPS_INT * 1e-9;
  constexpr float NORM_EPS = NORM_EPS_INT * 1e-9;
  using T4 = vec<T, 4>;
#if DEFERRED
#define HVAL(e) glm_hc_expand_value<T, D>(y, mm, post, (e))
#else
#define HVAL(e) x[(e)]
#endif

  // This thread's h values in rms_looped order.
  T hv[CH][4];
  for (int c = 0; c < CH; c++) {
    for (int i = 0; i < 4; i++) {
      hv[c][i] = HVAL(c * 4096 + int(lid) * 4 + i);
    }
  }
#if DEFERRED
  if (tile == 0) {
    for (int c = 0; c < CH; c++) {
      for (int i = 0; i < 4; i++) {
        h_out[c * 4096 + int(lid) * 4 + i] = hv[c][i];
      }
    }
  }
#endif

  // --- hc_mix: rms_looped (lsize = 1024, N_READS = 4) ---
  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[32];
  float acc = 0;
  for (int c = 0; c < CH; c++) {
    for (int i = 0; i < 4; i++) {
      float xi = static_cast<float>(hv[c][i]);
      acc += xi * xi;
    }
  }
  acc = simd_sum(acc);
  if (simd_gid == 0) {
    local_sums[simd_lid] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lid == 0) {
    local_sums[simd_gid] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    acc = simd_sum(local_sums[simd_lid]);
    if (simd_lid == 0) {
      local_inv_mean[0] = metal::precise::rsqrt(acc / KSZ + eps[0]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = local_inv_mean[0];

  // --- hc_mix gemv: 8 simdgroups per row, 4 rows per threadgroup ---
  const int slot = int(simd_gid) / 8;
  const int sgN = int(simd_gid) % 8;
  const int row = tile * 4 + slot;
  threadgroup float partial[4][8];
  threadgroup float row_total[4];
  float result = 0;
  {
    const device auto* mrow = fn + size_t(row) * KSZ;
    int bn = (32 * sgN + int(simd_lid)) * 4;
    for (int i = 0; i < KSZ / 1024; ++i) {
      float v_coeff[4];
      float inter[4];
      for (int tn = 0; tn < 4; tn++) {
        v_coeff[tn] = static_cast<float>(HVAL(bn + tn)) * inv;
      }
      for (int tn = 0; tn < 4; tn++) {
        inter[tn] = static_cast<float>(mrow[bn + tn]);
      }
      for (int tn = 0; tn < 4; tn++) {
        result += inter[tn] * v_coeff[tn];
      }
      bn += 1024;
    }
    for (ushort sn = 16; sn >= 1; sn >>= 1) {
      result += simd_shuffle_down(result, sn);
    }
  }
  if (simd_lid == 0) {
    partial[slot][sgN] = result;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgN == 0 && simd_lid == 0) {
    float total = partial[slot][0];
    for (int s = 1; s < 8; s++) {
      total += partial[slot][s];
    }
    mixes[row] = total;
    row_total[slot] = total;
  }
  if (tile != 0) {
    return;
  }

  // --- threadgroup 0 (the pre rows): exact_hc_norm's collapse + RMSNorm ---
  threadgroup float pre_shared[HC];
  threadgroup float norm_inv[1];
  threadgroup float norm_sums[32];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    const float pre_scale = scale[0];
    const uint llane = metal::min(simd_lid, (uint)(HC - 1));
    float pre_z = row_total[llane] * pre_scale + base[llane];
    float pre_v = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + HC_EPS;
    if (simd_lid < (uint)HC) {
      pre_shared[simd_lid] = pre_v;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  T4 rounded = T4(0);
  float accum = 0.0f;
  if (int(lid) < D4) {
    T4 xs[HC];
    for (int r = 0; r < HC; r++) {
      if (D == 4096) {
        // Same elements as this thread's rms_looped reads.
        xs[r] = T4(hv[r][0], hv[r][1], hv[r][2], hv[r][3]);
      } else {
        const int e = r * D + int(lid) * 4;
        xs[r] = T4(HVAL(e), HVAL(e + 1), HVAL(e + 2), HVAL(e + 3));
      }
    }
    float4 collapsed = fma(
        float4(pre_shared[0]), float4(xs[0]),
        fma(
            float4(pre_shared[1]), float4(xs[1]),
            fma(
                float4(pre_shared[2]), float4(xs[2]),
                float4(pre_shared[3]) * float4(xs[3]))));
    rounded = T4(collapsed);
    float4 rounded_float = float4(rounded);
    accum += rounded_float.x * rounded_float.x;
    accum += rounded_float.y * rounded_float.y;
    accum += rounded_float.z * rounded_float.z;
    accum += rounded_float.w * rounded_float.w;
  }
  accum = simd_sum(accum);
  if (simd_lid == 0) {
    norm_sums[simd_gid] = accum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    accum = simd_sum(norm_sums[simd_lid]);
    if (simd_lid == 0) {
      norm_inv[0] = metal::precise::rsqrt(accum / D + NORM_EPS);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(lid) < D4) {
    float ninv = norm_inv[0];
    const device T4* weights = (const device T4*)norm_weight;
    T4 scaled = T4(float4(rounded) * ninv);
    T4 weight = weights[lid];
    ((device T4*)normalized)[lid] = T4(
        weight.x * scaled.x,
        weight.y * scaled.y,
        weight.z * scaled.z,
        weight.w * scaled.w);
  }
#undef HVAL
"""

# exact_hc_norm's post/sinkhorn code, defined before the MPP include (so it
# compiles exactly as in that kernel), then hc_expand_one's comb product.
_HC_POST_COMB_HEADER = r"""
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

template <int HC, int ITERS, int HC_EPS_INT, typename MixPtr, typename BasePtr>
inline void glm_hc_post_comb(
    MixPtr mix, const float post_scale, const float comb_scale, BasePtr base,
    uint lane, thread float& post_v, thread float4& result) {
  constexpr int BASE_OFF = 2 * HC;
  constexpr float HC_EPS = HC_EPS_INT * 1e-9;
  const float active = lane < (uint)HC ? 1.0f : 0.0f;
  const uint llane = metal::min(lane, (uint)(HC - 1));

  float post_z = mix[HC + llane] * post_scale + base[HC + llane];
  post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));

  float4 value =
      (float4(mix[BASE_OFF + llane * HC], mix[BASE_OFF + llane * HC + 1],
              mix[BASE_OFF + llane * HC + 2], mix[BASE_OFF + llane * HC + 3]) * comb_scale +
       float4(base[BASE_OFF + llane * HC], base[BASE_OFF + llane * HC + 1],
              base[BASE_OFF + llane * HC + 2], base[BASE_OFF + llane * HC + 3])) * active;
  float row_max = metal::max(
      metal::max(value.x, value.y), metal::max(value.z, value.w));
  float4 exponent = metal::fast::exp(value - row_max) * active;
  result = exponent *
          (1.0f /
           (exponent.x + exponent.y + exponent.z + exponent.w + HC_EPS)) +
      HC_EPS * active;
  float4 column_inv = 1.0f /
      (float4(
           simd_sum(result.x), simd_sum(result.y),
           simd_sum(result.z), simd_sum(result.w)) +
       HC_EPS);
  result *= column_inv;
  for (int iter = 1; iter < ITERS; ++iter) {
    result *=
        (1.0f / (result.x + result.y + result.z + result.w + HC_EPS)) *
        active;
    column_inv = 1.0f /
        (float4(
             simd_sum(result.x), simd_sum(result.y),
             simd_sum(result.z), simd_sum(result.w)) +
         HC_EPS);
    result *= column_inv;
  }
}

"""

_HC_POST_HEADER = _HC_POST_COMB_HEADER + r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""

_HC_POST_PROLOGUE = r"""
  const ushort lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  static_assert(HC == 4, "one float4 comb row per lane");
  threadgroup float4 comb_rows[HC];
  if (sg == 0) {
    float post_v;
    float4 res;
    glm_hc_post_comb<HC, ITERS, HC_EPS_INT>(mixes, scale[1], scale[2], base, uint(lane), post_v, res);
    if (lane < HC) {
      comb_rows[lane] = res;
      if (threadgroup_position_in_grid.x == 0) {
        post_out[lane] = post_v;
        *(device float4*)(comb_out + lane * HC) = res;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int tile = int(threadgroup_position_in_grid.x) * SIMDS + int(sg);
  if (tile * 32 >= D) {
    return;
  }
  const threadgroup float* comb = (const threadgroup float*)comb_rows;
"""

_HC_POST_SOURCE = _HC_POST_PROLOGUE + r"""
  const short qid = lane >> 2;
  const short fm = ((qid & 4) | ((lane >> 1) & 3));
  const short fn = ((qid & 2) | (lane & 1)) * 4;
  constexpr auto desc = matmul2d_descriptor(
      16, 32, 16, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto ct_a = op.template get_left_input_cooperative_tensor<float, float, float>();
  auto ct_b = op.template get_right_input_cooperative_tensor<float, float, float>();
  auto ct_c = op.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(ct_a)>,
      metal::remove_addrspace_t<decltype(ct_b)>,
      float>();
  for (short i = 0; i < 8; i++) {
    const short r = fm + (i >> 2) * 8;
    const short c = fn + (i & 3);
    // A = comb^T (rows: output stream, cols: source stream), zero padded.
    ct_a[i] = (r < HC && c < HC) ? comb[c * HC + r] : 0.0f;
    ct_b[i] = (r < HC) ? static_cast<float>(residual[r * D + tile * 32 + c]) : 0.0f;
    ct_b[8 + i] = (r < HC) ? static_cast<float>(residual[r * D + tile * 32 + 16 + c]) : 0.0f;
    ct_c[i] = 0.0f;
    ct_c[8 + i] = 0.0f;
  }
  op.run(ct_a, ct_b, ct_c);
  for (short i = 0; i < 8; i++) {
    const short r = fm + (i >> 2) * 8;
    const short c = fn + (i & 3);
    if (r < HC) {
      for (short hh = 0; hh < 2; hh++) {
        const int col = tile * 32 + hh * 16 + c;
        mm[r * D + col] = ct_c[hh * 8 + i];
      }
    }
  }
"""

_HC_POST_SIMD_SOURCE = _HC_POST_PROLOGUE + r"""
  short fm, fn;
  float vals[4][2];
  glm_hc_comb_simd_mma<HC, D>(comb, residual, tile, lane, fm, fn, vals);
  if (fm < HC) {
    for (short t = 0; t < 4; t++) {
      for (short j = 0; j < 2; j++) {
        mm[fm * D + tile * 32 + t * 8 + fn + j] = vals[t][j];
      }
    }
  }
"""


@lru_cache(maxsize=None)
def _hc_pre_fused_kernel(deferred: bool):
    inputs = ["y", "mm", "post"] if deferred else ["x"]
    inputs += ["fn", "eps", "scale", "base", "norm_weight"]
    outputs = ["normalized", "mixes"] + (["h_out"] if deferred else [])
    return mx.fast.metal_kernel(
        name="glm5_hc_pre_fused" + ("_deferred" if deferred else ""),
        input_names=inputs,
        output_names=outputs,
        header=_HC_PRE_HEADER,
        source=_source(_HC_PRE_SOURCE, DEFERRED=int(deferred)),
    )


@lru_cache(maxsize=None)
def _hc_post_mm_kernel(product: str):
    simd = product == "simd"
    return mx.fast.metal_kernel(
        name="glm5_hc_post_comb_mm" + ("_simd" if simd else ""),
        input_names=["mixes", "scale", "base", "residual"],
        output_names=["post_out", "comb_out", "mm"],
        header=(_HC_POST_COMB_HEADER + _SIMD_COMB_HEADER) if simd else _HC_POST_HEADER,
        source=_HC_POST_SIMD_SOURCE if simd else _HC_POST_SOURCE,
    )


def hc_defer_supported(connection, norm, dtype, width: int) -> bool:
    """Whether ``hc_pre_fused`` / ``hc_post_mm`` cover this connection (one
    token, hidden ``width``, activations of ``dtype``): the shapes of the
    replicated hc_mix / exact_hc_norm / hc_expand_one configurations."""
    if dtype not in (mx.bfloat16, mx.float16) or connection.hc_mult != 4:
        return False
    fn = connection.fn
    mix = (2 + 4) * 4
    if fn.dtype != mx.float32 or fn.shape != (mix, 4 * width):
        return False
    if width % 1024 or width > 4096 or norm.weight.shape != (width,) or norm.weight.dtype != dtype:
        return False
    if connection.scale.shape != (3,) or connection.base.shape != (mix,):
        return False
    if connection.scale.dtype != mx.float32 or connection.base.dtype != mx.float32:
        return False
    return hc_comb_product() is not None


def hc_pre_fused(connection, norm, x=None, deferred=None):
    """One-token ``hc_mix`` + ``exact_hc_norm`` (the normalized branch input
    only) in one dispatch.

    ``x`` [1, 1, HC, D] is the layer input, or ``deferred = (y, mm, post)``
    the previous half-layer's branch output [1, 1, D], ``hc_post_mm``'s comb
    product [HC * D] fp32 and post weights [1, 1, HC] fp32, from which h =
    ``hc_expand_one(y, residual, post, comb)`` is recomputed exactly. Returns
    ``(normalized [1, 1, D], mixes [1, 1, 24] fp32, h [1, 1, HC, D] or None
    for ``x``)``; the caller checks ``hc_defer_supported`` first.
    """
    if deferred is not None:
        y, mm, post = deferred
        dtype, D = y.dtype, y.shape[-1]
        if y.shape != (1, 1, D) or mm.shape != (4 * D,) or post.shape != (1, 1, 4):
            return None
        inputs = [y, mm, post]
    else:
        dtype, D = x.dtype, x.shape[-1]
        if x.shape != (1, 1, 4, D):
            return None
        inputs = [x]
    mix = connection.fn.shape[0]
    inputs += [
        mix_weights(connection), mx.array([connection.norm_eps], dtype=mx.float32),
        connection.scale, connection.base, norm.weight,
    ]
    STATS["hc_pre_fused"] += 1
    outs = _hc_pre_fused_kernel(deferred is not None)(
        inputs=inputs,
        template=[
            ("T", dtype), ("HC", 4), ("D", D),
            ("HC_EPS_INT", round(connection.hc_eps / 1e-9)),
            ("NORM_EPS_INT", round(norm.eps / 1e-9)),
        ],
        grid=(1024 * (mix // 4), 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(1, 1, D), (1, 1, mix)] + ([(1, 1, 4, D)] if deferred is not None else []),
        output_dtypes=[dtype, mx.float32] + ([dtype] if deferred is not None else []),
    )
    return outs[0], outs[1], (outs[2] if deferred is not None else None)


def hc_post_mm(connection, mixes: mx.array, residual: mx.array):
    """The post weights, sinkhorn comb [1, 1, HC, HC] (fp32, as exact_hc_norm)
    and hc_expand_one's comb product ``mm`` [HC * D] fp32 of ``residual``
    [1, 1, HC, D] for one token."""
    D = residual.shape[-1]
    simds = 8
    tiles = D // 32
    STATS["hc_post_mm"] += 1
    return tuple(
        _hc_post_mm_kernel(hc_comb_product())(
            inputs=[mixes, connection.scale, connection.base, residual],
            template=[
                ("T", residual.dtype), ("HC", 4), ("D", D),
                ("ITERS", int(connection.sinkhorn_iters)),
                ("HC_EPS_INT", round(connection.hc_eps / 1e-9)), ("SIMDS", simds),
            ],
            grid=(32 * simds * ((tiles + simds - 1) // simds), 1, 1),
            threadgroup=(32 * simds, 1, 1),
            output_shapes=[(1, 1, 4), (1, 1, 4, 4), (4 * D,)],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
        )
    )


# ---------------------------------------------------------------------------
# MLA per-head projections (embed_q / unembed_out) for one token
# ---------------------------------------------------------------------------
#
# ``QuantizedMultiLinear`` runs a batched qmv: one 64-thread threadgroup per
# 8 rows of one head (4096 tiny threadgroups for 64 heads x 512 rows), well
# below the weight-streaming rate. The same per-row arithmetic (qmv_fast's
# lane mapping, or qmv's single tail block when K is one qmv block) with NSG
# simdgroups x 4 rows per threadgroup.
_MH_QMV_SOURCE = r"""
  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int h = int(threadgroup_position_in_grid.z);
  const int r0 = (int(threadgroup_position_in_grid.y) * NSG + sg) * 4;
  constexpr int WB = K * glm_bytes_per_pack<BITS>() / glm_pack_factor<BITS>();
  constexpr int G = K / GS;
  const device uint8_t* wh = (const device uint8_t*)w + (size_t(h) * N + r0) * WB;
  const device T* sh = scales + (size_t(h) * N + r0) * G;
  const device T* bh = biases + (size_t(h) * N + r0) * G;
  const device T* xh = x + size_t(h) * K;
  float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#if FAST
  glm_qmv_rows<T, K, GS, BITS, 4>(wh, sh, bh, xh, lane, result);
#else
  {
    // qmv_impl when K is exactly one block: every lane's values arrive in
    // the tail (load_vector_safe / qdot_safe with a full remainder).
    constexpr int VPT = glm_pack_factor<BITS>();
    static_assert(K == VPT * 32, "one qmv block");
    float x_thread[VPT];
    float sum = glm_load_vector<T, VPT, BITS>(xh + lane * VPT, x_thread);
    for (int row = 0; row < 4; row++) {
      const device uint8_t* wl = wh + row * WB + lane * glm_bytes_per_pack<BITS>();
      const float sc = sh[row * G + int(lane) / (GS / VPT)];
      const float bi = bh[row * G + int(lane) / (GS / VPT)];
      result[row] += glm_qdot<VPT, BITS>(wl, x_thread, sc, bi, sum);
    }
  }
#endif
  for (int r = 0; r < 4; r++) {
    float v = simd_sum(result[r]);
    if (lane == 0) {
      y[size_t(h) * N + r0 + r] = static_cast<T>(v);
    }
  }
"""


@lru_cache(maxsize=None)
def _mh_qmv_kernel(fast: bool):
    return mx.fast.metal_kernel(
        name="glm5_mla_head_qmv" + ("_fast" if fast else "_block"),
        input_names=["x", "w", "scales", "biases"],
        output_names=["y"],
        header=_QMV_HEADER,
        source=_source(_MH_QMV_SOURCE, FAST=int(fast)),
    )


def mla_head_qmv(x: mx.array, layer, nsg: int = 8) -> Optional[mx.array]:
    """``layer(x)`` (a ``QuantizedMultiLinear``, transpose=True) for one token:
    ``x`` [1, H, 1, K] -> [1, H, 1, N], or None when not covered."""
    if getattr(layer, "mode", "affine") != "affine" or layer.get("biases") is None:
        return None
    w, s, b = layer["weight"], layer["scales"], layer["biases"]
    bits, gs = layer.bits, layer.group_size
    if x.ndim != 4 or x.shape[0] != 1 or x.shape[2] != 1 or w.ndim != 3:
        return None
    H, K = x.shape[1], x.shape[3]
    N = w.shape[1]
    if w.shape[0] != H or s.shape != (H, N, K // gs) or b.shape != s.shape:
        return None
    if bits not in (4, 5, 6, 8) or K % gs or gs not in (32, 64, 128):
        return None
    if K in (64, 128) and bits in (4, 8):
        return None  # MLX routes these to qmv_quad
    if x.dtype not in (mx.bfloat16, mx.float16) or s.dtype != x.dtype or b.dtype != x.dtype:
        return None
    pack = {5: 8, 6: 4}.get(bits, 32 // bits)
    fast = N % 8 == 0 and K % (pack * 2 * 32) == 0
    if not fast and not (K == pack * 32 and N >= 8 and gs % pack == 0):
        return None
    if N % (4 * nsg):
        nsg = 2
        if N % 8:
            return None
    STATS["mla_head_qmv"] += 1
    return _mh_qmv_kernel(fast)(
        inputs=[x.reshape(H, K), w, s, b],
        template=[("T", x.dtype), ("K", K), ("N", N), ("BITS", bits), ("GS", gs), ("NSG", nsg)],
        grid=(32, (N // (4 * nsg)) * nsg, H),
        threadgroup=(32, nsg, 1),
        output_shapes=[(1, H, 1, N)],
        output_dtypes=[x.dtype],
    )[0]


# ---------------------------------------------------------------------------
# Several quantized projections of one input in a single dispatch
# ---------------------------------------------------------------------------
#
# One-token rows reproduce MLX's qmv_fast (2 simdgroups x 4 rows per 8-row
# tile, qdot per 512/256-wide K block, simd_sum); 2..8-token rows reproduce
# qmv_wide (8 lanes per row, per-group dequantize in 8-value sub-chunks, 4/2/1
# ladder) -- the kernels the separate projections run on. Each part keeps
# its own contiguous output, so nothing downstream changes.
_MULTI_QMV_SOURCE = r"""
  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int row0 = int(threadgroup_position_in_grid.x) * 8;
  constexpr int WB = K * BITS / 8;
  constexpr int G = K / GS;
  const device uint8_t* w;
  const device T* sc;
  const device T* bi;
  device T* y;
  int local;
  int n_rows;
  if (row0 < N0) {
    w = (const device uint8_t*)w0; sc = s0; bi = b0; y = y0; local = row0; n_rows = N0;
  }
#if NP > 1
  else if (row0 < N0 + N1) {
    w = (const device uint8_t*)w1; sc = s1; bi = b1; y = y1; local = row0 - N0; n_rows = N1;
  }
#endif
#if NP > 2
  else if (row0 < N0 + N1 + N2) {
    w = (const device uint8_t*)w2; sc = s2; bi = b2; y = y2; local = row0 - N0 - N1; n_rows = N2;
  }
#endif
#if NP > 3
  else {
    w = (const device uint8_t*)w3; sc = s3; bi = b3; y = y3; local = row0 - N0 - N1 - N2; n_rows = N3;
  }
#endif
#if TOK == 1
  const int r0 = local + sg * 4;
  float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  glm_qmv_rows<T, K, GS, BITS, 4>(
      w + size_t(r0) * WB, sc + r0 * G, bi + r0 * G, x, lane, result);
  for (int r = 0; r < 4; r++) {
    float v = simd_sum(result[r]);
    if (lane == 0) {
      y[r0 + r] = static_cast<T>(v);
    }
  }
#else
  const int k_lane = int(lane) % 8;
  const int row = local + sg * 4 + int(lane) / 8;
  float res[TOK];
  for (int v = 0; v < TOK; v++) {
    res[v] = 0.0f;
  }
  glm_qmv_wide_row<T, K, GS, BITS, TOK>(
      w + size_t(row) * WB, sc + row * G, bi + row * G, x, TOK, k_lane, res);
  for (int v = 0; v < TOK; v++) {
    res[v] += simd_shuffle_down(res[v], 4);
    res[v] += simd_shuffle_down(res[v], 2);
    res[v] += simd_shuffle_down(res[v], 1);
  }
  if (k_lane == 0) {
    for (int v = 0; v < TOK; v++) {
      y[size_t(v) * n_rows + row] = static_cast<T>(res[v]);
    }
  }
#endif
"""


@lru_cache(maxsize=None)
def _multi_qmv_kernel(n_parts: int, tokens: int):
    inputs = ["x"]
    for i in range(n_parts):
        inputs += [f"w{i}", f"s{i}", f"b{i}"]
    return mx.fast.metal_kernel(
        name=f"glm5_multi_qmv_p{n_parts}_t{tokens}",
        input_names=inputs,
        output_names=[f"y{i}" for i in range(n_parts)],
        header=_QMV_HEADER,
        source=_source(_MULTI_QMV_SOURCE, NP=n_parts, TOK=tokens),
    )


def multi_qmv(x: mx.array, layers) -> Optional[list]:
    """``[linear(x) for linear in layers]`` for 1..8 rows in one dispatch.

    ``x`` [T, K]; ``layers`` 1..4 affine quantized linears (no bias) with the
    same bits/group size, K inputs and output rows divisible by 8. Returns
    the [T, N_i] outputs or None when not covered.
    """
    if x.ndim != 2 or not 1 <= len(layers) <= 4 or x.dtype not in (mx.bfloat16, mx.float16):
        return None
    T, K = x.shape
    if not 1 <= T <= 8:
        return None
    parts = [_affine_parts(m) for m in layers]
    if any(p is None for p in parts):
        return None
    bits, gs = parts[0][3], parts[0][4]
    if any((p[3], p[4]) != (bits, gs) for p in parts):
        return None
    rows = []
    for w, s, b, _, _ in parts:
        if w.ndim != 2 or w.shape[1] * 32 // bits != K or s.dtype != x.dtype:
            return None
        rows.append(w.shape[0])
    if any(n % 8 for n in rows):
        return None
    if T == 1:
        # qmv_fast only (MLX routes the one-row product there when aligned).
        if not all(_qmv_fast_ok(bits, gs, n, K) for n in rows):
            return None
    elif bits not in (4, 5, 6, 8) or gs % 8 or K % gs or K in (64, 128):
        return None
    inputs = [x]
    for w, s, b, _, _ in parts:
        inputs += [w, s, b]
    template = [("T", x.dtype), ("K", K), ("BITS", bits), ("GS", gs)]
    template += [(f"N{i}", rows[i] if i < len(rows) else 0) for i in range(4)]
    STATS["multi_qmv"] += 1
    return list(_multi_qmv_kernel(len(layers), T)(
        inputs=inputs,
        template=template,
        grid=(64 * (sum(rows) // 8), 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(T, n) for n in rows],
        output_dtypes=[x.dtype] * len(rows),
    ))


# ---------------------------------------------------------------------------
# Row-exact verify GEMV: rows of x @ W.T with the one-row gemv arithmetic
# ---------------------------------------------------------------------------
#
# A row-exact MTP verify gives every row of a block the arithmetic of its
# one-token decode step. For a float weight that step is MLX's
# non-transposed gemv (gemv_axbpy with M == 1; SM = 1, SN = 32, TM = 4,
# TN = 4): BM simdgroups of 4 output rows each (BN = 1; BM = 8 when
# N >= 4096, else 4) or, when K >= 16 * N, one 4-row tile split along K
# over BN = 8 simdgroups whose partial sums simdgroup 0 adds in order
# through threadgroup memory. Lane l of K slice s accumulates inputs
# (32 s + l) * 4 .. + 3 of every BN * 128-wide block (the last block
# zero-filled past K), then the shuffle-down ladder; the tail tile shifts
# inwards to N - 4. A multi-row matmul instead runs gemv_wide or a GEMM
# tile, which sum in other orders. Here every row replays the one-row
# kernel, all rows sharing each weight load.
_ROWS_GEMV_SOURCE = r"""
  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int sgN = sg % BN;
  constexpr int BLOCK_N = BN * 128;
  constexpr int N_ITER = K / BLOCK_N;
  constexpr int LEFTOVER = K - N_ITER * BLOCK_N;
  threadgroup float partials[BN > 1 ? BM * BN * ROWS * 4 : 1];
  int out_row = int(threadgroup_position_in_grid.x) * (BM * 4) + (sg / BN) * 4;
  if (out_row >= N) {
    return;
  }
  out_row = out_row + 4 <= N ? out_row : N - 4;
  const device T* mat = w + size_t(out_row) * K;
  float result[ROWS][4];
  for (int r = 0; r < ROWS; r++) {
    for (int tm = 0; tm < 4; tm++) {
      result[r][tm] = 0.0f;
    }
  }
  int bn = (32 * sgN + int(lane)) * 4;
  for (int i = 0; i < N_ITER; ++i) {
    T inter[4][4];
    for (int tm = 0; tm < 4; tm++) {
      for (int tn = 0; tn < 4; tn++) {
        inter[tm][tn] = mat[tm * K + bn + tn];
      }
    }
    for (int r = 0; r < ROWS; r++) {
      float v_coeff[4];
      for (int tn = 0; tn < 4; tn++) {
        v_coeff[tn] = static_cast<float>(x[r * K + bn + tn]);
      }
      for (int tm = 0; tm < 4; tm++) {
        for (int tn = 0; tn < 4; tn++) {
          result[r][tm] += inter[tm][tn] * v_coeff[tn];
        }
      }
    }
    bn += BLOCK_N;
  }
  if (LEFTOVER > 0) {
    T inter[4][4];
    for (int tm = 0; tm < 4; tm++) {
      for (int tn = 0; tn < 4; tn++) {
        inter[tm][tn] = bn + tn < K ? mat[tm * K + bn + tn] : T(0);
      }
    }
    for (int r = 0; r < ROWS; r++) {
      float v_coeff[4];
      for (int tn = 0; tn < 4; tn++) {
        v_coeff[tn] = bn + tn < K ? static_cast<float>(x[r * K + bn + tn]) : 0.0f;
      }
      for (int tm = 0; tm < 4; tm++) {
        for (int tn = 0; tn < 4; tn++) {
          result[r][tm] += inter[tm][tn] * v_coeff[tn];
        }
      }
    }
  }
  for (int r = 0; r < ROWS; r++) {
    for (int tm = 0; tm < 4; tm++) {
      for (ushort sn = 16; sn >= 1; sn >>= 1) {
        result[r][tm] += simd_shuffle_down(result[r][tm], sn);
      }
    }
  }
  if (BN > 1) {
    threadgroup float* tile = partials + (sg / BN) * BN * ROWS * 4;
    if (lane == 0) {
      for (int r = 0; r < ROWS; r++) {
        for (int tm = 0; tm < 4; tm++) {
          tile[(sgN * ROWS + r) * 4 + tm] = result[r][tm];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgN == 0 && lane == 0) {
      for (int s = 1; s < BN; s++) {
        for (int r = 0; r < ROWS; r++) {
          for (int tm = 0; tm < 4; tm++) {
            result[r][tm] += tile[(s * ROWS + r) * 4 + tm];
          }
        }
      }
    }
  }
  if (sgN == 0 && lane == 0) {
    for (int r = 0; r < ROWS; r++) {
      for (int tm = 0; tm < 4; tm++) {
        y[size_t(r) * N + out_row + tm] = static_cast<T>(result[r][tm]);
      }
    }
  }
"""


@lru_cache(maxsize=None)
def _rows_gemv_kernel():
    return mx.fast.metal_kernel(
        name="glm5_rows_gemv",
        input_names=["x", "w"],
        output_names=["y"],
        source=_ROWS_GEMV_SOURCE,
    )


def rows_gemv(x: mx.array, weight: mx.array) -> Optional[mx.array]:
    """``x @ weight.T`` with every row as the one-row gemv computes it.

    ``x`` [..., K] (2..16 rows), ``weight`` [N, K] of x's dtype (bf16/fp16).
    Returns [..., N] or None when not covered (one row, or the gemv
    configurations for K <= 64 or N < 4).
    """
    if (
        weight.ndim != 2
        or x.dtype != weight.dtype
        or x.dtype not in (mx.bfloat16, mx.float16)
    ):
        return None
    N, K = weight.shape
    if x.ndim < 2 or x.shape[-1] != K or K <= 64 or N < 4:
        return None
    rows = x.size // K
    if not 2 <= rows <= 16:
        return None
    bm, bn = (1, 8) if K >= 16 * N else ((8 if N >= 4096 else 4), 1)
    STATS["rows_gemv"] += 1
    y = _rows_gemv_kernel()(
        inputs=[mx.contiguous(x).reshape(rows, K), weight],
        template=[
            ("T", x.dtype),
            ("K", K),
            ("N", N),
            ("ROWS", rows),
            ("BM", bm),
            ("BN", bn),
        ],
        grid=(32 * ((N + 4 * bm - 1) // (4 * bm)), bm * bn, 1),
        threadgroup=(32, bm * bn, 1),
        output_shapes=[(rows, N)],
        output_dtypes=[x.dtype],
    )[0]
    return y.reshape(*x.shape[:-1], N)
