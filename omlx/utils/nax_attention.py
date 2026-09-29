# SPDX-License-Identifier: Apache-2.0
#
# The attention kernel below is MLX's NAX flash-attention kernel
# (mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h,
# Copyright (c) 2024-25 Apple Inc., MIT License,
# https://github.com/ml-explore/mlx) with a separate value head dim.
"""Fused tensor-unit (NAX) prefill attention for a narrower value head.

MLA-style models such as MiMo-V2-Flash use query/key head dim 192 with value
head dim 128. MLX 0.32.2 has no fused kernel for that pair: its prefill SDPA
falls back to the unfused path (bf16 score matrix materialised in memory),
and on M5 (NAX) GPUs its tensor-unit attention kernel only exists for head
dims 64/96/128/256. ``mixed_head_dim_sdpa`` therefore zero-pads Q/K/V to 256
to reach the head-dim-split NAX kernel, which spends a third of its
multiply-adds on zero columns and copies Q/K/V every layer.

This module JIT-compiles (``mx.fast.metal_kernel``) MLX's own NAX attention
kernel with the value head dim as a separate template parameter (BD = 192,
BDV = 128): the query/key loop runs over 192 dims and the output tile and
the P @ V loop over 128, so no padding and no copies are needed. The kernel
body is MLX's (same tiles, same online softmax in fp32, same MPP matmul
calls); the MLX-side function constants (alignment, mask, causal, sinks) are
compile-time constants of one generated kernel per variant. On MLX builds
that carry the native kernel, the output is bit-identical to it.

Long contexts run as several dispatches over consecutive key ranges
(``OMLX_NAX_ATTN_PASS_KEYS`` keys each, default 8192) that hand the fp32
online-softmax row state (O accumulator, running max and sum) from one to
the next through device memory, so each row computes exactly what one
dispatch computes (bit-identical output). One dispatch over a long key range
lets its threadgroups drift apart until each streams its KV head from DRAM;
per-slice dispatches keep the K/V they share in the on-chip caches. Those
dispatches also split the head dims over simdgroup pairs (MLX's dsplit
scheme, ``OMLX_NAX_ATTN_DSPLIT``), which only reorders the fp32 sums of
Q @ K.T (last-bit differences in ~0.5% of the outputs).

Inputs are read through their strides (no contiguity copies: KV-cache
slices, transposed projections and the strided key windows of the blocked
sliding-window path are consumed in place); only the head dim must be
contiguous, as for MLX's SDPA. The output is written in MLX's SDPA layout
(``[B, L, H, V]`` rows returned as a ``[B, H, L, V]`` view), so the caller's
``swapaxes(1, 2).reshape(B, L, -1)`` stays free.

Fail-closed: only (192, 128) bf16/fp16 prefill on NAX GPUs is handled, the
first use runs a small self-check against an fp32 reference, and any
unsupported input returns None so the caller keeps its existing route.
Kill switch: ``OMLX_NAX_JIT_ATTENTION=0``.
"""

from __future__ import annotations

import logging
import os
import struct
from functools import lru_cache
from typing import Optional

import mlx.core as mx

from omlx.custom_kernels.nax import is_nax_available
from omlx.custom_kernels.nax_tiles import NAX_TILE_HEADER

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get("OMLX_NAX_JIT_ATTENTION", "1").strip().lower() not in {
    "0",
    "false",
    "off",
}

# (query/key head dim, value head dim) pairs this kernel is validated for.
SUPPORTED_HEAD_DIMS = frozenset({(192, 128)})

_BQ = 64
_BK = 32
_WM = 4
_THREADS = 32 * _WM


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# Key-range passes: keys per dispatch (0 = one dispatch). Every threadgroup
# of a dispatch streams this many keys of its KV head, so the key slice the
# threadgroups in flight share stays in the on-chip caches (one dispatch over
# 1M keys runs at ~85 TFLOPS against ~103 at 64k). Grids of a wave or two
# (query tails under ~512 rows at 64 heads) keep one dispatch: their
# threadgroups stay in step anyway.
_PASS_KEYS = _env_int("OMLX_NAX_ATTN_PASS_KEYS", 8192)
_PASS_MIN_GROUPS = 512
# Head-dim split over simdgroup pairs (MLX's attention_nax_dsplit scheme):
# 0 = never, 1 = for calls that run in key-range passes (long contexts,
# 3-4% faster there), 2 = always. It changes the summation order of
# Q @ K.T (two 96-dim fp32 partial sums added), so its output differs from
# the one-simdgroup kernel in the last bf16 bit of ~0.5% of the elements.
_DSPLIT = _env_int("OMLX_NAX_ATTN_DSPLIT", 1)

_METAL_TYPES = {mx.bfloat16: "bfloat16_t", mx.float16: "half"}

_HEADER = NAX_TILE_HEADER + r"""
namespace omlx_nax {

// Scalar parameters of one attention call (MLX's AttnParams without the
// strides, which the kernel reads from the inputs' own stride vectors).
struct AttnParams {
  int B;
  int H;
  int D;
  int qL;
  int kL;
  int gqa_factor;
  float scale;
  int NQ;
  int NK;
  int NQ_aligned;
  int NK_aligned;
  int qL_rem;
  int kL_rem;
  int qL_off;
  // Key blocks [kb_begin, kb_end) of this dispatch (key-range passes).
  int kb_begin;
  int kb_end;
};

struct MaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct SumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct ExpSubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return fast::exp2(x - y);
  }
};

// MLX's attention_nax with a value head dim BDV <= BD. The plumbing
// differs: strides come from the inputs, the output is written as
// [B, qL, H, BDV] rows (MLX's SDPA output layout), function constants are
// template arguments, and the mask column stride is honoured. Beyond that:
//
// * Key-range passes. One dispatch covers key blocks [kb_begin, kb_end).
//   Unless FIRST, the online-softmax state of every query row (fp32 O
//   accumulator, running max, running sum) is loaded from Sin; unless LAST it
//   is stored to Sout instead of normalising and writing the output. Every
//   row thus runs exactly the operations of one full dispatch, in the same
//   order (bit-identical output). Within a dispatch the threadgroups in
//   flight only stream one key slice of their KV head, which stays on chip;
//   over one long dispatch they drift apart and, once the KV head no longer
//   fits in the caches, each re-streams it from DRAM.
// * The query tile stays in registers instead of being re-read from
//   memory for every key block, and each row max is reduced over all of the
//   row's score fragments before the cross-lane shuffles (max is exact, so
//   the grouping does not matter).
// * WN = 2 splits the head dims over simdgroup pairs, as MLX's
//   attention_nax_dsplit does for 256-wide heads: each simdgroup of a pair
//   computes Q @ K.T over half of the query/key dims and P @ V for half of
//   the value dims, the pair adds its partial scores through threadgroup
//   memory, and both run the softmax on the full score tile. Half the O
//   accumulator and query registers per thread; the scores become the sum
//   of two fp32 partial dot products (a summation-order change).
template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int BDV,
    int WM,
    int WN,
    bool align_Q,
    bool align_K,
    bool has_mask,
    bool do_causal,
    bool has_sinks,
    bool FIRST,
    bool LAST,
    typename MaskType,
    typename AccumType,
    typename StridePtr,
    typename MaskPtr,
    typename SinkPtr,
    typename StatePtr>
METAL_FUNC void attention_nax_bdv(
    const device T* Q,
    const device T* K,
    const device T* V,
    device T* O,
    const device AttnParams* params,
    StridePtr q_str,
    StridePtr k_str,
    StridePtr v_str,
    StridePtr m_str,
    MaskPtr mask,
    SinkPtr sinks,
    StatePtr Sin,
    device float* Sout,
    threadgroup AccumType* xchg,
    uint simd_group_id,
    uint simd_lane_id,
    uint3 tid) {
  ulong3 tidl{tid.x, tid.y, tid.z};

  const int64_t Q_strides[3] = {q_str[0], q_str[1], q_str[2]};
  const int64_t K_strides[3] = {k_str[0], k_str[1], k_str[2]};
  const int64_t V_strides[3] = {v_str[0], v_str[1], v_str[2]};
  const int64_t O_strides[3] = {
      int64_t(params->qL) * params->H * BDV, BDV, int64_t(params->H) * BDV};

  Q += tidl.z * Q_strides[0] + // Batch
      tidl.y * Q_strides[1] + // Head
      tidl.x * BQ * Q_strides[2]; // Sequence

  ulong kv_head_idx = int(tid.y) / params->gqa_factor;
  K += tidl.z * K_strides[0] + // Batch
      kv_head_idx * K_strides[1]; // Head

  V += tidl.z * V_strides[0] + // Batch
      kv_head_idx * V_strides[1]; // Head

  O += tidl.z * O_strides[0] + // Batch
      tidl.y * O_strides[1] + // Head
      tidl.x * BQ * O_strides[2]; // Sequence

  if (has_mask) {
    mask += tidl.z * m_str[0] + // Batch
        tidl.y * m_str[1]; // Head
  }

  const metal::uniform<float> scale2 =
      make_uniform(params->scale) * make_uniform(1.44269504089f);

  // Prepare MMA tiles
  constexpr short kU = 16;

  // WM groups of 16 query rows; each group's WN simdgroups split the head
  // dims (WN = 2: MLX's attention_nax_dsplit scheme; see the note above).
  static_assert(BQ == WM * kU, "One 16-row fragment per row group");
  static_assert(WN == 1 || WN == 2, "Head dims split over 1 or 2 simdgroups");

  // Q seq frags per warp
  constexpr int TQ = 1;
  // HeadDim frags of this simdgroup
  constexpr int TD = BD / kU / WN;
  // Value head dim frags of this simdgroup
  constexpr int TDV = BDV / kU / WN;
  // KV seq frags per warp
  constexpr short TK = BK / kU;

  static_assert(TD * kU * WN == BD, "The head dim must split evenly");
  static_assert(TDV % 2 == 0, "P@V accumulates output fragments in pairs");
  using otile_t = NAXTile<AccumType, TQ, TDV>;
  otile_t Otile;

  Otile.clear();

  // Prepare mma tile offsets: rows of this row group, columns of this
  // simdgroup's share of the head dims.
  const short row_group = simd_group_id / WN;
  const short d_part = simd_group_id % WN;
  const short tm = kU * TQ * row_group;
  Q += tm * int(Q_strides[2]) + d_part * (BD / WN);
  K += d_part * (BD / WN);
  V += d_part * (BDV / WN);
  O += d_part * (BDV / WN);

  const short2 simd_coord = otile_t::NAXFrag_t::get_coord();
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;

  // Init row reduction variables
  constexpr short kRowsPT = otile_t::kRowsPerThread;

  metal::vec<AccumType, kRowsPT> max_score;
  metal::vec<AccumType, kRowsPT> sum_score{0};

  // Online-softmax state rows of this simdgroup in the pass buffers:
  // [B, H, NQ * BQ] rows of BDV + 2 floats (O accumulator, max, sum).
  constexpr int kSW = BDV + 2;
  const int64_t srow0 = (int64_t(tid.z) * params->H + tid.y) *
          (int64_t(params->NQ) * BQ) +
      int64_t(tid.x) * BQ + tm;

  if constexpr (FIRST) {
    // Init to -Inf
    OMLX_NAX_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = Limits<AccumType>::finite_min;
    }

    if (has_sinks) {
      OMLX_NAX_UNROLL
      for (short i = 0; i < kRowsPT; ++i) {
        max_score[i] = M_LOG2E_F * static_cast<AccumType>(sinks[tidl.y]);
        sum_score[i] = 1;
      }
    }
  } else {
    // Resume the previous pass (rows past a query tail are never output).
    Otile.load(Sin + srow0 * kSW + d_part * (BDV / WN), kSW);
    OMLX_NAX_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      const int64_t r = srow0 + sm + i * otile_t::kFragRowsJump;
      max_score[i] = Sin[r * kSW + BDV];
      sum_score[i] = Sin[r * kSW + BDV + 1];
    }
  }

  int kb_lim = params->NK;
  int kb_min_causal = params->NK;

  if (do_causal) {
    int q_max = (tid.x + 1) * BQ + params->qL_off;
    kb_lim = (q_max + BK - 1) / BK;
    kb_lim = min(params->NK, kb_lim);

    int q_min = tid.x * BQ + params->qL_off;
    q_min = max(0, q_min);
    kb_min_causal = (q_min / BK);
  }

  const bool is_last_bq = int(tid.x) == (params->NQ_aligned);
  const bool is_last_q = is_last_bq;

  const short lim_rows_q = params->qL_rem - tm;
  const short lim_rows_k = params->kL_rem;

  // This dispatch's key blocks.
  const int kb_lo = params->kb_begin;
  const int kb_hi = min(kb_lim, params->kb_end);
  K += int64_t(kb_lo) * BK * K_strides[2];
  V += int64_t(kb_lo) * BK * V_strides[2];

  // Query fragments, loaded once and kept in registers.
  NAXTile<T, TQ, TD> Qreg;
  if (!align_Q && is_last_q) {
    Qreg.load_rows(Q, int(Q_strides[2]), lim_rows_q);
  } else {
    Qreg.load(Q, int(Q_strides[2]));
  }

  // Loop over KV seq length
  for (int kb = kb_lo; kb < kb_hi; kb++) {
    const int is_last_k = (kb == (params->NK_aligned));

    // Do S = Q @ K.T
    using stile_t = NAXTile<AccumType, TQ, TK>;
    stile_t Stile;

    Stile.clear();

    OMLX_NAX_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      OMLX_NAX_UNROLL
      for (short ik = 0; ik < TK; ik += 2) {
        auto qk_step = [&](short id) {
          NAXTile<T, 1, 1> Qtile;
          NAXTile<T, 2, 1> Ktile;

          const int K_load_off = ik * kU * int(K_strides[2]) + id * kU;

          Qtile.frag_at(0, 0) = Qreg.frag_at(iq, id);

          if (!align_K && is_last_k) {
            Ktile.load_rows(
                K + K_load_off, int(K_strides[2]), lim_rows_k - ik * kU);
          } else {
            Ktile.load(K + K_load_off, int(K_strides[2]));
          }

          stile_t::NAXFrag_t::mma(
              Stile.frag_at(iq, ik),
              Stile.frag_at(iq, ik + 1),
              Qtile.frag_at(0, 0),
              metal::false_type{},
              Ktile.frag_at(0, 0),
              Ktile.frag_at(1, 0),
              metal::true_type{});
        };
        if constexpr (WN == 2) {
          // The head-dim split kernel keeps static fragment indices.
          OMLX_NAX_UNROLL
          for (short id = 0; id < TD; id++) {
            qk_step(id);
          }
        } else {
#pragma clang loop unroll_count(4)
          for (short id = 0; id < TD; id++) {
            qk_step(id);
          }
        }

        if constexpr (WN == 2) {
          // Add the peer's partial scores (its half of the head dims).
          constexpr short kEPF = stile_t::NAXFrag_t::kElemsPerFrag;
          threadgroup AccumType* slot =
              xchg + (row_group * WN + d_part) * (2 * kEPF * 32);
          const threadgroup AccumType* peer =
              xchg + (row_group * WN + 1 - d_part) * (2 * kEPF * 32);
          thread auto& s0 = Stile.frag_at(iq, ik);
          thread auto& s1 = Stile.frag_at(iq, ik + 1);
          const short base = short(simd_lane_id) * (2 * kEPF);
          OMLX_NAX_UNROLL
          for (short i = 0; i < kEPF; i++) {
            slot[base + i] = s0[i];
            slot[base + kEPF + i] = s1[i];
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          OMLX_NAX_UNROLL
          for (short i = 0; i < kEPF; i++) {
            s0[i] += peer[base + i];
            s1[i] += peer[base + kEPF + i];
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
      }
    }

    // Scale S
    OMLX_NAX_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= float(scale2);
    }

    // Mask out length sequence
    if (!align_K && is_last_k) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      OMLX_NAX_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        OMLX_NAX_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const short col_pos = ik * kU + sn;

          thread auto& fg = Stile.frag_at(iq, ik);

          OMLX_NAX_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            OMLX_NAX_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = ((col_pos + jj) < params->kL_rem) ? fg[loc] : neg_inf;
            }
          }
        }
      }
    }

    // Mask out if causal
    if (do_causal && kb >= kb_min_causal) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      const int base_row = tid.x * BQ + params->qL_off + tm;
      const int base_col = kb * BK;

      OMLX_NAX_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        OMLX_NAX_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          thread auto& fg = Stile.frag_at(iq, ik);

          OMLX_NAX_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            OMLX_NAX_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto r =
                  base_row + iq * kU + ii * stile_t::kFragRowsJump + sm;
              const auto c = base_col + ik * kU + jj + sn;
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = (r < c) ? neg_inf : fg[loc];
            }
          }
        }
      }
    }

    // Other masking as needed
    if (has_mask) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      const int base_row = tid.x * BQ + tm;
      const int base_col = kb * BK;

      constexpr bool is_bool = is_same_v<MaskType, bool>;
      using melem_t = typename metal::conditional_t<is_bool, bool, AccumType>;
      using mtile_t = NAXTile<melem_t, TQ, TK>;
      using mfrag_t = typename mtile_t::frag_type;

      if (base_row + BQ <= params->qL && base_col + BK <= params->kL) {
        for (short iq = 0; iq < TQ; iq++) {
          OMLX_NAX_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            const int row_pos = base_row + iq * kU;
            const int col_pos = base_col + ik * kU;

            mfrag_t mfrag;
            mtile_t::NAXFrag_t::load(
                mfrag,
                mask,
                int64_t(m_str[2]),
                int64_t(m_str[3]),
                row_pos,
                col_pos);

            thread auto& fg = Stile.frag_at(iq, ik);

            OMLX_NAX_UNROLL
            for (short jj = 0; jj < mtile_t::kElemsPerFrag; jj++) {
              if constexpr (is_bool) {
                fg[jj] = mfrag[jj] ? fg[jj] : neg_inf;
              } else {
                fg[jj] += M_LOG2E_F * AccumType(mfrag[jj]);
              }
            }
          }
        }
      } else {
        OMLX_NAX_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
          OMLX_NAX_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            const int row_pos = base_row + iq * kU;
            const int col_pos = base_col + ik * kU;

            mfrag_t mfrag;
            mtile_t::NAXFrag_t::load_safe(
                mfrag,
                mask,
                int64_t(m_str[2]),
                int64_t(m_str[3]),
                params->qL,
                params->kL,
                row_pos,
                col_pos);

            thread auto& fg = Stile.frag_at(iq, ik);

            OMLX_NAX_UNROLL
            for (short jj = 0; jj < mtile_t::kElemsPerFrag; jj++) {
              if constexpr (is_bool) {
                fg[jj] = mfrag[jj] ? fg[jj] : neg_inf;
              } else {
                fg[jj] += M_LOG2E_F * AccumType(mfrag[jj]);
              }
            }
          }
        }
      }
    }

    // Do softmax

    // Temp variables
    metal::vec<AccumType, kRowsPT> new_max;
    metal::vec<AccumType, kRowsPT> factor;
    OMLX_NAX_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    // Row max: all of a row's fragments first, then across lanes (the
    // grouping is free, max is exact).
    OMLX_NAX_UNROLL
    for (short iq = 0; iq < TQ; ++iq) {
      OMLX_NAX_UNROLL
      for (short ii = 0; ii < stile_t::kFragThrRows; ++ii) {
        const short loc0 = ii * stile_t::kFragThrCols;
        AccumType m = Stile.frag_at(iq, 0)[loc0];
        OMLX_NAX_UNROLL
        for (short ik = 0; ik < TK; ++ik) {
          OMLX_NAX_UNROLL
          for (short jj = 0; jj < stile_t::kFragThrCols; ++jj) {
            m = metal::max(m, Stile.frag_at(iq, ik)[loc0 + jj]);
          }
        }
        m = metal::max(m, simd_shuffle_xor(m, ushort(1)));
        m = metal::max(m, simd_shuffle_xor(m, ushort(8)));
        const short r = iq * stile_t::kFragThrRows + ii;
        new_max[r] = metal::max(new_max[r], m);
      }
    }

    // exp(Si - rowmax(Si))
    Stile.template row_bin_op<ExpSubOp>(new_max);

    // Factor exp(rowmax(Si) - rowmax(Si-1))
    OMLX_NAX_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }

    // Row Sum
    OMLX_NAX_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i];
    }

    Stile.template row_reduce<SumOp>(sum_score);

    // Update O
    Otile.template row_bin_op<MulOp>(factor);

    simdgroup_barrier(mem_flags::mem_none);

    // Do O = P @ V
    OMLX_NAX_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      OMLX_NAX_UNROLL
      for (short id = 0; id < TDV; id += 2) {
        if constexpr (BDV == 128 && WN == 1) {
          if (id == 4) {
            threadgroup_barrier(mem_flags::mem_none);
          }
        }

        OMLX_NAX_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          NAXTile<T, 1, 2> Vtile;

          const int V_load_off = ik * kU * int(V_strides[2]) + id * kU;

          if (!align_K && is_last_k) {
            Vtile.load_rows(
                V + V_load_off, int(V_strides[2]), lim_rows_k - ik * kU);
          } else {
            Vtile.load(V + V_load_off, int(V_strides[2]));
          }

          otile_t::NAXFrag_t::mma(
              Otile.frag_at(iq, id),
              Otile.frag_at(iq, id + 1),
              Stile.frag_at(iq, ik),
              metal::false_type{},
              Vtile.frag_at(0, 0),
              Vtile.frag_at(0, 1),
              metal::false_type{});
        }
      }
    }

    // Prepare for next iteration
    K += BK * int(K_strides[2]);
    V += BK * int(V_strides[2]);
  }

  threadgroup_barrier(mem_flags::mem_none);

  if constexpr (!LAST) {
    // Hand the row state to the next pass.
    Otile.store(Sout + srow0 * kSW + d_part * (BDV / WN), kSW);
    if (sn == 0 && d_part == 0) {
      OMLX_NAX_UNROLL
      for (short i = 0; i < kRowsPT; ++i) {
        const int64_t r = srow0 + sm + i * otile_t::kFragRowsJump;
        Sout[r * kSW + BDV] = max_score[i];
        Sout[r * kSW + BDV + 1] = sum_score[i];
      }
    }
    return;
  }

  // Normalize output

  metal::vec<AccumType, kRowsPT> rcp;
  OMLX_NAX_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    rcp[i] = 1.f / sum_score[i];
  }

  Otile.template row_bin_op<MulOp>(rcp);

  // Store results
  O += tm * int(O_strides[2]);

  if (!align_Q && is_last_q) {
    if (lim_rows_q <= 0)
      return;

    Otile.store_rows(O, int(O_strides[2]), lim_rows_q);
  } else {
    Otile.store(O, int(O_strides[2]));
  }
}

} // namespace omlx_nax
"""

# One generated kernel per variant; the MLX kernel's function constants are
# baked in as template arguments of the call.
_SOURCE = r"""
  threadgroup float xchg[{WN} == 2 ? {WM} * 2 * 16 * 32 : 1];
  omlx_nax::attention_nax_bdv<
      {T}, {BQ}, {BK}, {BD}, {BDV}, {WM}, {WN},
      {ALIGN_Q}, {ALIGN_K}, {HAS_MASK}, {DO_CAUSAL}, {HAS_SINKS},
      {FIRST}, {LAST}, bool, float>(
      q, k, v, out,
      reinterpret_cast<const device omlx_nax::AttnParams*>(params),
      q_strides, k_strides, v_strides, mask_strides,
      mask, sinks, state, state_out, xchg,
      simdgroup_index_in_threadgroup,
      thread_index_in_simdgroup,
      threadgroup_position_in_grid);
"""


def _flag(value: bool) -> str:
    return "true" if value else "false"


@lru_cache(maxsize=None)
def _kernel(
    dtype: mx.Dtype,
    bd: int,
    bdv: int,
    align_q: bool,
    align_k: bool,
    has_mask: bool,
    do_causal: bool,
    has_sinks: bool,
    first: bool = True,
    last: bool = True,
    wn: int = 1,
):
    wm = _WM // wn
    source = (
        _SOURCE.replace("{T}", _METAL_TYPES[dtype])
        .replace("{BQ}", str(16 * wm))
        .replace("{BK}", str(_BK))
        .replace("{BD}", str(bd))
        .replace("{BDV}", str(bdv))
        .replace("{WM}", str(wm))
        .replace("{WN}", str(wn))
        .replace("{ALIGN_Q}", _flag(align_q))
        .replace("{ALIGN_K}", _flag(align_k))
        .replace("{HAS_MASK}", _flag(has_mask))
        .replace("{DO_CAUSAL}", _flag(do_causal))
        .replace("{HAS_SINKS}", _flag(has_sinks))
        .replace("{FIRST}", _flag(first))
        .replace("{LAST}", _flag(last))
    )
    tag = "".join(
        "1" if f else "0"
        for f in (
            align_q,
            align_k,
            has_mask,
            do_causal,
            has_sinks,
            first,
            last,
        )
    )
    return mx.fast.metal_kernel(
        name=f"omlx_nax_attention_v2_bd{bd}_bdv{bdv}_wn{wn}_{tag}",
        input_names=["q", "k", "v", "mask", "sinks", "state", "params"],
        output_names=["out", "state_out"],
        header=_HEADER,
        source=source,
        ensure_row_contiguous=False,
    )


@lru_cache(maxsize=1)
def _nax_available() -> bool:
    try:
        return bool(is_nax_available())
    except Exception:  # noqa: BLE001
        return False


def _pass_edges(
    n_groups: int, nk: int, pass_keys: int, min_groups: Optional[int] = None
) -> list[int]:
    """Key-block boundaries of the dispatches for ``nk`` key blocks."""
    if min_groups is None:
        min_groups = _PASS_MIN_GROUPS
    if pass_keys <= 0 or n_groups < min_groups:
        return [0, nk]
    n_pass = max(1, min(nk, round(nk * _BK / pass_keys)))
    return [(i * nk) // n_pass for i in range(n_pass + 1)]


def _run(
    q, k, v, scale, mask, sinks, pass_keys=None, min_groups=None, dsplit=None
) -> mx.array:
    B, H, qL, _ = q.shape
    NK = (k.shape[2] + _BK - 1) // _BK
    edges = _pass_edges(
        B * H * ((qL + _BQ - 1) // _BQ),
        NK,
        _PASS_KEYS if pass_keys is None else pass_keys,
        min_groups,
    )
    mode = _DSPLIT if dsplit is None else dsplit
    wn = 2 if (mode == 2 or (mode == 1 and len(edges) > 2)) else 1
    if len(edges) > 2 and not _passes_check_passed(wn):
        edges, wn = [0, NK], 1
    return _run_edges(q, k, v, scale, mask, sinks, edges, wn)


def _run_edges(q, k, v, scale, mask, sinks, edges, wn) -> mx.array:
    """Dispatch the key blocks between consecutive ``edges`` in turn."""
    B, H, qL, D = q.shape
    kL = k.shape[2]
    DV = v.shape[3]
    do_causal = isinstance(mask, str)
    has_mask = isinstance(mask, mx.array)
    NK = (kL + _BK - 1) // _BK
    NK_aligned = kL // _BK
    n_pass = len(edges) - 1
    bq = 16 * (_WM // wn)
    NQ = (qL + bq - 1) // bq
    NQ_aligned = qL // bq
    has_sinks = sinks is not None
    # Unused inputs get a one-element placeholder (never read).
    if has_mask:
        mask = mx.broadcast_to(mask, (B, H, qL, kL))
    else:
        mask = mx.zeros((1,), dtype=mx.bool_)
    if has_sinks:
        sinks = mx.contiguous(sinks.astype(q.dtype))
    else:
        sinks = mx.zeros((1,), dtype=q.dtype)
    # fp32 row state between passes: O accumulator, running max, running sum.
    # Passes in flight each hold their state until their command buffer ends:
    # about 3 GB per call from 128k keys at 8192 queries x 64 heads.
    state_size = B * H * NQ * bq * (DV + 2)
    state = mx.zeros((1,), dtype=mx.float32)
    out = None
    for i in range(n_pass):
        first, last = i == 0, i == n_pass - 1
        params = struct.pack(
            "<6if9i",
            B,
            H,
            D,
            qL,
            kL,
            H // k.shape[1],
            float(scale),
            NQ,
            NK,
            NQ_aligned,
            NK_aligned,
            qL - NQ_aligned * bq,
            kL - NK_aligned * _BK,
            kL - qL,
            edges[i],
            edges[i + 1],
        )
        params = mx.array(memoryview(params), dtype=mx.uint8)
        kernel = _kernel(
            q.dtype,
            D,
            DV,
            qL % bq == 0,
            kL % _BK == 0,
            has_mask,
            do_causal,
            has_sinks,
            first,
            last,
            wn,
        )
        out, state = kernel(
            inputs=[q, k, v, mask, sinks, state, params],
            grid=(NQ * _THREADS, H, B),
            threadgroup=(_THREADS, 1, 1),
            output_shapes=[
                (B, qL, H, DV) if last else (1,),
                (1,) if last else (state_size,),
            ],
            output_dtypes=[q.dtype, mx.float32],
        )
    # [B, qL, H, DV] rows viewed as [B, H, qL, DV], like MLX's SDPA output.
    return out.transpose(0, 2, 1, 3)


def _reference(q, k, v, scale, mask):
    """fp32 causal attention (for the self-check)."""
    B, H, qL, _ = q.shape
    Hk, kL = k.shape[1], k.shape[2]
    g = H // Hk
    qf = q.astype(mx.float32).reshape(B, Hk, g, qL, -1) * scale
    s = qf @ k.astype(mx.float32)[:, :, None].swapaxes(-1, -2)
    causal = (mx.arange(qL)[:, None] + (kL - qL)) >= mx.arange(kL)[None]
    s = mx.where(causal, s, -mx.inf)
    p = mx.softmax(s, axis=-1)
    return (p @ v.astype(mx.float32)[:, :, None]).reshape(B, H, qL, -1)


@lru_cache(maxsize=1)
def _self_check_passed() -> bool:
    """Compile and run one small case against fp32 once per process.

    Uses the causal variant with unaligned query/key tails, the one a
    typical first prefill chunk (L - 1 prompt tokens) needs anyway.
    """
    try:
        key = mx.random.key(192128)
        kq, kk, kv = mx.random.split(key, 3)
        q = (0.5 * mx.random.normal((1, 4, 100, 192), key=kq)).astype(mx.bfloat16)
        k = (0.5 * mx.random.normal((1, 2, 300, 192), key=kk)).astype(mx.bfloat16)
        v = (0.5 * mx.random.normal((1, 2, 300, 128), key=kv)).astype(mx.bfloat16)
        scale = 192**-0.5
        out = _run_edges(q, k, v, scale, "causal", None, [0, 10], 1)
        ref = _reference(q, k, v, scale, "causal")
        err = mx.abs(out.astype(mx.float32) - ref).max().item()
        ok = err < 2e-2
    except Exception as exc:  # noqa: BLE001 - any failure disables the route
        logger.warning("NAX JIT attention disabled: self-check failed (%s)", exc)
        return False
    if not ok:
        logger.warning(
            "NAX JIT attention disabled: self-check error %.3g vs fp32", err
        )
        return False
    logger.info("NAX JIT attention (192/128 head dims) enabled")
    return True


@lru_cache(maxsize=None)
def _passes_check_passed(wn: int) -> bool:
    """First use of key-range passes (per head-dim split): one small case.

    Several dispatches must reproduce one dispatch of the same kernel bit
    for bit (the row state is resumed exactly), and match fp32. Any failure
    keeps long contexts on one dispatch.
    """
    try:
        key = mx.random.key(192129)
        kq, kk, kv = mx.random.split(key, 3)
        q = (0.5 * mx.random.normal((1, 4, 100, 192), key=kq)).astype(mx.bfloat16)
        k = (0.5 * mx.random.normal((1, 2, 300, 192), key=kk)).astype(mx.bfloat16)
        v = (0.5 * mx.random.normal((1, 2, 300, 128), key=kv)).astype(mx.bfloat16)
        scale = 192**-0.5
        one = _run_edges(q, k, v, scale, "causal", None, [0, 10], wn)
        many = _run_edges(q, k, v, scale, "causal", None, [0, 3, 6, 10], wn)
        ref = _reference(q, k, v, scale, "causal")
        err = mx.abs(many.astype(mx.float32) - ref).max().item()
        ok = err < 2e-2 and bool(mx.array_equal(many, one).item())
    except Exception as exc:  # noqa: BLE001 - any failure keeps one dispatch
        logger.warning("NAX JIT attention key-range passes disabled (%s)", exc)
        return False
    if not ok:
        logger.warning("NAX JIT attention key-range passes disabled: check failed")
    return ok


def uses_key_passes(queries: mx.array, keys: mx.array) -> bool:
    """True when ``nax_mixed_head_dim_attention`` would split the key range.

    Single-dispatch calls compute exactly what MLX's native 192/128 kernel
    (on MLX builds that carry it) computes, at the same speed; split calls
    (long contexts) are faster, so callers may prefer this kernel then.
    """
    if not _ENABLED or queries.ndim != 4 or keys.ndim != 4:
        return False
    B, H, qL, _ = queries.shape
    NQ = (qL + _BQ - 1) // _BQ
    NK = (keys.shape[2] + _BK - 1) // _BK
    return len(_pass_edges(B * H * NQ, NK, _PASS_KEYS)) > 2


def nax_mixed_head_dim_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    scale: float,
    mask=None,
    sinks: Optional[mx.array] = None,
) -> Optional[mx.array]:
    """Fused NAX attention for ``qk_dim > v_dim`` prefill; None if unsupported.

    ``queries`` [B, H, L, 192], ``keys`` [B, Hk, S, 192], ``values``
    [B, Hk, S, 128] (bf16 or fp16, any strides with a contiguous head dim);
    ``mask`` None, ``"causal"`` (bottom-right aligned, as in MLX) or a
    boolean array broadcastable to [B, H, L, S]; ``sinks`` [H] or None.
    Returns [B, H, L, 128] in the query dtype.
    """
    if not _ENABLED:
        return None
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        return None
    B, H, qL, qk_dim = queries.shape
    _, Hk, kL, k_dim = keys.shape
    v_dim = values.shape[-1]
    if (
        (qk_dim, v_dim) not in SUPPORTED_HEAD_DIMS
        or k_dim != qk_dim
        or queries.dtype not in _METAL_TYPES
        or keys.dtype != queries.dtype
        or values.dtype != queries.dtype
        or keys.shape[0] != B
        or tuple(values.shape[:3]) != (B, Hk, kL)
        or Hk == 0
        or H % Hk
        or qL <= 8
        or kL == 0
    ):
        return None
    if isinstance(mask, str):
        if mask != "causal":
            return None
    elif mask is not None:
        if (
            not isinstance(mask, mx.array)
            or mask.dtype != mx.bool_
            or mask.ndim > 4
            or mask.ndim < 1
        ):
            return None
        try:
            if mx.broadcast_shapes(mask.shape, (B, H, qL, kL)) != (B, H, qL, kL):
                return None
        except ValueError:
            return None
    if sinks is not None and (sinks.ndim != 1 or sinks.shape[0] != H):
        return None
    if not _nax_available() or not _self_check_passed():
        return None
    return _run(queries, keys, values, scale, mask, sinks)
