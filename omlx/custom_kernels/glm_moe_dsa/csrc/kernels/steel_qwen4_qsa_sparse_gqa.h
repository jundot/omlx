// SPDX-License-Identifier: Apache-2.0
// Copyright © 2026 OpenAI

#pragma once

#include "mlx/backend/metal/kernels/steel/attn/attn.h"
#include "mlx/backend/metal/kernels/steel/attn/params.h"

using namespace mlx::steel;

struct Qwen4SparseMaxOp {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct Qwen4SparseSumOp {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct Qwen4SparseMulOp {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct Qwen4SparseExpSubOp {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) {
    return fast::exp2(x - y);
  }
};

struct Qwen4SparseDivOp {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) {
    return x / y;
  }
};

// Exact Qwen4 QSA main attention over query-specific selected four-token
// blocks.
//
// One threadgroup owns one (query row, KV head). Qwen's 12 query heads per
// KV head are padded to one 16-row Steel MMA tile, letting both main GQA groups
// reuse each randomly addressed K/V tile without materializing the enormous
// [query, selected, kv-head, dim] gathered tensors. Invalid early-prefix
// blocks are masked before online FP32 softmax. The
// selected block list is chronological, and the zero-to-three causal tail is
// generated in-kernel, so accumulation follows the checkpoint's dense-mask
// token order without materializing an expanded 2,051-index tensor.
// The 128-bit global K/V staging pattern is adapted from mlx-serve's MIT
// ``msv_attn_p256`` kernel (Copyright 2026 David Dalcu); the query-specific
// direct-index/GQA organization and Steel MMA implementation are oMLX-specific.
// clang-format off
template <
    typename T,
    int BK,
    int DC,
    int GQA,
    int H_PAD,
    int D,
    int WM,
    typename IndexT,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * 32)]] void
qwen4_qsa_sparse_gqa_attention(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    const device IndexT* Topk [[buffer(3)]],
    device T* O [[buffer(4)]],
    const constant Qwen4QSASparseGQAParams* params [[buffer(5)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) { // clang-format on

  constexpr short kFragSize = 8;
  constexpr short padQ = 16 / sizeof(T);
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);

  constexpr short LDQ = DC + padQ;
  constexpr short LDK = BK + padK;
  constexpr short LDV = DC + padV;

  constexpr int kNWarps = WM;
  constexpr int TQ = H_PAD / (kNWarps * kFragSize);
  constexpr int TK = BK / kFragSize;
  constexpr int TDC = DC / kFragSize;
  constexpr int D_CHUNKS = D / DC;

  static_assert(GQA <= H_PAD, "Qwen GQA heads must fit the padded MMA tile.");
  static_assert(TQ == 1, "Qwen sparse GQA expects one query-head tile.");
  static_assert(H_PAD % (kNWarps * kFragSize) == 0,
                "Padded query heads must divide evenly across simdgroups.");
  static_assert(BK % kFragSize == 0, "BK must be a multiple of eight.");
  static_assert(DC % kFragSize == 0, "DC must be a multiple of eight.");
  static_assert(D % DC == 0, "Head dimension must divide DC.");

  constexpr int tgp_size = WM * 32;
  const int lane = int(simd_group_id * 32 + simd_lane_id);
  const int q_pos = int(tid.x);
  const int kv_head = int(tid.y);
  const int b = int(tid.z);

  threadgroup T Qs[H_PAD * LDQ];
  threadgroup T KVs[(BK * LDV > DC * LDK) ? BK * LDV : DC * LDK];
  threadgroup int selected[BK];

  using MMAFragAcc = BaseMMAFrag<AccumType, kFragSize, kFragSize>;
  MMATile<AccumType, TQ, 1, MMAFragAcc> Qtile;
  MMATile<AccumType, 1, TK, MMAFragAcc> Ktile;
  MMATile<AccumType, TQ, TK, MMAFragAcc> Stile;
  MMATile<AccumType, 1, 1, MMAFragAcc> Vtile;
  MMATile<AccumType, TQ, D_CHUNKS * TDC, MMAFragAcc> Otile;
  Otile.clear();

  const short2 simd_coord = MMAFragAcc::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = kFragSize * TQ * simd_group_id;
  const short Qs_offset = (tm + sm) * LDQ + sn;
  const short Ks_offset = sm * LDK + sn;
  const short Vs_offset = sm * LDV + sn;

  const AccumType scale = AccumType(params->scale * M_LOG2E_F);
  constexpr short rows_per_thread = decltype(Stile)::kRowsPerThread;
  AccumType max_score[rows_per_thread];
  AccumType sum_score[rows_per_thread] = {0};
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < rows_per_thread; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  const int query_head_base = kv_head * GQA;
  const device T *q_base = Q + size_t(b) * params->Q_strides[0] +
                           size_t(query_head_base) * params->Q_strides[1] +
                           size_t(q_pos) * params->Q_strides[2];
  const device T *k_base = K + size_t(b) * params->K_strides[0] +
                           size_t(kv_head) * params->K_strides[1];
  const device T *v_base = V + size_t(b) * params->V_strides[0] +
                           size_t(kv_head) * params->V_strides[1];
  const device IndexT *topk_base = Topk + size_t(b) * params->Topk_strides[0] +
                                   size_t(q_pos) * params->Topk_strides[2];

  const int q_abs = params->q_offset + q_pos;
  constexpr int kCompressRatio = 4;
  constexpr int kTail = kCompressRatio - 1;
  const int selected_tokens = params->topk * kCompressRatio + kTail;
  const int complete_blocks = (q_abs + 1) / kCompressRatio;
  const int valid_blocks = metal::min(params->topk, complete_blocks);
  const int n_tiles = (selected_tokens + BK - 1) / BK;

  for (int ktile = 0; ktile < n_tiles; ++ktile) {
    const int topk_off = ktile * BK;
    for (int k = lane; k < BK; k += tgp_size) {
      const int slot = topk_off + k;
      int k_pos = -1;
      if (slot < params->topk * kCompressRatio) {
        const int block_slot = slot / kCompressRatio;
        if (block_slot < valid_blocks) {
          const IndexT raw_block = topk_base[block_slot];
          const ulong candidate = ulong(raw_block) * ulong(kCompressRatio) +
                                  ulong(slot % kCompressRatio);
          if (candidate < ulong(params->kL) && candidate <= ulong(q_abs)) {
            k_pos = int(candidate);
          }
        }
      } else if (slot < selected_tokens) {
        const int tail_offset = slot - params->topk * kCompressRatio;
        const int candidate = complete_blocks * kCompressRatio + tail_offset;
        if (candidate < params->kL && candidate <= q_abs) {
          k_pos = candidate;
        }
      }
      selected[k] = k_pos;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    Stile.clear();
    STEEL_PRAGMA_UNROLL
    for (short dchunk = 0; dchunk < D_CHUNKS; ++dchunk) {
      const int dbase = int(dchunk) * DC;
      for (int elem = lane; elem < H_PAD * (DC / 8); elem += tgp_size) {
        const int h = elem / (DC / 8);
        const int d8 = elem - h * (DC / 8);
        uint4 word = uint4(0);
        if (h < GQA) {
          word = *((const device uint4 *)(q_base +
                                          size_t(h) * params->Q_strides[1] +
                                          dbase) +
                   d8);
        }
        *((threadgroup uint4 *)(Qs + h * LDQ) + d8) = word;
      }
      for (int elem = lane; elem < BK * (DC / 8); elem += tgp_size) {
        const int k = elem / (DC / 8);
        const int d8 = elem - k * (DC / 8);
        const int k_pos = selected[k];
        uint4 word = uint4(0);
        if (k_pos >= 0) {
          word = *((const device uint4 *)(k_base +
                                          size_t(k_pos) * params->K_strides[2] +
                                          dbase) +
                   d8);
        }
        thread T *values = (thread T *)&word;
        const int d = d8 * 8;
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < 8; ++e) {
          KVs[k + (d + e) * LDK] = values[e];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      STEEL_PRAGMA_UNROLL
      for (short dd = 0; dd < TDC; ++dd) {
        simdgroup_barrier(mem_flags::mem_none);
        Qtile.template load<T, 1, 1, LDQ, 1>(&Qs[Qs_offset + dd * kFragSize]);
        Ktile.template load<T, 1, 1, LDK, 1>(
            &KVs[Ks_offset + dd * kFragSize * LDK]);
        simdgroup_barrier(mem_flags::mem_none);
        tile_matmad(Stile, Qtile, Ktile, Stile);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(Stile)::kElemsPerTile; ++i) {
      Stile.elems()[i] *= scale;
    }
    {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      // Early first-chunk rows can have no complete block, so entire leading
      // tiles are invalid before the causal tail in the final tile. True
      // -INFINITY makes those tiles contribute exp2(-inf - finite_max) == 0;
      // finite_min would incorrectly add one to the softmax denominator.
      constexpr auto neg_inf = selem_t(-INFINITY);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; ++i) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; ++j) {
          const short col_pos = sn + j * stile_t::kFragCols;
          STEEL_PRAGMA_UNROLL
          for (short e = 0; e < stile_t::MMAFrag_t::kElemCols; ++e) {
            if (selected[col_pos + e] < 0) {
              Stile.frag_at(i, j)[e] = neg_inf;
            }
          }
        }
      }
    }

    AccumType new_max[rows_per_thread];
    AccumType factor[rows_per_thread];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < rows_per_thread; ++i) {
      new_max[i] = max_score[i];
    }
    Stile.template row_reduce<Qwen4SparseMaxOp>(new_max);
    Stile.template row_bin_op<Qwen4SparseExpSubOp>(new_max);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < rows_per_thread; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }
    AccumType sum_score_tmp[rows_per_thread] = {0};
    Stile.template row_reduce<Qwen4SparseSumOp>(sum_score_tmp);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < rows_per_thread; ++i) {
      sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    }
    Otile.template row_bin_op<Qwen4SparseMulOp>(factor);

    STEEL_PRAGMA_UNROLL
    for (short vchunk = 0; vchunk < D_CHUNKS; ++vchunk) {
      const int dbase = int(vchunk) * DC;
      for (int elem = lane; elem < BK * (DC / 8); elem += tgp_size) {
        const int k = elem / (DC / 8);
        const int d8 = elem - k * (DC / 8);
        const int k_pos = selected[k];
        uint4 word = uint4(0);
        if (k_pos >= 0) {
          word = *((const device uint4 *)(v_base +
                                          size_t(k_pos) * params->V_strides[2] +
                                          dbase) +
                   d8);
        }
        *((threadgroup uint4 *)(KVs + k * LDV) + d8) = word;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; ++iq) {
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < TDC; ++id) {
          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ++ik) {
            const short kk = ik * kFragSize;
            const short dd = id * kFragSize;
            Vtile.template load<T, 1, 1, LDV, 1>(
                &KVs[Vs_offset + kk * LDV + dd]);
            MMAFragAcc::mma(Otile.frag_at(iq, vchunk * TDC + id),
                            Stile.frag_at(iq, ik), Vtile.frag_at(0, 0),
                            Otile.frag_at(iq, vchunk * TDC + id));
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  Otile.template row_bin_op<Qwen4SparseDivOp>(sum_score);
  device T *out = O + size_t(b) * params->O_strides[0] +
                  size_t(query_head_base + tm + sm) * params->O_strides[1] +
                  size_t(q_pos) * params->O_strides[2] + sn;
  const short rows_left = short(GQA - (tm + sm));
  if (rows_left > 0) {
    Otile.template store_safe<T, 1, 1>(out, params->O_strides[1],
                                       short2(D - sn, rows_left));
  }
}
