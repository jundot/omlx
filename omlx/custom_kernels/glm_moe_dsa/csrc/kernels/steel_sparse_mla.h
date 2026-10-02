// Copyright © 2026 Apple Inc.

#pragma once

#include "mlx/backend/metal/kernels/steel/attn/attn.h"

using namespace mlx::steel;


struct SparseMlaMaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct SparseMlaSumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct SparseMlaMulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct SparseMlaExpSubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return fast::exp2(x - y);
  }
};

struct SparseMlaDivOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x / y;
  }
};

template <typename IndexT>
METAL_FUNC int sparse_mla_topk_token(
    const device IndexT* Topk,
    const device uint* TopkLength,
    const constant GlmDsaSparseMlaParams* params,
    int b,
    int q_pos,
    int slot) {
  if (q_pos < 0 || q_pos >= params->qL || slot < 0) {
    return -1;
  }

  const int q_abs = params->qL_off + q_pos;
  const int length_limit = params->has_topk_length
      ? int(TopkLength[size_t(b) * params->TopkLength_strides[0] +
                       size_t(q_pos) * params->TopkLength_strides[1]])
      : params->topk;
  const int causal_prefix_limit = params->topk_valid_prefix
      ? metal::min(params->topk, q_abs + 1)
      : params->topk;
  const int topk_limit =
      metal::min(params->topk, metal::min(length_limit, causal_prefix_limit));
  if (slot >= topk_limit) {
    return -1;
  }

  const bool prefix_row = params->causal_prefix_indices &&
      params->topk_valid_prefix && q_pos < params->causal_prefix_rows;
  const bool implicit_causal_prefix = params->causal_prefix_indices &&
      params->topk_valid_prefix &&
      (prefix_row || (q_abs < params->topk && length_limit >= q_abs + 1));
  int k_pos = implicit_causal_prefix ? slot
                                     : int(Topk[size_t(b) * params->Topk_strides[0] +
                                                size_t(q_pos -
                                                       params->causal_prefix_rows) *
                                                    params->Topk_strides[2] +
                                                slot]);
  if (k_pos < 0 || k_pos >= params->kL || (do_causal && k_pos > q_abs)) {
    k_pos = -1;
  }
  return k_pos;
}

// clang-format off
template <
    typename T,
    int BK,
    int DC,
    int H,
    int D_LATENT,
    int D_PE,
    int WM,
    typename IndexT,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * 32)]] void sparse_mla_attention(
    const device T* Q_latent [[buffer(0)]],
    const device T* Q_pe [[buffer(1)]],
    const device T* KV_latent [[buffer(2)]],
    const device T* K_pe [[buffer(3)]],
    const device IndexT* Topk [[buffer(4)]],
    const device uint* TopkLength [[buffer(5)]],
    device T* O [[buffer(6)]],
    const constant GlmDsaSparseMlaParams* params [[buffer(7)]],
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
  constexpr int TQ = H / (kNWarps * kFragSize);
  constexpr int TK = BK / kFragSize;
  constexpr int TDC = DC / kFragSize;
  constexpr int TD_LATENT = D_LATENT / kFragSize;
  constexpr int D_TOTAL = D_LATENT + D_PE;
  constexpr int D_CHUNKS = D_TOTAL / DC;
  constexpr int V_CHUNKS = D_LATENT / DC;

  static_assert(TQ >= 1, "Sparse MLA kernel expects at least one head tile.");
  static_assert(
      H % (kNWarps * kFragSize) == 0,
      "Sparse MLA head count must divide evenly across simdgroups.");
  static_assert(BK % kFragSize == 0, "BK must be a multiple of 8.");
  static_assert(DC % kFragSize == 0, "DC must be a multiple of 8.");
  static_assert(D_TOTAL % DC == 0, "QK total dimension must divide DC.");
  static_assert(D_LATENT % DC == 0, "Latent value dimension must divide DC.");

  constexpr int tgp_size = WM * 32;
  const int lane = int(simd_group_id * 32 + simd_lane_id);

  const int q_pos = int(tid.x);
  const int b = int(tid.y);
  const int q_abs = params->qL_off + q_pos;

  threadgroup T Qs[H * LDQ];
  threadgroup T KVs[(BK * LDV > DC * LDK) ? BK * LDV : DC * LDK];
  threadgroup int selected[BK];

  using MMAFragAcc = BaseMMAFrag<AccumType, kFragSize, kFragSize>;
  MMATile<AccumType, TQ, 1, MMAFragAcc> Qtile;
  MMATile<AccumType, 1, TK, MMAFragAcc> Ktile;
  MMATile<AccumType, TQ, TK, MMAFragAcc> Stile;
  MMATile<AccumType, 1, 1, MMAFragAcc> Vtile;
  MMATile<AccumType, TQ, TD_LATENT, MMAFragAcc> Otile;

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

  const device T* q_latent_base = Q_latent +
      size_t(b) * params->Q_latent_strides[0] +
      size_t(q_pos) * params->Q_latent_strides[2];
  const device T* q_pe_base = Q_pe + size_t(b) * params->Q_pe_strides[0] +
      size_t(q_pos) * params->Q_pe_strides[2];
  const device T* kv_latent_base =
      KV_latent + size_t(b) * params->KV_latent_strides[0];
  const device T* k_pe_base = K_pe + size_t(b) * params->K_pe_strides[0];
  const bool prefix_row = params->causal_prefix_indices &&
      params->topk_valid_prefix && q_pos < params->causal_prefix_rows;
  const int topk_row = prefix_row ? 0 : q_pos - params->causal_prefix_rows;
  const device IndexT* topk_base =
      Topk + size_t(b) * params->Topk_strides[0] +
      size_t(topk_row) * params->Topk_strides[2];

  const int length_limit = params->has_topk_length
      ? int(TopkLength[size_t(b) * params->TopkLength_strides[0] +
                       size_t(q_pos) * params->TopkLength_strides[1]])
      : params->topk;
  const int causal_prefix_limit = params->topk_valid_prefix
      ? metal::min(params->topk, q_abs + 1)
      : params->topk;
  const int topk_limit =
      metal::min(params->topk, metal::min(length_limit, causal_prefix_limit));
  const bool implicit_causal_prefix = params->causal_prefix_indices &&
      params->topk_valid_prefix &&
      (prefix_row || (q_abs < params->topk && length_limit >= q_abs + 1));
  const int n_tiles = (topk_limit + BK - 1) / BK;

  for (int ktile = 0; ktile < n_tiles; ++ktile) {
    const int topk_off = ktile * BK;

    for (int k = lane; k < BK; k += tgp_size) {
      const int slot = topk_off + k;
      int k_pos = -1;
      if (slot < topk_limit) {
        k_pos = implicit_causal_prefix ? slot : int(topk_base[slot]);
        if (k_pos < 0 || k_pos >= params->kL || (do_causal && k_pos > q_abs)) {
          k_pos = -1;
        }
      }
      selected[k] = k_pos;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    Stile.clear();

    STEEL_PRAGMA_UNROLL
    for (short dchunk = 0; dchunk < D_CHUNKS; ++dchunk) {
      const int dbase = int(dchunk) * DC;

      for (int elem = lane; elem < H * DC; elem += tgp_size) {
        const int h = elem / DC;
        const int d = elem - h * DC;
        Qs[h * LDQ + d] = dbase < D_LATENT
            ? q_latent_base[size_t(h) * params->Q_latent_strides[1] + dbase + d]
            : q_pe_base[size_t(h) * params->Q_pe_strides[1] +
                        (dbase - D_LATENT) + d];
      }

      for (int elem = lane; elem < BK * DC; elem += tgp_size) {
        const int k = elem / DC;
        const int d = elem - k * DC;
        const int k_pos = selected[k];
        T value = T(0);
        if (k_pos >= 0) {
          value = dbase < D_LATENT
              ? kv_latent_base[size_t(k_pos) * params->KV_latent_strides[2] +
                               dbase + d]
              : k_pe_base[size_t(k_pos) * params->K_pe_strides[2] +
                          (dbase - D_LATENT) + d];
        }
        KVs[k + d * LDK] = value;
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
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ++ii) {
      Stile.elems()[ii] *= scale;
    }

    {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; ++i) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; ++j) {
          const short col_pos = sn + j * stile_t::kFragCols;
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; ++jj) {
            if (selected[col_pos + jj] < 0) {
              Stile.frag_at(i, j)[jj] = neg_inf;
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

    Stile.template row_reduce<SparseMlaMaxOp>(new_max);
    Stile.template row_bin_op<SparseMlaExpSubOp>(new_max);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < rows_per_thread; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }

    AccumType sum_score_tmp[rows_per_thread] = {0};
    Stile.template row_reduce<SparseMlaSumOp>(sum_score_tmp);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < rows_per_thread; ++i) {
      sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    }

    Otile.template row_bin_op<SparseMlaMulOp>(factor);

    STEEL_PRAGMA_UNROLL
    for (short vchunk = 0; vchunk < V_CHUNKS; ++vchunk) {
      const int dbase = int(vchunk) * DC;

      for (int elem = lane; elem < BK * DC; elem += tgp_size) {
        const int k = elem / DC;
        const int d = elem - k * DC;
        const int k_pos = selected[k];
        KVs[k * LDV + d] = k_pos >= 0
            ? kv_latent_base[size_t(k_pos) * params->KV_latent_strides[2] +
                             dbase + d]
            : T(0);
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
            MMAFragAcc::mma(
                Otile.frag_at(iq, vchunk * TDC + id),
                Stile.frag_at(iq, ik),
                Vtile.frag_at(0, 0),
                Otile.frag_at(iq, vchunk * TDC + id));
          }
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  Otile.template row_bin_op<SparseMlaDivOp>(sum_score);

  device T* out = O + size_t(b) * params->O_strides[0] +
      size_t(q_pos) * params->O_strides[2] +
      size_t(tm + sm) * params->O_strides[1] + sn;
  Otile.template store<T, 1, 1>(out, params->O_strides[1]);
}
