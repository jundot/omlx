// Copyright © 2026 OpenAI

#pragma once

#include "mlx/backend/metal/kernels/steel/attn/attn.h"
#include "mlx/backend/metal/kernels/steel/attn/params.h"

using namespace mlx::steel;

struct DeepseekV4SparseMaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct DeepseekV4SparseSumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct DeepseekV4SparseMulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct DeepseekV4SparseExpSubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return fast::exp2(x - y);
  }
};

struct DeepseekV4SparseDivOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x / y;
  }
};

// clang-format off
template <
    typename T,
    int BK,
    int DC,
    int H,
    int D,
    int WM,
    typename IndexT,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * 32)]] void deepseek_v4_sparse_attention(
    const device T* Q [[buffer(0)]],
    const device T* LocalKV [[buffer(1)]],
    const device T* PooledKV [[buffer(2)]],
    const device IndexT* Topk [[buffer(3)]],
    const device T* Sinks [[buffer(4)]],
    device T* O [[buffer(5)]],
    const constant DeepseekV4SparseAttentionParams* params [[buffer(6)]],
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
  constexpr int D_CHUNKS = D / DC;

  static_assert(TQ >= 1, "DeepSeek sparse attention needs a head tile.");
  static_assert(
      H % (kNWarps * kFragSize) == 0,
      "Head count must divide evenly across simdgroups.");
  static_assert(BK % kFragSize == 0, "BK must be a multiple of 8.");
  static_assert(DC % kFragSize == 0, "DC must be a multiple of 8.");
  static_assert(D % DC == 0, "Head dimension must divide DC.");

  constexpr int tgp_size = WM * 32;
  const int lane = int(simd_group_id * 32 + simd_lane_id);

  const int q_pos = int(tid.x);
  const int b = int(tid.y);

  threadgroup T Qs[H * LDQ];
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
    const int head = int(tm + sm + i * kFragSize);
    if (head < params->H) {
      max_score[i] = AccumType(M_LOG2E_F) * AccumType(Sinks[head]);
      sum_score[i] = AccumType(1);
    } else {
      max_score[i] = Limits<AccumType>::finite_min;
    }
  }

  const device T* q_base = Q + size_t(b) * params->Q_strides[0] +
      size_t(q_pos) * params->Q_strides[2];
  const device T* local_base =
      LocalKV + size_t(b) * params->Local_strides[0];
  const device T* pooled_base =
      PooledKV + size_t(b) * params->Pooled_strides[0];
  const device IndexT* topk_base =
      Topk + size_t(b) * params->Topk_strides[0] +
      size_t(q_pos) * params->Topk_strides[2];

  const int local_offset = params->localL - params->qL;
  const int local_end =
      metal::min(params->localL, local_offset + q_pos + 1);
  const int local_start =
      metal::max(0, local_end - params->local_window);
  const int local_count = metal::max(0, local_end - local_start);
  const int local_tiles = (local_count + BK - 1) / BK;
  const int pooled_tiles = (params->topk + BK - 1) / BK;
  const int pooled_valid = metal::min(
      params->pooledL,
      (params->q_offset + q_pos + 1) / params->compress_ratio);
  const int total_tiles = local_tiles + pooled_tiles;

  for (int ktile = 0; ktile < total_tiles; ++ktile) {
    const bool is_pooled_tile = ktile >= local_tiles;
    const int tile_slot = is_pooled_tile ? (ktile - local_tiles) : ktile;
    const int slot_base = tile_slot * BK;

    for (int k = lane; k < BK; k += tgp_size) {
      const int slot = slot_base + k;
      int k_pos = -1;
      if (is_pooled_tile) {
        if (slot < params->topk) {
          const int pooled_pos = int(topk_base[slot]);
          if (pooled_pos >= 0 && pooled_pos < pooled_valid) {
            k_pos = pooled_pos;
          }
        }
      } else {
        if (slot < local_count) {
          k_pos = local_start + slot;
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
        Qs[h * LDQ + d] =
            q_base[size_t(h) * params->Q_strides[1] + dbase + d];
      }

      for (int elem = lane; elem < BK * DC; elem += tgp_size) {
        const int k = elem / DC;
        const int d = elem - k * DC;
        const int k_pos = selected[k];
        T value = T(0);
        if (k_pos >= 0) {
          value = is_pooled_tile
              ? pooled_base[size_t(k_pos) * params->Pooled_strides[1] +
                            dbase + d]
              : local_base[size_t(k_pos) * params->Local_strides[2] +
                           dbase + d];
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

    Stile.template row_reduce<DeepseekV4SparseMaxOp>(new_max);
    Stile.template row_bin_op<DeepseekV4SparseExpSubOp>(new_max);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < rows_per_thread; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }

    AccumType sum_score_tmp[rows_per_thread] = {0};
    Stile.template row_reduce<DeepseekV4SparseSumOp>(sum_score_tmp);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < rows_per_thread; ++i) {
      sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    }

    Otile.template row_bin_op<DeepseekV4SparseMulOp>(factor);

    STEEL_PRAGMA_UNROLL
    for (short vchunk = 0; vchunk < D_CHUNKS; ++vchunk) {
      const int dbase = int(vchunk) * DC;

      for (int elem = lane; elem < BK * DC; elem += tgp_size) {
        const int k = elem / DC;
        const int d = elem - k * DC;
        const int k_pos = selected[k];
        T value = T(0);
        if (k_pos >= 0) {
          value = is_pooled_tile
              ? pooled_base[size_t(k_pos) * params->Pooled_strides[1] +
                            dbase + d]
              : local_base[size_t(k_pos) * params->Local_strides[2] +
                           dbase + d];
        }
        KVs[k * LDV + d] = value;
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

  Otile.template row_bin_op<DeepseekV4SparseDivOp>(sum_score);

  device T* out = O + size_t(b) * params->O_strides[0] +
      size_t(q_pos) * params->O_strides[2] +
      size_t(tm + sm) * params->O_strides[1] + sn;
  Otile.template store<T, 1, 1>(out, params->O_strides[1]);
}
