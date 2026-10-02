// Copyright © 2024-25 Apple Inc.

#include "mlx/backend/metal/kernels/steel/attn/attn.h"

using namespace mlx::steel;

///////////////////////////////////////////////////////////////////////////////
// GEMM kernels
///////////////////////////////////////////////////////////////////////////////

constant bool align_Q [[function_constant(200)]];
constant bool align_K [[function_constant(201)]];

constant bool has_mask [[function_constant(300)]];
constant bool do_causal [[function_constant(301)]];
constant bool has_sinks [[function_constant(302)]];
constant bool has_block_mask [[function_constant(303)]];
constant bool has_block_token_mask [[function_constant(304)]];
constant bool has_block_indices [[function_constant(305)]];
// Chunked-dispatch partial outputs (issue #2225). Defaulted so pipelines
// built by callers that predate the constant (glm_moe_dsa) keep compiling.
constant bool output_partials [[function_constant(306)]];
constant bool output_partials_defined =
    is_function_constant_defined(output_partials);
constant bool use_output_partials =
    output_partials_defined && output_partials;

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

struct SubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x - y;
  }
};

struct ExpSubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return fast::exp2(x - y);
  }
};

struct DivOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x / y;
  }
};

// clang-format off
template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int WM,
    int WN,
    typename MaskType = float,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant AttnParams* params [[buffer(4)]],
    const constant AttnMaskParams* mask_params [[buffer(5), function_constant(has_mask)]],
    const device MaskType* mask [[buffer(6), function_constant(has_mask)]],
    const device T* sinks [[buffer(7), function_constant(has_sinks)]],
    const constant AttnBlockMaskParams* block_mask_params [[buffer(8), function_constant(has_block_mask)]],
    const device bool* block_mask [[buffer(9), function_constant(has_block_mask)]],
    const constant AttnBlockTokenMaskParams* block_token_mask_params [[buffer(10), function_constant(has_block_token_mask)]],
    const device uint* block_token_mask [[buffer(11), function_constant(has_block_token_mask)]],
    const constant AttnBlockIndexParams* block_index_params [[buffer(12), function_constant(has_block_indices)]],
    const device uint* block_indices [[buffer(13), function_constant(has_block_indices)]],
    device T* O_part [[buffer(14), function_constant(use_output_partials)]],
    device float* lse_part [[buffer(15), function_constant(use_output_partials)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on

  // Pacifying compiler
  (void)lid;

  // Move to correct block
  ulong3 tidl{tid.x, tid.y, tid.z};

  Q += tidl.z * params->Q_strides[0] + // Batch
      tidl.y * params->Q_strides[1] + // Head
      tidl.x * BQ * params->Q_strides[2]; // Sequence

  ulong kv_head_idx = int(tid.y) / params->gqa_factor;
  K += tidl.z * params->K_strides[0] + // Batch
      kv_head_idx * params->K_strides[1]; // Head

  V += tidl.z * params->V_strides[0] + // Batch
      kv_head_idx * params->V_strides[1]; // Head

  // In chunked-dispatch mode the (normalized) partial output goes to the
  // per-chunk slab instead of O; a reduce pass folds the chunks afterwards.
  device T* Odst = use_output_partials ? O_part : O;
  Odst += tidl.z * params->O_strides[0] + // Batch
      tidl.y * params->O_strides[1] + // Head
      tidl.x * BQ * params->O_strides[2]; // Sequence

  if (has_mask) {
    mask += tidl.z * mask_params->M_strides[0] + // Batch
        tidl.y * mask_params->M_strides[1]; // Head
  }
  if (has_block_mask) {
    block_mask += tidl.z * block_mask_params->BM_strides[0] + // Batch
        tidl.y * block_mask_params->BM_strides[1] + // Head
        tidl.x * block_mask_params->BM_strides[2]; // Query block
  }
  if (has_block_token_mask) {
    block_token_mask += tidl.z * block_token_mask_params->BTM_strides[0] + // Batch
        tidl.y * block_token_mask_params->BTM_strides[1]; // Head
  }
  if (has_block_indices) {
    block_indices += tidl.z * block_index_params->BI_strides[0] + // Batch
        tidl.y * block_index_params->BI_strides[1] + // Head
        tidl.x * block_index_params->BI_strides[2]; // Query block
  }

  // Prepare threadgroup memory
  constexpr short padQ = 16 / sizeof(T);
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);

  constexpr short LDQ_tgp = BD + padQ;
  constexpr short LDK_tgp = BK + padK;
  constexpr short LDV_tgp = BD + padV;

  constexpr short tgp_mem_0 = (BK + padK) * (BD);
  constexpr short tgp_mem_1 = BK * (BD + padV);
  constexpr short tgp_mem_s = tgp_mem_0 > tgp_mem_1 ? tgp_mem_0 : tgp_mem_1;

  threadgroup T Q_smem[BQ * (BD + padQ)];
  threadgroup T KV_smem[tgp_mem_s];

  threadgroup T* Qs = Q_smem;
  threadgroup T* Ks = KV_smem;
  threadgroup T* Vs = KV_smem;

  // Prepare block loaders
  using QBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BQ,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ LDQ_tgp,
      /* short kDstStrCol = */ 1,
      /* short reduction_dim = */ 1,
      /* short tgp_size = */ WM * WN * 32>;

  // K is loaded in transposed
  using KBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BK,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ 1,
      /* short kDstStrCol = */ LDK_tgp,
      /* short reduction_dim = */ 0,
      /* short tgp_size = */ WM * WN * 32>;

  using VBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BK,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ LDV_tgp,
      /* short kDstStrCol = */ 1,
      /* short reduction_dim = */ 0,
      /* short tgp_size = */ WM * WN * 32>;

  QBlockLoader loader_q(
      Q, params->Q_strides[2], Qs, simd_group_id, simd_lane_id);
  const AccumType scale = params->scale * M_LOG2E_F;

  // Prepare MMA tiles
  constexpr short kFragSize = 8; // MMAFrag size
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

  constexpr int kNWarps = WM * WN;
  static_assert(
      BQ >= (kNWarps * kFragSize) && BQ % (kNWarps * kFragSize) == 0,
      "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");

  // Q seq frags per warp
  constexpr int TQ = BQ / (kNWarps * kFragSize);
  // KV sequence frags (all warps load the same frags)
  constexpr int TK = BK / kFragSize;
  // HeadDim frags (all warps load the same frags)
  constexpr int TD = BD / kFragSize;

  static_assert(TQ == 1, "Check TQ");

  MMATile<AccumType, TQ, 1, MMAFrag_acc_t> Qtile;
  MMATile<AccumType, 1, TK, MMAFrag_acc_t> Ktile;
  MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Stile;
  MMATile<AccumType, 1, 1, MMAFrag_acc_t> Vtile;
  MMATile<AccumType, TQ, TD, MMAFrag_acc_t> Otile;

  Otile.clear();

  // Prepare mma tile offsets
  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = kFragSize * TQ * simd_group_id;

  const short Qs_offset = (tm + sm) * LDQ_tgp + sn;
  const short Ks_offset = sm * LDK_tgp + sn;
  const short Vs_offset = sm * LDV_tgp + sn;

  constexpr short Qs_tile_stride = kFragSize;
  constexpr short Ks_tile_stride = kFragSize * LDK_tgp;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Load Q blocks
  if (!align_Q && int(tid.x) == (params->NQ_aligned)) {
    loader_q.load_safe(short2(BD, params->qL_rem));
  } else {
    loader_q.load_unsafe();
  }

  // Init row reduction variables
  constexpr short kRowsPT = decltype(Stile)::kRowsPerThread;

  AccumType max_score[kRowsPT];
  AccumType sum_score[kRowsPT] = {0};

  // Init to -Inf
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  if (has_sinks) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = M_LOG2E_F * static_cast<AccumType>(sinks[tidl.y]);
      sum_score[i] = 1;
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

  const int kb_iter_lim =
      has_block_indices ? block_index_params->budget : kb_lim;

  // Loop over selected or dense KV blocks
  for (int kb_iter = 0; kb_iter < kb_iter_lim; kb_iter++) {
    int kb = has_block_indices ? int(block_indices[kb_iter]) : kb_iter;
    if (kb < 0 || kb >= kb_lim) {
      continue;
    }
    if (!has_block_indices && has_block_mask && !block_mask[kb]) {
      continue;
    }

    thread KBlockLoader loader_k(
        K + size_t(kb) * BK * params->K_strides[2],
        params->K_strides[2],
        Ks,
        simd_group_id,
        simd_lane_id);
    thread VBlockLoader loader_v(
        V + size_t(kb) * BK * params->V_strides[2],
        params->V_strides[2],
        Vs,
        simd_group_id,
        simd_lane_id);

    // Load K block and apply scale
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!align_K && kb == (params->NK_aligned)) {
      loader_k.load_safe(short2(BD, params->kL_rem));
    } else {
      loader_k.load_unsafe();
    }

    // Do S = Q @ K.T
    Stile.clear();

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);

      Qtile.template load<T, 1, 1, LDQ_tgp, 1>(
          &Qs[Qs_offset + dd * Qs_tile_stride]);
      Ktile.template load<T, 1, 1, LDK_tgp, 1>(
          &Ks[Ks_offset + dd * Ks_tile_stride]);

      simdgroup_barrier(mem_flags::mem_none);

      tile_matmad(Stile, Qtile, Ktile, Stile);
    }

    // Apply scale in float32
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale;
    }

    // Mask out length sequence
    if (!align_K && kb == (params->NK_aligned)) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          short col_pos = sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if ((col_pos + jj) >= params->kL_rem) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    // Mask out if causal
    if (do_causal && kb >= kb_min_causal) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos =
            tid.x * BQ + params->qL_off + tm + sm + (i * stile_t::kFragRows);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if (row_pos < (col_pos + jj)) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    if (has_block_token_mask) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos = tid.x * BQ + tm + sm + (i * stile_t::kFragRows);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);
          uint bits = 0;
          if (row_pos < params->qL) {
            bits = block_token_mask
                [row_pos * block_token_mask_params->BTM_strides[2] + kb];
          }
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            uint bit = 1u << uint((col_pos + jj) - kb * BK);
            if (row_pos >= params->qL || (bits & bit) == 0) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    // Other masking as needed
    if (has_mask) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      constexpr bool is_bool = is_same_v<MaskType, bool>;
      using melem_t = typename metal::conditional_t<is_bool, bool, selem_t>;

      using MMAFrag_mask_t = BaseMMAFrag<melem_t, kFragSize, kFragSize>;
      using frag_t = typename MMAFrag_mask_t::frag_type;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos = tid.x * BQ + tm + sm + (i * stile_t::kFragRows);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);

          frag_t mfrag;

          MMAFrag_mask_t::load_safe(
              mfrag,
              mask,
              int64_t(mask_params->M_strides[2]),
              Int<1>{},
              params->qL,
              params->kL,
              row_pos,
              col_pos);

          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemsPerFrag; jj++) {
            if constexpr (is_bool) {
              Stile.frag_at(i, j)[jj] =
                  mfrag[jj] ? Stile.frag_at(i, j)[jj] : neg_inf;
            } else {
              Stile.frag_at(i, j)[jj] += M_LOG2E_F * selem_t(mfrag[jj]);
            }
          }
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load V blocks
    if (!align_K && kb == (params->NK_aligned)) {
      loader_v.load_safe(short2(BD, params->kL_rem));
    } else {
      loader_v.load_unsafe();
    }

    // Do softmax

    // Temp variables
    AccumType new_max[kRowsPT];
    AccumType factor[kRowsPT];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    // Row max
    Stile.template row_reduce<MaxOp>(new_max);

    // exp(Si - rowmax(Si))
    Stile.template row_bin_op<ExpSubOp>(new_max);

    // Factor exp(rowmax(Si) - rowmax(Si-1))
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
    }

    // Save max for next iteration
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = new_max[i];
    }

    // Row Sum
    AccumType sum_score_tmp[kRowsPT] = {0};
    Stile.template row_reduce<SumOp>(sum_score_tmp);

    // Update norm
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    }

    // Update O
    Otile.template row_bin_op<MulOp>(factor);

    // Load V into registers
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TD; id++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          if constexpr (BD == 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }

          const short kk = ik * kFragSize;
          const short dd = id * kFragSize;

          Vtile.template load<T, 1, 1, LDV_tgp, 1>(
              &Vs[Vs_offset + kk * LDV_tgp + dd]);

          if constexpr (BD == 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }

          MMAFrag_acc_t::mma(
              Otile.frag_at(iq, id),
              Stile.frag_at(iq, ik),
              Vtile.frag_at(0, 0),
              Otile.frag_at(iq, id));
        }
      }
    }
  }

  // Emit per-row logsumexp for the chunk-reduce pass. Values stay in the
  // kernel's scaled log2 domain (scores were multiplied by scale * M_LOG2E);
  // the reduce only needs differences so the domain cancels. Rows whose KV
  // loop never ran (causally dead threadgroups in later chunks) get -INF so
  // the reduce skips their 0/0 partials.
  if (use_output_partials && sn == 0) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      const int row_pos = tid.x * BQ + tm + sm + (i * kFragSize);
      if (row_pos < params->qL) {
        const ulong lse_idx =
            (tidl.z * params->H + tidl.y) * ulong(params->qL) + row_pos;
        lse_part[lse_idx] = sum_score[i] <= 0
            ? -INFINITY
            : float(max_score[i]) + fast::log2(float(sum_score[i]));
      }
    }
  }

  // Normalize output
  Otile.template row_bin_op<DivOp>(sum_score);
  threadgroup_barrier(mem_flags::mem_none);

  // Store results
  Odst += (tm + sm) * params->O_strides[2] + sn;

  if (!align_Q && int(tid.x) == (params->NQ_aligned)) {
    auto dst_tile_dims = short2(BD - sn, params->qL_rem - (tm + sm));

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    Otile.template store_safe<T, 1, 1>(
        Odst, params->O_strides[2], dst_tile_dims);
  } else {
    Otile.template store<T, 1, 1>(Odst, params->O_strides[2]);
  }
}

///////////////////////////////////////////////////////////////////////////////
// Chunk reduce kernel (issue #2225)
///////////////////////////////////////////////////////////////////////////////

// Folds the per-chunk normalized partial outputs into the final output with
// logsumexp-weighted averaging (the standard flash-attention chunk
// combination). One thread covers 4 consecutive head-dim elements of one
// (batch, head, row); lse values are in the attention kernel's log2 domain.
template <typename T>
[[kernel]] void attention_chunk_reduce(
    const device T* O_part [[buffer(0)]],
    const device float* lse_part [[buffer(1)]],
    device T* O [[buffer(2)]],
    const constant AttnChunkReduceParams* params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {
  const int d4 = gid.x;
  const int row = gid.y;
  const int bh = gid.z;
  if (row >= params->qL || (4 * d4) >= params->D) {
    return;
  }
  const int b = bh / params->H;
  const int h = bh % params->H;

  const ulong row_off = ulong(bh) * ulong(params->qL) + row;
  const ulong o_off = row_off * ulong(params->D) + 4 * d4;

  float m = -INFINITY;
  for (int c = 0; c < params->C; ++c) {
    m = metal::max(m, lse_part[c * params->lse_chunk_stride + row_off]);
  }

  float4 acc = 0;
  float den = 0;
  for (int c = 0; c < params->C; ++c) {
    const float lse = lse_part[c * params->lse_chunk_stride + row_off];
    if (lse == -INFINITY) {
      continue; // dead chunk: partial is 0/0, weight is exactly 0
    }
    const float w = metal::exp2(lse - m);
    const device T* op = O_part + c * params->o_chunk_stride + o_off;
    acc += w * float4(op[0], op[1], op[2], op[3]);
    den += w;
  }
  const float4 out = den > 0 ? acc / den : float4(0);

  device T* dst = O + b * params->O_strides[0] + h * params->O_strides[1] +
      row * params->O_strides[2] + 4 * d4;
  dst[0] = static_cast<T>(out.x);
  dst[1] = static_cast<T>(out.y);
  dst[2] = static_cast<T>(out.z);
  dst[3] = static_cast<T>(out.w);
}
