// Copyright © 2026 Apple Inc.

#include <metal_atomic>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"

using namespace metal;
using namespace mlx::steel;

METAL_FUNC uint minimax_msa_ordered_key(float x) {
  uint bits = as_type<uint>(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

METAL_FUNC float minimax_msa_ordered_value(uint key) {
  uint bits = (key & 0x80000000u) ? (key & 0x7fffffffu) : ~key;
  return as_type<float>(bits);
}

template <typename T, int BM, int BN, int BK, int WM, int WN>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void
minimax_msa_block_scores(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    device float* O [[buffer(2)]],
    const constant GEMMParams* params [[buffer(3)]],
    const constant int& H [[buffer(4)]],
    const constant int& q_start [[buffer(5)]],
    const constant int& block_size [[buffer(6)]],
    const constant float& scale [[buffer(7)]],
    const constant int& num_blocks [[buffer(8)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  (void)block_size;
  (void)lid;

  using gemm_kernel = GEMMKernel<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      false,
      true,
      false,
      true,
      float>;

  using loader_a_t = typename gemm_kernel::loader_a_t;
  using loader_b_t = typename gemm_kernel::loader_b_t;
  using mma_t = typename gemm_kernel::mma_t;

  const int tid_y = int(tid.y);
  const int tid_x = int(tid.x);
  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }

  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const int M = params->M;
  const int N = params->N;
  const int D = params->K;
  const int z = int(tid.z);
  const int b = z / H;
  const int h = z - b * H;
  const int thread_idx = int(simd_group_id) * 32 + int(simd_lane_id);
  constexpr int THREADS = WM * WN * 32;

  threadgroup atomic_uint row_max[BM];
  if (thread_idx < BM) {
    atomic_store_explicit(
        &row_max[thread_idx],
        minimax_msa_ordered_key(-INFINITY),
        memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  Q += (size_t(b) * H + h) * M * D + size_t(c_row) * D;
  K += size_t(b) * N * D + size_t(c_col) * D;

  threadgroup T As[gemm_kernel::tgp_mem_size_a];
  threadgroup T Bs[gemm_kernel::tgp_mem_size_b];

  thread loader_a_t loader_a(Q, params->lda, As, simd_group_id, simd_lane_id);
  thread loader_b_t loader_b(K, params->ldb, Bs, simd_group_id, simd_lane_id);
  thread mma_t mma_op(simd_group_id, simd_lane_id);

  const short tgp_bm = short(min(BM, M - c_row));
  const short tgp_bn = short(min(BN, N - c_col));
  const short leftover_bk = short(D - params->gemm_k_iterations_aligned * BK);

  if (tgp_bm == BM && tgp_bn == BN) {
    gemm_kernel::template gemm_loop<true, true, true>(
        As,
        Bs,
        params->gemm_k_iterations_aligned,
        loader_a,
        loader_b,
        mma_op,
        tgp_bm,
        tgp_bn,
        leftover_bk);
  } else if (tgp_bn == BN) {
    gemm_kernel::template gemm_loop<false, true, true>(
        As,
        Bs,
        params->gemm_k_iterations_aligned,
        loader_a,
        loader_b,
        mma_op,
        tgp_bm,
        tgp_bn,
        leftover_bk);
  } else if (tgp_bm == BM) {
    gemm_kernel::template gemm_loop<true, false, true>(
        As,
        Bs,
        params->gemm_k_iterations_aligned,
        loader_a,
        loader_b,
        mma_op,
        tgp_bm,
        tgp_bn,
        leftover_bk);
  } else {
    gemm_kernel::template gemm_loop<false, false, true>(
        As,
        Bs,
        params->gemm_k_iterations_aligned,
        loader_a,
        loader_b,
        mma_op,
        tgp_bm,
        tgp_bn,
        leftover_bk);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < decltype(mma_op.Ctile)::kTileRows; ++i) {
    const int row = c_row + mma_op.sm + i * mma_t::TM_stride;
    const int row_local = mma_op.sm + i * mma_t::TM_stride;
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < decltype(mma_op.Ctile)::kTileCols; ++j) {
      const int col_base = c_col + mma_op.sn + j * mma_t::TN_stride;
      thread const auto& frag = mma_op.Ctile.frag_at(i, j);
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < decltype(mma_op.Ctile)::kElemsPerFrag; ++e) {
        const int col = col_base + e;
        const int q_abs = q_start + row;
        if (row < M && col < N && col <= q_abs) {
          float score = float(frag[e]) * scale;
          if (score == score) {
            atomic_fetch_max_explicit(
                &row_max[row_local],
                minimax_msa_ordered_key(score),
                memory_order_relaxed);
          }
        }
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int row_local = thread_idx; row_local < BM; row_local += THREADS) {
    const int row = c_row + row_local;
    if (row < M) {
      const uint key =
          atomic_load_explicit(&row_max[row_local], memory_order_relaxed);
      O[((size_t(b) * H + h) * M + row) * num_blocks + tid_x] =
          minimax_msa_ordered_value(key);
    }
  }
}

template <int TOPK, int THREADS>
[[kernel, max_total_threads_per_threadgroup(THREADS)]] void
minimax_msa_topk_select(
    const device float* block_scores [[buffer(0)]],
    device int* topk_idx [[buffer(1)]],
    const constant int& rows [[buffer(2)]],
    const constant int& H [[buffer(3)]],
    const constant int& L [[buffer(4)]],
    const constant int& num_blocks [[buffer(5)]],
    const constant int& q_start [[buffer(6)]],
    const constant int& block_size [[buffer(7)]],
    const constant int& init_blocks [[buffer(8)]],
    const constant int& local_blocks [[buffer(9)]],
    uint tid [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]) {
  if (row >= uint(rows)) {
    return;
  }

  const int q_idx = int(row % uint(L));
  const int h = int((row / uint(L)) % uint(H));
  const int b = int(row / uint(L * H));
  const int q_abs = q_start + q_idx;
  const int cur_block = q_abs / block_size;
  int local_start = cur_block - local_blocks + 1;
  if (local_start < 0) {
    local_start = 0;
  }

  threadgroup float scores_s[THREADS];
  threadgroup int indices_s[THREADS];
  threadgroup int selected_s[TOPK];

  if (tid < TOPK) {
    selected_s[tid] = num_blocks;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int slot = 0; slot < TOPK; ++slot) {
    float best_score = -INFINITY;
    int best_idx = num_blocks;
    for (int block = int(tid); block < num_blocks; block += THREADS) {
      bool valid = block <= cur_block;
      bool already = false;
      for (int prev = 0; prev < slot; ++prev) {
        already = already || (selected_s[prev] == block);
      }
      if (!valid || already) {
        continue;
      }

      const size_t offset = ((size_t(b) * H + h) * L + q_idx) * num_blocks +
          block;
      float score = block_scores[offset];
      if (score != score) {
        score = -INFINITY;
      }
      if (init_blocks > 0 && block < init_blocks) {
        score = 1.0e30f;
      }
      if (
          local_blocks > 0 && block >= local_start && block <= cur_block) {
        score = 1.0e29f;
      }
      if (score > best_score || (score == best_score && block < best_idx)) {
        best_score = score;
        best_idx = block;
      }
    }

    scores_s[tid] = best_score;
    indices_s[tid] = best_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = THREADS / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        float other_score = scores_s[tid + stride];
        int other_idx = indices_s[tid + stride];
        float cur_score = scores_s[tid];
        int cur_idx = indices_s[tid];
        if (
            other_score > cur_score ||
            (other_score == cur_score && other_idx < cur_idx)) {
          scores_s[tid] = other_score;
          indices_s[tid] = other_idx;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
      selected_s[slot] = indices_s[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0) {
    for (int i = 0; i < TOPK; ++i) {
      for (int j = i + 1; j < TOPK; ++j) {
        if (selected_s[j] < selected_s[i]) {
          int tmp = selected_s[i];
          selected_s[i] = selected_s[j];
          selected_s[j] = tmp;
        }
      }
    }

    const int out_base = int(row) * TOPK;
    for (int i = 0; i < TOPK; ++i) {
      int idx = selected_s[i];
      topk_idx[out_base + i] = idx < num_blocks ? idx : -1;
    }
  }
}

#define instantiate_minimax_msa_block_scores(iname, itype, bm, bn, bk, wm, wn) \
  instantiate_kernel(                                                          \
      "minimax_msa_block_scores_" #iname "_bm" #bm "_bn" #bn "_bk" #bk        \
      "_wm" #wm "_wn" #wn,                                                     \
      minimax_msa_block_scores,                                                \
      itype,                                                                   \
      bm,                                                                      \
      bn,                                                                      \
      bk,                                                                      \
      wm,                                                                      \
      wn)

#define instantiate_minimax_msa_topk(topk, threads)        \
  instantiate_kernel(                                      \
      "minimax_msa_topk_select_topk" #topk "_t" #threads, \
      minimax_msa_topk_select,                             \
      topk,                                                \
      threads)

instantiate_minimax_msa_block_scores(float32, float, 64, 128, 16, 2, 2);
instantiate_minimax_msa_block_scores(float16, half, 64, 128, 16, 2, 2);
instantiate_minimax_msa_block_scores(bfloat16, bfloat16_t, 64, 128, 16, 2, 2);

instantiate_minimax_msa_topk(16, 256);
