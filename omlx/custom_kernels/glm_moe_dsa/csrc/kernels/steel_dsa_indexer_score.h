// Copyright © 2026 Apple Inc.

#pragma once

#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"

using namespace mlx::steel;

constant bool do_causal [[function_constant(300)]];
constant bool weights_lh [[function_constant(301)]];
constant bool bucketed_topk_output [[function_constant(302)]];

template <typename T>
METAL_FUNC uint dsa_ordered_key_16(T x) {
  const ushort bits = as_type<ushort>(x);
  return (bits & 0x8000) ? uint((~bits) & 0xffff) : uint(bits | 0x8000);
}

METAL_FUNC uint dsa_ordered_key_16_bits(ushort bits) {
  return (bits & 0x8000) ? uint((~bits) & 0xffff) : uint(bits | 0x8000);
}

template <typename T, typename O, int TOPK, int THREADS>
[[kernel, max_total_threads_per_threadgroup(THREADS)]] void dsa_topk_indices_16bit(
    const device T* scores [[buffer(0)]],
    device O* out [[buffer(1)]],
    const constant DSATopKParams* params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]) {
  if (row >= uint(params->rows)) {
    return;
  }

  threadgroup atomic_uint hist[256];
  threadgroup atomic_uint counters[2];
  threadgroup uint state[4];

  if (tid < 256) {
    atomic_store_explicit(&hist[tid], 0, memory_order_relaxed);
  }
  if (tid < 2) {
    atomic_store_explicit(&counters[tid], 0, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const device T* row_scores = scores + size_t(row) * params->K;
  device O* row_out = out + size_t(row) * TOPK;

  int scan_limit = params->K;
  if (params->causal_valid_prefix) {
    const int q = int(row % uint(params->L));
    const int valid_length =
        metal::min(params->K, metal::max(0, params->K - params->L + q + 1));
    if (valid_length <= TOPK) {
      for (int i = int(tid); i < TOPK; i += THREADS) {
        row_out[i] = O(i < valid_length ? i : 0);
      }
      return;
    }
    scan_limit = valid_length;
  }

  for (int i = int(tid); i < scan_limit; i += THREADS) {
    const uint key = dsa_ordered_key_16(row_scores[i]);
    atomic_fetch_add_explicit(&hist[key >> 8], 1, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    uint greater = 0;
    uint threshold_hi = 0;
    for (int h = 255; h >= 0; --h) {
      const uint count = atomic_load_explicit(&hist[h], memory_order_relaxed);
      if (greater + count >= uint(TOPK)) {
        threshold_hi = uint(h);
        break;
      }
      greater += count;
    }
    state[0] = threshold_hi;
    state[1] = greater;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < 256) {
    atomic_store_explicit(&hist[tid], 0, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint threshold_hi = state[0];
  for (int i = int(tid); i < scan_limit; i += THREADS) {
    const uint key = dsa_ordered_key_16(row_scores[i]);
    if ((key >> 8) == threshold_hi) {
      atomic_fetch_add_explicit(&hist[key & 0xff], 1, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    uint greater = state[1];
    uint threshold_lo = 0;
    for (int l = 255; l >= 0; --l) {
      const uint count = atomic_load_explicit(&hist[l], memory_order_relaxed);
      if (greater + count >= uint(TOPK)) {
        threshold_lo = uint(l);
        break;
      }
      greater += count;
    }
    const uint threshold_key = (threshold_hi << 8) | threshold_lo;
    state[2] = threshold_key;
    state[3] = greater;
    atomic_store_explicit(&counters[0], 0, memory_order_relaxed);
    atomic_store_explicit(&counters[1], greater, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint threshold_key = state[2];
  if (bucketed_topk_output) {
    for (int base = 0; base < scan_limit; base += THREADS) {
      const int i = base + int(tid);
      if (i < scan_limit) {
        const uint key = dsa_ordered_key_16(row_scores[i]);
        if (key > threshold_key) {
          const uint pos =
              atomic_fetch_add_explicit(&counters[0], 1, memory_order_relaxed);
          if (pos < uint(TOPK)) {
            row_out[pos] = O(i);
          }
        } else if (key == threshold_key) {
          const uint pos =
              atomic_fetch_add_explicit(&counters[1], 1, memory_order_relaxed);
          if (pos < uint(TOPK)) {
            row_out[pos] = O(i);
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  } else {
    // Deterministic output pass (replaces the atomic-race append, which made both
    // tie MEMBERSHIP and output ORDER GPU-scheduling-dependent — measured: the
    // selected set changed on ~80% of realistic rows across re-runs, so replicated
    // TP ranks could pick different tie-band keys). Raking scan: each thread owns
    // one contiguous segment of the row (count pass -> one threadgroup-wide
    // exclusive scan of per-thread totals, ~3 barriers total -> emit pass).
    // Strictly-greater entries fill [0, n_greater) and threshold ties fill
    // [n_greater, TOPK) lowest-index-first — membership and order are functions of
    // the scores alone. Costs one extra read of the score row vs the racy append.
    constexpr uint kSimdgroups = THREADS / 32;
    threadgroup uint tg_partials_g[kSimdgroups];
    threadgroup uint tg_partials_t[kSimdgroups];
    const uint simd_id = tid / 32;
    const uint lane_id = tid % 32;
    const int seg = (scan_limit + THREADS - 1) / THREADS;
    const int s0 = int(tid) * seg;
    const int s1 = metal::min(s0 + seg, scan_limit);

    uint local_g = 0;
    uint local_t = 0;
    for (int i = s0; i < s1; ++i) {
      const uint key = dsa_ordered_key_16(row_scores[i]);
      local_g += key > threshold_key ? 1u : 0u;
      local_t += key == threshold_key ? 1u : 0u;
    }

    // Exclusive scan of the 1024 per-thread (greater, tie) totals.
    uint pre_g = metal::simd_prefix_exclusive_sum(local_g);
    uint pre_t = metal::simd_prefix_exclusive_sum(local_t);
    if (lane_id == 31) {
      tg_partials_g[simd_id] = pre_g + local_g;
      tg_partials_t[simd_id] = pre_t + local_t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
      const uint pg = lane_id < kSimdgroups ? tg_partials_g[lane_id] : 0u;
      const uint pt = lane_id < kSimdgroups ? tg_partials_t[lane_id] : 0u;
      const uint sg = metal::simd_prefix_exclusive_sum(pg);
      const uint st = metal::simd_prefix_exclusive_sum(pt);
      if (lane_id < kSimdgroups) {
        tg_partials_g[lane_id] = sg;
        tg_partials_t[lane_id] = st;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint pos_g = tg_partials_g[simd_id] + pre_g;
    uint pos_t = uint(state[3]) + tg_partials_t[simd_id] + pre_t;
    if (local_g > 0 || (local_t > 0 && pos_t < uint(TOPK))) {
      for (int j = s0; j < s1; ++j) {
        const uint key = dsa_ordered_key_16(row_scores[j]);
        if (key > threshold_key) {
          row_out[pos_g++] = O(j);
        } else if (key == threshold_key) {
          if (pos_t < uint(TOPK)) {
            row_out[pos_t++] = O(j);
          }
        }
      }
    }
  }
}

template <typename T, int BM, int BN, int BK, int WM, int WN>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void
dsa_indexer_score(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* W [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant GEMMParams* params [[buffer(4)]],
    const constant int& H [[buffer(5)]],
    const constant int& unused_causal_prefix_topk [[buffer(6)]],
    const constant bool& skip_causal_future_store [[buffer(7)]],
    const constant int& causal_q_offset [[buffer(8)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
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
      true,
      true,
      float>;

  using loader_a_t = typename gemm_kernel::loader_a_t;
  using loader_b_t = typename gemm_kernel::loader_b_t;
  using mma_t = typename gemm_kernel::mma_t;

  const int tid_y = ((tid.y) << params->swizzle_log) +
      ((tid.x) & ((1 << params->swizzle_log) - 1));
  const int tid_x = (tid.x) >> params->swizzle_log;

  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }

  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;

  const int M = params->M;
  const int N = params->N;
  const int D = params->K;
  const int q_offset = causal_q_offset >= 0 ? causal_q_offset : N - M;
  constexpr int THREADS = WM * WN * 32;
  const int thread_idx = int(simd_group_id) * 32 + int(simd_lane_id);

  if (do_causal) {
    const int row_limit = metal::min(c_row + BM, M);
    if (c_col > q_offset + row_limit - 1) {
      if (skip_causal_future_store) {
        return;
      }
      device T* Dst = O + size_t(tid.z) * M * N + size_t(c_row) * params->ldd +
          c_col;
      for (int e = thread_idx; e < BM * BN; e += THREADS) {
        const int row = e / BN;
        const int col = e - row * BN;
        if (c_row + row < M && c_col + col < N) {
          Dst[size_t(row) * params->ldd + col] = static_cast<T>(-INFINITY);
        }
      }
      return;
    }
  }

  if (do_causal && unused_causal_prefix_topk > 0) {
    const int row_limit = metal::min(c_row + BM, M);
    if (q_offset + row_limit <= unused_causal_prefix_topk) {
      return;
    }
  }

  Q += size_t(tid.z) * H * M * D;
  K += size_t(tid.z) * N * D;
  W += size_t(tid.z) * H * M;
  O += size_t(tid.z) * M * N + size_t(c_row) * params->ldd + c_col;

  threadgroup T As[gemm_kernel::tgp_mem_size_a];
  threadgroup T Bs[gemm_kernel::tgp_mem_size_b];

  thread mma_t mma_op(simd_group_id, simd_lane_id);

  float accum[decltype(mma_op.Ctile)::kElemsPerTile];
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < decltype(mma_op.Ctile)::kElemsPerTile; ++i) {
    accum[i] = 0.0f;
  }

  for (int h = 0; h < H; ++h) {
    mma_op.Ctile.clear();

    const device T* A = Q + size_t(h) * M * D + size_t(c_row) * D;
    const device T* B = K + size_t(c_col) * D;

    thread loader_a_t loader_a(A, params->lda, As, simd_group_id, simd_lane_id);
    thread loader_b_t loader_b(B, params->ldb, Bs, simd_group_id, simd_lane_id);

    for (int d = 0; d < params->gemm_k_iterations_aligned; ++d) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_unsafe();
      loader_b.load_unsafe();

      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs);

      loader_a.next();
      loader_b.next();
    }

    threadgroup_barrier(mem_flags::mem_none);

    short ai = 0;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(mma_op.Ctile)::kTileRows; ++i) {
      const int row = c_row + mma_op.sm + i * mma_t::TM_stride;
      const float weight = weights_lh
          ? static_cast<float>(W[size_t(row) * H + h])
          : static_cast<float>(W[size_t(h) * M + row]);
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < decltype(mma_op.Ctile)::kTileCols; ++j) {
        thread const auto& frag = mma_op.Ctile.frag_at(i, j);
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < decltype(mma_op.Ctile)::kElemsPerFrag; ++e) {
          accum[ai++] += max(frag[e], 0.0f) * weight;
        }
      }
    }
  }

  device T* Dst = O + size_t(mma_op.sm) * params->ldd + mma_op.sn;
  short ai = 0;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < decltype(mma_op.Ctile)::kTileRows; ++i) {
    const int row = c_row + mma_op.sm + i * mma_t::TM_stride;
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < decltype(mma_op.Ctile)::kTileCols; ++j) {
      const int col_base = c_col + mma_op.sn + j * mma_t::TN_stride;
      const int out_base =
          (i * decltype(mma_op.Ctile)::kFragRows) * WM * params->ldd +
          (j * decltype(mma_op.Ctile)::kFragCols) * WN;
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < decltype(mma_op.Ctile)::kElemsPerFrag; ++e) {
        const int col = col_base + e;
        const bool future = do_causal && col > q_offset + row;
        const T value = future ? static_cast<T>(-INFINITY)
                               : static_cast<T>(accum[ai]);
        Dst[out_base + e] = value;
        ai++;
      }
    }
  }
}
