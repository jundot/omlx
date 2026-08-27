// Copyright © 2026 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

// Optional M5/NAX DS4-Flash prefill indexer-score kernel.
//
// This is intentionally a separate macOS-26.2 metallib: the
// MetalPerformancePrimitives tensor API is unavailable at the classic
// extension deployment target. The C++ primitive loads it only for the exact
// BF16 DS4 domain and falls through to the existing Steel kernel if the
// library or pipeline is unavailable.

#if __has_include(<metal_tensor>) &&                                      \
    __has_include(<MetalPerformancePrimitives/MetalPerformancePrimitives.h>)

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;
using namespace mpp::tensor_ops;

struct OMLXNAXDSAScoreParams {
  int M;
  int N;
  int mask_ratio;
  int mask_q_offset;
};

// Ported from ds4-metal's kernel_dsv4_indexer_scores_nax tile, adapted to
// oMLX's native BF16 [B,H,M,D] / [B,1,N,D] / [B,M,H] ABI and BF16 output.
// The NAX matmul computes two heads at a time over four 32-wide K slices;
// relu, learned head weighting, and head accumulation remain in head order.
// Top-k is intentionally not fused: the existing deterministic radix kernel
// consumes this score sheet unchanged, preserving cutoff-tie membership and
// temporal output order.
[[kernel, max_total_threads_per_threadgroup(128)]] void
nax_dsa_indexer_score_bfloat16_h64_d128_ratio4(
    const device bfloat16_t* Q [[buffer(0)]],
    const device bfloat16_t* K [[buffer(1)]],
    const device bfloat16_t* W [[buffer(2)]],
    device bfloat16_t* O [[buffer(3)]],
    const constant OMLXNAXDSAScoreParams& params [[buffer(4)]],
    threadgroup bfloat16_t* shared [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
  constexpr uint TM = 16;
  constexpr uint TQ = 32; // two heads x 16 query rows
  constexpr uint TN = 32;
  constexpr uint NK = 32;
  constexpr uint D = 128;
  constexpr uint H = 64;
  constexpr uint NUM_THREADS = 128;

  const uint M = uint(params.M);
  const uint N = uint(params.N);
  const uint batch = tgpig.z;
  const uint c0 = tgpig.x * TN;
  const uint m0 = tgpig.y * TM;

  threadgroup bfloat16_t* qtg = shared;          // 2 x [TQ][NK]
  threadgroup bfloat16_t* ktg = qtg + 2*TQ*NK;  // [TN][D]
  threadgroup float* dot =
      reinterpret_cast<threadgroup float*>(ktg + TN*D); // [TQ][TN], col-major

  const device bfloat16_t* q_batch = Q + size_t(batch) * H * M * D;
  const device bfloat16_t* k_batch = K + size_t(batch) * N * D;
  const device bfloat16_t* w_batch = W + size_t(batch) * M * H;
  device bfloat16_t* o_batch = O + size_t(batch) * M * N;

  // A whole N tile can be known-future for every row in the M tile. Write the
  // exact BF16 finfo.min sentinel without staging Q/K or invoking TensorOps.
  const uint last_row = min(m0 + TM, M);
  const uint max_visible = last_row > m0
      ? min(uint((params.mask_q_offset + int(last_row)) / params.mask_ratio), N)
      : 0u;
  const bfloat16_t sentinel = as_type<bfloat16_t>(ushort(0xFF7F));
  if (c0 >= max_visible) {
    for (uint i = tid; i < TM*TN; i += NUM_THREADS) {
      const uint row = m0 + i / TN;
      const uint col = c0 + i % TN;
      if (row < M && col < N) {
        o_batch[size_t(row) * N + col] = sentinel;
      }
    }
    return;
  }

  // One pooled row per four threads; each thread copies 32 consecutive BF16
  // elements. Boundary columns are zero-filled and never stored.
  {
    const uint nc = tid / 4;
    const uint col = c0 + nc;
    const uint d0 = (tid % 4) * 32;
    const device bfloat16_t* krow = col < N ? k_batch + size_t(col) * D : nullptr;
    #pragma unroll
    for (uint j = 0; j < 8; ++j) {
      vec<bfloat16_t, 4> value = vec<bfloat16_t, 4>(bfloat16_t(0.0f));
      if (krow) {
        value = *reinterpret_cast<const device vec<bfloat16_t, 4>*>(
            krow + d0 + 4*j);
      }
      *reinterpret_cast<threadgroup vec<bfloat16_t, 4>*>(
          ktg + nc*D + d0 + 4*j) = value;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float acc[4];
  #pragma unroll
  for (uint j = 0; j < 4; ++j) {
    acc[j] = 0.0f;
  }

  auto tq0 = tensor(qtg, dextents<int32_t, 2>(NK, TQ));
  auto tq1 = tensor(qtg + TQ*NK, dextents<int32_t, 2>(NK, TQ));
  auto tk = tensor(ktg, dextents<int32_t, 2>(D, TN));
  auto td = tensor(dot, dextents<int32_t, 2>(TQ, TN), array<int, 2>({1, TQ}));

  matmul2d<
      matmul2d_descriptor(
          TN,
          TQ,
          NK,
          false,
          true,
          false,
          matmul2d_descriptor::mode::multiply_accumulate),
      execution_simdgroups<4>> mm;

  // One staged Q row per four threads. Row r is (head-in-pair, query-row),
  // with each thread copying eight consecutive BF16 values.
  const uint q_r = tid / 4;
  const uint q_k = (tid % 4) * 8;
  const uint q_head_in_pair = q_r / TM;
  const uint q_local_row = q_r % TM;
  const uint q_row = m0 + q_local_row;

  auto stage_q = [&](uint head0, uint k0, threadgroup bfloat16_t* dst) {
    const uint head = head0 + q_head_in_pair;
    vec<bfloat16_t, 4> v0 = vec<bfloat16_t, 4>(bfloat16_t(0.0f));
    vec<bfloat16_t, 4> v1 = vec<bfloat16_t, 4>(bfloat16_t(0.0f));
    if (q_row < M && head < H) {
      const device bfloat16_t* src =
          q_batch + size_t(head) * M * D + size_t(q_row) * D + k0 + q_k;
      v0 = *reinterpret_cast<const device vec<bfloat16_t, 4>*>(src);
      v1 = *reinterpret_cast<const device vec<bfloat16_t, 4>*>(src + 4);
    }
    *reinterpret_cast<threadgroup vec<bfloat16_t, 4>*>(dst + q_r*NK + q_k) = v0;
    *reinterpret_cast<threadgroup vec<bfloat16_t, 4>*>(dst + q_r*NK + q_k + 4) = v1;
  };

  for (uint head0 = 0; head0 < H; head0 += 2) {
    auto ct = mm.template get_destination_cooperative_tensor<
        decltype(tk), decltype(tq0), float>();
    #pragma unroll
    for (uint16_t i = 0; i < ct.get_capacity(); ++i) {
      if (ct.is_valid_element(i)) {
        ct[i] = 0.0f;
      }
    }

    stage_q(head0, 0, qtg);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint qsel = 0;
    #pragma unroll
    for (uint ki = 0; ki < D/NK; ++ki) {
      auto mk = tk.slice(ki*NK, 0);
      auto mq = (qsel ? tq1 : tq0).slice(0, 0);
      mm.run(mk, mq, ct);
      if (ki + 1 < D/NK) {
        qsel ^= 1u;
        stage_q(head0, (ki + 1)*NK, qsel ? qtg + TQ*NK : qtg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }

    ct.store(td);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma unroll
    for (uint j = 0; j < 4; ++j) {
      const uint linear = uint(tid) + j*NUM_THREADS;
      if (linear < TM*TN) {
        const uint local_row = linear / TN;
        const uint local_col = linear % TN;
        const uint row = m0 + local_row;
        if (row < M) {
          const float w0 = float(w_batch[size_t(row) * H + head0]);
          acc[j] += max(dot[local_col*TQ + local_row], 0.0f) * w0;
          const float w1 = float(w_batch[size_t(row) * H + head0 + 1]);
          acc[j] += max(dot[local_col*TQ + TM + local_row], 0.0f) * w1;
        }
      }
    }
    // The next pair stages Q in buffers disjoint from `dot`; its barrier also
    // separates the following cooperative-tensor store from these reads.
  }

  #pragma unroll
  for (uint j = 0; j < 4; ++j) {
    const uint linear = uint(tid) + j*NUM_THREADS;
    if (linear >= TM*TN) {
      continue;
    }
    const uint row = m0 + linear / TN;
    const uint col = c0 + linear % TN;
    if (row < M && col < N) {
      const uint visible = min(
          uint((params.mask_q_offset + int(row) + 1) / params.mask_ratio), N);
      o_batch[size_t(row) * N + col] =
          col < visible ? bfloat16_t(acc[j]) : sentinel;
    }
  }
}

#endif
