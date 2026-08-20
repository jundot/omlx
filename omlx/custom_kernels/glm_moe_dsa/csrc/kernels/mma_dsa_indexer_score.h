// v25 zero-per-head-barrier DSA indexer score kernel (M2-tuned, from-scratch
// simdgroup_matrix GEMM — not Steel). Non-causal, weights-LH, bf16, H=64,
// D=128 only; every other configuration must be routed to the Steel kernel
// by the host gate.
//
// HOST-SIDE HEADER: the kernel ships as a Metal source string compiled at
// runtime through mlx's Device::get_library(name, builder) — i.e. by the
// macOS runtime Metal compiler, NOT the Xcode CLI toolchain that builds
// omlx_glm_kernels.metallib. This is deliberate and measured: the CLI
// toolchain (metalfe-32023.921, Xcode 27 beta) generates code for this
// kernel that runs 3.4 %-points of peak slower (57.2ms vs 55.0ms at
// M=4096/N=16384) than the runtime compiler that produced every accepted
// benchmark. The one-time runtime compile (~100-300ms, cached by library
// name and by the OS shader cache) is paid at first prefill.
//
// Structure (why it beats Steel by ~1.37x on M2 Ultra — measured story in
// PR #2802):
//   1. K tile resident in threadgroup memory TRANSPOSED [k][n], PAD=0
//      (16KB = exactly half the M2 per-core tgp, preserving 2 resident
//      threadgroups). Loaded once; the ONLY threadgroup_barrier in the
//      kernel (Steel: 3 barriers per K-slice per head). The transposed
//      layout makes each B-fragment's two elements contiguous: one vec2
//      tgp load per fragment.
//   2. A(Q)-fragments load directly from device per MMA step — no staging
//      (proven latency-hidden; the staging round-trip was pure overhead).
//   3. multiply-init: the first MMA of each head writes the accumulator
//      tile directly (0 + a*b == a*b bit-exactly through the relu*w
//      epilogue), deleting the per-head tile clears.
//   4. Per-head epilogue relu*W with fp32 accumulation in head order —
//      byte-identical contract to Steel's dsa_indexer_score.
//
// BOUNDARY split: the kernel template is instantiated twice. BOUNDARY=false
// processes only fully-interior tiles with the unmodified hot loop;
// BOUNDARY=true processes only partial edge tiles with address-clamped
// loads. Clamping is bit-exact by row/column isolation (a clamped duplicate
// value only reaches output rows/cols the store guard drops). The split is
// load-bearing: a runtime branch or clamp in the shared hot loop measurably
// costs 3.4-10.5 %-points of peak because the register allocator provisions
// for the union of both paths.
//
// M, N, mask_ratio, mask_q_offset are RUNTIME params. Never make them
// compile-time: N grows and mask_q_offset changes every prefill chunk.

#pragma once

namespace omlx::glm_kernels {

struct OMLXMMADSAScoreParamsHost {
  int M;
  int N;
  int mask_ratio;
  int mask_q_offset;
};

inline constexpr const char* kMMADSAScoreKernelSource = R"MMADSA(
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

struct OMLXMMADSAScoreParams {
  int M;
  int N;
  int mask_ratio;
  int mask_q_offset;
};

// Lane -> (fn, fm) fragment coordinates for simdgroup_matrix<T, 8, 8>
// thread_elements() on AGX (matches the layout mlx::steel derives).
inline short2 mma_dsa_sg_coord(ushort lane) {
  const short qid = short(lane) / 4;
  const short fm = (qid & 4) + ((short(lane) / 2) % 4);
  const short fn = (qid & 2) * 2 + (short(lane) % 2) * 2;
  return short2(fn, fm);
}

// finfo(T).min bit pattern — the pooled-mask sentinel the call site's
// mx.where pass historically wrote (same values as steel_dsa_indexer_score).
template <typename T>
inline T mma_dsa_finfo_min() {
  return as_type<T>(
      ushort(metal::is_same<T, bfloat>::value ? 0xFF7F : 0xFBFF));
}

template <typename T, int BM, int BN, int WM, int WN, int H, int D,
          bool BOUNDARY>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void
mma_dsa_indexer_score(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* W [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant OMLXMMADSAScoreParams& params [[buffer(4)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tgpig [[threadgroup_position_in_grid]]) {
  constexpr int THREADS = WM * WN * 32;
  const int M = params.M;
  const int N = params.N;
  const uint tid = simd_group_id * 32 + simd_lane_id;
  const short2 coord = mma_dsa_sg_coord(ushort(simd_lane_id));
  const short fm = coord.y;
  const short fn = coord.x;

  constexpr int swizzle_log = 2;
  const int tiles_m = (M + BM - 1) / BM;
  const int tiles_n = (N + BN - 1) / BN;
  const int tid_y = (int(tgpig.y) << swizzle_log) +
      (int(tgpig.x) & ((1 << swizzle_log) - 1));
  const int tid_x = int(tgpig.x) >> swizzle_log;
  if (tid_x >= tiles_n || tid_y >= tiles_m) {
    return;
  }

  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  // Interior/boundary complement split. Uniform branch, resolved before
  // any load — zero hot-loop cost.
  const bool partial = (c_row + BM > M) || (c_col + BN > N);
  if (partial != BOUNDARY) {
    return;
  }

  const int batch = int(tgpig.z);
  const device T* Q_base = Q + size_t(batch) * H * M * D;
  const device T* K_base = K + size_t(batch) * N * D;
  const device T* W_base = W + size_t(batch) * M * H;
  device T* O_base = O + size_t(batch) * size_t(M) * N;

  constexpr int TM_STRIDE = 8 * WM;
  constexpr int TN_STRIDE = 8 * WN;
  constexpr int kTileRows = BM / TM_STRIDE;
  constexpr int kTileCols = BN / TN_STRIDE;
  constexpr int kSubTiles = kTileRows * kTileCols;
  const int sm = 8 * (int(simd_group_id) / WN) + fm;
  const int sn = 8 * (int(simd_group_id) % WN) + fn;

  float accum[kSubTiles * 2];
  #pragma unroll
  for (int i = 0; i < kSubTiles * 2; ++i) {
    accum[i] = 0.0f;
  }

  simdgroup_matrix<float, 8, 8> c_tiles[kSubTiles];
  constexpr int KSTEPS = D / 8;

  // ── K resident in tgp, transposed [k][n], loaded once ────────────────
  constexpr int BT_STRIDE = BN; // PAD=0: 16KB keeps 2 TGs per core
  threadgroup T buf_bt[D * BT_STRIDE];
  {
    const device T* B_tile = K_base + size_t(c_col) * D;
    constexpr int TOTAL = BN * D;
    constexpr int PER_THREAD = (TOTAL + THREADS - 1) / THREADS;
    #pragma unroll
    for (int i = 0; i < PER_THREAD; ++i) {
      int idx = int(tid) * PER_THREAD + i;
      if (idx < TOTAL) {
        int n = idx / D;
        int k = idx % D;
        int ns = BOUNDARY ? metal::min(n, N - c_col - 1) : n;
        buf_bt[size_t(k) * BT_STRIDE + n] = B_tile[size_t(ns) * D + k];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup); // the only barrier

  // ── Head loop ────────────────────────────────────────────────────────
  for (int h = 0; h < H; ++h) {
    const device T* A_tile = Q_base + size_t(h) * M * D + size_t(c_row) * D;

    #pragma unroll
    for (int kk = 0; kk < KSTEPS; ++kk) {
      const int kidx = kk * 8 + fn;
      simdgroup_matrix<T, 8, 8> a_frags[kTileRows];
      #pragma unroll
      for (int ti = 0; ti < kTileRows; ++ti) {
        const int a_row = sm + ti * TM_STRIDE;
        const int ar = BOUNDARY ? metal::min(a_row, M - c_row - 1) : a_row;
        const device T* ap = A_tile + size_t(ar) * D + kidx;
        vec<T, 2> v(ap[0], ap[1]);
        reinterpret_cast<thread vec<T, 2>&>(a_frags[ti].thread_elements()) =
            v;
      }
      simdgroup_matrix<T, 8, 8> b_frags[kTileCols];
      {
        const int krow = kk * 8 + fm;
        const threadgroup T* brow = buf_bt + size_t(krow) * BT_STRIDE;
        #pragma unroll
        for (int tj = 0; tj < kTileCols; ++tj) {
          const int b_col = sn + tj * TN_STRIDE;
          vec<T, 2> v =
              *reinterpret_cast<const threadgroup vec<T, 2>*>(brow + b_col);
          reinterpret_cast<thread vec<T, 2>&>(b_frags[tj].thread_elements()) =
              v;
        }
      }
      if (kk == 0) {
        // multiply-init: 0 + a*b == a*b bit-exactly through relu*w
        #pragma unroll
        for (int ti = 0; ti < kTileRows; ++ti) {
          #pragma unroll
          for (int tj = 0; tj < kTileCols; ++tj) {
            simdgroup_multiply(
                c_tiles[ti * kTileCols + tj], a_frags[ti], b_frags[tj]);
          }
        }
      } else {
        #pragma unroll
        for (int ti = 0; ti < kTileRows; ++ti) {
          #pragma unroll
          for (int tj = 0; tj < kTileCols; ++tj) {
            simdgroup_multiply_accumulate(
                c_tiles[ti * kTileCols + tj],
                a_frags[ti],
                b_frags[tj],
                c_tiles[ti * kTileCols + tj]);
          }
        }
      }
    }

    // ── Per-head epilogue: relu + weight + FP32 accumulate ─────────────
    #pragma unroll
    for (int ti = 0; ti < kTileRows; ++ti) {
      const int row = c_row + sm + ti * TM_STRIDE;
      const float weight =
          row < M ? static_cast<float>(W_base[size_t(row) * H + h]) : 0.0f;
      #pragma unroll
      for (int tj = 0; tj < kTileCols; ++tj) {
        vec<float, 2> cv = reinterpret_cast<thread vec<float, 2>&>(
            c_tiles[ti * kTileCols + tj].thread_elements());
        int ai = (ti * kTileCols + tj) * 2;
        accum[ai] += metal::max(cv[0], 0.0f) * weight;
        accum[ai + 1] += metal::max(cv[1], 0.0f) * weight;
      }
    }
  }

  // ── Store with pooled-ratio masking (identical to Steel's epilogue) ──
  const T pooled_sentinel = mma_dsa_finfo_min<T>();
  #pragma unroll
  for (int ti = 0; ti < kTileRows; ++ti) {
    const int row = c_row + sm + ti * TM_STRIDE;
    #pragma unroll
    for (int tj = 0; tj < kTileCols; ++tj) {
      const int col_base = c_col + sn + tj * TN_STRIDE;
      int ai = (ti * kTileCols + tj) * 2;
      #pragma unroll
      for (short e = 0; e < 2; ++e) {
        const int col = col_base + e;
        const bool pooled_masked = params.mask_ratio > 0 &&
            col >= (params.mask_q_offset + row + 1) / params.mask_ratio;
        const T value = pooled_masked ? pooled_sentinel
                                      : static_cast<T>(accum[ai + e]);
        if (row < M && col < N) {
          O_base[size_t(row) * N + col] = value;
        }
      }
    }
  }
}

template [[host_name("mma_dsa_indexer_score_bfloat16_interior")]] [[kernel]]
decltype(mma_dsa_indexer_score<bfloat, 64, 64, 2, 2, 64, 128, false>)
mma_dsa_indexer_score<bfloat, 64, 64, 2, 2, 64, 128, false>;
template [[host_name("mma_dsa_indexer_score_bfloat16_boundary")]] [[kernel]]
decltype(mma_dsa_indexer_score<bfloat, 64, 64, 2, 2, 64, 128, true>)
mma_dsa_indexer_score<bfloat, 64, 64, 2, 2, 64, 128, true>;
)MMADSA";

} // namespace omlx::glm_kernels
