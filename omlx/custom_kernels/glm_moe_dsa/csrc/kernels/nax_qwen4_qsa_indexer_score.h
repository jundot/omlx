// SPDX-License-Identifier: Apache-2.0
// Qwen4 QSA indexer scores on the tensor units: per head, a bf16 GEMM with fp32
// accumulate (mlx steel NAX tiles) followed by ReLU and the head-sum in
// registers, then 1/sqrt(D) and the pooled causal sentinel in the epilogue.
// Same contract as qwen4_qsa_indexer_score (steel_dsa_indexer_score.h); the
// fp32 sums differ from it only by accumulation order.
#pragma once

using namespace mlx::steel;

template <typename T, int BM, int BN, int BK, int WM, int WN>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void
qwen4_qsa_nax_indexer_score(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    device float* O [[buffer(2)]],
    const constant GEMMParams* params [[buffer(3)]],
    const constant int& mask_ratio [[buffer(4)]],
    const constant int& mask_q_offset [[buffer(5)]],
    const constant float& score_divisor [[buffer(6)]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  constexpr int H = 4;
  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  static_assert(SM % 16 == 0 && SN % 16 == 0 && BK % SK == 0, "NAX tiles are 16 wide");

  const int tid_x = int(tid.x);
  const int tid_y = int(tid.y);
  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }
  const int M = params->M;
  const int N = params->N;
  const int D = params->K;
  Q += size_t(tid.z) * H * M * D;
  K += size_t(tid.z) * N * D;
  O += size_t(tid.z) * M * N;

  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);
  const int row0 = tid_y * BM + tm;
  const int col0 = tid_x * BN + tn;
  const short sgp_sm = short(metal::min(int(SM), M - row0));
  const short sgp_sn = short(metal::min(int(SN), N - col0));
  const bool aligned_m = sgp_sm == SM;
  const bool aligned_n = sgp_sn == SN;

  using Tile = NAXTile<float, TM, TN>;
  const device T* Bp = K + size_t(metal::max(col0, 0)) * params->ldb;
  float acc[Tile::kElemsPerTile];
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < Tile::kElemsPerTile; ++i) {
    acc[i] = 0.0f;
  }

  STEEL_PRAGMA_UNROLL
  for (short h = 0; h < H; ++h) {
    const device T* Ap = Q + size_t(h) * M * D + size_t(metal::max(row0, 0)) * params->lda;
    Tile Dtile;
    dispatch_bool(aligned_m, [&](auto kAlignedM) {
      dispatch_bool(aligned_n, [&](auto kAlignedN) {
        Dtile = gemm_loop<
            T, SM, SN, SK, BK, false, true, kAlignedM.value, kAlignedN.value, true, float>(
            Ap, Bp, params->lda, params->ldb, params->K,
            params->gemm_k_iterations_aligned, sgp_sm, sgp_sn);
      });
    });
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < Tile::kNumFrags; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < Tile::kElemsPerFrag; ++e) {
        acc[i * Tile::kElemsPerFrag + e] += metal::max(Dtile.val_frags[i][e], 0.0f);
      }
    }
  }

  if (sgp_sm <= 0 || sgp_sn <= 0) {
    return;
  }
  const float sentinel = as_type<float>(uint(0xFF7FFFFF));
  STEEL_PRAGMA_UNROLL
  for (short mm = 0; mm < TM; ++mm) {
    STEEL_PRAGMA_UNROLL
    for (short nn = 0; nn < TN; ++nn) {
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < Tile::kElemsPerFrag; ++e) {
        const short2 c = Tile::NAXFrag_t::get_coord(e);
        const int row = row0 + mm * 16 + c.y;
        const int col = col0 + nn * 16 + c.x;
        if (row < M && col < N) {
          const bool masked = col >= (mask_q_offset + row + 1) / mask_ratio;
          O[size_t(row) * params->ldd + col] =
              masked ? sentinel : acc[(mm * TN + nn) * Tile::kElemsPerFrag + e] / score_divisor;
        }
      }
    }
  }
}
