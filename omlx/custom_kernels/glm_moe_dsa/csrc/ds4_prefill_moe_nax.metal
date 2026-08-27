// SPDX-License-Identifier: Apache-2.0
//
// Isolated M5/NAX DS4-Flash M=1024 routed-MoE projection. MLX's stock
// gather_qmm_rhs_nax kernel owns global BM64 route tiles and recomputes the
// full tile for every expert segment inside it. This kernel consumes oMLX's
// existing expert-local BM32 plan, so one threadgroup owns exactly one expert
// block while retaining stock NAX dequantization and arithmetic boundaries.

#if __has_include(<MetalPerformancePrimitives/MetalPerformancePrimitives.h>)

// clang-format off: MLX quantized headers require the Steel declarations first.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/fp_quantized_nax.h"
// clang-format on

using namespace metal;

template <typename T, int BM, int BN, int BK, int WM, int WN>
[[kernel]] void ds4_mxfp4_gather_qmm_blocks_nax(
    const device T *x [[buffer(0)]],
    const device uint32_t *weight [[buffer(1)]],
    const device uint8_t *scales [[buffer(2)]],
    const device int32_t *block_meta [[buffer(3)]],
    const device int32_t *block_count [[buffer(4)]],
    device T *output [[buffer(5)]],
    const constant int &max_blocks [[buffer(6)]],
    const constant int &K [[buffer(7)]],
    const constant int &N [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  constexpr int group_size = 32;
  constexpr int bits = 4;
  constexpr int pack_factor = get_pack_factor<8, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack();
  constexpr int BK_padded = BK + 16 / sizeof(bfloat);

  static_assert(BM == 32, "the first DS4 NAX expert block is BM32");
  static_assert(BN == 64 && BK == 64, "the first DS4 NAX tile is 64x64");
  static_assert(WM == 1 && WN == 2, "the first DS4 NAX warp tile is 1x2");

  using loader_w_t = QuantizedBlockLoader<
      bfloat, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits>;

  threadgroup bfloat Ws[BN * BK_padded];

  const int block_id = int(tid.y);
  if (block_id >= max_blocks || block_id >= block_count[0]) {
    return;
  }
  const int row_start = block_meta[block_id * 3 + 0];
  const int expert = block_meta[block_id * 3 + 1];
  const int rows = block_meta[block_id * 3 + 2];
  if (row_start < 0 || expert < 0 || expert >= 256 || rows <= 0 || rows > BM) {
    return;
  }

  const int out_col = int(tid.x) * BN;
  if (out_col >= N) {
    return;
  }

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const size_t weight_expert_stride = size_t(N) * size_t(K_w);
  const size_t scale_expert_stride = size_t(N) * size_t(K_g);
  const device uint8_t *weight_bytes =
      reinterpret_cast<const device uint8_t *>(weight) +
      size_t(expert) * weight_expert_stride + size_t(out_col) * size_t(K_w);
  const device uint8_t *scale_bytes =
      scales + size_t(expert) * scale_expert_stride +
      size_t(out_col) * size_t(K_g);
  const device T *input_block = x + size_t(row_start) * size_t(K);
  device T *output_block =
      output + size_t(row_start) * size_t(N) + size_t(out_col);

  thread loader_w_t loader_w(
      weight_bytes, scale_bytes, K, Ws, simd_group_id, simd_lane_id);

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;
  const short tn = SN * (simd_group_id % WN);

  NAXTile<float, TM, TN> accum;
  accum.clear();
  const device T *input_tile = input_block;

  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_w.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_NO_UNROLL
    for (int kk = 0; kk < BK; kk += SK) {
      NAXTile<T, TM, TK> Atile;
      NAXTile<bfloat, TN, TK> Btile;
      volatile int compiler_barrier;

      if (rows == BM) {
        Atile.load(input_tile + kk, K);
      } else {
        Atile.load_safe(input_tile + kk, K, short2(SK, short(rows)));
      }
      Btile.template load<bfloat, BK_padded, 1>(
          Ws + tn * BK_padded + kk);
      tile_matmad_nax(
          accum,
          Atile,
          metal::bool_constant<false>{},
          Btile,
          metal::bool_constant<true>{});
      (void)compiler_barrier;
    }

    input_tile += BK;
    loader_w.next();
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (rows == BM) {
    accum.store(output_block + tn, N);
  } else {
    accum.store_slice(
        output_block + tn, N, short2(0, 0), short2(SN, short(rows)));
  }
}

template <typename T, int BM, int BN, int BK, int WM, int WN>
[[kernel]] void ds4_mxfp4_gather_qmm_pair_blocks_nax(
    const device T *x [[buffer(0)]],
    const device uint32_t *weight0 [[buffer(1)]],
    const device uint8_t *scales0 [[buffer(2)]],
    const device uint32_t *weight1 [[buffer(3)]],
    const device uint8_t *scales1 [[buffer(4)]],
    const device int32_t *block_meta [[buffer(5)]],
    const device int32_t *block_count [[buffer(6)]],
    device T *output [[buffer(7)]],
    const constant int &max_blocks [[buffer(8)]],
    const constant int &K [[buffer(9)]],
    const constant int &N [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  constexpr int group_size = 32;
  constexpr int bits = 4;
  constexpr int pack_factor = get_pack_factor<8, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack();
  constexpr int BK_padded = BK + 16 / sizeof(bfloat);
  static_assert(BM == 32 && BN == 64 && BK == 64);
  static_assert(WM == 1 && WN == 2);

  using loader_w_t = QuantizedBlockLoader<
      bfloat, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits>;
  threadgroup bfloat Ws0[BN * BK_padded];
  threadgroup bfloat Ws1[BN * BK_padded];

  const int block_id = int(tid.y);
  if (block_id >= max_blocks || block_id >= block_count[0]) {
    return;
  }
  const int row_start = block_meta[block_id * 3 + 0];
  const int expert = block_meta[block_id * 3 + 1];
  const int rows = block_meta[block_id * 3 + 2];
  if (row_start < 0 || expert < 0 || expert >= 256 || rows <= 0 || rows > BM) {
    return;
  }
  const int out_col = int(tid.x) * BN;
  if (out_col >= N) {
    return;
  }

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const size_t weight_expert_stride = size_t(N) * size_t(K_w);
  const size_t scale_expert_stride = size_t(N) * size_t(K_g);
  const size_t weight_offset = size_t(expert) * weight_expert_stride +
      size_t(out_col) * size_t(K_w);
  const size_t scale_offset = size_t(expert) * scale_expert_stride +
      size_t(out_col) * size_t(K_g);
  const device uint8_t *weight_bytes0 =
      reinterpret_cast<const device uint8_t *>(weight0) + weight_offset;
  const device uint8_t *weight_bytes1 =
      reinterpret_cast<const device uint8_t *>(weight1) + weight_offset;
  const device uint8_t *scale_bytes0 = scales0 + scale_offset;
  const device uint8_t *scale_bytes1 = scales1 + scale_offset;
  const device T *input_block = x + size_t(row_start) * size_t(K);
  const size_t output_offset =
      size_t(row_start) * size_t(N) + size_t(out_col);
  device T *output0 = output + output_offset;
  device T *output1 = output + size_t(6144) * size_t(N) + output_offset;

  thread loader_w_t loader0(
      weight_bytes0, scale_bytes0, K, Ws0, simd_group_id, simd_lane_id);
  thread loader_w_t loader1(
      weight_bytes1, scale_bytes1, K, Ws1, simd_group_id, simd_lane_id);

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;
  const short tn = SN * (simd_group_id % WN);
  NAXTile<float, TM, TN> accum0;
  NAXTile<float, TM, TN> accum1;
  accum0.clear();
  accum1.clear();
  const device T *input_tile = input_block;

  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader0.load_unsafe();
    loader1.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_NO_UNROLL
    for (int kk = 0; kk < BK; kk += SK) {
      NAXTile<T, TM, TK> Atile;
      NAXTile<bfloat, TN, TK> Btile0;
      NAXTile<bfloat, TN, TK> Btile1;
      volatile int compiler_barrier;
      if (rows == BM) {
        Atile.load(input_tile + kk, K);
      } else {
        Atile.load_safe(input_tile + kk, K, short2(SK, short(rows)));
      }
      Btile0.template load<bfloat, BK_padded, 1>(Ws0 + tn * BK_padded + kk);
      Btile1.template load<bfloat, BK_padded, 1>(Ws1 + tn * BK_padded + kk);
      tile_matmad_nax(
          accum0, Atile, metal::bool_constant<false>{}, Btile0,
          metal::bool_constant<true>{});
      tile_matmad_nax(
          accum1, Atile, metal::bool_constant<false>{}, Btile1,
          metal::bool_constant<true>{});
      (void)compiler_barrier;
    }
    input_tile += BK;
    loader0.next();
    loader1.next();
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (rows == BM) {
    accum0.store(output0 + tn, N);
    accum1.store(output1 + tn, N);
  } else {
    const short2 extent(SN, short(rows));
    accum0.store_slice(output0 + tn, N, short2(0, 0), extent);
    accum1.store_slice(output1 + tn, N, short2(0, 0), extent);
  }
}

#define instantiate_ds4_mxfp4_blocks_nax(type, bm, bn, bk, wm, wn)          \
  instantiate_kernel(                                                        \
      "ds4_mxfp4_gather_qmm_blocks_nax_" #type "_bm" #bm "_bn" #bn       \
      "_bk" #bk "_wm" #wm "_wn" #wn,                                     \
      ds4_mxfp4_gather_qmm_blocks_nax, type, bm, bn, bk, wm, wn)

instantiate_ds4_mxfp4_blocks_nax(bfloat16_t, 32, 64, 64, 1, 2);

#define instantiate_ds4_mxfp4_pair_blocks_nax(type, bm, bn, bk, wm, wn)     \
  instantiate_kernel(                                                        \
      "ds4_mxfp4_gather_qmm_pair_blocks_nax_" #type "_bm" #bm "_bn" #bn \
      "_bk" #bk "_wm" #wm "_wn" #wn,                                     \
      ds4_mxfp4_gather_qmm_pair_blocks_nax, type, bm, bn, bk, wm, wn)

instantiate_ds4_mxfp4_pair_blocks_nax(bfloat16_t, 32, 64, 64, 1, 2);

#endif
