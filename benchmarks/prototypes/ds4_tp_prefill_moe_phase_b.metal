// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED PROTOTYPE ONLY. Deliberately absent from CMake and production
// dispatch. A future primitive named deepseek_mxfp4_down_top6_tiled owns one
// FP16 [routes, supertile] scratch allocation and encodes, in order:
//
//   for output_col_base in range(0, 4096, supertile):
//       prototype_ds4_mxfp4_down_tile_f16_bm32_bn32(...)
//       prototype_ds4_moe_top6_reduce_tile_bf16(...)
//
// Dispatch boundaries order the scratch producer and consumer before the next
// tile reuses it. Every global down-output column occurs in exactly one tile,
// so the expert-major schedule reads the down weights once, not once per slot.

#include "mlx/backend/metal/kernels/fp_quantized.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

[[host_name("prototype_ds4_mxfp4_down_tile_f16_bm32_bn32")]]
kernel void prototype_ds4_mxfp4_down_tile_f16_bm32_bn32(
    device const float16_t* activated [[buffer(0)]],
    device const uint32_t* down_w [[buffer(1)]],
    device const uint8_t* down_scales [[buffer(2)]],
    device const int32_t* block_meta [[buffer(3)]],
    device const int32_t* block_count [[buffer(4)]],
    device float16_t* route_tile [[buffer(5)]],
    constant int& max_blocks [[buffer(6)]],
    constant int& routes [[buffer(7)]],
    constant int& hidden [[buffer(8)]],
    constant int& intermediate [[buffer(9)]],
    constant int& output_col_base [[buffer(10)]],
    constant int& supertile [[buffer(11)]],
    uint3 tid [[threadgroup_position_in_grid]],
    ushort simd_group_id [[simdgroup_index_in_threadgroup]],
    ushort simd_lane_id [[thread_index_in_simdgroup]]) {
  (void)routes;
  using T = float16_t;
  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BK = 32;
  constexpr int WM = 1;
  constexpr int WN = 2;
  constexpr int group_size = 32;
  constexpr int bits = 4;
  constexpr int pack_factor = get_pack_factor<8, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack();
  constexpr int BK_padded = BK + 16 / sizeof(T);
  constexpr int threads = WM * WN * SIMD_SIZE;

  using mma_t = mlx::steel::BlockMMA<
      T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, threads>;
  using loader_w_t = QuantizedBlockLoader<
      T, BN, BK, BK_padded, true, threads, group_size, bits>;

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  const int block_id = int(tid.y);
  if (block_id >= max_blocks || block_id >= block_count[0]) {
    return;
  }
  const int tile_col = int(tid.x) * BN;
  const int output_col = output_col_base + tile_col;
  if (tile_col >= supertile || output_col >= hidden) {
    return;
  }

  const int row_start = block_meta[block_id * 3 + 0];
  const int expert = block_meta[block_id * 3 + 1];
  const int rows = block_meta[block_id * 3 + 2];
  if (rows <= 0) {
    return;
  }

  const int packed_intermediate = intermediate * bytes_per_pack / pack_factor;
  const int scale_intermediate = intermediate / group_size;
  const size_t weight_expert_stride = size_t(hidden) * packed_intermediate;
  const size_t scale_expert_stride = size_t(hidden) * scale_intermediate;
  const device T* x_block =
      activated + size_t(row_start) * intermediate;
  const device uint8_t* weight =
      reinterpret_cast<const device uint8_t*>(down_w) +
      size_t(expert) * weight_expert_stride +
      size_t(output_col) * packed_intermediate;
  const device uint8_t* scale =
      down_scales + size_t(expert) * scale_expert_stride +
      size_t(output_col) * scale_intermediate;

  thread loader_x_t loader_x(
      x_block, intermediate, Xs, simd_group_id, simd_lane_id);
  thread loader_w_t loader_w(
      weight, scale, intermediate, Ws, simd_group_id, simd_lane_id);
  thread mma_t mma(simd_group_id, simd_lane_id);
  gemm_loop_unaligned<false, true, true>(
      Xs,
      Ws,
      mma,
      loader_x,
      loader_w,
      intermediate / BK,
      short(rows),
      short(BN),
      short(BK));

  device T* output =
      route_tile + size_t(row_start) * supertile + tile_col;
  mma.store_result_slice(
      output, supertile, short2(0, 0), short2(BN, short(rows)));
}

// Exact current tail for one bounded output tile. The original top-k slot,
// not expert order or threadgroup completion order, owns every BF16 addition.
[[host_name("prototype_ds4_moe_top6_reduce_tile_bf16")]]
kernel void prototype_ds4_moe_top6_reduce_tile_bf16(
    device const float16_t* route_tile [[buffer(0)]],
    device const uint32_t* inverse_order [[buffer(1)]],
    device const float* scores [[buffer(2)]],
    device bfloat16_t* local_output [[buffer(3)]],
    constant int& tokens [[buffer(4)]],
    constant int& hidden [[buffer(5)]],
    constant int& output_col_base [[buffer(6)]],
    constant int& supertile [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int tile_col = int(gid.x);
  const int token = int(gid.y);
  if (tile_col >= supertile || token >= tokens) {
    return;
  }
  const int output_col = output_col_base + tile_col;
  if (output_col >= hidden) {
    return;
  }

  bfloat16_t total = bfloat16_t(0);
  for (int slot = 0; slot < 6; ++slot) {
    const int route_id = token * 6 + slot;
    const uint32_t sorted_row = inverse_order[route_id];
    const bfloat16_t route_value = bfloat16_t(
        route_tile[size_t(sorted_row) * supertile + tile_col]);
    const bfloat16_t route_score = bfloat16_t(scores[route_id]);
    const bfloat16_t weighted = bfloat16_t(route_value * route_score);
    total = bfloat16_t(total + weighted);
  }
  local_output[size_t(token) * hidden + output_col] = total;
}
