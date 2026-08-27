// SPDX-License-Identifier: Apache-2.0
//
// Isolated DS4-Flash phase-A prefill experiment. This kernel is callable only
// through a fixed-shape native symbol; no model/runtime dispatch references it.

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/fp_quantized.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/unary_ops.h"

using namespace metal;

// One threadgroup computes matching BM32xBN32 up and gate tiles. X is staged
// once per BK32 step while the two FP32 accumulator sets remain independent
// and keep the current Steel K order. Both accumulators narrow to FP16 before
// the exact LimitedSwiGLU operation sequence, matching the existing projection
// store boundary without materializing a [R,1,2I] pair tensor.
template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN>
[[kernel]] void deepseek_mxfp4_gather_qmm_pair_swiglu_blocks(
    const device T* x [[buffer(0)]],
    const device uint32_t* up_w [[buffer(1)]],
    const device uint8_t* up_scales [[buffer(2)]],
    const device uint32_t* gate_w [[buffer(3)]],
    const device uint8_t* gate_scales [[buffer(4)]],
    const device int32_t* block_meta [[buffer(5)]],
    const device int32_t* block_count [[buffer(6)]],
    device T* activated [[buffer(7)]],
    const constant int& max_blocks [[buffer(8)]],
    const constant int& routes [[buffer(9)]],
    const constant int& intermediate [[buffer(10)]],
    const constant int& hidden [[buffer(11)]],
    const constant float& activation_limit [[buffer(12)]],
    uint3 tid [[threadgroup_position_in_grid]],
    ushort thread_idx [[thread_index_in_threadgroup]],
    ushort simd_group_id [[simdgroup_index_in_threadgroup]],
    ushort simd_lane_id [[thread_index_in_simdgroup]]) {
  (void)routes;
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
  threadgroup T Wup[BN * BK_padded];
  threadgroup T Wgate[BN * BK_padded];

  const int block_id = int(tid.y);
  if (block_id >= max_blocks || block_id >= block_count[0]) {
    return;
  }
  const int out_col = int(tid.x) * BN;
  if (out_col >= intermediate) {
    return;
  }

  const int row_start = block_meta[block_id * 3 + 0];
  const int expert = block_meta[block_id * 3 + 1];
  const int rows = block_meta[block_id * 3 + 2];
  if (rows <= 0) {
    return;
  }

  const int packed_hidden = hidden * bytes_per_pack / pack_factor;
  const int scale_hidden = hidden / group_size;
  const size_t weight_expert_stride = size_t(intermediate) * packed_hidden;
  const size_t scale_expert_stride = size_t(intermediate) * scale_hidden;

  const device T* x_block = x + size_t(row_start) * hidden;
  const device uint8_t* up_weight =
      reinterpret_cast<const device uint8_t*>(up_w) +
      size_t(expert) * weight_expert_stride + size_t(out_col) * packed_hidden;
  const device uint8_t* gate_weight =
      reinterpret_cast<const device uint8_t*>(gate_w) +
      size_t(expert) * weight_expert_stride + size_t(out_col) * packed_hidden;
  const device uint8_t* up_scale =
      up_scales + size_t(expert) * scale_expert_stride +
      size_t(out_col) * scale_hidden;
  const device uint8_t* gate_scale =
      gate_scales + size_t(expert) * scale_expert_stride +
      size_t(out_col) * scale_hidden;

  thread loader_x_t loader_x(
      x_block, hidden, Xs, simd_group_id, simd_lane_id);
  thread loader_w_t loader_up(
      up_weight, up_scale, hidden, Wup, simd_group_id, simd_lane_id);
  thread loader_w_t loader_gate(
      gate_weight, gate_scale, hidden, Wgate, simd_group_id, simd_lane_id);
  thread mma_t mma_up(simd_group_id, simd_lane_id);
  thread mma_t mma_gate(simd_group_id, simd_lane_id);

  const short2 x_dims = short2(BK, short(rows));
  for (int k = 0; k < hidden / BK; ++k) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_x.load_safe(x_dims);
    loader_up.load_unsafe();
    loader_gate.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_up.mma(Xs, Wup);
    mma_gate.mma(Xs, Wgate);
    loader_x.next();
    loader_up.next();
    loader_gate.next();
  }

  // Reuse Xs/Wup after the K loop. MMATile::store performs the same
  // FP32->FP16 narrowing as the current pair-concat store_result.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup T* up_tile = Xs;
  threadgroup T* gate_tile = Wup;
  mma_up.Ctile.template store<T, WM, WN, BN, 1>(
      up_tile + mma_up.sm * BN + mma_up.sn);
  mma_gate.Ctile.template store<T, WM, WN, BN, 1>(
      gate_tile + mma_gate.sm * BN + mma_gate.sn);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int linear = int(thread_idx); linear < BM * BN; linear += threads) {
    const int row = linear / BN;
    const int col = linear - row * BN;
    if (row >= rows || out_col + col >= intermediate) {
      continue;
    }
    T up = up_tile[linear];
    T gate = gate_tile[linear];
    const T limit = T(activation_limit);
    gate = min(gate, limit);
    up = clamp(up, -limit, limit);
    // Use MLX's exact sigmoid functor (stable abs/branch form), not the
    // algebraically equivalent 1/(1+exp(-x)); the two differ in FP16.
    const T sigmoid_gate = Sigmoid{}(gate);
    const T value = T(T(gate * sigmoid_gate) * up);
    activated[size_t(row_start + row) * intermediate + out_col + col] = value;
  }
}

#define instantiate_ds4_pair_swiglu(type, bm, bn, bk, wm, wn)                 \
  instantiate_kernel(                                                         \
      "deepseek_mxfp4_gather_qmm_pair_swiglu_blocks_" #type                  \
      "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn,                \
      deepseek_mxfp4_gather_qmm_pair_swiglu_blocks,                           \
      type,                                                                   \
      bm,                                                                     \
      bn,                                                                     \
      bk,                                                                     \
      wm,                                                                     \
      wn)

instantiate_ds4_pair_swiglu(float16_t, 32, 32, 32, 1, 2);
