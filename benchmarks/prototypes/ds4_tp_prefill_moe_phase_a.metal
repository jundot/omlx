// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED PROTOTYPE ONLY. This file is deliberately absent from the native
// CMake target and production dispatch. Its fixed M=1024 TP4/4 contract is
// exercised by bench_ds4_tp_prefill_moe_campaign.py after it is promoted into
// a separately gated native experiment.

#include "mlx/backend/metal/kernels/fp_quantized.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

// Convert an already stable expert-sorted route list into a deterministic
// BM32 block list. Unlike the current atomic block allocator, block positions
// are a prefix sum over expert id, so repeated executions are byte-identical.
[[host_name("prototype_ds4_moe_deterministic_blocks_e256_bm32")]]
kernel void prototype_ds4_moe_deterministic_blocks_e256_bm32(
    device const int32_t* sorted_experts [[buffer(0)]],
    device int32_t* block_meta [[buffer(1)]],
    device int32_t* block_count [[buffer(2)]],
    constant int& routes [[buffer(3)]],
    ushort expert [[thread_position_in_threadgroup]]) {
  constexpr ushort experts = 256;
  constexpr int bm = 32;
  threadgroup ushort tile_counts[experts];

  int lo = 0;
  int hi = routes;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (sorted_experts[mid] < int(expert)) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  const int start = lo;
  hi = routes;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (sorted_experts[mid] <= int(expert)) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  const int end = lo;
  const ushort count = ushort((end - start + bm - 1) / bm);
  tile_counts[expert] = count;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  int block_base = 0;
  for (ushort prior = 0; prior < expert; ++prior) {
    block_base += int(tile_counts[prior]);
  }
  for (ushort tile = 0; tile < count; ++tile) {
    const int row = start + int(tile) * bm;
    const int slot = block_base + int(tile);
    block_meta[slot * 3 + 0] = row;
    block_meta[slot * 3 + 1] = int(expert);
    block_meta[slot * 3 + 2] = min(bm, end - row);
  }
  if (expert == experts - 1) {
    block_count[0] = block_base + int(count);
  }
}

// One threadgroup computes the same BM32xBN32 up and gate tiles as the two
// current pair-concat threadgroups. X is staged once per BK32 step. The two
// FP32 accumulator sets remain independent and keep the existing K order.
// Results are narrowed to FP16 in threadgroup memory before LimitedSwiGLU,
// preserving the current projection-store boundary without global gate/up.
[[host_name("prototype_ds4_mxfp4_pair_swiglu_f16_bm32_bn32")]]
kernel void prototype_ds4_mxfp4_pair_swiglu_f16_bm32_bn32(
    device const float16_t* x [[buffer(0)]],
    device const uint32_t* up_w [[buffer(1)]],
    device const uint8_t* up_scales [[buffer(2)]],
    device const uint32_t* gate_w [[buffer(3)]],
    device const uint8_t* gate_scales [[buffer(4)]],
    device const int32_t* block_meta [[buffer(5)]],
    device const int32_t* block_count [[buffer(6)]],
    device float16_t* activated [[buffer(7)]],
    constant int& max_blocks [[buffer(8)]],
    constant int& routes [[buffer(9)]],
    constant int& intermediate [[buffer(10)]],
    constant int& hidden [[buffer(11)]],
    constant float& activation_limit [[buffer(12)]],
    uint3 tid [[threadgroup_position_in_grid]],
    ushort thread_idx [[thread_index_in_threadgroup]],
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

  // Reuse Xs/Wup after the K loop. MMATile::store performs the same FP32->FP16
  // narrowing as BlockMMA::store_result in the current pair-concat kernel.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup T* up_tile = Xs;
  threadgroup T* gate_tile = Wup;
  mma_up.Ctile.template store<T, WM, WN>(
      up_tile + mma_up.sm * BN + mma_up.sn, BN);
  mma_gate.Ctile.template store<T, WM, WN>(
      gate_tile + mma_gate.sm * BN + mma_gate.sn, BN);
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
    // This expression is intentionally checked with array_equal at the
    // post-LimitedSwiGLU boundary before any native promotion.
    const T sigmoid_gate = T(T(1) / T(T(1) + T(exp(-gate))));
    const T value = T(T(gate * sigmoid_gate) * up);
    activated[(size_t(row_start + row) * intermediate) + out_col + col] = value;
  }
}

// Phase-B exact oracle. It retains the sorted [R,H] FP16 down tensor on
// purpose; its job is to freeze the post-down rounding and fixed-slot order
// before testing schedules that remove that tensor. No atomic accumulation is
// permitted in a lossless direct-output candidate.
[[host_name("prototype_ds4_moe_top6_post_down_bf16")]]
kernel void prototype_ds4_moe_top6_post_down_bf16(
    device const float16_t* sorted_down [[buffer(0)]],
    device const uint32_t* inverse_order [[buffer(1)]],
    device const float* scores [[buffer(2)]],
    device bfloat16_t* local_output [[buffer(3)]],
    constant int& tokens [[buffer(4)]],
    constant int& hidden [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int col = int(gid.x);
  const int token = int(gid.y);
  if (token >= tokens || col >= hidden) {
    return;
  }

  bfloat16_t total = bfloat16_t(0);
  for (int slot = 0; slot < 6; ++slot) {
    const int route_id = token * 6 + slot;
    const uint32_t sorted_row = inverse_order[route_id];
    const bfloat16_t route_value = bfloat16_t(
        sorted_down[size_t(sorted_row) * hidden + col]);
    const bfloat16_t route_score = bfloat16_t(scores[route_id]);
    const bfloat16_t weighted = bfloat16_t(route_value * route_score);
    total = bfloat16_t(total + weighted);
  }
  local_output[size_t(token) * hidden + col] = total;
}
