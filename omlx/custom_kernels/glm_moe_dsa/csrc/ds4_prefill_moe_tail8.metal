// SPDX-License-Identifier: Apache-2.0
//
// Isolated native tail-cull experiment. No model or runtime dispatch references it.
// The kernels retain one BM32 expert block and one weight-tile load, but split
// the route-row MMA into four independently gated BM8 microtiles. A 24-row
// block executes three route microtiles instead of multiplying eight padded
// zero rows. K order, FP32 accumulation, FP16 stores, and LimitedSwiGLU order
// remain unchanged.

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/fp_quantized.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"

using namespace metal;

template <typename T>
inline T ds4_tail8_sigmoid(T x) {
  auto y = 1 / (1 + metal::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}

template <typename T, int BN, int BK, int WM, int WN>
[[kernel]] void deepseek_mxfp4_gather_qmm_pair_swiglu_blocks_tail8(
    device const T* x [[buffer(0)]],
    device const uint32_t* up_w [[buffer(1)]],
    device const uint8_t* up_scales [[buffer(2)]],
    device const uint32_t* gate_w [[buffer(3)]],
    device const uint8_t* gate_scales [[buffer(4)]],
    device const int32_t* block_meta [[buffer(5)]],
    device const int32_t* block_count [[buffer(6)]],
    device T* activated [[buffer(7)]],
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
  constexpr int BM = 32;
  constexpr int MICRO_BM = 8;
  constexpr int group_size = 32;
  constexpr int bits = 4;
  constexpr int pack_factor = get_pack_factor<8, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack();
  constexpr int BK_padded = BK + 16 / sizeof(T);
  constexpr int threads = WM * WN * SIMD_SIZE;

  using micro_mma_t = mlx::steel::BlockMMA<
      T,
      T,
      MICRO_BM,
      BN,
      BK,
      WM,
      WN,
      false,
      true,
      BK_padded,
      BK_padded>;
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
  thread micro_mma_t up0(simd_group_id, simd_lane_id);
  thread micro_mma_t up1(simd_group_id, simd_lane_id);
  thread micro_mma_t up2(simd_group_id, simd_lane_id);
  thread micro_mma_t up3(simd_group_id, simd_lane_id);
  thread micro_mma_t gate0(simd_group_id, simd_lane_id);
  thread micro_mma_t gate1(simd_group_id, simd_lane_id);
  thread micro_mma_t gate2(simd_group_id, simd_lane_id);
  thread micro_mma_t gate3(simd_group_id, simd_lane_id);

  const short2 x_dims = short2(BK, short(rows));
  for (int k = 0; k < hidden / BK; ++k) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_x.load_safe(x_dims);
    loader_up.load_unsafe();
    loader_gate.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);

    up0.mma(Xs, Wup);
    gate0.mma(Xs, Wgate);
    if (rows > 8) {
      up1.mma(Xs + 8 * BK_padded, Wup);
      gate1.mma(Xs + 8 * BK_padded, Wgate);
    }
    if (rows > 16) {
      up2.mma(Xs + 16 * BK_padded, Wup);
      gate2.mma(Xs + 16 * BK_padded, Wgate);
    }
    if (rows > 24) {
      up3.mma(Xs + 24 * BK_padded, Wup);
      gate3.mma(Xs + 24 * BK_padded, Wgate);
    }

    loader_x.next();
    loader_up.next();
    loader_gate.next();
  }

  // Reuse the loader storage for the two FP16 projection boundaries.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup T* up_tile = Xs;
  threadgroup T* gate_tile = Wup;
#define STORE_PAIR_MICROTILE(q, up_mma, gate_mma)                            \
  if (rows > (q) * MICRO_BM) {                                               \
    up_mma.Ctile.template store<T, WM, WN, BN, 1>(                           \
        up_tile + ((q) * MICRO_BM + up_mma.sm) * BN + up_mma.sn);            \
    gate_mma.Ctile.template store<T, WM, WN, BN, 1>(                         \
        gate_tile + ((q) * MICRO_BM + gate_mma.sm) * BN + gate_mma.sn);      \
  }
  STORE_PAIR_MICROTILE(0, up0, gate0)
  STORE_PAIR_MICROTILE(1, up1, gate1)
  STORE_PAIR_MICROTILE(2, up2, gate2)
  STORE_PAIR_MICROTILE(3, up3, gate3)
#undef STORE_PAIR_MICROTILE
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
    const T sigmoid_gate = ds4_tail8_sigmoid(gate);
    const T value = T(T(gate * sigmoid_gate) * up);
    activated[size_t(row_start + row) * intermediate + out_col + col] = value;
  }
}

template <typename T, int BN, int BK, int WM, int WN>
[[kernel]] void deepseek_mxfp4_gather_qmm_blocks_tail8(
    device const T* x [[buffer(0)]],
    device const uint32_t* w [[buffer(1)]],
    device const uint8_t* scales [[buffer(2)]],
    device const int32_t* block_meta [[buffer(3)]],
    device const int32_t* block_count [[buffer(4)]],
    device T* y [[buffer(5)]],
    constant int& max_blocks [[buffer(6)]],
    constant int& routes [[buffer(7)]],
    constant int& output_width [[buffer(8)]],
    constant int& intermediate [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    ushort simd_group_id [[simdgroup_index_in_threadgroup]],
    ushort simd_lane_id [[thread_index_in_simdgroup]]) {
  (void)routes;
  using U = T;
  constexpr int BM = 32;
  constexpr int MICRO_BM = 8;
  constexpr int group_size = 32;
  constexpr int bits = 4;
  constexpr int pack_factor = get_pack_factor<8, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack();
  constexpr int BK_padded = BK + 16 / sizeof(T);
  constexpr int threads = WM * WN * SIMD_SIZE;

  using micro_mma_t = mlx::steel::BlockMMA<
      T,
      U,
      MICRO_BM,
      BN,
      BK,
      WM,
      WN,
      false,
      true,
      BK_padded,
      BK_padded>;
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
  const int out_col = int(tid.x) * BN;
  if (out_col >= output_width) {
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
  const size_t weight_expert_stride = size_t(output_width) * packed_intermediate;
  const size_t scale_expert_stride = size_t(output_width) * scale_intermediate;
  const device T* x_block = x + size_t(row_start) * intermediate;
  const device uint8_t* weight =
      reinterpret_cast<const device uint8_t*>(w) +
      size_t(expert) * weight_expert_stride + size_t(out_col) * packed_intermediate;
  const device uint8_t* scale =
      scales + size_t(expert) * scale_expert_stride +
      size_t(out_col) * scale_intermediate;

  thread loader_x_t loader_x(
      x_block, intermediate, Xs, simd_group_id, simd_lane_id);
  thread loader_w_t loader_w(
      weight, scale, intermediate, Ws, simd_group_id, simd_lane_id);
  thread micro_mma_t mma0(simd_group_id, simd_lane_id);
  thread micro_mma_t mma1(simd_group_id, simd_lane_id);
  thread micro_mma_t mma2(simd_group_id, simd_lane_id);
  thread micro_mma_t mma3(simd_group_id, simd_lane_id);

  const short2 x_dims = short2(BK, short(rows));
  for (int k = 0; k < intermediate / BK; ++k) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_x.load_safe(x_dims);
    loader_w.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma0.mma(Xs, Ws);
    if (rows > 8) {
      mma1.mma(Xs + 8 * BK_padded, Ws);
    }
    if (rows > 16) {
      mma2.mma(Xs + 16 * BK_padded, Ws);
    }
    if (rows > 24) {
      mma3.mma(Xs + 24 * BK_padded, Ws);
    }
    loader_x.next();
    loader_w.next();
  }

  device T* output = y + size_t(row_start) * output_width + out_col;
#define STORE_DOWN_MICROTILE(q, mma_op)                                      \
  if (rows > (q) * MICRO_BM) {                                               \
    const short valid_rows = short(min(MICRO_BM, rows - (q) * MICRO_BM));     \
    mma_op.store_result_slice(                                                \
        output + size_t((q) * MICRO_BM) * output_width,                      \
        output_width,                                                        \
        short2(0, 0),                                                        \
        short2(BN, valid_rows));                                             \
  }
  STORE_DOWN_MICROTILE(0, mma0)
  STORE_DOWN_MICROTILE(1, mma1)
  STORE_DOWN_MICROTILE(2, mma2)
  STORE_DOWN_MICROTILE(3, mma3)
#undef STORE_DOWN_MICROTILE
}

#define instantiate_tail8_pair(type, bn, bk, wm, wn)                        \
  instantiate_kernel(                                                        \
      "deepseek_mxfp4_gather_qmm_pair_swiglu_blocks_tail8_" #type "_bm32"                       \
      "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn,                       \
      deepseek_mxfp4_gather_qmm_pair_swiglu_blocks_tail8,                                 \
      type,                                                                  \
      bn,                                                                    \
      bk,                                                                    \
      wm,                                                                    \
      wn)

#define instantiate_tail8_down(type, bn, bk, wm, wn)                        \
  instantiate_kernel(                                                        \
      "deepseek_mxfp4_gather_qmm_blocks_tail8_" #type "_bm32"                              \
      "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn,                       \
      deepseek_mxfp4_gather_qmm_blocks_tail8,                                        \
      type,                                                                  \
      bn,                                                                    \
      bk,                                                                    \
      wm,                                                                    \
      wn)

instantiate_tail8_pair(float16_t, 32, 32, 1, 2);
instantiate_tail8_down(float16_t, 32, 32, 1, 2);
