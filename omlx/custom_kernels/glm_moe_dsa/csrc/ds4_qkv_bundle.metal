// SPDX-License-Identifier: Apache-2.0
// Isolated fixed B1 DS4 ratio-4 QKV/compressor bundle; no production dispatch.

#include <metal_simdgroup>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/fp8.h"

using namespace metal;

// Dispatch 1/2: Q-A (1024 rows) and raw KV (512 rows). This preserves the
// exact M=1 MXFP8 K walk and simd_sum reduction used by the established DSpark
// exact QMV kernel while allowing the two unequal banks in one launch.
[[kernel, max_total_threads_per_threadgroup(64)]] void
ds4_qkv_bundle_mxfp8_b1(
    const device bfloat16_t* input [[buffer(0)]],
    const device uint32_t* wq [[buffer(1)]],
    const device uint8_t* sq [[buffer(2)]],
    const device uint32_t* wkv [[buffer(3)]],
    const device uint8_t* skv [[buffer(4)]],
    device bfloat16_t* output [[buffer(5)]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  constexpr int K = 4096;
  constexpr int kValuesPerLane = 8;
  constexpr int kBlock = kValuesPerLane * 32;
  constexpr int kOutputsPerSimdgroup = 4;
  constexpr int kOutputsPerThreadgroup = 8;
  constexpr int kQRows = 1024;
  constexpr int kKVRows = 512;

  const bool second = group.z != 0;
  const int N = second ? kKVRows : kQRows;
  const int output_base = int(group.y) * kOutputsPerThreadgroup +
      int(simd_group) * kOutputsPerSimdgroup;
  if (output_base >= N) {
    return;
  }
  const device uint8_t* weight =
      reinterpret_cast<const device uint8_t*>(second ? wkv : wq);
  const device uint8_t* scales = second ? skv : sq;
  weight += output_base * K + int(lane) * kValuesPerLane;
  scales += output_base * (K / 32) + int(lane) / 4;
  const device bfloat16_t* input_lane = input + int(lane) * kValuesPerLane;
  device bfloat16_t* output_base_ptr =
      output + (second ? kQRows : 0) + output_base;

  float accum[kOutputsPerSimdgroup] = {0};
  for (int k = 0; k < K; k += kBlock) {
    float values[kValuesPerLane];
    #pragma unroll
    for (int idx = 0; idx < kValuesPerLane; ++idx) {
      values[idx] = float(input_lane[idx]);
    }
    #pragma unroll
    for (int result = 0; result < kOutputsPerSimdgroup; ++result) {
      const device uint8_t* row_weight = weight + result * K;
      const device uint8_t* row_scale = scales + result * (K / 32);
      uint8_t scale_byte = row_scale[0];
      const float scale = float(*(thread fp8_e8m0*)(&scale_byte));
      float dot = 0.0f;
      #pragma unroll
      for (int idx = 0; idx < kValuesPerLane; ++idx) {
        uint8_t packed = row_weight[idx];
        dot += values[idx] * float(*(thread fp8_e4m3*)(&packed));
      }
      accum[result] += scale * dot;
    }
    weight += kBlock;
    scales += kBlock / 32;
    input_lane += kBlock;
  }

  #pragma unroll
  for (int result = 0; result < kOutputsPerSimdgroup; ++result) {
    const float value = simd_sum(accum[result]);
    if (lane == 0) {
      output_base_ptr[result] = bfloat16_t(value);
    }
  }
}

// Dispatch 2/2: four BF16 dense banks as one logical 2560-row GEMV. The
// reduction geometry is copied exactly from MLX's B1 grouped-stock choice:
// GEMVKernel<bf16, BM4, BN1, SM1, SN32, TM4, TN4>. Bank boundaries are all
// multiples of its 16-row threadgroup tile, so no group spans two buffers.
[[kernel, max_total_threads_per_threadgroup(128)]] void
ds4_qkv_bundle_dense_b1(
    const device bfloat16_t* input [[buffer(0)]],
    const device bfloat16_t* compressor_kv [[buffer(1)]],
    const device bfloat16_t* compressor_gate [[buffer(2)]],
    const device bfloat16_t* index_kv [[buffer(3)]],
    const device bfloat16_t* index_gate [[buffer(4)]],
    device bfloat16_t* output [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int K = 4096;
  constexpr int TM = 4;
  constexpr int TN = 4;
  constexpr int blockM = 16;
  constexpr int blockN = 128;
  constexpr int packed_dense_offset = 1536;

  const int group_row = int(group.x) * blockM;
  const device bfloat16_t* matrix;
  int bank_start;
  if (group_row < 1024) {
    matrix = compressor_kv;
    bank_start = 0;
  } else if (group_row < 2048) {
    matrix = compressor_gate;
    bank_start = 1024;
  } else if (group_row < 2304) {
    matrix = index_kv;
    bank_start = 2048;
  } else {
    matrix = index_gate;
    bank_start = 2304;
  }

  const int global_out_row = group_row + int(simd_gid) * TM;
  const int local_out_row = global_out_row - bank_start;
  int k_col = int(lane) * TN;
  float result[TM] = {0};
  bfloat16_t values[TN];
  float coefficients[TN];

  for (int iteration = 0; iteration < K / blockN; ++iteration) {
    #pragma unroll
    for (int tn = 0; tn < TN; ++tn) {
      coefficients[tn] = float(input[k_col + tn]);
    }
    #pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
      const device bfloat16_t* row =
          matrix + size_t(local_out_row + tm) * K + k_col;
      #pragma unroll
      for (int tn = 0; tn < TN; ++tn) {
        values[tn] = row[tn];
      }
      #pragma unroll
      for (int tn = 0; tn < TN; ++tn) {
        result[tm] += float(values[tn]) * coefficients[tn];
      }
    }
    k_col += blockN;
  }

  #pragma unroll
  for (int tm = 0; tm < TM; ++tm) {
    #pragma unroll
    for (ushort offset = 16; offset >= 1; offset >>= 1) {
      result[tm] += simd_shuffle_down(result[tm], offset);
    }
  }
  if (lane == 0) {
    #pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
      output[packed_dense_offset + global_out_row + tm] =
          bfloat16_t(result[tm]);
    }
  }
}
