// SPDX-License-Identifier: Apache-2.0
// One-dispatch B1 DS4 ratio-128 Q-A/raw-KV/two-compressor bundle.

#include <metal_simdgroup>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/fp8.h"

using namespace metal;

[[kernel, max_total_threads_per_threadgroup(128)]] void
ds4_qkv_bundle128_all_b1(
    const device bfloat16_t* input [[buffer(0)]],
    const device uint32_t* wq [[buffer(1)]],
    const device uint8_t* sq [[buffer(2)]],
    const device uint32_t* wkv [[buffer(3)]],
    const device uint8_t* skv [[buffer(4)]],
    const device bfloat16_t* compressor_kv [[buffer(5)]],
    const device bfloat16_t* compressor_gate [[buffer(6)]],
    device bfloat16_t* output [[buffer(7)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int K = 4096;
  constexpr int kQRows = 1024;
  constexpr int kKVRows = 512;
  constexpr int kPackedDenseOffset = 1536;
  constexpr int kMXPhysicalGroups = 96;

  if (group_id < kMXPhysicalGroups) {
    constexpr int kValuesPerLane = 8;
    constexpr int kBlock = kValuesPerLane * 32;
    constexpr int kOutputsPerSimdgroup = 4;
    constexpr int kOutputsPerVirtualGroup = 8;
    const uint cohort = simd_gid >> 1;
    const uint virtual_simd = simd_gid & 1;
    const uint virtual_group = group_id * 2 + cohort;
    const bool second = virtual_group >= (kQRows / kOutputsPerVirtualGroup);
    const uint bank_group = second
        ? virtual_group - (kQRows / kOutputsPerVirtualGroup)
        : virtual_group;
    const int N = second ? kKVRows : kQRows;
    const int output_base = int(bank_group) * kOutputsPerVirtualGroup +
        int(virtual_simd) * kOutputsPerSimdgroup;
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
    return;
  }

  constexpr int TM = 4;
  constexpr int TN = 4;
  constexpr int blockM = 16;
  constexpr int blockN = 128;
  const int group_row = (int(group_id) - kMXPhysicalGroups) * blockM;
  const bool second = group_row >= 512;
  const device bfloat16_t* matrix = second ? compressor_gate : compressor_kv;
  const int bank_start = second ? 512 : 0;
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
      output[kPackedDenseOffset + global_out_row + tm] =
          bfloat16_t(result[tm]);
    }
  }
}
