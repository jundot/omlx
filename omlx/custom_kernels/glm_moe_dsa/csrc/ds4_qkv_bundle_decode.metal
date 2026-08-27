// SPDX-License-Identifier: Apache-2.0
// One-dispatch B1 DS4 ratio-4 QKV/compressor bundle. Kept in a separate
// metallib so decode experimentation cannot perturb the qualified classic
// prefill library's binary or pipeline cache.

#include <metal_simdgroup>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/fp8.h"

using namespace metal;

[[kernel]] void ds4_router_topk6_f32(
    const device float* scores [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  const device float* src = scores + size_t(row) * 256;
  device uint* dst = indices + size_t(row) * 6;
  threadgroup float score0[256];
  threadgroup int idx0[256];
  threadgroup float score1[256];
  threadgroup int idx1[256];
  float score = src[tid];
  int idx = int(tid);
  uint cross_stage = 0;
  for (uint k = 2; k <= 256; k <<= 1) {
    for (uint j = k >> 1; j > 0; j >>= 1) {
      float peer_score;
      int peer_idx;
      bool take_peer;
      const bool lower = (tid & j) == 0;
      const bool descending = (tid & k) == 0;
      if (j < 32) {
        peer_score = simd_shuffle_xor(score, ushort(j));
        peer_idx = simd_shuffle_xor(idx, ushort(j));
        take_peer = descending
            ? (lower ? score < peer_score : score > peer_score)
            : (lower ? score > peer_score : score < peer_score);
      } else {
        threadgroup float* score_tg =
            (cross_stage & 1u) != 0u ? score1 : score0;
        threadgroup int* idx_tg =
            (cross_stage & 1u) != 0u ? idx1 : idx0;
        score_tg[tid] = score;
        idx_tg[tid] = idx;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint other = tid ^ j;
        peer_score = score_tg[other];
        peer_idx = idx_tg[other];
        take_peer = descending
            ? (lower ? score < peer_score : score > peer_score)
            : (lower ? score > peer_score : score < peer_score);
        cross_stage++;
      }
      if (take_peer) {
        score = peer_score;
        idx = peer_idx;
      }
    }
  }
  if (tid < 6) dst[tid] = uint(idx);
}

// Threadgroups [0,96) host two virtual 64-thread MXFP8 groups (four
// simdgroups split 2+2), covering 1,536 Q-A/KV rows at 16 rows per physical
// group. Threadgroups [96,256) run the exact four-simdgroup dense body at 16
// rows per group. Lane K walks and reduction trees match the two-dispatch ABI.
[[kernel, max_total_threads_per_threadgroup(128)]] void
ds4_qkv_bundle_all_b1(
    const device bfloat16_t* input [[buffer(0)]],
    const device uint32_t* wq [[buffer(1)]],
    const device uint8_t* sq [[buffer(2)]],
    const device uint32_t* wkv [[buffer(3)]],
    const device uint8_t* skv [[buffer(4)]],
    const device bfloat16_t* compressor_kv [[buffer(5)]],
    const device bfloat16_t* compressor_gate [[buffer(6)]],
    const device bfloat16_t* index_kv [[buffer(7)]],
    const device bfloat16_t* index_gate [[buffer(8)]],
    device bfloat16_t* output [[buffer(9)]],
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
  const int dense_group = int(group_id) - kMXPhysicalGroups;
  const int group_row = dense_group * blockM;
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
      output[kPackedDenseOffset + global_out_row + tm] =
          bfloat16_t(result[tm]);
    }
  }
}
