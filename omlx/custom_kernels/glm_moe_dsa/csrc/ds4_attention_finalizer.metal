// SPDX-License-Identifier: Apache-2.0
// Isolated DS4 M=1024 BF16 RMSNorm+RoPE finalizers; no model dispatch.
// Reduction and rotation order are adapted from MLX v0.32.0's MIT-licensed
// rms_norm.metal and rope.metal; see MLX_LICENSE.txt in this directory.

#include <metal_math>
#include <metal_simdgroup>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

constant constexpr uint kHeadDim = 512;
constant constexpr uint kPairs = kHeadDim / 2;
constant constexpr uint kNormReads = 4;
constant constexpr uint kNormThreads = kHeadDim / kNormReads;
constant constexpr uint kHeadsPerGroup = 4;

inline float2 ds4_rope_pair(float2 value, float freq, int position) {
  const float inv_freq = 1.0f / freq;
  const float L = 1.0f * static_cast<float>(position);
  const float theta = L * inv_freq;
  const float cosine = metal::fast::cos(theta);
  const float sine = metal::fast::sin(theta);
  return float2(
      value.x * cosine - value.y * sine,
      value.x * sine + value.y * cosine);
}

// Q arrives in the stock pre-transpose layout [1, M, H, 512]. One 512-thread
// group owns four heads at one token. Each 128-thread head team reproduces
// MLX rms_single_row<bf16,N_READS=4> exactly, including its two-level simd_sum
// and precise rsqrt. The BF16-normalized values are then rotated into the
// stock contiguous [1, H, M, 512] output. Team zero computes the 256 explicit
// frequency pairs once, matching MLX rope_freqs' four-head inner loop.
[[kernel, max_total_threads_per_threadgroup(512)]] void
ds4_q_head_rms_rope_bf16(
    const device bfloat16_t* input [[buffer(0)]],
    const device float* freqs [[buffer(1)]],
    device bfloat16_t* normalized_output [[buffer(2)]],
    device bfloat16_t* rotated_output [[buffer(3)]],
    constant int& offset [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    constant uint& heads [[buffer(6)]],
    constant uint& write_normalized [[buffer(7)]],
    constant uint& tokens [[buffer(8)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float local_inv_mean[kHeadsPerGroup];
  threadgroup float local_sums[kHeadsPerGroup * 32];
  threadgroup float2 trig[kPairs];

  const uint local_head = tid / kNormThreads;
  const uint local_lid = tid - local_head * kNormThreads;
  const uint local_simd = simd_group & 3u;
  const uint head = group.x * kHeadsPerGroup + local_head;
  const uint token = group.y;
  const size_t input_row = (size_t(token) * heads + head) * kHeadDim;
  const size_t input_start = input_row + local_lid * kNormReads;

  float values[kNormReads];
  float acc = 0.0f;
  for (uint i = 0; i < kNormReads; ++i) {
    values[i] = static_cast<float>(input[input_start + i]);
    acc += values[i] * values[i];
  }
  acc = simd_sum(acc);

  if (local_simd == 0u) {
    local_sums[local_head * 32u + lane] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0u) {
    local_sums[local_head * 32u + local_simd] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (local_simd == 0u) {
    acc = simd_sum(local_sums[local_head * 32u + lane]);
    if (lane == 0u) {
      local_inv_mean[local_head] =
          metal::precise::rsqrt(acc / float(kHeadDim) + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  bfloat16_t normalized[kNormReads];
  for (uint i = 0; i < kNormReads; ++i) {
    normalized[i] = bfloat16_t(1.0f) * static_cast<bfloat16_t>(
        values[i] * local_inv_mean[local_head]);
    if (write_normalized != 0u) {
      normalized_output[input_start + i] = normalized[i];
    }
  }

  if (local_head == 0u) {
    const uint pair0 = local_lid * 2u;
    const uint pair1 = pair0 + 1u;
    const int position = int(token) + offset;
    const float L = 1.0f * static_cast<float>(position);
    const float theta0 = L * (1.0f / freqs[pair0]);
    const float theta1 = L * (1.0f / freqs[pair1]);
    trig[pair0] =
        float2(metal::fast::cos(theta0), metal::fast::sin(theta0));
    trig[pair1] =
        float2(metal::fast::cos(theta1), metal::fast::sin(theta1));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const size_t output_row = (size_t(head) * tokens + token) * kHeadDim;
  const uint pair_base = local_lid * 2u;
  for (uint i = 0; i < kNormReads; i += 2u) {
    const uint pair = pair_base + i / 2u;
    const float2 cs = trig[pair];
    const float2 value = float2(
        static_cast<float>(normalized[i]),
        static_cast<float>(normalized[i + 1u]));
    const float2 rotated = float2(
        value.x * cs.x - value.y * cs.y,
        value.x * cs.y + value.y * cs.x);
    rotated_output[output_row + local_lid * kNormReads + i] =
        static_cast<bfloat16_t>(rotated.x);
    rotated_output[output_row + local_lid * kNormReads + i + 1u] =
        static_cast<bfloat16_t>(rotated.y);
  }
}

// KV uses the same 128-thread reduction but retains MLX's weighted boundary:
// normalize to BF16 first, then multiply by the BF16 learned norm weight.
[[kernel, max_total_threads_per_threadgroup(128)]] void
ds4_kv_rms_rope_bf16(
    const device bfloat16_t* input [[buffer(0)]],
    const device bfloat16_t* weight [[buffer(1)]],
    const device float* freqs [[buffer(2)]],
    device bfloat16_t* normalized_output [[buffer(3)]],
    device bfloat16_t* rotated_output [[buffer(4)]],
    constant int& offset [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    constant uint& write_normalized [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[32];

  const size_t row = size_t(group) * kHeadDim;
  const size_t start = row + lid * kNormReads;
  float values[kNormReads];
  float acc = 0.0f;
  for (uint i = 0; i < kNormReads; ++i) {
    values[i] = static_cast<float>(input[start + i]);
    acc += values[i] * values[i];
  }
  acc = simd_sum(acc);
  if (simd_group == 0u) {
    local_sums[lane] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0u) {
    local_sums[simd_group] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0u) {
    acc = simd_sum(local_sums[lane]);
    if (lane == 0u) {
      local_inv_mean[0] =
          metal::precise::rsqrt(acc / float(kHeadDim) + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  bfloat16_t normalized[kNormReads];
  for (uint i = 0; i < kNormReads; ++i) {
    const uint column = lid * kNormReads + i;
    normalized[i] = weight[column] * static_cast<bfloat16_t>(
        values[i] * local_inv_mean[0]);
    if (write_normalized != 0u) {
      normalized_output[start + i] = normalized[i];
    }
  }

  const int position = int(group) + offset;
  const uint pair_base = lid * 2u;
  for (uint i = 0; i < kNormReads; i += 2u) {
    const uint pair = pair_base + i / 2u;
    const float2 value = float2(
        static_cast<float>(normalized[i]),
        static_cast<float>(normalized[i + 1u]));
    const float2 rotated = ds4_rope_pair(value, freqs[pair], position);
    rotated_output[start + i] = static_cast<bfloat16_t>(rotated.x);
    rotated_output[start + i + 1u] = static_cast<bfloat16_t>(rotated.y);
  }
}
