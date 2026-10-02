#include <metal_simdgroup>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/fp8.h"

using namespace metal;

struct DSparkQMVParams {
  int K;
  int N;
};

template <typename T, int M>
[[kernel, max_total_threads_per_threadgroup(64)]] void
dspark_exact_mxfp8_qmv_pair(
    const device T* input [[buffer(0)]],
    const device uint32_t* weight_a [[buffer(1)]],
    const device uint8_t* scales_a [[buffer(2)]],
    const device uint32_t* weight_b [[buffer(3)]],
    const device uint8_t* scales_b [[buffer(4)]],
    device T* output [[buffer(5)]],
    const constant DSparkQMVParams* params [[buffer(6)]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  constexpr int kValuesPerLane = 8;
  constexpr int kBlock = kValuesPerLane * 32;
  constexpr int kOutputsPerSimdgroup = 4;
  constexpr int kOutputsPerThreadgroup = 8;

  const int K = params->K;
  const int N = params->N;
  const int output_base = int(group.y) * kOutputsPerThreadgroup +
      int(simd_group) * kOutputsPerSimdgroup;
  const bool second = group.z != 0;
  const device uint8_t* weight =
      (const device uint8_t*)(second ? weight_b : weight_a);
  const device uint8_t* scales = second ? scales_b : scales_a;
  weight += output_base * K + int(lane) * kValuesPerLane;
  scales += output_base * (K / 32) + int(lane) / 4;
  const device T* input_lane = input + int(lane) * kValuesPerLane;
  device T* output_base_ptr = output + (second ? M * N : 0) + output_base;

  float accum[M][kOutputsPerSimdgroup] = {{0}};
  for (int k = 0; k < K; k += kBlock) {
    float values[M][kValuesPerLane];
#pragma unroll
    for (int row = 0; row < M; ++row) {
#pragma unroll
      for (int idx = 0; idx < kValuesPerLane; ++idx) {
        values[row][idx] = float(input_lane[row * K + idx]);
      }
    }
#pragma unroll
    for (int result = 0; result < kOutputsPerSimdgroup; ++result) {
      const device uint8_t* row_weight = weight + result * K;
      const device uint8_t* row_scale = scales + result * (K / 32);
      const uint8_t scale_byte = row_scale[0];
      const float scale = float(*(thread fp8_e8m0*)(&scale_byte));
#pragma unroll
      for (int row = 0; row < M; ++row) {
        float dot = 0.0f;
#pragma unroll
        for (int idx = 0; idx < kValuesPerLane; ++idx) {
          const uint8_t packed = row_weight[idx];
          dot += values[row][idx] * float(*(thread fp8_e4m3*)(&packed));
        }
        accum[row][result] += scale * dot;
      }
    }
    weight += kBlock;
    scales += kBlock / 32;
    input_lane += kBlock;
  }

#pragma unroll
  for (int row = 0; row < M; ++row) {
#pragma unroll
    for (int result = 0; result < kOutputsPerSimdgroup; ++result) {
      const float value = simd_sum(accum[row][result]);
      if (lane == 0) {
        output_base_ptr[row * N + result] = T(value);
      }
    }
  }
}

#define instantiate_dspark_qmv_pair(tname, dtype, m)                     \
  template [[host_name(                                                  \
      "omlx_dspark_exact_mxfp8_qmv_pair_" #tname "_m" #m)]]           \
  [[kernel]] decltype(dspark_exact_mxfp8_qmv_pair<dtype, m>)             \
      dspark_exact_mxfp8_qmv_pair<dtype, m>

instantiate_dspark_qmv_pair(float16, float16_t, 2);
instantiate_dspark_qmv_pair(float16, float16_t, 3);
instantiate_dspark_qmv_pair(float16, float16_t, 4);
instantiate_dspark_qmv_pair(float16, float16_t, 5);
instantiate_dspark_qmv_pair(float16, float16_t, 6);
instantiate_dspark_qmv_pair(bfloat16, bfloat16_t, 2);
instantiate_dspark_qmv_pair(bfloat16, bfloat16_t, 3);
instantiate_dspark_qmv_pair(bfloat16, bfloat16_t, 4);
instantiate_dspark_qmv_pair(bfloat16, bfloat16_t, 5);
instantiate_dspark_qmv_pair(bfloat16, bfloat16_t, 6);
