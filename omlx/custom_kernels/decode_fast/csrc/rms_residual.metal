// SPDX-License-Identifier: Apache-2.0
// omlx fused residual-add + RMS norm kernels (decode fast path).
//
// Ported from the mlx core PR ml-explore/mlx#4295 (closed unmerged upstream)
// so omlx can ship the fusion without waiting on an mlx release. The kernel
// bodies mirror mlx/backend/metal/kernels/rms_norm.metal @ jonathan308/mlx
// pr/fast-rmsnorm-residual with the symbol names prefixed (omlx_rms_*) to
// stay collision-free with mlx's own metallib. Fallback composition
// (add -> rms_norm) remains in fast.py for CPU / unsupported shapes.

#include <metal_common>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

// Fused residual add + RMS norm: computes out = rms_norm(x + res) * w and
// out_sum = x + res in a single pass. The reduction structure matches
// mlx's rms_single_row exactly.
template <typename T, int N_READS = RMS_N_READS>
[[kernel]] void omlx_rms_residual_single_row(
    const device T* x,
    const device T* res,
    const device T* w,
    device T* out,
    device T* out_sum,
    constant float& eps,
    constant uint& axis_size,
    constant uint& w_stride,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int SIMD_SIZE = 32;

  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[SIMD_SIZE];

  float acc = 0;
  float thread_x[N_READS];
  x += gid * size_t(axis_size) + lid * N_READS;
  res += gid * size_t(axis_size) + lid * N_READS;
  w += w_stride * lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      T s = x[i] + res[i];
      thread_x[i] = s;
      acc += thread_x[i] * thread_x[i];
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      T s = (lid * N_READS + i < axis_size) ? T(x[i] + res[i]) : T(0);
      thread_x[i] = s;
      acc += thread_x[i] * thread_x[i];
    }
  }
  acc = simd_sum(acc);
  //  Initialize shared memory
  if (simd_group_id == 0) {
    local_sums[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Write simd accumulations into shared memory
  if (simd_lane_id == 0) {
    local_sums[simd_group_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Accumulate over simd groups
  if (simd_group_id == 0) {
    acc = simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Write the outputs using the cached sums
  out += gid * size_t(axis_size) + lid * N_READS;
  out_sum += gid * size_t(axis_size) + lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      out_sum[i] = static_cast<T>(thread_x[i]);
      out[i] =
          w[w_stride * i] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        out_sum[i] = static_cast<T>(thread_x[i]);
        out[i] =
            w[w_stride * i] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
      }
    }
  }
}

// Looped variant for large axes. The sums are stored to out_sum during the
// accumulation pass and re-read (cache-hot) in the write pass instead of
// re-reading both inputs.
template <typename T, int N_READS = RMS_N_READS>
[[kernel]] void omlx_rms_residual_looped(
    const device T* x,
    const device T* res,
    const device T* w,
    device T* out,
    device T* out_sum,
    constant float& eps,
    constant uint& axis_size,
    constant uint& w_stride,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int SIMD_SIZE = 32;
  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[SIMD_SIZE];

  float acc = 0;
  x += gid * size_t(axis_size) + lid * N_READS;
  res += gid * size_t(axis_size) + lid * N_READS;
  w += w_stride * lid * N_READS;
  out_sum += gid * size_t(axis_size) + lid * N_READS;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        T s = x[i + r] + res[i + r];
        out_sum[i + r] = s;
        float sf = s;
        acc += sf * sf;
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          T s = x[i + r] + res[i + r];
          out_sum[i + r] = s;
          float sf = s;
          acc += sf * sf;
        }
      }
    }
  }
  acc = simd_sum(acc);
  //  Initialize shared memory
  if (simd_group_id == 0) {
    local_sums[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Write simd accumulations into shared memory
  if (simd_lane_id == 0) {
    local_sums[simd_group_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Accumulate over simd groups
  if (simd_group_id == 0) {
    acc = simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Write the outputs re-reading the cached sums
  out += gid * size_t(axis_size) + lid * N_READS;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        float sf = out_sum[r + i];
        out[r + i] =
            w[w_stride * (i + r)] * static_cast<T>(sf * local_inv_mean[0]);
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          float sf = out_sum[r + i];
          out[r + i] =
              w[w_stride * (i + r)] * static_cast<T>(sf * local_inv_mean[0]);
        }
      }
    }
  }
}

// clang-format off
#define instantiate_omlx_rms_residual(name, itype)                          \
  instantiate_kernel(                                                       \
      "omlx_rms_residual" #name, omlx_rms_residual_single_row, itype)       \
  instantiate_kernel(                                                       \
      "omlx_rms_residual16" #name, omlx_rms_residual_single_row, itype, 16) \
  instantiate_kernel(                                                       \
      "omlx_rms_residual_looped" #name, omlx_rms_residual_looped, itype)

instantiate_omlx_rms_residual(float32, float)
instantiate_omlx_rms_residual(float16, half)
instantiate_omlx_rms_residual(bfloat16, bfloat16_t)
// clang-format on
