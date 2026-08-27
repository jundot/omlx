// SPDX-License-Identifier: Apache-2.0
// omlx fused RoPE(K) + key-cache append kernels — port of
// ml-explore/mlx#4297 (closed unmerged upstream). Writes the rotated key
// pair directly into the (donated) K-cache buffer at `offset`, eliminating
// the rope-temp -> cache copy round-trip. Also carries a trivial flat copy
// kernel used for the non-donatable cache case (mlx's copy_gpu is not
// exported from libmlx).

#include <metal_math>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

constant bool traditional [[function_constant(2)]];

template <typename T, typename IdxT>
void rope_append_impl(
    const device T* k_in,
    device T* k_cache,
    const float inv_freq,
    constant const float& scale,
    constant const int& offset,
    constant const int& dims,
    constant const int& k_dims,
    constant const int64_t& k_cache_mat_stride,
    uint2 pos) {
  int dh = dims / 2;
  int kh = k_dims / 2;
  IdxT m = static_cast<IdxT>(pos.y);
  IdxT off = static_cast<IdxT>(offset);

  if (pos.x < static_cast<uint>(dh)) {
    // Same arithmetic as rope_single (forward direction)
    float L = scale * static_cast<float>(offset);
    float theta = L * inv_freq;
    float costheta = metal::fast::cos(theta);
    float sintheta = metal::fast::sin(theta);
    IdxT i1, i2;
    if (traditional) {
      i1 = 2 * pos.x;
      i2 = i1 + 1;
    } else {
      i1 = pos.x;
      i2 = pos.x + dh;
    }
    IdxT kr = m * static_cast<IdxT>(k_dims);
    IdxT cw = m * static_cast<IdxT>(k_cache_mat_stride) +
        off * static_cast<IdxT>(k_dims);
    float x1 = static_cast<float>(k_in[kr + i1]);
    float x2 = static_cast<float>(k_in[kr + i2]);
    k_cache[cw + i1] = static_cast<T>(x1 * costheta - x2 * sintheta);
    k_cache[cw + i2] = static_cast<T>(x1 * sintheta + x2 * costheta);
  } else if (pos.x < static_cast<uint>(kh)) {
    // Unrotated tail [dims, k_dims) is copied unchanged
    int64_t j = dims + 2 * (static_cast<int>(pos.x) - dh);
    IdxT kr = m * static_cast<IdxT>(k_dims);
    IdxT cw = m * static_cast<IdxT>(k_cache_mat_stride) +
        off * static_cast<IdxT>(k_dims);
    if (j < k_dims) {
      k_cache[cw + j] = k_in[kr + j];
      if (j + 1 < k_dims) {
        k_cache[cw + j + 1] = k_in[kr + j + 1];
      }
    }
  }
}

template <typename T, typename IdxT>
[[kernel]] void omlx_rope_append(
    const device T* k_in [[buffer(0)]],
    device T* k_cache [[buffer(1)]],
    constant const int& offset [[buffer(2)]],
    constant const float& scale [[buffer(3)]],
    constant const int& dims [[buffer(4)]],
    constant const int& k_dims [[buffer(5)]],
    constant const int64_t& k_cache_mat_stride [[buffer(6)]],
    constant const float& base [[buffer(7)]],
    uint2 pos [[thread_position_in_grid]]) {
  float inv_freq = 0.0f;
  if (pos.x < static_cast<uint>(dims / 2)) {
    float d = static_cast<float>(pos.x) / static_cast<float>(dims / 2);
    inv_freq = metal::exp2(-d * base);
  }
  rope_append_impl<T, IdxT>(
      k_in,
      k_cache,
      inv_freq,
      scale,
      offset,
      dims,
      k_dims,
      k_cache_mat_stride,
      pos);
}

template <typename T, typename IdxT>
[[kernel]] void omlx_rope_append_freqs(
    const device T* k_in [[buffer(0)]],
    device T* k_cache [[buffer(1)]],
    constant const int& offset [[buffer(2)]],
    constant const float& scale [[buffer(3)]],
    constant const int& dims [[buffer(4)]],
    constant const int& k_dims [[buffer(5)]],
    constant const int64_t& k_cache_mat_stride [[buffer(6)]],
    const device float* freqs [[buffer(7)]],
    constant const int64_t& freq_stride [[buffer(8)]],
    uint2 pos [[thread_position_in_grid]]) {
  float inv_freq = 0.0f;
  if (pos.x < static_cast<uint>(dims / 2)) {
    inv_freq = 1.0 / (freqs[freq_stride * pos.x]);
  }
  rope_append_impl<T, IdxT>(
      k_in,
      k_cache,
      inv_freq,
      scale,
      offset,
      dims,
      k_dims,
      k_cache_mat_stride,
      pos);
}

// Flat same-dtype copy for the non-donatable cache case.
template <typename T>
[[kernel]] void omlx_flat_copy(
    const device T* src [[buffer(0)]],
    device T* dst [[buffer(1)]],
    constant const int64_t& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint grid [[threads_per_grid]]) {
  for (int64_t i = gid; i < n; i += grid) {
    dst[i] = src[i];
  }
}

// clang-format off
#define instantiate_omlx_rope_append(name, type)                          \
  instantiate_kernel(                                                     \
      "omlx_rope_append_" #name, omlx_rope_append, type, int32_t)         \
  instantiate_kernel(                                                     \
      "omlx_rope_append_freqs_" #name,                                    \
      omlx_rope_append_freqs,                                             \
      type,                                                               \
      int32_t)                                                            \
  instantiate_kernel(                                                     \
      "omlx_rope_append_large_" #name, omlx_rope_append, type, int64_t)   \
  instantiate_kernel(                                                     \
      "omlx_rope_append_freqs_large_" #name,                              \
      omlx_rope_append_freqs,                                             \
      type,                                                               \
      int64_t)                                                            \
  instantiate_kernel("omlx_flat_copy_" #name, omlx_flat_copy, type)

instantiate_omlx_rope_append(float16, half)
instantiate_omlx_rope_append(bfloat16, bfloat16_t)
instantiate_omlx_rope_append(float32, float)
// clang-format on
