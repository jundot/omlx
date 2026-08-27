// SPDX-License-Identifier: Apache-2.0
// omlx decode SDPA (vector) kernels — port of ml-explore/mlx#4295's sibling
// PR #4294 (closed unmerged upstream): tiled online softmax, vectorized KV
// loads, context-scaled 2-pass split for 'd'-class GPUs, fp32 partials.
// Kernel bodies live in sdpa_decode_kernels.h (omlx_sdpa_decode*, renamed to
// stay collision-free with mlx's own sdpa_vector kernels).

#include <metal_stdlib>

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "sdpa_decode_kernels.h"

using namespace metal;

#define instantiate_omlx_sdpa_decode_aggregation(type, value_dim) \
  instantiate_kernel(                                             \
      "omlx_sdpa_decode_2pass_2_" #type "_" #value_dim,           \
      omlx_sdpa_decode_2pass_2,                                   \
      type,                                                       \
      value_dim)

#define instantiate_omlx_sdpa_decode(type, qk_dim, value_dim)       \
  instantiate_kernel(                                               \
      "omlx_sdpa_decode_" #type "_" #qk_dim "_" #value_dim,         \
      omlx_sdpa_decode,                                             \
      type,                                                         \
      qk_dim,                                                       \
      value_dim)                                                    \
  instantiate_kernel(                                               \
      "omlx_sdpa_decode_2pass_1_" #type "_" #qk_dim "_" #value_dim, \
      omlx_sdpa_decode_2pass_1,                                     \
      type,                                                         \
      qk_dim,                                                       \
      value_dim)

#define instantiate_omlx_sdpa_decode_heads(type)      \
  instantiate_omlx_sdpa_decode(type, 64, 64)          \
  instantiate_omlx_sdpa_decode(type, 96, 96)          \
  instantiate_omlx_sdpa_decode(type, 128, 128)        \
  instantiate_omlx_sdpa_decode(type, 192, 128)        \
  instantiate_omlx_sdpa_decode(type, 256, 256)        \
  instantiate_omlx_sdpa_decode_aggregation(type, 64)  \
  instantiate_omlx_sdpa_decode_aggregation(type, 96)  \
  instantiate_omlx_sdpa_decode_aggregation(type, 128) \
  instantiate_omlx_sdpa_decode_aggregation(type, 256)

instantiate_omlx_sdpa_decode_heads(float)
instantiate_omlx_sdpa_decode_heads(bfloat16_t)
instantiate_omlx_sdpa_decode_heads(float16_t)
// clang-format on
