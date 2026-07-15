// Copyright © 2026 oMLX contributors
// SPDX-License-Identifier: Apache-2.0
//
// Instantiates only the Bonsai-specific kernel variants that are not present
// in the stock mlx 0.32.0 metallib:
//   - affine_qmv_fast  for bits=1 (1-bit path added by Bonsai MLX fork)
//   - affine_qmv_wide  for bits=1 and bits=2 (new kernel from Bonsai fork)
//
// The vendored quantized.h in this directory is the Bonsai-patched version.
// It must shadow the mlx-installed copy; CMake sets -I for this directory
// before the system mlx include path.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "quantized.h"   // Bonsai-patched: 1-bit qmv_fast + qmv_wide

// ---- instantiation helpers ------------------------------------------------

#define bonsai_instantiate_qmv_fast(type, group_size, bits, batched)       \
  instantiate_kernel(                                                       \
      "affine_qmv_fast_" #type "_gs_" #group_size "_b_" #bits              \
          "_batch_" #batched,                                               \
      affine_qmv_fast, type, group_size, bits, batched)

#define bonsai_instantiate_qmv(type, group_size, bits, batched)            \
  instantiate_kernel(                                                       \
      "affine_qmv_" #type "_gs_" #group_size "_b_" #bits                   \
          "_batch_" #batched,                                               \
      affine_qmv, type, group_size, bits, batched)

#define bonsai_instantiate_qmv_wide(type, group_size, bits, nv, kl, batch) \
  instantiate_kernel(                                                       \
      "affine_qmv_wide_" #type "_gs_" #group_size "_b_" #bits              \
          "_nv_" #nv "_kl_" #kl "_batch_" #batch,                          \
      affine_qmv_wide, type, group_size, bits, nv, kl, batch)

// ---- 1-bit qmv_fast (new in Bonsai fork) ----------------------------------
// group_sizes 32, 64, 128; types float, float16_t, bfloat16_t

#define bonsai_qmv_fast_b1(type, gs) \
  bonsai_instantiate_qmv_fast(type, gs, 1, 0) \
  bonsai_instantiate_qmv_fast(type, gs, 1, 1) \
  bonsai_instantiate_qmv(type, gs, 1, 0) \
  bonsai_instantiate_qmv(type, gs, 1, 1)

#define bonsai_qmv_fast_b1_types(gs) \
  bonsai_qmv_fast_b1(float, gs) \
  bonsai_qmv_fast_b1(float16_t, gs) \
  bonsai_qmv_fast_b1(bfloat16_t, gs)

bonsai_qmv_fast_b1_types(32)
bonsai_qmv_fast_b1_types(64)
bonsai_qmv_fast_b1_types(128)

// ---- qmv_wide: bits=1 and bits=2 -----------------------------------------
// vecs_per_tg 2..5; k_lanes=8 for affine mode; batch 0 and 1

#define bonsai_qmv_wide_bit(type, gs, bits) \
  bonsai_instantiate_qmv_wide(type, gs, bits, 2, 8, 0) \
  bonsai_instantiate_qmv_wide(type, gs, bits, 2, 8, 1) \
  bonsai_instantiate_qmv_wide(type, gs, bits, 3, 8, 0) \
  bonsai_instantiate_qmv_wide(type, gs, bits, 3, 8, 1) \
  bonsai_instantiate_qmv_wide(type, gs, bits, 4, 8, 0) \
  bonsai_instantiate_qmv_wide(type, gs, bits, 4, 8, 1) \
  bonsai_instantiate_qmv_wide(type, gs, bits, 5, 8, 0) \
  bonsai_instantiate_qmv_wide(type, gs, bits, 5, 8, 1)

#define bonsai_qmv_wide_types(gs, bits) \
  bonsai_qmv_wide_bit(float, gs, bits) \
  bonsai_qmv_wide_bit(float16_t, gs, bits) \
  bonsai_qmv_wide_bit(bfloat16_t, gs, bits)

// bits=1
bonsai_qmv_wide_types(32, 1)
bonsai_qmv_wide_types(64, 1)
bonsai_qmv_wide_types(128, 1)

// bits=2
bonsai_qmv_wide_types(32, 2)
bonsai_qmv_wide_types(64, 2)
bonsai_qmv_wide_types(128, 2)
// clang-format on
