// SPDX-License-Identifier: Apache-2.0

// Include order is load-bearing: Steel's attention header provides Limits
// used by the specialized Qwen kernel.
// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"
#include "kernels/steel_qwen4_qsa_sparse_gqa.h"
// clang-format on

#define instantiate_qwen4_sparse_gqa(tname, dtype, bk, dc)                     \
  instantiate_kernel("qwen4_qsa_sparse_gqa_" #tname "_bk" #bk "_dc" #dc        \
                     "_gqa12_hp16_d256_wm2",                                   \
                     qwen4_qsa_sparse_gqa_attention, dtype, bk, dc, 12, 16,    \
                     256, 2, uint, float)

instantiate_qwen4_sparse_gqa(float16, half, 128, 32);
instantiate_qwen4_sparse_gqa(float16, half, 256, 32);
instantiate_qwen4_sparse_gqa(float16, half, 64, 64);
instantiate_qwen4_sparse_gqa(float16, half, 128, 64);
instantiate_qwen4_sparse_gqa(bfloat16, bfloat16_t, 128, 32);
instantiate_qwen4_sparse_gqa(bfloat16, bfloat16_t, 256, 32);
instantiate_qwen4_sparse_gqa(bfloat16, bfloat16_t, 64, 64);
instantiate_qwen4_sparse_gqa(bfloat16, bfloat16_t, 128, 64);
