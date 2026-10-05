// SPDX-License-Identifier: Apache-2.0

// Include order is load-bearing: Steel's attention header provides Limits
// used by the specialized Qwen kernels.
// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"
#include "kernels/steel_qwen4_qsa_sparse_gqa_tq.h"
// clang-format on

#define instantiate_qwen4_sparse_gqa_tq(tname, dtype, kb, vb, bk, dc)           \
  instantiate_kernel("qwen4_qsa_sparse_gqa_tq_" #tname "_kb" #kb "_vb" #vb     \
                     "_bk" #bk "_dc" #dc "_gqa12_hp16_d256_wm2",               \
                     qwen4_qsa_sparse_gqa_attention_tq, dtype, kb, vb, bk, dc, \
                     12, 16, 256, 2, uint, float)

// (BK,DC)=(64,64) is the production dispatch for every bit pair; the
// (128,32) tuning variant stays instantiated for the 4-bit pair only.
// Fractional cache widths ride their floor/ceil codec pairs (2.5 -> 2/3,
// 3.5 -> 3/4).
instantiate_qwen4_sparse_gqa_tq(bfloat16, bfloat16_t, 2, 2, 64, 64);
instantiate_qwen4_sparse_gqa_tq(bfloat16, bfloat16_t, 2, 3, 64, 64);
instantiate_qwen4_sparse_gqa_tq(bfloat16, bfloat16_t, 3, 3, 64, 64);
instantiate_qwen4_sparse_gqa_tq(bfloat16, bfloat16_t, 3, 4, 64, 64);
instantiate_qwen4_sparse_gqa_tq(bfloat16, bfloat16_t, 4, 4, 64, 64);
instantiate_qwen4_sparse_gqa_tq(bfloat16, bfloat16_t, 6, 6, 64, 64);
instantiate_qwen4_sparse_gqa_tq(bfloat16, bfloat16_t, 8, 8, 64, 64);
instantiate_qwen4_sparse_gqa_tq(bfloat16, bfloat16_t, 4, 4, 128, 32);
instantiate_qwen4_sparse_gqa_tq(float16, half, 2, 2, 64, 64);
instantiate_qwen4_sparse_gqa_tq(float16, half, 2, 3, 64, 64);
instantiate_qwen4_sparse_gqa_tq(float16, half, 3, 3, 64, 64);
instantiate_qwen4_sparse_gqa_tq(float16, half, 3, 4, 64, 64);
instantiate_qwen4_sparse_gqa_tq(float16, half, 4, 4, 64, 64);
instantiate_qwen4_sparse_gqa_tq(float16, half, 6, 6, 64, 64);
instantiate_qwen4_sparse_gqa_tq(float16, half, 8, 8, 64, 64);
instantiate_qwen4_sparse_gqa_tq(float16, half, 4, 4, 128, 32);
