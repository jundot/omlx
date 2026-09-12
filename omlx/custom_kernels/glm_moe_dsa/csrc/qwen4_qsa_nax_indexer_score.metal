// SPDX-License-Identifier: Apache-2.0
// Compiled into omlx_glm_kernels_nax.metallib (-mmacosx-version-min=26.2).
// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm_nax.h"
#include "kernels/nax_qwen4_qsa_indexer_score.h"
// clang-format on
using namespace mlx::steel;
#define instantiate_qwen4_nax_score(tname, dtype, bm, bn, bk, wm, wn)          \
  instantiate_kernel("qwen4_qsa_nax_indexer_score_" #tname "_bm" #bm "_bn" #bn \
                     "_bk" #bk "_wm" #wm "_wn" #wn,                            \
                     qwen4_qsa_nax_indexer_score, dtype, bm, bn, bk, wm, wn)
// 64x64 is the measured best on the M5 Max (3.1 ms per 2048x32768 vs 4.8 steel); 64x128 kept as the alternate.
instantiate_qwen4_nax_score(bfloat16, bfloat16_t, 64, 64, 64, 2, 2);
instantiate_qwen4_nax_score(float16, half, 64, 64, 64, 2, 2);
instantiate_qwen4_nax_score(bfloat16, bfloat16_t, 64, 128, 64, 2, 2);
instantiate_qwen4_nax_score(float16, half, 64, 128, 64, 2, 2);
