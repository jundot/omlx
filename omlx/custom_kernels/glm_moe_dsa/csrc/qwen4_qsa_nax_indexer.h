// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"
namespace mx = mlx::core;
namespace omlx::glm_kernels {
// True when the GPU has tensor units (macOS >= 26.2, applegpu gen >= 17) and
// omlx_glm_kernels_nax.metallib was built next to the extension.
bool qwen4_qsa_nax_indexer_available();
// Tensor-unit form of qwen4_qsa_indexer_scores: same inputs ([1,4,M,128] and
// [1,1,N,128] bf16/fp16), same fp32 [1,M,N] output with the pooled causal
// sentinel; fp32 sums differ from the steel kernel only by accumulation order.
mx::array qwen4_qsa_nax_indexer_scores(
    const mx::array& queries,
    const mx::array& pooled_keys,
    int mask_ratio = 4,
    int mask_q_offset = 0,
    int block_rows = 64,
    int block_cols = 64,
    mx::StreamOrDevice s = {});
} // namespace omlx::glm_kernels
