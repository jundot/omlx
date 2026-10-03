// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace omlx::glm_kernels {

mx::array qwen4_qsa_sparse_gqa_attention_tq(
    const mx::array &queries, const mx::array &key_norms,
    const mx::array &key_packed, const mx::array &value_norms,
    const mx::array &value_packed, const mx::array &codebook_k,
    const mx::array &codebook_v, const mx::array &selected_blocks, float scale,
    int q_offset, int key_tile = 64, int dimension_tile = 64,
    mx::StreamOrDevice s = {});

} // namespace omlx::glm_kernels
