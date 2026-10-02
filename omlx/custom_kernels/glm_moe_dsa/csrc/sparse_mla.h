#pragma once

#include <optional>

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace omlx::glm_kernels {

mx::array glm_dsa_sparse_mla_attention(
    const mx::array& q_latent,
    const mx::array& q_pe,
    const mx::array& kv_latent,
    const mx::array& k_pe,
    const mx::array& topk_indices,
    float scale,
    bool causal = true,
    bool topk_valid_prefix = false,
    bool causal_prefix_indices = false,
    const std::optional<mx::array>& topk_length = std::nullopt,
    int causal_prefix_rows = 0,
    mx::StreamOrDevice s = {});

} // namespace omlx::glm_kernels
