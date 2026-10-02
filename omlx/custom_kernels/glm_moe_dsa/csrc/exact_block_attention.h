#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace omlx::glm_kernels {

mx::array glm_dsa_exact_block_attention(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    const mx::array& block_mask,
    const mx::array& block_token_mask,
    float scale,
    bool causal = true,
    mx::StreamOrDevice s = {});

} // namespace omlx::glm_kernels
