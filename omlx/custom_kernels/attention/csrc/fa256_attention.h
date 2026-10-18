#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace omlx::attention {

// dispatch_budget bounds the per-dispatch work (B * H * qL * keys) by
// splitting the key axis into separately dispatched chunks combined with
// logsumexp weights; 0 keeps the single-dispatch behavior (issue #2225).
mx::array fa256_attention(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    float scale,
    bool causal = true,
    int q_block = 32,
    int k_block = 8,
    int64_t dispatch_budget = 0,
    mx::StreamOrDevice s = {});

} // namespace omlx::attention
