#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace omlx::glm_kernels {

mx::array dsa_indexer_scores(
    const mx::array& queries,
    const mx::array& keys,
    const mx::array& weights,
    bool causal = true,
    int unused_causal_prefix_topk = 0,
    bool skip_causal_future_store = false,
    int causal_q_offset = -1,
    mx::StreamOrDevice s = {});

mx::array dsa_topk_indices(
    const mx::array& scores,
    int topk,
    bool bucketed = false,
    bool causal_valid_prefix = false,
    mx::StreamOrDevice s = {});

} // namespace omlx::glm_kernels
