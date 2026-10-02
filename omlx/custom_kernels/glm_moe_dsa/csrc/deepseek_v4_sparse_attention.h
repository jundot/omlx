#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace omlx::glm_kernels {

mx::array deepseek_v4_sparse_attention(
    const mx::array& q,
    const mx::array& local_kv,
    const mx::array& pooled,
    const mx::array& topk_indices,
    const mx::array& sinks,
    float scale,
    int q_offset,
    int compress_ratio,
    int local_window,
    mx::StreamOrDevice s = {});

} // namespace omlx::glm_kernels
