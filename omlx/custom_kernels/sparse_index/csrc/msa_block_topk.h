#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace omlx::sparse_index {

mx::array msa_block_topk(
    const mx::array& idx_queries,
    const mx::array& idx_keys,
    int q_start,
    float scale,
    int block_size,
    int topk,
    int init_blocks,
    int local_blocks,
    mx::StreamOrDevice s = {});

} // namespace omlx::sparse_index
