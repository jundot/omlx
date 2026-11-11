#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace omlx::quant_matmul {

// Top-k values the weighted sum kernels are instantiated for.
inline constexpr int kMoeWeightedSumTopKs[] = {4, 6, 8, 10};

mx::array moe_weighted_sum(
    const mx::array& x_sorted,
    const mx::array& inv_order,
    const mx::array& scores,
    mx::StreamOrDevice s = {});

} // namespace omlx::quant_matmul
