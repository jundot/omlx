#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace omlx::quant_matmul {

mx::array q8_head_flat_qmm(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& scales,
    const mx::array& biases,
    mx::StreamOrDevice s = {});

} // namespace omlx::quant_matmul
