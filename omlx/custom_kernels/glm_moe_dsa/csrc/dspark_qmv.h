#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace omlx::glm_kernels {

mx::array dspark_exact_mxfp8_qmv_pair(
    const mx::array& input,
    const mx::array& weight_a,
    const mx::array& scales_a,
    const mx::array& weight_b,
    const mx::array& scales_b,
    mx::StreamOrDevice s = {});

} // namespace omlx::glm_kernels
