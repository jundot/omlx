#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace omlx::glm_kernels {

mx::array dspark_rowwise_gemm(
    const mx::array& lhs,
    const mx::array& rhs,
    bool transpose_rhs,
    mx::StreamOrDevice s = {});

mx::array dspark_ring_gemm(
    const mx::array& lhs,
    const mx::array& source,
    const mx::array& indices,
    bool transpose_rhs,
    mx::StreamOrDevice s = {});

} // namespace omlx::glm_kernels
