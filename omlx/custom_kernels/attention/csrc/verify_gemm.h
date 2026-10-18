#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace omlx::attention {

mx::array rowwise_gemm(
    const mx::array& lhs,
    const mx::array& rhs,
    bool transpose_rhs,
    mx::StreamOrDevice s = {});

mx::array ring_gemm(
    const mx::array& lhs,
    const mx::array& source,
    const mx::array& indices,
    bool transpose_rhs,
    mx::StreamOrDevice s = {});

} // namespace omlx::attention
