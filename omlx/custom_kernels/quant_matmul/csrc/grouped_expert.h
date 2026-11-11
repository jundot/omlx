#pragma once
#include "mlx/array.h"
namespace omlx::quant_matmul {
mlx::core::array grouped_expert(
    const mlx::core::array& gate, const mlx::core::array& up,
    const mlx::core::array& activation, const mlx::core::array& down);
}
