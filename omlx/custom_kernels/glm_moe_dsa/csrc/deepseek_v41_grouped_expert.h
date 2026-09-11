#pragma once
#include "mlx/array.h"
namespace omlx::glm_kernels {
mlx::core::array deepseek_v41_grouped_expert(
    const mlx::core::array& gate, const mlx::core::array& up,
    const mlx::core::array& activation, const mlx::core::array& down);
}
