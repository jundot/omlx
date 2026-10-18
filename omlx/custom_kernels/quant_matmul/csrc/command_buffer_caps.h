#pragma once

#include <tuple>

namespace omlx::quant_matmul {

// Sets MLX's per-command-buffer caps (ops, MB of inputs) on the GPU device and
// returns the previous ones. MLX reads them on every commit decision.
std::tuple<int, int> set_command_buffer_caps(int ops, int mb);

} // namespace omlx::quant_matmul
