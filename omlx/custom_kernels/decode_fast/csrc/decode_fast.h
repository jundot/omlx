// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vector>

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace omlx::decode_fast_kernels {

// True when the fused kernels apply: Metal stream, floating dtype, 1-D
// weight matching the last axis, residual shape identical to x.
bool rms_norm_residual_supported(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& residual,
    mx::StreamOrDevice s = {});

// Fused residual add + RMS norm: returns {rms_norm(x + residual) * weight,
// x + residual} in a single Metal dispatch (port of mlx#4295).
std::vector<mx::array> rms_norm_residual(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& residual,
    float eps,
    mx::StreamOrDevice s = {});

} // namespace omlx::decode_fast_kernels
