#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace omlx::glm_kernels {

// Isolated classic-Metal MXFP8 tile sweep. Production model code does not call
// this symbol; the projection campaign compares it with mx.quantized_matmul.
mx::array ds4_projection_mxfp8_qmm(const mx::array &x, const mx::array &weight,
                                   const mx::array &scales, int variant = 0,
                                   bool use_nax = false, int nax_variant = 0,
                                   mx::StreamOrDevice s = {});

bool ds4_projection_nax_kernels_built();
bool ds4_projection_nax_device_available();

} // namespace omlx::glm_kernels
