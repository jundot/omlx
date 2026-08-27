#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace omlx::glm_kernels {

mx::array ds4_q_head_rms_rope(const mx::array &q, const mx::array &freqs,
                              int offset, float eps,
                              bool return_normalized = false,
                              mx::StreamOrDevice s = {});

mx::array ds4_kv_rms_rope(const mx::array &kv, const mx::array &weight,
                          const mx::array &freqs, int offset, float eps,
                          bool return_normalized = false,
                          mx::StreamOrDevice s = {});

} // namespace omlx::glm_kernels
