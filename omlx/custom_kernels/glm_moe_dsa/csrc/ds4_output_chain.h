#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace omlx::glm_kernels {

// Boundary-probe companion for the chain primitive.  It returns the exact
// BF16 O-A result already laid out as [1,1024,8192], so benchmarks can gate
// the otherwise private intermediate bitwise against the stock graph.
mx::array ds4_output_oa_interleaved(const mx::array &x,
                                    const mx::array &o_a_weight,
                                    const mx::array &o_a_scales,
                                    int variant = 0, mx::StreamOrDevice s = {});

// Isolated exact DeepSeek-V4-Flash M=1024 O-A -> BF16 -> O-B chain.
// The primitive keeps MLX's MXFP8 dequantization, K walk, FP32 accumulation,
// and mandatory BF16 intermediate store.  Its O-A epilogue writes that BF16
// boundary directly in O-B's token-major layout, removing only the stock
// transpose/materialization round trip.
mx::array ds4_output_projection_chain(
    const mx::array &x, const mx::array &o_a_weight,
    const mx::array &o_a_scales, const mx::array &o_b_weight,
    const mx::array &o_b_scales, int variant = 0, mx::StreamOrDevice s = {});

} // namespace omlx::glm_kernels
