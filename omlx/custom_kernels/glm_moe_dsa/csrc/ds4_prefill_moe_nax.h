#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace omlx::glm_kernels {

// Isolated M5/NAX DS4-Flash M=1024 routed-MoE projection. The primitive
// consumes the existing expert-local BM32 block plan and preserves stock
// gather_qmm's BF16 output boundary. No model/runtime path calls this symbol.
mx::array deepseek_mxfp4_gather_qmm_blocks_nax(
    const mx::array &x, const mx::array &weight, const mx::array &scales,
    const mx::array &block_meta, const mx::array &block_count,
    mx::StreamOrDevice s = {});

// Two exact projection banks share each BF16 input tile load. The packed
// result is [2,6144,1,1280], preserving an independent BF16 store boundary for
// gate and up while removing one route-input walk and one dispatch.
mx::array deepseek_mxfp4_gather_qmm_pair_blocks_nax(
    const mx::array &x, const mx::array &weight0, const mx::array &scales0,
    const mx::array &weight1, const mx::array &scales1,
    const mx::array &block_meta, const mx::array &block_count,
    mx::StreamOrDevice s = {});

} // namespace omlx::glm_kernels
