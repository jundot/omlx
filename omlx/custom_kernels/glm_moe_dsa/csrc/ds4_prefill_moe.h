#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace omlx::glm_kernels {

// Isolated DS4-Flash phase-A experiment only. This symbol has no production
// dispatch site and deliberately accepts only the measured equal-TP2 shape:
// FP16 x [6144,1,4096], two MXFP4 expert tables [256,1024,4096/8], BM32 block
// metadata, and LimitedSwiGLU limit 10. It returns FP16 [6144,1,1024].
mx::array deepseek_mxfp4_gather_qmm_pair_swiglu_blocks(
    const mx::array& x,
    const mx::array& up_weight,
    const mx::array& up_scales,
    const mx::array& gate_weight,
    const mx::array& gate_scales,
    const mx::array& block_meta,
    const mx::array& block_count,
    float activation_limit = 10.0f,
    int variant = 2,
    mx::StreamOrDevice s = {});

// Isolated tail-cull follow-up for exact DS4 TP widths 768/1024/1280. The BM32
// expert plan and all dtype/arithmetic boundaries stay unchanged; BM8 route
// microtiles only skip padded-row MMA.
mx::array deepseek_mxfp4_gather_qmm_pair_swiglu_blocks_tail8(
    const mx::array& x,
    const mx::array& up_weight,
    const mx::array& up_scales,
    const mx::array& gate_weight,
    const mx::array& gate_scales,
    const mx::array& block_meta,
    const mx::array& block_count,
    float activation_limit = 10.0f,
    int variant = 2,
    mx::StreamOrDevice s = {});

mx::array deepseek_mxfp4_gather_qmm_blocks_tail8(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& scales,
    const mx::array& block_meta,
    const mx::array& block_count,
    int variant = 2,
    mx::StreamOrDevice s = {});

} // namespace omlx::glm_kernels
