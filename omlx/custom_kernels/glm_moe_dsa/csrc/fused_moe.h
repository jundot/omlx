#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace omlx::glm_kernels {

mx::array glm_dsa_q8_vup_flat(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& scales,
    const mx::array& biases,
    mx::StreamOrDevice s = {});

mx::array glm_moe_weighted_sum(
    const mx::array& x_sorted,
    const mx::array& inv_order,
    const mx::array& scores,
    mx::StreamOrDevice s = {});

mx::array deepseek_mxfp4_gather_qmm_blocks(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& scales,
    const mx::array& block_meta,
    const mx::array& block_count,
    int variant = 0,
    mx::StreamOrDevice s = {});

mx::array deepseek_mxfp4_gather_qmm_pair_blocks(
    const mx::array& x,
    const mx::array& weight0,
    const mx::array& scales0,
    const mx::array& weight1,
    const mx::array& scales1,
    const mx::array& block_meta,
    const mx::array& block_count,
    int variant = 0,
    mx::StreamOrDevice s = {});

mx::array deepseek_mxfp4_gather_qmm_pair_concat_blocks(
    const mx::array& x,
    const mx::array& weight0,
    const mx::array& scales0,
    const mx::array& weight1,
    const mx::array& scales1,
    const mx::array& block_meta,
    const mx::array& block_count,
    int variant = 0,
    mx::StreamOrDevice s = {});

mx::array deepseek_affine_gather_qmm_blocks(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& scales,
    const mx::array& biases,
    const mx::array& block_meta,
    const mx::array& block_count,
    int group_size,
    int bits,
    int variant = 0,
    mx::StreamOrDevice s = {});

mx::array deepseek_affine_gather_qmm_pair_concat_blocks(
    const mx::array& x,
    const mx::array& weight0,
    const mx::array& scales0,
    const mx::array& biases0,
    const mx::array& weight1,
    const mx::array& scales1,
    const mx::array& biases1,
    const mx::array& block_meta,
    const mx::array& block_count,
    int group_size,
    int bits,
    int variant = 0,
    mx::StreamOrDevice s = {});

mx::array deepseek_mxfp4_gather_qmm_expert(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& scales,
    const mx::array& indices,
    int variant = 0,
    mx::StreamOrDevice s = {});

} // namespace omlx::glm_kernels
