#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace omlx::qwen35_prefill_kernels {

mx::array qwen35_fa256_attention(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    float scale,
    bool causal = true,
    int q_block = 32,
    int k_block = 8,
    mx::StreamOrDevice s = {});

mx::array qwen35_q4_affine_qmm_t(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& scales,
    const mx::array& biases,
    int variant = 8,
    mx::StreamOrDevice s = {});

mx::array qwen35_q5_affine_qmm_t(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& scales,
    const mx::array& biases,
    int variant = 8,
    mx::StreamOrDevice s = {});

mx::array qwen35_q6_affine_qmm_t(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& scales,
    const mx::array& biases,
    int variant = 8,
    mx::StreamOrDevice s = {});

mx::array qwen35_q8_affine_qmm_t(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& scales,
    const mx::array& biases,
    int variant = 8,
    mx::StreamOrDevice s = {});

mx::array qwen35_moe_weighted_sum(
    const mx::array& x_sorted,
    const mx::array& inv_order,
    const mx::array& scores,
    mx::StreamOrDevice s = {});

} // namespace omlx::qwen35_prefill_kernels
