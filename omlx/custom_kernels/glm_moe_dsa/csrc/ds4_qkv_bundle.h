#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace omlx::glm_kernels {

mx::array deepseek_v4_qkv_pair_b1(
    const mx::array& x,
    const mx::array& wq_a_weight,
    const mx::array& wq_a_scales,
    const mx::array& wkv_weight,
    const mx::array& wkv_scales,
    mx::StreamOrDevice s = {});

mx::array deepseek_v4_qkv_compressor128_bundle_b1(
    const mx::array& x,
    const mx::array& wq_a_weight,
    const mx::array& wq_a_scales,
    const mx::array& wkv_weight,
    const mx::array& wkv_scales,
    const mx::array& compressor_wkv,
    const mx::array& compressor_wgate,
    mx::StreamOrDevice s = {});

// Isolated DS4 ratio-4 B1 projection bundle. There is no production dispatch.
// The output packs Q-A, raw KV, compressor KV/gate, and index-compressor
// KV/gate in the exact slice order frozen by the benchmark contract.
mx::array deepseek_v4_qkv_compressor_bundle_b1(
    const mx::array& x,
    const mx::array& wq_a_weight,
    const mx::array& wq_a_scales,
    const mx::array& wkv_weight,
    const mx::array& wkv_scales,
    const mx::array& compressor_wkv,
    const mx::array& compressor_wgate,
    const mx::array& index_compressor_wkv,
    const mx::array& index_compressor_wgate,
    mx::StreamOrDevice s = {});

// The promoted implementation uses one heterogeneous dispatch.
int deepseek_v4_qkv_compressor_bundle_b1_dispatches();

} // namespace omlx::glm_kernels
