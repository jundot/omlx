// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <vector>

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace omlx::decode_fast_kernels {

// True when the fused RoPE+append kernel applies: Metal stream, 4-D dense
// bf16/fp16/fp32, single-token decode, matching shapes, in-range offset.
bool rope_kv_append_supported(
    const mx::array& keys,
    const mx::array& values,
    const mx::array& key_cache,
    const mx::array& value_cache,
    int offset,
    int dims,
    mx::StreamOrDevice s = {});

// Fused RoPE(K) + K-cache append plus a V-cache slice update (port of
// mlx#4297). Returns {key_cache_updated, value_cache_updated}; the K update
// is done in place in the donated cache buffer when possible.
std::vector<mx::array> rope_kv_append(
    const mx::array& keys,
    const mx::array& values,
    const mx::array& key_cache,
    const mx::array& value_cache,
    int offset,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    const std::optional<mx::array>& freqs = std::nullopt,
    mx::StreamOrDevice s = {});

} // namespace omlx::decode_fast_kernels
