// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace omlx::decode_fast_kernels {

// True when the decode SDPA kernels apply: Metal stream, 4-D fp32/fp16/bf16
// q/k/v, query length <= 8, supported head dims, GQA fan-out within limits.
bool sdpa_decode_supported(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    mx::StreamOrDevice s = {});

// Decode-mode scaled dot product attention (port of mlx#4294). Returns an
// array of shape (B, H, qL, v_head_dim). Optional mask (broadcastable to
// (B, H, qL, kL), bool or q's dtype) and attention sinks (per query head).
mx::array sdpa_decode(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    float scale,
    bool causal,
    const std::optional<mx::array>& mask = std::nullopt,
    const std::optional<mx::array>& sinks = std::nullopt,
    mx::StreamOrDevice s = {});

} // namespace omlx::decode_fast_kernels
