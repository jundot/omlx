// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace omlx::decode_fast_kernels {

// True when the fused sparse decode attention kernel applies: Metal stream,
// q [B,H,1,512] fp32/fp16/bf16, local_kv [B,1,W,512], pooled [B,1,P,512],
// float32 sinks [H], 1 <= W+P <= 1024, B <= 8.
bool sparse_attn_decode_supported(
    const mx::array& q,
    const mx::array& local_kv,
    const mx::array& pooled,
    const mx::array& sinks,
    mx::StreamOrDevice s = {});

// Fused windowed-local + selected-pooled decode attention with sinks, for
// DeepSeek-V4-style shared K=V MLA layers. Returns [B,H,1,512] in q's dtype.
mx::array sparse_attn_decode(
    const mx::array& q,
    const mx::array& local_kv,
    const mx::array& pooled,
    const mx::array& sinks,
    mx::StreamOrDevice s = {});

} // namespace omlx::decode_fast_kernels
