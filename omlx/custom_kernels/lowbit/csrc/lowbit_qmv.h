// Copyright © 2026 oMLX contributors
// SPDX-License-Identifier: Apache-2.0
//
// Bonsai t5 (base-3 ternary, ~1.585 bpw) quantized kernels.

#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

#include <optional>

namespace omlx::lowbit {

using mlx::core::array;
using mlx::core::StreamOrDevice;

// ---------------------------------------------------------------------------
// t5: base-3 ternary qmv (Identity I-D, ~1.585 bpw)
// ---------------------------------------------------------------------------
// x      : [..., K]                float16 or bfloat16
// w      : [..., N, n_groups*bpg]  uint8 t5 bytes (bpg=26 for gs=128, 13 for gs=64)
// scales : [..., N, n_groups]      float16 or bfloat16 (no biases — always symmetric)
// Returns [..., N].
array t5_qmv(
    const array& x,
    const array& w,
    const array& scales,
    StreamOrDevice s = {});

array t5_qmv_wide(
    const array& x,
    const array& w,
    const array& scales,
    StreamOrDevice s = {});

// ---------------------------------------------------------------------------
// t5 MMA GEMM (Identity I-M): fused dequant + simdgroup matmul for prefill
// ---------------------------------------------------------------------------
// x      : [M, K]                   float16 or bfloat16
// w      : [N, n_groups * bpg]      uint8 t5 bytes
// scales : [N, n_groups]            float16 or bfloat16
// Returns [M, N].
array t5_qmm(
    const array& x,
    const array& w,
    const array& scales,
    StreamOrDevice s = {});

} // namespace omlx::lowbit
