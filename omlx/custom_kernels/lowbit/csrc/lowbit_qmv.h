// Copyright © 2026 oMLX contributors
// SPDX-License-Identifier: Apache-2.0
//
// Bonsai 1-bit / 2-bit affine quantized decode kernels.
//
// These wrap Metal kernels ported from the Bonsai MLX fork
// (github.com/PrismML-Eng/Bonsai-demo) which added:
//   - 1-bit qmv_fast support to mlx's quantized.h
//   - qmv_wide: small-batch (M=2..5) reuse kernel for 1/2-bit affine

#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

#include <optional>

namespace omlx::lowbit {

using mlx::core::array;
using mlx::core::StreamOrDevice;

// ---------------------------------------------------------------------------
// 1-bit affine qmv_fast (M = 1)
// ---------------------------------------------------------------------------
// x      : [..., K]         float16 or bfloat16
// w      : [..., N, K/8]    packed 1-bit weights (uint8)
// scales : [..., N, K/gs]   float16 or bfloat16
// biases : [..., N, K/gs]   float16 or bfloat16
// Returns [..., N].
array q1_affine_qmv(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    StreamOrDevice s = {});

// ---------------------------------------------------------------------------
// 2-bit affine qmv_fast (M = 1)
// ---------------------------------------------------------------------------
// Same layout as q1_affine_qmv but bits=2 (w has K/4 packed uint8 per row).
array q2_affine_qmv(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    StreamOrDevice s = {});

// ---------------------------------------------------------------------------
// 1-bit affine qmv_wide (M = 2..5)
// ---------------------------------------------------------------------------
// Same layout as q1_affine_qmv but uses the wide kernel for small
// batch, amortising weight loads across all M vectors.
// Caller is responsible for the routing decision (use_qmv_wide gate).
array q1_affine_qmv_wide(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    StreamOrDevice s = {});

// ---------------------------------------------------------------------------
// 2-bit affine qmv_wide (M = 2..5)
// ---------------------------------------------------------------------------
// Same layout as above but bits=2 (w has K/4 packed uint8 per row).
// Caller is responsible for the routing decision (use_qmv_wide gate).
array q2_affine_qmv_wide(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    StreamOrDevice s = {});

array q1_affine_qmv_sym(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    StreamOrDevice s = {});

array q2_affine_qmv_sym(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    StreamOrDevice s = {});

array q1_affine_qmv_wide_sym(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    StreamOrDevice s = {});

array q2_affine_qmv_wide_sym(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    StreamOrDevice s = {});

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
