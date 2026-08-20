// Copyright © 2026 oMLX contributors
// SPDX-License-Identifier: Apache-2.0
//
// NVFP4 (E4M3) decode kernels ported from Layr-Labs/mlxfast-challenge.
// See laguna_nvfp4.metal for the kernel sources and layouts.

#pragma once

#include <memory>

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace omlx::laguna_nvfp4 {

using mlx::core::array;
using mlx::core::StreamOrDevice;

// Shared-expert fused gate/up NVFP4 QMV with in-kernel SwiGLU.
//   x      [K]      bf16 activations (K = 2048)
//   w      [2N, K/2] uint8 fused gate/up NVFP4 codes (N = 512)
//   scales [2N, K/16] uint8 E4M3 group scales
// Returns [N] bf16 silu(gate) * up.
array shared_nvfp4_swiglu_qmv(
    const array& x,
    const array& w,
    const array& scales,
    StreamOrDevice s = {});

// Shared-expert down_proj with routed + residual adds fused in one kernel.
//   activated   [K2]      bf16 swiglu output (K2 = 512)
//   down_weight [N, K2/2] uint8 NVFP4 codes (N = 2048)
//   down_scales [N, K2/16] uint8 E4M3 group scales
//   routed      [N]       bf16 routed-expert output
//   residual    [N]       bf16 decoder residual
// Returns [N] bf16 residual + (routed + shared).
array shared_nvfp4_down_residual(
    const array& activated,
    const array& down_weight,
    const array& down_scales,
    const array& routed,
    const array& residual,
    StreamOrDevice s = {});

// Native extension availability probe for ABI verification.
int64_t abi_probe(const array& a);

}  // namespace omlx::laguna_nvfp4
