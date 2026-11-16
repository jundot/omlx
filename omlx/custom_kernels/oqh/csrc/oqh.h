// Copyright © 2026 oMLX contributors
// SPDX-License-Identifier: Apache-2.0
//
// oQh kernels: w = scale_row * (q - (2^bits - 1) / 2), uniform codes in MLX
// affine packing, weights in a signed block Walsh-Hadamard basis on the input
// axis. Producers write their output in the same basis.

#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

#include <optional>
#include <vector>

namespace omlx::oqh {

using mlx::core::array;
using mlx::core::StreamOrDevice;

// x: [..., K] float16/bfloat16, w: [N, K * bits / 32] uint32, rscale: [N].
// Returns [..., N]. Every row of x is one matrix-vector product.
array qmv(const array& x, const array& w, const array& rscale, int bits, StreamOrDevice s = {});

// Routed experts. x: [R, K] with R == 1 (shared by every pair) or R == P,
// w: [E, N, K * bits / 32], rscale: [E, N], indices: [P] expert ids. Returns [P, N].
array gather_qmv(
    const array& x,
    const array& w,
    const array& rscale,
    const array& indices,
    int bits,
    StreamOrDevice s = {});

// RMSNorm(x) * weight, then the signed Hadamard of each block of the row.
// Returns {rotated} or {rotated, plain}. D % 256 == 0, D <= 8192, block >= 256.
std::vector<array> rmsnorm_had(
    const array& x,
    const array& weight,
    const array& signs,
    float eps,
    int block,
    bool dual,
    StreamOrDevice s = {});

// h = x + r, then rmsnorm_had(h). Returns {h, rotated} or {h, rotated, plain}.
std::vector<array> add_rmsnorm_had(
    const array& x,
    const array& r,
    const array& weight,
    const array& signs,
    float eps,
    int block,
    bool dual,
    StreamOrDevice s = {});

// mode 0: silu(a) * b, 1: a * sigmoid(b), 3: a; then the rotation.
array act_had(
    const array& a,
    const std::optional<array>& b,
    const array& signs,
    int mode,
    int block,
    StreamOrDevice s = {});

// y: [..., 2L] holding gate | up halves of a fused projection. Returns the
// rotated silu(gate) * up, [..., L]; with a limit, the clamped SwiGLU
// silu(min(gate, limit)) * clamp(up, -limit, limit).
array act_split_had(
    const array& y,
    const array& signs,
    int block,
    std::optional<float> limit = std::nullopt,
    StreamOrDevice s = {});

// Gated RMSNorm over 128-wide heads (Qwen3-Next GDN), then the rotation of each row.
// Returns {rotated} or, when dual, {rotated, plain}.
std::vector<array> gnorm_had(
    const array& x,
    const array& z,
    const array& weight,
    const array& signs,
    float eps,
    int block,
    bool dual = false,
    StreamOrDevice s = {});

} // namespace omlx::oqh
