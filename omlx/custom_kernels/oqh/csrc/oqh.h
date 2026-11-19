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
// Returns [..., N]. Every row of x is one matrix-vector product. With w_b and
// rscale_b (a second matrix of the same shape, e.g. up next to gate) the result
// is [..., 2N] with w's rows first.
array qmv(
    const array& x,
    const array& w,
    const array& rscale,
    int bits,
    const std::optional<array>& w_b = std::nullopt,
    const std::optional<array>& rscale_b = std::nullopt,
    StreamOrDevice s = {});

// Routed experts. x: [R, K] with R == 1 (shared by every pair) or R == P,
// w: [E, N, K * bits / 32], rscale: [E, N], indices: [P] expert ids. Returns
// [P, N], or [P, 2N] with w_b / rscale_b as in qmv.
array gather_qmv(
    const array& x,
    const array& w,
    const array& rscale,
    const array& indices,
    int bits,
    const std::optional<array>& w_b = std::nullopt,
    const std::optional<array>& rscale_b = std::nullopt,
    StreamOrDevice s = {});

// RMSNorm(x) * weight as MLX rms_norm rounds it (plain), and plain rotated in
// 256-wide blocks. Returns {rotated, plain}. D % 256 == 0, D <= 8192.
std::vector<array> rmsnorm_had(
    const array& x,
    const array& weight,
    const array& signs,
    float eps,
    StreamOrDevice s = {});

// The rotation alone, for an activation that no producer rotated.
array rotate(const array& x, const array& signs, int block, StreamOrDevice s = {});

// y: [..., 2L] holding gate | up halves. Returns the rotated silu(gate) * up,
// [..., L]; with a limit, the clamped SwiGLU
// silu(min(gate, limit)) * clamp(up, -limit, limit).
array act_split_had(
    const array& y,
    const array& signs,
    int block,
    std::optional<float> limit = std::nullopt,
    StreamOrDevice s = {});

// act_split_had for gate and up as two [..., L] inputs.
array act_pair_had(
    const array& a,
    const array& b,
    const array& signs,
    int block,
    std::optional<float> limit = std::nullopt,
    StreamOrDevice s = {});

// Gated RMSNorm over 128-wide heads (Qwen3-Next GDN), then the rotation of each
// row. Returns {rotated, plain}.
std::vector<array> gnorm_had(
    const array& x,
    const array& z,
    const array& weight,
    const array& signs,
    float eps,
    int block,
    StreamOrDevice s = {});

} // namespace omlx::oqh
