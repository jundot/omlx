// Copyright © 2026 oMLX contributors
// SPDX-License-Identifier: Apache-2.0
//
// NVFP4 (E4M3) decode kernels ported from Layr-Labs/mlxfast-challenge.
// See laguna_nvfp4.metal for the kernel sources and layouts.

#pragma once

#include <memory>
#include <utility>

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

// Routed-expert fused gate/up NVFP4 QMV with in-kernel SwiGLU.
//   input        [K]      bf16 routed-expert input (K = 2048)
//   fused_weight [E, 2N, K/8] uint32 per-expert pair-interleaved gate/up
//                    planes (K/2 bytes/row): 32-row [gate; up] pairs
//                    (N = 512, E = 256)
//   fused_scales [E, 2N, K/16] uint8 E4M3 group scales
//   indices      [R] uint32 top-R routed expert ids (R = 8)
// Returns [R*N] bf16 per-slot silu(gate) * up.
array routed_nvfp4_swiglu_qmv(
    const array& input,
    const array& fused_weight,
    const array& fused_scales,
    const array& indices,
    StreamOrDevice s = {});

// Routed-expert down_proj + weighted reduction fused in one kernel.
//   activated      [R*K2] bf16      per-slot swiglu outputs (R=8, K2=512)
//   down_weight    [E, N, K2/8] uint32 per-expert NVFP4 planes (N=2048)
//   down_scales    [128 + E*N*16] uint8 halved group-32 planes
//   indices        [R] uint32 routed expert ids
//   router_weights [R] float32 routed scores
// Returns [N] bf16 sum_slots(act * w) * 2.5.
array routed_nvfp4_down_reduce(
    const array& activated,
    const array& down_weight,
    const array& down_scales,
    const array& indices,
    const array& router_weights,
    StreamOrDevice s = {});

// Fused Q/K RMSNorm + RoPE. Returns {queries [QH*128], keys [KH*128]}.
// Full-attention variant applies partial-RoPE with the YaRN mscale (48 q + 8 k
// heads, rotary 64); the sliding variant rotates the full 128 dims (64 q + 8 k
// heads).
std::pair<array, array> full_qk_norm_yarn(
    const array& raw_queries,
    const array& raw_keys,
    const array& query_weight,
    const array& key_weight,
    const array& angles,
    StreamOrDevice s = {});

std::pair<array, array> sliding_qk_norm_rope(
    const array& raw_queries,
    const array& raw_keys,
    const array& query_weight,
    const array& key_weight,
    const array& angles,
    StreamOrDevice s = {});

// Fused Q/K/V NVFP4 projection for one decode token (R1, one row per
// simdgroup). rows = (heads + 2*nkv_heads)*128.
//   normalized    [2048] bf16
//   weight_codes  [rows, 1024] uint8
//   weight_scales [rows, 128] uint8
// Returns [rows] bf16.
array decode_nvfp4_qkv_r1(
    const array& normalized,
    const array& weight_codes,
    const array& weight_scales,
    int heads,
    StreamOrDevice s = {});

// Gated affine o_proj (pre-activated per-head gate), fused in one kernel.
//   attention_output [heads*128] bf16
//   gate_values      [heads] bf16
//   weight_codes     [2048, heads*16] uint32
//   weight_scales    [2048, heads*8] uint8
// Returns [2048] bf16.
array oproj_act(
    const array& attention_output,
    const array& gate_values,
    const array& weight_codes,
    const array& weight_scales,
    int heads,
    StreamOrDevice s = {});

// Fused residual add + RMSNorm. Returns {summed, normalized} [2048] bf16.
std::pair<array, array> residual_rms(
    const array& residual,
    const array& branch,
    const array& weight,
    StreamOrDevice s = {});

// Decode router top-8 bitonic tournament. Returns {indices, scores} [8].
std::pair<array, array> decode_router_top8(
    const array& logits,
    const array& correction_bias,
    bool normalizing,
    StreamOrDevice s = {});

// Native extension availability probe for ABI verification.
int64_t abi_probe(const array& a);

}  // namespace omlx::laguna_nvfp4
