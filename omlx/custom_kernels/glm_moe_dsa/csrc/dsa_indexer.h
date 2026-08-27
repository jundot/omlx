#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace omlx::glm_kernels {

mx::array dsa_indexer_scores(
    const mx::array& queries,
    const mx::array& keys,
    const mx::array& weights,
    bool causal = true,
    int unused_causal_prefix_topk = 0,
    bool skip_causal_future_store = false,
    int causal_q_offset = -1,
    // Optional fused pooled-ratio causal mask (0 = disabled, the historical
    // behavior). When mask_ratio > 0, pooled column c is masked for query row
    // r iff c >= (mask_q_offset + r + 1) / mask_ratio, and masked positions
    // are written as finfo(dtype).min in the kernel epilogue — bit-identical
    // to the mx.where pass it replaces.
    int mask_ratio = 0,
    int mask_q_offset = 0,
    mx::StreamOrDevice s = {},
    // Prefer the optional M5/NAX score kernel for its deliberately narrow
    // DS4-Flash prefill domain (bf16, B=1, H=64, D=128, weights [B,L,H],
    // non-causal, ratio-4 pooled mask, L>=16). Every other configuration and
    // any NAX library/pipeline load failure stays on the Steel kernel above.
    bool use_nax = false);

// Availability diagnostics for the optional, separately compiled NAX
// metallib. Hardware eligibility is checked by the Python dispatch through
// the shared mirror of metal::is_nax_available(); these helpers distinguish a
// source-compatible/stale extension from one that actually ships the kernel.
bool dsa_indexer_nax_kernels_built();
bool dsa_indexer_nax_runtime_active();

// v25 M2 from-scratch MMA score kernel (zero-per-head-barrier structure,
// ~1.37x over the Steel kernel on M2 Ultra, bit-exact). Serves ONLY:
// bf16, H=64, D in {48,128}, weights rank 3 ([B, L, H]), non-causal. Callers must
// gate on those and fall back to dsa_indexer_scores otherwise. mask_ratio/
// mask_q_offset carry the same fused pooled-ratio mask semantics as
// dsa_indexer_scores.
mx::array dsa_indexer_scores_mma(
    const mx::array& queries,
    const mx::array& keys,
    const mx::array& weights,
    int mask_ratio = 0,
    int mask_q_offset = 0,
    mx::StreamOrDevice s = {},
    // Default-off physical A/B candidate for the exact DS4 D=128 prefill
    // domain. The production WM2xWN2 route remains the default.
    bool use_wm4_wn1 = false);

mx::array dsa_topk_indices(
    const mx::array& scores,
    int topk,
    bool bucketed = false,
    bool causal_valid_prefix = false,
    mx::StreamOrDevice s = {});

mx::array dspark_fp32_topk_indices(
    const mx::array& scores,
    int topk = 512,
    mx::StreamOrDevice s = {});

mx::array ds4_router_topk_indices(
    const mx::array& scores,
    mx::StreamOrDevice s = {});

// DC-1: fused DECODE indexer scan. One kernel computes the head-summed indexer
// scores for a single query position (s == 1) directly into [B,1,1,S] with fp32
// accumulation — replacing the q@k^T -> relu -> *w -> sum chain that materializes
// four S-sized tensors per layer per token. queries [B,32,1,128] (post-RoPE),
// keys [B,1,S,128] (the indexer k-cache), weights [B,1,32] or [B,32] (scaled).
mx::array dsa_decode_scores(
    const mx::array& queries,
    const mx::array& keys,
    const mx::array& weights,
    bool fp32_scores = false,
    mx::StreamOrDevice s = {});

} // namespace omlx::glm_kernels
