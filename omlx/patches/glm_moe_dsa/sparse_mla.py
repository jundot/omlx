# Copyright © 2026 Apple Inc.

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import mlx.core as mx

from .kernels import fast as glm_fast


@lru_cache(maxsize=None)
def _make_topk_indices_to_block_masks_kernel():
    if not mx.metal.is_available():
        return None

    source = r"""
        const uint elem = thread_position_in_grid.x;
        const uint total = B * L * TOPK;
        if (elem >= total) {
          return;
        }

        const uint j = elem % TOPK;
        const uint q_pos = (elem / TOPK) % L;
        const uint b = elem / (TOPK * L);

        const uint q_abs = (K - L) + q_pos;
        if (CAUSAL_PREFIX_INDICES && q_pos < PREFIX_ROWS) {
          const uint valid_length = min(uint(K), q_abs + 1);
          const uint prefix_blocks =
              (valid_length + uint(K_BLOCK) - 1) / uint(K_BLOCK);
          if (j >= prefix_blocks) {
            return;
          }

          const uint q_block = q_pos / Q_BLOCK;
          const uint k_block = j;
          if (q_block >= Q_BLOCKS || k_block >= K_BLOCKS) {
            return;
          }

          const uint block_idx = (b * Q_BLOCKS + q_block) * K_BLOCKS + k_block;
          block_mask[block_idx] = true;

          const uint block_start = k_block * uint(K_BLOCK);
          const uint tokens_in_block =
              min(uint(K_BLOCK), valid_length - block_start);
          const uint full_bits =
              K_BLOCK == 32 ? 0xffffffffu : ((1u << K_BLOCK) - 1u);
          const uint bits = tokens_in_block == K_BLOCK
              ? full_bits
              : ((1u << tokens_in_block) - 1u);
          const uint token_idx = (b * L + q_pos) * K_BLOCKS + k_block;
          block_token_mask[token_idx] = bits;
          return;
        }

        const uint topk_row = COMPACT_PREFIX_TOPK ? q_pos - PREFIX_ROWS : q_pos;
        const uint topk_elem = (b * TOPK_ROWS + topk_row) * TOPK + j;
        const uint k_pos = topk_indices[topk_elem];
        if (k_pos >= K) {
          return;
        }

        if (CAUSAL && k_pos > q_abs) {
          return;
        }

        const uint q_block = q_pos / Q_BLOCK;
        const uint k_block = k_pos / K_BLOCK;
        if (q_block >= Q_BLOCKS || k_block >= K_BLOCKS) {
          return;
        }

        const uint block_idx = (b * Q_BLOCKS + q_block) * K_BLOCKS + k_block;
        block_mask[block_idx] = true;

        const uint bit = 1u << (k_pos - k_block * K_BLOCK);
        const uint token_idx = (b * L + q_pos) * K_BLOCKS + k_block;
        device atomic_uint* atomic_token_mask =
            reinterpret_cast<device atomic_uint*>(block_token_mask);
        atomic_fetch_or_explicit(
            &atomic_token_mask[token_idx],
            bit,
            memory_order_relaxed);
    """

    return glm_fast.metal_kernel(
        name="glm_dsa_topk_indices_to_block_masks",
        input_names=["topk_indices"],
        output_names=["block_mask", "block_token_mask"],
        source=source,
    )


def topk_indices_to_block_masks(
    topk_indices: mx.array,
    *,
    L: Optional[int] = None,
    K: int,
    q_block_size: int = 32,
    k_block_size: int = 16,
    causal: bool = True,
    causal_prefix_indices: bool = False,
    causal_prefix_rows: int = 0,
    stream: Optional[mx.Stream] = None,
) -> Optional[tuple[mx.array, mx.array]]:
    """Build exact block and per-token masks for GLM DSA top-k attention."""

    if (
        topk_indices.ndim != 4
        or topk_indices.shape[1] != 1
        or k_block_size <= 0
        or k_block_size > 32
        or q_block_size <= 0
    ):
        return None

    B, _, L_in, topk = topk_indices.shape
    L = L_in if L is None else L
    causal_prefix_rows = max(0, min(causal_prefix_rows, L))
    compact_prefix_topk = (
        causal_prefix_indices
        and causal_prefix_rows > 0
        and L_in == L - causal_prefix_rows
    )
    if (L != L_in and not compact_prefix_topk) or K <= 0 or topk <= 0:
        return None

    q_blocks = (L + q_block_size - 1) // q_block_size
    k_blocks = (K + k_block_size - 1) // k_block_size

    kernel = _make_topk_indices_to_block_masks_kernel()
    if kernel is None:
        return None

    block_mask, block_token_mask = kernel(
        inputs=[topk_indices.astype(mx.uint32)],
        template=[
            ("B", B),
            ("L", L),
            ("K", K),
            ("TOPK", topk),
            ("TOPK_ROWS", L_in),
            ("Q_BLOCK", q_block_size),
            ("K_BLOCK", k_block_size),
            ("Q_BLOCKS", q_blocks),
            ("K_BLOCKS", k_blocks),
            ("CAUSAL", causal),
            ("CAUSAL_PREFIX_INDICES", causal_prefix_indices),
            ("PREFIX_ROWS", causal_prefix_rows),
            ("COMPACT_PREFIX_TOPK", compact_prefix_topk),
        ],
        grid=(B * L * topk, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[
            (B, 1, q_blocks, k_blocks),
            (B, 1, L, k_blocks),
        ],
        output_dtypes=[mx.bool_, mx.uint32],
        init_value=0,
        stream=stream or mx.gpu,
    )
    return block_mask, block_token_mask


def exact_block_token_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    topk_indices: mx.array,
    scale: float,
    *,
    q_block_size: int = 32,
    k_block_size: int = 8,
    causal_prefix_indices: bool = False,
    causal_prefix_rows: int = 0,
    stream: Optional[mx.Stream] = None,
) -> Optional[mx.array]:
    """Run fork-default exact block-token SDPA when the native op is available."""

    if (
        not hasattr(glm_fast, "glm_dsa_exact_block_attention")
        or q.ndim != 4
        or k.ndim != 4
        or v.ndim != 4
        or topk_indices.ndim != 4
        or q.shape[:2] != v.shape[:2]
        or k.shape[:3] != v.shape[:3]
        or q.shape[-1] != k.shape[-1]
        or q.shape[-1] != v.shape[-1]
        or q.shape[-1] != 256
        or q.shape[2] <= 8
    ):
        return None

    block_masks = topk_indices_to_block_masks(
        topk_indices,
        L=q.shape[2],
        K=k.shape[2],
        q_block_size=q_block_size,
        k_block_size=k_block_size,
        causal_prefix_indices=causal_prefix_indices,
        causal_prefix_rows=causal_prefix_rows,
        stream=stream,
    )
    if block_masks is None:
        return None
    block_mask, block_token_mask = block_masks
    try:
        return glm_fast.glm_dsa_exact_block_attention(
            q,
            k,
            v,
            block_mask,
            block_token_mask,
            scale,
            causal=True,
            stream=stream or mx.gpu,
        )
    except (AttributeError, RuntimeError, ValueError):
        return None


@lru_cache(maxsize=None)
def _make_index_score_reduce_kernel():
    if not mx.metal.is_available():
        return None

    source = r"""
        const uint elem = thread_position_in_grid.x;
        const uint total = B * L * K;
        if (elem >= total) {
          return;
        }

        const uint k_pos = elem % K;
        const uint q_pos = (elem / K) % L;
        const uint b = elem / (L * K);

        if (CAUSAL && k_pos > K - L + q_pos) {
          out[elem] = static_cast<T>(-INFINITY);
          return;
        }

        float acc = 0.0f;
        #pragma clang loop unroll(full)
        for (uint h = 0; h < H; ++h) {
          const uint score_idx = ((b * H + h) * L + q_pos) * K + k_pos;
          const uint weight_idx = (b * H + h) * L + q_pos;
          const float s = static_cast<float>(head_scores[score_idx]);
          const float w = static_cast<float>(weights[weight_idx]);
          acc += metal::max(s, 0.0f) * w;
        }
        out[elem] = static_cast<T>(acc);
    """

    return glm_fast.metal_kernel(
        name="glm_dsa_index_score_reduce",
        input_names=["head_scores", "weights"],
        output_names=["out"],
        source=source,
    )


def fused_index_score_reduce(
    head_scores: mx.array,
    weights: mx.array,
    *,
    causal: bool = False,
    stream: Optional[mx.Stream] = None,
) -> Optional[mx.array]:
    """Fuse ReLU, per-head weighting, causal fill, and head reduction."""

    kernel = _make_index_score_reduce_kernel()
    if (
        kernel is None
        or head_scores.ndim != 4
        or weights.ndim != 4
        or weights.shape[-1] != 1
        or head_scores.shape[:3] != weights.shape[:3]
    ):
        return None

    B, H, L, K = head_scores.shape
    short_k_threadgroup = 512
    long_k_threshold = 32768
    long_k_threadgroup = 256
    threadgroup_size = (
        long_k_threadgroup if K > long_k_threshold else short_k_threadgroup
    )
    return kernel(
        inputs=[head_scores, weights],
        template=[
            ("T", head_scores.dtype),
            ("B", B),
            ("H", H),
            ("L", L),
            ("K", K),
            ("CAUSAL", causal),
        ],
        grid=(B * L * K, 1, 1),
        threadgroup=(threadgroup_size, 1, 1),
        output_shapes=[(B, 1, L, K)],
        output_dtypes=[head_scores.dtype],
        stream=stream or mx.gpu,
    )[0]


def fused_indexer_scores(
    queries: mx.array,
    keys: mx.array,
    weights: mx.array,
    *,
    causal: bool = False,
    unused_causal_prefix_topk: int = 0,
    skip_causal_future_store: bool = False,
    causal_q_offset: int = -1,
    stream: Optional[mx.Stream] = None,
) -> Optional[mx.array]:
    """Compute GLM DSA indexer logits without materializing per-head scores.

    This is the MLX equivalent of the vLLM/SGLang MQA-logits indexer path:
    the Steel kernel computes ``sum_h relu(q_h @ k.T) * weight_h`` directly
    into [B, 1, L, K]. The Metal kernel is specialized for GLM-5.2's
    [H=32, D=128] indexer and 64-token M/N tiles, so non-multiple prompt
    lengths are padded and sliced back exactly.
    """

    if (
        not hasattr(glm_fast, "dsa_indexer_scores")
        or queries.ndim != 4
        or keys.ndim != 4
        or weights.ndim != 3
        or queries.shape[0] != keys.shape[0]
        or queries.shape[0] != weights.shape[0]
        or queries.shape[1] != 32
        or keys.shape[1] != 1
        or queries.shape[2] != weights.shape[1]
        or queries.shape[1] != weights.shape[2]
        or queries.shape[3] != 128
        or keys.shape[3] != 128
        or keys.shape[2] < 4096
        or queries.dtype != keys.dtype
        or queries.dtype != weights.dtype
    ):
        return None

    B, H, L, D = queries.shape
    K = keys.shape[2]
    q_pad = (-L) % 64
    k_pad = (-K) % 64
    if causal and causal_q_offset < 0 and (q_pad or k_pad):
        causal_q_offset = K - L

    q = queries
    k = keys
    w = weights
    if q_pad:
        q = mx.pad(q, [(0, 0), (0, 0), (0, q_pad), (0, 0)])
        w = mx.pad(w, [(0, 0), (0, q_pad), (0, 0)])
    if k_pad:
        k = mx.pad(k, [(0, 0), (0, 0), (0, k_pad), (0, 0)])
    if q_pad or k_pad:
        unused_causal_prefix_topk = 0

    scores = glm_fast.dsa_indexer_scores(
        q,
        k,
        w,
        causal=causal,
        unused_causal_prefix_topk=unused_causal_prefix_topk,
        skip_causal_future_store=skip_causal_future_store,
        causal_q_offset=causal_q_offset,
        stream=stream or mx.gpu,
    )
    if q_pad or k_pad:
        scores = scores[:, :, :L, :K]
    return scores


def sparse_mla_attention(
    q_latent: mx.array,
    q_pe: mx.array,
    kv_latent: mx.array,
    k_pe: mx.array,
    topk_indices: mx.array,
    scale: float,
    *,
    topk_valid_prefix: bool = False,
    causal_prefix_indices: bool = False,
    topk_length: Optional[mx.array] = None,
    causal_prefix_rows: int = 0,
    stream: Optional[mx.Stream] = None,
) -> Optional[mx.array]:
    """Sparse MLA prefill over per-query DSA top-k indices.

    This mirrors the FlashMLA sparse prefill contract used by vLLM/SGLang:
    attention scores are computed over [latent, rope] keys, values are the
    latent KV cache, and the caller applies the MLA output projection after.

    Shapes:
      q_latent: [B, H, L, 512]
      q_pe: [B, H, L, 64]
      kv_latent: [B, 1, K, 512]
      k_pe: [B, 1, K, 64]
      topk_indices: [B, 1, L, TOPK]
      topk_length: optional [B, L] or [B, 1, L] valid prefix length
    """

    if (
        q_latent.ndim != 4
        or q_pe.ndim != 4
        or kv_latent.ndim != 4
        or k_pe.ndim != 4
        or topk_indices.ndim != 4
        or kv_latent.shape[1] != 1
        or k_pe.shape[1] != 1
    ):
        return None

    B, H, L, D_LATENT = q_latent.shape
    K = kv_latent.shape[2]
    D_PE = q_pe.shape[-1]
    topk_rows = topk_indices.shape[2]
    compact_prefix = causal_prefix_rows > 0 and topk_rows != L

    if (
        L <= 1
        or q_pe.shape[:3] != (B, H, L)
        or kv_latent.shape[:3] != (B, 1, K)
        or k_pe.shape[:3] != (B, 1, K)
        or topk_indices.shape[:2] != (B, 1)
        or not (
            topk_rows == L
            or (
                compact_prefix
                and topk_rows + causal_prefix_rows == L
                and causal_prefix_indices
                and topk_valid_prefix
            )
        )
        or kv_latent.shape[-1] != D_LATENT
        or k_pe.shape[-1] != D_PE
        or D_LATENT != 512
        or D_PE != 64
        or q_latent.dtype not in (mx.float16, mx.bfloat16)
        or q_pe.dtype != q_latent.dtype
        or kv_latent.dtype != q_latent.dtype
        or k_pe.dtype != q_latent.dtype
    ):
        return None

    if not hasattr(glm_fast, "glm_dsa_sparse_mla_attention"):
        return None

    topk = (
        topk_indices
        if topk_indices.dtype == mx.uint32
        else topk_indices.astype(mx.uint32)
    )
    if topk_length is not None and topk_length.dtype != mx.uint32:
        topk_length = topk_length.astype(mx.uint32)
    return glm_fast.glm_dsa_sparse_mla_attention(
        q_latent,
        q_pe,
        kv_latent,
        k_pe,
        topk,
        scale,
        topk_valid_prefix=topk_valid_prefix,
        causal_prefix_indices=causal_prefix_indices,
        topk_length=topk_length,
        causal_prefix_rows=causal_prefix_rows,
        stream=stream or mx.gpu,
    )


def q8_vup_flat(
    x: mx.array,
    unembed_out,
    *,
    key_length: Optional[int] = None,
    stream: Optional[mx.Stream] = None,
) -> Optional[mx.array]:
    """Project GLM sparse MLA latent output directly to [B, L, H * 256]."""

    if key_length is None or key_length < 32768 or key_length > 65536:
        return None

    if (
        not hasattr(glm_fast, "glm_dsa_q8_vup_flat")
        or x.ndim != 4
        or x.shape[1] != 64
        or x.shape[-1] != 512
        or getattr(unembed_out, "bits", None) != 8
        or getattr(unembed_out, "group_size", None) != 64
        or getattr(unembed_out, "mode", None) != "affine"
        or not hasattr(unembed_out, "weight")
        or not hasattr(unembed_out, "scales")
    ):
        return None

    biases = unembed_out.get("biases") if hasattr(unembed_out, "get") else None
    if biases is None:
        return None
    weight = unembed_out["weight"]
    scales = unembed_out["scales"]
    if weight.shape != (64, 256, 128) or scales.shape != (64, 256, 8):
        return None
    return glm_fast.glm_dsa_q8_vup_flat(
        x,
        weight,
        scales,
        biases,
        stream=stream or mx.gpu,
    )
