# SPDX-License-Identifier: Apache-2.0
"""Collapse Qwen3.5/3.6 verify-width attention into causal vector-kernel calls.

mlx-vlm's ``Qwen3_5Attention`` serves a target-verify forward (q_len =
1 + draft depth) on a batch-1 dense cache with a per-row fallback: L
single-row SDPA calls over per-row K/V slices, concatenated. That costs L
dispatches plus 2L slice ops and a concat per attention layer per verify
cycle, and it is the reason deeper MTP chains stop paying on this family.

MLX's SDPA vector kernel serves causal blocks up to ``q_len * gqa <= 32``
(rows <= 5 on the dense 24/4 layout, <= 4 on the 16/2 MoE one), with
bottom-right alignment — exactly the per-row causal windows the loop
reproduces by hand. So a verify block of L rows needs only ceil(L / limit)
dispatches: rows are chunked at the vector-kernel row limit, each chunk c
covering rows [c0, c1) against ``keys[: kv_len - (L - c1)]`` with
``mask="causal"``. (Same construction as mlx-serve's ``splitCausalSdpa``,
measured +4..9% decode there with speculation on.)

The seam is ``_target_verify_left_padded_attention``: it runs FIRST in the
target-verify branch and its non-None result skips the row loop, while a
None keeps every existing path unchanged. This patch wraps it to claim the
batch-1 / dense-cache / head_dim-256 shape and delegate everything else
(left-padded batches, quantized caches) to the original.
"""

from __future__ import annotations

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)

_PATCHED = False
_ENGAGED_LOGGED: set[int] = set()

# MLX fast::ScaledDotProductAttention routes to the vector kernel only while
# q_len * gqa_factor <= 32; wider blocks fall to the composed unfused path.
_VECTOR_ROW_BUDGET = 32


def _log_engaged(q_len: int) -> None:
    # One log per width; a single one-shot log would only witness the first
    # width and hide whether deeper chains route.
    if q_len not in _ENGAGED_LOGGED:
        _ENGAGED_LOGGED.add(q_len)
        logger.info(
            "[verify-split] hd-256 causal vector attention engaged (q_len=%d)",
            q_len,
        )


def _chunked_causal_sdpa(queries, keys, values, scale, limit: int):
    q_len = queries.shape[-2]
    kv_len = keys.shape[-2]
    outs = []
    c0 = 0
    while c0 < q_len:
        c1 = min(c0 + limit, q_len)
        kv_end = kv_len - (q_len - c1)
        outs.append(
            mx.fast.scaled_dot_product_attention(
                queries[..., c0:c1, :],
                keys[..., :kv_end, :],
                values[..., :kv_end, :],
                scale=scale,
                mask="causal",
            )
        )
        c0 = c1
    if len(outs) == 1:
        return outs[0]
    return mx.concatenate(outs, axis=-2)


def _eligible(queries, keys, cache) -> int:
    """Return the vector-kernel row limit (>0) when this call is ours."""
    # A turboquant-quantized cache hands back `_QuantizedStateProxy` objects
    # (exposes .shape, deliberately not .ndim, to avoid dequantizing — see
    # mlx_vlm.turboquant). That's exactly the "quantized caches" case this
    # patch already means to delegate to the original path below, but the
    # attribute access below used to run before the `hasattr(cache, "bits")`
    # guard could rule it out, so it crashed instead of falling through.
    if getattr(queries, "ndim", None) != 4 or getattr(keys, "ndim", None) != 4:
        return 0
    if queries.shape[0] != 1:
        return 0
    if cache is not None and hasattr(cache, "bits"):
        return 0
    if queries.dtype not in (mx.float16, mx.bfloat16):
        return 0
    if queries.dtype != keys.dtype:
        return 0
    if queries.shape[-1] != 256 or keys.shape[-1] != 256:
        return 0
    q_heads = queries.shape[-3]
    kv_heads = keys.shape[-3]
    if kv_heads <= 0 or q_heads % kv_heads != 0:
        return 0
    limit = _VECTOR_ROW_BUDGET // (q_heads // kv_heads)
    if limit <= 0:
        return 0
    q_len = queries.shape[-2]
    if q_len <= 1 or q_len > keys.shape[-2]:
        return 0
    return limit


def apply_qwen35_verify_sdpa_split_patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    if not mx.metal.is_available():
        return False

    try:
        from mlx_vlm.models.qwen3_5 import language as q35_lang
    except ImportError:
        return False

    original = getattr(q35_lang, "_target_verify_left_padded_attention", None)
    if original is None:
        logger.debug("verify-split: target-verify seam not found; patch skipped")
        return False

    def patched_target_verify_attention(
        queries,
        keys,
        values,
        *,
        cache,
        scale,
        mask,
    ):
        # Only the batch-1 dense-cache shape is ours; a batch with real left
        # padding (or anything unexpected) keeps the original behavior.
        if mask is None or (isinstance(mask, str) and mask == "causal"):
            limit = _eligible(queries, keys, cache)
            if limit and getattr(cache, "left_padding", None) is None:
                try:
                    out = _chunked_causal_sdpa(
                        queries, keys, values, scale, limit
                    )
                    _log_engaged(queries.shape[-2])
                    return out
                except Exception:
                    logger.warning(
                        "verify-split attention failed; falling back",
                        exc_info=True,
                    )
        return original(
            queries, keys, values, cache=cache, scale=scale, mask=mask
        )

    q35_lang._target_verify_left_padded_attention = patched_target_verify_attention
    _PATCHED = True
    logger.info("Qwen3.5/3.6 verify-width causal vector attention patch applied")
    return True
