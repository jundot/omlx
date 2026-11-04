# SPDX-License-Identifier: Apache-2.0
"""Serve left-padded Qwen3.5/3.6/3.8 batches with the batch-1 attention kernels.

Two padded-batch attention paths fall back to slow code at long context:

* Decode (q_len 1): mlx-vlm's ragged SDPA kernel refuses a batch whose rows
  need different two-pass block counts (for example 24.7K and 100.7K tokens),
  and its fallback gathers each row's K/V with ``mx.take`` in every
  full-attention layer, every step.
* MTP verify (q_len > 1): ``qwen35_verify_attention`` claims padded batches
  only up to 8,192 keys, so above that the exact verifier runs one SDPA per
  query row over the whole padded K/V.

Attention has no shared weights, so a batched attention kernel saves no
weight bandwidth over one call per row. This patch hands each row of those
two cases to the attention its request would get alone: the row's queries and
a view of its own keys (``keys[i:i+1, :, pad_i:, :]``, no copy) go to the
original seam with no cache and a causal mask, which is the batch-1 route
(stock SDPA for decode, ``qwen35_verify_sdpa_split`` for verify). The output
of every row therefore equals that request decoding alone, bit for bit.

Everything else keeps its current path: batches the ragged kernel or the
bounded verify kernel serve, quantized caches, explicit masks other than the
left-padded causal mask, and any shape this patch does not recognize.

Row offsets are read once per forward, in the attention-mask hook, the way
mlx-vlm already reads them for left-padded decode, and are copied to every
full-attention cache so the layers do not synchronize with the GPU.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

_PATCHED = False
_PATCH_MARKER = "_omlx_qwen35_padded_row_attention_patch"
# Verify-row offsets for the current forward, stored on each full-attention
# cache: (pads, cache._idx before this forward's update, query length).
_ROWS_ATTR = "_omlx_qwen35_verify_row_pads"
# Matches the bounded padded-verify kernel in qwen35_verify_attention: at or
# below this many keys that kernel already serves the batch.
_VERIFY_KERNEL_MAX_KEYS = 8192


def _batch_cache_types() -> tuple[type, ...]:
    types: list[type] = []
    try:
        from mlx_lm.models.cache import BatchKVCache as LMBatchKVCache

        types.append(LMBatchKVCache)
    except ImportError:
        pass
    try:
        from mlx_vlm.models.cache import BatchKVCache

        types.append(BatchKVCache)
    except ImportError:
        pass
    return tuple(types)


def _plain_padded_cache(cache: Any) -> bool:
    """A dense, unquantized, left-padded batch cache with no pending right pad."""
    return (
        type(cache) in _batch_cache_types()
        and not hasattr(cache, "bits")
        and isinstance(getattr(cache, "left_padding", None), mx.array)
        and cache.left_padding.ndim == 1
        and cache.left_padding.size > 1
        and getattr(cache, "_right_padding", None) is None
        and isinstance(getattr(cache, "_idx", None), int)
    )


def _qkv_ok(queries, keys, values) -> bool:
    return (
        queries.ndim == 4
        and keys.ndim == 4
        and values.ndim == 4
        and queries.dtype in (mx.float16, mx.bfloat16)
        and keys.dtype == queries.dtype
        and values.dtype == queries.dtype
        and keys.shape[0] == queries.shape[0]
        and values.shape[:3] == keys.shape[:3]
        and keys.shape[1] > 0
        and queries.shape[1] % keys.shape[1] == 0
    )


def _decode_rows(q35, queries, keys, values, cache) -> list[int] | None:
    """Pads of a left-padded decode batch the ragged kernel would refuse."""
    pads = getattr(cache, "_qwen3_5_decode_left_padding", None)
    if pads is None or queries.shape[2] != 1 or len(pads) != queries.shape[0]:
        return None
    k_size = keys.shape[2]
    if any(p < 0 or p >= k_size for p in pads) or max(pads) <= 0:
        return None
    plans = {
        q35._qwen3_5_sdpa_vector_plan(k_size - p, queries.shape[1], keys.shape[1])
        for p in pads
    }
    # One plan: the ragged kernel serves the whole batch; leave it.
    return list(pads) if len(plans) > 1 else None


def _verify_rows(queries, keys, cache) -> list[int] | None:
    """Pads of a padded verify batch beyond the bounded verify kernel."""
    rows = getattr(cache, _ROWS_ATTR, None)
    if rows is None:
        return None
    pads, idx_before, length = rows
    if (
        queries.shape[2] != length
        or length <= 1
        or len(pads) != queries.shape[0]
        or cache._idx != idx_before + length
        or keys.shape[2] != cache._idx
        or keys.shape[2] <= _VERIFY_KERNEL_MAX_KEYS
        or any(p < 0 or p > keys.shape[2] - length for p in pads)
    ):
        return None
    return list(pads)


def _causal_padded_mask(mask, batch: int, length: int, k_size: int) -> bool:
    """The mask BatchKVCache.make_mask builds: left padding plus causal only."""
    if mask is None or (isinstance(mask, str) and mask == "causal"):
        return True
    return (
        isinstance(mask, mx.array)
        and mask.dtype == mx.bool_
        and mask.shape == (batch, 1, length, k_size)
    )


def _per_row(original, queries, keys, values, pads, scale):
    """Each row through its batch-1 route on a view of its own keys.

    Decode: a lone request's decode is stock SDPA with no mask (no seam
    claims it), so call that directly. Verify: the original seam with no
    cache and a causal mask, which is how a lone request's verify arrives.
    """
    outputs = []
    for row, pad in enumerate(pads):
        q = queries[row : row + 1]
        k = keys[row : row + 1, :, pad:, :]
        v = values[row : row + 1, :, pad:, :]
        if queries.shape[2] == 1:
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        else:
            out = original(q, k, v, cache=None, scale=scale, mask="causal")
            if out is None:
                return None
        outputs.append(out)
    return mx.concatenate(outputs, axis=0)


def apply_qwen35_padded_row_attention_patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return False
    try:
        from mlx_vlm.models.qwen3_5 import language as q35

        original_attention = q35._qwen3_5_left_padded_attention
        original_mask = q35._create_qwen3_5_attention_mask
        original_set_pads = q35._set_qwen3_5_decode_left_padding
    except (ImportError, AttributeError):
        return False
    if getattr(original_attention, _PATCH_MARKER, False):
        _PATCHED = True
        return False

    def attention_mask(h, cache):
        mask = original_mask(h, cache)
        if cache is not None and hasattr(cache, _ROWS_ATTR):
            delattr(cache, _ROWS_ATTR)
        length = h.shape[1]
        if (
            length > 1
            and _plain_padded_cache(cache)
            and cache._idx + length > _VERIFY_KERNEL_MAX_KEYS
            and _causal_padded_mask(mask, h.shape[0], length, cache._idx + length)
        ):
            # One small host read per forward, like mlx-vlm's decode pads.
            pads = tuple(int(p) for p in cache.left_padding.tolist())
            if len(pads) == h.shape[0] and max(pads) > 0:
                setattr(cache, _ROWS_ATTR, (pads, cache._idx, length))
        return mask

    def set_pads(caches, layers, pads):
        original_set_pads(caches, layers, pads)
        if caches is None:
            return
        rows = None
        for layer, entry in zip(layers, caches):
            if not layer.is_linear and entry is not None:
                rows = getattr(entry, _ROWS_ATTR, None)
                break
        for layer, entry in zip(layers, caches):
            if layer.is_linear or entry is None:
                continue
            if rows is None:
                if hasattr(entry, _ROWS_ATTR):
                    delattr(entry, _ROWS_ATTR)
            else:
                setattr(entry, _ROWS_ATTR, rows)

    def attention(queries, keys, values, *, cache, scale, mask):
        if (
            cache is not None
            and _plain_padded_cache(cache)
            and _qkv_ok(queries, keys, values)
        ):
            if queries.shape[2] == 1:
                pads = _decode_rows(q35, queries, keys, values, cache)
            elif _causal_padded_mask(
                mask, queries.shape[0], queries.shape[2], keys.shape[2]
            ):
                pads = _verify_rows(queries, keys, cache)
            else:
                pads = None
            if pads is not None:
                out = _per_row(original_attention, queries, keys, values, pads, scale)
                if out is not None:
                    return out
        return original_attention(
            queries, keys, values, cache=cache, scale=scale, mask=mask
        )

    setattr(attention, _PATCH_MARKER, True)
    q35._qwen3_5_left_padded_attention = attention
    q35._create_qwen3_5_attention_mask = attention_mask
    q35._set_qwen3_5_decode_left_padding = set_pads
    _PATCHED = True
    logger.info("qwen3_5 padded-row attention patch applied")
    return True
