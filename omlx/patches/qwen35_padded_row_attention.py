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

The multi-token case is recognized by shape, not by caller: any block of
2..MAX_LIGHTNING_MTP_DRAFT_TOKENS + 1 query tokens over more than 8,192 keys
whose mask is exactly left padding plus causal. A short prefill block of that
shape gets the same treatment, which is equally exact for it: the per-row
route attends to the same keys the mask allows.

Everything else keeps its current path: batches the ragged kernel or the
bounded verify kernel serve, quantized caches, wider blocks (prefill
chunks), masks other than left padding plus causal, and any shape this patch
does not recognize.

Row offsets and the mask check are read once per forward, in the
attention-mask hook, in one host read (the way mlx-vlm already reads offsets
for left-padded decode), and copied to every full-attention cache so the
layers do not synchronize with the GPU. The check costs about 0.4 ms per
forward at 100K keys and 0.75 ms at 262K (M5 Max), against about 4 ms per
layer saved at 100K.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

_PATCHED = False
_PATCH_MARKER = "_omlx_qwen35_padded_row_attention_patch"
# Verify-row offsets for the current forward, stored on each full-attention
# cache: (pads, cache._idx before this forward's update, query length, mask
# shape). The hook stores them only after checking that the forward's mask is
# exactly left padding + causal; each layer then receives a full-width slice
# of that mask (_prepare_projected_qkv), so the shape identifies it.
_ROWS_ATTR = "_omlx_qwen35_verify_row_pads"
try:
    from ..model_settings import MAX_LIGHTNING_MTP_DRAFT_TOKENS as _MAX_DRAFTS
except ImportError:  # pragma: no cover - keeps the patch importable alone
    _MAX_DRAFTS = 8
# Widest MTP verify block (target token + drafts). Wider multi-token blocks
# (prefill chunks) keep their current path.
_MAX_VERIFY_WIDTH = _MAX_DRAFTS + 1
# Matches the bounded padded-verify kernel in qwen35_verify_attention: at or
# below this many keys that kernel already serves the batch.
_VERIFY_KERNEL_MAX_KEYS = 8192
_ENGAGED_LOGGED: set[tuple[int, int]] = set()


def _log_engaged(rows: int, q_len: int) -> None:
    # One line per (rows, width), like the verify-split patch, so a log shows
    # which batch shapes this path served.
    if (rows, q_len) not in _ENGAGED_LOGGED:
        _ENGAGED_LOGGED.add((rows, q_len))
        logger.info(
            "[padded-row] batch-1 attention per row engaged (rows=%d, q_len=%d)",
            rows,
            q_len,
        )


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


def _verify_rows(queries, keys, cache, mask) -> list[int] | None:
    """Pads of a padded verify batch beyond the bounded verify kernel."""
    rows = getattr(cache, _ROWS_ATTR, None)
    if rows is None:
        return None
    pads, idx_before, length, mask_shape = rows
    if (
        not isinstance(mask, mx.array)
        or mask.shape != mask_shape
        or queries.shape[2] != length
        or not 1 < length <= _MAX_VERIFY_WIDTH
        or len(pads) != queries.shape[0]
        or cache._idx != idx_before + length
        or keys.shape[2] != cache._idx
        or keys.shape[2] <= _VERIFY_KERNEL_MAX_KEYS
        or any(p < 0 or p > keys.shape[2] - length for p in pads)
    ):
        return None
    return list(pads)


def _padded_causal_mask_shape(mask, batch: int, length: int, k_size: int) -> bool:
    """Shape of the mask BatchKVCache.make_mask builds for a padded block."""
    return (
        isinstance(mask, mx.array)
        and mask.dtype == mx.bool_
        and mask.shape == (batch, 1, length, k_size)
    )


def _loop_causal(q, k, v, scale):
    """Exact causal attention of a query block over one row, row by row."""
    prefix = k.shape[2] - q.shape[2]
    return mx.concatenate(
        [
            mx.fast.scaled_dot_product_attention(
                q[:, :, i : i + 1],
                k[:, :, : prefix + i + 1],
                v[:, :, : prefix + i + 1],
                scale=scale,
            )
            for i in range(q.shape[2])
        ],
        axis=2,
    )


def _per_row(original, queries, keys, values, pads, scale, tails=None):
    """Each row through its batch-1 route on a view of its own keys.

    Decode: a lone request's decode is stock SDPA with no mask (no seam
    claims it), so call that directly. Verify: the original seam with no
    cache and a causal mask, which is how a lone request's verify arrives.
    """
    outputs = []
    for row, pad in enumerate(pads):
        # A deferred-realignment tail ends the row before _idx.
        end = keys.shape[2] - (tails[row] if tails else 0)
        q = queries[row : row + 1]
        k = keys[row : row + 1, :, pad:end, :]
        v = values[row : row + 1, :, pad:end, :]
        if queries.shape[2] == 1:
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        else:
            out = original(q, k, v, cache=None, scale=scale, mask="causal")
            if out is None:
                if not tails:
                    return None
                out = _loop_causal(q, k, v, scale)
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
        from . import qwen35_deferred_realign as dr

        t = dr.tails(cache) if cache is not None else None
        if t and (
            h.shape[1] > _MAX_VERIFY_WIDTH
            or h.shape[0] != len(t)
            or (h.shape[1] > 1 and not dr.in_verify())
        ):
            # Not a block the per-row tail path serves (too wide, a different
            # batch, or a multi-token forward outside the speculative verifier,
            # which would take the stock masked SDPA): realign this cache now
            # (the mask is built from its padding) and every other layer in
            # set_pads, not just the layer the mask is built from.
            dr.realign(cache)
            dr.request_realign_all(cache)
            t = None
        mask = original_mask(h, cache)
        if cache is not None and hasattr(cache, _ROWS_ATTR):
            delattr(cache, _ROWS_ATTR)
        length = h.shape[1]
        if t and length == 1:
            # Rows end before _idx: every layer must take the per-row seam.
            cache._qwen3_5_decode_left_padding = [
                int(p) for p in cache.left_padding.tolist()
            ]
            return "left_padded_decode"
        if (
            1 < length <= _MAX_VERIFY_WIDTH
            and _plain_padded_cache(cache)
            and (t or cache._idx + length > _VERIFY_KERNEL_MAX_KEYS)
            and _padded_causal_mask_shape(mask, h.shape[0], length, cache._idx + length)
        ):
            from mlx_lm.models.base import create_causal_mask

            # The per-row route drops the mask and slices each row's keys
            # from its pad: only valid if the mask is exactly that.
            expected = create_causal_mask(
                length, offset=cache._idx, left_padding=cache.left_padding
            )
            same = mx.array_equal(mask, expected)
            # One small host read per forward, like mlx-vlm's decode pads.
            mx.eval(same, cache.left_padding)
            pads = tuple(int(p) for p in cache.left_padding.tolist())
            if same.item() and len(pads) == h.shape[0] and (t or max(pads) > 0):
                setattr(cache, _ROWS_ATTR, (pads, cache._idx, length, mask.shape))
            elif t:
                # The tail path drops the mask, which is only valid for the
                # padded causal mask: otherwise realign everything and serve
                # this forward through the stock masked path.
                dr.realign(cache)
                dr.request_realign_all(cache)
                return original_mask(h, cache)
        elif t and length > 1:
            dr.realign(cache)
            dr.request_realign_all(cache)
            return original_mask(h, cache)
        return mask

    def set_pads(caches, layers, pads):
        from . import qwen35_deferred_realign as dr

        original_set_pads(caches, layers, pads)
        if caches is None:
            return
        if any(
            entry is not None and entry.__dict__.pop("_omlx_realign_all", False)
            for entry in caches
            if hasattr(entry, "__dict__")
        ):
            for layer, entry in zip(layers, caches):
                if not layer.is_linear and entry is not None and dr.tails(entry):
                    dr.realign(entry)
        rows = None
        for layer, entry in zip(layers, caches):
            if not layer.is_linear and entry is not None:
                rows = getattr(entry, _ROWS_ATTR, None)
                break
        for layer, entry in zip(layers, caches):
            if layer.is_linear or entry is None:
                continue
            if _plain_padded_cache(entry):
                # Only caches this attention serves may defer realignment.
                entry.__dict__["_omlx_tail_ok"] = True
            if rows is None:
                if hasattr(entry, _ROWS_ATTR):
                    delattr(entry, _ROWS_ATTR)
            else:
                setattr(entry, _ROWS_ATTR, rows)

    def attention(queries, keys, values, *, cache, scale, mask):
        from . import qwen35_deferred_realign as dr

        t = dr.tails(cache) if cache is not None else None
        if t is not None:
            # Rows end at _idx - tail: no batched path knows that, so always
            # per row (decode pads from the hook, verify pads from the stash
            # or the cache).
            rows = getattr(cache, _ROWS_ATTR, None)
            if queries.shape[2] == 1:
                pads = getattr(cache, "_qwen3_5_decode_left_padding", None)
            else:
                pads = list(rows[0]) if rows is not None else None
            if pads is None or len(pads) != queries.shape[0]:
                pads = [int(p) for p in cache.left_padding.tolist()]
            out = _per_row(
                original_attention, queries, keys, values, pads, scale, tails=t
            )
            _log_engaged(len(pads), queries.shape[2])
            return out
        if (
            cache is not None
            and _plain_padded_cache(cache)
            and _qkv_ok(queries, keys, values)
        ):
            if queries.shape[2] == 1:
                pads = _decode_rows(q35, queries, keys, values, cache)
            else:
                pads = _verify_rows(queries, keys, cache, mask)
            if pads is not None:
                out = _per_row(original_attention, queries, keys, values, pads, scale)
                if out is not None:
                    _log_engaged(len(pads), queries.shape[2])
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
