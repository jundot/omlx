# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

_PATCHED = False
_PROBE_CACHE: dict[tuple[Any, ...], bool] = {}
_PATCH_MARKER = "_omlx_qwen35_ragged_decode_patch"
_ORIGINAL_ATTR = "_omlx_qwen35_ragged_decode_original"


def _threadgroup_limit_error(exc: Exception) -> bool:
    msg = str(exc)
    return "Thread group size" in msg or "threads per threadgroup" in msg


def _signature(q35, queries, keys, values, pads) -> tuple[Any, ...] | None:
    try:
        if (
            queries.ndim != 4
            or keys.ndim != 4
            or values.ndim != 4
            or queries.shape[2] != 1
            or queries.dtype not in (mx.bfloat16, mx.float16)
            or keys.dtype != queries.dtype
            or values.dtype != queries.dtype
        ):
            return None
        batch, q_heads, _, d_size = queries.shape
        pads_tuple = tuple(int(p) for p in pads)
        if len(pads_tuple) != batch or any(p < 0 for p in pads_tuple):
            return None
        kv_heads = keys.shape[1]
        k_size = keys.shape[2]
        v_size = values.shape[-1]
        if (
            q_heads % kv_heads != 0
            or d_size != v_size
            or d_size not in (64, 96, 128, 256)
            or any(p >= k_size for p in pads_tuple)
        ):
            return None
        plans = [
            q35._qwen3_5_sdpa_vector_plan(k_size - pad, q_heads, kv_heads)
            for pad in pads_tuple
        ]
        if len(set(plans)) != 1:
            return None
        mode, blocks = plans[0]
        return (
            mode,
            int(blocks),
            str(queries.dtype),
            int(d_size),
            int(v_size),
            int(q_heads),
            int(kv_heads),
        )
    except Exception:
        return None


def _call_with_probe(original, key, queries, keys, values, pads, scale):
    cached = _PROBE_CACHE.get(key)
    if cached is False:
        return None

    try:
        out = original(queries, keys, values, pads, scale)
        if cached is True or out is None:
            return out
        mx.eval(out)
        _PROBE_CACHE[key] = True
        return out
    except Exception as exc:
        if _threadgroup_limit_error(exc):
            logger.warning(
                "qwen3_5 ragged decode kernel %s exceeds threadgroup limit; "
                "using reference fallback",
                key,
            )
            _PROBE_CACHE[key] = False
            return None
        raise


def apply_qwen35_ragged_decode_patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return False

    try:
        from mlx_vlm.models.qwen3_5 import language as q35

        current = q35._qwen3_5_ragged_decode_attention
    except (ImportError, AttributeError):
        return False

    if getattr(current, _PATCH_MARKER, False):
        _PATCHED = True
        return False

    original = current

    def patched(queries, keys, values, pads, scale):
        key = _signature(q35, queries, keys, values, pads)
        if key is None:
            return original(queries, keys, values, pads, scale)
        return _call_with_probe(original, key, queries, keys, values, pads, scale)

    setattr(patched, _PATCH_MARKER, True)
    setattr(patched, _ORIGINAL_ATTR, original)
    q35._qwen3_5_ragged_decode_attention = patched
    _PATCHED = True
    logger.info("qwen3_5 ragged decode threadgroup fallback patch applied")
    return True
