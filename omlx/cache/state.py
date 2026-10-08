# SPDX-License-Identifier: Apache-2.0
"""Stable cache state shared by batched and speculative generation."""

from __future__ import annotations

import logging
from typing import Any, Optional
import mlx.core as mx
from .hybrid_cache import ModelCacheConfig
from .type_registry import CacheTypeRegistry

logger = logging.getLogger(__name__)
HAS_CACHE_TYPE_HANDLERS = True


def normalize_rotating_snapshot_state(
    layer_cache: Any,
    state: tuple[Any, Any],
    meta_state: Any,
    layer_idx: int | None = None,
) -> tuple[tuple[Any, Any], tuple[str, str, str, str]]:
    """
    Normalize RotatingKVCache state into merge-safe canonical form.

    Boundary snapshots captured mid-prefill can expose oversized rotating
    buffers (e.g., max_size + chunk_size - 1). Those states are valid for
    in-flight prefill but break BatchRotatingKVCache.merge() after SSD
    restore because merge expects per-request rotating buffers capped to
    max_size. This method canonicalizes to the latest max_size tokens.
    """
    if not isinstance(state, (list, tuple)) or len(state) < 2:
        return state, (
            tuple(meta_state) if isinstance(meta_state, (list, tuple)) else ()
        )

    keys = state[0]
    values = state[1]
    if keys is None or values is None or not hasattr(keys, "shape"):
        return state, (
            tuple(meta_state) if isinstance(meta_state, (list, tuple)) else ()
        )

    try:
        keep = (
            int(meta_state[0])
            if meta_state and len(meta_state) >= 1
            else int(getattr(layer_cache, "keep", 0))
        )
        max_size = (
            int(meta_state[1])
            if meta_state and len(meta_state) >= 2
            else int(getattr(layer_cache, "max_size", keys.shape[2]))
        )
        offset = (
            int(meta_state[2])
            if meta_state and len(meta_state) >= 3
            else int(getattr(layer_cache, "offset", keys.shape[2]))
        )
        idx = (
            int(meta_state[3])
            if meta_state and len(meta_state) >= 4
            else int(getattr(layer_cache, "_idx", keys.shape[2]))
        )
    except Exception:
        return state, (
            tuple(meta_state) if isinstance(meta_state, (list, tuple)) else ()
        )

    ordered_keys = keys
    ordered_values = values
    temporal_order = getattr(layer_cache, "_temporal_order", None)
    if callable(temporal_order):
        try:
            ordered_keys = temporal_order(keys)
            ordered_values = temporal_order(values)
        except Exception:
            ordered_keys = keys
            ordered_values = values

    original_len = int(ordered_keys.shape[2]) if len(ordered_keys.shape) >= 3 else 0
    normalized_keys = ordered_keys
    normalized_values = ordered_values

    if max_size > 0 and original_len > max_size:
        if keep > 0 and keep < max_size:
            tail_len = max_size - keep
            normalized_keys = mx.concatenate(
                [
                    ordered_keys[..., :keep, :],
                    ordered_keys[..., -tail_len:, :],
                ],
                axis=2,
            )
            normalized_values = mx.concatenate(
                [
                    ordered_values[..., :keep, :],
                    ordered_values[..., -tail_len:, :],
                ],
                axis=2,
            )
        else:
            normalized_keys = ordered_keys[..., -max_size:, :]
            normalized_values = ordered_values[..., -max_size:, :]

        try:
            normalized_keys = mx.contiguous(normalized_keys)
            normalized_values = mx.contiguous(normalized_values)
        except Exception:
            pass

    normalized_len = (
        int(normalized_keys.shape[2]) if len(normalized_keys.shape) >= 3 else 0
    )
    # Force case 1 of _temporal_order: _idx == keys.shape[2] means the
    # buffer is already in temporal order (which is exactly what the
    # oversized trim above produces — the contiguous tail of the most
    # recent tokens). Anything else lets _temporal_order re-slice the
    # buffer in the rotated branch (case 2), which is wasted work and
    # obscures the merge contract. See cache.py:431-447 for the branches.
    normalized_idx = normalized_len

    normalized_meta = (
        str(keep),
        str(max_size),
        str(offset),
        str(normalized_idx),
    )

    if original_len != normalized_len or idx != normalized_idx:
        layer_tag = f"layer {layer_idx}: " if layer_idx is not None else ""
        logger.debug(
            "%sNormalized RotatingKVCache snapshot: len %s->%s, idx %s->%s, "
            "offset=%s, max_size=%s",
            layer_tag,
            original_len,
            normalized_len,
            idx,
            normalized_idx,
            offset,
            max_size,
        )

    return (normalized_keys, normalized_values), normalized_meta


def extract_cache_states(
    raw_cache: list[Any],
    model_name: str = "",
) -> tuple[list[dict[str, Any]], Optional["ModelCacheConfig"]]:
    """
    Extract actual tensor state from each layer cache.

    This extracts the real KV data using mlx-lm's cache.state property,
    allowing the data to be stored and reconstructed later even after
    the BatchGenerator is recreated.

    Also creates a ModelCacheConfig with per-layer type information to
    support hybrid cache models (e.g., KVCache + ArraysCache).

    Args:
        raw_cache: List of cache objects from mlx-lm (KVCache, ArraysCache, etc.)

    Returns:
        Tuple of:
        - List of dicts with {state, meta_state, class_name, cache_type}
        - ModelCacheConfig with per-layer type information (or None)
    """
    if not raw_cache:
        return [], None

    # Build ModelCacheConfig for type information.
    # Skip if raw_cache contains None entries (boundary snapshots with
    # sliceable layers replaced by None) — from_cache_list expects real
    # cache objects and would log noisy NoneType warnings.
    model_cache_config = None
    has_none_layers = any(c is None for c in raw_cache)
    if HAS_CACHE_TYPE_HANDLERS and ModelCacheConfig is not None and not has_none_layers:
        try:
            model_cache_config = ModelCacheConfig.from_cache_list(
                raw_cache,
                model_name=model_name,
            )
        except Exception as e:
            logger.debug(f"Failed to build ModelCacheConfig: {e}")

    extracted = []
    for layer_idx, layer_cache in enumerate(raw_cache):
        # Boundary snapshots may contain None for sliceable layers
        # (KVCache) that were skipped during capture to save memory.
        # Insert a placeholder to preserve layer index alignment.
        if layer_cache is None:
            extracted.append(
                {
                    "state": (),
                    "meta_state": (),
                    "class_name": "KVCache",
                    "cache_type": "KVCache",
                }
            )
            continue
        try:
            class_name = CacheTypeRegistry.canonical_name(layer_cache)

            # Determine cache type using registry if available
            cache_type_name = class_name
            handler = None
            if HAS_CACHE_TYPE_HANDLERS and CacheTypeRegistry is not None:
                try:
                    cache_type = CacheTypeRegistry.detect_cache_type(layer_cache)
                    cache_type_name = cache_type.value
                    handler = CacheTypeRegistry.get_handler(cache_type)
                except Exception:
                    pass

            # CacheList: composite cache with multiple sub-caches
            if cache_type_name == "CacheList" or class_name == "CacheList":
                if HAS_CACHE_TYPE_HANDLERS and CacheTypeRegistry is not None:
                    try:
                        handler = CacheTypeRegistry.get_handler_by_class_name(
                            "CacheList"
                        )
                        state_dict = handler.extract_state(layer_cache)
                        sub_states = list(state_dict.get("sub_states", []))
                        sub_class_names = list(state_dict.get("sub_class_names", []))
                        sub_meta_states = list(state_dict.get("sub_meta_states", []))

                        sub_caches = getattr(layer_cache, "caches", ())
                        for sub_idx, sub_cache in enumerate(sub_caches):
                            if sub_idx >= len(sub_states):
                                break

                            sub_class_name = type(sub_cache).__name__
                            if sub_class_name in (
                                "RotatingKVCache",
                                "BatchRotatingKVCache",
                                "PrefillReadyRotatingKVCache",
                            ):
                                normalized_state, normalized_meta = (
                                    normalize_rotating_snapshot_state(
                                        sub_cache,
                                        sub_states[sub_idx],
                                        (
                                            sub_meta_states[sub_idx]
                                            if sub_idx < len(sub_meta_states)
                                            else getattr(sub_cache, "meta_state", ())
                                        ),
                                        layer_idx=layer_idx,
                                    )
                                )
                                sub_states[sub_idx] = normalized_state
                                if sub_idx < len(sub_meta_states):
                                    sub_meta_states[sub_idx] = normalized_meta

                        extracted.append(
                            {
                                "state": sub_states,
                                "meta_state": (
                                    sub_class_names,
                                    sub_meta_states,
                                ),
                                # prefix_cache's store path reads this to
                                # tag __nstate__ markers and to record
                                # per-sub class names in SSD metadata;
                                # without it both stay unnamed.
                                "sub_class_names": sub_class_names,
                                "class_name": "CacheList",
                                "cache_type": "CacheList",
                            }
                        )
                    except Exception as e:
                        logger.debug(f"CacheList handler extraction failed: {e}")
                        extracted.append(
                            {
                                "state": [],
                                "meta_state": ([], []),
                                "class_name": "CacheList",
                                "cache_type": "CacheList",
                            }
                        )
                else:
                    # Fallback: extract sub-cache state/meta without handlers
                    # MUST append to extracted to prevent layer count mismatch (Issue #1)
                    sub_caches = getattr(layer_cache, "caches", ())
                    sub_states = []
                    sub_class_names = []
                    sub_meta_states = []
                    for sc in sub_caches:
                        sub_states.append(sc.state if hasattr(sc, "state") else ())
                        sub_class_names.append(type(sc).__name__)
                        sub_meta_states.append(getattr(sc, "meta_state", ()))
                    extracted.append(
                        {
                            "state": sub_states,
                            "meta_state": (sub_class_names, sub_meta_states),
                            "sub_class_names": sub_class_names,
                            "class_name": "CacheList",
                            "cache_type": "CacheList",
                        }
                    )
                continue

            if hasattr(layer_cache, "state"):
                if handler is not None:
                    state = handler.serialize_state(layer_cache)
                    meta = handler.serialize_meta_state(layer_cache)
                else:
                    state = layer_cache.state
                    meta = getattr(layer_cache, "meta_state", ())

                is_rotating_cache = class_name in (
                    "RotatingKVCache",
                    "BatchRotatingKVCache",
                    "PrefillReadyRotatingKVCache",
                    "BufferedRotatingKVCache",
                )
                if HAS_CACHE_TYPE_HANDLERS and CacheTypeRegistry is not None:
                    is_rotating_cache = (
                        is_rotating_cache
                        or CacheTypeRegistry.is_rotating_family(class_name)
                    )
                if is_rotating_cache:
                    state, meta = normalize_rotating_snapshot_state(
                        layer_cache,
                        state,
                        meta,
                        layer_idx=layer_idx,
                    )

                # Preserve the full state tuple regardless of length.
                # Legacy 2-tuple caches (KVCache, RotatingKVCache, ...)
                # surface as (keys, values); 3-tuple caches like
                # PoolingCache surface as (buf_kv, buf_gate, pooled);
                # 4-tuple caches like BatchKVCache surface with the
                # extra offset/padding metadata. Downstream
                # serialization (paged_ssd_cache, boundary_snapshot)
                # is N-tuple aware after the cache architecture
                # refactor — see Section 6 of the implementation
                # plan.
                if isinstance(state, (list, tuple)) and len(state) >= 1:
                    # Validate non-None for legacy KV-style caches only.
                    # PoolingCache's buf_kv may legitimately be None
                    # (fresh cache before any update), so skip the
                    # null guard for non-KV cache classes.
                    if (
                        class_name in ("KVCache", "RotatingKVCache", "BatchKVCache")
                        or (
                            HAS_CACHE_TYPE_HANDLERS
                            and CacheTypeRegistry is not None
                            and CacheTypeRegistry.is_rotating_family(class_name)
                        )
                    ) and len(state) >= 2:
                        if state[0] is None or state[1] is None:
                            logger.debug(
                                f"Layer {layer_idx} ({class_name}) has None keys/values, "
                                f"skipping cache extraction"
                            )
                            return [], None  # Return empty - cache is corrupted

                    extracted.append(
                        {
                            "state": tuple(state),
                            "meta_state": meta,
                            "class_name": class_name,
                            "cache_type": cache_type_name,
                        }
                    )
                else:
                    # Unexpected state format (e.g. a non-tuple scalar).
                    logger.debug(
                        f"Layer {layer_idx} ({class_name}) has unexpected state format"
                    )
                    meta = getattr(layer_cache, "meta_state", ())
                    # Wrap the scalar so downstream code still gets a
                    # tuple-shaped state. This path is essentially dead
                    # in practice — kept defensive only.
                    extracted.append(
                        {
                            "state": (state,),
                            "meta_state": meta,
                            "class_name": class_name,
                            "cache_type": cache_type_name,
                        }
                    )
            elif hasattr(layer_cache, "cache"):
                # ArraysCache style: state stored in .cache list
                cache_list = layer_cache.cache
                if isinstance(cache_list, list) and len(cache_list) >= 2:
                    state = (cache_list[0], cache_list[1])
                    meta = getattr(layer_cache, "meta_state", ())
                    extracted.append(
                        {
                            "state": state,
                            "meta_state": meta,
                            "class_name": class_name,
                            "cache_type": cache_type_name,
                        }
                    )
                else:
                    logger.debug(
                        f"Layer {layer_idx} ({class_name}) has invalid cache list"
                    )
                    continue
            else:
                logger.debug(
                    f"Layer {layer_idx} ({class_name}) has no state or cache attribute"
                )
                continue

        except Exception as e:
            logger.debug(f"Failed to extract state from cache layer {layer_idx}: {e}")
            continue

    if len(extracted) != len(raw_cache):
        logger.debug(
            f"Incomplete cache extraction: {len(extracted)}/{len(raw_cache)} layers"
        )
        return [], None

    return extracted, model_cache_config


def restore_cache(caches, templates):
    """Bind the stable SSD tensor format to the active model's cache classes."""
    from mlx_lm.models import cache as lm_cache
    from mlx_vlm.models.cache import PoolingCache

    from ._rotating_subclass import (
        PrefillReadyRotatingKVCache as LMRestoredRotatingKVCache,
    )

    def restore(source, target):
        if hasattr(source, "_inner"):
            source = source._inner
        children = getattr(target, "caches", None)
        if children is not None:
            return type(target)(
                *(
                    restore(old, new)
                    for old, new in zip(source.caches, children, strict=True)
                )
            )
        restored_rotating = (
            type(source) is LMRestoredRotatingKVCache
            and type(target).__module__.startswith("mlx_vlm.")
            and type(target).__name__ == "RotatingKVCache"
        )
        if restored_rotating:
            from ..models.vlm import PrefillReadyRotatingKVCache

            target = PrefillReadyRotatingKVCache(source.max_size, source.keep)
        arrays_family = (
            CacheTypeRegistry.detect_cache_type(source).value == "ArraysCache"
            and CacheTypeRegistry.detect_cache_type(target).value == "ArraysCache"
        )
        if (
            not restored_rotating
            and not arrays_family
            and (
                type(source) is type(target)
                or type(source).__name__ != type(target).__name__
            )
        ):
            return source
        if arrays_family:
            if len(source.cache) != len(target.cache):
                raise ValueError("Reconstructed arrays cache state arity mismatch")
            target.cache = list(source.cache)
            target.left_padding = source.left_padding
            target.lengths = source.lengths
        elif type(source) in (
            lm_cache.KVCache,
            lm_cache.RotatingKVCache,
            lm_cache.ChunkedKVCache,
            LMRestoredRotatingKVCache,
        ):
            target.keys, target.values = source.keys, source.values
            target.offset = source.offset
            if isinstance(source, lm_cache.RotatingKVCache):
                target.keep, target.max_size = source.keep, source.max_size
                target._idx = source._idx
            elif type(source) is lm_cache.ChunkedKVCache:
                target.chunk_size = source.chunk_size
                target.start_position = source.start_position
        elif type(target) is PoolingCache:
            state = source.state
            if len(state) == 5:
                # SSD reconstruction uses the oMLX text cache layout.
                if any(value is not None for value in state[3:]):
                    raise ValueError(
                        "Cannot restore text pooling overlap into VLM cache"
                    )
                state = state[:3]
            target.meta_state = source.meta_state
            target.state = state
        else:
            target.meta_state = source.meta_state
            target.state = source.state
        return target

    return [restore(old, new) for old, new in zip(caches, templates, strict=True)]


def cache_block_size(
    block_size,
    *,
    window_sizes=(),
    has_pooling=False,
    is_mimo=False,
    has_arrays=False,
    prefill_step_size=0,
    prefill_floor=0,
    wide_prefill_step=0,
):
    """Native prefix geometry shared by scheduler and speculative engines."""
    windows = set(window_sizes)
    if len(windows) > 1:
        raise ValueError(f"Multiple rotating cache windows: {sorted(windows)}")
    if windows:
        window = next(iter(windows))
        lo, hi = 512, 1024
        if has_pooling or is_mimo:
            lo = hi = 2048
            if prefill_floor > hi and prefill_floor % window == 0:
                lo = hi = prefill_floor
        if window >= lo:
            return window
        target = ((lo + window - 1) // window) * window
        return target if target <= hi else max(window, (hi // window) * window)
    if has_arrays:
        return max(
            block_size, 2048, prefill_step_size, prefill_floor, wide_prefill_step
        )
    return block_size
