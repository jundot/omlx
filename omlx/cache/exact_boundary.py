# SPDX-License-Identifier: Apache-2.0
"""Fail-closed detached exact-boundary cache providers.

The first generic provider is intentionally narrow: an ordinary, unwrapped
``mlx_lm.models.cache.KVCache`` graph with batch size one.  Planning validates
the entire graph without allocating.  Materialization then copies only the
logical prefix into independently owned arrays and revalidates the source
before returning it to the scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx


@dataclass(frozen=True)
class _PlainKVLayerPlan:
    cache: Any
    keys: mx.array
    values: mx.array
    offset: int
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    key_dtype: Any
    value_dtype: Any


@dataclass(frozen=True)
class PlainKVBoundaryPlan:
    """Allocation-free proof for one exact plain-KV prompt boundary."""

    source_tokens: int
    source_cache_tokens: int
    target_tokens: int
    estimated_nbytes: int
    layers: tuple[_PlainKVLayerPlan, ...]


@dataclass(frozen=True)
class DetachedPlainKVBoundary:
    """Independently owned, evaluated plain-KV boundary."""

    cache: list[Any]
    arrays: tuple[mx.array, ...]
    nbytes: int
    token_count: int


@dataclass(frozen=True)
class _HybridArraysLayerPlan:
    cache: Any
    inner: Any
    wrapped: bool
    arrays: tuple[mx.array, ...]
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[Any, ...]


@dataclass(frozen=True)
class _HybridKVLayerPlan:
    cache: Any
    keys: mx.array
    values: mx.array
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    key_dtype: Any
    value_dtype: Any


@dataclass(frozen=True)
class HybridArraysKVBoundaryPlan:
    """Whole-graph proof for finalized B1 ArraysCache + KVCache state."""

    source_tokens: int
    target_tokens: int
    estimated_nbytes: int
    layers: tuple[_HybridArraysLayerPlan | _HybridKVLayerPlan, ...]


@dataclass(frozen=True)
class DetachedHybridArraysKVBoundary:
    """Independently owned, evaluated hybrid recurrent/KV boundary."""

    cache: list[Any]
    arrays: tuple[mx.array, ...]
    nbytes: int
    token_count: int


def _logical_prefix_nbytes(array: mx.array, target_tokens: int) -> int:
    shape = tuple(int(dim) for dim in array.shape)
    elements = 1
    for axis, dim in enumerate(shape):
        elements *= target_tokens if axis == 2 else dim
    return elements * int(array.itemsize)


def plan_plain_kv_boundary(
    cache_list: Any,
    *,
    source_tokens: int,
    source_cache_tokens: int,
    target_tokens: int,
) -> PlainKVBoundaryPlan | None:
    """Validate a whole plain-KV graph without allocating any MLX arrays."""

    try:
        from mlx_lm.models.cache import KVCache
    except ImportError:
        return None

    if (
        not isinstance(cache_list, list)
        or not cache_list
        or type(source_tokens) is not int
        or type(source_cache_tokens) is not int
        or type(target_tokens) is not int
        or source_tokens < 2
        or source_cache_tokens != source_tokens - 1
        or target_tokens <= 0
        or target_tokens > source_cache_tokens
    ):
        return None

    layers: list[_PlainKVLayerPlan] = []
    source_array_ids: set[int] = set()
    estimated_nbytes = 0
    for cache in cache_list:
        # Exact type is load-bearing. Subclasses may change layout, rotation,
        # quantization, or mutation semantics while retaining familiar fields.
        if type(cache) is not KVCache:
            return None
        keys = getattr(cache, "keys", None)
        values = getattr(cache, "values", None)
        offset = getattr(cache, "offset", None)
        if (
            not isinstance(keys, mx.array)
            or not isinstance(values, mx.array)
            or type(offset) is not int
            or offset != source_cache_tokens
            or keys.ndim != 4
            or values.ndim != 4
        ):
            return None
        key_shape = tuple(int(dim) for dim in keys.shape)
        value_shape = tuple(int(dim) for dim in values.shape)
        if (
            key_shape[0] != 1
            or value_shape[0] != 1
            or key_shape[:3] != value_shape[:3]
            or key_shape[2] < source_cache_tokens
            or value_shape[2] < source_cache_tokens
            or cache.size() != source_cache_tokens
            or id(keys) in source_array_ids
            or id(values) in source_array_ids
            or keys is values
        ):
            return None
        source_array_ids.update((id(keys), id(values)))
        estimated_nbytes += _logical_prefix_nbytes(keys, target_tokens)
        estimated_nbytes += _logical_prefix_nbytes(values, target_tokens)
        layers.append(
            _PlainKVLayerPlan(
                cache=cache,
                keys=keys,
                values=values,
                offset=offset,
                key_shape=key_shape,
                value_shape=value_shape,
                key_dtype=keys.dtype,
                value_dtype=values.dtype,
            )
        )

    if not layers or estimated_nbytes <= 0:
        return None
    return PlainKVBoundaryPlan(
        source_tokens=source_tokens,
        source_cache_tokens=source_cache_tokens,
        target_tokens=target_tokens,
        estimated_nbytes=estimated_nbytes,
        layers=tuple(layers),
    )


def _copy_prefix_array(array: mx.array, target_tokens: int) -> mx.array:
    """Build a distinct lazy output for the logical sequence prefix."""

    prefix = array[:, :, :target_tokens, :]
    try:
        detached = mx.copy(prefix)
    except AttributeError:
        # MLX before mx.copy: keep the established oMLX compatibility path.
        detached = prefix + mx.zeros((), dtype=prefix.dtype)
    # A plain slice may retain a much larger source allocation. Force the
    # detached logical boundary into its own compact row-contiguous buffer.
    return mx.contiguous(detached)


def _copy_exact_array(array: mx.array) -> mx.array:
    try:
        detached = mx.copy(array)
    except AttributeError:
        detached = array + mx.zeros((), dtype=array.dtype)
    return mx.contiguous(detached)


def plan_hybrid_arrays_kv_boundary(
    cache_list: Any,
    *,
    source_tokens: int,
    target_tokens: int,
) -> HybridArraysKVBoundaryPlan | None:
    """Prove an exact N-1 hybrid boundary without allocating.

    This V1 intentionally accepts only a flat graph containing both exact
    upstream ``ArraysCache`` and exact upstream ``KVCache`` leaves. Recurrent
    state must already be finalized B1 state; no rollback or trimming is
    attempted.
    """

    try:
        from mlx_lm.models.cache import ArraysCache, KVCache

        from .type_handlers import SizedArraysCache
    except ImportError:
        return None
    if (
        not isinstance(cache_list, list)
        or not cache_list
        or type(source_tokens) is not int
        or type(target_tokens) is not int
        or source_tokens < 2
        or target_tokens != source_tokens - 1
    ):
        return None

    layers: list[_HybridArraysLayerPlan | _HybridKVLayerPlan] = []
    source_array_ids: set[int] = set()
    estimated_nbytes = 0
    saw_arrays = False
    saw_kv = False
    for cache in cache_list:
        wrapped = type(cache) is SizedArraysCache
        inner = vars(cache).get("_inner") if wrapped else cache
        if type(cache) is ArraysCache or wrapped:
            if wrapped and (
                type(inner) is not ArraysCache
                or getattr(cache, "_token_count", None) != target_tokens
            ):
                return None
            state = getattr(inner, "cache", None)
            if (
                getattr(inner, "lengths", None) is not None
                or getattr(inner, "left_padding", None) is not None
                or not isinstance(state, list)
                or not state
                or any(not isinstance(value, mx.array) for value in state)
            ):
                return None
            arrays = tuple(state)
            shapes = tuple(tuple(int(dim) for dim in value.shape) for value in arrays)
            if any(
                not shape
                or shape[0] != 1
                or value.nbytes <= 0
                or id(value) in source_array_ids
                for value, shape in zip(arrays, shapes)
            ):
                return None
            source_array_ids.update(id(value) for value in arrays)
            estimated_nbytes += sum(int(value.nbytes) for value in arrays)
            layers.append(
                _HybridArraysLayerPlan(
                    cache=cache,
                    inner=inner,
                    wrapped=wrapped,
                    arrays=arrays,
                    shapes=shapes,
                    dtypes=tuple(value.dtype for value in arrays),
                )
            )
            saw_arrays = True
            continue

        if type(cache) is KVCache:
            keys = getattr(cache, "keys", None)
            values = getattr(cache, "values", None)
            if (
                not isinstance(keys, mx.array)
                or not isinstance(values, mx.array)
                or getattr(cache, "offset", None) != target_tokens
                or cache.size() != target_tokens
                or keys.ndim != 4
                or values.ndim != 4
            ):
                return None
            key_shape = tuple(int(dim) for dim in keys.shape)
            value_shape = tuple(int(dim) for dim in values.shape)
            if (
                key_shape[0] != 1
                or value_shape[0] != 1
                or key_shape[:3] != value_shape[:3]
                or key_shape[2] < target_tokens
                or value_shape[2] < target_tokens
                or id(keys) in source_array_ids
                or id(values) in source_array_ids
                or keys is values
            ):
                return None
            source_array_ids.update((id(keys), id(values)))
            estimated_nbytes += _logical_prefix_nbytes(keys, target_tokens)
            estimated_nbytes += _logical_prefix_nbytes(values, target_tokens)
            layers.append(
                _HybridKVLayerPlan(
                    cache=cache,
                    keys=keys,
                    values=values,
                    key_shape=key_shape,
                    value_shape=value_shape,
                    key_dtype=keys.dtype,
                    value_dtype=values.dtype,
                )
            )
            saw_kv = True
            continue

        # Subclasses, wrappers, quantized/rotating caches, and composites need
        # their own exact ownership/timeline proof.
        return None

    if not saw_arrays or not saw_kv or estimated_nbytes <= 0:
        return None
    return HybridArraysKVBoundaryPlan(
        source_tokens=source_tokens,
        target_tokens=target_tokens,
        estimated_nbytes=estimated_nbytes,
        layers=tuple(layers),
    )


def materialize_hybrid_arrays_kv_boundary(
    plan: HybridArraysKVBoundaryPlan,
    *,
    stream: Any,
) -> DetachedHybridArraysKVBoundary | None:
    """Detach and eagerly evaluate a proved hybrid recurrent/KV graph."""

    try:
        from mlx_lm.models.cache import ArraysCache, KVCache

        from .type_handlers import SizedArraysCache
    except ImportError:
        return None
    if not isinstance(plan, HybridArraysKVBoundaryPlan) or not plan.layers:
        return None

    # Revalidate the complete source graph before the first allocation.
    for layer in plan.layers:
        cache = layer.cache
        if isinstance(layer, _HybridArraysLayerPlan):
            inner = vars(cache).get("_inner") if layer.wrapped else cache
            if (
                (
                    type(cache) is not SizedArraysCache
                    if layer.wrapped
                    else type(cache) is not ArraysCache
                )
                or inner is not layer.inner
                or type(inner) is not ArraysCache
                or (
                    layer.wrapped
                    and getattr(cache, "_token_count", None)
                    != plan.target_tokens
                )
                or getattr(inner, "lengths", None) is not None
                or getattr(inner, "left_padding", None) is not None
                or not isinstance(getattr(inner, "cache", None), list)
                or len(inner.cache) != len(layer.arrays)
                or any(
                    current is not original
                    or tuple(int(dim) for dim in original.shape) != shape
                    or original.dtype != dtype
                    for current, original, shape, dtype in zip(
                        inner.cache,
                        layer.arrays,
                        layer.shapes,
                        layer.dtypes,
                    )
                )
            ):
                return None
        elif (
            type(cache) is not KVCache
            or cache.keys is not layer.keys
            or cache.values is not layer.values
            or cache.offset != plan.target_tokens
            or cache.size() != plan.target_tokens
            or tuple(int(dim) for dim in layer.keys.shape) != layer.key_shape
            or tuple(int(dim) for dim in layer.values.shape) != layer.value_shape
            or layer.keys.dtype != layer.key_dtype
            or layer.values.dtype != layer.value_dtype
        ):
            return None

    source_ids = {
        id(array)
        for layer in plan.layers
        for array in (
            layer.arrays
            if isinstance(layer, _HybridArraysLayerPlan)
            else (layer.keys, layer.values)
        )
    }
    cloned: list[Any] = []
    arrays: list[mx.array] = []
    try:
        with mx.stream(stream):
            for layer in plan.layers:
                if isinstance(layer, _HybridArraysLayerPlan):
                    copied = [_copy_exact_array(value) for value in layer.arrays]
                    inner_clone = ArraysCache(size=len(copied))
                    inner_clone.cache = copied
                    clone = (
                        SizedArraysCache(
                            inner_clone,
                            token_count=plan.target_tokens,
                        )
                        if layer.wrapped
                        else inner_clone
                    )
                    cloned.append(clone)
                    arrays.extend(copied)
                else:
                    keys = _copy_prefix_array(layer.keys, plan.target_tokens)
                    values = _copy_prefix_array(layer.values, plan.target_tokens)
                    clone = KVCache()
                    clone.keys = keys
                    clone.values = values
                    clone.offset = plan.target_tokens
                    cloned.append(clone)
                    arrays.extend((keys, values))
            if source_ids.intersection(id(array) for array in arrays):
                return None
            mx.eval(*arrays)
    except Exception:  # noqa: BLE001 - provider contract is fail closed
        return None

    # Re-prove source ownership/timeline after the eager barrier.
    for layer in plan.layers:
        cache = layer.cache
        if isinstance(layer, _HybridArraysLayerPlan):
            inner = vars(cache).get("_inner") if layer.wrapped else cache
            if (
                inner is not layer.inner
                or (
                    layer.wrapped
                    and getattr(cache, "_token_count", None)
                    != plan.target_tokens
                )
                or inner.lengths is not None
                or inner.left_padding is not None
                or any(
                    current is not original
                    or tuple(int(dim) for dim in original.shape) != shape
                    for current, original, shape in zip(
                        inner.cache,
                        layer.arrays,
                        layer.shapes,
                    )
                )
            ):
                return None
        elif (
            cache.keys is not layer.keys
            or cache.values is not layer.values
            or cache.offset != plan.target_tokens
            or tuple(int(dim) for dim in layer.keys.shape) != layer.key_shape
            or tuple(int(dim) for dim in layer.values.shape) != layer.value_shape
        ):
            return None
    nbytes = sum(int(array.nbytes) for array in arrays)
    if nbytes != plan.estimated_nbytes:
        return None
    return DetachedHybridArraysKVBoundary(
        cache=cloned,
        arrays=tuple(arrays),
        nbytes=nbytes,
        token_count=plan.target_tokens,
    )


def materialize_plain_kv_boundary(
    plan: PlainKVBoundaryPlan,
    *,
    stream: Any,
) -> DetachedPlainKVBoundary | None:
    """Copy and evaluate a previously proved plain-KV boundary.

    Any source mutation between planning and materialization rejects the whole
    graph. Partial clones remain private locals and are never published.
    """

    try:
        from mlx_lm.models.cache import KVCache
    except ImportError:
        return None
    if not isinstance(plan, PlainKVBoundaryPlan) or not plan.layers:
        return None

    # Phase two starts with a no-allocation revalidation of every source leaf.
    for layer in plan.layers:
        cache = layer.cache
        if (
            type(cache) is not KVCache
            or getattr(cache, "keys", None) is not layer.keys
            or getattr(cache, "values", None) is not layer.values
            or getattr(cache, "offset", None) != layer.offset
            or tuple(int(dim) for dim in layer.keys.shape) != layer.key_shape
            or tuple(int(dim) for dim in layer.values.shape) != layer.value_shape
            or layer.keys.dtype != layer.key_dtype
            or layer.values.dtype != layer.value_dtype
            or cache.size() != plan.source_cache_tokens
        ):
            return None

    source_ids = {
        identity
        for layer in plan.layers
        for identity in (id(layer.keys), id(layer.values))
    }
    cloned: list[Any] = []
    arrays: list[mx.array] = []
    try:
        with mx.stream(stream):
            for layer in plan.layers:
                keys = _copy_prefix_array(layer.keys, plan.target_tokens)
                values = _copy_prefix_array(layer.values, plan.target_tokens)
                clone = KVCache()
                clone.keys = keys
                clone.values = values
                clone.offset = plan.target_tokens
                cloned.append(clone)
                arrays.extend((keys, values))
            if source_ids.intersection(id(array) for array in arrays):
                return None
            mx.eval(*arrays)
    except Exception:  # noqa: BLE001 - provider contract is fail closed
        return None

    # Prove the source graph did not change while the detached graph evaluated.
    for layer in plan.layers:
        if (
            layer.cache.keys is not layer.keys
            or layer.cache.values is not layer.values
            or layer.cache.offset != layer.offset
            or tuple(int(dim) for dim in layer.keys.shape) != layer.key_shape
            or tuple(int(dim) for dim in layer.values.shape) != layer.value_shape
        ):
            return None
    if any(
        type(cache) is not KVCache
        or cache.offset != plan.target_tokens
        or cache.keys.shape[2] != plan.target_tokens
        or cache.values.shape[2] != plan.target_tokens
        for cache in cloned
    ):
        return None
    nbytes = sum(int(array.nbytes) for array in arrays)
    if nbytes != plan.estimated_nbytes:
        return None
    return DetachedPlainKVBoundary(
        cache=cloned,
        arrays=tuple(arrays),
        nbytes=nbytes,
        token_count=plan.target_tokens,
    )


__all__ = [
    "DetachedHybridArraysKVBoundary",
    "DetachedPlainKVBoundary",
    "HybridArraysKVBoundaryPlan",
    "PlainKVBoundaryPlan",
    "materialize_hybrid_arrays_kv_boundary",
    "materialize_plain_kv_boundary",
    "plan_hybrid_arrays_kv_boundary",
    "plan_plain_kv_boundary",
]
