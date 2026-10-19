# SPDX-License-Identifier: Apache-2.0
"""Bound the lazy graphs behind ArraysCache padding metadata.

Batched decode builds the linear-attention mask from one layer's cache, so the
other layers' ``left_padding`` and ``lengths`` are never evaluated. Every lazy
update keeps its operand buffers alive, and a long-lived BatchGenerator then
reaches the Metal resource limit (499000 buffers).
"""

from __future__ import annotations

import mlx.core as mx

_APPLIED = False
_FIELDS = ("left_padding", "lengths")


def apply_arrays_cache_metadata_patch() -> bool:
    """Bound ArraysCache metadata graphs in mlx-lm and mlx-vlm. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return True
    try:
        from mlx_lm.models.cache import ArraysCache
    except ImportError:
        return False

    cache_classes = [ArraysCache]
    try:
        from mlx_vlm.models.cache import ArraysCache as VLMArraysCache
    except ImportError:
        pass
    else:
        cache_classes.append(VLMArraysCache)

    for cache_class in cache_classes:
        _bound_metadata(cache_class)
    _APPLIED = True
    return True


def _bound_metadata(cache_class) -> None:
    if getattr(cache_class, "_omlx_metadata_bounded", False):
        return
    # mlx-vlm already counts advance() steps on the host with these fields.
    if not isinstance(cache_class.__dict__.get("left_padding"), property):
        for name in _FIELDS:
            setattr(cache_class, name, _pending_metadata(name))
        cache_class.advance = _advance

    original_filter = cache_class.filter
    original_extend = cache_class.extend

    def filter(self, batch_indices):
        original_filter(self, batch_indices)
        _schedule_metadata(self)

    def extend(self, other):
        original_extend(self, other)
        _schedule_metadata(self)

    cache_class.filter = filter
    cache_class.extend = extend
    cache_class._omlx_metadata_bounded = True


def _pending_metadata(name: str) -> property:
    base_key = f"_{name}"
    pending_key = f"_{name}_advance"

    def fget(self):
        base = self.__dict__.get(base_key)
        pending = self.__dict__.get(pending_key, 0)
        if base is None or pending == 0:
            return base
        return base - pending

    def fset(self, value):
        self.__dict__[base_key] = value
        self.__dict__[pending_key] = 0

    return property(fget, fset)


def _advance(self, n):
    for name in _FIELDS:
        if self.__dict__.get(f"_{name}") is not None:
            key = f"_{name}_advance"
            self.__dict__[key] = self.__dict__.get(key, 0) + n


def _schedule_metadata(cache) -> None:
    # A resize rebuilds the metadata lazily; evaluate it so the next resize
    # does not chain onto this one.
    arrays = [cache.__dict__.get(f"_{name}") for name in _FIELDS]
    arrays = [a for a in arrays if a is not None]
    if arrays:
        mx.async_eval(arrays)


__all__ = ["apply_arrays_cache_metadata_patch"]
