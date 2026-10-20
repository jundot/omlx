# SPDX-License-Identifier: Apache-2.0
"""Preserve untouched recurrent slots during LM and VLM cache row extraction."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_APPLIED = False


def apply_arrays_cache_extract_guard() -> bool:
    """Install the None-guarded ``ArraysCache.extract``. Idempotent."""
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

    _APPLIED = all(_guard_extract(cls) for cls in cache_classes)
    return _APPLIED


def _guard_extract(cache_class) -> bool:
    if getattr(cache_class.extract, "_omlx_none_guard", False):
        return True

    # Upstream already guarded? Probe with an all-None cache.
    try:
        probe = cache_class(1)
        cache_class.extract(probe, 0)
        return True
    except TypeError:
        pass
    except Exception:
        return False

    original_extract = cache_class.extract

    def _extract_with_none_guard(self, idx):
        cache = type(self)(len(self.cache))
        cache.cache = [
            None if c is None else c[idx : idx + 1] for c in self.cache
        ]
        return cache

    _extract_with_none_guard._omlx_none_guard = True
    _extract_with_none_guard._omlx_original = original_extract
    cache_class.extract = _extract_with_none_guard
    return True


def is_applied() -> bool:
    return _APPLIED


__all__ = ["apply_arrays_cache_extract_guard", "is_applied"]
