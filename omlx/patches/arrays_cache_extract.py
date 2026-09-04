# SPDX-License-Identifier: Apache-2.0
"""None-guard for ``mlx_lm.models.cache.ArraysCache.extract``.

Upstream ``ArraysCache.filter``/``extend``/``merge`` all tolerate ``None``
slots (a slot stays ``None`` until the layer's first forward touches it,
and ``merge`` of all-empty caches produces a cache whose every slot is
``None``), but ``extract`` indexes each slot unconditionally::

    cache.cache = [c[idx : idx + 1] for c in self.cache]

A request that is aborted or removed with ``return_prompt_caches=True``
before its first forward reaches ``BatchGenerator.extract_cache`` →
``CacheList.extract`` → ``ArraysCache.extract`` with such all-``None``
slots and dies on ``TypeError: 'NoneType' object is not subscriptable``.
Models wrapping ArraysCache inside CacheList per layer (inkling-style
hybrid attention + short-conv) hit this on every early abort.

The patch replaces ``extract`` with the same body plus the guard
``filter`` already uses. It retires itself automatically once upstream
adds the guard (detected by probing an all-``None`` extract).
"""

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

    if getattr(ArraysCache.extract, "_omlx_none_guard", False):
        _APPLIED = True
        return True

    # Upstream already guarded? Probe with an all-None cache.
    try:
        probe = ArraysCache(1)
        ArraysCache.extract(probe, 0)
        _APPLIED = True
        return True
    except TypeError:
        pass
    except Exception:
        return False

    original_extract = ArraysCache.extract

    def _extract_with_none_guard(self, idx):
        cache = ArraysCache(len(self.cache))
        cache.cache = [
            None if c is None else c[idx : idx + 1] for c in self.cache
        ]
        return cache

    _extract_with_none_guard._omlx_none_guard = True
    _extract_with_none_guard._omlx_original = original_extract
    ArraysCache.extract = _extract_with_none_guard
    _APPLIED = True
    return True


def is_applied() -> bool:
    return _APPLIED


__all__ = ["apply_arrays_cache_extract_guard", "is_applied"]
