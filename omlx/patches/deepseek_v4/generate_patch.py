# SPDX-License-Identifier: Apache-2.0
"""Patch ``mlx_lm.generate._make_cache`` for PoolingCache.

PR 1192 adds one ``elif`` branch to the ``to_batch_cache`` closure inside
``_make_cache``: ``isinstance(c, PoolingCache)`` → ``BatchPoolingCache``.

The closure is not externally hookable, so we replace ``_make_cache``
itself with a copy whose body is identical to PR 1192. ``_make_cache`` is
called via the module-level binding from inside ``mlx_lm.generate``, so
overwriting the attribute is sufficient.

When mlx-lm merges PR 1192 upstream this patch should be removed.
"""

from __future__ import annotations

import importlib
import logging

import mlx.core as mx
from mlx_lm.models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchRotatingKVCache,
    CacheList,
    KVCache,
    RotatingKVCache,
)

logger = logging.getLogger(__name__)
_PATCHED = False


def apply_generate_patch() -> bool:
    """Replace ``mlx_lm.generate._make_cache`` with a PoolingCache-aware copy.

    Must be called after ``cache_extras`` has injected ``PoolingCache`` and
    ``BatchPoolingCache`` into ``mlx_lm.models.cache`` — this function
    imports them from there.

    ``mlx_lm.__init__`` re-exports ``generate`` as a function via
    ``from .generate import generate``, which shadows the ``generate``
    submodule attribute. Use ``importlib.import_module`` to get the
    actual module object regardless.
    """
    global _PATCHED
    if _PATCHED:
        return False

    _gen = importlib.import_module("mlx_lm.generate")
    from mlx_lm.models.cache import BatchPoolingCache, PoolingCache

    def _patched_make_cache(model, left_padding, max_kv_size):
        """Convert a list of regular caches into their corresponding
        batch-aware caches.
        """

        def to_batch_cache(c):
            if type(c) is KVCache:
                return BatchKVCache(left_padding)
            elif isinstance(c, ArraysCache):
                c.left_padding = mx.array(left_padding)
                return c
            elif isinstance(c, PoolingCache):
                return BatchPoolingCache(c.ratio, left_padding)
            elif isinstance(c, RotatingKVCache):
                if c.keep > 0:
                    raise ValueError(
                        "RotatingKVCache with keep tokens is not supported."
                    )
                return BatchRotatingKVCache(c.max_size, left_padding)
            elif isinstance(c, CacheList):
                return CacheList(*(to_batch_cache(sub_c) for sub_c in c.caches))
            else:
                raise ValueError(f"{type(c)} does not yet support batching")

        if hasattr(model, "make_cache"):
            cache = model.make_cache()
            return [to_batch_cache(c) for c in cache]
        else:
            if max_kv_size is not None:
                return [
                    BatchRotatingKVCache(max_kv_size, left_padding)
                    for _ in model.layers
                ]
            return [BatchKVCache(left_padding) for _ in model.layers]

    _gen._make_cache = _patched_make_cache

    # ``BatchRotatingKVCache.merge`` normally initializes its host ``_offset``
    # from the retained physical window length. That is sufficient for generic
    # batched masking, but wrong for a singleton restored prefix: the public
    # per-row offset may be 4096 while the rotating window holds only 128 rows.
    # DS4's exact B1 WSDPA/native-mask route intentionally consumes the host
    # scalar to avoid synchronizing the public MLX vector on every layer. Keep
    # the absolute scalar for the singleton; real multi-row batches retain the
    # upstream physical-length behavior and their vector ABI.
    original_rotating_merge = BatchRotatingKVCache.merge

    def _merge_rotating_with_absolute_singleton(cls, caches):
        merged = original_rotating_merge(caches)
        if len(caches) == 1 and type(getattr(caches[0], "offset", None)) is int:
            merged._offset = int(caches[0].offset)
        return merged

    BatchRotatingKVCache.merge = classmethod(
        _merge_rotating_with_absolute_singleton
    )
    _PATCHED = True
    logger.info("mlx_lm.generate._make_cache replaced (PoolingCache aware)")
    return True
