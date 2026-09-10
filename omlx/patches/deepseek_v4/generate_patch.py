# SPDX-License-Identifier: Apache-2.0
"""Patch ``mlx_lm.generate._make_cache`` for PoolingCache.

PR 1192 adds one ``elif`` branch to the ``to_batch_cache`` closure inside
``_make_cache``: ``isinstance(c, PoolingCache)`` → ``BatchPoolingCache``.

The closure is not externally hookable, so we replace ``_make_cache``
with a compatible wrapper that adds PR 1192's PoolingCache conversion while
preserving any previously installed model-owned cache conversion.
``_make_cache`` is called via the module-level binding from inside
``mlx_lm.generate``, so overwriting the attribute is sufficient.

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

    previous_make_cache = _gen._make_cache

    def has_pooling_cache(cache_obj):
        if isinstance(cache_obj, PoolingCache):
            return True
        sub_caches = getattr(cache_obj, "caches", None)
        return isinstance(sub_caches, (list, tuple)) and any(
            has_pooling_cache(child) for child in sub_caches
        )

    def _patched_make_cache(model, left_padding, max_kv_size):
        """Convert a list of regular caches into their corresponding
        batch-aware caches.
        """

        def to_batch_cache(c):
            model_owned_to_batch = getattr(c, "to_batch", None)
            if callable(model_owned_to_batch):
                return model_owned_to_batch(left_padding)
            elif type(c) is KVCache:
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
            if not any(has_pooling_cache(c) for c in cache):

                class _CachedModel:
                    layers = getattr(model, "layers", ())

                    def make_cache(self):
                        return cache

                return previous_make_cache(_CachedModel(), left_padding, max_kv_size)
            return [to_batch_cache(c) for c in cache]
        else:
            if max_kv_size is not None:
                return [
                    BatchRotatingKVCache(max_kv_size, left_padding)
                    for _ in model.layers
                ]
            return [BatchKVCache(left_padding) for _ in model.layers]

    _gen._make_cache = _patched_make_cache
    _PATCHED = True
    logger.info("mlx_lm.generate._make_cache replaced (PoolingCache aware)")
    return True
