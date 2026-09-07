# SPDX-License-Identifier: Apache-2.0
"""Bridge oMLX scheduling to the ECO native Paged KV runtime."""

from __future__ import annotations

import inspect
import logging
import uuid
from typing import Any


class NativePagedKVConfigurationError(ValueError):
    """Raised when the native Paged KV runtime cannot start safely."""


class NativePagedPrefixCache:
    """Keep hybrid prefix checkpoints inside one loaded model instance."""

    def __init__(self, manager: Any, max_page_references: int):
        from mlx_lm.models.radix_cache import CacheNamespace, HybridRadixCache

        # The cache is never persisted or transferred across engine instances.
        identity = uuid.uuid4().hex
        namespace = CacheNamespace(
            model_revision=identity,
            config_fingerprint=identity,
            weights_fingerprint=identity,
            quantization=identity,
            adapter=identity,
            tokenizer_revision=identity,
            chat_template_fingerprint=identity,
            page_size=manager.page_size,
            dtype=str(manager.dtype),
        )
        self.engine = HybridRadixCache(
            namespace, manager, max_page_references=max_page_references
        )
        self.manager = manager
        self.max_page_references = max_page_references

    def acquire(self, tokens, *, request_id, max_tokens):
        result = self.engine.acquire(tokens, sequence_id=request_id)
        # Reclaim idle checkpoints before uncached prefill consumes the pool.
        from mlx_lm.models.paged_cache import PagedKVCache

        needed = (
            len(result.remaining_tokens) + max_tokens + self.manager.page_size - 1
        ) // self.manager.page_size + 1
        pools = [
            cache.pool for cache in result.caches if isinstance(cache, PagedKVCache)
        ]
        while any(pool.stats().free_pages < needed for pool in pools):
            if not self.engine.tree.evict_one(reason="pressure"):
                break
        return result

    def store(self, tokens, caches):
        from mlx_lm.models.paged_cache import PagedKVCache

        paged = [cache for cache in caches if isinstance(cache, PagedKVCache)]
        if (
            not tokens
            or not paged
            or any(cache.offset != len(tokens) for cache in paged)
        ):
            return False
        references = sum(len(cache.block_table.page_ids) for cache in paged)
        if references > self.max_page_references:
            return False
        self.engine.store(tokens, caches)
        return True

    def stats(self):
        from dataclasses import asdict

        return asdict(self.engine.stats())

    def clear(self):
        self.engine.clear()


def validate_batch_generator_contract(batch_generator_cls: type[Any]) -> None:
    """Require the mlx-lm hooks used to own physical KV pages."""
    parameters = inspect.signature(batch_generator_cls).parameters
    required = {"cache_factory", "admission_controller"}
    missing = sorted(required.difference(parameters))
    if missing:
        names = ", ".join(missing)
        raise NativePagedKVConfigurationError(
            "native Paged KV requires the IndenScale/mlx-lm fork; "
            f"BatchGenerator is missing: {names}"
        )


def create_native_paged_kv_manager(
    model: Any,
    *,
    capacity_pages: int | None,
    page_size: int = 64,
    attention_mode: str = "auto",
    batch_generator_cls: type[Any] | None = None,
) -> Any | None:
    """Create the shared GPU page pool for a registered decoder contract."""
    if capacity_pages is None:
        return None
    if capacity_pages <= 0:
        raise NativePagedKVConfigurationError(
            "native Paged KV capacity must be greater than zero"
        )
    if page_size <= 0:
        raise NativePagedKVConfigurationError(
            "native Paged KV page size must be greater than zero"
        )
    if attention_mode not in {"auto", "mlx", "gather", "direct"}:
        raise NativePagedKVConfigurationError(
            "native PagedAttention mode must be auto, mlx, gather, or direct"
        )
    if batch_generator_cls is None:
        from mlx_lm.generate import BatchGenerator

        batch_generator_cls = BatchGenerator
    validate_batch_generator_contract(batch_generator_cls)

    try:
        from mlx_lm.models.paged_cache import PagedKVCacheManager
        from mlx_lm.models.paged_capabilities import UnsupportedPagedModel
    except ImportError as exc:
        raise NativePagedKVConfigurationError(
            "native Paged KV requires the IndenScale/mlx-lm fork"
        ) from exc

    try:
        manager = PagedKVCacheManager(
            model,
            capacity_pages=capacity_pages,
            page_size=page_size,
            attention_mode=attention_mode,
        )
    except UnsupportedPagedModel as exc:
        if attention_mode != "auto":
            raise NativePagedKVConfigurationError(str(exc)) from exc
        logging.getLogger(__name__).warning(
            "Native Paged KV unavailable; using standard model cache: %s", exc
        )
        return None
    manager.materialize()
    return manager


def persistent_cache_view(caches):
    """Export physical pages as the existing portable KVCache SSD schema."""
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    result = []
    for cache in caches:
        if type(cache).__name__ == "PagedKVCache":
            keys, values = cache.state
            if keys is None:
                result.append(KVCache())
                continue
            # Resolve the gather before pages can be recycled by another request.
            mx.eval(keys, values)
            snapshot = KVCache()
            snapshot.state = (keys, values)
            result.append(snapshot)
        else:
            result.append(cache)
    return result


def restore_native_cache(manager, caches, *, sequence_id):
    """Import an oMLX prefix hit into this engine's page pool atomically."""
    import copy

    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache, KVCache
    from mlx_lm.models.paged_cache import PagedKVCache

    if len(caches) != manager.num_layers:
        raise ValueError("restored cache layer count does not match the model")
    from .cache.type_handlers import SizedArraysCache

    caches = [c._inner if isinstance(c, SizedArraysCache) else c for c in caches]
    restored = manager.make_cache(sequence_id=sequence_id)
    try:
        for index, (source, target) in enumerate(zip(caches, restored)):
            if isinstance(target, PagedKVCache):
                if isinstance(source, PagedKVCache):
                    if source.pool is not target.pool:
                        raise ValueError("restored pages belong to another model pool")
                    restored[index] = source.fork(sequence_id=target.sequence_id)
                    target.release()
                elif isinstance(source, KVCache):
                    keys, values = source.state
                    if keys is not None:
                        target._append(keys, values)
                else:
                    raise ValueError("unsupported restored full-attention cache")
            elif isinstance(source, ArraysCache):
                restored[index] = copy.deepcopy(source)
            else:
                raise ValueError("unsupported restored recurrent cache")
        manager.validate_cache(restored)
        mx.eval(
            [c.eval_state if isinstance(c, PagedKVCache) else c.state for c in restored]
        )
        return restored
    except Exception:
        manager.release(restored)
        raise
