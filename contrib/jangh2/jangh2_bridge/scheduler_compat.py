"""Process-local Qwen4-Exp VLM cache batching compatibility, installed process-wide.

PR #364's scheduler imports mlx_lm.generate._make_cache and knows only
MLX-LM cache classes. The qwen4_exp language model installed via MLX-VLM
creates VLM ArraysCache and QSAKVCache objects instead. Keep those native
classes: they implement the recurrent and QSA state needed for inference.

The adapter also normally wraps caches in an int-offset proxy. Qwen4's
native forward uses cache type checks and batched offsets, so retain its
original cache objects rather than hiding them behind that proxy.

The hook is process-global, so it must also serve every non-JANGH2 model
loaded by the same server. mlx-lm (0.31.x in the PR #364 venv) batch-converts
only its own cache classes: any other MLX-VLM checkpoint whose make_cache()
returns mlx_vlm-native caches (Qwen3.5 family: Qwen3.6/Qwen3.8 hybrids with
mlx_vlm ArraysCache/KVCache layers) makes the stock _make_cache raise
"does not yet support batching". For those, mirror mlx-lm's conversion
with MLX-VLM's own batch classes (BatchKVCache, BatchRotatingKVCache,
ArraysCache.left_padding). Keep the resulting hybrid cache list native:
Qwen3.5-family forwards inspect cache types, index each row's offset, and
replace list entries after extracting/merging singleton cache rows. Proxies
and a copied cache list both break those operations.
"""
from __future__ import annotations

import logging
import threading
from importlib import import_module
from typing import Any

log = logging.getLogger('jangh2_bridge')
_ACTIVE = False


def _bridge_model(model: Any) -> bool:
    vlm_model = getattr(model, '_vlm_model', None)
    return bool(getattr(vlm_model, '_jangh2_bridge_model', False))


def install():
    """Install the adapter (JANGH2-native plus non-JANGH2 VLM fallback); return undo callback."""
    global _ACTIVE
    if _ACTIVE:
        raise RuntimeError('JANGH2 batch-cache compatibility hook already installed')

    import mlx.core as mx
    from mlx_vlm.models.cache import (
        ArraysCache as VlmArraysCache,
        BatchKVCache as VlmBatchKVCache,
        BatchRotatingKVCache as VlmBatchRotatingKVCache,
        CacheList as VlmCacheList,
        KVCache as VlmKVCache,
        RotatingKVCache as VlmRotatingKVCache,
    )
    from mlx_vlm.models.qwen4_exp.language import QSAKVCache, BatchQSAKVCache
    # mlx_lm exports a same-named `generate` function from its package root.
    # Resolve the actual submodule explicitly rather than that exported callable.
    generation = import_module("mlx_lm.generate")
    import omlx.scheduler as scheduler
    from omlx.engine_core import get_mlx_executor
    import omlx.models.vlm as vlm_adapter

    original_scheduler = scheduler._make_cache
    original_generate = generation._make_cache
    original_wrap = vlm_adapter._wrap_caches
    original_generation_stream = generation.generation_stream
    original_scheduler_stream = scheduler.generation_stream

    # In PR #364 mlx_lm.generate constructs generation_stream on its import
    # thread, whereas EngineCore.scheduler.step runs inside the dedicated
    # (single-worker) get_mlx_executor() thread. MLX >= 0.31.2 rejects using
    # a stream registered to another thread, even for mx.eval() on an expert.
    # Allocate the replacement ON the actual worker, before any request can
    # construct a lazy computation graph there. Merely constructing a
    # ThreadLocalStream on the import thread is not reliable in the affected
    # MLX/MLX-LM combinations (see mlx-lm issue #1256).
    def _new_worker_stream():
        return mx.new_stream(mx.default_device()), threading.get_ident()

    executor = get_mlx_executor()
    worker_stream, owner_thread = executor.submit(_new_worker_stream).result()

    def _to_vlm_batch(c, left_padding):
        # Mirror mlx_lm.generate._make_cache's to_batch_cache() for the
        # same-named MLX-VLM classes. mlx-lm's isinstance checks are against
        # its own cache module, so VLM-native layers never match there.
        if type(c) is VlmKVCache:
            return VlmBatchKVCache(left_padding)
        if isinstance(c, VlmArraysCache):
            c.left_padding = mx.array(left_padding)
            return c
        # Exact type: a subclass such as BufferedRotatingKVCache must fail
        # closed below rather than silently lose its buffering.
        if type(c) is VlmRotatingKVCache:
            if c.keep > 0:
                raise ValueError('VLM RotatingKVCache with keep tokens is not supported.')
            return VlmBatchRotatingKVCache(c.max_size, left_padding)
        if isinstance(c, VlmCacheList):
            return VlmCacheList(*(_to_vlm_batch(sub, left_padding) for sub in c.caches))
        raise ValueError(f'{type(c)} does not yet support batching')

    def batch_cache(model, left_padding, max_kv_size):
        if not _bridge_model(model):
            try:
                # Unrelated models retain their original scheduler semantics.
                return original_scheduler(model, left_padding, max_kv_size)
            except ValueError as exc:
                # mlx-lm refused to batch the model's caches. VLM checkpoints
                # with mlx_vlm-native caches (Qwen3.5 family) land here and
                # get MLX-VLM's own batch conversion; every other error, and
                # anything the VLM classes cannot handle either, propagates.
                if 'does not yet support batching' not in str(exc):
                    raise
                if not hasattr(model, 'make_cache'):
                    raise
                caches = model.make_cache()
                if not isinstance(caches, (list, tuple)):
                    raise
                return [_to_vlm_batch(c, left_padding) for c in caches]
        if max_kv_size is not None:
            raise ValueError('JANGH2 hybrid QSA/PLE caches do not support max_kv_size')
        caches = model.make_cache()
        if not isinstance(caches, (list, tuple)):
            raise TypeError('Qwen4 JANGH2 make_cache must return a sequence')
        # Validate every type before assigning padding or converting any cache.
        for c in caches:
            if isinstance(c, VlmArraysCache):
                continue
            if type(c) is QSAKVCache and callable(getattr(c, 'to_batch', None)):
                continue
            raise TypeError(f'unsupported Qwen4 JANGH2 cache: {type(c)!r}')
        batched=[]
        for c in caches:
            if isinstance(c, VlmArraysCache):
                c.left_padding = mx.array(left_padding)
                batched.append(c)
            else:
                batched.append(c.to_batch(left_padding))
        log.debug('JANGH2: prepared %d VLM-native batch cache layers', len(batched))
        return batched

    def wrap_cache(caches):
        if caches is not None and isinstance(caches, (list, tuple)) and caches:
            if all(
                isinstance(c, (VlmArraysCache, BatchQSAKVCache, QSAKVCache)) for c in caches
            ):
                # Qwen4 uses isinstance(cache, ArraysCache) and native batch QSA
                # position handling. An int-offset proxy breaks those invariants.
                return caches
            if any(isinstance(c, VlmArraysCache) for c in caches):
                # Modern Qwen3.5-family models need native batched KV offsets
                # as well as ArraysCache type identity. They also replace
                # entries in this list after singleton extraction/merge, so
                # return the scheduler's actual list rather than a copy.
                native_types = (VlmArraysCache, VlmKVCache, VlmBatchKVCache,
                                VlmRotatingKVCache, VlmBatchRotatingKVCache,
                                VlmCacheList)
                if all(isinstance(c, native_types) for c in caches):
                    return caches
                # Preserve the historical fallback for unrelated mixed
                # cache implementations rather than bypassing every proxy.
                return [
                    c if isinstance(c, VlmArraysCache) else original_wrap([c])[0]
                    for c in caches
                ]
        return original_wrap(caches)

    # The PR scheduler holds its own from-import global; newer MLX-LM code
    # may also call the module global. Both must be patched and restored.
    from .lifecycle import install_scheduler_shutdown
    undo_shutdown = install_scheduler_shutdown(scheduler.Scheduler, executor, owner_thread)
    scheduler._make_cache = batch_cache
    generation._make_cache = batch_cache
    generation.generation_stream = worker_stream
    scheduler.generation_stream = worker_stream
    vlm_adapter._wrap_caches = wrap_cache
    _ACTIVE = True
    log.info('JANGH2: installed VLM-native ArraysCache + QSA batch adapter')
    log.info('JANGH2: generation stream created on oMLX single-worker MLX executor')
    log.info('JANGH2: VLM-native batch fallback enabled for non-JANGH2 caches')

    def undo():
        global _ACTIVE
        undo_shutdown()
        if scheduler._make_cache is batch_cache:
            scheduler._make_cache = original_scheduler
        if generation._make_cache is batch_cache:
            generation._make_cache = original_generate
        if vlm_adapter._wrap_caches is wrap_cache:
            vlm_adapter._wrap_caches = original_wrap
        if generation.generation_stream is worker_stream:
            generation.generation_stream = original_generation_stream
        if scheduler.generation_stream is worker_stream:
            scheduler.generation_stream = original_scheduler_stream
        _ACTIVE = False

    return undo
