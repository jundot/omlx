# SPDX-License-Identifier: Apache-2.0
"""Sync-before-clear primitive for the Metal buffer cache.

``mx.clear_cache()`` releases buffers from MLX's Metal buffer pool. If work
that references those buffers is still in flight, the driver can hit a
kernel panic, so every cache clear in oMLX has to drain the stream that
carried the work first. This module is the single home for that primitive
plus the lock that keeps it from racing the async store-cache worker.

Callers on an inference thread pass the stream their work rode on (the
per-engine stream for scheduler paths, ``BatchGenerator._stream`` inside
mlx-lm patches, the dependency's own stream where the dependency dispatched
the work). An mlx ``ThreadLocalStream`` resolves to a different concrete
``mx.Stream`` per calling thread, so the drain only covers the stream it is
given, resolved on the calling thread.
"""

import logging
import os
import threading
from contextlib import suppress

import mlx.core as mx
from mlx_lm.generate import generation_stream

logger = logging.getLogger(__name__)

# Module-level alias so callers can fall back to mlx-lm's default stream
# when no per-engine stream is provided.
_default_generation_stream = generation_stream

# Fase J Etapa D: byte threshold for the *gated* pool clear. A full
# mx.clear_cache() walks the whole allocator and hands every buffer back to
# Metal; on a streaming prefill that runs dozens of times per request and
# the retained pool is the page cache's main competitor (Fase G measured a
# 30.4 GiB pool and ~8x full-bank re-reads on an 8k prefill). Skipping the
# clear while the pool is small keeps the re-read cost down without
# letting the pool grow unbounded — the threshold bounds the steady state.
# Value is in GiB; 0 disables the gate (every clear runs).
_CACHE_CLEAR_THRESH_ENV = "OMLX_EXPERT_STREAMING_CACHE_THRESH"
_CACHE_CLEAR_THRESH_DEFAULT_BYTES = 2 * 1024**3

# Serializes Metal buffer-protocol access from the async store-cache worker
# against inference-thread mx.clear_cache / mx.synchronize calls that can
# invalidate the underlying buffer pool. Closes a SIGABRT path where
# _async_store_cache_worker reads tensor bytes via memoryview while the
# inference thread concurrently issues a reclaim-triggering mx op.
# See: https://github.com/jundot/omlx/issues/1106
_mx_buffer_access_lock = threading.RLock()


def clear_thread_streams() -> None:
    """Release every MLX stream owned by the current worker thread.

    MLX keeps a per-thread stream registry. Synchronizing and clearing the
    buffer cache does not remove those entries, so a worker that touched MLX
    must call ``mx.clear_streams()`` immediately before it exits.
    """
    # A ThreadPoolExecutor starts lazily. If this is the worker's first task,
    # no default stream exists yet and there is nothing to synchronize.
    with suppress(RuntimeError):
        mx.synchronize()
    mx.clear_streams()


def _sync_and_clear_cache(stream=None):
    """Synchronize in-flight GPU work before clearing the Metal buffer cache.

    Without synchronization, mx.clear_cache() can release Metal buffers that
    are still referenced by in-flight command buffers submitted via
    mx.async_eval(). This causes the GPU driver to hit a
    'completeMemory() prepare count underflow' kernel panic on M4 hardware
    (and SIGSEGV/SIGABRT on M3).

    Held under _mx_buffer_access_lock so the async store-cache worker cannot
    observe a half-reclaimed Metal buffer pool while it is in the middle of
    reading tensor bytes via the Python buffer protocol (#1106).

    See: https://github.com/jundot/omlx/issues/300, #888, #1106
    """
    with _mx_buffer_access_lock:
        # The engine stream may not have in-flight work on the current thread
        # (for example, during teardown before that thread submits work). On
        # some MLX builds mx.synchronize raises "There is no Stream(gpu, 0) in
        # current thread" in that case; swallow it since there is nothing to
        # drain.
        target = stream if stream is not None else _default_generation_stream
        try:
            mx.synchronize(target)
        except RuntimeError:
            pass
        mx.synchronize()  # default stream
        mx.clear_cache()


def cache_clear_threshold_bytes() -> int:
    """Pool size above which a gated ``mx.clear_cache()`` is worth running.

    Single source of truth for ``OMLX_EXPERT_STREAMING_CACHE_THRESH``
    (GiB): the scheduler's chunk-boundary clears and the streaming per-layer
    eval boundary gate on the same number, so they cannot drift apart.
    Falls back to 2 GiB when unset or unparseable.
    """
    raw = os.environ.get(_CACHE_CLEAR_THRESH_ENV)
    if raw:
        try:
            return max(0, int(float(raw) * 1024**3))
        except ValueError:
            logger.warning("Invalid %s=%r", _CACHE_CLEAR_THRESH_ENV, raw)
    return _CACHE_CLEAR_THRESH_DEFAULT_BYTES


def should_clear_cache(threshold_bytes: int | None = None) -> bool:
    """Whether the MLX buffer pool is worth trimming right now.

    Fase J Etapa D. The pool is only worth trimming once it has grown past
    *threshold_bytes* (``cache_clear_threshold_bytes`` by default); below
    that, clearing costs more than it saves — an allocator walk plus
    eviction of the page cache holding the mmap'd expert shards the next
    bank re-reads (Fase G measured ~8x full-bank re-reads at a 30.4 GiB
    pool).

    This returns only the *decision*. Callers keep their own
    ``_sync_and_clear_cache`` call site so the sync-before-clear primitive
    stays a single patchable choke point per module — the scheduler's is
    instrumented by the prefill tests, and routing through a shared helper
    would silently bypass that.

    When ``mx.get_cache_memory`` is unavailable the pool cannot be measured,
    so this returns True: the conservative, pre-Fase-J behavior.
    """
    get_cache_memory = getattr(mx, "get_cache_memory", None)
    if get_cache_memory is None:
        return True
    threshold = (
        cache_clear_threshold_bytes() if threshold_bytes is None else threshold_bytes
    )
    return get_cache_memory() >= threshold
