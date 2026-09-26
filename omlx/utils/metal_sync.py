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

import threading
import time
from collections import deque
from contextlib import suppress

import mlx.core as mx
from mlx_lm.generate import generation_stream

from .fatal import exit_if_gpu_submissions_ignored
from .proc_memory import get_graphics_footprint

# Module-level alias so callers can fall back to mlx-lm's default stream
# when no per-engine stream is provided.
_default_generation_stream = generation_stream

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
        except RuntimeError as exc:
            exit_if_gpu_submissions_ignored(exc)
        try:
            mx.synchronize()  # default stream
            mx.clear_cache()
        except RuntimeError as exc:
            exit_if_gpu_submissions_ignored(exc)
            raise


# The kernel footprint keeps charging freed Metal buffers until the driver
# finishes releasing them: 0.1-0.3s on macOS 27, over 1s on some macOS 26
# builds. MLX releases buffers on mx.clear_cache, when a pool trim makes room
# under its memory limit, and on free past the cache limit, so the lag is
# measured instead of tracked per release: the graphics footprint above MLX's
# own bytes, less the settled level of other Metal memory.
_RESIDUAL_WINDOW_S = 10.0
_FRESH_RESULT_S = 2.0
_residuals: deque[tuple[float, int]] = deque()
_last_unreleased: tuple[float, int] = (0.0, 0)
_residual_lock = threading.Lock()


def unreleased_graphics_bytes(mlx_bytes: int, *, fresh: bool = True) -> int:
    """Freed Metal bytes the kernel footprint still charges.

    ``mlx_bytes`` is MLX active + pool, which drops the moment MLX releases a
    buffer. The recent minimum of graphics footprint minus ``mlx_bytes`` is
    Metal memory outside MLX; anything above it is pending driver release.

    ``fresh=False`` marks a cached, possibly stale MLX sample (the enforcer
    thread must not call MLX): it reuses the last fresh result instead of
    deriving one from mismatched readings.
    """
    global _last_unreleased
    now = time.monotonic()
    if not fresh:
        with _residual_lock:
            at, value = _last_unreleased
        return value if now - at <= _FRESH_RESULT_S else 0
    graphics = get_graphics_footprint()
    if graphics <= 0:
        return 0
    residual = graphics - max(0, int(mlx_bytes))
    with _residual_lock:
        _residuals.append((now, residual))
        while _residuals and now - _residuals[0][0] > _RESIDUAL_WINDOW_S:
            _residuals.popleft()
        settled = max(0, min(value for _, value in _residuals))
        unreleased = max(0, residual - settled)
        _last_unreleased = (now, unreleased)
    return unreleased


# Per-owner MLX memory limits requested by in-flight prefill chunks. MLX's
# limit is process-wide, so the tightest request wins and the original limit
# returns when no chunk holds one.
_limit_lock = threading.Lock()
_chunk_memory_limits: dict[int, int] = {}
_default_memory_limit: int | None = None


def set_chunk_memory_limit(owner: int, limit: int | None) -> None:
    """Apply or drop one owner's MLX memory limit.

    Above the limit, MLX stops encoding ahead of the GPU until in-flight
    command buffers retire and free their intermediates, and it trims the
    buffer pool before growing it. It never refuses an allocation, so this
    bounds a lazy prefill graph's peak without failing the chunk.
    """
    global _default_memory_limit
    with _limit_lock:
        if limit is None:
            if _chunk_memory_limits.pop(owner, None) is None:
                return
        else:
            _chunk_memory_limits[owner] = max(1, int(limit))
        if _chunk_memory_limits:
            effective = min(_chunk_memory_limits.values())
            previous = mx.set_memory_limit(effective)
            if _default_memory_limit is None:
                _default_memory_limit = previous
        elif _default_memory_limit is not None:
            mx.set_memory_limit(_default_memory_limit)
            _default_memory_limit = None
