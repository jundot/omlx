# SPDX-License-Identifier: Apache-2.0
"""Thread-local MLX compile-cache clearing.

MLX moved its ``mx.compile`` graph cache to a C++ ``thread_local``
``CompilerCache`` (ml-explore/mlx #3280). When a thread that ran
``@mx.compile`` functions is destroyed, ``~CompilerCache`` frees the cached
graphs' Python objects from a thread-exit handler WITHOUT holding the GIL,
which aborts the process with ``Fatal Python error: PyThreadState_Get: ...
GIL is released`` (the underlying frame is ``_pthread_exit ->
ThreadLocalVariables::finalizeList -> CompilerCache::~CompilerCache ->
tupledealloc``). Any model with module-scope ``@mx.compile`` graphs hits this:
DeepSeek V4 on unload (the per-engine worker thread exits at close), Step 3.x
on process exit.

MLX exposes no Python API to clear the compile cache. Its own atexit-based
clear at ``transforms.cpp`` is dead code (the registration lambda is never
invoked) and would run on the main thread anyway, so it cannot clear a worker
thread's thread_local cache. But the C++ symbol
``mlx::core::detail::compile_clear_cache()`` is exported from ``libmlx.dylib``,
so we resolve it via ctypes and call it ON the worker thread (with the GIL
held, via ``ctypes.PyDLL``) right before that thread is torn down. The cache
ends up empty, so ``~CompilerCache`` becomes a no-op and the thread can exit
normally.

If the symbol cannot be resolved (e.g. a future MLX rename or a non-macOS
build), the helper is a no-op and callers fall back to keeping the worker
thread alive for the process lifetime instead of exiting it.
"""

import ctypes
import logging
import os
import threading

import mlx.core as mx

logger = logging.getLogger(__name__)

# Itanium-mangled ``mlx::core::detail::compile_clear_cache()`` (no args, void).
# "19" is the length of "compile_clear_cache".
_CLEAR_SYMBOL = "_ZN3mlx4core6detail19compile_clear_cacheEv"

_resolve_lock = threading.Lock()
_resolved = False
_clear_fn = None  # ctypes callable, or None if unresolvable


def _resolve_clear_fn():
    """Resolve compile_clear_cache from libmlx.dylib once (thread-safe)."""
    global _resolved, _clear_fn
    if _resolved:
        return _clear_fn
    with _resolve_lock:
        if _resolved:
            return _clear_fn
        _resolved = True
        try:
            lib_dir = os.path.join(os.path.dirname(mx.__file__), "lib")
            libmlx = os.path.join(lib_dir, "libmlx.dylib")
            if not os.path.exists(libmlx):
                logger.warning(
                    "MLX compile-cache clear unavailable: %s not found", libmlx
                )
                return None
            # PyDLL keeps the GIL held during the call (CDLL would release it).
            # The cache clear decrefs Python objects, so the GIL MUST be held.
            lib = ctypes.PyDLL(libmlx)
            fn = None
            for name in (_CLEAR_SYMBOL, "_" + _CLEAR_SYMBOL):
                try:
                    fn = getattr(lib, name)
                    break
                except AttributeError:
                    continue
            if fn is None:
                raise AttributeError(_CLEAR_SYMBOL)
            fn.restype = None
            fn.argtypes = []
            _clear_fn = fn
            logger.info("MLX compile-cache clear resolved (%s)", _CLEAR_SYMBOL)
        except (OSError, AttributeError) as e:
            logger.warning(
                "MLX compile-cache clear unavailable (%s); MLX worker threads "
                "will be kept alive as a fallback to avoid the thread-exit "
                "~CompilerCache crash",
                e,
            )
            _clear_fn = None
        return _clear_fn


def compile_cache_clear_available() -> bool:
    """True if the thread-local compile-cache clear symbol is resolvable."""
    return _resolve_clear_fn() is not None


def clear_thread_compile_cache() -> None:
    """Clear the CALLING thread's MLX thread-local compile cache.

    MUST run ON the thread whose cache should be cleared (the cache is C++
    ``thread_local``). Call it on an MLX worker thread right before that thread
    is destroyed so ``~CompilerCache`` runs on an empty cache and never frees
    Python objects from a GIL-less thread-exit handler. No-op if the symbol
    could not be resolved.
    """
    fn = _resolve_clear_fn()
    if fn is not None:
        fn()
