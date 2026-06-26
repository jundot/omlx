# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.utils.compile_cache (thread-local MLX compile-cache clear)."""

import concurrent.futures as cf

import mlx.core as mx

import omlx.utils.compile_cache as cc


def test_clear_available_returns_bool():
    # Contract: never raises, always returns a bool. On a real Apple-Silicon
    # MLX install the symbol resolves, but the helper must degrade gracefully.
    assert isinstance(cc.compile_cache_clear_available(), bool)


def test_clear_thread_compile_cache_does_not_raise():
    # Safe to call on any thread, even with an empty cache.
    cc.clear_thread_compile_cache()


def test_clear_after_compile_on_worker_then_shutdown():
    """Core scenario behind the fix: run an @mx.compile fn on a worker thread,
    clear that thread's cache on the same thread, then shut the worker down.
    Must not crash (~CompilerCache runs on an empty cache)."""

    @mx.compile
    def f(x):
        return x * 2 + 1

    ex = cf.ThreadPoolExecutor(max_workers=1)
    try:
        ex.submit(lambda: mx.eval(f(mx.arange(8)))).result()
        ex.submit(cc.clear_thread_compile_cache).result()
    finally:
        ex.shutdown(wait=True)


def test_noop_when_symbol_unavailable(monkeypatch):
    """When the symbol cannot be resolved, available() is False and clear() is
    a no-op (callers then fall back to keeping the worker thread alive)."""
    monkeypatch.setattr(cc, "_resolved", True)
    monkeypatch.setattr(cc, "_clear_fn", None)
    assert cc.compile_cache_clear_available() is False
    cc.clear_thread_compile_cache()  # must not raise
