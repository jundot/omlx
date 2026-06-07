# SPDX-License-Identifier: Apache-2.0
"""Tests for POST /admin/api/hot-cache/clear.

Covers the regression where clearing freed no RAM after every model was
unloaded: the clear must still run a buffer reclaim (and report the bytes
freed) even when no scheduler is loaded.
"""

import asyncio
import concurrent.futures
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import omlx.server  # noqa: F401 — triggers set_admin_getters
import omlx.admin.routes as admin_routes


MODEL_ID = "test-model"


def _run_clear():
    return asyncio.run(admin_routes.clear_hot_cache(is_admin=True))


def _pool(models, entries=None):
    """Mock engine pool exposing get_status()['models'] and _entries."""
    pool = MagicMock(spec=[])
    pool.get_status = MagicMock(return_value={"models": models})
    pool._entries = entries or {}
    return pool


def _loaded_entry(clear_hot_cache_mock):
    """Build the entry.engine._engine.engine.scheduler chain a loaded model has."""
    scheduler = SimpleNamespace(
        paged_ssd_cache_manager=SimpleNamespace(
            clear_hot_cache=clear_hot_cache_mock,
        ),
        _cache_rate_tracker=None,
    )
    return SimpleNamespace(
        engine=SimpleNamespace(
            _engine=SimpleNamespace(engine=SimpleNamespace(scheduler=scheduler)),
        )
    )


class _reclaim_env:
    """Patch the MLX reclaim dependencies the route imports lazily."""

    def __init__(self, footprint_before=1000, footprint_after=400):
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.clear_cache = MagicMock()
        self.synchronize = MagicMock()
        self.footprint = MagicMock(side_effect=[footprint_before, footprint_after])

    def __enter__(self):
        self._patches = [
            patch("mlx.core.clear_cache", self.clear_cache),
            patch("mlx.core.synchronize", self.synchronize),
            patch("omlx.engine_core.get_mlx_executor", return_value=self._executor),
            patch("omlx.utils.proc_memory.get_phys_footprint", self.footprint),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()
        self._executor.shutdown(wait=True)
        return False


class TestHotCacheClear:
    def test_reclaims_buffers_when_no_model_loaded(self):
        """The bug case: no model loaded, yet the pool still holds the buffers.

        The clear loop has nothing to iterate, but the reclaim must still run
        so the retained Metal pool is returned to the OS.
        """
        pool = _pool(models=[])
        with _reclaim_env() as env, patch.object(
            admin_routes, "_get_engine_pool", return_value=pool
        ):
            result = _run_clear()

        assert result["total_cleared"] == 0
        assert env.clear_cache.called, "mx.clear_cache must run even with no model loaded"
        assert env.synchronize.called, "synchronize() barrier must precede clear_cache()"
        assert result["bytes_reclaimed"] == 600

    def test_clears_loaded_model_then_reclaims(self):
        """Loaded model: its hot cache dict is cleared and buffers reclaimed."""
        clear_mock = MagicMock(return_value=7)
        entry = _loaded_entry(clear_mock)
        pool = _pool(
            models=[{"id": MODEL_ID, "loaded": True}],
            entries={MODEL_ID: entry},
        )
        with _reclaim_env() as env, patch.object(
            admin_routes, "_get_engine_pool", return_value=pool
        ):
            result = _run_clear()

        assert clear_mock.called
        assert result["total_cleared"] == 7
        assert env.clear_cache.called

    def test_response_shape(self):
        pool = _pool(models=[])
        with _reclaim_env(), patch.object(
            admin_routes, "_get_engine_pool", return_value=pool
        ):
            result = _run_clear()

        assert set(result.keys()) == {"status", "total_cleared", "bytes_reclaimed"}
        assert result["status"] == "ok"
