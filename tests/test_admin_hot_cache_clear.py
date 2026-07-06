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

import pytest

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


def _loaded_entry(clear_hot_cache_mock, executor, stream="engine-stream"):
    """Build the entry.engine._engine.engine.scheduler chain a loaded model has."""
    scheduler = SimpleNamespace(
        paged_ssd_cache_manager=SimpleNamespace(
            clear_hot_cache=clear_hot_cache_mock,
        ),
        _cache_rate_tracker=None,
        _stream=stream,
    )
    core = SimpleNamespace(scheduler=scheduler, _mlx_executor=executor)
    return SimpleNamespace(
        engine=SimpleNamespace(
            _engine=SimpleNamespace(engine=core),
        )
    )


class _reclaim_env:
    """Patch the MLX reclaim dependencies the route imports lazily."""

    def __init__(self, footprint_before=1000, footprint_after=400):
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.clear_cache = MagicMock()
        self.synchronize = MagicMock()
        self.footprint = MagicMock(side_effect=[footprint_before, footprint_after])
        self.get_mlx_executor = MagicMock(return_value=self._executor)
        self.relieve_malloc_pressure = MagicMock(return_value=128)

    def __enter__(self):
        self._patches = [
            patch("mlx.core.clear_cache", self.clear_cache),
            patch("mlx.core.synchronize", self.synchronize),
            patch("omlx.engine_core.get_mlx_executor", self.get_mlx_executor),
            patch("omlx.utils.proc_memory.get_phys_footprint", self.footprint),
            patch(
                "omlx.utils.proc_memory.relieve_malloc_pressure",
                self.relieve_malloc_pressure,
            ),
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
        assert env.get_mlx_executor.called
        assert env.clear_cache.called, "mx.clear_cache must run even with no model loaded"
        assert env.synchronize.called, "synchronize() barrier must precede clear_cache()"
        assert result["bytes_reclaimed"] == 600

    def test_clears_loaded_model_then_reclaims(self):
        """Loaded model: its hot cache dict is cleared and buffers reclaimed."""
        clear_mock = MagicMock(return_value=7)
        with _reclaim_env() as env:
            entry = _loaded_entry(clear_mock, env._executor, stream="loaded-stream")
            pool = _pool(
                models=[{"id": MODEL_ID, "loaded": True}],
                entries={MODEL_ID: entry},
            )
            with patch.object(admin_routes, "_get_engine_pool", return_value=pool):
                result = _run_clear()

        assert clear_mock.called
        assert result["total_cleared"] == 7
        assert env.clear_cache.called
        env.synchronize.assert_any_call("loaded-stream")
        env.get_mlx_executor.assert_not_called()

    def test_response_shape(self):
        pool = _pool(models=[])
        with _reclaim_env(), patch.object(
            admin_routes, "_get_engine_pool", return_value=pool
        ):
            result = _run_clear()

        assert set(result.keys()) == {
            "status",
            "total_cleared",
            "bytes_reclaimed",
            "malloc_bytes_relieved",
        }
        assert result["status"] == "ok"
        assert result["malloc_bytes_relieved"] == 128


class TestClearReachesOrphans:
    """Orphaned hot caches are reached through the shared budget."""

    def test_clears_orphan_via_budget_with_no_model_loaded(self):
        orphan_clear = MagicMock(return_value=5)
        pool = _pool(models=[])
        pool._scheduler_config = SimpleNamespace(
            hot_cache_budget=SimpleNamespace(clear_all_owners=orphan_clear)
        )
        with _reclaim_env(), patch.object(
            admin_routes, "_get_engine_pool", return_value=pool
        ):
            result = _run_clear()

        assert orphan_clear.called
        assert result["total_cleared"] == 5

    def test_no_budget_is_tolerated(self):
        pool = _pool(models=[])
        pool._scheduler_config = SimpleNamespace(hot_cache_budget=None)
        with _reclaim_env(), patch.object(
            admin_routes, "_get_engine_pool", return_value=pool
        ):
            result = _run_clear()

        assert result["total_cleared"] == 0


class TestClearReclaimBranches:
    """Cover the per-engine reclaim branches and their fallbacks."""

    def test_engine_executor_shutdown_falls_back_to_global(self):
        """A loaded engine whose executor is already shut down still reclaims.

        run_in_executor raises 'cannot schedule new futures after shutdown';
        the route must fall back to the global executor instead of failing.
        """
        dead = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        dead.shutdown(wait=True)
        with _reclaim_env() as env:
            entry = _loaded_entry(MagicMock(return_value=0), dead, stream="dead-stream")
            pool = _pool(
                models=[{"id": MODEL_ID, "loaded": True}],
                entries={MODEL_ID: entry},
            )
            with patch.object(admin_routes, "_get_engine_pool", return_value=pool):
                result = _run_clear()

        assert result["status"] == "ok"
        assert env.get_mlx_executor.called, "must fall back to the global executor"
        assert env.clear_cache.called

    def test_non_matching_runtime_error_propagates(self):
        """Any RuntimeError other than executor-shutdown must not be swallowed."""
        bad = MagicMock()
        bad.submit = MagicMock(side_effect=RuntimeError("boom"))
        entry = _loaded_entry(MagicMock(return_value=0), bad, stream="s")
        pool = _pool(
            models=[{"id": MODEL_ID, "loaded": True}],
            entries={MODEL_ID: entry},
        )
        with _reclaim_env() as env, patch.object(
            admin_routes, "_get_engine_pool", return_value=pool
        ):
            with pytest.raises(RuntimeError, match="boom"):
                _run_clear()
        assert not env.get_mlx_executor.called, "must not fall back on a non-shutdown error"

    def test_mixed_loaded_and_orphan_no_double_count(self):
        """Loaded model and a separate orphan owner are both counted, once each."""
        with _reclaim_env() as env:
            entry = _loaded_entry(MagicMock(return_value=7), env._executor, stream="mixed-stream")
            pool = _pool(
                models=[{"id": MODEL_ID, "loaded": True}],
                entries={MODEL_ID: entry},
            )
            pool._scheduler_config = SimpleNamespace(
                hot_cache_budget=SimpleNamespace(clear_all_owners=MagicMock(return_value=5))
            )
            with patch.object(admin_routes, "_get_engine_pool", return_value=pool):
                result = _run_clear()

        assert result["total_cleared"] == 12  # 7 loaded + 5 orphan, no double count
        env.synchronize.assert_any_call("mixed-stream")
        env.get_mlx_executor.assert_not_called()

    def test_two_loaded_engines_sync_each_stream(self):
        """Each loaded engine is synchronized on its own stream; no global fallback."""
        with _reclaim_env() as env:
            e1 = _loaded_entry(MagicMock(return_value=1), env._executor, stream="stream-a")
            e2 = _loaded_entry(MagicMock(return_value=1), env._executor, stream="stream-b")
            pool = _pool(
                models=[{"id": "m1", "loaded": True}, {"id": "m2", "loaded": True}],
                entries={"m1": e1, "m2": e2},
            )
            with patch.object(admin_routes, "_get_engine_pool", return_value=pool):
                result = _run_clear()

        assert result["total_cleared"] == 2
        env.synchronize.assert_any_call("stream-a")
        env.synchronize.assert_any_call("stream-b")
        env.get_mlx_executor.assert_not_called()
