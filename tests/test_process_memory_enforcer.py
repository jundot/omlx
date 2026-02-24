# SPDX-License-Identifier: Apache-2.0
"""Tests for ProcessMemoryEnforcer."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.process_memory_enforcer import ProcessMemoryEnforcer


def _make_entry(model_id, engine=None, is_loading=False, is_pinned=False):
    """Create a mock EngineEntry."""
    entry = MagicMock()
    entry.model_id = model_id
    entry.engine = engine
    entry.is_loading = is_loading
    entry.is_pinned = is_pinned
    entry.abort_loading = False
    return entry


@pytest.fixture
def mock_engine_pool():
    """Create a mock EnginePool with required methods."""
    pool = MagicMock()
    pool._lock = asyncio.Lock()
    pool._find_lru_victim = MagicMock(return_value="model-a")
    pool._unload_engine = AsyncMock()
    pool._entries = {}
    return pool


@pytest.fixture
def enforcer(mock_engine_pool):
    """Create an enforcer with 10GB limit."""
    return ProcessMemoryEnforcer(
        engine_pool=mock_engine_pool,
        max_bytes=10 * 1024**3,
        poll_interval=0.1,
    )


class TestCheckAndEnforce:
    """Tests for _check_and_enforce method."""

    @pytest.mark.asyncio
    async def test_no_action_when_under_limit(self, enforcer):
        """No eviction when memory is under limit."""
        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            mock_mx.get_active_memory.return_value = 5 * 1024**3
            await enforcer._check_and_enforce()
        enforcer._engine_pool._unload_engine.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_action_at_exact_limit(self, enforcer):
        """No eviction when memory is exactly at limit."""
        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            mock_mx.get_active_memory.return_value = 10 * 1024**3
            await enforcer._check_and_enforce()
        enforcer._engine_pool._unload_engine.assert_not_called()

    @pytest.mark.asyncio
    async def test_evicts_when_over_limit(self, enforcer):
        """Evicts LRU model when over limit."""
        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            mock_mx.get_active_memory.side_effect = [
                15 * 1024**3,  # Initial check (over limit)
                15 * 1024**3,  # Re-check before eviction loop
                8 * 1024**3,  # After eviction (under limit)
            ]
            await enforcer._check_and_enforce()
        enforcer._engine_pool._unload_engine.assert_called_once_with("model-a")

    @pytest.mark.asyncio
    async def test_stops_when_all_pinned(self, enforcer):
        """Stops eviction when all models are pinned (no victim)."""
        enforcer._engine_pool._find_lru_victim.return_value = None
        # Add a pinned loaded model so the log says "pinned"
        entry = _make_entry("pinned-model", engine=MagicMock(), is_pinned=True)
        enforcer._engine_pool._entries = {"pinned-model": entry}
        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            mock_mx.get_active_memory.side_effect = [
                15 * 1024**3,  # Initial check
                15 * 1024**3,  # Re-check in loop
            ]
            await enforcer._check_and_enforce()
        enforcer._engine_pool._unload_engine.assert_not_called()

    @pytest.mark.asyncio
    async def test_evicts_multiple_models(self, enforcer):
        """Evicts multiple models in sequence until under limit."""
        enforcer._engine_pool._find_lru_victim.side_effect = [
            "model-a",
            "model-b",
        ]
        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            mock_mx.get_active_memory.side_effect = [
                20 * 1024**3,  # Initial check
                20 * 1024**3,  # Re-check (still over)
                15 * 1024**3,  # After first eviction (still over)
                8 * 1024**3,  # After second eviction (under limit)
            ]
            await enforcer._check_and_enforce()
        assert enforcer._engine_pool._unload_engine.call_count == 2

    @pytest.mark.asyncio
    async def test_aborts_loading_model_when_no_lru_victim(self, enforcer):
        """Aborts a loading model when no LRU victim is available."""
        enforcer._engine_pool._find_lru_victim.return_value = None
        loading_entry = _make_entry(
            "loading-model", engine=None, is_loading=True
        )
        enforcer._engine_pool._entries = {"loading-model": loading_entry}

        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            mock_mx.get_active_memory.side_effect = [
                15 * 1024**3,  # Initial check
                15 * 1024**3,  # Re-check in loop
            ]
            await enforcer._check_and_enforce()

        assert loading_entry.abort_loading is True
        enforcer._engine_pool._unload_engine.assert_not_called()

    @pytest.mark.asyncio
    async def test_evicts_lru_before_aborting_loading(self, enforcer):
        """Evicts LRU models first, then aborts loading model."""
        # First call returns victim, second call returns None
        enforcer._engine_pool._find_lru_victim.side_effect = [
            "model-a",
            None,
        ]
        loading_entry = _make_entry(
            "loading-model", engine=None, is_loading=True
        )
        enforcer._engine_pool._entries = {"loading-model": loading_entry}

        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            mock_mx.get_active_memory.side_effect = [
                20 * 1024**3,  # Initial check
                20 * 1024**3,  # Re-check (still over)
                15 * 1024**3,  # After eviction (still over)
            ]
            await enforcer._check_and_enforce()

        # LRU victim evicted first
        enforcer._engine_pool._unload_engine.assert_called_once_with("model-a")
        # Then loading model abort requested
        assert loading_entry.abort_loading is True

    @pytest.mark.asyncio
    async def test_no_models_loaded_or_loading(self, enforcer):
        """Logs correctly when no models are loaded or loading."""
        enforcer._engine_pool._find_lru_victim.return_value = None
        enforcer._engine_pool._entries = {}

        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            mock_mx.get_active_memory.side_effect = [
                15 * 1024**3,  # Initial check
                15 * 1024**3,  # Re-check
            ]
            await enforcer._check_and_enforce()
        # Should not raise, just log warning


class TestProperties:
    """Tests for enforcer properties."""

    def test_max_bytes_getter(self, enforcer):
        """Test max_bytes property."""
        assert enforcer.max_bytes == 10 * 1024**3

    def test_max_bytes_setter(self, enforcer):
        """Test updating max_bytes at runtime."""
        enforcer.max_bytes = 20 * 1024**3
        assert enforcer.max_bytes == 20 * 1024**3

    def test_is_running_initially_false(self, enforcer):
        """Test is_running is False before start."""
        assert enforcer.is_running is False

    def test_get_status_when_not_running(self, enforcer):
        """Test get_status when enforcer is not running."""
        status = enforcer.get_status()
        assert status["enabled"] is False
        assert status["max_bytes"] == 10 * 1024**3
        assert status["current_bytes"] == 0


class TestLifecycle:
    """Tests for start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_stop(self, enforcer):
        """Test start and stop lifecycle."""
        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            mock_mx.get_active_memory.return_value = 0
            enforcer.start()
            assert enforcer.is_running is True
            await asyncio.sleep(0.05)
            await enforcer.stop()
            assert enforcer.is_running is False

    @pytest.mark.asyncio
    async def test_double_start_is_noop(self, enforcer):
        """Test calling start twice doesn't create duplicate tasks."""
        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            mock_mx.get_active_memory.return_value = 0
            enforcer.start()
            task1 = enforcer._task
            enforcer.start()
            task2 = enforcer._task
            assert task1 is task2
            await enforcer.stop()

    @pytest.mark.asyncio
    async def test_stop_when_not_started(self, enforcer):
        """Test stop when not started is safe."""
        await enforcer.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_get_status_when_running(self, enforcer):
        """Test get_status reflects running state."""
        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            mock_mx.get_active_memory.return_value = 5 * 1024**3
            enforcer.start()
            status = enforcer.get_status()
            assert status["enabled"] is True
            assert status["current_bytes"] == 5 * 1024**3
            await enforcer.stop()
