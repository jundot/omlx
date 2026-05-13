# SPDX-License-Identifier: Apache-2.0
"""Tests for ProcessMemoryEnforcer."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.process_memory_enforcer import ProcessMemoryEnforcer


def _cycling(values):
    """side_effect helper: yield each value, then repeat the last forever.

    Lets tests express the meaningful sequence of mocked memory values
    without having to count exact call sites in _check_and_enforce (the
    new 2-watermark path re-reads phys_footprint after eviction).
    """
    if not values:
        raise ValueError("need at least one value")
    state = {"i": 0}

    def _next(*_args, **_kwargs):
        i = state["i"]
        if i < len(values) - 1:
            state["i"] = i + 1
        return values[i]

    return _next


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
    """Create an enforcer with 10GB limit.

    Soft/hard thresholds set to 1.0 so legacy single-threshold tests keep
    treating max_bytes as the single trip point. Dedicated 2-watermark
    tests construct their own enforcer with default thresholds.
    """
    return ProcessMemoryEnforcer(
        engine_pool=mock_engine_pool,
        max_bytes=10 * 1024**3,
        poll_interval=0.1,
        soft_threshold=1.0,
        hard_threshold=1.0,
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
        """Evicts LRU model when over limit (multiple models loaded)."""
        # Need at least 2 loaded non-pinned models for eviction path
        engine_a = MagicMock()
        engine_a.abort_all_requests = AsyncMock(return_value=0)
        engine_b = MagicMock()
        engine_b.abort_all_requests = AsyncMock(return_value=0)
        entry_a = _make_entry("model-a", engine=engine_a)
        entry_b = _make_entry("model-b", engine=engine_b)
        enforcer._engine_pool._entries = {
            "model-a": entry_a,
            "model-b": entry_b,
        }
        enforcer._engine_pool._find_lru_victim.return_value = "model-a"

        async def fake_unload(model_id):
            enforcer._engine_pool._entries[model_id].engine = None

        enforcer._engine_pool._unload_engine.side_effect = fake_unload

        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            mock_mx.get_active_memory.side_effect = _cycling([
                15 * 1024**3,  # Initial check (over limit)
                15 * 1024**3,  # Re-check before eviction loop
                8 * 1024**3,  # After eviction (under limit)
            ])
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
            mock_mx.get_active_memory.side_effect = _cycling([
                15 * 1024**3,  # Initial check
                15 * 1024**3,  # Re-check in loop
            ])
            await enforcer._check_and_enforce()
        enforcer._engine_pool._unload_engine.assert_not_called()

    @pytest.mark.asyncio
    async def test_evicts_multiple_models(self, enforcer):
        """Evicts multiple models in sequence until under limit."""
        # Need 3 loaded non-pinned models for sequential eviction
        engine_a = MagicMock()
        engine_a.abort_all_requests = AsyncMock(return_value=0)
        engine_b = MagicMock()
        engine_b.abort_all_requests = AsyncMock(return_value=0)
        engine_c = MagicMock()
        engine_c.abort_all_requests = AsyncMock(return_value=0)
        entry_a = _make_entry("model-a", engine=engine_a)
        entry_b = _make_entry("model-b", engine=engine_b)
        entry_c = _make_entry("model-c", engine=engine_c)
        enforcer._engine_pool._entries = {
            "model-a": entry_a,
            "model-b": entry_b,
            "model-c": entry_c,
        }
        enforcer._engine_pool._find_lru_victim.side_effect = [
            "model-a",
            "model-b",
        ]

        async def fake_unload(model_id):
            enforcer._engine_pool._entries[model_id].engine = None

        enforcer._engine_pool._unload_engine.side_effect = fake_unload

        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            mock_mx.get_active_memory.side_effect = _cycling([
                20 * 1024**3,  # Initial check
                20 * 1024**3,  # Re-check (still over)
                15 * 1024**3,  # After first eviction (still over)
                8 * 1024**3,  # After second eviction (under limit)
            ])
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
            mock_mx.get_active_memory.side_effect = _cycling([
                15 * 1024**3,  # Initial check
                15 * 1024**3,  # Re-check in loop
            ])
            await enforcer._check_and_enforce()

        assert loading_entry.abort_loading is True
        enforcer._engine_pool._unload_engine.assert_not_called()

    @pytest.mark.asyncio
    async def test_evicts_lru_before_aborting_loading(self, enforcer):
        """Evicts LRU models first, then aborts loading model."""
        # Need 2 loaded non-pinned so model-a gets evicted (not abort path)
        engine_a = MagicMock()
        engine_a.abort_all_requests = AsyncMock(return_value=0)
        engine_b = MagicMock()
        engine_b.abort_all_requests = AsyncMock(return_value=0)
        entry_a = _make_entry("model-a", engine=engine_a)
        entry_b = _make_entry("model-b", engine=engine_b)
        loading_entry = _make_entry(
            "loading-model", engine=None, is_loading=True
        )
        enforcer._engine_pool._entries = {
            "model-a": entry_a,
            "model-b": entry_b,
            "loading-model": loading_entry,
        }

        async def fake_unload(model_id):
            enforcer._engine_pool._entries[model_id].engine = None

        enforcer._engine_pool._unload_engine.side_effect = fake_unload

        # First call returns victim, second call returns None
        enforcer._engine_pool._find_lru_victim.side_effect = [
            "model-a",
            None,
        ]

        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            mock_mx.get_active_memory.side_effect = _cycling([
                20 * 1024**3,  # Initial check
                20 * 1024**3,  # Re-check (still over)
                15 * 1024**3,  # After eviction (still over)
            ])
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
            mock_mx.get_active_memory.side_effect = _cycling([
                15 * 1024**3,  # Initial check
                15 * 1024**3,  # Re-check
            ])
            await enforcer._check_and_enforce()
        # Should not raise, just log warning


class TestDisabledWhenMaxBytesZero:
    """Tests for enforcement disabled when max_bytes <= 0."""

    @pytest.mark.asyncio
    async def test_no_enforce_when_max_bytes_zero(self, mock_engine_pool):
        """No enforcement when max_bytes is 0 (disabled)."""
        enforcer = ProcessMemoryEnforcer(
            engine_pool=mock_engine_pool, max_bytes=0
        )
        engine = MagicMock()
        engine.abort_all_requests = AsyncMock(return_value=0)
        entry = _make_entry("model-a", engine=engine)
        mock_engine_pool._entries = {"model-a": entry}

        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            mock_mx.get_active_memory.return_value = 50 * 1024**3
            await enforcer._check_and_enforce()

        engine.abort_all_requests.assert_not_awaited()
        mock_engine_pool._unload_engine.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_enforce_when_max_bytes_negative(self, mock_engine_pool):
        """No enforcement when max_bytes is negative."""
        enforcer = ProcessMemoryEnforcer(
            engine_pool=mock_engine_pool, max_bytes=-1
        )
        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            mock_mx.get_active_memory.return_value = 50 * 1024**3
            await enforcer._check_and_enforce()

        mock_engine_pool._unload_engine.assert_not_called()

    @pytest.mark.asyncio
    async def test_propagate_zero_disables_inline_prefill_check(
        self, mock_engine_pool
    ):
        """Propagating max_bytes=0 sets scheduler limit to 0 (disabled)."""
        enforcer = ProcessMemoryEnforcer(
            engine_pool=mock_engine_pool, max_bytes=0
        )
        bg = MagicMock(spec=[])
        bg._memory_limit_bytes = 999
        bg._memory_hard_limit_bytes = 999
        scheduler = MagicMock(spec=[])
        scheduler._memory_limit_bytes = 999
        scheduler._memory_hard_limit_bytes = 999
        scheduler.batch_generator = bg
        engine = MagicMock(spec=[])
        engine.scheduler = scheduler
        entry = _make_entry("model-a", engine=engine)
        mock_engine_pool._entries = {"model-a": entry}

        enforcer._propagate_memory_limit()

        assert scheduler._memory_limit_bytes == 0
        assert scheduler._memory_hard_limit_bytes == 0
        assert bg._memory_limit_bytes == 0
        assert bg._memory_hard_limit_bytes == 0


class TestPrefillMemoryGuardToggle:
    """Tests for prefill_memory_guard setter and Metal limit management."""

    def test_enable_guard_is_noop_for_metal_limits(self, enforcer):
        """Enabling guard does NOT call Metal limits (no-op since #429)."""
        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            enforcer._running = True

            enforcer.prefill_memory_guard = True
            assert enforcer.prefill_memory_guard is True
            mock_mx.set_memory_limit.assert_not_called()
            mock_mx.set_cache_limit.assert_not_called()

    def test_disable_guard_is_noop_for_metal_limits(self, enforcer):
        """Disabling guard does NOT call Metal limits (no-op since #429)."""
        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            enforcer._running = True

            enforcer.prefill_memory_guard = True
            enforcer.prefill_memory_guard = False
            assert enforcer.prefill_memory_guard is False
            mock_mx.set_memory_limit.assert_not_called()
            mock_mx.set_cache_limit.assert_not_called()

    def test_disable_guard_noop_without_prior_limits(self, enforcer):
        """Disabling guard when no limits were set does not call mx."""
        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            enforcer._running = True

            # Disable without enabling first
            enforcer.prefill_memory_guard = False
            mock_mx.set_memory_limit.assert_not_called()
            mock_mx.set_cache_limit.assert_not_called()


class TestHardLimitCalculation:
    """Tests for _get_hard_limit_bytes calculation."""

    def test_hard_limit_is_system_ram_minus_4gb(self, enforcer):
        """Hard limit = system_ram - 4GB."""
        with patch("omlx.settings.get_system_memory") as mock_mem:
            mock_mem.return_value = 96 * 1024**3
            result = enforcer._get_hard_limit_bytes()
        assert result == 92 * 1024**3

    def test_hard_limit_at_least_max_bytes(self, mock_engine_pool):
        """Hard limit is at least max_bytes (for small systems)."""
        # 16GB system, 14GB soft limit -> system-4GB = 12GB < 14GB
        enforcer = ProcessMemoryEnforcer(
            engine_pool=mock_engine_pool, max_bytes=14 * 1024**3
        )
        with patch("omlx.settings.get_system_memory") as mock_mem:
            mock_mem.return_value = 16 * 1024**3
            result = enforcer._get_hard_limit_bytes()
        assert result == 14 * 1024**3

    def test_hard_limit_zero_when_disabled(self, mock_engine_pool):
        """Hard limit is 0 when max_bytes <= 0 (disabled)."""
        enforcer = ProcessMemoryEnforcer(
            engine_pool=mock_engine_pool, max_bytes=0
        )
        assert enforcer._get_hard_limit_bytes() == 0


class TestSingleModelMemoryPressure:
    """Tests for single-model memory pressure handling (Issue #62).

    Verifies three scenarios:
    1. Two models, one inferring: evict idle LRU, inference continues
    2. Single model: abort requests, keep model loaded
    3. Two models both inferring: evict LRU, then abort remaining
    """

    @pytest.mark.asyncio
    async def test_single_model_aborts_not_evicts(self, enforcer):
        """Scenario 2: Single model aborts requests instead of evicting."""
        engine = MagicMock()
        engine.abort_all_requests = AsyncMock(return_value=3)
        entry = _make_entry("big-model", engine=engine)
        enforcer._engine_pool._entries = {"big-model": entry}
        enforcer._engine_pool._find_lru_victim.return_value = "big-model"

        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            mock_mx.get_active_memory.side_effect = _cycling([
                15 * 1024**3,  # Initial check
                15 * 1024**3,  # While loop check
            ])
            await enforcer._check_and_enforce()

        engine.abort_all_requests.assert_awaited_once()
        enforcer._engine_pool._unload_engine.assert_not_awaited()
        assert entry.engine is not None

    @pytest.mark.asyncio
    async def test_single_model_no_active_requests(self, enforcer):
        """Scenario 2 variant: No requests to abort, model still kept."""
        engine = MagicMock()
        engine.abort_all_requests = AsyncMock(return_value=0)
        entry = _make_entry("big-model", engine=engine)
        enforcer._engine_pool._entries = {"big-model": entry}
        enforcer._engine_pool._find_lru_victim.return_value = "big-model"

        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            mock_mx.get_active_memory.side_effect = _cycling([
                15 * 1024**3,
                15 * 1024**3,
            ])
            await enforcer._check_and_enforce()

        engine.abort_all_requests.assert_awaited_once()
        enforcer._engine_pool._unload_engine.assert_not_awaited()
        assert entry.engine is not None

    @pytest.mark.asyncio
    async def test_two_models_one_inferring_evicts_idle(self, enforcer):
        """Scenario 1: Two models, only one inferring. Evict idle LRU."""
        engine_active = MagicMock()
        engine_active.abort_all_requests = AsyncMock(return_value=0)
        engine_idle = MagicMock()
        engine_idle.abort_all_requests = AsyncMock(return_value=0)

        entry_active = _make_entry(
            "active-model", engine=engine_active
        )
        entry_idle = _make_entry(
            "idle-model", engine=engine_idle
        )
        enforcer._engine_pool._entries = {
            "active-model": entry_active,
            "idle-model": entry_idle,
        }
        enforcer._engine_pool._find_lru_victim.return_value = "idle-model"

        async def fake_unload(model_id):
            enforcer._engine_pool._entries[model_id].engine = None

        enforcer._engine_pool._unload_engine.side_effect = fake_unload

        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            mock_mx.get_active_memory.side_effect = _cycling([
                15 * 1024**3,  # Initial check
                15 * 1024**3,  # While loop check
                8 * 1024**3,  # After eviction (under limit)
            ])
            await enforcer._check_and_enforce()

        enforcer._engine_pool._unload_engine.assert_awaited_once_with(
            "idle-model"
        )
        # Idle model's requests aborted before eviction (0 requests)
        engine_idle.abort_all_requests.assert_awaited_once()
        # Active model's requests NOT aborted
        engine_active.abort_all_requests.assert_not_awaited()
        assert entry_active.engine is not None

    @pytest.mark.asyncio
    async def test_two_models_both_inferring_evict_then_abort(self, enforcer):
        """Scenario 3: Both models inferring. Evict LRU, abort remaining."""
        engine_a = MagicMock()
        engine_a.abort_all_requests = AsyncMock(return_value=2)
        engine_b = MagicMock()
        engine_b.abort_all_requests = AsyncMock(return_value=1)

        entry_a = _make_entry("model-a", engine=engine_a)
        entry_b = _make_entry("model-b", engine=engine_b)
        enforcer._engine_pool._entries = {
            "model-a": entry_a,
            "model-b": entry_b,
        }
        # First iteration: model-b is LRU. After eviction: model-a is sole.
        enforcer._engine_pool._find_lru_victim.side_effect = [
            "model-b",
            "model-a",
        ]

        async def fake_unload(model_id):
            enforcer._engine_pool._entries[model_id].engine = None

        enforcer._engine_pool._unload_engine.side_effect = fake_unload

        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            # Memory stays over limit throughout
            mock_mx.get_active_memory.return_value = 15 * 1024**3
            await enforcer._check_and_enforce()

        # model-b evicted (requests aborted before eviction)
        enforcer._engine_pool._unload_engine.assert_awaited_once_with(
            "model-b"
        )
        # model-b's requests aborted before eviction
        engine_b.abort_all_requests.assert_awaited_once()
        # model-a's requests aborted (single-model path, second iteration)
        engine_a.abort_all_requests.assert_awaited_once()
        # model-a still loaded
        assert entry_a.engine is not None


class TestMemoryLimitPropagation:
    """Tests for soft/hard memory limit propagation to schedulers."""

    def test_propagate_memory_limit(self, enforcer):
        """Propagates soft and hard limits to scheduler and batch_generator."""
        bg = MagicMock(spec=[])
        bg._memory_limit_bytes = 0
        bg._memory_hard_limit_bytes = 0
        scheduler = MagicMock(spec=[])
        scheduler._memory_limit_bytes = 0
        scheduler._memory_hard_limit_bytes = 0
        scheduler.batch_generator = bg
        engine = MagicMock(spec=[])
        engine.scheduler = scheduler
        entry = _make_entry("model-a", engine=engine)
        enforcer._engine_pool._entries = {"model-a": entry}

        with patch("omlx.settings.get_system_memory") as mock_mem:
            mock_mem.return_value = 96 * 1024**3
            enforcer._propagate_memory_limit()

        assert scheduler._memory_limit_bytes == 10 * 1024**3
        assert bg._memory_limit_bytes == 10 * 1024**3
        # hard limit = 96GB - 4GB = 92GB
        assert scheduler._memory_hard_limit_bytes == 92 * 1024**3
        assert bg._memory_hard_limit_bytes == 92 * 1024**3

    def test_propagates_on_max_bytes_change(self, enforcer):
        """Propagates updated limits when max_bytes is changed at runtime."""
        bg = MagicMock(spec=[])
        bg._memory_limit_bytes = 0
        bg._memory_hard_limit_bytes = 0
        scheduler = MagicMock(spec=[])
        scheduler._memory_limit_bytes = 0
        scheduler._memory_hard_limit_bytes = 0
        scheduler.batch_generator = bg
        engine = MagicMock(spec=[])
        engine.scheduler = scheduler
        entry = _make_entry("model-a", engine=engine)
        enforcer._engine_pool._entries = {"model-a": entry}

        enforcer._running = True
        with patch("omlx.settings.get_system_memory") as mock_mem:
            mock_mem.return_value = 96 * 1024**3
            enforcer.max_bytes = 20 * 1024**3

        assert scheduler._memory_limit_bytes == 20 * 1024**3
        assert bg._memory_limit_bytes == 20 * 1024**3

    def test_skips_engine_without_scheduler(self, enforcer):
        """Gracefully skips engines without scheduler attribute."""
        engine = MagicMock(spec=[])
        # No scheduler attribute (spec=[] prevents auto-creation)
        entry = _make_entry("model-a", engine=engine)
        enforcer._engine_pool._entries = {"model-a": entry}

        # Should not raise
        enforcer._propagate_memory_limit()

    def test_propagates_to_multiple_engines(self, enforcer):
        """Propagates to all engines."""
        schedulers = []
        entries = {}
        for i in range(3):
            bg = MagicMock(spec=[])
            bg._memory_limit_bytes = 0
            scheduler = MagicMock(spec=[])
            scheduler._memory_limit_bytes = 0
            scheduler.batch_generator = bg
            schedulers.append(scheduler)
            engine = MagicMock(spec=[])
            engine.scheduler = scheduler
            entry = _make_entry(f"model-{i}", engine=engine)
            entries[f"model-{i}"] = entry
        enforcer._engine_pool._entries = entries

        enforcer._propagate_memory_limit()

        for scheduler in schedulers:
            assert scheduler._memory_limit_bytes == 10 * 1024**3


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


class TestTwoWatermarkPressureLevels:
    """Tests for 2-watermark soft/hard pressure level handling."""

    @pytest.fixture
    def pool(self):
        p = MagicMock()
        p._lock = asyncio.Lock()
        p._find_lru_victim = MagicMock(return_value=None)
        p._unload_engine = AsyncMock()
        p._entries = {}
        return p

    @pytest.fixture
    def enforcer_2wm(self, pool):
        return ProcessMemoryEnforcer(
            engine_pool=pool,
            max_bytes=100 * 1024**3,
            poll_interval=0.1,
            soft_threshold=0.85,
            hard_threshold=0.95,
        )

    def test_soft_hard_bytes_computed(self, enforcer_2wm):
        assert enforcer_2wm._soft_bytes == int(100 * 1024**3 * 0.85)
        assert enforcer_2wm._hard_bytes == int(100 * 1024**3 * 0.95)

    def test_get_pressure_level_when_not_running(self, enforcer_2wm):
        # _running=False → always ok regardless of cached level
        enforcer_2wm._pressure_level = "hard"
        assert enforcer_2wm.get_pressure_level() == "ok"

    def test_get_pressure_level_when_running_returns_cached(self, enforcer_2wm):
        enforcer_2wm._running = True
        enforcer_2wm._pressure_level = "soft"
        assert enforcer_2wm.get_pressure_level() == "soft"

    @pytest.mark.asyncio
    async def test_ok_when_below_soft(self, enforcer_2wm):
        with patch("omlx.process_memory_enforcer.mx") as mock_mx, \
             patch("omlx.process_memory_enforcer.get_phys_footprint") as gpf:
            mock_mx.get_active_memory.return_value = 50 * 1024**3
            gpf.return_value = 50 * 1024**3
            await enforcer_2wm._check_and_enforce()
        assert enforcer_2wm._pressure_level == "ok"
        enforcer_2wm._engine_pool._unload_engine.assert_not_called()

    @pytest.mark.asyncio
    async def test_soft_when_active_low_but_phys_high(self, enforcer_2wm):
        """phys_footprint dominates active — the #702 case."""
        with patch("omlx.process_memory_enforcer.mx") as mock_mx, \
             patch("omlx.process_memory_enforcer.get_phys_footprint") as gpf:
            # active well below soft, phys above soft but below hard
            mock_mx.get_active_memory.return_value = 50 * 1024**3
            gpf.return_value = 88 * 1024**3
            await enforcer_2wm._check_and_enforce()
        assert enforcer_2wm._pressure_level == "soft"

    @pytest.mark.asyncio
    async def test_hard_when_phys_at_hard_threshold(self, enforcer_2wm):
        with patch("omlx.process_memory_enforcer.mx") as mock_mx, \
             patch("omlx.process_memory_enforcer.get_phys_footprint") as gpf:
            mock_mx.get_active_memory.return_value = 60 * 1024**3
            gpf.return_value = 98 * 1024**3
            await enforcer_2wm._check_and_enforce()
        assert enforcer_2wm._pressure_level == "hard"

    @pytest.mark.asyncio
    async def test_propagates_admission_paused_on_soft(self, enforcer_2wm, pool):
        # Wire a scheduler-like mock so propagate has something to set.
        engine = MagicMock()
        scheduler = MagicMock()
        scheduler._memory_limit_bytes = 0
        scheduler._memory_hard_limit_bytes = 0
        scheduler._prefill_memory_guard = False
        scheduler._admission_paused = False
        engine.scheduler = scheduler
        entry = _make_entry("m", engine=engine)
        pool._entries = {"m": entry}

        with patch("omlx.process_memory_enforcer.mx") as mock_mx, \
             patch("omlx.process_memory_enforcer.get_phys_footprint") as gpf:
            mock_mx.get_active_memory.return_value = 50 * 1024**3
            gpf.return_value = 88 * 1024**3
            await enforcer_2wm._check_and_enforce()

        assert scheduler._admission_paused is True

    @pytest.mark.asyncio
    async def test_clears_admission_paused_on_recovery(self, enforcer_2wm, pool):
        engine = MagicMock()
        scheduler = MagicMock()
        scheduler._memory_limit_bytes = 0
        scheduler._memory_hard_limit_bytes = 0
        scheduler._prefill_memory_guard = False
        scheduler._admission_paused = True
        engine.scheduler = scheduler
        entry = _make_entry("m", engine=engine, is_pinned=True)
        pool._entries = {"m": entry}

        # Force into soft first
        enforcer_2wm._pressure_level = "soft"

        with patch("omlx.process_memory_enforcer.mx") as mock_mx, \
             patch("omlx.process_memory_enforcer.get_phys_footprint") as gpf:
            mock_mx.get_active_memory.return_value = 30 * 1024**3
            gpf.return_value = 40 * 1024**3
            await enforcer_2wm._check_and_enforce()

        assert enforcer_2wm._pressure_level == "ok"
        assert scheduler._admission_paused is False

    @pytest.mark.asyncio
    async def test_hard_aborts_in_flight_when_all_pinned(self, enforcer_2wm, pool):
        engine = MagicMock()
        engine.abort_all_requests = AsyncMock(return_value=3)
        entry = _make_entry("pinned", engine=engine, is_pinned=True)
        pool._entries = {"pinned": entry}
        pool._find_lru_victim.return_value = "pinned"  # single non-pinned would route through abort_all; here all pinned route through loading abort. We test the single-non-pinned hard branch separately below.

        # Single pinned model means find_lru_victim returns None (pinned not victim).
        pool._find_lru_victim.return_value = None

        with patch("omlx.process_memory_enforcer.mx") as mock_mx, \
             patch("omlx.process_memory_enforcer.get_phys_footprint") as gpf:
            mock_mx.get_active_memory.return_value = 60 * 1024**3
            gpf.return_value = 99 * 1024**3
            await enforcer_2wm._check_and_enforce()

        # No in-progress loads to abort, all pinned → enforcer just logs warning,
        # doesn't crash.
        assert enforcer_2wm._pressure_level == "hard"

    @pytest.mark.asyncio
    async def test_soft_does_not_abort_loading(self, enforcer_2wm, pool):
        loading_entry = _make_entry("loading", engine=None, is_loading=True)
        pool._entries = {"loading": loading_entry}
        pool._find_lru_victim.return_value = None

        with patch("omlx.process_memory_enforcer.mx") as mock_mx, \
             patch("omlx.process_memory_enforcer.get_phys_footprint") as gpf:
            mock_mx.get_active_memory.return_value = 50 * 1024**3
            gpf.return_value = 88 * 1024**3  # soft
            await enforcer_2wm._check_and_enforce()

        assert loading_entry.abort_loading is False  # soft must not abort load

    @pytest.mark.asyncio
    async def test_hard_aborts_loading(self, enforcer_2wm, pool):
        loading_entry = _make_entry("loading", engine=None, is_loading=True)
        pool._entries = {"loading": loading_entry}
        pool._find_lru_victim.return_value = None

        with patch("omlx.process_memory_enforcer.mx") as mock_mx, \
             patch("omlx.process_memory_enforcer.get_phys_footprint") as gpf:
            mock_mx.get_active_memory.return_value = 60 * 1024**3
            gpf.return_value = 99 * 1024**3  # hard
            await enforcer_2wm._check_and_enforce()

        assert loading_entry.abort_loading is True
