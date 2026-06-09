# SPDX-License-Identifier: Apache-2.0
"""Tests for ProcessMemoryEnforcer."""

import asyncio
from contextlib import suppress
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import omlx.process_memory_enforcer as pme
from omlx.process_memory_enforcer import ProcessMemoryEnforcer


def _make_enforcer(
    engine_pool,
    ceiling: int = 10 * 1024**3,
    tier: str = "balanced",
    poll_interval: float = 0.1,
    soft_threshold: float = 1.0,
    hard_threshold: float = 1.0,
    **kwargs,
) -> ProcessMemoryEnforcer:
    """Build an enforcer with a deterministic hard ceiling.

    The new enforcer derives its ceiling from system_memory + tier +
    psutil.virtual_memory().available, which is impractical to mock per
    test. We replace `_get_hard_limit_bytes` with a constant so tests can
    exercise the watermark logic without juggling system mocks.

    Passing `ceiling=0` disables the limit.
    """
    enforcer = ProcessMemoryEnforcer(
        engine_pool=engine_pool,
        memory_guard_tier=tier,
        poll_interval=poll_interval,
        soft_threshold=soft_threshold,
        hard_threshold=hard_threshold,
        **kwargs,
    )
    enforcer._get_hard_limit_bytes = lambda: int(ceiling)
    return enforcer


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


def _close_coro(coro):
    """side_effect for patched asyncio.create_task that closes the coroutine.

    Tests that mock create_task to skip the background loop still pass the
    real `_enforcement_loop()` coroutine in. Closing it prevents the
    "coroutine was never awaited" RuntimeWarning at gc time.
    """
    if hasattr(coro, "close"):
        coro.close()
    return MagicMock()


class TestMacOSVMStats:
    """Tests for the host_statistics64 telemetry adapter."""

    def test_uses_max_sized_host_info64_buffer(self):
        """Newer macOS kernels can require a larger vm_statistics64 tail."""

        class FakeLibc:
            def host_statistics64(self, host, flavor, stats, count):
                assert host == 123
                assert flavor == pme._HOST_VM_INFO64
                assert count._obj.value == pme._HOST_INFO64_MAX_COUNT
                stats[0] = 10
                stats[1] = 20
                stats[2] = 30
                stats[3] = 40
                count._obj.value = 104
                return 0

        with patch.object(pme, "_libc", FakeLibc()), patch.object(
            pme, "_MACH_HOST", 123
        ), patch.object(pme, "_VM_PAGE_SIZE", 4096):
            stats = pme.get_macos_vm_stats()

        assert stats == {
            "free": 10 * 4096,
            "active": 20 * 4096,
            "inactive": 30 * 4096,
            "wired": 40 * 4096,
        }

    def test_short_host_info64_response_returns_none(self):
        class FakeLibc:
            def host_statistics64(self, host, flavor, stats, count):
                count._obj.value = 3
                return 0

        with patch.object(pme, "_libc", FakeLibc()), patch.object(
            pme, "_MACH_HOST", 123
        ):
            assert pme.get_macos_vm_stats() is None


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
    """Create an enforcer with a fixed 10GB ceiling.

    Soft/hard thresholds default to 1.0 so legacy single-threshold tests
    keep treating the ceiling as the single trip point. Dedicated
    2-watermark tests construct their own enforcer with default thresholds.
    """
    return _make_enforcer(mock_engine_pool, ceiling=10 * 1024**3)


class TestAdaptivePolling:
    """Tests for adaptive ProcessMemoryEnforcer polling cadence."""

    def test_no_loaded_or_loading_models_uses_unloaded_idle_interval(self, enforcer):
        enforcer._engine_pool._entries = {
            f"model-{i}": _make_entry(f"model-{i}") for i in range(3)
        }

        assert enforcer._select_poll_interval() == 30.0

    def test_loading_model_uses_active_interval(self, enforcer):
        enforcer._engine_pool._entries = {
            "loading": _make_entry("loading", is_loading=True)
        }

        assert enforcer._select_poll_interval() == enforcer._active_poll_interval

    def test_loaded_idle_model_uses_loaded_idle_interval(self, enforcer):
        engine = MagicMock()
        engine.has_active_requests.return_value = False
        enforcer._engine_pool._entries = {
            "loaded": _make_entry("loaded", engine=engine)
        }

        assert enforcer._select_poll_interval() == 10.0

    def test_loaded_active_model_uses_active_interval(self, enforcer):
        engine = MagicMock()
        engine.has_active_requests.return_value = True
        enforcer._engine_pool._entries = {
            "active": _make_entry("active", engine=engine)
        }

        assert enforcer._select_poll_interval() == enforcer._active_poll_interval

    def test_pressure_uses_active_interval(self, enforcer):
        enforcer._pressure_level = "soft"

        assert enforcer._select_poll_interval() == enforcer._active_poll_interval

    def test_activity_hint_uses_active_interval(self, enforcer):
        enforcer._engine_pool._entries = {}
        enforcer.wake(active=True)

        assert enforcer._select_poll_interval() == enforcer._active_poll_interval

    @pytest.mark.asyncio
    async def test_wake_interrupts_idle_sleep(self, enforcer):
        enforcer._engine_pool._entries = {}
        enforcer._running = True
        enforcer._check_and_enforce = AsyncMock()
        enforcer._check_ttl = AsyncMock()

        task = asyncio.create_task(enforcer._enforcement_loop())
        try:
            for _ in range(20):
                if enforcer._check_and_enforce.await_count >= 1:
                    break
                await asyncio.sleep(0.01)
            assert enforcer._check_and_enforce.await_count >= 1

            enforcer.wake()
            for _ in range(20):
                if enforcer._check_and_enforce.await_count >= 2:
                    break
                await asyncio.sleep(0.01)

            assert enforcer._check_and_enforce.await_count >= 2
        finally:
            enforcer._running = False
            enforcer.wake()
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


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


class TestDisabledWhenCeilingZero:
    """Tests for enforcement disabled when the ceiling is 0 (guard off)."""

    @pytest.mark.asyncio
    async def test_no_enforce_when_ceiling_zero(self, mock_engine_pool):
        """No enforcement when ceiling is 0 (guard disabled)."""
        enforcer = _make_enforcer(mock_engine_pool, ceiling=0)
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
    async def test_no_enforce_when_guard_off(self, mock_engine_pool):
        """No enforcement when memory guard toggle is off."""
        enforcer = _make_enforcer(mock_engine_pool, ceiling=0)
        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            mock_mx.get_active_memory.return_value = 50 * 1024**3
            await enforcer._check_and_enforce()

        mock_engine_pool._unload_engine.assert_not_called()

    @pytest.mark.asyncio
    async def test_propagate_zero_disables_inline_prefill_check(
        self, mock_engine_pool
    ):
        """Propagating ceiling=0 sets scheduler limit to 0 (disabled)."""
        enforcer = _make_enforcer(mock_engine_pool, ceiling=0)
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


class TestStaticCeiling:
    """Tier-driven static ceiling (`total_ram - tier.static_reserve`).

    >= 24 GB systems use a tier-scaled reserve. < 24 GB systems always
    use a 4 GB reserve regardless of tier.
    """

    @pytest.mark.parametrize(
        "tier,expected_reserve_gb",
        [("safe", 8), ("balanced", 6), ("aggressive", 4)],
    )
    def test_large_system_tier_reserve(
        self, mock_engine_pool, tier, expected_reserve_gb
    ):
        enforcer = ProcessMemoryEnforcer(
            engine_pool=mock_engine_pool, memory_guard_tier=tier
        )
        with patch("omlx.settings.get_system_memory") as mock_mem:
            mock_mem.return_value = 96 * 1024**3
            result = enforcer._get_static_ceiling()
        assert result == (96 - expected_reserve_gb) * 1024**3

    @pytest.mark.parametrize("tier", ["safe", "balanced", "aggressive"])
    def test_small_system_uses_4gb_reserve_regardless_of_tier(
        self, mock_engine_pool, tier
    ):
        enforcer = ProcessMemoryEnforcer(
            engine_pool=mock_engine_pool, memory_guard_tier=tier
        )
        with patch("omlx.settings.get_system_memory") as mock_mem:
            mock_mem.return_value = 12 * 1024**3
            result = enforcer._get_static_ceiling()
        assert result == 8 * 1024**3

    @pytest.mark.parametrize("tier", ["safe", "balanced", "aggressive"])
    def test_16gb_system_uses_4gb_reserve_regardless_of_tier(
        self, mock_engine_pool, tier
    ):
        enforcer = ProcessMemoryEnforcer(
            engine_pool=mock_engine_pool, memory_guard_tier=tier
        )
        with patch("omlx.settings.get_system_memory") as mock_mem:
            mock_mem.return_value = 16 * 1024**3
            result = enforcer._get_static_ceiling()
        assert result == 12 * 1024**3

    @pytest.mark.parametrize("tier", ["safe", "balanced", "aggressive"])
    def test_between_16gb_and_24gb_system_uses_4gb_reserve_regardless_of_tier(
        self, mock_engine_pool, tier
    ):
        enforcer = ProcessMemoryEnforcer(
            engine_pool=mock_engine_pool, memory_guard_tier=tier
        )
        with patch("omlx.settings.get_system_memory") as mock_mem:
            mock_mem.return_value = 16 * 1024**3 + 256 * 1024**2
            result = enforcer._get_static_ceiling()
        assert result == 12 * 1024**3 + 256 * 1024**2

    def test_18gb_system_uses_4gb_reserve(self, mock_engine_pool):
        enforcer = ProcessMemoryEnforcer(
            engine_pool=mock_engine_pool, memory_guard_tier="balanced"
        )
        with patch("omlx.settings.get_system_memory") as mock_mem:
            mock_mem.return_value = 18 * 1024**3
            result = enforcer._get_static_ceiling()
        assert result == 14 * 1024**3

    def test_24gb_system_uses_tier_reserve(self, mock_engine_pool):
        enforcer = ProcessMemoryEnforcer(
            engine_pool=mock_engine_pool, memory_guard_tier="balanced"
        )
        with patch("omlx.settings.get_system_memory") as mock_mem:
            mock_mem.return_value = 24 * 1024**3
            result = enforcer._get_static_ceiling()
        assert result == 18 * 1024**3

    def test_custom_uses_2gb_reserve_on_large_system(self, mock_engine_pool):
        enforcer = ProcessMemoryEnforcer(
            engine_pool=mock_engine_pool,
            memory_guard_tier="custom",
            memory_guard_custom_ceiling_gb=50.0,
        )
        with patch("omlx.settings.get_system_memory") as mock_mem:
            mock_mem.return_value = 64 * 1024**3
            result = enforcer._get_static_ceiling()
        assert result == 62 * 1024**3

    def test_custom_uses_2gb_reserve_on_small_system(self, mock_engine_pool):
        """Custom bypasses the 4 GB small-system reserve."""
        enforcer = ProcessMemoryEnforcer(
            engine_pool=mock_engine_pool,
            memory_guard_tier="custom",
            memory_guard_custom_ceiling_gb=8.0,
        )
        with patch("omlx.settings.get_system_memory") as mock_mem:
            mock_mem.return_value = 12 * 1024**3
            result = enforcer._get_static_ceiling()
        assert result == 10 * 1024**3


class TestDynamicCeilingActiveRatio:
    """Dynamic ceiling sums free + inactive + active * tier ratio
    (host_statistics64 path) for safe / balanced / aggressive."""

    @pytest.mark.parametrize(
        "tier,ratio",
        [("safe", 0.2), ("balanced", 0.5), ("aggressive", 0.8)],
    )
    def test_active_ratio_per_tier(self, mock_engine_pool, tier, ratio):
        enforcer = ProcessMemoryEnforcer(
            engine_pool=mock_engine_pool, memory_guard_tier=tier
        )
        with patch(
            "omlx.process_memory_enforcer.get_phys_footprint",
            return_value=1 * 1024**3,
        ), patch(
            "omlx.process_memory_enforcer.get_macos_vm_stats",
            return_value={
                "free": 10 * 1024**3,
                "inactive": 4 * 1024**3,
                "active": 8 * 1024**3,
                "wired": 2 * 1024**3,
            },
        ):
            result = enforcer._get_dynamic_ceiling()
        expected = (
            1 * 1024**3
            + 10 * 1024**3
            + 4 * 1024**3
            + int(8 * 1024**3 * ratio)
        )
        assert result == expected

    def test_non_macos_falls_back_to_psutil_available(self, mock_engine_pool):
        enforcer = ProcessMemoryEnforcer(
            engine_pool=mock_engine_pool, memory_guard_tier="balanced"
        )
        with patch(
            "omlx.process_memory_enforcer.get_phys_footprint",
            return_value=2 * 1024**3,
        ), patch(
            "omlx.process_memory_enforcer.get_macos_vm_stats",
            return_value=None,
        ), patch(
            "omlx.process_memory_enforcer.psutil"
        ) as mock_psutil:
            mock_psutil.virtual_memory.return_value.available = 15 * 1024**3
            result = enforcer._get_dynamic_ceiling()
        assert result == 2 * 1024**3 + 15 * 1024**3

    def test_psutil_failure_falls_back_to_static_ceiling(self, mock_engine_pool):
        enforcer = ProcessMemoryEnforcer(
            engine_pool=mock_engine_pool, memory_guard_tier="balanced"
        )
        with patch(
            "omlx.process_memory_enforcer.get_phys_footprint",
            return_value=2 * 1024**3,
        ), patch(
            "omlx.process_memory_enforcer.get_macos_vm_stats",
            return_value=None,
        ), patch(
            "omlx.process_memory_enforcer.psutil.virtual_memory",
            side_effect=RuntimeError(
                "host_statistics64(HOST_VM_INFO64) syscall failed"
            ),
        ), patch("omlx.settings.get_system_memory", return_value=64 * 1024**3):
            result = enforcer._get_dynamic_ceiling()

        assert result == 58 * 1024**3


class TestDynamicCeilingCustom:
    """tier == custom: dynamic is the user-specified value, panic-safe via min()."""

    def test_custom_ceiling_returned_verbatim(self, mock_engine_pool):
        enforcer = ProcessMemoryEnforcer(
            engine_pool=mock_engine_pool,
            memory_guard_tier="custom",
            memory_guard_custom_ceiling_gb=24.0,
        )
        assert enforcer._get_dynamic_ceiling() == 24 * 1024**3

    def test_custom_setter_updates_ceiling(self, mock_engine_pool):
        enforcer = ProcessMemoryEnforcer(
            engine_pool=mock_engine_pool,
            memory_guard_tier="custom",
            memory_guard_custom_ceiling_gb=10.0,
        )
        enforcer.memory_guard_custom_ceiling_bytes = 30 * 1024**3
        assert enforcer._get_dynamic_ceiling() == 30 * 1024**3

    def test_custom_clamped_by_static_and_metal_cap(self, mock_engine_pool):
        """User can type any number; the final ceiling is still min(static,
        dynamic, metal_cap) so out-of-range values are panic-safe."""
        enforcer = ProcessMemoryEnforcer(
            engine_pool=mock_engine_pool,
            memory_guard_tier="custom",
            memory_guard_custom_ceiling_gb=1024.0,  # absurdly large
        )
        with patch(
            "omlx.settings.get_system_memory", return_value=64 * 1024**3
        ), patch(
            "omlx.process_memory_enforcer.get_effective_metal_cap_bytes",
            return_value=48 * 1024**3,
        ):
            ceiling = enforcer._get_hard_limit_bytes()
        # static = 64 - 2 = 62 GB; metal = 48 GB; custom = 1024 GB
        # → clamped to metal 48 GB
        assert ceiling == 48 * 1024**3


class TestHardLimitCalculation:
    """`_get_hard_limit_bytes` returns min(static, dynamic), or 0 when guard off."""

    def test_picks_static_when_smaller(self, mock_engine_pool):
        enforcer = ProcessMemoryEnforcer(
            engine_pool=mock_engine_pool, memory_guard_tier="balanced"
        )
        with patch("omlx.settings.get_system_memory") as mock_mem, patch(
            "omlx.process_memory_enforcer.get_phys_footprint",
            return_value=2 * 1024**3,
        ), patch(
            "omlx.process_memory_enforcer.get_macos_vm_stats",
            return_value={
                "free": 30 * 1024**3,
                "inactive": 10 * 1024**3,
                "active": 5 * 1024**3,
                "wired": 1 * 1024**3,
            },
        ), patch(
            "omlx.process_memory_enforcer.get_effective_metal_cap_bytes",
            return_value=100 * 1024**3,
        ):
            mock_mem.return_value = 48 * 1024**3  # static = 42 GB
            # dynamic balanced = 2 + 30 + 10 + 5*0.5 = 44.5 GB
            # static (42) wins → final ceiling is 42 GB
            assert enforcer._get_hard_limit_bytes() == 42 * 1024**3

    def test_picks_dynamic_when_smaller(self, mock_engine_pool):
        enforcer = ProcessMemoryEnforcer(
            engine_pool=mock_engine_pool, memory_guard_tier="balanced"
        )
        with patch("omlx.settings.get_system_memory") as mock_mem, patch(
            "omlx.process_memory_enforcer.get_phys_footprint",
            return_value=1 * 1024**3,
        ), patch(
            "omlx.process_memory_enforcer.get_macos_vm_stats",
            return_value={
                "free": 5 * 1024**3,
                "inactive": 2 * 1024**3,
                "active": 4 * 1024**3,
                "wired": 1 * 1024**3,
            },
        ), patch(
            "omlx.process_memory_enforcer.get_effective_metal_cap_bytes",
            return_value=100 * 1024**3,
        ):
            mock_mem.return_value = 48 * 1024**3  # static = 42 GB
            # dynamic balanced = 1 + 5 + 2 + int(4 * 0.5) = 10 GB
            # → dynamic wins
            assert enforcer._get_hard_limit_bytes() == 10 * 1024**3


class TestAbortLimitCalculation:
    """`_get_abort_limit_bytes` = min(static, metal_cap); ignores the jittery
    dynamic ceiling so a transient dip can't kill an in-flight prefill."""

    def test_ignores_dynamic_ceiling(self, mock_engine_pool):
        """Even when the dynamic ceiling is tiny, the abort limit stays at the
        stable min(static, metal_cap)."""
        enforcer = ProcessMemoryEnforcer(
            engine_pool=mock_engine_pool, memory_guard_tier="balanced"
        )
        with patch("omlx.settings.get_system_memory") as mock_mem, patch(
            "omlx.process_memory_enforcer.get_phys_footprint",
            return_value=1 * 1024**3,
        ), patch(
            "omlx.process_memory_enforcer.get_macos_vm_stats",
            return_value={  # dynamic would compute to ~10 GB (depressed)
                "free": 5 * 1024**3,
                "inactive": 2 * 1024**3,
                "active": 4 * 1024**3,
                "wired": 1 * 1024**3,
            },
        ), patch(
            "omlx.process_memory_enforcer.get_effective_metal_cap_bytes",
            return_value=100 * 1024**3,
        ):
            mock_mem.return_value = 48 * 1024**3  # static = 42 GB
            # dynamic (~10 GB) is far below, but the abort limit ignores it.
            assert enforcer._get_abort_limit_bytes() == 42 * 1024**3
            # Sanity: the (jittery) hard limit DID drop to dynamic.
            assert enforcer._get_hard_limit_bytes() == 10 * 1024**3

    def test_picks_metal_cap_when_smaller_than_static(self, mock_engine_pool):
        enforcer = ProcessMemoryEnforcer(
            engine_pool=mock_engine_pool, memory_guard_tier="balanced"
        )
        with patch(
            "omlx.settings.get_system_memory", return_value=64 * 1024**3
        ), patch(
            "omlx.process_memory_enforcer.get_effective_metal_cap_bytes",
            return_value=43 * 1024**3,
        ):
            # static = 64 - 8 = 56 GB; metal = 43 GB → min = 43 GB
            assert enforcer._get_abort_limit_bytes() == 43 * 1024**3

    def test_falls_back_to_static_when_metal_cap_unknown(self, mock_engine_pool):
        enforcer = ProcessMemoryEnforcer(
            engine_pool=mock_engine_pool, memory_guard_tier="aggressive"
        )
        with patch(
            "omlx.settings.get_system_memory", return_value=48 * 1024**3
        ), patch(
            "omlx.process_memory_enforcer.get_effective_metal_cap_bytes",
            return_value=0,  # unknown
        ):
            # aggressive reserve = 4 GB → static = 44 GB
            assert enforcer._get_abort_limit_bytes() == 44 * 1024**3

    def test_zero_when_guard_disabled(self, mock_engine_pool):
        enforcer = ProcessMemoryEnforcer(
            engine_pool=mock_engine_pool, prefill_memory_guard=False
        )
        assert enforcer._get_abort_limit_bytes() == 0

    def test_zero_when_guard_disabled(self, mock_engine_pool):
        enforcer = ProcessMemoryEnforcer(
            engine_pool=mock_engine_pool,
            memory_guard_tier="balanced",
            prefill_memory_guard=False,
        )
        assert enforcer._get_hard_limit_bytes() == 0
        assert enforcer.get_final_ceiling() == 0

    def test_unknown_tier_falls_back_to_balanced(self, mock_engine_pool):
        enforcer = ProcessMemoryEnforcer(
            engine_pool=mock_engine_pool, memory_guard_tier="extreme"
        )
        assert enforcer.memory_guard_tier == "balanced"


class TestMetalWiredLimit:
    """enforcer.start() applies MLX wired limits only for explicit sysctl caps."""

    def test_start_calls_set_wired_limit_with_static_ceiling(
        self, mock_engine_pool
    ):
        enforcer = ProcessMemoryEnforcer(
            engine_pool=mock_engine_pool, memory_guard_tier="balanced"
        )
        with patch(
            "omlx.settings.get_system_memory", return_value=48 * 1024**3
        ), patch(
            "omlx.process_memory_enforcer.get_effective_metal_cap_bytes",
            return_value=64 * 1024**3,  # cap above static ceiling
        ), patch(
            "omlx.process_memory_enforcer.get_iogpu_wired_limit_bytes",
            return_value=64 * 1024**3,
        ), patch(
            "omlx.process_memory_enforcer.mx"
        ) as mock_mx, patch.object(
            asyncio, "create_task", side_effect=_close_coro
        ):
            mock_mx.set_wired_limit.return_value = 36 * 1024**3
            enforcer.start()
        # balanced @ 48 GB => static_ceiling = 42 GB
        mock_mx.set_wired_limit.assert_called_once_with(42 * 1024**3)
        # Stored value is the desired ceiling (not the post-clamp value)
        # so the admin UI can detect a kernel cap that's below the
        # request and surface the sysctl-raise hint.
        assert enforcer._metal_wired_limit_request == 42 * 1024**3

    def test_start_clamps_to_effective_cap_when_lower(self, mock_engine_pool):
        enforcer = ProcessMemoryEnforcer(
            engine_pool=mock_engine_pool, memory_guard_tier="aggressive"
        )
        with patch(
            "omlx.settings.get_system_memory", return_value=64 * 1024**3
        ), patch(
            "omlx.process_memory_enforcer.get_effective_metal_cap_bytes",
            return_value=42 * 1024**3,  # cap < static ceiling
        ), patch(
            "omlx.process_memory_enforcer.get_iogpu_wired_limit_bytes",
            return_value=42 * 1024**3,
        ), patch(
            "omlx.process_memory_enforcer.mx"
        ) as mock_mx, patch.object(
            asyncio, "create_task", side_effect=_close_coro
        ):
            mock_mx.set_wired_limit.return_value = 48 * 1024**3
            enforcer.start()
        # aggressive @ 64 GB static = 60 GB, clamped to cap 42 GB
        mock_mx.set_wired_limit.assert_called_once_with(42 * 1024**3)
        # Desired (60 GB) is stored, not the post-clamp 42 GB.
        assert enforcer._metal_wired_limit_request == 60 * 1024**3

    def test_start_skips_set_wired_limit_when_sysctl_unset(
        self, mock_engine_pool, caplog
    ):
        """sysctl=0 path: plan against Apple cap, but don't touch MLX state."""
        enforcer = ProcessMemoryEnforcer(
            engine_pool=mock_engine_pool, memory_guard_tier="balanced"
        )
        with caplog.at_level("WARNING", logger="omlx.process_memory_enforcer"):
            with patch(
                "omlx.settings.get_system_memory", return_value=512 * 1024**3
            ), patch(
                "omlx.process_memory_enforcer.get_effective_metal_cap_bytes",
                return_value=128 * 1024**3,  # Apple default below static ceiling
            ), patch(
                "omlx.process_memory_enforcer.get_iogpu_wired_limit_bytes",
                return_value=0,  # sysctl unset; cap comes from working set
            ), patch(
                "omlx.process_memory_enforcer.mx"
            ) as mock_mx, patch.object(
                asyncio, "create_task", side_effect=_close_coro
            ):
                enforcer.start()
        # balanced @ 512 GB static = 506 GB. The scheduler still clamps to the
        # 128 GB effective cap, but MLX's wired limit is left untouched.
        mock_mx.set_wired_limit.assert_not_called()
        assert enforcer._metal_wired_limit_request == 506 * 1024**3
        assert "leaving Apple's default Metal cap active" in caplog.text

    def test_start_handles_set_wired_limit_error(self, mock_engine_pool):
        """Older macOS (<15) raises on the call; enforcer keeps going."""
        enforcer = ProcessMemoryEnforcer(
            engine_pool=mock_engine_pool, memory_guard_tier="balanced"
        )
        with patch(
            "omlx.settings.get_system_memory", return_value=48 * 1024**3
        ), patch(
            "omlx.process_memory_enforcer.get_effective_metal_cap_bytes",
            return_value=64 * 1024**3,
        ), patch(
            "omlx.process_memory_enforcer.get_iogpu_wired_limit_bytes",
            return_value=64 * 1024**3,
        ), patch(
            "omlx.process_memory_enforcer.mx"
        ) as mock_mx, patch.object(
            asyncio, "create_task", side_effect=_close_coro
        ):
            mock_mx.set_wired_limit.side_effect = RuntimeError("unsupported")
            enforcer.start()  # must not raise
        # We store the desired static_ceiling even when the call fails,
        # so the admin UI can still surface a warning.
        assert enforcer._metal_wired_limit_request == 42 * 1024**3

    def test_start_skips_when_guard_disabled(self, mock_engine_pool):
        """Guard off means we should not touch Metal limits either."""
        enforcer = ProcessMemoryEnforcer(
            engine_pool=mock_engine_pool,
            memory_guard_tier="balanced",
            prefill_memory_guard=False,
        )
        with patch(
            "omlx.process_memory_enforcer.mx"
        ) as mock_mx, patch.object(
            asyncio, "create_task", side_effect=_close_coro
        ):
            enforcer.start()
        mock_mx.set_wired_limit.assert_not_called()


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
        scheduler.batch_generator = bg
        engine = MagicMock(spec=[])
        engine.scheduler = scheduler
        entry = _make_entry("model-a", engine=engine)
        enforcer._engine_pool._entries = {"model-a": entry}

        enforcer._propagate_memory_limit()

        # Fixture stubs the ceiling at 10 GB with soft_threshold = 1.0,
        # so the scheduler soft limit equals the ceiling.
        assert scheduler._memory_limit_bytes == 10 * 1024**3
        assert bg._memory_limit_bytes == 10 * 1024**3
        assert scheduler._memory_hard_limit_bytes == 10 * 1024**3
        assert bg._memory_hard_limit_bytes == 10 * 1024**3

    def test_propagate_with_guard_disabled(self, enforcer):
        """When the guard is disabled the field reflects it; hard limit is
        still propagated for observability — the reader's early-return on
        ``prefill_memory_guard=False`` makes the value moot for the
        rejection path."""
        scheduler = MagicMock(spec=[])
        engine = MagicMock(spec=[])
        engine.scheduler = scheduler
        entry = _make_entry("model-a", engine=engine)
        enforcer._engine_pool._entries = {"model-a": entry}
        enforcer._prefill_memory_guard = False

        enforcer._propagate_memory_limit()

        assert scheduler._prefill_memory_guard is False
        # Hard limit is still propagated for observability — the reader's
        # early-return on ``prefill_memory_guard=False`` makes the value
        # moot for the rejection path. (The fixture's monkey-patched
        # ceiling stays at 10 GB; the production ``_get_hard_limit_bytes``
        # would return 0 when the guard is disabled, but the fixture
        # bypasses that branch — see ``_make_enforcer``.)
        assert scheduler._memory_hard_limit_bytes == 10 * 1024**3

    def test_propagates_on_tier_change(self, enforcer):
        """Changing the tier at runtime triggers re-propagation."""
        bg = MagicMock(spec=[])
        bg._memory_limit_bytes = 0
        bg._memory_hard_limit_bytes = 0
        scheduler = MagicMock(spec=[])
        scheduler.batch_generator = bg
        engine = MagicMock(spec=[])
        engine.scheduler = scheduler
        entry = _make_entry("model-a", engine=engine)
        enforcer._engine_pool._entries = {"model-a": entry}

        enforcer._running = True
        # Simulate the ceiling shrinking after the tier flip.
        enforcer._get_hard_limit_bytes = lambda: 20 * 1024**3
        enforcer.memory_guard_tier = "safe"

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

    async def test_check_and_enforce_propagates_every_poll(self, enforcer):
        """Regression: a fresh engine loaded AFTER enforcer.start() must pick
        up its limits within one poll interval — even when pressure stays
        "ok" the whole time.

        Before this guarantee the propagation only fired on pressure-level
        changes. On a host where the first prefill stayed below soft until
        a few seconds in, the scheduler kept _prefill_memory_guard=False /
        _memory_hard_limit_bytes=0 (their __init__ defaults), the guard
        short-circuited, the request entered prefill, and the underlying
        Apple IOGPUFamily bug (FB22091885) panicked the kernel mid-chunk.
        """
        # Engine pool starts empty (mirrors real startup: lazy load on first
        # request, well after enforcer.start()).
        enforcer._engine_pool._entries = {}
        # Engine loads at t1 — the enforcer hasn't seen it yet.
        bg = MagicMock(spec=[])
        bg._memory_limit_bytes = 0
        bg._memory_hard_limit_bytes = 0
        scheduler = MagicMock(spec=[])
        scheduler.batch_generator = bg
        engine = MagicMock(spec=[])
        engine.scheduler = scheduler
        entry = _make_entry("model-a", engine=engine)
        enforcer._engine_pool._entries = {"model-a": entry}

        # One poll iteration with pressure well below soft — pressure level
        # does NOT change. Before the fix this returned without propagating.
        with patch.object(
            enforcer, "_current_usage_bytes", return_value=1 * 1024**3
        ):
            await enforcer._check_and_enforce()

        # Within one poll, the freshly-loaded engine has the user-configured
        # ceiling and the guard flag.
        assert scheduler._memory_hard_limit_bytes == 10 * 1024**3
        assert scheduler._memory_limit_bytes == 10 * 1024**3
        assert scheduler._prefill_memory_guard is True

    def test_propagates_through_batched_engine_wrapper(self, enforcer):
        """Regression: live engines in EnginePool don't expose ``.scheduler``
        on the top-level wrapper — BatchedEngine and VLMBatchedEngine both
        hold the real Scheduler at ``self._engine.engine.scheduler``. The
        propagation must traverse that chain, otherwise the prefill memory
        guard flag never reaches the scheduler and the guard short-circuits
        on every request (observed end-to-end 2026-05-15: three kernel
        panics from 110k-token Qwen3.6-VL prefills the guard "should" have
        rejected).
        """
        # Build the real wrapper shape:
        #   entry.engine                  → BatchedEngine / VLMBatchedEngine
        #   entry.engine._engine          → AsyncEngineCore
        #   entry.engine._engine.engine   → EngineCore
        #   entry.engine._engine.engine.scheduler → Scheduler  ← target
        scheduler = MagicMock(spec=[])
        scheduler.batch_generator = None
        engine_core = MagicMock(spec=["scheduler"])
        engine_core.scheduler = scheduler
        async_engine_core = MagicMock(spec=["engine"])
        async_engine_core.engine = engine_core
        # Wrapper deliberately does NOT expose top-level ``.scheduler`` — only
        # ``._engine`` like the real BatchedEngine.
        wrapper = MagicMock(spec=["_engine"])
        wrapper._engine = async_engine_core

        entry = _make_entry("model-a", engine=wrapper)
        enforcer._engine_pool._entries = {"model-a": entry}

        enforcer._propagate_memory_limit()

        assert scheduler._memory_limit_bytes == 10 * 1024**3
        assert scheduler._memory_hard_limit_bytes == 10 * 1024**3
        assert scheduler._prefill_memory_guard is True

    def test_unresolvable_scheduler_logs_warning_once(self, enforcer, caplog):
        """If the wrapper-chain traversal fails (no ``scheduler`` anywhere
        in the chain), ``_propagate_memory_limit`` must log a WARNING
        naming the engine type so the silent no-op failure mode that
        originally hid the dead memory guard is loud in CI / oncall. The
        warning is rate-limited per engine type so a misconfigured
        engine polled every second doesn't spam.
        """
        # Wrapper chain that bottoms out without a scheduler.
        wrapper = MagicMock(spec=["_engine"])
        wrapper._engine = MagicMock(spec=["engine"])
        wrapper._engine.engine = MagicMock(spec=[])  # no .scheduler
        wrapper.__class__.__name__ = "BrokenEngine"

        entry = _make_entry("model-broken", engine=wrapper)
        enforcer._engine_pool._entries = {"model-broken": entry}

        with caplog.at_level("WARNING", logger="omlx.process_memory_enforcer"):
            enforcer._propagate_memory_limit()
            # Second call: no extra log line — rate limit holds.
            enforcer._propagate_memory_limit()

        matching = [
            r for r in caplog.records
            if "could not resolve scheduler" in r.message
        ]
        assert len(matching) == 1, (
            f"expected 1 warning, got {[r.message for r in matching]}"
        )


class TestUnresolvableSchedulerWarning:
    """``_propagate_memory_limit`` must complain when a wrapper-chain
    change makes the scheduler unreachable via ``_resolve_scheduler``.

    Silent no-op was the failure mode that originally hid the
    dead-memory-guard bug for months — surfacing it as a per-engine-type
    rate-limited warning closes that gap without spamming logs at
    request rate.
    """

    def test_warns_once_per_engine_type(self, enforcer, caplog):
        """Three propagation calls with an unreachable scheduler emit
        exactly one WARNING (not three)."""
        # Engine wrapper with no resolvable scheduler chain: spec=[]
        # blocks both ``.scheduler`` and ``._engine`` lookups so
        # ``_resolve_scheduler`` returns None.
        engine = MagicMock(spec=[])
        entry = _make_entry("model-broken", engine=engine)
        enforcer._engine_pool._entries = {"model-broken": entry}

        with caplog.at_level(
            "WARNING", logger="omlx.process_memory_enforcer"
        ):
            enforcer._propagate_memory_limit()
            enforcer._propagate_memory_limit()
            enforcer._propagate_memory_limit()

        warnings = [
            r for r in caplog.records
            if "could not resolve scheduler" in r.getMessage()
        ]
        assert len(warnings) == 1, (
            f"Expected exactly one warning per engine type per lifetime, "
            f"got {len(warnings)}"
        )

    def test_unresolvable_does_not_block_other_engines(self, enforcer):
        """If engine A is unresolvable but engine B has a real scheduler,
        B must still receive the propagation."""
        # A: unresolvable (no .scheduler / no ._engine).
        engine_a = MagicMock(spec=[])
        entry_a = _make_entry("model-a", engine=engine_a)

        # B: real scheduler chain.
        scheduler_b = MagicMock(spec=[])
        scheduler_b._memory_limit_bytes = 0
        scheduler_b._memory_hard_limit_bytes = 0
        scheduler_b._prefill_memory_guard = False
        scheduler_b._admission_paused = False
        scheduler_b._prefill_safe_zone_ratio = 0.0
        scheduler_b._prefill_min_chunk_tokens = 0
        scheduler_b.batch_generator = None
        engine_b = MagicMock(spec=[])
        engine_b.scheduler = scheduler_b
        entry_b = _make_entry("model-b", engine=engine_b)

        enforcer._engine_pool._entries = {
            "model-a": entry_a,
            "model-b": entry_b,
        }

        enforcer._propagate_memory_limit()

        # B got it; A's resolve-failure didn't poison the loop.
        assert scheduler_b._memory_limit_bytes == 10 * 1024**3
        assert scheduler_b._prefill_memory_guard is True

    def test_no_warning_for_unloaded_engine(self, enforcer, caplog):
        """A discovered-but-unloaded entry (``engine is None``) is a normal
        state, not a wrapper break, so it must not emit the warning. The
        pool keeps these entries for every model it has discovered but not
        yet loaded, so warning here would fire on a routine startup."""
        entry = _make_entry("model-unloaded", engine=None)
        enforcer._engine_pool._entries = {"model-unloaded": entry}

        with caplog.at_level(
            "WARNING", logger="omlx.process_memory_enforcer"
        ):
            enforcer._propagate_memory_limit()
            enforcer._propagate_memory_limit()

        warnings = [
            r for r in caplog.records
            if "could not resolve scheduler" in r.getMessage()
        ]
        assert warnings == [], (
            f"Unloaded engine must not warn, got {len(warnings)} warning(s)"
        )


class TestStoreCacheCapWalk:
    """Tests for _walk_store_cache_caps — store-cache gate adjustment (#1383)."""

    def _scheduler_with_adjust(self):
        scheduler = MagicMock(spec=[])
        scheduler.adjust_store_cache_cap = MagicMock()
        return scheduler

    def test_calls_adjust_with_current_pressure(self, enforcer):
        scheduler = self._scheduler_with_adjust()
        engine = MagicMock(spec=[])
        engine.scheduler = scheduler
        enforcer._engine_pool._entries = {"m": _make_entry("m", engine=engine)}
        enforcer._pressure_level = "soft"

        enforcer._walk_store_cache_caps()

        scheduler.adjust_store_cache_cap.assert_called_once_with("soft")

    def test_no_op_when_engine_missing(self, enforcer):
        entry = _make_entry("m", engine=None)
        enforcer._engine_pool._entries = {"m": entry}
        # Should not raise.
        enforcer._walk_store_cache_caps()

    def test_no_op_when_scheduler_lacks_method(self, enforcer):
        engine = MagicMock(spec=[])  # no scheduler attr
        entry = _make_entry("m", engine=engine)
        enforcer._engine_pool._entries = {"m": entry}
        # Should not raise.
        enforcer._walk_store_cache_caps()

    @pytest.mark.asyncio
    async def test_check_and_enforce_walks_caps_on_ok(self, enforcer):
        scheduler = self._scheduler_with_adjust()
        engine = MagicMock(spec=[])
        engine.scheduler = scheduler
        enforcer._engine_pool._entries = {"m": _make_entry("m", engine=engine)}

        with patch("omlx.process_memory_enforcer.mx") as mock_mx, patch(
            "omlx.process_memory_enforcer.get_phys_footprint", return_value=0
        ):
            mock_mx.get_active_memory.return_value = 1 * 1024**3  # ok
            await enforcer._check_and_enforce()

        scheduler.adjust_store_cache_cap.assert_called_with("ok")

    @pytest.mark.asyncio
    async def test_check_and_enforce_walks_caps_on_soft(self, enforcer):
        # Force a 0.85/0.95 split so 9GB lands in the soft band.
        enforcer._soft_threshold = 0.85
        enforcer._hard_threshold = 0.95
        scheduler = self._scheduler_with_adjust()
        engine = MagicMock(spec=[])
        engine.scheduler = scheduler
        enforcer._engine_pool._entries = {"m": _make_entry("m", engine=engine)}
        enforcer._engine_pool._find_lru_victim = MagicMock(return_value=None)

        with patch("omlx.process_memory_enforcer.mx") as mock_mx, patch(
            "omlx.process_memory_enforcer.get_phys_footprint", return_value=0
        ):
            mock_mx.get_active_memory.return_value = 9 * 1024**3  # soft
            await enforcer._check_and_enforce()

        scheduler.adjust_store_cache_cap.assert_called_with("soft")


class TestProperties:
    """Tests for enforcer properties."""

    def test_memory_guard_tier_default(self, enforcer):
        """Default tier from `_make_enforcer` is balanced."""
        assert enforcer.memory_guard_tier == "balanced"

    def test_memory_guard_tier_setter(self, enforcer):
        """Setting a new tier updates the internal state."""
        enforcer.memory_guard_tier = "safe"
        assert enforcer.memory_guard_tier == "safe"

    def test_memory_guard_tier_setter_ignores_unknown_value(self, enforcer):
        """Unknown tier values normalize to balanced."""
        enforcer.memory_guard_tier = "extreme"
        assert enforcer.memory_guard_tier == "balanced"

    def test_get_final_ceiling_matches_hard_limit(self, enforcer):
        assert enforcer.get_final_ceiling() == enforcer._get_hard_limit_bytes()

    def test_is_running_initially_false(self, enforcer):
        """Test is_running is False before start."""
        assert enforcer.is_running is False

    def test_get_status_when_not_running(self, enforcer):
        """Test get_status when enforcer is not running."""
        status = enforcer.get_status()
        assert status["enabled"] is False
        assert status["ceiling_bytes"] == 0
        assert status["current_bytes"] == 0
        assert status["memory_guard_tier"] == "balanced"


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
        return _make_enforcer(
            pool,
            ceiling=100 * 1024**3,
            soft_threshold=0.85,
            hard_threshold=0.95,
        )

    def test_soft_hard_bytes_computed(self, enforcer_2wm):
        assert enforcer_2wm._soft_bytes() == int(100 * 1024**3 * 0.85)
        assert enforcer_2wm._hard_bytes() == int(100 * 1024**3 * 0.95)

    def test_prefill_abort_margin_is_tier_specific(self, pool):
        balanced = _make_enforcer(pool, tier="balanced")
        aggressive = _make_enforcer(pool, tier="aggressive")
        custom = _make_enforcer(pool, tier="custom")

        assert balanced._get_prefill_abort_margin() == 0.90
        assert aggressive._get_prefill_abort_margin() == 0.95
        assert custom._get_prefill_abort_margin() == 0.95

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
    async def test_propagates_abort_limit_to_scheduler(self, enforcer_2wm, pool):
        """The stable abort ceiling is pushed to scheduler._memory_abort_limit_bytes
        every tick, independent of the (jittery) dynamic hard limit."""
        engine = MagicMock()
        scheduler = MagicMock()
        scheduler._memory_limit_bytes = 0
        scheduler._memory_hard_limit_bytes = 0
        scheduler._memory_abort_limit_bytes = 0
        scheduler._prefill_memory_guard = False
        scheduler._admission_paused = False
        engine.scheduler = scheduler
        pool._entries = {"m": _make_entry("m", engine=engine)}

        # Stub the abort ceiling to a known stable value.
        enforcer_2wm._get_abort_limit_bytes = lambda: 42 * 1024**3

        with patch("omlx.process_memory_enforcer.mx") as mock_mx, \
             patch("omlx.process_memory_enforcer.get_phys_footprint") as gpf:
            mock_mx.get_active_memory.return_value = 50 * 1024**3
            gpf.return_value = 50 * 1024**3
            await enforcer_2wm._check_and_enforce()

        assert scheduler._memory_abort_limit_bytes == 42 * 1024**3

    @pytest.mark.asyncio
    async def test_propagates_prefill_abort_margin_to_scheduler(self, pool):
        enforcer = _make_enforcer(pool, ceiling=100 * 1024**3, tier="custom")
        engine = MagicMock()
        scheduler = MagicMock()
        scheduler._memory_limit_bytes = 0
        scheduler._memory_hard_limit_bytes = 0
        scheduler._memory_abort_limit_bytes = 0
        scheduler._prefill_abort_margin = 0.90
        scheduler._prefill_memory_guard = False
        scheduler._admission_paused = False
        engine.scheduler = scheduler
        pool._entries = {"m": _make_entry("m", engine=engine)}

        with patch("omlx.process_memory_enforcer.mx") as mock_mx, \
             patch("omlx.process_memory_enforcer.get_phys_footprint") as gpf:
            mock_mx.get_active_memory.return_value = 50 * 1024**3
            gpf.return_value = 50 * 1024**3
            await enforcer._check_and_enforce()

        assert scheduler._prefill_abort_margin == 0.95

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

    def test_get_status_uses_max_active_and_phys(self, enforcer_2wm):
        """get_status must report the same value enforcer compares against,
        so admin UI / /health utilization matches the watermark logic."""
        enforcer_2wm._running = True
        with patch("omlx.process_memory_enforcer.mx") as mock_mx, \
             patch("omlx.process_memory_enforcer.get_phys_footprint") as gpf:
            mock_mx.get_active_memory.return_value = 50 * 1024**3
            gpf.return_value = 88 * 1024**3  # phys dominates
            status = enforcer_2wm.get_status()
        assert status["current_bytes"] == 88 * 1024**3
        # Utilization computed against the max value
        assert abs(status["utilization"] - 0.88) < 0.01
