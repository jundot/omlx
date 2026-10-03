# SPDX-License-Identifier: Apache-2.0
"""Tests for the idle-time, cross-consumer disk-pressure guard (design doc R2).

The guard is exercised entirely through fault-injected `disk_usage` and
mocked managers/stores — no test touches a real filesystem's actual free
space, and nothing here runs against the live `~/.omlx` directory.
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from omlx.disk_pressure_guard import (
    _MIN_SCALE,
    disk_pressure_guard_loop,
    run_disk_pressure_guard_tick,
)
from omlx.settings import DiskSettings


def _usage(total: int, free: int):
    return SimpleNamespace(total=total, used=total - free, free=free)


def _manager(*, evicted_bytes: int = 0, reconciled: int = 0) -> MagicMock:
    """A MagicMock manager with realistic (int) return values — a bare
    MagicMock() would make `reconciled += manager.reconcile_tracked_sizes()`
    add an int to a MagicMock and raise, silently swallowed by the tick's
    own per-manager try/except and therefore invisible without this."""
    manager = MagicMock()
    manager.enforce_size_limit.return_value = evicted_bytes
    manager.reconcile_tracked_sizes.return_value = reconciled
    return manager


def _settings(**overrides) -> DiskSettings:
    return DiskSettings(**overrides)


class TestTickTiers:
    async def test_ok_tier_leaves_managers_untouched_by_eviction(self):
        disk_settings = _settings(soft_free_floor_gb=20.0, hard_free_floor_gb=10.0)
        manager = _manager()
        store = MagicMock()

        def disk_usage(_path):
            return _usage(total=1000 * 1024**3, free=500 * 1024**3)

        result = await run_disk_pressure_guard_tick(
            volume_path=Path("/fake"),
            disk_settings=disk_settings,
            get_managers=lambda: [manager],
            get_boundary_stores=lambda: [store],
            disk_usage=disk_usage,
        )

        assert result.tier == "ok"
        manager.set_disk_pressure_hard.assert_called_once_with(False)
        store.set_disk_pressure_hard.assert_called_once_with(False)
        manager.enforce_size_limit.assert_not_called()

    async def test_soft_tier_scales_eviction_without_refusing_writes(self):
        disk_settings = _settings(
            soft_free_floor_gb=20.0,
            soft_free_floor_fraction=0.0,
            hard_free_floor_gb=10.0,
            hard_free_floor_fraction=0.0,
        )
        manager = _manager(evicted_bytes=123)
        store = MagicMock()

        # 15 GiB free: below the 20 GiB soft floor, above the 10 GiB hard floor.
        def disk_usage(_path):
            return _usage(total=1000 * 1024**3, free=15 * 1024**3)

        result = await run_disk_pressure_guard_tick(
            volume_path=Path("/fake"),
            disk_settings=disk_settings,
            get_managers=lambda: [manager],
            get_boundary_stores=lambda: [store],
            disk_usage=disk_usage,
        )

        assert result.tier == "soft"
        manager.set_disk_pressure_hard.assert_called_once_with(False)
        store.set_disk_pressure_hard.assert_called_once_with(False)
        manager.enforce_size_limit.assert_called_once()
        _, kwargs = manager.enforce_size_limit.call_args
        assert 0 < kwargs["trigger_fraction"] < 1.0
        assert 0 < kwargs["target_fraction"] < kwargs["trigger_fraction"]
        assert result.managers_evicted_bytes == 123

    async def test_hard_tier_refuses_writes_and_evicts_aggressively(self):
        disk_settings = _settings(
            soft_free_floor_gb=20.0,
            soft_free_floor_fraction=0.0,
            hard_free_floor_gb=10.0,
            hard_free_floor_fraction=0.0,
        )
        manager = _manager()
        store = MagicMock()

        # 5 GiB free: below the 10 GiB hard floor.
        def disk_usage(_path):
            return _usage(total=1000 * 1024**3, free=5 * 1024**3)

        result = await run_disk_pressure_guard_tick(
            volume_path=Path("/fake"),
            disk_settings=disk_settings,
            get_managers=lambda: [manager],
            get_boundary_stores=lambda: [store],
            disk_usage=disk_usage,
        )

        assert result.tier == "hard"
        assert result.scale == _MIN_SCALE
        manager.set_disk_pressure_hard.assert_called_once_with(True)
        store.set_disk_pressure_hard.assert_called_once_with(True)
        manager.enforce_size_limit.assert_called_once()

    async def test_pressure_clears_reactivates_writes(self):
        """A manager left in hard-refusal mode from a prior tick must be
        released once free disk recovers — never a one-way switch."""
        disk_settings = _settings(soft_free_floor_gb=20.0, hard_free_floor_gb=10.0)
        manager = _manager()

        def plentiful(_path):
            return _usage(total=1000 * 1024**3, free=500 * 1024**3)

        result = await run_disk_pressure_guard_tick(
            volume_path=Path("/fake"),
            disk_settings=disk_settings,
            get_managers=lambda: [manager],
            get_boundary_stores=lambda: [],
            disk_usage=plentiful,
        )

        assert result.tier == "ok"
        manager.set_disk_pressure_hard.assert_called_once_with(False)


class TestTickReconcile:
    """design doc §A5/2.5: multi-manager index drift decays on every tick,
    independent of pressure tier — not just under soft/hard pressure."""

    async def test_reconcile_runs_even_at_ok_tier(self):
        disk_settings = _settings(soft_free_floor_gb=20.0, hard_free_floor_gb=10.0)
        manager = _manager(reconciled=3)

        def plentiful(_path):
            return _usage(total=1000 * 1024**3, free=500 * 1024**3)

        result = await run_disk_pressure_guard_tick(
            volume_path=Path("/fake"),
            disk_settings=disk_settings,
            get_managers=lambda: [manager],
            get_boundary_stores=lambda: [],
            disk_usage=plentiful,
        )

        assert result.tier == "ok"
        manager.reconcile_tracked_sizes.assert_called_once()
        assert result.reconciled_entries == 3

    async def test_manager_without_reconcile_method_is_tolerated(self):
        """Older/mocked managers without the method must not break the tick."""
        disk_settings = _settings()
        manager = MagicMock(spec=["set_disk_pressure_hard", "enforce_size_limit"])
        manager.enforce_size_limit.return_value = 0

        def plentiful(_path):
            return _usage(total=1000 * 1024**3, free=500 * 1024**3)

        result = await run_disk_pressure_guard_tick(
            volume_path=Path("/fake"),
            disk_settings=disk_settings,
            get_managers=lambda: [manager],
            get_boundary_stores=lambda: [],
            disk_usage=plentiful,
        )

        assert result.tier == "ok"
        assert result.reconciled_entries == 0


class TestTickLogReaperWiring:
    """design doc §R4: 'trigger: server startup + the R2 tick'."""

    async def test_log_reaper_runs_every_tick(self):
        disk_settings = _settings()
        manager = _manager()
        calls = []

        def plentiful(_path):
            return _usage(total=1000 * 1024**3, free=500 * 1024**3)

        await run_disk_pressure_guard_tick(
            volume_path=Path("/fake"),
            disk_settings=disk_settings,
            get_managers=lambda: [manager],
            get_boundary_stores=lambda: [],
            disk_usage=plentiful,
            run_log_reaper=lambda: calls.append(1),
        )

        assert calls == [1]

    async def test_log_reaper_failure_does_not_break_the_tick(self):
        disk_settings = _settings()
        manager = _manager()

        def plentiful(_path):
            return _usage(total=1000 * 1024**3, free=500 * 1024**3)

        def broken_reaper():
            raise RuntimeError("boom")

        result = await run_disk_pressure_guard_tick(
            volume_path=Path("/fake"),
            disk_settings=disk_settings,
            get_managers=lambda: [manager],
            get_boundary_stores=lambda: [],
            disk_usage=plentiful,
            run_log_reaper=broken_reaper,
        )

        assert result.tier == "ok"

    async def test_log_reaper_still_runs_when_disk_usage_fails(self):
        """Log hygiene is independent of disk-pressure evaluation — a
        volume that can't be statted shouldn't also block log rotation."""
        disk_settings = _settings()
        calls = []

        def broken(_path):
            raise OSError("no such volume")

        result = await run_disk_pressure_guard_tick(
            volume_path=Path("/fake"),
            disk_settings=disk_settings,
            get_managers=lambda: [],
            get_boundary_stores=lambda: [],
            disk_usage=broken,
            run_log_reaper=lambda: calls.append(1),
        )

        assert result is None
        assert calls == [1]


class TestTickFaultInjection:
    async def test_disk_usage_oserror_returns_none_and_touches_nothing(self):
        disk_settings = _settings()
        manager = _manager()

        def broken(_path):
            raise OSError("no such volume")

        result = await run_disk_pressure_guard_tick(
            volume_path=Path("/fake"),
            disk_settings=disk_settings,
            get_managers=lambda: [manager],
            get_boundary_stores=lambda: [],
            disk_usage=broken,
        )

        assert result is None
        manager.set_disk_pressure_hard.assert_not_called()
        manager.enforce_size_limit.assert_not_called()

    async def test_one_manager_raising_does_not_block_the_others(self):
        disk_settings = _settings(soft_free_floor_gb=20.0, hard_free_floor_gb=10.0)
        broken_manager = _manager()
        broken_manager.set_disk_pressure_hard.side_effect = RuntimeError("boom")
        healthy_manager = _manager()

        def plentiful(_path):
            return _usage(total=1000 * 1024**3, free=500 * 1024**3)

        result = await run_disk_pressure_guard_tick(
            volume_path=Path("/fake"),
            disk_settings=disk_settings,
            get_managers=lambda: [broken_manager, healthy_manager],
            get_boundary_stores=lambda: [],
            disk_usage=plentiful,
        )

        assert result.tier == "ok"
        healthy_manager.set_disk_pressure_hard.assert_called_once_with(False)


class TestGuardLoop:
    async def test_loop_exits_promptly_on_stop_event(self):
        disk_settings = _settings(guard_tick_interval_seconds=0.01)
        stop_event = asyncio.Event()
        ticks = []

        def plentiful(_path):
            ticks.append(1)
            return _usage(total=1000 * 1024**3, free=500 * 1024**3)

        async def stopper():
            await asyncio.sleep(0.05)
            stop_event.set()

        await asyncio.wait_for(
            asyncio.gather(
                disk_pressure_guard_loop(
                    volume_path=Path("/fake"),
                    disk_settings=disk_settings,
                    get_managers=lambda: [],
                    get_boundary_stores=lambda: [],
                    stop_event=stop_event,
                ),
                stopper(),
            ),
            timeout=5.0,
        )

        assert stop_event.is_set()

    async def test_loop_survives_a_raising_tick(self, monkeypatch):
        import omlx.disk_pressure_guard as guard_mod

        disk_settings = _settings(guard_tick_interval_seconds=0.01)
        stop_event = asyncio.Event()
        call_count = {"n": 0}

        async def raising_tick(**_kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("tick blew up")
            stop_event.set()
            return None

        monkeypatch.setattr(guard_mod, "run_disk_pressure_guard_tick", raising_tick)

        await asyncio.wait_for(
            disk_pressure_guard_loop(
                volume_path=Path("/fake"),
                disk_settings=disk_settings,
                get_managers=lambda: [],
                get_boundary_stores=lambda: [],
                stop_event=stop_event,
            ),
            timeout=5.0,
        )

        assert call_count["n"] >= 2
