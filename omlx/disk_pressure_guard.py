# SPDX-License-Identifier: Apache-2.0
"""Idle-time, cross-consumer disk-pressure guard (design doc R2).

The only three triggers of SSD-cache eviction were a block save, a sidecar
commit, and the startup scan — a server that is loaded but idle held its
full cache budget while another process (a model download, Time Machine,
the user's own files) filled the remaining disk, reacting only at the next
write, with a 32-unlink burst landing on the inference hot path at the
worst possible time. This module is the periodic, out-of-band actor that
was missing: it reads free disk directly (soft/hard floors are expressed
in absolute free bytes, not cache-relative, so pressure attributed to
"other people's data" by the save-time clamp is still visible here) and
acts on every live manager/store without waiting for a write to happen.

Admission (serving a request) is never touched by this guard — only
optional cache writes are refused or throttled.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)

# Floor for the soft-pressure eviction scale (free / soft_floor). Never let
# a single tick collapse the cache to near-zero in one pass — the walk is
# already bounded per call site; this just keeps each pass's target sane.
_MIN_SCALE = 0.1


@dataclass
class DiskPressureTickResult:
    """What one guard tick observed and did — surfaced via get_stats_dict."""

    free_bytes: int
    total_bytes: int
    soft_floor_bytes: int
    hard_floor_bytes: int
    tier: str  # "ok" | "soft" | "hard"
    scale: float
    managers_evicted_bytes: int
    managers_count: int
    stores_count: int
    reconciled_entries: int


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _evaluate_tick(
    *,
    volume_path: Path,
    disk_settings: Any,
    managers: list[Any],
    stores: list[Any],
    disk_usage: Callable[[Path], Any],
) -> DiskPressureTickResult | None:
    """Synchronous body of one tick — run off the event loop by the caller."""
    try:
        usage = disk_usage(volume_path)
    except OSError as exc:
        logger.warning(
            "Disk pressure guard: failed to read disk usage for %s: %s",
            volume_path,
            exc,
        )
        return None

    total_bytes = usage.total
    free_bytes = usage.free
    soft_floor = disk_settings.get_soft_free_floor_bytes(total_bytes)
    hard_floor = disk_settings.get_hard_free_floor_bytes(total_bytes)

    if free_bytes < hard_floor:
        tier = "hard"
        scale = _MIN_SCALE
    elif free_bytes < soft_floor:
        tier = "soft"
        scale = (
            _clamp(free_bytes / soft_floor, _MIN_SCALE, 1.0)
            if soft_floor > 0
            else _MIN_SCALE
        )
    else:
        tier = "ok"
        scale = 1.0

    hard_active = tier == "hard"
    for manager in managers:
        try:
            manager.set_disk_pressure_hard(hard_active)
        except Exception:
            logger.debug(
                "Disk pressure guard: manager set_disk_pressure_hard failed",
                exc_info=True,
            )
    for store in stores:
        try:
            store.set_disk_pressure_hard(hard_active)
        except Exception:
            logger.debug(
                "Disk pressure guard: store set_disk_pressure_hard failed",
                exc_info=True,
            )

    evicted_bytes = 0
    if tier in ("soft", "hard"):
        for manager in managers:
            try:
                evicted_bytes += manager.enforce_size_limit(
                    trigger_fraction=scale,
                    target_fraction=max(_MIN_SCALE, scale * 0.9),
                )
            except Exception:
                logger.warning(
                    "Disk pressure guard: enforce_size_limit failed", exc_info=True
                )

    # Multi-manager index drift (design doc §A5) decays on every tick,
    # independent of pressure tier — a bounded, cheap stat-check batch.
    reconciled = 0
    for manager in managers:
        try:
            reconcile = getattr(manager, "reconcile_tracked_sizes", None)
            if reconcile is not None:
                reconciled += reconcile()
        except Exception:
            logger.warning(
                "Disk pressure guard: reconcile_tracked_sizes failed", exc_info=True
            )

    if tier != "ok" or evicted_bytes or reconciled:
        logger.info(
            "Disk pressure guard tick: tier=%s free=%d soft_floor=%d "
            "hard_floor=%d evicted=%d reconciled=%d managers=%d",
            tier,
            free_bytes,
            soft_floor,
            hard_floor,
            evicted_bytes,
            reconciled,
            len(managers),
        )

    return DiskPressureTickResult(
        free_bytes=free_bytes,
        total_bytes=total_bytes,
        soft_floor_bytes=soft_floor,
        hard_floor_bytes=hard_floor,
        tier=tier,
        scale=scale,
        managers_evicted_bytes=evicted_bytes,
        managers_count=len(managers),
        stores_count=len(stores),
        reconciled_entries=reconciled,
    )


async def run_disk_pressure_guard_tick(
    *,
    volume_path: Path,
    disk_settings: Any,
    get_managers: Callable[[], Iterable[Any]],
    get_boundary_stores: Callable[[], Iterable[Any]],
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
    run_log_reaper: Callable[[], Any] | None = None,
) -> DiskPressureTickResult | None:
    """One guard evaluation. The (blocking) disk_usage syscall and eviction
    walk run in a worker thread so a slow/contested filesystem cannot stall
    the event loop.

    ``run_log_reaper``, when given, is the log/artifact reaper (design doc
    §R4 — "trigger: server startup + the R2 tick"), invoked once per pass
    independent of disk-pressure tier: it's cheap directory hygiene, not a
    pressure response.
    """
    managers = list(get_managers())
    stores = list(get_boundary_stores())
    result = await asyncio.to_thread(
        _evaluate_tick,
        volume_path=volume_path,
        disk_settings=disk_settings,
        managers=managers,
        stores=stores,
        disk_usage=disk_usage,
    )
    if run_log_reaper is not None:
        try:
            await asyncio.to_thread(run_log_reaper)
        except Exception:
            logger.warning(
                "Disk pressure guard: log/artifact reaper failed", exc_info=True
            )
    return result


async def disk_pressure_guard_loop(
    *,
    volume_path: Path,
    disk_settings: Any,
    get_managers: Callable[[], Iterable[Any]],
    get_boundary_stores: Callable[[], Iterable[Any]],
    stop_event: asyncio.Event,
    run_log_reaper: Callable[[], Any] | None = None,
) -> None:
    """The guard's own asyncio task — scheduled once from `lifespan()`.

    Runs until `stop_event` is set (server shutdown). A tick that raises
    never kills the loop; the guard is a housekeeping actor, not something
    a transient failure should permanently disable.
    """
    while not stop_event.is_set():
        try:
            await run_disk_pressure_guard_tick(
                volume_path=volume_path,
                disk_settings=disk_settings,
                get_managers=get_managers,
                get_boundary_stores=get_boundary_stores,
                run_log_reaper=run_log_reaper,
            )
        except Exception:
            logger.warning("Disk pressure guard tick raised", exc_info=True)

        interval = max(1.0, float(disk_settings.guard_tick_interval_seconds))
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass
