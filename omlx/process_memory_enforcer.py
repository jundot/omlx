# SPDX-License-Identifier: Apache-2.0
"""
Process-level memory enforcer for oMLX.

Monitors total Metal memory usage via mx.get_active_memory() and enforces
the max_process_memory limit by unloading LRU models from EnginePool.

The enforcer runs as a background asyncio task that polls memory usage at
a configurable interval (default: 1 second). When usage exceeds the limit,
it immediately unloads the least-recently-used non-pinned model. If the
model is mid-inference, the inference is aborted as part of engine shutdown.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import mlx.core as mx

from .utils.proc_memory import get_phys_footprint

if TYPE_CHECKING:
    from .engine_pool import EnginePool
    from .model_settings import ModelSettingsManager
    from .settings import GlobalSettings

logger = logging.getLogger(__name__)


def _format_gb(b: int) -> str:
    """Format bytes as GB string."""
    return f"{b / 1024**3:.1f}GB"


class ProcessMemoryEnforcer:
    """
    Background task that enforces process-level memory limits.

    Polls mx.get_active_memory() every poll_interval seconds and unloads
    LRU models from EnginePool when the limit is exceeded.
    """

    def __init__(
        self,
        engine_pool: EnginePool,
        max_bytes: int,
        poll_interval: float = 1.0,
        settings_manager: ModelSettingsManager | None = None,
        prefill_memory_guard: bool = True,
        global_settings: GlobalSettings | None = None,
        soft_threshold: float = 0.85,
        hard_threshold: float = 0.95,
    ):
        """
        Initialize the process memory enforcer.

        Args:
            engine_pool: The engine pool to evict models from.
            max_bytes: Maximum allowed process memory in bytes (compared
                against max(mx.get_active_memory(), phys_footprint)).
            poll_interval: Seconds between memory checks.
            settings_manager: Optional settings manager for TTL checks.
            prefill_memory_guard: Whether to enable pre-flight memory
                estimation to reject requests that would exceed limits.
            global_settings: Optional global settings for idle timeout.
            soft_threshold: Fraction of max_bytes that triggers soft action
                (LRU non-pinned eviction + admission pause; in-flight allowed).
            hard_threshold: Fraction of max_bytes that triggers hard action
                (also abort in-flight when all loaded models are pinned).
        """
        self._engine_pool = engine_pool
        self._max_bytes = max_bytes
        self._poll_interval = poll_interval
        self._settings_manager = settings_manager
        self._prefill_memory_guard = prefill_memory_guard
        self._global_settings = global_settings
        self._soft_threshold = soft_threshold
        self._hard_threshold = hard_threshold
        self._task: asyncio.Task | None = None
        self._running = False
        # Most recently observed pressure level, consumed by scheduler /
        # admission control. Updated on every poll iteration.
        self._pressure_level: str = "ok"

    @property
    def max_bytes(self) -> int:
        """Maximum allowed Metal memory in bytes."""
        return self._max_bytes

    @max_bytes.setter
    def max_bytes(self, value: int) -> None:
        old = self._max_bytes
        self._max_bytes = value
        if self._running:
            self._propagate_memory_limit()
            self._set_metal_memory_limit()
        logger.info(
            f"Process memory limit changed: "
            f"{_format_gb(old)} -> {_format_gb(value)}"
        )

    @property
    def is_running(self) -> bool:
        """Whether the enforcement loop is active."""
        return self._running

    def start(self) -> None:
        """Start the background enforcement loop."""
        if self._running:
            return
        self._running = True
        self._propagate_memory_limit()
        self._set_metal_memory_limit()
        self._task = asyncio.create_task(self._enforcement_loop())
        logger.info(
            f"Process memory enforcer started "
            f"(limit: {_format_gb(self._max_bytes)}, "
            f"interval: {self._poll_interval}s)"
        )

    def _get_hard_limit_bytes(self) -> int:
        """Hard limit for inline prefill check: system_ram - 4GB.

        Returns 0 if enforcement is disabled (max_bytes <= 0).
        Always >= max_bytes so prefill gets headroom above the soft limit.

        Note: this is the absolute system ceiling for the scheduler's prefill
        check, distinct from the enforcer's own soft/hard watermarks
        (`_soft_bytes` / `_hard_bytes`) which trigger LRU eviction.
        """
        if self._max_bytes <= 0:
            return 0
        from .settings import get_system_memory

        return max(get_system_memory() - 4 * 1024**3, self._max_bytes)

    @property
    def _soft_bytes(self) -> int:
        """Soft watermark: max_bytes * soft_threshold."""
        if self._max_bytes <= 0:
            return 0
        return int(self._max_bytes * self._soft_threshold)

    @property
    def _hard_bytes(self) -> int:
        """Hard watermark: max_bytes * hard_threshold."""
        if self._max_bytes <= 0:
            return 0
        return int(self._max_bytes * self._hard_threshold)

    def _current_usage_bytes(self) -> int:
        """Process memory usage as seen by macOS jetsam.

        Combines MLX-reported active memory and the kernel phys_footprint
        ledger. phys_footprint covers anonymous + IOAccelerator + dirty
        file-backed, so it usually dominates; we take max() so MLX-internal
        cache that hasn't been mirrored into phys yet still triggers.
        """
        return max(mx.get_active_memory(), get_phys_footprint())

    def get_pressure_level(self) -> str:
        """Return cached pressure level: 'ok', 'soft', or 'hard'.

        Consumed by scheduler `_schedule_waiting` and HTTP admission control.
        Updated on every enforcer poll iteration.
        """
        return self._pressure_level if self._running else "ok"

    def _set_metal_memory_limit(self) -> None:
        """No-op. Metal-level limits removed to prevent model load swap.

        mx.set_memory_limit() causes MLX to aggressively reclaim cached
        buffers during model loading, creating alloc/free churn that
        pushes the system into swap. All memory enforcement is handled
        by mx.get_active_memory() polling instead. (#429)
        """
        pass

    def _clear_metal_memory_limit(self) -> None:
        """No-op. See _set_metal_memory_limit."""
        pass

    @property
    def prefill_memory_guard(self) -> bool:
        """Whether prefill memory guard is enabled."""
        return self._prefill_memory_guard

    @prefill_memory_guard.setter
    def prefill_memory_guard(self, value: bool) -> None:
        self._prefill_memory_guard = value
        if self._running:
            self._propagate_memory_limit()
            if value:
                self._set_metal_memory_limit()
            else:
                self._clear_metal_memory_limit()
        logger.info(f"Prefill memory guard: {'enabled' if value else 'disabled'}")

    def _propagate_memory_limit(self) -> None:
        """Propagate soft/hard memory limits to schedulers for inline prefill checking."""
        hard_limit = self._get_hard_limit_bytes()
        admission_paused = self._pressure_level != "ok"
        for entry in self._engine_pool._entries.values():
            if entry.engine is not None:
                scheduler = getattr(entry.engine, "scheduler", None)
                if scheduler is not None:
                    scheduler._memory_limit_bytes = self._max_bytes
                    scheduler._memory_hard_limit_bytes = hard_limit
                    scheduler._prefill_memory_guard = self._prefill_memory_guard
                    scheduler._admission_paused = admission_paused
                    bg = getattr(scheduler, "batch_generator", None)
                    if bg is not None and hasattr(bg, "_memory_limit_bytes"):
                        bg._memory_limit_bytes = self._max_bytes
                        bg._memory_hard_limit_bytes = hard_limit

    async def stop(self) -> None:
        """Stop the background enforcement loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Process memory enforcer stopped")

    async def _enforcement_loop(self) -> None:
        """Main polling loop."""
        while self._running:
            try:
                await self._check_and_enforce()
                await self._check_ttl()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Process memory enforcer error: {e}")
            await asyncio.sleep(self._poll_interval)

    async def _check_ttl(self) -> None:
        """Check and unload models that exceeded their TTL."""
        if self._settings_manager is None:
            return
        await self._engine_pool.check_ttl_expirations(
            self._settings_manager,
            global_idle_timeout_seconds=(
                self._global_settings.idle_timeout.idle_timeout_seconds
                if self._global_settings else None
            ),
        )

    async def _check_and_enforce(self) -> None:
        """Check current memory and enforce 2-watermark policy.

        Pressure levels:
        - ok (current < soft): no action, ensure admission unpaused.
        - soft (soft <= current < hard): LRU non-pinned eviction + signal
          schedulers to pause new admissions (in-flight requests proceed).
        - hard (current >= hard): full enforcement — LRU evict, abort
          in-flight when only pinned remain, abort in-progress model loads.

        Pressure target on recovery is the soft threshold (always evict
        back below soft to avoid oscillation when single eviction lands
        just under hard).
        """
        if self._max_bytes <= 0:
            self._pressure_level = "ok"
            return

        current = self._current_usage_bytes()
        soft = self._soft_bytes
        hard = self._hard_bytes
        prev_level = self._pressure_level

        if current < soft:
            new_level = "ok"
        elif current < hard:
            new_level = "soft"
        else:
            new_level = "hard"

        # Update cached level and propagate admission_paused immediately so
        # the scheduler stops admitting new prefills before we start evicting.
        if new_level != prev_level:
            self._pressure_level = new_level
            self._propagate_memory_limit()
            logger.info(
                f"Memory pressure level: {prev_level} -> {new_level} "
                f"(current={_format_gb(current)}, "
                f"soft={_format_gb(soft)}, hard={_format_gb(hard)})"
            )

        if new_level == "ok":
            return

        # Recover below soft regardless of level — prevents oscillation
        # at the boundary.
        target = soft

        async with self._engine_pool._lock:
            while self._current_usage_bytes() > target:
                victim = self._engine_pool._find_lru_victim()
                if victim is not None:
                    loaded_non_pinned = [
                        mid
                        for mid, e in self._engine_pool._entries.items()
                        if e.engine is not None and not e.is_pinned
                    ]
                    if len(loaded_non_pinned) > 1:
                        # Multiple non-pinned: evict LRU victim cleanly.
                        # abort_all_requests is fired before _unload_engine
                        # so clients receive proper error responses instead
                        # of silent disconnect.
                        entry = self._engine_pool._entries.get(victim)
                        if entry and entry.engine is not None:
                            if hasattr(entry.engine, "abort_all_requests"):
                                aborted = await entry.engine.abort_all_requests()
                                if aborted > 0:
                                    logger.warning(
                                        f"Aborted {aborted} requests on "
                                        f"'{victim}' before eviction"
                                    )
                        logger.warning(
                            f"Evicting model '{victim}' (pressure={new_level})"
                        )
                        await self._engine_pool._unload_engine(victim)
                        continue

                    # Only one non-pinned model remains.
                    if new_level == "hard":
                        # Abort in-flight requests, keep model loaded —
                        # frees KV blocks so short-context follow-ups work.
                        entry = self._engine_pool._entries.get(victim)
                        if entry and entry.engine is not None:
                            if hasattr(entry.engine, "abort_all_requests"):
                                aborted = await entry.engine.abort_all_requests()
                                if aborted > 0:
                                    logger.warning(
                                        f"Aborted {aborted} requests on "
                                        f"'{victim}' due to hard memory "
                                        f"pressure (model kept loaded)"
                                    )
                    # soft: leave in-flight alone — admission pause already
                    # signaled, eviction can't help further without aborts.
                    break

                # No non-pinned victim. All loaded models are pinned.
                if new_level == "hard":
                    # Hard only: abort any in-progress model loads.
                    aborted_any = False
                    for entry in self._engine_pool._entries.values():
                        if entry.is_loading and not entry.abort_loading:
                            logger.warning(
                                f"Aborting in-progress load of "
                                f"'{entry.model_id}' (hard memory pressure)"
                            )
                            entry.abort_loading = True
                            aborted_any = True
                    if not aborted_any:
                        has_loaded = any(
                            e.engine is not None
                            for e in self._engine_pool._entries.values()
                        )
                        if has_loaded:
                            logger.warning(
                                "Hard memory pressure but all loaded models "
                                "are pinned and no loads in progress."
                            )
                        else:
                            logger.warning(
                                "Hard memory pressure but no models loaded."
                            )
                # soft + all pinned: nothing to do beyond admission pause.
                break

        # Re-evaluate level after eviction completes so admission state
        # reflects post-eviction reality on the next propagate.
        post_current = self._current_usage_bytes()
        if post_current < soft:
            post_level = "ok"
        elif post_current < hard:
            post_level = "soft"
        else:
            post_level = "hard"
        if post_level != self._pressure_level:
            self._pressure_level = post_level
            self._propagate_memory_limit()
            logger.info(
                f"Memory pressure post-eviction: {new_level} -> {post_level} "
                f"(current={_format_gb(post_current)})"
            )

    def get_status(self) -> dict:
        """Get enforcer status for monitoring endpoints."""
        current = mx.get_active_memory() if self._running else 0
        return {
            "enabled": self._running,
            "max_bytes": self._max_bytes,
            "max_formatted": _format_gb(self._max_bytes),
            "current_bytes": current,
            "current_formatted": _format_gb(current),
            "utilization": (
                current / self._max_bytes if self._max_bytes > 0 else 0.0
            ),
        }
