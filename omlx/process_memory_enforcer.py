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
import gc
import logging
from enum import Enum
from typing import TYPE_CHECKING

import mlx.core as mx

from .engine_core import get_mlx_executor

if TYPE_CHECKING:
    from .engine_pool import EnginePool
    from .model_settings import ModelSettingsManager

logger = logging.getLogger(__name__)


def _format_gb(b: int) -> str:
    """Format bytes as GB string."""
    return f"{b / 1024**3:.1f}GB"


class MemoryWatermark(Enum):
    """Memory pressure levels for pre-load safety decisions."""
    GREEN = "green"    # < 65% utilization
    YELLOW = "yellow"  # 65-80% utilization
    RED = "red"        # 80-90% utilization
    FATAL = "fatal"    # > 90% utilization

    @classmethod
    def from_utilization(cls, utilization: float) -> "MemoryWatermark":
        if utilization < 0.65:
            return cls.GREEN
        elif utilization < 0.80:
            return cls.YELLOW
        elif utilization < 0.90:
            return cls.RED
        else:
            return cls.FATAL


class WatermarkAction(Enum):
    LOAD_DIRECTLY = "load_directly"
    RECLAIM_THEN_LOAD = "reclaim_then_load"
    RESTART_THEN_LOAD = "restart_then_load"
    QUEUE_AND_WAIT = "queue_and_wait"


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
    ):
        """
        Initialize the process memory enforcer.

        Args:
            engine_pool: The engine pool to evict models from.
            max_bytes: Maximum allowed Metal memory in bytes.
            poll_interval: Seconds between memory checks.
            settings_manager: Optional settings manager for TTL checks.
            prefill_memory_guard: Whether to enable pre-flight memory
                estimation to reject requests that would exceed limits.
        """
        self._engine_pool = engine_pool
        self._max_bytes = max_bytes
        self._poll_interval = poll_interval
        self._settings_manager = settings_manager
        self._prefill_memory_guard = prefill_memory_guard
        self._task: asyncio.Task | None = None
        self._running = False

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
        """
        if self._max_bytes <= 0:
            return 0
        from .settings import get_system_memory

        return max(get_system_memory() - 4 * 1024**3, self._max_bytes)

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
        for entry in self._engine_pool._entries.values():
            if entry.engine is not None:
                scheduler = getattr(entry.engine, "scheduler", None)
                if scheduler is not None:
                    scheduler._memory_limit_bytes = self._max_bytes
                    scheduler._memory_hard_limit_bytes = hard_limit
                    scheduler._prefill_memory_guard = self._prefill_memory_guard
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
        await self._engine_pool.check_ttl_expirations(self._settings_manager)

    async def _check_and_enforce(self) -> None:
        """Check current memory and enforce limit if exceeded.

        Handles three scenarios via the while loop:
        1. Multiple models, one inferring: evict LRU (idle) model,
           inference on the other continues.
        2. Single model: abort all requests, keep model loaded.
           Short-context requests can be served afterward.
        3. Multiple models, both inferring: first iteration evicts LRU
           (aborting its requests), second iteration aborts remaining
           single model's requests.
        """
        if self._max_bytes <= 0:
            return

        current = mx.get_active_memory()
        if current <= self._max_bytes:
            return

        overage = current - self._max_bytes
        logger.warning(
            f"Process memory limit exceeded: "
            f"{_format_gb(current)} / {_format_gb(self._max_bytes)} "
            f"(over by {_format_gb(overage)})"
        )

        # Acquire EnginePool lock and unload LRU models until under limit.
        # Note: prefill loops self-check via _memory_limit_bytes (same thread,
        # no GIL issue), so they will abort independently of this enforcer.
        async with self._engine_pool._lock:
            while mx.get_active_memory() > self._max_bytes:
                victim = self._engine_pool._find_lru_victim()
                if victim is not None:
                    # Count loaded non-pinned models
                    loaded_non_pinned = [
                        mid
                        for mid, e in self._engine_pool._entries.items()
                        if e.engine is not None and not e.is_pinned
                    ]
                    if len(loaded_non_pinned) > 1:
                        # Multiple models: evict LRU victim.
                        # First abort active requests so clients receive
                        # error messages — EngineCore.stop() only cancels
                        # the engine loop silently without notifying collectors.
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
                            f"Evicting model '{victim}' to enforce "
                            f"process memory limit"
                        )
                        await self._engine_pool._unload_engine(victim)
                        continue
                    else:
                        # Single model: abort all requests, keep model
                        # loaded. This frees KV cache blocks internally
                        # so short-context requests can be served without
                        # new Metal allocation.
                        entry = self._engine_pool._entries.get(victim)
                        if entry and entry.engine is not None:
                            if hasattr(entry.engine, "abort_all_requests"):
                                aborted = await entry.engine.abort_all_requests()
                                if aborted > 0:
                                    logger.warning(
                                        f"Aborted {aborted} requests on "
                                        f"'{victim}' due to memory pressure "
                                        f"(model kept loaded)"
                                    )
                        break

                # No loaded non-pinned model to evict.
                # Check if any model is currently loading — request abort.
                aborted_any = False
                for entry in self._engine_pool._entries.values():
                    if entry.is_loading and not entry.abort_loading:
                        logger.warning(
                            f"Requesting abort of loading model "
                            f"'{entry.model_id}' — process memory "
                            f"limit exceeded"
                        )
                        entry.abort_loading = True
                        aborted_any = True

                if not aborted_any:
                    # Nothing we can do — all models are either pinned
                    # or there are no loaded/loading models
                    has_loaded = any(
                        e.engine is not None
                        for e in self._engine_pool._entries.values()
                    )
                    if has_loaded:
                        logger.warning(
                            "Process memory limit exceeded but all "
                            "loaded models are pinned — cannot evict."
                        )
                    else:
                        logger.warning(
                            "Process memory limit exceeded but no "
                            "models are loaded to evict."
                        )
                break

    def get_status(self) -> dict:
        """Get enforcer status for monitoring endpoints."""
        current = mx.get_active_memory() if self._running else 0
        utilization = current / self._max_bytes if self._max_bytes > 0 else 0.0
        return {
            "enabled": self._running,
            "max_bytes": self._max_bytes,
            "max_formatted": _format_gb(self._max_bytes),
            "current_bytes": current,
            "current_formatted": _format_gb(current),
            "utilization": utilization,
            "watermark": MemoryWatermark.from_utilization(utilization).value,
        }

    def get_watermark_level(self) -> MemoryWatermark:
        """Get current memory watermark level."""
        if self._max_bytes <= 0:
            return MemoryWatermark.GREEN
        current = mx.get_active_memory()
        utilization = current / self._max_bytes
        return MemoryWatermark.from_utilization(utilization)

    def get_memory_diagnostics(self) -> dict:
        """Get comprehensive memory diagnostics."""
        current = mx.get_active_memory()
        peak = mx.get_peak_memory()
        cache = mx.get_cache_memory()
        model_est = self._engine_pool.current_model_memory
        utilization = current / self._max_bytes if self._max_bytes > 0 else 0.0
        watermark = MemoryWatermark.from_utilization(utilization)
        return {
            "active_gb": round(current / 1024**3, 2),
            "peak_gb": round(peak / 1024**3, 2),
            "cache_gb": round(cache / 1024**3, 2),
            "model_est_gb": round(model_est / 1024**3, 2),
            "loaded_models": self._engine_pool.loaded_model_count,
            "limit_gb": round(self._max_bytes / 1024**3, 2),
            "utilization_pct": round(utilization * 100, 1),
            "watermark": watermark.value,
        }

    async def pre_load_check(
        self, new_model_size_bytes: int, engine_type: str = "batched",
    ) -> tuple:
        """
        Pre-load memory safety check.

        Deducts reclaimable Metal cache from current usage and scales
        runtime overhead by engine type for more accurate projections.

        Returns (WatermarkAction, diagnostics_dict).
        """
        current_active = mx.get_active_memory()
        current_cache = mx.get_cache_memory()
        effective_current = current_active - current_cache

        if engine_type in ("embedding", "reranker", "audio_stt", "audio_tts", "audio_sts"):
            overhead_pct = 0.05
        else:
            overhead_pct = max(0.10, min(0.25, 0.30 - new_model_size_bytes / (200 * 1024**3)))
        runtime_overhead = int(new_model_size_bytes * overhead_pct)

        projected = effective_current + new_model_size_bytes + runtime_overhead
        utilization = projected / self._max_bytes if self._max_bytes > 0 else 0.0
        watermark = MemoryWatermark.from_utilization(utilization)

        diagnostics = {
            "current_gb": round(current_active / 1024**3, 2),
            "cache_gb": round(current_cache / 1024**3, 2),
            "effective_current_gb": round(effective_current / 1024**3, 2),
            "new_model_gb": round(new_model_size_bytes / 1024**3, 2),
            "overhead_gb": round(runtime_overhead / 1024**3, 2),
            "overhead_pct": round(overhead_pct * 100, 1),
            "projected_gb": round(projected / 1024**3, 2),
            "limit_gb": round(self._max_bytes / 1024**3, 2),
            "utilization_pct": round(utilization * 100, 1),
            "watermark": watermark.value,
            "loaded_model_count": self._engine_pool.loaded_model_count,
        }

        action_map = {
            MemoryWatermark.GREEN: WatermarkAction.LOAD_DIRECTLY,
            MemoryWatermark.YELLOW: WatermarkAction.RECLAIM_THEN_LOAD,
            MemoryWatermark.RED: WatermarkAction.RESTART_THEN_LOAD,
            MemoryWatermark.FATAL: WatermarkAction.QUEUE_AND_WAIT,
        }
        return action_map[watermark], diagnostics

    async def emergency_reclaim(self) -> bool:
        """
        Emergency memory reclaim: aggressive GC + Metal cache clear.

        Uses the dedicated MLX executor to avoid concurrent Metal operations.
        Returns True if memory dropped.
        """
        logger.warning("Performing emergency memory reclaim...")
        loop = asyncio.get_running_loop()
        before = mx.get_active_memory()

        for _i in range(3):
            gc.collect()
            await loop.run_in_executor(
                get_mlx_executor(), lambda: (mx.synchronize(), mx.clear_cache())
            )
            await asyncio.sleep(1.0)

        gc.collect()
        await loop.run_in_executor(
            get_mlx_executor(), lambda: (mx.synchronize(), mx.clear_cache())
        )
        await asyncio.sleep(2.0)
        after = mx.get_active_memory()

        freed = before - after
        logger.info(
            f"Emergency reclaim complete: freed {_format_gb(freed)} "
            f"({_format_gb(before)} -> {_format_gb(after)})"
        )
        return after < before
