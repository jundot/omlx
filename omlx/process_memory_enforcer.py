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

if TYPE_CHECKING:
    from .engine_pool import EnginePool

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
    ):
        """
        Initialize the process memory enforcer.

        Args:
            engine_pool: The engine pool to evict models from.
            max_bytes: Maximum allowed Metal memory in bytes.
            poll_interval: Seconds between memory checks.
        """
        self._engine_pool = engine_pool
        self._max_bytes = max_bytes
        self._poll_interval = poll_interval
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
        self._task = asyncio.create_task(self._enforcement_loop())
        logger.info(
            f"Process memory enforcer started "
            f"(limit: {_format_gb(self._max_bytes)}, "
            f"interval: {self._poll_interval}s)"
        )

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
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Process memory enforcer error: {e}")
            await asyncio.sleep(self._poll_interval)

    async def _check_and_enforce(self) -> None:
        """Check current memory and enforce limit if exceeded."""
        current = mx.get_active_memory()
        if current <= self._max_bytes:
            return

        overage = current - self._max_bytes
        logger.warning(
            f"Process memory limit exceeded: "
            f"{_format_gb(current)} / {_format_gb(self._max_bytes)} "
            f"(over by {_format_gb(overage)})"
        )

        # Acquire EnginePool lock and unload LRU models until under limit
        async with self._engine_pool._lock:
            while mx.get_active_memory() > self._max_bytes:
                victim = self._engine_pool._find_lru_victim()
                if victim is not None:
                    logger.warning(
                        f"Evicting model '{victim}' to enforce "
                        f"process memory limit"
                    )
                    await self._engine_pool._unload_engine(victim)
                    continue

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
