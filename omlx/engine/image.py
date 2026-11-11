# SPDX-License-Identifier: Apache-2.0
"""Unadvertised image engine: one request, bounded executor steps, drain on cancel."""

import asyncio
import contextlib
import threading
import time
from pathlib import Path

import mlx.core as mx

from ..engine_core import get_mlx_executor
from ..image.pipeline import generate_steps
from ..image_inspect import inspect_bundle
from ..utils.proc_memory import get_phys_footprint
from .base import BaseNonStreamingEngine
from .forward_fairness import ForwardFairnessGate


async def _drain(future, *, propagate_cancel=False):
    """Wait through repeated cancellation without abandoning worker ownership."""
    cancelled = False
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError:
            cancelled = True
            continue
        except Exception:
            break
    result = future.result()
    if cancelled and propagate_cancel:
        raise asyncio.CancelledError
    return result


class ImageEngine(BaseNonStreamingEngine):
    def __init__(self, model_name, *, scheduler_config=None):
        super().__init__()
        self._path = str(Path(model_name).resolve())
        self._lock = asyncio.Lock()
        self._loaded = False
        self._stopping = False
        self._cancel = None
        self._required_bytes = 0
        self._soft_limit = 0
        self._fairness = ForwardFairnessGate(self._path, scheduler_config)

    @property
    def model_name(self):
        return self._path

    def set_memory_soft_limit(self, soft_limit_bytes):
        self._soft_limit = max(0, soft_limit_bytes)
        self._fairness.set_memory_soft_limit(soft_limit_bytes)

    async def start(self):
        async with self._lock:
            report = await asyncio.to_thread(inspect_bundle, self._path)
            if report["structural_status"] != "valid":
                raise ValueError(report["errors"])
            # Deliberately conservative versus the measured ~15.6 GB peak.
            # This is not a universal estimate for other models or recipes.
            self._required_bytes = int(report["physical_weight_bytes"] * 1.10)
            self._loaded = True
            self._stopping = False

    async def stop(self):
        pending = asyncio.create_task(self._stop())
        try:
            await asyncio.shield(pending)
        except asyncio.CancelledError:
            await _drain(pending)
            raise

    async def _stop(self):
        self._stopping = True
        if self._cancel is not None:
            self._cancel.set()
        async with self._lock:
            self._loaded = False

    def _admit(self):
        limit = self._soft_limit or int(
            mx.device_info().get("max_recommended_working_set_size", 0)
        )
        used = max(mx.get_active_memory(), get_phys_footprint())
        if not limit or used + self._required_bytes > limit:
            raise MemoryError("insufficient memory headroom for the fixed image recipe")

    def _step(self, steps):
        contended = self._fairness.wait_turn()
        started = time.perf_counter()
        try:
            return False, next(steps)
        except StopIteration as stop:
            return True, stop.value
        finally:
            mx.synchronize()
            self._fairness.settle(time.perf_counter() - started, 1, contended)

    async def generate(self, prompt, seed):
        if not self._loaded or self._stopping:
            raise RuntimeError("image engine is not running")
        if self._lock.locked():
            raise RuntimeError("image engine is busy; retry after the active request")
        async with self._lock:
            if self._stopping:
                raise RuntimeError("image engine is stopping")
            cancel = self._cancel = threading.Event()
            activity = self._begin_activity(
                "image", detail="Generating image", total_items=1
            )
            loop = asyncio.get_running_loop()
            steps = generate_steps(
                self._path, prompt, seed, cancel=cancel, measure_peak=False
            )
            pending = None
            try:
                pending = loop.run_in_executor(get_mlx_executor(), self._admit)
                await asyncio.shield(pending)
                while True:
                    pending = loop.run_in_executor(
                        get_mlx_executor(), self._step, steps
                    )
                    done, result = await asyncio.shield(pending)
                    if done:
                        return result
                    phase, step, total = result
                    self._update_activity(
                        activity, detail=phase, current_step=step, total_steps=total
                    )
            except BaseException:
                cancel.set()
                if pending is not None:
                    with contextlib.suppress(Exception):
                        await _drain(pending)
                raise
            finally:
                # No close() while next() is still running, including repeated cancel.
                closing = loop.run_in_executor(get_mlx_executor(), steps.close)
                try:
                    await _drain(closing, propagate_cancel=True)
                finally:
                    self._cancel = None
                    self._end_activity(activity)

    def get_stats(self):
        return {
            "model_name": self.model_name,
            "loaded": self._loaded,
            "active_requests": int(self.has_active_requests()),
            "required_headroom_bytes": self._required_bytes,
        }
