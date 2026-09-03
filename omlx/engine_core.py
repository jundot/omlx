# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-mlx (https://github.com/vllm-project/vllm-mlx).
"""
Engine Core for oMLX continuous batching.

This module provides the EngineCore class that coordinates:
- Model loading and management
- Request scheduling via Scheduler
- Async request processing
- Output streaming

The design follows vLLM's engine architecture adapted for MLX.
"""

import asyncio
import concurrent.futures
import gc
import logging
import os
import threading
import time
import uuid
import weakref
from contextlib import suppress
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import mlx.core as mx

from .exceptions import (
    PrefillMemoryAbortedError,
    PrefillMemoryExceededError,
    describe_ceiling_binding,
)
from .keepwarm import (
    CompiledMetalKeepwarmTouch,
    KeepwarmAction,
    KeepwarmConfig,
    KeepwarmController,
)
from .model_registry import get_registry
from .output_collector import RequestOutputCollector, RequestStreamState
from .request import Request, RequestOutput, SamplingParams
from .scheduler import Scheduler, SchedulerConfig, _sync_and_clear_cache
from .utils.fatal import FATAL_TEARDOWN_TIMEOUT_S, fatal_exit
from .utils.hardware import format_bytes
from .utils.metal_sync import clear_thread_streams

logger = logging.getLogger(__name__)


# Prompt-tail maintenance is process-global GPU work even though each model
# has its own EngineCore and MLX stream.  A single execution lease prevents
# two individually-safe memory estimates from overcommitting together, while
# a cheap epoch lets any engine admission cancel hidden work without waiting
# for its full suffix.  Weak membership never extends an engine lifetime.
_prompt_tail_global_state_lock = threading.RLock()
_prompt_tail_global_run_lock = threading.Lock()
_prompt_tail_global_epoch = 0
_prompt_tail_global_engines: weakref.WeakSet[Any] = weakref.WeakSet()


def _register_prompt_tail_engine(engine: Any) -> None:
    global _prompt_tail_global_epoch
    with _prompt_tail_global_state_lock:
        _prompt_tail_global_epoch += 1
        _prompt_tail_global_engines.add(engine)


def _unregister_prompt_tail_engine(engine: Any) -> None:
    global _prompt_tail_global_epoch
    with _prompt_tail_global_state_lock:
        _prompt_tail_global_epoch += 1
        _prompt_tail_global_engines.discard(engine)


def _invalidate_global_prompt_tail() -> int:
    global _prompt_tail_global_epoch
    with _prompt_tail_global_state_lock:
        _prompt_tail_global_epoch += 1
        return _prompt_tail_global_epoch


def _global_prompt_tail_epoch() -> int:
    with _prompt_tail_global_state_lock:
        return _prompt_tail_global_epoch


def _other_prompt_tail_engine_busy(owner: Any) -> bool:
    with _prompt_tail_global_state_lock:
        engines = list(_prompt_tail_global_engines)
    for engine in engines:
        if engine is owner or getattr(engine, "_closed", True):
            continue
        if not getattr(engine, "_running", False):
            continue
        try:
            if (
                engine._pending_admission_count()
                or engine._scheduler_has_active_user_requests()
                or engine.scheduler.has_requests()
            ):
                return True
        except Exception:
            # Unknown engine state is not an idle guarantee.
            return True
    return False


def _global_prompt_tail_invalid(owner: Any, epoch: int) -> bool:
    with _prompt_tail_global_state_lock:
        if epoch != _prompt_tail_global_epoch:
            return True
    return _other_prompt_tail_engine_busy(owner)


def _raise_request_output_error(output: RequestOutput) -> None:
    if output.error_code in ("prefill_memory_exceeded", "prefill_memory_aborted"):
        metadata = output.error_metadata or {}
        request_id = metadata.get("request_id")
        estimated_bytes = metadata.get("estimated_bytes")
        limit_bytes = metadata.get("limit_bytes")
        # Both are memory-guard outcomes and share the HTTP 400 mapping; the
        # aborted subclass only changes the wording (admitted then killed
        # mid-prefill vs rejected before it started).
        error_type = (
            PrefillMemoryAbortedError
            if output.error_code == "prefill_memory_aborted"
            else PrefillMemoryExceededError
        )
        raise error_type(
            message=output.error or "Prefill memory exceeded",
            request_id=str(request_id) if request_id is not None else output.request_id,
            estimated_bytes=(
                int(estimated_bytes) if estimated_bytes is not None else None
            ),
            limit_bytes=int(limit_bytes) if limit_bytes is not None else None,
        )
    raise RuntimeError(output.error)


_global_mlx_executor: concurrent.futures.ThreadPoolExecutor | None = None


def _final_engine_thread_reclaim(stream: Any) -> None:
    """Drop Python cycles and reclaim MLX buffers on the engine worker thread."""
    gc.collect()
    _sync_and_clear_cache(stream)
    gc.collect()
    clear_thread_streams()


def _final_global_mlx_thread_reclaim() -> None:
    """Reclaim MLX state before the process-wide worker thread exits."""
    gc.collect()
    _sync_and_clear_cache()
    gc.collect()
    clear_thread_streams()


def _init_mlx_thread() -> None:
    """Replace generation_stream with a thread-local stream on the executor thread.

    mlx-lm's module-level ``generation_stream`` is created at import time in
    whichever thread imported it first (the main thread at server startup).
    Arrays produced inside ``with mx.stream(generation_stream):`` blocks carry
    that stream reference.  If the stream was created on the main thread,
    subsequent ``.item()`` / ``mx.synchronize()`` calls from the executor
    thread fail with "There is no Stream(gpu, 0) in current thread".

    Fix: create a thread-local stream HERE and replace the module-level
    ``generation_stream`` in mlx_lm.generate and omlx.scheduler.
    """
    import sys

    import mlx.core as mx

    stream = mx.new_thread_local_stream(mx.default_device())

    gen_mod = sys.modules.get("mlx_lm.generate")
    if gen_mod is not None:
        gen_mod.generation_stream = stream

    sched_mod = sys.modules.get("omlx.scheduler")
    if sched_mod is not None:
        sched_mod.generation_stream = stream

    logger.info(f"MLX executor thread initialized: generation_stream = {stream}")


def get_mlx_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Get or create the global MLX executor (lazy singleton).

    mlx-lm's BatchGenerator uses a module-level Metal stream
    (generation_stream), so ALL MLX GPU operations across all models
    MUST be serialized onto one thread to prevent Metal command buffer
    races that cause segfaults. See issue #85.
    """
    global _global_mlx_executor
    if _global_mlx_executor is None:
        _global_mlx_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mlx-global",
            initializer=_init_mlx_thread,
        )
    return _global_mlx_executor


def shutdown_mlx_executor() -> None:
    """Shut down the global MLX worker after clearing its thread-local streams."""
    global _global_mlx_executor
    executor = _global_mlx_executor
    if executor is None:
        return

    try:
        future = executor.submit(_final_global_mlx_thread_reclaim)
    except RuntimeError:
        # An embedding application may have shut the executor down directly.
        future = None

    if future is not None:
        try:
            future.result(timeout=FATAL_TEARDOWN_TIMEOUT_S)
        except concurrent.futures.TimeoutError:
            fatal_exit(
                "Global MLX executor teardown timed out after "
                f"{FATAL_TEARDOWN_TIMEOUT_S:.0f}s"
            )
        except Exception as exc:
            fatal_exit(f"Global MLX executor teardown failed: {exc!r}")

    executor.shutdown(wait=False)
    _global_mlx_executor = None


@dataclass
class EngineConfig:
    """Configuration for the engine."""

    model_name: str = ""
    scheduler_config: Optional[SchedulerConfig] = None
    step_interval: float = 0.05  # Idle wait timeout; requests wake the loop
    stream_interval: int = 1  # Tokens to batch before streaming (1=every token)
    prefill_eviction_callback: Optional[Callable[[Any], Awaitable[bool]]] = None
    keepwarm_config: KeepwarmConfig = field(
        default_factory=KeepwarmConfig.for_local_engine
    )
    # Decode burst: run several scheduler.step() calls per run_in_executor
    # hand-off instead of one. Each decode token otherwise bounces back to the
    # event loop, ping-ponging the GIL with the asyncio loop + uvicorn on the
    # main thread; bursting keeps the MLX thread holding the GIL continuously.
    # scheduler.step() services aborts/admission/finish every step, so
    # correctness is unchanged and memory is identical (same tokens decoded,
    # same KV cache; only a small list of K SchedulerOutputs is held per
    # burst). The budget is a TIME ceiling so the event-loop pause (and thus
    # new-request admission / abort / HTTP latency) is bounded consistently
    # across hardware, and a slow prefill-chunk step ends the burst.
    #
    # Adaptive: with a single active request (the common local/single-user
    # case) there is no concurrent request to stay responsive to, so we burst
    # aggressively (decode_burst_budget_single_s). Once concurrent, we use the
    # tight decode_burst_budget_s to keep admission/abort latency low.
    # max_steps is a safety cap (bounds the host-side output list), NOT a
    # memory knob. Set both budgets <= 0, or max_steps <= 1, to disable.
    decode_burst_max_steps: int = field(
        default_factory=lambda: int(os.environ.get("OMLX_DECODE_BURST_MAX_STEPS", "64"))
    )
    decode_burst_budget_single_s: float = field(
        default_factory=lambda: float(
            os.environ.get("OMLX_DECODE_BURST_BUDGET_SINGLE_S", "0.1")
        )
    )
    decode_burst_budget_s: float = field(
        default_factory=lambda: float(
            os.environ.get("OMLX_DECODE_BURST_BUDGET_S", "0.03")
        )
    )


class EngineCore:
    """
    Core engine for oMLX inference with continuous batching.

    This engine runs the generation loop and manages request lifecycle.
    It provides both sync and async interfaces for request handling.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[EngineConfig] = None,
        engine_id: Optional[str] = None,
        force_model_ownership: bool = False,
    ):
        """
        Initialize the engine.

        Args:
            model: The MLX model
            tokenizer: The tokenizer
            config: Engine configuration
            engine_id: Optional unique ID for this engine (auto-generated if None)
            force_model_ownership: Legacy compatibility flag. A stale registry
                entry may be replaced, but a different live owner always raises
                ModelOwnershipError and must be closed first.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or EngineConfig()
        self._engine_id = engine_id or str(uuid.uuid4())
        self._owns_model = False
        self._closed = False

        # Acquire model ownership
        # Engine construction is itself a process-global admission: cancel any
        # hidden idle maintenance before a forced ownership transfer can queue
        # teardown on the previous owner's executor.
        _invalidate_global_prompt_tail()
        registry = get_registry()
        registry.acquire(
            model=model,
            engine=self,
            engine_id=self._engine_id,
            force=force_model_ownership,
        )
        self._owns_model = True

        # Per-engine executor with dedicated mx.Stream (#1248).
        # Each EngineCore gets its own thread + GPU stream so different
        # models can run scheduler.step() concurrently.
        self._mlx_stream = mx.new_thread_local_stream(mx.default_device())
        self._mlx_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"mlx-engine-{self._engine_id[:8]}",
        )

        # Create scheduler with per-engine stream
        scheduler_config = self.config.scheduler_config or SchedulerConfig()
        self.scheduler = Scheduler(
            model=model,
            tokenizer=tokenizer,
            config=scheduler_config,
            stream=self._mlx_stream,
        )

        # Output collectors for low-latency streaming (vLLM pattern)
        self._output_collectors: Dict[str, RequestOutputCollector] = {}
        self._stream_states: Dict[str, RequestStreamState] = {}
        self._finished_events: Dict[str, asyncio.Event] = {}
        # Finish timestamps for orphan-collector reaping (#1154).
        # Normally a consumer drains and removes its own collector, but if the
        # client disconnects mid-stream the SSE generator chain is abandoned and
        # its cleanup finally only runs at GC time, so the collector lingers and
        # the dashboard keeps showing the request as "Generating".
        self._finished_at: Dict[str, float] = {}
        self._last_reap = 0.0

        # Engine state
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._wake_event: Optional[asyncio.Event] = None
        self._start_time: Optional[float] = None
        self._steps_executed = 0
        resident_cache = getattr(self.scheduler, "_exact_resident_cache", None)
        self._prompt_tail_resident_baseline_entries = int(
            getattr(resident_cache, "max_entries", 0) or 0
        )
        self._keepwarm = KeepwarmController(self.config.keepwarm_config)
        self._ensure_prompt_tail_resident_capacity()
        # Lazily created and used only on _mlx_executor. The dedicated pulse
        # stream cannot drain the model stream, and the object retains one
        # four-byte input rather than model, request, or cache state.
        self._compiled_metal_keepwarm = CompiledMetalKeepwarmTouch(mx)
        self._pending_admissions = 0
        self._pending_admissions_lock = threading.Lock()
        self._next_keepwarm_check_at = 0.0
        self._prompt_tail_lock = threading.Lock()
        self._prompt_tail_plan: tuple[str | list[int], float, int] | None = None
        self._prompt_tail_epoch = 0
        self._prompt_tail_inflight_epoch: int | None = None
        self._prompt_tail_stats = {
            "scheduled": 0,
            "published": 0,
            "skipped": 0,
            "cancelled": 0,
            "failures": 0,
            "last_result": None,
        }
        _register_prompt_tail_engine(self)

        # Drop transient aliases after ownership moves to the engine/scheduler
        # graph, so close()/deep_reset() can make that graph unreachable.
        model = None
        tokenizer = None

        logger.debug(f"Engine {self._engine_id} initialized")

    async def start(self) -> None:
        """Start the engine loop."""
        if self._running:
            return

        self._loop = asyncio.get_running_loop()
        self._wake_event = asyncio.Event()
        self._running = True
        self._start_time = time.time()
        self._task = asyncio.create_task(self._engine_loop())
        logger.info("Engine started")

    async def stop(self) -> None:
        """Stop the engine loop."""
        self._cancel_prompt_tail_prewarm("engine-stop")
        self._running = False
        if self._wake_event is not None:
            self._wake_event.set()
        if self._task:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self._wake_event = None
        self._loop = None
        logger.info("Engine stopped")

    def is_running(self) -> bool:
        """Check if engine is running."""
        return self._running

    def _wake_engine_loop(self) -> None:
        """Wake the idle engine loop after scheduler-visible state changes."""
        event = getattr(self, "_wake_event", None)
        loop = getattr(self, "_loop", None)
        if event is None or loop is None or loop.is_closed():
            return
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is loop:
            event.set()
        else:
            loop.call_soon_threadsafe(event.set)

    def _step_burst(self) -> list:
        """Run scheduler.step() several times in one executor hand-off.

        Each decode token otherwise bounces back to the event loop, which
        ping-pongs the GIL with the asyncio loop + uvicorn on the main thread
        (~1ms/token of contention). Chaining a few steps lets the MLX thread
        hold the GIL continuously (in-process sync loop hits ~80 tok/s vs ~74
        through the per-token async hand-off).

        scheduler.step() services aborts/admission/finish every step, so
        correctness is unchanged; the only cost is event-loop responsiveness,
        bounded by decode_burst_budget_s. Stops early when no work remains, a
        prefill eviction needs the (async) callback, or the budget elapses —
        the budget also ends the burst when a slow prefill-chunk step lands.

        Runs on the MLX executor thread. Returns the SchedulerOutputs in order.
        """
        max_steps = self.config.decode_burst_max_steps
        outputs = [self.scheduler.step()]
        if max_steps <= 1:
            return outputs
        # Adaptive budget: single active request -> aggressive (nothing else to
        # stay responsive to); concurrent -> tight to keep admission/abort low.
        running = getattr(self.scheduler, "running", None)
        single = running is None or len(running) <= 1
        budget = (
            self.config.decode_burst_budget_single_s
            if single
            else self.config.decode_burst_budget_s
        )
        if budget <= 0:
            return outputs
        deadline = time.monotonic() + budget
        while len(outputs) < max_steps:
            last = outputs[-1]
            if (
                not last.has_work  # throttled/idle: stop and let the loop wait
                or not self.scheduler.has_requests()
                or last.prefill_eviction_request is not None
                or time.monotonic() >= deadline
            ):
                break
            outputs.append(self.scheduler.step())
        return outputs

    def configure_keepwarm(self, enabled: bool) -> None:
        """Apply the experimental master switch to this loaded engine."""

        self._keepwarm.configure(enabled)
        self.config.keepwarm_config = self._keepwarm.config
        if enabled:
            self._ensure_prompt_tail_resident_capacity()
        else:
            self._restore_prompt_tail_resident_capacity()
        if not enabled:
            self._cancel_prompt_tail_prewarm("keepwarm-disabled")
        self._wake_engine_loop()

    def _ensure_prompt_tail_resident_capacity(self) -> None:
        """Give prompt-tail fallback two slots under the existing byte cap.

        One slot keeps the longer validated terminal state; the second can
        hold the guaranteed input-prompt prefix when chat-template rendering
        makes that terminal diverge on the next turn.  The configured byte
        ceiling remains authoritative, and an explicit zero-slot environment
        setting remains a hard disable.
        """

        config = getattr(self, "_keepwarm", None)
        config = getattr(config, "config", None)
        if not (
            config is not None
            and config.enabled
            and config.prompt_tail_prewarm_enabled
        ):
            return
        resident = getattr(getattr(self, "scheduler", None), "_exact_resident_cache", None)
        if resident is None or int(getattr(resident, "max_bytes", 0) or 0) <= 0:
            return
        explicit_slots = os.environ.get(
            "OMLX_EXACT_RESIDENT_MAX_ENTRIES",
            os.environ.get("OMLX_EXACT_RESIDENT_CACHE_SLOTS"),
        )
        if explicit_slots is not None:
            try:
                if int(explicit_slots) <= 0:
                    return
            except ValueError:
                return
        resident.max_entries = max(2, int(resident.max_entries or 0))

    def _restore_prompt_tail_resident_capacity(self) -> None:
        """Restore the resident-tier limit that preceded UI keepwarm."""

        resident = getattr(
            getattr(self, "scheduler", None),
            "_exact_resident_cache",
            None,
        )
        if resident is None:
            return
        baseline = max(
            0,
            int(getattr(self, "_prompt_tail_resident_baseline_entries", 0) or 0),
        )
        resize = getattr(resident, "resize", None)
        if callable(resize):
            resize(baseline)
        else:
            resident.max_entries = baseline
            if baseline == 0 and hasattr(resident, "clear"):
                resident.clear()

    def disarm_keepwarm_cache(self) -> None:
        """Stop idle touches after an explicit in-memory cache clear."""

        self._keepwarm.disarm_cache()
        self._cancel_prompt_tail_prewarm("cache-disarmed")

    def schedule_prompt_tail_prewarm(self, prompt: str | list[int]) -> bool:
        """Arm one latest-wins, idle-only prompt-tail prewarm candidate."""

        config = self.config.keepwarm_config
        if (
            self._closed
            or not self._running
            or not config.enabled
            or not config.prompt_tail_prewarm_enabled
            or not isinstance(prompt, (str, list))
        ):
            return False
        if (
            self._pending_admission_count()
            or self._scheduler_has_active_user_requests()
        ):
            with self._prompt_tail_lock:
                self._prompt_tail_stats["skipped"] += 1
                self._prompt_tail_stats["last_result"] = {
                    "status": "skipped",
                    "reason": "request-already-active",
                }
            return False
        value = list(prompt) if isinstance(prompt, list) else prompt
        due = time.monotonic() + config.prompt_tail_prewarm_delay_seconds
        with self._prompt_tail_lock:
            self._prompt_tail_epoch += 1
            self._prompt_tail_plan = (value, due, self._prompt_tail_epoch)
            self._prompt_tail_stats["scheduled"] += 1
        self._wake_engine_loop()
        return True

    def notify_admission_pending(self) -> None:
        """Cancel hidden maintenance at the public request boundary.

        Chat/VLM engines can perform template or media preparation on this
        same one-worker MLX lane before ``add_request`` is reached. Signalling
        here prevents that preparation from sitting behind an idle prompt-tail
        pass; the pass observes the epoch between bounded chunks and exits.
        """

        _invalidate_global_prompt_tail()
        self._cancel_prompt_tail_prewarm("external-admission")
        self._wake_engine_loop()

    def _cancel_prompt_tail_prewarm(self, reason: str) -> None:
        lock = getattr(self, "_prompt_tail_lock", None)
        if lock is None:
            return
        with lock:
            had_work = bool(
                self._prompt_tail_plan is not None
                or self._prompt_tail_inflight_epoch is not None
            )
            self._prompt_tail_epoch += 1
            self._prompt_tail_plan = None
            if had_work:
                self._prompt_tail_stats["cancelled"] += 1
                self._prompt_tail_stats["last_result"] = {
                    "status": "cancelled",
                    "reason": reason,
                }

    def _take_prompt_tail_prewarm(
        self,
        now: float,
    ) -> tuple[str | list[int], int] | None:
        with self._prompt_tail_lock:
            plan = self._prompt_tail_plan
            if plan is None or now < plan[1]:
                return None
            if (
                self._pending_admission_count()
                or self._scheduler_has_active_user_requests()
            ):
                self._prompt_tail_plan = None
                self._prompt_tail_epoch += 1
                self._prompt_tail_stats["skipped"] += 1
                self._prompt_tail_stats["last_result"] = {
                    "status": "skipped",
                    "reason": "request-became-active",
                }
                return None
            # Async store/deferred-clear work belongs to the completed source
            # request. Keep the latest plan queued until that housekeeping
            # drains; the durable prefix must exist before prewarm rebuilds
            # its non-block tail.
            if self.scheduler.has_requests():
                return None
            self._prompt_tail_plan = None
            self._prompt_tail_inflight_epoch = plan[2]
            return plan[0], plan[2]

    def _scheduler_has_active_user_requests(self) -> bool:
        active = getattr(self.scheduler, "has_active_user_requests", None)
        if callable(active):
            return bool(active())
        return bool(
            getattr(self.scheduler, "waiting", ())
            or getattr(self.scheduler, "prefilling", ())
            or getattr(self.scheduler, "running", {})
        )

    def _prompt_tail_abort_requested(self, epoch: int) -> bool:
        with self._prompt_tail_lock:
            invalidated = bool(
                epoch != self._prompt_tail_epoch
                or self._prompt_tail_inflight_epoch != epoch
            )
        return bool(
            invalidated
            or self._closed
            or not self._running
            or not self.config.keepwarm_config.enabled
            or not self.config.keepwarm_config.prompt_tail_prewarm_enabled
            or self._pending_admission_count()
            or self._scheduler_has_active_user_requests()
            or self.scheduler.has_requests()
        )

    def _run_prompt_tail_prewarm(
        self,
        prompt: str | list[int],
        epoch: int,
    ) -> dict[str, Any]:
        if not _prompt_tail_global_run_lock.acquire(blocking=False):
            return {"status": "skipped", "reason": "global-maintenance-busy"}
        try:
            config = self.config.keepwarm_config
            global_epoch = _global_prompt_tail_epoch()
            if (
                not config.enabled
                or not config.prompt_tail_prewarm_enabled
                or self._prompt_tail_abort_requested(epoch)
                or _other_prompt_tail_engine_busy(self)
            ):
                return {"status": "skipped", "reason": "engine-busy-or-disabled"}

            def abort_requested() -> bool:
                return bool(
                    self._prompt_tail_abort_requested(epoch)
                    or _global_prompt_tail_invalid(self, global_epoch)
                )

            return self.scheduler.prewarm_prompt_tail(
                prompt,
                min_tokens=config.prompt_tail_prewarm_min_tokens,
                max_tokens=config.prompt_tail_prewarm_max_tokens,
                max_suffix_tokens=config.prompt_tail_prewarm_max_suffix_tokens,
                chunk_size=config.prompt_tail_prewarm_chunk_size,
                abort_requested=abort_requested,
                publish_if_current=lambda tokens, cache, nbytes, durable: (
                    self._publish_prompt_tail_if_current(
                        epoch,
                        tokens,
                        cache,
                        nbytes,
                        durable,
                        global_epoch=global_epoch,
                    )
                ),
            )
        finally:
            _prompt_tail_global_run_lock.release()

    def _publish_prompt_tail_if_current(
        self,
        epoch: int,
        tokens: list[int],
        cache: list[Any],
        cache_nbytes: int,
        durable_tokens: int,
        *,
        global_epoch: int | None = None,
    ) -> bool:
        """Atomically validate the lifecycle epoch and replace resident L0."""

        with _prompt_tail_global_state_lock:
            if global_epoch is not None:
                if global_epoch != _prompt_tail_global_epoch:
                    logger.info(
                        "Latent prompt-tail publication rejected: global-epoch-changed"
                    )
                    return False
                if _other_prompt_tail_engine_busy(self):
                    logger.info(
                        "Latent prompt-tail publication rejected: other-engine-busy"
                    )
                    return False
            with self._prompt_tail_lock:
                reject_reason = None
                if epoch != self._prompt_tail_epoch:
                    reject_reason = "local-epoch-changed"
                elif self._prompt_tail_inflight_epoch != epoch:
                    reject_reason = "inflight-epoch-changed"
                elif self._closed or not self._running:
                    reject_reason = "engine-stopped"
                elif not self.config.keepwarm_config.enabled:
                    reject_reason = "keepwarm-disabled"
                elif not self.config.keepwarm_config.prompt_tail_prewarm_enabled:
                    reject_reason = "tail-disabled"
                elif self._pending_admission_count():
                    reject_reason = "admission-pending"
                elif self._scheduler_has_active_user_requests():
                    reject_reason = "user-request-active"
                elif self.scheduler.has_requests():
                    reject_reason = "scheduler-housekeeping"
                if reject_reason is not None:
                    logger.info(
                        "Latent prompt-tail publication rejected: %s",
                        reject_reason,
                    )
                    return False
                published = self.scheduler._exact_resident_cache.put(
                    tokens,
                    cache,
                    cache_nbytes=cache_nbytes,
                    durable_tokens=durable_tokens,
                    protect_longer_prefix=True,
                    terminal_proof=(
                        "mtp-target-only-stable-v1"
                        if self.scheduler._qwen35_mtp_target_only_prewarm_enabled()
                        else None
                    ),
                )
                if not published:
                    logger.info(
                        "Latent prompt-tail publication rejected: resident-policy"
                    )
                return published

    def _record_prompt_tail_result(self, epoch: int, result: dict[str, Any]) -> None:
        with self._prompt_tail_lock:
            if self._prompt_tail_inflight_epoch == epoch:
                self._prompt_tail_inflight_epoch = None
            status = result.get("status")
            if status == "published":
                self._prompt_tail_stats["published"] += 1
            elif status == "cancelled":
                self._prompt_tail_stats["cancelled"] += 1
            elif status == "skipped":
                self._prompt_tail_stats["skipped"] += 1
            else:
                self._prompt_tail_stats["failures"] += 1
            self._prompt_tail_stats["last_result"] = dict(result)
        logger.info(
            "Latent prompt-tail result: epoch=%d status=%s reason=%s "
            "prompt=%s cached=%s suffix=%s elapsed=%.1fms",
            epoch,
            result.get("status", "unknown"),
            result.get("reason", "unknown"),
            result.get("prompt_tokens", 0),
            result.get("cached_tokens", 0),
            result.get("suffix_tokens", 0),
            float(result.get("elapsed_ms", 0.0) or 0.0),
        )

    def _pending_admission_count(self) -> int:
        with self._pending_admissions_lock:
            return self._pending_admissions

    def _resident_cache_tokens(self) -> int:
        """Best-effort current prefix-cache size, read on the MLX lane."""

        cache_tokens = 0
        prefix_cache = getattr(self.scheduler, "block_aware_cache", None)
        paged_cache = getattr(prefix_cache, "paged_cache", None)
        try:
            stats = getattr(paged_cache, "stats", None)
            cache_tokens = max(
                cache_tokens,
                int(getattr(stats, "total_tokens_cached", 0) or 0),
            )
        except Exception:
            logger.debug("Unable to read cache size for keepwarm", exc_info=True)
        exact_stats = getattr(self.scheduler, "_exact_resident_stats", None)
        if callable(exact_stats):
            try:
                cache_tokens = max(
                    cache_tokens,
                    int(exact_stats().get("max_token_count", 0) or 0),
                )
            except Exception:
                logger.debug(
                    "Unable to read exact-resident size for keepwarm",
                    exc_info=True,
                )
        return max(0, cache_tokens)

    def _run_keepwarm_action(self, action: KeepwarmAction) -> bool:
        """Run one touch on this engine's existing serialized MLX lane."""

        if not self._keepwarm.should_execute(action):
            self._keepwarm.skip("disabled, closed, or request state changed")
            return False
        pending = self._pending_admission_count()
        scheduler_busy = self.scheduler.has_requests()
        if action.kind == "request_start" and (pending != 1 or scheduler_busy):
            self._keepwarm.observe_request_state(True)
            self._keepwarm.skip("request-start lost exclusive admission")
            return False
        if action.kind != "request_start" and (pending or scheduler_busy):
            self._keepwarm.observe_request_state(True)
            self._keepwarm.skip("admission or scheduler became busy")
            return False
        started = time.monotonic()
        try:
            compiled_touch = getattr(self, "_compiled_metal_keepwarm", None)
            if compiled_touch is None:
                # Focused embedders/tests may construct EngineCore via __new__.
                # Never reintroduce the allocation-heavy local matmul fallback.
                self._keepwarm.skip("local Metal pulse is unavailable")
                return False
            touch_result = compiled_touch.touch(action)
            if touch_result is None:
                # Request admission must not allocate, create a stream, or JIT.
                self._keepwarm.skip("request-start Metal pulse is not prepared")
                return False
            elapsed = touch_result.elapsed_seconds
        except Exception as exc:  # keep a model usable if experimental warming fails
            elapsed = max(0.0, time.monotonic() - started)
            self._keepwarm.record(
                action,
                elapsed_seconds=elapsed,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
            logger.warning("Local keepwarm %s failed: %s", action.kind, exc)
            return False
        self._keepwarm.record(
            action,
            elapsed_seconds=elapsed,
            ok=True,
            execution_mode=touch_result.execution_mode,
        )
        if elapsed >= self.config.keepwarm_config.slow_threshold_seconds:
            logger.warning(
                "Local keepwarm %s %s was slow (%.1f ms); backing off",
                action.kind,
                (
                    "async submission"
                    if touch_result.execution_mode == "async_submitted"
                    else "asynchronous preparation"
                ),
                elapsed * 1000.0,
            )
        else:
            logger.debug(
                "Local keepwarm %s %s in %.1f ms",
                action.kind,
                (
                    "submitted asynchronously"
                    if touch_result.execution_mode == "async_submitted"
                    else "prepared asynchronously"
                ),
                elapsed * 1000.0,
            )
        return True

    def _admit_request(self, request: Request) -> None:
        """Optionally warm after idle, then admit on the same MLX lane."""

        keepwarm = getattr(self, "_keepwarm", None)
        if keepwarm is None:
            self.scheduler.add_request(request)
            return
        pending = self._pending_admission_count()
        exclusive_idle_admission = pending == 1 and not self.scheduler.has_requests()
        if exclusive_idle_admission:
            action = keepwarm.request_start_action()
            if action is not None:
                self._run_keepwarm_action(action)
        else:
            if self.config.keepwarm_config.enabled:
                keepwarm.skip("concurrent admission or scheduler busy")
        try:
            self.scheduler.add_request(request)
        except BaseException:
            if exclusive_idle_admission:
                keepwarm.cancel_unstarted_request()
            raise
        keepwarm.observe_request_state(True)

    def _idle_keepwarm_if_due(self) -> None:
        """Re-check quiescence on the MLX lane and skip rather than queue."""

        if self._pending_admission_count() or self.scheduler.has_requests():
            self._keepwarm.observe_request_state(True)
            return
        cache_tokens = self._resident_cache_tokens()
        action = self._keepwarm.idle_action(cache_tokens=cache_tokens)
        if action is not None:
            self._run_keepwarm_action(action)

    async def _engine_loop(self) -> None:
        """Main engine loop - runs scheduler steps on the MLX executor.

        All scheduler steps run on _mlx_executor (single-worker thread) to
        guarantee that MLX GPU operations are never concurrent.  VLM vision
        encoding also runs on the same executor, so inline scheduler.step()
        on the event loop would race with vision mx.eval() and segfault.
        """
        loop = asyncio.get_running_loop()

        step_interval = self.config.step_interval
        stream_interval = self.config.stream_interval
        use_simple_streaming = stream_interval == 1

        while self._running:
            try:
                # Sweep collectors orphaned by client disconnects (throttled).
                now = time.monotonic()
                if now - self._last_reap >= 1.0:
                    self._last_reap = now
                    self._reap_orphaned_collectors(now)

                if self.scheduler.has_requests():
                    self._keepwarm.observe_request_state(True)
                    step_outputs = await loop.run_in_executor(
                        self._mlx_executor, self._step_burst
                    )
                    self._steps_executed += len(step_outputs)

                    # Distribute every step's outputs to collectors (one or
                    # more decode tokens per burst). collector.put() runs on the
                    # loop thread, keeping the asyncio.Event signalling
                    # thread-safe and per-token streaming intact.
                    collectors = self._output_collectors
                    states = self._stream_states
                    eviction_request = None
                    distributed = False

                    for output in step_outputs:
                        if (
                            eviction_request is None
                            and output.prefill_eviction_request is not None
                        ):
                            eviction_request = output.prefill_eviction_request

                        outputs = output.outputs
                        if not outputs:
                            continue
                        distributed = True

                        for req_output in outputs:
                            rid = req_output.request_id
                            collector = collectors.get(rid)

                            if collector is not None:
                                # Optimized: skip stream_interval check when interval=1
                                if use_simple_streaming:
                                    collector.put(req_output)
                                else:
                                    state = states.get(rid)
                                    if state and state.should_send(
                                        req_output.completion_tokens,
                                        req_output.finished,
                                    ):
                                        collector.put(req_output)
                                        state.mark_sent(req_output.completion_tokens)

                            if req_output.finished:
                                self._mark_request_finished(rid)
                                # Cleanup normally happens in the consumer
                                # (stream_outputs()/generate()); collectors left
                                # behind by a disconnected client are swept by
                                # _reap_orphaned_collectors() via _finished_at.

                    if distributed:
                        # Always yield to prevent event loop starvation.
                        # Without this, orphaned requests (client disconnected but
                        # request still in scheduler) block the entire event loop,
                        # making the server unresponsive to all HTTP requests.
                        await asyncio.sleep(0)

                    if eviction_request is not None:
                        callback = self.config.prefill_eviction_callback
                        if callback is not None:
                            logger.info(
                                "Running prefill LRU eviction for request %s",
                                eviction_request.request_id,
                            )
                            evicted = await callback(eviction_request)
                            if evicted:
                                logger.info(
                                    "Prefill LRU eviction completed for request %s",
                                    eviction_request.request_id,
                                )
                            else:
                                logger.info(
                                    "No idle model evicted for request %s; "
                                    "scheduler will fall back to throttling",
                                    eviction_request.request_id,
                                )
                        else:
                            logger.debug(
                                "Prefill eviction requested for %s but no callback "
                                "is configured",
                                eviction_request.request_id,
                            )
                        continue
                    if not step_outputs[-1].has_work:
                        # Requests may be queued while scheduler admission is
                        # intentionally throttled by async cache cleanup. Avoid
                        # spinning the engine loop, but still let new requests
                        # wake the wait immediately.
                        event = self._wake_event
                        if event is None:
                            await asyncio.sleep(step_interval)
                        else:
                            event.clear()
                            with suppress(TimeoutError):
                                await asyncio.wait_for(
                                    event.wait(), timeout=step_interval
                                )
                else:
                    self._keepwarm.observe_request_state(False)
                    prompt_tail = self._take_prompt_tail_prewarm(now)
                    if prompt_tail is not None:
                        prompt, prompt_epoch = prompt_tail
                        prompt_tail_result = await loop.run_in_executor(
                            self._mlx_executor,
                            self._run_prompt_tail_prewarm,
                            prompt,
                            prompt_epoch,
                        )
                        self._record_prompt_tail_result(
                            prompt_epoch,
                            prompt_tail_result,
                        )
                        continue
                    if (
                        self.config.keepwarm_config.enabled
                        and now >= self._next_keepwarm_check_at
                    ):
                        # The executor helper performs the decisive busy check.
                        # New admissions increment their counter before queuing
                        # scheduler insertion, so a racing touch skips.
                        self._next_keepwarm_check_at = now + 0.25
                        await loop.run_in_executor(
                            self._mlx_executor,
                            self._idle_keepwarm_if_due,
                        )
                    event = self._wake_event
                    if event is None:
                        await asyncio.sleep(step_interval)
                    else:
                        event.clear()
                        # Avoid losing a request that arrived between
                        # has_requests() and clear().
                        if self.scheduler.has_requests():
                            continue
                        with suppress(TimeoutError):
                            await asyncio.wait_for(event.wait(), timeout=step_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                import traceback

                logger.error(f"Engine loop error: {e}\n{traceback.format_exc()}")
                # Fail all requests and remove from scheduler to prevent
                # infinite loop (has_requests() must go False; a pending
                # idle reclaim may hold it True for one extra step, which
                # drains and clears it).
                failed_ids = await loop.run_in_executor(
                    self._mlx_executor, self.scheduler.fail_all_requests
                )
                for rid in failed_ids:
                    collector = self._output_collectors.get(rid)
                    if collector is not None:
                        collector.put(
                            RequestOutput(
                                request_id=rid,
                                finished=True,
                                finish_reason="error",
                                error=str(e),
                            )
                        )
                    self._mark_request_finished(rid)
                await asyncio.sleep(0.1)

    async def add_request(
        self,
        prompt: Union[str, List[int]],
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        images: Optional[List[Any]] = None,
        videos: Optional[List[Any]] = None,
        vlm_inputs_embeds: Optional[Any] = None,
        vlm_extra_kwargs: Optional[Dict[str, Any]] = None,
        vlm_image_hash: Optional[str] = None,
        vlm_cache_key_start: int = 0,
        vlm_cache_key_ranges: Optional[List[Tuple[int, str]]] = None,
        specprefill: Optional[bool] = None,
        specprefill_keep_pct: Optional[float] = None,
        specprefill_threshold: Optional[int] = None,
        specprefill_system_end: Optional[int] = None,
        skip_cache_store: bool = False,
        benchmark_trace: bool = False,
        benchmark_ane_sequence_length: int = 0,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """
        Add a request for processing.

        Args:
            prompt: Input prompt (string or token IDs)
            sampling_params: Generation parameters
            request_id: Optional custom request ID
            images: Optional images for multimodal
            videos: Optional videos for multimodal
            vlm_inputs_embeds: Pre-computed vision+text embeddings for VLM
            vlm_extra_kwargs: Model-specific VLM kwargs (e.g., position_ids)
            vlm_image_hash: SHA256 hash of images for prefix cache
            specprefill: Per-request SpecPrefill override (True/False/None)
            specprefill_keep_pct: Per-request keep rate override
            specprefill_threshold: Per-request threshold override (min tokens)

        Returns:
            The request ID
        """
        _invalidate_global_prompt_tail()
        self._cancel_prompt_tail_prewarm("new-admission")
        if request_id is None:
            request_id = str(uuid.uuid4())

        if sampling_params is None:
            sampling_params = SamplingParams()

        request = Request(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            tools=tools,
            images=images,
            videos=videos,
            vlm_inputs_embeds=vlm_inputs_embeds,
            vlm_extra_kwargs=vlm_extra_kwargs,
            vlm_image_hash=vlm_image_hash,
            vlm_cache_key_start=vlm_cache_key_start,
            vlm_cache_key_ranges=vlm_cache_key_ranges,
            skip_cache_store=skip_cache_store,
            benchmark_trace=benchmark_trace,
            benchmark_ane_sequence_length=benchmark_ane_sequence_length,
        )

        # SpecPrefill: resolve per-request settings.
        # The scheduler checks _specprefill_enabled to decide whether to score.
        if specprefill is not None:
            request._specprefill_enabled = specprefill
        elif self.scheduler._specprefill_draft_model is not None:
            # Draft model is loaded → enable by default
            request._specprefill_enabled = True
        if specprefill_keep_pct is not None:
            request._specprefill_keep_pct = specprefill_keep_pct
        if specprefill_threshold is not None:
            request._specprefill_threshold = specprefill_threshold
        if specprefill_system_end is not None and specprefill_system_end > 0:
            request.specprefill_system_end = specprefill_system_end

        # Setup output collector with stream_interval from config
        self._output_collectors[request_id] = RequestOutputCollector(aggregate=True)
        self._stream_states[request_id] = RequestStreamState(
            stream_interval=self.config.stream_interval
        )
        self._finished_events[request_id] = asyncio.Event()

        # Add to scheduler — route through the MLX executor so that
        # prefix cache reconstruction (mx.load, mx.concatenate) never
        # races with scheduler.step() on the Metal stream.  See #95.
        #
        # The scheduler may raise (PrefillMemoryExceededError, or other
        # validation errors) before the request enters self.waiting. In
        # that case the consumer in stream_outputs / generate never sees
        # the request_id and its finally-block cleanup never fires —
        # without the explicit cleanup below the per-rejection leak
        # accumulates one collector + one stream_state + one
        # asyncio.Event per refused request. Re-raise after cleanup so
        # the typed exception still reaches the FastAPI 400 handler.
        loop = asyncio.get_running_loop()
        # Focused embedders/tests may construct EngineCore via __new__.
        if not hasattr(self, "_pending_admissions_lock"):
            self._pending_admissions_lock = threading.Lock()
            self._pending_admissions = 0
        with self._pending_admissions_lock:
            self._pending_admissions += 1
        try:
            await loop.run_in_executor(self._mlx_executor, self._admit_request, request)
        except BaseException:
            # If the caller is cancelled here (e.g. the client disconnected
            # before the SSE stream began) — or the insert fails — the request
            # never reaches stream_outputs()/generate()'s try/finally, so
            # nothing would mark it finished or clean it up. The collector
            # created above would then linger forever as a phantom the reaper
            # cannot see (it was never stamped _finished_at), and the dashboard
            # would show it as "Generating" indefinitely (#1154).
            # Drop the tracking and abort any partial scheduler insert (the
            # deferred abort is idempotent and harmless if it never landed).
            try:
                self.scheduler.abort_request(request_id)
            except Exception as abort_exc:  # noqa: BLE001
                logger.debug(
                    f"Abort of partial insert for {request_id} failed: {abort_exc}"
                )
            self._cleanup_request(request_id)
            raise
        finally:
            with self._pending_admissions_lock:
                self._pending_admissions = max(0, self._pending_admissions - 1)
        self._wake_engine_loop()

        return request_id

    async def abort_request(self, request_id: str) -> bool:
        """Abort a request.

        Uses deferred abort pattern: scheduler.abort_request() just enqueues
        the request ID into a thread-safe set. The actual abort is processed
        at the start of the next scheduler.step() call, ensuring it runs in
        the same execution context as generation (no race conditions).

        Signals the consumer (stream_outputs/generate) with an abort error
        so it can exit gracefully. Cleanup is handled by the consumer's
        finally block, NOT here -- calling _cleanup_request() immediately
        after put() would clear the output before the consumer can drain it.
        """
        scheduler = getattr(self, "scheduler", None)
        if getattr(self, "_closed", False) or scheduler is None:
            logger.debug(
                "Skipping abort for request %s because engine is already closed",
                request_id,
            )
            return False

        result = scheduler.abort_request(request_id)

        # Signal consumer with abort error so any waiting
        # stream_outputs() / generate() can exit gracefully.
        # Matches abort_all_requests() pattern.
        collector = self._output_collectors.get(request_id)
        if collector is not None:
            collector.put(
                RequestOutput(
                    request_id=request_id,
                    finished=True,
                    finish_reason="abort",
                    error="Request aborted",
                )
            )
        self._mark_request_finished(request_id)
        self._wake_engine_loop()

        return result

    async def abort_all_requests(
        self,
        *,
        reason: str | None = None,
        error_code: str | None = None,
    ) -> int:
        """Abort all active requests without stopping the engine.

        Sends error output to all active collectors and marks requests
        for deferred abort in the scheduler. Cleanup is handled by
        the consumer (stream_outputs/generate). Without an explicit reason,
        preserve the memory-pressure error used by the process enforcer.
        """
        from .utils.proc_memory import get_phys_footprint

        # Only abort requests not already finished or pending abort. The
        # enforcer calls this once per poll (1s) while pressure persists; a
        # deferred abort can take many seconds to drain (it executes at the
        # next step()/chunk boundary), so without this filter the same
        # request is re-aborted every poll: duplicate [Error: ...] frames
        # reach streaming clients and the returned count reports stale
        # aborts as fresh progress, hiding a stuck scheduler from the
        # enforcer's escalation logic.
        pending_aborts: set[str] = set()
        sched_for_filter = self.scheduler
        if sched_for_filter is not None:
            pending_aborts = set(
                getattr(sched_for_filter, "_pending_abort_ids", None) or ()
            )
        request_ids = [
            rid
            for rid in self._output_collectors
            if rid not in self._finished_at and rid not in pending_aborts
        ]
        ceiling = 0
        watermark = 0
        sched = self.scheduler
        if reason is None and sched is not None:
            ceiling = int(getattr(sched, "_memory_hard_limit_bytes", 0) or 0)
            watermark = int(getattr(sched, "_memory_hard_watermark_bytes", 0) or 0)
        usage = get_phys_footprint() if reason is None else 0
        usage_gb = usage / (1024**3)
        ceiling_gb = ceiling / (1024**3) if ceiling > 0 else 0.0
        watermark_gb = watermark / (1024**3) if watermark > 0 else 0.0
        # Name the component ceiling that actually produced this abort. A
        # generic "loosen memory_guard_tier" leaves users who are already on
        # `aggressive` with nothing to try (#2362); the ladder points at the
        # constraint that is really binding, or falls back to the same
        # generic advice when the enforcer has not propagated a breakdown.
        binding = "effective"
        advice = ""
        if reason is None:
            binding, advice = describe_ceiling_binding(
                static=int(getattr(sched, "_memory_static_ceiling_bytes", 0) or 0),
                dynamic=int(getattr(sched, "_memory_dynamic_ceiling_bytes", 0) or 0),
                metal_cap=int(getattr(sched, "_memory_metal_cap_bytes", 0) or 0),
                tier=str(getattr(sched, "_memory_guard_tier", "") or ""),
                current=usage,
                fmt=format_bytes,
                tail="reduce context length",
            )
            advice = f"{advice}."
        ceiling_label = f"{binding} ceiling" if binding != "effective" else "ceiling"
        for rid in request_ids:
            self.scheduler.abort_request(rid)
            collector = self._output_collectors.get(rid)
            if collector is not None:
                # Name the watermark that actually tripped; printing only the
                # ceiling reads as "usage below limit yet aborted" (#2321).
                if reason is not None:
                    error_msg = reason
                elif watermark > 0 and ceiling > 0:
                    error_msg = (
                        f"Request aborted: process memory limit exceeded "
                        f"(usage {usage_gb:.1f} GB, abort threshold "
                        f"(hard watermark) {watermark_gb:.1f} GB, "
                        f"{ceiling_label} {ceiling_gb:.1f} GB). "
                        f"{advice}"
                    )
                elif ceiling > 0:
                    error_msg = (
                        f"Request aborted: process memory limit exceeded "
                        f"(usage {usage_gb:.1f} GB, "
                        f"{ceiling_label} {ceiling_gb:.1f} GB). "
                        f"{advice}"
                    )
                else:
                    error_msg = (
                        f"Request aborted: process memory limit exceeded "
                        f"(usage {usage_gb:.1f} GB). "
                        f"{advice}"
                    )
                collector.put(
                    RequestOutput(
                        request_id=rid,
                        finished=True,
                        finish_reason="error",
                        new_text=f"\n\n[Error: {error_msg}]",
                        error=error_msg,
                        # Without a code this surfaced as a bare RuntimeError:
                        # the JSON keepalive wrapper never saw a memory error,
                        # so the response generator died mid-body and the
                        # client got a truncated read plus a 500 traceback
                        # instead of the same actionable 400 the pre-flight
                        # guard returns.
                        error_code=error_code or "prefill_memory_aborted",
                        error_metadata={
                            "request_id": rid,
                            # Only the limit is reported: the exception's
                            # estimated_bytes means "predicted peak", and this
                            # abort fires on measured usage, which the message
                            # already states.
                            "limit_bytes": (
                                watermark or ceiling or None
                                if reason is None
                                else None
                            ),
                        },
                    )
                )
            self._mark_request_finished(rid)
        if request_ids:
            if reason is None:
                logger.warning(
                    f"Aborted {len(request_ids)} requests due to memory pressure"
                )
            else:
                logger.warning("Aborted %d requests: %s", len(request_ids), reason)
            self._wake_engine_loop()
        return len(request_ids)

    def _mark_request_finished(self, request_id: str) -> None:
        """Signal the consumer a request finished and stamp the finish time.

        The timestamp lets _reap_orphaned_collectors() drop collectors whose
        consumer never cleaned up (e.g. the client disconnected mid-stream and
        the SSE generator chain was abandoned rather than closed).
        """
        self._finished_at.setdefault(request_id, time.monotonic())
        event = self._finished_events.get(request_id)
        if event is not None:
            event.set()

    def _reap_orphaned_collectors(self, now: float, grace: float = 5.0) -> int:
        """Drop tracking for finished requests whose consumer never cleaned up.

        stream_outputs()/generate() normally remove their own collector via
        _cleanup_request() once the final output is drained. But when a client
        disconnects mid-stream the SSE generator chain is abandoned instead of
        closed, so that finally block only runs at non-deterministic GC time —
        the collector lingers in _output_collectors and the request shows as
        "Generating" forever on the dashboard (#1154).

        This sweep removes the dict tracking for any request finished more than
        ``grace`` seconds ago. It is intentionally pop-only and never calls
        ``collector.clear()``: stream_outputs()/generate() hold their own
        reference to the collector object, so dropping the dict entry cannot
        truncate a slow-but-live consumer's output — only an over-eager clear()
        could (which is what made the earlier _delayed_cleanup() approach race).
        The grace period guarantees a live consumer (which drains in the same
        event-loop turn the request finishes) has already self-cleaned.
        """
        if not self._finished_at:
            return 0
        stale = [rid for rid, ts in self._finished_at.items() if now - ts >= grace]
        for rid in stale:
            # pop-only — see docstring; never clear() the collector object.
            self._output_collectors.pop(rid, None)
            self._stream_states.pop(rid, None)
            self._finished_events.pop(rid, None)
            self._finished_at.pop(rid, None)
        if stale:
            logger.debug(
                "Reaped %d orphaned output collector(s) after disconnect: %s",
                len(stale),
                stale,
            )
        return len(stale)

    def _cleanup_request(self, request_id: str) -> None:
        """Clean up request tracking.

        Only cleans engine-core level state (collectors, events).
        Scheduler state is cleaned by _do_abort_request (deferred abort)
        or _cleanup_finished (normal completion).
        """
        collector = self._output_collectors.pop(request_id, None)
        if collector:
            collector.clear()
        self._stream_states.pop(request_id, None)
        self._finished_events.pop(request_id, None)
        self._finished_at.pop(request_id, None)

    async def _delayed_cleanup(self, request_id: str, delay: float = 5.0) -> None:
        """
        Cleanup request after delay if not already cleaned.

        This handles the case where a client disconnects before consuming
        the stream_outputs() generator, which would prevent the finally
        block from running.
        """
        await asyncio.sleep(delay)
        if request_id in self._output_collectors:
            logger.debug(f"Delayed cleanup for request {request_id}")
            self._cleanup_request(request_id)

    async def stream_outputs(
        self,
        request_id: str,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[RequestOutput]:
        """
        Stream outputs for a request with low-latency non-blocking pattern.

        Uses the vLLM pattern: get_nowait() or await get()
        This avoids unnecessary task switches when output is available.

        Args:
            request_id: The request ID
            timeout: Optional timeout in seconds

        Yields:
            RequestOutput objects as tokens are generated
        """
        collector = self._output_collectors.get(request_id)
        if collector is None:
            # Request might not be added yet or already cleaned up
            return

        try:
            while True:
                try:
                    # Non-blocking drain pattern from vLLM
                    # Try get_nowait first to avoid task switch if output ready
                    if timeout:
                        output = collector.get_nowait()
                        if output is None:
                            output = await asyncio.wait_for(
                                collector.get(), timeout=timeout
                            )
                    else:
                        output = collector.get_nowait() or await collector.get()

                    yield output

                    if output.error:
                        _raise_request_output_error(output)

                    if output.finished:
                        break

                except asyncio.TimeoutError:
                    logger.warning(f"Timeout waiting for request {request_id}")
                    break

        finally:
            self._cleanup_request(request_id)

    async def generate(
        self,
        prompt: Union[str, List[int]],
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> RequestOutput:
        """
        Generate a complete response (non-streaming).

        This method is optimized to avoid streaming overhead when
        you only need the final result.

        Args:
            prompt: Input prompt
            sampling_params: Generation parameters
            request_id: Optional request ID

        Returns:
            Final RequestOutput with complete text
        """
        request_id = await self.add_request(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            **kwargs,
        )

        # Wait for completion using event instead of streaming
        # This avoids the waiting_consumer tracking overhead
        event = self._finished_events.get(request_id)
        if event is None:
            raise RuntimeError(f"No event for request {request_id}")

        # Capture the collector reference BEFORE awaiting, mirroring
        # stream_outputs(): the orphan reaper is pop-only and may drop the dict
        # entry once the request is finished, but a held reference still drains.
        # Re-fetching after the await would race the reaper if this coroutine is
        # starved past the grace window under heavy load (#1154).
        collector = self._output_collectors.get(request_id)
        if collector is None:
            raise RuntimeError(f"No collector for request {request_id}")

        try:
            # Wait for the request to finish
            await event.wait()
        except asyncio.CancelledError:
            # Client disconnected or task was cancelled - abort the request
            # to free scheduler/GPU resources (prevents orphaned requests)
            logger.info(f"Request {request_id} cancelled, aborting")
            await self.abort_request(request_id)
            self._cleanup_request(request_id)
            raise

        # Drain all outputs and get the last one (using the captured reference)
        final_output = None
        first_token_at = None
        while True:
            output = collector.get_nowait()
            if output is None:
                break
            if first_token_at is None and output.generated_at is not None:
                first_token_at = output.generated_at
            final_output = output

        # Cleanup
        self._cleanup_request(request_id)

        if final_output is None:
            raise RuntimeError(f"No output for request {request_id}")

        if final_output.error:
            _raise_request_output_error(final_output)

        final_output.first_token_at = first_token_at
        return final_output

    def generate_batch_sync(
        self,
        prompts: List[Union[str, List[int]]],
        sampling_params: Optional[SamplingParams] = None,
    ) -> List[RequestOutput]:
        """
        Generate responses synchronously for maximum throughput.

        This bypasses the async engine loop entirely, running the scheduler
        directly for optimal batching performance. Use this when you don't
        need streaming and want maximum throughput.

        Args:
            prompts: List of input prompts
            sampling_params: Generation parameters (same for all)

        Returns:
            List of RequestOutput in same order as prompts
        """
        import uuid as uuid_module

        from .request import Request

        if sampling_params is None:
            sampling_params = SamplingParams()

        # Add all requests to scheduler
        request_ids = []
        for prompt in prompts:
            request_id = str(uuid_module.uuid4())
            request = Request(
                request_id=request_id,
                prompt=prompt,
                sampling_params=sampling_params,
            )
            self.scheduler.add_request(request)
            request_ids.append(request_id)

        # Process until all done - direct scheduler access, no async overhead
        results: Dict[str, RequestOutput] = {}
        while self.scheduler.has_requests():
            output = self.scheduler.step()
            for req_output in output.outputs:
                if req_output.finished:
                    results[req_output.request_id] = req_output

        # Cleanup
        for rid in request_ids:
            self.scheduler.remove_finished_request(rid)

        # Return in original order
        return [results[rid] for rid in request_ids]

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        scheduler_stats = self.scheduler.get_stats()
        uptime = time.time() - self._start_time if self._start_time else 0

        return {
            "running": self._running,
            "uptime_seconds": uptime,
            "steps_executed": self._steps_executed,
            "active_requests": len(self._output_collectors),
            "stream_interval": self.config.stream_interval,
            "keepwarm": (
                self._keepwarm.snapshot()
                if getattr(self, "_keepwarm", None) is not None
                else None
            ),
            "prompt_tail_prewarm": dict(
                getattr(self, "_prompt_tail_stats", {})
            ),
            **scheduler_stats,
        }

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get prefix cache statistics."""
        return self.scheduler.get_cache_stats()

    def _release_model(self) -> None:
        """Release model ownership."""
        if self._owns_model and not self._closed:
            registry = get_registry()
            registry.release(self.model, self._engine_id)
            self._owns_model = False
            logger.debug(f"Engine {self._engine_id} released model ownership")

    def close(self) -> None:
        """
        Explicitly close the engine and release resources.

        This should be called when done using the engine, especially
        if you plan to create another engine with the same model.
        """
        if self._closed:
            return

        # Preserve only the registry's non-owning integer key. Holding a strong
        # model reference through final executor-thread reclaim would keep all
        # weights alive and defeat that reclaim.
        owned_model_id = id(self.model) if self._owns_model else None
        self._cancel_prompt_tail_prewarm("engine-close")
        _unregister_prompt_tail_engine(self)

        keepwarm = getattr(self, "_keepwarm", None)
        if keepwarm is not None:
            keepwarm.shutdown()

        compiled_touch = getattr(self, "_compiled_metal_keepwarm", None)

        def shutdown_after_keepwarm_close() -> None:
            # Cancellation cannot interrupt executor work that already began.
            # FIFO this close behind any in-flight pulse, drain only its
            # dedicated stream, then continue scheduler/model teardown.
            try:
                if compiled_touch is not None:
                    compiled_touch.close()
            finally:
                self.scheduler.shutdown()

        self._closed = True

        # Both shutdown() and deep_reset() touch the engine stream (directly
        # or via _drain_pending_async_removes / _do_abort_request). The
        # stream is bound to the engine's executor thread, so dispatch both
        # through the executor; fall back to a direct call if the executor
        # is already shut down.
        for fn in (shutdown_after_keepwarm_close, self.scheduler.deep_reset):
            fn_name = getattr(fn, "__name__", repr(fn))
            try:
                self._mlx_executor.submit(fn).result(timeout=FATAL_TEARDOWN_TIMEOUT_S)
            except concurrent.futures.TimeoutError:
                fatal_exit(
                    f"Engine teardown timed out after "
                    f"{FATAL_TEARDOWN_TIMEOUT_S:.0f}s while running "
                    f"{fn_name} for engine {self._engine_id}"
                )
            except RuntimeError:
                try:
                    fn()
                except RuntimeError:
                    pass
                except Exception:
                    logger.warning(
                        "Engine %s: %s raised during close() fallback",
                        self._engine_id,
                        getattr(fn, "__name__", fn),
                        exc_info=True,
                    )
            except Exception:
                # A failing shutdown/deep_reset must not abort close(), or the
                # SSD cache manager below stays open and its writer thread keeps
                # the manager (and its hot cache) alive until restart.
                logger.warning(
                    "Engine %s: %s raised during close()",
                    self._engine_id,
                    getattr(fn, "__name__", fn),
                    exc_info=True,
                )

        # Drop the last bound-method reference from the teardown loop before
        # the final GC/reclaim pass below.
        fn = None
        shutdown_after_keepwarm_close = None
        compiled_touch = None
        self._compiled_metal_keepwarm = None

        # Guarantee the SSD cache manager is released even if shutdown() did not
        # reach its own close() above. The manager's writer thread holds a strong
        # reference to it, so an unclosed manager leaks until restart.
        manager = getattr(self.scheduler, "paged_ssd_cache_manager", None)
        if manager is not None:
            try:
                manager.close()
            except Exception:
                logger.warning(
                    "Engine %s: SSD cache manager close() failed during teardown",
                    self._engine_id,
                    exc_info=True,
                )
            self.scheduler.paged_ssd_cache_manager = None
        manager = None

        # Clear output collectors before dropping model/scheduler references so
        # any request-side caches they retain are eligible for the final reclaim.
        for collector in self._output_collectors.values():
            collector.clear()
        self._output_collectors.clear()
        self._stream_states.clear()
        self._finished_events.clear()
        self._finished_at.clear()

        release_model_resources = getattr(self.model, "release_resources", None)
        if callable(release_model_resources):
            try:
                release_model_resources()
            except Exception:
                logger.warning(
                    "Engine %s: model resource release failed during close()",
                    self._engine_id,
                    exc_info=True,
                )
        release_model_resources = None

        # Release model, tokenizer, and scheduler references before the final
        # MLX reclaim. The reclaim must run on this engine's worker thread and
        # stream; clearing on the global executor cannot reliably return this
        # thread/stream-local Metal memory to MLX.
        self.model = None
        self.tokenizer = None
        self.scheduler = None

        if self._mlx_executor is not None:
            try:
                reclaim_future = self._mlx_executor.submit(
                    _final_engine_thread_reclaim, self._mlx_stream
                )
            except RuntimeError:
                reclaim_future = None

            if reclaim_future is not None:
                try:
                    reclaim_future.result(timeout=FATAL_TEARDOWN_TIMEOUT_S)
                except concurrent.futures.TimeoutError:
                    fatal_exit(
                        f"Engine teardown timed out after "
                        f"{FATAL_TEARDOWN_TIMEOUT_S:.0f}s while reclaiming "
                        f"MLX memory for engine {self._engine_id}"
                    )
                except Exception as exc:
                    fatal_exit(
                        f"Engine {self._engine_id}: final MLX reclaim failed: "
                        f"{exc!r}"
                    )

            # MLX 0.32.2 makes compiled AttachedData destruction GIL-safe, and
            # _final_engine_thread_reclaim clears the worker's per-thread stream
            # registry before ThreadPoolExecutor lets that worker exit.
            self._mlx_executor.shutdown(wait=False)
            self._mlx_executor = None

        # This is the model-ownership barrier, not merely the request-worker
        # barrier above.  Keep the lease while release_resources(), reference
        # detachment, final Metal reclaim, and compile-cache retirement can
        # still touch the old graph.  A replacement engine may acquire this
        # object only after every old-engine operation is complete.
        if self._owns_model and owned_model_id is not None:
            registry = get_registry()
            registry.release_by_id(owned_model_id, self._engine_id)
            self._owns_model = False
            logger.debug(f"Engine {self._engine_id} released model ownership")
        owned_model_id = None

        logger.debug(f"Engine {self._engine_id} closed")

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self._release_model()
        except Exception:
            # Ignore errors during garbage collection
            pass

    @property
    def engine_id(self) -> str:
        """Get the engine ID."""
        return self._engine_id


class AsyncEngineCore:
    """
    Async context manager wrapper for EngineCore.

    Usage:
        async with AsyncEngineCore(model, tokenizer) as engine:
            request_id = await engine.add_request("Hello")
            async for output in engine.stream_outputs(request_id):
                print(output.new_text)
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[EngineConfig] = None,
    ):
        self.engine = EngineCore(model, tokenizer, config)
        # Drop wrapper-local aliases after EngineCore takes ownership.
        model = None
        tokenizer = None

    @property
    def _mlx_executor(self):
        """Expose the MLX executor for VLM vision encoding."""
        return self.engine._mlx_executor

    async def __aenter__(self) -> "AsyncEngineCore":
        await self.engine.start()
        return self

    async def __aexit__(self, *args) -> None:
        await self.stop()

    def start(self) -> None:
        """Start engine (creates task in current loop)."""
        asyncio.create_task(self.engine.start())

    async def stop(self) -> None:
        """Stop the engine."""
        engine = getattr(self, "engine", None)
        if engine is None:
            return
        await engine.stop()

    async def add_request(
        self,
        prompt: Union[str, List[int]],
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Add a request."""
        return await self.engine.add_request(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            **kwargs,
        )

    async def abort_request(self, request_id: str) -> bool:
        """Abort a request."""
        engine = getattr(self, "engine", None)
        if engine is None:
            logger.debug(
                "Skipping abort for request %s because async engine is closed",
                request_id,
            )
            return False
        return await engine.abort_request(request_id)

    def schedule_prompt_tail_prewarm(self, prompt: str | list[int]) -> bool:
        """Arm an idle-only exact prompt-tail prewarm on the owning engine."""

        engine = getattr(self, "engine", None)
        if engine is None:
            return False
        return engine.schedule_prompt_tail_prewarm(prompt)

    def notify_admission_pending(self) -> None:
        """Tell the owning core that real request preparation has begun."""

        engine = getattr(self, "engine", None)
        if engine is not None:
            engine.notify_admission_pending()

    async def abort_all_requests(
        self,
        *,
        reason: str | None = None,
        error_code: str | None = None,
    ) -> int:
        """Abort all active requests without stopping the engine."""
        engine = getattr(self, "engine", None)
        if engine is None:
            return 0
        return await engine.abort_all_requests(reason=reason, error_code=error_code)

    async def stream_outputs(
        self,
        request_id: str,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[RequestOutput]:
        """Stream outputs."""
        async for output in self.engine.stream_outputs(request_id, timeout):
            yield output

    async def generate(
        self,
        prompt: Union[str, List[int]],
        sampling_params: Optional[SamplingParams] = None,
        **kwargs,
    ) -> RequestOutput:
        """Generate complete response."""
        return await self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            **kwargs,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get engine stats."""
        return self.engine.get_stats()

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get prefix cache statistics."""
        return self.engine.get_cache_stats()
