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
import logging
import os
import time
import uuid
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
    Set,
    Tuple,
    Union,
)

import mlx.core as mx

from .request import Request, RequestOutput, RequestStatus, SamplingParams
from .scheduler import Scheduler, SchedulerConfig, SchedulerOutput
from .output_collector import RequestOutputCollector, RequestStreamState
from .model_registry import get_registry, ModelOwnershipError
from .utils.compile_cache import (
    clear_thread_compile_cache,
    compile_cache_clear_available,
)

logger = logging.getLogger(__name__)

_global_mlx_executor: concurrent.futures.ThreadPoolExecutor | None = None

# Fallback only: used when the MLX compile-cache clear symbol is unavailable
# (see omlx/utils/compile_cache.py). In that case a per-engine MLX worker
# thread cannot exit safely (its thread_local ~CompilerCache would free
# @mx.compile graphs' Python objects without the GIL -> crash), so close()
# keeps the executor + stream alive here for the process lifetime instead of
# shutting the thread down. With the clear symbol present (the normal path)
# these stay empty and the worker threads shut down normally.
_immortal_mlx_executors: list = []
_immortal_mlx_streams: list = []


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


@dataclass
class EngineConfig:
    """Configuration for the engine."""

    model_name: str = ""
    scheduler_config: Optional[SchedulerConfig] = None
    step_interval: float = 0.05  # Idle wait timeout; requests wake the loop
    stream_interval: int = 1  # Tokens to batch before streaming (1=every token)
    prefill_eviction_callback: Optional[Callable[[Any], Awaitable[bool]]] = None
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
        force_model_ownership: bool = True,
    ):
        """
        Initialize the engine.

        Args:
            model: The MLX model
            tokenizer: The tokenizer
            config: Engine configuration
            engine_id: Optional unique ID for this engine (auto-generated if None)
            force_model_ownership: If True (default), forcibly take model ownership
                                   from any existing engine. If False, raises
                                   ModelOwnershipError if model is in use.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or EngineConfig()
        self._engine_id = engine_id or str(uuid.uuid4())
        self._owns_model = False
        self._closed = False

        # Acquire model ownership
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

        # Engine state
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._wake_event: Optional[asyncio.Event] = None
        self._start_time: Optional[float] = None
        self._steps_executed = 0

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
                if self.scheduler.has_requests():
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
                    events = self._finished_events
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
                                event = events.get(rid)
                                if event:
                                    event.set()
                                # Note: cleanup is handled by stream_outputs() finally block
                                # _delayed_cleanup() was causing double cleanup race condition

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
                            await asyncio.wait_for(
                                event.wait(), timeout=step_interval
                            )

            except asyncio.CancelledError:
                break
            except Exception as e:
                import traceback

                logger.error(f"Engine loop error: {e}\n{traceback.format_exc()}")
                # Fail all requests and remove from scheduler to prevent
                # infinite loop (has_requests() must return False).
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
                    event = self._finished_events.get(rid)
                    if event:
                        event.set()
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
        if request_id is None:
            request_id = str(uuid.uuid4())

        if sampling_params is None:
            sampling_params = SamplingParams()

        request = Request(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            images=images,
            videos=videos,
            vlm_inputs_embeds=vlm_inputs_embeds,
            vlm_extra_kwargs=vlm_extra_kwargs,
            vlm_image_hash=vlm_image_hash,
            vlm_cache_key_start=vlm_cache_key_start,
            vlm_cache_key_ranges=vlm_cache_key_ranges,
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
        # the typed exception still reaches the FastAPI 413 handler.
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                self._mlx_executor, self.scheduler.add_request, request
            )
        except BaseException:
            self._cleanup_request(request_id)
            raise
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
        event = self._finished_events.get(request_id)
        if event is not None:
            event.set()
        self._wake_engine_loop()

        return result

    async def abort_all_requests(self) -> int:
        """Abort all active requests without stopping the engine.

        Sends error output to all active collectors and marks requests
        for deferred abort in the scheduler. Cleanup is handled by
        the consumer (stream_outputs/generate).
        """
        from .utils.proc_memory import get_phys_footprint

        request_ids = list(self._output_collectors.keys())
        ceiling = 0
        sched = self.scheduler
        if sched is not None:
            ceiling = int(getattr(sched, "_memory_hard_limit_bytes", 0) or 0)
        usage = get_phys_footprint()
        usage_gb = usage / (1024**3)
        ceiling_gb = ceiling / (1024**3) if ceiling > 0 else 0.0
        for rid in request_ids:
            self.scheduler.abort_request(rid)
            collector = self._output_collectors.get(rid)
            if collector is not None:
                if ceiling > 0:
                    error_msg = (
                        f"Request aborted: process memory limit exceeded "
                        f"(usage {usage_gb:.1f} GB, ceiling {ceiling_gb:.1f} GB). "
                        "Reduce context size or lower memory_guard_tier."
                    )
                else:
                    error_msg = (
                        f"Request aborted: process memory limit exceeded "
                        f"(usage {usage_gb:.1f} GB). "
                        "Reduce context size or lower memory_guard_tier."
                    )
                collector.put(
                    RequestOutput(
                        request_id=rid,
                        finished=True,
                        finish_reason="error",
                        new_text=f"\n\n[Error: {error_msg}]",
                        error=error_msg,
                    )
                )
            event = self._finished_events.get(rid)
            if event is not None:
                event.set()
        if request_ids:
            logger.warning(
                f"Aborted {len(request_ids)} requests due to memory pressure"
            )
            self._wake_engine_loop()
        return len(request_ids)

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
                        raise RuntimeError(output.error)

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

        # Get the final output from collector
        collector = self._output_collectors.get(request_id)
        if collector is None:
            raise RuntimeError(f"No collector for request {request_id}")

        # Drain all outputs and get the last one
        final_output = None
        while True:
            output = collector.get_nowait()
            if output is None:
                break
            final_output = output

        # Cleanup
        self._cleanup_request(request_id)

        if final_output is None:
            raise RuntimeError(f"No output for request {request_id}")

        if final_output.error:
            raise RuntimeError(final_output.error)

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
        from .request import Request
        import uuid as uuid_module

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

        # Release model ownership BEFORE setting _closed
        # (_release_model checks not self._closed)
        if self._owns_model:
            registry = get_registry()
            registry.release(self.model, self._engine_id)
            self._owns_model = False
            logger.debug(f"Engine {self._engine_id} released model ownership")

        self._closed = True

        # Both shutdown() and deep_reset() touch the engine stream (directly
        # or via _drain_pending_async_removes / _do_abort_request). The
        # stream is bound to the engine's executor thread, so dispatch both
        # through the executor; fall back to a direct call if the executor
        # is already shut down.
        for fn in (self.scheduler.shutdown, self.scheduler.deep_reset):
            try:
                self._mlx_executor.submit(fn).result()
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

        if self._mlx_executor is not None:
            # MLX's @mx.compile cache is a C++ thread_local CompilerCache. If
            # this worker thread exits with a non-empty cache, ~CompilerCache
            # frees the cached graphs' Python objects from a thread-exit handler
            # WITHOUT the GIL -> "PyThreadState_Get: GIL is released" crash for
            # models with module-scope @mx.compile graphs (DeepSeek V4 unload,
            # ml-explore/mlx #3280). Clear the cache ON this worker thread (GIL
            # held) before the thread is torn down so the destructor runs on an
            # empty cache, then shut down normally. See utils/compile_cache.py.
            if compile_cache_clear_available():
                try:
                    self._mlx_executor.submit(clear_thread_compile_cache).result()
                except RuntimeError:
                    pass
                self._mlx_executor.shutdown(wait=True)
            else:
                # Fallback: the clear symbol is unavailable, so do NOT exit the
                # worker thread (that would run the unsafe destructor). Keep it
                # alive for the process lifetime via the module-global registry.
                _immortal_mlx_executors.append(self._mlx_executor)
                _immortal_mlx_streams.append(self._mlx_stream)
            self._mlx_executor = None

        # Clear output collectors
        for collector in self._output_collectors.values():
            collector.clear()
        self._output_collectors.clear()
        self._stream_states.clear()
        self._finished_events.clear()

        # Release model and tokenizer references for GC
        self.model = None
        self.tokenizer = None
        self.scheduler = None

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

    async def abort_all_requests(self) -> int:
        """Abort all active requests without stopping the engine."""
        engine = getattr(self, "engine", None)
        if engine is None:
            return 0
        return await engine.abort_all_requests()

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
