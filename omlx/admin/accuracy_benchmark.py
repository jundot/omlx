# SPDX-License-Identifier: Apache-2.0
"""Accuracy benchmark execution logic for oMLX admin panel.

Orchestrates MMLU, HellaSwag, TruthfulQA, GSM8K, and LiveCodeBench
evaluations with real-time progress reporting via SSE events.

Supports server-side queue and persistent result accumulation.
Results survive browser close and persist until explicitly reset.
"""

import asyncio
import logging
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from pydantic import BaseModel, field_validator, model_validator

from .external_api import (
    ExternalAPIClient,
    ExternalChatAdapter,
    ExternalEndpointConfig,
)

logger = logging.getLogger(__name__)

# Module-level storage for active benchmark runs
_accuracy_runs: dict[str, "AccuracyBenchmarkRun"] = {}

# Accumulated results — persists until explicit reset
_accumulated_results: list[dict] = []

# Server-side queue
_queue: list["AccuracyBenchmarkRequest"] = []
_queue_running: bool = False
_current_run_id: Optional[str] = None
_current_model: Optional[str] = None
_engine_pool_ref: Any = None
# Chain-ownership token. A "chain" is one start_next_from_queue call plus
# the _continue_queue tail it spawns. Each chain captures the token current
# at its start; cancel_queue and start_next_from_queue bump it. A chain
# whose token is stale (e.g. it was soft-cancelled and only noticed at its
# next checkpoint, after the user already started a new chain) must not pop
# the queue or mutate _queue_running/_current_run_id — otherwise it starts
# a run concurrently with the live chain, whose Phase 1 "unload all models"
# rips the engine out from under the active run.
_chain_id: int = 0

VALID_BENCHMARKS = [
    "mmlu", "mmlu_pro", "kmmlu", "cmmlu", "jmmlu",
    "hellaswag", "truthfulqa", "arc_challenge", "winogrande",
    "gsm8k", "mathqa", "humaneval", "mbpp", "livecodebench",
    "bbq", "safetybench",
]

# Sampling profile for an accuracy run. "deterministic" (default) runs greedy
# (temperature 0) so saved scores stay reproducible; "model_settings" opts in to
# the model's configured sampling (temperature, top_p, …) for a real-world score.
SamplingProfile = Literal["deterministic", "model_settings"]


class AccuracyBenchmarkRequest(BaseModel):
    """Request model for starting an accuracy benchmark."""

    model_id: str
    benchmarks: dict[str, int]  # name -> sample_size (0 = full dataset)
    batch_size: int = 1
    enable_thinking: bool = False
    sampling_profile: SamplingProfile = "deterministic"
    # When set, the benchmark runs against a remote OpenAI-compatible
    # endpoint instead of a local engine and model_id is the remote
    # model name (not validated against the local catalog).
    external: Optional[ExternalEndpointConfig] = None

    @model_validator(mode="after")
    def _force_thinking_off_for_external(self) -> "AccuracyBenchmarkRequest":
        # enable_thinking is a local chat-template kwarg; external requests
        # never send it, so keep the stored flag honest.
        if self.external is not None:
            self.enable_thinking = False
        return self

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        if v not in (1, 2, 4, 8, 16, 32):
            raise ValueError("batch_size must be 1, 2, 4, 8, 16, or 32")
        return v

    @field_validator("benchmarks")
    @classmethod
    def validate_benchmarks(cls, v: dict[str, int]) -> dict[str, int]:
        if not v:
            raise ValueError("At least one benchmark is required")
        for name, size in v.items():
            if name not in VALID_BENCHMARKS:
                raise ValueError(
                    f"Invalid benchmark '{name}'. Must be one of {VALID_BENCHMARKS}"
                )
            if size < 0:
                raise ValueError(f"Sample size for '{name}' must be >= 0")
        return v


@dataclass
class AccuracyBenchmarkRun:
    """Tracks the state of a running accuracy benchmark.

    SSE delivery model mirrors `BenchmarkRun`: append-only `events`
    log + `cond` for live notification + `terminal` flag set on the
    final event. See benchmark.py for the rationale.
    """

    bench_id: str
    request: AccuracyBenchmarkRequest
    status: str = "running"  # running, completed, cancelled, error
    events: list[dict] = field(default_factory=list)
    cond: asyncio.Condition = field(default_factory=asyncio.Condition)
    terminal: bool = False
    task: Optional[asyncio.Task] = None
    results: list[dict] = field(default_factory=list)
    error_message: str = ""
    last_progress: Optional[dict] = None  # last progress event for reconnect
    # Finer-grained lifecycle than `status` — surfaces the difference between
    # "still scoring questions" and "cleaning up after the last result was
    # emitted". The serialization gate (_queue_running) stays True across
    # both, but a UI rendering the running row wants to hide it once
    # phase=="unloading" so the user isn't told "still running" when the
    # result card has already appeared on screen. Transitions:
    #   pending → loading → evaluating → unloading → completed
    # (cancelled / error replace the terminal phase on those branches.)
    phase: str = "pending"


# Accuracy stream closes on `done` (run finished) or `error`. Unlike the
# throughput bench there's no separate upload phase to ride out.
_ACCURACY_TERMINAL_TYPES = frozenset({"done", "error"})


# --- Run management ---


def get_run(bench_id: str) -> Optional[AccuracyBenchmarkRun]:
    """Get an accuracy benchmark run by ID."""
    return _accuracy_runs.get(bench_id)


def create_run(request: AccuracyBenchmarkRequest) -> AccuracyBenchmarkRun:
    """Create a new accuracy benchmark run."""
    bench_id = str(uuid.uuid4())[:8]
    run = AccuracyBenchmarkRun(bench_id=bench_id, request=request)
    _accuracy_runs[bench_id] = run
    return run


def cleanup_old_runs() -> None:
    """Remove completed/errored runs to prevent memory leaks."""
    to_remove = []
    for bid, run in _accuracy_runs.items():
        if run.status in ("completed", "cancelled", "error"):
            to_remove.append(bid)
    for bid in to_remove:
        del _accuracy_runs[bid]


# --- Accumulated results ---


def get_accumulated_results() -> list[dict]:
    """Get all accumulated benchmark results."""
    return _accumulated_results


def reset_accumulated_results() -> None:
    """Clear all accumulated results."""
    _accumulated_results.clear()


# --- Queue management ---


def add_to_queue(request: AccuracyBenchmarkRequest) -> None:
    """Add a benchmark request to the queue."""
    _queue.append(request)


def get_queue_status() -> dict:
    """Get current queue status."""
    last_progress = None
    phase = None
    if _current_run_id:
        run = get_run(_current_run_id)
        if run:
            last_progress = run.last_progress
            phase = run.phase
    return {
        "running": _queue_running,
        "current_model": _current_model,
        "current_bench_id": _current_run_id,
        "last_progress": last_progress,
        # Finer-grained than `running`: distinguishes "still scoring" from
        # "cleaning up after the last result emitted". Polling UIs hide
        # the running row once phase becomes "unloading" / "completed" so
        # the result card alone tells the story.
        "phase": phase,
        "queue": [
            {
                "model_id": r.model_id,
                "benchmarks": list(r.benchmarks.keys()),
                "external": r.external is not None,
            }
            for r in _queue
        ],
    }


def remove_from_queue(idx: int) -> bool:
    """Remove an item from the queue by index."""
    if 0 <= idx < len(_queue):
        _queue.pop(idx)
        return True
    return False


def start_next_from_queue(engine_pool: Any) -> Optional[str]:
    """Pop next item from queue, create run, start background task.

    Returns bench_id if a run was started, None if already running or queue empty.
    This is synchronous so the caller gets the bench_id immediately.
    """
    global _queue_running, _current_run_id, _current_model, _engine_pool_ref
    global _chain_id

    _engine_pool_ref = engine_pool

    if _queue_running:
        return None

    if not _queue:
        return None

    request = _queue.pop(0)
    _queue_running = True
    _current_model = request.model_id
    # This chain takes ownership of the queue; any earlier chain still
    # draining a soft-cancelled run bails at its next _continue_queue call.
    _chain_id += 1
    my_chain = _chain_id

    cleanup_old_runs()
    run = create_run(request)
    _current_run_id = run.bench_id

    logger.info(
        f"Queue: starting {request.model_id} "
        f"benchmarks={list(request.benchmarks.keys())}"
    )

    async def _run_and_continue():
        try:
            await run_accuracy_benchmark(run, engine_pool)
        except Exception as e:
            logger.error(f"Queue: error running {request.model_id}: {e}")
        # Auto-continue with next in queue
        await _continue_queue(engine_pool, my_chain)

    run.task = asyncio.create_task(_run_and_continue())
    return run.bench_id


async def _continue_queue(engine_pool: Any, chain_id: int) -> None:
    """Continue processing the queue after a run completes.

    `chain_id` is the ownership token captured when this chain started.
    A stale chain (orphaned by cancel_queue, with a new chain started by
    the user since) returns without popping the queue or touching the
    gate, so it cannot start a run concurrently with the live chain.
    """
    global _queue_running, _current_run_id, _current_model

    if chain_id != _chain_id:
        return

    if not _queue:
        _queue_running = False
        _current_run_id = None
        _current_model = None
        return

    request = _queue.pop(0)
    _current_model = request.model_id

    cleanup_old_runs()
    run = create_run(request)
    _current_run_id = run.bench_id
    # Queue-continued runs execute inside this chain's own task; record it
    # so cancel_queue can hard-cancel them instead of waiting for the next
    # on_progress checkpoint (up to a full generation batch away).
    run.task = asyncio.current_task()

    logger.info(
        f"Queue: continuing with {request.model_id} "
        f"benchmarks={list(request.benchmarks.keys())}"
    )

    try:
        await run_accuracy_benchmark(run, engine_pool)
    except Exception as e:
        logger.error(f"Queue: error running {request.model_id}: {e}")

    await _continue_queue(engine_pool, chain_id)


async def cancel_queue() -> None:
    """Cancel the current run and clear the queue."""
    global _queue_running, _current_run_id, _current_model, _chain_id

    _queue.clear()
    # Orphan the live chain before releasing the gate: if the cancelled run
    # only notices at its next checkpoint, its trailing _continue_queue must
    # not race whatever chain the user starts after this cancel.
    _chain_id += 1

    if _current_run_id:
        run = get_run(_current_run_id)
        if run and run.status == "running":
            run.status = "cancelled"
            if run.task and not run.task.done():
                run.task.cancel()

    _queue_running = False
    _current_run_id = None
    _current_model = None


# --- SSE ---


async def _send_event(run: AccuracyBenchmarkRun, event: dict) -> None:
    """Append an event to the run's log and wake subscribers.

    Updates `last_progress` (used by the REST `queue/status` endpoint
    for reconnect hints) and sets `run.terminal` on the final event.
    """
    if event.get("type") == "progress":
        run.last_progress = event
    async with run.cond:
        run.events.append(event)
        if event.get("type") in _ACCURACY_TERMINAL_TYPES:
            run.terminal = True
        run.cond.notify_all()


# --- Benchmark execution ---


async def run_accuracy_benchmark(
    run: AccuracyBenchmarkRun, engine_pool: Any
) -> None:
    """Execute accuracy benchmark run.

    Phases:
    1. Unload all models
    2. Load target model
    3. For each selected benchmark: load data, evaluate, report
    4. Unload model
    5. Send done event
    """
    from ..eval import BENCHMARKS

    request = run.request

    # Suppress TTL auto-unload during benchmark (local engines only)
    if request.external is None:
        engine_pool._suppress_ttl = True
    start_time = time.time()
    client: Optional[ExternalAPIClient] = None

    try:
        run.phase = "loading"
        if request.external is not None:
            # External endpoint: no local model lifecycle. The adapter owns
            # the sampling-profile mapping, so sampling_kwargs stays empty
            # (enable_thinking is already forced off by request validation).
            await _send_event(run, {
                "type": "progress",
                "phase": "connect",
                "model_id": request.model_id,
                "benchmark": "",
                "message": f"Connecting to {request.external.base_url}...",
                "current": 0,
                "total": len(request.benchmarks),
            })
            client = ExternalAPIClient(request.external)
            engine = ExternalChatAdapter(client, request.sampling_profile)
            # Fail fast on auth/URL/model errors so a wrong API key cannot
            # silently produce a 0% score.
            await engine.preflight()
            sampling_kwargs = {}
        else:
            # Phase 1: Unload all models
            loaded_ids = engine_pool.get_loaded_model_ids()
            if loaded_ids:
                await _send_event(run, {
                    "type": "progress",
                    "phase": "unload",
                    "model_id": request.model_id,
                    "benchmark": "",
                    "message": f"Unloading {len(loaded_ids)} model(s)...",
                    "current": 0,
                    "total": len(request.benchmarks),
                })
                for model_id in loaded_ids:
                    try:
                        await engine_pool._unload_engine(model_id)
                    except Exception as e:
                        logger.warning(f"Failed to unload {model_id}: {e}")

            # Phase 2: Load target model
            await _send_event(run, {
                "type": "progress",
                "phase": "load",
                "model_id": request.model_id,
                "benchmark": "",
                "message": f"Loading {request.model_id}...",
                "current": 0,
                "total": len(request.benchmarks),
            })

            # Force LM engine for accuracy benchmarks — text-only tasks
            # don't need VLM and the VLM adapter can produce empty responses.
            engine = await engine_pool.get_engine(request.model_id, force_lm=True)

            # Load model sampling settings. Under the default "deterministic"
            # profile sampling params are not read — the benchmark runs greedy
            # (temperature 0) so saved scores stay reproducible. Only the
            # explicit "model_settings" opt-in honors the model's configured
            # sampling. chat_template_kwargs is prompt construction, not
            # sampling, so it is forwarded in both profiles.
            sampling_kwargs = {}
            if engine_pool._settings_manager is not None:
                ms = engine_pool._settings_manager.get_settings(request.model_id)
                if ms.chat_template_kwargs:
                    sampling_kwargs["chat_template_kwargs"] = ms.chat_template_kwargs
                if request.sampling_profile == "model_settings":
                    if ms.temperature is not None:
                        sampling_kwargs["temperature"] = ms.temperature
                    if ms.top_p is not None:
                        sampling_kwargs["top_p"] = ms.top_p
                    if ms.top_k is not None:
                        sampling_kwargs["top_k"] = ms.top_k
                    if ms.min_p is not None:
                        sampling_kwargs["min_p"] = ms.min_p
                    if ms.repetition_penalty is not None:
                        sampling_kwargs["repetition_penalty"] = ms.repetition_penalty
                    if ms.presence_penalty is not None:
                        sampling_kwargs["presence_penalty"] = ms.presence_penalty

        # Phase 3: Run each benchmark
        run.phase = "evaluating"
        completed = 0
        for bench_name, sample_size in request.benchmarks.items():
            if run.status == "cancelled":
                break

            bench_cls = BENCHMARKS.get(bench_name)
            if bench_cls is None:
                logger.warning(f"Unknown benchmark: {bench_name}")
                continue

            evaluator = bench_cls()

            # Load dataset
            await _send_event(run, {
                "type": "progress",
                "phase": "download",
                "model_id": request.model_id,
                "benchmark": bench_name,
                "message": f"Loading {bench_name} dataset...",
                "current": completed,
                "total": len(request.benchmarks),
            })

            try:
                items = await evaluator.load_dataset(sample_size=sample_size)
            except Exception as e:
                logger.error(f"Failed to load {bench_name} dataset: {e}")
                await _send_event(run, {
                    "type": "error",
                    "message": f"Failed to load {bench_name} dataset: {e}",
                })
                run.status = "error"
                run.error_message = str(e)
                return

            # Run evaluation with progress
            total_items = len(items)

            async def on_progress(current: int, total: int) -> None:
                if run.status == "cancelled":
                    raise asyncio.CancelledError()
                await _send_event(run, {
                    "type": "progress",
                    "phase": "eval",
                    "model_id": request.model_id,
                    "benchmark": bench_name,
                    "message": f"Evaluating {bench_name} ({current}/{total})...",
                    "current": completed,
                    "total": len(request.benchmarks),
                    "bench_current": current,
                    "bench_total": total,
                })

            await _send_event(run, {
                "type": "progress",
                "phase": "eval",
                "model_id": request.model_id,
                "benchmark": bench_name,
                "message": f"Evaluating {bench_name} (0/{total_items})...",
                "current": completed,
                "total": len(request.benchmarks),
                "bench_current": 0,
                "bench_total": total_items,
            })

            try:
                result = await evaluator.run(
                    engine, items, on_progress,
                    batch_size=request.batch_size,
                    sampling_kwargs=sampling_kwargs,
                    enable_thinking=request.enable_thinking,
                )
            except asyncio.CancelledError:
                run.status = "cancelled"
                await _send_event(run, {
                    "type": "error",
                    "message": "Benchmark cancelled",
                })
                return
            except Exception as e:
                logger.error(f"Error running {bench_name}: {e}")
                await _send_event(run, {
                    "type": "error",
                    "message": f"Error running {bench_name}: {e}",
                })
                run.status = "error"
                run.error_message = str(e)
                return

            question_results = []
            for qr in result.question_results:
                question_data = {
                    "id": qr.question_id,
                    "correct": qr.correct,
                    "expected": qr.expected,
                    "predicted": qr.predicted,
                    "question": qr.question_text,
                    "raw_response": qr.raw_response,
                    "category": qr.category,
                    "time_s": round(qr.time_seconds, 3),
                }
                if request.external is not None:
                    question_data.update({
                        "status": qr.status,
                        "finish_reason": qr.finish_reason,
                        "reasoning_fields_present": qr.reasoning_fields_present,
                        "reasoning_fields_nonempty": qr.reasoning_fields_nonempty,
                        "prompt_tokens": qr.prompt_tokens,
                        "completion_tokens": qr.completion_tokens,
                        "error_message": qr.error_message,
                    })
                question_results.append(question_data)

            result_data = {
                "model_id": request.model_id,
                "external": request.external is not None,
                "benchmark": result.benchmark_name,
                "accuracy": round(result.accuracy, 4),
                "thinking_used": result.thinking_used,
                "total": result.total_questions,
                "correct": result.correct_count,
                "time_s": round(result.time_seconds, 1),
                "question_results": question_results,
            }
            if request.external is not None:
                status_counts = Counter(
                    qr.status or "invalid_response" for qr in result.question_results
                )
                valid_responses = status_counts["correct"] + status_counts["wrong"]
                total_questions = result.total_questions
                result_data.update({
                    "valid_response_count": valid_responses,
                    "empty_content_count": status_counts["empty_content"],
                    "truncated_count": status_counts["truncated"],
                    "timeout_count": status_counts["timeout"],
                    "http_error_count": status_counts["http_error"],
                    "connection_error_count": status_counts["connection_error"],
                    "invalid_response_count": status_counts["invalid_response"],
                    "parse_error_count": status_counts["parse_error"],
                    "wrong_count": status_counts["wrong"],
                    "valid_response_rate": round(
                        valid_responses / total_questions
                        if total_questions > 0 else 0.0,
                        4,
                    ),
                    "valid_answer_accuracy": round(
                        result.correct_count / valid_responses
                        if valid_responses > 0 else 0.0,
                        4,
                    ),
                    "reliability_warning": any(
                        status_counts[name] > 0
                        for name in (
                            "empty_content",
                            "truncated",
                            "timeout",
                            "http_error",
                            "connection_error",
                            "invalid_response",
                            "parse_error",
                        )
                    ),
                })
            if result.category_scores:
                result_data["category_scores"] = {
                    k: round(v, 4) for k, v in result.category_scores.items()
                }

            # Accumulate persistently
            _accumulated_results.append(result_data)

            run.results.append(result_data)
            completed += 1

            await _send_event(run, {
                "type": "result",
                "data": result_data,
            })

        # Phase 4: Unload model. The result(s) are already emitted by now,
        # so flip phase so polling clients hide the running indicator
        # (the result card has already appeared on screen — telling the
        # user "still running" while we clean up reads as a bug).
        run.phase = "unloading"
        if request.external is None:
            try:
                await engine_pool._unload_engine(request.model_id)
            except Exception:
                pass

        # Phase 5: Done
        total_time = time.time() - start_time
        run.status = "completed"
        run.phase = "completed"

        await _send_event(run, {
            "type": "done",
            "summary": {
                "model_id": request.model_id,
                "total_time": round(total_time, 1),
                "benchmarks_completed": completed,
            },
        })

    except asyncio.CancelledError:
        run.status = "cancelled"
        run.phase = "cancelled"
        await _send_event(run, {
            "type": "error",
            "message": "Benchmark cancelled",
        })
    except Exception as e:
        logger.exception(f"Accuracy benchmark error: {e}")
        run.status = "error"
        run.phase = "error"
        run.error_message = str(e)
        await _send_event(run, {
            "type": "error",
            "message": str(e),
        })
    finally:
        # Re-enable TTL auto-unload
        engine_pool._suppress_ttl = False
        if client is not None:
            await client.aclose()
