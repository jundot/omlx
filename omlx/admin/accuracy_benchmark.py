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
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, field_validator

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

VALID_BENCHMARKS = ["mmlu", "hellaswag", "truthfulqa", "gsm8k", "livecodebench"]


class AccuracyBenchmarkRequest(BaseModel):
    """Request model for starting an accuracy benchmark."""

    model_id: str
    benchmarks: dict[str, int]  # name -> sample_size (0 = full dataset)
    batch_size: int = 1

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        if v not in (1, 2, 4, 8):
            raise ValueError("batch_size must be 1, 2, 4, or 8")
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
    """Tracks the state of a running accuracy benchmark."""

    bench_id: str
    request: AccuracyBenchmarkRequest
    status: str = "running"  # running, completed, cancelled, error
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    task: Optional[asyncio.Task] = None
    results: list[dict] = field(default_factory=list)
    error_message: str = ""
    last_progress: Optional[dict] = None  # last progress event for reconnect


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
    if _current_run_id:
        run = get_run(_current_run_id)
        if run:
            last_progress = run.last_progress
    return {
        "running": _queue_running,
        "current_model": _current_model,
        "current_bench_id": _current_run_id,
        "last_progress": last_progress,
        "queue": [
            {"model_id": r.model_id, "benchmarks": list(r.benchmarks.keys())}
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

    _engine_pool_ref = engine_pool

    if _queue_running:
        return None

    if not _queue:
        return None

    request = _queue.pop(0)
    _queue_running = True
    _current_model = request.model_id

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
        await _continue_queue(engine_pool)

    run.task = asyncio.create_task(_run_and_continue())
    return run.bench_id


async def _continue_queue(engine_pool: Any) -> None:
    """Continue processing the queue after a run completes."""
    global _queue_running, _current_run_id, _current_model

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

    logger.info(
        f"Queue: continuing with {request.model_id} "
        f"benchmarks={list(request.benchmarks.keys())}"
    )

    try:
        await run_accuracy_benchmark(run, engine_pool)
    except Exception as e:
        logger.error(f"Queue: error running {request.model_id}: {e}")

    await _continue_queue(engine_pool)


async def cancel_queue() -> None:
    """Cancel the current run and clear the queue."""
    global _queue_running, _current_run_id, _current_model

    _queue.clear()

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
    """Send an SSE event to the client."""
    if event.get("type") == "progress":
        run.last_progress = event
    try:
        await run.queue.put(event)
    except Exception:
        pass


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
    start_time = time.time()

    try:
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

        engine = await engine_pool.get_engine(request.model_id)

        # Phase 3: Run each benchmark
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

            # Build result
            result_data = {
                "model_id": request.model_id,
                "benchmark": result.benchmark_name,
                "accuracy": round(result.accuracy, 4),
                "total": result.total_questions,
                "correct": result.correct_count,
                "time_s": round(result.time_seconds, 1),
            }
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

        # Phase 4: Unload model
        try:
            await engine_pool._unload_engine(request.model_id)
        except Exception:
            pass

        # Phase 5: Done
        total_time = time.time() - start_time
        run.status = "completed"

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
        await _send_event(run, {
            "type": "error",
            "message": "Benchmark cancelled",
        })
    except Exception as e:
        logger.exception(f"Accuracy benchmark error: {e}")
        run.status = "error"
        run.error_message = str(e)
        await _send_event(run, {
            "type": "error",
            "message": str(e),
        })
