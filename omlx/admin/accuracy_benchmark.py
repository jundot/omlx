# SPDX-License-Identifier: Apache-2.0
"""Accuracy benchmark execution logic for oMLX admin panel.

Orchestrates MMLU, HellaSwag, TruthfulQA, GSM8K, and LiveCodeBench
evaluations with real-time progress reporting via SSE events.
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

VALID_BENCHMARKS = ["mmlu", "hellaswag", "truthfulqa", "gsm8k", "livecodebench"]


class AccuracyBenchmarkRequest(BaseModel):
    """Request model for starting an accuracy benchmark."""

    model_id: str
    benchmarks: list[str]
    full_dataset: bool = False
    batch_size: int = 1

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        if v not in (1, 2, 4, 8):
            raise ValueError("batch_size must be 1, 2, 4, or 8")
        return v

    @field_validator("benchmarks")
    @classmethod
    def validate_benchmarks(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("At least one benchmark is required")
        for b in v:
            if b not in VALID_BENCHMARKS:
                raise ValueError(
                    f"Invalid benchmark '{b}'. Must be one of {VALID_BENCHMARKS}"
                )
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


async def _send_event(run: AccuracyBenchmarkRun, event: dict) -> None:
    """Send an SSE event to the client."""
    try:
        await run.queue.put(event)
    except Exception:
        pass


async def run_accuracy_benchmark(
    run: AccuracyBenchmarkRun, engine_pool: Any
) -> None:
    """Execute accuracy benchmark run.

    Phases:
    1. Unload all models
    2. Load target model
    3. For each selected benchmark:
       a. Download dataset (with progress)
       b. Run evaluator (with per-question progress)
       c. Report results
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
            "benchmark": "",
            "message": f"Loading {request.model_id}...",
            "current": 0,
            "total": len(request.benchmarks),
        })

        engine = await engine_pool.get_engine(request.model_id)

        # Phase 3: Run each benchmark
        completed = 0
        for bench_name in request.benchmarks:
            if run.status == "cancelled":
                break

            bench_cls = BENCHMARKS.get(bench_name)
            if bench_cls is None:
                logger.warning(f"Unknown benchmark: {bench_name}")
                continue

            evaluator = bench_cls()

            # Download dataset
            await _send_event(run, {
                "type": "progress",
                "phase": "download",
                "benchmark": bench_name,
                "message": f"Downloading {bench_name} dataset...",
                "current": completed,
                "total": len(request.benchmarks),
            })

            try:
                items = await evaluator.load_dataset(full=request.full_dataset)
            except Exception as e:
                logger.error(f"Failed to load {bench_name} dataset: {e}")
                await _send_event(run, {
                    "type": "error",
                    "message": f"Failed to download {bench_name} dataset: {e}",
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

            # Send result
            result_data = {
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
