# SPDX-License-Identifier: Apache-2.0
"""Accuracy benchmark execution logic for oMLX admin panel.

Orchestrates MMLU, HellaSwag, TruthfulQA, GSM8K, and LiveCodeBench
evaluations with real-time progress reporting via SSE events.

Supports server-side queue and saved result history.
Completed results persist under the oMLX base path until explicitly deleted.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

# Module-level storage for active benchmark runs
_accuracy_runs: dict[str, "AccuracyBenchmarkRun"] = {}

# Accumulated results loaded from/saved to process-local history storage.
_accumulated_results: list[dict] = []
_result_storage_dir: Optional[Path] = None
_result_paths: dict[str, Path] = {}
_model_catalog_ref: Any = None

# Server-side queue
_queue: list["AccuracyBenchmarkRequest"] = []
_queue_running: bool = False
_current_run_id: Optional[str] = None
_current_model: Optional[str] = None
_engine_pool_ref: Any = None

VALID_BENCHMARKS = [
    "mmlu",
    "mmlu_pro",
    "kmmlu",
    "cmmlu",
    "jmmlu",
    "hellaswag",
    "truthfulqa",
    "arc_challenge",
    "winogrande",
    "gsm8k",
    "mathqa",
    "humaneval",
    "mbpp",
    "livecodebench",
    "bbq",
    "safetybench",
]
VALID_SAMPLING_PROFILES = ("model_settings", "deterministic")


class AccuracyBenchmarkRequest(BaseModel):
    """Request model for starting an accuracy benchmark."""

    model_id: str
    benchmarks: dict[str, int]  # name -> sample_size (0 = full dataset)
    batch_size: int = 1
    enable_thinking: bool = False
    sampling_profile: str = "deterministic"

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        if v not in (1, 2, 4, 8, 16, 32):
            raise ValueError("batch_size must be 1, 2, 4, 8, 16, or 32")
        return v

    @field_validator("sampling_profile")
    @classmethod
    def validate_sampling_profile(cls, v: str) -> str:
        if v not in VALID_SAMPLING_PROFILES:
            raise ValueError(
                f"sampling_profile must be one of {VALID_SAMPLING_PROFILES}"
            )
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


def configure_accuracy_result_storage(
    base_path: Optional[Path],
    model_catalog: Any = None,
) -> None:
    """Configure persisted accuracy result storage and load saved history."""
    global _result_storage_dir, _model_catalog_ref

    _accumulated_results.clear()
    _result_paths.clear()
    _model_catalog_ref = model_catalog

    if base_path is None:
        _result_storage_dir = None
        _rebuild_accuracy_catalog_summaries()
        return

    _result_storage_dir = Path(base_path) / "benchmarks" / "accuracy" / "results"
    _load_accumulated_results()
    _rebuild_accuracy_catalog_summaries()


def _load_accumulated_results() -> None:
    """Load saved benchmark results from disk into memory."""
    if _result_storage_dir is None:
        return

    try:
        _result_storage_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create accuracy result directory: {e}")
        return

    for path in sorted(_result_storage_dir.glob("*.json"), reverse=True):
        try:
            with open(path, "r", encoding="utf-8") as f:
                result = json.load(f)
        except Exception as e:
            logger.warning(f"Skipping invalid accuracy result {path}: {e}")
            continue

        if not isinstance(result, dict):
            logger.warning(f"Skipping non-object accuracy result {path}")
            continue

        result_id = result.get("result_id")
        if not isinstance(result_id, str) or not result_id:
            result_id = path.stem.rsplit("-", 1)[-1] or str(uuid.uuid4())[:8]
            result["result_id"] = result_id

        _accumulated_results.append(result)
        _result_paths[result_id] = path


def _save_accumulated_result(result: dict) -> None:
    """Persist one completed benchmark result if storage is configured."""
    if _result_storage_dir is None:
        return

    try:
        _result_storage_dir.mkdir(parents=True, exist_ok=True)
        created_at = result.setdefault(
            "created_at",
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        result_id = result.get("result_id") or str(uuid.uuid4())[:8]
        result["result_id"] = result_id
        stamp = str(created_at).replace(":", "").replace("+", "").replace("/", "-")
        path = _result_storage_dir / f"{stamp}-{result_id}.json"
        tmp_path = path.with_suffix(".json.tmp")

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            f.write("\n")

        tmp_path.replace(path)
        _result_paths[result_id] = path
    except Exception as e:
        logger.error(f"Failed to persist accuracy result: {e}")


def get_accumulated_results() -> list[dict]:
    """Get all accumulated benchmark results."""
    return _accumulated_results


def delete_accumulated_result(result_id: str) -> bool:
    """Delete one accumulated result by ID, including its saved file."""
    for idx, result in enumerate(_accumulated_results):
        if result.get("result_id") == result_id:
            del _accumulated_results[idx]
            path = _result_paths.pop(result_id, None)
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError as e:
                    logger.warning(f"Failed to delete accuracy result {path}: {e}")
            _rebuild_accuracy_catalog_summaries()
            return True
    return False


def reset_accumulated_results() -> None:
    """Clear all accumulated results, including saved files."""
    _accumulated_results.clear()
    if _result_storage_dir is not None:
        paths = set(_result_paths.values())
        if _result_storage_dir.exists():
            paths.update(_result_storage_dir.glob("*.json"))
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError as e:
                logger.warning(f"Failed to delete accuracy result {path}: {e}")
    _result_paths.clear()
    _rebuild_accuracy_catalog_summaries()


def _append_accumulated_result(result: dict) -> None:
    """Append and persist one completed benchmark result."""
    result.setdefault(
        "created_at",
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    _accumulated_results.append(result)
    _save_accumulated_result(result)
    _rebuild_accuracy_catalog_summaries()


def _result_created_at(result: dict) -> str:
    return str(result.get("created_at") or "")


def _accuracy_result_summary(result: dict) -> dict[str, Any]:
    return {
        "result_id": result.get("result_id", ""),
        "created_at": result.get("created_at", ""),
        "benchmark": result.get("benchmark", ""),
        "benchmark_variant": result.get("benchmark_variant"),
        "accuracy": result.get("accuracy", 0),
        "correct": result.get("correct", 0),
        "total": result.get("total", 0),
        "thinking_used": bool(result.get("thinking_used", False)),
        "batch_size": result.get("batch_size", 1),
        "sampling_profile": result.get("sampling_profile", "deterministic"),
        "temperature": (result.get("effective_sampling") or {}).get("temperature"),
    }


def _build_accuracy_catalog_snapshot(results: list[dict]) -> dict[str, dict[str, Any]]:
    by_model: dict[str, list[dict]] = {}
    for result in results:
        model_id = result.get("model_id")
        if model_id:
            by_model.setdefault(model_id, []).append(result)

    snapshot: dict[str, dict[str, Any]] = {}
    for model_id, model_results in by_model.items():
        latest = model_results[-1]
        best = max(
            model_results,
            key=lambda r: (float(r.get("accuracy") or 0), _result_created_at(r)),
        )
        by_benchmark: dict[str, dict[str, Any]] = {}
        for result in model_results:
            benchmark = result.get("benchmark")
            if not benchmark:
                continue
            existing = by_benchmark.get(benchmark)
            if existing is None or (
                float(result.get("accuracy") or 0),
                _result_created_at(result),
            ) > (
                float(existing.get("accuracy") or 0),
                str(existing.get("created_at") or ""),
            ):
                by_benchmark[benchmark] = _accuracy_result_summary(result)

        snapshot[model_id] = {
            "last_accuracy_result_id": latest.get("result_id", ""),
            "best_accuracy_summary": _accuracy_result_summary(best),
            "accuracy_summaries_by_benchmark": by_benchmark,
        }
    return snapshot


def _rebuild_accuracy_catalog_summaries() -> None:
    if _model_catalog_ref is None:
        return
    try:
        _model_catalog_ref.replace_accuracy_summaries(
            _build_accuracy_catalog_snapshot(_accumulated_results)
        )
    except Exception as e:
        logger.warning(f"Failed to update accuracy catalog summaries: {e}")


def _global_sampling_defaults() -> dict[str, Any]:
    try:
        from ..server import _server_state

        sampling = getattr(_server_state, "sampling", None)
    except Exception:
        sampling = None

    return {
        "temperature": getattr(sampling, "temperature", 1.0),
        "top_p": getattr(sampling, "top_p", 0.95),
        "top_k": getattr(sampling, "top_k", 0),
        "min_p": 0.0,
        "repetition_penalty": getattr(sampling, "repetition_penalty", 1.0),
        "presence_penalty": 0.0,
    }


def _get_model_settings(engine_pool: Any, model_id: str) -> Any:
    settings_manager = getattr(engine_pool, "_settings_manager", None)
    if settings_manager is None:
        return None
    try:
        return settings_manager.get_settings(model_id)
    except Exception as e:
        logger.warning(f"Failed to load model settings for {model_id}: {e}")
        return None


def _build_sampling_kwargs(
    engine_pool: Any,
    request: AccuracyBenchmarkRequest,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build request kwargs and serializable effective sampling metadata."""
    if request.sampling_profile == "deterministic":
        effective = {
            "sampling_profile": "deterministic",
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "chat_template_kwargs": {},
        }
    else:
        effective = _global_sampling_defaults()
        effective["sampling_profile"] = "model_settings"
        ms = _get_model_settings(engine_pool, request.model_id)
        if ms is not None:
            for key in (
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "repetition_penalty",
                "presence_penalty",
            ):
                value = getattr(ms, key, None)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    effective[key] = value
            ct_kwargs = getattr(ms, "chat_template_kwargs", None)
            effective["chat_template_kwargs"] = (
                dict(ct_kwargs) if isinstance(ct_kwargs, dict) else {}
            )
        else:
            effective["chat_template_kwargs"] = {}

    effective["enable_thinking"] = request.enable_thinking
    effective["batch_size"] = request.batch_size

    kwargs = {
        "temperature": effective["temperature"],
        "top_p": effective["top_p"],
        "top_k": effective["top_k"],
        "min_p": effective["min_p"],
        "repetition_penalty": effective["repetition_penalty"],
        "presence_penalty": effective["presence_penalty"],
        "chat_template_kwargs": dict(effective.get("chat_template_kwargs") or {}),
    }
    return kwargs, effective


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
                "batch_size": r.batch_size,
                "enable_thinking": r.enable_thinking,
                "sampling_profile": r.sampling_profile,
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


async def run_accuracy_benchmark(run: AccuracyBenchmarkRun, engine_pool: Any) -> None:
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

    # Suppress TTL auto-unload during benchmark
    engine_pool._suppress_ttl = True
    start_time = time.time()

    try:
        # Phase 1: Unload all models
        run.phase = "loading"
        loaded_ids = engine_pool.get_loaded_model_ids()
        if loaded_ids:
            await _send_event(
                run,
                {
                    "type": "progress",
                    "phase": "unload",
                    "model_id": request.model_id,
                    "benchmark": "",
                    "message": f"Unloading {len(loaded_ids)} model(s)...",
                    "current": 0,
                    "total": len(request.benchmarks),
                },
            )
            for model_id in loaded_ids:
                try:
                    await engine_pool._unload_engine(model_id)
                except Exception as e:
                    logger.warning(f"Failed to unload {model_id}: {e}")

        # Phase 2: Load target model
        await _send_event(
            run,
            {
                "type": "progress",
                "phase": "load",
                "model_id": request.model_id,
                "benchmark": "",
                "message": f"Loading {request.model_id}...",
                "current": 0,
                "total": len(request.benchmarks),
            },
        )

        # Force LM engine for accuracy benchmarks — text-only tasks
        # don't need VLM and the VLM adapter can produce empty responses.
        engine = await engine_pool.get_engine(request.model_id, force_lm=True)

        # Load benchmark sampling profile once per run. Individual evaluators
        # still own answer budgets via resolve_max_tokens().
        sampling_kwargs, effective_sampling_base = _build_sampling_kwargs(
            engine_pool, request
        )

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
            await _send_event(
                run,
                {
                    "type": "progress",
                    "phase": "download",
                    "model_id": request.model_id,
                    "benchmark": bench_name,
                    "message": f"Loading {bench_name} dataset...",
                    "current": completed,
                    "total": len(request.benchmarks),
                },
            )

            try:
                items = await evaluator.load_dataset(sample_size=sample_size)
            except Exception as e:
                logger.error(f"Failed to load {bench_name} dataset: {e}")
                await _send_event(
                    run,
                    {
                        "type": "error",
                        "message": f"Failed to load {bench_name} dataset: {e}",
                    },
                )
                run.status = "error"
                run.error_message = str(e)
                return

            # Run evaluation with progress
            total_items = len(items)

            async def on_progress(current: int, total: int) -> None:
                if run.status == "cancelled":
                    raise asyncio.CancelledError()
                await _send_event(
                    run,
                    {
                        "type": "progress",
                        "phase": "eval",
                        "model_id": request.model_id,
                        "benchmark": bench_name,
                        "message": f"Evaluating {bench_name} ({current}/{total})...",
                        "current": completed,
                        "total": len(request.benchmarks),
                        "bench_current": current,
                        "bench_total": total,
                    },
                )

            await _send_event(
                run,
                {
                    "type": "progress",
                    "phase": "eval",
                    "model_id": request.model_id,
                    "benchmark": bench_name,
                    "message": f"Evaluating {bench_name} (0/{total_items})...",
                    "current": completed,
                    "total": len(request.benchmarks),
                    "bench_current": 0,
                    "bench_total": total_items,
                },
            )

            try:
                effective_sampling = dict(effective_sampling_base)
                effective_sampling["max_tokens"] = evaluator.resolve_max_tokens(
                    engine, request.enable_thinking
                )
                result = await evaluator.run(
                    engine,
                    items,
                    on_progress,
                    batch_size=request.batch_size,
                    sampling_kwargs=sampling_kwargs,
                    enable_thinking=request.enable_thinking,
                )
            except asyncio.CancelledError:
                run.status = "cancelled"
                await _send_event(
                    run,
                    {
                        "type": "error",
                        "message": "Benchmark cancelled",
                    },
                )
                return
            except Exception as e:
                logger.error(f"Error running {bench_name}: {e}")
                await _send_event(
                    run,
                    {
                        "type": "error",
                        "message": f"Error running {bench_name}: {e}",
                    },
                )
                run.status = "error"
                run.error_message = str(e)
                return

            # Build result
            effective_sampling["enable_thinking"] = result.thinking_used
            effective_sampling["max_tokens"] = evaluator.resolve_max_tokens(
                engine, result.thinking_used
            )
            result_data = {
                "result_id": str(uuid.uuid4())[:8],
                "model_id": request.model_id,
                "benchmark": result.benchmark_name,
                "benchmark_variant": result.benchmark_variant,
                "batch_size": request.batch_size,
                "sampling_profile": request.sampling_profile,
                "effective_sampling": effective_sampling,
                "accuracy": round(result.accuracy, 4),
                "thinking_used": result.thinking_used,
                "total": result.total_questions,
                "correct": result.correct_count,
                "time_s": round(result.time_seconds, 1),
                "question_results": [
                    {
                        "id": qr.question_id,
                        "correct": qr.correct,
                        "expected": qr.expected,
                        "predicted": qr.predicted,
                        "question": qr.question_text,
                        "raw_response": qr.raw_response,
                        "category": qr.category,
                        "pass_mode": qr.pass_mode,
                        "failure_type": qr.failure_type,
                        "error": qr.error,
                        "time_s": round(qr.time_seconds, 3),
                    }
                    for qr in result.question_results
                ],
            }
            if result.category_scores:
                result_data["category_scores"] = {
                    k: round(v, 4) for k, v in result.category_scores.items()
                }

            # Accumulate and persist completed results.
            _append_accumulated_result(result_data)

            run.results.append(result_data)
            completed += 1

            await _send_event(
                run,
                {
                    "type": "result",
                    "data": result_data,
                },
            )

        # Phase 4: Unload model. The result(s) are already emitted by now,
        # so flip phase so polling clients hide the running indicator
        # (the result card has already appeared on screen — telling the
        # user "still running" while we clean up reads as a bug).
        run.phase = "unloading"
        try:
            await engine_pool._unload_engine(request.model_id)
        except Exception:
            pass

        # Phase 5: Done
        total_time = time.time() - start_time
        run.status = "completed"
        run.phase = "completed"

        await _send_event(
            run,
            {
                "type": "done",
                "summary": {
                    "model_id": request.model_id,
                    "total_time": round(total_time, 1),
                    "benchmarks_completed": completed,
                },
            },
        )

    except asyncio.CancelledError:
        run.status = "cancelled"
        run.phase = "cancelled"
        await _send_event(
            run,
            {
                "type": "error",
                "message": "Benchmark cancelled",
            },
        )
    except Exception as e:
        logger.exception(f"Accuracy benchmark error: {e}")
        run.status = "error"
        run.phase = "error"
        run.error_message = str(e)
        await _send_event(
            run,
            {
                "type": "error",
                "message": str(e),
            },
        )
    finally:
        # Re-enable TTL auto-unload
        engine_pool._suppress_ttl = False
