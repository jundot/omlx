# SPDX-License-Identifier: Apache-2.0
"""Unit tests for accuracy benchmark orchestration."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.admin.accuracy_benchmark import (
    VALID_BENCHMARKS,
    AccuracyBenchmarkRequest,
    AccuracyBenchmarkRun,
    _accumulated_results,
    add_to_queue,
    cleanup_old_runs,
    create_run,
    get_accumulated_results,
    get_queue_status,
    get_run,
    reset_accumulated_results,
    run_accuracy_benchmark,
)


class TestAccuracyBenchmarkRequest:
    def test_valid_request(self):
        req = AccuracyBenchmarkRequest(
            model_id="test-model",
            benchmarks={"mmlu": 300, "gsm8k": 100},
        )
        assert req.model_id == "test-model"
        assert "mmlu" in req.benchmarks
        assert req.benchmarks["gsm8k"] == 100

    def test_full_dataset_size_zero(self):
        req = AccuracyBenchmarkRequest(
            model_id="test-model",
            benchmarks={"mmlu": 0},
        )
        assert req.benchmarks["mmlu"] == 0

    def test_empty_benchmarks_rejected(self):
        with pytest.raises(Exception):
            AccuracyBenchmarkRequest(
                model_id="test-model",
                benchmarks={},
            )

    def test_invalid_benchmark_rejected(self):
        with pytest.raises(Exception):
            AccuracyBenchmarkRequest(
                model_id="test-model",
                benchmarks={"invalid_bench": 100},
            )

    def test_all_valid_benchmarks(self):
        req = AccuracyBenchmarkRequest(
            model_id="test-model",
            benchmarks={b: 100 for b in VALID_BENCHMARKS},
        )
        assert len(req.benchmarks) == len(VALID_BENCHMARKS)

    def test_enable_thinking_default_false(self):
        req = AccuracyBenchmarkRequest(
            model_id="test-model",
            benchmarks={"mmlu": 100},
        )
        assert req.enable_thinking is False

    def test_enable_thinking_true(self):
        req = AccuracyBenchmarkRequest(
            model_id="test-model",
            benchmarks={"mmlu": 100},
            enable_thinking=True,
        )
        assert req.enable_thinking is True


class TestQueueAndResults:
    def setup_method(self):
        from omlx.admin.accuracy_benchmark import _queue
        _queue.clear()
        reset_accumulated_results()

    def test_add_to_queue(self):
        req = AccuracyBenchmarkRequest(
            model_id="model-a",
            benchmarks={"mmlu": 100},
        )
        add_to_queue(req)
        status = get_queue_status()
        assert len(status["queue"]) == 1
        assert status["queue"][0]["model_id"] == "model-a"

    def test_queue_status_empty(self):
        status = get_queue_status()
        assert status["running"] is False
        assert len(status["queue"]) == 0

    def test_accumulated_results(self):
        _accumulated_results.append({"model_id": "m1", "benchmark": "mmlu", "accuracy": 0.5})
        results = get_accumulated_results()
        assert len(results) == 1
        assert results[0]["model_id"] == "m1"

    def test_reset_accumulated_results(self):
        _accumulated_results.append({"model_id": "m1", "benchmark": "mmlu", "accuracy": 0.5})
        reset_accumulated_results()
        assert len(get_accumulated_results()) == 0


class TestRunLifecycle:
    def setup_method(self):
        from omlx.admin.accuracy_benchmark import _accuracy_runs
        _accuracy_runs.clear()

    def test_create_run(self):
        req = AccuracyBenchmarkRequest(
            model_id="test-model",
            benchmarks={"mmlu": 100},
        )
        run = create_run(req)
        assert run.bench_id is not None
        assert run.status == "running"
        assert run.request == req

    def test_get_run(self):
        req = AccuracyBenchmarkRequest(
            model_id="test-model",
            benchmarks={"mmlu": 100},
        )
        run = create_run(req)
        found = get_run(run.bench_id)
        assert found is run

    def test_get_run_not_found(self):
        assert get_run("nonexistent") is None

    def test_cleanup_old_runs(self):
        req = AccuracyBenchmarkRequest(
            model_id="test-model",
            benchmarks={"mmlu": 100},
        )
        run1 = create_run(req)
        run2 = create_run(req)
        run1.status = "completed"
        run2.status = "running"

        cleanup_old_runs()

        assert get_run(run1.bench_id) is None
        assert get_run(run2.bench_id) is run2

    def test_cleanup_error_runs(self):
        req = AccuracyBenchmarkRequest(
            model_id="test-model",
            benchmarks={"mmlu": 100},
        )
        run = create_run(req)
        run.status = "error"

        cleanup_old_runs()
        assert get_run(run.bench_id) is None


class TestRunAccuracyBenchmark:
    @pytest.mark.asyncio
    async def test_sends_done_event(self):
        """Verify that a successful run sends a done event."""
        req = AccuracyBenchmarkRequest(
            model_id="test-model",
            benchmarks={"mmlu": 100},
        )
        run = create_run(req)

        # Mock engine_pool
        mock_engine = AsyncMock()
        mock_engine.chat = AsyncMock(return_value=MagicMock(text="A"))

        mock_pool = MagicMock()
        mock_pool.get_loaded_model_ids = MagicMock(return_value=[])
        mock_pool.get_engine = AsyncMock(return_value=mock_engine)
        mock_pool._unload_engine = AsyncMock()

        # Mock evaluator
        mock_result = MagicMock()
        mock_result.benchmark_name = "mmlu"
        mock_result.accuracy = 0.75
        mock_result.total_questions = 4
        mock_result.correct_count = 3
        mock_result.time_seconds = 1.0
        mock_result.category_scores = None
        mock_result.thinking_used = False

        mock_evaluator = MagicMock()
        mock_evaluator.load_dataset = AsyncMock(return_value=[{"id": "1"}])
        mock_evaluator.run = AsyncMock(return_value=mock_result)

        mock_bench_cls = MagicMock(return_value=mock_evaluator)

        with patch.dict("omlx.eval.BENCHMARKS", {"mmlu": mock_bench_cls}, clear=True):
            await run_accuracy_benchmark(run, mock_pool)

        # Collect all events
        events = []
        while not run.queue.empty():
            events.append(await run.queue.get())

        event_types = [e["type"] for e in events]
        assert "done" in event_types
        assert run.status == "completed"

    @pytest.mark.asyncio
    async def test_cancellation(self):
        """Verify that cancelling stops the run."""
        req = AccuracyBenchmarkRequest(
            model_id="test-model",
            benchmarks={"mmlu": 100},
        )
        run = create_run(req)
        run.status = "cancelled"  # Pre-cancel

        mock_pool = MagicMock()
        mock_pool.get_loaded_model_ids = MagicMock(return_value=[])
        mock_pool.get_engine = AsyncMock(return_value=MagicMock())
        mock_pool._unload_engine = AsyncMock()

        mock_evaluator = MagicMock()
        mock_evaluator.load_dataset = AsyncMock(return_value=[])
        mock_evaluator.run = AsyncMock(return_value=MagicMock(
            benchmark_name="mmlu",
            accuracy=0.0,
            total_questions=0,
            correct_count=0,
            time_seconds=0.0,
            category_scores=None,
        ))

        mock_bench_cls = MagicMock(return_value=mock_evaluator)

        with patch.dict("omlx.eval.BENCHMARKS", {"mmlu": mock_bench_cls}):
            await run_accuracy_benchmark(run, mock_pool)

        # Should have stopped early
        assert len(run.results) == 0
