# SPDX-License-Identifier: Apache-2.0
"""Unit tests for accuracy benchmark orchestration."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.admin.accuracy_benchmark import (
    VALID_BENCHMARKS,
    AccuracyBenchmarkRequest,
    _accumulated_results,
    _append_accumulated_result,
    _build_sampling_kwargs,
    add_to_queue,
    cleanup_old_runs,
    configure_accuracy_result_storage,
    create_run,
    delete_accumulated_result,
    get_accumulated_results,
    get_queue_status,
    get_run,
    reset_accumulated_results,
    run_accuracy_benchmark,
)
from omlx.model_catalog import ModelCatalog
from omlx.model_settings import ModelSettings, ModelSettingsManager


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

    def test_sampling_profile_defaults_to_deterministic(self):
        req = AccuracyBenchmarkRequest(
            model_id="test-model",
            benchmarks={"mmlu": 100},
        )
        assert req.sampling_profile == "deterministic"

    def test_invalid_sampling_profile_rejected(self):
        with pytest.raises(Exception):
            AccuracyBenchmarkRequest(
                model_id="test-model",
                benchmarks={"mmlu": 100},
                sampling_profile="random-chaos",
            )


class TestQueueAndResults:
    def setup_method(self):
        from omlx.admin.accuracy_benchmark import _queue

        _queue.clear()
        configure_accuracy_result_storage(None)
        reset_accumulated_results()

    def test_add_to_queue(self):
        req = AccuracyBenchmarkRequest(
            model_id="model-a",
            benchmarks={"mmlu": 100},
            sampling_profile="deterministic",
        )
        add_to_queue(req)
        status = get_queue_status()
        assert len(status["queue"]) == 1
        assert status["queue"][0]["model_id"] == "model-a"
        assert status["queue"][0]["sampling_profile"] == "deterministic"

    def test_queue_status_empty(self):
        status = get_queue_status()
        assert status["running"] is False
        assert len(status["queue"]) == 0

    def test_accumulated_results(self):
        _accumulated_results.append(
            {"model_id": "m1", "benchmark": "mmlu", "accuracy": 0.5}
        )
        results = get_accumulated_results()
        assert len(results) == 1
        assert results[0]["model_id"] == "m1"

    def test_delete_accumulated_result(self):
        _accumulated_results.append({"result_id": "a1", "model_id": "m1"})
        _accumulated_results.append({"result_id": "b2", "model_id": "m2"})

        assert delete_accumulated_result("a1") is True
        results = get_accumulated_results()
        assert len(results) == 1
        assert results[0]["result_id"] == "b2"

    def test_delete_accumulated_result_not_found(self):
        _accumulated_results.append({"result_id": "a1", "model_id": "m1"})

        assert delete_accumulated_result("missing") is False
        assert len(get_accumulated_results()) == 1

    def test_reset_accumulated_results(self):
        _accumulated_results.append(
            {"model_id": "m1", "benchmark": "mmlu", "accuracy": 0.5}
        )
        reset_accumulated_results()
        assert len(get_accumulated_results()) == 0

    def test_persist_and_reload_accumulated_result(self, tmp_path):
        configure_accuracy_result_storage(tmp_path)
        _append_accumulated_result(
            {
                "result_id": "a1",
                "model_id": "m1",
                "benchmark": "mmlu",
                "accuracy": 0.5,
            }
        )

        result_files = list(
            (tmp_path / "benchmarks" / "accuracy" / "results").glob("*.json")
        )
        assert len(result_files) == 1

        configure_accuracy_result_storage(tmp_path)
        results = get_accumulated_results()
        assert len(results) == 1
        assert results[0]["result_id"] == "a1"
        assert results[0]["model_id"] == "m1"
        assert results[0]["created_at"]

    def test_catalog_accuracy_summary_updates_and_recomputes(self, tmp_path):
        catalog = ModelCatalog(tmp_path)
        configure_accuracy_result_storage(tmp_path, catalog)
        _append_accumulated_result(
            {
                "result_id": "a1",
                "model_id": "m1",
                "benchmark": "humaneval",
                "accuracy": 0.75,
                "correct": 3,
                "total": 4,
                "batch_size": 1,
                "sampling_profile": "deterministic",
                "effective_sampling": {"temperature": 0.0},
                "thinking_used": False,
            }
        )
        _append_accumulated_result(
            {
                "result_id": "b2",
                "model_id": "m1",
                "benchmark": "humaneval",
                "accuracy": 0.9,
                "correct": 9,
                "total": 10,
                "batch_size": 1,
                "sampling_profile": "model_settings",
                "effective_sampling": {"temperature": 0.6},
                "thinking_used": False,
            }
        )

        entry = catalog.get_public("m1")
        assert entry["last_accuracy_result_id"] == "b2"
        assert entry["best_accuracy_summary"]["result_id"] == "b2"
        assert entry["accuracy_summaries_by_benchmark"]["humaneval"]["accuracy"] == 0.9

        assert delete_accumulated_result("b2") is True
        entry = catalog.get_public("m1")
        assert entry["last_accuracy_result_id"] == "a1"
        assert entry["best_accuracy_summary"]["result_id"] == "a1"

        reset_accumulated_results()
        entry = catalog.get_public("m1")
        assert entry["last_accuracy_result_id"] == ""
        assert entry["best_accuracy_summary"] == {}

    def test_old_model_catalog_json_loads_without_accuracy_fields(self, tmp_path):
        catalog_path = tmp_path / "model_catalog.json"
        catalog_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "models": {
                        "m1": {
                            "model_id": "m1",
                            "path": "/tmp/m1",
                            "source": "local",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        catalog = ModelCatalog(tmp_path)
        entry = catalog.get_public("m1")
        assert entry["last_accuracy_result_id"] == ""
        assert entry["best_accuracy_summary"] == {}

    def test_delete_accumulated_result_removes_file(self, tmp_path):
        configure_accuracy_result_storage(tmp_path)
        _append_accumulated_result(
            {
                "result_id": "a1",
                "model_id": "m1",
                "benchmark": "mmlu",
                "accuracy": 0.5,
            }
        )
        result_files = list(
            (tmp_path / "benchmarks" / "accuracy" / "results").glob("*.json")
        )
        assert len(result_files) == 1

        assert delete_accumulated_result("a1") is True
        assert get_accumulated_results() == []
        assert not result_files[0].exists()

    def test_reset_accumulated_results_removes_files(self, tmp_path):
        configure_accuracy_result_storage(tmp_path)
        _append_accumulated_result({"result_id": "a1", "model_id": "m1"})
        _append_accumulated_result({"result_id": "b2", "model_id": "m2"})
        result_dir = tmp_path / "benchmarks" / "accuracy" / "results"
        (result_dir / "stale.json").write_text("{not json", encoding="utf-8")

        reset_accumulated_results()

        assert get_accumulated_results() == []
        assert list(result_dir.glob("*.json")) == []

    def test_invalid_result_file_is_skipped(self, tmp_path):
        result_dir = tmp_path / "benchmarks" / "accuracy" / "results"
        result_dir.mkdir(parents=True)
        (result_dir / "bad.json").write_text("{not json", encoding="utf-8")
        (result_dir / "good.json").write_text(
            json.dumps({"result_id": "ok", "model_id": "m1"}),
            encoding="utf-8",
        )

        configure_accuracy_result_storage(tmp_path)

        results = get_accumulated_results()
        assert len(results) == 1
        assert results[0]["result_id"] == "ok"


class TestRunLifecycle:
    def setup_method(self):
        from omlx.admin.accuracy_benchmark import _accuracy_runs

        _accuracy_runs.clear()
        configure_accuracy_result_storage(None)

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


class TestAccuracySampling:
    def test_deterministic_sampling_profile_forces_neutral_values(self):
        req = AccuracyBenchmarkRequest(
            model_id="test-model",
            benchmarks={"mmlu": 100},
            sampling_profile="deterministic",
            enable_thinking=True,
        )
        kwargs, effective = _build_sampling_kwargs(MagicMock(), req)

        assert kwargs["temperature"] == 0.0
        assert kwargs["top_p"] == 1.0
        assert kwargs["repetition_penalty"] == 1.0
        assert kwargs["presence_penalty"] == 0.0
        assert effective["enable_thinking"] is True

    def test_model_settings_sampling_profile_uses_model_values(self, tmp_path):
        manager = ModelSettingsManager(tmp_path)
        manager.set_settings(
            "test-model",
            ModelSettings(
                temperature=0.6,
                top_p=0.8,
                top_k=20,
                min_p=0.05,
                repetition_penalty=1.1,
                presence_penalty=0.2,
                max_tokens=9999,
                chat_template_kwargs={"foo": "bar"},
                enable_thinking=False,
            ),
        )
        pool = MagicMock()
        pool._settings_manager = manager
        req = AccuracyBenchmarkRequest(
            model_id="test-model",
            benchmarks={"humaneval": 0},
            sampling_profile="model_settings",
            enable_thinking=True,
        )

        kwargs, effective = _build_sampling_kwargs(pool, req)

        assert kwargs["temperature"] == 0.6
        assert kwargs["top_p"] == 0.8
        assert kwargs["top_k"] == 20
        assert kwargs["min_p"] == 0.05
        assert kwargs["repetition_penalty"] == 1.1
        assert kwargs["presence_penalty"] == 0.2
        assert kwargs["chat_template_kwargs"] == {"foo": "bar"}
        assert effective["enable_thinking"] is True
        assert "max_tokens" not in effective


class TestRunAccuracyBenchmark:
    def setup_method(self):
        configure_accuracy_result_storage(None)
        reset_accumulated_results()

    @pytest.mark.asyncio
    async def test_sends_done_event(self, tmp_path):
        """Verify that a successful run sends a done event."""
        configure_accuracy_result_storage(tmp_path)
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
        mock_result.benchmark_variant = None
        mock_result.question_results = []

        mock_evaluator = MagicMock()
        mock_evaluator.load_dataset = AsyncMock(return_value=[{"id": "1"}])
        mock_evaluator.run = AsyncMock(return_value=mock_result)
        mock_evaluator.resolve_max_tokens.return_value = 128

        mock_bench_cls = MagicMock(return_value=mock_evaluator)

        with patch.dict("omlx.eval.BENCHMARKS", {"mmlu": mock_bench_cls}, clear=True):
            await run_accuracy_benchmark(run, mock_pool)

        # Collect all events from the replay log.
        events = list(run.events)

        event_types = [e["type"] for e in events]
        assert "done" in event_types
        result_event = next(e for e in events if e["type"] == "result")
        assert result_event["data"]["batch_size"] == 1
        assert result_event["data"]["result_id"]
        json.dumps(result_event)
        result_files = list(
            (tmp_path / "benchmarks" / "accuracy" / "results").glob("*.json")
        )
        assert len(result_files) == 1
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
        mock_evaluator.run = AsyncMock(
            return_value=MagicMock(
                benchmark_name="mmlu",
                accuracy=0.0,
                total_questions=0,
                correct_count=0,
                time_seconds=0.0,
                category_scores=None,
            )
        )

        mock_bench_cls = MagicMock(return_value=mock_evaluator)

        with patch.dict("omlx.eval.BENCHMARKS", {"mmlu": mock_bench_cls}):
            await run_accuracy_benchmark(run, mock_pool)

        # Should have stopped early
        assert len(run.results) == 0
