# SPDX-License-Identifier: Apache-2.0
"""Unit tests for accuracy benchmark orchestration."""

import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import omlx.admin.accuracy_benchmark as accuracy_benchmark
from omlx.admin.accuracy_benchmark import (
    VALID_BENCHMARKS,
    AccuracyBenchmarkRequest,
    AccuracyBenchmarkRun,
    _accumulated_results,
    add_to_queue,
    cancel_queue,
    cleanup_old_runs,
    create_run,
    get_accumulated_results,
    get_queue_status,
    get_run,
    reset_accumulated_results,
    run_accuracy_benchmark,
    start_next_from_queue,
)
from omlx.model_settings import ModelSettings


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

    def test_sampling_profile_default_deterministic(self):
        req = AccuracyBenchmarkRequest(
            model_id="test-model",
            benchmarks={"mmlu": 100},
        )
        assert req.sampling_profile == "deterministic"

    def test_sampling_profile_model_settings_accepted(self):
        req = AccuracyBenchmarkRequest(
            model_id="test-model",
            benchmarks={"mmlu": 100},
            sampling_profile="model_settings",
        )
        assert req.sampling_profile == "model_settings"

    def test_sampling_profile_invalid_rejected(self):
        with pytest.raises(Exception):
            AccuracyBenchmarkRequest(
                model_id="test-model",
                benchmarks={"mmlu": 100},
                sampling_profile="wild",
            )


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

        # Collect all events from the replay log.
        events = list(run.events)

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


class TestSamplingProfile:
    """sampling_profile gates whether per-model sampling reaches the evaluator.

    Default "deterministic" must read nothing (reproducible greedy scores);
    "model_settings" must forward the model's configured sampling. See #606 /
    the #1254 deterministic-default request.
    """

    def _mock_pool(self, model_settings):
        mock_engine = AsyncMock()
        mock_engine.chat = AsyncMock(return_value=MagicMock(text="A"))
        mock_pool = MagicMock()
        mock_pool.get_loaded_model_ids = MagicMock(return_value=[])
        mock_pool.get_engine = AsyncMock(return_value=mock_engine)
        mock_pool._unload_engine = AsyncMock()
        mock_pool._settings_manager.get_settings = MagicMock(return_value=model_settings)
        return mock_pool

    async def _captured_sampling_kwargs(self, req, mock_pool):
        run = create_run(req)
        mock_result = MagicMock(
            benchmark_name="mmlu", accuracy=0.5, total_questions=1,
            correct_count=1, time_seconds=0.1, category_scores=None,
            thinking_used=False,
        )
        mock_evaluator = MagicMock()
        mock_evaluator.load_dataset = AsyncMock(return_value=[{"id": "1"}])
        mock_evaluator.run = AsyncMock(return_value=mock_result)
        mock_bench_cls = MagicMock(return_value=mock_evaluator)
        with patch.dict("omlx.eval.BENCHMARKS", {"mmlu": mock_bench_cls}, clear=True):
            await run_accuracy_benchmark(run, mock_pool)
        return mock_evaluator.run.call_args.kwargs["sampling_kwargs"]

    @pytest.mark.asyncio
    async def test_deterministic_ignores_model_settings(self):
        # Default profile is "deterministic".
        req = AccuracyBenchmarkRequest(model_id="test-model", benchmarks={"mmlu": 1})
        mock_pool = self._mock_pool(ModelSettings(temperature=0.9, top_p=0.95))
        assert await self._captured_sampling_kwargs(req, mock_pool) == {}

    @pytest.mark.asyncio
    async def test_model_settings_forwards_sampling(self):
        req = AccuracyBenchmarkRequest(
            model_id="test-model",
            benchmarks={"mmlu": 1},
            sampling_profile="model_settings",
        )
        mock_pool = self._mock_pool(ModelSettings(temperature=0.9, top_p=0.95))
        sampling_kwargs = await self._captured_sampling_kwargs(req, mock_pool)
        assert sampling_kwargs["temperature"] == 0.9
        assert sampling_kwargs["top_p"] == 0.95

    @pytest.mark.asyncio
    async def test_deterministic_keeps_chat_template_kwargs(self):
        # Template kwargs are prompt construction, not sampling — forwarded
        # even under the deterministic profile.
        req = AccuracyBenchmarkRequest(model_id="test-model", benchmarks={"mmlu": 1})
        mock_pool = self._mock_pool(
            ModelSettings(temperature=0.9, chat_template_kwargs={"custom_flag": True})
        )
        sampling_kwargs = await self._captured_sampling_kwargs(req, mock_pool)
        assert sampling_kwargs == {"chat_template_kwargs": {"custom_flag": True}}


# =============================================================================
# External endpoint accuracy benchmark tests
# =============================================================================


def _external_dict():
    return {
        "base_url": "http://localhost:8001/v1",
        "api_key": "sk-test",
        "model": "remote-model",
    }


class TestExternalAccuracyRequest:
    def test_external_accepted(self):
        req = AccuracyBenchmarkRequest(
            model_id="remote-model",
            benchmarks={"mmlu": 100},
            external=_external_dict(),
        )
        assert req.external is not None
        assert req.external.model == "remote-model"

    def test_external_forces_thinking_off(self):
        req = AccuracyBenchmarkRequest(
            model_id="remote-model",
            benchmarks={"mmlu": 100},
            enable_thinking=True,
            external=_external_dict(),
        )
        assert req.enable_thinking is False

    def test_local_keeps_thinking(self):
        req = AccuracyBenchmarkRequest(
            model_id="local-model",
            benchmarks={"mmlu": 100},
            enable_thinking=True,
        )
        assert req.enable_thinking is True

    def test_queue_status_flags_external(self):
        req = AccuracyBenchmarkRequest(
            model_id="remote-model",
            benchmarks={"mmlu": 100},
            external=_external_dict(),
        )
        add_to_queue(req)
        try:
            entry = get_queue_status()["queue"][-1]
            assert entry["external"] is True
        finally:
            from omlx.admin.accuracy_benchmark import _queue

            _queue.clear()


class TestExternalAccuracyRun:
    def _mock_result(self):
        return MagicMock(
            benchmark_name="mmlu",
            accuracy=0.5,
            total_questions=2,
            correct_count=1,
            time_seconds=0.1,
            category_scores=None,
            thinking_used=False,
            question_results=[],
        )

    def _mock_evaluator(self):
        mock_evaluator = MagicMock()
        mock_evaluator.load_dataset = AsyncMock(return_value=[{"id": "1"}])
        mock_evaluator.run = AsyncMock(return_value=self._mock_result())
        return mock_evaluator

    def _external_request(self):
        return AccuracyBenchmarkRequest(
            model_id="remote-model",
            benchmarks={"mmlu": 100},
            batch_size=4,
            external=_external_dict(),
        )

    @pytest.mark.asyncio
    async def test_external_run_uses_adapter_and_skips_pool(self):
        run = create_run(self._external_request())
        mock_pool = MagicMock()
        mock_evaluator = self._mock_evaluator()
        mock_bench_cls = MagicMock(return_value=mock_evaluator)

        mock_adapter = MagicMock()
        mock_adapter.preflight = AsyncMock()
        mock_client = MagicMock()
        mock_client.aclose = AsyncMock()

        with (
            patch.dict("omlx.eval.BENCHMARKS", {"mmlu": mock_bench_cls}, clear=True),
            patch(
                "omlx.admin.accuracy_benchmark.ExternalAPIClient",
                return_value=mock_client,
            ),
            patch(
                "omlx.admin.accuracy_benchmark.ExternalChatAdapter",
                return_value=mock_adapter,
            ) as adapter_cls,
        ):
            await run_accuracy_benchmark(run, mock_pool)

        assert run.status == "completed"
        mock_pool.get_engine.assert_not_called()
        mock_pool.get_loaded_model_ids.assert_not_called()
        mock_pool._unload_engine.assert_not_called()
        mock_adapter.preflight.assert_awaited_once()
        adapter_cls.assert_called_once_with(mock_client, "deterministic")
        # Evaluator got the adapter, empty sampling kwargs, thinking off
        call = mock_evaluator.run.call_args
        assert call.args[0] is mock_adapter
        assert call.kwargs["sampling_kwargs"] == {}
        assert call.kwargs["enable_thinking"] is False
        assert call.kwargs["batch_size"] == 4
        # Result carries the external flag and the remote model name
        assert run.results[0]["external"] is True
        assert run.results[0]["model_id"] == "remote-model"
        mock_client.aclose.assert_awaited()
        # Clean up accumulated results this test appended
        reset_accumulated_results()

    @pytest.mark.asyncio
    async def test_external_preflight_failure_emits_error(self):
        from omlx.admin.external_api import ExternalEndpointError

        run = create_run(self._external_request())
        mock_pool = MagicMock()

        mock_adapter = MagicMock()
        mock_adapter.preflight = AsyncMock(
            side_effect=ExternalEndpointError(
                "External endpoint rejected the API key (HTTP 401)"
            )
        )
        mock_client = MagicMock()
        mock_client.aclose = AsyncMock()

        with (
            patch(
                "omlx.admin.accuracy_benchmark.ExternalAPIClient",
                return_value=mock_client,
            ),
            patch(
                "omlx.admin.accuracy_benchmark.ExternalChatAdapter",
                return_value=mock_adapter,
            ),
        ):
            await run_accuracy_benchmark(run, mock_pool)

        assert run.status == "error"
        assert "rejected the API key" in run.error_message
        error_events = [e for e in run.events if e["type"] == "error"]
        assert error_events
        mock_client.aclose.assert_awaited()

    @pytest.mark.asyncio
    async def test_external_result_separates_failures_from_wrong_answers(self):
        questions = [
            SimpleNamespace(
                question_id=str(index),
                correct=status == "correct",
                expected="A",
                predicted="A" if status == "correct" else "",
                question_text="question",
                raw_response="answer",
                category="test",
                time_seconds=0.1,
                status=status,
                finish_reason="stop",
                reasoning_fields_present=[],
                reasoning_fields_nonempty=[],
                prompt_tokens=10,
                completion_tokens=1,
                error_message="timed out" if status == "timeout" else "",
            )
            for index, status in enumerate(
                ["correct", "wrong", "parse_error", "timeout"]
            )
        ]
        mock_result = MagicMock(
            benchmark_name="mmlu",
            accuracy=0.25,
            total_questions=4,
            correct_count=1,
            time_seconds=0.4,
            category_scores=None,
            thinking_used=False,
            question_results=questions,
        )
        mock_evaluator = MagicMock()
        mock_evaluator.load_dataset = AsyncMock(return_value=[{"id": "1"}])
        mock_evaluator.run = AsyncMock(return_value=mock_result)
        mock_adapter = MagicMock()
        mock_adapter.preflight = AsyncMock()
        mock_client = MagicMock()
        mock_client.aclose = AsyncMock()
        run = create_run(self._external_request())

        with (
            patch.dict(
                "omlx.eval.BENCHMARKS",
                {"mmlu": MagicMock(return_value=mock_evaluator)},
                clear=True,
            ),
            patch(
                "omlx.admin.accuracy_benchmark.ExternalAPIClient",
                return_value=mock_client,
            ),
            patch(
                "omlx.admin.accuracy_benchmark.ExternalChatAdapter",
                return_value=mock_adapter,
            ),
        ):
            await run_accuracy_benchmark(run, MagicMock())

        result = run.results[0]
        assert result["valid_response_count"] == 2
        assert result["valid_response_rate"] == 0.5
        assert result["valid_answer_accuracy"] == 0.5
        assert result["wrong_count"] == 1
        assert result["parse_error_count"] == 1
        assert result["timeout_count"] == 1
        assert result["reliability_warning"] is True
        assert result["question_results"][3]["status"] == "timeout"
        reset_accumulated_results()


class _StubResult:
    """Minimal stand-in for an eval BenchmarkResult."""

    def __init__(self):
        self.benchmark_name = "mmlu"
        self.accuracy = 1.0
        self.total_questions = 1
        self.correct_count = 1
        self.time_seconds = 0.0
        self.question_results = []
        self.category_scores = None
        self.thinking_used = False


class _StubEnginePool:
    """Engine pool stub that tracks which model engines are loaded."""

    def __init__(self):
        self._suppress_ttl = False
        self._settings_manager = None
        self.loaded: list[str] = []

    def get_loaded_model_ids(self):
        return list(self.loaded)

    async def _unload_engine(self, model_id):
        if model_id in self.loaded:
            self.loaded.remove(model_id)

    async def get_engine(self, model_id, force_lm=False):
        self.loaded.append(model_id)
        return SimpleNamespace(model_id=model_id)


class TestQueueChainOwnership:
    """Regression tests for the cancel→re-add queue race (issue 1655).

    Runs started by _continue_queue used to have task=None, so cancel_queue
    could only soft-cancel them; the orphaned chain's trailing
    _continue_queue then popped the NEW queue and ran its item concurrently
    with the chain the user started after the cancel, whose Phase 1
    "unload all models" killed the active run (accuracy collapsed to 0.0%).
    """

    def setup_method(self):
        self._reset_module_state()

    def teardown_method(self):
        # Leave no queue/gate state behind for other test files.
        self._reset_module_state()

    @staticmethod
    def _reset_module_state():
        accuracy_benchmark._queue.clear()
        accuracy_benchmark._accuracy_runs.clear()
        accuracy_benchmark._queue_running = False
        accuracy_benchmark._current_run_id = None
        accuracy_benchmark._current_model = None
        reset_accumulated_results()

    @staticmethod
    def _request(model_id: str) -> AccuracyBenchmarkRequest:
        return AccuracyBenchmarkRequest(model_id=model_id, benchmarks={"mmlu": 1})

    @staticmethod
    async def _wait_for(predicate, timeout=5.0):
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not predicate():
            assert loop.time() < deadline, "timed out waiting for condition"
            await asyncio.sleep(0.01)

    @pytest.mark.asyncio
    async def test_continue_queue_run_is_hard_cancellable(self):
        """A queue-continued run records its chain task, so cancel_queue
        cancels it immediately instead of leaving it to run until its next
        on_progress checkpoint (up to a full generation batch away)."""
        b_entered = asyncio.Event()
        b_finished_normally = False

        class StubEval:
            async def load_dataset(self, sample_size=0):
                return [{"id": "1"}]

            async def run(self, engine, items, on_progress, batch_size=1,
                          sampling_kwargs=None, enable_thinking=False):
                nonlocal b_finished_normally
                if engine.model_id == "model-b":
                    b_entered.set()
                    # Blocks until hard-cancelled; never returns on its own.
                    await asyncio.Event().wait()
                    b_finished_normally = True
                return _StubResult()

        pool = _StubEnginePool()
        with patch.dict("omlx.eval.BENCHMARKS", {"mmlu": StubEval}, clear=True):
            add_to_queue(self._request("model-a"))
            add_to_queue(self._request("model-b"))
            start_next_from_queue(pool)

            # model-a completes instantly; model-b is started by
            # _continue_queue, the path that used to leave task=None.
            await asyncio.wait_for(b_entered.wait(), timeout=5)
            run_b = get_run(get_queue_status()["current_bench_id"])
            assert run_b.request.model_id == "model-b"
            assert run_b.task is not None

            await cancel_queue()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(run_b.task, timeout=5)

        assert run_b.status == "cancelled"
        assert not b_finished_normally
        # The cancelled run still emits its terminal event so attached SSE
        # streams close.
        assert any(e["type"] == "error" for e in run_b.events)
        assert run_b.terminal is True

    @pytest.mark.asyncio
    async def test_stale_chain_leaves_queue_and_gate_alone(self):
        """_continue_queue holding a stale ownership token returns without
        popping the queue or mutating the running gate."""
        pool = _StubEnginePool()
        add_to_queue(self._request("model-d"))
        accuracy_benchmark._queue_running = True
        accuracy_benchmark._current_run_id = "live-run"
        accuracy_benchmark._current_model = "model-c"

        await accuracy_benchmark._continue_queue(
            pool, accuracy_benchmark._chain_id - 1
        )

        status = get_queue_status()
        assert [q["model_id"] for q in status["queue"]] == ["model-d"]
        assert status["running"] is True
        assert accuracy_benchmark._current_run_id == "live-run"
        assert accuracy_benchmark._current_model == "model-c"
        assert pool.loaded == []  # stale chain never started a run

    @pytest.mark.asyncio
    async def test_cancel_then_requeue_does_not_corrupt_new_chain(self):
        """The reported sequence: cancel during a queue-continued run, then
        immediately queue new models. The orphaned chain must not pop the
        new queue, flip the gate, or run anything concurrently with the
        chain started after the cancel."""
        b_entered = asyncio.Event()
        b_release = asyncio.Event()
        c_entered = asyncio.Event()
        c_release = asyncio.Event()
        # (event, model_id) log of evaluator.run entries/exits. The cancelled
        # model-b run may legitimately overlap model-c while its last batch
        # drains; the regression is model-d entering while model-c runs.
        run_log: list[tuple[str, str]] = []

        class StubEval:
            async def load_dataset(self, sample_size=0):
                return [{"id": "1"}]

            async def run(self, engine, items, on_progress, batch_size=1,
                          sampling_kwargs=None, enable_thinking=False):
                run_log.append(("enter", engine.model_id))
                try:
                    if engine.model_id == "model-b":
                        b_entered.set()
                        # Simulate an in-flight generation batch: it keeps
                        # running past the cancel and only notices it at the
                        # next on_progress checkpoint.
                        with contextlib.suppress(
                            asyncio.TimeoutError, asyncio.CancelledError
                        ):
                            await asyncio.wait_for(asyncio.Event().wait(), 0.05)
                        await b_release.wait()
                        # Checkpoint: raises CancelledError, run is cancelled.
                        await on_progress(1, 1)
                    elif engine.model_id == "model-c":
                        c_entered.set()
                        await c_release.wait()
                finally:
                    run_log.append(("exit", engine.model_id))
                return _StubResult()

        pool = _StubEnginePool()
        with patch.dict("omlx.eval.BENCHMARKS", {"mmlu": StubEval}, clear=True):
            add_to_queue(self._request("model-a"))
            add_to_queue(self._request("model-b"))
            start_next_from_queue(pool)

            await asyncio.wait_for(b_entered.wait(), timeout=5)
            run_b = get_run(get_queue_status()["current_bench_id"])

            # User cancels, then immediately queues two new models.
            await cancel_queue()
            add_to_queue(self._request("model-c"))
            start_next_from_queue(pool)
            add_to_queue(self._request("model-d"))
            start_next_from_queue(pool)  # no-op: gate is held by model-c

            await asyncio.wait_for(c_entered.wait(), timeout=5)
            status = get_queue_status()
            assert status["current_model"] == "model-c"
            assert [q["model_id"] for q in status["queue"]] == ["model-d"]
            c_bench_id = status["current_bench_id"]

            # Let the soft-cancel window close: model-b's batch finishes and
            # the orphaned chain reaches its trailing _continue_queue.
            b_release.set()
            await self._wait_for(lambda: run_b.terminal)
            await asyncio.sleep(0.05)

            # The stale chain must not have popped model-d, flipped the
            # gate, or started anything next to the live chain.
            status = get_queue_status()
            assert [q["model_id"] for q in status["queue"]] == ["model-d"]
            assert status["running"] is True
            assert status["current_bench_id"] == c_bench_id
            assert ("enter", "model-d") not in run_log

            # The live chain finishes model-c, then runs model-d normally.
            c_release.set()
            await self._wait_for(
                lambda: not get_queue_status()["running"]
                and not get_queue_status()["queue"]
            )

        # model-d ran strictly after model-c finished — never concurrently.
        assert run_log.index(("enter", "model-d")) > run_log.index(
            ("exit", "model-c")
        )
        completed = [r["model_id"] for r in get_accumulated_results()]
        assert completed == ["model-a", "model-c", "model-d"]
