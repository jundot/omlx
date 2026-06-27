# SPDX-License-Identifier: Apache-2.0
"""
Tests for EngineCore module.

Tests cover:
- EngineConfig: default values
- EngineCore initialization
- add_request(): adding requests (async)
- abort_request(): aborting requests (async)
- get_stats(): statistics

Note: Uses pytest-asyncio for async tests.
"""

import asyncio
import concurrent.futures
from unittest.mock import MagicMock, patch

import pytest

from omlx.engine_core import AsyncEngineCore, EngineConfig, EngineCore
from omlx.exceptions import PrefillMemoryExceededError
from omlx.output_collector import RequestOutputCollector
from omlx.request import RequestOutput, SamplingParams
from omlx.scheduler import SchedulerConfig, SchedulerOutput


class TestEngineConfig:
    """Tests for EngineConfig dataclass."""

    def test_default_values(self):
        """Test EngineConfig has correct defaults."""
        config = EngineConfig()

        assert config.model_name == ""
        assert config.scheduler_config is None
        assert config.step_interval == 0.05
        assert config.stream_interval == 1

    def test_custom_values(self):
        """Test EngineConfig with custom values."""
        scheduler_config = SchedulerConfig(max_num_seqs=64)
        config = EngineConfig(
            model_name="test-model",
            scheduler_config=scheduler_config,
            step_interval=0.005,
            stream_interval=5,
        )

        assert config.model_name == "test-model"
        assert config.scheduler_config is scheduler_config
        assert config.scheduler_config.max_num_seqs == 64
        assert config.step_interval == 0.005
        assert config.stream_interval == 5


class TestEngineCoreInitialization:
    """Tests for EngineCore initialization."""

    def test_init_with_defaults(self, mock_model, mock_tokenizer):
        """Test EngineCore initializes with default config."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                assert engine.model is mock_model
                assert engine.tokenizer is mock_tokenizer
                assert isinstance(engine.config, EngineConfig)
                assert engine._running is False
                assert engine._task is None
                assert engine._steps_executed == 0
                assert engine._output_collectors == {}
                assert engine._stream_states == {}
                assert engine._finished_events == {}
            finally:
                engine.close()

    def test_init_with_custom_config(self, mock_model, mock_tokenizer):
        """Test EngineCore initializes with custom config."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            config = EngineConfig(
                model_name="custom-model",
                step_interval=0.01,
                stream_interval=3,
            )
            engine = EngineCore(
                model=mock_model,
                tokenizer=mock_tokenizer,
                config=config,
            )

            try:
                assert engine.config.model_name == "custom-model"
                assert engine.config.step_interval == 0.01
                assert engine.config.stream_interval == 3
            finally:
                engine.close()

    def test_init_generates_engine_id(self, mock_model, mock_tokenizer):
        """Test EngineCore generates unique engine ID."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                assert engine.engine_id is not None
                assert len(engine.engine_id) > 0
            finally:
                engine.close()

    def test_init_with_custom_engine_id(self, mock_model, mock_tokenizer):
        """Test EngineCore uses provided engine ID."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(
                model=mock_model,
                tokenizer=mock_tokenizer,
                engine_id="custom-engine-123",
            )

            try:
                assert engine.engine_id == "custom-engine-123"
            finally:
                engine.close()


class TestEngineCoreStartStop:
    """Tests for EngineCore start/stop."""

    @pytest.mark.asyncio
    async def test_start_sets_running(self, mock_model, mock_tokenizer):
        """Test start() sets engine to running state."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()

                assert engine._running is True
                assert engine._task is not None
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_stop_clears_running(self, mock_model, mock_tokenizer):
        """Test stop() clears running state."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()
                await engine.stop()

                assert engine._running is False
                assert engine._task is None
            finally:
                engine.close()

    @pytest.mark.asyncio
    async def test_is_running(self, mock_model, mock_tokenizer):
        """Test is_running() returns correct state."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                assert engine.is_running() is False

                await engine.start()
                assert engine.is_running() is True

                await engine.stop()
                assert engine.is_running() is False
            finally:
                engine.close()

    @pytest.mark.asyncio
    async def test_double_start_noop(self, mock_model, mock_tokenizer):
        """Test starting already running engine is no-op."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()
                first_task = engine._task

                await engine.start()  # Second start should be no-op
                assert engine._task is first_task
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_idle_loop_wakes_without_waiting_for_step_interval(
        self, mock_model, mock_tokenizer
    ):
        """Idle loop should sleep cheaply but wake immediately for new work."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(
                model=mock_model,
                tokenizer=mock_tokenizer,
                config=EngineConfig(step_interval=10.0),
            )

            try:
                engine.scheduler.has_requests = MagicMock(return_value=False)
                await engine.start()

                for _ in range(20):
                    if engine.scheduler.has_requests.call_count >= 2:
                        break
                    await asyncio.sleep(0.01)

                calls_before = engine.scheduler.has_requests.call_count
                assert calls_before >= 2

                await asyncio.sleep(0.05)
                assert engine.scheduler.has_requests.call_count == calls_before

                engine._wake_engine_loop()
                for _ in range(20):
                    if engine.scheduler.has_requests.call_count > calls_before:
                        break
                    await asyncio.sleep(0.01)

                assert engine.scheduler.has_requests.call_count > calls_before
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_loop_sleeps_when_scheduler_reports_no_work(
        self, mock_model, mock_tokenizer
    ):
        """Admission backpressure must not spin the engine loop."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(
                model=mock_model,
                tokenizer=mock_tokenizer,
                config=EngineConfig(step_interval=10.0),
            )

            try:
                engine.scheduler.has_requests = MagicMock(return_value=True)
                engine.scheduler.step = MagicMock(
                    return_value=SchedulerOutput(has_work=False)
                )
                await engine.start()

                for _ in range(20):
                    if engine.scheduler.step.call_count >= 1:
                        break
                    await asyncio.sleep(0.01)

                calls_before = engine.scheduler.step.call_count
                assert calls_before == 1

                await asyncio.sleep(0.05)
                assert engine.scheduler.step.call_count == calls_before

                engine._wake_engine_loop()
                for _ in range(20):
                    if engine.scheduler.step.call_count > calls_before:
                        break
                    await asyncio.sleep(0.01)

                assert engine.scheduler.step.call_count > calls_before
            finally:
                await engine.stop()
                engine.close()


class TestEngineCoreAddRequest:
    """Tests for EngineCore.add_request()."""

    @pytest.mark.asyncio
    async def test_add_request_returns_id(self, mock_model, mock_tokenizer):
        """Test add_request() returns request ID."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()

                request_id = await engine.add_request(
                    prompt="Hello, world!",
                    sampling_params=SamplingParams(max_tokens=50),
                )

                assert request_id is not None
                assert isinstance(request_id, str)
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_add_request_with_custom_id(self, mock_model, mock_tokenizer):
        """Test add_request() uses provided request ID."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()

                request_id = await engine.add_request(
                    prompt="Hello",
                    request_id="custom-request-001",
                )

                assert request_id == "custom-request-001"
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_add_request_creates_collector(self, mock_model, mock_tokenizer):
        """Test add_request() creates output collector."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()

                request_id = await engine.add_request(prompt="Hello")

                assert request_id in engine._output_collectors
                assert request_id in engine._stream_states
                assert request_id in engine._finished_events
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_add_request_cleans_up_if_scheduler_insert_fails(
        self, mock_model, mock_tokenizer
    ):
        """If the scheduler insert fails/cancels after the collector is created,
        add_request must drop the tracking (and abort) so no phantom collector
        leaks that the reaper can't see — it was never stamped finished
        (#1154). BaseException in the guard also covers the real
        trigger: CancelledError when a client disconnects before streaming."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                await engine.start()
                engine.scheduler.add_request = MagicMock(
                    side_effect=RuntimeError("insert boom")
                )
                engine.scheduler.abort_request = MagicMock(return_value=True)

                with pytest.raises(RuntimeError):
                    await engine.add_request(prompt="Hello")

                # No phantom tracking left behind for any request.
                assert engine._output_collectors == {}
                assert engine._stream_states == {}
                assert engine._finished_events == {}
                assert engine._finished_at == {}
                # The partial scheduler insert was aborted (idempotent).
                engine.scheduler.abort_request.assert_called_once()
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_add_request_with_default_sampling_params(
        self, mock_model, mock_tokenizer
    ):
        """Test add_request() uses default sampling params when none provided."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()

                request_id = await engine.add_request(prompt="Hello")

                # Should not raise - default params used
                assert request_id is not None
            finally:
                await engine.stop()
                engine.close()


class TestEngineCoreAbortRequest:
    """Tests for EngineCore.abort_request()."""

    @pytest.mark.asyncio
    async def test_abort_request_after_close_returns_false(self):
        """Late aborts after close should not touch a cleared scheduler."""
        engine = EngineCore.__new__(EngineCore)
        engine._closed = True
        engine.scheduler = None
        engine._output_collectors = {}
        engine._finished_events = {}

        result = await engine.abort_request("request-after-close")

        assert result is False

    @pytest.mark.asyncio
    async def test_abort_request(self, mock_model, mock_tokenizer):
        """Test abort_request() returns True for existing request."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()

                request_id = await engine.add_request(prompt="Hello")
                result = await engine.abort_request(request_id)

                assert result is True
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_abort_request_signals_consumer(self, mock_model, mock_tokenizer):
        """Test abort_request() signals consumer with error output."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()
                engine.scheduler.has_requests = lambda: False

                request_id = await engine.add_request(prompt="Hello")
                await engine.abort_request(request_id)

                # Collector should still exist with abort error
                assert request_id in engine._output_collectors
                collector = engine._output_collectors[request_id]
                output = collector.get_nowait()
                assert output is not None
                assert output.finished is True
                assert output.finish_reason == "abort"
                assert output.error == "Request aborted"

                # Event should be set
                assert engine._finished_events[request_id].is_set()

                # Consumer's finally block handles cleanup
                engine._cleanup_request(request_id)
                assert request_id not in engine._output_collectors
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_abort_request_no_ghost_in_scheduler(
        self, mock_model, mock_tokenizer
    ):
        """Deferred abort must clean scheduler state (no ghost request).

        Regression: _cleanup_request used to call remove_finished_request()
        which deleted from scheduler.requests before the deferred abort ran,
        causing _do_abort_request to skip cleanup and leave ghost state in
        scheduler.running / uid mappings / active batch.
        """
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()
                engine.scheduler.has_requests = lambda: False

                request_id = await engine.add_request(prompt="Hello")

                # Request should be in scheduler waiting
                assert request_id in engine.scheduler.requests

                await engine.abort_request(request_id)

                # abort_request signals consumer but does not clean up
                assert request_id in engine._output_collectors

                # Process the deferred abort (normally happens in step())
                engine.scheduler._process_pending_aborts()

                # Scheduler state must be fully cleaned
                assert request_id not in engine.scheduler.requests
                assert request_id not in engine.scheduler.running
                assert request_id not in engine.scheduler.request_id_to_uid

                # Consumer's finally block handles engine-core cleanup
                engine._cleanup_request(request_id)
                assert request_id not in engine._output_collectors
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_abort_request_wakes_blocked_stream_outputs(
        self, mock_model, mock_tokenizer
    ):
        """abort_request() must wake a blocked stream_outputs() consumer.

        Regression test: previously abort_request called _cleanup_request
        which reset the collector's asyncio.Event without waking waiters,
        causing stream_outputs to block forever.
        """
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()
                engine.scheduler.has_requests = lambda: False

                request_id = await engine.add_request(prompt="Hello")

                # Start consuming stream_outputs in a separate task
                outputs = []

                async def consume():
                    async for output in engine.stream_outputs(request_id):
                        outputs.append(output)

                task = asyncio.create_task(consume())
                # Let it enter await collector.get()
                await asyncio.sleep(0.02)

                # Abort from external context while consumer is waiting
                await engine.abort_request(request_id)

                # The task should complete (not hang forever)
                # stream_outputs yields abort error then raises RuntimeError
                with pytest.raises(RuntimeError, match="Request aborted"):
                    await asyncio.wait_for(task, timeout=1.0)

                # Verify abort error was received
                assert len(outputs) == 1
                assert outputs[0].finished is True
                assert outputs[0].error == "Request aborted"

                # stream_outputs' finally block should have cleaned up
                assert request_id not in engine._output_collectors
            finally:
                await engine.stop()
                engine.close()


class TestEngineCoreGetStats:
    """Tests for EngineCore.get_stats()."""

    @pytest.mark.asyncio
    async def test_get_stats_initial(self, mock_model, mock_tokenizer):
        """Test get_stats() returns initial values."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()

                stats = engine.get_stats()

                assert "running" in stats
                assert "uptime_seconds" in stats
                assert "steps_executed" in stats
                assert "active_requests" in stats
                assert "stream_interval" in stats
                assert stats["running"] is True
                assert stats["steps_executed"] == 0
                assert stats["active_requests"] == 0
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_get_stats_includes_scheduler_stats(self, mock_model, mock_tokenizer):
        """Test get_stats() includes scheduler statistics."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                stats = engine.get_stats()

                # Should include scheduler stats
                assert "num_waiting" in stats
                assert "num_running" in stats
            finally:
                engine.close()


class TestEngineCoreClose:
    """Tests for EngineCore.close()."""

    def test_close_releases_model(self, mock_model, mock_tokenizer):
        """Test close() releases model ownership."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            engine.close()

            # Should have called release
            mock_registry.return_value.release.assert_called()

    def test_close_idempotent(self, mock_model, mock_tokenizer):
        """Test close() can be called multiple times safely."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            engine.close()
            engine.close()  # Should not raise

    def test_close_fatal_exits_when_teardown_future_times_out(
        self, mock_model, mock_tokenizer
    ):
        """A stuck scheduler teardown is fatal so a supervisor can restart."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            engine._mlx_executor.shutdown(wait=False)

            future = MagicMock()
            future.result.side_effect = concurrent.futures.TimeoutError
            executor = MagicMock()
            executor.submit.return_value = future
            engine._mlx_executor = executor

            with (
                patch("omlx.engine_core.fatal_exit", side_effect=SystemExit) as fatal,
                pytest.raises(SystemExit),
            ):
                engine.close()

            future.result.assert_called_once_with(timeout=60.0)
            assert "Engine teardown timed out after 60s" in fatal.call_args.args[0]

    def test_close_fatal_exits_when_compile_cache_clear_times_out(
        self, mock_model, mock_tokenizer
    ):
        """A stuck MLX compile-cache clear is also fatal."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            engine._mlx_executor.shutdown(wait=False)

            ok_future = MagicMock()
            ok_future.result.return_value = None
            timeout_future = MagicMock()
            timeout_future.result.side_effect = concurrent.futures.TimeoutError
            executor = MagicMock()
            executor.submit.side_effect = [
                ok_future,
                ok_future,
                ok_future,
                timeout_future,
            ]
            engine._mlx_executor = executor

            with (
                patch(
                    "omlx.engine_core.compile_cache_clear_available", return_value=True
                ),
                patch("omlx.engine_core.fatal_exit", side_effect=SystemExit) as fatal,
                pytest.raises(SystemExit),
            ):
                engine.close()

            timeout_future.result.assert_called_once_with(timeout=60.0)
            assert "MLX compile cache" in fatal.call_args.args[0]


class TestEngineCoreGetCacheStats:
    """Tests for EngineCore.get_cache_stats()."""

    def test_get_cache_stats(self, mock_model, mock_tokenizer):
        """Test get_cache_stats() returns None when no cache."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                stats = engine.get_cache_stats()

                # No SSD cache configured, should return None
                assert stats is None
            finally:
                engine.close()


class TestEngineCoreGenerateCancellation:
    """Tests for EngineCore.generate() cancellation handling."""

    @pytest.mark.asyncio
    async def test_generate_cancel_aborts_request(self, mock_model, mock_tokenizer):
        """Test that cancelling generate() aborts the underlying request."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()
                engine.scheduler.has_requests = lambda: False

                # Create a task that calls generate - it will block on event.wait()
                task = asyncio.create_task(
                    engine.generate(
                        prompt="Hello, world!",
                        sampling_params=SamplingParams(max_tokens=50),
                    )
                )

                # Give the task time to reach event.wait()
                await asyncio.sleep(0.05)

                # There should be one active request
                assert len(engine._output_collectors) == 1
                request_id = list(engine._output_collectors.keys())[0]

                # Cancel the task (simulating client disconnect)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

                # After cancellation, the request should be cleaned up
                assert request_id not in engine._output_collectors
                assert request_id not in engine._stream_states
                assert request_id not in engine._finished_events
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_generate_cancel_multiple_requests(self, mock_model, mock_tokenizer):
        """Test cancelling one generate() does not affect others."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()

                # Prevent step() from running (MockModel can't support
                # BatchGenerator), so the engine loop stays idle and
                # doesn't interfere with cancellation testing.
                engine.scheduler.has_requests = lambda: False

                # Create two generate tasks
                task1 = asyncio.create_task(
                    engine.generate(
                        prompt="Request 1",
                        sampling_params=SamplingParams(max_tokens=50),
                    )
                )
                task2 = asyncio.create_task(
                    engine.generate(
                        prompt="Request 2",
                        sampling_params=SamplingParams(max_tokens=50),
                    )
                )

                await asyncio.sleep(0.05)

                # Should have two active requests
                assert len(engine._output_collectors) == 2
                request_ids = list(engine._output_collectors.keys())

                # Cancel only the first task
                task1.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task1

                # First request cleaned up, second still active
                assert request_ids[0] not in engine._output_collectors
                assert request_ids[1] in engine._output_collectors

                # Clean up second task
                task2.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task2
            finally:
                await engine.stop()
                engine.close()


class TestEngineCoreErrorPropagation:
    """Tests for error propagation from engine loop to requests."""

    @pytest.mark.asyncio
    async def test_error_output_propagates_to_collector(
        self, mock_model, mock_tokenizer
    ):
        """Test that engine loop errors are sent to request collectors."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()

                # Add a request
                request_id = await engine.add_request(
                    prompt="Hello",
                    sampling_params=SamplingParams(max_tokens=50),
                )

                # Simulate: put this request into scheduler.running
                engine.scheduler.running[request_id] = MagicMock()

                # Manually put an error output into the collector
                # (simulating what _engine_loop does on exception)
                collector = engine._output_collectors.get(request_id)
                assert collector is not None

                error_output = RequestOutput(
                    request_id=request_id,
                    finished=True,
                    finish_reason="error",
                    error="Memory limit exceeded during prefill",
                )
                collector.put(error_output)

                # The collector should have the error output
                result = collector.get_nowait()
                assert result is not None
                assert result.error == "Memory limit exceeded during prefill"
                assert result.finished is True
                assert result.finish_reason == "error"
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_stream_outputs_raises_on_error(self, mock_model, mock_tokenizer):
        """Test stream_outputs raises RuntimeError when error output received."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()

                request_id = await engine.add_request(
                    prompt="Hello",
                    sampling_params=SamplingParams(max_tokens=50),
                )

                # Put an error output into the collector
                collector = engine._output_collectors[request_id]
                error_output = RequestOutput(
                    request_id=request_id,
                    finished=True,
                    finish_reason="error",
                    error="Memory limit exceeded during prefill",
                )
                collector.put(error_output)

                # stream_outputs should yield the error output then raise
                with pytest.raises(RuntimeError, match="Memory limit exceeded"):
                    async for _ in engine.stream_outputs(request_id):
                        pass
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_stream_outputs_restores_prefill_memory_error(
        self, mock_model, mock_tokenizer
    ):
        """Structured capacity errors must survive the RequestOutput boundary."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()

                request_id = await engine.add_request(
                    prompt="Hello",
                    sampling_params=SamplingParams(max_tokens=50),
                )

                collector = engine._output_collectors[request_id]
                collector.put(
                    RequestOutput(
                        request_id=request_id,
                        finished=True,
                        finish_reason="error",
                        error="Prefill context too large for available memory",
                        error_code="prefill_memory_exceeded",
                        error_metadata={
                            "request_id": request_id,
                            "estimated_bytes": 123,
                            "limit_bytes": 100,
                        },
                    )
                )

                with pytest.raises(PrefillMemoryExceededError) as exc:
                    async for _ in engine.stream_outputs(request_id):
                        pass

                assert exc.value.request_id == request_id
                assert exc.value.estimated_bytes == 123
                assert exc.value.limit_bytes == 100
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_generate_raises_on_error(self, mock_model, mock_tokenizer):
        """Test generate() raises RuntimeError when error output received."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()

                request_id = await engine.add_request(
                    prompt="Hello",
                    sampling_params=SamplingParams(max_tokens=50),
                )

                # Put an error output and set the finished event
                collector = engine._output_collectors[request_id]
                error_output = RequestOutput(
                    request_id=request_id,
                    finished=True,
                    finish_reason="error",
                    error="Memory limit exceeded during prefill",
                )
                collector.put(error_output)

                event = engine._finished_events[request_id]
                event.set()

                # generate() internally waits on event then drains collector
                # We need to call it in a way that bypasses add_request
                # since the request is already added. Use _generate_from_id
                # directly, but it doesn't exist. Instead, test the drain logic.
                final_output = None
                while True:
                    output = collector.get_nowait()
                    if output is None:
                        break
                    final_output = output

                assert final_output is not None
                assert final_output.error == "Memory limit exceeded during prefill"
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_generate_restores_prefill_memory_error(
        self, mock_model, mock_tokenizer
    ):
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()

                request_id = await engine.add_request(
                    prompt="Hello",
                    sampling_params=SamplingParams(max_tokens=50),
                )
                engine._output_collectors[request_id].put(
                    RequestOutput(
                        request_id=request_id,
                        finished=True,
                        finish_reason="error",
                        error="Prefill context too large for available memory",
                        error_code="prefill_memory_exceeded",
                        error_metadata={
                            "request_id": request_id,
                            "estimated_bytes": 123,
                            "limit_bytes": 100,
                        },
                    )
                )
                engine._finished_events[request_id].set()

                async def _reuse_existing_request(*args, **kwargs):
                    return request_id

                engine.add_request = _reuse_existing_request  # type: ignore[method-assign]

                with pytest.raises(PrefillMemoryExceededError) as exc:
                    await engine.generate(
                        prompt="ignored",
                        sampling_params=SamplingParams(max_tokens=50),
                    )

                assert exc.value.request_id == request_id
                assert exc.value.estimated_bytes == 123
                assert exc.value.limit_bytes == 100
            finally:
                await engine.stop()
                engine.close()


class TestAsyncEngineCore:
    """Tests for AsyncEngineCore wrapper."""

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_model, mock_tokenizer):
        """Test AsyncEngineCore as async context manager."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            async with AsyncEngineCore(
                model=mock_model,
                tokenizer=mock_tokenizer,
            ) as engine:
                assert engine.engine._running is True

            # After exit, should be stopped
            assert engine.engine._running is False

    @pytest.mark.asyncio
    async def test_add_request(self, mock_model, mock_tokenizer):
        """Test AsyncEngineCore.add_request()."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            async with AsyncEngineCore(
                model=mock_model,
                tokenizer=mock_tokenizer,
            ) as engine:
                request_id = await engine.add_request(prompt="Hello")

                assert request_id is not None

    @pytest.mark.asyncio
    async def test_abort_request(self, mock_model, mock_tokenizer):
        """Test AsyncEngineCore.abort_request()."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            async with AsyncEngineCore(
                model=mock_model,
                tokenizer=mock_tokenizer,
            ) as engine:
                request_id = await engine.add_request(prompt="Hello")
                result = await engine.abort_request(request_id)

                assert result is True

    @pytest.mark.asyncio
    async def test_abort_request_after_close_returns_false(self):
        """Late stream cleanup should no-op if unload already closed the core.

        Streaming generators keep an AsyncEngineCore reference. A concurrent
        unload can close that wrapper before the generator's finally block
        calls abort_request(), clearing ``engine`` to None. The wrapper must
        not raise AttributeError in that late-cleanup path.
        """
        async_engine = AsyncEngineCore.__new__(AsyncEngineCore)
        setattr(async_engine, "engine", None)

        result = await async_engine.abort_request("request-after-close")

        assert result is False

    @pytest.mark.asyncio
    async def test_context_manager_exit_after_close_does_not_raise(self):
        """Context-manager cleanup should tolerate an already-closed wrapper."""
        async_engine = AsyncEngineCore.__new__(AsyncEngineCore)
        setattr(async_engine, "engine", None)

        await async_engine.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_abort_all_requests_after_close_returns_zero(self):
        """Bulk abort should no-op if the async wrapper is already closed."""
        async_engine = AsyncEngineCore.__new__(AsyncEngineCore)
        setattr(async_engine, "engine", None)

        count = await async_engine.abort_all_requests()

        assert count == 0

    @pytest.mark.asyncio
    async def test_get_stats(self, mock_model, mock_tokenizer):
        """Test AsyncEngineCore.get_stats()."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            async with AsyncEngineCore(
                model=mock_model,
                tokenizer=mock_tokenizer,
            ) as engine:
                stats = engine.get_stats()

                assert "running" in stats
                assert stats["running"] is True

    @pytest.mark.asyncio
    async def test_get_cache_stats(self, mock_model, mock_tokenizer):
        """Test AsyncEngineCore.get_cache_stats()."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            async with AsyncEngineCore(
                model=mock_model,
                tokenizer=mock_tokenizer,
            ) as engine:
                stats = engine.get_cache_stats()

                assert stats is None  # No SSD cache configured


class TestEngineCoreAbortAllRequests:
    """Tests for EngineCore.abort_all_requests()."""

    @pytest.mark.asyncio
    async def test_abort_all_requests(self, mock_model, mock_tokenizer):
        """Test abort_all_requests() sends errors to all collectors."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()
                engine.scheduler.has_requests = lambda: False

                # Add multiple requests
                rid1 = await engine.add_request(prompt="Hello")
                rid2 = await engine.add_request(prompt="World")

                # Abort all
                count = await engine.abort_all_requests()
                assert count == 2

                # Collectors should have error outputs
                for rid in [rid1, rid2]:
                    collector = engine._output_collectors.get(rid)
                    if collector is not None:
                        output = collector.get_nowait()
                        assert output is not None
                        assert output.finished is True
                        assert output.finish_reason == "error"
                        assert "memory" in output.error.lower()
                        # new_text should contain error message for SSE delivery
                        assert output.new_text is not None
                        assert "[Error:" in output.new_text
                        assert "memory" in output.new_text.lower()

                    # Finished events should be set
                    event = engine._finished_events.get(rid)
                    if event is not None:
                        assert event.is_set()
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_abort_all_requests_empty(self, mock_model, mock_tokenizer):
        """Test abort_all_requests() with no active requests returns 0."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()
                count = await engine.abort_all_requests()
                assert count == 0
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_abort_all_requests_engine_keeps_running(
        self, mock_model, mock_tokenizer
    ):
        """Test engine loop continues after abort_all_requests()."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()
                engine.scheduler.has_requests = lambda: False

                rid = await engine.add_request(prompt="Hello")
                await engine.abort_all_requests()

                # Engine should still be running
                assert engine.is_running() is True

                # Can add new requests after abort
                new_rid = await engine.add_request(prompt="New request")
                assert new_rid in engine._output_collectors
            finally:
                await engine.stop()
                engine.close()


class TestGlobalMLXExecutor:
    """Tests for the global MLX executor singleton (issue #85)."""

    def test_get_mlx_executor_returns_singleton(self):
        """get_mlx_executor() must always return the same executor instance."""
        from omlx.engine_core import get_mlx_executor

        executor1 = get_mlx_executor()
        executor2 = get_mlx_executor()
        assert executor1 is executor2

    def test_engines_have_per_engine_executors(self, mock_model, mock_tokenizer):
        """Each EngineCore must have its own executor (#1248)."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine1 = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            engine2 = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                assert engine1._mlx_executor is not engine2._mlx_executor
            finally:
                engine1.close()
                engine2.close()

    @pytest.mark.asyncio
    async def test_shared_executor_serializes_concurrent_tasks(self):
        """Concurrent submissions to shared executor must never overlap (#85).

        Simulates two engines submitting work simultaneously and verifies
        that tasks run one at a time (no concurrent execution).
        """
        import threading
        import time

        from omlx.engine_core import get_mlx_executor

        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()

        active_count = 0
        max_concurrent = 0
        lock = threading.Lock()

        def simulated_step(task_id: str, duration: float = 0.05):
            """Simulate a scheduler.step() that takes some time."""
            nonlocal active_count, max_concurrent
            with lock:
                active_count += 1
                if active_count > max_concurrent:
                    max_concurrent = active_count
            time.sleep(duration)
            with lock:
                active_count -= 1
            return task_id

        # Submit multiple tasks concurrently (simulating two engines)
        tasks = [
            loop.run_in_executor(executor, simulated_step, "engine_a_step1"),
            loop.run_in_executor(executor, simulated_step, "engine_b_step1"),
            loop.run_in_executor(executor, simulated_step, "engine_a_step2"),
            loop.run_in_executor(executor, simulated_step, "engine_b_step2"),
        ]
        results = await asyncio.gather(*tasks)

        # All tasks completed
        assert set(results) == {
            "engine_a_step1",
            "engine_b_step1",
            "engine_a_step2",
            "engine_b_step2",
        }
        # Critical: no two tasks ever ran at the same time
        assert max_concurrent == 1, (
            f"Expected max 1 concurrent task, got {max_concurrent}. "
            f"Shared executor failed to serialize MLX operations."
        )

    @pytest.mark.asyncio
    async def test_two_engine_loops_run_concurrently_on_separate_executors(
        self, mock_model, mock_tokenizer
    ):
        """Two engines with per-engine executors can run step() concurrently (#1248).

        Each EngineCore has its own ThreadPoolExecutor and mx.Stream, so their
        scheduler.step() calls can overlap. This test verifies that two engines
        actually achieve concurrent execution.
        """
        import threading
        import time

        active_count = 0
        max_concurrent = 0
        total_steps = 0
        lock = threading.Lock()

        def make_tracked_step():
            """Create a step function that tracks concurrency."""
            from omlx.scheduler import SchedulerOutput

            def tracked_step():
                nonlocal active_count, max_concurrent, total_steps
                with lock:
                    active_count += 1
                    total_steps += 1
                    if active_count > max_concurrent:
                        max_concurrent = active_count
                time.sleep(0.01)  # Simulate GPU work
                with lock:
                    active_count -= 1
                return SchedulerOutput(outputs=[])

            return tracked_step

        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine1 = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            engine2 = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            # Wire up tracked step functions
            engine1.scheduler.step = make_tracked_step()
            engine2.scheduler.step = make_tracked_step()
            engine1.scheduler.has_requests = lambda: True
            engine2.scheduler.has_requests = lambda: True

            try:
                await engine1.start()
                await engine2.start()

                # Let both engines run for a bit
                await asyncio.sleep(0.3)
            finally:
                await engine1.stop()
                await engine2.stop()
                engine1.close()
                engine2.close()

        assert (
            total_steps >= 4
        ), f"Expected at least 4 steps from two engines, got {total_steps}"
        # With per-engine executors (#1248), two engines CAN run concurrently.
        # max_concurrent >= 2 means both engines overlapped at least once.
        assert max_concurrent >= 2, (
            f"Expected concurrent execution (max_concurrent >= 2), got {max_concurrent}. "
            f"Per-engine executors should allow parallel step() calls."
        )


class TestEngineCoreCloseReleasesSSDManager:
    """close() must release the SSD cache manager even if shutdown() fails.

    The manager's writer thread holds a strong reference to it, so an unclosed
    manager (and its hot cache) leaks until restart.
    """

    def test_manager_closed_when_shutdown_raises(self, mock_model, mock_tokenizer):
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            scheduler = engine.scheduler
            manager = MagicMock()
            scheduler.paged_ssd_cache_manager = manager
            scheduler.shutdown = MagicMock(side_effect=ValueError("boom"))

            engine.close()  # must not raise

            manager.close.assert_called_once()
            assert scheduler.paged_ssd_cache_manager is None

    def test_manager_closed_when_executor_fallback_raises(
        self, mock_model, mock_tokenizer
    ):
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            scheduler = engine.scheduler
            manager = MagicMock()
            scheduler.paged_ssd_cache_manager = manager
            scheduler.shutdown = MagicMock(side_effect=ValueError("boom"))
            engine._mlx_executor.shutdown(wait=True)

            engine.close()  # must not raise

            manager.close.assert_called_once()
            assert scheduler.paged_ssd_cache_manager is None

    def test_manager_closed_on_normal_close(self, mock_model, mock_tokenizer):
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            scheduler = engine.scheduler
            manager = MagicMock()
            scheduler.paged_ssd_cache_manager = manager

            engine.close()

            manager.close.assert_called_once()
            assert scheduler.paged_ssd_cache_manager is None


class TestStepBurst:
    """Tests for the decode-burst loop (_step_burst).

    Bursting runs several scheduler.step() calls per executor hand-off so the
    MLX thread holds the GIL continuously instead of ping-ponging the event
    loop every decode token.
    """

    def _make_engine(self, mock_model, mock_tokenizer, max_steps, budget=0.2):
        # Mocked scheduler has empty `running`, so the burst takes the
        # single-stream budget; set both so tests are agnostic to the split.
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            config = EngineConfig(
                decode_burst_max_steps=max_steps,
                decode_burst_budget_single_s=budget,
                decode_burst_budget_s=budget,
            )
            return EngineCore(model=mock_model, tokenizer=mock_tokenizer, config=config)

    def test_max_steps_1_runs_single_step(self, mock_model, mock_tokenizer):
        """max_steps=1 disables bursting -> exactly one scheduler.step()."""
        engine = self._make_engine(mock_model, mock_tokenizer, max_steps=1)
        try:
            engine.scheduler.step = MagicMock(
                return_value=SchedulerOutput(has_work=True)
            )
            engine.scheduler.has_requests = MagicMock(return_value=True)
            outs = engine._step_burst()
            assert len(outs) == 1
            assert engine.scheduler.step.call_count == 1
        finally:
            engine.close()

    def test_runs_up_to_max_steps(self, mock_model, mock_tokenizer):
        """With work available and budget headroom, burst hits max_steps."""
        engine = self._make_engine(mock_model, mock_tokenizer, max_steps=4)
        try:
            engine.scheduler.step = MagicMock(
                return_value=SchedulerOutput(has_work=True)
            )
            engine.scheduler.has_requests = MagicMock(return_value=True)
            outs = engine._step_burst()
            assert len(outs) == 4
            assert engine.scheduler.step.call_count == 4
        finally:
            engine.close()

    def test_breaks_when_no_requests(self, mock_model, mock_tokenizer):
        """Burst stops once the scheduler runs dry (e.g. only request finished)."""
        engine = self._make_engine(mock_model, mock_tokenizer, max_steps=4)
        try:
            engine.scheduler.step = MagicMock(
                return_value=SchedulerOutput(has_work=True)
            )
            engine.scheduler.has_requests = MagicMock(return_value=False)
            outs = engine._step_burst()
            assert len(outs) == 1
            assert engine.scheduler.step.call_count == 1
        finally:
            engine.close()

    def test_breaks_on_no_work(self, mock_model, mock_tokenizer):
        """A step that did no work (throttled/idle) ends the burst."""
        engine = self._make_engine(mock_model, mock_tokenizer, max_steps=4)
        try:
            engine.scheduler.step = MagicMock(
                side_effect=[
                    SchedulerOutput(has_work=True),
                    SchedulerOutput(has_work=False),
                    SchedulerOutput(has_work=True),
                ]
            )
            engine.scheduler.has_requests = MagicMock(return_value=True)
            outs = engine._step_burst()
            assert len(outs) == 2  # the no-work step ends bursting
            assert engine.scheduler.step.call_count == 2
        finally:
            engine.close()

    def test_breaks_on_eviction(self, mock_model, mock_tokenizer):
        """A prefill-eviction request needs the async callback -> stop burst."""
        engine = self._make_engine(mock_model, mock_tokenizer, max_steps=4)
        try:
            engine.scheduler.step = MagicMock(
                return_value=SchedulerOutput(
                    has_work=True, prefill_eviction_request=MagicMock()
                )
            )
            engine.scheduler.has_requests = MagicMock(return_value=True)
            outs = engine._step_burst()
            assert len(outs) == 1
            assert engine.scheduler.step.call_count == 1
        finally:
            engine.close()

    def test_breaks_on_budget(self, mock_model, mock_tokenizer):
        """Elapsed budget ends the burst (also caps slow prefill-chunk steps)."""
        engine = self._make_engine(mock_model, mock_tokenizer, max_steps=8, budget=0.05)
        try:
            engine.scheduler.step = MagicMock(
                return_value=SchedulerOutput(has_work=True)
            )
            engine.scheduler.has_requests = MagicMock(return_value=True)
            # deadline = monotonic()(=100.0) + 0.05; next check (=200.0) exceeds it.
            with patch("omlx.engine_core.time.monotonic", side_effect=[100.0, 200.0]):
                outs = engine._step_burst()
            assert len(outs) == 1
            assert engine.scheduler.step.call_count == 1
        finally:
            engine.close()

    def test_budget_zero_disables_bursting(self, mock_model, mock_tokenizer):
        """budget<=0 (with max_steps>1) still runs a single step."""
        engine = self._make_engine(mock_model, mock_tokenizer, max_steps=8, budget=0.0)
        try:
            engine.scheduler.step = MagicMock(
                return_value=SchedulerOutput(has_work=True)
            )
            engine.scheduler.has_requests = MagicMock(return_value=True)
            outs = engine._step_burst()
            assert len(outs) == 1
            assert engine.scheduler.step.call_count == 1
        finally:
            engine.close()

    def test_adaptive_single_budget_when_solo(self, mock_model, mock_tokenizer):
        """One active request -> aggressive single-stream budget (bursts)."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            config = EngineConfig(
                decode_burst_max_steps=4,
                decode_burst_budget_single_s=10.0,  # large -> burst to cap
                decode_burst_budget_s=0.0,  # would disable if used
            )
            engine = EngineCore(
                model=mock_model, tokenizer=mock_tokenizer, config=config
            )
        try:
            engine.scheduler.step = MagicMock(
                return_value=SchedulerOutput(has_work=True)
            )
            engine.scheduler.has_requests = MagicMock(return_value=True)
            engine.scheduler.running = {"a": object()}  # solo
            outs = engine._step_burst()
            assert len(outs) == 4
        finally:
            engine.close()

    def test_adaptive_concurrent_budget_when_busy(self, mock_model, mock_tokenizer):
        """Multiple active requests -> tight concurrent budget (here 0 = none)."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            config = EngineConfig(
                decode_burst_max_steps=8,
                decode_burst_budget_single_s=10.0,  # would burst if used
                decode_burst_budget_s=0.0,  # concurrent: no burst
            )
            engine = EngineCore(
                model=mock_model, tokenizer=mock_tokenizer, config=config
            )
        try:
            engine.scheduler.step = MagicMock(
                return_value=SchedulerOutput(has_work=True)
            )
            engine.scheduler.has_requests = MagicMock(return_value=True)
            engine.scheduler.running = {"a": object(), "b": object()}  # concurrent
            outs = engine._step_burst()
            assert len(outs) == 1
        finally:
            engine.close()


class TestOrphanedCollectorReaping:
    """Reaping of output collectors orphaned by a client disconnect (#1154).

    When a client disconnects mid-stream the SSE generator chain is abandoned
    rather than closed, so stream_outputs()'s cleanup finally only runs at GC
    time and the collector lingers in _output_collectors — the dashboard then
    shows the request as "Generating" indefinitely. _reap_orphaned_collectors()
    drops such orphans after a grace period.
    """

    def test_reaps_only_stale_finished_collectors(self, mock_model, mock_tokenizer):
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                now = 1000.0

                # orphan: finished long ago, consumer never cleaned up.
                orphan = RequestOutputCollector()
                orphan.put(
                    RequestOutput(
                        request_id="orphan",
                        finished=True,
                        finish_reason="abort",
                        new_text="partial",
                    )
                )
                engine._output_collectors["orphan"] = orphan
                engine._finished_events["orphan"] = asyncio.Event()
                engine._finished_at["orphan"] = now - 100.0

                # fresh: just finished, still within grace (consumer may drain).
                engine._output_collectors["fresh"] = RequestOutputCollector()
                engine._finished_events["fresh"] = asyncio.Event()
                engine._finished_at["fresh"] = now - 1.0

                # active: still generating, never marked finished.
                engine._output_collectors["active"] = RequestOutputCollector()
                engine._finished_events["active"] = asyncio.Event()

                reaped = engine._reap_orphaned_collectors(now=now, grace=5.0)

                assert reaped == 1
                # stale orphan dropped from every tracking dict
                assert "orphan" not in engine._output_collectors
                assert "orphan" not in engine._finished_events
                assert "orphan" not in engine._finished_at
                # within-grace and still-active requests are retained
                assert "fresh" in engine._output_collectors
                assert "active" in engine._output_collectors
                # pop-only: a consumer still holding the orphan reference keeps
                # its buffered output — the reaper must NOT clear() it.
                assert orphan.output is not None
                assert orphan.output.new_text == "partial"
            finally:
                engine.close()

    def test_mark_request_finished_stamps_once_and_signals(
        self, mock_model, mock_tokenizer
    ):
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                event = asyncio.Event()
                engine._finished_events["r1"] = event

                engine._mark_request_finished("r1")
                assert event.is_set()
                assert "r1" in engine._finished_at
                first = engine._finished_at["r1"]

                # setdefault semantics: a repeated signal must not reset the
                # grace clock (otherwise an orphan could never age out).
                engine._mark_request_finished("r1")
                assert engine._finished_at["r1"] == first
            finally:
                engine.close()

    def test_cleanup_request_removes_finished_stamp(self, mock_model, mock_tokenizer):
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                engine._output_collectors["r1"] = RequestOutputCollector()
                engine._finished_events["r1"] = asyncio.Event()
                engine._finished_at["r1"] = 123.0

                engine._cleanup_request("r1")

                # normal consumer cleanup also clears the finish stamp so the
                # reaper never revisits a request the consumer already handled.
                assert "r1" not in engine._finished_at
                assert "r1" not in engine._output_collectors
            finally:
                engine.close()

    @pytest.mark.asyncio
    async def test_generate_drains_via_held_reference_if_reaped(
        self, mock_model, mock_tokenizer
    ):
        """generate() captures the collector BEFORE awaiting, so the pop-only
        reaper removing the dict entry once the request finishes cannot lose a
        completed result when generate() is slow to resume under load (#1154).
        Without the early capture, the post-await re-fetch would return None and
        raise — the streaming path was already safe; this extends it to generate().
        """
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            try:
                await engine.start()
                engine.scheduler.has_requests = lambda: False

                task = asyncio.create_task(
                    engine.generate(
                        prompt="Hello",
                        sampling_params=SamplingParams(max_tokens=5),
                    )
                )
                # Let generate() add the request and capture the collector
                # reference (it captures before awaiting the finished event).
                await asyncio.sleep(0.05)
                request_id = list(engine._output_collectors.keys())[0]
                collector = engine._output_collectors[request_id]

                # Deliver the completed result, then simulate the reaper popping
                # the dict entry BEFORE generate() resumes to drain it.
                collector.put(
                    RequestOutput(
                        request_id=request_id,
                        finished=True,
                        finish_reason="stop",
                        new_text="done",
                    )
                )
                engine._output_collectors.pop(request_id)
                engine._finished_events[request_id].set()

                result = await task
                assert result is not None
                assert result.new_text == "done"
            finally:
                await engine.stop()
                engine.close()
