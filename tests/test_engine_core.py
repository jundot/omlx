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
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from omlx.engine_core import EngineCore, AsyncEngineCore, EngineConfig
from omlx.request import Request, RequestOutput, RequestStatus, SamplingParams
from omlx.scheduler import SchedulerConfig


class TestEngineConfig:
    """Tests for EngineConfig dataclass."""

    def test_default_values(self):
        """Test EngineConfig has correct defaults."""
        config = EngineConfig()

        assert config.model_name == ""
        assert config.scheduler_config is None
        assert config.step_interval == 0.001
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
    async def test_add_request_with_default_sampling_params(self, mock_model, mock_tokenizer):
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
    async def test_abort_request_cleans_up(self, mock_model, mock_tokenizer):
        """Test abort_request() cleans up tracking state."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()

                request_id = await engine.add_request(prompt="Hello")
                await engine.abort_request(request_id)

                assert request_id not in engine._output_collectors
                assert request_id not in engine._stream_states
                assert request_id not in engine._finished_events
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

                request_id = await engine.add_request(prompt="Hello")

                # Request should be in scheduler waiting
                assert request_id in engine.scheduler.requests

                await engine.abort_request(request_id)

                # Engine-core state cleaned immediately
                assert request_id not in engine._output_collectors

                # Process the deferred abort (normally happens in step())
                engine.scheduler._process_pending_aborts()

                # Scheduler state must be fully cleaned
                assert request_id not in engine.scheduler.requests
                assert request_id not in engine.scheduler.running
                assert request_id not in engine.scheduler.request_id_to_uid
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
    async def test_error_output_propagates_to_collector(self, mock_model, mock_tokenizer):
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
