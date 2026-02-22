# SPDX-License-Identifier: Apache-2.0
"""
Tests for Scheduler module.

Tests cover:
- SchedulerConfig: default values, custom values
- SchedulerOutput: dataclass behavior
- Scheduler initialization with mock model/tokenizer
- add_request(): adding requests, tokenization
- abort_request(): aborting waiting/running requests
- has_requests(), get_num_waiting(), get_num_running()
- get_request(): request lookup
- get_stats(): statistics

Note: BatchGenerator is mocked; step() is too complex for unit tests.
"""

from collections import deque
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from omlx.request import Request, RequestOutput, RequestStatus, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig, SchedulerOutput, SchedulingPolicy


class TestSchedulerConfig:
    """Tests for SchedulerConfig dataclass."""

    def test_default_values(self):
        """Test SchedulerConfig has correct defaults."""
        config = SchedulerConfig()

        assert config.max_num_seqs == 256
        assert config.max_num_batched_tokens == 8192
        assert config.policy == SchedulingPolicy.FCFS
        assert config.prefill_batch_size == 8
        assert config.completion_batch_size == 32
        assert config.prefill_step_size == 2048
        assert config.paged_cache_block_size == 256
        assert config.max_cache_blocks is None
        assert config.initial_cache_blocks == 256
        assert config.paged_ssd_cache_dir is None
        assert config.paged_ssd_cache_max_size == 100 * 1024 * 1024 * 1024  # 100GB
        assert config.model_name == ""
        assert config.gc_cleanup_interval == 10
        assert config.mlx_cache_cleanup_interval == 10

    def test_custom_values(self):
        """Test SchedulerConfig with custom values."""
        config = SchedulerConfig(
            max_num_seqs=128,
            max_num_batched_tokens=4096,
            policy=SchedulingPolicy.PRIORITY,
            prefill_batch_size=4,
            completion_batch_size=16,
            prefill_step_size=1024,
            paged_cache_block_size=128,
            max_cache_blocks=500,
            initial_cache_blocks=100,
            paged_ssd_cache_dir="/tmp/cache",
            paged_ssd_cache_max_size=50 * 1024 * 1024 * 1024,
            model_name="test-model",
            gc_cleanup_interval=5,
            mlx_cache_cleanup_interval=20,
        )

        assert config.max_num_seqs == 128
        assert config.max_num_batched_tokens == 4096
        assert config.policy == SchedulingPolicy.PRIORITY
        assert config.prefill_batch_size == 4
        assert config.completion_batch_size == 16
        assert config.prefill_step_size == 1024
        assert config.paged_cache_block_size == 128
        assert config.max_cache_blocks == 500
        assert config.initial_cache_blocks == 100
        assert config.paged_ssd_cache_dir == "/tmp/cache"
        assert config.paged_ssd_cache_max_size == 50 * 1024 * 1024 * 1024
        assert config.model_name == "test-model"
        assert config.gc_cleanup_interval == 5
        assert config.mlx_cache_cleanup_interval == 20


class TestSchedulingPolicy:
    """Tests for SchedulingPolicy enum."""

    def test_fcfs_policy(self):
        """Test FCFS policy value."""
        assert SchedulingPolicy.FCFS.value == "fcfs"

    def test_priority_policy(self):
        """Test Priority policy value."""
        assert SchedulingPolicy.PRIORITY.value == "priority"


class TestSchedulerOutput:
    """Tests for SchedulerOutput dataclass."""

    def test_default_values(self):
        """Test SchedulerOutput has correct defaults."""
        output = SchedulerOutput()

        assert output.scheduled_request_ids == []
        assert output.num_scheduled_tokens == 0
        assert output.finished_request_ids == set()
        assert output.outputs == []
        assert output.has_work is False

    def test_custom_values(self):
        """Test SchedulerOutput with custom values."""
        outputs = [
            RequestOutput(
                request_id="req-1",
                new_token_ids=[100],
                new_text="hello",
            )
        ]
        output = SchedulerOutput(
            scheduled_request_ids=["req-1", "req-2"],
            num_scheduled_tokens=100,
            finished_request_ids={"req-1"},
            outputs=outputs,
            has_work=True,
        )

        assert output.scheduled_request_ids == ["req-1", "req-2"]
        assert output.num_scheduled_tokens == 100
        assert output.finished_request_ids == {"req-1"}
        assert len(output.outputs) == 1
        assert output.outputs[0].request_id == "req-1"
        assert output.has_work is True


class TestSchedulerInitialization:
    """Tests for Scheduler initialization."""

    def test_init_with_defaults(self, mock_model, mock_tokenizer):
        """Test Scheduler initializes with default config."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        assert scheduler.model is mock_model
        assert scheduler.tokenizer is mock_tokenizer
        assert isinstance(scheduler.config, SchedulerConfig)
        assert isinstance(scheduler.waiting, deque)
        assert len(scheduler.waiting) == 0
        assert scheduler.running == {}
        assert scheduler.requests == {}
        assert scheduler.finished_req_ids == set()
        assert scheduler.request_id_to_uid == {}
        assert scheduler.uid_to_request_id == {}
        assert scheduler.batch_generator is None

    def test_init_with_custom_config(self, mock_model, mock_tokenizer):
        """Test Scheduler initializes with custom config."""
        config = SchedulerConfig(
            max_num_seqs=64,
            prefill_batch_size=2,
        )
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=config,
        )

        assert scheduler.config.max_num_seqs == 64
        assert scheduler.config.prefill_batch_size == 2

    def test_init_statistics_zero(self, mock_model, mock_tokenizer):
        """Test Scheduler initializes with zero statistics."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        assert scheduler.num_requests_processed == 0
        assert scheduler.total_prompt_tokens == 0
        assert scheduler.total_completion_tokens == 0


class TestSchedulerAddRequest:
    """Tests for Scheduler.add_request()."""

    def test_add_request_with_string_prompt(self, mock_model, mock_tokenizer):
        """Test adding a request with string prompt."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        request = Request(
            request_id="test-001",
            prompt="Hello, world!",
            sampling_params=SamplingParams(max_tokens=50),
        )
        scheduler.add_request(request)

        assert "test-001" in scheduler.requests
        assert request in scheduler.waiting
        assert request.prompt_token_ids is not None
        assert len(request.prompt_token_ids) > 0
        assert request.num_prompt_tokens == len(request.prompt_token_ids)

    def test_add_request_with_token_ids(self, mock_model, mock_tokenizer):
        """Test adding a request with pre-tokenized prompt."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        token_ids = [1, 100, 200, 300]
        request = Request(
            request_id="test-002",
            prompt=token_ids,
            sampling_params=SamplingParams(max_tokens=50),
        )
        # Pre-set token IDs
        request.prompt_token_ids = token_ids
        request.num_prompt_tokens = len(token_ids)

        scheduler.add_request(request)

        assert "test-002" in scheduler.requests
        assert request.prompt_token_ids == token_ids
        assert request.num_prompt_tokens == 4

    def test_add_duplicate_request_raises(self, mock_model, mock_tokenizer):
        """Test adding duplicate request raises ValueError."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        request = Request(
            request_id="test-001",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        scheduler.add_request(request)

        with pytest.raises(ValueError, match="already exists"):
            scheduler.add_request(request)

    def test_add_multiple_requests(self, mock_model, mock_tokenizer):
        """Test adding multiple requests."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        for i in range(5):
            request = Request(
                request_id=f"test-{i:03d}",
                prompt=f"Prompt {i}",
                sampling_params=SamplingParams(),
            )
            scheduler.add_request(request)

        assert len(scheduler.requests) == 5
        assert len(scheduler.waiting) == 5


class TestSchedulerAbortRequest:
    """Tests for Scheduler.abort_request()."""

    def test_abort_waiting_request(self, mock_model, mock_tokenizer):
        """Test aborting a waiting request."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        request = Request(
            request_id="test-001",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        scheduler.add_request(request)

        result = scheduler.abort_request("test-001")

        assert result is True
        assert request.status == RequestStatus.FINISHED_ABORTED
        assert request not in scheduler.waiting
        assert "test-001" in scheduler.finished_req_ids

    def test_abort_nonexistent_request(self, mock_model, mock_tokenizer):
        """Test aborting a non-existent request returns False."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        result = scheduler.abort_request("nonexistent")

        assert result is False

    def test_abort_sets_finish_reason(self, mock_model, mock_tokenizer):
        """Test aborting sets correct finish reason."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        request = Request(
            request_id="test-001",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        scheduler.add_request(request)
        scheduler.abort_request("test-001")

        assert request.get_finish_reason() == "abort"

    def test_abort_running_request_removes_on_generation_stream(
        self, mock_model, mock_tokenizer
    ):
        """Abort must remove active UID on generation_stream."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        request = Request(
            request_id="req-run",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = [1]
        request.num_prompt_tokens = 1
        request.status = RequestStatus.RUNNING

        uid = 7
        scheduler.requests["req-run"] = request
        scheduler.running["req-run"] = request
        scheduler.request_id_to_uid["req-run"] = uid
        scheduler.uid_to_request_id[uid] = "req-run"

        scheduler.batch_generator = MagicMock()
        scheduler.batch_generator.active_batch = MagicMock(uids=[uid])

        with patch("omlx.scheduler.mx") as mock_mx:
            result = scheduler.abort_request("req-run")

        assert result is True
        scheduler.batch_generator.remove.assert_called_once_with([uid])
        mock_mx.stream.assert_called_once()

    def test_abort_running_request_skips_remove_when_uid_not_in_active_batch(
        self, mock_model, mock_tokenizer
    ):
        """Abort must not call remove() when UID is already absent."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        request = Request(
            request_id="req-run-missing",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = [1]
        request.num_prompt_tokens = 1
        request.status = RequestStatus.RUNNING

        uid = 8
        scheduler.requests["req-run-missing"] = request
        scheduler.running["req-run-missing"] = request
        scheduler.request_id_to_uid["req-run-missing"] = uid
        scheduler.uid_to_request_id[uid] = "req-run-missing"

        scheduler.batch_generator = MagicMock()
        scheduler.batch_generator.active_batch = MagicMock(uids=[999])

        scheduler.abort_request("req-run-missing")

        scheduler.batch_generator.remove.assert_not_called()


class TestSchedulerQueryMethods:
    """Tests for Scheduler query methods."""

    def test_has_requests_empty(self, mock_model, mock_tokenizer):
        """Test has_requests() returns False when empty."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        assert scheduler.has_requests() is False

    def test_has_requests_with_waiting(self, mock_model, mock_tokenizer):
        """Test has_requests() returns True with waiting requests."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        request = Request(
            request_id="test-001",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        scheduler.add_request(request)
        assert scheduler.has_requests() is True

    def test_get_num_waiting(self, mock_model, mock_tokenizer):
        """Test get_num_waiting() returns correct count."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        assert scheduler.get_num_waiting() == 0

        for i in range(3):
            request = Request(
                request_id=f"test-{i}",
                prompt=f"Prompt {i}",
                sampling_params=SamplingParams(),
            )
            scheduler.add_request(request)

        assert scheduler.get_num_waiting() == 3

    def test_get_num_running(self, mock_model, mock_tokenizer):
        """Test get_num_running() returns correct count."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        assert scheduler.get_num_running() == 0

        # Manually add to running for testing
        request = Request(
            request_id="test-001",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        scheduler.running["test-001"] = request

        assert scheduler.get_num_running() == 1

    def test_get_request(self, mock_model, mock_tokenizer):
        """Test get_request() returns correct request."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        request = Request(
            request_id="test-001",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        scheduler.add_request(request)

        retrieved = scheduler.get_request("test-001")
        assert retrieved is request

    def test_get_request_nonexistent(self, mock_model, mock_tokenizer):
        """Test get_request() returns None for nonexistent request."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        assert scheduler.get_request("nonexistent") is None


class TestSchedulerStatistics:
    """Tests for Scheduler.get_stats()."""

    def test_get_stats_initial(self, mock_model, mock_tokenizer):
        """Test get_stats() returns correct initial values."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        stats = scheduler.get_stats()

        assert stats["num_waiting"] == 0
        assert stats["num_running"] == 0
        assert stats["num_requests_processed"] == 0
        assert stats["total_prompt_tokens"] == 0
        assert stats["total_completion_tokens"] == 0

    def test_get_stats_with_requests(self, mock_model, mock_tokenizer):
        """Test get_stats() reflects added requests."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        for i in range(3):
            request = Request(
                request_id=f"test-{i}",
                prompt=f"Prompt {i}",
                sampling_params=SamplingParams(),
            )
            scheduler.add_request(request)

        stats = scheduler.get_stats()

        assert stats["num_waiting"] == 3
        assert stats["num_running"] == 0


class TestSchedulerReset:
    """Tests for Scheduler reset methods."""

    def test_reset_clears_state(self, mock_model, mock_tokenizer):
        """Test reset() clears all scheduler state."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        # Add some requests
        for i in range(3):
            request = Request(
                request_id=f"test-{i}",
                prompt=f"Prompt {i}",
                sampling_params=SamplingParams(),
            )
            scheduler.add_request(request)

        scheduler.reset()

        assert len(scheduler.waiting) == 0
        assert len(scheduler.running) == 0
        assert len(scheduler.requests) == 0
        assert scheduler.batch_generator is None


class TestSchedulerStopTokens:
    """Tests for stop token handling."""

    def test_get_stop_tokens(self, mock_model, mock_tokenizer):
        """Test _get_stop_tokens() retrieves EOS token."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        stop_tokens = scheduler._get_stop_tokens()

        # MockTokenizer has eos_token_id = 2
        assert mock_tokenizer.eos_token_id in stop_tokens


class TestSchedulerFormatBytes:
    """Tests for Scheduler._format_bytes()."""

    def test_format_bytes_bytes(self):
        """Test formatting bytes."""
        assert Scheduler._format_bytes(100) == "100 B"
        assert Scheduler._format_bytes(1023) == "1023 B"

    def test_format_bytes_kilobytes(self):
        """Test formatting kilobytes."""
        result = Scheduler._format_bytes(1024)
        assert "KB" in result

        result = Scheduler._format_bytes(2048)
        assert "2.00 KB" in result

    def test_format_bytes_megabytes(self):
        """Test formatting megabytes."""
        result = Scheduler._format_bytes(1024 * 1024)
        assert "MB" in result

        result = Scheduler._format_bytes(5 * 1024 * 1024)
        assert "5.00 MB" in result

    def test_format_bytes_gigabytes(self):
        """Test formatting gigabytes."""
        result = Scheduler._format_bytes(1024 * 1024 * 1024)
        assert "GB" in result

        result = Scheduler._format_bytes(2 * 1024 * 1024 * 1024)
        assert "2.00 GB" in result


class TestSchedulerRemoveFinishedRequest:
    """Tests for Scheduler.remove_finished_request()."""

    def test_remove_finished_request(self, mock_model, mock_tokenizer):
        """Test removing a finished request from tracking."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        request = Request(
            request_id="test-001",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        scheduler.add_request(request)

        removed = scheduler.remove_finished_request("test-001")

        assert removed is request
        assert "test-001" not in scheduler.requests

    def test_remove_nonexistent_request(self, mock_model, mock_tokenizer):
        """Test removing nonexistent request returns None."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        result = scheduler.remove_finished_request("nonexistent")

        assert result is None


class TestSchedulerBoundarySnapshots:
    """Tests for boundary cache snapshots on non-sliceable cache models."""

    def test_capture_boundary_snapshot_at_block_boundary(self, mock_model, mock_tokenizer):
        """Capture snapshot when total tokens land exactly on block boundary."""
        config = SchedulerConfig(paged_cache_block_size=4)
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer, config=config)
        scheduler.block_aware_cache = MagicMock()
        scheduler._boundary_snapshot_required = True

        mock_batch = MagicMock()
        mock_batch.uids = [123]
        snapshot_cache = [MagicMock()]
        mock_batch.extract_cache.return_value = snapshot_cache

        scheduler.batch_generator = MagicMock()
        scheduler.batch_generator.active_batch = mock_batch

        request = Request(
            request_id="req-boundary",
            prompt="hello",
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = [10, 11]
        request.num_prompt_tokens = 2
        request.output_token_ids = [12, 13]  # Total = 4 (boundary)

        scheduler._maybe_capture_boundary_snapshot(request, 123)

        assert scheduler._boundary_cache_snapshots["req-boundary"][0] == snapshot_cache
        assert scheduler._boundary_cache_snapshots["req-boundary"][1] == 4
        mock_batch.extract_cache.assert_called_once_with(0)

    def test_cleanup_finished_uses_boundary_snapshot_for_partial_trailing_tokens(
        self, mock_model, mock_tokenizer
    ):
        """When final length has trailing partial tokens, store boundary snapshot."""
        config = SchedulerConfig(paged_cache_block_size=4)
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer, config=config)
        scheduler.block_aware_cache = MagicMock()
        scheduler.paged_cache_manager = None

        request = Request(
            request_id="req-partial",
            prompt="hello",
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = [1, 2, 3, 4]
        request.num_prompt_tokens = 4
        request.output_token_ids = [5, 6, 7]  # Total = 7 (partial trailing block)
        request._extracted_cache = [{"state": "final-cache"}]
        request._model_cache_config = "final-config"

        scheduler.running["req-partial"] = request
        scheduler.requests["req-partial"] = request
        scheduler._boundary_cache_snapshots["req-partial"] = ([MagicMock()], 4)

        snapshot_extracted = [{"state": "boundary-cache"}]
        with patch.object(
            scheduler,
            "_extract_cache_states",
            return_value=(snapshot_extracted, "boundary-config"),
        ):
            scheduler._cleanup_finished({"req-partial"})

        scheduler.block_aware_cache.store_cache.assert_called_once()
        args, kwargs = scheduler.block_aware_cache.store_cache.call_args
        assert args[0] == "req-partial"
        assert args[1] == [1, 2, 3, 4]
        assert args[2] == snapshot_extracted
        assert kwargs["model_cache_config"] == "boundary-config"
        assert "req-partial" not in scheduler._boundary_cache_snapshots

    def test_boundary_snapshot_synchronizes_generation_stream(
        self, mock_model, mock_tokenizer
    ):
        """Boundary snapshot extraction must synchronize generation_stream
        before accessing batch cache tensors to prevent Metal command buffer conflicts."""
        config = SchedulerConfig(paged_cache_block_size=4)
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer, config=config)
        scheduler.block_aware_cache = MagicMock()
        scheduler._boundary_snapshot_required = True

        mock_batch = MagicMock()
        mock_batch.uids = [42]
        mock_batch.extract_cache.return_value = [MagicMock()]

        scheduler.batch_generator = MagicMock()
        scheduler.batch_generator.active_batch = mock_batch

        request = Request(
            request_id="req-sync",
            prompt="hello",
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = [1, 2]
        request.num_prompt_tokens = 2
        request.output_token_ids = [3, 4]  # Total = 4 (boundary)

        with patch("omlx.scheduler.mx") as mock_mx:
            scheduler._maybe_capture_boundary_snapshot(request, 42)
            mock_mx.synchronize.assert_called()
            mock_mx.stream.assert_called()

    def test_cleanup_finished_synchronizes_before_cache_store(
        self, mock_model, mock_tokenizer
    ):
        """_cleanup_finished must synchronize generation_stream before cache
        storage even when active_batch is None (all requests finished)."""
        config = SchedulerConfig(paged_cache_block_size=4)
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer, config=config)
        scheduler.block_aware_cache = MagicMock()
        scheduler.paged_cache_manager = None

        # Simulate active_batch = None (all requests finished in this step)
        scheduler.batch_generator = MagicMock()
        scheduler.batch_generator.active_batch = None

        request = Request(
            request_id="req-cleanup-sync",
            prompt="hello",
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = [1, 2, 3, 4]
        request.num_prompt_tokens = 4
        request.output_token_ids = [5]
        request._extracted_cache = [{"state": "cache"}]
        request._model_cache_config = None

        scheduler.running["req-cleanup-sync"] = request
        scheduler.requests["req-cleanup-sync"] = request

        with patch("omlx.scheduler.mx") as mock_mx:
            scheduler._cleanup_finished({"req-cleanup-sync"})
            mock_mx.synchronize.assert_called()
            mock_mx.stream.assert_called()

    def test_prefill_boundary_snapshot_records_rotating_cache(
        self, mock_model, mock_tokenizer
    ):
        """Prefill callback should store rotating boundary snapshots."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=SchedulerConfig(paged_cache_block_size=4),
        )
        scheduler.block_aware_cache = MagicMock()

        request = Request(
            request_id="req-prefill-boundary",
            prompt="hello",
            sampling_params=SamplingParams(),
        )
        uid = 77
        scheduler.requests[request.request_id] = request
        scheduler.running[request.request_id] = request
        scheduler.request_id_to_uid[request.request_id] = uid
        scheduler.uid_to_request_id[uid] = request.request_id

        RotatingStub = type("RotatingKVCache", (), {})
        snapshot_cache = [RotatingStub()]

        scheduler._on_prefill_boundary_snapshot(uid, snapshot_cache, 4)

        assert scheduler._boundary_cache_snapshots[request.request_id][0] == snapshot_cache
        assert scheduler._boundary_cache_snapshots[request.request_id][1] == 4
        assert scheduler._boundary_snapshot_required is True

    def test_prefill_boundary_snapshot_ignores_non_boundary_token_count(
        self, mock_model, mock_tokenizer
    ):
        """Prefill callback should ignore non-boundary token counts."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=SchedulerConfig(paged_cache_block_size=4),
        )
        scheduler.block_aware_cache = MagicMock()

        request = Request(
            request_id="req-prefill-non-boundary",
            prompt="hello",
            sampling_params=SamplingParams(),
        )
        uid = 78
        scheduler.requests[request.request_id] = request
        scheduler.running[request.request_id] = request
        scheduler.request_id_to_uid[request.request_id] = uid
        scheduler.uid_to_request_id[uid] = request.request_id

        RotatingStub = type("RotatingKVCache", (), {})
        scheduler._on_prefill_boundary_snapshot(uid, [RotatingStub()], 3)

        assert request.request_id not in scheduler._boundary_cache_snapshots


class TestSchedulerRotatingBlockAlignment:
    """Tests for rotating window/block-size alignment."""

    def test_aligns_block_size_to_rotating_window(self, mock_tokenizer):
        RotatingStub = type("RotatingKVCache", (), {})

        class RotatingModel:
            def __init__(self):
                self.config = MagicMock()
                self.config.num_hidden_layers = 1

            def make_cache(self):
                cache = RotatingStub()
                cache.max_size = 128
                return [cache]

        scheduler = Scheduler(
            model=RotatingModel(),
            tokenizer=mock_tokenizer,
            config=SchedulerConfig(paged_cache_block_size=256),
        )
        scheduler.config.paged_ssd_cache_dir = "/tmp/cache"
        scheduler._align_block_size_with_rotating_window()

        assert scheduler.config.paged_cache_block_size == 128

    def test_multiple_rotating_window_sizes_raise(self, mock_tokenizer):
        RotatingStub = type("RotatingKVCache", (), {})

        class MultiRotatingModel:
            def __init__(self):
                self.config = MagicMock()
                self.config.num_hidden_layers = 2

            def make_cache(self):
                c1 = RotatingStub()
                c1.max_size = 128
                c2 = RotatingStub()
                c2.max_size = 256
                return [c1, c2]

        scheduler = Scheduler(
            model=MultiRotatingModel(),
            tokenizer=mock_tokenizer,
            config=SchedulerConfig(paged_cache_block_size=256),
        )
        scheduler.config.paged_ssd_cache_dir = "/tmp/cache"

        with pytest.raises(ValueError):
            scheduler._align_block_size_with_rotating_window()

    def test_cleanup_finished_skips_remove_when_uid_not_in_active_batch(
        self, mock_model, mock_tokenizer
    ):
        """_cleanup_finished should not call remove() for already-filtered UIDs."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        request = Request(
            request_id="req-skip-remove",
            prompt="hello",
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = [1, 2]
        request.num_prompt_tokens = 2
        request.output_token_ids = [3]

        uid = 55
        scheduler.running["req-skip-remove"] = request
        scheduler.requests["req-skip-remove"] = request
        scheduler.request_id_to_uid["req-skip-remove"] = uid
        scheduler.uid_to_request_id[uid] = "req-skip-remove"

        scheduler.batch_generator = MagicMock()
        scheduler.batch_generator.active_batch = MagicMock(uids=[77])

        scheduler._cleanup_finished({"req-skip-remove"})

        scheduler.batch_generator.remove.assert_not_called()

    def test_cleanup_finished_removes_on_generation_stream_when_uid_active(
        self, mock_model, mock_tokenizer
    ):
        """_cleanup_finished should remove active UID on generation_stream."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        request = Request(
            request_id="req-remove-active",
            prompt="hello",
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = [1, 2]
        request.num_prompt_tokens = 2
        request.output_token_ids = [3]

        uid = 56
        scheduler.running["req-remove-active"] = request
        scheduler.requests["req-remove-active"] = request
        scheduler.request_id_to_uid["req-remove-active"] = uid
        scheduler.uid_to_request_id[uid] = "req-remove-active"

        scheduler.batch_generator = MagicMock()
        scheduler.batch_generator.active_batch = MagicMock(uids=[uid])

        with patch("omlx.scheduler.mx") as mock_mx:
            scheduler._cleanup_finished({"req-remove-active"})

        scheduler.batch_generator.remove.assert_called_once_with([uid])
        mock_mx.stream.assert_called_once()


class TestExtractCacheStatesCacheList:
    """Tests for CacheList handling in _extract_cache_states."""

    @pytest.fixture
    def scheduler(self):
        """Create a minimal scheduler mock for testing _extract_cache_states."""
        from omlx.scheduler import Scheduler

        mock_scheduler = MagicMock(spec=Scheduler)
        mock_scheduler.model_name = "test"
        mock_scheduler._extract_cache_states = Scheduler._extract_cache_states.__get__(
            mock_scheduler, Scheduler
        )
        return mock_scheduler

    def test_extract_cache_states_cache_list(self, scheduler):
        """Test CacheList layer extraction."""
        # Create a mock CacheList object
        mock_kv_sub = MagicMock(spec=[])
        mock_kv_sub.__class__ = type("KVCache", (), {})
        mock_kv_sub.state = (MagicMock(), MagicMock())
        mock_kv_sub.meta_state = (32,)

        mock_cache_list = MagicMock(spec=[])
        mock_cache_list.__class__ = type("CacheList", (), {})
        mock_cache_list.caches = (mock_kv_sub,)
        mock_cache_list.state = [(MagicMock(), MagicMock())]  # CacheList.state
        mock_cache_list.meta_state = (["KVCache"], [(32,)])

        # Standard KVCache layer
        mock_kv = MagicMock(spec=[])
        mock_kv.__class__ = type("KVCache", (), {})
        mock_kv.state = (MagicMock(), MagicMock())
        mock_kv.meta_state = (64,)

        raw_cache = [mock_cache_list, mock_kv]

        extracted, config = scheduler._extract_cache_states(raw_cache)

        assert len(extracted) == 2
        assert extracted[0]['class_name'] == 'CacheList'
        assert extracted[0]['cache_type'] == 'CacheList'
        assert isinstance(extracted[0]['state'], list)
        assert isinstance(extracted[0]['meta_state'], tuple)
        assert len(extracted[0]['meta_state']) == 2

    def test_extract_cache_states_cache_list_no_handlers(self, scheduler):
        """Test CacheList extraction when HAS_CACHE_TYPE_HANDLERS=False."""
        # Use real stub classes so type(obj).__name__ returns the correct name
        # (needed because the fallback branch uses type().__name__ for detection)
        KVCacheStub = type("KVCache", (), {
            "state": (MagicMock(), MagicMock()),
            "meta_state": (32,),
        })
        mock_kv_sub = KVCacheStub()

        CacheListStub = type("CacheList", (), {
            "caches": (mock_kv_sub,),
            "state": [(MagicMock(), MagicMock())],
            "meta_state": (["KVCache"], [(32,)]),
        })
        mock_cache_list = CacheListStub()

        raw_cache = [mock_cache_list]

        # Patch HAS_CACHE_TYPE_HANDLERS to False
        with patch('omlx.scheduler.HAS_CACHE_TYPE_HANDLERS', False):
            extracted, config = scheduler._extract_cache_states(raw_cache)

        # Must still have 1 extracted entry (Issue #1: no layer count mismatch)
        assert len(extracted) == 1
        assert extracted[0]['class_name'] == 'CacheList'
        assert isinstance(extracted[0]['state'], list)


class TestExtractCacheStatesRotatingNormalization:
    """Tests for RotatingKVCache snapshot normalization during extraction."""

    def test_extract_cache_states_normalizes_oversized_rotating_snapshot(
        self, mock_model, mock_tokenizer
    ):
        """Oversized rotating snapshot should be canonicalized to max_size."""
        mx = pytest.importorskip("mlx.core")
        cache_mod = pytest.importorskip("mlx_lm.models.cache")
        RotatingKVCache = cache_mod.RotatingKVCache

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        rotating = RotatingKVCache(max_size=128, keep=0)
        rotating.keys = mx.arange(255).reshape(1, 1, 255, 1)
        rotating.values = mx.arange(1000, 1255).reshape(1, 1, 255, 1)
        rotating.offset = 1280
        rotating._idx = 255

        expected_keys = rotating.keys[..., -128:, :]
        expected_values = rotating.values[..., -128:, :]

        extracted, _ = scheduler._extract_cache_states([rotating])

        assert len(extracted) == 1
        normalized_keys, normalized_values = extracted[0]["state"]
        normalized_meta = tuple(extracted[0]["meta_state"])

        assert normalized_keys.shape == (1, 1, 128, 1)
        assert normalized_values.shape == (1, 1, 128, 1)
        assert bool(mx.all(normalized_keys == expected_keys).item())
        assert bool(mx.all(normalized_values == expected_values).item())
        assert normalized_meta == ("0", "128", "1280", "128")
