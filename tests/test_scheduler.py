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

import mlx.core as mx
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
        assert config.completion_batch_size == 32
        assert config.prefill_step_size == 2048
        assert config.paged_cache_block_size == 256
        assert config.max_cache_blocks is None
        assert config.initial_cache_blocks == 256
        assert config.paged_ssd_cache_dir is None
        assert config.paged_ssd_cache_max_size == 100 * 1024 * 1024 * 1024  # 100GB
        assert config.model_name == ""
        assert config.gc_cleanup_interval == 0
        assert config.mlx_cache_cleanup_interval == 512

    def test_custom_values(self):
        """Test SchedulerConfig with custom values."""
        config = SchedulerConfig(
            max_num_seqs=128,
            max_num_batched_tokens=4096,
            policy=SchedulingPolicy.PRIORITY,
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
        # Scheduler deep-copies tokenizer for thread safety (Rust RefCell
        # isolation between event loop and MLX executor threads).
        assert scheduler.tokenizer is not mock_tokenizer
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
        )
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=config,
        )

        assert scheduler.config.max_num_seqs == 64

    def test_init_statistics_zero(self, mock_model, mock_tokenizer):
        """Test Scheduler initializes with zero statistics."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        assert scheduler.num_requests_processed == 0
        assert scheduler.total_prompt_tokens == 0
        assert scheduler.total_completion_tokens == 0

    def test_snapshot_for_admin_is_isolated_from_live_state(
        self, mock_model, mock_tokenizer
    ):
        """Published admin snapshot must not mutate when live state changes."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        request = Request(
            request_id="req-snap",
            prompt=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=8),
        )
        request.prompt_token_ids = [1, 2, 3]
        request.num_prompt_tokens = 3

        scheduler.waiting.append(request)
        scheduler.running["req-snap"] = request
        scheduler._publish_admin_snapshot()

        snap = scheduler.snapshot_for_admin()
        assert snap["running_by_id"] == {"req-snap": request}
        assert snap["waiting"] == [request]

        scheduler.running.clear()
        scheduler.waiting.clear()
        # Snapshot reflects the published moment, not the live state.
        assert snap["running_by_id"] == {"req-snap": request}
        assert snap["waiting"] == [request]


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

    def test_add_request_exact_cache_hit_trims_one_token(
        self, mock_model, mock_tokenizer
    ):
        """Exact cache hit should use (N-1) cache + last token for kickoff."""
        from omlx.cache.paged_cache import BlockTable

        class TrimCache:
            def __init__(self):
                self.trim_calls = 0

            def trim(self, n):
                self.trim_calls += 1
                return n

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.block_aware_cache = MagicMock()
        scheduler.paged_cache_manager = MagicMock()

        block_table = BlockTable(request_id="req-exact", block_ids=[1, 2], num_tokens=4)
        trim_cache_a = TrimCache()
        trim_cache_b = TrimCache()

        scheduler.block_aware_cache.fetch_cache.return_value = (block_table, [])
        scheduler.block_aware_cache.reconstruct_cache.return_value = [trim_cache_a, trim_cache_b]

        request = Request(
            request_id="req-exact",
            prompt=[11, 12, 13, 14],
            sampling_params=SamplingParams(max_tokens=16),
        )

        scheduler.add_request(request)

        assert request.cached_tokens == 3
        assert request.remaining_tokens == [14]
        assert request.prompt_cache is not None
        assert trim_cache_a.trim_calls == 1
        assert trim_cache_b.trim_calls == 1

    def test_add_request_exact_cache_hit_falls_back_if_not_trimmable(
        self, mock_model, mock_tokenizer
    ):
        """Exact cache hit should fallback when any layer cannot trim."""
        from omlx.cache.paged_cache import BlockTable

        class NonTrimmableCache:
            pass

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.block_aware_cache = MagicMock()
        scheduler.paged_cache_manager = MagicMock()

        block_table = BlockTable(request_id="req-fallback", block_ids=[3], num_tokens=4)
        scheduler.block_aware_cache.fetch_cache.return_value = (block_table, [])
        scheduler.block_aware_cache.reconstruct_cache.return_value = [NonTrimmableCache()]

        request = Request(
            request_id="req-fallback",
            prompt=[21, 22, 23, 24],
            sampling_params=SamplingParams(max_tokens=16),
        )

        scheduler.add_request(request)

        assert request.cached_tokens == 0
        assert request.remaining_tokens == [21, 22, 23, 24]
        assert request.prompt_cache is None
        scheduler.paged_cache_manager.delete_block_table.assert_called_once_with("req-fallback")

    def test_add_request_exact_cache_hit_rotating_forces_fallback(
        self, mock_model, mock_tokenizer
    ):
        """Rotating cache exact hit should fallback to full prefill."""
        from omlx.cache.paged_cache import BlockTable

        RotatingCacheWithTrim = type(
            "RotatingKVCache",
            (),
            {"trim": lambda self, n: n},
        )

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.block_aware_cache = MagicMock()
        scheduler.paged_cache_manager = MagicMock()

        block_table = BlockTable(request_id="req-rotating", block_ids=[9], num_tokens=4)
        scheduler.block_aware_cache.fetch_cache.return_value = (block_table, [])
        scheduler.block_aware_cache.reconstruct_cache.return_value = [RotatingCacheWithTrim()]

        request = Request(
            request_id="req-rotating",
            prompt=[31, 32, 33, 34],
            sampling_params=SamplingParams(max_tokens=16),
        )

        scheduler.add_request(request)

        assert request.cached_tokens == 0
        assert request.remaining_tokens == [31, 32, 33, 34]
        assert request.prompt_cache is None
        scheduler.paged_cache_manager.delete_block_table.assert_called_once_with("req-rotating")


class TestSchedulerAbortRequest:
    """Tests for Scheduler.abort_request() (deferred abort pattern)."""

    def test_abort_enqueues_request(self, mock_model, mock_tokenizer):
        """Test abort_request() enqueues for deferred processing."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        request = Request(
            request_id="test-001",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        scheduler.add_request(request)

        result = scheduler.abort_request("test-001")

        # abort_request always returns True (enqueue is always successful)
        assert result is True
        # Request should still be in waiting (not yet processed)
        assert "test-001" in scheduler._pending_abort_ids

    def test_abort_waiting_request(self, mock_model, mock_tokenizer):
        """Test aborting a waiting request via deferred processing."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        request = Request(
            request_id="test-001",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        scheduler.add_request(request)

        scheduler.abort_request("test-001")
        scheduler._process_pending_aborts()

        assert request.status == RequestStatus.FINISHED_ABORTED
        assert request not in scheduler.waiting
        assert "test-001" in scheduler.finished_req_ids

    def test_abort_nonexistent_request(self, mock_model, mock_tokenizer):
        """Test aborting a non-existent request is silently ignored."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        result = scheduler.abort_request("nonexistent")
        # Enqueue always succeeds
        assert result is True
        # Processing a non-existent abort is a no-op
        scheduler._process_pending_aborts()

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
        scheduler._process_pending_aborts()

        assert request.get_finish_reason() == "abort"

    def test_abort_running_request_removes_from_batch(
        self, mock_model, mock_tokenizer
    ):
        """Abort must remove active UID from BatchGenerator."""
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

        scheduler.abort_request("req-run")
        scheduler._process_pending_aborts()

        scheduler.batch_generator.remove.assert_called_once_with([uid])

    def test_abort_running_request_always_calls_remove(
        self, mock_model, mock_tokenizer
    ):
        """Abort always calls remove() — BatchGenerator handles unknown UIDs."""
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

        scheduler.abort_request("req-run-missing")
        scheduler._process_pending_aborts()

        scheduler.batch_generator.remove.assert_called_once_with([uid])

    def test_abort_cleans_all_scheduler_state(self, mock_model, mock_tokenizer):
        """Abort must clean running, uid mappings, and requests dict.

        Regression test: previously _cleanup_request (engine_core) removed
        the request from self.requests before the deferred abort ran,
        causing _do_abort_request to early-return and leave ghost state
        in running/uid mappings/active batch.
        """
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        request = Request(
            request_id="req-ghost",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = [1]
        request.num_prompt_tokens = 1
        request.status = RequestStatus.RUNNING

        uid = 10
        scheduler.requests["req-ghost"] = request
        scheduler.running["req-ghost"] = request
        scheduler.request_id_to_uid["req-ghost"] = uid
        scheduler.uid_to_request_id[uid] = "req-ghost"

        scheduler.batch_generator = MagicMock()
        scheduler.batch_generator.active_batch = MagicMock(uids=[uid])

        scheduler.abort_request("req-ghost")
        scheduler._process_pending_aborts()

        # All scheduler state must be cleaned
        assert "req-ghost" not in scheduler.running
        assert "req-ghost" not in scheduler.requests
        assert "req-ghost" not in scheduler.request_id_to_uid
        assert uid not in scheduler.uid_to_request_id


class TestPrefillAbortInterrupt:
    """Tests for prefill abort interrupt via _check_pending_aborts_for_uids."""

    def test_check_pending_aborts_returns_aborted_uids(
        self, mock_model, mock_tokenizer
    ):
        """_check_pending_aborts_for_uids returns UIDs with pending aborts."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        # Set up UID mapping
        scheduler.uid_to_request_id[0] = "req-a"
        scheduler.uid_to_request_id[1] = "req-b"
        scheduler._pending_abort_ids.add("req-a")

        result = scheduler._check_pending_aborts_for_uids([0, 1])
        assert result == [0]

    def test_check_pending_aborts_empty_when_no_aborts(
        self, mock_model, mock_tokenizer
    ):
        """Returns empty list when no pending aborts."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.uid_to_request_id[0] = "req-a"

        result = scheduler._check_pending_aborts_for_uids([0])
        assert result == []

    def test_prefill_aborted_error_resets_batch_generator(
        self, mock_model, mock_tokenizer
    ):
        """_PrefillAbortedError in step() resets batch_generator to None."""
        from omlx.scheduler import _PrefillAbortedError

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.batch_generator = MagicMock()

        # Make next_generated() raise _PrefillAbortedError
        # (simulates abort during external prefill in _schedule_waiting)
        scheduler.batch_generator.next_generated.side_effect = _PrefillAbortedError(
            [0], 1024
        )
        # Need running requests for next_generated() to be called
        request = Request(
            request_id="req-prefill",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = [1]
        request.num_prompt_tokens = 1
        request.status = RequestStatus.RUNNING
        scheduler.running["req-prefill"] = request
        scheduler.requests["req-prefill"] = request

        output = scheduler.step()

        # batch_generator should be reset
        assert scheduler.batch_generator is None
        # Request should be moved back to waiting
        assert "req-prefill" not in scheduler.running
        assert len(scheduler.waiting) > 0


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


class TestSchedulerXtcSpecialTokens:
    """Tests for _get_xtc_special_tokens()."""

    def test_includes_newline_and_eos(self, mock_model, mock_tokenizer):
        """Test that XTC special tokens include newline encoding and EOS."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        tokens = scheduler._get_xtc_special_tokens()

        # Should include tokens from encoding "\n"
        newline_tokens = mock_tokenizer.encode("\n")
        for t in newline_tokens:
            assert t in tokens

        # Should include eos_token_id (MockTokenizer has eos_token_id=2)
        assert mock_tokenizer.eos_token_id in tokens

    def test_includes_eos_token_ids_plural(self, mock_model, mock_tokenizer):
        """Test that eos_token_ids (plural) is used when available."""
        mock_tokenizer.eos_token_ids = [2, 100, 200]
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        tokens = scheduler._get_xtc_special_tokens()

        for eos_id in [2, 100, 200]:
            assert eos_id in tokens

    def test_falls_back_to_singular_eos(self, mock_model, mock_tokenizer):
        """Test fallback to eos_token_id when eos_token_ids is absent."""
        # MockTokenizer has eos_token_id=2 but no eos_token_ids
        assert not hasattr(mock_tokenizer, 'eos_token_ids')
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        tokens = scheduler._get_xtc_special_tokens()

        assert 2 in tokens


class TestSyncAndClearCache:
    """Tests for module-level _sync_and_clear_cache() helper (#300, #888)."""

    def test_swallows_generation_stream_thread_error(self):
        """generation_stream sync failing must not break cache clear.

        Reproduces #888: on some MLX builds mx.synchronize(generation_stream)
        raises 'There is no Stream(gpu, 0) in current thread' when called
        from an executor thread that has not submitted work to that stream
        (e.g. during _do_external_prefill). The helper must swallow that
        RuntimeError and still drain the default stream + clear the cache.
        """
        from omlx import scheduler as sched_mod

        calls = []

        def fake_gen_sync(stream):
            calls.append(("gen_sync", stream))
            raise RuntimeError("There is no Stream(gpu, 0) in current thread.")

        def fake_default_sync():
            calls.append(("default_sync",))

        def fake_clear_cache():
            calls.append(("clear_cache",))

        def dispatch(*args, **kwargs):
            if args:
                fake_gen_sync(args[0])
            else:
                fake_default_sync()

        with patch.object(sched_mod.mx, "synchronize", side_effect=dispatch), \
             patch.object(sched_mod.mx, "clear_cache", side_effect=fake_clear_cache):
            sched_mod._sync_and_clear_cache()

        assert calls[0][0] == "gen_sync"
        assert ("default_sync",) in calls
        assert ("clear_cache",) in calls

    def test_propagates_default_stream_error(self):
        """Errors on the default stream sync are not swallowed."""
        from omlx import scheduler as sched_mod

        def dispatch(*args, **kwargs):
            if not args:
                raise RuntimeError("default stream failure")

        with patch.object(sched_mod.mx, "synchronize", side_effect=dispatch), \
             patch.object(sched_mod.mx, "clear_cache") as clear_cache:
            with pytest.raises(RuntimeError, match="default stream failure"):
                sched_mod._sync_and_clear_cache()
            clear_cache.assert_not_called()


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

        # New API: _extract_boundary_snapshot uses batch_generator.extract_cache()
        # which returns {uid: (cache_list, tokens_list)}.
        mock_layer_cache = MagicMock()
        type(mock_layer_cache).__name__ = "BatchArraysCache"

        scheduler.batch_generator = MagicMock()
        scheduler.batch_generator.extract_cache.return_value = {
            123: ([mock_layer_cache], [10, 11, 12, 13])
        }

        request = Request(
            request_id="req-boundary",
            prompt="hello",
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = [10, 11]
        request.num_prompt_tokens = 2
        request.output_token_ids = [12, 13]  # Total = 4 (boundary)

        scheduler._maybe_capture_boundary_snapshot(request, 123)

        assert 4 in scheduler._boundary_cache_snapshots["req-boundary"]
        snapshot = scheduler._boundary_cache_snapshots["req-boundary"][4]
        # Non-sliceable cache layer is kept as-is in the snapshot
        assert snapshot == [mock_layer_cache]

    def test_cleanup_finished_skips_output_tokens_for_reasoning_model(
        self, mock_model, mock_tokenizer
    ):
        """Reasoning models (needs_think_prefix=True) should store only prompt tokens."""
        config = SchedulerConfig(paged_cache_block_size=4)
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer, config=config)
        scheduler.block_aware_cache = MagicMock()
        scheduler.paged_cache_manager = None

        request = Request(
            request_id="req-reasoning",
            prompt="hello",
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        request.num_prompt_tokens = 8
        request.output_token_ids = [9, 10, 11, 12]
        request.needs_think_prefix = True
        request._extracted_cache = [{"state": "cache"}]
        request._model_cache_config = None

        scheduler.running["req-reasoning"] = request
        scheduler.requests["req-reasoning"] = request

        scheduler._cleanup_finished({"req-reasoning"})

        scheduler.block_aware_cache.store_cache.assert_called_once()
        args, kwargs = scheduler.block_aware_cache.store_cache.call_args
        assert args[0] == "req-reasoning"
        assert args[1] == [1, 2, 3, 4, 5, 6, 7, 8]  # prompt only

    def test_cleanup_finished_stores_output_tokens_for_non_reasoning_model(
        self, mock_model, mock_tokenizer
    ):
        """Non-reasoning models should store prompt + output tokens."""
        config = SchedulerConfig(paged_cache_block_size=4)
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer, config=config)
        scheduler.block_aware_cache = MagicMock()
        scheduler.paged_cache_manager = None

        request = Request(
            request_id="req-nonreasoning",
            prompt="hello",
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = [1, 2, 3, 4]
        request.num_prompt_tokens = 4
        request.output_token_ids = [5, 6, 7, 8]
        request._extracted_cache = [{"state": "cache"}]
        request._model_cache_config = None

        scheduler.running["req-nonreasoning"] = request
        scheduler.requests["req-nonreasoning"] = request

        scheduler._cleanup_finished({"req-nonreasoning"})

        scheduler.block_aware_cache.store_cache.assert_called_once()
        args, kwargs = scheduler.block_aware_cache.store_cache.call_args
        assert args[1] == [1, 2, 3, 4, 5, 6, 7, 8]  # prompt + output

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
        scheduler._boundary_cache_snapshots["req-partial"] = {4: [MagicMock()]}

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
            # Metal buffer cache clear is now DEFERRED by _DEFERRED_CLEAR_DELAY
            # generation steps to avoid IOKit completeMemory() race (#435).
            # It should NOT be called immediately in _cleanup_finished.
            mock_mx.clear_cache.assert_not_called()
            assert scheduler._deferred_clear_at == (
                scheduler._step_counter + Scheduler._DEFERRED_CLEAR_DELAY
            )

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

        assert 4 in scheduler._boundary_cache_snapshots[request.request_id]
        assert scheduler._boundary_cache_snapshots[request.request_id][4] == snapshot_cache
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

        # window_size=128 is below _ROTATING_BLOCK_SIZE_MIN (512),
        # so it gets rounded up to 512 (smallest multiple of 128 >= 512).
        assert scheduler.config.paged_cache_block_size == 512

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

    def test_cleanup_finished_always_calls_remove_for_mapped_uid(
        self, mock_model, mock_tokenizer
    ):
        """_cleanup_finished should always call remove() for UIDs with mapping,
        regardless of whether the UID is in the active batch.
        The new _remove_uid_from_active_batch unconditionally calls remove()."""
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

        scheduler.batch_generator.remove.assert_called_once_with([uid])

    def test_cleanup_finished_removes_uid_from_active_batch(
        self, mock_model, mock_tokenizer
    ):
        """_cleanup_finished should remove active UID from batch."""
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

        scheduler._cleanup_finished({"req-remove-active"})

        scheduler.batch_generator.remove.assert_called_once_with([uid])

    def test_cleanup_finished_defers_metal_buffer_cache_clear(
        self, mock_model, mock_tokenizer
    ):
        """_cleanup_finished must defer Metal buffer cache clear (#435).

        Immediate mx.clear_cache() after request completion races with
        IOKit's asynchronous completeMemory() callbacks. Instead,
        _cleanup_finished sets _deferred_clear_at so the clear happens
        after _DEFERRED_CLEAR_DELAY generation steps.
        """
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        request = Request(
            request_id="req-clear-cache",
            prompt="hello",
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = [1, 2]
        request.num_prompt_tokens = 2
        request.output_token_ids = [3]

        scheduler.running["req-clear-cache"] = request
        scheduler.requests["req-clear-cache"] = request

        with patch("omlx.scheduler.mx") as mock_mx:
            scheduler._cleanup_finished({"req-clear-cache"})
            # Should NOT clear immediately — deferred to avoid IOKit race
            mock_mx.clear_cache.assert_not_called()
            # Target step should be set for deferred clearing
            assert scheduler._deferred_clear_at == (
                scheduler._step_counter + Scheduler._DEFERRED_CLEAR_DELAY
            )

    def test_cleanup_finished_skips_clear_cache_when_no_finished(
        self, mock_model, mock_tokenizer
    ):
        """_cleanup_finished must not schedule deferred clear when no requests finished."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        with patch("omlx.scheduler.mx") as mock_mx:
            scheduler._cleanup_finished(set())
            mock_mx.clear_cache.assert_not_called()
            assert scheduler._deferred_clear_at is None

    def test_cleanup_finished_extends_deferred_clear_for_concurrent_completions(
        self, mock_model, mock_tokenizer
    ):
        """Concurrent completions must extend the deferred clear window (#557).

        With max_num_seqs > 1, two requests can finish in the same batch
        or in consecutive steps. Each completion must get a full
        _DEFERRED_CLEAR_DELAY window from its own finish step, otherwise
        the second request's KV cache blocks can be re-allocated before
        IOKit finishes completeMemory() callbacks.
        """
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        # Simulate first completion at step 0
        req1 = Request(
            request_id="req-concurrent-1",
            prompt="hello",
            sampling_params=SamplingParams(),
        )
        req1.prompt_token_ids = [1, 2]
        req1.num_prompt_tokens = 2
        req1.output_token_ids = [3]
        scheduler.running["req-concurrent-1"] = req1
        scheduler.requests["req-concurrent-1"] = req1

        with patch("omlx.scheduler.mx"):
            scheduler._cleanup_finished({"req-concurrent-1"})
        first_target = scheduler._deferred_clear_at
        assert first_target == scheduler._step_counter + Scheduler._DEFERRED_CLEAR_DELAY

        # Advance step counter to simulate a later step
        scheduler._step_counter += 3

        # Simulate second completion at step 3
        req2 = Request(
            request_id="req-concurrent-2",
            prompt="world",
            sampling_params=SamplingParams(),
        )
        req2.prompt_token_ids = [4, 5]
        req2.num_prompt_tokens = 2
        req2.output_token_ids = [6]
        scheduler.running["req-concurrent-2"] = req2
        scheduler.requests["req-concurrent-2"] = req2

        with patch("omlx.scheduler.mx"):
            scheduler._cleanup_finished({"req-concurrent-2"})

        # Target must be extended to cover the second completion's full window
        second_target = scheduler._step_counter + Scheduler._DEFERRED_CLEAR_DELAY
        assert scheduler._deferred_clear_at == second_target
        assert scheduler._deferred_clear_at > first_target


class TestPeriodicClearGating:
    """Tests for the conditional periodic clear (#978/#1040 mitigation)."""

    def test_periodic_clear_skipped_when_cache_below_threshold(
        self, mock_model, mock_tokenizer
    ):
        """Periodic clear should NOT fire when MLX buffer pool is small.

        The pre-fix behavior fired every mlx_cache_cleanup_interval steps
        unconditionally, producing IOGPUFamily refcount transitions even
        when there was nothing meaningful to release. After the fix, the
        clear only fires when accumulated cache memory exceeds the
        threshold (memory_limit/3 or absolute 2 GiB floor).
        """
        from omlx import scheduler as sched_mod

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler._step_counter = scheduler.config.mlx_cache_cleanup_interval
        scheduler._memory_limit_bytes = 0  # → use absolute 2 GiB threshold

        # 1 GiB cached, well under the 2 GiB threshold
        with patch.object(
            sched_mod.mx, "get_cache_memory", return_value=1 * 1024**3
        ):
            assert scheduler._should_periodic_clear_cache() is False

    def test_periodic_clear_fires_when_cache_above_threshold(
        self, mock_model, mock_tokenizer
    ):
        """Periodic clear must fire when MLX buffer pool exceeds threshold."""
        from omlx import scheduler as sched_mod

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler._step_counter = scheduler.config.mlx_cache_cleanup_interval
        scheduler._memory_limit_bytes = 0  # → 2 GiB absolute floor

        # 3 GiB cached, exceeds the 2 GiB threshold
        with patch.object(
            sched_mod.mx, "get_cache_memory", return_value=3 * 1024**3
        ):
            assert scheduler._should_periodic_clear_cache() is True

    def test_periodic_clear_threshold_scales_with_memory_limit(
        self, mock_model, mock_tokenizer
    ):
        """Threshold must be max(memory_limit/3, 2 GiB) when limit is set."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        # Limit 30 GiB → threshold 10 GiB (memory_limit / 3)
        scheduler._memory_limit_bytes = 30 * 1024**3
        assert scheduler._periodic_clear_threshold_bytes() == 10 * 1024**3

        # Limit 3 GiB → threshold 2 GiB (floor wins)
        scheduler._memory_limit_bytes = 3 * 1024**3
        assert scheduler._periodic_clear_threshold_bytes() == 2 * 1024**3

        # No limit → 2 GiB absolute floor
        scheduler._memory_limit_bytes = 0
        assert scheduler._periodic_clear_threshold_bytes() == 2 * 1024**3


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


class TestCacheCorruptionRecovery:
    """Tests for cache corruption detection and recovery."""

    def _make_scheduler(self, mock_model, mock_tokenizer):
        """Create a Scheduler with requests in running state."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        for i in range(3):
            req = Request(
                request_id=f"req-{i}",
                prompt=f"test prompt {i}",
                sampling_params=SamplingParams(),
                prompt_token_ids=[1, 2, 3],
                num_prompt_tokens=3,
                status=RequestStatus.RUNNING,
                batch_uid=i,
                remaining_tokens=[1, 2, 3],
            )
            scheduler.running[req.request_id] = req
            scheduler.requests[req.request_id] = req
        return scheduler

    def test_reschedule_resets_all_fields(self, mock_model, mock_tokenizer):
        """Rescheduling must reset all request fields for clean re-prefill."""
        scheduler = self._make_scheduler(mock_model, mock_tokenizer)
        req = scheduler.running["req-0"]
        # Simulate partial generation state
        req.output_token_ids = [10, 20, 30]
        req.output_text = "partial output"
        req.num_computed_tokens = 5
        req.block_table = MagicMock()
        req.shared_prefix_blocks = 4
        req._extracted_cache = MagicMock()
        req._model_cache_config = MagicMock()
        req.think_prefix_sent = True
        req.prompt_cache = MagicMock()
        req.cached_tokens = 10

        scheduler._reschedule_running_requests()

        assert req.status == RequestStatus.WAITING
        assert req.batch_uid is None
        assert req.prompt_cache is None
        assert req.cached_tokens == 0
        assert req.remaining_tokens == [1, 2, 3]
        assert req.block_table is None
        assert req.shared_prefix_blocks == 0
        assert req.output_token_ids == []
        assert req.output_text == ""
        assert req.num_computed_tokens == 0
        assert req._extracted_cache is None
        assert req._model_cache_config is None
        assert req.think_prefix_sent is False

    def test_reschedule_corruption_increments_counter(
        self, mock_model, mock_tokenizer
    ):
        """Corruption reschedule increments per-request retry counter."""
        scheduler = self._make_scheduler(mock_model, mock_tokenizer)

        failed = scheduler._reschedule_running_requests(is_corruption=True)

        assert failed == []
        for req in scheduler.waiting:
            assert req.cache_corruption_retries == 1

    def test_reschedule_corruption_fails_after_max_retries(
        self, mock_model, mock_tokenizer
    ):
        """Requests exceeding max corruption retries are failed."""
        scheduler = self._make_scheduler(mock_model, mock_tokenizer)
        # Set one request near the limit
        scheduler.running["req-1"].cache_corruption_retries = 3

        failed = scheduler._reschedule_running_requests(
            is_corruption=True, max_corruption_retries=3
        )

        assert failed == ["req-1"]
        assert "req-1" not in scheduler.running
        assert "req-1" not in scheduler.requests
        # Other requests should be rescheduled
        waiting_ids = {r.request_id for r in scheduler.waiting}
        assert "req-0" in waiting_ids
        assert "req-2" in waiting_ids

    def test_reschedule_no_corruption_does_not_increment_counter(
        self, mock_model, mock_tokenizer
    ):
        """Non-corruption reschedule (e.g. PrefillAborted) does not touch counter."""
        scheduler = self._make_scheduler(mock_model, mock_tokenizer)

        failed = scheduler._reschedule_running_requests(is_corruption=False)

        assert failed == []
        for req in scheduler.waiting:
            assert req.cache_corruption_retries == 0

    def test_fail_all_requests_clears_everything(
        self, mock_model, mock_tokenizer
    ):
        """fail_all_requests removes all running and waiting requests."""
        scheduler = self._make_scheduler(mock_model, mock_tokenizer)
        # Also add a waiting request
        wait_req = Request(
            request_id="req-wait",
            prompt="waiting",
            sampling_params=SamplingParams(),
            prompt_token_ids=[4, 5],
            num_prompt_tokens=2,
        )
        scheduler.waiting.append(wait_req)
        scheduler.requests[wait_req.request_id] = wait_req

        failed_ids = scheduler.fail_all_requests()

        assert set(failed_ids) == {"req-0", "req-1", "req-2", "req-wait"}
        assert len(scheduler.running) == 0
        assert len(scheduler.waiting) == 0
        assert not scheduler.has_requests()
        # Verify requests dict is also cleaned up (no memory leak)
        for rid in failed_ids:
            assert rid not in scheduler.requests

    def test_fail_all_requests_preserves_cache(
        self, mock_model, mock_tokenizer
    ):
        """fail_all_requests resets batch_generator but preserves block cache."""
        scheduler = self._make_scheduler(mock_model, mock_tokenizer)
        scheduler.batch_generator = MagicMock()
        scheduler.block_aware_cache = MagicMock()

        scheduler.fail_all_requests()

        assert scheduler.batch_generator is None
        assert scheduler._current_sampler_params is None
        # Cache should NOT be cleared (not a corruption error)
        scheduler.block_aware_cache.clear.assert_not_called()


class TestDetectNeedsThinkPrefix:
    """Tests for _detect_needs_think_prefix() method.

    Verifies that <think></think> (disabled thinking) patterns are correctly
    distinguished from <think>\\n (enabled thinking) patterns.
    """

    def _make_scheduler(self, mock_model, think_start_id, think_end_id=None):
        """Create scheduler with think token IDs on the tokenizer."""
        from conftest import MockTokenizer

        tokenizer = MockTokenizer()
        tokenizer.think_start_id = think_start_id
        if think_end_id is not None:
            tokenizer.think_end_id = think_end_id
        return Scheduler(model=mock_model, tokenizer=tokenizer)

    def _make_request(self, prompt_token_ids):
        """Create a request with given prompt token IDs."""
        return Request(
            request_id="test-think",
            prompt="test",
            sampling_params=SamplingParams(),
            prompt_token_ids=list(prompt_token_ids),
            num_prompt_tokens=len(prompt_token_ids),
        )

    def test_enabled_thinking_with_newline(self, mock_model):
        """<think> + \\n at end -> True (enabled thinking, e.g. DeepSeek)."""
        scheduler = self._make_scheduler(mock_model, think_start_id=100, think_end_id=101)
        request = self._make_request([1, 2, 3, 100, 198])  # 198 = \n
        assert scheduler._detect_needs_think_prefix(request) is True

    def test_enabled_thinking_last_token(self, mock_model):
        """<think> as last token -> True."""
        scheduler = self._make_scheduler(mock_model, think_start_id=100, think_end_id=101)
        request = self._make_request([1, 2, 3, 100])
        assert scheduler._detect_needs_think_prefix(request) is True

    def test_disabled_thinking_adjacent(self, mock_model):
        """<think></think> adjacent -> False (disabled, e.g. Nemotron)."""
        scheduler = self._make_scheduler(mock_model, think_start_id=100, think_end_id=101)
        request = self._make_request([1, 2, 3, 100, 101])
        assert scheduler._detect_needs_think_prefix(request) is False

    def test_disabled_thinking_with_prefix(self, mock_model):
        """X <think></think> -> False (disabled with preceding token)."""
        scheduler = self._make_scheduler(mock_model, think_start_id=100, think_end_id=101)
        request = self._make_request([1, 2, 50, 100, 101])
        assert scheduler._detect_needs_think_prefix(request) is False

    def test_no_think_token_in_tail(self, mock_model):
        """No <think> in last 3 tokens -> False."""
        scheduler = self._make_scheduler(mock_model, think_start_id=100, think_end_id=101)
        request = self._make_request([1, 2, 3, 4, 5])
        assert scheduler._detect_needs_think_prefix(request) is False

    def test_no_think_start_id_on_tokenizer(self, mock_model):
        """Tokenizer without think_start_id -> False."""
        from conftest import MockTokenizer

        tokenizer = MockTokenizer()
        scheduler = Scheduler(model=mock_model, tokenizer=tokenizer)
        request = self._make_request([1, 2, 3])
        assert scheduler._detect_needs_think_prefix(request) is False

    def test_empty_prompt(self, mock_model):
        """Empty prompt -> False."""
        scheduler = self._make_scheduler(mock_model, think_start_id=100, think_end_id=101)
        request = self._make_request([])
        assert scheduler._detect_needs_think_prefix(request) is False

    def test_no_think_end_id_still_sets_true(self, mock_model):
        """<think> found but no think_end_id resolvable -> True (safe fallback)."""
        scheduler = self._make_scheduler(mock_model, think_start_id=100)
        request = self._make_request([1, 2, 100, 101])
        assert scheduler._detect_needs_think_prefix(request) is True

    def test_think_start_id_raises_type_error(self, mock_model):
        """Tokenizer whose think_start_id raises TypeError -> False.

        Models like context-1 (harmony parser) have _think_start_tokens=None
        in their mlx-lm tokenizer, causing think_start_id to raise TypeError.
        """
        from unittest.mock import PropertyMock
        from conftest import MockTokenizer

        tokenizer = MockTokenizer()
        type(tokenizer).think_start_id = PropertyMock(
            side_effect=TypeError("object of type 'NoneType' has no len()")
        )
        scheduler = Scheduler(model=mock_model, tokenizer=tokenizer)
        request = self._make_request([1, 2, 3])
        assert scheduler._detect_needs_think_prefix(request) is False


class TestOutputParserSmoke:
    """Smoke tests for scheduler output parser session integration."""

    class _Detokenizer:
        def __init__(self, decode_one):
            self._decode_one = decode_one
            self.last_segment = ""

        def reset(self):
            self.last_segment = ""

        def add_token(self, token_id):
            self.last_segment = self._decode_one(token_id)

        def finalize(self):
            self.last_segment = ""

    class _GemmaTokenizer:
        def __init__(self, token_map):
            self._token_map = token_map
            self.eos_token_id = 2
            self.pad_token_id = 0
            self.bos_token_id = 1

        @property
        def detokenizer(self):
            return TestOutputParserSmoke._Detokenizer(
                lambda token_id: self._token_map[token_id]
            )

        def encode(self, text: str, add_special_tokens: bool = True):
            if text == "\n":
                return [198]
            return [10]

        def decode(self, token_ids, skip_special_tokens: bool = True):
            return "".join(self._token_map.get(token_id, "") for token_id in token_ids)

    def test_gemma4_session_selected_and_markers_hidden(self, mock_model):
        mock_model.config.model_type = "gemma4"
        tokenizer = self._GemmaTokenizer(
            {
                11: "<|channel>",
                12: "thought\n",
                13: "reasoning",
                14: "<channel|>",
                15: "answer",
                16: "<turn|>",
            }
        )
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=tokenizer,
            config=SchedulerConfig(model_name="google/gemma-4b"),
        )

        assert scheduler._output_parser_kind == "gemma4"

        request = Request(
            request_id="gemma-req",
            prompt="prompt",
            sampling_params=SamplingParams(max_tokens=5),
            prompt_token_ids=[1, 2, 3],
            num_prompt_tokens=3,
            status=RequestStatus.RUNNING,
            batch_uid=99,
        )
        scheduler.running[request.request_id] = request
        scheduler.requests[request.request_id] = request
        scheduler.uid_to_request_id[99] = request.request_id
        scheduler.request_id_to_uid[request.request_id] = 99

        responses = [
            type("Resp", (), {"uid": 99, "token": 11, "finish_reason": None})(),
            type("Resp", (), {"uid": 99, "token": 12, "finish_reason": None})(),
            type("Resp", (), {"uid": 99, "token": 13, "finish_reason": None})(),
            type("Resp", (), {"uid": 99, "token": 14, "finish_reason": None})(),
            type("Resp", (), {"uid": 99, "token": 15, "finish_reason": None})(),
            type("Resp", (), {"uid": 99, "token": 16, "finish_reason": "length"})(),
        ]

        outputs, finished_ids = scheduler._process_batch_responses(responses)

        assert finished_ids == {"gemma-req"}
        assert outputs[-1].finished is True
        assert outputs[-1].output_text == "<think>\nreasoning</think>\nanswer"

        full_stream = "".join(output.new_text for output in outputs)
        assert "<|channel>" not in full_stream
        assert "<channel|>" not in full_stream
        assert full_stream == "<think>\nreasoning</think>\nanswer"


class TestVLMPositionStateClearing:
    """Tests for conditional mRoPE position state clearing (#531).

    VLM batches must preserve position state set by get_input_embeddings();
    text-only batches must clear stale VLM position state.
    """

    def _make_vlm_model(self):
        """Create a mock model with clear_vlm_position_state.

        Includes make_cache (returning empty list) so that
        _do_external_prefill can call make_prompt_cache(model)
        without hitting AttributeError on model.layers.
        """
        model = MagicMock(spec=[
            "__call__", "clear_vlm_position_state", "parameters",
            "make_cache",
        ])
        model.clear_vlm_position_state = MagicMock()
        model.make_cache.return_value = []
        return model

    def test_schedule_waiting_preserves_vlm_position_state(
        self, mock_tokenizer
    ):
        """VLM request in _schedule_waiting should NOT clear position state.

        With external prefill, clear_vlm_position_state is called inside
        _do_external_prefill only for text-only requests (vlm_embeds is None).
        VLM requests pass vlm_embeds, so the clear is skipped.
        """
        model = self._make_vlm_model()
        scheduler = Scheduler(model=model, tokenizer=mock_tokenizer)

        # Minimal batch generator mock
        mock_bg = MagicMock()
        mock_bg.insert = MagicMock(return_value=[42])
        scheduler.batch_generator = mock_bg

        request = Request(
            request_id="vlm-001",
            prompt="describe this image",
            sampling_params=SamplingParams(max_tokens=50),
        )
        request.prompt_token_ids = [1, 2, 3, 4, 5]
        request.num_prompt_tokens = 5
        # Use a real mx.array so _do_external_prefill can slice it
        request.vlm_inputs_embeds = mx.zeros((1, 5, 64))

        scheduler.waiting.append(request)
        scheduler.requests[request.request_id] = request

        scheduler._schedule_waiting()

        model.clear_vlm_position_state.assert_not_called()

    def test_schedule_waiting_clears_text_only_position_state(
        self, mock_tokenizer
    ):
        """Text-only request in _schedule_waiting should clear position state.

        With external prefill, clear_vlm_position_state is called inside
        _do_external_prefill when vlm_embeds is None (text-only).
        """
        model = self._make_vlm_model()
        scheduler = Scheduler(model=model, tokenizer=mock_tokenizer)

        mock_bg = MagicMock()
        mock_bg.insert = MagicMock(return_value=[42])
        scheduler.batch_generator = mock_bg

        request = Request(
            request_id="text-001",
            prompt="hello world",
            sampling_params=SamplingParams(max_tokens=50),
        )
        request.prompt_token_ids = [1, 2, 3, 4, 5]
        request.num_prompt_tokens = 5
        # vlm_inputs_embeds is None by default (text-only)

        scheduler.waiting.append(request)
        scheduler.requests[request.request_id] = request

        scheduler._schedule_waiting()

        model.clear_vlm_position_state.assert_called_once()
