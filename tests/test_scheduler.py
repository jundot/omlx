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

Note: BatchGenerator is mocked; step() coverage is limited to targeted paths.
"""

import concurrent.futures
import json
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import mlx.core as mx
import pytest

import omlx.scheduler as scheduler_module
from omlx.request import Request, RequestOutput, RequestStatus, SamplingParams
from omlx.scheduler import (
    Scheduler,
    SchedulerConfig,
    SchedulerOutput,
    SchedulingPolicy,
    _PrefillState,
    _StoreCacheGate,
    _VLMMTPDecodeState,
)


class _ParserStopFactory:
    kind = "test"
    stop_token_ids = set()
    thinking_end_text = None

    def create_session(self, tokenizer):
        return _ParserStopSession()


class _ParserStopSession:
    def process_token(self, token_id):
        from omlx.adapter.output_parser import OutputParserTokenResult

        return OutputParserTokenResult(
            stream_text="",
            visible_text="",
            is_stop=True,
            record_token=False,
        )

    def finalize(self):
        from omlx.adapter.output_parser import OutputParserFinalizeResult

        return OutputParserFinalizeResult()


class TestSchedulerConfig:
    """Tests for SchedulerConfig dataclass."""

    def test_default_values(self):
        """Test SchedulerConfig has correct defaults."""
        config = SchedulerConfig()

        assert config.max_num_seqs == 256
        assert config.max_num_batched_tokens == 8192
        assert config.policy == SchedulingPolicy.FCFS
        assert config.completion_batch_size == 32
        assert config.embedding_batch_size == 32
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
            embedding_batch_size=12,
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
        assert config.embedding_batch_size == 12
        assert config.prefill_step_size == 1024
        assert config.paged_cache_block_size == 128
        assert config.max_cache_blocks == 500
        assert config.initial_cache_blocks == 100
        assert config.paged_ssd_cache_dir == "/tmp/cache"
        assert config.paged_ssd_cache_max_size == 50 * 1024 * 1024 * 1024
        assert config.model_name == "test-model"
        assert config.gc_cleanup_interval == 5
        assert config.mlx_cache_cleanup_interval == 20


class TestVLMExtraSlicing:
    """Tests for VLM prompt-aligned extra kwargs used during external prefill."""

    def test_slice_and_advance_token_type_ids(self):
        """Multimodal token types should stay aligned with inputs_embeds chunks."""
        extra = {
            "mm_token_type_ids": mx.array([[0, 1, 1, 0]]),
            "token_type_ids": mx.array([[0, 1, 1, 0]]),
            "per_layer_inputs": mx.zeros((1, 4, 2, 3)),
            "scalar": mx.array(7),
        }

        sliced = scheduler_module._slice_vlm_extra(extra, 3)
        assert sliced["mm_token_type_ids"].tolist() == [[0, 1, 1]]
        assert sliced["token_type_ids"].tolist() == [[0, 1, 1]]
        assert sliced["per_layer_inputs"].shape == (1, 3, 2, 3)
        assert sliced["scalar"] is extra["scalar"]

        advanced = scheduler_module._advance_vlm_extra(extra, 1)
        assert advanced["mm_token_type_ids"].tolist() == [[1, 1, 0]]
        assert advanced["token_type_ids"].tolist() == [[1, 1, 0]]
        assert advanced["per_layer_inputs"].shape == (1, 3, 2, 3)
        assert advanced["scalar"] is extra["scalar"]


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


class TestSchedulerStepOutputs:
    """Tests for Scheduler.step output assembly."""

    def test_decode_outputs_preserve_prefill_rejections(
        self, mock_model, mock_tokenizer
    ):
        """Decode responses must not overwrite earlier rejection outputs."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        prefill_error = RequestOutput(
            request_id="prefill-failed",
            finished=True,
            finish_reason="error",
            error="Memory limit exceeded during prefill",
        )
        decode_output = RequestOutput(
            request_id="running",
            new_token_ids=[123],
            new_text="x",
        )

        scheduler._schedule_waiting = MagicMock(return_value=([], [prefill_error]))
        scheduler._process_batch_responses = MagicMock(
            return_value=([decode_output], {"running"})
        )
        scheduler._cleanup_finished = MagicMock()

        scheduler.running = {"running": MagicMock()}
        scheduler.batch_generator = MagicMock()
        scheduler.batch_generator.next_generated.return_value = iter([MagicMock()])

        output = scheduler.step()

        assert output.outputs == [prefill_error, decode_output]
        assert output.finished_request_ids == {"running"}
        scheduler._cleanup_finished.assert_called_once_with({"running"})


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

    def test_llama4_effective_cap_is_serial(self, mock_model, mock_tokenizer):
        """Llama 4 uses ChunkedKVCache layers that are serialized for now."""
        mock_model.config.model_type = "llama4"
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=SchedulerConfig(max_num_seqs=8),
        )

        assert scheduler._effective_max_num_seqs() == 1

    def test_init_falls_back_when_paged_ssd_cache_unavailable(
        self, mock_model, mock_tokenizer, tmp_path, monkeypatch, caplog
    ):
        """Unusable SSD cache directories should not leave partial cache state."""
        if not scheduler_module.HAS_TIERED_CACHE:
            pytest.skip("tiered cache modules are unavailable")

        class BrokenPagedSSDCacheManager:
            def __init__(self, *args, **kwargs):
                raise OSError("cache directory is not writable")

        monkeypatch.setattr(
            scheduler_module,
            "PagedSSDCacheManager",
            BrokenPagedSSDCacheManager,
        )

        config = SchedulerConfig(paged_ssd_cache_dir=str(tmp_path / "missing-drive"))

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer, config=config)

        assert scheduler.paged_ssd_cache_manager is None
        assert scheduler.paged_cache_manager is None
        assert scheduler.block_aware_cache is None
        assert scheduler._boundary_snapshot_store is None
        assert scheduler._store_cache_executor is None
        assert scheduler._store_cache_gate is None
        assert "Failed to initialize paged SSD cache" in caplog.text

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

    def _scheduler_with_mock_block_cache(
        self,
        mock_model,
        mock_tokenizer,
        *,
        hot_cache_max_size: int = 1024,
        hot_cache_only: bool = False,
    ):
        config = SchedulerConfig(
            hot_cache_max_size=hot_cache_max_size,
            hot_cache_only=hot_cache_only,
        )
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=config,
        )
        scheduler.block_aware_cache = MagicMock()
        scheduler.paged_cache_manager = MagicMock()
        scheduler.paged_ssd_cache_manager = MagicMock()
        scheduler._prefill_memory_guard = True
        scheduler._memory_limit_bytes = 100
        return scheduler

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
        scheduler.block_aware_cache.reconstruct_cache.return_value = [
            trim_cache_a,
            trim_cache_b,
        ]

        request = Request(
            request_id="req-exact",
            prompt=[11, 12, 13, 14],
            sampling_params=SamplingParams(max_tokens=16),
        )

        scheduler.add_request(request)
        scheduler._prepare_prefix_cache_for_request(request)

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
        scheduler.block_aware_cache.reconstruct_cache.return_value = [
            NonTrimmableCache()
        ]

        request = Request(
            request_id="req-fallback",
            prompt=[21, 22, 23, 24],
            sampling_params=SamplingParams(max_tokens=16),
        )

        scheduler.add_request(request)
        scheduler._prepare_prefix_cache_for_request(request)

        assert request.cached_tokens == 0
        assert request.remaining_tokens == [21, 22, 23, 24]
        assert request.prompt_cache is None
        scheduler.paged_cache_manager.delete_block_table.assert_called_once_with(
            "req-fallback"
        )

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
        scheduler.block_aware_cache.reconstruct_cache.return_value = [
            RotatingCacheWithTrim()
        ]

        request = Request(
            request_id="req-rotating",
            prompt=[31, 32, 33, 34],
            sampling_params=SamplingParams(max_tokens=16),
        )

        scheduler.add_request(request)
        scheduler._prepare_prefix_cache_for_request(request)

        assert request.cached_tokens == 0
        assert request.remaining_tokens == [31, 32, 33, 34]
        assert request.prompt_cache is None
        scheduler.paged_cache_manager.delete_block_table.assert_called_once_with(
            "req-rotating"
        )

    def test_add_request_under_pressure_skips_hot_cache_preload_and_promotion(
        self, mock_model, mock_tokenizer
    ):
        """Memory pressure should bypass optional SSD hot-cache RAM copies."""
        from omlx.cache.paged_cache import BlockTable

        scheduler = self._scheduler_with_mock_block_cache(
            mock_model,
            mock_tokenizer,
        )
        scheduler._current_usage_bytes = MagicMock(return_value=100)

        block_table = BlockTable(
            request_id="req-pressure",
            block_ids=[1],
            num_tokens=2,
        )
        scheduler.block_aware_cache.fetch_cache.return_value = (
            block_table,
            [13, 14],
        )
        scheduler.block_aware_cache.reconstruct_cache.return_value = [MagicMock()]

        request = Request(
            request_id="req-pressure",
            prompt=[11, 12, 13, 14],
            sampling_params=SamplingParams(max_tokens=16),
        )

        scheduler.add_request(request)
        scheduler._prepare_prefix_cache_for_request(request)

        scheduler.block_aware_cache.preload_blocks.assert_not_called()
        scheduler.block_aware_cache.reconstruct_cache.assert_called_once_with(
            block_table,
            promote_to_hot_cache=False,
        )
        scheduler._current_usage_bytes.assert_called_once_with()

    def test_add_request_below_pressure_keeps_hot_cache_preload_and_promotion(
        self, mock_model, mock_tokenizer
    ):
        """Normal memory state should preserve existing hot-cache acceleration."""
        from omlx.cache.paged_cache import BlockTable

        scheduler = self._scheduler_with_mock_block_cache(
            mock_model,
            mock_tokenizer,
        )
        scheduler._current_usage_bytes = MagicMock(return_value=99)

        block_table = BlockTable(
            request_id="req-normal",
            block_ids=[1],
            num_tokens=2,
        )
        scheduler.block_aware_cache.fetch_cache.return_value = (
            block_table,
            [13, 14],
        )
        scheduler.block_aware_cache.reconstruct_cache.return_value = [MagicMock()]

        request = Request(
            request_id="req-normal",
            prompt=[11, 12, 13, 14],
            sampling_params=SamplingParams(max_tokens=16),
        )

        scheduler.add_request(request)
        scheduler._prepare_prefix_cache_for_request(request)

        scheduler.block_aware_cache.preload_blocks.assert_called_once_with(block_table)
        scheduler.block_aware_cache.reconstruct_cache.assert_called_once_with(
            block_table
        )

    def test_add_request_hot_cache_only_ignores_pressure_bypass(
        self, mock_model, mock_tokenizer
    ):
        """hot_cache_only mode must keep RAM hot-cache behavior unchanged."""
        from omlx.cache.paged_cache import BlockTable

        scheduler = self._scheduler_with_mock_block_cache(
            mock_model,
            mock_tokenizer,
            hot_cache_only=True,
        )
        scheduler._current_usage_bytes = MagicMock(return_value=100)

        block_table = BlockTable(
            request_id="req-hot-only",
            block_ids=[1],
            num_tokens=2,
        )
        scheduler.block_aware_cache.fetch_cache.return_value = (
            block_table,
            [13, 14],
        )
        scheduler.block_aware_cache.reconstruct_cache.return_value = [MagicMock()]

        request = Request(
            request_id="req-hot-only",
            prompt=[11, 12, 13, 14],
            sampling_params=SamplingParams(max_tokens=16),
        )

        scheduler.add_request(request)
        scheduler._prepare_prefix_cache_for_request(request)

        scheduler.block_aware_cache.preload_blocks.assert_called_once_with(block_table)
        scheduler.block_aware_cache.reconstruct_cache.assert_called_once_with(
            block_table
        )
        scheduler._current_usage_bytes.assert_not_called()

    def test_admission_defers_for_relevant_inflight_store(
        self, mock_model, mock_tokenizer
    ):
        """A same-conversation request should defer lookup until store finishes."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.block_aware_cache = MagicMock()
        prompt = list(range(9001))
        scheduler.block_aware_cache.fetch_cache.return_value = (None, prompt)

        future = MagicMock()
        future.done.return_value = False
        future.result.return_value = None
        scheduler._inflight_store_futures["req-prev"] = future
        scheduler._inflight_store_info["req-prev"] = (
            scheduler_module._InflightStoreInfo(tokens=list(range(9000)))
        )

        request = Request(
            request_id="req-next",
            prompt=prompt,
            sampling_params=SamplingParams(max_tokens=16),
        )

        scheduler.add_request(request)

        future.result.assert_not_called()
        scheduler.block_aware_cache.fetch_cache.assert_not_called()
        assert scheduler._should_defer_for_cache_freshness(request) is True
        future.result.assert_not_called()
        scheduler.block_aware_cache.fetch_cache.assert_not_called()
        assert request.request_id in scheduler._cache_freshness_waits

        future.done.return_value = True
        assert scheduler._should_defer_for_cache_freshness(request) is False
        scheduler._prepare_prefix_cache_for_request(request)
        scheduler.block_aware_cache.fetch_cache.assert_called_once()

    def test_admission_defers_for_ratio_relevant_inflight_store(
        self, mock_model, mock_tokenizer
    ):
        """A large prompt should defer when the shared prefix is a high ratio."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.block_aware_cache = MagicMock()
        prompt = list(range(9001))
        scheduler.block_aware_cache.fetch_cache.return_value = (None, prompt)

        future = MagicMock()
        future.done.return_value = False
        future.result.return_value = None
        scheduler._inflight_store_futures["req-prev"] = future
        scheduler._inflight_store_info["req-prev"] = (
            scheduler_module._InflightStoreInfo(tokens=list(range(3000)))
        )

        request = Request(
            request_id="req-next",
            prompt=prompt,
            sampling_params=SamplingParams(max_tokens=16),
        )

        scheduler.add_request(request)

        future.result.assert_not_called()
        scheduler.block_aware_cache.fetch_cache.assert_not_called()
        assert scheduler._should_defer_for_cache_freshness(request) is True
        assert request.request_id in scheduler._cache_freshness_waits

    def test_admission_does_not_defer_below_common_and_ratio_thresholds(
        self, mock_model, mock_tokenizer
    ):
        """A moderate shared prefix should not defer below both relevance gates."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.block_aware_cache = MagicMock()
        prompt = list(range(30000))

        future = MagicMock()
        future.done.return_value = False
        future.result.return_value = None
        scheduler._inflight_store_futures["req-prev"] = future
        scheduler._inflight_store_info["req-prev"] = (
            scheduler_module._InflightStoreInfo(tokens=list(range(7000)))
        )

        request = Request(
            request_id="req-next",
            prompt=prompt,
            sampling_params=SamplingParams(max_tokens=16),
        )

        scheduler.add_request(request)

        future.result.assert_not_called()
        scheduler.block_aware_cache.fetch_cache.assert_not_called()
        assert scheduler._should_defer_for_cache_freshness(request) is False
        assert request.request_id not in scheduler._cache_freshness_waits

    def test_admission_does_not_defer_for_short_prompt_even_with_high_ratio(
        self, mock_model, mock_tokenizer
    ):
        """Prompts below the freshness minimum should never wait on store_cache."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.block_aware_cache = MagicMock()
        prompt = list(range(7000))

        future = MagicMock()
        future.done.return_value = False
        future.result.return_value = None
        scheduler._inflight_store_futures["req-prev"] = future
        scheduler._inflight_store_info["req-prev"] = (
            scheduler_module._InflightStoreInfo(tokens=list(range(6000)))
        )

        request = Request(
            request_id="req-next",
            prompt=prompt,
            sampling_params=SamplingParams(max_tokens=16),
        )

        scheduler.add_request(request)

        future.result.assert_not_called()
        scheduler.block_aware_cache.fetch_cache.assert_not_called()
        assert scheduler._should_defer_for_cache_freshness(request) is False
        assert request.request_id not in scheduler._cache_freshness_waits

    def test_admission_defers_for_relevant_store_during_active_work(
        self, mock_model, mock_tokenizer
    ):
        """Active decode/prefill rows should not skip a highly relevant store."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.block_aware_cache = MagicMock()
        prompt = list(range(9001))
        scheduler.block_aware_cache.fetch_cache.return_value = (None, prompt)
        scheduler.running["req-running"] = MagicMock()
        scheduler.prefilling.append(MagicMock())

        future = MagicMock()
        future.done.return_value = False
        future.result.return_value = None
        scheduler._inflight_store_futures["req-prev"] = future
        scheduler._inflight_store_info["req-prev"] = (
            scheduler_module._InflightStoreInfo(tokens=list(range(9000)))
        )

        request = Request(
            request_id="req-next",
            prompt=prompt,
            sampling_params=SamplingParams(max_tokens=16),
        )

        scheduler.add_request(request)

        future.result.assert_not_called()
        scheduler.block_aware_cache.fetch_cache.assert_not_called()
        assert scheduler._should_defer_for_cache_freshness(request) is True

    def test_schedule_waiting_defers_cache_freshness_without_blocking(
        self, mock_model, mock_tokenizer
    ):
        """Freshness waits must defer admission, not block scheduler execution."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.block_aware_cache = MagicMock()
        prompt = list(range(9001))

        future = MagicMock()
        future.done.return_value = False
        scheduler._inflight_store_futures["req-prev"] = future
        scheduler._inflight_store_info["req-prev"] = (
            scheduler_module._InflightStoreInfo(tokens=list(range(9000)))
        )
        scheduler._ensure_batch_generator = MagicMock()

        request = Request(
            request_id="req-next",
            prompt=prompt,
            sampling_params=SamplingParams(max_tokens=16),
        )
        scheduler.add_request(request)

        scheduled, rejected = scheduler._schedule_waiting()

        assert scheduled == []
        assert rejected == []
        assert list(scheduler.waiting) == [request]
        assert request.request_id in scheduler._cache_freshness_waits
        future.result.assert_not_called()
        scheduler.block_aware_cache.fetch_cache.assert_not_called()
        scheduler._ensure_batch_generator.assert_not_called()

    def test_admission_does_not_defer_for_mismatched_extra_keys(
        self, mock_model, mock_tokenizer
    ):
        """Token-only overlap must not delay VLM requests with different cache keys."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.block_aware_cache = MagicMock()
        prompt = list(range(9001))
        scheduler.block_aware_cache.fetch_cache.return_value = (None, prompt)

        future = MagicMock()
        future.done.return_value = False
        scheduler._inflight_store_futures["req-prev"] = future
        scheduler._inflight_store_info["req-prev"] = (
            scheduler_module._InflightStoreInfo(
                tokens=list(range(9000)),
                extra_keys=("image-a",),
                extra_key_token_start=0,
            )
        )

        request = Request(
            request_id="req-next",
            prompt=prompt,
            sampling_params=SamplingParams(max_tokens=16),
            vlm_image_hash="image-b",
            vlm_cache_key_start=0,
        )

        scheduler.add_request(request)

        future.result.assert_not_called()
        assert scheduler._should_defer_for_cache_freshness(request) is False
        scheduler._prepare_prefix_cache_for_request(request)
        scheduler.block_aware_cache.fetch_cache.assert_called_once()

    def test_async_store_cache_worker_forwards_hot_cache_write_back_flag(
        self, mock_model, mock_tokenizer
    ):
        """The async store worker must pass pressure mode to store_cache."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.block_aware_cache = MagicMock()
        scheduler.block_aware_cache.store_cache.return_value = None
        scheduler.paged_cache_manager = MagicMock()
        scheduler.paged_cache_manager.get_block_table.return_value = None

        with patch("omlx.scheduler._safe_sync_stream"):
            scheduler._async_store_cache_worker(
                "req-store",
                [1, 2, 3, 4],
                [],
                None,
                None,
                None,
                None,
                None,
                hot_cache_write_back=False,
            )

        scheduler.block_aware_cache.store_cache.assert_called_once_with(
            "req-store",
            [1, 2, 3, 4],
            [],
            model_cache_config=None,
            boundary_snapshots=None,
            extra_keys=None,
            extra_key_token_start=None,
            extra_key_ranges=None,
            hot_cache_write_back=False,
        )


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

    def test_abort_running_request_removes_from_batch(self, mock_model, mock_tokenizer):
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

    def test_abort_vlm_mtp_request_clears_active_generator(
        self, mock_model, mock_tokenizer
    ):
        """Aborting negative vlm_mtp UIDs must release the serialized drafter."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        request = Request(
            request_id="req-vlm-mtp",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = [1]
        request.num_prompt_tokens = 1
        request.status = RequestStatus.RUNNING

        class ClosableGenerator:
            closed = False

            def __next__(self):
                return 1

            def close(self):
                self.closed = True

        generator = ClosableGenerator()
        uid = -1
        scheduler.requests[request.request_id] = request
        scheduler.running[request.request_id] = request
        scheduler.request_id_to_uid[request.request_id] = uid
        scheduler.uid_to_request_id[uid] = request.request_id
        scheduler._vlm_mtp_active[uid] = _VLMMTPDecodeState(
            generator=generator,
            request=request,
            prompt_cache=[],
            sampler=MagicMock(),
            state_machine=MagicMock(),
            max_tokens=16,
        )
        scheduler.batch_generator = MagicMock()

        scheduler.abort_request(request.request_id)
        scheduler._process_pending_aborts()

        assert uid not in scheduler._vlm_mtp_active
        assert generator.closed is True
        scheduler.batch_generator.remove.assert_not_called()

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

    def test_external_prefill_abort_reclaims_metal_before_raise(
        self, mock_model, mock_tokenizer
    ):
        """Aborted external prefill must clear transients before unwinding."""
        from omlx.scheduler import _PrefillAbortedError

        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=SchedulerConfig(prefill_step_size=2),
        )
        request = Request(
            request_id="req-prefill-abort",
            prompt=[1, 2, 3],
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = [1, 2, 3]
        request.num_prompt_tokens = 3
        cache = [SimpleNamespace(state=mx.array([0]))]

        uid = 42
        scheduler.request_id_to_uid[request.request_id] = uid
        scheduler.uid_to_request_id[uid] = request.request_id
        scheduler._pending_abort_ids.add(request.request_id)

        with patch.object(scheduler_module, "_sync_and_clear_cache") as clear_cache:
            with pytest.raises(_PrefillAbortedError):
                scheduler._do_external_prefill(
                    request,
                    request.prompt_token_ids,
                    cache,
                )

        assert clear_cache.call_args_list == [
            call(scheduler._stream),
            call(scheduler._stream),
        ]
        assert request._prefill_saved_rope_deltas is None

    def test_prefill_abort_cleanup_removes_temp_uid_and_pending_abort(
        self, mock_model, mock_tokenizer
    ):
        """Schedule-time prefill aborts must not leave orphan scheduler state."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        request = Request(
            request_id="req-clean-prefill-abort",
            prompt=[1, 2, 3],
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = [1, 2, 3]
        request.num_prompt_tokens = 3
        scheduler.requests[request.request_id] = request

        temp_uid = id(request)
        scheduler.request_id_to_uid[request.request_id] = temp_uid
        scheduler.uid_to_request_id[temp_uid] = request.request_id
        scheduler._pending_abort_ids.add(request.request_id)

        scheduler._cleanup_prefill_abort_request(request, temp_uid=temp_uid)

        assert request.status == RequestStatus.FINISHED_ABORTED
        assert request.request_id not in scheduler.requests
        assert request.request_id not in scheduler.request_id_to_uid
        assert temp_uid not in scheduler.uid_to_request_id
        assert request.request_id not in scheduler._pending_abort_ids

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

    def test_has_requests_with_pending_async_cleanup(self, mock_model, mock_tokenizer):
        """Async store-cache cleanup keeps the scheduler stepping."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        future = concurrent.futures.Future()
        scheduler._pending_async_removes.append((123, "req-cleanup", future))

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

    def test_reset_clears_async_store_cache_bookkeeping(
        self, mock_model, mock_tokenizer
    ):
        """reset() must drop _pending_async_removes and _inflight_store_futures.

        Regression for #1459: a slow async store_cache worker can leave the
        deferred _drain_pending_async_removes step that nulls req._extracted_cache
        pending. If reset() leaves these two containers populated, the futures
        keep Request references alive and the KV cache stays pinned for the rest
        of the process lifetime.
        """
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        fake_future = MagicMock()
        scheduler._pending_async_removes.append((999, "req-leaked", fake_future))
        scheduler._inflight_store_futures["req-leaked"] = fake_future

        scheduler.reset()

        assert len(scheduler._pending_async_removes) == 0
        assert len(scheduler._inflight_store_futures) == 0

    def test_shutdown_drains_after_bounded_wait(self, mock_model, mock_tokenizer):
        """shutdown() must drain pending removes after the bounded wait.

        Regression for #1459. If the bounded wait completes, every future is
        done and the second drain releases skipped entries. If it does not
        complete, shutdown takes the fatal-exit path instead of leaving a
        partially torn-down engine alive.

        Asserts: drain runs both before and after executor.shutdown(wait=False).
        """
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        fake_executor = MagicMock()
        scheduler._store_cache_executor = fake_executor
        scheduler._store_cache_gate = MagicMock()

        # Seed an inflight future so shutdown() enters the wait branch.
        scheduler._inflight_store_futures["req-slow"] = MagicMock()

        call_order = []
        original_drain = scheduler._drain_pending_async_removes

        def record_drain():
            call_order.append("drain")
            original_drain()

        def record_executor_shutdown(wait=True):
            call_order.append("executor_shutdown")

        fake_executor.shutdown.side_effect = record_executor_shutdown
        scheduler._drain_pending_async_removes = record_drain

        with patch("concurrent.futures.wait", return_value=({object()}, set())):
            scheduler.shutdown()

        assert call_order == [
            "drain",
            "executor_shutdown",
            "drain",
        ], f"Expected drain to bracket executor.shutdown, got: {call_order}"
        fake_executor.shutdown.assert_called_once_with(wait=False)

    def test_shutdown_fatal_exits_when_store_cache_worker_times_out(
        self, mock_model, mock_tokenizer
    ):
        """A stuck store-cache worker is fatal during scheduler shutdown."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        fake_executor = MagicMock()
        scheduler._store_cache_executor = fake_executor
        scheduler._store_cache_gate = MagicMock()
        future = MagicMock()
        scheduler._inflight_store_futures["req-stuck"] = future

        with (
            patch("concurrent.futures.wait", return_value=(set(), {future})),
            patch("omlx.scheduler.fatal_exit", side_effect=SystemExit) as fatal,
            pytest.raises(SystemExit),
        ):
            scheduler.shutdown()

        assert "Scheduler shutdown timed out after 60s" in fatal.call_args.args[0]
        fake_executor.shutdown.assert_not_called()


class TestSchedulerStopTokens:
    """Tests for stop token handling."""

    def test_get_stop_tokens(self, mock_model, mock_tokenizer):
        """Test _get_stop_tokens() retrieves EOS token."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        stop_tokens = scheduler._get_stop_tokens()

        # MockTokenizer has eos_token_id = 2
        assert mock_tokenizer.eos_token_id in stop_tokens

    def test_includes_eot_token_id(self, mock_model, mock_tokenizer):
        """Test _get_stop_tokens() includes end-of-turn token when available."""
        # eot_token_id as a single int
        mock_tokenizer.eot_token_id = 106
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        stop_tokens = scheduler._get_stop_tokens()
        assert 106 in stop_tokens
        assert mock_tokenizer.eos_token_id in stop_tokens  # EOS still there too

    def test_includes_eot_token_id_list(self, mock_model, mock_tokenizer):
        """Test _get_stop_tokens() handles eot_token_id as a list."""
        mock_tokenizer.eot_token_id = [106, 107]
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        stop_tokens = scheduler._get_stop_tokens()
        assert 106 in stop_tokens
        assert 107 in stop_tokens

    def test_falls_back_to_eot_token_encoding(self, mock_model, mock_tokenizer):
        """When eot_token_id is absent but eot_token string is present, encode it."""
        mock_tokenizer.eot_token = "<turn|>"
        # Ensure eot_token_id is NOT present
        assert (
            not hasattr(mock_tokenizer, "eot_token_id")
            or mock_tokenizer.eot_token_id is None
        )
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        stop_tokens = scheduler._get_stop_tokens()
        # The MockTokenizer.encode() returns hash-based IDs, so we get something
        assert len([t for t in stop_tokens if t != mock_tokenizer.eos_token_id]) > 0

    def test_no_eot_token_when_absent(self, mock_model, mock_tokenizer):
        """When neither eot_token_id nor eot_token string is present, no crash."""
        # MockTokenizer has no eot_token_id or eot_token by default
        assert not hasattr(mock_tokenizer, "eot_token_id")
        assert not hasattr(mock_tokenizer, "eot_token")
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        stop_tokens = scheduler._get_stop_tokens()
        assert mock_tokenizer.eos_token_id in stop_tokens


class TestSchedulerSuppressTokens:
    """Tests for generation_config.suppress_tokens handling."""

    def test_loads_generation_config_suppress_tokens(
        self, mock_model, mock_tokenizer, tmp_path
    ):
        (tmp_path / "generation_config.json").write_text(
            json.dumps({"suppress_tokens": [258883, 258882]}),
            encoding="utf-8",
        )
        mock_tokenizer.name_or_path = str(tmp_path)

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        assert scheduler._model_suppress_tokens == {258883, 258882}

    def test_suppress_logits_processor_masks_configured_ids(
        self, mock_model, mock_tokenizer
    ):
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler._model_suppress_tokens = {2}

        _, processors = scheduler._build_sampler_and_processors(SamplingParams())

        assert processors
        logits = mx.array([[0.0, 4.0, 100.0, 2.0]])
        masked = processors[-1](mx.array([1]), logits)
        mx.eval(masked)

        assert float(masked[0, 2].item()) == float("-inf")
        assert float(masked[0, 1].item()) == 4.0

    def test_vlm_mtp_first_bonus_uses_suppressing_sampler(
        self, mock_model, mock_tokenizer
    ):
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler._model_suppress_tokens = {3}
        scheduler._vlm_mtp_drafter = MagicMock()

        class FakeLanguageModel:
            def rollback_speculative_cache(self):
                pass

            def __call__(self, *args, **kwargs):
                return SimpleNamespace(
                    logits=mx.array([[[0.0, 0.0, 5.0, 99.0, 0.0]]]),
                    hidden_states=mx.zeros((1, 1, 4)),
                    shared_kv_states={},
                )

        class FakeVLMAdapter:
            def __init__(self):
                self._language_model = FakeLanguageModel()
                self.calls = []
                self.batch_rope_deltas = None

            def set_batch_rope_deltas(self, deltas):
                self.batch_rope_deltas = deltas

            def __call__(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return self._language_model(*args, **kwargs)

        mock_model = FakeVLMAdapter()
        scheduler.model = mock_model
        request = Request(
            request_id="req-mtp",
            prompt=[1],
            sampling_params=SamplingParams(max_tokens=4),
        )
        request.prompt_token_ids = [1]
        request.rope_deltas = 123.0
        cache = [SimpleNamespace(state=mx.array([0]))]

        def sampler(logits):
            return mx.argmax(logits, axis=-1)

        captured = {}

        def fake_run_vlm_mtp_decode(**kwargs):
            captured.update(kwargs)

            def _gen():
                yield kwargs["first_bonus"]

            return _gen()

        with patch.object(
            scheduler_module,
            "run_vlm_mtp_decode",
            side_effect=fake_run_vlm_mtp_decode,
        ):
            uid = scheduler._route_to_vlm_mtp(
                request,
                cache,
                [1],
                sampler,
                state_machine=object(),
            )

        assert uid is not None
        assert mock_model.calls
        assert captured["target_language_model"] is mock_model
        assert float(mock_model.batch_rope_deltas.item()) == 123.0
        assert captured["first_bonus"] == 2
        assert "prompt_tokens" not in captured

        round_logits = mx.array([[0.0, 0.0, 1.0, 99.0, 0.0]])
        round_token = captured["sampler"](round_logits)
        mx.eval(round_token)
        assert int(round_token.item()) == 2


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
        assert not hasattr(mock_tokenizer, "eos_token_ids")
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        tokens = scheduler._get_xtc_special_tokens()

        assert 2 in tokens

    def test_includes_parser_stop_tokens_without_base_stop(
        self, mock_model, mock_tokenizer
    ):
        """Parser stop tokens are XTC-protected but not BatchGenerator stops."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler._output_parser_factory = _ParserStopFactory()
        scheduler._output_parser_factory.stop_token_ids = {101, 102}

        stop_tokens = scheduler._get_stop_tokens()
        xtc_tokens = scheduler._get_xtc_special_tokens()

        assert 101 not in stop_tokens
        assert 102 not in stop_tokens
        assert 101 in xtc_tokens
        assert 102 in xtc_tokens


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

        with (
            patch.object(sched_mod.mx, "synchronize", side_effect=dispatch),
            patch.object(sched_mod.mx, "clear_cache", side_effect=fake_clear_cache),
        ):
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

        with (
            patch.object(sched_mod.mx, "synchronize", side_effect=dispatch),
            patch.object(sched_mod.mx, "clear_cache") as clear_cache,
        ):
            with pytest.raises(RuntimeError, match="default stream failure"):
                sched_mod._sync_and_clear_cache()
            clear_cache.assert_not_called()


class TestStoreCacheWorkerSync:
    """Tests for store-cache worker stream-scoped sync (#1437).

    Worker must wait on generation_stream specifically, not on the
    default stream. mx.synchronize() with no args only blocks on the
    default stream (gpu:0) and leaves the gpu:2 dispatched work
    unwaited, racing the buffer-protocol access in
    _extract_tensor_bytes -> SIGABRT in get_command_encoder(gpu:2).
    """

    def test_safe_sync_passes_generation_stream(self):
        """_safe_sync_stream() with no args must invoke mx.synchronize with
        the module-level _default_generation_stream object, not call the
        no-args variant.

        Regression: PR #1146 wired the worker to bare mx.synchronize()
        under the (incorrect) assumption that it was a global barrier
        and that stream-scoped sync was unsafe cross-thread. Both
        assumptions are wrong: synchronize() defaults to a single
        stream, and Stream objects are not thread-local. The worker
        path now routes through this helper so the regression has a
        single chokepoint to assert against.
        """
        from omlx import scheduler as sched_mod

        calls = []

        def fake_sync(*args, **kwargs):
            calls.append(args)

        with patch.object(sched_mod.mx, "synchronize", side_effect=fake_sync):
            sched_mod._safe_sync_stream()

        assert len(calls) == 1
        assert (
            calls[0] and calls[0][0] is sched_mod._default_generation_stream
        ), f"Worker sync must target _default_generation_stream, got: {calls}"

    def test_safe_sync_swallows_no_stream_runtime_error(self):
        """A 'no Stream' RuntimeError from cross-thread sync must be
        swallowed so the worker can still proceed to extract bytes.

        On some MLX builds mx.synchronize(stream) raises 'There is no
        Stream(gpu, X) in current thread' from a thread that has not
        submitted work to that stream. In the store-cache worker that
        condition means there is no in-flight gpu:2 work to drain, so
        it is safe to continue.
        """
        from omlx import scheduler as sched_mod

        def fake_sync(*args, **kwargs):
            raise RuntimeError("There is no Stream(gpu, 2) in current thread.")

        with patch.object(sched_mod.mx, "synchronize", side_effect=fake_sync):
            sched_mod._safe_sync_stream()

    def test_safe_sync_propagates_other_runtime_errors(self):
        """Real GPU errors must not be silently swallowed."""
        from omlx import scheduler as sched_mod

        def fake_sync(*args, **kwargs):
            raise RuntimeError("Metal command buffer execution failed")

        with patch.object(sched_mod.mx, "synchronize", side_effect=fake_sync):
            with pytest.raises(RuntimeError, match="command buffer execution failed"):
                sched_mod._safe_sync_stream()


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

    def test_capture_boundary_snapshot_at_block_boundary(
        self, mock_model, mock_tokenizer
    ):
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

    def test_boundary_override_preextracts_in_memory_intermediate_snapshots(
        self, mock_model, mock_tokenizer
    ):
        """In-memory boundary snapshots must be extracted before worker access."""
        config = SchedulerConfig(paged_cache_block_size=4)
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer, config=config)

        request = Request(
            request_id="req-hot-cache",
            prompt="hello",
            sampling_params=SamplingParams(),
        )
        scheduler.requests["req-hot-cache"] = request

        raw_intermediate = object()
        raw_latest = object()
        extracted_intermediate = [{"state": ("intermediate",)}]
        extracted_latest = [{"state": ("latest",)}]
        scheduler._boundary_cache_snapshots["req-hot-cache"] = {
            4: raw_intermediate,
            8: raw_latest,
        }

        def extract(raw_cache):
            if raw_cache is raw_latest:
                return extracted_latest, "latest-config"
            if raw_cache is raw_intermediate:
                return extracted_intermediate, "intermediate-config"
            raise AssertionError("unexpected raw cache")

        with patch.object(
            scheduler, "_extract_cache_states", side_effect=extract
        ) as ex:
            result = scheduler._get_boundary_store_override(
                "req-hot-cache", list(range(10))
            )

        assert result is not None
        token_sequence, cache_to_store, model_config, provider = result
        assert token_sequence == list(range(8))
        assert cache_to_store is extracted_latest
        assert model_config == "latest-config"
        assert 4 in provider

        ex.reset_mock()
        assert provider[4] is extracted_intermediate
        ex.assert_not_called()

    def test_cleanup_finished_pre_evals_intermediate_boundary_snapshots(
        self, mock_model, mock_tokenizer
    ):
        """Intermediate boundary snapshot arrays are materialized on engine thread."""
        from omlx import scheduler as sched_mod

        config = SchedulerConfig(paged_cache_block_size=4)
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer, config=config)
        scheduler.block_aware_cache = MagicMock()
        scheduler.paged_cache_manager = None

        latest_arr = mx.zeros((1,))
        intermediate_arr = mx.ones((1,))
        latest_cache = [{"state": (latest_arr,), "cache_type": "ArraysCache"}]
        intermediate_cache = [
            {"state": (intermediate_arr,), "cache_type": "ArraysCache"}
        ]
        provider = sched_mod._BoundarySnapshotProvider(
            store=None,
            request_id="req-hot-cache",
            valid_tcs=[4],
            in_memory_snapshots={4: intermediate_cache},
        )

        request = Request(
            request_id="req-hot-cache",
            prompt="hello",
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = [1, 2, 3, 4]
        request.num_prompt_tokens = 4
        request.output_token_ids = [5, 6, 7]
        request._extracted_cache = [{"state": ("final",)}]
        request._model_cache_config = None
        scheduler.running["req-hot-cache"] = request
        scheduler.requests["req-hot-cache"] = request

        with (
            patch.object(
                scheduler,
                "_get_boundary_store_override",
                return_value=([1, 2, 3, 4], latest_cache, None, provider),
            ),
            patch.object(sched_mod.mx, "eval") as eval_,
            patch.object(sched_mod, "_safe_sync_stream"),
        ):
            scheduler._cleanup_finished({"req-hot-cache"})

        eval_.assert_called_once()
        assert eval_.call_args.args == (latest_arr, intermediate_arr)

    def test_boundary_snapshot_synchronizes_generation_stream(
        self, mock_model, mock_tokenizer
    ):
        """Boundary snapshot extraction must synchronize generation_stream
        before accessing batch cache tensors to prevent Metal command buffer conflicts.
        """
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
        """Prefill callback should store rotating boundary snapshots.

        Regression: deliberately leave ``request_id_to_uid`` /
        ``uid_to_request_id`` unset, matching what happens in production
        during prefill (the request has not been inserted into
        BatchGenerator yet). The earlier shape passed a uid that
        resolved to None and silently dropped the snapshot.
        """
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
        scheduler.requests[request.request_id] = request
        scheduler.running[request.request_id] = request

        RotatingStub = type("RotatingKVCache", (), {})
        snapshot_cache = [RotatingStub()]

        scheduler._on_prefill_boundary_snapshot(request.request_id, snapshot_cache, 4)

        assert 4 in scheduler._boundary_cache_snapshots[request.request_id]
        assert (
            scheduler._boundary_cache_snapshots[request.request_id][4] == snapshot_cache
        )
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
        scheduler.requests[request.request_id] = request
        scheduler.running[request.request_id] = request

        RotatingStub = type("RotatingKVCache", (), {})
        scheduler._on_prefill_boundary_snapshot(request.request_id, [RotatingStub()], 3)

        assert request.request_id not in scheduler._boundary_cache_snapshots

    def test_emit_prefill_boundary_snapshot_persists_before_uid_assignment(
        self, mock_model, mock_tokenizer
    ):
        """Snapshots emitted during prefill must persist even though the
        request has not yet been inserted into BatchGenerator.

        The regression this guards against: the wrapper used to route
        through ``request_id_to_uid.get(rid, -1)`` →
        ``uid_to_request_id.get(-1)`` → ``None`` → silent return, so
        every block-boundary snapshot during prefill was dropped. For
        hybrid (ArraysCache / GDN) models that meant every non-last
        cached block stored a placeholder and identical-prefix re-
        uploads re-prefilled from scratch.
        """
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=SchedulerConfig(paged_cache_block_size=4),
        )
        scheduler.block_aware_cache = MagicMock()

        request = Request(
            request_id="req-prefill-pre-insert",
            prompt="hello",
            sampling_params=SamplingParams(),
        )
        scheduler.requests[request.request_id] = request
        scheduler.running[request.request_id] = request
        # Deliberately do NOT populate request_id_to_uid /
        # uid_to_request_id — that mirrors production state at the
        # time _emit_prefill_boundary_snapshot fires.
        assert request.request_id not in scheduler.request_id_to_uid

        RotatingStub = type("RotatingKVCache", (), {})
        prompt_cache = [RotatingStub()]

        scheduler._emit_prefill_boundary_snapshot(request, prompt_cache, 4)

        assert request.request_id in scheduler._boundary_cache_snapshots
        assert 4 in scheduler._boundary_cache_snapshots[request.request_id]


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
        with patch.object(sched_mod.mx, "get_cache_memory", return_value=1 * 1024**3):
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
        with patch.object(sched_mod.mx, "get_cache_memory", return_value=3 * 1024**3):
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
        assert extracted[0]["class_name"] == "CacheList"
        assert extracted[0]["cache_type"] == "CacheList"
        assert isinstance(extracted[0]["state"], list)
        assert isinstance(extracted[0]["meta_state"], tuple)
        assert len(extracted[0]["meta_state"]) == 2

    def test_extract_cache_states_cache_list_no_handlers(self, scheduler):
        """Test CacheList extraction when HAS_CACHE_TYPE_HANDLERS=False."""
        # Use real stub classes so type(obj).__name__ returns the correct name
        # (needed because the fallback branch uses type().__name__ for detection)
        KVCacheStub = type(
            "KVCache",
            (),
            {
                "state": (MagicMock(), MagicMock()),
                "meta_state": (32,),
            },
        )
        mock_kv_sub = KVCacheStub()

        CacheListStub = type(
            "CacheList",
            (),
            {
                "caches": (mock_kv_sub,),
                "state": [(MagicMock(), MagicMock())],
                "meta_state": (["KVCache"], [(32,)]),
            },
        )
        mock_cache_list = CacheListStub()

        raw_cache = [mock_cache_list]

        # Patch HAS_CACHE_TYPE_HANDLERS to False
        with patch("omlx.scheduler.HAS_CACHE_TYPE_HANDLERS", False):
            extracted, config = scheduler._extract_cache_states(raw_cache)

        # Must still have 1 extracted entry (Issue #1: no layer count mismatch)
        assert len(extracted) == 1
        assert extracted[0]["class_name"] == "CacheList"
        assert isinstance(extracted[0]["state"], list)


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


class TestSchedulerSSDLayerSignature:
    """Tests for pre-lookup SSD layer signature refresh."""

    def test_refresh_uses_final_turboquant_layout_and_sweeps(
        self, mock_tokenizer, tmp_path
    ):
        from mlx_lm.models.cache import KVCache

        from omlx.cache.paged_ssd_cache import PagedSSDBlockMetadata

        class TwoLayerModel:
            config = SimpleNamespace(
                num_hidden_layers=2,
                num_key_value_heads=2,
                num_attention_heads=2,
                head_dim=32,
            )

            def make_cache(self):
                return [KVCache(), KVCache()]

        scheduler = Scheduler(
            model=TwoLayerModel(),
            tokenizer=mock_tokenizer,
            config=SchedulerConfig(
                paged_ssd_cache_dir=str(tmp_path),
                paged_cache_block_size=4,
                model_name="test-model",
            ),
        )
        try:
            manager = scheduler.paged_ssd_cache_manager
            assert manager is not None
            assert manager._expected_layer_cache_types is None

            stale = PagedSSDBlockMetadata(
                block_hash=b"stale".ljust(32, b"\0"),
                file_path=tmp_path / "stale.safetensors",
                file_size=1024,
                token_count=4,
                created_at=0.0,
                last_access=0.0,
                num_layers=2,
                model_name="test-model",
                block_size=4,
                layer_cache_types=["KVCache", "KVCache"],
            )
            fresh = PagedSSDBlockMetadata(
                block_hash=b"fresh".ljust(32, b"\0"),
                file_path=tmp_path / "fresh.safetensors",
                file_size=1024,
                token_count=4,
                created_at=0.0,
                last_access=0.0,
                num_layers=2,
                model_name="test-model",
                block_size=4,
                layer_cache_types=["TurboQuantKVCache", "KVCache"],
            )
            manager._index.add(stale)
            manager._index.add(fresh)

            scheduler._turboquant_kv_bits = 4.0
            scheduler._turboquant_skip_last = True

            layer_cache_types = scheduler.refresh_ssd_layer_signature()

            assert layer_cache_types == ["TurboQuantKVCache", "KVCache"]
            assert manager._expected_layer_cache_types == layer_cache_types
            assert manager._index.get(stale.block_hash) is None
            assert manager._index.get(fresh.block_hash) is not None
        finally:
            scheduler.shutdown()


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

    def test_reschedule_corruption_increments_counter(self, mock_model, mock_tokenizer):
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

    def test_fail_all_requests_clears_everything(self, mock_model, mock_tokenizer):
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

    def test_fail_all_requests_preserves_cache(self, mock_model, mock_tokenizer):
        """fail_all_requests resets batch_generator but preserves block cache."""
        scheduler = self._make_scheduler(mock_model, mock_tokenizer)
        scheduler.batch_generator = MagicMock()
        scheduler.block_aware_cache = MagicMock()

        scheduler.fail_all_requests()

        assert scheduler.batch_generator is None
        assert scheduler._current_sampler_params is None
        # Cache should NOT be cleared (not a corruption error)
        scheduler.block_aware_cache.clear.assert_not_called()

    def test_fail_all_requests_includes_in_flight_orphans(
        self, mock_model, mock_tokenizer
    ):
        """Catch requests popped from self.waiting but not yet in self.running.

        Regression test for the hang triggered when ``_do_external_prefill``
        raises inside ``_schedule_waiting``: the request has already been
        popped from ``self.waiting`` and has not yet been inserted into
        ``self.running``, so the three-queue sweep misses it. The orphan
        still lives in ``self.requests`` and the HTTP collector for its id
        keeps awaiting a result that never arrives.
        """
        scheduler = self._make_scheduler(mock_model, mock_tokenizer)
        orphan = Request(
            request_id="req-orphan",
            prompt="orphan",
            sampling_params=SamplingParams(),
            prompt_token_ids=[6, 7],
            num_prompt_tokens=2,
        )
        # Orphan only: present in self.requests, absent from all three queues.
        scheduler.requests[orphan.request_id] = orphan
        # _schedule_waiting assigns a temp_uid (id(request)) before prefill and
        # only clears it on the success path, so an orphan leaves both uid maps
        # populated.
        temp_uid = id(orphan)
        scheduler.request_id_to_uid[orphan.request_id] = temp_uid
        scheduler.uid_to_request_id[temp_uid] = orphan.request_id
        assert orphan.request_id not in scheduler.waiting
        assert orphan.request_id not in scheduler.running
        assert orphan.request_id not in scheduler.prefilling

        failed_ids = scheduler.fail_all_requests()

        assert "req-orphan" in failed_ids
        assert "req-orphan" not in scheduler.requests
        # Stale uid mappings for the orphan must be cleared too.
        assert "req-orphan" not in scheduler.request_id_to_uid
        assert temp_uid not in scheduler.uid_to_request_id

    def test_fail_all_requests_excludes_async_cleanup_in_flight(
        self, mock_model, mock_tokenizer
    ):
        """Finished requests awaiting async cache-store cleanup must not be failed.

        ``_cleanup_finished`` keeps a finished request in ``self.requests``
        and registers its store future in ``_inflight_store_futures`` until
        ``_drain_pending_async_removes`` finalizes the cleanup. That request
        has already emitted ``finished=True`` to its collector; if
        ``fail_all_requests`` runs during this window, appending an error
        output via the orphan sweep would override the success for
        non-streaming ``generate()`` (engine_core returns the last queued
        output). The sweep must skip ids present in
        ``_inflight_store_futures`` and leave them for the async drain.
        """
        scheduler = self._make_scheduler(mock_model, mock_tokenizer)
        finished_pending_cleanup = Request(
            request_id="req-async-cleanup",
            prompt="finished",
            sampling_params=SamplingParams(),
            prompt_token_ids=[8, 9],
            num_prompt_tokens=2,
        )
        # Simulate _cleanup_finished's terminal state: request lives in
        # self.requests + _inflight_store_futures, absent from all three queues.
        scheduler.requests[finished_pending_cleanup.request_id] = (
            finished_pending_cleanup
        )
        scheduler._inflight_store_futures[finished_pending_cleanup.request_id] = (
            MagicMock()
        )
        # Its uid mapping is still live for _drain_pending_async_removes and
        # must survive fail_all_requests untouched.
        scheduler.request_id_to_uid[finished_pending_cleanup.request_id] = 999
        scheduler.uid_to_request_id[999] = finished_pending_cleanup.request_id

        failed_ids = scheduler.fail_all_requests()

        assert "req-async-cleanup" not in failed_ids
        assert "req-async-cleanup" in scheduler.requests
        assert "req-async-cleanup" in scheduler._inflight_store_futures
        # uid mapping preserved for the async drain.
        assert scheduler.request_id_to_uid["req-async-cleanup"] == 999
        assert scheduler.uid_to_request_id[999] == "req-async-cleanup"


class TestGenerationOverflowRecovery:
    """Tests for MLX __next_prime overflow recovery."""

    def _make_scheduler(self, mock_model, mock_tokenizer, count: int = 2):
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.batch_generator = MagicMock()
        scheduler.batch_generator.next_generated.side_effect = OverflowError(
            "__next_prime overflow"
        )
        for i in range(count):
            request = Request(
                request_id=f"req-overflow-{i}",
                prompt=f"prompt {i}",
                sampling_params=SamplingParams(max_tokens=4),
                prompt_token_ids=[1, 2, 3],
                num_prompt_tokens=3,
                status=RequestStatus.RUNNING,
                batch_uid=i,
                remaining_tokens=[1, 2, 3],
            )
            request.output_token_ids = [10, 11]
            request.output_text = "partial"
            request.num_computed_tokens = 2
            scheduler.running[request.request_id] = request
            scheduler.requests[request.request_id] = request
            scheduler.request_id_to_uid[request.request_id] = i
            scheduler.uid_to_request_id[i] = request.request_id
        return scheduler

    def test_generation_overflow_detection(self, mock_model, mock_tokenizer):
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        assert scheduler._is_generation_overflow_error(
            OverflowError("__next_prime overflow")
        )
        assert not scheduler._is_generation_overflow_error(
            OverflowError("integer conversion overflow")
        )
        assert not scheduler._is_generation_overflow_error(
            RuntimeError("__next_prime overflow")
        )

    def test_generation_overflow_reschedules_for_serial_retry(
        self, mock_model, mock_tokenizer
    ):
        scheduler = self._make_scheduler(mock_model, mock_tokenizer, count=3)
        scheduler.config.max_num_seqs = 8

        with patch("omlx.scheduler._sync_and_clear_cache"):
            output = scheduler.step()

        assert output.outputs == []
        assert output.has_work is True
        assert scheduler.batch_generator is None
        assert scheduler.running == {}
        assert list(scheduler.request_id_to_uid) == []
        waiting_ids = [request.request_id for request in scheduler.waiting]
        assert waiting_ids == [
            "req-overflow-0",
            "req-overflow-1",
            "req-overflow-2",
        ]
        for request in scheduler.waiting:
            assert request.generation_overflow_retries == 1
            assert request.output_token_ids == []
            assert request.output_text == ""
            assert request.num_computed_tokens == 0
        assert scheduler._effective_max_num_seqs() == 1

    def test_generation_overflow_fails_after_serial_retry(
        self, mock_model, mock_tokenizer
    ):
        scheduler = self._make_scheduler(mock_model, mock_tokenizer, count=1)
        request = next(iter(scheduler.running.values()))
        request.generation_overflow_retries = 1

        with patch("omlx.scheduler._sync_and_clear_cache"):
            output = scheduler.step()

        assert len(output.outputs) == 1
        error_output = output.outputs[0]
        assert error_output.request_id == request.request_id
        assert error_output.finished is True
        assert error_output.finish_reason == "error"
        assert "Generation overflow not recoverable" in error_output.error
        assert request.request_id in output.finished_request_ids
        assert scheduler.running == {}
        assert scheduler.waiting == deque()
        assert request.request_id not in scheduler.requests
        assert scheduler.has_requests() is False

    def test_unrelated_overflow_still_raises(self, mock_model, mock_tokenizer):
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.batch_generator = MagicMock()
        scheduler.batch_generator.next_generated.side_effect = OverflowError(
            "integer conversion overflow"
        )
        request = Request(
            request_id="req-other-overflow",
            prompt="prompt",
            sampling_params=SamplingParams(max_tokens=4),
            prompt_token_ids=[1],
            num_prompt_tokens=1,
            status=RequestStatus.RUNNING,
        )
        scheduler.running[request.request_id] = request
        scheduler.requests[request.request_id] = request

        with pytest.raises(OverflowError, match="integer conversion overflow"):
            scheduler.step()


class TestStoreCacheAdmissionBackpressure:
    """Tests for store-cache admission backpressure (#1684)."""

    def _make_request(self, request_id: str = "req-store-gate") -> Request:
        return Request(
            request_id=request_id,
            prompt="hello",
            sampling_params=SamplingParams(max_tokens=4),
            prompt_token_ids=[1],
            num_prompt_tokens=1,
        )

    def _queue_request(self, scheduler: Scheduler, request: Request) -> None:
        scheduler.waiting.append(request)
        scheduler.requests[request.request_id] = request

    def test_schedule_waiting_defers_when_gate_full_without_running(
        self, mock_model, mock_tokenizer
    ):
        """Sequential turns must respect store-cache backpressure."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        gate = _StoreCacheGate(cap=1)
        gate.note_submitted()
        scheduler._store_cache_gate = gate
        request = self._make_request()
        self._queue_request(scheduler, request)
        scheduler._ensure_batch_generator = MagicMock()

        scheduled, rejected = scheduler._schedule_waiting()

        assert scheduled == []
        assert rejected == []
        assert list(scheduler.waiting) == [request]
        assert scheduler.running == {}
        scheduler._ensure_batch_generator.assert_not_called()

    def test_memory_guard_fails_persistent_admission_stall(
        self, mock_model, mock_tokenizer
    ):
        """A persistent memory-gated head-of-line wait should fail cleanly."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        request = self._make_request("req-stalled")
        self._queue_request(scheduler, request)
        running = self._make_request("req-running")
        scheduler.running[running.request_id] = running
        scheduler.requests[running.request_id] = running
        scheduler._prefill_memory_guard = True
        scheduler._memory_limit_bytes = 100
        scheduler._current_usage_bytes = MagicMock(return_value=101)
        scheduler._memory_admission_blocked_request_id = request.request_id
        scheduler._memory_admission_blocked_since = 0.0

        with patch("omlx.scheduler.time.monotonic", return_value=61.0):
            scheduled, rejected = scheduler._schedule_waiting()

        assert scheduled == []
        assert len(rejected) == 1
        assert rejected[0].request_id == request.request_id
        assert rejected[0].finish_reason == "error"
        assert rejected[0].error_code == "memory_admission_stalled"
        assert request.request_id not in scheduler.requests
        assert list(scheduler.waiting) == []
        assert scheduler.running[running.request_id] is running

    def test_store_cache_gate_fails_persistent_non_memory_stall(
        self, mock_model, mock_tokenizer
    ):
        """A stuck store-cache gate must not leave admission waiting forever."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        gate = _StoreCacheGate(cap=1)
        gate.note_submitted()
        scheduler._store_cache_gate = gate
        request = self._make_request("req-store-stalled")
        self._queue_request(scheduler, request)
        scheduler._prefill_memory_guard = True
        scheduler._memory_limit_bytes = 100
        scheduler._current_usage_bytes = MagicMock(return_value=50)
        scheduler._store_cache_admission_blocked_request_id = request.request_id
        scheduler._store_cache_admission_blocked_since = 0.0
        scheduler._ensure_batch_generator = MagicMock()

        with patch("omlx.scheduler.time.monotonic", return_value=61.0):
            scheduled, rejected = scheduler._schedule_waiting()

        assert scheduled == []
        assert len(rejected) == 1
        assert rejected[0].request_id == request.request_id
        assert rejected[0].finish_reason == "error"
        assert rejected[0].error_code == "store_cache_admission_stalled"
        assert rejected[0].error_metadata["store_cache_in_flight"] == 1
        assert request.request_id not in scheduler.requests
        assert list(scheduler.waiting) == []
        scheduler._ensure_batch_generator.assert_not_called()

    def test_store_cache_stall_timer_clears_when_gate_recovers_before_freshness_wait(
        self, mock_model, mock_tokenizer
    ):
        """A recovered store-cache gate must not accumulate stale stall time."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        gate = _StoreCacheGate(cap=2)
        gate.note_submitted()
        gate.note_submitted()
        scheduler._store_cache_gate = gate
        request = self._make_request("req-store-recovered")
        self._queue_request(scheduler, request)
        scheduler._prefill_memory_guard = True
        scheduler._memory_limit_bytes = 100
        scheduler._current_usage_bytes = MagicMock(return_value=50)
        scheduler._ensure_batch_generator = MagicMock()

        with patch("omlx.scheduler.time.monotonic", return_value=0.0):
            scheduled, rejected = scheduler._schedule_waiting()

        assert scheduled == []
        assert rejected == []
        assert scheduler._store_cache_admission_blocked_request_id == request.request_id

        gate.note_done()
        scheduler._should_defer_for_cache_freshness = MagicMock(return_value=True)
        with patch("omlx.scheduler.time.monotonic", return_value=30.0):
            scheduled, rejected = scheduler._schedule_waiting()

        assert scheduled == []
        assert rejected == []
        assert list(scheduler.waiting) == [request]
        assert scheduler._store_cache_admission_blocked_request_id is None

        gate.note_submitted()
        scheduler._should_defer_for_cache_freshness = MagicMock(return_value=False)
        with patch("omlx.scheduler.time.monotonic", return_value=61.0):
            scheduled, rejected = scheduler._schedule_waiting()

        assert scheduled == []
        assert rejected == []
        assert list(scheduler.waiting) == [request]
        assert scheduler._store_cache_admission_blocked_request_id == request.request_id
        assert scheduler._store_cache_admission_blocked_since == 61.0
        scheduler._ensure_batch_generator.assert_not_called()

    def test_schedule_waiting_allows_when_gate_has_capacity(
        self, mock_model, mock_tokenizer
    ):
        """Below-cap store-cache cleanup must not block admission."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        gate = _StoreCacheGate(cap=2)
        gate.note_submitted()
        scheduler._store_cache_gate = gate
        request = self._make_request()
        self._queue_request(scheduler, request)

        batch_generator = MagicMock()
        batch_generator.insert.return_value = [42]
        scheduler.batch_generator = batch_generator
        scheduler._ensure_batch_generator = MagicMock()
        scheduler._build_sampler_and_processors = MagicMock(
            return_value=(MagicMock(), [])
        )
        scheduler._build_state_machine = MagicMock(return_value=MagicMock())
        scheduler._preflight_memory_check = MagicMock(return_value=None)

        scheduled, rejected = scheduler._schedule_waiting()

        assert rejected == []
        assert scheduled == [request]
        assert scheduler.waiting == deque()
        assert scheduler.running[request.request_id] is request
        scheduler._ensure_batch_generator.assert_called_once_with(
            request.sampling_params
        )

    def test_llama4_admission_serializes_waiting_requests(
        self, mock_model, mock_tokenizer
    ):
        """A second Llama 4 request stays queued instead of forming a batch."""
        mock_model.config.model_type = "llama4"
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=SchedulerConfig(max_num_seqs=4),
        )
        first = self._make_request("req-llama4-1")
        second = self._make_request("req-llama4-2")
        self._queue_request(scheduler, first)
        self._queue_request(scheduler, second)

        batch_generator = MagicMock()
        batch_generator.insert.return_value = [42]
        scheduler.batch_generator = batch_generator
        scheduler._ensure_batch_generator = MagicMock()
        scheduler._build_sampler_and_processors = MagicMock(
            return_value=(MagicMock(), [])
        )
        scheduler._build_state_machine = MagicMock(return_value=MagicMock())
        scheduler._preflight_memory_check = MagicMock(return_value=None)

        scheduled, rejected = scheduler._schedule_waiting()

        assert rejected == []
        assert scheduled == [first]
        assert list(scheduler.waiting) == [second]
        assert list(scheduler.running) == [first.request_id]
        assert batch_generator.insert.call_count == 1

    def test_schedule_waiting_defers_when_pending_cleanups_reach_cap(
        self, mock_model, mock_tokenizer
    ):
        """Deferred removals still own cache refs even if the gate has room."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler._store_cache_gate = _StoreCacheGate(cap=2)
        scheduler._pending_async_removes.append(
            (1, "req-cleanup-1", concurrent.futures.Future())
        )
        scheduler._pending_async_removes.append(
            (2, "req-cleanup-2", concurrent.futures.Future())
        )
        request = self._make_request()
        self._queue_request(scheduler, request)
        scheduler._ensure_batch_generator = MagicMock()

        scheduled, rejected = scheduler._schedule_waiting()

        assert scheduled == []
        assert rejected == []
        assert list(scheduler.waiting) == [request]
        scheduler._ensure_batch_generator.assert_not_called()

    def test_drain_pending_async_removes_releases_done_entries_out_of_order(
        self, mock_model, mock_tokenizer
    ):
        """One slow store-cache future must not pin later completed caches."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.batch_generator = MagicMock()
        gate = _StoreCacheGate(cap=2)
        gate.note_submitted()
        gate.note_submitted()
        scheduler._store_cache_gate = gate

        slow_future = concurrent.futures.Future()
        done_future = concurrent.futures.Future()
        done_future.set_result(None)

        slow_request = self._make_request("req-slow")
        done_request = self._make_request("req-done")
        slow_request._extracted_cache = object()
        done_request._extracted_cache = object()
        scheduler.requests[slow_request.request_id] = slow_request
        scheduler.requests[done_request.request_id] = done_request
        scheduler._inflight_store_futures[slow_request.request_id] = slow_future
        scheduler._inflight_store_futures[done_request.request_id] = done_future
        scheduler._inflight_store_info[slow_request.request_id] = (
            scheduler_module._InflightStoreInfo(tokens=[1, 2, 3])
        )
        scheduler._inflight_store_info[done_request.request_id] = (
            scheduler_module._InflightStoreInfo(tokens=[1, 2, 3])
        )
        scheduler.request_id_to_uid[slow_request.request_id] = 1
        scheduler.request_id_to_uid[done_request.request_id] = 2
        scheduler.uid_to_request_id[1] = slow_request.request_id
        scheduler.uid_to_request_id[2] = done_request.request_id
        scheduler._pending_async_removes.append(
            (1, slow_request.request_id, slow_future)
        )
        scheduler._pending_async_removes.append(
            (2, done_request.request_id, done_future)
        )

        with patch("omlx.scheduler._safe_sync_stream"):
            drained = scheduler._drain_pending_async_removes()

        assert drained is True
        assert list(scheduler._pending_async_removes) == [
            (1, slow_request.request_id, slow_future)
        ]
        assert slow_request.request_id in scheduler.requests
        assert done_request.request_id not in scheduler.requests
        assert done_request.request_id not in scheduler._inflight_store_futures
        assert done_request.request_id not in scheduler._inflight_store_info
        assert slow_request.request_id in scheduler._inflight_store_info
        assert done_request.request_id not in scheduler.request_id_to_uid
        assert 2 not in scheduler.uid_to_request_id
        assert gate.in_flight == 1
        scheduler.batch_generator.remove.assert_called_once_with([2])


class TestBatchGeneratorAllTokens:
    """TokenBuffer seed passed to mlx-lm BatchGenerator.insert."""

    def _make_scheduler(self, mock_model, mock_tokenizer):
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        batch_generator = MagicMock()
        batch_generator.insert.return_value = [42]
        scheduler.batch_generator = batch_generator
        scheduler._ensure_batch_generator = MagicMock()
        scheduler._build_sampler_and_processors = MagicMock(
            return_value=(MagicMock(), [])
        )
        scheduler._build_state_machine = MagicMock(return_value=MagicMock())
        scheduler._preflight_memory_check = MagicMock(return_value=None)
        scheduler._validate_cache = MagicMock(return_value=True)
        return scheduler

    def _queue_request(
        self,
        scheduler: Scheduler,
        request: Request,
        *,
        prompt_tokens: list[int],
        remaining_tokens: list[int],
        cached_tokens: int = 0,
        prompt_cache=None,
    ) -> None:
        request.prompt_token_ids = prompt_tokens
        request.num_prompt_tokens = len(prompt_tokens)
        request.remaining_tokens = remaining_tokens
        request.cached_tokens = cached_tokens
        request.prompt_cache = prompt_cache
        scheduler.waiting.append(request)
        scheduler.requests[request.request_id] = request

    def test_cache_hit_insert_seeds_prompt_prefix(self, mock_model, mock_tokenizer):
        scheduler = self._make_scheduler(mock_model, mock_tokenizer)
        request = Request(
            request_id="req-cache-hit-all-tokens",
            prompt=[11, 12, 13, 14],
            sampling_params=SamplingParams(max_tokens=4),
        )
        self._queue_request(
            scheduler,
            request,
            prompt_tokens=[11, 12, 13, 14],
            remaining_tokens=[14],
            cached_tokens=3,
            prompt_cache=[MagicMock()],
        )

        scheduler._schedule_waiting()

        call_kwargs = scheduler.batch_generator.insert.call_args.kwargs
        assert call_kwargs["all_tokens"] == [[11, 12, 13]]

    def test_external_prefill_insert_seeds_prompt_prefix(
        self, mock_model, mock_tokenizer
    ):
        scheduler = self._make_scheduler(mock_model, mock_tokenizer)
        scheduler._do_external_prefill = MagicMock(return_value=([MagicMock()], [14]))
        request = Request(
            request_id="req-prefill-all-tokens",
            prompt=[11, 12, 13, 14],
            sampling_params=SamplingParams(max_tokens=4),
        )
        self._queue_request(
            scheduler,
            request,
            prompt_tokens=[11, 12, 13, 14],
            remaining_tokens=[11, 12, 13, 14],
        )

        scheduler._schedule_waiting()

        call_kwargs = scheduler.batch_generator.insert.call_args.kwargs
        assert call_kwargs["all_tokens"] == [[11, 12, 13]]

    def test_single_token_prompt_uses_empty_seed(self, mock_model, mock_tokenizer):
        scheduler = self._make_scheduler(mock_model, mock_tokenizer)
        request = Request(
            request_id="req-single-token-all-tokens",
            prompt=[99],
            sampling_params=SamplingParams(max_tokens=4),
        )
        self._queue_request(
            scheduler,
            request,
            prompt_tokens=[99],
            remaining_tokens=[99],
        )

        scheduler._schedule_waiting()

        call_kwargs = scheduler.batch_generator.insert.call_args.kwargs
        assert call_kwargs["all_tokens"] == [[]]

    def test_concurrent_inserts_keep_per_request_seed(self, mock_model, mock_tokenizer):
        scheduler = self._make_scheduler(mock_model, mock_tokenizer)
        first = Request(
            request_id="req-concurrent-a",
            prompt=[11, 12, 13],
            sampling_params=SamplingParams(max_tokens=4),
        )
        second = Request(
            request_id="req-concurrent-b",
            prompt=[21, 22, 23, 24],
            sampling_params=SamplingParams(max_tokens=4),
        )
        self._queue_request(
            scheduler,
            first,
            prompt_tokens=[11, 12, 13],
            remaining_tokens=[13],
            prompt_cache=[MagicMock()],
        )
        self._queue_request(
            scheduler,
            second,
            prompt_tokens=[21, 22, 23, 24],
            remaining_tokens=[24],
            prompt_cache=[MagicMock()],
        )

        scheduler._schedule_waiting()

        calls = scheduler.batch_generator.insert.call_args_list
        assert [call.kwargs["all_tokens"] for call in calls] == [
            [[11, 12]],
            [[21, 22, 23]],
        ]

    def test_chunked_prefill_insert_seeds_prompt_prefix(
        self, mock_model, mock_tokenizer
    ):
        scheduler = self._make_scheduler(mock_model, mock_tokenizer)
        request = Request(
            request_id="req-chunked-all-tokens",
            prompt=[11, 12, 13, 14],
            sampling_params=SamplingParams(max_tokens=4),
        )
        request.prompt_token_ids = [11, 12, 13, 14]
        request.num_prompt_tokens = 4
        state = _PrefillState(
            request=request,
            cache=[MagicMock()],
            tokens_remaining=mx.array([[]]),
            last_token=[14],
            tokens_processed=3,
            base_size=0,
            emitted_boundaries={},
            boundary_enabled=False,
            block_size=0,
            total_length=4,
            sampler=MagicMock(),
            sm=MagicMock(),
            per_row_lps=[],
        )
        scheduled = []

        scheduler._insert_prefilled_request(request, state, scheduled)

        call_kwargs = scheduler.batch_generator.insert.call_args.kwargs
        assert call_kwargs["all_tokens"] == [[11, 12, 13]]
        assert scheduled == [request]

    def test_chunked_prefill_converts_turboquant_cache_before_insert(
        self, mock_model, mock_tokenizer
    ):
        """Chunked prefill must mirror external prefill's TQ epilogue."""
        from mlx_lm.models.cache import KVCache
        from mlx_vlm.turboquant import TurboQuantKVCache

        scheduler = self._make_scheduler(mock_model, mock_tokenizer)
        scheduler._turboquant_kv_bits = 4.0
        scheduler._turboquant_skip_last = False

        kv_cache = KVCache()
        kv_cache.update_and_fetch(
            mx.random.normal((1, 2, 4, 32)),
            mx.random.normal((1, 2, 4, 32)),
        )

        request = Request(
            request_id="req-chunked-tq",
            prompt=[11, 12, 13, 14, 15],
            sampling_params=SamplingParams(max_tokens=4),
        )
        request.prompt_token_ids = [11, 12, 13, 14, 15]
        request.num_prompt_tokens = 5
        request.cached_tokens = 4
        state = _PrefillState(
            request=request,
            cache=[kv_cache],
            tokens_remaining=mx.array([[]]),
            last_token=[15],
            tokens_processed=4,
            base_size=4,
            emitted_boundaries={},
            boundary_enabled=False,
            block_size=0,
            total_length=5,
            sampler=MagicMock(),
            sm=MagicMock(),
            per_row_lps=[],
        )
        scheduled = []

        with patch("omlx.scheduler._materialize_cache_storage") as materialize:
            with patch("omlx.scheduler._sync_and_clear_cache") as sync_clear:
                scheduler._insert_prefilled_request(request, state, scheduled)

        call_kwargs = scheduler.batch_generator.insert.call_args.kwargs
        inserted_cache = call_kwargs["caches"][0][0]
        assert isinstance(inserted_cache, TurboQuantKVCache)
        assert state.cache[0] is inserted_cache
        materialize.assert_called_once_with(state.cache)
        sync_clear.assert_called_once_with(scheduler._stream)
        assert scheduled == [request]

    def test_chunked_prefill_converts_after_sized_arrays_restore(
        self, mock_model, mock_tokenizer
    ):
        """Restored ArraysCache wrappers must not skip the TQ epilogue."""
        from mlx_lm.models.cache import ArraysCache, KVCache
        from mlx_vlm.turboquant import TurboQuantKVCache

        from omlx.cache.type_handlers import SizedArraysCache

        scheduler = self._make_scheduler(mock_model, mock_tokenizer)
        scheduler._turboquant_kv_bits = 4.0
        scheduler._turboquant_skip_last = False

        arrays_cache = ArraysCache(size=2)
        arrays_cache.cache[0] = mx.random.normal((1, 3, 16))
        arrays_cache.cache[1] = mx.random.normal((1, 2, 16, 16))
        sized_arrays_cache = SizedArraysCache(arrays_cache, token_count=4)

        kv_cache = KVCache()
        kv_cache.update_and_fetch(
            mx.random.normal((1, 2, 4, 32)),
            mx.random.normal((1, 2, 4, 32)),
        )

        request = Request(
            request_id="req-chunked-tq-sized-arrays",
            prompt=[11, 12, 13, 14, 15],
            sampling_params=SamplingParams(max_tokens=4),
        )
        request.prompt_token_ids = [11, 12, 13, 14, 15]
        request.num_prompt_tokens = 5
        request.cached_tokens = 4
        state = _PrefillState(
            request=request,
            cache=[sized_arrays_cache, kv_cache],
            tokens_remaining=mx.array([[]]),
            last_token=[15],
            tokens_processed=4,
            base_size=4,
            emitted_boundaries={},
            boundary_enabled=False,
            block_size=0,
            total_length=5,
            sampler=MagicMock(),
            sm=MagicMock(),
            per_row_lps=[],
        )
        scheduled = []

        with patch("omlx.scheduler._materialize_cache_storage") as materialize:
            with patch("omlx.scheduler._sync_and_clear_cache") as sync_clear:
                scheduler._insert_prefilled_request(request, state, scheduled)

        call_kwargs = scheduler.batch_generator.insert.call_args.kwargs
        inserted_cache = call_kwargs["caches"][0]
        assert inserted_cache[0] is sized_arrays_cache
        assert isinstance(inserted_cache[1], TurboQuantKVCache)
        assert state.cache[1] is inserted_cache[1]
        materialize.assert_called_once_with(state.cache)
        sync_clear.assert_called_once_with(scheduler._stream)
        assert scheduled == [request]


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
        scheduler = self._make_scheduler(
            mock_model, think_start_id=100, think_end_id=101
        )
        request = self._make_request([1, 2, 3, 100, 198])  # 198 = \n
        assert scheduler._detect_needs_think_prefix(request) is True

    def test_enabled_thinking_last_token(self, mock_model):
        """<think> as last token -> True."""
        scheduler = self._make_scheduler(
            mock_model, think_start_id=100, think_end_id=101
        )
        request = self._make_request([1, 2, 3, 100])
        assert scheduler._detect_needs_think_prefix(request) is True

    def test_disabled_thinking_adjacent(self, mock_model):
        """<think></think> adjacent -> False (disabled, e.g. Nemotron)."""
        scheduler = self._make_scheduler(
            mock_model, think_start_id=100, think_end_id=101
        )
        request = self._make_request([1, 2, 3, 100, 101])
        assert scheduler._detect_needs_think_prefix(request) is False

    def test_disabled_thinking_with_prefix(self, mock_model):
        """X <think></think> -> False (disabled with preceding token)."""
        scheduler = self._make_scheduler(
            mock_model, think_start_id=100, think_end_id=101
        )
        request = self._make_request([1, 2, 50, 100, 101])
        assert scheduler._detect_needs_think_prefix(request) is False

    def test_no_think_token_in_tail(self, mock_model):
        """No <think> in last 3 tokens -> False."""
        scheduler = self._make_scheduler(
            mock_model, think_start_id=100, think_end_id=101
        )
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
        scheduler = self._make_scheduler(
            mock_model, think_start_id=100, think_end_id=101
        )
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

    def test_parser_stop_sets_finish_reason(self, mock_model):
        tokenizer = self._GemmaTokenizer({11: "<|return|>"})
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=tokenizer,
            config=SchedulerConfig(model_name="test-model"),
        )
        scheduler._output_parser_factory = _ParserStopFactory()

        request = Request(
            request_id="parser-stop-req",
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
        ]

        outputs, finished_ids = scheduler._process_batch_responses(responses)

        assert finished_ids == {"parser-stop-req"}
        assert outputs[-1].finished is True
        assert outputs[-1].finish_reason == "stop"


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
        model = MagicMock(
            spec=[
                "__call__",
                "clear_vlm_position_state",
                "parameters",
                "make_cache",
            ]
        )
        model.clear_vlm_position_state = MagicMock()
        model.make_cache.return_value = []
        return model

    def test_schedule_waiting_preserves_vlm_position_state(self, mock_tokenizer):
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

    def test_schedule_waiting_clears_text_only_position_state(self, mock_tokenizer):
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

    def test_cached_text_only_prefill_seeds_zero_mrope_delta(self, mock_tokenizer):
        """Cached text-only mRoPE suffixes must start at the restored offset."""
        model = self._make_vlm_model()
        model._language_model = MagicMock()
        model._language_model._rope_deltas = mx.array([[123]])
        scheduler = Scheduler(model=model, tokenizer=mock_tokenizer)

        request = Request(
            request_id="text-cached-001",
            prompt="hello world",
            sampling_params=SamplingParams(max_tokens=50),
        )
        request.prompt_token_ids = [1, 2, 3, 4]
        request.num_prompt_tokens = 4
        request.cached_tokens = 2048

        scheduler._do_external_prefill(
            request,
            tokens=[1, 2, 3, 4],
            existing_cache=[],
            vlm_embeds=None,
        )

        model.clear_vlm_position_state.assert_called_once()
        seeded = model._language_model._rope_deltas
        assert seeded.shape == (1, 1)
        assert seeded.item() == 0


class TestBuildStateMachineStopStrings:
    """Tests for _build_state_machine stop-string tokenization.

    The scheduler must convert SamplingParams.stop (a list of strings)
    into token-sequence transitions on the per-request state machine,
    so mlx-lm's BatchGenerator can halt on user-supplied stop sequences.
    """

    def _make_scheduler(self, mock_model, mock_tokenizer):
        return Scheduler(model=mock_model, tokenizer=mock_tokenizer)

    def _request_with_stop(self, stop):
        return Request(
            request_id="stop-001",
            prompt="hello",
            sampling_params=SamplingParams(max_tokens=10, stop=stop),
        )

    def test_no_stop_string_only_eos_transitions(self, mock_model, mock_tokenizer):
        scheduler = self._make_scheduler(mock_model, mock_tokenizer)
        sm = scheduler._build_state_machine(self._request_with_stop([]))
        # SequenceStateMachine has internal _states dict; non-empty implies
        # at least the EOS transitions are present.
        assert sm._states

    def test_stop_string_added_as_token_sequence(self, mock_model, mock_tokenizer):
        scheduler = self._make_scheduler(mock_model, mock_tokenizer)
        # MockTokenizer encodes "delta" to a single hash-derived token id.
        expected_seq = mock_tokenizer.encode("delta", add_special_tokens=False)
        assert expected_seq, "MockTokenizer must produce a token for 'delta'"

        sm = scheduler._build_state_machine(self._request_with_stop(["delta"]))
        # Walk the trie following expected_seq; the terminal node must
        # have a __match__ entry, meaning the sequence is registered.
        node = sm._states["normal"][0]
        for tok in expected_seq:
            assert tok in node, f"token {tok} missing from trie"
            node = node[tok]
        assert "__match__" in node, "stop sequence not terminated in trie"

    def test_empty_or_non_string_entries_skipped(self, mock_model, mock_tokenizer):
        scheduler = self._make_scheduler(mock_model, mock_tokenizer)
        # Mixed list with empty string and non-string entry; only "real"
        # should be tokenized.
        sm = scheduler._build_state_machine(self._request_with_stop(["", "real", 123]))
        real_seq = mock_tokenizer.encode("real", add_special_tokens=False)
        node = sm._states["normal"][0]
        for tok in real_seq:
            assert tok in node
            node = node[tok]
        assert "__match__" in node

    def test_multiple_stop_strings_all_registered(self, mock_model, mock_tokenizer):
        scheduler = self._make_scheduler(mock_model, mock_tokenizer)
        sm = scheduler._build_state_machine(self._request_with_stop(["foo", "bar"]))
        for stop_str in ("foo", "bar"):
            seq = mock_tokenizer.encode(stop_str, add_special_tokens=False)
            node = sm._states["normal"][0]
            for tok in seq:
                assert tok in node
                node = node[tok]
            assert "__match__" in node


class TestTurboQuantMLAGuard:
    """Regression tests for #1613: MLA models must not be TurboQuant-converted.

    GLM-4.7-Flash / DeepSeek use Multi-head Latent Attention and read fetched
    cache tensors directly (k_pe.swapaxes(...)), which crashes on TurboQuant's
    quantized NamedTuple states ('TurboQuantMSEState' object has no attribute
    'swapaxes'). _turboquant_eligible() must return False for them so they stay
    fp16.
    """

    def test_mla_model_ineligible_by_config(self, mock_model, mock_tokenizer):
        from mlx_lm.models.cache import KVCache

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler._turboquant_kv_bits = 4.0
        # MLA config exposes kv_lora_rank (GLM-4.7-Flash, DeepSeek-V*).
        scheduler.model = SimpleNamespace(args=SimpleNamespace(kv_lora_rank=512))
        scheduler._mla_model = None

        assert scheduler._model_uses_mla() is True
        assert scheduler._turboquant_eligible([KVCache()]) is False

    def test_mla_model_ineligible_by_architecture(self, mock_model, mock_tokenizer):
        from mlx_lm.models.cache import KVCache

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler._turboquant_kv_bits = 4.0
        # No kv_lora_rank in config, but an attention submodule with the MLA
        # down-projection / latent layernorm.
        attn = SimpleNamespace(
            kv_a_proj_with_mqa=object(),
            kv_a_layernorm=object(),
            kv_lora_rank=512,
        )
        scheduler.model = SimpleNamespace(
            args=SimpleNamespace(), modules=lambda: [attn]
        )
        scheduler._mla_model = None

        assert scheduler._model_uses_mla() is True
        assert scheduler._turboquant_eligible([KVCache()]) is False

    def test_standard_model_still_eligible(self, mock_model, mock_tokenizer):
        from mlx_lm.models.cache import KVCache

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler._turboquant_kv_bits = 4.0
        # Standard MHA/GQA model: no kv_lora_rank, no MLA submodules.
        scheduler.model = SimpleNamespace(
            args=SimpleNamespace(num_hidden_layers=4), modules=lambda: []
        )
        scheduler._mla_model = None

        assert scheduler._model_uses_mla() is False
        assert scheduler._turboquant_eligible([KVCache()]) is True

    def test_mla_model_ineligible_nested_text_config(self, mock_model, mock_tokenizer):
        # VLM MLA (e.g. kimi_vl): kv_lora_rank is nested under text_config, not
        # on the top-level args.
        from mlx_lm.models.cache import KVCache

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler._turboquant_kv_bits = 4.0
        scheduler.model = SimpleNamespace(
            args=SimpleNamespace(text_config=SimpleNamespace(kv_lora_rank=512))
        )
        scheduler._mla_model = None

        assert scheduler._model_uses_mla() is True
        assert scheduler._turboquant_eligible([KVCache()]) is False

    def test_mla_vlm_adapter_delegates_to_language_model(
        self, mock_model, mock_tokenizer
    ):
        # VLMModelAdapter exposes _language_model; its args surface kv_lora_rank.
        from mlx_lm.models.cache import KVCache

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler._turboquant_kv_bits = 4.0
        lm = SimpleNamespace(args=SimpleNamespace(kv_lora_rank=512))
        scheduler.model = SimpleNamespace(args=SimpleNamespace(), _language_model=lm)
        scheduler._mla_model = None

        assert scheduler._model_uses_mla() is True
        assert scheduler._turboquant_eligible([KVCache()]) is False

    def test_mla_detection_is_memoized(self, mock_model, mock_tokenizer):
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        calls = {"n": 0}

        def _modules():
            calls["n"] += 1
            return []

        scheduler.model = SimpleNamespace(args=SimpleNamespace(), modules=_modules)
        scheduler._mla_model = None

        assert scheduler._model_uses_mla() is False
        assert scheduler._model_uses_mla() is False
        assert calls["n"] == 1  # walked once, then cached


class TestTurboQuantAttentionSinkGuard:
    """Attention-sink models must not use TQ kernels that drop sink logits."""

    def test_attention_sink_model_ineligible_by_module_mapping(
        self, mock_model, mock_tokenizer
    ):
        from mlx_lm.models.cache import KVCache

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler._turboquant_kv_bits = 4.0
        scheduler.model = SimpleNamespace(
            args=SimpleNamespace(),
            modules=lambda: [{"sinks": mx.zeros((8,))}],
        )
        scheduler._mla_model = None
        scheduler._attention_sink_model = None

        assert scheduler._model_uses_attention_sinks() is True
        assert scheduler._turboquant_eligible([KVCache()]) is False

    def test_attention_sink_model_ineligible_by_module_attribute(
        self, mock_model, mock_tokenizer
    ):
        from mlx_lm.models.cache import KVCache

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler._turboquant_kv_bits = 4.0
        attn = SimpleNamespace(sinks=mx.zeros((8,)))
        scheduler.model = SimpleNamespace(
            args=SimpleNamespace(), modules=lambda: [attn]
        )
        scheduler._mla_model = None
        scheduler._attention_sink_model = None

        assert scheduler._model_uses_attention_sinks() is True
        assert scheduler._turboquant_eligible([KVCache()]) is False

    def test_standard_model_without_sinks_still_eligible(
        self, mock_model, mock_tokenizer
    ):
        from mlx_lm.models.cache import KVCache

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler._turboquant_kv_bits = 4.0
        scheduler.model = SimpleNamespace(
            args=SimpleNamespace(model_type="llama"), modules=lambda: []
        )
        scheduler._mla_model = None
        scheduler._attention_sink_model = None

        assert scheduler._model_uses_attention_sinks() is False
        assert scheduler._turboquant_eligible([KVCache()]) is True
