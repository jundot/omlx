# SPDX-License-Identifier: Apache-2.0
"""Serving cache metrics count restored KV, not stale prefix-index matches."""

from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache

from omlx.cache.paged_cache import PagedCacheManager
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.cache.prefix_cache import BlockAwarePrefixCache
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig


@pytest.fixture
def cached_scheduler(tmp_path, mock_model, mock_tokenizer, request):
    mock_model.config.num_hidden_layers = 1
    stateful = getattr(request, "param", False)
    if stateful:
        mock_model.make_cache = lambda: [ArraysCache(2)]
    scheduler = Scheduler(
        model=mock_model,
        tokenizer=mock_tokenizer,
        config=SchedulerConfig(paged_cache_block_size=4),
    )
    paged = PagedCacheManager(
        block_size=4, max_blocks=100, initial_blocks=100, model_name="test-model"
    )
    ssd = PagedSSDCacheManager(
        cache_dir=tmp_path / "cache",
        max_size_bytes=1024**2,
        hot_cache_max_bytes=1024**2,
        hot_cache_only=True,
        expected_model_name="test-model",
        expected_num_layers=1,
        expected_block_size=4,
    )
    paged.set_paged_ssd_cache_manager(ssd)
    prefix = BlockAwarePrefixCache(mock_model, paged, ssd)
    scheduler.block_aware_cache = prefix
    scheduler.paged_cache_manager = paged
    scheduler.paged_ssd_cache_manager = ssd
    scheduler.batch_generator = MagicMock()
    scheduler.batch_generator.insert.return_value = [42]
    scheduler._ensure_batch_generator = MagicMock()
    scheduler._build_sampler_and_processors = MagicMock(return_value=(None, []))
    scheduler._build_state_machine = MagicMock(return_value=None)
    scheduler._preflight_memory_check = MagicMock(return_value=None)
    scheduler._do_external_prefill = MagicMock(
        return_value=(mock_model.make_cache(), [8])
    )

    tokens = list(range(8))
    keys = mx.arange(8, dtype=mx.float32).reshape(1, 1, 8, 1)
    cache_data = [
        {"state": (keys, keys), "cache_type": "KVCache", "class_name": "KVCache"}
    ]
    if stateful:
        state = (mx.zeros((1, 2)), mx.zeros((1, 2)))
        cache_data = [
            {"state": state, "cache_type": "ArraysCache", "class_name": "ArraysCache"}
        ]
    table = prefix.store_cache(
        "stored",
        tokens,
        cache_data,
        boundary_snapshots={4: cache_data, 8: cache_data} if stateful else None,
    )
    assert table is not None
    prefix.clear_request_entry("stored")
    yield scheduler, table
    ssd.clear_hot_cache()
    scheduler.shutdown()


@pytest.mark.parametrize(
    ("missing_block", "prompt_length", "minimum_prefix", "expected_reused"),
    [
        (None, 9, 0, 8),
        (0, 9, 0, 0),
        (1, 9, 0, 4),
        (None, 8, 0, 7),
        (None, 9, 9, 0),
        (1, 9, 8, 0),
    ],
    ids=[
        "restored",
        "first-block-evicted",
        "partial-restore",
        "exact-hit-kickoff-trim",
        "image-prefix-rejected",
        "partial-below-image-prefix",
    ],
)
def test_serving_stats_count_only_restored_tokens(
    cached_scheduler, missing_block, prompt_length, minimum_prefix, expected_reused
):
    scheduler, table = cached_scheduler
    if missing_block is not None:
        block = scheduler.paged_cache_manager.allocated_blocks[
            table.block_ids[missing_block]
        ]
        scheduler.paged_ssd_cache_manager._hot_cache_remove(block.block_hash)
    scheduler.model.minimum_prefill_prefix = lambda tokens: minimum_prefix
    request = Request(
        request_id="lookup",
        prompt=list(range(prompt_length)),
        sampling_params=SamplingParams(max_tokens=1),
    )
    scheduler.add_request(request)
    scheduled, rejected = scheduler._schedule_waiting()

    assert scheduled == [request]
    assert rejected == []
    assert request.cached_tokens == expected_reused
    counters = scheduler._collect_cache_counters()
    assert counters["prefix_hits"] == int(expected_reused > 0)
    assert counters["prefix_misses"] == int(expected_reused == 0)
    assert counters["prefix_tokens_saved"] == expected_reused
    assert counters["prefix_tokens_matched"] == expected_reused
    assert counters["prefix_tokens_requested"] == prompt_length

    # Re-entering preparation for an already prepared request cannot count twice.
    scheduler._prepare_prefix_cache_for_request(request)
    assert scheduler._collect_cache_counters() == counters


@pytest.mark.parametrize("cached_scheduler", [True], indirect=True)
def test_stateful_exact_hit_fallback_is_counted_as_a_miss(cached_scheduler):
    scheduler, _ = cached_scheduler
    request = Request(
        request_id="stateful", prompt=list(range(8)), sampling_params=SamplingParams()
    )
    scheduler.add_request(request)
    scheduler._schedule_waiting()

    assert request.cached_tokens == 0
    assert request.prompt_cache is None
    counters = scheduler._collect_cache_counters()
    assert counters["prefix_hits"] == 0
    assert counters["prefix_misses"] == 1
    assert counters["prefix_tokens_saved"] == 0
    assert counters["prefix_tokens_matched"] == 0
    assert counters["prefix_tokens_requested"] == 8


def test_cold_lookup_is_counted_as_a_miss(cached_scheduler):
    scheduler, _ = cached_scheduler
    request = Request(
        request_id="cold", prompt=list(range(10, 19)), sampling_params=SamplingParams()
    )
    scheduler.add_request(request)
    scheduler._schedule_waiting()

    counters = scheduler._collect_cache_counters()
    assert counters["prefix_hits"] == 0
    assert counters["prefix_misses"] == 1
    assert counters["prefix_tokens_saved"] == 0
    assert counters["prefix_tokens_requested"] == 9


def test_deferred_admission_records_the_lookup_once(cached_scheduler):
    scheduler, _ = cached_scheduler
    request = Request(
        request_id="deferred", prompt=list(range(9)), sampling_params=SamplingParams()
    )
    scheduler.add_request(request)
    scheduler._specprefill_active_request_id = "other-request"
    scheduled, rejected = scheduler._schedule_waiting()
    assert scheduled == rejected == []
    counters = scheduler._collect_cache_counters()
    assert counters["prefix_hits"] == counters["prefix_misses"] == 0
    assert counters["prefix_tokens_requested"] == 0

    scheduler._specprefill_active_request_id = None
    scheduled, rejected = scheduler._schedule_waiting()
    assert scheduled == [request]
    assert rejected == []
    counters = scheduler._collect_cache_counters()
    assert counters["prefix_hits"] == 1
    assert counters["prefix_tokens_saved"] == 8
    assert counters["prefix_tokens_requested"] == 9


def test_prefill_memory_requeue_does_not_count_another_cache_hit(cached_scheduler):
    scheduler, _ = cached_scheduler
    request = Request(
        request_id="lookup", prompt=list(range(9)), sampling_params=SamplingParams()
    )
    scheduler.add_request(request)
    scheduler._schedule_waiting()
    counters = scheduler._collect_cache_counters()
    assert counters["prefix_hits"] == 1

    request.memory_requeues = 1
    request.requeue_reported_cached_tokens = request.cached_tokens
    scheduler.running.pop(request.request_id)
    scheduler.waiting.append(request)
    scheduler._prefix_cache_prepared.discard(request.request_id)
    scheduler.paged_cache_manager.delete_block_table(request.request_id)
    scheduler._schedule_waiting()

    after_requeue = scheduler._collect_cache_counters()
    assert {k: v for k, v in after_requeue.items() if k.startswith("prefix_")} == {
        k: v for k, v in counters.items() if k.startswith("prefix_")
    }


@pytest.mark.parametrize("fallback", ["validation", "sparse-prefill"])
def test_scheduling_fallback_counts_a_miss(cached_scheduler, fallback):
    scheduler, _ = cached_scheduler
    request = Request(
        request_id="fallback", prompt=list(range(9)), sampling_params=SamplingParams()
    )
    scheduler.add_request(request)
    scheduler._prepare_prefix_cache_for_request(request)
    assert request.cached_tokens == 8
    assert scheduler._collect_cache_counters()["prefix_hits"] == 0
    if fallback == "validation":
        scheduler._validate_cache = MagicMock(return_value=False)
    else:
        scheduler.model.layers = []
        request.specprefill_indices = mx.array([0])
        request.specprefill_system_end = 8
        request._specprefill_system_tokens = 0
    with patch(
        "omlx.specprefill.target.run_specprefill_target_prefill",
        side_effect=RuntimeError("sparse prefill failed mid-append"),
    ):
        scheduled, rejected = scheduler._schedule_waiting()

    assert scheduled == [request]
    assert rejected == []
    assert request.cached_tokens == 0
    assert scheduler._do_external_prefill.call_args.args[1:] == (
        request.prompt_token_ids,
        None,
    )
    counters = scheduler._collect_cache_counters()
    assert counters["prefix_hits"] == 0
    assert counters["prefix_misses"] == 1
    assert counters["prefix_tokens_saved"] == 0
    assert counters["prefix_tokens_matched"] == 0
    assert counters["prefix_tokens_requested"] == 9
