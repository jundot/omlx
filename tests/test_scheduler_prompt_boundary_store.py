# SPDX-License-Identifier: Apache-2.0
"""Tests for parser-stop prompt-boundary cache storage."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig


def _scheduler() -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.block_aware_cache = object()
    scheduler.config = SchedulerConfig(paged_cache_block_size=4)
    return scheduler


def _request(prompt_tokens):
    return SimpleNamespace(
        prompt_token_ids=prompt_tokens,
        specprefill_indices=None,
    )


def test_prompt_boundary_store_fills_only_sliceable_snapshot_placeholders():
    scheduler = _scheduler()
    prompt_tokens = list(range(10))
    boundary_tokens = prompt_tokens[:8]
    boundary_cache = [
        {"state": (), "class_name": "KVCache", "cache_type": "KVCache"},
        {
            "state": ("rotating-at-boundary",),
            "class_name": "RotatingKVCache",
            "cache_type": "RotatingKVCache",
        },
    ]
    live_cache = [
        {"state": ("kv-live",), "class_name": "KVCache", "cache_type": "KVCache"},
        {
            "state": ("rotating-live-tail",),
            "class_name": "RotatingKVCache",
            "cache_type": "RotatingKVCache",
        },
    ]

    scheduler._get_boundary_store_override = MagicMock(
        return_value=(boundary_tokens, boundary_cache, None, {})
    )
    scheduler._extract_live_request_cache_for_store = MagicMock(
        return_value=(live_cache, "live-config")
    )

    result = scheduler._prepare_prompt_boundary_cache_store(
        "req-parser-stop",
        _request(prompt_tokens),
        uid=7,
    )

    assert result is not None
    token_sequence, cache_to_store, model_config, intermediate_snapshots = result
    assert token_sequence == boundary_tokens
    assert cache_to_store == [live_cache[0], boundary_cache[1]]
    assert model_config == "live-config"
    assert intermediate_snapshots == {}
    scheduler._extract_live_request_cache_for_store.assert_called_once_with(
        "req-parser-stop",
        7,
        boundary_tokens,
    )


def test_prompt_boundary_store_skips_missing_snapshot_for_snapshot_models():
    scheduler = _scheduler()
    scheduler._get_boundary_store_override = MagicMock(return_value=None)
    scheduler._detect_boundary_snapshot_need = MagicMock(return_value=True)
    scheduler._extract_live_request_cache_for_store = MagicMock()

    result = scheduler._prepare_prompt_boundary_cache_store(
        "req-parser-stop",
        _request(list(range(10))),
        uid=7,
    )

    assert result is None
    scheduler._extract_live_request_cache_for_store.assert_not_called()


def test_prompt_boundary_store_uses_live_cache_for_sliceable_models():
    scheduler = _scheduler()
    prompt_tokens = list(range(10))
    boundary_tokens = prompt_tokens[:8]
    live_cache = [
        {"state": ("kv-live",), "class_name": "KVCache", "cache_type": "KVCache"},
        {
            "state": ("batch-kv-live",),
            "class_name": "BatchKVCache",
            "cache_type": "BatchKVCache",
        },
    ]
    scheduler._get_boundary_store_override = MagicMock(return_value=None)
    scheduler._detect_boundary_snapshot_need = MagicMock(return_value=False)
    scheduler._extract_live_request_cache_for_store = MagicMock(
        return_value=(live_cache, "live-config")
    )

    result = scheduler._prepare_prompt_boundary_cache_store(
        "req-parser-stop",
        _request(prompt_tokens),
        uid=7,
    )

    assert result == (boundary_tokens, live_cache, "live-config", None)
    scheduler._extract_live_request_cache_for_store.assert_called_once_with(
        "req-parser-stop",
        7,
        boundary_tokens,
    )


def test_cleanup_finished_stores_prompt_boundary_without_extracted_cache(
    mock_model,
    mock_tokenizer,
):
    scheduler = Scheduler(
        model=mock_model,
        tokenizer=mock_tokenizer,
        config=SchedulerConfig(paged_cache_block_size=4),
    )
    scheduler.block_aware_cache = MagicMock()
    scheduler.paged_cache_manager = None

    request = Request(
        request_id="req-parser-stop",
        prompt="prompt",
        sampling_params=SamplingParams(),
    )
    request.prompt_token_ids = list(range(10))
    request.num_prompt_tokens = 10
    request.output_token_ids = [100, 101]
    request._extracted_cache = None

    boundary_tokens = list(range(8))
    boundary_cache = [
        {"state": ("kv-at-boundary",), "class_name": "KVCache", "cache_type": "KVCache"}
    ]
    scheduler.running[request.request_id] = request
    scheduler.requests[request.request_id] = request
    scheduler.request_id_to_uid[request.request_id] = 7
    scheduler.uid_to_request_id[7] = request.request_id

    with (
        patch.object(
            scheduler,
            "_prepare_prompt_boundary_cache_store",
            return_value=(boundary_tokens, boundary_cache, "boundary-config", None),
        ) as prepare,
        patch.object(scheduler, "_remove_uid_from_active_batch"),
    ):
        scheduler._cleanup_finished({request.request_id})

    prepare.assert_called_once_with(request.request_id, request, 7)
    scheduler.block_aware_cache.store_cache.assert_called_once()
    args, kwargs = scheduler.block_aware_cache.store_cache.call_args
    assert args[0] == request.request_id
    assert args[1] == boundary_tokens
    assert args[2] == boundary_cache
    assert kwargs["model_cache_config"] == "boundary-config"
