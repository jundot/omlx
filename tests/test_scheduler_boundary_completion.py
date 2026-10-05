# SPDX-License-Identifier: Apache-2.0
"""Exercise completed hybrid requests through durable prefix reuse."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import pytest
from mlx_lm.generate import BatchGenerator
from mlx_lm.models.qwen3_5 import Model, ModelArgs

from omlx.cache.boundary_snapshot_store import BoundarySnapshotSSDStore
from omlx.cache.paged_cache import PagedCacheManager
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.cache.prefix_cache import BlockAwarePrefixCache
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig


def _tiny_hybrid_model():
    return Model(
        ModelArgs(
            model_type="qwen3_5",
            text_config=dict(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=16,
                full_attention_interval=2,
                vocab_size=32,
                linear_num_value_heads=2,
                linear_num_key_heads=2,
                linear_key_head_dim=16,
                linear_value_head_dim=16,
                linear_conv_kernel_dim=4,
                partial_rotary_factor=1.0,
            ),
        )
    )


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("total_tokens", [3, 4, 5])
def test_completed_hybrid_boundary_reuses_final_response(
    mock_tokenizer, tmp_path, split, total_tokens
):
    mx.random.seed(7)
    model = _tiny_hybrid_model()
    scheduler = Scheduler(
        model=model,
        tokenizer=mock_tokenizer,
        config=SchedulerConfig(paged_cache_block_size=4, gdn_ssd_split_enabled=split),
    )
    paged = PagedCacheManager(
        block_size=4, max_blocks=16, initial_blocks=16, model_name="tiny-qwen"
    )
    ssd = PagedSSDCacheManager(
        cache_dir=tmp_path / "cache",
        max_size_bytes=16 * 1024**2,
        expected_model_name="tiny-qwen",
        expected_num_layers=2,
        expected_block_size=4,
        expected_layer_cache_types=["ArraysCache", "KVCache"],
        gdn_ssd_split_enabled=split,
    )
    boundary = BoundarySnapshotSSDStore(tmp_path / "cache")
    prefix = BlockAwarePrefixCache(
        model=model,
        paged_cache_manager=paged,
        paged_ssd_cache_manager=ssd,
        gdn_ssd_split_enabled=split,
    )
    prefix.set_gdn_checkpoint_loader(boundary.load_file)
    scheduler.paged_cache_manager = paged
    scheduler.paged_ssd_cache_manager = ssd
    scheduler._boundary_snapshot_store = boundary
    scheduler.block_aware_cache = prefix
    generator = BatchGenerator(
        model,
        max_tokens=1,
        sampler=lambda logits: mx.zeros(logits.shape[:-1], dtype=mx.int32),
        stream=scheduler._stream,
    )
    scheduler.batch_generator = generator
    tokens = list(range(3, 3 + total_tokens - 1))
    request = Request(
        request_id="finished",
        prompt=tokens,
        sampling_params=SamplingParams(max_tokens=1),
    )
    scheduler.add_request(request)
    scheduler.waiting.clear()
    scheduler.running[request.request_id] = request
    uid = generator.insert([tokens])[0]
    scheduler.request_id_to_uid[request.request_id] = uid
    scheduler.uid_to_request_id[uid] = request.request_id

    try:
        for _ in range(5):
            _, responses = generator.next()
            if responses:
                break
        assert responses[0].finish_reason == "length"
        assert responses[0].prompt_cache[-1].offset == total_tokens
        assert generator.extract_cache([uid]) == {}
        _, finished = scheduler._process_batch_responses(responses)
        assert finished == {request.request_id}
        scheduler._cleanup_finished(finished)
        for future in list(scheduler._inflight_store_futures.values()):
            future.result(timeout=10)
        scheduler._drain_pending_async_removes()

        extended = tokens + [0, 9, 10]
        table, remaining = prefix.fetch_cache("next", extended)
        # The generator prefilled directly, so no prefill tail was captured;
        # only an aligned completion boundary is reusable here. Unaligned
        # replies are not stored: the next turn renders them differently on
        # reasoning templates.
        if total_tokens % 4:
            assert table is None
            assert remaining == extended
        else:
            assert table is not None and table.num_tokens == total_tokens
            restored = prefix.reconstruct_cache(table)
            assert restored is not None
            assert remaining == [9, 10]
            with mx.stream(scheduler._stream):
                warm_logits = model(mx.array([remaining]), cache=restored)
                cold_logits = model(mx.array([extended]), cache=model.make_cache())
                mx.eval(warm_logits, cold_logits)
            assert mx.allclose(warm_logits[:, -1], cold_logits[:, -1], atol=1e-4)
    finally:
        scheduler.shutdown()


@pytest.mark.parametrize(
    "offsets,expected",
    [([None], False), ([5], False), ([4, 5], False), ([4], True)],
)
def test_completion_boundary_requires_known_consistent_positions(
    mock_model, mock_tokenizer, offsets, expected
):
    scheduler = Scheduler(
        model=mock_model,
        tokenizer=mock_tokenizer,
        config=SchedulerConfig(paged_cache_block_size=4),
    )
    scheduler._boundary_snapshot_required = True
    scheduler._on_prefill_boundary_snapshot = MagicMock()
    request = Request(
        request_id="finished",
        prompt=[3, 4, 5],
        sampling_params=SamplingParams(max_tokens=1),
    )
    request.prompt_token_ids = [3, 4, 5]
    request.num_prompt_tokens = 3
    request.output_token_ids = [6]
    cache = [
        SimpleNamespace(
            caches=[
                SimpleNamespace(offset=offset),
                SimpleNamespace(),
            ]
        )
        for offset in offsets
    ]

    scheduler._capture_finished_boundary_snapshot(request, cache)

    assert scheduler._on_prefill_boundary_snapshot.called is expected
    scheduler.shutdown()


@pytest.mark.parametrize("preserve_reasoning, expected", [(False, False), (True, True)])
def test_completion_boundary_counts_output_only_when_reasoning_is_preserved(
    mock_model, mock_tokenizer, preserve_reasoning, expected
):
    """With a think prefix the terminal boundary covers prompt + output only if the history keeps the reasoning."""
    scheduler = Scheduler(
        model=mock_model,
        tokenizer=mock_tokenizer,
        config=SchedulerConfig(paged_cache_block_size=4),
    )
    scheduler._boundary_snapshot_required = True
    scheduler._on_prefill_boundary_snapshot = MagicMock()
    request = Request(
        request_id="finished-think",
        prompt=[3, 4, 5],
        sampling_params=SamplingParams(max_tokens=1),
        preserve_reasoning=preserve_reasoning,
    )
    request.prompt_token_ids = [3, 4, 5]
    request.num_prompt_tokens = 3
    request.output_token_ids = [6]
    request.needs_think_prefix = True
    cache = [SimpleNamespace(caches=[SimpleNamespace(offset=4), SimpleNamespace()])]

    scheduler._capture_finished_boundary_snapshot(request, cache)

    assert scheduler._on_prefill_boundary_snapshot.called is expected
    scheduler.shutdown()


@pytest.mark.parametrize(
    "total_tokens,existing,expected",
    [
        (8, {}, True),
        (8, {3: object()}, True),
        (8, {8: object()}, False),
        (5, {}, False),
    ],
)
def test_completion_capture_ignores_older_tail_but_stays_aligned(
    mock_model, mock_tokenizer, total_tokens, existing, expected
):
    """A prefill tail below the end must not block the aligned completion capture."""
    scheduler = Scheduler(
        model=mock_model,
        tokenizer=mock_tokenizer,
        config=SchedulerConfig(paged_cache_block_size=4),
    )
    scheduler._boundary_snapshot_required = True
    scheduler._on_prefill_boundary_snapshot = MagicMock()
    request = Request(
        request_id="finished-tail",
        prompt=[3, 4, 5],
        sampling_params=SamplingParams(max_tokens=8),
    )
    request.prompt_token_ids = [3, 4, 5]
    request.num_prompt_tokens = 3
    request.output_token_ids = list(range(6, 6 + total_tokens - 3))
    scheduler._boundary_cache_snapshots[request.request_id] = dict(existing)
    cache = [
        SimpleNamespace(
            caches=[SimpleNamespace(offset=total_tokens), SimpleNamespace()]
        )
    ]

    scheduler._capture_finished_boundary_snapshot(request, cache)

    assert scheduler._on_prefill_boundary_snapshot.called is expected
    if expected:
        args = scheduler._on_prefill_boundary_snapshot.call_args.args
        kwargs = scheduler._on_prefill_boundary_snapshot.call_args.kwargs
        assert args[2] == total_tokens
        assert kwargs["source"] == "completion"
    scheduler.shutdown()


def _tiny_rotating_model():
    from mlx_lm.models.gemma3_text import Model, ModelArgs

    return Model(
        ModelArgs(
            model_type="gemma3_text",
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=2,
            head_dim=16,
            vocab_size=32,
            num_key_value_heads=1,
            sliding_window=4,
            sliding_window_pattern=2,
            query_pre_attn_scalar=16,
        )
    )


def _run_through_scheduler(scheduler, tokens, marker, request_id):
    """Serve one request end to end: prefix lookup, prefill, decode, store."""
    request = Request(
        request_id=request_id,
        prompt=list(tokens),
        sampling_params=SamplingParams(max_tokens=1, temperature=0),
    )
    scheduler.add_request(request)
    request.generation_prompt_start = marker
    for _ in range(50):
        scheduler.step()
        if request_id not in scheduler.requests:
            break
    for future in list(scheduler._inflight_store_futures.values()):
        future.result(timeout=30)
    scheduler._drain_pending_async_removes()
    return request


def _assert_restored_matches_cold(model, scheduler, prefix, tokens, expected_cached):
    table, remaining = prefix.fetch_cache("probe", tokens)
    assert table is not None and table.num_tokens == expected_cached
    restored = prefix.reconstruct_cache(table)
    assert restored is not None
    assert remaining == tokens[expected_cached:]
    with mx.stream(scheduler._stream):
        cold = model(mx.array([tokens]), cache=model.make_cache())[:, -1]
        warm = model(mx.array([remaining]), cache=restored)[:, -1]
        mx.eval(cold, warm)
    assert mx.allclose(warm, cold, atol=1e-4)
    prefix.release_cache("probe")


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize(
    "kind,split",
    [("gdn", False), ("gdn", True), ("rotating", False)],
)
def test_prefill_tail_reuse_matches_cold_forward(
    mock_tokenizer, tmp_path, kind, split, chunked
):
    """Tail blocks stored by the scheduler restore the same state a cold forward
    produces, for GDN (embedded and sidecar) and rotating layouts on both
    prefill paths, across a repeat, an extended prompt, and a second turn."""
    mx.random.seed(7)
    if kind == "gdn":
        model = _tiny_hybrid_model()
        layer_types = ["ArraysCache", "KVCache"]
    else:
        model = _tiny_rotating_model()
        layer_types = ["RotatingKVCache", "KVCache"]
    scheduler = Scheduler(
        model=model,
        tokenizer=mock_tokenizer,
        config=SchedulerConfig(
            paged_cache_block_size=4,
            prefill_step_size=4,
            gdn_ssd_split_enabled=split,
            chunked_prefill=chunked,
        ),
    )
    paged = PagedCacheManager(
        block_size=4, max_blocks=64, initial_blocks=64, model_name="tiny"
    )
    ssd = PagedSSDCacheManager(
        cache_dir=tmp_path / "cache",
        max_size_bytes=16 * 1024**2,
        expected_model_name="tiny",
        expected_num_layers=2,
        expected_block_size=4,
        expected_layer_cache_types=layer_types,
        gdn_ssd_split_enabled=split,
    )
    boundary = BoundarySnapshotSSDStore(tmp_path / "cache")
    prefix = BlockAwarePrefixCache(
        model=model,
        paged_cache_manager=paged,
        paged_ssd_cache_manager=ssd,
        gdn_ssd_split_enabled=split,
    )
    prefix.set_gdn_checkpoint_loader(boundary.load_file)
    scheduler.paged_cache_manager = paged
    scheduler.paged_ssd_cache_manager = ssd
    scheduler._boundary_snapshot_store = boundary
    scheduler.block_aware_cache = prefix

    try:
        # Ten tokens, marker at 7: the tail [4:7) precedes the "generation
        # prompt", the prefill then crosses the boundary at 8, and the one
        # generated token leaves the sequence off the grid. Without the
        # persists flag the boundary at 8 must not outrank the tail.
        first = list(range(3, 13))
        _run_through_scheduler(scheduler, first, 7, "turn-1")
        _assert_restored_matches_cold(model, scheduler, prefix, first, 7)
        _assert_restored_matches_cold(model, scheduler, prefix, first + [1, 2, 5], 7)

        # Second turn reuses the tail, then stores its own tail at the
        # prefill end (its marker is the last prefilled token).
        second = first + [1, 2, 5, 6]
        request = _run_through_scheduler(scheduler, second, 13, "turn-2")
        assert request.cached_tokens == 7
        _assert_restored_matches_cold(model, scheduler, prefix, second + [7, 8], 13)
    finally:
        scheduler.shutdown()
