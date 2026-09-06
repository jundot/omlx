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
