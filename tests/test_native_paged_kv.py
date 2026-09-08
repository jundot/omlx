# SPDX-License-Identifier: Apache-2.0

import mlx.core as mx
import pytest

from omlx.native_paged_kv import (
    NativePagedKVConfigurationError,
    NativePagedPrefixCache,
    create_native_paged_kv_manager,
    validate_batch_generator_contract,
)
from omlx.scheduler import (
    _eval_prompt_cache,
    _patched_extend_cache,
    _patched_merge_caches,
)
from omlx.settings import SchedulerSettings

paged_cache = pytest.importorskip("mlx_lm.models.paged_cache")
KVBlockPool = paged_cache.KVBlockPool
PagedKVCache = paged_cache.PagedKVCache


class CompatibleBatchGenerator:
    def __init__(self, *, cache_factory=None, admission_controller=None):
        pass


class IncompatibleBatchGenerator:
    def __init__(self, *, model=None):
        pass


def test_prefill_sync_preserves_paged_and_recurrent_state_without_gather(monkeypatch):
    from mlx_lm.models.cache import ArraysCache

    pool = KVBlockPool(capacity_pages=8, page_size=4, num_kv_heads=2, key_head_dim=8)
    cache = PagedKVCache(pool)
    values = mx.ones((1, 2, 7, 8), dtype=mx.float16)
    cache._append(values, values * 2)
    recurrent = ArraysCache(size=2)
    recurrent[0] = mx.ones((1, 8)) * 3
    recurrent[1] = mx.ones((1, 8)) * 4

    def unexpected_gather():
        raise AssertionError("prefill synchronization materialized contiguous KV")

    with monkeypatch.context() as patcher:
        patcher.setattr(cache, "gather", unexpected_gather)
        _eval_prompt_cache([cache, recurrent])
    keys, actual_values = cache.gather()
    assert mx.array_equal(keys, values).item()
    assert mx.array_equal(actual_values, values * 2).item()
    assert recurrent[0].sum().item() == 24
    assert recurrent[1].sum().item() == 32
    cache.release()
    assert pool.stats().live_pages == 0


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe"])
def test_native_prefix_sharing_matches_cold_hybrid_forward(model_type):
    from mlx_lm.models.paged_cache import PagedKVCacheManager
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

    mx.random.seed(41)
    model = TextModel(
        TextModelArgs(
            model_type=model_type,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            rms_norm_eps=1e-6,
            vocab_size=128,
            max_position_embeddings=128,
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_conv_kernel_dim=4,
            full_attention_interval=2,
            num_experts=2 if model_type.endswith("moe") else 0,
            num_experts_per_tok=1 if model_type.endswith("moe") else 0,
            moe_intermediate_size=16 if model_type.endswith("moe") else 0,
            shared_expert_intermediate_size=16 if model_type.endswith("moe") else 0,
        )
    )
    manager = PagedKVCacheManager(model, capacity_pages=32, page_size=4)
    prefixes = NativePagedPrefixCache(manager, max_page_references=24)
    seed = list(range(1, 10))
    base = manager.make_cache()
    mx.eval(model(mx.array([seed]), cache=base))
    assert prefixes.store(seed, base)
    manager.release(base)
    acquired = []
    for suffix in ([31, 32, 33], [41, 42], [51, 52, 53, 54, 55]):
        tokens = seed + suffix
        hit = prefixes.acquire(tokens, request_id=str(suffix[0]), max_tokens=3)
        assert hit.matched_tokens == len(seed)
        acquired.append(hit.caches)
        cold = manager.make_cache()
        expected = model(mx.array([tokens]), cache=cold)[:, -1]
        actual = model(mx.array([suffix]), cache=hit.caches)[:, -1]
        mx.eval(actual, expected)
        assert mx.max(mx.abs(actual - expected)).item() < 1e-4
        assert mx.argmax(actual, axis=-1).item() == mx.argmax(expected, axis=-1).item()
        manager.release(cold)
    assert manager.stats().shared_pages > 0
    assert prefixes.stats()["hits"] == 3
    prefixes.clear()
    # Eviction releases tree references while active branches remain valid.
    for caches in acquired:
        mx.eval(model(mx.array([[63]]), cache=caches))
        manager.release(caches)
    assert manager.stats().live_pages == 0
    assert manager.stats().references == 0


def test_native_prefix_settings_round_trip():
    settings = SchedulerSettings.from_dict(
        {
            "native_paged_kv_cache_pages": 64,
            "native_paged_prefix_cache_pages": 128,
        }
    )
    assert settings.to_dict()["native_paged_prefix_cache_pages"] == 128


def test_disabled_native_paged_kv_does_not_require_fork():
    assert (
        create_native_paged_kv_manager(
            object(),
            capacity_pages=None,
            batch_generator_cls=IncompatibleBatchGenerator,
        )
        is None
    )


def test_batch_generator_contract_requires_page_hooks():
    with pytest.raises(NativePagedKVConfigurationError, match="cache_factory"):
        validate_batch_generator_contract(IncompatibleBatchGenerator)
    validate_batch_generator_contract(CompatibleBatchGenerator)


def test_scheduler_settings_round_trip_native_paged_kv():
    settings = SchedulerSettings.from_dict(
        {
            "native_paged_kv_cache_pages": 640,
            "native_paged_kv_page_size": 64,
            "native_paged_attention_mode": "mlx",
        }
    )
    assert settings.native_paged_kv_cache_pages == 640
    assert settings.native_paged_kv_page_size == 64
    assert settings.native_paged_attention_mode == "mlx"
    assert settings.to_dict()["native_paged_kv_cache_pages"] == 640


def test_continuous_batch_extend_joins_page_tables():
    pool = KVBlockPool(
        capacity_pages=8,
        page_size=4,
        num_kv_heads=2,
        key_head_dim=8,
        dtype=mx.float16,
    )
    first = _patched_merge_caches([[PagedKVCache(pool)]])
    second = _patched_merge_caches([[PagedKVCache(pool)]])

    merged = _patched_extend_cache(first, second)

    assert type(merged[0]).__name__ == "BatchPagedKVCache"
    assert len(merged[0].caches) == 2


@pytest.fixture
def tiny_hybrid_model():
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

    mx.random.seed(41)
    return TextModel(
        TextModelArgs(
            model_type="qwen3_5",
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            rms_norm_eps=1e-6,
            vocab_size=128,
            max_position_embeddings=128,
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_conv_kernel_dim=4,
            full_attention_interval=2,
        )
    )


def test_native_tiered_ssd_roundtrip_matches_cold(tiny_hybrid_model, tmp_path):
    from omlx.cache.paged_cache import PagedCacheManager
    from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
    from omlx.cache.prefix_cache import BlockAwarePrefixCache
    from omlx.native_paged_kv import restore_native_cache
    from omlx.scheduler import Scheduler

    model = tiny_hybrid_model
    manager = create_native_paged_kv_manager(
        model,
        capacity_pages=32,
        page_size=4,
        batch_generator_cls=CompatibleBatchGenerator,
    )
    ssd = PagedSSDCacheManager(
        cache_dir=tmp_path,
        max_size_bytes=10 * 1024**2,
        hot_cache_max_bytes=0,
        expected_model_name="tiny-hybrid",
        expected_num_layers=4,
        expected_block_size=4,
    )
    blocks = PagedCacheManager(
        block_size=4, max_blocks=32, initial_blocks=32, model_name="tiny-hybrid"
    )
    blocks.set_paged_ssd_cache_manager(ssd)
    prefix = BlockAwarePrefixCache(model, blocks, paged_ssd_cache_manager=ssd)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model_name = "tiny-hybrid"
    tokens = list(range(1, 9))
    caches = manager.make_cache()
    try:
        mx.eval(model(mx.array([tokens]), cache=caches))
        extracted, config = scheduler._extract_cache_states(caches)
        assert {item["class_name"] for item in extracted} == {"KVCache", "ArraysCache"}
        table = prefix.store_cache(
            "seed",
            tokens,
            extracted,
            model_cache_config=config,
            hot_cache_write_back=False,
        )
        assert table is not None
        blocks.delete_block_table("seed")
        manager.release(caches)
        assert manager.stats().live_pages == 0
        assert list(tmp_path.rglob("*.safetensors"))
        for suffix in ([31, 32], [41, 42, 43]):
            request_id = str(suffix[0])
            table, remaining = prefix.fetch_cache(request_id, tokens + suffix)
            assert table is not None and table.num_tokens == len(tokens)
            assert remaining == suffix
            ordinary = prefix.reconstruct_cache(table)
            restored = restore_native_cache(manager, ordinary, sequence_id=request_id)
            manager.validate_cache(restored)
            cold = manager.make_cache()
            actual = model(mx.array([suffix]), cache=restored)[:, -1]
            expected = model(mx.array([tokens + suffix]), cache=cold)[:, -1]
            mx.eval(actual, expected)
            assert mx.max(mx.abs(actual - expected)).item() < 1e-4
            manager.release(restored)
            manager.release(cold)
            blocks.delete_block_table(request_id)
        assert manager.stats().live_pages == 0
        assert manager.stats().references == 0
    finally:
        ssd.close()


def test_native_restore_rolls_back_partial_allocation(tiny_hybrid_model):
    from mlx_lm.models.cache import KVCache
    from mlx_lm.models.paged_cache import PageAllocationError, PagedKVCacheManager

    from omlx.native_paged_kv import persistent_cache_view, restore_native_cache

    manager = PagedKVCacheManager(tiny_hybrid_model, capacity_pages=2, page_size=4)
    original = manager.make_cache()
    mx.eval(tiny_hybrid_model(mx.array([[1, 2, 3, 4]]), cache=original))
    ordinary = persistent_cache_view(original)
    manager.release(original)
    # A later layer exhausts capacity after the earlier layer was imported.
    large = KVCache()
    values = mx.ones((1, 2, 9, 8))
    large.state = (values, values)
    ordinary[3] = large
    with pytest.raises(PageAllocationError):
        restore_native_cache(manager, ordinary, sequence_id="failed")
    assert manager.stats().live_pages == 0
    assert manager.stats().references == 0
    assert manager.stats().reserved_pages == 0


def test_persistent_snapshot_survives_page_reuse(tiny_hybrid_model):
    from mlx_lm.models.paged_cache import PagedKVCacheManager

    from omlx.native_paged_kv import persistent_cache_view

    manager = PagedKVCacheManager(tiny_hybrid_model, capacity_pages=2, page_size=4)
    original = manager.make_cache()
    mx.eval(tiny_hybrid_model(mx.array([[1, 2, 3, 4]]), cache=original))
    snapshot = persistent_cache_view(original)
    before = snapshot[1].state[0].tolist()
    manager.release(original)
    reuse = manager.make_cache()
    mx.eval(tiny_hybrid_model(mx.array([[11, 12, 13, 14]]), cache=reuse))
    assert snapshot[1].state[0].tolist() == before
    manager.release(reuse)


@pytest.mark.parametrize("hot_bytes", [0, 1024 * 1024])
@pytest.mark.parametrize("gdn_split", [False, True])
def test_scheduler_native_tiered_warm_decode_and_reload(
    tiny_hybrid_model, mock_tokenizer, tmp_path, hot_bytes, gdn_split
):
    import gc

    from omlx.request import Request, SamplingParams
    from omlx.scheduler import Scheduler, SchedulerConfig

    config = SchedulerConfig(
        model_name="tiny-hybrid",
        max_num_seqs=2,
        completion_batch_size=2,
        prefill_step_size=2048,
        paged_cache_block_size=2048,
        max_cache_blocks=32,
        initial_cache_blocks=32,
        native_paged_kv_cache_pages=64,
        native_paged_kv_page_size=64,
        native_paged_prefix_cache_pages=16,
        paged_ssd_cache_dir=str(tmp_path),
        paged_ssd_cache_max_size=10 * 1024**2,
        hot_cache_max_size=hot_bytes,
        gdn_ssd_split_enabled=gdn_split,
    )
    scheduler = Scheduler(tiny_hybrid_model, mock_tokenizer, config)
    manager = scheduler.native_paged_kv_manager
    tokens = [10 + i % 80 for i in range(2051)]

    def generate(scheduler, request_id, tokens):
        request = Request(
            request_id=request_id,
            prompt="",
            prompt_token_ids=tokens,
            num_prompt_tokens=len(tokens),
            sampling_params=SamplingParams(max_tokens=3, temperature=0),
        )
        scheduler.add_request(request)
        for _ in range(40):
            result = scheduler.step()
            for output in result.outputs:
                assert not output.error, output.error
            if request.is_finished():
                break
        assert request.is_finished()
        for future in list(scheduler._inflight_store_futures.values()):
            future.result(timeout=10)
        scheduler._drain_pending_async_removes()
        return request

    try:
        assert scheduler.block_aware_cache is not None
        assert scheduler.native_paged_prefix_cache is None
        cold = generate(scheduler, "cold", tokens)
        warm = generate(scheduler, "warm", tokens)
        assert warm.cached_tokens > 0
        assert warm.output_token_ids == cold.output_token_ids
        expected_tokens = list(cold.output_token_ids)
        cancelled = Request(
            request_id="cancelled",
            prompt="",
            prompt_token_ids=tokens,
            num_prompt_tokens=len(tokens),
            sampling_params=SamplingParams(max_tokens=32, temperature=0),
        )
        scheduler.add_request(cancelled)
        scheduler.step()
        assert cancelled.cached_tokens > 0
        scheduler.abort_request("cancelled")
        scheduler.step()
        assert cancelled.is_finished()
    finally:
        scheduler.shutdown()
    del scheduler, cold, warm
    gc.collect()
    assert manager.stats().live_pages == 0
    reloaded = Scheduler(tiny_hybrid_model, mock_tokenizer, config)
    try:
        restored = generate(reloaded, "reload", tokens)
        assert restored.cached_tokens > 0
        assert restored.output_token_ids == expected_tokens
    finally:
        reloaded.shutdown()


def test_settings_allow_native_pages_with_default_tiered_cache():
    from omlx.settings import GlobalSettings

    settings = GlobalSettings()
    settings.scheduler.native_paged_kv_cache_pages = 64
    settings.cache.enabled = True
    assert settings.validate() == []
