# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Qwen sparse-attention prefix-cache persistence."""

from types import SimpleNamespace


def test_qwen4_qsa_prefix_cache_round_trip_preserves_indexer_state(tmp_path):
    """QSA cache restore must retain both attention KV and indexer keys."""
    import mlx.core as mx

    from omlx.cache.hybrid_cache import ModelCacheConfig
    from omlx.cache.paged_cache import PagedCacheManager
    from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
    from omlx.cache.prefix_cache import BlockAwarePrefixCache
    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
    )
    from omlx.scheduler import Scheduler

    apply_mlx_vlm_qwen4_exp_compat_patch()

    from mlx_vlm.models.qwen4_exp.language import QSAKVCache

    source = QSAKVCache()
    keys = mx.arange(1 * 2 * 8 * 3, dtype=mx.float32).reshape(1, 2, 8, 3)
    values = keys + 100
    indexer_keys = mx.arange(1 * 8 * 5, dtype=mx.float32).reshape(1, 8, 5)
    source.update_and_fetch(keys, values)
    source.update_indexer(indexer_keys)
    mx.eval(source.keys, source.values, source.indexer_keys)

    scheduler = Scheduler.__new__(Scheduler)
    extracted, model_cache_config = scheduler._extract_cache_states([source])

    assert model_cache_config is not None
    assert extracted[0]["class_name"] == "QSAKVCache"
    assert extracted[0]["cache_type"] == "QSAKVCache"
    assert len(extracted[0]["state"]) == 3
    assert mx.array_equal(extracted[0]["state"][2], indexer_keys).item()

    paged_cache = PagedCacheManager(
        block_size=4,
        max_blocks=100,
        model_name="test-model",
        initial_blocks=100,
    )
    ssd_cache = PagedSSDCacheManager(
        cache_dir=tmp_path / "qsa-prefix-cache",
        max_size_bytes=100 * 1024**2,
        hot_cache_max_bytes=0,
        expected_model_name="test-model",
        expected_num_layers=1,
        expected_block_size=4,
    )
    paged_cache.set_paged_ssd_cache_manager(ssd_cache)
    prefix_cache = BlockAwarePrefixCache(
        model=SimpleNamespace(
            layers=[object()],
            args=SimpleNamespace(num_hidden_layers=1),
        ),
        paged_cache_manager=paged_cache,
        paged_ssd_cache_manager=ssd_cache,
    )

    try:
        model_cache_config = ModelCacheConfig.from_cache_list(
            [source], model_name="test-model"
        )
        block_table = prefix_cache.store_cache(
            "qsa-source",
            list(range(8)),
            extracted,
            model_cache_config=model_cache_config,
            hot_cache_write_back=False,
        )
        assert block_table is not None

        restored_layers = prefix_cache.reconstruct_cache(
            block_table, promote_to_hot_cache=False
        )
        assert restored_layers is not None
        restored = restored_layers[0]
        assert isinstance(restored, QSAKVCache)
        restored_keys, restored_values = restored.state
        assert restored.offset == 8
        assert restored.indexer_offset == 8
        assert mx.array_equal(restored_keys, keys).item()
        assert mx.array_equal(restored_values, values).item()
        assert mx.array_equal(restored.indexer_keys, indexer_keys).item()

        continued = restored.update_indexer(mx.full((1, 1, 5), 999.0))
        mx.eval(continued)
        assert restored.indexer_offset == 9
        assert continued.shape == (1, 9, 5)
        assert continued[0, -1, :].tolist() == [999.0] * 5
    finally:
        ssd_cache.close()
