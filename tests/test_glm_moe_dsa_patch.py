# SPDX-License-Identifier: Apache-2.0
"""Tests for the GLM-5.2 glm_moe_dsa monkey-patch."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from omlx.utils import model_loading
from omlx.utils.model_loading import maybe_apply_pre_load_patches


def _write_config(tmp_path, body: str) -> str:
    (tmp_path / "config.json").write_text(body)
    return str(tmp_path)


def _load_patched_glm_module():
    from omlx.patches.glm_moe_dsa import apply_glm_moe_dsa_patch

    apply_glm_moe_dsa_patch()
    from mlx_lm.models import glm_moe_dsa

    return glm_moe_dsa


def _small_glm_args(glm_moe_dsa):
    return glm_moe_dsa.ModelArgs(
        model_type="glm_moe_dsa",
        vocab_size=1024,
        hidden_size=128,
        index_head_dim=16,
        index_n_heads=4,
        index_topk=4,
        intermediate_size=256,
        moe_intermediate_size=256,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=4,
        routed_scaling_factor=2.5,
        kv_lora_rank=16,
        q_lora_rank=24,
        qk_rope_head_dim=16,
        v_head_dim=32,
        qk_nope_head_dim=16,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        norm_topk_prob=True,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=2,
        moe_layer_freq=1,
        first_k_dense_replace=1,
        max_position_embeddings=1024,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_theta": 10000.0},
        attention_bias=False,
        index_topk_pattern="FSFSFS",
    )


def _wait_for_pending_writes(manager):
    import time

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        with manager._pending_write_hashes_lock:
            if not manager._pending_write_hashes:
                return
        time.sleep(0.01)
    raise AssertionError("timed out waiting for pending SSD cache writes")


def test_pre_load_dispatch_applies_glm_patch(tmp_path, monkeypatch):
    monkeypatch.setattr(model_loading, "_patch_mlx_lm_load_config", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "omlx.patches.mlx_lm_mtp",
        MagicMock(set_mtp_active=MagicMock()),
    )
    apply_mock = MagicMock(return_value=True)
    monkeypatch.setitem(
        sys.modules,
        "omlx.patches.glm_moe_dsa",
        MagicMock(apply_glm_moe_dsa_patch=apply_mock),
    )

    path = _write_config(tmp_path, '{"model_type": "glm_moe_dsa"}')
    maybe_apply_pre_load_patches(path)

    apply_mock.assert_called_once_with()


def test_glm_patch_installs_native_indexer_schedule():
    glm_moe_dsa = _load_patched_glm_module()

    fields = glm_moe_dsa.ModelArgs.__dataclass_fields__
    assert "indexer_types" in fields
    assert hasattr(glm_moe_dsa, "GlmMoeDsaModel")

    args = _small_glm_args(glm_moe_dsa)
    assert args.indexer_types == [
        "full",
        "shared",
        "full",
        "shared",
        "full",
        "shared",
    ]

    model = glm_moe_dsa.Model(args)
    assert [
        layer.self_attn.indexer is not None for layer in model.model.layers
    ] == [True, False, True, False, True, False]
    assert [len(c.caches) for c in model.make_cache()] == [2, 1, 2, 1, 2, 1]


def test_glm_patch_forward_sparse_path_and_cache_state():
    mx = pytest.importorskip("mlx.core")
    glm_moe_dsa = _load_patched_glm_module()

    args = _small_glm_args(glm_moe_dsa)
    model = glm_moe_dsa.Model(args)
    cache = model.make_cache()

    prompt = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    logits = model(prompt, cache=cache)
    assert logits.shape == (1, 8, args.vocab_size)

    nxt = mx.argmax(logits[0, -1:, :], keepdims=True)
    logits = model(nxt, cache=cache)
    assert logits.shape == (1, 1, args.vocab_size)
    assert mx.all(mx.isfinite(logits)).item()

    mx.eval([c.state for c in cache])
    full_state = cache[0].state
    shared_state = cache[1].state
    assert len(full_state) == 2
    assert len(shared_state) == 1
    assert full_state[1][1].shape[-1] == 0


def test_glm_cachelist_hot_and_cold_round_trip(tmp_path):
    mx = pytest.importorskip("mlx.core")
    glm_moe_dsa = _load_patched_glm_module()

    from omlx.cache.paged_cache import PagedCacheManager
    from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
    from omlx.cache.prefix_cache import BlockAwarePrefixCache
    from omlx.scheduler import Scheduler

    args = _small_glm_args(glm_moe_dsa)
    model = glm_moe_dsa.Model(args)
    cache = model.make_cache()
    logits = model(mx.array([[1, 2, 3, 4, 5, 6, 7, 8]]), cache=cache)
    mx.eval(logits, [c.state for c in cache])

    scheduler = MagicMock(spec=Scheduler)
    scheduler.model_name = "glm-test"
    scheduler._normalize_rotating_snapshot_state = (
        Scheduler._normalize_rotating_snapshot_state.__get__(scheduler, Scheduler)
    )
    scheduler._extract_cache_states = Scheduler._extract_cache_states.__get__(
        scheduler, Scheduler
    )
    extracted, model_cache_config = scheduler._extract_cache_states(cache)
    assert model_cache_config is not None
    assert model_cache_config.get_type_names() == ["CacheList"] * args.num_hidden_layers

    prefix_cache = BlockAwarePrefixCache(
        model=model,
        paged_cache_manager=PagedCacheManager(
            block_size=4,
            max_blocks=16,
            model_name="glm-test",
            initial_blocks=16,
        ),
    )
    block_data = prefix_cache._extract_block_tensor_slice(
        extracted,
        0,
        4,
        model_cache_config=model_cache_config,
        is_last_block=False,
    )
    assert block_data is not None
    assert block_data[0][0] == "__cache_list__"
    assert len(block_data[0][1]) == 2
    assert len(block_data[1][1]) == 1
    assert block_data[0][1][1][1].shape[-1] == 0

    block_hash = b"glm_moe_dsa_cache"
    layer_types = model_cache_config.get_type_names()
    layer_meta = model_cache_config.get_meta_states(cache)
    cache_dir = tmp_path / "glm_cache"

    manager = PagedSSDCacheManager(
        cache_dir=cache_dir,
        max_size_bytes=64 * 1024**2,
        hot_cache_max_bytes=16 * 1024**2,
    )
    try:
        assert manager.save_block(
            block_hash,
            block_data,
            token_count=4,
            model_name="glm-test",
            layer_cache_types=layer_types,
            layer_meta_states=layer_meta,
        )
        assert manager._hot_cache_get(block_hash) is not None
        hot_loaded = manager.load_block(block_hash)
        assert hot_loaded is not None
        assert len(hot_loaded[0]) == 2
        assert len(hot_loaded[1]) == 1
        assert hot_loaded[0][1][1].shape[-1] == 0
    finally:
        manager.close()

    cold_manager = PagedSSDCacheManager(
        cache_dir=cache_dir,
        max_size_bytes=64 * 1024**2,
        hot_cache_max_bytes=16 * 1024**2,
    )
    try:
        _wait_for_pending_writes(cold_manager)
        cold_loaded = cold_manager.load_block(block_hash)
        assert cold_loaded is not None
        assert len(cold_loaded[0]) == 2
        assert len(cold_loaded[1]) == 1
        assert cold_loaded[0][1][1].shape[-1] == 0
        assert cold_manager._hot_cache_get(block_hash) is not None
    finally:
        cold_manager.close()
