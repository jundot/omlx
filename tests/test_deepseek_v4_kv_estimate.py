# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 guard-side KV pricing and prefill-profile plumbing.

The scheduler installs a V4-aware prefill profile and a latent-cache KV
override; the engines that bypass it (``set_model_info_from_model``: the
DFlash guard and every cluster rank's prefill guard) previously fell back to
dense full-context math on V4 — the same family of mis-price the planner
half of this fix removes.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.memory_monitor import (
    MemoryMonitor,
    estimate_mla_kv_bytes_per_token,
    set_model_info_from_model,
)

V4_LAYERS = 61
V4_RATIOS = [0] + [4] * 30 + [128] * 30
# pooled elems = (512 + 128) x2 (PoolingCache geometric backing capacity)
# = 1280; 30 layers at /4 + 30 at /128, ratio-0 free:
# (30 * 320 + 30 * 10) * 2 bytes = 19800 bytes/token across all layers.
V4_TOTAL_BYTES_PER_TOKEN = 19800.0


def _v4_config(**overrides):
    fields = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": V4_LAYERS,
        "num_attention_heads": 128,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "hidden_size": 7168,
        "q_lora_rank": 1024,
        "compress_ratios": list(V4_RATIOS),
        "sliding_window": 128,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "index_topk": 1024,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


class _FakeModel:
    def __init__(self, config):
        self.config = config
        self.dtype = mx.bfloat16


# ── estimate_mla_kv_bytes_per_token ──────────────────────────────────────────


def test_v4_estimate_prices_pools_not_dense_kv():
    got = estimate_mla_kv_bytes_per_token(_v4_config(), None, 2)
    assert got == pytest.approx(V4_TOTAL_BYTES_PER_TOKEN)
    # The dense fallback it replaces charged 2048 B/tok/layer * 61 layers.
    assert got < (1 * 512 * 2 * 2 * V4_LAYERS) / 6


def test_v4_estimate_scales_to_local_cache_layers():
    cache_list = [object()] * 30  # a pipeline rank's slice
    got = estimate_mla_kv_bytes_per_token(_v4_config(), cache_list, 2)
    assert got == pytest.approx(V4_TOTAL_BYTES_PER_TOKEN * 30 / V4_LAYERS)


def test_v4_estimate_without_ratios_uses_min_pool_ratio():
    config = _v4_config()
    del config.compress_ratios
    got = estimate_mla_kv_bytes_per_token(config, None, 2)
    assert got == pytest.approx(2 * (512 + 128) / 4 * 2 * V4_LAYERS)


def test_v4_estimate_requires_the_v4_shape():
    assert (
        estimate_mla_kv_bytes_per_token(
            _v4_config(num_key_value_heads=8), None, 2
        )
        is None
    )
    config = _v4_config()
    del config.q_lora_rank
    assert estimate_mla_kv_bytes_per_token(config, None, 2) is None


def test_mla_models_keep_the_latent_path():
    class _LayerCache:
        def __init__(self, n):
            self.caches = [object()] * n

    config = SimpleNamespace(
        model_type="glm_moe_dsa",
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
    )
    cache_list = [_LayerCache(2), _LayerCache(1)]
    got = estimate_mla_kv_bytes_per_token(config, cache_list, 2)
    assert got == pytest.approx((2 * (512 + 64) + 1 * 128) * 2)


def test_pool_capacity_charge_covers_real_geometric_growth():
    """The x2 pooled charge is not a fudge factor: PoolingCache regrows its
    backing buffer to ``max(needed, 2 * cap)``, so physical residency may
    reach twice the logical rows. Drive the real cache through adoption,
    chunked appends, and row-at-a-time decode growth and check the charged
    capacity bounds the actual backing at every step."""
    pytest.importorskip("mlx_lm")
    from omlx.patches.deepseek_v4.cache_extras import PoolingCache

    ratio, pooled_dim = 4, 16
    pool = PoolingCache(ratio)
    for chunk_rows in [8] + [3] * 4 + [1] * 40:
        pool.update_and_fetch(mx.zeros((1, chunk_rows, pooled_dim)))
        capacity = pool._pool_buf.shape[1]
        assert capacity <= 2 * pool._pool_len

    # The production accounting must charge at least the real backing.
    monitor = MemoryMonitor(max_kv_cache_memory=None, eviction_enabled=False)
    set_model_info_from_model(monitor, _FakeModel(_v4_config()))
    profile = monitor._prefill_memory_profile
    assert profile is not None
    charged_elems = profile._pool_cache_elements(
        pool._pool_len * ratio, ratio=ratio, pooled_dim=pooled_dim, overlap=True
    )
    assert charged_elems >= pool._pool_buf.shape[1] * pooled_dim


# ── set_model_info_from_model installs the V4 profile ────────────────────────


def test_model_info_extraction_installs_v4_profile_and_override():
    monitor = MemoryMonitor(max_kv_cache_memory=None, eviction_enabled=False)
    set_model_info_from_model(monitor, _FakeModel(_v4_config()))

    profile = getattr(monitor, "_prefill_memory_profile", None)
    assert profile is not None
    assert profile.num_attention_heads == 128
    assert profile.wsdpa_dtype_supported is True
    assert monitor._kv_bytes_per_token_override == pytest.approx(
        V4_TOTAL_BYTES_PER_TOKEN
    )


def test_non_v4_models_get_no_profile():
    monitor = MemoryMonitor(max_kv_cache_memory=None, eviction_enabled=False)
    config = SimpleNamespace(
        model_type="llama",
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        hidden_size=4096,
    )
    set_model_info_from_model(monitor, _FakeModel(config))
    assert getattr(monitor, "_prefill_memory_profile", None) is None


# ── the cluster rank guard keeps and rescales what extraction found ──────────


def test_rank_monitor_scales_profile_layers_for_pipeline_stage():
    from omlx.cluster.prefill_guard import rank_monitor

    monitor = rank_monitor(
        _FakeModel(_v4_config()), layer_count=20, tensor_parallel_size=1
    )
    assert monitor is not None
    profile = monitor._prefill_memory_profile
    assert profile is not None
    # 20 of 61 layers: whole-model pooled/window terms would over-reject
    # prompts this stage can serve; counts scale proportionally (ceil).
    assert profile.local_layers == 20
    assert profile.ratio4_layers == 10  # ceil(30 * 20/61)
    assert profile.ratio128_layers == 10
    assert profile.num_attention_heads == 128  # tp=1: heads untouched


def test_rank_monitor_prefers_exact_stage_ratios_over_proportional():
    """With the stage's layer range known, the profile must count the
    ratios actually inside it. V4's ratio-4 layers sit in one contiguous
    band; a stage holding all 30 of them prefills ~2x what the
    proportional split (15/15) would reserve for."""
    from omlx.cluster.prefill_guard import rank_monitor

    head = rank_monitor(
        _FakeModel(_v4_config()),
        layer_count=30,
        tensor_parallel_size=1,
        start_layer=1,
    )
    assert head is not None
    profile = head._prefill_memory_profile
    assert profile.local_layers == 30
    assert profile.ratio4_layers == 30
    assert profile.ratio128_layers == 0

    tail = rank_monitor(
        _FakeModel(_v4_config()),
        layer_count=30,
        tensor_parallel_size=1,
        start_layer=31,
    )
    assert tail._prefill_memory_profile.ratio4_layers == 0
    assert tail._prefill_memory_profile.ratio128_layers == 30


def test_rank_monitor_survives_unscalable_profile():
    """A profile without the expected fields must pass through unscaled,
    not break monitor construction."""
    from omlx import memory_monitor as mm
    from omlx.cluster.prefill_guard import rank_monitor

    class _Model(_FakeModel):
        pass

    model = _Model(_v4_config())
    original = mm.make_prefill_memory_profile
    try:
        mm.make_prefill_memory_profile = lambda *a, **k: SimpleNamespace(other=1)
        monitor = rank_monitor(model, layer_count=20, tensor_parallel_size=2)
    finally:
        mm.make_prefill_memory_profile = original
    assert monitor is not None
    assert getattr(monitor._prefill_memory_profile, "other", None) == 1


def test_rank_monitor_keeps_replicated_kv_override_under_tp():
    from omlx.cluster.prefill_guard import rank_monitor

    monitor = rank_monitor(
        _FakeModel(_v4_config()), layer_count=V4_LAYERS, tensor_parallel_size=2
    )
    assert monitor is not None
    # The pooled cache is replicated on every TP member — the override must
    # not be halved the way per-head KV is.
    assert monitor._kv_bytes_per_token_override == pytest.approx(
        V4_TOTAL_BYTES_PER_TOKEN
    )
    # The attention transient does shrink with this rank's head shard, and
    # at 128 heads / TP=2 the profile must see the 64-head shard the fused
    # WSDPA route actually runs with.
    assert monitor._num_attention_heads == 64
    profile = getattr(monitor, "_prefill_memory_profile", None)
    assert profile is not None
    assert profile.num_attention_heads == 64
    assert profile.index_n_heads == 64  # indexer is replicated, not sharded
    # The second set_model_info call must not reset the score dtype either.
    assert monitor._score_dtype_size == 2
