"""Regression smoke tests for Qwen4 QSA long-prefill memory routing."""

import importlib
import sys
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache
from mlx_vlm.turboquant import TurboQuantKVCache


class QSAKVCache:
    def __init__(self, token_count: int):
        self.state = (
            mx.zeros((1, 2, token_count, 32), dtype=mx.float16),
            mx.zeros((1, 2, token_count, 32), dtype=mx.float16),
        )


class QSAQuantizedKVCache(QSAKVCache):
    pass


@pytest.mark.parametrize("cache_cls", [QSAKVCache, QSAQuantizedKVCache])
def test_qsa_cache_is_not_retained_in_boundary_snapshots(cache_cls):
    from omlx.scheduler import Scheduler

    captured = []
    scheduler = SimpleNamespace(
        _on_prefill_boundary_snapshot=(
            lambda request_id, snapshot_cache, token_count: captured.append(
                snapshot_cache
            )
        )
    )
    request = SimpleNamespace(request_id="qsa-smoke")

    for token_count in (16, 32, 48):
        cache = cache_cls(token_count)
        mx.eval(*cache.state)
        Scheduler._emit_prefill_boundary_snapshot(
            scheduler, request, [cache], token_count
        )

    retained_bytes = sum(
        sum(array.nbytes for array in snapshot[0].state)
        for snapshot in captured
        if snapshot[0] is not None
    )
    assert retained_bytes == 0


def test_bool_mask_uses_tiled_sdpa_and_matches_dense(monkeypatch):
    import threading

    from omlx.patches import sdpa256_attention as sdpa256

    monkeypatch.setattr(sdpa256, "_HEADROOM_PROVIDER_LOCAL", threading.local())
    monkeypatch.setattr(sdpa256, "_FORCE_TILED", None)
    monkeypatch.setattr(sdpa256, "_SDPA256_MIN_KV_LEN", 64)
    monkeypatch.setattr(sdpa256, "_Q_TILE", 16)
    monkeypatch.setattr(sdpa256, "_KV_TILE", 64)
    monkeypatch.setattr(sdpa256.mx.metal, "is_available", lambda: False)

    mx.random.seed(0)
    queries = mx.random.normal((1, 4, 32, 256)).astype(mx.float16)
    keys = mx.random.normal((1, 2, 128, 256)).astype(mx.float16)
    values = mx.random.normal((1, 2, 128, 256)).astype(mx.float16)
    mask = mx.zeros((1, 1, 32, 128), dtype=mx.bool_)
    mask[..., 64:] = True
    mx.eval(queries, keys, values, mask)

    assert sdpa256._should_route(queries, keys, None, mask, None) is True
    tiled = sdpa256._flash_sdpa256(queries, keys, values, 256**-0.5, mask)
    dense = mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=256**-0.5, mask=mask
    )
    mx.eval(tiled, dense)
    error = mx.max(mx.abs(tiled.astype(mx.float32) - dense.astype(mx.float32))).item()
    assert error < 2e-2


def test_long_bool_mask_turboquant_prefill_is_tiled_first(monkeypatch):
    from mlx_lm.models import base as mlx_base

    from omlx.patches import turboquant_attention as tq_attention

    tq_attention.apply_turboquant_attention_patch()
    monkeypatch.setattr(tq_attention, "_LONG_PREFILL_QUANTIZED_THRESHOLD", 4)

    fp_cache = KVCache()
    fp_cache.update_and_fetch(
        mx.random.normal((1, 2, 8, 32)),
        mx.random.normal((1, 2, 8, 32)),
    )
    cache = TurboQuantKVCache.from_cache(fp_cache, bits=4.0)
    keys, values = cache.state
    calls = []

    def fake_prefill(self, *args, **kwargs):
        calls.append("prefill")
        return mx.zeros_like(args[0])

    def fake_quantized(self, *args, **kwargs):
        calls.append("quantized")
        return mx.zeros_like(args[0])

    monkeypatch.setattr(TurboQuantKVCache, "prefill_attention", fake_prefill)
    monkeypatch.setattr(TurboQuantKVCache, "quantized_attention", fake_quantized)

    queries = mx.random.normal((1, 4, 2, 32))
    mask = mx.ones((1, 1, 2, 8), dtype=mx.bool_)
    result = mlx_base.scaled_dot_product_attention(
        queries, keys, values, cache, scale=32**-0.5, mask=mask
    )

    assert result.shape == queries.shape
    assert calls == ["quantized"]


def test_qwen4_mask_dense_seam_reaches_array_tiled_sdpa256(monkeypatch):
    """Production seam: on the official mask_dense path the QSA indexer
    builds an explicit array mask; with the sdpa256 patch installed that mask
    must reach _array_tiled_sdpa256 (bounded) and never the native fused call
    whose array-mask support could silently unfuse into the O(L^2) fp32
    score matrix. Uses a real Qwen4ExpAttention at production head_dim=256
    with gathered attention disabled by non-broadcast MRoPE positions."""
    from omlx import memory_monitor
    from omlx.patches import mlx_vlm_qwen4_exp_compat as compat
    from omlx.patches import sdpa256_attention as sdpa256

    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp import TextConfig
    from mlx_vlm.models.qwen4_exp.language import QSAKVCache, Qwen4ExpAttention

    cfg = TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=512,
        num_hidden_layers=1,
        num_attention_heads=8,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=3,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=16,
        moe_intermediate_size=16,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        hc_count=2,
        hc_lowrank=8,
        head_dim=256,
        layer_types=["full_attention"],
        ple_layer_ids=[],
        ple_embed_dim=32,
        ple_conv_kernel_size=3,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=16,
        indexer_compress_ratio=4,
        eos_token_id=1,
        rope_parameters={
            "rope_type": "default",
            "mrope_section": [2, 1, 1],
            "rope_theta": 10_000,
            "partial_rotary_factor": 1.0,
        },
    )
    attn = Qwen4ExpAttention(cfg)
    mx.eval(attn.parameters())

    # Keep the gathered fast path off so the official mask_dense path runs:
    # 3-D MRoPE positions with differing planes fail the gathered text
    # eligibility on both this branch (broadcast check) and upstream main
    # (2-D-only predicate), unlike the env knob which only exists here.
    prefill_len = 64  # > indexer_budget(16): the sparse mask engages
    position_ids = mx.stack(
        [
            mx.arange(prefill_len, dtype=mx.int32) * (1 + plane)
            for plane in range(3)
        ]
    ).reshape(3, 1, prefill_len)

    # Fresh sdpa256 install with test-sized KV floor.
    importlib.import_module("mlx_lm.models.base")
    importlib.import_module("mlx_vlm.models.base")
    sdpa_snap = {
        mod: mod.scaled_dot_product_attention
        for name, mod in tuple(sys.modules.items())
        if mod is not None
        and name.startswith(("mlx_lm.models.", "mlx_vlm.models."))
        and hasattr(mod, "scaled_dot_product_attention")
    }
    min_kv_len_snap = sdpa256._SDPA256_MIN_KV_LEN
    routes_snap = memory_monitor._SDPA_TILED_PREFILL_HEAD_DIMS.get(256)
    monkeypatch.setattr(sdpa256, "_PATCHED", False, raising=False)
    monkeypatch.setattr(sdpa256, "_HEADROOM_PROVIDER", None, raising=False)
    monkeypatch.setattr(sdpa256, "_FORCE_TILED", None, raising=False)
    assert sdpa256.apply_sdpa256_attention_patch(min_kv_len=32) is True

    calls = []

    def tiled(queries, keys, values, scale, mask, sinks=None):
        calls.append(mask)
        return mx.zeros(queries.shape, queries.dtype)

    def boom(*args, **kwargs):
        raise AssertionError(
            "native fused SDPA must not see the Qwen4 explicit array mask"
        )

    monkeypatch.setattr(sdpa256, "_array_tiled_sdpa256", tiled)
    monkeypatch.setattr(sdpa256.mx.fast, "scaled_dot_product_attention", boom)

    try:
        mx.random.seed(7)
        x = mx.random.normal((1, prefill_len, 512))
        cache = QSAKVCache()
        out = attn(x, mask="causal", cache=cache, position_ids=position_ids)
        mx.eval(out)

        assert out.shape == (1, prefill_len, 512)
        assert len(calls) == 1
        mask = calls[0]
        assert isinstance(mask, mx.array)
        assert 1 <= mask.ndim <= 4
        assert cache._omlx_last_prefill_gathered is False
    finally:
        for mod, fn in sdpa_snap.items():
            mod.scaled_dot_product_attention = fn
        sdpa256._SDPA256_MIN_KV_LEN = min_kv_len_snap
        if routes_snap is None:
            memory_monitor._SDPA_TILED_PREFILL_HEAD_DIMS.pop(256, None)
        else:
            memory_monitor._SDPA_TILED_PREFILL_HEAD_DIMS[256] = routes_snap
        monkeypatch.setattr(sdpa256, "_PATCHED", False, raising=False)
