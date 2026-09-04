"""Regression smoke tests for Qwen4 QSA long-prefill memory routing."""

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
    from omlx.patches import sdpa256_attention as sdpa256

    monkeypatch.setattr(sdpa256, "_HEADROOM_PROVIDER", None)
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
