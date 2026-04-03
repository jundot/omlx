"""Tests for TurboQuant KV cache (mlx-vlm backend + omlx BatchTurboQuantKVCache)."""

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from mlx_vlm.turboquant import (
    TurboQuantKVCache,
    _TurboQuantMSECodec,
    _TurboQuantProdCodec,
    _build_codec,
    turboquant_enabled,
)

from omlx.turboquant_kv import BatchTurboQuantKVCache


def _sample_unit_vectors(count: int, dim: int) -> mx.array:
    vectors = mx.random.normal((count, dim))
    return vectors / mx.linalg.norm(vectors, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Codec tests (ported from mlx-vlm)
# ---------------------------------------------------------------------------


def test_turboquant_mse_matches_paper_small_bit_distortions():
    vectors = _sample_unit_vectors(256, 64)
    expected = {1: 0.36, 2: 0.117, 3: 0.03}

    for bits, target in expected.items():
        codec = _TurboQuantMSECodec(64, bits, seed=0)
        state = codec.quantize(vectors)
        reconstructed = codec.dequantize(state)
        mse = mx.mean(mx.sum((vectors - reconstructed) ** 2, axis=-1)).item()
        assert mse == pytest.approx(target, rel=0.25, abs=0.02)


def test_turboquant_prod_is_nearly_unbiased_across_seeds():
    keys = _sample_unit_vectors(128, 64)
    queries = mx.random.normal((128, 64))
    true_inner_products = mx.sum(keys * queries, axis=-1)

    estimates = []
    for seed in range(16):
        codec = _TurboQuantProdCodec(64, 2, seed=seed)
        state = codec.quantize(keys)
        reconstructed = codec.dequantize(state)
        estimates.append(mx.sum(reconstructed * queries, axis=-1))

    mean_estimate = mx.mean(mx.stack(estimates), axis=0)
    bias = mx.mean(mean_estimate - true_inner_products).item()
    assert abs(bias) < 0.03


def test_fractional_turboquant_improves_reconstruction():
    vectors = mx.random.normal((1, 2, 32, 64))

    codec_3bit = _build_codec(vectors, 3.0, mode="mse", seed=0)
    codec_35bit = _build_codec(vectors, 3.5, mode="mse", seed=0)

    state_3bit = codec_3bit.quantize(vectors)
    state_35bit = codec_35bit.quantize(vectors)

    mse_3bit = mx.mean((vectors - codec_3bit.dequantize(state_3bit)) ** 2).item()
    mse_35bit = mx.mean((vectors - codec_35bit.dequantize(state_35bit)) ** 2).item()

    assert turboquant_enabled(3.5)
    assert not turboquant_enabled(3.0)
    assert mse_35bit < mse_3bit


# ---------------------------------------------------------------------------
# TurboQuantKVCache round-trip
# ---------------------------------------------------------------------------


def test_turboquant_cache_round_trip():
    keys = mx.random.normal((1, 2, 16, 32))
    values = mx.random.normal((1, 2, 16, 32))

    fp_cache = KVCache()
    fp_cache.update_and_fetch(keys, values)
    turbo_cache = TurboQuantKVCache.from_cache(fp_cache, bits=3.5)

    assert turbo_cache.offset == 16
    assert turbo_cache.nbytes < fp_cache.nbytes

    dk, dv = turbo_cache.dequantize()
    diff = mx.mean(mx.abs(keys - dk)).item()
    assert diff < 0.5  # Lossy but reasonable


# ---------------------------------------------------------------------------
# BatchTurboQuantKVCache tests (omlx-specific)
# ---------------------------------------------------------------------------


def test_batch_tq_prefill_quantizes_immediately():
    batch = BatchTurboQuantKVCache([0, 0], bits=4.0)
    keys = mx.random.normal((2, 4, 8, 32))
    values = mx.random.normal((2, 4, 8, 32))
    k_out, v_out = batch.update_and_fetch(keys, values)
    # Internal storage should be quantized immediately
    assert batch._key_state is not None
    # Returns raw fp16 input (current chunk) for model compatibility
    assert isinstance(k_out, mx.array)
    assert k_out.shape == (2, 4, 8, 32)


def test_batch_tq_decode_appends():
    batch = BatchTurboQuantKVCache([0, 0], bits=4.0)
    # Prefill
    keys = mx.random.normal((2, 4, 8, 32))
    values = mx.random.normal((2, 4, 8, 32))
    batch.update_and_fetch(keys, values)
    # Decode buffers fp16, batch-quantizes later
    dk = mx.random.normal((2, 4, 1, 32))
    dv = mx.random.normal((2, 4, 1, 32))
    batch.update_and_fetch(dk, dv)
    assert batch.total_tokens == 9
    assert batch._decode_buf_count == 1  # buffered, not yet quantized


def test_batch_tq_merge_extract():
    # Create two individual TQ caches
    c1 = TurboQuantKVCache(bits=4.0)
    c1.update_and_fetch(
        mx.random.normal((1, 2, 8, 32)),
        mx.random.normal((1, 2, 8, 32)),
    )
    c2 = TurboQuantKVCache(bits=4.0)
    c2.update_and_fetch(
        mx.random.normal((1, 2, 4, 32)),
        mx.random.normal((1, 2, 4, 32)),
    )
    mx.eval(c1.keys, c1.values, c2.keys, c2.values)

    # Merge into batch
    batch = BatchTurboQuantKVCache.merge([c1, c2])
    assert batch._key_state is not None
    assert batch._idx == 8  # max(8, 4)
    # left_padding: c1 needs 0, c2 needs 4
    assert batch.left_padding[0].item() == 0
    assert batch.left_padding[1].item() == 4

    # Extract back
    e1 = batch.extract(0)
    e2 = batch.extract(1)
    assert e1.offset == 8
    assert e2.offset == 4


def test_batch_tq_filter():
    batch = BatchTurboQuantKVCache([0, 0, 0], bits=4.0)
    keys = mx.random.normal((3, 2, 8, 32))
    values = mx.random.normal((3, 2, 8, 32))
    batch.update_and_fetch(keys, values)
    # Filter to keep only requests 0 and 2
    batch.filter([0, 2])
    assert batch._key_state.norms.shape[0] == 2


def test_batch_tq_extend():
    b1 = BatchTurboQuantKVCache([0], bits=4.0)
    b1.update_and_fetch(mx.random.normal((1, 2, 8, 32)), mx.random.normal((1, 2, 8, 32)))

    b2 = BatchTurboQuantKVCache([0], bits=4.0)
    b2.update_and_fetch(mx.random.normal((1, 2, 4, 32)), mx.random.normal((1, 2, 4, 32)))

    b1.extend(b2)
    assert b1._key_state.norms.shape[0] == 2  # Two requests in batch


def test_batch_tq_dequantize():
    batch = BatchTurboQuantKVCache([0], bits=4.0)
    keys = mx.random.normal((1, 2, 8, 32))
    values = mx.random.normal((1, 2, 8, 32))
    batch.update_and_fetch(keys, values)
    batch.update_and_fetch(mx.random.normal((1, 2, 1, 32)), mx.random.normal((1, 2, 1, 32)))
    dk, dv = batch.dequantize()
    assert dk.shape[2] == 9  # 8 prefill + 1 decode
    assert dv.shape[2] == 9


def test_batch_tq_state_property():
    batch = BatchTurboQuantKVCache([2, 0], bits=4.0)
    s = batch.state
    assert s[0] is None
    assert s[1] is None

    # Prefill — should be quantized immediately
    keys = mx.random.normal((2, 2, 4, 32))
    values = mx.random.normal((2, 2, 4, 32))
    batch.update_and_fetch(keys, values)
    s = batch.state
    assert hasattr(s[0], 'norms')  # NamedTuple state


def test_batch_tq_finalize_with_right_padding():
    batch = BatchTurboQuantKVCache([0, 0], bits=4.0)
    batch.prepare(right_padding=[2, 0])
    # Prefill (quantizes immediately)
    keys = mx.random.normal((2, 2, 8, 32))
    values = mx.random.normal((2, 2, 8, 32))
    batch.update_and_fetch(keys, values)
    assert batch._key_state is not None
    # Finalize applies right-padding via dequantize-roll-requantize
    batch.finalize()
    assert batch._right_padding is None
    assert batch._idx == 8
    # left_padding should be adjusted
    assert batch.left_padding[0].item() == 2
    assert batch.left_padding[1].item() == 0


def test_batch_tq_meta_state_round_trip():
    batch = BatchTurboQuantKVCache([0], bits=3.5, seed=42)
    batch.update_and_fetch(mx.random.normal((1, 2, 4, 32)), mx.random.normal((1, 2, 4, 32)))
    batch.update_and_fetch(mx.random.normal((1, 2, 1, 32)), mx.random.normal((1, 2, 1, 32)))

    ms = batch.meta_state
    batch2 = BatchTurboQuantKVCache([0], bits=4.0)
    batch2.meta_state = ms
    assert batch2.bits == pytest.approx(3.5)
    assert batch2.seed == 42


# ---------------------------------------------------------------------------
# Attention patch test
# ---------------------------------------------------------------------------


def test_attention_patch_routes_tq():
    from omlx.patches.turboquant_attention import apply_turboquant_attention_patch

    apply_turboquant_attention_patch()

    from mlx_lm.models import base as mlx_base

    # Create a TQ cache with some data
    fp_cache = KVCache()
    keys = mx.random.normal((1, 2, 8, 32))
    values = mx.random.normal((1, 2, 8, 32))
    fp_cache.update_and_fetch(keys, values)
    tq = TurboQuantKVCache.from_cache(fp_cache, bits=4.0)
    ks, vs = tq.state

    # Decode query (L=1) should not crash
    queries = mx.random.normal((1, 4, 1, 32))
    out = mlx_base.scaled_dot_product_attention(
        queries, ks, vs, tq, scale=32**-0.5, mask=None
    )
    assert out.shape == (1, 4, 1, 32)
