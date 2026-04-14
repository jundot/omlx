# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for PlanarQuantKVCache with packed storage + deferred quant."""

from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.cache.planarquant.constants import PLANAR_D
from omlx.cache.planarquant.kv_cache import (
    FP16State,
    PlanarQuantKVCache,
)


@pytest.fixture
def seeded_inputs():
    mx.random.seed(1)
    return mx.random.normal((1, 8, 4, PLANAR_D)) * 0.1


def test_empty_cache_state():
    cache = PlanarQuantKVCache()
    assert cache.empty()
    assert cache.size() == 0
    assert cache.offset == 0
    assert cache.nbytes == 0


def test_head_dim_not_multiple_of_planar_d_raises():
    cache = PlanarQuantKVCache()
    k = mx.zeros((1, 1, 1, 127))
    v = mx.zeros((1, 1, 1, 127))
    with pytest.raises(ValueError, match="even"):
        cache.update_and_fetch(k, v)


def test_deferred_mode_returns_fp16_states(seeded_inputs):
    """Before finalize_prefill, cache should return FP16State objects."""
    cache = PlanarQuantKVCache()
    ks, vs = cache.update_and_fetch(seeded_inputs, seeded_inputs)
    assert isinstance(ks, FP16State)
    assert isinstance(vs, FP16State)
    assert ks.shape == (1, 8, 4, PLANAR_D)
    assert not cache._finalized


def test_finalize_prefill_converts_to_packed(seeded_inputs):
    """After finalize_prefill, internal storage should be packed PlanarQuant3."""
    cache = PlanarQuantKVCache()
    cache.update_and_fetch(seeded_inputs, seeded_inputs)
    assert cache._k_fp16 is not None  # Still FP16
    cache.finalize_prefill()
    assert cache._finalized
    assert cache._k_fp16 is None  # FP16 freed
    assert cache._k_packed is not None  # Now packed


def test_decode_after_finalize_returns_fp16_via_dequant_cache(seeded_inputs):
    """After finalize_prefill + decode, update_and_fetch returns FP16 states
    backed by the dequant cache. Per-token quantization is deferred; the
    packed buffers are populated lazily on state-save via _flush_unpacked."""
    cache = PlanarQuantKVCache()
    cache.update_and_fetch(seeded_inputs, seeded_inputs)
    cache.finalize_prefill()

    # Decode token
    t = mx.random.normal((1, 8, 1, PLANAR_D)) * 0.1
    ks, vs = cache.update_and_fetch(t, t)
    assert isinstance(ks, FP16State)
    assert isinstance(vs, FP16State)
    assert cache.offset == 5  # 4 prefill + 1 decode
    # Decode row is unpacked until state serialization flushes it
    assert cache._k_unpacked_start == 4
    assert cache._k_unpacked_end == 5
    # Accessing state triggers lazy pack of the unpacked decode rows
    _ = cache.state
    assert cache._k_unpacked_start is None
    assert cache._k_unpacked_end is None


def test_multi_step_growth(seeded_inputs):
    cache = PlanarQuantKVCache()
    cache.update_and_fetch(seeded_inputs, seeded_inputs)
    cache.finalize_prefill()
    for _ in range(3):
        t = mx.random.normal((1, 8, 1, PLANAR_D)) * 0.1
        cache.update_and_fetch(t, t)
    assert cache.offset == 7


def test_dequantize_preserves_shape_and_dtype(seeded_inputs):
    cache = PlanarQuantKVCache()
    cache.update_and_fetch(seeded_inputs, seeded_inputs)
    cache.finalize_prefill()
    k, v = cache.dequantize()
    assert k.shape == seeded_inputs.shape
    assert v.shape == seeded_inputs.shape
    assert k.dtype == mx.float16
    k32, v32 = cache.dequantize(out_dtype=mx.float32)
    assert k32.dtype == mx.float32


def test_decode_attention_matches_manual_dequant_sdpa(seeded_inputs):
    cache = PlanarQuantKVCache()
    cache.update_and_fetch(seeded_inputs, seeded_inputs)
    cache.finalize_prefill()

    q = (mx.random.normal((1, 8, 1, PLANAR_D)) * 0.1).astype(mx.float16)
    scale = 1.0 / PLANAR_D**0.5

    out = cache.decode_attention(q, scale=scale)

    dq_k, dq_v = cache.dequantize(out_dtype=mx.float16)
    ref = mx.fast.scaled_dot_product_attention(q, dq_k, dq_v, scale=scale)

    out_flat = out.reshape(-1).astype(mx.float32)
    ref_flat = ref.reshape(-1).astype(mx.float32)
    dot = float(mx.sum(out_flat * ref_flat).item())
    no = float(mx.sqrt(mx.sum(out_flat * out_flat)).item())
    nr = float(mx.sqrt(mx.sum(ref_flat * ref_flat)).item())
    cos_sim = dot / (no * nr + 1e-10)
    assert cos_sim > 0.9999, f"fused vs materialized SDPA drift: cos={cos_sim}"


def test_state_meta_state_roundtrip(seeded_inputs):
    cache = PlanarQuantKVCache(bits=3.0)
    cache.update_and_fetch(seeded_inputs, seeded_inputs)
    cache.finalize_prefill()

    meta = cache.meta_state
    packed = cache.state

    cache2 = PlanarQuantKVCache()
    cache2.meta_state = meta
    cache2.state = packed

    assert cache2.offset == cache.offset
    k1, v1 = cache.dequantize()
    k2, v2 = cache2.dequantize()
    assert float(mx.max(mx.abs(k1 - k2)).item()) < 1e-4
    assert float(mx.max(mx.abs(v1 - v2)).item()) < 1e-4


def test_state_reset_via_none():
    cache = PlanarQuantKVCache()
    x = mx.random.normal((1, 1, 2, PLANAR_D))
    cache.update_and_fetch(x, x)
    cache.finalize_prefill()
    cache.state = None
    assert cache.empty()
    assert cache.offset == 0


def test_trim_reduces_offset(seeded_inputs):
    cache = PlanarQuantKVCache()
    cache.update_and_fetch(seeded_inputs, seeded_inputs)
    cache.finalize_prefill()
    n = cache.trim(2)
    assert n == 2
    assert cache.offset == 2
    n = cache.trim(100)
    assert n == 2
    assert cache.offset == 0


def test_nbytes_nonzero_after_write(seeded_inputs):
    cache = PlanarQuantKVCache()
    assert cache.nbytes == 0
    cache.update_and_fetch(seeded_inputs, seeded_inputs)
    assert cache.nbytes > 0


def test_asymmetric_v_fp16(seeded_inputs):
    """quantize_v=False keeps V as FP16 while K is PlanarQuant3 on disk.
    During decode, update_and_fetch returns FP16States (K via dequant cache,
    V from the FP16 buffer). The persisted K state is still packed — see the
    roundtrip test for serialization semantics."""
    cache = PlanarQuantKVCache(quantize_v=False)
    ks, vs = cache.update_and_fetch(seeded_inputs, seeded_inputs)
    assert isinstance(ks, FP16State)
    assert isinstance(vs, FP16State)

    cache.finalize_prefill()

    # Decode: fast path returns FP16 states pointing at the dequant K cache
    # and the FP16 V buffer, so Apple's MPS SDPA can be called directly.
    t = mx.random.normal((1, 8, 1, PLANAR_D)) * 0.1
    ks, vs = cache.update_and_fetch(t, t)
    assert isinstance(ks, FP16State)
    assert isinstance(vs, FP16State)
    # Packed K is still maintained on disk (lazy-flushed on state save)
    assert cache._k_packed is not None

    # Decode attention should work with mixed state
    q = (mx.random.normal((1, 8, 1, PLANAR_D)) * 0.1).astype(mx.float16)
    out = cache.decode_attention(q, scale=1.0/PLANAR_D**0.5)
    assert out.shape == (1, 8, 1, PLANAR_D)


def test_memory_compression_ratio(seeded_inputs):
    """Verify packed storage achieves ~5x compression vs FP16."""
    cache = PlanarQuantKVCache()
    cache.update_and_fetch(seeded_inputs, seeded_inputs)
    cache.finalize_prefill()

    # FP16 baseline: B * H * T * D * 2 bytes per K/V
    B, H, T, D = 1, 8, 4, PLANAR_D
    fp16_bytes = B * H * T * D * 2 * 2  # K + V

    pq_bytes = cache.nbytes
    ratio = fp16_bytes / pq_bytes
    # Packed: 50 bytes per 128-elem block per K and V per head
    # K+V: 2 * B * H * T * 50 = 2 * 1 * 8 * 4 * 50 = 3200
    # FP16: 2 * B * H * T * D * 2 = 2 * 1 * 8 * 4 * 128 * 2 = 16384
    # Expected ratio: ~5.12x
    assert ratio > 4.5, f"Compression ratio too low: {ratio:.2f}x (expected >4.5x)"


def test_batch_cache_b1_delegates_to_base():
    from omlx.cache.planarquant.kv_cache import BatchPlanarQuantKVCache
    cache = BatchPlanarQuantKVCache(left_padding=[0], bits=3.0)
    assert cache.offset == 0
    x = mx.random.normal((1, 4, 3, PLANAR_D)) * 0.1
    cache.update_and_fetch(x, x)
    assert cache.offset == 3


def test_make_mask_signature_delegates_to_mlx_lm():
    cache = PlanarQuantKVCache()
    assert callable(cache.make_mask)
