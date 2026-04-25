# SPDX-License-Identifier: Apache-2.0
"""Tests for tiled decode attention with online softmax accumulation.

Claim-2 from PR #757 review: decompress 4K-token tiles with online softmax
keeps throughput flat from 1K→100K context where the monolithic dequant
degrades 2.1× or OOMs.

These tests verify:
  1. Tiled output matches monolithic decode (within fp32 online-softmax
     precision when compared to the MPS reference).
  2. Memory-pressure mode disables dequant caches.
  3. Eager-packing path in update_and_fetch is consistent with the
     lazy-packing path.
"""
from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.cache.planarquant.constants import PLANAR_D
from omlx.cache.planarquant.kv_cache import PlanarQuantKVCache


def _cos_sim(a: mx.array, b: mx.array) -> float:
    af = a.astype(mx.float32).flatten()
    bf = b.astype(mx.float32).flatten()
    num = (af * bf).sum()
    den = mx.sqrt((af * af).sum()) * mx.sqrt((bf * bf).sum()) + 1e-9
    return float((num / den).item())


def _fill(seq_len: int, quantize_v: bool = True, h_k: int = 4, d: int = PLANAR_D):
    mx.random.seed(7)
    k = mx.random.normal((1, h_k, seq_len, d)) * 0.1
    v = mx.random.normal((1, h_k, seq_len, d)) * 0.1
    c = PlanarQuantKVCache(bits=3, quantize_v=quantize_v)
    c.update_and_fetch(k, v)
    c.finalize_prefill()
    return c, k, v


# ---------------------------------------------------------------------------
# Correctness: tiled vs monolithic decode attention
# ---------------------------------------------------------------------------

def test_tiled_matches_monolithic_kv_quantized():
    """decode_attention_tiled output ≈ decode_attention (both paths sum to
    the same attention function; tiled uses online softmax with fp32 acc)."""
    cache, _, _ = _fill(seq_len=128, quantize_v=True)
    q = mx.random.normal((1, 4, 1, PLANAR_D)).astype(mx.float16)
    scale = 1.0 / (PLANAR_D ** 0.5)

    out_mono = cache.decode_attention(q, scale=scale)
    out_tiled = cache.decode_attention_tiled(q, scale=scale, tile_size=32)
    sim = _cos_sim(out_mono, out_tiled)
    assert sim > 0.9995, f"tiled vs mono cos_sim={sim}"


def test_tiled_matches_monolithic_k_only():
    """quantize_v=False: V stored as fp16 — tile path must still use it."""
    cache, _, _ = _fill(seq_len=96, quantize_v=False)
    q = mx.random.normal((1, 4, 1, PLANAR_D)).astype(mx.float16)
    scale = 1.0 / (PLANAR_D ** 0.5)

    out_mono = cache.decode_attention(q, scale=scale)
    out_tiled = cache.decode_attention_tiled(q, scale=scale, tile_size=24)
    sim = _cos_sim(out_mono, out_tiled)
    assert sim > 0.9995, f"k-only tiled vs mono cos_sim={sim}"


def test_tiled_single_tile_equals_full():
    """tile_size >= offset means one tile — must match non-tiled path."""
    cache, _, _ = _fill(seq_len=48)
    q = mx.random.normal((1, 4, 1, PLANAR_D)).astype(mx.float16)
    scale = 1.0 / (PLANAR_D ** 0.5)
    out_mono = cache.decode_attention(q, scale=scale)
    out_tiled = cache.decode_attention_tiled(q, scale=scale, tile_size=1024)
    sim = _cos_sim(out_mono, out_tiled)
    assert sim > 0.9999, f"single-tile cos_sim={sim}"


def test_tiled_many_small_tiles():
    """Highly fragmented tiling exercises the online-softmax recurrence."""
    cache, _, _ = _fill(seq_len=200)
    q = mx.random.normal((1, 4, 1, PLANAR_D)).astype(mx.float16)
    scale = 1.0 / (PLANAR_D ** 0.5)
    out_mono = cache.decode_attention(q, scale=scale)
    out_tiled = cache.decode_attention_tiled(q, scale=scale, tile_size=8)
    sim = _cos_sim(out_mono, out_tiled)
    assert sim > 0.995, f"many-small-tiles cos_sim={sim}"


def test_tiled_gqa_head_repeat():
    """H_q > H_k: tiled path must repeat K/V heads to match queries."""
    # K/V has 4 heads, queries have 16 heads (n_rep=4) — typical Qwen GQA
    cache, _, _ = _fill(seq_len=96, h_k=4)
    q = mx.random.normal((1, 16, 1, PLANAR_D)).astype(mx.float16)
    scale = 1.0 / (PLANAR_D ** 0.5)
    out_mono = cache.decode_attention(q, scale=scale)
    out_tiled = cache.decode_attention_tiled(q, scale=scale, tile_size=32)
    sim = _cos_sim(out_mono, out_tiled)
    assert sim > 0.9995, f"GQA tiled cos_sim={sim}"


def test_tile_size_auto_routes_decode_attention():
    """When self.tile_size is set, decode_attention auto-routes to tiled."""
    cache, _, _ = _fill(seq_len=64)
    q = mx.random.normal((1, 4, 1, PLANAR_D)).astype(mx.float16)
    scale = 1.0 / (PLANAR_D ** 0.5)

    cache.tile_size = 16
    out_auto = cache.decode_attention(q, scale=scale)
    out_explicit = cache.decode_attention_tiled(q, scale=scale, tile_size=16)
    assert mx.array_equal(out_auto, out_explicit)


# ---------------------------------------------------------------------------
# Memory-pressure mode: dequant caches never allocated
# ---------------------------------------------------------------------------

def test_memory_pressure_mode_evicts_caches():
    """enable_memory_pressure_mode() frees _k_dequant_cache immediately."""
    cache, _, _ = _fill(seq_len=128)
    # Trigger cache allocation via a normal decode step
    q = mx.random.normal((1, 4, 1, PLANAR_D)).astype(mx.float16)
    _ = cache.decode_attention(q, scale=1.0)
    assert cache._k_dequant_cache is not None

    cache.enable_memory_pressure_mode(tile_size=32)
    assert cache._k_dequant_cache is None
    assert cache._v_dequant_cache is None
    assert cache.tile_size == 32
    assert cache.memory_pressure is True


def test_memory_pressure_default_tile_autotunes_long_contexts():
    cache = PlanarQuantKVCache(bits=3, quantize_v=True)
    cache.enable_memory_pressure_mode()

    assert cache.tile_size == 4096
    assert cache._effective_tile_size(4096, 8192) == 4096
    assert cache._effective_tile_size(4096, 16_384) == 16_384


def test_memory_pressure_explicit_tile_can_disable_autotune():
    cache = PlanarQuantKVCache(bits=3, quantize_v=True)
    cache.enable_memory_pressure_mode(auto_tile_size=False)

    assert cache._effective_tile_size(4096, 32_768) == 4096
    assert cache._effective_tile_size(1024, 32_768) == 1024


def test_memory_pressure_update_and_fetch_eager_packs():
    """Under memory_pressure, update_and_fetch writes to _k_packed directly
    and never allocates a dequant cache."""
    cache, _, _ = _fill(seq_len=64)
    cache.enable_memory_pressure_mode(tile_size=32)

    # Simulate a decode step
    new_k = mx.random.normal((1, 4, 1, PLANAR_D)) * 0.1
    new_v = mx.random.normal((1, 4, 1, PLANAR_D)) * 0.1
    offset_before = cache.offset
    cache.update_and_fetch(new_k, new_v)

    assert cache.offset == offset_before + 1
    assert cache._k_dequant_cache is None, "dequant cache should stay None"
    assert cache._v_dequant_cache is None
    # New row must be present in packed buffer
    assert cache._k_packed is not None
    assert cache._k_packed.shape[2] >= cache.offset


def test_memory_pressure_tiled_attention_correct():
    """Full memory-pressure pipeline: decode step + tiled attention produces
    the same output as the normal path (within tile-softmax precision)."""
    # Build two identical caches — one normal, one memory-pressure
    mx.random.seed(7)
    k = mx.random.normal((1, 4, 64, PLANAR_D)) * 0.1
    v = mx.random.normal((1, 4, 64, PLANAR_D)) * 0.1

    c_normal = PlanarQuantKVCache(bits=3, quantize_v=True)
    c_normal.update_and_fetch(k, v)
    c_normal.finalize_prefill()

    c_mp = PlanarQuantKVCache(bits=3, quantize_v=True)
    c_mp.update_and_fetch(k, v)
    c_mp.finalize_prefill()
    c_mp.enable_memory_pressure_mode(tile_size=16)

    # Same decode-step input
    new_k = mx.random.normal((1, 4, 1, PLANAR_D)) * 0.1
    new_v = mx.random.normal((1, 4, 1, PLANAR_D)) * 0.1
    c_normal.update_and_fetch(new_k, new_v)
    c_mp.update_and_fetch(new_k, new_v)

    q = mx.random.normal((1, 4, 1, PLANAR_D)).astype(mx.float16)
    scale = 1.0 / (PLANAR_D ** 0.5)
    out_normal = c_normal.decode_attention(q, scale=scale)
    out_mp = c_mp.decode_attention(q, scale=scale)

    sim = _cos_sim(out_normal, out_mp)
    # Normal path appends fp16 K to dequant cache; MP path quantizes new K
    # to 3-bit. One extra round of quantization — expect cos_sim > 0.99 but
    # not bit-equal.
    assert sim > 0.99, f"memory-pressure vs normal cos_sim={sim}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_tiled_empty_cache_returns_zeros():
    """tile_size set but offset=0 → zero output, no crash."""
    cache = PlanarQuantKVCache(bits=3, quantize_v=True)
    # finalize_prefill requires some init; force the minimum
    k = mx.random.normal((1, 4, 2, PLANAR_D)) * 0.1
    cache.update_and_fetch(k, k)
    cache.finalize_prefill()
    # Reset offset artificially to test T=0 branch
    cache.offset = 0
    q = mx.random.normal((1, 4, 1, PLANAR_D)).astype(mx.float16)
    out = cache.decode_attention_tiled(q, scale=1.0, tile_size=16)
    assert out.shape == q.shape
    assert float(out.abs().sum().item()) < 1e-6


def test_tiled_requires_finalize():
    """Tiled path asserts _finalized — fails helpfully in deferred mode."""
    cache = PlanarQuantKVCache(bits=3, quantize_v=True)
    k = mx.random.normal((1, 4, 8, PLANAR_D)) * 0.1
    cache.update_and_fetch(k, k)
    # NOT finalized
    q = mx.random.normal((1, 4, 1, PLANAR_D)).astype(mx.float16)
    with pytest.raises(AssertionError):
        cache.decode_attention_tiled(q, scale=1.0, tile_size=16)
