# SPDX-License-Identifier: Apache-2.0
"""SSD offload integration tests for PlanarQuantKVCache.

Exercises the block-level slice + concatenate + reconstruct round-trip that
the prefix-cache / paged-SSD pipeline performs when PlanarQuant3 is enabled.

No mocks — round-trip uses real mx.array operations on real packed state.
"""

from __future__ import annotations

import mlx.core as mx

from omlx.cache.planarquant.constants import PLANAR_D
from omlx.cache.planarquant.kv_cache import (
    PlanarQuantKVCache,
    _unpack_state,
)
from omlx.cache.type_registry import CacheTypeRegistry
from omlx.cache.type_handlers import CacheType


def _cos_sim(a: mx.array, b: mx.array) -> float:
    af = a.astype(mx.float32).flatten()
    bf = b.astype(mx.float32).flatten()
    num = (af * bf).sum()
    den = mx.sqrt((af * af).sum()) * mx.sqrt((bf * bf).sum()) + 1e-9
    return float((num / den).item())


def _fill_cache(quantize_v: bool, seq_len: int = 64, B: int = 1, H: int = 4):
    """Create a finalized PlanarQuantKVCache with `seq_len` tokens."""
    mx.random.seed(7)
    k = mx.random.normal((B, H, seq_len, PLANAR_D)) * 0.1
    v = mx.random.normal((B, H, seq_len, PLANAR_D)) * 0.1
    cache = PlanarQuantKVCache(bits=3, quantize_v=quantize_v)
    cache.update_and_fetch(k, v)
    cache.finalize_prefill()
    return cache, k, v


def _reconstruct(cat_k: mx.array, cat_v: mx.array, ms: tuple) -> PlanarQuantKVCache:
    """Mirror prefix_cache.py reconstruction branch for PlanarQuant."""
    bits = float(ms[1])
    quantize_v = bool(int(ms[2]))
    D_k = int(ms[3]) or None
    D_v = int(ms[4]) or None
    packed_last_k = int(ms[5]) or None
    packed_last_v = int(ms[6]) or None

    c = PlanarQuantKVCache(bits=bits, quantize_v=quantize_v)
    c._D_k = D_k
    c._D_v = D_v
    c._packed_last_k = packed_last_k
    c._packed_last_v = packed_last_v

    k_idx, k_norm = _unpack_state(cat_k, D_k, packed_last_k)
    B, H_k, T, _ = k_idx.shape
    c._B = B
    c._H_k = H_k
    c._k_packed = k_idx
    c._k_norms = k_norm
    c.offset = T
    c._cap = T
    c._finalized = True

    if quantize_v:
        v_idx, v_norm = _unpack_state(cat_v, D_v, packed_last_v)
        c._H_v = v_idx.shape[1]
        c._v_packed = v_idx
        c._v_norms = v_norm
    else:
        c._H_v = cat_v.shape[1]
        c._v_fp16 = cat_v
    return c


# ---------------------------------------------------------------------------
# 1. handler/registry wiring
# ---------------------------------------------------------------------------

def test_planarquant_registered_as_kvcache_type():
    """PlanarQuantKVCache should route to KVCACHE for block-slicing support."""
    handler = CacheTypeRegistry.get_handler_by_class_name("PlanarQuantKVCache")
    assert handler.cache_type == CacheType.KVCACHE
    assert handler.supports_block_slicing is True

    handler_batch = CacheTypeRegistry.get_handler_by_class_name(
        "BatchPlanarQuantKVCache"
    )
    assert handler_batch.cache_type == CacheType.KVCACHE
    assert handler_batch.supports_block_slicing is True


def test_meta_state_is_seven_stringified_fields():
    """Reconstruction relies on exactly these 7 fields."""
    cache, _, _ = _fill_cache(quantize_v=True, seq_len=32)
    ms = cache.meta_state
    assert isinstance(ms, tuple)
    assert len(ms) == 7
    for field in ms:
        assert isinstance(field, str)
    # offset, bits, quantize_v, D_k, D_v, packed_last_k, packed_last_v
    assert int(ms[0]) == 32
    assert float(ms[1]) == 3.0
    assert int(ms[2]) == 1  # quantize_v=True
    assert int(ms[3]) == PLANAR_D
    assert int(ms[4]) == PLANAR_D


# ---------------------------------------------------------------------------
# 2. K+V quantized round-trip
# ---------------------------------------------------------------------------

def test_kv_quantized_single_block_roundtrip():
    """1 block: extract state → reconstruct → same packed content."""
    orig, _, _ = _fill_cache(quantize_v=True, seq_len=48)
    k_state, v_state = orig.state
    ms = orig.meta_state

    restored = _reconstruct(k_state, v_state, ms)

    assert restored.offset == orig.offset
    T = restored.offset
    assert mx.array_equal(restored._k_packed, orig._k_packed[..., :T, :])
    assert mx.array_equal(restored._v_packed, orig._v_packed[..., :T, :])
    assert mx.allclose(restored._k_norms, orig._k_norms[..., :T, :]).item()
    assert mx.allclose(restored._v_norms, orig._v_norms[..., :T, :]).item()


def test_kv_quantized_multi_block_concat_roundtrip():
    """Split state into 3 blocks along seq axis, concat back, reconstruct."""
    orig, _, _ = _fill_cache(quantize_v=True, seq_len=48)
    k_state, v_state = orig.state
    ms = orig.meta_state

    # Split seq_len=48 into 3 blocks of 16
    k_blocks = [k_state[:, :, 0:16, :], k_state[:, :, 16:32, :], k_state[:, :, 32:48, :]]
    v_blocks = [v_state[:, :, 0:16, :], v_state[:, :, 16:32, :], v_state[:, :, 32:48, :]]

    cat_k = mx.concatenate(k_blocks, axis=2)
    cat_v = mx.concatenate(v_blocks, axis=2)

    restored = _reconstruct(cat_k, cat_v, ms)

    assert restored.offset == 48
    T = restored.offset
    assert mx.array_equal(restored._k_packed, orig._k_packed[..., :T, :])
    assert mx.array_equal(restored._v_packed, orig._v_packed[..., :T, :])


# ---------------------------------------------------------------------------
# 3. K-only quantized (quantize_v=False) round-trip
# ---------------------------------------------------------------------------

def test_k_only_single_block_roundtrip():
    orig, _, v = _fill_cache(quantize_v=False, seq_len=32)
    k_state, v_state = orig.state
    ms = orig.meta_state

    # v_state is plain fp16 tensor with shape (B, H, T, D)
    assert v_state.dtype == mx.float16
    assert v_state.shape[2] == 32

    restored = _reconstruct(k_state, v_state, ms)

    assert restored.offset == 32
    assert restored.quantize_v is False
    assert restored._v_packed is None
    assert restored._v_fp16 is not None
    T = restored.offset
    assert mx.array_equal(restored._k_packed, orig._k_packed[..., :T, :])
    assert mx.allclose(restored._v_fp16, orig._v_fp16[..., :T, :]).item()


def test_k_only_multi_block_concat_roundtrip():
    orig, _, v = _fill_cache(quantize_v=False, seq_len=48)
    k_state, v_state = orig.state
    ms = orig.meta_state

    # 2 blocks of 24
    cat_k = mx.concatenate(
        [k_state[:, :, :24, :], k_state[:, :, 24:, :]], axis=2
    )
    cat_v = mx.concatenate(
        [v_state[:, :, :24, :], v_state[:, :, 24:, :]], axis=2
    )

    restored = _reconstruct(cat_k, cat_v, ms)

    assert restored.offset == 48
    assert restored.quantize_v is False
    T = restored.offset
    assert mx.array_equal(restored._k_packed, orig._k_packed[..., :T, :])
    assert mx.allclose(restored._v_fp16, orig._v_fp16[..., :T, :]).item()


# ---------------------------------------------------------------------------
# 4. Dequantization quality survives round-trip (cos_sim)
# ---------------------------------------------------------------------------

def test_dequant_output_cos_sim_after_roundtrip():
    """Dequantized K from reconstructed cache ≈ original dequantized K."""
    orig, k_orig, _ = _fill_cache(quantize_v=True, seq_len=64)
    k_state, v_state = orig.state
    ms = orig.meta_state

    # 4 blocks of 16
    blocks_k = [k_state[:, :, i * 16:(i + 1) * 16, :] for i in range(4)]
    blocks_v = [v_state[:, :, i * 16:(i + 1) * 16, :] for i in range(4)]
    cat_k = mx.concatenate(blocks_k, axis=2)
    cat_v = mx.concatenate(blocks_v, axis=2)

    restored = _reconstruct(cat_k, cat_v, ms)

    orig._ensure_k_dequant_cache()
    restored._ensure_k_dequant_cache()
    # Compare dequantized K rows for valid tokens
    k_orig_dq = orig._k_dequant_cache[..., :orig.offset, :]
    k_rest_dq = restored._k_dequant_cache[..., :restored.offset, :]
    assert _cos_sim(k_orig_dq, k_rest_dq) > 0.9999


# ---------------------------------------------------------------------------
# 5. Meta-state tuple round-trip through str() (paged_ssd_cache's channel)
# ---------------------------------------------------------------------------

def test_meta_state_str_roundtrip():
    """paged_ssd_cache JSON-encodes meta_state as list of strings. Ensure
    round-tripping (tuple → list of str → tuple) preserves reconstruction."""
    orig, _, _ = _fill_cache(quantize_v=True, seq_len=24)
    ms_original = orig.meta_state
    # Mimic paged_ssd_cache storage: list of str → tuple after reload
    ms_listform = [str(x) for x in ms_original]
    ms_restored = tuple(ms_listform)
    assert ms_restored == ms_original

    k_state, v_state = orig.state
    restored = _reconstruct(k_state, v_state, ms_restored)
    assert restored.offset == orig.offset
    assert restored.bits == orig.bits
    assert restored.quantize_v == orig.quantize_v
