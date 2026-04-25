# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for BatchPlanarQuantKVCache — continuous batching ops."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from omlx.cache.planarquant.constants import PLANAR_D
from omlx.cache.planarquant.kv_cache import (
    BatchPlanarQuantKVCache,
    FP16State,
    PlanarQuantKVCache,
    PlanarQuantState,
    _concat_packed_batch,
    _filter_packed_state,
    _packed_state_length,
    _pad_packed_left,
    _slice_packed_range,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_single_cache(T: int = 4, H: int = 4, bits: float = 3.0,
                        quantize_v: bool = True) -> PlanarQuantKVCache:
    """Create a finalized PlanarQuantKVCache with T tokens."""
    cache = PlanarQuantKVCache(bits=bits, quantize_v=quantize_v)
    x = mx.random.normal((1, H, T, PLANAR_D)) * 0.1
    cache.update_and_fetch(x, x)
    cache.finalize_prefill()
    return cache


def _make_deferred_cache(T: int = 4, H: int = 4) -> PlanarQuantKVCache:
    """Create a deferred (un-finalized) PlanarQuantKVCache."""
    cache = PlanarQuantKVCache()
    x = mx.random.normal((1, H, T, PLANAR_D)) * 0.1
    cache.update_and_fetch(x, x)
    return cache


@pytest.fixture(autouse=True)
def _seed():
    mx.random.seed(42)


# ---------------------------------------------------------------------------
# Packed-state batch helpers
# ---------------------------------------------------------------------------

class TestFilterPackedState:
    def test_basic(self):
        packed = mx.ones((4, 2, 8, 48), dtype=mx.uint8)
        norms = mx.ones((4, 2, 8, 1), dtype=mx.float16) * 2.0
        state = PlanarQuantState(packed, norms)
        filtered = _filter_packed_state(state, slice(0, 2))
        assert filtered.packed.shape == (2, 2, 8, 48)
        assert filtered.norms.shape == (2, 2, 8, 1)

    def test_list_indices(self):
        packed = mx.arange(12).reshape(3, 1, 4, 1).astype(mx.uint8)
        norms = mx.arange(12, dtype=mx.float16).reshape(3, 1, 4, 1)
        state = PlanarQuantState(packed, norms)
        filtered = _filter_packed_state(state, [2, 0])
        assert filtered.packed.shape == (2, 1, 4, 1)
        # Should have rows from index 2 and 0
        assert int(filtered.packed[0, 0, 0, 0].item()) == 8  # row 2
        assert int(filtered.packed[1, 0, 0, 0].item()) == 0  # row 0


class TestConcatPackedBatch:
    def test_two_states(self):
        s1 = PlanarQuantState(
            mx.ones((2, 2, 4, 48), dtype=mx.uint8),
            mx.ones((2, 2, 4, 1), dtype=mx.float16),
        )
        s2 = PlanarQuantState(
            mx.ones((3, 2, 4, 48), dtype=mx.uint8) * 2,
            mx.ones((3, 2, 4, 1), dtype=mx.float16) * 2,
        )
        result = _concat_packed_batch([s1, s2])
        assert result.packed.shape == (5, 2, 4, 48)
        assert result.norms.shape == (5, 2, 4, 1)


class TestPadPackedLeft:
    def test_no_pad(self):
        state = PlanarQuantState(
            mx.ones((1, 2, 4, 48), dtype=mx.uint8),
            mx.ones((1, 2, 4, 1), dtype=mx.float16),
        )
        result = _pad_packed_left(state, 0)
        assert result.packed.shape == state.packed.shape

    def test_pad_3(self):
        state = PlanarQuantState(
            mx.ones((1, 2, 4, 48), dtype=mx.uint8),
            mx.ones((1, 2, 4, 1), dtype=mx.float16),
        )
        result = _pad_packed_left(state, 3)
        assert result.packed.shape == (1, 2, 7, 48)
        assert result.norms.shape == (1, 2, 7, 1)
        # Padded rows should be zero
        assert float(mx.sum(result.packed[:, :, :3, :]).item()) == 0.0
        assert float(mx.sum(result.norms[:, :, :3, :]).item()) == 0.0
        # Original rows preserved: 1 * 2 * 4 * 48 = 384
        assert float(mx.sum(result.packed[:, :, 3:, :]).item()) == 1 * 2 * 4 * 48


class TestSlicePackedRange:
    def test_slice(self):
        packed = mx.arange(80).reshape(1, 2, 10, 4).astype(mx.uint8)
        norms = mx.arange(20, dtype=mx.float16).reshape(1, 2, 10, 1)
        state = PlanarQuantState(packed, norms)
        sliced = _slice_packed_range(state, 3, 7)
        assert sliced.packed.shape == (1, 2, 4, 4)
        assert sliced.norms.shape == (1, 2, 4, 1)


class TestPackedStateLength:
    def test_length(self):
        state = PlanarQuantState(
            mx.zeros((2, 4, 10, 48), dtype=mx.uint8),
            mx.zeros((2, 4, 10, 1), dtype=mx.float16),
        )
        assert _packed_state_length(state) == 10


# ---------------------------------------------------------------------------
# BatchPlanarQuantKVCache — init
# ---------------------------------------------------------------------------

class TestBatchInit:
    def test_b1_int_offset(self):
        cache = BatchPlanarQuantKVCache(left_padding=[0])
        assert isinstance(cache.offset, int)
        assert cache.offset == 0

    def test_b3_array_offset(self):
        cache = BatchPlanarQuantKVCache(left_padding=[2, 0, 1])
        assert isinstance(cache.offset, mx.array)
        assert cache._batch_size == 3
        # offset = [-2, 0, -1]
        assert int(cache.offset[0].item()) == -2
        assert int(cache.offset[1].item()) == 0
        assert int(cache.offset[2].item()) == -1


# ---------------------------------------------------------------------------
# update_and_fetch with B>1
# ---------------------------------------------------------------------------

class TestBatchUpdateAndFetch:
    def test_b1_delegates_to_parent(self):
        cache = BatchPlanarQuantKVCache(left_padding=[0])
        x = mx.random.normal((1, 4, 3, PLANAR_D)) * 0.1
        ks, vs = cache.update_and_fetch(x, x)
        assert isinstance(ks, FP16State)
        assert cache.offset == 3

    def test_b2_array_offset_update(self):
        cache = BatchPlanarQuantKVCache(left_padding=[1, 0])
        # Batch prefill: B=2, H=4, T=4 (with left padding)
        x = mx.random.normal((2, 4, 4, PLANAR_D)) * 0.1
        ks, vs = cache.update_and_fetch(x, x)
        # offset should have advanced by T=4 for each request
        assert isinstance(cache.offset, mx.array)
        # Initial offset: [-1, 0], after T=4: [3, 4]
        assert int(cache.offset[0].item()) == 3
        assert int(cache.offset[1].item()) == 4

    def test_left_padding_crossing_zero_preserves_valid_suffix_deferred(self):
        cache = BatchPlanarQuantKVCache(left_padding=[1, 0])
        x_np = np.zeros((2, 1, 3, PLANAR_D), dtype=np.float32)
        x_np[0, 0, 0, :] = 99.0
        x_np[0, 0, 1, :] = 1.0
        x_np[0, 0, 2, :] = 2.0
        x_np[1, 0, 0, :] = 3.0
        x_np[1, 0, 1, :] = 4.0
        x_np[1, 0, 2, :] = 5.0
        x = mx.array(x_np)

        cache.update_and_fetch(x, x)

        assert float(cache._k_fp16[0, 0, 0, 0].item()) == pytest.approx(1.0)
        assert float(cache._k_fp16[0, 0, 1, 0].item()) == pytest.approx(2.0)
        assert float(cache._k_fp16[1, 0, 0, 0].item()) == pytest.approx(3.0)
        assert float(cache._k_fp16[1, 0, 2, 0].item()) == pytest.approx(5.0)

    def test_uniform_batch_write_fast_path_matches_row_loop(self):
        buf_slow = mx.zeros((4, 2, 8, PLANAR_D), dtype=mx.float16)
        buf_fast = mx.zeros((4, 2, 8, PLANAR_D), dtype=mx.float16)
        x = mx.random.normal((4, 2, 2, PLANAR_D)).astype(mx.float16)
        offsets = mx.array([3, 3, 3, 3])

        slow, slow_start, slow_end = BatchPlanarQuantKVCache._write_batch_rows(
            buf_slow, x, offsets, try_uniform=False
        )
        fast, fast_start, fast_end = BatchPlanarQuantKVCache._write_batch_rows(
            buf_fast, x, offsets, try_uniform=True
        )

        assert slow_start == fast_start == 3
        assert slow_end == fast_end == 5
        assert mx.array_equal(slow, fast)


# ---------------------------------------------------------------------------
# make_mask
# ---------------------------------------------------------------------------

class TestBatchMakeMask:
    def test_b1_int_offset(self):
        cache = BatchPlanarQuantKVCache(left_padding=[0])
        x = mx.random.normal((1, 4, 3, PLANAR_D)) * 0.1
        cache.update_and_fetch(x, x)
        # make_mask delegates correctly for int offset
        assert callable(cache.make_mask)

    def test_b2_offset_is_array(self):
        cache = BatchPlanarQuantKVCache(left_padding=[1, 0])
        x = mx.random.normal((2, 4, 4, PLANAR_D)) * 0.1
        cache.update_and_fetch(x, x)
        # Verify offset is an array for B>1
        assert isinstance(cache.offset, mx.array)


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------

class TestBatchPrepare:
    def test_left_padding_on_empty(self):
        cache = BatchPlanarQuantKVCache(left_padding=[0, 0])
        cache.prepare(left_padding=mx.array([2, 1]))
        assert int(cache.left_padding[0].item()) == 2
        assert int(cache.left_padding[1].item()) == 1
        # offset should have decreased
        assert int(cache.offset[0].item()) == -2
        assert int(cache.offset[1].item()) == -1

    def test_right_padding_stored(self):
        cache = BatchPlanarQuantKVCache(left_padding=[0, 0])
        cache.prepare(right_padding=[1, 2])
        assert cache._right_padding is not None
        assert int(cache._right_padding[0].item()) == 1
        assert int(cache._right_padding[1].item()) == 2

    def test_left_padding_on_non_empty_raises(self):
        cache = BatchPlanarQuantKVCache(left_padding=[0])
        x = mx.random.normal((1, 4, 2, PLANAR_D)) * 0.1
        cache.update_and_fetch(x, x)
        cache.finalize_prefill()
        with pytest.raises(ValueError, match="empty"):
            cache.prepare(left_padding=mx.array([1]))


# ---------------------------------------------------------------------------
# finalize (right-padding roll)
# ---------------------------------------------------------------------------

class TestBatchFinalize:
    def test_finalize_deferred_mode(self):
        """finalize with right padding in deferred mode rolls fp16 buffers."""
        cache = BatchPlanarQuantKVCache(left_padding=[0, 0])
        # Simulate right-padded prefill
        cache.prepare(right_padding=[1, 0])
        x = mx.random.normal((2, 4, 4, PLANAR_D)) * 0.1
        cache.update_and_fetch(x, x)
        # Before finalize: right_padding is set
        assert cache._right_padding is not None
        cache.finalize()
        # After finalize: right_padding cleared, left_padding adjusted
        assert cache._right_padding is None

    def test_finalize_quantized_mode(self):
        """finalize with right padding in quantized mode rolls packed+norms."""
        cache = BatchPlanarQuantKVCache(left_padding=[0, 0])
        x = mx.random.normal((2, 4, 4, PLANAR_D)) * 0.1
        cache.update_and_fetch(x, x)
        cache.finalize_prefill()
        # Simulate right padding from a subsequent prepare
        cache._right_padding = mx.array([2, 0])
        cache.finalize()
        # After: rolled, right_padding cleared
        assert cache._right_padding is None
        # Left padding adjusted by right_padding amount
        assert int(cache.left_padding[0].item()) == 2
        assert int(cache.left_padding[1].item()) == 0

    def test_finalize_no_right_padding_noop(self):
        cache = BatchPlanarQuantKVCache(left_padding=[0, 0])
        x = mx.random.normal((2, 4, 2, PLANAR_D)) * 0.1
        cache.update_and_fetch(x, x)
        cache.finalize()  # No right padding set — no-op
        assert cache._right_padding is None


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------

class TestBatchFilter:
    def test_filter_keeps_subset(self):
        cache = BatchPlanarQuantKVCache(left_padding=[0, 0, 0])
        x = mx.random.normal((3, 4, 3, PLANAR_D)) * 0.1
        cache.update_and_fetch(x, x)
        cache.finalize_prefill()
        b_before = cache._k_packed.shape[0]
        assert b_before == 3

        cache.filter([0, 2])
        assert cache._k_packed.shape[0] == 2
        assert cache._batch_size == 2
        assert cache.offset.shape[0] == 2
        assert cache.left_padding.shape[0] == 2

    def test_filter_deferred_mode(self):
        cache = BatchPlanarQuantKVCache(left_padding=[0, 0, 0])
        x = mx.random.normal((3, 4, 3, PLANAR_D)) * 0.1
        cache.update_and_fetch(x, x)
        # Still in deferred mode
        assert not cache._finalized
        cache.filter([1])
        assert cache._k_fp16.shape[0] == 1
        assert cache._batch_size == 1

    def test_filter_resets_unpacked_ranges(self):
        cache = BatchPlanarQuantKVCache(left_padding=[0, 0])
        x = mx.random.normal((2, 4, 3, PLANAR_D)) * 0.1
        cache.update_and_fetch(x, x)
        cache.finalize_prefill()
        # Decode to create unpacked ranges
        t = mx.random.normal((2, 4, 1, PLANAR_D)) * 0.1
        cache.update_and_fetch(t, t)
        assert cache._k_unpacked_start is not None
        cache.filter([0])
        assert cache._k_unpacked_start is None
        assert cache._k_unpacked_end is None


# ---------------------------------------------------------------------------
# extend
# ---------------------------------------------------------------------------

class TestBatchExtend:
    def test_extend_two_quantized_batches(self):
        c1 = BatchPlanarQuantKVCache(left_padding=[0, 0])
        x1 = mx.random.normal((2, 4, 4, PLANAR_D)) * 0.1
        c1.update_and_fetch(x1, x1)
        c1.finalize_prefill()

        c2 = BatchPlanarQuantKVCache(left_padding=[1, 0])
        x2 = mx.random.normal((2, 4, 3, PLANAR_D)) * 0.1
        c2.update_and_fetch(x2, x2)
        c2.finalize_prefill()

        c1.extend(c2)
        assert c1._k_packed.shape[0] == 4
        assert c1._batch_size == 4
        assert c1.offset.shape[0] == 4
        assert c1.left_padding.shape[0] == 4

    def test_extend_single_to_batch(self):
        """Extend a single-request batch with a single-request batch."""
        c1 = BatchPlanarQuantKVCache(left_padding=[0])
        x1 = mx.random.normal((1, 4, 4, PLANAR_D)) * 0.1
        c1.update_and_fetch(x1, x1)
        c1.finalize_prefill()

        c2 = BatchPlanarQuantKVCache(left_padding=[0])
        x2 = mx.random.normal((1, 4, 3, PLANAR_D)) * 0.1
        c2.update_and_fetch(x2, x2)
        c2.finalize_prefill()

        c1.extend(c2)
        assert c1._k_packed.shape[0] == 2
        assert c1._batch_size == 2


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

class TestBatchMerge:
    def test_merge_two_single_caches(self):
        c1 = _make_single_cache(T=4, H=4)
        c2 = _make_single_cache(T=3, H=4)

        batch = BatchPlanarQuantKVCache.merge([c1, c2])
        assert batch._batch_size == 2
        assert batch._k_packed is not None
        assert batch._k_packed.shape[0] == 2
        # max_length = 4, so c2 (T=3) gets 1 row of left padding
        assert int(batch.left_padding[0].item()) == 0
        assert int(batch.left_padding[1].item()) == 1
        assert int(batch.offset[0].item()) == 4
        assert int(batch.offset[1].item()) == 3

    def test_merge_auto_finalizes(self):
        """merge should finalize any deferred input caches."""
        c1 = _make_deferred_cache(T=4, H=4)
        assert not c1._finalized
        BatchPlanarQuantKVCache.merge([c1])
        assert c1._finalized  # Side effect: input is finalized

    def test_merge_preserves_quantize_v(self):
        c1 = _make_single_cache(T=3, H=4, quantize_v=False)
        batch = BatchPlanarQuantKVCache.merge([c1])
        assert not batch.quantize_v

    def test_merge_three_caches(self):
        caches = [_make_single_cache(T=i + 2, H=4) for i in range(3)]
        batch = BatchPlanarQuantKVCache.merge(caches)
        assert batch._batch_size == 3
        assert batch._k_packed.shape[0] == 3

    def test_merge_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            BatchPlanarQuantKVCache.merge([])

    def test_merge_dequant_caches_carried(self):
        """Dequant caches from input caches should be carried into merged batch."""
        c1 = _make_single_cache(T=4, H=4)
        c2 = _make_single_cache(T=3, H=4)
        # Force dequant caches to exist
        c1._ensure_k_dequant_cache()
        c2._ensure_k_dequant_cache()
        batch = BatchPlanarQuantKVCache.merge([c1, c2])
        assert batch._k_dequant_cache is not None


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------

class TestBatchExtract:
    def test_extract_from_merged(self):
        c1 = _make_single_cache(T=4, H=4)
        c2 = _make_single_cache(T=3, H=4)
        batch = BatchPlanarQuantKVCache.merge([c1, c2])

        # Extract first request (no left padding)
        extracted = batch.extract(0)
        assert isinstance(extracted, PlanarQuantKVCache)
        assert extracted.offset == 4
        assert extracted._k_packed is not None

    def test_extract_with_left_padding(self):
        c1 = _make_single_cache(T=4, H=4)
        c2 = _make_single_cache(T=3, H=4)
        batch = BatchPlanarQuantKVCache.merge([c1, c2])

        # Extract second request (has left_padding=1)
        extracted = batch.extract(1)
        assert extracted.offset == 3
        assert extracted._k_packed.shape[2] == 3

    def test_extract_roundtrip_cosine(self):
        """Extracted cache should dequantize to match original."""
        c1 = _make_single_cache(T=4, H=4)
        batch = BatchPlanarQuantKVCache.merge([c1])

        extracted = batch.extract(0)
        k_orig, _ = c1.dequantize()
        k_ext, _ = extracted.dequantize()

        k1 = k_orig.reshape(-1).astype(mx.float32)
        k2 = k_ext.reshape(-1).astype(mx.float32)
        dot = float(mx.sum(k1 * k2).item())
        n1 = float(mx.sqrt(mx.sum(k1 * k1)).item())
        n2 = float(mx.sqrt(mx.sum(k2 * k2)).item())
        cos_sim = dot / (n1 * n2 + 1e-10)
        assert cos_sim > 0.999, f"Extract roundtrip cos_sim={cos_sim}"


# ---------------------------------------------------------------------------
# evict_dequant_caches
# ---------------------------------------------------------------------------

class TestBatchEvictDequantCaches:
    def test_evict_frees_memory(self):
        cache = BatchPlanarQuantKVCache(left_padding=[0])
        x = mx.random.normal((1, 4, 4, PLANAR_D)) * 0.1
        cache.update_and_fetch(x, x)
        cache.finalize_prefill()
        # Build dequant caches
        cache._ensure_k_dequant_cache()
        assert cache._k_dequant_cache is not None
        freed = cache.evict_dequant_caches()
        assert freed > 0
        assert cache._k_dequant_cache is None
        assert cache._k_dequant_offset == 0

    def test_evict_rebuild_on_decode(self):
        cache = BatchPlanarQuantKVCache(left_padding=[0])
        x = mx.random.normal((1, 4, 4, PLANAR_D)) * 0.1
        cache.update_and_fetch(x, x)
        cache.finalize_prefill()

        # Decode, then evict
        t = mx.random.normal((1, 4, 1, PLANAR_D)) * 0.1
        cache.update_and_fetch(t, t)
        cache.evict_dequant_caches()

        # Next decode should rebuild dequant caches lazily
        t2 = mx.random.normal((1, 4, 1, PLANAR_D)) * 0.1
        cache.update_and_fetch(t2, t2)
        # Dequant caches should be rebuilt
        cache._ensure_k_dequant_cache()
        assert cache._k_dequant_cache is not None


# ---------------------------------------------------------------------------
# Invariant checks
# ---------------------------------------------------------------------------

class TestBatchInvariants:
    def test_valid_after_merge(self):
        caches = [_make_single_cache(T=i + 2, H=4) for i in range(3)]
        batch = BatchPlanarQuantKVCache.merge(caches)
        violations = batch._check_invariants()
        assert violations == [], f"Invariant violations: {violations}"

    def test_valid_after_filter(self):
        cache = BatchPlanarQuantKVCache(left_padding=[0, 0, 0])
        x = mx.random.normal((3, 4, 4, PLANAR_D)) * 0.1
        cache.update_and_fetch(x, x)
        cache.finalize_prefill()
        cache.filter([0, 2])
        violations = cache._check_invariants()
        assert violations == [], f"Invariant violations: {violations}"

    def test_valid_after_extend(self):
        c1 = BatchPlanarQuantKVCache(left_padding=[0, 0])
        x1 = mx.random.normal((2, 4, 4, PLANAR_D)) * 0.1
        c1.update_and_fetch(x1, x1)
        c1.finalize_prefill()

        c2 = BatchPlanarQuantKVCache(left_padding=[0])
        x2 = mx.random.normal((1, 4, 3, PLANAR_D)) * 0.1
        c2.update_and_fetch(x2, x2)
        c2.finalize_prefill()

        c1.extend(c2)
        violations = c1._check_invariants()
        assert violations == [], f"Invariant violations: {violations}"

    def test_mismatch_detected(self):
        cache = BatchPlanarQuantKVCache(left_padding=[0])
        x = mx.random.normal((1, 4, 4, PLANAR_D)) * 0.1
        cache.update_and_fetch(x, x)
        cache.finalize_prefill()
        # Corrupt: make norms have wrong T
        cache._k_norms = mx.zeros((1, 4, 3, 1), dtype=mx.float16)
        violations = cache._check_invariants()
        assert len(violations) > 0
        assert "norms T=" in violations[0]


# ---------------------------------------------------------------------------
# Integration: batch decode attention
# ---------------------------------------------------------------------------

class TestBatchDecodeAttention:
    def test_b2_decode_after_merge(self):
        """Merged batch of 2 should support decode_attention."""
        c1 = _make_single_cache(T=4, H=4)
        c2 = _make_single_cache(T=3, H=4)
        batch = BatchPlanarQuantKVCache.merge([c1, c2])

        # Batch decode: B=2, H=4, L=1
        q = (mx.random.normal((2, 4, 1, PLANAR_D)) * 0.1).astype(mx.float16)
        out = batch.decode_attention(q, scale=1.0 / PLANAR_D**0.5)
        assert out.shape == (2, 4, 1, PLANAR_D)

    def test_decode_attention_accepts_explicit_states_from_sdpa_patch(self):
        c1 = _make_single_cache(T=4, H=4)
        c2 = _make_single_cache(T=3, H=4)
        batch = BatchPlanarQuantKVCache.merge([c1, c2])
        keys_state, values_state = batch._current_state()

        q = (mx.random.normal((2, 4, 1, PLANAR_D)) * 0.1).astype(mx.float16)
        scale = 1.0 / PLANAR_D**0.5
        out = batch.decode_attention(
            q,
            keys_state=keys_state,
            values_state=values_state,
            scale=scale,
        )

        keys, values = batch.dequantize(
            keys_state=keys_state, values_state=values_state, out_dtype=mx.float16
        )
        ref = mx.fast.scaled_dot_product_attention(q, keys, values, scale=scale)
        assert out.shape == (2, 4, 1, PLANAR_D)
        assert float(mx.max(mx.abs(out - ref)).item()) < 1e-4

    def test_state_serializes_batch_array_offset(self):
        c1 = _make_single_cache(T=4, H=4)
        c2 = _make_single_cache(T=3, H=4)
        batch = BatchPlanarQuantKVCache.merge([c1, c2])

        k_tensor, v_tensor = batch.state

        assert k_tensor.shape[0] == 2
        assert k_tensor.shape[2] == 4
        assert v_tensor.shape[0] == 2
        assert v_tensor.shape[2] == 4


# ---------------------------------------------------------------------------
# Full lifecycle: merge → extend → filter → extract
# ---------------------------------------------------------------------------

class TestBatchLifecycle:
    def test_full_lifecycle(self):
        """End-to-end: merge, decode, extend, filter, extract."""
        # 1. Create and merge 3 caches
        caches = [_make_single_cache(T=i + 3, H=4) for i in range(3)]
        batch = BatchPlanarQuantKVCache.merge(caches)
        assert batch._batch_size == 3

        # 2. Batch decode
        q = (mx.random.normal((3, 4, 1, PLANAR_D)) * 0.1).astype(mx.float16)
        t = mx.random.normal((3, 4, 1, PLANAR_D)) * 0.1
        batch.update_and_fetch(t, t)
        out = batch.decode_attention(q, scale=1.0 / PLANAR_D**0.5)
        assert out.shape == (3, 4, 1, PLANAR_D)

        # 3. Extend with a new cache
        c4 = _make_single_cache(T=5, H=4)
        c4_batch = BatchPlanarQuantKVCache.merge([c4])
        batch.extend(c4_batch)
        assert batch._batch_size == 4

        # 4. Filter out first request
        batch.filter([1, 2, 3])
        assert batch._batch_size == 3

        # 5. Extract one request
        extracted = batch.extract(0)
        assert isinstance(extracted, PlanarQuantKVCache)
        assert extracted._finalized

        # 6. Invariant check
        violations = batch._check_invariants()
        assert violations == [], f"Invariant violations after lifecycle: {violations}"

    def test_lifecycle_asymmetric_v(self):
        """Full lifecycle with quantize_v=False."""
        caches = [
            _make_single_cache(T=3, H=4, quantize_v=False),
            _make_single_cache(T=4, H=4, quantize_v=False),
        ]
        batch = BatchPlanarQuantKVCache.merge(caches)
        assert not batch.quantize_v
        assert batch._v_fp16 is not None
        assert batch._v_packed is None

        # Decode
        q = (mx.random.normal((2, 4, 1, PLANAR_D)) * 0.1).astype(mx.float16)
        out = batch.decode_attention(q, scale=1.0 / PLANAR_D**0.5)
        assert out.shape == (2, 4, 1, PLANAR_D)

        # Extract
        extracted = batch.extract(0)
        assert not extracted.quantize_v
