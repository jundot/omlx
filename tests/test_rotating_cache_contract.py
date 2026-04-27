# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the RotatingKVCache contract with mlx-lm 0.31.3.

Issues #934 (infinite loop), #903 (empty content), and #900 (preserve_thinking
flakiness) all traced back to omlx feeding mlx-lm a RotatingKVCache whose
shape and meta_state did not match the new BatchRotatingKVCache.merge()
semantics introduced by mlx-lm PR #1072. After the fix, omlx's restore path
must satisfy:

1. ``size()`` reports a length the merge slice can actually fill — clamping
   to ``keys.shape[2]`` when the buffer is shorter than ``min(offset, max_size)``.
2. The buffer is in temporal order (case 1 of ``_temporal_order``), so
   ``_idx == keys.shape[2]``. No phantom zero positions get padded into the
   merge buffer.
3. Heterogeneous merges (a restored cache and a fresh empty one) place the
   restored row's real tokens at the right position, with left_padding
   covering the gap.

These tests pin those invariants by exercising the real mlx-lm classes.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def mx():
    try:
        import mlx.core as mx_module

        return mx_module
    except ImportError:
        pytest.skip("MLX not available")


@pytest.fixture
def mlx_lm_classes():
    try:
        from mlx_lm.models.cache import BatchRotatingKVCache, RotatingKVCache
    except ImportError:
        pytest.skip("mlx-lm not available")
    return RotatingKVCache, BatchRotatingKVCache


class TestPrefillReadyRotatingKVCacheSize:
    """size() must reflect the actual buffer length, not just the offset."""

    def test_zero_length_buffer_reports_zero(self, mx, mlx_lm_classes):
        from omlx.cache._rotating_subclass import PrefillReadyRotatingKVCache

        cache = PrefillReadyRotatingKVCache(max_size=128, keep=0)
        cache.keys = mx.zeros((1, 4, 0, 32))
        cache.values = mx.zeros((1, 4, 0, 32))
        cache.offset = 4096  # rotation wrapped many times
        cache._idx = 0

        # super().size() would return min(4096, 128) = 128 here.
        assert cache.size() == 0

    def test_partial_buffer_clamps_to_buffer_length(self, mx, mlx_lm_classes):
        from omlx.cache._rotating_subclass import PrefillReadyRotatingKVCache

        cache = PrefillReadyRotatingKVCache(max_size=128, keep=0)
        cache.keys = mx.zeros((1, 4, 64, 32))
        cache.values = mx.zeros((1, 4, 64, 32))
        cache.offset = 4096
        cache._idx = 64

        # super().size() == min(4096, 128) == 128, but only 64 entries exist.
        assert cache.size() == 64

    def test_full_buffer_unchanged(self, mx, mlx_lm_classes):
        from omlx.cache._rotating_subclass import PrefillReadyRotatingKVCache

        cache = PrefillReadyRotatingKVCache(max_size=128, keep=0)
        cache.keys = mx.zeros((1, 4, 128, 32))
        cache.values = mx.zeros((1, 4, 128, 32))
        cache.offset = 256
        cache._idx = 128

        assert cache.size() == 128

    def test_empty_keys_none_reports_zero(self, mx, mlx_lm_classes):
        from omlx.cache._rotating_subclass import PrefillReadyRotatingKVCache

        cache = PrefillReadyRotatingKVCache(max_size=128, keep=0)
        # keys never populated
        assert cache.size() == 0


class TestReconstructCacheUndersized:
    """Restored caches with shorter buffers must round-trip through merge."""

    def test_merge_with_fresh_empty_does_not_overshoot(self, mx, mlx_lm_classes):
        from omlx.cache._rotating_subclass import PrefillReadyRotatingKVCache
        from omlx.cache.type_handlers import RotatingKVCacheHandler

        _, batch_rotating = mlx_lm_classes
        handler = RotatingKVCacheHandler()

        # Emulate an extract() output: 100 real tokens, max_size 128,
        # offset 4096 (rotation has wrapped). The buffer is in temporal
        # order with the most recent 100 tokens at indices [0..99].
        keys = mx.arange(100, dtype=mx.float32).reshape(1, 1, 100, 1)
        values = mx.arange(100, dtype=mx.float32).reshape(1, 1, 100, 1)
        state = {"keys": keys, "values": values}
        meta_state = (0, 128, 4096, 100)

        restored = handler.reconstruct_cache(state, meta_state)
        assert restored is not None
        assert isinstance(restored, PrefillReadyRotatingKVCache)
        assert restored.size() == 100
        assert restored._idx == 100
        assert restored.keys.shape[2] == 100

        # Build a fresh empty cache (the kind a brand-new request brings).
        fresh = PrefillReadyRotatingKVCache(max_size=128, keep=0)
        fresh.keys = mx.zeros((1, 1, 0, 1))
        fresh.values = mx.zeros((1, 1, 0, 1))
        fresh.offset = 0
        fresh._idx = 0

        merged = batch_rotating.merge([restored, fresh])
        # max_length = max(100, 0) = 100, so the merged buffer is 100 wide.
        assert merged.keys.shape[2] == 100
        # Row 0 (restored) holds the original tokens at the trailing 100 slots.
        row0 = merged.keys[0, 0, :, 0]
        for i in range(100):
            assert float(row0[i].item()) == float(i)
        # Row 1 (fresh) is left-padded by 100 zeros.
        row1 = merged.keys[1, 0, :, 0]
        assert all(float(v.item()) == 0.0 for v in row1)

    def test_two_restored_caches_align_correctly(self, mx, mlx_lm_classes):
        from omlx.cache.type_handlers import RotatingKVCacheHandler

        _, batch_rotating = mlx_lm_classes
        handler = RotatingKVCacheHandler()

        long_keys = mx.arange(80, dtype=mx.float32).reshape(1, 1, 80, 1)
        short_keys = mx.arange(80, 80 + 30, dtype=mx.float32).reshape(1, 1, 30, 1)
        long = handler.reconstruct_cache(
            {"keys": long_keys, "values": long_keys},
            (0, 128, 4096, 80),
        )
        short = handler.reconstruct_cache(
            {"keys": short_keys, "values": short_keys},
            (0, 128, 200, 30),
        )
        assert long.size() == 80
        assert short.size() == 30

        merged = batch_rotating.merge([long, short])
        # max_length = 80, padding = [0, 50].
        assert merged.keys.shape[2] == 80
        # Long row gets 80 entries starting at offset 0.
        row_long = merged.keys[0, 0, :, 0]
        for i in range(80):
            assert float(row_long[i].item()) == float(i)
        # Short row: 50 zero pads + 30 real entries (80..109).
        row_short = merged.keys[1, 0, :, 0]
        for i in range(50):
            assert float(row_short[i].item()) == 0.0
        for i in range(30):
            assert float(row_short[50 + i].item()) == float(80 + i)


class TestNormalizeRotatingStateIdx:
    """``_normalize_rotating_state`` should always emit ``_idx == keys.shape[2]``."""

    def test_oversized_normalizes_idx_to_buffer_length(self, mx, mlx_lm_classes):
        from omlx.scheduler import Scheduler

        # Build a stand-in oversized state. _normalize_rotating_state is a
        # method on Scheduler; calling it directly avoids spinning up the
        # full engine. The MagicMock layer_cache only needs _temporal_order.
        max_size = 128
        oversize = max_size + 16
        keys = mx.arange(oversize, dtype=mx.float32).reshape(1, 1, oversize, 1)
        values = mx.arange(oversize, dtype=mx.float32).reshape(1, 1, oversize, 1)

        class _FakeLayerCache:
            keep = 0
            max_size = 128
            offset = oversize
            _idx = oversize

            @staticmethod
            def _temporal_order(v):
                # Identity: pretend the buffer is already in temporal order.
                return v

        state = (keys, values)
        meta_state = (0, max_size, oversize, oversize)

        result_state, result_meta = Scheduler._normalize_rotating_snapshot_state(
            None, _FakeLayerCache(), state, meta_state
        )

        normalized_keys, _ = result_state
        # Trimmed to max_size.
        assert normalized_keys.shape[2] == max_size
        # _idx (4th field) equals the normalized buffer length.
        assert int(result_meta[3]) == max_size
