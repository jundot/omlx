# SPDX-License-Identifier: Apache-2.0
"""Tests for the N-tuple state interface on CacheTypeHandler.

The legacy interface in `extract_state` / `reconstruct_cache` modeled
state as a 2-tuple `(keys, values)` dict. omlx core had hard-coded
`state[0], state[1]` unpacking sprinkled across `prefix_cache.py`,
`paged_ssd_cache.py`, and `boundary_snapshot_store.py`, which silently
dropped the third+ element of N-tuple state caches like DeepSeek V4's
`PoolingCache` (`(buf_kv, buf_gate, pooled)`).

This test module pins the new handler-driven interface introduced in
Commit 1 of the cache architecture refactor: per-element axis metadata,
generic serialize/deserialize, and seq-len recovery from a raw state
tuple. Subsequent commits wire omlx core to use this interface; this
test establishes the contract those changes must keep stable.
"""

from __future__ import annotations


class TestCacheStateAxisInfoDefault:
    """Default axis_info matches the legacy 2-tuple (keys, values) contract."""

    def test_default_axis_info_two_elements(self):
        from omlx.cache.type_handlers import KVCacheHandler

        info = KVCacheHandler().get_state_axis_info()
        assert len(info) == 2
        assert info[0].name == "keys"
        assert info[1].name == "values"
        assert info[0].sequence_axis == 2
        assert info[1].sequence_axis == 2
        assert info[0].sliceable is True
        assert info[1].sliceable is True

    def test_rotating_axis_info_marks_non_sliceable(self):
        """RotatingKVCache uses circular buffer, must not be per-block sliced."""
        from omlx.cache.type_handlers import RotatingKVCacheHandler

        info = RotatingKVCacheHandler().get_state_axis_info()
        assert len(info) == 2
        assert info[0].sliceable is False
        assert info[1].sliceable is False
        # Sequence axis is still axis 2 (the circular buffer dim) even
        # though slicing along it is unsafe.
        assert info[0].sequence_axis == 2

    def test_arrays_cache_marked_variable_length(self):
        from omlx.cache.type_handlers import ArraysCacheHandler

        h = ArraysCacheHandler()
        assert h.is_variable_length_state() is True
        # Variable-length caches return empty axis info — omlx core
        # consults the `is_variable_length_state` flag instead.
        assert h.get_state_axis_info() == ()

    def test_cache_list_marked_composite(self):
        from omlx.cache.type_handlers import CacheListHandler

        h = CacheListHandler()
        assert h.is_composite_cache() is True
        assert h.get_state_axis_info() == ()


class TestSerializeStatePassthrough:
    """Default serialize_state passes through cache_obj.state as a tuple."""

    def test_kvcache_state_serialized_as_2tuple(self):
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from omlx.cache.type_handlers import KVCacheHandler

        cache = KVCache()
        cache.update_and_fetch(mx.zeros((1, 4, 8, 16)), mx.zeros((1, 4, 8, 16)))
        elements = KVCacheHandler().serialize_state(cache)
        assert isinstance(elements, tuple)
        assert len(elements) == 2

    def test_serialize_state_handles_missing_state_attr(self):
        from omlx.cache.type_handlers import KVCacheHandler

        class _Empty:
            pass

        elements = KVCacheHandler().serialize_state(_Empty())
        assert elements == ()


class TestDeserializeStateLegacyContract:
    """Default deserialize_state maps tuple elements to legacy keys/values dict."""

    def test_kvcache_round_trip_via_new_interface(self):
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from omlx.cache.type_handlers import KVCacheHandler

        original = KVCache()
        original.update_and_fetch(
            mx.arange(1 * 4 * 8 * 16, dtype=mx.float32).reshape(1, 4, 8, 16),
            mx.zeros((1, 4, 8, 16)),
        )
        h = KVCacheHandler()
        elements = h.serialize_state(original)
        restored = h.deserialize_state(elements, meta_state=original.meta_state)
        assert restored is not None
        # Compare trimmed state tuples (KVCache.state returns sliced view
        # without internal padding chunks).
        orig_keys, orig_values = original.state
        rest_keys, rest_values = restored.state
        assert orig_keys.shape == rest_keys.shape
        assert mx.max(mx.abs(rest_keys - orig_keys)).item() == 0.0
        assert mx.max(mx.abs(rest_values - orig_values)).item() == 0.0


class TestSeqLenFromTuple:
    """get_state_seq_len_from_tuple recovers length from first sliceable elem."""

    def test_kvcache_seq_len_from_tuple(self):
        import mlx.core as mx

        from omlx.cache.type_handlers import KVCacheHandler

        keys = mx.zeros((1, 4, 13, 16))  # seq_len = 13 on axis 2
        values = mx.zeros((1, 4, 13, 16))
        seq_len = KVCacheHandler().get_state_seq_len_from_tuple((keys, values))
        assert seq_len == 13

    def test_rotating_returns_full_length_even_when_non_sliceable(self):
        """Non-sliceable elements still report seq length on the seq axis;
        the *sliceable* flag controls per-block slicing, not length lookup.
        Default impl skips non-sliceable, so RotatingKVCache reports 0
        until a handler explicitly overrides this method."""
        import mlx.core as mx

        from omlx.cache.type_handlers import RotatingKVCacheHandler

        keys = mx.zeros((1, 4, 128, 16))
        values = mx.zeros((1, 4, 128, 16))
        # Default impl walks for first sliceable element. Rotating has no
        # sliceable elements → returns 0. This is the expected contract.
        assert (
            RotatingKVCacheHandler().get_state_seq_len_from_tuple((keys, values)) == 0
        )

    def test_seq_len_returns_zero_for_empty_tuple(self):
        from omlx.cache.type_handlers import KVCacheHandler

        assert KVCacheHandler().get_state_seq_len_from_tuple(()) == 0

    def test_seq_len_returns_zero_for_none_element(self):
        from omlx.cache.type_handlers import KVCacheHandler

        assert KVCacheHandler().get_state_seq_len_from_tuple((None, None)) == 0
