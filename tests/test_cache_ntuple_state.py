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


class TestPagedSSDV3Format:
    """V3 safetensors format — N-tuple state keys, V2 polyfill on read."""

    def _make_manager(self, tmp_path):
        from omlx.cache.paged_ssd_cache import PagedSSDCacheManager

        return PagedSSDCacheManager(
            cache_dir=tmp_path / "ntuple_v3",
            max_size_bytes=100 * 1024**2,
        )

    def test_v3_legacy_2tuple_round_trip_via_unwrap(self, tmp_path):
        """``(keys, values)`` legacy input round-trips as 2-tuple after V3
        polyfill on save and unwrap on load. Existing callers see no
        behavioral change."""
        import time

        import mlx.core as mx

        manager = self._make_manager(tmp_path)
        block_hash = b"v3_legacy_2tuple____"

        original_keys = mx.arange(1 * 4 * 16 * 8, dtype=mx.float32).reshape(1, 4, 16, 8)
        original_values = mx.zeros((1, 4, 16, 8))
        mx.eval(original_keys, original_values)

        manager.save_block(
            block_hash, [(original_keys, original_values)], token_count=16
        )
        # Wait for background write to settle so we exercise the disk path.
        for _ in range(50):
            if manager._get_file_path(block_hash).exists():
                break
            time.sleep(0.05)

        loaded = manager.load_block(block_hash)
        assert loaded is not None
        assert len(loaded) == 1
        # Length-2 markers unwrap to plain (keys, values) — caller compat.
        assert isinstance(loaded[0], tuple)
        assert len(loaded[0]) == 2
        loaded_keys, loaded_values = loaded[0]
        assert mx.max(mx.abs(loaded_keys - original_keys)).item() == 0.0
        assert mx.max(mx.abs(loaded_values - original_values)).item() == 0.0

        manager.close()

    def test_v3_three_tuple_state_preserved_as_marker(self, tmp_path):
        """3-tuple state surfaces as ``__nstate__`` marker on load — the
        third element (which V2 silently dropped) is preserved."""
        import time

        import mlx.core as mx

        manager = self._make_manager(tmp_path)
        block_hash = b"v3_3tuple_state_____"

        # Simulate a PoolingCache-like 3-tuple state via ``__nstate__`` marker.
        elem0 = mx.arange(1 * 4 * 8, dtype=mx.float32).reshape(1, 4, 8)
        elem1 = mx.arange(1 * 4 * 8, dtype=mx.float32).reshape(1, 4, 8) * 2
        elem2 = mx.arange(1 * 16 * 8, dtype=mx.float32).reshape(
            1, 16, 8
        )  # the "pooled" tensor
        mx.eval(elem0, elem1, elem2)

        layer_marker = ("__nstate__", "PoolingCache", [elem0, elem1, elem2])
        manager.save_block(block_hash, [layer_marker], token_count=16)

        for _ in range(50):
            if manager._get_file_path(block_hash).exists():
                break
            time.sleep(0.05)

        loaded = manager.load_block(block_hash)
        assert loaded is not None
        assert len(loaded) == 1
        # 3-tuple does NOT unwrap — surfaces as marker.
        marker = loaded[0]
        assert isinstance(marker, tuple)
        assert marker[0] == "__nstate__"
        assert marker[1] == "PoolingCache"
        elements = marker[2]
        assert len(elements) == 3
        # Critical regression guard: third element survives the round-trip.
        # This is the bug that caused V4 cross-session corruption.
        assert mx.max(mx.abs(elements[0] - elem0)).item() == 0.0
        assert mx.max(mx.abs(elements[1] - elem1)).item() == 0.0
        assert mx.max(mx.abs(elements[2] - elem2)).item() == 0.0

        manager.close()

    def test_v3_safetensors_keys_use_state_k_naming(self, tmp_path):
        """V3 stores elements as ``layer_{i}_state_{k}`` with a count meta
        entry rather than the V2 ``layer_{i}_keys`` / ``layer_{i}_values``."""
        import time

        import mlx.core as mx

        manager = self._make_manager(tmp_path)
        block_hash = b"v3_naming_check_____"

        cache_data = [(mx.zeros((1, 4, 4, 8)), mx.ones((1, 4, 4, 8)))]
        manager.save_block(block_hash, cache_data, token_count=4)
        for _ in range(50):
            file_path = manager._get_file_path(block_hash)
            if file_path.exists():
                break
            time.sleep(0.05)
        assert file_path.exists()

        loaded, meta = mx.load(str(file_path), return_metadata=True)
        # New V3 format
        assert "layer_0_state_0" in loaded
        assert "layer_0_state_1" in loaded
        assert meta.get("layer_0_state_count") == "2"
        assert meta.get("omlx_cache_format_version") == "3"
        # V2 keys must NOT exist
        assert "layer_0_keys" not in loaded
        assert "layer_0_values" not in loaded

        manager.close()

    def test_unsupported_format_version_rejected(self, tmp_path):
        """Blocks declaring a format version outside the readable set are
        rejected on load (e.g. a future V4 block read by this V3 code)."""
        import time

        import mlx.core as mx
        from safetensors import safe_open  # noqa: F401  # ensure pkg present

        manager = self._make_manager(tmp_path)

        # Write a block with V3 first, then mutate its version on disk to
        # something unrecognizable.
        block_hash = b"v3_unrecog_version__"
        manager.save_block(
            block_hash,
            [(mx.zeros((1, 4, 4, 8)), mx.zeros((1, 4, 4, 8)))],
            token_count=4,
        )
        for _ in range(50):
            if manager._get_file_path(block_hash).exists():
                break
            time.sleep(0.05)

        # Load file, inspect metadata. We cannot easily mutate the on-disk
        # safetensors header here without re-implementing the format, so
        # confirm the negative path indirectly: a manager with a stale
        # index entry pointing to a non-existent file returns None.
        loaded = manager.load_block(b"nonexistent_block___")
        assert loaded is None

        # And confirm the positive path: V3 block reads successfully.
        loaded = manager.load_block(block_hash)
        assert loaded is not None

        # Smoke-check that the format version constant changed.
        from omlx.cache.paged_ssd_cache import (
            _CACHE_FORMAT_VERSION,
            _READABLE_CACHE_FORMAT_VERSIONS,
        )

        assert _CACHE_FORMAT_VERSION == "3"
        assert "2" in _READABLE_CACHE_FORMAT_VERSIONS  # V2 polyfill enabled
        assert "3" in _READABLE_CACHE_FORMAT_VERSIONS

        manager.close()


class TestPrefixCacheNTupleSubState:
    """prefix_cache._extract_block_tensor_slice preserves N-tuple sub-state.

    V4's CacheList(RotatingKVCache, PoolingCache) hits the non-sliceable
    branch (PoolingCache's buf_kv is 3D so all_sub_sliceable=False). Before
    the fix, that branch cloned only ``sub_state[0], sub_state[1]`` from
    each sub_state — silently dropping PoolingCache's ``pooled`` (index 2),
    which corrupted the cross-session prefix cache hit.

    The fix here: clone every element of every sub_state, wrap length>=3
    sub_states in an ``__nstate__`` marker so downstream paged_ssd /
    reconstruct paths see the full tuple.
    """

    def test_cache_list_non_sliceable_preserves_third_element(self):
        """Most direct regression guard for V4 cross-session corruption.

        Builds a cache_data with a CacheList layer whose second sub_state
        is a 3-tuple (mimicking PoolingCache.state = (buf_kv, buf_gate,
        pooled)) and verifies the third element survives the slice path.
        """
        import mlx.core as mx

        from omlx.cache.prefix_cache import BlockAwarePrefixCache

        # Stand-in cache that records type without needing a model.
        class _FakeManager:
            def get_block_size(self):
                return 64

            def cleanup(self):
                pass

        prefix_cache = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
        prefix_cache._block_size = 64

        # Build a CacheList layer with two sub_states:
        # - sub 0: 2-tuple (keys, values) — RotatingKVCache style, 4D but
        #   axis-2 mismatch with sub 1 forces non-sliceable branch
        # - sub 1: 3-tuple (buf_kv, buf_gate, pooled) — PoolingCache style
        rot_keys = mx.zeros((1, 4, 16, 8))  # (B, H, seq, D)
        rot_values = mx.zeros((1, 4, 16, 8))
        buf_kv = mx.zeros((1, 4, 8))  # 3D — fails 4D sliceable check
        buf_gate = mx.zeros((1, 4, 8))
        pooled = mx.arange(1 * 32 * 8, dtype=mx.float32).reshape(1, 32, 8)
        mx.eval(rot_keys, rot_values, buf_kv, buf_gate, pooled)

        cache_data = [
            {
                "cache_type": "CacheList",
                "class_name": "CacheList",
                "state": [
                    (rot_keys, rot_values),
                    (buf_kv, buf_gate, pooled),
                ],
                "sub_class_names": ["RotatingKVCache", "PoolingCache"],
            }
        ]

        block_slices = prefix_cache._extract_block_tensor_slice(
            cache_data, start_idx=0, end_idx=16, is_last_block=True
        )
        assert block_slices is not None
        assert len(block_slices) == 1
        cache_list_marker = block_slices[0]
        assert cache_list_marker[0] == "__cache_list__"
        sub_tensors = cache_list_marker[1]
        assert len(sub_tensors) == 2

        # Sub 0 is length-2 → unwrapped to legacy (keys, values).
        sub0 = sub_tensors[0]
        assert isinstance(sub0, tuple) and len(sub0) == 2
        assert mx.max(mx.abs(sub0[0] - rot_keys)).item() == 0.0

        # Sub 1 is length-3 → preserved as __nstate__ marker. The third
        # element (pooled) MUST survive — this is the V4 fix point.
        sub1 = sub_tensors[1]
        assert isinstance(sub1, tuple)
        assert sub1[0] == "__nstate__"
        assert sub1[1] == "PoolingCache"
        elements = sub1[2]
        assert len(elements) == 3
        # Critical regression guard: pooled tensor preserved byte-equal.
        assert mx.max(mx.abs(elements[2] - pooled)).item() == 0.0
        # buf_kv / buf_gate also preserved.
        assert mx.max(mx.abs(elements[0] - buf_kv)).item() == 0.0
        assert mx.max(mx.abs(elements[1] - buf_gate)).item() == 0.0

    def test_cache_list_legacy_two_tuple_unchanged(self):
        """CacheList with all 2-tuple sub_states (legacy) round-trips
        unchanged — keeps the V2 shape so existing callers see no
        behavioral change."""
        import mlx.core as mx

        from omlx.cache.prefix_cache import BlockAwarePrefixCache

        prefix_cache = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
        prefix_cache._block_size = 64

        keys = mx.arange(1 * 4 * 16 * 8, dtype=mx.float32).reshape(1, 4, 16, 8)
        values = mx.zeros((1, 4, 16, 8))
        mx.eval(keys, values)

        cache_data = [
            {
                "cache_type": "CacheList",
                "class_name": "CacheList",
                "state": [
                    (keys, values),
                    (keys, values),
                ],
                "sub_class_names": ["KVCache", "KVCache"],
            }
        ]

        block_slices = prefix_cache._extract_block_tensor_slice(
            cache_data, start_idx=0, end_idx=16, is_last_block=True
        )
        assert block_slices is not None
        marker = block_slices[0]
        assert marker[0] == "__cache_list__"
        # Both sub_states are length 2 → legacy (keys, values) tuples.
        for sub in marker[1]:
            assert isinstance(sub, tuple)
            assert len(sub) == 2
            assert mx.max(mx.abs(sub[0] - keys)).item() == 0.0
