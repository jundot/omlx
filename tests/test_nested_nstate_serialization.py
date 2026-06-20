# SPDX-License-Identifier: Apache-2.0
"""Round-trip tests for nstate elements that are themselves composite.

DeepSeek-V4-Flash layers are ``CacheList(RotatingKVCache, PoolingCache,
PoolingCache)``. On the store path a layer can reach the paged-SSD serializer
as an ``__nstate__`` marker whose *elements* are not flat ``mx.array`` values
but composite sub-states:

  - a 2-tuple of arrays ``(keys, values)`` (the rotating sub-state), and
  - a nested ``('__nstate__', class_name, [None, None, pooled])`` marker (the
    pooling sub-state, whose first two elements are ``None``).

The pre-fix ``_store_nstate_elements`` assumed every element was an
``mx.array`` and stored it directly, so ``_extract_tensor_bytes`` raised
``'tuple' object has no attribute 'dtype'``. That failed the whole block save
(and, via the caller's break-on-failure, every later block), which bled the
prefix-cache hit rate down over a multi-turn conversation.

These tests pin the contract: composite nstate elements round-trip
byte-identically through ``save_block`` -> ``load_block``, including ``None``
sub-elements and zero-*size* arrays (MLA stores an empty values tensor).
"""

from __future__ import annotations

import time

import mlx.core as mx

from omlx.cache.paged_ssd_cache import PagedSSDCacheManager


def _make_manager(tmp_path):
    return PagedSSDCacheManager(
        cache_dir=tmp_path / "nested_nstate",
        max_size_bytes=100 * 1024**2,
    )


def _wait_for_file(manager, block_hash):
    for _ in range(100):
        if manager._get_file_path(block_hash).exists():
            return True
        time.sleep(0.05)
    return False


def _eq(a, b):
    return mx.max(mx.abs(a - b)).item() == 0.0


class TestNestedNStateElements:
    """An nstate element may itself be a composite sub-state."""

    def test_element_is_tuple_of_arrays_round_trips(self, tmp_path):
        """A layer nstate whose element is a 2-tuple ``(keys, values)`` of
        arrays survives the round-trip with both arrays byte-identical."""
        manager = _make_manager(tmp_path)
        block_hash = b"nested_tuple_elem___"

        keys = mx.arange(1 * 1 * 16 * 8, dtype=mx.float32).reshape(1, 1, 16, 8)
        values = (mx.arange(1 * 1 * 16 * 8, dtype=mx.float32) + 7.0).reshape(1, 1, 16, 8)
        mx.eval(keys, values)

        layer_marker = ("__nstate__", "DeepSeekV4Composite", [(keys, values)])
        manager.save_block(block_hash, [layer_marker], token_count=16)
        assert _wait_for_file(manager, block_hash)

        loaded = manager.load_block(block_hash)
        assert loaded is not None
        assert len(loaded) == 1
        marker = loaded[0]
        assert marker[0] == "__nstate__"
        elements = marker[2]
        assert len(elements) == 1
        elem = elements[0]
        assert isinstance(elem, (tuple, list)) and len(elem) == 2
        assert _eq(elem[0], keys)
        assert _eq(elem[1], values)

        manager.close()

    def test_element_is_nested_nstate_marker_round_trips(self, tmp_path):
        """The real DeepSeek shape: an element that is a nested
        ``('__nstate__', class, [None, None, pooled])`` marker."""
        manager = _make_manager(tmp_path)
        block_hash = b"nested_marker_elem__"

        keys = mx.arange(1 * 1 * 16 * 8, dtype=mx.float32).reshape(1, 1, 16, 8)
        values = mx.zeros((1, 1, 16, 8))
        pooled = mx.arange(1 * 4 * 8, dtype=mx.float32).reshape(1, 4, 8) * 3.0
        mx.eval(keys, values, pooled)

        layer_marker = (
            "__nstate__",
            "DeepSeekV4Composite",
            [
                (keys, values),
                ("__nstate__", "PoolingCache", [None, None, pooled]),
            ],
        )
        manager.save_block(block_hash, [layer_marker], token_count=16)
        assert _wait_for_file(manager, block_hash)

        loaded = manager.load_block(block_hash)
        assert loaded is not None
        marker = loaded[0]
        assert marker[0] == "__nstate__"
        elements = marker[2]
        assert len(elements) == 2

        # element 0: tuple of arrays
        assert _eq(elements[0][0], keys)
        assert _eq(elements[0][1], values)

        # element 1: nested __nstate__ marker, None positions preserved, pooled byte-equal
        nested = elements[1]
        assert isinstance(nested, tuple)
        assert nested[0] == "__nstate__"
        sub_elems = nested[2]
        assert len(sub_elems) == 3
        assert sub_elems[0] is None
        assert sub_elems[1] is None
        assert _eq(sub_elems[2], pooled)

        manager.close()

    def test_length2_nested_nstate_marker_stays_a_marker(self, tmp_path):
        """A length-2, EXPLICITLY-marked nested ``__nstate__`` element must
        round-trip as an ``__nstate__`` marker (class_name intact), not get
        unwrapped to a bare 2-tuple. Length-2 is the heuristic-collision case
        the length-3 tests miss; unwrapping it loses the marker and breaks
        callers that index ``elem[2]``."""
        manager = _make_manager(tmp_path)
        block_hash = b"len2_nested_marker__"

        c = (mx.arange(1 * 4 * 8, dtype=mx.float32) * 5.0).reshape(1, 4, 8)
        mx.eval(c)

        layer_marker = (
            "__nstate__",
            "DeepSeekV4Composite",
            [("__nstate__", "PoolingCache", [None, c])],
        )
        manager.save_block(block_hash, [layer_marker], token_count=8)
        assert _wait_for_file(manager, block_hash)

        loaded = manager.load_block(block_hash)
        assert loaded is not None
        nested = loaded[0][2][0]
        assert isinstance(nested, tuple)
        assert nested[0] == "__nstate__", f"expected marker, got {nested!r}"
        assert nested[1] == "PoolingCache"  # class_name preserved
        assert len(nested[2]) == 2
        assert nested[2][0] is None
        assert _eq(nested[2][1], c)

        manager.close()

    def test_zero_size_array_element_round_trips(self, tmp_path):
        """MLA stores an empty values tensor (a 0-length trailing axis).
        Shape must be preserved across the round-trip."""
        manager = _make_manager(tmp_path)
        block_hash = b"zero_size_elem______"

        keys = mx.arange(1 * 1 * 8 * 16, dtype=mx.float32).reshape(1, 1, 8, 16)
        values = mx.zeros((1, 1, 8, 0))  # zero-size last axis, like MLA
        mx.eval(keys, values)

        layer_marker = ("__nstate__", "DeepSeekV4Composite", [(keys, values)])
        manager.save_block(block_hash, [layer_marker], token_count=8)
        assert _wait_for_file(manager, block_hash)

        loaded = manager.load_block(block_hash)
        assert loaded is not None
        elem = loaded[0][2][0]
        assert _eq(elem[0], keys)
        assert tuple(elem[1].shape) == (1, 1, 8, 0)

        manager.close()

    def test_flat_three_tuple_still_works(self, tmp_path):
        """Regression guard: a flat N-tuple of plain arrays (the existing
        PoolingCache case) must keep working unchanged."""
        manager = _make_manager(tmp_path)
        block_hash = b"flat_three_tuple____"

        e0 = mx.arange(1 * 4 * 8, dtype=mx.float32).reshape(1, 4, 8)
        e1 = e0 * 2.0
        e2 = e0 * 3.0
        mx.eval(e0, e1, e2)

        layer_marker = ("__nstate__", "PoolingCache", [e0, e1, e2])
        manager.save_block(block_hash, [layer_marker], token_count=16)
        assert _wait_for_file(manager, block_hash)

        loaded = manager.load_block(block_hash)
        assert loaded is not None
        elements = loaded[0][2]
        assert len(elements) == 3
        assert _eq(elements[0], e0)
        assert _eq(elements[1], e1)
        assert _eq(elements[2], e2)

        manager.close()

    def test_format_version_unchanged(self):
        """The fix is additive; existing v2/v3 caches stay readable and the
        write version is not bumped (no mass cache invalidation)."""
        from omlx.cache.paged_ssd_cache import (
            _CACHE_FORMAT_VERSION,
            _READABLE_CACHE_FORMAT_VERSIONS,
        )

        assert _CACHE_FORMAT_VERSION == "3"
        assert {"2", "3"} <= set(_READABLE_CACHE_FORMAT_VERSIONS)
