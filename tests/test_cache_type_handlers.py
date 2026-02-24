# SPDX-License-Identifier: Apache-2.0
"""
Tests for cache type handlers and registry.

This module tests the abstract and concrete handlers for various cache types
from mlx-lm, enabling type-aware cache operations like slicing and reconstruction.
"""

from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from omlx.cache.type_handlers import (
    ArraysCacheHandler,
    CacheListHandler,
    CacheStateInfo,
    CacheType,
    CacheTypeHandler,
    DefaultCacheHandler,
    KVCacheHandler,
    RotatingKVCacheHandler,
    SizedArraysCache,
)
from omlx.cache.type_registry import CacheTypeRegistry


class TestCacheTypeEnum:
    """Tests for CacheType enum."""

    def test_kvcache_value(self):
        """Test KVCache enum value."""
        assert CacheType.KVCACHE.value == "KVCache"

    def test_rotating_kvcache_value(self):
        """Test RotatingKVCache enum value."""
        assert CacheType.ROTATING_KVCACHE.value == "RotatingKVCache"

    def test_arrays_cache_value(self):
        """Test ArraysCache enum value."""
        assert CacheType.ARRAYS_CACHE.value == "ArraysCache"

    def test_batch_kvcache_value(self):
        """Test BatchKVCache enum value."""
        assert CacheType.BATCH_KVCACHE.value == "BatchKVCache"

    def test_all_values_unique(self):
        """Test all enum values are unique."""
        values = [ct.value for ct in CacheType]
        assert len(values) == len(set(values))


class TestCacheStateInfo:
    """Tests for CacheStateInfo dataclass."""

    def test_creation(self):
        """Test creating CacheStateInfo."""
        info = CacheStateInfo(
            cache_type="KVCache",
            state_keys=("keys", "values"),
            meta_state_keys=("offset",),
            supports_block_slicing=True,
        )

        assert info.cache_type == "KVCache"
        assert info.state_keys == ("keys", "values")
        assert info.meta_state_keys == ("offset",)
        assert info.supports_block_slicing is True
        assert info.is_full_state is False

    def test_with_full_state(self):
        """Test CacheStateInfo with is_full_state."""
        info = CacheStateInfo(
            cache_type="ArraysCache",
            state_keys=("states",),
            meta_state_keys=(),
            supports_block_slicing=False,
            is_full_state=True,
        )

        assert info.is_full_state is True


class TestKVCacheHandler:
    """Tests for KVCacheHandler."""

    @pytest.fixture
    def handler(self):
        """Create a KVCacheHandler."""
        return KVCacheHandler()

    def test_cache_type(self, handler):
        """Test cache_type property."""
        assert handler.cache_type == CacheType.KVCACHE

    def test_supports_block_slicing(self, handler):
        """Test supports_block_slicing property."""
        assert handler.supports_block_slicing is True

    def test_extract_state(self, handler):
        """Test extracting state from cache object."""
        mock_cache = MagicMock()
        mock_keys = MagicMock()
        mock_keys.shape = (1, 8, 64, 64)
        mock_values = MagicMock()
        mock_cache.state = (mock_keys, mock_values)
        mock_cache.offset = 64

        state = handler.extract_state(mock_cache)

        assert state["keys"] is mock_keys
        assert state["values"] is mock_values
        assert state["offset"] == 64
        assert state["cache_type"] == "KVCache"

    def test_get_seq_len(self, handler):
        """Test getting sequence length."""
        mock_keys = MagicMock()
        mock_keys.shape = (1, 8, 128, 64)

        state = {"keys": mock_keys, "offset": 100}

        seq_len = handler.get_seq_len(state)
        assert seq_len == 128

    def test_get_seq_len_from_offset(self, handler):
        """Test getting sequence length from offset when keys unavailable."""
        state = {"keys": None, "offset": 50}

        seq_len = handler.get_seq_len(state)
        assert seq_len == 50

    def test_get_state_info(self, handler):
        """Test getting state info."""
        info = handler.get_state_info()

        assert isinstance(info, CacheStateInfo)
        assert info.cache_type == "KVCache"
        assert info.supports_block_slicing is True
        assert info.state_keys == ("keys", "values")

    def test_default_state_keys(self, handler):
        """Test default state keys."""
        keys = handler._get_state_keys()
        assert keys == ("keys", "values")

    def test_default_meta_state_keys(self, handler):
        """Test default meta state keys."""
        keys = handler._get_meta_state_keys()
        assert keys == ("offset",)


class TestKVCacheHandlerWithMLX:
    """Tests for KVCacheHandler that require MLX."""

    @pytest.fixture
    def mx(self):
        """Import MLX or skip."""
        try:
            import mlx.core as mx
            return mx
        except ImportError:
            pytest.skip("MLX not available")

    @pytest.fixture
    def handler(self):
        """Create a KVCacheHandler."""
        return KVCacheHandler()

    def test_slice_state(self, handler, mx):
        """Test slicing state."""
        keys = mx.zeros((1, 8, 64, 64))
        values = mx.zeros((1, 8, 64, 64))
        state = {"keys": keys, "values": values}

        sliced = handler.slice_state(state, 16, 32)

        assert sliced is not None
        assert sliced["keys"].shape == (1, 8, 16, 64)
        assert sliced["values"].shape == (1, 8, 16, 64)

    def test_slice_state_none_keys(self, handler, mx):
        """Test slicing with None keys returns None."""
        state = {"keys": None, "values": mx.zeros((1, 8, 64, 64))}

        sliced = handler.slice_state(state, 0, 16)

        assert sliced is None

    def test_concatenate_states(self, handler, mx):
        """Test concatenating states."""
        states = [
            {"keys": mx.zeros((1, 8, 16, 64)), "values": mx.zeros((1, 8, 16, 64))},
            {"keys": mx.zeros((1, 8, 32, 64)), "values": mx.zeros((1, 8, 32, 64))},
        ]

        concatenated = handler.concatenate_states(states)

        assert concatenated["keys"].shape == (1, 8, 48, 64)
        assert concatenated["values"].shape == (1, 8, 48, 64)
        assert concatenated["offset"] == 48

    def test_concatenate_empty_states(self, handler):
        """Test concatenating empty states list."""
        result = handler.concatenate_states([])
        assert result == {}

    def test_reconstruct_cache(self, handler, mx):
        """Test reconstructing cache object."""
        state = {
            "keys": mx.zeros((1, 8, 64, 64)),
            "values": mx.zeros((1, 8, 64, 64)),
            "offset": 64,
        }

        cache = handler.reconstruct_cache(state)

        assert cache is not None
        assert cache.offset == 64

    def test_reconstruct_cache_with_meta_state(self, handler, mx):
        """Test reconstructing cache with meta_state."""
        state = {
            "keys": mx.zeros((1, 8, 32, 64)),
            "values": mx.zeros((1, 8, 32, 64)),
        }
        meta_state = (100,)  # offset = 100

        cache = handler.reconstruct_cache(state, meta_state)

        assert cache is not None
        assert cache.offset == 100


class TestRotatingKVCacheHandler:
    """Tests for RotatingKVCacheHandler."""

    @pytest.fixture
    def handler(self):
        """Create a RotatingKVCacheHandler."""
        return RotatingKVCacheHandler()

    def test_cache_type(self, handler):
        """Test cache_type property."""
        assert handler.cache_type == CacheType.ROTATING_KVCACHE

    def test_supports_block_slicing(self, handler):
        """Test supports_block_slicing is False."""
        assert handler.supports_block_slicing is False

    def test_extract_state(self, handler):
        """Test extracting state from RotatingKVCache."""
        mock_cache = MagicMock()
        mock_keys = MagicMock()
        mock_keys.shape = (1, 8, 256, 64)
        mock_values = MagicMock()
        mock_cache.state = (mock_keys, mock_values)
        mock_cache.offset = 1000
        mock_cache.max_size = 256
        mock_cache.keep = 4
        mock_cache._idx = 100
        mock_cache.meta_state = (4, 256, 1000, 100)

        state = handler.extract_state(mock_cache)

        assert state["keys"] is mock_keys
        assert state["values"] is mock_values
        assert state["offset"] == 1000
        assert state["max_size"] == 256
        assert state["keep"] == 4
        assert state["_idx"] == 100
        assert state["cache_type"] == "RotatingKVCache"

    def test_get_seq_len(self, handler):
        """Test getting sequence length (returns offset)."""
        state = {"offset": 500}
        seq_len = handler.get_seq_len(state)
        assert seq_len == 500

    def test_slice_state_returns_full(self, handler):
        """Test slice_state returns full state (not sliceable)."""
        mock_keys = MagicMock()
        mock_values = MagicMock()
        state = {
            "keys": mock_keys,
            "values": mock_values,
            "meta_state": (4, 256, 1000, 100),
            "max_size": 256,
            "offset": 1000,
            "keep": 4,
            "_idx": 100,
        }

        sliced = handler.slice_state(state, 0, 64)

        assert sliced["keys"] is mock_keys
        assert sliced["values"] is mock_values
        assert sliced["is_full_state"] is True

    def test_concatenate_states_uses_latest(self, handler):
        """Test concatenate uses most recent state."""
        states = [
            {"keys": MagicMock(name="keys1"), "values": MagicMock()},
            {"keys": MagicMock(name="keys2"), "values": MagicMock()},
        ]

        result = handler.concatenate_states(states)

        # Should use last state
        assert result["keys"] is states[-1]["keys"]
        assert result["is_full_state"] is True

    def test_meta_state_keys(self, handler):
        """Test meta_state keys for RotatingKVCache."""
        keys = handler._get_meta_state_keys()
        assert keys == ("keep", "max_size", "offset", "_idx")


class TestRotatingKVCacheHandlerWithMLX:
    """Tests for RotatingKVCacheHandler requiring MLX."""

    @pytest.fixture
    def mx(self):
        """Import MLX or skip."""
        try:
            import mlx.core as mx
            return mx
        except ImportError:
            pytest.skip("MLX not available")

    @pytest.fixture
    def handler(self):
        """Create handler."""
        return RotatingKVCacheHandler()

    def test_reconstruct_cache(self, handler, mx):
        """Test reconstructing RotatingKVCache."""
        state = {
            "keys": mx.zeros((1, 8, 256, 64)),
            "values": mx.zeros((1, 8, 256, 64)),
            "max_size": 256,
            "keep": 4,
            "offset": 500,
            "_idx": 100,
        }
        meta_state = (4, 256, 500, 100)

        cache = handler.reconstruct_cache(state, meta_state)

        assert cache is not None
        assert cache.max_size == 256
        assert cache.keep == 4
        assert cache.offset == 500
        assert cache._idx == 100


class TestArraysCacheHandler:
    """Tests for ArraysCacheHandler."""

    @pytest.fixture
    def handler(self):
        """Create an ArraysCacheHandler."""
        return ArraysCacheHandler()

    def test_cache_type(self, handler):
        """Test cache_type property."""
        assert handler.cache_type == CacheType.ARRAYS_CACHE

    def test_supports_block_slicing(self, handler):
        """Test supports_block_slicing is False."""
        assert handler.supports_block_slicing is False

    def test_extract_state(self, handler):
        """Test extracting state from ArraysCache."""
        mock_cache = MagicMock()
        mock_states = [MagicMock(), MagicMock(), MagicMock()]
        mock_cache.state = mock_states
        mock_cache.cache = mock_states

        state = handler.extract_state(mock_cache)

        assert len(state["states"]) == 3
        assert state["is_full_state"] is True

    def test_get_seq_len(self, handler):
        """Test getting sequence length."""
        state = {"token_count": 256}
        assert handler.get_seq_len(state) == 256

    def test_slice_state_returns_full(self, handler):
        """Test slice_state returns full state."""
        state = {"states": [MagicMock()]}

        sliced = handler.slice_state(state, 0, 64)

        assert sliced["is_full_state"] is True

    def test_concatenate_states(self, handler):
        """Test concatenate uses latest state."""
        states = [
            {"states": [1, 2]},
            {"states": [3, 4, 5]},
        ]

        result = handler.concatenate_states(states)

        assert result["states"] == [3, 4, 5]

    def test_state_keys(self, handler):
        """Test state keys."""
        assert handler._get_state_keys() == ("states",)


class TestDefaultCacheHandler:
    """Tests for DefaultCacheHandler (fallback)."""

    @pytest.fixture
    def handler(self):
        """Create a DefaultCacheHandler."""
        return DefaultCacheHandler()

    def test_cache_type(self, handler):
        """Test cache_type property (defaults to KVCACHE)."""
        assert handler.cache_type == CacheType.KVCACHE

    def test_extract_state_kvcache_like(self, handler):
        """Test extracting state from KVCache-like object."""
        mock_cache = MagicMock()
        mock_keys = MagicMock()
        mock_values = MagicMock()
        mock_cache.state = (mock_keys, mock_values)
        mock_cache.offset = 64

        state = handler.extract_state(mock_cache)

        assert state["keys"] is mock_keys
        assert state["values"] is mock_values
        assert state["cache_type"] == "Unknown"

    def test_extract_state_fails_gracefully(self, handler):
        """Test extract_state handles unknown objects."""
        mock_cache = MagicMock()
        mock_cache.state = "not a tuple"

        state = handler.extract_state(mock_cache)

        assert state["cache_type"] == "Unknown"


class TestSizedArraysCache:
    """Tests for SizedArraysCache wrapper."""

    def test_creation(self):
        """Test creating SizedArraysCache."""
        mock_inner = MagicMock()
        wrapper = SizedArraysCache(mock_inner, token_count=100)

        assert wrapper._inner is mock_inner
        assert wrapper._token_count == 100

    def test_size_returns_token_count(self):
        """Test size() returns token_count instead of 0."""
        mock_inner = MagicMock()
        mock_inner.size.return_value = 0

        wrapper = SizedArraysCache(mock_inner, token_count=256)

        assert wrapper.size() == 256

    def test_empty_delegation(self):
        """Test empty() delegates to inner."""
        mock_inner = MagicMock()
        mock_inner.empty.return_value = False

        wrapper = SizedArraysCache(mock_inner, token_count=100)

        assert wrapper.empty() is False
        mock_inner.empty.assert_called_once()

    def test_state_property(self):
        """Test state property delegation."""
        mock_inner = MagicMock()
        mock_state = (MagicMock(), MagicMock())
        mock_inner.state = mock_state

        wrapper = SizedArraysCache(mock_inner, token_count=100)

        assert wrapper.state is mock_state

    def test_state_setter(self):
        """Test state setter delegation."""
        mock_inner = MagicMock()
        wrapper = SizedArraysCache(mock_inner, token_count=100)

        new_state = (MagicMock(), MagicMock())
        wrapper.state = new_state

        assert mock_inner.state == new_state

    def test_cache_property(self):
        """Test cache property delegation."""
        mock_inner = MagicMock()
        mock_cache = [MagicMock()]
        mock_inner.cache = mock_cache

        wrapper = SizedArraysCache(mock_inner, token_count=100)

        assert wrapper.cache is mock_cache

    def test_getitem(self):
        """Test __getitem__ delegation."""
        mock_inner = MagicMock()
        mock_value = MagicMock()
        mock_inner.__getitem__.return_value = mock_value

        wrapper = SizedArraysCache(mock_inner, token_count=100)

        result = wrapper[0]

        assert result is mock_value
        mock_inner.__getitem__.assert_called_once_with(0)

    def test_setitem(self):
        """Test __setitem__ delegation."""
        mock_inner = MagicMock()
        wrapper = SizedArraysCache(mock_inner, token_count=100)

        wrapper[0] = "value"

        mock_inner.__setitem__.assert_called_once_with(0, "value")

    def test_len(self):
        """Test __len__ returns inner cache length."""
        mock_inner = MagicMock()
        mock_inner.cache = [1, 2, 3]

        wrapper = SizedArraysCache(mock_inner, token_count=100)

        assert len(wrapper) == 3

    def test_getattr_delegation(self):
        """Test unknown attributes delegate to inner."""
        mock_inner = MagicMock()
        mock_inner.custom_attr = "custom_value"

        wrapper = SizedArraysCache(mock_inner, token_count=100)

        assert wrapper.custom_attr == "custom_value"

    def test_setattr_delegation(self):
        """Test setting unknown attributes on inner."""
        mock_inner = MagicMock()
        wrapper = SizedArraysCache(mock_inner, token_count=100)

        wrapper.custom_attr = "custom_value"

        assert mock_inner.custom_attr == "custom_value"

    def test_prepare_delegation(self):
        """Test prepare() delegates to inner."""
        mock_inner = MagicMock()
        wrapper = SizedArraysCache(mock_inner, token_count=100)

        wrapper.prepare(arg1="val1")

        mock_inner.prepare.assert_called_once_with(arg1="val1")

    def test_finalize_delegation(self):
        """Test finalize() delegates to inner."""
        mock_inner = MagicMock()
        wrapper = SizedArraysCache(mock_inner, token_count=100)

        wrapper.finalize()

        mock_inner.finalize.assert_called_once()

    def test_advance_delegation(self):
        """Test advance() delegates to inner."""
        mock_inner = MagicMock()
        wrapper = SizedArraysCache(mock_inner, token_count=100)

        wrapper.advance(10)

        mock_inner.advance.assert_called_once_with(10)

    def test_extract_preserves_token_count(self):
        """Test extract() preserves token_count."""
        mock_inner = MagicMock()
        mock_extracted = MagicMock()
        mock_inner.extract.return_value = mock_extracted

        wrapper = SizedArraysCache(mock_inner, token_count=100)
        result = wrapper.extract(0)

        assert isinstance(result, SizedArraysCache)
        assert result._token_count == 100

    def test_extend_unwraps_other(self):
        """Test extend() unwraps SizedArraysCache argument."""
        mock_inner1 = MagicMock()
        mock_inner2 = MagicMock()

        wrapper1 = SizedArraysCache(mock_inner1, token_count=100)
        wrapper2 = SizedArraysCache(mock_inner2, token_count=50)

        wrapper1.extend(wrapper2)

        mock_inner1.extend.assert_called_once_with(mock_inner2)


class TestCacheTypeRegistry:
    """Tests for CacheTypeRegistry."""

    def test_get_handler(self):
        """Test getting handler by CacheType."""
        handler = CacheTypeRegistry.get_handler(CacheType.KVCACHE)
        assert isinstance(handler, KVCacheHandler)

        handler = CacheTypeRegistry.get_handler(CacheType.ROTATING_KVCACHE)
        assert isinstance(handler, RotatingKVCacheHandler)

    def test_get_handler_by_class_name(self):
        """Test getting handler by class name string."""
        handler = CacheTypeRegistry.get_handler_by_class_name("KVCache")
        assert isinstance(handler, KVCacheHandler)

        handler = CacheTypeRegistry.get_handler_by_class_name("RotatingKVCache")
        assert isinstance(handler, RotatingKVCacheHandler)

    def test_get_handler_by_class_name_unknown(self):
        """Test getting handler for unknown class name."""
        handler = CacheTypeRegistry.get_handler_by_class_name("UnknownCache")
        # Should return default handler
        assert isinstance(handler, DefaultCacheHandler)

    def test_get_handler_for_sized_arrays_cache(self):
        """Test getting handler for SizedArraysCache wrapper."""
        handler = CacheTypeRegistry.get_handler_by_class_name("SizedArraysCache")
        assert isinstance(handler, ArraysCacheHandler)

    def test_detect_cache_type(self):
        """Test detecting cache type from object."""
        # KVCache-like
        mock_kv = MagicMock()
        mock_kv.__class__.__name__ = "KVCache"
        assert CacheTypeRegistry.detect_cache_type(mock_kv) == CacheType.KVCACHE

        # RotatingKVCache-like
        mock_rotating = MagicMock()
        mock_rotating.__class__.__name__ = "RotatingKVCache"
        assert CacheTypeRegistry.detect_cache_type(mock_rotating) == CacheType.ROTATING_KVCACHE

    def test_detect_cache_type_by_attributes(self):
        """Test detecting cache type by attributes when class name unknown."""
        # Object with max_size and _idx -> RotatingKVCache
        mock_obj = MagicMock()
        mock_obj.__class__.__name__ = "CustomRotating"
        mock_obj.max_size = 256
        mock_obj._idx = 0

        cache_type = CacheTypeRegistry.detect_cache_type(mock_obj)
        assert cache_type == CacheType.ROTATING_KVCACHE

    def test_detect_cache_type_sized_arrays_wrapper(self):
        """Test detecting cache type from SizedArraysCache wrapper."""
        mock_inner = MagicMock()
        mock_inner.__class__.__name__ = "ArraysCache"
        wrapper = SizedArraysCache(mock_inner, token_count=100)

        cache_type = CacheTypeRegistry.detect_cache_type(wrapper)
        assert cache_type == CacheType.ARRAYS_CACHE

    def test_get_handler_for_object(self):
        """Test getting handler for cache object."""
        mock_kv = MagicMock()
        mock_kv.__class__.__name__ = "KVCache"

        handler = CacheTypeRegistry.get_handler_for_object(mock_kv)
        assert isinstance(handler, KVCacheHandler)

    def test_is_sliceable(self):
        """Test checking if cache is sliceable."""
        mock_kv = MagicMock()
        mock_kv.__class__.__name__ = "KVCache"
        assert CacheTypeRegistry.is_sliceable(mock_kv) is True

        mock_arrays = MagicMock()
        mock_arrays.__class__.__name__ = "ArraysCache"
        assert CacheTypeRegistry.is_sliceable(mock_arrays) is False

    def test_get_class_name_for_type(self):
        """Test getting class name for cache type."""
        name = CacheTypeRegistry.get_class_name_for_type(CacheType.KVCACHE)
        assert name == "KVCache"

        name = CacheTypeRegistry.get_class_name_for_type(CacheType.ARRAYS_CACHE)
        assert name == "ArraysCache"

    def test_list_registered_types(self):
        """Test listing registered cache types."""
        types = CacheTypeRegistry.list_registered_types()
        assert CacheType.KVCACHE in types
        assert CacheType.ROTATING_KVCACHE in types
        assert CacheType.ARRAYS_CACHE in types

    def test_list_known_class_names(self):
        """Test listing known class names."""
        names = CacheTypeRegistry.list_known_class_names()
        assert "KVCache" in names
        assert "RotatingKVCache" in names
        assert "ArraysCache" in names

    def test_register_handler(self):
        """Test registering a custom handler."""

        class CustomHandler(KVCacheHandler):
            pass

        custom = CustomHandler()
        # This would override the existing handler
        # Just verify registration works without error
        CacheTypeRegistry.register(custom)


class TestCacheListHandler:
    """Tests for CacheListHandler."""

    @pytest.fixture
    def handler(self):
        """Create a CacheListHandler."""
        return CacheListHandler()

    def test_cache_type(self, handler):
        """Test cache_type property."""
        assert handler.cache_type == CacheType.CACHE_LIST

    def test_supports_block_slicing(self, handler):
        """Test supports_block_slicing is False."""
        assert handler.supports_block_slicing is False

    def test_extract_state_with_sub_caches(self, handler):
        """Test extracting state from CacheList with sub-caches."""
        mock_cache = MagicMock(spec=[])  # spec=[] to avoid spurious attributes

        # Use real lightweight stub classes so type(obj).__name__ returns correct names
        class KVCache:
            def __init__(self, state, meta_state):
                self.state = state
                self.meta_state = meta_state

        sub_kv = KVCache(state=(MagicMock(), MagicMock()), meta_state=(64,))
        sub_kv2 = KVCache(state=(MagicMock(), MagicMock()), meta_state=(32,))

        mock_cache.caches = (sub_kv, sub_kv2)

        state = handler.extract_state(mock_cache)

        assert state["cache_type"] == "CacheList"
        assert state["is_full_state"] is True
        assert len(state["sub_states"]) == 2
        assert len(state["sub_class_names"]) == 2
        assert len(state["sub_meta_states"]) == 2
        assert state["sub_class_names"] == ["KVCache", "KVCache"]

    def test_extract_state_normalizes_sized_arrays_cache(self, handler):
        """Test that SizedArraysCache class name is normalized to ArraysCache."""
        from omlx.cache.type_handlers import SizedArraysCache

        inner_mock = MagicMock(spec=[])
        inner_mock.state = [MagicMock(), MagicMock()]
        inner_mock.meta_state = ()
        inner_mock.cache = [MagicMock(), MagicMock()]

        sized = SizedArraysCache(inner_mock, 128)

        mock_cache = MagicMock(spec=[])
        mock_cache.caches = (sized,)

        state = handler.extract_state(mock_cache)

        assert state["sub_class_names"] == ["ArraysCache"]

    def test_extract_state_empty_caches(self, handler):
        """Test extracting state from CacheList with no sub-caches."""
        mock_cache = MagicMock(spec=[])
        mock_cache.caches = None

        state = handler.extract_state(mock_cache)

        assert state["sub_states"] == []
        assert state["sub_class_names"] == []
        assert state["sub_meta_states"] == []

    def test_get_seq_len_from_4d_tensor(self, handler):
        """Test getting seq_len from 4D tensor sub-states."""
        mock_keys = MagicMock()
        mock_keys.shape = (1, 8, 128, 64)
        mock_values = MagicMock()

        state = {
            "sub_states": [(mock_keys, mock_values)],
        }

        seq_len = handler.get_seq_len(state)
        assert seq_len == 128

    def test_get_seq_len_no_4d_tensors(self, handler):
        """Test get_seq_len returns 0 when no 4D tensors."""
        state = {"sub_states": []}
        assert handler.get_seq_len(state) == 0

    def test_slice_state_returns_full(self, handler):
        """Test slice_state returns full state (non-sliceable)."""
        state = {
            "sub_states": ["s1", "s2"],
            "sub_class_names": ["KVCache", "KVCache"],
            "sub_meta_states": [(1,), (2,)],
        }

        sliced = handler.slice_state(state, 0, 64)

        assert sliced["sub_states"] == ["s1", "s2"]
        assert sliced["is_full_state"] is True
        assert sliced["cache_type"] == "CacheList"

    def test_concatenate_states_uses_latest(self, handler):
        """Test concatenate_states returns latest state."""
        states = [
            {"sub_states": ["old"]},
            {"sub_states": ["new"]},
        ]

        result = handler.concatenate_states(states)
        assert result["sub_states"] == ["new"]

    def test_concatenate_empty_states(self, handler):
        """Test concatenate_states with empty list."""
        assert handler.concatenate_states([]) == {}

    def test_reconstruct_cache_missing_meta_state(self, handler):
        """Test reconstruct returns None without meta_state."""
        result = handler.reconstruct_cache({"sub_states": []}, None)
        assert result is None

    def test_reconstruct_cache_length_mismatch(self, handler):
        """Test reconstruct returns None on length mismatch (zip truncate prevention)."""
        state = {"sub_states": ["s1", "s2"]}
        meta_state = (["KVCache"], [(1,)])  # Only 1 class name, 2 sub_states

        result = handler.reconstruct_cache(state, meta_state)
        assert result is None

    def test_reconstruct_cache_length_mismatch_meta(self, handler):
        """Test reconstruct returns None when sub_meta_states length mismatches."""
        state = {"sub_states": ["s1"]}
        meta_state = (["KVCache"], [(1,), (2,)])  # 1 class name, 2 sub_meta_states

        result = handler.reconstruct_cache(state, meta_state)
        assert result is None

    def test_get_state_keys(self, handler):
        """Test state keys."""
        keys = handler._get_state_keys()
        assert "sub_states" in keys
        assert "sub_class_names" in keys

    def test_get_meta_state_keys(self, handler):
        """Test meta state keys."""
        keys = handler._get_meta_state_keys()
        assert "class_names" in keys
        assert "sub_meta_states" in keys

    def test_get_seq_len_max_across_sub_caches(self, handler):
        """Test get_seq_len returns max seq_len across multiple sub-caches."""
        mock_keys1 = MagicMock()
        mock_keys1.shape = (1, 8, 64, 64)
        mock_keys2 = MagicMock()
        mock_keys2.shape = (1, 8, 256, 64)

        state = {
            "sub_states": [
                (mock_keys1, MagicMock()),
                (mock_keys2, MagicMock()),
            ],
        }

        seq_len = handler.get_seq_len(state)
        assert seq_len == 256


class TestCacheListHandlerWithMLX:
    """Tests for CacheListHandler that require MLX."""

    @pytest.fixture
    def mx(self):
        """Import MLX or skip."""
        try:
            import mlx.core as mx
            return mx
        except ImportError:
            pytest.skip("MLX not available")

    @pytest.fixture
    def handler(self):
        """Create a CacheListHandler."""
        return CacheListHandler()

    def test_reconstruct_cache_with_kvcache_subs(self, handler, mx):
        """Test reconstructing CacheList with KVCache sub-caches."""
        keys1 = mx.zeros((1, 8, 32, 64))
        values1 = mx.zeros((1, 8, 32, 64))
        keys2 = mx.zeros((1, 8, 32, 64))
        values2 = mx.zeros((1, 8, 32, 64))

        state = {
            "sub_states": [(keys1, values1), (keys2, values2)],
        }
        meta_state = (
            ["KVCache", "KVCache"],
            [(32,), (32,)],  # offset meta_state for each sub-cache
        )

        cache = handler.reconstruct_cache(state, meta_state)

        assert cache is not None
        assert hasattr(cache, "caches")
        assert len(cache.caches) == 2

    def test_reconstruct_cache_kvcache_no_fallback(self, handler, mx):
        """Test CacheList with KVCache sub-caches succeeds via from_state()
        without falling back to manual reconstruction (GLM-5 scenario)."""
        keys1 = mx.zeros((1, 8, 64, 64))
        values1 = mx.zeros((1, 8, 64, 64))
        keys2 = mx.zeros((1, 8, 64, 64))
        values2 = mx.zeros((1, 8, 64, 64))

        state = {
            "sub_states": [(keys1, values1), (keys2, values2)],
        }
        # Simulate the meta_state that prefix_cache creates after fix:
        # empty strings for KVCache sub-caches (no meta_state needed)
        meta_state = (
            ["KVCache", "KVCache"],
            ["", ""],
        )

        import logging
        from unittest.mock import patch

        with patch.object(
            logging.getLogger("omlx.cache.type_handlers"), "debug"
        ) as mock_debug:
            cache = handler.reconstruct_cache(state, meta_state)

        assert cache is not None
        assert hasattr(cache, "caches")
        assert len(cache.caches) == 2
        # Verify from_state() succeeded without fallback
        for call_args in mock_debug.call_args_list:
            assert "from_state() unavailable or failed" not in str(call_args)

    def test_reconstruct_cache_mixed_types(self, handler, mx):
        """Test reconstructing CacheList with ArraysCache + KVCache."""
        # ArraysCache sub-state: list of arrays
        arrays_state = [mx.zeros((1, 16)), mx.zeros((1, 32))]
        # KVCache sub-state: (keys, values)
        kv_keys = mx.zeros((1, 8, 32, 64))
        kv_values = mx.zeros((1, 8, 32, 64))

        state = {
            "sub_states": [arrays_state, (kv_keys, kv_values)],
        }
        meta_state = (
            ["ArraysCache", "KVCache"],
            [(), (32,)],
        )

        cache = handler.reconstruct_cache(state, meta_state)

        assert cache is not None
        assert hasattr(cache, "caches")
        assert len(cache.caches) == 2
