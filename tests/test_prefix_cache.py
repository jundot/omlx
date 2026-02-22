# SPDX-License-Identifier: Apache-2.0
"""
Tests for BlockAwarePrefixCache and related components.

This module tests the block-aware prefix caching system that uses
PagedCacheManager for block-based storage with SSD persistence.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from omlx.cache.paged_cache import (
    BlockHash,
    BlockTable,
    CacheBlock,
    PagedCacheManager,
    compute_block_hash,
)
from omlx.cache.prefix_cache import BlockAwarePrefixCache, BlockCacheEntry
from omlx.cache.stats import PrefixCacheStats


class MockModel:
    """Mock model for testing."""

    def __init__(self, num_layers: int = 32):
        self._num_layers = num_layers
        self.layers = [MagicMock() for _ in range(num_layers)]

    @property
    def args(self):
        mock_args = MagicMock()
        mock_args.num_hidden_layers = self._num_layers
        return mock_args


class TestBlockCacheEntry:
    """Tests for BlockCacheEntry dataclass."""

    def test_creation(self):
        """Test creating a cache entry."""
        block_table = BlockTable(request_id="req-001")
        entry = BlockCacheEntry(
            block_table=block_table,
            last_access=time.time(),
        )

        assert entry.block_table is block_table
        assert entry.last_access > 0


class TestBlockAwarePrefixCache:
    """Tests for BlockAwarePrefixCache."""

    @pytest.fixture
    def paged_cache(self):
        """Create a PagedCacheManager for testing."""
        return PagedCacheManager(
            block_size=4,  # Small block size for testing
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        return MockModel(num_layers=4)

    @pytest.fixture
    def prefix_cache(self, mock_model, paged_cache):
        """Create a BlockAwarePrefixCache for testing."""
        return BlockAwarePrefixCache(
            model=mock_model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=None,
        )

    def test_initialization(self, prefix_cache, mock_model, paged_cache):
        """Test cache initialization."""
        assert prefix_cache.model is mock_model
        assert prefix_cache.paged_cache is paged_cache
        assert prefix_cache.block_size == 4
        assert prefix_cache.expected_num_layers == 4

    def test_get_model_num_layers(self, paged_cache):
        """Test getting model num_layers from various sources."""
        # Model with layers attribute
        model1 = MockModel(num_layers=16)
        cache1 = BlockAwarePrefixCache(model1, paged_cache)
        assert cache1.expected_num_layers == 16

        # Model with args.num_hidden_layers
        model2 = MagicMock()
        model2.layers = None
        model2.args.num_hidden_layers = 24
        delattr(model2, 'layers')
        cache2 = BlockAwarePrefixCache(model2, paged_cache)
        assert cache2.expected_num_layers == 24

    def test_fetch_cache_miss(self, prefix_cache):
        """Test fetch_cache returns miss when no cache exists."""
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]

        block_table, remaining = prefix_cache.fetch_cache("req-001", tokens)

        assert block_table is None
        assert remaining == tokens
        assert prefix_cache._misses == 1

    def test_fetch_cache_empty_tokens(self, prefix_cache):
        """Test fetch_cache with empty tokens."""
        block_table, remaining = prefix_cache.fetch_cache("req-001", [])

        assert block_table is None
        assert remaining == []

    def test_store_cache_empty_tokens(self, prefix_cache):
        """Test store_cache with empty tokens returns None."""
        result = prefix_cache.store_cache("req-001", [], [])
        assert result is None

    def test_store_cache_creates_block_table(self, prefix_cache):
        """Test store_cache creates a block table."""
        tokens = [1, 2, 3, 4]  # One full block
        # Mock cache data (tensor data format)
        cache_data = [
            {"state": (MagicMock(), MagicMock()), "cache_type": "KVCache"}
            for _ in range(4)
        ]

        result = prefix_cache.store_cache("req-001", tokens, cache_data)

        assert result is not None
        assert result.request_id == "req-001"
        assert "req-001" in prefix_cache._request_tables

    def test_release_cache(self, prefix_cache, paged_cache):
        """Test releasing cache for a request."""
        # First create a block table with blocks
        table = paged_cache.create_block_table("req-001")
        block = paged_cache.allocate_block()
        table.block_ids.append(block.block_id)

        # Add to prefix cache tracking
        prefix_cache._request_tables["req-001"] = BlockCacheEntry(
            block_table=table,
            last_access=time.time(),
        )

        initial_free = paged_cache.free_blocks

        prefix_cache.release_cache("req-001")

        assert "req-001" not in prefix_cache._request_tables
        assert paged_cache.free_blocks == initial_free + 1

    def test_clear_request_entry(self, prefix_cache, paged_cache):
        """Test clearing request entry without freeing blocks."""
        table = paged_cache.create_block_table("req-001")
        prefix_cache._request_tables["req-001"] = BlockCacheEntry(
            block_table=table,
            last_access=time.time(),
        )

        prefix_cache.clear_request_entry("req-001")

        assert "req-001" not in prefix_cache._request_tables
        # Blocks should still be tracked in paged_cache
        assert "req-001" in paged_cache.request_tables

    def test_fork_cache(self, prefix_cache, paged_cache):
        """Test forking cache from one request to another."""
        # Create source with blocks
        source_table = paged_cache.create_block_table("req-source")
        block = paged_cache.allocate_block()
        source_table.block_ids.append(block.block_id)

        prefix_cache._request_tables["req-source"] = BlockCacheEntry(
            block_table=source_table,
            last_access=time.time(),
        )

        forked = prefix_cache.fork_cache("req-source", "req-forked")

        assert forked is not None
        assert forked.request_id == "req-forked"
        assert "req-forked" in prefix_cache._request_tables
        # Ref count should be incremented
        assert block.ref_count == 2

    def test_fork_cache_nonexistent(self, prefix_cache):
        """Test forking nonexistent cache returns None."""
        result = prefix_cache.fork_cache("nonexistent", "new-req")
        assert result is None

    def test_get_stats(self, prefix_cache):
        """Test getting statistics."""
        prefix_cache._hits = 5
        prefix_cache._misses = 3
        prefix_cache._tokens_saved = 100

        stats = prefix_cache.get_stats()

        assert isinstance(stats, PrefixCacheStats)
        assert stats.hits == 5
        assert stats.misses == 3
        assert stats.tokens_saved == 100

    def test_get_stats_dict(self, prefix_cache):
        """Test getting statistics as dictionary."""
        prefix_cache._hits = 10
        prefix_cache._misses = 5

        stats_dict = prefix_cache.get_stats_dict()

        assert "hits" in stats_dict
        assert "misses" in stats_dict
        assert "hit_rate" in stats_dict
        assert stats_dict["hit_rate"] == pytest.approx(10 / 15)

    def test_reset_stats(self, prefix_cache):
        """Test resetting statistics."""
        prefix_cache._hits = 10
        prefix_cache._misses = 5
        prefix_cache._tokens_saved = 500

        prefix_cache.reset_stats()

        assert prefix_cache._hits == 0
        assert prefix_cache._misses == 0
        assert prefix_cache._tokens_saved == 0

    def test_clear(self, prefix_cache, paged_cache):
        """Test clearing all cache data."""
        # Add some data
        table = paged_cache.create_block_table("req-001")
        prefix_cache._request_tables["req-001"] = BlockCacheEntry(
            block_table=table,
            last_access=time.time(),
        )
        prefix_cache._prefix_index[b"test_hash"] = (10, [1, 2], 2)
        prefix_cache._hits = 10

        cleared = prefix_cache.clear()

        assert cleared > 0
        assert len(prefix_cache._request_tables) == 0
        assert len(prefix_cache._prefix_index) == 0
        assert prefix_cache._hits == 0

    def test_len(self, prefix_cache):
        """Test __len__ returns number of request entries."""
        assert len(prefix_cache) == 0

        table = BlockTable(request_id="req-001")
        prefix_cache._request_tables["req-001"] = BlockCacheEntry(
            block_table=table,
            last_access=time.time(),
        )
        prefix_cache._request_tables["req-002"] = BlockCacheEntry(
            block_table=BlockTable(request_id="req-002"),
            last_access=time.time(),
        )

        assert len(prefix_cache) == 2

    def test_cache_manager_interface_fetch(self, prefix_cache):
        """Test CacheManager ABC fetch interface."""
        # Invalid key format
        value, hit = prefix_cache.fetch("invalid")
        assert hit is False
        assert value is None

        # Valid key format but miss
        value, hit = prefix_cache.fetch(("req-001", [1, 2, 3, 4]))
        assert hit is False

    def test_cache_manager_interface_store(self, prefix_cache):
        """Test CacheManager ABC store interface."""
        # Invalid key format
        result = prefix_cache.store("invalid", [])
        assert result is False

    def test_cache_manager_interface_evict(self, prefix_cache, paged_cache):
        """Test CacheManager ABC evict interface."""
        # Evict nonexistent
        result = prefix_cache.evict("req-nonexistent")
        assert result is False

        # Create entry then evict
        table = paged_cache.create_block_table("req-001")
        prefix_cache._request_tables["req-001"] = BlockCacheEntry(
            block_table=table,
            last_access=time.time(),
        )

        result = prefix_cache.evict("req-001")
        assert result is True
        assert "req-001" not in prefix_cache._request_tables

    def test_cache_manager_interface_properties(self, prefix_cache):
        """Test CacheManager ABC property interface."""
        assert prefix_cache.size == 0
        assert prefix_cache.max_size == 100  # max_blocks

    def test_set_paged_ssd_cache_manager(self, prefix_cache):
        """Test setting SSD cache manager."""
        mock_ssd_cache = MagicMock()

        prefix_cache.set_paged_ssd_cache_manager(mock_ssd_cache)

        assert prefix_cache.paged_ssd_cache is mock_ssd_cache

    def test_set_cold_restore_callback(self, prefix_cache):
        """Test setting cold restore callback."""

        def restore_callback(block_id: int, block_hash: bytes) -> bool:
            return True

        prefix_cache.set_cold_restore_callback(restore_callback)
        assert prefix_cache._cold_restore_callback is restore_callback


class TestBlockAwarePrefixCacheWithSSD:
    """Tests for BlockAwarePrefixCache with SSD cache manager."""

    @pytest.fixture
    def paged_cache(self):
        """Create a PagedCacheManager."""
        return PagedCacheManager(
            block_size=4,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )

    @pytest.fixture
    def mock_ssd_cache(self):
        """Create a mock SSD cache manager."""
        mock = MagicMock()
        mock.save_block.return_value = True
        mock.load_block.return_value = None
        mock.load_block_with_metadata.return_value = (None, None)
        mock.has_block.return_value = False
        return mock

    @pytest.fixture
    def prefix_cache_with_ssd(self, paged_cache, mock_ssd_cache):
        """Create a BlockAwarePrefixCache with SSD manager."""
        model = MockModel(num_layers=4)
        return BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd_cache,
        )

    def test_get_cache_for_generation_no_entry(self, prefix_cache_with_ssd):
        """Test get_cache_for_generation with no entry."""
        result, was_loaded = prefix_cache_with_ssd.get_cache_for_generation("req-001")

        assert result is None
        assert was_loaded is False

    def test_reconstruct_cache_empty_table(self, prefix_cache_with_ssd):
        """Test reconstruct_cache with empty block table."""
        table = BlockTable(request_id="req-001")

        result = prefix_cache_with_ssd.reconstruct_cache(table)

        assert result is None

    def test_reconstruct_cache_no_ssd_manager(self, paged_cache):
        """Test reconstruct_cache without SSD manager."""
        model = MockModel(num_layers=4)
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=None,
        )

        table = BlockTable(request_id="req-001", block_ids=[1, 2])

        result = cache.reconstruct_cache(table)

        assert result is None


class TestPrefixIndexOperations:
    """Tests for prefix index operations."""

    @pytest.fixture
    def paged_cache(self):
        """Create a PagedCacheManager."""
        return PagedCacheManager(
            block_size=4,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )

    @pytest.fixture
    def prefix_cache(self, paged_cache):
        """Create a BlockAwarePrefixCache."""
        model = MockModel(num_layers=4)
        return BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
        )

    def test_update_prefix_index(self, prefix_cache, paged_cache):
        """Test _update_prefix_index method."""
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]  # 2 blocks
        block_ids = [1, 2]

        # Allocate blocks and set them up
        blocks = paged_cache.get_new_blocks(2)
        block_ids = [b.block_id for b in blocks]

        prefix_cache._update_prefix_index(tokens, block_ids)

        # Prefix index should have entries
        assert len(prefix_cache._prefix_index) > 0

    def test_find_best_prefix_match_no_match(self, prefix_cache):
        """Test _find_best_prefix_match with no matching prefix."""
        tokens = [1, 2, 3, 4]

        result = prefix_cache._find_best_prefix_match(tokens)

        assert result is None

    def test_find_best_prefix_match_with_match(self, prefix_cache, paged_cache):
        """Test _find_best_prefix_match finding a matching prefix."""
        tokens = [1, 2, 3, 4]
        block_ids = [1, 2]

        # Manually add to prefix index
        block_hash = compute_block_hash(
            b"", tokens, model_name=paged_cache.model_name
        )
        prefix_cache._prefix_index[block_hash] = (4, block_ids, 1)

        result = prefix_cache._find_best_prefix_match(tokens)

        assert result is not None
        prefix_len, matched_ids, num_blocks = result
        assert prefix_len == 4


class TestValidateBlockCacheData:
    """Tests for _validate_block_cache_data method."""

    @pytest.fixture
    def prefix_cache(self):
        """Create a BlockAwarePrefixCache."""
        paged_cache = PagedCacheManager(
            block_size=4,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        model = MockModel(num_layers=4)
        return BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
        )

    def test_validate_empty_data(self, prefix_cache):
        """Test validation of empty cache data."""
        assert prefix_cache._validate_block_cache_data([]) is False

    def test_validate_with_none_values(self, prefix_cache):
        """Test validation fails with None keys/values."""
        cache_data = [(None, MagicMock())]
        assert prefix_cache._validate_block_cache_data(cache_data) is False

        cache_data = [(MagicMock(), None)]
        assert prefix_cache._validate_block_cache_data(cache_data) is False

    def test_validate_with_arrays_cache_skipped(self, prefix_cache):
        """Test validation skips ArraysCache layers for seq_len check."""
        # Create mock tensors with shape
        mock_keys1 = MagicMock()
        mock_keys1.shape = (1, 8, 64, 64)  # KVCache shape
        mock_values1 = MagicMock()

        mock_keys2 = MagicMock()
        mock_keys2.shape = (1, 128)  # ArraysCache has different shape
        mock_values2 = MagicMock()

        cache_data = [
            (mock_keys1, mock_values1),
            (mock_keys2, mock_values2),
        ]
        layer_cache_types = ["KVCache", "ArraysCache"]

        # Should pass because ArraysCache is skipped in validation
        result = prefix_cache._validate_block_cache_data(
            cache_data, layer_cache_types
        )
        assert result is True

    def test_validate_seq_len_mismatch(self, prefix_cache):
        """Test validation fails on seq_len mismatch."""
        mock_keys1 = MagicMock()
        mock_keys1.shape = (1, 8, 64, 64)
        mock_values1 = MagicMock()

        mock_keys2 = MagicMock()
        mock_keys2.shape = (1, 8, 32, 64)  # Different seq_len
        mock_values2 = MagicMock()

        cache_data = [
            (mock_keys1, mock_values1),
            (mock_keys2, mock_values2),
        ]
        layer_cache_types = ["KVCache", "KVCache"]

        result = prefix_cache._validate_block_cache_data(
            cache_data, layer_cache_types
        )
        assert result is False


class TestArraysCacheLastBlockOnly:
    """Tests for ArraysCache last-block-only storage and partial match rejection."""

    @pytest.fixture
    def mx(self):
        """Import MLX or skip."""
        try:
            import mlx.core as mx
            return mx
        except ImportError:
            pytest.skip("MLX not available")

    @pytest.fixture
    def paged_cache(self):
        """Create a PagedCacheManager."""
        return PagedCacheManager(
            block_size=4,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )

    @pytest.fixture
    def prefix_cache(self, paged_cache):
        """Create a BlockAwarePrefixCache."""
        model = MockModel(num_layers=2)
        return BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
        )

    def test_extract_block_arrays_cache_last_block_stores_full_state(
        self, prefix_cache, mx
    ):
        """Last block should store the full ArraysCache state."""
        conv_state = mx.ones((1, 3, 64))
        ssm_state = mx.ones((1, 32, 128, 128))
        cache_data = [
            {
                "state": (conv_state, ssm_state),
                "cache_type": "ArraysCache",
                "class_name": "ArraysCache",
            },
        ]

        result = prefix_cache._extract_block_tensor_slice(
            cache_data, 0, 4, model_cache_config=None, is_last_block=True,
        )

        assert result is not None
        assert len(result) == 1
        keys, values = result[0]
        # Should be full state, not placeholder
        assert keys.shape == (1, 3, 64)
        assert values.shape == (1, 32, 128, 128)

    def test_extract_block_arrays_cache_non_last_block_stores_placeholder(
        self, prefix_cache, mx
    ):
        """Non-last block should store placeholder (1,) for ArraysCache."""
        conv_state = mx.ones((1, 3, 64))
        ssm_state = mx.ones((1, 32, 128, 128))
        cache_data = [
            {
                "state": (conv_state, ssm_state),
                "cache_type": "ArraysCache",
                "class_name": "ArraysCache",
            },
        ]

        result = prefix_cache._extract_block_tensor_slice(
            cache_data, 0, 4, model_cache_config=None, is_last_block=False,
        )

        assert result is not None
        assert len(result) == 1
        keys, values = result[0]
        # Should be placeholder
        assert keys.shape == (1,)
        assert values.shape == (1,)

    def test_extract_block_hybrid_model_arrays_cache_and_kvcache(
        self, prefix_cache, mx
    ):
        """Hybrid model: KVCache sliced normally, ArraysCache last-block-only."""
        from omlx.cache.hybrid_cache import ModelCacheConfig

        kv_keys = mx.ones((1, 8, 8, 64))
        kv_values = mx.ones((1, 8, 8, 64))
        conv_state = mx.ones((1, 3, 64))
        ssm_state = mx.ones((1, 32, 128, 128))

        cache_data = [
            {
                "state": (kv_keys, kv_values),
                "cache_type": "KVCache",
                "class_name": "KVCache",
            },
            {
                "state": (conv_state, ssm_state),
                "cache_type": "ArraysCache",
                "class_name": "ArraysCache",
            },
        ]

        config = ModelCacheConfig.from_type_list(
            ["KVCache", "ArraysCache"], model_name="test"
        )

        # Non-last block
        result = prefix_cache._extract_block_tensor_slice(
            cache_data, 0, 4, model_cache_config=config, is_last_block=False,
        )
        assert result is not None
        assert len(result) == 2
        # KVCache layer should be sliced normally
        assert result[0][0].shape[2] == 4  # seq_len slice
        # ArraysCache layer should be placeholder
        assert result[1][0].shape == (1,)

        # Last block
        result = prefix_cache._extract_block_tensor_slice(
            cache_data, 4, 8, model_cache_config=config, is_last_block=True,
        )
        assert result is not None
        assert len(result) == 2
        # KVCache layer should be sliced
        assert result[0][0].shape[2] == 4
        # ArraysCache layer should have full state
        assert result[1][0].shape == (1, 3, 64)

    def test_reconstruct_arrays_cache_partial_match_returns_none(
        self, prefix_cache, mx
    ):
        """Partial match (placeholder in last block) should return None."""
        from omlx.cache.paged_ssd_cache import PagedSSDCacheManager

        # Create mock SSD cache
        mock_ssd = MagicMock(spec=PagedSSDCacheManager)

        model = MockModel(num_layers=2)
        paged_cache = PagedCacheManager(
            block_size=4, max_blocks=100, model_name="test-model",
            initial_blocks=100,
        )
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )

        # Allocate blocks
        block1 = paged_cache.allocate_block()
        block1.block_hash = b"hash1"
        block1.token_count = 4
        block2 = paged_cache.allocate_block()
        block2.block_hash = b"hash2"
        block2.token_count = 4

        block_table = BlockTable(
            request_id="req-001",
            block_ids=[block1.block_id, block2.block_id],
            num_tokens=8,
        )

        # Block data: KVCache layer sliced, ArraysCache layer placeholder
        kv_slice = (mx.ones((1, 8, 4, 64)), mx.ones((1, 8, 4, 64)))
        placeholder = (mx.zeros((1,)), mx.zeros((1,)))

        block1_data = [kv_slice, placeholder]  # ArraysCache = placeholder (non-last)
        block2_data = [kv_slice, placeholder]  # ArraysCache = placeholder (still non-last in original)

        mock_ssd.load_block_with_metadata.side_effect = [
            (block1_data, {
                "model_name": "test-model",
                "num_layers": 2,
                "layer_cache_types": ["KVCache", "ArraysCache"],
                "layer_meta_states": [(), ()],
            }),
            (block2_data, {
                "model_name": "test-model",
                "num_layers": 2,
                "layer_cache_types": ["KVCache", "ArraysCache"],
                "layer_meta_states": [(), ()],
            }),
        ]

        result = cache.reconstruct_cache(block_table)

        # Should return None because ArraysCache layer has placeholder
        assert result is None

    def test_reconstruct_arrays_cache_exact_match_succeeds(
        self, prefix_cache, mx
    ):
        """Exact match (full state in last block) should reconstruct successfully."""
        from omlx.cache.paged_ssd_cache import PagedSSDCacheManager

        mock_ssd = MagicMock(spec=PagedSSDCacheManager)

        model = MockModel(num_layers=1)
        paged_cache = PagedCacheManager(
            block_size=4, max_blocks=100, model_name="test-model",
            initial_blocks=100,
        )
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )

        block1 = paged_cache.allocate_block()
        block1.block_hash = b"hash1"
        block1.token_count = 4
        block2 = paged_cache.allocate_block()
        block2.block_hash = b"hash2"
        block2.token_count = 4

        block_table = BlockTable(
            request_id="req-001",
            block_ids=[block1.block_id, block2.block_id],
            num_tokens=8,
        )

        # First block: placeholder, Second block (last): full state
        conv_state = mx.ones((1, 3, 64))
        ssm_state = mx.ones((1, 32, 128, 128))

        block1_data = [(mx.zeros((1,)), mx.zeros((1,)))]  # placeholder
        block2_data = [(conv_state, ssm_state)]  # full state

        mock_ssd.load_block_with_metadata.side_effect = [
            (block1_data, {
                "model_name": "test-model",
                "num_layers": 1,
                "layer_cache_types": ["ArraysCache"],
                "layer_meta_states": [()],
            }),
            (block2_data, {
                "model_name": "test-model",
                "num_layers": 1,
                "layer_cache_types": ["ArraysCache"],
                "layer_meta_states": [()],
            }),
        ]

        result = cache.reconstruct_cache(block_table)

        # Should succeed because last block has full state
        assert result is not None
        assert len(result) == 1

    def test_store_cache_skips_partial_blocks(self, mx):
        """store_cache should only create full blocks, skipping partial trailing tokens.

        get_computed_blocks() matches full blocks only (floor division), so
        partial blocks are never matched. Skipping them ensures is_last_block
        points to the last full block, which is critical for ArraysCache/
        RotatingKVCache last-block-only storage.
        """
        block_size = 4
        paged_cache = PagedCacheManager(
            block_size=block_size,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        model = MockModel(num_layers=1)
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
        )

        # 10 tokens = 2 full blocks (8 tokens) + 2 partial tokens
        tokens = list(range(10))
        cache_data = [
            {"state": (mx.ones((1, 8, 10, 64)), mx.ones((1, 8, 10, 64))),
             "cache_type": "KVCache", "class_name": "KVCache"}
        ]

        result = cache.store_cache("req-001", tokens, cache_data)

        assert result is not None
        # Should create exactly 2 blocks (10 // 4 = 2), not 3
        assert len(result.block_ids) == 2
        # num_tokens should reflect only full blocks
        assert result.num_tokens == 8  # 2 blocks * 4 tokens

    def test_store_cache_arrayscache_partial_trailing_uses_last_full_block_state(self, mx):
        """ArraysCache with trailing partial tokens stores only full blocks safely."""
        from omlx.cache.hybrid_cache import ModelCacheConfig

        block_size = 4
        paged_cache = PagedCacheManager(
            block_size=block_size,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        mock_ssd = MagicMock()
        mock_ssd.save_block.return_value = True

        model = MockModel(num_layers=1)
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )

        # 7 tokens = 1 full block (4) + 3 trailing partial tokens
        tokens = list(range(7))
        conv_state = mx.ones((1, 3, 64))
        ssm_state = mx.ones((1, 32, 128, 128))
        cache_data = [
            {
                "state": (conv_state, ssm_state),
                "cache_type": "ArraysCache",
                "class_name": "ArraysCache",
            }
        ]
        model_cache_config = ModelCacheConfig.from_type_list(
            ["ArraysCache"], model_name="test-model"
        )

        result = cache.store_cache(
            "req-001",
            tokens,
            cache_data,
            model_cache_config=model_cache_config,
        )

        assert result is not None
        assert len(result.block_ids) == 1
        assert result.num_tokens == 4
        mock_ssd.save_block.assert_called_once()

        saved_data = mock_ssd.save_block.call_args.kwargs["cache_data"]
        saved_conv_state, saved_ssm_state = saved_data[0]
        assert saved_conv_state.shape == conv_state.shape
        assert saved_ssm_state.shape == ssm_state.shape

    def test_store_cache_all_partial_creates_no_blocks(self, mx):
        """Tokens fewer than block_size should create no blocks."""
        block_size = 4
        paged_cache = PagedCacheManager(
            block_size=block_size,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        model = MockModel(num_layers=1)
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
        )

        # 3 tokens < block_size=4 -> 0 full blocks
        tokens = [1, 2, 3]
        cache_data = [
            {"state": (mx.ones((1, 8, 3, 64)), mx.ones((1, 8, 3, 64))),
             "cache_type": "KVCache", "class_name": "KVCache"}
        ]

        result = cache.store_cache("req-001", tokens, cache_data)

        assert result is not None
        assert len(result.block_ids) == 0
        assert result.num_tokens == 0

    def test_store_cache_exact_multiple_creates_all_blocks(self, mx):
        """Tokens exactly divisible by block_size should create all blocks."""
        block_size = 4
        paged_cache = PagedCacheManager(
            block_size=block_size,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        model = MockModel(num_layers=1)
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
        )

        # 8 tokens = exactly 2 blocks
        tokens = list(range(8))
        cache_data = [
            {"state": (mx.ones((1, 8, 8, 64)), mx.ones((1, 8, 8, 64))),
             "cache_type": "KVCache", "class_name": "KVCache"}
        ]

        result = cache.store_cache("req-001", tokens, cache_data)

        assert result is not None
        assert len(result.block_ids) == 2
        assert result.num_tokens == 8

    def test_store_cache_with_existing_prefix_uses_global_cache_indices(self, mx):
        """Store new blocks from full-sequence cache slices after cache hit.

        When a request reuses prefix blocks, extracted cache data includes the
        full sequence (prefix + newly processed suffix). New block slicing must
        use global token indices, otherwise wrong KV ranges are persisted.
        """
        block_size = 4
        paged_cache = PagedCacheManager(
            block_size=block_size,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        mock_ssd = MagicMock()
        mock_ssd.save_block.return_value = True

        model = MockModel(num_layers=1)
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )

        # Existing cached prefix (1 full block).
        existing_tokens = [10, 11, 12, 13]
        block_table = paged_cache.create_block_table("req-001")
        existing_block = paged_cache.allocate_block()
        assert existing_block is not None
        existing_hash = compute_block_hash(None, existing_tokens, model_name="test-model")
        existing_block.block_hash = existing_hash
        existing_block.token_count = block_size
        block_table.block_ids.append(existing_block.block_id)
        block_table.num_tokens = block_size
        paged_cache.register_block_hash(existing_block, existing_tokens, None)

        # Full sequence and full-sequence cache tensors.
        full_tokens = existing_tokens + [20, 21, 22, 23]
        keys = mx.arange(8, dtype=mx.float32).reshape(1, 1, 8, 1)
        values = (mx.arange(8, dtype=mx.float32) + 100).reshape(1, 1, 8, 1)
        cache_data = [
            {"state": (keys, values), "cache_type": "KVCache", "class_name": "KVCache"}
        ]

        result = cache.store_cache("req-001", full_tokens, cache_data)

        assert result is not None
        # Existing 1 block + newly stored 1 block
        assert len(result.block_ids) == 2
        assert mock_ssd.save_block.call_count == 1

        saved_block_data = mock_ssd.save_block.call_args.kwargs["cache_data"]
        saved_keys, saved_values = saved_block_data[0]

        # New block must use global slice [4:8], not [0:4].
        expected_keys = keys[:, :, 4:8, :]
        expected_values = values[:, :, 4:8, :]
        assert saved_keys.tolist() == expected_keys.tolist()
        assert saved_values.tolist() == expected_values.tolist()


class TestPrefixCacheCacheList:
    """Tests for CacheList support in BlockAwarePrefixCache."""

    @pytest.fixture
    def mx(self):
        """Import MLX or skip."""
        try:
            import mlx.core as mx
            return mx
        except ImportError:
            pytest.skip("MLX not available")

    @pytest.fixture
    def paged_cache(self):
        """Create a PagedCacheManager for testing."""
        return PagedCacheManager(
            block_size=4,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        return MockModel(num_layers=4)

    @pytest.fixture
    def prefix_cache(self, mock_model, paged_cache):
        """Create a BlockAwarePrefixCache for testing."""
        return BlockAwarePrefixCache(
            model=mock_model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=None,
        )

    def test_get_cache_seq_len_cache_list_only(self, prefix_cache, mx):
        """Test _get_cache_seq_len with all-CacheList model (e.g., deepseek_v32)."""
        # CacheList sub-states with 4D tensors
        sub_keys = mx.zeros((1, 8, 128, 64))
        sub_values = mx.zeros((1, 8, 128, 64))

        cache_data = [
            {
                'state': [(sub_keys, sub_values)],  # CacheList sub-states
                'cache_type': 'CacheList',
                'class_name': 'CacheList',
            },
            {
                'state': [(sub_keys, sub_values)],
                'cache_type': 'CacheList',
                'class_name': 'CacheList',
            },
        ]

        seq_len = prefix_cache._get_cache_seq_len(cache_data)
        assert seq_len == 128

    def test_get_cache_seq_len_mixed_kvcache_and_cache_list(self, prefix_cache, mx):
        """Test _get_cache_seq_len with mixed KVCache + CacheList model."""
        kv_keys = mx.zeros((1, 8, 256, 64))
        kv_values = mx.zeros((1, 8, 256, 64))
        sub_keys = mx.zeros((1, 4, 128, 64))

        cache_data = [
            {
                'state': (kv_keys, kv_values),
                'cache_type': 'KVCache',
                'class_name': 'KVCache',
            },
            {
                'state': [(sub_keys, MagicMock())],
                'cache_type': 'CacheList',
                'class_name': 'CacheList',
            },
        ]

        seq_len = prefix_cache._get_cache_seq_len(cache_data)
        # Should use KVCache layer (Step 1), not CacheList
        assert seq_len == 256

    def test_get_cache_seq_len_cache_list_skipped_in_step1(self, prefix_cache, mx):
        """Test CacheList is skipped in Step 1 and Step 2."""
        sub_keys = mx.zeros((1, 8, 64, 64))

        cache_data = [
            {
                'state': [(sub_keys, MagicMock())],
                'cache_type': 'CacheList',
                'class_name': 'CacheList',
            },
        ]

        # CacheList is in non_sliceable_types, so Step 1 & 2 skip it
        # Step 3 finds the sub-cache seq_len
        seq_len = prefix_cache._get_cache_seq_len(cache_data)
        assert seq_len == 64

    def test_get_cache_seq_len_pure_rotating_kvcache(self, prefix_cache, mx):
        """Test _get_cache_seq_len for pure RotatingKVCache model."""
        rot_keys = mx.zeros((1, 8, 96, 64))
        rot_values = mx.zeros((1, 8, 96, 64))

        cache_data = [
            {
                'state': (rot_keys, rot_values),
                'cache_type': 'RotatingKVCache',
                'class_name': 'RotatingKVCache',
            },
        ]

        # Step 1 skips RotatingKVCache; Step 2 must still recover seq_len.
        seq_len = prefix_cache._get_cache_seq_len(cache_data)
        assert seq_len == 96

    def test_extract_block_tensor_slice_cache_list_last_block(self, prefix_cache, mx):
        """Test _extract_block_tensor_slice for CacheList on last block."""
        from omlx.cache.hybrid_cache import ModelCacheConfig

        sub_keys = mx.zeros((1, 8, 32, 64))
        sub_values = mx.ones((1, 8, 32, 64))

        cache_data = [
            {
                'state': [(sub_keys, sub_values)],
                'cache_type': 'CacheList',
                'class_name': 'CacheList',
            },
        ]
        config = ModelCacheConfig.from_type_list(["CacheList"])

        result = prefix_cache._extract_block_tensor_slice(
            cache_data, 0, 32, model_cache_config=config, is_last_block=True,
        )

        assert result is not None
        assert len(result) == 1
        # CacheList marker format
        assert result[0][0] == '__cache_list__'
        assert len(result[0][1]) == 1  # One sub-cache
        assert result[0][1][0][0].shape == (1, 8, 32, 64)

    def test_extract_block_tensor_slice_cache_list_non_last_sliceable(self, prefix_cache, mx):
        """Test _extract_block_tensor_slice for CacheList with sliceable sub-caches on non-last block.

        When all sub-caches are 4D KVCache tensors, they should be sliced
        per-block instead of using last-block-only placeholder storage.
        """
        from omlx.cache.hybrid_cache import ModelCacheConfig

        sub_keys = mx.zeros((1, 8, 32, 64))
        sub_values = mx.ones((1, 8, 32, 64))

        cache_data = [
            {
                'state': [(sub_keys, sub_values)],
                'cache_type': 'CacheList',
                'class_name': 'CacheList',
            },
        ]
        config = ModelCacheConfig.from_type_list(["CacheList"])

        result = prefix_cache._extract_block_tensor_slice(
            cache_data, 0, 16, model_cache_config=config, is_last_block=False,
        )

        assert result is not None
        assert len(result) == 1
        # Sliceable sub-caches: per-block sliced data, not placeholder
        assert result[0][0] == '__cache_list__'
        assert len(result[0][1]) == 1
        assert result[0][1][0][0].shape == (1, 8, 16, 64)
        assert result[0][1][0][1].shape == (1, 8, 16, 64)

    def test_extract_block_tensor_slice_cache_list_zero_dim_values(self, prefix_cache, mx):
        """Test per-block slicing for CacheList with zero-dim values (DSA indexer)."""
        from omlx.cache.hybrid_cache import ModelCacheConfig

        # GLM-5 style: main attention + indexer with zero head_dim
        sub_keys1 = mx.zeros((1, 1, 64, 512))
        sub_values1 = mx.zeros((1, 1, 64, 64))
        sub_keys2 = mx.zeros((1, 1, 64, 128))
        sub_values2 = mx.zeros((1, 1, 64, 0))  # zero head_dim

        cache_data = [
            {
                'state': [(sub_keys1, sub_values1), (sub_keys2, sub_values2)],
                'cache_type': 'CacheList',
                'class_name': 'CacheList',
            },
        ]
        config = ModelCacheConfig.from_type_list(["CacheList"])

        result = prefix_cache._extract_block_tensor_slice(
            cache_data, 0, 32, model_cache_config=config, is_last_block=False,
        )

        assert result is not None
        assert result[0][0] == '__cache_list__'
        assert len(result[0][1]) == 2
        # Sub-cache 0: sliced normally
        assert result[0][1][0][0].shape == (1, 1, 32, 512)
        assert result[0][1][0][1].shape == (1, 1, 32, 64)
        # Sub-cache 1: sliced, values remain zero-dim
        assert result[0][1][1][0].shape == (1, 1, 32, 128)
        assert result[0][1][1][1].shape == (1, 1, 32, 0)

    def test_validate_block_cache_data_cache_list(self, prefix_cache, mx):
        """Test _validate_block_cache_data with CacheList layers."""
        # CacheList as list format (last block)
        cache_data = [
            [(mx.zeros((1, 8, 32, 64)), mx.zeros((1, 8, 32, 64)))],  # CacheList sub-cache list
            (mx.zeros((1, 8, 32, 64)), mx.zeros((1, 8, 32, 64))),  # Standard KVCache
        ]
        layer_cache_types = ["CacheList", "KVCache"]

        result = prefix_cache._validate_block_cache_data(cache_data, layer_cache_types)
        assert result is True

    def test_validate_block_cache_data_cache_list_placeholder(self, prefix_cache, mx):
        """Test _validate_block_cache_data with CacheList placeholder."""
        cache_data = [
            (mx.zeros((1,)), mx.zeros((1,))),  # CacheList placeholder (falls through to tuple check)
            (mx.zeros((1, 8, 32, 64)), mx.zeros((1, 8, 32, 64))),  # Standard KVCache
        ]
        layer_cache_types = ["CacheList", "KVCache"]

        result = prefix_cache._validate_block_cache_data(cache_data, layer_cache_types)
        assert result is True

    def test_find_kv_shape_ref_skips_cache_list(self, prefix_cache, mx):
        """Test _find_kv_shape_ref skips CacheList layers."""
        all_block_data = [
            [
                [(mx.zeros((1, 8, 32, 64)), mx.zeros((1, 8, 32, 64)))],  # CacheList: List[Tuple]
                (mx.zeros((1, 4, 32, 128)), mx.zeros((1, 4, 32, 128))),  # KVCache
            ]
        ]
        layer_cache_types = ["CacheList", "KVCache"]

        result = prefix_cache._find_kv_shape_ref(all_block_data, layer_cache_types)
        assert result == (4, 128)  # From KVCache layer, not CacheList

    def test_reconstruct_cache_list_partial_match_reject(self, mx):
        """Test reconstruct_cache rejects CacheList with placeholder (partial match)."""
        from omlx.cache.paged_ssd_cache import PagedSSDCacheManager

        mock_ssd = MagicMock(spec=PagedSSDCacheManager)

        model = MockModel(num_layers=1)
        paged_cache = PagedCacheManager(
            block_size=4, max_blocks=100, model_name="test-model",
            initial_blocks=100,
        )
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )

        # Allocate a block
        block1 = paged_cache.allocate_block()
        block1.block_hash = b"hash_cl1"
        block1.token_count = 4

        block_table = BlockTable(
            request_id="req-cl-partial",
            block_ids=[block1.block_id],
            num_tokens=4,
        )

        # Block data: CacheList placeholder (partial match)
        placeholder = (mx.zeros((1,)), mx.zeros((1,)))
        block1_data = [placeholder]

        mock_ssd.load_block_with_metadata.return_value = (
            block1_data,
            {
                "model_name": "test-model",
                "num_layers": 1,
                "layer_cache_types": ["CacheList"],
                "layer_meta_states": [()],
            },
        )

        result = cache.reconstruct_cache(block_table)
        # Should return None because CacheList layer has placeholder
        assert result is None


class TestWalkBackTruncation:
    """Tests for walk-back truncation of non-sliceable caches."""

    @pytest.fixture
    def mx(self):
        """Import MLX or skip."""
        try:
            import mlx.core as mx
            return mx
        except ImportError:
            pytest.skip("MLX not available")

    # ------------------------------------------------------------------
    # _is_placeholder_state
    # ------------------------------------------------------------------

    def test_is_placeholder_state_detects_placeholder(self, mx):
        """Placeholder tuple with shape (1,) should be detected."""
        placeholder = (mx.zeros((1,)), mx.zeros((1,)))
        assert BlockAwarePrefixCache._is_placeholder_state(placeholder) is True

    def test_is_placeholder_state_rejects_real_arrays_cache(self, mx):
        """Real ArraysCache state should not be flagged as placeholder."""
        real_state = (mx.ones((1, 3, 64)), mx.ones((1, 32, 128, 128)))
        assert BlockAwarePrefixCache._is_placeholder_state(real_state) is False

    def test_is_placeholder_state_rejects_kv_cache(self, mx):
        """Standard 4D KVCache tensors should not be flagged."""
        kv_state = (mx.ones((1, 8, 4, 64)), mx.ones((1, 8, 4, 64)))
        assert BlockAwarePrefixCache._is_placeholder_state(kv_state) is False

    def test_is_placeholder_state_rejects_list(self, mx):
        """CacheList real data (list format) should not be flagged."""
        cache_list_data = [
            (mx.ones((1, 8, 4, 64)), mx.ones((1, 8, 4, 64))),
        ]
        assert BlockAwarePrefixCache._is_placeholder_state(cache_list_data) is False

    # ------------------------------------------------------------------
    # _find_walk_back_truncation_point
    # ------------------------------------------------------------------

    def test_walk_back_no_truncation_when_last_block_valid(self, mx):
        """No truncation when the last block has real state."""
        model = MockModel(num_layers=2)
        paged_cache = PagedCacheManager(
            block_size=4, max_blocks=100, model_name="test-model",
            initial_blocks=100,
        )
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=None,
        )

        placeholder = (mx.zeros((1,)), mx.zeros((1,)))
        real_state = (mx.ones((1, 3, 64)), mx.ones((1, 32, 128, 128)))
        kv = (mx.ones((1, 8, 4, 64)), mx.ones((1, 8, 4, 64)))

        all_block_data = [
            [kv, placeholder],  # block 0
            [kv, real_state],   # block 1 (last, valid)
        ]
        layer_cache_types = ["KVCache", "ArraysCache"]

        result = cache._find_walk_back_truncation_point(
            all_block_data, layer_cache_types
        )
        assert result is None  # No truncation needed

    def test_walk_back_multi_turn_pattern(self, mx):
        """Walk-back finds the latest block with valid ArraysCache state.

        Simulates multi-turn pattern:
        A[p] B[p] C[real] D[p] E[real] F[p]
        Should walk back to E (index 4).
        """
        model = MockModel(num_layers=2)
        paged_cache = PagedCacheManager(
            block_size=4, max_blocks=100, model_name="test-model",
            initial_blocks=100,
        )
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=None,
        )

        placeholder = (mx.zeros((1,)), mx.zeros((1,)))
        real_state = (mx.ones((1, 3, 64)), mx.ones((1, 32, 128, 128)))
        kv = (mx.ones((1, 8, 4, 64)), mx.ones((1, 8, 4, 64)))

        # A[p] B[p] C[real] D[p] E[real] F[p]
        all_block_data = [
            [kv, placeholder],  # A
            [kv, placeholder],  # B
            [kv, real_state],   # C (turn 1 last block)
            [kv, placeholder],  # D
            [kv, real_state],   # E (turn 2 last block)
            [kv, placeholder],  # F (last loaded, placeholder)
        ]
        layer_cache_types = ["KVCache", "ArraysCache"]

        result = cache._find_walk_back_truncation_point(
            all_block_data, layer_cache_types
        )
        assert result == 4  # Block E (index 4)

    def test_walk_back_all_placeholders_returns_none(self, mx):
        """All blocks have placeholders -- no valid fallback exists."""
        model = MockModel(num_layers=2)
        paged_cache = PagedCacheManager(
            block_size=4, max_blocks=100, model_name="test-model",
            initial_blocks=100,
        )
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=None,
        )

        placeholder = (mx.zeros((1,)), mx.zeros((1,)))
        kv = (mx.ones((1, 8, 4, 64)), mx.ones((1, 8, 4, 64)))

        all_block_data = [
            [kv, placeholder],
            [kv, placeholder],
            [kv, placeholder],
        ]
        layer_cache_types = ["KVCache", "ArraysCache"]

        result = cache._find_walk_back_truncation_point(
            all_block_data, layer_cache_types
        )
        assert result is None  # No valid block found

    def test_walk_back_includes_rotating_kv_cache(self, mx):
        """RotatingKVCache placeholders should walk back to latest valid block."""
        model = MockModel(num_layers=2)
        paged_cache = PagedCacheManager(
            block_size=4, max_blocks=100, model_name="test-model",
            initial_blocks=100,
        )
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=None,
        )

        placeholder = (mx.zeros((1,)), mx.zeros((1,)))
        rotating_real = (mx.ones((1, 8, 4, 64)), mx.ones((1, 8, 4, 64)))
        kv = (mx.ones((1, 8, 4, 64)), mx.ones((1, 8, 4, 64)))

        # Last block placeholder, previous block valid rotating state.
        all_block_data = [
            [kv, rotating_real],
            [kv, placeholder],
        ]
        layer_cache_types = ["KVCache", "RotatingKVCache"]

        result = cache._find_walk_back_truncation_point(
            all_block_data, layer_cache_types
        )
        assert result == 0

    # ------------------------------------------------------------------
    # Full reconstruct_cache integration with walk-back
    # ------------------------------------------------------------------

    def test_reconstruct_arrays_cache_walks_back_to_valid_block(self, mx):
        """Partial match should walk back instead of rejecting entirely.

        3 blocks: block0[p] block1[real] block2[p]
        Should truncate to blocks 0-1, returning valid cache.
        """
        from omlx.cache.paged_ssd_cache import PagedSSDCacheManager

        mock_ssd = MagicMock(spec=PagedSSDCacheManager)

        model = MockModel(num_layers=2)
        paged_cache = PagedCacheManager(
            block_size=4, max_blocks=100, model_name="test-model",
            initial_blocks=100,
        )
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )

        # Allocate 3 blocks
        blocks = []
        for i in range(3):
            b = paged_cache.allocate_block()
            b.block_hash = f"hash{i}".encode()
            b.token_count = 4
            b.ref_count = 2  # Simulate fetch_cache having incremented ref
            blocks.append(b)

        block_table = BlockTable(
            request_id="req-001",
            block_ids=[b.block_id for b in blocks],
            num_tokens=12,
        )

        kv_slice = (mx.ones((1, 8, 4, 64)), mx.ones((1, 8, 4, 64)))
        placeholder = (mx.zeros((1,)), mx.zeros((1,)))
        conv_state = mx.ones((1, 3, 64))
        ssm_state = mx.ones((1, 32, 128, 128))
        real_state = (conv_state, ssm_state)

        metadata = {
            "model_name": "test-model",
            "num_layers": 2,
            "layer_cache_types": ["KVCache", "ArraysCache"],
            "layer_meta_states": [(), ()],
        }

        mock_ssd.load_block_with_metadata.side_effect = [
            ([kv_slice, placeholder], metadata),   # block 0: placeholder
            ([kv_slice, real_state], metadata),     # block 1: real state (turn 1 last)
            ([kv_slice, placeholder], metadata),    # block 2: placeholder
        ]

        result = cache.reconstruct_cache(block_table)

        # Should NOT be None -- walk-back recovered blocks 0-1
        assert result is not None
        assert len(result) == 2  # 2 layers reconstructed

        # block_table should be truncated to 2 blocks
        assert len(block_table.block_ids) == 2
        assert block_table.num_tokens == 8

        # Block 2 ref_count should have been decremented (freed)
        assert blocks[2].ref_count == 1

    def test_reconstruct_rotating_cache_walks_back_to_valid_block(self, mx):
        """Rotating partial match should walk back to latest valid block."""
        from omlx.cache.paged_ssd_cache import PagedSSDCacheManager

        mock_ssd = MagicMock(spec=PagedSSDCacheManager)

        model = MockModel(num_layers=2)
        paged_cache = PagedCacheManager(
            block_size=4, max_blocks=100, model_name="test-model",
            initial_blocks=100,
        )
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )

        # Allocate 3 blocks
        blocks = []
        for i in range(3):
            b = paged_cache.allocate_block()
            b.block_hash = f"hash{i}".encode()
            b.token_count = 4
            b.ref_count = 2  # Simulate fetch_cache having incremented ref
            blocks.append(b)

        block_table = BlockTable(
            request_id="req-001",
            block_ids=[b.block_id for b in blocks],
            num_tokens=12,
        )

        kv_slice = (mx.ones((1, 8, 4, 64)), mx.ones((1, 8, 4, 64)))
        placeholder = (mx.zeros((1,)), mx.zeros((1,)))
        rotating_real = (mx.ones((1, 8, 4, 64)), mx.ones((1, 8, 4, 64)))

        metadata = {
            "model_name": "test-model",
            "num_layers": 2,
            "layer_cache_types": ["KVCache", "RotatingKVCache"],
            "layer_meta_states": [(), (0, 4, 4, 4)],
        }

        mock_ssd.load_block_with_metadata.side_effect = [
            ([kv_slice, placeholder], metadata),     # block 0: placeholder
            ([kv_slice, rotating_real], metadata),   # block 1: real rotating state
            ([kv_slice, placeholder], metadata),     # block 2: placeholder
        ]

        result = cache.reconstruct_cache(block_table)

        # Should recover blocks 0-1 via walk-back.
        assert result is not None
        assert len(result) == 2
        assert len(block_table.block_ids) == 2
        assert block_table.num_tokens == 8
        assert blocks[2].ref_count == 1

        rotating_cache = result[1]
        assert hasattr(rotating_cache, "max_size")
        assert rotating_cache.max_size == 4

    def test_reconstruct_all_placeholders_still_rejects(self, mx):
        """When no block has valid state, walk-back finds nothing and
        the existing per-layer rejection returns None."""
        from omlx.cache.paged_ssd_cache import PagedSSDCacheManager

        mock_ssd = MagicMock(spec=PagedSSDCacheManager)

        model = MockModel(num_layers=2)
        paged_cache = PagedCacheManager(
            block_size=4, max_blocks=100, model_name="test-model",
            initial_blocks=100,
        )
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )

        blocks = []
        for i in range(2):
            b = paged_cache.allocate_block()
            b.block_hash = f"hash{i}".encode()
            b.token_count = 4
            blocks.append(b)

        block_table = BlockTable(
            request_id="req-001",
            block_ids=[b.block_id for b in blocks],
            num_tokens=8,
        )

        kv_slice = (mx.ones((1, 8, 4, 64)), mx.ones((1, 8, 4, 64)))
        placeholder = (mx.zeros((1,)), mx.zeros((1,)))

        metadata = {
            "model_name": "test-model",
            "num_layers": 2,
            "layer_cache_types": ["KVCache", "ArraysCache"],
            "layer_meta_states": [(), ()],
        }

        mock_ssd.load_block_with_metadata.side_effect = [
            ([kv_slice, placeholder], metadata),
            ([kv_slice, placeholder], metadata),
        ]

        result = cache.reconstruct_cache(block_table)

        # Should still return None -- all placeholders, no walk-back target
        assert result is None

    def test_partial_reconstruction_frees_dropped_blocks(self, mx):
        """Blocks dropped during partial reconstruction should have
        their ref_counts decremented."""
        from omlx.cache.paged_ssd_cache import PagedSSDCacheManager

        mock_ssd = MagicMock(spec=PagedSSDCacheManager)

        model = MockModel(num_layers=1)
        paged_cache = PagedCacheManager(
            block_size=4, max_blocks=100, model_name="test-model",
            initial_blocks=100,
        )
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )

        block1 = paged_cache.allocate_block()
        block1.block_hash = b"hash1"
        block1.token_count = 4
        block1.ref_count = 2  # Simulate fetch_cache increment

        block2 = paged_cache.allocate_block()
        block2.block_hash = b"hash2"
        block2.token_count = 4
        block2.ref_count = 2  # Simulate fetch_cache increment

        block_table = BlockTable(
            request_id="req-001",
            block_ids=[block1.block_id, block2.block_id],
            num_tokens=8,
        )

        kv_slice = (mx.ones((1, 8, 4, 64)), mx.ones((1, 8, 4, 64)))

        # First block loads fine, second fails
        mock_ssd.load_block_with_metadata.side_effect = [
            ([kv_slice], {
                "model_name": "test-model",
                "num_layers": 1,
                "layer_cache_types": ["KVCache"],
                "layer_meta_states": [()],
            }),
            (None, None),  # Second block fails to load
        ]

        result = cache.reconstruct_cache(block_table)

        # Should partially reconstruct with block1 only
        assert result is not None
        assert len(block_table.block_ids) == 1
        assert block_table.num_tokens == 4

        # block2 ref_count should have been decremented
        assert block2.ref_count == 1
