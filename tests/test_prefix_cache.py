# SPDX-License-Identifier: Apache-2.0
"""
Tests for BlockAwarePrefixCache and related components.

This module tests the block-aware prefix caching system that uses
PagedCacheManager for block-based storage with SSD persistence.
"""

import logging
import sys
import time
import types
from unittest.mock import MagicMock, patch

import pytest

from omlx.cache.paged_cache import (
    BlockTable,
    PagedCacheManager,
    compute_block_hash,
)
from omlx.cache.prefix_cache import (
    BlockAwarePrefixCache,
    BlockCacheEntry,
    _MRUPartialBlock,
)
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
        delattr(model2, "layers")
        cache2 = BlockAwarePrefixCache(model2, paged_cache)
        assert cache2.expected_num_layers == 24

    def test_get_model_num_layers_with_make_cache(self, paged_cache):
        """Test make_cache() takes priority over model.layers for hybrid models."""
        # Hybrid model: 88 architectural layers but only 48 produce cache
        model = MockModel(num_layers=88)
        model.make_cache = lambda: [MagicMock() for _ in range(48)]
        cache = BlockAwarePrefixCache(model, paged_cache)
        assert cache.expected_num_layers == 48

        # Non-hybrid model: make_cache() returns same count as model.layers
        model2 = MockModel(num_layers=32)
        model2.make_cache = lambda: [MagicMock() for _ in range(32)]
        cache2 = BlockAwarePrefixCache(model2, paged_cache)
        assert cache2.expected_num_layers == 32

    def test_get_model_num_layers_make_cache_exception(self, paged_cache):
        """Test fallback to model.layers when make_cache() raises."""
        model = MockModel(num_layers=88)

        def bad_make_cache():
            raise RuntimeError("model not initialized")

        model.make_cache = bad_make_cache
        cache = BlockAwarePrefixCache(model, paged_cache)
        assert cache.expected_num_layers == 88

    def test_get_model_num_layers_make_cache_empty(self, paged_cache):
        """Test fallback when make_cache() returns empty list."""
        model = MockModel(num_layers=32)
        model.make_cache = lambda: []
        cache = BlockAwarePrefixCache(model, paged_cache)
        assert cache.expected_num_layers == 32

    def test_get_model_num_layers_make_cache_non_list(self, paged_cache):
        """Test fallback when make_cache() returns non-list."""
        model = MockModel(num_layers=32)
        model.make_cache = lambda: None
        cache = BlockAwarePrefixCache(model, paged_cache)
        assert cache.expected_num_layers == 32

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
        assert stats.partial_block_skips == 0
        assert stats.partial_tokens_skipped == 0
        assert stats.block_size == prefix_cache.block_size

    def test_get_stats_dict(self, prefix_cache):
        """Test getting statistics as dictionary."""
        prefix_cache._hits = 10
        prefix_cache._misses = 5

        stats_dict = prefix_cache.get_stats_dict()

        assert "hits" in stats_dict
        assert "misses" in stats_dict
        assert "hit_rate" in stats_dict
        assert "partial_block_skips" in stats_dict
        assert "partial_tokens_skipped" in stats_dict
        assert "last_tokens_to_next_block" in stats_dict
        assert stats_dict["hit_rate"] == pytest.approx(10 / 15)

    def test_reset_stats(self, prefix_cache):
        """Test resetting statistics."""
        prefix_cache._hits = 10
        prefix_cache._misses = 5
        prefix_cache._tokens_saved = 500
        prefix_cache._partial_block_skips = 3
        prefix_cache._partial_tokens_skipped = 42
        prefix_cache._last_partial_tokens_skipped = 2
        prefix_cache._last_tokens_to_next_block = 254

        prefix_cache.reset_stats()

        assert prefix_cache._hits == 0
        assert prefix_cache._misses == 0
        assert prefix_cache._tokens_saved == 0
        assert prefix_cache._partial_block_skips == 0
        assert prefix_cache._partial_tokens_skipped == 0
        assert prefix_cache._last_partial_tokens_skipped == 0
        assert prefix_cache._last_tokens_to_next_block == 0

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

    def test_forget_incompatible_ssd_block_calls_forget(self, prefix_cache_with_ssd):
        """The helper clears local SSD indexes without deleting the file."""
        block_hash = b"\xab" * 32
        prefix_cache_with_ssd._forget_incompatible_ssd_block(block_hash)

        prefix_cache_with_ssd.paged_ssd_cache.forget_block.assert_called_once_with(
            block_hash
        )

    def test_forget_incompatible_ssd_block_noop_when_hash_missing(
        self, prefix_cache_with_ssd
    ):
        """No hash means the block was never persisted to SSD — nothing to
        forget."""
        prefix_cache_with_ssd._forget_incompatible_ssd_block(None)
        prefix_cache_with_ssd.paged_ssd_cache.forget_block.assert_not_called()

    def test_forget_incompatible_ssd_block_noop_without_ssd(self, paged_cache):
        """Pure paged-cache deployments have no SSD manager attached — the
        helper must not raise."""
        model = MockModel(num_layers=4)
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=None,
        )
        # Should not raise.
        cache._forget_incompatible_ssd_block(b"\xab" * 32)

    def test_reconstruct_block_size_mismatch_forgets_without_delete(
        self, prefix_cache_with_ssd
    ):
        """A block from another block_size is rejected without unlinking SSD."""
        block = prefix_cache_with_ssd.paged_cache.allocate_block()
        block.block_hash = b"\xcd" * 32
        block.token_count = 4
        table = BlockTable(
            request_id="req-001",
            block_ids=[block.block_id],
            num_tokens=4,
        )
        prefix_cache_with_ssd.paged_ssd_cache.load_block_with_metadata.return_value = (
            [(MagicMock(), MagicMock()) for _ in range(4)],
            {
                "model_name": "test-model",
                "num_layers": 4,
                "block_size": 2048,
                "layer_cache_types": ["KVCache"] * 4,
            },
        )

        result = prefix_cache_with_ssd.reconstruct_cache(table)

        assert result is None
        prefix_cache_with_ssd.paged_ssd_cache.forget_block.assert_called_once_with(
            block.block_hash
        )
        prefix_cache_with_ssd.paged_ssd_cache.delete_block.assert_not_called()

    def test_minimax_m3_sliceable_nstate_blocks_round_trip(self, monkeypatch):
        """MiniMax M3 single-cache blocks store K/V/index slices, not snapshots."""
        mx = pytest.importorskip("mlx.core")

        module = types.ModuleType("mlx_vlm.models.minimax_m3_vl.language")

        class FakeInnerKVCache:
            def __init__(self):
                self.state = None

        class MiniMaxM3KVCache:
            def __init__(self):
                self.kv_cache = FakeInnerKVCache()
                self.index_keys = None
                self.index_offset = 0

        module.MiniMaxM3KVCache = MiniMaxM3KVCache
        monkeypatch.setitem(
            sys.modules,
            "mlx_vlm.models.minimax_m3_vl.language",
            module,
        )

        paged_cache = PagedCacheManager(
            block_size=4,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        mock_ssd = MagicMock()
        saved_by_hash = {}

        def save_block(**kwargs):
            saved_by_hash[kwargs["block_hash"]] = (
                kwargs["cache_data"],
                {
                    "model_name": "test-model",
                    "num_layers": 1,
                    "block_size": 4,
                    "layer_cache_types": kwargs["layer_cache_types"],
                    "layer_meta_states": kwargs["layer_meta_states"],
                },
            )
            return True

        mock_ssd.save_block.side_effect = save_block
        cache = BlockAwarePrefixCache(
            model=MockModel(num_layers=1),
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )

        keys = mx.arange(1 * 2 * 8 * 3, dtype=mx.float32).reshape(1, 2, 8, 3)
        values = keys + 100
        index_keys = mx.arange(1 * 1 * 8 * 3, dtype=mx.float32).reshape(1, 1, 8, 3)
        mx.eval(keys, values, index_keys)
        cache_data = [
            {
                "state": (keys, values, index_keys),
                "cache_type": "MiniMaxM3KVCache",
                "class_name": "MiniMaxM3KVCache",
                "meta_state": (999,),
            }
        ]

        block_table = cache.store_cache("req-minimax", list(range(8)), cache_data)

        assert block_table is not None
        assert len(block_table.block_ids) == 2
        assert mock_ssd.save_block.call_count == 2
        for idx, call in enumerate(mock_ssd.save_block.call_args_list):
            marker = call.kwargs["cache_data"][0]
            assert marker[0] == "__nstate__"
            assert marker[1] == "MiniMaxM3KVCache"
            saved_keys, saved_values, saved_index = marker[2]
            start = idx * 4
            end = start + 4
            assert saved_keys.tolist() == keys[:, :, start:end, :].tolist()
            assert saved_values.tolist() == values[:, :, start:end, :].tolist()
            assert saved_index.tolist() == index_keys[:, :, start:end, :].tolist()

        def load_block_with_metadata(block_hash, **kwargs):
            return saved_by_hash.get(block_hash, (None, None))

        mock_ssd.load_block_with_metadata.side_effect = load_block_with_metadata
        restored = cache.reconstruct_cache(block_table)

        assert restored is not None
        restored_cache = restored[0]
        restored_keys, restored_values = restored_cache.kv_cache.state
        assert restored_keys.tolist() == keys.tolist()
        assert restored_values.tolist() == values.tolist()
        assert restored_cache.index_keys.tolist() == index_keys.tolist()
        assert restored_cache.index_offset == 8


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
        block_hash = compute_block_hash(b"", tokens, model_name=paged_cache.model_name)
        prefix_cache._prefix_index[block_hash] = (4, block_ids, 1)

        result = prefix_cache._find_best_prefix_match(tokens)

        assert result is not None
        prefix_len, matched_ids, num_blocks, chain_hashes = result
        assert prefix_len == 4
        assert chain_hashes == [block_hash]

    def test_prefix_index_immutable_after_store(self, prefix_cache, paged_cache):
        """Test that _prefix_index entries are not affected by later mutations
        of the original block_ids list (e.g., from CoW or block reallocation).

        Regression test for: storing a mutable list reference in _prefix_index
        allows CoW operations to silently corrupt the index.
        """
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]  # 2 blocks (block_size=4)

        # Allocate blocks
        blocks = paged_cache.get_new_blocks(2)
        block_ids = [b.block_id for b in blocks]
        original_ids = list(block_ids)  # snapshot for assertion

        # Store into prefix index
        prefix_cache._update_prefix_index(tokens, block_ids)

        # Simulate CoW: mutate the original list in-place
        block_ids[0] = 9999

        # Verify: prefix_index must still contain the original block IDs
        result = prefix_cache._find_best_prefix_match(tokens)
        assert result is not None
        _, matched_ids, num_blocks, _ = result
        assert list(matched_ids[:num_blocks]) == original_ids[:num_blocks]


class TestPrefixIndexLifecycle:
    """Tests for the prefix-index / block-hash lifecycle coupling.

    Regression tests for unbounded _prefix_index growth: entries used to be
    dropped only by clear(), so a long-lived server leaked Python heap for
    every distinct stored prefix even after the blocks themselves were freed
    or reused.
    """

    @pytest.fixture
    def paged_cache(self):
        return PagedCacheManager(
            block_size=4,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )

    @pytest.fixture
    def prefix_cache(self, paged_cache):
        return BlockAwarePrefixCache(
            model=MockModel(num_layers=4),
            paged_cache_manager=paged_cache,
        )

    def _indexed_blocks(self, prefix_cache, paged_cache, tokens):
        """Allocate blocks for tokens and index them; returns the blocks."""
        num = len(tokens) // paged_cache.block_size
        blocks = paged_cache.get_new_blocks(num)
        for block in blocks:
            block.token_count = paged_cache.block_size
        prefix_cache._update_prefix_index(
            tokens, [block.block_id for block in blocks]
        )
        return blocks

    def test_lifecycle_hooks_registered(self, prefix_cache, paged_cache):
        assert paged_cache.on_block_hash_dropped is not None
        assert paged_cache.on_hash_map_cleared is not None

    def test_index_entry_dropped_on_block_free(self, prefix_cache, paged_cache):
        blocks = self._indexed_blocks(
            prefix_cache, paged_cache, [1, 2, 3, 4, 5, 6, 7, 8]
        )
        assert len(prefix_cache._prefix_index) == 2

        for block in blocks:
            paged_cache.free_block(block.block_id)

        assert len(prefix_cache._prefix_index) == 0

    def test_index_entry_dropped_on_hash_eviction(self, prefix_cache, paged_cache):
        blocks = self._indexed_blocks(prefix_cache, paged_cache, [1, 2, 3, 4])
        block = blocks[0]
        paged_cache.cached_block_hash_to_block.insert(block.block_hash, block)
        assert len(prefix_cache._prefix_index) == 1

        paged_cache._maybe_evict_cached_block(block)

        assert len(prefix_cache._prefix_index) == 0

    def test_index_survives_while_other_block_shares_hash(
        self, prefix_cache, paged_cache
    ):
        """Hybrid models map the same hash to several blocks (one per KV
        cache group); the index entry must only die with the last one."""
        blocks = self._indexed_blocks(prefix_cache, paged_cache, [1, 2, 3, 4])
        block = blocks[0]
        block_hash = block.block_hash
        other = paged_cache.get_new_blocks(1)[0]
        other.block_hash = block_hash
        paged_cache.cached_block_hash_to_block.insert(block_hash, block)
        paged_cache.cached_block_hash_to_block.insert(block_hash, other)

        paged_cache._maybe_evict_cached_block(block)
        assert block_hash in prefix_cache._prefix_index

        paged_cache._maybe_evict_cached_block(other)
        assert block_hash not in prefix_cache._prefix_index

    def test_index_cleared_on_paged_clear(self, prefix_cache, paged_cache):
        self._indexed_blocks(prefix_cache, paged_cache, [1, 2, 3, 4])
        assert prefix_cache._prefix_index

        paged_cache.clear()

        assert not prefix_cache._prefix_index

    def test_index_entry_dropped_on_forget_incompatible(
        self, prefix_cache, paged_cache
    ):
        """_forget_incompatible_ssd_block pops the hash map directly; the
        prefix index must follow, or it keeps retrying the dead chain."""
        blocks = self._indexed_blocks(prefix_cache, paged_cache, [1, 2, 3, 4])
        block = blocks[0]
        paged_cache.cached_block_hash_to_block.insert(block.block_hash, block)
        assert len(prefix_cache._prefix_index) == 1

        prefix_cache._forget_incompatible_ssd_block(
            block.block_hash, block.block_id
        )

        assert len(prefix_cache._prefix_index) == 0

    def test_update_stops_at_unallocated_block_id(self, prefix_cache, paged_cache):
        """Unallocated block ids must not be indexed: no block owns the
        hash, so the lifecycle hooks could never drop the entry."""
        blocks = paged_cache.get_new_blocks(1)
        blocks[0].token_count = paged_cache.block_size

        prefix_cache._update_prefix_index(
            [1, 2, 3, 4, 5, 6, 7, 8], [blocks[0].block_id, 9999]
        )

        assert len(prefix_cache._prefix_index) == 1


class TestPrefixIndexValidation:
    """Tests for chain re-validation on the prefix-index hit path.

    Index entries can outlive their blocks; reusing a reassigned block would
    splice foreign KV into the request. The hit path must acquire block by
    block, stop at the first hash mismatch, and self-heal the stale entry.
    """

    @pytest.fixture
    def paged_cache(self):
        return PagedCacheManager(
            block_size=4,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )

    @pytest.fixture
    def prefix_cache(self, paged_cache):
        return BlockAwarePrefixCache(
            model=MockModel(num_layers=4),
            paged_cache_manager=paged_cache,
        )

    def _indexed_blocks(self, prefix_cache, paged_cache, tokens):
        num = len(tokens) // paged_cache.block_size
        blocks = paged_cache.get_new_blocks(num)
        for block in blocks:
            block.token_count = paged_cache.block_size
        prefix_cache._update_prefix_index(
            tokens, [block.block_id for block in blocks]
        )
        return blocks

    def test_valid_index_hit_returns_full_prefix(self, prefix_cache, paged_cache):
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        blocks = self._indexed_blocks(prefix_cache, paged_cache, tokens)

        table, remaining = prefix_cache.fetch_cache("req-valid", tokens)

        assert table is not None
        assert table.block_ids == [b.block_id for b in blocks]
        assert table.num_tokens == 8
        assert remaining == []
        assert all(b.ref_count == 2 for b in blocks)

    def test_stale_tail_truncates_to_valid_prefix(self, prefix_cache, paged_cache):
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        blocks = self._indexed_blocks(prefix_cache, paged_cache, tokens)
        stale_entry_key = blocks[1].block_hash

        # Simulate the tail block being reused for other content after the
        # index was written.
        blocks[1].block_hash = b"reassigned"

        table, remaining = prefix_cache.fetch_cache("req-stale", tokens)

        assert table is not None
        assert table.block_ids == [blocks[0].block_id]
        assert table.num_tokens == 4
        assert remaining == tokens[4:]
        # Reference taken only on the validated block.
        assert blocks[0].ref_count == 2
        assert blocks[1].ref_count == 1
        # The dead-chain entry self-healed out of the index.
        assert stale_entry_key not in prefix_cache._prefix_index

    def test_fully_stale_chain_is_a_miss(self, prefix_cache, paged_cache):
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        blocks = self._indexed_blocks(prefix_cache, paged_cache, tokens)
        blocks[0].block_hash = b"reassigned"

        table, remaining = prefix_cache.fetch_cache("req-miss", tokens)

        assert table is None
        assert remaining == tokens
        assert blocks[0].ref_count == 1
        assert blocks[1].ref_count == 1


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
        result = prefix_cache._validate_block_cache_data(cache_data, layer_cache_types)
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

        result = prefix_cache._validate_block_cache_data(cache_data, layer_cache_types)
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
            cache_data,
            0,
            4,
            model_cache_config=None,
            is_last_block=True,
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
            cache_data,
            0,
            4,
            model_cache_config=None,
            is_last_block=False,
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
            cache_data,
            0,
            4,
            model_cache_config=config,
            is_last_block=False,
        )
        assert result is not None
        assert len(result) == 2
        # KVCache layer should be sliced normally
        assert result[0][0].shape[2] == 4  # seq_len slice
        # ArraysCache layer should be placeholder
        assert result[1][0].shape == (1,)

        # Last block
        result = prefix_cache._extract_block_tensor_slice(
            cache_data,
            4,
            8,
            model_cache_config=config,
            is_last_block=True,
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
            block_size=4,
            max_blocks=100,
            model_name="test-model",
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
        block2_data = [
            kv_slice,
            placeholder,
        ]  # ArraysCache = placeholder (still non-last in original)

        mock_ssd.load_block_with_metadata.side_effect = [
            (
                block1_data,
                {
                    "model_name": "test-model",
                    "num_layers": 2,
                    "layer_cache_types": ["KVCache", "ArraysCache"],
                    "layer_meta_states": [(), ()],
                },
            ),
            (
                block2_data,
                {
                    "model_name": "test-model",
                    "num_layers": 2,
                    "layer_cache_types": ["KVCache", "ArraysCache"],
                    "layer_meta_states": [(), ()],
                },
            ),
        ]

        result = cache.reconstruct_cache(block_table)

        # Should return None because ArraysCache layer has placeholder
        assert result is None

    def test_reconstruct_arrays_cache_exact_match_succeeds(self, prefix_cache, mx):
        """Exact match (full state in last block) should reconstruct successfully."""
        from omlx.cache.paged_ssd_cache import PagedSSDCacheManager

        mock_ssd = MagicMock(spec=PagedSSDCacheManager)

        model = MockModel(num_layers=1)
        paged_cache = PagedCacheManager(
            block_size=4,
            max_blocks=100,
            model_name="test-model",
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
            (
                block1_data,
                {
                    "model_name": "test-model",
                    "num_layers": 1,
                    "layer_cache_types": ["ArraysCache"],
                    "layer_meta_states": [()],
                },
            ),
            (
                block2_data,
                {
                    "model_name": "test-model",
                    "num_layers": 1,
                    "layer_cache_types": ["ArraysCache"],
                    "layer_meta_states": [()],
                },
            ),
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
            {
                "state": (mx.ones((1, 8, 10, 64)), mx.ones((1, 8, 10, 64))),
                "cache_type": "KVCache",
                "class_name": "KVCache",
            }
        ]

        result = cache.store_cache("req-001", tokens, cache_data)

        assert result is not None
        # Should create exactly 2 blocks (10 // 4 = 2), not 3
        assert len(result.block_ids) == 2
        # num_tokens should reflect only full blocks
        assert result.num_tokens == 8  # 2 blocks * 4 tokens

        stats = cache.get_stats()
        assert stats.partial_block_skips == 1
        assert stats.partial_tokens_skipped == 2
        assert stats.last_partial_tokens_skipped == 2
        assert stats.last_tokens_to_next_block == 2

    def test_store_cache_arrayscache_partial_trailing_uses_last_full_block_state(
        self, mx
    ):
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
            {
                "state": (mx.ones((1, 8, 3, 64)), mx.ones((1, 8, 3, 64))),
                "cache_type": "KVCache",
                "class_name": "KVCache",
            }
        ]

        result = cache.store_cache("req-001", tokens, cache_data)

        assert result is not None
        assert len(result.block_ids) == 0
        assert result.num_tokens == 0

        stats = cache.get_stats()
        assert stats.partial_block_skips == 1
        assert stats.partial_tokens_skipped == 3
        assert stats.last_partial_tokens_skipped == 3
        assert stats.last_tokens_to_next_block == 1

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
            {
                "state": (mx.ones((1, 8, 8, 64)), mx.ones((1, 8, 8, 64))),
                "cache_type": "KVCache",
                "class_name": "KVCache",
            }
        ]

        result = cache.store_cache("req-001", tokens, cache_data)

        assert result is not None
        assert len(result.block_ids) == 2
        assert result.num_tokens == 8

    def test_fetch_cache_with_segmented_extra_key_ranges(self):
        """Later image changes should preserve reuse before their boundary."""
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

        tokens = list(range(12))
        original_ranges = [
            (5, ("image-1",)),
            (9, ("image-1", "image-2")),
        ]

        stored = cache.store_cache(
            "req-store",
            tokens,
            [],
            extra_key_ranges=original_ranges,
        )
        assert stored is not None
        assert stored.num_tokens == 12

        exact_table, exact_remaining = cache.fetch_cache(
            "req-exact",
            tokens,
            extra_key_ranges=original_ranges,
        )
        assert exact_table is not None
        assert exact_table.num_tokens == 12
        assert exact_remaining == []

        changed_later_image_table, changed_later_image_remaining = cache.fetch_cache(
            "req-later-image",
            tokens,
            extra_key_ranges=[
                (5, ("image-1",)),
                (9, ("image-1", "image-3")),
            ],
        )
        assert changed_later_image_table is not None
        assert changed_later_image_table.num_tokens == 8
        assert changed_later_image_remaining == tokens[8:]

        changed_earlier_image_table, changed_earlier_image_remaining = (
            cache.fetch_cache(
                "req-earlier-image",
                tokens,
                extra_key_ranges=[
                    (5, ("image-x",)),
                    (9, ("image-x", "image-2")),
                ],
            )
        )
        assert changed_earlier_image_table is not None
        assert changed_earlier_image_table.num_tokens == 4
        assert changed_earlier_image_remaining == tokens[4:]

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
        existing_hash = compute_block_hash(
            None, existing_tokens, model_name="test-model"
        )
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

    def test_store_cache_rolls_back_when_ssd_save_fails(self, mx):
        """Failed SSD save should not retain block metadata in paged cache."""
        block_size = 4
        paged_cache = PagedCacheManager(
            block_size=block_size,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        mock_ssd = MagicMock()
        mock_ssd.save_block.return_value = False

        model = MockModel(num_layers=1)
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )

        tokens = [1, 2, 3, 4]  # exactly one full block
        keys = mx.ones((1, 8, 4, 64))
        values = mx.ones((1, 8, 4, 64))
        cache_data = [
            {"state": (keys, values), "cache_type": "KVCache", "class_name": "KVCache"}
        ]

        result = cache.store_cache("req-rollback", tokens, cache_data)

        assert result is not None
        # If persistence fails, block should be rolled back (not indexed/retained).
        assert len(result.block_ids) == 0
        assert result.num_tokens == 0
        assert paged_cache.stats.allocated_blocks == 1  # null block only

        failed_hash = compute_block_hash(None, tokens, model_name="test-model")
        assert paged_cache.cached_block_hash_to_block.get_block(failed_hash) is None

    def test_store_cache_keeps_valid_prefix_when_later_ssd_save_fails(self, mx):
        """A later SSD save failure should roll back only the failed block."""
        block_size = 4
        paged_cache = PagedCacheManager(
            block_size=block_size,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        mock_ssd = MagicMock()
        mock_ssd.save_block.side_effect = [True, False]

        model = MockModel(num_layers=1)
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )

        # Seed request with one already-cached prefix block.
        existing_tokens = [10, 11, 12, 13]
        existing_block = paged_cache.allocate_block()
        assert existing_block is not None
        existing_block.token_count = block_size
        block_table = paged_cache.create_block_table("req-partial-rollback")
        block_table.block_ids.append(existing_block.block_id)
        block_table.num_tokens = block_size
        paged_cache.register_block_hash(existing_block, existing_tokens, None)
        assert existing_block.block_hash is not None

        # Add two more blocks; first save succeeds, second save fails.
        tokens = existing_tokens + [20, 21, 22, 23, 30, 31, 32, 33]
        keys = mx.arange(12, dtype=mx.float32).reshape(1, 1, 12, 1)
        values = (mx.arange(12, dtype=mx.float32) + 100).reshape(1, 1, 12, 1)
        cache_data = [
            {"state": (keys, values), "cache_type": "KVCache", "class_name": "KVCache"}
        ]

        result = cache.store_cache("req-partial-rollback", tokens, cache_data)

        assert result is not None

        first_new_hash = compute_block_hash(
            existing_block.block_hash, tokens[4:8], model_name="test-model"
        )
        failed_hash = compute_block_hash(
            first_new_hash, tokens[8:12], model_name="test-model"
        )

        # Keep existing block + first new block; drop only the failed second new block.
        assert len(result.block_ids) == 2
        assert result.num_tokens == 8
        assert result.block_ids[0] == existing_block.block_id

        first_new_block = paged_cache.cached_block_hash_to_block.get_block(
            first_new_hash
        )
        assert first_new_block is not None
        assert result.block_ids[1] == first_new_block.block_id
        assert result.block_ids == [existing_block.block_id, first_new_block.block_id]

        # save_block should attempt exactly two writes in this scenario.
        calls = mock_ssd.save_block.call_args_list
        assert len(calls) == 2
        attempted_hashes = [call.kwargs["block_hash"] for call in calls]
        assert attempted_hashes == [first_new_hash, failed_hash]
        assert [call.kwargs["token_count"] for call in calls] == [
            block_size,
            block_size,
        ]

        # Verify global-index slices were persisted for both attempted new blocks.
        first_saved_keys, first_saved_values = calls[0].kwargs["cache_data"][0]
        failed_saved_keys, failed_saved_values = calls[1].kwargs["cache_data"][0]
        assert first_saved_keys.tolist() == keys[:, :, 4:8, :].tolist()
        assert first_saved_values.tolist() == values[:, :, 4:8, :].tolist()
        assert failed_saved_keys.tolist() == keys[:, :, 8:12, :].tolist()
        assert failed_saved_values.tolist() == values[:, :, 8:12, :].tolist()

        assert (
            paged_cache.cached_block_hash_to_block.get_block(first_new_hash) is not None
        )
        assert paged_cache.cached_block_hash_to_block.get_block(failed_hash) is None
        # Failed block should be freed, not just removed from hash index.
        allocated_non_null_ids = {
            block.block_id
            for block in paged_cache.allocated_blocks.values()
            if not block.is_null
        }
        assert allocated_non_null_ids == {
            existing_block.block_id,
            first_new_block.block_id,
        }
        assert all(
            b.block_hash != failed_hash for b in paged_cache.allocated_blocks.values()
        )

        # Public contract after partial failure: only valid prefix should be reused.
        expected_partial_ids = [existing_block.block_id, first_new_block.block_id]
        fetched_partial, remaining_partial = cache.fetch_cache(
            "req-partial-rollback-hit", tokens
        )
        assert fetched_partial is not None
        assert fetched_partial.block_ids == expected_partial_ids
        assert fetched_partial.num_tokens == 8
        assert remaining_partial == tokens[8:12]

    def test_store_cache_retry_after_partial_failure_saves_only_missing_tail(self, mx):
        """Retry should preserve valid prefix and only save the missing tail block."""
        block_size = 4
        paged_cache = PagedCacheManager(
            block_size=block_size,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        mock_ssd = MagicMock()
        mock_ssd.save_block.side_effect = [True, False, True]

        model = MockModel(num_layers=1)
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )

        existing_tokens = [10, 11, 12, 13]
        existing_block = paged_cache.allocate_block()
        assert existing_block is not None
        existing_block.token_count = block_size
        block_table = paged_cache.create_block_table("req-retry")
        block_table.block_ids.append(existing_block.block_id)
        block_table.num_tokens = block_size
        paged_cache.register_block_hash(existing_block, existing_tokens, None)
        assert existing_block.block_hash is not None

        tokens = existing_tokens + [20, 21, 22, 23, 30, 31, 32, 33]
        keys = mx.arange(12, dtype=mx.float32).reshape(1, 1, 12, 1)
        values = (mx.arange(12, dtype=mx.float32) + 100).reshape(1, 1, 12, 1)
        cache_data = [
            {"state": (keys, values), "cache_type": "KVCache", "class_name": "KVCache"}
        ]

        first_result = cache.store_cache("req-retry", tokens, cache_data)
        assert first_result is not None

        first_new_hash = compute_block_hash(
            existing_block.block_hash, tokens[4:8], model_name="test-model"
        )
        tail_hash = compute_block_hash(
            first_new_hash, tokens[8:12], model_name="test-model"
        )
        first_new_block = paged_cache.cached_block_hash_to_block.get_block(
            first_new_hash
        )
        assert first_new_block is not None
        retained_prefix_ids = first_result.block_ids.copy()
        assert retained_prefix_ids == [
            existing_block.block_id,
            first_new_block.block_id,
        ]

        retry_result = cache.store_cache("req-retry", tokens, cache_data)
        assert retry_result is not None

        calls = mock_ssd.save_block.call_args_list
        assert len(calls) == 3
        attempted_hashes = [call.kwargs["block_hash"] for call in calls]
        assert attempted_hashes == [first_new_hash, tail_hash, tail_hash]
        assert attempted_hashes.count(first_new_hash) == 1
        assert attempted_hashes.count(tail_hash) == 2
        assert [call.kwargs["token_count"] for call in calls] == [
            block_size,
            block_size,
            block_size,
        ]
        retry_saved_keys, retry_saved_values = calls[2].kwargs["cache_data"][0]
        assert retry_saved_keys.tolist() == keys[:, :, 8:12, :].tolist()
        assert retry_saved_values.tolist() == values[:, :, 8:12, :].tolist()

        assert len(retry_result.block_ids) == 3
        assert retry_result.num_tokens == 12
        assert retry_result.block_ids[:2] == retained_prefix_ids
        assert len(set(retry_result.block_ids)) == 3

        tail_block = paged_cache.cached_block_hash_to_block.get_block(tail_hash)
        assert tail_block is not None
        assert retry_result.block_ids[2] == tail_block.block_id

        # Reconstruct full cache and verify tensor content for retry flow.
        saved_by_hash = {
            existing_block.block_hash: [
                (keys[:, :, 0:4, :], values[:, :, 0:4, :]),
            ],
        }
        for call in calls:
            saved_by_hash[call.kwargs["block_hash"]] = call.kwargs["cache_data"]

        def load_block_with_metadata(block_hash):
            block_data = saved_by_hash.get(block_hash)
            if block_data is None:
                return None, None
            return (
                block_data,
                {
                    "model_name": "test-model",
                    "num_layers": 1,
                    "layer_cache_types": ["KVCache"],
                    "layer_meta_states": [()],
                },
            )

        mock_ssd.load_block_with_metadata.side_effect = load_block_with_metadata
        reconstructed = cache.reconstruct_cache(retry_result)
        assert reconstructed is not None
        assert len(reconstructed) == 1
        layer_cache = reconstructed[0]
        if hasattr(layer_cache, "state"):
            reconstructed_keys, reconstructed_values = layer_cache.state
        elif isinstance(layer_cache, (list, tuple)) and len(layer_cache) == 2:
            reconstructed_keys, reconstructed_values = layer_cache
        else:
            reconstructed_keys, reconstructed_values = (
                layer_cache.keys,
                layer_cache.values,
            )

        assert reconstructed_keys.tolist() == keys.tolist()
        assert reconstructed_values.tolist() == values.tolist()

        # Force prefix-index fallback by removing chain-hash index entries.
        for block_id in retry_result.block_ids:
            block = paged_cache.allocated_blocks.get(block_id)
            assert block is not None
            assert block.block_hash is not None
            paged_cache.cached_block_hash_to_block.pop(block.block_hash, block.block_id)

        # Explicitly prove shared-hash path cannot succeed in this fixture.
        assert paged_cache._paged_ssd_cache_manager is None
        shared_block_ids, _ = paged_cache.find_shared_prefix(tokens)
        assert shared_block_ids == []

        expected_ids = retry_result.block_ids.copy()
        # Public contract via prefix-index fallback: full prefix hit, no remaining tokens.
        fetched_table, remaining = cache.fetch_cache(
            "req-retry-prefix-index-hit", tokens
        )
        assert fetched_table is not None
        assert fetched_table.block_ids == expected_ids
        assert fetched_table.num_tokens == 12
        assert remaining == []


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
                "state": [(sub_keys, sub_values)],  # CacheList sub-states
                "cache_type": "CacheList",
                "class_name": "CacheList",
            },
            {
                "state": [(sub_keys, sub_values)],
                "cache_type": "CacheList",
                "class_name": "CacheList",
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
                "state": (kv_keys, kv_values),
                "cache_type": "KVCache",
                "class_name": "KVCache",
            },
            {
                "state": [(sub_keys, MagicMock())],
                "cache_type": "CacheList",
                "class_name": "CacheList",
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
                "state": [(sub_keys, MagicMock())],
                "cache_type": "CacheList",
                "class_name": "CacheList",
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
                "state": (rot_keys, rot_values),
                "cache_type": "RotatingKVCache",
                "class_name": "RotatingKVCache",
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
                "state": [(sub_keys, sub_values)],
                "cache_type": "CacheList",
                "class_name": "CacheList",
            },
        ]
        config = ModelCacheConfig.from_type_list(["CacheList"])

        result = prefix_cache._extract_block_tensor_slice(
            cache_data,
            0,
            32,
            model_cache_config=config,
            is_last_block=True,
        )

        assert result is not None
        assert len(result) == 1
        # CacheList marker format
        assert result[0][0] == "__cache_list__"
        assert len(result[0][1]) == 1  # One sub-cache
        assert result[0][1][0][0].shape == (1, 8, 32, 64)

    def test_extract_block_tensor_slice_cache_list_non_last_sliceable(
        self, prefix_cache, mx
    ):
        """Test _extract_block_tensor_slice for CacheList with sliceable sub-caches on non-last block.

        When all sub-caches are 4D KVCache tensors, they should be sliced
        per-block instead of using last-block-only placeholder storage.
        """
        from omlx.cache.hybrid_cache import ModelCacheConfig

        sub_keys = mx.zeros((1, 8, 32, 64))
        sub_values = mx.ones((1, 8, 32, 64))

        cache_data = [
            {
                "state": [(sub_keys, sub_values)],
                "cache_type": "CacheList",
                "class_name": "CacheList",
            },
        ]
        config = ModelCacheConfig.from_type_list(["CacheList"])

        result = prefix_cache._extract_block_tensor_slice(
            cache_data,
            0,
            16,
            model_cache_config=config,
            is_last_block=False,
        )

        assert result is not None
        assert len(result) == 1
        # Sliceable sub-caches: per-block sliced data, not placeholder
        assert result[0][0] == "__cache_list__"
        assert len(result[0][1]) == 1
        assert result[0][1][0][0].shape == (1, 8, 16, 64)
        assert result[0][1][0][1].shape == (1, 8, 16, 64)

    def test_extract_block_tensor_slice_cache_list_zero_dim_values(
        self, prefix_cache, mx
    ):
        """Test per-block slicing for CacheList with zero-dim values (DSA indexer)."""
        from omlx.cache.hybrid_cache import ModelCacheConfig

        # GLM-5 style: main attention + indexer with zero head_dim
        sub_keys1 = mx.zeros((1, 1, 64, 512))
        sub_values1 = mx.zeros((1, 1, 64, 64))
        sub_keys2 = mx.zeros((1, 1, 64, 128))
        sub_values2 = mx.zeros((1, 1, 64, 0))  # zero head_dim

        cache_data = [
            {
                "state": [(sub_keys1, sub_values1), (sub_keys2, sub_values2)],
                "cache_type": "CacheList",
                "class_name": "CacheList",
            },
        ]
        config = ModelCacheConfig.from_type_list(["CacheList"])

        result = prefix_cache._extract_block_tensor_slice(
            cache_data,
            0,
            32,
            model_cache_config=config,
            is_last_block=False,
        )

        assert result is not None
        assert result[0][0] == "__cache_list__"
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
            [
                (mx.zeros((1, 8, 32, 64)), mx.zeros((1, 8, 32, 64)))
            ],  # CacheList sub-cache list
            (mx.zeros((1, 8, 32, 64)), mx.zeros((1, 8, 32, 64))),  # Standard KVCache
        ]
        layer_cache_types = ["CacheList", "KVCache"]

        result = prefix_cache._validate_block_cache_data(cache_data, layer_cache_types)
        assert result is True

    def test_validate_block_cache_data_cache_list_placeholder(self, prefix_cache, mx):
        """Test _validate_block_cache_data with CacheList placeholder."""
        cache_data = [
            (
                mx.zeros((1,)),
                mx.zeros((1,)),
            ),  # CacheList placeholder (falls through to tuple check)
            (mx.zeros((1, 8, 32, 64)), mx.zeros((1, 8, 32, 64))),  # Standard KVCache
        ]
        layer_cache_types = ["CacheList", "KVCache"]

        result = prefix_cache._validate_block_cache_data(cache_data, layer_cache_types)
        assert result is True

    def test_find_kv_shape_ref_skips_cache_list(self, prefix_cache, mx):
        """Test _find_kv_shape_ref skips CacheList layers."""
        all_block_data = [
            [
                [
                    (mx.zeros((1, 8, 32, 64)), mx.zeros((1, 8, 32, 64)))
                ],  # CacheList: List[Tuple]
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
            block_size=4,
            max_blocks=100,
            model_name="test-model",
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
            block_size=4,
            max_blocks=100,
            model_name="test-model",
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
            [kv, real_state],  # block 1 (last, valid)
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
            block_size=4,
            max_blocks=100,
            model_name="test-model",
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
            [kv, real_state],  # C (turn 1 last block)
            [kv, placeholder],  # D
            [kv, real_state],  # E (turn 2 last block)
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
            block_size=4,
            max_blocks=100,
            model_name="test-model",
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
            block_size=4,
            max_blocks=100,
            model_name="test-model",
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
            block_size=4,
            max_blocks=100,
            model_name="test-model",
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
            ([kv_slice, placeholder], metadata),  # block 0: placeholder
            ([kv_slice, real_state], metadata),  # block 1: real state (turn 1 last)
            ([kv_slice, placeholder], metadata),  # block 2: placeholder
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
            block_size=4,
            max_blocks=100,
            model_name="test-model",
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
            ([kv_slice, placeholder], metadata),  # block 0: placeholder
            ([kv_slice, rotating_real], metadata),  # block 1: real rotating state
            ([kv_slice, placeholder], metadata),  # block 2: placeholder
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
            block_size=4,
            max_blocks=100,
            model_name="test-model",
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
            block_size=4,
            max_blocks=100,
            model_name="test-model",
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
            (
                [kv_slice],
                {
                    "model_name": "test-model",
                    "num_layers": 1,
                    "layer_cache_types": ["KVCache"],
                    "layer_meta_states": [()],
                },
            ),
            (None, None),  # Second block fails to load
        ]

        result = cache.reconstruct_cache(block_table)

        # Should partially reconstruct with block1 only
        assert result is not None
        assert len(block_table.block_ids) == 1
        assert block_table.num_tokens == 4

        # block2 ref_count should have been decremented
        assert block2.ref_count == 1


class TestTurboQuantFormatMismatchRecovery:
    """Regression tests for TurboQuant/fp16 prefix-chain format mismatches."""

    @pytest.fixture
    def mx(self):
        """Import MLX or skip."""
        try:
            import mlx.core as mx

            return mx
        except ImportError:
            pytest.skip("MLX not available")

    def test_reconstruct_truncates_turboquant_chain_at_fp16_tail(self, mx):
        """A pre-fix fp16 tail after TQ blocks should heal by truncation."""
        from mlx_lm.models.cache import KVCache
        from mlx_vlm.turboquant import TurboQuantKVCache

        from omlx.cache.paged_ssd_cache import PagedSSDCacheManager

        mock_ssd = MagicMock(spec=PagedSSDCacheManager)
        mock_ssd.forget_block.return_value = True

        model = MockModel(num_layers=1)
        paged_cache = PagedCacheManager(
            block_size=4,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )

        blocks = []
        for i in range(2):
            block = paged_cache.allocate_block()
            block.block_hash = f"hash-tq-{i}".encode()
            block.token_count = 4
            block.ref_count = 2
            paged_cache.cached_block_hash_to_block.insert(block.block_hash, block)
            blocks.append(block)

        block_table = BlockTable(
            request_id="req-tq-mixed",
            block_ids=[block.block_id for block in blocks],
            num_tokens=8,
        )

        kv_cache = KVCache()
        kv_cache.update_and_fetch(
            mx.random.normal((1, 2, 4, 32)),
            mx.random.normal((1, 2, 4, 32)),
        )
        tq_cache = TurboQuantKVCache.from_cache(kv_cache, bits=4.0)
        tq_keys, tq_values = tq_cache.state
        tq_block = [("__turboquant_v2__", (tq_keys, tq_values))]

        fp16_tail = [
            (
                mx.random.normal((1, 2, 4, 32)),
                mx.random.normal((1, 2, 4, 32)),
            )
        ]
        tq_metadata = {
            "model_name": "test-model",
            "num_layers": 1,
            "block_size": 4,
            "layer_cache_types": ["TurboQuantKVCache"],
            "layer_meta_states": [tq_cache.meta_state],
        }
        fp16_metadata = {
            "model_name": "test-model",
            "num_layers": 1,
            "block_size": 4,
            "layer_cache_types": ["KVCache"],
            "layer_meta_states": [(4,)],
        }
        mock_ssd.load_block_with_metadata.side_effect = [
            (tq_block, tq_metadata),
            (fp16_tail, fp16_metadata),
        ]

        result = cache.reconstruct_cache(block_table)

        assert result is not None
        assert len(result) == 1
        assert isinstance(result[0], TurboQuantKVCache)
        assert block_table.block_ids == [blocks[0].block_id]
        assert block_table.num_tokens == 4
        assert blocks[1].ref_count == 1
        mock_ssd.forget_block.assert_called_once_with(blocks[1].block_hash)
        assert paged_cache.cached_block_hash_to_block.get_block(
            blocks[1].block_hash
        ) is None

    def test_reconstruct_rejects_stale_first_block_with_manager_signature(self, mx):
        """A live manager signature must make stale block 0 fail its own check."""
        from omlx.cache.paged_ssd_cache import PagedSSDCacheManager

        mock_ssd = MagicMock(spec=PagedSSDCacheManager)
        mock_ssd._expected_layer_cache_types = ["TurboQuantKVCache"]
        mock_ssd.forget_block.return_value = True

        paged_cache = PagedCacheManager(
            block_size=4,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        cache = BlockAwarePrefixCache(
            model=MockModel(num_layers=1),
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )
        block = paged_cache.allocate_block()
        block.block_hash = b"stale-first"
        block.token_count = 4
        block.ref_count = 2
        paged_cache.cached_block_hash_to_block.insert(block.block_hash, block)
        block_table = BlockTable(
            request_id="req-stale-first",
            block_ids=[block.block_id],
            num_tokens=4,
        )

        stale_block = [
            (
                mx.random.normal((1, 2, 4, 32)),
                mx.random.normal((1, 2, 4, 32)),
            )
        ]
        stale_metadata = {
            "model_name": "test-model",
            "num_layers": 1,
            "block_size": 4,
            "layer_cache_types": ["KVCache"],
            "layer_meta_states": [(4,)],
        }
        mock_ssd.load_block_with_metadata.return_value = (
            stale_block,
            stale_metadata,
        )

        result = cache.reconstruct_cache(block_table)

        assert result is None
        assert block_table.block_ids == []
        assert block_table.num_tokens == 0
        mock_ssd.forget_block.assert_called_once_with(block.block_hash)

    def test_reconstruct_accepts_sized_arrays_metadata_with_turboquant(self, mx):
        """SizedArraysCache is a restored ArraysCache wrapper, not a mismatch."""
        from mlx_lm.models.cache import KVCache
        from mlx_vlm.turboquant import TurboQuantKVCache

        from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
        from omlx.cache.type_handlers import SizedArraysCache

        mock_ssd = MagicMock(spec=PagedSSDCacheManager)

        model = MockModel(num_layers=2)
        paged_cache = PagedCacheManager(
            block_size=4,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )

        blocks = []
        for i in range(2):
            block = paged_cache.allocate_block()
            block.block_hash = f"hash-arrays-tq-{i}".encode()
            block.token_count = 4
            block.ref_count = 2
            paged_cache.cached_block_hash_to_block.insert(block.block_hash, block)
            blocks.append(block)

        block_table = BlockTable(
            request_id="req-arrays-sized-tq",
            block_ids=[block.block_id for block in blocks],
            num_tokens=8,
        )

        arrays_state_1 = (
            mx.ones((1, 3, 16)),
            mx.ones((1, 2, 16, 16)),
        )
        arrays_state_2 = (
            mx.ones((1, 3, 16)) * 2,
            mx.ones((1, 2, 16, 16)) * 2,
        )

        kv_cache = KVCache()
        kv_cache.update_and_fetch(
            mx.random.normal((1, 2, 4, 32)),
            mx.random.normal((1, 2, 4, 32)),
        )
        tq_cache = TurboQuantKVCache.from_cache(kv_cache, bits=4.0)
        tq_keys, tq_values = tq_cache.state
        tq_block = ("__turboquant_v2__", (tq_keys, tq_values))

        arrays_metadata = {
            "model_name": "test-model",
            "num_layers": 2,
            "block_size": 4,
            "layer_cache_types": ["ArraysCache", "TurboQuantKVCache"],
            "layer_meta_states": [(), tq_cache.meta_state],
        }
        sized_metadata = {
            "model_name": "test-model",
            "num_layers": 2,
            "block_size": 4,
            "layer_cache_types": ["SizedArraysCache", "TurboQuantKVCache"],
            "layer_meta_states": [(), tq_cache.meta_state],
        }
        mock_ssd.load_block_with_metadata.side_effect = [
            ([arrays_state_1, tq_block], arrays_metadata),
            ([arrays_state_2, tq_block], sized_metadata),
        ]

        result = cache.reconstruct_cache(block_table)

        assert result is not None
        assert len(result) == 2
        assert isinstance(result[0], SizedArraysCache)
        assert isinstance(result[1], TurboQuantKVCache)
        assert block_table.block_ids == [blocks[0].block_id, blocks[1].block_id]
        assert block_table.num_tokens == 8
        mock_ssd.forget_block.assert_not_called()


class TestPerBlockMetaStates:
    """Tests for per-block meta_states in store_cache with boundary snapshots.

    Verifies that blocks stored with boundary snapshots use the snapshot's
    meta_state (correct per-boundary offset) rather than the shared final
    meta_state from _extract_cache_states.
    """

    @pytest.fixture
    def mx(self):
        """Import MLX or skip."""
        try:
            import mlx.core as mx

            return mx
        except ImportError:
            pytest.skip("MLX not available")

    def test_store_cache_uses_snapshot_meta_for_rotating_cache(self, mx):
        """Boundary snapshot meta_state should override shared meta for RotatingKVCache blocks."""
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

        model = MockModel(num_layers=2)
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )

        # 8 tokens = 2 full blocks of 4
        tokens = list(range(8))

        # Simulate a hybrid model: layer 0 = KVCache, layer 1 = RotatingKVCache
        # Final cache state has offset=8 (end of request)
        cache_data = [
            {
                "state": (mx.ones((1, 4, 8, 64)), mx.ones((1, 4, 8, 64))),
                "cache_type": "KVCache",
                "class_name": "KVCache",
                "meta_state": ("8",),
            },
            {
                "state": (mx.ones((1, 1, 4, 256)), mx.ones((1, 1, 4, 256))),
                "cache_type": "RotatingKVCache",
                "class_name": "RotatingKVCache",
                "meta_state": (
                    "0",
                    "4",
                    "8",
                    "4",
                ),  # keep, max_size, offset=8 (final), _idx
            },
        ]
        model_cache_config = ModelCacheConfig.from_type_list(
            ["KVCache", "RotatingKVCache"], model_name="test-model"
        )

        # Boundary snapshot at token 4 (end of block 1) with correct offset=4
        boundary_snapshots = {
            4: [
                {
                    "state": (),
                    "meta_state": (),
                    "class_name": "KVCache",
                    "cache_type": "KVCache",
                },
                {
                    "state": (mx.ones((1, 1, 4, 256)), mx.ones((1, 1, 4, 256))),
                    "meta_state": ("0", "4", "4", "4"),  # offset=4 at boundary
                    "class_name": "RotatingKVCache",
                    "cache_type": "RotatingKVCache",
                },
            ],
        }

        result = cache.store_cache(
            "req-001",
            tokens,
            cache_data,
            model_cache_config=model_cache_config,
            boundary_snapshots=boundary_snapshots,
        )

        assert result is not None
        assert len(result.block_ids) == 2

        # Verify save_block was called twice (one per block)
        assert mock_ssd.save_block.call_count == 2

        # Block 1 (has boundary snapshot): should use snapshot meta for
        # RotatingKVCache layer (offset=4), not shared meta (offset=8)
        block1_call = mock_ssd.save_block.call_args_list[0]
        block1_meta = block1_call.kwargs["layer_meta_states"]
        # RotatingKVCache meta (layer 1): offset should be 4 from snapshot
        assert block1_meta[1] == ("0", "4", "4", "4"), (
            f"Block 1 RotatingKVCache meta should use snapshot offset=4, "
            f"got {block1_meta[1]}"
        )

        # Block 2 (last block, uses main state): should use shared meta
        block2_call = mock_ssd.save_block.call_args_list[1]
        block2_meta = block2_call.kwargs["layer_meta_states"]
        # Last block has no separate boundary snapshot override (boundary at
        # token 8 matches the request end), so it uses the shared meta
        assert block2_meta[1] == ("0", "4", "8", "4"), (
            f"Block 2 RotatingKVCache meta should use shared meta offset=8, "
            f"got {block2_meta[1]}"
        )

    def test_store_cache_kvcache_meta_falls_back_to_shared(self, mx):
        """KVCache layers in boundary snapshots have empty meta, should fall back to shared."""
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

        tokens = list(range(8))
        cache_data = [
            {
                "state": (mx.ones((1, 4, 8, 64)), mx.ones((1, 4, 8, 64))),
                "cache_type": "KVCache",
                "class_name": "KVCache",
                "meta_state": ("8",),
            },
        ]
        model_cache_config = ModelCacheConfig.from_type_list(
            ["KVCache"], model_name="test-model"
        )

        # KVCache layers have empty meta in boundary snapshots
        boundary_snapshots = {
            4: [
                {
                    "state": (),
                    "meta_state": (),
                    "class_name": "KVCache",
                    "cache_type": "KVCache",
                },
            ],
        }

        result = cache.store_cache(
            "req-001",
            tokens,
            cache_data,
            model_cache_config=model_cache_config,
            boundary_snapshots=boundary_snapshots,
        )

        assert result is not None
        assert mock_ssd.save_block.call_count == 2

        # Block 1: KVCache meta should fall back to shared meta (empty snapshot meta)
        block1_call = mock_ssd.save_block.call_args_list[0]
        block1_meta = block1_call.kwargs["layer_meta_states"]
        assert block1_meta[0] == (
            "8",
        ), f"KVCache should fall back to shared meta, got {block1_meta[0]}"

    def test_store_cache_no_snapshot_uses_shared_meta(self, mx):
        """Blocks without boundary snapshots should use shared meta (existing behavior)."""
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

        tokens = list(range(4))
        cache_data = [
            {
                "state": (mx.ones((1, 4, 4, 64)), mx.ones((1, 4, 4, 64))),
                "cache_type": "KVCache",
                "class_name": "KVCache",
                "meta_state": ("4",),
            },
        ]

        # No boundary snapshots
        result = cache.store_cache("req-001", tokens, cache_data)

        assert result is not None
        assert mock_ssd.save_block.call_count == 1

        block_call = mock_ssd.save_block.call_args_list[0]
        block_meta = block_call.kwargs["layer_meta_states"]
        assert block_meta[0] == ("4",)

    def test_store_cache_last_block_with_snapshot_uses_snapshot_meta(self, mx):
        """Last block should also prefer snapshot meta when a boundary snapshot exists."""
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

        # Hybrid model: KVCache + RotatingKVCache (need KVCache so
        # _get_cache_seq_len can determine the full sequence length)
        model = MockModel(num_layers=2)
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )

        # 8 tokens = 2 blocks, but cache_data reflects processing of
        # 11 total tokens (3 trailing partial not stored).
        # Shared meta has offset=11 (final state) for RotatingKVCache.
        tokens = list(range(8))
        cache_data = [
            {
                "state": (mx.ones((1, 4, 11, 64)), mx.ones((1, 4, 11, 64))),
                "cache_type": "KVCache",
                "class_name": "KVCache",
                "meta_state": ("11",),
            },
            {
                "state": (mx.ones((1, 1, 4, 256)), mx.ones((1, 1, 4, 256))),
                "cache_type": "RotatingKVCache",
                "class_name": "RotatingKVCache",
                "meta_state": ("0", "4", "11", "4"),  # offset=11 (final)
            },
        ]
        model_cache_config = ModelCacheConfig.from_type_list(
            ["KVCache", "RotatingKVCache"], model_name="test-model"
        )

        # Both blocks have boundary snapshots with correct per-boundary offsets
        boundary_snapshots = {
            4: [
                {
                    "state": (),
                    "meta_state": (),
                    "class_name": "KVCache",
                    "cache_type": "KVCache",
                },
                {
                    "state": (mx.ones((1, 1, 4, 256)), mx.ones((1, 1, 4, 256))),
                    "meta_state": ("0", "4", "4", "4"),
                    "class_name": "RotatingKVCache",
                    "cache_type": "RotatingKVCache",
                },
            ],
            8: [
                {
                    "state": (),
                    "meta_state": (),
                    "class_name": "KVCache",
                    "cache_type": "KVCache",
                },
                {
                    "state": (mx.ones((1, 1, 4, 256)), mx.ones((1, 1, 4, 256))),
                    "meta_state": ("0", "4", "8", "4"),  # offset=8 at boundary
                    "class_name": "RotatingKVCache",
                    "cache_type": "RotatingKVCache",
                },
            ],
        }

        result = cache.store_cache(
            "req-001",
            tokens,
            cache_data,
            model_cache_config=model_cache_config,
            boundary_snapshots=boundary_snapshots,
        )

        assert result is not None
        assert mock_ssd.save_block.call_count == 2

        # Block 1: RotatingKVCache offset=4 from snapshot
        b1_meta = mock_ssd.save_block.call_args_list[0].kwargs["layer_meta_states"]
        assert b1_meta[1] == ("0", "4", "4", "4")

        # Block 2 (last): RotatingKVCache offset=8 from snapshot,
        # NOT offset=11 from shared meta
        b2_meta = mock_ssd.save_block.call_args_list[1].kwargs["layer_meta_states"]
        assert b2_meta[1] == ("0", "4", "8", "4"), (
            f"Last block should use snapshot offset=8, not shared offset=11, "
            f"got {b2_meta[1]}"
        )


def _get_mru_partial(cache, parent_hash):
    """Test-only accessor for one MRU partial entry by parent_hash key.

    Lives in the test module rather than on ``BlockAwarePrefixCache``
    itself: production code has no consumer that looks up entries by
    arbitrary key (the scheduler only needs the boolean predicate via
    ``has_mru_partial()`` and the dict lookup inside ``apply_mru_partial``).
    Tests use this helper to assert on individual entries without
    coupling to the internal ``_mru_partials`` container shape.
    """
    return cache._mru_partials.get(parent_hash)


def _layer(mx, n_tokens, *, class_name="KVCache", head_dim=4, n_kv_heads=1, fill=1.0):
    """Build a layer-state dict for store_cache.

    ``class_name`` selects the cache type (e.g. ``"KVCache"``,
    ``"RotatingKVCache"``, ``"BatchRotatingKVCache"``).  Both
    ``cache_type`` and ``class_name`` keys are populated with the same
    string — ``store_cache`` consults whichever the layer provides.
    """
    return {
        "state": (
            mx.full((1, n_kv_heads, n_tokens, head_dim), fill),
            mx.full((1, n_kv_heads, n_tokens, head_dim), fill),
        ),
        "cache_type": class_name,
        "class_name": class_name,
    }


def _kv_layer(mx, n_tokens, head_dim=4, n_kv_heads=1, fill=1.0):
    return _layer(
        mx, n_tokens,
        class_name="KVCache",
        head_dim=head_dim, n_kv_heads=n_kv_heads, fill=fill,
    )


def _rotating_layer(mx, n_tokens, head_dim=4, n_kv_heads=1):
    return _layer(
        mx, n_tokens,
        class_name="RotatingKVCache",
        head_dim=head_dim, n_kv_heads=n_kv_heads,
    )


def _make_reconstructed_cache(mx, n_layers, n_tokens, head_dim=4):
    """Build a list of MockKVCache objects matching what reconstruct_cache
    would produce: keys.shape[2] == offset, valid region only."""
    class MockKVCache:
        def __init__(self, k, v, offset):
            self.keys = k
            self.values = v
            self.offset = offset

    return [
        MockKVCache(
            mx.ones((1, 1, n_tokens, head_dim)),
            mx.ones((1, 1, n_tokens, head_dim)),
            n_tokens,
        )
        for _ in range(n_layers)
    ]


def _make_mru_cache(paged_cache, mock_ssd, max_entries=4, num_layers=4):
    """Construct a ``BlockAwarePrefixCache`` with a custom MRU capacity."""
    return BlockAwarePrefixCache(
        model=MockModel(num_layers=num_layers),
        paged_cache_manager=paged_cache,
        paged_ssd_cache_manager=mock_ssd,
        mru_partial_max_entries=max_entries,
    )


def _stash_with_prefix(cache, mx, prefix_marker, tail_token):
    """Stash a partial under a distinct parent_hash for multi-slot tests.

    Builds a prompt whose first 4 tokens are unique to ``prefix_marker``
    (forcing a unique parent block hash) and whose 5th token is the
    partial tail.  Returns ``(block_table, parent_hash)``.
    """
    tokens = [prefix_marker * 10 + i for i in range(4)] + [tail_token]
    cache_data = [_kv_layer(mx, 5) for _ in range(4)]
    block_table = cache.store_cache(f"req-{prefix_marker}", tokens, cache_data)
    parent_hash = cache.paged_cache.allocated_blocks[
        block_table.block_ids[-1]
    ].block_hash
    return block_table, parent_hash


class TestMRUPartialBlockCache:
    """Tests for the MRU partial block cache.

    The MRU is a bounded LRU dict of trailing sub-block tails keyed by
    ``parent_hash``.  It lets exact-repeat requests skip re-prefilling
    those tail tokens, and tolerates interleaving (multi-user / multi-
    conversation workloads) by keeping multiple distinct-prefix entries
    coexistent up to ``mru_partial_max_entries``.

    Threat-model coverage these tests enforce:

    - **Hybrid refusal:** when any layer is non-sliceable (RotatingKVCache,
      ArraysCache, etc.), the stash is suppressed entirely.  Splicing into
      only the sliceable layers would create per-layer offset skew at
      decode time.
    - **Transactional splice:** if any layer's concatenate fails, no layer
      sees a mutated keys/values/offset.  Half-mutated caches are silent
      generation corruption.
    - **Real round-trip:** ``store_cache`` populates entries via the
      production extraction path; ``apply_mru_partial`` then splices.
      Tests do not hand-build ``_MRUPartialBlock`` objects for splice
      cases — that hides the extraction-vs-apply boundary the original
      single-slot branch's tests missed.
    """

    @pytest.fixture
    def mx(self):
        try:
            import mlx.core as mx
            return mx
        except ImportError:
            pytest.skip("MLX not available")

    @pytest.fixture
    def paged_cache(self):
        return PagedCacheManager(
            block_size=4,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )

    @pytest.fixture
    def mock_ssd(self):
        """An SSD manager mock — present, not used.

        The MRU stash gates on ``paged_ssd_cache is not None`` because in
        the no-SSD configuration ``reconstruct_cache`` returns ``None`` and
        ``apply_mru_partial`` is unreachable; stashing then would only
        produce dead memory.  Tests that exercise stash/apply directly
        need an SSD instance present even though the mocked save/load
        paths are not exercised.
        """
        mock = MagicMock()
        mock.save_block.return_value = True
        mock.load_block.return_value = None
        mock.load_block_with_metadata.return_value = (None, None)
        mock.has_block.return_value = False
        return mock

    @pytest.fixture
    def prefix_cache(self, paged_cache, mock_ssd):
        return BlockAwarePrefixCache(
            model=MockModel(num_layers=4),
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )

    # --- initial state ---

    def test_init_state_empty(self, prefix_cache):
        assert not prefix_cache._mru_partials
        assert prefix_cache.has_mru_partial() is False

    # --- stash semantics on uniformly sliceable layers ---

    def test_stash_after_store_with_trailing_tokens(self, prefix_cache, mx):
        """6 tokens, block_size=4 → 1 full block + 2 trailing → stash captured."""
        tokens = [10, 20, 30, 40, 50, 60]
        cache_data = [_kv_layer(mx, 6) for _ in range(4)]

        block_table = prefix_cache.store_cache("req-stash", tokens, cache_data)

        parent_hash = prefix_cache.paged_cache.allocated_blocks[
            block_table.block_ids[-1]
        ].block_hash
        partial = _get_mru_partial(prefix_cache, parent_hash)
        assert partial is not None
        assert partial.tokens == [50, 60]
        assert len(partial.kv_data) == 4
        assert prefix_cache.has_mru_partial() is True

    def test_no_stash_when_block_aligned(self, prefix_cache, mx):
        """Block-aligned tokens leave no trailing partial → no entry written."""
        tokens = [10, 20, 30, 40]
        cache_data = [_kv_layer(mx, 4) for _ in range(4)]

        prefix_cache.store_cache("req-aligned", tokens, cache_data)

        assert not prefix_cache._mru_partials
        assert prefix_cache.has_mru_partial() is False

    def test_same_prefix_store_replaces_entry(self, prefix_cache, mx):
        """Same prefix → same parent_hash → same dict key → replace.

        Two stores with identical prefix tokens but different tails
        collide on the same key (parent_hash chains from identical
        prefix blocks).  The newer tail replaces the older one in the
        single dict entry — that is correct LRU put behavior.
        """
        for tail in (50, 99):
            tokens = [10, 20, 30, 40, tail]
            cache_data = [_kv_layer(mx, 5) for _ in range(4)]
            prefix_cache.store_cache(f"req-{tail}", tokens, cache_data)

        # Exactly one entry; its tokens are the latest tail.
        assert len(prefix_cache._mru_partials) == 1
        partial = next(iter(prefix_cache._mru_partials.values()))
        assert partial.tokens == [99]

    def test_no_eligible_tail_does_not_evict_siblings(
        self, prefix_cache, mx
    ):
        """Behavioral change vs single-slot: a block-aligned store (no
        trailing tail) MUST NOT wipe sibling entries from other prefixes.

        Single-slot mode used to clear the lone slot in this branch.
        Multi-slot mode treats "nothing eligible to stash this time"
        as a local signal — sibling entries for distinct prefixes are
        unrelated and stay.
        """
        # First: stash a partial via prefix A.
        prefix_cache.store_cache(
            "req-a", [10, 20, 30, 40, 50], [_kv_layer(mx, 5) for _ in range(4)]
        )
        assert len(prefix_cache._mru_partials) == 1
        before_key = next(iter(prefix_cache._mru_partials.keys()))

        # Second: block-aligned store on a DIFFERENT prefix — no tail to
        # stash, but must not evict the existing sibling.
        prefix_cache.store_cache(
            "req-b", [11, 22, 33, 44], [_kv_layer(mx, 4) for _ in range(4)]
        )
        assert len(prefix_cache._mru_partials) == 1
        assert next(iter(prefix_cache._mru_partials.keys())) == before_key

    def test_stash_records_parent_hash_from_last_block(self, prefix_cache, mx):
        """Stashed entry is keyed by the hash of the last full block."""
        tokens = [10, 20, 30, 40, 50, 60]
        cache_data = [_kv_layer(mx, 6) for _ in range(4)]

        block_table = prefix_cache.store_cache("req-hash", tokens, cache_data)

        # The last (and only) block's hash should be the dict key AND
        # the partial's stored parent_hash.
        last_block = prefix_cache.paged_cache.allocated_blocks[
            block_table.block_ids[-1]
        ]
        assert last_block.block_hash is not None
        partial = _get_mru_partial(prefix_cache, last_block.block_hash)
        assert partial is not None
        assert partial.parent_hash == last_block.block_hash

    # --- threat model: hybrid refusal (B1, B2) ---

    def test_refuse_stash_when_any_layer_non_sliceable_hybrid(
        self, paged_cache, mock_ssd, mx
    ):
        """Hybrid model (KVCache + RotatingKVCache): no stash.

        Splicing into only the sliceable layers produces per-layer offset
        skew at decode time (review B2). The only correct behavior is to
        refuse the partial entirely for hybrid models.
        """
        from omlx.cache.hybrid_cache import ModelCacheConfig

        cache = BlockAwarePrefixCache(
            model=MockModel(num_layers=2),
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )
        tokens = [10, 20, 30, 40, 50, 60]
        cache_data = [
            _kv_layer(mx, 6),
            _rotating_layer(mx, 6),
        ]
        config = ModelCacheConfig.from_type_list(
            ["KVCache", "RotatingKVCache"], model_name="test"
        )

        cache.store_cache("req-hybrid", tokens, cache_data, model_cache_config=config)

        assert not cache._mru_partials
        assert cache.has_mru_partial() is False

    def test_refuse_stash_when_all_layers_non_sliceable(self, prefix_cache, mx):
        """Pure RotatingKVCache model also refuses stash."""
        from omlx.cache.hybrid_cache import ModelCacheConfig

        tokens = [10, 20, 30, 40, 50]
        cache_data = [_rotating_layer(mx, 5) for _ in range(4)]
        config = ModelCacheConfig.from_type_list(
            ["RotatingKVCache"] * 4, model_name="test"
        )

        prefix_cache.store_cache(
            "req-rotating", tokens, cache_data, model_cache_config=config
        )

        assert not prefix_cache._mru_partials

    def test_refuse_stash_when_layer_falls_through_to_default_handler(
        self, prefix_cache, mx
    ):
        """Non-sliceable types whose handler is unregistered (fall through
        to ``DefaultCacheHandler``, which inherits ``KVCacheHandler``'s
        ``supports_block_slicing=True``) must still be refused.

        Concrete case: ``BatchRotatingKVCache`` is mapped in
        ``_class_name_map`` to ``BATCH_ROTATING_KVCACHE`` but no handler
        is registered for that enum.  The original rewrite's
        registry-based gate would have classified it as sliceable,
        recreating exactly the silent-corruption hazard the rewrite was
        supposed to close, just from a different angle.  The fix uses an
        explicit class-name whitelist (``KNOWN_SLICEABLE_CACHE_TYPES``)
        instead of the registry.
        """
        from omlx.cache.hybrid_cache import ModelCacheConfig
        from omlx.cache.type_registry import (
            KNOWN_SLICEABLE_CACHE_TYPES,
            CacheTypeRegistry,
        )

        # Sanity: the registry would lie about this class name.
        handler = CacheTypeRegistry.get_handler_by_class_name("BatchRotatingKVCache")
        assert handler.supports_block_slicing is True
        # And the whitelist correctly excludes it.
        assert "BatchRotatingKVCache" not in KNOWN_SLICEABLE_CACHE_TYPES

        tokens = [10, 20, 30, 40, 50, 60]
        # cache_data shape doesn't matter — store_cache must refuse before
        # any extraction is attempted.
        cache_data = [_kv_layer(mx, 6) for _ in range(4)]
        config = ModelCacheConfig.from_type_list(
            ["BatchRotatingKVCache"] * 4, model_name="test"
        )

        prefix_cache.store_cache(
            "req-batch-rotating", tokens, cache_data, model_cache_config=config
        )

        assert not prefix_cache._mru_partials
        assert prefix_cache.has_mru_partial() is False

    # --- threat model: stale-slot eviction at clear() (C2) ---

    def test_clear_wipes_mru_partials(self, prefix_cache, mx):
        """``BlockAwarePrefixCache.clear()`` must drop the entire MRU dict.

        The scheduler's cache-corruption recovery routes through
        ``clear()``.  Surviving partials chain from paged-block hashes
        whose backing blocks were just freed; the dict is wiped so no
        entry can survive into the recovery path that exists because
        something was wrong.
        """
        prefix_cache.store_cache(
            "req-clear",
            [10, 20, 30, 40, 50, 60],
            [_kv_layer(mx, 6) for _ in range(4)],
        )
        assert bool(prefix_cache._mru_partials)

        prefix_cache.clear()

        assert not prefix_cache._mru_partials
        assert prefix_cache.has_mru_partial() is False

    # --- threat model: H2 ambiguous cache layout ---

    def test_refuse_stash_on_ambiguous_cache_layout(
        self, prefix_cache, mx
    ):
        """Cache lengths that don't unambiguously map to global or local
        indexing must refuse the stash.

        Multi-turn requests can produce ``cache_seq_len ==
        existing_tokens`` or shapes between local and global.  The
        previous heuristic (``cache_seq_len >= existing_tokens + 1``)
        silently picked "local" on the boundary, slicing local indices
        out of a global-indexed cache and capturing tokens from the
        prefix instead of the trailing tail.  parent_hash still matched,
        and a future apply spliced wrong KV — silent generation
        corruption.

        Drive that boundary directly: cache_seq_len falls strictly
        between global_end and local_len.
        """
        # First turn: cache 4 tokens.
        prefix_cache.store_cache(
            "req-turn-1",
            [1, 2, 3, 4],
            [_kv_layer(mx, 4) for _ in range(4)],
        )

        # Second turn: 8 prefix-aligned tokens (1 full block + 1 partial-block).
        # Hand a cache_data whose cache_seq_len is 6 — strictly between:
        #   - local_len = len(new_tokens) = len(tokens) - existing_tokens = 4
        #   - global_end = existing_tokens + new_count = 4 + 4 = 8
        # global_end (8) > cache_seq_len (6) > local_len (4): ambiguous.
        full_tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        cache_data = [_kv_layer(mx, 6) for _ in range(4)]

        prefix_cache.store_cache(
            "req-turn-2-ambiguous", full_tokens, cache_data
        )

        # Refuse rather than guess.  Stash must be the previous turn's
        # state cleared (block-aligned turn 1 has no stash anyway), not
        # a guessed-wrong turn 2.
        assert not prefix_cache._mru_partials

    # --- accounting invariant ---

    def test_kv_data_holds_mlx_arrays_for_active_memory_accounting(
        self, prefix_cache, mx
    ):
        """The MRU slot's memory must flow through ``mx.get_active_memory()``.

        The codebase enforces all KV-memory limits via ``mx.get_active_memory()``
        (process_memory_enforcer, the three scheduler memory checkpoints,
        the periodic-clear threshold, telemetry).  The MRU slot has no
        separate accounting hook — it relies on the invariant that
        ``kv_data`` holds real ``mx.array`` allocations, which MLX counts
        in active memory automatically.

        A "helpful" future change that stored CPU-side copies (e.g.
        ``np.ndarray`` to dodge a perceived GPU-memory cost) would silently
        escape every existing memory limit and only manifest as system OOM
        under load.  Pin the invariant so that change is caught at test
        time, not in production.
        """
        block_table = prefix_cache.store_cache(
            "req-accounting",
            [10, 20, 30, 40, 50, 60],
            [_kv_layer(mx, 6) for _ in range(4)],
        )

        parent_hash = prefix_cache.paged_cache.allocated_blocks[
            block_table.block_ids[-1]
        ].block_hash
        partial = _get_mru_partial(prefix_cache, parent_hash)
        assert partial is not None
        assert len(partial.kv_data) == 4
        for layer_idx, (keys, values) in enumerate(partial.kv_data):
            assert isinstance(keys, mx.array), (
                f"layer {layer_idx} keys is {type(keys).__name__}, not mx.array. "
                f"MRU memory accounting depends on mx.array storage so the "
                f"slot is visible to mx.get_active_memory()."
            )
            assert isinstance(values, mx.array), (
                f"layer {layer_idx} values is {type(values).__name__}, "
                f"not mx.array. See above."
            )

    # --- threat model: no-reconstruct-path config ---

    def test_no_stash_when_paged_ssd_cache_is_none(self, paged_cache, mx):
        """Without a ``PagedSSDCacheManager`` instance, ``reconstruct_cache``
        returns ``None`` (``_can_reconstruct() is False``) and
        ``apply_mru_partial`` is unreachable from the scheduler.  Stashing
        in this configuration would only produce dead memory.

        Note: this is distinct from ``hot_cache_only=True``, where the
        manager IS present (the disk writer thread is what's disabled,
        not the manager itself).  In that mode the MRU stash IS expected
        to populate — ``load_block_with_metadata`` short-circuits to the
        hot tier and reconstruct still works.  The gate keys on manager
        presence, not on whether SSD writes are happening.
        """
        cache = BlockAwarePrefixCache(
            model=MockModel(num_layers=4),
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=None,
        )

        cache.store_cache(
            "req-no-ssd",
            [10, 20, 30, 40, 50, 60],
            [_kv_layer(mx, 6) for _ in range(4)],
        )

        assert not cache._mru_partials
        assert cache.has_mru_partial() is False

    def test_can_reconstruct_helper_reflects_manager_presence(
        self, paged_cache, mock_ssd
    ):
        """``_can_reconstruct`` is the canonical predicate keeping the
        MRU stash gate and the ``reconstruct_cache`` guard in lockstep.

        It returns False only when no manager is configured at all.
        ``hot_cache_only=True`` configurations (manager present, disk
        writer disabled) return True because reconstruct still works
        via the hot-tier short-circuit in
        ``PagedSSDCacheManager.load_block_with_metadata``.
        """
        cache_with = BlockAwarePrefixCache(
            model=MockModel(num_layers=2),
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )
        assert cache_with._can_reconstruct() is True

        cache_without = BlockAwarePrefixCache(
            model=MockModel(num_layers=2),
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=None,
        )
        assert cache_without._can_reconstruct() is False

    # --- apply: real round-trip ---

    def test_apply_round_trip_exact_match(self, prefix_cache, mx):
        """Real store → apply round-trip: partial produced by extraction
        is consumed by the splice path, no hand-built _MRUPartialBlock."""
        tokens = [10, 20, 30, 40, 50, 60]
        cache_data = [_kv_layer(mx, 6) for _ in range(4)]
        block_table = prefix_cache.store_cache("req-rt", tokens, cache_data)

        # Reconstructed cache: 4 layers × 4 tokens (the prefix only).
        reconstructed = _make_reconstructed_cache(mx, n_layers=4, n_tokens=4)
        remaining = [50, 60]

        result, new_remaining, applied = prefix_cache.apply_mru_partial(
            reconstructed, block_table, remaining,
        )

        assert applied == 2
        assert new_remaining == []
        assert all(layer.offset == 6 for layer in result)
        assert all(layer.keys.shape[2] == 6 for layer in result)
        assert all(layer.values.shape[2] == 6 for layer in result)

    def test_apply_round_trip_prefix_match_leaves_extra_tokens(
        self, prefix_cache, mx
    ):
        """When remaining is longer than the partial, the partial covers
        its prefix and the rest is left for normal prefill."""
        tokens = [10, 20, 30, 40, 50, 60]
        cache_data = [_kv_layer(mx, 6) for _ in range(4)]
        block_table = prefix_cache.store_cache("req-rt-prefix", tokens, cache_data)

        reconstructed = _make_reconstructed_cache(mx, n_layers=4, n_tokens=4)
        remaining = [50, 60, 70, 80]  # partial is [50, 60]; [70, 80] left over

        _, new_remaining, applied = prefix_cache.apply_mru_partial(
            reconstructed, block_table, remaining,
        )

        assert applied == 2
        assert new_remaining == [70, 80]

    # --- apply: eviction reasons ---

    def test_apply_noop_on_parent_hash_mismatch_preserves_sibling(
        self, prefix_cache, mx, paged_cache
    ):
        """A request keyed by a parent_hash that isn't in the dict
        returns no-op WITHOUT evicting unrelated sibling entries.

        Behavioral change from single-slot: the single slot used to be
        evicted whenever the lookup key didn't match.  In multi-slot,
        the lookup simply misses and other entries are preserved.
        """
        # Stash a partial under prefix A.
        tokens = [10, 20, 30, 40, 50, 60]
        block_table_a = prefix_cache.store_cache(
            "req-a", tokens, [_kv_layer(mx, 6) for _ in range(4)]
        )
        before = dict(prefix_cache._mru_partials)
        assert len(before) == 1

        # Construct a synthetic block_table pointing at a block whose
        # hash is NOT a key in the dict (simulate "request for a
        # different prefix that has its own paged block").
        other_block = paged_cache.allocate_block()
        other_block.block_hash = b"\x00" * 32  # not in the MRU dict
        synthetic_bt = BlockTable(request_id="req-other")
        synthetic_bt.block_ids.append(other_block.block_id)
        synthetic_bt.num_tokens = 4

        reconstructed = _make_reconstructed_cache(mx, n_layers=4, n_tokens=4)
        _, new_remaining, applied = prefix_cache.apply_mru_partial(
            reconstructed, synthetic_bt, [50, 60],
        )

        assert applied == 0
        assert new_remaining == [50, 60]
        # Prefix A's entry must still be present — no false eviction.
        assert dict(prefix_cache._mru_partials) == before

    def test_apply_evicts_on_token_mismatch(self, prefix_cache, mx):
        """Different trailing tokens → partial cannot apply, evict."""
        tokens = [10, 20, 30, 40, 50, 60]
        cache_data = [_kv_layer(mx, 6) for _ in range(4)]
        block_table = prefix_cache.store_cache("req-evict-t", tokens, cache_data)

        reconstructed = _make_reconstructed_cache(mx, n_layers=4, n_tokens=4)
        _, new_remaining, applied = prefix_cache.apply_mru_partial(
            reconstructed, block_table, [99, 60],  # first token doesn't match
        )

        assert applied == 0
        assert new_remaining == [99, 60]
        assert not prefix_cache._mru_partials

    def test_apply_evicts_on_remaining_shorter_than_partial(
        self, prefix_cache, mx
    ):
        """If remaining_tokens is shorter than the partial it cannot match."""
        tokens = [10, 20, 30, 40, 50, 60, 70]
        cache_data = [_kv_layer(mx, 7) for _ in range(4)]
        block_table = prefix_cache.store_cache("req-evict-s", tokens, cache_data)
        # Partial is [50, 60, 70]; remaining is shorter → must evict.

        reconstructed = _make_reconstructed_cache(mx, n_layers=4, n_tokens=4)
        _, new_remaining, applied = prefix_cache.apply_mru_partial(
            reconstructed, block_table, [50, 60],
        )

        assert applied == 0
        assert not prefix_cache._mru_partials

    def test_apply_evicts_on_layer_count_mismatch(self, prefix_cache, mx):
        """If the reconstructed cache layer count differs from the
        stashed partial, evict — likely a model swap or bug, not safe."""
        tokens = [10, 20, 30, 40, 50, 60]
        cache_data = [_kv_layer(mx, 6) for _ in range(4)]
        block_table = prefix_cache.store_cache("req-evict-lc", tokens, cache_data)

        # Reconstructed has only 2 layers, partial has 4 → mismatch.
        reconstructed = _make_reconstructed_cache(mx, n_layers=2, n_tokens=4)
        _, _, applied = prefix_cache.apply_mru_partial(
            reconstructed, block_table, [50, 60],
        )

        assert applied == 0
        assert not prefix_cache._mru_partials

    def test_apply_noop_when_no_stash(self, prefix_cache, paged_cache, mx):
        """No partial → no-op, remaining unchanged."""
        block_table = paged_cache.create_block_table("req-noop")
        reconstructed = _make_reconstructed_cache(mx, n_layers=4, n_tokens=0)

        result, remaining, applied = prefix_cache.apply_mru_partial(
            reconstructed, block_table, [10, 20],
        )

        assert applied == 0
        assert remaining == [10, 20]
        assert result is reconstructed

    def test_apply_noop_when_remaining_empty(self, prefix_cache, mx):
        """Empty remaining → exact prefix hit already; no MRU work."""
        tokens = [10, 20, 30, 40, 50, 60]
        cache_data = [_kv_layer(mx, 6) for _ in range(4)]
        block_table = prefix_cache.store_cache("req-noop-empty", tokens, cache_data)

        reconstructed = _make_reconstructed_cache(mx, n_layers=4, n_tokens=4)
        _, _, applied = prefix_cache.apply_mru_partial(
            reconstructed, block_table, [],
        )

        assert applied == 0
        # Stash must NOT be evicted on empty-remaining no-op — it could
        # still match a *future* request that does have tail tokens.
        assert bool(prefix_cache._mru_partials)

    # --- threat model: transactional splice rollback (B3) ---

    def test_splice_failure_does_not_mutate_any_layer(
        self, prefix_cache, mx
    ):
        """If any layer's concatenate fails, NO layer is mutated.

        Review B3: the original implementation's try/except wrapped the
        whole loop, so failure on layer N>0 left layers 0..N-1 mutated
        with cache.offset += n_partial while the caller was told nothing
        was applied. The rewrite must build replacements first and commit
        atomically.
        """
        tokens = [10, 20, 30, 40, 50, 60]
        cache_data = [_kv_layer(mx, 6) for _ in range(4)]
        block_table = prefix_cache.store_cache("req-rollback", tokens, cache_data)

        reconstructed = _make_reconstructed_cache(mx, n_layers=4, n_tokens=4)
        # Snapshot the pre-splice state of every layer for an after-comparison.
        before_offsets = [layer.offset for layer in reconstructed]
        before_key_shapes = [layer.keys.shape for layer in reconstructed]

        # Make mx.concatenate explode on the third call (layer 1's keys).
        # Calls go: layer0 keys, layer0 values, layer1 keys (boom).
        from omlx.cache import prefix_cache as pc_mod
        real_concatenate = pc_mod.mx.concatenate
        call_count = {"n": 0}

        def flaky_concatenate(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 3:
                raise RuntimeError("synthetic concatenate failure")
            return real_concatenate(*args, **kwargs)

        with patch.object(pc_mod.mx, "concatenate", side_effect=flaky_concatenate):
            _, new_remaining, applied = prefix_cache.apply_mru_partial(
                reconstructed, block_table, [50, 60],
            )

        assert applied == 0
        assert new_remaining == [50, 60]
        # No layer's offset advanced.
        assert [layer.offset for layer in reconstructed] == before_offsets
        # No layer's keys shape changed.
        assert [layer.keys.shape for layer in reconstructed] == before_key_shapes
        # Slot is evicted on splice failure (don't retry a failing partial).
        assert not prefix_cache._mru_partials

    # --- threat model: multi-turn (existing_tokens > 0) ---

    def test_stash_correct_indices_when_existing_tokens_present(
        self, prefix_cache, mx
    ):
        """When store_cache is called with existing_tokens > 0 (multi-turn),
        the stash slices the partial from the correct cache region.

        cache_data is full-sequence (system prompt + new turn), so the
        partial extraction must use global indices, not relative ones.
        """
        # Pretend a previous turn already cached 4 tokens.
        prev_tokens = [1, 2, 3, 4]
        prev_cache = [_kv_layer(mx, 4) for _ in range(4)]
        prefix_cache.store_cache("req-turn-1", prev_tokens, prev_cache)

        # Second turn: 4 prev + 4 new = 8 prefix block tokens, then 2 trailing.
        # Distinct fill values let us verify the stash sliced the *right*
        # tokens — the partial should contain the trailing region's data
        # (fill=2.0), not the prefix region (fill=1.0).
        full_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        full_cache = []
        for _ in range(4):
            # First 8 positions = old (1.0), last 2 positions = new (2.0)
            keys = mx.concatenate(
                [
                    mx.full((1, 1, 8, 4), 1.0),
                    mx.full((1, 1, 2, 4), 2.0),
                ],
                axis=2,
            )
            values = mx.concatenate(
                [
                    mx.full((1, 1, 8, 4), 1.0),
                    mx.full((1, 1, 2, 4), 2.0),
                ],
                axis=2,
            )
            full_cache.append({
                "state": (keys, values),
                "cache_type": "KVCache",
                "class_name": "KVCache",
            })

        block_table = prefix_cache.store_cache(
            "req-turn-2", full_tokens, full_cache
        )

        parent_hash = prefix_cache.paged_cache.allocated_blocks[
            block_table.block_ids[-1]
        ].block_hash
        partial = _get_mru_partial(prefix_cache, parent_hash)
        assert partial is not None
        assert partial.tokens == [9, 10]
        # Each layer's stashed slice must be the trailing region (fill=2.0).
        for keys, values in partial.kv_data:
            assert keys.shape[2] == 2
            assert mx.allclose(keys, mx.full((1, 1, 2, 4), 2.0))
            assert mx.allclose(values, mx.full((1, 1, 2, 4), 2.0))


class TestMRUPartialMultiSlot:
    """Multi-slot LRU semantics: coexistence, capacity, eviction discipline.

    These tests cover the mechanics that single-slot mode could not
    exercise — multiple entries keyed by distinct ``parent_hash`` values,
    LRU promotion on apply success, sibling preservation on apply miss,
    capacity-bounded eviction, ``max_entries=0`` feature disable, and
    the freed-paged-block guard introduced by the multi-slot design.
    """

    @pytest.fixture
    def mx(self):
        try:
            import mlx.core as mx
            return mx
        except ImportError:
            pytest.skip("MLX not available")

    @pytest.fixture
    def paged_cache(self):
        return PagedCacheManager(
            block_size=4,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )

    @pytest.fixture
    def mock_ssd(self):
        mock = MagicMock()
        mock.save_block.return_value = True
        mock.load_block.return_value = None
        mock.load_block_with_metadata.return_value = (None, None)
        mock.has_block.return_value = False
        return mock

    # --- multi-entry coexistence ---

    def test_distinct_prefixes_coexist_as_separate_entries(
        self, paged_cache, mock_ssd, mx
    ):
        """Two stashes with different parent prefixes produce two entries."""
        cache = _make_mru_cache(paged_cache, mock_ssd, max_entries=4)
        _, hash_a = _stash_with_prefix(cache, mx, prefix_marker=1, tail_token=99)
        _, hash_b = _stash_with_prefix(cache, mx, prefix_marker=2, tail_token=88)

        assert hash_a != hash_b
        assert len(cache._mru_partials) == 2
        assert hash_a in cache._mru_partials
        assert hash_b in cache._mru_partials

    # --- LRU mechanics (parameterized) ---

    @pytest.mark.parametrize(
        "scenario",
        [
            # (capacity, n_stashes_in_order, expected_dict_keys_in_order)
            # Capacity respected; oldest evicted on overflow.
            ("evict_oldest_at_capacity", 2, [1, 2, 3], [2, 3]),
            # Below capacity, all retained, insertion order preserved.
            ("under_capacity_keeps_all", 4, [1, 2, 3], [1, 2, 3]),
        ],
        ids=lambda s: s[0] if isinstance(s, tuple) else str(s),
    )
    def test_lru_capacity_bounds(
        self, paged_cache, mock_ssd, mx, scenario
    ):
        _, capacity, order, expected = scenario
        cache = _make_mru_cache(paged_cache, mock_ssd, max_entries=capacity)
        hashes = {}
        for marker in order:
            _, h = _stash_with_prefix(
                cache, mx, prefix_marker=marker, tail_token=900 + marker
            )
            hashes[marker] = h

        expected_keys = [hashes[m] for m in expected]
        assert list(cache._mru_partials.keys()) == expected_keys

    def test_apply_success_promotes_entry_to_lru_tail(
        self, paged_cache, mock_ssd, mx
    ):
        """Applying an entry moves it to the LRU tail; a subsequent
        capacity-eviction drops a now-older sibling, not the just-used
        entry."""
        cache = _make_mru_cache(paged_cache, mock_ssd, max_entries=2)
        bt_a, hash_a = _stash_with_prefix(
            cache, mx, prefix_marker=1, tail_token=901
        )
        _, hash_b = _stash_with_prefix(
            cache, mx, prefix_marker=2, tail_token=902
        )
        assert list(cache._mru_partials.keys()) == [hash_a, hash_b]

        # Apply A → A promoted to tail.
        reconstructed = _make_reconstructed_cache(mx, n_layers=4, n_tokens=4)
        # Remaining must equal A's stashed tokens.  Tail token was 901,
        # placed at index 4 of A's prompt.
        _, _, applied = cache.apply_mru_partial(reconstructed, bt_a, [901])
        assert applied == 1
        assert list(cache._mru_partials.keys()) == [hash_b, hash_a]

        # Stash C at capacity 2 → B evicted (oldest after promote), A kept.
        _, hash_c = _stash_with_prefix(
            cache, mx, prefix_marker=3, tail_token=903
        )
        assert list(cache._mru_partials.keys()) == [hash_a, hash_c]
        assert hash_b not in cache._mru_partials

    # --- max_entries=0 disables ---

    def test_max_entries_zero_disables_stashing(
        self, paged_cache, mock_ssd, mx
    ):
        cache = _make_mru_cache(paged_cache, mock_ssd, max_entries=0)
        _stash_with_prefix(cache, mx, prefix_marker=1, tail_token=99)

        assert len(cache._mru_partials) == 0
        assert cache.has_mru_partial() is False

    # --- clear_mru_partials() leaves siblings alone ---

    def test_clear_mru_partials_wipes_only_partials(
        self, paged_cache, mock_ssd, mx
    ):
        """``clear_mru_partials()`` is the admin-clear hook.  It wipes the
        MRU dict but must not touch ``paged_cache``, the prefix index,
        or stats — those have their own clear paths."""
        cache = _make_mru_cache(paged_cache, mock_ssd, max_entries=4)
        bt, _ = _stash_with_prefix(cache, mx, prefix_marker=1, tail_token=99)

        prefix_index_before = dict(cache._prefix_index)
        request_tables_before = dict(cache._request_tables)
        assert len(cache._mru_partials) == 1
        assert prefix_index_before  # was populated by store_cache

        n_wiped = cache.clear_mru_partials()

        assert n_wiped == 1
        assert len(cache._mru_partials) == 0
        # Paged blocks, prefix index, request tables all unchanged.
        assert cache._prefix_index == prefix_index_before
        assert cache._request_tables == request_tables_before
        assert bt.block_ids[-1] in cache.paged_cache.allocated_blocks

    # --- freed-block guard (new in multi-slot) ---

    def test_apply_noop_when_parent_block_freed(
        self, paged_cache, mock_ssd, mx
    ):
        """If the parent paged block is freed between stash and apply,
        the apply path must not fall through to a None-keyed lookup
        (which could falsely match a short-prompt entry).

        This race is new in multi-slot: single-slot tolerated it because
        there was only ever one slot to match against.
        """
        cache = _make_mru_cache(paged_cache, mock_ssd, max_entries=4)
        # Stash a short-prompt entry (parent_hash=None) — the false-match
        # bait for the freed-block scenario.
        short_tokens = [99, 100, 101]  # < block_size=4
        cache.store_cache(
            "req-short", short_tokens, [_kv_layer(mx, 3) for _ in range(4)]
        )
        # Confirm the short-prompt entry landed under None.
        assert None in cache._mru_partials

        # Construct a block_table whose last block has been freed.
        freed_block = paged_cache.allocate_block()
        freed_block_id = freed_block.block_id
        paged_cache.free_block(freed_block_id)
        bt = BlockTable(request_id="req-freed")
        bt.block_ids.append(freed_block_id)
        bt.num_tokens = 4

        reconstructed = _make_reconstructed_cache(mx, n_layers=4, n_tokens=4)
        _, new_remaining, applied = cache.apply_mru_partial(
            reconstructed, bt, [99, 100, 101]
        )

        # Must NOT splice the short-prompt entry even though the
        # remaining tokens happen to match.
        assert applied == 0
        assert new_remaining == [99, 100, 101]
        # Short-prompt entry preserved (not falsely evicted by the guard).
        assert None in cache._mru_partials

    # --- short-prompt None-key coexists with hash-keyed entry ---

    def test_short_prompt_none_key_coexists_with_block_aligned_entry(
        self, paged_cache, mock_ssd, mx
    ):
        cache = _make_mru_cache(paged_cache, mock_ssd, max_entries=4)
        # Short prompt (< block_size) → parent_hash=None
        cache.store_cache(
            "req-short", [10, 20, 30],
            [_kv_layer(mx, 3) for _ in range(4)],
        )
        # Longer prompt → distinct parent_hash
        _, hash_long = _stash_with_prefix(
            cache, mx, prefix_marker=1, tail_token=99
        )

        assert None in cache._mru_partials
        assert hash_long in cache._mru_partials
        assert len(cache._mru_partials) == 2


class TestMRUPartialCounters:
    """The observability counters mirror PR #1183's pattern so operators
    can answer "is the MRU cache paying off" with the same dashboard
    surface they use for prefix-hit and memory-hit rates.

    Counters: ``mru_partial_stashes``, ``mru_partial_hits``,
    ``mru_partial_evictions``, ``mru_partial_tokens_saved``.
    Gauges:    ``mru_partial_entries``, ``mru_partial_max_entries``.
    """

    @pytest.fixture
    def mx(self):
        try:
            import mlx.core as mx
            return mx
        except ImportError:
            pytest.skip("MLX not available")

    @pytest.fixture
    def paged_cache(self):
        return PagedCacheManager(
            block_size=4,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )

    @pytest.fixture
    def mock_ssd(self):
        mock = MagicMock()
        mock.save_block.return_value = True
        mock.load_block.return_value = None
        mock.load_block_with_metadata.return_value = (None, None)
        mock.has_block.return_value = False
        return mock

    def test_initial_counters_are_zero(self, paged_cache, mock_ssd):
        cache = _make_mru_cache(paged_cache, mock_ssd, max_entries=4)
        stats = cache.get_stats()
        assert stats.mru_partial_stashes == 0
        assert stats.mru_partial_hits == 0
        assert stats.mru_partial_evictions == 0
        assert stats.mru_partial_tokens_saved == 0
        assert stats.mru_partial_entries == 0
        assert stats.mru_partial_max_entries == 4

    def test_stash_increments_stash_counter(self, paged_cache, mock_ssd, mx):
        cache = _make_mru_cache(paged_cache, mock_ssd, max_entries=4)
        _stash_with_prefix(cache, mx, prefix_marker=1, tail_token=99)
        _stash_with_prefix(cache, mx, prefix_marker=2, tail_token=88)

        stats = cache.get_stats()
        assert stats.mru_partial_stashes == 2
        assert stats.mru_partial_entries == 2

    def test_same_key_replacement_counts_as_stash_not_eviction(
        self, paged_cache, mock_ssd, mx
    ):
        """Replacing an existing entry under the same key counts as a
        stash but NOT as an eviction.  Eviction is reserved for entries
        that leave the dict (capacity overflow, apply-miss, clear)."""
        cache = _make_mru_cache(paged_cache, mock_ssd, max_entries=4)
        _stash_with_prefix(cache, mx, prefix_marker=1, tail_token=99)
        _stash_with_prefix(cache, mx, prefix_marker=1, tail_token=77)

        stats = cache.get_stats()
        assert stats.mru_partial_stashes == 2
        assert stats.mru_partial_evictions == 0
        assert stats.mru_partial_entries == 1

    def test_capacity_overflow_increments_eviction_counter(
        self, paged_cache, mock_ssd, mx
    ):
        cache = _make_mru_cache(paged_cache, mock_ssd, max_entries=2)
        for i in (1, 2, 3):
            _stash_with_prefix(cache, mx, prefix_marker=i, tail_token=100 + i)

        stats = cache.get_stats()
        assert stats.mru_partial_stashes == 3
        assert stats.mru_partial_evictions == 1  # one entry pushed out
        assert stats.mru_partial_entries == 2

    def test_apply_success_increments_hits_and_tokens_saved(
        self, paged_cache, mock_ssd, mx
    ):
        cache = _make_mru_cache(paged_cache, mock_ssd, max_entries=4)
        bt, _ = _stash_with_prefix(
            cache, mx, prefix_marker=1, tail_token=901
        )

        reconstructed = _make_reconstructed_cache(mx, n_layers=4, n_tokens=4)
        _, _, applied = cache.apply_mru_partial(reconstructed, bt, [901])
        assert applied == 1

        stats = cache.get_stats()
        assert stats.mru_partial_hits == 1
        assert stats.mru_partial_tokens_saved == 1
        assert stats.mru_partial_evictions == 0  # success, not eviction

    def test_apply_miss_on_found_key_increments_eviction(
        self, paged_cache, mock_ssd, mx
    ):
        """Token-mismatch eviction pops the matched key and counts."""
        cache = _make_mru_cache(paged_cache, mock_ssd, max_entries=4)
        bt, _ = _stash_with_prefix(
            cache, mx, prefix_marker=1, tail_token=99
        )

        reconstructed = _make_reconstructed_cache(mx, n_layers=4, n_tokens=4)
        _, _, applied = cache.apply_mru_partial(reconstructed, bt, [77])  # wrong tail
        assert applied == 0

        stats = cache.get_stats()
        assert stats.mru_partial_hits == 0
        assert stats.mru_partial_evictions == 1

    def test_clear_mru_partials_counts_all_wiped_entries(
        self, paged_cache, mock_ssd, mx
    ):
        cache = _make_mru_cache(paged_cache, mock_ssd, max_entries=4)
        _stash_with_prefix(cache, mx, prefix_marker=1, tail_token=99)
        _stash_with_prefix(cache, mx, prefix_marker=2, tail_token=88)
        _stash_with_prefix(cache, mx, prefix_marker=3, tail_token=77)

        n = cache.clear_mru_partials()
        assert n == 3

        stats = cache.get_stats()
        assert stats.mru_partial_evictions == 3
        assert stats.mru_partial_entries == 0

    def test_clear_wipes_partials_and_resets_counters(
        self, paged_cache, mock_ssd, mx
    ):
        """clear() is the "restart everything" path (cache-corruption
        recovery).  It wipes the dict AND resets every counter,
        including mru_partial_evictions — incrementing evictions just
        to have them zeroed by the same call would be incoherent.
        Operators tracking partial wipes specifically use
        clear_mru_partials() instead.
        """
        cache = _make_mru_cache(paged_cache, mock_ssd, max_entries=4)
        _stash_with_prefix(cache, mx, prefix_marker=1, tail_token=99)
        _stash_with_prefix(cache, mx, prefix_marker=2, tail_token=88)

        cache.clear()

        stats = cache.get_stats()
        assert stats.mru_partial_entries == 0  # dict wiped
        assert stats.mru_partial_stashes == 0  # counters reset
        assert stats.mru_partial_evictions == 0
        assert stats.mru_partial_hits == 0

    def test_reset_stats_zeros_mru_counters_but_keeps_live_state(
        self, paged_cache, mock_ssd, mx
    ):
        """reset_stats() is the analyst's reset — it zeros cumulative
        counters but leaves the live cache state alone.  Use
        clear_mru_partials() if entries should be dropped too."""
        cache = _make_mru_cache(paged_cache, mock_ssd, max_entries=4)
        _stash_with_prefix(cache, mx, prefix_marker=1, tail_token=99)

        cache.reset_stats()

        stats = cache.get_stats()
        assert stats.mru_partial_stashes == 0
        assert stats.mru_partial_hits == 0
        assert stats.mru_partial_evictions == 0
        assert stats.mru_partial_tokens_saved == 0
        # But the live entry is still there.
        assert stats.mru_partial_entries == 1

    def test_get_stats_dict_mirrors_dataclass_after_round_trip(
        self, paged_cache, mock_ssd, mx
    ):
        """``get_stats_dict`` must surface every MRU field that ``get_stats``
        (the dataclass) does.  The admin dashboard reads MRU state via the
        dict path (``Scheduler.get_ssd_cache_stats`` -> ``get_stats_dict``);
        when the dict drops any of these keys, the admin payload's
        ``mru_partial_max_entries`` aggregates to 0 and the dashboard's
        ``mruEnabled`` gate hides every MRU panel even when the feature
        is enabled.

        Uses the production round-trip (real stashes via ``store_cache``,
        real apply via ``apply_mru_partial``, real capacity overflow) so
        the live gauge ``mru_partial_entries`` and every counter reach the
        dict via the same path the scheduler exercises.
        """
        cache = _make_mru_cache(paged_cache, mock_ssd, max_entries=2)
        # Stash two distinct prefixes (both fit in the cap).  The first
        # one (``bt_kept``) will be applied below to bump hits; the LRU
        # touch from a successful apply leaves it at the MRU end, so the
        # later capacity-overflow stash evicts the *other* survivor and
        # not the one we just touched.
        bt_kept, _ = _stash_with_prefix(
            cache, mx, prefix_marker=2, tail_token=88
        )
        _stash_with_prefix(cache, mx, prefix_marker=3, tail_token=77)

        reconstructed = _make_reconstructed_cache(mx, n_layers=4, n_tokens=4)
        _, _, applied = cache.apply_mru_partial(reconstructed, bt_kept, [88])
        assert applied == 1  # guard: the apply path actually fired

        # Third stash forces capacity-overflow eviction of the un-touched
        # entry.  After this: 3 stashes, 1 hit, 1 token saved, 1 eviction,
        # 2 live entries.
        _stash_with_prefix(cache, mx, prefix_marker=4, tail_token=66)

        stats = cache.get_stats()
        stats_dict = cache.get_stats_dict()

        # Every MRU field on the dataclass must surface in the dict with
        # the same value — this is the contract the admin route depends on.
        for field in (
            "mru_partial_stashes",
            "mru_partial_hits",
            "mru_partial_evictions",
            "mru_partial_tokens_saved",
            "mru_partial_entries",
            "mru_partial_max_entries",
        ):
            assert field in stats_dict, f"{field} missing from get_stats_dict()"
            assert stats_dict[field] == getattr(stats, field), (
                f"{field}: dict={stats_dict[field]} dataclass={getattr(stats, field)}"
            )

        # Sanity: the round-trip actually moved every counter off its
        # initial zero, so a future regression that hardwires zeros into
        # the dict would still fail this test.
        assert stats_dict["mru_partial_stashes"] == 3
        assert stats_dict["mru_partial_hits"] == 1
        assert stats_dict["mru_partial_evictions"] == 1  # capacity overflow
        assert stats_dict["mru_partial_tokens_saved"] == 1
        assert stats_dict["mru_partial_entries"] == 2
        assert stats_dict["mru_partial_max_entries"] == 2


def _model_with_make_cache(num_layers: int, layer_class_names: list[str]):
    """Build a MockModel whose ``make_cache()`` returns objects whose
    ``type(obj).__name__`` matches the requested cache class names.

    ``ModelCacheConfig.from_cache_list`` identifies cache types by class
    name (with isinstance fallback for SizedArraysCache only), so dynamic
    classes are enough to exercise the eager init-time eligibility check
    without pulling in real mlx-lm cache implementations.
    """
    cache_objs = [
        type(name, (object,), {"max_size": 64})()
        for name in layer_class_names
    ]
    model = MockModel(num_layers=num_layers)
    model.make_cache = lambda: cache_objs  # type: ignore[attr-defined]
    return model


class TestMRUPartialEligibility:
    """The ``mru_partial_supported`` tri-state flag and its one-shot
    warning.  Surfaces structurally-incompatible models on the admin
    dashboard so operators see ``N/A (see log)`` instead of a misleading
    ``0/N entries`` gauge.  Mirrors the prior-art Pattern B (real
    ``store_cache`` round-trip) for the lazy fallback path, and exercises
    the eager init-time path through a model with ``make_cache()``.
    """

    @pytest.fixture
    def mx(self):
        try:
            import mlx.core as mx
            return mx
        except ImportError:
            pytest.skip("MLX not available")

    @pytest.fixture
    def paged_cache(self):
        return PagedCacheManager(
            block_size=4,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )

    @pytest.fixture
    def mock_ssd(self):
        mock = MagicMock()
        mock.save_block.return_value = True
        mock.load_block.return_value = None
        mock.load_block_with_metadata.return_value = (None, None)
        mock.has_block.return_value = False
        return mock

    def test_supported_is_none_without_make_cache_and_no_inference(
        self, paged_cache, mock_ssd
    ):
        """``MockModel`` has no ``make_cache``; eager check bare-returns and
        lazy fallback hasn't fired yet — flag stays ``None``."""
        cache = _make_mru_cache(paged_cache, mock_ssd, max_entries=4)
        stats = cache.get_stats()
        assert stats.mru_partial_supported is None
        assert cache.get_stats_dict()["mru_partial_supported"] is None

    def test_supported_latches_true_on_sliceable_observation(
        self, paged_cache, mock_ssd, mx
    ):
        """A successful KVCache stash latches ``supported=True`` via the
        lazy path (eager skipped because MockModel has no make_cache)."""
        cache = _make_mru_cache(paged_cache, mock_ssd, max_entries=4)
        _stash_with_prefix(cache, mx, prefix_marker=1, tail_token=99)
        assert cache.get_stats().mru_partial_supported is True

    def test_supported_latches_false_lazy_on_non_sliceable(
        self, paged_cache, mock_ssd, mx, caplog
    ):
        """A store_cache with RotatingKVCache layers latches
        ``supported=False`` and emits exactly one warning."""
        from omlx.cache.hybrid_cache import ModelCacheConfig

        cache = _make_mru_cache(paged_cache, mock_ssd, max_entries=4)
        # Eager init skipped (MockModel has no make_cache), so flag is
        # None at this point — the lazy path is what we're testing.
        assert cache.get_stats().mru_partial_supported is None

        tokens = [10, 20, 30, 40, 50]
        cache_data = [_rotating_layer(mx, 5) for _ in range(4)]
        config = ModelCacheConfig.from_type_list(
            ["RotatingKVCache"] * 4, model_name="test"
        )
        with caplog.at_level(logging.WARNING, logger="omlx.cache.prefix_cache"):
            cache.store_cache("req-rot", tokens, cache_data, model_cache_config=config)

        stats = cache.get_stats()
        assert stats.mru_partial_supported is False
        assert stats.mru_partial_stashes == 0  # gate refused, no stash
        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warns) == 1
        assert "MRU tails will be inactive" in warns[0].getMessage()
        assert "incompatible" in warns[0].getMessage()
        assert "RotatingKVCache" in warns[0].getMessage()

    def test_warning_does_not_repeat_on_subsequent_non_sliceable(
        self, paged_cache, mock_ssd, mx, caplog
    ):
        """Once the flag is latched False, further non-sliceable store_cache
        calls must NOT re-emit the warning (operator log spam guard)."""
        from omlx.cache.hybrid_cache import ModelCacheConfig

        cache = _make_mru_cache(paged_cache, mock_ssd, max_entries=4)
        config = ModelCacheConfig.from_type_list(
            ["RotatingKVCache"] * 4, model_name="test"
        )
        with caplog.at_level(logging.WARNING, logger="omlx.cache.prefix_cache"):
            for i in range(3):
                cache.store_cache(
                    f"req-rot-{i}",
                    [10 * i + j for j in range(5)],
                    [_rotating_layer(mx, 5) for _ in range(4)],
                    model_cache_config=config,
                )

        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warns) == 1
        assert cache.get_stats().mru_partial_supported is False

    def test_eager_check_latches_false_at_init_with_non_sliceable_make_cache(
        self, paged_cache, mock_ssd, caplog
    ):
        """When ``model.make_cache()`` is available and returns non-sliceable
        cache instances, the flag latches False at construction and the
        warning fires BEFORE any inference — true model-load-time signal."""
        model = _model_with_make_cache(
            num_layers=4,
            layer_class_names=["RotatingKVCache"] * 4,
        )
        with caplog.at_level(logging.WARNING, logger="omlx.cache.prefix_cache"):
            cache = BlockAwarePrefixCache(
                model=model,
                paged_cache_manager=paged_cache,
                paged_ssd_cache_manager=mock_ssd,
                mru_partial_max_entries=4,
            )

        assert cache.get_stats().mru_partial_supported is False
        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warns) == 1
        assert "RotatingKVCache" in warns[0].getMessage()

    def test_eager_check_latches_true_at_init_with_sliceable_make_cache(
        self, paged_cache, mock_ssd, caplog
    ):
        """When ``model.make_cache()`` returns only sliceable cache
        instances, the flag latches True at construction and no warning
        is emitted."""
        model = _model_with_make_cache(
            num_layers=4,
            layer_class_names=["KVCache"] * 4,
        )
        with caplog.at_level(logging.WARNING, logger="omlx.cache.prefix_cache"):
            cache = BlockAwarePrefixCache(
                model=model,
                paged_cache_manager=paged_cache,
                paged_ssd_cache_manager=mock_ssd,
                mru_partial_max_entries=4,
            )

        assert cache.get_stats().mru_partial_supported is True
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_eager_check_skipped_when_feature_disabled(
        self, paged_cache, mock_ssd, caplog
    ):
        """``max_entries=0`` disables the feature; no eager check runs even
        for an obviously incompatible model.  No warning, no flag change —
        the operator already opted out by setting the capacity to zero."""
        model = _model_with_make_cache(
            num_layers=4,
            layer_class_names=["RotatingKVCache"] * 4,
        )
        with caplog.at_level(logging.WARNING, logger="omlx.cache.prefix_cache"):
            cache = BlockAwarePrefixCache(
                model=model,
                paged_cache_manager=paged_cache,
                paged_ssd_cache_manager=mock_ssd,
                mru_partial_max_entries=0,
            )

        assert cache.get_stats().mru_partial_supported is None
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_eager_check_survives_make_cache_failure(
        self, paged_cache, mock_ssd, caplog
    ):
        """If ``model.make_cache()`` raises, the eager check bare-returns
        and the flag stays ``None``.  Lazy fallback picks up at first
        inference instead — no startup crash."""
        model = MockModel(num_layers=4)
        model.make_cache = lambda: (_ for _ in ()).throw(  # type: ignore[attr-defined]
            RuntimeError("model not fully initialized")
        )
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
            mru_partial_max_entries=4,
        )
        assert cache.get_stats().mru_partial_supported is None


def _store_seq(cache, mx, request_id, tokens, *, prompt_token_count=None):
    """``store_cache`` a full token sequence; return the BlockTable.

    ``cache_data`` spans the whole sequence so ``_update_mru_partial``
    takes the global index path (the cache covers ``prompt + output``),
    matching the production resubmission layout.
    """
    cache_data = [_kv_layer(mx, len(tokens)) for _ in range(4)]
    return cache.store_cache(
        request_id, tokens, cache_data, prompt_token_count=prompt_token_count,
    )


class TestMRUPromptBoundaryStash:
    """The MRU stash must key off the *prompt's* trailing partial, not the
    stored sequence's.

    ``store_cache`` is handed ``prompt + output``, but a repeat request
    resubmits the prompt only and ``apply_mru_partial`` looks the entry
    up by the prompt's last full block.  Before the prompt boundary was
    threaded in, the stash keyed off ``prompt + output``'s last full
    block — a key a prompt-only resubmit could never compute — so the
    feature never produced a hit for ordinary chat completions.

    block_size is 4 in this fixture.
    """

    @pytest.fixture
    def mx(self):
        try:
            import mlx.core as mx
            return mx
        except ImportError:
            pytest.skip("MLX not available")

    @pytest.fixture
    def paged_cache(self):
        return PagedCacheManager(
            block_size=4,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )

    @pytest.fixture
    def mock_ssd(self):
        mock = MagicMock()
        mock.save_block.return_value = True
        mock.load_block.return_value = None
        mock.load_block_with_metadata.return_value = (None, None)
        mock.has_block.return_value = False
        return mock

    def test_prompt_boundary_stash_hits_on_prompt_only_resubmit(
        self, paged_cache, mock_ssd, mx
    ):
        """The decisive regression test: store ``prompt + output`` with the
        prompt boundary, then resubmit the prompt only — apply must HIT.
        """
        cache = _make_mru_cache(paged_cache, mock_ssd, max_entries=4)
        # prompt = 1 full block [10..13] + 2-token tail [14,15]; output = 3.
        prompt = [10, 11, 12, 13, 14, 15]
        stored = prompt + [90, 91, 92]
        store_bt = _store_seq(
            cache, mx, "store", stored, prompt_token_count=len(prompt)
        )

        # Keyed by the prompt's last full block (block 0) — NOT the stored
        # sequence's last full block (block 1) — and stashing the prompt
        # tail [14,15], not the sequence tail [92].
        prompt_block = paged_cache.allocated_blocks[store_bt.block_ids[0]]
        assert prompt_block.block_hash in cache._mru_partials
        assert cache._mru_partials[prompt_block.block_hash].tokens == [14, 15]

        # Simulate fetch_cache(prompt): a block table with the prompt's
        # full blocks only — apply_mru_partial keys off block_ids[-1].
        fetch_bt = BlockTable(request_id="resubmit")
        fetch_bt.block_ids = [store_bt.block_ids[0]]
        fetch_bt.num_tokens = 4
        reconstructed = _make_reconstructed_cache(mx, n_layers=4, n_tokens=4)

        _, new_remaining, applied = cache.apply_mru_partial(
            reconstructed, fetch_bt, [14, 15]
        )
        assert applied == 2
        assert new_remaining == []
        stats = cache.get_stats()
        assert stats.mru_partial_hits == 1
        assert stats.mru_partial_tokens_saved == 2

    def test_whole_sequence_stash_misses_on_prompt_only_resubmit(
        self, paged_cache, mock_ssd, mx
    ):
        """Pins the original bug: with no prompt boundary
        (``prompt_token_count=None``) the stash keys off the stored
        sequence's last full block, which a prompt-only resubmit never
        reaches — 0 hits, and 0 evictions because the lookup key is never
        found (no entry to evict).
        """
        cache = _make_mru_cache(paged_cache, mock_ssd, max_entries=4)
        prompt = [10, 11, 12, 13, 14, 15]
        stored = prompt + [90, 91, 92]
        store_bt = _store_seq(cache, mx, "store", stored)  # boundary unknown

        # None falls back to whole-sequence: keyed off block 1 (the stored
        # sequence's last full block), unreachable by a prompt-only fetch.
        seq_block = paged_cache.allocated_blocks[store_bt.block_ids[1]]
        assert seq_block.block_hash in cache._mru_partials

        fetch_bt = BlockTable(request_id="resubmit")
        fetch_bt.block_ids = [store_bt.block_ids[0]]
        fetch_bt.num_tokens = 4
        reconstructed = _make_reconstructed_cache(mx, n_layers=4, n_tokens=4)

        _, _, applied = cache.apply_mru_partial(reconstructed, fetch_bt, [14, 15])
        assert applied == 0
        stats = cache.get_stats()
        assert stats.mru_partial_hits == 0
        assert stats.mru_partial_evictions == 0  # key miss — nothing evicted

    def test_block_aligned_prompt_does_not_stash(
        self, paged_cache, mock_ssd, mx
    ):
        """A prompt that is an exact multiple of block_size has no partial
        tail — every prompt token lands in a full paged block, so the MRU
        has nothing to add."""
        cache = _make_mru_cache(paged_cache, mock_ssd, max_entries=4)
        prompt = [10, 11, 12, 13, 14, 15, 16, 17]  # 8 tokens = 2 full blocks
        stored = prompt + [90, 91, 92]
        _store_seq(cache, mx, "store", stored, prompt_token_count=len(prompt))
        assert not cache._mru_partials
        assert cache.get_stats().mru_partial_stashes == 0

    def test_short_prompt_stashes_under_none_key(
        self, paged_cache, mock_ssd, mx
    ):
        """A prompt shorter than one block has no last full block; the
        stash is keyed by None (the short-prompt path) and holds the
        whole prompt as its tail."""
        cache = _make_mru_cache(paged_cache, mock_ssd, max_entries=4)
        prompt = [10, 11, 12]  # 3 tokens < block_size 4
        stored = prompt + [90, 91, 92, 93, 94]
        _store_seq(cache, mx, "store", stored, prompt_token_count=len(prompt))
        assert None in cache._mru_partials
        assert cache._mru_partials[None].tokens == [10, 11, 12]

    def test_prompt_boundary_stash_with_existing_cached_prefix(
        self, paged_cache, mock_ssd, mx
    ):
        """Resubmission path: store_cache runs with ``existing_tokens > 0``
        (the prompt's leading blocks are already cached) and works in
        ``new_tokens`` space.  The prompt-boundary arithmetic must still
        resolve the prompt's last full block — not an index shifted by
        ``existing_tokens``.
        """
        cache = _make_mru_cache(paged_cache, mock_ssd, max_entries=4)
        # prompt = 2 full blocks [10..17] + 2-token tail [18,19]; output = 3.
        prompt = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        stored = prompt + [90, 91, 92]
        # Round 1: store the prompt's first full block only, same
        # request_id, so round 2's store sees existing_tokens > 0.
        _store_seq(cache, mx, "req", prompt[:8], prompt_token_count=8)
        assert not cache._mru_partials  # block-aligned round-1, no stash
        # Round 2: same request_id — existing_tokens == 8.
        store_bt = _store_seq(
            cache, mx, "req", stored, prompt_token_count=len(prompt)
        )

        # Prompt's last full block is index 1 ([14..17]); the stash must
        # key off it despite existing_tokens=8 and new_tokens-space math.
        prompt_block = paged_cache.allocated_blocks[store_bt.block_ids[1]]
        assert prompt_block.block_hash in cache._mru_partials
        assert cache._mru_partials[prompt_block.block_hash].tokens == [18, 19]

        fetch_bt = BlockTable(request_id="resubmit")
        fetch_bt.block_ids = store_bt.block_ids[:2]
        fetch_bt.num_tokens = 8
        reconstructed = _make_reconstructed_cache(mx, n_layers=4, n_tokens=8)
        _, new_remaining, applied = cache.apply_mru_partial(
            reconstructed, fetch_bt, [18, 19]
        )
        assert applied == 2
        assert new_remaining == []
        assert cache.get_stats().mru_partial_hits == 1


class TestMRUPartialCrossThreadSafety:
    """An MRU partial is extracted on the store-cache worker thread but
    spliced into a live cache on the separate inference thread.

    ``_extract_block_tensor_slice`` builds the partial as lazy
    ``mx.copy`` ops; an unevaluated tensor carries a pending op bound to
    the worker's per-thread MLX stream, and evaluating the splice on the
    inference thread raises ``RuntimeError: There is no Stream(gpu, N)
    in current thread``.  ``_update_mru_partial`` must materialize the
    partial at stash time so the stashed data is concrete and
    stream-free.
    """

    @pytest.fixture
    def mx(self):
        try:
            import mlx.core as mx
            return mx
        except ImportError:
            pytest.skip("MLX not available")

    @pytest.fixture
    def paged_cache(self):
        return PagedCacheManager(
            block_size=4,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )

    @pytest.fixture
    def mock_ssd(self):
        mock = MagicMock()
        mock.save_block.return_value = True
        mock.load_block.return_value = None
        mock.load_block_with_metadata.return_value = (None, None)
        mock.has_block.return_value = False
        return mock

    def test_materialize_mru_kv_handles_extract_shapes(
        self, paged_cache, mock_ssd, mx
    ):
        """``_materialize_mru_kv`` evaluates every ``mx.array`` leaf across
        the plain ``(keys, values)`` and TurboQuant ``(tag, (k, v))``
        shapes ``_extract_block_tensor_slice`` returns, and tolerates the
        non-array tag string and an empty list."""
        cache = _make_mru_cache(paged_cache, mock_ssd)
        plain = [(mx.ones((1, 1, 2, 4)), mx.ones((1, 1, 2, 4)))]
        tagged = [
            ("__turboquant_v2__", (mx.ones((1, 1, 2, 4)), mx.ones((1, 1, 2, 4))))
        ]
        # None of these should raise.
        cache._materialize_mru_kv(plain)
        cache._materialize_mru_kv(tagged)
        cache._materialize_mru_kv([])

    def test_stashed_partial_splices_across_threads(
        self, paged_cache, mock_ssd, mx
    ):
        """Extract+stash on a worker thread, splice+evaluate on the main
        thread.  Without stash-time materialization the final ``mx.eval``
        raises a foreign-stream ``RuntimeError``; with it, the cross-
        thread handoff is clean.
        """
        import concurrent.futures

        cache = _make_mru_cache(paged_cache, mock_ssd, max_entries=4)
        prompt = [10, 11, 12, 13, 14, 15]
        stored = prompt + [90, 91, 92]
        cache_data = [_kv_layer(mx, len(stored)) for _ in range(4)]
        # Mirror production: the inference thread materializes the
        # extracted cache (mx.async_eval + the worker's mx.synchronize)
        # before the store worker runs.  Without this the cache_data
        # arrays would still be lazy ops bound to THIS thread's stream —
        # a different cross-thread failure than the one under test.
        for layer in cache_data:
            mx.eval(*layer["state"])

        # Stash on a dedicated worker thread, mirroring the production
        # _store_cache_executor (a pool distinct from the inference one).
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            store_bt = pool.submit(
                cache.store_cache,
                "store",
                stored,
                cache_data,
                prompt_token_count=len(prompt),
            ).result()

        # Splice on this (the "inference") thread.
        fetch_bt = BlockTable(request_id="resubmit")
        fetch_bt.block_ids = [store_bt.block_ids[0]]
        fetch_bt.num_tokens = 4
        reconstructed = _make_reconstructed_cache(mx, n_layers=4, n_tokens=4)

        spliced, new_remaining, applied = cache.apply_mru_partial(
            reconstructed, fetch_bt, [14, 15]
        )
        assert applied == 2
        assert new_remaining == []

        # Force evaluation on this thread — the point a partial still
        # carrying the worker thread's stream would fail.
        for layer in spliced:
            mx.eval(layer.keys, layer.values)


class TestHasMRUPartial:
    """The has_mru_partial() accessor is the public API the scheduler
    uses to decide whether to suppress the deferred Metal cache clear."""

    def test_has_mru_partial_reflects_dict_emptiness(self):
        cache = BlockAwarePrefixCache(
            model=MockModel(num_layers=2),
            paged_cache_manager=PagedCacheManager(
                block_size=4, max_blocks=10, model_name="t", initial_blocks=10,
            ),
            paged_ssd_cache_manager=None,
        )
        assert cache.has_mru_partial() is False

        cache._mru_partials[b"x"] = _MRUPartialBlock(
            parent_hash=b"x", tokens=[1], kv_data=[],
        )
        assert cache.has_mru_partial() is True

        cache._mru_partials.clear()
        assert cache.has_mru_partial() is False


class TestSetPagedSSDCacheManagerTriggersSweep:
    """``set_paged_ssd_cache_manager`` must trigger the one-shot sweep so
    stale-signature blocks left over from a previous cache-config run for
    this model are evicted before the first prefix lookup runs."""

    @pytest.fixture
    def prefix_cache(self):
        return BlockAwarePrefixCache(
            model=MockModel(num_layers=4),
            paged_cache_manager=PagedCacheManager(
                block_size=4,
                max_blocks=16,
                model_name="test",
                initial_blocks=16,
            ),
            paged_ssd_cache_manager=None,
        )

    def test_attach_calls_invalidate(self, prefix_cache):
        mock_mgr = MagicMock()
        mock_mgr.invalidate_stale_layer_signature.return_value = 0

        prefix_cache.set_paged_ssd_cache_manager(mock_mgr)

        mock_mgr.invalidate_stale_layer_signature.assert_called_once_with()
        assert prefix_cache.paged_ssd_cache is mock_mgr

    def test_attach_survives_sweep_exception(self, prefix_cache):
        mock_mgr = MagicMock()
        mock_mgr.invalidate_stale_layer_signature.side_effect = RuntimeError("boom")

        # Must not raise — sweep failure is logged but the manager
        # connection must still complete.
        prefix_cache.set_paged_ssd_cache_manager(mock_mgr)

        assert prefix_cache.paged_ssd_cache is mock_mgr

    def test_attach_none_no_call(self, prefix_cache):
        # Detaching the manager must not invoke anything.
        prefix_cache.set_paged_ssd_cache_manager(None)
        assert prefix_cache.paged_ssd_cache is None


class TestCanonicalLayerCacheTypes:
    """The canonicalizer normalizes wrapper class names but must NOT
    collapse types that change tensor representation (TurboQuantKVCache
    stores 4-bit packed tensors; KVCache stores fp16) — collapsing those
    would silently mix incompatible cache blocks."""

    def test_none_passthrough(self):
        assert BlockAwarePrefixCache._canonical_layer_cache_types(None) is None

    def test_sized_arrays_normalized(self):
        result = BlockAwarePrefixCache._canonical_layer_cache_types(
            ["SizedArraysCache", "SizedArraysCache", "KVCache"]
        )
        assert result == ["ArraysCache", "ArraysCache", "KVCache"]

    def test_prefill_ready_rotating_normalized(self):
        result = BlockAwarePrefixCache._canonical_layer_cache_types(
            ["KVCache", "PrefillReadyRotatingKVCache", "RotatingKVCache"]
        )
        assert result == ["KVCache", "RotatingKVCache", "RotatingKVCache"]

    def test_turboquant_not_collapsed(self):
        result = BlockAwarePrefixCache._canonical_layer_cache_types(
            ["ArraysCache", "TurboQuantKVCache", "KVCache"]
        )
        # TurboQuant must remain distinct from plain KVCache.
        assert "TurboQuantKVCache" in result
        assert result != ["ArraysCache", "KVCache", "KVCache"]


class TestTurboQuantMixedPayloadReconstruction:
    """Payload-driven reconstruction of TurboQuant prefix-cache chains.

    A block chain can mix payload formats: blocks stored while TurboQuant KV
    conversion was active carry tagged ``('__turboquant_v2__', (ks, vs))``
    NamedTuple states, while blocks stored without conversion (chains
    written by versions that skipped it on chunked-prefill completion, or
    stored under a different TQ setting) carry plain dense
    ``(keys, values)`` tensors in the SAME chain via dedup.

    The per-block ``layer_cache_types`` mismatch check truncates such chains
    when every block's type metadata is present and accurate (covered by
    ``TestTurboQuantFormatMismatchRecovery``), but block metadata is
    optional at load time: metadata files can be missing or corrupt, the
    ``layer_cache_types`` JSON can fail to round-trip, and a chain whose
    first block lacks types gets typed from a later block. These tests model
    such metadata-blind chains (mixed payloads with absent
    ``layer_cache_types``), where the payload itself is the only ground
    truth. Typing the whole chain from chain-level metadata fed dense arrays
    into TurboQuant ``_concat_state`` and crashed with
    ``AttributeError: 'array' object has no attribute 'norms'``, rejecting
    the entire cache hit (full re-prefill on long-context hybrid models).
    """

    BLOCK = 4
    HEADS = 2
    HDIM = 64

    @pytest.fixture
    def mx(self):
        """Import MLX or skip."""
        try:
            import mlx.core as mx

            return mx
        except ImportError:
            pytest.skip("MLX not available")

    @pytest.fixture
    def tq_mod(self):
        """Import mlx_vlm.turboquant or skip."""
        return pytest.importorskip("mlx_vlm.turboquant")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_cache(self, num_layers=1):
        """Build a prefix cache with a mocked SSD manager."""
        from omlx.cache.paged_ssd_cache import PagedSSDCacheManager

        mock_ssd = MagicMock(spec=PagedSSDCacheManager)
        paged_cache = PagedCacheManager(
            block_size=self.BLOCK,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        cache = BlockAwarePrefixCache(
            model=MockModel(num_layers=num_layers),
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )
        return cache, paged_cache, mock_ssd

    def _alloc_block_table(self, paged_cache, n_blocks):
        """Allocate n hashed blocks and a block table referencing them."""
        blocks = []
        for i in range(n_blocks):
            b = paged_cache.allocate_block()
            b.block_hash = f"hash{i}".encode()
            b.token_count = self.BLOCK
            b.ref_count = 2  # Simulate fetch_cache having incremented ref
            blocks.append(b)
        return BlockTable(
            request_id="req-001",
            block_ids=[b.block_id for b in blocks],
            num_tokens=n_blocks * self.BLOCK,
        )

    @staticmethod
    def _unwrap(state):
        """Unwrap a _QuantizedStateProxy to the raw NamedTuple state."""
        return state._state if hasattr(state, "_state") else state

    def _tq_quantize(self, mx, tq_mod, keys, values, bits=4.0, seed=0):
        """Quantize (keys, values) with a real TurboQuantKVCache.

        Returns (tq_cache, ks_state, vs_state, ref_keys, ref_values) where
        ref_* is the full-state dequantized reference.
        """
        tq = tq_mod.TurboQuantKVCache(bits=bits, seed=seed)
        tq.update_and_fetch(keys, values)
        ks = self._unwrap(tq.keys)
        vs = self._unwrap(tq.values)
        ref_keys, ref_values = tq.dequantize(ks, vs)
        mx.eval(ref_keys, ref_values)
        return tq, ks, vs, ref_keys, ref_values

    def _tq_block_payload(self, tq_mod, ks, vs, block_idx):
        """Build one block's tagged TurboQuant payload by state slicing."""
        start = block_idx * self.BLOCK
        end = start + self.BLOCK
        return (
            "__turboquant_v2__",
            (
                tq_mod._slice_state_range(ks, start, end),
                tq_mod._slice_state_range(vs, start, end),
            ),
        )

    def _metadata(self, layer_cache_types, layer_meta_states, num_layers):
        return {
            "model_name": "test-model",
            "num_layers": num_layers,
            "block_size": self.BLOCK,
            "layer_cache_types": layer_cache_types,
            "layer_meta_states": layer_meta_states,
        }

    # ------------------------------------------------------------------
    # Mixed-format chains (the issue regression)
    # ------------------------------------------------------------------

    def test_mixed_chain_tq_head_plain_tail_reconstructs(self, mx, tq_mod):
        """Chain typed TurboQuant (first block) with a plain dense tail block.

        Regression for the multi-turn pattern: turn 1 stores TQ-converted
        blocks; a later turn appends a plain dense block to the same chain.
        The tail block's layer_cache_types is absent (block metadata is
        optional at load time), so the type-mismatch truncation cannot see
        the format change. Reconstruction must dequantize the TQ blocks per
        block and pass the dense block through instead of rejecting the hit.
        """
        from mlx_lm.models.cache import KVCache

        cache, paged_cache, mock_ssd = self._make_cache(num_layers=1)
        block_table = self._alloc_block_table(paged_cache, 3)

        # Blocks 0-1: TurboQuant-converted store (8 tokens quantized)
        full_keys = mx.random.normal((1, self.HEADS, 2 * self.BLOCK, self.HDIM))
        full_values = mx.random.normal((1, self.HEADS, 2 * self.BLOCK, self.HDIM))
        tq, ks, vs, ref_keys, ref_values = self._tq_quantize(
            mx, tq_mod, full_keys, full_values
        )

        # Block 2: plain dense store (skipped TQ conversion), stored
        # without layer_cache_types metadata.
        plain_k = mx.random.normal((1, self.HEADS, self.BLOCK, self.HDIM)).astype(
            mx.float16
        )
        plain_v = mx.random.normal((1, self.HEADS, self.BLOCK, self.HDIM)).astype(
            mx.float16
        )

        tq_metadata = self._metadata(["TurboQuantKVCache"], [tq.meta_state], 1)
        plain_metadata = self._metadata(None, [()], 1)

        mock_ssd.load_block_with_metadata.side_effect = [
            ([self._tq_block_payload(tq_mod, ks, vs, 0)], tq_metadata),
            ([self._tq_block_payload(tq_mod, ks, vs, 1)], tq_metadata),
            ([(plain_k, plain_v)], plain_metadata),
        ]

        result = cache.reconstruct_cache(block_table)

        assert result is not None, "mixed TQ+plain chain must reconstruct"
        assert len(result) == 1
        layer0 = result[0]
        assert isinstance(layer0, KVCache)
        assert layer0.offset == 3 * self.BLOCK
        assert layer0.keys.shape == (1, self.HEADS, 3 * self.BLOCK, self.HDIM)
        # Healed layer is cast back to the dense blocks' stored dtype so an
        # fp32 layer never leaks into batch merge on non-requantizing servers.
        assert layer0.keys.dtype == mx.float16
        assert layer0.values.dtype == mx.float16

        # TQ blocks: per-block dequantize must equal the full-state
        # dequantize reference rounded to the healed layer's dense dtype.
        assert mx.array_equal(
            layer0.keys[:, :, : 2 * self.BLOCK, :],
            ref_keys.astype(layer0.keys.dtype),
        )
        assert mx.array_equal(
            layer0.values[:, :, : 2 * self.BLOCK, :],
            ref_values.astype(layer0.values.dtype),
        )
        # Plain block: passed through unmodified (promoted dtype only).
        assert mx.allclose(
            layer0.keys[:, :, 2 * self.BLOCK :, :],
            plain_k.astype(layer0.keys.dtype),
            atol=1e-3,
        )
        assert mx.allclose(
            layer0.values[:, :, 2 * self.BLOCK :, :],
            plain_v.astype(layer0.values.dtype),
            atol=1e-3,
        )

    def test_mixed_chain_plain_head_tq_tail_reconstructs(self, mx, tq_mod):
        """Chain typed KVCache (first block dense) with TQ-tagged tail blocks.

        The reverse mixing direction: the chain-level type from the first
        block routes through the standard KVCache branch. The TQ tail
        blocks carry no layer_cache_types metadata (only their per-block
        meta_states), so the type-mismatch truncation cannot see them; the
        payload scan must route the layer into the TurboQuant branch, which
        decodes tagged blocks with their own per-block meta instead of
        unpacking the tag string as a tensor.
        """
        from mlx_lm.models.cache import KVCache

        cache, paged_cache, mock_ssd = self._make_cache(num_layers=1)
        block_table = self._alloc_block_table(paged_cache, 3)

        # Block 0: plain dense store
        plain_k = mx.random.normal((1, self.HEADS, self.BLOCK, self.HDIM)).astype(
            mx.float16
        )
        plain_v = mx.random.normal((1, self.HEADS, self.BLOCK, self.HDIM)).astype(
            mx.float16
        )

        # Blocks 1-2: TurboQuant-converted store, stored without
        # layer_cache_types metadata.
        full_keys = mx.random.normal((1, self.HEADS, 2 * self.BLOCK, self.HDIM))
        full_values = mx.random.normal((1, self.HEADS, 2 * self.BLOCK, self.HDIM))
        tq, ks, vs, ref_keys, ref_values = self._tq_quantize(
            mx, tq_mod, full_keys, full_values
        )

        plain_metadata = self._metadata(["KVCache"], [()], 1)
        tq_metadata = self._metadata(None, [tq.meta_state], 1)

        mock_ssd.load_block_with_metadata.side_effect = [
            ([(plain_k, plain_v)], plain_metadata),
            ([self._tq_block_payload(tq_mod, ks, vs, 0)], tq_metadata),
            ([self._tq_block_payload(tq_mod, ks, vs, 1)], tq_metadata),
        ]

        result = cache.reconstruct_cache(block_table)

        assert result is not None, "mixed plain+TQ chain must reconstruct"
        assert len(result) == 1
        layer0 = result[0]
        assert isinstance(layer0, KVCache)
        assert layer0.offset == 3 * self.BLOCK
        assert layer0.keys.shape == (1, self.HEADS, 3 * self.BLOCK, self.HDIM)
        assert layer0.keys.dtype == mx.float16
        assert layer0.values.dtype == mx.float16

        assert mx.allclose(
            layer0.keys[:, :, : self.BLOCK, :],
            plain_k.astype(layer0.keys.dtype),
            atol=1e-3,
        )
        assert mx.array_equal(
            layer0.keys[:, :, self.BLOCK :, :],
            ref_keys.astype(layer0.keys.dtype),
        )
        assert mx.array_equal(
            layer0.values[:, :, self.BLOCK :, :],
            ref_values.astype(layer0.values.dtype),
        )

    def test_mixed_chain_hybrid_model_with_arrays_cache_layer(self, mx, tq_mod):
        """Hybrid (Qwen3.5-style) layout: TQ attention layer + ArraysCache GDN
        layer, with a plain dense block (no layer_cache_types metadata)
        appended to the TQ chain."""
        from mlx_lm.models.cache import KVCache

        cache, paged_cache, mock_ssd = self._make_cache(num_layers=2)
        block_table = self._alloc_block_table(paged_cache, 3)

        full_keys = mx.random.normal((1, self.HEADS, 2 * self.BLOCK, self.HDIM))
        full_values = mx.random.normal((1, self.HEADS, 2 * self.BLOCK, self.HDIM))
        tq, ks, vs, _, _ = self._tq_quantize(mx, tq_mod, full_keys, full_values)

        plain_k = mx.random.normal((1, self.HEADS, self.BLOCK, self.HDIM)).astype(
            mx.float16
        )
        plain_v = mx.random.normal((1, self.HEADS, self.BLOCK, self.HDIM)).astype(
            mx.float16
        )

        placeholder = (mx.zeros((1,)), mx.zeros((1,)))
        # ArraysCache (GDN) real state lives in the last stored block.
        conv_state = mx.ones((1, 3, 64))
        ssm_state = mx.ones((1, 32, 16, 16))

        tq_metadata = self._metadata(
            ["TurboQuantKVCache", "ArraysCache"], [tq.meta_state, ()], 2
        )
        plain_metadata = self._metadata(None, [(), ()], 2)

        mock_ssd.load_block_with_metadata.side_effect = [
            ([self._tq_block_payload(tq_mod, ks, vs, 0), placeholder], tq_metadata),
            ([self._tq_block_payload(tq_mod, ks, vs, 1), placeholder], tq_metadata),
            ([(plain_k, plain_v), (conv_state, ssm_state)], plain_metadata),
        ]

        result = cache.reconstruct_cache(block_table)

        assert result is not None, "hybrid mixed chain must reconstruct"
        assert len(result) == 2
        assert isinstance(result[0], KVCache)
        assert result[0].offset == 3 * self.BLOCK
        assert result[0].keys.dtype == mx.float16
        # GDN layer reconstructed from the last block's real state.
        assert not isinstance(result[1], KVCache)

    # ------------------------------------------------------------------
    # Homogeneous chains: zero behavior change
    # ------------------------------------------------------------------

    def test_all_tq_chain_reconstructs_unchanged(self, mx, tq_mod):
        """Uniform TQ chain stays quantized (single-run fast path).

        A homogeneous TQ chain is a single run, so it is restored as a
        quantized ``TurboQuantKVCache`` (upstream #1842: restored
        long-context prefixes stay quantized and avoid full-state
        materialization) rather than being dequantized into a dense
        ``KVCache``. The dequantized state must still match the full-state
        reference (TQ states are per-token, so per-block concat equals the
        full-state concat).
        """
        cache, paged_cache, mock_ssd = self._make_cache(num_layers=1)
        block_table = self._alloc_block_table(paged_cache, 3)

        full_keys = mx.random.normal((1, self.HEADS, 3 * self.BLOCK, self.HDIM))
        full_values = mx.random.normal((1, self.HEADS, 3 * self.BLOCK, self.HDIM))
        tq, ks, vs, ref_keys, ref_values = self._tq_quantize(
            mx, tq_mod, full_keys, full_values
        )

        tq_metadata = self._metadata(["TurboQuantKVCache"], [tq.meta_state], 1)
        mock_ssd.load_block_with_metadata.side_effect = [
            ([self._tq_block_payload(tq_mod, ks, vs, i)], tq_metadata)
            for i in range(3)
        ]

        result = cache.reconstruct_cache(block_table)

        assert result is not None
        layer0 = result[0]
        assert isinstance(layer0, tq_mod.TurboQuantKVCache)
        assert layer0.offset == 3 * self.BLOCK
        rebuilt_keys, rebuilt_values = layer0.dequantize()
        assert mx.allclose(rebuilt_keys, ref_keys, atol=1e-5)
        assert mx.allclose(rebuilt_values, ref_values, atol=1e-5)

    def test_all_plain_chain_reconstructs_unchanged(self, mx):
        """Uniform dense chain keeps exact values and stored dtype."""
        from mlx_lm.models.cache import KVCache

        cache, paged_cache, mock_ssd = self._make_cache(num_layers=1)
        block_table = self._alloc_block_table(paged_cache, 3)

        slices = [
            (
                mx.random.normal((1, self.HEADS, self.BLOCK, self.HDIM)).astype(
                    mx.float16
                ),
                mx.random.normal((1, self.HEADS, self.BLOCK, self.HDIM)).astype(
                    mx.float16
                ),
            )
            for _ in range(3)
        ]

        plain_metadata = self._metadata(["KVCache"], [()], 1)
        mock_ssd.load_block_with_metadata.side_effect = [
            ([s], plain_metadata) for s in slices
        ]

        result = cache.reconstruct_cache(block_table)

        assert result is not None
        layer0 = result[0]
        assert isinstance(layer0, KVCache)
        assert layer0.offset == 3 * self.BLOCK
        assert layer0.keys.dtype == mx.float16
        expected_keys = mx.concatenate([k for k, _ in slices], axis=2)
        expected_values = mx.concatenate([v for _, v in slices], axis=2)
        assert mx.array_equal(layer0.keys, expected_keys)
        assert mx.array_equal(layer0.values, expected_values)

    # ------------------------------------------------------------------
    # Corrupt / placeholder payloads: clean rejection
    # ------------------------------------------------------------------

    def test_tq_chain_placeholder_block_rejected(self, mx, tq_mod):
        """A (1,) empty-slice placeholder in a TQ-typed chain rejects the hit
        cleanly (None) instead of corrupting the concatenated KV."""
        cache, paged_cache, mock_ssd = self._make_cache(num_layers=1)
        block_table = self._alloc_block_table(paged_cache, 2)

        full_keys = mx.random.normal((1, self.HEADS, self.BLOCK, self.HDIM))
        full_values = mx.random.normal((1, self.HEADS, self.BLOCK, self.HDIM))
        tq, ks, vs, _, _ = self._tq_quantize(mx, tq_mod, full_keys, full_values)

        placeholder = (mx.zeros((1,)), mx.zeros((1,)))
        tq_metadata = self._metadata(["TurboQuantKVCache"], [tq.meta_state], 1)

        mock_ssd.load_block_with_metadata.side_effect = [
            ([self._tq_block_payload(tq_mod, ks, vs, 0)], tq_metadata),
            ([placeholder], tq_metadata),
        ]

        assert cache.reconstruct_cache(block_table) is None

    def test_plain_chain_tq_placeholder_block_rejected(self, mx):
        """A (1,) placeholder from a TQ-regime store inside a KVCache-typed
        chain rejects the hit cleanly (None)."""
        cache, paged_cache, mock_ssd = self._make_cache(num_layers=1)
        block_table = self._alloc_block_table(paged_cache, 2)

        plain_k = mx.random.normal((1, self.HEADS, self.BLOCK, self.HDIM)).astype(
            mx.float16
        )
        plain_v = mx.random.normal((1, self.HEADS, self.BLOCK, self.HDIM)).astype(
            mx.float16
        )
        placeholder = (mx.zeros((1,)), mx.zeros((1,)))
        plain_metadata = self._metadata(["KVCache"], [()], 1)

        mock_ssd.load_block_with_metadata.side_effect = [
            ([(plain_k, plain_v)], plain_metadata),
            ([placeholder], plain_metadata),
        ]

        assert cache.reconstruct_cache(block_table) is None

    def test_tq_chain_corrupt_payload_rejected(self, mx, tq_mod):
        """A payload that is neither tagged, placeholder, nor dense 4D KV
        rejects the hit cleanly (None) instead of producing wrong KV."""
        cache, paged_cache, mock_ssd = self._make_cache(num_layers=1)
        block_table = self._alloc_block_table(paged_cache, 2)

        full_keys = mx.random.normal((1, self.HEADS, self.BLOCK, self.HDIM))
        full_values = mx.random.normal((1, self.HEADS, self.BLOCK, self.HDIM))
        tq, ks, vs, _, _ = self._tq_quantize(mx, tq_mod, full_keys, full_values)

        corrupt = (mx.zeros((2, 2)), mx.zeros((2, 2)))  # 2D, not a KV slice
        tq_metadata = self._metadata(["TurboQuantKVCache"], [tq.meta_state], 1)

        mock_ssd.load_block_with_metadata.side_effect = [
            ([self._tq_block_payload(tq_mod, ks, vs, 0)], tq_metadata),
            ([corrupt], tq_metadata),
        ]

        assert cache.reconstruct_cache(block_table) is None

    def test_tq_chain_absent_meta_uses_configured_bits(self, mx, tq_mod):
        """Uniform TQ chain whose (bits, seed) metadata never round-tripped.

        Both blocks' metadata carry no layer_meta_states (the JSON failed
        to round-trip on save). Neither
        per-block nor chain-level meta can resolve (bits, seed). The
        server-configured TurboQuant KV bit depth is authoritative for
        admitted blocks (cache eligibility already keys on it), so
        reconstruction must rebuild the codec at that depth — 8 bits here —
        rather than the historical bits=4.0 guess, which dequantized 8-bit
        states into plausible-but-wrong tensors: silent output corruption
        after a cache hit instead of a loud failure.
        """
        cache, paged_cache, mock_ssd = self._make_cache(num_layers=1)
        mock_ssd._expected_turboquant_kv_bits = 8.0
        block_table = self._alloc_block_table(paged_cache, 2)

        full_keys = mx.random.normal((1, self.HEADS, 2 * self.BLOCK, self.HDIM))
        full_values = mx.random.normal(
            (1, self.HEADS, 2 * self.BLOCK, self.HDIM)
        )
        tq, ks, vs, ref_keys, ref_values = self._tq_quantize(
            mx, tq_mod, full_keys, full_values, bits=8.0
        )

        no_meta_states = self._metadata(["TurboQuantKVCache"], None, 1)
        mock_ssd.load_block_with_metadata.side_effect = [
            ([self._tq_block_payload(tq_mod, ks, vs, 0)], no_meta_states),
            ([self._tq_block_payload(tq_mod, ks, vs, 1)], no_meta_states),
        ]

        result = cache.reconstruct_cache(block_table)

        assert result is not None, "configured bit depth must resolve the codec"
        assert len(result) == 1
        layer0 = result[0]
        assert layer0.offset == 2 * self.BLOCK
        got_keys, got_values = layer0.dequantize(
            self._unwrap(layer0.keys), self._unwrap(layer0.values)
        )
        assert mx.allclose(got_keys, ref_keys, atol=1e-5)
        assert mx.allclose(got_values, ref_values, atol=1e-5)

    def test_tq_chain_unresolvable_params_rejected(self, mx, tq_mod):
        """No meta anywhere and no configured bit depth: reject, don't guess.

        Historically this path silently rebuilt the codec at bits=4.0. For a
        chain stored at any other depth that dequantizes to wrong-width
        tensors and the request degenerates after the cache hit (upstream
        issue: `!!!!` output after prefix-cache restores). A rejected hit
        merely re-prefills — strictly safer than plausible-but-wrong KV.
        """
        cache, paged_cache, mock_ssd = self._make_cache(num_layers=1)
        mock_ssd._expected_turboquant_kv_bits = None
        block_table = self._alloc_block_table(paged_cache, 2)

        full_keys = mx.random.normal((1, self.HEADS, 2 * self.BLOCK, self.HDIM))
        full_values = mx.random.normal(
            (1, self.HEADS, 2 * self.BLOCK, self.HDIM)
        )
        tq, ks, vs, _, _ = self._tq_quantize(
            mx, tq_mod, full_keys, full_values, bits=8.0
        )

        no_meta_states = self._metadata(["TurboQuantKVCache"], None, 1)
        mock_ssd.load_block_with_metadata.side_effect = [
            ([self._tq_block_payload(tq_mod, ks, vs, 0)], no_meta_states),
            ([self._tq_block_payload(tq_mod, ks, vs, 1)], no_meta_states),
        ]

        assert cache.reconstruct_cache(block_table) is None


class TestReconstructionSilentFallbackHardening:
    """Guessed-default fallbacks in reconstruction reject instead of guessing.

    Follow-up to the PR #2272 direction (payload over guesswork): a cache hit
    is an optimization, so any ambiguity about reconstruction inputs must
    resolve to re-prefill (block drop / chain truncation / loud rejection),
    never to a plausible-but-wrong default.
    """

    BLOCK = 4
    HEADS = 2
    HDIM = 64

    @pytest.fixture
    def mx(self):
        """Import MLX or skip."""
        try:
            import mlx.core as mx

            return mx
        except ImportError:
            pytest.skip("MLX not available")

    def _make_cache(self, num_layers=1):
        """Build a prefix cache with a mocked SSD manager."""
        from omlx.cache.paged_ssd_cache import PagedSSDCacheManager

        mock_ssd = MagicMock(spec=PagedSSDCacheManager)
        paged_cache = PagedCacheManager(
            block_size=self.BLOCK,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        cache = BlockAwarePrefixCache(
            model=MockModel(num_layers=num_layers),
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )
        return cache, paged_cache, mock_ssd

    def _alloc_block_table(self, paged_cache, n_blocks):
        """Allocate n hashed blocks and a block table referencing them."""
        blocks = []
        for i in range(n_blocks):
            b = paged_cache.allocate_block()
            b.block_hash = f"hash{i}".encode()
            b.token_count = self.BLOCK
            b.ref_count = 2
            blocks.append(b)
        return BlockTable(
            request_id="req-001",
            block_ids=[b.block_id for b in blocks],
            num_tokens=n_blocks * self.BLOCK,
        )

    def _plain_slice(self, mx):
        return (
            mx.random.normal((1, self.HEADS, self.BLOCK, self.HDIM)).astype(
                mx.float16
            ),
            mx.random.normal((1, self.HEADS, self.BLOCK, self.HDIM)).astype(
                mx.float16
            ),
        )

    def _metadata(self, num_layers=1):
        return {
            "model_name": "test-model",
            "num_layers": num_layers,
            "block_size": self.BLOCK,
            "layer_cache_types": ["KVCache"] * num_layers,
            "layer_meta_states": [()] * num_layers,
        }

    def test_metadata_less_block_truncates_chain(self, mx):
        """A block that loads with data but no metadata truncates the chain.

        Continuing past it kept the first/last meta_state trackers stale
        (pairing a later block's tensors with an earlier block's meta) and
        skipped every per-block validation gate (model_name, num_layers,
        block_size, layer types). Treat it like a load failure: keep the
        valid prefix, drop the untrusted tail.
        """
        from mlx_lm.models.cache import KVCache

        cache, paged_cache, mock_ssd = self._make_cache(num_layers=1)
        block_table = self._alloc_block_table(paged_cache, 3)

        slices = [self._plain_slice(mx) for _ in range(3)]
        mock_ssd.load_block_with_metadata.side_effect = [
            ([slices[0]], self._metadata()),
            ([slices[1]], self._metadata()),
            ([slices[2]], None),
        ]

        result = cache.reconstruct_cache(block_table)

        assert result is not None, "valid prefix must survive truncation"
        layer0 = result[0]
        assert isinstance(layer0, KVCache)
        assert layer0.offset == 2 * self.BLOCK
        assert block_table.num_tokens == 2 * self.BLOCK
        expected_keys = mx.concatenate([k for k, _ in slices[:2]], axis=2)
        assert mx.array_equal(layer0.keys, expected_keys)

    def test_fallback_reconstruct_refuses_non_kvcache_types(self, mx):
        """Handler-declared failure for a stateful cache type must reject.

        _fallback_reconstruct_layer rebuilds a plain KVCache, which is the
        wrong cache class for rotating/arrays/composite layers: a rotating
        buffer restored as KVCache carries wrong positions and merge-unsafe
        state. Only KVCache-typed layers may take the fallback.
        """
        cache, _, _ = self._make_cache(num_layers=1)
        layer_states = [
            {
                "keys": mx.zeros((1, self.HEADS, self.BLOCK, self.HDIM)),
                "values": mx.zeros((1, self.HEADS, self.BLOCK, self.HDIM)),
            }
        ]

        for type_name in ("RotatingKVCache", "ArraysCache", "CacheList"):
            assert (
                cache._fallback_reconstruct_layer(layer_states, type_name) is None
            ), f"{type_name} must not fall back to a plain KVCache rebuild"

    def test_fallback_reconstruct_still_serves_kvcache(self, mx):
        """The fallback keeps working for the type it can actually rebuild."""
        cache, _, _ = self._make_cache(num_layers=1)
        layer_states = [
            {
                "keys": mx.zeros((1, self.HEADS, self.BLOCK, self.HDIM)),
                "values": mx.zeros((1, self.HEADS, self.BLOCK, self.HDIM)),
            }
        ]

        rebuilt = cache._fallback_reconstruct_layer(layer_states, "KVCache")

        assert rebuilt is not None
        assert rebuilt.offset == self.BLOCK
