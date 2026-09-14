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
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from omlx.cache.paged_cache import (
    BlockTable,
    PagedCacheManager,
    compute_block_hash,
)
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
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

    @staticmethod
    def _make_ssd_prefix_cache(
        cache_directory: Path,
        *,
        num_layers: int = 1,
        hot_cache_max_bytes: int = 0,
        hot_cache_only: bool = False,
    ) -> tuple[BlockAwarePrefixCache, PagedCacheManager, PagedSSDCacheManager]:
        ssd_manager = PagedSSDCacheManager(
            cache_dir=cache_directory,
            max_size_bytes=100 * 1024**2,
            hot_cache_max_bytes=hot_cache_max_bytes,
            hot_cache_only=hot_cache_only,
            expected_model_name="test-model",
            expected_num_layers=num_layers,
            expected_block_size=4,
        )
        paged_manager = PagedCacheManager(
            block_size=4,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        paged_manager.set_paged_ssd_cache_manager(ssd_manager)
        prefix_cache = BlockAwarePrefixCache(
            model=MockModel(num_layers=num_layers),
            paged_cache_manager=paged_manager,
            paged_ssd_cache_manager=ssd_manager,
        )
        return prefix_cache, paged_manager, ssd_manager

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

    def test_shared_prefix_hit_rejects_reassigned_block(self, prefix_cache):
        """A2 regression: a block reassigned to different content between
        find_shared_prefix's lookup and the caller acquiring it must not be
        spliced into the returned chain. The race is injected at
        acquire_cached_block itself -- the one point genuinely between the
        two steps -- by mutating the block's hash there, as a concurrent
        eviction + reallocation would, before delegating to the real
        implementation."""
        import mlx.core as mx

        tokens = list(range(8))  # 2 full blocks at block_size=4
        keys = mx.arange(8, dtype=mx.float32).reshape(1, 1, 8, 1)
        values = (keys + 100).astype(mx.float32)
        cache_data = [
            {"state": (keys, values), "cache_type": "KVCache", "class_name": "KVCache"}
        ]

        table = prefix_cache.store_cache("req-store", tokens, cache_data)
        assert table is not None
        assert len(table.block_ids) == 2
        prefix_cache.clear_request_entry("req-store")

        second_block_id = table.block_ids[1]
        second_block = prefix_cache.paged_cache.allocated_blocks[second_block_id]
        paged_cache = prefix_cache.paged_cache
        real_acquire = paged_cache.acquire_cached_block

        def racy_acquire(block_id, expected_hash):
            if block_id == second_block_id:
                # Concurrent eviction + reallocation lands here, between
                # find_shared_prefix's lookup and this acquire.
                second_block.block_hash = b"reassigned-to-other-content__"
            return real_acquire(block_id, expected_hash)

        paged_cache.acquire_cached_block = racy_acquire
        try:
            hit_table, remaining = prefix_cache.fetch_cache("req-fetch", tokens)
        finally:
            paged_cache.acquire_cached_block = real_acquire

        # Chain stops at the first mismatch: only the still-valid first
        # block is returned, never the reassigned second block.
        assert hit_table is not None
        assert hit_table.block_ids == [table.block_ids[0]]
        assert remaining == tokens[4:]

    def test_exact_prefix_survives_ssd_manager_restart(self, tmp_path):
        """An exact static prefix includes its partial terminal block on SSD."""
        import mlx.core as mx

        cache_directory = tmp_path / "exact-prefix"
        tokens = list(range(7))
        keys = mx.arange(7, dtype=mx.float32).reshape(1, 1, 7, 1)
        values = (keys + 10).astype(mx.float32)
        extracted_cache = [
            {
                "state": (keys, values),
                "meta_state": (7,),
                "class_name": "KVCache",
                "cache_type": "KVCache",
            }
        ]

        first_prefix_cache, _, first_ssd_manager = self._make_ssd_prefix_cache(
            cache_directory
        )

        stored_table = first_prefix_cache.store_exact_prefix(
            "cold-static-prefix",
            tokens,
            extracted_cache,
        )
        assert stored_table is not None
        assert stored_table.num_tokens == len(tokens)

        ordinary_table, ordinary_remaining = first_prefix_cache.fetch_cache(
            "ordinary-prefix",
            [*tokens, 8],
        )
        assert ordinary_table is not None
        assert ordinary_table.num_tokens == 4
        assert ordinary_remaining == [4, 5, 6, 8]
        first_prefix_cache.release_cache("ordinary-prefix")
        first_ssd_manager.close()

        restarted_prefix_cache, _, restarted_ssd_manager = self._make_ssd_prefix_cache(
            cache_directory
        )

        try:
            restored_table = restarted_prefix_cache.fetch_exact_prefix(
                "warm-static-prefix",
                tokens,
            )
            assert restored_table is not None
            restored_cache = restarted_prefix_cache.reconstruct_cache(
                restored_table,
                promote_to_hot_cache=False,
            )
            assert restored_cache is not None
            restored_keys, restored_values = restored_cache[0].state
            assert restored_table.num_tokens == len(tokens)
            assert mx.array_equal(restored_keys, keys).item()
            assert mx.array_equal(restored_values, values).item()

            changed_tokens = [*tokens[:-1], 99]
            assert (
                restarted_prefix_cache.fetch_exact_prefix(
                    "changed-static-prefix",
                    changed_tokens,
                )
                is None
            )
        finally:
            restarted_ssd_manager.close()

    def test_aligned_exact_prefix_does_not_reuse_placeholder_terminal(self, tmp_path):
        """An aligned static tip must not collide with an ordinary chain block."""
        import mlx.core as mx

        from omlx.cache.hybrid_cache import ModelCacheConfig

        prefix_cache, _, ssd_manager = self._make_ssd_prefix_cache(
            tmp_path / "aligned-exact-prefix",
            num_layers=2,
        )
        model_cache_config = ModelCacheConfig.from_type_list(
            ["KVCache", "ArraysCache"],
            model_name="test-model",
        )
        keys = mx.arange(12, dtype=mx.float32).reshape(1, 1, 12, 1)
        values = keys + 20
        first_recurrent_state = mx.ones((1, 2), dtype=mx.float32)
        second_recurrent_state = mx.ones((1, 2), dtype=mx.float32) * 2
        extracted_cache = [
            {
                "state": (keys, values),
                "meta_state": (12,),
                "class_name": "KVCache",
                "cache_type": "KVCache",
            },
            {
                "state": (first_recurrent_state, second_recurrent_state),
                "meta_state": (),
                "class_name": "ArraysCache",
                "cache_type": "ArraysCache",
            },
        ]

        try:
            ordinary_table = prefix_cache.store_cache(
                "ordinary-longer-prompt",
                list(range(12)),
                extracted_cache,
                model_cache_config=model_cache_config,
                hot_cache_write_back=False,
            )
            assert ordinary_table is not None

            exact_table = prefix_cache.store_exact_prefix(
                "aligned-static-prefix",
                list(range(8)),
                extracted_cache,
                model_cache_config=model_cache_config,
            )
            assert exact_table is not None
            prefix_cache.clear_request_entry("aligned-static-prefix")

            restored_table = prefix_cache.fetch_exact_prefix(
                "aligned-static-restore",
                list(range(8)),
            )
            assert restored_table is not None
            restored_cache = prefix_cache.reconstruct_cache(
                restored_table,
                promote_to_hot_cache=False,
            )
            assert restored_cache is not None
            assert restored_cache[0].state[0].shape[2] == 8
            restored_arrays_cache = restored_cache[1]
            restored_arrays_cache[0] = mx.zeros((1, 2), dtype=mx.float32)
            assert mx.array_equal(
                restored_arrays_cache[0],
                mx.zeros((1, 2), dtype=mx.float32),
            ).item()
            assert mx.array_equal(
                restored_arrays_cache[1],
                second_recurrent_state,
            ).item()
        finally:
            ssd_manager.close()

    def test_exact_prefix_missing_terminal_is_a_complete_miss(self, tmp_path):
        import mlx.core as mx

        prefix_cache, paged_manager, ssd_manager = self._make_ssd_prefix_cache(
            tmp_path / "missing-exact-terminal"
        )
        tokens = list(range(7))
        keys = mx.ones((1, 1, 7, 1))
        extracted_cache = [
            {
                "state": (keys, keys),
                "meta_state": (7,),
                "class_name": "KVCache",
                "cache_type": "KVCache",
            }
        ]

        try:
            assert (
                prefix_cache.store_exact_prefix(
                    "stored-exact-prefix", tokens, extracted_cache
                )
                is not None
            )
            terminal_hash = prefix_cache._exact_prefix_hashes(tokens)[-1][0]
            assert ssd_manager.delete_block(terminal_hash)

            assert (
                prefix_cache.restore_exact_prefix(
                    "missing-terminal-restore",
                    tokens,
                    promote_to_hot_cache=False,
                )
                is None
            )
            assert "missing-terminal-restore" not in paged_manager.request_tables
            assert prefix_cache.get_stats().exact_prefix_misses == 1
        finally:
            ssd_manager.close()

    def test_exact_prefix_uses_global_hot_tier_without_requiring_it(self, tmp_path):
        import mlx.core as mx

        prefix_cache, _, ssd_manager = self._make_ssd_prefix_cache(
            tmp_path / "exact-prefix-hot-tier",
            hot_cache_max_bytes=10 * 1024**2,
        )
        tokens = list(range(7))
        keys = mx.ones((1, 1, 7, 1))
        extracted_cache = [
            {
                "state": (keys, keys),
                "meta_state": (7,),
                "class_name": "KVCache",
                "cache_type": "KVCache",
            }
        ]

        try:
            assert (
                prefix_cache.store_exact_prefix(
                    "write-through-static-prefix", tokens, extracted_cache
                )
                is not None
            )
            assert ssd_manager.get_stats().hot_cache_entries == 0

            assert (
                prefix_cache.restore_exact_prefix(
                    "hot-promoting-restore",
                    tokens,
                    promote_to_hot_cache=True,
                )
                is not None
            )
            assert ssd_manager.get_stats().hot_cache_entries > 0

            ssd_manager.clear_hot_cache()
            assert (
                prefix_cache.restore_exact_prefix(
                    "pressure-bypass-restore",
                    tokens,
                    promote_to_hot_cache=False,
                )
                is not None
            )
            assert ssd_manager.get_stats().hot_cache_entries == 0
        finally:
            ssd_manager.close()

    def test_exact_prefix_honors_global_hot_cache_only_mode(self, tmp_path):
        import mlx.core as mx

        prefix_cache, _, ssd_manager = self._make_ssd_prefix_cache(
            tmp_path / "unused-hot-only-cache",
            hot_cache_max_bytes=10 * 1024**2,
            hot_cache_only=True,
        )
        tokens = list(range(7))
        keys = mx.ones((1, 1, 7, 1))
        extracted_cache = [
            {
                "state": (keys, keys),
                "meta_state": (7,),
                "class_name": "KVCache",
                "cache_type": "KVCache",
            }
        ]

        try:
            assert (
                prefix_cache.store_exact_prefix(
                    "hot-only-static-prefix", tokens, extracted_cache
                )
                is not None
            )
            assert ssd_manager.get_stats().hot_cache_entries > 0
            assert (
                prefix_cache.restore_exact_prefix(
                    "hot-only-restore",
                    tokens,
                    promote_to_hot_cache=True,
                )
                is not None
            )
        finally:
            ssd_manager.close()

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

    def test_qwen4_qsa_misaligned_auxiliary_state_is_not_stored(
        self, prefix_cache_with_ssd
    ):
        """A QSA block with shorter indexer state must fail closed."""
        mx = pytest.importorskip("mlx.core")
        keys = mx.zeros((1, 2, 8, 64))
        values = mx.zeros((1, 2, 8, 64))
        index_keys = mx.zeros((1, 4, 64))
        positions = mx.zeros((1, 1, 4), dtype=mx.int32)

        block = prefix_cache_with_ssd._extract_block_tensor_slice(
            [
                {
                    "state": (keys, values, index_keys, positions),
                    "cache_type": "QSAKVCache",
                }
            ],
            4,
            8,
        )

        assert block is None

    def test_qwen4_qsa_sliceable_nstate_blocks_round_trip(self):
        """Valid QSA blocks preserve K/V, index keys, and positions."""
        mx = pytest.importorskip("mlx.core")
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

        keys = mx.arange(1 * 2 * 8 * 64, dtype=mx.float32).reshape(1, 2, 8, 64)
        values = keys + 1_000
        index_keys = mx.arange(1 * 8 * 64, dtype=mx.float32).reshape(1, 8, 64)
        positions = mx.arange(8, dtype=mx.int32).reshape(1, 1, 8)
        mx.eval(keys, values, index_keys, positions)
        cache_data = [
            {
                "state": (keys, values, index_keys, positions),
                "cache_type": "QSAKVCache",
                "class_name": "QSAKVCache",
                "meta_state": (),
            }
        ]

        block_table = cache.store_cache("req-qsa", list(range(8)), cache_data)

        assert block_table is not None
        assert len(block_table.block_ids) == 2

        def load_block_with_metadata(block_hash, **kwargs):
            return saved_by_hash.get(block_hash, (None, None))

        mock_ssd.load_block_with_metadata.side_effect = load_block_with_metadata
        restored = cache.reconstruct_cache(block_table)

        assert restored is not None
        restored_state = restored[0].state
        assert restored_state[0].tolist() == keys.tolist()
        assert restored_state[1].tolist() == values.tolist()
        assert restored_state[2].tolist() == index_keys.tolist()
        assert restored_state[3].tolist() == positions[:, 0, :].tolist()


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
        prefix_cache._update_prefix_index(tokens, [block.block_id for block in blocks])
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

        prefix_cache._forget_incompatible_ssd_block(block.block_hash, block.block_id)

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
        prefix_cache._update_prefix_index(tokens, [block.block_id for block in blocks])
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

    def test_extract_block_arrays_cache_last_block_without_snapshot_is_placeholder(
        self, prefix_cache, mx
    ):
        """A1 fix: is_last_block alone no longer justifies storing live
        ArraysCache state -- store_cache's last FULL block may sit behind
        trailing tokens the live state has already ingested (skipped
        partial block), so a missing boundary snapshot falls to the
        placeholder even on the last block. See
        docs/qwen35-hardening-and-optimization.md A1."""
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
        assert keys.shape == (1,)
        assert values.shape == (1,)

    def test_extract_block_arrays_cache_last_block_with_snapshot_stores_it(
        self, prefix_cache, mx
    ):
        """With a matching boundary snapshot, the last block stores real
        state -- sourced from the snapshot, never from live cache_data."""
        conv_state = mx.ones((1, 3, 64))
        ssm_state = mx.ones((1, 32, 128, 128))
        cache_data = [
            {
                "state": (conv_state, ssm_state),
                "cache_type": "ArraysCache",
                "class_name": "ArraysCache",
            },
        ]
        snapshot_conv_state = mx.zeros((1, 3, 64))
        snapshot_ssm_state = mx.zeros((1, 32, 128, 128))
        snapshot_cache_data = [
            {"state": (snapshot_conv_state, snapshot_ssm_state)},
        ]

        result = prefix_cache._extract_block_tensor_slice(
            cache_data,
            0,
            4,
            model_cache_config=None,
            is_last_block=True,
            snapshot_cache_data=snapshot_cache_data,
        )

        assert result is not None
        keys, values = result[0]
        assert keys.shape == (1, 3, 64)
        assert values.shape == (1, 32, 128, 128)
        assert bool((keys == snapshot_conv_state).all().item())

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
        # ArraysCache layer: placeholder, no snapshot provided (A1 fix --
        # is_last_block alone is not enough, see the test above)
        assert result[1][0].shape == (1,)

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

    def test_store_cache_arrayscache_partial_trailing_stores_placeholder(self, mx):
        """A1 regression: 7 tokens / block_size=4 stores 1 full block (4
        tokens) and skips 3 trailing partial tokens -- but the ArraysCache
        conv/ssm state passed in is the LIVE state after all 7 tokens, not
        just the first 4. Storing it as this block's boundary state would
        make a later restore replay tokens 4-6 against a state that has
        already seen them (silent GDN corruption). Without a boundary
        snapshot, this must store a placeholder instead.
        See docs/qwen35-hardening-and-optimization.md A1."""
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
        assert saved_conv_state.shape == (1,)  # placeholder, not live state
        assert saved_ssm_state.shape == (1,)

    def test_store_cache_arrayscache_exact_multiple_stores_live_state(self, mx):
        """A1 companion case: 8 tokens / block_size=4 has NO trailing
        partial tokens skipped -- the live ArraysCache state genuinely IS
        this (only, last) block's boundary state, so live_state_at_true_end
        lets it be stored directly instead of a placeholder. Distinguishes
        this safe case from the unsafe partial-trailing case above.
        See docs/qwen35-hardening-and-optimization.md A1."""
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

        # 8 tokens = 2 full blocks (4+4), no trailing partial.
        tokens = list(range(8))
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
            "req-002",
            tokens,
            cache_data,
            model_cache_config=model_cache_config,
        )

        assert result is not None
        assert len(result.block_ids) == 2
        assert result.num_tokens == 8
        assert mock_ssd.save_block.call_count == 2

        # Block 0 (non-last): placeholder regardless.
        block0_data = mock_ssd.save_block.call_args_list[0].kwargs["cache_data"]
        assert block0_data[0][0].shape == (1,)

        # Block 1 (last, no trailing partial skipped): live state stored.
        block1_data = mock_ssd.save_block.call_args_list[1].kwargs["cache_data"]
        saved_conv_state, saved_ssm_state = block1_data[0]
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

    def test_store_cache_degrades_gracefully_when_out_of_blocks(self, mx, caplog):
        """F3: handle_memory_pressure/evict_lru_blocks were removed --
        they only ever recycled blocks already in the free queue (ref_count
        == 0), never anything held by a live request, so they could never
        gain capacity beyond what a bare retry already sees. Confirm the
        simplified retry-once-then-give-up path still degrades gracefully
        (keeps the valid prefix, logs a warning, doesn't raise) when
        allocation genuinely has nothing left to give.
        See docs/qwen35-hardening-and-optimization.md F3."""
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

        # 8 tokens = 2 full blocks; force the underlying allocator to
        # simulate genuinely running out of blocks after the first one.
        tokens = list(range(8))
        keys = mx.arange(8, dtype=mx.float32).reshape(1, 1, 8, 1)
        values = (keys + 100).astype(mx.float32)
        cache_data = [
            {"state": (keys, values), "cache_type": "KVCache", "class_name": "KVCache"}
        ]

        real_allocate = paged_cache.allocate_block
        call_index = {"n": 0}

        def flaky_allocate():
            call_index["n"] += 1
            if call_index["n"] > 1:
                return None
            return real_allocate()

        with patch.object(paged_cache, "allocate_block", flaky_allocate):
            with caplog.at_level(logging.WARNING):
                result = cache.store_cache("req-out-of-blocks", tokens, cache_data)

        assert result is not None
        assert len(result.block_ids) == 1
        assert result.num_tokens == 4
        assert any(
            "Cannot allocate block for req-out-of-blocks" in r.getMessage()
            for r in caplog.records
        )

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

    @pytest.mark.parametrize("truncated_side", ["keys", "values"])
    def test_store_cache_rejects_block_when_either_kv_slice_length_disagrees(
        self, mx, truncated_side
    ):
        """A3: a bug anywhere upstream of _extract_block_tensor_slice (wrong
        start/end indices, a stale cache_data snapshot, etc.) could silently
        produce a sliced tensor whose sequence length doesn't match the
        block's own token_count -- persisting that would corrupt a later
        restore with no error until it's replayed. Simulate such a bug by
        monkeypatching _extract_block_tensor_slice to return a 3-token
        slice on either side of the KV pair for what store_cache believes is
        a 4-token block, and confirm the block is rejected (not persisted)
        while the valid prefix before it survives -- the same discipline
        test_store_cache_keeps_valid_prefix_when_later_ssd_save_fails already
        establishes for SSD write failures.
        See docs/qwen35-hardening-and-optimization.md A3."""
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

        tokens = list(range(8))  # 2 full blocks
        keys = mx.arange(8, dtype=mx.float32).reshape(1, 1, 8, 1)
        values = (keys + 100).astype(mx.float32)
        cache_data = [
            {"state": (keys, values), "cache_type": "KVCache", "class_name": "KVCache"}
        ]

        original_extract = cache._extract_block_tensor_slice
        call_index = {"n": 0}

        def buggy_extract(*args, **kwargs):
            call_index["n"] += 1
            result = original_extract(*args, **kwargs)
            if call_index["n"] == 2:
                # Simulate an upstream indexing bug on the second block:
                # truncate one side to 3 tokens instead of the real 4.
                bad_keys, bad_values = result[0]
                if truncated_side == "keys":
                    bad_keys = bad_keys[:, :, :3, :]
                else:
                    bad_values = bad_values[:, :, :3, :]
                result = [(bad_keys, bad_values)]
            return result

        with patch.object(cache, "_extract_block_tensor_slice", buggy_extract):
            result = cache.store_cache("req-mismatch", tokens, cache_data)

        assert result is not None
        # Only the first (correct) block is kept; the mismatched second
        # block is rejected, not persisted.
        assert len(result.block_ids) == 1
        assert result.num_tokens == 4
        assert mock_ssd.save_block.call_count == 1
        saved_keys, saved_values = mock_ssd.save_block.call_args.kwargs["cache_data"][0]
        assert saved_keys.tolist() == keys[:, :, 0:4, :].tolist()
        assert saved_values.tolist() == values[:, :, 0:4, :].tolist()

        # The rejected block must not linger as an allocated/hashed block.
        allocated_non_null_ids = {
            block.block_id
            for block in paged_cache.allocated_blocks.values()
            if not block.is_null
        }
        assert allocated_non_null_ids == set(result.block_ids)

    def test_store_cache_rotating_window_state_passes_token_count_check(self, mx):
        """The A3 length check must not reject non-sliceable layers whose
        real state length is legitimately unrelated to token_count: the
        aligned block size is a MULTIPLE of the rotating window (e.g.
        window 128 -> block 512), so a rotating layer's last-block window
        state is shorter than the block's token span by design."""
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

        # 8 tokens = 2 full blocks, no trailing partial (live state at true
        # end). Window 2, block 4 = 2x window, as production alignment
        # produces.
        tokens = list(range(8))
        window = 2
        rot_keys = mx.ones((1, 8, window, 64))
        rot_values = mx.ones((1, 8, window, 64))
        cache_data = [
            {
                "state": (rot_keys, rot_values),
                "cache_type": "RotatingKVCache",
                "class_name": "RotatingKVCache",
                "meta_state": (0, window, 8, 0),
            }
        ]
        model_cache_config = ModelCacheConfig.from_type_list(
            ["RotatingKVCache"], model_name="test-model"
        )

        result = cache.store_cache(
            "req-rot",
            tokens,
            cache_data,
            model_cache_config=model_cache_config,
        )

        assert result is not None
        # Both blocks survive: block 0 = placeholder, block 1 (last) = the
        # real rotating window state, its window-length shape intact.
        assert len(result.block_ids) == 2
        assert result.num_tokens == 8
        assert mock_ssd.save_block.call_count == 2
        block1_data = mock_ssd.save_block.call_args_list[1].kwargs["cache_data"]
        saved_keys, _saved_values = block1_data[0]
        assert saved_keys.shape == (1, 8, window, 64)

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
        shared_block_ids, _, _ = paged_cache.find_shared_prefix(tokens)
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
        assert (
            paged_cache.cached_block_hash_to_block.get_block(blocks[1].block_hash)
            is None
        )

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

    def test_reconstruct_preserves_variable_length_arrays_state(self, mx):
        """N-tuple ArraysCache markers restore every recurrent state slot."""
        from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
        from omlx.cache.type_handlers import SizedArraysCache

        mock_ssd = MagicMock(spec=PagedSSDCacheManager)
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
        block.block_hash = b"arrays-four-state"
        block.token_count = 4
        block.ref_count = 2
        paged_cache.cached_block_hash_to_block.insert(block.block_hash, block)
        block_table = BlockTable(
            request_id="req-arrays-four-state",
            block_ids=[block.block_id],
            num_tokens=4,
        )
        states = [mx.full((1, 2, 3), i) for i in range(4)]
        mock_ssd.load_block_with_metadata.return_value = (
            [("__nstate__", "ArraysCache", states)],
            {
                "model_name": "test-model",
                "num_layers": 1,
                "block_size": 4,
                "layer_cache_types": ["ArraysCache"],
                "layer_meta_states": [()],
            },
        )

        result = cache.reconstruct_cache(block_table)

        assert result is not None
        assert isinstance(result[0], SizedArraysCache)
        assert result[0].size() == 4
        assert len(result[0].state) == 4
        for expected, actual in zip(states, result[0].state):
            assert mx.array_equal(expected, actual).item()


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
            ([self._tq_block_payload(tq_mod, ks, vs, i)], tq_metadata) for i in range(3)
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
        full_values = mx.random.normal((1, self.HEADS, 2 * self.BLOCK, self.HDIM))
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
        full_values = mx.random.normal((1, self.HEADS, 2 * self.BLOCK, self.HDIM))
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
            mx.random.normal((1, self.HEADS, self.BLOCK, self.HDIM)).astype(mx.float16),
            mx.random.normal((1, self.HEADS, self.BLOCK, self.HDIM)).astype(mx.float16),
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
