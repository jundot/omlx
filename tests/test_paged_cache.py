# SPDX-License-Identifier: Apache-2.0
"""
Tests for PagedCacheManager and related components.

This module tests the block-based paged KV cache management following vLLM's
architecture, adapted for MLX on Apple Silicon.
"""

import time
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from omlx.cache.paged_cache import (
    BlockHash,
    BlockHashToBlockMap,
    BlockTable,
    CacheBlock,
    FreeKVCacheBlockQueue,
    PagedCacheManager,
    compute_block_hash,
)


class TestComputeBlockHash:
    """Tests for compute_block_hash function."""

    def test_same_tokens_same_hash(self):
        """Same tokens with same parent should produce same hash."""
        tokens = [1, 2, 3, 4]
        hash1 = compute_block_hash(None, tokens)
        hash2 = compute_block_hash(None, tokens)
        assert hash1 == hash2

    def test_different_tokens_different_hash(self):
        """Different tokens should produce different hashes."""
        hash1 = compute_block_hash(None, [1, 2, 3, 4])
        hash2 = compute_block_hash(None, [5, 6, 7, 8])
        assert hash1 != hash2

    def test_parent_hash_affects_result(self):
        """Different parent hashes should produce different results."""
        tokens = [1, 2, 3, 4]
        parent1 = b"parent1_hash_bytes"
        parent2 = b"parent2_hash_bytes"
        hash1 = compute_block_hash(BlockHash(parent1), tokens)
        hash2 = compute_block_hash(BlockHash(parent2), tokens)
        assert hash1 != hash2

    def test_model_name_isolation(self):
        """Different model names should produce different hashes."""
        tokens = [1, 2, 3, 4]
        hash1 = compute_block_hash(None, tokens, model_name="model_a")
        hash2 = compute_block_hash(None, tokens, model_name="model_b")
        assert hash1 != hash2

    def test_extra_keys_affect_hash(self):
        """Extra keys should affect the resulting hash."""
        tokens = [1, 2, 3, 4]
        hash1 = compute_block_hash(None, tokens)
        hash2 = compute_block_hash(None, tokens, extra_keys=("lora_1",))
        assert hash1 != hash2

    def test_hash_is_bytes(self):
        """Hash should be bytes type."""
        hash_val = compute_block_hash(None, [1, 2, 3])
        assert isinstance(hash_val, bytes)

    def test_hash_length(self):
        """Hash should be 32 bytes (SHA256)."""
        hash_val = compute_block_hash(None, [1, 2, 3])
        assert len(hash_val) == 32

    def test_chain_hash(self):
        """Test chain hashing (each block depends on previous)."""
        # Block 1
        hash1 = compute_block_hash(None, [1, 2, 3, 4])
        # Block 2 depends on Block 1
        hash2 = compute_block_hash(hash1, [5, 6, 7, 8])
        # Block 3 depends on Block 2
        hash3 = compute_block_hash(hash2, [9, 10, 11, 12])

        # Each hash should be unique
        assert len({hash1, hash2, hash3}) == 3


class TestCacheBlock:
    """Tests for CacheBlock dataclass."""

    def test_default_values(self):
        """Test default values on initialization."""
        block = CacheBlock(block_id=0)
        assert block.block_id == 0
        assert block.ref_count == 0
        assert block.block_hash is None
        assert block.prev_free_block is None
        assert block.next_free_block is None
        assert block.is_null is False
        assert block.token_count == 0

    def test_is_full(self):
        """Test is_full method."""
        block = CacheBlock(block_id=0, token_count=32)
        assert not block.is_full(64)
        assert block.is_full(32)
        assert block.is_full(16)

    def test_is_shared(self):
        """Test is_shared method."""
        block = CacheBlock(block_id=0, ref_count=1)
        assert not block.is_shared()

        block.ref_count = 2
        assert block.is_shared()

        block.ref_count = 5
        assert block.is_shared()

    def test_reset_hash(self):
        """Test reset_hash method."""
        block = CacheBlock(block_id=0)
        block.block_hash = BlockHash(b"test_hash")
        assert block.block_hash is not None

        block.reset_hash()
        assert block.block_hash is None

    def test_touch(self):
        """Test touch method updates last_access."""
        block = CacheBlock(block_id=0)
        old_access = block.last_access

        time.sleep(0.01)
        block.touch()

        assert block.last_access > old_access

    def test_repr(self):
        """Test string representation."""
        block = CacheBlock(block_id=5, ref_count=2, token_count=32)
        repr_str = repr(block)
        assert "id=5" in repr_str
        assert "ref=2" in repr_str
        assert "tokens=32" in repr_str


class TestFreeKVCacheBlockQueue:
    """Tests for FreeKVCacheBlockQueue (doubly linked list)."""

    def test_empty_queue(self):
        """Test creating queue with no blocks."""
        queue = FreeKVCacheBlockQueue([])
        assert queue.num_free_blocks == 0

    def test_initial_state(self):
        """Test queue initialization with blocks."""
        blocks = [CacheBlock(block_id=i) for i in range(5)]
        queue = FreeKVCacheBlockQueue(blocks)
        assert queue.num_free_blocks == 5

    def test_popleft(self):
        """Test popping from front (LRU order)."""
        blocks = [CacheBlock(block_id=i) for i in range(5)]
        queue = FreeKVCacheBlockQueue(blocks)

        popped = queue.popleft()
        assert popped.block_id == 0
        assert queue.num_free_blocks == 4

        popped = queue.popleft()
        assert popped.block_id == 1
        assert queue.num_free_blocks == 3

    def test_popleft_empty_raises(self):
        """Test popleft on empty queue raises ValueError."""
        queue = FreeKVCacheBlockQueue([])
        with pytest.raises(ValueError, match="No free blocks"):
            queue.popleft()

    def test_popleft_n(self):
        """Test popping multiple blocks at once."""
        blocks = [CacheBlock(block_id=i) for i in range(5)]
        queue = FreeKVCacheBlockQueue(blocks)

        popped = queue.popleft_n(3)
        assert len(popped) == 3
        assert [b.block_id for b in popped] == [0, 1, 2]
        assert queue.num_free_blocks == 2

    def test_popleft_n_zero(self):
        """Test popping zero blocks returns empty list."""
        blocks = [CacheBlock(block_id=i) for i in range(5)]
        queue = FreeKVCacheBlockQueue(blocks)

        popped = queue.popleft_n(0)
        assert popped == []
        assert queue.num_free_blocks == 5

    def test_popleft_n_insufficient_raises(self):
        """Test popping more blocks than available raises."""
        blocks = [CacheBlock(block_id=i) for i in range(3)]
        queue = FreeKVCacheBlockQueue(blocks)

        with pytest.raises(AssertionError):
            queue.popleft_n(5)

    def test_append(self):
        """Test appending a block to end (MRU position)."""
        blocks = [CacheBlock(block_id=i) for i in range(3)]
        queue = FreeKVCacheBlockQueue(blocks)

        # Pop all
        queue.popleft_n(3)
        assert queue.num_free_blocks == 0

        # Append a block
        new_block = CacheBlock(block_id=10)
        queue.append(new_block)
        assert queue.num_free_blocks == 1

        # Pop should return the appended block
        popped = queue.popleft()
        assert popped.block_id == 10

    def test_append_n(self):
        """Test appending multiple blocks."""
        blocks = [CacheBlock(block_id=i) for i in range(2)]
        queue = FreeKVCacheBlockQueue(blocks)

        # Append multiple
        new_blocks = [CacheBlock(block_id=i) for i in range(10, 13)]
        queue.append_n(new_blocks)

        assert queue.num_free_blocks == 5

    def test_remove(self):
        """Test removing a block from the middle."""
        blocks = [CacheBlock(block_id=i) for i in range(5)]
        queue = FreeKVCacheBlockQueue(blocks)

        # Remove block 2 from middle
        queue.remove(blocks[2])
        assert queue.num_free_blocks == 4

        # Verify block 2 is no longer in queue
        all_free = queue.get_all_free_blocks()
        assert blocks[2] not in all_free
        assert len(all_free) == 4

    def test_remove_not_in_queue_raises(self):
        """Test removing a block not in queue raises."""
        blocks = [CacheBlock(block_id=i) for i in range(5)]
        queue = FreeKVCacheBlockQueue(blocks)

        # Pop a block (removes it from queue)
        popped = queue.popleft()

        # Try to remove again
        with pytest.raises(RuntimeError, match="not in free queue"):
            queue.remove(popped)

    def test_lru_order_maintained(self):
        """Test LRU order: front is LRU, back is MRU."""
        blocks = [CacheBlock(block_id=i) for i in range(5)]
        queue = FreeKVCacheBlockQueue(blocks)

        # Pop 2, append them back in reverse order
        b0 = queue.popleft()
        b1 = queue.popleft()

        queue.append(b1)
        queue.append(b0)

        # Now order should be: 2, 3, 4, 1, 0
        all_free = queue.get_all_free_blocks()
        ids = [b.block_id for b in all_free]
        assert ids == [2, 3, 4, 1, 0]

    def test_get_all_free_blocks(self):
        """Test getting all free blocks."""
        blocks = [CacheBlock(block_id=i) for i in range(5)]
        queue = FreeKVCacheBlockQueue(blocks)

        all_free = queue.get_all_free_blocks()
        assert len(all_free) == 5
        assert [b.block_id for b in all_free] == [0, 1, 2, 3, 4]


class TestBlockHashToBlockMap:
    """Tests for BlockHashToBlockMap (hash-based prefix cache)."""

    def test_empty_map(self):
        """Test empty map."""
        cache = BlockHashToBlockMap()
        assert len(cache) == 0
        assert cache.get_block(BlockHash(b"nonexistent")) is None

    def test_insert_and_get(self):
        """Test inserting and retrieving a block."""
        cache = BlockHashToBlockMap()
        block = CacheBlock(block_id=1)
        block_hash = BlockHash(b"test_hash_bytes_1234")

        cache.insert(block_hash, block)
        assert len(cache) == 1

        retrieved = cache.get_block(block_hash)
        assert retrieved is block

    def test_insert_same_hash_multiple_blocks(self):
        """Test inserting multiple blocks with same hash (hybrid models)."""
        cache = BlockHashToBlockMap()
        block1 = CacheBlock(block_id=1)
        block2 = CacheBlock(block_id=2)
        block_hash = BlockHash(b"shared_hash_bytes_12")

        cache.insert(block_hash, block1)
        cache.insert(block_hash, block2)

        # get_block returns any block with that hash
        retrieved = cache.get_block(block_hash)
        assert retrieved in (block1, block2)

    def test_pop(self):
        """Test popping a specific block."""
        cache = BlockHashToBlockMap()
        block = CacheBlock(block_id=5)
        block_hash = BlockHash(b"test_hash_for_pop_12")

        cache.insert(block_hash, block)
        assert len(cache) == 1

        popped = cache.pop(block_hash, block.block_id)
        assert popped is block
        assert len(cache) == 0

    def test_pop_wrong_block_id(self):
        """Test popping with wrong block_id returns None."""
        cache = BlockHashToBlockMap()
        block = CacheBlock(block_id=5)
        block_hash = BlockHash(b"test_hash_wrong_id_1")

        cache.insert(block_hash, block)

        # Try to pop with wrong block_id
        popped = cache.pop(block_hash, 999)
        assert popped is None
        # Original block should still be in cache
        assert len(cache) == 1

    def test_pop_from_multiple(self):
        """Test popping one block when multiple exist for same hash."""
        cache = BlockHashToBlockMap()
        block1 = CacheBlock(block_id=1)
        block2 = CacheBlock(block_id=2)
        block_hash = BlockHash(b"multi_block_hash_123")

        cache.insert(block_hash, block1)
        cache.insert(block_hash, block2)

        # Pop block1
        popped = cache.pop(block_hash, 1)
        assert popped is block1

        # block2 should still be retrievable
        retrieved = cache.get_block(block_hash)
        assert retrieved is block2

    def test_clear(self):
        """Test clearing the map."""
        cache = BlockHashToBlockMap()
        for i in range(5):
            block = CacheBlock(block_id=i)
            cache.insert(BlockHash(f"hash_{i}_bytes_padding".encode()), block)

        assert len(cache) == 5

        cache.clear()
        assert len(cache) == 0


class TestBlockTable:
    """Tests for BlockTable (per-request block mapping)."""

    def test_default_values(self):
        """Test default values."""
        table = BlockTable(request_id="req-001")
        assert table.request_id == "req-001"
        assert table.block_ids == []
        assert table.num_tokens == 0

    def test_add_block(self):
        """Test adding a block."""
        table = BlockTable(request_id="req-001")
        table.add_block(block_id=5, num_tokens=64)

        assert table.block_ids == [5]
        assert table.num_tokens == 64

        table.add_block(block_id=10, num_tokens=32)
        assert table.block_ids == [5, 10]
        assert table.num_tokens == 96

    def test_len(self):
        """Test length (number of blocks)."""
        table = BlockTable(request_id="req-001", block_ids=[1, 2, 3])
        assert len(table) == 3

    def test_copy(self):
        """Test copying with new request ID."""
        table = BlockTable(
            request_id="req-001",
            block_ids=[1, 2, 3],
            num_tokens=192,
        )

        copied = table.copy("req-002")
        assert copied.request_id == "req-002"
        assert copied.block_ids == [1, 2, 3]
        assert copied.num_tokens == 192

        # Ensure it's a deep copy of block_ids
        copied.block_ids.append(4)
        assert table.block_ids == [1, 2, 3]


class TestPagedCacheManager:
    """Tests for PagedCacheManager."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = PagedCacheManager(
            block_size=64,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=50,
        )

        assert manager.block_size == 64
        assert manager.max_blocks == 100
        assert manager.model_name == "test-model"
        # Null block is reserved
        assert manager.free_blocks == 49  # 50 initial - 1 null

    def test_null_block_reserved(self):
        """Test that null block is properly reserved."""
        manager = PagedCacheManager(block_size=64, max_blocks=100, initial_blocks=100)

        assert manager.null_block is not None
        assert manager.null_block.is_null is True
        assert manager.null_block.ref_count == 1
        assert manager.null_block.block_id in manager.allocated_blocks

    def test_allocate_block(self):
        """Test allocating a single block."""
        manager = PagedCacheManager(block_size=64, max_blocks=100, initial_blocks=100)
        initial_free = manager.free_blocks

        block = manager.allocate_block()
        assert block is not None
        assert block.ref_count == 1
        assert block.block_id in manager.allocated_blocks
        assert manager.free_blocks == initial_free - 1

    def test_get_new_blocks(self):
        """Test allocating multiple blocks."""
        manager = PagedCacheManager(block_size=64, max_blocks=100, initial_blocks=100)

        blocks = manager.get_new_blocks(5)
        assert len(blocks) == 5
        for block in blocks:
            assert block.ref_count == 1
            assert block.block_id in manager.allocated_blocks

    def test_free_block(self):
        """Test freeing a block."""
        manager = PagedCacheManager(block_size=64, max_blocks=100, initial_blocks=100)

        block = manager.allocate_block()
        block_id = block.block_id
        initial_free = manager.free_blocks

        result = manager.free_block(block_id)
        assert result is True
        assert manager.free_blocks == initial_free + 1
        assert block_id not in manager.allocated_blocks

    def test_free_block_shared(self):
        """Test freeing a shared block only decrements ref_count."""
        manager = PagedCacheManager(block_size=64, max_blocks=100, initial_blocks=100)

        block = manager.allocate_block()
        block.ref_count = 2  # Simulating shared block
        block_id = block.block_id
        initial_free = manager.free_blocks

        result = manager.free_block(block_id)
        assert result is False  # Not actually freed
        assert block.ref_count == 1
        assert manager.free_blocks == initial_free  # No change

    def test_increment_ref(self):
        """Test incrementing reference count."""
        manager = PagedCacheManager(block_size=64, max_blocks=100, initial_blocks=100)

        block = manager.allocate_block()
        assert block.ref_count == 1

        result = manager.increment_ref(block.block_id)
        assert result is True
        assert block.ref_count == 2

    def test_create_block_table(self):
        """Test creating a block table for a request."""
        manager = PagedCacheManager(block_size=64, max_blocks=100, initial_blocks=100)

        table = manager.create_block_table("req-001")
        assert table.request_id == "req-001"
        assert "req-001" in manager.request_tables

    def test_get_block_table(self):
        """Test getting an existing block table."""
        manager = PagedCacheManager(block_size=64, max_blocks=100, initial_blocks=100)

        manager.create_block_table("req-001")
        table = manager.get_block_table("req-001")
        assert table is not None
        assert table.request_id == "req-001"

        # Non-existent table
        assert manager.get_block_table("req-nonexistent") is None

    def test_delete_block_table(self):
        """Test deleting a block table frees associated blocks."""
        manager = PagedCacheManager(block_size=64, max_blocks=100, initial_blocks=100)

        # Create table and add blocks
        table = manager.create_block_table("req-001")
        blocks = manager.get_new_blocks(3)
        for block in blocks:
            table.block_ids.append(block.block_id)

        initial_free = manager.free_blocks

        # Delete table
        manager.delete_block_table("req-001")
        assert "req-001" not in manager.request_tables
        assert manager.free_blocks == initial_free + 3

    def test_find_shared_prefix(self):
        """Test finding shared prefix blocks."""
        manager = PagedCacheManager(block_size=4, max_blocks=100, initial_blocks=100)

        # Create and cache some blocks
        tokens1 = [1, 2, 3, 4, 5, 6, 7, 8]  # 2 full blocks
        blocks = manager.get_new_blocks(2)

        # Cache the blocks with their hashes
        parent_hash = None
        for i, block in enumerate(blocks):
            start = i * 4
            end = start + 4
            block_tokens = tokens1[start:end]
            manager.register_block_hash(block, block_tokens, parent_hash)
            parent_hash = block.block_hash

        # Find shared prefix for same tokens
        shared_ids, remaining = manager.find_shared_prefix(tokens1)
        assert len(shared_ids) == 2
        assert remaining == []

    def test_fork_block_table(self):
        """Test forking a block table (COW)."""
        manager = PagedCacheManager(block_size=64, max_blocks=100, initial_blocks=100)

        # Create source table with blocks
        source_table = manager.create_block_table("req-source")
        blocks = manager.get_new_blocks(2)
        for block in blocks:
            source_table.block_ids.append(block.block_id)

        # Fork
        forked = manager.fork_block_table(source_table, "req-forked")

        assert forked.request_id == "req-forked"
        assert forked.block_ids == source_table.block_ids

        # Ref counts should be incremented
        for block_id in source_table.block_ids:
            block = manager.allocated_blocks[block_id]
            assert block.ref_count == 2

    def test_evict_block_permanently(self):
        """Test permanent block eviction."""
        manager = PagedCacheManager(block_size=64, max_blocks=100, initial_blocks=100)

        block = manager.allocate_block()
        block_id = block.block_id
        block.ref_count = 0  # Make evictable
        block.block_hash = BlockHash(b"test_hash_for_evict")
        manager.cached_block_hash_to_block.insert(block.block_hash, block)

        initial_free = manager.free_blocks

        result = manager.evict_block_permanently(block_id)
        assert result is True
        assert block_id not in manager.allocated_blocks
        assert manager.free_blocks == initial_free + 1

    def test_evict_block_in_use_fails(self):
        """Test evicting block in use fails."""
        manager = PagedCacheManager(block_size=64, max_blocks=100, initial_blocks=100)

        block = manager.allocate_block()
        block_id = block.block_id
        # block.ref_count is already 1 (in use)

        result = manager.evict_block_permanently(block_id)
        assert result is False
        assert block_id in manager.allocated_blocks

    def test_dynamic_block_growth(self):
        """Test dynamic block pool growth."""
        manager = PagedCacheManager(
            block_size=64,
            max_blocks=100,
            initial_blocks=10,
        )

        # Initially should have 10 blocks (9 free after null block)
        assert manager._current_allocated_count == 10
        assert manager.free_blocks == 9

        # Allocate more than initial - should grow
        blocks = manager.get_new_blocks(15)
        assert len(blocks) == 15

        # Pool should have grown
        assert manager._current_allocated_count > 10

    def test_stats(self):
        """Test cache statistics."""
        manager = PagedCacheManager(block_size=64, max_blocks=100, initial_blocks=100)

        # Initial stats
        stats = manager.get_stats()
        assert stats.allocated_blocks == 1  # null block
        assert stats.free_blocks == 99

        # Allocate some blocks
        manager.get_new_blocks(5)
        stats = manager.get_stats()
        assert stats.allocated_blocks == 6

    def test_clear(self):
        """Test clearing all cache data."""
        manager = PagedCacheManager(
            block_size=64,
            max_blocks=100,
            initial_blocks=50,
        )

        # Allocate blocks and create tables
        manager.get_new_blocks(10)
        manager.create_block_table("req-001")

        cleared = manager.clear()
        assert cleared > 0

        # After clear, should be reset to initial state
        assert len(manager.request_tables) == 0
        # Only null block should be allocated
        assert len(manager.allocated_blocks) == 1

    def test_usage_property(self):
        """Test usage ratio property."""
        manager = PagedCacheManager(block_size=64, max_blocks=10, initial_blocks=10)

        # Initially low usage (only null block)
        assert manager.usage < 0.2

        # Allocate more blocks
        manager.get_new_blocks(5)
        # Usage should increase
        assert manager.usage > 0.5

    def test_cache_manager_interface(self):
        """Test CacheManager ABC interface implementation."""
        manager = PagedCacheManager(block_size=64, max_blocks=100, initial_blocks=100)

        # Test fetch (by hash)
        block = manager.allocate_block()
        block_hash = BlockHash(b"test_interface_hash1")
        block.block_hash = block_hash
        manager.cached_block_hash_to_block.insert(block_hash, block)

        value, hit = manager.fetch(block_hash)
        assert hit is True
        assert value is block

        # Test fetch miss
        value, hit = manager.fetch(b"nonexistent_hash_byte")
        assert hit is False
        assert value is None

        # Test size and max_size
        assert manager.size >= 0
        assert manager.max_size == 100

    def test_get_evictable_blocks(self):
        """Test getting evictable blocks in LRU order."""
        manager = PagedCacheManager(block_size=64, max_blocks=100, initial_blocks=100)

        # Allocate blocks then free them (add to free queue)
        blocks = manager.get_new_blocks(5)
        for block in blocks:
            block.ref_count = 0

        evictable = manager.get_evictable_blocks(3)
        # Should get blocks from free queue in LRU order
        assert len(evictable) <= 3

    def test_handle_memory_pressure(self):
        """Test handling memory pressure."""
        manager = PagedCacheManager(block_size=64, max_blocks=100, initial_blocks=100)

        # Should return True when enough blocks available
        result = manager.handle_memory_pressure(5)
        assert result is True

    def test_allocate_blocks_for_tokens(self):
        """Test allocating blocks for a given number of tokens."""
        manager = PagedCacheManager(block_size=64, max_blocks=100, initial_blocks=100)

        # Need enough blocks for 150 tokens with block_size=64
        # ceil(150/64) = 3 blocks
        blocks = manager.allocate_blocks_for_tokens(150)
        assert len(blocks) == 3

    def test_get_computed_blocks_ssd_fallback(self):
        """Test that get_computed_blocks falls back to SSD cache on in-memory miss."""
        manager = PagedCacheManager(
            block_size=4, max_blocks=100, model_name="test-model", initial_blocks=100
        )

        tokens = [1, 2, 3, 4, 5, 6, 7, 8]  # 2 full blocks

        # Compute block hashes for these tokens
        hash1 = compute_block_hash(None, [1, 2, 3, 4], model_name="test-model")
        hash2 = compute_block_hash(hash1, [5, 6, 7, 8], model_name="test-model")

        # Mock SSD cache manager that reports both blocks exist
        mock_ssd = MagicMock(spec=[])
        mock_ssd.has_block = MagicMock(side_effect=lambda h: h in (hash1, hash2))
        manager._paged_ssd_cache_manager = mock_ssd

        # In-memory cache is empty, but SSD has the blocks
        cached_blocks, num_tokens = manager.get_computed_blocks(tokens)

        assert num_tokens == 8
        assert len(cached_blocks) == 2
        assert mock_ssd.has_block.call_count == 2

        # Blocks should now be registered in memory
        assert manager.cached_block_hash_to_block.get_block(hash1) is not None
        assert manager.cached_block_hash_to_block.get_block(hash2) is not None

    def test_get_computed_blocks_ssd_fallback_partial(self):
        """Test SSD fallback with partial hit (first block on SSD, second not)."""
        manager = PagedCacheManager(
            block_size=4, max_blocks=100, model_name="test-model", initial_blocks=100
        )

        tokens = [1, 2, 3, 4, 5, 6, 7, 8]

        hash1 = compute_block_hash(None, [1, 2, 3, 4], model_name="test-model")

        # Only first block exists on SSD
        mock_ssd = MagicMock(spec=[])
        mock_ssd.has_block = MagicMock(side_effect=lambda h: h == hash1)
        manager._paged_ssd_cache_manager = mock_ssd

        cached_blocks, num_tokens = manager.get_computed_blocks(tokens)

        assert num_tokens == 4
        assert len(cached_blocks) == 1

    def test_get_computed_blocks_no_ssd_no_regression(self):
        """Test that without SSD cache manager, behavior is unchanged."""
        manager = PagedCacheManager(
            block_size=4, max_blocks=100, model_name="test-model", initial_blocks=100
        )

        tokens = [1, 2, 3, 4, 5, 6, 7, 8]

        # No SSD manager set
        assert manager._paged_ssd_cache_manager is None

        cached_blocks, num_tokens = manager.get_computed_blocks(tokens)

        assert num_tokens == 0
        assert len(cached_blocks) == 0
