# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-mlx (https://github.com/vllm-project/vllm-mlx).
"""
Paged KV Cache Manager for oMLX.

This module implements block-based paged KV cache management following vLLM's
architecture (vllm/v1/core/block_pool.py), adapted for MLX on Apple Silicon.

Key components:
- KVCacheBlock: Metadata for each cache block with doubly linked list pointers
- FreeKVCacheBlockQueue: O(1) doubly linked list for LRU block allocation
- BlockHashToBlockMap: Hash-to-block cache for prefix caching
- PagedCacheManager: Main manager with block allocation, prefix caching, and COW

Features:
- Block-based allocation (configurable tokens per block)
- Reference counting for shared blocks
- Copy-on-Write (COW) for efficient prefix sharing
- LRU eviction using doubly linked list (O(1) operations)
- Chain hashing for prefix caching (hash depends on parent block)

Reference: vLLM v1 - vllm/v1/core/block_pool.py, vllm/v1/core/kv_cache_utils.py
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Dict, List, NewType, Optional, Tuple

from .interface import CacheManager
from .stats import BaseCacheStats, PagedCacheStats

logger = logging.getLogger(__name__)

# Type alias for block hash (content-based hash for prefix caching)
BlockHash = NewType("BlockHash", bytes)


def compute_block_hash(
    parent_hash: Optional[BlockHash],
    token_ids: List[int],
    extra_keys: Optional[Tuple[Any, ...]] = None,
    model_name: Optional[str] = None,
) -> BlockHash:
    """
    Compute hash for a block based on its content and parent block.

    This enables prefix caching by creating a chain of hashes where
    each block's hash depends on all previous blocks (similar to vLLM).

    Args:
        parent_hash: Hash of the previous block, or None for first block
        token_ids: Token IDs in this block
        extra_keys: Additional keys (e.g., LoRA, multimodal)
        model_name: Model name for cache isolation between different models

    Returns:
        Content-based hash for this block
    """
    hasher = hashlib.sha256()

    # Include model name first to isolate caches between different models
    if model_name:
        hasher.update(model_name.encode("utf-8"))

    # Include parent hash for chain
    if parent_hash:
        hasher.update(parent_hash)
    else:
        # Use fixed seed for reproducibility
        hasher.update(b"omlx-root")

    # Include token content
    hasher.update(bytes(str(tuple(token_ids)), "utf-8"))

    # Include extra keys if present
    if extra_keys:
        hasher.update(bytes(str(extra_keys), "utf-8"))

    return BlockHash(hasher.digest())


# =============================================================================
# KVCacheBlock - Following vLLM's design
# =============================================================================

@dataclass
class CacheBlock:
    """
    KV cache block metadata following vLLM's design.

    Each block represents a fixed number of tokens (block_size) worth
    of KV cache data. Blocks can be shared across requests via
    reference counting for prefix caching.

    NOTE: In paged SSD-only mode, blocks do NOT store cache_data in GPU memory.
    All KV cache data is stored on paged SSD via PagedSSDCacheManager, and only
    loaded when needed for inference via BatchGenerator.

    Attributes:
        block_id: Physical block index (0 to num_blocks - 1)
        ref_count: Reference count for sharing (0 = can be evicted)
        block_hash: Content hash for prefix caching and paged SSD storage key
        prev_free_block: Previous block in free list (doubly linked)
        next_free_block: Next block in free list (doubly linked)
        is_null: True if this is the null/placeholder block
        token_count: Number of tokens stored in this block
    """

    block_id: int
    ref_count: int = 0
    block_hash: Optional[BlockHash] = None

    # Doubly linked list pointers for FreeKVCacheBlockQueue
    prev_free_block: Optional["CacheBlock"] = None
    next_free_block: Optional["CacheBlock"] = None

    # Special flags
    is_null: bool = False

    # Metadata
    token_count: int = 0
    last_access: float = field(default_factory=time.time)

    def is_full(self, block_size: int) -> bool:
        """Check if block is at capacity."""
        return self.token_count >= block_size

    def is_shared(self) -> bool:
        """Check if block is shared (ref_count > 1)."""
        return self.ref_count > 1

    def reset_hash(self) -> None:
        """Reset block hash when evicted from cache."""
        self.block_hash = None

    def touch(self) -> None:
        """Update last access time."""
        self.last_access = time.time()

    def __repr__(self) -> str:
        prev_id = self.prev_free_block.block_id if self.prev_free_block else None
        next_id = self.next_free_block.block_id if self.next_free_block else None
        hash_str = f", hash={self.block_hash.hex()[:8]}..." if self.block_hash else ""
        return (
            f"CacheBlock(id={self.block_id}, ref={self.ref_count}, "
            f"tokens={self.token_count}, prev={prev_id}, next={next_id}{hash_str})"
        )


# =============================================================================
# FreeKVCacheBlockQueue - O(1) Doubly Linked List (vLLM style)
# =============================================================================

class FreeKVCacheBlockQueue:
    """
    Doubly linked list of free blocks following vLLM's design.

    Provides O(1) operations for:
    - popleft(): Allocate block from front (LRU order)
    - remove(): Remove block from middle (when touched by cache hit)
    - append(): Return block to end (when freed)

    The queue maintains LRU eviction order:
    - Front = least recently used (evict first)
    - Back = most recently used (evict last)

    Uses fake head/tail sentinels to simplify edge cases.
    """

    def __init__(self, blocks: List[CacheBlock]) -> None:
        """
        Initialize queue with all blocks as free.

        Args:
            blocks: List of all CacheBlock objects
        """
        self.num_free_blocks = len(blocks)

        # Initialize doubly linked list
        for i in range(len(blocks)):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < len(blocks) - 1:
                blocks[i].next_free_block = blocks[i + 1]

        # Create sentinel nodes (never popped)
        self.fake_head = CacheBlock(block_id=-1)
        self.fake_tail = CacheBlock(block_id=-2)

        if blocks:
            self.fake_head.next_free_block = blocks[0]
            blocks[0].prev_free_block = self.fake_head
            self.fake_tail.prev_free_block = blocks[-1]
            blocks[-1].next_free_block = self.fake_tail
        else:
            self.fake_head.next_free_block = self.fake_tail
            self.fake_tail.prev_free_block = self.fake_head

    def popleft(self) -> CacheBlock:
        """
        Pop and return the first (LRU) free block.

        Raises:
            ValueError: If no free blocks available
        """
        if self.fake_head.next_free_block is self.fake_tail:
            raise ValueError("No free blocks available")

        block = self.fake_head.next_free_block
        assert block is not None

        # Remove from list
        self.fake_head.next_free_block = block.next_free_block
        if block.next_free_block:
            block.next_free_block.prev_free_block = self.fake_head

        block.prev_free_block = None
        block.next_free_block = None
        self.num_free_blocks -= 1

        return block

    def popleft_n(self, n: int) -> List[CacheBlock]:
        """
        Pop n blocks from the front.

        Args:
            n: Number of blocks to allocate

        Returns:
            List of n free blocks

        Raises:
            AssertionError: If not enough free blocks
        """
        if n == 0:
            return []

        assert self.num_free_blocks >= n, f"Need {n} blocks, have {self.num_free_blocks}"

        result = []
        curr = self.fake_head.next_free_block

        for _ in range(n):
            assert curr is not None and curr is not self.fake_tail
            result.append(curr)
            last = curr
            curr = curr.next_free_block
            # Clear pointers
            last.prev_free_block = None
            last.next_free_block = None

        # Reconnect list
        self.fake_head.next_free_block = curr
        if curr:
            curr.prev_free_block = self.fake_head

        self.num_free_blocks -= n
        return result

    def remove(self, block: CacheBlock) -> None:
        """
        Remove a block from the middle of the queue.

        Used when a free block is "touched" (reused by prefix cache hit).

        Args:
            block: Block to remove

        Raises:
            RuntimeError: If block not in queue
        """
        if block.prev_free_block is None or block.next_free_block is None:
            raise RuntimeError(f"Block {block.block_id} not in free queue")

        # Unlink
        block.prev_free_block.next_free_block = block.next_free_block
        block.next_free_block.prev_free_block = block.prev_free_block
        block.prev_free_block = None
        block.next_free_block = None

        self.num_free_blocks -= 1

    def append(self, block: CacheBlock) -> None:
        """
        Append a block to the end (MRU position).

        Args:
            block: Block to append
        """
        last = self.fake_tail.prev_free_block
        assert last is not None

        last.next_free_block = block
        block.prev_free_block = last
        block.next_free_block = self.fake_tail
        self.fake_tail.prev_free_block = block

        self.num_free_blocks += 1

    def append_n(self, blocks: List[CacheBlock]) -> None:
        """
        Append multiple blocks to the end.

        Args:
            blocks: Blocks to append (in order)
        """
        if not blocks:
            return

        last = self.fake_tail.prev_free_block
        assert last is not None

        for block in blocks:
            block.prev_free_block = last
            last.next_free_block = block
            last = block

        last.next_free_block = self.fake_tail
        self.fake_tail.prev_free_block = last

        self.num_free_blocks += len(blocks)

    def get_all_free_blocks(self) -> List[CacheBlock]:
        """Get all free blocks (for testing)."""
        result = []
        curr = self.fake_head.next_free_block
        while curr and curr is not self.fake_tail:
            result.append(curr)
            curr = curr.next_free_block
        return result


# =============================================================================
# BlockHashToBlockMap - Hash-based prefix cache (vLLM style)
# =============================================================================

class BlockHashToBlockMap:
    """
    Cache mapping block hashes to blocks for prefix caching.

    Follows vLLM's design where the same hash can map to multiple
    blocks (for different KV cache groups in hybrid models).
    """

    def __init__(self) -> None:
        self._cache: Dict[BlockHash, CacheBlock | Dict[int, CacheBlock]] = {}

    def get_block(self, block_hash: BlockHash) -> Optional[CacheBlock]:
        """Get any block with the given hash."""
        blocks = self._cache.get(block_hash)
        if blocks is None:
            return None
        if isinstance(blocks, CacheBlock):
            return blocks
        if isinstance(blocks, dict):
            return next(iter(blocks.values()))
        return None

    def insert(self, block_hash: BlockHash, block: CacheBlock) -> None:
        """Insert a block into the cache."""
        existing = self._cache.get(block_hash)
        if existing is None:
            self._cache[block_hash] = block
        elif isinstance(existing, CacheBlock):
            self._cache[block_hash] = {
                existing.block_id: existing,
                block.block_id: block,
            }
        elif isinstance(existing, dict):
            existing[block.block_id] = block

    def pop(self, block_hash: BlockHash, block_id: int) -> Optional[CacheBlock]:
        """Remove and return a specific block from the cache."""
        blocks = self._cache.pop(block_hash, None)
        if blocks is None:
            return None

        if isinstance(blocks, CacheBlock):
            if blocks.block_id == block_id:
                return blocks
            # Wrong block ID, put it back
            self._cache[block_hash] = blocks
            return None

        if isinstance(blocks, dict):
            block = blocks.pop(block_id, None)
            if blocks:  # Still has other blocks
                self._cache[block_hash] = blocks
            return block

        return None

    def __len__(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()


# =============================================================================
# BlockTable - Per-request block mapping
# =============================================================================

@dataclass
class BlockTable:
    """
    Per-request block table mapping logical to physical blocks.

    Similar to vLLM's block table, this maps a request's token positions
    to physical cache blocks.

    Attributes:
        request_id: Unique request identifier
        block_ids: List of physical block IDs
        num_tokens: Total number of cached tokens
    """

    request_id: str
    block_ids: List[int] = field(default_factory=list)
    num_tokens: int = 0

    def add_block(self, block_id: int, num_tokens: int) -> None:
        """Add a block to the table."""
        self.block_ids.append(block_id)
        self.num_tokens += num_tokens

    def __len__(self) -> int:
        return len(self.block_ids)

    def copy(self, new_request_id: str) -> "BlockTable":
        """Create a copy with new request ID."""
        return BlockTable(
            request_id=new_request_id,
            block_ids=self.block_ids.copy(),
            num_tokens=self.num_tokens,
        )


# =============================================================================
# PagedCacheManager - Main manager (vLLM BlockPool style)
# =============================================================================

class PagedCacheManager(CacheManager):
    """
    Paged KV cache manager following vLLM's BlockPool architecture.

    Features:
    - Block allocation/deallocation with reference counting
    - Prefix sharing via chain-based hash deduplication
    - Copy-on-Write for efficient forking
    - O(1) LRU eviction using doubly linked list

    Implements the CacheManager ABC interface for consistency with other
    cache implementations in oMLX.

    Args:
        block_size: Number of tokens per block (default: 64)
        max_blocks: Maximum number of blocks to allocate (default: 1000)
        enable_caching: Whether to enable prefix caching (default: True)
    """

    def __init__(
        self,
        block_size: int = 64,
        max_blocks: int = 1000,
        enable_caching: bool = True,
        model_name: str = "",
        initial_blocks: int = 256,
    ):
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.enable_caching = enable_caching
        self.model_name = model_name
        self.initial_blocks = initial_blocks

        # Warn if model_name is not set (cache isolation may not work)
        if not model_name:
            logger.warning(
                "PagedCacheManager initialized without model_name. "
                "Cache isolation between models may not work correctly."
            )

        # Dynamic allocation tracking
        # Only create initial blocks; grow dynamically as needed up to max_blocks
        initial_count = min(initial_blocks, max_blocks)
        self._current_allocated_count = initial_count

        # Create only initial blocks (memory optimization)
        self.blocks: List[CacheBlock] = [
            CacheBlock(block_id=i) for i in range(initial_count)
        ]

        # Free block queue (doubly linked list for O(1) LRU)
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # Hash-to-block cache for prefix caching
        self.cached_block_hash_to_block = BlockHashToBlockMap()

        # Request to block table mapping
        self.request_tables: Dict[str, BlockTable] = {}

        # Allocated blocks (for fast lookup)
        self.allocated_blocks: Dict[int, CacheBlock] = {}

        # Reserve null block (block 0) - never freed
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True
        self.null_block.ref_count = 1
        self.allocated_blocks[self.null_block.block_id] = self.null_block

        # Statistics - track actual created blocks, not max
        self.stats = PagedCacheStats(
            total_blocks=initial_count,
            allocated_blocks=1,  # null block
            free_blocks=initial_count - 1,
        )

        # Thread safety
        self._lock = threading.RLock()

        # paged SSD cache manager for storage (set via set_paged_ssd_cache_manager)
        self._paged_ssd_cache_manager: Optional[Any] = None

        logger.info(
            f"PagedCacheManager initialized: block_size={block_size}, "
            f"initial_blocks={initial_count}, max_blocks={max_blocks}, "
            f"max_tokens={block_size * max_blocks}"
        )

    def set_paged_ssd_cache_manager(self, paged_ssd_cache_manager: Any) -> None:
        """
        Set paged SSD cache manager for tiered storage.

        When set, evicted blocks will be saved to paged SSD before clearing
        GPU memory, allowing them to be restored later.

        Args:
            paged_ssd_cache_manager: PagedSSDCacheManager instance
        """
        self._paged_ssd_cache_manager = paged_ssd_cache_manager
        logger.info("paged SSD cache manager connected to PagedCacheManager")

    # =========================================================================
    # Dynamic Block Pool Growth (Elastic KV Cache)
    # =========================================================================

    def _grow_blocks(self, additional_blocks: int) -> int:
        """
        Dynamically expand the block pool.

        This method is called when more blocks are needed but the current
        pool is exhausted. It creates new blocks up to max_blocks limit.

        Args:
            additional_blocks: Number of blocks to add.

        Returns:
            Number of blocks actually created.
        """
        with self._lock:
            available = self.max_blocks - self._current_allocated_count
            to_create = min(additional_blocks, available)

            if to_create <= 0:
                return 0

            start_id = self._current_allocated_count
            new_blocks = [
                CacheBlock(block_id=i)
                for i in range(start_id, start_id + to_create)
            ]

            self.blocks.extend(new_blocks)
            self.free_block_queue.append_n(new_blocks)
            self._current_allocated_count += to_create

            self.stats.total_blocks = self._current_allocated_count
            self.stats.free_blocks = self.free_block_queue.num_free_blocks

            logger.info(
                f"Block pool grown: +{to_create}, "
                f"total={self._current_allocated_count}/{self.max_blocks}"
            )
            return to_create

    # =========================================================================
    # Block Allocation (vLLM style)
    # =========================================================================

    def allocate_block(self) -> Optional[CacheBlock]:
        """
        Allocate a new cache block.

        Returns:
            CacheBlock if available, None if out of memory.
        """
        with self._lock:
            if self.free_block_queue.num_free_blocks == 0:
                # Try to grow the block pool dynamically
                grown = self._grow_blocks(min(256, self.max_blocks - self._current_allocated_count))
                if grown == 0:
                    logger.warning("Out of cache blocks (max reached)")
                    return None

            block = self.free_block_queue.popleft()

            # Evict from hash cache if needed
            if self.enable_caching:
                self._maybe_evict_cached_block(block)

            block.ref_count = 1
            block.touch()
            self.allocated_blocks[block.block_id] = block

            self.stats.allocated_blocks += 1
            self.stats.free_blocks -= 1

            return block

    def get_new_blocks(self, num_blocks: int) -> List[CacheBlock]:
        """
        Allocate multiple blocks at once (vLLM style).

        Args:
            num_blocks: Number of blocks to allocate

        Returns:
            List of allocated blocks

        Raises:
            ValueError: If not enough free blocks even after dynamic growth
        """
        with self._lock:
            if num_blocks > self.free_block_queue.num_free_blocks:
                # Try to grow the block pool dynamically
                needed = num_blocks - self.free_block_queue.num_free_blocks
                self._grow_blocks(needed + 128)  # Extra buffer for future allocations

            if num_blocks > self.free_block_queue.num_free_blocks:
                raise ValueError(
                    f"Cannot allocate {num_blocks} blocks, "
                    f"only {self.free_block_queue.num_free_blocks} available "
                    f"(max={self.max_blocks})"
                )

            blocks = self.free_block_queue.popleft_n(num_blocks)

            for block in blocks:
                if self.enable_caching:
                    self._maybe_evict_cached_block(block)

                block.ref_count = 1
                block.touch()
                self.allocated_blocks[block.block_id] = block

            self.stats.allocated_blocks += num_blocks
            self.stats.free_blocks -= num_blocks

            return blocks

    def _maybe_evict_cached_block(self, block: CacheBlock) -> bool:
        """
        Evict a block from the hash cache if present.

        In paged SSD-only mode, block data is always on paged SSD, so this just
        removes the block from the cache index.

        Args:
            block: Block to evict

        Returns:
            True if block was evicted from cache
        """
        if block.block_hash is None:
            return False

        evicted = self.cached_block_hash_to_block.pop(
            block.block_hash, block.block_id
        )

        if evicted:
            block.reset_hash()
            self.stats.evictions += 1
            return True

        return False

    def free_block(self, block_id: int) -> bool:
        """
        Free a cache block (decrements ref_count, frees if 0).

        Returns:
            True if block was freed, False if still referenced.
        """
        with self._lock:
            if block_id not in self.allocated_blocks:
                logger.warning(f"Attempted to free unknown block: {block_id}")
                return False

            block = self.allocated_blocks[block_id]
            if block.is_null:
                return False  # Never free null block

            block.ref_count -= 1

            if block.ref_count <= 0:
                # Remove from hash cache
                if block.block_hash is not None:
                    self.cached_block_hash_to_block.pop(block.block_hash, block.block_id)

                # Remove from allocated
                del self.allocated_blocks[block_id]

                # Add to free queue (back = MRU)
                self.free_block_queue.append(block)

                self.stats.allocated_blocks -= 1
                self.stats.free_blocks += 1
                self.stats.total_tokens_cached -= block.token_count

                return True

            return False

    def free_blocks(self, blocks: Iterable[CacheBlock]) -> None:
        """
        Free multiple blocks (vLLM style).

        Blocks with ref_count=0 are added to the free queue.

        Args:
            blocks: Blocks to free (in eviction order)
        """
        with self._lock:
            blocks_list = list(blocks)
            to_free = []

            for block in blocks_list:
                if block.is_null:
                    continue

                block.ref_count -= 1

                if block.ref_count <= 0:
                    # Remove from hash cache
                    if block.block_hash is not None:
                        self.cached_block_hash_to_block.pop(block.block_hash, block.block_id)

                    del self.allocated_blocks[block.block_id]
                    to_free.append(block)
                    self.stats.allocated_blocks -= 1
                    self.stats.free_blocks += 1
                    self.stats.total_tokens_cached -= block.token_count

            # Add to free queue (back = MRU, evicted last)
            self.free_block_queue.append_n(to_free)

    def touch(self, blocks: Iterable[CacheBlock]) -> None:
        """
        Touch blocks to prevent eviction (cache hit, vLLM style).

        Increments ref_count and removes from free queue if needed.

        Args:
            blocks: Blocks to touch
        """
        with self._lock:
            for block in blocks:
                if block.ref_count == 0 and not block.is_null:
                    # Block is in free queue, remove it
                    try:
                        self.free_block_queue.remove(block)
                        self.stats.free_blocks -= 1
                        self.stats.allocated_blocks += 1
                        self.allocated_blocks[block.block_id] = block
                    except RuntimeError:
                        pass  # Block not in queue

                block.ref_count += 1
                block.touch()

    # =========================================================================
    # Reference Counting
    # =========================================================================

    def increment_ref(self, block_id: int) -> bool:
        """Increment reference count for a block."""
        with self._lock:
            if block_id not in self.allocated_blocks:
                return False

            block = self.allocated_blocks[block_id]
            block.ref_count += 1
            block.touch()

            if block.ref_count == 2:
                self.stats.shared_blocks += 1

            return True

    def decrement_ref(self, block_id: int) -> bool:
        """Decrement reference count (alias for free_block)."""
        return self.free_block(block_id)

    def release_for_eviction(self, block_ids: List[int]) -> int:
        """
        Release blocks for eviction without removing from allocated_blocks.

        Decrements ref_count so blocks become evictable, but keeps them in
        allocated_blocks so they can still be found by cache lookups.

        Args:
            block_ids: List of block IDs to release.

        Returns:
            Number of blocks released.
        """
        released = 0
        with self._lock:
            for block_id in block_ids:
                block = self.allocated_blocks.get(block_id)
                if block is None or block.is_null:
                    continue

                if block.ref_count > 0:
                    block.ref_count -= 1
                    released += 1

                    if block.ref_count == 1:
                        self.stats.shared_blocks = max(0, self.stats.shared_blocks - 1)

        return released

    # =========================================================================
    # Prefix Caching (vLLM chain-hash style)
    # =========================================================================

    def get_cached_block(self, block_hash: BlockHash) -> Optional[CacheBlock]:
        """
        Get a cached block by its hash (vLLM style).

        Args:
            block_hash: Content hash of the block

        Returns:
            Cached block if found, None otherwise
        """
        if not self.enable_caching:
            return None

        with self._lock:
            block = self.cached_block_hash_to_block.get_block(block_hash)
            if block:
                self.stats.hits += 1
            else:
                self.stats.misses += 1
            return block

    def cache_full_blocks(
        self,
        blocks: List[CacheBlock],
        token_ids: List[int],
        num_cached_blocks: int,
        num_full_blocks: int,
    ) -> None:
        """
        Cache full blocks for prefix caching (vLLM style).

        Computes chain hashes and adds blocks to the cache.

        Args:
            blocks: All blocks for the request
            token_ids: All token IDs for the request
            num_cached_blocks: Number of blocks already cached
            num_full_blocks: Number of full blocks to cache
        """
        if not self.enable_caching:
            return

        if num_cached_blocks >= num_full_blocks:
            return

        with self._lock:
            # Get parent hash from last cached block
            parent_hash = None
            if num_cached_blocks > 0:
                parent_hash = blocks[num_cached_blocks - 1].block_hash

            for i in range(num_cached_blocks, num_full_blocks):
                block = blocks[i]
                if block.block_hash is not None:
                    parent_hash = block.block_hash
                    continue  # Already cached

                # Get tokens for this block
                start = i * self.block_size
                end = start + self.block_size
                block_tokens = token_ids[start:end]

                # Compute chain hash
                block_hash = compute_block_hash(parent_hash, block_tokens, model_name=self.model_name)
                block.block_hash = block_hash
                block.token_count = len(block_tokens)

                # Add to cache
                self.cached_block_hash_to_block.insert(block_hash, block)

                parent_hash = block_hash

    def get_computed_blocks(
        self,
        token_ids: List[int],
    ) -> Tuple[List[CacheBlock], int]:
        """
        Find cached blocks for a token prefix (vLLM style).

        Args:
            token_ids: Token IDs to look up

        Returns:
            Tuple of (cached_blocks, num_cached_tokens)
        """
        if not self.enable_caching:
            return [], 0

        with self._lock:
            cached_blocks = []
            parent_hash = None
            num_cached_tokens = 0

            num_full_blocks = len(token_ids) // self.block_size

            for i in range(num_full_blocks):
                start = i * self.block_size
                end = start + self.block_size
                block_tokens = token_ids[start:end]

                # Compute expected hash
                block_hash = compute_block_hash(parent_hash, block_tokens, model_name=self.model_name)

                # Look up in cache
                cached_block = self.cached_block_hash_to_block.get_block(block_hash)

                # Lazy restore: if not in memory but exists on SSD, register it
                if cached_block is None and self._paged_ssd_cache_manager is not None:
                    if self._paged_ssd_cache_manager.has_block(block_hash):
                        block = self.free_block_queue.popleft()
                        if block is not None:
                            block.block_hash = block_hash
                            block.token_count = self.block_size
                            block.ref_count = 0
                            self.cached_block_hash_to_block.insert(
                                block_hash, block
                            )
                            self.allocated_blocks[block.block_id] = block
                            cached_block = block

                if cached_block is None:
                    self.stats.misses += 1
                    break  # Cache miss, stop here

                cached_blocks.append(cached_block)
                parent_hash = block_hash
                num_cached_tokens += self.block_size
                self.stats.hits += 1

            return cached_blocks, num_cached_tokens

    # =========================================================================
    # Legacy hash methods (for backwards compatibility)
    # =========================================================================

    def find_cached_block(
        self,
        tokens: List[int],
        parent_hash: Optional[BlockHash] = None,
    ) -> Optional[CacheBlock]:
        """
        Find a cached block matching the given tokens using chain hash.

        Args:
            tokens: Token IDs to look up
            parent_hash: Hash of the parent block (for chain), or None for first block

        Returns:
            Cached block if found, None otherwise
        """
        if not self.enable_caching:
            return None

        with self._lock:
            block_hash = compute_block_hash(parent_hash, tokens, model_name=self.model_name)
            block = self.cached_block_hash_to_block.get_block(block_hash)
            if block:
                block.touch()
                self.stats.hits += 1
                return block

            self.stats.misses += 1
            return None

    def register_block_hash(
        self,
        block: CacheBlock,
        tokens: List[int],
        parent_hash: Optional[BlockHash] = None,
    ) -> None:
        """
        Register a block's hash for deduplication using chain hash.

        Args:
            block: Block to register
            tokens: Token IDs in this block
            parent_hash: Hash of the parent block (for chain), or None for first block
        """
        if not self.enable_caching:
            return

        with self._lock:
            block_hash = compute_block_hash(parent_hash, tokens, model_name=self.model_name)
            block.block_hash = block_hash
            self.cached_block_hash_to_block.insert(block_hash, block)

    # =========================================================================
    # Block Table Management
    # =========================================================================

    def create_block_table(self, request_id: str) -> BlockTable:
        """Create a new block table for a request."""
        with self._lock:
            table = BlockTable(request_id=request_id)
            self.request_tables[request_id] = table
            return table

    def get_block_table(self, request_id: str) -> Optional[BlockTable]:
        """Get block table for a request."""
        with self._lock:
            return self.request_tables.get(request_id)

    def get_or_create_block_table(self, request_id: str) -> BlockTable:
        """Get or create block table for a request."""
        with self._lock:
            if request_id not in self.request_tables:
                self.request_tables[request_id] = BlockTable(request_id=request_id)
            return self.request_tables[request_id]

    def delete_block_table(self, request_id: str) -> None:
        """Delete block table and free associated blocks."""
        with self._lock:
            table = self.request_tables.pop(request_id, None)
            if table:
                for block_id in table.block_ids:
                    self.free_block(block_id)

    def add_block_to_table(
        self,
        table: BlockTable,
        block: CacheBlock,
        tokens_in_block: int,
    ) -> None:
        """Add a block to a block table."""
        with self._lock:
            table.block_ids.append(block.block_id)
            block.token_count = tokens_in_block
            table.num_tokens += tokens_in_block
            self.stats.total_tokens_cached += tokens_in_block

    # =========================================================================
    # Prefix Sharing & COW
    # =========================================================================

    def find_shared_prefix(
        self,
        tokens: List[int],
    ) -> Tuple[List[int], List[int]]:
        """
        Find shared prefix blocks for a token sequence.

        Uses get_computed_blocks for consistent chain-hash lookup.
        """
        cached_blocks, num_cached_tokens = self.get_computed_blocks(tokens)

        shared_block_ids = [b.block_id for b in cached_blocks]
        remaining_tokens = tokens[num_cached_tokens:]

        return shared_block_ids, remaining_tokens

    def fork_block_table(
        self,
        source_table: BlockTable,
        new_request_id: str,
    ) -> BlockTable:
        """
        Fork a block table for a new request (COW).
        """
        with self._lock:
            new_table = source_table.copy(new_request_id)

            for block_id in new_table.block_ids:
                self.increment_ref(block_id)

            self.request_tables[new_request_id] = new_table

            logger.debug(
                f"Forked block table: {source_table.request_id} -> {new_request_id}, "
                f"blocks={len(new_table.block_ids)}"
            )

            return new_table

    def get_blocks_for_generation(
        self,
        table: BlockTable,
    ) -> Tuple[List[CacheBlock], bool]:
        """
        Get blocks for generation, applying COW if needed.
        """
        with self._lock:
            blocks = []
            was_copied = False

            for i, block_id in enumerate(table.block_ids):
                block = self.allocated_blocks.get(block_id)
                if not block:
                    continue

                if block.is_shared():
                    new_block = self._cow_copy_block(block)
                    if new_block:
                        table.block_ids[i] = new_block.block_id
                        blocks.append(new_block)
                        was_copied = True
                        self.stats.cow_copies += 1
                    else:
                        blocks.append(block)
                else:
                    blocks.append(block)

                block.touch()

            return blocks, was_copied

    def _cow_copy_block(self, source_block: CacheBlock) -> Optional[CacheBlock]:
        """
        Create a copy of a block for COW.

        In paged SSD-only mode, we don't copy data - we just allocate a new block
        with the same metadata. The actual KV data will be loaded from paged SSD
        when needed.
        """
        new_block = self.allocate_block()
        if not new_block:
            return None

        new_block.token_count = source_block.token_count
        new_block.block_hash = source_block.block_hash

        source_block.ref_count -= 1
        if source_block.ref_count == 1:
            self.stats.shared_blocks -= 1

        logger.debug(
            f"COW copy: block {source_block.block_id} -> {new_block.block_id}"
        )

        return new_block

    # =========================================================================
    # Legacy allocation methods (for backwards compatibility)
    # =========================================================================

    def allocate_blocks_for_tokens(self, num_tokens: int) -> List[CacheBlock]:
        """Allocate enough blocks to hold num_tokens."""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        return self.get_new_blocks(num_blocks_needed)

    # =========================================================================
    # Eviction
    # =========================================================================

    def evict_lru_blocks(self, num_blocks: int) -> int:
        """
        Evict least recently used blocks.

        With the doubly linked list, LRU blocks are already at the front
        of the free queue. We just need to pop from front.
        """
        with self._lock:
            evicted = 0

            # Get evictable blocks from free queue (they're already LRU ordered)
            for _ in range(min(num_blocks, self.free_block_queue.num_free_blocks)):
                try:
                    block = self.free_block_queue.popleft()
                    self._maybe_evict_cached_block(block)
                    # Put back at end (now available for allocation)
                    self.free_block_queue.append(block)
                    evicted += 1
                except ValueError:
                    break

            if evicted > 0:
                logger.info(f"Evicted {evicted} LRU blocks from cache")

            return evicted

    def handle_memory_pressure(self, requested_blocks: int) -> bool:
        """Handle memory pressure by evicting blocks."""
        with self._lock:
            if self.free_block_queue.num_free_blocks >= requested_blocks:
                return True

            needed = requested_blocks - self.free_block_queue.num_free_blocks
            self.evict_lru_blocks(needed)

            return self.free_block_queue.num_free_blocks >= requested_blocks

    # =========================================================================
    # Statistics and Properties
    # =========================================================================

    @property
    def free_blocks(self) -> int:
        """Number of free blocks available."""
        return self.free_block_queue.num_free_blocks

    @property
    def usage(self) -> float:
        """Cache usage ratio (0.0 to 1.0)."""
        total = self.max_blocks - 1  # Exclude null block
        if total == 0:
            return 0.0
        return 1.0 - (self.free_blocks / total)

    def get_stats(self) -> PagedCacheStats:
        """Get current cache statistics."""
        with self._lock:
            self.stats.shared_blocks = sum(
                1 for b in self.allocated_blocks.values() if b.ref_count > 1
            )
            self.stats.free_blocks = self.free_block_queue.num_free_blocks
            return self.stats

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information."""
        with self._lock:
            stats = self.get_stats()
            return {
                "block_size": self.block_size,
                "max_blocks": self.max_blocks,
                "allocated_blocks": stats.allocated_blocks,
                "free_blocks": stats.free_blocks,
                "shared_blocks": stats.shared_blocks,
                "total_tokens_cached": stats.total_tokens_cached,
                "utilization": stats.allocated_blocks / self.max_blocks,
                "cache_hit_rate": (
                    stats.hits / (stats.hits + stats.misses)
                    if (stats.hits + stats.misses) > 0 else 0
                ),
            }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        with self._lock:
            self.stats.hits = 0
            self.stats.misses = 0
            self.stats.cow_copies = 0
            self.stats.evictions = 0

    def reset_prefix_cache(self) -> bool:
        """Reset the prefix cache."""
        with self._lock:
            num_used = self.max_blocks - self.free_block_queue.num_free_blocks
            if num_used > 1:  # null_block is always "used"
                logger.warning(f"Cannot reset cache: {num_used - 1} blocks in use")
                return False

            self.cached_block_hash_to_block.clear()

            for block in self.blocks:
                block.reset_hash()

            self.stats.evictions = 0
            self.stats.hits = 0
            self.stats.misses = 0

            logger.info("Prefix cache reset successfully")
            return True

    def clear(self) -> int:
        """
        Clear all cached data and reset to initial block count.

        Returns:
            Number of entries cleared.
        """
        with self._lock:
            # Count entries before clearing
            cleared_count = len(self.allocated_blocks) - 1  # Exclude null block

            # Reset to initial blocks (memory optimization)
            initial_count = min(self.initial_blocks, self.max_blocks)
            self._current_allocated_count = initial_count

            # Recreate blocks and queue with only initial blocks
            self.blocks = [
                CacheBlock(block_id=i) for i in range(initial_count)
            ]
            self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

            self.cached_block_hash_to_block.clear()
            self.request_tables.clear()
            self.allocated_blocks.clear()

            # Reserve null block
            self.null_block = self.free_block_queue.popleft()
            self.null_block.is_null = True
            self.null_block.ref_count = 1
            self.allocated_blocks[self.null_block.block_id] = self.null_block

            self.stats = PagedCacheStats(
                total_blocks=initial_count,
                allocated_blocks=1,
                free_blocks=initial_count - 1,
            )

            logger.info(
                f"PagedCacheManager cleared (reset to {initial_count} initial blocks)"
            )

            return max(0, cleared_count)

    # =========================================================================
    # SSD Cache Support
    # =========================================================================

    def get_evictable_blocks(self, count: int) -> List[CacheBlock]:
        """
        Get LRU blocks that can be evicted (metadata cleared).

        In paged SSD-only mode, blocks don't store data in GPU memory,
        so this returns blocks that can be freed from the metadata index.

        Args:
            count: Maximum number of blocks to return.

        Returns:
            List of evictable blocks in LRU order.
        """
        with self._lock:
            candidates = []

            # Iterate through free queue (LRU order)
            current = self.free_block_queue.fake_head.next_free_block
            while (
                current is not None
                and current != self.free_block_queue.fake_tail
                and len(candidates) < count
            ):
                # Block must not be null and have ref_count == 0
                if not current.is_null and current.ref_count == 0:
                    candidates.append(current)
                current = current.next_free_block

            # Also check allocated blocks with ref_count == 0 (not in free queue yet)
            if len(candidates) < count:
                # Sort by last_access (LRU)
                remaining = []
                for block in self.allocated_blocks.values():
                    if (
                        not block.is_null
                        and block.ref_count == 0
                        and block not in candidates
                    ):
                        remaining.append(block)

                remaining.sort(key=lambda b: b.last_access)
                candidates.extend(remaining[: count - len(candidates)])

            return candidates[:count]

    def mark_block_cold(self, block_id: int) -> bool:
        """
        Mark a block as evicted (metadata preserved).

        In paged SSD-only mode, this is a no-op since block data is always on paged SSD.
        Kept for API compatibility.

        Args:
            block_id: Block ID to mark.

        Returns:
            True if successful, False if block not found or has data users.
        """
        with self._lock:
            block = self.blocks[block_id] if block_id < len(self.blocks) else None
            if block is None:
                logger.warning(f"Block {block_id} not found")
                return False

            if block.ref_count > 0:
                logger.warning(
                    f"Cannot mark block {block_id}: ref_count={block.ref_count}"
                )
                return False

            if block.is_null:
                logger.warning(f"Cannot mark null block")
                return False

            # In paged SSD-only mode, data is already on paged SSD
            self.stats.evictions += 1

            logger.debug(
                f"Marked block {block_id} "
                f"(hash={block.block_hash.hex()[:16] if block.block_hash else 'None'}...)"
            )
            return True

    def evict_block_permanently(self, block_id: int) -> bool:
        """
        Evict a block permanently (removes from metadata index).

        This method:
        - Removes from hash cache (block won't be found in cache lookups)
        - Returns block to free queue (can be reallocated)
        - Removes from allocated_blocks

        Note: In paged SSD-only mode, the data remains on paged SSD and may be deleted
        by PagedSSDCacheManager's LRU eviction if needed.

        Args:
            block_id: Block ID to evict.

        Returns:
            True if successful, False if block not found or in use.
        """
        with self._lock:
            block = self.blocks[block_id] if block_id < len(self.blocks) else None
            if block is None:
                logger.warning(f"Block {block_id} not found for permanent eviction")
                return False

            if block.ref_count > 0:
                logger.warning(
                    f"Cannot permanently evict block {block_id}: ref_count={block.ref_count}"
                )
                return False

            if block.is_null:
                logger.warning(f"Cannot evict null block")
                return False

            # Remove from hash cache
            if block.block_hash is not None:
                self.cached_block_hash_to_block.pop(block.block_hash, block.block_id)

            # Clear metadata
            block.reset_hash()
            block.token_count = 0

            # Remove from allocated_blocks and add to free queue
            if block_id in self.allocated_blocks:
                del self.allocated_blocks[block_id]
                self.stats.allocated_blocks -= 1

            self.free_block_queue.append(block)
            self.stats.free_blocks += 1
            self.stats.evictions += 1

            logger.debug(f"Permanently evicted block {block_id}")
            return True

    def restore_block(
        self,
        block_id: int,
        cache_data: List[Tuple[Any, Any]],
    ) -> bool:
        """
        Restore block data from cold storage.

        In paged SSD-only mode, this is a no-op since data is always loaded
        directly from paged SSD when needed. Kept for API compatibility.

        Args:
            block_id: Block ID to restore.
            cache_data: KV cache data (unused in paged SSD-only mode).

        Returns:
            True if successful, False if block not found.
        """
        with self._lock:
            block = self.blocks[block_id] if block_id < len(self.blocks) else None
            if block is None:
                logger.warning(f"Block {block_id} not found for restoration")
                return False

            block.touch()

            logger.debug(
                f"Block {block_id} touched "
                f"(hash={block.block_hash.hex()[:16] if block.block_hash else 'None'}...)"
            )
            return True

    def get_cold_blocks(self) -> List[CacheBlock]:
        """
        Get all blocks that have data on paged SSD.

        In paged SSD-only mode, returns all blocks with block_hash set
        (i.e., blocks that have data stored on paged SSD).

        Returns:
            List of blocks with paged SSD data.
        """
        with self._lock:
            return [b for b in self.blocks if b.block_hash is not None and not b.is_null]

    @property
    def cold_block_count(self) -> int:
        """Number of blocks with data on paged SSD."""
        with self._lock:
            return sum(1 for b in self.blocks if b.block_hash is not None and not b.is_null)

    def get_ref_count_distribution(self) -> Dict[int, int]:
        """
        Get distribution of blocks by ref_count.

        Returns:
            Dict mapping ref_count -> number of blocks with that count.
            Only includes ref_counts that have at least one block.
        """
        with self._lock:
            distribution: Dict[int, int] = {}
            for block in self.allocated_blocks.values():
                rc = block.ref_count
                distribution[rc] = distribution.get(rc, 0) + 1
            return distribution

    def get_ref_count_summary(self) -> str:
        """
        Get a compact string summary of ref_count distribution.

        Returns:
            String like "rc0=5(ssd=3),rc1=100" showing counts per ref_count.
            In paged SSD-only mode, all blocks have data on paged SSD.
        """
        dist = self.get_ref_count_distribution()
        if not dist:
            return "rc=none"

        # Count blocks with paged SSD data (blocks with block_hash)
        ssd_count = 0
        with self._lock:
            for block in self.allocated_blocks.values():
                if block.ref_count == 0 and block.block_hash is not None:
                    ssd_count += 1

        parts = []
        for k, v in sorted(dist.items()):
            if k == 0:
                # Show paged SSD count for rc0
                parts.append(f"rc0={v}(ssd={ssd_count})")
            else:
                parts.append(f"rc{k}={v}")
        return ",".join(parts)

    # =========================================================================
    # CacheManager ABC Interface Implementation
    # =========================================================================

    def fetch(self, key: Any) -> Tuple[Optional[Any], bool]:
        """
        Fetch a cached block by its hash.

        Args:
            key: BlockHash (bytes) to look up.

        Returns:
            Tuple of (CacheBlock, True) if found, (None, False) otherwise.
        """
        if not isinstance(key, bytes):
            return None, False

        block = self.get_cached_block(BlockHash(key))
        if block is not None:
            return block, True
        return None, False

    def store(self, key: Any, value: Any) -> bool:
        """
        Store a block in the cache.

        For PagedCacheManager, use allocate_block() and register_block_hash()
        for the full workflow. This method provides a simplified interface.

        Args:
            key: BlockHash (bytes) for the block.
            value: CacheBlock to store.

        Returns:
            True if stored successfully.
        """
        if not isinstance(key, bytes) or not isinstance(value, CacheBlock):
            return False

        with self._lock:
            block_hash = BlockHash(key)
            value.block_hash = block_hash
            self.cached_block_hash_to_block.insert(block_hash, value)
            return True

    def evict(self, key: Any) -> bool:
        """
        Evict a specific block from the cache.

        Args:
            key: BlockHash (bytes) or block_id (int) to evict.

        Returns:
            True if evicted, False if not found.
        """
        if isinstance(key, bytes):
            # Evict by block hash
            block = self.cached_block_hash_to_block.get_block(BlockHash(key))
            if block is not None:
                return self.evict_block_permanently(block.block_id)
            return False
        elif isinstance(key, int):
            # Evict by block ID
            return self.evict_block_permanently(key)
        return False

    @property
    def size(self) -> int:
        """
        Get the current number of allocated blocks.

        Returns:
            Number of allocated blocks (excluding null block).
        """
        return max(0, len(self.allocated_blocks) - 1)  # Exclude null block

    @property
    def max_size(self) -> int:
        """
        Get the maximum number of blocks.

        Returns:
            Maximum number of blocks.
        """
        return self.max_blocks
