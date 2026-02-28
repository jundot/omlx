# SPDX-License-Identifier: Apache-2.0
"""
Tiered Cache Manager for oMLX.

This module manages hot/cold tiered KV caching, enabling automatic paged SSD offloading
when GPU memory is under pressure.

In paged SSD-only mode:
- All KV cache data is stored on paged SSD via PagedSSDCacheManager
- PagedCacheManager only stores block metadata (no GPU memory for cache data)
- BatchGenerator handles GPU memory for active inference
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from omlx.utils.formatting import format_bytes

if TYPE_CHECKING:
    from ..paged_cache import PagedCacheManager
    from ..prefix_cache import BlockAwarePrefixCache
    from ..paged_ssd_cache import PagedSSDCacheManager
    from ..memory_monitor import MemoryMonitor

logger = logging.getLogger(__name__)


class TieredCacheManager:
    """
    Manages hot/cold tiered KV caching.

    This class coordinates between PagedCacheManager (hot cache in GPU memory)
    and PagedSSDCacheManager (cold cache on disk) to provide efficient memory usage.

    In paged SSD-only mode, all KV cache data is stored on paged SSD. The PagedCacheManager
    only manages block metadata, and BatchGenerator handles GPU memory for
    active inference.
    """

    def __init__(
        self,
        paged_cache_manager: Optional["PagedCacheManager"] = None,
        block_aware_cache: Optional["BlockAwarePrefixCache"] = None,
        paged_ssd_cache_manager: Optional["PagedSSDCacheManager"] = None,
        memory_monitor: Optional["MemoryMonitor"] = None,
        block_size: int = 256,
    ):
        """
        Initialize the tiered cache manager.

        Args:
            paged_cache_manager: Manager for paged KV cache.
            block_aware_cache: Block-aware prefix cache.
            paged_ssd_cache_manager: Manager for paged SSD storage.
            memory_monitor: Monitor for memory pressure.
            block_size: Tokens per block.
        """
        self.paged_cache_manager = paged_cache_manager
        self.block_aware_cache = block_aware_cache
        self.paged_ssd_cache_manager = paged_ssd_cache_manager
        self.memory_monitor = memory_monitor
        self.block_size = block_size

    @classmethod
    def from_config(
        cls,
        paged_cache_manager: Optional["PagedCacheManager"],
        block_aware_cache: Optional["BlockAwarePrefixCache"],
        paged_ssd_cache_dir: Optional[str],
        paged_ssd_cache_max_size: int,
        block_size: int,
        model: Any = None,
        hot_cache_max_bytes: int = 0,
    ) -> Optional["TieredCacheManager"]:
        """
        Create a TieredCacheManager from configuration.

        Args:
            paged_cache_manager: Manager for paged KV cache.
            block_aware_cache: Block-aware prefix cache.
            paged_ssd_cache_dir: Path for paged SSD cache storage (None = disabled).
            paged_ssd_cache_max_size: Maximum paged SSD cache size in bytes.
            block_size: Tokens per block.
            model: The model (for extracting KV cache dimensions).
            hot_cache_max_bytes: Maximum in-memory hot cache size (0 = disabled).

        Returns:
            TieredCacheManager instance or None if tiered caching is disabled.
        """
        # Import here to avoid circular imports
        try:
            from ..paged_ssd_cache import PagedSSDCacheManager
            from ..memory_monitor import MemoryMonitor
        except ImportError:
            if paged_ssd_cache_dir:
                logger.warning(
                    "Paged paged SSD cache requested but paged_ssd_cache/memory_monitor modules "
                    "not available. Install required dependencies."
                )
            return None

        if not paged_cache_manager:
            if paged_ssd_cache_dir:
                logger.warning(
                    "Paged paged SSD cache requires paged cache. Ignoring paged_ssd_cache_dir."
                )
            return None

        if not paged_ssd_cache_dir:
            logger.debug("Paged paged SSD cache not configured (no --paged-ssd-cache-dir specified)")
            return None

        try:
            # Initialize paged SSD cache manager
            paged_ssd_cache_manager = PagedSSDCacheManager(
                cache_dir=Path(paged_ssd_cache_dir),
                max_size_bytes=paged_ssd_cache_max_size,
                hot_cache_max_bytes=hot_cache_max_bytes,
            )

            # Connect paged SSD cache manager to PagedCacheManager
            paged_cache_manager.set_paged_ssd_cache_manager(paged_ssd_cache_manager)

            # Connect paged SSD cache manager to BlockAwarePrefixCache for paged SSD-only mode
            if block_aware_cache is not None:
                block_aware_cache.set_paged_ssd_cache_manager(paged_ssd_cache_manager)

            manager = cls(
                paged_cache_manager=paged_cache_manager,
                block_aware_cache=block_aware_cache,
                paged_ssd_cache_manager=paged_ssd_cache_manager,
                memory_monitor=None,  # Memory monitor not used in paged SSD-only mode
                block_size=block_size,
            )

            logger.info(
                f"Paged paged SSD cache enabled: "
                f"cache_dir={paged_ssd_cache_dir}, "
                f"max_size={format_bytes(paged_ssd_cache_max_size)}, "
                f"block_size={block_size} tokens"
            )

            return manager

        except Exception as e:
            logger.error(f"Failed to initialize paged SSD cache: {e}")
            return None

    def check_memory_pressure(self) -> bool:
        """
        Check memory and evict blocks if needed.

        In paged SSD-only mode, memory pressure is not monitored since
        KV cache data is stored on paged SSD, not GPU memory.

        Returns:
            True if eviction was performed.
        """
        # In paged SSD-only mode, memory_monitor is not used
        # All KV cache data is on paged SSD, so no GPU memory pressure from PagedCache
        return False

    def evict_blocks_permanently(self, bytes_to_free: int) -> int:
        """
        Evict LRU blocks permanently (metadata cleanup).

        In paged SSD-only mode, blocks don't store data in GPU memory.
        This method just removes block metadata to free up slots.

        Args:
            bytes_to_free: Target bytes to free (used for estimation).

        Returns:
            Number of bytes freed (estimated).
        """
        if self.paged_cache_manager is None or self.memory_monitor is None:
            return 0

        # Estimate how many blocks to evict
        num_blocks_to_evict = self.memory_monitor.estimate_blocks_to_free(
            bytes_to_free, self.block_size
        )

        # Get evictable blocks in LRU order
        evictable = self.paged_cache_manager.get_evictable_blocks(num_blocks_to_evict)

        if not evictable:
            logger.debug("No evictable blocks found for permanent eviction")
            return 0

        freed = 0
        evicted_count = 0

        for block in evictable:
            # In paged SSD-only mode, just clear metadata (data is on paged SSD)
            if self.paged_cache_manager.evict_block_permanently(block.block_id):
                freed += self.memory_monitor.estimate_block_memory(self.block_size)
                evicted_count += 1

            if freed >= bytes_to_free:
                break

        if evicted_count > 0:
            logger.info(
                f"Evicted {evicted_count} blocks permanently "
                f"(~{format_bytes(freed)} estimated)"
            )

        return freed

    def evict_blocks_to_cold(self, bytes_to_free: int) -> int:
        """
        Evict LRU blocks (with paged SSD cache configured).

        In paged SSD-only mode, data is already on paged SSD, so this just evicts
        block metadata from the index. The data remains on paged SSD and can
        be re-discovered if the same token sequence is requested.

        Args:
            bytes_to_free: Target bytes to free (used for estimation).

        Returns:
            Number of bytes freed (estimated).
        """
        if self.paged_cache_manager is None or self.paged_ssd_cache_manager is None:
            return 0

        if self.memory_monitor is None:
            return 0

        # Estimate how many blocks to evict
        num_blocks_to_evict = self.memory_monitor.estimate_blocks_to_free(
            bytes_to_free, self.block_size
        )

        # Get evictable blocks in LRU order
        evictable = self.paged_cache_manager.get_evictable_blocks(num_blocks_to_evict)

        if not evictable:
            logger.debug("No evictable blocks found")
            return 0

        evicted_count = 0

        for block in evictable:
            # In paged SSD-only mode, data is already on paged SSD
            # Just evict the block metadata
            if self.paged_cache_manager.evict_block_permanently(block.block_id):
                evicted_count += 1

        # Estimate bytes freed based on block count
        estimated_freed = evicted_count * self.memory_monitor.estimate_block_memory(
            self.block_size
        )

        if evicted_count > 0:
            logger.info(
                f"Evicted {evicted_count} blocks from index "
                f"(data preserved on paged SSD, ~{format_bytes(estimated_freed)} metadata freed)"
            )

        return estimated_freed

    def restore_block_from_cold(self, block_id: int, block_hash: bytes) -> bool:
        """
        Restore a block from cold storage (deprecated in paged SSD-only mode).

        In paged SSD-only mode, blocks don't store cache_data. Data is loaded
        directly from SSD when needed via reconstruct_cache().

        Kept for API compatibility.

        Args:
            block_id: Block ID to restore.
            block_hash: Block's content hash.

        Returns:
            True if block exists in cold storage.
        """
        if self.paged_ssd_cache_manager is None or self.paged_cache_manager is None:
            return False

        # In paged SSD-only mode, just verify block exists on paged SSD
        if not self.paged_ssd_cache_manager.has_block(block_hash):
            logger.warning(f"Block {block_id} not found in cold storage")
            return False

        # Touch the block to update LRU
        blocks = self.paged_cache_manager.blocks
        if block_id < len(blocks):
            block = blocks[block_id]
            if block:
                block.touch()

        logger.debug(
            f"Block {block_id} verified on paged SSD (hash={block_hash.hex()[:16]}...)"
        )
        return True

    def restore_cold_blocks_for_request(self, request_id: str) -> int:
        """
        Verify all blocks needed for a request exist on paged SSD.

        In paged SSD-only mode, blocks don't store cache_data. This method
        just verifies that blocks exist on paged SSD.

        Args:
            request_id: Request ID.

        Returns:
            Number of blocks verified on paged SSD.
        """
        if self.paged_cache_manager is None or self.paged_ssd_cache_manager is None:
            return 0

        if self.block_aware_cache is None:
            return 0

        # Get block table for request
        block_table = self.paged_cache_manager.request_tables.get(request_id)
        if block_table is None:
            return 0

        verified = 0
        for block_id in block_table.block_ids:
            blocks = self.paged_cache_manager.blocks
            if block_id < len(blocks):
                block = blocks[block_id]
                if block and block.block_hash is not None:
                    if self.restore_block_from_cold(block_id, block.block_hash):
                        verified += 1

        return verified

    def get_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get tiered cache statistics.

        Returns:
            Dictionary with cache statistics.
        """
        stats = {}

        if self.paged_ssd_cache_manager is not None:
            stats["ssd_cache"] = self.paged_ssd_cache_manager.get_stats()

        if self.paged_cache_manager is not None:
            # In paged SSD-only mode, all cache data is on paged SSD
            stats["indexed_blocks"] = self.paged_cache_manager.cold_block_count
            stats["block_size"] = self.block_size

        return stats if stats else None
