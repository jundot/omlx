# SPDX-License-Identifier: Apache-2.0
"""
Unified cache statistics for oMLX.

This module provides base classes and utilities for tracking cache performance
metrics across different cache implementations (prefix, paged, VLM, paged SSD).
"""

from dataclasses import dataclass, asdict, field
from typing import Any, Dict


@dataclass
class BaseCacheStats:
    """
    Base statistics for all cache implementations.

    This class provides common metrics shared across different cache types.
    Subclasses can extend with additional type-specific metrics.
    """

    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def total_queries(self) -> int:
        """Get total number of cache queries."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """
        Calculate cache hit rate.

        Returns:
            Hit rate as a float between 0.0 and 1.0.
        """
        total = self.total_queries
        if total == 0:
            return 0.0
        return self.hits / total

    def record_hit(self) -> None:
        """Record a cache hit."""
        self.hits += 1

    def record_miss(self) -> None:
        """Record a cache miss."""
        self.misses += 1

    def record_eviction(self) -> None:
        """Record a cache eviction."""
        self.evictions += 1

    def reset(self) -> None:
        """Reset all statistics to zero."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert stats to dictionary.

        Returns:
            Dictionary with all stats fields.
        """
        d = asdict(self)
        # Add computed properties
        d["total_queries"] = self.total_queries
        d["hit_rate"] = self.hit_rate
        return d


@dataclass
class PrefixCacheStats(BaseCacheStats):
    """
    Statistics for prefix cache performance.

    Extends base stats with tokens_saved to track efficiency and
    partial-block skip metrics for observability.
    """

    tokens_saved: int = 0
    partial_block_skips: int = 0
    partial_tokens_skipped: int = 0
    block_size: int = 0
    last_partial_tokens_skipped: int = 0
    last_tokens_to_next_block: int = 0
    _total_queries: int = field(default=0, repr=False)

    @property
    def total_queries(self) -> int:
        """Get total number of cache queries."""
        # Use explicit counter if set, otherwise compute from hits + misses
        if self._total_queries > 0:
            return self._total_queries
        return self.hits + self.misses

    @total_queries.setter
    def total_queries(self, value: int) -> None:
        """Set total queries counter (for legacy compatibility)."""
        self._total_queries = value

    def reset(self) -> None:
        """Reset all statistics to zero."""
        super().reset()
        self.tokens_saved = 0
        self.partial_block_skips = 0
        self.partial_tokens_skipped = 0
        self.last_partial_tokens_skipped = 0
        self.last_tokens_to_next_block = 0
        self._total_queries = 0


@dataclass
class PagedCacheStats(BaseCacheStats):
    """
    Statistics for paged KV cache performance.

    Extends base stats with block-level metrics.
    """

    total_blocks: int = 0
    allocated_blocks: int = 0
    free_blocks: int = 0
    shared_blocks: int = 0  # Blocks with ref_count > 1
    total_tokens_cached: int = 0
    cow_copies: int = 0  # Copy-on-write operations

    @property
    def utilization(self) -> float:
        """
        Calculate block utilization rate.

        Returns:
            Utilization as a float between 0.0 and 1.0.
        """
        if self.total_blocks == 0:
            return 0.0
        return self.allocated_blocks / self.total_blocks

    def reset(self) -> None:
        """Reset runtime statistics (not capacity metrics)."""
        super().reset()
        self.cow_copies = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        d = super().to_dict()
        d["utilization"] = self.utilization
        return d


@dataclass
class VLMCacheStats(BaseCacheStats):
    """
    Statistics for VLM (Vision Language Model) cache performance.

    Extends base stats with VLM-specific metrics.
    """

    tokens_saved: int = 0
    image_cache_hits: int = 0

    def record_image_hit(self) -> None:
        """Record an image cache hit."""
        self.image_cache_hits += 1

    def reset(self) -> None:
        """Reset all statistics to zero."""
        super().reset()
        self.tokens_saved = 0
        self.image_cache_hits = 0


@dataclass
class PagedSSDCacheStats(BaseCacheStats):
    """
    Statistics for paged SSD cache performance.

    Extends base stats with storage-specific and hot cache metrics.
    """

    saves: int = 0
    loads: int = 0
    errors: int = 0
    total_size_bytes: int = 0
    max_size_bytes: int = 0
    configured_max_size_bytes: int = 0
    num_files: int = 0

    # Hot cache (in-memory tier) metrics
    hot_cache_entries: int = 0
    hot_cache_size_bytes: int = 0
    hot_cache_max_bytes: int = 0
    hot_cache_hits: int = 0
    hot_cache_evictions: int = 0
    hot_cache_promotions: int = 0

    @property
    def save_rate(self) -> float:
        """Calculate successful save rate."""
        total = self.saves + self.errors
        if total == 0:
            return 0.0
        return self.saves / total

    def record_save(self) -> None:
        """Record a successful save operation."""
        self.saves += 1

    def record_load(self) -> None:
        """Record a successful load operation."""
        self.loads += 1
        self.hits += 1

    def record_error(self) -> None:
        """Record an error."""
        self.errors += 1

    def reset(self) -> None:
        """Reset runtime statistics."""
        super().reset()
        self.saves = 0
        self.loads = 0
        self.errors = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        d = super().to_dict()
        d["save_rate"] = self.save_rate
        return d
