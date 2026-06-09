# SPDX-License-Identifier: Apache-2.0
"""
Factory for creating cache instances from configuration.

This module provides a unified way to instantiate cache components
based on configuration settings.

Note: oMLX only supports paged SSD-based caching. Memory KV cache is managed
by mlx-lm's BatchGenerator. When paged SSD cache is disabled, no oMLX caching
is performed.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .paged_cache import PagedCacheManager
    from .prefix_cache import BlockAwarePrefixCache
    from .paged_ssd_cache import PagedSSDCacheManager
    from ..memory_monitor import MemoryMonitor


@dataclass
class CacheConfig:
    """
    Configuration for cache system (paged SSD-only).

    Note: oMLX only supports paged SSD-based caching via BlockAwarePrefixCache.
    Memory KV cache is managed entirely by mlx-lm's BatchGenerator.

    Attributes:
        block_size: Number of tokens per cache block.
        max_num_blocks: Maximum number of cache blocks.
        initial_blocks: Initial number of blocks to allocate.
        paged_ssd_cache_dir: Directory for paged SSD cache storage. If None, caching is disabled.
        max_paged_ssd_cache_size: Maximum size of paged SSD cache in bytes.
        max_kv_cache_memory: Maximum GPU memory for KV cache.
        model_name: Model name for cache isolation.
    """

    block_size: int = 64
    max_num_blocks: int = 1024
    initial_blocks: int = 256
    paged_ssd_cache_dir: Optional[Path] = None
    max_paged_ssd_cache_size: int = 100 * 1024 * 1024 * 1024  # 100GB
    max_kv_cache_memory: Optional[int] = None
    model_name: str = ""


class CacheFactory:
    """
    Factory for creating cache instances (paged SSD-only).

    This class provides static methods to create cache components
    based on a unified CacheConfig. oMLX only supports paged SSD-based caching
    via BlockAwarePrefixCache.

    Example:
        config = CacheConfig(
            block_size=64,
            paged_ssd_cache_dir=Path("/tmp/cache"),
        )

        paged_cache = CacheFactory.create_paged_cache(config)
        paged_ssd_cache = CacheFactory.create_paged_ssd_cache(config, "llama-3b")
        prefix_cache = CacheFactory.create_prefix_cache(
            config, paged_cache, paged_ssd_cache
        )
    """

    @staticmethod
    def create_paged_cache(
        config: CacheConfig,
        num_layers: Optional[int] = None,
    ) -> Optional["PagedCacheManager"]:
        """
        Create a PagedCacheManager instance.

        Args:
            config: Cache configuration.
            num_layers: Number of model layers (unused, kept for API compat).

        Returns:
            Configured PagedCacheManager instance, or None if paged SSD cache is disabled.
        """
        if config.paged_ssd_cache_dir is None:
            return None

        from .paged_cache import PagedCacheManager

        return PagedCacheManager(
            block_size=config.block_size,
            max_blocks=config.max_num_blocks,
            enable_caching=True,
            model_name=config.model_name,
            initial_blocks=config.initial_blocks,
        )

    @staticmethod
    def create_paged_ssd_cache(
        config: CacheConfig,
        model_name: Optional[str] = None,
    ) -> Optional["PagedSSDCacheManager"]:
        """
        Create a PagedSSDCacheManager instance.

        Args:
            config: Cache configuration.
            model_name: Override model name for cache isolation.

        Returns:
            Configured PagedSSDCacheManager instance, or None if disabled.
        """
        if config.paged_ssd_cache_dir is None:
            return None

        from .paged_ssd_cache import PagedSSDCacheManager

        cache_dir = config.paged_ssd_cache_dir
        if model_name:
            cache_dir = cache_dir / model_name

        return PagedSSDCacheManager(
            cache_dir=cache_dir,
            max_size_bytes=config.max_paged_ssd_cache_size,
            expected_model_name=config.model_name or "",
            expected_block_size=config.block_size,
            expected_block_size_tokens=config.block_size,
            # No memory monitor available here, so leave
            # ``expected_kv_bytes_per_token`` at its 200 KB default.
            # The Scheduler construction path is the one that benefits
            # from the model-derived value; this factory is used mostly
            # by direct/test callers.
        )

    @staticmethod
    def create_prefix_cache(
        config: CacheConfig,
        model: Any = None,
        paged_cache: Optional["PagedCacheManager"] = None,
        paged_ssd_cache: Optional["PagedSSDCacheManager"] = None,
    ) -> Optional["BlockAwarePrefixCache"]:
        """
        Create a BlockAwarePrefixCache instance.

        Note: oMLX only supports paged SSD-based caching via BlockAwarePrefixCache.
        Returns None if paged SSD cache is disabled.

        Args:
            config: Cache configuration.
            model: Model instance for cache identification.
            paged_cache: PagedCacheManager for block-based caching.
            paged_ssd_cache: PagedSSDCacheManager for SSD storage.

        Returns:
            Configured BlockAwarePrefixCache instance, or None if disabled.
        """
        if config.paged_ssd_cache_dir is None or paged_cache is None:
            return None

        from .prefix_cache import BlockAwarePrefixCache

        return BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=paged_ssd_cache,
        )

    @staticmethod
    def create_memory_monitor(
        config: CacheConfig,
        paged_cache: Optional["PagedCacheManager"] = None,
    ) -> "MemoryMonitor":
        """
        Create a MemoryMonitor instance.

        Args:
            config: Cache configuration.
            paged_cache: PagedCacheManager to monitor.

        Returns:
            Configured MemoryMonitor instance.
        """
        from ..memory_monitor import MemoryMonitor

        # Use configured max KV cache memory, or default to 4GB
        max_kv_cache_memory = config.max_kv_cache_memory or (4 * 1024 * 1024 * 1024)

        monitor = MemoryMonitor(
            max_kv_cache_memory=max_kv_cache_memory,
        )

        if paged_cache is not None:
            monitor.set_paged_cache_manager(paged_cache)

        return monitor

    @staticmethod
    def create_full_cache_stack(
        config: CacheConfig,
        model: Any = None,
        num_layers: Optional[int] = None,
    ) -> dict:
        """
        Create a complete cache stack with all components.

        This is a convenience method that creates all cache components
        and wires them together. Returns all None values if paged SSD cache
        is disabled (oMLX only supports paged SSD-based caching).

        Args:
            config: Cache configuration.
            model: Model instance for cache identification.
            num_layers: Number of model layers.

        Returns:
            Dictionary with keys: paged_cache, paged_ssd_cache, prefix_cache, memory_monitor
        """
        paged_cache = None
        paged_ssd_cache = None
        prefix_cache = None
        memory_monitor = None

        # Only create cache components if paged SSD cache is enabled
        if config.paged_ssd_cache_dir is not None:
            paged_cache = CacheFactory.create_paged_cache(config, num_layers)
            paged_ssd_cache = CacheFactory.create_paged_ssd_cache(
                config, config.model_name
            )

            if paged_cache is not None and paged_ssd_cache is not None:
                paged_cache.set_paged_ssd_cache_manager(paged_ssd_cache)

            prefix_cache = CacheFactory.create_prefix_cache(
                config, model, paged_cache, paged_ssd_cache
            )

            if paged_cache is not None:
                memory_monitor = CacheFactory.create_memory_monitor(config, paged_cache)

        return {
            "paged_cache": paged_cache,
            "paged_ssd_cache": paged_ssd_cache,
            "prefix_cache": prefix_cache,
            "memory_monitor": memory_monitor,
        }
