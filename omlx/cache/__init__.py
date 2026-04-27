# SPDX-License-Identifier: Apache-2.0
"""
Cache module - unified cache management for oMLX.

This package contains cache implementations including:
- Prefix caching for KV state reuse
- Paged cache for memory-efficient KV storage
- VLM cache for vision-language model optimization
- SSD cache for disk-based persistence
"""

# Stats
from .stats import (
    BaseCacheStats,
    PagedCacheStats,
    VLMCacheStats,
    PagedSSDCacheStats,
)

# Interfaces
from .interface import CacheManager

# Paged cache implementations
from .paged_cache import (
    PagedCacheManager,
    CacheBlock,
    BlockTable,
    FreeKVCacheBlockQueue,
    BlockHashToBlockMap,
    BlockHash,
    compute_block_hash,
)

# Prefix cache implementations (SSD-only)
from .prefix_cache import (
    BlockAwarePrefixCache,
    BlockCacheEntry,
)

# Paged SSD cache implementations
from .paged_ssd_cache import (
    PagedSSDCacheManager,
    PagedSSDBlockMetadata,
    PagedSSDCacheIndex,
    parse_size,
)

# Vision feature cache
from .vision_feature_cache import (
    VisionFeatureSSDCache,
    VisionFeatureSSDEntry,
)

# Managers
from .tiered_manager import TieredCacheManager
from .recovery import CacheRecoveryManager

# Factory
from .factory import CacheConfig, CacheFactory

# Type handlers
from .type_handlers import (
    CacheType,
    CacheTypeHandler,
    CacheStateInfo,
    KVCacheHandler,
    RotatingKVCacheHandler,
    ArraysCacheHandler,
    CacheListHandler,
    DefaultCacheHandler,
    SizedArraysCache,
)

# Type registry
from .type_registry import CacheTypeRegistry

# Hybrid cache config
from .hybrid_cache import (
    LayerCacheConfig,
    ModelCacheConfig,
    create_default_kvcache_config,
)

__all__ = [
    # Stats
    "BaseCacheStats",
    "PagedCacheStats",
    "VLMCacheStats",
    "PagedSSDCacheStats",
    # Interfaces
    "CacheManager",
    # Paged cache
    "PagedCacheManager",
    "CacheBlock",
    "BlockTable",
    "FreeKVCacheBlockQueue",
    "BlockHashToBlockMap",
    "BlockHash",
    "compute_block_hash",
    # Prefix cache (SSD-only)
    "BlockAwarePrefixCache",
    "BlockCacheEntry",
    # Paged SSD cache
    "PagedSSDCacheManager",
    "PagedSSDBlockMetadata",
    "PagedSSDCacheIndex",
    "parse_size",
    # Vision feature cache
    "VisionFeatureSSDCache",
    "VisionFeatureSSDEntry",
    # Managers
    "TieredCacheManager",
    "CacheRecoveryManager",
    # Factory
    "CacheConfig",
    "CacheFactory",
    # Type handlers
    "CacheType",
    "CacheTypeHandler",
    "CacheStateInfo",
    "KVCacheHandler",
    "RotatingKVCacheHandler",
    "ArraysCacheHandler",
    "CacheListHandler",
    "DefaultCacheHandler",
    "SizedArraysCache",
    # Type registry
    "CacheTypeRegistry",
    # Hybrid cache config
    "LayerCacheConfig",
    "ModelCacheConfig",
    "create_default_kvcache_config",
]
