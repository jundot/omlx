# SPDX-License-Identifier: Apache-2.0
"""
Cache type registry for handler lookup.

This module provides a registry for looking up cache type handlers
by cache type enum or class name string.
"""

from typing import Any, Dict, Optional, Type
import logging

from .type_handlers import (
    CacheType,
    CacheTypeHandler,
    KVCacheHandler,
    RotatingKVCacheHandler,
    ArraysCacheHandler,
    CacheListHandler,
    DefaultCacheHandler,
    SizedArraysCache,
)

logger = logging.getLogger(__name__)


class CacheTypeRegistry:
    """Registry for cache type handlers.

    Provides lookup of handlers by:
    - CacheType enum value
    - Class name string (e.g., "KVCache", "ArraysCache")

    Usage:
        handler = CacheTypeRegistry.get_handler(CacheType.KVCACHE)
        handler = CacheTypeRegistry.get_handler_by_class_name("RotatingKVCache")
        cache_type = CacheTypeRegistry.detect_cache_type(cache_obj)
    """

    # Handler instances by cache type
    _handlers: Dict[CacheType, CacheTypeHandler] = {}

    # Mapping from mlx-lm class names to cache types
    _class_name_map: Dict[str, CacheType] = {
        "KVCache": CacheType.KVCACHE,
        "RotatingKVCache": CacheType.ROTATING_KVCACHE,
        # omlx subclass that overrides size() to clamp by actual buffer
        # length (defined in omlx/cache/_rotating_subclass.py). Cache
        # restore serializes type(cache).__name__, so the registry must
        # recognize this name to route through RotatingKVCacheHandler;
        # otherwise the default handler reconstructs vanilla
        # RotatingKVCache and the size() override is lost.
        "PrefillReadyRotatingKVCache": CacheType.ROTATING_KVCACHE,
        "BatchKVCache": CacheType.BATCH_KVCACHE,
        "BatchRotatingKVCache": CacheType.BATCH_ROTATING_KVCACHE,
        "ArraysCache": CacheType.ARRAYS_CACHE,
        "QuantizedKVCache": CacheType.QUANTIZED_KVCACHE,
        "CacheList": CacheType.CACHE_LIST,
        # TurboQuant: handled specially in prefix_cache/paged_ssd_cache,
        # mapped to KVCACHE so supports_block_slicing = True (but prefix_cache
        # checks the class name first and routes to TQ-specific handling)
        "TurboQuantKVCache": CacheType.KVCACHE,
        "BatchTurboQuantKVCache": CacheType.KVCACHE,
    }

    # Default handler instance
    _default_handler: CacheTypeHandler = DefaultCacheHandler()

    @classmethod
    def register(cls, handler: CacheTypeHandler) -> None:
        """Register a handler for a cache type.

        Args:
            handler: Handler instance to register
        """
        cls._handlers[handler.cache_type] = handler
        logger.debug(f"Registered handler for {handler.cache_type.value}")

    @classmethod
    def get_handler(cls, cache_type: CacheType) -> CacheTypeHandler:
        """Get handler for a cache type.

        Args:
            cache_type: The cache type enum

        Returns:
            Handler for the cache type, or default handler if not found
        """
        handler = cls._handlers.get(cache_type)
        if handler is None:
            logger.debug(f"No handler for {cache_type}, using default")
            return cls._default_handler
        return handler

    @classmethod
    def get_handler_by_class_name(cls, class_name: str) -> CacheTypeHandler:
        """Get handler by mlx-lm class name.

        Args:
            class_name: The class name string (e.g., "KVCache")

        Returns:
            Handler for the cache type, or default handler if not found
        """
        # Handle SizedArraysCache wrapper - use ArraysCache handler
        if class_name == "SizedArraysCache":
            return cls.get_handler(CacheType.ARRAYS_CACHE)

        cache_type = cls._class_name_map.get(class_name)
        if cache_type is None:
            logger.debug(f"Unknown cache class '{class_name}', using default handler")
            return cls._default_handler
        return cls.get_handler(cache_type)

    @classmethod
    def detect_cache_type(cls, cache_obj: Any) -> CacheType:
        """Detect cache type from object.

        Args:
            cache_obj: An mlx-lm cache object

        Returns:
            Detected CacheType enum value
        """
        # Handle SizedArraysCache wrapper - detect inner cache type
        if isinstance(cache_obj, SizedArraysCache):
            return CacheType.ARRAYS_CACHE

        class_name = type(cache_obj).__name__
        cache_type = cls._class_name_map.get(class_name)

        if cache_type is None:
            # CacheList: has .caches attribute (tuple/list of sub-caches)
            sub_caches = getattr(cache_obj, "caches", None)
            if isinstance(sub_caches, (list, tuple)) and len(sub_caches) > 0:
                return CacheType.CACHE_LIST

            # Try to detect by checking for known attributes
            if hasattr(cache_obj, "max_size") and hasattr(cache_obj, "_idx"):
                return CacheType.ROTATING_KVCACHE
            elif hasattr(cache_obj, "keys") and hasattr(cache_obj, "values"):
                return CacheType.KVCACHE
            elif hasattr(cache_obj, "cache") and isinstance(
                getattr(cache_obj, "cache", None), list
            ):
                return CacheType.ARRAYS_CACHE

            logger.debug(f"Could not detect cache type for {class_name}, assuming KVCache")
            return CacheType.KVCACHE

        return cache_type

    @classmethod
    def get_handler_for_object(cls, cache_obj: Any) -> CacheTypeHandler:
        """Get handler for a cache object.

        Convenience method that detects type and returns handler.

        Args:
            cache_obj: An mlx-lm cache object

        Returns:
            Appropriate handler for the object
        """
        cache_type = cls.detect_cache_type(cache_obj)
        return cls.get_handler(cache_type)

    @classmethod
    def is_sliceable(cls, cache_obj: Any) -> bool:
        """Check if a cache object supports block slicing.

        Args:
            cache_obj: An mlx-lm cache object

        Returns:
            True if the cache supports sequence-level slicing
        """
        handler = cls.get_handler_for_object(cache_obj)
        return handler.supports_block_slicing

    @classmethod
    def get_class_name_for_type(cls, cache_type: CacheType) -> str:
        """Get mlx-lm class name for a cache type.

        Args:
            cache_type: CacheType enum

        Returns:
            Class name string
        """
        # Reverse lookup
        for class_name, ct in cls._class_name_map.items():
            if ct == cache_type:
                return class_name
        return cache_type.value

    @classmethod
    def list_registered_types(cls) -> list:
        """List all registered cache types.

        Returns:
            List of registered CacheType enums
        """
        return list(cls._handlers.keys())

    @classmethod
    def list_known_class_names(cls) -> list:
        """List all known mlx-lm class names.

        Returns:
            List of class name strings
        """
        return list(cls._class_name_map.keys())


def _initialize_default_handlers() -> None:
    """Initialize default handlers on module load."""
    CacheTypeRegistry.register(KVCacheHandler())
    CacheTypeRegistry.register(RotatingKVCacheHandler())
    CacheTypeRegistry.register(ArraysCacheHandler())
    CacheTypeRegistry.register(CacheListHandler())


# Initialize handlers when module is imported
_initialize_default_handlers()
