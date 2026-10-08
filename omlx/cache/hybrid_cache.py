# SPDX-License-Identifier: Apache-2.0
"""
Hybrid cache configuration for models with mixed cache types.

This module provides configuration classes for models that use different
cache types across layers (e.g., Qwen3-Next with ArraysCache + KVCache).
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple
import logging

from .type_handlers import CacheType, CacheTypeHandler
from .type_registry import CacheTypeRegistry

logger = logging.getLogger(__name__)


@dataclass
class LayerCacheConfig:
    """Configuration for a single layer's cache.

    Attributes:
        layer_idx: Index of the layer
        cache_type: Type of cache used by this layer
        supports_block_slicing: Whether this layer's cache can be block-sliced
        class_name: Original mlx-lm class name
    """

    layer_idx: int
    cache_type: CacheType
    supports_block_slicing: bool
    class_name: str = ""

    @property
    def handler(self) -> CacheTypeHandler:
        """Get the handler for this layer's cache type."""
        return CacheTypeRegistry.get_handler(self.cache_type)


@dataclass
class ModelCacheConfig:
    """Cache configuration for an entire model.

    Contains per-layer cache type information and model-level metadata.
    This is used to properly handle hybrid models that mix different
    cache types (e.g., KVCache + ArraysCache).

    Attributes:
        model_name: Name of the model
        num_layers: Total number of layers
        layer_configs: Per-layer cache configurations
        is_hybrid: True if model uses multiple cache types
        sliceable_layer_count: Number of layers that support block slicing
    """

    model_name: str = ""
    num_layers: int = 0
    layer_configs: List[LayerCacheConfig] = field(default_factory=list)
    is_hybrid: bool = False
    sliceable_layer_count: int = 0

    @classmethod
    def from_cache_list(
        cls,
        cache_list: List[Any],
        model_name: str = "",
    ) -> "ModelCacheConfig":
        """Create configuration from mlx-lm cache list.

        Args:
            cache_list: List of cache objects from model.make_cache()
            model_name: Optional model name for identification

        Returns:
            ModelCacheConfig with per-layer type information
        """
        if not cache_list:
            return cls(model_name=model_name)

        layer_configs = []
        cache_types_seen = set()
        sliceable_count = 0
        max_window_size = 0

        for idx, cache_obj in enumerate(cache_list):
            cache_type = CacheTypeRegistry.detect_cache_type(cache_obj)
            handler = CacheTypeRegistry.get_handler(cache_type)
            class_name = CacheTypeRegistry.canonical_name(cache_obj)

            cache_types_seen.add(cache_type)
            if handler.supports_block_slicing:
                sliceable_count += 1

            # Extract window_size from RotatingKVCache layers
            if cache_type == CacheType.ROTATING_KVCACHE:
                window_size = getattr(cache_obj, "max_size", 0)
                if window_size > max_window_size:
                    max_window_size = window_size

            # Extract window_size from CacheList sub-caches (e.g., RotatingKVCache inside)
            if cache_type == CacheType.CACHE_LIST:
                sub_caches = getattr(cache_obj, "caches", ())
                for sub_c in sub_caches:
                    sub_type = CacheTypeRegistry.detect_cache_type(sub_c)
                    if sub_type == CacheType.ROTATING_KVCACHE:
                        ws = getattr(sub_c, "max_size", 0)
                        if ws > max_window_size:
                            max_window_size = ws

            layer_configs.append(
                LayerCacheConfig(
                    layer_idx=idx,
                    cache_type=cache_type,
                    supports_block_slicing=handler.supports_block_slicing,
                    class_name=class_name,
                )
            )

        config = cls(
            model_name=model_name,
            num_layers=len(cache_list),
            layer_configs=layer_configs,
            is_hybrid=len(cache_types_seen) > 1,
            sliceable_layer_count=sliceable_count,
        )
        config._max_window_size = max_window_size
        return config

    @classmethod
    def from_type_list(
        cls,
        cache_types: List[str],
        model_name: str = "",
    ) -> "ModelCacheConfig":
        """Create configuration from list of type names.

        Useful for reconstructing config from serialized metadata.

        Args:
            cache_types: List of cache type names (e.g., ["KVCache", "ArraysCache"])
            model_name: Optional model name

        Returns:
            ModelCacheConfig
        """
        if not cache_types:
            return cls(model_name=model_name)

        layer_configs = []
        cache_types_seen = set()
        sliceable_count = 0

        for idx, type_name in enumerate(cache_types):
            handler = CacheTypeRegistry.get_handler_by_class_name(type_name)
            cache_type = handler.cache_type

            cache_types_seen.add(cache_type)
            if handler.supports_block_slicing:
                sliceable_count += 1

            layer_configs.append(
                LayerCacheConfig(
                    layer_idx=idx,
                    cache_type=cache_type,
                    supports_block_slicing=handler.supports_block_slicing,
                    class_name=type_name,
                )
            )

        return cls(
            model_name=model_name,
            num_layers=len(cache_types),
            layer_configs=layer_configs,
            is_hybrid=len(cache_types_seen) > 1,
            sliceable_layer_count=sliceable_count,
        )

    def get_sliceable_layers(self) -> List[int]:
        """Get indices of layers that support block slicing.

        Returns:
            List of layer indices
        """
        return [cfg.layer_idx for cfg in self.layer_configs if cfg.supports_block_slicing]

    def get_non_sliceable_layers(self) -> List[int]:
        """Get indices of layers that don't support block slicing.

        Returns:
            List of layer indices (e.g., ArraysCache, RotatingKVCache layers)
        """
        return [
            cfg.layer_idx for cfg in self.layer_configs if not cfg.supports_block_slicing
        ]

    def get_layer_type(self, layer_idx: int) -> CacheType:
        """Get cache type for a specific layer.

        Args:
            layer_idx: Layer index

        Returns:
            CacheType enum
        """
        if 0 <= layer_idx < len(self.layer_configs):
            return self.layer_configs[layer_idx].cache_type
        return CacheType.KVCACHE  # Default

    def get_layer_handler(self, layer_idx: int) -> CacheTypeHandler:
        """Get handler for a specific layer.

        Args:
            layer_idx: Layer index

        Returns:
            CacheTypeHandler instance
        """
        if 0 <= layer_idx < len(self.layer_configs):
            return self.layer_configs[layer_idx].handler
        return CacheTypeRegistry.get_handler(CacheType.KVCACHE)

    def get_type_names(self) -> List[str]:
        """Get list of cache type names for serialization.

        Returns:
            List of class name strings
        """
        return [cfg.class_name for cfg in self.layer_configs]

    def get_meta_states(self, cache_list: List[Any]) -> List[Tuple]:
        """Extract meta_states from cache objects.

        For CacheList layers, the meta_state is a composite:
        ([class_names], [sub_meta_states]).

        Args:
            cache_list: List of cache objects

        Returns:
            List of meta_state tuples
        """
        meta_states = []
        for idx, cache_obj in enumerate(cache_list):
            if idx < len(self.layer_configs):
                handler = self.layer_configs[idx].handler

                if self.layer_configs[idx].cache_type == CacheType.CACHE_LIST:
                    # CacheList: extract composite meta_state from sub-caches
                    state = handler.extract_state(cache_obj)
                    meta_state = (
                        state.get("sub_class_names", []),
                        state.get("sub_meta_states", []),
                    )
                    meta_states.append(meta_state)
                else:
                    state = handler.extract_state(cache_obj)
                    meta_state = state.get("meta_state", ())
                    if not meta_state:
                        # Try to extract offset at minimum
                        offset = state.get("offset", 0)
                        meta_state = (offset,)
                    meta_states.append(meta_state)
            else:
                meta_states.append(())
        return meta_states

    def supports_full_block_slicing(self) -> bool:
        """Check if all layers support block slicing.

        Returns:
            True if all layers can be block-sliced
        """
        return self.sliceable_layer_count == self.num_layers

    def has_rotating_layers(self) -> bool:
        """Check if model has any RotatingKVCache layers.

        Returns:
            True if any layer uses RotatingKVCache
        """
        return any(
            cfg.cache_type == CacheType.ROTATING_KVCACHE for cfg in self.layer_configs
        )

    def get_max_window_size(self) -> int:
        """Get maximum window size from RotatingKVCache layers.

        The window size is extracted from RotatingKVCache meta_state during
        from_cache_list() construction, or can be set manually via
        _max_window_size attribute.

        Returns:
            Maximum window_size, or 0 if no RotatingKVCache layers
        """
        return getattr(self, "_max_window_size", 0)

    def __repr__(self) -> str:
        types = [cfg.class_name for cfg in self.layer_configs[:5]]
        if len(self.layer_configs) > 5:
            types.append(f"... +{len(self.layer_configs) - 5} more")
        return (
            f"ModelCacheConfig(model='{self.model_name}', "
            f"layers={self.num_layers}, "
            f"hybrid={self.is_hybrid}, "
            f"types={types})"
        )


def create_default_kvcache_config(num_layers: int, model_name: str = "") -> ModelCacheConfig:
    """Create a default KVCache-only configuration.

    Convenience function for models that use only KVCache.

    Args:
        num_layers: Number of model layers
        model_name: Optional model name

    Returns:
        ModelCacheConfig with all KVCache layers
    """
    layer_configs = [
        LayerCacheConfig(
            layer_idx=idx,
            cache_type=CacheType.KVCACHE,
            supports_block_slicing=True,
            class_name="KVCache",
        )
        for idx in range(num_layers)
    ]

    return ModelCacheConfig(
        model_name=model_name,
        num_layers=num_layers,
        layer_configs=layer_configs,
        is_hybrid=False,
        sliceable_layer_count=num_layers,
    )
