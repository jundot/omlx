# SPDX-License-Identifier: Apache-2.0
"""
Cache type handlers for different KV cache implementations.

This module provides abstract and concrete handlers for various cache types
from mlx-lm, enabling type-aware cache operations like slicing and reconstruction.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Try to import mlx for tensor operations
try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None


class CacheType(Enum):
    """Supported cache types from mlx-lm."""

    KVCACHE = "KVCache"
    ROTATING_KVCACHE = "RotatingKVCache"
    BATCH_KVCACHE = "BatchKVCache"
    BATCH_ROTATING_KVCACHE = "BatchRotatingKVCache"
    ARRAYS_CACHE = "ArraysCache"
    QUANTIZED_KVCACHE = "QuantizedKVCache"
    CACHE_LIST = "CacheList"


@dataclass
class CacheStateInfo:
    """Information about a cache state for serialization."""

    cache_type: str
    state_keys: Tuple[str, ...]
    meta_state_keys: Tuple[str, ...]
    supports_block_slicing: bool
    is_full_state: bool = False


class CacheTypeHandler(ABC):
    """Abstract handler for cache type-specific operations.

    Each handler implements operations specific to a cache type,
    including state extraction, slicing, and reconstruction.
    """

    @property
    @abstractmethod
    def cache_type(self) -> CacheType:
        """Return the cache type this handler manages."""
        pass

    @property
    @abstractmethod
    def supports_block_slicing(self) -> bool:
        """Whether this cache type supports sequence-level block slicing."""
        pass

    @abstractmethod
    def extract_state(self, cache_obj: Any) -> Dict[str, Any]:
        """Extract serializable state from cache object.

        Args:
            cache_obj: The mlx-lm cache object (KVCache, ArraysCache, etc.)

        Returns:
            Dictionary containing state tensors and metadata
        """
        pass

    @abstractmethod
    def get_seq_len(self, state: Dict[str, Any]) -> int:
        """Get sequence length from state.

        Args:
            state: State dictionary from extract_state()

        Returns:
            Sequence length (number of tokens)
        """
        pass

    @abstractmethod
    def slice_state(
        self,
        state: Dict[str, Any],
        start_idx: int,
        end_idx: int,
    ) -> Optional[Dict[str, Any]]:
        """Slice state for block-level storage.

        Args:
            state: State dictionary from extract_state()
            start_idx: Start token index
            end_idx: End token index (exclusive)

        Returns:
            Sliced state dictionary, or None if slicing not supported
        """
        pass

    @abstractmethod
    def concatenate_states(
        self,
        states: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Concatenate multiple block states into one.

        Args:
            states: List of state dictionaries to concatenate

        Returns:
            Combined state dictionary
        """
        pass

    @abstractmethod
    def reconstruct_cache(
        self,
        state: Dict[str, Any],
        meta_state: Optional[Tuple] = None,
    ) -> Any:
        """Reconstruct cache object from stored state.

        Args:
            state: State dictionary (may be concatenated)
            meta_state: Optional metadata (offset, etc.)

        Returns:
            Reconstructed mlx-lm cache object
        """
        pass

    def get_state_info(self) -> CacheStateInfo:
        """Get information about this cache type's state structure."""
        return CacheStateInfo(
            cache_type=self.cache_type.value,
            state_keys=self._get_state_keys(),
            meta_state_keys=self._get_meta_state_keys(),
            supports_block_slicing=self.supports_block_slicing,
        )

    def _get_state_keys(self) -> Tuple[str, ...]:
        """Return keys used in state dictionary."""
        return ("keys", "values")

    def _get_meta_state_keys(self) -> Tuple[str, ...]:
        """Return keys used in meta_state."""
        return ("offset",)


class KVCacheHandler(CacheTypeHandler):
    """Handler for standard KVCache (4D tensors).

    KVCache uses:
    - keys: shape (batch, n_kv_heads, seq_len, head_dim)
    - values: shape (batch, n_kv_heads, seq_len, head_dim)
    - offset: current sequence length
    """

    @property
    def cache_type(self) -> CacheType:
        return CacheType.KVCACHE

    @property
    def supports_block_slicing(self) -> bool:
        return True

    def extract_state(self, cache_obj: Any) -> Dict[str, Any]:
        """Extract state from KVCache object."""
        keys, values = cache_obj.state
        return {
            "keys": keys,
            "values": values,
            "offset": getattr(cache_obj, "offset", keys.shape[2] if keys is not None else 0),
            "cache_type": self.cache_type.value,
        }

    def get_seq_len(self, state: Dict[str, Any]) -> int:
        """Get sequence length from keys tensor."""
        keys = state.get("keys")
        if keys is not None and hasattr(keys, "shape") and len(keys.shape) >= 3:
            return keys.shape[2]
        return state.get("offset", 0)

    def slice_state(
        self,
        state: Dict[str, Any],
        start_idx: int,
        end_idx: int,
    ) -> Optional[Dict[str, Any]]:
        """Slice keys and values along sequence dimension (axis 2)."""
        if not HAS_MLX:
            return None

        keys = state.get("keys")
        values = state.get("values")

        if keys is None or values is None:
            return None

        try:
            # Slice along axis 2 (sequence dimension)
            # Shape: (batch, n_kv_heads, seq_len, head_dim)
            keys_slice = keys[:, :, start_idx:end_idx, :]
            values_slice = values[:, :, start_idx:end_idx, :]

            return {
                "keys": keys_slice,
                "values": values_slice,
                "cache_type": self.cache_type.value,
            }
        except Exception as e:
            logger.warning(f"Failed to slice KVCache state: {e}")
            return None

    def concatenate_states(
        self,
        states: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Concatenate multiple KVCache states along sequence dimension."""
        if not HAS_MLX or not states:
            return {}

        keys_list = [s["keys"] for s in states if s.get("keys") is not None]
        values_list = [s["values"] for s in states if s.get("values") is not None]

        if not keys_list or not values_list:
            return {}

        concat_keys = mx.concatenate(keys_list, axis=2)
        concat_values = mx.concatenate(values_list, axis=2)

        return {
            "keys": concat_keys,
            "values": concat_values,
            "offset": concat_keys.shape[2],
            "cache_type": self.cache_type.value,
        }

    def reconstruct_cache(
        self,
        state: Dict[str, Any],
        meta_state: Optional[Tuple] = None,
    ) -> Any:
        """Reconstruct KVCache from state."""
        try:
            from mlx_lm.models.cache import KVCache
        except ImportError:
            logger.error("mlx_lm not available for cache reconstruction")
            return None

        keys = state.get("keys")
        values = state.get("values")

        if keys is None or values is None:
            return None

        cache = KVCache()
        cache.keys = keys
        cache.values = values

        # Set offset from meta_state or infer from tensor shape
        if meta_state and len(meta_state) > 0:
            cache.offset = int(meta_state[0])
        else:
            cache.offset = state.get("offset", keys.shape[2])

        return cache


class RotatingKVCacheHandler(CacheTypeHandler):
    """Handler for RotatingKVCache (sliding window attention).

    RotatingKVCache uses:
    - keys/values: shape (batch, n_kv_heads, max_size, head_dim)
    - offset: total tokens processed
    - _idx: current rotation index
    - max_size: maximum window size
    - keep: tokens to always keep

    IMPORTANT: RotatingKVCache does NOT support block slicing because:
    1. The cache has a fixed max_size and uses circular buffer semantics
    2. The _idx pointer tracks the current position in the circular buffer
    3. Slicing would break the rotation index and cause shape mismatches
    4. When merged with other caches, all must have same max_size
    """

    @property
    def cache_type(self) -> CacheType:
        return CacheType.ROTATING_KVCACHE

    @property
    def supports_block_slicing(self) -> bool:
        return False  # Cannot safely slice rotating cache

    def extract_state(self, cache_obj: Any) -> Dict[str, Any]:
        """Extract state from RotatingKVCache object."""
        keys, values = cache_obj.state

        # Get meta_state: (keep, max_size, offset, _idx)
        meta_state = getattr(cache_obj, "meta_state", ())

        return {
            "keys": keys,
            "values": values,
            "offset": getattr(cache_obj, "offset", 0),
            "max_size": getattr(cache_obj, "max_size", keys.shape[2] if keys is not None else 0),
            "keep": getattr(cache_obj, "keep", 0),
            "_idx": getattr(cache_obj, "_idx", 0),
            "meta_state": meta_state,
            "cache_type": self.cache_type.value,
        }

    def get_seq_len(self, state: Dict[str, Any]) -> int:
        """Get effective sequence length (offset, not buffer size)."""
        return state.get("offset", 0)

    def slice_state(
        self,
        state: Dict[str, Any],
        start_idx: int,
        end_idx: int,
    ) -> Optional[Dict[str, Any]]:
        """RotatingKVCache cannot be sliced by sequence position.

        Returns the full state instead, similar to ArraysCache.
        The circular buffer semantics make slicing unsafe.
        """
        return {
            "keys": state.get("keys"),
            "values": state.get("values"),
            "meta_state": state.get("meta_state", ()),
            "max_size": state.get("max_size"),
            "offset": state.get("offset"),
            "keep": state.get("keep"),
            "_idx": state.get("_idx"),
            "is_full_state": True,
            "cache_type": self.cache_type.value,
        }

    def concatenate_states(
        self,
        states: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """For RotatingKVCache, use the most recent state.

        Rotating cache states cannot be concatenated like KV caches
        because they use circular buffer semantics.
        """
        if not states:
            return {}

        # Use the last (most recent) state
        latest = states[-1]
        return {
            "keys": latest.get("keys"),
            "values": latest.get("values"),
            "meta_state": latest.get("meta_state", ()),
            "max_size": latest.get("max_size"),
            "offset": latest.get("offset"),
            "keep": latest.get("keep"),
            "_idx": latest.get("_idx"),
            "is_full_state": True,
            "cache_type": self.cache_type.value,
        }

    def reconstruct_cache(
        self,
        state: Dict[str, Any],
        meta_state: Optional[Tuple] = None,
    ) -> Any:
        """Reconstruct RotatingKVCache from state."""
        try:
            from mlx_lm.models.cache import RotatingKVCache
        except ImportError:
            logger.error("mlx_lm not available for cache reconstruction")
            return None

        keys = state.get("keys")
        values = state.get("values")

        if keys is None or values is None:
            return None

        # Parse meta_state: (keep, max_size, offset, _idx)
        if meta_state and len(meta_state) >= 4:
            keep, max_size, offset, _idx = map(int, meta_state[:4])
        else:
            # Use defaults from state or tensor shape
            keep = state.get("keep", 0)
            max_size = state.get("max_size", keys.shape[2])
            offset = state.get("offset", keys.shape[2])
            _idx = state.get("_idx", 0)

        # Backward-compatible canonicalization for oversized rotating snapshots.
        # Older snapshots could persist prefill-internal states where seq_len >
        # max_size (e.g., max_size + chunk_size - 1). Those states are not
        # merge-safe when reintroduced as per-request prompt caches.
        if (
            hasattr(keys, "shape")
            and len(keys.shape) >= 3
            and max_size > 0
            and keys.shape[2] > max_size
        ):
            if keep > 0 and keep < max_size and HAS_MLX and mx is not None:
                tail_len = max_size - keep
                keys = mx.concatenate(
                    [keys[..., :keep, :], keys[..., -tail_len:, :]],
                    axis=2,
                )
                values = mx.concatenate(
                    [values[..., :keep, :], values[..., -tail_len:, :]],
                    axis=2,
                )
            else:
                keys = keys[..., -max_size:, :]
                values = values[..., -max_size:, :]

            if HAS_MLX and mx is not None:
                keys = mx.contiguous(keys)
                values = mx.contiguous(values)

            _idx = min(max_size, keys.shape[2])

        if hasattr(keys, "shape") and len(keys.shape) >= 3:
            seq_len = keys.shape[2]
            _idx = min(max(0, int(_idx)), seq_len, max_size if max_size > 0 else seq_len)

        cache = RotatingKVCache(max_size=max_size, keep=keep)
        cache.keys = keys
        cache.values = values
        cache.offset = offset
        cache._idx = _idx

        return cache

    def _get_meta_state_keys(self) -> Tuple[str, ...]:
        return ("keep", "max_size", "offset", "_idx")


class SizedArraysCache:
    """ArraysCache wrapper that provides a correct size() method.

    mlx-lm's ArraysCache.size() always returns 0 because _BaseCache.size()
    returns 0 by default. This causes BatchGenerator batch ordering issues
    when sorting by prompt length + cache size.

    This wrapper tracks token_count and delegates all other methods to the
    inner ArraysCache, ensuring BatchGenerator sees the correct cache size.
    """

    def __init__(self, inner_cache: Any, token_count: int = 0):
        """Initialize the wrapper.

        Args:
            inner_cache: The ArraysCache to wrap.
            token_count: Number of tokens this cache represents.
        """
        self._inner = inner_cache
        self._token_count = token_count

    def size(self) -> int:
        """Return the cached token count (instead of 0)."""
        return self._token_count

    def empty(self) -> bool:
        """Delegate to inner cache."""
        return self._inner.empty()

    @property
    def state(self):
        """Delegate to inner cache."""
        return self._inner.state

    @state.setter
    def state(self, v):
        """Delegate to inner cache."""
        self._inner.state = v

    @property
    def cache(self):
        """Delegate to inner cache."""
        return self._inner.cache

    def __getitem__(self, idx):
        """Delegate to inner cache."""
        return self._inner[idx]

    def __setitem__(self, idx, value):
        """Delegate to inner cache."""
        self._inner[idx] = value

    def __len__(self):
        """Return length of inner cache's state list.

        ArraysCache doesn't have __len__, so we return len(cache) instead.
        """
        return len(self._inner.cache)

    def __getattr__(self, name):
        """Delegate unknown attributes to inner cache.

        This handles attributes like 'lengths', 'left_padding', etc.
        that may be set by BatchGenerator.prepare().
        """
        # Avoid infinite recursion for _inner
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        """Set attributes on inner cache for non-wrapper attributes."""
        if name in ("_inner", "_token_count"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._inner, name, value)

    # BatchGenerator interface methods
    def prepare(self, **kwargs):
        """Delegate to inner cache."""
        return self._inner.prepare(**kwargs)

    def finalize(self):
        """Delegate to inner cache."""
        return self._inner.finalize()

    def advance(self, N):
        """Delegate to inner cache."""
        return self._inner.advance(N)

    def make_mask(self, N):
        """Delegate to inner cache."""
        return self._inner.make_mask(N)

    def filter(self, batch_indices):
        """Delegate to inner cache."""
        return self._inner.filter(batch_indices)

    def extend(self, other):
        """Delegate to inner cache."""
        # Unwrap if other is also a SizedArraysCache
        other_inner = other._inner if isinstance(other, SizedArraysCache) else other
        return self._inner.extend(other_inner)

    def extract(self, idx):
        """Extract and wrap to preserve token_count."""
        extracted = self._inner.extract(idx)
        return SizedArraysCache(extracted, self._token_count)

    @classmethod
    def merge(cls, caches: List["SizedArraysCache"]) -> "SizedArraysCache":
        """Merge multiple caches, preserving size information."""
        inner_caches = [c._inner if isinstance(c, cls) else c for c in caches]
        # Use first inner cache's merge method
        merged_inner = inner_caches[0].merge(inner_caches)
        # Preserve token_count from first cache
        token_count = caches[0]._token_count if isinstance(caches[0], cls) else 0
        return cls(merged_inner, token_count)


class ArraysCacheHandler(CacheTypeHandler):
    """Handler for generic ArraysCache (multiple state arrays).

    ArraysCache is a base class for caches with variable number of states.
    """

    @property
    def cache_type(self) -> CacheType:
        return CacheType.ARRAYS_CACHE

    @property
    def supports_block_slicing(self) -> bool:
        return False  # Generic arrays may not be sequence-indexed

    def extract_state(self, cache_obj: Any) -> Dict[str, Any]:
        """Extract state from ArraysCache object."""
        # Unwrap if wrapped in SizedArraysCache
        inner = cache_obj._inner if isinstance(cache_obj, SizedArraysCache) else cache_obj
        state_list = inner.state if hasattr(inner, "state") else inner.cache

        return {
            "states": list(state_list) if state_list else [],
            "is_full_state": True,
            "cache_type": self.cache_type.value,
        }

    def get_seq_len(self, state: Dict[str, Any]) -> int:
        return state.get("token_count", 0)

    def slice_state(
        self,
        state: Dict[str, Any],
        start_idx: int,
        end_idx: int,
    ) -> Optional[Dict[str, Any]]:
        # Return full state
        return {
            "states": state.get("states", []),
            "is_full_state": True,
            "cache_type": self.cache_type.value,
        }

    def concatenate_states(
        self,
        states: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not states:
            return {}
        # Use latest state
        return states[-1]

    def reconstruct_cache(
        self,
        state: Dict[str, Any],
        meta_state: Optional[Tuple] = None,
        token_count: int = 0,
    ) -> Any:
        """Reconstruct ArraysCache from state.

        Args:
            state: State dictionary with 'states' key.
            meta_state: Optional metadata (unused for ArraysCache).
            token_count: Number of tokens this cache represents.
                Used by SizedArraysCache wrapper for correct size() return.

        Returns:
            SizedArraysCache wrapping the reconstructed ArraysCache.
        """
        try:
            from mlx_lm.models.cache import ArraysCache
        except ImportError:
            logger.error("mlx_lm not available for cache reconstruction")
            return None

        states = state.get("states", [])
        cache = ArraysCache(size=len(states))
        for i, s in enumerate(states):
            cache.cache[i] = s

        # Wrap with SizedArraysCache to provide correct size()
        return SizedArraysCache(cache, token_count)

    def _get_state_keys(self) -> Tuple[str, ...]:
        return ("states",)

    def _get_meta_state_keys(self) -> Tuple[str, ...]:
        return ()


class CacheListHandler(CacheTypeHandler):
    """Handler for CacheList (composite cache with multiple sub-caches).

    CacheList wraps multiple sub-caches (e.g., KVCache + ArraysCache) into a
    single per-layer cache object. Used by models like deepseek_v32 (MLA),
    falcon_h1 (Mamba + Attention), baichuan_m1 (SSM + Attention).

    Uses last-block-only storage: only the last block stores full state,
    non-last blocks get placeholders. Partial prefix match → reject.
    """

    # Normalize sub-cache class names for mlx-lm CacheList.from_state() compat
    _CLASS_NAME_NORMALIZE = {
        "SizedArraysCache": "ArraysCache",
    }

    @property
    def cache_type(self) -> CacheType:
        return CacheType.CACHE_LIST

    @property
    def supports_block_slicing(self) -> bool:
        return False  # Mixed sub-cache types prevent slicing

    def extract_state(self, cache_obj: Any) -> Dict[str, Any]:
        """Extract state from CacheList object.

        Iterates over sub-caches and extracts each one's state, meta_state,
        and class_name individually.

        Returns:
            Dictionary with sub_states, sub_class_names, sub_meta_states,
            cache_type, and is_full_state fields.
        """
        sub_caches = getattr(cache_obj, "caches", None)
        if not sub_caches:
            return {
                "sub_states": [],
                "sub_class_names": [],
                "sub_meta_states": [],
                "cache_type": self.cache_type.value,
                "is_full_state": True,
            }

        sub_states = []
        sub_class_names = []
        sub_meta_states = []

        for sc in sub_caches:
            # Get state
            if hasattr(sc, "state"):
                sub_states.append(sc.state)
            else:
                sub_states.append(())

            # Get class name (normalize SizedArraysCache → ArraysCache)
            raw_name = type(sc).__name__
            # Unwrap SizedArraysCache
            if isinstance(sc, SizedArraysCache):
                raw_name = "ArraysCache"
            normalized = self._CLASS_NAME_NORMALIZE.get(raw_name, raw_name)
            sub_class_names.append(normalized)

            # Get meta_state
            sub_meta_states.append(getattr(sc, "meta_state", ()))

        return {
            "sub_states": sub_states,
            "sub_class_names": sub_class_names,
            "sub_meta_states": sub_meta_states,
            "cache_type": self.cache_type.value,
            "is_full_state": True,
        }

    def get_seq_len(self, state: Dict[str, Any]) -> int:
        """Get sequence length from sub-caches.

        Returns the maximum seq_len found among sub-caches that have
        4D tensors (batch, n_kv_heads, seq_len, head_dim).
        """
        max_seq_len = 0
        sub_states = state.get("sub_states", [])
        for sub_state in sub_states:
            if isinstance(sub_state, (list, tuple)) and len(sub_state) >= 2:
                sub_keys = sub_state[0]
                if hasattr(sub_keys, "shape") and len(sub_keys.shape) == 4:
                    max_seq_len = max(max_seq_len, sub_keys.shape[2])
        return max_seq_len

    def slice_state(
        self,
        state: Dict[str, Any],
        start_idx: int,
        end_idx: int,
    ) -> Optional[Dict[str, Any]]:
        """CacheList cannot be sliced — return full state."""
        return {
            "sub_states": state.get("sub_states", []),
            "sub_class_names": state.get("sub_class_names", []),
            "sub_meta_states": state.get("sub_meta_states", []),
            "is_full_state": True,
            "cache_type": self.cache_type.value,
        }

    def concatenate_states(
        self,
        states: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Use the most recent (last) state."""
        if not states:
            return {}
        return states[-1]

    def reconstruct_cache(
        self,
        state: Dict[str, Any],
        meta_state: Optional[Tuple] = None,
    ) -> Any:
        """Reconstruct CacheList from stored state.

        Tries new mlx-lm CacheList.from_state() first, then falls back to
        manual sub-cache reconstruction using CacheTypeRegistry (local import
        to avoid circular dependency).

        Args:
            state: Dict with 'sub_states' key containing per-sub-cache states.
            meta_state: Tuple of ([class_names], [sub_meta_states]).

        Returns:
            Reconstructed CacheList object, or None on failure.
        """
        sub_states = state.get("sub_states", [])
        if not meta_state or not isinstance(meta_state, (list, tuple)) or len(meta_state) < 2:
            logger.error("CacheList reconstruct: missing or invalid meta_state")
            return None

        class_names, sub_meta_states = meta_state[0], meta_state[1]

        # Validate lengths match to prevent silent zip truncation
        if len(sub_states) != len(class_names) or len(sub_states) != len(sub_meta_states):
            logger.error(
                f"CacheList reconstruct: length mismatch — "
                f"sub_states={len(sub_states)}, class_names={len(class_names)}, "
                f"sub_meta_states={len(sub_meta_states)}"
            )
            return None

        # Sanitize sub_meta_states for sub-cache types that don't support
        # meta_state (inherit _BaseCache's strict setter which rejects
        # truthy values).  Use "" to match _BaseCache.meta_state getter.
        _NO_META_STATE_TYPES = frozenset({"KVCache", "ConcatenateKVCache", "ArraysCache"})
        sanitized_sub_meta_states = [
            "" if cls_name in _NO_META_STATE_TYPES else sub_meta
            for cls_name, sub_meta in zip(class_names, sub_meta_states)
        ]

        # Try new mlx-lm CacheList.from_state() first
        try:
            from mlx_lm.models.cache import CacheList

            return CacheList.from_state(sub_states, (class_names, sanitized_sub_meta_states))
        except (ImportError, AttributeError, TypeError, KeyError, Exception) as e:
            logger.debug(f"CacheList.from_state() unavailable or failed: {e}")

        # Fallback: manually reconstruct sub-caches
        # NOTE: CacheTypeRegistry must be imported locally to avoid circular import
        # (type_handlers.py is imported by type_registry.py)
        try:
            from mlx_lm.models.cache import CacheList
        except ImportError:
            logger.error("mlx_lm not available for CacheList reconstruction")
            return None

        from .type_registry import CacheTypeRegistry as _Registry  # local import

        sub_caches = []
        for sub_state, cls_name, sub_meta in zip(sub_states, class_names, sub_meta_states):
            # Normalize class name for handler lookup
            normalized_name = self._CLASS_NAME_NORMALIZE.get(cls_name, cls_name)
            sub_handler = _Registry.get_handler_by_class_name(normalized_name)

            try:
                if normalized_name in ("ArraysCache", "SizedArraysCache"):
                    sub_cache = sub_handler.reconstruct_cache(
                        {"states": list(sub_state)}, sub_meta
                    )
                else:
                    # KVCache / RotatingKVCache: state is (keys, values)
                    if isinstance(sub_state, (list, tuple)) and len(sub_state) >= 2:
                        sub_cache = sub_handler.reconstruct_cache(
                            {"keys": sub_state[0], "values": sub_state[1]}, sub_meta
                        )
                    else:
                        logger.error(
                            f"CacheList fallback: unexpected sub_state format "
                            f"for {cls_name}"
                        )
                        return None
            except Exception as e:
                logger.error(f"CacheList fallback: failed to reconstruct {cls_name}: {e}")
                return None

            if sub_cache is None:
                logger.error(f"CacheList fallback: sub-cache {cls_name} returned None")
                return None

            # Unwrap SizedArraysCache for CacheList (CacheList expects raw ArraysCache)
            if isinstance(sub_cache, SizedArraysCache):
                sub_cache = sub_cache._inner

            sub_caches.append(sub_cache)

        return CacheList(*sub_caches)

    def _get_state_keys(self) -> Tuple[str, ...]:
        return ("sub_states", "sub_class_names", "sub_meta_states")

    def _get_meta_state_keys(self) -> Tuple[str, ...]:
        return ("class_names", "sub_meta_states")


# Default handler for unknown types - falls back to KVCache behavior
class DefaultCacheHandler(KVCacheHandler):
    """Default handler that assumes KVCache-like behavior.

    Used as fallback for unknown cache types.
    """

    @property
    def cache_type(self) -> CacheType:
        return CacheType.KVCACHE

    def extract_state(self, cache_obj: Any) -> Dict[str, Any]:
        """Try to extract state assuming KVCache-like structure."""
        try:
            if hasattr(cache_obj, "state"):
                state = cache_obj.state
                if isinstance(state, tuple) and len(state) == 2:
                    keys, values = state
                    return {
                        "keys": keys,
                        "values": values,
                        "offset": getattr(cache_obj, "offset", 0),
                        "cache_type": "Unknown",
                    }
        except Exception as e:
            logger.warning(f"Failed to extract state from unknown cache type: {e}")

        return {"cache_type": "Unknown"}
