# SPDX-License-Identifier: Apache-2.0
"""Cache type handlers for PoolingCache and BatchPoolingCache.

DeepSeek V4 uses these caches for the compressed (sliding-window) attention
path. They are not sliceable on a per-token basis because the pool is
compressed in fixed ``ratio``-sized windows. Both handlers expose a full
state round-trip (extract → reconstruct) so SSD eviction and recovery work,
but ``supports_block_slicing = False`` keeps the prefix cache from trying
to dedup partial windows.
"""

from __future__ import annotations

import logging
from typing import Any

from omlx.cache.type_handlers import (
    CacheStateAxisInfo,
    CacheType,
    CacheTypeHandler,
)

logger = logging.getLogger(__name__)


class PoolingCacheHandler(CacheTypeHandler):
    """Handler for ``mlx_lm.models.cache.PoolingCache`` (single-sequence)."""

    @property
    def cache_type(self) -> CacheType:
        return CacheType.POOLING_CACHE

    @property
    def supports_block_slicing(self) -> bool:
        # Compressed pool — partial slicing is not meaningful.
        return False

    def extract_state(self, cache_obj: Any) -> dict[str, Any]:
        buf_kv, buf_gate, pooled = cache_obj.state
        return {
            "buf_kv": buf_kv,
            "buf_gate": buf_gate,
            "pooled": pooled,
            "cache_type": self.cache_type.value,
        }

    def get_seq_len(self, state: dict[str, Any]) -> int:
        pooled = state.get("pooled")
        if pooled is not None and hasattr(pooled, "shape") and len(pooled.shape) >= 2:
            return int(pooled.shape[1])
        return 0

    def slice_state(
        self,
        state: dict[str, Any],
        start_idx: int,
        end_idx: int,
    ) -> dict[str, Any] | None:
        # PoolingCache is not block-sliceable; return the full state with a
        # marker so the storage layer treats it as opaque.
        return {**state, "is_full_state": True}

    def concatenate_states(
        self,
        states: list[dict[str, Any]],
    ) -> dict[str, Any]:
        # Per-block concatenation is not supported. Use the most recent
        # state — same convention as RotatingKVCacheHandler.
        return states[-1] if states else {}

    def reconstruct_cache(
        self,
        state: dict[str, Any],
        meta_state: tuple | None = None,
    ) -> Any:
        try:
            from mlx_lm.models.cache import PoolingCache
        except ImportError:
            logger.error("mlx_lm.models.cache.PoolingCache unavailable for reconstruct")
            return None

        ratio = meta_state if isinstance(meta_state, int) else 1
        cache = PoolingCache(ratio=ratio)
        cache.state = (
            state.get("buf_kv"),
            state.get("buf_gate"),
            state.get("pooled"),
        )
        return cache

    def get_state_axis_info(self) -> tuple[CacheStateAxisInfo, ...]:
        # PoolingCache.state = (buf_kv, buf_gate, pooled).
        # buf_kv / buf_gate are remainder windows of shape (B, ratio, D);
        # pooled is the accumulated compressed sequence (B, P, D). The
        # quantization at ``ratio`` makes per-token slicing unsafe — keep
        # all three elements non-sliceable so omlx core takes the
        # last-block-only / boundary-snapshot path.
        return (
            CacheStateAxisInfo(name="buf_kv", sequence_axis=1, sliceable=False),
            CacheStateAxisInfo(name="buf_gate", sequence_axis=1, sliceable=False),
            CacheStateAxisInfo(name="pooled", sequence_axis=1, sliceable=False),
        )

    def deserialize_state(
        self,
        elements: tuple[Any, ...],
        meta_state: Any | None = None,
    ) -> Any:
        """Reconstruct PoolingCache from a 3-tuple state directly.

        omlx core dispatches handlers via ``deserialize_state`` instead of
        the legacy keys/values dict so 3-tuple state survives without
        getting truncated by the default 2-tuple mapping.
        """
        if not isinstance(elements, (list, tuple)):
            logger.error(
                "PoolingCache deserialize: expected tuple, got %s",
                type(elements).__name__,
            )
            return None
        # Tolerate length-2 input (legacy V2-truncated state); fill the
        # missing pooled with None so reconstruct doesn't crash.
        if len(elements) == 2:
            buf_kv, buf_gate = elements
            pooled = None
        elif len(elements) == 3:
            buf_kv, buf_gate, pooled = elements
        else:
            logger.error(
                "PoolingCache deserialize: expected 2 or 3 elements, got %d",
                len(elements),
            )
            return None
        return self.reconstruct_cache(
            {"buf_kv": buf_kv, "buf_gate": buf_gate, "pooled": pooled},
            meta_state,
        )

    def _get_state_keys(self) -> tuple[str, ...]:
        return ("buf_kv", "buf_gate", "pooled")

    def _get_meta_state_keys(self) -> tuple[str, ...]:
        return ("ratio",)


class BatchPoolingCacheHandler(CacheTypeHandler):
    """Handler for ``mlx_lm.models.cache.BatchPoolingCache``.

    BatchPoolingCache state is the same 3-tuple as PoolingCache but kept
    untrimmed; meta_state is a 4-tuple
    ``(ratio, remainder, _pool_lengths, _processed)``.
    """

    @property
    def cache_type(self) -> CacheType:
        return CacheType.BATCH_POOLING_CACHE

    @property
    def supports_block_slicing(self) -> bool:
        return False

    def extract_state(self, cache_obj: Any) -> dict[str, Any]:
        buf_kv, buf_gate, pooled = cache_obj.state
        return {
            "buf_kv": buf_kv,
            "buf_gate": buf_gate,
            "pooled": pooled,
            "cache_type": self.cache_type.value,
            "is_full_state": True,
        }

    def get_seq_len(self, state: dict[str, Any]) -> int:
        pooled = state.get("pooled")
        if pooled is not None and hasattr(pooled, "shape") and len(pooled.shape) >= 2:
            return int(pooled.shape[1])
        return 0

    def slice_state(
        self,
        state: dict[str, Any],
        start_idx: int,
        end_idx: int,
    ) -> dict[str, Any] | None:
        return {**state, "is_full_state": True}

    def concatenate_states(
        self,
        states: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return states[-1] if states else {}

    def reconstruct_cache(
        self,
        state: dict[str, Any],
        meta_state: tuple | None = None,
    ) -> Any:
        try:
            from mlx_lm.models.cache import BatchPoolingCache
        except ImportError:
            logger.error(
                "mlx_lm.models.cache.BatchPoolingCache unavailable for reconstruct"
            )
            return None

        if not isinstance(meta_state, tuple) or len(meta_state) != 4:
            logger.error(
                "BatchPoolingCache reconstruct expects 4-tuple meta_state "
                "(ratio, remainder, pool_lengths, processed); got %r",
                type(meta_state).__name__,
            )
            return None

        ratio, remainder, pool_lengths, processed = meta_state
        batch_size = len(remainder)
        cache = BatchPoolingCache(ratio=ratio, left_padding=[0] * batch_size)
        cache.state = (
            state.get("buf_kv"),
            state.get("buf_gate"),
            state.get("pooled"),
        )
        cache.meta_state = (ratio, list(remainder), list(pool_lengths), list(processed))
        return cache

    def get_state_axis_info(self) -> tuple[CacheStateAxisInfo, ...]:
        return (
            CacheStateAxisInfo(name="buf_kv", sequence_axis=1, sliceable=False),
            CacheStateAxisInfo(name="buf_gate", sequence_axis=1, sliceable=False),
            CacheStateAxisInfo(name="pooled", sequence_axis=1, sliceable=False),
        )

    def deserialize_state(
        self,
        elements: tuple[Any, ...],
        meta_state: Any | None = None,
    ) -> Any:
        if not isinstance(elements, (list, tuple)):
            logger.error(
                "BatchPoolingCache deserialize: expected tuple, got %s",
                type(elements).__name__,
            )
            return None
        if len(elements) == 2:
            buf_kv, buf_gate = elements
            pooled = None
        elif len(elements) == 3:
            buf_kv, buf_gate, pooled = elements
        else:
            logger.error(
                "BatchPoolingCache deserialize: expected 2 or 3 elements, got %d",
                len(elements),
            )
            return None
        return self.reconstruct_cache(
            {"buf_kv": buf_kv, "buf_gate": buf_gate, "pooled": pooled},
            meta_state,
        )

    def _get_state_keys(self) -> tuple[str, ...]:
        return ("buf_kv", "buf_gate", "pooled")

    def _get_meta_state_keys(self) -> tuple[str, ...]:
        return ("ratio", "remainder", "pool_lengths", "processed")
