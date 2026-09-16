# SPDX-License-Identifier: Apache-2.0
"""Cross-layer shared attention state for DeepSeek-V4.1 CSA2.

Matches official ``SharedAttentionRuntime`` in inference/model.py: sources
write before consumers read within a single forward, so one slot each is
enough. Pool *identity* is the PoolingCache (or tensor buffer) object that
kv/index source layers own and that consumers reuse.
"""
from __future__ import annotations

from typing import Any, Optional


class SharedAttentionRuntime:
    """Mutable hand-off between CSA2 layers in stack order."""

    def __init__(self) -> None:
        self.compress_kv: Any = None
        self.index_k: Any = None
        self.topk_idxs: Any = None
        self.candidates: Any = None
        # Stable pool objects keyed by owning source layer id (for make_cache
        # identity tests and cross-layer reuse).
        self.kv_pool_by_source: dict[int, Any] = {}
        self.index_pool_by_source: dict[int, Any] = {}

    def reset_forward_slots(self) -> None:
        """Clear per-forward pointers (not the registered pools)."""
        self.compress_kv = None
        self.index_k = None
        self.topk_idxs = None
        self.candidates = None

    def register_kv_pool(self, source_layer_id: int, pool: Any) -> Any:
        existing = self.kv_pool_by_source.get(source_layer_id)
        if existing is not None:
            return existing
        self.kv_pool_by_source[source_layer_id] = pool
        return pool

    def register_index_pool(self, source_layer_id: int, pool: Any) -> Any:
        existing = self.index_pool_by_source.get(source_layer_id)
        if existing is not None:
            return existing
        self.index_pool_by_source[source_layer_id] = pool
        return pool

    def kv_source_for(self, layer_id: int, source_ids: list[int]) -> Optional[int]:
        """Largest kv_source_layer_id <= layer_id (official reuse rule)."""
        candidates = [s for s in source_ids if s <= layer_id]
        return max(candidates) if candidates else None

    def index_source_for(self, layer_id: int, source_ids: list[int]) -> Optional[int]:
        candidates = [s for s in source_ids if s <= layer_id]
        return max(candidates) if candidates else None


# Process-global runtime matching the official module-level singleton.
shared_attn = SharedAttentionRuntime()
