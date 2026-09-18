# SPDX-License-Identifier: Apache-2.0
"""Non-default eviction policy for the expert cache.

``S3FIFOExpertCache`` subclasses ``ExpertLRUCache`` and keeps its
per-layer caps, retain_hot, stats and the speculation
hooks — only the victim-selection order differs. Selected via
OMLX_EXPERT_STREAMING_CACHE (or the per-model
``expert_streaming_cache_policy`` setting) in
``expert_cache.make_expert_cache``; the default stays ``lru``.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict

from .expert_cache import (
    ExpertLRUCache,
    _layer_index_add,
    _layer_index_drop,
    _layer_index_touch,
)

# Membership sentinel for the ghost queue (its stored values are all
# None, so a plain ``pop(key, None)`` cannot detect a hit).
_MISSING: Any = object()


class S3FIFOExpertCache(ExpertLRUCache):
    """S3-FIFO eviction behind the ExpertLRUCache interface.

    LRU keeps recency; MoE routing is skewed (heavy hitters + scan-like
    prefill demand), where S3-FIFO's scan resistance wins: a small FIFO
    filters one-hit wonders, the main queue holds reuse, and a ghost
    queue promotes re-referenced entries (2nd chance). Per-layer caps,
    retain_hot, stats and the spec hooks are
    inherited unchanged — only the global eviction order differs.
    Select with OMLX_EXPERT_STREAMING_CACHE=s3fifo (default lru).
    """

    def __init__(
        self,
        budget_bytes: int,
        per_slot_bytes: int | None = None,
        num_layers: int | None = None,
        *,
        per_expert_bytes: int | None = None,
    ):
        super().__init__(
            budget_bytes,
            per_slot_bytes,
            num_layers,
            per_expert_bytes=per_expert_bytes,
        )
        self._small: OrderedDict = OrderedDict()
        self._ghost: OrderedDict = OrderedDict()
        # Per-queue, per-layer victim indexes: the base class's
        # _layer_orders only covers _store. _small_layers[layer] /
        # _main_layers[layer] order each queue's keys for one layer; a
        # layer-local victim takes the small queue's head first
        # (probation semantics: one-hit wonders leave before
        # re-referenced main entries).
        self._small_layers: Dict[int, OrderedDict] = {}
        self._main_layers: Dict[int, OrderedDict] = {}
        self._derive_queue_caps()

    def _derive_queue_caps(self) -> None:
        """Re-derive the small/ghost bounds from the current capacity."""
        cap = max(1, self.capacity)
        # Small must cover a full decode token across all layers —
        # per_layer_cap slots x num_layers — or every small entry churns
        # before its re-reference and main stays empty. With per-layer
        # caps active this approaches capacity (a documented limit of
        # S3-FIFO under per-layer quotas).
        per_layer = max(1, self._per_layer_cap) if self.num_layers > 0 else 0
        working = per_layer * max(1, self.num_layers) if self.num_layers > 0 else cap // 10
        self._small_cap = max(1, min(cap - 1, max(cap // 10, working)))
        self._ghost_cap = max(self._small_cap, cap // 10)

    @property
    def policy(self) -> str:
        return "s3fifo"

    def __contains__(self, key: tuple[int, int, str]) -> bool:
        with self._lock:
            return key in self._small or key in self._store

    def peek(self, key: tuple[int, int, str]) -> Any | None:
        """Non-counting probe (small queue counts as resident)."""
        with self._lock:
            return self._small.get(key, self._store.get(key))

    def peek_many(self, keys: list) -> list:
        with self._lock:
            small, store = self._small, self._store
            return [small.get(k, store.get(k)) for k in keys]

    def resident_keys(self) -> set:
        """Both queues are resident: probationary ``_small`` rows serve
        demand reads exactly like main-queue rows, so the adaptive-topk
        residency probe must count them (the base implementation only
        sees ``_store``)."""
        with self._lock:
            return set(self._store) | set(self._small)

    # -- per-queue per-layer victim indexes ---------------------------------
    # _small_layers / _main_layers ride the same module-level index
    # helpers as the base class's _layer_orders (index map as first arg).

    def _get_unlocked(self, key: tuple[int, int, str]) -> Any | None:
        layer = self._layer_of(key)
        if key in self._small:
            # Promote small -> main on re-reference (frequency signal).
            val = self._small.pop(key)
            _layer_index_drop(self._small_layers, layer, key)
            self._store[key] = val
            _layer_index_add(self._main_layers, layer, key)
            self.stats.hits += 1
            return val
        if key in self._store:
            self._store.move_to_end(key)
            _layer_index_touch(self._main_layers, layer, key)
            self.stats.hits += 1
            return self._store[key]
        self.stats.misses += 1
        # Parity with the base class: decode-phase misses feed the
        # governor's per-layer targeting map.
        if self._active_decode:
            self.stats.note_miss(layer)
        return None

    def _put_unlocked(self, key: tuple[int, int, str], value: Any) -> None:
        if self.capacity <= 0:
            return
        layer = self._layer_of(key)
        if key in self._small:
            self._small[key] = value
            return
        if key in self._store:
            self._store.move_to_end(key)
            _layer_index_touch(self._main_layers, layer, key)
            self._store[key] = value
            return
        # Per-layer cap (governor overrides + phase pair); a cap hit that
        # found no layer victim falls back to the global queue drain.
        if not self._enforce_layer_cap(layer):
            self._evict_one_global_unlocked()
        # Ghost hit -> main queue (2nd chance); else small queue.
        target_main = self._ghost.pop(key, _MISSING) is not _MISSING
        while self._resident_count_unlocked() >= self._global_cap_active():
            self._evict_one_global_unlocked()
        if target_main:
            self._store[key] = value
            _layer_index_add(self._main_layers, layer, key)
        else:
            if len(self._small) >= self._small_cap:
                old_k = self._pop_small_head_unlocked()
                self.stats.evictions += 1
                self._dec_layer_count(self._layer_of(old_k))
            self._small[key] = value
            _layer_index_add(self._small_layers, layer, key)
        self._layer_counts[layer] = self._layer_counts.get(layer, 0) + 1

    def _pop_small_head_unlocked(self):
        """Evict the small FIFO's head into the ghost queue; return it."""
        old_k, _ = self._small.popitem(last=False)
        _layer_index_drop(self._small_layers, self._layer_of(old_k), old_k)
        self._ghost[old_k] = None
        while len(self._ghost) > self._ghost_cap:
            self._ghost.popitem(last=False)
        return old_k

    def _evict_layer_unlocked(self, layer: int) -> bool:
        """Evict one entry of `layer` — small queue first (probation)."""
        for store, index in (
            (self._small, self._small_layers.get(layer)),
            (self._store, self._main_layers.get(layer)),
        ):
            if self._evict_indexed_unlocked(layer, store, index):
                return True
        # Fallback scan: entries that bypassed index maintenance (none
        # should exist — defensive only).
        return self._evict_layer_scan_unlocked(
            layer, (self._small, self._store)
        )

    def _evict_one_global_unlocked(self) -> None:
        if len(self._small):
            old_k = self._pop_small_head_unlocked()
        elif len(self._store):
            old_k, _ = self._store.popitem(last=False)
            _layer_index_drop(self._main_layers, self._layer_of(old_k), old_k)
        else:
            return
        self.stats.evictions += 1
        self._dec_layer_count(self._layer_of(old_k))

    def _clear_unlocked(self) -> None:
        super()._clear_unlocked()
        self._small.clear()
        self._ghost.clear()
        self._small_layers.clear()
        self._main_layers.clear()

    def _resize_unlocked(self, capacity: int, per_layer_cap: int | None = None) -> None:
        super()._resize_unlocked(capacity, per_layer_cap)
        # The queue bounds derive from capacity: a shrink that left them
        # stale would refill the small FIFO to the old bound before the
        # global drain could act — the cache never really shrank.
        self._derive_queue_caps()
        while len(self._small) > self._small_cap:
            old_k = self._pop_small_head_unlocked()
            self.stats.evictions += 1
            self._dec_layer_count(self._layer_of(old_k))

    def _drain_to_unlocked(self, cap: int) -> None:
        """S3-FIFO: drain both queues, demoting small victims to ghost."""
        while self._resident_count_unlocked() > cap:
            self._evict_one_global_unlocked()

    def _retain_hot_unlocked(self, hot_pairs: set) -> int:
        if self.capacity <= 0 or (not self._store and not self._small):
            return 0
        evicted = 0
        for store in (self._small, self._store):
            for key in list(store.keys()):
                if (key[0], key[1]) not in hot_pairs:
                    del store[key]
                    evicted += 1
        if evicted:
            self._layer_counts, (self._small_layers, self._main_layers) = (
                self._rebuild_layer_index(self._small, self._store)
            )
            self.stats.evictions += evicted
        return evicted

    def _resident_count_unlocked(self) -> int:
        # Both queues hold resident rows — probationary small-queue rows
        # serve demand reads exactly like main-queue rows, so occupancy
        # (admission fullness, size) counts them.
        return len(self._small) + len(self._store)

