# SPDX-License-Identifier: Apache-2.0
"""Quantized Engram hot-row cache (FP8 E4M3 + UE8M0 scales).

**Point here for the Engram RAM working-set cache.**

* Env: ``OMLX_DSV41_ENGRAM_CACHE_GB`` (default 50),
  ``OMLX_DSV41_ENGRAM_CACHE_POLICY`` (``lru`` | ``clock``)
* Class: :class:`QuantHotRowCache`
* Bound on: ``DiskEngramEmbedding`` via ``bind_hot_cache`` (SSD offload mode)
* Stats: :func:`engram_cache_stats` (also re-exported from ``engram``)

Full Engram tables stay on SSD via ``numpy.memmap``; only touched rows are
stored here as on-disk FP8+scale bytes (~264 B/row at D=256).
"""
from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .engram_fp8 import _e4m3_lut, _ue8m0_to_float32

def engram_cache_gb() -> float:
    """Total RAM budget (GB) for quantized Engram hot-row cache across layers.

    Default 50. Set 0 to disable. Rows are stored as FP8+UE8M0 (~264 B/row
    at D=256), so 50 GB holds ~200M rows (~2× a bf16 cache).
    """
    raw = os.environ.get("OMLX_DSV41_ENGRAM_CACHE_GB", "50").strip()
    if not raw:
        return 0.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 50.0


def engram_cache_policy() -> str:
    """``lru`` (default, oldest-used) or ``clock``."""
    return os.environ.get("OMLX_DSV41_ENGRAM_CACHE_POLICY", "lru").strip().lower()

class QuantHotRowCache:
    """Fixed-capacity hot-row cache storing **quantized** Engram rows.

    Each slot keeps FP8 E4M3 weights ``[D]`` plus UE8M0 scales ``[D/block]`` —
    the on-disk layout — so a 50 GB budget holds ~2× as many rows as bf16.
    Hits dequant in RAM (LUT); misses come from SSD then populate the cache.
    Eviction: LRU (oldest-used, default) or CLOCK.
    """

    def __init__(
        self,
        capacity_rows: int,
        head_dim: int,
        block_size: int = 32,
        policy: str = "lru",
    ):
        self.capacity = max(0, int(capacity_rows))
        self.head_dim = int(head_dim)
        self.block_size = int(block_size)
        self.n_scales = self.head_dim // self.block_size
        self.policy = (policy or "lru").strip().lower()
        self.enabled = self.capacity > 0
        # uint8 weight + uint8 scale (no padding bookkeeping beyond slot_key/ref)
        self.bytes_per_row = self.head_dim + self.n_scales
        self.hits = 0
        self.misses = 0
        self.inserts = 0
        self.evictions = 0
        self._hand = 0
        self._n = 0
        self._key_to_slot: Dict[int, int] = {}
        self._lru: Optional["OrderedDict[int, int]"] = None
        if not self.enabled:
            self.weight = None
            self.scale = None
            self.slot_key = None
            self.ref = None
            return
        self.weight = np.empty((self.capacity, self.head_dim), dtype=np.uint8)
        self.scale = np.empty((self.capacity, self.n_scales), dtype=np.uint8)
        self.slot_key = np.full(self.capacity, -1, dtype=np.int64)
        self.ref = np.zeros(self.capacity, dtype=np.uint8)
        if self.policy == "lru":
            self._lru = OrderedDict()

    @property
    def size(self) -> int:
        return self._n

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total else 0.0

    def stats(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "capacity_rows": self.capacity,
            "size": self._n,
            "bytes_per_row": self.bytes_per_row,
            "budget_gb": self.capacity * self.bytes_per_row / (1024**3),
            "policy": self.policy,
            "storage": "fp8_e4m3+ue8m0",
            "hits": self.hits,
            "misses": self.misses,
            "inserts": self.inserts,
            "evictions": self.evictions,
            "hit_rate": self.hit_rate,
        }

    def reset_stats(self) -> None:
        self.hits = self.misses = self.inserts = self.evictions = 0

    def _dequant_slots(self, slots: np.ndarray) -> np.ndarray:
        """Dequant cached slots -> float32 [N, D]."""
        if slots.size == 0:
            return np.empty((0, self.head_dim), dtype=np.float32)
        w = self.weight[slots]
        s = self.scale[slots]
        w_f = _e4m3_lut()[w].reshape(slots.shape[0], -1, self.block_size)
        s_f = _ue8m0_to_float32(s)[..., None]
        return (w_f * s_f).reshape(slots.shape[0], self.head_dim).astype(
            np.float32, copy=False
        )

    def get_many_raw(
        self, keys: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Lookup unique keys. Returns (hit_mask, weight_u8, scale_u8).

        ``weight_u8`` / ``scale_u8`` rows are valid only where ``hit_mask`` is true.
        Callers should run the same dequant path used for SSD misses.
        """
        n = int(keys.shape[0])
        hit = np.zeros(n, dtype=bool)
        w_out = np.empty((n, self.head_dim), dtype=np.uint8)
        s_out = np.empty((n, self.n_scales), dtype=np.uint8)
        if not self.enabled or n == 0:
            self.misses += n
            return hit, w_out, s_out
        kmap = self._key_to_slot
        lru = self._lru
        for i in range(n):
            k = int(keys[i])
            slot = kmap.get(k)
            if slot is None:
                self.misses += 1
                continue
            hit[i] = True
            self.hits += 1
            w_out[i] = self.weight[slot]
            s_out[i] = self.scale[slot]
            self.ref[slot] = 1
            if lru is not None:
                lru.move_to_end(k)
        return hit, w_out, s_out

    def get_many(self, keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        """Lookup unique keys. Returns (hit_mask [N], rows_f32 [N, D] valid on hits)."""
        n = int(keys.shape[0])
        hit = np.zeros(n, dtype=bool)
        out = np.empty((n, self.head_dim), dtype=np.float32)
        if not self.enabled or n == 0:
            self.misses += n
            return hit, out
        kmap = self._key_to_slot
        lru = self._lru
        hit_idx: List[int] = []
        slots: List[int] = []
        for i in range(n):
            k = int(keys[i])
            slot = kmap.get(k)
            if slot is None:
                self.misses += 1
                continue
            hit[i] = True
            hit_idx.append(i)
            slots.append(slot)
            self.hits += 1
            if lru is not None:
                lru.move_to_end(k)
        if slots:
            slot_arr = np.asarray(slots, dtype=np.int64)
            out[np.asarray(hit_idx, dtype=np.int64)] = self._dequant_slots(slot_arr)
            self.ref[slot_arr] = 1
        return hit, out

    def _evict_clock(self) -> int:
        cap = self.capacity
        hand = self._hand
        ref = self.ref
        for _ in range(cap * 2):
            if ref[hand] == 0:
                victim = hand
                self._hand = (hand + 1) % cap
                return victim
            ref[hand] = 0
            hand = (hand + 1) % cap
        self._hand = (hand + 1) % cap
        return hand

    def _evict_lru(self) -> int:
        lru = self._lru
        assert lru is not None
        old_key, _ = lru.popitem(last=False)
        slot = self._key_to_slot.pop(old_key)
        self.slot_key[slot] = -1
        return slot

    def put_many(
        self, keys: np.ndarray, weight_u8: np.ndarray, scale_u8: np.ndarray
    ) -> None:
        """Insert quantized rows (uint8 weight + scale) into the cache."""
        if not self.enabled or keys.size == 0:
            return
        kmap = self._key_to_slot
        lru = self._lru
        for i in range(int(keys.shape[0])):
            k = int(keys[i])
            slot = kmap.get(k)
            if slot is not None:
                self.weight[slot] = weight_u8[i]
                self.scale[slot] = scale_u8[i]
                self.ref[slot] = 1
                if lru is not None:
                    lru.move_to_end(k)
                continue
            if self._n < self.capacity:
                slot = self._n
                self._n += 1
            else:
                if self.policy == "lru" and lru is not None:
                    slot = self._evict_lru()
                else:
                    slot = self._evict_clock()
                    old = int(self.slot_key[slot])
                    if old >= 0:
                        kmap.pop(old, None)
                        if lru is not None:
                            lru.pop(old, None)
                self.evictions += 1
            old = int(self.slot_key[slot])
            if old >= 0 and old != k:
                kmap.pop(old, None)
                if lru is not None:
                    lru.pop(old, None)
            self.slot_key[slot] = k
            kmap[k] = slot
            self.weight[slot] = weight_u8[i]
            self.scale[slot] = scale_u8[i]
            self.ref[slot] = 1
            if lru is not None:
                lru[k] = slot
                lru.move_to_end(k)
            self.inserts += 1

def _quant_cache_rows_per_layer(n_layers: int, head_dim: int, block_size: int = 32) -> int:
    """Split ``OMLX_DSV41_ENGRAM_CACHE_GB`` evenly; row = FP8[D] + UE8M0[D/block]."""
    gb = engram_cache_gb()
    if gb <= 0 or n_layers <= 0:
        return 0
    bytes_per_row = int(head_dim) + int(head_dim) // int(block_size)
    total_bytes = int(gb * (1024**3))
    per_layer = total_bytes // n_layers
    return max(0, per_layer // bytes_per_row)

def engram_cache_stats(model: Any = None) -> Dict[int, Dict[str, Any]]:
    """Return per-layer quantized hot-cache stats from a bound model."""
    out: Dict[int, Dict[str, Any]] = {}
    if model is None:
        return out
    root = getattr(model, "language_model", None) or getattr(model, "model", model)
    layers = getattr(root, "layers", None) or getattr(model, "layers", [])
    for i, layer in enumerate(layers):
        eng = getattr(layer, "engram", None) if hasattr(layer, "engram") or (hasattr(layer, "__contains__") and "engram" in layer) else None
        if eng is None:
            try:
                eng = layer.engram if "engram" in layer else None
            except Exception:
                eng = None
        if eng is None:
            continue
        embed = getattr(eng, "embed", None)
        hot = getattr(embed, "_hot_cache", None) if embed is not None else None
        if hot is None or not hot.enabled:
            continue
        layer_id = int(getattr(eng, "layer_id", i))
        out[layer_id] = hot.stats()
    return out
