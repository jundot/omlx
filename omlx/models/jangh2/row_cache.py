"""Bounded CPU packed-row cache: exact checkpoint data, never GPU tensor graphs."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np


class PackedRowCache:
    """Payload is pre-sized on first insertion and never exceeds byte_limit.

    The payload budget excludes the bounded Python LRU dictionary / integer
    bookkeeping. Access is serialized by the owning PLE reader's lock.
    """

    def __init__(self, packed_width: int, aux_width: int, byte_limit: int):
        if type(byte_limit) is not int or byte_limit < 0:
            raise ValueError("row_cache_bytes must be a nonnegative integer")
        self.packed_width, self.aux_width = packed_width, aux_width
        self.row_bytes = 4 * (packed_width + 2 * aux_width)
        self.capacity = byte_limit // self.row_bytes
        self._slots: OrderedDict[int, int] = OrderedDict()
        self._arrays = None
        self._next = 0
        self.hits = self.misses = self.evictions = 0

    def get(self, row_id: int):
        slot = self._slots.get(int(row_id))
        if slot is None:
            self.misses += 1
            return None
        self.hits += 1
        self._slots.move_to_end(int(row_id))
        # Copies never become dangling aliases when this slot is evicted.
        return tuple(a[slot].copy() for a in self._arrays)

    def get_many(self, row_ids):
        """Gather cache hits in three bulk copies instead of three per row.

        LRU order and hit/miss counts match scalar get() in the input order.
        Returned arrays never alias cache storage. Caller holds the reader lock.
        """
        row_ids = np.asarray(row_ids)
        if row_ids.ndim != 1 or not np.issubdtype(row_ids.dtype, np.integer):
            raise ValueError(
                "Packed row cache IDs must be a one-dimensional integer array"
            )
        hits, slots, misses = [], [], []
        for position, row_id in enumerate(row_ids):
            key = int(row_id)
            slot = self._slots.get(key)
            if slot is None:
                misses.append(position)
            else:
                hits.append(position)
                slots.append(slot)
                self._slots.move_to_end(key)
        self.hits += len(hits)
        self.misses += len(misses)
        positions = np.asarray(hits, dtype=np.intp)
        if slots:
            slot_ids = np.asarray(slots, dtype=np.intp)
            values = tuple(a[slot_ids] for a in self._arrays)
        else:
            values = (
                np.empty((0, self.packed_width), np.uint32),
                np.empty((0, self.aux_width), np.float32),
                np.empty((0, self.aux_width), np.float32),
            )
        return positions, values, np.asarray(misses, dtype=np.intp)

    def put(self, row_id: int, packed, scales, biases):
        if not self.capacity:
            return
        if self._arrays is None:
            self._arrays = (
                np.empty((self.capacity, self.packed_width), np.uint32),
                np.empty((self.capacity, self.aux_width), np.float32),
                np.empty((self.capacity, self.aux_width), np.float32),
            )
        key = int(row_id)
        slot = self._slots.get(key)
        if slot is None:
            if len(self._slots) >= self.capacity:
                _, slot = self._slots.popitem(last=False)
                self.evictions += 1
            else:
                slot = self._next
                self._next += 1
        self._slots[key] = slot
        self._slots.move_to_end(key)
        for arr, value in zip(self._arrays, (packed, scales, biases)):
            arr[slot] = value

    def stats(self):
        return {
            "row_cache_capacity": self.capacity,
            "row_cache_rows": len(self._slots),
            "row_cache_payload_bytes": sum(a.nbytes for a in self._arrays)
            if self._arrays
            else 0,
            "row_cache_hits": self.hits,
            "row_cache_misses": self.misses,
            "row_cache_evictions": self.evictions,
        }

    def close(self):
        self._slots.clear()
        self._arrays = None
        self._next = 0
