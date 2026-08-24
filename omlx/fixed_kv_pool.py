# SPDX-License-Identifier: Apache-2.0
"""Materialized, fixed-capacity KV cache storage for MLX text models.

The pool owns every backing array for the lifetime of a loaded engine.  A
minimal one-token forward discovers the cache tree and the exact tensor dtypes
and non-sequence dimensions produced by the loaded model.  Request caches are
lightweight views into rows of those arrays.

mlx-lm normally grows standalone caches and then copies them into new batch
caches.  The wrappers below keep merge, extend, and filter operations inside
the already-allocated rows.  No cache tensor is allowed to grow past the
configured context capacity.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchRotatingKVCache,
    CacheList,
    ChunkedKVCache,
    KVCache,
    RotatingKVCache,
    make_prompt_cache,
)

from omlx.patches.deepseek_v4.cache_extras import BatchPoolingCache, PoolingCache

logger = logging.getLogger(__name__)


class FixedKVCacheError(RuntimeError):
    """Raised when a cache tree cannot use fixed-capacity storage."""


class FixedKVCacheCapacityError(FixedKVCacheError):
    """Raised when a request attempts to write beyond its fixed slot."""


def _filter_indices(indices: Any, size: int) -> list[int]:
    values = indices.tolist() if isinstance(indices, mx.array) else list(indices)
    resolved = [int(value) for value in values]
    if any(value < 0 or value >= size for value in resolved):
        raise FixedKVCacheError(
            f"Fixed-cache filter indices {resolved} exceed batch size {size}"
        )
    if any(left >= right for left, right in zip(resolved, resolved[1:])):
        raise FixedKVCacheError(
            "Fixed-cache compaction requires unique ascending survivor indices"
        )
    return resolved


def _round_up(value: int, step: int) -> int:
    return ((int(value) + step - 1) // step) * step


def _iter_arrays(value: Any) -> Iterable[mx.array]:
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_arrays(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_arrays(item)


def _cache_arrays(cache_tree: Any) -> list[mx.array]:
    arrays: list[mx.array] = []
    seen: set[int] = set()
    for cache in cache_tree:
        found = list(_iter_arrays(getattr(cache, "state", None)))
        if not found:
            # A few architecture-owned caches (notably DeepSeek DSpark) expose
            # their committed tensor directly but intentionally have no
            # mlx-lm ``state`` property.
            found.extend(_iter_arrays(getattr(cache, "keys", None)))
            found.extend(_iter_arrays(getattr(cache, "values", None)))
        for array in found:
            if id(array) not in seen:
                seen.add(id(array))
                arrays.append(array)
    return arrays


def _cache_slot(cache_tree: Any) -> int | None:
    """Find the fixed session row owned by a nested cache tree."""

    if cache_tree is None:
        return None
    if isinstance(cache_tree, (list, tuple)):
        for item in cache_tree:
            slot = _cache_slot(item)
            if slot is not None:
                return slot
        return None
    slot = getattr(cache_tree, "slot", None)
    if slot is not None:
        return int(slot)
    slots = getattr(cache_tree, "slots", None)
    if isinstance(slots, list) and len(slots) == 1:
        return int(slots[0])
    children = getattr(cache_tree, "caches", None)
    if isinstance(children, (list, tuple)):
        return _cache_slot(children)
    return None


def _mtp_hidden(output: Any) -> mx.array | None:
    if isinstance(output, tuple) and len(output) >= 2:
        hidden = output[1]
    else:
        hidden = getattr(output, "hidden_states", None)
    if isinstance(hidden, list):
        hidden = hidden[-1] if hidden else None
    return hidden if isinstance(hidden, mx.array) else None


def _model_flag(model: Any, name: str) -> bool:
    for candidate in (
        model,
        getattr(model, "language_model", None),
        getattr(model, "_language_model", None),
    ):
        if candidate is not None and bool(getattr(candidate, name, False)):
            return True
    return False


class _FixedMTPList(list):
    """MTP cache list with a precommitted speculative-copy factory."""

    def __init__(self, values: list[Any], clone_factory: Any | None = None):
        super().__init__(values)
        self._omlx_fixed_clone_factory = clone_factory


@dataclass
class _KVStorage:
    keys: mx.array
    values: mx.array
    capacity: int
    logical_capacity: int
    rotating: bool = False
    scratch_keys: mx.array | None = None
    scratch_values: mx.array | None = None

    @property
    def slots(self) -> int:
        return int(self.keys.shape[0])

    @property
    def per_slot_nbytes(self) -> int:
        return int((self.keys.nbytes + self.values.nbytes) // self.slots)

    def _require_scratch(self) -> tuple[mx.array, mx.array]:
        if self.scratch_keys is None or self.scratch_values is None:
            raise FixedKVCacheError(
                "Fixed cache row movement requires the launch-time scratch row"
            )
        return self.scratch_keys, self.scratch_values

    def write_row_from(
        self,
        row: int,
        keys: mx.array,
        values: mx.array,
        *,
        length: int,
        destination: int = 0,
    ) -> None:
        """Copy through the committed scratch row without duplicating the pool."""

        if length <= 0:
            return
        scratch_k, scratch_v = self._require_scratch()
        # The scalar operation forces an independent row buffer. A plain view
        # would still alias the pool, causing MLX's immutable slice update to
        # retain a second full pool buffer until evaluation completed.
        scratch_k[:, :, :length] = keys[:, :, :length] + mx.array(
            0, dtype=keys.dtype
        )
        scratch_v[:, :, :length] = values[:, :, :length] + mx.array(
            0, dtype=values.dtype
        )
        mx.eval(scratch_k, scratch_v)
        self.keys[row : row + 1, :, destination : destination + length] = (
            scratch_k[:, :, :length]
        )
        self.values[row : row + 1, :, destination : destination + length] = (
            scratch_v[:, :, :length]
        )
        mx.eval(self.keys, self.values)

    def move_row(self, source: int, destination: int, *, length: int) -> None:
        if source == destination or length <= 0:
            return
        self.write_row_from(
            destination,
            self.keys[source : source + 1, :, :length],
            self.values[source : source + 1, :, :length],
            length=length,
        )

    def shift_row_right(self, row: int, *, amount: int, length: int) -> None:
        if amount <= 0 or length <= 0:
            return
        self.write_row_from(
            row,
            self.keys[row : row + 1, :, :length],
            self.values[row : row + 1, :, :length],
            length=length,
            destination=amount,
        )

    def roll_row_right(self, row: int, *, amount: int, length: int) -> None:
        """Rotate one logical row using only the committed scratch arrays."""

        if length <= 0:
            return
        amount = int(amount) % int(length)
        if amount == 0:
            return
        scratch_k, scratch_v = self._require_scratch()
        scratch_k[:, :, :length] = self.keys[
            row : row + 1, :, :length
        ] + mx.array(0, dtype=self.keys.dtype)
        scratch_v[:, :, :length] = self.values[
            row : row + 1, :, :length
        ] + mx.array(0, dtype=self.values.dtype)
        mx.eval(scratch_k, scratch_v)
        split = length - amount
        self.keys[row : row + 1, :, :amount] = scratch_k[:, :, split:length]
        self.values[row : row + 1, :, :amount] = scratch_v[:, :, split:length]
        self.keys[row : row + 1, :, amount:length] = scratch_k[:, :, :split]
        self.values[row : row + 1, :, amount:length] = scratch_v[:, :, :split]
        mx.eval(self.keys, self.values)


@dataclass
class _DSparkStorage:
    keys: mx.array
    max_size: int
    scratch: mx.array

    @property
    def slots(self) -> int:
        return int(self.keys.shape[0])

    @property
    def per_slot_nbytes(self) -> int:
        return int(self.keys.nbytes // self.slots)


class FixedDSparkContextCache:
    """Committed-only DSpark context ring backed by one fixed pool row."""

    def __init__(self, storage: _DSparkStorage, slot: int):
        self._storage = storage
        self.slot = int(slot)
        self.max_size = int(storage.max_size)
        self.offset = 0
        self._length = 0

    @property
    def keys(self):
        if self._length <= 0:
            return None
        row = self._storage.keys[self.slot : self.slot + 1]
        if self._length < self.max_size:
            return row[:, :, : self._length]
        return row

    @keys.setter
    def keys(self, value):
        if value is None:
            self.offset = 0
            self._length = 0
            return
        length = int(value.shape[2])
        if length > self.max_size:
            raise FixedKVCacheCapacityError(
                f"Restored DSpark context has {length:,} tokens but its fixed "
                f"ring holds {self.max_size:,}."
            )
        self._storage.keys[self.slot : self.slot + 1, :, :length] = value[
            :, :, :length
        ]
        self.offset = length
        self._length = length
        mx.eval(self._storage.keys)

    def _store_full_chronological(self, value: mx.array, next_offset: int) -> None:
        """Store one chronological full window in DSpark physical-ring order."""

        storage = self._storage.keys
        row_index = slice(self.slot, self.slot + 1)
        cutoff = next_offset % self.max_size
        if cutoff == 0:
            storage[row_index] = value
        else:
            split = self.max_size - cutoff
            storage[row_index, :, :cutoff] = value[:, :, split:]
            storage[row_index, :, cutoff:] = value[:, :, :split]
        self._length = self.max_size
        self.offset = next_offset
        mx.eval(self._storage.keys)

    def append(self, keys: mx.array, *, start_offset: int | None = None) -> None:
        if start_offset is not None:
            start = int(start_offset)
            if self.offset not in (0, start):
                raise ValueError(
                    "DSpark context is not contiguous: "
                    f"cache={self.offset}, append={start}"
                )
            if self.offset == 0:
                self.offset = start

        length = int(keys.shape[2])
        if length <= 0:
            return
        next_offset = self.offset + length
        if length >= self.max_size:
            tail = keys[:, :, -self.max_size :]
            self._store_full_chronological(tail, next_offset)
            return

        storage = self._storage.keys
        row_index = slice(self.slot, self.slot + 1)
        row = storage[row_index]
        if self._length == self.max_size:
            start = self.offset % self.max_size
            first = min(length, self.max_size - start)
            storage[row_index, :, start : start + first] = keys[:, :, :first]
            if first < length:
                storage[row_index, :, : length - first] = keys[:, :, first:]
            self.offset = next_offset
            mx.eval(self._storage.keys)
            return

        total = self._length + length
        if total < self.max_size:
            storage[row_index, :, self._length : total] = keys
            self._length = total
            self.offset = next_offset
            mx.eval(self._storage.keys)
            return

        scratch = self._storage.scratch
        keep_old = self.max_size - length
        if keep_old > 0:
            old_start = max(0, self._length - keep_old)
            scratch[:, :, :keep_old] = row[:, :, old_start : self._length]
        scratch[:, :, keep_old:] = keys
        mx.eval(scratch)
        self._store_full_chronological(scratch, next_offset)

    @property
    def logical_length(self) -> int:
        return self._length

    def reset(self) -> None:
        self.offset = 0
        self._length = 0

    def trim(self, n: int) -> None:
        n = min(max(0, int(n)), self._length)
        self._length -= n
        self.offset = max(0, self.offset - n)
        mx.eval(self._storage.keys)

    @property
    def state(self):
        return (self.keys,)

    @state.setter
    def state(self, value):
        keys = value[0] if isinstance(value, (list, tuple)) else value
        self.keys = keys

    def empty(self) -> bool:
        return self._length == 0

    @property
    def nbytes(self) -> int:
        return self._storage.per_slot_nbytes


@dataclass
class _ArraysStorage:
    arrays: list[mx.array | None]
    slots: int
    scratch: list[mx.array | None]

    @property
    def per_slot_nbytes(self) -> int:
        return sum(
            int(array.nbytes // self.slots)
            for array in self.arrays
            if array is not None
        )

    def move_row(self, source: int, destination: int) -> None:
        if source == destination:
            return
        materialized: list[mx.array] = []
        for array, scratch in zip(self.arrays, self.scratch):
            if array is None:
                continue
            if scratch is None:
                raise FixedKVCacheError(
                    "Fixed recurrent row movement requires launch-time scratch"
                )
            scratch[:] = array[source : source + 1] + mx.array(
                0, dtype=array.dtype
            )
            materialized.append(scratch)
        if materialized:
            mx.eval(*materialized)
        updated: list[mx.array] = []
        for array, scratch in zip(self.arrays, self.scratch):
            if array is not None and scratch is not None:
                array[destination : destination + 1] = scratch
                updated.append(array)
        if updated:
            mx.eval(*updated)


@dataclass
class _PoolingStorage:
    pooled: mx.array
    buf_kv: mx.array
    buf_gate: mx.array
    ratio: int
    logical_capacity: int
    prev_win_kv: mx.array | None = None
    prev_win_gate: mx.array | None = None
    undo_buf_kv: mx.array | None = None
    undo_buf_gate: mx.array | None = None
    scratch_pooled: mx.array | None = None
    scratch_buf_kv: mx.array | None = None
    scratch_buf_gate: mx.array | None = None
    scratch_prev_win_kv: mx.array | None = None
    scratch_prev_win_gate: mx.array | None = None

    @property
    def slots(self) -> int:
        return int(self.pooled.shape[0])

    @property
    def per_slot_nbytes(self) -> int:
        return sum(int(array.nbytes // self.slots) for array in self.arrays)

    @property
    def arrays(self) -> tuple[mx.array, ...]:
        return tuple(
            array
            for array in (
                self.pooled,
                self.buf_kv,
                self.buf_gate,
                self.prev_win_kv,
                self.prev_win_gate,
            )
            if array is not None
        )

    @property
    def scratch_arrays(self) -> tuple[mx.array | None, ...]:
        return (
            self.scratch_pooled,
            self.scratch_buf_kv,
            self.scratch_buf_gate,
            self.scratch_prev_win_kv,
            self.scratch_prev_win_gate,
        )

    def move_row(self, source: int, destination: int) -> None:
        if source == destination:
            return
        materialized: list[mx.array] = []
        pairs = list(zip(self.arrays, self.scratch_arrays))
        for array, scratch in pairs:
            if scratch is None:
                raise FixedKVCacheError(
                    "Fixed pooled-cache row movement requires launch-time scratch"
                )
            scratch[:] = array[source : source + 1] + mx.array(
                0, dtype=array.dtype
            )
            materialized.append(scratch)
        if materialized:
            mx.eval(*materialized)
        updated: list[mx.array] = []
        for array, scratch in pairs:
            if scratch is not None:
                array[destination : destination + 1] = scratch
                updated.append(array)
        if updated:
            mx.eval(*updated)


class FixedKVCache(KVCache):
    """Single-sequence view into one row of a fixed full-attention pool."""

    def __init__(self, storage: _KVStorage, slot: int):
        self._storage = storage
        self.slot = int(slot)
        self.offset = 0

    @property
    def keys(self):
        return self._storage.keys[self.slot : self.slot + 1]

    @keys.setter
    def keys(self, value):
        if value is None:
            self.offset = 0
            return
        if int(value.shape[0]) != 1:
            raise FixedKVCacheError(
                f"Single-row fixed cache cannot restore {int(value.shape[0])} rows"
            )
        length = int(value.shape[2])
        if length > self._storage.logical_capacity:
            raise FixedKVCacheCapacityError(
                f"Restored key cache has {length:,} tokens but the fixed slot holds "
                f"{self._storage.logical_capacity:,}."
            )
        self._storage.keys[self.slot : self.slot + 1, :, :length] = value[
            ..., :length, :
        ]

    @property
    def values(self):
        return self._storage.values[self.slot : self.slot + 1]

    @values.setter
    def values(self, value):
        if value is None:
            self.offset = 0
            return
        if int(value.shape[0]) != 1:
            raise FixedKVCacheError(
                f"Single-row fixed cache cannot restore {int(value.shape[0])} rows"
            )
        length = int(value.shape[2])
        if length > self._storage.logical_capacity:
            raise FixedKVCacheCapacityError(
                f"Restored value cache has {length:,} tokens but the fixed slot holds "
                f"{self._storage.logical_capacity:,}."
            )
        self._storage.values[self.slot : self.slot + 1, :, :length] = value[
            ..., :length, :
        ]

    def update_and_fetch(self, keys, values):
        end = self.offset + int(keys.shape[2])
        if end > self._storage.logical_capacity:
            raise FixedKVCacheCapacityError(
                f"KV cache slot capacity is {self._storage.logical_capacity:,} tokens; "
                f"write would end at {end:,}. Reduce prompt or output tokens."
            )
        self._storage.keys[self.slot : self.slot + 1, :, self.offset : end] = keys
        self._storage.values[self.slot : self.slot + 1, :, self.offset : end] = values
        self.offset = end
        return self.keys[..., :end, :], self.values[..., :end, :]

    def compact_to(self, base: int = 0) -> None:
        destination = int(base)
        if self.slot == destination:
            return
        if self.slot < destination:
            raise FixedKVCacheError("Fixed KV rows cannot compact upward")
        self._storage.move_row(self.slot, destination, length=self.offset)
        self.slot = destination

    @property
    def state(self):
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    @state.setter
    def state(self, v):
        keys, values = v
        if keys is None:
            self.offset = 0
            return
        length = int(keys.shape[2])
        if length > self._storage.logical_capacity:
            raise FixedKVCacheCapacityError(
                f"Restored cache has {length:,} tokens but the fixed slot holds "
                f"{self._storage.logical_capacity:,}."
            )
        self._storage.keys[self.slot : self.slot + 1, :, :length] = keys
        self._storage.values[self.slot : self.slot + 1, :, :length] = values
        self.offset = length

    def empty(self):
        return self.offset == 0

    @property
    def nbytes(self):
        return self._storage.per_slot_nbytes

    @classmethod
    def merge(cls, caches):
        if not caches:
            raise ValueError("Cannot merge an empty fixed cache list")
        storage = caches[0]._storage
        if any(cache._storage is not storage for cache in caches):
            raise FixedKVCacheError("Cannot merge caches from different fixed pools")
        return FixedBatchKVCache.from_caches(caches)


class FixedBatchKVCache(BatchKVCache):
    """Dense batch view backed by fixed rows instead of merged allocations."""

    def __init__(
        self,
        storage: _KVStorage,
        slots: list[int],
        *,
        offsets: list[int],
        left_padding: list[int],
        index: int,
    ):
        self._storage = storage
        self.slots = list(slots)
        self.offset = mx.array(offsets)
        self.left_padding = mx.array(left_padding)
        self._idx = int(index)
        self._right_padding = None
        self._require_compact_slots()

    def _require_compact_slots(self) -> None:
        start = self.slots[0] if self.slots else 0
        expected = list(range(start, start + len(self.slots)))
        if self.slots != expected:
            raise FixedKVCacheError(
                f"Fixed cache rows must stay compact; got {self.slots}, expected {expected}"
            )

    def compact_to(self, base: int = 0) -> None:
        target = list(range(int(base), int(base) + len(self.slots)))
        if self.slots == target:
            return
        if any(source < destination for source, destination in zip(self.slots, target)):
            raise FixedKVCacheError("Fixed KV rows cannot compact upward")
        for source, destination in zip(self.slots, target):
            self._storage.move_row(source, destination, length=self._idx)
        self.slots = target

    @property
    def keys(self):
        start = self.slots[0] if self.slots else 0
        return self._storage.keys[start : start + len(self.slots)]

    @keys.setter
    def keys(self, value):
        if value is None:
            return
        rows = int(value.shape[0])
        if rows != len(self.slots):
            raise FixedKVCacheError(
                f"Restored key batch has {rows} rows but the fixed batch owns "
                f"{len(self.slots)} slots"
            )
        length = int(value.shape[2])
        if length > self._storage.logical_capacity:
            raise FixedKVCacheCapacityError(
                f"Restored key batch has {length:,} tokens but each fixed slot "
                f"holds {self._storage.logical_capacity:,}."
            )
        start = self.slots[0] if self.slots else 0
        self._storage.keys[start : start + rows, :, :length] = value[:rows, :, :length]

    @property
    def values(self):
        start = self.slots[0] if self.slots else 0
        return self._storage.values[start : start + len(self.slots)]

    @values.setter
    def values(self, value):
        if value is None:
            return
        rows = int(value.shape[0])
        if rows != len(self.slots):
            raise FixedKVCacheError(
                f"Restored value batch has {rows} rows but the fixed batch owns "
                f"{len(self.slots)} slots"
            )
        length = int(value.shape[2])
        if length > self._storage.logical_capacity:
            raise FixedKVCacheCapacityError(
                f"Restored value batch has {length:,} tokens but each fixed slot "
                f"holds {self._storage.logical_capacity:,}."
            )
        start = self.slots[0] if self.slots else 0
        self._storage.values[start : start + rows, :, :length] = value[
            :rows, :, :length
        ]

    @classmethod
    def from_caches(cls, caches: list[FixedKVCache]):
        if not caches:
            raise FixedKVCacheError("Cannot merge an empty fixed cache list")
        storage = caches[0]._storage
        if any(cache._storage is not storage for cache in caches):
            raise FixedKVCacheError("Cannot merge caches from different fixed pools")
        slots = [int(cache.slot) for cache in caches]
        if any(slot < 0 or slot >= storage.slots for slot in slots):
            raise FixedKVCacheError(
                f"Invalid fixed cache slots {slots} for a {storage.slots}-slot pool"
            )
        if len(set(slots)) != len(slots):
            raise FixedKVCacheError(
                f"Cannot merge duplicate fixed cache slots {slots}"
            )
        start = slots[0]
        expected = list(range(start, start + len(slots)))
        if slots != expected:
            raise FixedKVCacheError(
                f"Fixed cache rows must merge compactly; got {slots}, "
                f"expected {expected}"
            )
        lengths = [cache.offset for cache in caches]
        max_length = max(lengths, default=0)
        padding = [max_length - length for length in lengths]
        for cache, pad in zip(caches, padding):
            if pad:
                storage.shift_row_right(
                    cache.slot,
                    amount=pad,
                    length=cache.offset,
                )
        return cls(
            storage,
            slots,
            offsets=lengths,
            left_padding=padding,
            index=max_length,
        )

    @classmethod
    def merge(cls, caches):
        if not caches:
            raise FixedKVCacheError("Cannot merge an empty fixed batch list")
        if len(caches) == 1 and isinstance(caches[0], cls):
            return caches[0]
        if any(isinstance(cache, cls) for cache in caches):
            raise FixedKVCacheError(
                "Fixed KV batches can only be remerged one batch at a time"
            )
        return cls.from_caches(caches)

    def update_and_fetch(self, keys, values):
        end = self._idx + int(keys.shape[2])
        if end > self._storage.logical_capacity:
            raise FixedKVCacheCapacityError(
                f"KV cache slot capacity is {self._storage.logical_capacity:,} tokens; "
                f"batch write would end at {end:,}."
            )
        start = self.slots[0] if self.slots else 0
        stop = start + len(self.slots)
        self._storage.keys[start:stop, :, self._idx : end] = keys
        self._storage.values[start:stop, :, self._idx : end] = values
        self.offset += keys.shape[2]
        self._idx = end
        return self.keys[..., :end, :], self.values[..., :end, :]

    def _shift_right(self, row: int, amount: int, length: int) -> None:
        self._storage.shift_row_right(row, amount=amount, length=length)

    def extend(self, other):
        if (
            not isinstance(other, FixedBatchKVCache)
            or other._storage is not self._storage
        ):
            raise FixedKVCacheError(
                "Fixed KV batches can only extend from the same pool"
            )
        if len(other.slots) != 1:
            raise FixedKVCacheError("Fixed KV admission adds one cache slot at a time")
        old_rows = len(self.slots)
        base = 0
        expected_slot = base + old_rows
        source_slot = other.slots[0]
        if source_slot < expected_slot:
            raise FixedKVCacheError(
                f"Cannot admit fixed row {source_slot} behind active row "
                f"{expected_slot}; pool row ownership is inconsistent"
            )
        if source_slot != expected_slot:
            self._storage.move_row(
                source_slot,
                expected_slot,
                length=other._idx,
            )
            other.slots = [expected_slot]
        new_index = max(self._idx, other._idx)
        if new_index > self._storage.capacity:
            raise FixedKVCacheCapacityError("Merged fixed KV batch exceeds capacity")
        if new_index > self._idx:
            delta = new_index - self._idx
            for row in self.slots:
                self._shift_right(row, delta, self._idx)
            self.left_padding += delta
        if new_index > other._idx:
            self._shift_right(
                expected_slot,
                new_index - other._idx,
                other._idx,
            )
            other.left_padding += new_index - other._idx
        self.slots.append(expected_slot)
        self.offset = mx.concatenate([self.offset, other.offset])
        self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
        self._idx = new_index

    def filter(self, batch_indices):
        indices = _filter_indices(batch_indices, len(self.slots))
        base = 0
        old_slots = list(self.slots)
        for dst, src in enumerate(indices):
            destination = base + dst
            source = old_slots[src]
            if destination == source:
                continue
            self._storage.move_row(source, destination, length=self._idx)
        self.offset = self.offset[indices]
        self.left_padding = self.left_padding[indices]
        self.slots = list(range(base, base + len(indices)))
        if indices:
            min_left_pad = int(self.left_padding.min().item())
            if min_left_pad > 0:
                length = self._idx - min_left_pad
                for row in self.slots:
                    self._storage.write_row_from(
                        row,
                        self._storage.keys[
                            row : row + 1, :, min_left_pad : self._idx
                        ],
                        self._storage.values[
                            row : row + 1, :, min_left_pad : self._idx
                        ],
                        length=length,
                    )
                self._idx = length
                self.left_padding -= min_left_pad

    def finalize(self):
        right_padding: Any = self._right_padding
        if right_padding is None:
            return
        padding = [int(value) for value in right_padding.tolist()]
        for slot, amount in zip(self.slots, padding):
            self._storage.roll_row_right(
                slot,
                amount=amount,
                length=self._idx,
            )
        padding_array = right_padding
        self.offset -= padding_array
        self.left_padding += padding_array
        self._right_padding = None

    def extract(self, idx):
        padding = int(self.left_padding[idx].item())
        physical_slot = self.slots[int(idx)]
        cache = FixedKVCache(self._storage, physical_slot)
        if padding:
            length = self._idx - padding
            self._storage.write_row_from(
                physical_slot,
                self._storage.keys[
                    physical_slot : physical_slot + 1, :, padding : self._idx
                ],
                self._storage.values[
                    physical_slot : physical_slot + 1, :, padding : self._idx
                ],
                length=length,
            )
            cache.offset = length
        else:
            cache.offset = self._idx
        return cache

    @property
    def state(self):
        return (
            self.keys[..., : self._idx, :],
            self.values[..., : self._idx, :],
            self.offset,
            self.left_padding,
        )

    @state.setter
    def state(self, v):
        keys, values, offsets, left_padding = v
        length = int(keys.shape[2])
        if length > self._storage.logical_capacity:
            raise FixedKVCacheCapacityError(
                f"Restored batch cache has {length:,} tokens but each fixed slot "
                f"holds {self._storage.logical_capacity:,}."
            )
        self.keys = keys
        self.values = values
        self.offset = offsets
        self.left_padding = left_padding
        self._idx = length

    def empty(self):
        return self._idx == 0

    @property
    def nbytes(self):
        return self._storage.per_slot_nbytes * len(self.slots)


class FixedMiniMaxM3KVCache:
    """MiniMax M3 K/V and sparse-index views backed by two fixed rows."""

    step = KVCache.step

    def __init__(self, kv_cache: FixedKVCache, index_cache: FixedKVCache):
        self.kv_cache = kv_cache
        self._index_cache = index_cache

    @property
    def offset(self):
        return self.kv_cache.offset

    @offset.setter
    def offset(self, value):
        self.kv_cache.offset = int(value)
        self._index_cache.offset = int(value)

    @property
    def index_offset(self):
        return self._index_cache.offset

    @index_offset.setter
    def index_offset(self, value):
        self._index_cache.offset = int(value)

    @property
    def index_keys(self):
        return None if self.index_offset == 0 else self._index_cache.keys

    @index_keys.setter
    def index_keys(self, value):
        if value is None:
            self._index_cache.offset = 0
            return
        dummy = mx.zeros((*value.shape[:-1], 0), dtype=value.dtype)
        self._index_cache.state = value, dummy

    def update_and_fetch(self, keys, values):
        return self.kv_cache.update_and_fetch(keys, values)

    def update_index_and_fetch(self, keys):
        dummy = mx.zeros((*keys.shape[:-1], 0), dtype=keys.dtype)
        indexed, _ = self._index_cache.update_and_fetch(keys, dummy)
        return indexed

    def make_mask(self, *args, **kwargs):
        return self.kv_cache.make_mask(*args, **kwargs)

    def size(self):
        return self.kv_cache.size()

    def empty(self):
        return self.kv_cache.empty()

    def is_trimmable(self):
        return True

    def trim(self, n):
        trimmed = self.kv_cache.trim(n)
        self._index_cache.trim(trimmed)
        return trimmed

    def compact_to(self, base: int = 0) -> None:
        self.kv_cache.compact_to(base)
        self._index_cache.compact_to(base)

    @classmethod
    def merge(cls, caches, prefix_lens=None):
        del prefix_lens
        if not caches:
            raise FixedKVCacheError("Cannot merge an empty MiniMax M3 cache list")
        return FixedMiniMaxM3BatchKVCache(
            FixedBatchKVCache.from_caches([cache.kv_cache for cache in caches]),
            FixedBatchKVCache.from_caches(
                [cache._index_cache for cache in caches]
            ),
        )

    def to_batch(self, left_padding):
        padding = list(left_padding)
        if padding != [0]:
            raise FixedKVCacheError(
                "A fixed MiniMax M3 cache can only become an unpadded singleton batch"
            )
        return self.merge([self])

    @property
    def state(self):
        kv_state = None if self.kv_cache.empty() else self.kv_cache.state
        index_keys = self.index_keys
        index_state = (
            None
            if self.index_offset == 0 or index_keys is None
            else index_keys[..., : self.index_offset, :]
        )
        return kv_state, index_state

    @state.setter
    def state(self, value):
        kv_state, index_state = value
        if kv_state is None or kv_state[0] is None:
            self.kv_cache.offset = 0
        else:
            self.kv_cache.state = kv_state[:2]
        self.index_keys = index_state

    @property
    def meta_state(self):
        return str(self.index_offset)

    @meta_state.setter
    def meta_state(self, value):
        self.index_offset = int(value) if value else 0

    @property
    def nbytes(self):
        return self.kv_cache.nbytes + self._index_cache.nbytes


class FixedMiniMaxM3BatchKVCache:
    """Batched MiniMax M3 composite cache that never leaves fixed rows."""

    step = BatchKVCache.step

    def __init__(
        self,
        kv_cache: FixedBatchKVCache,
        index_cache: FixedBatchKVCache,
    ):
        self.kv_cache = kv_cache
        self._index_cache = index_cache

    @property
    def offset(self):
        return self.kv_cache.offset

    @property
    def left_padding(self):
        return self.kv_cache.left_padding

    @property
    def _idx(self):
        return self.kv_cache._idx

    @property
    def index_offset(self):
        return self._index_cache._idx

    @index_offset.setter
    def index_offset(self, value):
        self._index_cache._idx = int(value)

    @property
    def index_keys(self):
        return None if self.index_offset == 0 else self._index_cache.keys

    @index_keys.setter
    def index_keys(self, value):
        if value is None:
            self._index_cache._idx = 0
            return
        dummy = mx.zeros((*value.shape[:-1], 0), dtype=value.dtype)
        self._index_cache.state = (
            value,
            dummy,
            self.kv_cache.offset,
            self.kv_cache.left_padding,
        )

    def update_and_fetch(self, keys, values):
        return self.kv_cache.update_and_fetch(keys, values)

    def update_index_and_fetch(self, keys):
        dummy = mx.zeros((*keys.shape[:-1], 0), dtype=keys.dtype)
        indexed, _ = self._index_cache.update_and_fetch(keys, dummy)
        return indexed

    def prepare(self, **kwargs):
        self.kv_cache.prepare(**kwargs)
        self._index_cache.prepare(**kwargs)

    def finalize(self):
        self.kv_cache.finalize()
        self._index_cache.finalize()

    def make_mask(self, *args, **kwargs):
        return self.kv_cache.make_mask(*args, **kwargs)

    def filter(self, batch_indices):
        self.kv_cache.filter(batch_indices)
        self._index_cache.filter(batch_indices)

    def extend(self, other):
        if not isinstance(other, FixedMiniMaxM3BatchKVCache):
            raise FixedKVCacheError("MiniMax M3 fixed batches do not match")
        self.kv_cache.extend(other.kv_cache)
        self._index_cache.extend(other._index_cache)

    @classmethod
    def merge(cls, caches, prefix_lens=None):
        del prefix_lens
        if len(caches) == 1 and isinstance(caches[0], cls):
            return caches[0]
        raise FixedKVCacheError(
            "MiniMax M3 fixed batches can only be remerged one batch at a time"
        )

    def compact_to(self, base: int = 0) -> None:
        self.kv_cache.compact_to(base)
        self._index_cache.compact_to(base)

    def extract(self, idx):
        return FixedMiniMaxM3KVCache(
            self.kv_cache.extract(idx),
            self._index_cache.extract(idx),
        )

    def size(self):
        return self.kv_cache.size()

    def empty(self):
        return self.kv_cache.empty()

    def is_trimmable(self):
        return True

    def trim(self, n):
        trimmed = self.kv_cache.trim(n)
        self._index_cache.trim(trimmed)
        return trimmed

    @property
    def state(self):
        kv_state = self.kv_cache.state
        index_keys = self.index_keys
        index_state = (
            None
            if self.index_offset == 0 or index_keys is None
            else index_keys[..., : self.index_offset, :]
        )
        return kv_state, index_state

    @state.setter
    def state(self, value):
        kv_state, index_state = value
        self.kv_cache.state = kv_state
        self.index_keys = index_state

    @property
    def meta_state(self):
        return str(self.index_offset)

    @meta_state.setter
    def meta_state(self, value):
        self.index_offset = int(value) if value else 0

    @property
    def nbytes(self):
        return self.kv_cache.nbytes + self._index_cache.nbytes


# Prefix-cache type routing uses the concrete class name. Preserve the cache
# wire identity while keeping this adapter independent of mlx-vlm import order.
FixedMiniMaxM3KVCache.__name__ = "MiniMaxM3KVCache"
FixedMiniMaxM3BatchKVCache.__name__ = "MiniMaxM3BatchKVCache"


class FixedChunkedKVCache(ChunkedKVCache):
    """Serialized Llama 4 cache with a fixed resident chunk buffer."""

    def __init__(
        self,
        storage: _KVStorage,
        slot: int,
        chunk_size: int,
    ) -> None:
        self._storage = storage
        self.slot = int(slot)
        self.chunk_size = int(chunk_size)
        self.offset = 0
        self.start_position = 0

    @property
    def keys(self):
        return self._storage.keys[self.slot : self.slot + 1]

    @keys.setter
    def keys(self, value):
        if value is None:
            self.offset = 0
            self.start_position = 0
            return
        length = int(value.shape[2])
        if length > self._storage.capacity:
            raise FixedKVCacheCapacityError(
                f"Chunked cache resident capacity is {self._storage.capacity:,} "
                f"tokens; restore requires {length:,}."
            )
        self._storage.keys[self.slot : self.slot + 1, :, :length] = value

    @property
    def values(self):
        return self._storage.values[self.slot : self.slot + 1]

    @values.setter
    def values(self, value):
        if value is None:
            self.offset = 0
            self.start_position = 0
            return
        length = int(value.shape[2])
        if length > self._storage.capacity:
            raise FixedKVCacheCapacityError(
                f"Chunked cache resident capacity is {self._storage.capacity:,} "
                f"tokens; restore requires {length:,}."
            )
        self._storage.values[self.slot : self.slot + 1, :, :length] = value

    def maybe_trim_front(self):
        resident = self.offset - self.start_position
        if resident < self.chunk_size:
            return
        keep = min(resident, self.chunk_size)
        drop = resident - keep
        if drop:
            self._storage.write_row_from(
                self.slot,
                self.keys[..., drop:resident, :],
                self.values[..., drop:resident, :],
                length=keep,
            )
            self.start_position += drop

    def update_and_fetch(self, keys, values):
        incoming = int(keys.shape[2])
        absolute_end = self.offset + incoming
        if absolute_end > self._storage.logical_capacity:
            raise FixedKVCacheCapacityError(
                f"KV cache slot capacity is {self._storage.logical_capacity:,} tokens; "
                f"write would end at {absolute_end:,}. Reduce prompt or output tokens."
            )
        resident = self.offset - self.start_position
        resident_end = resident + incoming
        if resident_end > self._storage.capacity:
            raise FixedKVCacheCapacityError(
                f"Chunked cache resident capacity is {self._storage.capacity:,} "
                f"tokens; one forward requires {resident_end:,}. Reduce the "
                "prefill step size."
            )
        self._storage.keys[
            self.slot : self.slot + 1, :, resident:resident_end
        ] = keys
        self._storage.values[
            self.slot : self.slot + 1, :, resident:resident_end
        ] = values
        self.offset = absolute_end
        return (
            self.keys[..., :resident_end, :],
            self.values[..., :resident_end, :],
        )

    def compact_to(self, base: int = 0) -> None:
        destination = int(base)
        if self.slot == destination:
            return
        if self.slot < destination:
            raise FixedKVCacheError("Fixed chunked rows cannot compact upward")
        self._storage.move_row(
            self.slot,
            destination,
            length=self.offset - self.start_position,
        )
        self.slot = destination

    @property
    def state(self):
        resident = self.offset - self.start_position
        return self.keys[..., :resident, :], self.values[..., :resident, :]

    @state.setter
    def state(self, v):
        keys, values = v
        if keys is None:
            self.offset = 0
            self.start_position = 0
            return
        length = int(keys.shape[2])
        self.keys = keys
        self.values = values
        self.offset = length
        self.start_position = 0

    @property
    def meta_state(self):
        return tuple(map(str, (self.chunk_size, self.start_position)))

    @meta_state.setter
    def meta_state(self, v):
        chunk_size, start_position = map(int, v)
        resident = self.offset - self.start_position
        self.chunk_size = chunk_size
        self.start_position = start_position
        self.offset = start_position + resident

    def trim(self, n):
        trimmed = min(self.offset - self.start_position, int(n))
        self.offset -= trimmed
        return trimmed

    def empty(self):
        return self.offset == 0

    def size(self):  # type: ignore[override]
        return max(0, self.offset - self.start_position)

    def filter(self, batch_indices):
        indices = _filter_indices(batch_indices, 1)
        if len(indices) == 0:
            self.offset = 0
            self.start_position = 0
        elif indices != [0]:
            raise FixedKVCacheError(
                "Llama 4 chunked cache is serialized and only has row zero"
            )

    def extract(self, idx):
        if int(idx) != 0:
            raise FixedKVCacheError("Llama 4 chunked cache only has row zero")
        return self

    def extend(self, other):
        if other is None or other.empty():
            return
        if not self.empty():
            raise FixedKVCacheError(
                "Llama 4 chunked cache cannot batch non-empty sessions"
            )
        self.state = other.state
        self.meta_state = other.meta_state

    @classmethod
    def merge(cls, caches):
        if len(caches) != 1:
            raise FixedKVCacheError(
                "Llama 4 chunked cache supports one concurrent session"
            )
        return caches[0]

    @property
    def nbytes(self):
        return self._storage.per_slot_nbytes


class FixedRingSlidingKVCache(FixedKVCache):
    """Unlimited OCR prompt-plus-ring semantics inside one fixed context row."""

    def __init__(self, storage: _KVStorage, slot: int, window_size: int):
        super().__init__(storage, slot)
        self.window_size = int(window_size)
        self.prefill_length: int | None = None
        self._ring_pos = 0

    def update_and_fetch(self, keys, values):
        incoming = int(keys.shape[2])
        if self.offset + incoming > self._storage.logical_capacity:
            raise FixedKVCacheCapacityError(
                f"KV cache slot capacity is {self._storage.logical_capacity:,} tokens; "
                f"write would end at {self.offset + incoming:,}. Reduce prompt or "
                "output tokens."
            )
        if self.prefill_length is None:
            if incoming > 1:
                return super().update_and_fetch(keys, values)
            self.prefill_length = self.offset

        ring_end = self.prefill_length + self.window_size
        if self.offset < ring_end:
            fetched = super().update_and_fetch(keys, values)
            if self.offset >= ring_end:
                self._ring_pos = 0
            return fetched

        for index in range(incoming):
            destination = self.prefill_length + self._ring_pos
            self._storage.keys[
                self.slot : self.slot + 1, :, destination : destination + 1
            ] = keys[..., index : index + 1, :]
            self._storage.values[
                self.slot : self.slot + 1, :, destination : destination + 1
            ] = values[..., index : index + 1, :]
            self._ring_pos = (self._ring_pos + 1) % self.window_size
        self.offset += incoming
        return self.keys[..., :ring_end, :], self.values[..., :ring_end, :]

    def make_mask(self, n: int, return_array: bool = False, **kwargs):
        kwargs.pop("window_size", None)
        if (
            self.prefill_length is not None
            and self.offset >= self.prefill_length + self.window_size
            and n == 1
            and not return_array
        ):
            return None
        return super().make_mask(
            n,
            return_array=return_array,
            window_size=None,
            **kwargs,
        )

    def filter(self, batch_indices):
        indices = _filter_indices(batch_indices, 1)
        if len(indices) == 0:
            self.offset = 0
            self.prefill_length = None
            self._ring_pos = 0
        elif indices != [0]:
            raise FixedKVCacheError(
                "Unlimited OCR's ring cache is serialized and only has row zero"
            )

    def extract(self, idx):
        if int(idx) != 0:
            raise FixedKVCacheError(
                "Unlimited OCR's ring cache only has row zero"
            )
        return self

    def extend(self, other):
        if other is None or other.empty():
            return
        if not self.empty():
            raise FixedKVCacheError(
                "Unlimited OCR's ring cache cannot batch non-empty sessions"
            )
        self.state = other.state
        self.meta_state = other.meta_state

    @classmethod
    def merge(cls, caches):
        if len(caches) != 1:
            raise FixedKVCacheError(
                "Unlimited OCR's ring cache supports one concurrent session"
            )
        return caches[0]

    @property
    def state(self):
        end = (
            self.offset
            if self.prefill_length is None
            else min(self.offset, self.prefill_length + self.window_size)
        )
        return self.keys[..., :end, :], self.values[..., :end, :]

    @state.setter
    def state(self, v):
        keys, values = v
        if keys is None:
            self.offset = 0
        else:
            length = int(keys.shape[2])
            if length > self._storage.logical_capacity:
                raise FixedKVCacheCapacityError(
                    f"Restored cache has {length:,} tokens but the fixed slot holds "
                    f"{self._storage.logical_capacity:,}."
                )
            self._storage.keys[self.slot : self.slot + 1, :, :length] = keys
            self._storage.values[self.slot : self.slot + 1, :, :length] = values
            self.offset = length
        self.prefill_length = None
        self._ring_pos = 0

    @property
    def meta_state(self):  # type: ignore[override]
        return tuple(
            map(
                str,
                (
                    self.window_size,
                    -1 if self.prefill_length is None else self.prefill_length,
                    self.offset,
                    self._ring_pos,
                ),
            )
        )

    @meta_state.setter
    def meta_state(self, v):
        window, prefill, offset, ring_pos = map(int, v)
        self.window_size = window
        self.prefill_length = None if prefill < 0 else prefill
        self.offset = offset
        self._ring_pos = ring_pos


class FixedRotatingKVCache(RotatingKVCache):
    """Single-sequence sliding cache with a materialized resident window."""

    def __init__(self, storage: _KVStorage, slot: int, max_size: int, keep: int = 0):
        self._storage = storage
        self.slot = int(slot)
        self.keep = int(keep)
        self.max_size = int(max_size)
        self.offset = 0
        self._idx = 0

    @property
    def keys(self):
        return self._storage.keys[self.slot : self.slot + 1, :, : self.max_size]

    @keys.setter
    def keys(self, value):
        if value is None:
            self.offset = self._idx = 0
            return
        length = min(int(value.shape[2]), self.max_size)
        self._storage.keys[self.slot : self.slot + 1, :, :length] = value[
            ..., -length:, :
        ]

    @property
    def values(self):
        return self._storage.values[self.slot : self.slot + 1, :, : self.max_size]

    @values.setter
    def values(self, value):
        if value is None:
            self.offset = self._idx = 0
            return
        length = min(int(value.shape[2]), self.max_size)
        self._storage.values[self.slot : self.slot + 1, :, :length] = value[
            ..., -length:, :
        ]

    def _update_concat(self, keys, values):
        if self.offset == 0:
            fetched_k, fetched_v = keys, values
        else:
            old_k = self._temporal_order(self.keys)
            old_v = self._temporal_order(self.values)
            trim_size = int(old_k.shape[2]) - self.max_size + 1
            fetched_k = self._trim(trim_size, old_k, keys)
            fetched_v = self._trim(trim_size, old_v, values)
        resident = min(self.max_size, int(fetched_k.shape[2]))
        self._storage.keys[self.slot : self.slot + 1, :, :resident] = fetched_k[
            ..., -resident:, :
        ]
        self._storage.values[self.slot : self.slot + 1, :, :resident] = fetched_v[
            ..., -resident:, :
        ]
        self.offset += int(keys.shape[2])
        self._idx = resident
        return fetched_k, fetched_v

    def _update_in_place(self, keys, values):
        if self._idx >= self.max_size:
            self._idx = self.keep
        end = self._idx + int(keys.shape[2])
        if end > self.max_size:
            return self._update_concat(keys, values)
        self._storage.keys[self.slot : self.slot + 1, :, self._idx : end] = keys
        self._storage.values[self.slot : self.slot + 1, :, self._idx : end] = values
        self.offset += int(keys.shape[2])
        self._idx = end
        if self.offset < self.max_size:
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
        return self.keys, self.values

    def update_and_fetch(self, keys, values):
        if int(keys.shape[2]) == 1:
            return self._update_in_place(keys, values)
        return self._update_concat(keys, values)

    def compact_to(self, base: int = 0) -> None:
        destination = int(base)
        if self.slot == destination:
            return
        if self.slot < destination:
            raise FixedKVCacheError("Fixed rotating rows cannot compact upward")
        if self.offset:
            self._storage.move_row(
                self.slot,
                destination,
                length=self.max_size,
            )
        self.slot = destination

    def empty(self):
        return self.offset == 0

    @property
    def nbytes(self):
        return self._storage.per_slot_nbytes

    @classmethod
    def merge(cls, caches):
        return FixedBatchRotatingKVCache.from_caches(caches)


class FixedBatchRotatingKVCache(BatchRotatingKVCache):
    """Batch rotating cache backed by the fixed resident-window rows."""

    def __init__(self, storage: _KVStorage, slots: list[int], max_size: int):
        super().__init__(max_size=max_size, left_padding=[0] * len(slots))
        self._storage = storage
        self.slots = list(slots)

    @classmethod
    def from_cache(cls, cache: FixedRotatingKVCache):
        return cls.from_caches([cache])

    @classmethod
    def from_caches(cls, caches: list[FixedRotatingKVCache]):
        if not caches:
            raise FixedKVCacheError("Cannot merge an empty rotating cache list")
        storage = caches[0]._storage
        if any(cache._storage is not storage for cache in caches):
            raise FixedKVCacheError("Cannot merge rotating caches from different pools")
        if any(cache.max_size != caches[0].max_size for cache in caches):
            raise FixedKVCacheError("Fixed rotating cache windows do not match")

        lengths = [min(cache.offset, cache.max_size) for cache in caches]
        max_length = max(lengths, default=0)
        padding = [max_length - length for length in lengths]
        for cache, length, pad in zip(caches, lengths, padding):
            if length:
                storage.write_row_from(
                    cache.slot,
                    cache._temporal_order(cache.keys)[..., -length:, :],
                    cache._temporal_order(cache.values)[..., -length:, :],
                    length=length,
                    destination=pad,
                )

        obj = cls(storage, [cache.slot for cache in caches], caches[0].max_size)
        obj.offset = mx.array([cache.offset for cache in caches])
        obj.left_padding = mx.array(padding)
        obj._offset = max_length
        obj._idx = max_length
        obj.rotated = False
        return obj

    @classmethod
    def merge(cls, caches):
        if not caches:
            raise FixedKVCacheError("Cannot merge an empty rotating batch list")
        if len(caches) == 1 and isinstance(caches[0], cls):
            return caches[0]
        if any(isinstance(cache, cls) for cache in caches):
            raise FixedKVCacheError(
                "Fixed rotating batches can only be remerged one batch at a time"
            )
        return cls.from_caches(caches)

    @property
    def keys(self):
        start = self.slots[0] if self.slots else 0
        return self._storage.keys[start : start + len(self.slots), :, : self.max_size]

    @keys.setter
    def keys(self, value):
        if value is None:
            return
        rows = min(int(value.shape[0]), len(self.slots))
        length = min(int(value.shape[2]), self.max_size)
        start = self.slots[0] if self.slots else 0
        self._storage.keys[start : start + rows, :, :length] = value[:rows, :, -length:]

    @property
    def values(self):
        start = self.slots[0] if self.slots else 0
        return self._storage.values[start : start + len(self.slots), :, : self.max_size]

    def compact_to(self, base: int = 0) -> None:
        target = list(range(int(base), int(base) + len(self.slots)))
        if self.slots == target:
            return
        if any(source < destination for source, destination in zip(self.slots, target)):
            raise FixedKVCacheError("Fixed rotating rows cannot compact upward")
        for source, destination in zip(self.slots, target):
            self._storage.move_row(source, destination, length=self.max_size)
        self.slots = target

    @values.setter
    def values(self, value):
        if value is None:
            return
        rows = min(int(value.shape[0]), len(self.slots))
        length = min(int(value.shape[2]), self.max_size)
        start = self.slots[0] if self.slots else 0
        self._storage.values[start : start + rows, :, :length] = value[
            :rows, :, -length:
        ]

    def extend(self, other):
        if (
            not isinstance(other, FixedBatchRotatingKVCache)
            or other._storage is not self._storage
        ):
            raise FixedKVCacheError("Fixed rotating batches must share one pool")
        if len(other.slots) != 1:
            raise FixedKVCacheError(
                "Fixed rotating admission adds one cache slot at a time"
            )
        base = 0
        expected_slot = base + len(self.slots)
        source_slot = other.slots[0]
        if source_slot < expected_slot:
            raise FixedKVCacheError(
                "Fixed rotating pool row ownership is inconsistent"
            )
        # Canonicalize the current ring before aligning a newly admitted row.
        if self.rotated:
            for row in self.slots:
                self._storage.roll_row_right(
                    row,
                    amount=self.max_size - self._idx,
                    length=self.max_size,
                )
            self._idx = self.max_size
            self.rotated = False
        if other.rotated:
            self._storage.roll_row_right(
                source_slot,
                amount=self.max_size - other._idx,
                length=self.max_size,
            )
            other._idx = self.max_size
            other.rotated = False
        if source_slot != expected_slot:
            self._storage.move_row(
                source_slot,
                expected_slot,
                length=self.max_size,
            )
            other.slots = [expected_slot]
        target = max(self._idx, other._idx)
        if self._idx < target:
            delta = target - self._idx
            for row in self.slots:
                self._storage.shift_row_right(
                    row,
                    amount=delta,
                    length=self._idx,
                )
            self.left_padding += delta
        if other._idx < target:
            delta = target - other._idx
            self._storage.shift_row_right(
                expected_slot,
                amount=delta,
                length=other._idx,
            )
            other.left_padding += delta
        self.slots.append(expected_slot)
        self.offset = mx.concatenate([self.offset, other.offset])
        self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
        self._idx = target
        self._offset = target

    def finalize(self):
        if self._lengths is None:
            return
        roll = mx.maximum(0, self.offset - self._lengths)
        roll_values = [int(value) for value in roll.tolist()]
        resident = min(int(self._offset), int(self.max_size))
        for slot, amount in zip(self.slots, roll_values):
            self._storage.roll_row_right(
                slot,
                amount=amount,
                length=resident,
            )
        self.left_padding += roll
        self.offset -= roll
        self._lengths = None

    def filter(self, batch_indices):
        indices = _filter_indices(batch_indices, len(self.slots))
        base = 0
        old_slots = list(self.slots)
        for dst, src in enumerate(indices):
            destination = base + dst
            source = old_slots[src]
            if destination != source:
                self._storage.move_row(
                    source,
                    destination,
                    length=self.max_size,
                )
        self.offset = self.offset[indices]
        self.left_padding = self.left_padding[indices]
        self.slots = list(range(base, base + len(indices)))

    def extract(self, idx):
        cache = FixedRotatingKVCache(
            self._storage,
            self.slots[int(idx)],
            self.max_size,
        )
        cache.offset = int(self.offset[idx].item())
        cache._idx = self._idx
        return cache

    @property
    def nbytes(self):
        return self._storage.per_slot_nbytes * len(self.slots)


class FixedArraysCache(ArraysCache):
    """Fixed recurrent state rows used by hybrid GDN/Mamba-style layers."""

    def __init__(self, storage: _ArraysStorage, slots: list[int]):
        self._storage = storage
        self.slots = list(slots)
        self.left_padding = None
        self.lengths = None

    @property
    def cache(self):
        if not self.slots:
            return [None for _ in self._storage.arrays]
        if self.slots == list(range(len(self.slots))):
            return [
                None if array is None else array[: len(self.slots)]
                for array in self._storage.arrays
            ]
        return [
            None if array is None else array[self.slots]
            for array in self._storage.arrays
        ]

    @cache.setter
    def cache(self, values):
        for idx, value in enumerate(values):
            if value is not None:
                self[idx] = value

    def __setitem__(self, idx, value):
        if value is None:
            return
        storage = self._storage.arrays[idx]
        if storage is None:
            raise FixedKVCacheError("Recurrent cache probe did not expose this state")
        for row, slot in enumerate(self.slots):
            storage[slot : slot + 1] = value[row : row + 1]

    def __getitem__(self, idx):
        storage = self._storage.arrays[idx]
        if storage is None:
            return None
        if self.slots == list(range(len(self.slots))):
            return storage[: len(self.slots)]
        return storage[self.slots]

    @classmethod
    def merge(cls, caches):
        if not caches:
            raise FixedKVCacheError("Cannot merge empty recurrent cache list")
        if len(caches) == 1:
            return caches[0]
        storage = caches[0]._storage
        if any(cache._storage is not storage for cache in caches):
            raise FixedKVCacheError("Recurrent caches must share one fixed pool")
        slots = [slot for cache in caches for slot in cache.slots]
        return cls(storage, slots)

    def extend(self, other):
        if other._storage is not self._storage:
            raise FixedKVCacheError("Recurrent batches must share one fixed pool")
        if len(other.slots) != 1:
            raise FixedKVCacheError(
                "Fixed recurrent admission adds one cache slot at a time"
            )
        base = 0
        expected_slot = base + len(self.slots)
        source_slot = other.slots[0]
        if source_slot < expected_slot:
            raise FixedKVCacheError("Recurrent pool row ownership is inconsistent")
        if source_slot != expected_slot:
            self._storage.move_row(source_slot, expected_slot)
            other.slots = [expected_slot]
        self.slots.append(expected_slot)

    def compact_to(self, base: int = 0) -> None:
        target = list(range(int(base), int(base) + len(self.slots)))
        if self.slots == target:
            return
        if any(source < destination for source, destination in zip(self.slots, target)):
            raise FixedKVCacheError("Fixed recurrent rows cannot compact upward")
        for source, destination in zip(self.slots, target):
            self._storage.move_row(source, destination)
        self.slots = target

    def filter(self, batch_indices):
        indices = _filter_indices(batch_indices, len(self.slots))
        base = 0
        old_slots = list(self.slots)
        for dst, src in enumerate(indices):
            destination = base + dst
            source = old_slots[src]
            if destination == source:
                continue
            self._storage.move_row(source, destination)
        self.slots = list(range(base, base + len(indices)))
        if self.left_padding is not None:
            self.left_padding = self.left_padding[indices]
        if self.lengths is not None:
            self.lengths = self.lengths[indices]

    def extract(self, idx):
        return FixedArraysCache(self._storage, [self.slots[int(idx)]])

    def empty(self):
        return False

    @property
    def nbytes(self):
        return self._storage.per_slot_nbytes * len(self.slots)


class FixedNullCache:
    """Zero-byte cache placeholder for architecture-declared linear/no-op blocks."""

    def __init__(self, slots: list[int]):
        self.slots = list(slots)
        self.offset = 0
        self.left_padding = None
        self.lengths = None

    @property
    def state(self):
        return ()

    @state.setter
    def state(self, value):
        if value not in (None, (), []):
            raise FixedKVCacheError("A no-op cache cannot accept tensor state")

    @classmethod
    def merge(cls, caches):
        if not caches:
            raise FixedKVCacheError("Cannot merge an empty no-op cache list")
        if len(caches) == 1:
            return caches[0]
        return cls([slot for cache in caches for slot in cache.slots])

    def extend(self, other):
        self.slots.extend(other.slots)

    def filter(self, indices):
        values = indices.tolist() if isinstance(indices, mx.array) else list(indices)
        self.slots = [self.slots[int(index)] for index in values]

    def extract(self, idx):
        return FixedNullCache([self.slots[int(idx)]])

    def compact_to(self, base: int = 0) -> None:
        self.slots = list(range(int(base), int(base) + len(self.slots)))

    def prepare(self, *, lengths=None, right_padding=None):
        self.lengths = None if lengths is None else mx.array(lengths)

    def finalize(self):
        self.lengths = None

    def trim(self, n: int):
        return None

    def update_and_fetch(self, keys, values):
        raise FixedKVCacheError(
            "An architecture-declared no-op cache unexpectedly received K/V tensors"
        )

    def empty(self):
        return True

    @property
    def nbytes(self):
        return 0


class FixedPoolingCache(PoolingCache):
    """One DeepSeek pooled-cache row backed by launch-time storage."""

    def __init__(self, storage: _PoolingStorage, slot: int):
        self._storage = storage
        self.slot = int(slot)
        self.ratio = int(storage.ratio)
        self.remainder = 0
        self._pool_len = 0
        self._undo = None
        self._undo_chain = False
        self._mtp_cross_boundary_rollback = True
        self._prev_valid = False

    @property
    def _pool_buf(self):
        return self._storage.pooled[self.slot : self.slot + 1]

    @_pool_buf.setter
    def _pool_buf(self, value):
        if value is None:
            self._pool_len = 0
            return
        length = int(value.shape[1])
        if length > self._storage.logical_capacity:
            raise FixedKVCacheCapacityError(
                f"Pooled cache capacity is {self._storage.logical_capacity:,} rows; "
                f"restore requires {length:,}."
            )
        self._storage.pooled[self.slot : self.slot + 1, :length] = value[:, :length]

    @property
    def pooled(self):
        if self._pool_len == 0:
            return None
        return self._pool_buf[:, : self._pool_len]

    @pooled.setter
    def pooled(self, v):
        if v is None:
            self._pool_len = 0
            return
        self._pool_buf = v
        self._pool_len = int(v.shape[1])

    @property
    def buf_kv(self):
        return self._storage.buf_kv[self.slot : self.slot + 1]

    @buf_kv.setter
    def buf_kv(self, value):
        self._restore_buffer(self._storage.buf_kv, value, "KV")

    @property
    def buf_gate(self):
        return self._storage.buf_gate[self.slot : self.slot + 1]

    @buf_gate.setter
    def buf_gate(self, value):
        self._restore_buffer(self._storage.buf_gate, value, "gate")

    def _restore_buffer(self, storage, value, label: str) -> None:
        """Restore one row without rebinding or clearing its source view."""
        row = slice(self.slot, self.slot + 1)
        if value is None:
            storage[row] = 0
            mx.eval(storage)
            return
        rows = int(value.shape[0])
        length = int(value.shape[1])
        if rows != 1 or length > self.ratio:
            raise FixedKVCacheError(
                f"Pooled {label} restore shape {tuple(value.shape)} exceeds the "
                f"fixed one-row, {self.ratio}-token remainder buffer"
            )
        storage[row, :length] = value[:, :length]
        if length < self.ratio:
            storage[row, length:] = 0
        mx.eval(storage)

    def _stash_undo_buffers(self) -> tuple[mx.array | None, mx.array | None]:
        if self.remainder <= 0:
            return None, None
        undo_kv = self._storage.undo_buf_kv
        undo_gate = self._storage.undo_buf_gate
        if undo_kv is None or undo_gate is None:
            raise FixedKVCacheError(
                "Pooled rollback requires launch-time per-session undo storage"
            )
        row = slice(self.slot, self.slot + 1)
        undo_kv[row] = self._storage.buf_kv[row]
        undo_gate[row] = self._storage.buf_gate[row]
        mx.eval(undo_kv, undo_gate)
        return (
            undo_kv[row, : self.remainder],
            undo_gate[row, : self.remainder],
        )

    def accumulate_windows(self, kv, gate, offset):
        """PoolingCache accumulation with writes routed to committed storage."""
        batch, length, dim_kv = kv.shape
        _, _, dim_gate = gate.shape
        if int(batch) != 1:
            raise FixedKVCacheError("A single fixed pooled row requires batch size 1")

        if length <= 8:
            try:
                from omlx.patches.mlx_lm_mtp import cache_rollback

                decode_consistent = cache_rollback._is_undo_armed() and (
                    length == 1 or cache_rollback._is_decode_consistent_armed()
                )
            except Exception:
                decode_consistent = False
            if decode_consistent and self._undo_chain:
                undo: Any = self._undo
                self._undo = (
                    *undo[:4],
                    mx.concatenate([undo[4], kv], axis=1),
                    mx.concatenate([undo[5], gate], axis=1),
                    *undo[6:],
                )
            else:
                undo_kv, undo_gate = self._stash_undo_buffers()
                self._undo = (
                    undo_kv,
                    undo_gate,
                    self.remainder,
                    self.pooled,
                    kv,
                    gate,
                    self.prev_win_kv,
                    self.prev_win_gate,
                )
            self._undo_chain = decode_consistent
        else:
            self._undo = None
            self._undo_chain = False

        row = slice(self.slot, self.slot + 1)
        if length > 1:
            total = int(length) + self.remainder
            usable = (total // self.ratio) * self.ratio
            new_remainder = total % self.ratio
            if usable > 0:
                ready_kv = mx.concatenate(
                    [
                        self._storage.buf_kv[row, : self.remainder],
                        kv[:, : (usable - self.remainder)],
                    ],
                    axis=1,
                )
                ready_gate = mx.concatenate(
                    [
                        self._storage.buf_gate[row, : self.remainder],
                        gate[:, : (usable - self.remainder)],
                    ],
                    axis=1,
                )
                ready_base = offset - self.remainder
            else:
                ready_kv = mx.zeros((1, 0, dim_kv), dtype=kv.dtype)
                ready_gate = mx.zeros((1, 0, dim_gate), dtype=gate.dtype)
                ready_base = 0
            if new_remainder:
                self._storage.buf_kv[row, :new_remainder] = kv[
                    :, -new_remainder:
                ]
                self._storage.buf_gate[row, :new_remainder] = gate[
                    :, -new_remainder:
                ]
            self.remainder = new_remainder
            return ready_kv, ready_gate, ready_base

        index = self.remainder
        self._storage.buf_kv[row, index : index + 1] = kv
        self._storage.buf_gate[row, index : index + 1] = gate
        self.remainder = (index + 1) % self.ratio
        if self.remainder == 0:
            return (
                self._storage.buf_kv[row],
                self._storage.buf_gate[row],
                offset - self.ratio + 1,
            )
        return (
            mx.zeros((1, 0, dim_kv), dtype=kv.dtype),
            mx.zeros((1, 0, dim_gate), dtype=gate.dtype),
            0,
        )

    @property
    def prev_win_kv(self):
        if not self._prev_valid or self._storage.prev_win_kv is None:
            return None
        return self._storage.prev_win_kv[self.slot : self.slot + 1]

    @prev_win_kv.setter
    def prev_win_kv(self, value):
        if value is None:
            self._prev_valid = False
            return
        if self._storage.prev_win_kv is None:
            raise FixedKVCacheError("This pooled cache has no overlap carry storage")
        self._storage.prev_win_kv[self.slot : self.slot + 1] = value
        self._prev_valid = True

    @property
    def prev_win_gate(self):
        if not self._prev_valid or self._storage.prev_win_gate is None:
            return None
        return self._storage.prev_win_gate[self.slot : self.slot + 1]

    @prev_win_gate.setter
    def prev_win_gate(self, value):
        if value is None:
            self._prev_valid = False
            return
        if self._storage.prev_win_gate is None:
            raise FixedKVCacheError("This pooled cache has no overlap carry storage")
        self._storage.prev_win_gate[self.slot : self.slot + 1] = value
        self._prev_valid = True

    def _grow_pool(self, needed: int) -> None:
        raise FixedKVCacheCapacityError(
            f"Pooled cache capacity is {self._storage.logical_capacity:,} rows; "
            f"write requires {needed:,}. Reduce prompt or output tokens."
        )

    def update_and_fetch(self, px):
        length = int(px.shape[1])
        if length == 0:
            if self._pool_len == 0:
                return mx.zeros(
                    (int(px.shape[0]), 0, int(px.shape[2])), dtype=px.dtype
                )
            return self.pooled
        needed = self._pool_len + length
        if needed > self._storage.logical_capacity:
            self._grow_pool(needed)
        row = slice(self.slot, self.slot + 1)
        self._storage.pooled[row, self._pool_len : needed] = px
        self._pool_len = needed
        return self.pooled

    def compact_to(self, base: int = 0) -> None:
        destination = int(base)
        if self.slot == destination:
            return
        if self.slot < destination:
            raise FixedKVCacheError("Fixed pooled-cache rows cannot compact upward")
        self._storage.move_row(self.slot, destination)
        self.slot = destination

    @property
    def state(self):
        buf_kv = self.buf_kv[:, : self.remainder] if self.remainder else None
        buf_gate = self.buf_gate[:, : self.remainder] if self.remainder else None
        return (
            buf_kv,
            buf_gate,
            self.pooled,
            self.prev_win_kv,
            self.prev_win_gate,
        )

    @state.setter
    def state(self, v):
        if len(v) == 3:
            buf_kv, buf_gate, pooled = v
            prev_win_kv = prev_win_gate = None
        elif len(v) == 5:
            buf_kv, buf_gate, pooled, prev_win_kv, prev_win_gate = v
        else:
            raise ValueError(
                f"PoolingCache state must have 3 or 5 elements, got {len(v)}"
            )
        self.buf_kv = buf_kv
        self.buf_gate = buf_gate
        self.remainder = 0 if buf_kv is None else int(buf_kv.shape[1])
        self.pooled = pooled
        self.prev_win_kv = prev_win_kv
        self.prev_win_gate = prev_win_gate
        self._undo = None
        self._undo_chain = False

    def is_trimmable(self):
        if self._pool_len == 0 or self.remainder >= 1:
            return True
        return self._can_undo(1)

    def trim(self, n):
        if n <= self.remainder:
            self.remainder -= n
            self._undo = None
            self._undo_chain = False
            return n
        if not self._can_undo(n):
            return 0
        undo: Any = self._undo
        buf_kv, buf_gate, _rem, pooled, kv, gate, prev_kv, prev_gate = undo
        self._undo = None
        self._undo_chain = False
        keep = int(kv.shape[1]) - n
        prefix_kv = kv[:, :keep]
        prefix_gate = gate[:, :keep]
        if buf_kv is not None:
            prefix_kv = mx.concatenate([buf_kv, prefix_kv], axis=1)
            prefix_gate = mx.concatenate([buf_gate, prefix_gate], axis=1)

        completed = int(prefix_kv.shape[1]) // self.ratio
        previous_pooled = 0 if pooled is None else int(pooled.shape[1])
        if completed == 0:
            self._pool_len = previous_pooled
            self.prev_win_kv = prev_kv
            self.prev_win_gate = prev_gate
        else:
            self._pool_len = previous_pooled + completed
            end = completed * self.ratio
            start = end - self.ratio
            self.prev_win_kv = prefix_kv[:, start:end, :][:, None]
            self.prev_win_gate = prefix_gate[:, start:end, :][:, None]

        used = completed * self.ratio
        remainder_kv = prefix_kv[:, used:]
        remainder_gate = prefix_gate[:, used:]
        self.remainder = int(remainder_kv.shape[1])
        self.buf_kv = remainder_kv if self.remainder else None
        self.buf_gate = remainder_gate if self.remainder else None
        return n

    def empty(self):
        return self._pool_len == 0 and self.remainder == 0

    @property
    def nbytes(self):
        return self._storage.per_slot_nbytes

    @classmethod
    def merge(cls, caches):
        return FixedBatchPoolingCache.merge(caches)


class FixedBatchPoolingCache(BatchPoolingCache):
    """DeepSeek batched pooled cache backed by fixed session rows."""

    def __init__(self, storage: _PoolingStorage, slots: list[int]):
        self._storage = storage
        self.slots = list(slots)
        self.ratio = int(storage.ratio)
        size = len(slots)
        self.remainder = [0] * size
        self._pool_lengths = [0] * size
        self._pool_extent = 0
        self._lengths = [2**31] * size
        self._processed = [0] * size
        self._undo = None
        self._undo_chain = False
        self._mtp_cross_boundary_rollback = size == 1
        self._prev_valid = [False] * size
        self._last_usable = [0] * size
        self._require_compact_slots()

    def _require_compact_slots(self) -> None:
        start = self.slots[0] if self.slots else 0
        expected = list(range(start, start + len(self.slots)))
        if self.slots != expected:
            raise FixedKVCacheError(
                f"Fixed pooled rows must stay compact; got {self.slots}, expected {expected}"
            )

    def _rows(self, array: mx.array) -> mx.array:
        start = self.slots[0] if self.slots else 0
        return array[start : start + len(self.slots)]

    @property
    def _pool_buf(self):
        return self._rows(self._storage.pooled)

    @_pool_buf.setter
    def _pool_buf(self, value):
        if value is None:
            self._pool_extent = 0
            return
        length = int(value.shape[1])
        if length > self._storage.logical_capacity:
            raise FixedKVCacheCapacityError(
                f"Pooled cache capacity is {self._storage.logical_capacity:,} rows; "
                f"batch write requires {length:,}."
            )
        start = self.slots[0] if self.slots else 0
        stop = start + len(self.slots)
        self._storage.pooled[start:stop, :length] = value[:, :length]

    @property
    def pooled(self):
        if self._pool_extent == 0:
            return None
        return self._pool_buf[:, : self._pool_extent]

    @pooled.setter
    def pooled(self, v):
        if v is None:
            self._pool_extent = 0
            return
        self._pool_buf = v
        self._pool_extent = int(v.shape[1])

    @property
    def buf_kv(self):
        return self._rows(self._storage.buf_kv)

    @buf_kv.setter
    def buf_kv(self, value):
        self._restore_buffer_rows(self._storage.buf_kv, value, "KV")

    @property
    def buf_gate(self):
        return self._rows(self._storage.buf_gate)

    @buf_gate.setter
    def buf_gate(self, value):
        self._restore_buffer_rows(self._storage.buf_gate, value, "gate")

    def _restore_buffer_rows(self, storage, value, label: str) -> None:
        """Restore compact batch rows without rebinding committed storage."""
        start = self.slots[0] if self.slots else 0
        stop = start + len(self.slots)
        if value is None:
            storage[start:stop] = 0
            mx.eval(storage)
            return
        rows = int(value.shape[0])
        length = int(value.shape[1])
        if rows != len(self.slots) or length > self.ratio:
            raise FixedKVCacheError(
                f"Pooled batch {label} restore shape {tuple(value.shape)} does "
                f"not fit {len(self.slots)} fixed rows of {self.ratio} tokens"
            )
        storage[start:stop, :length] = value[:, :length]
        if length < self.ratio:
            storage[start:stop, length:] = 0
        mx.eval(storage)

    @property
    def prev_win_kv(self):
        if not any(self._prev_valid) or self._storage.prev_win_kv is None:
            return None
        return self._rows(self._storage.prev_win_kv)

    @prev_win_kv.setter
    def prev_win_kv(self, value):
        if value is None:
            self._prev_valid = [False] * len(self.slots)
            return
        if self._storage.prev_win_kv is None:
            raise FixedKVCacheError("This pooled batch has no overlap carry storage")
        start = self.slots[0] if self.slots else 0
        stop = start + len(self.slots)
        self._storage.prev_win_kv[start:stop] = value

    @property
    def prev_win_gate(self):
        if not any(self._prev_valid) or self._storage.prev_win_gate is None:
            return None
        return self._rows(self._storage.prev_win_gate)

    @prev_win_gate.setter
    def prev_win_gate(self, value):
        if value is None:
            self._prev_valid = [False] * len(self.slots)
            return
        if self._storage.prev_win_gate is None:
            raise FixedKVCacheError("This pooled batch has no overlap carry storage")
        start = self.slots[0] if self.slots else 0
        stop = start + len(self.slots)
        self._storage.prev_win_gate[start:stop] = value

    def update_and_fetch(self, px):
        batch, columns, dim = (int(value) for value in px.shape)
        if columns == 0:
            if self._pool_extent == 0:
                return mx.zeros((batch, 0, dim), dtype=px.dtype)
            return self.pooled
        counts = [
            (self._processed[i] - self.remainder[i]) // self.ratio
            - self._pool_lengths[i]
            for i in range(batch)
        ]
        max_new = max(counts)
        if max_new == 0:
            if self._pool_extent == 0:
                return mx.zeros((batch, 0, dim), dtype=px.dtype)
            return self.pooled
        max_pool = max(self._pool_lengths) + max_new
        if max_pool > self._storage.logical_capacity:
            raise FixedKVCacheCapacityError(
                f"Pooled cache capacity is {self._storage.logical_capacity:,} rows; "
                f"batch write requires {max_pool:,}."
            )
        start = self.slots[0] if self.slots else 0
        for row, count in enumerate(counts):
            if count <= 0:
                continue
            previous = self._pool_lengths[row]
            self._storage.pooled[
                start + row, previous : previous + count
            ] = px[row, :count]
            self._pool_lengths[row] = previous + count
        self._pool_extent = max(self._pool_extent, max_pool)
        return self.pooled

    def accumulate_windows(self, kv, gate, offset):
        """Batch pooling with all persistent writes kept in reserved rows."""
        batch, length, dim_kv = (int(value) for value in kv.shape)
        _, _, dim_gate = (int(value) for value in gate.shape)
        if batch != len(self.slots):
            raise FixedKVCacheError(
                f"Pooled batch has {batch} rows for {len(self.slots)} fixed slots"
            )
        start = self.slots[0] if self.slots else 0
        stop = start + batch

        if length <= 8:
            try:
                from omlx.patches.mlx_lm_mtp import cache_rollback

                decode_consistent = cache_rollback._is_undo_armed() and (
                    length == 1 or cache_rollback._is_decode_consistent_armed()
                )
            except Exception:
                decode_consistent = False
            if decode_consistent and self._undo_chain:
                undo: Any = self._undo
                self._undo = (
                    *undo[:5],
                    mx.concatenate([undo[5], kv], axis=1),
                    mx.concatenate([undo[6], gate], axis=1),
                    *undo[7:],
                )
            else:
                undo_kv = self._storage.undo_buf_kv
                undo_gate = self._storage.undo_buf_gate
                if undo_kv is None or undo_gate is None:
                    raise FixedKVCacheError(
                        "Pooled rollback requires launch-time per-session undo storage"
                    )
                undo_kv[start:stop] = self._storage.buf_kv[start:stop]
                undo_gate[start:stop] = self._storage.buf_gate[start:stop]
                mx.eval(undo_kv, undo_gate)
                self._undo = (
                    undo_kv[start:stop],
                    undo_gate[start:stop],
                    list(self.remainder),
                    list(self._pool_lengths),
                    list(self._processed),
                    kv,
                    gate,
                    self.prev_win_kv,
                    self.prev_win_gate,
                    list(self._prev_valid),
                )
            self._undo_chain = decode_consistent
        else:
            self._undo = None
            self._undo_chain = False

        valid = [
            min(limit - processed, length)
            for limit, processed in zip(self._lengths, self._processed)
        ]
        if max(valid) != length:
            raise RuntimeError()
        for row in range(batch):
            self._processed[row] += valid[row]
        totals = [count + rem for count, rem in zip(valid, self.remainder)]
        usable = [(total // self.ratio) * self.ratio for total in totals]
        max_usable = max(usable)
        next_remainder = [total % self.ratio for total in totals]
        self._last_usable = usable

        if max_usable == 0:
            for row in range(batch):
                rem = self.remainder[row]
                count = valid[row]
                self._storage.buf_kv[
                    start + row, rem : rem + count
                ] = kv[row, :count]
                self._storage.buf_gate[
                    start + row, rem : rem + count
                ] = gate[row, :count]
            self.remainder = next_remainder
            return (
                mx.zeros((batch, 0, dim_kv), dtype=kv.dtype),
                mx.zeros((batch, 0, dim_gate), dtype=gate.dtype),
                0,
            )

        ready_kv = mx.zeros((batch, max_usable, dim_kv), dtype=kv.dtype)
        ready_gate = mx.zeros((batch, max_usable, dim_gate), dtype=gate.dtype)
        ready_base = [0] * batch
        for row in range(batch):
            rem = self.remainder[row]
            count = valid[row]
            used = usable[row]
            if used > 0:
                if rem:
                    ready_kv[row, :rem] = self._storage.buf_kv[
                        start + row, :rem
                    ]
                    ready_gate[row, :rem] = self._storage.buf_gate[
                        start + row, :rem
                    ]
                consume = used - rem
                ready_kv[row, rem : rem + consume] = kv[row, :consume]
                ready_gate[row, rem : rem + consume] = gate[row, :consume]
                ready_base[row] = (
                    int(offset[row]) - rem
                    if isinstance(offset, mx.array)
                    else offset - rem
                )
        mx.eval(ready_kv, ready_gate)
        for row in range(batch):
            count = valid[row]
            used = usable[row]
            rem = self.remainder[row]
            next_rem = next_remainder[row]
            if used > 0:
                self._storage.buf_kv[start + row] = 0
                self._storage.buf_gate[start + row] = 0
                if next_rem:
                    self._storage.buf_kv[start + row, :next_rem] = kv[
                        row, count - next_rem : count
                    ]
                    self._storage.buf_gate[start + row, :next_rem] = gate[
                        row, count - next_rem : count
                    ]
            else:
                if count:
                    self._storage.buf_kv[
                        start + row, rem : rem + count
                    ] = kv[row, :count]
                    self._storage.buf_gate[
                        start + row, rem : rem + count
                    ] = gate[row, :count]
        self.remainder = next_remainder
        return ready_kv, ready_gate, mx.array(ready_base)

    def trim(self, n):
        if n <= min(self.remainder):
            for row in range(len(self.remainder)):
                self.remainder[row] -= n
                self._processed[row] -= n
            self._truncate_pooled_tail()
            self._undo = None
            self._undo_chain = False
            return n
        if not self._can_undo(n):
            return 0
        undo: Any = self._undo
        (
            buf_kv,
            buf_gate,
            remainder,
            pool_lengths,
            processed,
            kv,
            gate,
            prev_kv,
            prev_gate,
            prev_valid,
        ) = undo
        if self._mtp_cross_boundary_rollback:
            self._undo = None
            self._undo_chain = False
            keep = int(kv.shape[1]) - n
            prefix_kv = mx.concatenate(
                [buf_kv[:, : remainder[0]], kv[:, :keep]], axis=1
            )
            prefix_gate = mx.concatenate(
                [buf_gate[:, : remainder[0]], gate[:, :keep]], axis=1
            )
            completed = int(prefix_kv.shape[1]) // self.ratio
            next_pool = pool_lengths[0] + completed
            self._pool_extent = next_pool
            self._pool_lengths = [next_pool]
            self._processed = [processed[0] + keep]

            used = completed * self.ratio
            remainder_kv = prefix_kv[:, used:]
            remainder_gate = prefix_gate[:, used:]
            next_rem = int(remainder_kv.shape[1])
            self.remainder = [next_rem]
            self.buf_kv = remainder_kv if next_rem else None
            self.buf_gate = remainder_gate if next_rem else None
            if completed:
                start = used - self.ratio
                self.prev_win_kv = prefix_kv[:, start:used, :][:, None]
                self.prev_win_gate = prefix_gate[:, start:used, :][:, None]
                self._prev_valid = [True]
            else:
                self.prev_win_kv = prev_kv
                self.prev_win_gate = prev_gate
                self._prev_valid = list(prev_valid)
            self._last_usable = [used]
            return n

        self._undo = None
        decode_consistent = self._undo_chain
        self._undo_chain = False
        keep = int(kv.shape[1]) - n
        self.buf_kv = buf_kv
        self.buf_gate = buf_gate
        self.remainder = list(remainder)
        self._pool_lengths = list(pool_lengths)
        self._processed = list(processed)
        self._truncate_pooled_tail()
        self.prev_win_kv = prev_kv
        self.prev_win_gate = prev_gate
        self._prev_valid = list(prev_valid)
        if keep > 0:
            if decode_consistent:
                for index in range(keep):
                    self.accumulate_windows(
                        kv[:, index : index + 1], gate[:, index : index + 1], 0
                    )
            else:
                self.accumulate_windows(kv[:, :keep], gate[:, :keep], 0)
            self._undo = None
            self._undo_chain = False
        return n

    @classmethod
    def merge(cls, caches):
        if not caches:
            raise FixedKVCacheError("Cannot merge an empty pooled-cache list")
        if len(caches) == 1 and isinstance(caches[0], cls):
            return caches[0]
        if any(isinstance(cache, cls) for cache in caches):
            raise FixedKVCacheError(
                "Fixed pooled batches can only be remerged one batch at a time"
            )
        storage = caches[0]._storage
        if any(cache._storage is not storage for cache in caches):
            raise FixedKVCacheError("Cannot merge pooled caches from different pools")
        if any(cache.ratio != caches[0].ratio for cache in caches):
            raise FixedKVCacheError("Fixed pooled-cache ratios do not match")
        slots = [cache.slot for cache in caches]
        obj = cls(storage, slots)
        obj.remainder = [cache.remainder for cache in caches]
        obj._pool_lengths = [cache._pool_len for cache in caches]
        obj._pool_extent = max(obj._pool_lengths, default=0)
        obj._processed = [
            cache.remainder + cache._pool_len * cache.ratio for cache in caches
        ]
        obj._prev_valid = [cache._prev_valid for cache in caches]
        return obj

    def extend(self, other):
        if (
            not isinstance(other, FixedBatchPoolingCache)
            or other._storage is not self._storage
        ):
            raise FixedKVCacheError("Fixed pooled batches must share one pool")
        if len(other.slots) != 1:
            raise FixedKVCacheError(
                "Fixed pooled admission adds one cache slot at a time"
            )
        expected_slot = len(self.slots)
        source_slot = other.slots[0]
        if source_slot < expected_slot:
            raise FixedKVCacheError("Pooled-cache row ownership is inconsistent")
        if source_slot != expected_slot:
            self._storage.move_row(source_slot, expected_slot)
            other.slots = [expected_slot]
        self.slots.append(expected_slot)
        self.remainder.extend(other.remainder)
        self._pool_lengths.extend(other._pool_lengths)
        self._pool_extent = max(self._pool_extent, other._pool_extent)
        self._lengths.extend(other._lengths)
        self._processed.extend(other._processed)
        self._prev_valid.extend(other._prev_valid)
        self._last_usable.extend(other._last_usable)
        self._mtp_cross_boundary_rollback = len(self.slots) == 1

    def filter(self, batch_indices):
        indices = _filter_indices(batch_indices, len(self.slots))
        old_slots = list(self.slots)
        for destination, source_index in enumerate(indices):
            source = old_slots[int(source_index)]
            if source != destination:
                self._storage.move_row(source, destination)
        self.remainder = [self.remainder[i] for i in indices]
        self._pool_lengths = [self._pool_lengths[i] for i in indices]
        self._lengths = [self._lengths[i] for i in indices]
        self._processed = [self._processed[i] for i in indices]
        self._prev_valid = [self._prev_valid[i] for i in indices]
        self._last_usable = [self._last_usable[i] for i in indices]
        self.slots = list(range(len(indices)))
        self._pool_extent = max(self._pool_lengths, default=0)
        self._undo = None
        self._undo_chain = False

    def compact_to(self, base: int = 0) -> None:
        target = list(range(int(base), int(base) + len(self.slots)))
        if self.slots == target:
            return
        if any(source < destination for source, destination in zip(self.slots, target)):
            raise FixedKVCacheError("Fixed pooled-cache rows cannot compact upward")
        for source, destination in zip(self.slots, target):
            self._storage.move_row(source, destination)
        self.slots = target

    def extract(self, idx):
        index = int(idx)
        cache = FixedPoolingCache(self._storage, self.slots[index])
        cache.remainder = self.remainder[index]
        cache._pool_len = self._pool_lengths[index]
        cache._prev_valid = self._prev_valid[index]
        return cache

    def empty(self):
        return self._pool_extent == 0 and all(value == 0 for value in self.remainder)

    @property
    def nbytes(self):
        return self._storage.per_slot_nbytes * len(self.slots)


@dataclass
class _KVBlueprint:
    storage: _KVStorage
    rotating_max_size: int | None = None
    keep: int = 0

    def make(self, slot: int):
        if self.rotating_max_size is None:
            return FixedKVCache(self.storage, slot)
        return FixedRotatingKVCache(
            self.storage,
            slot,
            self.rotating_max_size,
            self.keep,
        )


@dataclass
class _ArraysBlueprint:
    storage: _ArraysStorage

    def make(self, slot: int):
        return FixedArraysCache(self.storage, [slot])


@dataclass
class _NullBlueprint:
    def make(self, slot: int):
        return FixedNullCache([slot])


@dataclass
class _PoolingBlueprint:
    storage: _PoolingStorage

    def make(self, slot: int):
        return FixedPoolingCache(self.storage, slot)


@dataclass
class _MiniMaxM3Blueprint:
    kv: _KVBlueprint
    index: _KVBlueprint

    def make(self, slot: int):
        kv_cache = self.kv.make(slot)
        index_cache = self.index.make(slot)
        if not isinstance(kv_cache, FixedKVCache) or not isinstance(
            index_cache, FixedKVCache
        ):
            raise FixedKVCacheError("MiniMax M3 requires linear fixed cache rows")
        return FixedMiniMaxM3KVCache(kv_cache, index_cache)


@dataclass
class _ChunkedBlueprint:
    storage: _KVStorage
    chunk_size: int

    def make(self, slot: int):
        return FixedChunkedKVCache(self.storage, slot, self.chunk_size)


@dataclass
class _RingSlidingBlueprint:
    storage: _KVStorage
    window_size: int

    def make(self, slot: int):
        return FixedRingSlidingKVCache(self.storage, slot, self.window_size)


@dataclass
class _DSparkBlueprint:
    storage: _DSparkStorage

    def make(self, slot: int):
        return FixedDSparkContextCache(self.storage, slot)


def _blueprint_per_slot_bytes(blueprint: Any) -> int:
    if isinstance(blueprint, _ListBlueprint):
        return sum(_blueprint_per_slot_bytes(child) for child in blueprint.children)
    if isinstance(blueprint, _MiniMaxM3Blueprint):
        return _blueprint_per_slot_bytes(blueprint.kv) + _blueprint_per_slot_bytes(
            blueprint.index
        )
    storage = getattr(blueprint, "storage", None)
    return int(getattr(storage, "per_slot_nbytes", 0) or 0)


@dataclass
class _ListBlueprint:
    children: tuple[Any, ...]

    def make(self, slot: int):
        return CacheList(*(child.make(slot) for child in self.children))


class FixedKVCachePool:
    """Own all fixed cache arrays for one loaded model."""

    def __init__(
        self,
        *,
        context_window: int,
        slots: int,
        blueprints: list[Any],
        mtp_blueprints: list[Any],
        mtp_clone_blueprints: list[Any],
        arrays: list[mx.array],
        pool_scratch_bytes: int,
        probe_bytes: int,
        active_memory_before: int,
        active_memory_after: int,
    ) -> None:
        self.context_window = int(context_window)
        self.slots = int(slots)
        self._blueprints = blueprints
        self._mtp_blueprints = mtp_blueprints
        self._mtp_clone_blueprints = mtp_clone_blueprints
        self._arrays = arrays
        self._provider_owners: list[Any] = []
        self.native_mtp_bytes_per_session = sum(
            _blueprint_per_slot_bytes(blueprint)
            for blueprint in (*mtp_blueprints, *mtp_clone_blueprints)
        )
        self.pool_scratch_bytes = int(pool_scratch_bytes)
        self.probe_bytes = int(probe_bytes)
        self.active_memory_before = int(active_memory_before)
        self.active_memory_after = int(active_memory_after)
        self.committed_bytes = sum(int(array.nbytes) for array in arrays)
        self.serving_bytes = self.committed_bytes - self.pool_scratch_bytes
        self.materialized_delta_bytes = max(
            0, active_memory_after - active_memory_before
        )

    @classmethod
    def create(
        cls,
        model: Any,
        *,
        context_window: int,
        slots: int,
        probe_token_id: int = 0,
        prefill_step_size: int = 2048,
        native_mtp_enabled: bool = False,
    ) -> FixedKVCachePool:
        if context_window <= 0:
            raise FixedKVCacheError("Fixed KV context window must be positive")
        if slots <= 0:
            raise FixedKVCacheError(
                "Fixed KV cache must reserve at least one session slot"
            )

        probe_cache = make_prompt_cache(model)
        if not isinstance(probe_cache, (list, tuple)) or not probe_cache:
            raise FixedKVCacheError("Model did not return a usable cache tree")

        model_args = getattr(model, "args", None)
        model_type = str(getattr(model_args, "model_type", "")).lower()
        null_cache_indices: set[int] = set()
        if model_type == "nemotron-nas":
            cache_index = 0
            for block in getattr(model_args, "block_configs", ()):
                attention = getattr(block, "attention", None)
                if bool(getattr(attention, "no_op", False)):
                    continue
                if bool(getattr(attention, "replace_with_linear", False)):
                    null_cache_indices.add(cache_index)
                cache_index += 1

        probe_length = 1
        if model_type == "longcat_flash_ngram":
            probe_length = min(
                context_window,
                max(1, int(getattr(model_args, "emb_neighbor_num", 4)) - 1),
            )
        probe_tokens = mx.full(
            (1, probe_length), int(probe_token_id), dtype=mx.int32
        )
        if native_mtp_enabled:
            try:
                output = model(
                    probe_tokens,
                    cache=probe_cache,
                    return_hidden=True,
                )
            except TypeError:
                output = model(probe_tokens, cache=probe_cache)
        else:
            output = model(probe_tokens, cache=probe_cache)
        probe_arrays = _cache_arrays(probe_cache)
        if not probe_arrays and len(null_cache_indices) != len(probe_cache):
            raise FixedKVCacheError("The cache probe produced no tensors")
        mx.eval(*probe_arrays)
        mx.synchronize()
        mtp_probe_cache: list[Any] = []
        mtp_head_clone = False
        if native_mtp_enabled and callable(getattr(model, "make_mtp_cache", None)):
            made = model.make_mtp_cache()
            if made:
                if not isinstance(made, (list, tuple)):
                    raise FixedKVCacheError("Native MTP returned an invalid cache tree")
                mtp_probe_cache = list(made)
                hidden = _mtp_hidden(output)
                if hidden is None:
                    raise FixedKVCacheError(
                        "Native MTP is active but the model cache probe did not "
                        "return hidden states"
                    )
                mtp_forward = getattr(model, "mtp_forward", None)
                if not callable(mtp_forward):
                    raise FixedKVCacheError(
                        "Native MTP is active but the model has no mtp_forward method"
                    )
                mtp_output = mtp_forward(
                    hidden[:, -1:],
                    probe_tokens[:, -1:],
                    mtp_probe_cache,
                )
                mtp_arrays = _cache_arrays(mtp_probe_cache)
                if not mtp_arrays:
                    raise FixedKVCacheError(
                        "Native MTP cache probe produced no materialized tensors"
                    )
                mx.eval(*mtp_arrays)
                probe_arrays.extend(mtp_arrays)
                del mtp_output
                mtp_head_clone = _model_flag(model, "_omlx_mtp_head_clone")
        probe_bytes = sum(int(array.nbytes) for array in probe_arrays)
        del output

        allocated: list[mx.array] = []
        scratch_allocated: list[mx.array] = []
        scratch_registry: dict[tuple[str, str, tuple[int, ...]], mx.array] = {}

        def shared_scratch(
            role: str,
            shape: tuple[int, ...],
            dtype: Any,
            *,
            force: bool = False,
        ) -> mx.array | None:
            if slots <= 1 and not force:
                return None
            scratch_shape = (1,) + shape[1:]
            signature = (role, str(dtype), scratch_shape[1:])
            existing = scratch_registry.get(signature)
            if existing is not None:
                return existing
            scratch = mx.zeros(scratch_shape, dtype=dtype)
            scratch_registry[signature] = scratch
            allocated.append(scratch)
            scratch_allocated.append(scratch)
            return scratch

        def blueprint(cache: Any, path: str, *, allow_empty: bool = False):
            if type(cache).__name__ == "DSparkContextCache":
                keys = getattr(cache, "keys", None)
                max_size = int(getattr(cache, "max_size", 0) or 0)
                if keys is None or max_size <= 0:
                    raise FixedKVCacheError(
                        f"{path}: DSpark context cache probe stayed empty"
                    )
                physical_size = min(context_window, max_size)
                key_shape = (
                    slots,
                    int(keys.shape[1]),
                    physical_size,
                    int(keys.shape[3]),
                )
                fixed_keys = mx.zeros(key_shape, dtype=keys.dtype)
                allocated.append(fixed_keys)
                scratch = shared_scratch(
                    "keys",
                    key_shape,
                    keys.dtype,
                    force=True,
                )
                if scratch is None:
                    raise FixedKVCacheError(
                        f"{path}: DSpark ring rotation scratch was not reserved"
                    )
                return _DSparkBlueprint(
                    _DSparkStorage(fixed_keys, physical_size, scratch)
                )

            if type(cache).__name__ == "MiniMaxM3KVCache":
                inner = getattr(cache, "kv_cache", None)
                index_keys = getattr(cache, "index_keys", None)
                if not isinstance(inner, KVCache):
                    raise FixedKVCacheError(
                        f"{path}: MiniMax M3 inner cache is not linear K/V"
                    )
                if inner.keys is None:
                    raise FixedKVCacheError(
                        f"{path}: MiniMax M3 K/V probe stayed empty"
                    )
                if index_keys is None:
                    raise FixedKVCacheError(
                        f"{path}: MiniMax M3 sparse index probe stayed empty"
                    )

                kv_blueprint = blueprint(inner, f"{path}.kv")
                if not isinstance(kv_blueprint, _KVBlueprint):
                    raise FixedKVCacheError(
                        f"{path}: MiniMax M3 inner cache is not a linear K/V cache"
                    )

                capacity = _round_up(
                    context_window,
                    int(getattr(cache, "step", 256) or 256),
                )
                index_shape = (
                    slots,
                    int(index_keys.shape[1]),
                    capacity,
                    int(index_keys.shape[3]),
                )
                value_shape = (
                    slots,
                    int(index_keys.shape[1]),
                    capacity,
                    0,
                )
                index_array = mx.zeros(index_shape, dtype=index_keys.dtype)
                empty_values = mx.zeros(value_shape, dtype=index_keys.dtype)
                allocated.extend([index_array, empty_values])
                index_storage = _KVStorage(
                    index_array,
                    empty_values,
                    capacity,
                    context_window,
                    scratch_keys=shared_scratch(
                        "index_keys", index_shape, index_keys.dtype
                    ),
                    scratch_values=shared_scratch(
                        "index_values", value_shape, index_keys.dtype
                    ),
                )
                return _MiniMaxM3Blueprint(
                    kv_blueprint,
                    _KVBlueprint(index_storage),
                )

            if type(cache).__name__ == "RingSlidingKVCache":
                if cache.keys is None or cache.values is None:
                    raise FixedKVCacheError(
                        f"{path}: ring-sliding cache probe stayed empty"
                    )
                capacity = _round_up(
                    context_window,
                    int(getattr(cache, "step", 256) or 256),
                )
                key_shape = (
                    slots,
                    int(cache.keys.shape[1]),
                    capacity,
                    int(cache.keys.shape[3]),
                )
                value_shape = (
                    slots,
                    int(cache.values.shape[1]),
                    capacity,
                    int(cache.values.shape[3]),
                )
                keys = mx.zeros(key_shape, dtype=cache.keys.dtype)
                values = mx.zeros(value_shape, dtype=cache.values.dtype)
                allocated.extend([keys, values])
                return _RingSlidingBlueprint(
                    _KVStorage(
                        keys,
                        values,
                        capacity,
                        context_window,
                        scratch_keys=shared_scratch(
                            "keys", key_shape, cache.keys.dtype
                        ),
                        scratch_values=shared_scratch(
                            "values", value_shape, cache.values.dtype
                        ),
                    ),
                    int(cache.window_size),
                )

            if isinstance(cache, ChunkedKVCache):
                if cache.keys is None or cache.values is None:
                    raise FixedKVCacheError(
                        f"{path}: chunked cache probe stayed empty"
                    )
                context_capacity = _round_up(
                    context_window,
                    int(getattr(cache, "step", 256) or 256),
                )
                incoming_capacity = _round_up(
                    min(context_window, max(1, int(prefill_step_size))),
                    int(getattr(cache, "step", 256) or 256),
                )
                physical_capacity = min(
                    context_capacity,
                    int(cache.chunk_size) + incoming_capacity,
                )
                key_shape = (
                    slots,
                    int(cache.keys.shape[1]),
                    physical_capacity,
                    int(cache.keys.shape[3]),
                )
                value_shape = (
                    slots,
                    int(cache.values.shape[1]),
                    physical_capacity,
                    int(cache.values.shape[3]),
                )
                keys = mx.zeros(key_shape, dtype=cache.keys.dtype)
                values = mx.zeros(value_shape, dtype=cache.values.dtype)
                allocated.extend([keys, values])
                return _ChunkedBlueprint(
                    _KVStorage(
                        keys,
                        values,
                        physical_capacity,
                        context_window,
                        scratch_keys=shared_scratch(
                            "keys", key_shape, cache.keys.dtype, force=True
                        ),
                        scratch_values=shared_scratch(
                            "values", value_shape, cache.values.dtype, force=True
                        ),
                    ),
                    int(cache.chunk_size),
                )

            if type(cache) is KVCache or (
                isinstance(cache, KVCache)
                and not isinstance(cache, RotatingKVCache)
                and type(cache).__name__ in {"KVCache", "PrefillReadyKVCache"}
            ):
                if cache.keys is None or cache.values is None:
                    if allow_empty:
                        return _NullBlueprint()
                    raise FixedKVCacheError(f"{path}: KV cache probe stayed empty")
                capacity = _round_up(
                    context_window, int(getattr(cache, "step", 256) or 256)
                )
                key_shape = (
                    slots,
                    int(cache.keys.shape[1]),
                    capacity,
                    int(cache.keys.shape[3]),
                )
                val_shape = (
                    slots,
                    int(cache.values.shape[1]),
                    capacity,
                    int(cache.values.shape[3]),
                )
                keys = mx.zeros(key_shape, dtype=cache.keys.dtype)
                values = mx.zeros(val_shape, dtype=cache.values.dtype)
                allocated.extend([keys, values])
                scratch_keys = shared_scratch(
                    "keys",
                    key_shape,
                    cache.keys.dtype,
                )
                scratch_values = shared_scratch(
                    "values",
                    val_shape,
                    cache.values.dtype,
                )
                return _KVBlueprint(
                    _KVStorage(
                        keys,
                        values,
                        capacity,
                        context_window,
                        scratch_keys=scratch_keys,
                        scratch_values=scratch_values,
                    )
                )

            if isinstance(cache, RotatingKVCache):
                if cache.keys is None or cache.values is None:
                    raise FixedKVCacheError(
                        f"{path}: rotating cache probe stayed empty"
                    )
                physical_size = min(
                    _round_up(
                        context_window,
                        int(getattr(cache, "step", 256) or 256),
                    ),
                    int(cache.max_size),
                )
                max_size = min(context_window, int(cache.max_size))
                key_shape = (
                    slots,
                    int(cache.keys.shape[1]),
                    physical_size,
                    int(cache.keys.shape[3]),
                )
                val_shape = (
                    slots,
                    int(cache.values.shape[1]),
                    physical_size,
                    int(cache.values.shape[3]),
                )
                keys = mx.zeros(key_shape, dtype=cache.keys.dtype)
                values = mx.zeros(val_shape, dtype=cache.values.dtype)
                allocated.extend([keys, values])
                scratch_keys = shared_scratch(
                    "keys",
                    key_shape,
                    cache.keys.dtype,
                )
                scratch_values = shared_scratch(
                    "values",
                    val_shape,
                    cache.values.dtype,
                )
                storage = _KVStorage(
                    keys,
                    values,
                    physical_size,
                    max_size,
                    rotating=True,
                    scratch_keys=scratch_keys,
                    scratch_values=scratch_values,
                )
                return _KVBlueprint(storage, max_size, int(getattr(cache, "keep", 0)))

            if isinstance(cache, PoolingCache):
                if cache.buf_kv is None or cache.buf_gate is None:
                    raise FixedKVCacheError(
                        f"{path}: pooled-cache probe did not expose its buffers"
                    )
                ratio = int(cache.ratio)
                if ratio not in (4, 128):
                    raise FixedKVCacheError(
                        f"{path}: unsupported pooled-cache ratio {ratio}"
                    )
                projection_dim = int(cache.buf_kv.shape[2])
                pooled_dim = projection_dim // 2 if ratio == 4 else projection_dim
                if pooled_dim <= 0 or (ratio == 4 and projection_dim % 2):
                    raise FixedKVCacheError(
                        f"{path}: pooled-cache projection dimension is invalid"
                    )
                pooled_capacity = context_window // ratio
                pooled_shape = (slots, pooled_capacity, pooled_dim)
                buf_kv_shape = (slots, ratio, projection_dim)
                buf_gate_shape = (
                    slots,
                    ratio,
                    int(cache.buf_gate.shape[2]),
                )
                pooled = mx.zeros(pooled_shape, dtype=cache.buf_kv.dtype)
                buf_kv = mx.zeros(buf_kv_shape, dtype=cache.buf_kv.dtype)
                buf_gate = mx.zeros(buf_gate_shape, dtype=cache.buf_gate.dtype)
                allocated.extend([pooled, buf_kv, buf_gate])
                undo_buf_kv = mx.zeros(buf_kv_shape, dtype=cache.buf_kv.dtype)
                undo_buf_gate = mx.zeros(
                    buf_gate_shape, dtype=cache.buf_gate.dtype
                )
                allocated.extend([undo_buf_kv, undo_buf_gate])
                scratch_allocated.extend([undo_buf_kv, undo_buf_gate])

                prev_win_kv = prev_win_gate = None
                if ratio == 4 and context_window >= ratio:
                    prev_kv_shape = (slots, 1, ratio, projection_dim)
                    prev_gate_shape = (
                        slots,
                        1,
                        ratio,
                        int(cache.buf_gate.shape[2]),
                    )
                    prev_win_kv = mx.zeros(prev_kv_shape, dtype=cache.buf_kv.dtype)
                    prev_win_gate = mx.zeros(
                        prev_gate_shape,
                        dtype=cache.buf_gate.dtype,
                    )
                    allocated.extend([prev_win_kv, prev_win_gate])

                return _PoolingBlueprint(
                    _PoolingStorage(
                        pooled=pooled,
                        buf_kv=buf_kv,
                        buf_gate=buf_gate,
                        ratio=ratio,
                        logical_capacity=pooled_capacity,
                        prev_win_kv=prev_win_kv,
                        prev_win_gate=prev_win_gate,
                        undo_buf_kv=undo_buf_kv,
                        undo_buf_gate=undo_buf_gate,
                        scratch_pooled=shared_scratch(
                            "pooled", pooled_shape, cache.buf_kv.dtype
                        ),
                        scratch_buf_kv=shared_scratch(
                            "buffer_kv",
                            buf_kv_shape,
                            cache.buf_kv.dtype,
                        ),
                        scratch_buf_gate=shared_scratch(
                            "buffer_gate",
                            buf_gate_shape,
                            cache.buf_gate.dtype,
                        ),
                        scratch_prev_win_kv=(
                            shared_scratch(
                                "previous_window_kv",
                                tuple(int(dim) for dim in prev_win_kv.shape),
                                prev_win_kv.dtype,
                            )
                            if prev_win_kv is not None
                            else None
                        ),
                        scratch_prev_win_gate=(
                            shared_scratch(
                                "previous_window_gate",
                                tuple(int(dim) for dim in prev_win_gate.shape),
                                prev_win_gate.dtype,
                            )
                            if prev_win_gate is not None
                            else None
                        ),
                    )
                )

            if isinstance(cache, ArraysCache):
                state_arrays: list[mx.array | None] = []
                scratch_arrays: list[mx.array | None] = []
                for value in cache.cache:
                    if value is None:
                        state_arrays.append(None)
                        scratch_arrays.append(None)
                        continue
                    shape = (slots,) + tuple(int(dim) for dim in value.shape[1:])
                    array = mx.zeros(shape, dtype=value.dtype)
                    allocated.append(array)
                    state_arrays.append(array)
                    scratch = shared_scratch(
                        f"array-state-{len(state_arrays) - 1}",
                        shape,
                        value.dtype,
                    )
                    scratch_arrays.append(scratch)
                if not any(array is not None for array in state_arrays):
                    raise FixedKVCacheError(
                        f"{path}: recurrent cache probe stayed empty"
                    )
                return _ArraysBlueprint(
                    _ArraysStorage(state_arrays, slots, scratch_arrays)
                )

            if isinstance(cache, CacheList):
                return _ListBlueprint(
                    tuple(
                        blueprint(child, f"{path}.{index}")
                        for index, child in enumerate(cache.caches)
                    )
                )

            raise FixedKVCacheError(
                f"{path}: cache layout {type(cache).__name__} has no fixed-pool adapter"
            )

        blueprints = [
            blueprint(
                cache,
                f"layer[{index}]",
                allow_empty=index in null_cache_indices,
            )
            for index, cache in enumerate(probe_cache)
        ]
        mtp_blueprints = [
            blueprint(cache, f"mtp[{index}]")
            for index, cache in enumerate(mtp_probe_cache)
        ]
        mtp_clone_blueprints = (
            [
                blueprint(cache, f"mtp_clone[{index}]")
                for index, cache in enumerate(mtp_probe_cache)
            ]
            if mtp_head_clone
            else []
        )

        # Drop the probe before materializing the reservation so the active
        # delta cannot contain both.  clear_cache only releases unused probe
        # buffers; the loaded model weights remain referenced.
        del probe_cache
        del mtp_probe_cache
        del probe_arrays
        for candidate in (
            model,
            getattr(model, "language_model", None),
            getattr(model, "_language_model", None),
        ):
            if candidate is not None:
                with suppress(AttributeError):
                    delattr(candidate, "_omlx_mtp_prime_ctx")
        mx.synchronize()
        mx.clear_cache()
        active_before = int(mx.get_active_memory())
        if allocated:
            mx.eval(*allocated)
        mx.synchronize()
        active_after = int(mx.get_active_memory())

        pool = cls(
            context_window=context_window,
            slots=slots,
            blueprints=blueprints,
            mtp_blueprints=mtp_blueprints,
            mtp_clone_blueprints=mtp_clone_blueprints,
            arrays=allocated,
            pool_scratch_bytes=sum(
                int(array.nbytes) for array in scratch_allocated
            ),
            probe_bytes=probe_bytes,
            active_memory_before=active_before,
            active_memory_after=active_after,
        )
        # MLX can differ by a handful of allocator bookkeeping bytes, but a
        # material deficit means the supposedly reserved arrays were not all
        # made resident.  Fail before the scheduler can admit a request.
        materialization_deficit = (
            active_before + pool.committed_bytes - active_after
        )
        if materialization_deficit > 64 * 1024:
            pool.close()
            mx.synchronize()
            mx.clear_cache()
            raise FixedKVCacheError(
                "Fixed KV pool did not fully materialize: requested "
                f"{pool.committed_bytes:,} bytes but MLX active memory increased "
                f"by only {pool.materialized_delta_bytes:,} bytes. Launch was "
                "aborted before serving requests."
            )
        logger.info(
            "Materialized fixed KV pool: context=%d slots=%d bytes=%d active_delta=%d",
            context_window,
            slots,
            pool.committed_bytes,
            pool.materialized_delta_bytes,
        )
        pool._install_mtp_provider(model)
        return pool

    def _install_mtp_provider(self, model: Any) -> None:
        if not self._mtp_blueprints:
            return

        def provider(
            target_cache: Any | None = None,
            *,
            scratch: bool = False,
        ) -> list[Any]:
            slot = _cache_slot(target_cache)
            if slot is None:
                if self.slots != 1:
                    raise FixedKVCacheError(
                        "Native MTP cache allocation needs the target session row"
                    )
                slot = 0
            return self.make_mtp_cache(slot, scratch=scratch)

        seen: set[int] = set()
        for owner in (
            model,
            getattr(model, "language_model", None),
            getattr(model, "_language_model", None),
        ):
            if owner is None or id(owner) in seen:
                continue
            seen.add(id(owner))
            previous = getattr(owner, "_omlx_fixed_mtp_cache_provider", None)
            self._provider_owners.append((owner, previous, provider))
            owner._omlx_fixed_mtp_cache_provider = provider

    def make_mtp_cache(self, slot: int, *, scratch: bool = False) -> list[Any]:
        blueprints = (
            self._mtp_clone_blueprints if scratch else self._mtp_blueprints
        )
        if not blueprints:
            raise FixedKVCacheError(
                "This fixed pool has no precommitted native MTP scratch cache"
                if scratch
                else "This fixed pool has no native MTP cache"
            )
        slot = int(slot)
        if slot < 0 or slot >= self.slots:
            raise FixedKVCacheCapacityError(
                f"Native MTP row {slot} is outside the {self.slots}-slot pool"
            )
        values = [blueprint.make(slot) for blueprint in blueprints]
        clone_factory = (
            (lambda: self.make_mtp_cache(slot, scratch=True))
            if self._mtp_clone_blueprints and not scratch
            else None
        )
        return _FixedMTPList(values, clone_factory)

    def make_cache(self, slot: int) -> list[Any]:
        slot = int(slot)
        if slot < 0 or slot >= self.slots:
            raise FixedKVCacheCapacityError(
                f"All {self.slots} fixed KV session slots are in use"
            )
        return [blueprint.make(slot) for blueprint in self._blueprints]

    def load_cache(self, slot: int, source: list[Any]) -> list[Any]:
        target = self.make_cache(slot)
        if len(target) != len(source):
            raise FixedKVCacheError(
                f"Restored cache has {len(source)} layers; fixed pool expects {len(target)}"
            )
        for dst, src in zip(target, source):
            self._copy_cache(dst, src)
        return target

    @classmethod
    def _copy_cache(cls, target: Any, source: Any) -> None:
        if isinstance(target, CacheList):
            source_children = getattr(source, "caches", None)
            if source_children is None or len(target.caches) != len(source_children):
                raise FixedKVCacheError(
                    "Restored CacheList layout does not match fixed pool"
                )
            for dst, src in zip(target.caches, source_children):
                cls._copy_cache(dst, src)
            return
        target.state = source.state
        with suppress(AttributeError, TypeError, ValueError):
            target.meta_state = source.meta_state

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_window": self.context_window,
            "reserved_session_slots": self.slots,
            "fixed_kv_cache_bytes": self.serving_bytes,
            "per_session_kv_bytes": self.serving_bytes // self.slots,
            "pool_scratch_bytes": self.pool_scratch_bytes,
            "committed_kv_cache_bytes": self.serving_bytes,
            "committed_pool_bytes": self.committed_bytes,
            "materialized_delta_bytes": self.materialized_delta_bytes,
            "native_mtp_kv_bytes_per_session": self.native_mtp_bytes_per_session,
            "lifecycle": "committed",
        }

    def close(self) -> None:
        for owner, previous, provider in self._provider_owners:
            if getattr(owner, "_omlx_fixed_mtp_cache_provider", None) is not provider:
                continue
            if previous is None:
                with suppress(AttributeError):
                    delattr(owner, "_omlx_fixed_mtp_cache_provider")
            else:
                owner._omlx_fixed_mtp_cache_provider = previous
        self._provider_owners.clear()
        self._blueprints.clear()
        self._mtp_blueprints.clear()
        self._mtp_clone_blueprints.clear()
        self._arrays.clear()
