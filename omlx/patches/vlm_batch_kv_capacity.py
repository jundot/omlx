# SPDX-License-Identifier: Apache-2.0
"""Reuse spare KV capacity in mlx-vlm and mlx-lm ``BatchKVCache``.

The stock classes regrow by concatenation, so every 256-token step, merge,
join and filter copies the whole bank. These methods keep the stock logical
layout (``_width``) inside a larger buffer that grows on a ratio ladder.
A filter refills freed rows from the last rows and drops common padding in
place. A later join writes into those spare rows (``_rows``), and a ragged
rollback rolls the bank in place.
Prefix restore builds mlx-lm caches even for VLM models, so both backends are
patched in place. Capacity layout adapted from PR #4030.
"""

import weakref

import mlx.core as mx
from mlx_lm.generate import GenerationBatch
from mlx_lm.models.cache import BatchKVCache as LMBatchKVCache
from mlx_lm.models.cache import KVCache as LMKVCache
from mlx_vlm.models.cache import BatchKVCache, KVCache, dynamic_roll

from omlx.custom_kernels.attention import fast as _attention_fast

# Larger shifts keep the MLX gather.
# The native roll stages this many columns per block; an MTP rollback shifts at most one verify window.
_NATIVE_ROLL_MAX_SHIFT = 16


def _native_roll(x, shifts, length: int, axis: int, max_shift: int):
    """Roll the rows of ``x`` in place when the kernel applies, else ``None``."""
    if (
        not 0 < max_shift < length
        or max_shift > _NATIVE_ROLL_MAX_SHIFT
        or x.itemsize not in (1, 2, 4)
        or mx.default_device() != mx.gpu
        or not _attention_fast.has_symbol("roll_rows")
    ):
        return None
    rows = mx.broadcast_to(shifts, x.shape[:axis]).reshape(-1).astype(mx.int32)
    return _attention_fast.roll_rows(x, rows, axis % x.ndim, length, max_shift)


def _roll_logical_prefix(x, shifts, length: int, axis: int, max_shift=None):
    """``dynamic_roll`` of the first ``length`` columns; later columns stay put.

    ``max_shift`` is the largest shift when the caller knows it on the host.
    """
    if max_shift == 0:
        return x
    if max_shift is not None:
        rolled = _native_roll(x, shifts, length, axis, max_shift)
        if rolled is not None:
            return rolled
    n = x.shape[axis]
    if n == length:
        return dynamic_roll(x, shifts, axis=axis)
    if length <= 0:
        return x
    expand_shifts = (...,) + (None,) * (x.ndim - axis)
    expand_indices = expand_shifts[:-1]
    positions = mx.arange(n)[expand_indices]
    rolled = (positions - shifts[expand_shifts]) % length
    idx = mx.where(positions < length, rolled, positions)
    return mx.take_along_axis(x, idx, axis=axis)


def _ladder_capacity(needed: int, step: int, fraction: int = 16) -> int:
    """Capacity for ``needed`` columns with spare in ``[g, 2g)``.

    ``g`` is ``1/fraction`` of the largest power of two <= ``needed``, at least one ``step``.
    Sizes are piecewise constant, so MLX's pool can reuse a released bank.
    """
    needed = max(int(needed), 1)
    grain = max(step, (1 << (needed.bit_length() - 1)) // fraction)
    grain = ((grain + step - 1) // step) * step
    return ((needed + 2 * grain - 1) // grain) * grain


# A batch bank regrow copies every row, so batch banks keep twice the spare.
# At long context that spare covers several agent turns.
_BANK_FRACTION = 8

# Moving the live rows inside the bank copies at most this much per chunk.
_MOVE_CHUNK_BYTES = 512 * 1024**2

# Banks this large grow early, one layer per synchronized pool clear, so the pool holds one old bank.
# Smaller banks grow all layers in the step that needs the columns.
_EARLY_GROWTH_MIN_BYTES = 256 * 1024**2
_early_growth = {"allowed": False, "grown": False}


def allow_early_kv_growth(allowed: bool) -> None:
    """Let the next decode step grow one large bank before it runs out of columns."""
    _early_growth["allowed"] = allowed


def take_early_kv_growth() -> bool:
    """True once after a bank grew early; its old buffer then waits in the MLX pool."""
    grown = _early_growth["grown"]
    _early_growth["grown"] = False
    return grown


def _logical_kv(cache):
    """``(keys, values)`` cut to the stock (concatenate) width."""
    keys, values = cache.keys, cache.values
    width = getattr(cache, "_width", None)
    if keys is None or width is None or width == keys.shape[2]:
        return keys, values
    return keys[..., :width, :], values[..., :width, :]


class _CapacityMethods:
    # Live rows when the bank has spare rows, else None.
    _rows = None

    @property
    def keys(self):
        if self._rows is None or self._keys is None:
            return self._keys
        return self._keys[: self._rows]

    @keys.setter
    def keys(self, value):
        self._set_bank("_keys", value)
        self._drop_spare_rows("_values")

    @property
    def values(self):
        if self._rows is None or self._values is None:
            return self._values
        return self._values[: self._rows]

    @values.setter
    def values(self, value):
        self._set_bank("_values", value)
        self._drop_spare_rows("_keys")

    def _set_bank(self, name: str, value) -> None:
        """Adopt ``value`` as the bank; a same-width set keeps the logical width.

        ``cache.keys = mx.depends(cache.keys, deps)`` writes back the whole bank,
        spare columns included, and must not turn them into logical columns.
        """
        old = getattr(self, name, None)
        if old is None or value is None or value.shape[2] != old.shape[2]:
            self._width = None
        setattr(self, name, value)

    def _drop_spare_rows(self, partner: str) -> None:
        """Cut the partner bank to the live rows when one bank is set alone.

        ``cache.keys = mx.depends(cache.keys, deps)`` sets only the keys view.
        """
        if self._rows is None:
            return
        other = getattr(self, partner)
        if other is not None:
            setattr(self, partner, other[: self._rows])
        self._rows = None

    def _logical_width(self) -> int:
        if self._width is not None:
            return self._width
        return 0 if self._keys is None else int(self._keys.shape[2])

    def _grow(self, keys, values, base: int, width: int) -> None:
        """Match stock growth: keep ``[0, base)`` and zero-extend to ``width``."""
        old_k, old_v = self.keys, self.values
        batch_size, n_kv_heads, _, k_head_dim = keys.shape
        v_head_dim = values.shape[3]
        if old_k is not None and (
            old_k.dtype != keys.dtype
            or old_v.dtype != values.dtype
            or old_k.shape[:2] != keys.shape[:2]
            or old_k.shape[3] != k_head_dim
            or old_v.shape[:2] != values.shape[:2]
            or old_v.shape[3] != v_head_dim
        ):
            # Stock concatenate promotes or raises here; keep that behavior.
            fresh = width - base
            self.keys = mx.concatenate(
                [
                    old_k[..., :base, :],
                    mx.zeros((batch_size, n_kv_heads, fresh, k_head_dim), keys.dtype),
                ],
                axis=2,
            )
            self.values = mx.concatenate(
                [
                    old_v[..., :base, :],
                    mx.zeros((batch_size, n_kv_heads, fresh, v_head_dim), values.dtype),
                ],
                axis=2,
            )
            self._width = None
            return
        capacity = 0 if old_k is None else int(old_k.shape[2])
        if width > capacity:
            capacity = _ladder_capacity(width, self.step, _BANK_FRACTION)
            new_k = mx.zeros((batch_size, n_kv_heads, capacity, k_head_dim), keys.dtype)
            new_v = mx.zeros(
                (batch_size, n_kv_heads, capacity, v_head_dim), values.dtype
            )
            if base:
                new_k[..., :base, :] = old_k[..., :base, :]
                new_v[..., :base, :] = old_v[..., :base, :]
            self._keys, self._values = new_k, new_v
            self._rows = None
        # Columns past the old width are still zero, and the append that
        # triggered growth overwrites [base, old width).
        self._width = width

    def _grow_early(self, count: int) -> None:
        """Grow a large bank whose spare columns fall under half a grain.

        A lone live row drops the spare rows; otherwise up to as many spare rows as live rows stay.
        """
        old_k, old_v = self._keys, self._values
        if old_k is None or old_k.nbytes + old_v.nbytes < _EARLY_GROWTH_MIN_BYTES:
            return
        width = self._logical_width()
        needed = max(width, self._idx + count)
        current = int(old_k.shape[2])
        grain = max(self.step, (1 << (current.bit_length() - 1)) // _BANK_FRACTION)
        if current - needed >= grain // 2:
            return
        capacity = _ladder_capacity(needed, self.step, _BANK_FRACTION)
        if capacity <= current:
            return
        live = int(self.keys.shape[0])
        rows = 1 if live == 1 else min(int(old_k.shape[0]), 2 * live)
        _, heads, _, key_dim = old_k.shape
        keys = mx.zeros((rows, heads, capacity, key_dim), old_k.dtype)
        values = mx.zeros((rows, heads, capacity, old_v.shape[3]), old_v.dtype)
        keys[:live, :, :width] = old_k[:live, :, :width]
        values[:live, :, :width] = old_v[:live, :, :width]
        self._keys, self._values = keys, values
        self._width = width
        self._rows = live if live < rows else None
        _early_growth["allowed"] = False
        _early_growth["grown"] = True

    def update_and_fetch(self, keys, values):
        if self.__dict__.get("_extracts"):
            self._settle_extracts()
        if _early_growth["allowed"] and keys.shape[2] < self.step:
            self._grow_early(keys.shape[2])
        prev = self._idx
        width = self._logical_width()
        if self._keys is None or (prev + keys.shape[2]) > width:
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            if self._keys is None:
                base = 0
            elif prev % self.step != 0:
                base = prev
            else:
                base = width
            self._grow(keys, values, base, base + n_steps * self.step)

        self.offset += keys.shape[2]
        self._idx += keys.shape[2]
        # Index the whole bank: a slice assignment into a live-row view copies it.
        rows = slice(self._rows)
        self._keys[rows, :, prev : self._idx, :] = keys
        self._values[rows, :, prev : self._idx, :] = values
        return (
            self._keys[rows, :, : self._idx, :],
            self._values[rows, :, : self._idx, :],
        )

    def finalize(self):
        if self._right_padding is not None:
            if self.keys is None:
                self._right_padding = None
                return
            padding = self._right_padding
            max_shift = max(padding.tolist())
            self._settle_extracts()
            # Roll only the stock width; the zero tail stays in place.
            # Spare rows roll too, with no shift, so the bank keeps them.
            width = self._logical_width()
            spare = int(self._keys.shape[0]) - int(padding.shape[0])
            shifts = padding
            if spare:
                shifts = mx.concatenate([padding, mx.zeros((spare,), padding.dtype)])
            self._keys = _roll_logical_prefix(
                self._keys, shifts[:, None], width, axis=2, max_shift=max_shift
            )
            self._values = _roll_logical_prefix(
                self._values, shifts[:, None], width, axis=2, max_shift=max_shift
            )
            self.offset -= padding
            self.left_padding += padding
            self._right_padding = None

    def filter(self, batch_indices):
        """Keep the given rows in place, or copy survivors once into a ladder bank."""
        kept = (
            batch_indices.tolist()
            if isinstance(batch_indices, mx.array)
            else list(batch_indices)
        )
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]
        if self._right_padding is not None and not isinstance(self, LMBatchKVCache):
            self._right_padding = self._right_padding[batch_indices]

        # Shift left to reduce padding
        min_left_pad = min(self.left_padding.min().item(), self._idx) if kept else 0
        if self._keys is not None:
            if not kept:
                self._keys = self._keys[batch_indices]
                self._values = self._values[batch_indices]
                self._rows = None
            elif not self._move_rows(kept, min_left_pad):
                self._compact_rows(kept, min_left_pad)
            elif min_left_pad:
                self._drop_left_columns(len(kept), min_left_pad)
        if min_left_pad > 0:
            self._idx -= min_left_pad
            self.left_padding -= min_left_pad

    def _move_rows(self, kept, drop: int) -> bool:
        """Refill freed rows in place from rows past ``len(kept)``.

        Freed rows stay for later joins until column growth trims them or the batch empties.
        False for other orders or when ``drop`` takes half the width, which compacts instead.
        """
        count = len(kept)
        moves = [(dst, src) for dst, src in enumerate(kept) if src != dst]
        if drop * 2 >= self._logical_width() or any(src < count for _, src in moves):
            return False
        self._settle_extracts()
        if moves:
            dst = mx.array([d for d, _ in moves])
            src = mx.array([s for _, s in moves])
            moved_k, moved_v = self._keys[src], self._values[src]
            # A pending gather of the bank would make the scatter copy it.
            mx.eval(moved_k, moved_v)
            self._keys[dst] = moved_k
            self._values[dst] = moved_v
        self._rows = count if count < int(self._keys.shape[0]) else None
        return True

    def _compact_rows(self, kept, start: int) -> None:
        """Keep rows ``kept`` and drop the first ``start`` columns."""
        old_k, old_v = self.keys, self.values
        if not start and kept == list(range(int(old_k.shape[0]))):
            return
        width = self._logical_width()
        length = width - start
        capacity = _ladder_capacity(length, self.step, _BANK_FRACTION)
        _, heads, _, key_dim = old_k.shape
        keys = mx.zeros((len(kept), heads, capacity, key_dim), old_k.dtype)
        values = mx.zeros((len(kept), heads, capacity, old_v.shape[3]), old_v.dtype)
        for row, old in enumerate(kept):
            keys[row : row + 1, :, :length] = old_k[old : old + 1, :, start:width]
            values[row : row + 1, :, :length] = old_v[old : old + 1, :, start:width]
        self._keys, self._values = keys, values
        self._width = length
        self._rows = None

    def extend(self, other):
        """In-place extend this cache with the other cache."""
        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return

        logical = {id(c): _logical_kv(c) for c in (self, other)}
        max_idx = max(self._idx, other._idx)
        length_a = length_b = 0
        if self.keys is not None:
            batch_size, heads, length_a, head_dim = logical[id(self)][0].shape
            value_dim = self.values.shape[3]
        if other.keys is not None:
            batch_size, heads, length_b, head_dim = logical[id(other)][0].shape
            value_dim = other.values.shape[3]
        max_size = max(length_a, length_b)
        empty_dtype = (
            (self.keys if self.keys is not None else other.keys).dtype
            if isinstance(self, LMBatchKVCache)
            else mx.float32
        )

        if self._extend_in_place(other, logical, max_idx, max_size):
            return
        if self._extend_into_capacity(other, logical, max_idx, max_size):
            return

        # Stock concatenate join on the logical widths.
        def pad(c):
            k, v = logical[id(c)]
            if k is None:
                rows = c.offset.shape[0]
                k = mx.zeros((rows, heads, 0, head_dim), dtype=empty_dtype)
                v = mx.zeros((rows, heads, 0, value_dim), dtype=empty_dtype)
            left = max_idx - c._idx
            right = max_size - k.shape[2] - left
            if right < 0:
                k = k[..., :right, :]
                v = v[..., :right, :]
                right = 0
            if left != 0 or right != 0:
                pad = [(0, 0), (0, 0), (left, right), (0, 0)]
                k = mx.pad(k, pad)
                v = mx.pad(v, pad)
            left_padding = c.left_padding + left
            return k, v, c.offset, left_padding

        self.keys, self.values, self.offset, self.left_padding = map(
            mx.concatenate, zip(*(pad(self), pad(other)))
        )
        self._width = None
        self._idx = max_idx

    def _extend_in_place(self, other, logical, max_idx: int, max_size: int) -> bool:
        """Write the stock join layout into the spare rows a filter left.

        When ``other`` is longer, the stock layout moves the live rows right; they
        move inside the bank. False when the rows, width or dtypes do not fit.
        """
        if self._keys is None or other.keys is None:
            return False
        keys, values = logical[id(other)]
        live, new = int(self.keys.shape[0]), int(keys.shape[0])
        for bank, part in ((self._keys, keys), (self._values, values)):
            if (
                live + new > int(bank.shape[0])
                or max_size > int(bank.shape[2])
                or part.dtype != bank.dtype
                or part.shape[1] != bank.shape[1]
                or part.shape[3] != bank.shape[3]
            ):
                return False
        # The caller's view of this bank would make every write below copy it.
        del logical[id(self)]
        self._settle_extracts()
        shift = max_idx - self._idx
        if shift:
            length = min(self._logical_width(), max_size - shift)
            self._move_columns(live, 0, shift, length)
            for bank in (self._keys, self._values):
                bank[:live, :, :shift] = 0
        left = max_idx - other._idx
        length = min(int(keys.shape[2]), max_size - left)
        rows = slice(live, live + new)
        for bank, part in ((self._keys, keys), (self._values, values)):
            # Freed rows still hold old tokens.
            bank[rows, :, :max_size] = 0
            bank[rows, :, left : left + length] = part[..., :length, :]
        self.offset = mx.concatenate([self.offset, other.offset])
        self.left_padding = mx.concatenate(
            [self.left_padding + shift, other.left_padding + left]
        )
        self._idx = max_idx
        self._width = max_size
        self._rows = live + new if live + new < int(self._keys.shape[0]) else None
        return True

    def _move_columns(self, live: int, src: int, dst: int, length: int) -> None:
        """Move live-row columns ``[src, src + length)`` to ``dst`` in the bank.

        Chunks start at the end the move leaves behind, so none reads a column a write changed.
        Each chunk is evaluated before its write, or the write copies the bank.
        """
        rows = slice(live)
        per_column = sum(
            live * bank.shape[1] * bank.shape[3] * bank.itemsize
            for bank in (self._keys, self._values)
        )
        chunk = max(1, _MOVE_CHUNK_BYTES // max(1, per_column))
        starts = range(0, length, chunk)
        for start in reversed(starts) if dst > src else starts:
            end = min(start + chunk, length)
            columns = mx.arange(src + start, src + end)
            moved = [
                mx.take(bank[rows], columns, axis=2)
                for bank in (self._keys, self._values)
            ]
            mx.eval(moved)
            for bank, part in zip((self._keys, self._values), moved):
                bank[rows, :, dst + start : dst + end] = part

    def _drop_left_columns(self, live: int, count: int) -> None:
        """Drop the first ``count`` logical columns of the live rows inside the bank."""
        width = self._logical_width()
        length = max(0, width - count)
        self._move_columns(live, count, 0, length)
        # Columns past the logical width stay zero for later growth, spare rows included.
        # A join writes a spare row only up to its own width.
        for bank in (self._keys, self._values):
            bank[:, :, length:width] = 0
        self._width = length

    def _extend_into_capacity(self, other, logical, max_idx: int, max_size: int):
        """Write the stock join layout straight into a ladder bank.

        Returns False when a side is empty or dtypes or head shapes differ,
        where stock concatenate would promote or raise.
        """
        sides = (self, other)
        banks = [logical[id(c)] for c in sides]
        if any(k is None for k, _ in banks):
            return False
        (k0, v0), (k1, v1) = banks
        if (
            k0.dtype != k1.dtype
            or v0.dtype != v1.dtype
            or k0.shape[1] != k1.shape[1]
            or k0.shape[3] != k1.shape[3]
            or v0.shape[1] != v1.shape[1]
            or v0.shape[3] != v1.shape[3]
        ):
            return False
        rows = [int(k.shape[0]) for k, _ in banks]
        # Joins keep arriving while the batch runs, so the new bank keeps the spare rows.
        total = max(sum(rows), int(self._keys.shape[0]))
        width = max_size + self.step
        capacity = _ladder_capacity(width, self.step, _BANK_FRACTION)
        keys = mx.zeros((total, k0.shape[1], capacity, k0.shape[3]), k0.dtype)
        values = mx.zeros((total, v0.shape[1], capacity, v0.shape[3]), v0.dtype)
        start = 0
        left_padding = []
        for c, (k, v), n in zip(sides, banks, rows):
            left = max_idx - c._idx
            length = min(int(k.shape[2]), max_size - left)
            keys[start : start + n, :, left : left + length] = k[..., :length, :]
            values[start : start + n, :, left : left + length] = v[..., :length, :]
            left_padding.append(c.left_padding + left)
            start += n
        self.offset = mx.concatenate([self.offset, other.offset])
        self.left_padding = mx.concatenate(left_padding)
        self._keys, self._values = keys, values
        self._width = max_size
        self._rows = start if start < total else None
        self._idx = max_idx
        return True

    @classmethod
    def merge(cls, caches):
        lengths = [c.size() for c in caches]
        max_length = max(lengths)

        # No cache has content so make an empty one
        if max_length == 0:
            return cls([0] * len(caches))

        padding = [max_length - length for length in lengths]
        batch_size = len(caches)
        heads = max(c.keys.shape[1] for c in caches if c.keys is not None)
        key_dim = max(c.keys.shape[3] for c in caches if c.keys is not None)
        value_dim = max(c.values.shape[3] for c in caches if c.values is not None)
        dt = next(iter(c.keys.dtype for c in caches if c.keys is not None))

        # Size for the merge plus its first append, on the capacity ladder.
        width = max_length + cls.step
        capacity = _ladder_capacity(width, cls.step, _BANK_FRACTION)
        keys = mx.zeros((batch_size, heads, capacity, key_dim), dtype=dt)
        values = mx.zeros((batch_size, heads, capacity, value_dim), dtype=dt)
        for i, (p, c) in enumerate(zip(padding, caches)):
            if c.keys is None:
                continue
            keys[i : i + 1, :, p : p + c.offset] = c.keys[..., : c.offset, :]
            values[i : i + 1, :, p : p + c.offset] = c.values[..., : c.offset, :]

        cache = cls(padding)
        cache.keys = keys
        cache.values = values
        cache._width = max_length
        cache.offset += max_length
        cache._idx = max_length

        return cache

    def extract(self, idx):
        padding = int(self.left_padding[idx].item())
        length = self._idx - padding
        # mlx-lm KVCache.state exposes the whole buffer, so its rows keep the
        # exact width.
        capacity = (
            length
            if isinstance(self, LMBatchKVCache)
            else _ladder_capacity(length + self.step, self.step)
        )
        cache = LMKVCache() if isinstance(self, LMBatchKVCache) else KVCache()
        cache.keys = mx.zeros(
            (1, self.keys.shape[1], capacity, self.keys.shape[3]), self.keys.dtype
        )
        cache.values = mx.zeros(
            (1, self.values.shape[1], capacity, self.values.shape[3]), self.values.dtype
        )
        cache.keys[..., :length, :] = self.keys[
            idx : idx + 1, :, padding : self._idx, :
        ]
        cache.values[..., :length, :] = self.values[
            idx : idx + 1, :, padding : self._idx, :
        ]
        cache.offset = length
        # The copy stays lazy and reads the bank until it is evaluated. Wrappers
        # such as the qwen4 QSA cache keep these arrays but drop this object.
        pending = self.__dict__.setdefault("_extracts", [])
        pending += [weakref.ref(cache.keys), weakref.ref(cache.values)]
        return cache

    def _settle_extracts(self) -> None:
        """Evaluate live extracts so in-place writes do not copy the bank.

        MLX copies a buffer instead of writing it in place while a pending read still uses it.
        mlx-lm extracts a finished row and filters at once.
        """
        pending = self.__dict__.pop("_extracts", None)
        arrays = [a for a in (ref() for ref in pending or ()) if a is not None]
        if arrays:
            mx.eval(arrays)

    @property
    def nbytes(self):
        """Bytes of the whole bank, spare rows and columns included."""
        if self._keys is None:
            return 0
        return self._keys.nbytes + self._values.nbytes


def _lm_state(cache):
    # mlx-lm serializes the whole logical bank plus the write index.
    keys, values = _logical_kv(cache)
    return keys, values, cache.offset, cache.left_padding, cache._idx


def _set_lm_state(cache, state):
    cache.keys, cache.values, cache.offset, cache.left_padding, cache._idx = state
    cache._width = None


def _refill_order(keep):
    """Order sorted ``keep`` so each freed row below ``len(keep)`` takes a later row."""
    count = len(keep)
    kept = set(keep)
    later = [row for row in keep if row >= count]
    return [row if row in kept else later.pop() for row in range(count)]


def _install_refill_order() -> None:
    """Let a finished row's slot take a later row instead of shifting every row."""
    if getattr(GenerationBatch, "_omlx_refill_order", False):
        return
    original = GenerationBatch.filter

    def filter(self, keep, *args, **kwargs):
        keep = list(keep)
        if keep == sorted(keep):
            keep = _refill_order(keep)
        return original(self, keep, *args, **kwargs)

    GenerationBatch.filter = filter
    GenerationBatch._omlx_refill_order = True


def install_capacity_methods(cls) -> None:
    """Put one ``BatchKVCache`` class on the capacity ladder (idempotent).

    Model packages that carry their own copy of the class (qwen4_exp) call
    this for it.
    """
    if getattr(cls, "_omlx_capacity_managed", False):
        return
    for name, member in vars(_CapacityMethods).items():
        if not name.startswith("__"):
            setattr(cls, name, member)
    if cls is LMBatchKVCache:
        cls.state = property(_lm_state, _set_lm_state)
    cls._omlx_capacity_managed = True


def apply_batch_kv_capacity_patch() -> bool:
    """Patch in place before cache creation; repeated installation is a no-op."""
    for cls in (BatchKVCache, LMBatchKVCache):
        install_capacity_methods(cls)
    _install_refill_order()
    return True
