# SPDX-License-Identifier: Apache-2.0
"""Reuse spare KV capacity in mlx-vlm and mlx-lm ``BatchKVCache``.

The stock classes regrow by concatenation, so every 256-token step, merge,
join and filter copies the whole bank. These methods keep the stock logical
layout (``_width``) inside a larger buffer that grows on a ratio ladder.
Prefix restore builds mlx-lm caches even for VLM models, so both backends are
patched in place. Capacity layout adapted from PR #4030.
"""

import mlx.core as mx
from mlx_lm.models.cache import BatchKVCache as LMBatchKVCache
from mlx_lm.models.cache import KVCache as LMKVCache
from mlx_vlm.models.cache import BatchKVCache, KVCache, dynamic_roll


def _roll_logical_prefix(x, shifts, length: int, axis: int):
    """``dynamic_roll`` of the first ``length`` columns; later columns stay put."""
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


def _ladder_capacity(needed: int, step: int) -> int:
    """Capacity for ``needed`` columns with spare in ``[g, 2g)``.

    ``g`` is 1/16 of the largest power of two <= ``needed``, at least one
    ``step``. Sizes are piecewise constant, so MLX's pool can reuse the bank
    a previous batch released.
    """
    needed = max(int(needed), 1)
    grain = max(step, 1 << max(0, needed.bit_length() - 5))
    grain = ((grain + step - 1) // step) * step
    return ((needed + 2 * grain - 1) // grain) * grain


def _logical_kv(cache):
    """``(keys, values)`` cut to the stock (concatenate) width."""
    keys, values = cache.keys, cache.values
    width = getattr(cache, "_width", None)
    if keys is None or width is None or width == keys.shape[2]:
        return keys, values
    return keys[..., :width, :], values[..., :width, :]


class _CapacityMethods:
    @property
    def keys(self):
        return self._keys

    @keys.setter
    def keys(self, value):
        self._keys = value
        self._width = None

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, value):
        self._values = value
        self._width = None

    def _logical_width(self) -> int:
        if self._width is not None:
            return self._width
        return 0 if self._keys is None else int(self._keys.shape[2])

    def _grow(self, keys, values, base: int, width: int) -> None:
        """Match stock growth: keep ``[0, base)`` and zero-extend to ``width``."""
        old_k, old_v = self._keys, self._values
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
            return
        capacity = 0 if old_k is None else int(old_k.shape[2])
        if width > capacity:
            capacity = _ladder_capacity(width, self.step)
            new_k = mx.zeros((batch_size, n_kv_heads, capacity, k_head_dim), keys.dtype)
            new_v = mx.zeros(
                (batch_size, n_kv_heads, capacity, v_head_dim), values.dtype
            )
            if base:
                new_k[..., :base, :] = old_k[..., :base, :]
                new_v[..., :base, :] = old_v[..., :base, :]
            self._keys, self._values = new_k, new_v
        # Columns past the old width are still zero, and the append that
        # triggered growth overwrites [base, old width).
        self._width = width

    def update_and_fetch(self, keys, values):
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
        self._keys[..., prev : self._idx, :] = keys
        self._values[..., prev : self._idx, :] = values
        return self._keys[..., : self._idx, :], self._values[..., : self._idx, :]

    def finalize(self):
        if self._right_padding is not None:
            if self.keys is None:
                self._right_padding = None
                return
            padding = self._right_padding
            # Roll only the stock width; the zero tail stays in place.
            width = self._logical_width()
            self._keys = _roll_logical_prefix(
                self._keys, padding[:, None], width, axis=2
            )
            self._values = _roll_logical_prefix(
                self._values, padding[:, None], width, axis=2
            )
            self.offset -= padding
            self.left_padding += padding
            self._right_padding = None

    def filter(self, batch_indices):
        """Keep the given rows, copying survivors once into a ladder bank."""
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
        min_left_pad = self.left_padding.min().item() if kept else 0
        if self._keys is not None:
            if kept:
                self._compact_rows(kept, min_left_pad)
            else:
                self._keys = self._keys[batch_indices]
                self._values = self._values[batch_indices]
        if min_left_pad > 0:
            self._idx -= min_left_pad
            self.left_padding -= min_left_pad

    def _compact_rows(self, kept, start: int) -> None:
        """Keep rows ``kept`` and drop the first ``start`` columns."""
        old_k, old_v = self._keys, self._values
        if not start and kept == list(range(int(old_k.shape[0]))):
            return
        width = self._logical_width()
        length = width - start
        capacity = _ladder_capacity(length, self.step)
        _, heads, _, key_dim = old_k.shape
        keys = mx.zeros((len(kept), heads, capacity, key_dim), old_k.dtype)
        values = mx.zeros((len(kept), heads, capacity, old_v.shape[3]), old_v.dtype)
        for row, old in enumerate(kept):
            keys[row : row + 1, :, :length] = old_k[old : old + 1, :, start:width]
            values[row : row + 1, :, :length] = old_v[old : old + 1, :, start:width]
        self._keys, self._values = keys, values
        self._width = length

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
        self._idx = max_idx

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
        width = max_size + self.step
        capacity = _ladder_capacity(width, self.step)
        keys = mx.zeros((sum(rows), k0.shape[1], capacity, k0.shape[3]), k0.dtype)
        values = mx.zeros((sum(rows), v0.shape[1], capacity, v0.shape[3]), v0.dtype)
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
        capacity = _ladder_capacity(width, cls.step)
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
        return cache


def _lm_state(cache):
    # mlx-lm serializes the whole logical bank plus the write index.
    keys, values = _logical_kv(cache)
    return keys, values, cache.offset, cache.left_padding, cache._idx


def _set_lm_state(cache, state):
    cache.keys, cache.values, cache.offset, cache.left_padding, cache._idx = state


def apply_batch_kv_capacity_patch() -> bool:
    """Patch in place before cache creation; repeated installation is a no-op."""
    for cls in (BatchKVCache, LMBatchKVCache):
        if getattr(cls, "_omlx_capacity_managed", False):
            continue
        for name, member in vars(_CapacityMethods).items():
            if not name.startswith("__"):
                setattr(cls, name, member)
        if cls is LMBatchKVCache:
            cls.state = property(_lm_state, _set_lm_state)
        cls._omlx_capacity_managed = True
    return True
