# SPDX-License-Identifier: Apache-2.0
"""PoolingCache + BatchPoolingCache from mlx-lm PR 1192.

These two cache classes are copied 1:1 from
https://github.com/Blaizzy/mlx-lm/blob/5c10538136b9038b9626c134612b08afc18d697a/mlx_lm/models/cache.py
(lines 903-1447), and injected into ``mlx_lm.models.cache`` at runtime
by patches/deepseek_v4/__init__.py so DeepSeek V4 model code can do
``from .cache import PoolingCache, BatchPoolingCache`` transparently.

When mlx-lm merges PR 1192 upstream, this file should be deleted along
with the rest of the deepseek_v4 patch directory.
"""
from typing import List

import mlx.core as mx

from mlx_lm.models.cache import _BaseCache


class PoolingCache(_BaseCache):
    """Cache for pooled (compressed) KV tokens with a remainder buffer.

    Stores two things:
      1. A growing pool of compressed tokens (step-allocated).
      2. A small remainder buffer of tokens not yet forming a full window.
    """

    def __init__(self, ratio: int):
        self.ratio = ratio

        self.buf_kv = None
        self.buf_gate = None
        self.remainder = 0

        self.pooled = None

    @property
    def offset(self):
        return 0 if self.pooled is None else self.pooled.shape[1]

    def accumulate_windows(self, kv: mx.array, gate: mx.array, offset):
        B, L, D1 = kv.shape
        _, _, D2 = gate.shape

        if self.buf_kv is None:
            self.buf_kv = mx.zeros((B, self.ratio, D1), dtype=kv.dtype)
            self.buf_gate = mx.zeros((B, self.ratio, D2), dtype=gate.dtype)

        # Prompt mode
        if L > 1:
            total = L + self.remainder
            usable = (total // self.ratio) * self.ratio
            new_remainder = total % self.ratio

            if usable > 0:
                r_kv = mx.concatenate(
                    [
                        self.buf_kv[:, : self.remainder],
                        kv[:, : (usable - self.remainder)],
                    ],
                    axis=1,
                )
                r_gate = mx.concatenate(
                    [
                        self.buf_gate[:, : self.remainder],
                        gate[:, : (usable - self.remainder)],
                    ],
                    axis=1,
                )
                r_base = offset - self.remainder
                self.remainder = 0
            else:
                r_kv = mx.zeros((B, 0, D1), dtype=kv.dtype)
                r_gate = mx.zeros((B, 0, D2), dtype=gate.dtype)
                r_base = 0

            if new_remainder > 0:
                self.buf_kv[:, self.remainder : new_remainder] = kv[:, -new_remainder:]
                self.buf_gate[:, self.remainder : new_remainder] = gate[
                    :, -new_remainder:
                ]
            self.remainder = new_remainder

            return r_kv, r_gate, r_base

        # Decode mode
        else:
            self.buf_kv[:, self.remainder : self.remainder + 1] = kv
            self.buf_gate[:, self.remainder : self.remainder + 1] = gate
            self.remainder = (self.remainder + 1) % self.ratio

            if self.remainder == 0:
                r_kv = self.buf_kv
                r_gate = self.buf_gate
                r_base = offset - self.ratio + 1
            else:
                r_kv = mx.zeros((B, 0, D1), dtype=kv.dtype)
                r_gate = mx.zeros((B, 0, D2), dtype=gate.dtype)
                r_base = 0

            return r_kv, r_gate, r_base

    def update_and_fetch(self, px: mx.array):
        if px.shape[1] == 0:
            if self.pooled is None:
                return mx.zeros((px.shape[0], 0, px.shape[-1]), dtype=px.dtype)
            return self.pooled

        if self.pooled is None:
            self.pooled = px
        else:
            self.pooled = mx.concatenate([self.pooled, px], axis=1)
        return self.pooled

    def make_mask(self, L: int = 1, offset: int = 0):
        """Build a causal validity mask for pooled positions.

        Query at absolute position ``offset + j`` can attend to pooled token
        ``i`` iff ``i < (offset + j) // ratio``.

        Returns ``(N, P)`` bool mask, or ``None`` when every pooled position
        is visible to every query (common during decode).
        """
        if self.pooled is None or L == 1:
            return None

        pool_idx = mx.arange(self.pooled.shape[1])
        query_idx = mx.arange(offset + 1, offset + L + 1)
        return pool_idx < query_idx[:, None] // self.ratio

    @property
    def state(self):
        buf_kv = self.buf_kv[:, : self.remainder] if self.remainder > 0 else None
        buf_gate = self.buf_gate[:, : self.remainder] if self.remainder > 0 else None
        return (buf_kv, buf_gate, self.pooled)

    @state.setter
    def state(self, v):
        buf_kv, buf_gate, pooled = v
        self.remainder = 0
        self.buf_kv = self.buf_gate = None
        if buf_kv is not None:
            self.accumulate_windows(buf_kv, buf_gate, 0)
        self.pooled = pooled

    @property
    def meta_state(self):
        return self.ratio

    @meta_state.setter
    def meta_state(self, v):
        self.ratio = v

    def is_trimmable(self):
        return self.pooled is None

    def trim(self, n):
        n = min(self.remainder, n)
        self.remainder -= n
        return n

    def size(self):
        return 0 if self.pooled is None else self.pooled.shape[1]

    def empty(self):
        return self.pooled is None and self.remainder == 0

    @property
    def nbytes(self):
        total = 0
        if self.buf_kv is not None:
            total += self.buf_kv.nbytes + self.buf_gate.nbytes
        if self.pooled is not None:
            total += self.pooled.nbytes
        return total

    @classmethod
    def merge(cls, caches):
        return BatchPoolingCache.merge(caches)


class BatchPoolingCache(_BaseCache):
    """Batched pooling cache with per-element variable-length tracking."""

    def __init__(self, ratio: int, left_padding: List[int]):
        self.ratio = ratio

        if not all(p == 0 for p in left_padding):
            raise RuntimeError("BatchPoolingCache does not support left padding")

        batch_size = len(left_padding)

        self.buf_kv = None
        self.buf_gate = None
        self.remainder = [0] * batch_size

        self.pooled = None
        self._pool_lengths = [0] * batch_size

        self._lengths = [2**31] * batch_size
        self._processed = [0] * batch_size

    @property
    def offset(self):
        return mx.array(self._pool_lengths, dtype=mx.int32)

    def prepare(self, *, lengths=None, right_padding=None, left_padding=None):
        if left_padding is not None:
            raise RuntimeError("BatchPoolingCache does not support left padding")
        if lengths is not None:
            self._lengths = [p + l for p, l in zip(self._processed, lengths)]

    def finalize(self):
        self._lengths = [2**31] * len(self._pool_lengths)

    def accumulate_windows(self, kv: mx.array, gate: mx.array, offset):
        B, L, D1 = kv.shape
        _, _, D2 = gate.shape
        ratio = self.ratio

        if self.buf_kv is None:
            self.buf_kv = mx.zeros((B, ratio, D1), dtype=kv.dtype)
            self.buf_gate = mx.zeros((B, ratio, D2), dtype=gate.dtype)

        valid_lengths = [min(l - p, L) for l, p in zip(self._lengths, self._processed)]
        if max(valid_lengths) != L:
            raise RuntimeError()
        for i in range(B):
            self._processed[i] += valid_lengths[i]

        totals = [vl + r for vl, r in zip(valid_lengths, self.remainder)]
        usable = [(t // ratio) * ratio for t in totals]
        max_usable = max(usable)
        new_remainder = [t % ratio for t in totals]

        # No sequence produced a full window yet
        if max_usable == 0:
            for i in range(B):
                r = self.remainder[i]
                vl = valid_lengths[i]
                self.buf_kv[i, r : r + vl] = kv[i, :vl]
                self.buf_gate[i, r : r + vl] = gate[i, :vl]
            self.remainder = new_remainder

            r_kv = mx.zeros((B, 0, D1), dtype=kv.dtype)
            r_gate = mx.zeros((B, 0, D2), dtype=gate.dtype)
            r_base = 0
            return r_kv, r_gate, r_base

        # At least one sequence completed a window
        r_kv = mx.zeros((B, max_usable, D1), dtype=kv.dtype)
        r_gate = mx.zeros((B, max_usable, D2), dtype=gate.dtype)
        r_base = [0] * B

        new_buf_kv = mx.zeros_like(self.buf_kv)
        new_buf_gate = mx.zeros_like(self.buf_gate)

        for i in range(B):
            r = self.remainder[i]
            vl = valid_lengths[i]
            u = usable[i]
            nr = new_remainder[i]

            if u > 0:
                # Tokens from the buffer (the leftover from last call)
                if r > 0:
                    r_kv[i, :r] = self.buf_kv[i, :r]
                    r_gate[i, :r] = self.buf_gate[i, :r]

                # Tokens from the new input that complete full windows
                consume = u - r
                r_kv[i, r : r + consume] = kv[i, :consume]
                r_gate[i, r : r + consume] = gate[i, :consume]

                r_base[i] = (
                    offset[i] - r if isinstance(offset, mx.array) else offset - r
                )

            # Fill new remainder buffer from the tail of the input
            if nr > 0:
                if u > 0:
                    # Old remainder was consumed into usable output;
                    # new remainder is purely from the tail of new input.
                    new_buf_kv[i, :nr] = kv[i, vl - nr : vl]
                    new_buf_gate[i, :nr] = gate[i, vl - nr : vl]
                else:
                    # No full window produced: carry over old buffer and
                    # append any new valid tokens.
                    if r > 0:
                        new_buf_kv[i, :r] = self.buf_kv[i, :r]
                        new_buf_gate[i, :r] = self.buf_gate[i, :r]
                    if vl > 0:
                        new_buf_kv[i, r : r + vl] = kv[i, :vl]
                        new_buf_gate[i, r : r + vl] = gate[i, :vl]

        self.buf_kv = new_buf_kv
        self.buf_gate = new_buf_gate
        self.remainder = new_remainder

        r_base = mx.array(r_base)
        return r_kv, r_gate, r_base

    def update_and_fetch(self, px: mx.array):
        B, N, D = px.shape

        if N == 0:
            if self.pooled is None:
                return mx.zeros((B, 0, D), dtype=px.dtype)
            return self.pooled

        # Derive how many new pooled tokens each sequence actually produced.
        new_counts = [
            (self._processed[i] - self.remainder[i]) // self.ratio
            - self._pool_lengths[i]
            for i in range(B)
        ]
        max_new = max(new_counts)
        if max_new == 0:
            if self.pooled is None:
                return mx.zeros((B, 0, D), dtype=px.dtype)
            return self.pooled

        max_pool = max(self._pool_lengths) + max_new

        if self.pooled is None:
            self.pooled = mx.zeros((B, max_pool, D), dtype=px.dtype)
        elif self.pooled.shape[1] < max_pool:
            pad = mx.zeros((B, max_pool - self.pooled.shape[1], D), dtype=px.dtype)
            self.pooled = mx.concatenate([self.pooled, pad], axis=1)

        for i in range(B):
            nc = new_counts[i]
            if nc > 0:
                pl = self._pool_lengths[i]
                self.pooled[i, pl : pl + nc] = px[i, :nc]
                self._pool_lengths[i] = pl + nc

        return self.pooled

    def make_mask(self, L: int = 1, offset=0):
        if self.pooled is None:
            return None

        B, P, _ = self.pooled.shape
        pool_lengths = mx.array(self._pool_lengths)

        # Length based mask
        pool_idx = mx.arange(P)[None, None, :]
        valid = pool_idx < pool_lengths[:, None, None]

        # Decode so no need for causal masking
        if L == 1:
            if all(pl == P for pl in self._pool_lengths):
                return None
            return valid

        # Prompt so we need to combine with causal
        if isinstance(offset, mx.array):
            query_pos = offset[:, None] + mx.arange(1, L + 1)
        else:
            query_pos = offset + mx.arange(offset + 1, offset + L + 1)[None]

        causal = pool_idx < (query_pos[..., None] // self.ratio)
        mask = causal & valid
        return mask

    @property
    def state(self):
        return (self.buf_kv, self.buf_gate, self.pooled)

    @state.setter
    def state(self, v):
        self.buf_kv, self.buf_gate, self.pooled = v

    @property
    def meta_state(self):
        return (self.ratio, self.remainder, self._pool_lengths, self._processed)

    @meta_state.setter
    def meta_state(self, v):
        self.ratio, self.remainder, self._pool_lengths, self._processed = v

    def is_trimmable(self):
        return self.pooled is None

    def trim(self, n):
        n = min(min(self.remainder), n)
        for i in range(len(self.remainder)):
            self.remainder[i] -= n
            self._processed[i] -= n
        return n

    def size(self):
        return 0 if self.pooled is None else self.pooled.shape[1]

    def empty(self):
        return self.pooled is None and all(r == 0 for r in self.remainder)

    @property
    def nbytes(self):
        total = 0
        if self.buf_kv is not None:
            total += self.buf_kv.nbytes + self.buf_gate.nbytes
        if self.pooled is not None:
            total += self.pooled.nbytes
        return total

    def filter(self, batch_indices):
        if isinstance(batch_indices, mx.array):
            idx_list = batch_indices.tolist()
        else:
            idx_list = list(batch_indices)

        if self.buf_kv is not None:
            self.buf_kv = self.buf_kv[batch_indices]
            self.buf_gate = self.buf_gate[batch_indices]
        if self.pooled is not None:
            self.pooled = self.pooled[batch_indices]

        self.remainder = [self.remainder[i] for i in idx_list]
        self._pool_lengths = [self._pool_lengths[i] for i in idx_list]
        self._lengths = [self._lengths[i] for i in idx_list]
        self._processed = [self._processed[i] for i in idx_list]

    def extend(self, other):
        # Merge the remainder buffers
        if self.buf_kv is None and other.buf_kv is None:
            pass
        elif self.buf_kv is not None and other.buf_kv is not None:
            self.buf_kv = mx.concatenate([self.buf_kv, other.buf_kv], axis=0)
            self.buf_gate = mx.concatenate([self.buf_gate, other.buf_gate], axis=0)
        elif self.buf_kv is None:
            B = len(self.remainder)
            D1 = other.buf_kv.shape[2]
            D2 = other.buf_gate.shape[2]
            self.buf_kv = mx.concatenate(
                [mx.zeros((B, self.ratio, D1), dtype=other.buf_kv.dtype), other.buf_kv],
                axis=0,
            )
            self.buf_gate = mx.concatenate(
                [
                    mx.zeros((B, self.ratio, D2), dtype=other.buf_gate.dtype),
                    other.buf_gate,
                ],
                axis=0,
            )
        else:
            B2 = len(other.remainder)
            D1 = self.buf_kv.shape[2]
            D2 = self.buf_gate.shape[2]
            self.buf_kv = mx.concatenate(
                [self.buf_kv, mx.zeros((B2, self.ratio, D1), dtype=self.buf_kv.dtype)],
                axis=0,
            )
            self.buf_gate = mx.concatenate(
                [
                    self.buf_gate,
                    mx.zeros((B2, self.ratio, D2), dtype=self.buf_gate.dtype),
                ],
                axis=0,
            )

        # Merge the pooled buffers
        if self.pooled is None and other.pooled is None:
            pass
        else:
            B1 = len(self.remainder)
            B2 = len(other.remainder)
            P1 = 0 if self.pooled is None else self.pooled.shape[1]
            P2 = 0 if other.pooled is None else other.pooled.shape[1]
            max_P = max(P1, P2)

            if max_P > 0:
                if self.pooled is not None:
                    D = self.pooled.shape[2]
                else:
                    D = other.pooled.shape[2]
                dt = (self.pooled if self.pooled is not None else other.pooled).dtype

                def pad_pool(pooled, B, P):
                    if pooled is None:
                        return mx.zeros((B, max_P, D), dtype=dt)
                    if P < max_P:
                        pad = mx.zeros((pooled.shape[0], max_P - P, D), dtype=dt)
                        return mx.concatenate([pooled, pad], axis=1)
                    return pooled

                self.pooled = mx.concatenate(
                    [pad_pool(self.pooled, B1, P1), pad_pool(other.pooled, B2, P2)],
                    axis=0,
                )

        self.remainder = self.remainder + other.remainder
        self._pool_lengths = self._pool_lengths + other._pool_lengths
        self._lengths = self._lengths + other._lengths
        self._processed = self._processed + other._processed

    def extract(self, idx):
        cache = PoolingCache(self.ratio)
        pl = self._pool_lengths[idx]
        r = self.remainder[idx]

        if self.pooled is not None and pl > 0:
            cache.pooled = mx.contiguous(self.pooled[idx : idx + 1, :pl])

        if self.buf_kv is not None and r > 0:
            cache.buf_kv = mx.contiguous(self.buf_kv[idx : idx + 1])
            cache.buf_gate = mx.contiguous(self.buf_gate[idx : idx + 1])
            cache.remainder = r

        return cache

    @classmethod
    def merge(cls, caches):
        """Merge a list of PoolingCache instances into a BatchPoolingCache."""
        B = len(caches)
        if not all(c.ratio == caches[0].ratio for c in caches):
            raise ValueError(
                "BatchPoolingCache can only merge caches with the same ratio"
            )
        ratio = caches[0].ratio
        batch_cache = cls(ratio, [0] * B)

        # Check if all caches are empty
        if all(c.empty() for c in caches):
            return batch_cache

        # Merge pooled buffers
        pool_sizes = [c.size() for c in caches]
        max_pool = max(pool_sizes)
        if max_pool > 0:
            D = next(c.pooled.shape[2] for c in caches if c.pooled is not None)
            dt = next(c.pooled.dtype for c in caches if c.pooled is not None)
            pooled = mx.zeros((B, max_pool, D), dtype=dt)
            for i, c in enumerate(caches):
                if c.pooled is not None:
                    ps = c.pooled.shape[1]
                    pooled[i, :ps] = c.pooled[0]
            batch_cache.pooled = pooled

        batch_cache._pool_lengths = pool_sizes
        batch_cache.remainder = [c.remainder for c in caches]
        batch_cache._processed = [
            c.remainder + ps * ratio for c, ps in zip(caches, pool_sizes)
        ]

        # Merge remainder buffers
        has_buf = any(c.buf_kv is not None for c in caches)
        if has_buf:
            D1 = next(c.buf_kv.shape[2] for c in caches if c.buf_kv is not None)
            D2 = next(c.buf_gate.shape[2] for c in caches if c.buf_gate is not None)
            dt = next(c.buf_kv.dtype for c in caches if c.buf_kv is not None)
            buf_kv = mx.zeros((B, ratio, D1), dtype=dt)
            buf_gate = mx.zeros((B, ratio, D2), dtype=dt)
            for i, c in enumerate(caches):
                if c.buf_kv is not None and c.remainder > 0:
                    buf_kv[i, : c.remainder] = c.buf_kv[0, : c.remainder]
                    buf_gate[i, : c.remainder] = c.buf_gate[0, : c.remainder]
            batch_cache.buf_kv = buf_kv
            batch_cache.buf_gate = buf_gate

        return batch_cache


