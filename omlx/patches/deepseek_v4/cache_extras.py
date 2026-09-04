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

        # Append-in-place pooled storage. ``_pool_buf`` is the backing
        # allocation (capacity >= logical length); ``_pool_len`` is the
        # logical row count that the old step-allocated ``pooled`` tensor
        # reported as ``pooled.shape[1]``. Appends only ever write rows at
        # [old_len, new_len), so a view captured before an append keeps
        # reading the same bytes afterwards; regrowth allocates a fresh
        # buffer and leaves any outstanding views on the old one intact.
        self._pool_buf = None
        self._pool_len = 0
        self._undo = None
        self._undo_chain = False

        # Previous completed window's raw (pre-compression) KV/gate. Only
        # used by overlap (ratio==4) compressors: decode hands a single
        # window to _overlap_compress_kv, whose `kv_a[:, :-1]` lane-A shift
        # collapses to zero-padding and drops the cross-window overlap that
        # native DS4 keeps via a rolling double buffer (ds4.c
        # compressor_decode_one). Carrying the last window here lets the
        # Compressor prepend it so decode preserves the overlap. Simple
        # (ratio==128) layers leave these None.
        self.prev_win_kv = None
        self.prev_win_gate = None

        # A verify-sized update keeps both its raw inputs and the resulting
        # pooled rows, so trim() can retain an accepted prefix even when that
        # prefix crosses a compression boundary.
        self._mtp_cross_boundary_rollback = True

    @property
    def pooled(self):
        """Logical pooled tensor: a view of the backing buffer's first
        ``_pool_len`` rows, with exactly the shape/contents the old
        step-allocated ``self.pooled`` array had.

        Long-lived consumers that must survive later appends must copy
        (see ``state`` and ``BatchPoolingCache.extract``); rows below the
        current logical length are never rewritten, so views used within
        the current chunk/step graph stay correct.
        """
        if self._pool_buf is None:
            return None
        return self._pool_buf[:, : self._pool_len]

    @pooled.setter
    def pooled(self, v):
        # Full-tensor rebinding (state restore, merge/extract targets,
        # trim restore). Capacity collapses to the logical length; the next
        # append regrows geometrically.
        if v is None:
            self._pool_buf = None
            self._pool_len = 0
        else:
            self._pool_buf = v
            self._pool_len = v.shape[1]

    @property
    def offset(self):
        return self._pool_len

    def _grow_pool(self, needed: int) -> None:
        """Ensure backing capacity for ``needed`` rows (geometric growth).

        Copies only the logical region; rows outside it were never visible
        through ``pooled``. Outstanding views keep referencing the old
        buffer, whose committed bytes stay valid.
        """
        old = self._pool_buf
        capacity = max(needed, 2 * old.shape[1])
        new = mx.zeros((old.shape[0], capacity, old.shape[2]), dtype=old.dtype)
        new[:, : self._pool_len] = old[:, : self._pool_len]
        self._pool_buf = new

    def accumulate_windows(self, kv: mx.array, gate: mx.array, offset):
        B, L, D1 = kv.shape
        _, _, D2 = gate.shape

        if self.buf_kv is None:
            self.buf_kv = mx.zeros((B, self.ratio, D1), dtype=kv.dtype)
            self.buf_gate = mx.zeros((B, self.ratio, D2), dtype=gate.dtype)

        # One-update undo log for MTP draft rejection: trim() needs the
        # pre-update state plus this update's raw inputs to undo the last
        # token when it completed a pool window. Only decode / MTP-verify
        # sized updates (L <= 8 covers depth-k chain verify windows) are
        # ever trimmed; skipping the stash for prompt chunks avoids pinning
        # large prefill projections. Buffer slices are taken before any
        # mutation, so they reference the pre-update array node.
        if L <= 8:
            try:
                from omlx.patches.mlx_lm_mtp import cache_rollback

                decode_consistent = cache_rollback._is_undo_armed() and (
                    L == 1 or cache_rollback._is_decode_consistent_armed()
                )
            except Exception:
                decode_consistent = False
            if decode_consistent and getattr(self, "_undo_chain", False):
                undo = self._undo
                self._undo = (
                    *undo[:4],
                    mx.concatenate([undo[4], kv], axis=1),
                    mx.concatenate([undo[5], gate], axis=1),
                    *undo[6:],
                )
            else:
                self._undo = (
                    self.buf_kv[:, : self.remainder] if self.remainder > 0 else None,
                    self.buf_gate[:, : self.remainder] if self.remainder > 0 else None,
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
            if self._pool_buf is None:
                return mx.zeros((px.shape[0], 0, px.shape[-1]), dtype=px.dtype)
            return self.pooled

        n = px.shape[1]
        if self._pool_buf is None:
            # First append: adopt the compressor output as the backing
            # buffer (exact fit, no copy) — identical to the old
            # ``self.pooled = px`` rebinding.
            self._pool_buf = px
            self._pool_len = n
        else:
            if self._pool_len + n > self._pool_buf.shape[1]:
                self._grow_pool(self._pool_len + n)
            self._pool_buf[:, self._pool_len : self._pool_len + n] = px
            self._pool_len += n
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
        self.remainder = 0
        self.buf_kv = self.buf_gate = None
        if buf_kv is not None:
            self.accumulate_windows(buf_kv, buf_gate, 0)
        self.pooled = pooled
        self._undo = None
        self._undo_chain = False
        self.prev_win_kv = prev_win_kv
        self.prev_win_gate = prev_win_gate

    @property
    def meta_state(self):
        return self.ratio

    @meta_state.setter
    def meta_state(self, v):
        self.ratio = v

    def is_trimmable(self):
        # Trim-by-1 contract (MTP draft rejection): possible while the last
        # token still sits in the remainder buffer, or via the one-update
        # undo log when it completed a pool window.
        if self.pooled is None or self.remainder >= 1:
            return True
        return self._can_undo(1)

    def _can_undo(self, n):
        undo = self._undo
        if undo is None:
            return False
        k = undo[4].shape[1] - n
        return k >= 0

    def trim(self, n):
        if n <= self.remainder:
            self.remainder -= n
            self._undo = None
            self._undo_chain = False
            return n
        if not self._can_undo(n):
            return 0
        buf_kv, buf_gate, rem_prev, pooled_prev, kv, gate, prev_kv, prev_gate = (
            self._undo
        )
        self._undo = None
        self._undo_chain = False
        k = kv.shape[1] - n
        prefix_kv = kv[:, :k]
        prefix_gate = gate[:, :k]
        if buf_kv is not None:
            prefix_kv = mx.concatenate([buf_kv, prefix_kv], axis=1)
            prefix_gate = mx.concatenate([buf_gate, prefix_gate], axis=1)

        completed = prefix_kv.shape[1] // self.ratio
        previous_pooled = 0 if pooled_prev is None else pooled_prev.shape[1]
        if completed == 0:
            # Restore the pre-update logical length. The appended rows were
            # only ever written at [previous_pooled, ...), so dropping the
            # logical length discards exactly what the old
            # ``self.pooled = pooled_prev`` rebinding discarded.
            if previous_pooled == 0:
                self._pool_buf = None
                self._pool_len = 0
            else:
                self._pool_len = previous_pooled
            self.prev_win_kv = prev_kv
            self.prev_win_gate = prev_gate
        else:
            # The full verify already computed these prefix windows. Keep
            # their exact rows instead of recompressing them during rollback.
            # Old code rebound ``self.pooled = pooled_after[:, :N]``; the
            # buffer's first N rows are exactly that slice.
            self._pool_len = previous_pooled + completed
            end = completed * self.ratio
            start = end - self.ratio
            self.prev_win_kv = prefix_kv[:, start:end, :][:, None]
            self.prev_win_gate = prefix_gate[:, start:end, :][:, None]

        used = completed * self.ratio
        remainder_kv = prefix_kv[:, used:]
        remainder_gate = prefix_gate[:, used:]
        self.remainder = remainder_kv.shape[1]
        if self.remainder:
            self.buf_kv[:, : self.remainder] = remainder_kv
            self.buf_gate[:, : self.remainder] = remainder_gate
        return n

    def prev_for_prepend(self):
        """Previous completed window for the Compressor to prepend, or
        ``(None, None)`` when no overlap carry is available."""
        if self.prev_win_kv is None:
            return None, None
        return self.prev_win_kv, self.prev_win_gate

    def store_prev(self, kv, gate, dropped):
        """Roll the prev window after a compression step.

        ``kv``/``gate`` are the (possibly prepended) window tensors the
        Compressor just pooled; the last window is always this sequence's
        newest completed window. ``dropped`` is unused for the single
        sequence cache, kept for signature parity with BatchPoolingCache.
        """
        self.prev_win_kv = kv[:, -1:]
        self.prev_win_gate = gate[:, -1:]

    def size(self):
        return self._pool_len

    def empty(self):
        return self._pool_buf is None and self.remainder == 0

    @property
    def nbytes(self):
        total = 0
        if self.buf_kv is not None:
            total += self.buf_kv.nbytes + self.buf_gate.nbytes
        if self._pool_buf is not None:
            # Resident allocation (capacity), not just the logical view.
            total += self._pool_buf.nbytes
        if self.prev_win_kv is not None:
            total += self.prev_win_kv.nbytes + self.prev_win_gate.nbytes
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

        # Append-in-place pooled storage (see PoolingCache). ``_pool_buf``
        # is the backing allocation; ``_pool_lengths`` (already tracked for
        # offset/mask bookkeeping) is the per-row logical length;
        # ``_pool_extent`` reproduces the physical ``pooled.shape[1]`` the
        # old code exposed, which could overshoot max(_pool_lengths) when
        # the longest row and the row completing windows differed
        # (old: max(lengths_before) + max_new).
        self._pool_buf = None
        self._pool_extent = 0
        self._pool_lengths = [0] * batch_size

        self._lengths = [2**31] * batch_size
        self._processed = [0] * batch_size
        self._undo = None
        self._undo_chain = False
        self._mtp_cross_boundary_rollback = batch_size == 1

        # Previous completed window's raw KV/gate per batch row, for overlap
        # (ratio==4) compressors — see PoolingCache.prev_win_kv docstring.
        # Rows complete windows at different steps (their window phase is
        # their token count mod ratio), so the carry is tracked per row:
        # _prev_valid[i] says whether row i's slot holds a real window.
        # Invalid rows are masked with -inf gates at prepend time, which
        # reproduces the kernel's own zero lane-A masking exactly.
        self.prev_win_kv = None
        self.prev_win_gate = None
        self._prev_valid = [False] * batch_size
        self._last_usable = [0] * batch_size

    @property
    def pooled(self):
        """Logical pooled tensor: view of the backing buffer's first
        ``_pool_extent`` columns, matching the old physical tensor's
        shape/contents exactly (per-row validity lives in
        ``_pool_lengths`` and is applied via ``make_mask``).

        Rows/columns below the current extent are never rewritten, so
        views used inside the current chunk/step graph stay correct;
        long-lived consumers must copy.
        """
        if self._pool_buf is None:
            return None
        return self._pool_buf[:, : self._pool_extent]

    @pooled.setter
    def pooled(self, v):
        # Full-tensor rebinding (state restore, filter/extend/merge).
        if v is None:
            self._pool_buf = None
            self._pool_extent = 0
        else:
            self._pool_buf = v
            self._pool_extent = v.shape[1]

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

        # One-update undo log for MTP draft rejection (see PoolingCache).
        # The buffer references are only consulted when a window completed,
        # in which case this method rebinds self.buf_* to fresh arrays and
        # the stashed objects keep the pre-update contents. The pooled
        # tensor needs no snapshot: update_and_fetch only writes beyond the
        # old _pool_lengths.  trim() drops that speculative physical tail
        # after restoring the logical lengths.
        if L <= 8:
            try:
                from omlx.patches.mlx_lm_mtp import cache_rollback

                decode_consistent = cache_rollback._is_undo_armed() and (
                    L == 1 or cache_rollback._is_decode_consistent_armed()
                )
            except Exception:
                decode_consistent = False
            if decode_consistent and getattr(self, "_undo_chain", False):
                undo = self._undo
                self._undo = (
                    *undo[:5],
                    mx.concatenate([undo[5], kv], axis=1),
                    mx.concatenate([undo[6], gate], axis=1),
                    *undo[7:],
                )
            else:
                self._undo = (
                    self.buf_kv + 0 if decode_consistent else self.buf_kv,
                    self.buf_gate + 0 if decode_consistent else self.buf_gate,
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

        valid_lengths = [min(l - p, L) for l, p in zip(self._lengths, self._processed)]
        if max(valid_lengths) != L:
            raise RuntimeError()
        for i in range(B):
            self._processed[i] += valid_lengths[i]

        totals = [vl + r for vl, r in zip(valid_lengths, self.remainder)]
        usable = [(t // ratio) * ratio for t in totals]
        max_usable = max(usable)
        new_remainder = [t % ratio for t in totals]
        # Consumed by store_prev right after this step's compression to
        # locate each row's newest real window inside the ready tensor.
        self._last_usable = usable

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
            if self._pool_buf is None:
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
            if self._pool_buf is None:
                return mx.zeros((B, 0, D), dtype=px.dtype)
            return self.pooled

        # Physical extent exactly as the old code computed it, including the
        # overshoot when the longest row is not the one completing windows.
        max_pool = max(self._pool_lengths) + max_new

        if self._pool_buf is None:
            self._pool_buf = mx.zeros((B, max_pool, D), dtype=px.dtype)
        elif self._pool_buf.shape[1] < max_pool:
            # Geometric regrowth; copy only the visible region so columns
            # beyond the extent stay zero-filled like the old pad path.
            capacity = max(max_pool, 2 * self._pool_buf.shape[1])
            new_buf = mx.zeros((B, capacity, D), dtype=px.dtype)
            new_buf[:, : self._pool_extent] = self._pool_buf[:, : self._pool_extent]
            self._pool_buf = new_buf
        self._pool_extent = max(self._pool_extent, max_pool)

        # Append in place. The old singleton path rebound
        # ``self.pooled = concatenate([self.pooled[:, :current], new_rows])``;
        # writing rows [pl, pl+nc) into the buffer is value-identical, and
        # the B > 1 path already used exactly this in-place scheme.
        for i in range(B):
            nc = new_counts[i]
            if nc > 0:
                pl = self._pool_lengths[i]
                self._pool_buf[i, pl : pl + nc] = px[i, :nc]
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
        self._undo = None
        self._undo_chain = False
        # prev_win is runtime-only and not part of the persisted state (see
        # PoolingCache.state); the first window completed after a restore
        # pools with a zero lane-A once, then the carry repopulates.
        self.prev_win_kv = None
        self.prev_win_gate = None
        self._prev_valid = [False] * len(self.remainder)

    @property
    def meta_state(self):
        return (self.ratio, self.remainder, self._pool_lengths, self._processed)

    @meta_state.setter
    def meta_state(self, v):
        self.ratio, self.remainder, self._pool_lengths, self._processed = v
        # Restore order between state and meta_state is not fixed; reset in
        # both so _prev_valid always matches the restored batch size.
        self.prev_win_kv = None
        self.prev_win_gate = None
        self._prev_valid = [False] * len(self.remainder)

    def is_trimmable(self):
        # Trim-by-1 contract (MTP draft rejection): possible while every
        # row's last token still sits in the remainder buffer, or via the
        # one-update undo log when a row completed a pool window.
        if self.pooled is None or min(self.remainder) >= 1:
            return True
        return self._can_undo(1)

    def _can_undo(self, n):
        undo = self._undo
        if undo is None:
            return False
        k = undo[5].shape[1] - n
        if self._mtp_cross_boundary_rollback:
            return k >= 0
        # The replayed confirmed prefix must stay inside the buffer for
        # every row (a replay that pools again cannot be reconstructed).
        return k >= 0 and all(r + k < self.ratio for r in undo[2])

    def trim(self, n):
        if n <= min(self.remainder):
            for i in range(len(self.remainder)):
                self.remainder[i] -= n
                self._processed[i] -= n
            self._truncate_pooled_tail()
            self._undo = None
            self._undo_chain = False
            return n
        if not self._can_undo(n):
            return 0
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
        ) = self._undo
        if self._mtp_cross_boundary_rollback:
            self._undo = None
            self._undo_chain = False
            k = kv.shape[1] - n
            prefix_kv = mx.concatenate(
                [buf_kv[:, : remainder[0]], kv[:, :k]],
                axis=1,
            )
            prefix_gate = mx.concatenate(
                [buf_gate[:, : remainder[0]], gate[:, :k]],
                axis=1,
            )
            completed = prefix_kv.shape[1] // self.ratio
            next_pool_length = pool_lengths[0] + completed
            # Old code rebound ``self.pooled = pooled_after[:, :N]``; the
            # buffer's first N columns are exactly that slice (appends only
            # wrote at [pool_lengths[0], ...)).
            if next_pool_length:
                self._pool_extent = next_pool_length
            else:
                self._pool_buf = None
                self._pool_extent = 0
            self._pool_lengths = [next_pool_length]
            self._processed = [processed[0] + k]

            used = completed * self.ratio
            remainder_kv = prefix_kv[:, used:]
            remainder_gate = prefix_gate[:, used:]
            self.remainder = [remainder_kv.shape[1]]
            self.buf_kv = buf_kv
            self.buf_gate = buf_gate
            if self.remainder[0]:
                self.buf_kv[:, : self.remainder[0]] = remainder_kv
                self.buf_gate[:, : self.remainder[0]] = remainder_gate

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
        decode_consistent = getattr(self, "_undo_chain", False)
        self._undo_chain = False
        k = kv.shape[1] - n
        # The undo path only triggers when some row completed a window,
        # which rebinds self.buf_* to fresh arrays — the stashed objects
        # still hold the pre-update contents.
        self.buf_kv = buf_kv
        self.buf_gate = buf_gate
        self.remainder = list(remainder)
        self._pool_lengths = list(pool_lengths)
        self._processed = list(processed)
        self._truncate_pooled_tail()
        self.prev_win_kv = prev_kv
        self.prev_win_gate = prev_gate
        self._prev_valid = list(prev_valid)
        if k > 0:
            # Replay the confirmed prefix; _can_undo guarantees it stays in
            # the buffer, so no window is recompressed.
            if decode_consistent:
                for idx in range(k):
                    self.accumulate_windows(
                        kv[:, idx : idx + 1], gate[:, idx : idx + 1], 0
                    )
            else:
                self.accumulate_windows(kv[:, :k], gate[:, :k], 0)
            self._undo = None
            self._undo_chain = False
        return n

    def _truncate_pooled_tail(self):
        """Drop pooled rows written by a rejected speculative suffix."""
        if self._pool_buf is None:
            return
        logical_size = max(self._pool_lengths, default=0)
        if self._pool_extent > logical_size:
            # Logical truncation only; the buffer's first logical_size
            # columns are exactly the old ``pooled[:, :logical_size]``.
            self._pool_extent = logical_size

    def prev_for_prepend(self):
        """Per-row previous window with invalid rows masked via -inf gates.

        A row that has not completed a window since its carry was reset
        must pool with a zero lane-A exactly like the kernel's own
        first-window padding, so its prepended gate is forced to -inf
        (softmax weight 0) instead of leaking zero-filled or stale data
        at finite gate values.
        """
        if self.prev_win_kv is None or not any(self._prev_valid):
            return None, None
        if all(self._prev_valid):
            return self.prev_win_kv, self.prev_win_gate
        mask = mx.array(self._prev_valid).reshape(-1, 1, 1, 1)
        gate = mx.where(mask, self.prev_win_gate, -mx.inf)
        return self.prev_win_kv, gate

    def store_prev(self, kv, gate, dropped):
        """Roll the per-row prev window after a compression step.

        ``kv``/``gate`` hold ``dropped`` prepended window(s) followed by
        this step's ready windows, where row ``i`` contributed
        ``self._last_usable[i] // ratio`` real windows left-aligned and
        zero-filled up to the batch max. Rows that completed a window
        advance to their newest real window; rows that did not keep their
        old carry (slot 0 when prepended) and stay masked via
        ``_prev_valid``.
        """
        n_new = [u // self.ratio for u in self._last_usable]
        idx = [dropped + n - 1 if n > 0 else 0 for n in n_new]
        take = mx.array(idx, dtype=mx.int32).reshape(-1, 1, 1, 1)
        self.prev_win_kv = mx.take_along_axis(kv, take, axis=1)
        self.prev_win_gate = mx.take_along_axis(gate, take, axis=1)
        self._prev_valid = [v or n > 0 for v, n in zip(self._prev_valid, n_new)]

    def size(self):
        return 0 if self._pool_buf is None else self._pool_extent

    def empty(self):
        return self._pool_buf is None and all(r == 0 for r in self.remainder)

    @property
    def nbytes(self):
        total = 0
        if self.buf_kv is not None:
            total += self.buf_kv.nbytes + self.buf_gate.nbytes
        if self._pool_buf is not None:
            # Resident allocation (capacity), not just the logical view.
            total += self._pool_buf.nbytes
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
        if self.prev_win_kv is not None:
            self.prev_win_kv = self.prev_win_kv[batch_indices]
            self.prev_win_gate = self.prev_win_gate[batch_indices]
        self._prev_valid = [self._prev_valid[i] for i in idx_list]

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

        # Merge the prev-window carries; rows on a side without one stay
        # invalid and get -inf masked at prepend time.
        if self.prev_win_kv is not None or other.prev_win_kv is not None:
            B1 = len(self.remainder)
            B2 = len(other.remainder)
            ref_kv = (
                self.prev_win_kv
                if self.prev_win_kv is not None
                else other.prev_win_kv
            )
            ref_gate = (
                self.prev_win_gate
                if self.prev_win_gate is not None
                else other.prev_win_gate
            )

            def pad_prev(arr, ref, B):
                if arr is None:
                    return mx.zeros((B,) + ref.shape[1:], dtype=ref.dtype)
                return arr

            self.prev_win_kv = mx.concatenate(
                [
                    pad_prev(self.prev_win_kv, ref_kv, B1),
                    pad_prev(other.prev_win_kv, ref_kv, B2),
                ],
                axis=0,
            )
            self.prev_win_gate = mx.concatenate(
                [
                    pad_prev(self.prev_win_gate, ref_gate, B1),
                    pad_prev(other.prev_win_gate, ref_gate, B2),
                ],
                axis=0,
            )

        self.remainder = self.remainder + other.remainder
        self._pool_lengths = self._pool_lengths + other._pool_lengths
        self._lengths = self._lengths + other._lengths
        self._processed = self._processed + other._processed
        self._prev_valid = self._prev_valid + other._prev_valid

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

        if self.prev_win_kv is not None and self._prev_valid[idx]:
            cache.prev_win_kv = mx.contiguous(self.prev_win_kv[idx : idx + 1])
            cache.prev_win_gate = mx.contiguous(self.prev_win_gate[idx : idx + 1])

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

        # Carry prev windows from members that have one
        if any(c.prev_win_kv is not None for c in caches):
            ref_kv = next(c.prev_win_kv for c in caches if c.prev_win_kv is not None)
            ref_gate = next(
                c.prev_win_gate for c in caches if c.prev_win_gate is not None
            )
            prev_kv = mx.zeros((B,) + ref_kv.shape[1:], dtype=ref_kv.dtype)
            prev_gate = mx.zeros((B,) + ref_gate.shape[1:], dtype=ref_gate.dtype)
            for i, c in enumerate(caches):
                if c.prev_win_kv is not None:
                    prev_kv[i : i + 1] = c.prev_win_kv
                    prev_gate[i : i + 1] = c.prev_win_gate
            batch_cache.prev_win_kv = prev_kv
            batch_cache.prev_win_gate = prev_gate
            batch_cache._prev_valid = [c.prev_win_kv is not None for c in caches]

        return batch_cache
