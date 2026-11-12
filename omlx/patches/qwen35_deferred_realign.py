# SPDX-License-Identifier: Apache-2.0
"""Defer the KV-bank realignment of a ragged MTP rollback (opt-in).

When the rows of a batch accept different draft counts, mlx-vlm's speculative
rollback trims to the longest row and ``finalize()`` rolls every lagging row
over the whole logical width so all rows end at ``_idx`` again. At long
context that roll costs about 3.6 ms per full-attention layer and cycle
(24.7K + 100.7K on an M5 Max), which a lone request never pays.

Here a ragged rollback inside an MTP transaction records a per-row *tail*
instead: row ``r`` keeps its keys in ``[left_padding[r], _idx - tail[r])``
and the next block of row ``r`` is written at ``_idx - tail[r]``, so each row
stays contiguous and ``_idx`` still advances by exactly the block size (the
transaction checks that). The bank is realigned (the same roll as before, all
tails at once) only when a tail exceeds ``_MAX_TAIL`` or right before anything
else touches the cache: ``filter``, ``extend``, ``merge``, ``extract``,
``state`` and external reads of ``keys`` / ``values``. Everything outside the
steady MTP loop therefore sees the stock layout.

Only caches a Qwen3.5 forward has marked (full-attention caches seen by the
padded-row attention hook) defer, because only that attention knows tails:
while a cache has tails, every attention call on it goes per row over
``[pad, _idx - tail)``. Enabled only with ``OMLX_DEFER_REALIGN=1``.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading

import mlx.core as mx

logger = logging.getLogger(__name__)

TAIL = "_omlx_tail"  # list[int] per live row
MARK = "_omlx_tail_ok"  # set on full-attention caches of a Qwen3.5 forward
_PENDING = "_omlx_pending_tail"
# Realign before any tail exceeds this, so every realignment shift stays
# within the native in-place roll (16 columns) after one more verify block.
_MAX_TAIL = 8
_state = threading.local()
_PATCHED = False


def enabled() -> bool:
    return os.environ.get("OMLX_DEFER_REALIGN") == "1"


def tails(cache):
    """The cache's per-row tails when any is non-zero, else None."""
    t = getattr(cache, "__dict__", {}).get(TAIL)
    return t if t and any(t) else None


def realign(cache) -> None:
    """Roll every row right by its tail (the stock realignment), clear tails.

    The new banks and padding are built first and committed together; the
    tail record is cleared only after that, so a failure leaves the cache
    still described by its tails (fail closed).
    """
    t = cache.__dict__.get(TAIL)
    if not t or not any(t):
        cache.__dict__.pop(TAIL, None)
        return
    from .vlm_batch_kv_capacity import _roll_logical_prefix

    _state.internal = True
    try:
        cache._settle_extracts()
        width = cache._logical_width()
        spare = int(cache._keys.shape[0]) - len(t)
        shifts = mx.array(list(t) + [0] * spare, dtype=cache.left_padding.dtype)
        keys = _roll_logical_prefix(
            cache._keys, shifts[:, None], width, axis=2, max_shift=max(t)
        )
        values = _roll_logical_prefix(
            cache._values, shifts[:, None], width, axis=2, max_shift=max(t)
        )
        left = cache.left_padding + mx.array(list(t), dtype=cache.left_padding.dtype)
        # Evaluate before committing, so a failing roll raises here with the
        # tails still recorded. Realignment is rare (a tail above _MAX_TAIL or
        # a foreign read), so the sync is cheap. Limit: a native in-place roll
        # that fails midway on the GPU leaves the bank unrecoverable in any
        # layout.
        mx.eval(keys, values, left)
        cache._keys, cache._values, cache.left_padding = keys, values, left
        cache.__dict__.pop(TAIL, None)
    finally:
        _state.internal = False


def request_realign_all(cache) -> None:
    """Ask the next ``_set_qwen3_5_decode_left_padding`` to realign every layer."""
    cache.__dict__["_omlx_realign_all"] = True


def _internal() -> bool:
    return getattr(_state, "internal", False)


@contextlib.contextmanager
def verifying():
    """Mark a speculative verify forward on this thread.

    Only mlx-vlm's batch-invariant verifier routes every multi-token block
    through the tail-aware attention seam; an ordinary model forward runs the
    stock masked SDPA over the whole bank for any non-decode mask. So tails
    may serve a multi-token forward only inside this context: an ordinary
    2-9 token forward would read a lagging row's stale columns.
    """
    prev = getattr(_state, "verify", False)
    _state.verify = True
    try:
        yield
    finally:
        _state.verify = prev


def in_verify() -> bool:
    return getattr(_state, "verify", False)


def _mark_verifier() -> None:
    """Run the Qwen3.5 batch-invariant verifier forward inside ``verifying``.

    If the class is missing, nothing is marked and every multi-token forward
    realigns first: correct, just without the deferral.
    """
    try:
        from mlx_vlm.models.qwen3_5.speculative_verifier import (
            Qwen3_5BatchInvariantForward as verifier,
        )
    except ImportError:
        logger.warning(
            "deferred realignment: verifier not found, verify forwards realign"
        )
        return
    original = verifier._model
    if getattr(original, "_omlx_verify_marked", False):
        return

    def _model(self, *args, **kwargs):
        with verifying():
            return original(self, *args, **kwargs)

    _model._omlx_verify_marked = True
    verifier._model = _model


def _wrap_class(cls) -> None:
    if getattr(cls, "_omlx_deferred_realign", False):
        return
    from . import vlm_batch_kv_capacity as cap

    keys_prop, values_prop = cls.keys, cls.values
    orig_prepare, orig_finalize = cls.prepare, cls.finalize
    orig_update = cls.update_and_fetch

    def _guarded_prop(prop):
        def getter(self):
            if not _internal() and tails(self):
                realign(self)
            return prop.fget(self)

        def setter(self, value):
            if not _internal() and tails(self):
                realign(self)
            prop.fset(self, value)

        return property(getter, setter)

    cls.keys = _guarded_prop(keys_prop)
    cls.values = _guarded_prop(values_prop)

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if (
            right_padding is not None
            and getattr(_state, "defer", False)
            and self.__dict__.get(MARK)
            and max(right_padding) > 0
            and left_padding is None
        ):
            self.__dict__[_PENDING] = [int(p) for p in right_padding]
        return orig_prepare(
            self,
            left_padding=left_padding,
            lengths=lengths,
            right_padding=right_padding,
        )

    def finalize(self):
        pending = self.__dict__.pop(_PENDING, None)
        if pending is None or self._right_padding is None or self._keys is None:
            return orig_finalize(self)
        # The stock finalize: offset -= padding; roll; left_padding += padding.
        # Keep the offsets (RoPE positions), record the tail, skip the roll.
        self.offset = self.offset - self._right_padding
        self._right_padding = None
        prior = self.__dict__.get(TAIL) or [0] * len(pending)
        self.__dict__[TAIL] = [a + b for a, b in zip(prior, pending)]
        if max(self.__dict__[TAIL]) > _MAX_TAIL:
            realign(self)

    def update_and_fetch(self, keys, values):
        t = tails(self)
        if t is None:
            return orig_update(self, keys, values)
        if len(t) != int(keys.shape[0]):
            realign(self)
            return orig_update(self, keys, values)
        # The capacity update with each row written at its own end. Growth
        # keeps columns [0, base) in place, so tails survive it.
        _state.internal = True
        try:
            if self.__dict__.get("_extracts"):
                self._settle_extracts()
            length = int(keys.shape[2])
            if cap._early_growth["allowed"] and length < self.step:
                self._grow_early(length)
            prev = self._idx
            width = self._logical_width()
            if self._keys is None or (prev + length) > width:
                n_steps = (self.step + length - 1) // self.step
                if self._keys is None:
                    base = 0
                elif prev % self.step != 0:
                    base = prev
                else:
                    base = width
                self._grow(keys, values, base, base + n_steps * self.step)
            self.offset += length
            self._idx += length
            for row, tail in enumerate(t):
                start = prev - tail
                self._keys[row : row + 1, :, start : start + length, :] = keys[
                    row : row + 1
                ]
                self._values[row : row + 1, :, start : start + length, :] = values[
                    row : row + 1
                ]
            rows = slice(self._rows)
            return (
                self._keys[rows, :, : self._idx, :],
                self._values[rows, :, : self._idx, :],
            )
        finally:
            _state.internal = False

    cls.prepare = prepare
    cls.finalize = finalize
    cls.update_and_fetch = update_and_fetch

    for name in ("filter", "extend", "extract"):
        orig = getattr(cls, name)

        def foreign(self, *args, _orig=orig, **kwargs):
            realign(self)
            return _orig(self, *args, **kwargs)

        setattr(cls, name, foreign)

    orig_merge = cls.__dict__["merge"].__func__

    def merge(klass, caches):
        for c in caches:
            realign(c)
        return orig_merge(klass, caches)

    cls.merge = classmethod(merge)
    cls._omlx_deferred_realign = True


def apply_qwen35_deferred_realign_patch() -> bool:
    global _PATCHED
    if _PATCHED or not enabled():
        return False
    try:
        from mlx_lm.models.cache import BatchKVCache as LMBatchKVCache
        from mlx_vlm.models.cache import BatchKVCache
        from mlx_vlm.speculative import cache_state

        from .vlm_batch_kv_capacity import apply_batch_kv_capacity_patch
    except ImportError:
        return False
    apply_batch_kv_capacity_patch()
    for cls in (BatchKVCache, LMBatchKVCache):
        _wrap_class(cls)

    orig_trim = cache_state._trim_append_caches

    def trim_append_caches(*args, **kwargs):
        # Only a speculative rollback defers; prefill right padding still rolls.
        # It reads ``cache.keys`` itself, which must not realign here.
        _state.defer = True
        _state.internal = True
        try:
            return orig_trim(*args, **kwargs)
        finally:
            _state.defer = False
            _state.internal = False

    cache_state._trim_append_caches = trim_append_caches
    _mark_verifier()
    _PATCHED = True
    logger.info("qwen3_5 deferred ragged-rollback realignment patch applied")
    return True
