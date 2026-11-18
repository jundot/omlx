# SPDX-License-Identifier: Apache-2.0
"""Defer the KV-bank realignment of a ragged MTP rollback.

When batch rows accept different draft counts, the stock rollback rolls every
lagging row over the whole logical width so all rows end at ``_idx`` again.
Here a rollback on a marked cache records a per-row *tail* instead: row ``r``
keeps its keys in ``[left_padding[r], _idx - tail[r])`` and writes its next
block at ``_idx - tail[r]``. Only the padded-row attention reads tails, so it
marks the full-attention caches it serves.

The bank is realigned (the stock roll, all tails at once) when a tail exceeds
``_MAX_TAIL`` or before anything else touches the cache: ``filter``,
``extend``, ``extract`` and outside reads of ``keys`` / ``values``.
"""

from __future__ import annotations

import contextlib
import logging
import threading

import mlx.core as mx
from mlx_lm.models.cache import BatchKVCache as LMBatchKVCache
from mlx_vlm.models.cache import BatchKVCache
from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward
from mlx_vlm.speculative import cache_state

from .vlm_batch_kv_capacity import (
    _early_growth,
    _roll_logical_prefix,
    apply_batch_kv_capacity_patch,
)

logger = logging.getLogger(__name__)

TAIL = "_omlx_tail"  # list[int] per live row
MARK = "_omlx_tail_ok"
REALIGN_ALL = "_omlx_realign_all"
_PENDING = "_omlx_pending_tail"
# Realign once a tail exceeds this, so the realignment shift stays within the
# native in-place roll (16 columns) after one more verify block.
_MAX_TAIL = 8
_state = threading.local()
_PATCHED = False


def tails(cache):
    """The cache's per-row tails when any is non-zero, else None."""
    t = getattr(cache, "__dict__", {}).get(TAIL)
    return t if t and any(t) else None


def realign(cache) -> None:
    """Roll every row right by its tail, as the stock finalize would have."""
    t = cache.__dict__.pop(TAIL, None)
    if not t or not any(t):
        return
    cache._settle_extracts()
    width = cache._logical_width()
    spare = int(cache._keys.shape[0]) - len(t)
    dtype = cache.left_padding.dtype
    shifts = mx.array(t + [0] * spare, dtype=dtype)[:, None]
    cache._keys = _roll_logical_prefix(
        cache._keys, shifts, width, axis=2, max_shift=max(t)
    )
    cache._values = _roll_logical_prefix(
        cache._values, shifts, width, axis=2, max_shift=max(t)
    )
    cache.left_padding = cache.left_padding + mx.array(t, dtype=dtype)


def realign_all(cache) -> None:
    """Realign ``cache`` now and the other layers in the next ``set_pads``."""
    realign(cache)
    cache.__dict__[REALIGN_ALL] = True


def _internal() -> bool:
    return getattr(_state, "internal", False)


@contextlib.contextmanager
def verifying():
    """Mark the batch-invariant verifier forward on this thread.

    Only that verifier sends every multi-token block through the tail-aware
    attention seam; an ordinary forward runs masked SDPA over the whole bank.
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
    original = Qwen3_5BatchInvariantForward._model

    def _model(self, *args, **kwargs):
        with verifying():
            return original(self, *args, **kwargs)

    Qwen3_5BatchInvariantForward._model = _model


def _wrap_class(cls) -> None:
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
        # The stock finalize minus the roll: offsets (RoPE positions) still drop.
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
            if _early_growth["allowed"] and length < self.step:
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


def apply_qwen35_deferred_realign_patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return False
    # Install the capacity methods first; installing them later replaces the wraps.
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
