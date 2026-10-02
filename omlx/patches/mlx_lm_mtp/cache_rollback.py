# SPDX-License-Identifier: Apache-2.0
"""MTP rollback support for mlx-lm cache classes.

Two pieces:

1. ``rollback_state`` slot on ``ArraysCache`` (PR 990): GatedDeltaNet
   snapshots ``(conv_state, ssm_state)`` after the confirmed prefix of an
   MTP draft+verify forward, then restores it when the draft is rejected.

2. A verify-block undo log on ``RotatingKVCache`` / ``BatchRotatingKVCache``:
   a rotated rotating cache is not trimmable (the slot of the evicted token
   has been overwritten), so an MTP draft rejection could not discard the
   rejected positions exactly and phantom tokens progressively corrupted
   output (hit on DeepSeek-V4-Flash, sliding_window=128). A multi-token verify
   update takes the
   ``_update_concat`` path, which only rebinds ``keys``/``values`` (no
   in-place setitem), so stashing the pre-update attribute references plus
   the update's inputs gives an exact undo. ``trim(rejected)`` then restores
   the snapshot and replays the confirmed/accepted prefix. Stashing is armed
   only around MTP-managed backbone forwards (``batch_generator._call_backbone``)
   so non-MTP flows keep stock trim semantics.
"""

from __future__ import annotations

import logging
import sys
import threading

logger = logging.getLogger(__name__)

# Thread-local so concurrent engine steps of other models never see the
# flag armed by an MTP verify forward running on a different engine thread.
_UNDO_ARMED = threading.local()


def set_undo_armed(flag: bool) -> None:
    """Arm/disarm the rotating-cache undo stash (MTP backbone forwards only)."""
    _UNDO_ARMED.value = bool(flag)


def _is_undo_armed() -> bool:
    # Resolve the flag through sys.modules: the wrapped methods live on
    # foreign (mlx-lm) classes and outlive this module object if it is ever
    # re-imported (e.g. tests that patch.dict sys.modules), so a closure
    # over this instance's _UNDO_ARMED could go stale.
    mod = sys.modules.get(__name__)
    armed = getattr(mod, "_UNDO_ARMED", None) if mod is not None else None
    if armed is None:
        armed = _UNDO_ARMED
    return getattr(armed, "value", False)


def _is_decode_consistent_armed() -> bool:
    try:
        from omlx.patches.deepseek_v4.decode_consistency import is_armed

        return is_armed()
    except Exception:
        return False


def stage_functional_rotating_update(self, keys, values) -> None:
    """Stage undo for a rotating-cache update that only rebinds arrays.

    DeepSeek's decode-consistent verify builds its physical-ring snapshots
    from the pre-update cache and then rebinds the cache to the final one.
    The old array wrappers are never mutated, so retaining their references
    is sufficient and avoids the defensive full-cache copy used for setitem.
    """
    if not _is_undo_armed():
        self._mtp_undo = None
        return
    fields = ("keys", "values", "offset", "_idx")
    fields += tuple(
        field
        for field in ("_offset", "rotated", "left_padding")
        if hasattr(self, field)
    )
    snap = {field: getattr(self, field) for field in fields}
    self._mtp_undo = (snap, keys, values, True)


def _wrap_rotating(cls, fields) -> None:
    """Wrap update_and_fetch / is_trimmable / trim with the MTP undo log."""
    if getattr(cls, "_omlx_mtp_undo_attached", False):
        return

    import mlx.core as mx

    orig_update = cls.update_and_fetch
    orig_is_trimmable = cls.is_trimmable
    orig_trim = cls.trim

    def update_and_fetch(self, keys, values):
        # DSpark's decode-consistent verify advances the cache with several
        # M=1 updates; a depth-k Lightning verify uses one update containing the
        # confirmed token plus every draft token (currently up to M=9 for
        # depth-8 drafting). Preserve one pre-block snapshot for both forms
        # so rejection can restore exactly. Undo is scoped by _call_backbone;
        # record every non-empty armed update instead of duplicating the
        # supported draft-depth limit here.
        steps = keys.shape[2]
        armed = _is_undo_armed()
        existing = getattr(self, "_mtp_undo", None)
        decode_consistent = steps == 1 or _is_decode_consistent_armed()
        chained = decode_consistent and armed and existing is not None and existing[3]
        if chained:
            snap, old_keys, old_values, _ = existing
            self._mtp_undo = (
                snap,
                mx.concatenate([old_keys, keys], axis=2),
                mx.concatenate([old_values, values], axis=2),
                True,
            )
        elif armed and steps >= 1:
            snap = {}
            for f in fields:
                v = getattr(self, f)
                if isinstance(v, mx.array):
                    # Detach: += / setitem mutate the same wrapper object,
                    # so a plain reference would see the post-update value.
                    v = v + 0
                snap[f] = v
            self._mtp_undo = (snap, keys, values, decode_consistent)
        else:
            self._mtp_undo = None
        return orig_update(self, keys, values)

    def is_trimmable(self):
        if orig_is_trimmable(self):
            return True
        return getattr(self, "_mtp_undo", None) is not None

    def trim(self, n):
        if orig_is_trimmable(self):
            self._mtp_undo = None
            return orig_trim(self, n)
        undo = getattr(self, "_mtp_undo", None)
        self._mtp_undo = None
        if undo is None:
            return 0
        snap, keys, values, decode_consistent = undo
        k = keys.shape[2] - n
        if k < 0:
            return 0
        for f, v in snap.items():
            setattr(self, f, v)
        if k > 0:
            # A decode-consistent verify must also leave the committed cache
            # in the same physical ring layout as k ordinary M=1 updates.
            if decode_consistent:
                for idx in range(k):
                    orig_update(
                        self,
                        keys[..., idx : idx + 1, :],
                        values[..., idx : idx + 1, :],
                    )
            else:
                orig_update(self, keys[..., :k, :], values[..., :k, :])
            self._mtp_undo = None
        return n

    cls.update_and_fetch = update_and_fetch
    cls.is_trimmable = is_trimmable
    cls.trim = trim
    cls._mtp_undo = None
    cls._omlx_mtp_undo_attached = True


def _attach_rotating_undo() -> bool:
    try:
        from mlx_lm.models.cache import BatchRotatingKVCache, RotatingKVCache
    except ImportError:
        logger.debug("mlx_lm.models.cache not importable; skipping rotating undo")
        return False

    _wrap_rotating(RotatingKVCache, ("keys", "values", "offset", "_idx"))
    _wrap_rotating(
        BatchRotatingKVCache,
        ("keys", "values", "offset", "_offset", "_idx", "rotated", "left_padding"),
    )
    return True


def apply() -> bool:
    """Attach ``rollback_state = None`` to ``ArraysCache`` (idempotent).

    Idempotency is checked against the live class attribute, not a
    module-level flag — keeps the patch consistent with the rest of
    mlx_lm_mtp after the #1388 self-healing refactor.
    """
    try:
        from mlx_lm.models.cache import ArraysCache
    except ImportError:
        logger.debug("mlx_lm.models.cache not importable; skipping rollback_state")
        return False

    _attach_rotating_undo()

    if hasattr(ArraysCache, "_omlx_rollback_attached"):
        return True

    if hasattr(ArraysCache, "rollback_state"):
        # Upstream may have added it natively (e.g. once PR 990 lands).
        ArraysCache._omlx_rollback_attached = "upstream"
        return True

    ArraysCache.rollback_state = None
    ArraysCache._omlx_rollback_attached = "patch"
    return True
