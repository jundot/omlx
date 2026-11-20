# SPDX-License-Identifier: Apache-2.0
"""SenseNova Loop MTP module: loop-aware cache rollback and MTP specialization.

This module carries the SenseNova Loop MTP-specific logic (slot-based
rollback, skipping untouched slots), isolated from the generic
``qwen35_model.py``. It overrides ``qwen3_5.TextModel.mtp_partial_rollback``
with the loop-aware version and **falls back to the original generic version**
on non-Loop models (behavior identical to before the split).

This module is **independent** of the ``sensenova_loop`` architecture module
(``sensenova_loop.py``).  It only reads the ``_loop_plan`` slot layout from a
loaded model and, when present, applies the slot-based rollback.  The MTP
head attachment itself is handled by ``qwen35_model`` (generic); this module
only adapts the cache-rollback step to the Loop execution plan.

Key background:
  Under the Loop architecture the verify forward executes only the **first
  segment** (primary slice) of the execution plan (``_loop_plan``); the
  repeated-slice slots that follow are **never touched** (linear layers have
  no stash, KV offset == 0). Unless the rollback is slot-based rather than
  layer-based, the length check never passes and the whole chain falls back
  to standard stepping, disabling speculative decoding entirely.

Design principle (separation of concerns):
  ``qwen35_model.py``'s ``mtp_partial_rollback`` is the **generic version**
  (identity slot layout, zero Loop concept); this module **wraps** it with
  the loop-aware version:
    - Loop models   -> loop-aware logic (slot-based, skip untouched slots);
    - non-Loop models -> fall back to the original generic version (no effect).

  Dependency direction: sensenova_loop_mtp -> qwen35_model (reads via cls
  method); qwen35_model knows nothing about SenseNova.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Tuple

logger = logging.getLogger(__name__)

_MARKER = "_omlx_sensenova_loop_mtp"

# Original generic mtp_partial_rollback (saved before apply() overrides it,
# used as the non-Loop fallback). None if apply() has not run or the original
# is missing (defensive).
_FALLBACK = None


def mtp_slot_layout(model: Any) -> Optional[Tuple[int, Callable[[int], int]]]:
    """Return the slot layout ``(n_slots, phys_fn)`` for MTP rollback.

    - ``n_slots`` -- cache slot count (== len(_loop_plan) for Loop models);
    - ``phys_fn(slot)`` -- slot-to-physical-layer index mapping.

    Non-Loop models return None and the caller falls back to the generic
    identity layout (``n_slots = len(model.layers)``).
    """
    plan = getattr(model, "_loop_plan", None)
    if plan is None:
        return None
    # The loop-aware forward (Qwen3_5TextModel.__call__ in sensenova_loop.py)
    # consumes the full ``_loop_plan``: ``for slot, phys in enumerate(plan)``,
    # and slot's cache is written by physical layer ``plan[slot]`` (Loop
    # repeated slices: layers 4-27 are visited once in slots 4-27 and once in
    # slots 28-51, while physical layers 28-31 are visited only once in slots
    # 52-55).
    #
    # Therefore rollback's phys_fn must return ``plan[slot]`` -- unlike the old
    # assumption (forward consumed n_phys slots via zip(layers, cache) only).
    # The old phys_fn returning ``slot if slot < n_phys else None`` was wrong
    # under the loop-aware forward: slots 28-55 were misjudged as dead slots
    # and skipped from rollback, leaving KV/ssm_state residue that corrupted
    # MTP-on output (off was correct, on was wrong).
    #
    # Historical lesson: an even earlier version used plan[slot] but the
    # forward then consumed only 32 slots (identity), so slots 28-31 mapped to
    # layers 4-7 and rollback replayed the wrong layer (layer 4 vs 28 have
    # different weights), scrambling the GDN ssm_state. The forward is now
    # loop-aware, so plan[slot] is correct again.
    return (
        len(plan),
        lambda slot: plan[slot],
    )


def _slot_layout(model: Any) -> Optional[tuple]:
    """Get the Loop slot layout ``(n_slots, phys_fn)``; None for non-Loop models."""
    try:
        return mtp_slot_layout(model)
    except Exception:
        return None


def _loop_rollback(model, cache, accepted: int, num_drafts: int, layout: tuple) -> bool:
    """Loop-aware cache rollback core (slot-based, skips untouched slots)."""
    n_slots, phys_fn = layout
    layers = model.layers
    if len(cache) != n_slots:
        return False
    trim_n = num_drafts - accepted
    if trim_n <= 0:
        return True
    keep = 1 + accepted  # confirmed token + accepted drafts

    # --- First pass: verify every touched slot has rollback state ---
    for slot in range(n_slots):
        phys = phys_fn(slot)
        if phys is None:
            continue  # Dead slot not consumed by the forward (slot >= physical count)
        layer = layers[phys]
        c = cache[slot]
        if getattr(layer, "is_linear", False):
            rs = getattr(c, "rollback_state", None)
            ds = getattr(c, "_mtp_draft_stash", None)
            if rs is None:
                # Untouched linear layer: no stash, skip; anomalous state
                # (stash present but no state) fails loud.
                if ds is not None:
                    return False
                continue
            if ds is None:
                return False
        else:
            # KV slot: offset == 0 means untouched, skip; otherwise must be trimmable.
            off = getattr(c, "offset", 0)
            if off is None or off <= 0:
                continue
            if not (hasattr(c, "is_trimmable") and c.is_trimmable()):
                return False

    # --- Second pass: execute the rollback ---
    for slot in range(n_slots):
        phys = phys_fn(slot)
        if phys is None:
            continue  # Dead slot
        layer = layers[phys]
        c = cache[slot]
        if getattr(layer, "is_linear", False):
            if getattr(c, "rollback_state", None) is None:
                continue  # Untouched linear layer, skip
            conv_0, ssm_0 = c.rollback_state
            qkv_s, a_s, b_s = c._mtp_draft_stash
            _, conv_m, ssm_m = layer.linear_attn._process_chunk(
                qkv_s[:, :keep],
                a_s[:, :keep],
                b_s[:, :keep],
                conv_0,
                ssm_0,
                None,
            )
            c[0] = conv_m
            c[1] = ssm_m
            c.rollback_state = None
            c._mtp_draft_stash = None
        else:
            off = getattr(c, "offset", 0)
            if off is None or off <= 0:
                continue  # Untouched KV slot, skip
            c.trim(trim_n)
    return True


def mtp_partial_rollback(self, cache, accepted: int, num_drafts: int) -> bool:
    """Loop-aware wrapper (installed on TextModel, replacing the generic version).

    - Loop model (``_loop_plan`` present) -> loop-aware logic;
    - non-Loop model -> fall back to the original generic version (behavior
      identical to before the split).
    """
    model = self.model
    layout = _slot_layout(model)
    if layout is not None:
        return _loop_rollback(model, cache, accepted, num_drafts, layout)
    # Non-Loop: fall back to the original generic version (defensive False
    # if the original is unavailable).
    if _FALLBACK is not None:
        return _FALLBACK(self, cache, accepted, num_drafts)
    return False


def apply_mtp() -> bool:
    """Overlay TextModel.mtp_partial_rollback with the loop-aware version (idempotent).

    The original generic version is saved as ``_FALLBACK`` before the override
    (for the non-Loop fallback). Idempotence is enforced via a ``cls.__dict__``
    marker. The outer ``q35.Model`` forwarding version is left untouched (it
    forwards via ``self.language_model`` and ultimately calls the TextModel
    wrapper).

    Only overrides when mlx_lm.models.qwen3_5 is importable and TextModel
    already has mtp_partial_rollback; otherwise it skips (defensive, returns
    False).

    This is the **MTP-only** entry point.  It assumes the
    ``mlx_lm.models.sensenova_loop`` module is already registered (by
    ``sensenova_loop.apply()``) and the generic ``mtp_partial_rollback`` is
    installed (by ``qwen35_model``).  The Loop/MHC architecture is *not*
    installed here -- it lives in ``sensenova_loop.py`` and loads
    independently of MTP.
    """
    global _FALLBACK

    try:
        from mlx_lm.models import sensenova_loop as sl
    except Exception as e:  # pragma: no cover
        logger.debug("sensenova_loop not importable; skipping sensenova_loop_mtp: %s", e)
        return False

    cls = sl.TextModel
    # ``mtp_partial_rollback`` is installed on qwen3_5.TextModel by
    # qwen35_model; the subclass inherits it, so read via getattr (including
    # inherited) rather than ``__dict__.get``.
    current = getattr(cls, "mtp_partial_rollback", None)
    if current is None:
        return False  # Original generic version not installed (qwen35_model not applied)

    already_marked = getattr(current, _MARKER, False)
    if not already_marked:
        # Save the original generic version as the non-Loop fallback.
        _FALLBACK = current
        cls.mtp_partial_rollback = mtp_partial_rollback
        setattr(cls.mtp_partial_rollback, _MARKER, True)
        logger.info("SenseNova Loop MTP patch applied (wrap mtp_partial_rollback)")
    return True
