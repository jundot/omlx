# SPDX-License-Identifier: Apache-2.0
"""Shared governor-facing adapter for partitioned per-layer expert caches.

The two non-unified backends — DeepSeek V4.1's ``_ExpertSlots`` backing
(``deepseek_v41.streaming_backing.V41StreamingBacking``) and the legacy
fetch-on-miss per-layer ``ExpertCache``s
(``moe_expert_offload.LegacyOffloadState``) — present the same duck-type
to ``ExpertResidencyGovernor``: ``capacity`` / ``resize`` / ``clear`` /
``set_layer_caps`` / ``layer_cap_overrides`` / ``stats`` plus a
``note_visit`` decode counter feeding the hunger window. This module
carries everything shared; subclasses supply the layer iteration, the
cap application, and the floor rule.

Unit adaptation: partitioned stores have no single pool, so the adapter
reports capacity in per-layer units — ``per_slot`` is one resident row
summed across every wrapped layer and the governor runs
``num_layers=1``, making every budget<->slots conversion exact and a
global shrink unable to starve one layer below its decode working set.
Targeting overrides (``set_layer_caps``) are per-layer ceilings;
``capacity`` reports the uniform base.

Subclass hooks (each called with ``self._lock`` held unless noted):

``_iter_units()``        — the per-layer stores in stable order.
``_per_slot_bytes()``    — bytes of one resident row across all layers.
``_initial_base_cap()``  — the uniform per-layer slot ceiling at build.
``_floor_cap()``         — per-layer floor: never shrink below one
                           token's decode working set.
``_apply_layer_caps()``  — push effective caps onto every unit
                           (``self._cap_for`` computes each).
``_clear_units()``       — drop all resident rows.
``_make_stats()``        — the ``DecodeVisitStats`` instance (override to
                           carry extra per-backend counters).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from .slot_cache import DecodeVisitStats

logger = logging.getLogger(__name__)


class PerLayerResidencyAdapter:
    """Governor-facing cache over partitioned per-layer expert stores."""

    # Governor arm label; subclasses override with their backend name.
    _GOVERNOR_LABEL = "expert streaming"

    def __init__(
        self,
        cache: Any,
        *,
        dynamic: bool = True,
        max_budget_bytes: int | None = None,
        min_budget_bytes: int | None = None,
        stall_target: float | None = None,
        min_cap: int | None = None,
    ) -> None:
        self._lock = threading.RLock()
        # The subclass-shaped per-layer store collection; the hooks read
        # it. ``self`` — not ``cache`` — is what the governor resizes.
        self.cache = cache
        self.overrides: dict = {}
        self.stats = self._make_stats()
        self.num_layers = len(list(self._iter_units()))
        self.per_slot = max(1, int(self._per_slot_bytes()))
        self.base_cap = max(0, int(self._initial_base_cap()))
        self.governor = None
        if not dynamic or self.num_layers == 0 or self.base_cap <= 0:
            return
        from .governor import arm_dynamic

        # Per-layer units (see module docstring): the governor sees
        # num_layers=1 and per_slot spanning every layer. floor_default
        # covers the no-explicit-min_cap case (never below one token's
        # working set); an explicit min_cap wins inside arm_dynamic.
        self.governor = arm_dynamic(
            self,
            self.per_slot,
            self.base_cap,
            floor_default=max(1, min(self._floor_cap(), self.base_cap)),
            max_budget_bytes=max_budget_bytes,
            min_budget_bytes=min_budget_bytes,
            stall_target=stall_target,
            min_cap=min_cap,
            num_layers=1,
            label=self._GOVERNOR_LABEL,
        )

    # -- subclass hooks ----------------------------------------------------

    def _iter_units(self):
        """Iterate the per-layer stores (stable order)."""
        raise NotImplementedError

    def _per_slot_bytes(self) -> int:
        """Bytes of one resident row summed across every layer."""
        raise NotImplementedError

    def _initial_base_cap(self) -> int:
        """Uniform per-layer slot ceiling the units were built with."""
        raise NotImplementedError

    def _floor_cap(self) -> int:
        """Per-layer slot floor — one token's decode working set.

        Applied both as the clamp on effective caps (``_cap_for``) and,
        bounded by ``base_cap``, as ``arm_dynamic``'s ``floor_default``.
        """
        return 1

    def _apply_layer_caps(self) -> None:
        """Push the effective caps onto the units (under ``_lock``).

        Read ``self._cap_for(key)`` per unit — override when present,
        floored ``base_cap`` otherwise.
        """
        raise NotImplementedError

    def _clear_units(self) -> None:
        """Drop every resident row in the per-layer stores (under ``_lock``)."""
        raise NotImplementedError

    def _make_stats(self) -> DecodeVisitStats:
        """The visit-stats object ``ExpertResidencyGovernor._window``
        duck-reads. Override to carry extra per-backend counters (e.g.
        V4.1's verify visits)."""
        return DecodeVisitStats()

    # -- governor duck-type ------------------------------------------------

    def _cap_for(self, key: int) -> int:
        """Effective cap for one unit: its override else the floored base."""
        return max(self._floor_cap(), int(self.overrides.get(key, self.base_cap)))

    def resize(self, cap: int, per_layer: int | None) -> None:
        """Retarget the uniform per-layer base (atomic under the lock).

        With ``num_layers=1`` the governor always passes ``per_layer ==
        cap``; growth materializes lazily on demand, so resize only ever
        sets ceilings and compacts overs. Targeting overrides reset to
        uniform; the governor re-applies them via ``set_layer_caps``
        right after (same contract as the generic cache). ``base_cap``
        keeps the governor's raw request so consecutive shrinks do not
        flap against the floor.
        """
        want = max(1, int(per_layer if per_layer is not None else cap))
        with self._lock:
            self.base_cap = want
            self.overrides = {}
            self._apply_layer_caps()

    def clear(self) -> None:
        with self._lock:
            self._clear_units()

    def set_layer_caps(self, caps: dict) -> None:
        floor = max(1, int(self._floor_cap()))
        with self._lock:
            self.overrides = {}
            for k, v in (caps or {}).items():
                eff = max(floor, int(v))
                # An override that lands on the applied base is not
                # targeting — dropping it keeps layer_cap_overrides()
                # truthful (the governor's num_layers==1 retarget sends
                # exactly this no-op).
                if eff != max(floor, self.base_cap):
                    self.overrides[int(k)] = eff
            self._apply_layer_caps()

    def layer_cap_overrides(self) -> dict:
        return dict(self.overrides)

    @property
    def capacity(self) -> int:
        # Per-layer-units contract (see module docstring): the uniform
        # per-layer base, not a total.
        return self.base_cap

    @property
    def evictions(self) -> int:
        # Per-layer counters live on the units; the adapter aggregates.
        return sum(int(getattr(u, "evictions", 0) or 0) for u in self._iter_units())

    def note_visit(self, layer_idx: int, missed: bool) -> None:
        """Count one decode-shaped layer-call, then tick the governor.

        Called by the units AFTER their own lock is released — taking
        the adapter lock here keeps the unit -> adapter order impossible
        to invert. tick() self-throttles on the governor's tick interval,
        so the cost is a monotonic compare until it elapses; it runs
        outside the adapter lock because observe() may resize(), which
        retakes it (RLock is reentrant; unscoped matches the mandated
        governor->cache lock order).
        """
        with self._lock:
            self.stats.note_visit(layer_idx, missed)
        gov = self.governor
        if gov is not None:
            try:
                gov.tick()
            except Exception:
                logger.debug("governor tick failed", exc_info=True)
