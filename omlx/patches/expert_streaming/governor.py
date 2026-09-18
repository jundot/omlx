# SPDX-License-Identifier: Apache-2.0
"""Dynamic expert-residency governor (auto default).

A fixed streaming budget is decided at load time and never revisited, but
system free memory keeps moving. The
governor revisits the cache capacity at request boundaries from TWO
signals — pressure (free memory) and hunger (windowed decode stall):

- free < LOW  -> clear() the cache (desperate; pages are re-readable)
- free < TGT  -> halve capacity (floor: min budget)
- hunger (windowed decode-layer stall > target) + free > TGT -> grow
  additively (+25%, ceiling max_budget), targeted at the layers that
  missed most (per-layer cap overrides; the global trim still rules)
- free > HIGH -> double capacity (abundance; ceiling max_budget)
- in between  -> stable (hysteresis; no churn)

Pressure always wins over hunger, and one shared cooldown blocks a
shrink-then-regrow flap. Hunger uses DECODE-layer counters only, so a
prefill-heavy request cannot steer decode residency; windows too small
to trust are skipped.

Auto by default (no explicit budget pin): a budget-0 run is
page-cache-only by operator choice and stays untouched. Actions log one
line each.

Threading: observe() is called at request boundaries from
``_log_streaming_summary``, which runs on the asyncio event loop (batched.py
and vlm.py), while tick() runs mid-request on the inference thread from
``_LayerLoadContext.close()``. Both serialize on the governor's own lock,
and every cache mutation goes through the cache's lock
(``ExpertLRUCache.resize`` / ``clear``) in governor->cache order, so the
two callers cannot interleave inside one update or deadlock a drain.
"""

from __future__ import annotations

import logging
import threading
import time

from ._env import env_bool, env_float

logger = logging.getLogger(__name__)

_DYNAMIC_ENV = env_bool("OMLX_EXPERT_STREAMING_DYNAMIC", False)

# Mid-request ticks: request-boundary observe() plus the 30s cooldown
# allowed ~1 action per long decode — the budget never caught the working
# set inside one request. tick() runs from _LayerLoadContext.close() (once
# per MoE layer-call on the inference thread) with its own throttle;
# actions keep the normal pressure/hunger rules but use a shorter spacing
# so the cap converges in ~15 s. OMLX_EXPERT_STREAMING_GOV_TICK=0 restores
# boundary-only behavior.
_GOV_TICK_ENV = env_bool("OMLX_EXPERT_STREAMING_GOV_TICK", True)
# Tick throttle constants (dev tunables, not envs): observe spacing and the
# shorter mid-request action spacing tick() substitutes for cooldown_s.
_GOV_TICK_S = 1.0
_GOV_TICK_ACTION_S = 3.0


def dynamic_residency_enabled() -> bool:
    return _DYNAMIC_ENV


def _max_dynamic_budget_bytes(default_gib: float = 0.0) -> int:
    gib = env_float("OMLX_EXPERT_STREAMING_DYNAMIC_MAX_GIB", default_gib)
    if gib <= 0:
        # Dynamic ceiling: a quarter of RAM, bounded — a bigger private
        # LRU squeezes the OS page cache and turns the remaining misses
        # into physical SSD reads, which costs more than the extra hits
        # save.
        gib = min(24.0, _total_ram_bytes() / 1024**3 * 0.25)
    return max(0, int(gib * 1024**3))


def _free_bytes() -> int:
    """Best-effort system free memory (``virtual_memory().available``).

    Shared with the rest of oMLX via ``utils.psutil_compat`` — native
    ``host_statistics64`` first, TTL-cached ``vm_stat`` subprocess only as
    the fallback (no per-observe subprocess churn, real page size, no
    hardcoded guess). ``available`` = free + inactive: psutil's primary
    path never counted purgeable pages, so the old vm_stat fallback's
    purgeable-inclusive figure was the outlier, not the contract. 0 when
    every probe fails — callers treat it as "unknown" and idle.
    """
    try:
        from ...utils.psutil_compat import virtual_memory

        return int(virtual_memory().available)
    except Exception:
        return 0


def _total_ram_bytes() -> int:
    """Total physical RAM (psutil_compat's sysconf/sysctl/psutil chain)."""
    try:
        from ...utils.psutil_compat import get_total_memory

        total = int(get_total_memory())
    except Exception:
        total = 0
    # Keep the 64 GiB guess when every probe fails: the watermark
    # fractions need a nonzero RAM figure — 0 would freeze the governor.
    return total if total > 0 else 64 * 1024**3


def _clampf(v, lo: float, hi: float, default: float) -> float:
    """float(v) clamped to [lo, hi]; unparseable -> default."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return min(hi, max(lo, f))


def _bytes_or(override, ram: int, env: str, frac: float) -> int:
    """Explicit byte watermark wins; else ram * env-clamped fraction.

    The env is a 0..1 fraction of physical RAM (``OMLX_GOV_*_FRAC``);
    invalid values degrade to *frac* via ``env_float``.
    """
    if override is not None:
        return int(override)
    return int(ram * min(0.95, max(0.0, env_float(env, frac))))


class ExpertResidencyGovernor:
    """Resize the expert cache capacity from system free memory."""

    def __init__(
        self,
        cache,
        per_slot: int,
        num_layers: int,
        max_budget_bytes: int,
        *,
        low_free_bytes: int | None = None,
        target_free_bytes: int | None = None,
        high_free_bytes: int | None = None,
        cooldown_s: float = 30.0,
        min_cap: int = 32,
        min_budget_bytes: int | None = None,
        stall_target: float = 0.05,
        grow_add_frac: float = 0.25,
        min_window_layers: int = 16,
    ) -> None:
        # Thresholds scale with PHYSICAL memory: absolute watermarks
        # misfire when a dirty page cache keeps "available" permanently
        # low. Fractions: 10% desperate, 20% shrink, 40% grow — portable
        # defaults; envs OMLX_GOV_*_FRAC tune per machine (explicit
        # kwargs always win).
        ram = _total_ram_bytes()
        self.cache = cache
        self.per_slot = max(1, int(per_slot))
        self.num_layers = max(0, int(num_layers))
        self.max_budget_bytes = int(max_budget_bytes)
        self.low_free_bytes = _bytes_or(low_free_bytes, ram, "OMLX_GOV_LOW_FRAC", 0.10)
        self.target_free_bytes = _bytes_or(
            target_free_bytes, ram, "OMLX_GOV_TARGET_FRAC", 0.20
        )
        self.high_free_bytes = _bytes_or(
            high_free_bytes, ram, "OMLX_GOV_HIGH_FRAC", 0.40
        )
        self.cooldown_s = float(cooldown_s)
        self.min_cap = max(1, int(min_cap))
        self.min_budget_bytes = max(0, int(min_budget_bytes or 0))
        self.stall_target = _clampf(stall_target, 0.0, 0.9, 0.05)
        self.grow_add_frac = _clampf(grow_add_frac, 0.0, 4.0, 0.25)
        self.min_window_layers = max(1, int(min_window_layers))
        # Governor lock: serializes observe()/tick() state updates between
        # the asyncio event loop (request boundaries) and the inference
        # thread (mid-request ticks). Lock order is governor -> cache:
        # every cache mutation goes through cache.resize/clear under this
        # lock, and no cache-locked path calls back into the governor, so
        # the inference thread can never hold cache->_lock while waiting
        # on _lock (that inversion would deadlock a drain).
        self._lock = threading.RLock()
        self._last_action_at = time.monotonic()  # cooldown counts from creation
        self._last_tick_at = 0.0
        self._last_free_gib = 0.0
        self.actions = 0
        self.last_action = "init"
        # Hunger baselines (cumulative stats snapshots for windowed deltas).
        self._last_decode_layers = 0
        self._last_decode_missed = 0
        self._last_miss_by_layer: dict = {}
        self.last_window_stall = 0.0
        self.last_window_layers = 0

    def _caps_for(self, budget_bytes: int) -> tuple:
        cap = max(1, budget_bytes // self.per_slot)
        per_layer = max(1, cap // self.num_layers) if self.num_layers > 0 else 0
        return cap, per_layer

    def _apply(self, cap: int, per_layer: int, layers: list | None = None) -> None:
        """Retarget the cache capacity via ``cache.resize()``.

        The supported cache contract requires resize() so the whole
        retarget (capacity, per-layer cap, drain, per-layer trim) is one
        atomic step under the cache lock. resize() resets overrides to
        uniform, so per-layer targeting is re-applied right after: a grow
        keeps its targeting and a shrink returns to uniform (layers
        empty/None). Each override is bounded by the global cap
        (overrides are ceilings; the global trim still rules), so
        targeting can never over-admit.
        """
        self.cache.resize(cap, per_layer if self.num_layers > 0 else None)
        try:
            setter = getattr(self.cache, "set_layer_caps", None)
            if not callable(setter):
                return
            if not layers:
                setter({})
                return
            per_layer = max(1, int(per_layer))
            setter({int(l): min(int(cap), per_layer * 2) for l in layers})
        except Exception:
            logger.debug("governor retarget failed", exc_info=True)

    def reconcile_per_slot(self, per_slot: int) -> None:
        """Adopt the post-conversion per-slot bytes.

        The converter picks ``per_slot`` before it knows the majority
        projection layout (``__init__.py`` hardcodes ``per_expert // 3``),
        then reconciles ``cache.per_slot_bytes`` / ``capacity`` /
        ``_per_layer_cap`` from the real layout. Without this the governor
        keeps the pre-reconciliation number, and on a fused ``gate_up``
        model its first grow pins a capacity ~1.5x above the budget the
        user asked for -- silently over the ``max_budget_bytes`` ceiling.
        """
        try:
            per_slot = int(per_slot)
        except (TypeError, ValueError):
            return
        with self._lock:
            if per_slot > 0 and per_slot != self.per_slot:
                logger.info(
                    "expert_streaming governor: per_slot reconciled %d -> %d",
                    self.per_slot,
                    per_slot,
                )
                self.per_slot = per_slot

    # -- public introspection (contract API for the streaming paths) ------

    def min_capacity(self) -> int:
        """Floor (slots) the governor never shrinks the cache below."""
        with self._lock:
            return self._min_cap_slots()

    def at_floor(self) -> bool:
        """True when the cache capacity sits at the governor's floor.

        Speculative staging reads this to stop prefetching into a pool
        that has no slack — a staged row admitted at the floor just
        evicts a demanded one.
        """
        cache = self.cache
        if cache is None:
            return False
        try:
            cap = int(getattr(cache, "capacity", 0) or 0)
        except Exception:
            return False
        with self._lock:
            # capacity <= floor covers the budget-0 page-cache-only case:
            # there is no slack to stage into at all.
            return cap <= self._min_cap_slots()

    def in_desperate_band(self) -> bool:
        """True when last-observed free memory is inside the clear band.

        Unknown free (no observe yet, ``_last_free_gib == 0``) reports
        False — suppressing speculation on a never-observed machine would
        disable it entirely.
        """
        with self._lock:
            free = float(self._last_free_gib or 0.0)
        return 0.0 < free < (self.low_free_bytes / 1024**3)

    def _min_cap_slots(self) -> int:
        if self.min_budget_bytes > 0 and self.per_slot > 0:
            return max(self.min_cap, self.min_budget_bytes // self.per_slot)
        return self.min_cap

    def _window(self) -> tuple[int, int, dict]:
        """Windowed (layers, missed, per-layer misses) since last observe.

        Stats are cumulative and reset() can wipe them mid-run: a backwards
        counter means a reset happened, so re-baseline and report an empty
        window (never a negative one).
        """
        st = getattr(self.cache, "stats", None)
        if st is None:
            return 0, 0, {}
        try:
            layers = int(getattr(st, "decode_layers", 0) or 0)
            missed = int(getattr(st, "decode_layers_missed", 0) or 0)
            by_layer = dict(getattr(st, "decode_misses_by_layer", None) or {})
        except (TypeError, ValueError):
            return 0, 0, {}
        if layers < self._last_decode_layers or missed < self._last_decode_missed:
            self._last_decode_layers = layers
            self._last_decode_missed = missed
            self._last_miss_by_layer = by_layer
            return 0, 0, {}
        win_layers = layers - self._last_decode_layers
        win_missed = missed - self._last_decode_missed
        win_by_layer = {
            k: by_layer.get(k, 0) - self._last_miss_by_layer.get(k, 0)
            for k in by_layer
        }
        win_by_layer = {k: v for k, v in win_by_layer.items() if v > 0}
        self._last_decode_layers = layers
        self._last_decode_missed = missed
        self._last_miss_by_layer = by_layer
        return win_layers, win_missed, win_by_layer

    def _target_layers(self, win_by_layer: dict) -> list:
        """Top-missing layers for targeted growth (bounded count)."""
        if not win_by_layer or self.num_layers <= 0:
            return []
        ranked = sorted(win_by_layer.items(), key=lambda kv: kv[1], reverse=True)
        return [k for k, _ in ranked[: max(1, self.num_layers // 6)]]

    def observe(self, force: bool = False, *, cooldown: float | None = None) -> str:
        """One governor step; returns the action taken (empty when idle).

        Serialized under the governor lock: request-boundary calls (event
        loop) and mid-request ticks (inference thread) cannot interleave
        baselines, watermarks, or the cooldown. The lock is re-entrant so
        ``tick()`` can hold it across its throttle bookkeeping. ``cooldown``
        overrides ``self.cooldown_s`` for this step only (tick's shorter
        action spacing).
        """
        with self._lock:
            return self._observe_locked(force, cooldown=cooldown)

    def _grow(
        self, free: int, factor: float, win_by_layer: dict | None = None
    ) -> tuple | None:
        """One growth step toward the budget/headroom ceiling.

        Shared by the hunger branch (additive ``1 + grow_add_frac``,
        targeted at the top-missing layers) and the abundance branch
        (``2.0``, uniform): the ceiling is dynamic — the cache may grow
        while free stays above ``target_free_bytes``; the static
        ``max_budget_bytes`` remains the outer bound. Returns
        ``(cap, target_layers)`` when the capacity grew, else None.
        """
        cache = self.cache
        cap_now = cache.capacity
        budget_now = cap_now * self.per_slot
        headroom_budget = budget_now + max(0, free - self.target_free_bytes)
        want_budget = min(
            self.max_budget_bytes,
            headroom_budget,
            int(budget_now * factor),
        )
        if want_budget <= budget_now:
            return None
        cap, per_layer = self._caps_for(want_budget)
        if cap <= cap_now:
            return None
        layers = self._target_layers(win_by_layer)
        self._apply(cap, per_layer, layers)
        return cap, layers

    def _observe_locked(self, force: bool, cooldown: float | None = None) -> str:
        try:
            cache = self.cache
            if cache is None or getattr(cache, "capacity", 0) <= 0:
                return ""
            free = _free_bytes()
            if free <= 0:
                return ""
            self._last_free_gib = free / 1024**3
            now = time.monotonic()
            win_layers, win_missed, win_by_layer = self._window()
            self.last_window_layers = win_layers
            self.last_window_stall = (
                win_missed / win_layers if win_layers > 0 else 0.0
            )
            cooldown_s = self.cooldown_s if cooldown is None else float(cooldown)
            if not force and (now - self._last_action_at) < cooldown_s:
                return ""
            action = ""
            if free < self.low_free_bytes:
                cache.clear()
                action = "clear (free=%.1fG)" % (free / 1024**3)
            elif free < self.target_free_bytes:
                floor = self._min_cap_slots()
                # Proportional correction: shed only what restores the
                # target headroom. Halving on a small overshoot caused a
                # grow/shrink sawtooth that evicted tens of thousands of
                # hot entries mid-decode.
                give_back = self.target_free_bytes - free
                want = max(
                    floor,
                    (cache.capacity * self.per_slot - give_back)
                    // self.per_slot,
                )
                if want < cache.capacity:
                    cap, per_layer = self._caps_for(want * self.per_slot)
                    self._apply(cap, per_layer)
                    action = "shrink cap=%d (free=%.1fG)" % (cap, free / 1024**3)
            elif (
                win_layers >= self.min_window_layers
                and self.last_window_stall > self.stall_target
                and free > self.target_free_bytes
            ):
                # Hunger: proven decode stalls + headroom (not abundance).
                grew = self._grow(free, 1.0 + self.grow_add_frac, win_by_layer)
                if grew is not None:
                    cap, layers = grew
                    action = "grow cap=%d layers=%d (stall=%.2f free=%.1fG)" % (
                        cap,
                        len(layers),
                        self.last_window_stall,
                        free / 1024**3,
                    )
            elif free > self.high_free_bytes:
                # Abundance: double capacity; between TGT and HIGH the
                # band is stable (hysteresis, no churn).
                grew = self._grow(free, 2.0)
                if grew is not None:
                    action = "grow cap=%d (free=%.1fG)" % (grew[0], free / 1024**3)
            if action:
                self._last_action_at = now
                self.actions += 1
                self.last_action = action
                logger.info("expert_streaming governor: %s", action)
            return action
        except Exception:
            logger.debug("governor observe failed", exc_info=True)
            return ""

    def tick(self) -> str:
        """Mid-request observation point.

        Called once per MoE layer-call from ``_LayerLoadContext.close()``
        on the inference thread — so it must be cheap: a monotonic
        compare until the tick interval elapses, then one observe() step
        with a shorter action spacing than the request-boundary
        cooldown. Resize/clear still go through the cache lock; the
        occasional drain cost is amortized by the tick throttle.
        """
        if not _GOV_TICK_ENV:
            return ""
        with self._lock:
            now = time.monotonic()
            if now - self._last_tick_at < _GOV_TICK_S:
                return ""
            self._last_tick_at = now
            return self.observe(cooldown=_GOV_TICK_ACTION_S)

    def summary(self) -> dict:
        cache = self.cache
        try:
            overrides = len(getattr(cache, "layer_cap_overrides", lambda: {})())
        except Exception:
            overrides = 0
        with self._lock:
            return {
                "actions": self.actions,
                "last_action": self.last_action,
                "last_free_gib": round(self._last_free_gib, 1),
                "capacity": getattr(cache, "capacity", 0),
                "per_slot": self.per_slot,
                "window_stall": round(self.last_window_stall, 3),
                "window_layers": self.last_window_layers,
                "stall_target": self.stall_target,
                "layer_overrides": overrides,
            }


def arm_dynamic(
    cache,
    per_slot: int,
    base_cap: int,
    *,
    floor_default: int | None,
    max_budget_bytes: int | None,
    min_budget_bytes: int | None,
    stall_target: float | None,
    min_cap: int | None = None,
    num_layers: int = 1,
    initial_total_bytes: int | None = None,
    label: str = "expert streaming",
    **governor_kwargs,
) -> ExpertResidencyGovernor | None:
    """Arm an :class:`ExpertResidencyGovernor` on a dynamic cache.

    Shared arm block for the governor duck-types (the V4.1 streaming
    backing, the legacy offload state — per-layer-units caches whose
    ``capacity`` is a per-layer slot count). Resolves the byte budgets:
    an explicit ``max_budget_bytes`` pins the ceiling, otherwise the
    DYNAMIC_MAX_GIB env / quarter-of-RAM default applies; the floor is
    ``min_budget_bytes`` or a quarter of the initial total (never below
    0.25 GiB). ``floor_default`` is the caller's model-specific min-cap
    fallback (e.g. one token's routed working set); an explicit
    ``min_cap`` always wins. ``num_layers`` is the governor's layer count
    for windowing (1 for per-layer-units caches). ``initial_total_bytes``
    overrides the initial total budget estimate (defaults to
    ``base_cap * per_slot``) for caches whose slot bookkeeping makes that
    product a poor measure of the real starting footprint. Extra governor
    kwargs pass through. Arming failure is soft: residency just stays
    static (returns None).
    """
    try:
        initial_total = (
            int(base_cap) * max(1, int(per_slot))
            if initial_total_bytes is None
            else int(initial_total_bytes)
        )
        gov_max = (
            int(max_budget_bytes)
            if max_budget_bytes is not None
            else _max_dynamic_budget_bytes()
        )
        gov_min = (
            int(min_budget_bytes)
            if min_budget_bytes is not None
            else max(int(0.25 * 1024**3), initial_total // 4)
        )
        # Never shrink a layer below one token's decode working set —
        # explicit min_cap wins, then the caller's floor_default, then
        # the governor's own default.
        floor = min_cap if min_cap is not None else floor_default
        kwargs = dict(governor_kwargs)
        if stall_target is not None:
            kwargs["stall_target"] = stall_target
        if floor is not None:
            kwargs["min_cap"] = max(1, int(floor))
        governor = ExpertResidencyGovernor(
            cache,
            max(1, int(per_slot)),
            num_layers,
            max(gov_max, initial_total),
            min_budget_bytes=gov_min,
            **kwargs,
        )
        logger.info(
            "%s: dynamic residency governor armed "
            "(per-layer %d slots, min %d, max %.2f GiB)",
            label,
            int(base_cap),
            governor._min_cap_slots(),
            governor.max_budget_bytes / 1024**3,
        )
        return governor
    except Exception:
        logger.debug("%s: governor arming failed", label, exc_info=True)
        return None
