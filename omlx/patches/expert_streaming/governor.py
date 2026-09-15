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
import os
import threading
import time

logger = logging.getLogger(__name__)

_DYNAMIC_ENV = os.environ.get("OMLX_EXPERT_STREAMING_DYNAMIC", "").strip() == "1"

# Mid-request ticks: request-boundary observe() plus the 30s cooldown
# allowed ~1 action per long decode — the budget never caught the working
# set inside one request. tick() runs from _LayerLoadContext.close() (once
# per MoE layer-call on the inference thread) with its own throttle;
# actions keep the normal pressure/hunger rules but use a shorter spacing
# so the cap converges in ~15 s. OMLX_EXPERT_STREAMING_GOV_TICK=0 restores
# boundary-only behavior.
_GOV_TICK_ENV = os.environ.get("OMLX_EXPERT_STREAMING_GOV_TICK", "1") != "0"
_GOV_TICK_S = float(
    os.environ.get("OMLX_EXPERT_STREAMING_GOV_TICK_S", "1.0") or 1.0
)
_GOV_TICK_ACTION_S = float(
    os.environ.get("OMLX_EXPERT_STREAMING_GOV_TICK_ACTION_S", "3.0") or 3.0
)


def dynamic_residency_enabled() -> bool:
    return _DYNAMIC_ENV


def _max_dynamic_budget_bytes(default_gib: float = 0.0) -> int:
    raw = os.environ.get("OMLX_EXPERT_STREAMING_DYNAMIC_MAX_GIB", "").strip()
    try:
        gib = float(raw) if raw else default_gib
    except ValueError:
        gib = default_gib
    if gib <= 0:
        # Dynamic ceiling: a quarter of RAM, bounded — a bigger private
        # LRU squeezes the OS page cache and turns the remaining misses
        # into physical SSD reads, which costs more than the extra hits
        # save.
        gib = min(24.0, _total_ram_bytes() / 1024**3 * 0.25)
    return max(0, int(gib * 1024**3))


def _frac_env(name: str, default: float) -> float:
    """Watermark fraction env override (0..1); invalid values keep default."""
    try:
        v = float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default
    return min(0.95, max(0.0, v))


def _free_bytes() -> int:
    """Best-effort system free memory (psutil.available; vm_stat fallback)."""
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        import subprocess

        out = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=2
        ).stdout
        page = 16384
        free = inactive = purgeable = 0
        for line in out.splitlines():
            if "Pages free:" in line:
                free = int(line.split(":")[1].strip().rstrip("."))
            elif "Pages inactive:" in line:
                inactive = int(line.split(":")[1].strip().rstrip("."))
            elif "Pages purgeable:" in line:
                purgeable = int(line.split(":")[1].strip().rstrip("."))
        return (free + inactive + purgeable) * page
    except Exception:
        return 0


def _total_ram_bytes() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    try:
        import subprocess

        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=2
        ).stdout.strip()
        return int(out) if out else 64 * 1024**3
    except Exception:
        return 64 * 1024**3


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
        self.low_free_bytes = int(
            low_free_bytes
            if low_free_bytes is not None
            else int(ram * _frac_env("OMLX_GOV_LOW_FRAC", 0.10))
        )
        self.target_free_bytes = int(
            target_free_bytes
            if target_free_bytes is not None
            else int(ram * _frac_env("OMLX_GOV_TARGET_FRAC", 0.20))
        )
        self.high_free_bytes = int(
            high_free_bytes
            if high_free_bytes is not None
            else int(ram * _frac_env("OMLX_GOV_HIGH_FRAC", 0.40))
        )
        self.cooldown_s = float(cooldown_s)
        self.min_cap = max(1, int(min_cap))
        self.min_budget_bytes = max(0, int(min_budget_bytes or 0))
        try:
            self.stall_target = float(stall_target)
        except (TypeError, ValueError):
            self.stall_target = 0.05
        self.stall_target = min(0.9, max(0.0, self.stall_target))
        try:
            self.grow_add_frac = float(grow_add_frac)
        except (TypeError, ValueError):
            self.grow_add_frac = 0.25
        self.grow_add_frac = min(4.0, max(0.0, self.grow_add_frac))
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

    def _apply(self, cap: int, per_layer: int) -> None:
        """Retarget the cache capacity via ``cache.resize()``.

        The supported cache contract requires resize() so the whole
        retarget (capacity, per-layer cap, drain, per-layer trim) is one
        atomic step under the cache lock.
        """
        self.cache.resize(cap, per_layer if self.num_layers > 0 else None)

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

    def observe(self, force: bool = False) -> str:
        """One governor step; returns the action taken (empty when idle).

        Serialized under the governor lock: request-boundary calls (event
        loop) and mid-request ticks (inference thread) cannot interleave
        baselines, watermarks, or the cooldown. The lock is re-entrant so
        ``tick()`` can hold it across its throttle bookkeeping.
        """
        with self._lock:
            return self._observe_locked(force)

    def _observe_locked(self, force: bool) -> str:
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
            if not force and (now - self._last_action_at) < self.cooldown_s:
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
                    self._retarget(cache, cap, per_layer)
                    action = "shrink cap=%d (free=%.1fG)" % (cap, free / 1024**3)
            elif (
                win_layers >= self.min_window_layers
                and self.last_window_stall > self.stall_target
                and free > self.target_free_bytes
            ):
                # Hunger: proven decode stalls + headroom (not abundance).
                # The ceiling is dynamic: the cache may grow while free
                # stays above `target_free_bytes`; the static max_budget
                # remains the outer bound.
                cap_now = cache.capacity
                budget_now = cap_now * self.per_slot
                headroom_budget = budget_now + max(
                    0, free - self.target_free_bytes
                )
                want_budget = min(
                    self.max_budget_bytes,
                    headroom_budget,
                    int(budget_now * (1.0 + self.grow_add_frac)),
                )
                if want_budget > budget_now:
                    cap, per_layer = self._caps_for(want_budget)
                    if cap > cap_now:
                        self._apply(cap, per_layer)
                        layers = self._target_layers(win_by_layer)
                        self._retarget(cache, cap, per_layer, layers)
                        action = (
                            "grow cap=%d layers=%d (stall=%.2f free=%.1fG)"
                            % (cap, len(layers), self.last_window_stall, free / 1024**3)
                        )
            else:
                cap_now = cache.capacity
                budget_now = cap_now * self.per_slot
                headroom_budget = budget_now + max(
                    0, free - self.target_free_bytes
                )
                want_budget = min(
                    self.max_budget_bytes, headroom_budget, budget_now * 2
                )
                if want_budget > budget_now and free > self.high_free_bytes:
                    cap, per_layer = self._caps_for(want_budget)
                    if cap > cap_now:
                        self._apply(cap, per_layer)
                        self._retarget(cache, cap, per_layer)
                        action = "grow cap=%d (free=%.1fG)" % (cap, free / 1024**3)
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
            saved = self.cooldown_s
            self.cooldown_s = _GOV_TICK_ACTION_S
            try:
                return self.observe()
            finally:
                self.cooldown_s = saved

    def _retarget(
        self, cache, cap: int, per_layer: int, layers: list | None = None
    ) -> None:
        """Apply per-layer targeting after a resize (best-effort).

        resize() resets overrides to uniform; re-apply here so a grow keeps
        its targeting and a shrink returns to uniform (layers=None). Each
        override is bounded by the global cap (overrides are ceilings; the
global trim still rules), so targeting can never over-admit.
        """
        try:
            setter = getattr(cache, "set_layer_caps", None)
            if not callable(setter):
                return
            if not layers:
                setter({})
                return
            per_layer = max(1, int(per_layer))
            setter({int(l): min(int(cap), per_layer * 2) for l in layers})
        except Exception:
            logger.debug("governor retarget failed", exc_info=True)

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
