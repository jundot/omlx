# SPDX-License-Identifier: Apache-2.0
"""Dynamic expert-residency governor.

A fixed streaming budget is decided at load time and never revisited, but
system free memory keeps moving (observed on the 36-cell matrix: RSS peak
~25-27G on 51G while decode runs mostly under 50% utilization). The
governor revisits the cache capacity at request boundaries:

- free < LOW  -> clear() the cache (desperate; pages are re-readable)
- free < TGT  -> halve capacity (floor MIN_CAP)
- free > HIGH -> double capacity (ceiling max_budget)
- in between  -> stable (hysteresis; no churn)

Opt-in: OMLX_EXPERT_STREAMING_DYNAMIC=1 (a budget-0 run is page-cache-only
by operator choice and stays untouched). Actions log one line each; a
cooldown prevents oscillation. observe() runs on the inference thread at
request boundaries -- same thread that calls cache.put/get -- so no
cross-thread races on the OrderedDict.
"""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)

_DYNAMIC_ENV = os.environ.get("OMLX_EXPERT_STREAMING_DYNAMIC", "").strip() == "1"


def dynamic_residency_enabled() -> bool:
    return _DYNAMIC_ENV


def _max_dynamic_budget_bytes(default_gib: float = 6.0) -> int:
    raw = os.environ.get("OMLX_EXPERT_STREAMING_DYNAMIC_MAX_GIB", "").strip()
    try:
        gib = float(raw) if raw else default_gib
    except ValueError:
        gib = default_gib
    return max(0, int(gib * 1024**3))


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
    ) -> None:
        # Thresholds default scale with PHYSICAL memory (observed on the
        # 51G box: a dirty page cache from prior benches normalizes
        # available to ~24G, so absolute 24G-high never fires). Fractions:
        # 10% desperate, 20% shrink, 40% grow.
        ram = _total_ram_bytes()
        self.cache = cache
        self.per_slot = max(1, int(per_slot))
        self.num_layers = max(0, int(num_layers))
        self.max_budget_bytes = int(max_budget_bytes)
        self.low_free_bytes = int(low_free_bytes if low_free_bytes is not None else int(ram * 0.10))
        self.target_free_bytes = int(target_free_bytes if target_free_bytes is not None else int(ram * 0.20))
        self.high_free_bytes = int(high_free_bytes if high_free_bytes is not None else int(ram * 0.40))
        self.cooldown_s = float(cooldown_s)
        self.min_cap = max(1, int(min_cap))
        self._last_action_at = time.monotonic()  # cooldown counts from creation
        self._last_free_gib = 0.0
        self.actions = 0
        self.last_action = "init"

    def _caps_for(self, budget_bytes: int) -> tuple:
        cap = max(1, budget_bytes // self.per_slot)
        per_layer = max(1, cap // self.num_layers) if self.num_layers > 0 else 0
        return cap, per_layer

    def _apply(self, cap: int, per_layer: int) -> None:
        cache = self.cache
        cache.capacity = cap
        if self.num_layers > 0:
            cache._per_layer_cap = per_layer
        cache._global_cap = cap
        while len(cache._store) > cap:
            old_k, _ = cache._store.popitem(last=False)
            old_layer = cache._layer_of(old_k)
            cache._layer_counts[old_layer] = max(0, cache._layer_counts.get(old_layer, 1) - 1)
            cache.stats.evictions += 1
        if self.num_layers > 0:
            for layer in list(getattr(cache, "_layer_counts", {})):
                while cache._layer_counts.get(layer, 0) > per_layer:
                    victim = None
                    for k in cache._store:
                        if cache._layer_of(k) == layer:
                            victim = k
                            break
                    if victim is None:
                        break
                    cache._store.pop(victim)
                    cache._layer_counts[layer] = max(0, cache._layer_counts.get(layer, 1) - 1)
                    cache.stats.evictions += 1

    def observe(self, force: bool = False) -> str:
        """One governor step; returns the action taken (empty when idle)."""
        try:
            cache = self.cache
            if cache is None or getattr(cache, "capacity", 0) <= 0:
                return ""
            free = _free_bytes()
            if free <= 0:
                return ""
            self._last_free_gib = free / 1024**3
            now = time.monotonic()
            if not force and (now - self._last_action_at) < self.cooldown_s:
                return ""
            action = ""
            if free < self.low_free_bytes:
                cache.clear()
                action = "clear (free=%.1fG)" % (free / 1024**3)
            elif free < self.target_free_bytes:
                want = max(self.min_cap, cache.capacity // 2)
                if want < cache.capacity:
                    cap, per_layer = self._caps_for(want * self.per_slot)
                    self._apply(cap, per_layer)
                    action = "shrink cap=%d (free=%.1fG)" % (cap, free / 1024**3)
            else:
                cap_now = cache.capacity
                budget_now = cap_now * self.per_slot
                want_budget = min(self.max_budget_bytes, budget_now * 2)
                if want_budget > budget_now and free > self.high_free_bytes:
                    cap, per_layer = self._caps_for(want_budget)
                    if cap > cap_now:
                        self._apply(cap, per_layer)
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

    def summary(self) -> dict:
        return {
            "actions": self.actions,
            "last_action": self.last_action,
            "last_free_gib": round(self._last_free_gib, 1),
            "capacity": getattr(self.cache, "capacity", 0),
            "per_slot": self.per_slot,
        }
