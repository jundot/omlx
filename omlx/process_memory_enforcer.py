# SPDX-License-Identifier: Apache-2.0
"""
Process-level memory enforcer for oMLX.

The enforcer derives a hard ceiling from the configured memory_guard_tier
(safe / balanced / aggressive / custom) and the current system state,
then drives soft / hard watermarks from that ceiling. When usage crosses
a watermark it unloads LRU models from EnginePool and pauses admission
for new prefills.

Ceiling = min(static_ceiling, dynamic_ceiling, metal_cap):
  static_ceiling  = total_ram - tier.static_reserve
  dynamic_ceiling depends on tier:
    safe / balanced / aggressive
      = omlx_phys + free + inactive + active * tier.reclaim_ratio
        (free / inactive / active from host_statistics64; active reclaim
        ratio is 0.2 / 0.5 / 0.8 — the fraction of active memory the OS
        can compress / swap out under pressure)
    custom
      = user-specified custom_ceiling_bytes (set via the admin dashboard)

static_ceiling caps absolute Metal pressure. dynamic_ceiling moves with
system state every poll so the cap shrinks or grows as other apps come
and go. metal_cap guards against panics from Apple's per-process Metal
limit being below the chosen ceiling.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import subprocess
import time
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import mlx.core as mx

from . import settings as _settings
from .utils import psutil_compat
from .utils.proc_memory import get_phys_footprint

if TYPE_CHECKING:
    from .engine_pool import EnginePool
    from .model_settings import ModelSettingsManager
    from .settings import GlobalSettings

logger = logging.getLogger(__name__)


# Reserve sub-24 GB systems regardless of tier. Small Macs cannot afford a
# tier-scaled cut and still load any useful model.
_SMALL_SYSTEM_RESERVE = 4 * 1024**3
_SMALL_SYSTEM_THRESHOLD = 24 * 1024**3

# Tier map: static reserve for systems at or above the small-system threshold.
# `custom` shares the `balanced` reserve so the static cap stays sane
# regardless of what the user types into the custom ceiling field.
_STATIC_RESERVE_LARGE: dict[str, int] = {
    "safe": 8 * 1024**3,
    "balanced": 6 * 1024**3,
    "aggressive": 4 * 1024**3,
    "custom": 2 * 1024**3,
}

# Fraction of "active" pages we count as reclaimable via macOS
# compression / swap. macOS's compressor averages 2-3x so ~60-67% of
# active is realistically reclaimable; 0.8 pushes into swap territory.
_ACTIVE_RECLAIM_RATIO: dict[str, float] = {
    "safe": 0.2,
    "balanced": 0.5,
    "aggressive": 0.8,
}

# Default soft watermark per tier. A saved 0.85 from older configs is treated
# as the legacy default so balanced / aggressive can move to their tier defaults.
_LEGACY_SOFT_THRESHOLD = 0.85
_SOFT_THRESHOLD_BY_TIER: dict[str, float] = {
    "safe": 0.85,
    "balanced": 0.90,
    "aggressive": 0.925,
    "custom": 0.85,
}

# Fraction of the hard ceiling used by the adaptive prefill chunk sizer.
_PREFILL_HEADROOM_SAFETY: dict[str, float] = {
    "safe": 0.90,
    "balanced": 0.90,
    "aggressive": 0.925,
    "custom": 0.90,
}

# Fraction of the effective physical cap used by the pre-chunk prediction
# guard. Aggressive/custom are user-directed and can run closer to the
# configured ceiling.
_PREFILL_ABORT_MARGIN: dict[str, float] = {
    "safe": 0.90,
    "balanced": 0.90,
    "aggressive": 0.95,
    "custom": 0.95,
}

# Last-resort active-request brake. Normal hard pressure starts at the
# hard watermark (usually 95% of ceiling); pinned workloads are only aborted
# after the process crosses the actual ceiling by a small margin, or stays
# over the ceiling for consecutive polls.
_EMERGENCY_OVER_CEILING_MARGIN_BYTES = 2 * 1024**3
_EMERGENCY_OVER_CEILING_POLLS = 2
_HOT_CACHE_RESERVATION_SLACK_BYTES = 512 * 1024**2


def _format_gb(b: int) -> str:
    """Format bytes as GB string."""
    return f"{b / 1024**3:.1f}GB"


def get_macos_vm_stats() -> dict[str, int] | None:
    """Snapshot of mach `vm_statistics64` in bytes.

    Returns None on non-macOS or when the host call fails. ~0.8 us per
    call so this is safe inside the enforcer poll loop and inside
    per-chunk memcheck.

    The dict exposes only the first four page counters we use for the
    dynamic ceiling math. Those counters are stable at the front of
    `vm_statistics64`; using a max-sized `host_info64_t` buffer avoids
    pinning oMLX to an SDK-specific tail layout.
    """
    return psutil_compat.get_macos_vm_stats()


def get_iogpu_wired_limit_bytes() -> int:
    """Read the kernel's `iogpu.wired_limit_mb` sysctl in bytes.

    Returns 0 when the value is unset (`0` in sysctl means "use the system
    default", typically ~75% of RAM) or when the read fails. Callers
    should treat 0 as "limit unknown / not enforced" and fall back to a
    different source (e.g. mx.device_info()'s working set size).
    """
    try:
        out = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "iogpu.wired_limit_mb"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        value_mb = int(out.stdout.strip())
        if value_mb <= 0:
            return 0
        return value_mb * 1024**2
    except (subprocess.SubprocessError, ValueError, OSError):
        return 0


def _get_max_metal_working_set_bytes() -> int:
    """Apple's default Metal cap as reported by MLX (~75% of RAM).

    `mx.set_wired_limit` refuses any value above this when the kernel
    iogpu.wired_limit_mb is unset (= 0).
    """
    try:
        info = mx.device_info()
        size = int(info.get("max_recommended_working_set_size", 0) or 0)
        return max(0, size)
    except Exception:  # noqa: BLE001
        return 0


def get_effective_metal_cap_bytes() -> int:
    """Effective per-process Metal allocation cap.

    Uses the kernel iogpu.wired_limit_mb when explicitly set (> 0).
    Otherwise falls back to Apple's max_recommended_working_set_size.
    This is the value above which `mx.set_wired_limit` will reject the
    request, so callers should clamp against it before calling MLX.
    """
    sysctl_cap = get_iogpu_wired_limit_bytes()
    if sysctl_cap > 0:
        return sysctl_cap
    return _get_max_metal_working_set_bytes()


def _wired_limit_suggestion_bytes(desired_bytes: int) -> int:
    """Clamp a wired-limit recommendation to leave the OS 5% of RAM.

    Wiring within a few GiB of physical RAM invites jetsam during a
    large-model load burst, and a jetsammed/hard-killed process strands
    its wired allocation at kernel level until reboot (#2184). 5% matches
    the field-stable margin from that report (488 GiB stable on a 512 GiB
    box, 510 GiB crash-looped). On small-memory machines 5% is below the
    tier static reserve, so their recommendation is unchanged.

    Only shapes the suggested sysctl value in logs and the admin banner;
    the enforcement path still honors whatever the kernel sysctl allows,
    including user-set values above this recommendation.

    Resolved through the settings module at call time so tests that patch
    omlx.settings.get_system_memory control this the same way they control
    the static ceiling.
    """
    try:
        total = int(_settings.get_system_memory())
    except Exception:  # noqa: BLE001
        return desired_bytes
    if total <= 0:
        return desired_bytes
    return max(0, min(desired_bytes, total - total // 20))


def _apply_metal_wired_limit(desired_bytes: int) -> tuple[int, int | None]:
    """Try to raise Metal wired limit for this process to `desired_bytes`.

    Returns (applied_bytes, previous_bytes). `applied_bytes` is what we
    actually told MLX (clamped to the kernel iogpu.wired_limit_mb if it
    is lower); `previous_bytes` is what MLX reports the prior limit was,
    or None on failure / older macOS where the call is unavailable.

    Emits a WARNING when the kernel sysctl caps us below the recommended
    wired limit so the user sees the hint in logs in addition to the admin
    UI red banner. The recommended value is clamped below physical RAM
    (_wired_limit_suggestion_bytes) so following it cannot push the OS
    into jetsam during a large-model load (#2184).

    When iogpu.wired_limit_mb is unset (0), leave Apple's default Metal
    cap active instead of calling mx.set_wired_limit with the same default
    cap. The scheduler still clamps against get_effective_metal_cap_bytes();
    this only avoids changing MLX allocator state unless the user explicitly
    raised the kernel cap.
    """
    if desired_bytes <= 0:
        return 0, None

    suggestion = _wired_limit_suggestion_bytes(desired_bytes)
    sysctl_cap = get_iogpu_wired_limit_bytes()
    if sysctl_cap <= 0:
        effective_cap = get_effective_metal_cap_bytes()
        if effective_cap > 0 and effective_cap < suggestion:
            logger.warning(
                "Metal cap (%s, Apple max_recommended_working_set_size) is "
                "below the oMLX static ceiling (%s); leaving Apple's default "
                "Metal cap active because iogpu.wired_limit_mb is unset. "
                "Raise it with: sudo sysctl iogpu.wired_limit_mb=%d",
                _format_gb(effective_cap),
                _format_gb(desired_bytes),
                suggestion // (1024**2),
            )
        else:
            logger.debug(
                "Skipping mx.set_wired_limit because iogpu.wired_limit_mb is "
                "unset (target=%s, Apple cap=%s)",
                _format_gb(desired_bytes),
                _format_gb(effective_cap),
            )
        return 0, None

    try:
        _total = int(_settings.get_system_memory())
    except Exception:  # noqa: BLE001
        _total = 0
    if _total > 0:
        _reserve = _total // 20
        if sysctl_cap > _total - _reserve:
            # The kernel cap leaves the OS less headroom than the
            # recommendation floor. A big-model load burst can jetsam the
            # server at this setting, and the stranded wired memory then
            # needs a reboot to reclaim (#2184).
            logger.warning(
                "iogpu.wired_limit_mb (%s) leaves the OS less than %s of "
                "physical RAM (%s); a large model load can trigger jetsam "
                "at this setting and strand the wired memory until reboot "
                "(#2184). Recommended maximum: %d MB.",
                _format_gb(sysctl_cap),
                _format_gb(_reserve),
                _format_gb(_total),
                (_total - _reserve) // (1024**2),
            )

    effective_cap = sysctl_cap
    capped = effective_cap > 0 and effective_cap < desired_bytes
    applied = effective_cap if capped else desired_bytes
    try:
        previous = mx.set_wired_limit(applied)
        if capped and effective_cap < suggestion:
            logger.warning(
                "Metal cap (%s, %s) is below the oMLX static ceiling (%s); "
                "Metal will clamp allocations to the cap and panic if a "
                "request exceeds it. Raise it with: sudo sysctl "
                "iogpu.wired_limit_mb=%d",
                _format_gb(effective_cap),
                "kernel iogpu.wired_limit_mb",
                _format_gb(desired_bytes),
                suggestion // (1024**2),
            )
        return applied, int(previous)
    except Exception as exc:  # noqa: BLE001
        # Older macOS (<15) or the API just isn't available. Log + skip.
        logger.warning(
            "mx.set_wired_limit(%s) failed; Metal will use its default cap (%s)",
            _format_gb(applied),
            exc,
        )
        return 0, None


class ProcessMemoryEnforcer:
    """
    Background task that enforces process-level memory limits.

    Polls usage every poll_interval seconds. On every tick it recomputes
    the dynamic ceiling from system_available, so other-app pressure is
    reflected immediately without restarting the enforcer.
    """

    def __init__(
        self,
        engine_pool: EnginePool,
        memory_guard_tier: str = "balanced",
        memory_guard_custom_ceiling_gb: float = 0.0,
        poll_interval: float = 1.0,
        settings_manager: ModelSettingsManager | None = None,
        prefill_memory_guard: bool = True,
        global_settings: GlobalSettings | None = None,
        soft_threshold: float | None = None,
        hard_threshold: float = 0.95,
        prefill_safe_zone_ratio: float = 0.89,
        prefill_min_chunk_tokens: int = 256,
    ):
        """
        Initialize the process memory enforcer.

        Args:
            engine_pool: The engine pool to evict models from.
            memory_guard_tier: One of "safe", "balanced", "aggressive", "custom".
                Picks the active-memory reclaim ratio (0.2 / 0.5 / 0.8) and
                the static reserve. "custom" uses
                memory_guard_custom_ceiling_gb directly for the dynamic
                ceiling instead of computing from vm_stat.
            memory_guard_custom_ceiling_gb: Custom ceiling in GB. Only
                used when tier == "custom". Clamped by static_ceiling and
                metal_cap so a too-large value is panic-safe.
            poll_interval: Seconds between memory checks.
            settings_manager: Optional settings manager for TTL checks.
            prefill_memory_guard: When False, returns a ceiling of 0 so
                callers treat the limit as disabled.
            global_settings: Optional global settings for idle timeout.
            soft_threshold: Optional explicit fraction of ceiling that triggers
                soft action. None, or the legacy 0.85 default, uses the tier
                default instead.
            hard_threshold: Fraction of ceiling that triggers hard action
                (LRU/non-pinned aborts, loading aborts, and idle reclaim).
            prefill_safe_zone_ratio: Fraction of hard cap below which prefill
                runs at full chunk size; above triggers adaptive shrink.
            prefill_min_chunk_tokens: Floor for adaptive shrink.
        """
        self._engine_pool = engine_pool
        self._memory_guard_tier = self._normalize_tier(memory_guard_tier)
        self._memory_guard_custom_ceiling_bytes = max(
            0, int(memory_guard_custom_ceiling_gb * 1024**3)
        )
        self._active_poll_interval = poll_interval
        self._loaded_idle_poll_interval = 10.0
        self._unloaded_idle_poll_interval = 30.0
        self._current_poll_interval = poll_interval
        self._settings_manager = settings_manager
        self._prefill_memory_guard = prefill_memory_guard
        self._global_settings = global_settings
        self._soft_threshold_override = self._normalize_soft_threshold_override(
            soft_threshold
        )
        self._soft_threshold = self._get_soft_threshold()
        self._hard_threshold = hard_threshold
        self._prefill_headroom_safety = self._get_prefill_headroom_safety()
        self._prefill_safe_zone_ratio = prefill_safe_zone_ratio
        self._prefill_min_chunk_tokens = prefill_min_chunk_tokens
        self._task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wake_event: asyncio.Event | None = None
        self._running = False
        self._activity_hint_until = 0.0
        # Most recently observed pressure level, consumed by scheduler /
        # admission control. Updated on every poll iteration.
        self._pressure_level: str = "ok"
        self._over_ceiling_polls: int = 0
        # Last value passed to mx.set_wired_limit (0 if not yet applied
        # or the call failed). Used by the admin dashboard to surface a
        # warning when the kernel iogpu.wired_limit_mb is below this.
        self._metal_wired_limit_request: int = 0
        # Cached Metal cap used by the background poll loop. Reading the Apple
        # default cap falls back to mx.device_info(), so keep it out of active
        # decode ticks once the enforcer is running.
        self._effective_metal_cap_bytes: int | None = None
        # Engine types we've already complained about in
        # ``_propagate_memory_limit``'s "scheduler unreachable" path.
        # Prevents the per-poll warning from spamming logs while keeping
        # the first occurrence loud enough to alert CI / oncall.
        self._scheduler_resolve_warned: set[str] = set()

    @staticmethod
    def _normalize_tier(tier: str) -> str:
        t = (tier or "").strip().lower()
        if t not in _STATIC_RESERVE_LARGE:
            return "balanced"
        return t

    @staticmethod
    def _normalize_soft_threshold_override(value: float | None) -> float | None:
        if value is None:
            return None
        threshold = float(value)
        if threshold <= 0:
            return None
        if abs(threshold - _LEGACY_SOFT_THRESHOLD) < 1e-9:
            return None
        return threshold

    def _refresh_tier_thresholds(self) -> None:
        self._soft_threshold = self._get_soft_threshold()
        self._prefill_headroom_safety = self._get_prefill_headroom_safety()

    def _get_soft_threshold(self) -> float:
        if self._soft_threshold_override is not None:
            return self._soft_threshold_override
        return _SOFT_THRESHOLD_BY_TIER[self._memory_guard_tier]

    def _get_prefill_headroom_safety(self) -> float:
        return _PREFILL_HEADROOM_SAFETY[self._memory_guard_tier]

    @property
    def memory_guard_tier(self) -> str:
        return self._memory_guard_tier

    @memory_guard_tier.setter
    def memory_guard_tier(self, value: str) -> None:
        new_tier = self._normalize_tier(value)
        if new_tier == self._memory_guard_tier:
            return
        old = self._memory_guard_tier
        self._memory_guard_tier = new_tier
        self._refresh_tier_thresholds()
        if self._running:
            if self._prefill_memory_guard:
                self._refresh_effective_metal_cap_bytes()
            self._propagate_memory_limit()
        logger.info(f"Memory guard tier changed: {old} -> {new_tier}")

    @property
    def memory_guard_custom_ceiling_bytes(self) -> int:
        return self._memory_guard_custom_ceiling_bytes

    @memory_guard_custom_ceiling_bytes.setter
    def memory_guard_custom_ceiling_bytes(self, value: int) -> None:
        new_value = max(0, int(value))
        if new_value == self._memory_guard_custom_ceiling_bytes:
            return
        old = self._memory_guard_custom_ceiling_bytes
        self._memory_guard_custom_ceiling_bytes = new_value
        if self._running:
            if self._prefill_memory_guard:
                self._refresh_effective_metal_cap_bytes()
            self._propagate_memory_limit()
        logger.info(
            "Memory guard custom ceiling changed: %s -> %s",
            _format_gb(old),
            _format_gb(new_value),
        )

    @property
    def is_running(self) -> bool:
        """Whether the enforcement loop is active."""
        return self._running

    def start(self) -> None:
        """Start the background enforcement loop.

        Also mirrors the static ceiling into MLX's wired-memory limit when
        the user explicitly raised iogpu.wired_limit_mb. When the kernel
        sysctl is unset, the scheduler still clamps against Apple's default
        Metal cap, but oMLX leaves MLX allocator state untouched.
        """
        if self._running:
            return
        if self._prefill_memory_guard:
            self._refresh_effective_metal_cap_bytes()
        self._running = True
        self._propagate_memory_limit()
        ceiling = self._get_hard_limit_bytes()

        if self._prefill_memory_guard:
            static_ceiling = self._get_static_ceiling()
            applied, previous = _apply_metal_wired_limit(static_ceiling)
            # Store the *recommended* limit (static ceiling clamped below
            # physical RAM, see _wired_limit_suggestion_bytes) rather than
            # the post-clamp applied value. The admin UI compares this
            # against the live iogpu.wired_limit_mb so a kernel cap below
            # the recommendation triggers the red sysctl-command banner;
            # clamping keeps the banner from telling users to wire nearly
            # all of physical RAM (#2184).
            self._metal_wired_limit_request = _wired_limit_suggestion_bytes(
                static_ceiling
            )
            if applied > 0:
                logger.info(
                    "Metal wired limit raised: %s -> %s "
                    "(target=%s, iogpu sysctl cap=%s)",
                    _format_gb(previous or 0),
                    _format_gb(applied),
                    _format_gb(static_ceiling),
                    _format_gb(get_iogpu_wired_limit_bytes()),
                )

        self._task = asyncio.create_task(self._enforcement_loop())
        logger.info(
            f"Process memory enforcer started "
            f"(tier={self._memory_guard_tier}, "
            f"ceiling={_format_gb(ceiling)}, "
            f"interval={self._active_poll_interval}s)"
        )

    def wake(self, *, active: bool = False) -> None:
        """Wake the polling loop before its current sleep timeout expires.

        ``active=True`` keeps the loop on the fast interval briefly. This covers
        request/model-load entry points before an engine's active-request
        collectors are visible to ``_select_poll_interval``.
        """
        if active:
            self._activity_hint_until = max(
                self._activity_hint_until,
                time.monotonic() + max(2.0, self._active_poll_interval * 2),
            )

        event = self._wake_event
        loop = self._loop
        if event is None or loop is None or loop.is_closed():
            return

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is loop:
            event.set()
        else:
            loop.call_soon_threadsafe(event.set)

    def _get_static_ceiling(self) -> int:
        """Total RAM minus tier-scaled static reserve."""
        from .settings import get_system_memory

        system_bytes = get_system_memory()
        if self._memory_guard_tier == "custom":
            return max(0, system_bytes - _STATIC_RESERVE_LARGE["custom"])
        if system_bytes < _SMALL_SYSTEM_THRESHOLD:
            reserve = _SMALL_SYSTEM_RESERVE
        else:
            reserve = _STATIC_RESERVE_LARGE[self._memory_guard_tier]
        return max(0, system_bytes - reserve)

    def _get_dynamic_ceiling(self) -> int:
        """Tier-aware reclaimable-memory ceiling.

        custom:
            Returns the user-supplied ceiling verbatim (clamped >= 0).
            min() with static / metal_cap still applies in
            `_get_hard_limit_bytes` so out-of-range input is panic safe.

        safe / balanced / aggressive:
            omlx_phys + free + inactive + active * ratio

            free / inactive / active come from `host_statistics64`
            (recomputed every call — never cached). active * ratio
            approximates how much active memory macOS can compress or
            swap out under pressure. Speculative and purgeable pages are
            subsets of free / inactive, so we deliberately do not add
            them (would double count).

        VM stats failure: falls back to psutil_compat.virtual_memory().available
        (= roughly free + inactive on macOS, similar elsewhere). On macOS that
        compat layer avoids psutil's HOST_VM_INFO64 adapter and uses a cached
        vm_stat fallback only if the fast host_statistics64 path is unavailable.
        If all telemetry is unavailable, fall back to the static ceiling so
        server health endpoints and the enforcer keep running.
        """
        if self._memory_guard_tier == "custom":
            return max(0, self._memory_guard_custom_ceiling_bytes)

        omlx_usage = get_phys_footprint()
        stats = get_macos_vm_stats()
        if stats is None:
            try:
                available = int(psutil_compat.virtual_memory().available)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Memory guard could not read available memory; "
                    "using static ceiling fallback: %s",
                    exc,
                )
                return self._get_static_ceiling()
            return max(0, omlx_usage + available)
        ratio = _ACTIVE_RECLAIM_RATIO[self._memory_guard_tier]
        reclaimable = stats["free"] + stats["inactive"] + int(stats["active"] * ratio)
        return max(0, omlx_usage + reclaimable)

    def _get_hard_limit_bytes(self) -> int:
        """Final hard ceiling = min(static, dynamic, metal_cap).

        Thin wrapper over ``_get_ceiling_breakdown`` that discards the
        component breakdown. Hot callers that don't need to know which
        ceiling is binding should keep using this helper.

        `metal_cap` is the effective Metal allocation cap (kernel
        iogpu.wired_limit_mb when set, otherwise Apple's
        max_recommended_working_set_size). Including it here means oMLX
        never plans allocations above what Metal will actually accept,
        so users who have not raised iogpu.wired_limit_mb still get a
        safe (smaller) ceiling rather than a panic.

        For `custom` tier the dynamic ceiling (vm_stat-based) is skipped;
        the user-specified value is capped only by static (total - 2 GB)
        and metal_cap.

        Returns 0 if the memory guard is disabled (callers treat 0 as
        "no limit").
        """
        return self._get_ceiling_breakdown()["hard_limit"]

    def _get_ceiling_breakdown(self) -> dict[str, int]:
        """Compute the hard limit AND the three component ceilings.

        Returns a dict with keys ``static``, ``dynamic``, ``metal_cap``,
        ``hard_limit`` (= min of the three non-zero values, or 0 when
        the guard is disabled). Used by ``_propagate_memory_limit`` to
        push the breakdown to schedulers so the prefill-rejection error
        message can identify which constraint is binding and suggest the
        right remedy. Single computation so the subprocess to ``sysctl``
        (inside ``get_effective_metal_cap_bytes``) only fires once per
        call.
        """
        if not self._prefill_memory_guard:
            return {"static": 0, "dynamic": 0, "metal_cap": 0, "hard_limit": 0}
        static_ceiling = self._get_static_ceiling()
        if self._memory_guard_tier == "custom":
            dynamic_ceiling = max(0, self._memory_guard_custom_ceiling_bytes)
        else:
            dynamic_ceiling = self._get_dynamic_ceiling()
        metal_cap = self._get_effective_metal_cap_bytes()
        candidates = [static_ceiling, dynamic_ceiling]
        if metal_cap > 0:
            candidates.append(metal_cap)
        return {
            "static": static_ceiling,
            "dynamic": dynamic_ceiling,
            "metal_cap": metal_cap,
            "hard_limit": min(candidates),
        }

    def get_final_ceiling(self) -> int:
        """Public accessor used by engine_pool pre-load admission."""
        return self._get_hard_limit_bytes()

    def _get_abort_limit_bytes(self) -> int:
        """Stable physical cap used to ABORT an in-flight prefill.

        Deliberately excludes the dynamic ceiling: that value jitters every
        poll with other-app pressure, and a transient dip must not kill a
        near-complete prefill whose usage actually fits the physical envelope.
        We use ``min(static_ceiling, metal_cap)`` — exactly the limit
        ``start()`` arms via ``mx.set_wired_limit`` — so allocating up to it
        cannot trigger a Metal clamp/panic. The dynamic ceiling still governs
        chunk-size throttling and admission elsewhere; this is only the
        last-resort kill threshold.

        Returns 0 when the guard is disabled (callers treat 0 as "no limit"
        and fall back to the dynamic hard limit).
        """
        if not self._prefill_memory_guard:
            return 0
        static_ceiling = self._get_static_ceiling()
        metal_cap = self._get_effective_metal_cap_bytes()
        if metal_cap > 0:
            return min(static_ceiling, metal_cap)
        return static_ceiling

    def _get_prefill_abort_margin(self) -> float:
        """Tier-specific prediction margin for pre-chunk safety checks."""
        return _PREFILL_ABORT_MARGIN[self._memory_guard_tier]

    def _soft_bytes(self) -> int:
        """Soft watermark: ceiling * soft_threshold."""
        ceiling = self._get_hard_limit_bytes()
        if ceiling <= 0:
            return 0
        return int(ceiling * self._soft_threshold)

    def _hard_bytes(self) -> int:
        """Hard watermark: ceiling * hard_threshold."""
        ceiling = self._get_hard_limit_bytes()
        if ceiling <= 0:
            return 0
        return int(ceiling * self._hard_threshold)

    def _current_usage_bytes(self) -> int:
        """Process memory usage as seen by macOS jetsam.

        During active requests this must not call MLX/Metal APIs from the
        background enforcer thread. The scheduler records the last
        mx.get_active_memory() sample on the MLX executor thread; the enforcer
        combines that cached value with the kernel phys_footprint ledger.

        When no request is active we keep the legacy direct MLX telemetry path
        so idle/status accounting remains as precise as before.
        """
        phys = get_phys_footprint()
        if self._has_active_requests():
            return max(self._cached_executor_active_memory_bytes(), phys)
        return max(mx.get_active_memory(), phys)

    def _is_emergency_pressure(self, current: int, ceiling: int) -> bool:
        """Return True only for pressure beyond the configured ceiling.

        This deliberately does not fire at the hard watermark. A pinned model
        means "do not unload"; active requests remain protected until the
        process is at or above the real ceiling and needs a last-resort brake.
        """
        if ceiling <= 0 or current < ceiling:
            self._over_ceiling_polls = 0
            return False

        self._over_ceiling_polls += 1
        if current >= ceiling + _EMERGENCY_OVER_CEILING_MARGIN_BYTES:
            return True
        return self._over_ceiling_polls >= _EMERGENCY_OVER_CEILING_POLLS

    def _refresh_effective_metal_cap_bytes(self) -> int:
        """Refresh the cached effective Metal cap outside the poll hot path."""
        self._effective_metal_cap_bytes = get_effective_metal_cap_bytes()
        return self._effective_metal_cap_bytes

    def _get_effective_metal_cap_bytes(self) -> int:
        """Return the cached Metal cap, populating it on first use."""
        if self._effective_metal_cap_bytes is None:
            return self._refresh_effective_metal_cap_bytes()
        return self._effective_metal_cap_bytes

    def _has_active_requests(self) -> bool:
        """Best-effort active-request detection without touching MLX."""
        for entry in self._engine_pool._entries.values():
            engine = getattr(entry, "engine", None)
            if engine is None:
                continue
            has_active_requests = getattr(engine, "has_active_requests", None)
            if not callable(has_active_requests):
                continue
            try:
                if has_active_requests() is True:
                    return True
            except Exception:
                return True
        return False

    def _cached_executor_active_memory_bytes(self) -> int:
        """Max MLX active-memory sample recorded by scheduler executor threads."""
        cached = 0
        for entry in self._engine_pool._entries.values():
            scheduler = self._resolve_scheduler(entry)
            if scheduler is None:
                continue
            getter = getattr(scheduler, "get_cached_mlx_active_memory_bytes", None)
            try:
                value = (
                    getter()
                    if callable(getter)
                    else getattr(scheduler, "_last_mlx_active_memory_bytes", 0)
                )
            except Exception:
                continue
            if isinstance(value, (int, float)):
                cached = max(cached, int(value))
        return cached

    @staticmethod
    def _nonnegative_bytes(value: Any) -> int | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return max(0, int(value))

    @classmethod
    def _nonnegative_byte_attr(cls, obj: Any, name: str) -> int | None:
        try:
            value = getattr(obj, name)
        except Exception:
            return None
        return cls._nonnegative_bytes(value)

    def _hot_cache_budget(self) -> Any | None:
        config = getattr(self._engine_pool, "_scheduler_config", None)
        if config is None:
            return None
        budget = getattr(config, "hot_cache_budget", None)
        if budget is None:
            return None
        max_bytes = self._nonnegative_byte_attr(budget, "max_bytes")
        total_bytes = self._nonnegative_byte_attr(budget, "total_bytes")
        if max_bytes is None or max_bytes <= 0 or total_bytes is None:
            return None
        return budget

    def _hot_cache_max_bytes(self) -> int:
        budget = self._hot_cache_budget()
        if budget is not None:
            return self._nonnegative_byte_attr(budget, "max_bytes") or 0
        config = getattr(self._engine_pool, "_scheduler_config", None)
        if config is None:
            return 0
        return self._nonnegative_byte_attr(config, "hot_cache_max_size") or 0

    @staticmethod
    def _manager_hot_cache_bytes(manager: Any) -> int:
        try:
            stats = manager.get_stats()
            size = ProcessMemoryEnforcer._nonnegative_byte_attr(
                stats, "hot_cache_size_bytes"
            )
            if size is not None:
                return size
        except Exception:
            pass
        return (
            ProcessMemoryEnforcer._nonnegative_byte_attr(
                manager, "_hot_cache_total_bytes"
            )
            or 0
        )

    def _hot_cache_used_bytes(self) -> int:
        budget = self._hot_cache_budget()
        if budget is not None:
            return self._nonnegative_byte_attr(budget, "total_bytes") or 0

        total = 0
        seen_managers: set[int] = set()
        for entry in self._engine_pool._entries.values():
            scheduler = self._resolve_scheduler(entry)
            if scheduler is None:
                continue
            manager = getattr(scheduler, "paged_ssd_cache_manager", None)
            if manager is None or id(manager) in seen_managers:
                continue
            seen_managers.add(id(manager))
            total += self._manager_hot_cache_bytes(manager)
        return total

    def _hot_cache_reserved_bytes(self) -> int:
        max_bytes = self._hot_cache_max_bytes()
        if max_bytes <= 0:
            return 0
        used = self._hot_cache_used_bytes()
        return min(max_bytes, used + _HOT_CACHE_RESERVATION_SLACK_BYTES)

    def _scheduler_limit_bytes(
        self, process_limit: int, *, reserved: int | None = None
    ) -> int:
        if process_limit <= 0:
            return 0
        if reserved is None:
            reserved = self._hot_cache_reserved_bytes()
        if reserved <= 0:
            return process_limit
        return max(1, process_limit - reserved)

    def _active_hot_cache_block_hashes(self) -> set[bytes]:
        hashes: set[bytes] = set()
        for entry in self._engine_pool._entries.values():
            scheduler = self._resolve_scheduler(entry)
            if scheduler is None:
                continue
            getter = getattr(scheduler, "get_active_hot_cache_block_hashes", None)
            if not callable(getter):
                continue
            try:
                hashes.update(bytes(h) for h in getter())
            except Exception:
                logger.debug("Failed to read active hot-cache block hashes")
        return hashes

    def _shrink_hot_cache_for_pressure(self, current: int, target: int) -> int:
        """Try to shrink hot cache enough to move process usage toward target."""
        hot_used = self._hot_cache_used_bytes()
        if hot_used <= 0 or current <= target:
            return 0

        target_hot_bytes = max(0, hot_used - (current - target))
        protected_hashes = self._active_hot_cache_block_hashes()
        budget = self._hot_cache_budget()
        if budget is not None:
            shrink = getattr(budget, "shrink_to", None)
            if callable(shrink):
                freed = int(
                    shrink(target_hot_bytes, protected_hashes=protected_hashes) or 0
                )
                if freed > 0:
                    logger.warning(
                        "Shrank shared hot cache under memory pressure: "
                        "freed=%s target_hot=%s protected=%d",
                        _format_gb(freed),
                        _format_gb(target_hot_bytes),
                        len(protected_hashes),
                    )
                return freed

        freed_total = 0
        remaining_to_free = current - target
        seen_managers: set[int] = set()
        for entry in self._engine_pool._entries.values():
            if remaining_to_free <= 0:
                break
            scheduler = self._resolve_scheduler(entry)
            if scheduler is None:
                continue
            manager = getattr(scheduler, "paged_ssd_cache_manager", None)
            if manager is None or id(manager) in seen_managers:
                continue
            seen_managers.add(id(manager))
            manager_used = self._manager_hot_cache_bytes(manager)
            if manager_used <= 0:
                continue
            target_manager_bytes = max(0, manager_used - remaining_to_free)
            shrink = getattr(manager, "shrink_hot_cache_to", None)
            if not callable(shrink):
                continue
            freed = int(
                shrink(target_manager_bytes, protected_hashes=protected_hashes) or 0
            )
            freed_total += freed
            remaining_to_free = max(0, remaining_to_free - freed)

        if freed_total > 0:
            logger.warning(
                "Shrank hot cache under memory pressure: freed=%s protected=%d",
                _format_gb(freed_total),
                len(protected_hashes),
            )
        return freed_total

    def get_pressure_level(self) -> str:
        """Return cached pressure level: 'ok', 'soft', or 'hard'.

        Consumed by scheduler `_schedule_waiting` and HTTP admission control.
        Updated on every enforcer poll iteration.
        """
        return self._pressure_level if self._running else "ok"

    @property
    def prefill_memory_guard(self) -> bool:
        """Whether prefill memory guard is enabled."""
        return self._prefill_memory_guard

    @prefill_memory_guard.setter
    def prefill_memory_guard(self, value: bool) -> None:
        self._prefill_memory_guard = value
        if self._running:
            self._propagate_memory_limit()
        logger.info(f"Prefill memory guard: {'enabled' if value else 'disabled'}")

    @staticmethod
    def _resolve_scheduler(entry: Any) -> Any | None:
        """Resolve the watermark target (Scheduler or DFlash guard) from an
        EnginePool entry.

        Most engines (BatchedEngine, VLMBatchedEngine) wrap the scheduler
        as ``entry.engine._engine.engine.scheduler`` (AsyncEngineCore →
        EngineCore → Scheduler). Some non-streaming engines may expose
        ``entry.engine.scheduler`` directly — including ``DFlashEngine`` in
        fallback mode, whose ``scheduler`` property resolves the fallback
        engine's real scheduler. DFlash's *primary* speculative path has no
        scheduler at all, so it exposes a lightweight ``_prefill_guard`` that
        carries the same watermark attrs; tried last so standard engines
        resolve unchanged. Returns None if nothing resolves.
        """
        eng = entry.engine
        if eng is None:
            return None
        sched = getattr(eng, "scheduler", None)
        if sched is not None:
            return sched
        inner = getattr(eng, "_engine", None)
        if inner is not None:
            inner_engine = getattr(inner, "engine", None)
            if inner_engine is not None:
                sched = getattr(inner_engine, "scheduler", None)
                if sched is not None:
                    return sched
        # DFlash primary mode bypasses the scheduler entirely; the enforcer
        # still needs to push the ceiling somewhere, so it lands on the
        # engine's ``_prefill_guard`` (None for every non-DFlash engine).
        return getattr(eng, "_prefill_guard", None)

    def _propagate_memory_limit(self) -> None:
        """Propagate ceiling-derived watermarks to all schedulers.

        Called on every enforcer tick so the dynamic ceiling reaches the
        schedulers as fast as the poll interval allows.
        """
        breakdown = self._get_ceiling_breakdown()
        ceiling = breakdown["hard_limit"]
        abort_limit = self._get_abort_limit_bytes()
        hot_cache_reserved = (
            self._hot_cache_reserved_bytes() if ceiling > 0 or abort_limit > 0 else 0
        )
        scheduler_ceiling = self._scheduler_limit_bytes(
            ceiling, reserved=hot_cache_reserved
        )
        soft_limit = (
            int(scheduler_ceiling * self._soft_threshold)
            if scheduler_ceiling > 0
            else 0
        )
        scheduler_abort_limit = self._scheduler_limit_bytes(
            abort_limit, reserved=hot_cache_reserved
        )
        admission_paused = self._pressure_level != "ok"
        for entry in self._engine_pool._entries.values():
            scheduler = self._resolve_scheduler(entry)
            if scheduler is None:
                engine = getattr(entry, "engine", None)
                if engine is None:
                    # Discovered-but-not-loaded entry. There is no
                    # scheduler to propagate to yet and that is normal,
                    # not a wrapper break, so skip silently. Warning here
                    # would fire on a routine startup before any model is
                    # loaded and turn the signal into noise.
                    continue
                if getattr(engine, "_loaded", True) is False:
                    # EnginePool clears entry.engine after stop(), but an
                    # enforcer tick can observe the engine in the small
                    # teardown window after its scheduler has been released.
                    # Treat that like an unloaded entry, not a wrapper break.
                    continue
                if (
                    type(engine).__name__ == "DFlashEngine"
                    and getattr(engine, "_fallback_engine", None) is None
                ):
                    continue
                if getattr(engine, "is_diffusion_model", False):
                    continue
                # Silent no-op was the failure mode that originally hid
                # the dead memory guard: a wrapper-chain change made
                # ``_resolve_scheduler()`` return None on a loaded engine
                # and the loop kept iterating without complaining. Surface
                # it now — once per engine type per enforcer lifetime so
                # the regression is loud in CI / oncall but a misconfigured
                # engine polled every second doesn't spam.
                engine_type = type(engine).__name__
                if engine_type not in self._scheduler_resolve_warned:
                    self._scheduler_resolve_warned.add(engine_type)
                    logger.warning(
                        "ProcessMemoryEnforcer: could not resolve "
                        "scheduler for engine type %s — prefill memory "
                        "guard will not propagate to this engine. "
                        "Verify the wrapper chain "
                        "(engine._engine.engine.scheduler) still holds.",
                        engine_type,
                    )
                continue
            scheduler._memory_limit_bytes = soft_limit
            scheduler._memory_hard_limit_bytes = scheduler_ceiling
            scheduler._memory_abort_limit_bytes = scheduler_abort_limit
            scheduler._prefill_abort_margin = self._get_prefill_abort_margin()
            # Propagate the component ceilings too so the rejection
            # message in ``_preflight_memory_check`` can name the binding
            # constraint and steer the user toward the right remedy
            # (close apps for dynamic, raise sysctl for metal_cap, raise
            # tier or reduce context for static).
            scheduler._memory_static_ceiling_bytes = breakdown["static"]
            scheduler._memory_dynamic_ceiling_bytes = breakdown["dynamic"]
            scheduler._memory_metal_cap_bytes = breakdown["metal_cap"]
            scheduler._memory_hot_cache_reserved_bytes = hot_cache_reserved
            # Tier name disambiguates dynamic = computed reclaimable
            # (safe/balanced/aggressive) from dynamic = user-pinned
            # custom_ceiling_bytes (custom). The advice ladder needs
            # the distinction to point at the right knob.
            scheduler._memory_guard_tier = self._memory_guard_tier
            scheduler._prefill_memory_guard = self._prefill_memory_guard
            scheduler._admission_paused = admission_paused
            scheduler._prefill_headroom_safety = self._prefill_headroom_safety
            scheduler._prefill_safe_zone_ratio = self._prefill_safe_zone_ratio
            scheduler._prefill_min_chunk_tokens = self._prefill_min_chunk_tokens
            bg = getattr(scheduler, "batch_generator", None)
            if bg is not None and hasattr(bg, "_memory_limit_bytes"):
                bg._memory_limit_bytes = soft_limit
                bg._memory_hard_limit_bytes = scheduler_ceiling

    def _walk_store_cache_caps(self) -> None:
        """Walk each scheduler's store-cache gate one step per poll (#1383).

        Driven on every enforcement tick, not just on pressure transitions,
        so the cap converges ±1 per poll toward its pressure-driven target
        (ok -> max_num_seqs, soft/hard -> 1). Decoupled from
        `_propagate_memory_limit` to avoid double-stepping the cap when
        a transition fires.
        """
        for entry in self._engine_pool._entries.values():
            scheduler = self._resolve_scheduler(entry)
            if scheduler is None:
                continue
            adjust = getattr(scheduler, "adjust_store_cache_cap", None)
            if adjust is not None:
                adjust(self._pressure_level)

    async def _abort_loaded_requests_for_memory_emergency(self) -> int:
        """Abort active requests on loaded models without unloading them."""
        aborted_total = 0
        for entry in self._engine_pool._entries.values():
            engine = getattr(entry, "engine", None)
            if engine is None:
                continue

            abort_all = getattr(engine, "abort_all_requests", None)
            if not callable(abort_all):
                continue

            try:
                result = abort_all()
                if inspect.isawaitable(result):
                    result = await result
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Emergency memory abort failed for '%s': %s",
                    getattr(entry, "model_id", "<unknown>"),
                    exc,
                )
                continue

            if isinstance(result, (int, float)):
                aborted_total += max(0, int(result))
        return aborted_total

    def _find_lru_busy_non_pinned_victim_locked(self) -> str | None:
        """Find a non-pinned loaded model that is busy but abortable.

        Caller must hold the engine-pool lock. This is used only at hard
        pressure, after idle victims have already been considered.
        """
        candidates: list[tuple[float, str]] = []
        for mid, entry in self._engine_pool._entries.items():
            if (
                getattr(entry, "engine", None) is None
                or getattr(entry, "is_pinned", False)
                or getattr(entry, "is_loading", False)
            ):
                continue

            busy = getattr(entry, "in_use", 0) > 0
            if not busy:
                engine = getattr(entry, "engine", None)
                has_active = getattr(engine, "has_active_requests", None)
                if callable(has_active):
                    try:
                        busy = has_active() is True
                    except Exception:  # noqa: BLE001
                        busy = True
            if busy:
                candidates.append((getattr(entry, "last_access", 0.0), mid))

        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]

    async def stop(self) -> None:
        """Stop the background enforcement loop."""
        self._running = False
        if self._wake_event is not None:
            self._wake_event.set()
        if self._task:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self._wake_event = None
        self._loop = None
        logger.info("Process memory enforcer stopped")

    async def _enforcement_loop(self) -> None:
        """Main polling loop."""
        self._loop = asyncio.get_running_loop()
        self._wake_event = asyncio.Event()
        while self._running:
            if self._wake_event is not None:
                self._wake_event.clear()
            try:
                await self._check_and_enforce()
                await self._check_ttl()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Process memory enforcer error: {e}")
            interval = self._select_poll_interval()
            self._current_poll_interval = interval
            if self._wake_event is None:
                await asyncio.sleep(interval)
                continue
            with suppress(TimeoutError):
                await asyncio.wait_for(self._wake_event.wait(), timeout=interval)

    def _select_poll_interval(self) -> float:
        """Choose the next polling interval from current engine activity."""
        if self._pressure_level != "ok":
            return self._active_poll_interval

        if time.monotonic() < self._activity_hint_until:
            return self._active_poll_interval

        has_loaded = False
        for entry in self._engine_pool._entries.values():
            if getattr(entry, "is_loading", False):
                return self._active_poll_interval

            engine = getattr(entry, "engine", None)
            if engine is None:
                continue

            has_loaded = True
            has_active_requests = getattr(engine, "has_active_requests", None)
            if not callable(has_active_requests):
                return self._active_poll_interval
            try:
                if has_active_requests():
                    return self._active_poll_interval
            except Exception:
                # If activity detection itself fails, bias toward safety.
                return self._active_poll_interval

        if has_loaded:
            return self._loaded_idle_poll_interval
        return self._unloaded_idle_poll_interval

    async def _check_ttl(self) -> None:
        """Check and unload models that exceeded their TTL."""
        if self._settings_manager is None:
            return
        await self._engine_pool.check_ttl_expirations(
            self._settings_manager,
            global_idle_timeout_seconds=(
                self._global_settings.idle_timeout.idle_timeout_seconds
                if self._global_settings
                else None
            ),
        )

    async def _check_and_enforce(self) -> None:
        """Check current memory and enforce 2-watermark policy.

        The ceiling is recomputed on every tick (dynamic ceiling moves
        with system_available), so watermarks shift as other apps take
        or release memory.

        Pressure levels:
        - ok (current < soft): no action, ensure admission unpaused.
        - soft (soft <= current < hard): LRU non-pinned eviction + signal
          schedulers to pause new admissions (in-flight requests proceed).
        - hard (current >= hard): LRU evict, abort a sole non-pinned
          victim's in-flight requests, abort in-progress model loads, and
          request idle reclaim when no victim exists. Pinned active requests
          are only aborted under emergency pressure beyond the real ceiling.
        """
        # Always propagate so the scheduler sees the latest ceiling /
        # admission_paused, even when usage stays below the soft mark.
        self._propagate_memory_limit()

        ceiling = self._get_hard_limit_bytes()
        if ceiling <= 0:
            self._pressure_level = "ok"
            self._over_ceiling_polls = 0
            return

        current = self._current_usage_bytes()
        soft = int(ceiling * self._soft_threshold)
        hard = int(ceiling * self._hard_threshold)
        prev_level = self._pressure_level
        emergency = self._is_emergency_pressure(current, ceiling)

        if current < soft:
            new_level = "ok"
        elif current < hard:
            new_level = "soft"
        else:
            new_level = "hard"

        if new_level != prev_level:
            self._pressure_level = new_level
            self._propagate_memory_limit()
            logger.info(
                f"Memory pressure level: {prev_level} -> {new_level} "
                f"(current={_format_gb(current)}, "
                f"soft={_format_gb(soft)}, hard={_format_gb(hard)}, "
                f"ceiling={_format_gb(ceiling)})"
            )

        if new_level == "hard":
            freed_hot = await asyncio.to_thread(
                self._shrink_hot_cache_for_pressure,
                current,
                soft,
            )
            if freed_hot > 0:
                current = self._current_usage_bytes()
                emergency = self._is_emergency_pressure(current, ceiling)
                if current < soft:
                    recovered_level = "ok"
                elif current < hard:
                    recovered_level = "soft"
                else:
                    recovered_level = "hard"
                if recovered_level != new_level:
                    logger.info(
                        "Memory pressure after hot-cache shrink: %s -> %s "
                        "(current=%s, freed_hot=%s)",
                        new_level,
                        recovered_level,
                        _format_gb(current),
                        _format_gb(freed_hot),
                    )
                    new_level = recovered_level
                    self._pressure_level = new_level
                    self._propagate_memory_limit()

        if new_level == "ok":
            # Still walk the store-cache cap so it can recover toward
            # max_num_seqs while pressure stays low (#1383).
            self._walk_store_cache_caps()
            return

        # Recover below soft regardless of level — prevents oscillation
        # at the boundary.
        target = soft

        async with self._engine_pool._lock:
            while self._current_usage_bytes() > target:
                pending = self._engine_pool._find_pending_unload_ready_locked()
                if pending is not None:
                    await self._engine_pool._unload_pending_if_idle_locked(pending)
                    continue

                victim = self._engine_pool._find_lru_victim()
                if victim is not None:
                    loaded_non_pinned = [
                        mid
                        for mid, e in self._engine_pool._entries.items()
                        if e.engine is not None and not e.is_pinned
                    ]
                    if new_level == "hard" or len(loaded_non_pinned) > 1:
                        # Evict idle LRU victims cleanly. At hard pressure even
                        # the last idle non-pinned model should be unloaded; it
                        # has no active KV to preserve and keeping it resident
                        # does not reduce pressure.
                        entry = self._engine_pool._entries.get(victim)
                        if (
                            entry
                            and entry.engine is not None
                            and hasattr(entry.engine, "abort_all_requests")
                        ):
                            result = await entry.engine.abort_all_requests()
                            aborted = max(0, int(result or 0))
                            if aborted > 0:
                                logger.warning(
                                    f"Aborted {aborted} requests on "
                                    f"'{victim}' before eviction"
                                )
                        logger.warning(
                            f"Evicting model '{victim}' (pressure={new_level})"
                        )
                        await self._engine_pool._unload_engine(victim)
                        continue

                    # soft: leave in-flight alone — admission pause already
                    # signaled, eviction can't help further without aborts.
                    break

                # No non-pinned victim. Loaded models are pinned or no loaded
                # engines exist at all.
                if new_level == "hard":
                    busy_victim = self._find_lru_busy_non_pinned_victim_locked()
                    if busy_victim is not None:
                        entry = self._engine_pool._entries.get(busy_victim)
                        aborted = 0
                        if (
                            entry
                            and entry.engine is not None
                            and hasattr(entry.engine, "abort_all_requests")
                        ):
                            aborted = await entry.engine.abort_all_requests()
                        if not emergency:
                            logger.warning(
                                "Hard memory pressure: aborted %d request(s) on "
                                "'%s' and kept model loaded",
                                aborted,
                                busy_victim,
                            )
                            break
                        if entry is not None:
                            self._engine_pool._mark_pending_unload_locked(
                                busy_victim,
                                "hard memory pressure",
                                abort_requested=True,
                            )
                            await self._engine_pool._unload_pending_if_idle_locked(
                                busy_victim
                            )
                        logger.warning(
                            "Hard memory pressure: requested abort/unload for "
                            "'%s' (aborted=%d)",
                            busy_victim,
                            aborted,
                        )
                        break

                    # Hard only: abort any in-progress model loads.
                    aborted_any = False
                    for entry in self._engine_pool._entries.values():
                        if entry.is_loading and not entry.abort_loading:
                            logger.warning(
                                f"Aborting in-progress load of "
                                f"'{entry.model_id}' (hard memory pressure)"
                            )
                            entry.abort_loading = True
                            aborted_any = True
                    if not aborted_any:
                        has_loaded = any(
                            e.engine is not None
                            for e in self._engine_pool._entries.values()
                        )
                        if has_loaded:
                            if emergency:
                                emergency_current = self._current_usage_bytes()
                            else:
                                emergency_current = 0
                            if emergency and emergency_current >= ceiling:
                                aborted = await (
                                    self._abort_loaded_requests_for_memory_emergency()
                                )
                                if aborted > 0:
                                    logger.warning(
                                        "Emergency memory pressure: aborted "
                                        "%d in-flight request(s) "
                                        "(current=%s, ceiling=%s); models "
                                        "kept loaded.",
                                        aborted,
                                        _format_gb(emergency_current),
                                        _format_gb(ceiling),
                                    )
                                    break

                            # Nothing to evict (all pinned) and no load to
                            # abort — but the resident footprint may still hold
                            # reclaimable Metal transients from a finished turn.
                            # Ask each loaded scheduler to trim them between
                            # turns. This only sets a flag; the actual reclaim
                            # runs on the inference thread when it is idle, so
                            # we never touch Metal from the enforcer thread.
                            requested = 0
                            for entry in self._engine_pool._entries.values():
                                sched = self._resolve_scheduler(entry)
                                if sched is not None and hasattr(
                                    sched, "request_idle_reclaim"
                                ):
                                    sched.request_idle_reclaim()
                                    requested += 1
                            logger.warning(
                                "Hard memory pressure, no evictable models "
                                "and no loads in progress: requested "
                                "idle reclaim on %d scheduler(s).",
                                requested,
                            )
                        else:
                            logger.warning("Hard memory pressure but no models loaded.")
                # soft + all pinned: nothing to do beyond admission pause.
                break

        # Re-evaluate level after eviction completes so admission state
        # reflects post-eviction reality on the next propagate.
        post_current = self._current_usage_bytes()
        # Recompute ceiling again — eviction may free phys, shifting the
        # dynamic ceiling.
        post_ceiling = self._get_hard_limit_bytes()
        post_soft = int(post_ceiling * self._soft_threshold) if post_ceiling > 0 else 0
        post_hard = int(post_ceiling * self._hard_threshold) if post_ceiling > 0 else 0
        if post_ceiling <= 0 or post_current < post_soft:
            post_level = "ok"
        elif post_current < post_hard:
            post_level = "soft"
        else:
            post_level = "hard"
        if post_ceiling <= 0 or post_current < post_ceiling:
            self._over_ceiling_polls = 0
        if post_level != self._pressure_level:
            self._pressure_level = post_level
            self._propagate_memory_limit()
            logger.info(
                f"Memory pressure post-eviction: {new_level} -> {post_level} "
                f"(current={_format_gb(post_current)})"
            )

        # Walk each scheduler's store-cache gate ±1 toward its
        # pressure-driven target every poll (#1383).
        self._walk_store_cache_caps()

    def get_status(self) -> dict:
        """Get enforcer status for monitoring endpoints.

        Reports the same `max(active, phys_footprint)` value the enforcer
        uses internally so admin UI / /health utilization matches the
        watermark the enforcer is actually comparing against.
        """
        ceiling = self._get_hard_limit_bytes() if self._running else 0
        static_ceiling = self._get_static_ceiling() if self._running else 0
        dynamic_ceiling = self._get_dynamic_ceiling() if self._running else 0
        current = self._current_usage_bytes() if self._running else 0
        hot_reserved = self._hot_cache_reserved_bytes() if self._running else 0
        scheduler_ceiling = (
            self._scheduler_limit_bytes(ceiling, reserved=hot_reserved)
            if self._running
            else 0
        )
        scheduler_abort = (
            self._scheduler_limit_bytes(
                self._get_abort_limit_bytes(), reserved=hot_reserved
            )
            if self._running
            else 0
        )
        soft = int(ceiling * self._soft_threshold) if ceiling > 0 else 0
        hard = int(ceiling * self._hard_threshold) if ceiling > 0 else 0
        return {
            "enabled": self._running,
            "memory_guard_tier": self._memory_guard_tier,
            "memory_guard_custom_ceiling_bytes": self._memory_guard_custom_ceiling_bytes,
            "ceiling_bytes": ceiling,
            "ceiling_formatted": _format_gb(ceiling),
            "static_ceiling_bytes": static_ceiling,
            "static_ceiling_formatted": _format_gb(static_ceiling),
            "dynamic_ceiling_bytes": dynamic_ceiling,
            "dynamic_ceiling_formatted": _format_gb(dynamic_ceiling),
            "hot_cache_reserved_bytes": hot_reserved,
            "hot_cache_reserved_formatted": _format_gb(hot_reserved),
            "scheduler_ceiling_bytes": scheduler_ceiling,
            "scheduler_ceiling_formatted": _format_gb(scheduler_ceiling),
            "scheduler_abort_limit_bytes": scheduler_abort,
            "scheduler_abort_limit_formatted": _format_gb(scheduler_abort),
            "soft_threshold": self._soft_threshold,
            "hard_threshold": self._hard_threshold,
            "soft_bytes": soft,
            "soft_formatted": _format_gb(soft),
            "hard_bytes": hard,
            "hard_formatted": _format_gb(hard),
            "current_bytes": current,
            "current_formatted": _format_gb(current),
            "pressure_level": self._pressure_level if self._running else "ok",
            "utilization": (current / ceiling if ceiling > 0 else 0.0),
            "poll_interval_seconds": (
                self._current_poll_interval if self._running else 0.0
            ),
        }
