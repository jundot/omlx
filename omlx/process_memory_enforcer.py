# SPDX-License-Identifier: Apache-2.0
"""
Process-level memory enforcer for oMLX.

The enforcer derives a hard ceiling from the configured memory_guard_tier
(safe / balanced / aggressive) and the current system state, then drives
soft / hard watermarks from that ceiling. When usage crosses a watermark
it unloads LRU models from EnginePool and pauses admission for new
prefills.

Ceiling = min(static_ceiling, dynamic_ceiling):
  static_ceiling  = total_ram - tier.static_reserve
  dynamic_ceiling = omlx_phys_footprint + system_available - tier.other_app_reserve

static_ceiling caps the absolute Metal pressure. dynamic_ceiling moves
in real time with system_available so that when other apps grab memory
the ceiling drops immediately and LRU eviction or scheduler abort fires.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import psutil

from .utils.proc_memory import get_phys_footprint

if TYPE_CHECKING:
    from .engine_pool import EnginePool
    from .model_settings import ModelSettingsManager
    from .settings import GlobalSettings

logger = logging.getLogger(__name__)


# Reserve below 16 GB systems regardless of tier — small Macs cannot
# afford a tier-scaled cut and still load any useful model.
_SMALL_SYSTEM_RESERVE = 4 * 1024**3
_SMALL_SYSTEM_THRESHOLD = 16 * 1024**3

# Tier maps: static reserve (>= 16 GB systems) + real-time spike buffer.
_STATIC_RESERVE_LARGE: dict[str, int] = {
    "safe": 12 * 1024**3,  # aligned with Apple iogpu.wired_limit 75%
    "balanced": 8 * 1024**3,
    "aggressive": 6 * 1024**3,
}
_OTHER_APP_RESERVE: dict[str, int] = {
    "safe": 2 * 1024**3,
    "balanced": 1 * 1024**3,
    "aggressive": 512 * 1024**2,
}


def _format_gb(b: int) -> str:
    """Format bytes as GB string."""
    return f"{b / 1024**3:.1f}GB"


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


def _apply_metal_wired_limit(desired_bytes: int) -> tuple[int, int | None]:
    """Try to raise Metal wired limit for this process to `desired_bytes`.

    Returns (applied_bytes, previous_bytes). `applied_bytes` is what we
    actually told MLX (clamped to the kernel iogpu.wired_limit_mb if it
    is lower); `previous_bytes` is what MLX reports the prior limit was,
    or None on failure / older macOS where the call is unavailable.

    Emits a WARNING when the kernel sysctl caps us below `desired_bytes`
    so the user sees the hint in logs in addition to the admin UI red
    banner.

    `mx.set_wired_limit` raises when asked for more than the kernel
    sysctl allows, so we clamp before calling.
    """
    if desired_bytes <= 0:
        return 0, None
    effective_cap = get_effective_metal_cap_bytes()
    capped = effective_cap > 0 and effective_cap < desired_bytes
    applied = effective_cap if capped else desired_bytes
    try:
        previous = mx.set_wired_limit(applied)
        if capped:
            source = (
                "kernel iogpu.wired_limit_mb"
                if get_iogpu_wired_limit_bytes() > 0
                else "Apple max_recommended_working_set_size"
            )
            logger.warning(
                "Metal cap (%s, %s) is below the oMLX static ceiling (%s); "
                "Metal will clamp allocations to the cap and panic if a "
                "request exceeds it. Raise it with: sudo sysctl "
                "iogpu.wired_limit_mb=%d",
                _format_gb(effective_cap),
                source,
                _format_gb(desired_bytes),
                desired_bytes // (1024**2),
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
        poll_interval: float = 1.0,
        settings_manager: ModelSettingsManager | None = None,
        prefill_memory_guard: bool = True,
        global_settings: GlobalSettings | None = None,
        soft_threshold: float = 0.85,
        hard_threshold: float = 0.95,
        prefill_safe_zone_ratio: float = 0.80,
        prefill_min_chunk_tokens: int = 32,
    ):
        """
        Initialize the process memory enforcer.

        Args:
            engine_pool: The engine pool to evict models from.
            memory_guard_tier: One of "safe", "balanced", "aggressive".
                Selects static_reserve (>= 16 GB systems) and
                other_app_reserve buffers. Below 16 GB systems always
                use a 4 GB static reserve regardless of tier.
            poll_interval: Seconds between memory checks.
            settings_manager: Optional settings manager for TTL checks.
            prefill_memory_guard: When False, returns a ceiling of 0 so
                callers treat the limit as disabled.
            global_settings: Optional global settings for idle timeout.
            soft_threshold: Fraction of ceiling that triggers soft action
                (LRU non-pinned eviction + admission pause; in-flight allowed).
            hard_threshold: Fraction of ceiling that triggers hard action
                (also abort in-flight when all loaded models are pinned).
            prefill_safe_zone_ratio: Fraction of hard cap below which prefill
                runs at full chunk size; above triggers adaptive shrink.
            prefill_min_chunk_tokens: Floor for adaptive shrink.
        """
        self._engine_pool = engine_pool
        self._memory_guard_tier = self._normalize_tier(memory_guard_tier)
        self._poll_interval = poll_interval
        self._settings_manager = settings_manager
        self._prefill_memory_guard = prefill_memory_guard
        self._global_settings = global_settings
        self._soft_threshold = soft_threshold
        self._hard_threshold = hard_threshold
        self._prefill_safe_zone_ratio = prefill_safe_zone_ratio
        self._prefill_min_chunk_tokens = prefill_min_chunk_tokens
        self._task: asyncio.Task | None = None
        self._running = False
        # Most recently observed pressure level, consumed by scheduler /
        # admission control. Updated on every poll iteration.
        self._pressure_level: str = "ok"
        # Last value passed to mx.set_wired_limit (0 if not yet applied
        # or the call failed). Used by the admin dashboard to surface a
        # warning when the kernel iogpu.wired_limit_mb is below this.
        self._metal_wired_limit_request: int = 0

    @staticmethod
    def _normalize_tier(tier: str) -> str:
        t = (tier or "").strip().lower()
        if t not in _STATIC_RESERVE_LARGE:
            return "balanced"
        return t

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
        if self._running:
            self._propagate_memory_limit()
        logger.info(f"Memory guard tier changed: {old} -> {new_tier}")

    @property
    def is_running(self) -> bool:
        """Whether the enforcement loop is active."""
        return self._running

    def start(self) -> None:
        """Start the background enforcement loop.

        Also raises this process's Metal wired-memory limit to the static
        ceiling so allocations within ceiling don't bounce off Apple's
        default ~75% cap. Static ceiling is used (not dynamic) because
        dynamic shrinks with other-app pressure and would oscillate the
        Metal limit if used here.
        """
        if self._running:
            return
        self._running = True
        self._propagate_memory_limit()
        ceiling = self._get_hard_limit_bytes()

        if self._prefill_memory_guard:
            static_ceiling = self._get_static_ceiling()
            applied, previous = _apply_metal_wired_limit(static_ceiling)
            # Store the *desired* limit (= static ceiling) rather than the
            # post-clamp applied value. The admin UI compares this against
            # the live iogpu.wired_limit_mb so a kernel cap below the
            # desired limit triggers the red sysctl-command banner.
            self._metal_wired_limit_request = static_ceiling
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
            f"interval={self._poll_interval}s)"
        )

    def _get_static_ceiling(self) -> int:
        """Total RAM minus tier-scaled static reserve."""
        from .settings import get_system_memory

        system_bytes = get_system_memory()
        if system_bytes < _SMALL_SYSTEM_THRESHOLD:
            reserve = _SMALL_SYSTEM_RESERVE
        else:
            reserve = _STATIC_RESERVE_LARGE[self._memory_guard_tier]
        return max(0, system_bytes - reserve)

    def _get_dynamic_ceiling(self) -> int:
        """Real-time ceiling based on `psutil.virtual_memory().available`.

        Recomputed on every call — never cached — so other-app spikes
        (Chrome / Xcode / video editor) shrink the ceiling within one
        poll. other_app_reserve absorbs sub-1s spikes that fall between
        polling ticks.
        """
        omlx_usage = get_phys_footprint()
        system_available = psutil.virtual_memory().available
        buffer = _OTHER_APP_RESERVE[self._memory_guard_tier]
        return max(0, omlx_usage + system_available - buffer)

    def _get_hard_limit_bytes(self) -> int:
        """Final hard ceiling = min(static, dynamic, metal_cap).

        `metal_cap` is the effective Metal allocation cap (kernel
        iogpu.wired_limit_mb when set, otherwise Apple's
        max_recommended_working_set_size). Including it here means oMLX
        never plans allocations above what Metal will actually accept,
        so users who have not raised iogpu.wired_limit_mb still get a
        safe (smaller) ceiling rather than a panic.

        Returns 0 if the memory guard is disabled (callers treat 0 as
        "no limit").
        """
        if not self._prefill_memory_guard:
            return 0
        candidates = [
            self._get_static_ceiling(),
            self._get_dynamic_ceiling(),
        ]
        metal_cap = get_effective_metal_cap_bytes()
        if metal_cap > 0:
            candidates.append(metal_cap)
        return min(candidates)

    def get_final_ceiling(self) -> int:
        """Public accessor used by engine_pool pre-load admission."""
        return self._get_hard_limit_bytes()

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

        Combines MLX-reported active memory and the kernel phys_footprint
        ledger. phys_footprint covers anonymous + IOAccelerator + dirty
        file-backed, so it usually dominates; we take max() so MLX-internal
        cache that hasn't been mirrored into phys yet still triggers.
        """
        return max(mx.get_active_memory(), get_phys_footprint())

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
        """Resolve the Scheduler instance from an EnginePool entry.

        Most engines (BatchedEngine, VLMBatchedEngine) wrap the scheduler
        as ``entry.engine._engine.engine.scheduler`` (AsyncEngineCore →
        EngineCore → Scheduler). Some non-streaming engines may expose
        ``entry.engine.scheduler`` directly. Returns None if neither
        path resolves.
        """
        eng = entry.engine
        if eng is None:
            return None
        sched = getattr(eng, "scheduler", None)
        if sched is not None:
            return sched
        inner = getattr(eng, "_engine", None)
        if inner is None:
            return None
        inner_engine = getattr(inner, "engine", None)
        if inner_engine is None:
            return None
        return getattr(inner_engine, "scheduler", None)

    def _propagate_memory_limit(self) -> None:
        """Propagate ceiling-derived watermarks to all schedulers.

        Called on every enforcer tick so the dynamic ceiling reaches the
        schedulers as fast as the poll interval allows.
        """
        ceiling = self._get_hard_limit_bytes()
        soft_limit = int(ceiling * self._soft_threshold) if ceiling > 0 else 0
        admission_paused = self._pressure_level != "ok"
        for entry in self._engine_pool._entries.values():
            scheduler = self._resolve_scheduler(entry)
            if scheduler is not None:
                scheduler._memory_limit_bytes = soft_limit
                scheduler._memory_hard_limit_bytes = ceiling
                scheduler._prefill_memory_guard = self._prefill_memory_guard
                scheduler._admission_paused = admission_paused
                scheduler._prefill_safe_zone_ratio = self._prefill_safe_zone_ratio
                scheduler._prefill_min_chunk_tokens = self._prefill_min_chunk_tokens
                bg = getattr(scheduler, "batch_generator", None)
                if bg is not None and hasattr(bg, "_memory_limit_bytes"):
                    bg._memory_limit_bytes = soft_limit
                    bg._memory_hard_limit_bytes = ceiling

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

    async def stop(self) -> None:
        """Stop the background enforcement loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Process memory enforcer stopped")

    async def _enforcement_loop(self) -> None:
        """Main polling loop."""
        while self._running:
            try:
                await self._check_and_enforce()
                await self._check_ttl()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Process memory enforcer error: {e}")
            await asyncio.sleep(self._poll_interval)

    async def _check_ttl(self) -> None:
        """Check and unload models that exceeded their TTL."""
        if self._settings_manager is None:
            return
        await self._engine_pool.check_ttl_expirations(
            self._settings_manager,
            global_idle_timeout_seconds=(
                self._global_settings.idle_timeout.idle_timeout_seconds
                if self._global_settings else None
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
        - hard (current >= hard): full enforcement — LRU evict, abort
          in-flight when only pinned remain, abort in-progress model loads.
        """
        # Always propagate so the scheduler sees the latest ceiling /
        # admission_paused, even when usage stays below the soft mark.
        self._propagate_memory_limit()

        ceiling = self._get_hard_limit_bytes()
        if ceiling <= 0:
            self._pressure_level = "ok"
            return

        current = self._current_usage_bytes()
        soft = int(ceiling * self._soft_threshold)
        hard = int(ceiling * self._hard_threshold)
        prev_level = self._pressure_level

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
                victim = self._engine_pool._find_lru_victim()
                if victim is not None:
                    loaded_non_pinned = [
                        mid
                        for mid, e in self._engine_pool._entries.items()
                        if e.engine is not None and not e.is_pinned
                    ]
                    if len(loaded_non_pinned) > 1:
                        # Multiple non-pinned: evict LRU victim cleanly.
                        # abort_all_requests is fired before _unload_engine
                        # so clients receive proper error responses instead
                        # of silent disconnect.
                        entry = self._engine_pool._entries.get(victim)
                        if entry and entry.engine is not None:
                            if hasattr(entry.engine, "abort_all_requests"):
                                aborted = await entry.engine.abort_all_requests()
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

                    # Only one non-pinned model remains.
                    if new_level == "hard":
                        # Abort in-flight requests, keep model loaded —
                        # frees KV blocks so short-context follow-ups work.
                        entry = self._engine_pool._entries.get(victim)
                        if entry and entry.engine is not None:
                            if hasattr(entry.engine, "abort_all_requests"):
                                aborted = await entry.engine.abort_all_requests()
                                if aborted > 0:
                                    logger.warning(
                                        f"Aborted {aborted} requests on "
                                        f"'{victim}' due to hard memory "
                                        f"pressure (model kept loaded)"
                                    )
                    # soft: leave in-flight alone — admission pause already
                    # signaled, eviction can't help further without aborts.
                    break

                # No non-pinned victim. All loaded models are pinned.
                if new_level == "hard":
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
                            logger.warning(
                                "Hard memory pressure but all loaded models "
                                "are pinned and no loads in progress."
                            )
                        else:
                            logger.warning(
                                "Hard memory pressure but no models loaded."
                            )
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
        soft = int(ceiling * self._soft_threshold) if ceiling > 0 else 0
        hard = int(ceiling * self._hard_threshold) if ceiling > 0 else 0
        return {
            "enabled": self._running,
            "memory_guard_tier": self._memory_guard_tier,
            "ceiling_bytes": ceiling,
            "ceiling_formatted": _format_gb(ceiling),
            "static_ceiling_bytes": static_ceiling,
            "static_ceiling_formatted": _format_gb(static_ceiling),
            "dynamic_ceiling_bytes": dynamic_ceiling,
            "dynamic_ceiling_formatted": _format_gb(dynamic_ceiling),
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
        }
