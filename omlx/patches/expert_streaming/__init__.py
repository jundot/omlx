# SPDX-License-Identifier: Apache-2.0
"""Expert streaming (SSD) patch for MoE models (glm_moe_dsa, deepseek_v4, ...)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Re-exported from the leaf modules — they hold the single copy, so the
# gates cannot drift apart. Both names stay importable from the package
# root; there is exactly one definition.
from .model_hooks import (
    DEFAULT_PREFIX_TEMPLATES as DEFAULT_PREFIX_TEMPLATES,
    find_moe_container as find_moe_container,
    find_moe_owner as find_moe_owner,
    find_mtp_stages as find_mtp_stages,
    hooks_for as hooks_for,
)
from .residency import SUPPORTED_TYPES, normalize_model_type

try:
    # Public cache-policy API at the package root.
    from .governor import dynamic_residency_enabled
except Exception:  # pragma: no cover - governor has no mlx dependency
    dynamic_residency_enabled = lambda: False  # type: ignore[assignment]

try:
    from .streaming_switch import S3FIFOExpertCache, make_expert_cache
except Exception:  # pragma: no cover - streaming_switch imports mlx
    S3FIFOExpertCache = None  # type: ignore[assignment]
    make_expert_cache = None  # type: ignore[assignment]

# Conversion pipeline lives in conversion.py (stacked-key resolution, the
# switch-MLP rewrite, the orchestrator, transition-profile persistence).
# These names stay importable from the package root: tests and engines
# import them from here, and shutdown_expert_streaming calls
# save_transition_profile below. conversion.py top-level only touches leaf
# modules, so this cannot cycle.
from .conversion import (
    _candidate_stacked_keys as _candidate_stacked_keys,
    _mtp_candidate_stacked_keys as _mtp_candidate_stacked_keys,
    _resolve_moe_dims as _resolve_moe_dims,
    _resolve_stacked_key as _resolve_stacked_key,
    convert_model_to_streaming as convert_model_to_streaming,
    load_transition_profile as load_transition_profile,
    save_transition_profile as save_transition_profile,
)


def is_supported_model_type(model_type: str | None) -> bool:
    if not model_type:
        return False
    return normalize_model_type(model_type) in SUPPORTED_TYPES


# Upper bound on the expert LRU budget. The admin PUT handler rejects
# anything outside 0-64 GiB (routes.py); the loader path -- a hand-edited
# settings file, an autotune --apply, or an env override -- bypasses that
# check, so clamp here too.
MAX_EXPERT_STREAMING_BUDGET_BYTES = 64 * 1024**3


def _clamp_budget_bytes(raw: float) -> int:
    """Clamp a raw byte budget to [0, 64 GiB]. 0 means page-cache only."""
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return 0
    return max(0, min(MAX_EXPERT_STREAMING_BUDGET_BYTES, val))


def _dynamic_armed(setting_val: Any | None, model_settings: Any | None) -> bool:
    """Resolve the dynamic-governor switch (user-testable auto rule).

    Explicit setting wins (True forces on over a pinned budget, False opts
    out); then the OMLX_EXPERT_STREAMING_DYNAMIC env; then the auto rule:
    on when the budget itself is automatic (no explicit pin), off when the
    user pinned a budget (manual mode stays put unless forced).
    """
    if setting_val is not None:
        return bool(setting_val)
    try:
        if dynamic_residency_enabled():
            return True
    except Exception:
        pass
    return not _budget_is_pinned(model_settings)


def _budget_is_pinned(model_settings: Any | None) -> bool:
    """True when the user pinned an explicit budget (manual mode)."""
    if model_settings is None:
        return False
    for attr in (
        "expert_streaming_budget_gib",
        "expert_cache_budget_gib",
        "expert_streaming_budget_mib",
        "expert_cache_budget_mib",
    ):
        if getattr(model_settings, attr, None) is not None:
            return True
    return False


def _auto_budget_bytes() -> int:
    """RAM-scaled starting budget for the auto stack (~5% of total RAM).

    Only the STARTING point: the dynamic governor adapts it at runtime
    (grow on proven hunger, shrink on pressure). Clamped to [0.5, 4] GiB
    so small machines stay safe and big ones do not pre-grab memory the
    model weights may still need during load.
    """
    try:
        import psutil

        total = int(psutil.virtual_memory().total)
    except Exception:
        total = 0
    if total <= 0:
        try:
            from .governor import _total_ram_bytes

            total = int(_total_ram_bytes())
        except Exception:
            total = 0
    if total <= 0:
        return int(1.0 * 1024**3)
    return _clamp_budget_bytes(
        max(0.5 * 1024**3, min(4.0 * 1024**3, int(total * 0.05)))
    )


def resolve_budget_bytes(model_settings: Any | None) -> int:
    """Resolved app-level LRU budget for the streaming heap.

    Public (admission path): ``engine_pool`` charges exactly this — the
    real commitment a streaming load makes — instead of a fraction-based
    heuristic. Settings precedence: explicit GiB > explicit MiB > the
    auto stack (RAM-scaled, governor-adapted) > 0 when
    ``expert_streaming_budget_auto`` is False or no settings object
    exists.
    """
    return _resolve_budget_bytes(model_settings)


def _resolve_budget_bytes(model_settings: Any | None) -> int:
    if model_settings is not None:
        # Preferred name (model_settings.py:222) + legacy cache name.
        # Explicit 0 = page-cache only (no app-level LRU); None falls through
        # to the auto stack below.
        for attr in ("expert_streaming_budget_gib", "expert_cache_budget_gib"):
            gib = getattr(model_settings, attr, None)
            if gib is not None:
                try:
                    return _clamp_budget_bytes(float(gib) * 1024**3)
                except (TypeError, ValueError):
                    continue
        # legacy mib
        for attr in ("expert_streaming_budget_mib", "expert_cache_budget_mib"):
            mib = getattr(model_settings, attr, None)
            if mib is not None:
                try:
                    if int(mib) <= 0:
                        continue
                except (TypeError, ValueError):
                    continue
                return _clamp_budget_bytes(int(mib) * 1024 * 1024)
        # Auto (default): RAM-scaled starting budget for the dynamic
        # governor. False opts out to page-cache only (the OS file cache
        # serves reuse from clean pages).
        if getattr(model_settings, "expert_streaming_budget_auto", True) is False:
            return 0
        return _auto_budget_bytes()
    # No settings object (bench direct path): stay page-cache only unless
    # the caller passes budget_bytes explicitly.
    return 0


def _prior_usable(cache: Any) -> bool:
    """Cache-prior needs app-level LRU residency as its signal.

    With a page-cache-only budget the resident set is always empty and the
    rerank is pure overhead — refuse it so budget-0 stays on the stock
    path."""
    try:
        return int(getattr(cache, "capacity", 0) or 0) > 0
    except (TypeError, ValueError):
        return False


# Per-model streaming IO override keys (None = unset: keep the env-var /
# built-in default). Validators below normalize the value or drop it back
# to None when it is unusable.
#
# NOT here on purpose: expert_streaming_cache_prior,
# expert_streaming_cold_tier, expert_streaming_topk_threshold. They are
# routing/tier knobs, not IO-path overrides — their consumers read
# model_settings through dedicated resolvers (adaptive_topk.resolve_*,
# conversion._resolve_cold_tier_root) that each coerce defensively and
# fall back to env/exact on garbage, and the admin PUT validates them at
# the write boundary. Listing them here would create a second, divergent
# contract.
_IO_OVERRIDE_KEYS = (
    "expert_streaming_io_depth",
    "expert_streaming_coalesce",
    "expert_streaming_readahead",
    "expert_streaming_seed",
    "expert_streaming_per_layer_eval",
    "expert_streaming_pins",
    "expert_streaming_hot_fraction",
    "expert_streaming_pin_gib",
    "expert_streaming_pin_sync",
    "expert_streaming_pin_regime",
    "expert_streaming_cache_policy",
    "expert_streaming_dynamic",
    "expert_streaming_dynamic_max_gib",
    "expert_streaming_dynamic_min_gib",
    "expert_streaming_dynamic_stall_target",
    "expert_streaming_prefill_budget_gib",
)


def _clamped_int(value: Any, lo: int, hi: int) -> int | None:
    """Coerce to int; below lo is invalid, above hi clamps."""
    # bools are not numbers for these knobs: True must not silently
    # become 1 GiB of pinned budget (admin rejects bools too).
    if isinstance(value, bool):
        return None
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    return max(lo, min(hi, v)) if v >= lo else None


def _bounded_float(value: Any, lo: float, hi: float, lo_open: bool) -> float | None:
    """Coerce to float accepted only inside (lo, hi] or [lo, hi]."""
    if isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    ok = (lo < v <= hi) if lo_open else (lo <= v <= hi)
    return v if ok else None


def _policy_choice(value: Any) -> str | None:
    p = str(value).strip().lower()
    return p if p in ("lru", "s3fifo") else None


def _pin_regime_choice(value: Any) -> str | None:
    # Consumed as a lowercase regime name (warmer.PinController.pin_regime).
    p = str(value).strip().lower()
    return p if p in ("decode", "prefill") else None


def _nonneg_float(value: Any) -> float | None:
    """Coerce to float accepted only for finite values >= 0."""
    if isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if 0 <= v < float("inf") else None


_IO_OVERRIDE_VALIDATORS = {
    "expert_streaming_io_depth": lambda v: _clamped_int(v, 1, 64),
    "expert_streaming_cache_policy": _policy_choice,
    "expert_streaming_hot_fraction": lambda v: _bounded_float(v, 0, 1, False),
    "expert_streaming_pin_gib": lambda v: _bounded_float(v, 0, 64, True),
    "expert_streaming_pin_regime": _pin_regime_choice,
    "expert_streaming_dynamic_max_gib": lambda v: _bounded_float(v, 0, 64, True),
    "expert_streaming_dynamic_min_gib": lambda v: _bounded_float(v, 0, 64, False),
    "expert_streaming_dynamic_stall_target": lambda v: _bounded_float(v, 0.0, 0.9, False),
    "expert_streaming_prefill_budget_gib": lambda v: _bounded_float(v, 0, 64, True),
}


def _io_overrides(model_settings: Any | None) -> dict[str, Any]:
    """Per-model streaming IO overrides with env-fallback semantics.

    Returns a dict whose values are None when the setting is unset (keep the
    env-var / built-in default) or the requested override otherwise.
    """
    raw = {
        key: getattr(model_settings, key, None) if model_settings is not None else None
        for key in _IO_OVERRIDE_KEYS
    }
    for key, validate in _IO_OVERRIDE_VALIDATORS.items():
        if raw[key] is not None:
            raw[key] = validate(raw[key])
    return raw



def expert_streaming_summary(cache: Any, backing: Any | None = None) -> dict:
    """One-line request/bench summary of streaming health.

    Aggregates the counters the implementation already keeps (LRU hits,
    advisor, ctx fallbacks) into a single
    dict for per-request logs and the admin payload. Never raises;
    missing pieces report as None/0.
    """
    out: dict = {}
    try:
        stats = getattr(cache, "stats", None)
        hits = int(getattr(stats, "hits", 0) or 0)
        misses = int(getattr(stats, "misses", 0) or 0)
        out["lru_hit_rate"] = hits / (hits + misses) if (hits + misses) else 0.0
        out["lru_hits"] = hits
        out["lru_misses"] = misses
        out["lru_evictions"] = int(getattr(stats, "evictions", 0) or 0)
        out["lru_size"] = int(getattr(cache, "size", 0) or 0)
        out["lru_capacity"] = int(getattr(cache, "capacity", 0) or 0)
    except Exception:
        pass
    try:
        # Partitioned backends (V4.1 dedicated adapter, legacy
        # fetch-on-miss) have no app-level LRU cache — the backing's own
        # summary carries the counters, folded into the lru_* slots the
        # request log reads so the line stops reporting a permanent 0.
        if cache is None and backing is not None:
            bsum = getattr(backing, "summary", None)
            if callable(bsum):
                b = bsum()
                if isinstance(b, dict):
                    out["backing"] = b
                    hits = int(b.get("hits", 0) or 0)
                    misses = int(b.get("misses", 0) or 0)
                    out["lru_hit_rate"] = (
                        hits / (hits + misses) if (hits + misses) else 0.0
                    )
                    out["lru_hits"] = hits
                    out["lru_misses"] = misses
                    out["lru_evictions"] = int(b.get("evictions", 0) or 0)
                    out["lru_size"] = int(b.get("resident", 0) or 0)
                    out["lru_capacity"] = int(
                        b.get("capacity_per_layer", 0) or 0
                    ) * int(b.get("layers", 0) or 0)
    except Exception:
        pass
    try:
        spec = getattr(cache, "spec_state", None)
        sstats = getattr(spec, "stats", None) if spec is not None else None
        if sstats is not None:
            out["advised"] = int(sstats.get("advised", 0) or 0)
    except Exception:
        pass
    try:
        out["ctx_fallbacks"] = dict(getattr(cache, "ctx_fallback_stats", lambda: {})())
    except Exception:
        out["ctx_fallbacks"] = {}
    try:
        gov = getattr(backing, "governor", None) if backing is not None else None
        if gov is not None:
            out["governor"] = gov.summary()
    except Exception:
        pass
    try:
        out["cache_policy"] = getattr(cache, "policy", "lru")
    except Exception:
        pass
    try:
        spec = getattr(cache, "spec_state", None)
        if spec is not None:
            out["trans_updates"] = int(getattr(spec, "trans_updates", 0) or 0)
            out["trans_sources"] = len(getattr(spec, "trans", {}) or {})
            sstats = getattr(spec, "stats", None) or {}
            if isinstance(sstats, dict) and sstats.get("trans_overfetch"):
                out["trans_overfetch"] = int(sstats["trans_overfetch"])
    except Exception:
        pass
    return out


def resolve_streaming_backing(engine: Any) -> Any | None:
    """Return the engine's streaming backing, walking the holder chain.

    The unified converter stamps the backing on engine and model; the
    legacy adapter (and the alias' streaming fallback) stamps its store
    on the model only. Used by the engine stop paths so legacy fds/mmaps
    release on teardown instead of whenever GC gets around to it.
    """
    backing = getattr(engine, "_expert_streaming_backing", None)
    if backing is None:
        for holder in (
            getattr(engine, "_model", None),
            getattr(engine, "_vlm_model", None),
        ):
            backing = getattr(holder, "_expert_streaming_backing", None)
            if backing is not None:
                break
    return backing


def streaming_summary_backing(engine: Any) -> Any | None:
    """The object owning governor + summary for this engine.

    The legacy adapter aggregates its per-layer caches under a
    governor-facing state stamped as ``_moe_offload_legacy_state`` (same
    duck-type as the V4.1 backing); when present it overrides the plain
    backing for reporting.
    """
    for holder in (
        engine,
        getattr(engine, "_model", None),
        getattr(engine, "_vlm_model", None),
    ):
        state = getattr(holder, "_moe_offload_legacy_state", None)
        if state is not None:
            return state
    return resolve_streaming_backing(engine)


def log_expert_streaming_summary(
    engine: Any, *, prompt_tokens: int = 0, completion_tokens: int = 0
) -> None:
    """One-line MoE streaming health log per completed request.

    No-op unless expert streaming is active. Also drives the dynamic
    governor's per-request observation. Never raises.
    """
    try:
        backing = streaming_summary_backing(engine)
        if backing is None:
            return
        # Dynamic residency: revisit cache capacity from free memory
        # once per request boundary (opt-in; never raises).
        governor = getattr(backing, "governor", None)
        if governor is not None:
            try:
                action = governor.observe()
            except Exception:
                logger.debug("governor observe failed", exc_info=True)
            else:
                logger.info("expert_streaming governor: %s", action)
        cache = getattr(backing, "_streaming_cache", None)
        summary = expert_streaming_summary(cache, backing)
        if not summary:
            return
        logger.info(
            "expert_streaming req prompt=%d completion=%d lru_hit=%.3f "
            "(h=%d m=%d evict=%d size=%d/%d) advised=%d "
            "ctx_fallbacks=%s",
            prompt_tokens,
            completion_tokens,
            summary.get("lru_hit_rate", 0.0),
            summary.get("lru_hits", 0),
            summary.get("lru_misses", 0),
            summary.get("lru_evictions", 0),
            summary.get("lru_size", 0),
            summary.get("lru_capacity", 0),
            summary.get("advised", 0),
            summary.get("ctx_fallbacks", {}),
        )
    except Exception:
        pass


def ensure_streaming_backing_or_raise(
    model: Any,
    backing: Any | None,
    *,
    requested: bool,
    model_name: str,
) -> None:
    """Fail a lazy load whose streaming conversion produced nothing.

    A model lazy-loaded on the promise of SSD streaming must not reach
    materialize_lazy_state with every expert bank resident — that is the
    silent-OOM path the conversion exists to prevent. Raises when the
    checkpoint structurally supports streaming but neither the unified
    converter nor the legacy adapter claimed any MoE layer. No-op when the
    intent was off, the estimate declines the checkpoint, or real
    conversion happened — backing presence alone is not evidence of
    conversion.
    """
    if not requested:
        return
    if backing is not None and int(
        getattr(backing, "streaming_converted", 0) or 0
    ) > 0:
        return
    try:
        from .residency import expert_streaming_estimate

        estimate = expert_streaming_estimate(str(model_name))
    except Exception:
        return
    if not estimate.supported:
        return
    # A legacy OffloadSwitchGLU wrap also satisfies the intent (resident
    # fraction by design). Class-name match avoids a circular import.
    try:
        for _, mod in model.named_modules():
            cls_name = mod.__class__.__name__
            if cls_name in ("StreamingSwitchGLU", "OffloadSwitchGLU"):
                return
    except Exception:
        pass
    raise RuntimeError(
        f"Expert streaming: {model_name} supports streaming and was "
        "lazy-loaded for it, but no MoE layer was converted — refusing to "
        "materialize every expert bank in RAM"
    )


async def streaming_offload_load(
    model: Any,
    model_name: str,
    settings: Any,
    *,
    label: str = "model",
) -> tuple[Any | None, int]:
    """The load-time offload sequence shared by the engine paths.

    Streaming conversion -> legacy MoE offload -> zero-conversion guard.
    Must run BEFORE ``materialize_lazy_state`` and gate/up fusion: the
    checkpoint stayed lazy on this path and dropping the stock MoE modules
    is what keeps non-resident experts from ever materializing. All MLX
    work runs on the MLX executor because it allocates the resident slot
    tensors (#1304).

    Returns ``(backing, wrapped)`` — the streaming backing for the caller
    to keep alive, and the legacy-offload wrapped-layer count (0 when the
    unified path served or offload was not requested). The caller stamps
    ``backing`` onto the engine itself.
    """
    import asyncio

    from ..engine_core import get_mlx_executor
    from ..model_settings import moe_offload_requested

    loop = asyncio.get_running_loop()
    backing = None
    if getattr(settings, "expert_streaming_enabled", False):
        try:
            def _do_streaming():
                _, b = convert_model_to_streaming(model, model_name, settings)
                # keep backing alive on the model
                if b is not None:
                    try:
                        model._expert_streaming_backing = b  # type: ignore[attr-defined]
                    except Exception:
                        pass
                return b

            backing = await loop.run_in_executor(get_mlx_executor(), _do_streaming)
            logger.info("Expert streaming enabled for %s %s", label, model_name)
        except Exception as e:
            # Fail clean: streaming was explicitly enabled, so a backing
            # failure must fail the load — continuing to materialize
            # would retain all expert banks in RAM (OOM).
            logger.error(
                "Expert streaming conversion failed for %s %s: %s",
                label,
                model_name,
                e,
                exc_info=True,
            )
            raise RuntimeError(
                f"Expert streaming conversion failed for {label} {model_name}: {e}"
            ) from e

    # MoE expert offload: replace covered SwitchGLU layers with a
    # fetch-on-miss LRU cache streaming experts from the checkpoint's
    # own safetensors. The caches' slot maps and resident slots live on
    # plain attributes outside the module tree, so the lazy-state
    # materialization never reaches them; left lazy they stay bound to
    # the loader stream and the first request from an inference thread
    # dies with "There is no Stream(gpu, N) in current thread" — the
    # post-apply materialize fixes that on the same executor.
    wrapped = 0
    if moe_offload_requested(settings):
        from ..patches.moe_expert_offload import (
            apply_moe_expert_offload,
            materialize_offload_state,
        )

        fraction = float(
            getattr(settings, "moe_expert_offload_resident_fraction", 0.25)
        )
        wrapped = await loop.run_in_executor(
            get_mlx_executor(),
            apply_moe_expert_offload,
            model,
            model_name,
            fraction,
            settings,
        )
        if wrapped:
            await loop.run_in_executor(
                get_mlx_executor(), materialize_offload_state, model
            )

    # Fail before materializing: a lazy-loaded streaming checkpoint with
    # zero converted layers would evaluate every expert bank in RAM.
    if moe_offload_requested(settings):
        ensure_streaming_backing_or_raise(
            model, backing, requested=True, model_name=model_name
        )
    return backing, wrapped


def shutdown_expert_streaming(backing: Any) -> None:
    """Release MoE streaming resources held by *backing*.

    Persists the transition table, then closes shard fds/mmaps.
    Idempotent; safe to call with None or a RAM-dict backing. Engines
    call this in stop() and before replacing the model on reload so
    threads/fds never leak across model lifetimes.
    """
    if backing is None or isinstance(backing, dict):
        return
    # Persist the learned transition table before threads/fds die.
    try:
        save_transition_profile(backing)
    except Exception:
        pass
    try:
        close = getattr(backing, "close", None)
        if callable(close):
            close()
    except Exception:
        pass


def save_expert_pin_profile(engine: Any) -> None:
    """Persist the learned pin profile of a streaming engine, if any.

    Called from the engine ``stop()`` paths while the backing store (and the
    PinController attached to it) is still reachable — before teardown drops
    the references. Never raises: a failed save only costs the learned hot
    set, never correctness.
    """
    for holder in (
        engine,
        getattr(engine, "_model", None),
        getattr(engine, "_vlm_model", None),
    ):
        if holder is None:
            continue
        backing = getattr(holder, "_expert_streaming_backing", None)
        pinner = getattr(backing, "_pin_controller", None)
        if pinner is not None:
            try:
                pinner.save_profile()
            except Exception:
                logger.debug(
                    "Expert streaming: pin profile save failed", exc_info=True
                )
            return


__all__ = [
    "convert_model_to_streaming",
    "ensure_streaming_backing_or_raise",
    "resolve_budget_bytes",
    "save_expert_pin_profile",
    "is_supported_model_type",
    "normalize_model_type",
    "SUPPORTED_TYPES",
]
