# SPDX-License-Identifier: Apache-2.0
"""Streaming settings/budget policy and per-model override validation.

Pure-policy leaf of the expert_streaming package: the budget resolver
(``resolve_budget_bytes`` and its helpers), the dynamic-governor switch
(``_dynamic_armed``), the ``expert_streaming_*`` schema coercion shared by
the admin write boundary and the load path, and the
``is_supported_model_type`` allowlist gate. Top-level imports stay light —
stdlib, ``model_profiles``, and ``.residency`` are all mlx-free — so this
module is safe to pull from any context; heavier pieces stay deferred to
call time exactly as they were in the package ``__init__``.
"""

from __future__ import annotations

from typing import Any

from ...model_profiles import STREAMING_SETTING_SCHEMA
from .residency import streaming_owns_model


def is_supported_model_type(model_type: str | None) -> bool:
    # Same allowlist gate as residency.streaming_owns_model — that one
    # also accepts a local checkpoint dir, this one takes the type string
    # directly.
    return streaming_owns_model(model_type)


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
        # Resolved through the package namespace, not ``.governor``
        # directly, so monkeypatches of the re-exported
        # ``expert_streaming.dynamic_residency_enabled`` keep working.
        from . import dynamic_residency_enabled

        if dynamic_residency_enabled():
            return True
    except Exception:
        pass
    return not _budget_is_pinned(model_settings)


# Budget attribute spellings on the settings object — GiB pair preferred
# over the legacy MiB pair; any explicit value pins the budget (manual
# mode in _dynamic_armed).
_BUDGET_GIB_ATTRS = ("expert_streaming_budget_gib", "expert_cache_budget_gib")
_BUDGET_MIB_ATTRS = ("expert_streaming_budget_mib", "expert_cache_budget_mib")


def _budget_is_pinned(model_settings: Any | None) -> bool:
    """True when the user pinned an explicit budget (manual mode)."""
    if model_settings is None:
        return False
    for attr in _BUDGET_GIB_ATTRS + _BUDGET_MIB_ATTRS:
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
        from ...utils.psutil_compat import get_total_memory

        total = int(get_total_memory())
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
    if model_settings is not None:
        # Preferred name (model_settings.py:222) + legacy cache name.
        # Explicit 0 = page-cache only (no app-level LRU); None falls through
        # to the auto stack below.
        for attr in _BUDGET_GIB_ATTRS:
            gib = getattr(model_settings, attr, None)
            if gib is not None:
                try:
                    return _clamp_budget_bytes(float(gib) * 1024**3)
                except (TypeError, ValueError):
                    continue
        # legacy mib
        for attr in _BUDGET_MIB_ATTRS:
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


def validate_streaming_setting(name: str, value: Any) -> Any | None:
    """Normalize one ``expert_streaming_*`` value, or None when unusable.

    The runtime-side reader of ``model_profiles.STREAMING_SETTING_SCHEMA``
    — the single bounds table shared with the admin write path. Per type:
    bools stay strict (True/False only — "true"/1 are rejected, not
    coerced, matching the admin contract); ints coerce, reject below lo
    and clamp above hi; floats coerce and must sit inside (lo, hi] /
    [lo, hi] / [lo, ∞); choices lower-case and match. None in means
    unset — None out (keep the env / built-in default).
    """
    if value is None:
        return None
    spec = STREAMING_SETTING_SCHEMA.get(name)
    if spec is None:
        return None
    kind = spec["type"]
    if kind == "bool":
        return value if isinstance(value, bool) else None
    if kind == "choice":
        v = str(value).strip().lower()
        return v if v in spec["choices"] else None
    # Numeric: bools are not numbers for these knobs — True must not
    # silently become 1 GiB of pinned budget (admin rejects bools too).
    if isinstance(value, bool):
        return None
    lo, hi, lo_open = spec["bounds"]
    if kind == "int":
        try:
            v = int(value)
        except (TypeError, ValueError):
            return None
        if v < lo:
            return None
        return min(hi, v) if hi is not None else v
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not ((lo < v) if lo_open else (lo <= v)):
        return None
    if hi is not None and v > hi:
        return None
    return v


# Bounds now live in model_profiles.STREAMING_SETTING_SCHEMA; the
# validators delegate so the write-boundary table and this load-time
# coercion share one source. Only non-bool io keys get a validator — the
# bool flags keep their historical raw pass-through (None = unset), so
# _io_overrides() semantics are unchanged.
_IO_OVERRIDE_VALIDATORS = {
    key: (lambda _k: lambda v: validate_streaming_setting(_k, v))(key)
    for key in _IO_OVERRIDE_KEYS
    if STREAMING_SETTING_SCHEMA.get(key, {}).get("type") != "bool"
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
