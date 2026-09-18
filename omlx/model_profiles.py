# SPDX-License-Identifier: Apache-2.0
"""Profile and template primitives for per-model settings.

Defines the field allowlists used to split ModelSettings values into:
- Universal fields (shared via global templates)
- Model-specific fields (profiles only)
- Excluded fields (identity/management, never in profiles or templates)

Also defines the serializable ``ModelProfile`` and ``GlobalTemplate``
dataclasses and helpers to filter incoming setting dicts to the
allowed keys.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# Universal fields — eligible for global templates AND per-model profiles.
UNIVERSAL_PROFILE_FIELDS = (
    "max_context_window",
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "force_sampling",
    "enable_thinking",
    "preserve_thinking",
    "cache_reasoning_output",
    "thinking_budget_enabled",
    "thinking_budget_tokens",
    "reasoning_parser",
    "guided_grammar_enabled",
    "guided_grammar",
    "max_tool_result_tokens",
    "chat_template_kwargs",
    "forced_ct_kwargs",
)

# Machine-readable schema for the expert_streaming_* ModelSettings fields
# (model_settings.py) — one ordered bounds table so the admin write
# validation, the load-time io-overrides coercion
# (patches/expert_streaming/__init__.py) and tests share the ranges
# instead of drifting copies.
#
# spec keys:
#   "type"    : "bool" | "int" | "float" | "choice"
#   "default" : the ModelSettings field default (None = unset marker)
#   "bounds"  : (lo, hi, lo_open) for int/float — hi=None is unbounded,
#               lo_open excludes the low end
#   "choices" : allowed normalized strings for "choice"
#   "tunable" : False marks the routing switch itself — excluded from
#               EXPERT_STREAMING_TUNABLE_KEYS below
#
# Bounds mirror the runtime's own coercion points: the GiB ceilings match
# MAX_EXPERT_STREAMING_BUDGET_BYTES (64 GiB); topk_threshold's floor is
# adaptive_topk._MIN_THRESHOLD (below it the runtime drops to exact
# routing, so persisting it would store a knob that never engages);
# cold_tier accepts any "2".."8" digit label plus "" = off
# (conversion._resolve_cold_tier_root). Explicit 0 on budget_gib is legal
# (page-cache only); the other GiB fields are lo-open — 0 is rejected.
STREAMING_SETTING_SCHEMA: dict[str, dict[str, Any]] = {
    "expert_streaming_enabled": {
        "type": "bool",
        "default": False,
        "tunable": False,
    },
    "expert_streaming_budget_gib": {
        "type": "float",
        "default": None,
        "bounds": (0.0, 64.0, False),
    },
    "expert_streaming_budget_auto": {"type": "bool", "default": True},
    "expert_streaming_dynamic": {"type": "bool", "default": None},
    "expert_streaming_dynamic_max_gib": {
        "type": "float",
        "default": None,
        "bounds": (0.0, 64.0, True),
    },
    "expert_streaming_dynamic_min_gib": {
        "type": "float",
        "default": None,
        "bounds": (0.0, 64.0, False),
    },
    "expert_streaming_dynamic_stall_target": {
        "type": "float",
        "default": None,
        "bounds": (0.0, 0.9, False),
    },
    "expert_streaming_prefill_budget_gib": {
        "type": "float",
        "default": None,
        "bounds": (0.0, 64.0, True),
    },
    "expert_streaming_io_depth": {
        "type": "int",
        "default": None,
        "bounds": (1, 64, False),
    },
    "expert_streaming_coalesce": {"type": "bool", "default": None},
    "expert_streaming_readahead": {"type": "bool", "default": None},
    "expert_streaming_seed": {"type": "bool", "default": None},
    "expert_streaming_per_layer_eval": {"type": "bool", "default": None},
    "expert_streaming_pins": {"type": "bool", "default": None},
    "expert_streaming_pin_gib": {
        "type": "float",
        "default": None,
        "bounds": (0.0, 64.0, True),
    },
    "expert_streaming_pin_sync": {"type": "bool", "default": None},
    "expert_streaming_pin_regime": {
        "type": "choice",
        "default": None,
        "choices": ("decode", "prefill"),
    },
    "expert_streaming_cold_tier": {
        "type": "choice",
        "default": None,
        "choices": ("", "2", "3", "4", "5", "6", "7", "8"),
    },
    "expert_streaming_hot_fraction": {
        "type": "float",
        "default": None,
        "bounds": (0.0, 1.0, False),
    },
    "expert_streaming_cache_policy": {
        "type": "choice",
        "default": None,
        "choices": ("lru", "s3fifo"),
    },
    "expert_streaming_topk_threshold": {
        "type": "float",
        "default": None,
        "bounds": (0.05, 1.0, False),
    },
    "expert_streaming_cache_prior": {
        "type": "float",
        "default": None,
        "bounds": (0.0, None, False),
    },
}

# The expert_streaming_* tunable family consumed at engine construction
# (everything except the `enabled` switch). Derived from the schema so a
# new knob cannot drift out of the profile allowlist: spliced into
# MODEL_SPECIFIC_PROFILE_FIELDS below and re-exported through
# model_settings for EnginePool's reload signature (a changed value must
# force a reload) and the admin sanitizers (these keys must not survive
# on a diffusion model).
EXPERT_STREAMING_TUNABLE_KEYS = tuple(
    key for key, spec in STREAMING_SETTING_SCHEMA.items() if spec.get("tunable", True)
)

# Model-specific fields — eligible for per-model profiles only (never templates).
MODEL_SPECIFIC_PROFILE_FIELDS = (
    "turboquant_kv_enabled",
    "turboquant_kv_bits",
    "turboquant_skip_last",
    "qwen35_ane_prefill_enabled",
    "qwen35_ane_prefill_sequence_length",
    "qwen35_ane_prefill_tail_padding_min_tokens",
    "qwen35_ane_prefill_fraction",
    "qwen35_ane_prefill_shared_fraction",
    "qwen35_ane_prefill_fused_down",
    "qwen35_ane_prefill_max_layers",
    "qwen35_ane_prefill_dual_ane",
    "qwen35_ane_prefill_gdn",
    "qwen35_ane_prefill_gdn_fraction",
    "qwen35_ane_prefill_gdn_max_layers",
    "qwen35_ane_prefill_cpu_enabled",
    "qwen35_ane_prefill_cpu_fraction",
    "qwen35_ane_prefill_cpu_down_fraction",
    "qwen35_ane_prefill_cpu_gdn_fraction",
    "qwen35_ane_prefill_cpu_threads",
    "qwen35_ane_prefill_cpu_shared_resource",
    "qwen35_oq_a8_enabled",
    "qwen35_oq_a8_min_tokens",
    "moe_expert_offload_enabled",
    "moe_expert_offload_resident_fraction",
    # Validation scope for model-dependent rules (e.g. the DSpark offload
    # exception). Per-model only — never templates. Always re-derived
    # from the checkpoint at validation time, so a copied profile cannot
    # carry a stale scope.
    "model_type",
    # Unified expert-streaming backend (canonical keys; the moe_* pair above
    # stays as load-time aliases served by the same backend). `enabled` is
    # the routing switch and stays spelled out; the IO/pinning/policy
    # tunables splice in from EXPERT_STREAMING_TUNABLE_KEYS. Per-model by
    # construction: several are checkpoint- or hardware-bound (pins, seed)
    # and must never propagate across models via templates.
    "expert_streaming_enabled",
    *EXPERT_STREAMING_TUNABLE_KEYS,
    "dflash_enabled",
    "dflash_draft_model",
    "dflash_draft_quant_enabled",
    "dflash_draft_quant_weight_bits",
    "dflash_draft_quant_activation_bits",
    "dflash_draft_quant_group_size",
    "dflash_max_ctx",
    "dflash_in_memory_cache",
    "dflash_in_memory_cache_max_entries",
    "dflash_in_memory_cache_max_bytes",
    "dflash_ssd_cache",
    "dflash_ssd_cache_max_bytes",
    "dflash_draft_window_size",
    "dflash_draft_sink_size",
    "dflash_block_size",
    "dflash_verify_mode",
    "mtp_enabled",
    "mtp_num_draft_tokens",
    "vlm_mtp_enabled",
    "vlm_mtp_draft_model",
    "vlm_mtp_draft_block_size",
    "specprefill_enabled",
    "specprefill_draft_model",
    "specprefill_keep_pct",
    "specprefill_threshold",
    "index_cache_freq",
)

# Excluded — never stored in a profile or template.
EXCLUDED_FROM_PROFILES = frozenset(
    {
        "is_pinned",
        "is_default",
        "is_hidden",
        "is_favorite",
        "display_name",
        "description",
        "model_alias",
        "model_type_override",
        "active_profile_name",
        "ttl_seconds",
        # Hardware-specific residency choice; never propagate across models.
        "qwen4_ple_ssd_offload",
        "deepseek_v41_engram_ssd_offload",
        # Security flag must be explicit per model — never propagated via profiles.
        "trust_remote_code",
    }
)


UNIVERSAL_FIELDS_SET = frozenset(UNIVERSAL_PROFILE_FIELDS)
PROFILE_FIELDS_SET = UNIVERSAL_FIELDS_SET | frozenset(MODEL_SPECIFIC_PROFILE_FIELDS)


def _filter_and_sanitize(
    data: dict[str, Any], allowed: frozenset[str]
) -> dict[str, Any]:
    """Keep allowlisted keys that carry a real value.

    None and "" are "unset" markers (older clients stored them for cleared
    inputs); under snapshot apply an unset field must be absent, so both are
    dropped on save and when overlaying stored (possibly legacy) records.
    """
    return {
        k: v for k, v in data.items() if k in allowed and v is not None and v != ""
    }


def filter_universal_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict of UNIVERSAL_PROFILE_FIELDS keys with real values."""
    return _filter_and_sanitize(data, UNIVERSAL_FIELDS_SET)


def filter_profile_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict of UNIVERSAL + MODEL_SPECIFIC keys with real values."""
    return _filter_and_sanitize(data, PROFILE_FIELDS_SET)


@dataclass
class ModelProfile:
    """A per-model saved bundle of ModelSettings values."""

    name: str
    display_name: str
    created_at: datetime
    updated_at: datetime
    api_name: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    description: str | None = None
    source_template: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "api_name": self.api_name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "settings": dict(self.settings),
            "source_template": self.source_template,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelProfile":
        return cls(
            name=data["name"],
            display_name=data["display_name"],
            api_name=data.get("api_name"),
            description=data.get("description"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            settings=dict(data.get("settings") or {}),
            source_template=data.get("source_template"),
        )


@dataclass
class GlobalTemplate:
    """A globally-shared bundle of universal ModelSettings values."""

    name: str
    display_name: str
    created_at: datetime
    updated_at: datetime
    settings: dict[str, Any] = field(default_factory=dict)
    description: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "settings": dict(self.settings),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GlobalTemplate":
        return cls(
            name=data["name"],
            display_name=data["display_name"],
            description=data.get("description"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            settings=dict(data.get("settings") or {}),
        )


def utcnow() -> datetime:
    """Return current UTC time (single-source helper for testability)."""
    return datetime.now(timezone.utc)


class InvalidProfileNameError(ValueError):
    """Raised when a profile or template name fails validation."""


_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


def validate_profile_name(name: str) -> None:
    """Raise InvalidProfileNameError if ``name`` is not a valid slug.

    Valid: lowercase letters/digits, underscores, dashes. Must start with
    a letter or digit. 1-32 characters.
    """
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise InvalidProfileNameError(
            f"Invalid profile/template name: {name!r}. "
            f"Must match ^[a-z0-9][a-z0-9_-]{{0,31}}$"
        )


def slugify_profile_api_name(value: str | None, fallback: str = "profile") -> str:
    """Derive a stable API-safe profile suffix from user-facing text."""
    text = unicodedata.normalize("NFKD", value or "")
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9_-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-_")
    if not text or not re.match(r"^[a-z0-9]", text):
        text = fallback
    text = text[:32].rstrip("-_")
    if not text or not _NAME_RE.match(text):
        text = fallback[:32].rstrip("-_") or "profile"
    return text
