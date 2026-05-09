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
    "thinking_budget_enabled",
    "thinking_budget_tokens",
    "reasoning_parser",
    "max_tool_result_tokens",
    "chat_template_kwargs",
    "forced_ct_kwargs",
)

# Model-specific fields — eligible for per-model profiles only (never templates).
MODEL_SPECIFIC_PROFILE_FIELDS = (
    "turboquant_kv_enabled",
    "turboquant_kv_bits",
    "turboquant_skip_last",
    "dflash_enabled",
    "dflash_draft_model",
    "dflash_draft_quant_bits",
    "mtp_enabled",
    "specprefill_enabled",
    "specprefill_draft_model",
    "specprefill_keep_pct",
    "specprefill_threshold",
    "index_cache_freq",
)

# Excluded — never stored in a profile or template.
EXCLUDED_FROM_PROFILES = frozenset({
    "is_pinned",
    "is_default",
    "display_name",
    "description",
    "model_alias",
    "model_type_override",
    "active_profile_name",
    "ttl_seconds",
    # Security flag must be explicit per model — never propagated via profiles.
    "trust_remote_code",
})


def filter_universal_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict containing only UNIVERSAL_PROFILE_FIELDS keys."""
    allowed = set(UNIVERSAL_PROFILE_FIELDS)
    return {k: v for k, v in data.items() if k in allowed}


def filter_profile_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict containing UNIVERSAL + MODEL_SPECIFIC keys."""
    allowed = set(UNIVERSAL_PROFILE_FIELDS) | set(MODEL_SPECIFIC_PROFILE_FIELDS)
    return {k: v for k, v in data.items() if k in allowed}


@dataclass
class ModelProfile:
    """A per-model saved bundle of ModelSettings values."""

    name: str
    display_name: str
    created_at: datetime
    updated_at: datetime
    settings: dict[str, Any] = field(default_factory=dict)
    description: str | None = None
    source_template: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
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
