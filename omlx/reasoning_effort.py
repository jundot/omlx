# SPDX-License-Identifier: Apache-2.0
"""Compatibility handling for chat-template reasoning effort values."""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

_STRING_LEVELS = {
    "off",
    "none",
    "minimal",
    "low",
    "moderate",
    "medium",
    "high",
    "xhigh",
    "max",
    "maximum",
    "ultra",
}
_ALIAS_FALLBACKS: dict[str, str] = {
    "off": "low",
    "none": "low",
    "minimal": "low",
    "moderate": "medium",
    "medium": "high",
    "high": "xhigh",
    "xhigh": "max",
    "max": "xhigh",
    "maximum": "max",
    "ultra": "max",
}
_HARMONY_ALIASES: dict[str, str] = {
    "off": "low",
    "none": "low",
    "minimal": "low",
    "low": "low",
    "moderate": "medium",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
    "maximum": "high",
    "ultra": "high",
}


def _normalized_input(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    normalized = value.strip().lower()
    if normalized in _STRING_LEVELS:
        return normalized
    try:
        numeric = float(normalized)
    except ValueError:
        return value
    return normalized if math.isfinite(numeric) else value


def _fallback_candidate(value: Any) -> Any | None:
    if not isinstance(value, str):
        return None
    try:
        numeric = float(value)
    except ValueError:
        return _ALIAS_FALLBACKS.get(value)
    return numeric if math.isfinite(numeric) else None


def _harmony_effort(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return _HARMONY_ALIASES.get(value.strip().lower())


def apply_chat_template_with_reasoning_effort_fallback(
    target: Any,
    messages: Any,
    template_kwargs: dict[str, Any],
    *,
    is_harmony: bool = False,
) -> Any:
    """Render with one alias retry, then the template's native default."""
    original_kwargs = dict(template_kwargs)
    if "reasoning_effort" not in original_kwargs:
        return target.apply_chat_template(messages, **original_kwargs)

    original_value = original_kwargs["reasoning_effort"]
    if is_harmony:
        mapped = _harmony_effort(original_value)
        if mapped is None:
            original_kwargs.pop("reasoning_effort", None)
            logger.debug(
                "Harmony ignored reasoning_effort=%r and used its native default",
                original_value,
            )
        else:
            original_kwargs["reasoning_effort"] = mapped
            if mapped != original_value:
                logger.debug(
                    "Harmony remapped reasoning_effort=%r to %r",
                    original_value,
                    mapped,
                )
        return target.apply_chat_template(messages, **original_kwargs)

    normalized = _normalized_input(original_value)
    original_kwargs["reasoning_effort"] = normalized
    try:
        return target.apply_chat_template(messages, **original_kwargs)
    except Exception as original_error:
        candidate = _fallback_candidate(normalized)
        if candidate is not None and candidate != normalized:
            candidate_kwargs = dict(original_kwargs)
            candidate_kwargs["reasoning_effort"] = candidate
            try:
                rendered = target.apply_chat_template(messages, **candidate_kwargs)
            except Exception:
                pass
            else:
                logger.debug(
                    "Chat template remapped reasoning_effort=%r to %r",
                    original_value,
                    candidate,
                )
                return rendered

        fallback_kwargs = dict(original_kwargs)
        fallback_kwargs.pop("reasoning_effort", None)
        try:
            rendered = target.apply_chat_template(messages, **fallback_kwargs)
        except Exception:
            raise original_error.with_traceback(original_error.__traceback__) from None

        logger.debug(
            "Chat template ignored reasoning_effort=%r and used its native default",
            original_value,
        )
        return rendered
