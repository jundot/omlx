# SPDX-License-Identifier: Apache-2.0
"""Helpers for reading generation_config.json generation policy."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _resolve_generation_config_path(model_path_or_id: str | Path | None) -> Path | None:
    if not model_path_or_id:
        return None

    model_ref = str(model_path_or_id)
    candidate = Path(model_ref)
    if candidate.name == "generation_config.json" and candidate.exists():
        return candidate

    local_path = candidate / "generation_config.json"
    if local_path.exists():
        return local_path

    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(model_ref, "generation_config.json")
    except Exception:
        cached = None

    if cached and isinstance(cached, str):
        cached_path = Path(cached)
        if cached_path.exists():
            return cached_path

    return None


def load_generation_config(model_path_or_id: str | Path | None) -> dict[str, Any] | None:
    """Load generation_config.json from a local path or Hugging Face cache."""

    path = _resolve_generation_config_path(model_path_or_id)
    if path is None:
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Could not read generation_config.json from %s: %s", path, exc)
        return None

    return data if isinstance(data, dict) else None


def _as_token_id_set(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, bool):
        return set()
    if isinstance(value, int):
        return {value}
    if isinstance(value, (list, tuple, set)):
        result: set[int] = set()
        for item in value:
            if isinstance(item, bool):
                continue
            if isinstance(item, int):
                result.add(item)
        return result
    return set()


def load_generation_config_token_ids(
    model_path_or_id: str | Path | None,
    key: str,
) -> set[int] | None:
    """Return token IDs from a generation config field.

    Returns None when no config/key is available, and an empty set when the
    config explicitly contains the key but no valid integer token IDs.
    """

    config = load_generation_config(model_path_or_id)
    if config is None or key not in config:
        return None
    return _as_token_id_set(config.get(key))


SAMPLING_SETTING_KEYS = ("temperature", "top_p", "top_k", "repetition_penalty")

# Used when generation_config.json leaves these keys out. Qwen uses top_k 20
# in every release; top_p 0.95 is its thinking-mode value. Gemma 3/4 use
# 64/0.95 throughout. Temperature differs per variant, so it is not filled.
_FAMILY_SAMPLING_DEFAULTS: tuple[tuple[tuple[str, ...], dict[str, Any]], ...] = (
    (("qwen3", "qwen4"), {"top_k": 20, "top_p": 0.95}),
    (("gemma3", "gemma4"), {"top_k": 64, "top_p": 0.95}),
)


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def load_sampling_defaults(
    model_path_or_id: str | Path | None,
    model_type: str | None = None,
) -> dict[str, float | int]:
    """Return the sampling settings a model recommends.

    Values come from generation_config.json. top_k and top_p that the file
    leaves out come from the model family's published defaults. ``do_sample``
    is ignored, so an explicit temperature is kept even next to
    ``do_sample: false``. Out-of-range values are skipped.
    """

    config = load_generation_config(model_path_or_id) or {}
    result: dict[str, float | int] = {}

    temperature = _as_number(config.get("temperature"))
    if temperature is not None and temperature >= 0:
        result["temperature"] = temperature
    top_p = _as_number(config.get("top_p"))
    if top_p is not None and 0 < top_p <= 1:
        result["top_p"] = top_p
    top_k = config.get("top_k")
    if isinstance(top_k, int) and not isinstance(top_k, bool):
        result["top_k"] = max(top_k, 0)  # -1 also means disabled.
    repetition_penalty = _as_number(config.get("repetition_penalty"))
    if repetition_penalty is not None and repetition_penalty > 0:
        result["repetition_penalty"] = repetition_penalty

    family = (model_type or "").lower().replace("-", "_")
    for prefixes, defaults in _FAMILY_SAMPLING_DEFAULTS:
        if family.startswith(prefixes):
            for key, value in defaults.items():
                result.setdefault(key, value)
            break
    return result
