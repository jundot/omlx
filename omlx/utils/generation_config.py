# SPDX-License-Identifier: Apache-2.0
"""Helpers for reading generation_config.json generation policy."""

from __future__ import annotations

import json
import logging
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
