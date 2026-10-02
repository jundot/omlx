# SPDX-License-Identifier: Apache-2.0
"""MiniMax M3 adaptive prefill helpers for oMLX scheduler paths."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MINIMAX_M3_MODEL_TYPES = {"minimax_m3", "minimax_m3_vl"}


@dataclass(frozen=True)
class _AdaptivePrefillConfig:
    step_size: int
    after: int
    min_remaining: int


def _get_attr_or_key(obj: Any, name: str) -> Any:
    if type(obj).__module__.startswith("unittest.mock"):
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    try:
        return getattr(obj, name)
    except Exception:
        return None


def _declares_minimax_m3(model: Any) -> bool:
    seen: set[int] = set()
    stack = [model]
    while stack:
        obj = stack.pop()
        if obj is None:
            continue
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)

        model_type = _get_attr_or_key(obj, "model_type")
        if model_type in _MINIMAX_M3_MODEL_TYPES:
            return True
        if _get_attr_or_key(obj, "_uses_minimax_m3_positions") is True:
            return True

        for attr in (
            "args",
            "config",
            "text_config",
            "language_config",
            "llm_config",
            "_vlm_model",
            "_language_model",
            "language_model",
            "model",
            "vlm_model",
        ):
            child = _get_attr_or_key(obj, attr)
            if child is not None and not isinstance(
                child, (str, bytes, int, float, bool)
            ):
                stack.append(child)
    return False


def _declares_minimax_m3_model_path(model_path: Any) -> bool:
    if not model_path:
        return False
    try:
        config_path = Path(model_path).expanduser() / "config.json"
    except TypeError:
        return False
    if not config_path.is_file():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return config.get("model_type") in _MINIMAX_M3_MODEL_TYPES


def _minimax_m3_adaptive_prefill_config(
    model: Any, prefill_step_size: int, model_path: Any = None
) -> _AdaptivePrefillConfig | None:
    if (
        prefill_step_size != 2048
        or os.environ.get("MLX_MINIMAX_M3_ADAPTIVE_PREFILL_STEP", "1") != "1"
        or not (
            _declares_minimax_m3(model)
            or _declares_minimax_m3_model_path(model_path)
        )
    ):
        return None

    return _AdaptivePrefillConfig(
        step_size=int(
            os.environ.get("MLX_MINIMAX_M3_ADAPTIVE_PREFILL_STEP_SIZE", "4096")
        ),
        after=int(os.environ.get("MLX_MINIMAX_M3_ADAPTIVE_PREFILL_AFTER", "0")),
        min_remaining=int(
            os.environ.get("MLX_MINIMAX_M3_ADAPTIVE_PREFILL_MIN_REMAINING", "4096")
        ),
    )


def _prefill_step_size_for_progress(
    prefill_step_size: int,
    processed_tokens: int,
    remaining_tokens: int,
    adaptive_prefill: _AdaptivePrefillConfig | None,
) -> int:
    if (
        adaptive_prefill is not None
        and processed_tokens >= adaptive_prefill.after
        and remaining_tokens >= adaptive_prefill.min_remaining
    ):
        return adaptive_prefill.step_size
    return prefill_step_size


__all__ = [
    "_AdaptivePrefillConfig",
    "_declares_minimax_m3",
    "_declares_minimax_m3_model_path",
    "_minimax_m3_adaptive_prefill_config",
    "_prefill_step_size_for_progress",
]
