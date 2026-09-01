# SPDX-License-Identifier: Apache-2.0
"""Narrow capability helpers for DeepSeek-V4-Flash-Vision checkpoints.

The experimental checkpoint deliberately keeps ``model_type=deepseek_v4`` and
stores its vision configuration as flat top-level fields.  Consequently the
usual ``vision_config`` heuristic cannot distinguish it from the text model.
Keep that distinction in one dependency-free module so discovery, planning,
and the distributed runtime make the same fail-closed decision.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"
IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEWLINE, IMAGE_END = range(5)

_REQUIRED_POSITIVE_FIELDS = (
    "vision_n_layers",
    "vision_dim",
    "vision_n_heads",
    "vision_inter_dim",
    "vision_patch_size",
    "vision_downsample_ratio",
    "vision_max_n_token",
    "vision_min_pixels",
    "vision_max_wh_ratio",
)
_REQUIRED_POSITIVE_NUMERIC_FIELDS = ("vision_rope_theta",)
_COORDINATOR_EXACT_WEIGHTS = frozenset(
    {"image_start", "image_end", "image_newline", "image_pad"}
)


def is_deepseek_v4_vision_config(config: Mapping[str, Any] | None) -> bool:
    """Return whether *config* has the checkpoint's exact multimodal shape.

    Requiring all published flat fields is intentional.  A regular DeepSeek-V4
    config has the same model type and causal-LM architecture, and must remain
    on the text path.
    """

    if not isinstance(config, Mapping):
        return False
    model_type = config.get("model_type")
    if (
        not isinstance(model_type, str)
        or model_type.lower().replace("-", "_") != "deepseek_v4"
    ):
        return False
    for field in _REQUIRED_POSITIVE_FIELDS:
        value = config.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            return False
    for field in _REQUIRED_POSITIVE_NUMERIC_FIELDS:
        value = config.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            return False
    return True


def is_deepseek_v4_vision_weight(name: str) -> bool:
    """Return whether a checkpoint tensor is owned by coordinator/rank zero."""

    normalized = name.removeprefix("model.")
    return (
        normalized.startswith("vision.")
        or normalized.startswith("aligner.")
        or normalized in _COORDINATOR_EXACT_WEIGHTS
    )


def require_supported_distributed_vlm(config: Mapping[str, Any]) -> None:
    """Fail closed unless a VLM is the explicitly supported DS-V4 family."""

    if is_deepseek_v4_vision_config(config):
        return
    raise ValueError(
        "distributed VLM inference is experimental and currently supports only "
        "DeepSeek-V4-Flash-Vision checkpoints with the published flat vision "
        "configuration"
    )
