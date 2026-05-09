# SPDX-License-Identifier: Apache-2.0
"""Patch mlx-vlm's qwen3_5_moe VLM sanitize for Qwen3.6's nested visual layout.

Qwen3.6-35B-A3B's HF checkpoint nests the ViT weights inside the language
model submodule: `model.language_model.visual.*` instead of the flat
`model.visual.*` layout that other Qwen VLMs use. After mlx-vlm's default
sanitize applies `model.language_model → language_model.model`, the visual
keys land at `language_model.model.visual.*`, which the instantiated
`Qwen3_5MoeForConditionalGeneration` model class does not have — the ViT
lives at `self.vision_tower`. Result: 333 visual parameters are silently
dropped on load and any image input produces garbage.

This patch wraps `Model.sanitize` in mlx-vlm's `qwen3_5_moe` module to
remap `language_model.model.visual.*` -> `vision_tower.*` after the
original sanitize runs.

Target commit: mlx-vlm 3472132... (pyproject pin). Should become a no-op
once upstream mlx-vlm adds the rule itself; the patch auto-skips if it
detects the rule is already there.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_NESTED_PREFIX = "language_model.model.visual."
_TARGET_PREFIX = "vision_tower."

_class_patch_applied = False


def _rewrite_key(key: str) -> str:
    if key.startswith(_NESTED_PREFIX):
        return _TARGET_PREFIX + key[len(_NESTED_PREFIX) :]
    return key


def _make_patched_sanitize(original_sanitize):
    # mlx-vlm's Model.sanitize is an instance method: def sanitize(self, weights).
    # Preserve that signature so the bound-method call site keeps working.
    def patched_sanitize(self, weights):
        sanitized = original_sanitize(self, weights)
        remapped = 0
        out: dict = {}
        for k, v in sanitized.items():
            new_k = _rewrite_key(k)
            if new_k != k:
                remapped += 1
            out[new_k] = v
        if remapped:
            logger.info(
                "qwen3_6_nested_visual: remapped %d tensor keys "
                "'language_model.model.visual.*' -> 'vision_tower.*'",
                remapped,
            )
        return out

    return patched_sanitize


def apply_qwen3_6_nested_visual_patch() -> bool:
    """Install the sanitize wrapper on mlx-vlm's Qwen3_5MoE VLM Model class.

    Idempotent. Returns True on first successful application, False if the
    module is unavailable or the patch was already applied.
    """
    global _class_patch_applied

    if _class_patch_applied:
        return False

    try:
        from mlx_vlm.models.qwen3_5_moe import qwen3_5_moe as qwen3_5_moe_module
    except ImportError:
        logger.debug("qwen3_6_nested_visual: mlx_vlm.models.qwen3_5_moe not available")
        return False

    model_cls = getattr(qwen3_5_moe_module, "Model", None)
    if model_cls is None:
        logger.debug("qwen3_6_nested_visual: Model class not found on module")
        return False

    original_sanitize = getattr(model_cls, "sanitize", None)
    if original_sanitize is None:
        logger.debug("qwen3_6_nested_visual: Model has no sanitize attr")
        return False

    try:
        import inspect

        source = inspect.getsource(original_sanitize)
        if _NESTED_PREFIX in source or _TARGET_PREFIX in source:
            logger.debug(
                "qwen3_6_nested_visual: upstream sanitize already handles "
                "nested visual; skipping"
            )
            _class_patch_applied = True
            return False
    except (OSError, TypeError):
        pass

    model_cls.sanitize = _make_patched_sanitize(original_sanitize)
    _class_patch_applied = True
    logger.info("qwen3_6_nested_visual: patched mlx_vlm.qwen3_5_moe Model.sanitize")
    return True
