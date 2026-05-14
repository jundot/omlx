# SPDX-License-Identifier: Apache-2.0
"""Patch mlx-vlm's qwen3_5_moe VLM sanitize for Qwen3.6's nested visual layout.

Qwen3.6-35B-A3B's HF checkpoint nests the ViT weights inside the language
model submodule: `model.language_model.visual.*` instead of the flat
`model.visual.*` layout that other Qwen VLMs use. mlx-vlm's sanitize_key
uses if/elif: it matches `model.language_model` first, rewrites to
`language_model.model`, and the `model.visual -> vision_tower` branch
never fires. Result: keys land at `language_model.model.visual.*`, which
the instantiated `Qwen3_5MoeForConditionalGeneration` model class does
not have (its ViT lives at `self.vision_tower`). 333 visual params get
silently dropped on load and any image input produces garbage.

The mlx_vlm_mtp runtime sanitize (omlx/patches/mlx_vlm_mtp) re-implements
the same if/elif shape, so the bug also carries through mtp_enabled=True.

This patch wraps `Model.sanitize` on mlx-vlm's `qwen3_5_moe` module to
remap `language_model.model.visual.* -> vision_tower.*` after the
original sanitize runs. Wired from
``omlx.utils.model_loading.maybe_apply_pre_load_patches`` (after
``apply_mlx_vlm_mtp_runtime_patch`` so it covers whichever sanitize the
class currently has) and from ``omlx.oq._build_model_sanitizer``.

Self-guards: if upstream mlx-vlm adds the rule itself, source inspection
picks up ``language_model.model.visual.`` or ``vision_tower.`` and the
patch skips.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_NESTED_PREFIX = "language_model.model.visual."
_TARGET_PREFIX = "vision_tower."


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

    patched_sanitize._omlx_nested_visual_wrapped = True
    return patched_sanitize


def apply_qwen3_6_nested_visual_patch() -> bool:
    """Install the sanitize wrapper on mlx-vlm's Qwen3_5MoE VLM Model class.

    Idempotent: skips if the current ``Model.sanitize`` already carries the
    ``_omlx_nested_visual_wrapped`` marker. Uses a function-attribute marker
    instead of a module-level flag so that if another patch replaces
    ``Model.sanitize`` (e.g. MTP runtime), this wrapper can re-apply.
    """
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

    if getattr(original_sanitize, "_omlx_nested_visual_wrapped", False):
        return False

    try:
        import inspect

        source = inspect.getsource(original_sanitize)
        if _NESTED_PREFIX in source or _TARGET_PREFIX in source:
            logger.debug(
                "qwen3_6_nested_visual: upstream sanitize already handles "
                "nested visual; skipping"
            )
            return False
    except (OSError, TypeError):
        pass

    model_cls.sanitize = _make_patched_sanitize(original_sanitize)
    logger.info("qwen3_6_nested_visual: patched mlx_vlm.qwen3_5_moe Model.sanitize")
    return True
