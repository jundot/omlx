# SPDX-License-Identifier: Apache-2.0
"""Patch mlx-vlm's Qwen3.5 ``Model.sanitize`` to preserve MTP weights.

The stock implementation in
``mlx_vlm/models/qwen3_5/qwen3_5.py`` strips every ``mtp.*`` key on the
first line and only shifts a fixed list of backbone norm suffixes by
+1. With Qwen3.5/3.6 native MTP heads (PR 990 layout), the MTP block
also has its own RMSNorm weights that need the +1 shift — both the
in-block ``input_layernorm`` / ``post_attention_layernorm`` /
``q_norm`` / ``k_norm`` (caught by the existing suffix patterns once
``mtp.*`` is no longer stripped) and three MTP-specific norms
(``mtp.norm.weight``, ``mtp.pre_fc_norm_hidden.weight``,
``mtp.pre_fc_norm_embedding.weight``) that have no analogue in the
backbone and are missed entirely.

After this patch the sanitize body keeps mtp tensors intact, applies
the +1 shift to MTP norms (raw HF only), and remains idempotent for
already-converted checkpoints by guarding the shift on the unsanitized
conv1d marker (matches mlx_lm_mtp/qwen35_model.py).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_APPLIED = False


def apply() -> bool:
    global _APPLIED
    if _APPLIED:
        return True

    try:
        from mlx_vlm.models.qwen3_5 import qwen3_5 as q35vlm
    except Exception as e:
        logger.debug(f"mlx_vlm qwen3_5 not importable: {e}")
        return False

    cls = q35vlm.Model
    if getattr(cls, "_omlx_mtp_vlm_patched", False):
        _APPLIED = True
        return True

    def sanitize(self, weights):
        # Detect raw-HF input via unsanitized conv1d shape (matches
        # mlx_lm_mtp/qwen35_model.py). Already-sanitized checkpoints
        # (e.g. an oQ output passed through this sanitize again) keep
        # their norms unshifted.
        has_unsanitized_conv1d = any(
            "conv1d.weight" in k and getattr(v, "shape", (1,))[-1] != 1
            for k, v in weights.items()
        )
        should_shift_norm_weights = has_unsanitized_conv1d

        if self.config.text_config.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        norm_keys = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            # MTP-specific norms that the stock list misses.
            ".pre_fc_norm_hidden.weight",
            ".pre_fc_norm_embedding.weight",
            "mtp.norm.weight",
        )

        sanitized_weights = {}
        for key, value in weights.items():
            if "model" in key:
                if "model.language_model" in key:
                    key = key.replace(
                        "model.language_model", "language_model.model"
                    )
                elif "model.visual" in key:
                    key = key.replace("model.visual", "vision_tower")
            elif "lm_head" in key:
                key = key.replace("lm_head", "language_model.lm_head")
            elif key.startswith("mtp."):
                # MTP weights live under ``language_model.mtp.*`` in the
                # mlx-lm model hierarchy (outer Model wraps language_model).
                # Without this prefix the oQ quantization config keys would
                # be ``mtp.*`` while load-time class_predicate looks up
                # ``language_model.mtp.*`` — mismatch → default 4-bit init
                # vs e.g. 6-bit packed weight on disk → shape error.
                key = "language_model." + key

            if "conv1d.weight" in key and value.shape[-1] != 1:
                value = value.moveaxis(2, 1)
            if should_shift_norm_weights and any(
                key.endswith(sfx) for sfx in norm_keys
            ):
                if value.ndim == 1:
                    value = value + 1.0

            sanitized_weights[key] = value

        return sanitized_weights

    cls.sanitize = sanitize
    cls._omlx_mtp_vlm_patched = True
    _APPLIED = True
    logger.info("Patched mlx_vlm.models.qwen3_5.Model.sanitize for MTP")
    return True
