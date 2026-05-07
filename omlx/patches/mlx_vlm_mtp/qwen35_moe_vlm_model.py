# SPDX-License-Identifier: Apache-2.0
"""Patch mlx-vlm's Qwen3.5 MoE ``Model.sanitize`` to preserve MTP weights.

Same fix as ``qwen35_vlm_model.py`` but for the MoE variant in
``mlx_vlm/models/qwen3_5_moe/qwen3_5_moe.py``. The expert weight repacking
(``gate_up_proj`` split into ``gate_proj`` / ``up_proj``) is preserved
verbatim from the upstream body.
"""

from __future__ import annotations

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)

_APPLIED = False


def apply() -> bool:
    global _APPLIED
    if _APPLIED:
        return True

    try:
        from mlx_vlm.models.qwen3_5_moe import qwen3_5_moe as q35moevlm
    except Exception as e:
        logger.debug(f"mlx_vlm qwen3_5_moe not importable: {e}")
        return False

    cls = q35moevlm.Model
    if cls.__dict__.get("_omlx_mtp_vlm_patched", False):
        _APPLIED = True
        return True

    def sanitize(self, weights):
        has_unsanitized_conv1d = any(
            "conv1d.weight" in k and getattr(v, "shape", (1,))[-1] != 1
            for k, v in weights.items()
        )
        should_shift_norm_weights = has_unsanitized_conv1d

        if self.config.text_config.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        # Expert repack: gate_up_proj → gate_proj + up_proj per layer.
        def _unfuse_layer_experts(prefix):
            gate_up_key = f"{prefix}.experts.gate_up_proj"
            if gate_up_key not in weights:
                return
            gate_up_weight = weights.pop(gate_up_key)
            gate_weight, up_weights = mx.split(gate_up_weight, 2, axis=-2)
            weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_weight
            weights[f"{prefix}.switch_mlp.up_proj.weight"] = up_weights
            down_key = f"{prefix}.experts.down_proj"
            if down_key in weights:
                weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(
                    down_key
                )

        for layer_idx in range(self.config.text_config.num_hidden_layers):
            _unfuse_layer_experts(f"model.language_model.layers.{layer_idx}.mlp")

        # MTP expert layers also ship in fused form (Qwen3.6) and must be
        # unfused here so the oQ quantization sees the same per-projection
        # shapes as the model class expects (switch_mlp.{gate,up,down}_proj).
        # Without this, oQ stores fused experts.gate_up_proj on disk and
        # mlx-lm's load-time unfuse can't recover the per-tensor quantization
        # bits — class_predicate misses the lookup → wrong-bit init → shape
        # mismatch.
        # Note: mlx-vlm's TextConfig dataclass doesn't expose
        # ``mtp_num_hidden_layers``, so we discover MTP layer indices from
        # the weight keys themselves.
        mtp_layer_idxs = sorted(
            {
                int(k.split(".")[2])
                for k in weights
                if k.startswith("mtp.layers.")
                and len(k.split(".")) > 2
                and k.split(".")[2].isdigit()
            }
        )
        for layer_idx in mtp_layer_idxs:
            _unfuse_layer_experts(f"mtp.layers.{layer_idx}.mlp")

        norm_keys = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
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
                # mlx-lm model hierarchy. See qwen35_vlm_model.py for why.
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
    logger.info("Patched mlx_vlm.models.qwen3_5_moe.Model.sanitize for MTP")
    return True
