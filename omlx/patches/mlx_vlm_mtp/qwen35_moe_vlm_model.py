# SPDX-License-Identifier: Apache-2.0
"""Patch mlx-vlm's Qwen3.5 MoE ``Model.sanitize`` to preserve MTP weights.

Same fix as ``qwen35_vlm_model.py`` but for the MoE variant in
``mlx_vlm/models/qwen3_5_moe/qwen3_5_moe.py``. The expert weight repacking
(``gate_up_proj`` split into ``gate_proj`` / ``up_proj``) follows the pinned
upstream sanitize body.
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

        num_experts = int(getattr(self.config.text_config, "num_experts", 0) or 0)

        # Expert repack: normalise to switch_mlp form.
        # Two source formats are seen in the wild:
        #   - Fused ``experts.gate_up_proj`` (Qwen3.6 checkpoints)
        #   - Per-expert ``experts.{N}.{gate,up,down}_proj.weight`` (Ornith / Qwen3.5)
        def _unfuse_layer_experts(prefix):
            if f"{prefix}.switch_mlp.gate_proj.weight" in weights:
                return  # already in switch_mlp form
            gate_up_key = f"{prefix}.experts.gate_up_proj"
            if gate_up_key in weights:
                gate_up_weight = weights.pop(gate_up_key)
                gate_weight, up_weights = mx.split(gate_up_weight, 2, axis=-2)
                weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_weight
                weights[f"{prefix}.switch_mlp.up_proj.weight"] = up_weights
                down_key = f"{prefix}.experts.down_proj"
                if down_key in weights:
                    weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(
                        down_key
                    )
            elif num_experts > 0 and f"{prefix}.experts.0.gate_proj.weight" in weights:
                # Also stack quantization metadata (.scales, .biases) so a
                # per-expert *quantized* checkpoint loads cleanly. Without this,
                # only .weight is stacked and the per-expert scales/biases
                # survive as orphan keys -> "Received N parameters not in model".
                for n in ("gate_proj", "up_proj", "down_proj"):
                    for suffix in ("weight", "scales", "biases"):
                        first_key = f"{prefix}.experts.0.{n}.{suffix}"
                        if first_key not in weights:
                            continue
                        weights[f"{prefix}.switch_mlp.{n}.{suffix}"] = mx.stack(
                            [
                                weights.pop(f"{prefix}.experts.{e}.{n}.{suffix}")
                                for e in range(num_experts)
                            ]
                        )

        for layer_idx in range(self.config.text_config.num_hidden_layers):
            _unfuse_layer_experts(f"model.language_model.layers.{layer_idx}.mlp")

        # MTP expert layers ship in two possible formats:
        #   - Fused ``experts.gate_up_proj`` (Qwen3.6)
        #   - Per-expert ``experts.{N}.{gate,up,down}_proj.weight`` (Qwen3.5)
        # Both must be normalised to the switch_mlp form here so oQ
        # quantization sees the same per-projection shapes as the model
        # class expects.
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
        num_experts = int(getattr(self.config.text_config, "num_experts", 0) or 0)
        for layer_idx in mtp_layer_idxs:
            prefix = f"mtp.layers.{layer_idx}.mlp"
            if f"{prefix}.switch_mlp.gate_proj.weight" in weights:
                continue  # already in switch_mlp form
            _unfuse_layer_experts(prefix)

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

        # MTP-head norm conventions: raw-HF sources (unsanitized conv1d
        # present) shift every head norm uniformly with the backbone — see
        # the raw-HF branch below. Pre-converted checkpoints can ship MIXED
        # head conventions (JANG MXFP4 Qwen3.6 bundles keep ``mtp.norm`` in
        # MLX's +1 convention while the per-layer head norms remain raw-HF),
        # so that branch decides PER-KEY from each weight's own magnitude
        # (raw-HF center ~0, MLX-shifted ~1). Mirrors
        # mlx_lm_mtp/qwen35_model.py. The magnitude is unreadable during oQ
        # streaming plan discovery (the weight is a no-data ``_TrackedTensor``
        # and ``mx.mean(...).item()`` raises), so emit a conditional replay
        # transform there.
        def _is_oq_tracked_tensor(_w):
            return _w.__class__.__name__ == "_TrackedTensor" and hasattr(_w, "_clone")

        def _mark_mtp_norm_conditional_add(_w):
            return _w._clone(transform="add_if_mean_lt_0_5")

        def _mtp_norm_is_raw_hf(_w, _fallback):
            try:
                return float(mx.mean(_w.astype(mx.float32)).item()) < 0.5
            except Exception:
                return _fallback

        sanitized_weights = {}
        for key, value in weights.items():
            if "model" in key:
                if "model.language_model" in key:
                    key = key.replace("model.language_model", "language_model.model")
                elif "model.visual" in key:
                    key = key.replace("model.visual", "vision_tower")
            elif "lm_head" in key:
                key = key.replace("lm_head", "language_model.lm_head")
            elif key.startswith("mtp."):
                # MTP weights live under ``language_model.mtp.*`` in the
                # mlx-lm model hierarchy. See qwen35_vlm_model.py for why.
                key = "language_model." + key

            if key.startswith("language_model.model.visual."):
                key = "vision_tower." + key[len("language_model.model.visual.") :]

            if "conv1d.weight" in key and value.shape[-1] != 1:
                # mx.moveaxis goes through the streaming-discovery
                # monkey-patch in omlx.oq when called with _TrackedTensor;
                # the instance method on the tracker doesn't exist.
                value = mx.moveaxis(value, 2, 1)
            if value.ndim == 1 and any(key.endswith(sfx) for sfx in norm_keys):
                # ``key`` is already remapped to ``language_model.mtp.*`` for
                # MTP weights here, so test the ``mtp.`` substring.
                if "mtp." in key:
                    if should_shift_norm_weights:
                        # Raw-HF source: every Qwen3-Next RMSNorm gamma is
                        # zero-centered — shift head norms uniformly like the
                        # backbone's. Per-key mean heuristics misclassify
                        # q_norm/k_norm (raw ~0.75), post_attention_layernorm
                        # (raw ~0.87 on 35B-A3B), and mtp.norm (raw ~1.3-1.9),
                        # costing tens of points of draft acceptance.
                        value = value + 1.0
                    # Pre-converted checkpoints: per-key decision (JANG
                    # mixed-convention bundles).
                    elif _is_oq_tracked_tensor(value):
                        value = _mark_mtp_norm_conditional_add(value)
                    elif _mtp_norm_is_raw_hf(value, should_shift_norm_weights):
                        value = value + 1.0
                elif should_shift_norm_weights:
                    value = value + 1.0

            sanitized_weights[key] = value

        return sanitized_weights

    cls.sanitize = sanitize
    cls._omlx_mtp_vlm_patched = True
    _APPLIED = True
    logger.info("Patched mlx_vlm.models.qwen3_5_moe.Model.sanitize for MTP")
    return True
