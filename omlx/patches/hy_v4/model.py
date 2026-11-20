# SPDX-License-Identifier: Apache-2.0
"""mlx-vlm model module for Hy4 (``hy_v4_vl``) and Hy4-flash (``hy_v4_flash``).

Registered as ``mlx_vlm.models.hy_v4_vl`` and ``mlx_vlm.models.hy_v4_flash``.
Vision input is not wired yet and the vision tower is skipped at load. The
DFly2 drafter is attached as ``language_model.dfly`` when the checkpoint
ships it.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.base import InputEmbeddingsFeatures

from ..mlx_lm_mtp import get_mtp_depth, is_mtp_active, is_mtp_depth_fixed
from .config import ModelConfig, TextConfig, VisionConfig
from .dfly import DFlyConfig, DFlyModel
from .language import LanguageModel

__all__ = ["Model", "ModelConfig", "TextConfig", "VisionConfig", "LanguageModel"]

_FP8_BLOCK = 128
_SKIPPED_PREFIXES = (
    "vit.",
    "visual.",
    "model.mtp_layers.",
    "model.dfly_layers.confidence_head.",
)
_DFLY_PREFIX = "model.dfly_layers."
_EXPERT_NAMES = ("gate_proj", "up_proj", "down_proj")


def dequantize_block_fp8(weight: mx.array, scale: mx.array) -> mx.array:
    """E4M3 bytes [..., R, C] times a [..., ceil(R/128), ceil(C/128)] scale grid."""
    *lead, rows, cols = weight.shape
    pad_r, pad_c = (-rows) % _FP8_BLOCK, (-cols) % _FP8_BLOCK
    w = mx.from_fp8(weight, dtype=mx.float32)
    if pad_r or pad_c:
        w = mx.pad(w, [(0, 0)] * len(lead) + [(0, pad_r), (0, pad_c)])
    w = w.reshape(
        *lead,
        (rows + pad_r) // _FP8_BLOCK,
        _FP8_BLOCK,
        (cols + pad_c) // _FP8_BLOCK,
        _FP8_BLOCK,
    )
    w = w * scale.astype(mx.float32)[..., :, None, :, None]
    w = w.reshape(*lead, rows + pad_r, cols + pad_c)[..., :rows, :cols]
    return w.astype(mx.bfloat16)


def _fp8_scale_base(key: str) -> str | None:
    if key.endswith("_scale_inv"):
        return key[: -len("_scale_inv")]
    if key.endswith(".weight_scale"):
        return key[: -len("_scale")]
    return None


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config.text_config)
        self.language_model.media_token_ids = (
            config.image_token_id,
            config.video_token_id,
        )
        if config.dfly_num_hidden_layers:
            text = config.text_config
            # Both releases size the draft FFN at 3x and the correction MLP at
            # 1x the hidden size; sanitize checks the shapes.
            dfly = DFlyConfig.from_model(
                config, 3 * text.hidden_size, post_proj_norm=text.is_flash
            )
            self.language_model.dfly = DFlyModel(text, dfly, text.hidden_size)
            self.language_model.configure_mtp(is_mtp_active(), get_mtp_depth())
            self.language_model._omlx_mtp_depth_fixed = is_mtp_depth_fixed()

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        if pixel_values is not None:
            raise ValueError("Hy4 image input is not supported yet")
        embeds = self.language_model.model.embed_tokens(input_ids)
        return InputEmbeddingsFeatures(inputs_embeds=embeds)

    def __call__(self, input_ids, pixel_values=None, mask=None, cache=None, **kwargs):
        if pixel_values is not None:
            raise ValueError("Hy4 image input is not supported yet")
        return self.language_model(input_ids, cache=cache, **kwargs)

    def sanitize(self, weights: dict) -> dict:
        if any(k.startswith("language_model.") for k in weights):
            return weights
        args = self.config.text_config
        weights = {
            k: v for k, v in weights.items() if not k.startswith(_SKIPPED_PREFIXES)
        }
        # mlx-vlm's FP8 transform already turned 2D block-FP8 linears into
        # MXFP8 when it ran; keep the routed experts in the same format then.
        experts_mxfp8 = any(k.endswith(".scales") for k in weights)
        for key in [k for k in weights if _fp8_scale_base(k)]:
            base = _fp8_scale_base(key)
            if base in weights and weights[base].dtype == mx.uint8:
                weights[base] = dequantize_block_fp8(weights[base], weights.pop(key))

        for i in range(args.num_hidden_layers):
            self._sanitize_experts(weights, i, args, experts_mxfp8)
            self._split_kv_b(weights, i, args)
        self._sanitize_dfly(weights, args)

        if args.enable_lm_head_fp32 and "lm_head.weight" in weights:
            head = weights["lm_head.weight"]
            if mx.issubdtype(head.dtype, mx.floating):
                weights["lm_head.weight"] = head.astype(mx.float32)

        out = {}
        for k, v in weights.items():
            if k.startswith(("model.", "lm_head.", "dfly.")):
                k = "language_model." + k
            out[k] = v
        return out

    @staticmethod
    def _sanitize_experts(weights, layer, args, mxfp8):
        prefix = f"model.layers.{layer}.mlp"
        fused = weights.pop(f"{prefix}.experts.gate_up_proj", None)
        if fused is not None:
            mid = fused.shape[-2] // 2
            parts = {
                "gate_proj": fused[:, :mid],
                "up_proj": fused[:, mid:],
                "down_proj": weights.pop(f"{prefix}.experts.down_proj"),
            }
        elif f"{prefix}.experts.0.gate_proj.weight" in weights:
            parts = {
                name: mx.stack(
                    [
                        weights.pop(f"{prefix}.experts.{e}.{name}.weight")
                        for e in range(args.n_routed_experts)
                    ]
                )
                for name in _EXPERT_NAMES
            }
        else:
            return
        for name, w in parts.items():
            target = f"{prefix}.switch_mlp.{name}"
            if mxfp8:
                packed, scales = mx.quantize(w, group_size=32, bits=8, mode="mxfp8")
                weights[f"{target}.weight"] = packed
                weights[f"{target}.scales"] = scales
            else:
                weights[f"{target}.weight"] = w

    def _sanitize_dfly(self, weights, args):
        """Rename ``model.dfly_layers.*`` and detach the drafter when absent."""
        names = [k for k in weights if k.startswith(_DFLY_PREFIX)]
        lm = self.language_model
        if not names:
            if "dfly" in lm:
                del lm["dfly"]
                lm.configure_mtp(False)
            return
        for key in names:
            rest = key[len(_DFLY_PREFIX) :]
            if rest[0].isdigit():
                rest = "layers." + rest
            weights["dfly." + rest] = weights.pop(key)
        # Draft layers have the shape of the target's sparse attention layers.
        dims = args.head_dims(args.attention_kinds.index("sparse"))
        for i in range(len(lm.dfly.layers)):
            self._split_kv_b(
                weights, i, args, prefix=f"dfly.layers.{i}.self_attn", dims=dims
            )
        expected = {
            "dfly.layers.0.mlp.gate_proj.weight": lm.dfly.layers[0].mlp.gate_proj,
            "dfly.hidden_correction.gate_proj.weight": (
                lm.dfly.hidden_correction.gate_proj
            ),
        }
        for name, module in expected.items():
            got = weights.get(name)
            if got is not None and got.shape[0] != module.weight.shape[0]:
                raise ValueError(
                    f"{name} has {got.shape[0]} rows, expected {module.weight.shape[0]}"
                )

    @staticmethod
    def _split_kv_b(weights, layer, args, prefix=None, dims=None):
        prefix = prefix or f"model.layers.{layer}.self_attn"
        w = weights.pop(f"{prefix}.kv_b_proj.weight", None)
        if w is None:
            return
        heads, nope, _, value = dims or args.head_dims(layer)
        w = w.reshape(heads, nope + value, args.kv_lora_rank)
        weights[f"{prefix}.embed_q.weight"] = mx.contiguous(
            w[:, :nope, :].swapaxes(-1, -2)
        )
        weights[f"{prefix}.unembed_out.weight"] = mx.contiguous(w[:, nope:, :])
