# SPDX-License-Identifier: Apache-2.0
"""mlx-vlm model module for Hy4 (``hy_v4_vl``) and Hy4-flash (``hy_v4_flash``).

Registered as ``mlx_vlm.models.hy_v4_vl`` and ``mlx_vlm.models.hy_v4_flash``.
Image features replace the image placeholder slots of the prompt embeddings;
the prompt's rotary positions come from ``rope_index``. The DFly2 drafter is
attached as ``language_model.dfly`` when the checkpoint ships it.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm.models.base import InputEmbeddingsFeatures

from ..mlx_lm_mtp import get_mtp_depth, is_mtp_active, is_mtp_depth_fixed
from .config import ModelConfig, TextConfig, VisionConfig
from .dfly import DFlyConfig, DFlyModel
from .language import LanguageModel
from .rope import rope_index
from .vision import VisionModel, grid_rows

__all__ = [
    "Model",
    "ModelConfig",
    "TextConfig",
    "VisionConfig",
    "LanguageModel",
    "VisionModel",
]

_FP8_BLOCK = 128
_SKIPPED_PREFIXES = (
    "visual.",
    "model.mtp_layers.",
    "model.dfly_layers.confidence_head.",
)
_VISION_PREFIX = "vit."
_DFLY_PREFIX = "model.dfly_layers."
_EXPERT_NAMES = ("gate_proj", "up_proj", "down_proj")
_DFLY_FFN_RATIO = 3


# Float tensors oQ keeps at their checkpoint precision (FP32): the
# hyper-connection mixes, attention sinks, the router bias and the compressed
# indexer's key pooling.
_FP32_KEYS = (
    ".hc_attn_layer.",
    ".hc_mlp_layer.",
    ".hc_head.",
    "learnable_sink_param",
    "e_score_correction_bias",
    ".compressed_indexer.",
)


def cast_predicate(key: str) -> bool:
    """Whether oQ may cast this float tensor to the model dtype."""
    return not any(part in key for part in _FP32_KEYS)


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


def _split_kv_b(weights, layer, args, prefix=None, dims=None):
    prefix = prefix or f"model.layers.{layer}.self_attn"
    w = weights.pop(f"{prefix}.kv_b_proj.weight", None)
    if w is None:
        return
    heads, nope, _, value = dims or args.head_dims(layer)
    w = w.reshape(heads, nope + value, args.kv_lora_rank)
    weights[f"{prefix}.embed_q.weight"] = mx.contiguous(w[:, :nope, :].swapaxes(-1, -2))
    weights[f"{prefix}.unembed_out.weight"] = mx.contiguous(w[:, nope:, :])


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.vision_tower = (
            None if config.vision_config is None else VisionModel(config.vision_config)
        )
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
                config, _DFLY_FFN_RATIO * text.hidden_size, post_proj_norm=text.is_flash
            )
            self.language_model.dfly = DFlyModel(text, dfly, text.hidden_size)
            self.language_model.configure_mtp(is_mtp_active(), get_mtp_depth())
            self.language_model._omlx_mtp_depth_fixed = is_mtp_depth_fixed()

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def encode_image(self, pixel_values, image_grid_thw=None, **kwargs):
        """[image tokens, hidden] features of the images, in order."""
        if self.vision_tower is None:
            raise ValueError("This Hy4 checkpoint was loaded without a vision tower")
        if image_grid_thw is None:
            raise ValueError("Hy4 image input needs image_grid_thw")
        return self.vision_tower(pixel_values, image_grid_thw)

    def get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        embeds = self.language_model.model.embed_tokens(input_ids)
        features = kwargs.get("cached_image_features")
        if pixel_values is None and features is None:
            return InputEmbeddingsFeatures(inputs_embeds=embeds)
        if kwargs.get("pixel_values_videos") is not None:
            raise ValueError("Hy4 video input is not supported")
        grids = kwargs.get("image_grid_thw")
        if features is None:
            features = self.encode_image(pixel_values, grids)
        elif grids is None:
            raise ValueError("Hy4 image input needs image_grid_thw")

        ids = np.asarray(input_ids)
        slots = np.flatnonzero(ids == self.config.image_token_id)
        if len(slots) != features.shape[0]:
            raise ValueError(
                f"{features.shape[0]} image features for {len(slots)} image tokens"
            )
        flat = embeds.reshape(-1, embeds.shape[-1])
        flat[mx.array(slots)] = features.astype(embeds.dtype)

        mask = kwargs.get("mask")
        planes, deltas = rope_index(
            ids,
            self.config.image_token_id,
            grid_rows(grids),
            self.config.vision_config.spatial_merge_size,
            None if mask is None else np.asarray(mask),
        )
        return InputEmbeddingsFeatures(
            inputs_embeds=flat.reshape(embeds.shape),
            position_ids=mx.array(planes),
            rope_deltas=mx.array(deltas),
        )

    def __call__(self, input_ids, pixel_values=None, mask=None, cache=None, **kwargs):
        if pixel_values is not None:
            merged = self.get_input_embeddings(
                input_ids, pixel_values, mask=mask, **kwargs
            )
            kwargs["inputs_embeds"] = merged.inputs_embeds
            kwargs["position_ids"] = merged.position_ids
        return self.language_model(input_ids, cache=cache, **kwargs)

    def sanitize(self, weights: dict) -> dict:
        if any(k.startswith("language_model.") for k in weights):
            return weights
        args = self.config.text_config
        weights = {
            k: v for k, v in weights.items() if not k.startswith(_SKIPPED_PREFIXES)
        }
        self._sanitize_vision(weights)
        # mlx-vlm's FP8 transform already turned 2D block-FP8 linears into
        # MXFP8 when it ran; keep the routed experts in the same format then.
        experts_mxfp8 = any(k.endswith(".scales") for k in weights)
        for key in [k for k in weights if _fp8_scale_base(k)]:
            base = _fp8_scale_base(key)
            if base in weights and weights[base].dtype == mx.uint8:
                weights[base] = dequantize_block_fp8(weights[base], weights.pop(key))

        for i in range(args.num_hidden_layers):
            _sanitize_experts(weights, i, args, experts_mxfp8)
            _split_kv_b(weights, i, args)
        self._sanitize_dfly(weights, args)

        out = {}
        for k, v in weights.items():
            if k.startswith(("model.", "lm_head.", "dfly.")):
                k = "language_model." + k
            out[k] = v
        return out

    def _sanitize_vision(self, weights):
        """Rename ``vit.*`` to ``vision_tower.*`` and drop it without a vision config.

        The patch embedding and the 1x1 projector conv become matrices; the 2x2
        projector conv becomes a channels-last kernel.
        """
        for key in [k for k in weights if k.startswith(_VISION_PREFIX)]:
            value = weights.pop(key)
            if self.config.vision_config is None:
                continue
            name = "vision_tower." + key[len(_VISION_PREFIX) :]
            name = name.replace("perceive.proj.0.", "perceive.proj_conv.")
            name = name.replace("perceive.proj.2.", "perceive.proj_out.")
            if value.ndim == 4:
                if name.endswith("proj_conv.weight"):
                    value = value.transpose(0, 2, 3, 1)
                else:
                    value = value.reshape(value.shape[0], -1)
            weights[name] = value

    def _sanitize_dfly(self, weights, args):
        """Rename ``model.dfly_layers.*`` and detach the drafter when absent.

        oQ runs sanitize on a proxy without modules, so the layout comes from
        the config.
        """
        names = [k for k in weights if k.startswith(_DFLY_PREFIX)]
        lm = self.language_model
        if not names:
            if isinstance(lm, nn.Module) and "dfly" in lm:
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
        for i in range(int(self.config.dfly_num_hidden_layers)):
            _split_kv_b(
                weights, i, args, prefix=f"dfly.layers.{i}.self_attn", dims=dims
            )
        expected = {
            "dfly.layers.0.mlp.gate_proj.weight": _DFLY_FFN_RATIO * args.hidden_size,
            "dfly.hidden_correction.gate_proj.weight": args.hidden_size,
        }
        for name, rows in expected.items():
            got = weights.get(name)
            if got is not None and got.shape[0] != rows:
                raise ValueError(f"{name} has {got.shape[0]} rows, expected {rows}")
