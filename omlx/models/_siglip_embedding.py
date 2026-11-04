# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_vlm.encoder_loader import _weight_files
from mlx_vlm.models.granite_vision.config import VisionConfig as GraniteVisionConfig
from mlx_vlm.models.granite_vision.vision import VisionModel
from mlx_vlm.models.pooling import normalize_embeddings
from mlx_vlm.models.siglip.config import VisionConfig as SiglipVisionConfig
from mlx_vlm.utils import _quantization_for_module_path

from ..utils.image import load_image


class SiglipImageEncoder:
    def __init__(self, vision: VisionModel, image_processor):
        self.vision = vision
        self.image_processor = image_processor

    def embed(self, items: list[dict[str, str]]) -> mx.array:
        images = [load_image(item["image"], field="items[].image") for item in items]
        pixels = np.asarray(
            self.image_processor(images=images, return_tensors="np")["pixel_values"]
        )
        channels = self.vision.vision_model.embeddings.config.num_channels
        if pixels.ndim != 4:
            raise ValueError("SigLIP image processor must return a batch of images")
        if pixels.shape[1] == channels:
            pixels = pixels.transpose(0, 2, 3, 1)
        elif pixels.shape[-1] != channels:
            raise ValueError("SigLIP image processor returned an invalid channel count")
        dtype = self.vision.vision_model.embeddings.patch_embedding.weight.dtype
        return embed_siglip_images(self.vision, mx.array(pixels).astype(dtype))


def set_siglip_activation(vision: VisionModel, approximation: str) -> None:
    for layer in vision.vision_model.encoder.layers:
        layer.mlp.activation_fn = nn.GELU(approx=approximation)
    vision.vision_model.head.mlp.activation_fn = nn.GELU(approx=approximation)


def load_siglip_vision(model_path: Path, config: dict[str, Any]) -> VisionModel | None:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open() as handle:
            weight_map = json.load(handle).get("weight_map", {})
        missing = sorted(
            shard
            for shard in set(weight_map.values())
            if not (model_path / shard).is_file()
        )
        if missing:
            raise ValueError(f"Missing SigLIP weight shards: {', '.join(missing)}")

    weight_files = [(Path(path), "") for path in _weight_files(model_path)]
    weight_files.extend(
        (path, f"{path.parent.name}.")
        for path in sorted(model_path.glob("*/*.safetensors"))
    )
    weights = {}
    for path, prefix in weight_files:
        for key, value in mx.load(str(path)).items():
            key = prefix + key
            if not key.startswith("vision_model.") or "position_ids" in key:
                continue
            if key.startswith("vision_model.vision_model."):
                key = key.removeprefix("vision_model.")
            key = key.replace(
                "head.attention.in_proj_weight", "head.attention.in_proj.weight"
            )
            key = key.replace(
                "head.attention.in_proj_bias", "head.attention.in_proj.bias"
            )
            if key in weights:
                raise ValueError(f"Duplicate SigLIP vision weight: {key}")
            weights[key] = value
    if not weights:
        return None

    vision_config = config.get("vision_config") or {}
    if vision_config.get("vision_use_head") is False:
        return None
    siglip_config = SiglipVisionConfig.from_dict(vision_config)
    vision = VisionModel(GraniteVisionConfig.from_dict(asdict(siglip_config)))

    activation = vision_config.get("hidden_act", "gelu_pytorch_tanh")
    approximations = {
        "gelu_pytorch_tanh": "precise",
        "gelu_new": "precise",
        "gelu": "none",
    }
    if activation not in approximations:
        raise ValueError(f"Unsupported SigLIP vision activation: {activation!r}")
    set_siglip_activation(vision, approximations[activation])

    patch_key = "vision_model.embeddings.patch_embedding.weight"
    if patch_key in weights:
        weight = weights[patch_key]
        channels = siglip_config.num_channels
        patch = siglip_config.patch_size
        pytorch_shape = (siglip_config.hidden_size, channels, patch, patch)
        mlx_shape = (siglip_config.hidden_size, patch, patch, channels)
        if weight.shape == pytorch_shape and weight.shape == mlx_shape:
            raise ValueError("Ambiguous SigLIP patch embedding weight layout")
        if weight.shape == pytorch_shape:
            weights[patch_key] = weight.transpose(0, 2, 3, 1)

    quantization = config.get("quantization")
    if quantization is not None:
        if not isinstance(quantization, dict):
            raise ValueError("Invalid SigLIP quantization configuration")
        quantization = dict(quantization)
        for key in list(quantization):
            if key.startswith("vision_model.vision_model."):
                alias = key.removeprefix("vision_model.")
                if alias in quantization and quantization[alias] != quantization[key]:
                    raise ValueError(
                        f"Conflicting SigLIP quantization settings: {alias}"
                    )
                quantization[alias] = quantization.pop(key)

        def quantization_predicate(path, module):
            if not hasattr(module, "to_quantized"):
                return False
            recipe = _quantization_for_module_path(quantization, path, vision)
            if recipe is not None:
                return recipe
            if hasattr(module, "weight") and module.weight.size % 64 != 0:
                return False
            return f"{path}.scales" in weights

        nn.quantize(
            vision,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization.get("mode", "affine"),
            class_predicate=quantization_predicate,
        )

    expected = dict(tree_flatten(vision.parameters()))
    for key, value in weights.items():
        target = expected.get(key)
        if target is not None:
            if mx.issubdtype(target.dtype, mx.floating):
                compatible = mx.issubdtype(value.dtype, mx.floating)
            else:
                compatible = value.dtype == target.dtype
            if not compatible:
                raise ValueError(
                    f"Invalid dtype for SigLIP vision weight {key}: "
                    f"{value.dtype}, expected {target.dtype}"
                )
    vision.load_weights(list(weights.items()), strict=True)
    vision.eval()
    mx.eval(vision.parameters())
    return vision


def embed_siglip_images(vision: VisionModel, pixel_values: mx.array) -> mx.array:
    patches = vision(pixel_values)[0]
    head = vision.vision_model.head
    probe = mx.repeat(head.probe, patches.shape[0], axis=0)
    queries, _, _ = mx.split(head.attention.in_proj(probe), 3, axis=-1)
    _, keys, values = mx.split(head.attention.in_proj(patches), 3, axis=-1)
    batch, length, dimensions = queries.shape
    heads = head.attention.num_heads
    queries = queries.reshape(batch, length, heads, -1).transpose(0, 2, 1, 3)
    keys = keys.reshape(batch, patches.shape[1], heads, -1).transpose(0, 2, 1, 3)
    values = values.reshape(batch, patches.shape[1], heads, -1).transpose(0, 2, 1, 3)
    pooled = mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=head.attention.scale
    )
    pooled = pooled.transpose(0, 2, 1, 3).reshape(batch, length, dimensions)
    pooled = head.attention.out_proj(pooled)
    pooled = pooled + head.mlp(head.layernorm(pooled))
    return normalize_embeddings(pooled[:, 0])
