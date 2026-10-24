# SPDX-License-Identifier: Apache-2.0
"""Serve mlx-lm's Muse Spark model.

mlx-lm implements Muse Spark with its vision tower and reads the HF, ModelOpt
NVFP4 and mlx-lm checkpoint layouts. ``Model`` adds to mlx-lm's class:

* the mlx-vlm interface the VLM engine calls: ``language_model``,
  ``get_input_embeddings``, and ``encode_image`` for the patchified
  ``pixel_values`` and ``image_grid_thw`` of the Muse Glimmer image processor
* the layout that oQ and oMLX-based conversions wrote before oMLX served
  mlx-lm's model: ``switch_mlp`` experts, separate q, k, v and gate
  projections, and a quantized lm_head

The processor is the mlx-vlm Muse Spark one. It expands each image into the
same ``<|patch|>`` tokens as mlx-lm's own processor.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
from mlx_lm.models import muse_spark
from mlx_lm.utils import load_config, load_model

from ..mlx_vlm_muse_spark_compat import apply_mlx_vlm_muse_spark_compat_patch

try:
    from mlx_vlm.models.base import InputEmbeddingsFeatures
    from mlx_vlm.utils import get_model_path, load_processor
except ImportError:
    InputEmbeddingsFeatures = get_model_path = load_processor = None

_UNSUPPORTED_INPUTS = ("pixel_values_videos", "input_values")

# The earlier layout: (its prefix, the prefix mlx-lm's model reads).
_LEGACY_PREFIXES = (
    ("language_model.model.", "model.language_model."),
    ("language_model.lm_head.", "lm_head."),
    ("model.", "model.language_model."),
    ("vision_", "model.vision_"),
)
_LEGACY_RENAMES = (
    (".mlp.switch_mlp.", ".mlp.experts."),
    (".mlp.post_expert_norm.", ".mlp.experts.post_expert_norm."),
)
_ATTENTION_PARTS = ("q_proj", "k_proj", "v_proj", "gate_proj")


def _is_legacy(names) -> bool:
    return any(".switch_mlp." in name for name in names)


def _affine_format(weights: dict, prefix: str, in_dims: int) -> tuple[int, int]:
    """(bits, group_size) of an affine-quantized layer, from its shapes."""
    weight, scales = weights[f"{prefix}.weight"], weights[f"{prefix}.scales"]
    return weight.shape[-1] * 32 // in_dims, in_dims // scales.shape[-1]


def _dequantize(weights: dict, prefix: str, in_dims: int) -> mx.array:
    bits, group_size = _affine_format(weights, prefix, in_dims)
    return mx.dequantize(
        weights.pop(f"{prefix}.weight"),
        weights.pop(f"{prefix}.scales"),
        weights.pop(f"{prefix}.biases"),
        group_size=group_size,
        bits=bits,
    )


def _unify_attention(weights: dict, attention: str, in_dims: int) -> None:
    """Give q, k, v and the gate one format, so mlx-lm can fuse them.

    The parts in other formats are requantized to the widest one, as oQ now
    writes them. With a BF16 part all four stay BF16.
    """
    parts = [attention + part for part in _ATTENTION_PARTS]
    if not all(f"{part}.scales" in weights for part in parts):
        for part in parts:
            if f"{part}.scales" in weights:
                weights[f"{part}.weight"] = _dequantize(weights, part, in_dims)
        return
    formats = {part: _affine_format(weights, part, in_dims) for part in parts}
    # Widest bits; the finer group size on a tie.
    bits, group_size = max(formats.values(), key=lambda f: (f[0], -f[1]))
    for part, fmt in formats.items():
        if fmt != (bits, group_size):
            weight = _dequantize(weights, part, in_dims)
            (
                weights[f"{part}.weight"],
                weights[f"{part}.scales"],
                weights[f"{part}.biases"],
            ) = mx.quantize(weight, group_size=group_size, bits=bits)


def _from_legacy(weights: dict, hidden_size: int) -> dict:
    """Rename the earlier layout to the HF names mlx-lm's sanitize reads."""
    out = {}
    for key, value in weights.items():
        for old, new in _LEGACY_PREFIXES:
            if key.startswith(old):
                key = new + key[len(old) :]
                break
        for old, new in _LEGACY_RENAMES:
            key = key.replace(old, new)
        out[key] = value
    attentions = {
        key[: -len("q_proj.weight")]
        for key in out
        if key.endswith("self_attn.q_proj.weight") and "vision" not in key
    }
    for attention in attentions:
        _unify_attention(out, attention, hidden_size)
    if "lm_head.scales" in out:
        # mlx-lm casts the lm_head input to the weight dtype, which only works
        # for a float lm_head.
        out["lm_head.weight"] = _dequantize(out, "lm_head", hidden_size)
    return out


class _LanguageModel:
    """The text forward of ``Model`` with the mlx-vlm call signature.

    It is not a module, so the model tree holds the text layers once.
    """

    def __init__(self, model: Model):
        self._model = model

    def __call__(self, inputs, cache=None, inputs_embeds=None, **kwargs):
        return self._model(inputs, cache=cache, input_embeddings=inputs_embeds)

    @property
    def model(self):
        return self._model.model

    @property
    def layers(self):
        return self._model.layers

    @property
    def args(self):
        return self._model.args.text

    def make_cache(self):
        return self._model.make_cache()


class Model(muse_spark.Model):
    @property
    def language_model(self) -> _LanguageModel:
        return _LanguageModel(self)

    def sanitize(self, weights: dict) -> dict:
        if _is_legacy(weights):
            weights = _from_legacy(weights, self.args.text.hidden_size)
        return super().sanitize(weights)

    def encode_image(self, pixel_values, image_grid_thw=None, **kwargs):
        if image_grid_thw is not None:
            pixel_values = self._images(pixel_values, image_grid_thw)
        return super().encode_image(pixel_values)

    def _images(self, patches: mx.array, grid) -> list[mx.array]:
        """The (C, H, W) images of the image processor's patch rows."""
        vision = self.args.vision
        p, t, c = vision.patch_size, vision.patch_temporal, vision.num_channels
        images, start = [], 0
        for _, gh, gw in mx.array(grid).tolist():
            rows = patches[start : start + gh * gw].reshape(gh, gw, t, c, p, p)
            start += gh * gw
            # Each temporal slot holds the same frame.
            image = rows[:, :, 0].transpose(2, 0, 3, 1, 4)
            images.append(image.reshape(c, gh * p, gw * p))
        return images

    def get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        for name in _UNSUPPORTED_INPUTS:
            if kwargs.get(name) is not None:
                raise ValueError(f"Muse Spark does not support {name} in oMLX.")

        embeds = self.model.language_model.embed(input_ids)
        if pixel_values is None:
            return InputEmbeddingsFeatures(inputs_embeds=embeds)

        features = kwargs.get("cached_image_features")
        if features is None:
            features = self.encode_image(pixel_values, kwargs.get("image_grid_thw"))
        mask = input_ids == self.args.image_token_id
        count = int(mask.sum().item())
        if count != features.shape[0]:
            raise ValueError(
                "Image features and image tokens do not match: "
                f"tokens={count}, features={features.shape[0]}"
            )
        # The feature row of an image token: the image tokens before it.
        rows = mx.cumsum(mask.flatten()).reshape(mask.shape) - 1
        rows = features[mx.maximum(rows, 0)].astype(embeds.dtype)
        embeds = mx.where(mask[..., None], rows, embeds)
        return InputEmbeddingsFeatures(inputs_embeds=embeds)


def _weight_names(path: Path) -> list[str]:
    index = path / "model.safetensors.index.json"
    if index.is_file():
        return list(json.loads(index.read_text())["weight_map"])
    names = []
    for shard in sorted(path.glob("model*.safetensors")):
        with open(shard, "rb") as f:
            (size,) = struct.unpack("<Q", f.read(8))
            names.extend(k for k in json.loads(f.read(size)) if k != "__metadata__")
    return names


def load_kwargs(path: str | Path, *, text_only: bool = False) -> dict:
    """``mlx_lm.utils.load_model`` arguments for a Muse Spark checkpoint."""
    path = Path(path)
    config = load_config(path)
    names = _weight_names(path)
    model_config = {}
    if _is_legacy(names) and isinstance(config.get("quantization"), dict):
        # Its per-module map names the earlier modules. The loader infers each
        # format from the weights instead.
        model_config["quantization"] = {
            key: value
            for key, value in config["quantization"].items()
            if not isinstance(value, (dict, bool))
        }
    has_vision = any(".vision_tower." in f".{name}" for name in names)
    if config.get("vision_config") and (text_only or not has_vision):
        model_config["vision_config"] = None
    kwargs = {"get_model_classes": lambda config: (Model, muse_spark.ModelArgs)}
    if model_config:
        kwargs["model_config"] = model_config
    return kwargs


def load(model_path: str | Path, *, lazy: bool = False):
    """Load a Muse Spark checkpoint with mlx-lm and its mlx-vlm processor."""
    # Registers the processor with AutoProcessor and the chat prompt format.
    apply_mlx_vlm_muse_spark_compat_patch()
    path = get_model_path(str(model_path))
    model, config = load_model(path, lazy=lazy, **load_kwargs(path))
    text = config.get("text_config") or {}
    model.config = SimpleNamespace(
        model_type=model.model_type,
        image_token_id=model.args.image_token_id,
        eos_token_id=config.get("eos_token_id", text.get("eos_token_id")),
        vocab_size=model.args.text.vocab_size,
        text_config=model.args.text,
        vision_config=model.args.vision,
    )
    processor = load_processor(path, True, eos_token_ids=model.config.eos_token_id)
    return model, processor
