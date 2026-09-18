# SPDX-License-Identifier: Apache-2.0
# Portions Copyright 2026-present Prism ML, Inc. See NOTICE in this directory.
"""Load Prism Hadamard Qwen3.5 packs without executing checkpoint Python.

Adapted from runtime/{runtime,artifact,vision_artifact}.py in
prism-ml/Ternary-Bonsai-2-27B-mlx-2bit at
3f926b415992eaa2ae9dd7b573706494d6bbf787. Packed projections need a signed
Hadamard activation transform; embeddings need its inverse. Treating these
weights as ordinary affine quantization silently changes the model.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

MODEL_TYPE = "prism_hadamard_qwen35"
_BLOCKS = (0, 512, 1024, 2048, 4096)


def hadamard(x, block, signs, *, inverse=False):
    """Apply the pack's orthonormal signed transform, accumulating in fp32."""
    shape, dtype = x.shape, x.dtype
    x = x.astype(mx.float32)
    if not inverse:
        x = x * signs
    x = mx.hadamard_transform(x.reshape(-1, block), scale=1 / math.sqrt(block)).reshape(
        shape
    )
    if inverse:
        x = x * signs
    return x.astype(dtype)


class Packed(nn.Module):
    """The published 2-bit/group-128 carrier, including its basis transform."""

    def __init__(self, arrays, block=0, signs=None, embedding=False):
        super().__init__()
        self.weight, self.scales, self.biases = arrays
        self.block, self.signs, self.embedding = block, signs, embedding
        self.activation_dtype = None

    def __call__(self, x):
        if self.embedding:
            out = (
                mx.dequantize(
                    self.weight[x.reshape(-1)],
                    self.scales[x.reshape(-1)],
                    self.biases[x.reshape(-1)],
                    group_size=128,
                    bits=2,
                )
                .reshape(*x.shape, -1)
                .astype(mx.float16)
            )
            return (
                hadamard(out, self.block, self.signs, inverse=True)
                if self.block
                else out
            )
        return self.as_linear(x)

    def as_linear(self, x):
        if self.activation_dtype is not None:
            x = x.astype(self.activation_dtype)
        if self.block:
            x = hadamard(x, self.block, self.signs)
        out = mx.quantized_matmul(
            x,
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=128,
            bits=2,
        )
        return out.astype(self.activation_dtype) if self.activation_dtype else out


class _ActivationRMSNorm(nn.RMSNorm):
    def __call__(self, x):
        return super().__call__(x).astype(x.dtype)


class _ActivationConv1d(nn.Conv1d):
    def __call__(self, x):
        return super().__call__(x).astype(x.dtype)


def _enable_fp16_activations(model):
    """Keep checkpoint weights and recurrent state intact; narrow activations.

    FP32 norm/conv weights otherwise promote the entire residual stream and
    attention cache. Keep their arithmetic in FP32, then return to FP16 at
    module boundaries. This is opt-in because it changes rounding.
    """
    for _, module in model.named_modules():
        if isinstance(module, Packed):
            module.activation_dtype = mx.float16
        elif type(module) is nn.RMSNorm:
            module.__class__ = _ActivationRMSNorm
        elif type(module) is nn.Conv1d:
            module.__class__ = _ActivationConv1d


def _validate_config(config, *, is_vlm):
    vision = config.get("components", {}).get("vision", False)
    if config.get("model_type") != MODEL_TYPE or config.get("schema_version") not in (
        1,
        2,
    ):
        raise ValueError("Unsupported Prism Hadamard pack schema")
    if bool(vision) != is_vlm:
        raise ValueError("Prism Hadamard pack requires its matching text/vision loader")
    if config.get("base_model_type", "qwen3_5") != "qwen3_5":
        raise ValueError("Unsupported Prism Hadamard base model")
    if config.get("quantization") != {"bits": 2, "group_size": 128, "mode": "affine"}:
        raise ValueError("Prism Hadamard requires affine 2-bit/group-128 weights")
    if config.get("gdn_activation_layout", "grouped") != "grouped":
        raise ValueError("Unsupported Prism Hadamard GDN layout")
    if not isinstance(config.get("modules"), list) or not config["modules"]:
        raise ValueError("Missing Prism Hadamard module manifest")


def _install_packed(model, records, weights, *, prefix=""):
    modules = dict(model.named_modules())
    replacements, seen = [], set()
    for record in records:
        path = record["path"]
        if path in seen:
            raise ValueError(f"Duplicate packed module: {path}")
        seen.add(path)
        original = modules.get(path)
        if not isinstance(original, (nn.Linear, nn.Embedding)):
            raise ValueError(f"Unsupported packed module target: {path}")
        if record["embedding"] != isinstance(original, nn.Embedding):
            raise ValueError(f"Packed module kind mismatch: {path}")
        if record["dtype"] != "float16":
            raise ValueError("Unsupported Prism Hadamard activation dtype")
        block = record["block"]
        if block not in _BLOCKS:
            raise ValueError("Unsupported Prism Hadamard block size")
        key = prefix + path
        arrays = [
            weights[key + "." + suffix] for suffix in ("weight", "scales", "biases")
        ]
        rows, width = original.weight.shape
        shapes = [(rows, width // 16), (rows, width // 128), (rows, width // 128)]
        if (
            width % 128
            or [a.shape for a in arrays] != shapes
            or arrays[0].dtype != mx.uint32
        ):
            raise ValueError(f"Invalid packed tensor shapes or storage dtype: {path}")
        for array in arrays[1:]:
            if array.dtype not in (mx.float16, mx.float32, mx.bfloat16):
                raise ValueError(f"Invalid affine dtype: {path}")
            if not mx.all(mx.isfinite(array)).item():
                raise ValueError(f"Non-finite affine parameters: {path}")
        signs = weights.get(key + ".signs")
        if block:
            if width % block or signs is None or signs.shape != (width,):
                raise ValueError(f"Invalid transform dimensions: {path}")
            if not mx.all((signs == 1) | (signs == -1)).item():
                raise ValueError(f"Invalid sign values: {path}")
        elif signs is not None:
            raise ValueError(f"Unexpected sign vector: {path}")
        replacements.append((path, Packed(arrays, block, signs, record["embedding"])))
    model.update_modules(tree_unflatten(replacements))


def _build_processor(directory):
    # Use Qwen's PIL path: AutoProcessor cannot resolve the custom model_type.
    from mlx_vlm.models.qwen3_5 import Qwen3VLProcessor
    from mlx_vlm.tokenizer_utils import load_tokenizer
    from mlx_vlm.utils import StoppingCriteria
    from transformers import AutoTokenizer
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
        Qwen2VLImageProcessorPil,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(directory), trust_remote_code=False)
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessorPil.from_pretrained(str(directory)),
        tokenizer=tokenizer,
        video_processor=None,
        chat_template=(directory / "chat_template.jinja").read_text(),
    )
    processor.detokenizer = load_tokenizer(directory, return_tokenizer=False)(tokenizer)
    eos = getattr(tokenizer, "eos_token_ids", None) or tokenizer.eos_token_id
    criteria = StoppingCriteria(eos, tokenizer)
    processor.tokenizer.stopping_criteria = criteria
    processor.stopping_criteria = criteria
    return processor


def load(model_path, *, is_vlm):
    """Return the normal oMLX model/processor pair for a Hadamard checkpoint."""
    directory = Path(model_path)
    config = json.loads((directory / "config.json").read_text())
    _validate_config(config, is_vlm=is_vlm)
    if is_vlm:
        from mlx_vlm.models.qwen3_5 import Model, ModelConfig
        from mlx_vlm.prompt_utils import MODEL_CONFIG

        model = Model(ModelConfig.from_dict(config))
        language_model, prefix = model.language_model, "language_model."
        # All turns, including text-only turns in a vision conversation, must
        # use Qwen's image-aware message format rather than generic fallback.
        MODEL_CONFIG.setdefault(MODEL_TYPE, MODEL_CONFIG["qwen3_5"])
    else:
        from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

        model = TextModel(TextModelArgs.from_dict(config["text_config"]))
        language_model, prefix = model, ""
    shards = sorted(directory.glob("model*.safetensors"))
    if not shards:
        raise ValueError("No Prism Hadamard model weights found")
    weights = {}
    for shard in shards:
        chunk = mx.load(str(shard))
        if weights.keys() & chunk.keys():
            raise ValueError("Duplicate Prism Hadamard tensors across shards")
        weights.update(chunk)
    _install_packed(language_model, config["modules"], weights, prefix=prefix)
    model.load_weights(list(weights.items()), strict=True)
    if os.environ.get("OMLX_PRISM_FP16_ACTIVATIONS", "0") == "1":
        _enable_fp16_activations(language_model)
        model._omlx_prism_activation_signature = "prism_fp16_v1"
    model.eval()
    mx.eval(model.parameters())
    if is_vlm:
        processor = _build_processor(directory)
    else:
        from mlx_lm.utils import load_tokenizer

        processor = load_tokenizer(
            directory, tokenizer_config_extra={"trust_remote_code": False}
        )
    return model, processor
