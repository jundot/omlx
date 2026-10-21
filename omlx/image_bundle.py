# SPDX-License-Identifier: Apache-2.0
"""Metadata-only image pipeline boundaries. Never import an inference backend."""

import json
import os
import stat
from pathlib import Path

MAX_METADATA = 16 * 1024 * 1024
SIDECAR = "omlx-pipeline.json"
PIPELINE = "qwen_image_2_1_turbo_text_to_image"
IMAGE_CLASSES = {"QwenImagePipeline", "QwenImageEditPipeline", "QwenImage21Pipeline"}
COMPONENT_CLASSES = {
    "QwenImageTransformer2DModel",
    "QwenImage21Transformer2DModel",
    "AutoencoderKLQwenImage",
    "AutoencoderKLQwenImage21",
}


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path):
    """Read bounded regular-file JSON, including safe handling of FIFOs."""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError(f"not a regular file: {path.name}")
        data = stream.read(MAX_METADATA + 1)
    if len(data) > MAX_METADATA:
        raise ValueError(f"metadata too large: {path.name}")
    value = json.loads(
        data,
        object_pairs_hook=_pairs,
        parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)),
    )
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def image_boundary(path: Path) -> bool:
    """Recognize explicit ownership, not folder names or general model indexes.

    Even a malformed explicit sidecar reserves its directory. Legacy research
    metadata and Diffusers indexes require a known pipeline identifier.
    """
    if os.path.lexists(path / SIDECAR):
        return True
    for filename, key, known in (
        ("pipeline.json", "pipeline", {PIPELINE}),
        ("model_index.json", "_class_name", IMAGE_CLASSES),
        ("config.json", "_class_name", COMPONENT_CLASSES),
        ("config.json", "model_type", {"qwen3_vl_image_text_encoder"}),
    ):
        try:
            if read_json(path / filename).get(key) in known:
                return True
        except (OSError, ValueError, TypeError, RecursionError):
            continue
    return False


def bundle_contract(root, *, resolve, metadata=read_json):
    """Normalize a model index or the legacy research sidecar without imports.

    Model-index component class labels are descriptive, never dynamically imported.
    A known MLX layout is mandatory: the same architecture can have incompatible
    tensor names, packing and convolution layouts in different converters.
    """
    if not os.path.lexists(root / "model_index.json"):
        return metadata(resolve(root, SIDECAR))
    index = metadata(resolve(root, "model_index.json"))
    expected = {
        "_class_name": "QwenImage21Pipeline",
        "mlx_layout": "native-bf16-f32-v1",
        "processor": ["transformers", "Qwen3VLProcessor"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
        "transformer": ["diffusers", "QwenImage21Transformer2DModel"],
        "vae": ["diffusers", "AutoencoderKLQwenImage21"],
    }
    if index.get("mlx_format") is not True or any(
        index.get(k) != v for k, v in expected.items()
    ):
        raise ValueError("unsupported model_index MLX layout or components")
    recipe = metadata(resolve(root, "generation_config.json"))
    expected_recipe = {
        "recipe": "qwen-image-2.1-turbo-euler-8-v1",
        "num_inference_steps": 8,
        "guidance_scale": 1.0,
        "height": 1024,
        "width": 1024,
        "sigmas": [
            1.0,
            0.978453,
            0.95418,
            0.926626,
            0.89508,
            0.845148,
            0.704534,
            0.414568,
            0.0,
        ],
    }
    if any(recipe.get(k) != v for k, v in expected_recipe.items()) or any(
        isinstance(recipe.get(k), bool) for k in expected_recipe
    ):
        raise ValueError("unsupported generation recipe")
    components = {
        name: {"path": name} for name in ("text_encoder", "transformer", "vae")
    }
    for name in components:
        config = metadata(resolve(root, name + "/config.json"))
        if (
            config.get("quantization") is not None
            or config.get("quantization_config") is not None
        ):
            raise ValueError("quantized image weights are not supported by native v1")
    if os.path.lexists(root / "vae/temporal_auxiliary.safetensors"):
        components["vae"]["auxiliary"] = ["temporal_auxiliary.safetensors"]
    hashes = {}
    if os.path.lexists(root / "checksums.json"):
        hashes = metadata(resolve(root, "checksums.json"))
    return {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "layout": index["mlx_layout"],
        "recipe": recipe["recipe"],
        "processor": "processor",
        "components": components,
        "sha256": hashes,
    }
