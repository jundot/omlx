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
