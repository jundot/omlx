#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Add model-index metadata to the pinned research export; never rewrite weights.

Run from a source checkout: python -m tools.migrate_image_bundle PATH
Existing release metadata is never replaced. This is not a model converter.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

from omlx.image_bundle import PIPELINE, read_json
from omlx.image_inspect import COMPONENTS, RECIPE, contained


def migrate(path):
    root = Path(path).resolve(strict=True)
    legacy = read_json(contained(root, "pipeline.json"))
    if legacy.get("pipeline") != PIPELINE or legacy.get("format") != "mlx":
        raise ValueError("not the supported research export")
    components = legacy.get("components", {})
    output = {}
    for name in COMPONENTS:
        component = components[name]
        directory = contained(root, component["path"])
        read_json(contained(root, str(directory.relative_to(root) / "config.json")))
        output[name] = {"path": component["path"]}
        auxiliary = directory / "temporal_auxiliary.safetensors"
        if name == "vae" and auxiliary.exists():
            contained(root, str(auxiliary.relative_to(root)))
            output[name]["auxiliary"] = ["temporal_auxiliary.safetensors"]
    sampling = legacy.get("sampling", {})
    expected = {
        "steps": 8,
        "guidance": 1.0,
        "sampler": "euler",
        "resolution": [1024, 1024],
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
    if any(sampling.get(k) != v for k, v in expected.items()):
        raise ValueError("research recipe differs from the supported recipe")
    contained(root, "processor/tokenizer.json")
    if any(component["path"] != name for name, component in output.items()):
        raise ValueError("model-index releases require canonical component paths")
    index = {
        "_class_name": "QwenImage21Pipeline",
        "mlx_format": True,
        "mlx_layout": "native-bf16-f32-v1",
        "processor": ["transformers", "Qwen3VLProcessor"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
        "transformer": ["diffusers", "QwenImage21Transformer2DModel"],
        "vae": ["diffusers", "AutoencoderKLQwenImage21"],
    }
    recipe = {
        "recipe": RECIPE,
        "num_inference_steps": 8,
        "guidance_scale": 1.0,
        "height": 1024,
        "width": 1024,
        "sigmas": expected["sigmas"],
    }
    # Publish the model index last. Roll back only files this call created if
    # exclusive publication fails; existing files and weights are untouched.
    payloads = {
        "PROVENANCE.json": {"sources": components},
        "generation_config.json": recipe,
        "model_index.json": index,
    }
    for name in payloads:
        if os.path.lexists(root / name):
            raise FileExistsError(root / name)
    created = []
    temporary = None
    try:
        for name, value in payloads.items():
            with tempfile.NamedTemporaryFile(
                mode="w", dir=root, prefix=".pipeline-", delete=False
            ) as stream:
                temporary = Path(stream.name)
                json.dump(value, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, root / name)
            created.append(root / name)
            temporary.unlink()
            temporary = None
    except BaseException:
        for path in reversed(created):
            path.unlink()
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return root / "model_index.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    args = parser.parse_args()
    print(migrate(args.path))
