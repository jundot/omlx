#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Add a sidecar to the pinned research export; never rewrite weights.

Run from a source checkout: python -m tools.migrate_image_bundle PATH
Existing sidecars are never replaced. This is not a model converter.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

from omlx.image_bundle import PIPELINE, SIDECAR, read_json
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
    manifest = dict(
        schema_version=1,
        pipeline=PIPELINE,
        layout="native-bf16-f32-v1",
        recipe=RECIPE,
        processor="processor",
        components=output,
        provenance=components,
    )
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=root, prefix=".pipeline-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Link atomically, failing if the destination already exists.
        os.link(temporary, root / SIDECAR)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return root / SIDECAR


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    args = parser.parse_args()
    print(migrate(args.path))
