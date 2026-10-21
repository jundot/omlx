# SPDX-License-Identifier: Apache-2.0
"""Model-index release contracts must not silently accept incompatible weights."""

import pytest

from omlx.image_bundle import bundle_contract, read_json
from omlx.image_inspect import contained, inspect_bundle, tensor_header
from tests import test_image_import as helpers

write_json = helpers.write_json


@pytest.fixture
def bundle():
    fixture = helpers.ImageImportTests()
    fixture.setUp()
    fixture.bundle()
    root = fixture.root.resolve()
    # Independent metadata fixtures, no downloaded or real weight dependencies.
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
    write_json(root / "model_index.json", index)
    write_json(root / "generation_config.json", recipe)
    (root / "omlx-pipeline.json").unlink()
    write_json(
        root / "checksums.json",
        {
            str(p.relative_to(root)): tensor_header(p, True)[2]
            for p in root.rglob("*.safetensors")
        },
    )
    try:
        yield root
    finally:
        fixture.doCleanups()


def test_model_index_only_bundle_inspects_and_loads_contract(bundle):
    report = inspect_bundle(bundle, True)
    assert report["structural_status"] == "valid", report
    assert report["integrity_status"] == "matched_manifest"
    assert bundle_contract(bundle, resolve=contained)["processor"] == "processor"


@pytest.mark.parametrize(
    "file,key,value",
    [
        ("model_index.json", "mlx_layout", "unknown"),
        ("model_index.json", "mlx_format", False),
        ("model_index.json", "transformer", ["arbitrary", "Class"]),
        ("generation_config.json", "num_inference_steps", 4),
        ("generation_config.json", "guidance_scale", True),
        ("generation_config.json", "sigmas", [1, 0]),
        ("transformer/config.json", "quantization", {"bits": 4}),
    ],
)
def test_reject_incompatible_contract(bundle, file, key, value):
    p = bundle / file
    d = read_json(p)
    d[key] = value
    write_json(p, d)
    assert inspect_bundle(bundle)["structural_status"] == "invalid"
    with pytest.raises(ValueError):
        bundle_contract(bundle, resolve=contained)


def test_bad_index_does_not_fall_back_to_legacy(bundle):
    write_json(bundle / "omlx-pipeline.json", {"schema_version": 1})
    (bundle / "model_index.json").write_text("{")
    assert inspect_bundle(bundle)["structural_status"] == "invalid"


def test_missing_hashes_are_not_integrity_success(bundle):
    (bundle / "checksums.json").unlink()
    assert (
        inspect_bundle(bundle, True)["integrity_status"] == "missing_reference_hashes"
    )


def test_generation_metadata_cannot_escape_bundle(bundle, tmp_path):
    external = tmp_path / "external.json"
    external.write_text("{}")
    (bundle / "generation_config.json").unlink()
    (bundle / "generation_config.json").symlink_to(external)
    assert inspect_bundle(bundle)["structural_status"] == "invalid"
