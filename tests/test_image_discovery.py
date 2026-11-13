# SPDX-License-Identifier: Apache-2.0
"""Image pipeline discovery regressions, without inference or model weights."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from omlx.image_bundle import PIPELINE, SIDECAR


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def tensor(path):
    # Discovery only checks for weight-file presence.
    path.write_bytes(b"fixture")


class DiscoveryTests(unittest.TestCase):
    def test_discovery_layouts(self):
        from omlx.model_discovery import discover_models

        for layout in ("direct", "flat", "grouped", "cache"):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                bundle = {
                    "direct": root,
                    "flat": root / "image",
                    "grouped": root / "org" / "image",
                    "cache": root / "models--owner--image" / "snapshots" / "a",
                }.get(layout)
                if layout == "cache":
                    bundle = root / "models--owner--image" / "snapshots" / "a"
                bundle.mkdir(parents=True, exist_ok=True)
                write_json(bundle / "pipeline.json", {"pipeline": PIPELINE})
                for name in ("text_encoder", "transformer", "vae"):
                    write_json(bundle / name / "config.json", {"model_type": "qwen2"})
                    tensor(bundle / name / "model.safetensors")
                with (
                    patch("omlx.model_discovery._register_model") as register,
                    patch(
                        "omlx.model_discovery._register_gguf_folder", return_value=False
                    ),
                ):
                    self.assertEqual(discover_models(root), {})
                    register.assert_not_called()

    def test_sidecar_precedes_gguf(self):
        from omlx.model_discovery import discover_models

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "image"
            bundle.mkdir()
            (bundle / SIDECAR).write_text("{")
            (bundle / "encoder.gguf").write_bytes(b"GGUF")
            with patch(
                "omlx.model_discovery._register_gguf_folder", return_value=False
            ) as register:
                discover_models(root)
                self.assertNotIn(bundle, [c.args[1] for c in register.call_args_list])

    def test_ordinary_chat_and_gguf_still_reach_registration(self):
        from omlx.model_discovery import discover_models

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "chat" / "config.json", {"model_type": "qwen2"})
            tensor(root / "chat" / "model.safetensors")
            with (
                patch("omlx.model_discovery._register_model") as register,
                patch(
                    "omlx.model_discovery._register_gguf_folder", return_value=False
                ) as gguf,
            ):
                discover_models(root)
                register.assert_called_once()
                self.assertIn(root / "chat", [c.args[1] for c in gguf.call_args_list])


# Exercise actual discovery/registration as well as boundary ordering above.


@pytest.mark.parametrize("layout", ["direct", "flat", "grouped", "cache"])
@pytest.mark.parametrize("marker", ["legacy", "sidecar", "broken", "diffusers"])
def test_real_discovery_keeps_pipeline_atomic(tmp_path, layout, marker):
    from omlx.model_discovery import discover_models

    bundle = {
        "direct": tmp_path,
        "flat": tmp_path / "image",
        "grouped": tmp_path / "org" / "image",
        "cache": tmp_path / "models--owner--image" / "snapshots" / "abc",
    }[layout]
    bundle.mkdir(parents=True, exist_ok=True)
    if marker == "legacy":
        write_json(bundle / "pipeline.json", {"pipeline": PIPELINE})
    elif marker == "diffusers":
        write_json(bundle / "model_index.json", {"_class_name": "QwenImagePipeline"})
    else:
        (bundle / SIDECAR).write_text(
            "{" if marker == "broken" else '{"schema_version":999}'
        )
    for name in ("text_encoder", "transformer", "vae"):
        write_json(bundle / name / "config.json", {"model_type": "qwen2"})
        tensor(bundle / name / "model.safetensors")
    assert discover_models(tmp_path) == {}


@pytest.mark.parametrize(
    "config",
    [
        {"_class_name": "QwenImage21Transformer2DModel"},
        {"_class_name": "AutoencoderKLQwenImage21"},
        {"model_type": "qwen3_vl_image_text_encoder"},
    ],
)
def test_known_component_alone(tmp_path, config):
    from omlx.model_discovery import discover_models

    write_json(tmp_path / "config.json", config)
    tensor(tmp_path / "model.safetensors")
    assert discover_models(tmp_path) == {}


def test_ordinary_named_components_remain_models(tmp_path):
    from omlx.model_discovery import discover_models

    for name in ("transformer", "vae", "text_encoder"):
        write_json(tmp_path / "org" / name / "config.json", {"model_type": "qwen2"})
        tensor(tmp_path / "org" / name / "model.safetensors")
    assert set(discover_models(tmp_path)) == {"transformer", "vae", "text_encoder"}


@pytest.mark.parametrize(
    "contents",
    [
        b"[]",
        b"{",
        b"\xff",
        b'{"_class_name":[]}',
        b'{"a":NaN}',
        b'{"a":1,"a":2}',
        b"[" * 2000,
    ],
)
def test_unrelated_bad_metadata_does_not_break_discovery(tmp_path, contents):
    from omlx.image_bundle import image_boundary

    (tmp_path / "model_index.json").write_bytes(contents)
    assert not image_boundary(tmp_path)


def test_pipeline_cannot_register_gguf_component(tmp_path, write_gguf):
    from omlx.model_discovery import discover_models

    bundle = tmp_path / "models" / "image"
    bundle.mkdir(parents=True)
    (bundle / SIDECAR).write_text("{")
    write_gguf(bundle / "encoder.gguf", {"general.architecture": "qwen3vl"})
    generated = tmp_path / "generated"
    assert discover_models(tmp_path / "models", generated) == {}
    assert not generated.exists()
