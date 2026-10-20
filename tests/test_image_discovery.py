# SPDX-License-Identifier: Apache-2.0
"""Image pipeline discovery regressions, without inference or model weights."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
