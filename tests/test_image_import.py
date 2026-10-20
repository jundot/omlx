# SPDX-License-Identifier: Apache-2.0
"""Offline tests; also runnable with python -S -m unittest discover."""

import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from omlx.image_bundle import PIPELINE, SIDECAR, image_boundary, read_json
from omlx.image_inspect import RECIPE, inspect_bundle, tensor_header


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def tensor(path, header=None, payload=b"\0\0"):
    header = header or {"w": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


class ImageImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def bundle(self):
        manifest = dict(
            schema_version=1,
            pipeline=PIPELINE,
            layout="native-bf16-f32-v1",
            recipe=RECIPE,
            processor="processor",
            components={},
        )
        for name in ("text_encoder", "transformer", "vae"):
            manifest["components"][name] = {"path": name}
            write_json(self.root / name / "config.json", {})
            tensor(self.root / name / "model.safetensors")
        for name in ("tokenizer.json", "tokenizer_config.json"):
            write_json(self.root / "processor" / name, {})
        write_json(self.root / SIDECAR, manifest)
        return manifest

    def test_valid_structure_does_not_claim_serving(self):
        self.bundle()
        report = inspect_bundle(self.root)
        self.assertEqual(report["structural_status"], "valid")
        self.assertFalse(report["serving_enabled"])
        self.assertEqual(report["runtime_status"], "unvalidated")
        self.assertEqual(report["integrity_status"], "not_checked")

    def test_bad_sidecar_reserves_boundary(self):
        (self.root / SIDECAR).write_bytes(b"\xff")
        self.assertTrue(image_boundary(self.root))
        self.assertEqual(inspect_bundle(self.root)["structural_status"], "invalid")

    def test_names_and_unrelated_indexes_not_boundaries(self):
        write_json(
            self.root / "model_index.json", {"_class_name": "NotAnImagePipeline"}
        )
        (self.root / "transformer").mkdir()
        self.assertFalse(image_boundary(self.root))

    def test_explicit_legacy_and_component(self):
        for filename, value in [
            ("pipeline.json", {"pipeline": PIPELINE}),
            ("config.json", {"_class_name": "AutoencoderKLQwenImage21"}),
            ("model_index.json", {"_class_name": "QwenImagePipeline"}),
        ]:
            with self.subTest(filename=filename):
                write_json(self.root / filename, value)
                self.assertTrue(image_boundary(self.root))
                (self.root / filename).unlink()

    def test_invalid_json(self):
        for data in [b'{"x":1,"x":2}', b"[]", b'{"x":NaN}', b"\xff"]:
            (self.root / "a.json").write_bytes(data)
            with self.assertRaises(ValueError):
                read_json(self.root / "a.json")

    def test_fifo_does_not_block(self):
        os.mkfifo(self.root / "fifo")
        with self.assertRaises(ValueError):
            read_json(self.root / "fifo")
        with self.assertRaises(ValueError):
            tensor_header(self.root / "fifo", False)

    def test_header_corruption(self):
        path = self.root / "model.safetensors"
        for field, value in [
            ("shape", [-1]),
            ("dtype", "Q4"),
            ("data_offsets", [0, 3]),
        ]:
            head = {"w": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}
            head["w"][field] = value
            tensor(path, head)
            with self.subTest(field=field), self.assertRaises(ValueError):
                tensor_header(path, False)
        path.write_bytes(struct.pack("<Q", 2**63))
        with self.assertRaises(ValueError):
            tensor_header(path, False)

    def test_overlapping_tensors(self):
        item = {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}
        tensor(self.root / "a", {"a": item, "b": item})
        with self.assertRaises(ValueError):
            tensor_header(self.root / "a", False)

    def test_unknown_contract(self):
        manifest = self.bundle()
        for key, value in [
            ("schema_version", True),
            ("layout", "quantized"),
            ("recipe", "custom"),
        ]:
            bad = dict(manifest)
            bad[key] = value
            write_json(self.root / SIDECAR, bad)
            self.assertEqual(inspect_bundle(self.root)["structural_status"], "invalid")

    def test_escape(self):
        manifest = self.bundle()
        for value in ["../outside", "/tmp", "escape"]:
            if value == "escape":
                (self.root / "escape").symlink_to("/tmp", target_is_directory=True)
            manifest["components"]["vae"]["path"] = value
            write_json(self.root / SIDECAR, manifest)
            self.assertEqual(inspect_bundle(self.root)["structural_status"], "invalid")

    def test_index_mismatch(self):
        self.bundle()
        write_json(
            self.root / "vae" / "model.safetensors.index.json",
            {"weight_map": {"missing": "model.safetensors"}},
        )
        self.assertEqual(inspect_bundle(self.root)["structural_status"], "invalid")

    def test_hashes(self):
        manifest = self.bundle()
        self.assertEqual(
            inspect_bundle(self.root, True)["integrity_status"],
            "missing_reference_hashes",
        )
        manifest["sha256"] = {}
        for name in manifest["components"]:
            p = self.root / name / "model.safetensors"
            manifest["sha256"][str(p.relative_to(self.root))] = tensor_header(p, True)[
                2
            ]
        write_json(self.root / SIDECAR, manifest)
        self.assertEqual(
            inspect_bundle(self.root, True)["integrity_status"], "matched_manifest"
        )
        p = self.root / "vae" / "model.safetensors"
        p.write_bytes(p.read_bytes()[:-1] + b"1")
        self.assertEqual(
            inspect_bundle(self.root, True)["structural_status"], "invalid"
        )

    def test_cli_clean_process(self):
        self.bundle()
        command = [
            sys.executable,
            "-S",
            "-m",
            "omlx.cli",
            "inspect-image",
            str(self.root),
            "--json",
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout)["serving_enabled"])
        code = "import omlx.image_inspect, sys; assert not any(x in sys.modules for x in ('mlx','torch','mflux','mlx_audio','mlx_vlm'))"
        subprocess.run([sys.executable, "-S", "-c", code], check=True, timeout=10)

    def test_migration_is_atomic_and_preserves_existing_sidecar(self):
        from tools.migrate_image_bundle import migrate

        manifest = self.bundle()
        (self.root / SIDECAR).unlink()
        legacy = dict(
            format="mlx",
            pipeline=PIPELINE,
            components=manifest["components"],
            sampling=dict(
                steps=8,
                guidance=1.0,
                sampler="euler",
                resolution=[1024, 1024],
                sigmas=[
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
            ),
        )
        write_json(self.root / "pipeline.json", legacy)
        before = (self.root / "vae/model.safetensors").read_bytes()
        migrate(self.root)
        saved = (self.root / SIDECAR).read_bytes()
        with self.assertRaises(FileExistsError):
            migrate(self.root)
        self.assertEqual(saved, (self.root / SIDECAR).read_bytes())
        self.assertEqual(before, (self.root / "vae/model.safetensors").read_bytes())
        self.assertEqual(list(self.root.glob(".pipeline-*")), [])
        self.assertEqual(inspect_bundle(self.root)["structural_status"], "valid")

    def test_auxiliary_inventory(self):
        manifest = self.bundle()
        tensor(
            self.root / "vae/aux.safetensors",
            {"aux": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
            b"0000",
        )
        manifest["components"]["vae"]["auxiliary"] = ["aux.safetensors"]
        write_json(self.root / SIDECAR, manifest)
        self.assertEqual(
            inspect_bundle(self.root)["components"]["vae"]["tensor_count"], 2
        )

    def test_internal_hardlink_counts_once(self):
        self.bundle()
        target = self.root / "vae/model.safetensors"
        target.unlink()
        os.link(self.root / "transformer/model.safetensors", target)
        expected = sum(
            (self.root / name / "model.safetensors").stat().st_size
            for name in ("text_encoder", "transformer")
        )
        self.assertEqual(inspect_bundle(self.root)["physical_weight_bytes"], expected)

    def test_metadata_size_bound(self):
        from unittest.mock import patch

        (self.root / "large.json").write_text('{"value":"long string"}')
        with patch("omlx.image_bundle.MAX_METADATA", 8), self.assertRaises(ValueError):
            read_json(self.root / "large.json")


class InspectionFailureTests(unittest.TestCase):
    setUp = ImageImportTests.setUp
    bundle = ImageImportTests.bundle

    def test_symlink_loop(self):
        manifest = self.bundle()
        (self.root / "loop").symlink_to("loop")
        manifest["components"]["vae"]["path"] = "loop"
        write_json(self.root / SIDECAR, manifest)
        self.assertEqual(inspect_bundle(self.root)["structural_status"], "invalid")

    def test_file_changes_during_inspection(self):
        from unittest.mock import patch

        from omlx.image_inspect import tensor_header as original

        self.bundle()

        def changing(path, verify):
            result = original(path, verify)
            with path.open("ab") as stream:
                stream.write(b"changed")
            return result

        with patch("omlx.image_inspect.tensor_header", side_effect=changing):
            self.assertEqual(inspect_bundle(self.root)["structural_status"], "invalid")

    def test_metadata_changes_between_components(self):
        from unittest.mock import patch

        from omlx.image_inspect import tensor_header as original

        self.bundle()

        def changing(path, verify):
            result = original(path, verify)
            write_json(self.root / SIDECAR, {"schema_version": 999})
            return result

        with patch("omlx.image_inspect.tensor_header", side_effect=changing):
            self.assertEqual(inspect_bundle(self.root)["structural_status"], "invalid")

    def test_cancel_returns_non_success(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from omlx.image_inspect import inspect_command

        with (
            patch("omlx.image_inspect.inspect_bundle", side_effect=KeyboardInterrupt),
            patch("builtins.print"),
        ):
            self.assertEqual(
                inspect_command(
                    SimpleNamespace(path=self.root, verify_hashes=True, json=True)
                ),
                130,
            )

    def test_hashes_cannot_claim_uninspected_files(self):
        manifest = self.bundle()
        manifest["sha256"] = {"outside.safetensors": "0" * 64}
        write_json(self.root / SIDECAR, manifest)
        self.assertEqual(
            inspect_bundle(self.root, True)["structural_status"], "invalid"
        )


if __name__ == "__main__":
    unittest.main()
