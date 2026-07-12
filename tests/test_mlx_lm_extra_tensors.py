# SPDX-License-Identifier: Apache-2.0
"""Tests for the MTP sidecar glob patch (issue #2062).

Covers ``sidecar_files_for`` (index scan), the process-wide file-list
flag, ``_GlobProxy`` (augmented glob), ``apply()`` (idempotent install),
and the ``maybe_apply_pre_load_patches`` dispatch wiring.
"""

from __future__ import annotations

import json

import pytest

from omlx.patches import mlx_lm_extra_tensors as extra_tensors


@pytest.fixture(autouse=True)
def _reset_extra_tensor_files():
    extra_tensors.set_extra_tensor_files([])
    yield
    extra_tensors.set_extra_tensor_files([])


def _write_index(tmp_path, weight_map: dict) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )


class TestSidecarFilesFor:
    def test_returns_empty_when_no_index(self, tmp_path):
        assert extra_tensors.sidecar_files_for(tmp_path) == []

    def test_returns_empty_on_malformed_index(self, tmp_path):
        (tmp_path / "model.safetensors.index.json").write_text("{not valid")
        assert extra_tensors.sidecar_files_for(tmp_path) == []

    def test_returns_empty_when_no_mtp_keys(self, tmp_path):
        _write_index(
            tmp_path,
            {"model.embed_tokens.weight": "model-00001-of-00001.safetensors"},
        )
        assert extra_tensors.sidecar_files_for(tmp_path) == []

    def test_finds_sidecar_referenced_by_bare_mtp_prefix(self, tmp_path):
        # Ornith-1.0-35B-MTPLX layout: index lists mtp.* keys pointing at a
        # sidecar file that doesn't match model*.safetensors.
        _write_index(
            tmp_path,
            {
                "model.embed_tokens.weight": "model-00001-of-00001.safetensors",
                "mtp.fc.weight": "mtp.safetensors",
                "mtp.layers.0.self_attn.q_proj.weight": "mtp.safetensors",
            },
        )
        assert extra_tensors.sidecar_files_for(tmp_path) == [
            str(tmp_path / "mtp.safetensors")
        ]

    def test_finds_sidecar_for_prefixed_mtp_key_variants(self, tmp_path):
        for prefix in (
            "language_model.mtp.",
            "model.mtp.",
            "model.language_model.mtp.",
        ):
            _write_index(tmp_path, {f"{prefix}fc.weight": "mtp.safetensors"})
            assert extra_tensors.sidecar_files_for(tmp_path) == [
                str(tmp_path / "mtp.safetensors")
            ]

    def test_ignores_mtp_keys_already_in_a_model_shard(self, tmp_path):
        # mlx-lm's own glob already covers this file; don't duplicate it.
        (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"")
        _write_index(
            tmp_path,
            {"mtp.fc.weight": "model-00001-of-00001.safetensors"},
        )
        assert extra_tensors.sidecar_files_for(tmp_path) == []

    def test_dedups_and_sorts_multiple_sidecar_files(self, tmp_path):
        _write_index(
            tmp_path,
            {
                "mtp.fc.weight": "mtp-b.safetensors",
                "mtp.norm.weight": "mtp-b.safetensors",
                "mtp.layers.0.mlp.gate_proj.weight": "mtp-a.safetensors",
            },
        )
        assert extra_tensors.sidecar_files_for(tmp_path) == [
            str(tmp_path / "mtp-a.safetensors"),
            str(tmp_path / "mtp-b.safetensors"),
        ]

    def test_ignores_non_string_weight_map_entries(self, tmp_path):
        _write_index(tmp_path, {"mtp.fc.weight": 123})
        assert extra_tensors.sidecar_files_for(tmp_path) == []


class TestExtraTensorFilesFlag:
    def test_default_is_empty(self):
        assert extra_tensors.get_extra_tensor_files() == []

    def test_set_and_get_roundtrip(self):
        extra_tensors.set_extra_tensor_files(["/a/mtp.safetensors"])
        assert extra_tensors.get_extra_tensor_files() == ["/a/mtp.safetensors"]

    def test_get_returns_a_copy(self):
        extra_tensors.set_extra_tensor_files(["/a/mtp.safetensors"])
        files = extra_tensors.get_extra_tensor_files()
        files.append("/b/other.safetensors")
        assert extra_tensors.get_extra_tensor_files() == ["/a/mtp.safetensors"]


class TestGlobProxy:
    def test_passthrough_for_unrelated_pattern(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        proxy = extra_tensors._GlobProxy()
        assert proxy.glob(str(tmp_path / "*.json")) == [str(tmp_path / "config.json")]

    def test_augments_model_safetensors_pattern(self, tmp_path):
        (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"")
        sidecar = str(tmp_path / "mtp.safetensors")
        (tmp_path / "mtp.safetensors").write_bytes(b"")
        extra_tensors.set_extra_tensor_files([sidecar])

        proxy = extra_tensors._GlobProxy()
        matches = proxy.glob(str(tmp_path / "model*.safetensors"))
        assert set(matches) == {
            str(tmp_path / "model-00001-of-00001.safetensors"),
            sidecar,
        }

    def test_no_augmentation_when_flag_empty(self, tmp_path):
        (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"")
        proxy = extra_tensors._GlobProxy()
        matches = proxy.glob(str(tmp_path / "model*.safetensors"))
        assert matches == [str(tmp_path / "model-00001-of-00001.safetensors")]

    def test_does_not_duplicate_a_match_already_found(self, tmp_path):
        shard = tmp_path / "model-00001-of-00001.safetensors"
        shard.write_bytes(b"")
        extra_tensors.set_extra_tensor_files([str(shard)])

        proxy = extra_tensors._GlobProxy()
        matches = proxy.glob(str(tmp_path / "model*.safetensors"))
        assert matches == [str(shard)]

    def test_getattr_delegates_to_real_glob_module(self):
        proxy = extra_tensors._GlobProxy()
        assert proxy.escape("a*b") == extra_tensors._glob_module.escape("a*b")

    def test_marker_attribute_present(self):
        proxy = extra_tensors._GlobProxy()
        assert proxy._omlx_extra_tensors_proxy is True


class TestApply:
    def test_apply_installs_proxy_on_mlx_lm_utils(self):
        try:
            from mlx_lm import utils as mlx_lm_utils
        except ImportError:
            pytest.skip("mlx-lm not importable")

        original_glob = mlx_lm_utils.glob
        try:
            assert extra_tensors.apply() is True
            assert (
                getattr(mlx_lm_utils.glob, "_omlx_extra_tensors_proxy", False) is True
            )
        finally:
            mlx_lm_utils.glob = original_glob

    def test_apply_idempotent(self):
        try:
            from mlx_lm import utils as mlx_lm_utils
        except ImportError:
            pytest.skip("mlx-lm not importable")

        original_glob = mlx_lm_utils.glob
        try:
            assert extra_tensors.apply() is True
            installed = mlx_lm_utils.glob
            assert extra_tensors.apply() is True
            # Second call is a no-op — same proxy instance stays installed.
            assert mlx_lm_utils.glob is installed
        finally:
            mlx_lm_utils.glob = original_glob

    def test_apply_returns_false_when_mlx_lm_not_importable(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "mlx_lm" or name.startswith("mlx_lm."):
                raise ImportError("simulated missing mlx-lm")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert extra_tensors.apply() is False

    def test_installed_proxy_actually_augments_load_model_glob(self, tmp_path):
        try:
            from mlx_lm import utils as mlx_lm_utils
        except ImportError:
            pytest.skip("mlx-lm not importable")

        (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"")
        sidecar = tmp_path / "mtp.safetensors"
        sidecar.write_bytes(b"")

        original_glob = mlx_lm_utils.glob
        try:
            extra_tensors.apply()
            extra_tensors.set_extra_tensor_files([str(sidecar)])
            matches = mlx_lm_utils.glob.glob(str(tmp_path / "model*.safetensors"))
            assert str(sidecar) in matches
        finally:
            mlx_lm_utils.glob = original_glob


class TestMaybeApplyPreLoadPatchesDispatch:
    """Integration: the loader-level dispatch in model_loading.py."""

    def _write_mtp_model(self, tmp_path, *, sidecar: bool):
        config = {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "mtp_num_hidden_layers": 1,
                "num_hidden_layers": 2,
            },
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        weight_map = {"model.embed_tokens.weight": "model-00001-of-00001.safetensors"}
        (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"")
        if sidecar:
            weight_map["mtp.fc.weight"] = "mtp.safetensors"
            (tmp_path / "mtp.safetensors").write_bytes(b"")
        _write_index(tmp_path, weight_map)

    def test_registers_sidecar_for_mtp_model(self, tmp_path):
        from omlx.utils.model_loading import maybe_apply_pre_load_patches

        self._write_mtp_model(tmp_path, sidecar=True)
        maybe_apply_pre_load_patches(str(tmp_path))
        assert extra_tensors.get_extra_tensor_files() == [
            str(tmp_path / "mtp.safetensors")
        ]

    def test_no_sidecar_registered_when_index_has_no_extra_files(self, tmp_path):
        from omlx.utils.model_loading import maybe_apply_pre_load_patches

        self._write_mtp_model(tmp_path, sidecar=False)
        maybe_apply_pre_load_patches(str(tmp_path))
        assert extra_tensors.get_extra_tensor_files() == []

    def test_flag_resets_between_loads(self, tmp_path):
        """A model without MTP heads must not inherit a prior sidecar list."""
        from omlx.utils.model_loading import maybe_apply_pre_load_patches

        mtp_dir = tmp_path / "mtp_model"
        mtp_dir.mkdir()
        self._write_mtp_model(mtp_dir, sidecar=True)
        maybe_apply_pre_load_patches(str(mtp_dir))
        assert extra_tensors.get_extra_tensor_files() != []

        plain_dir = tmp_path / "plain_model"
        plain_dir.mkdir()
        (plain_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
        maybe_apply_pre_load_patches(str(plain_dir))
        assert extra_tensors.get_extra_tensor_files() == []
