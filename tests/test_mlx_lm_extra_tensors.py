# SPDX-License-Identifier: Apache-2.0
"""Tests for the MTP sidecar glob patch (issue #2062).

Covers ``sidecar_files_for`` (config + index scan), model-scoped
``_GlobProxy`` augmentation, ``apply()`` (idempotent install), and the
``maybe_apply_pre_load_patches`` dispatch wiring.
"""

from __future__ import annotations

import json

import pytest

from omlx.patches import mlx_lm_extra_tensors as extra_tensors


def _write_index(tmp_path, weight_map: dict) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )


def _write_config(tmp_path, config: dict) -> None:
    (tmp_path / "config.json").write_text(json.dumps(config))


class TestSidecarFilesFor:
    def test_returns_empty_when_no_index(self, tmp_path):
        assert extra_tensors.sidecar_files_for(tmp_path) == []

    def test_returns_empty_on_malformed_index(self, tmp_path):
        (tmp_path / "model.safetensors.index.json").write_text("{not valid")
        assert extra_tensors.sidecar_files_for(tmp_path) == []

    def test_returns_empty_on_malformed_config(self, tmp_path):
        (tmp_path / "config.json").write_text("{not valid")
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

    def test_finds_nested_sidecar_declared_by_config(self, tmp_path):
        _write_config(
            tmp_path,
            {
                "mlx_lm_extra_tensors": {
                    "mtp_file": "mtp/weights.safetensors",
                    "mtp_tensor_count": 15,
                }
            },
        )
        assert extra_tensors.sidecar_files_for(tmp_path) == [
            str(tmp_path / "mtp" / "weights.safetensors")
        ]

    def test_merges_and_deduplicates_config_and_index_sidecars(self, tmp_path):
        _write_config(
            tmp_path,
            {"mlx_lm_extra_tensors": {"mtp_file": "mtp/weights.safetensors"}},
        )
        _write_index(
            tmp_path,
            {
                "mtp.fc.weight": "mtp/weights.safetensors",
                "mtp.norm.weight": "mtp-extra.safetensors",
            },
        )
        assert extra_tensors.sidecar_files_for(tmp_path) == [
            str(tmp_path / "mtp-extra.safetensors"),
            str(tmp_path / "mtp" / "weights.safetensors"),
        ]

    @pytest.mark.parametrize(
        "extra_config",
        [
            None,
            "mtp.safetensors",
            {"mtp_file": None},
            {"mtp_file": 123},
            {"mtp_file": ""},
        ],
    )
    def test_ignores_invalid_config_declaration(self, tmp_path, extra_config):
        _write_config(tmp_path, {"mlx_lm_extra_tensors": extra_config})
        assert extra_tensors.sidecar_files_for(tmp_path) == []

    def test_ignores_sidecar_path_outside_model_directory(self, tmp_path):
        _write_config(
            tmp_path,
            {"mlx_lm_extra_tensors": {"mtp_file": "../outside.safetensors"}},
        )
        assert extra_tensors.sidecar_files_for(tmp_path) == []

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


class TestGlobProxy:
    def test_passthrough_for_unrelated_pattern(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        proxy = extra_tensors._GlobProxy()
        assert proxy.glob(str(tmp_path / "*.json")) == [str(tmp_path / "config.json")]

    def test_augments_model_safetensors_pattern(self, tmp_path):
        (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"")
        sidecar = str(tmp_path / "mtp.safetensors")
        (tmp_path / "mtp.safetensors").write_bytes(b"")
        _write_index(tmp_path, {"mtp.fc.weight": "mtp.safetensors"})

        proxy = extra_tensors._GlobProxy()
        matches = proxy.glob(str(tmp_path / "model*.safetensors"))
        assert set(matches) == {
            str(tmp_path / "model-00001-of-00001.safetensors"),
            sidecar,
        }

    def test_no_augmentation_when_model_has_no_sidecars(self, tmp_path):
        (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"")
        proxy = extra_tensors._GlobProxy()
        matches = proxy.glob(str(tmp_path / "model*.safetensors"))
        assert matches == [str(tmp_path / "model-00001-of-00001.safetensors")]

    def test_does_not_duplicate_a_match_already_found(self, tmp_path):
        shard = tmp_path / "model-00001-of-00001.safetensors"
        shard.write_bytes(b"")
        _write_index(tmp_path, {"mtp.fc.weight": shard.name})

        proxy = extra_tensors._GlobProxy()
        matches = proxy.glob(str(tmp_path / "model*.safetensors"))
        assert matches == [str(shard)]

    def test_resolves_sidecars_from_each_pattern_model_directory(self, tmp_path):
        target = tmp_path / "target"
        target.mkdir()
        target_shard = target / "model.safetensors"
        target_shard.write_bytes(b"")
        target_sidecar = target / "mtp.safetensors"
        target_sidecar.write_bytes(b"")
        _write_index(target, {"mtp.fc.weight": target_sidecar.name})

        draft = tmp_path / "draft"
        draft.mkdir()
        draft_shard = draft / "model.safetensors"
        draft_shard.write_bytes(b"")

        proxy = extra_tensors._GlobProxy()
        assert set(proxy.glob(str(target / "model*.safetensors"))) == {
            str(target_shard),
            str(target_sidecar),
        }
        assert proxy.glob(str(draft / "model*.safetensors")) == [str(draft_shard)]

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
            _write_index(tmp_path, {"mtp.fc.weight": sidecar.name})
            matches = mlx_lm_utils.glob.glob(str(tmp_path / "model*.safetensors"))
            assert str(sidecar) in matches
        finally:
            mlx_lm_utils.glob = original_glob


class TestMaybeApplyPreLoadPatchesDispatch:
    """Integration: the loader-level dispatch in model_loading.py."""

    def _write_mtp_model(
        self,
        tmp_path,
        *,
        indexed_sidecar: bool = False,
        config_sidecar: str | None = None,
    ):
        config = {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "mtp_num_hidden_layers": 1,
                "num_hidden_layers": 2,
            },
        }
        if config_sidecar is not None:
            config["mlx_lm_extra_tensors"] = {"mtp_file": config_sidecar}
        _write_config(tmp_path, config)
        weight_map = {"model.embed_tokens.weight": "model-00001-of-00001.safetensors"}
        (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"")
        if indexed_sidecar:
            weight_map["mtp.fc.weight"] = "mtp.safetensors"
            (tmp_path / "mtp.safetensors").write_bytes(b"")
        if config_sidecar is not None:
            sidecar_path = tmp_path / config_sidecar
            sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            sidecar_path.write_bytes(b"")
        _write_index(tmp_path, weight_map)

    def test_installs_proxy_for_indexed_sidecar(self, tmp_path):
        from omlx.utils.model_loading import maybe_apply_pre_load_patches

        try:
            from mlx_lm import utils as mlx_lm_utils
        except ImportError:
            pytest.skip("mlx-lm not importable")

        self._write_mtp_model(tmp_path, indexed_sidecar=True)
        original_glob = mlx_lm_utils.glob
        try:
            maybe_apply_pre_load_patches(str(tmp_path))
            matches = mlx_lm_utils.glob.glob(str(tmp_path / "model*.safetensors"))
            assert str(tmp_path / "mtp.safetensors") in matches
        finally:
            mlx_lm_utils.glob = original_glob

    def test_installs_proxy_for_config_declared_nested_sidecar(self, tmp_path):
        from omlx.utils.model_loading import maybe_apply_pre_load_patches

        try:
            from mlx_lm import utils as mlx_lm_utils
        except ImportError:
            pytest.skip("mlx-lm not importable")

        self._write_mtp_model(tmp_path, config_sidecar="mtp/weights.safetensors")
        original_glob = mlx_lm_utils.glob
        try:
            maybe_apply_pre_load_patches(str(tmp_path))
            matches = mlx_lm_utils.glob.glob(str(tmp_path / "model*.safetensors"))
            assert str(tmp_path / "mtp" / "weights.safetensors") in matches
        finally:
            mlx_lm_utils.glob = original_glob

    def test_target_sidecar_does_not_leak_into_specprefill_draft(
        self, tmp_path, monkeypatch
    ):
        """DeepSeek-patched second load stays scoped without redispatch."""
        from omlx.patches.deepseek_v4.utils_patch import apply_utils_patch
        from omlx.utils.model_loading import maybe_apply_pre_load_patches

        try:
            from mlx_lm import utils as mlx_lm_utils
        except ImportError:
            pytest.skip("mlx-lm not importable")

        target = tmp_path / "target"
        target.mkdir()
        self._write_mtp_model(target, indexed_sidecar=True)

        draft = tmp_path / "draft"
        draft.mkdir()
        draft_shard = draft / "model.safetensors"
        draft_shard.write_bytes(b"")
        _write_config(draft, {"model_type": "llama"})

        class FakeModelArgs:
            @classmethod
            def from_dict(cls, config):
                return cls()

        class FakeModel:
            def __init__(self, args):
                self.loaded_weights = []

            def eval(self):
                pass

            def load_weights(self, weights, strict=True):
                self.loaded_weights = list(weights)

        loaded_files = []
        monkeypatch.setattr(
            mlx_lm_utils.mx,
            "load",
            lambda path: loaded_files.append(str(path)) or {},
        )

        def get_model_classes(config):
            return FakeModel, FakeModelArgs

        original_glob = mlx_lm_utils.glob
        try:
            # DeepSeek V4 permanently replaces mlx-lm's loader. Its replacement
            # must still resolve the live mlx_lm.utils.glob proxy.
            apply_utils_patch()
            mlx_lm_utils.glob = extra_tensors._GlobProxy()
            maybe_apply_pre_load_patches(str(target))
            mlx_lm_utils.load_model(
                target,
                lazy=True,
                get_model_classes=get_model_classes,
            )
            target_files = set(loaded_files)

            loaded_files.clear()
            mlx_lm_utils.load_model(
                draft,
                lazy=True,
                get_model_classes=get_model_classes,
            )

            assert target_files == {
                str(target / "model-00001-of-00001.safetensors"),
                str(target / "mtp.safetensors"),
            }
            assert loaded_files == [str(draft_shard)]
        finally:
            mlx_lm_utils.glob = original_glob
