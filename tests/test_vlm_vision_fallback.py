"""Tests for the vision_tower fallback in VLM loading.

Background: a text-only oQ quant of `diffusion_gemma` (oQ's `text_only`,
admin 「仅文本」) strips every `vision_tower.*` tensor and pops `vision_config`
out of `config.json`, but keeps `embed_vision.*`. oMLX still routes that model
type to the VLM engine (mlx-lm has no `diffusion_gemma` class, and the
block-diffusion decode lane only exists there), where mlx-vlm's `load_model`
runs `config.setdefault("vision_config", {})`. Two sites then turn that empty
dict into a *default* Gemma 4 vision tower — `ModelConfig.from_dict`
(`is not None`) and `update_module_configs` after it — so strict loading fails
with "Missing 210 parameters: model.encoder.vision_tower.*", naming parameters
the checkpoint never had.

`_strip_vision_config_if_orphaned` nulls `vision_config` on the finished
`ModelConfig` and drops the checkpoint's orphan `vision_tower.*` /
`embed_vision.*` tensors for the duration of one `vlm_load(...)` call.
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx_vlm.utils as _vu
import pytest

from omlx.engine.vlm import (
    _has_vision_tower_weights,
    _is_vision_tensor_key,
    _strip_vision_config_if_orphaned,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _write_safetensors(
    path: Path,
    keys: list[str],
    *,
    metadata: dict[str, str] | None = None,
) -> None:
    """Write a tiny safetensors file with the given parameter keys."""
    import numpy as np
    from safetensors.numpy import save_file

    payload = {k: np.zeros((1,), dtype=np.float32) for k in keys}
    save_file(payload, str(path), metadata=metadata)


def _build_model_dir(
    tmp_path: Path,
    *,
    name: str,
    has_vision_config: bool,
    has_vision_weights: bool,
    with_embed_vision: bool = True,
) -> Path:
    """A diffusion_gemma-shaped directory: `model.encoder.*` key layout."""
    model_dir = tmp_path / name
    model_dir.mkdir()

    config: dict = {
        "architectures": ["DiffusionGemmaForBlockDiffusion"],
        "model_type": "diffusion_gemma",
        "text_config": {"hidden_size": 32, "num_hidden_layers": 1},
    }
    if has_vision_config:
        config["vision_config"] = {
            "model_type": "gemma4_vision",
            "hidden_size": 1152,
            "num_hidden_layers": 27,
        }
    (model_dir / "config.json").write_text(json.dumps(config))

    keys = ["model.decoder.layers.0.self_attn.q_proj.weight"]
    if with_embed_vision:
        # oQ's text_only keeps the vision projection even without a tower.
        keys.append("model.encoder.embed_vision.embedding_projection.weight")
    if has_vision_weights:
        keys.append("model.encoder.vision_tower.patch_embedder.input_proj.weight")
    _write_safetensors(
        model_dir / "model.safetensors", keys, metadata={"format": "mlx"}
    )

    return model_dir


class _FakeModule:
    """Stands in for the model under `load_weights`."""

    def __init__(self, owned_keys: list[str]):
        self._params = {k: mx.zeros((1,), dtype=mx.float32) for k in owned_keys}

    def parameters(self):
        return self._params


def _capture_load_weights(monkeypatch):
    captured = {}

    def fake_load_weights(self, weights_items, *args, **kwargs):
        captured["items"] = list(weights_items)
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "loaded"

    monkeypatch.setattr(nn.Module, "load_weights", fake_load_weights)
    return captured, fake_load_weights


# ---------------------------------------------------------------------------
# _is_vision_tensor_key / _has_vision_tower_weights
# ---------------------------------------------------------------------------


class TestVisionTensorKey:
    @pytest.mark.parametrize(
        "key",
        [
            "model.encoder.vision_tower.patch_embedder.input_proj.weight",
            "model.encoder.vision_tower.encoder.layers.0.self_attn.q_norm.weight",
            "model.encoder.embed_vision.embedding_projection.weight",
            "vision_tower.layers.0.mlp.up_proj.weight",
            "embed_vision.embedding_projection.scales",
        ],
    )
    def test_matches_vision_paths(self, key: str):
        assert _is_vision_tensor_key(key) is True

    @pytest.mark.parametrize(
        "key",
        [
            "model.decoder.layers.0.self_attn.q_proj.weight",
            "model.encoder.language_model.layers.0.layer_scalar",
            # "vision" as part of a larger path component is not a vision path.
            "model.decoder.vision_projection.weight",
        ],
    )
    def test_ignores_other_paths(self, key: str):
        assert _is_vision_tensor_key(key) is False


class TestHasVisionTowerWeights:
    def test_true_when_tower_present(self, tmp_path: Path):
        model_dir = _build_model_dir(
            tmp_path, name="v1", has_vision_config=True, has_vision_weights=True
        )
        assert _has_vision_tower_weights(model_dir) is True

    def test_false_for_text_only_checkpoint(self, tmp_path: Path):
        # embed_vision alone does not count: the tower is what builds it.
        model_dir = _build_model_dir(
            tmp_path, name="v2", has_vision_config=False, has_vision_weights=False
        )
        assert _has_vision_tower_weights(model_dir) is False

    def test_false_for_empty_dir(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert _has_vision_tower_weights(empty) is False


# ---------------------------------------------------------------------------
# _strip_vision_config_if_orphaned
# ---------------------------------------------------------------------------


class TestStripVisionConfigIfOrphaned:
    def _fake_update_module_configs(self, monkeypatch):
        """Simulate mlx-vlm re-deserializing an empty vision_config."""
        calls = []

        def fake(model_config, model_class, config, modules):
            calls.append(config)
            if config.get("vision_config") is not None:
                model_config.vision_config = "DEFAULT-TOWER"
            return model_config

        monkeypatch.setattr(_vu, "update_module_configs", fake)
        return calls

    def test_passthrough_when_config_declares_vision(
        self, tmp_path: Path, monkeypatch
    ):
        # Healthy VLM: the tower is real, nothing may be stripped.
        model_dir = _build_model_dir(
            tmp_path, name="healthy", has_vision_config=True, has_vision_weights=True
        )
        _capture_load_weights(monkeypatch)
        before_update = _vu.update_module_configs
        before_load = nn.Module.load_weights

        with _strip_vision_config_if_orphaned(model_dir):
            assert _vu.update_module_configs is before_update
            assert nn.Module.load_weights is before_load

    def test_passthrough_when_vision_weights_present(
        self, tmp_path: Path, monkeypatch
    ):
        # Config is silent but the tower is in the shards — the config is the
        # thing that is wrong, not the checkpoint. Leave mlx-vlm alone.
        model_dir = _build_model_dir(
            tmp_path, name="weights-only",
            has_vision_config=False,
            has_vision_weights=True,
        )
        _capture_load_weights(monkeypatch)
        before_update = _vu.update_module_configs

        with _strip_vision_config_if_orphaned(model_dir):
            assert _vu.update_module_configs is before_update

    def test_nulls_fabricated_vision_config(self, tmp_path: Path, monkeypatch):
        model_dir = _build_model_dir(
            tmp_path, name="text-only",
            has_vision_config=False,
            has_vision_weights=False,
        )
        calls = self._fake_update_module_configs(monkeypatch)

        class _ModelConfig:
            vision_config = None

        with _strip_vision_config_if_orphaned(model_dir):
            result = _vu.update_module_configs(_ModelConfig(), object(), {}, ["vision"])

        assert len(calls) == 1
        assert result.vision_config is None

    def test_keeps_vision_config_when_config_has_one(
        self, tmp_path: Path, monkeypatch
    ):
        # Defensive: even inside the patched window, a truthy vision_config is
        # never nulled.
        model_dir = _build_model_dir(
            tmp_path, name="keep", has_vision_config=True, has_vision_weights=True
        )
        self._fake_update_module_configs(monkeypatch)

        class _ModelConfig:
            vision_config = None

        with _strip_vision_config_if_orphaned(model_dir):
            result = _vu.update_module_configs(
                _ModelConfig(), object(), {"vision_config": {"hidden_size": 8}}, ["vision"]
            )

        assert result.vision_config == "DEFAULT-TOWER"

    def test_drops_orphan_vision_tensors_only(self, tmp_path: Path, monkeypatch):
        model_dir = _build_model_dir(
            tmp_path, name="dropped",
            has_vision_config=False,
            has_vision_weights=False,
        )
        captured, fake_load_weights = _capture_load_weights(monkeypatch)
        module = _FakeModule(
            [
                "model.decoder.layers.0.self_attn.q_proj.weight",
                "model.encoder.language_model.layers.0.layer_scalar",
            ]
        )
        weights = [
            ("model.decoder.layers.0.self_attn.q_proj.weight", 1),
            ("model.encoder.language_model.layers.0.layer_scalar", 2),
            ("model.encoder.embed_vision.embedding_projection.weight", 3),
            ("model.encoder.embed_vision.embedding_projection.scales", 4),
        ]

        with _strip_vision_config_if_orphaned(model_dir):
            result = nn.Module.load_weights(module, weights, strict=True)

        assert result == "loaded"
        assert nn.Module.load_weights is fake_load_weights
        assert captured["kwargs"] == {"strict": True}
        assert [k for k, _ in captured["items"]] == [
            "model.decoder.layers.0.self_attn.q_proj.weight",
            "model.encoder.language_model.layers.0.layer_scalar",
        ]

    def test_keeps_vision_tensors_the_model_owns(self, tmp_path: Path, monkeypatch):
        # The filter is "keys with no slot", not "keys that look like vision".
        model_dir = _build_model_dir(
            tmp_path, name="owns-vision",
            has_vision_config=False,
            has_vision_weights=False,
        )
        captured, _ = _capture_load_weights(monkeypatch)
        module = _FakeModule(
            ["model.encoder.embed_vision.embedding_projection.weight"]
        )
        weights = [("model.encoder.embed_vision.embedding_projection.weight", 1)]

        with _strip_vision_config_if_orphaned(model_dir):
            nn.Module.load_weights(module, weights)

        assert captured["items"] == weights

    def test_warning_logged_once(self, tmp_path: Path, monkeypatch, caplog):
        model_dir = _build_model_dir(
            tmp_path, name="warned",
            has_vision_config=False,
            has_vision_weights=False,
        )
        _capture_load_weights(monkeypatch)
        module = _FakeModule(["model.decoder.norm.weight"])
        weights = [("model.encoder.embed_vision.embedding_projection.weight", 1)]

        with caplog.at_level("WARNING"):
            with _strip_vision_config_if_orphaned(model_dir):
                nn.Module.load_weights(module, weights)
                nn.Module.load_weights(module, weights)

        warnings = [
            rec for rec in caplog.records
            if "vision_tower weights missing" in rec.message
        ]
        assert len(warnings) == 1

    def test_attributes_restored_on_normal_exit(self, tmp_path: Path, monkeypatch):
        model_dir = _build_model_dir(
            tmp_path, name="r1", has_vision_config=False, has_vision_weights=False
        )
        _capture_load_weights(monkeypatch)
        before_update = _vu.update_module_configs
        before_load = nn.Module.load_weights

        with _strip_vision_config_if_orphaned(model_dir):
            assert _vu.update_module_configs is not before_update
            assert nn.Module.load_weights is not before_load

        assert _vu.update_module_configs is before_update
        assert nn.Module.load_weights is before_load

    def test_attributes_restored_on_exception(self, tmp_path: Path, monkeypatch):
        model_dir = _build_model_dir(
            tmp_path, name="r2", has_vision_config=False, has_vision_weights=False
        )
        _capture_load_weights(monkeypatch)
        before_update = _vu.update_module_configs
        before_load = nn.Module.load_weights

        with pytest.raises(RuntimeError, match="boom"):
            with _strip_vision_config_if_orphaned(model_dir):
                raise RuntimeError("boom")

        assert _vu.update_module_configs is before_update
        assert nn.Module.load_weights is before_load

    def test_noop_when_config_json_missing(self, tmp_path: Path, monkeypatch):
        model_dir = tmp_path / "no-config"
        model_dir.mkdir()
        _capture_load_weights(monkeypatch)
        before_update = _vu.update_module_configs

        with _strip_vision_config_if_orphaned(model_dir):
            assert _vu.update_module_configs is before_update
