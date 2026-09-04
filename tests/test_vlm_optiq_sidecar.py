"""Tests for config-declared OptiQ multimodal sidecar loading."""

import json
from pathlib import Path

import mlx.nn as nn
import numpy as np
import pytest
from safetensors.numpy import save_file

from omlx.engine.vlm import (
    _has_audio_weights,
    _load_optiq_vision_sidecar_on_load,
    _resolve_optiq_vision_sidecar,
)


def _write_safetensors(path: Path, keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: np.zeros((1,), dtype=np.float32) for key in keys}
    save_file(payload, str(path), metadata={"format": "mlx"})


def _build_model_dir(
    tmp_path: Path,
    *,
    sidecar: str | None = "optiq/optiq_vision.safetensors",
    sidecar_keys: list[str] | None = None,
) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config = {
        "model_type": "gemma4",
        "vision_config": {"hidden_size": 16},
    }
    if sidecar is not None:
        config["optiq_vision"] = {
            "sidecar": sidecar,
            "n_tensors": len(sidecar_keys or []),
        }
    (model_dir / "config.json").write_text(json.dumps(config))
    _write_safetensors(
        model_dir / "model.safetensors",
        ["language_model.model.layers.0.self_attn.q_proj.weight"],
    )
    if sidecar is not None and sidecar_keys is not None:
        _write_safetensors(model_dir / sidecar, sidecar_keys)
    return model_dir


def _capture_load_weights(monkeypatch):
    captured = {}

    def fake_load_weights(self, weights_items, *args, **kwargs):
        captured["items"] = list(weights_items)
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "loaded"

    monkeypatch.setattr(nn.Module, "load_weights", fake_load_weights)
    return captured, fake_load_weights


class TestResolveOptiqVisionSidecar:
    def test_resolves_nested_declared_sidecar(self, tmp_path: Path):
        model_dir = _build_model_dir(
            tmp_path,
            sidecar_keys=["vision_tower.blocks.0.attn.qkv.weight"],
        )

        assert (
            _resolve_optiq_vision_sidecar(model_dir)
            == (model_dir / "optiq/optiq_vision.safetensors").resolve()
        )

    def test_returns_none_without_declaration(self, tmp_path: Path):
        model_dir = _build_model_dir(tmp_path, sidecar=None)

        assert _resolve_optiq_vision_sidecar(model_dir) is None

    def test_rejects_path_outside_model_directory(self, tmp_path: Path):
        outside = tmp_path / "outside.safetensors"
        _write_safetensors(outside, ["vision_tower.weight"])
        model_dir = _build_model_dir(tmp_path, sidecar=None)
        config = {
            "model_type": "gemma4",
            "optiq_vision": {"sidecar": "../outside.safetensors"},
        }
        (model_dir / "config.json").write_text(json.dumps(config))

        with pytest.raises(ValueError, match="inside the model directory"):
            _resolve_optiq_vision_sidecar(model_dir)

    def test_rejects_missing_declared_sidecar(self, tmp_path: Path):
        model_dir = _build_model_dir(tmp_path, sidecar_keys=None)

        with pytest.raises(FileNotFoundError, match="sidecar not found"):
            _resolve_optiq_vision_sidecar(model_dir)


class TestLoadOptiqVisionSidecar:
    def test_injects_nested_sidecar(self, tmp_path: Path, monkeypatch):
        model_dir = _build_model_dir(
            tmp_path,
            sidecar_keys=[
                "vision_tower.blocks.0.attn.qkv.weight",
                "embed_vision.embedding_projection.weight",
            ],
        )
        captured, original = _capture_load_weights(monkeypatch)
        root_weights = [("language_model.model.embed_tokens.weight", object())]

        with _load_optiq_vision_sidecar_on_load(model_dir):
            result = nn.Module.load_weights(
                object(),
                root_weights,
                strict=True,
            )

        assert result == "loaded"
        assert nn.Module.load_weights is original
        assert captured["kwargs"] == {"strict": True}
        assert {key for key, _ in captured["items"]} == {
            "language_model.model.embed_tokens.weight",
            "vision_tower.blocks.0.attn.qkv.weight",
            "embed_vision.embedding_projection.weight",
        }

    def test_root_sidecar_is_left_to_native_glob(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        model_dir = _build_model_dir(
            tmp_path,
            sidecar="optiq_vision.safetensors",
            sidecar_keys=["vision_tower.weight"],
        )
        captured, original = _capture_load_weights(monkeypatch)
        root_weights = [("language_model.weight", object())]

        with _load_optiq_vision_sidecar_on_load(model_dir):
            nn.Module.load_weights(object(), root_weights)

        assert nn.Module.load_weights is original
        assert captured["items"] == root_weights

    def test_rejects_duplicate_model_weight(self, tmp_path: Path, monkeypatch):
        duplicate = "vision_tower.blocks.0.attn.qkv.weight"
        model_dir = _build_model_dir(
            tmp_path,
            sidecar_keys=[duplicate],
        )
        _, original = _capture_load_weights(monkeypatch)

        with (
            pytest.raises(
                ValueError,
                match="duplicates model weights",
            ),
            _load_optiq_vision_sidecar_on_load(model_dir),
        ):
            nn.Module.load_weights(object(), [(duplicate, object())])

        assert nn.Module.load_weights is original

    def test_restores_load_weights_on_exception(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        model_dir = _build_model_dir(
            tmp_path,
            sidecar_keys=["vision_tower.weight"],
        )
        _, original = _capture_load_weights(monkeypatch)

        with (
            pytest.raises(
                RuntimeError,
                match="boom",
            ),
            _load_optiq_vision_sidecar_on_load(model_dir),
        ):
            raise RuntimeError("boom")

        assert nn.Module.load_weights is original


def test_audio_weights_are_detected_in_optiq_sidecar(tmp_path: Path):
    model_dir = _build_model_dir(
        tmp_path,
        sidecar_keys=[
            "audio_tower.layers.0.feed_forward1.linear.weight",
            "embed_audio.embedding_projection.weight",
        ],
    )

    assert _has_audio_weights(model_dir) is True
