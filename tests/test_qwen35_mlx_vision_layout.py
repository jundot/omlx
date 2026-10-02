"""Regression tests for Qwen3.5 MLX-format vision patch embeddings."""

import json
from pathlib import Path

import mlx_vlm.utils as _vu
import numpy as np
import pytest

from omlx.engine.vlm import _transpose_qwen35_mlx_vision_patch_embed_on_load


def _model_dir(tmp_path: Path, *, model_type="qwen3_5", mlx_format=True) -> Path:
    from safetensors.numpy import save_file

    model_dir = tmp_path / model_type
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": model_type}))
    metadata = {"format": "mlx"} if mlx_format else None
    save_file(
        {"placeholder": np.zeros((1,), dtype=np.float32)},
        str(model_dir / "model.safetensors"),
        metadata=metadata,
    )
    return model_dir


def _loader_for(weight):
    def _loader(_):
        return {"vision_tower.patch_embed.proj.weight": weight}

    return _loader


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe"])
def test_transposes_channels_first_qwen35_patch_embed(
    tmp_path, monkeypatch, model_type
):
    model_dir = _model_dir(tmp_path, model_type=model_type)
    weight = np.zeros((1152, 3, 2, 16, 16), dtype=np.float32)
    loader = _loader_for(weight)
    monkeypatch.setattr(_vu, "_load_safetensors", loader)

    with _transpose_qwen35_mlx_vision_patch_embed_on_load(model_dir):
        result = _vu._load_safetensors("model-vision.safetensors")

    assert _vu._load_safetensors is loader
    assert result["vision_tower.patch_embed.proj.weight"].shape == (
        1152,
        2,
        16,
        16,
        3,
    )


def test_preserves_already_correct_patch_embed(tmp_path, monkeypatch):
    model_dir = _model_dir(tmp_path)
    weight = np.zeros((1152, 2, 16, 16, 3), dtype=np.float32)
    loader = _loader_for(weight)
    monkeypatch.setattr(_vu, "_load_safetensors", loader)

    with _transpose_qwen35_mlx_vision_patch_embed_on_load(model_dir):
        result = _vu._load_safetensors("model-vision.safetensors")

    assert result["vision_tower.patch_embed.proj.weight"] is weight


def test_noop_for_non_mlx_checkpoint(tmp_path, monkeypatch):
    model_dir = _model_dir(tmp_path, mlx_format=False)
    weight = np.zeros((1152, 3, 2, 16, 16), dtype=np.float32)
    loader = _loader_for(weight)
    monkeypatch.setattr(_vu, "_load_safetensors", loader)

    with _transpose_qwen35_mlx_vision_patch_embed_on_load(model_dir):
        assert _vu._load_safetensors is loader

    assert _vu._load_safetensors is loader


def test_noop_for_other_model_type(tmp_path, monkeypatch):
    model_dir = _model_dir(tmp_path, model_type="qwen3_vl")
    weight = np.zeros((1152, 3, 2, 16, 16), dtype=np.float32)
    loader = _loader_for(weight)
    monkeypatch.setattr(_vu, "_load_safetensors", loader)

    with _transpose_qwen35_mlx_vision_patch_embed_on_load(model_dir):
        assert _vu._load_safetensors is loader

    assert _vu._load_safetensors is loader
