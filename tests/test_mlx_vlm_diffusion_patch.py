# SPDX-License-Identifier: Apache-2.0
"""Tests for mlx-vlm diffusion compatibility patches."""

import importlib

import mlx.nn as nn


def test_diffusion_embedding_dequantize_passes_quantization_mode(monkeypatch):
    diffusion_mod = importlib.import_module("mlx_vlm.generate.diffusion")

    from omlx.patches import mlx_vlm_diffusion

    monkeypatch.setattr(mlx_vlm_diffusion, "_APPLIED", False)

    captured = {}

    def fake_dequantize(weight, scales, biases, **kwargs):
        captured["biases"] = biases
        captured.update(kwargs)
        return "dequantized"

    monkeypatch.setattr(diffusion_mod.mx, "dequantize", fake_dequantize)

    assert mlx_vlm_diffusion.apply_mlx_vlm_diffusion_patch() is True

    embedding = nn.QuantizedEmbedding(
        num_embeddings=64,
        dims=352,
        group_size=32,
        bits=4,
        mode="mxfp4",
    )

    result = diffusion_mod._diffusion_soft_embedding_weight(embedding)

    assert result == "dequantized"
    assert captured["group_size"] == 32
    assert captured["bits"] == 4
    assert captured["mode"] == "mxfp4"
    assert captured["biases"] is None


def test_pre_load_dispatch_applies_diffusion_patch(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text('{"model_type": "diffusion_gemma"}')

    from omlx.patches import mlx_vlm_diffusion
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    calls = []
    monkeypatch.setattr(mlx_vlm_diffusion, "_APPLIED", False)
    monkeypatch.setattr(
        mlx_vlm_diffusion,
        "apply_mlx_vlm_diffusion_patch",
        lambda: calls.append(True) or True,
    )

    maybe_apply_pre_load_patches(str(tmp_path), for_vlm=True)

    assert calls == [True]
