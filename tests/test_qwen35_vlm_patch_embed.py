# SPDX-License-Identifier: Apache-2.0
"""Idempotent Qwen3.5 vision patch weight layout sanitization."""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")


@pytest.mark.parametrize(
    "sanitizer", ["qwen35_vlm_model", "qwen35_moe_vlm_model", "qwen35_moe_vlm_runtime"]
)
@pytest.mark.parametrize("channels_first", [True, False])
def test_qwen35_patch_embed_sanitize_is_idempotent(
    sanitizer, channels_first, monkeypatch
):
    import importlib

    class DisposableModel:
        pass

    patch = importlib.import_module(f"omlx.patches.mlx_vlm_mtp.{sanitizer}")
    if sanitizer == "qwen35_moe_vlm_runtime":
        patch._patch_vlm_outer_model_sanitize(SimpleNamespace(Model=DisposableModel))
    else:
        family = "qwen3_5_moe" if sanitizer == "qwen35_moe_vlm_model" else "qwen3_5"
        module = importlib.import_module(f"mlx_vlm.models.{family}.{family}")
        monkeypatch.setattr(module, "Model", DisposableModel)
        monkeypatch.setattr(patch, "_APPLIED", False)
        assert patch.apply()
    assert DisposableModel.sanitize.__module__ == patch.__name__
    model = DisposableModel()
    model.config = SimpleNamespace(
        text_config=SimpleNamespace(tie_word_embeddings=True, num_hidden_layers=0),
        vision_config=SimpleNamespace(in_channels=3),
    )
    shape = (4, 3, 2, 16, 16) if channels_first else (4, 2, 16, 16, 3)
    weight = mx.zeros(shape, dtype=mx.bfloat16)
    key = "vision_tower.patch_embed.proj.weight"
    result = model.sanitize({"model.visual.patch_embed.proj.weight": weight})
    assert result[key].shape == (4, 2, 16, 16, 3)
    if not channels_first:
        assert result[key] is weight
    again = model.sanitize(result)
    assert again[key] is result[key]
