"""Strict loading and patch lifetime for text-only VLM checkpoints."""

import json

import mlx.core as mx
import mlx.nn as nn
import mlx_vlm.utils as vu
import pytest
from mlx.utils import tree_flatten
from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

from omlx.engine.vlm import _strip_vision_config_if_orphaned
from omlx.utils.model_loading import maybe_apply_pre_load_patches


_TOWER_KEYS = {
    "tower": "model.encoder.vision_tower.weight",
    "moondream_tower": "model.vision.encoder.blocks.0.attn.qkv.weight",
    "moondream_legacy_tower": "vision_encoder.encoder.model.visual.pos_embed",
}


@pytest.mark.parametrize(
    "case",
    [
        "declared",
        "tower",
        "moondream_tower",
        "moondream_legacy_tower",
        "unreadable",
        "no_config",
    ],
)
def test_leaves_loader_unchanged(tmp_path, case):
    config = {"model_type": "diffusion_gemma"}
    if case == "declared":
        config["vision_config"] = {"hidden_size": 32}
    if case != "no_config":
        (tmp_path / "config.json").write_text(json.dumps(config))
    shard = tmp_path / "model.safetensors"
    if case == "unreadable":
        shard.write_bytes(b"broken")
    else:
        key = _TOWER_KEYS.get(case, "weight")
        mx.save_safetensors(str(shard), {key: mx.zeros((1,))})
    before = (vu.update_module_configs, nn.Module.load_weights)
    with _strip_vision_config_if_orphaned(tmp_path):
        assert (vu.update_module_configs, nn.Module.load_weights) == before


def test_quantized_diffusion_gemma_loads_without_vision(tmp_path):
    config = {
        "model_type": "diffusion_gemma",
        "canvas_length": 4,
        "text_config": {
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "moe_intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 1,
            "head_dim": 32,
            "global_head_dim": 32,
            "num_experts": 4,
            "top_k_experts": 2,
        },
        "quantization": {"bits": 4, "group_size": 32},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    maybe_apply_pre_load_patches(str(tmp_path), for_vlm=True)
    model = Model(ModelConfig.from_dict(config))
    nn.quantize(model, bits=4, group_size=32)
    weights = dict(tree_flatten(model.parameters()))
    weights["model.encoder.embed_vision.embedding_projection.weight"] = mx.zeros(
        (32, 32)
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), weights, metadata={"format": "mlx"}
    )
    before = (vu.update_module_configs, nn.Module.load_weights)
    with _strip_vision_config_if_orphaned(tmp_path):
        loaded = vu.load_model(tmp_path, lazy=True)
    assert (vu.update_module_configs, nn.Module.load_weights) == before
    assert loaded.model.encoder.vision_tower is None
    inputs = mx.array([[2, 3]])
    canvas = mx.array([[4, 5, 6, 7]])
    actual = loaded(inputs, canvas_ids=canvas).logits
    expected = model(inputs, canvas_ids=canvas).logits
    assert mx.all(mx.isfinite(actual)).item()
    assert mx.array_equal(actual, expected).item()
    with pytest.raises(ValueError, match="does not include a vision tower"):
        loaded.model.encoder.get_image_features(mx.zeros((1, 3, 8, 8)))


def test_keeps_owned_vision_weights_and_restores_after_load_error(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "diffusion_gemma"}')
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {"weight": mx.zeros((1,))})
    model = nn.Module()
    model.embed_vision = nn.Linear(2, 2, bias=False)
    weights = [("embed_vision.weight", mx.ones((2, 2)))]
    before = (vu.update_module_configs, nn.Module.load_weights)
    with pytest.raises(ValueError, match="Missing"):
        with _strip_vision_config_if_orphaned(tmp_path):
            model.load_weights(weights, strict=True)
            assert mx.array_equal(model.embed_vision.weight, mx.ones((2, 2))).item()
            model.load_weights([], strict=True)
    assert (vu.update_module_configs, nn.Module.load_weights) == before
