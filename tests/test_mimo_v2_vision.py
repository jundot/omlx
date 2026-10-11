# SPDX-License-Identifier: Apache-2.0
"""Tests for the MiMo V2.6 image sidecar."""

import json
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from omlx.patches.mimo_v2.omnimodal import (
    MiMoLanguageAdapter,
    MiMoOmnimodalModel,
    has_vision_sidecar,
    load,
)
from omlx.patches.mimo_v2.vision import VisionAttention, VisionConfig, VisionModel


def _tiny_config(**overrides):
    values = {
        "depth": 2,
        "hidden_size": 8,
        "intermediate_size": 16,
        "out_hidden_size": 12,
        "num_heads": 2,
        "num_key_value_heads": 1,
        "patch_size": 2,
        "temporal_patch_size": 2,
        "in_chans": 3,
        "spatial_merge_size": 2,
        "fullatt_block_indexes": [1],
        "use_sink": True,
        "visual_token_window_size": 1,
        "vit_window_attn_types": [-1, 1],
        "qk_channels": 4,
    }
    values.update(overrides)
    return VisionConfig(**values)


def test_has_vision_sidecar_requires_both_files(tmp_path):
    sidecar = tmp_path / "omnimodal"
    sidecar.mkdir()
    (sidecar / "vision_encoder.safetensors").write_bytes(b"weights")
    assert has_vision_sidecar(tmp_path) is False

    (sidecar / "config.json").write_text("{}")
    assert has_vision_sidecar(tmp_path) is True


def test_vision_config_accepts_upstream_in_channels_alias():
    config = VisionConfig.from_dict(
        {
            "hidden_size": 32,
            "in_channels": 1,
            "unrelated_future_field": "ignored",
        }
    )

    assert config.hidden_size == 32
    assert config.in_chans == 1


def test_window_attention_mask_matches_mimo_sink_semantics():
    attention = VisionAttention(_tiny_config(), use_sink=True)
    attention.sinks = mx.array([2.0, 3.0])

    mask = attention._mask(4, mx.float32, full_attention=False)
    mx.eval(mask)

    assert mask.shape == (1, 2, 4, 4)
    assert mask[0, 0, 0].tolist() == [2.0, 0.0, float("-inf"), float("-inf")]
    assert mask[0, 1, 0].tolist() == [3.0, 0.0, float("-inf"), float("-inf")]
    assert attention._mask(4, mx.float32, full_attention=True) is None


def test_vision_parameter_names_match_sidecar_contract():
    model = VisionModel(
        _tiny_config(
            depth=1,
            fullatt_block_indexes=[],
            vit_window_attn_types=[-1],
        )
    )
    parameters = {name: value for name, value in tree_flatten(model.parameters())}

    assert parameters["patch_embed.proj.weight"].shape == (8, 2, 2, 2, 3)
    assert parameters["blocks.0.attn.sinks"].shape == (2,)
    assert parameters["merger.ln_q.weight"].shape == (8,)
    assert "merger.ln_q.bias" not in parameters
    assert "merger.mlp.0.bias" not in parameters
    assert "merger.mlp.2.bias" not in parameters


def test_vision_forward_emits_one_feature_per_merged_patch():
    model = VisionModel(_tiny_config())
    # One image with grid t=2, h=4, w=4 has 32 patch vectors and 8 merged tokens.
    pixel_values = mx.arange(32 * 24, dtype=mx.float32).reshape(32, 24) / 100
    output = model(pixel_values, mx.array([[2, 4, 4]], dtype=mx.int32))
    mx.eval(output)

    assert output.shape == (8, 12)
    assert bool(mx.all(mx.isfinite(output)).item())


def test_sanitize_strips_visual_prefix_and_transposes_conv3d():
    torch_weight = mx.zeros((8, 3, 2, 2, 2))
    sanitized = VisionModel.sanitize(
        {
            "visual.patch_embed.proj.weight": torch_weight,
            "visual.blocks.0.norm1.weight": mx.ones((8,)),
        }
    )

    assert sanitized["patch_embed.proj.weight"].shape == (8, 2, 2, 2, 3)
    assert "blocks.0.norm1.weight" in sanitized


def test_sidecar_loader_composes_text_vision_and_processor(
    tmp_path, monkeypatch
):
    from mlx_lm.models.mimo_v2_flash import Model, ModelArgs

    vision_config = _tiny_config(depth=1, fullatt_block_indexes=[], vit_window_attn_types=[-1])
    sidecar_dir = tmp_path / "omnimodal"
    sidecar_dir.mkdir()
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "mimo_v2_flash", "image_token_id": 99})
    )
    (sidecar_dir / "config.json").write_text(
        json.dumps({"vision_config": vision_config.__dict__})
    )

    source_model = VisionModel(vision_config)
    source_weights = {}
    for name, value in tree_flatten(source_model.parameters()):
        if name == "patch_embed.proj.weight":
            value = value.transpose(0, 4, 1, 2, 3)
        source_weights[f"visual.{name}"] = value
    mx.eval(source_weights)
    mx.save_safetensors(
        str(sidecar_dir / "vision_encoder.safetensors"), source_weights
    )

    text_args = ModelArgs(
        model_type="mimo_v2_flash",
        num_experts_per_tok=1,
        hybrid_layer_pattern=[0, 1],
        moe_layer_freq=[0, 0],
        add_swa_attention_sink_bias=True,
        add_full_attention_sink_bias=False,
        sliding_window_size=8,
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        n_shared_experts=None,
        n_routed_experts=None,
        routed_scaling_factor=None,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        max_position_embeddings=32,
        layernorm_epsilon=1e-6,
        rope_theta=1000.0,
        swa_rope_theta=1000.0,
        swa_num_attention_heads=2,
        swa_num_key_value_heads=1,
        head_dim=8,
        v_head_dim=8,
        swa_head_dim=8,
        swa_v_head_dim=8,
        partial_rotary_factor=0.5,
    )
    text_model = Model(text_args)
    tokenizer = object()
    processor = SimpleNamespace()
    processor_paths = []

    def fake_load_processor(model_path, **_kwargs):
        processor_paths.append(model_path)
        return processor

    monkeypatch.setattr(
        "omlx.utils.model_loading.load_text_model",
        lambda *_args, **_kwargs: (text_model, tokenizer),
    )
    monkeypatch.setattr(
        "mlx_vlm.utils.load_processor",
        fake_load_processor,
    )

    model, loaded_processor = load(tmp_path)

    assert isinstance(model, MiMoOmnimodalModel)
    assert model.config.model_type == "mimo_v2_flash"
    assert loaded_processor is processor
    assert processor_paths == [tmp_path]
    assert processor.tokenizer is tokenizer


def test_language_adapter_translates_inputs_embeds_keyword():
    class Target:
        def __init__(self):
            self.model = SimpleNamespace(layers=[object()])
            self.args = SimpleNamespace()
            self.calls = []

        def __call__(
            self,
            input_ids,
            *,
            cache=None,
            input_embeddings=None,
            return_hidden=False,
        ):
            self.calls.append((input_ids, cache, input_embeddings, return_hidden))
            if return_hidden:
                return input_embeddings, input_embeddings
            return input_embeddings

        def make_cache(self):
            return ["cache"]

    target = Target()
    adapter = MiMoLanguageAdapter(target)
    input_ids = mx.array([[1, 2]])
    embeddings = mx.ones((1, 2, 4))

    result = adapter(
        input_ids,
        cache=["active"],
        inputs_embeds=embeddings,
        return_hidden=True,
        logits_keep=1,
    )

    assert result[0] is embeddings
    assert result[1] is embeddings
    assert len(target.calls) == 1
    called_ids, called_cache, called_embeddings, called_return_hidden = target.calls[0]
    assert called_ids is input_ids
    assert called_cache == ["active"]
    assert called_embeddings is embeddings
    assert called_return_hidden is True
    assert adapter.make_cache() == ["cache"]


def test_language_adapter_injects_embeddings_into_upstream_flash_model():
    from mlx_lm.models.mimo_v2_flash import Model, ModelArgs

    args = ModelArgs(
        model_type="mimo_v2_flash",
        num_experts_per_tok=1,
        hybrid_layer_pattern=[0, 1],
        moe_layer_freq=[0, 0],
        add_swa_attention_sink_bias=True,
        add_full_attention_sink_bias=False,
        sliding_window_size=8,
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        n_shared_experts=None,
        n_routed_experts=None,
        routed_scaling_factor=None,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        max_position_embeddings=32,
        layernorm_epsilon=1e-6,
        rope_theta=1000.0,
        swa_rope_theta=1000.0,
        swa_num_attention_heads=2,
        swa_num_key_value_heads=1,
        head_dim=8,
        v_head_dim=8,
        swa_head_dim=8,
        swa_v_head_dim=8,
        partial_rotary_factor=0.5,
    )
    target = Model(args)
    adapter = MiMoLanguageAdapter(target)
    input_ids = mx.array([[1, 2, 3]])
    embeddings = target.model.embed_tokens(input_ids)

    expected = target(input_ids)
    actual = adapter(input_ids, inputs_embeds=embeddings)
    mx.eval(expected, actual)

    assert bool(mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item())


def test_image_feature_merge_is_batch_ordered_and_strict():
    input_ids = mx.array([[1, 99, 2], [99, 3, 99]])
    text = mx.zeros((2, 3, 2))
    features = mx.array([[10, 11], [20, 21], [30, 31]])

    merged = MiMoOmnimodalModel._merge_image_features(
        input_ids, text, features, image_token_id=99
    )
    mx.eval(merged)

    assert merged[0, 1].tolist() == [10, 11]
    assert merged[1, 0].tolist() == [20, 21]
    assert merged[1, 2].tolist() == [30, 31]

    with pytest.raises(ValueError, match="placeholder count"):
        MiMoOmnimodalModel._merge_image_features(
            input_ids, text, features[:2], image_token_id=99
        )


def test_export_sidecars_keeps_original_media_weights_and_audio_tokenizer(tmp_path):
    from omlx.patches.mimo_v2.omnimodal import export_sidecars

    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    config = _tiny_config(depth=1, fullatt_block_indexes=[], vit_window_attn_types=[-1])
    vision = VisionModel(config)
    weights = {}
    for key, value in tree_flatten(vision.parameters()):
        if key == "patch_embed.proj.weight":
            value = value.transpose(0, 4, 1, 2, 3)
        weights[f"visual.{key}"] = value
    audio_key = "speech_embeddings.0.weight"
    weights[audio_key] = mx.ones((8, 4), dtype=mx.bfloat16)
    weights["model.layers.0.weight"] = mx.zeros((4, 4))
    mx.save_safetensors(str(source / "model.safetensors"), weights)
    audio_root = source / "audio_tokenizer"
    audio_root.mkdir()
    (audio_root / "model.safetensors").write_bytes(b"audio tokenizer")
    (audio_root / "config.json").write_text("{}")

    export_sidecars(source, output, {"vision_config": config.__dict__})

    exported = mx.load(str(output / "omnimodal/vision_encoder.safetensors"))
    restored = VisionModel(config)
    restored.load_weights(list(restored.sanitize(exported).items()), strict=True)
    for key, value in exported.items():
        assert mx.array_equal(value, weights[key]).item()
    audio = mx.load(str(output / "omnimodal/audio_encoder.safetensors"))
    assert list(audio) == [audio_key]
    assert mx.array_equal(audio[audio_key], weights[audio_key]).item()
    assert (
        output / "audio_tokenizer/model.safetensors"
    ).read_bytes() == b"audio tokenizer"
    assert has_vision_sidecar(output)
