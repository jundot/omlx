# SPDX-License-Identifier: Apache-2.0
"""Tencent WeMM-Video (wemm_qwen3_5_moe & wemm_qwen3_5) mlx-vlm compatibility tests."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


@pytest.fixture(scope="module")
def applied():
    from omlx.patches.mlx_vlm_wemm_compat import (
        apply_mlx_vlm_wemm_compat_patch,
        is_applied,
    )

    apply_mlx_vlm_wemm_compat_patch()
    assert is_applied()
    return True


def _tiny_wemm_moe_config():
    from mlx_vlm.models.wemm_qwen3_5_moe.config import (
        AudioConfig,
        ModelConfig,
        QFormerConfig,
        TextConfig,
        VisionConfig,
    )

    return ModelConfig(
        model_type="wemm_qwen3_5_moe",
        image_token_id=248056,
        video_token_id=248057,
        audio_token_id=248076,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
        text_config=TextConfig(
            model_type="qwen3_5_moe_text",
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
            num_experts=4,
            num_experts_per_tok=2,
            shared_expert_intermediate_size=32,
            moe_intermediate_size=32,
            rms_norm_eps=1e-6,
            vocab_size=248320,
            max_position_embeddings=4096,
            full_attention_interval=4,
            rope_parameters={
                "type": "default",
                "mrope_section": [1, 1, 2],
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
            },
        ),
        vision_config=VisionConfig(
            model_type="qwen3_5_moe",
            depth=1,
            hidden_size=64,
            intermediate_size=128,
            num_heads=4,
            in_channels=3,
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            out_hidden_size=64,
            num_position_embeddings=64,
            deepstack_visual_indexes=[],
        ),
        audio_config=AudioConfig(
            d_model=32,
            encoder_layers=2,
            encoder_attention_heads=4,
            encoder_ffn_dim=64,
            num_mel_bins=128,
            max_source_positions=1500,
            second_per_window=0.1,
            second_stride=0.1,
            qformer_config=QFormerConfig(
                hidden_size=48,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=96,
                encoder_width=32,
                query_length=1,
                out_hidden_size=64,
            ),
        ),
    )


def test_vendor_modules_resolve(applied):
    import mlx_vlm.prompt_utils as prompt_utils
    import mlx_vlm.utils as vlm_utils

    for mt in ("wemm_qwen3_5_moe", "wemm_qwen3_5"):
        module = importlib.import_module(f"mlx_vlm.models.{mt}")
        assert hasattr(module, "Model")
        assert hasattr(module, "LanguageModel")
        assert hasattr(module, "WeMMProcessor")
        res = vlm_utils.get_model_and_args({"model_type": mt})
        assert res[0] is module
        assert mt in prompt_utils.MODEL_CONFIG


def test_double_apply_is_noop(applied):
    from omlx.patches.mlx_vlm_wemm_compat import (
        apply_mlx_vlm_wemm_compat_patch,
        is_applied,
    )

    assert apply_mlx_vlm_wemm_compat_patch() is False
    assert is_applied() is True


def test_dense_wemm_config_detection(applied):
    from omlx.patches.mlx_vlm_wemm_compat import (
        is_wemm_config,
        normalize_wemm_model_type,
    )

    dense_cfg = {
        "architectures": ["WeMMVideoForConditionalGeneration"],
        "model_type": "qwen3_5",
        "audio_config": {"model_type": "whisper", "qformer_config": {"model_type": "bert"}},
        "text_config": {"model_type": "qwen3_5_text", "hidden_size": 2048},
    }
    assert is_wemm_config(dense_cfg) is True
    assert normalize_wemm_model_type(dense_cfg) == "wemm_qwen3_5"

    plain_q35 = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5"}
    assert is_wemm_config(plain_q35) is False


def test_audio_tower_and_mrope_interleave(applied):
    from mlx_vlm.models.wemm_qwen3_5_moe import Model, WeMMAudioTower

    cfg = _tiny_wemm_moe_config()
    tower = WeMMAudioTower(cfg.audio_config)
    mel = mx.zeros((1, 128, 3000), dtype=mx.float32)
    out = tower(mel)
    mx.eval(out)
    assert out.shape == (1, 300, 64)

    model = Model(cfg)
    vs, ve, vp, ap = 248053, 248054, 248057, 248076
    tokens = [10, 11, vs, vp, vp, vp, vp, ap, ap, ap, ve, 12]
    input_ids = mx.array([tokens], dtype=mx.int32)
    video_grid_thw = mx.array([[1, 4, 4]], dtype=mx.int32)
    pos_ids, _ = model.language_model.get_rope_index(
        input_ids, video_grid_thw=video_grid_thw
    )
    mx.eval(pos_ids)
    assert pos_ids.shape == (3, 1, len(tokens))
    # Visual tokens get 2D spatial grid positions; interleaved audio tokens get 1D sequential positions
    assert pos_ids[1, 0, 3:7].tolist() == [3, 3, 4, 4]
    assert pos_ids[2, 0, 3:7].tolist() == [3, 4, 3, 4]
    assert pos_ids[0, 0, 7:10].tolist() == [5, 6, 7]


def test_sanitize_per_expert_and_qformer_keys(applied):
    from mlx_vlm.models.wemm_qwen3_5_moe import Model

    cfg = _tiny_wemm_moe_config()
    model = Model(cfg)
    raw = {
        "model.audio.conv1.weight": mx.zeros((32, 128, 3)),
        "model.audio.qformer.bert.embeddings.LayerNorm.gamma": mx.ones((48,)),
        "model.audio.qformer.bert.encoder.layer.0.crossattention.output.LayerNorm.beta": mx.zeros((48,)),
        "model.language_model.layers.0.input_layernorm.weight": mx.zeros((64,)),
        "model.language_model.layers.0.linear_attn.conv1d.weight": mx.zeros((128, 1, 4)),
    }
    for l in range(cfg.text_config.num_hidden_layers):
        for e in range(cfg.text_config.num_experts):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                raw[f"model.language_model.layers.{l}.mlp.experts.{e}.{proj}.weight"] = mx.ones((32, 64))

    san = model.sanitize(raw)
    assert san["audio_tower.conv1.weight"].shape == (32, 3, 128)
    assert "audio_tower.qformer_embeddings_ln.weight" in san
    assert "audio_tower.qformer_layers.0.crossattention.LayerNorm.bias" in san
    assert san["language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"].shape == (4, 32, 64)
    assert np.allclose(np.array(san["language_model.model.layers.0.input_layernorm.weight"]), 1.0)
