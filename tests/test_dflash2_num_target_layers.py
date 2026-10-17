# SPDX-License-Identifier: Apache-2.0
"""DFlash2 checkpoints converted without ``num_target_layers`` still load."""

from __future__ import annotations

import types

import pytest

dflash2_config = pytest.importorskip("mlx_vlm.speculative.drafters.dflash2.config")

from omlx.speculative import dflash as dflash_drafter  # noqa: E402

# config.json of mlx-community/Qwen3.6-35B-A3B-DFlash2-4bit (no num_target_layers)
_CONVERTED = {
    "architectures": ["DFlash2DraftModel"],
    "model_type": "qwen3",
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "num_hidden_layers": 6,
    "vocab_size": 248320,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-06,
    "max_position_embeddings": 262144,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000000.0},
    "block_size": 8,
    "is_causal": False,
    "dflash_config": {
        "mask_token_id": 248077,
        "target_layer_ids": [1, 6, 11, 16, 22, 27, 32, 37],
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "selector_rank": 256,
        "selector_top_k": 16,
    },
}


def _target(layers: int):
    inner = types.SimpleNamespace(layers=[object()] * layers)
    return types.SimpleNamespace(language_model=types.SimpleNamespace(model=inner))


def _build(params):
    config = dflash2_config.DFlash2Config.from_dict(dict(params))
    config.validate()
    return config


def test_missing_field_is_rejected_without_the_fallback():
    with pytest.raises((ValueError, TypeError)):
        _build(_CONVERTED)


def test_target_layer_count_fills_the_gap():
    count = dflash_drafter._target_layer_count(_target(40))
    assert count == 40
    with dflash_drafter._default_num_target_layers(count):
        config = _build(_CONVERTED)
    assert config.num_target_layers == 40
    # restored afterwards
    with pytest.raises((ValueError, TypeError)):
        _build(_CONVERTED)


def test_explicit_value_wins():
    params = {**_CONVERTED, "num_target_layers": 48}
    with dflash_drafter._default_num_target_layers(40):
        config = _build(params)
    assert config.num_target_layers == 48
