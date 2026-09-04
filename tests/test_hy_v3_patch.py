# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Hy3 checkpoint compatibility."""

from __future__ import annotations

from copy import deepcopy

import mlx_lm.utils as mlx_lm_utils
import pytest

from omlx.utils import model_loading
from omlx.utils.model_loading import normalize_hy_v3_rope_config


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "hy_v3", "rope_theta": 11158840.0},
        {
            "model_type": "hy_v3",
            "rope_theta": 11158840.0,
            "rope_parameters": None,
        },
    ],
)
def test_normalize_hy_v3_rope_config_fills_legacy_layout(config):
    result = normalize_hy_v3_rope_config(config)

    assert result is config
    assert config["rope_theta"] == 11158840.0
    assert config["rope_parameters"] == {
        "rope_theta": 11158840.0,
        "rope_type": "default",
    }


def test_normalize_hy_v3_rope_config_preserves_structured_layout():
    rope_parameters = {
        "rope_theta": 500000.0,
        "rope_type": "yarn",
        "factor": 4.0,
    }
    config = {
        "model_type": "hy_v3",
        "rope_theta": 11158840.0,
        "rope_parameters": rope_parameters,
    }

    normalize_hy_v3_rope_config(config)

    assert config["rope_parameters"] is rope_parameters


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "llama", "rope_theta": 11158840.0},
        {"model_type": "hy_v3"},
        {"model_type": "hy_v3", "rope_theta": None},
        {
            "model_type": "hy_v3",
            "rope_theta": 11158840.0,
            "rope_parameters": "invalid",
        },
    ],
)
def test_normalize_hy_v3_rope_config_does_not_invent_or_repair_values(config):
    original = deepcopy(config)

    normalize_hy_v3_rope_config(config)

    assert config == original


def test_mlx_lm_load_config_patch_applies_hy_v3_normalization(monkeypatch):
    monkeypatch.setattr(
        mlx_lm_utils,
        "load_config",
        lambda _model_path: {
            "model_type": "hy_v3",
            "rope_theta": 11158840.0,
        },
    )
    monkeypatch.setattr(model_loading, "_MLX_LM_LOAD_CONFIG_PATCHED", False)

    model_loading._patch_mlx_lm_load_config()
    config = mlx_lm_utils.load_config("unused")

    assert config["rope_parameters"] == {
        "rope_theta": 11158840.0,
        "rope_type": "default",
    }


def test_oq_sanitizer_normalizes_legacy_hy_v3_config():
    from omlx.oq import _build_model_sanitizer

    config = {
        "architectures": ["HYV3ForCausalLM"],
        "model_type": "hy_v3",
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "num_shared_experts": 1,
        "expert_hidden_dim": 32,
        "first_k_dense_replace": 1,
        "rms_norm_eps": 1e-5,
        "rope_theta": 11158840.0,
    }

    sanitizer = _build_model_sanitizer(config)

    assert callable(sanitizer)
    assert config["rope_parameters"] == {
        "rope_theta": 11158840.0,
        "rope_type": "default",
    }
