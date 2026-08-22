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
