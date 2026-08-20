# SPDX-License-Identifier: Apache-2.0
"""Tests for the Qwen3.8 mixed ModelOpt DFlash target bridge."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from omlx.patches import qwen38_modelopt_dflash as dflash_bridge
from omlx.patches import qwen38_modelopt_mixed as mixed_bridge


def _config() -> dict:
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "hidden_size": 5120,
            "num_hidden_layers": 64,
        },
        "vision_config": {
            "model_type": "qwen3_5_vision",
            "hidden_size": 1152,
            "out_hidden_size": 5120,
        },
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "mixed-precision",
            "config_groups": {
                "group_0": {
                    "format": "float-quantized",
                    "targets": list(mixed_bridge._FP8_TARGETS),
                    "weights": {
                        "type": "float",
                        "num_bits": 8,
                        "strategy": "channel",
                        "group_size": None,
                        "dynamic": False,
                        "symmetric": True,
                    },
                },
                "group_1": {
                    "format": "nvfp4-pack-quantized",
                    "targets": list(mixed_bridge._NVFP4_TARGETS),
                    "weights": {
                        "type": "float",
                        "num_bits": 4,
                        "strategy": "tensor_group",
                        "group_size": 16,
                        "dynamic": False,
                        "symmetric": True,
                    },
                },
            },
        },
    }


def _write_config(path, config=None):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config or _config()))


def test_supported_model_ref_is_local_and_strict(tmp_path):
    target = tmp_path / "target"
    _write_config(target)
    assert dflash_bridge.is_supported_model_ref(target)

    unsupported = tmp_path / "other"
    config = _config()
    config["text_config"]["num_hidden_layers"] = 63
    _write_config(unsupported, config)
    assert not dflash_bridge.is_supported_model_ref(unsupported)
    assert not dflash_bridge.is_supported_model_ref("org/not-a-local-checkpoint")


def test_install_routes_only_supported_target(tmp_path, monkeypatch):
    from dflash_mlx.runtime import loading as dflash_loading

    target = tmp_path / "target"
    _write_config(target)

    original_load = MagicMock(return_value=("ORIGINAL", "TOKENIZER", {"kind": "orig"}))
    exact_load = MagicMock(return_value=("EXACT", "TOKENIZER", {"kind": "exact"}))
    monkeypatch.setattr(dflash_loading, "load", original_load)
    monkeypatch.setattr(dflash_bridge, "load_lm", exact_load)

    assert dflash_bridge.install_dflash_modelopt_loader()
    result = dflash_loading.load(str(target), lazy=True, return_config=True)
    assert result[0] == "EXACT"
    exact_load.assert_called_once_with(str(target), lazy=True, return_config=True)
    original_load.assert_not_called()

    result = dflash_loading.load("org/ordinary-model", lazy=True, return_config=True)
    assert result[0] == "ORIGINAL"
    original_load.assert_called_once_with(
        "org/ordinary-model", lazy=True, return_config=True
    )


def test_install_is_idempotent(tmp_path, monkeypatch):
    from dflash_mlx.runtime import loading as dflash_loading

    target = tmp_path / "target"
    _write_config(target)

    original_load = MagicMock(return_value=("ORIGINAL", "TOKENIZER", {}))
    exact_load = MagicMock(return_value=("EXACT", "TOKENIZER", {}))
    monkeypatch.setattr(dflash_loading, "load", original_load)
    monkeypatch.setattr(dflash_bridge, "load_lm", exact_load)

    assert dflash_bridge.install_dflash_modelopt_loader()
    first_wrapper = dflash_loading.load
    assert not dflash_bridge.install_dflash_modelopt_loader()
    assert dflash_loading.load is first_wrapper

    dflash_loading.load(str(target), lazy=True, return_config=True)
    exact_load.assert_called_once()


def test_load_lm_suppresses_generic_quantization_but_reports_source_config(
    tmp_path, monkeypatch
):
    import mlx_lm.utils as lm_utils
    from mlx_lm.models import qwen3_5

    target = tmp_path / "target"
    _write_config(target)

    load_model = MagicMock(return_value=("MODEL", {"quantization_config": None}))
    load_tokenizer = MagicMock(return_value="TOKENIZER")
    monkeypatch.setattr(lm_utils, "load_model", load_model)
    monkeypatch.setattr(lm_utils, "load_tokenizer", load_tokenizer)

    model, tokenizer, config = dflash_bridge.load_lm(
        target, lazy=True, return_config=True
    )

    assert model == "MODEL"
    assert tokenizer == "TOKENIZER"
    assert config["quantization_config"]["format"] == "mixed-precision"

    kwargs = load_model.call_args.kwargs
    assert kwargs["lazy"] is True
    assert kwargs["model_config"]["quantization"] is None
    assert kwargs["model_config"]["quantization_config"] is None
    assert kwargs["model_config"]["quantize_activations"] is False

    model_class, args_class = kwargs["get_model_classes"](_config())
    assert issubclass(model_class, qwen3_5.Model)
    assert args_class is qwen3_5.ModelArgs


def test_lifecycle_install_wires_modelopt_dispatch(monkeypatch):
    from omlx.patches import dflash_lifecycle

    install = MagicMock(return_value=True)
    monkeypatch.setattr(dflash_bridge, "install_dflash_modelopt_loader", install)

    dflash_lifecycle.install_dflash_lifecycle_wrap()
    install.assert_called_once_with()
