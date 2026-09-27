import json
import struct
from unittest.mock import patch

import pytest

from omlx.quantization.exl3 import validate_checkpoint_headers
from omlx.utils.model_loading import maybe_load_custom_quantization


def fixture(tmp_path):
    config = {
        "model_type": "qwen4_exp",
        "expert_quant": {
            "format": "exl3",
            "codebook": "mcg",
            "out_scales": "svh",
            "k": 2.625,
            "window": 15,
        },
        "text_config": {
            "num_hidden_layers": 1,
            "hidden_size": 128,
            "moe_intermediate_size": 128,
            "num_experts": 2,
        },
    }
    header = {}
    offset = 0
    for projection in ("gate_proj", "up_proj", "down_proj"):
        for part, shape, dtype in [
            ("trellis", [2, 8, 8, 42], "U16"),
            ("suh", [2, 128], "F16"),
            ("svh", [2, 128], "F16"),
        ]:
            count = 1
            for n in shape:
                count *= n
            key = f"language_model.model.layers.0.mlp.switch_mlp.{projection}.{part}"
            header[key] = {
                "shape": shape,
                "dtype": dtype,
                "data_offsets": [offset, offset + count * 2],
            }
            offset += count * 2
    raw = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(
        struct.pack("<Q", len(raw)) + raw + bytes(offset)
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    return config


def test_header_validation_and_missing_duplicate(tmp_path):
    config = fixture(tmp_path)
    validate_checkpoint_headers(tmp_path, config)
    duplicate = tmp_path / "duplicate.safetensors"
    duplicate.write_bytes((tmp_path / "model.safetensors").read_bytes())
    with pytest.raises(ValueError, match="Duplicate"):
        validate_checkpoint_headers(tmp_path, config)
    duplicate.unlink()
    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="Missing"):
        validate_checkpoint_headers(tmp_path, config)


def test_opt_in_before_any_load(tmp_path, monkeypatch):
    fixture(tmp_path)
    monkeypatch.delenv("OMLX_EXL3_ENABLED", raising=False)
    with patch("mlx_vlm.utils.load") as load:
        with pytest.raises(ValueError, match="opt-in"):
            maybe_load_custom_quantization(str(tmp_path), is_vlm=True)
        load.assert_not_called()


def test_native_dispatch_strict_no_checkpoint_code(tmp_path, monkeypatch):
    fixture(tmp_path)
    monkeypatch.setenv("OMLX_EXL3_ENABLED", "1")
    with (
        patch(
            "omlx.patches.mlx_vlm_qwen4_exp_compat.configure_qwen4_exp_runtime"
        ) as configure,
        patch("mlx_vlm.utils.load", return_value=("model", "processor")) as load,
    ):
        assert maybe_load_custom_quantization(str(tmp_path), is_vlm=True) == (
            "model",
            "processor",
        )
        configure.assert_called_once_with(str(tmp_path), mode="mmap", mtp_enabled=False)
        load.assert_called_once_with(
            str(tmp_path), trust_remote_code=False, strict=True
        )


def test_text_fallback_refused(tmp_path, monkeypatch):
    fixture(tmp_path)
    monkeypatch.setenv("OMLX_EXL3_ENABLED", "1")
    with pytest.raises(ValueError, match="native Qwen4"):
        maybe_load_custom_quantization(str(tmp_path), is_vlm=False)
