import json
import struct
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest

from omlx.quantization.exl3 import install_packed_experts, validate_checkpoint_headers
from omlx.utils.model_loading import (
    expand_per_layer_quant_keys,
    maybe_load_custom_quantization,
)
from tests.test_exl3_loading import fixture


def add_head(tmp_path, config, prefix="language_model.mtp."):
    shard = tmp_path / "model.safetensors"
    with shard.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(size))
    head = {}
    offset = 0
    for key, entry in header.items():
        entry = dict(entry)
        count = entry["data_offsets"][1] - entry["data_offsets"][0]
        entry["data_offsets"] = [offset, offset + count]
        head[prefix + key.removeprefix("language_model.model.")] = entry
        offset += count
    raw = json.dumps(head).encode()
    (tmp_path / "mtp.safetensors").write_bytes(
        struct.pack("<Q", len(raw)) + raw + bytes(offset)
    )


@pytest.mark.parametrize(
    "prefix", ["language_model.mtp.", "model.language_model.mtp.", "model.mtp.", "mtp."]
)
def test_head_prefix_and_missing_validation(tmp_path, prefix):
    config = fixture(tmp_path)
    with pytest.raises(ValueError, match="Missing EXL3"):
        validate_checkpoint_headers(tmp_path, config, include_mtp=True)
    add_head(tmp_path, config, prefix)
    validate_checkpoint_headers(tmp_path, config, include_mtp=True)
    config["text_config"]["mtp_num_experts"] = 3
    with pytest.raises(ValueError, match="Invalid EXL3"):
        validate_checkpoint_headers(tmp_path, config, include_mtp=True)


def test_mtp_opt_in_respects_dispatch_and_does_not_leak(tmp_path, monkeypatch):
    config = fixture(tmp_path)
    add_head(tmp_path, config)
    monkeypatch.setenv("OMLX_EXL3_ENABLED", "1")
    with (
        patch("omlx.patches.mlx_lm_mtp.is_mtp_active", return_value=True),
        patch(
            "omlx.patches.mlx_vlm_qwen4_exp_compat.configure_qwen4_exp_runtime"
        ) as configure,
        patch("mlx_vlm.utils.load", return_value=("model", "processor")),
    ):
        maybe_load_custom_quantization(
            str(tmp_path), is_vlm=True, model_settings=SimpleNamespace(mtp_enabled=True)
        )
        configure.assert_called_with(str(tmp_path), mode="mmap", mtp_enabled=True)
        maybe_load_custom_quantization(str(tmp_path), is_vlm=True)
        configure.assert_called_with(str(tmp_path), mode="mmap", mtp_enabled=False)


def test_mtp_quantization_keys_follow_sanitized_module_tree():
    q = {
        "bits": 8,
        "group_size": 64,
        "language_model.mtp.layers.0.self_attn.q_proj": {"bits": 8, "group_size": 128},
    }
    c = {"quantization": q}
    expand_per_layer_quant_keys(c)
    assert q["mtp.layers.0.self_attn.q_proj"] == {"bits": 8, "group_size": 128}


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_installs_packed_head_with_its_own_expert_count(monkeypatch):
    monkeypatch.setenv("OMLX_EXL3_ENABLED", "1")

    def layer(e):
        projections = SimpleNamespace(
            **{
                name: SimpleNamespace(weight=mx.zeros((e, 128, 128)))
                for name in ("gate_proj", "up_proj", "down_proj")
            }
        )
        return SimpleNamespace(mlp=SimpleNamespace(switch_mlp=projections))

    main, head = layer(2), layer(3)
    model = SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(layers=[main])),
        mtp=SimpleNamespace(layers=[head]),
    )
    install_packed_experts(
        model,
        SimpleNamespace(
            expert_quant={
                "format": "exl3",
                "codebook": "mcg",
                "out_scales": "svh",
                "k": 2.625,
                "window": 15,
            }
        ),
    )
    assert main.mlp.switch_mlp.down_proj.num_experts == 2
    assert head.mlp.switch_mlp.down_proj.num_experts == 3


def test_head_duplicate_and_unsupported_depth_rejected_before_loading(tmp_path):
    config = fixture(tmp_path)
    add_head(tmp_path, config)
    config["text_config"]["mtp_num_hidden_layers"] = 2
    with pytest.raises(ValueError, match="one-layer draft head"):
        validate_checkpoint_headers(tmp_path, config, include_mtp=True)
    config["text_config"]["mtp_num_hidden_layers"] = 1
    (tmp_path / "duplicate.safetensors").write_bytes(
        (tmp_path / "mtp.safetensors").read_bytes()
    )
    with pytest.raises(ValueError, match="Duplicate packed tensor"):
        validate_checkpoint_headers(tmp_path, config, include_mtp=True)
