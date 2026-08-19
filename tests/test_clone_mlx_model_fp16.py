# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json

import mlx.core as mx
import pytest
from safetensors import safe_open

from tools.clone_mlx_model_fp16 import clone_model


def _source_model(tmp_path, values: list[float]):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "dtype": "bfloat16"})
    )
    mx.save_safetensors(
        str(source / "model.safetensors"),
        {
            "model.layers.0.weight": mx.array(values, dtype=mx.bfloat16),
            "model.layers.0.packed": mx.array([1, 2], dtype=mx.uint32),
        },
    )
    return source


def test_clone_converts_valid_bfloat16_and_preserves_packed_weights(tmp_path):
    source = _source_model(tmp_path, [1.0, -2.0])
    destination = tmp_path / "clone"

    clone_model(source, destination)

    config = json.loads((destination / "config.json").read_text())
    assert config["dtype"] == "float16"
    with safe_open(destination / "model.safetensors", framework="np") as handle:
        assert handle.get_tensor("model.layers.0.weight").dtype.name == "float16"
        assert handle.get_tensor("model.layers.0.packed").dtype.name == "uint32"


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (70000.0, "exceeds the FP16 limit"),
        (float("nan"), "NaN or infinite"),
        (float("inf"), "NaN or infinite"),
    ],
)
def test_clone_rejects_unsafe_values_before_writing(tmp_path, value, message):
    source = _source_model(tmp_path, [value])
    destination = tmp_path / "clone"

    with pytest.raises(ValueError, match=message):
        clone_model(source, destination)

    assert not destination.exists()
