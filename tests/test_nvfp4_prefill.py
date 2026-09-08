# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches.nvfp4_prefill import NVFP4PrefillLinear, apply_nvfp4_prefill


@pytest.mark.parametrize(
    "shape", [(1, 1, 128), (8, 1, 128), (1, 1024, 128), (4, 256, 128)]
)
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_projection_and_decode_fallback(shape, dtype):
    layer = nn.QuantizedLinear(128, 64, bias=True, mode="nvfp4")
    x = mx.random.normal(shape).astype(dtype)
    expected = layer(x)
    mx.eval(expected)
    weights = layer.weight
    layer.__class__ = NVFP4PrefillLinear
    original = mx.dequantize
    with patch.object(mx, "dequantize", wraps=original) as dequantize:
        actual = layer(x)
        mx.eval(actual)
        assert dequantize.call_count == int(
            dtype == mx.bfloat16 and shape[0] * shape[1] >= 1024
        )
    assert layer.weight is weights
    assert mx.allclose(actual, expected, atol=0.02, rtol=0.02).item()


def fake_model():
    # Tiny projections exercise installation without allocating 27B parameters.
    cls = type("TextModel", (), {"__module__": "mlx_lm.models.qwen3_5"})
    model = cls()
    model.args = SimpleNamespace(hidden_size=5120, intermediate_size=17408)
    model.layers = [
        SimpleNamespace(
            mlp=SimpleNamespace(
                **{
                    name: nn.QuantizedLinear(16, 16, mode="nvfp4")
                    for name in ("gate_proj", "up_proj", "down_proj")
                }
            )
        )
        for _ in range(64)
    ]
    return model


def test_opt_in_is_instance_local_and_idempotent(monkeypatch):
    a, b = fake_model(), fake_model()
    monkeypatch.delenv("OMLX_NVFP4_PREFILL", raising=False)
    assert apply_nvfp4_prefill(a) == 0
    monkeypatch.setenv("OMLX_NVFP4_PREFILL", "1")
    original_call = nn.QuantizedLinear.__call__
    tensor = a.layers[0].mlp.gate_proj.weight
    assert apply_nvfp4_prefill(a) == 189
    assert apply_nvfp4_prefill(a) == 0
    assert a.layers[0].mlp.gate_proj.weight is tensor
    assert type(a.layers[-1].mlp.gate_proj) is nn.QuantizedLinear
    assert type(b.layers[0].mlp.gate_proj) is nn.QuantizedLinear
    assert nn.QuantizedLinear.__call__ is original_call


def test_unsupported_models_are_untouched(monkeypatch):
    monkeypatch.setenv("OMLX_NVFP4_PREFILL", "1")
    model = fake_model()
    model.args.hidden_size = 1024
    assert apply_nvfp4_prefill(model) == 0
    model.args.hidden_size = 5120
    model.layers[1].mlp.up_proj = nn.Linear(16, 16)
    assert apply_nvfp4_prefill(model) == 0
    assert type(model.layers[0].mlp.gate_proj) is nn.QuantizedLinear
