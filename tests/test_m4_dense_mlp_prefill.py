# SPDX-License-Identifier: Apache-2.0
"""Loaded-instance routing and numerical checks for M4 dense MLP prefill."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.llama import MLP

from omlx.patches import m4_dense_mlp_prefill as patch


@pytest.fixture
def enabled(monkeypatch):
    from omlx.patches import qwen35_q4_mlp

    monkeypatch.setenv("OMLX_M4_DENSE_MLP_PREFILL", "1")
    monkeypatch.setattr(patch, "_on_m4_max", lambda: True)
    monkeypatch.setattr(qwen35_q4_mlp, "_qmm_supports_group_size", lambda gs: gs == 64)
    monkeypatch.setattr(qwen35_q4_mlp, "_native_qmm_for_bits", lambda bits: object())


def _linear(input_dim=3840, output_dim=15360, bits=4, dtype=mx.bfloat16):
    linear = nn.QuantizedLinear(
        input_dim, output_dim, bias=False, bits=bits, group_size=64
    )
    linear.set_dtype(dtype)
    return linear


def _model(module):
    return SimpleNamespace(named_modules=lambda: [("model.layers.0.mlp", module)])


def _args(hidden=3840, intermediate=15360):
    return SimpleNamespace(
        hidden_size=hidden, intermediate_size=intermediate, mlp_bias=False
    )


def _mlp():
    module = MLP(_args())
    module.gate_proj = _linear()
    module.up_proj = _linear(bits=8)
    module.down_proj = _linear(15360, 3840)
    return module


def test_disabled_does_not_probe_hardware(monkeypatch):
    monkeypatch.delenv("OMLX_M4_DENSE_MLP_PREFILL", raising=False)
    probe = MagicMock(side_effect=AssertionError("disabled must not probe"))
    monkeypatch.setattr(patch, "_on_m4_max", probe)
    assert patch.apply_m4_dense_mlp_prefill(None) == 0
    probe.assert_not_called()


@pytest.mark.parametrize("name", ["Apple M3 Max", "Apple M4 Pro", "Apple M5 Max", None])
def test_other_hardware_is_unchanged(monkeypatch, name):
    monkeypatch.setenv("OMLX_M4_DENSE_MLP_PREFILL", "1")
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "device_info", lambda: {"device_name": name})
    assert patch.apply_m4_dense_mlp_prefill(None) == 0


def test_rebinds_only_mlp_instances_and_keeps_weights(enabled, monkeypatch):
    monkeypatch.setattr(patch, "_native_qmm_for_bits", lambda bits: object())
    first = _mlp()
    other = _mlp()
    weight, scales, biases = (
        first.gate_proj.weight,
        first.gate_proj.scales,
        first.gate_proj.biases,
    )
    original_call = type(first).__call__
    assert patch.apply_m4_dense_mlp_prefill(_model(first)) == 3
    assert patch.apply_m4_dense_mlp_prefill(_model(first)) == 0
    assert first.gate_proj.weight is weight
    assert first.gate_proj.scales is scales
    assert first.gate_proj.biases is biases
    assert type(other.gate_proj) is nn.QuantizedLinear
    assert type(first).__call__ is original_call


def test_missing_native_symbols_leaves_stock(enabled, monkeypatch):
    from omlx.patches import qwen35_q4_mlp

    monkeypatch.setattr(patch, "_native_qmm_for_bits", lambda bits: None)
    monkeypatch.setattr(qwen35_q4_mlp, "_native_qmm_for_bits", lambda bits: None)
    module = _mlp()
    assert patch.apply_m4_dense_mlp_prefill(_model(module)) == 0
    assert type(module.up_proj) is nn.QuantizedLinear


def test_custom_mlp_and_linear_subclasses_are_untouched(enabled, monkeypatch):
    monkeypatch.setattr(patch, "_native_qmm_for_bits", lambda bits: object())

    class CustomMLP(MLP):
        pass

    assert patch.apply_m4_dense_mlp_prefill(_model(CustomMLP(_args(64, 128)))) == 0
    module = _mlp()

    class CustomLinear(nn.QuantizedLinear):
        pass

    module.gate_proj.__class__ = CustomLinear
    assert patch.apply_m4_dense_mlp_prefill(_model(module)) == 2
    assert type(module.gate_proj) is CustomLinear


@pytest.mark.parametrize("rows,batch", [(1, 1), (8, 1), (511, 1), (512, 2)])
def test_decode_verify_short_and_multirow_fall_back(enabled, monkeypatch, rows, batch):
    monkeypatch.setattr(patch, "_native_qmm_for_bits", lambda bits: object())
    module = _mlp()
    patch.apply_m4_dense_mlp_prefill(_model(module))
    native = MagicMock(side_effect=AssertionError("must keep stock"))
    monkeypatch.setattr(patch, "_native_qmm_for_bits", native)
    x = mx.zeros((batch, rows, 3840), dtype=mx.bfloat16)
    got = module.gate_proj(x)
    ref = nn.QuantizedLinear.__call__(module.gate_proj, x)
    mx.eval(got, ref)
    assert mx.array_equal(got, ref).item()
    native.assert_not_called()


def test_exact_threshold_routes_using_owning_stream(enabled, monkeypatch):
    calls = []

    def native(x, weight, scales, biases, variant, group_size):
        calls.append((variant, group_size, mx.default_stream(mx.gpu)))
        return mx.quantized_matmul(
            x, weight, scales, biases, bits=4, group_size=64, transpose=True
        )

    monkeypatch.setattr(patch, "_native_qmm_for_bits", lambda bits: native)
    module = _mlp()
    patch.apply_m4_dense_mlp_prefill(_model(module))
    stream = mx.new_stream(mx.gpu)
    with mx.stream(stream):
        module.gate_proj(mx.zeros((1, 512, 3840), dtype=mx.bfloat16))
    assert calls == [(8, 64, stream)]


@pytest.mark.parametrize("shape", [(64, 128), (3840, 15296), (4096, 16384)])
def test_unmeasured_shapes_stay_stock(enabled, monkeypatch, shape):
    monkeypatch.setattr(patch, "_native_qmm_for_bits", lambda bits: object())
    module = MLP(_args(*shape))
    module.gate_proj = _linear(*shape)
    assert patch.apply_m4_dense_mlp_prefill(_model(module)) == 0


@pytest.mark.parametrize("change", ["dtype", "bits", "group", "mode", "bias"])
def test_nonstandard_quantization_is_not_wrapped(enabled, monkeypatch, change):
    monkeypatch.setattr(patch, "_native_qmm_for_bits", lambda bits: object())
    module = _mlp()
    linear = module.gate_proj
    if change == "dtype":
        linear.scales = linear.scales.astype(mx.float32)
    elif change == "bits":
        linear.bits = 6
    elif change == "group":
        linear.group_size = 128
    elif change == "mode":
        linear.mode = "mxfp4"
    else:
        linear.bias = mx.zeros((15360,), dtype=mx.bfloat16)
    assert patch.apply_m4_dense_mlp_prefill(_model(module)) == 2
    assert type(module.gate_proj) is nn.QuantizedLinear


def test_dtype_change_after_load_falls_back(enabled, monkeypatch):
    monkeypatch.setattr(patch, "_native_qmm_for_bits", lambda bits: object())
    module = _mlp()
    patch.apply_m4_dense_mlp_prefill(_model(module))
    native = MagicMock(side_effect=AssertionError("mixed dtype must keep stock"))
    monkeypatch.setattr(patch, "_native_qmm_for_bits", native)
    x = mx.zeros((1, 512, 3840), dtype=mx.float16)
    got = module.gate_proj(x)
    ref = nn.QuantizedLinear.__call__(module.gate_proj, x)
    mx.eval(got, ref)
    assert mx.array_equal(got, ref).item()
    native.assert_not_called()


def test_post_load_hook_runs_without_model_settings(enabled, monkeypatch):
    from omlx.utils.model_loading import apply_post_load_transforms

    monkeypatch.setattr(patch, "_native_qmm_for_bits", lambda bits: object())
    model = _model(_mlp())
    assert apply_post_load_transforms(model, None) is model
    assert type(model.named_modules()[0][1].gate_proj) is patch._M4PrefillLinear


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_native_output_matches_stock_on_measured_shape(enabled, bits, dtype):
    if not mx.metal.is_available() or patch._native_qmm_for_bits(bits) is None:
        pytest.skip("native QMM is not built")
    linear = _linear(bits=bits, dtype=dtype)
    x = mx.random.normal((1, 513, 3840)).astype(dtype)
    reference = linear(x)
    linear.__class__ = patch._M4PrefillLinear
    got = linear(x)
    mx.eval(reference, got)
    assert mx.array_equal(got, reference).item()


@pytest.mark.parametrize("evaluate_slice", [False, True])
def test_strided_input_matches_stock_on_native_path(
    enabled, monkeypatch, evaluate_slice
):
    if not mx.metal.is_available() or patch._native_qmm_for_bits(4) is None:
        pytest.skip("native QMM is not built")
    linear = _linear()
    x = mx.random.normal((1, 1024, 3840)).astype(mx.bfloat16)[:, ::2, :]
    if evaluate_slice:
        mx.eval(x)
    reference = linear(x)
    linear.__class__ = patch._M4PrefillLinear
    native = patch._native_qmm_for_bits(4)
    calls = []

    def spy(*args):
        calls.append(True)
        return native(*args)

    monkeypatch.setattr(patch, "_native_qmm_for_bits", lambda bits: spy)
    got = linear(x)
    mx.eval(reference, got)
    assert calls == [True]
    assert mx.array_equal(got, reference).item()


def test_strided_quantized_weights_match_stock(enabled):
    if not mx.metal.is_available() or patch._native_qmm_for_bits(4) is None:
        pytest.skip("native QMM is not built")
    linear = _linear(output_dim=30720)
    linear.weight = linear.weight[::2]
    linear.scales = linear.scales[::2]
    linear.biases = linear.biases[::2]
    x = mx.random.normal((1, 512, 3840)).astype(mx.bfloat16)
    reference = linear(x)
    linear.__class__ = patch._M4PrefillLinear
    # The row slice returns the measured 15360 output width, so this must
    # normalize the lazy weight layout before taking the native route.
    got = linear(x)
    mx.eval(got, reference)
    assert mx.array_equal(got, reference).item()
