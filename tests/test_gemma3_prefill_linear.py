# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_vlm.models.gemma3 import language

from omlx.patches import gemma3_prefill_linear as patch


def _module(cls, **attributes):
    module = cls.__new__(cls)
    nn.Module.__init__(module)
    for name, value in attributes.items():
        setattr(module, name, value)
    return module


def _linear(input_dim=3840, output_dim=2048, cls=nn.QuantizedLinear):
    return _module(
        cls,
        weight=mx.broadcast_to(
            mx.array(0xFEDCBA98, mx.uint32), (output_dim, input_dim // 8)
        ),
        scales=mx.full((output_dim, input_dim // 64), 0.03125, mx.bfloat16),
        biases=mx.full((output_dim, input_dim // 64), -0.25, mx.bfloat16),
        group_size=64,
        bits=4,
        mode="affine",
    )


_PROJECTIONS = (
    ("self_attn", "q_proj", 3840, 4096),
    ("self_attn", "k_proj", 3840, 2048),
    ("self_attn", "v_proj", 3840, 2048),
    ("self_attn", "o_proj", 4096, 3840),
    ("mlp", "gate_proj", 3840, 15360),
    ("mlp", "up_proj", 3840, 15360),
    ("mlp", "down_proj", 15360, 3840),
)


def _model():
    attention = _module(language.Attention)
    mlp = _module(language.MLP)
    layer = _module(language.TransformerBlock, self_attn=attention, mlp=mlp)
    for parent, name, input_dim, output_dim in _PROJECTIONS:
        setattr(getattr(layer, parent), name, _linear(input_dim, output_dim))
    config = SimpleNamespace(
        hidden_size=3840,
        intermediate_size=15360,
        num_hidden_layers=48,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=256,
    )
    # Sharing the block keeps this installation test allocation-light.
    text_model = _module(
        language.Gemma3Model,
        config=config,
        layers=[layer] * 48,
        embed_tokens=_module(nn.QuantizedEmbedding),
    )
    language_model = _module(
        language.LanguageModel, model=text_model, lm_head=_linear()
    )
    return SimpleNamespace(language_model=language_model, vision_tower=_linear())


@pytest.fixture
def m1_max(monkeypatch):
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "device_info", lambda: {"device_name": "Apple M1 Max"})


def test_installation_is_instance_scoped_and_idempotent(m1_max):
    model = _model()
    other = _model()
    original_call = nn.QuantizedLinear.__call__
    layer = model.language_model.model.layers[0]
    weight = layer.mlp.gate_proj.weight

    assert patch.apply_gemma3_prefill_linear_patch(model) == 7
    assert model.language_model._omlx_prefill_scratch_bytes == 225 * 1024**2
    assert model.language_model._omlx_prefill_scratch_min_tokens == 512
    assert patch.apply_gemma3_prefill_linear_patch(model) == 0
    assert model.language_model._omlx_prefill_scratch_bytes == 225 * 1024**2
    assert layer.mlp.gate_proj.weight is weight
    for parent, name, _, _ in _PROJECTIONS:
        assert type(getattr(getattr(layer, parent), name)) is patch.Gemma3PrefillLinear
    assert (
        type(other.language_model.model.layers[0].mlp.gate_proj) is nn.QuantizedLinear
    )
    assert type(model.language_model.lm_head) is nn.QuantizedLinear
    assert type(model.vision_tower) is nn.QuantizedLinear
    assert type(model.language_model.model.embed_tokens) is nn.QuantizedEmbedding
    assert nn.QuantizedLinear.__call__ is original_call


def test_installation_excludes_linear_subclasses(m1_max):
    class CustomLinear(nn.QuantizedLinear):
        pass

    model = _model()
    attention = model.language_model.model.layers[0].self_attn
    attention.k_proj = _linear(cls=CustomLinear)
    assert patch.apply_gemma3_prefill_linear_patch(model) == 6
    assert type(attention.k_proj) is CustomLinear


def test_scratch_bound_counts_only_simultaneous_patched_groups(m1_max):
    model = _model()
    layer = model.language_model.model.layers[0]
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(layer.mlp, name, nn.Identity())
    assert patch.apply_gemma3_prefill_linear_patch(model) == 4
    assert model.language_model._omlx_prefill_scratch_bytes == 60 * 1024**2


def test_installation_declines_unmeasured_device(m1_max, monkeypatch):
    monkeypatch.setattr(mx, "device_info", lambda: {"device_name": "Apple M1 Pro"})
    assert patch.apply_gemma3_prefill_linear_patch(_model()) == 0


def test_installation_declines_other_model_geometry(m1_max):
    model = _model()
    model.language_model.model.config.intermediate_size = 16384
    assert patch.apply_gemma3_prefill_linear_patch(model) == 0


@pytest.mark.parametrize(
    "input_dim,output_dim,rows",
    [(3840, 2048, 512), (3840, 15360, 2047), (15360, 3840, 2048)],
)
def test_dense_projection_equals_packed_reference(input_dim, output_dim, rows):
    linear = _linear(input_dim, output_dim, cls=patch.Gemma3PrefillLinear)
    row = mx.arange(output_dim, dtype=mx.uint32)[:, None]
    word = mx.arange(input_dim // 8, dtype=mx.uint32)[None, :]
    linear.weight = (row + word) * mx.array(0x1234567, mx.uint32)
    group = mx.arange(input_dim // 64, dtype=mx.uint32)[None, :]
    linear.scales = ((row + group) % 4 + 1).astype(mx.bfloat16) / 64
    linear.biases = ((row + group) % 3).astype(mx.bfloat16) / 8 - 0.125
    # Binary fractions keep products and accumulator sums exactly representable.
    x = mx.random.randint(-8, 9, (1, rows, input_dim), key=mx.random.key(17))
    x = x.astype(mx.bfloat16) / 8
    reference = nn.QuantizedLinear.__call__(linear, x)
    actual = linear(x)
    mx.eval(reference, actual)
    assert actual.dtype == mx.bfloat16
    assert mx.array_equal(actual, reference).item()


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((1, 1, 3840), mx.bfloat16),
        ((1, 511, 3840), mx.bfloat16),
        ((1, 2049, 3840), mx.bfloat16),
        ((2, 512, 3840), mx.bfloat16),
        ((512, 3840), mx.bfloat16),
        ((1, 512, 3840), mx.float16),
    ],
)
def test_unsupported_calls_bypass_dense_path(monkeypatch, shape, dtype):
    sentinel = object()
    monkeypatch.setattr(nn.QuantizedLinear, "__call__", lambda self, x: sentinel)
    monkeypatch.setattr(
        mx, "dequantize", lambda *args, **kwargs: pytest.fail("dense path")
    )
    linear = _linear(cls=patch.Gemma3PrefillLinear)
    assert linear(mx.zeros(shape, dtype)) is sentinel


@pytest.mark.parametrize(
    "field,value", [("bits", 8), ("group_size", 128), ("mode", "mxfp4")]
)
def test_installation_declines_other_quantization(m1_max, field, value):
    model = _model()
    linear = model.language_model.model.layers[0].self_attn.k_proj
    setattr(linear, field, value)
    assert patch.apply_gemma3_prefill_linear_patch(model) == 6
    assert type(linear) is nn.QuantizedLinear


def test_installation_declines_other_parameter_layouts(m1_max):
    model = _model()
    attention = model.language_model.model.layers[0].self_attn
    attention.q_proj = _linear(3840, 4095)
    attention.k_proj.scales = attention.k_proj.scales.astype(mx.float16)
    attention.v_proj.bias = mx.zeros((2048,), mx.bfloat16)
    attention.o_proj.biases = None

    assert patch.apply_gemma3_prefill_linear_patch(model) == 3
    assert all(
        type(linear) is nn.QuantizedLinear
        for linear in (
            attention.q_proj,
            attention.k_proj,
            attention.v_proj,
            attention.o_proj,
        )
    )


def test_dense_weight_depends_on_activation_and_is_transient(monkeypatch):
    linear = _linear(cls=patch.Gemma3PrefillLinear)
    x = mx.zeros((1, 512, 3840), mx.bfloat16)
    saved_keys = set(linear)
    depends = mx.depends
    dequantize = mx.dequantize
    events = []

    def traced_depends(weight, activation):
        assert weight is linear.weight
        assert activation is x
        events.append("depends")
        return depends(weight, activation)

    def traced_dequantize(*args, **kwargs):
        events.append("dequantize")
        return dequantize(*args, **kwargs)

    monkeypatch.setattr(mx, "depends", traced_depends)
    monkeypatch.setattr(mx, "dequantize", traced_dequantize)
    linear(x)
    linear(x)
    assert events == ["depends", "dequantize", "depends", "dequantize"]
    assert set(linear) == saved_keys
