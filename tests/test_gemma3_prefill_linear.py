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


def _model():
    attention = _module(language.Attention)
    mlp = _module(language.MLP)
    layer = _module(language.TransformerBlock, self_attn=attention, mlp=mlp)
    for parent, name, input_dim, output_dim in patch._PROJECTIONS:
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
    assert layer.mlp.gate_proj.weight is weight
    for parent, name, _, _ in patch._PROJECTIONS:
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
    assert patch.apply_gemma3_prefill_linear_patch(model) == 7
    language_model = model.language_model
    assert patch._prefill_scratch_bytes(language_model) == 225 * 1024**2
    layer = language_model.model.layers[0]
    layer.mlp = SimpleNamespace()
    assert patch._prefill_scratch_bytes(language_model) == 60 * 1024**2
    layer.self_attn = SimpleNamespace()
    assert patch._prefill_scratch_bytes(language_model) == 0


@pytest.mark.parametrize("device", ["Apple M1 Pro", "Apple M2 Max", "Apple M5 Max"])
def test_installation_declines_unmeasured_devices(m1_max, monkeypatch, device):
    monkeypatch.setattr(mx, "device_info", lambda: {"device_name": device})
    assert patch.apply_gemma3_prefill_linear_patch(_model()) == 0


def test_installation_declines_other_model_geometry(m1_max):
    model = _model()
    model.language_model.model.config.intermediate_size = 16384
    assert patch.apply_gemma3_prefill_linear_patch(model) == 0


@pytest.mark.parametrize(
    "input_dim,output_dim", [(3840, 2048), (3840, 15360), (15360, 3840)]
)
@pytest.mark.parametrize("rows", [512, 2047, 2048])
def test_dense_projection_equals_packed_reference(input_dim, output_dim, rows):
    linear = _linear(input_dim, output_dim, cls=patch.Gemma3PrefillLinear)
    # Powers-of-two inputs and affine metadata make this an exact arithmetic
    # check of dequantization and projection, including ragged query rows.
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
        ((1, 128, 3840), mx.bfloat16),
        ((1, 511, 3840), mx.bfloat16),
        ((1, 2049, 3840), mx.bfloat16),
        ((2, 512, 3840), mx.bfloat16),
        ((512, 3840), mx.bfloat16),
        ((1, 512, 3840), mx.float16),
        ((1, 512, 3840), mx.float32),
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
def test_changed_quantization_uses_original_call(monkeypatch, field, value):
    sentinel = object()
    monkeypatch.setattr(nn.QuantizedLinear, "__call__", lambda self, x: sentinel)
    linear = _linear(cls=patch.Gemma3PrefillLinear)
    setattr(linear, field, value)
    assert linear(mx.zeros((1, 512, 3840), mx.bfloat16)) is sentinel


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
