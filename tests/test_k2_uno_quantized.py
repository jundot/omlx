# SPDX-License-Identifier: Apache-2.0
"""Q8 block arithmetic and eligibility for the Uno projection path."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches.k2_horizon import quantized
from omlx.patches.k2_horizon.uno_adapter import TARGETS, ConditionalLoRALinear


def qlinear(k=4096, n=1024, *, bits=8, group=64, dtype=mx.bfloat16):
    linear = nn.Linear(k, n, bias=False)
    linear.set_dtype(dtype)
    return nn.QuantizedLinear.from_linear(linear, group_size=group, bits=bits)


def shell(linear):
    layer = SimpleNamespace(
        **{
            scope: SimpleNamespace(**dict.fromkeys(names, linear))
            for scope, names in TARGETS.items()
        }
    )
    return SimpleNamespace(
        layers=[layer],
        args=SimpleNamespace(tie_word_embeddings=False),
        lm_head=linear,
        _uno_adapter_loaded=True,
    )


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({}, 8),
        ({"bits": 4}, 0),
        ({"group": 128}, 0),
        ({"dtype": mx.float16}, 0),
        ({"k": 2048}, 8),
        ({"k": 768, "n": 16}, 8),
        ({"k": 64, "n": 8}, 8),
        ({"n": 1025}, 0),
    ],
)
def test_q8_block_marks_only_eligible_weights(monkeypatch, changes, expected):
    monkeypatch.setattr(mx, "default_device", lambda: mx.Device(mx.gpu))
    linear = qlinear(**changes)
    assert quantized.enable_q8_blocks(shell(linear)) == expected
    assert bool(getattr(linear, "_omlx_uno_q8_block", False)) == bool(expected)


def test_q8_block_does_not_restrict_gpu_architecture(monkeypatch):
    def architecture_not_needed():
        raise AssertionError("GPU architecture must not gate Uno Q8")

    monkeypatch.setattr(mx, "device_info", architecture_not_needed)
    monkeypatch.setattr(mx, "default_device", lambda: mx.Device(mx.gpu))
    assert quantized.enable_q8_blocks(shell(qlinear())) == 8


def test_q8_block_requires_uno():
    model = shell(qlinear())
    model._uno_adapter_loaded = False
    assert quantized.enable_q8_blocks(model) == 0


def test_q8_block_does_not_enable_on_cpu(monkeypatch):
    monkeypatch.setattr(mx, "default_device", lambda: mx.Device(mx.cpu))
    model = SimpleNamespace(_uno_adapter_loaded=True)
    assert quantized.enable_q8_blocks(model) == 0


def test_uno_engine_enables_q8_before_compilation_and_reports_it(monkeypatch):
    from omlx.engine.batched import BatchedEngine
    from omlx.engine.uno import UnoEngine
    from omlx.patches.k2_horizon import compiled, uno_adapter, uno_batch

    events = []
    engine = object.__new__(UnoEngine)
    engine._model = SimpleNamespace(parameters=lambda: {})
    engine._model_name = "base"
    engine._adapter_path = "adapter"
    engine._tokenizer = SimpleNamespace(eos_token_ids=[0])
    bundle = SimpleNamespace(
        adapter_path=Path("adapter"), base_model_id="IFM/K2-Horizon-7B"
    )
    monkeypatch.setattr("omlx.engine.uno.resolve_uno_bundle", lambda *args: bundle)
    monkeypatch.setattr(
        uno_adapter, "load_uno_adapter", lambda *a, **kw: events.append("adapter")
    )
    monkeypatch.setattr(mx, "eval", lambda *args: None)
    monkeypatch.setattr(
        quantized, "enable_q8_blocks", lambda model: events.append("q8") or 253
    )
    monkeypatch.setattr(compiled, "can_compile_blocks", lambda model: True)

    def compile_blocks(model):
        assert model._omlx_uno_q8_block_projections == 253
        events.append("compile")

    monkeypatch.setattr(compiled, "install_compiled_blocks", compile_blocks)
    monkeypatch.setattr(uno_batch, "install_cache_hooks", lambda: None)
    monkeypatch.setattr(BatchedEngine, "get_stats", lambda self: {})
    engine._prepare_uno_model()
    assert events == ["adapter", "q8", "compile"]
    assert engine.get_stats()["uno"]["q8_block_projections"] == 253


def test_grouped_norm_compilation_isolates_model_widths():
    from omlx.patches.k2_horizon.k2_horizon_model import GroupedRMSNorm

    for width, tokens in [(4096, 8), (64, 4), (4096, 3), (64, 1)]:
        layer = GroupedRMSNorm(width, groups=2, eps=1e-5)
        layer.set_dtype(mx.bfloat16)
        x = mx.random.normal((1, tokens, width)).astype(mx.bfloat16)
        expected = (
            mx.fast.rms_norm(
                x.astype(mx.float32).reshape(1, tokens, 2, width // 2), None, 1e-5
            )
            .reshape(x.shape)
            .astype(mx.bfloat16)
        )
        actual = layer(x)
        assert actual.shape == x.shape
        assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((1, 1, 4096), mx.bfloat16),
        ((1, 7, 4096), mx.bfloat16),
        ((1, 9, 4096), mx.bfloat16),
        ((2, 8, 4096), mx.bfloat16),
        ((8, 4096), mx.bfloat16),
        ((1, 8, 4096), mx.float16),
        ((1, 8, 2048), mx.bfloat16),
    ],
)
def test_q8_block_leaves_other_calls_on_native_path(monkeypatch, shape, dtype):
    native = Mock(return_value="native")
    native._omlx_uno_q8_block = True
    native.weight = SimpleNamespace(shape=(1024, 1024))
    kernel = Mock(side_effect=AssertionError("unexpected custom kernel"))
    monkeypatch.setattr(quantized, "_kernel", kernel)
    x = mx.zeros(shape, dtype=dtype)
    assert quantized.project_linear(native, x) == "native"
    native.assert_called_once_with(x)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize(
    "k,n",
    [
        (64, 8),
        (128, 8),
        (192, 16),
        (768, 16),
        (2048, 1024),
        (4096, 1024),
        (4096, 4096),
        (4096, 12288),
        (12288, 4096),
        (4096, 65536),
    ],
)
@pytest.mark.parametrize("bias", [False, True])
def test_q8_block_matches_mlx_and_conditional_clean_row(k, n, bias):
    mx.random.seed(123)
    linear = qlinear(k, n)
    if bias:
        linear.bias = mx.random.normal((n,)).astype(mx.bfloat16)
    x = mx.random.normal((1, 8, k)).astype(mx.bfloat16)
    # Include zeros and signed values at several scales, plus a noncontiguous view.
    x = (
        x
        * mx.array([1, 0, -0.001, 0.1, -1, 10, -100, 1000], mx.bfloat16)[None, :, None]
    )
    x = mx.concatenate([x, x], axis=-1)[..., ::2]
    expected = linear(x)
    mx.eval(expected)
    linear._omlx_uno_q8_block = True
    actual = mx.compile(lambda x: quantized.project_linear(linear, x))(x)
    assert mx.array_equal(actual, expected).item()
    a = (mx.random.normal((8, k)) * 0.01).astype(mx.bfloat16)
    b = (mx.random.normal((n, 8)) * 0.01).astype(mx.bfloat16)
    adapter = ConditionalLoRALinear(linear, a, b, 2.0)
    mask = mx.array([[0] + [1] * 7])
    adapted = adapter.conditional_forward(x, mask)
    linear._omlx_uno_q8_block = False
    reference_adapted = adapter.conditional_forward(x, mask)
    assert mx.array_equal(adapted, reference_adapted).item()
    assert mx.array_equal(adapted[:, :1], expected[:, :1]).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("compiled", [False, True])
def test_q8_block_preserves_full_model_logits_and_kv(compiled):
    from mlx_lm.models.cache import make_prompt_cache
    from test_k2_horizon import small_config

    from omlx.patches.k2_horizon.compiled import install_compiled_blocks
    from omlx.patches.k2_horizon.k2_horizon_model import Model, ModelArgs

    mx.random.seed(345)
    model = Model(
        ModelArgs.from_dict(
            small_config(
                hidden_size=4096,
                intermediate_size=4096,
                num_hidden_layers=1,
                num_attention_heads=32,
                num_key_value_heads=8,
                head_dim=128,
                vocab_size=1024,
            )
        )
    )
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, bits=8, group_size=64)
    for scope, names in TARGETS.items():
        owner = getattr(model.layers[0], scope)
        for name in names:
            linear = getattr(owner, name)
            n, packed_k = linear.weight.shape
            a = (mx.random.normal((8, packed_k * 4)) * 0.01).astype(mx.bfloat16)
            b = (mx.random.normal((n, 8)) * 0.01).astype(mx.bfloat16)
            setattr(owner, name, ConditionalLoRALinear(linear, a, b, 2.0))
    model._uno_adapter_loaded = True
    native = model.model
    before = []
    for enabled in [False, True]:
        if enabled:
            assert quantized.enable_q8_blocks(model) == 8
        model.model = native
        if compiled:
            install_compiled_blocks(model)
        cache = make_prompt_cache(model)
        mx.eval(model(mx.array([[2, 3, 5]]), cache=cache))
        # Rollback after conditional proposals, then materialize clean verification.
        for index, mask in enumerate([mx.array([[0] + [1] * 7]), None]):
            result = model(
                mx.array([[7, 11, 13, 17, 19, 23, 29, 31]]), cache=cache, lora_mask=mask
            )
            mx.eval(result, [c.state for c in cache])
            if not enabled:
                before.append((result, [c.state for c in cache]))
            else:
                assert mx.array_equal(result, before[index][0]).item()
                for current, saved in zip(cache, before[index][1]):
                    assert current.offset == 11
                    assert all(
                        mx.array_equal(a, b).item()
                        for a, b in zip(current.state, saved)
                    )
            for c in cache:
                c.trim(8)
