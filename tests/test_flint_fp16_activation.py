# SPDX-License-Identifier: Apache-2.0
"""Tests for Flint's float16 range-safe SwiGLU."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
flint = pytest.importorskip("mlx_vlm.models.flint.flint")

from mlx.utils import tree_flatten, tree_map  # noqa: E402
from mlx_vlm.models.flint.config import ModelConfig  # noqa: E402
from mlx_vlm.models.mlp import SwiGLUMLP  # noqa: E402

from omlx.patches import flint_decode, flint_fp16_activation  # noqa: E402

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="Flint decode kernels need Metal"
)

# silu(gate) * up grows by ~GAIN**2 (~1.5e5 here, past float16's 65504)
# and down_proj takes one GAIN back, so every layer's output stays in range.
GAIN = 256.0


def _tiny_model():
    config = ModelConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=128,
        layer_types=["full_attention", "sliding_attention"],
        sliding_window=16,
        num_experts=32,
        num_experts_per_tok=4,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=128,
        mlp_only_layers=[0],
    )
    mx.random.seed(11)
    model = flint.Model(config)
    for layer in model.layers:
        mlp = layer.mlp
        for glu in (
            getattr(mlp, "switch_mlp", None),
            getattr(mlp, "shared_expert", mlp),
        ):
            if glu is None:
                continue
            glu.gate_proj.weight = glu.gate_proj.weight * GAIN
            glu.up_proj.weight = glu.up_proj.weight * GAIN
            glu.down_proj.weight = glu.down_proj.weight / GAIN
    mx.eval(model.parameters())
    return model


def _cast(model, dtype):
    model.update(tree_map(lambda p: p.astype(dtype), model.parameters()))
    mx.eval(model.parameters())
    return model


def _logits(model, tokens):
    return model(tokens).logits.astype(mx.float32)


def _ref_logits(tokens):
    # The float16 weights, run in fp32.
    model = _cast(_cast(_tiny_model(), mx.float16), mx.float32)
    return _logits(model, tokens)


TOKENS = mx.array([[3, 17, 42, 99, 5, 64, 7, 120]])


def test_stock_float16_overflows():
    """The fixture reproduces #4784: float16 logits go NaN on mlx-vlm's path."""
    model = _cast(_tiny_model(), mx.float16)
    assert not mx.isnan(_ref_logits(TOKENS)).any().item()
    assert mx.isnan(_logits(model, TOKENS)).any().item()


def test_patched_float16_matches_fp32_reference():
    model = _cast(_tiny_model(), mx.float16)
    assert flint_fp16_activation.apply_flint_fp16_activation_patch(model)
    out = _logits(model, TOKENS)
    ref = _ref_logits(TOKENS)
    assert not mx.isnan(out).any().item()
    assert mx.allclose(out, ref, atol=5e-2, rtol=5e-2).item()


def test_patched_float16_one_token_decode_path():
    """The fused one-token decode path undoes the routed scale too."""
    model = _cast(_tiny_model(), mx.float16)
    assert flint_fp16_activation.apply_flint_fp16_activation_patch(model)
    assert flint_decode.apply_flint_decode_patch()
    token = TOKENS[:, :1]
    out = _logits(model, token)
    assert mx.allclose(out, _ref_logits(token), atol=5e-2, rtol=5e-2).item()


def test_patch_keeps_parameters():
    model = _cast(_tiny_model(), mx.float16)
    before = [k for k, _ in tree_flatten(model.parameters())]
    assert flint_fp16_activation.apply_flint_fp16_activation_patch(model)
    assert [k for k, _ in tree_flatten(model.parameters())] == before


def test_bfloat16_model_is_untouched():
    model = _cast(_tiny_model(), mx.bfloat16)
    assert not flint_fp16_activation.apply_flint_fp16_activation_patch(model)
    assert type(model.layers[0].mlp) is SwiGLUMLP
    activation = model.layers[1].mlp.switch_mlp.activation
    assert not isinstance(activation, flint_fp16_activation.ScaledSwiGLU)


@pytest.mark.parametrize("size", [512, 3072, 49152])
@pytest.mark.parametrize("peak", [1.0, 1e3])
def test_kernel_matches_compiled_graph(size, peak):
    """The single-launch kernel against the compiled graph, on strided views
    (the halves of a fused [gate; up] projection)."""
    mx.random.seed(size)
    gate_up = (mx.random.normal((1, size // 512, 1024)) * peak).astype(mx.float16)
    gate, up = mx.split(gate_up, 2, axis=-1)
    out, scale = flint_fp16_activation._scaled_swiglu_launch(gate, up)
    ref, ref_scale = flint_fp16_activation._scaled_swiglu_fp16(gate, up)
    assert out.shape == up.shape and out.dtype == mx.float16
    assert scale.item() == ref_scale.item()
    if peak > 1:
        assert scale.item() > 1
    diff = mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))
    assert (diff / (mx.abs(ref.astype(mx.float32)) + 1e-3)).max().item() < 2e-3


def test_scaled_combine_matches_separate_multiplies():
    mx.random.seed(5)
    top_k, dim = 6, 2048
    dtype = mx.float16
    routed = mx.random.normal((1, 1, top_k, dim)).astype(dtype)
    weights = mx.softmax(mx.random.normal((top_k,)))
    shared = mx.random.normal((1, 1, dim)).astype(dtype)
    residual = mx.random.normal((1, 1, dim)).astype(dtype)
    rs, ss = mx.array([3.5], dtype), mx.array([1.25], dtype)
    moe = ((routed * rs).astype(mx.float32) * weights[:, None]).sum(axis=-2)
    ref = residual + (moe.astype(dtype) + shared * ss)
    out = flint_decode.fused_combine(routed, weights, shared, residual, rs, ss)
    assert mx.array_equal(out, ref).item()
