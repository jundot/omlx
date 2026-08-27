# SPDX-License-Identifier: Apache-2.0
"""Parity and dispatch gates for the experimental DS4 full-MoE decode path."""

from __future__ import annotations

from unittest.mock import patch

import pytest


def _native_fast():
    pytest.importorskip("mlx.core")
    from omlx.custom_kernels.glm_moe_dsa import fast

    if not fast.is_native_available() or not fast.has_symbol(
        "deepseek_mxfp4_full_decode"
    ):
        pytest.skip("experimental DeepSeek full-MoE native kernel is unavailable")
    return fast


def _fixture(dtype, tokens: int, index_dtype, hidden_dims: int = 512):
    import mlx.core as mx

    mx.random.seed(20260822 + tokens)
    experts = 8
    input_dims = 512
    output_dims = 512

    def random_weight(shape):
        return mx.random.randint(
            0,
            0x7FFFFFFF,
            shape=shape,
            dtype=mx.uint32,
        )

    def random_scales(shape):
        return mx.random.randint(116, 125, shape=shape, dtype=mx.uint8)

    up_weight = random_weight((experts, hidden_dims, input_dims // 8))
    up_scales = random_scales((experts, hidden_dims, input_dims // 32))
    gate_weight = random_weight((experts, hidden_dims, input_dims // 8))
    gate_scales = random_scales((experts, hidden_dims, input_dims // 32))
    down_weight = random_weight((experts, output_dims, hidden_dims // 8))
    down_scales = random_scales((experts, output_dims, hidden_dims // 32))
    x = mx.random.normal((1, tokens, input_dims)).astype(dtype)
    indices = mx.array(
        [[[(row + slot) % experts for slot in range(6)] for row in range(tokens)]],
        dtype=index_dtype,
    )
    scores = mx.softmax(
        mx.random.normal((1, tokens, 6)).astype(mx.float32),
        axis=-1,
    )
    return (
        x,
        up_weight,
        up_scales,
        gate_weight,
        gate_scales,
        down_weight,
        down_scales,
        indices,
        scores,
    )


def _reference(args, activation_limit: float = 10.0):
    import mlx.core as mx

    (
        x,
        up_weight,
        up_scales,
        gate_weight,
        gate_scales,
        down_weight,
        down_scales,
        indices,
        scores,
    ) = args
    expanded = mx.expand_dims(x, (-2, -3))
    kwargs = {
        "transpose": True,
        "group_size": 32,
        "bits": 4,
        "mode": "mxfp4",
        "sorted_indices": False,
    }
    up = mx.gather_qmm(
        expanded, up_weight, up_scales, None, rhs_indices=indices, **kwargs
    )
    gate = mx.gather_qmm(
        expanded, gate_weight, gate_scales, None, rhs_indices=indices, **kwargs
    )
    gate = mx.minimum(gate, activation_limit)
    up = mx.clip(up, -activation_limit, activation_limit)
    activated = (gate * mx.sigmoid(gate)) * up
    down = mx.gather_qmm(
        activated, down_weight, down_scales, None, rhs_indices=indices, **kwargs
    ).squeeze(-2)
    return (down * scores[..., None].astype(down.dtype)).sum(-2)


@pytest.mark.parametrize(
    ("dtype_name", "tokens", "index_dtype_name"),
    [
        ("float16", 1, "uint32"),
        ("bfloat16", 1, "uint32"),
        # Fixed-depth DS4 verification is six rows (one target row plus five
        # drafts). Keep the real B=6 geometry in the parity matrix; B=1 alone
        # can hide batch-stride and ordered-accumulation bugs in the native
        # kernel.
        ("bfloat16", 6, "uint32"),
        ("float16", 4, "int32"),
        ("bfloat16", 4, "int32"),
    ],
)
def test_deepseek_mxfp4_full_decode_is_bit_exact(
    dtype_name, tokens, index_dtype_name
):
    mx = pytest.importorskip("mlx.core")
    fast = _native_fast()
    args = _fixture(getattr(mx, dtype_name), tokens, getattr(mx, index_dtype_name))

    reference = _reference(args)
    candidate = fast.deepseek_mxfp4_full_decode(*args, 10.0)
    mx.eval(reference, candidate)

    assert candidate.shape == reference.shape
    assert candidate.dtype == reference.dtype
    assert mx.array_equal(candidate, reference).item()


@pytest.mark.parametrize("hidden_dims", (768, 1280))
@pytest.mark.parametrize("tokens", (1, 2, 4))
def test_deepseek_mxfp4_full_decode_is_exact_on_asymmetric_tp_tails(
    hidden_dims, tokens
):
    mx = pytest.importorskip("mlx.core")
    fast = _native_fast()
    args = _fixture(mx.bfloat16, tokens, mx.uint32, hidden_dims=hidden_dims)

    reference = _reference(args)
    candidate = fast.deepseek_mxfp4_full_decode(*args, 10.0)
    mx.eval(reference, candidate)

    assert mx.array_equal(candidate, reference).item()


@pytest.mark.parametrize("rows", (1, 2, 4))
def test_deepseek_mxfp4_full_decode_row_probe_is_exact(monkeypatch, rows):
    """The isolated row-tile probe must preserve the stock B1 boundary."""
    mx = pytest.importorskip("mlx.core")
    fast = _native_fast()
    args = _fixture(mx.bfloat16, 1, mx.uint32, hidden_dims=768)
    monkeypatch.setenv("OMLX_DSV4_FULL_DECODE_ROWS", str(rows))

    reference = _reference(args)
    candidate = fast.deepseek_mxfp4_full_decode(*args, 10.0)
    mx.eval(reference, candidate)

    assert mx.array_equal(candidate, reference).item()


def test_switchglu_full_decode_seam_is_opt_in_and_returns_reduced_rows(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    fast = _native_fast()
    from omlx.patches.deepseek_v4 import switch_layers

    args = _fixture(mx.bfloat16, 1, mx.uint32, hidden_dims=768)
    (
        x,
        up_weight,
        up_scales,
        gate_weight,
        gate_scales,
        down_weight,
        down_scales,
        indices,
        scores,
    ) = args

    switch = switch_layers.SwitchGLU(512, 768, 8)
    switch.eval()
    for projection, weight, scales in (
        (switch.up_proj, up_weight, up_scales),
        (switch.gate_proj, gate_weight, gate_scales),
        (switch.down_proj, down_weight, down_scales),
    ):
        quantized = projection.to_quantized(group_size=32, bits=4, mode="mxfp4")
        quantized.weight = weight
        quantized.scales = scales
        quantized.biases = None
        if projection is switch.up_proj:
            switch.up_proj = quantized
        elif projection is switch.gate_proj:
            switch.gate_proj = quantized
        else:
            switch.down_proj = quantized

    switch.activation = type(
        "LimitedActivation",
        (),
        {
            "limit": 10.0,
            "fp32": False,
            "__call__": lambda self, up, gate: up,
        },
    )()

    calls = 0
    real_call = fast.deepseek_mxfp4_full_decode

    def counting_call(*call_args, **kwargs):
        nonlocal calls
        calls += 1
        return real_call(*call_args, **kwargs)

    monkeypatch.setattr(switch_layers, "_DEEPSEEK_MXFP4_FULL_DECODE", True)
    monkeypatch.setattr(
        switch_layers.mx,
        "device_info",
        lambda: {"device_name": "Apple M3 Ultra"},
    )
    with patch.object(
        switch_layers.glm_fast,
        "deepseek_mxfp4_full_decode",
        counting_call,
    ):
        output = switch(x, indices, scores=scores)
        mx.eval(output)

    assert calls == 1
    assert output.shape == x.shape
    assert mx.array_equal(output, _reference(args)).item()


def test_switchglu_full_decode_default_remains_disabled(monkeypatch):
    pytest.importorskip("mlx.core")
    from omlx.patches.deepseek_v4 import switch_layers

    monkeypatch.setattr(switch_layers, "_DEEPSEEK_MXFP4_FULL_DECODE", False)
    switch = switch_layers.SwitchGLU(16, 16, 8)
    assert not switch._can_use_mxfp4_full_decode(None, None, None)


def test_switchglu_full_decode_accepts_served_asymmetric_tp_width(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from omlx.patches.deepseek_v4 import switch_layers

    args = _fixture(mx.bfloat16, 1, mx.uint32, hidden_dims=768)
    x, up_w, up_s, gate_w, gate_s, down_w, down_s, indices, scores = args
    switch = switch_layers.SwitchGLU(512, 768, 8)
    switch.eval()
    for name, weight, scales in (
        ("up_proj", up_w, up_s),
        ("gate_proj", gate_w, gate_s),
        ("down_proj", down_w, down_s),
    ):
        projection = getattr(switch, name).to_quantized(
            group_size=32, bits=4, mode="mxfp4"
        )
        projection.weight = weight
        projection.scales = scales
        projection.biases = None
        setattr(switch, name, projection)
    switch.activation = type(
        "LimitedActivation",
        (),
        {"limit": 10.0, "fp32": False},
    )()
    monkeypatch.setattr(switch_layers, "_DEEPSEEK_MXFP4_FULL_DECODE", True)
    monkeypatch.setattr(
        switch_layers.mx,
        "device_info",
        lambda: {"device_name": "Apple M3 Ultra"},
    )
    monkeypatch.setattr(
        switch_layers, "_DEEPSEEK_MXFP4_FULL_DECODE_MAX_TOKENS", 1
    )
    monkeypatch.setattr(
        switch_layers.glm_fast,
        "has_symbol",
        lambda name: name == "deepseek_mxfp4_full_decode",
    )
    assert switch._can_use_mxfp4_full_decode(x, indices, scores)
