# SPDX-License-Identifier: Apache-2.0
"""Parity and lifecycle coverage for Qwen native-MTP optimizations."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.activations import swiglu

from omlx.patches.qwen35_dense_gate_up import _ProjectionSlice, _SharedProjection
from omlx.patches.qwen35_exact_crossrow_qmm import exact_crossrow
from omlx.patches.qwen35_fused_swiglu_qmm import try_fast_swiglu
from omlx.patches.qwen35_verify_qmm import set_verify_qmm_armed, vk_qmm


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize(
    ("bits", "rows", "outputs"),
    [
        (4, 2, 1024),
        (4, 3, 1024),
        (4, 4, 1024),
        (5, 2, 1024),
        (5, 3, 1024),
        (5, 4, 1024),
        (4, 4, 48),
        (5, 4, 48),
    ],
)
def test_exact_crossrow_matches_independent_serial_rows(bits, rows, outputs):
    linear = nn.QuantizedLinear(
        512,
        outputs,
        bias=False,
        group_size=64,
        bits=bits,
        mode="affine",
    )
    linear.scales = linear.scales.astype(mx.bfloat16)
    linear.biases = linear.biases.astype(mx.bfloat16)
    x = mx.random.normal((1, rows, 512)).astype(mx.bfloat16)

    expected = mx.concatenate(
        [linear(x[:, row : row + 1]) for row in range(rows)],
        axis=1,
    )
    actual = exact_crossrow(linear, x)

    assert actual is not None
    mx.eval(expected, actual)
    assert bool(mx.array_equal(expected, actual).item())


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("rows", [3, 4])
def test_fused_q4_swiglu_matches_verify_projection_boundary(rows):
    linear = nn.QuantizedLinear(
        512,
        128,
        bias=False,
        group_size=64,
        bits=4,
        mode="affine",
    )
    linear.scales = linear.scales.astype(mx.bfloat16)
    linear.biases = linear.biases.astype(mx.bfloat16)
    x = mx.random.normal((1, rows, 512)).astype(mx.bfloat16)

    projected = vk_qmm(
        x[0],
        linear.weight,
        linear.scales,
        linear.biases,
        bits=4,
        group_size=64,
    )
    gate, up = mx.split(projected, 2, axis=-1)
    expected = swiglu(gate, up)

    set_verify_qmm_armed(True)
    try:
        actual = try_fast_swiglu(linear, x, target_verify=True)
    finally:
        set_verify_qmm_armed(False)

    assert actual is not None
    mx.eval(expected, actual)
    assert bool(mx.array_equal(expected, actual[0]).item())


def test_mtp_qkv_projection_slices_share_once_and_clear_after_v():
    calls = []

    def fused(x):
        calls.append(x)
        return mx.arange(12).reshape(1, 12)

    shared = _SharedProjection(fused)
    q_proj = _ProjectionSlice(shared, 0, 6, last=False)
    k_proj = _ProjectionSlice(shared, 6, 9, last=False)
    v_proj = _ProjectionSlice(shared, 9, 12, last=True)
    x = mx.zeros((1, 4))

    assert q_proj(x).shape == (1, 6)
    assert k_proj(x).shape == (1, 3)
    assert v_proj(x).shape == (1, 3)
    assert calls == [x]
    assert shared.input is None
    assert shared.output is None

    q_proj(x)
    assert calls == [x, x]
