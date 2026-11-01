# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp vocabulary head launches keep stock quantized_matmul bits."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches import qwen4_head_qmv

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


@pytest.fixture(autouse=True)
def fresh_verdicts():
    qwen4_head_qmv._VERDICTS.clear()
    yield
    qwen4_head_qmv._VERDICTS.clear()


def _head(k, n, bits, group_size, seed):
    mx.random.seed(seed)
    linear = nn.QuantizedLinear(k, n, bias=False, group_size=group_size, bits=bits)
    weight = (mx.random.normal((n, k)) * 0.05).astype(mx.bfloat16)
    linear.weight, linear.scales, linear.biases = mx.quantize(
        weight, group_size=group_size, bits=bits
    )
    return linear


def _routed(linear):
    """The same parameters behind the routed head class."""
    routed = nn.QuantizedLinear(
        linear.weight.shape[1] * 32 // linear.bits,
        linear.weight.shape[0],
        bias=False,
        group_size=linear.group_size,
        bits=linear.bits,
    )
    routed.weight, routed.scales, routed.biases = (
        linear.weight,
        linear.scales,
        linear.biases,
    )
    routed.__class__ = qwen4_head_qmv._Qwen4HeadLinear
    return routed


def _bits(a):
    return a.view(mx.uint16)


@pytest.mark.parametrize("bits, group_size", [(8, 64), (4, 64), (5, 128), (6, 64)])
@pytest.mark.parametrize("rows", [1, 2, 3, 5, 6, 7, 8, 9, 11])
def test_head_rows_equal_stock_quantized_matmul(bits, group_size, rows):
    k, n = 2560, 16384
    stock = _head(k, n, bits, group_size, seed=rows + bits)
    routed = _routed(stock)
    x = mx.random.normal((rows, 1, k)).astype(mx.bfloat16)
    want, got = stock(x), routed(x)
    assert want.shape == got.shape and want.dtype == got.dtype
    assert mx.array_equal(_bits(want), _bits(got)).item()
    # Shapes that take the fewer-pass launch were checked on this GPU and kept it.
    passes = qwen4_head_qmv.stock_qmv_wide_passes(rows, k, n, bits, "affine")
    if qwen4_head_qmv._rows_per_pass(rows, k, n, bits, "affine"):
        assert passes > 1 and list(qwen4_head_qmv._VERDICTS.values()) == [True]


def test_stock_dispatch_model(monkeypatch):
    """MLX 0.32.4's qmv_wide choice: affine, family >= 15, rows below the batch limit."""
    monkeypatch.setattr(qwen4_head_qmv.mx, "__version__", "0.32.4")
    monkeypatch.setattr(qwen4_head_qmv, "_gpu", lambda: (15, "d"))  # M3 Ultra
    passes = qwen4_head_qmv.stock_qmv_wide_passes
    assert [passes(m, 2560, 248320, 8, "affine") for m in range(1, 13)] == [
        0,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        3,
        0,
    ]
    assert passes(8, 2560, 248320, 8, "mxfp4") == 0
    assert passes(8, 128, 4096, 4, "affine") == 0  # qmv_quad
    # Fewer passes only where they help: 6-8 rows in one pass, 11 in two.
    assert [
        qwen4_head_qmv._rows_per_pass(m, 2560, 248320, 8, "affine")
        for m in range(1, 13)
    ] == [0, 0, 0, 0, 0, 6, 7, 8, 0, 0, 6, 0]
    monkeypatch.setattr(
        qwen4_head_qmv, "_gpu", lambda: (14, "d")
    )  # affine qmv_wide needs family 15
    assert passes(8, 2560, 248320, 8, "affine") == 0
    monkeypatch.setattr(
        qwen4_head_qmv, "_gpu", lambda: (17, "s")
    )  # M5 Max: limit 13 for wide heads
    assert passes(12, 2560, 248320, 8, "affine") == 3
    monkeypatch.setattr(qwen4_head_qmv.mx, "__version__", "0.33.0")
    assert passes(8, 2560, 248320, 8, "affine") == 0


def test_mismatching_shape_keeps_stock(monkeypatch):
    stock = _head(2560, 8192, 8, 64, seed=3)
    routed = _routed(stock)
    monkeypatch.setattr(
        qwen4_head_qmv,
        "_wide",
        lambda linear, x, per_pass: mx.zeros((x.shape[0], 8192), x.dtype),
    )
    monkeypatch.setattr(qwen4_head_qmv, "_rows_per_pass", lambda *a: 8)
    x = mx.random.normal((8, 1, 2560)).astype(mx.bfloat16)
    assert mx.array_equal(_bits(stock(x)), _bits(routed(x))).item()
    assert list(qwen4_head_qmv._VERDICTS.values()) == [False]


def test_patch_only_routes_qwen4_heads():
    class Plain(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = _head(64, 64, 8, 64, seed=1)

    model = Plain()
    assert not qwen4_head_qmv.apply_qwen4_head_qmv_patch(model)
    assert type(model.lm_head) is nn.QuantizedLinear
