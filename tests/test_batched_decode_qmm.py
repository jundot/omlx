# SPDX-License-Identifier: Apache-2.0
"""Batched decode routing on M1-class GPUs: which row counts leave the stock qmm."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches import (
    batched_decode_qmm,
    m1_verify_mma,
    m1_verify_qmv,
    qwen35_verify_qmm,
)

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

K, N = 2048, 4096


@pytest.fixture
def m1_batched_decode(monkeypatch):
    monkeypatch.setattr(m1_verify_qmv, "enabled", lambda: True)
    monkeypatch.setattr(m1_verify_mma, "enabled", lambda: True)
    batched_decode_qmm._route.cache_clear()
    batched_decode_qmm._CHECKED.clear()
    assert batched_decode_qmm.apply_batched_decode_qmm_patch()
    yield
    batched_decode_qmm.remove_batched_decode_qmm_patch()
    batched_decode_qmm._route.cache_clear()
    batched_decode_qmm._CHECKED.clear()


def _linear(bits, dtype, n=N):
    mx.random.seed(0)
    linear = nn.QuantizedLinear(K, n, bias=False, group_size=64, bits=bits)
    weight = (mx.random.normal((n, K)) * 0.05).astype(dtype)
    linear.weight, linear.scales, linear.biases = mx.quantize(
        weight, group_size=64, bits=bits
    )
    return linear


def _routes_taken(monkeypatch, linear, x):
    taken = []
    real = qwen35_verify_qmm.run_route

    def spy(layer, x2, route, **kwargs):
        taken.append(route)
        return real(layer, x2, route, **kwargs)

    monkeypatch.setattr(qwen35_verify_qmm, "run_route", spy)
    return taken, linear(x)


@pytest.mark.parametrize("rows,route", [(3, "m1"), (4, "m1"), (5, "m1_mma")])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_few_decode_rows_take_the_m1_kernels(
    monkeypatch, m1_batched_decode, rows, route, dtype
):
    linear = _linear(4, dtype)
    x = mx.random.normal((rows, 1, K)).astype(dtype)
    stock = mx.quantized_matmul(
        x.reshape(rows, K),
        linear.weight,
        linear.scales,
        linear.biases,
        transpose=True,
        group_size=64,
        bits=4,
    ).reshape(rows, 1, N)

    taken, routed = _routes_taken(monkeypatch, linear, x)

    assert taken == [route]
    assert routed.shape == (rows, 1, N)
    assert mx.allclose(routed, stock, atol=2e-2, rtol=2e-2).item()


def test_one_row_and_mismatched_dtype_keep_the_stock_path(
    monkeypatch, m1_batched_decode
):
    taken, _ = _routes_taken(
        monkeypatch, _linear(4, mx.float16), mx.ones((1, 1, K), dtype=mx.float16)
    )
    assert taken == []
    taken, _ = _routes_taken(
        monkeypatch, _linear(4, mx.float16), mx.ones((4, 1, K), dtype=mx.bfloat16)
    )
    assert taken == []


def test_two_rows_keep_the_stock_path(monkeypatch, m1_batched_decode):
    # Two-row calls measured slower through the M1 kernels even at the large
    # projections the size floor admits, so the floor starts at three rows.
    taken, _ = _routes_taken(
        monkeypatch, _linear(4, mx.float16), mx.ones((2, 1, K), dtype=mx.float16)
    )
    assert taken == []


@pytest.mark.parametrize("rows", [2, 3, 4, 5])
def test_small_projections_keep_the_stock_path(m1_batched_decode, rows):
    # 512x2048 is eligible for the M1 kernels but measured slower than stock.
    assert qwen35_verify_qmm._verify_route(rows, 512, 2048, 4, 64, mx.float16) in (
        "m1",
        "m1_mma",
    )
    assert batched_decode_qmm._route(rows, 512, 2048, 4, 64, mx.float16) is None


def test_other_routes_stay_off_below_six_rows(monkeypatch, m1_batched_decode):
    # No M1 kernel reads 6-bit weights; sg8 would, from four rows.
    assert qwen35_verify_qmm._verify_route(4, K, N, 6, 64, mx.float16) == "sg8"
    taken, _ = _routes_taken(
        monkeypatch, _linear(6, mx.float16), mx.ones((4, 1, K), dtype=mx.float16)
    )
    assert taken == []


@pytest.mark.parametrize("bits,rows", [(4, 2), (4, 5), (5, 3)])
def test_other_gpus_stay_on_the_stock_path(monkeypatch, bits, rows):
    monkeypatch.setattr(m1_verify_qmv, "enabled", lambda: False)
    monkeypatch.setattr(m1_verify_mma, "enabled", lambda: False)
    batched_decode_qmm._route.cache_clear()
    assert batched_decode_qmm.apply_batched_decode_qmm_patch()
    try:
        taken, _ = _routes_taken(
            monkeypatch,
            _linear(bits, mx.float16),
            mx.ones((rows, 1, K), dtype=mx.float16),
        )
    finally:
        batched_decode_qmm.remove_batched_decode_qmm_patch()
        batched_decode_qmm._route.cache_clear()
    assert taken == []


def test_routed_rows_keep_the_bias(monkeypatch, m1_batched_decode):
    mx.random.seed(1)
    linear = nn.QuantizedLinear(K, N, bias=True, group_size=64, bits=4)
    weight = (mx.random.normal((N, K)) * 0.05).astype(mx.float16)
    linear.weight, linear.scales, linear.biases = mx.quantize(
        weight, group_size=64, bits=4
    )
    linear.bias = mx.random.normal((N,)).astype(mx.float16)
    x = mx.random.normal((4, 1, K)).astype(mx.float16)
    stock = (
        mx.quantized_matmul(
            x.reshape(4, K),
            linear.weight,
            linear.scales,
            linear.biases,
            transpose=True,
            group_size=64,
            bits=4,
        )
        + linear.bias
    ).reshape(4, 1, N)
    taken, routed = _routes_taken(monkeypatch, linear, x)
    assert taken == ["m1"]
    assert mx.allclose(routed, stock, atol=2e-2, rtol=2e-2).item()
