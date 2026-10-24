# SPDX-License-Identifier: Apache-2.0
"""M1-class verify kernels (multi-row qmv, 5-16 row MMA): numerics, gating, routing."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches import m1_verify_mma, m1_verify_qmv, qwen35_verify_qmm

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


@pytest.fixture
def qmv_only(monkeypatch):
    monkeypatch.setattr(m1_verify_qmv, "enabled", lambda: True)
    # Pin the 5-16 row MMA route off so the sg8/qmv boundaries stay visible
    # (the MMA tests below cover it).
    monkeypatch.setattr(m1_verify_mma, "enabled", lambda: False)
    yield
    qwen35_verify_qmm.set_verify_qmm_armed(False)


@pytest.fixture
def qmv_off(monkeypatch):
    monkeypatch.setattr(m1_verify_qmv, "enabled", lambda: False)


def _quantized(k, n, bits, group_size, dtype, seed=0):
    mx.random.seed(seed)
    weight = (mx.random.normal((n, k)) * 0.05).astype(dtype)
    return mx.quantize(weight, group_size=group_size, bits=bits)


@pytest.mark.parametrize("bits,rows", [(4, 2), (4, 3), (4, 4), (5, 2), (5, 3)])
@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_qmv_matches_fp32_reference(qmv_only, bits, rows, group_size, dtype):
    K, N = 2048, 1024
    w, scales, biases = _quantized(K, N, bits, group_size, dtype)
    x = mx.random.normal((rows, K)).astype(dtype)
    assert m1_verify_qmv.eligible(rows, K, N, bits, group_size, dtype)

    out = m1_verify_qmv.m1_qmm(x, w, scales, biases, bits=bits, group_size=group_size)
    ref = mx.quantized_matmul(
        x.astype(mx.float32),
        w,
        scales.astype(mx.float32),
        biases.astype(mx.float32),
        transpose=True,
        group_size=group_size,
        bits=bits,
    )
    assert out.shape == (rows, N) and out.dtype == dtype
    err = mx.max(mx.abs(out.astype(mx.float32) - ref)).item()
    assert err <= 5e-3 * mx.max(mx.abs(ref)).item()


def test_row_and_shape_gating(qmv_only):
    f16 = mx.float16
    assert not m1_verify_qmv.eligible(1, 2048, 1024, 4, 64, f16)
    assert m1_verify_qmv.eligible(4, 2048, 1024, 4, 64, f16)
    assert not m1_verify_qmv.eligible(5, 2048, 1024, 4, 64, f16)
    assert m1_verify_qmv.eligible(3, 2048, 1024, 5, 64, f16)
    assert not m1_verify_qmv.eligible(4, 2048, 1024, 5, 64, f16)
    # unsupported bit width, small N, K off the lane block, fp32 rows
    assert not m1_verify_qmv.eligible(2, 2048, 1024, 8, 64, f16)
    assert not m1_verify_qmv.eligible(2, 2048, 512, 4, 64, f16)
    assert not m1_verify_qmv.eligible(2, 2048 + 64, 1024, 4, 64, f16)
    assert not m1_verify_qmv.eligible(2, 2048, 1024, 4, 64, mx.float32)


def test_qmv_disabled_when_the_gate_is_off(qmv_off):
    assert not m1_verify_qmv.eligible(2, 2048, 1024, 4, 64, mx.float16)
    assert qwen35_verify_qmm._verify_route(4, 5120, 17408, 4, 64, mx.float16) == "sg8"


def test_takes_priority_over_sg8_and_split_k(qmv_only):
    f16 = mx.float16
    assert qwen35_verify_qmm._verify_route(4, 5120, 17408, 4, 64, f16) == "m1"
    assert qwen35_verify_qmm._verify_route(3, 5120, 17408, 4, 64, f16) == "m1"
    # sg8 must decline so the fused swiglu path falls back per linear
    assert not qwen35_verify_qmm.sg8_eligible(4, 5120, 17408, 4, 64, f16)
    # past the per-bit row cap the previous routes are untouched
    assert qwen35_verify_qmm._verify_route(4, 5120, 17408, 5, 64, f16) == "sg8"
    assert qwen35_verify_qmm._verify_route(6, 5120, 17408, 4, 64, f16) == "sg8"


def test_armed_quantized_linear_uses_m1_route(qmv_only):
    assert qwen35_verify_qmm.apply_verify_qmm_patch()
    linear = nn.QuantizedLinear(2048, 1024, bias=False, group_size=64, bits=4)
    linear.weight, linear.scales, linear.biases = _quantized(
        2048, 1024, 4, 64, mx.float16
    )
    x = mx.random.normal((1, 3, 2048)).astype(mx.float16)
    stock = linear(x)

    calls = []
    real = m1_verify_qmv.m1_qmm

    def spy(*args, **kwargs):
        calls.append(args[0].shape)
        return real(*args, **kwargs)

    m1_verify_qmv.m1_qmm = spy
    try:
        qwen35_verify_qmm.set_verify_qmm_armed(True)
        routed = linear(x)
    finally:
        qwen35_verify_qmm.set_verify_qmm_armed(False)
        m1_verify_qmv.m1_qmm = real
    assert calls == [(3, 2048)]
    assert routed.shape == stock.shape
    assert mx.allclose(routed, stock, atol=2e-2, rtol=2e-2).item()


# --- 5-16 row MMA kernel --------------------------------------------------


@pytest.fixture
def mma_on(monkeypatch):
    monkeypatch.setattr(m1_verify_mma, "enabled", lambda: True)
    monkeypatch.setattr(m1_verify_qmv, "enabled", lambda: True)


@pytest.mark.parametrize("rows", [5, 7, 8, 9, 13, 16])
@pytest.mark.parametrize("n", [1024, 4096])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_mma_matches_fp32_reference(mma_on, rows, n, dtype):
    k = 2048
    mx.random.seed(rows + n)
    w, scales, biases = mx.quantize(
        (mx.random.normal((n, k)) * 0.05).astype(dtype), group_size=64, bits=4
    )
    x = mx.random.normal((rows, k)).astype(dtype)
    assert m1_verify_mma.eligible(rows, k, n, 4, 64, dtype)
    out = m1_verify_mma.m1_mma_qmm(x, w, scales, biases)
    ref = mx.quantized_matmul(
        x.astype(mx.float32),
        w,
        scales.astype(mx.float32),
        biases.astype(mx.float32),
        transpose=True,
        group_size=64,
        bits=4,
    )
    assert out.shape == (rows, n) and out.dtype == dtype
    err = mx.max(mx.abs(out.astype(mx.float32) - ref)).item()
    assert err <= 5e-3 * mx.max(mx.abs(ref)).item()


def test_window_and_route_order(mma_on):
    f16 = mx.float16
    assert not m1_verify_mma.eligible(4, 2048, 4096, 4, 64, f16)
    assert not m1_verify_mma.eligible(17, 2048, 4096, 4, 64, f16)
    assert not m1_verify_mma.eligible(8, 2048, 4096, 5, 64, f16)
    assert not m1_verify_mma.eligible(8, 2048, 512, 4, 64, f16)
    route = qwen35_verify_qmm._verify_route
    assert route(3, 5120, 17408, 4, 64, f16) == "m1"
    assert route(8, 5120, 17408, 4, 64, f16) == "m1_mma"
    assert route(12, 5120, 17408, 4, 64, f16) == "m1_mma"
    # 5-bit beyond the qmv window keeps the previous kernels
    assert route(8, 5120, 17408, 5, 64, f16) == "sg8"


def test_disabled_when_the_gate_is_off(monkeypatch):
    monkeypatch.setattr(m1_verify_mma, "enabled", lambda: False)
    assert not m1_verify_mma.eligible(8, 2048, 4096, 4, 64, mx.float16)
