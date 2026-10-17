# SPDX-License-Identifier: Apache-2.0
"""M1-class multi-row verify qmv: numerics, gating and verify routing."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches import m1_verify_mma, m1_verify_qmv, qwen35_verify_qmm

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


@pytest.fixture
def force_enabled(monkeypatch):
    monkeypatch.setenv("OMLX_M1_VERIFY_QMV", "1")
    # Pin the 5-16 row MMA route off so the sg8/qmv boundaries stay visible
    # (tests/test_m1_verify_mma.py covers it).
    monkeypatch.setenv("OMLX_M1_VERIFY_MMA", "0")
    m1_verify_qmv.enabled.cache_clear()
    m1_verify_mma.enabled.cache_clear()
    yield
    m1_verify_qmv.enabled.cache_clear()
    m1_verify_mma.enabled.cache_clear()
    qwen35_verify_qmm.set_verify_qmm_armed(False)


@pytest.fixture
def force_disabled(monkeypatch):
    monkeypatch.setenv("OMLX_M1_VERIFY_QMV", "0")
    m1_verify_qmv.enabled.cache_clear()
    yield
    m1_verify_qmv.enabled.cache_clear()


def _quantized(k, n, bits, group_size, dtype, seed=0):
    mx.random.seed(seed)
    weight = (mx.random.normal((n, k)) * 0.05).astype(dtype)
    return mx.quantize(weight, group_size=group_size, bits=bits)


@pytest.mark.parametrize("bits,rows", [(4, 2), (4, 3), (4, 4), (5, 2), (5, 3)])
@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_matches_fp32_reference(force_enabled, bits, rows, group_size, dtype):
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


def test_row_and_shape_gating(force_enabled):
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


def test_disabled_by_env(force_disabled):
    assert not m1_verify_qmv.eligible(2, 2048, 1024, 4, 64, mx.float16)
    assert qwen35_verify_qmm._verify_route(4, 5120, 17408, 4, 64, mx.float16) == "sg8"


def test_takes_priority_over_sg8_and_split_k(force_enabled):
    f16 = mx.float16
    assert qwen35_verify_qmm._verify_route(4, 5120, 17408, 4, 64, f16) == "m1"
    assert qwen35_verify_qmm._verify_route(3, 5120, 17408, 4, 64, f16) == "m1"
    # sg8 must decline so the fused swiglu path falls back per linear
    assert not qwen35_verify_qmm.sg8_eligible(4, 5120, 17408, 4, 64, f16)
    # past the per-bit row cap the previous routes are untouched
    assert qwen35_verify_qmm._verify_route(4, 5120, 17408, 5, 64, f16) == "sg8"
    assert qwen35_verify_qmm._verify_route(6, 5120, 17408, 4, 64, f16) == "sg8"


def test_armed_quantized_linear_uses_m1_route(force_enabled):
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
