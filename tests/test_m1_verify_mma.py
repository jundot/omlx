# SPDX-License-Identifier: Apache-2.0
"""M1-class verify MMA kernel (5-16 rows, 4-bit): numerics and routing."""

from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.patches import m1_verify_mma, m1_verify_qmv, qwen35_verify_qmm

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


@pytest.fixture
def force_enabled(monkeypatch):
    monkeypatch.setenv("OMLX_M1_VERIFY_MMA", "1")
    monkeypatch.setenv("OMLX_M1_VERIFY_QMV", "1")
    m1_verify_mma.enabled.cache_clear()
    m1_verify_qmv.enabled.cache_clear()
    yield
    m1_verify_mma.enabled.cache_clear()
    m1_verify_qmv.enabled.cache_clear()


@pytest.mark.parametrize("rows", [5, 7, 8, 9, 13, 16])
@pytest.mark.parametrize("n", [1024, 4096])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_matches_fp32_reference(force_enabled, rows, n, dtype):
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


def test_window_and_route_order(force_enabled):
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


def test_disabled_by_env(monkeypatch):
    monkeypatch.setenv("OMLX_M1_VERIFY_MMA", "0")
    m1_verify_mma.enabled.cache_clear()
    try:
        assert not m1_verify_mma.eligible(8, 2048, 4096, 4, 64, mx.float16)
    finally:
        m1_verify_mma.enabled.cache_clear()
