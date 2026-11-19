# SPDX-License-Identifier: Apache-2.0
"""Tests for the skinny (9..48 row) kernel on MLX-layout weights and its verify route."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.custom_kernels import nax_skinny_qmm as nsq
from omlx.custom_kernels.nax import is_nax_available
from omlx.patches import qwen35_verify_qmm as vq

needs_nax = pytest.mark.skipif(not is_nax_available(), reason="needs an M5 (NAX) GPU")


def _weights(K: int, N: int, bits: int, dtype, seed: int):
    w = (mx.random.normal((N, K), key=mx.random.key(seed)) * 0.02).astype(dtype)
    wq, scales, biases = mx.quantize(w, group_size=64, bits=bits)
    return wq, scales.astype(dtype), biases.astype(dtype)


def _rows(M: int, K: int, dtype, seed: int) -> mx.array:
    return (mx.random.normal((M, K), key=mx.random.key(seed)) * 0.1).astype(dtype)


def _max_rel(y: mx.array, ref: mx.array) -> float:
    return (mx.abs(y.astype(mx.float32) - ref).max() / mx.abs(ref).max()).item()


@needs_nax
@pytest.mark.parametrize("bits", nsq.BITS)
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_matches_reference_at_every_row_count(bits, dtype):
    K, N = 1024, 384
    wq, s, b = _weights(K, N, bits, dtype, seed=bits)
    x = _rows(nsq.MAX_ROWS, K, dtype, seed=4)
    truth = x.astype(mx.float32) @ (
        mx.dequantize(wq, s, b, group_size=64, bits=bits).astype(mx.float32).T
    )
    stock = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=bits)
    full = nsq.skinny_qmm(x, wq, s, b, K, N, bits=bits)
    assert full.shape == (nsq.MAX_ROWS, N) and full.dtype == dtype
    assert _max_rel(full, truth) <= 1.25 * _max_rel(stock, truth) + 1e-4
    # Rows keep their bits at every row count, across 16-row fragment edges.
    for M in (9, 16, 17, 33):
        part = nsq.skinny_qmm(x[:M], wq, s, b, K, N, bits=bits)
        assert mx.array_equal(part, full[:M]).item(), M


def test_route_takes_only_supported_9_to_48_row_calls(monkeypatch):
    monkeypatch.setattr(vq, "is_nax_available", lambda: True)
    bf16 = mx.bfloat16
    for bits in nsq.BITS:
        for M in (9, 48):
            assert vq._verify_route(M, 6144, 5120, bits, 64, bf16) == "skinny"
    for M in (8, 49):
        assert vq._verify_route(M, 6144, 5120, 8, 64, bf16) != "skinny"
    # 9..16 rows replace mma only; narrower outputs keep stock's exact-code path.
    assert not vq.skinny_eligible(16, 4096, 1024, 4, 64, bf16)
    assert vq.skinny_eligible(17, 4096, 1024, 4, 64, bf16)
    for args in (
        (17, 4096, 512, 4, 64, bf16),
        (17, 6144, 5120, 3, 64, bf16),
        (17, 6144, 5120, 5, 128, bf16),
        (17, 6144, 5120, 5, 64, mx.float32),
    ):
        assert not vq.skinny_eligible(*args), args
    monkeypatch.setattr(vq, "is_nax_available", lambda: False)
    assert not vq.skinny_eligible(17, 6144, 5120, 5, 64, bf16)


@needs_nax
def test_armed_quantized_linear_takes_the_skinny_route(monkeypatch):
    assert vq.apply_verify_qmm_patch()
    K, N = 1024, 1024
    linear = nn.QuantizedLinear(K, N, bias=False, group_size=64, bits=5)
    linear.weight, linear.scales, linear.biases = _weights(K, N, 5, mx.bfloat16, seed=7)
    x = _rows(34, K, mx.bfloat16, seed=8).reshape(2, 17, K)
    stock = linear(x)
    calls = []
    real = nsq.skinny_qmm

    def spy(x2, *args, **kwargs):
        calls.append(int(x2.shape[0]))
        return real(x2, *args, **kwargs)

    monkeypatch.setattr(nsq, "skinny_qmm", spy)
    vq.set_verify_qmm_armed(True)
    try:
        routed = linear(x)
        linear(x.astype(mx.float16))  # scales in another dtype: the route declines
    finally:
        vq.set_verify_qmm_armed(False)
    assert calls == [34]
    assert mx.allclose(routed, stock, atol=2e-2, rtol=2e-2).item()
