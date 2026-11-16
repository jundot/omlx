# SPDX-License-Identifier: Apache-2.0
"""Tests for the packed 4-bit projections' verify-width (skinny) kernel.

``packed_matmul`` sends 9..48 rows to the skinny kernel.
It must match a float32 reference as closely as stock ``quantized_matmul``,
give every row the same bits whatever the row count in that range, serve
grouped projections through tile-offset views, and leave 1..8 rows and wider
calls on their kernels.
"""

from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches import qwen35_packed_linear as pl


def _on_nax() -> bool:
    if not mx.metal.is_available():
        return False
    try:
        from omlx.custom_kernels.nax import is_nax_available

        return bool(is_nax_available())
    except Exception:  # noqa: BLE001
        return False


needs_nax = pytest.mark.skipif(not _on_nax(), reason="needs an M5 (NAX) GPU")


def _linear(K: int, N: int, seed: int) -> nn.QuantizedLinear:
    layer = nn.QuantizedLinear(K, N, bias=False, group_size=64, bits=4)
    w = (mx.random.normal((N, K), key=mx.random.key(seed)) * 0.02).astype(mx.bfloat16)
    layer.weight, scales, biases = mx.quantize(w, group_size=64, bits=4)
    layer.scales = scales.astype(mx.bfloat16)
    layer.biases = biases.astype(mx.bfloat16)
    mx.eval(layer.parameters())
    return layer


def _rows(M: int, K: int, seed: int) -> mx.array:
    return (mx.random.normal((M, K), key=mx.random.key(seed)) * 0.1).astype(mx.bfloat16)


def _max_rel(y: mx.array, ref: mx.array) -> float:
    return (mx.abs(y.astype(mx.float32) - ref).max() / mx.abs(ref).max()).item()


@needs_nax
@pytest.mark.parametrize("K,N", [(1024, 384), (2048, 1280), (512, 25600)])
@pytest.mark.parametrize("M", [9, 16, 17, 33, 47, 48])
def test_skinny_matches_float32_reference_like_stock(K, N, M):
    layer = _linear(K, N, seed=K + N)
    w, s, b = layer.weight, layer.scales, layer.biases
    packed = pl._pack([layer])[0]
    x = _rows(M, K, seed=M)
    truth = (
        x.astype(mx.float32)
        @ mx.dequantize(w, s, b, group_size=64, bits=4).astype(mx.float32).T
    )
    stock = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=4)
    y = packed(x)
    assert y.shape == (M, N) and y.dtype == mx.bfloat16
    assert _max_rel(y, truth) <= 1.25 * _max_rel(stock, truth) + 1e-4


@needs_nax
@pytest.mark.parametrize("K,N", [(1024, 384), (512, 25600)])
def test_skinny_rows_do_not_depend_on_row_count(K, N):
    layer = _linear(K, N, seed=3)
    packed = pl._pack([layer])[0]
    top = pl._SKINNY_MAX_ROWS
    x = _rows(top, K, seed=4)
    full = packed(x)
    for M in (9, 15, 16, 17, 31, 33, top - 1):
        assert mx.array_equal(packed(x[:M]), full[:M]).item(), M


@needs_nax
def test_grouped_projection_views_match_each_layer():
    K = 1024
    layers = [_linear(K, n, seed=10 + i) for i, n in enumerate((768, 128, 128))]
    refs = [(lin.weight, lin.scales, lin.biases) for lin in layers]
    packed = pl._pack(layers)
    x = _rows(24, K, seed=7).reshape(1, 24, K)
    grouped = pl.project(packed, x)
    for (w, s, b), g, p in zip(refs, grouped, packed):
        ref = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=4)
        one = p(x)
        assert mx.array_equal(g, one).item()
        assert _max_rel(one, ref.astype(mx.float32)) < 2e-2


@needs_nax
def test_dispatch_keeps_small_and_wide_calls_off_the_skinny_kernel(monkeypatch):
    calls = []
    original = pl._skinny

    def spy(x2, *args):
        calls.append(int(x2.shape[0]))
        return original(x2, *args)

    monkeypatch.setattr(pl, "_skinny", spy)
    K, N = 512, 25600
    packed = pl._pack([_linear(K, N, seed=5)])[0]
    for M in (1, 2, 8, 9, 48, 49, 128):
        mx.eval(packed(_rows(M, K, seed=M)))
    assert calls == [9, 48]


def test_skinny_split_follows_the_core_budget(monkeypatch):
    pl._skinny_split.cache_clear()
    monkeypatch.setattr(pl, "_gpu_cores", lambda: 80)
    try:
        assert pl._skinny_split(5120) == 8  # 160 threadgroups
        assert pl._skinny_split(16384) == 2  # 512 threadgroups
        assert pl._skinny_split(248320) == 1
        pl._skinny_split.cache_clear()
        monkeypatch.setattr(pl, "_gpu_cores", lambda: 40)
        assert pl._skinny_split(5120) == 4
        for N in (5120, 10240, 34816):
            split = pl._skinny_split(N)
            assert split in (1, 2, 4, 8)
            assert split == 1 or (N // 32) * split <= 16 * 40
    finally:
        pl._skinny_split.cache_clear()


def test_gpu_cores_fallback_warns_once(monkeypatch, caplog):
    from omlx.utils import hardware

    pl._gpu_cores.cache_clear()
    monkeypatch.setattr(hardware, "get_gpu_core_count", lambda: None)
    try:
        with caplog.at_level(logging.WARNING, logger=pl.__name__):
            assert pl._gpu_cores() == pl._FALLBACK_GPU_CORES
            assert pl._gpu_cores() == pl._FALLBACK_GPU_CORES
        warnings = [r for r in caplog.records if r.name == pl.__name__]
        assert len(warnings) == 1 and "core count" in warnings[0].getMessage()
        pl._gpu_cores.cache_clear()
        monkeypatch.setattr(hardware, "get_gpu_core_count", lambda: 20)
        assert pl._gpu_cores() == 20
    finally:
        pl._gpu_cores.cache_clear()
