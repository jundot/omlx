# SPDX-License-Identifier: Apache-2.0
"""On a tensor-unit GPU a single verify projection of four or five rows runs on stock qmv."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches import m1_verify_mma, m1_verify_qmv
from omlx.patches import qwen35_verify_qmm as vq

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

K, N = 6144, 5120


@pytest.fixture
def armed(monkeypatch):
    # Keep the M1 windows out of the way so only the tensor-unit floor is tested.
    monkeypatch.setattr(m1_verify_qmv, "enabled", lambda: False)
    monkeypatch.setattr(m1_verify_mma, "enabled", lambda: False)
    assert vq.apply_verify_qmm_patch()
    vq.set_verify_qmm_armed(True)
    try:
        yield
    finally:
        vq.set_verify_qmm_armed(False)


def _linear(k, n, bits):
    mx.random.seed(3)
    layer = nn.QuantizedLinear(k, n, bias=False, group_size=64, bits=bits)
    layer.weight, layer.scales, layer.biases = mx.quantize(
        (mx.random.normal((n, k)) * 0.05).astype(mx.bfloat16), group_size=64, bits=bits
    )
    return layer


@pytest.mark.parametrize("bits", [4, 5])
@pytest.mark.parametrize(
    "rows,nax,route",
    [(4, True, None), (5, True, None), (6, True, "sg8"), (4, False, "sg8"), (5, False, "sg8")],
)
def test_single_projection_route_by_rows(monkeypatch, armed, bits, rows, nax, route):
    monkeypatch.setattr(vq, "is_nax_available", lambda: nax)
    assert vq._verify_route(rows, K, N, bits, 64, mx.bfloat16) == route


@pytest.mark.parametrize("bits", [4, 5])
@pytest.mark.parametrize("rows", [4, 5])
def test_four_and_five_rows_match_the_stock_call(monkeypatch, bits, rows):
    monkeypatch.setattr(vq, "is_nax_available", lambda: True)
    monkeypatch.setattr(m1_verify_qmv, "enabled", lambda: False)
    monkeypatch.setattr(m1_verify_mma, "enabled", lambda: False)
    assert vq.apply_verify_qmm_patch()
    layer = _linear(K, N, bits)
    x = mx.random.normal((1, rows, K)).astype(mx.bfloat16)
    expected = layer(x)  # disarmed: stock qmv
    vq.set_verify_qmm_armed(True)
    try:
        routed = layer(x)
    finally:
        vq.set_verify_qmm_armed(False)
    assert mx.array_equal(routed, expected).item()


@pytest.mark.parametrize("rows", [4, 5])
def test_fused_swiglu_and_grouped_launches_stay_on_sg8(monkeypatch, armed, rows):
    monkeypatch.setattr(vq, "is_nax_available", lambda: True)
    gate, up = _linear(5120, 6144, 4), _linear(5120, 6144, 4)
    x = mx.random.normal((1, rows, 5120)).astype(mx.bfloat16)
    assert vq.sg8_swiglu_eligible(gate, up, x)
    assert vq.sg8_group_eligible([gate, up], x)
