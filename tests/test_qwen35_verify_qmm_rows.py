# SPDX-License-Identifier: Apache-2.0
"""Width-stable verify: a row's projection must not depend on the block width.

The MTP depth controller changes the block width between cycles. If a
position's logits depend on that width, greedy output changes from run to run
whenever draft acceptance does.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches import qwen35_verify_qmm as vq

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


def _linear(k: int, n: int) -> nn.QuantizedLinear:
    lin = nn.QuantizedLinear(k, n, bias=False, group_size=64, bits=4)
    w = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    lin.weight, lin.scales, lin.biases = mx.quantize(w, group_size=64, bits=4)
    return lin


# Shapes reaching each route: vk/sg8 projections and the msg lm_head kernel.
@pytest.mark.parametrize(
    "k,n", [(1024, 16384), (5120, 17408), (17408, 5120), (1024, 100352)]
)
def test_row_result_is_independent_of_verify_width(k, n):
    assert vq.apply_verify_qmm_patch()
    mx.random.seed(0)
    lin = _linear(k, n)
    x = mx.random.normal((1, 6, k)).astype(mx.bfloat16)
    vq.set_verify_qmm_armed(True)
    vq.set_verify_width_stable(True)
    try:
        outs = {w: lin(x[:, :w]) for w in range(2, 7)}
        mx.eval(outs)
    finally:
        vq.set_verify_width_stable(False)
        vq.set_verify_qmm_armed(False)
    for w in range(2, 7):
        for row in range(w):
            assert mx.array_equal(outs[w][:, row], outs[6][:, row]).item(), (w, row)


def test_sampled_verify_keeps_width_specific_routing():
    """Without width-stable verify, a depth-1 block still takes the stock path."""
    assert vq.apply_verify_qmm_patch()
    mx.random.seed(0)
    lin = _linear(1024, 16384)
    x = mx.random.normal((1, 2, 1024)).astype(mx.bfloat16)
    stock = lin(x)
    vq.set_verify_qmm_armed(True)
    try:
        routed = lin(x)
        mx.eval(stock, routed)
    finally:
        vq.set_verify_qmm_armed(False)
    assert mx.array_equal(stock, routed).item()


@pytest.mark.parametrize("temp,armed", [(0.0, True), (0.7, False)])
def test_singleton_step_arms_width_stable_only_for_greedy(monkeypatch, temp, armed):
    from types import SimpleNamespace

    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    gen_batch = SimpleNamespace(samplers=[SimpleNamespace(temp=temp)])
    monkeypatch.setattr(bg, "_resolve_sampler", lambda gb: gb.samplers[0])
    with bg._width_stable_verify(gen_batch):
        seen = vq.is_width_stable_armed()
    assert seen is armed
    assert not vq.is_width_stable_armed()
