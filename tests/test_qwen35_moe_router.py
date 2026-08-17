# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the fused Qwen MoE router top-k.

The routed expert SET must match the composed chain exactly — including
on equal rounded probabilities, where mlx's ``argpartition`` keeps the
HIGHEST index (pinned here so an mlx behavior change fails loudly).
Scores may differ from the composed sum/divide by reduction-order ulp.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.qwen35_moe_router import fused_router_topk, router_eligible

K = 8
NE = 256


def _composed(p):
    inds = mx.argpartition(p, kth=-K, axis=-1)[..., -K:]
    scores = mx.take_along_axis(p, inds, axis=-1)
    scores = scores / scores.sum(axis=-1, keepdims=True)
    return inds, scores


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_selected_set_matches_composed():
    mx.random.seed(3)
    for _ in range(300):
        g = (mx.random.normal((1, 1, NE)) * 2.0).astype(mx.bfloat16)
        p = mx.softmax(g, axis=-1, precise=True)
        ci, cs = _composed(p)
        fi, fs = fused_router_topk(p, K)
        assert sorted(ci[0, 0].tolist()) == sorted(fi[0, 0].tolist())
        cmap = dict(zip(ci[0, 0].tolist(), cs[0, 0].astype(mx.float32).tolist()))
        fmap = dict(zip(fi[0, 0].tolist(), fs[0, 0].astype(mx.float32).tolist()))
        for idx, ref in cmap.items():
            assert abs(ref - fmap[idx]) <= max(2e-2 * ref, 1e-4)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_tie_break_matches_argpartition_highest_index():
    base = np.full(NE, 0.001, dtype=np.float32)
    for t in range(0, NE, 16):
        base[t] = 0.1
    for arr in (base, base[::-1].copy()):
        p = mx.array(arr, dtype=mx.bfloat16).reshape(1, 1, -1)
        ci, _ = _composed(p)
        fi, _ = fused_router_topk(p, K)
        assert sorted(ci[0, 0].tolist()) == sorted(fi[0, 0].tolist())


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_multi_row_verify_widths():
    mx.random.seed(5)
    for rows in (1, 4, 6):
        g = (mx.random.normal((1, rows, NE)) * 2.0).astype(mx.bfloat16)
        p = mx.softmax(g, axis=-1, precise=True)
        ci, _ = _composed(p)
        fi, _ = fused_router_topk(p, K)
        for r in range(rows):
            assert sorted(ci[0, r].tolist()) == sorted(fi[0, r].tolist())


def test_eligibility_gates():
    x1 = mx.zeros((1, 1, 2048), dtype=mx.bfloat16)
    assert router_eligible(x1, NE)
    xp = mx.zeros((1, 2048, 2048), dtype=mx.bfloat16)
    assert not router_eligible(xp, NE)  # prefill rows stay composed
    assert not router_eligible(x1, 250)  # NE % 32 != 0
    xf = mx.zeros((1, 1, 2048), dtype=mx.float32)
    assert not router_eligible(xf, NE)
