# SPDX-License-Identifier: Apache-2.0
"""The Inkling 2 decode kernel must match the masked reference path, including
batch caches that carry left padding."""

import mlx.core as mx
import pytest

from omlx.patches.inkling2 import kernels


def _inputs(b, s, hq=8, hkv=2, d=64, r=16, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((b, hq, 1, d)).astype(mx.bfloat16)
    k = mx.random.normal((b, hkv, s, d)).astype(mx.bfloat16)
    v = mx.random.normal((b, hkv, s, d)).astype(mx.bfloat16)
    rel = mx.random.normal((b, 1, hq, r)).astype(mx.bfloat16)
    sink = mx.random.normal((b, 1, hq)).astype(mx.bfloat16)
    return q, k, v, rel, sink


@pytest.mark.parametrize(
    "b,s,window,pad",
    [
        (1, 40, 0, None),
        (1, 40, 0, [0]),
        (1, 40, 0, [7]),
        (1, 40, 16, [0]),
        (1, 40, 16, [20]),
        (1, 40, 16, [30]),
        (3, 40, 0, [0, 5, 30]),
        (3, 40, 16, [0, 5, 30]),
        (2, 10, 16, [0, 3]),
    ],
)
def test_decode_matches_masked(b, s, window, pad, monkeypatch):
    q, k, v, rel, sink = _inputs(b, s)
    lp = None if pad is None else mx.array(pad)
    scale = 64**-0.5
    want = kernels._masked_attention(q, k, v, rel, sink, scale, window, lp)

    def no_masked(*a, **kw):
        raise AssertionError("single-token decode took the masked path")

    # Batch caches always pass left_padding; decode must still use the
    # windowed decode kernel instead of masking every cached key.
    monkeypatch.setattr(kernels, "_masked_attention", no_masked)
    got = kernels.banded_sink_attention(
        q, k, v, rel, sink, scale=scale, window=window, left_padding=lp
    )
    assert got.shape == want.shape
    assert mx.allclose(
        got.astype(mx.float32), want.astype(mx.float32), atol=2e-2
    ).item()
