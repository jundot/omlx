# SPDX-License-Identifier: Apache-2.0
"""Tests for the LFM3 one-launch RoPE patch."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
lfm3 = pytest.importorskip("mlx_vlm.models.lfm3.language")

from omlx.patches.lfm3_rope import apply_lfm3_rope_patch  # noqa: E402

_STOCK_ROPE = lfm3.RotaryEmbedding.__call__


@pytest.fixture
def patched():
    assert apply_lfm3_rope_patch()
    yield
    cls = lfm3.RotaryEmbedding
    cls.__call__ = _STOCK_ROPE
    del cls._omlx_lfm3_fast_rope, cls._omlx_lfm3_original_call


@pytest.mark.parametrize(
    "params", [{"rope_theta": 1e6, "partial_rotary_factor": 0.5}, {"rope_theta": 1e4}]
)
@pytest.mark.parametrize(
    "shape,offset",
    [((1, 4, 1, 128), 37), ((1, 4, 9, 128), 0), ((3, 4, 1, 128), "array")],
)
def test_fast_rope_matches_stock(patched, params, shape, offset):
    rope = lfm3.RotaryEmbedding(128, params)
    if offset == "array":
        # Per-row offsets of a batch cache.
        offset = mx.array([3, 500, 41])
    x = mx.random.normal(shape, key=mx.random.key(1))
    ref = _STOCK_ROPE(rope, x, offset)
    out = rope(x, offset)
    assert mx.allclose(out, ref, atol=1e-4, rtol=1e-4).item()
