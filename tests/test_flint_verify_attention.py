# SPDX-License-Identifier: Apache-2.0
"""Tests for Flint's row-chunked verify attention."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
flint = pytest.importorskip("mlx_vlm.models.flint.language")

from omlx.patches import flint_verify_attention  # noqa: E402

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="Flint verify attention needs Metal"
)

SCALE = 128**-0.5


def _block(rows, kv_len, dtype, seed=0):
    """Flint's attention shapes: 32 query heads on 4 KV heads, head dim 128."""
    mx.random.seed(seed)
    q = mx.random.normal((1, 32, rows, 128)).astype(dtype)
    k = mx.random.normal((1, 4, kv_len, 128)).astype(dtype)
    v = mx.random.normal((1, 4, kv_len, 128)).astype(dtype)
    sinks = mx.random.normal((32,)).astype(dtype)
    return q, k, v, sinks


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("kv_len", [600, 4098])
@pytest.mark.parametrize("rows", [2, 4, 5, 8, 9])
def test_verify_rows_match_their_one_row_decode(dtype, kv_len, rows):
    """Each verify row equals the one-row decode call over its own keys,
    also where the rows straddle MLX's two-pass threshold (4096 keys)."""
    assert flint_verify_attention.apply_flint_verify_attention_patch()
    q, k, v, sinks = _block(rows, kv_len, dtype)
    out = flint.scaled_dot_product_attention(
        q, k, v, cache=None, scale=SCALE, mask="causal", sinks=sinks
    )
    for r in range(rows):
        end = kv_len - (rows - 1 - r)
        decode = mx.fast.scaled_dot_product_attention(
            q[:, :, r : r + 1],
            k[:, :, :end],
            v[:, :, :end],
            scale=SCALE,
            mask=None,
            sinks=sinks,
        )
        assert mx.array_equal(out[:, :, r : r + 1], decode).item(), r


def test_array_mask_matches_stock_sdpa():
    """A sliding-window keep-mask is sliced per chunk, not rebuilt."""
    assert flint_verify_attention.apply_flint_verify_attention_patch()
    rows, kv_len, window = 6, 40, 16
    q, k, v, sinks = _block(rows, kv_len, mx.float32, seed=1)
    query_pos = mx.arange(kv_len - rows, kv_len)[:, None]
    key_pos = mx.arange(kv_len)[None]
    mask = ((key_pos <= query_pos) & (key_pos > query_pos - window))[None, None]
    out = flint.scaled_dot_product_attention(
        q, k, v, cache=None, scale=SCALE, mask=mask, sinks=sinks
    )
    ref = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=SCALE, mask=mask, sinks=sinks
    )
    assert mx.allclose(out, ref, atol=1e-5, rtol=1e-5).item()


@pytest.mark.parametrize(
    ("rows", "mask"), [(32, "causal"), (4, None)], ids=["prefill", "no-mask"]
)
def test_prefill_and_unmasked_blocks_keep_stock_path(monkeypatch, rows, mask):
    assert flint_verify_attention.apply_flint_verify_attention_patch()

    def fail(*args, **kwargs):
        pytest.fail("only causal verify-width blocks take the chunked route")

    monkeypatch.setattr(flint_verify_attention, "row_exact_causal_sdpa", fail)
    q, k, v, sinks = _block(rows, 64, mx.float16)
    out = flint.scaled_dot_product_attention(
        q, k, v, cache=None, scale=SCALE, mask=mask, sinks=sinks
    )
    assert out.shape == q.shape
