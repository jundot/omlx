# SPDX-License-Identifier: Apache-2.0
"""The QSA row gather must read only the selected rows of the stored (B, H, N, D) cache."""

from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat


@pytest.fixture(autouse=True)
def _vendored_qwen4():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()


def _reference(kv, indices):
    """The previous token-major path: transpose, gather per batch (the old helper, inlined), transpose back."""
    rows = kv.transpose(0, 2, 1, 3)
    batch, tokens = rows.shape[:2]
    trailing = rows.shape[2:]
    offsets = mx.arange(batch, dtype=mx.int32).reshape((batch,) + (1,) * (indices.ndim - 1)) * tokens
    flat = (indices.astype(mx.int32) + offsets).reshape(-1)
    gathered = rows.reshape(batch * tokens, *trailing)[flat].reshape(*indices.shape, *trailing)
    axes = (0, 2, 1, 3) if indices.ndim == 2 else (0, 1, 3, 2, 4)
    return mx.contiguous(gathered.transpose(*axes))


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("form", ["_gather_kv_rows", "_gather_kv_rows_stored", "_gather_kv_rows_token_major"])
def test_gather_kv_rows_matches_token_major_gather(batch, form):
    from mlx_vlm.models.qwen4_exp import qsa_fast

    mx.random.seed(3)
    kv = mx.random.normal((batch, 2, 300, 16)).astype(mx.bfloat16)
    indices = mx.sort(mx.random.randint(0, 300, (batch, 37)).astype(mx.int32), axis=-1)
    out = getattr(qsa_fast, form)(kv, indices)
    mx.eval(out)
    assert out.shape == (batch, 2, 37, 16)
    assert mx.array_equal(out.view(mx.uint16), _reference(kv, indices).view(mx.uint16)).item()


def _dispatch(monkeypatch, per_query, tokens):
    """Which form the dispatcher picks for a (1, per_query, 2051) gather from a cache of ``tokens`` rows."""
    from mlx_vlm.models.qwen4_exp import qsa_fast

    picked = []
    for name in ("_gather_kv_rows_stored", "_gather_kv_rows_token_major"):
        monkeypatch.setattr(qsa_fast, name, lambda kv, idx, _n=name: picked.append(_n))
    kv = mx.zeros((1, 2, tokens, 16), dtype=mx.bfloat16)
    idx = mx.zeros((1, per_query, 2051), dtype=mx.int32)
    qsa_fast._gather_kv_rows(kv, idx)
    return picked


@pytest.mark.parametrize("per_query", [1, 4, 16])
@pytest.mark.parametrize("tokens", [4096, 65536, 206848])
def test_gather_kv_rows_decode_and_verify_widths_use_stored_layout(monkeypatch, per_query, tokens):
    assert _dispatch(monkeypatch, per_query, tokens) == ["_gather_kv_rows_stored"]


@pytest.mark.parametrize("tokens", [4096, 16384, 65536])
def test_gather_kv_rows_prefill_width_copies_token_major_below_threshold(monkeypatch, tokens):
    assert _dispatch(monkeypatch, 64, tokens) == ["_gather_kv_rows_token_major"]


def test_gather_kv_rows_prefill_width_uses_stored_layout_at_long_context(monkeypatch):
    assert _dispatch(monkeypatch, 64, 131072) == ["_gather_kv_rows_stored"]


def test_gather_kv_rows_rank_three_forms_agree():
    """The two forms must be bit-identical on a prefill-shaped (B, T, S) gather."""
    from mlx_vlm.models.qwen4_exp import qsa_fast

    mx.random.seed(5)
    kv = mx.random.normal((2, 2, 500, 16)).astype(mx.bfloat16)
    indices = mx.sort(mx.random.randint(0, 500, (2, 8, 21)).astype(mx.int32), axis=-1)
    stored, token_major = qsa_fast._gather_kv_rows_stored(kv, indices), qsa_fast._gather_kv_rows_token_major(kv, indices)
    mx.eval(stored, token_major)
    assert stored.shape == token_major.shape == (2, 8, 2, 21, 16)
    assert mx.array_equal(stored.view(mx.uint16), token_major.view(mx.uint16)).item()
    assert mx.array_equal(stored.view(mx.uint16), _reference(kv, indices).view(mx.uint16)).item()
