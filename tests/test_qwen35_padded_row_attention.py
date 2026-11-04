# SPDX-License-Identifier: Apache-2.0
"""Padded Qwen3.5 batches served per row by the batch-1 attention routes."""

from __future__ import annotations

import types

import mlx.core as mx
import pytest

q35 = pytest.importorskip("mlx_vlm.models.qwen3_5.language")
from mlx_lm.models.cache import BatchKVCache  # noqa: E402

DTYPE = mx.bfloat16
Q_HEADS, KV_HEADS, D = 24, 4, 256
SCALE = D**-0.5
# Below 1024 keys every GPU uses the one-pass plan; above 8192 every GPU uses
# a two-pass plan, so these rows never share a ragged-kernel plan.
SHORT, LONG = 600, 9000


@pytest.fixture
def patched(monkeypatch):
    from omlx.patches import qwen35_padded_row_attention as mod

    for name in (
        "_qwen3_5_left_padded_attention",
        "_create_qwen3_5_attention_mask",
        "_set_qwen3_5_decode_left_padding",
    ):
        monkeypatch.setattr(q35, name, getattr(q35, name))
    monkeypatch.setattr(mod, "_PATCHED", False)
    # The seams a served Qwen3.5 model already has: the bounded padded-verify
    # kernel (MTP runtime) and the batch-1 verify split.
    from omlx.patches import qwen35_verify_sdpa_split as split
    from omlx.patches.mlx_vlm_mtp import qwen35_verify_attention

    monkeypatch.setattr(split, "_PATCHED", False, raising=False)
    monkeypatch.setattr(q35, "_omlx_verify_attention", False, raising=False)
    qwen35_verify_attention.apply(q35)
    split.apply_qwen35_verify_sdpa_split_patch()
    original = q35._qwen3_5_left_padded_attention
    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs.get("cache"))
        return original(*args, **kwargs)

    monkeypatch.setattr(q35, "_qwen3_5_left_padded_attention", spy)
    assert mod.apply_qwen35_padded_row_attention_patch()
    return types.SimpleNamespace(mod=mod, original=original, calls=calls)


def _layers():
    # Two full-attention layers around one linear layer, like the hybrid stack.
    return [
        types.SimpleNamespace(is_linear=False),
        types.SimpleNamespace(is_linear=True),
        types.SimpleNamespace(is_linear=False),
    ]


def _step(lengths, length, seed=0):
    """Caches for two full-attention layers, filled to len - length, then one step."""
    mx.random.seed(seed)
    total = max(lengths)
    caches = [BatchKVCache([total - n for n in lengths]), None, BatchKVCache([total - n for n in lengths])]
    batch = len(lengths)
    for c in (caches[0], caches[2]):
        k = mx.random.normal((batch, KV_HEADS, total - length, D)).astype(DTYPE)
        v = mx.random.normal((batch, KV_HEADS, total - length, D)).astype(DTYPE)
        c.update_and_fetch(k, v)
    h = mx.zeros((batch, length, 8), dtype=DTYPE)
    mask = q35._create_qwen3_5_attention_mask(h, caches[0])
    decode_pads = (
        getattr(caches[0], "_qwen3_5_decode_left_padding", None)
        if isinstance(mask, str) and mask == "left_padded_decode"
        else None
    )
    q35._set_qwen3_5_decode_left_padding(caches, _layers(), decode_pads)
    cache = caches[2]
    nk = mx.random.normal((batch, KV_HEADS, length, D)).astype(DTYPE)
    nv = mx.random.normal((batch, KV_HEADS, length, D)).astype(DTYPE)
    keys, values = cache.update_and_fetch(nk, nv)
    queries = mx.random.normal((batch, Q_HEADS, length, D)).astype(DTYPE)
    attn_mask = None if isinstance(mask, str) and mask == "left_padded_decode" else mask
    return cache, queries, keys, values, attn_mask, [total - n for n in lengths]


def _solo(original, queries, keys, values, pads):
    """What each row gets as a lone request."""
    rows = []
    for i, p in enumerate(pads):
        q, k, v = queries[i : i + 1], keys[i : i + 1, :, p:, :], values[i : i + 1, :, p:, :]
        if queries.shape[2] == 1:
            rows.append(mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE))
        else:
            out = original(q, k, v, cache=None, scale=SCALE, mask="causal")
            rows.append(_loop(q, k, v) if out is None else out)
    return mx.concatenate(rows, axis=0)


def _loop(q, k, v):
    prefix = k.shape[2] - q.shape[2]
    return mx.concatenate(
        [
            mx.fast.scaled_dot_product_attention(
                q[:, :, i : i + 1], k[:, :, : prefix + i + 1], v[:, :, : prefix + i + 1], scale=SCALE
            )
            for i in range(q.shape[2])
        ],
        axis=2,
    )


def _call(queries, keys, values, cache, mask):
    return q35._qwen3_5_left_padded_attention(
        queries, keys, values, cache=cache, scale=SCALE, mask=mask
    )


def test_decode_rows_with_different_plans_match_solo(patched):
    cache, q, k, v, mask, pads = _step([SHORT, LONG], 1)
    out = _call(q, k, v, cache, mask)
    assert out is not None
    assert mx.array_equal(out, _solo(patched.original, q, k, v, pads)).item()
    assert cache not in patched.calls  # the batched original never ran


def test_decode_rows_sharing_a_plan_keep_the_ragged_kernel(patched):
    cache, q, k, v, mask, _ = _step([LONG - 200, LONG], 1)
    _call(q, k, v, cache, mask)
    assert cache in patched.calls


@pytest.mark.parametrize("length", [2, 4, 5])
def test_long_verify_rows_match_solo(patched, length):
    cache, q, k, v, mask, pads = _step([SHORT + 3000, LONG], length)
    assert isinstance(mask, mx.array)
    out = _call(q, k, v, cache, mask)
    assert out is not None
    assert mx.array_equal(out, _solo(patched.original, q, k, v, pads)).item()
    assert cache not in patched.calls


def test_short_verify_keeps_the_bounded_kernel(patched):
    cache, q, k, v, mask, _ = _step([SHORT, 4000], 4)
    _call(q, k, v, cache, mask)
    assert cache in patched.calls


def test_stale_verify_rows_are_ignored(patched):
    cache, q, k, v, mask, _ = _step([SHORT + 3000, LONG], 4)
    pads, idx, length = getattr(cache, patched.mod._ROWS_ATTR)
    setattr(cache, patched.mod._ROWS_ATTR, (pads, idx - 1, length))
    _call(q, k, v, cache, mask)
    assert cache in patched.calls


def test_unexpected_mask_is_delegated(patched):
    cache, q, k, v, _, _ = _step([SHORT + 3000, LONG], 4)
    odd = mx.ones((1, 1, 4, k.shape[2]), dtype=mx.bool_)
    _call(q, k, v, cache, odd)
    assert cache in patched.calls


def test_rows_are_cleared_when_the_next_forward_is_not_padded(patched):
    cache, *_ = _step([SHORT + 3000, LONG], 4)
    assert hasattr(cache, patched.mod._ROWS_ATTR)
    h = mx.zeros((2, 1, 8), dtype=DTYPE)
    caches = [cache, None, cache]
    q35._create_qwen3_5_attention_mask(h, cache)
    q35._set_qwen3_5_decode_left_padding(caches, _layers(), None)
    assert not hasattr(cache, patched.mod._ROWS_ATTR)
