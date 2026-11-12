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


def _fresh_seams(monkeypatch):
    from omlx.patches import qwen35_padded_row_attention as mod
    from omlx.patches import qwen35_verify_sdpa_split as split

    for name in (
        "_qwen3_5_left_padded_attention",
        "_create_qwen3_5_attention_mask",
        "_set_qwen3_5_decode_left_padding",
    ):
        monkeypatch.setattr(q35, name, getattr(q35, name))
    monkeypatch.setattr(mod, "_PATCHED", False)
    monkeypatch.setattr(split, "_PATCHED", False, raising=False)
    monkeypatch.setattr(q35, "_omlx_verify_attention", False, raising=False)
    return mod, split


@pytest.fixture
def patched(monkeypatch):
    mod, split = _fresh_seams(monkeypatch)
    # The seams a served Qwen3.5 model already has: the bounded padded-verify
    # kernel (MTP runtime) and the batch-1 verify split.
    from omlx.patches.mlx_vlm_mtp import qwen35_verify_attention

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


def test_long_verify_rows_are_claimed_when_another_model_started_first(monkeypatch):
    # A non-Qwen VLM start installs the verify split and this patch before a
    # Qwen3.5 load applies the bounded verify kernel. Without the bounded
    # kernel inside both wrappers it returns None for these rows.
    from omlx.patches.mlx_vlm_mtp import qwen35_verify_attention

    mod, split = _fresh_seams(monkeypatch)
    split.apply_qwen35_verify_sdpa_split_patch()
    assert mod.apply_qwen35_padded_row_attention_patch()
    qwen35_verify_attention.apply(q35)
    cache, q, k, v, mask, _ = _step([SHORT + 3000, LONG], 4)
    assert _call(q, k, v, cache, mask) is not None


def test_short_verify_keeps_the_bounded_kernel(patched):
    cache, q, k, v, mask, _ = _step([SHORT, 4000], 4)
    _call(q, k, v, cache, mask)
    assert cache in patched.calls


def test_stale_verify_rows_are_ignored(patched):
    cache, q, k, v, mask, _ = _step([SHORT + 3000, LONG], 4)
    pads, idx, length, shape = getattr(cache, patched.mod._ROWS_ATTR)
    setattr(cache, patched.mod._ROWS_ATTR, (pads, idx - 1, length, shape))
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



def _padded_causal(cache, length):
    """The mask a padded batch should have: left padding plus causal."""
    from mlx_lm.models.base import create_causal_mask

    return create_causal_mask(
        length, offset=cache._idx - length, left_padding=cache.left_padding
    )


def _reference(queries, keys, values, mask):
    """Plain float32 attention over the whole padded batch with its mask."""
    q = queries.astype(mx.float32)
    k = mx.repeat(keys.astype(mx.float32), Q_HEADS // KV_HEADS, axis=1)
    v = mx.repeat(values.astype(mx.float32), Q_HEADS // KV_HEADS, axis=1)
    scores = mx.where(mask, (q @ k.transpose(0, 1, 3, 2)) * SCALE, -mx.inf)
    return mx.softmax(scores, axis=-1) @ v


@pytest.mark.parametrize("length", [1, 4])
def test_rows_match_an_independent_masked_reference(patched, length):
    # Unlike _solo, this shares no slicing with the patch.
    cache, q, k, v, mask, _ = _step([SHORT + 3000, LONG], length)
    out = _call(q, k, v, cache, mask)
    assert out is not None and cache not in patched.calls
    ref = _reference(q, k, v, _padded_causal(cache, length))
    assert mx.allclose(out.astype(mx.float32), ref, atol=2e-2, rtol=2e-2).item()


def test_forwards_wider_than_a_verify_block_keep_their_path(patched):
    # A prefill chunk through the verifier is not a verify block.
    cache, q, k, v, mask, _ = _step([SHORT + 3000, LONG], patched.mod._MAX_VERIFY_WIDTH + 7)
    assert not hasattr(cache, patched.mod._ROWS_ATTR)
    _call(q, k, v, cache, mask)
    assert cache in patched.calls


def test_a_mask_that_hides_more_than_padding_is_not_claimed(patched, monkeypatch):
    from mlx_lm.models.base import create_causal_mask

    def stricter(h, cache):
        # Same shape as the padded causal mask, but row 1 may not see its
        # first real key: dropping the mask per row would change its output.
        mask = create_causal_mask(
            h.shape[1], offset=cache._idx, left_padding=cache.left_padding
        )
        first = int(cache.left_padding[1].item())
        return mask & ~(mx.arange(mask.shape[-1]) == first)[None, None, None, :]

    # Re-apply the hook over a mask creator that adds that restriction.
    monkeypatch.setattr(patched.mod, "_PATCHED", False)
    monkeypatch.setattr(q35, "_create_qwen3_5_attention_mask", stricter)
    monkeypatch.setattr(q35, "_qwen3_5_left_padded_attention", patched.original)
    assert patched.mod.apply_qwen35_padded_row_attention_patch()
    cache, *_ = _step([SHORT + 3000, LONG], 4)
    assert not hasattr(cache, patched.mod._ROWS_ATTR)
