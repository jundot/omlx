# SPDX-License-Identifier: Apache-2.0
"""Deferred ragged-rollback realignment keeps every row's KV exact."""

from __future__ import annotations

import os
import random
import types

import mlx.core as mx
import pytest

q35 = pytest.importorskip("mlx_vlm.models.qwen3_5.language")
from mlx_vlm.models.cache import BatchKVCache  # noqa: E402
from mlx_vlm.speculative import cache_state  # noqa: E402

KVH, D, QH = 4, 256, 24
DTYPE = mx.bfloat16
SCALE = D**-0.5


@pytest.fixture(scope="module")
def dr():
    os.environ["OMLX_DEFER_REALIGN"] = "1"
    from omlx.patches import qwen35_deferred_realign as mod
    from omlx.patches import qwen35_padded_row_attention as pr
    from omlx.patches import qwen35_verify_sdpa_split as split
    from omlx.patches.mlx_vlm_mtp import qwen35_verify_attention

    mod.apply_qwen35_deferred_realign_patch()
    assert mod._PATCHED
    qwen35_verify_attention.apply(q35)
    split.apply_qwen35_verify_sdpa_split_patch()
    pr.apply_qwen35_padded_row_attention_patch()
    return mod


def _fill(lengths, mark, dr, seed=0):
    mx.random.seed(seed)
    total = max(lengths)
    cache = BatchKVCache([total - n for n in lengths])
    if mark:
        cache.__dict__[dr.MARK] = True
    k = mx.random.normal((len(lengths), KVH, total, D)).astype(DTYPE)
    v = mx.random.normal((len(lengths), KVH, total, D)).astype(DTYPE)
    cache.update_and_fetch(k, v)
    mx.eval(cache._keys, cache._values)
    return cache


def _rows(cache, dr):
    """Each row's valid keys/values without triggering a realignment."""
    tails = cache.__dict__.get(dr.TAIL) or [0] * int(cache.left_padding.size)
    pads = [int(p) for p in cache.left_padding.tolist()]
    return [
        (
            cache._keys[r : r + 1, :, pads[r] : cache._idx - tails[r], :],
            cache._values[r : r + 1, :, pads[r] : cache._idx - tails[r], :],
        )
        for r in range(len(pads))
    ]


def _step(caches, block, accepted, seed):
    mx.random.seed(seed)
    batch = int(caches[0].left_padding.size)
    k = mx.random.normal((batch, KVH, block, D)).astype(DTYPE)
    v = mx.random.normal((batch, KVH, block, D)).astype(DTYPE)
    for c in caches:
        c.update_and_fetch(k, v)
        # The rollback an MTP commit runs for these accepted counts.
        cache_state._trim_append_caches([c], [a + 1 for a in accepted], block)


def test_rows_stay_exact_through_ragged_cycles(dr):
    rng = random.Random(7)
    lengths = [3000, 9000]
    ref = _fill(lengths, False, dr)
    exp = _fill(lengths, True, dr)
    saw_tail = 0
    for step in range(60):
        block = rng.choice([2, 3, 4, 5])
        accepted = [rng.randrange(block) for _ in lengths]
        _step([ref, exp], block, accepted, seed=100 + step)
        assert exp._idx >= ref._idx
        tails = exp.__dict__.get(dr.TAIL)
        saw_tail += bool(tails and any(tails))
        assert not tails or max(tails) <= dr._MAX_TAIL
        assert mx.array_equal(exp.offset, ref.offset).item()
        for (rk, rv), (ek, ev) in zip(_rows(ref, dr), _rows(exp, dr)):
            assert rk.shape == ek.shape
            assert mx.array_equal(rk, ek).item() and mx.array_equal(rv, ev).item()
    assert saw_tail > 10  # the patch actually deferred
    # A foreign read realigns to exactly the stock layout.
    keys, values, offset, left = exp.state
    assert dr.tails(exp) is None
    rk, rv, roff, rleft = ref.state
    assert mx.array_equal(left, rleft).item() and mx.array_equal(offset, roff).item()
    for r, pad in enumerate(rleft.tolist()):
        assert mx.array_equal(keys[r, :, pad:], rk[r, :, pad:]).item()
        assert mx.array_equal(values[r, :, pad:], rv[r, :, pad:]).item()


def test_extract_realigns_first(dr):
    ref = _fill([3000, 9000], False, dr)
    exp = _fill([3000, 9000], True, dr)
    _step([ref, exp], 4, [3, 0], seed=5)
    assert dr.tails(exp) == [0, 3]
    a, b = ref.extract(1), exp.extract(1)
    assert dr.tails(exp) is None
    assert mx.array_equal(a.keys[..., : a.offset, :], b.keys[..., : b.offset, :]).item()


def _layers():
    return [types.SimpleNamespace(is_linear=False)]


@pytest.mark.parametrize("block", [1, 4])
def test_attention_over_tails_equals_the_aligned_cache(dr, block):
    ref = _fill([3000, 9000], False, dr)
    exp = _fill([3000, 9000], True, dr)
    _step([ref, exp], 4, [3, 1], seed=11)  # leaves exp with tails [0, 2]
    assert dr.tails(exp) == [0, 2]
    for c in (ref, exp):
        c.__dict__[dr.MARK] = True
    mx.random.seed(21)
    q = mx.random.normal((2, QH, block, D)).astype(DTYPE)
    nk = mx.random.normal((2, KVH, block, D)).astype(DTYPE)
    outs = []
    for c in (ref, exp):
        # The route mlx-vlm's speculative verifier takes: every block through
        # the attention seam, inside the verify mark its _model sets.
        with dr.verifying():
            h = mx.zeros((2, block, 8), dtype=DTYPE)
            mask = q35._create_qwen3_5_attention_mask(h, c)
            pads = (
                getattr(c, "_qwen3_5_decode_left_padding", None)
                if isinstance(mask, str) and mask == "left_padded_decode"
                else None
            )
            q35._set_qwen3_5_decode_left_padding([c], _layers(), pads)
            keys, values = c.update_and_fetch(nk, nk)
            attn_mask = None if isinstance(mask, str) else mask
            outs.append(
                q35._qwen3_5_left_padded_attention(
                    q, keys, values, cache=c, scale=SCALE, mask=attn_mask
                )
            )
    assert dr.tails(exp) == [0, 2]
    assert outs[0] is not None and outs[1] is not None
    assert mx.array_equal(outs[0], outs[1]).item()


def test_prefill_right_padding_still_rolls(dr):
    # Outside a speculative rollback, finalize keeps the stock roll.
    c = _fill([3000, 9000], True, dr)
    k = mx.random.normal((2, KVH, 4, D)).astype(DTYPE)
    c.update_and_fetch(k, k)
    c.prepare(right_padding=[2, 0])
    c.finalize()
    assert dr.tails(c) is None


def _layers3():
    return [
        types.SimpleNamespace(is_linear=False),
        types.SimpleNamespace(is_linear=True),
        types.SimpleNamespace(is_linear=False),
    ]


def test_failed_realign_keeps_the_tails(dr, monkeypatch):
    from omlx.patches import vlm_batch_kv_capacity as cap

    c = _fill([3000, 9000], True, dr)
    _step([c], 4, [3, 0], seed=3)
    assert dr.tails(c) == [0, 3]

    def boom(*args, **kwargs):
        raise RuntimeError("roll failed")

    monkeypatch.setattr(cap, "_roll_logical_prefix", boom)
    with pytest.raises(RuntimeError):
        dr.realign(c)
    assert dr.tails(c) == [0, 3]  # still described by its tails


def test_a_wide_forward_realigns_every_layer(dr):
    from mlx_lm.models.base import create_causal_mask

    caches = [_fill([3000, 9000], True, dr), None, _fill([3000, 9000], True, dr)]
    _step([caches[0], caches[2]], 4, [3, 0], seed=9)
    assert dr.tails(caches[0]) == dr.tails(caches[2]) == [0, 3]
    width = 16  # wider than any verify block
    h = mx.zeros((2, width, 8), dtype=DTYPE)
    mask = q35._create_qwen3_5_attention_mask(h, caches[0])
    q35._set_qwen3_5_decode_left_padding(caches, _layers3(), None)
    assert dr.tails(caches[0]) is None and dr.tails(caches[2]) is None
    expected = create_causal_mask(
        width, offset=caches[0]._idx, left_padding=caches[0].left_padding
    )
    assert mx.array_equal(mask, expected).item()


def test_a_stricter_mask_with_tails_realigns_instead_of_dropping_it(dr):
    from mlx_lm.models.base import create_causal_mask

    caches = [_fill([3000, 9000], True, dr), None, _fill([3000, 9000], True, dr)]
    _step([caches[0], caches[2]], 4, [3, 0], seed=13)
    assert dr.tails(caches[0]) == [0, 3]
    fa = caches[0]

    def stricter(n, return_array=False, **kwargs):
        base = create_causal_mask(n, offset=fa._idx, left_padding=fa.left_padding)
        first = int(fa.left_padding[1].item())
        return base & ~(mx.arange(base.shape[-1]) == first)[None, None, None, :]

    fa.make_mask = stricter  # instance override: this forward's mask hides a key
    h = mx.zeros((2, 4, 8), dtype=DTYPE)
    q35._create_qwen3_5_attention_mask(h, fa)
    q35._set_qwen3_5_decode_left_padding(caches, _layers3(), None)
    assert dr.tails(caches[0]) is None and dr.tails(caches[2]) is None
    assert not hasattr(fa, "_omlx_qwen35_verify_row_pads")


def _assert_same_rows(ref, exp, dr):
    assert mx.array_equal(exp.offset, ref.offset).item()
    for (rk, rv), (ek, ev) in zip(_rows(ref, dr), _rows(exp, dr), strict=True):
        assert rk.shape == ek.shape
        assert mx.array_equal(rk, ek).item() and mx.array_equal(rv, ev).item()


@pytest.mark.parametrize("keep", [[1], [0], [1, 0]])
def test_a_row_that_finishes_realigns_first(dr, keep):
    ref = _fill([3000, 9000], False, dr)
    exp = _fill([3000, 9000], True, dr)
    _step([ref, exp], 4, [0, 3], seed=31)
    assert dr.tails(exp) == [3, 0]
    ref.filter(keep)
    exp.filter(keep)
    assert dr.tails(exp) is None
    _assert_same_rows(ref, exp, dr)
    # The survivors keep deferring after the batch shrank.
    if len(keep) == 2:
        _step([ref, exp], 4, [3, 1], seed=32)
        assert dr.tails(exp) == [0, 2]
        _assert_same_rows(ref, exp, dr)


def test_a_row_that_joins_realigns_first(dr):
    ref = _fill([3000, 9000], False, dr)
    exp = _fill([3000, 9000], True, dr)
    _step([ref, exp], 4, [0, 2], seed=41)
    assert dr.tails(exp) == [2, 0]
    ref.extend(_fill([5000], False, dr, seed=42))
    exp.extend(_fill([5000], False, dr, seed=42))
    assert dr.tails(exp) is None
    _assert_same_rows(ref, exp, dr)
    _step([ref, exp], 3, [2, 0, 1], seed=43)
    assert dr.tails(exp) == [0, 2, 1]
    _assert_same_rows(ref, exp, dr)


def test_rows_stay_exact_when_the_bank_grows_under_tails(dr):
    rng = random.Random(3)
    ref = _fill([400, 750], False, dr)
    exp = _fill([400, 750], True, dr)
    start = int(exp._keys.shape[2])
    grew_with_tails = False
    for step in range(400):  # each cycle commits 1-5 tokens; stop once grown
        if int(exp._keys.shape[2]) > start and step % 8 == 0:
            break
        block = rng.choice([3, 4, 5])
        before = int(exp._keys.shape[2])
        had_tails = dr.tails(exp) is not None
        # Row 1 lags by one token per cycle, so tails build up to the cap.
        accepted = [block - 1, block - 2]
        _step([ref, exp], block, accepted, seed=300 + step)
        grew_with_tails |= had_tails and int(exp._keys.shape[2]) > before
        _assert_same_rows(ref, exp, dr)
    assert int(exp._keys.shape[2]) > start
    assert grew_with_tails  # the growth happened while rows had tails


def test_reinstalling_the_capacity_patch_keeps_the_wrap(dr):
    from mlx_lm.models.cache import BatchKVCache as LMBatchKVCache

    from omlx.patches.vlm_batch_kv_capacity import apply_batch_kv_capacity_patch

    wrapped = {
        cls: (cls.update_and_fetch, cls.prepare, cls.finalize)
        for cls in (BatchKVCache, LMBatchKVCache)
    }
    apply_batch_kv_capacity_patch()
    assert not dr.apply_qwen35_deferred_realign_patch()  # second apply: no-op
    for cls, methods in wrapped.items():
        assert (cls.update_and_fetch, cls.prepare, cls.finalize) == methods
        assert "_wrap_class" in cls.update_and_fetch.__qualname__


def test_an_ordinary_short_forward_with_tails_realigns(dr):
    # Review finding (Codex, 2026-11-10): a 2-9 token forward outside the
    # speculative verifier takes the stock masked SDPA over the whole bank, so
    # it must not see tails. Equal-length rows: no padding, only the tails.
    ref = _fill([9000, 9000], False, dr)
    exp = _fill([9000, 9000], True, dr)
    _step([ref, exp], 4, [3, 1], seed=51)
    assert dr.tails(exp) == [0, 2]
    for c in (ref, exp):
        c.__dict__[dr.MARK] = True
    mx.random.seed(52)
    q = mx.random.normal((2, QH, 4, D)).astype(DTYPE)
    nk = mx.random.normal((2, KVH, 4, D)).astype(DTYPE)
    outs = []
    for c in (ref, exp):
        h = mx.zeros((2, 4, 8), dtype=DTYPE)
        mask = q35._create_qwen3_5_attention_mask(h, c)
        q35._set_qwen3_5_decode_left_padding([c], _layers(), None)
        keys, values = c.update_and_fetch(nk, nk)
        # What Qwen3_5 Attention.__call__ runs for a non-string mask.
        outs.append(
            q35.scaled_dot_product_attention(
                q, keys, values, cache=c, scale=SCALE, mask=mask
            )
        )
    assert dr.tails(exp) is None
    assert mx.array_equal(outs[0], outs[1]).item()


def test_the_verifier_forward_runs_inside_the_verify_mark(dr):
    from mlx_vlm.models.qwen3_5.speculative_verifier import (
        Qwen3_5BatchInvariantForward,
    )

    assert getattr(Qwen3_5BatchInvariantForward._model, "_omlx_verify_marked", False)
    seen = []

    def stop():
        seen.append(dr.in_verify())
        raise RuntimeError("stop after the mark check")

    # Other patches may wrap _model under ours and touch the model first, so
    # record the mark at whichever the call chain reaches first.
    class Probe(Qwen3_5BatchInvariantForward):
        @staticmethod
        def _helpers():
            stop()

    class Model:
        def __getattr__(self, name):
            stop()

    assert not dr.in_verify()
    with pytest.raises(RuntimeError):
        Probe()._model(Model(), None, None, None, None, None, None)
    assert seen == [True] and not dr.in_verify()
