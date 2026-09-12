# SPDX-License-Identifier: Apache-2.0
"""Parked-decode head folds: the MTP head cache stays warm while the controller
parks speculation, so the re-entry probe starts primed."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from omlx.patches.mlx_lm_mtp import prompt_priming as pp
from tests.test_mtp_prompt_priming import (  # noqa: F401 - autouse fixture
    TINY_CONFIG,
    _apply_patch,
    _make_tiny_model,
)

H = TINY_CONFIG["hidden_size"]


class _Anchor:
    def __init__(self, offset):
        self.offset = offset


def _cache_at(n):
    return [_Anchor(n)]


def _head_offset(cache):
    return int(cache[0].offset)


def _fold_hidden(model, h):
    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    return bg._head_input_hidden(model, h)


def _hidden(value, rows=1):
    return (mx.ones((1, rows, H)) * value).astype(mx.bfloat16)


def test_parked_ctx_folds_in_blocks_and_flushes_at_probe(monkeypatch):
    monkeypatch.setenv("OMLX_MTP_PARK_FOLD_BLOCK", "8")
    model = _make_tiny_model()
    pre = 5  # pairs already in the head cache when parking
    seed_h = _fold_hidden(model, _hidden(0.03, pre))
    seed_t = mx.arange(pre, dtype=mx.uint32).reshape(1, pre)
    head_cache = model.make_mtp_cache()
    model.mtp_forward(seed_h, seed_t, head_cache, logits_keep=1)
    ref_cache = model.make_mtp_cache()
    model.mtp_forward(seed_h, seed_t, ref_cache, logits_keep=1)

    hidden0 = _fold_hidden(model, _hidden(0.02))
    ctx = pp.begin_parked_ctx(model, head_cache, hidden0, folded=pre, expected_offset=10)
    assert ctx is not None and ctx.parked and pp._find_ctx(model) is ctx

    pairs_h, pairs_t = [hidden0], []
    for step in range(20):
        tok = mx.array([[step % 7]], dtype=mx.uint32)
        h = _fold_hidden(model, _hidden((step + 1) * 0.01))
        pairs_t.append(tok)
        pp.maybe_capture(model, tok, h, _cache_at(11 + step))
        pairs_h.append(h)
    assert pp._find_ctx(model) is ctx
    assert len(ctx.pending_pairs) < 8  # folded in blocks of 8
    assert ctx.folded >= pre + 16

    main_tok = mx.array([[3]], dtype=mx.uint32)
    primed = pp.take_primed(model, _cache_at(31), main_tok)
    assert primed is not None
    cache, hist = primed
    assert hist == pre + 20 + 1
    assert pp._find_ctx(model) is None

    model.mtp_forward(
        mx.concatenate(pairs_h[:-1], axis=1), mx.concatenate(pairs_t, axis=1), ref_cache, logits_keep=1
    )
    model.mtp_forward(pairs_h[-1], main_tok, ref_cache, logits_keep=1)
    mx.eval(cache[0].keys, cache[0].values, ref_cache[0].keys, ref_cache[0].values)
    assert _head_offset(cache) == _head_offset(ref_cache) == pre + 20 + 1
    assert mx.allclose(cache[0].keys, ref_cache[0].keys, atol=1e-5).item()
    assert mx.allclose(cache[0].values, ref_cache[0].values, atol=1e-5).item()


def test_parked_ctx_drops_on_discontiguous_step():
    model = _make_tiny_model()
    head_cache = model.make_mtp_cache()
    ctx = pp.begin_parked_ctx(model, head_cache, _hidden(0.02), folded=0, expected_offset=10)
    assert ctx is not None
    pp.maybe_capture(model, mx.array([[1]], dtype=mx.uint32), _hidden(0.01), _cache_at(11))
    assert pp._find_ctx(model) is ctx
    # A rewind (offset going backwards) must invalidate the parked timeline.
    pp.maybe_capture(model, mx.array([[2]], dtype=mx.uint32), _hidden(0.01), _cache_at(9))
    assert pp._find_ctx(model) is None


def test_parked_ctx_ignores_prime_window(monkeypatch):
    monkeypatch.setenv("OMLX_MTP_PRIME_WINDOW", "4")
    model = _make_tiny_model()
    head_cache = model.make_mtp_cache()
    ctx = pp.begin_parked_ctx(model, head_cache, _hidden(0.02), folded=0, expected_offset=10)
    for step in range(10):
        pp.maybe_capture(model, mx.array([[step]], dtype=mx.uint32), _hidden(0.01), _cache_at(11 + step))
    assert pp._find_ctx(model) is ctx and not ctx.window_exceeded


def test_feed_to_standard_survives_a_backbone_without_hidden(monkeypatch):
    """The parked fold is optional: a backbone that yields no hidden must not fail the park."""
    from types import SimpleNamespace

    from omlx.patches.mlx_lm_mtp import batch_generator as bg
    from omlx.patches.mlx_lm_mtp.batch_generator import _MtpState

    logits = mx.zeros((1, 1, 8))
    monkeypatch.setattr(bg, "_call_backbone", lambda model, x, cache: (logits, None, None))
    monkeypatch.setattr(bg, "_proc_list", lambda gb: None)
    monkeypatch.setattr(bg, "_set_singleton_mrope_delta", lambda gb: None)
    monkeypatch.setattr(bg, "_resolve_sampler", lambda gb: (lambda lp: mx.argmax(lp, axis=-1)))
    monkeypatch.setattr(bg, "_clear_rollback", lambda cache: None)
    state = _MtpState(uid=7, next_main=mx.array([3], dtype=mx.uint32))
    gen_batch = SimpleNamespace(model=object(), prompt_cache=[], _next_tokens=None, _next_logprobs=None)
    assert bg._feed_next_main_to_standard(gen_batch, state) is True
    assert state.park_hidden is None
    assert gen_batch._next_tokens is not None
