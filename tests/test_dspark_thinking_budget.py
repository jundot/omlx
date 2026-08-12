# SPDX-License-Identifier: Apache-2.0
"""Thinking budget × DSpark chain-verify: repro for the early-fire bug.

The DSpark chain path (``batch_generator._run_verify_cycle_chain``) applies
logits processors **2k+1 times per cycle**:

  - k draft-gen calls (``_chain_next_drafts`` / ``_dspark_next_drafts``,
    one per speculative draft position), and
  - k+1 verify calls (one per row of ``[next_main, d1..dk]``).

Only ``m+1`` tokens are actually emitted (m accepted drafts + 1 bonus/verify
correction). ``ThinkingBudgetProcessor.__call__`` increments
``_thinking_tokens`` on every invocation while thinking (thinking.py:465), so
the budget fires early by ``(2k+1) - (m+1) = 2k - m`` tokens per cycle — even
on full accept (drift = k per cycle).

These tests drive the REAL verify cycle with a real budget processor and a
real token buffer. RED (pre-fix): the counter drifts ahead of emitted tokens.
GREEN (post-fix, snapshot/restore like MTPProcessingSampler): counter ==
emitted at every cycle.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.api.thinking import ThinkingBudgetProcessor
from omlx.patches.mlx_lm_mtp import batch_generator as bg

VOCAB = 32
CLOSE = 21  # single-token close-think
FILL = 7  # the model's "thinking filler" token
REJECT = 30  # valid vocab id that never equals a draft id (1..k)
PROMPT = [1, 2, 3, 10]  # prompt ends with <think> (10) -> _in_thinking=True


def _make_budget_processor(budget: int) -> ThinkingBudgetProcessor:
    return ThinkingBudgetProcessor(
        think_end_token_ids=[CLOSE],
        budget=budget,
        think_start_token_id=None,
        leading_token_ids=[],  # force sequence = just [CLOSE]
        trailing_token_ids=[],
        token_to_piece=None,
    )


def _greedy(logprobs):
    return mx.argmax(logprobs, axis=-1).astype(mx.uint32)


def _logits_for(targets):
    rows = []
    for target in targets:
        row = [-100.0] * VOCAB
        row[target] = 0.0
        rows.append(row)
    return mx.array([rows], dtype=mx.float32)


class _Counter:
    """Mimics a real TokenBuffer append without mlx_lm imports."""

    def __init__(self):
        self._tokens = list(PROMPT)
        self._size = len(PROMPT)

    def update_and_fetch(self, toks):
        t = toks.tolist()
        if isinstance(t, int):
            t = [t]
        self._tokens.extend(t)
        self._size = len(self._tokens)
        return mx.array(self._tokens, dtype=mx.int32)

    @property
    def tokens(self):
        return mx.array(self._tokens[: self._size], dtype=mx.int32)


def _make_state(k: int, draft_ids, emitted: int):
    state = bg._MtpState(
        uid=1,
        chain=True,
        depth=k,
        mtp_cache=[],
        next_main=mx.array([15], dtype=mx.uint32),
        drafts=mx.array(draft_ids, dtype=mx.uint32),
        draft_lps=[mx.zeros((VOCAB,)) for _ in draft_ids],
    )
    return state


def _make_batch(proc, emitted: int, k: int):
    cache = SimpleNamespace(offset=emitted - 1)

    def mtp_forward(hidden_rows, committed, mtp_cache, **kwargs):
        # MTP head: propose FILL for every draft position.
        n = int(committed.shape[1])
        return _logits_for([FILL] * n), mx.zeros((1, n, 8), dtype=mx.float32)

    model = SimpleNamespace(
        _omlx_mtp_commit_align=0,
        _omlx_mtp_head_prenorm=True,  # skip trunk-norm path in draft-gen
        mtp_forward=mtp_forward,
    )
    buf = _Counter()
    batch = SimpleNamespace(
        model=model,
        prompt_cache=[cache],
        tokens=[list(range(emitted))],
        samplers=[None],
        fallback_sampler=_greedy,
        logits_processors=[[proc]],
        _token_context=[buf],
    )
    return batch, cache, buf


def _run_cycle(monkeypatch, proc, emitted, k, accept_m, draft_ids=None):
    """One real ``_run_verify_cycle_chain``; ``accept_m`` drafts accepted."""
    batch, cache, buf = _make_batch(proc, emitted, k)
    if draft_ids is None:
        draft_ids = [FILL] * k
    state = _make_state(k, draft_ids, emitted)

    def fake_backbone(_model, inputs, _cache, **_kwargs):
        width = int(inputs.shape[1])
        cache.offset += width
        # Row j predicts drafts[j] for j < k; accept m of them, then a
        # non-draft correction token at row m (and beyond).
        targets = draft_ids[:] + [20]
        for j in range(accept_m, k):
            targets[j] = REJECT  # mismatch -> draft j rejected
        return (
            _logits_for(targets),
            mx.zeros((1, width, 8), dtype=mx.float32),
            None,
        )

    def fake_rollback(_model, _cache, accepted, num_drafts, _gdn_states):
        cache.offset -= num_drafts - accepted
        return True

    monkeypatch.setattr(bg, "_call_backbone", fake_backbone)
    monkeypatch.setattr(bg, "_chain_rollback", fake_rollback)
    # REAL _chain_next_drafts: model.mtp_forward proposes FILL drafts, and
    # the draft-gen loop applies the budget processor once per draft
    # position (the second over-counting site).
    monkeypatch.setattr(bg, "_clear_rollback", lambda _cache: None)

    before = proc._thinking_tokens
    bg._run_verify_cycle_chain(batch, state)
    emitted_this = len(state.queue)
    delta = proc._thinking_tokens - before
    return emitted_this, delta


class TestBudgetCounterDrift:
    """RED: the real chain cycle over-counts the budget on speculative
    positions. After the fix these become contract assertions."""

    def test_full_accept_drifts_by_k(self, monkeypatch):
        # k=3, all 3 drafts accepted: 4 tokens emitted but (with the bug)
        # 2k+1 = 7 processor calls fire.
        proc = _make_budget_processor(10_000)
        emitted, delta = _run_cycle(monkeypatch, proc, emitted=10, k=3, accept_m=3)
        assert emitted == 4
        # Post-fix contract: one call per emitted token.
        assert delta == emitted, (
            f"budget advanced {delta} for {emitted} emitted tokens "
            f"(overcount {delta - emitted})"
        )

    def test_partial_accept_drifts_more(self, monkeypatch):
        proc = _make_budget_processor(10_000)
        emitted, delta = _run_cycle(monkeypatch, proc, emitted=10, k=3, accept_m=1)
        assert emitted == 2
        assert delta == emitted, (
            f"budget advanced {delta} for {emitted} emitted tokens "
            f"(overcount {delta - emitted})"
        )

    def test_no_drafts_k0_single_call(self, monkeypatch):
        proc = _make_budget_processor(10_000)
        # k=0: single plain step, 1 call, 1 emit.
        emitted, delta = _run_cycle(monkeypatch, proc, emitted=10, k=0, accept_m=0, draft_ids=[])
        assert emitted == 1
        assert delta == emitted


class TestBudgetFiresAtBudgetTokens:
    """End-to-end: budget fires only after exactly `budget` thinking tokens
    have been emitted (not early)."""

    def test_fires_at_emitted_budget(self, monkeypatch):
        budget = 12
        proc = _make_budget_processor(budget)
        emitted_total = 0
        cycle = 0
        # k=2; mix of full and partial accepts. Loop until the processor
        # starts forcing (budget reached).
        while not proc._forcing and cycle < 50:
            m = 2 if cycle % 3 else 1
            emitted, delta = _run_cycle(monkeypatch, proc, emitted=10 + emitted_total, k=2, accept_m=m)
            emitted_total += emitted
            cycle += 1
        assert proc._forcing, "budget should have forced close-think"
        # The counter at force time must equal the number of emitted
        # thinking tokens so far (contract: 1 call per emitted token).
        assert proc._thinking_tokens == emitted_total, (
            f"budget fired with counter={proc._thinking_tokens} "
            f"after {emitted_total} emitted (early-fire {proc._thinking_tokens - emitted_total})"
        )
        assert proc._thinking_tokens >= budget


class TestDraftGenDoesNotCount:
    """The draft-generation processor calls (k per cycle) shape drafts but
    must NOT advance the budget — drafts are speculative until verified."""

    def test_draft_gen_shapes_without_counting(self):
        proc = _make_budget_processor(10_000)
        batch, cache, buf = _make_batch(proc, emitted=10, k=2)
        state = _make_state(2, [FILL, FILL], 10)
        # Prime the processor so _accepted_up_to is set (post-init did one
        # real emit already).
        buf.update_and_fetch(mx.array([5], dtype=mx.uint32))
        proc(buf.tokens, _logits_for([FILL]))
        before = proc._thinking_tokens

        # Real draft-gen: one batch head forward + per-position processor
        # calls for 2 drafts. committed = the anchor token.
        hidden = mx.zeros((1, 1, 8), dtype=mx.float32)
        committed = mx.array([5], dtype=mx.uint32)
        bg._chain_next_drafts(batch, state, hidden, committed, buf.tokens)

        delta = proc._thinking_tokens - before
        assert state.drafts.shape[0] == 2
        # Draft-gen calls must be rewound: zero budget advance.
        assert delta == 0, (
            f"draft-gen leaked {delta} into the budget counter "
            f"(speculative drafts must not count until emitted)"
        )
