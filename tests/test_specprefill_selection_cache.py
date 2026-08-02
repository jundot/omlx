# SPDX-License-Identifier: Apache-2.0
"""Tests for SpecPrefill selection memoization (#1079).

The core guarantee: a byte-identical prompt (same draft model + scoring params)
runs the score->select pipeline exactly once and reuses the selection thereafter,
while still recomputing the per-request position bookkeeping.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.specprefill.policy import SpecPrefillScoringPlan
from omlx.specprefill.selection_cache import (
    SpecPrefillSelectionCache,
    apply_cached_selection,
    build_selection_key,
    memoize_scoring,
)


def _plan(tokens, keep_pct=0.5, effective_system=0):
    return SpecPrefillScoringPlan(
        tokens_to_score=list(tokens),
        effective_system=effective_system,
        keep_pct=keep_pct,
        n_to_score=len(tokens),
    )


def _request(cached_tokens=0):
    return SimpleNamespace(
        specprefill_indices=None,
        specprefill_total_tokens=None,
        specprefill_position_offset=None,
        _specprefill_system_tokens=None,
        cached_tokens=cached_tokens,
    )


# --------------------------------------------------------------------------- #
# LRU cache
# --------------------------------------------------------------------------- #
def test_lru_evicts_least_recently_used_beyond_capacity():
    cache = SpecPrefillSelectionCache(max_entries=2)
    cache.put(("a",), [1])
    cache.put(("b",), [2])
    cache.put(("c",), [3])  # evicts oldest "a"
    assert cache.get(("a",)) is None
    assert cache.get(("b",)) == [2]
    assert cache.get(("c",)) == [3]
    assert len(cache) == 2


def test_get_refreshes_recency_so_touched_entry_survives():
    cache = SpecPrefillSelectionCache(max_entries=2)
    cache.put(("a",), [1])
    cache.put(("b",), [2])
    assert cache.get(("a",)) == [1]  # "a" is now most-recently used
    cache.put(("c",), [3])  # should evict "b", not "a"
    assert cache.get(("a",)) == [1]
    assert cache.get(("b",)) is None


def test_cached_values_are_copies_not_caller_aliases():
    cache = SpecPrefillSelectionCache()
    original = [1, 2, 3]
    cache.put(("k",), original)
    original.append(999)  # mutate after put
    got = cache.get(("k",))
    assert got == [1, 2, 3]
    got.append(0)  # mutate after get
    assert cache.get(("k",)) == [1, 2, 3]


def test_hit_and_miss_counters():
    cache = SpecPrefillSelectionCache()
    cache.get(("missing",))
    cache.put(("k",), [1])
    cache.get(("k",))
    assert cache.misses == 1
    assert cache.hits == 1


# --------------------------------------------------------------------------- #
# Cache key
# --------------------------------------------------------------------------- #
def test_key_is_stable_for_identical_inputs():
    a = build_selection_key(draft_model_id="d", tokens_to_score=[1, 2, 3, 4], keep_pct=0.5)
    b = build_selection_key(draft_model_id="d", tokens_to_score=[1, 2, 3, 4], keep_pct=0.5)
    assert a == b


@pytest.mark.parametrize(
    "mutation",
    [
        {"tokens_to_score": [1, 2, 3, 5]},  # one token differs
        {"keep_pct": 0.3},
        {"draft_model_id": "other-draft"},
        {"chunk_size": 16},
        {"n_lookahead": 4},
        {"pool_kernel": 0},
        {"scorer": "self_layers"},
    ],
)
def test_key_changes_when_any_deterministic_input_changes(mutation):
    base = {"draft_model_id": "d", "tokens_to_score": [1, 2, 3, 4], "keep_pct": 0.5}
    assert build_selection_key(**base) != build_selection_key(**{**base, **mutation})


def test_key_handles_non_list_token_sequences():
    # tokens_to_score may arrive as a list or an mx.array slice.
    as_list = build_selection_key(draft_model_id="d", tokens_to_score=[7, 8, 9], keep_pct=0.5)
    as_array = build_selection_key(
        draft_model_id="d", tokens_to_score=mx.array([7, 8, 9]), keep_pct=0.5
    )
    assert as_list == as_array


# --------------------------------------------------------------------------- #
# apply_cached_selection
# --------------------------------------------------------------------------- #
def test_apply_reuses_selection_and_recomputes_position_bookkeeping():
    plan = _plan(range(6), keep_pct=0.5, effective_system=2)
    req = _request(cached_tokens=7)
    apply_cached_selection(req, plan, [0, 1, 4, 5])
    assert req.specprefill_indices.tolist() == [0, 1, 4, 5]
    assert req.specprefill_indices.dtype == mx.int32
    assert req.specprefill_total_tokens == 6
    # position_offset must be recomputed from THIS request's cached_tokens.
    assert req.specprefill_position_offset == 7 + 2
    assert req._specprefill_system_tokens == 2


# --------------------------------------------------------------------------- #
# End-to-end memoization (the fix)
# --------------------------------------------------------------------------- #
def test_identical_prompt_scores_once_and_reuses_selection():
    cache = SpecPrefillSelectionCache()
    tokens = list(range(9000))  # realistically long scoring target
    plan = _plan(tokens, keep_pct=0.5)
    key = build_selection_key(draft_model_id="d", tokens_to_score=tokens, keep_pct=0.5)

    scoring_calls = {"n": 0}

    def scoring_for(req):
        def _run():
            scoring_calls["n"] += 1
            # Emulate run_specprefill_draft_scoring's effect on the request.
            req.specprefill_indices = mx.array([0, 1, 2, 3, 4], dtype=mx.int32)
            req.specprefill_total_tokens = plan.n_to_score

        return _run

    r1 = _request(cached_tokens=0)
    hit1 = memoize_scoring(cache=cache, key=key, request=r1, plan=plan, run_scoring=scoring_for(r1))

    # Same prompt again, but a different prefix-cache warmth (cached_tokens).
    r2 = _request(cached_tokens=100)
    hit2 = memoize_scoring(cache=cache, key=key, request=r2, plan=plan, run_scoring=scoring_for(r2))

    assert scoring_calls["n"] == 1  # <-- the whole point: scored once, not twice
    assert hit1 is False
    assert hit2 is True
    assert r2.specprefill_indices.tolist() == [0, 1, 2, 3, 4]  # selection reused
    assert r2.specprefill_total_tokens == plan.n_to_score
    # Recomputed for THIS request even on a cache hit.
    assert r2.specprefill_position_offset == 100


def test_different_prompts_each_score():
    cache = SpecPrefillSelectionCache()
    calls = {"n": 0}

    def run_for(req, idx):
        def _run():
            calls["n"] += 1
            req.specprefill_indices = mx.array([idx], dtype=mx.int32)

        return _run

    for i in range(3):
        plan = _plan(range(9000 + i))  # each prompt differs
        key = build_selection_key(
            draft_model_id="d", tokens_to_score=plan.tokens_to_score, keep_pct=plan.keep_pct
        )
        req = _request()
        memoize_scoring(cache=cache, key=key, request=req, plan=plan, run_scoring=run_for(req, i))

    assert calls["n"] == 3  # no false sharing across distinct prompts


def test_failed_scoring_is_not_cached():
    cache = SpecPrefillSelectionCache()
    plan = _plan(range(100))
    key = build_selection_key(
        draft_model_id="d", tokens_to_score=plan.tokens_to_score, keep_pct=plan.keep_pct
    )
    req = _request()

    def failing_scoring():
        # run_specprefill_draft_scoring sets indices to None on failure.
        req.specprefill_indices = None

    hit = memoize_scoring(cache=cache, key=key, request=req, plan=plan, run_scoring=failing_scoring)
    assert hit is False
    assert len(cache) == 0  # a transient failure must not poison future requests


def test_none_cache_falls_back_to_plain_scoring():
    # Defensive: memoize_scoring with cache=None just runs scoring.
    plan = _plan(range(50))
    key = build_selection_key(
        draft_model_id="d", tokens_to_score=plan.tokens_to_score, keep_pct=plan.keep_pct
    )
    req = _request()
    ran = {"n": 0}

    def _run():
        ran["n"] += 1
        req.specprefill_indices = mx.array([0], dtype=mx.int32)

    hit = memoize_scoring(cache=None, key=key, request=req, plan=plan, run_scoring=_run)
    assert hit is False
    assert ran["n"] == 1


# --------------------------------------------------------------------------- #
# Scheduler integration: the real _try_specprefill_scoring memoizes end-to-end
# --------------------------------------------------------------------------- #
def test_scheduler_try_specprefill_scoring_scores_identical_prompt_once(monkeypatch):
    """Drive the real Scheduler method twice with a byte-identical prompt.

    Proves the wiring (admission plan -> key -> memoize -> reuse) short-circuits
    the draft scoring on the second, identical request.
    """
    import omlx.specprefill.draft as draft_mod
    from omlx.scheduler import Scheduler

    calls = {"n": 0}

    def fake_scoring(*, request, plan, **_):
        calls["n"] += 1
        request.specprefill_indices = mx.array([0, 1, 2, 3], dtype=mx.int32)
        request.specprefill_total_tokens = plan.n_to_score

    monkeypatch.setattr(draft_mod, "run_specprefill_draft_scoring", fake_scoring)

    sched = SimpleNamespace(
        _specprefill_draft_model=object(),  # non-None => SpecPrefill applies
        _specprefill_draft_model_name="draft-model-x",
        _specprefill_selection_cache=SpecPrefillSelectionCache(),
        _draft_prefix_cache=None,
        config=SimpleNamespace(model_name="target-model", prefill_step_size=2048),
        _stream=None,
        _extract_cache_states=lambda used: ([], None),
    )

    def make_request(cached_tokens):
        return SimpleNamespace(
            _specprefill_enabled=True,
            vlm_inputs_embeds=None,
            remaining_tokens=list(range(9000)),  # > DEFAULT_THRESHOLD (8192)
            prompt_token_ids=None,
            specprefill_system_end=0,
            cached_tokens=cached_tokens,
            _specprefill_threshold=None,
            _specprefill_keep_pct=0.5,
            specprefill_indices=None,
            specprefill_total_tokens=None,
            specprefill_position_offset=None,
            _specprefill_system_tokens=None,
        )

    r1 = make_request(cached_tokens=0)
    Scheduler._try_specprefill_scoring(sched, r1)

    r2 = make_request(cached_tokens=42)  # same prompt, different prefix-cache warmth
    Scheduler._try_specprefill_scoring(sched, r2)

    assert calls["n"] == 1  # <-- scored once across two identical requests
    assert r1.specprefill_indices.tolist() == [0, 1, 2, 3]
    assert r2.specprefill_indices.tolist() == [0, 1, 2, 3]  # reused, not re-scored
    assert r2.specprefill_position_offset == 42  # recomputed for THIS request
