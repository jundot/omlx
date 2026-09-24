"""Ordering, token accounting, and cost limits for prefill plans."""

from dataclasses import replace

import pytest

from omlx.prefill.planning import (
    PrefillCandidate,
    PrefillDefer,
    PrefillEstimate,
    PrefillFallback,
    PrefillReason,
    PrefillRun,
    plan_prefill_batch,
)


def _run_plan(*args, **kwargs):
    decision = plan_prefill_batch(*args, **kwargs)
    assert isinstance(decision, PrefillRun)
    return decision.plan


def _candidates(*remaining: int, chunk_cap: int = 512) -> list[PrefillCandidate]:
    return [
        PrefillCandidate(f"request-{index}", tokens, chunk_cap)
        for index, tokens in enumerate(remaining)
    ]


def test_different_prompt_lengths_share_equal_unpadded_chunks():
    candidates = _candidates(1_000, 1_800)

    first = _run_plan(candidates, max_batch_size=4, token_budget=8_192)

    assert first.request_ids == ("request-0", "request-1")
    assert first.chunk_tokens == 512
    assert first.total_tokens == 1_024

    remaining = [
        replace(candidate, remaining_tokens=candidate.remaining_tokens - 512)
        for candidate in candidates
    ]
    second = _run_plan(remaining, max_batch_size=4, token_budget=8_192)

    assert second.chunk_tokens == 488
    assert second.total_tokens == 976


def test_budget_counts_every_row_and_preserves_remainder():
    plan = _run_plan(_candidates(800, 900, 1_000), max_batch_size=4, token_budget=1_000)

    assert plan.chunk_tokens == 333
    assert plan.total_tokens == 999


@pytest.mark.parametrize("max_batch_size", [0, 1, 2.5, "2", None, True])
def test_disabled_batching_leaves_work_for_singleton(max_batch_size):
    assert not isinstance(
        plan_prefill_batch(
            _candidates(100, 100),
            max_batch_size=max_batch_size,
            token_budget=1_000,
        ),
        PrefillRun,
    )


@pytest.mark.parametrize("remaining", [(), (100,), (0, 100), (100, 0, 100)])
def test_empty_singleton_and_finished_head_return_without_waiting(remaining):
    assert not isinstance(
        plan_prefill_batch(
            _candidates(*remaining), max_batch_size=4, token_budget=1_000
        ),
        PrefillRun,
    )


@pytest.mark.parametrize("token_budget", [-1, 0, 1, 2.5, "2", None, True])
def test_exhausted_budget_does_not_schedule(token_budget):
    assert not isinstance(
        plan_prefill_batch(
            _candidates(100, 100), max_batch_size=4, token_budget=token_budget
        ),
        PrefillRun,
    )


@pytest.mark.parametrize(
    "changed",
    [
        {"compatible": False},
        {"priority": 1},
        {"cache_tokens": 32},
        {"max_chunk_tokens": 0},
        {"max_chunk_tokens": 2.5},
        {"remaining_tokens": True},
        {"remaining_tokens": None},
        {"cache_tokens": -1},
        {"cache_tokens": 0.0},
    ],
)
def test_incompatible_middle_request_cannot_be_bypassed(changed):
    candidates = _candidates(100, 100, 100)
    candidates[1] = replace(candidates[1], **changed)

    assert isinstance(
        plan_prefill_batch(candidates, max_batch_size=4, token_budget=1_000),
        PrefillFallback,
    )


def test_incompatible_later_request_keeps_earlier_batch():
    candidates = _candidates(100, 100, 100, 100)
    candidates[2] = replace(candidates[2], compatible=False)

    plan = _run_plan(candidates, max_batch_size=4, token_budget=1_000)

    assert plan.request_ids == ("request-0", "request-1")


def test_cap_never_increases_even_for_tiny_tail():
    candidates = _candidates(100, 1)
    candidates[0] = replace(candidates[0], max_chunk_tokens=2)

    plan = _run_plan(candidates, max_batch_size=4, token_budget=8_192)

    assert plan.chunk_tokens == 1
    assert plan.total_tokens == 2


def test_duplicate_request_identity_is_rejected():
    candidate = PrefillCandidate("duplicate", 100, 10)

    with pytest.raises(ValueError, match="duplicate request IDs"):
        plan_prefill_batch([candidate, candidate], max_batch_size=2, token_budget=100)


def test_memory_estimate_shrinks_chunk_to_largest_feasible_size():
    estimates = []

    def estimate(batch_size, chunk_tokens, cache_tokens):
        estimates.append((batch_size, chunk_tokens, cache_tokens))
        return PrefillEstimate(100 * batch_size + batch_size * chunk_tokens)

    plan = _run_plan(
        _candidates(1_000, 1_000, 1_000),
        max_batch_size=3,
        token_budget=4_000,
        estimate=estimate,
        max_memory_bytes=600,
    )

    assert plan.request_ids == ("request-0", "request-1", "request-2")
    assert plan.chunk_tokens == 100
    assert plan.estimate.memory_bytes == 600
    assert len(estimates) <= 11


def test_memory_limit_can_reduce_width_without_reordering():
    plan = _run_plan(
        _candidates(100, 100, 100, 100),
        max_batch_size=4,
        token_budget=1_000,
        estimate=lambda width, chunk, cache: PrefillEstimate(width * (100 + chunk)),
        max_memory_bytes=250,
    )

    assert plan.request_ids == ("request-0", "request-1")
    assert plan.chunk_tokens == 25


def test_estimator_receives_current_cache_length():
    candidates = [
        replace(candidate, cache_tokens=512) for candidate in _candidates(100, 100)
    ]
    seen_offsets = []

    def estimate(width, chunk, cache_tokens):
        seen_offsets.append(cache_tokens)
        return PrefillEstimate(width * (cache_tokens + chunk))

    plan = _run_plan(
        candidates,
        max_batch_size=2,
        token_budget=200,
        estimate=estimate,
        max_memory_bytes=1_100,
    )

    assert plan.chunk_tokens == 38
    assert set(seen_offsets) == {512}


def test_duration_and_memory_limits_both_apply():
    plan = _run_plan(
        _candidates(100, 100),
        max_batch_size=2,
        token_budget=200,
        estimate=lambda width, chunk, cache: PrefillEstimate(
            width * chunk, width * chunk / 100
        ),
        max_memory_bytes=100,
        max_duration_seconds=0.5,
    )

    assert plan.chunk_tokens == 25
    assert plan.total_tokens == 50
    assert plan.estimate.duration_seconds == 0.5


@pytest.mark.parametrize(
    "options",
    [
        {"max_memory_bytes": 100},
        {"max_duration_seconds": 0.5},
        {"max_memory_bytes": 100, "estimate": lambda *args: None},
        {
            "max_duration_seconds": 0.5,
            "estimate": lambda *args: PrefillEstimate(0),
        },
        {"max_memory_bytes": -1},
        {
            "max_memory_bytes": float("nan"),
            "estimate": lambda *args: PrefillEstimate(1_000),
        },
        {
            "max_memory_bytes": float("inf"),
            "estimate": lambda *args: PrefillEstimate(1_000),
        },
        {"max_memory_bytes": True, "estimate": lambda *args: PrefillEstimate(0)},
        {"max_duration_seconds": float("nan")},
        {"max_duration_seconds": True},
        {"estimate": lambda *args: PrefillEstimate(-1)},
        {"estimate": lambda *args: PrefillEstimate(float("nan"))},
        {"estimate": lambda *args: PrefillEstimate(0.5)},
        {"estimate": lambda *args: PrefillEstimate(True)},
        {"estimate": lambda *args: PrefillEstimate(0, float("nan"))},
        {"estimate": lambda *args: PrefillEstimate(0, True)},
    ],
)
def test_missing_or_invalid_costs_cannot_bypass_limits(options):
    assert not isinstance(
        plan_prefill_batch(
            _candidates(10, 10), max_batch_size=2, token_budget=20, **options
        ),
        PrefillRun,
    )


def test_minimum_batch_that_cannot_fit_falls_back():
    assert not isinstance(
        plan_prefill_batch(
            _candidates(100, 100),
            max_batch_size=2,
            token_budget=200,
            estimate=lambda *args: PrefillEstimate(100),
            max_memory_bytes=99,
        ),
        PrefillRun,
    )


@pytest.mark.parametrize("token_budget", range(2, 33))
def test_plans_never_exceed_token_or_individual_chunk_bounds(token_budget):
    candidates = _candidates(9, 19, 29, chunk_cap=7)
    candidates[1] = replace(candidates[1], max_chunk_tokens=3)

    plan = _run_plan(candidates, max_batch_size=3, token_budget=token_budget)

    assert 0 < plan.total_tokens <= token_budget
    assert plan.request_ids == tuple(
        candidate.request_id for candidate in candidates[: len(plan.request_ids)]
    )
    for candidate in candidates[: len(plan.request_ids)]:
        assert plan.chunk_tokens <= candidate.remaining_tokens
        assert plan.chunk_tokens <= candidate.max_chunk_tokens


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({"token_budget": 1}, PrefillDefer(PrefillReason.TOKEN_BUDGET)),
        ({"max_batch_size": 1}, PrefillFallback(PrefillReason.DISABLED)),
        ({"max_memory_bytes": 1}, PrefillDefer(PrefillReason.MEMORY_LIMIT)),
        ({"max_duration_seconds": 0.1}, PrefillFallback(PrefillReason.DURATION_LIMIT)),
        (
            {"estimate": lambda *args: None},
            PrefillFallback(PrefillReason.ESTIMATE_UNAVAILABLE),
        ),
        ({"max_duration_seconds": -1}, PrefillFallback(PrefillReason.INVALID_LIMITS)),
        (
            {"allow_width_reduction": "false"},
            PrefillFallback(PrefillReason.INVALID_LIMITS),
        ),
    ],
)
def test_decisions_distinguish_backpressure_from_scalar_fallback(options, expected):
    arguments = {
        "max_batch_size": 3,
        "token_budget": 200,
        "estimate": lambda width, chunk, cache: PrefillEstimate(width * chunk, chunk),
        "max_memory_bytes": 200,
    }
    arguments.update(options)
    assert plan_prefill_batch(_candidates(10, 10, 10), **arguments) == expected


def test_live_group_cannot_silently_drop_a_row_to_fit_memory():
    arguments = {
        "max_batch_size": 3,
        "token_budget": 30,
        "estimate": lambda width, chunk, cache: PrefillEstimate(width * 10),
        "max_memory_bytes": 20,
    }
    candidates = _candidates(10, 10, 10)
    assert isinstance(plan_prefill_batch(candidates, **arguments), PrefillRun)
    assert plan_prefill_batch(
        candidates, allow_width_reduction=False, **arguments
    ) == PrefillDefer(PrefillReason.MEMORY_LIMIT)


def test_live_group_defers_when_budget_cannot_advance_every_row():
    assert plan_prefill_batch(
        _candidates(10, 10, 10),
        max_batch_size=3,
        token_budget=2,
        allow_width_reduction=False,
    ) == PrefillDefer(PrefillReason.TOKEN_BUDGET)


def test_memory_refusal_takes_precedence_when_no_chunk_is_safe():
    assert plan_prefill_batch(
        _candidates(10, 10),
        max_batch_size=2,
        token_budget=20,
        estimate=lambda *args: PrefillEstimate(100, 100),
        max_memory_bytes=99,
        max_duration_seconds=1,
    ) == PrefillDefer(PrefillReason.MEMORY_LIMIT)
