"""Retention-policy tests for per-boundary recurrent-state checkpoints.

Regression guard for the measured cost of unbounded retention: one 234,070-token
prefill of Qwen3.8-Flash-Next captured 113 checkpoints of 115.6MB each -- 12.06GB
resident for the whole prefill, 52.5KB per prompt token against the 27KB of the
entire live attention cache. Retention must bound that to the configured budget,
keep the boundary a continuation resumes from, and spread what it keeps: keeping
only the deepest checkpoints drives the hit rate of an early-diverging branch to
zero, and one dominant gap re-prefills most of the prompt.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from omlx.cache.boundary_retention import (
    MAX_RETAINED_BOUNDARIES,
    select_spaced_boundaries,
)

BLOCK = 2048
# The boundaries recorded by the measured 234k prefill.
MEASURED = list(range(BLOCK, 233_472 + BLOCK, BLOCK))  # 114 entries
LATEST = MEASURED[-1]


def test_zero_budget_retains_everything():
    assert select_spaced_boundaries(MEASURED, LATEST, BLOCK, 0) == set(MEASURED)
    assert select_spaced_boundaries(MEASURED, LATEST, BLOCK, -1) == set(MEASURED)


def test_a_short_prompt_fits_the_budget_untouched():
    recorded = [BLOCK * i for i in range(1, 5)]
    assert select_spaced_boundaries(recorded, BLOCK * 4, BLOCK, keep=4) == set(recorded)


def test_measured_run_drops_almost_every_checkpoint():
    retained = select_spaced_boundaries(MEASURED, LATEST, BLOCK, keep=4)

    assert len(retained) <= 4 + 1
    assert len(MEASURED) - len(retained) >= 108  # ~12GB no longer held resident
    assert LATEST in retained


def test_newest_boundary_is_always_retained():
    for count in (9, 64, 1024, 8192):
        recorded = [BLOCK * i for i in range(1, count + 1)]
        assert recorded[-1] in select_spaced_boundaries(
            recorded, recorded[-1], BLOCK, 3
        )


def test_cost_is_bounded_independent_of_prefill_length():
    for count in (9, 64, 1024, 8192, 131_072):
        recorded = [BLOCK * i for i in range(1, count + 1)]
        retained = select_spaced_boundaries(recorded, recorded[-1], BLOCK, 3)
        assert len(retained) <= 3 + 1


def test_kept_checkpoints_are_evenly_spaced():
    keep = 4
    retained = sorted(select_spaced_boundaries(MEASURED, LATEST, BLOCK, keep=keep))

    step = retained[1] - retained[0]
    # Uniform interior spacing bounds the worst-case recompute gap at one step.
    assert [b - a for a, b in zip(retained, retained[1:])][:-1] == [step] * (
        len(retained) - 2
    )
    assert max(b - a for a, b in zip([0, *retained], retained)) <= step
    assert len(retained) >= 2  # a budget of 4 must not collapse to one point


def test_lattice_is_stable_as_the_prompt_grows():
    # The scheduler prunes in place, so the lattice has to be nested: growing the
    # prompt must never delete a checkpoint the wider lattice still wants, and
    # must not leave a hole spanning most of the prompt.
    recorded: list[int] = []
    for i in range(1, 513):
        newest = BLOCK * i
        recorded = sorted(
            select_spaced_boundaries([*recorded, newest], newest, BLOCK, keep=4)
        )
        assert recorded[-1] == newest
        assert len(recorded) <= 5

    gaps = [b - a for a, b in zip(recorded, recorded[1:])]
    assert max(gaps) <= recorded[-1] // 4
    assert len(recorded) >= 3


def test_never_returns_empty_for_a_non_empty_input():
    assert select_spaced_boundaries([BLOCK], BLOCK, BLOCK, keep=1) == {BLOCK}


def test_budget_is_capped_at_a_sane_ceiling():
    recorded = [BLOCK * i for i in range(1, 5000)]
    assert (
        len(select_spaced_boundaries(recorded, recorded[-1], BLOCK, 10_000))
        <= MAX_RETAINED_BOUNDARIES + 1
    )


def test_the_resume_boundary_survives_a_drifting_reference():
    """``limit`` is the cacheable depth, not the newest capture.

    A decoding request records boundaries past the end of its prompt, which the
    store filters out. Spacing the lattice over the cacheable limit instead of
    the newest capture keeps the boundary the next turn resumes from, even when
    the newest capture is far beyond it.
    """
    recorded = [BLOCK, 2 * BLOCK, 3 * BLOCK]  # 3 interior boundaries in range
    limit = 3 * BLOCK  # end of the cacheable prompt
    newest = 8 * BLOCK  # emitted tokens pushed the capture past it

    retained = select_spaced_boundaries(
        sorted([*recorded, newest]), limit, BLOCK, keep=1
    )

    assert limit in retained  # the resume point of the next turn
    assert newest in retained  # the store's latest_tc
    assert len(retained) <= 1 + 2
