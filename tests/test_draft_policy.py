"""Draft policy: source and window by expected tokens per millisecond."""

from __future__ import annotations

import itertools

from omlx.speculative.draft_policy import (
    Choice,
    DraftPolicy,
    Source,
    allocate_batch_verify_budget,
)

# Inkling 2 Mini verify rounds on M5 Pro, ms for 1..8 rows.
M5_PRO = (0.0, 26.6, 39.0, 46.7, 54.7, 62.8, 70.9, 79.0, 87.1)


def _policy(**kwargs):
    return DraftPolicy(7, timed=False, cost_shape=M5_PRO, draft_ratio=0.2, **kwargs)


def test_source_follows_the_content():
    """A long match takes a wide lookup window, a short noisy one leaves the
    window to the drafter, and plain decoding runs once drafts stop paying."""
    policy = _policy()
    quoted = policy.choose(lookup_k=7, match=20, drafter_k=7)
    assert quoted.source is Source.LOOKUP and quoted.k >= 5
    # A trigram match proposes as many tokens, but its window would cost more
    # than the drafter's for fewer expected tokens.
    novel = policy.choose(lookup_k=7, match=3, drafter_k=7)
    assert novel.source is Source.DRAFTER and novel.k == 2
    for _ in range(40):
        policy.observe(policy.choose(drafter_k=7), 0)
    parked = policy.choose(drafter_k=7)
    assert parked.source is Source.NONE and parked.k == 0
    # A lookup still wins from plain decoding when its match is long.
    assert policy.choose(lookup_k=7, match=20, drafter_k=7).source is Source.LOOKUP


def test_parked_drafter_is_measured_again():
    """Nothing is learned about the drafter while no draft is verified, so one
    is verified again after a stretch of plain rounds, and good drafts return."""
    policy = _policy()
    for _ in range(40):
        policy.observe(policy.choose(drafter_k=7), 0)
    sources = []
    for _ in range(400):
        choice = policy.choose(drafter_k=7)
        sources.append(choice.source)
        policy.observe(choice, choice.k)  # every draft verified is accepted
    assert sources[0] is Source.NONE
    assert sources[-1] is Source.DRAFTER and policy.choose(drafter_k=7).k >= 3


def test_lookup_acceptance_is_learned_per_match_length():
    policy = _policy()
    before = policy.lookup_rate(3), policy.lookup_rate(20)
    for _ in range(30):
        policy.observe(Choice(Source.LOOKUP, 4, match=3), 0)
        policy.observe(Choice(Source.LOOKUP, 7, match=20), 7)
    assert policy.lookup_rate(3) < 0.1 < before[0]
    assert policy.lookup_rate(20) > 0.97 > before[1]
    # Buckets: 3-4, 5-7, 8-15, 16 and more.
    assert policy.lookup_rate(4) == policy.lookup_rate(3)
    assert len({policy.lookup_rate(m) for m in (4, 5, 8, 16)}) == 4
    assert policy.lookup_rate(16) == policy.lookup_rate(32)


def test_min_gain_and_margin_guard_the_switch():
    # One draft accepted 70% of the time beats plain decoding by 2% here: not
    # enough to start drafting, enough to go on once the drafter runs.
    policy = _policy(position_prior=(0.7,), min_gain=0.05)
    assert policy.choose(drafter_k=1).source is Source.NONE
    assert _policy(position_prior=(0.7,), min_gain=0.0).choose(drafter_k=1).k == 1
    policy.observe(Choice(Source.DRAFTER, 1), 1)
    assert policy.choose(drafter_k=1).source is Source.DRAFTER
    policy.observe(Choice(Source.NONE, 0), 0)
    assert policy.choose(drafter_k=1).source is Source.NONE
    # The host's own depth stays, however weak, until a lookup beats it.
    held = policy.choose(drafter_k=2, drafter_window=2)
    assert (held.source, held.k) == (Source.DRAFTER, 2)
    # A lookup window worth 1.5% more than the drafter's does not take over.
    for margin, source in ((0.03, Source.DRAFTER), (0.0, Source.LOOKUP)):
        even = _policy(position_prior=(0.8,), margin=margin)
        for _ in range(200):
            for accepted in (2, 2, 1, 0):
                even.observe(Choice(Source.LOOKUP, 2, match=20), accepted)
        assert 0.70 < even.lookup_rate(20) < 0.73
        assert even.choose(lookup_k=2, match=20, drafter_k=2).source is source
    # Nor does a lookup worth 2% over plain decoding, with no drafter to beat.
    for min_gain, source in ((0.05, Source.NONE), (0.0, Source.LOOKUP)):
        thin = _policy(min_gain=min_gain)
        for _ in range(200):
            for accepted in (1, 0):
                thin.observe(Choice(Source.LOOKUP, 1, match=3), accepted)
        assert thin.choose(lookup_k=2, match=3).source is source
    # With the drafter's block already made, the lookup saves nothing of it.
    saved = _policy(position_prior=(0.92,))
    assert saved.choose(lookup_k=3, match=20, drafter_k=3).source is Source.LOOKUP
    assert (
        saved.choose(lookup_k=3, match=20, drafter_k=3, drafted=True).source
        is Source.DRAFTER
    )


def test_stated_chances_pick_the_window_and_are_calibrated():
    policy = _policy()
    assert policy.window([0.99, 0.99, 0.98, 0.3, 0.2, 0.2, 0.2]) == 4
    assert policy.window([0.2] * 7) == 1
    assert policy.window([0.99] * 7) >= 7
    # Chances stated as near sure and rejected half the time stop buying rows.
    for _ in range(100):
        choice = Choice(Source.DRAFTER, 3)
        policy.observe(choice, 0, conf=[0.99, 0.99, 0.99])
        policy.observe(choice, 3, conf=[0.99, 0.99, 0.99])
    assert 0.7 < policy.chance_rate(0.99) < 0.8
    assert policy.chance_rate(0.6) == 0.6  # another bin keeps its stated chance
    assert policy.window([0.99] * 7) < 7


def test_costs_are_measured_and_the_drafter_cost_is_split_off():
    """Round times replace the prior shape; a drafter round's extra time over
    a draft-free round of the same size is the drafter's cost, which is what a
    lookup saves; a slowdown moves every size at once."""
    policy = DraftPolicy(7)
    verify = {1: 30.0, 2: 50.0, 4: 80.0, 8: 150.0}
    for _ in range(60):
        for rows, ms in verify.items():
            policy.observe(Choice(Source.LOOKUP, rows - 1, match=20), rows - 1, ms)
            policy.observe(Choice(Source.DRAFTER, rows - 1), 0, ms + 9.0)
    for rows, ms in verify.items():
        assert abs(policy.cost_ms(rows) - ms) < 0.05 * ms
    assert abs(policy.cost_ms(2, drafted=True) - policy.cost_ms(2) - 9.0) < 1.5
    # Sizes never run lie between their neighbours.
    assert policy.cost_ms(2) < policy.cost_ms(3) < policy.cost_ms(4)
    assert policy.cost_ms(4) < policy.cost_ms(6) < policy.cost_ms(8)
    # The machine gets busy: rounds of one size take twice as long.
    for _ in range(30):
        policy.observe(Choice(Source.LOOKUP, 1, match=20), 1, 100.0)
    assert policy.cost_ms(8) > 1.5 * 150.0


def test_unmeasured_window_is_tried_before_its_guess_can_veto_it():
    policy = DraftPolicy(7)
    for _ in range(20):  # only small windows measured, and rows look dear
        policy.observe(Choice(Source.LOOKUP, 1, match=20), 1, 60.0)
        policy.observe(Choice(Source.NONE, 0), 0, 20.0)
    for _ in range(40):
        policy.observe(Choice(Source.LOOKUP, 1, match=20), 1)
    probes = 0
    for _ in range(10):
        choice = policy.choose(lookup_k=7, match=20)
        probes += choice.probe
        if choice.k == 7:
            policy.observe(choice, 7, 24.0)  # wide windows are cheap after all
    assert probes == 3
    wide = policy.choose(lookup_k=7, match=20)
    assert wide.k == 7 and not wide.probe


def test_joint_budget_spends_rows_where_drafts_survive():
    def cost(rows):
        return M5_PRO[rows] if rows <= 4 else 54.7 + 8.1 * (rows - 4)

    quoted = [0.95**i for i in range(1, 8)]
    prose = [0.68, 0.47, 0.28, 0.18]
    # Any row may join the pass: the quoting stream takes nearly all of them.
    assert allocate_batch_verify_budget([quoted, prose], cost) == [7, 1]
    # One window size for every stream: narrow, the weak stream pays per row.
    assert allocate_batch_verify_budget([quoted, prose], cost, rectangular=True) == [
        2,
        2,
    ]
    assert allocate_batch_verify_budget([[0.1, 0.01]] * 2, cost) == [0, 0]
    assert allocate_batch_verify_budget([], cost) == []

    def rate(counts, streams):
        expected = sum(1 + sum(s[:k]) for s, k in zip(streams, counts))
        return expected / cost(len(streams) + sum(counts))

    # Against every allocation of three streams.
    streams = [quoted[:4], prose, [0.9, 0.5, 0.45, 0.1]]
    best = max(itertools.product(range(5), repeat=3), key=lambda c: rate(c, streams))
    got = allocate_batch_verify_budget(streams, cost)
    assert abs(rate(got, streams) - rate(best, streams)) < 1e-12
    # A gain under the guard is not worth leaving plain decoding for.
    assert allocate_batch_verify_budget([[0.6], [0.6]], cost, min_gain=0.5) == [0, 0]


def test_fair_budget_does_not_starve_the_weaker_stream():
    """The most tokens in total gives the quoting stream every row, and the
    prose stream then commits one token per long pass: under its plain rate.
    The fair budget keeps both streams above plain decoding."""

    def cost(rows):
        return M5_PRO[rows] if rows <= 4 else 54.7 + 8.1 * (rows - 4)

    streams = [[0.95**i for i in range(1, 8)], [0.68, 0.47, 0.28, 0.18]]

    def rates(counts):
        ms = cost(2 + sum(counts))
        return [(1 + sum(s[:k])) / ms for s, k in zip(streams, counts)]

    plain = 1 / cost(2)
    greedy = allocate_batch_verify_budget(streams, cost)
    fair = allocate_batch_verify_budget(streams, cost, fair=True)
    assert rates(greedy)[1] < 0.7 * plain
    assert min(rates(fair)) > plain
    assert fair[0] > fair[1] >= 1
    # Of the allocations that keep both streams at plain decoding's rate or
    # better, the fair one has the largest product of rates.
    best = max(
        (c for c in itertools.product(range(8), range(5)) if min(rates(c)) >= plain),
        key=lambda c: rates(c)[0] * rates(c)[1],
    )
    assert rates(fair)[0] * rates(fair)[1] >= 0.999 * rates(best)[0] * rates(best)[1]
    assert allocate_batch_verify_budget([[0.1], [0.1]], cost, fair=True) == [0, 0]
