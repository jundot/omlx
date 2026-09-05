# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the LRC analysis math (bench/lrc_analysis.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

from lrc_analysis import load_trace, sch, srp  # noqa: E402


def test_sch_alternating_period_two_one_slot():
    # A,B,A,B,... with 1 slot: Belady keeps A (recurs every 2 calls) over
    # each one-need-away B whose next use is strictly farther — the best a
    # single slot can do is every A call after the first (19/40).
    seq = [frozenset({0}), frozenset({1})] * 20
    rate, hits, needed = sch(seq, cache_size=1)
    assert hits == 19
    assert needed == 40
    assert abs(rate - 19 / 40) < 1e-9


def test_sch_constant_expert_warms_after_first_call():
    seq = [frozenset({5})] * 10
    rate, hits, needed = sch(seq, cache_size=1)
    assert hits == 9
    assert needed == 10
    assert abs(rate - 0.9) < 1e-9


def test_sch_zero_locality_degrades_to_capacity():
    # Every call needs a brand-new expert: hit rate stays 0 for any size.
    seq = [frozenset({i}) for i in range(50)]
    rate, hits, _ = sch(seq, cache_size=8)
    assert hits == 0 and rate == 0.0


def test_sch_mixed_pattern_beats_lru_baseline():
    # One hot expert every other call, cold experts otherwise: the oracle
    # cache of 1 slot keeps the hot expert whenever the next call allows.
    seq = []
    for i in range(40):
        seq.append(frozenset({0}))
        seq.append(frozenset({100 + i}))
    rate, hits, needed = sch(seq, cache_size=1)
    # After serving A at call 2k, next call needs X (next use: never again);
    # after serving X, next call needs A (next use: 1 ahead) -> retained.
    # So every A call after the first hits.
    assert hits >= needed // 2 - 1
    assert rate > 0.45


def test_sch_never_again_evicted_first():
    # Hot expert H alternates with distinct cold experts; a 2-slot oracle
    # cache keeps H and the single most-soon-needed expert.
    seq = []
    for i in range(20):
        seq.append(frozenset({0, 50 + i}))  # H + a one-shot expert
    rate, hits, needed = sch(seq, cache_size=2)
    # H is retained throughout (needed every call); one-shots are misses
    # and the very first call is a miss too.
    assert hits == 19
    assert needed == 40
    assert abs(rate - 19 / 40) < 1e-9


def test_srp_demand_weighted_reflects_frequency():
    # Segment where expert 0 is demanded by every call: the top-1 fixed
    # group covers half the demand mass (3/6) but only 1/4 of the union.
    seq = [frozenset({0, 1}), frozenset({0, 2}), frozenset({0, 3})]
    covs = srp(seq, group_size=1, segment=10)
    demand, distinct = covs[0]
    assert abs(demand - 0.5) < 1e-9  # expert 0 = 3 of 6 demand slots
    assert abs(distinct - 1 / 4) < 1e-9  # union = {0,1,2,3}


def test_load_trace_orders_calls_and_layers():
    import tempfile

    rows = [
        {"call": 2, "layer": 0, "positions": 1, "uniq": [3]},
        {"call": 1, "layer": 1, "positions": 1, "uniq": [1]},
        {"call": 1, "layer": 0, "positions": 1, "uniq": [0]},
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for r in rows:
            f.write(__import__("json").dumps(r) + "\n")
        path = f.name
    per_layer = load_trace(path)
    assert per_layer[0] == [frozenset({0}), frozenset({3})]
    assert per_layer[1] == [frozenset({1})]
