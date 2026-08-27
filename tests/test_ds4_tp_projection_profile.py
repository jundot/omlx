from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.bench_ds4_tp_projection_profile import (
    _candidate_result,
    _projected_tps,
)
from benchmarks.bench_ds4_tp_stage_profile import (
    ATTENTION_PROJECTION_DETAILS,
    INDEXER_PROJECTION_DETAILS,
    PROJECTION_DETAIL_CATEGORIES,
)


def test_projection_schema_covers_attention_compressor_indexer_and_output():
    assert ATTENTION_PROJECTION_DETAILS == (
        "q_a",
        "q_b",
        "raw_wkv",
        "compressor_wkv",
        "compressor_gate",
        "o_a",
        "o_b",
    )
    assert INDEXER_PROJECTION_DETAILS == (
        "indexer_q",
        "indexer_weights",
        "indexer_compressor_wkv",
        "indexer_compressor_gate",
    )
    assert len(PROJECTION_DETAIL_CATEGORIES) == len(
        set(PROJECTION_DETAIL_CATEGORIES)
    )


def test_candidate_rank_requires_exact_parity_before_projecting_tps():
    timings = {
        "separate": {"median_ms": 10.0, "min_ms": 9.0, "max_ms": 11.0},
        "concat": {"median_ms": 5.0, "min_ms": 4.0, "max_ms": 6.0},
    }
    fractions = {name: 0.0 for name in PROJECTION_DETAIL_CATEGORIES}
    fractions["q_a"] = 0.1
    exact = _candidate_result(
        "q",
        "separate",
        "concat",
        timings,
        {"array_equal": True, "slices": [True]},
        ("q_a",),
        fractions,
        baseline_tps=600,
    )
    drifted = _candidate_result(
        "q",
        "separate",
        "concat",
        timings,
        {"array_equal": False, "slices": [False]},
        ("q_a",),
        fractions,
        baseline_tps=600,
    )

    assert exact["local_speedup"] == 2.0
    assert exact["projected_tps"] == pytest.approx(600 / 0.95)
    assert exact["eligible"] is True
    assert drifted["projected_tps"] is None
    assert drifted["eligible"] is False


def test_projection_amdahl_uses_exposed_wall_not_projection_bucket_share():
    # A 2x local win on a projection that owns 20% of end-to-end wall is 1.11x,
    # not 2x and not a projection-bucket-local claim.
    assert _projected_tps(0.2, 2.0, 600.0) == pytest.approx(600 / 0.9)


def test_projection_harness_stays_out_of_production_dispatch():
    root = Path(__file__).parents[1]
    symbol = "bench_ds4_tp_projection_profile"
    assert all(
        symbol not in path.read_text() for path in (root / "omlx").rglob("*.py")
    )
