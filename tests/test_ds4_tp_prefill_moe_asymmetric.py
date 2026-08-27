import json
from pathlib import Path

from benchmarks.bench_ds4_tp_prefill_moe_asymmetric import (
    SHARD_WEIGHTS,
    shard_bounds,
)


def test_exact_three_five_tp_slice_boundaries():
    assert SHARD_WEIGHTS == (3, 5)
    assert shard_bounds(0) == (0, 768)
    assert shard_bounds(1) == (768, 2048)
    assert (768 // 8, 768 // 32) == (96, 24)
    assert ((2048 - 768) // 8, (2048 - 768) // 32) == (160, 40)


def test_explicit_equal_tp_slice_boundaries_are_available_to_benchmarks():
    assert shard_bounds(0, (4, 4)) == (0, 1024)
    assert shard_bounds(1, (4, 4)) == (1024, 2048)


def test_tail8_native_guards_accept_only_exact_ds4_tp_widths():
    source = (
        Path(__file__).parents[1]
        / "omlx/custom_kernels/glm_moe_dsa/csrc/ds4_prefill_moe.cpp"
    ).read_text()
    assert "intermediate != 768" in source
    assert "intermediate != 1024" in source
    assert "intermediate != 1280" in source


def test_recorded_asymmetric_gate_is_exact_and_over_one_point_ten():
    results = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/experimental/"
            "ds4_tp_prefill_moe_asymmetric_results_2026-08-22.json"
        ).read_text()
    )
    assert results["tp"]["local_intermediate"] == 768
    assert results["dominant"]["primitive"] == "pair"
    assert results["gate"]["all_boundaries_exact"] is True
    assert results["gate"]["both_tail8_runs_pass"] is True
    assert all(
        run["full_speedup"] >= 1.10 for run in results["tail8_vs_variant2"]
    )
    assert results["decision"]["production_dispatch"] == "off"
