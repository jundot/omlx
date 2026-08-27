from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

from benchmarks.bench_ds4_tp_stage_profile import (
    CATEGORIES,
    DS4TPShape,
    FINE_DETAIL_CATEGORIES,
    TimingRecorder,
    amdahl_table,
    modeled_collective_ms,
    normalize_to_observed_wall,
    optimization_scenarios,
    parse_args,
    profile_real_layers,
    projected_tps,
    required_group_speedup,
)


def test_three_five_shape_matches_exact_ds4_rank_geometry():
    m3 = DS4TPShape(rank=0)
    m5 = DS4TPShape(rank=1)

    assert (m3.local_heads, m3.local_intermediate) == (24, 768)
    assert (m5.local_heads, m5.local_intermediate) == (40, 1280)
    assert m3.route_rows == m5.route_rows == 6144


def test_collective_model_counts_two_activation_reductions_per_layer():
    shape = DS4TPShape(rank=1, tokens=1024, prefix_tokens=8192)
    report = modeled_collective_ms(
        shape,
        bandwidth_gbps=6.2,
        latency_us=30,
    )

    assert report["activation_all_sum_calls"] == 86
    assert report["activation_payload_bytes"] == 1024 * 4096 * 2
    assert report["indexer_all_gather_calls"] == 21
    assert report["indexer_payload_bytes_per_rank"] == 512 * 512 * 4
    assert report["total_ms"] == pytest.approx(
        report["activation_ms"] + report["indexer_ms"]
    )


def test_m6_verify_collective_geometry_has_no_short_context_indexer_exchange():
    shape = DS4TPShape(rank=1, tokens=6, prefix_tokens=65)
    report = modeled_collective_ms(shape, bandwidth_gbps=6.2, latency_us=30)

    assert shape.route_rows == 36
    assert report["activation_all_sum_calls"] == 86
    assert report["activation_payload_bytes"] == 6 * 4096 * 2
    assert report["indexer_all_gather_calls"] == 0
    assert report["total_ms"] == pytest.approx(3.2617858064516128)


def test_observed_normalization_preserves_wall_and_builds_amdahl_rows():
    compute = {
        "attention_projections": 30,
        "routed_moe_pair": 30,
        "routed_moe_down": 15,
        "indexer": 5,
        "misc": 20,
    }
    result = normalize_to_observed_wall(
        compute,
        collective_ms=100,
        tokens=1024,
        baseline_tps=628.76,
        target_tps=1000,
    )

    assert sum(result["attributed_ms"].values()) == pytest.approx(
        result["observed_wall_ms"]
    )
    assert sum(result["fractions"].values()) == pytest.approx(1.0)
    names = {row["component"] for row in result["amdahl"]}
    assert names == set(CATEGORIES) | {
        "routed_moe_total",
        "kernel_hotset",
        "kernel_hotset_plus_collectives",
    }


def test_amdahl_required_speedup_and_impossible_single_component():
    fractions = {
        "attention_projections": 0.30,
        "routed_moe_pair": 0.30,
        "routed_moe_down": 0.15,
        "indexer": 0.05,
        "collectives": 0.10,
        "misc": 0.10,
    }
    rows = {
        row["component"]: row
        for row in amdahl_table(
            fractions,
            baseline_tps=628.76,
            target_tps=1000,
        )
    }

    assert rows["attention_projections"]["target_possible_alone"] is False
    assert rows["routed_moe_total"]["target_possible_alone"] is True
    f = 0.45
    expected = f / (628.76 / 1000 - (1 - f))
    assert rows["routed_moe_total"]["required_speedup_for_target"] == pytest.approx(
        expected
    )


def test_joint_scenarios_solve_the_remaining_target_budget():
    fractions = {
        "attention_projections": 0.44,
        "routed_moe_pair": 0.24,
        "routed_moe_down": 0.11,
        "indexer": 0.035,
        "collectives": 0.075,
        "misc": 0.10,
    }
    required_moe = required_group_speedup(
        fractions,
        ("routed_moe_pair", "routed_moe_down"),
        baseline_tps=628.76,
        target_tps=1000,
        fixed_speedups={"attention_projections": 2.0},
    )
    speedups = {
        "attention_projections": 2.0,
        "routed_moe_pair": required_moe,
        "routed_moe_down": required_moe,
    }

    assert required_moe is not None
    assert projected_tps(fractions, speedups, baseline_tps=628.76) == pytest.approx(
        1000
    )
    scenarios = optimization_scenarios(
        fractions,
        baseline_tps=628.76,
        target_tps=1000,
    )
    assert scenarios[0]["projected_tps"] > 1000
    assert scenarios[-1]["required_group_speedup"] < 2


def test_timing_recorder_is_exclusive_and_subtracts_barrier_overhead():
    ticks = iter((100, 250))
    recorder = TimingRecorder(
        synchronize=lambda: None,
        evaluate=lambda _value: None,
        clock_ns=lambda: next(ticks),
        barrier_overhead_ns=50,
    )
    recorder.active = True

    result = recorder.call("indexer", lambda: "result")

    assert result == "result"
    assert recorder.nanoseconds["indexer"] == 100
    assert recorder.calls["indexer"] == 1


def test_timing_recorder_assigns_nested_work_to_the_outer_category():
    ticks = iter((100, 300))
    recorder = TimingRecorder(
        synchronize=lambda: None,
        evaluate=lambda _value: None,
        clock_ns=lambda: next(ticks),
    )
    recorder.active = True

    def outer():
        return recorder.call("attention_projections", lambda: "nested")

    assert recorder.call("indexer", outer) == "nested"
    assert recorder.nanoseconds == {"indexer": 200}
    assert recorder.calls == {"indexer": 1}


def test_invalid_shape_and_fraction_contracts_fail_closed():
    with pytest.raises(ValueError, match="sum to eight"):
        DS4TPShape(rank=0, shard_weights=(3, 4))
    with pytest.raises(ValueError, match="sum to one"):
        amdahl_table(
            {category: 0.1 for category in CATEGORIES},
            baseline_tps=628.76,
            target_tps=1000,
        )


def test_verify_and_fine_attribution_are_explicit_cli_modes(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["profile", "--model", "/tmp/model"])
    defaults = parse_args()
    assert defaults.dspark_verify is False
    assert defaults.fine_detail is False

    monkeypatch.setattr(
        sys,
        "argv",
        ["profile", "--model", "/tmp/model", "--dspark-verify", "--fine-detail"],
    )
    enabled = parse_args()
    assert enabled.dspark_verify is True
    assert enabled.fine_detail is True


def test_verify_profile_arms_exact_dispatch_and_rollback_inside_finally():
    source = inspect.getsource(profile_real_layers)
    arm_undo = source.index("cache_rollback.set_undo_armed(True)")
    arm_verify = source.index("set_dspark_verify_armed(True)")
    finally_block = source.index("finally:", arm_verify)
    disarm_verify = source.index("set_dspark_verify_armed(False)", finally_block)
    disarm_undo = source.index(
        "cache_rollback.set_undo_armed(False)", disarm_verify
    )
    assert arm_undo < arm_verify < finally_block < disarm_verify < disarm_undo


def test_fine_detail_taxonomy_covers_the_unattributed_verify_hotset():
    assert set(FINE_DETAIL_CATEGORIES) == {
        "router",
        "shared_moe",
        "hyperconnection",
        "norms",
        "attention_qkv_bank",
        "attention_q_b",
        "attention_core",
        "attention_output",
    }


def test_fine_detail_brackets_prefill_attention_implementations():
    source = inspect.getsource(
        __import__(
            "benchmarks.bench_ds4_tp_stage_profile",
            fromlist=["DS4LayerInstrumentation"],
        ).DS4LayerInstrumentation.__enter__
    )
    for symbol in (
        "scaled_dot_product_attention",
        "wsdpa_prefill",
        "wsdpa_topk_prefill",
    ):
        assert f'"{symbol}"' in source


def test_decode_profile_warms_cache_with_production_prefill_chunks():
    from benchmarks.bench_ds4_tp_stage_profile import _warm_layer_cache

    source = inspect.getsource(_warm_layer_cache)
    assert "step = min(1024" in source
    assert "step = shape.tokens" not in source


def test_profiler_is_not_imported_or_dispatched_by_production_code():
    root = Path(__file__).parents[1]
    symbol = "bench_ds4_tp_stage_profile"
    assert all(
        symbol not in path.read_text() for path in (root / "omlx").rglob("*.py")
    )
    head_symbol = "bench_ds4_mtp_vocab_head_tiles"
    assert all(
        head_symbol not in path.read_text()
        for path in (root / "omlx").rglob("*.py")
    )
