from __future__ import annotations

import json
from pathlib import Path

from benchmarks.bench_ds4_qkv_compressor_bundle import (
    LAYER_RATIOS,
    M1024_ROLLBACK_ENV,
    NATIVE_B1_SYMBOL,
    byte_ledger,
    dispatch_ledger,
    packed_slices,
    projections,
    promotion_contract,
)


def test_real_ds4_layer_schedule_and_projection_extents():
    assert {ratio: LAYER_RATIOS.count(ratio) for ratio in (0, 4, 128)} == {
        0: 2,
        4: 21,
        128: 20,
    }
    assert [item.rows for item in projections(4)] == [1024, 512, 1024, 1024, 256, 256]
    assert [item.rows for item in projections(128)] == [1024, 512, 512, 512]
    assert packed_slices(4)["indexer.compressor.wgate"] == (3840, 4096)


def test_checkpoint_and_runtime_byte_ledger_is_exact():
    ledger = byte_ledger()
    assert ledger["per_layer"]["0"]["checkpoint_bytes"] == 6_488_064
    assert ledger["per_layer"]["4"]["checkpoint_bytes"] == 27_459_584
    assert ledger["per_layer"]["128"]["checkpoint_bytes"] == 14_876_672
    assert ledger["all_layers_checkpoint_bytes"] == 887_160_832
    assert ledger["all_layers_runtime_bytes"] == 887_160_832


def test_dispatch_ledger_is_per_rank_and_collective_neutral():
    ledger = dispatch_ledger()
    assert ledger["current_projection_dispatches_per_rank"] == 210
    assert ledger["two_bank_dispatches_per_rank"] == 84
    assert ledger["full_bundle_dispatches_per_rank"] == 43
    assert ledger["full_bundle_dispatches_saved_per_rank"] == 167
    assert ledger["collectives_changed"] == 0


def test_first_native_abi_stops_before_ape_and_cache_mutation():
    contract = promotion_contract()
    assert contract["first_native_symbol"] == NATIVE_B1_SYMBOL
    assert contract["shape"] == {"batch": 1, "rows": 1, "hidden": 4096, "ratio": 4}
    assert contract["output"] == "packed_bf16[1,4096]"
    assert contract["forbidden_inputs"] == [
        "ape",
        "cache",
        "position",
        "distributed_group",
    ]
    assert contract["parity"]["decode_positions"] == [0, 1, 2, 3]


def test_m1024_candidate_is_lossless_rank_local_and_reversible():
    candidate = promotion_contract()["m1024_production_candidate"]
    assert candidate["shape"] == {
        "batch": 1,
        "rows": 1024,
        "hidden": 4096,
        "ratio": 4,
    }
    assert candidate["storage"]["requantization"] is False
    assert candidate["storage"]["steady_state_duplicate_weight_bytes"] == 0
    assert candidate["dispatches"] == {
        "Apple M3 Ultra": 3,
        "Apple M5 Max": 4,
    }
    assert candidate["single_node"] is True
    assert candidate["tp2_rank_local"] is True
    assert candidate["collectives_changed"] == 0
    assert candidate["rollback_env"] == M1024_ROLLBACK_ENV
    assert candidate["default_enabled"] is False


def test_symbol_has_only_the_exact_promoted_production_seam():
    root = Path(__file__).parents[1]
    model = root / "omlx/patches/deepseek_v4/deepseek_v4_model.py"
    allowed = {
        root / "omlx/custom_kernels/glm_moe_dsa/fast.py",
        model,
    }
    hits = []
    for path in (root / "omlx").rglob("*.py"):
        if path not in allowed and NATIVE_B1_SYMBOL in path.read_text():
            hits.append(path)
    assert hits == []
    source = model.read_text()
    assert '"OMLX_DSV4_QKV_BUNDLE_DECODE", "1"' in source
    assert "_decode_qkv_projection_bundle(self, x)" in source


def test_recorded_b1_gate_is_lossless_but_rejected_on_both_hosts():
    report = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/experimental/ds4_qkv_compressor_bundle_b1_results_2026-08-22.json"
        ).read_text()
    )
    assert report["dispatches"] == {
        "separate": 6,
        "grouped_stock": 2,
        "native": 2,
    }
    for host in report["hosts"].values():
        assert host["all_six_slices_exact_vs_separate"] is True
        assert host["all_six_slices_exact_vs_grouped"] is True
        assert host["speedup_vs_faster_baseline"] < 1.05
        assert host["passed"] is False
    assert report["outcome"]["promoted"] is False
    assert report["outcome"]["production_dispatch"] is False


def test_recorded_m1024_gate_preserves_storage_and_passes_both_hosts():
    report = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/experimental"
            / "ds4_qkv_compressor_bundle_m1024_results_2026-08-23.json"
        ).read_text()
    )
    assert report["storage"]["requantized"] is False
    assert report["storage"]["steady_state_duplicate_weight_bytes"] == 0
    assert report["rejected_m5_three_dispatch_schedule"]["projection_array_equal"] == [
        True,
        True,
        False,
        False,
        True,
        True,
    ]
    for host in report["hosts"].values():
        assert host["projection_array_equal"] == [True] * 6
        assert host["fallback_views_array_equal"] == [True] * 6
        assert host["speedup"] >= 1.05
    assert report["gate"]["passed"] is True
    assert report["outcome"]["cluster_default_enabled"] is False
