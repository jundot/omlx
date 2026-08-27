from __future__ import annotations

import inspect
import json
from pathlib import Path

from benchmarks.bench_ds4_attention_finalizer import (
    SUPPORTED_HEADS,
    analysis_report,
    byte_ledger,
)
from omlx.custom_kernels.glm_moe_dsa import fast


def test_intermediate_traffic_ledger_matches_all_tp_head_shapes():
    assert SUPPORTED_HEADS == (24, 32, 40, 64)
    assert byte_ledger(24)["candidate_removed_mib"] == 50.0
    assert byte_ledger(32)["candidate_removed_mib"] == 66.0
    assert byte_ledger(40)["candidate_removed_mib"] == 82.0
    assert byte_ledger(64)["candidate_removed_mib"] == 130.0


def test_contract_requires_exact_boundaries_and_1_10x_combined():
    report = analysis_report()
    assert report["dispatches"] == {"current": 4, "candidate": 2}
    assert report["gate"] == {
        "normalized_array_equal": True,
        "rotated_array_equal": True,
        "minimum_combined_speedup": 1.10,
        "production_dispatch": False,
    }


def test_metal_source_freezes_mlx_reduction_rounding_and_rope_order():
    root = Path(__file__).parents[1]
    source = (
        root / "omlx/custom_kernels/glm_moe_dsa/csrc/ds4_attention_finalizer.metal"
    ).read_text()

    assert "kNormReads = 4" in source
    assert "max_total_threads_per_threadgroup(512)" in source
    assert "max_total_threads_per_threadgroup(128)" in source
    assert source.count("metal::precise::rsqrt") == 2
    assert "bfloat16_t(1.0f) * static_cast<bfloat16_t>" in source
    assert "weight[column] * static_cast<bfloat16_t>" in source
    assert "1.0f / freqs" in source
    assert "metal::fast::cos" in source
    assert "metal::fast::sin" in source


def test_symbols_are_built_and_reached_only_through_default_off_pair_gate():
    root = Path(__file__).parents[1]
    cmake = (root / "omlx/custom_kernels/glm_moe_dsa/csrc/CMakeLists.txt").read_text()
    bindings = (root / "omlx/custom_kernels/glm_moe_dsa/csrc/bindings.cpp").read_text()
    model = (root / "omlx/patches/deepseek_v4/deepseek_v4_model.py").read_text()

    assert "ds4_attention_finalizer.cpp" in cmake
    assert "ds4_attention_finalizer.metal" in cmake
    assert '"OMLX_DSV4_ATTN_FINALIZER_PREFILL", "0"' in model
    assert '"OMLX_DSV4_ATTN_FINALIZER_VERIFY", "0"' in model
    assert "_attention_finalizer_native_inputs" in model
    for symbol in ("ds4_q_head_rms_rope", "ds4_kv_rms_rope"):
        assert symbol in bindings
        assert symbol in fast.NATIVE_SYMBOLS
        assert symbol in model


def test_python_wrappers_expose_normalized_debug_boundary():
    q_signature = inspect.signature(fast.ds4_q_head_rms_rope)
    kv_signature = inspect.signature(fast.ds4_kv_rms_rope)
    assert q_signature.parameters["return_normalized"].default is False
    assert kv_signature.parameters["return_normalized"].default is False


def test_mlx_mit_notice_is_retained_with_adapted_source():
    root = Path(__file__).parents[1]
    notice = (root / "omlx/custom_kernels/glm_moe_dsa/csrc/MLX_LICENSE.txt").read_text()
    assert "MIT License" in notice
    assert "Copyright © 2023 Apple Inc." in notice
    assert "Permission is hereby granted" in notice


def test_recorded_real_gate_is_exact_and_above_threshold():
    root = Path(__file__).parents[1]
    result = json.loads(
        (
            root / "docs/experimental/ds4_attention_finalizer_results_2026-08-22.json"
        ).read_text()
    )
    assert result["all_boundaries_array_equal"] is True
    assert result["all_storage_bits_equal"] is True
    assert result["exceptional_signed_zero_inf_nan_bits_equal"] is True
    assert result["max_abs"] == 0.0
    assert result["gate"]["passed"] is True
    assert result["rank0_confirmation"]["combined_speedup"] >= 1.10
    assert result["gate"]["production_dispatch"] == "default_off"
    assert result["gate"]["environment"] == "OMLX_DSV4_ATTN_FINALIZER_PREFILL"
