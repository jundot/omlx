# SPDX-License-Identifier: Apache-2.0
"""Real-model calibration conversion (fake-driven, spec-grounded).

The calibration runs dev-1.5b on each Mac and converts its real throughput into
the planner's ComputeProfile. These tests pin the conversion physics and the
published-spec ordering (REVIEW-2026-11-13.md B10 / E4=B; the live-run notes):

* decode tok/s is bandwidth-proportional -> M2 Ultra > M5 Max > M1 Max.
* prefill tok/s is compute-proportional  -> M5 Max > M2 Ultra > M1 Max.

No real Mac/model here (AGENTS.md): timings are constructed, the conversion and
the real planner timing function are exercised. Live numbers live in the doc.
"""

from __future__ import annotations

import pytest

from omlx.cluster.assistant.calibration import (
    CALIBRATION_MAX_TOKENS,
    CALIBRATION_MODEL_REPO,
    CompletionTimings,
    compute_profile_from_timings,
)
from omlx.cluster.assistant.election import NodeBudget
from omlx.cluster.planner import _predict_stage_seconds

GIB = 1024**3
# dev-1.5b is ~1 GB of weights; use a round number so rates read as GB/s.
WEIGHT_BYTES = 1_000_000_000

# Spec-proportional real throughput for the SAME calibration model. Decode
# scales with bandwidth (400/614/800 GB/s); prefill scales with AI compute
# (M5 Max ~6x M1 Max, M2 Ultra ~2.6x). Prompt is fixed at 10 tokens.
DECODE_TOK_S = {"m1max": 400.0, "m5max": 614.0, "m2ultra": 800.0}
PREFILL_TOK_S = {"m1max": 1000.0, "m2ultra": 2600.0, "m5max": 6000.0}
PROMPT_TOKENS = 10


def _profile(node_id: str):
    return compute_profile_from_timings(
        CompletionTimings(
            decode_tok_per_s=DECODE_TOK_S[node_id],
            prefill_tok_per_s=PREFILL_TOK_S[node_id],
            prompt_tokens=PROMPT_TOKENS,
        ),
        model_weight_bytes=WEIGHT_BYTES,
        measured_at="2026-11-13T00:00:00+00:00",
    )


def test_calibration_model_is_the_reference_small_model() -> None:
    assert CALIBRATION_MODEL_REPO == "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    assert CALIBRATION_MAX_TOKENS >= 64  # enough tokens to mean something


def test_decode_rate_is_bandwidth_proportional() -> None:
    # decode reads the whole model once per token: rate = tok/s * weight bytes.
    profile = _profile("m2ultra")
    assert profile.decode_weight_bytes_per_second == pytest.approx(800e9)
    assert (
        _profile("m2ultra").decode_weight_bytes_per_second
        > _profile("m5max").decode_weight_bytes_per_second
        > _profile("m1max").decode_weight_bytes_per_second
    )


def test_prefill_rate_is_compute_proportional_and_prompt_normalized() -> None:
    # prefill reads the weights once per prompt: rate = tok/s * bytes / prompt.
    profile = _profile("m5max")
    assert profile.prefill_weight_bytes_per_second == pytest.approx(600e9)
    assert (
        _profile("m5max").prefill_weight_bytes_per_second
        > _profile("m2ultra").prefill_weight_bytes_per_second
        > _profile("m1max").prefill_weight_bytes_per_second
    )


def test_no_decode_rate_is_unknown_not_zero() -> None:
    assert (
        compute_profile_from_timings(
            CompletionTimings(decode_tok_per_s=0.0), model_weight_bytes=WEIGHT_BYTES
        )
        is None
    )


def test_missing_prefill_falls_back_to_decode_not_invented() -> None:
    profile = compute_profile_from_timings(
        CompletionTimings(decode_tok_per_s=500.0), model_weight_bytes=WEIGHT_BYTES
    )
    assert profile is not None
    # Documented conservative stand-in, not a fabricated prefill throughput.
    assert profile.prefill_weight_bytes_per_second == profile.decode_weight_bytes_per_second


def test_invalid_weight_bytes_rejected() -> None:
    with pytest.raises(ValueError):
        compute_profile_from_timings(
            CompletionTimings(decode_tok_per_s=500.0), model_weight_bytes=0
        )


def _budget(node_id: str) -> NodeBudget:
    profile = _profile(node_id)
    from omlx.cluster.performance import NodePerformanceProfile

    return NodeBudget(
        node_id=node_id,
        capacity_bytes=128 * GIB,
        reserve_bytes=64 * GIB,
        rank=1,
        memory_guard_tier="balanced",
        performance=NodePerformanceProfile(
            node_id=node_id,
            rank=1,
            decode_weight_bytes_per_second=profile.decode_weight_bytes_per_second,
            prefill_weight_bytes_per_second=profile.prefill_weight_bytes_per_second,
            collective_latency_seconds=0.0002,
            collective_bandwidth_bytes_per_second=4e9,
            backend="ring",
            measured_at="2026-11-13T00:00:00+00:00",
            samples=1,
        ),
    )


def test_real_calibration_orders_decode_heavy_by_bandwidth() -> None:
    compute = {
        node_id: _predict_stage_seconds(
            2 * GIB,
            _budget(node_id),
            activation_bytes_per_token=0,
            workload_profile="interactive",
            microbatch_size=1,
        )[0]
        for node_id in DECODE_TOK_S
    }
    # With prefill normalized, the decode-heavy workload still ranks bandwidth:
    # M2 Ultra fastest decode.
    assert compute["m2ultra"] < compute["m1max"]
    assert compute["m5max"] < compute["m1max"]


def test_real_calibration_orders_prefill_heavy_by_compute() -> None:
    compute = {
        node_id: _predict_stage_seconds(
            2 * GIB,
            _budget(node_id),
            activation_bytes_per_token=0,
            workload_profile="throughput",
            microbatch_size=1,
        )[0]
        for node_id in PREFILL_TOK_S
    }
    # Prefill-heavy ranks compute: M5 Max fastest, M1 Max slowest.
    assert compute["m5max"] < compute["m2ultra"] < compute["m1max"]
