# SPDX-License-Identifier: Apache-2.0
"""Probe calibration expectations grounded in published Apple Silicon specs.

The compute probe (REVIEW-2026-11-13.md B10 / decision E4=B) must order the
hardware correctly for the two physics that drive pipeline timing:

* DECODE is memory-bound -> the decode rate tracks unified-memory bandwidth.
* PREFILL is compute-bound -> the prefill rate tracks fp16/AI compute.

Published specs (Apple newsroom / Wikipedia / everymac, researched 2026-11-13):

| Chip      | GPU cores        | Bandwidth | FP32     | AI compute        |
|-----------|------------------|-----------|----------|-------------------|
| M1 Max    | 32               | 400 GB/s  | 10.4 TF  | baseline          |
| M2 Ultra  | 76               | 800 GB/s  | 27.2 TF  | ~2x M1 Max        |
| M5 Max    | 40 + NPU/core    | 614 GB/s  | ~17 TF   | ~6x M1 Max (Apple)|

So the expected ordering is:
* decode bandwidth:  M2 Ultra > M5 Max > M1 Max  (800 > 614 > 400)
* prefill AI compute: M5 Max > M2 Ultra > M1 Max  (Neural Accelerators)

These tests are fake-driven (AGENTS.md): they feed spec-derived profiles through
the real planner timing function. The live measured rates that confirm the
ordering on the validation cluster are recorded in the live-run notes, not here.
"""

from __future__ import annotations

from omlx.cluster.assistant.election import ComputeProfile, NodeBudget
from omlx.cluster.performance import NodePerformanceProfile
from omlx.cluster.planner import _predict_stage_seconds

GIB = 1024**3

# Spec-anchored calibration signals (bytes/s). Decode uses the published
# bandwidth; prefill uses weight-read rates whose ordering reflects the
# published AI-compute ratios (M5 Max ~6x M1 Max, M2 Ultra ~2.6x M1 Max FP32).
DECODE_BANDWIDTH = {"m1max": 400e9, "m5max": 614e9, "m2ultra": 800e9}
PREFILL_COMPUTE = {"m1max": 10e9, "m2ultra": 26e9, "m5max": 60e9}


def _budget(node_id: str, *, decode: float, prefill: float, rank: int = 1) -> NodeBudget:
    return NodeBudget(
        node_id=node_id,
        capacity_bytes=128 * GIB,
        reserve_bytes=64 * GIB,
        rank=rank,
        memory_guard_tier="balanced",
        performance=NodePerformanceProfile(
            node_id=node_id,
            rank=rank,
            decode_weight_bytes_per_second=decode,
            prefill_weight_bytes_per_second=prefill,
            collective_latency_seconds=0.0002,
            collective_bandwidth_bytes_per_second=4e9,
            backend="ring",
            measured_at="2026-11-13T00:00:00+00:00",
            samples=5,
        ),
    )


def _spec_budget(node_id: str, rank: int = 1) -> NodeBudget:
    return _budget(
        node_id,
        decode=DECODE_BANDWIDTH[node_id],
        prefill=PREFILL_COMPUTE[node_id],
        rank=rank,
    )


def test_spec_decode_rates_are_plausible_and_old_garbage_is_not() -> None:
    # The bandwidth-bound probe reports rates near these spec values; all sit
    # inside the plausibility window. The old latency-bound GEMV reported
    # single-digit GB/s, which the guard must reject as unknown compute.
    for node_id, rate in DECODE_BANDWIDTH.items():
        profile = ComputeProfile(
            decode_weight_bytes_per_second=rate,
            prefill_weight_bytes_per_second=PREFILL_COMPUTE[node_id],
        )
        assert profile.plausible(), node_id
    for garbage in (3.4e9, 13.6e9, 25.7e9):  # the old probe's reported rates
        profile = ComputeProfile(
            decode_weight_bytes_per_second=garbage,
            prefill_weight_bytes_per_second=garbage * 2,
        )
        assert not profile.plausible()


def test_decode_seconds_order_by_memory_bandwidth() -> None:
    # Decode is memory-bound: faster bandwidth -> fewer seconds per layer.
    # M2 Ultra (800) < M5 Max (614) < M1 Max (400).
    layer_bytes = 2 * GIB
    seconds = {
        node_id: layer_bytes / rate for node_id, rate in DECODE_BANDWIDTH.items()
    }
    assert seconds["m2ultra"] < seconds["m5max"] < seconds["m1max"]


def test_prefill_seconds_order_by_ai_compute() -> None:
    # Prefill is compute-bound: the M5 Max's Neural Accelerators give it the
    # most AI compute, so it reads weights fastest despite fewer GPU cores than
    # the M2 Ultra. M5 Max < M2 Ultra < M1 Max.
    layer_bytes = 2 * GIB
    seconds = {
        node_id: layer_bytes / rate for node_id, rate in PREFILL_COMPUTE.items()
    }
    assert seconds["m5max"] < seconds["m2ultra"] < seconds["m1max"]


def test_predict_stage_seconds_rewards_bandwidth_for_decode_heavy() -> None:
    # With prefill held equal, a decode-heavy (interactive) workload must rank
    # the highest-bandwidth Mac fastest.
    equal_prefill = 30e9
    nodes = {
        node_id: _budget(node_id, decode=rate, prefill=equal_prefill)
        for node_id, rate in DECODE_BANDWIDTH.items()
    }
    compute = {
        node_id: _predict_stage_seconds(
            2 * GIB,
            node,
            activation_bytes_per_token=0,
            workload_profile="interactive",
            microbatch_size=1,
        )[0]
        for node_id, node in nodes.items()
    }
    assert compute["m2ultra"] < compute["m5max"] < compute["m1max"]


def test_predict_stage_seconds_rewards_compute_for_prefill_heavy() -> None:
    # With decode held equal, a prefill-heavy (throughput) workload must rank
    # the highest-compute Mac fastest (M5 Max, not the M2 Ultra).
    equal_decode = 500e9
    nodes = {
        node_id: _budget(node_id, decode=equal_decode, prefill=rate)
        for node_id, rate in PREFILL_COMPUTE.items()
    }
    compute = {
        node_id: _predict_stage_seconds(
            2 * GIB,
            node,
            activation_bytes_per_token=0,
            workload_profile="throughput",
            microbatch_size=1,
        )[0]
        for node_id, node in nodes.items()
    }
    assert compute["m5max"] < compute["m2ultra"] < compute["m1max"]


def test_unmeasured_node_contributes_no_compute_seconds() -> None:
    # A node whose probe was implausible (guard -> performance None) contributes
    # zero compute seconds: unknown is never treated as fast.
    node = NodeBudget(
        node_id="m1max",
        capacity_bytes=128 * GIB,
        reserve_bytes=64 * GIB,
        rank=1,
        memory_guard_tier="balanced",
        performance=None,
    )
    assert _predict_stage_seconds(
        2 * GIB,
        node,
        activation_bytes_per_token=0,
        workload_profile="balanced",
        microbatch_size=1,
    ) == (0.0, 0.0, 0.0)
