# SPDX-License-Identifier: Apache-2.0
"""Capability and role planning for full-replica prefill/decode serving."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DisaggregatedNodeProfile:
    """One node's admitted full-replica capacity and measured phase rates."""

    node_id: str
    admission_budget_bytes: int
    model_resident_bytes: int
    prefill_tokens_per_second: float
    decode_tokens_per_second: float

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("disaggregated node profile requires a node id")
        if self.admission_budget_bytes <= 0 or self.model_resident_bytes <= 0:
            raise ValueError("disaggregated node memory values must be positive")
        if (
            not math.isfinite(self.prefill_tokens_per_second)
            or self.prefill_tokens_per_second <= 0
            or not math.isfinite(self.decode_tokens_per_second)
            or self.decode_tokens_per_second <= 0
        ):
            raise ValueError("disaggregated phase rates must be finite and positive")


@dataclass(frozen=True)
class DisaggregatedWorkload:
    prompt_tokens: int
    completion_tokens: int
    cache_transfer_bytes: int
    fabric_bytes_per_second: float
    handoff_fixed_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.prompt_tokens < 1 or self.completion_tokens < 1:
            raise ValueError("disaggregated workload token counts must be positive")
        if self.cache_transfer_bytes < 0:
            raise ValueError("cache transfer bytes cannot be negative")
        if (
            not math.isfinite(self.fabric_bytes_per_second)
            or self.fabric_bytes_per_second <= 0
            or not math.isfinite(self.handoff_fixed_seconds)
            or self.handoff_fixed_seconds < 0
        ):
            raise ValueError("disaggregated fabric values are invalid")


@dataclass(frozen=True)
class DisaggregatedOrientation:
    prefill_node_id: str
    decode_node_id: str
    prefill_seconds: float
    handoff_seconds: float
    decode_seconds: float
    first_request_seconds: float
    steady_interval_seconds: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DisaggregatedPlan:
    supported: bool
    recommended: bool
    reason: str
    orientation: DisaggregatedOrientation | None
    best_single_node_id: str | None
    best_single_interval_seconds: float | None
    estimated_steady_speedup: float | None
    candidates: tuple[DisaggregatedOrientation, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "supported": self.supported,
            "recommended": self.recommended,
            "reason": self.reason,
            "orientation": (
                self.orientation.to_dict() if self.orientation is not None else None
            ),
            "best_single_node_id": self.best_single_node_id,
            "best_single_interval_seconds": self.best_single_interval_seconds,
            "estimated_steady_speedup": self.estimated_steady_speedup,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def _single_interval(
    node: DisaggregatedNodeProfile, workload: DisaggregatedWorkload
) -> float:
    return (
        workload.prompt_tokens / node.prefill_tokens_per_second
        + workload.completion_tokens / node.decode_tokens_per_second
    )


def _orientation(
    prefill: DisaggregatedNodeProfile,
    decode: DisaggregatedNodeProfile,
    workload: DisaggregatedWorkload,
) -> DisaggregatedOrientation:
    prefill_seconds = (
        workload.prompt_tokens / prefill.prefill_tokens_per_second
    )
    decode_seconds = (
        workload.completion_tokens / decode.decode_tokens_per_second
    )
    handoff_seconds = (
        workload.handoff_fixed_seconds
        + workload.cache_transfer_bytes / workload.fabric_bytes_per_second
    )
    return DisaggregatedOrientation(
        prefill_node_id=prefill.node_id,
        decode_node_id=decode.node_id,
        prefill_seconds=prefill_seconds,
        handoff_seconds=handoff_seconds,
        decode_seconds=decode_seconds,
        first_request_seconds=prefill_seconds + handoff_seconds + decode_seconds,
        # The cache boundary uses both endpoints and therefore remains outside
        # the compute-stage max until a future double-buffered transport gate.
        steady_interval_seconds=(
            max(prefill_seconds, decode_seconds) + handoff_seconds
        ),
    )


def plan_disaggregated_prefill_decode(
    nodes: Sequence[DisaggregatedNodeProfile],
    workload: DisaggregatedWorkload,
    *,
    minimum_steady_speedup: float = 1.05,
) -> DisaggregatedPlan:
    """Choose prefill/decode roles or fail closed to ordinary serving.

    The first implementation intentionally admits exactly two full replicas.
    A future multi-node form can pass logical shard-group profiles through this
    same planner without changing the workload or orientation math.
    """

    if len(nodes) != 2 or len({node.node_id for node in nodes}) != 2:
        return DisaggregatedPlan(
            supported=False,
            recommended=False,
            reason="full-replica disaggregation currently requires two distinct nodes",
            orientation=None,
            best_single_node_id=None,
            best_single_interval_seconds=None,
            estimated_steady_speedup=None,
        )
    if (
        not math.isfinite(minimum_steady_speedup)
        or minimum_steady_speedup < 1.0
    ):
        raise ValueError("minimum steady speedup must be finite and at least one")

    non_fitting = [
        node.node_id
        for node in nodes
        if node.model_resident_bytes > node.admission_budget_bytes
    ]
    if non_fitting:
        return DisaggregatedPlan(
            supported=False,
            recommended=False,
            reason=(
                "the complete model replica exceeds the admitted budget on: "
                + ", ".join(sorted(non_fitting))
            ),
            orientation=None,
            best_single_node_id=None,
            best_single_interval_seconds=None,
            estimated_steady_speedup=None,
        )

    single = sorted(
        ((_single_interval(node, workload), node.node_id) for node in nodes),
        key=lambda item: (item[0], item[1]),
    )
    best_single_seconds, best_single_node = single[0]
    candidates = tuple(
        sorted(
            (
                _orientation(nodes[0], nodes[1], workload),
                _orientation(nodes[1], nodes[0], workload),
            ),
            key=lambda item: (
                item.steady_interval_seconds,
                item.first_request_seconds,
                item.prefill_node_id,
            ),
        )
    )
    chosen = candidates[0]
    speedup = best_single_seconds / chosen.steady_interval_seconds
    recommended = speedup >= minimum_steady_speedup
    return DisaggregatedPlan(
        supported=True,
        recommended=recommended,
        reason=(
            f"estimated steady request throughput improves {speedup:.3f}x"
            if recommended
            else (
                f"estimated steady speedup {speedup:.3f}x is below the "
                f"{minimum_steady_speedup:.3f}x admission threshold"
            )
        ),
        orientation=chosen,
        best_single_node_id=best_single_node,
        best_single_interval_seconds=best_single_seconds,
        estimated_steady_speedup=speedup,
        candidates=candidates,
    )


def build_full_replica_shard_plan(
    model: object,
    nodes: Sequence[object],
    *,
    prefill_rank: int,
    decode_rank: int,
    context_tokens: int,
    workload_profile: str = "throughput",
) -> object:
    """Build the signed memory plan consumed by a persistent phase split.

    Kept in this module so ordinary tensor/pipeline planning remains unchanged.
    Both ranks receive the complete layer range and full KV reservation; the
    deployment validator later refuses any partial or mismatched replica.
    """

    from .planner import (
        PipelineAssignment,
        PlanningError,
        ShardPlan,
        _kv_bytes_for_stage,
        _kv_bytes_per_token_for_stage,
        _max_context_for_stage,
    )

    if len(nodes) != 2 or {prefill_rank, decode_rank} != {0, 1}:
        raise PlanningError(
            "disaggregated serving requires two distinct phase ranks"
        )
    layer_weights = tuple(int(value) for value in model.layer_weight_bytes)
    if not layer_weights:
        raise PlanningError("disaggregated serving requires a layered model")
    layer_bytes = sum(layer_weights)
    fixed_bytes = int(model.fixed_weight_bytes)
    layer_count = len(layer_weights)
    kv_bytes = _kv_bytes_for_stage(model, layer_count, context_tokens)
    kv_bytes_per_token = _kv_bytes_per_token_for_stage(model, layer_count)
    assignments = []
    for node in nodes:
        planned_weights = fixed_bytes + layer_bytes
        planned = planned_weights + kv_bytes
        if planned > node.usable_bytes:
            raise PlanningError(
                f"full replica does not fit node {node.node_id}: {planned} > "
                f"{node.usable_bytes} (weights {planned_weights}, KV {kv_bytes})"
            )
        if planned_weights > node.weight_ceiling_bytes:
            raise PlanningError(
                f"full replica exceeds the weight limit on node {node.node_id}: "
                f"{planned_weights} > {node.weight_ceiling_bytes}"
            )
        assignments.append(
            PipelineAssignment(
                node_id=node.node_id,
                rank=node.rank,
                start_layer=0,
                end_layer=layer_count,
                layer_weight_bytes=layer_bytes,
                fixed_weight_bytes=fixed_bytes,
                reserve_bytes=node.reserve_bytes,
                capacity_bytes=node.capacity_bytes,
                manual_memory_limit=node.manual_memory_limit,
                role=node.role,
                memory_guard_tier=node.memory_guard_tier,
                tensor_parallel_rank=0,
                tensor_parallel_size=1,
                tensor_parallel_shard_weight=1,
                sharded_weight_bytes=0,
                kv_cache_bytes=kv_bytes,
                kv_bytes_per_token=kv_bytes_per_token,
                max_context_tokens=_max_context_for_stage(
                    model,
                    node,
                    layer_count=layer_count,
                    weight_bytes=planned_weights,
                ),
            )
        )
    assignments.sort(key=lambda item: item.rank)
    hash_payload = {
        "serving_mode": "disaggregated",
        "prefill_rank": prefill_rank,
        "decode_rank": decode_rank,
        "model": model.to_dict(),
        "context_tokens": context_tokens,
        "workload_profile": workload_profile,
        "assignments": [item.to_dict() for item in assignments],
    }
    plan_hash = hashlib.sha256(
        json.dumps(
            hash_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return ShardPlan(
        model=model,
        assignments=tuple(assignments),
        plan_hash=plan_hash,
        optimization="disaggregated",
        workload_profile=workload_profile,
        performance_profiles=tuple(
            node.performance
            for node in nodes
            if getattr(node, "performance", None) is not None
        ),
        tensor_parallel_size=1,
        pipeline_stages=1,
        target_context_tokens=context_tokens,
        serving_mode="disaggregated",
        prefill_rank=prefill_rank,
        decode_rank=decode_rank,
    )


__all__ = [
    "DisaggregatedNodeProfile",
    "DisaggregatedOrientation",
    "DisaggregatedPlan",
    "DisaggregatedWorkload",
    "plan_disaggregated_prefill_decode",
    "build_full_replica_shard_plan",
]
