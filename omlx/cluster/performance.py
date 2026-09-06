# SPDX-License-Identifier: Apache-2.0
"""Typed performance inputs and execution profiles for distributed inference."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Literal

ExecutionProfileName = Literal["interactive", "balanced", "throughput"]

_MAX_RATE = 1e18
_MAX_LATENCY_SECONDS = 60.0
_MAX_CONNECTIONS_PER_IP = 32
_GIB = 1024**3


def _finite_rate(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite positive number") from exc
    if not math.isfinite(result) or not 0 < result <= _MAX_RATE:
        raise ValueError(f"{label} must be a finite positive number")
    return result


def _finite_latency(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite non-negative number") from exc
    if not math.isfinite(result) or result < 0 or result > _MAX_LATENCY_SECONDS:
        raise ValueError(f"{label} must be a finite non-negative number")
    return result


@dataclass(frozen=True)
class NodePerformanceProfile:
    """Bounded synthetic compute and collective measurements for one rank.

    The compute rates are deliberately expressed as effective model-weight
    bytes per second. They are calibration signals for relative partitioning,
    not promises about end-to-end model throughput.
    """

    node_id: str
    rank: int
    decode_weight_bytes_per_second: float
    prefill_weight_bytes_per_second: float
    collective_latency_seconds: float
    collective_bandwidth_bytes_per_second: float
    backend: str
    measured_at: str
    samples: int
    source: str = "synthetic_mlx_probe"

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("performance profile node_id is required")
        if self.rank < 0:
            raise ValueError("performance profile rank must be non-negative")
        if self.backend not in {"ring", "jaccl", "jaccl-ring"}:
            raise ValueError("performance profile backend is invalid")
        if not self.measured_at:
            raise ValueError("performance profile measured_at is required")
        if not 1 <= self.samples <= 10_000:
            raise ValueError("performance profile samples are out of range")
        if self.source != "synthetic_mlx_probe":
            raise ValueError("performance profile source is unsupported")
        object.__setattr__(
            self,
            "decode_weight_bytes_per_second",
            _finite_rate(
                self.decode_weight_bytes_per_second,
                "decode weight rate",
            ),
        )
        object.__setattr__(
            self,
            "prefill_weight_bytes_per_second",
            _finite_rate(
                self.prefill_weight_bytes_per_second,
                "prefill weight rate",
            ),
        )
        object.__setattr__(
            self,
            "collective_latency_seconds",
            _finite_latency(
                self.collective_latency_seconds,
                "collective latency",
            ),
        )
        object.__setattr__(
            self,
            "collective_bandwidth_bytes_per_second",
            _finite_rate(
                self.collective_bandwidth_bytes_per_second,
                "collective bandwidth",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "rank": self.rank,
            "decode_weight_bytes_per_second": self.decode_weight_bytes_per_second,
            "prefill_weight_bytes_per_second": self.prefill_weight_bytes_per_second,
            "collective_latency_seconds": self.collective_latency_seconds,
            "collective_bandwidth_bytes_per_second": (
                self.collective_bandwidth_bytes_per_second
            ),
            "backend": self.backend,
            "measured_at": self.measured_at,
            "samples": self.samples,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NodePerformanceProfile:
        if not isinstance(payload, dict):
            raise ValueError("performance profile must be an object")
        try:
            return cls(
                node_id=str(payload["node_id"]),
                rank=int(payload["rank"]),
                decode_weight_bytes_per_second=payload[
                    "decode_weight_bytes_per_second"
                ],
                prefill_weight_bytes_per_second=payload[
                    "prefill_weight_bytes_per_second"
                ],
                collective_latency_seconds=payload["collective_latency_seconds"],
                collective_bandwidth_bytes_per_second=payload[
                    "collective_bandwidth_bytes_per_second"
                ],
                backend=str(payload["backend"]),
                measured_at=str(payload["measured_at"]),
                samples=int(payload["samples"]),
                source=str(payload.get("source", "synthetic_mlx_probe")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and not isinstance(exc, KeyError):
                raise
            raise ValueError("performance profile is missing required fields") from exc


@dataclass(frozen=True)
class ExecutionSettings:
    """Resolved worker controls stored with a cluster deployment."""

    profile: ExecutionProfileName = "balanced"
    auto_tune: bool = True
    decode_concurrency: int = 8
    prompt_concurrency: int = 4
    prefill_step_size: int = 2048
    prompt_cache_size: int = 8
    prompt_cache_bytes: int | None = None
    # The slot count the operator or profile asked for, before the headroom
    # tuner clamped it. Tuning can run more than once on the activation path
    # (plan-time budgets, then measured rank profiles); clamping from the
    # already-clamped value would pin the first tier's slot count forever.
    requested_prompt_cache_size: int | None = None
    max_kv_size: int | None = None
    pipeline_microbatch_size: int = 4
    cache_affinity: bool = True
    # Snapshot the prompt cache to SSD at prefill boundaries so a model whose
    # per-layer state cannot be sliced (rotating window, gated-delta-net) still
    # reuses a long prefix across requests instead of recomputing it.
    prompt_cache_ssd: bool = True
    sampling_rank_only: bool = True
    async_overlap: bool = True
    ring_connections_per_ip: int = 2
    tuning_reason: str = "balanced profile defaults"

    def __post_init__(self) -> None:
        if self.profile not in {"interactive", "balanced", "throughput"}:
            raise ValueError("execution profile is invalid")
        for name in (
            "decode_concurrency",
            "prompt_concurrency",
            "prefill_step_size",
            "prompt_cache_size",
            "pipeline_microbatch_size",
            "ring_connections_per_ip",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.prompt_concurrency > self.decode_concurrency:
            raise ValueError("prompt concurrency cannot exceed decode concurrency")
        if self.pipeline_microbatch_size > self.decode_concurrency:
            raise ValueError(
                "pipeline microbatch size cannot exceed decode concurrency"
            )
        if self.ring_connections_per_ip > _MAX_CONNECTIONS_PER_IP:
            raise ValueError(
                f"ring connections per IP cannot exceed {_MAX_CONNECTIONS_PER_IP}"
            )
        for name in (
            "prompt_cache_bytes",
            "requested_prompt_cache_size",
            "max_kv_size",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer when set")
        if not isinstance(self.tuning_reason, str) or not self.tuning_reason:
            raise ValueError("tuning_reason is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "auto_tune": self.auto_tune,
            "decode_concurrency": self.decode_concurrency,
            "prompt_concurrency": self.prompt_concurrency,
            "prefill_step_size": self.prefill_step_size,
            "prompt_cache_size": self.prompt_cache_size,
            "prompt_cache_bytes": self.prompt_cache_bytes,
            "requested_prompt_cache_size": self.requested_prompt_cache_size,
            "max_kv_size": self.max_kv_size,
            "pipeline_microbatch_size": self.pipeline_microbatch_size,
            "cache_affinity": self.cache_affinity,
            "prompt_cache_ssd": self.prompt_cache_ssd,
            "sampling_rank_only": self.sampling_rank_only,
            "async_overlap": self.async_overlap,
            "ring_connections_per_ip": self.ring_connections_per_ip,
            "tuning_reason": self.tuning_reason,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> ExecutionSettings:
        if payload is None:
            return execution_profile("balanced")
        if not isinstance(payload, dict):
            raise ValueError("execution settings must be an object")
        profile = payload.get("profile", "balanced")
        if profile not in {"interactive", "balanced", "throughput"}:
            raise ValueError("execution profile is invalid")
        defaults = execution_profile(profile)
        values = defaults.to_dict()
        for key in values:
            if key in payload:
                values[key] = payload[key]
        try:
            return cls(**values)
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError("execution settings contain an invalid field") from exc


def execution_profile(
    profile: ExecutionProfileName,
    *,
    auto_tune: bool = True,
    sampling_rank_only: bool = True,
) -> ExecutionSettings:
    """Return conservative presets which the headroom tuner may reduce."""

    presets: dict[ExecutionProfileName, dict[str, Any]] = {
        "interactive": {
            "decode_concurrency": 4,
            "prompt_concurrency": 2,
            "prefill_step_size": 1024,
            "prompt_cache_size": 4,
            "pipeline_microbatch_size": 2,
            "ring_connections_per_ip": 1,
        },
        "balanced": {
            "decode_concurrency": 8,
            "prompt_concurrency": 4,
            "prefill_step_size": 2048,
            "prompt_cache_size": 8,
            "pipeline_microbatch_size": 4,
            "ring_connections_per_ip": 2,
        },
        "throughput": {
            "decode_concurrency": 16,
            "prompt_concurrency": 8,
            "prefill_step_size": 4096,
            "prompt_cache_size": 16,
            "pipeline_microbatch_size": 8,
            "ring_connections_per_ip": 4,
        },
    }
    if profile not in presets:
        raise ValueError("execution profile is invalid")
    return ExecutionSettings(
        profile=profile,
        auto_tune=auto_tune,
        sampling_rank_only=sampling_rank_only,
        tuning_reason=f"{profile} profile defaults",
        **presets[profile],
    )


def tune_execution_settings(
    settings: ExecutionSettings,
    assignments: Sequence[Any],
    *,
    backend: str,
) -> ExecutionSettings:
    """Bound concurrency while keeping distributed prompt caches coherent.

    MLX-LM keeps a prompt cache on every rank, so cache state must stay
    identical across ranks or a request starts at different token offsets on
    each rank and blocks forever in the first unmatched collective. Two
    conditions guarantee that:

    - Which entry is evicted must match on every rank. Eviction order is pure
      recency over the insert/fetch event stream, and every rank observes the
      same stream in the same order, so count-based LRU stays in lockstep for
      any slot count.
    - Whether an eviction happens must not depend on rank-local accounting:
      the same tokens occupy different bytes on different pipeline stages, so
      a byte budget crosses its threshold at different requests on different
      ranks and the ranks then retain different prefixes. Byte budgets are
      therefore always disabled.

    The slot count tiers with minimum stage headroom. Slots accrue lazily as
    requests run, so nothing is reserved up front. The slot count bounds how
    many prefixes are retained, not their byte cost: a slot holding a
    near-limit-context prefix costs that prefix's full KV, and byte budgets
    stay disabled for rank coherence, so total cache pressure is bounded by
    the memory guard rather than by this policy. This correctness invariant
    applies even when the user disables the other automatic tuning.
    """

    if not settings.auto_tune or not assignments:
        return replace(
            settings,
            prompt_cache_size=1,
            prompt_cache_bytes=None,
            tuning_reason=(
                f"{settings.tuning_reason}; synchronized single-slot prompt cache"
            ),
        )
    minimum_headroom = min(
        max(0, int(getattr(item, "headroom_bytes", 0))) for item in assignments
    )
    if minimum_headroom < 4 * _GIB:
        caps = (2, 1, 512, 2, 1)
        slot_cap = 1
        tier = "critical headroom"
    elif minimum_headroom < 8 * _GIB:
        caps = (4, 2, 1024, 4, 2)
        slot_cap = 1
        tier = "low headroom"
    elif minimum_headroom < 16 * _GIB:
        caps = (8, 4, 2048, 8, 4)
        slot_cap = 2
        tier = "moderate headroom"
    else:
        caps = (
            settings.decode_concurrency,
            settings.prompt_concurrency,
            settings.prefill_step_size,
            settings.prompt_cache_size,
            settings.pipeline_microbatch_size,
        )
        slot_cap = 4
        tier = "ample headroom"

    decode = min(settings.decode_concurrency, caps[0])
    prompt = min(settings.prompt_concurrency, caps[1], decode)
    prefill = min(settings.prefill_step_size, caps[2])
    microbatch = min(settings.pipeline_microbatch_size, caps[4], decode)
    connections = settings.ring_connections_per_ip if backend == "ring" else 1
    # Clamp from the requested slot count, not the resolved one. Tuning runs
    # again after the performance probe against measured headroom, and an
    # earlier pass with tighter planned budgets may have held the slot count
    # down; clamping an already-clamped value could never raise it back to
    # the tier the measured headroom supports.
    requested_slots = (
        settings.requested_prompt_cache_size
        if settings.requested_prompt_cache_size is not None
        else settings.prompt_cache_size
    )
    cache_slots = max(1, min(requested_slots, slot_cap))
    return replace(
        settings,
        decode_concurrency=decode,
        prompt_concurrency=prompt,
        prefill_step_size=prefill,
        prompt_cache_size=cache_slots,
        prompt_cache_bytes=None,
        requested_prompt_cache_size=requested_slots,
        pipeline_microbatch_size=microbatch,
        ring_connections_per_ip=connections,
        tuning_reason=(
            f"{settings.profile} profile auto-tuned for {tier}; "
            f"minimum stage headroom {minimum_headroom / _GIB:.2f} GiB; "
            f"synchronized {cache_slots}-slot prompt cache"
        ),
    )


def performance_profiles_from_records(
    records: Sequence[dict[str, Any]],
    *,
    node_ids: Sequence[str],
    backend: str,
) -> tuple[NodePerformanceProfile, ...]:
    """Validate one benchmark record per rank and bind it to trusted node IDs."""

    by_rank: dict[int, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or record.get("type") != "performance_result":
            continue
        rank = record.get("rank")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank in by_rank:
            raise ValueError("performance probe returned duplicate or invalid ranks")
        by_rank[rank] = record
    if set(by_rank) != set(range(len(node_ids))):
        raise ValueError("performance probe did not return every cluster rank")
    profiles = []
    for rank, node_id in enumerate(node_ids):
        record = by_rank[rank]
        profiles.append(
            NodePerformanceProfile(
                node_id=node_id,
                rank=rank,
                decode_weight_bytes_per_second=record.get(
                    "decode_weight_bytes_per_second"
                ),
                prefill_weight_bytes_per_second=record.get(
                    "prefill_weight_bytes_per_second"
                ),
                collective_latency_seconds=record.get("collective_latency_seconds"),
                collective_bandwidth_bytes_per_second=record.get(
                    "collective_bandwidth_bytes_per_second"
                ),
                backend=backend,
                measured_at=str(
                    record.get("measured_at") or datetime.now(UTC).isoformat()
                ),
                samples=int(record.get("samples", 1)),
            )
        )
    return tuple(profiles)
