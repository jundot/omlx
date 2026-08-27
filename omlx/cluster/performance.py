# SPDX-License-Identifier: Apache-2.0
"""Typed performance inputs and execution profiles for distributed inference."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Literal

ExecutionProfileName = Literal["interactive", "balanced", "throughput"]

_MAX_RATE = 1e18
_MAX_LATENCY_SECONDS = 60.0
_MAX_CONNECTIONS_PER_IP = 32
_GIB = 1024**3

# Persistent prompt snapshots are one file per aligned boundary and per rank.
# A single long-context conversation can therefore consume many GiB even with
# the 512-entry count guard. Keep a conservative, explicit per-rank ceiling in
# the signed execution contract; operators can raise it for larger SSDs.
DEFAULT_PROMPT_CACHE_SSD_MAX_BYTES = 20 * _GIB


@dataclass(frozen=True)
class DeepseekAnePrefillSettings:
    """Signed per-rank DeepSeek-V4 ANE/GPU/CPU prefill contract."""

    enabled: bool = False
    sequence_length: int = 4096
    tail_padding_min_tokens: int = 0
    down_enabled: bool = True
    down_fraction: float = 0.5
    wo_a_enabled: bool = False
    wo_a_fraction: float = 0.5
    cpu_enabled: bool = False
    cpu_fraction: float = 0.125
    cpu_threads: int = 12
    cpu_shared_resource: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("DeepSeek ANE enabled flag must be boolean")
        if (
            not isinstance(self.sequence_length, int)
            or isinstance(self.sequence_length, bool)
            or self.sequence_length < 1024
            or self.sequence_length % 64
        ):
            raise ValueError(
                "DeepSeek ANE sequence length must be a multiple of 64 >= 1024"
            )
        if (
            not isinstance(self.tail_padding_min_tokens, int)
            or isinstance(self.tail_padding_min_tokens, bool)
            or not (
                self.tail_padding_min_tokens == 0
                or 2 <= self.tail_padding_min_tokens < self.sequence_length
            )
        ):
            raise ValueError(
                "DeepSeek ANE tail padding must be zero or within the tile"
            )
        for name in ("down_fraction", "wo_a_fraction"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 < float(value) < 1.0
            ):
                raise ValueError(f"DeepSeek ANE {name} must be between zero and one")
        if (
            isinstance(self.cpu_fraction, bool)
            or not isinstance(self.cpu_fraction, (int, float))
            or not math.isfinite(float(self.cpu_fraction))
            or not 0.0 <= float(self.cpu_fraction) < 0.5
        ):
            raise ValueError("DeepSeek ANE CPU fraction must be in [0, 0.5)")
        if (
            not isinstance(self.cpu_threads, int)
            or isinstance(self.cpu_threads, bool)
            or not 0 <= self.cpu_threads <= 64
        ):
            raise ValueError("DeepSeek ANE CPU threads must be in [0, 64]")
        for name in (
            "down_enabled",
            "wo_a_enabled",
            "cpu_enabled",
            "cpu_shared_resource",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"DeepSeek ANE {name} must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "sequence_length": self.sequence_length,
            "tail_padding_min_tokens": self.tail_padding_min_tokens,
            "down_enabled": self.down_enabled,
            "down_fraction": self.down_fraction,
            "wo_a_enabled": self.wo_a_enabled,
            "wo_a_fraction": self.wo_a_fraction,
            "cpu_enabled": self.cpu_enabled,
            "cpu_fraction": self.cpu_fraction,
            "cpu_threads": self.cpu_threads,
            "cpu_shared_resource": self.cpu_shared_resource,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> DeepseekAnePrefillSettings:
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise ValueError("DeepSeek ANE settings must be an object")
        allowed = cls().to_dict()
        unknown = set(payload) - set(allowed)
        if unknown:
            raise ValueError(
                "DeepSeek ANE settings contain unknown fields: "
                + ", ".join(sorted(unknown))
            )
        allowed.update(payload)
        return cls(**allowed)


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
    promotable: bool = True
    qualification_reason: str = ""

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
        if not isinstance(self.promotable, bool):
            raise ValueError("performance profile promotable must be a boolean")
        if not isinstance(self.qualification_reason, str):
            raise ValueError("performance profile qualification reason must be text")
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
            "promotable": self.promotable,
            "qualification_reason": self.qualification_reason,
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
                promotable=payload.get("promotable", True),
                qualification_reason=str(payload.get("qualification_reason", "")),
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
    max_kv_size: int | None = None
    pipeline_microbatch_size: int = 4
    cache_affinity: bool = True
    # Snapshot the prompt cache to SSD at prefill boundaries so a model whose
    # per-layer state cannot be sliced (rotating window, gated-delta-net) still
    # reuses a long prefix across requests instead of recomputing it.
    prompt_cache_ssd: bool = False
    prompt_cache_ssd_max_bytes: int = DEFAULT_PROMPT_CACHE_SSD_MAX_BYTES
    sampling_rank_only: bool = True
    async_overlap: bool = True
    ring_connections_per_ip: int = 2
    tuning_reason: str = "balanced profile defaults"
    deepseek_ane_prefill: DeepseekAnePrefillSettings = field(
        default_factory=DeepseekAnePrefillSettings
    )

    def __post_init__(self) -> None:
        if self.profile not in {"interactive", "balanced", "throughput"}:
            raise ValueError("execution profile is invalid")
        if not isinstance(self.deepseek_ane_prefill, DeepseekAnePrefillSettings):
            raise ValueError("deepseek_ane_prefill must be typed settings")
        for name in (
            "decode_concurrency",
            "prompt_concurrency",
            "prefill_step_size",
            "prompt_cache_size",
            "pipeline_microbatch_size",
            "ring_connections_per_ip",
            "prompt_cache_ssd_max_bytes",
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
        result = {
            "profile": self.profile,
            "auto_tune": self.auto_tune,
            "decode_concurrency": self.decode_concurrency,
            "prompt_concurrency": self.prompt_concurrency,
            "prefill_step_size": self.prefill_step_size,
            "prompt_cache_size": self.prompt_cache_size,
            "prompt_cache_bytes": self.prompt_cache_bytes,
            "max_kv_size": self.max_kv_size,
            "pipeline_microbatch_size": self.pipeline_microbatch_size,
            "cache_affinity": self.cache_affinity,
            "prompt_cache_ssd": self.prompt_cache_ssd,
            "prompt_cache_ssd_max_bytes": self.prompt_cache_ssd_max_bytes,
            "sampling_rank_only": self.sampling_rank_only,
            "async_overlap": self.async_overlap,
            "ring_connections_per_ip": self.ring_connections_per_ip,
            "tuning_reason": self.tuning_reason,
        }
        if self.deepseek_ane_prefill.enabled:
            result["deepseek_ane_prefill"] = self.deepseek_ane_prefill.to_dict()
        return result

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
        deepseek_ane = DeepseekAnePrefillSettings.from_dict(
            payload.get("deepseek_ane_prefill")
        )
        for key in values:
            if key in payload:
                values[key] = payload[key]
        try:
            return cls(**values, deepseek_ane_prefill=deepseek_ane)
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

    MLX-LM's prompt cache evicts independently on every rank. A byte limit is
    therefore unsafe for unequal pipeline stages: the same prompt occupies
    different bytes on each Mac, so one rank can retain a prefix another rank
    evicts. The next request then starts at different token offsets and blocks
    forever in the first unmatched collective.

    Disable byte-based eviction, but retain the profile's deterministic entry
    budget.  The sequence-count policy is rank-symmetric because all ranks see
    the same ordered request stream.  MLX-LM may retain system, user and
    assistant segment boundaries separately, so one slot per admitted prompt
    is still too small: two active conversations can consume more than four
    entries.  Collapsing the cache to one slot made every alternating turn pay
    a full prefill tax despite the advertised prompt concurrency.  This
    correctness invariant applies even when the user disables the other
    automatic tuning.
    """

    def synchronized_cache(entry_limit: int) -> dict[str, int | None]:
        return {
            "prompt_cache_size": max(
                1,
                min(settings.prompt_cache_size, entry_limit),
            ),
            # Unequal pipeline stages hold different byte counts for the same
            # logical prefix. A per-rank byte LRU can therefore evict different
            # keys; the shared sequence-count LRU cannot.
            "prompt_cache_bytes": None,
        }

    if not settings.auto_tune or not assignments:
        return replace(
            settings,
            **synchronized_cache(settings.prompt_cache_size),
            tuning_reason=(
                f"{settings.tuning_reason}; synchronized count-bounded prompt cache"
            ),
        )
    minimum_headroom = min(
        max(0, int(getattr(item, "headroom_bytes", 0))) for item in assignments
    )
    if minimum_headroom < 4 * _GIB:
        caps = (2, 1, 512, 2, 1)
        tier = "critical headroom"
    elif minimum_headroom < 8 * _GIB:
        caps = (4, 2, 1024, 4, 2)
        tier = "low headroom"
    elif minimum_headroom < 16 * _GIB:
        caps = (8, 4, 2048, 8, 4)
        tier = "moderate headroom"
    else:
        caps = (
            settings.decode_concurrency,
            settings.prompt_concurrency,
            settings.prefill_step_size,
            settings.prompt_cache_size,
            settings.pipeline_microbatch_size,
        )
        tier = "ample headroom"

    decode = min(settings.decode_concurrency, caps[0])
    prompt = min(settings.prompt_concurrency, caps[1], decode)
    prefill = min(settings.prefill_step_size, caps[2])
    prompt_cache_size = min(settings.prompt_cache_size, caps[3])
    microbatch = min(settings.pipeline_microbatch_size, caps[4], decode)
    connections = settings.ring_connections_per_ip if backend == "ring" else 1
    return replace(
        settings,
        decode_concurrency=decode,
        prompt_concurrency=prompt,
        prefill_step_size=prefill,
        **synchronized_cache(prompt_cache_size),
        pipeline_microbatch_size=microbatch,
        ring_connections_per_ip=connections,
        tuning_reason=(
            f"{settings.profile} profile auto-tuned for {tier}; "
            f"minimum stage headroom {minimum_headroom / _GIB:.2f} GiB; "
            "synchronized count-bounded prompt cache"
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
                promotable=record.get("promotable", True),
                qualification_reason=str(record.get("qualification_reason", "")),
            )
        )
    return tuple(profiles)
