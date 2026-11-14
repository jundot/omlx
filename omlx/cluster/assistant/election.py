# SPDX-License-Identifier: Apache-2.0
"""Coordinator election for the oMLX Cluster Assistant.

Rank 0 is not the Mac the user opened the app on. In the MVP rank 0 is also the
coordinator, the endpoint owner and the supervisor host, so choosing it decides
where the cluster *is*. The rule this module implements is the one frozen in
``docs/cluster-assistant/PROJECT-BRIEF.md`` section 7:

1. every healthy Mac is tried as rank 0, with every order of the other Macs;
2. a candidate is kept only if a legal PP split exists for it — every stage
   inside its measured admission, and rank 0 inside its admission *including*
   the tail, the LM/MTP head and the coordinator overhead;
3. survivors are scored: predicted critical stage first, then headroom, then
   coordinator suitability;
4. free admission is a preference between otherwise equal plans, never a reason
   to pick a Mac that cannot hold the stage;
5. no feasible candidate is an explicit error. There is deliberately no
   "fall back to this Mac" branch anywhere in this file.

The layer split itself is not re-implemented here. ``plan_hybrid`` in
``omlx/cluster/planner.py`` already owns the PP contract (contiguous layers,
KV owner boundaries, per-stage admission, MTP head on the last layer), and the
election only feeds it one rank order at a time and reads back the verdict.
Keeping one planner is what makes an elected plan the same object the rest of
oMLX approves, launches and hashes.

The module is pure: no HTTP, no SSH, no model loading, no GPU. Inputs are
plain measured numbers, so the whole election is unit-testable off-cluster.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import permutations
from typing import Any

from ..performance import ExecutionProfileName, NodePerformanceProfile
from ..planner import (
    ModelLayout,
    NodeBudget,
    PipelineAssignment,
    PlanningError,
    ShardPlan,
    normalize_memory_guard_tier,
    plan_hybrid,
)
from .topology import missing_line_links

GIB = 1024**3

# Leave a little admission for the supervisor/HTTP and for the difference
# between the measurement's p50 and the live ceiling at Start. A live GLM 32k
# plan elected a 32 GB Studio at 21.2 GiB while Start admitted 21.1 GiB; the
# 0.1 GiB miss failed an otherwise feasible plan. This is a planning margin,
# never a lowering of the measured p05 floor.
RANK0_ADMISSION_MARGIN_BYTES = 256 * 1024**2

# The MVP is 2–4 Macs, so the exhaustive search is at most 4 * 3! rank orders.
# Beyond that every added Mac multiplies the permutations, and the product does
# not support it: refuse loudly instead of planning for minutes.
MAX_CANDIDATE_NODES = 4

# Nominal Thunderbolt/TCP ring used when a caller has no measured link. It only
# shapes relative send cost inside one election; it is not a throughput promise.
_NOMINAL_COLLECTIVE_BANDWIDTH_BYTES_PER_SECOND = 4 * 10**9
_NOMINAL_COLLECTIVE_LATENCY_SECONDS = 0.0002

_LINK_TRANSPORTS = frozenset({"ring", "jaccl", "jaccl-ring"})

# A bandwidth-bound decode probe reports model-weight bytes/s close to the
# Mac's unified-memory bandwidth. Apple Silicon spans roughly 68 GB/s (base M1)
# to ~800 GB/s (M2 Ultra); this generous window rejects the old latency-bound
# probe's hardware-impossible rates (single-digit GB/s) and any nonsense
# ceiling without rejecting a real chip. An implausible probe is treated as
# unknown compute (decision E4=B): headroom preference, never a fabricated
# ranking.
_PLAUSIBLE_DECODE_RATE_FLOOR = 50 * 10**9  # 50 GB/s
_PLAUSIBLE_DECODE_RATE_CEILING = 1_200 * 10**9  # 1.2 TB/s


class ElectionError(PlanningError):
    """Base error for coordinator election.

    Subclasses ``PlanningError`` so cluster code that already handles "this
    model does not fit these budgets" also handles "no Mac can be rank 0".
    """


class UnsupportedParallelismError(ElectionError):
    """Raised for a tensor-parallel request; the MVP is PP only."""


class NoFeasiblePlanError(ElectionError):
    """No Mac can host rank 0 for this model at this context.

    ``result`` carries the same information in the serializable shape an API
    would return, so a caller can show which constraint failed per Mac without
    parsing the message. ``action`` is the one next step: a context/tier
    refusal asks to adjust the plan, while an architecture mlx-lm cannot split
    asks for a pipeline-capable model.
    """

    def __init__(
        self,
        message: str,
        result: ElectionResult,
        *,
        action: str = "adjust_model_or_context",
    ) -> None:
        super().__init__(message)
        self.result = result
        self.action = action


@dataclass(frozen=True)
class ComputeProfile:
    """Measured compute rates for one Mac, in model-weight bytes per second.

    These are the same units ``NodePerformanceProfile`` uses: calibration
    signals for *relative* partitioning, not tokens/s promises.
    """

    decode_weight_bytes_per_second: float
    prefill_weight_bytes_per_second: float
    # Link numbers are only read for the non-rank-0 send cost. The defaults
    # describe a nominal wired/Thunderbolt ring and exist so a caller who
    # measured compute but not the fabric still gets a stable relative score.
    collective_latency_seconds: float = _NOMINAL_COLLECTIVE_LATENCY_SECONDS
    collective_bandwidth_bytes_per_second: float = (
        _NOMINAL_COLLECTIVE_BANDWIDTH_BYTES_PER_SECOND
    )
    backend: str = "ring"
    measured_at: str = ""
    samples: int = 1

    def __post_init__(self) -> None:
        if self.decode_weight_bytes_per_second <= 0:
            raise ValueError("decode weight rate must be positive")
        if self.prefill_weight_bytes_per_second <= 0:
            raise ValueError("prefill weight rate must be positive")
        if self.collective_latency_seconds < 0:
            raise ValueError("collective latency must be non-negative")
        if self.collective_bandwidth_bytes_per_second <= 0:
            raise ValueError("collective bandwidth must be positive")
        if self.backend not in _LINK_TRANSPORTS:
            raise ValueError(f"unknown collective backend: {self.backend!r}")
        if not 1 <= self.samples <= 10_000:
            raise ValueError("samples must be between 1 and 10000")

    def relative_speed(self, reference_decode_rate: float) -> float:
        """This Mac's decode rate as a fraction of the fastest known one."""

        if reference_decode_rate <= 0:
            raise ValueError("reference decode rate must be positive")
        return self.decode_weight_bytes_per_second / reference_decode_rate

    def plausible(self) -> bool:
        """Whether the decode rate is a believable memory-bandwidth reading.

        The bandwidth-bound decode probe (rows=1) reads weights at roughly the
        Mac's unified-memory bandwidth. A rate outside the plausible window is
        a latency-bound or corrupt measurement, not a slow Mac; the election
        treats it as unknown compute (E4=B) instead of ranking by it.
        """

        return (
            _PLAUSIBLE_DECODE_RATE_FLOOR
            <= self.decode_weight_bytes_per_second
            <= _PLAUSIBLE_DECODE_RATE_CEILING
        )

    def performance_profile(self, node_id: str, rank: int) -> NodePerformanceProfile:
        """The planner's profile type for this rank."""

        return NodePerformanceProfile(
            node_id=node_id,
            rank=rank,
            decode_weight_bytes_per_second=self.decode_weight_bytes_per_second,
            prefill_weight_bytes_per_second=self.prefill_weight_bytes_per_second,
            collective_latency_seconds=self.collective_latency_seconds,
            collective_bandwidth_bytes_per_second=(
                self.collective_bandwidth_bytes_per_second
            ),
            backend=self.backend,
            measured_at=self.measured_at or "election",
            samples=self.samples,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decode_weight_bytes_per_second": self.decode_weight_bytes_per_second,
            "prefill_weight_bytes_per_second": self.prefill_weight_bytes_per_second,
            "collective_latency_seconds": self.collective_latency_seconds,
            "collective_bandwidth_bytes_per_second": (
                self.collective_bandwidth_bytes_per_second
            ),
            "backend": self.backend,
            "measured_at": self.measured_at,
            "samples": self.samples,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ComputeProfile:
        # ``performance_profile`` injects a placeholder timestamp; round-tripping
        # it would turn "unmeasured" into "measured at election".
        measured_at = str(payload.get("measured_at", ""))
        return cls(
            decode_weight_bytes_per_second=float(
                payload["decode_weight_bytes_per_second"]
            ),
            prefill_weight_bytes_per_second=float(
                payload["prefill_weight_bytes_per_second"]
            ),
            collective_latency_seconds=float(
                payload.get(
                    "collective_latency_seconds",
                    _NOMINAL_COLLECTIVE_LATENCY_SECONDS,
                )
            ),
            collective_bandwidth_bytes_per_second=float(
                payload.get(
                    "collective_bandwidth_bytes_per_second",
                    _NOMINAL_COLLECTIVE_BANDWIDTH_BYTES_PER_SECOND,
                )
            ),
            backend=str(payload.get("backend", "ring")),
            measured_at="" if measured_at == "election" else measured_at,
            samples=int(payload.get("samples", 1)),
        )


@dataclass(frozen=True)
class LinkHints:
    """How this Mac is cabled to the others. Hints, not measurements."""

    transport: str = "ring"
    bandwidth_gbps: float | None = None
    latency_ms: float | None = None

    def __post_init__(self) -> None:
        if self.transport not in _LINK_TRANSPORTS:
            raise ValueError(f"unknown transport: {self.transport!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "transport": self.transport,
            "bandwidth_gbps": self.bandwidth_gbps,
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LinkHints:
        return cls(
            transport=str(payload.get("transport", "ring")),
            bandwidth_gbps=(
                None
                if payload.get("bandwidth_gbps") is None
                else float(payload["bandwidth_gbps"])
            ),
            latency_ms=(
                None
                if payload.get("latency_ms") is None
                else float(payload["latency_ms"])
            ),
        )


@dataclass(frozen=True)
class NodeFacts:
    """What the election knows about one Mac.

    Admission is the measured memory ceiling a rank may use (brief section 7.3).
    ``admission_p05`` is the conservative floor and the number a plan is built
    against; p50/p95 ride along for confidence and for the human-readable
    reason. Nothing here says whether this Mac is the app origin — the origin
    is deliberately not an input to any decision in this module.
    """

    node_id: str
    capacity_gib: float
    admission_p05_gib: float
    admission_p50_gib: float | None = None
    admission_p95_gib: float | None = None
    compute: ComputeProfile | None = None
    # A Mac without the model is only ruled out when we can prove it cannot be
    # staged either. "No measurement" must not read as "no disk".
    model_present: bool = True
    disk_free_gib: float | None = None
    link: LinkHints = LinkHints()
    healthy: bool = True
    # The private address a client would reach the endpoint on. Empty means
    # "not measured yet"; the election only records it, it never dials it.
    endpoint_host: str = ""
    # Checkpoints this Mac's own catalogue listed, as the measurement recorded
    # them. Empty means "not measured" — it never means "holds nothing", so a
    # recommendation can answer "already present" for any model without claiming
    # knowledge it does not have.
    model_paths: tuple[str, ...] = ()
    # The process memory policy this Mac saved. It is the default tier for a
    # plan that does not raise it; an explicit plan policy still wins.
    memory_guard_tier: str = "balanced"
    # ``iogpu.wired_limit_mb`` when that Mac reported it. ``None`` is "not
    # measured", never zero; the preflight only refuses a known-too-low limit.
    wired_limit_mb: int | None = None
    #: The absolute interpreter that Mac reported for cluster work. Empty means
    #: "not measured"; the deployment then falls back to the coordinator's own
    #: interpreter, which is only correct when the worker shares that path.
    python_executable: str = ""

    def __post_init__(self) -> None:
        if not self.node_id or not self.node_id.strip():
            raise ValueError("node_id is required")
        if self.capacity_gib <= 0:
            raise ValueError("capacity_gib must be positive")
        if not 0 < self.admission_p05_gib <= self.capacity_gib:
            raise ValueError(
                "admission_p05_gib must be positive and at most capacity_gib"
            )
        for name in ("admission_p50_gib", "admission_p95_gib"):
            value = getattr(self, name)
            if value is not None and not value >= 0:
                raise ValueError(f"{name} must be non-negative")
        if self.disk_free_gib is not None and self.disk_free_gib < 0:
            raise ValueError("disk_free_gib must be non-negative")
        if self.wired_limit_mb is not None and self.wired_limit_mb < 0:
            raise ValueError("wired_limit_mb must be non-negative")
        object.__setattr__(
            self,
            "memory_guard_tier",
            normalize_memory_guard_tier(self.memory_guard_tier),
        )

    @property
    def capacity_bytes(self) -> int:
        return _gib_to_bytes(self.capacity_gib)

    @property
    def admission_bytes(self) -> int:
        """What this Mac may hand a rank — the conservative p05 measurement."""

        return _gib_to_bytes(self.admission_p05_gib)

    @property
    def admission_p50_bytes(self) -> int | None:
        return (
            None
            if self.admission_p50_gib is None
            else _gib_to_bytes(self.admission_p50_gib)
        )

    @property
    def admission_p95_bytes(self) -> int | None:
        return (
            None
            if self.admission_p95_gib is None
            else _gib_to_bytes(self.admission_p95_gib)
        )

    @property
    def disk_free_bytes(self) -> int | None:
        return None if self.disk_free_gib is None else _gib_to_bytes(self.disk_free_gib)

    def can_stage(self, model_bytes: int) -> bool:
        """Whether the model could be copied here when it is not present yet.

        Unknown free space counts as "not disproven": the election must not
        reject a Mac because a measurement is missing.
        """

        if self.model_present:
            return True
        free = self.disk_free_bytes
        return True if free is None else free >= model_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "capacity_gib": self.capacity_gib,
            "admission_p05_gib": self.admission_p05_gib,
            "admission_p50_gib": self.admission_p50_gib,
            "admission_p95_gib": self.admission_p95_gib,
            "compute": None if self.compute is None else self.compute.to_dict(),
            "model_present": self.model_present,
            "disk_free_gib": self.disk_free_gib,
            "link": self.link.to_dict(),
            "healthy": self.healthy,
            "endpoint_host": self.endpoint_host,
            "model_paths": list(self.model_paths),
            "memory_guard_tier": self.memory_guard_tier,
            "wired_limit_mb": self.wired_limit_mb,
            "python_executable": self.python_executable,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NodeFacts:
        compute = payload.get("compute")
        link = payload.get("link")
        return cls(
            node_id=str(payload["node_id"]),
            capacity_gib=float(payload["capacity_gib"]),
            admission_p05_gib=float(payload["admission_p05_gib"]),
            admission_p50_gib=_optional_float(payload.get("admission_p50_gib")),
            admission_p95_gib=_optional_float(payload.get("admission_p95_gib")),
            compute=(None if not compute else ComputeProfile.from_dict(compute)),
            model_present=bool(payload.get("model_present", True)),
            disk_free_gib=_optional_float(payload.get("disk_free_gib")),
            link=LinkHints.from_dict(link or {}),
            healthy=bool(payload.get("healthy", True)),
            model_paths=tuple(
                str(item)
                for item in (payload.get("model_paths") or ())
                if isinstance(item, str) and item.strip()
            ),
            endpoint_host=str(payload.get("endpoint_host", "")),
            memory_guard_tier=str(payload.get("memory_guard_tier") or "balanced"),
            wired_limit_mb=_optional_int(payload.get("wired_limit_mb")),
            python_executable=str(payload.get("python_executable") or ""),
        )


@dataclass(frozen=True)
class StagePlan:
    """One pipeline stage of a candidate plan, as oMLX's planner produced it."""

    rank: int
    node_id: str
    first_layer: int
    last_layer: int
    layer_count: int
    resident_bytes: int
    admission_bytes: int
    headroom_bytes: int
    predicted_seconds: float | None = None
    critical: bool = False

    @property
    def resident_gib(self) -> float:
        return self.resident_bytes / GIB

    @property
    def headroom_gib(self) -> float:
        return self.headroom_bytes / GIB

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "node_id": self.node_id,
            "first_layer": self.first_layer,
            "last_layer": self.last_layer,
            "layer_count": self.layer_count,
            "resident_bytes": self.resident_bytes,
            "resident_gib": self.resident_gib,
            "admission_bytes": self.admission_bytes,
            "headroom_bytes": self.headroom_bytes,
            "headroom_gib": self.headroom_gib,
            "predicted_seconds": self.predicted_seconds,
            "critical": self.critical,
        }


@dataclass(frozen=True)
class PlanCandidate:
    """One rank order, cut and verdict.

    ``feasible`` is the only field that decides whether this may be elected;
    everything else is scoring and explanation. ``rank0_admission_bytes`` is the
    free admission of the rank-0 Mac, kept because the brief makes it a
    preference between plans that are otherwise equal.
    """

    rank_order: tuple[str, ...]
    stages: tuple[StagePlan, ...]
    feasible: bool
    reason: str
    rank0_extra_bytes: int = 0
    rank0_headroom_bytes: int = 0
    rank0_admission_bytes: int = 0
    min_headroom_bytes: int = 0
    critical_stage_seconds: float | None = None
    context_tokens: int = 0
    plan_hash: str = ""
    # The planner's own plan, so a caller that starts the cluster does not have
    # to re-derive a split that must stay identical to the one elected. It is
    # left out of ``to_dict`` (it is derivable from the cuts above and is far
    # too large for an API response); ``replace``d plans keep it consistent.
    shard_plan: ShardPlan | None = None
    # The resolved resource policy this candidate was elected with (a
    # ``policy.ResourcePolicy``, typed Any to avoid an import cycle). The search
    # loop passes it back to ``plan_apply`` so the re-election reproduces the
    # exact signature instead of falling back to the policy-less election.
    resource_policy: Any = None

    @property
    def coordinator_node(self) -> str:
        return self.rank_order[0] if self.rank_order else ""

    @property
    def groups(self) -> tuple[int, ...]:
        """Layers per rank, in rank order."""

        return tuple(stage.layer_count for stage in self.stages)

    @property
    def critical_stage(self) -> StagePlan | None:
        return next((stage for stage in self.stages if stage.critical), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feasible": self.feasible,
            "reason": self.reason,
            "coordinator_node": self.coordinator_node,
            "rank_order": list(self.rank_order),
            "groups": list(self.groups),
            "stages": [stage.to_dict() for stage in self.stages],
            "rank0_extra_bytes": self.rank0_extra_bytes,
            "rank0_headroom_bytes": self.rank0_headroom_bytes,
            "rank0_headroom_gib": self.rank0_headroom_bytes / GIB,
            "rank0_admission_bytes": self.rank0_admission_bytes,
            "min_headroom_bytes": self.min_headroom_bytes,
            "min_headroom_gib": self.min_headroom_bytes / GIB,
            "critical_stage_seconds": self.critical_stage_seconds,
            "context_tokens": self.context_tokens,
            "plan_hash": self.plan_hash,
        }


@dataclass(frozen=True)
class RejectedCandidate:
    """A Mac that could not be rank 0, and why, in the caller's words."""

    node_id: str
    reason: str
    attempts: int = 0
    details: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "reason": self.reason,
            "attempts": self.attempts,
            "details": list(self.details),
        }


@dataclass(frozen=True)
class ElectionPolicy:
    """The rule set that was applied, recorded with the result.

    §7.5's hard order is fixed in code; this only records which preferences
    were on, so a persisted election can explain itself later.
    """

    name: str = "feasibility_first"
    free_admission_preference: bool = True
    stop_on_infeasible: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "free_admission_preference": self.free_admission_preference,
            "stop_on_infeasible": self.stop_on_infeasible,
        }


@dataclass(frozen=True)
class EndpointPlaceholder:
    """Who will own the endpoint. The address itself is not an election input.

    The MVP puts the endpoint on the elected Mac, but the port and binding are
    runtime decisions (brief §9.4), so this only names the owner and carries the
    measured address when one exists. ``port`` 0 means "the coordinator's
    configured port".
    """

    node_id: str
    host: str = ""
    port: int = 0

    @property
    def url(self) -> str:
        if not self.host:
            return ""
        return f"http://{self.host}:{self.port}" if self.port else f"http://{self.host}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "url": self.url,
        }


@dataclass(frozen=True)
class ElectionResult:
    """The elected coordinator plus everything that was ruled out.

    ``feasible`` false is only ever produced for the ``NoFeasiblePlanError``
    payload: an infeasible election names no coordinator and no rank order, so
    nothing downstream can mistake the origin Mac for the answer.
    """

    feasible: bool
    coordinator_node: str
    rank_order: tuple[str, ...]
    endpoint: EndpointPlaceholder | None
    selected: PlanCandidate | None
    considered: tuple[PlanCandidate, ...]
    rejected: tuple[RejectedCandidate, ...]
    policy: ElectionPolicy
    origin_node: str
    plan_signature: str
    reasons: tuple[str, ...]
    tensor_parallel_size: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "feasible": self.feasible,
            "coordinator_node": self.coordinator_node,
            "rank_order": list(self.rank_order),
            "endpoint": None if self.endpoint is None else self.endpoint.to_dict(),
            "selected": None if self.selected is None else self.selected.to_dict(),
            "considered": [candidate.to_dict() for candidate in self.considered],
            "rejected": [item.to_dict() for item in self.rejected],
            "policy": self.policy.to_dict(),
            "origin_node": self.origin_node,
            "plan_signature": self.plan_signature,
            "reasons": list(self.reasons),
            "tensor_parallel_size": self.tensor_parallel_size,
        }


def elect(
    nodes: Sequence[NodeFacts],
    model: ModelLayout,
    *,
    context_tokens: int,
    rank0_extra_bytes: int = 0,
    mtp: bool = False,
    workload_profile: ExecutionProfileName = "balanced",
    microbatch_size: int = 1,
    tensor_parallel_size: int = 1,
    origin_node: str | None = None,
    policy: ElectionPolicy | None = None,
    memory_guard_tier: str | None = None,
    admission: str = "p05",
    table_offload: str = "none",
    moe_offload_layer_bytes: Sequence[int] = (),
    moe_expert_offload_resident_fraction: float = 0.0,
    fabric_links: Sequence[Sequence[str]] | None = None,
    two_mac_fallback: bool = False,
) -> ElectionResult:
    """Elect the coordinator/rank 0 and return the plan that justifies it.

    ``rank0_extra_bytes`` is memory only rank 0 holds beyond its pipeline stage:
    an LM head the layout does not already charge, plus the coordinator/HTTP
    overhead. ``mtp=True`` adds the checkpoint's Lightning MTP head. Both are
    charged to the last layer, which the planner always places on rank 0.

    ``origin_node`` is accepted so the caller can record where the user was
    sitting, and is used *only* to fill in ``ElectionResult.origin_node``. It is
    not read anywhere in candidate generation or scoring.

    The resource levers are the coordinator's own planning inputs:
    ``memory_guard_tier`` (``None`` means each Mac's saved tier), the admission
    percentile (p05 floor, p50/live default, p95 only with explicit consent),
    ``table_offload`` (``engram``/``ple``/``none``) and routed-expert offload.
    Passing them here is what keeps the election from refusing a model the
    coordinator would happily launch with the same levers.

    ``fabric_links`` is the measured line, as node-ID pairs. When it is given,
    a rank order is only feasible if every consecutive pair has a link; ``None``
    means the fabric was not measured and keeps the nominal-ring behaviour.
    ``two_mac_fallback`` is the user's explicit acceptance of the offered 2-Mac
    plan; it salts the signature so apply/start must repeat that consent.
    """

    _reject_tensor_parallel(tensor_parallel_size)
    tier = None if memory_guard_tier is None else _tier(memory_guard_tier)
    if admission not in ("p05", "p50", "p95"):
        raise ValueError(f"unknown admission percentile: {admission!r}")
    if table_offload not in ("engram", "ple", "none"):
        raise ValueError(f"unknown table offload: {table_offload!r}")
    if not 0 <= moe_expert_offload_resident_fraction <= 1:
        raise ValueError("moe resident fraction must be in [0, 1]")
    moe_bytes = tuple(int(value) for value in moe_offload_layer_bytes)
    if moe_bytes and len(moe_bytes) != model.layer_count:
        raise ValueError("MoE offload layer bytes must cover every model layer")
    if moe_bytes and not moe_expert_offload_resident_fraction:
        raise ValueError("MoE offload bytes require a resident fraction")

    if not nodes:
        raise ValueError("at least one node is required")
    if len(nodes) > MAX_CANDIDATE_NODES:
        raise ElectionError(
            f"coordinator election supports at most {MAX_CANDIDATE_NODES} Macs; "
            f"got {len(nodes)}"
        )
    node_ids = [facts.node_id for facts in nodes]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("node IDs must be unique")
    if context_tokens <= 0:
        raise ValueError("context_tokens must be positive")
    if rank0_extra_bytes < 0:
        raise ValueError("rank0_extra_bytes must be non-negative")

    resolved_policy = policy or ElectionPolicy()
    facts_by_id = {facts.node_id: facts for facts in nodes}
    extra = rank0_extra_bytes + (model.mtp_weight_bytes if mtp else 0)

    considered: list[PlanCandidate] = []
    rejected: list[RejectedCandidate] = []

    for candidate in nodes:
        refusal = _ineligible_reason(candidate, model)
        if refusal:
            rejected.append(
                RejectedCandidate(node_id=candidate.node_id, reason=refusal)
            )
            continue

        rest = [facts.node_id for facts in nodes if facts.node_id != candidate.node_id]
        failures: list[str] = []
        attempts = 0
        for peers in permutations(rest):
            attempts += 1
            rank_order = (candidate.node_id, *peers)
            try:
                plan = _plan_for_order(
                    rank_order,
                    facts_by_id,
                    model,
                    context_tokens=context_tokens,
                    rank0_extra_bytes=extra,
                    workload_profile=workload_profile,
                    microbatch_size=microbatch_size,
                    memory_guard_tier=tier,
                    admission=admission,
                    table_offload=table_offload,
                    moe_offload_layer_bytes=moe_bytes,
                    moe_expert_offload_resident_fraction=(
                        moe_expert_offload_resident_fraction
                    ),
                )
            except PlanningError as exc:
                failures.append(str(exc))
                continue
            if fabric_links is not None:
                missing = missing_line_links(rank_order, fabric_links)
                if missing:
                    failures.append(
                        "no measured link between "
                        + " and ".join(
                            f"{first} and {second}" for first, second in missing
                        )
                    )
                    continue
            considered.append(
                _candidate_from_plan(
                    plan,
                    rank_order=rank_order,
                    rank0_extra_bytes=extra,
                    context_tokens=context_tokens,
                )
            )

        if attempts and not any(
            candidate_node.rank_order[0] == candidate.node_id
            for candidate_node in considered
        ):
            rejected.append(
                RejectedCandidate(
                    node_id=candidate.node_id,
                    reason=(
                        "no legal pipeline split with this Mac as rank 0: "
                        + _summarize_failures(failures)
                    ),
                    attempts=attempts,
                    details=tuple(dict.fromkeys(failures))[:3],
                )
            )

    ordered = rank_candidates(considered, nodes, policy=resolved_policy)

    if not ordered:
        result = ElectionResult(
            feasible=False,
            coordinator_node="",
            rank_order=(),
            endpoint=None,
            selected=None,
            considered=(),
            rejected=tuple(rejected),
            policy=resolved_policy,
            origin_node=origin_node or "",
            plan_signature="",
            reasons=(
                "no Mac can host rank 0 for this model at this context",
                *(f"{item.node_id}: {item.reason}" for item in rejected),
            ),
            tensor_parallel_size=1,
        )
        raise NoFeasiblePlanError(
            "no feasible coordinator: "
            + _summarize_failures([item.reason for item in rejected]),
            result,
        )

    selected = ordered[0]
    runner_up = ordered[1] if len(ordered) > 1 else None
    coordinator = facts_by_id[selected.coordinator_node]
    signature = selected.plan_hash
    if two_mac_fallback:
        # The explicit consent is part of what the user approved: a caller that
        # only replays the rank order must not accidentally accept a fallback.
        signature = hashlib.sha256(
            f"{signature}:two_mac_fallback".encode()
        ).hexdigest()
    result = ElectionResult(
        feasible=True,
        coordinator_node=selected.coordinator_node,
        rank_order=selected.rank_order,
        endpoint=EndpointPlaceholder(
            node_id=coordinator.node_id,
            host=coordinator.endpoint_host,
        ),
        selected=selected,
        considered=tuple(ordered),
        rejected=tuple(rejected),
        policy=resolved_policy,
        origin_node=origin_node or "",
        plan_signature=signature,
        reasons=_reasons(selected, runner_up, facts_by_id, rejected, resolved_policy),
        tensor_parallel_size=1,
    )
    return result


def _reject_tensor_parallel(tensor_parallel_size: int) -> None:
    if tensor_parallel_size == 1:
        return
    if tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be at least 1")
    # §8.1: the assistant API rejects TP with a clear message instead of
    # silently planning PP behind the caller's back.
    raise UnsupportedParallelismError(
        f"tensor parallelism ({tensor_parallel_size}-way) is not supported by "
        "the Cluster Assistant; the MVP plans pipeline parallelism only"
    )


def _ineligible_reason(facts: NodeFacts, model: ModelLayout) -> str:
    if not facts.healthy:
        return "Mac is not healthy"
    # Stage the whole checkpoint, draft head included: a Mac missing the model
    # needs room for every byte that has to be copied, not just layer weights.
    needed = model.total_weight_bytes + model.mtp_weight_bytes
    if not facts.can_stage(needed):
        return (
            "model is not present and there is not enough disk to stage it "
            f"({_gib(needed)} GiB needed)"
        )
    return ""


def rank_candidates(
    candidates: Sequence[PlanCandidate],
    nodes: Sequence[NodeFacts],
    *,
    policy: ElectionPolicy | None = None,
) -> tuple[PlanCandidate, ...]:
    """Order feasible candidates best-first under the election policy.

    Public because a caller that persisted a set of plans (or re-ran a subset)
    must rank them the same way the election did. Infeasible candidates are
    dropped rather than sorted last: they were never electable.
    """

    facts_by_id = {facts.node_id: facts for facts in nodes}
    resolved = policy or ElectionPolicy()
    unknown = [
        candidate.coordinator_node
        for candidate in candidates
        if candidate.feasible and candidate.coordinator_node not in facts_by_id
    ]
    if unknown:
        raise ValueError(
            "candidate rank-0 Mac has no NodeFacts: " + ", ".join(sorted(set(unknown)))
        )
    feasible = [candidate for candidate in candidates if candidate.feasible]
    return tuple(sorted(feasible, key=lambda item: _score(item, facts_by_id, resolved)))


def _plan_for_order(
    rank_order: Sequence[str],
    facts_by_id: dict[str, NodeFacts],
    model: ModelLayout,
    *,
    context_tokens: int,
    rank0_extra_bytes: int,
    workload_profile: ExecutionProfileName,
    microbatch_size: int,
    memory_guard_tier: str | None = None,
    admission: str = "p05",
    table_offload: str = "none",
    moe_offload_layer_bytes: Sequence[int] = (),
    moe_expert_offload_resident_fraction: float = 0.0,
) -> ShardPlan:
    """Ask the oMLX planner for one rank order.

    Admission becomes the budget: ``reserve_bytes`` is what the Mac may *not*
    use, so the plan's headroom is measured against measured admission rather
    than installed RAM. The rank-0 extra is charged through the planner's
    existing head mechanism, which puts it on the last layer and therefore on
    rank 0, so it goes through the same admission check as any other weight.

    The offload switches are passed straight to ``plan_hybrid``; it already owns
    the SSD and routed-expert arithmetic, so this function must not rederive it.
    """

    budgets = [
        budget_for(
            facts_by_id[node_id],
            rank=rank,
            tier=memory_guard_tier,
            admission=admission,
        )
        for rank, node_id in enumerate(rank_order)
    ]
    return plan_hybrid(
        model,
        budgets,
        tensor_parallel_size=1,
        workload_profile=workload_profile,
        microbatch_size=microbatch_size,
        context_tokens=context_tokens,
        head_rank_extra_bytes=rank0_extra_bytes,
        engram_ssd_offload=table_offload != "none",
        moe_offload_layer_bytes=tuple(moe_offload_layer_bytes),
        moe_expert_offload_resident_fraction=(
            moe_expert_offload_resident_fraction
        ),
    )


def budget_for(
    facts: NodeFacts,
    *,
    rank: int = 0,
    tier: str | None = None,
    admission: str = "p05",
) -> NodeBudget:
    """The planner budget one Mac contributes at one pipeline rank.

    Public because this is the *only* place measured admission becomes a
    ``NodeBudget``. The election needs it to plan, and a recommendation needs it
    to ask the same question about a model the measurement did not cover; a
    second copy of these three lines would be a second admission rule, and the
    two would eventually disagree about what a Mac can hold.

    ``reserve_bytes`` is what the Mac may *not* use, so every headroom the
    planner reports is measured against measured admission rather than installed
    RAM (see ``measurement.py`` and ``node_role.py``).

    ``admission`` chooses the percentile the plan is built against. p05 is the
    measured safety floor and the default for a direct call; the solver plans
    against p50/live admission and only tries p95 with explicit consent. The
    floor is always enforced: a measured p50 below p05 is never used. Rank 0
    additionally leaves ``RANK0_ADMISSION_MARGIN_BYTES`` for the supervisor and
    for the live-start drift, but never below that p05 floor.
    """

    capacity = facts.capacity_bytes
    floor = min(facts.admission_bytes, capacity)
    if admission == "p50":
        chosen = facts.admission_p50_bytes or floor
    elif admission == "p95":
        chosen = facts.admission_p95_bytes or facts.admission_p50_bytes or floor
    else:
        chosen = floor
    # A raised tier is a real lever: the guard keeps a smaller reserve, so the
    # rank may hold more than it held under the saved tier. Ask oMLX's own tier
    # arithmetic; never invent a new reserve. The measured p95 (the optimistic
    # end of what this Mac actually admitted) still caps it.
    if tier is not None:
        chosen_tier = normalize_memory_guard_tier(tier)
        if chosen_tier != facts.memory_guard_tier:
            from ...process_memory_enforcer import tier_reserve_bytes

            static = capacity - int(tier_reserve_bytes(chosen_tier, capacity))
            optimistic = facts.admission_p95_bytes or capacity
            chosen = max(chosen, min(static, optimistic))
    admission_bytes = min(max(chosen, floor), capacity)
    if rank == 0:
        admission_bytes = max(
            floor, admission_bytes - RANK0_ADMISSION_MARGIN_BYTES
        )
    return NodeBudget(
        node_id=facts.node_id,
        capacity_bytes=capacity,
        reserve_bytes=capacity - admission_bytes,
        rank=rank,
        memory_guard_tier=tier or facts.memory_guard_tier,
        performance=(
            None
            if facts.compute is None or not facts.compute.plausible()
            else facts.compute.performance_profile(facts.node_id, rank)
        ),
    )


def _candidate_from_plan(
    plan: ShardPlan,
    *,
    rank_order: tuple[str, ...],
    rank0_extra_bytes: int,
    context_tokens: int,
) -> PlanCandidate:
    assignments = sorted(plan.assignments, key=lambda item: item.rank)
    rank0 = next((item for item in assignments if item.rank == 0), None)
    if rank0 is None or rank0.node_id != rank_order[0]:
        raise ElectionError(
            "planner returned a plan whose rank 0 is not the candidate Mac"
        )
    # Rank 0 must own the tail. If this ever changes in the planner, charging
    # the head to the last layer would silently stop landing it on rank 0.
    if rank0.end_layer != plan.model.layer_count:
        raise ElectionError(
            "planner returned a plan whose rank 0 does not hold the tail layers"
        )

    critical_rank = _critical_rank(assignments)
    stages = tuple(
        StagePlan(
            rank=item.rank,
            node_id=item.node_id,
            first_layer=item.start_layer,
            last_layer=item.end_layer,
            layer_count=item.layer_count,
            resident_bytes=item.planned_weight_bytes,
            admission_bytes=item.capacity_bytes - item.reserve_bytes,
            headroom_bytes=item.headroom_bytes,
            predicted_seconds=item.predicted_stage_seconds,
            critical=item.rank == critical_rank,
        )
        for item in assignments
    )

    return PlanCandidate(
        rank_order=rank_order,
        stages=stages,
        feasible=True,
        reason=_describe_plan(stages),
        rank0_extra_bytes=rank0_extra_bytes,
        rank0_headroom_bytes=rank0.headroom_bytes,
        rank0_admission_bytes=rank0.capacity_bytes - rank0.reserve_bytes,
        min_headroom_bytes=min(stage.headroom_bytes for stage in stages),
        critical_stage_seconds=_critical_seconds(assignments),
        context_tokens=context_tokens,
        plan_hash=plan.plan_hash,
        shard_plan=plan,
    )


def _critical_rank(assignments: Sequence[PipelineAssignment]) -> int | None:
    """The stage that decides throughput: slowest predicted, else tightest."""

    seconds = {
        item.rank: item.predicted_stage_seconds
        for item in assignments
        if item.predicted_stage_seconds is not None
    }
    if seconds:
        return max(seconds, key=lambda rank: seconds[rank] or 0.0)
    return min(assignments, key=lambda item: item.headroom_bytes).rank


def _critical_seconds(assignments: Sequence[PipelineAssignment]) -> float | None:
    """Predicted slowest stage, or ``None`` when nothing was measured.

    ``None`` is not zero: an unmeasured plan cannot claim to be faster than a
    measured one, so scoring treats it as worst-case (see ``_score``).
    """

    values = [
        item.predicted_stage_seconds
        for item in assignments
        if item.predicted_stage_seconds is not None
    ]
    if not values:
        return None
    return max(values)


def _score(
    candidate: PlanCandidate,
    facts_by_id: dict[str, NodeFacts],
    policy: ElectionPolicy,
) -> tuple[Any, ...]:
    """Ascending sort key: first element decides, later ones break ties.

    Order is §7.5's, with one deliberate reading: a plan with no measurement
    cannot outrank a measured one, because "unknown" is not "fast".
    """

    critical = (
        float("inf")
        if candidate.critical_stage_seconds is None
        else candidate.critical_stage_seconds
    )
    rank0_facts = facts_by_id[candidate.coordinator_node]
    return (
        critical,
        -candidate.min_headroom_bytes,
        -candidate.rank0_headroom_bytes,
        _coordinator_preference(rank0_facts),
        -candidate.rank0_admission_bytes if policy.free_admission_preference else 0,
        candidate.rank_order,
    )


def _coordinator_preference(facts: NodeFacts) -> tuple[int, int, int]:
    """§7.5 step 4: a stable host with the model already on it starts sooner."""

    return (
        0 if facts.model_present else 1,
        0 if facts.endpoint_host else 1,
        0 if facts.link.transport != "tcp" else 1,
    )


_CRITERIA = (
    "the fastest predicted critical stage",
    "the most headroom on the tightest stage",
    "the most rank-0 headroom",
    "coordinator suitability",
    "free admission as rank 0",
    "the rank order",
)


def _deciding_criterion(
    selected: PlanCandidate,
    runner_up: PlanCandidate,
    facts_by_id: dict[str, NodeFacts],
    policy: ElectionPolicy,
) -> str:
    left = _score(selected, facts_by_id, policy)
    right = _score(runner_up, facts_by_id, policy)
    for index, (mine, theirs) in enumerate(zip(left, right)):
        if mine != theirs:
            return _CRITERIA[index] if index < len(_CRITERIA) else "the rank order"
    return "the rank order"


def _reasons(
    selected: PlanCandidate,
    runner_up: PlanCandidate | None,
    facts_by_id: dict[str, NodeFacts],
    rejected: Sequence[RejectedCandidate],
    policy: ElectionPolicy,
) -> tuple[str, ...]:
    coordinator = facts_by_id[selected.coordinator_node]
    if runner_up is None:
        headline = (
            f"{coordinator.node_id} is the only Mac that can hold rank 0 for "
            "this model at this context"
        )
    else:
        headline = (
            f"{coordinator.node_id} was elected on "
            f"{_deciding_criterion(selected, runner_up, facts_by_id, policy)}"
        )
    details = [headline]
    if selected.critical_stage_seconds is not None:
        critical = selected.critical_stage
        if critical is not None:
            details.append(
                f"slowest stage: rank {critical.rank} on {critical.node_id} "
                f"({selected.critical_stage_seconds:.3f} s)"
            )
    details.append(
        f"rank 0 holds layers {selected.stages[0].first_layer}-"
        f"{selected.stages[0].last_layer} with "
        f"{_gib(selected.rank0_headroom_bytes)} GiB headroom inside "
        f"{_gib(selected.rank0_admission_bytes)} GiB admission"
    )
    if selected.rank0_extra_bytes:
        details.append(
            f"rank 0 also holds {_gib(selected.rank0_extra_bytes)} GiB of "
            "head/coordinator memory"
        )
    for item in rejected:
        details.append(f"{item.node_id} rejected: {item.reason}")
    return tuple(details)


def _describe_plan(stages: Sequence[StagePlan]) -> str:
    groups = ", ".join(
        f"rank {stage.rank} {stage.node_id}: {stage.layer_count} layers "
        f"({_gib(stage.resident_bytes)} GiB)"
        for stage in stages
    )
    return groups


def _summarize_failures(failures: Sequence[str]) -> str:
    unique = list(dict.fromkeys(failures))
    if not unique:
        return "no rank order produced a plan"
    if len(unique) == 1:
        return unique[0]
    return f"{unique[0]} (and {len(unique) - 1} other reason(s))"


def _gib(value: int) -> str:
    return f"{value / GIB:.1f}"


def _tier(value: str) -> str:
    return normalize_memory_guard_tier(value)


def _gib_to_bytes(value: float) -> int:
    return int(value * GIB)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
