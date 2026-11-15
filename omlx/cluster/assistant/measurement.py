# SPDX-License-Identifier: Apache-2.0
"""Turn oMLX's own measurements into coordinator-election inputs.

``election.NodeFacts`` is a pure decision type: it says nothing about where its
numbers came from. This module is the one place that reads them out of what
oMLX already measures, so the election never has to care whether a Mac's
admission came from the memory guard's ceiling, a wired-limit ``sysctl`` or a
role's policy — and so that a measurement which never happened can stay
missing instead of becoming a zero.

Three rules are deliberate.

1. **Admission is what was measured, not a spread.** ``NodeFacts.admission_p05``
   is the budget every plan is built against. When only one admission number
   exists, that number *is* p05 and p50/p95 stay ``None``. Synthesising a
   confidence interval would turn one number into three that nobody measured.
2. **Compute is only ever a measurement.** A Mac without a performance probe
   or a real-model calibration gets ``compute=None``, which the election
   scores as worst case — never as fast. Nothing here estimates a rate from a
   chip name. When both exist, the real calibration wins: ranking rests on
   real completions, and the synthetic probe stays the fallback for a Mac
   that could not run the calibration model (decision 2026-11-13).
3. **Installed RAM is not a GPU ceiling.** ``node_role.py`` documents the
   incident: a 128 GiB MacBook addresses 107.5 GiB. A budget whose capacity came
   from ``hw.memsize`` is refused by default, because planning against it
   proposes a stage the memory guard then rejects at load.

The module is pure: no HTTP, no SSH, no model loading, no GPU. Callers pass in
the payloads oMLX's ``/admin/api/cluster/node-budgets`` route or performance
probe already produced.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

from ..autoconfigure import choose_backend
from ..link_bandwidth import LinkBandwidth
from ..node_role import role_for
from ..performance import NodePerformanceProfile
from .election import ComputeProfile, LinkHints, NodeFacts

GIB = 1024**3

# ``NodeBudgetSuggestion.capacity_source`` values that describe what the GPU can
# actually address. Everything else — notably ``installed_ram`` — is a larger
# number than the Mac can hand a rank.
ADDRESSABLE_CAPACITY_SOURCES = frozenset(
    {"admission_ceiling", "metal_cap", "recommended_working_set"}
)
INSTALLED_RAM_CAPACITY_SOURCE = "installed_ram"

WARNING_UNREADABLE_CAPACITY = "unreadable_capacity"
WARNING_UNVERIFIED_CAPACITY = "unverified_capacity_source"
WARNING_INSTALLED_RAM_CAPACITY = "installed_ram_capacity"
WARNING_UNREADABLE_ADMISSION = "unreadable_admission"
WARNING_ADMISSION_ABOVE_CAPACITY = "admission_above_capacity"
WARNING_ADMISSION_SPREAD_WITHOUT_P05 = "admission_spread_without_p05"
WARNING_ADMISSION_SPREAD_INCONSISTENT = "admission_spread_inconsistent"
WARNING_ADMISSION_DERIVED_FROM_ROLE = "admission_derived_from_role"
WARNING_UNMEASURED_COMPUTE = "unmeasured_compute"
WARNING_PERFORMANCE_NODE_MISMATCH = "performance_profile_node_mismatch"

# The cluster default port. It is only ever used when a Mac's own
# ``ServerSettings.port`` cannot be read — never as a guess between two known
# numbers, and never as 0, which no client can dial.
DEFAULT_ENDPOINT_PORT = 8099

# Addresses that identify this process, not this Mac on the network. An
# endpoint other Macs dial is never loopback: binding there is exactly the
# mistake the design notes names.
LOOPBACK_ENDPOINT_HOSTS = frozenset(
    {"", "127.0.0.1", "localhost", "::1", "0.0.0.0", "::"}
)


@dataclass(frozen=True)
class MeasurementWarning:
    """One thing the measurement layer could not measure, or would not trust.

    Warnings are returned, never logged and swallowed: an optimistically planned
    cluster is exactly what the admission rules in this file exist to prevent,
    so the caller has to be able to show this to a person.
    """

    code: str
    message: str
    node_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "node_id": self.node_id, "message": self.message}


@dataclass(frozen=True)
class MeasuredNode:
    """One Mac, as oMLX measured it.

    ``capacity_bytes``/``capacity_source`` come from ``NodeBudgetSuggestion``;
    ``admission_p05_bytes`` is an independently measured admission floor when
    the caller has one, and otherwise the role's planning ceiling is derived
    from the capacity. ``performance`` is a ``NodePerformanceProfile`` from the
    synthetic performance probe; ``calibration`` is a real-model
    ``ComputeProfile`` from ``calibration.calibrate_node`` and is preferred
    over the probe when present. ``link`` is the slowest ``LinkBandwidth``
    incident to this Mac (``transport_kinds`` covers the case where only the
    fabric kinds are known).
    """

    node_id: str
    capacity_bytes: int
    capacity_source: str = "admission_ceiling"
    role: str = "headless"
    reserve_bytes: int = 0
    admission_p05_bytes: int | None = None
    admission_p50_bytes: int | None = None
    admission_p95_bytes: int | None = None
    performance: NodePerformanceProfile | None = None
    #: A real-model calibration profile for this Mac, as
    #: ``calibration.calibrate_node`` produces it. Preferred over the synthetic
    #: ``performance`` probe when both exist (decision 2026-11-13); ``None``
    #: means no calibration ran and the probe stays the fallback.
    calibration: ComputeProfile | None = None
    transport_kinds: tuple[str, ...] = ()
    link: LinkBandwidth | None = None
    model_present: bool = True
    #: Checkpoint paths this Mac's own catalogue reported. Empty means "not
    #: measured", never "holds nothing": without it a recommendation could only
    #: answer "already present" for the checkpoint the measurement covered.
    model_paths: tuple[str, ...] = ()
    disk_free_bytes: int | None = None
    healthy: bool = True
    endpoint_host: str = ""
    # The port this Mac serves on, resolved where its settings could be read.
    # ``None`` means "not measured"; 8099 is decided by
    # ``measured_endpoint_port`` and never invented here.
    endpoint_port: int | None = None
    #: The process memory policy that Mac saved. ``balanced`` is the documented
    #: default when nothing could be read.
    memory_guard_tier: str = "balanced"
    #: ``iogpu.wired_limit_mb`` when the Mac reported it. ``None`` is unknown.
    wired_limit_mb: int | None = None
    #: The absolute cluster interpreter that Mac reported, or "" when unknown.
    python_executable: str = ""

    def __post_init__(self) -> None:
        if not self.node_id or not self.node_id.strip():
            raise ValueError("node_id is required")
        if self.capacity_bytes < 0:
            raise ValueError("capacity_bytes must be non-negative")
        if self.reserve_bytes < 0:
            raise ValueError("reserve_bytes must be non-negative")
        if self.disk_free_bytes is not None and self.disk_free_bytes < 0:
            raise ValueError("disk_free_bytes must be non-negative")
        if self.endpoint_port is not None and not 1 <= self.endpoint_port <= 65535:
            raise ValueError("endpoint_port must be between 1 and 65535")
        if self.wired_limit_mb is not None and self.wired_limit_mb < 0:
            raise ValueError("wired_limit_mb must be non-negative")
        for name in (
            "admission_p05_bytes",
            "admission_p50_bytes",
            "admission_p95_bytes",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when measured")

    @property
    def capacity_is_addressable(self) -> bool:
        """Whether the capacity is what the GPU can address, not installed RAM."""

        return self.capacity_source in ADDRESSABLE_CAPACITY_SOURCES

    @property
    def rolled_up_admission_bytes(self) -> int:
        """The role's planning ceiling for this capacity. Not a measurement."""

        return role_for(self.role).usable_for(self.capacity_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "capacity_bytes": self.capacity_bytes,
            "capacity_source": self.capacity_source,
            "role": self.role,
            "reserve_bytes": self.reserve_bytes,
            "admission_p05_bytes": self.admission_p05_bytes,
            "admission_p50_bytes": self.admission_p50_bytes,
            "admission_p95_bytes": self.admission_p95_bytes,
            "performance": (
                None if self.performance is None else self.performance.to_dict()
            ),
            "calibration": (
                None if self.calibration is None else self.calibration.to_dict()
            ),
            "transport_kinds": list(self.transport_kinds),
            "link": None if self.link is None else self.link.to_dict(),
            "model_present": self.model_present,
            "model_paths": list(self.model_paths),
            "disk_free_bytes": self.disk_free_bytes,
            "healthy": self.healthy,
            "endpoint_host": self.endpoint_host,
            "endpoint_port": self.endpoint_port,
            "memory_guard_tier": self.memory_guard_tier,
            "wired_limit_mb": self.wired_limit_mb,
            "python_executable": self.python_executable,
        }


@dataclass(frozen=True)
class MeasurementSet:
    """The election's node input, plus when it was taken and what was refused.

    ``measured_at`` is the timestamp the caller measured at; it is *not* taken
    while building the facts, because a timestamp from this process would claim
    a freshness the numbers do not have. Empty means "unknown", and that is
    allowed to reach the plan object as an empty string.
    """

    nodes: tuple[NodeFacts, ...]
    measured_at: str = ""
    warnings: tuple[MeasurementWarning, ...] = ()

    @property
    def measured(self) -> bool:
        return bool(self.nodes)

    def warning_messages(self) -> tuple[str, ...]:
        return tuple(
            f"{item.node_id}: {item.message}" if item.node_id else item.message
            for item in self.warnings
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "measured_at": self.measured_at,
            "warnings": [item.to_dict() for item in self.warnings],
        }


def measurement_timestamp() -> str:
    """Now, in the format a probe writes. Callers pass this in explicitly."""

    return datetime.now(UTC).isoformat()


def measured_endpoint_port(*, configured_port: Any = None, settings: Any = None) -> int:
    """The port a Mac serves on — never 0, and 8099 only when unreadable.

    ``configured_port`` is what the measuring side already knows about this Mac
    (the HTTP port its paired record advertises, or a value a probe carried).
    ``settings`` is that Mac's own ``ServerSettings`` when this process owns the
    Mac. Anything unusable — ``0``, ``None``, out of range, a settings object
    without a port — falls back to the documented cluster default, because a Mac
    whose port nobody could read is a Mac whose port we do not know.
    """

    for candidate in (configured_port, getattr(settings, "port", None)):
        port = _port_int(candidate)
        if port is not None:
            return port
    return DEFAULT_ENDPOINT_PORT


def reachable_endpoint_host(*candidates: Any) -> str:
    """The first address another Mac could dial, or "" when there is none.

    Loopback is not an endpoint: discovery and pairing report the private
    address a peer reached this Mac on, and only that address may become a URL.
    Returning "" keeps "not measured" separate from "this Mac's own name".
    """

    for value in candidates:
        host = str(value or "").strip()
        if host and host not in LOOPBACK_ENDPOINT_HOSTS:
            return host
    return ""


def measured_node_from_budget(payload: Mapping[str, Any]) -> MeasuredNode:
    """One Mac from a ``/admin/api/cluster/node-budgets`` node entry.

    That payload already carries the measured ceiling (``capacity_bytes``), the
    role reserve, the resulting ``usable_bytes`` and where the ceiling came
    from — the only place the installed-RAM mistake can be caught. Any admission
    spread, probe profile, real calibration, link or disk figure a caller has is
    folded in when present; nothing is guessed when it is absent.
    """

    node_id = str(payload.get("node_id") or "").strip()
    if not node_id:
        raise ValueError("node budget payload has no node_id")
    disk_free = _optional_int(payload.get("disk_free_bytes"))

    return MeasuredNode(
        node_id=node_id,
        capacity_bytes=_optional_int(payload.get("capacity_bytes")) or 0,
        capacity_source=str(payload.get("capacity_source") or ""),
        role=str(payload.get("role") or "headless"),
        reserve_bytes=_optional_int(payload.get("reserve_bytes")) or 0,
        admission_p05_bytes=(
            _positive_int(payload.get("admission_p05_bytes"))
            # ``usable_bytes`` is what the node-budgets route reported and what
            # oMLX's own planner would hand this Mac, so it is the admission
            # floor when no separate admission measurement exists.
            or _positive_int(payload.get("usable_bytes"))
        ),
        admission_p50_bytes=_positive_int(payload.get("admission_p50_bytes")),
        admission_p95_bytes=_positive_int(payload.get("admission_p95_bytes")),
        performance=_performance_from_payload(payload.get("performance")),
        calibration=_calibration_from_payload(payload.get("calibration")),
        transport_kinds=_transport_kinds(payload),
        link=_link_from_payload(payload.get("link")),
        model_present=bool(payload.get("model_present", True)),
        model_paths=tuple(
            str(item)
            for item in (payload.get("model_paths") or ())
            if isinstance(item, str) and item.strip()
        ),
        disk_free_bytes=disk_free,
        healthy=bool(payload.get("healthy", True)),
        endpoint_host=str(payload.get("endpoint_host") or ""),
        endpoint_port=_port_int(payload.get("endpoint_port")),
        memory_guard_tier=str(payload.get("memory_guard_tier") or "balanced"),
        wired_limit_mb=_optional_int(payload.get("wired_limit_mb")),
        python_executable=str(payload.get("python_executable") or ""),
    )


def with_calibration(
    node: MeasuredNode, profile: ComputeProfile | None
) -> MeasuredNode:
    """Fold a real calibration profile into a measured Mac, as a copy.

    This is how a ``calibrate_node`` result becomes the node's compute: the
    election prefers it over the synthetic probe, and ``None`` clears it. The
    admission/memory fields are untouched — calibration measures compute only.
    """

    return replace(node, calibration=profile)


def measure_nodes(
    nodes: Sequence[MeasuredNode],
    *,
    measured_at: str | None = None,
    allow_installed_ram_capacity: bool = False,
) -> MeasurementSet:
    """Build ``NodeFacts`` for every Mac that was measured well enough.

    ``allow_installed_ram_capacity`` exists for a Mac whose GPU ceiling cannot be
    read at all. It is off by default: admitting a Mac on installed RAM is how a
    plan gets approved against 128 GiB of a 107.5 GiB machine. Either way the
    node is named in ``warnings`` — a Mac may never disappear from a plan
    quietly.
    """

    facts: list[NodeFacts] = []
    warnings: list[MeasurementWarning] = []
    for node in nodes:
        node_facts, node_warnings = _facts_for(
            node, allow_installed_ram_capacity=allow_installed_ram_capacity
        )
        warnings.extend(node_warnings)
        if node_facts is not None:
            facts.append(node_facts)
    return MeasurementSet(
        nodes=tuple(facts),
        measured_at=measured_at or "",
        warnings=tuple(warnings),
    )


def measure_budget_payloads(
    payloads: Sequence[Mapping[str, Any]],
    *,
    measured_at: str | None = None,
    allow_installed_ram_capacity: bool = False,
) -> MeasurementSet:
    """Shortcut for the node-budgets route's own ``{"nodes": [...]}`` entries."""

    return measure_nodes(
        [measured_node_from_budget(payload) for payload in payloads],
        measured_at=measured_at,
        allow_installed_ram_capacity=allow_installed_ram_capacity,
    )


def _facts_for(
    node: MeasuredNode, *, allow_installed_ram_capacity: bool
) -> tuple[NodeFacts | None, list[MeasurementWarning]]:
    warnings: list[MeasurementWarning] = []

    def warn(code: str, message: str) -> None:
        warnings.append(
            MeasurementWarning(code=code, node_id=node.node_id, message=message)
        )

    if node.capacity_bytes <= 0:
        warn(
            WARNING_UNREADABLE_CAPACITY,
            "no memory ceiling was measured for this Mac, so it cannot be planned",
        )
        return None, warnings

    if not node.capacity_is_addressable:
        if node.capacity_source == INSTALLED_RAM_CAPACITY_SOURCE:
            warn(
                WARNING_INSTALLED_RAM_CAPACITY,
                "capacity is installed RAM, not what the GPU can address; a plan "
                "built on it proposes a stage the memory guard rejects at load",
            )
        else:
            source = node.capacity_source or "an unrecorded source"
            warn(
                WARNING_UNVERIFIED_CAPACITY,
                f"capacity comes from {source}, which does not say what the GPU can "
                "address, so this Mac cannot be planned against",
            )
        if not allow_installed_ram_capacity:
            return None, warnings

    admission, p50, p95, admission_warnings = _admission_for(node)
    warnings.extend(admission_warnings)
    if admission is None:
        return None, warnings

    performance = node.performance
    if performance is not None and performance.node_id != node.node_id:
        warn(
            WARNING_PERFORMANCE_NODE_MISMATCH,
            f"measured performance belongs to {performance.node_id}, not this Mac; "
            "it is ignored rather than attributed to the wrong node",
        )
        performance = None
    # A real calibration outranks the synthetic probe: ranking must rest on
    # real completions (decision 2026-11-13). The probe stays the fallback for
    # a Mac that could not run the calibration model.
    compute = (
        node.calibration
        if node.calibration is not None
        else _compute_profile(performance)
    )
    if compute is None:
        warn(
            WARNING_UNMEASURED_COMPUTE,
            "no performance profile was measured, so the election scores this Mac "
            "as worst case instead of guessing a rate",
        )

    facts = NodeFacts(
        node_id=node.node_id,
        capacity_gib=node.capacity_bytes / GIB,
        admission_p05_gib=admission / GIB,
        admission_p50_gib=_gib(p50),
        admission_p95_gib=_gib(p95),
        compute=compute,
        model_present=node.model_present,
        # Carried through so a recommendation can answer "already present" for
        # any model the measured Macs listed, not only the measured checkpoint.
        model_paths=node.model_paths,
        disk_free_gib=_gib(node.disk_free_bytes),
        link=_link_hints(node),
        healthy=node.healthy,
        endpoint_host=node.endpoint_host,
        memory_guard_tier=node.memory_guard_tier,
        wired_limit_mb=node.wired_limit_mb,
        python_executable=node.python_executable,
    )
    return facts, warnings


def _admission_for(
    node: MeasuredNode,
) -> tuple[int | None, int | None, int | None, list[MeasurementWarning]]:
    """The measured admission floor, or the role's ceiling over a measured cap."""

    warnings: list[MeasurementWarning] = []
    p05 = node.admission_p05_bytes
    p50 = node.admission_p50_bytes
    p95 = node.admission_p95_bytes

    def warn(code: str, message: str) -> None:
        warnings.append(
            MeasurementWarning(code=code, node_id=node.node_id, message=message)
        )

    if p05 is None:
        if p50 is not None or p95 is not None:
            warn(
                WARNING_ADMISSION_SPREAD_WITHOUT_P05,
                "a p50/p95 admission was reported without the p05 floor; the "
                "spread is dropped instead of being used as the budget",
            )
            p50 = p95 = None
        p05 = node.rolled_up_admission_bytes
        if p05 <= 0:
            warn(
                WARNING_UNREADABLE_ADMISSION,
                "no admission was measured and the role's reserve leaves nothing "
                "for a rank",
            )
            return None, None, None, warnings
        warn(
            WARNING_ADMISSION_DERIVED_FROM_ROLE,
            f"admission is the {node.role} role's planning ceiling "
            f"({p05 / GIB:.1f} GiB) over the measured capacity, not a measured "
            "admission",
        )
    elif (p50 is not None and p50 < p05) or (p95 is not None and p95 < p05):
        warn(
            WARNING_ADMISSION_SPREAD_INCONSISTENT,
            "a reported admission percentile is below the measured p05 floor; the "
            "spread is dropped rather than reversed",
        )
        p50 = p95 = None

    if p05 > node.capacity_bytes:
        warn(
            WARNING_ADMISSION_ABOVE_CAPACITY,
            f"measured admission {p05 / GIB:.1f} GiB exceeds the "
            f"{node.capacity_bytes / GIB:.1f} GiB ceiling; it is clamped, because "
            "a Mac cannot admit more than it can address",
        )
        p05 = node.capacity_bytes
    return p05, p50, p95, warnings


def _compute_profile(
    performance: NodePerformanceProfile | None,
) -> ComputeProfile | None:
    """The probe's numbers, unchanged. ``None`` stays ``None``."""

    if performance is None:
        return None
    return ComputeProfile(
        decode_weight_bytes_per_second=performance.decode_weight_bytes_per_second,
        prefill_weight_bytes_per_second=performance.prefill_weight_bytes_per_second,
        collective_latency_seconds=performance.collective_latency_seconds,
        collective_bandwidth_bytes_per_second=(
            performance.collective_bandwidth_bytes_per_second
        ),
        backend=performance.backend,
        measured_at=performance.measured_at,
        samples=performance.samples,
    )


@dataclass(frozen=True)
class _DetectedKind:
    """Adapter so ``choose_backend`` can name the transport for one link kind."""

    kind: str


def _link_hints(node: MeasuredNode) -> LinkHints:
    """Fabric hints for the election's coordinator-suitability preference.

    The backend itself is not decided here: ``autoconfigure.choose_backend``
    already owns the kind → backend rule (RDMA → jaccl, Thunderbolt →
    jaccl-ring, everything else → TCP ring). Two answers to that question is how
    a plan and a launch end up disagreeing.
    """

    kinds = tuple(
        dict.fromkeys(
            kind
            for kind in (
                node.link.kind if node.link is not None else "",
                *node.transport_kinds,
            )
            if kind
        )
    )
    transport = (
        choose_backend([_DetectedKind(kind=kind) for kind in kinds])[0]
        if kinds
        else "ring"
    )
    bandwidth_gbps = (
        None
        if node.link is None or node.link.bytes_per_second <= 0
        else node.link.bytes_per_second * 8 / 10**9
    )
    latency_ms = (
        None
        if node.performance is None
        else node.performance.collective_latency_seconds * 1000.0
    )
    return LinkHints(
        transport=transport,
        bandwidth_gbps=bandwidth_gbps,
        latency_ms=latency_ms,
    )


def _performance_from_payload(value: Any) -> NodePerformanceProfile | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return NodePerformanceProfile.from_dict(dict(value))
    except ValueError:
        # A malformed probe record is not a measurement. The node keeps
        # ``compute=None`` and the warning says so.
        return None


def _calibration_from_payload(value: Any) -> ComputeProfile | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return ComputeProfile.from_dict(dict(value))
    except (KeyError, TypeError, ValueError):
        # A malformed calibration record is not a measurement. The node keeps
        # the synthetic probe (or unknown compute) instead.
        return None


def _link_from_payload(value: Any) -> LinkBandwidth | None:
    if not isinstance(value, Mapping):
        return None
    try:
        bytes_per_second = float(value["bytes_per_second"])
    except (KeyError, TypeError, ValueError):
        return None
    if bytes_per_second <= 0:
        return None
    return LinkBandwidth(
        source_node_id=str(value.get("source_node_id") or ""),
        peer_node_id=str(value.get("peer_node_id") or ""),
        bytes_per_second=bytes_per_second,
        source=str(value.get("source") or "assumed"),
        kind=str(value.get("kind") or "unknown"),
    )


def _transport_kinds(payload: Mapping[str, Any]) -> tuple[str, ...]:
    kinds: list[str] = []
    for item in payload.get("transports") or ():
        kind = (
            str(item.get("kind") or "")
            if isinstance(item, Mapping)
            else str(getattr(item, "kind", "") or "")
        )
        if kind:
            kinds.append(kind)
    raw = payload.get("transport_kinds")
    if isinstance(raw, (list, tuple)):
        kinds.extend(str(kind) for kind in raw if kind)
    return tuple(dict.fromkeys(kinds))


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _positive_int(value: Any) -> int | None:
    number = _optional_int(value)
    return number if number is not None and number > 0 else None


def _port_int(value: Any) -> int | None:
    """A usable TCP port, or ``None``. 0 is "no port", not a port."""

    number = _optional_int(value)
    return number if number is not None and 1 <= number <= 65535 else None


def _gib(value: int | None) -> float | None:
    return None if value is None else value / GIB
