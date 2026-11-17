# SPDX-License-Identifier: Apache-2.0
"""Admin API for the Cluster Assistant: measure, status, recommend, plan.

Five routes, the whole decision surface of the assistant::

    POST   /admin/api/cluster/assistant/measure              start a job, return immediately
    GET    /admin/api/cluster/assistant/status               poll that job
    POST   /admin/api/cluster/assistant/recommend            which model should we run?
    POST   /admin/api/cluster/assistant/plan                 plan from the stored facts
    DELETE /admin/api/cluster/assistant/models/{node}/{id}   free space on a short Mac

Why the measurement is a job: the probes read memory ceilings, model
inventories and links over SSH, which takes seconds and fails in ways a plan
may not hide. So ``measure`` persists a ``MeasurementSet`` under a
``measurement_id`` and returns before the first probe finishes, and neither
``recommend`` nor ``plan`` probes at all — they load the record and decide. A
slow probe can therefore never look like a hung plan, and a plan can never be
justified by a number it measured itself.

``recommend`` and ``plan`` answer different questions (the notes). Recommendation
reads only the hardware half of a measurement and may evaluate any candidate
model, because "which model should we run?" is asked before a model is chosen.
``plan`` keeps the strict Chapter 3 match, so the chosen model is measured again
(model-bound) before it is split. Recommendation never weakens that check.

The measurement job does not stage, launch, register a deployment or write the
device registry: it reads budgets, profiles, links, model inventories and disk
space through oMLX's existing probes, and runs the small calibration model on
each Mac that holds it (one warmup plus one measured completion, through the
same admin seam a later post-``ready`` observe measurement will use) so ranking
rests on real compute. Only one job runs at a time; a second ``measure`` is
refused with ``409 measurement_in_progress`` rather than joined, because joining
would hand its caller a measurement of a different node set or model.

Tensor parallelism is refused with an explicit 4xx rather than planned as PP
behind the caller's back (the notes); a candidate that would need it is reported as
unsupported with one action, never offered. Refusals are distinguishable
without parsing prose: ``stale_measurement`` (no usable measurement — one
action: measure), ``no_feasible_candidate`` (measured cluster, no model fits —
choose a different model) and ``no_feasible_plan`` (measured cluster and chosen
model, no legal split — change the context or model). Nothing in this module can
name the app origin as a fallback coordinator: the election has no such branch
and these routes add none.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi import Path as PathParam
from fastapi.responses import JSONResponse

from ..deployment import validate_ssh_target
from ..performance import NodePerformanceProfile
from ..planner import (
    LOCAL_NODE,
    ModelLayout,
    PlanningError,
    inspect_safetensors_layout,
)
from .calibration import (
    CALIBRATION_MODEL_DIR,
    calibrate_node,
    calibration_weight_bytes,
)
from .election import (
    ComputeProfile,
    ElectionError,
    NoFeasiblePlanError,
    UnsupportedParallelismError,
    _reject_tensor_parallel,
)
from .measurement import (
    DEFAULT_ENDPOINT_PORT,
    MeasurementSet,
    MeasurementWarning,
    measure_budget_payloads,
    measured_endpoint_port,
    measured_node_from_budget,
    measurement_timestamp,
    reachable_endpoint_host,
)
from .models import (
    ACTION_ADJUST_PLAN,
    ACTION_CONFIGURE_ENDPOINT,
    NO_FEASIBLE_CANDIDATE_CODE,
    AssistantMeasurementStatus,
    AssistantMeasureNode,
    AssistantMeasureRequest,
    AssistantMeasureResponse,
    AssistantPlanRequest,
    AssistantPlanResponse,
    AssistantRecommendRequest,
    AssistantRecommendResponse,
    build_plan_response,
    build_recommend_response,
    measurement_status,
    plan_error_payload,
)
from .operations import (
    ACTION_DIAGNOSTICS,
    ACTION_FREE_DISK_SPACE,
    ACTION_PAIR_MAC,
    ACTION_PLAN,
    ACTION_RETRY_COORDINATOR,
    ACTION_STATUS,
)
from .performance_store import AssistantPerformanceStore
from .policy import (
    SavedPolicy,
    solve_cluster_plan,
)
from .provisioning import ProvisioningSeams, model_layout_for
from .recommendation import (
    ModelCandidate,
    RecommendationResult,
    load_model_candidate,
    recommend,
)
from .start import RemoteAdminError, delete_model_on_mac, stream_completion
from .state import (
    ACTION_MEASURE,
    STALE_MEASUREMENT_CODE,
    MeasurementEndpoint,
    MeasurementRecord,
    MeasurementStore,
    ModelIdentity,
    measurement_refusal,
)
from .topology import missing_line_links

router = APIRouter(prefix="/admin/api/cluster/assistant", tags=["cluster-assistant"])

logger = logging.getLogger(__name__)

# No Mac can host rank 0 for this model at this context. Documented, tested and
# the only status this route uses for it; the body is the plan object with
# ``feasible`` false, no coordinator, ``code`` and one ``action``.
NO_FEASIBLE_PLAN_STATUS = 409
# The measurement is missing, unknown, running, stale or covers another cluster.
STALE_MEASUREMENT_STATUS = 409
# The measurement is usable but no candidate model fits these Macs.
NO_FEASIBLE_CANDIDATE_STATUS = 409
# One measurement job at a time; the caller polls the running one instead.
MEASUREMENT_IN_PROGRESS_STATUS = 409
MEASUREMENT_NOT_FOUND_STATUS = 404
# A well-formed request the MVP cannot serve, i.e. tensor parallelism.
UNSUPPORTED_PARALLELISM_STATUS = 400

MEASURE_STATUS_PATH = "/admin/api/cluster/assistant/status"

WARNING_LINK_UNMEASURED = "link_unmeasured"
#: A workstation reserve is >= 32 GiB; below this much usable admission the
#: role is meaningless and the Mac is measured as Headless instead.
WARNING_WORKSTATION_TOO_LARGE = "workstation_reserve_too_large"
WORKSTATION_ROLE = "workstation"
DEFAULT_NODE_ROLE = "headless"
WORKSTATION_MIN_USABLE_BYTES = 4 * 1024**3
WARNING_MODEL_INVENTORY_UNREADABLE = "model_inventory_unreadable"
WARNING_DISK_FREE_UNREADABLE = "disk_free_unreadable"
WARNING_LAYOUT_FROM_PEER = "layout_read_on_peer"
WARNING_MODEL_SETTINGS_UNREADABLE = "model_settings_unreadable"

_LOCAL_SSH_TARGETS = frozenset({LOCAL_NODE, "127.0.0.1", "localhost", "::1"})


class MeasureJobError(RuntimeError):
    """A measurement failure with exactly one actionable next step."""

    def __init__(self, code: str, message: str, action: str = ACTION_MEASURE) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.action = action


@dataclass(frozen=True)
class MeasureProbes:
    """The oMLX calls a measurement job may make.

    Every entry is an existing oMLX path; nothing here is a second probe. Tests
    inject fakes so the suite never touches SSH, a model or the validation
    cluster, and the server's own wiring is the default.
    """

    #: (ssh target, python_executable) -> the peer's live memory ceiling.
    admission_ceiling: Callable[..., int]
    #: keyword-only role/ssh/capacity -> a ``NodeBudgetSuggestion``.
    budget: Callable[..., Any]
    #: (ssh targets) -> a ``TransportMatrix``.
    transports: Callable[..., Any]
    #: (ssh target, python_executable) -> the peer's model inventory.
    model_inventory: Callable[..., list[dict[str, Any]]]
    #: (path, ssh_target) -> free bytes on that Mac.
    disk_free: Callable[..., int]
    #: node_id -> that Mac's paired registry record, if any.
    device_record: Callable[[str], Mapping[str, Any] | None]
    #: (ssh target, model path) -> that Mac's planner layout for the checkpoint.
    #: Used when the origin cannot read the model itself, which is the normal
    #: case when the model lives on a bigger Mac.
    model_layout: Callable[..., dict[str, Any]] | None = None
    #: Optional distributed performance probe. It boots a launcher on every Mac
    #: and needs an approved plan, so it is only ever called when a server (or
    #: a test) injects one and the request opts in.
    performance: Callable[..., Sequence[Any]] | None = None
    #: (model path) -> the saved per-model settings this Mac can read. The
    #: resource policy starts from these; ``None`` means "not readable", which
    #: keeps the documented defaults rather than inventing values.
    model_settings: Callable[[str], Mapping[str, Any]] | None = None


_lock = threading.RLock()
_store: MeasurementStore | None = None
_probes: MeasureProbes | None = None
_performance: AssistantPerformanceStore | None = None
_seams: ProvisioningSeams | None = None
_get_settings_manager: Callable[[], Any] | None = None
_get_server_settings: Callable[[], Any] | None = None
_tasks: set[asyncio.Task[Any]] = set()


def set_assistant_getters(
    settings_manager_getter: Callable[[], Any] | None = None,
    server_settings_getter: Callable[[], Any] | None = None,
) -> None:
    """Inject server-owned dependencies without importing ``omlx.server``.

    Called once next to ``set_cluster_getters`` in
    ``omlx/server.py::_register_cluster_routes``: the settings manager supplies
    the base path the measurement store lives under and the server settings
    supply this Mac's own endpoint port. Both are read lazily, because the
    cluster routes are registered before the settings manager exists.
    """

    global _get_settings_manager, _get_server_settings
    _get_settings_manager = settings_manager_getter
    _get_server_settings = server_settings_getter


def configure_assistant_measurements(
    store: MeasurementStore | None = None,
    *,
    probes: MeasureProbes | None = None,
) -> None:
    """Point the assistant at a measurement store and/or its probes.

    Production resolves the store lazily from the settings manager; tests call
    this with a temporary store so no test writes to the user's real ``~/.omlx``
    and no test reaches a Mac.
    """

    global _store, _probes, _performance
    with _lock:
        if store is not None:
            _store = store
            # A new measurement store means a new base path; the compute
            # profiles live beside it and must not leak from a previous store
            # (or a previous test).
            _performance = None
        if probes is not None:
            _probes = probes


def configure_assistant_performance(
    store: AssistantPerformanceStore | None = None,
) -> None:
    """Point the compute-profile store at a store, or let it resolve lazily."""

    global _performance
    with _lock:
        if store is not None:
            _performance = store


def configure_assistant_seams(seams: ProvisioningSeams | None = None) -> None:
    """Point target resolution at injected read-only lookups.

    Production resolves the registry and this Mac's addresses lazily; tests
    pass fakes so no test reads a real registry or the network.
    """

    global _seams
    with _lock:
        if seams is not None:
            _seams = seams


def _provisioning_seams() -> ProvisioningSeams:
    global _seams
    with _lock:
        if _seams is None:
            from .provisioning import default_provisioning_seams

            _seams = default_provisioning_seams()
        return _seams




def default_measure_probes() -> MeasureProbes:
    """The real probes, imported lazily so importing this module stays cheap."""

    from ..launch import probe_remote_admission_ceiling
    from ..model_inventory import remote_model_inventory, remote_model_layout
    from ..node_role import suggest_budget
    from ..staging import free_disk_bytes
    from ..transport import detect_cluster_transports

    return MeasureProbes(
        admission_ceiling=probe_remote_admission_ceiling,
        budget=suggest_budget,
        transports=detect_cluster_transports,
        model_inventory=remote_model_inventory,
        # The peer's own view of a checkpoint it holds, so a model that lives on
        # another Mac can still be measured, planned and staged.
        model_layout=remote_model_layout,
        # Read-only ``df``: staging's own helper, not staging itself.
        disk_free=free_disk_bytes,
        device_record=_paired_device_record,
        model_settings=_saved_model_settings_snapshot,
    )


def _saved_model_settings_snapshot(model_path: str) -> Mapping[str, Any]:
    """The saved per-model levers the resource policy starts from.

    Read on this Mac (the one posting the plan) with oMLX's own settings
    manager. The model id is the checkpoint folder name, which is how oMLX keys
    per-model settings; a manager that has no record answers the documented
    defaults rather than writing anything. This function never raises: an
    unreadable settings file must not stop a measurement.
    """

    if not model_path:
        return {}
    try:
        manager = _get_settings_manager() if _get_settings_manager else None
        if manager is None or not hasattr(manager, "get_settings"):
            return {}
        model_id = Path(model_path).name
        settings = manager.get_settings(model_id)
    except Exception:
        return {}
    return {
        "deepseek_v41_engram_ssd_offload": bool(
            getattr(settings, "deepseek_v41_engram_ssd_offload", False)
        ),
        "qwen4_ple_ssd_offload": bool(
            getattr(settings, "qwen4_ple_ssd_offload", False)
        ),
        "moe_expert_offload_enabled": bool(
            getattr(settings, "moe_expert_offload_enabled", False)
        ),
        "moe_expert_offload_resident_fraction": float(
            getattr(settings, "moe_expert_offload_resident_fraction", 0.25) or 0.25
        ),
        "max_context_window": getattr(settings, "max_context_window", None),
        "mtp_enabled": bool(getattr(settings, "mtp_enabled", False)),
    }


def _saved_memory_guard_tier() -> str | None:
    """This Mac's saved process memory tier, or ``None`` when unreadable."""

    try:
        from omlx.settings import get_settings

        tier = getattr(get_settings().memory, "memory_guard_tier", None)
    except Exception:
        return None
    return str(tier) if isinstance(tier, str) and tier.strip() else None


def _paired_device_record(node_id: str) -> Mapping[str, Any] | None:
    """The paired record discovery/pairing kept for this Mac, if any.

    Its ``last_addrs`` are the private addresses a peer reached this Mac on and
    its ``http_port`` is the port that Mac serves on — the two numbers the plan
    needs and must not invent.
    """

    try:
        from ..registry import get_device_registry

        return get_device_registry().get(node_id)
    except (RuntimeError, ValueError):
        return None


def _measurement_store() -> MeasurementStore:
    """The process-wide measurement store, resolved once."""

    with _lock:
        if _store is not None:
            return _store
        manager = _get_settings_manager() if _get_settings_manager else None
        base = getattr(manager, "base_path", None)
        store = MeasurementStore(Path(base) if base else Path.home() / ".omlx")
        configure_assistant_measurements(store)
        return store


def _measure_probes() -> MeasureProbes:
    with _lock:
        if _probes is not None:
            return _probes
        probes = default_measure_probes()
        configure_assistant_measurements(probes=probes)
        return probes


def _performance_store() -> AssistantPerformanceStore:
    """The process-wide compute-profile store, resolved once."""

    global _performance
    with _lock:
        if _performance is not None:
            return _performance
        manager = _get_settings_manager() if _get_settings_manager else None
        base = getattr(manager, "base_path", None)
        store = AssistantPerformanceStore(Path(base) if base else Path.home() / ".omlx")
        configure_assistant_performance(store)
        return store


def _server_settings() -> Any | None:
    """This Mac's ``ServerSettings``, or ``None`` when it cannot be read.

    Only the measurement job calls this. The plan deliberately does not, so a
    plan cannot depend on the settings of whichever Mac happens to answer it.
    """

    getter = _get_server_settings
    if getter is None:
        try:
            from omlx.settings import get_settings

            return get_settings().server
        except Exception:
            return None
    try:
        return getter()
    except Exception:
        return None


@router.post("/measure", response_model=AssistantMeasureResponse)
async def cluster_assistant_measure(request: AssistantMeasureRequest) -> Any:
    """Start measuring these Macs. Returns a handle, never a result."""

    nodes = _validated_nodes(request.nodes)
    try:
        identity = request.model_identity()
    except PlanningError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_measure_request", "message": str(exc)},
        ) from exc

    store = _measurement_store()
    running = store.active()
    if running is not None:
        return JSONResponse(
            status_code=MEASUREMENT_IN_PROGRESS_STATUS,
            content={
                **plan_error_payload(
                    "measurement_in_progress",
                    f"measurement {running.measurement_id} is still running; "
                    "wait for it instead of starting a second one",
                    ACTION_STATUS,
                ),
                "measurement_id": running.measurement_id,
            },
        )

    record = store.start(model=identity, node_ids=[node.node_id for node in nodes])
    _start_job(store, _measure_probes(), record.measurement_id, request)
    return AssistantMeasureResponse(
        measurement_id=record.measurement_id,
        state="running",
        poll=f"{MEASURE_STATUS_PATH}?measurement_id={record.measurement_id}",
    )


def _start_job(
    store: MeasurementStore,
    probes: MeasureProbes,
    measurement_id: str,
    request: AssistantMeasureRequest,
) -> None:
    """Run the probes on a worker thread; the request returns immediately."""

    task = asyncio.create_task(
        asyncio.to_thread(_measure_job, store, probes, measurement_id, request)
    )
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


def _measure_job(
    store: MeasurementStore,
    probes: MeasureProbes,
    measurement_id: str,
    request: AssistantMeasureRequest,
) -> None:
    """The whole measurement, off the event loop, ending in ready or failed."""

    try:
        (
            measurement,
            endpoints,
            inventories,
            model_settings,
            fabric_links,
            fabric_measured,
        ) = _run_probes(probes, request)
    except MeasureJobError as exc:
        _fail_job(store, measurement_id, exc.code, exc.message, exc.action)
        return
    except Exception as exc:  # a job may never stay "running"
        _fail_job(
            store,
            measurement_id,
            "measurement_failed",
            _redacted(exc),
            ACTION_MEASURE,
        )
        return

    if not measurement.nodes:
        _fail_job(
            store,
            measurement_id,
            "measurement_refused",
            "no Mac could be measured: "
            + ("; ".join(measurement.warning_messages()) or "no facts were produced"),
            ACTION_MEASURE,
        )
        return

    try:
        layout = None
        layout_warnings: list[MeasurementWarning] = []
        if request.model_path or request.model_size_bytes is not None:
            try:
                layout = model_layout_for(request.model_identity()).to_dict()
            except Exception:  # not on this Mac: ask the Mac that holds it
                layout = _layout_from_peer(probes, request, layout_warnings)
        if layout_warnings:
            measurement = replace(
                measurement,
                warnings=measurement.warnings + tuple(layout_warnings),
            )
        store.finish(
            measurement_id,
            measurement=measurement,
            endpoints=endpoints,
            layout=layout,
            inventories=inventories,
            model_settings=model_settings,
            fabric_links=fabric_links,
            fabric_measured=fabric_measured,
        )
    except KeyError:
        return
    except Exception as exc:  # the result could not be stored
        _fail_job(
            store,
            measurement_id,
            "measurement_not_stored",
            _redacted(exc),
            ACTION_MEASURE,
        )


def _fail_job(
    store: MeasurementStore,
    measurement_id: str,
    code: str,
    message: str,
    action: str,
) -> None:
    try:
        store.fail(measurement_id, code=code, message=message, action=action)
    except KeyError:
        # The record was pruned; there is nothing left to tell the caller,
        # and the job thread must not raise.
        return
    except OSError as exc:
        # The store is unwritable. The failed state stays in memory — the
        # store keeps terminal states even when the save fails — so the
        # status poll still reports it with its one action. Log the
        # persistence failure instead of hiding it (review B6); the job
        # thread still must not raise.
        logger.warning(
            "could not persist the failed state of measurement %s: %s",
            measurement_id,
            exc,
        )


def _validated_nodes(
    nodes: Sequence[AssistantMeasureNode],
) -> list[AssistantMeasureNode]:
    seen: set[str] = set()
    validated: list[AssistantMeasureNode] = []
    for node in nodes:
        try:
            target = validate_ssh_target(node.ssh.strip())
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={"code": "invalid_measure_request", "message": str(exc)},
            ) from exc
        node_id = node.node_id.strip()
        if node_id in seen:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "invalid_measure_request",
                    "message": f"node ID {node_id!r} appears more than once",
                },
            )
        seen.add(node_id)
        validated.append(node.model_copy(update={"node_id": node_id, "ssh": target}))
    return validated


def _run_probes(
    probes: MeasureProbes, request: AssistantMeasureRequest
) -> tuple[
    MeasurementSet,
    dict[str, MeasurementEndpoint],
    dict[str, tuple[dict[str, Any], ...]],
    dict[str, Any],
    tuple[tuple[str, str], ...],
    bool,
]:
    """Probe every Mac and turn the answers into one measured set.

    Budgets are the measurement's backbone: if one Mac's ceiling cannot be
    read, the job fails, because a plan built without it would be a guess.
    Links, model availability, disk space and compute degrade to warnings — a
    missing measurement stays missing (``measurement.py``) instead of becoming
    a zero.

    The third return value is each Mac's own model list, with sizes: the
    election only needs the paths, but the free-space affordance has to show a
    person what it may delete on the Mac that is short.
    """

    settings = _server_settings()
    performance = _measure_performance(probes, request)
    stored_performance = _stored_performance_profiles(request)
    links, link_warnings, fabric_links, fabric_measured = _measure_links(
        probes, request
    )
    warnings: list[MeasurementWarning] = list(link_warnings)
    model_settings = _measure_saved_settings(probes, request, warnings)
    inventories: dict[str, list[dict[str, Any]] | None] = {}
    payloads: list[dict[str, Any]] = []
    endpoints: dict[str, MeasurementEndpoint] = {}
    measured_models: dict[str, tuple[dict[str, Any], ...]] = {}

    for node in request.nodes:
        node_id = node.node_id.strip()
        ssh = node.ssh.strip()
        local = ssh in _LOCAL_SSH_TARGETS

        capacity = 0
        capacity_source: str | None = None
        if not local:
            try:
                capacity = int(
                    probes.admission_ceiling(
                        ssh, python_executable=node.python_executable
                    )
                )
            except Exception as exc:  # one Mac, one reason
                raise MeasureJobError(
                    "admission_probe_failed",
                    f"could not read {node_id}'s memory ceiling: {_redacted(exc)}",
                ) from exc
            capacity_source = "admission_ceiling"

        role = request.roles.get(node_id, "headless")
        try:
            budget = probes.budget(
                role=role,
                ssh_target=ssh,
                capacity_bytes=capacity,
                capacity_source=capacity_source,
            )
        except Exception as exc:  # same rule for the budget
            raise MeasureJobError(
                "budget_probe_failed",
                f"could not compute {node_id}'s budget: {_redacted(exc)}",
            ) from exc

        # A workstation reserve is at least 32 GiB. On a Mac whose ceiling is
        # close to that, honouring it leaves ~1 GiB of admission and no model
        # fits — the live run's "none of the 7 candidate models fits" dead end.
        # The check is on the *budget*, so it covers this Mac as well as a peer.
        if role == WORKSTATION_ROLE:
            usable = budget.capacity_bytes - budget.reserve_bytes
            if budget.capacity_bytes and usable < WORKSTATION_MIN_USABLE_BYTES:
                role = DEFAULT_NODE_ROLE
                warnings.append(
                    MeasurementWarning(
                        code=WARNING_WORKSTATION_TOO_LARGE,
                        node_id=node_id,
                        message=(
                            "this Mac is too small for a workstation reserve, so "
                            "the cluster uses it as Headless and keeps its memory "
                            "available"
                        ),
                    )
                )
                try:
                    budget = probes.budget(
                        role=role,
                        ssh_target=ssh,
                        capacity_bytes=capacity,
                        capacity_source=capacity_source,
                    )
                except Exception as exc:
                    raise MeasureJobError(
                        "budget_probe_failed",
                        f"could not compute {node_id}'s budget: {_redacted(exc)}",
                    ) from exc

        inventory = _model_inventory(probes, node, request, inventories, warnings)
        present = (
            True
            if inventory is None or not request.model_path
            else _model_on_node(inventory, request.model_path)
        )
        measured_models[node_id] = _model_entries(inventory)
        disk = _disk_free(probes, node, request, warnings)
        device = _device_record(probes, node_id)

        addresses = _addresses_of(device)
        host = reachable_endpoint_host(
            *addresses, getattr(settings, "host", None) if local else None
        )
        if local and not host:
            # This Mac has no paired record of itself and its own configured
            # host is not dialable (``127.0.0.1`` / ``0.0.0.0``), so take the
            # addresses it would announce to a peer instead of reporting that
            # the coordinator owns no endpoint.
            host = reachable_endpoint_host(*_local_endpoint_addresses())
        port = measured_endpoint_port(
            # The owning Mac's own settings win; a paired record is the only
            # evidence for a Mac this process does not own.
            configured_port=None if local else (device or {}).get("http_port"),
            settings=settings if local else None,
        )
        endpoints[node_id] = MeasurementEndpoint(host=host, port=port)

        profile = next(
            (item for item in performance if _profile_node_id(item) == node_id),
            None,
        ) or stored_performance.get(node_id)
        payload = {
            "node_id": node_id,
            "ssh": ssh,
            **budget.to_dict(),
            "endpoint_host": host,
            "endpoint_port": port,
            "disk_free_bytes": disk,
            "model_present": present,
            # What this Mac's own catalogue listed. Presence for *any*
            # recommended model can then be answered from the measurement
            # instead of assuming a copy is needed for everything but the
            # checkpoint that happened to be measured.
            "model_paths": []
            if inventory is None
            else sorted(
                {
                    str(item.get("model_path"))
                    for item in inventory
                    if isinstance(item.get("model_path"), str)
                    and item.get("model_path")
                }
            ),
            "healthy": True,
            "transports": [{"kind": kind} for kind in links.get(node_id, ())],
            "memory_guard_tier": (
                _saved_memory_guard_tier() if local else "balanced"
            ),
            "wired_limit_mb": _local_wired_limit_mb() if local else None,
            "python_executable": (node.python_executable or "").strip(),
            "performance": None if profile is None else _profile_dict(profile),
            "link": (
                None
                if links.get("__slowest__") is None
                else _link_dict(links["__slowest__"])
            ),
        }
        # Read the endpoint back out of the measured node rather than from the
        # locals: the record must hold exactly what the measurement carried.
        measured = measured_node_from_budget(payload)
        endpoints[node_id] = MeasurementEndpoint(
            host=measured.endpoint_host,
            port=measured.endpoint_port or DEFAULT_ENDPOINT_PORT,
        )
        payloads.append(payload)

    # Real-model calibration (decision 2026-11-13): every Mac that holds the
    # calibration model is measured with real completions, and its profile
    # outranks the synthetic probe. Runs after the probes, never fails the job:
    # a Mac that cannot calibrate keeps the probe (or unknown compute).
    calibrations = _measure_calibrations(probes, request, endpoints, inventories)
    for payload in payloads:
        profile = calibrations.get(str(payload.get("node_id") or ""))
        payload["calibration"] = None if profile is None else profile.to_dict()

    measurement = measure_budget_payloads(payloads, measured_at=measurement_timestamp())
    if warnings:
        measurement = replace(
            measurement, warnings=measurement.warnings + tuple(warnings)
        )
    return (
        measurement,
        endpoints,
        measured_models,
        model_settings,
        fabric_links,
        fabric_measured,
    )


def _measure_saved_settings(
    probes: MeasureProbes,
    request: AssistantMeasureRequest,
    warnings: list[MeasurementWarning],
) -> dict[str, Any]:
    """The saved per-model levers, or none. Never guesses a value."""

    if probes.model_settings is None or not request.model_path:
        return {}
    try:
        saved = probes.model_settings(request.model_path)
    except Exception as exc:  # settings are evidence, not a gate
        warnings.append(
            MeasurementWarning(
                code=WARNING_MODEL_SETTINGS_UNREADABLE,
                message=(
                    "the saved per-model settings could not be read, so the plan "
                    f"starts from the documented defaults: {_redacted(exc)}"
                ),
            )
        )
        return {}
    return dict(saved) if isinstance(saved, Mapping) else {}


def _local_wired_limit_mb() -> int | None:
    """This Mac's ``iogpu.wired_limit_mb``, or ``None`` when unreadable.

    A kernel value of 0 means "Apple's default", not "no limit"; the preflight
    only refuses a known limit, so 0 is reported as unknown.
    """

    try:
        from ...process_memory_enforcer import get_iogpu_wired_limit_bytes

        value = int(get_iogpu_wired_limit_bytes())
    except Exception:
        return None
    if value <= 0:
        return None
    return value // (1024 * 1024)


def _model_entries(
    inventory: Sequence[Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], ...]:
    """The model list one Mac reported, in the shape the card may show.

    ``None`` (unreadable inventory) and ``[]`` (nothing listed) both become "no
    entries": the card then offers no delete for that Mac rather than an empty
    claim. A size of 0 means that Mac reported none, never "zero bytes".
    """

    if not inventory:
        return ()
    entries: dict[str, dict[str, Any]] = {}
    for item in inventory:
        model_path = str(item.get("model_path") or "").strip()
        model_id = str(item.get("id") or item.get("model_id") or "").strip()
        if not model_path or not model_id:
            continue
        size = item.get("estimated_size")
        known_size = (
            int(size)
            if isinstance(size, int) and not isinstance(size, bool) and size > 0
            else 0
        )
        entries[model_path] = {
            "model_id": model_id,
            "model_path": model_path,
            "size_bytes": known_size,
        }
    return tuple(entries[path] for path in sorted(entries))


def _measure_performance(
    probes: MeasureProbes, request: AssistantMeasureRequest
) -> list[Any]:
    """Measured compute profiles, or none. Never an estimated rate."""

    if not request.include_performance_probe:
        return []
    if probes.performance is None:
        raise MeasureJobError(
            "performance_probe_unavailable",
            "this server has no performance probe wired, so measured compute "
            "cannot be requested; measure again without it",
        )
    try:
        return [item for item in (probes.performance(request.nodes) or ())]
    except Exception as exc:  # the probe is one measurement
        raise MeasureJobError(
            "performance_probe_failed",
            f"the performance probe failed: {_redacted(exc)}",
        ) from exc


def _stored_performance_profiles(
    request: AssistantMeasureRequest,
) -> dict[str, NodePerformanceProfile]:
    """Measured compute from an earlier measurement, for the Macs it covered.

    The synthetic probe runs a plan's Macs once and its profiles are stored on
    this Mac; the next measurement reads them. A stored profile is still a measurement
    (with its own ``measured_at``), never an estimate. A malformed or mismatched
    record is skipped, so that node keeps ``unmeasured_compute``.
    """

    try:
        stored = _performance_store().for_nodes(
            [node.node_id.strip() for node in request.nodes]
        )
    except Exception:  # a broken scratch store must not fail a measurement
        return {}
    profiles: dict[str, NodePerformanceProfile] = {}
    for node_id, item in stored.items():
        try:
            profile = NodePerformanceProfile.from_dict(dict(item.profile))
        except (ValueError, TypeError):
            continue
        if profile.node_id == node_id:
            profiles[node_id] = profile
    return profiles


def _measure_calibrations(
    probes: MeasureProbes,
    request: AssistantMeasureRequest,
    endpoints: dict[str, MeasurementEndpoint],
    inventories: dict[str, list[dict[str, Any]] | None],
) -> dict[str, ComputeProfile]:
    """Real-model calibration for every Mac that can run it, or nothing.

    Decision 2026-11-13: ranking rests on REAL completions. Each Mac that
    lists the calibration model runs it through the same completion seam the
    post-``ready`` observe measurement uses, and the resulting profile
    outranks the synthetic probe (``measurement._facts_for``). A Mac without
    the model, an unreadable layout or a failed completion keeps the probe —
    or unknown compute. Calibration never fails the job and never touches
    admission: it measures compute only.
    """

    profiles: dict[str, ComputeProfile] = {}
    for node in request.nodes:
        node_id = node.node_id.strip()
        endpoint = endpoints.get(node_id)
        if endpoint is None:
            continue
        profile = _calibration_for_node(
            probes,
            node,
            inventory=inventories.get(node.ssh.strip()),
            port=endpoint.port,
        )
        if profile is not None:
            profiles[node_id] = profile
    return profiles


def _calibration_for_node(
    probes: MeasureProbes,
    node: AssistantMeasureNode,
    *,
    inventory: list[dict[str, Any]] | None,
    port: int,
) -> ComputeProfile | None:
    """One Mac's real calibration profile, or ``None`` (the probe stays).

    Never raises: every missing piece — no catalogue entry, no layout, no
    answer — is unknown compute, which the measurement already knows how to
    carry. The two completions (warmup discarded, measured second) reuse the
    observe seam's transport and timeout; nothing here stages or launches.
    """

    ssh = node.ssh.strip()
    local = ssh in _LOCAL_SSH_TARGETS
    try:
        if inventory is None:
            # A hardware-only measurement reads no inventory; calibration
            # still needs the Mac's own catalogue to name the serving id.
            # One bounded read, and it never changes the payload's inventory
            # semantics (``model_paths`` stays "not measured").
            inventory = probes.model_inventory(ssh, **_inventory_call(node))
        entry = _calibration_entry(inventory)
        if entry is None:
            return None
        model_id = str(entry.get("id") or entry.get("model_id") or "").strip()
        model_path = str(entry.get("model_path") or "").strip()
        if not model_id or not model_path:
            return None
        layout = _calibration_layout(probes, ssh, model_path, local=local)
        if layout is None:
            return None
        weight_bytes = calibration_weight_bytes(layout)
        if weight_bytes <= 0:
            return None

        def seam(body: Mapping[str, Any]) -> Mapping[str, Any]:
            # This Mac in-process; any other Mac through its own admin seam.
            return stream_completion(
                port=port, body=body, target="" if local else ssh
            )

        # The job runs on a worker thread, so it owns this small event loop;
        # calibrate_node answers None on any failed completion.
        return asyncio.run(
            calibrate_node(
                seam,
                model_weight_bytes=weight_bytes,
                model_id=model_id,
                measured_at=measurement_timestamp(),
            )
        )
    except Exception:  # calibration is evidence, never a gate
        return None


def _calibration_entry(
    inventory: Sequence[Mapping[str, Any]] | None,
) -> Mapping[str, Any] | None:
    """The calibration model in one Mac's catalogue, or ``None``.

    Matched by the checkpoint path first; the folder name is the fallback
    identity because each Mac expands ``~`` to its own home. The folder name
    is how oMLX keys per-model settings, so it is the model's stable name.
    """

    target = _normalized_path(CALIBRATION_MODEL_DIR)
    name = Path(CALIBRATION_MODEL_DIR).name
    fallback: Mapping[str, Any] | None = None
    for item in inventory or ():
        if not isinstance(item, Mapping):
            continue
        path = _normalized_path(str(item.get("model_path") or ""))
        if not path:
            continue
        if path == target:
            return item
        if fallback is None and Path(path).name == name:
            fallback = item
    return fallback


def _calibration_layout(
    probes: MeasureProbes, ssh: str, model_path: str, *, local: bool
) -> ModelLayout | None:
    """The calibration checkpoint's shape, read on the Mac that holds it.

    The weight bytes convert token rates into the planner's units; they come
    from the real checkpoint headers, never from an advertised size. An
    unreadable layout is unknown compute, not a guessed rate.
    """

    try:
        if local:
            return inspect_safetensors_layout(model_path)
        if probes.model_layout is None:
            return None
        return ModelLayout.from_dict(dict(probes.model_layout(ssh, model_path)))
    except Exception:  # an unreadable layout is unknown compute
        return None


def _measure_links(
    probes: MeasureProbes, request: AssistantMeasureRequest
) -> tuple[
    dict[str, Any],
    list[MeasurementWarning],
    tuple[tuple[str, str], ...],
    bool,
]:
    """Transport kinds per Mac and the slowest link inside the group.

    Link evidence is optional: a cluster measured without it is planned on the
    election's nominal ring, and the warning says so. The one thing this must
    not do is attribute a link to a Mac that was not on it.

    The fourth return value is whether the matrix answered with any link at all.
    An empty matrix is "not measured", not "no links": the plan keeps the
    documented nominal-ring fallback instead of refusing every 2-Mac plan.
    """

    from ..link_bandwidth import bandwidth_graph, slowest_link_in
    from ..transport import TransportInfo

    aliases = {
        alias: node.node_id.strip()
        for node in request.nodes
        for alias in (node.node_id.strip(), node.ssh.strip())
    }
    try:
        matrix = probes.transports([node.ssh.strip() for node in request.nodes])
        transports = [
            replace(
                item,
                source_node_id=aliases.get(item.source_node_id, ""),
                peer_node_id=aliases.get(item.peer_node_id, ""),
            )
            for item in (getattr(matrix, "transports", ()) or ())
            if isinstance(item, TransportInfo)
        ]
    except Exception as exc:  # links are hints, not budgets
        return (
            {},
            [
                MeasurementWarning(
                    code=WARNING_LINK_UNMEASURED,
                    message=(
                        "the fabric between these Macs could not be measured, so "
                        f"the plan uses the nominal ring: {_redacted(exc)}"
                    ),
                )
            ],
            (),
            False,
        )

    graph = bandwidth_graph(transports)
    members = [node.node_id.strip() for node in request.nodes]
    slowest = slowest_link_in(graph, members)
    kinds: dict[str, Any] = {}
    for item in transports:
        for node_id in (item.source_node_id, item.peer_node_id):
            if node_id and item.kind not in kinds.setdefault(node_id, []):
                kinds[node_id].append(item.kind)
    kinds["__slowest__"] = slowest
    # ``bandwidth_graph`` is undirected and keyed by the sorted node pair. It is
    # the same evidence the plan uses, so the line check cannot disagree with
    # the link bandwidths.
    pairs = tuple(sorted(graph))
    return kinds, [], pairs, bool(pairs)


def _inventory_call(node: AssistantMeasureNode) -> dict[str, Any]:
    """``model_inventory``'s keyword arguments for one Mac.

    An unknown interpreter is not an interpreter called "None": the probe
    discovers the peer's own when the argument is absent and refuses a
    non-absolute path when it is present.
    """

    executable = (node.python_executable or "").strip()
    return {"python_executable": executable} if executable else {}


def _model_inventory(
    probes: MeasureProbes,
    node: AssistantMeasureNode,
    request: AssistantMeasureRequest,
    cache: dict[str, list[dict[str, Any]] | None],
    warnings: list[MeasurementWarning],
) -> list[dict[str, Any]] | None:
    """The peer's model inventory, or ``None`` when it could not be read.

    ``None`` is not "no models": a Mac whose inventory is unreadable may still
    hold the checkpoint, so ``model_present`` stays true and the warning says
    why.
    """

    ssh = node.ssh.strip()
    if ssh in cache:
        return cache[ssh]
    if not request.model_path:
        cache[ssh] = None
        return None
    try:
        # An unknown interpreter is not an interpreter called "None": the probe
        # discovers the peer's own when the argument is absent, and refuses a
        # non-absolute path when it is present. Passing the caller's empty value
        # through therefore turned every ordinary measurement into
        # ``model_inventory_unreadable``, which reads as "the model is not on
        # that Mac" and makes apply stage a copy that is not needed.
        inventory = probes.model_inventory(ssh, **_inventory_call(node))
    except Exception as exc:  # availability is optional
        warnings.append(
            MeasurementWarning(
                code=WARNING_MODEL_INVENTORY_UNREADABLE,
                node_id=node.node_id,
                message=(
                    "this Mac's model inventory could not be read, so model "
                    f"availability was not measured: {_redacted(exc)}"
                ),
            )
        )
        cache[ssh] = None
        return None
    cache[ssh] = inventory
    return inventory


def _model_on_node(inventory: Sequence[Mapping[str, Any]], model_path: str) -> bool:
    target = _normalized_path(model_path)
    return any(
        _normalized_path(str(item.get("model_path") or "")) == target
        for item in inventory
    )


def _disk_free(
    probes: MeasureProbes,
    node: AssistantMeasureNode,
    request: AssistantMeasureRequest,
    warnings: list[MeasurementWarning],
) -> int | None:
    path = request.model_path or "~"
    try:
        free = int(probes.disk_free(path, ssh_target=node.ssh.strip()))
    except Exception as exc:  # disk is optional evidence
        warnings.append(
            MeasurementWarning(
                code=WARNING_DISK_FREE_UNREADABLE,
                node_id=node.node_id,
                message=(f"free disk space could not be read: {_redacted(exc)}"),
            )
        )
        return None
    if free <= 0:
        # The remote probe returns 0 when df itself failed (for example when
        # the checkpoint is not on that Mac yet). Silence made the
        # recommendation treat a missing measurement as "can stage".
        warnings.append(
            MeasurementWarning(
                code=WARNING_DISK_FREE_UNREADABLE,
                node_id=node.node_id,
                message="free disk space could not be read on this Mac",
            )
        )
        return None
    return free


def _device_record(probes: MeasureProbes, node_id: str) -> Mapping[str, Any] | None:
    try:
        return probes.device_record(node_id)
    except Exception:  # a missing record is not a failure
        return None


def _addresses_of(device: Mapping[str, Any] | None) -> list[str]:
    """The paired addresses for one Mac, ignoring anything malformed."""

    raw = (device or {}).get("last_addrs")
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(item) for item in raw if isinstance(item, str) and item.strip()]


def _dialable(host: str) -> bool:
    """Whether a URL with this host could reach this Mac from another one."""

    text = (host or "").strip()
    if not text:
        return False
    try:
        address = ipaddress.ip_address(text.split("%", 1)[0])
    except ValueError:
        return text not in {"localhost"}
    return not (address.is_loopback or address.is_unspecified)


def _local_endpoint_addresses() -> list[str]:
    """Addresses another Mac could dial this Mac on, most useful first.

    A paired record says which address a *peer* reached this Mac on, but a Mac
    has no paired record of itself. Without this fallback a plan whose
    coordinator is the app origin — the ordinary local case — resolved no
    endpoint at all, so a started cluster could never show a URL.

    Only addresses that can appear in a URL are offered: an IPv4 private
    address first, then any other IPv4. A Thunderbolt self-assigned
    ``169.254.x`` address is still better than nothing when that is all this Mac
    has, and an IPv6 link-local address (``fe80::…%bridge0``) is not offered
    because it is not dialable.
    """

    try:
        from ..discovery import local_addr_dicts

        rows = local_addr_dicts()
    except Exception:  # discovery is optional for a measurement
        return []
    addresses: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        host = str(row.get("ip") or "").strip()
        if not _dialable(host) or ":" in host:
            continue
        addresses.append(host)
    private = [
        host
        for host in addresses
        if ipaddress.ip_address(host).is_private
        and not ipaddress.ip_address(host).is_link_local
    ]
    return private + [host for host in addresses if host not in private]


def _profile_node_id(profile: Any) -> str:
    if isinstance(profile, Mapping):
        return str(profile.get("node_id") or "")
    return str(getattr(profile, "node_id", "") or "")


def _profile_dict(profile: Any) -> dict[str, Any]:
    to_dict = getattr(profile, "to_dict", None)
    return dict(to_dict()) if callable(to_dict) else dict(profile)


def _link_dict(link: Any) -> dict[str, Any]:
    to_dict = getattr(link, "to_dict", None)
    return dict(to_dict()) if callable(to_dict) else dict(link)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@router.get("/status", response_model=AssistantMeasurementStatus)
async def cluster_assistant_measurement_status(
    measurement_id: str = Query(min_length=1, max_length=128),
) -> Any:
    """Report one measurement job. Reads the store; changes nothing."""

    record = _measurement_store().get(measurement_id)
    if record is None:
        return JSONResponse(
            status_code=MEASUREMENT_NOT_FOUND_STATUS,
            content=plan_error_payload(
                "measurement_not_found",
                f"no measurement {measurement_id} is stored; start one",
                ACTION_MEASURE,
            ),
        )
    return measurement_status(record)


# ---------------------------------------------------------------------------
# recommend
# ---------------------------------------------------------------------------

_NO_CANDIDATE_REASON = "no candidate model fits these Macs"


@router.post(
    "/recommend",
    response_model=AssistantRecommendResponse,
    responses={NO_FEASIBLE_CANDIDATE_STATUS: {"model": AssistantRecommendResponse}},
)
async def cluster_assistant_recommend(request: AssistantRecommendRequest) -> Any:
    """Recommend a model for these measured Macs. Synchronous, pure, idempotent.

    Reads the stored hardware facts and the candidates' own headers; it never
    probes, reads this Mac's settings or writes the store. The model identity
    the measurement was taken for is deliberately *not* compared with the
    candidates (the notes): recommending a model is the step that decides which
    one to measure next.
    """

    record = _measurement_store().get(request.measurement_id)
    if record is None:
        return _stale_measurement_response(
            "unknown",
            f"no measurement {request.measurement_id} is stored; start one",
            request.measurement_id,
        )
    # Asking the freshness rule about the record's *own* identity means only
    # state and age can refuse it. That is the whole Option A contract: the
    # hardware was measured, which model it was measured for does not matter
    # here — and ``plan`` still refuses a mismatched model on its own.
    refusal = measurement_refusal(
        record, model=record.model, hardware_only_ok=True
    )
    if refusal is not None:
        return _stale_measurement_response(
            refusal.reason, refusal.message, record.measurement_id
        )

    try:
        result = await asyncio.to_thread(_recommend_from_record, request, record)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_recommend_request", "message": str(exc)},
        ) from exc

    response = build_recommend_response(result, request=request, measurement=record)
    if not result.feasible:
        # Same object as the success path plus the machine-readable
        # discriminators, including the ranked near misses: a client shows the
        # refusal and the next thing worth trying without a second request.
        body = response.model_dump(mode="json")
        body["code"] = NO_FEASIBLE_CANDIDATE_CODE
        body["action"] = result.action
        body["message"] = result.reasons[0] if result.reasons else _NO_CANDIDATE_REASON
        return JSONResponse(status_code=NO_FEASIBLE_CANDIDATE_STATUS, content=body)
    return response


def _recommend_from_record(
    request: AssistantRecommendRequest, record: MeasurementRecord
) -> RecommendationResult:
    """Load the candidates' layouts and rank them against the stored facts.

    Pure apart from ``load_model_candidate`` reading headers, and off the event
    loop because that read can touch many shards. Presence comes from the
    catalogue each Mac reported in the measurement (``NodeFacts.model_paths``),
    so a model that is already on every Mac is not reported as "must be copied"
    just because the measurement covered a different checkpoint.
    """

    nodes = record.measurement_set().nodes

    candidates: list[ModelCandidate] = []
    seen: set[str] = set()
    for item in request.candidates:
        key = _model_key(item.model_path)
        if key in seen:
            raise ValueError(f"model path {item.model_path!r} appears more than once")
        seen.add(key)
        candidates.append(
            load_model_candidate(
                item.model_path,
                friendly_name=item.friendly_name or "",
                declared_context_tokens=item.declared_context_tokens,
            )
        )

    return recommend(
        nodes,
        candidates,
        context_tokens=request.context_tokens,
        execution_profile=request.execution_profile,
        # The Lightning setting the measurement was taken with: the chosen model
        # will be measured again and planned with exactly this one.
        mtp=record.model.mtp,
        saved_settings=record.model_settings,
    )


def _model_key(model_path: str) -> str:
    """One normalized key per checkpoint, so a path cannot be listed twice."""

    return ModelIdentity(model_path=model_path).identity_key


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


@router.post(
    "/plan",
    response_model=AssistantPlanResponse,
    responses={NO_FEASIBLE_PLAN_STATUS: {"model": AssistantPlanResponse}},
)
async def cluster_assistant_plan(request: AssistantPlanRequest) -> Any:
    """Elect rank 0 from a stored measurement. Synchronous, pure, idempotent."""

    try:
        identity = request.model_identity()
    except PlanningError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "planning_error", "message": str(exc)},
        ) from exc

    record = _measurement_store().get(request.measurement_id)
    if record is None:
        return _stale_measurement_response(
            "unknown",
            f"no measurement {request.measurement_id} is stored; start one",
            request.measurement_id,
        )
    refusal = measurement_refusal(record, node_ids=request.node_ids, model=identity)
    if refusal is not None:
        return _stale_measurement_response(
            refusal.reason, refusal.message, record.measurement_id
        )

    try:
        outcome = await asyncio.to_thread(_plan_selection, record, request)
    except UnsupportedParallelismError as exc:
        raise HTTPException(
            status_code=UNSUPPORTED_PARALLELISM_STATUS,
            detail={"code": "unsupported_parallelism", "message": str(exc)},
        ) from exc
    except NoFeasiblePlanError as exc:
        # Same object as the success path plus two machine-readable
        # discriminators; a client cannot mistake "no plan" for a plan it may
        # start, and does not have to read the prose to know which 409 this is.
        payload = build_plan_response(exc.result, request=request, measurement=record)
        body = payload.model_dump(mode="json")
        body["code"] = "no_feasible_plan"
        body["action"] = exc.action
        body["message"] = str(exc)
        return JSONResponse(status_code=NO_FEASIBLE_PLAN_STATUS, content=body)
    except (ElectionError, PlanningError) as exc:
        # The election's input contract: too many Macs, a model mlx-lm cannot
        # split, or a layout that does not fit the planner.
        raise HTTPException(
            status_code=400,
            detail={"code": "planning_error", "message": str(exc)},
        ) from exc
    except ValueError as exc:
        # ``NodeFacts``' own cross-field rules, reached through the measurement.
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_plan_request", "message": str(exc)},
        ) from exc

    if isinstance(outcome, JSONResponse):
        return outcome
    selection = outcome
    alternative = await asyncio.to_thread(
        _speed_alternative, record, request, selection
    )
    return build_plan_response(
        selection.election,
        request=request,
        measurement=record,
        selection=selection,
        alternatives=() if alternative is None else (alternative,),
    )


def _speed_alternative(
    record: MeasurementRecord,
    request: AssistantPlanRequest,
    selection: Any,
) -> Any | None:
    """The best plan that keeps routed experts resident, if it is a real trade.

    The returned plan is quality-first: it may offload experts to keep a larger
    context. Decision 2026-11-14 makes that the user's call, so when the plan does
    offload experts, re-solve with the expert rungs removed and hand back the best
    experts-resident plan (a smaller context) as the alternative. ``None`` when
    the returned plan already keeps experts resident, or no experts-resident plan
    fits — there is then nothing to trade.
    """

    if not selection.policy.moe_expert_offload:
        return None
    try:
        model = _model_layout(record, request)
    except (ElectionError, PlanningError, ValueError):
        return None
    fabric = record.fabric_links if record.fabric_measured else None
    try:
        if request.confirm_two_mac_fallback:
            alternative = _solve_plan(
                record,
                request,
                model,
                fabric_links=fabric,
                node_selection="minimal",
                max_size=2,
                two_mac_fallback=True,
                forced_node_ids=(),
                exclude_expert_offload=True,
            )
        else:
            alternative = _solve_plan(
                record,
                request,
                model,
                fabric_links=fabric,
                node_selection=request.node_selection,
                exclude_expert_offload=True,
            )
    except (NoFeasiblePlanError, ElectionError, PlanningError, ValueError):
        return None
    if alternative.policy.moe_expert_offload:
        return None
    return alternative


def _plan_selection(
    record: MeasurementRecord, request: AssistantPlanRequest
) -> Any:
    """The synchronous solver half of ``plan``, run off the event loop.

    Returns the selection, or the one topology-refusal response. The CPU-bound
    election (and the topology re-solve) must not block other requests; the
    async handler wraps this in ``asyncio.to_thread`` and keeps the error
    contract.
    """

    if request.tensor_parallel_size != 1:
        _reject_tensor_parallel(request.tensor_parallel_size)
    model = _model_layout(record, request)
    fabric = record.fabric_links if record.fabric_measured else None
    if request.confirm_two_mac_fallback:
        try:
            return _solve_plan(
                record,
                request,
                model,
                fabric_links=fabric,
                node_selection="minimal",
                max_size=2,
                two_mac_fallback=True,
                forced_node_ids=(),
            )
        except NoFeasiblePlanError:
            refusal = _topology_refusal(
                record,
                request,
                model,
                fabric=fabric,
                node_selection="minimal",
                max_size=2,
                offer_fallback=False,
            )
            if refusal is not None:
                return refusal
            raise
    try:
        return _solve_plan(
            record,
            request,
            model,
            fabric_links=fabric,
            node_selection=request.node_selection,
        )
    except NoFeasiblePlanError:
        refusal = _topology_refusal(
            record,
            request,
            model,
            fabric=fabric,
            node_selection=request.node_selection,
            max_size=None,
            offer_fallback=True,
        )
        if refusal is not None:
            return refusal
        raise


def _saved_policy_for(
    record: MeasurementRecord, request: AssistantPlanRequest
) -> SavedPolicy:
    """The saved policy plus the explicit choices this plan overrides.

    ``request.table_offload``/``moe_*``/``memory_guard_tier`` are the user's
    visible choices; unset means "keep the saved setting". The solver may still
    propose a more invasive lever if the chosen one does not fit, and the plan
    says so before Start setup.
    """

    saved = SavedPolicy.from_mapping(record.model_settings)
    tier = request.memory_guard_tier or saved.memory_guard_tier
    engram = saved.engram_ssd_offload
    ple = saved.ple_ssd_offload
    if request.table_offload == "engram":
        engram = True
    elif request.table_offload == "ple":
        ple = True
    elif request.table_offload == "none":
        engram = ple = False
    moe = (
        saved.moe_expert_offload_enabled
        if request.moe_expert_offload is None
        else request.moe_expert_offload
    )
    fraction = (
        saved.moe_expert_offload_resident_fraction
        if request.moe_resident_fraction is None
        else request.moe_resident_fraction
    )
    return SavedPolicy(
        memory_guard_tier=tier,
        engram_ssd_offload=engram,
        ple_ssd_offload=ple,
        moe_expert_offload_enabled=moe,
        moe_expert_offload_resident_fraction=fraction,
        max_context_window=saved.max_context_window,
    )


def _solve_plan(
    record: MeasurementRecord,
    request: AssistantPlanRequest,
    model: ModelLayout,
    *,
    fabric_links,
    node_selection: str,
    max_size: int | None = None,
    two_mac_fallback: bool = False,
    forced_node_ids=None,
    exclude_expert_offload: bool = False,
):
    """One solve with every plan-time input, so plan and fallback agree."""

    return solve_cluster_plan(
        record.measurement_set().nodes,
        model,
        target_context_tokens=request.context_tokens,
        rank0_extra_bytes=request.rank0_extra_bytes,
        mtp=request.mtp,
        execution_profile=request.execution_profile,
        microbatch_size=request.microbatch_size or 1,
        saved=_saved_policy_for(record, request),
        saved_baseline=SavedPolicy.from_mapping(record.model_settings),
        node_selection=node_selection,
        allow_optimistic_admission=request.allow_optimistic_admission,
        forced_node_ids=(
            request.node_ids if forced_node_ids is None else forced_node_ids
        ),
        origin_node=request.origin_node,
        fabric_links=fabric_links,
        max_size=max_size,
        two_mac_fallback=two_mac_fallback,
        exclude_expert_offload=exclude_expert_offload,
    )


def _topology_refusal(
    record: MeasurementRecord,
    request: AssistantPlanRequest,
    model: ModelLayout,
    *,
    fabric,
    node_selection: str,
    max_size: int | None,
    offer_fallback: bool,
):
    """Name the missing cable, and offer the 2-Mac plan only with consent.

    The unconstrained solve is the near miss: it is the plan the user would
    have received without the line requirement, so its rank order names the
    exact consecutive pairs that have no measured link. Nothing here runs when
    the fabric was not measured.
    """

    if fabric is None:
        return None
    try:
        unconstrained = _solve_plan(
            record,
            request,
            model,
            fabric_links=None,
            node_selection=node_selection,
            max_size=max_size,
        )
    except NoFeasiblePlanError:
        return None
    missing = missing_line_links(unconstrained.election.rank_order, fabric)
    if not missing:
        return None
    body = build_plan_response(
        unconstrained.election,
        request=request,
        measurement=record,
        selection=unconstrained,
    ).model_dump(mode="json")
    body["code"] = "topology_incomplete"
    body["action"] = ACTION_ADJUST_PLAN
    body["missing_links"] = [[first, second] for first, second in missing]
    body["message"] = (
        "the measured fabric has no line for this rank order: "
        + ", ".join(f"{first} <-> {second}" for first, second in missing)
        + ". Check the Thunderbolt cable, or run with fewer Macs"
    )
    if offer_fallback and len(unconstrained.election.rank_order) > 2:
        try:
            fallback = _solve_plan(
                record,
                request,
                model,
                fabric_links=fabric,
                node_selection="minimal",
                max_size=2,
                two_mac_fallback=True,
                forced_node_ids=(),
            )
        except NoFeasiblePlanError:
            fallback = None
        if fallback is not None:
            fallback_request = request.model_copy(
                update={"confirm_two_mac_fallback": True}
            )
            body["fallback"] = build_plan_response(
                fallback.election,
                request=fallback_request,
                measurement=record,
                selection=fallback,
            ).model_dump(mode="json")
            body["action"] = ACTION_PLAN
            body["message"] += "; run with 2 Macs instead?"
    return JSONResponse(status_code=NO_FEASIBLE_PLAN_STATUS, content=body)


def _stale_measurement_response(
    reason: str, message: str, measurement_id: str
) -> JSONResponse:
    """409 for "measure again", with no plan object at all."""

    return JSONResponse(
        status_code=STALE_MEASUREMENT_STATUS,
        content={
            **plan_error_payload(STALE_MEASUREMENT_CODE, message, ACTION_MEASURE),
            "reason": reason,
            "measurement_id": measurement_id,
        },
    )


def _layout_from_peer(
    probes: MeasureProbes,
    request: AssistantMeasureRequest,
    warnings: list[MeasurementWarning],
) -> dict[str, Any] | None:
    """Read the checkpoint's shape on a peer that reported holding it.

    A model on a bigger Mac is the case this exists for: the origin cannot read
    it, so the plan would refuse a checkpoint that is perfectly usable. The
    layout is recorded with the measurement and every later step re-elects from
    it, so the peers' answers cannot drift apart.
    """

    if not request.model_path or probes.model_layout is None:
        return None
    target = _normalized_path(request.model_path)
    for node in request.nodes:
        ssh = node.ssh.strip()
        try:
            inventory = probes.model_inventory(ssh, **_inventory_call(node))
        except Exception:
            continue
        if not any(
            _normalized_path(str(item.get("model_path") or "")) == target
            for item in inventory
            if isinstance(item, dict)
        ):
            continue
        try:
            layout = probes.model_layout(ssh, request.model_path)
        except Exception:
            continue
        if isinstance(layout, dict) and layout.get("layer_weight_bytes"):
            warnings.append(
                MeasurementWarning(
                    code=WARNING_LAYOUT_FROM_PEER,
                    node_id=node.node_id,
                    message=(
                        "this Mac does not hold the model, so its shape was read "
                        f"on {node.node_id} and every later step plans from that"
                    ),
                )
            )
            return layout
    return None


def _model_layout(record: MeasurementRecord, request: AssistantPlanRequest) -> ModelLayout:
    """The model the plan is built for: the measured layout, or a fresh read.

    The measurement records the checkpoint's shape once, so the plan describes
    what was measured even if a shard file changes underneath it. A record from
    before that field existed still plans from a fresh read of this Mac's copy.
    """

    stored = record.model_layout()
    if stored is not None:
        return stored
    return model_layout_for(request.model_identity())


def _normalized_path(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    return str(Path(text).expanduser())


def _redacted(exc: BaseException) -> str:
    """A short failure message that cannot carry a key or a whole command."""

    text = str(exc).strip() or exc.__class__.__name__
    return text[-300:]


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# free space (the live-run notes: the delete affordance is a relay)
# ---------------------------------------------------------------------------


@router.delete("/models/{node_id}/{model_id:path}")
async def cluster_assistant_delete_model(
    node_id: str = PathParam(min_length=1, max_length=128),
    model_id: str = PathParam(min_length=1, max_length=512),
    measurement_id: str = Query(..., min_length=1, max_length=128),
) -> Any:
    """Delete one model on the measured Mac that is short of space.

    A copy that cannot fit is refused with ``insufficient_disk`` and the one
    action ``free_disk_space``; freeing space is what answers it. This route is
    that step and nothing else: it deletes exactly one model, only on a Mac the
    named measurement covered, only one that Mac itself listed, and never the
    checkpoint that measurement is about to run. The deleting is oMLX's own per
    Mac route, reached over the assistant's admin seam (``start.py``), so the
    browser never touches SSH and a worker — the Mac that actually runs out of
    room — is as reachable as the coordinator.

    ``model_id`` is the id the card showed from the measured inventory; the
    relay passes that model's folder name, which is what ``/admin/api/hf/
    models/{name}`` resolves. A Mac that no longer holds it answers 404, and
    the one action is then ``measure`` — the list is stale, not the model.
    """

    record = _measurement_store().get(measurement_id)
    if record is None or not record.ready:
        return JSONResponse(
            status_code=404,
            content=plan_error_payload(
                "unknown_measurement",
                f"no ready measurement {measurement_id} is stored; measure the "
                "Macs again before deleting a model",
                ACTION_MEASURE,
            ),
        )

    entry = _measured_model(record, node_id, model_id)
    if isinstance(entry, JSONResponse):
        return entry
    if _is_planned_model(record, entry):
        return JSONResponse(
            status_code=409,
            content=plan_error_payload(
                "model_in_use",
                f"{model_id} is the model this measurement plans to run; the "
                "assistant never deletes it. Delete a model the cluster is not "
                "about to start",
                ACTION_ADJUST_PLAN,
            ),
        )
    if _is_last_cluster_copy(record, node_id, entry):
        return JSONResponse(
            status_code=409,
            content=plan_error_payload(
                "last_cluster_copy",
                f"{model_id} is the only complete copy of this checkpoint on the "
                "measured Macs; deleting it would lose the model. Delete a "
                "different model on this Mac, or copy this one to another Mac "
                "first",
                ACTION_FREE_DISK_SPACE,
            ),
        )

    target = _delete_target(record, node_id)
    if isinstance(target, JSONResponse):
        return target
    ssh, port = target

    try:
        await asyncio.to_thread(
            delete_model_on_mac,
            port=port,
            model_name=Path(str(entry.get("model_path") or "")).name,
            target=ssh,
        )
    except RemoteAdminError as exc:
        if exc.status == 404:
            return _refusal_response(
                "unknown_model",
                f"{node_id} does not hold {model_id} any more; measure again to "
                "list what that Mac has",
                ACTION_MEASURE,
            )
        if exc.status in (0, 502):
            return _refusal_response(
                "delete_unreachable",
                f"{node_id} did not answer its own delete route; open and "
                f"configure oMLX on it, then try again ({_redacted(exc)})",
                ACTION_RETRY_COORDINATOR,
                status_code=502,
            )
        return _refusal_response(
            "delete_failed",
            f"{node_id} refused to delete {model_id}: {_redacted(exc)}",
            ACTION_DIAGNOSTICS,
            status_code=502,
        )
    return {"node_id": node_id, "model_id": model_id, "deleted": True}


def _measured_model(
    record: MeasurementRecord, node_id: str, model_id: str
) -> dict[str, Any] | JSONResponse:
    """The measured inventory entry for this Mac and model, or the refusal.

    The inventory is the authority, not the request: the route deletes a model
    that Mac reported, so a wrong node id or a stale list cannot reach a disk.
    """

    if node_id not in set(record.node_ids):
        return _refusal_response(
            "measurement_node_mismatch",
            f"{node_id} is not one of the Macs measurement "
            f"{record.measurement_id} covers; measure again",
            ACTION_MEASURE,
        )
    wanted = _model_key(model_id)
    entry = next(
        (
            item
            for item in record.inventories.get(node_id, ())
            if _model_key(str(item.get("model_id") or "")) == wanted
        ),
        None,
    )
    if entry is None:
        return _refusal_response(
            "unknown_model",
            f"measurement {record.measurement_id} did not list {model_id} on "
            f"{node_id}; measure again before deleting",
            ACTION_MEASURE,
        )
    return entry


def _is_planned_model(
    record: MeasurementRecord, entry: Mapping[str, Any]
) -> bool:
    """Whether this model is the one the measurement is about to run.

    Checked against both the entry's path and its id, because the inventory may
    name the same checkpoint either way. The assistant never deletes the model
    the cluster was just planned to start.
    """

    planned = record.model.model_path
    if not planned:
        return False
    planned_key = _model_key(planned)
    return any(
        _model_key(str(value)) == planned_key
        for value in (
            entry.get("model_path") or "",
            entry.get("model_id") or "",
        )
        if str(value)
    )


def _is_last_cluster_copy(
    record: MeasurementRecord, node_id: str, entry: Mapping[str, Any]
) -> bool:
    """Whether deleting this entry would leave no measured copy of the model.

    The measurement's per-Mac inventories are the evidence. "No other Mac
    listed it" is not the same as "no other Mac has it", so the safe answer is
    to refuse and ask for another model; losing the last copy is data loss.
    """

    key = _model_key(str(entry.get("model_path") or ""))
    if not key:
        return True
    holders = {
        holder
        for holder, entries in record.inventories.items()
        if any(
            _model_key(str(item.get("model_path") or "")) == key
            for item in entries
            if isinstance(item, Mapping)
        )
    }
    return holders <= {node_id}


def _delete_target(
    record: MeasurementRecord, node_id: str
) -> tuple[str, int] | JSONResponse:
    """Where to send the delete: ``(ssh target, admin port)``, empty = this Mac.

    Resolved from the measured address, the measurement's endpoint and the
    paired record — because "which Mac can I reach" must not have two answers. A Mac that was never paired is
    ``pair_mac``; a paired Mac with no address is ``configure_endpoint``; in
    neither case is this Mac substituted for it, and an unknown port is a
    refusal rather than a guessed 8099.
    """

    seams = _provisioning_seams()
    facts = next(
        (node for node in record.measurement_set().nodes if node.node_id == node_id),
        None,
    )
    endpoint = record.endpoint_for(node_id)
    device = seams.device_record(node_id) or {}
    paired = [
        str(item).strip()
        for item in (device.get("last_addrs") or ())
        if isinstance(item, str) and str(item).strip()
    ]
    local_addresses = {
        str(address) for address in seams.local_addresses() if str(address)
    }
    candidates = [
        "" if facts is None else facts.endpoint_host,
        "" if endpoint is None else endpoint.host,
        *paired,
    ]
    local = any(item in local_addresses for item in candidates if item)
    ssh_user = str(device.get("ssh_user") or "").strip()
    paired_host = next((item for item in paired if item), "")
    ssh = f"{ssh_user}@{paired_host}" if paired_host and ssh_user else paired_host
    if not local and not device:
        return _refusal_response(
            "node_not_paired",
            f"{node_id} is not paired with this Mac, so it cannot be reached to "
            "delete a model; pair it and measure again",
            ACTION_PAIR_MAC,
        )
    if not local and not ssh:
        return _refusal_response(
            "node_unresolved",
            f"{node_id} reported no address this Mac can reach; bind oMLX to "
            "the cluster's private interface and measure again",
            ACTION_CONFIGURE_ENDPOINT,
        )
    port = 0 if endpoint is None else int(endpoint.port or 0)
    if not 1 <= port <= 65535:
        return _refusal_response(
            "endpoint_unresolved",
            f"{node_id}'s admin port was not measured; make sure oMLX runs on "
            "that Mac and measure again",
            ACTION_CONFIGURE_ENDPOINT,
        )
    return ("" if local else ssh), port


# ---------------------------------------------------------------------------
# shared apply/start helpers
# ---------------------------------------------------------------------------



def _refusal_response(
    code: str, message: str, action: str, *, status_code: int = 409
) -> JSONResponse:
    """409/400 with a machine-readable code and exactly one next step."""

    return JSONResponse(
        status_code=status_code,
        content=plan_error_payload(code, message, action),
    )
