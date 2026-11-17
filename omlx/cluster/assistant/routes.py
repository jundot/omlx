# SPDX-License-Identifier: Apache-2.0
"""Admin API for the Cluster Assistant: measure, plan, then apply and start.

Twenty-two routes: the browser's assistant surface, plus the two read/write routes
a second origin and the origin itself exchange with the elected coordinator —
the cluster-wide "one apply/start at a time" lock — and the redacted support
bundle::

    POST   /admin/api/cluster/assistant/measure             start a job, return immediately
    GET    /admin/api/cluster/assistant/status              poll that job
    POST   /admin/api/cluster/assistant/recommend           which model should we run?
    POST   /admin/api/cluster/assistant/plan                plan from the stored facts
    POST   /admin/api/cluster/assistant/apply               stage the approved plan
    POST   /admin/api/cluster/assistant/start               start it on the elected Mac
    GET    /admin/api/cluster/assistant/session             last deployment after reload
    POST   /admin/api/cluster/assistant/stop                stop the ready deployment
    GET    /admin/api/cluster/assistant/performance-probe   read the probe switch
    POST   /admin/api/cluster/assistant/performance-probe   write the probe switch
    GET    /admin/api/cluster/assistant/operations/active   is the coordinator busy?
    POST   /admin/api/cluster/assistant/operations/mirror   register/release another origin's operation
    GET    /admin/api/cluster/assistant/operations/{id}     poll that operation
    DELETE /admin/api/cluster/assistant/models/{node}/{id}  free space on a short Mac
    GET    /admin/api/cluster/assistant/diagnostics         redacted support bundle
    GET    /admin/api/cluster/assistant/network-preflight   read-only local network plan

Why the split exists: the probes measure memory ceilings, model inventories and
links over SSH, which takes seconds and fails in ways a plan may not hide. So
``measure`` persists a ``MeasurementSet`` under a ``measurement_id`` and returns
before the first probe finishes, and neither ``recommend`` nor ``plan`` probes at
all — they load the record and decide. A slow probe can therefore never look
like a hung plan, and a plan can never be justified by a number it measured
itself.

``recommend`` and ``plan`` answer different questions (§11.1a). Recommendation
reads only the hardware half of a measurement and may evaluate any candidate
model, because "which model should we run?" is asked before a model is chosen.
``plan`` keeps the strict Chapter 3 match, so the chosen model is measured again
(model-bound) before it is split. Recommendation never weakens that check.

The measurement job does not stage, launch, register a deployment or write the
device registry: it reads budgets, profiles, links, model inventories and disk
space through oMLX's existing probes, and runs the small calibration model on
each Mac that holds it (one warmup plus one measured completion, through the
same seam the post-ready observe measurement uses) so ranking rests on real
compute. Only one job runs at a time; a second ``measure`` is refused with
``409 measurement_in_progress`` rather than joined, because joining would hand
its caller a measurement of a different node set or model.

``apply`` and ``start`` are the only routes that touch the cluster, and they are
jobs for the same reason: staging copies tens of gigabytes and the coordinator
loads weights for minutes. ``apply`` stages and starts nothing; ``start`` stages
nothing and launches only on the elected coordinator. Both re-elect from the
stored measurement and refuse a signature that the election does not reproduce,
so a client-supplied stage table is never obeyed.

Tensor parallelism is refused with an explicit 4xx rather than planned as PP
behind the caller's back (§8.1); a candidate that would need it is reported as
unsupported with one action, never offered. Refusals are distinguishable without
parsing prose: ``stale_measurement`` (no usable measurement — one action:
measure), ``no_feasible_candidate`` (measured cluster, no model fits — choose a
different model), ``no_feasible_plan`` (measured cluster and chosen model, no
legal split — change the context or model), ``plan_mismatch`` (the approved plan
is not the election's — plan again), ``single_node_plan`` (add a Mac),
``node_not_paired`` (pair it), ``operation_not_applied`` (apply first) and
``operation_in_progress`` (poll the running one). Nothing in this module can name
the app origin as a fallback coordinator: the election has no such branch and
these routes add none.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query
from fastapi import Path as PathParam
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

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
from .diagnostics import build_diagnostics
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
    AssistantApplyRequest,
    AssistantDiagnosticsResponse,
    AssistantMeasurementStatus,
    AssistantMeasureNode,
    AssistantMeasureRequest,
    AssistantMeasureResponse,
    AssistantOperationHandle,
    AssistantOperationStatus,
    AssistantPerformanceProbeRequest,
    AssistantPlanRequest,
    AssistantPlanResponse,
    AssistantRecommendRequest,
    AssistantRecommendResponse,
    AssistantStartRequest,
    AssistantStopRequest,
    AssistantStopResponse,
    NetworkPreflightResponse,
    build_plan_response,
    build_recommend_response,
    measurement_status,
    operation_handle,
    operation_status,
    plan_error_payload,
    session_view,
)
from .network_preflight import build_network_preflight
from .operations import (
    ACTION_APPLY,
    ACTION_DIAGNOSTICS,
    ACTION_FREE_DISK_SPACE,
    ACTION_PAIR_MAC,
    ACTION_PLAN,
    ACTION_RETRY_COORDINATOR,
    ACTION_STATUS,
    KIND_APPLY,
    KIND_START,
    OPERATION_IN_PROGRESS_CODE,
    OPERATION_NOT_FOUND_CODE,
    STATE_APPLIED,
    STATE_APPLYING,
    STATE_READY,
    STATE_STARTING,
    OperationInProgressError,
    OperationProgress,
    OperationRecord,
    OperationStore,
    OperationWarning,
)
from .performance_store import AssistantPerformanceStore
from .policy import (
    SavedPolicy,
    solve_cluster_plan,
)
from .provisioning import (
    ApplyPlan,
    PlanInputs,
    ProvisioningError,
    ProvisioningSeams,
    model_layout_for,
    plan_apply,
)
from .recommendation import (
    ModelCandidate,
    RecommendationResult,
    load_model_candidate,
    recommend,
)
from .start import (
    ACTIVE_OPERATION_PATH,
    COORDINATOR_REFUSALS,
    MODEL_UNLOAD_PATH,
    CoordinatorOps,
    RemoteAdminError,
    _local_admin,
    coordinator_ops,
    delete_model_on_mac,
    remote_admin,
    run_apply_job,
    run_start_job,
    start_inputs,
    stream_completion,
)
from .start import (
    coordinator_active_operation as active_operation,
)
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
from .tuning import hardware_key
from .tuning_store import KEEP, TuningPresetStore, merge_preset

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
# One apply/start at a time; the caller polls the running operation instead.
OPERATION_IN_PROGRESS_STATUS = 409
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
_tuning: TuningPresetStore | None = None
_operations: OperationStore | None = None
_seams: ProvisioningSeams | None = None
_coordinator: Callable[[ApplyPlan], CoordinatorOps] | None = None
_get_settings_manager: Callable[[], Any] | None = None
_get_server_settings: Callable[[], Any] | None = None
_get_global_settings: Callable[[], Any] | None = None
_tasks: set[asyncio.Task[Any]] = set()


def set_assistant_getters(
    settings_manager_getter: Callable[[], Any] | None = None,
    server_settings_getter: Callable[[], Any] | None = None,
    global_settings_getter: Callable[[], Any] | None = None,
) -> None:
    """Inject server-owned dependencies without importing ``omlx.server``.

    Called once next to ``set_cluster_getters`` in
    ``omlx/server.py::_register_cluster_routes``: the settings manager supplies
    the base path the measurement store lives under, the server settings
    supply this Mac's own endpoint port and probe flag, and the global settings
    let the performance-probe toggle persist without an edit-JSON step. All are
    read lazily, because the cluster routes are registered before the settings
    manager exists.
    """

    global _get_settings_manager, _get_server_settings, _get_global_settings
    _get_settings_manager = settings_manager_getter
    _get_server_settings = server_settings_getter
    _get_global_settings = global_settings_getter


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


def configure_assistant_tuning(store: TuningPresetStore | None = None) -> None:
    """Point the tuning-preset store at a store, or let it resolve lazily."""

    global _tuning
    with _lock:
        if store is not None:
            _tuning = store


def configure_assistant_operations(
    store: OperationStore | None = None,
    *,
    provisioning: ProvisioningSeams | None = None,
    coordinator: Callable[[ApplyPlan], CoordinatorOps] | None = None,
) -> None:
    """Point apply/start at an operation store, lookup seams and coordinator.

    Production resolves all three lazily (settings manager base path, device
    registry, elected coordinator). Tests pass a temporary store and fakes so no
    test writes to the user's real ``~/.omlx`` or reaches a Mac.
    """

    global _operations, _seams, _coordinator
    with _lock:
        if store is not None:
            _operations = store
        if provisioning is not None:
            _seams = provisioning
        if coordinator is not None:
            _coordinator = coordinator


def _operation_store() -> OperationStore:
    """The process-wide operation store, resolved once."""

    with _lock:
        if _operations is not None:
            return _operations
        manager = _get_settings_manager() if _get_settings_manager else None
        base = getattr(manager, "base_path", None)
        store = OperationStore(Path(base) if base else Path.home() / ".omlx")
        configure_assistant_operations(store)
        return store


def _provisioning_seams() -> ProvisioningSeams:
    global _seams
    with _lock:
        if _seams is None:
            from .provisioning import default_provisioning_seams

            _seams = default_provisioning_seams()
        return _seams


def _coordinator_ops(plan: ApplyPlan) -> CoordinatorOps:
    """How this apply/start reaches the elected coordinator.

    Defaults to ``start.coordinator_ops``: the coordinator's own pipeline in
    this process when it is this Mac, and the coordinator's admin API over the
    existing SSH seam when it is not.
    """

    factory = _coordinator or coordinator_ops
    return factory(plan)


def _schedule(task: Any) -> None:
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


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


def _tuning_store() -> TuningPresetStore:
    """The process-wide tuning-preset store, resolved once."""

    global _tuning
    with _lock:
        if _tuning is not None:
            return _tuning
        manager = _get_settings_manager() if _get_settings_manager else None
        base = getattr(manager, "base_path", None)
        store = TuningPresetStore(Path(base) if base else Path.home() / ".omlx")
        configure_assistant_tuning(store)
        return store


def _global_settings() -> Any | None:
    """This Mac's ``GlobalSettings``, so the probe toggle can persist."""

    getter = _get_global_settings
    if getter is None:
        try:
            from omlx.settings import get_settings

            return get_settings()
        except Exception:
            return None
    try:
        return getter()
    except Exception:
        return None


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


def _performance_probe_enabled() -> bool:
    """Whether this Mac's settings ask for the post-``ready`` measurement.

    Default off. The flag is read on the Mac that posts ``start`` (the origin),
    because that is the Mac whose operation record receives ``observed``; the
    probe itself still runs on the elected coordinator through the existing
    seam. ``auto_tune`` stays false regardless, so the approved placement is the
    launched placement.
    """

    settings = _server_settings()
    return bool(getattr(settings, "assistant_performance_probe", False))


# ---------------------------------------------------------------------------
# measure
# ---------------------------------------------------------------------------


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

    # The calibration just loaded and released a checkpoint on those Macs, and
    # the freed bytes do not go back to the OS immediately (mlx keeps them in the
    # process). The launcher charges the *dynamic* ceiling it reads at load time,
    # so a plan priced from the probe taken before the calibration is refused at
    # start — measured live 2026-11-16: 47.8 GiB needed against 45.4 GiB
    # admitted, four minutes later, on a Mac that showed 55.4 GiB idle.
    # Re-probe now and keep the smaller capacity: the plan must be priced against
    # what the launch will actually see.
    if calibrations:
        _reprice_after_calibration(probes, request, payloads)

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
    """Measured compute from the last apply, for the Macs it covered.

    The synthetic probe runs after apply; its profiles are stored on this Mac
    and used by the next measurement. A stored profile is still a measurement
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


def _reprice_after_calibration(
    probes: MeasureProbes,
    request: AssistantMeasureRequest,
    payloads: list[dict[str, Any]],
) -> None:
    """Lower each calibrated Mac's budget to its post-calibration ceiling.

    Never raises the number: a Mac that shows *more* room after the calibration
    keeps the conservative figure the plan was already built from, so this can
    only make a plan safer. A failed re-probe leaves the payload untouched.
    """

    roles = {node.node_id.strip(): node for node in request.nodes}
    for payload in payloads:
        node_id = str(payload.get("node_id") or "")
        node = roles.get(node_id)
        if node is None:
            continue
        if payload.get("calibration") is None:
            # Only a Mac that actually ran the calibration changed its ceiling.
            continue
        ssh = str(payload.get("ssh") or "").strip()
        local = ssh in _LOCAL_SSH_TARGETS
        try:
            fresh = (
                0
                if local
                else int(
                    probes.admission_ceiling(
                        ssh, python_executable=node.python_executable
                    )
                )
            )
            budget = probes.budget(
                role=str(payload.get("role") or "headless"),
                ssh_target=ssh,
                capacity_bytes=fresh,
                capacity_source=None if local else "admission_ceiling",
            )
        except Exception:  # noqa: BLE001 - a failed re-probe keeps the old number
            continue
        current = int(payload.get("capacity_bytes") or 0)
        candidate = int(getattr(budget, "capacity_bytes", 0) or 0)
        if current > 0 and candidate >= current:
            continue
        payload.update(budget.to_dict())


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
        profile = asyncio.run(
            calibrate_node(
                seam,
                model_weight_bytes=weight_bytes,
                model_id=model_id,
                measured_at=measurement_timestamp(),
            )
        )
        # The warmup loaded this checkpoint into the Mac's pool and nothing
        # else would unload it. Left resident it lowers the admission ceiling
        # the *launcher* sees, so a plan priced from this measurement is then
        # refused — measured live 2026-11-16: a plan admitted at 49.3 GiB was
        # rejected at 41.6 GiB because every Mac still held `dev-1.5b`.
        _release_calibration_model(
            port=port, ssh="" if local else ssh, model_id=model_id
        )
        return profile
    except Exception:  # calibration is evidence, never a gate
        return None


def _release_calibration_model(*, port: int, ssh: str, model_id: str) -> None:
    """Unload the calibration checkpoint this Mac just ran.

    ``calibrate_node`` runs two real completions with the calibration model, so
    oMLX's pool holds it afterwards. Nothing else unloads it, and a resident
    calibration model is exactly the difference between the admission the
    measurement priced and the admission the launcher's guard reports — the
    plan then fails at start with "needs 47.8 GiB but can admit 41.6 GiB".

    Best effort by design: an unload that fails (busy, already gone, unreadable)
    only leaves the model where it was, and the measurement itself is complete.
    ``ssh`` empty means this Mac (in-process seam), otherwise the Mac's own
    admin route over the existing SSH seam.
    """

    if not model_id:
        return
    path = MODEL_UNLOAD_PATH.format(model_id=model_id)
    try:
        if ssh:
            remote_admin(ssh, port, path, {}, method="POST")
        else:
            _local_admin(port, path, {}, method="POST")
    except Exception:  # noqa: BLE001 - cleanup never fails the measurement
        logger.debug("calibration release failed for %s", model_id, exc_info=True)


def _calibration_entry(
    inventory: Sequence[Any] | None,
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
                message=f"free disk space could not be read: {_redacted(exc)}",
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

        rows: list[Any] = local_addr_dicts()
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
    candidates (§11.1a): recommending a model is the step that decides which
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
        # The user's explicit rank-0 pin (and with it the endpoint host). It is
        # a plan-time input, so apply/start send it as well; an unknown Mac is
        # a ValueError and surfaces as a 422 rather than a silent fallback.
        preferred_coordinator=request.preferred_coordinator,
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


@router.post("/apply", response_model=AssistantOperationHandle)
async def cluster_assistant_apply(request: AssistantApplyRequest) -> Any:
    """Stage the approved plan on the elected coordinator. Starts nothing.

    The plan is not persisted (Chapter 3 kept ``plan`` pure), so this route
    re-elects from the stored measurement with the same plan-time inputs and
    refuses a signature the election does not reproduce. Staging runs on the
    coordinator through oMLX's existing staging job; the request returns before
    the first byte is copied.
    """

    try:
        identity = request.model_identity()
    except PlanningError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_apply_request", "message": str(exc)},
        ) from exc

    resolved = _usable_measurement(
        request.measurement_id,
        node_ids=request.rank_order,
        identity=identity,
    )
    if isinstance(resolved, JSONResponse):
        return resolved
    record = resolved

    planned = await _plan_with_auto_free(record, request.plan_inputs())
    plan, reclaim_warnings = planned
    if isinstance(plan, JSONResponse):
        return plan

    busy = await asyncio.to_thread(_busy_on_coordinator, plan)
    if busy is not None:
        return _operation_in_progress_elsewhere(busy)

    store = _operation_store()
    try:
        operation = store.start(
            kind=KIND_APPLY,
            measurement_id=record.measurement_id,
            plan_signature=request.plan_signature,
            coordinator_node=plan.coordinator.node_id,
            coordinator_local=plan.coordinator.local,
            rank_order=plan.rank_order,
            context_tokens=request.context_tokens,
            target_context_tokens=(
                request.target_context_tokens or request.context_tokens
            ),
            model_path=request.model_path or "",
            rank0_extra_bytes=request.rank0_extra_bytes,
            microbatch_size=request.microbatch_size or 1,
            mtp=request.mtp,
            execution_profile=request.execution_profile,
            two_mac_fallback=request.confirm_two_mac_fallback,
            preferred_coordinator=request.preferred_coordinator or "",
            policy=request.resource_policy().to_dict(),
            endpoint=plan.endpoint,
        )
    except OperationInProgressError as exc:
        return _operation_in_progress_response(exc)

    _begin(
        store,
        operation.operation_id,
        STATE_APPLYING,
        "staging",
        plan.coordinator.node_id,
        nodes_total=len(plan.rank_order),
    )
    try:
        ops = _coordinator_ops(plan)
    except Exception as exc:  # a resolved target that cannot be dialled
        store.fail(
            operation.operation_id,
            code="coordinator_unreachable",
            message=_redacted(exc),
            action=ACTION_RETRY_COORDINATOR,
        )
        return operation_handle(store.get(operation.operation_id) or operation)

    _schedule(
        asyncio.create_task(
            asyncio.to_thread(
                run_apply_job,
                store,
                operation.operation_id,
                plan,
                ops,
                performance_store=_performance_store(),
                pre_warnings=reclaim_warnings,
            )
        )
    )
    return operation_handle(store.get(operation.operation_id) or operation)


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------


@router.post("/start", response_model=AssistantOperationHandle)
async def cluster_assistant_start(request: AssistantStartRequest) -> Any:
    """Start the staged deployment on the elected coordinator.

    Only an applied plan may start: the deployment that staging prepared is the
    deployment that launches, and a changed placement is refused rather than
    launched. The request returns a handle; readiness is proven later.
    """

    store = _operation_store()
    applied = _applied_operation(store, request)
    if applied is None:
        return _refusal_response(
            "operation_not_applied",
            "no applied plan matches this measurement, signature and "
            "coordinator; apply it before starting it",
            ACTION_APPLY,
        )

    resolved = _usable_measurement(request.measurement_id, node_ids=None, identity=None)
    if isinstance(resolved, JSONResponse):
        return resolved
    record = resolved

    planned = await asyncio.to_thread(
        _planned_deployment, record, start_inputs(applied, model=record.model)
    )
    if isinstance(planned, JSONResponse):
        return planned
    plan = planned

    busy = await asyncio.to_thread(_busy_on_coordinator, plan)
    if busy is not None:
        return _operation_in_progress_elsewhere(busy)

    try:
        operation = store.start(
            kind=KIND_START,
            measurement_id=record.measurement_id,
            plan_signature=applied.plan_signature,
            coordinator_node=plan.coordinator.node_id,
            coordinator_local=plan.coordinator.local,
            rank_order=plan.rank_order,
            context_tokens=applied.context_tokens,
            target_context_tokens=applied.target_context_tokens,
            model_path=applied.model_path,
            rank0_extra_bytes=applied.rank0_extra_bytes,
            microbatch_size=applied.microbatch_size,
            mtp=applied.mtp,
            execution_profile=applied.execution_profile,
            two_mac_fallback=applied.two_mac_fallback,
            preferred_coordinator=applied.preferred_coordinator,
            policy=dict(applied.policy),
            endpoint=plan.endpoint,
        )
    except OperationInProgressError as exc:
        return _operation_in_progress_response(exc)

    _begin(
        store,
        operation.operation_id,
        STATE_STARTING,
        "starting",
        plan.coordinator.node_id,
        nodes_total=len(plan.rank_order),
    )
    try:
        ops = _coordinator_ops(plan)
    except Exception as exc:
        store.fail(
            operation.operation_id,
            code="coordinator_unreachable",
            message=_redacted(exc),
            action=ACTION_RETRY_COORDINATOR,
        )
        return operation_handle(store.get(operation.operation_id) or operation)

    _schedule(
        asyncio.create_task(
            run_start_job(
                store,
                operation.operation_id,
                plan.request,
                ops,
                expected_placement=applied.placement_signature,
                probe_enabled=_performance_probe_enabled(),
                # The saved tuning preset for this model + node config is
                # written by the job before activation, so its MTP depth is the
                # one the launch uses. Best-effort; never blocks the start.
                preset=_saved_preset_patch(applied),
            )
        )
    )
    return operation_handle(store.get(operation.operation_id) or operation)


# ---------------------------------------------------------------------------
# stop (Screen 6: state + stop/restart, nothing more)
# ---------------------------------------------------------------------------

STOP_DEPLOYMENT_PATH = "/admin/api/cluster/deployments/{deployment_id}/unload"
STOP_TIMEOUT_SECONDS = 300.0
WARNING_DEPLOYMENT_STOPPED = "deployment_stopped"


def _coordinator_admin(
    record: OperationRecord,
    path: str,
    *,
    body: Mapping[str, Any] | None = None,
    method: str = "POST",
    timeout: float = STOP_TIMEOUT_SECONDS,
) -> Mapping[str, Any]:
    """Call one admin route on the operation's coordinator, local or over SSH.

    The coordinator is known from the operation record: no re-election, so a
    stale measurement cannot prevent a stop or a status question. This Mac uses
    its own loopback admin key; another Mac is reached through the existing SSH
    seam, which authenticates locally on that Mac. The key never crosses the
    assistant API.
    """

    endpoint = record.endpoint
    port = 0 if endpoint is None else int(endpoint.port)
    if not 1 <= port <= 65535:
        raise ProvisioningError(
            "configure_endpoint",
            "the coordinator's endpoint port is not known; open oMLX on it and "
            "retry the coordinator",
            ACTION_CONFIGURE_ENDPOINT,
        )
    if record.coordinator_local:
        return _local_admin(port, path, body, method=method, timeout=timeout)
    device = _paired_device_record(record.coordinator_node) or {}
    ssh_user = str(device.get("ssh_user") or "").strip()
    host = "" if endpoint is None else str(endpoint.host or "").strip()
    if not host:
        raise ProvisioningError(
            "configure_endpoint",
            "the coordinator's address is not known; open oMLX on it and retry "
            "the coordinator",
            ACTION_CONFIGURE_ENDPOINT,
        )
    if not device:
        raise ProvisioningError(
            "node_not_paired",
            f"{record.coordinator_node} has no paired SSH target on this Mac; "
            "pair it again and retry the coordinator",
            ACTION_PAIR_MAC,
        )
    target = f"{ssh_user}@{host}" if ssh_user else host
    return remote_admin(
        validate_ssh_target(target),
        port,
        path,
        body,
        method=method,
        timeout=timeout,
    )


def _coordinator_active_operation(record: OperationRecord) -> str:
    """The operation the coordinator is running, or ``""`` when unanswerable.

    An unanswered question is not a lock: the stop then runs and reports the
    real refusal, rather than turning a diagnostic problem into a fake busy.
    """

    try:
        reply: Any = _coordinator_admin(
            record, ACTIVE_OPERATION_PATH, method="GET", timeout=30.0
        )
    except Exception:  # noqa: BLE001 - unanswerable is not "busy"
        return ""
    if not isinstance(reply, Mapping):
        return ""
    operation = reply.get("operation")
    if not isinstance(operation, Mapping):
        return ""
    running = operation.get("running")
    operation_id = operation.get("operation_id")
    if running is False or not isinstance(operation_id, str):
        return ""
    return operation_id


def _stop_on_coordinator(
    record: OperationRecord, deployment_id: str
) -> Mapping[str, Any]:
    """Unload one deployment on the elected coordinator through the same seam."""

    path = STOP_DEPLOYMENT_PATH.format(
        deployment_id=quote(str(deployment_id).strip(), safe="")
    )
    return _coordinator_admin(record, path)


@router.post("/stop", response_model=AssistantStopResponse)
async def cluster_assistant_stop(request: AssistantStopRequest) -> Any:
    """Stop the deployment a ready start operation launched.

    Idempotent: a second stop answers the stored record instead of relaying
    again. ``state`` stays ``ready`` (the frozen state machine has no stopped
    value); ``stopped`` is what the Screen-6 card reads.
    """

    store = _operation_store()
    active = store.active()
    if active is not None:
        # "One apply/start at a time" is cluster-wide. A stop must not unload a
        # deployment while another operation is staging or launching; the user
        # polls the running one instead.
        return _operation_in_progress_response(
            OperationInProgressError(active.operation_id, active.kind)
        )
    record = store.get(request.operation_id)
    if record is None:
        return JSONResponse(
            status_code=404,
            content=plan_error_payload(
                "unknown_operation",
                f"no operation {request.operation_id} is stored; poll the "
                "operation you started",
                ACTION_STATUS,
            ),
        )
    if record.kind != KIND_START or record.state != "ready" or not record.deployment_id:
        return _refusal_response(
            "operation_not_ready",
            "only a start operation that is ready and knows its deployment can "
            "be stopped",
            ACTION_STATUS,
        )
    if record.stopped:
        return AssistantStopResponse(
            operation_id=record.operation_id,
            deployment_id=record.deployment_id,
            stopped=True,
            warnings=operation_status(record).warnings,
        )
    busy = (
        ""
        if record.coordinator_local
        else await asyncio.to_thread(_coordinator_active_operation, record)
    )
    if busy and busy != record.operation_id:
        return _operation_in_progress_elsewhere(busy)
    try:
        reply = await asyncio.to_thread(
            _stop_on_coordinator, record, record.deployment_id
        )
    except ProvisioningError as exc:
        return _refusal_response(exc.code, exc.message, exc.action)
    except RemoteAdminError as exc:
        if exc.status == 404:
            # The coordinator no longer has this deployment: it is already
            # stopped, which is the state the caller asked for.
            reply = {"ok": True, "stopped": True}
        elif exc.status in COORDINATOR_REFUSALS:
            return _refusal_response(
                "coordinator_unreachable",
                f"{record.coordinator_node} did not answer; open oMLX on it and "
                "retry the coordinator",
                ACTION_RETRY_COORDINATOR,
            )
        else:
            return _refusal_response(
                "stop_failed",
                f"the coordinator refused to stop {record.deployment_id}: "
                f"{exc.detail}",
                ACTION_DIAGNOSTICS,
            )
    except OSError as exc:  # a refused loopback port is "no answer"
        return _refusal_response(
            "coordinator_unreachable",
            f"{record.coordinator_node} did not answer: {_redacted(exc)}; open "
            "oMLX on it and retry the coordinator",
            ACTION_RETRY_COORDINATOR,
        )
    if not isinstance(reply, Mapping) or reply.get("stopped") is not True:
        return _refusal_response(
            "stop_failed",
            f"the coordinator did not confirm that {record.deployment_id} "
            "stopped",
            ACTION_DIAGNOSTICS,
        )
    updated = store.mark_stopped(
        record.operation_id,
        warning=OperationWarning(
            code=WARNING_DEPLOYMENT_STOPPED,
            node_id=record.deployment_id,
            message=(
                f"the deployment on {record.coordinator_node} was stopped; the "
                "cluster is not serving"
            ),
        ),
    )
    return AssistantStopResponse(
        operation_id=updated.operation_id,
        deployment_id=updated.deployment_id,
        stopped=updated.stopped,
        warnings=operation_status(updated).warnings,
    )


# ---------------------------------------------------------------------------
# search (the automatic constellation optimization loop)
# ---------------------------------------------------------------------------


class AssistantSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    measurement_id: str = Field(min_length=1, max_length=128)
    context_tokens: int = Field(default=32768, ge=1)
    priority: Literal["balanced", "speed", "ttft", "quality", "context"] = "balanced"
    max_count: int = Field(default=3, ge=1, le=4)
    mtp: bool = False
    execution_profile: Literal["interactive", "balanced", "throughput"] = "balanced"
    rank0_extra_bytes: int = Field(default=0, ge=0)
    microbatch_size: int | None = Field(default=None, ge=1)


def _constellation_view(row: Any) -> dict[str, Any]:
    return {
        "rank_order": list(row.rank_order),
        "model_path": row.model_path,
        "context_tokens": row.context_tokens,
        "plan_signature": row.plan_signature,
        "node_count": row.node_count,
        "feasible": row.feasible,
        "measured": row.measured,
        "decode_tok_per_s": row.decode_tok_per_s,
        "prefill_tok_per_s": row.prefill_tok_per_s,
        "ttft_ms": row.ttft_ms,
        "deployment_id": row.deployment_id,
        "coordinator_node": row.coordinator_node,
    }


@router.post("/search")
async def cluster_assistant_search(request: AssistantSearchRequest) -> Any:
    """Measure up to ``max_count`` diverse constellations, keep the best measured.

    The loop ranks beforehand from the measurement (hardware estimate), then oMLX
    plans + starts each constellation, measures it for real, picks the best
    measured and stops the losers. oMLX's planner owns every layer split; this
    only chooses rank orders / subsets and measures. One apply/start at a time
    (the cluster-wide lock); a failed constellation is recorded, never fatal.
    """

    from .policy import diverse_candidates
    from .search_wiring import evaluate_constellation

    record = _measurement_store().get(request.measurement_id)
    if record is None:
        return _stale_measurement_response(
            "unknown",
            f"no measurement {request.measurement_id} is stored; start one",
            request.measurement_id,
        )
    refusal = measurement_refusal(record, model=record.model)
    if refusal is not None:
        return _stale_measurement_response(
            refusal.reason, refusal.message, record.measurement_id
        )
    model = record.model_layout()
    if model is None:
        return _refusal_response(
            "no_model_layout",
            "the measurement has no model layout; measure the chosen model first",
            ACTION_MEASURE,
        )
    candidates = await asyncio.to_thread(
        diverse_candidates,
        record.measurement_set().nodes,
        model,
        target_context_tokens=request.context_tokens,
        rank0_extra_bytes=request.rank0_extra_bytes,
        mtp=request.mtp,
        execution_profile=request.execution_profile,
        microbatch_size=request.microbatch_size or 1,
        fabric_links=record.fabric_links if record.fabric_measured else None,
        max_count=request.max_count,
    )
    if not candidates:
        return _refusal_response(
            "no_feasible_plan",
            "no constellation fits this model on the measured Macs; choose a "
            "smaller model, less context, or add a Mac",
            ACTION_ADJUST_PLAN,
        )

    store = _operation_store()
    seams = _provisioning_seams()
    performance_store = _performance_store()
    probe_enabled = _performance_probe_enabled()

    async def evaluate(candidate: Any) -> Any:
        return await evaluate_constellation(
            candidate,
            record=record,
            store=store,
            seams=seams,
            ops_for=_coordinator_ops,
            mtp=request.mtp,
            execution_profile=request.execution_profile,
            rank0_extra_bytes=request.rank0_extra_bytes,
            microbatch_size=request.microbatch_size,
            performance_store=performance_store,
            probe_enabled=probe_enabled,
        )

    async def stop(result: Any) -> None:
        op = store.get(result.operation_id)
        if op is not None and result.deployment_id:
            await asyncio.to_thread(_stop_on_coordinator, op, result.deployment_id)
            store.mark_stopped(op.operation_id)

    job_id = _create_search_job(total=len(candidates))
    _schedule(
        asyncio.create_task(
            _run_search_job(job_id, candidates, evaluate, stop, request)
        )
    )
    return {
        "job_id": job_id,
        "state": "queued",
        "total": len(candidates),
        "poll": f"/admin/api/cluster/assistant/search/{job_id}",
    }


# The search loop runs for minutes (it deploys + measures several constellations),
# so it is a background job, not a blocking request: POST /search returns a handle
# immediately and GET /search/{job_id} polls progress + the final outcome. Jobs
# are ephemeral (in-memory): a server restart loses a running search, which the
# caller restarts; the deployments it created remain in the operation store.
_search_jobs: dict[str, dict[str, Any]] = {}
_search_jobs_lock = threading.Lock()


def _create_search_job(*, total: int) -> str:
    job_id = f"search-{uuid.uuid4().hex[:12]}"
    with _search_jobs_lock:
        _search_jobs[job_id] = {
            "job_id": job_id,
            "state": "queued",
            "total": total,
            "done": 0,
            "message": "queued",
            "results": [],
            "best": None,
            "tested": [],
            "stopped": [],
            "errors": [],
        }
    return job_id


def _update_search_job(job_id: str, **fields: Any) -> None:
    with _search_jobs_lock:
        job = _search_jobs.get(job_id)
        if job is not None:
            job.update(fields)


async def _run_search_job(
    job_id: str,
    candidates: list,
    evaluate: Callable,
    stop: Callable,
    request: AssistantSearchRequest,
) -> None:
    from .search import run_search

    _update_search_job(job_id, state="running", message="measuring constellations")
    done = 0
    partial: list = []

    async def progress_evaluate(candidate: Any) -> Any:
        nonlocal done
        _update_search_job(
            job_id,
            message=f"measuring constellation {done + 1} of {len(candidates)}",
        )
        result = await evaluate(candidate)
        done += 1
        partial.append(result)
        _update_search_job(
            job_id,
            done=done,
            results=[_constellation_view(row) for row in partial],
        )
        return result

    try:
        outcome = await run_search(
            candidates,
            evaluate=progress_evaluate,
            stop=stop,
            max_count=request.max_count,
            priority=request.priority,
        )
        _update_search_job(
            job_id,
            state="completed",
            message="done",
            best=_constellation_view(outcome.best) if outcome.best else None,
            results=[_constellation_view(row) for row in outcome.results],
            tested=[list(candidate.rank_order) for candidate in outcome.tested],
            stopped=[_constellation_view(row) for row in outcome.stopped],
            errors=list(outcome.errors),
        )
    except Exception as exc:  # noqa: BLE001 - a failed search is a failed job
        _update_search_job(job_id, state="failed", message=str(exc)[:300])


@router.get("/search/{job_id}")
async def cluster_assistant_search_status(job_id: str = PathParam(min_length=1, max_length=128)) -> Any:
    """Poll a running search job: state, progress, and the final outcome."""

    with _search_jobs_lock:
        job = _search_jobs.get(job_id)
        snapshot = dict(job) if job is not None else None
    if snapshot is None:
        return _refusal_response(
            "search_not_found",
            f"no search job {job_id}; start one with POST /search",
            ACTION_STATUS,
        )
    return snapshot


# ---------------------------------------------------------------------------
# tune (fine-tune the winning constellation with real completions)
# ---------------------------------------------------------------------------


class AssistantTuneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation_id: str = Field(min_length=1, max_length=128)


def _tune_mtp_capable(model_path: str) -> bool:
    """Whether this model has an MTP head (so the MTP A/B is meaningful).

    A model without an MTP head cannot draft, so the tune skips the A/B instead
    of measuring two identical no-ops. Unreadable → False (not proven capable).
    """

    if not model_path:
        return False
    try:
        from ..model_inventory import mtp_compatible_for_model

        return bool(mtp_compatible_for_model(model_path))
    except Exception:  # noqa: BLE001 - an unproven head is treated as absent
        return False


@router.post("/tune")
async def cluster_assistant_tune(request: AssistantTuneRequest) -> Any:
    """Fine-tune the deployment a ready start operation launched.

    Sweeps prefill chunk, sampler (golden-prompt quality) and MTP on the winning
    coordinator through the same completion seam the calibration uses, with REAL
    completions. The planner is not involved; this only tunes the running model.
    """

    from pathlib import Path

    from .start import stream_completion
    from .tuning_wiring import fine_tune_constellation

    store = _operation_store()
    record = store.get(request.operation_id)
    if record is None or record.kind != KIND_START or record.state != STATE_READY:
        return _refusal_response(
            "operation_not_ready",
            "tuning needs a ready start operation; run the search and pick a "
            "winner first",
            ACTION_STATUS,
        )
    endpoint = record.endpoint
    port = int(endpoint.port) if endpoint is not None else 0
    if not 1 <= port <= 65535:
        return _refusal_response(
            "coordinator_unresolved",
            "the winning coordinator's port is unknown; cannot tune it",
            ACTION_CONFIGURE_ENDPOINT,
        )
    if record.coordinator_local:
        target = ""
    else:
        device = _provisioning_seams().device_record(record.coordinator_node) or {}
        addrs = [
            str(item).strip()
            for item in (device.get("last_addrs") or ())
            if isinstance(item, str) and str(item).strip()
        ]
        ssh_user = str(device.get("ssh_user") or "").strip()
        # Fall back to the measured endpoint host when the device record carries
        # no address (resolve_targets does the same); the SSH user defaults to
        # the login user when the record does not name one.
        host = addrs[0] if addrs else (endpoint.host if endpoint is not None else "")
        target = f"{ssh_user}@{host}" if host and ssh_user else host
        if not target:
            return _refusal_response(
                "node_not_paired",
                f"{record.coordinator_node} has no paired SSH target; pair it "
                "again to tune it",
                ACTION_PAIR_MAC,
            )

    async def complete(body: dict) -> Any:
        # Tuning bodies carry only the sampling knobs; force a streamed
        # completion so the server returns the usage chunk (decode/TTFT) the
        # tuning reads. stream_completion is blocking I/O, so run it off the
        # event loop too (else the server cannot serve the stream).
        streamed = {
            **body,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        return await asyncio.to_thread(
            stream_completion, port=port, body=streamed, target=target
        )

    model_id = Path(record.model_path).name if record.model_path else ""
    result = await fine_tune_constellation(
        complete,
        model_id=model_id,
        mtp_capable=_tune_mtp_capable(record.model_path),
        # The chunk is a per-model setting, not a chat-request field; a
        # per-request sweep would measure noise. Tuning it for real needs a
        # redeploy per chunk, so the route marks it not-applicable here.
        chunk_applicable=False,
    )
    preset_key = _save_tuning_preset(record, result)
    return {
        "operation_id": record.operation_id,
        "model_id": model_id,
        "chunk": result.chunk,
        "chunk_applicable": result.chunk_applicable,
        "sampler": result.sampler,
        "mtp": result.mtp,
        "mtp_applicable": result.mtp_applicable,
        "chunk_rows": result.chunk_rows,
        "sampler_rows": result.sampler_rows,
        "mtp_rates": {str(k): v for k, v in result.mtp_rates.items()},
        "preset_key": preset_key,
    }


def _save_tuning_preset(record: Any, result: Any) -> str:
    """Merge the tuned sampler (and MTP switch) into the model's preset.

    Best-effort: a failed save never fails the tune, and a missing model path or
    rank order simply skips persistence. Returns the preset key ("" when skipped).
    The save merges: ``/tune`` does not measure the MTP depth, so a depth a
    redeploy sweep saved is kept (and vice versa the sampler is kept by the
    sweep), giving one combined preset a Start can replay.
    """

    return _merge_and_save_preset(
        record,
        chunk=_measured(result.chunk),
        mtp=_measured(result.mtp),
        sampler=_measured(result.sampler),
    )


def _measured(value: Any) -> Any:
    """A tuner value as a merge input: an absent measurement keeps the preset."""

    return KEEP if value is None else value


def _merge_and_save_preset(
    record: Any,
    *,
    chunk: Any = KEEP,
    mtp: Any = KEEP,
    mtp_depth: Any = KEEP,
    sampler: Any = KEEP,
) -> str:
    """One combined preset per (model, node set): merge the measured knobs.

    ``/tune`` and ``/tune-redeploy`` each measure only part of the tuning, so
    both save through here: an existing field is kept unless this call measured
    it. That is what makes "redeploy for MTP depth, then tune the sampler" — and
    the reverse order — end with both knobs in one preset.
    """

    return _merge_and_save_preset_for(
        str(getattr(record, "model_path", "") or ""),
        tuple(getattr(record, "rank_order", ()) or ()),
        chunk=chunk,
        mtp=mtp,
        mtp_depth=mtp_depth,
        sampler=sampler,
    )


def _merge_and_save_preset_for(
    model_path: str,
    node_ids: Sequence[str],
    *,
    chunk: Any = KEEP,
    mtp: Any = KEEP,
    mtp_depth: Any = KEEP,
    sampler: Any = KEEP,
) -> str:
    """The same save for any (model, node set), cluster or one-Mac local run.

    The key is ``preset_key(model_path, hardware_key(node_ids))`` — the same one
    ``_tuning_preset_patch`` reads back — so a preset saved by a local tune is
    the preset a single-node replay reads back, and a cluster Start keys its own on
    its elected node set. Returns "" when nothing could be saved (no model path,
    no nodes, or an unwritable store): tuning never fails on a preset write.
    """

    from datetime import UTC, datetime

    if not model_path or not node_ids:
        return ""
    node_key = hardware_key(node_ids)
    try:
        existing = _tuning_store().get(model_path, node_key)
    except OSError:
        existing = None
    preset = merge_preset(
        existing,
        model_signature=model_path,
        hardware_key=node_key,
        chunk=chunk,
        mtp=mtp,
        mtp_depth=mtp_depth,
        sampler=sampler,
        measured_at=datetime.now(UTC).isoformat(),
    )
    try:
        _tuning_store().save(preset)
    except OSError:
        return ""
    return preset.key


def _saved_preset_patch(record: Any) -> dict[str, Any] | None:
    """The saved tuning preset for this model + node config, as a settings patch.

    Only the knobs that are real ``ModelSettings`` fields are returned: the MTP
    adaptive depth and the sampler (temperature/top_p/top_k). The chunk is not a
    model setting in this build, so it is not resumed from a preset. No preset,
    or a preset whose knobs are all ``None``, yields ``None`` and Start behaves
    exactly as before.
    """

    model_path = str(getattr(record, "model_path", "") or "")
    rank_order = getattr(record, "rank_order", ()) or ()
    return _tuning_preset_patch(model_path, rank_order)


def _tuning_preset_patch(
    model_path: str, node_ids: Sequence[str]
) -> dict[str, Any] | None:
    """The same patch for any (model, node set) — one lookup for Start and any single-node replay.

    Start replays the preset for the elected rank order; a single-node run
    replays it for this one Mac (``hardware_key([this_mac])``). Both must read
    the store the same way, or a model tuned on one Mac set would be replayed
    differently by the two paths.
    """

    nodes = tuple(node_ids)
    if not model_path or not nodes:
        return None
    preset = _tuning_store().get(model_path, hardware_key(nodes))
    if preset is None:
        return None
    patch: dict[str, Any] = {}
    if preset.mtp_depth is not None:
        patch["mtp_adaptive_max_depth"] = preset.mtp_depth
    if isinstance(preset.sampler, Mapping):
        for key in ("temperature", "top_p", "top_k"):
            value = preset.sampler.get(key)
            if value is not None:
                patch[key] = value
    return patch or None


# ---------------------------------------------------------------------------
# tune-redeploy (MTP on/off + MTP depth need a redeploy per config)
# ---------------------------------------------------------------------------


class AssistantRedeployTuneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation_id: str = Field(min_length=1, max_length=128)


# A redeploy sweep deploys + measures the winning constellation once per config
# (MTP off, MTP adaptive, one per depth). That is minutes, so it is a background
# job like /search: POST returns a handle, GET polls. Jobs are ephemeral.
_redeploy_jobs: dict[str, dict[str, Any]] = {}
_redeploy_jobs_lock = threading.Lock()


def _create_redeploy_job(*, total: int) -> str:
    job_id = f"redeploy-{uuid.uuid4().hex[:12]}"
    with _redeploy_jobs_lock:
        _redeploy_jobs[job_id] = {
            "job_id": job_id,
            "state": "queued",
            "total": total,
            "done": 0,
            "message": "queued",
            "rows": [],
            "best": None,
            "preset_key": "",
        }
    return job_id


def _update_redeploy_job(job_id: str, **fields: Any) -> None:
    with _redeploy_jobs_lock:
        job = _redeploy_jobs.get(job_id)
        if job is not None:
            job.update(fields)


@router.post("/tune-redeploy")
async def cluster_assistant_tune_redeploy(
    request: AssistantRedeployTuneRequest,
) -> Any:
    """Tune the per-model knobs (MTP on/off, MTP depth) by redeploying per config.

    The sampler is a per-request knob and lives in ``/tune``; the MTP switch and
    the adaptive depth are per-model settings, so the only honest sweep is to
    deploy each config and measure its real rate (mtplx does the same; a one-off
    cost). The winner is persisted per (model, node config) so a later Start
    replays it without re-tuning.
    """

    from .redeploy_tuning import plan_configs

    store = _operation_store()
    record = store.get(request.operation_id)
    if record is None or record.kind != KIND_START or record.state != STATE_READY:
        return _refusal_response(
            "operation_not_ready",
            "redeploy tuning needs a ready start operation; run the search and "
            "pick a winner first",
            ACTION_STATUS,
        )
    measurement = _measurement_store().get(record.measurement_id)
    if measurement is None or measurement.model_layout() is None:
        return _stale_measurement_response(
            "unknown",
            "the winning operation's measurement is gone; measure the model "
            "again before redeploy tuning",
            record.measurement_id,
        )

    mtp_capable = _tune_mtp_capable(record.model_path)
    configs = plan_configs(mtp_capable=mtp_capable)
    job_id = _create_redeploy_job(total=len(configs))
    _schedule(
        asyncio.create_task(
            _run_redeploy_tune_job(
                job_id, store, record, measurement, configs, mtp_capable
            )
        )
    )
    return {
        "job_id": job_id,
        "state": "queued",
        "total": len(configs),
        "mtp_applicable": mtp_capable,
        "poll": f"/admin/api/cluster/assistant/tune-redeploy/{job_id}",
    }


@router.get("/tune-redeploy/{job_id}")
async def cluster_assistant_tune_redeploy_status(
    job_id: str = PathParam(min_length=1, max_length=128),
) -> Any:
    """Poll a redeploy tuning job: state, progress rows and the winning config."""

    with _redeploy_jobs_lock:
        job = _redeploy_jobs.get(job_id)
        snapshot = dict(job) if job is not None else None
    if snapshot is None:
        return _refusal_response(
            "redeploy_not_found",
            f"no redeploy tune job {job_id}; start one with POST /tune-redeploy",
            ACTION_STATUS,
        )
    return snapshot


async def _run_redeploy_tune_job(
    job_id: str,
    store: Any,
    record: Any,
    measurement: Any,
    configs: list,
    mtp_capable: bool,
) -> None:
    """Deploy each config once, measure it, stop it, and rank the real rates."""

    from .policy import diverse_candidates
    from .redeploy_tuning import RedeployRow, choose_candidate, sweep
    from .search_wiring import evaluate_constellation

    seams = _provisioning_seams()
    performance_store = _performance_store()
    probe_enabled = _performance_probe_enabled()
    model = measurement.model_layout()
    nodes = measurement.measurement_set().nodes
    fabric = measurement.fabric_links if measurement.fabric_measured else None
    wanted_order = tuple(record.rank_order)
    wanted_context = record.context_tokens

    _update_redeploy_job(job_id, state="running", message="measuring configs")
    # Tuning takes over the cluster: the ready deployment this operation launched
    # still holds the model in memory, so a bare redeploy would be refused with
    # "does not fit ... right now". Unload it once before the first config loads;
    # each config is stopped after its own measurement.
    if getattr(record, "deployment_id", ""):
        _update_redeploy_job(job_id, message="stopping the running cluster first")
        try:
            await asyncio.to_thread(
                _stop_on_coordinator, record, record.deployment_id
            )
            store.mark_stopped(record.operation_id)
        except Exception:  # a failed unload is reported by the next deploy
            pass
    done = 0
    rows: list[RedeployRow] = []

    async def measure(config: Any) -> RedeployRow:
        nonlocal done
        _update_redeploy_job(
            job_id,
            message=f"measuring {config.label} ({done + 1} of {len(configs)})",
        )
        generation_error = ""
        try:
            candidates = await asyncio.to_thread(
                diverse_candidates,
                nodes,
                model,
                target_context_tokens=wanted_context,
                rank0_extra_bytes=record.rank0_extra_bytes,
                mtp=config.mtp,
                execution_profile=record.execution_profile,
                microbatch_size=record.microbatch_size or 1,
                fabric_links=fabric,
            )
        except Exception as exc:  # a bad config is a row, never fatal
            candidates = []
            generation_error = f"{type(exc).__name__}: {exc}"
        chosen = choose_candidate(
            candidates,
            wanted_order=wanted_order,
            wanted_context=wanted_context,
        )
        if chosen is None:
            row = RedeployRow(
                config=config,
                feasible=False,
                error=(
                    generation_error
                    or "the winner's rank order/context is infeasible at this "
                    "config"
                ),
            )
        else:
            result = await evaluate_constellation(
                chosen,
                record=measurement,
                store=store,
                seams=seams,
                ops_for=_coordinator_ops,
                mtp=config.mtp,
                execution_profile=record.execution_profile,
                rank0_extra_bytes=record.rank0_extra_bytes,
                microbatch_size=record.microbatch_size,
                performance_store=performance_store,
                probe_enabled=probe_enabled,
                model_settings_patch=config.settings_patch() or None,
            )
            # The sweep is not the running cluster: stop each measured config so
            # the next one can deploy and the caller's cluster is not left
            # dangling. A failed stop is reported by the next deploy, not hidden.
            op = store.get(result.operation_id)
            if op is not None and result.deployment_id:
                try:
                    await asyncio.to_thread(
                        _stop_on_coordinator, op, result.deployment_id
                    )
                    store.mark_stopped(op.operation_id)
                except Exception:
                    pass
            row = RedeployRow(
                config=config,
                feasible=result.feasible,
                decode_tok_per_s=result.decode_tok_per_s,
                prefill_tok_per_s=result.prefill_tok_per_s,
                ttft_ms=result.ttft_ms,
                operation_id=result.operation_id,
            )
        done += 1
        rows.append(row)
        _update_redeploy_job(job_id, done=done, rows=[r.to_dict() for r in rows])
        return row

    try:
        outcome = await sweep(measure, configs, mtp_capable=mtp_capable)
        preset_key = _save_redeploy_preset(record, outcome)
        _update_redeploy_job(
            job_id,
            state="completed",
            message="done",
            rows=[r.to_dict() for r in outcome.rows],
            best=None
            if outcome.best is None
            else {
                "config": outcome.best.label,
                "mtp": outcome.best.mtp,
                "mtp_depth": outcome.best.mtp_depth,
            },
            mtp_applicable=outcome.mtp_applicable,
            preset_key=preset_key,
        )
    except Exception as exc:  # noqa: BLE001 - a failed job is a failed job
        _update_redeploy_job(job_id, state="failed", message=str(exc)[:300])


def _save_redeploy_preset(record: Any, outcome: Any) -> str:
    """Merge the redeploy winner (mtp + depth) into the model's tuning preset.

    The sampler preset from ``/tune`` is kept, so one Start applies both. A model
    without an MTP head keeps its previous mtp fields and saves only the row it
    measured. When the sweep measured a winner, its depth is the answer even if
    that answer is ``None`` (the adaptive config won) — a leftover depth from an
    earlier sweep must not outlive the measurement that replaced it.
    """

    best = outcome.best
    if best is None:
        return _merge_and_save_preset(record)
    return _merge_and_save_preset(
        record,
        mtp=best.mtp,
        mtp_depth=best.mtp_depth,
    )


# ---------------------------------------------------------------------------
# operations
# ---------------------------------------------------------------------------


@router.get("/tuning-preset")
async def cluster_assistant_tuning_preset(
    operation_id: str = Query(min_length=1, max_length=128),
) -> Any:
    """Read the saved tuning preset for an operation's model + node set.

    The UI shows what a later Start will replay, so it reads the preset through
    the same key Start uses: the operation's ``model_path`` and the *sorted* node
    set (not the rank order, so a flipped rank 0 still finds it). Any apply or
    start operation id is accepted. ``{"preset": null}`` means "nothing tuned for
    this model on these Macs yet" — never an error. No key, SSH target or secret
    is in the answer.
    """

    store = _operation_store()
    record = store.get(operation_id)
    if record is None:
        return _refusal_response(
            "operation_not_found",
            f"no operation {operation_id}; use a known apply or start operation id",
            ACTION_STATUS,
        )
    model_path = str(getattr(record, "model_path", "") or "")
    rank_order = getattr(record, "rank_order", ()) or ()
    if not model_path or not rank_order:
        return {"preset": None}
    preset = _tuning_store().get(model_path, hardware_key(rank_order))
    if preset is None:
        return {"preset": None}
    return preset.to_dict()






@router.get("/performance-probe")
async def get_assistant_performance_probe() -> Any:
    """Whether the post-``ready`` measurement is on for this Mac."""

    return {"enabled": _performance_probe_enabled()}


@router.post("/performance-probe")
async def set_assistant_performance_probe(
    request: AssistantPerformanceProbeRequest,
) -> Any:
    """Turn the post-``ready`` measurement on/off and persist the setting.

    It is a server setting, so it lives in ``settings.json``; this route is the
    in-app switch the Screen-6 card uses instead of an edit-JSON step. It never
    touches a running deployment.
    """

    settings = _global_settings()
    if settings is None:
        raise HTTPException(
            status_code=503,
            detail="global settings are not available on this server",
        )
    settings.server.assistant_performance_probe = bool(request.enabled)
    await asyncio.to_thread(settings.save)
    return {"enabled": bool(settings.server.assistant_performance_probe)}


@router.get("/session")
async def cluster_assistant_session() -> Any:
    """The last deployment, so a reloaded card can show Running/Stopped.

    Reads the persisted operation store only: no re-election, no probe, no
    secret. The returned inputs are enough to post apply and start again.
    """

    record = _operation_store().latest_session()
    return {
        "session": None if record is None else session_view(record).model_dump(mode="json")
    }


@router.get("/diagnostics", response_model=AssistantDiagnosticsResponse)
async def cluster_assistant_diagnostics(
    measurement_id: str | None = Query(None, min_length=1, max_length=128),
    operation_id: str | None = Query(None, min_length=1, max_length=128),
) -> Any:
    """A redacted support bundle for the last measurement and operation.

    Reads only the persisted records: no re-election, probe, stage or launch.
    An unknown id is an app-only bundle, never a 404, so a user can copy
    diagnostics even after the record was pruned. No key, token, SSH target,
    model path, prompt or completion is copied.
    """

    operations = _operation_store()
    if operation_id:
        operation = operations.get(operation_id)
    elif measurement_id:
        # Explicitly ask about one measurement: never mix it with the latest
        # operation of a different measurement.
        operation = None
    else:
        operation = operations.latest_session()
    if measurement_id:
        record = _measurement_store().get(measurement_id)
    elif operation is not None:
        record = _measurement_store().get(operation.measurement_id)
    else:
        records = _measurement_store().list()
        record = records[-1] if records else None
    return build_diagnostics(record, operation)


@router.get("/network-preflight", response_model=NetworkPreflightResponse)
async def cluster_assistant_network_preflight() -> Any:
    """Read the local interfaces and return exact bridge commands.

    Read-only by construction: the only work is ``ifconfig``/``networksetup``
    reads through the existing transport probe. Nothing is executed, no admin
    dialog opens and no peer is touched. The card shows the commands and asks
    the user to review them; the eventual privileged apply is a separate step.
    """

    return build_network_preflight(("127.0.0.1",))


@router.get("/operations/active")
async def cluster_assistant_active_operation() -> Any:
    """The apply/start this Mac is running, if any. Read-only.

    One origin's store cannot see another origin's, so a second Mac driving the
    same coordinator has to ask *the coordinator* this question before it
    activates. Declared before ``/operations/{operation_id}`` so "active" is not
    read as an operation id.
    """

    record = _operation_store().active()
    if record is None:
        return {"operation": None}
    return {
        "operation": {
            "operation_id": record.operation_id,
            "kind": record.kind,
            "state": record.state,
            "coordinator_node": record.coordinator_node,
            "running": True,
        }
    }


class AssistantOperationMirrorRequest(BaseModel):
    """One lifecycle state of another origin's operation (server-to-server).

    The browser never calls this: the origin mirrors its apply/start onto the
    elected coordinator through the admin seam so the coordinator's own store
    can answer ``operations/active`` — the cluster-wide lock (the apply/start design notes).
    """

    model_config = ConfigDict(extra="forbid")

    operation_id: str = Field(min_length=1, max_length=128)
    kind: str = Field(min_length=1, max_length=16)
    coordinator_node: str = Field(min_length=1, max_length=128)
    state: str = Field(min_length=1, max_length=16)
    model_path: str = Field(default="", max_length=512)
    context_tokens: int = Field(default=0, ge=0)


@router.post("/operations/mirror")
async def cluster_assistant_mirror_operation(
    request: AssistantOperationMirrorRequest,
) -> Any:
    """Register or release another origin's operation on this coordinator.

    Server-to-server (origin → coordinator over the admin seam); never called
    by the browser. A running mirror takes the cluster-wide lock and a terminal
    mirror releases it; both reuse the operation store, so ``operations/active``
    answers unchanged. Mirrors are lock markers, not this Mac's sessions.
    Declared before ``/operations/{operation_id}`` so "mirror" is not read as
    an operation id.
    """

    store = _operation_store()
    try:
        record = store.mirror(
            request.operation_id,
            request.kind,
            request.coordinator_node,
            request.state,
            model_path=request.model_path,
            context_tokens=request.context_tokens,
        )
    except OperationInProgressError as exc:
        return _operation_in_progress_response(exc)
    except ValueError as exc:
        return _refusal_response(
            "invalid_mirror_request", str(exc), ACTION_STATUS, status_code=400
        )
    return {"operation_id": record.operation_id, "running": record.running}


@router.get("/operations/{operation_id}", response_model=AssistantOperationStatus)
async def cluster_assistant_operation_status(
    operation_id: str = PathParam(min_length=1, max_length=128),
) -> Any:
    """Report one apply/start. Reads the store; changes nothing."""

    record = _operation_store().get(operation_id)
    if record is None:
        return JSONResponse(
            status_code=404,
            content=plan_error_payload(
                OPERATION_NOT_FOUND_CODE,
                f"no operation {operation_id} is stored; apply the plan first",
                ACTION_APPLY,
            ),
        )
    return operation_status(record)


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
    Mac route, reached over the admin seam apply/start already use, so the
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

    The same evidence apply/start resolve a target from — the measured address,
    the measurement's endpoint and the paired record — because "which Mac can I
    reach" must not have two answers. A Mac that was never paired is
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


def _usable_measurement(
    measurement_id: str,
    *,
    node_ids: Sequence[str] | None,
    identity: ModelIdentity | None,
) -> MeasurementRecord | JSONResponse:
    """The stored measurement, or the flat 409 ``plan`` returns for no answer.

    One shape for every reason (unknown, running, failed, stale, wrong node set,
    wrong model, wrong MTP) so a client cannot mistake "measure again" for a plan
    and does not have to parse prose.
    """

    record = _measurement_store().get(measurement_id)
    if record is None:
        return _stale_measurement_response(
            "unknown",
            f"no measurement {measurement_id} is stored; start one",
            measurement_id,
        )
    refusal = measurement_refusal(
        record,
        node_ids=node_ids,
        model=identity if identity is not None else record.model,
    )
    if refusal is not None:
        return _stale_measurement_response(
            refusal.reason, refusal.message, record.measurement_id
        )
    return record


WARNING_SPACE_FREED = "space_freed"


async def _plan_with_auto_free(
    record: MeasurementRecord, inputs: PlanInputs
) -> tuple[ApplyPlan | JSONResponse, list[OperationWarning]]:
    """Plan; if a Mac is short of disk, free safe models per the rule, then retry.

    Decision 2026-11-14: the delete rule is enforced automatically when a model
    must be distributed, instead of returning ``insufficient_disk`` and stopping.
    The rule itself is unchanged and lives in ``reclaim.deletable_models``: only
    a model that Mac listed, never the checkpoint the cluster is about to run,
    never the last cluster copy, largest first. A Mac that cannot free enough is
    left short and the retry returns the same one-action refusal.
    """

    planned = await asyncio.to_thread(_planned_deployment, record, inputs)
    if not isinstance(planned, JSONResponse):
        return planned, []
    try:
        body = json.loads(planned.body)
    except (TypeError, ValueError):
        return planned, []
    details = body.get("details")
    if body.get("code") != "insufficient_disk" or not isinstance(details, Mapping):
        return planned, []

    deleted, freed_by_node = await _auto_free_space(record, details)
    if not freed_by_node:
        return planned, []
    retried = await asyncio.to_thread(
        _planned_deployment, record, inputs, free_override=freed_by_node
    )
    warnings: list[OperationWarning] = []
    if deleted:
        names = ", ".join(str(item.get("model_id") or "") for item in deleted)
        warnings.append(
            OperationWarning(
                code=WARNING_SPACE_FREED,
                node_id=str(details.get("node_id") or ""),
                message=(
                    "the assistant freed space on the short Mac before staging by "
                    f"deleting unused model(s) per the delete rule: {names}"
                ),
            )
        )
    return retried, warnings


async def _auto_free_space(
    record: MeasurementRecord, details: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Delete the safe models that cover one short Mac's shortfall."""

    from .reclaim import deletable_models, select_for_space

    node_id = str(details.get("node_id") or "")
    needed = int(details.get("needed_bytes") or 0)
    free = int(details.get("free_bytes") or 0)
    if not node_id or needed <= 0:
        return [], {}
    candidates = deletable_models(record, node_id)
    selected = select_for_space(candidates, max(0, needed - free))
    if not selected:
        return [], {}
    target = _delete_target(record, node_id)
    if isinstance(target, JSONResponse):
        return [], {}
    ssh, port = target
    deleted: list[dict[str, Any]] = []
    freed = 0
    for entry in selected:
        try:
            await asyncio.to_thread(
                delete_model_on_mac,
                port=port,
                model_name=entry.model_id,
                target=ssh,
            )
        except Exception:  # a delete that fails leaves the shortfall in place
            continue
        deleted.append(entry.to_dict())
        freed += entry.size_bytes
    return deleted, ({node_id: freed} if freed else {})


def _planned_deployment(
    record: MeasurementRecord,
    inputs: PlanInputs,
    *,
    free_override: Mapping[str, int] | None = None,
) -> ApplyPlan | JSONResponse:
    """Re-elect and build the deployment, or the one refusal that answers it.

    An ``insufficient_disk`` refusal carries the short Mac's numbers in
    ``details`` so the route can free space per the frozen rule and retry.
    """

    try:
        return plan_apply(
            record,
            inputs,
            seams=_provisioning_seams(),
            free_override=free_override,
        )
    except ProvisioningError as exc:
        if exc.code == "insufficient_disk" and exc.details is not None:
            body = plan_error_payload(exc.code, exc.message, exc.action)
            body["details"] = exc.details
            return JSONResponse(status_code=409, content=body)
        return _refusal_response(exc.code, exc.message, exc.action)
    except NoFeasiblePlanError as exc:
        body = plan_error_payload("no_feasible_plan", str(exc), exc.action)
        body["measurement_id"] = record.measurement_id
        return JSONResponse(status_code=NO_FEASIBLE_PLAN_STATUS, content=body)
    except UnsupportedParallelismError as exc:
        return _refusal_response(
            "unsupported_parallelism", str(exc), ACTION_ADJUST_PLAN
        )
    except (ElectionError, PlanningError) as exc:
        return _refusal_response("planning_error", str(exc), ACTION_PLAN)
    except ValueError as exc:
        return _refusal_response("invalid_apply_request", str(exc), ACTION_PLAN)


def _applied_operation(
    store: OperationStore, request: AssistantStartRequest
) -> OperationRecord | None:
    """The applied plan a start may launch, matched by every identity it has."""

    if request.operation_id:
        record = store.get(request.operation_id)
        if (
            record is None
            or record.kind != KIND_APPLY
            or record.state != STATE_APPLIED
            or record.measurement_id != request.measurement_id
            or record.plan_signature != request.plan_signature
            or record.coordinator_node != request.coordinator_node
        ):
            return None
        return record
    return store.applied(
        measurement_id=request.measurement_id,
        plan_signature=request.plan_signature,
        coordinator_node=request.coordinator_node,
    )


def _busy_on_coordinator(plan: ApplyPlan) -> str | None:
    """The operation another origin is running on this plan's coordinator.

    An unanswerable question is not a busy coordinator: the operation then runs
    and reports the real reason the coordinator refused.
    """

    try:
        operation_id = active_operation(plan)
    except Exception:  # noqa: BLE001 - never turn a probe failure into a lock
        return None
    return operation_id or None


def _operation_in_progress_elsewhere(operation_id: str) -> JSONResponse:
    """409 for "the coordinator is already running something".

    Same code and action as the per-origin refusal, because the next step is the
    same: poll the operation that is running.
    """

    return JSONResponse(
        status_code=OPERATION_IN_PROGRESS_STATUS,
        content={
            **plan_error_payload(
                OPERATION_IN_PROGRESS_CODE,
                f"operation {operation_id} is running on the elected "
                "coordinator; poll it instead of starting a second one",
                ACTION_STATUS,
            ),
            "operation_id": operation_id,
        },
    )


def _operation_in_progress_response(exc: OperationInProgressError) -> JSONResponse:
    return JSONResponse(
        status_code=OPERATION_IN_PROGRESS_STATUS,
        content={
            **plan_error_payload(
                OPERATION_IN_PROGRESS_CODE,
                f"{exc.kind} operation {exc.operation_id} is still running; poll "
                "it instead of starting a second one",
                ACTION_STATUS,
            ),
            "operation_id": exc.operation_id,
        },
    )


def _begin(
    store: OperationStore,
    operation_id: str,
    state: str,
    phase: str,
    coordinator: str,
    *,
    nodes_total: int = 2,
) -> None:
    """Mark the job as started before the handle is returned.

    A client that polls immediately must not read ``queued`` for a job that is
    already running, and the state is forward-only, so this can only ever move
    the record.
    """

    store.advance(
        operation_id,
        state,
        progress=OperationProgress(
            phase=phase,
            message=f"the operation is running on {coordinator}",
            nodes_total=nodes_total,
        ),
    )


def _refusal_response(
    code: str, message: str, action: str, *, status_code: int = 409
) -> JSONResponse:
    """409/400 with a machine-readable code and exactly one next step."""

    return JSONResponse(
        status_code=status_code,
        content=plan_error_payload(code, message, action),
    )
