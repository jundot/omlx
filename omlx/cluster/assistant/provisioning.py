# SPDX-License-Identifier: Apache-2.0
"""Turn an elected plan into the deployment oMLX already knows how to run.

``plan`` answers "which Macs, in which order, holding which layers". The
activation pipeline in ``cluster/routes.py`` answers a different question: it
re-derives a ``ClusterDeployment`` from a ``ClusterDeploymentRequest`` and
refuses anything whose placement does not match what the caller approved. This
module is the single bridge between the two, and it is deliberately thin:

* it re-elects from the stored measurement — never from the request body — so
  an approved ``plan_signature`` is re-proved instead of trusted;
* it resolves each Mac's SSH target and address from the paired device
  registry at request time (never persisted, never guessed);
* it builds the request with measured budgets only: ``capacity_bytes`` and
  ``reserve_bytes`` come from ``NodeFacts`` through ``election.budget_for``'s
  arithmetic, so the deployment cannot be planned against a different ceiling
  than the election approved;
* it names the elected coordinator as ``hosts[0]`` and spawns rank 0 on that
  Mac, either as a local child or over SSH to the coordinator's own LAN address
  (the macOS Tahoe ring workaround). It is never another Mac: the address is one
  the coordinator owns, and ``resolve_targets`` proves that before apply.

The request is then prepared by the coordinator itself (``prepare`` in
``start.py``), because only the Mac that will launch the ranks can sign the
placement it is about to run. This module never activates anything, and it
never picks the app origin as a fallback: there is no branch here that prefers
the local Mac, and an unresolvable coordinator is a refusal with one action.

No secrets are stored or returned: SSH targets and addresses live in the
in-memory request for one apply, and the operation record keeps only node IDs,
signatures and the endpoint.
"""

from __future__ import annotations

import ipaddress
import json
import os
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from ..performance import ExecutionProfileName
from ..planner import (
    ModelLayout,
    inspect_safetensors_layout,
    synthetic_model_layout,
)
from ..staging import DEFAULT_REMOTE_PYTHON
from .election import ElectionResult, NodeFacts, elect
from .measurement import LOOPBACK_ENDPOINT_HOSTS
from .operations import (
    ACTION_ADD_MAC,
    ACTION_CONFIGURE_ENDPOINT,
    ACTION_COPY_MODEL,
    ACTION_DIAGNOSTICS,
    ACTION_FREE_DISK_SPACE,
    ACTION_PAIR_MAC,
    ACTION_PLAN,
    OperationEndpoint,
    OperationWarning,
)
from .policy import (
    ClusterPlanSelection,
    ResourcePolicy,
    SavedPolicy,
    solve_cluster_plan,
)
from .state import MeasurementRecord, ModelIdentity

# The value a request carries until the coordinator has signed the placement it
# will actually launch. It is not "approved": ``prepare`` replaces it with the
# 16-hex ``placement_signature`` the activation compares against, and any plan
# whose signature still equals this string is refused by that guard.
PENDING_PLACEMENT = "not-approved-yet"


# One node in the deployable hostfile shape. ``ssh`` is empty when the Mac has
# no paired SSH target; the coordinator's own entry always uses 127.0.0.1.
@dataclass(frozen=True)
class NodeTarget:
    node_id: str
    address: str
    ssh: str
    local: bool
    #: Whether this Mac had a *paired* record at all. An unpaired Mac that the
    #: measurement still saw (discovery, a manual address) is not a launch
    #: target and must ask for the pairing, not for an interface to be bound.
    paired: bool = True
    #: That Mac's own measured admin port. 0 means "not measured".
    port: int = 0

    @property
    def resolvable(self) -> bool:
        return bool(self.address) and (self.local or bool(self.ssh))


@dataclass(frozen=True)
class PlanInputs:
    """The plan-time inputs ``apply`` and ``start`` must agree on.

    These are exactly the fields that change the election without changing what
    was measured, which is why they are not part of a measurement's identity.
    Everything else — the Macs, admission, compute, links, endpoints, model and
    MTP setting — comes from the stored measurement.
    """

    measurement_id: str
    plan_signature: str
    model: ModelIdentity
    context_tokens: int
    execution_profile: str = "balanced"
    rank0_extra_bytes: int = 0
    mtp: bool = False
    coordinator_node: str = ""
    rank_order: tuple[str, ...] = ()
    microbatch_size: int | None = None
    origin_node: str = ""
    #: The context the user asked for, before the solver reduced it.
    target_context_tokens: int | None = None
    #: The resolved levers. ``None`` means an older caller that only has the
    #: context/MTP inputs; the election then keeps its conservative defaults.
    resource_policy: ResourcePolicy | None = None
    #: The user's explicit consent to the 2-Mac fallback. It salts the plan
    #: signature, so apply/start must carry it or the signature will not match.
    two_mac_fallback: bool = False


@dataclass(frozen=True)
class ApplyPlan:
    """Everything one apply needs, resolved once and checked by recomputation."""

    measurement: MeasurementRecord
    election: ElectionResult
    inputs: PlanInputs
    request: Any  # ClusterDeploymentRequest, imported lazily to keep this pure
    coordinator: NodeTarget
    targets: Mapping[str, NodeTarget] = field(default_factory=dict)
    source_host: str = "127.0.0.1"
    endpoint: OperationEndpoint | None = None
    warnings: tuple[OperationWarning, ...] = ()
    #: Writes the chosen per-model levers on one selected Mac. Injected through
    #: the seams so tests never touch a Mac; ``None`` means "no settings write",
    #: which is only correct for an older coordinator or a pure test.
    settings_writer: Any | None = None
    selection: ClusterPlanSelection | None = None

    @property
    def rank_order(self) -> tuple[str, ...]:
        return tuple(self.election.rank_order)

    def target_for(self, node_id: str) -> NodeTarget:
        return self.targets[node_id]


class ProvisioningError(RuntimeError):
    """Why an approved plan cannot become a deployment, and the one next step."""

    def __init__(
        self,
        code: str,
        message: str,
        action: str,
        details: Any = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.action = action
        #: Optional structured evidence (for ``insufficient_disk``: which Mac is
        #: short and by how much), so a caller can free space and retry instead
        #: of only returning the refusal.
        self.details = details


@dataclass(frozen=True)
class ProvisioningSeams:
    """The two read-only oMLX lookups this module performs.

    Both are injected so tests never touch a real registry or the network, and
    so the resolution rule is visible in one place.
    """

    #: node_id -> that Mac's paired registry record, if any.
    device_record: Callable[[str], Mapping[str, Any] | None]
    #: Addresses this Mac owns; used only to prove "the coordinator is me".
    local_addresses: Callable[[], Sequence[str]]
    #: (target, model path, per-model settings patch) -> None. Writes the
    #: approved levers on one selected Mac through the existing admin seam.
    #: ``None`` means the caller did not wire one (a pure test or an older
    #: coordinator); apply then skips the write and records no claim.
    settings_writer: Callable[[NodeTarget, str, Mapping[str, Any]], None] | None = (
        None
    )


def default_provisioning_seams() -> ProvisioningSeams:
    """The real registry lookup, imported lazily so this module stays cheap."""

    from ..discovery import local_addr_dicts
    from ..registry import get_device_registry

    def device_record(node_id: str) -> Mapping[str, Any] | None:
        try:
            registry = get_device_registry()
            # Only a *paired* Mac may become a launch target. Discovery also
            # lists unpaired Macs, and their record carries whatever addresses
            # the network happened to announce — including IPv6 link-local
            # ones. Taking one of those as an SSH target made apply try to
            # launch a rank on a Mac the user never approved
            # ("invalid SSH target: fe80::…") instead of asking for the pairing.
            row = registry.get(node_id)
            if row is None or not registry.is_paired(node_id):
                return None
            return row
        except (RuntimeError, ValueError):
            return None

    def local_addresses() -> Sequence[str]:
        try:
            # ``local_addr_dicts`` reports ``{"ip": …, "if_type": …}``. Reading a
            # different key made this list empty, so "is the coordinator this
            # Mac?" could never be true and apply refused the local coordinator
            # with "invalid SSH target: ''" — the ordinary case.
            return [
                str(item.get("ip") or item.get("address") or "")
                for item in local_addr_dicts()
            ]
        except Exception:  # noqa: BLE001 - address discovery is best effort
            return ()

    return ProvisioningSeams(
        device_record=device_record,
        local_addresses=local_addresses,
        settings_writer=write_model_settings_on_mac,
    )


def write_model_settings_on_mac(
    target: NodeTarget,
    model_path: str,
    patch: Mapping[str, Any],
) -> None:
    """Write one model's settings on a selected Mac through the admin seam.

    The coordinator's own oMLX is reached on loopback with the key that Mac
    stores; a worker is reached through the same SSH seam apply/start use. The
    model id is the Mac's own catalogue id for that path, falling back to the
    checkpoint folder name, which is how oMLX keys per-model settings. Nothing
    here reads, returns or stores a key.
    """

    if not patch:
        return
    from .start import (
        MODEL_SETTINGS_PATH,
        MODELS_PATH,
        RemoteAdminError,
        _local_admin,
        _model_entry,
        remote_admin,
    )

    port = target.port
    if not 1 <= int(port) <= 65535:
        raise ProvisioningError(
            "configure_endpoint",
            f"the endpoint port on {target.node_id} was not measured, so its "
            "settings cannot be written; measure again",
            ACTION_CONFIGURE_ENDPOINT,
        )
    model_id = ""
    try:
        if target.local:
            listing = _local_admin(port, MODELS_PATH)
        else:
            listing = remote_admin(target.ssh, port, MODELS_PATH, None, method="GET")
        entry = _model_entry(listing, model_path)
        model_id = "" if entry is None else str(entry.get("id") or "").strip()
    except Exception:
        # The catalogue is evidence for the model id; the folder name is the
        # documented fallback and still names the same settings record.
        model_id = ""
    if not model_id:
        model_id = Path(model_path).name
    if not model_id:
        raise ProvisioningError(
            "policy_settings_failed",
            f"the model has no id on {target.node_id}, so its settings could "
            "not be written; check the model on that Mac and apply again",
            ACTION_DIAGNOSTICS,
        )
    path = MODEL_SETTINGS_PATH.format(model_id=model_id)
    try:
        if target.local:
            _local_admin(port, path, dict(patch), method="PUT")
        else:
            remote_admin(target.ssh, port, path, dict(patch), method="PUT")
    except RemoteAdminError as exc:
        # A Mac that received the checkpoint through staging may have it on
        # disk before its model pool lists it, so the settings route answers
        # 404 "Model not found". The thin layer writes the same per-model JSON
        # before the load; do that rather than leaving a half-written policy.
        if exc.status != 404:
            raise
        _write_model_settings_file(target, model_id, patch)


# The worker copy of the fallback below. It runs as a bare ``python -c`` on the
# worker, so it resolves the base path the same way the inventory snippet does
# (``GlobalSettings.load()``: ``OMLX_BASE_PATH`` > the app bootstrap > ~/.omlx)
# and is stdlib-only after that one oMLX import. Keep it in sync with
# ``_merge_model_settings_file``: a missing file may start a fresh document, but
# any other read error aborts (never reset an unreadable file), and the swap
# uses a unique temp name so concurrent writers never rename each other's
# partial write.
_SETTINGS_FILE_SNIPPET = (
    "import json,os,sys,uuid\n"
    "from omlx.settings import GlobalSettings\n"
    "payload=json.loads(sys.argv[1])\n"
    "path=os.path.join(str(GlobalSettings.load().base_path),'model_settings.json')\n"
    "try:\n"
    "    handle=open(path,encoding='utf-8')\n"
    "except FileNotFoundError:\n"
    "    document={'version':1,'models':{}}\n"
    "else:\n"
    "    with handle:\n"
    "        document=json.load(handle)\n"
    "if not isinstance(document,dict) or not isinstance(document.get('models',{}),dict):\n"
    "    raise ValueError('unexpected model settings document: '+path)\n"
    "entry=document.setdefault('models',{}).setdefault(payload['model_id'],{})\n"
    "entry.update(payload['patch'])\n"
    "os.makedirs(os.path.dirname(path),exist_ok=True)\n"
    "temporary=path+'.'+uuid.uuid4().hex+'.tmp'\n"
    "with open(temporary,'w',encoding='utf-8') as handle:\n"
    "    json.dump(document,handle,indent=2)\n"
    "os.replace(temporary,path)\n"
    "print(json.dumps({'ok':True}))\n"
)

# Serializes the local read-modify-write so two in-process writers keep both
# model entries; the unique temp name alone only prevents the rename race.
_SETTINGS_FILE_LOCK = threading.Lock()


def _local_model_settings_path() -> Path:
    """The settings file this Mac's oMLX actually reads.

    ``ModelSettingsManager`` keys the file off the server's base path, which
    ``OMLX_BASE_PATH`` or the app's bootstrap file can move away from ``~/.omlx``.
    Resolve it the same way — the running server's settings when initialized,
    else ``GlobalSettings.load()`` (the pattern ``memory_guard`` documents) —
    instead of hardcoding the default, or the policy lands in the wrong place on
    a moved data root.
    """

    from ...settings import GlobalSettings, get_settings

    try:
        settings = get_settings()
    except RuntimeError:
        settings = GlobalSettings.load()
    return Path(settings.base_path) / "model_settings.json"


def _merge_model_settings_file(
    path: Path, model_id: str, patch: Mapping[str, Any]
) -> None:
    """Merge one model's patch into the settings file, or leave it untouched.

    A missing file starts a fresh document (the first write). Any other read
    error — permissions, corrupt JSON — aborts: replacing an unreadable file
    with an empty document would wipe every model's settings on this Mac. The
    merge semantics are unchanged (``entry.update(patch)`` per model id, the
    version/models structure preserved). Mirrors ``_SETTINGS_FILE_SNIPPET``.
    """

    with _SETTINGS_FILE_LOCK:
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            document: Any = {"version": 1, "models": {}}
        else:
            document = json.loads(raw)
        if not isinstance(document, dict) or not isinstance(
            document.get("models", {}), dict
        ):
            raise ValueError(f"unexpected model settings document: {path}")
        entry = document.setdefault("models", {}).setdefault(model_id, {})
        entry.update(dict(patch))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(document, indent=2), encoding="utf-8")
        os.replace(temporary, path)


def _write_model_settings_file(
    target: NodeTarget, model_id: str, patch: Mapping[str, Any]
) -> None:
    """Update that Mac's per-model settings file when its pool does not list it.

    Stdlib only, on the Mac itself, exactly like the thin layer: the settings
    are read when the rank loads the model, so this is valid both after staging
    and before the pool refresh. No key crosses the seam. A read error aborts
    the write and leaves the file unchanged; the exception propagates into the
    apply failure path (``_write_policy_settings``), never a silent reset.
    """

    payload = json.dumps({"model_id": model_id, "patch": dict(patch)}, sort_keys=True)
    if target.local:
        path = _local_model_settings_path()
        try:
            _merge_model_settings_file(path, model_id, patch)
        except Exception as exc:
            raise ProvisioningError(
                "policy_settings_failed",
                f"the model settings file at {path} could not be updated and was "
                f"left unchanged: {exc}; fix the file on this Mac and apply again",
                ACTION_DIAGNOSTICS,
            ) from exc
        return
    from ..staging import DEFAULT_REMOTE_PYTHON, run_remote_python

    run_remote_python(
        target.ssh,
        _SETTINGS_FILE_SNIPPET,
        payload,
        description="write the approved model settings on the worker",
        python_executable=DEFAULT_REMOTE_PYTHON,
        timeout=120.0,
    )


def model_layout_for(model: ModelIdentity) -> ModelLayout:
    """The planner layout for the checkpoint a plan is about.

    One implementation, used by ``plan``, ``apply`` and ``start``: a real
    checkpoint is inspected on this Mac, a pre-download size plus layer count is
    planned as a synthetic layout. Reading headers is not a probe and not a
    network call.
    """

    if model.model_path:
        return inspect_safetensors_layout(model.model_path)
    return synthetic_model_layout(
        total_weight_bytes=model.model_size_bytes or 0,
        layer_count=model.layer_count,
    )


def elect_for_apply(
    record: MeasurementRecord,
    inputs: PlanInputs,
) -> tuple[ElectionResult, ClusterPlanSelection | None]:
    """Re-elect from the stored measurement with the apply-time inputs.

    The election is the authority on rank 0. Nothing here reads the origin, the
    request's opinion about the coordinator, or this Mac's own identity.

    When the request carries a resolved resource policy, the same solver that
    produced the plan is asked again with that policy and the approved Macs, so
    a drifted lever or a grown subset is caught by the signature check. An
    older caller without a policy keeps the conservative election.
    """

    layout = record.model_layout() or model_layout_for(inputs.model)
    fabric_links = record.fabric_links if record.fabric_measured else None
    if inputs.resource_policy is None:
        return (
            elect(
                record.measurement_set().nodes,
                layout,
                context_tokens=inputs.context_tokens,
                rank0_extra_bytes=inputs.rank0_extra_bytes,
                mtp=inputs.mtp,
                workload_profile=cast(
                    ExecutionProfileName, inputs.execution_profile
                ),
                microbatch_size=inputs.microbatch_size or 1,
                tensor_parallel_size=1,
                origin_node=inputs.origin_node or None,
                fabric_links=fabric_links,
                two_mac_fallback=inputs.two_mac_fallback,
            ),
            None,
        )
    selection = solve_cluster_plan(
        record.measurement_set().nodes,
        layout,
        target_context_tokens=inputs.target_context_tokens or inputs.context_tokens,
        rank0_extra_bytes=inputs.rank0_extra_bytes,
        mtp=inputs.mtp,
        execution_profile=cast(ExecutionProfileName, inputs.execution_profile),
        microbatch_size=inputs.microbatch_size or 1,
        saved=SavedPolicy.from_mapping(record.model_settings),
        policy=inputs.resource_policy,
        forced_node_ids=inputs.rank_order,
        origin_node=inputs.origin_node or None,
        fabric_links=fabric_links,
        two_mac_fallback=inputs.two_mac_fallback,
    )
    return selection.election, selection


def verify_approved_plan(result: ElectionResult, inputs: PlanInputs) -> None:
    """Refuse a signature, coordinator or rank order the election did not produce.

    The client's claim is only a claim: the plan it approved is re-derived from
    the measurement and must agree in all three parts. A drifted plan is not
    "close enough" — it names different Macs in a different order.
    """

    if result.plan_signature != inputs.plan_signature:
        raise ProvisioningError(
            "plan_mismatch",
            "the plan you approved no longer matches the election for this "
            "measurement, context and MTP setting; plan again and approve what "
            "it shows",
            ACTION_PLAN,
        )
    if inputs.coordinator_node and result.coordinator_node != inputs.coordinator_node:
        raise ProvisioningError(
            "plan_mismatch",
            f"the approved plan names {inputs.coordinator_node} as coordinator, "
            f"but the election for this measurement elects "
            f"{result.coordinator_node}; plan again",
            ACTION_PLAN,
        )
    if inputs.rank_order and tuple(inputs.rank_order) != tuple(result.rank_order):
        raise ProvisioningError(
            "plan_mismatch",
            "the approved rank order is not the order the election produces for "
            f"this measurement ({' -> '.join(result.rank_order)}); plan again",
            ACTION_PLAN,
        )


def resolve_targets(
    record: MeasurementRecord,
    result: ElectionResult,
    seams: ProvisioningSeams,
) -> dict[str, NodeTarget]:
    """Resolve every elected Mac's paired SSH target and communication address.

    Read-only, at request time, and never persisted. The coordinator is called
    local only when one of its own addresses is an address this Mac owns —
    "there is no paired record" is a missing pairing, not proof of locality, and
    is never used to fall back to the local Mac.
    """

    measured = {node.node_id: node for node in record.measurement_set().nodes}
    local = {str(address) for address in seams.local_addresses() if address}
    targets: dict[str, NodeTarget] = {}
    for node_id in result.rank_order:
        facts = measured.get(node_id)
        if facts is None:
            raise ProvisioningError(
                "plan_mismatch",
                f"the election names {node_id}, which the measurement does not "
                "cover; measure again",
                ACTION_PLAN,
            )
        device = seams.device_record(node_id) or {}
        paired = [
            str(item).strip()
            for item in (device.get("last_addrs") or ())
            if isinstance(item, str) and str(item).strip()
        ]
        endpoint = record.endpoint_for(node_id)
        endpoint_host = "" if endpoint is None else endpoint.host
        candidates = [facts.endpoint_host, endpoint_host, *paired]
        address = _first_address(*candidates)
        ssh_user = str(device.get("ssh_user") or "").strip()
        paired_host = _first_address(*paired)
        ssh = f"{ssh_user}@{paired_host}" if paired_host and ssh_user else paired_host
        is_local = any(candidate in local for candidate in candidates if candidate)
        if is_local:
            # Rank 0 on this Mac runs in-process. If another Mac is rank 0 then
            # this Mac is a worker that the coordinator has to dial, and a Mac
            # has no paired record of itself — its own measured address is the
            # one piece of evidence about where it is. Leaving it empty refused
            # the whole apply with ``node_not_paired``.
            ssh = "" if node_id == result.coordinator_node else ssh or address
        targets[node_id] = NodeTarget(
            node_id=node_id,
            address=address,
            ssh=ssh,
            local=is_local,
            paired=not is_local and bool(device),
            port=0 if endpoint is None else endpoint.port,
        )
    return targets


def plan_apply(
    record: MeasurementRecord,
    inputs: PlanInputs,
    *,
    seams: ProvisioningSeams | None = None,
    free_override: Mapping[str, int] | None = None,
) -> ApplyPlan:
    """Re-elect, verify, resolve and build one apply's deployment request.

    ``free_override`` counts space a caller already freed on a Mac (the
    automatic reclaim path) so the room check does not refuse a copy that now
    fits.
    """

    seams = seams or default_provisioning_seams()
    result, selection = elect_for_apply(record, inputs)
    verify_approved_plan(result, inputs)

    if len(result.rank_order) < 2:
        # ``ClusterDeployment`` is 2–64 hosts, and a one-Mac "cluster" is a
        # local engine, not a pipeline. Refuse with one action instead of
        # inventing a second launch mode.
        raise ProvisioningError(
            "single_node_plan",
            "this plan runs the model on one Mac, and the distributed cluster "
            "path needs at least two; add another Mac and measure again",
            ACTION_ADD_MAC,
        )
    targets = resolve_targets(record, result, seams)
    coordinator = targets[result.coordinator_node]
    if not coordinator.address and not coordinator.paired:
        # Nothing about this Mac is trusted: it was never paired, so it has no
        # address, no SSH target and no cluster key. Telling the user to bind an
        # interface would be misleading — the next step is the pairing flow.
        raise ProvisioningError(
            "node_not_paired",
            f"{coordinator.node_id} is not paired with this Mac; pair it and "
            "measure again",
            ACTION_PAIR_MAC,
        )
    if not coordinator.address:
        raise ProvisioningError(
            "coordinator_unresolved",
            f"{coordinator.node_id} reported no private address to launch on; "
            "bind oMLX to this cluster's private interface and measure again",
            ACTION_CONFIGURE_ENDPOINT,
        )
    for node_id, target in targets.items():
        if node_id != coordinator.node_id and not target.ssh:
            raise ProvisioningError(
                "node_not_paired",
                f"{node_id} has no paired SSH target on this Mac; pair it again "
                "and measure again",
                ACTION_PAIR_MAC,
            )
    _require_room_for_the_copy(
        record, result, inputs, free_override=free_override
    )
    _require_wired_limit(record, result)
    endpoint = _endpoint(record, coordinator)
    if endpoint is None:
        raise ProvisioningError(
            "coordinator_unresolved",
            f"{coordinator.node_id} reported no dialable endpoint; bind oMLX to "
            "this cluster's private interface and measure again",
            ACTION_CONFIGURE_ENDPOINT,
        )
    source_host = model_source(
        record, result, targets, coordinator=coordinator, inputs=inputs
    )
    request = build_deployment_request(
        record,
        result,
        targets,
        coordinator=coordinator,
        source_host=source_host,
        inputs=inputs,
    )
    return ApplyPlan(
        measurement=record,
        election=result,
        inputs=inputs,
        request=request,
        coordinator=coordinator,
        targets=targets,
        source_host=source_host,
        endpoint=endpoint,
        warnings=_plan_warnings(record, selection),
        settings_writer=seams.settings_writer,
        selection=selection,
    )


def _plan_warnings(
    record: MeasurementRecord, selection: ClusterPlanSelection | None
) -> tuple[OperationWarning, ...]:
    """The policy warnings, in the operation's own warning shape."""

    if selection is None:
        return ()
    return tuple(
        OperationWarning(
            code=item.code,
            node_id=item.node_id,
            message=item.message,
        )
        for item in selection.warnings
    )


def _require_wired_limit(
    record: MeasurementRecord, result: ElectionResult
) -> None:
    """Refuse a stage a known ``iogpu.wired_limit_mb`` would reject at load.

    The full helper that raises the kernel limit is Phase 5. Until then a
    too-low limit is a preflight refusal with one explicit action, instead of a
    forty-minute staging job that ends in oMLX refusing the load. An unreadable
    limit is unknown, never "zero", and never refuses a plan.
    """

    selected = result.selected
    if selected is None:
        return
    measured = {node.node_id: node for node in record.measurement_set().nodes}
    for stage in selected.stages:
        facts = measured.get(stage.node_id)
        if facts is None or facts.wired_limit_mb is None:
            continue
        limit = facts.wired_limit_mb * 1024 * 1024
        if limit <= 0 or stage.resident_bytes <= limit:
            continue
        wanted = (stage.resident_bytes + 1024 * 1024 - 1) // (1024 * 1024)
        raise ProvisioningError(
            "wired_limit_too_low",
            f"{stage.node_id} can wire {facts.wired_limit_mb} MiB but this plan "
            f"needs {wanted} MiB; raise the limit on that Mac with "
            f"`sudo sysctl iogpu.wired_limit_mb={wanted}` and apply again",
            ACTION_DIAGNOSTICS,
        )


def model_source(
    record: MeasurementRecord,
    result: ElectionResult,
    targets: Mapping[str, NodeTarget],
    *,
    coordinator: NodeTarget,
    inputs: PlanInputs,
) -> str:
    """Which Mac owns the complete checkpoint, in the coordinator's own terms.

    ``127.0.0.1`` means the coordinator (the process that copies the files), so
    "the coordinator has it" is the only case that may be local. Otherwise a
    peer's paired SSH target is used — never the app origin as an assumption,
    and never an invented path.
    """

    measured = {node.node_id: node for node in record.measurement_set().nodes}
    if measured[coordinator.node_id].model_present:
        return "127.0.0.1"
    for node_id in result.rank_order:
        facts = measured[node_id]
        target = targets[node_id]
        if not facts.model_present or not target.ssh:
            continue
        # A local *worker* is a valid source too when the elected coordinator is
        # another Mac.  `target.ssh` is the local address in that case; only a
        # local *coordinator* has an empty ssh here, and that case was already
        # answered by the 127.0.0.1 branch above.
        return target.ssh
    raise ProvisioningError(
        "model_source_missing",
        "the model is not on the coordinator and no other measured Mac with the "
        "checkpoint can be reached over SSH; copy the model to "
        f"{coordinator.node_id} and apply again",
        ACTION_COPY_MODEL,
    )


def _require_room_for_the_copy(
    record: MeasurementRecord,
    result: ElectionResult,
    inputs: PlanInputs,
    *,
    free_override: Mapping[str, int] | None = None,
) -> None:
    """Refuse before staging when a Mac cannot hold the copy it needs.

    "Not enough disk space on Mac Studio" is a one-action problem with a known
    fix — delete a model you no longer use on that Mac. Discovering it half way
    through a 100 GiB copy only wastes the user's evening. ``free_override``
    lets a caller that already freed space (the automatic reclaim path) count
    the bytes it made available before this re-check.
    """

    layout = record.model_layout()
    if layout is None or not inputs.model.model_path:
        return
    needed = layout.fixed_weight_bytes + sum(layout.layer_weight_bytes)
    if inputs.mtp:
        needed += layout.mtp_weight_bytes
    for node_id in result.rank_order:
        facts = next(
            (node for node in record.measurement_set().nodes if node.node_id == node_id),
            None,
        )
        if facts is None or facts.model_present:
            continue
        free = facts.disk_free_bytes
        if free_override and node_id in free_override:
            free = (free or 0) + int(free_override[node_id])
        if free is None or free >= needed:
            continue
        raise ProvisioningError(
            "insufficient_disk",
            f"{node_id} needs {needed / 1024**3:.1f} GiB free for this model and has "
            f"{free / 1024**3:.1f} GiB; delete a model you no longer use on that Mac "
            "or choose a smaller model, then measure again",
            ACTION_FREE_DISK_SPACE,
            # Structured evidence so the apply route may free space and retry.
            {
                "node_id": node_id,
                "needed_bytes": int(needed),
                "free_bytes": int(free),
            },
        )


def build_deployment_request(
    record: MeasurementRecord,
    result: ElectionResult,
    targets: Mapping[str, NodeTarget],
    *,
    coordinator: NodeTarget,
    source_host: str,
    inputs: PlanInputs,
) -> Any:
    """The ``ClusterDeploymentRequest`` the existing activation expects.

    Budgets come from the measurement through the same arithmetic as
    ``election.budget_for``: ``reserve_bytes`` is what the Mac may *not* use, so
    the planner's headroom is measured against measured admission. They are
    marked ``manual_memory_limit`` because measured admission *is* the explicit
    ceiling — a role reserve or a default must not silently widen it between the
    plan and the launch.
    """

    from ..routes import (
        ClusterDeploymentRequest,
        ClusterHostRequest,
        ClusterPlanNodeRequest,
    )
    from .election import budget_for

    if not inputs.model.model_path:
        raise ProvisioningError(
            "model_not_downloaded",
            "this plan was built for a model size, not a checkpoint on disk; "
            "download the model on one of these Macs, measure it again and plan",
            ACTION_COPY_MODEL,
        )

    measured = {node.node_id: node for node in record.measurement_set().nodes}
    nodes: list[Any] = []
    hosts: list[Any] = []
    for rank, node_id in enumerate(result.rank_order):
        facts: NodeFacts = measured[node_id]
        target = targets[node_id]
        capacity = facts.capacity_bytes
        # The same admission and tier the election planned against. Reading the
        # p05 floor here while the election used p50 would hand the coordinator
        # a smaller Mac and fail the placement it just approved.
        policy = inputs.resource_policy
        budget = budget_for(
            facts,
            rank=rank,
            tier=None if policy is None else policy.memory_guard_tier,
            admission="p05" if policy is None else policy.admission,
        )
        nodes.append(
            ClusterPlanNodeRequest(
                node_id=node_id,
                capacity_bytes=capacity,
                reserve_bytes=max(0, budget.reserve_bytes),
                manual_memory_limit=True,
                role="headless",
                memory_guard_tier=budget.memory_guard_tier,
                performance=(
                    None
                    if facts.compute is None
                    else facts.compute.performance_profile(node_id, rank).to_dict()
                ),
            )
        )
        hosts.append(
            ClusterHostRequest(
                node_id=node_id,
                # Rank 0 runs on the coordinator as its own local child. The
                # deployment contract requires exactly "127.0.0.1" here
                # (``ClusterDeployment``), and the validation cluster runs as a
                # *line* rather than a macOS bridge ring, so the Tahoe ring
                # workaround (spawn rank 0 over SSH to this Mac's own LAN
                # address) does not apply and would be rejected.
                ssh=("127.0.0.1" if rank == 0 else target.ssh),
                ips=[target.address],
                # The interpreter that Mac itself reported to the measurement.
                # Rank 0 is spawned by the MLX launcher over SSH to the
                # coordinator's own LAN address (Tahoe ring workaround), so it is
                # not "the local process" any more. The published shim exports
                # the app bundle's PYTHONPATH and is the safe fallback when the
                # measurement did not report one (the slim app's venv no longer
                # installs the omlx package). The deployment contract requires
                # an absolute path, so resolve the shim's ~ here.
                python_executable=str(
                    Path(
                        facts.python_executable or DEFAULT_REMOTE_PYTHON
                    ).expanduser()
                ),
            )
        )

    return ClusterDeploymentRequest(
        model_path=inputs.model.model_path,
        model_source=None if source_host == "127.0.0.1" else source_host,
        # Ring over the measured link, always. The measurement's transport
        # ("ring" / "jaccl" / "jaccl-ring") is evidence of what the cabling can
        # do, not a deployment backend the assistant can honour: JACCL requires
        # a full per-host RDMA connectivity matrix, and this path has no pairwise
        # path data to fill it with. Copying the measured value through made
        # apply fail on a 3-Mac cluster with "JACCL requires a full RDMA
        # connectivity matrix". A ring over the same Thunderbolt bridge measured
        # 26.9 Gbit/s between two of these Macs, which is the MVP transport.
        backend="ring",
        nodes=nodes,
        hosts=hosts,
        preflight=True,
        execution_profile=cast(ExecutionProfileName, inputs.execution_profile),
        allocation="balanced",
        # The assistant never runs the distributed performance probe as part of
        # a start: it needs an approved deployment and would only tune execution
        # settings here. Leaving it off keeps the plan the user approved the
        # plan that launches, and keeps an unmeasured rate unmeasured.
        auto_tune=False,
        sampling_rank_only=True,
        target_context_tokens=inputs.context_tokens,
        tensor_parallel_size=1,
        approved_placement=PENDING_PLACEMENT,
        mtp=inputs.mtp,
    )


def placement_warning(
    result: ElectionResult, signed_plan: Mapping[str, Any]
) -> OperationWarning | None:
    """Say so when the coordinator's planner cut the layers differently.

    The rank order, the Macs and the budgets are the approved ones — the host
    list is the request's. The layer cut is re-derived by oMLX's own planner on
    the coordinator (it adds MoE/Engram offload the election's memory-only plan
    does not model), so it can differ. That is a fact worth recording, not a
    reason to refuse: the alternative is either launching a split nobody
    approved silently, or blocking real models whose offload the election
    deliberately ignores.
    """

    selected = result.selected
    if selected is None:
        return None
    elected = {
        stage.rank: (stage.first_layer, stage.last_layer) for stage in selected.stages
    }
    rows: dict[int, tuple[int, int]] = {}
    for item in signed_plan.get("assignments") or ():
        if not isinstance(item, Mapping):
            continue
        rank = _optional_int(item.get("rank"))
        start = _optional_int(item.get("start_layer"))
        end = _optional_int(item.get("end_layer"))
        if rank is None or start is None or end is None:
            continue
        rows[rank] = (start, end)
    if not rows or rows == elected:
        return None
    return OperationWarning(
        code="placement_recomputed",
        node_id=result.coordinator_node,
        message=(
            "the coordinator's planner placed its own layer cut for these Macs "
            "(the rank order and budgets are the approved ones); review the "
            "deployment on the coordinator if the split matters"
        ),
    )


def _endpoint(
    record: MeasurementRecord, coordinator: NodeTarget
) -> OperationEndpoint | None:
    """The coordinator's dialable endpoint from the measurement, or ``None``."""

    endpoint = record.endpoint_for(coordinator.node_id)
    if endpoint is None or not endpoint.resolved:
        return None
    host = endpoint.host.strip()
    if host in LOOPBACK_ENDPOINT_HOSTS:
        return None
    return OperationEndpoint(node=coordinator.node_id, host=host, port=endpoint.port)


def _first_address(*candidates: Any) -> str:
    """The first value that is a communication address a rank can bind to.

    IPv4 wins over IPv6: a link-local ``fe80::…`` address is announced by every
    interface but cannot be typed into an SSH target or dialled from another
    Mac without a scope id, so taking it because it came first in a discovered
    record sent apply after a Mac it could not reach.
    """

    usable: list[str] = []
    for value in candidates:
        text = str(value or "").strip()
        if not text or text in LOOPBACK_ENDPOINT_HOSTS:
            continue
        try:
            ipaddress.ip_address(text.split("%", 1)[0])
        except ValueError:
            continue
        usable.append(text.split("%", 1)[0])
    for address in usable:
        if ":" not in address:
            return address
    return usable[0] if usable else ""


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
