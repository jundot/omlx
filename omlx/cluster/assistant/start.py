# SPDX-License-Identifier: Apache-2.0
"""Run apply and start on the *elected* coordinator, never on the origin.

The two jobs this module drives are the ones the project design notes describes:

``apply``  prepare and stage the model on the elected Mac; start nothing.
``start``  health, activation, load, readiness canary, endpoint ready.

Both run where rank 0 runs. The deployment requires rank 0 to be spawned by the
coordinator itself — as a local child or over SSH to the coordinator's own LAN
address (the macOS Tahoe ring workaround). The only process that can stage,
preflight and launch a deployment is one running *on the coordinator*. When the app
origin is not that Mac, this module reaches the coordinator over the existing
SSH seam (``staging.run_remote_python``, which goes through ``ssh_policy``) and
asks the coordinator's own oMLX server to run the activation it already has.
Nothing here starts a second runtime, keeps a second credential, or launches
rank 0 anywhere else: there is deliberately no branch that names the origin as
a coordinator, and a failed coordinator start fails the operation.

Two rules keep the approved plan honest.

* ``prepare`` runs on the coordinator and returns *its* signed placement, which
  becomes ``approved_placement``. The activation then re-checks it against the
  plan it recomputes, so "the plan you approved" is still a fact.
* ``start`` re-prepares and refuses if the placement changed since ``apply``,
  so a start can never launch a rank map that apply did not stage.

Progress is recorded through ``operations.OperationStore``; ``endpoint.ready``
only becomes true after the activation reports a passed readiness canary, i.e.
one real completion with every rank resident. A missing distributed performance
probe is not a failure: this module never invents a rate, and the record carries
no tokens-per-second field at all.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ..deployment import validate_ssh_target
from ..liveness import PeerLostError, check_peers, probe_peer, raise_if_peer_lost
from ..staging import DEFAULT_REMOTE_PYTHON, run_remote_python
from .models import ACTION_ADJUST_PLAN
from .operations import (
    ACTION_APPLY,
    ACTION_DIAGNOSTICS,
    ACTION_PAIR_MAC,
    ACTION_PLAN,
    ACTION_RETRY_COORDINATOR,
    ACTION_STATUS,
    OPERATION_IN_PROGRESS_CODE,
    STATE_APPLIED,
    STATE_APPLYING,
    STATE_LOADING,
    STATE_READY,
    STATE_STARTING,
    OperationEndpoint,
    OperationObserved,
    OperationProgress,
    OperationStore,
    OperationWarning,
)
from .policy import ResourcePolicy, policy_model_settings
from .provisioning import (
    ApplyPlan,
    PlanInputs,
    ProvisioningError,
    placement_warning,
)

# The coordinator's own admin routes. The assistant never re-implements them;
# it calls the same handlers the dashboard calls, on the Mac that owns rank 0.
PLAN_PATH = "/admin/api/cluster/plan"
STAGING_PATH = "/admin/api/cluster/stage"
STAGING_STATUS_PATH = "/admin/api/cluster/stage/{job_id}"
ACTIVATION_PATH = "/admin/api/cluster/deployments"
PERFORMANCE_PROBE_PATH = "/admin/api/cluster/performance-probe"

# How often a running operation re-reads its progress.
POLL_SECONDS = 1.0
# Staging and activation are long by design (tens of GB, then a weight load).
REMOTE_TIMEOUT_SECONDS = 3600.0

WARNING_PLACEMENT_RECOMPUTED = "placement_recomputed"
WARNING_PERFORMANCE_UNMEASURED = "performance_unmeasured"
WARNING_PERFORMANCE_PROBE_FAILED = "performance_probe_failed"
WARNING_COMPUTE_PROBE_FAILED = "compute_probe_failed"
WARNING_CHAT_PROFILE_APPLIED = "chat_profile_applied"
WARNING_CHAT_PROFILE_SKIPPED = "chat_profile_skipped"
WARNING_TUNING_PRESET_APPLIED = "tuning_preset_applied"
WARNING_TUNING_PRESET_SKIPPED = "tuning_preset_skipped"

# The post-``ready`` measurement asks the coordinator for one real completion
# and reads the timing fields oMLX itself computed for it. It is deliberately
# not the readiness canary (two tokens cannot measure decode) and not the
# synthetic distributed probe (weight bytes per second are not token rates).
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
OBSERVE_PROMPT = (
    "Write one short sentence about why a quiet library is a good place to "
    "read. Reply with the sentence only."
)
# Enough completion tokens that decode has more than one interval to measure,
# and short enough that a thinking model cannot spend minutes on a probe.
OBSERVE_MAX_TOKENS = 96
OBSERVE_TIMEOUT_SECONDS = 300.0

# The generation settings a freshly started cluster needs to be usable for a
# normal question. Measured on this project's validation pair with
# GLM-5.3-Flash-oQ2e-mtp: the model's own defaults (temperature 1.0, no token
# cap) let one factual question think for 1149 tokens and 48 seconds, and a
# smaller cap truncates the answer entirely. These values answered the same
# question in 3-16 s with correct content.
#
# ``enable_thinking`` is deliberately absent: forcing it off at the model level
# behaved worse than leaving it to the model, and the per-request
# ``chat_template_kwargs`` toggle is what actually shortens the reasoning. The
# assistant does not own a chat request, so it does not fake that decision.
CHAT_PROFILE: dict[str, Any] = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 0,
    "max_tokens": 2048,
    "force_sampling": True,
}
ACTIVE_OPERATION_PATH = "/admin/api/cluster/assistant/operations/active"
# The cluster-wide lock (the apply/start design notes): the origin mirrors its operation onto
# the coordinator's own store through the same admin seam, so the coordinator
# can answer ``operations/active`` while a remote apply/start is working.
# Server-to-server only; the browser never calls it.
MIRROR_PATH = "/admin/api/cluster/assistant/operations/mirror"
MODELS_PATH = "/admin/api/models"
RELOAD_PATH = "/admin/api/reload"
DEVICES_PATH = "/api/cluster/devices"
MODEL_SETTINGS_PATH = "/admin/api/models/{model_id}/settings"
# oMLX's own Mac-wide settings route. Used for levers that are not per model —
# today the memory-guard tier, which decides the ceiling a Mac admits. It is
# written on every selected Mac before staging, because the launch preflight
# reads that Mac's *live* ceiling: a plan priced at `aggressive` and a Mac
# still guarding at `balanced` is a stage refused by the reserve difference.
GLOBAL_SETTINGS_PATH = "/admin/api/global-settings"
# oMLX's own manual load route: the single-node ("run on this Mac") flow asks
# this Mac's server to load the checkpoint into its engine pool. It is the same
# route the dashboard uses; the assistant never launches a process itself.
MODEL_LOAD_PATH = "/admin/api/models/{model_id}/load"
# oMLX's own manual unload route: a stopped local run frees this Mac's memory
# through the same route the dashboard's Unload button calls.
MODEL_UNLOAD_PATH = "/admin/api/models/{model_id}/unload"
# oMLX's own per-Mac delete route: it removes one model from that Mac's disk and
# refreshes its pool. The assistant relays to it instead of deleting anything
# itself, so the path a model lives in stays that Mac's business.
MODEL_DELETE_PATH = "/admin/api/hf/models/{model_name}"

logger = logging.getLogger(__name__)


class RemoteAdminError(RuntimeError):
    """The coordinator's admin route answered, but not with a result."""

    def __init__(self, status: int, detail: str) -> None:
        self.status = status
        self.detail = detail
        super().__init__(f"the coordinator answered {status}: {detail}"[:500])


@dataclass(frozen=True)
class PreparedDeployment:
    """The coordinator's signed plan for one deployment request.

    ``plan`` is the coordinator's own plan payload, including the
    ``placement_signature`` its activation will compare. ``deployment`` is the
    parsed ``ClusterDeployment`` when preparation happened in this process; it
    is ``None`` for a remote coordinator, where only the coordinator can build
    it.
    """

    plan: Mapping[str, Any]
    deployment: Any | None = None

    @property
    def placement_signature(self) -> str:
        signature = str(self.plan.get("placement_signature") or "").strip()
        if not signature:
            raise RemoteAdminError(
                502, "the coordinator returned a plan with no placement signature"
            )
        return signature


# ``target``, ``port``, ``path``, optional JSON body -> parsed JSON reply.
RemoteAdmin = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True)
class CoordinatorOps:
    """How one apply/start reaches the elected coordinator.

    Tests replace this whole object, so no test reaches SSH, a model or a Mac.
    The four synchronous calls run on a worker thread; ``health`` and
    ``activate`` are awaited, and their default implementations move the
    blocking oMLX calls into threads themselves.
    """

    local: bool
    target: str
    prepare: Callable[[Any], PreparedDeployment]
    stage: Callable[[PreparedDeployment, Any, str], Mapping[str, Any]]
    staging_status: Callable[[str], Mapping[str, Any] | None]
    health: Callable[[], Any]
    activate: Callable[[Any], Any]
    #: The coordinator's model list, and a per-model settings write. Both are
    #: optional so a test (or an older coordinator) can leave them out; the
    #: chat-profile step then simply does nothing.
    models: Callable[[], Any] | None = None
    write_model_settings: Callable[[str, Mapping[str, Any]], Any] | None = None
    #: The coordinator's own paired-device snapshot. Apply uses it to refuse a
    #: plan whose coordinator cannot reach a selected peer, before it stages.
    devices: Callable[[], Any] | None = None
    #: Re-read the coordinator's model list and per-model settings. Needed when
    #: a model was staged/copied outside the pool: the settings file exists but
    #: ``prepare`` would otherwise use the stale, resident settings. It unloads
    #: every loaded model, so apply/start only call it when the catalogue does
    #: not list the target model (``_reload_if_model_unlisted``).
    reload: Callable[[], Any] | None = None
    #: One real non-streaming completion on the coordinator, returning the
    #: OpenAI reply whose ``usage`` carries the measured timing fields. Optional
    #: so a test (or an older coordinator) can leave it out; the probe then
    #: records a warning instead of a rate.
    observe: Callable[[str], Mapping[str, Any]] | None = None
    #: One streaming completion whose final usage chunk carries TTFT/prefill.
    stream_observe: Callable[[str], Mapping[str, Any]] | None = None
    #: The synthetic distributed compute probe, run after staging. It returns
    #: the coordinator's profiles. Optional for tests/older coordinators.
    probe: Callable[[Any], Mapping[str, Any]] | None = None
    #: Mirror one operation lifecycle state onto the coordinator's own store
    #: (the cluster-wide lock, the apply/start design notes). The body carries the origin's
    #: operation ID, kind, coordinator, state and model/context for display.
    #: Remote coordinators only: a local coordinator shares this store, so
    #: ``store.start`` already refuses a second operation and the mirror stays
    #: unset. Optional so a test can leave it out.
    mirror: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None


# ---------------------------------------------------------------------------
# Reaching the coordinator
# ---------------------------------------------------------------------------


def coordinator_ops(
    plan: ApplyPlan, *, remote: RemoteAdmin | None = None
) -> CoordinatorOps:
    """The coordinator's control surface: local when it is this Mac, SSH otherwise.

    ``plan.coordinator.local`` is proven from measured addresses, never assumed.
    The remote variant always dials ``plan.coordinator.ssh`` — the elected Mac —
    and never the app origin.
    """

    if plan.coordinator.local:
        return _local_ops(plan)
    target = validate_ssh_target(plan.coordinator.ssh)
    return _remote_ops(target, plan.endpoint.port if plan.endpoint else 0, remote)


def coordinator_active_operation(
    plan: ApplyPlan, *, remote: RemoteAdmin | None = None
) -> str:
    """The operation the elected coordinator is already running, or ``""``.

    "One apply/start at a time" is enforced by each origin's own store, which
    says nothing about a *second* origin driving the same coordinator: both
    would activate, and the second replaces the first. The coordinator is the
    only Mac that can answer this, so the question is asked over the same admin
    seam the rest of apply/start uses.

    Unanswered means "cannot prove it is busy": the caller proceeds and the
    operation itself reports why the coordinator refused, instead of turning a
    diagnostic problem into a fake lock.
    """

    if plan.coordinator.local:
        return ""  # its own store is this store; ``store.start`` already refuses
    port = plan.endpoint.port if plan.endpoint else 0
    if not 1 <= int(port) <= 65535:
        return ""
    admin = remote or remote_admin
    try:
        reply: Any = admin(
            validate_ssh_target(plan.coordinator.ssh),
            port,
            ACTIVE_OPERATION_PATH,
            None,
            method="GET",
            timeout=30.0,
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


def _remote_ops(target: str, port: int, remote: RemoteAdmin | None) -> CoordinatorOps:
    admin = remote or remote_admin

    def prepare(request: Any) -> PreparedDeployment:
        return PreparedDeployment(
            plan=admin(target, port, PLAN_PATH, _plan_body(request))
        )

    def stage(
        prepared: PreparedDeployment, request: Any, source_host: str
    ) -> Mapping[str, Any]:
        snapshot = admin(
            target,
            port,
            STAGING_PATH,
            {
                "activation": request.model_dump(mode="json"),
                "parallel": 4,
            },
        )
        if not isinstance(snapshot, Mapping) or not snapshot.get("job_id"):
            raise RemoteAdminError(502, "the coordinator returned no staging job")
        return snapshot

    def staging_status(job_id: str) -> Mapping[str, Any] | None:
        reply = admin(
            target,
            port,
            STAGING_STATUS_PATH.format(job_id=job_id),
            None,
            method="GET",
        )
        return reply if isinstance(reply, Mapping) else None

    async def health() -> Mapping[str, Any]:
        reachable = await asyncio.to_thread(probe_peer, target)
        return {
            "ok": reachable,
            "message": (
                "the elected coordinator answers"
                if reachable
                else f"{target} did not answer; open oMLX on the coordinator and "
                "retry the coordinator"
            ),
        }

    async def activate(request: Any) -> Mapping[str, Any]:
        return await asyncio.to_thread(
            admin,
            target,
            port,
            ACTIVATION_PATH,
            request.model_dump(mode="json"),
            timeout=REMOTE_TIMEOUT_SECONDS,
        )

    return CoordinatorOps(
        local=False,
        target=target,
        prepare=prepare,
        stage=stage,
        staging_status=staging_status,
        health=health,
        activate=activate,
        models=lambda: admin(target, port, MODELS_PATH, None, method="GET"),
        write_model_settings=lambda model_id, patch: admin(
            target,
            port,
            MODEL_SETTINGS_PATH.format(model_id=model_id),
            dict(patch),
            method="PUT",
        ),
        devices=lambda: admin(target, port, DEVICES_PATH, None, method="GET"),
        reload=lambda: admin(target, port, RELOAD_PATH, {}, method="POST"),
        observe=lambda model_id: admin(
            target,
            port,
            CHAT_COMPLETIONS_PATH,
            _observe_body(model_id),
            timeout=OBSERVE_TIMEOUT_SECONDS,
        ),
        probe=lambda request: admin(
            target,
            port,
            PERFORMANCE_PROBE_PATH,
            request.model_dump(mode="json"),
            timeout=REMOTE_TIMEOUT_SECONDS,
        ),
        stream_observe=lambda model_id: remote_stream_observe(
            target, port, model_id
        ),
        mirror=lambda body: admin(target, port, MIRROR_PATH, dict(body)),
    )


def _local_ops(plan: ApplyPlan) -> CoordinatorOps:
    """The coordinator is this Mac: use the pipeline in this process."""

    from .. import routes as cluster_routes

    def prepare(request: Any) -> PreparedDeployment:
        _, deployment, signed_plan = cluster_routes.prepare_cluster_deployment(request)
        return PreparedDeployment(plan=signed_plan, deployment=deployment)

    def stage(
        prepared: PreparedDeployment, request: Any, source_host: str
    ) -> Mapping[str, Any]:
        if prepared.deployment is None:
            raise RemoteAdminError(500, "the local deployment was not prepared")
        return cluster_routes.start_cluster_staging(
            prepared.deployment,
            source_host=source_host,
            parallel=4,
        )

    def staging_status(job_id: str) -> Mapping[str, Any] | None:
        return cluster_routes.read_cluster_staging_job(job_id)

    async def health() -> Mapping[str, Any]:
        hosts = {
            rank: (
                node_id,
                "127.0.0.1" if rank == 0 else plan.targets[node_id].ssh,
            )
            for rank, node_id in enumerate(plan.rank_order)
        }

        def check() -> Mapping[str, Any]:
            peers = check_peers(hosts)
            raise_if_peer_lost(peers)
            return {"ok": True, "message": "every rank answers"}

        try:
            return await asyncio.to_thread(check)
        except PeerLostError as exc:
            return {"ok": False, "message": str(exc)}

    async def activate(request: Any) -> Mapping[str, Any]:
        # The route handler *is* the activation pipeline; calling it directly
        # keeps the engine pool on the server's event loop.
        report: Mapping[str, Any] = await cluster_routes.activate_cluster_deployment(
            request
        )
        return report

    return CoordinatorOps(
        local=True,
        target="127.0.0.1",
        prepare=prepare,
        stage=stage,
        staging_status=staging_status,
        health=health,
        activate=activate,
        models=lambda: _local_admin(plan.endpoint.port if plan.endpoint else 0, MODELS_PATH),
        write_model_settings=lambda model_id, patch: _local_admin(
            plan.endpoint.port if plan.endpoint else 0,
            MODEL_SETTINGS_PATH.format(model_id=model_id),
            dict(patch),
            method="PUT",
        ),
        devices=lambda: _local_admin(
            plan.endpoint.port if plan.endpoint else 0,
            DEVICES_PATH,
            None,
            method="GET",
        ),
        reload=lambda: _local_admin(
            plan.endpoint.port if plan.endpoint else 0,
            RELOAD_PATH,
            {},
            method="POST",
        ),
        observe=lambda model_id: _local_admin(
            plan.endpoint.port if plan.endpoint else 0,
            CHAT_COMPLETIONS_PATH,
            _observe_body(model_id),
            method="POST",
            timeout=OBSERVE_TIMEOUT_SECONDS,
        ),
        probe=lambda request: _local_performance_probe(request),
        stream_observe=lambda model_id: _local_stream_observe(
            plan.endpoint.port if plan.endpoint else 0, model_id
        ),
        # No mirror: the coordinator *is* this Mac, so the operation record
        # already lives in the store ``operations/active`` reads, and
        # ``store.start`` refuses a second apply/start by itself.
    )


def _local_performance_probe(request: Any) -> Mapping[str, Any]:
    """Run the synthetic compute probe in-process on this coordinator."""

    from .. import routes as cluster_routes

    _, deployment, _ = cluster_routes.prepare_cluster_deployment(request)
    return cluster_routes.run_cluster_performance_probe(deployment)


def _plan_body(request: Any) -> dict[str, Any]:
    """The ``ClusterPlanRequest`` the coordinator's ``/plan`` route expects.

    Derived from the deployment request so both sides plan the same thing: the
    coordinator re-runs its own planner and signs the placement it would launch.
    """

    return {
        "model_path": request.model_path,
        "model_source": request.model_source,
        "nodes": [node.model_dump(mode="json") for node in request.nodes],
        "execution_profile": request.execution_profile,
        "allocation": request.allocation,
        "tensor_parallel_size": request.tensor_parallel_size,
        "target_context_tokens": request.target_context_tokens,
        "path_map": request.path_map,
        "mtp": request.mtp,
    }


_REMOTE_ADMIN_SNIPPET = (
    "import json,sys\n"
    "import urllib.error,urllib.request\n"
    "from omlx.settings import get_settings,init_settings\n"
    "payload=json.loads(sys.argv[1])\n"
    "body=payload.get('body')\n"
    "data=json.dumps(body).encode('utf-8') if body is not None else None\n"
    "url='http://127.0.0.1:%d%s'%(int(payload['port']),payload['path'])\n"
    "request=urllib.request.Request(url,data=data,method=payload.get('method','POST'))\n"
    "request.add_header('Content-Type','application/json')\n"
    # This process is a bare ``python -c`` on the coordinator, not that Mac's
    # server, so nothing has loaded its settings yet. Reading the settings
    # object without that step raised "Settings not initialized" and made every
    # remote coordinator call fail; load them the documented way instead. The
    # key still never leaves the coordinator.
    "try:\n"
    "    key=getattr(get_settings().auth,'api_key',None)\n"
    "except RuntimeError:\n"
    "    key=getattr(init_settings().auth,'api_key',None)\n"
    "if key:\n"
    "    request.add_header('Authorization','Bearer '+key)\n"
    "try:\n"
    "    with urllib.request.urlopen(request,timeout=float(payload['timeout'])) as reply:\n"
    "        print(json.dumps({'ok':True,'status':getattr(reply,'status',200),"
    "'body':json.loads(reply.read().decode('utf-8') or '{}')}))\n"
    "except urllib.error.HTTPError as exc:\n"
    "    detail=exc.read().decode('utf-8','replace')\n"
    "    print(json.dumps({'ok':False,'status':exc.code,'detail':detail[:1000]}))\n"
    # A closed coordinator (or a refused loopback port) never answers, so it
    # cannot report an HTTP status. Without this the snippet died with a
    # traceback and the origin could only say "apply_failed" + diagnostics,
    # which is the wrong next step for "open oMLX on that Mac". Status 0 means
    # "no answer" and maps to retry_coordinator.
    "except (urllib.error.URLError,OSError) as exc:\n"
    "    print(json.dumps({'ok':False,'status':0,"
    "'detail':('the coordinator did not answer: '+str(exc))[:1000]}))\n"
)


def remote_admin(
    target: str,
    port: int,
    path: str,
    body: Mapping[str, Any] | None,
    *,
    method: str = "POST",
    timeout: float = 120.0,
) -> Mapping[str, Any]:
    """Call one admin route on the coordinator through the existing SSH seam.

    A local process on the coordinator authenticates its own loopback call with
    the key that Mac already stores; the origin never sees it and never stores
    it. That is why no new credential is exchanged by pairing.
    """

    if not 1 <= int(port) <= 65535:
        raise RemoteAdminError(502, "the coordinator's admin port is not known")
    reply = run_remote_python(
        validate_ssh_target(target),
        _REMOTE_ADMIN_SNIPPET,
        json.dumps(
            {
                "port": int(port),
                "path": path,
                "method": method,
                "body": None if body is None else dict(body),
                "timeout": float(timeout),
            },
            sort_keys=True,
        ),
        description=f"reach the coordinator's admin API at {path}",
        python_executable=DEFAULT_REMOTE_PYTHON,
        timeout=float(timeout) + 60.0,
    )
    if not isinstance(reply, Mapping):
        raise RemoteAdminError(502, "the coordinator answered with no JSON")
    if not reply.get("ok"):
        raw_status = reply.get("status")
        status = int(raw_status) if isinstance(raw_status, int) else 502
        raise RemoteAdminError(
            status,
            str(reply.get("detail") or "the coordinator refused the call"),
        )
    body_out = reply.get("body")
    return body_out if isinstance(body_out, Mapping) else {}


def delete_model_on_mac(
    *,
    port: int,
    model_name: str,
    target: str = "",
    timeout: float = 120.0,
) -> Mapping[str, Any]:
    """Delete one model on a measured Mac through the same admin seam.

    This is the free-space affordance's one write, and it is a relay: the card
    asks for a model it saw in that Mac's measured inventory, and the Mac's own
    ``DELETE /admin/api/hf/models/{name}`` does the deleting and the pool
    refresh. An empty ``target`` means this Mac (loopback with the key it
    stores); otherwise the coordinator story applies unchanged — a bare
    interpreter on that Mac authenticates its own loopback call, so the key
    never leaves the Mac and pairing exchanges no new credential.

    ``model_name`` is the folder name of the model on that Mac, which is what
    oMLX's route resolves. A name that Mac cannot resolve comes back as its own
    404, never as a silent success.
    """

    path = MODEL_DELETE_PATH.format(
        model_name=quote(str(model_name).strip(), safe="")
    )
    if target:
        return remote_admin(
            validate_ssh_target(target),
            port,
            path,
            None,
            method="DELETE",
            timeout=timeout,
        )
    try:
        return _local_admin(port, path, method="DELETE", timeout=timeout)
    except RemoteAdminError:
        raise
    except OSError as exc:  # a refused loopback port is "no answer", not a crash
        raise RemoteAdminError(0, f"this Mac did not answer: {_redacted(exc)}") from exc


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------


def run_apply_job(
    store: OperationStore,
    operation_id: str,
    plan: ApplyPlan,
    ops: CoordinatorOps,
    *,
    poll_seconds: float = POLL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    performance_store: Any | None = None,
    pre_warnings: Sequence[OperationWarning] | None = None,
) -> None:
    """Prepare and stage the model. Inference is never started here."""

    coordinator = plan.coordinator.node_id
    try:
        # The cluster-wide lock: register the running operation on the
        # coordinator before any staging work (remote coordinators only).
        refusal = _mirror_running(store, operation_id, ops, STATE_APPLYING)
        if refusal is not None:
            _fail(
                store,
                operation_id,
                OPERATION_IN_PROGRESS_CODE,
                f"operation {refusal or 'another origin'} is running on "
                f"{coordinator}; poll it instead of starting a second one",
                ACTION_STATUS,
            )
            return
        _advance(
            store,
            operation_id,
            STATE_APPLYING,
            OperationProgress(
                phase="staging",
                message=f"writing the approved memory policy on {len(plan.rank_order)} Macs",
                nodes_total=len(plan.rank_order),
            ),
            warnings=pre_warnings,
        )
        pairing = _coordinator_pairing_failure(ops, plan)
        if pairing:
            _fail(store, operation_id, "node_not_paired", pairing, ACTION_PAIR_MAC)
            return
        _write_tier_settings(plan, plan.rank_order)
        _write_policy_settings(plan, [plan.coordinator.node_id])
        # A checkpoint copied/staged outside the coordinator's pool is on
        # disk but not in its in-memory catalogue yet; the settings file is
        # written, and prepare must see it. Re-read only then — a reload
        # unloads every loaded model, including a chat the user is running.
        _reload_if_model_unlisted(ops, plan.inputs.model.model_path)
        _advance(
            store,
            operation_id,
            STATE_APPLYING,
            OperationProgress(
                phase="staging",
                message=f"preparing the deployment on {coordinator}",
                nodes_total=len(plan.rank_order),
            ),
        )
        prepared = ops.prepare(plan.request)
        signed = plan.request.model_copy(
            update={"approved_placement": prepared.placement_signature}
        )
        warnings = list(plan.warnings)
        placement = placement_warning(plan.election, prepared.plan)
        if placement is not None:
            warnings.append(placement)
        _advance(
            store,
            operation_id,
            STATE_APPLYING,
            OperationProgress(
                phase="staging",
                message=f"staging the model through {coordinator}",
                nodes_total=len(plan.rank_order),
            ),
            placement_signature=prepared.placement_signature,
            warnings=warnings,
        )
        snapshot = ops.stage(prepared, signed, plan.source_host)
        job_id = str(snapshot.get("job_id") or "")
        if not job_id:
            raise RemoteAdminError(502, "the coordinator returned no staging job")
        _poll_staging(
            store,
            operation_id,
            ops,
            job_id,
            plan,
            warnings,
            poll_seconds,
            sleep,
            signed=signed,
            performance_store=performance_store,
        )
    except ProvisioningError as exc:
        _fail(store, operation_id, exc.code, exc.message, exc.action)
    except RemoteAdminError as exc:
        _fail(
            store,
            operation_id,
            "staging_refused",
            _coordinator_message(exc, plan),
            (
                ACTION_RETRY_COORDINATOR
                if exc.status in COORDINATOR_REFUSALS
                else ACTION_APPLY
            ),
        )
    except Exception as exc:  # noqa: BLE001 - a job may never stay applying
        _fail(
            store,
            operation_id,
            "apply_failed",
            _redacted(exc),
            ACTION_DIAGNOSTICS,
        )
    finally:
        # Every terminal path (applied or failed) releases the coordinator's
        # lock marker; the release is best-effort and never fails the job.
        _mirror_terminal(store, operation_id, ops)


def _coordinator_pairing_failure(
    ops: CoordinatorOps, plan: ApplyPlan
) -> str:
    """Why the elected coordinator cannot launch every selected peer, or "".

    The origin's device registry only proves that *it* is paired with each
    selected Mac. The coordinator performs staging and launch itself, so it
    must also know every selected peer. A Studio coordinator paired only with
    the hub fails later with ``staging_failed``; ask the coordinator for its
    own snapshot first and turn that into the documented ``pair_mac`` action.
    """

    if ops.devices is None:
        return ""
    payload = ops.devices()
    paired = payload.get("paired") if isinstance(payload, Mapping) else None
    if not isinstance(paired, list):
        return ""
    paired_ids = {
        str(item.get("node_id") or "")
        for item in paired
        if isinstance(item, Mapping) and item.get("node_id")
    }
    missing = [
        node_id
        for node_id in plan.rank_order
        if node_id != plan.coordinator.node_id and node_id not in paired_ids
    ]
    if not missing:
        return ""
    names = ", ".join(missing)
    return (
        f"{names} is not paired with coordinator {plan.coordinator.node_id}; "
        "pair it with that Mac and apply again"
    )


def _write_tier_settings(plan: ApplyPlan, node_ids: Sequence[str]) -> None:
    """Write the plan's memory-guard tier on the given selected Macs, or fail.

    The plan prices every stage with the tier it shows the user
    (``election.budget_for(..., tier=…)``) and the rank processes are launched
    with that tier as a flag, but the Mac's own server — the one whose live
    ceiling the launch preflight reads — kept whatever tier was last saved. A
    plan that raised the tier to `aggressive` then met a guard still reserving
    `balanced`: rank 1 of a clean apply was refused at "needs 40.9 GiB but can
    admit 40.4 GiB right now", which is exactly the reserve difference.

    So the lever is applied where it is priced, on every selected Mac, and it
    runs **before** staging because it changes the ceiling the start preflight
    will read. A Mac that cannot be written fails apply with one action: a
    half-raised cluster must never start.
    """

    policy = plan.inputs.resource_policy
    tier = "" if policy is None else str(policy.memory_guard_tier or "").strip()
    if not tier or plan.global_settings_writer is None:
        # No tier was chosen, so each Mac keeps its own saved tier and the
        # election priced it that way. Nothing to write, nothing to claim.
        return
    patch = {"memory_guard_tier": tier}
    for node_id in node_ids:
        target = plan.target_for(node_id)
        try:
            plan.global_settings_writer(target, patch)
        except ProvisioningError:
            raise
        except Exception as exc:
            raise ProvisioningError(
                "policy_settings_failed",
                f"the approved memory-guard tier ({tier}) could not be written "
                f"on {node_id}: {_redacted(exc)}",
                ACTION_DIAGNOSTICS,
            ) from exc


def _write_policy_settings(plan: ApplyPlan, node_ids: Sequence[str] | None = None) -> None:
    """Write the approved levers on the given selected Macs, or fail apply.

    The coordinator is written **before** ``prepare`` because its ``/plan``
    reads those settings to re-derive the placement. A worker that does not
    have the model yet cannot be written through its model-settings route
    (``404 Model not found``), so workers are written **after** staging, when
    the copy exists. Either way a failure fails the operation with one action;
    a mixed policy is never started. Excluded Macs are never written to.
    """

    policy = plan.inputs.resource_policy
    if policy is None or plan.settings_writer is None:
        return
    patch = policy_model_settings(policy)
    if not patch:
        return
    model_path = plan.inputs.model.model_path
    for node_id in (node_ids if node_ids is not None else plan.rank_order):
        target = plan.target_for(node_id)
        try:
            plan.settings_writer(target, model_path, patch)
        except ProvisioningError:
            raise
        except Exception as exc:
            raise ProvisioningError(
                "policy_settings_failed",
                f"the approved memory policy could not be written on {node_id}: "
                f"{_redacted(exc)}",
                ACTION_DIAGNOSTICS,
            ) from exc


def _poll_staging(
    store: OperationStore,
    operation_id: str,
    ops: CoordinatorOps,
    job_id: str,
    plan: ApplyPlan,
    warnings: Sequence[OperationWarning],
    poll_seconds: float,
    sleep: Callable[[float], None],
    *,
    signed: Any | None = None,
    performance_store: Any | None = None,
    deadline_seconds: float = 3600.0,
) -> None:
    # A staging job that never completes (or one the coordinator has lost, so
    # every status read is empty) must not spin forever: bound the total wait
    # and fail with one action (B13). Elapsed is accumulated from the injected
    # sleep so a fake clock drives it in tests.
    elapsed = 0.0
    while True:
        snapshot = ops.staging_status(job_id)
        status = str((snapshot or {}).get("status") or "")
        progress = _staging_progress(snapshot, plan)
        if status == "failed":
            failed = sorted(
                node_id
                for node_id, node in ((snapshot or {}).get("nodes") or {}).items()
                if isinstance(node, Mapping) and node.get("status") == "failed"
            )
            _fail(
                store,
                operation_id,
                "staging_failed",
                "the model could not be copied to "
                + (", ".join(failed) or plan.coordinator.node_id),
                ACTION_APPLY,
            )
            return
        if status == "completed":
            # A worker that received the model through staging only now has it
            # in its own catalogue, so its model-settings route can accept the
            # approved levers. Write them before the operation is `applied`;
            # a failure must not leave a half-written policy to start.
            try:
                _write_policy_settings(
                    plan,
                    [
                        node_id
                        for node_id in plan.rank_order
                        if node_id != plan.coordinator.node_id
                    ],
                )
            except ProvisioningError as exc:
                _fail(store, operation_id, exc.code, exc.message, exc.action)
                return
            # The synthetic compute probe runs on the coordinator after the
            # model is staged and never gates the operation. Its profiles are
            # used by the *next* measurement/plan, not this one.
            if signed is not None and performance_store is not None:
                _advance(
                    store,
                    operation_id,
                    STATE_APPLYING,
                    OperationProgress(
                        phase="staging",
                        message=f"measuring compute on {plan.coordinator.node_id}",
                        nodes_ready=progress.nodes_ready,
                        nodes_total=progress.nodes_total,
                    ),
                    warnings=warnings,
                )
                warnings = _run_compute_probe(
                    ops, signed, plan, performance_store, warnings
                )
            _advance(
                store,
                operation_id,
                STATE_APPLIED,
                OperationProgress(
                    phase="applied",
                    message=(
                        f"the model is staged on {plan.coordinator.node_id}; "
                        "nothing has been started"
                    ),
                    nodes_ready=progress.nodes_ready,
                    nodes_total=progress.nodes_total,
                ),
                warnings=warnings,
            )
            return
        if elapsed >= deadline_seconds:
            _fail(
                store,
                operation_id,
                "staging_timeout",
                f"staging job {job_id} did not finish within "
                f"{int(deadline_seconds)} s; the coordinator may have lost it",
                ACTION_APPLY,
            )
            return
        _advance(store, operation_id, STATE_APPLYING, progress)
        sleep(poll_seconds)
        elapsed += poll_seconds


def _run_compute_probe(
    ops: CoordinatorOps,
    request: Any,
    plan: ApplyPlan,
    performance_store: Any,
    warnings: Sequence[OperationWarning],
) -> tuple[OperationWarning, ...]:
    """Run the synthetic probe after staging; record a warning on failure.

    It never gates apply or start, and its profiles belong to the *next* plan.
    A failed probe leaves the existing stored profiles alone.
    """

    result = list(warnings)
    if ops.probe is None:
        return tuple(result)
    try:
        reply = ops.probe(request)
    except Exception as exc:  # a probe is calibration, never a gate
        result.append(
            OperationWarning(
                code=WARNING_COMPUTE_PROBE_FAILED,
                node_id=plan.coordinator.node_id,
                message=(
                    "the model is staged; the synthetic compute probe did not "
                    f"answer, so this apply stores no new compute profiles: "
                    f"{_redacted(exc)}"
                ),
            )
        )
        return tuple(result)
    profiles = reply.get("profiles") if isinstance(reply, Mapping) else None
    if not isinstance(profiles, list) or not profiles:
        result.append(
            OperationWarning(
                code=WARNING_COMPUTE_PROBE_FAILED,
                node_id=plan.coordinator.node_id,
                message=(
                    "the synthetic compute probe answered without profiles; "
                    "the next plan keeps using the previously measured compute"
                ),
            )
        )
        return tuple(result)
    try:
        performance_store.save(
            profiles,
            deployment_id=plan.election.plan_signature,
            world_size=len(plan.rank_order),
            backend=str(reply.get("backend") or "ring"),
        )
    except Exception as exc:
        result.append(
            OperationWarning(
                code=WARNING_COMPUTE_PROBE_FAILED,
                node_id=plan.coordinator.node_id,
                message=(
                    "the synthetic compute probe measured this deployment, but "
                    f"its profiles could not be stored: {_redacted(exc)}"
                ),
            )
        )
    return tuple(result)


def _staging_progress(
    snapshot: Mapping[str, Any] | None, plan: ApplyPlan
) -> OperationProgress:
    nodes = (snapshot or {}).get("nodes") or {}
    total = len(plan.rank_order) or len(nodes)
    ready = sum(
        1
        for node in nodes.values()
        if isinstance(node, Mapping) and node.get("status") == "ready"
    )
    return OperationProgress(
        phase="staging",
        message=f"staged {ready} of {total} Macs",
        nodes_ready=ready,
        nodes_total=total,
    )


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------


async def run_start_job(
    store: OperationStore,
    operation_id: str,
    request: Any,
    ops: CoordinatorOps,
    *,
    expected_placement: str,
    poll_seconds: float = POLL_SECONDS,
    probe_enabled: bool = False,
    preset: Mapping[str, Any] | None = None,
) -> None:
    """Health, activation, load, readiness canary — then ``endpoint.ready``."""

    record = store.get(operation_id)
    coordinator = "" if record is None else record.coordinator_node
    try:
        # The cluster-wide lock: register the running operation on the
        # coordinator before activation (remote coordinators only).
        refusal = await asyncio.to_thread(
            _mirror_running, store, operation_id, ops, STATE_STARTING
        )
        if refusal is not None:
            _fail(
                store,
                operation_id,
                OPERATION_IN_PROGRESS_CODE,
                f"operation {refusal or 'another origin'} is running on "
                f"{coordinator}; poll it instead of starting a second one",
                ACTION_STATUS,
            )
            return
        _advance(
            store,
            operation_id,
            STATE_STARTING,
            OperationProgress(
                phase="starting",
                message=f"checking that {coordinator} and its Macs answer",
            ),
        )
        # Replay the saved tuning preset so the launch uses the tuned MTP depth
        # (and sampler) instead of re-tuning. Best-effort; one warning records
        # the outcome and a failure never blocks the start.
        preset_warning = await apply_tuning_preset(
            ops, str(getattr(request, "model_path", "") or ""), preset
        )
        # The model may have been staged/copied after the apply's reload;
        # re-read it before the placement is prepared and launched, but only
        # when the catalogue does not list it yet (a reload unloads every
        # loaded model on the coordinator).
        await asyncio.to_thread(
            _reload_if_model_unlisted,
            ops,
            str(getattr(request, "model_path", "") or ""),
        )
        prepared = await asyncio.to_thread(ops.prepare, request)
        if prepared.placement_signature != expected_placement:
            _fail(
                store,
                operation_id,
                "placement_changed",
                "the coordinator would now launch a different placement than the "
                "one apply staged; plan and apply again",
                ACTION_PLAN,
            )
            return
        signed = request.model_copy(
            update={"approved_placement": prepared.placement_signature}
        )
        health = await ops.health()
        if not health.get("ok"):
            _fail(
                store,
                operation_id,
                "coordinator_unreachable",
                str(health.get("message") or "the elected coordinator did not answer"),
                ACTION_RETRY_COORDINATOR,
            )
            return
        _advance(
            store,
            operation_id,
            STATE_LOADING,
            OperationProgress(
                phase="loading",
                message=f"{coordinator} is loading the model and proving readiness",
            ),
        )
        report = await ops.activate(signed)
        if not _readiness_proven(report):
            _fail(
                store,
                operation_id,
                "readiness_unproven",
                "the coordinator reported no successful readiness completion, so "
                "the endpoint is not ready",
                ACTION_DIAGNOSTICS,
            )
            return
        record = store.get(operation_id)
        endpoint = None if record is None else record.endpoint
        deployment_id = _deployment_id_from_report(report)
        ready_endpoint = (
            None
            if endpoint is None
            else OperationEndpoint(
                node=endpoint.node,
                host=endpoint.host,
                port=endpoint.port,
                ready=True,
            )
        )
        warnings = list(record.warnings if record is not None else ())
        if not probe_enabled:
            # With the probe enabled the measured block (or its failure
            # warning) is the authority; claiming "unmeasured" next to a
            # filled observed block contradicts the measurement.
            warnings.append(
                OperationWarning(
                    code=WARNING_PERFORMANCE_UNMEASURED,
                    node_id=coordinator,
                    message=(
                        "the cluster is ready and no throughput was measured; this "
                        "operation claims no rate"
                    ),
                )
            )
        profile_warning = await ensure_chat_profile(
            ops, str(getattr(request, "model_path", "") or "")
        )
        if profile_warning is not None:
            warnings.append(profile_warning)
        if preset_warning is not None:
            warnings.append(preset_warning)
        _advance(
            store,
            operation_id,
            STATE_READY,
            OperationProgress(
                phase="ready",
                message=f"the endpoint on {coordinator} answered one completion",
            ),
            endpoint=ready_endpoint,
            warnings=warnings,
            deployment_id=deployment_id,
        )
        # The probe runs after the operation the poll returns is already
        # ``ready``: it can add one measured block or one warning, and it can
        # never unready the cluster or fail the job. Keeping ``auto_tune`` off
        # is what keeps the approved placement the launched one.
        if probe_enabled:
            await observe_performance(
                store,
                operation_id,
                ops,
                str(getattr(request, "model_path", "") or ""),
            )
            # The non-streaming completion reported decode; one streaming
            # completion reads the final usage chunk's TTFT/prefill. A missing
            # streaming path or a failed chunk is a warning, never an unready.
            await observe_streaming(
                store,
                operation_id,
                ops,
                str(getattr(request, "model_path", "") or ""),
            )
    except RemoteAdminError as exc:
        message = _coordinator_message(exc, None)
        action = ACTION_RETRY_COORDINATOR
        if any(hint in message for hint in _MEMORY_PRESSURE_HINTS):
            # "does not fit ... right now" is memory pressure on the selected
            # Macs (a previous deployment or an app still holds it), not a
            # coordinator that is down. Retrying without freeing memory just
            # fails again, so name the cause and the fix instead.
            action = ACTION_ADJUST_PLAN
            message = _memory_refusal_detail(message)
        _fail(store, operation_id, "coordinator_start_failed", message, action)
    except ProvisioningError as exc:
        _fail(store, operation_id, exc.code, exc.message, exc.action)
    except Exception as exc:  # noqa: BLE001 - a job may never stay starting
        message = _redacted(exc)
        action = ACTION_RETRY_COORDINATOR
        if any(hint in message for hint in _MEMORY_PRESSURE_HINTS):
            # Measured live 2026-11-16: the coordinator answered "503: Model
            # does not fit the current per-node memory limits ... can admit
            # 45.2 GiB" and the card then told the user oMLX "is not reachable"
            # and to open it — on a Mac that was running the whole time.
            # Nothing was started, and the fix is the plan or the memory, so
            # the one action is the model/context step.
            action = ACTION_ADJUST_PLAN
            message = _memory_refusal_detail(message)
        _fail(store, operation_id, "start_failed", message, action)
    finally:
        # Every terminal path (ready or failed) releases the coordinator's
        # lock marker; the release is best-effort and never fails the job.
        await asyncio.to_thread(_mirror_terminal, store, operation_id, ops)


def _deployment_id_from_report(report: Mapping[str, Any]) -> str:
    """The deployment ID the coordinator assigned to the activation."""

    deployment = report.get("deployment") if isinstance(report, Mapping) else None
    if not isinstance(deployment, Mapping):
        return ""
    return str(deployment.get("deployment_id") or "").strip()


def _readiness_proven(report: Mapping[str, Any]) -> bool:
    """One completion with every rank resident, or the endpoint is not ready."""

    if not isinstance(report, Mapping) or not report.get("ok"):
        return False
    readiness = report.get("readiness")
    if not isinstance(readiness, Mapping):
        return False
    if readiness.get("state") != "ready" or not readiness.get("canary_passed"):
        return False
    if readiness.get("all_ranks_ready") is False:
        return False
    tokens = readiness.get("canary_completion_tokens")
    return isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 1


def start_inputs(record: Any, *, model: Any) -> PlanInputs:
    """Rebuild the plan-time inputs a start needs from an applied record.

    ``plan`` is pure and did not persist its result, so the operation record is
    the only place the approved inputs survive. Re-electing from them is what
    re-proves the signature instead of trusting it.
    """

    return PlanInputs(
        measurement_id=record.measurement_id,
        plan_signature=record.plan_signature,
        model=model,
        context_tokens=record.context_tokens,
        execution_profile=record.execution_profile,
        rank0_extra_bytes=record.rank0_extra_bytes,
        mtp=record.mtp,
        coordinator_node=record.coordinator_node,
        rank_order=record.rank_order,
        microbatch_size=record.microbatch_size,
        target_context_tokens=_optional_int(
            record.policy.get("target_context_tokens")
        ),
        resource_policy=ResourcePolicy.from_dict(record.policy),
        two_mac_fallback=bool(getattr(record, "two_mac_fallback", False)),
        preferred_coordinator=str(
            getattr(record, "preferred_coordinator", "") or ""
        ),
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _advance(
    store: OperationStore,
    operation_id: str,
    state: str,
    progress: OperationProgress,
    *,
    endpoint: OperationEndpoint | None = None,
    warnings: Sequence[OperationWarning] | None = None,
    placement_signature: str | None = None,
    deployment_id: str | None = None,
) -> None:
    try:
        store.advance(
            operation_id,
            state,
            progress=progress,
            endpoint=endpoint,
            warnings=warnings,
            placement_signature=placement_signature,
            deployment_id=deployment_id,
        )
    except KeyError:
        # The record was pruned; the job still must not raise.
        return
    except OSError as exc:
        # A save failure keeps the true state in memory (the store no longer
        # rolls back terminal states); log and continue rather than crash the
        # job task and lose the operation (B6 follow-up).
        logger.warning(
            "could not persist operation %s advance: %s", operation_id, _redacted(exc)
        )


def _fail(
    store: OperationStore,
    operation_id: str,
    code: str,
    message: str,
    action: str,
) -> None:
    try:
        store.fail(operation_id, code=code, message=message, action=action)
    except KeyError:
        return
    except OSError as exc:
        # The failed state stays in memory even if the save failed; the status
        # poll still surfaces it. Never crash the job task over persistence.
        logger.warning(
            "could not persist operation %s failure: %s", operation_id, _redacted(exc)
        )


# ---------------------------------------------------------------------------
# The cluster-wide lock: mirror the operation onto the coordinator (§2.2)
# ---------------------------------------------------------------------------


def _mirror_body(
    store: OperationStore, operation_id: str, state: str
) -> dict[str, Any] | None:
    """The mirror payload, read from the origin's own record (one source)."""

    record = store.get(operation_id)
    if record is None:
        return None
    return {
        "operation_id": record.operation_id,
        "kind": record.kind,
        "coordinator_node": record.coordinator_node,
        "state": state,
        "model_path": record.model_path,
        "context_tokens": record.context_tokens,
    }


def _mirror_busy_id(exc: RemoteAdminError) -> str | None:
    """The running operation ID when the coordinator refused a busy mirror.

    The seam reports the coordinator's 409 body as text; only the documented
    ``operation_in_progress`` code is a lock refusal, never any other 409.
    """

    if exc.status != 409:
        return None
    try:
        payload = json.loads(exc.detail)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    if payload.get("code") != OPERATION_IN_PROGRESS_CODE:
        return None
    return str(payload.get("operation_id") or "")


def _mirror_running(
    store: OperationStore,
    operation_id: str,
    ops: CoordinatorOps,
    state: str,
) -> str | None:
    """Register the running operation on the coordinator before any work.

    Returns ``None`` when the mirror landed — or when there is nothing to
    mirror (a local coordinator shares this store; a test may leave the seam
    out). Returns the running operation's ID when the coordinator refused the
    mirror because another operation holds the cluster-wide lock; the caller
    fails with ``operation_in_progress``. Any other refusal propagates to the
    job's own error handling.
    """

    if ops.mirror is None:
        return None
    body = _mirror_body(store, operation_id, state)
    if body is None:
        return None
    try:
        ops.mirror(body)
    except RemoteAdminError as exc:
        busy_id = _mirror_busy_id(exc)
        if busy_id is not None:
            return busy_id
        if exc.status in _MIRROR_ROUTE_ABSENT:
            # An older coordinator build has no lock-mirror route. That is not
            # a running operation (the cluster-wide lock is simply unavailable
            # on this Mac), so the apply/start must not be blocked: a mixed
            # version cluster would otherwise be unable to deploy at all. The
            # absence is logged, never silent.
            logger.warning(
                "the coordinator %s has no operation-lock route (HTTP %s); "
                "the cluster-wide lock is unavailable for this operation",
                ops.target,
                exc.status,
            )
            return None
        raise
    return None


def _mirror_terminal(
    store: OperationStore, operation_id: str, ops: CoordinatorOps
) -> None:
    """Release the coordinator's lock marker — best-effort, never a gate.

    The operation already reached its terminal state in this store. A
    coordinator that cannot be reached for the release logs a warning; the
    stale-mirror TTL frees the lock eventually. It never fails an
    applied/ready cluster.
    """

    if ops.mirror is None:
        return
    record = store.get(operation_id)
    if record is None or not record.finished:
        return
    body = _mirror_body(store, operation_id, record.state)
    if body is None:
        return
    try:
        ops.mirror(body)
    except Exception as exc:  # noqa: BLE001 - a release is never a gate
        logger.warning(
            "could not release the operation lock on coordinator %s for %s: %s",
            record.coordinator_node,
            operation_id,
            _redacted(exc),
        )


#: Statuses that mean "this Mac's oMLX is not answering as a coordinator".
#: 0 is the transport answer the snippet reports for a closed app; 401/403 mean
#: it answered but is not configured for this cluster.
COORDINATOR_REFUSALS = (0, 401, 403)
#: Statuses that mean "this coordinator build has no lock-mirror route". The
#: cluster-wide lock is unavailable on an older Mac, but that must not block a
#: deploy (a mixed-version cluster is a supported reality, not an operation).
_MIRROR_ROUTE_ABSENT = (404, 405)
#: A coordinator refusing activation with these words is out of per-node memory
#: (a previous deployment/app still holds it), not unreachable. Retrying without
#: freeing memory cannot help, so the failure names the cause instead.
_MEMORY_PRESSURE_HINTS = ("does not fit", "per-node memory limits")


def _memory_refusal_detail(message: str) -> str:
    """The cause plus the one step that helps, for a memory refusal.

    Nothing was started, so "retry the coordinator" (its UI text says oMLX "is
    not reachable") is wrong — measured live 2026-11-16. Freeing memory or a
    smaller plan is the fix, and that is what the card's single action opens.
    """

    return (
        f"{message} No ranks were started. Free memory on the Mac named above "
        "(stop the previous cluster, close apps) or rebuild the plan with a "
        "smaller context, then Start again."
    )


def _coordinator_message(exc: RemoteAdminError, plan: ApplyPlan | None) -> str:
    where = "the elected coordinator" if plan is None else plan.coordinator.node_id
    if exc.status in COORDINATOR_REFUSALS:
        return (
            f"{where} did not answer as the coordinator; open oMLX on it, make "
            "sure its admin API key is set, and retry the coordinator"
        )
    return f"{where} did not accept the request: {exc.detail}"[:400]


def _redacted(exc: BaseException) -> str:
    """A short failure message that cannot carry a key or a whole command."""

    text = str(exc).strip() or exc.__class__.__name__
    return text[-300:]


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Chat profile: the sampling a freshly started cluster needs to be usable
# ---------------------------------------------------------------------------

_LOCAL_ADMIN_TIMEOUT = 30.0


def _local_admin(
    port: int,
    path: str,
    body: Mapping[str, Any] | None = None,
    *,
    method: str = "GET",
    timeout: float = _LOCAL_ADMIN_TIMEOUT,
) -> Any:
    """Call one admin route on this Mac's own server with the key it stores.

    The local coordinator *is* this process, so the same loopback contract the
    remote path uses applies here: the key never leaves the Mac and never
    crosses the assistant API. Callers run this off the event loop.
    """

    import urllib.error
    import urllib.request

    from ...settings import get_settings

    if not 1 <= int(port) <= 65535:
        raise RemoteAdminError(502, "the coordinator's admin port is not known")
    data = None if body is None else json.dumps(dict(body)).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{int(port)}{path}", data=data, method=method
    )
    request.add_header("Content-Type", "application/json")
    try:
        key = getattr(get_settings().auth, "api_key", None)
    except RuntimeError:  # a bare interpreter without loaded settings
        key = None
    if key:
        request.add_header("Authorization", "Bearer " + key)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            raw = reply.read().decode("utf-8", "replace")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RemoteAdminError(exc.code, detail[:1000]) from exc


def _model_entry(listing: Any, model_path: str) -> Mapping[str, Any] | None:
    """The coordinator's catalogue entry for the model this cluster launched."""

    entries = listing.get("models") if isinstance(listing, Mapping) else listing
    if not isinstance(entries, (list, tuple)):
        return None
    wanted = str(model_path or "").strip()
    tail = Path(wanted).name if wanted else ""
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        if wanted and str(entry.get("model_path") or "") == wanted:
            return entry
    for entry in entries:
        if isinstance(entry, Mapping) and tail and str(entry.get("id") or "") == tail:
            return entry
    return None


def _reload_if_model_unlisted(ops: CoordinatorOps, model_path: str) -> None:
    """Re-read the coordinator's catalogue only when it does not list the model.

    The reload unloads *every* loaded model on the coordinator, so it must not
    evict a chat model the user is running in parallel. A model already listed
    needs no re-read: its settings write went through the admin route and is
    already visible to prepare. Only a checkpoint staged/copied outside the
    pool — or a catalogue that cannot be read, which cannot prove the model is
    listed — keeps the re-read, so prepare still sees it.
    """

    if ops.reload is None:
        return
    listing = None
    if ops.models is not None:
        try:
            listing = ops.models()
        except Exception:  # noqa: BLE001 - unreadable is not "listed"
            listing = None
    if _model_entry(listing, model_path) is not None:
        return
    ops.reload()


def _observe_body(model_id: str, *, stream: bool = False) -> dict[str, Any]:
    """One harmless completion whose own timing fields are the measurement."""

    body: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": OBSERVE_PROMPT}],
        "max_tokens": OBSERVE_MAX_TOKENS,
        "temperature": 0.0,
        "stream": stream,
    }
    if stream:
        # OpenAI-compatible servers put the final usage (and oMLX's TTFT /
        # prompt rate) in the last chunk only when usage is requested.
        body["stream_options"] = {"include_usage": True}
    return body


def _local_stream_observe(
    port: int, model_id: str = "", *, body: Mapping[str, Any] | None = None
) -> Mapping[str, Any]:
    """Read the final usage chunk of one streaming completion on this Mac.

    The non-streaming path reports decode; TTFT and prefill live in the
    streaming chunk. This reads only oMLX's own usage fields and returns the
    last one seen; a missing chunk is ``{}`` (not a zero). ``body`` overrides
    the fixed observe request — the calibration runner sends its own prompt
    and token budget through the same loopback call and timeout.
    """

    import urllib.error
    import urllib.request

    from ...settings import get_settings

    if not 1 <= int(port) <= 65535:
        raise RemoteAdminError(502, "the coordinator's admin port is not known")
    request = urllib.request.Request(
        f"http://127.0.0.1:{int(port)}{CHAT_COMPLETIONS_PATH}",
        data=json.dumps(
            dict(body) if body is not None else _observe_body(model_id, stream=True)
        ).encode("utf-8"),
        method="POST",
    )
    request.add_header("Content-Type", "application/json")
    request.add_header("Accept", "text/event-stream")
    try:
        key = getattr(get_settings().auth, "api_key", None)
    except RuntimeError:
        key = None
    if key:
        request.add_header("Authorization", "Bearer " + key)
    usage: Mapping[str, Any] = {}
    try:
        with urllib.request.urlopen(
            request, timeout=OBSERVE_TIMEOUT_SECONDS
        ) as reply:
            for raw in reply:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                if isinstance(chunk, Mapping) and isinstance(
                    chunk.get("usage"), Mapping
                ):
                    usage = dict(chunk["usage"])
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RemoteAdminError(exc.code, detail[:1000]) from exc
    return {"usage": usage}


_REMOTE_STREAM_OBSERVE_SNIPPET = (
    "import json,sys\n"
    "import urllib.error,urllib.request\n"
    "from omlx.settings import get_settings,init_settings\n"
    "payload=json.loads(sys.argv[1])\n"
    # An explicit body (the calibration runner's) wins; otherwise the fixed
    # observe request is built from the prompt/budget fields as before.
    "body=json.dumps(payload.get('body') or {'model':payload['model'],"
    "'messages':[{'role':'user','content':payload['prompt']}],"
    "'max_tokens':int(payload['max_tokens']),'temperature':0.0,"
    "'stream':True,'stream_options':{'include_usage':True}}).encode('utf-8')\n"
    "url='http://127.0.0.1:%d/v1/chat/completions'%int(payload['port'])\n"
    "request=urllib.request.Request(url,data=body,method='POST')\n"
    "request.add_header('Content-Type','application/json')\n"
    "request.add_header('Accept','text/event-stream')\n"
    "try:\n"
    "    key=getattr(get_settings().auth,'api_key',None)\n"
    "except RuntimeError:\n"
    "    key=getattr(init_settings().auth,'api_key',None)\n"
    "if key:\n"
    "    request.add_header('Authorization','Bearer '+key)\n"
    "usage={}\n"
    "try:\n"
    "    with urllib.request.urlopen(request,timeout=float(payload['timeout'])) as reply:\n"
    "        for raw in reply:\n"
    "            line=raw.decode('utf-8','replace').strip()\n"
    "            if not line.startswith('data:'):\n"
    "                continue\n"
    "            data=line[5:].strip()\n"
    "            if data=='[DONE]':\n"
    "                break\n"
    "            try:\n"
    "                chunk=json.loads(data)\n"
    "            except ValueError:\n"
    "                continue\n"
    "            if isinstance(chunk,dict) and isinstance(chunk.get('usage'),dict):\n"
    "                usage=chunk['usage']\n"
    "    print(json.dumps({'ok':True,'usage':usage}))\n"
    "except urllib.error.HTTPError as exc:\n"
    "    detail=exc.read().decode('utf-8','replace')\n"
    "    print(json.dumps({'ok':False,'status':exc.code,'detail':detail[:1000]}))\n"
    "except (urllib.error.URLError,OSError) as exc:\n"
    "    print(json.dumps({'ok':False,'status':0,"
    "'detail':('the coordinator did not answer: '+str(exc))[:1000]}))\n"
)


def remote_stream_observe(
    target: str, port: int, model_id: str = "", *, body: Mapping[str, Any] | None = None
) -> Mapping[str, Any]:
    """Read one streaming completion's final usage on a remote coordinator.

    ``body`` overrides the fixed observe request when given: the calibration
    runner sends its own prompt and token budget through the same SSH seam,
    with the same timeout. The key is still read and used on the Mac itself.
    """

    if not 1 <= int(port) <= 65535:
        raise RemoteAdminError(502, "the coordinator's admin port is not known")
    payload: dict[str, Any] = {
        "port": int(port),
        "model": model_id,
        "prompt": OBSERVE_PROMPT,
        "max_tokens": OBSERVE_MAX_TOKENS,
        "timeout": OBSERVE_TIMEOUT_SECONDS,
    }
    if body is not None:
        payload["body"] = dict(body)
    reply = run_remote_python(
        validate_ssh_target(target),
        _REMOTE_STREAM_OBSERVE_SNIPPET,
        json.dumps(payload, sort_keys=True),
        description="measure TTFT/prefill on the coordinator",
        python_executable=DEFAULT_REMOTE_PYTHON,
        timeout=OBSERVE_TIMEOUT_SECONDS + 60.0,
    )
    if not isinstance(reply, Mapping):
        raise RemoteAdminError(502, "the coordinator answered with no JSON")
    if not reply.get("ok"):
        status = reply.get("status")
        raise RemoteAdminError(
            int(status) if isinstance(status, int) else 502,
            str(reply.get("detail") or "the coordinator refused the measurement"),
        )
    usage = reply.get("usage")
    return {"usage": dict(usage) if isinstance(usage, Mapping) else {}}


def stream_completion(
    *, port: int, body: Mapping[str, Any], target: str = ""
) -> Mapping[str, Any]:
    """One streaming completion on a measured Mac: body in, usage reply out.

    The per-node transport for ``calibration.calibrate_node``'s
    ``CompletionSeam`` contract: the same loopback/SSH seam the post-``ready``
    observe measurement uses, with the caller's completion body (produced by
    ``calibration.calibration_body``) instead of the fixed observe prompt. An
    empty ``target`` means this Mac — the in-process loopback call with the
    key it stores; any other Mac is reached through its own bare interpreter
    over the existing SSH seam, so no key leaves that Mac. The reply is the
    final usage chunk as ``{"usage": ...}``; a missing chunk is ``{}``, never
    a zero. The observe timeout bounds each completion.
    """

    if target:
        return remote_stream_observe(validate_ssh_target(target), port, body=body)
    return _local_stream_observe(port, body=body)


def _serving_model_id(listing: Any, model_path: str) -> str:
    """The coordinator's own serving id for the model this cluster launched."""

    entry = _model_entry(listing, model_path)
    if entry is None:
        return ""
    return str(entry.get("id") or "").strip()


def _positive_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def observed_from_reply(reply: Any) -> OperationObserved | None:
    """Read oMLX's own end-to-end timing fields, or ``None``.

    ``decode_tok_per_s`` is required: without it there is no rate at all. TTFT
    and prefill are recorded only when the coordinator's reply carries them —
    the distributed non-streaming path currently reports decode, while TTFT is
    populated on the chat streaming path. A missing value is "unknown", never
    zero, and a reply with no timing at all is "not measured", not a rate.
    """

    usage = reply.get("usage") if isinstance(reply, Mapping) else None
    if not isinstance(usage, Mapping):
        return None
    decode = _positive_float(usage.get("generation_tokens_per_second"))
    if decode is None:
        return None
    ttft_seconds = _positive_float(usage.get("time_to_first_token"))
    return OperationObserved(
        prefill_tok_per_s=_positive_float(usage.get("prompt_tokens_per_second")),
        decode_tok_per_s=decode,
        ttft_ms=None if ttft_seconds is None else ttft_seconds * 1000.0,
        measured_at=datetime.now(UTC).isoformat(),
    )


async def observe_performance(
    store: OperationStore,
    operation_id: str,
    ops: CoordinatorOps,
    model_path: str,
) -> OperationObserved | None:
    """Ask the ready coordinator for one real completion and record its timings.

    Runs on the coordinator through the same seam apply/start use: local
    in-process when this Mac is the coordinator, otherwise the coordinator's own
    bare interpreter, which authenticates its loopback call with the key that
    Mac stores. The origin never sees the key and no new credential is
    exchanged.

    Never raises and never moves the operation: a probe that cannot answer adds
    one warning to the already-``ready`` record.
    """

    if ops.observe is None:
        store.observe(
            operation_id,
            warning=OperationWarning(
                code=WARNING_PERFORMANCE_PROBE_FAILED,
                message=(
                    "the cluster is ready; this coordinator cannot serve a "
                    "measurement request, so no throughput is claimed"
                ),
            ),
        )
        return None
    try:
        listing = await asyncio.to_thread(ops.models) if ops.models else None
        model_id = _serving_model_id(listing, model_path)
        if not model_id:
            raise ValueError(
                "the coordinator's catalogue does not name the model this "
                "cluster launched"
            )
        reply = await asyncio.to_thread(ops.observe, model_id)
    except Exception as exc:  # a probe failure is a warning, never a failure
        store.observe(
            operation_id,
            warning=OperationWarning(
                code=WARNING_PERFORMANCE_PROBE_FAILED,
                message=(
                    "the cluster is ready, but the post-start measurement did "
                    f"not answer: {_redacted(exc)}"
                ),
            ),
        )
        return None
    observed = observed_from_reply(reply)
    if observed is None:
        store.observe(
            operation_id,
            warning=OperationWarning(
                code=WARNING_PERFORMANCE_PROBE_FAILED,
                message=(
                    "the cluster is ready, but the measurement completion "
                    "reported no timing, so no throughput is claimed"
                ),
            ),
        )
        return None
    store.observe(operation_id, observed=observed)
    return observed


async def observe_streaming(
    store: OperationStore,
    operation_id: str,
    ops: CoordinatorOps,
    model_path: str,
) -> OperationObserved | None:
    """Fill TTFT/prefill from one streaming completion's final usage chunk.

    The non-streaming measurement reports decode; this adds the two fields that
    only the streaming path carries. It never fails or unreadies the operation:
    an older coordinator that cannot stream, a failed request or a chunk with no
    timing adds nothing, and a real failure adds one warning while the existing
    decode block stays.
    """

    if ops.stream_observe is None:
        # An older coordinator (or a test) has no streaming path; the
        # non-streaming measurement is still valid and remains untouched.
        return None
    try:
        listing = await asyncio.to_thread(ops.models) if ops.models else None
        model_id = _serving_model_id(listing, model_path)
        if not model_id:
            raise ValueError(
                "the coordinator's catalogue does not name the model this "
                "cluster launched"
            )
        reply = await asyncio.to_thread(ops.stream_observe, model_id)
    except Exception as exc:
        store.observe(
            operation_id,
            warning=OperationWarning(
                code=WARNING_PERFORMANCE_PROBE_FAILED,
                message=(
                    "the cluster is ready and decode was measured; the "
                    f"streaming TTFT/prefill measurement did not answer: "
                    f"{_redacted(exc)}"
                ),
            ),
        )
        return None
    usage = reply.get("usage") if isinstance(reply, Mapping) else None
    if not isinstance(usage, Mapping):
        return None
    ttft_seconds = _positive_float(usage.get("time_to_first_token"))
    prefill = _positive_float(usage.get("prompt_tokens_per_second"))
    if ttft_seconds is None and prefill is None:
        return None
    record = store.get(operation_id)
    base = record.observed if record is not None else None
    merged = OperationObserved(
        prefill_tok_per_s=(
            prefill if prefill is not None else getattr(base, "prefill_tok_per_s", None)
        ),
        decode_tok_per_s=getattr(base, "decode_tok_per_s", None),
        ttft_ms=(
            ttft_seconds * 1000.0
            if ttft_seconds is not None
            else getattr(base, "ttft_ms", None)
        ),
        measured_at=(
            getattr(base, "measured_at", "")
            or datetime.now(UTC).isoformat()
        ),
    )
    store.observe(operation_id, observed=merged)
    return merged


async def apply_tuning_preset(
    ops: CoordinatorOps,
    model_path: str,
    preset: Mapping[str, Any] | None,
) -> OperationWarning | None:
    """Write a saved tuning preset's model settings before the launch.

    Tuning a large model is expensive (each config is a model setting, so a real
    sweep redeploys per config). The tuned knobs are persisted per (model, node
    config) and replayed here so a later Start uses them without re-tuning.

    Only knobs that are still unset on the coordinator's model are written: a
    value the user set by hand is never overwritten (the same rule the chat
    profile follows). The MTP depth has to be set *before* activation, so this
    runs before the launch, not after the cluster is ready. Every outcome — a
    write, or a refusal with its reason — is one warning, never silent. Nothing
    here can fail the start: a coordinator that cannot be reached simply stays
    without the preset.
    """

    if not preset:
        return None
    if ops.models is None or ops.write_model_settings is None:
        return None
    try:
        listing = await asyncio.to_thread(ops.models)
    except Exception as exc:  # the preset is a preference, not a gate
        return OperationWarning(
            code=WARNING_TUNING_PRESET_SKIPPED,
            message=(
                "the saved tuning preset was not applied: the model settings "
                f"could not be read: {_redacted(exc)}"
            ),
        )
    entry = _model_entry(listing, model_path)
    if entry is None:
        return None
    model_id = str(entry.get("id") or "").strip()
    if not model_id:
        return None
    # The tuning preset is the user's measured choice, so it is authoritative:
    # a leftover value from a sweep's last config must not win over the winner.
    # (The chat profile keeps the opposite rule — it only fills empty fields —
    # because it is a convenience default, not a tuned decision.)
    patch = {
        key: value for key, value in preset.items() if value is not None
    }
    if not patch:
        return OperationWarning(
            code=WARNING_TUNING_PRESET_SKIPPED,
            node_id=model_id,
            message="the saved tuning preset carries no value to apply",
        )
    try:
        await asyncio.to_thread(ops.write_model_settings, model_id, patch)
    except Exception as exc:
        return OperationWarning(
            code=WARNING_TUNING_PRESET_SKIPPED,
            node_id=model_id,
            message=(
                "the saved tuning preset could not be written: "
                f"{_redacted(exc)}"
            ),
        )
    return OperationWarning(
        code=WARNING_TUNING_PRESET_APPLIED,
        node_id=model_id,
        message=(
            "the saved tuning preset was applied before launch: "
            + ", ".join(sorted(patch))
        ),
    )


async def ensure_chat_profile(
    ops: CoordinatorOps, model_path: str
) -> OperationWarning | None:
    """Give the launched model sampling settings a chat can actually use.

    A cluster that starts with the model's own generation defaults lets one
    factual question think for a thousand tokens, and a token cap small enough
    to stop that truncates the answer instead. This writes the documented
    ``CHAT_PROFILE`` **only when the model has no sampling value of its own**,
    all-or-nothing: one user-set value keeps the whole profile untouched,
    because the settings write merges field by field. The outcome is recorded
    as a warning rather than being silent.

    Never fails the operation: the cluster is already ready at this point.
    """

    if ops.models is None or ops.write_model_settings is None:
        return None
    try:
        listing = await asyncio.to_thread(ops.models)
    except Exception as exc:  # the model list is evidence, not a gate
        return OperationWarning(
            code=WARNING_CHAT_PROFILE_SKIPPED,
            message=(
                "the cluster is ready; the model's chat settings could not be "
                f"read, so they were left untouched: {_redacted(exc)}"
            ),
        )
    entry = _model_entry(listing, model_path)
    if entry is None:
        return None
    # The real admin catalogue nests sampling under ``settings``
    # (``asdict(ModelSettings)``); there are no top-level sampling keys.
    # ``force_sampling`` defaults to False, so only a truthy value is a user
    # choice — an ``is not None`` check would see the default on every model.
    settings = entry.get("settings") or {}
    if any(
        settings.get(key) is not None
        for key in ("temperature", "top_p", "top_k", "max_tokens")
    ) or settings.get("force_sampling"):
        return None
    model_id = str(entry.get("id") or "").strip()
    if not model_id:
        return None
    try:
        await asyncio.to_thread(ops.write_model_settings, model_id, CHAT_PROFILE)
    except Exception as exc:
        return OperationWarning(
            code=WARNING_CHAT_PROFILE_SKIPPED,
            node_id=model_id,
            message=(
                "the cluster is ready, but the model's chat settings could not "
                f"be written: {_redacted(exc)}"
            ),
        )
    return OperationWarning(
        code=WARNING_CHAT_PROFILE_APPLIED,
        node_id=model_id,
        message=(
            "the assistant set this model's chat settings (temperature 0.6, "
            "top_p 0.95, top_k 0, max_tokens 2048) because it had none; change "
            "them in the model settings at any time"
        ),
    )
