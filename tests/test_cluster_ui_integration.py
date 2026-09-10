# SPDX-License-Identifier: Apache-2.0
"""Exercise the UI's actual backend contracts and delayed-response boundaries."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from omlx.cluster import pairing_routes, routes, runtime
from omlx.cluster.performance import ExecutionSettings
from omlx.cluster.telemetry import RuntimeTelemetry
from tests import test_cluster_replan
from tests.test_cluster_autoconfigure import _app, _autoconfigure_payload
from tests.test_cluster_pairing import _loopback_pair
from tests.test_cluster_runtime import _marker
from tests.ui.test_cluster_v2_wizard import _WIZARD_TWO_MACS, _run_wizard

active_deployment = test_cluster_replan.active_deployment


def test_join_http_flow_completes_both_sides_and_cancels(tmp_path, monkeypatch):
    coordinator, joiner, _, enrollments, _ = _loopback_pair(tmp_path)
    monkeypatch.setattr(pairing_routes, "_get_pairing_manager", lambda: joiner)
    app = FastAPI()
    app.include_router(pairing_routes.pair_admin_router)
    client = TestClient(app)
    assert client.get("/api/cluster/pair/join").json()["state"] == "idle"
    response = client.post(
        "/api/cluster/pair/join", json={"coordinator_addr": "127.0.0.1:8000"}
    )
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert client.get("/api/cluster/pair/join").headers["cache-control"] == "no-store"
    assert (
        client.post(
            "/api/cluster/pair/join", json={"coordinator_addr": "other:8000"}
        ).status_code
        == 409
    )
    assert client.get("/api/cluster/pair/join").json()["state"] == "awaiting_approval"
    coordinator.approve(joiner.node_id, response.json()["code"])
    assert client.get("/api/cluster/pair/join").json()["state"] == "approved"
    assert client.get("/api/cluster/pair/join").json()["state"] == "approved"
    assert len(enrollments) == 2
    assert joiner._devices.get(coordinator.node_id)["state"] == "paired"
    cancelled = client.post("/api/cluster/pair/join/cancel").json()
    assert cancelled["state"] == "idle"
    assert cancelled["code"] is None


def test_join_mutations_are_on_admin_router():
    def deny():
        raise HTTPException(401, "admin required")

    app = FastAPI()
    app.include_router(pairing_routes.pair_router)
    app.include_router(pairing_routes.pair_admin_router, dependencies=[Depends(deny)])
    client = TestClient(app)
    assert (
        client.post(
            "/api/cluster/pair/join", json={"coordinator_addr": "worker:8000"}
        ).status_code
        == 401
    )
    assert client.get("/api/cluster/pair/join").status_code == 401
    assert client.post("/api/cluster/pair/join/cancel").status_code == 401


def test_expired_join_retains_retry_address_and_clears_code(tmp_path):
    from omlx.cluster.pairing import CODE_TTL_SECONDS

    _, joiner, *_ = _loopback_pair(tmp_path)
    joiner.ui_session.begin("coordinator:8000")
    joiner._clock.now += CODE_TTL_SECONDS + 1
    status = joiner.ui_session.poll()
    assert status["state"] == "error"
    assert status["coordinator_addr"] == "coordinator:8000"
    assert status["code"] is None


def test_cancel_prevents_delayed_approval_from_completing_a_new_join(tmp_path):
    coordinator, joiner, _, enrollments, _ = _loopback_pair(tmp_path)
    shown = joiner.ui_session.begin("coordinator:8000")
    coordinator.approve(joiner.node_id, shown["code"])
    approval = coordinator.join_status(joiner.node_id)
    entered, release = Event(), Event()

    def delayed(*args):
        entered.set()
        assert release.wait(5)
        return approval

    joiner._http_get = delayed
    with ThreadPoolExecutor() as executor:
        pending = executor.submit(joiner.ui_session.poll)
        assert entered.wait(5)
        assert joiner.ui_session.poll()["state"] == "awaiting_approval"
        assert joiner.ui_session.cancel()["state"] == "idle"
        joiner._http_post = lambda *args: {"state": "awaiting_approval"}
        joiner.ui_session.begin("other:8000")
        release.set()
        assert pending.result(5)["coordinator_addr"] == "other:8000"
    assert len(enrollments) == 1  # coordinator only; no joiner trust installed
    assert joiner._devices.get(coordinator.node_id) is None


@pytest.mark.parametrize(
    "address",
    ["http://user:pass@host", "https://host", "host/path", "[broken", "host:0"],
)
def test_join_rejects_invalid_coordinator_addresses(tmp_path, address):
    _, joiner, *_ = _loopback_pair(tmp_path)
    from omlx.cluster.pairing import PairingRequestError

    with pytest.raises(PairingRequestError):
        joiner.ui_session.begin(address)
    assert joiner.ui_session.snapshot()["state"] == "idle"


def test_proposal_can_stage_without_launching_and_preserves_identity(monkeypatch):
    monkeypatch.setattr(
        routes,
        "_staging_for",
        lambda *_: {"ready": False, "nodes": [], "total_missing_bytes": 100},
    )
    payload = _autoconfigure_payload() | {
        "deployment_id": "existing-pool",
        "path_map": {"node-0": "/models/local", "node-1": "/peer/model"},
        "prompt_cache_ssd": False,
        "prompt_cache_ssd_max_bytes": 123456,
    }
    client = TestClient(_app())
    result = client.post("/admin/api/cluster/autoconfigure", json=payload).json()
    assert result["ready_to_stage"] is True
    assert result["ready_to_activate"] is False
    for key in (
        "deployment_id",
        "path_map",
        "prompt_cache_ssd",
        "prompt_cache_ssd_max_bytes",
    ):
        assert result["activation"][key] == payload[key]
    monkeypatch.setattr(
        routes,
        "_staging_for",
        lambda *_: {"ready": False, "error": "source unavailable"},
    )
    blocked = client.post("/admin/api/cluster/autoconfigure", json=payload).json()
    assert not blocked["ready_to_stage"] and not blocked["ready_to_activate"]


def test_replan_applies_ssd_settings_to_the_persisted_execution(active_deployment):
    client = TestClient(_app())
    payload = {
        "deployment_id": active_deployment.deployment["deployment_id"],
        "prompt_cache_ssd": False,
        "prompt_cache_ssd_max_bytes": 987654,
    }
    preview = client.post("/admin/api/cluster/replan", json=payload)
    assert preview.status_code == 200, preview.text
    payload["approved_placement"] = preview.json()["plan"]["placement_signature"]
    applied = client.post("/admin/api/cluster/replan", json=payload)
    assert applied.status_code == 200, applied.text
    execution = routes.get_cluster_registry().get(payload["deployment_id"]).execution
    assert execution.prompt_cache_ssd is False
    assert execution.prompt_cache_ssd_max_bytes == 987654
    assert ExecutionSettings.from_dict(execution.to_dict()) == execution


def test_runtime_reconciles_loaded_loading_and_detached(monkeypatch):
    loaded = SimpleNamespace(
        engine=SimpleNamespace(cluster_status=lambda: {"deployment_id": "loaded"}),
        is_loading=False,
    )
    loading = SimpleNamespace(engine=None, is_loading=True, model_path="/loading")
    pool = SimpleNamespace(
        get_loaded_model_ids=lambda: ["a"],
        get_model_ids=lambda: ["a", "b"],
        get_entry=lambda model: loaded if model == "a" else loading,
    )
    monkeypatch.setattr(
        routes,
        "get_cluster_registry",
        lambda: SimpleNamespace(
            get_for_model=lambda _: SimpleNamespace(deployment_id="loading")
        ),
    )
    payload = {
        "jobs": [
            {"deployment_id": name, "live": True}
            for name in ("loaded", "loading", "old")
        ]
    }
    routes._reconcile_runtime_ownership(payload, pool)
    assert [job["ownership"] for job in payload["jobs"]] == [
        "loaded",
        "loading",
        "detached",
    ]
    assert payload["jobs"][0]["live"] is True
    assert payload["jobs"][2]["live"] is False
    assert any(item.get("phase") == "loading" for item in payload["launchers"])


@pytest.mark.parametrize(
    "stage",
    [
        "initializing_full_replica",
        "materializing_fixed",
        "materializing_layers",
        "tensor_ready",
        "weights_resident",
        "warming_prefill_shape",
    ],
)
def test_runtime_accepts_worker_loading_stages(stage):
    assert (
        runtime._validated_marker(_marker(phase="loading", load_stage=stage))[
            "load_stage"
        ]
        == stage
    )


def test_request_metrics_survive_the_runtime_validator_without_content():
    telemetry = RuntimeTelemetry(
        SimpleNamespace(update=lambda *args, **kwargs: None), clock=lambda: 1.0
    )
    ids = [telemetry.begin_request() for _ in range(70)]
    value = runtime._validated_metrics(telemetry.snapshot())
    assert [row["request_id"] for row in value["active_request_metrics"]] == ids[:64]
    assert value["active_request_metrics_truncated"] == 6
    assert all("prompt" not in row for row in value["active_request_metrics"])
    value["active_request_metrics"][1]["request_id"] = ids[0]
    with pytest.raises(ValueError, match="identities"):
        runtime._validated_metrics(value)


def test_initial_plan_uses_measured_budgets_and_role_fraction():
    result = _run_wizard(_WIZARD_TWO_MACS + """
component.selectedModelPath = '/models/m';
component.modelOptions = [{model_path: '/models/m'}];
let posted;
component.apiFetch = async (url, options) => {
  const body = JSON.parse(options.body);
  if (url.endsWith('/node-budgets')) return {nodes: body.hosts.map((host) => ({node_id: host.node_id, capacity_bytes: 32 * 1024**3, reserve_bytes: 8 * 1024**3}))};
  posted = body;
  return {ready_to_activate: true, plan: {placement_signature: 'a'.repeat(16)}, activation: {approved_placement: 'a'.repeat(16)}};
};
(async () => {await component.runPlan(); process.stdout.write(JSON.stringify(posted.nodes));})();
""")
    assert len(result) == 2
    assert all(
        node["capacity_bytes"] == 32 * 1024**3 and node["reserve_bytes"] == 8 * 1024**3
        for node in result
    )
    roles = asyncio.run(routes.cluster_node_roles())["roles"]
    assert all("reserve_fraction" in role for role in roles)


def test_staging_poll_is_single_flight_and_old_generation_cannot_activate():
    result = _run_wizard("""
let resolve, reads = 0, activations = [];
component.apiFetch = () => {reads++; return new Promise((done) => {resolve = done;});};
component.postActivation = async (activation) => activations.push(activation.deployment_id);
component.stagingJob = {job_id: 'old'};
component.stagingActivation = {deployment_id: 'old-pool'};
(async () => {
  const first = component.pollStagingJob();
  await component.pollStagingJob();
  component.dismissStaging();
  component.stagingJob = {job_id: 'new'};
  component.stagingActivation = {deployment_id: 'new-pool'};
  resolve({job_id: 'old', status: 'completed', ready: true});
  await first;
  const retained = component.stagingJob.job_id;
  component.apiFetch = async () => ({job_id: 'new', status: 'completed', ready: true});
  await Promise.all([component.pollStagingJob(), component.pollStagingJob()]);
  await component.pollStagingJob();
  process.stdout.write(JSON.stringify({reads, activations, retained}));
})();
""")
    assert result == {"reads": 1, "activations": ["new-pool"], "retained": "new"}


def test_blockers_and_unpaired_versions_do_not_start_work():
    result = _run_wizard(_WIZARD_TWO_MACS + """
component.devicesPayload.self.version = '1.0';
component.devicesPayload.paired[0].version = '1.0';
component.devicesPayload.discovered = [{node_id: 'unpaired', version: 'wrong', paired: false}];
component.plan = {};
component.planProposal = {activation: {}, ready_to_activate: false, ready_to_stage: false};
let calls = 0;
component.postActivation = async () => calls++;
(async () => {await component.activatePlan(); process.stdout.write(JSON.stringify({calls, mismatches: component.versionMismatches()}));})();
""")
    assert result == {"calls": 0, "mismatches": []}


def test_autoconfigure_paths_match_activation_signature(active_deployment, monkeypatch):
    from omlx.cluster.deployment import ClusterDeployment
    from omlx.cluster.replan import hosts_from_deployment, nodes_from_deployment

    current = ClusterDeployment.from_dict(active_deployment.deployment)
    monkeypatch.setattr(routes, "_staging_for", lambda *_: {"ready": False})
    payload = {
        "deployment_id": current.deployment_id,
        "model_path": current.model,
        "nodes": nodes_from_deployment(current),
        "hosts": hosts_from_deployment(current),
        "path_map": {"large": current.model, "small": "/different/model"},
        "detect_transports": False,
        "preflight": False,
        "auto_tune": False,
        "measure_performance": False,
        "strategy": "pipeline",
        "prompt_cache_ssd": False,
        "prompt_cache_ssd_max_bytes": 345678,
    }
    response = TestClient(_app()).post("/admin/api/cluster/autoconfigure", json=payload)
    assert response.status_code == 200, response.text
    proposal = response.json()
    request = routes.ClusterDeploymentRequest(**proposal["activation"])
    deployment, plan = routes._create_deployment(request)
    assert routes._placement_signature(plan) == proposal["plan"]["placement_signature"]
    assert deployment.deployment_id == current.deployment_id
    assert deployment.path_map == payload["path_map"]
    assert deployment.execution.prompt_cache_ssd is False
    assert deployment.execution.prompt_cache_ssd_max_bytes == 345678


@pytest.mark.parametrize("enabled", [True, False])
def test_worker_argument_roundtrip_keeps_ssd_limit(tmp_path, enabled):
    from omlx.cluster.inference_worker import _execution_settings
    from tests.test_cluster_launch import _deployment, _parsed_plan

    deployment = _deployment()
    deployment = replace(
        deployment,
        execution=replace(
            deployment.execution,
            prompt_cache_ssd=enabled,
            prompt_cache_ssd_max_bytes=654321,
        ),
    )
    args, *_ = _parsed_plan(deployment, tmp_path)
    execution = _execution_settings(args)
    assert execution.prompt_cache_ssd is enabled
    assert execution.prompt_cache_ssd_max_bytes == 654321


def test_staging_reads_destination_path_map(tmp_path, monkeypatch):
    from omlx.cluster import staging
    from tests.test_cluster_staging import _model

    model = _model(tmp_path / "model", layers=2, per_file=1)
    calls = []
    monkeypatch.setattr(staging, "remote_model_dir", lambda host, path: path)
    monkeypatch.setattr(
        staging,
        "remote_file_sizes",
        lambda host, path: calls.append((host, path)) or {},
    )
    assignment = SimpleNamespace(node_id="peer", start_layer=0, end_layer=2)
    result = staging.stage_manifest(
        model,
        [assignment],
        {"peer": "worker.local"},
        path_map={"peer": "/custom/model"},
    )
    assert calls == [("worker.local", "/custom/model")]
    assert result["ready"] is False


def test_expansion_allows_copy_first_and_init_is_idempotent():
    result = _run_wizard("""
let timers = 0, ticks = 0, copied = [];
global.setInterval = () => ++timers;
global.clearInterval = () => {};
component.tick = () => ticks++;
component.init(); component.init();
component.membershipProposal = {ready_to_activate: false, ready_to_stage: true, activation: {deployment_id: 'saved', path_map: {peer: '/peer/model'}}};
component.stageModelToPeers = async (activation) => copied.push(activation);
(async () => {await component.applyMembershipExpansion(); process.stdout.write(JSON.stringify({timers, ticks, copied}));})();
""")
    assert result["timers"] == result["ticks"] == 1
    assert result["copied"] == [
        {"deployment_id": "saved", "path_map": {"peer": "/peer/model"}}
    ]


def test_cancel_then_retry_same_coordinator_over_http(tmp_path, monkeypatch):
    coordinator, joiner, _, enrollments, _ = _loopback_pair(tmp_path)
    monkeypatch.setattr(pairing_routes, "_get_pairing_manager", lambda: joiner)
    app = FastAPI()
    app.include_router(pairing_routes.pair_admin_router)
    client = TestClient(app)
    body = {"coordinator_addr": "127.0.0.1:8000"}
    first = client.post("/api/cluster/pair/join", json=body)
    assert first.status_code == 200
    old_token = joiner.ui_session.attempt["cancel_token"]
    assert "cancel_token" not in first.json()
    assert client.post("/api/cluster/pair/join/cancel").status_code == 200
    assert not coordinator.pending_requests()
    retry = client.post("/api/cluster/pair/join", json=body)
    assert retry.status_code == 200
    from omlx.cluster.pairing import PairingCodeError, PairingStateError

    with pytest.raises(PairingCodeError):
        coordinator.cancel_join_request(joiner.node_id, old_token)
    token = joiner.ui_session.attempt["cancel_token"]
    coordinator._pending[joiner.node_id].approving = True
    with pytest.raises(PairingStateError):
        coordinator.cancel_join_request(joiner.node_id, token)
    coordinator._pending[joiner.node_id].approving = False
    coordinator.approve(joiner.node_id, retry.json()["code"])
    assert client.get("/api/cluster/pair/join").json()["state"] == "approved"
    assert len(enrollments) == 2
    assert client.post("/api/cluster/pair/join/cancel").status_code == 200
    assert coordinator.join_status(joiner.node_id)["state"] == "approved"


def test_lost_join_response_can_be_cancelled_before_retry(tmp_path):
    coordinator, joiner, *_ = _loopback_pair(tmp_path)
    original = joiner._http_post

    def lose_reply(url, payload, timeout):
        result = original(url, payload, timeout)
        if url.endswith("/pair/request"):
            raise OSError("response lost")
        return result

    joiner._http_post = lose_reply
    from omlx.cluster.pairing import PairingRequestError

    with pytest.raises(PairingRequestError):
        joiner.ui_session.begin("coordinator:8000")
    assert coordinator.pending_requests()
    joiner._http_post = original
    retry = joiner.ui_session.begin("coordinator:8000")
    coordinator.approve(joiner.node_id, retry["code"])
    assert joiner.ui_session.poll()["state"] == "approved"


def test_cancel_waits_for_outbound_join_before_withdrawing(tmp_path):
    coordinator, joiner, *_ = _loopback_pair(tmp_path)
    original = joiner._http_post
    entered, release = Event(), Event()

    def delay_request(url, payload, timeout):
        if url.endswith("/pair/request"):
            entered.set()
            assert release.wait(5)
        return original(url, payload, timeout)

    joiner._http_post = delay_request
    with ThreadPoolExecutor() as executor:
        beginning = executor.submit(joiner.ui_session.begin, "coordinator:8000")
        assert entered.wait(5)
        cancelling = executor.submit(joiner.ui_session.cancel)
        release.set()
        beginning.result(5)
        assert cancelling.result(5)["state"] == "idle"
    assert not coordinator.pending_requests()


@pytest.mark.parametrize("ip", ["fd00::2", "fe80::2%en0"])
def test_ui_ipv6_address_reaches_join_request(tmp_path, ip):
    import json

    address = _run_wizard(
        "process.stdout.write(JSON.stringify(component.coordinatorAddrFor("
        + json.dumps({"http_port": 8123, "addrs": [{"ip": ip, "if_type": "ethernet"}]})
        + ")));"
    )
    _, joiner, *_ = _loopback_pair(tmp_path)
    sent = []
    joiner._http_post = lambda url, *_: sent.append(url) or {}
    assert joiner.ui_session.begin(address)["state"] == "awaiting_approval"
    assert sent == [f"http://[{ip}]:8123/api/cluster/pair/request"]


def test_model_change_keeps_other_saved_setup_and_opens_picker():
    result = _run_wizard(_WIZARD_TWO_MACS + """
component.deploymentsPayload = [{deployment_id:'A',model:'/a'}, {deployment_id:'B',model:'/b'}];
component.runtimeLoaded = true; component.runtimePayload = {jobs:[],launchers:[]};
component.apiFetch = async () => ({}); component.notify = () => {};
component.refreshDeployments = async () => {component.deploymentsPayload = [{deployment_id:'B',model:'/b'}];};
component.refreshRuntime = async () => {}; component.loadModels = () => {}; component.loadNodeRoles = () => {};
(async () => {
 const a = component.deploymentsPayload[0]; component.beginModelChange(a); await component.changeClusterModel(a);
 process.stdout.write(JSON.stringify({screen:component.wizardState(),saved:component.deploymentsPayload.map(d=>d.deployment_id)}));
})();
""")
    assert result == {"screen": "plan", "saved": ["B"]}


@pytest.mark.parametrize("old_fails", [False, True])
def test_runtime_ignores_old_response_after_lifecycle_refresh(old_fails):
    result = _run_wizard(
        """
let pending = []; component.apiFetch = () => new Promise((resolve,reject) => pending.push({resolve,reject}));
(async () => {
 const old = component.refreshRuntime(); const current = component.refreshRuntime();
 pending[1].resolve({jobs:[],launchers:[],revision:'unloaded'}); await current;
 """
        + (
            "pending[0].reject(new Error('old failure'));"
            if old_fails
            else "pending[0].resolve({jobs:[],revision:'old ready'});"
        )
        + """
 await old; process.stdout.write(JSON.stringify({revision:component.runtimePayload.revision,loaded:component.runtimeLoaded}));
})();
"""
    )
    assert result == {"revision": "unloaded", "loaded": True}


@pytest.mark.parametrize("during_stage", [False, True])
def test_expansion_conflict_refreshes_membership_proposal(during_stage):
    result = _run_wizard(
        """
component.membershipProposal = {ready_to_stage:true,activation:{deployment_id:'saved',path_map:{a:'/a'}}};
let normal = 0, expanded = 0;
component.notify = () => {}; component.apiFetch = async () => {throw {status:409};};
component.runPlan = async () => normal++;
component.previewMembershipExpansion = async () => {if (!component.membershipBusy) {expanded++; component.membershipProposal={activation:{deployment_id:'saved'},ready_to_stage:true};}};
(async () => {
"""
        + (
            "await component.applyMembershipExpansion();"
            if during_stage
            else "await component.postActivation(component.membershipProposal.activation, 'membership');"
        )
        + """
 process.stdout.write(JSON.stringify({normal,expanded,busy:component.activateBusy,id:component.membershipProposal.activation.deployment_id}));
})();
"""
    )
    assert result == {"normal": 0, "expanded": 1, "busy": False, "id": "saved"}


def test_coordinator_cancel_endpoint_requires_attempt_token(tmp_path, monkeypatch):
    coordinator, joiner, *_ = _loopback_pair(tmp_path)
    monkeypatch.setattr(pairing_routes, "_get_pairing_manager", lambda: coordinator)
    app = FastAPI()
    app.include_router(pairing_routes.pair_router)
    client = TestClient(app, client=("127.0.0.1", 50000))

    def send(url, payload, timeout):
        from urllib.parse import urlsplit

        response = client.post(urlsplit(url).path, json=payload)
        response.raise_for_status()
        return response.json()

    joiner._http_post = send
    joiner.ui_session.begin("coordinator:8000")
    wrong = client.post(
        "/api/cluster/pair/request/cancel",
        json={"node_id": joiner.node_id, "token": "0" * 64},
    )
    assert wrong.status_code == 403
    assert coordinator.pending_requests()
    assert joiner.ui_session.cancel()["state"] == "idle"
    assert not coordinator.pending_requests()


def test_failed_cancellation_keeps_proof_for_retry(tmp_path):
    coordinator, joiner, *_ = _loopback_pair(tmp_path)
    original = joiner._http_post
    joiner.ui_session.begin("coordinator:8000")

    def disconnected(*args):
        raise OSError("coordinator offline")

    joiner._http_post = disconnected
    from omlx.cluster.pairing import PairingRequestError

    with pytest.raises(PairingRequestError):
        joiner.ui_session.cancel()
    assert joiner.ui_session.snapshot()["state"] == "error"
    assert coordinator.pending_requests()
    joiner._http_post = original
    assert joiner.ui_session.cancel()["state"] == "idle"
    assert not coordinator.pending_requests()
