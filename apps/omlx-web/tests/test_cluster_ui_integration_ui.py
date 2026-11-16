# SPDX-License-Identifier: Apache-2.0
"""Drive the cluster wizard against the backend contracts it depends on."""

# The fixture templates use %-formatting on purpose: their JS bodies contain
# braces, so .format()/f-strings would need escaping that makes them unreadable.
# ruff: noqa: UP031

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from tests.ui.test_cluster_v2_wizard import _WIZARD_TWO_MACS, _fixtures, _run_wizard

from omlx.cluster import routes
from tests.test_cluster_pairing import _loopback_pair

ROOT = Path(__file__).resolve().parents[1]
CHAT_TEMPLATE = ROOT / "omlx_web/templates/chat.html"


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


# --- Review slice 6: the assistant card is mounted independently --------------
#
# REVIEW-2026-11-13.md B3: a configured deployment (wizardState 'active') must
# not hide the assistant's own surface, and the entry must stay reachable on a
# Mac that already has a saved setup.


def test_assistant_surface_stays_visible_with_a_configured_deployment():
    result = _run_wizard(_WIZARD_TWO_MACS + """
component.deploymentsPayload = [{deployment_id:'A',model:'/a'}];
component.runtimePayload = {jobs:[],launchers:[]};
component.assistant.state = 'ready';
component.assistant.endpoint = {ready:true, url:'http://172.16.1.11:8099'};
process.stdout.write(JSON.stringify({
  wizard: component.wizardState(),
  surface: component.assistantSurfaceVisible(),
}));
""")
    # The wizard shows its management panel, but the assistant stays mounted.
    assert result == {"wizard": "active", "surface": True}


def test_assistant_surface_hidden_before_any_peer_or_state():
    result = _run_wizard("""
component.devicesPayload = {
  self: {node_id:'node-a', caps:{ram_gb:256}, addrs:[]},
  paired: [], discovered: [],
};
component.assistant.state = 'idle';
process.stdout.write(JSON.stringify({surface: component.assistantSurfaceVisible()}));
""")
    assert result == {"surface": False}


def test_assistant_surface_visible_when_a_session_exists_without_peers():
    result = _run_wizard("""
component.devicesPayload = {
  self: {node_id:'node-a', caps:{ram_gb:256}, addrs:[]},
  paired: [], discovered: [],
};
component.assistant.session = {operation_id:'op', state:'ready', stopped:true};
process.stdout.write(JSON.stringify({surface: component.assistantSurfaceVisible()}));
""")
    assert result == {"surface": True}


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


# --- Cluster Assistant (minimal Screen-3 slice) -------------------------------
#
# The assistant card is driven entirely through the seven assistant admin
# routes. These tests execute the shipped component under Node (skipped when
# node is absent) and assert the request order, the frozen plan-time inputs, the
# ready-only endpoint and the one action per failure. No real cluster, no SSH.

_ASSISTANT_TIMER_STUBS = """
global.setInterval = () => 1;
global.clearInterval = () => {};
global.setTimeout = () => 0;
"""

_ASSISTANT_PRELUDE = """
component.modelOptions = [{
  model_path: '/models/qwen3-32b', id: 'qwen',
  display_name: 'Qwen3 32B', model_context_length: 131072,
}];
component.selectedModelPath = '/models/qwen3-32b';
const recommendation = %s;
const planFixture = %s;
const applyingFixture = %s;
const loadingFixture = %s;
const readyFixture = %s;
const failureFixture = %s;
const calls = [];
const measured = {};
let sequence = 0;
const applyQueue = [];
const startQueue = [];
component.apiFetch = async (url, options = {}) => {
  const body = options && options.body ? JSON.parse(options.body) : null;
  calls.push({url, method: (options && options.method) || 'GET', body});
  if (url.endsWith('/assistant/measure')) {
    sequence += 1;
    const id = 'm' + sequence;
    measured[id] = body.model_path;
    return {measurement_id: id, state: 'running', poll: 'status?measurement_id=' + id};
  }
  if (url.includes('/assistant/status')) {
    const id = url.split('measurement_id=')[1];
    return {measurement_id: id, state: 'ready', model: measured[id],
      node_count: 2, nodes: [], warnings: [], error: null};
  }
  if (url.endsWith('/assistant/recommend')) return recommendation;
  if (url.endsWith('/assistant/plan')) return planFixture;
  if (url.endsWith('/assistant/apply')) {
    return {operation_id: 'op-apply', state: 'applying', poll: 'operations/op-apply'};
  }
  if (url.endsWith('/assistant/start')) {
    return {operation_id: 'op-start', state: 'starting', poll: 'operations/op-start'};
  }
  if (url.includes('/assistant/models/')) {
    return {node_id: 'node-b', model_id: 'old', deleted: true};
  }
  if (url.endsWith('/assistant/stop')) {
    return {
      operation_id: 'op-start', deployment_id: 'dev-1.5b-abc123',
      stopped: true, warnings: [],
    };
  }
  if (url.includes('/assistant/operations/op-apply')) {
    return applyQueue.length ? applyQueue.shift() : applyingFixture;
  }
  if (url.includes('/assistant/operations/op-start')) {
    return startQueue.length ? startQueue.shift() : loadingFixture;
  }
  throw new Error('unexpected URL ' + url);
};
async function reachPlan() {
  await component.startAssistant();
  await component.pollAssistantMeasurement();   // entry measurement → recommend
  await component.pollAssistantMeasurement();   // recommended model → plan
}
%s
"""


def _assistant_flow(extra: str) -> str:
    fixtures = _fixtures()
    prelude = _ASSISTANT_PRELUDE % (
        json.dumps(fixtures["assistant_recommendation.json"]),
        json.dumps(fixtures["assistant_plan.json"]),
        json.dumps(fixtures["assistant_operation_applying.json"]),
        json.dumps(fixtures["assistant_operation_loading.json"]),
        json.dumps(fixtures["assistant_operation_ready.json"]),
        json.dumps(fixtures["assistant_error_retry_coordinator.json"]),
        extra,
    )
    return _WIZARD_TWO_MACS + _ASSISTANT_TIMER_STUBS + prelude


def test_assistant_recommendation_binds_selected_model_and_context():
    result = _run_wizard(
        _assistant_flow(
            """
(async () => {
  await reachPlan();
  process.stdout.write(JSON.stringify({
    state: component.assistantState(),
    selectedModel: component.assistant.selectedModel,
    selectedModelId: component.assistant.selectedModelId,
    selectedContext: component.assistant.selectedContextTokens,
    pickerModel: component.selectedModelPath,
    pickerContext: component.targetContextTokens,
    planSignature: component.assistant.plan && component.assistant.plan.plan_signature,
    recommendBody: calls.find((c) => c.url.endsWith('/assistant/recommend')).body,
    planBody: calls.find((c) => c.url.endsWith('/assistant/plan')).body,
    measureModels: calls.filter((c) => c.url.endsWith('/assistant/measure'))
      .map((c) => c.body.model_path),
  }));
})();
"""
        )
    )

    assert result["state"] == "planning"
    assert result["selectedModel"] == "/models/qwen3-32b-4bit"
    assert result["selectedModelId"] == "mlx-community/Qwen3-32B-4bit"
    assert result["selectedContext"] == 32768
    assert result["pickerModel"] == "/models/qwen3-32b-4bit"
    assert result["pickerContext"] == 32768
    assert result["planSignature"] == "3f9a1c7d20b45e68"
    # recommend reads the hardware half of the entry measurement and evaluates
    # the model list; the manager does not invent a candidate.
    assert result["recommendBody"]["measurement_id"] == "m1"
    assert [
        candidate["model_path"]
        for candidate in result["recommendBody"]["candidates"]
    ] == ["/models/qwen3-32b"]
    assert result["recommendBody"]["context_tokens"] == 32768
    # Phase C3 (ROADMAP-TO-CHAPTER-10.md §6, recorded in SCREEN-3-MINIMAL.md
    # §4.2): the entry measurement is hardware-only (no model_path), and the
    # recommended model is measured model-bound before plan; plan then uses
    # that measurement and exactly that model.
    assert result["measureModels"] == [
        None,
        "/models/qwen3-32b-4bit",
    ]
    assert result["planBody"]["measurement_id"] == "m2"
    assert result["planBody"]["model_path"] == "/models/qwen3-32b-4bit"
    assert result["planBody"]["context_tokens"] == 32768


def test_assistant_plan_and_apply_share_frozen_inputs_and_apply_precedes_start():
    result = _run_wizard(
        _assistant_flow(
            """
(async () => {
  await reachPlan();
  await component.runAssistantApply();
  const startsBefore = calls.filter((c) => c.url.endsWith('/assistant/start')).length;
  applyQueue.push(Object.assign({}, applyingFixture, {
    state: 'applied', operation_id: 'op-apply',
  }));
  await component.pollAssistantOperation();
  const startCall = calls.find((c) => c.url.endsWith('/assistant/start'));
  process.stdout.write(JSON.stringify({
    planBody: calls.find((c) => c.url.endsWith('/assistant/plan')).body,
    applyBody: calls.find((c) => c.url.endsWith('/assistant/apply')).body,
    startsBefore,
    startBody: startCall && startCall.body,
    state: component.assistantState(),
    applyOperationId: component.assistant.applyOperationId,
  }));
})();
"""
        )
    )

    plan_body = result["planBody"]
    apply_body = result["applyBody"]
    for key in (
        "measurement_id",
        "model_path",
        "context_tokens",
        "rank0_extra_bytes",
        "mtp",
        "execution_profile",
        "microbatch_size",
    ):
        assert apply_body[key] == plan_body[key], key
    assert apply_body["plan_signature"] == "3f9a1c7d20b45e68"
    assert apply_body["coordinator_node"] == "node-b"
    assert apply_body["rank_order"] == ["node-b", "node-a"]
    assert apply_body["rank0_extra_bytes"] == 0
    assert apply_body["mtp"] is False
    assert apply_body["execution_profile"] == "balanced"
    assert apply_body["microbatch_size"] is None
    # start is not posted while the apply operation is still applying.
    assert result["startsBefore"] == 0
    assert result["applyOperationId"] == "op-apply"
    assert result["startBody"]["operation_id"] == "op-apply"
    assert result["startBody"]["plan_signature"] == "3f9a1c7d20b45e68"
    assert result["startBody"]["coordinator_node"] == "node-b"
    assert result["state"] == "starting"


def test_assistant_start_is_not_posted_for_an_unapplied_or_failed_apply():
    result = _run_wizard(
        _assistant_flow(
            """
(async () => {
  await reachPlan();
  await component.runAssistantApply();
  await component.pollAssistantOperation();   // applying
  const whileApplying = calls.filter((c) => c.url.endsWith('/assistant/start')).length;
  applyQueue.push(Object.assign({}, applyingFixture, {
    state: 'failed',
    error: {code: 'staging_failed', message: 'copy failed', action: 'diagnostics'},
  }));
  await component.pollAssistantOperation();
  process.stdout.write(JSON.stringify({
    whileApplying,
    starts: calls.filter((c) => c.url.endsWith('/assistant/start')).length,
    state: component.assistantState(),
    action: component.assistant.error && component.assistant.error.action,
  }));
})();
"""
        )
    )

    assert result["whileApplying"] == 0
    assert result["starts"] == 0
    assert result["state"] == "failed"
    assert result["action"] == "diagnostics"


def test_assistant_operation_poll_stops_at_ready_and_drops_a_stale_generation():
    result = _run_wizard(
        _assistant_flow(
            """
(async () => {
  await reachPlan();
  await component.runAssistantApply();
  applyQueue.push(Object.assign({}, applyingFixture, {
    state: 'applied', operation_id: 'op-apply',
  }));
  await component.pollAssistantOperation();   // start op created
  startQueue.push(loadingFixture);
  await component.pollAssistantOperation();   // loading
  const loading = {
    state: component.assistantState(),
    timer: component.assistant.operationTimer,
    ready: component.assistantReady(),
  };
  startQueue.push(readyFixture);
  await component.pollAssistantOperation();   // ready
  const ready = {
    state: component.assistantState(),
    timer: component.assistant.operationTimer,
    ready: component.assistantReady(),
    url: component.assistant.endpoint && component.assistant.endpoint.url,
  };
  let reads = 0;
  let resolveOld;
  component.apiFetch = () => {
    reads += 1;
    return new Promise((done) => { resolveOld = done; });
  };
  component.assistant.operationId = 'op-stale';
  const pending = component.pollAssistantOperation();
  await component.pollAssistantOperation();   // single flight
  component.assistant.generation += 1;
  component.assistant.operationId = 'op-new';
  component.assistant.state = 'loading';
  resolveOld(readyFixture);
  await pending;
  process.stdout.write(JSON.stringify({
    loading, ready, reads,
    staleState: component.assistantState(),
  }));
})();
"""
        )
    )

    assert result["loading"] == {"state": "loading", "timer": 1, "ready": False}
    assert result["ready"] == {
        "state": "ready",
        "timer": None,
        "ready": True,
        "url": "http://172.16.1.11:8099",
    }
    # The second call was single-flight, and the late old-generation response
    # could not flip the newer state back to ready.
    assert result["reads"] == 1
    assert result["staleState"] == "loading"


def test_assistant_endpoint_is_hidden_until_the_operation_is_ready():
    result = _run_wizard(
        _assistant_flow(
            """
(async () => {
  await reachPlan();
  await component.runAssistantApply();
  applyQueue.push(Object.assign({}, applyingFixture, {
    state: 'applied', operation_id: 'op-apply',
  }));
  await component.pollAssistantOperation();
  startQueue.push(loadingFixture);
  await component.pollAssistantOperation();
  const loading = {
    state: component.assistantState(),
    endpointReady: component.assistant.endpoint && component.assistant.endpoint.ready,
    exposed: component.assistantReady(),
  };
  startQueue.push(readyFixture);
  await component.pollAssistantOperation();
  const ready = {
    state: component.assistantState(),
    endpointReady: component.assistant.endpoint && component.assistant.endpoint.ready,
    exposed: component.assistantReady(),
    url: component.assistant.endpoint.url,
  };
  process.stdout.write(JSON.stringify({loading, ready}));
})();
"""
        )
    )

    assert result["loading"] == {
        "state": "loading",
        "endpointReady": False,
        "exposed": False,
    }
    assert result["ready"]["state"] == "ready"
    assert result["ready"]["endpointReady"] is True
    assert result["ready"]["exposed"] is True
    assert result["ready"]["url"] == "http://172.16.1.11:8099"


def test_assistant_retry_coordinator_retries_the_elected_coordinator_only():
    result = _run_wizard(
        _assistant_flow(
            """
(async () => {
  await reachPlan();
  await component.runAssistantApply();
  applyQueue.push(Object.assign({}, applyingFixture, {
    state: 'applied', operation_id: 'op-apply',
  }));
  await component.pollAssistantOperation();
  startQueue.push(failureFixture);
  await component.pollAssistantOperation();
  const failed = {
    state: component.assistantState(),
    action: component.assistant.error && component.assistant.error.action,
    coordinator: component.assistantCoordinatorName(),
    note: component.assistantCoordinatorInstruction(),
    label: component.assistantActionLabel('retry_coordinator'),
  };
  const counts = () => ({
    measure: calls.filter((c) => c.url.endsWith('/assistant/measure')).length,
    apply: calls.filter((c) => c.url.endsWith('/assistant/apply')).length,
    start: calls.filter((c) => c.url.endsWith('/assistant/start')).length,
  });
  const before = counts();
  await component.assistantActionHandler('retry_coordinator');
  const after = counts();
  const starts = calls.filter((c) => c.url.endsWith('/assistant/start'));
  process.stdout.write(JSON.stringify({
    failed, before, after,
    coordinatorNode: starts[starts.length - 1].body.coordinator_node,
    planSignature: starts[starts.length - 1].body.plan_signature,
    operationId: starts[starts.length - 1].body.operation_id,
    selfNode: component.selfDevice().node_id,
  }));
})();
"""
        )
    )

    assert result["failed"]["state"] == "failed"
    assert result["failed"]["action"] == "retry_coordinator"
    assert result["failed"]["coordinator"] == "Node B"
    assert "Open and configure oMLX on Node B" in result["failed"]["note"]
    assert result["failed"]["label"] == "Retry on Node B"
    # Retry re-posts start for the already-applied plan: no new measurement and
    # no second apply, and the coordinator never becomes this Mac.
    assert result["after"]["measure"] == result["before"]["measure"]
    assert result["after"]["apply"] == result["before"]["apply"]
    assert result["after"]["start"] == result["before"]["start"] + 1
    assert result["coordinatorNode"] == "node-b"
    assert result["coordinatorNode"] != result["selfNode"]
    assert result["planSignature"] == "3f9a1c7d20b45e68"
    assert result["operationId"] == "op-apply"


def test_assistant_pair_mac_opens_the_existing_pairing_flow_and_starts_nothing():
    result = _run_wizard(
        """
global.setTimeout = () => 0;
component.devicesPayload = {
  self: {node_id: 'node-a', paired: true, caps: {}, addrs: []},
  paired: [{node_id: 'node-b', paired: true, caps: {}, addrs: []}],
  discovered: [
    {node_id: 'node-c', paired: false, state: 'discovered', friendly_name: 'Mini'},
  ],
};
component.assistant.state = 'failed';
component.assistant.error = {
  code: 'node_not_paired', message: 'pair it', action: 'pair_mac',
};
const urls = [];
component.apiFetch = async (url) => {
  urls.push(url);
  throw new Error('the pair action starts nothing');
};
component.assistantActionHandler('pair_mac');
process.stdout.write(JSON.stringify({
  target: component.pairing.target && component.pairing.target.node_id,
  state: component.assistantState(),
  error: component.assistant.error,
  urls,
}));
"""
    )

    assert result["target"] == "node-c"
    assert result["state"] == "idle"
    assert result["error"] is None
    # The existing pairing flow may open its invite; it must never start an
    # assistant operation (measure/plan/apply/start).
    assert all(
        url.startswith("/api/cluster/pair/") for url in result["urls"]
    ), result["urls"]
    assert not any(
        "/admin/api/cluster/assistant/" in url for url in result["urls"]
    )


def test_assistant_flow_never_calls_stage_deploy_replan_or_autoconfigure():
    result = _run_wizard(
        _assistant_flow(
            """
(async () => {
  await reachPlan();
  await component.runAssistantApply();
  applyQueue.push(Object.assign({}, applyingFixture, {
    state: 'applied', operation_id: 'op-apply',
  }));
  await component.pollAssistantOperation();
  startQueue.push(readyFixture);
  await component.pollAssistantOperation();
  const forbidden = [
    '/deployments', '/stage', '/replan', '/autoconfigure',
    '/ssh-key', '/peer-probe',
  ];
  process.stdout.write(JSON.stringify({
    state: component.assistantState(),
    urls: calls.map((c) => c.url),
    offending: calls
      .filter((c) => forbidden.some((f) => c.url.includes(f)))
      .map((c) => c.url),
  }));
})();
"""
        )
    )

    assert result["state"] == "ready"
    assert result["offending"] == []
    assert all("/assistant/" in url for url in result["urls"])


# --- Screen 4: the progress surface ------------------------------------------
#
# SCREEN-4-PROGRESS.md §2 pins which backend value makes which row current. These
# tests walk a real apply/start against the recorded operation payloads and read
# the rows the card would render.

_SCREEN_4_ROW = """
const currentRow = () => {
  const rows = component.assistantProgressRows();
  const row = rows.find(
    (item) => item.state === 'current' || item.state === 'failed',
  );
  return row ? row.key : '';
};
"""


def test_assistant_walks_the_screen_4_rows_in_the_backend_order():
    fixtures = _fixtures()
    result = _run_wizard(
        _assistant_flow(
            _SCREEN_4_ROW
            + """
const startingFixture = %s;
const appliedSnapshot = Object.assign({}, applyingFixture, {
  state: 'applied', operation_id: 'op-apply',
  progress: {
    phase: 'applied',
    message: 'the model is staged on node-b; nothing has been started',
    nodes_ready: 2, nodes_total: 2,
  },
});
const rowOf = (key) => component.assistantProgressRows()
  .find((row) => row.key === key);
(async () => {
  const seen = [];
  await component.startAssistant();
  seen.push(currentRow());                      // the entry measurement runs
  await component.pollAssistantMeasurement();   // recommend → model measurement
  seen.push(currentRow());                      // still checking that model
  await component.pollAssistantMeasurement();   // model-bound → plan
  seen.push(currentRow());
  await component.runAssistantApply();
  applyQueue.push(applyingFixture);
  await component.pollAssistantOperation();
  seen.push(currentRow());
  const preparing = rowOf('preparing');
  applyQueue.push(appliedSnapshot);
  await component.pollAssistantOperation();     // applied → start posted
  seen.push(currentRow());
  startQueue.push(startingFixture);
  await component.pollAssistantOperation();
  seen.push(currentRow());
  startQueue.push(loadingFixture);
  await component.pollAssistantOperation();
  seen.push(currentRow());
  const loading = rowOf('loading');
  startQueue.push(readyFixture);
  await component.pollAssistantOperation();
  seen.push(currentRow());
  const rows = component.assistantProgressRows();
  process.stdout.write(JSON.stringify({
    seen,
    order: rows.map((row) => row.key + ':' + row.state),
    preparing: {message: preparing.message, count: preparing.count},
    loading: {message: loading.message},
    endpoint: component.assistantReady() && component.assistant.endpoint.url,
  }));
})();
"""
            % json.dumps(fixtures["assistant_operation_starting.json"])
        )
    )

    # The six rows appear in the corrected order — planning before staging — and
    # no row is ever skipped or invented.
    distinct = []
    for row in result["seen"]:
        if not distinct or distinct[-1] != row:
            distinct.append(row)
    assert distinct == [
        "checking",
        "planning",
        "preparing",
        "starting",
        "loading",
        "ready",
    ]
    # Every earlier row is done once Ready is reached.
    assert result["order"] == [
        "checking:done",
        "planning:done",
        "preparing:done",
        "starting:done",
        "loading:done",
        "ready:current",
    ]
    # The backend's own sentence and Mac count sit under the current row.
    assert result["preparing"]["message"] == "the operation is running on node-b"
    assert result["preparing"]["count"] == "1 of 2 Macs ready"
    assert result["loading"]["message"] == "loading weights on node-b"
    assert result["endpoint"] == "http://172.16.1.11:8099"


def test_assistant_ready_row_is_not_reached_without_a_ready_endpoint():
    result = _run_wizard(
        _assistant_flow(
            """
const readyWithoutEndpoint = Object.assign({}, readyFixture, {
  endpoint: Object.assign({}, readyFixture.endpoint, {ready: false}),
});
(async () => {
  await reachPlan();
  await component.runAssistantApply();
  applyQueue.push(Object.assign({}, applyingFixture, {
    state: 'applied', operation_id: 'op-apply',
  }));
  await component.pollAssistantOperation();
  startQueue.push(readyWithoutEndpoint);
  await component.pollAssistantOperation();
  const rows = component.assistantProgressRows();
  process.stdout.write(JSON.stringify({
    state: component.assistantState(),
    ready: component.assistantReady(),
    current: rows.filter((row) => row.state === 'current').map((row) => row.key),
    readyRow: rows.find((row) => row.key === 'ready').state,
  }));
})();
"""
        )
    )

    # ``ready`` without a proven endpoint never becomes the Ready row and never
    # exposes a URL: the last proven row (load + canary) stays.
    assert result["ready"] is False
    assert result["current"] == ["loading"]
    assert result["readyRow"] == "pending"


def test_assistant_unknown_phase_keeps_the_previous_row_and_shows_the_message():
    fixtures = _fixtures()
    result = _run_wizard(
        _assistant_flow(
            """
const unknownPhase = %s;
(async () => {
  await reachPlan();
  await component.runAssistantApply();
  applyQueue.push(Object.assign({}, applyingFixture, {
    state: 'applied', operation_id: 'op-apply',
  }));
  await component.pollAssistantOperation();
  startQueue.push(loadingFixture);
  await component.pollAssistantOperation();
  startQueue.push(unknownPhase);
  await component.pollAssistantOperation();
  const rows = component.assistantProgressRows();
  process.stdout.write(JSON.stringify({
    current: rows.filter((row) => row.state === 'current').map((row) => row.key),
    messages: rows.filter((row) => row.message).map((row) => row.message),
    state: component.assistantState(),
  }));
})();
"""
            % json.dumps(fixtures["assistant_operation_unknown_phase.json"])
        )
    )

    # A phase this build does not know never blanks the row and never guesses:
    # the row that was current stays, and the raw message is shown under it.
    assert result["current"] == ["loading"]
    assert result["messages"] == [
        "the coordinator is working through a step this build does not know"
    ]


def test_assistant_failure_keeps_the_row_it_stopped_in():
    fixtures = _fixtures()
    result = _run_wizard(
        _assistant_flow(
            """
const diskFailure = %s;
(async () => {
  await reachPlan();
  await component.runAssistantApply();
  applyQueue.push(applyingFixture);
  await component.pollAssistantOperation();
  applyQueue.push(diskFailure);
  await component.pollAssistantOperation();
  const rows = component.assistantProgressRows();
  process.stdout.write(JSON.stringify({
    state: component.assistantState(),
    stopped: rows.filter(
      (row) => row.state === 'failed' || row.state === 'current',
    ).map((row) => row.key + ':' + row.state),
    message: (rows.find((row) => row.message) || {}).message,
    action: component.assistant.error && component.assistant.error.action,
    label: component.assistantActionLabel(component.assistant.error.action),
  }));
})();
"""
            % json.dumps(fixtures["assistant_error_free_disk_space.json"])
        )
    )

    assert result["state"] == "failed"
    # The phase the backend last reported stays, marked failed, with its own
    # message — and exactly the one action the backend named.
    assert result["stopped"] == ["preparing:failed"]
    assert result["message"] == "staged 1 of 2 Macs"
    assert result["action"] == "free_disk_space"
    assert result["label"] == "Free disk space"


def test_assistant_free_space_lists_only_picked_mac_and_deletes_in_two_steps():
    result = _run_wizard(
        """
global.setInterval = () => 1;
global.clearInterval = () => {};
global.setTimeout = () => 0;
component.devicesPayload = {
  self: {node_id: 'node-b', friendly_name: 'Node B', caps: {}, addrs: []},
  paired: [{node_id: 'node-a', friendly_name: 'Node A', paired: true, caps: {}, addrs: []}],
  discovered: [],
};
component.devicesLoaded = true;
component.modelOptions = [{model_path: '/models/qwen3-32b-4bit', id: 'qwen'}];
component.selectedModelPath = '/models/qwen3-32b-4bit';
const calls = [];
component.apiFetch = async (url, options = {}) => {
  calls.push({url, method: options.method || 'GET', body: options.body || null});
  if (url.endsWith('/assistant/measure')) {
    return {measurement_id: 'm3', state: 'running', poll: 'status'};
  }
  if (url.includes('/assistant/models/')) return {deleted: true};
  return {state: 'running'};
};
component.notify = () => {};
component.assistant.state = 'failed';
component.assistant.error = {
  code: 'insufficient_disk',
  message: 'Node A needs 18.0 GiB for its stage and has 4.0 GiB free',
  action: 'free_disk_space',
};
component.assistant.measurementId = 'm2';
component.assistant.selectedModel = '/models/qwen3-32b-4bit';
component.assistant.measurementNodes = [
  {node_id: 'node-b', models: [
    {model_id: 'old', model_path: '/models/old', size_bytes: 12 * 1024 ** 3},
    {model_id: 'mlx-community/Qwen3-32B-4bit',
     model_path: '/models/qwen3-32b-4bit', size_bytes: 18 * 1024 ** 3},
  ]},
  {node_id: 'node-a', models: [
    {model_id: 'unused', model_path: '/models/unused', size_bytes: 5 * 1024 ** 3},
  ]},
];
const deletes = () => calls.filter((call) => call.method === 'DELETE');
(async () => {
  component.assistantActionHandler('free_disk_space');
  const opened = {
    open: component.assistant.freeSpaceOpen,
    node: component.assistant.freeSpaceNode,
    listed: component.assistantFreeSpaceModels().length,
  };
  component.chooseAssistantFreeSpaceNode('node-b');
  const listed = component.assistantFreeSpaceModels().map((model) => model.model_id);
  const model = component.assistantFreeSpaceModels()[0];
  const before = component.assistantDeleteLabel(model);
  await component.deleteAssistantModel(model);   // arms only
  const armed = {
    confirm: component.assistant.freeSpaceConfirm,
    label: component.assistantDeleteLabel(model),
    deletes: deletes().length,
  };
  await component.deleteAssistantModel(model);   // confirms and sends
  process.stdout.write(JSON.stringify({
    opened,
    listed,
    before,
    armed,
    deletes: deletes().map((call) => call.url),
    state: component.assistantState(),
    measurementId: component.assistant.measurementId,
    measured: calls.filter((call) => call.url.endsWith('/assistant/measure')).length,
    open: component.assistant.freeSpaceOpen,
  }));
})();
"""
    )

    # Nothing is listed until the user picks that Mac, and never another Mac's
    # models. The checkpoint the cluster is about to run is not offered.
    assert result["opened"] == {"open": True, "node": "", "listed": 0}
    assert result["listed"] == ["old"]
    assert result["before"] == "Delete"
    # Destructive needs the second click: the first only arms the row.
    assert result["armed"] == {"confirm": "old", "label": "Delete old?", "deletes": 0}
    assert result["deletes"] == [
        "/admin/api/cluster/assistant/models/node-b/old?measurement_id=m2"
    ]
    # After a delete the assistant measures again instead of assuming the copy
    # now fits.
    assert result["state"] == "measuring"
    assert result["measurementId"] == "m3"
    assert result["measured"] == 1
    assert result["open"] is False


def test_assistant_resumes_the_running_operation_the_backend_names():
    """Reload mid-operation: the backend names the operation, the card polls it.

    A reloaded page has no plan-time inputs, so the honest resume point is the
    answer to the next request: ``409 operation_in_progress`` carries the running
    ``operation_id`` and the one action ``status``. The card then renders the
    phase that stored record reports (SCREEN-4-PROGRESS.md §5).
    """

    result = _run_wizard(
        _assistant_flow(
            """
(async () => {
  await reachPlan();
  const callsAfter = [];
  component.apiFetch = async (url) => {
    callsAfter.push(url);
    if (url.endsWith('/assistant/apply')) {
      const error = new Error('the operation is still running');
      error.payload = {
        code: 'operation_in_progress',
        message: 'start operation op-live is still running; poll it instead',
        action: 'status',
        operation_id: 'op-live',
      };
      throw error;
    }
    return loadingFixture;
  };
  await component.runAssistantApply();
  const refused = {
    state: component.assistantState(),
    action: component.assistant.error.action,
    adopted: component.assistant.operationId,
  };
  component.assistantActionHandler('status');
  await component.pollAssistantOperation();
  const rows = component.assistantProgressRows();
  process.stdout.write(JSON.stringify({
    refused,
    polls: callsAfter.filter((url) => url.includes('/operations/op-live')).length,
    state: component.assistantState(),
    current: rows.filter((row) => row.state === 'current').map((row) => row.key),
    message: (rows.find((row) => row.message) || {}).message,
  }));
})();
"""
        )
    )

    # The refusal carries the running operation and exactly the status action.
    assert result["refused"] == {
        "state": "failed",
        "action": "status",
        "adopted": "op-live",
    }
    # Resuming renders the phase the stored record reports, with its message.
    assert result["polls"] == 1
    assert result["state"] == "loading"
    assert result["current"] == ["loading"]
    assert result["message"] == "loading weights on node-b"


# --- Review slice 2: measurement recovery and poll robustness -----------------
#
# REVIEW-2026-11-13.md U2/U7: a transient blip or a named 409 must recover, not
# wedge the card; a resumed apply without plan-time inputs rests on the session
# surface instead of failing a healthy cluster.


def test_assistant_measurement_409_recovery_adopts_the_measurement_id():
    """U2: the refused measure names the running measurement; status resumes it."""

    result = _run_wizard(
        _assistant_flow(
            """
(async () => {
  await component.startAssistant();   // entry measurement m1 is running
  let statusPolls = 0;
  component.apiFetch = async (url, options) => {
    const body = options && options.body ? JSON.parse(options.body) : null;
    calls.push({url, method: (options && options.method) || 'GET', body});
    if (url.endsWith('/assistant/measure')) {
      const error = new Error('measurement running');
      error.payload = {
        code: 'measurement_in_progress',
        message: 'measurement m1 is still running; poll it instead',
        action: 'status',
        measurement_id: 'm1',
      };
      throw error;
    }
    if (url.includes('/assistant/status')) {
      statusPolls += 1;
      return {measurement_id: 'm1', state: 'running', model: null,
        node_count: 2, nodes: [], warnings: [], error: null};
    }
    throw new Error('unexpected URL ' + url);
  };
  await component.runAssistantMeasure();   // posts measure -> 409 -> failed
  const refused = {
    state: component.assistantState(),
    action: component.assistant.error.action,
    measurementId: component.assistant.measurementId,
  };
  component.assistantActionHandler('status');
  const resumedState = component.assistantState();
  await component.pollAssistantMeasurement();
  process.stdout.write(JSON.stringify({
    refused, resumedState, statusPolls,
    state: component.assistantState(),
    error: component.assistant.error,
  }));
})();
"""
        )
    )
    assert result["refused"] == {
        "state": "failed",
        "action": "status",
        "measurementId": "m1",
    }
    # Resume restores the measuring state, clears the banner, and really polls.
    assert result["resumedState"] == "measuring"
    assert result["statusPolls"] >= 1
    assert result["state"] == "measuring"
    assert result["error"] is None


def test_assistant_measure_action_resumes_a_named_running_measurement():
    """U2: the measure action resumes the named measurement, never re-POSTs it."""

    result = _run_wizard(
        _assistant_flow(
            """
(async () => {
  await component.startAssistant();
  let statusPolls = 0;
  component.apiFetch = async (url, options) => {
    const body = options && options.body ? JSON.parse(options.body) : null;
    calls.push({url, method: (options && options.method) || 'GET', body});
    if (url.endsWith('/assistant/measure')) {
      const error = new Error('measurement running');
      error.payload = {
        code: 'measurement_in_progress', message: 'still running',
        action: 'status', measurement_id: 'm1',
      };
      throw error;
    }
    if (url.includes('/assistant/status')) {
      statusPolls += 1;
      return {measurement_id: 'm1', state: 'running', model: null,
        node_count: 2, nodes: [], warnings: [], error: null};
    }
    throw new Error('unexpected URL ' + url);
  };
  await component.runAssistantMeasure();
  const measureBefore = calls.filter((c) => c.url.endsWith('/assistant/measure')).length;
  component.assistantActionHandler('measure');   // must resume, not reset+POST
  const measureAfter = calls.filter((c) => c.url.endsWith('/assistant/measure')).length;
  await component.pollAssistantMeasurement();
  process.stdout.write(JSON.stringify({
    measureBefore, measureAfter, statusPolls,
    state: component.assistantState(),
  }));
})();
"""
        )
    )
    assert result["measureAfter"] == result["measureBefore"]
    assert result["statusPolls"] >= 1
    assert result["state"] == "measuring"


def test_assistant_poll_survives_transient_errors_then_fails_after_the_limit():
    """A couple of network blips recover; a permanent outage still fails."""

    result = _run_wizard(
        _assistant_flow(
            """
(async () => {
  await component.startAssistant();
  let failures = 2;
  component.apiFetch = async (url) => {
    if (url.includes('/assistant/status')) {
      if (failures > 0) { failures -= 1; throw new Error('network blip'); }
      return {measurement_id: 'm1', state: 'running', model: null,
        node_count: 2, nodes: [], warnings: [], error: null};
    }
    throw new Error('unexpected URL ' + url);
  };
  for (let i = 0; i < 12; i++) { await component.pollAssistantMeasurement(); }
  const recovered = {
    state: component.assistantState(),
    errors: component.assistant.measurePollErrors,
  };
  component.apiFetch = async (url) => {
    if (url.includes('/assistant/status')) throw new Error('down for good');
    throw new Error('unexpected URL ' + url);
  };
  for (let i = 0; i < 20; i++) { await component.pollAssistantMeasurement(); }
  process.stdout.write(JSON.stringify({
    recovered, finalState: component.assistantState(),
  }));
})();
"""
        )
    )
    assert result["recovered"] == {"state": "measuring", "errors": 0}
    assert result["finalState"] == "failed"


def test_assistant_resumed_apply_without_inputs_rests_on_the_session():
    """U7: a resumed apply that reaches applied has no inputs; rest, don't fail."""

    result = _run_wizard(
        _assistant_flow(
            """
(async () => {
  await reachPlan();
  // Simulate a reload: the plan-time inputs are gone, a running apply is named.
  component.assistant.plan = null;
  component.assistant.inputs = null;
  component.assistant.operationId = 'op-apply';
  component.assistant.operation = {kind: 'apply', state: 'applying', operation_id: 'op-apply'};
  let startCalls = 0, sessionCalls = 0;
  component.apiFetch = async (url) => {
    if (url.includes('/assistant/operations/op-apply')) {
      return {kind: 'apply', state: 'applied', operation_id: 'op-apply',
        progress: null, warnings: []};
    }
    if (url.endsWith('/assistant/start')) {
      startCalls += 1;
      return {operation_id: 'op-start', state: 'starting'};
    }
    if (url.endsWith('/assistant/session')) {
      sessionCalls += 1;
      return {session: {operation_id: 'op-apply', kind: 'apply', state: 'applied',
        model_path: '/models/qwen3-32b-4bit', context_tokens: 32768}};
    }
    throw new Error('unexpected URL ' + url);
  };
  await component.pollAssistantOperation();
  process.stdout.write(JSON.stringify({
    state: component.assistantState(),
    startCalls, sessionCalls,
    operationId: component.assistant.operationId,
    sessionVisible: component.assistantSessionVisible(),
    error: component.assistant.error,
  }));
})();
"""
        )
    )
    assert result["state"] == "idle"
    assert result["startCalls"] == 0
    assert result["sessionCalls"] == 1
    assert result["operationId"] == ""
    assert result["sessionVisible"] is True
    assert result["error"] is None


# --- Screen 5: the handoff and the measured line ------------------------------
#
# The ready card opens the existing /chat with the cluster's serving model id,
# and shows a rate only when the operation carries `observed`. These tests
# execute the shipped component, plus the shipped chat preselection methods,
# under Node. No real cluster, no key, no network.

_ASSISTANT_REACH_READY = """
async function reachReady() {
  await reachPlan();
  await component.runAssistantApply();
  applyQueue.push(Object.assign({}, applyingFixture, {
    state: 'applied', operation_id: 'op-apply',
  }));
  await component.pollAssistantOperation();
  startQueue.push(loadingFixture);
  await component.pollAssistantOperation();
}
"""


def test_assistant_open_chat_targets_the_coordinator_and_the_serving_model_id():
    result = _run_wizard(
        _assistant_flow(
            _ASSISTANT_REACH_READY
            + """
(async () => {
  await reachReady();
  component.modelOptions.push({
    model_path: '/models/qwen3-32b-4bit', id: 'qwen3-32b-4bit',
  });
  startQueue.push(readyFixture);
  await component.pollAssistantOperation();
  const remote = {
    base: component.assistantChatBase(),
    url: component.assistantChatUrl(),
    agentBase: component.assistantAgentBaseUrl(),
    agentModel: component.assistantAgentModelId(),
    note: component.assistantRemoteChatNote(),
    local: component.assistantCoordinatorIsLocal(),
  };
  // The coordinator is this Mac: a relative page keeps this host's session.
  component.devicesPayload.self.node_id = 'node-b';
  const local = {
    base: component.assistantChatBase(),
    url: component.assistantChatUrl(),
    note: component.assistantRemoteChatNote(),
    local: component.assistantCoordinatorIsLocal(),
  };
  // No catalogue entry: link without a model parameter, never a guess.
  component.devicesPayload.self.node_id = 'node-a';
  component.modelOptions = component.modelOptions.filter(
    (model) => model.model_path !== '/models/qwen3-32b-4bit',
  );
  const unresolved = component.assistantChatUrl();
  process.stdout.write(JSON.stringify({remote, local, unresolved}));
})();
"""
        )
    )

    # The fixture's coordinator is node-b (http://172.16.1.11:8099) and the
    # wizard's own Mac is node-a, so this is the remote case: that Mac's page.
    assert result["remote"]["local"] is False
    assert result["remote"]["base"] == "http://172.16.1.11:8099/chat"
    assert (
        result["remote"]["url"]
        == "http://172.16.1.11:8099/chat?model=qwen3-32b-4bit"
    )
    assert result["remote"]["agentBase"] == "http://172.16.1.11:8099/v1"
    assert result["remote"]["agentModel"] == "qwen3-32b-4bit"
    assert "Node B" in result["remote"]["note"]
    assert "sign in" in result["remote"]["note"]

    assert result["local"]["local"] is True
    assert result["local"]["base"] == "/chat"
    assert result["local"]["url"] == "/chat?model=qwen3-32b-4bit"
    assert result["local"]["note"] == ""

    assert result["unresolved"] == "http://172.16.1.11:8099/chat"


def test_assistant_ready_renders_only_the_measured_rate():
    result = _run_wizard(
        _assistant_flow(
            _ASSISTANT_REACH_READY
            + """
const observedReady = Object.assign({}, readyFixture, {
  observed: {
    prefill_tok_per_s: 320.5,
    decode_tok_per_s: 42.4,
    ttft_ms: 250.0,
    measured_at: '2026-11-07T23:40:12+00:00',
  },
});
(async () => {
  await reachReady();
  startQueue.push(observedReady);
  await component.pollAssistantOperation();
  const measured = {
    state: component.assistantState(),
    label: component.assistantObservedLabel(),
    observed: component.assistant.observed,
  };
  process.stdout.write(JSON.stringify(measured));
})();
"""
        )
    )

    assert result["state"] == "ready"
    assert result["observed"]["decode_tok_per_s"] == 42.4
    assert "42.4 tok/s decode" in result["label"]
    assert "320.5 tok/s prefill" in result["label"]
    assert "first token in 250 ms" in result["label"]
    assert "measured 2026-11-07T23:40:12+00:00" in result["label"]


def test_assistant_ready_says_nothing_about_speed_without_observed():
    result = _run_wizard(
        _assistant_flow(
            _ASSISTANT_REACH_READY
            + """
(async () => {
  await reachReady();
  startQueue.push(readyFixture);   // observed: null
  await component.pollAssistantOperation();
  process.stdout.write(JSON.stringify({
    state: component.assistantState(),
    observed: component.assistant.observed,
    label: component.assistantObservedLabel(),
  }));
})();
"""
        )
    )

    assert result["state"] == "ready"
    assert result["observed"] is None
    assert result["label"] == ""


def _chat_method(name: str) -> str:
    text = CHAT_TEMPLATE.read_text()
    start = text.index(f"            {name}() {{")
    depth = 0
    index = start
    while index < len(text):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
        index += 1
    raise AssertionError(f"unterminated chat method {name}")


def test_chat_preselection_selects_a_known_model_and_ignores_an_unknown_one():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to execute the chat preselection")

    methods = ",\n".join(
        _chat_method(name)
        for name in ("requestedModelFromUrl", "applyRequestedModelFromUrl")
    )
    script = f"""
global.window = {{location: {{search: process.argv[1]}}}};
const component = {{
  requestedModelParam: null,
  requestedModelApplied: false,
  availableModels: [{{id: 'dev-1.5b'}}, {{id: 'other'}}],
  currentModel: 'other',
  isModelAvailableOnServer(id) {{
    return this.availableModels.some((model) => model.id === id);
  }},
  resolveGatewayModelId(id) {{ return id; }},
}};
Object.assign(component, {{ {methods} }});
component.applyRequestedModelFromUrl();
process.stdout.write(JSON.stringify({{
  model: component.currentModel,
  applied: component.requestedModelApplied,
}}));
"""
    checked = {"known": "?model=dev-1.5b", "unknown": "?model=not-a-model", "none": ""}
    results = {}
    for name, search in checked.items():
        completed = subprocess.run(
            [node, "-e", script, search],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert completed.returncode == 0, completed.stderr
        results[name] = json.loads(completed.stdout)

    assert results["known"] == {"model": "dev-1.5b", "applied": True}
    # An unknown value leaves the normal default selection untouched.
    assert results["unknown"] == {"model": "other", "applied": True}
    assert results["none"] == {"model": "other", "applied": True}


# --- Screen 6: state + stop/restart -----------------------------------------
#
# Stop is one assistant route; the browser never calls a coordinator's unload
# route itself. Start again re-applies the frozen plan inputs and lets the
# existing poll start it — no plan, no measure, no model/context edit.

def test_assistant_stop_closes_the_endpoint_and_offers_start_again():
    result = _run_wizard(
        _assistant_flow(
            _ASSISTANT_REACH_READY
            + """
(async () => {
  await reachReady();
  startQueue.push(readyFixture);
  await component.pollAssistantOperation();
  const before = {
    ready: component.assistantReady(),
    url: component.assistant.endpoint && component.assistant.endpoint.url,
    deployment: component.assistant.deploymentId,
  };
  const beforeCalls = calls.length;
  await component.stopAssistantCluster();
  const afterCalls = calls.slice(beforeCalls);
  const stopCall = afterCalls.find((call) => call.url.endsWith('/assistant/stop'));
  process.stdout.write(JSON.stringify({
    before,
    stopped: component.assistant.stopped,
    ready: component.assistantReady(),
    endpoint: component.assistant.endpoint,
    deployed: component.assistant.deploymentId,
    current: component.assistantProgressRows()
      .filter((row) => row.state === 'current').map((row) => row.key),
    stopBody: stopCall && stopCall.body,
    unloadCalls: calls.filter(
      (call) => call.url.includes('/unload') || call.url.includes('/deployments'),
    ).length,
  }));
})();
"""
        )
    )

    assert result["before"]["ready"] is True
    assert result["before"]["url"] == "http://172.16.1.11:8099"
    assert result["before"]["deployment"] == "dev-1.5b-abc123"
    assert result["stopped"] is True
    assert result["ready"] is False
    assert result["endpoint"] is None
    assert result["current"] == ["ready"]
    assert result["stopBody"] == {"operation_id": "op-start"}
    assert result["unloadCalls"] == 0


def test_assistant_start_again_reapplies_without_replanning():
    result = _run_wizard(
        _assistant_flow(
            _ASSISTANT_REACH_READY
            + """
(async () => {
  await reachReady();
  startQueue.push(readyFixture);
  await component.pollAssistantOperation();
  await component.stopAssistantCluster();
  const beforeCalls = calls.length;
  const targetBefore = {
    measurementId: component.assistant.measurementId,
    context: component.assistant.selectedContextTokens,
  };
  await component.startAssistantAgain();
  const afterCalls = calls.slice(beforeCalls);
  const applyCall = afterCalls.find((call) => call.url.endsWith('/assistant/apply'));
  process.stdout.write(JSON.stringify({
    state: component.assistantState(),
    stopped: component.assistant.stopped,
    measurementId: component.assistant.measurementId,
    context: component.assistant.selectedContextTokens,
    planCalls: afterCalls.filter((call) => call.url.endsWith('/assistant/plan')).length,
    recommendCalls: afterCalls.filter(
      (call) => call.url.endsWith('/assistant/recommend'),
    ).length,
    measureCalls: afterCalls.filter(
      (call) => call.url.endsWith('/assistant/measure'),
    ).length,
    applyBody: applyCall && applyCall.body,
    signature: planFixture.plan_signature,
    targetBefore,
  }));
})();
"""
        )
    )

    # The first step is apply, with the same frozen plan signature and inputs.
    assert result["state"] == "applying"
    assert result["stopped"] is False
    assert result["planCalls"] == 0
    assert result["recommendCalls"] == 0
    assert result["measureCalls"] == 0
    assert result["applyBody"]["plan_signature"] == result["signature"]
    assert result["applyBody"]["measurement_id"] == result["targetBefore"]["measurementId"]
    assert result["measurementId"] == result["targetBefore"]["measurementId"]
    assert result["context"] == result["targetBefore"]["context"]


def test_assistant_renders_a_backend_stopped_operation_without_an_endpoint():
    fixtures = _fixtures()
    result = _run_wizard(
        _assistant_flow(
            _ASSISTANT_REACH_READY
            + """
const stoppedFixture = %s;
(async () => {
  await reachReady();
  startQueue.push(stoppedFixture);
  await component.pollAssistantOperation();
  process.stdout.write(JSON.stringify({
    state: component.assistantState(),
    stopped: component.assistant.stopped,
    ready: component.assistantReady(),
    current: component.assistantProgressRows()
      .filter((row) => row.state === 'current').map((row) => row.key),
    deployment: component.assistant.deploymentId,
  }));
})();
"""
            % json.dumps(fixtures["assistant_operation_ready_stopped.json"])
        )
    )

    assert result["state"] == "ready"
    assert result["stopped"] is True
    assert result["ready"] is False
    assert result["current"] == ["ready"]
    assert result["deployment"] == "dev-1.5b-abc123"


# --- Review slice 11: remaining UI mediums/lows ------------------------------
#
# REVIEW-2026-11-13.md §3 (UI): the clipboard reported success on a rejected
# write, the Details JSON kept a ready endpoint after Stop, Refresh left
# state/progress stale with no generation guard, and the debug pane printed a
# not-ready endpoint URL. These tests execute the shipped component under Node.


def test_assistant_copy_notifies_success_only_on_a_real_copy():
    """L2: a rejected or unavailable clipboard is an error, never a fake copy."""

    result = _run_wizard(
        """
global.setTimeout = () => 0;
const toasts = [];
component.notify = (type, message) => toasts.push({type, message});
const flush = () => new Promise((resolve) => setImmediate(resolve));
// Node exposes a getter-only `navigator`; replace it via defineProperty.
const setNavigator = (value) => Object.defineProperty(
  globalThis, 'navigator', {value, configurable: true, writable: true},
);
(async () => {
  setNavigator({clipboard: {writeText: async () => {}}});
  component.copyAssistantValue('http://172.16.1.11:8099');
  await flush();
  const ok = {toasts: toasts.slice(), copied: component.assistant.copied};
  toasts.length = 0;
  component.assistant.copied = '';
  setNavigator({
    clipboard: {writeText: () => Promise.reject(new Error('denied'))},
  });
  component.copyAssistantValue('http://172.16.1.11:8099');
  await flush();
  const rejected = {toasts: toasts.slice(), copied: component.assistant.copied};
  toasts.length = 0;
  setNavigator({});
  component.copyAssistantValue('http://172.16.1.11:8099');
  await flush();
  const unavailable = {toasts: toasts.slice(), copied: component.assistant.copied};
  process.stdout.write(JSON.stringify({ok, rejected, unavailable}));
})();
"""
    )

    # A real copy marks the value and notifies success.
    assert result["ok"]["copied"] == "http://172.16.1.11:8099"
    assert [toast["type"] for toast in result["ok"]["toasts"]] == ["success"]
    # A rejected write notifies an error and never marks the value copied.
    assert [toast["type"] for toast in result["rejected"]["toasts"]] == ["error"]
    assert result["rejected"]["copied"] == ""
    # No clipboard at all is an error, not a silent success.
    assert [toast["type"] for toast in result["unavailable"]["toasts"]] == ["error"]
    assert result["unavailable"]["copied"] == ""


def test_assistant_details_json_after_stop_has_no_ready_endpoint():
    """L2: Stop updates the operation snapshot so Details no longer reads ready."""

    result = _run_wizard(
        _assistant_flow(
            _ASSISTANT_REACH_READY
            + """
(async () => {
  await reachReady();
  startQueue.push(readyFixture);
  await component.pollAssistantOperation();
  await component.stopAssistantCluster();
  const json = component.assistantOperationJson();
  const parsed = JSON.parse(json);
  process.stdout.write(JSON.stringify({
    stoppedFlag: component.assistant.stopped,
    state: component.assistantState(),
    snapStopped: parsed.stopped,
    snapEndpoint: parsed.endpoint,
    hasUrl: json.includes('http://172.16.1.11:8099'),
  }));
})();
"""
        )
    )

    # The stopped flag and the ready state machine value survive the stop.
    assert result["stoppedFlag"] is True
    assert result["state"] == "ready"
    # The snapshot itself now reports stopped and carries no live endpoint.
    assert result["snapStopped"] is True
    assert result["snapEndpoint"] is None
    assert result["hasUrl"] is False


def test_assistant_refresh_syncs_state_and_progress():
    """L3: a bounded Refresh routes through the poll's applier, not a partial set."""

    result = _run_wizard(
        _assistant_flow(
            _ASSISTANT_REACH_READY
            + """
(async () => {
  await reachReady();   // rests in 'loading'
  const before = {
    state: component.assistantState(),
    phase: component.assistant.progress && component.assistant.progress.phase,
  };
  startQueue.push(readyFixture);
  await component.refreshAssistantOperation();
  process.stdout.write(JSON.stringify({
    before,
    state: component.assistantState(),
    phase: component.assistant.progress && component.assistant.progress.phase,
    ready: component.assistantReady(),
    url: component.assistant.endpoint && component.assistant.endpoint.url,
    busy: component.assistant.busy,
  }));
})();
"""
        )
    )

    assert result["before"] == {"state": "loading", "phase": "loading"}
    assert result["state"] == "ready"
    assert result["phase"] == "ready"
    assert result["ready"] is True
    assert result["url"] == "http://172.16.1.11:8099"
    assert result["busy"] is False


def test_assistant_refresh_is_busy_guarded_and_generation_safe():
    """L3: Refresh reads once, refuses while busy, and drops a stale generation."""

    result = _run_wizard(
        _assistant_flow(
            _ASSISTANT_REACH_READY
            + """
(async () => {
  await reachReady();
  let reads = 0;
  let resolveFetch;
  component.apiFetch = () => {
    reads += 1;
    return new Promise((resolve) => { resolveFetch = resolve; });
  };
  component.assistant.busy = true;
  await component.refreshAssistantOperation();
  const busyReads = reads;
  component.assistant.busy = false;
  component.assistant.state = 'loading';
  component.assistant.endpoint = null;
  component.assistant.operation = null;
  const pending = component.refreshAssistantOperation();
  component.assistant.generation += 1;
  component.assistant.state = 'starting';
  resolveFetch(readyFixture);
  await pending;
  process.stdout.write(JSON.stringify({
    busyReads,
    reads,
    state: component.assistantState(),
    endpoint: component.assistant.endpoint,
    operation: component.assistant.operation,
    busy: component.assistant.busy,
  }));
})();
"""
        )
    )

    # Busy guard: no read while busy; exactly one bounded read afterwards.
    assert result["busyReads"] == 0
    assert result["reads"] == 1
    # The late old-generation response could not clobber the newer state.
    assert result["state"] == "starting"
    assert result["endpoint"] is None
    assert result["operation"] is None
    assert result["busy"] is False


def test_assistant_operation_json_redacts_the_endpoint_until_ready():
    """L4: the debug pane mirrors the card's gate and never prints a port-0 URL."""

    result = _run_wizard(
        _assistant_flow(
            _ASSISTANT_REACH_READY
            + """
(async () => {
  await reachReady();   // loading, endpoint.ready false
  const loadingJson = component.assistantOperationJson();
  startQueue.push(readyFixture);
  await component.pollAssistantOperation();   // ready
  const readyJson = component.assistantOperationJson();
  await component.stopAssistantCluster();     // stopped
  const stoppedJson = component.assistantOperationJson();
  // A port-0 URL is never rendered, even on an otherwise ready snapshot.
  component.assistant.stopped = false;
  component.assistant.state = 'ready';
  component.assistant.endpoint = {ready: true, port: 0, url: 'http://172.16.1.11:0'};
  component.assistant.operation = Object.assign({}, readyFixture, {
    endpoint: {ready: true, port: 0, url: 'http://172.16.1.11:0'},
  });
  const portZeroJson = component.assistantOperationJson();
  process.stdout.write(JSON.stringify({
    loadingEndpoint: JSON.parse(loadingJson).endpoint,
    loadingHasUrl: loadingJson.includes('http://172.16.1.11:8099'),
    readyUrl: JSON.parse(readyJson).endpoint.url,
    stoppedEndpoint: JSON.parse(stoppedJson).endpoint,
    stoppedHasUrl: stoppedJson.includes('http://172.16.1.11:8099'),
    portZeroEndpoint: JSON.parse(portZeroJson).endpoint,
    portZeroHasUrl: portZeroJson.includes('http://172.16.1.11:0'),
  }));
})();
"""
        )
    )

    # Before ready the endpoint is redacted; the URL never leaks into the pane.
    assert result["loadingEndpoint"] is None
    assert result["loadingHasUrl"] is False
    # When the card would show it, the pane shows it too.
    assert result["readyUrl"] == "http://172.16.1.11:8099"
    # After stop it is redacted again.
    assert result["stoppedEndpoint"] is None
    assert result["stoppedHasUrl"] is False
    # A port-0 URL is never rendered.
    assert result["portZeroEndpoint"] is None
    assert result["portZeroHasUrl"] is False


# --- Review slice 11: session-card runtime coverage (Phase C1 reload resume) --
#
# SCREEN-6-MANAGE.md §1: after a reload the idle card shows the persisted
# session (Running / Stopped / Applied) and offers Start again / Stop. These
# tests execute the shipped session methods under Node with the fake apiFetch;
# no real cluster, no SSH. The session payload mirrors the backend's
# AssistantSessionResponse (models.py) and the plan/ready fixtures.


def test_assistant_session_label_reports_running_stopped_and_applied():
    result = _run_wizard(
        _WIZARD_TWO_MACS
        + """
const base = {
  model_path: '/models/qwen3-32b-4bit', context_tokens: 32768,
  coordinator_node: 'node-b',
};
// The label is `<model> · <context> tokens · <coordinator> · <state>`; build
// the expected context with the same locale call so the pin is locale-safe.
const expected = (state) =>
  'qwen3-32b-4bit · ' + Number(32768).toLocaleString() + ' tokens · Node B · ' + state;
component.assistant.session = Object.assign({}, base, {state: 'ready', stopped: false});
const runningLabel = component.assistantSessionLabel();
component.assistant.session = Object.assign({}, base, {state: 'ready', stopped: true});
const stoppedLabel = component.assistantSessionLabel();
component.assistant.session = Object.assign({}, base, {state: 'applied', stopped: false});
const appliedLabel = component.assistantSessionLabel();
component.assistant.session = null;
const emptyLabel = component.assistantSessionLabel();
process.stdout.write(JSON.stringify({
  runningLabel, stoppedLabel, appliedLabel, emptyLabel,
  expectedRunning: expected('Running'),
  expectedStopped: expected('Stopped'),
  expectedApplied: expected('Applied'),
}));
"""
    )
    # ready + not stopped -> Running; ready + stopped -> Stopped; an applied
    # apply -> Applied; no session -> empty.
    assert result["runningLabel"] == result["expectedRunning"]
    assert result["stoppedLabel"] == result["expectedStopped"]
    assert result["appliedLabel"] == result["expectedApplied"]
    assert result["emptyLabel"] == ""


def test_assistant_session_endpoint_only_when_ready_and_not_stopped():
    """Slice 6 (U5): a stopped or non-ready session never offers a dead link."""

    result = _run_wizard(
        _WIZARD_TWO_MACS
        + """
const url = 'http://172.16.1.11:8099';
const base = {endpoint: {ready: true, url}};
component.assistant.session = Object.assign({}, base, {state: 'ready', stopped: false});
const running = component.assistantSessionEndpoint();
component.assistant.session = {state: 'ready', stopped: false, endpoint: {ready: false, url}};
const notReady = component.assistantSessionEndpoint();
component.assistant.session = Object.assign({}, base, {state: 'ready', stopped: true});
const stopped = component.assistantSessionEndpoint();
component.assistant.session = Object.assign({}, base, {state: 'applied', stopped: false});
const applied = component.assistantSessionEndpoint();
component.assistant.session = {state: 'ready', stopped: false};   // no endpoint
const noEndpoint = component.assistantSessionEndpoint();
component.assistant.session = null;
const noSession = component.assistantSessionEndpoint();
process.stdout.write(JSON.stringify({running, notReady, stopped, applied, noEndpoint, noSession}));
"""
    )
    assert result["running"] == "http://172.16.1.11:8099"
    assert result["notReady"] == ""
    assert result["stopped"] == ""
    assert result["applied"] == ""
    assert result["noEndpoint"] == ""
    assert result["noSession"] == ""


def test_assistant_session_visible_only_when_idle_without_plan_or_operation():
    result = _run_wizard(
        _WIZARD_TWO_MACS
        + """
const session = {operation_id: 'op-start', state: 'ready', stopped: false};
const visible = () => component.assistantSessionVisible();
component.assistant.session = session;
component.assistant.state = 'idle';
component.assistant.plan = null;
component.assistant.operationId = '';
const idle = visible();
component.assistant.plan = {feasible: true};
const withPlan = visible();
component.assistant.plan = null;
component.assistant.operationId = 'op-start';
const withOperation = visible();
component.assistant.operationId = '';
component.assistant.state = 'measuring';
const notIdle = visible();
component.assistant.state = 'idle';
component.assistant.session = null;
const noSession = visible();
process.stdout.write(JSON.stringify({idle, withPlan, withOperation, notIdle, noSession}));
"""
    )
    assert result == {
        "idle": True,
        "withPlan": False,
        "withOperation": False,
        "notIdle": False,
        "noSession": False,
    }


def test_assistant_load_session_fetches_stores_and_is_guarded():
    """The reload read: one session fetch when idle, guarded otherwise."""

    result = _run_wizard(
        _assistant_flow(
            """
(async () => {
  const session = {operation_id: 'op-start', kind: 'start', state: 'ready',
    stopped: false, model_path: '/models/qwen3-32b-4bit', context_tokens: 32768,
    coordinator_node: 'node-b'};
  let reads = 0;
  component.apiFetch = async (url) => {
    if (url.endsWith('/assistant/session')) { reads += 1; return {session}; }
    throw new Error('unexpected URL ' + url);
  };
  component.assistant.state = 'idle';
  component.assistant.plan = null;
  component.assistant.operationId = '';
  component.assistant.busy = false;
  component.assistant.session = null;
  await component.loadAssistantSession();
  const loaded = component.assistant.session;
  const loadedReads = reads;
  // A plan, an operation id, or busy each suppress the read.
  component.assistant.plan = {feasible: true};
  await component.loadAssistantSession();
  const withPlan = reads;
  component.assistant.plan = null;
  component.assistant.operationId = 'op-start';
  await component.loadAssistantSession();
  const withOperation = reads;
  component.assistant.operationId = '';
  component.assistant.busy = true;
  await component.loadAssistantSession();
  const whileBusy = reads;
  component.assistant.busy = false;
  // A failed read clears the session rather than throwing.
  component.apiFetch = async () => { throw new Error('down'); };
  await component.loadAssistantSession();
  const afterFailure = component.assistant.session;
  process.stdout.write(JSON.stringify({
    loaded, loadedReads, withPlan, withOperation, whileBusy, afterFailure,
  }));
})();
"""
        )
    )
    assert result["loaded"]["operation_id"] == "op-start"
    assert result["loadedReads"] == 1
    assert result["withPlan"] == 1
    assert result["withOperation"] == 1
    assert result["whileBusy"] == 1
    assert result["afterFailure"] is None


def test_assistant_stop_session_posts_operation_id_and_is_busy_guarded():
    result = _run_wizard(
        _assistant_flow(
            """
(async () => {
  const running = {operation_id: 'op-start', kind: 'start', state: 'ready',
    stopped: false, model_path: '/models/qwen3-32b-4bit', context_tokens: 32768,
    coordinator_node: 'node-b'};
  component.assistant.session = running;
  component.assistant.state = 'idle';
  const toasts = [];
  component.notify = (type, message) => toasts.push({type, message});
  let sessionReads = 0;
  const seen = [];
  component.apiFetch = async (url, options = {}) => {
    const body = options && options.body ? JSON.parse(options.body) : null;
    seen.push({url, method: (options && options.method) || 'GET', body});
    if (url.endsWith('/assistant/stop')) {
      return {operation_id: 'op-start', deployment_id: 'dev-1.5b-abc123',
        stopped: true, warnings: []};
    }
    if (url.endsWith('/assistant/session')) {
      sessionReads += 1;
      return {session: Object.assign({}, running, {stopped: true})};
    }
    throw new Error('unexpected URL ' + url);
  };
  await component.stopAssistantSession();
  const sessionAfter = component.assistant.session;
  const busyAfter = component.assistant.busy;
  // Busy guard: a second call while busy posts nothing.
  component.assistant.busy = true;
  const beforeBusy = seen.length;
  await component.stopAssistantSession();
  const busyPosts = seen.length - beforeBusy;
  component.assistant.busy = false;
  // No session: nothing to stop.
  component.assistant.session = null;
  const beforeEmpty = seen.length;
  await component.stopAssistantSession();
  const emptyPosts = seen.length - beforeEmpty;
  const stopCall = seen.find((c) => c.url.endsWith('/assistant/stop'));
  process.stdout.write(JSON.stringify({
    sessionAfter, busyAfter, sessionReads, busyPosts, emptyPosts,
    stopBody: stopCall && stopCall.body,
    stopMethod: stopCall && stopCall.method,
    unloadCalls: seen.filter(
      (c) => c.url.includes('/unload') || c.url.includes('/deployments'),
    ).length,
    toasts: toasts.map((t) => t.type),
  }));
})();
"""
        )
    )
    # Stop is one assistant route with the session's operation id; the browser
    # never dials a coordinator's unload/deployments route itself.
    assert result["stopBody"] == {"operation_id": "op-start"}
    assert result["stopMethod"] == "POST"
    assert result["unloadCalls"] == 0
    assert result["toasts"] == ["success"]
    assert result["busyAfter"] is False
    assert result["busyPosts"] == 0
    assert result["emptyPosts"] == 0
    # After a successful stop the handler reloads the session (busy cleared
    # first), so the strip shows the stopped deployment with "Start again"
    # rather than vanishing (SCREEN-6-MANAGE.md §1).
    assert result["sessionReads"] == 1
    assert result["sessionAfter"]["operation_id"] == "op-start"
    assert result["sessionAfter"]["stopped"] is True


def test_assistant_resume_reapplies_stored_inputs_without_measuring():
    result = _run_wizard(
        _assistant_flow(
            """
(async () => {
  const session = {
    operation_id: 'op-start', kind: 'start', state: 'ready', stopped: true,
    measurement_id: 'msr-a1b2c3d4', plan_signature: '3f9a1c7d20b45e68',
    model_path: '/models/qwen3-32b-4bit', context_tokens: 32768,
    target_context_tokens: 32768, rank0_extra_bytes: 0, mtp: false,
    execution_profile: 'balanced', microbatch_size: 1,
    coordinator_node: 'node-b', rank_order: ['node-b', 'node-a'],
    two_mac_fallback: false,
    resource_policy: {context_tokens: 32768, target_context_tokens: 32768,
      table_offload: 'none', table_kind: '', moe_expert_offload: false,
      moe_resident_fraction: 1, memory_guard_tier: null, microbatch_size: 1,
      admission: 'p50'},
  };
  const seen = [];
  component.apiFetch = async (url, options = {}) => {
    const body = options && options.body ? JSON.parse(options.body) : null;
    seen.push({url, method: (options && options.method) || 'GET', body});
    if (url.endsWith('/assistant/apply')) {
      return {operation_id: 'op-apply', state: 'applying', poll: 'operations/op-apply'};
    }
    throw new Error('unexpected URL ' + url);
  };
  component.assistant.session = session;
  component.assistant.state = 'idle';
  // Running guard: a busy assistant does not resume.
  component.assistant.busy = true;
  await component.runAssistantResume();
  const busyPosts = seen.length;
  component.assistant.busy = false;
  await component.runAssistantResume();
  const applyCall = seen.find((c) => c.url.endsWith('/assistant/apply'));
  process.stdout.write(JSON.stringify({
    busyPosts,
    state: component.assistantState(),
    session: component.assistant.session,
    selectedModel: component.assistant.selectedModel,
    measurementId: component.assistant.measurementId,
    applyBody: applyCall && applyCall.body,
    urls: seen.map((c) => c.url),
    measureCalls: seen.filter((c) => c.url.endsWith('/assistant/measure')).length,
    recommendCalls: seen.filter((c) => c.url.endsWith('/assistant/recommend')).length,
    planCalls: seen.filter((c) => c.url.endsWith('/assistant/plan')).length,
    startCalls: seen.filter((c) => c.url.endsWith('/assistant/start')).length,
  }));
})();
"""
        )
    )
    # The guard posted nothing; the resume posted exactly one apply.
    assert result["busyPosts"] == 0
    assert result["urls"] == ["/admin/api/cluster/assistant/apply"]
    assert result["measureCalls"] == 0
    assert result["recommendCalls"] == 0
    assert result["planCalls"] == 0
    assert result["startCalls"] == 0
    # The frozen plan inputs are re-applied verbatim; no re-election.
    body = result["applyBody"]
    assert body["measurement_id"] == "msr-a1b2c3d4"
    assert body["model_path"] == "/models/qwen3-32b-4bit"
    assert body["context_tokens"] == 32768
    assert body["mtp"] is False
    assert body["execution_profile"] == "balanced"
    assert body["plan_signature"] == "3f9a1c7d20b45e68"
    assert body["coordinator_node"] == "node-b"
    assert body["rank_order"] == ["node-b", "node-a"]
    # KNOWN DEFERRED LIMITATION (product): runAssistantResume hardcodes
    # node_selection 'minimal' instead of preserving the session's selection.
    # This pins the current behaviour; preserving it is a later product slice.
    assert body["node_selection"] == "minimal"
    # The card binds the stored model/measurement and rests in applying.
    assert result["state"] == "applying"
    assert result["session"] is None
    assert result["selectedModel"] == "/models/qwen3-32b-4bit"
    assert result["measurementId"] == "msr-a1b2c3d4"


# --- Cluster Assistant: the automatic constellation search --------------------
#
# POST /assistant/search is synchronous and minutes-long: it deploys and
# measures several constellations, keeps the best measured and stops the losers.
# The UI is an action on the plan-stage card (no new state), single-flight and
# generation-guarded, and renders a rate only for a measured constellation.
# These tests execute the shipped component under Node with a fake apiFetch;
# no real cluster, no SSH, no endpoint is ever exposed from a search result.

_SEARCH_RESPONSE = {
    "best": {
        "rank_order": ["node-b", "node-a"],
        "model_path": "/models/qwen3-32b",
        "context_tokens": 32768,
        "plan_signature": "sig-b",
        "node_count": 2,
        "feasible": True,
        "measured": True,
        "decode_tok_per_s": 44.5,
        "prefill_tok_per_s": 900.0,
        "ttft_ms": 120.0,
        "deployment_id": "dep-best",
        "coordinator_node": "node-b",
    },
    "results": [
        {
            "rank_order": ["node-b", "node-a"],
            "model_path": "/models/qwen3-32b",
            "context_tokens": 32768,
            "plan_signature": "sig-b",
            "node_count": 2,
            "feasible": True,
            "measured": True,
            "decode_tok_per_s": 44.5,
            "prefill_tok_per_s": 900.0,
            "ttft_ms": 120.0,
            "deployment_id": "dep-best",
            "coordinator_node": "node-b",
        },
        {
            "rank_order": ["node-a", "node-b"],
            "model_path": "/models/qwen3-32b",
            "context_tokens": 32768,
            "plan_signature": "sig-a",
            "node_count": 2,
            "feasible": True,
            "measured": False,
            "decode_tok_per_s": None,
            "prefill_tok_per_s": None,
            "ttft_ms": None,
            "deployment_id": "dep-loser",
            "coordinator_node": "node-a",
        },
    ],
    "tested": [["node-b", "node-a"], ["node-a", "node-b"]],
    "stopped": [
        {
            "rank_order": ["node-a", "node-b"],
            "model_path": "/models/qwen3-32b",
            "context_tokens": 32768,
            "plan_signature": "sig-a",
            "node_count": 2,
            "feasible": True,
            "measured": False,
            "decode_tok_per_s": None,
            "prefill_tok_per_s": None,
            "ttft_ms": None,
            "deployment_id": "dep-loser",
            "coordinator_node": "node-a",
        },
    ],
    "errors": ["stop sig-a: coordinator unreachable"],
}

_SEARCH_SETUP = """
component.assistant.measurementId = 'm1';
component.assistant.measurementModel = '/models/qwen3-32b';
component.assistant.selectedModel = '/models/qwen3-32b';
component.assistant.selectedContextTokens = 32768;
component.assistant.state = 'planning';
component.assistant.plan = {feasible:true, plan_signature:'sig', coordinator_node:'node-a', rank_order:['node-a','node-b'], context_tokens:32768};
component.modelOptions = [{model_path:'/models/qwen3-32b', id:'qwen', mtp_compatible:true}];
component.selectedModelPath = '/models/qwen3-32b';
component.executionProfile = 'balanced';
const searchResponse = %s;
""" % json.dumps(_SEARCH_RESPONSE)


def _search_flow(extra: str) -> str:
    return _WIZARD_TWO_MACS + _ASSISTANT_TIMER_STUBS + _SEARCH_SETUP + extra


def test_assistant_search_posts_measurement_and_options_and_renders_the_best():
    result = _run_wizard(
        _search_flow(
            """
const calls = [];
component.apiFetch = async (url, options = {}) => {
  const body = options && options.body ? JSON.parse(options.body) : null;
  calls.push({url, method: (options && options.method) || 'GET', body});
  if (url.endsWith('/assistant/search')) return {job_id:'search-1', state:'queued', total:2, poll:'/admin/api/cluster/assistant/search/search-1'};
  if (url.endsWith('/assistant/search/search-1')) return Object.assign({state:'completed', message:'done', done:2, total:2}, searchResponse);
  throw new Error('unexpected URL ' + url);
};
(async () => {
  const available = component.assistantSearchAvailable();
  component.setAssistantSearchPriority('speed');
  component.setAssistantSearchMaxCount(2);
  await component.runAssistantSearch();
  process.stdout.write(JSON.stringify({
    available,
    calls,
    busy: component.assistant.search.busy,
    assistantBusy: component.assistant.busy,
    state: component.assistantState(),
    bestRank: component.assistantSearchRankOrder(component.assistantSearchBest()),
    bestMacs: component.assistantSearchBest().node_count,
    bestMeasured: component.assistantSearchMeasuredLabel(component.assistantSearchBest()),
    results: component.assistantSearchResults().length,
    errors: component.assistantSearchErrors(),
    endpointExposed: component.assistantReady(),
  }));
})();
"""
        )
    )

    assert result["available"] is True
    # Exactly one POST to the search route with the measurement + chosen options
    # (the GET poll(s) that follow are the job status, not a second search).
    posts = [c for c in result["calls"] if c["method"] == "POST"]
    assert len(posts) == 1
    call = posts[0]
    assert call["url"] == "/admin/api/cluster/assistant/search"
    assert call["method"] == "POST"
    body = call["body"]
    assert body["measurement_id"] == "m1"
    assert body["priority"] == "speed"
    assert body["max_count"] == 2
    assert body["context_tokens"] == 32768
    assert body["mtp"] is True
    assert body["execution_profile"] == "balanced"
    # The card rests in the same state; search invents no state-machine value.
    assert result["state"] == "planning"
    assert result["busy"] is False
    assert result["assistantBusy"] is False
    # The best measured constellation is rendered with its measured rates.
    assert result["bestRank"] == "Node B → Node A"
    assert result["bestMacs"] == 2
    assert result["bestMeasured"] == (
        "44.5 tok/s decode · 900.0 tok/s prefill · first token in 120 ms"
    )
    assert result["results"] == 2
    assert result["errors"] == ["stop sig-a: coordinator unreachable"]
    # No endpoint is exposed from a search result (endpoint stays ready-gated).
    assert result["endpointExposed"] is False


def test_assistant_search_busy_guard_prevents_a_double_post():
    result = _run_wizard(
        _search_flow(
            """
let resolveSearch; let searchCalls = 0;
component.apiFetch = (url) => {
  if (url.endsWith('/assistant/search')) {
    searchCalls += 1;
    return new Promise((resolve) => { resolveSearch = resolve; });
  }
  if (url.endsWith('/assistant/search/search-1')) return Promise.resolve(Object.assign({state:'completed', message:'done', done:2, total:2}, searchResponse));
  throw new Error('unexpected URL ' + url);
};
(async () => {
  const first = component.runAssistantSearch();
  // While the first request is in flight a second click must not POST again.
  const second = await component.runAssistantSearch();
  const busyDuring = component.assistant.search.busy;
  const callsDuring = searchCalls;
  resolveSearch({job_id:'search-1', state:'queued', total:2});
  await first;
  process.stdout.write(JSON.stringify({
    callsDuring, busyDuring, second,
    callsAfter: searchCalls,
    busyAfter: component.assistant.search.busy,
    hasResult: Boolean(component.assistant.search.result),
  }));
})();
"""
        )
    )
    assert result["callsDuring"] == 1
    assert result["busyDuring"] is True
    assert result["second"] is None
    assert result["callsAfter"] == 1
    assert result["busyAfter"] is False
    assert result["hasResult"] is True


def test_assistant_search_ignores_a_stale_generation_response():
    result = _run_wizard(
        _search_flow(
            """
let resolveSearch;
component.apiFetch = (url) => {
  if (url.endsWith('/assistant/search')) {
    return new Promise((resolve) => { resolveSearch = resolve; });
  }
  throw new Error('unexpected URL ' + url);
};
(async () => {
  const flight = component.runAssistantSearch();
  // A newer action bumps the generation; the in-flight answer is now stale.
  component.assistant.generation += 1;
  resolveSearch({job_id:'search-1', state:'queued', total:2});
  await flight;
  process.stdout.write(JSON.stringify({
    result: component.assistant.search.result,
    best: component.assistantSearchBest(),
  }));
})();
"""
        )
    )
    assert result["result"] is None
    assert result["best"] is None


def test_assistant_search_unmeasured_rows_show_no_rate():
    result = _run_wizard(
        _search_flow(
            """
component.apiFetch = async (url) => {
  if (url.endsWith('/assistant/search')) return {job_id:'search-1', state:'queued', total:2};
  if (url.endsWith('/assistant/search/search-1')) return Object.assign({state:'completed', message:'done', done:2, total:2}, searchResponse);
  throw new Error('unexpected URL ' + url);
};
(async () => {
  await component.runAssistantSearch();
  const rows = component.assistantSearchResults().map((row) => ({
    measured: row.measured,
    rate: component.assistantSearchRateLabel(row),
    measuredLabel: component.assistantSearchMeasuredLabel(row),
    feasible: component.assistantSearchFeasibleLabel(row),
  }));
  process.stdout.write(JSON.stringify({rows}));
})();
"""
        )
    )
    measured_row, unmeasured_row = result["rows"]
    assert measured_row["measured"] is True
    assert "tok/s" in measured_row["rate"]
    assert measured_row["feasible"] == "Yes"
    # An unmeasured constellation never shows an invented rate.
    assert unmeasured_row["measured"] is False
    assert unmeasured_row["rate"] == "not measured"
    assert unmeasured_row["measuredLabel"] == ""
    assert "tok/s" not in unmeasured_row["rate"]


def test_assistant_search_requires_a_model_bound_measurement():
    result = _run_wizard(
        _search_flow(
            """
let searchCalls = 0;
component.apiFetch = async (url) => {
  if (url.endsWith('/assistant/search')) { searchCalls += 1; return searchResponse; }
  throw new Error('unexpected URL ' + url);
};
(async () => {
  // A hardware-only measurement (no model, no plan) is not searchable.
  component.assistant.measurementModel = '';
  component.assistant.plan = null;
  const available = component.assistantSearchAvailable();
  const returned = await component.runAssistantSearch();
  process.stdout.write(JSON.stringify({available, returned, searchCalls}));
})();
"""
        )
    )
    assert result["available"] is False
    assert result["returned"] is None
    assert result["searchCalls"] == 0


def test_assistant_search_refusal_renders_one_message_and_no_result():
    result = _run_wizard(
        _search_flow(
            """
component.apiFetch = async (url) => {
  if (url.endsWith('/assistant/search')) {
    const error = new Error('no constellation fits');
    error.payload = {code: 'no_feasible_plan', message: 'no constellation fits', action: 'adjust_model_or_context'};
    throw error;
  }
  throw new Error('unexpected URL ' + url);
};
(async () => {
  await component.runAssistantSearch();
  process.stdout.write(JSON.stringify({
    error: component.assistant.search.error,
    result: component.assistant.search.result,
    busy: component.assistant.search.busy,
    state: component.assistantState(),
    assistantError: component.assistant.error,
  }));
})();
"""
        )
    )
    assert result["error"] == "no constellation fits"
    assert result["result"] is None
    assert result["busy"] is False
    # Search reports its own error; it does not flip the state machine to failed.
    assert result["state"] == "planning"
    assert result["assistantError"] is None


# --- Cluster Assistant tuning -------------------------------------------------
#
# AGENTS.md §2 "Tuning is a saved preset, not a re-tune on every Start": /tune is
# a quick per-request sampler sweep on the running cluster, /tune-redeploy is a
# minutes-long background job that redeploys each MTP config and therefore ends
# with the cluster stopped. The card renders the backend's measured numbers only
# and never invents a rate.

_TUNE_RESPONSE = {
    "operation_id": "op-start",
    "model_id": "qwen3-32b",
    "chunk": None,
    "chunk_applicable": False,
    "sampler": {
        "sampler": "balanced",
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 0,
        "quality": 1.0,
        "reasoning_tokens": 40,
        "completion_tokens": 120,
        "tok_per_s": 41.2,
    },
    "mtp": False,
    "mtp_applicable": True,
    "chunk_rows": [],
    "sampler_rows": [],
    "mtp_rates": {"false": 22.1, "true": 31.6},
    "preset_key": "qwen3-32b|node-a+node-b",
}

_REDEPLOY_ROWS = [
    {
        "config": "mtp_off",
        "mtp": False,
        "mtp_depth": None,
        "feasible": True,
        "decode_tok_per_s": 22.07,
        "prefill_tok_per_s": 900.0,
        "ttft_ms": 120.0,
        "operation_id": "op-1",
        "error": "",
    },
    {
        "config": "mtp_on_adaptive",
        "mtp": True,
        "mtp_depth": None,
        "feasible": True,
        "decode_tok_per_s": 31.6,
        "prefill_tok_per_s": 950.0,
        "ttft_ms": 110.0,
        "operation_id": "op-2",
        "error": "",
    },
    {
        # A config that could not be measured is a row, never a rate.
        "config": "mtp_on_depth_3",
        "mtp": True,
        "mtp_depth": 3,
        "feasible": False,
        "decode_tok_per_s": None,
        "prefill_tok_per_s": None,
        "ttft_ms": None,
        "operation_id": "",
        "error": "the winner's rank order/context is infeasible at this config",
    },
]

_REDEPLOY_JOB = {
    "job_id": "redeploy-1",
    "state": "completed",
    "total": 3,
    "done": 3,
    "message": "done",
    "rows": _REDEPLOY_ROWS,
    "best": {"config": "mtp_on_adaptive", "mtp": True, "mtp_depth": None},
    "mtp_applicable": True,
    "preset_key": "qwen3-32b|node-a+node-b",
}

_TUNE_SETUP = """
component.assistant.state = 'ready';
component.assistant.operationId = 'op-start';
component.assistant.applyOperationId = 'op-apply';
component.assistant.endpoint = {ready:true, url:'http://172.16.1.11:8099', port:8099, node:'node-a'};
component.assistant.operation = {
  operation_id: 'op-start', kind: 'start', state: 'ready',
  endpoint: {ready:true, url:'http://172.16.1.11:8099', port:8099, node:'node-a'},
};
component.assistant.selectedModel = '/models/qwen3-32b';
component.assistant.selectedContextTokens = 32768;
component.assistant.plan = {feasible:true, plan_signature:'sig', coordinator_node:'node-a', rank_order:['node-a','node-b'], context_tokens:32768};
component.modelOptions = [{model_path:'/models/qwen3-32b', id:'qwen', mtp_compatible:true}];
component.selectedModelPath = '/models/qwen3-32b';
const tuneResponse = %s;
const redeployRows = %s;
const redeployJob = %s;
""" % (
    json.dumps(_TUNE_RESPONSE),
    json.dumps(_REDEPLOY_ROWS),
    json.dumps(_REDEPLOY_JOB),
)


def _tune_flow(extra: str) -> str:
    return _WIZARD_TWO_MACS + _ASSISTANT_TIMER_STUBS + _TUNE_SETUP + extra


def test_assistant_tune_posts_the_ready_operation_and_renders_measured_knobs():
    result = _run_wizard(
        _tune_flow(
            """
const calls = [];
component.apiFetch = async (url, options = {}) => {
  calls.push({
    url,
    method: (options && options.method) || 'GET',
    body: options && options.body ? JSON.parse(options.body) : null,
  });
  if (url.endsWith('/assistant/tune')) return tuneResponse;
  throw new Error('unexpected URL ' + url);
};
(async () => {
  const available = component.assistantTuneAvailable();
  const returned = await component.runAssistantTune();
  process.stdout.write(JSON.stringify({
    available,
    calls,
    sampler: component.assistantTuneSamplerLabel(),
    chunk: component.assistantTuneChunkLabel(),
    mtp: component.assistantTuneMtpLabel(),
    mtpRates: component.assistantTuneMtpRatesLabel(),
    preset: component.assistantTunePresetLabel(),
    busy: component.assistant.tune.busy,
    assistantBusy: component.assistant.busy,
    state: component.assistantState(),
    ready: component.assistantReady(),
    returnedPreset: returned && returned.preset_key,
  }));
})();
"""
        )
    )

    assert result["available"] is True
    # Exactly one POST, carrying the ready start operation whose deployment is
    # the one being tuned.
    assert len(result["calls"]) == 1
    call = result["calls"][0]
    assert call["url"] == "/admin/api/cluster/assistant/tune"
    assert call["method"] == "POST"
    assert call["body"] == {"operation_id": "op-start"}
    # The sampler is named with the knobs it was measured with.
    assert result["sampler"] == (
        "balanced (temperature 0.6, top_p 0.95, top_k 0) · 41.2 tok/s decode"
    )
    # The chunk is a per-model setting in this build, so it is not applicable —
    # never a fake winner.
    assert result["chunk"] == "not applicable"
    assert result["mtp"] == "off"
    assert result["mtpRates"] == "off 22.1 tok/s · on 31.6 tok/s"
    assert result["preset"] == "qwen3-32b|node-a+node-b"
    assert result["returnedPreset"] == "qwen3-32b|node-a+node-b"
    assert result["busy"] is False
    assert result["assistantBusy"] is False
    # Fine-tune keeps the cluster serving: no state change, endpoint still live.
    assert result["state"] == "ready"
    assert result["ready"] is True


def test_assistant_tune_refusal_keeps_its_message_and_action():
    result = _run_wizard(
        _tune_flow(
            """
component.apiFetch = async (url) => {
  const error = new Error('tuning needs a ready start operation');
  error.status = 409;
  error.payload = {
    code: 'operation_not_ready',
    message: 'tuning needs a ready start operation; run the search and pick a winner first',
    action: 'status',
  };
  throw error;
};
(async () => {
  const returned = await component.runAssistantTune();
  process.stdout.write(JSON.stringify({
    returned,
    error: component.assistant.tune.error,
    action: component.assistant.tune.errorAction,
    actionLabel: component.assistantActionLabel(component.assistant.tune.errorAction),
    result: component.assistant.tune.result,
    busy: component.assistant.tune.busy,
    state: component.assistantState(),
    assistantError: component.assistant.error,
  }));
})();
"""
        )
    )

    assert result["returned"] is None
    assert result["error"] == (
        "tuning needs a ready start operation; run the search and pick a winner first"
    )
    # The refusal's one action is rendered, not swallowed.
    assert result["action"] == "status"
    assert result["actionLabel"]
    assert result["result"] is None
    assert result["busy"] is False
    # A refused tune is not a failed cluster: the card stays on ready.
    assert result["state"] == "ready"
    assert result["assistantError"] is None


def test_assistant_tune_redeploy_polls_to_completion_and_leaves_the_cluster_stopped():
    result = _run_wizard(
        _tune_flow(
            """
const calls = [];
// The sweep takes minutes; run the loop's wait synchronously so the test can
// follow every poll without a real clock.
let progressDuringPoll = '';
global.setTimeout = (fn) => {
  progressDuringPoll = component.assistantTuneRedeployProgressLabel();
  if (typeof fn === 'function') fn();
  return 0;
};
const polls = [
  {
    job_id: 'redeploy-1', state: 'running', total: 3, done: 1,
    message: 'measuring mtp_on_adaptive (2 of 3)',
    rows: [redeployRows[0]], best: null, mtp_applicable: true, preset_key: '',
  },
  redeployJob,
];
component.apiFetch = async (url, options = {}) => {
  calls.push({
    url,
    method: (options && options.method) || 'GET',
    body: options && options.body ? JSON.parse(options.body) : null,
  });
  if (url.endsWith('/assistant/tune-redeploy')) {
    return {job_id:'redeploy-1', state:'queued', total:3, mtp_applicable:true, poll:'/admin/api/cluster/assistant/tune-redeploy/redeploy-1'};
  }
  if (url.endsWith('/assistant/tune-redeploy/redeploy-1')) return polls.shift();
  throw new Error('unexpected URL ' + url);
};
(async () => {
  const available = component.assistantTuneAvailable();
  const returned = await component.runAssistantTuneRedeploy();
  process.stdout.write(JSON.stringify({
    available,
    calls,
    progressDuringPoll,
    progress: component.assistantTuneRedeployProgressLabel(),
    returned,
    state: component.assistant.tuneRedeploy.state,
    rows: component.assistantTuneRedeployRows().map((row) => ({
      config: row.config,
      mtp: component.assistantTuneRedeployMtpLabel(row),
      rate: component.assistantTuneRedeployRateLabel(row),
      error: row.error,
    })),
    best: component.assistantTuneRedeployBestLabel(),
    preset: component.assistant.tuneRedeploy.presetKey,
    stopped: component.assistant.stopped,
    endpoint: component.assistant.endpoint,
    ready: component.assistantReady(),
    tuneAvailable: component.assistantTuneAvailable(),
    busy: component.assistant.tuneRedeploy.busy,
    assistantBusy: component.assistant.busy,
    cardState: component.assistantState(),
  }));
})();
"""
        )
    )

    assert result["available"] is True
    posts = [call for call in result["calls"] if call["method"] == "POST"]
    assert len(posts) == 1
    assert posts[0]["url"] == "/admin/api/cluster/assistant/tune-redeploy"
    assert posts[0]["body"] == {"operation_id": "op-start"}
    gets = [call for call in result["calls"] if call["method"] == "GET"]
    assert [call["url"] for call in gets] == [
        "/admin/api/cluster/assistant/tune-redeploy/redeploy-1",
        "/admin/api/cluster/assistant/tune-redeploy/redeploy-1",
    ]
    # Progress is the backend's own sentence, not an invented percentage.
    assert result["progressDuringPoll"] == "measuring mtp_on_adaptive (2 of 3)"
    assert result["state"] == "completed"
    assert result["returned"]["best"] == {
        "config": "mtp_on_adaptive",
        "mtp": True,
        "mtp_depth": None,
    }
    # Each row shows its real measured decode rate; an unmeasured config says so.
    assert result["rows"] == [
        {
            "config": "mtp_off",
            "mtp": "off",
            "rate": "22.1 tok/s decode",
            "error": "",
        },
        {
            "config": "mtp_on_adaptive",
            "mtp": "adaptive",
            "rate": "31.6 tok/s decode",
            "error": "",
        },
        {
            "config": "mtp_on_depth_3",
            "mtp": "depth 3",
            "rate": "not measured",
            "error": "the winner's rank order/context is infeasible at this config",
        },
    ]
    assert "not measured" not in result["rows"][0]["rate"]
    # The winner is named with the measured rate of its own row.
    assert result["best"] == "mtp_on_adaptive · 31.6 tok/s decode"
    assert result["preset"] == "qwen3-32b|node-a+node-b"
    # The sweep stops every config it measures: the card must not claim the
    # cluster still serves, and Start again is the way back.
    assert result["stopped"] is True
    assert result["endpoint"] is None
    assert result["ready"] is False
    assert result["tuneAvailable"] is False
    assert result["busy"] is False
    assert result["assistantBusy"] is False
    assert result["cardState"] == "ready"


def test_assistant_tune_redeploy_busy_guard_prevents_a_double_job():
    result = _run_wizard(
        _tune_flow(
            """
let resolvePost;
let posts = 0;
component.apiFetch = (url, options = {}) => {
  if (url.endsWith('/assistant/tune-redeploy')) {
    posts += 1;
    return new Promise((resolve) => { resolvePost = resolve; });
  }
  if (url.endsWith('/assistant/tune-redeploy/redeploy-1')) return Promise.resolve(redeployJob);
  throw new Error('unexpected URL ' + url);
};
(async () => {
  const first = component.runAssistantTuneRedeploy();
  // While the POST is in flight a second click must not start a second sweep.
  const second = await component.runAssistantTuneRedeploy();
  const busyDuring = component.assistant.tuneRedeploy.busy;
  const postsDuring = posts;
  resolvePost({job_id:'redeploy-1', state:'queued', total:3, mtp_applicable:true});
  await first;
  process.stdout.write(JSON.stringify({
    second, busyDuring, postsDuring, postsAfter: posts,
    busyAfter: component.assistant.tuneRedeploy.busy,
    state: component.assistant.tuneRedeploy.state,
  }));
})();
"""
        )
    )

    assert result["second"] is None
    assert result["busyDuring"] is True
    assert result["postsDuring"] == 1
    assert result["postsAfter"] == 1
    assert result["busyAfter"] is False
    assert result["state"] == "completed"


def test_assistant_tune_redeploy_ignores_a_stale_generation_response():
    result = _run_wizard(
        _tune_flow(
            """
let resolvePost;
let gets = 0;
component.apiFetch = (url) => {
  if (url.endsWith('/assistant/tune-redeploy')) {
    return new Promise((resolve) => { resolvePost = resolve; });
  }
  if (url.endsWith('/assistant/tune-redeploy/redeploy-1')) { gets += 1; return Promise.resolve(redeployJob); }
  throw new Error('unexpected URL ' + url);
};
(async () => {
  const flight = component.runAssistantTuneRedeploy();
  // A newer job replaces this one; its answer must not land on the card.
  component.assistant.tuneRedeploy.generation += 1;
  resolvePost({job_id:'redeploy-1', state:'queued', total:3, mtp_applicable:true});
  await flight;
  process.stdout.write(JSON.stringify({
    gets,
    rows: component.assistant.tuneRedeploy.rows.length,
    best: component.assistantTuneRedeployBest(),
    jobId: component.assistant.tuneRedeploy.jobId,
  }));
})();
"""
        )
    )

    # The stale job never polled and never rendered a row or a winner.
    assert result["gets"] == 0
    assert result["rows"] == 0
    assert result["best"] is None
    assert result["jobId"] == ""


def test_assistant_tune_redeploy_failure_reports_the_error_and_the_stop():
    result = _run_wizard(
        _tune_flow(
            """
component.apiFetch = async (url) => {
  if (url.endsWith('/assistant/tune-redeploy')) {
    return {job_id:'redeploy-1', state:'queued', total:3, mtp_applicable:true};
  }
  if (url.endsWith('/assistant/tune-redeploy/redeploy-1')) {
    return {job_id:'redeploy-1', state:'failed', total:3, done:1, message:'deploy failed: port 8099 busy', rows:[redeployRows[0]], best:null, mtp_applicable:true, preset_key:''};
  }
  throw new Error('unexpected URL ' + url);
};
(async () => {
  const returned = await component.runAssistantTuneRedeploy();
  process.stdout.write(JSON.stringify({
    returned,
    state: component.assistant.tuneRedeploy.state,
    error: component.assistant.tuneRedeploy.error,
    best: component.assistantTuneRedeployBest(),
    rows: component.assistantTuneRedeployRows().length,
    stopped: component.assistant.stopped,
    busy: component.assistant.tuneRedeploy.busy,
  }));
})();
"""
        )
    )

    assert result["returned"] is None
    assert result["state"] == "failed"
    assert result["error"] == "deploy failed: port 8099 busy"
    # No winner is ever invented for a failed sweep.
    assert result["best"] is None
    assert result["rows"] == 1
    # The sweep already unloaded the cluster before its first config.
    assert result["stopped"] is True
    assert result["busy"] is False


def test_assistant_tune_needs_a_ready_running_operation():
    result = _run_wizard(
        _tune_flow(
            """
let calls = 0;
component.apiFetch = async (url) => { calls += 1; return tuneResponse; };
(async () => {
  // A stopped cluster has nothing to sweep.
  component.assistant.stopped = true;
  const stoppedAvailable = component.assistantTuneAvailable();
  const stoppedTune = await component.runAssistantTune();
  const stoppedRedeploy = await component.runAssistantTuneRedeploy();
  // An unfinished operation is not tunable either.
  component.assistant.stopped = false;
  component.assistant.state = 'planning';
  const planningAvailable = component.assistantTuneAvailable();
  const planningTune = await component.runAssistantTune();
  // Without an operation id there is nothing to name in the request.
  component.assistant.state = 'ready';
  component.assistant.operationId = '';
  const noOperationAvailable = component.assistantTuneAvailable();
  process.stdout.write(JSON.stringify({
    stoppedAvailable, stoppedTune, stoppedRedeploy,
    planningAvailable, planningTune,
    noOperationAvailable,
    calls,
  }));
})();
"""
        )
    )

    assert result["stoppedAvailable"] is False
    assert result["stoppedTune"] is None
    assert result["stoppedRedeploy"] is None
    assert result["planningAvailable"] is False
    assert result["planningTune"] is None
    assert result["noOperationAvailable"] is False
    assert result["calls"] == 0


def test_assistant_tuning_markup_is_ready_gated_and_measured_only():
    template = (ROOT / "omlx_web/templates/dashboard/_cluster_v2.html").read_text()
    javascript = (ROOT / "omlx_web/static/js/cluster_v2.js").read_text()

    start = template.index("<!-- Tuning the winning deployment.")
    block = template[start : template.index("<!-- failed: one message", start)]

    # The block lives on the ready surface only, and every affordance/result
    # carries a stable marker for review and for these tests.
    assert 'x-show="assistantState() === \'ready\'"' in block
    for marker in (
        "data-cluster-v2-assistant-tune>",
        "data-cluster-v2-assistant-tune-redeploy>",
        "data-cluster-v2-assistant-tune-busy",
        "data-cluster-v2-assistant-tune-error",
        "data-cluster-v2-assistant-tune-action",
        "data-cluster-v2-assistant-tune-result",
        "data-cluster-v2-assistant-tune-sampler",
        "data-cluster-v2-assistant-tune-chunk",
        "data-cluster-v2-assistant-tune-mtp",
        "data-cluster-v2-assistant-tune-preset",
        "data-cluster-v2-assistant-tune-redeploy-busy",
        "data-cluster-v2-assistant-tune-redeploy-progress",
        "data-cluster-v2-assistant-tune-redeploy-error",
        "data-cluster-v2-assistant-tune-redeploy-row",
        "data-cluster-v2-assistant-tune-redeploy-best",
        "data-cluster-v2-assistant-tune-redeploy-stopped",
    ):
        assert marker in block, marker

    # A second click cannot double-start while a sweep runs.
    assert block.count("assistantTuneBusy()") >= 2
    # The sweep ends with the cluster stopped: the card says so instead of
    # leaving a live endpoint behind, and the tuning block never links one.
    assert "Start again" in block
    assert "assistant.endpoint" not in block
    # No SSE anywhere, and every rate comes from a measured field.
    assert "EventSource" not in javascript
    assert "'not applicable'" in javascript
    assert "'not measured'" in javascript


def test_assistant_tuning_endpoints_and_poll_are_declared_and_generation_guarded():
    javascript = (ROOT / "omlx_web/static/js/cluster_v2.js").read_text()

    assert "tune: '/admin/api/cluster/assistant/tune'" in javascript
    assert "tuneRedeploy: '/admin/api/cluster/assistant/tune-redeploy'" in javascript
    assert "/admin/api/cluster/assistant/tune-redeploy/${encodeURIComponent(jobId)}" in javascript

    # One job, one loop: the poll applies a snapshot only for its own
    # generation and stops at completed/failed (pollStagingJob's rule).
    poll = javascript.split("async pollAssistantTuneRedeploy(", 1)[1].split(
        "applyAssistantTuneRedeploy(snapshot) {", 1
    )[0]
    assert "generation !== job.generation" in poll
    assert "job.state === 'completed'" in poll
    assert "job.state === 'failed'" in poll
    assert "await new Promise" in poll
    assert "EventSource" not in poll

    # A refusal keeps its action, and the buttons are busy- and stop-guarded.
    tune = javascript.split("async runAssistantTune(", 1)[1].split(
        "async runAssistantTuneRedeploy(", 1
    )[0]
    assert "assistantTuneBusy()" in tune
    assert "assistantRunning()" in tune
    assert "refusal.action" in tune
    redeploy = javascript.split("async runAssistantTuneRedeploy(", 1)[1].split(
        "async pollAssistantTuneRedeploy(", 1
    )[0]
    assert "assistantTuneBusy()" in redeploy
    assert "assistantRunning()" in redeploy
    assert "refusal.action" in redeploy


def test_the_user_can_pick_the_experts_resident_tradeoff_for_apply():
    """Decision 2026-11-14: choosing the speed alternative applies *its* plan.

    The card offers the experts-resident plan the backend returned as a tradeoff;
    selecting it must send that plan's signature, coordinator and levers, not the
    quality plan's.
    """

    result = _run_wizard(
        _assistant_flow(
            """
(async () => {
  await reachPlan();
  component.assistant.plan.tradeoffs = [{
    label: 'speed', context_tokens: 16384,
    plan_signature: 'speedalt0000000000000000000000',
    coordinator_node: 'node-a', rank_order: ['node-a', 'node-b'],
    resource_policy: {
      context_tokens: 16384, target_context_tokens: 32768,
      table_offload: 'ple', table_kind: 'ple',
      moe_expert_offload: false, moe_resident_fraction: 1,
      memory_guard_tier: null, admission: 'p50',
    },
  }];
  const tradeoffsShown = component.assistantTradeoffs().length;
  component.assistant.selectedTradeoff = component.assistant.plan.tradeoffs[0];
  const selected = component.assistantTradeoffSelected();
  await component.runAssistantApply();
  process.stdout.write(JSON.stringify({
    applyBody: calls.find((c) => c.url.endsWith('/assistant/apply')).body,
    tradeoffsShown, selected,
  }));
})();
"""
        )
    )

    body = result["applyBody"]
    assert result["tradeoffsShown"] == 1
    assert result["selected"] is True
    assert body["plan_signature"] == "speedalt0000000000000000000000"
    assert body["coordinator_node"] == "node-a"
    assert body["rank_order"] == ["node-a", "node-b"]
    assert body["context_tokens"] == 16384
    assert body["moe_expert_offload"] is False
    assert body["table_offload"] == "ple"


# --- Run on this Mac: single-node inference, no cluster ----------------------
#
# PHASE-2B-RUN.md §22: a model that fits on one Mac is faster served by that
# Mac's own engine than by a pipeline. The recommendation names it
# (`single_node`), and the card offers one action that POSTs an async job. The
# browser never launches anything, and the endpoint/chat link stay gated on the
# job's own ready answer.


def test_assistant_run_on_this_mac_posts_the_job_and_reaches_ready():
    result = _run_wizard(
        _assistant_flow(
            """
(async () => {
  await reachPlan();
  const calls = [];
  const polls = [
    {
      job_id: 'local-1', state: 'running',
      message: 'applying the saved tuning preset before the load',
      model_id: 'qwen', context_tokens: 32768, mtp: false,
      endpoint: null, warnings: [], error: null,
    },
    {
      job_id: 'local-1', state: 'ready',
      message: 'qwen is ready on this Mac',
      model_id: 'qwen', context_tokens: 32768, mtp: false,
      endpoint: {
        host: '192.168.1.10', port: 8099,
        url: 'http://192.168.1.10:8099', ready: true,
      },
      warnings: [{
        code: 'tuning_preset_applied',
        message: 'the saved tuning preset was applied before the load: mtp_adaptive_max_depth',
      }],
      error: null,
    },
  ];
  let progressDuringPoll = '';
  global.setTimeout = (fn) => {
    progressDuringPoll = component.assistantLocalRunProgressLabel();
    if (typeof fn === 'function') fn();
    return 0;
  };
  component.apiFetch = async (url, options = {}) => {
    calls.push({
      url,
      method: (options && options.method) || 'GET',
      body: options && options.body ? JSON.parse(options.body) : null,
    });
    if (url.endsWith('/assistant/run-local')) {
      return {job_id: 'local-1', state: 'queued', poll: '/admin/api/cluster/assistant/run-local/local-1'};
    }
    if (url.endsWith('/assistant/run-local/local-1')) return polls.shift();
    throw new Error('unexpected URL ' + url);
  };
  const available = component.assistantLocalRunAvailable();
  const returned = await component.runAssistantLocal();
  process.stdout.write(JSON.stringify({
    available,
    calls,
    progressDuringPoll,
    returned,
    state: component.assistant.localRun.state,
    modelId: component.assistant.localRun.modelId,
    endpoint: component.assistantLocalEndpointUrl(),
    chat: component.assistantLocalChatUrl(),
    clusterChat: component.assistantChatUrl(),
    warnings: component.assistantLocalRunWarnings(),
    cardState: component.assistantState(),
    busy: component.assistant.localRun.busy,
    assistantBusy: component.assistant.busy,
  }));
})();
"""
        )
    )

    assert result["available"] is True
    posts = [call for call in result["calls"] if call["method"] == "POST"]
    assert len(posts) == 1
    assert posts[0]["url"] == "/admin/api/cluster/assistant/run-local"
    # The single-node answer's own context and layer count are what is asserted
    # to this Mac; the measurement is the model-bound one (m2), never the entry.
    assert posts[0]["body"] == {
        "measurement_id": "m2",
        "model_path": "/models/qwen3-32b-4bit",
        "layer_count": 80,
        "context_tokens": 32768,
        "mtp": False,
    }
    gets = [call for call in result["calls"] if call["method"] == "GET"]
    assert [call["url"] for call in gets] == [
        "/admin/api/cluster/assistant/run-local/local-1",
        "/admin/api/cluster/assistant/run-local/local-1",
    ]
    # Progress is the backend's sentence, never an invented percentage.
    assert result["progressDuringPoll"] == (
        "applying the saved tuning preset before the load"
    )
    assert result["state"] == "ready"
    assert result["returned"]["state"] == "ready"
    assert result["modelId"] == "qwen"
    assert result["endpoint"] == "http://192.168.1.10:8099"
    # Screen 5, model-only: this Mac serves, so the link is this Mac's own page.
    assert result["chat"] == "/chat?model=qwen"
    # The cluster endpoint/chat stay untouched: run-local is not a cluster ready.
    assert result["clusterChat"] == ""
    assert result["warnings"] == [
        "the saved tuning preset was applied before the load: mtp_adaptive_max_depth"
    ]
    # No new assistant state-machine value: the card rests where it was.
    assert result["cardState"] == "planning"
    assert result["busy"] is False
    assert result["assistantBusy"] is False


def test_assistant_run_on_this_mac_is_only_offered_for_this_mac_and_guarded():
    result = _run_wizard(
        _assistant_flow(
            """
let postedCalls = 0;
(async () => {
  await reachPlan();
  component.apiFetch = async () => { postedCalls += 1; return {}; };
  const offeredForSelf = component.assistantLocalRunAvailable();
  // The best single node may be another Mac; this flow is local by contract.
  component.assistant.recommendation.single_node.rank0_node = 'node-b';
  const offeredForOther = component.assistantLocalRunAvailable();
  const postedForOther = await component.runAssistantLocal();
  component.assistant.recommendation.single_node.rank0_node = 'node-a';
  // A pipeline answer is not one Mac and is never offered here.
  component.assistant.recommendation.single_node.nodes_required = 2;
  const offeredForPipeline = component.assistantLocalRunAvailable();
  component.assistant.recommendation.single_node.nodes_required = 1;
  // A second click must not start a second load on this Mac.
  component.assistant.localRun.busy = true;
  const postedWhileBusy = await component.runAssistantLocal();
  component.assistant.localRun.busy = false;
  process.stdout.write(JSON.stringify({
    offeredForSelf, offeredForOther, postedForOther,
    offeredForPipeline, postedWhileBusy, calls: postedCalls,
  }));
})();
"""
        )
    )

    assert result == {
        "offeredForSelf": True,
        "offeredForOther": False,
        "postedForOther": None,
        "offeredForPipeline": False,
        "postedWhileBusy": None,
        "calls": 0,
    }


def test_assistant_run_on_this_mac_failure_keeps_the_one_action():
    result = _run_wizard(
        _assistant_flow(
            """
(async () => {
  await reachPlan();
  global.setTimeout = () => 0;
  component.apiFetch = async (url) => {
    if (url.endsWith('/assistant/run-local')) {
      return {job_id: 'local-2', state: 'queued', poll: '/admin/api/cluster/assistant/run-local/local-2'};
    }
    return {
      job_id: 'local-2', state: 'failed',
      message: 'only 2,048 tokens of context fit on node-a alone, below the requested 32,768; lower the context or choose a smaller model',
      model_id: 'qwen', context_tokens: 0, mtp: false,
      endpoint: null, warnings: [],
      error: {
        code: 'context_too_large',
        message: 'only 2,048 tokens of context fit on node-a alone, below the requested 32,768; lower the context or choose a smaller model',
        action: 'adjust_model_or_context',
      },
    };
  };
  const returned = await component.runAssistantLocal();
  process.stdout.write(JSON.stringify({
    returned,
    state: component.assistant.localRun.state,
    error: component.assistant.localRun.error,
    action: component.assistant.localRun.errorAction,
    endpoint: component.assistantLocalEndpointUrl(),
    chat: component.assistantLocalChatUrl(),
    clusterChat: component.assistantChatUrl(),
  }));
})();
"""
        )
    )

    assert result["returned"] is None
    assert result["state"] == "failed"
    assert result["action"] == "adjust_model_or_context"
    assert "below the requested 32,768" in result["error"]
    # A failed run never shows an endpoint or a chat link.
    assert result["endpoint"] == ""
    assert result["chat"] == ""
    assert result["clusterChat"] == ""


def test_assistant_local_run_markup_and_poll_are_declared_and_ready_gated():
    template = (ROOT / "omlx_web/templates/dashboard/_cluster_v2.html").read_text()
    javascript = (ROOT / "omlx_web/static/js/cluster_v2.js").read_text()

    start = template.index("<!-- Run on this Mac: single-node inference")
    block = template[start : template.index("<!-- Search: the automatic", start)]
    for marker in (
        "data-cluster-v2-assistant-local-run>",
        "data-cluster-v2-assistant-local-run-button",
        "data-cluster-v2-assistant-local-run-busy",
        "data-cluster-v2-assistant-local-run-progress",
        "data-cluster-v2-assistant-local-run-error",
        "data-cluster-v2-assistant-local-run-action",
        "data-cluster-v2-assistant-local-run-warning",
        "data-cluster-v2-assistant-local-run-endpoint",
        "data-cluster-v2-assistant-local-run-endpoint-url",
        "data-cluster-v2-assistant-local-run-open-chat",
        "data-cluster-v2-assistant-local-run-model",
        "data-cluster-v2-assistant-local-run-context",
    ):
        assert marker in block, marker
    # The button is only offered for this Mac's own single-node answer, and the
    # endpoint + chat link live inside a ready-gated container.
    assert "assistantLocalRunAvailable()" in block
    assert "assistantLocalRunVisible()" in block
    assert 'x-show="assistantLocalEndpointUrl()"' in block
    assert "openAssistantLocalChat()" in block
    # No token rate or percentage is rendered by this block.
    assert "tok/s" not in block
    assert "%" not in block

    assert "runLocal: '/admin/api/cluster/assistant/run-local'" in javascript
    assert "/admin/api/cluster/assistant/run-local/${encodeURIComponent(jobId)}" in javascript

    # One job, one loop, generation-guarded; it stops at ready/failed and never
    # opens a stream.
    poll = javascript.split("async pollAssistantLocalRun(", 1)[1].split(
        "applyAssistantLocalRun(snapshot) {", 1
    )[0]
    assert "generation !== job.generation" in poll
    assert "job.state === 'ready'" in poll
    assert "job.state === 'failed'" in poll
    assert "await new Promise" in poll
    assert "EventSource" not in poll

    run = javascript.split("async runAssistantLocal(", 1)[1].split(
        "async pollAssistantLocalRun(", 1
    )[0]
    assert "assistantRunning()" in run
    assert "assistant.localRun.busy" in run
    assert "refusal.action" in run
    # The browser asks only the assistant route: it never deploys, stages,
    # launches or dials another Mac.
    assert "/admin/api/cluster/deployments" not in run
    assert "EventSource" not in javascript


# --- Run on this Mac: Stop/Unload + strict readiness -------------------------
#
# PHASE-2B-RUN.md §28 recorded that a local run had no stop/unload affordance:
# the engine kept serving until the model unloaded. Stop is the job's own stop
# route (idempotent on the backend), posted once, and followed by one re-read so
# the card never claims "stopped" without the job saying so. The endpoint and
# Open Chat stay gated on state === 'ready' *and* endpoint.ready === true.


def test_assistant_local_run_stop_posts_once_and_retires_the_serving_surface():
    result = _run_wizard(
        _assistant_flow(
            """
(async () => {
  await reachPlan();
  const notifications = [];
  component.notify = (type, message) => notifications.push({type, message});
  const calls = [];
  let stopPosts = 0;
  const ready = {
    job_id: 'local-3', state: 'ready',
    message: 'qwen is ready on this Mac (readiness check passed)',
    model_id: 'qwen', context_tokens: 32768, mtp: false,
    endpoint: {
      host: '127.0.0.1', port: 8099,
      url: 'http://127.0.0.1:8099', ready: true,
    },
    warnings: [], error: null,
  };
  const stopped = {
    job_id: 'local-3', state: 'failed',
    message: 'the local run was stopped on this Mac',
    model_id: 'qwen', context_tokens: 0, mtp: false,
    endpoint: null, warnings: [],
    error: {
      code: 'stopped',
      message: 'the local run was stopped on this Mac',
      action: '',
    },
  };
  const polls = [
    {
      job_id: 'local-3', state: 'running',
      message: 'applying the saved tuning preset before the load',
      endpoint: null, warnings: [], error: null,
    },
    ready,
  ];
  global.setTimeout = (fn) => {
    if (typeof fn === 'function') fn();
    return 0;
  };
  component.apiFetch = async (url, options = {}) => {
    calls.push({url, method: (options && options.method) || 'GET'});
    if (url.endsWith('/assistant/run-local')) {
      return {job_id: 'local-3', state: 'queued'};
    }
    if (url.endsWith('/assistant/run-local/local-3/stop')) {
      stopPosts += 1;
      return {job_id: 'local-3', stopped: true};
    }
    if (url.endsWith('/assistant/run-local/local-3')) {
      return polls.length ? polls.shift() : stopped;
    }
    throw new Error('unexpected URL ' + url);
  };
  const ran = await component.runAssistantLocal();
  const before = {
    state: component.assistant.localRun.state,
    endpoint: component.assistantLocalEndpointUrl(),
    chat: component.assistantLocalChatUrl(),
    stoppable: component.assistantLocalRunStoppable(),
  };
  const stopResult = await component.stopAssistantLocalRun();
  const after = {
    state: component.assistant.localRun.state,
    stopped: component.assistant.localRun.stopped,
    stopping: component.assistant.localRun.stopping,
    endpoint: component.assistantLocalEndpointUrl(),
    chat: component.assistantLocalChatUrl(),
    stoppable: component.assistantLocalRunStoppable(),
    error: component.assistant.localRun.error,
    busy: component.assistant.localRun.busy,
    assistantBusy: component.assistant.busy,
  };
  // A second click must not post the same stop again.
  const again = await component.stopAssistantLocalRun();
  // A snapshot that was already in flight cannot resurrect the stopped job.
  component.applyAssistantLocalRun(ready);
  process.stdout.write(JSON.stringify({
    ran, before, stopResult, after, again, stopPosts,
    calls, notifications,
    afterLateSnapshot: component.assistantLocalEndpointUrl(),
  }));
})();
"""
        )
    )

    assert result["ran"]["state"] == "ready"
    assert result["before"] == {
        "state": "ready",
        "endpoint": "http://127.0.0.1:8099",
        "chat": "/chat?model=qwen",
        "stoppable": True,
    }
    # Exactly one stop POST, to the job's own route, and exactly one re-read.
    posts = [call for call in result["calls"] if call["method"] == "POST"]
    assert [(call["url"], call["method"]) for call in posts] == [
        (
            "/admin/api/cluster/assistant/run-local",
            "POST",
        ),
        (
            "/admin/api/cluster/assistant/run-local/local-3/stop",
            "POST",
        ),
    ]
    assert result["stopPosts"] == 1
    assert result["stopResult"] == {"job_id": "local-3", "stopped": True}
    # The serving surface is retired: no endpoint, no chat link, no error.
    assert result["after"] == {
        "state": "",
        "stopped": True,
        "stopping": False,
        "endpoint": "",
        "chat": "",
        "stoppable": False,
        "error": "",
        "busy": False,
        "assistantBusy": False,
    }
    assert result["again"] is None
    assert result["stopPosts"] == 1
    assert result["afterLateSnapshot"] == ""
    assert [entry["type"] for entry in result["notifications"]] == ["success"]


def test_assistant_local_run_stop_failure_keeps_serving_and_the_one_action():
    result = _run_wizard(
        _assistant_flow(
            """
(async () => {
  await reachPlan();
  const notifications = [];
  component.notify = (type, message) => notifications.push({type, message});
  const calls = [];
  let stopPosts = 0;
  component.apiFetch = async (url, options = {}) => {
    calls.push({url, method: (options && options.method) || 'GET'});
    if (url.endsWith('/assistant/run-local/local-6/stop')) {
      stopPosts += 1;
      throw {
        status: 502,
        payload: {
          message: 'the engine pool did not answer the unload',
          action: 'diagnostics',
        },
      };
    }
    throw new Error('unexpected URL ' + url);
  };
  component.applyAssistantLocalRun({
    job_id: 'local-6', state: 'ready',
    message: 'qwen is ready on this Mac',
    model_id: 'qwen', context_tokens: 32768, mtp: false,
    endpoint: {
      host: '127.0.0.1', port: 8099,
      url: 'http://127.0.0.1:8099', ready: true,
    },
    warnings: [], error: null,
  });
  const stopResult = await component.stopAssistantLocalRun();
  process.stdout.write(JSON.stringify({
    stopResult, stopPosts, calls,
    state: component.assistant.localRun.state,
    endpoint: component.assistantLocalEndpointUrl(),
    chat: component.assistantLocalChatUrl(),
    error: component.assistant.localRun.error,
    action: component.assistant.localRun.errorAction,
    stopping: component.assistant.localRun.stopping,
    stoppable: component.assistantLocalRunStoppable(),
    notifications,
  }));
})();
"""
        )
    )

    assert result["stopResult"] is None
    assert result["stopPosts"] == 1
    assert [call["method"] for call in result["calls"]] == ["POST"]
    # The engine is still loaded, so the card keeps the endpoint; the failure
    # keeps exactly one action and Stop stays the retry.
    assert result["state"] == "ready"
    assert result["endpoint"] == "http://127.0.0.1:8099"
    assert result["chat"] == "/chat?model=qwen"
    assert result["error"] == "the engine pool did not answer the unload"
    assert result["action"] == "diagnostics"
    assert result["stopping"] is False
    assert result["stoppable"] is True
    assert [entry["type"] for entry in result["notifications"]] == ["error"]


def test_assistant_local_run_endpoint_and_chat_need_ready_and_endpoint_ready():
    result = _run_wizard(
        _assistant_flow(
            """
(async () => {
  await reachPlan();
  component.apiFetch = async () => ({});
  const endpoint = {
    host: '127.0.0.1', port: 8099,
    url: 'http://127.0.0.1:8099', ready: true,
  };
  // A load that is still running never shows a URL, even if the endpoint block
  // already says ready: the readiness canary decides what "ready" means.
  component.applyAssistantLocalRun({
    job_id: 'local-7', state: 'running',
    message: 'checking the endpoint with one tiny completion',
    model_id: 'qwen', context_tokens: 32768,
    endpoint, warnings: [], error: null,
  });
  const running = {
    message: component.assistantLocalRunProgressLabel(),
    busy: component.assistantLocalRunBusy(),
    endpoint: component.assistantLocalEndpointUrl(),
    chat: component.assistantLocalChatUrl(),
  };
  // state === 'ready' is not enough while the job's own endpoint is unverified.
  component.applyAssistantLocalRun({
    job_id: 'local-7', state: 'ready',
    message: 'qwen loaded',
    model_id: 'qwen', context_tokens: 32768,
    endpoint: Object.assign({}, endpoint, {ready: false}),
    warnings: [], error: null,
  });
  const readyButUnverified = {
    busy: component.assistantLocalRunBusy(),
    endpoint: component.assistantLocalEndpointUrl(),
    chat: component.assistantLocalChatUrl(),
  };
  component.applyAssistantLocalRun({
    job_id: 'local-7', state: 'ready',
    message: 'qwen is ready on this Mac',
    model_id: 'qwen', context_tokens: 32768,
    endpoint, warnings: [], error: null,
  });
  const ready = {
    busy: component.assistantLocalRunBusy(),
    endpoint: component.assistantLocalEndpointUrl(),
    chat: component.assistantLocalChatUrl(),
  };
  process.stdout.write(JSON.stringify({running, readyButUnverified, ready}));
})();
"""
        )
    )

    # The in-flight message is the backend's canary sentence, never invented.
    assert result["running"] == {
        "message": "checking the endpoint with one tiny completion",
        "busy": True,
        "endpoint": "",
        "chat": "",
    }
    assert result["readyButUnverified"] == {"busy": False, "endpoint": "", "chat": ""}
    assert result["ready"] == {
        "busy": False,
        "endpoint": "http://127.0.0.1:8099",
        "chat": "/chat?model=qwen",
    }


def test_assistant_local_run_stop_during_load_retires_the_poll_loop():
    result = _run_wizard(
        _assistant_flow(
            """
const flush = () => new Promise((resolve) => setImmediate(resolve));
(async () => {
  await reachPlan();
  component.notify = () => {};
  const calls = [];
  const timers = [];
  let resolveGet = null;
  let openedGets = 0;
  let stopPosts = 0;
  // The poll loop waits between snapshots; capture the timer so the test can
  // decide when the loop wakes up again.
  global.setTimeout = (fn) => {
    timers.push(fn);
    return 0;
  };
  component.apiFetch = async (url, options = {}) => {
    calls.push({url, method: (options && options.method) || 'GET'});
    if (url.endsWith('/assistant/run-local')) {
      return {job_id: 'local-4', state: 'queued'};
    }
    if (url.endsWith('/assistant/run-local/local-4/stop')) {
      stopPosts += 1;
      return {job_id: 'local-4', stopped: true};
    }
    if (url.endsWith('/assistant/run-local/local-4')) {
      openedGets += 1;
      if (openedGets === 1) {
        return new Promise((resolve) => { resolveGet = resolve; });
      }
      return {
        job_id: 'local-4', state: 'failed',
        message: 'the local run was stopped on this Mac',
        endpoint: null, warnings: [], error: null,
      };
    }
    throw new Error('unexpected URL ' + url);
  };
  const running = component.runAssistantLocal();
  await flush();   // the POST resolved and the first snapshot is in flight
  resolveGet({
    job_id: 'local-4', state: 'running',
    message: 'loading the model on this Mac',
    model_id: 'qwen', context_tokens: 32768,
    endpoint: null, warnings: [], error: null,
  });
  await flush();   // applied; the loop now waits for the next snapshot
  const duringLoad = {
    label: component.assistantLocalRunProgressLabel(),
    busy: component.assistantLocalRunBusy(),
    stoppable: component.assistantLocalRunStoppable(),
  };
  const stopResult = await component.stopAssistantLocalRun();
  const timersBeforeRelease = timers.length;
  if (timers.length) timers[0]();   // let the suspended loop wake up
  await running;
  process.stdout.write(JSON.stringify({
    duringLoad, stopResult, stopPosts, openedGets,
    timersBeforeRelease,
    state: component.assistant.localRun.state,
    stopped: component.assistant.localRun.stopped,
    endpoint: component.assistantLocalEndpointUrl(),
    localBusy: component.assistant.localRun.busy,
    assistantBusy: component.assistant.busy,
    calls,
  }));
})();
"""
        )
    )

    # The load is narrated while it runs, and it is stoppable before it is ready.
    assert result["duringLoad"] == {
        "label": "loading the model on this Mac",
        "busy": True,
        "stoppable": True,
    }
    assert result["stopResult"] == {"job_id": "local-4", "stopped": True}
    assert result["stopPosts"] == 1
    assert result["timersBeforeRelease"] >= 1
    # The loop read one snapshot and the stop's re-read once; the retired job
    # never issues another snapshot request.
    assert result["openedGets"] == 2
    assert [call["url"] for call in result["calls"] if call["method"] == "POST"] == [
        "/admin/api/cluster/assistant/run-local",
        "/admin/api/cluster/assistant/run-local/local-4/stop",
    ]
    assert result["state"] == ""
    assert result["stopped"] is True
    assert result["endpoint"] == ""
    # A retired load never freezes the card's busy flags.
    assert result["localBusy"] is False
    assert result["assistantBusy"] is False


def test_assistant_local_run_stop_markup_and_wiring_are_declared():
    template = (ROOT / "omlx_web/templates/dashboard/_cluster_v2.html").read_text()
    javascript = (ROOT / "omlx_web/static/js/cluster_v2.js").read_text()

    start = template.index("<!-- Run on this Mac: single-node inference")
    block = template[start : template.index("<!-- Search: the automatic", start)]
    for marker in (
        "data-cluster-v2-assistant-local-run-stop",
        "data-cluster-v2-assistant-local-run-stopped",
        "data-cluster-v2-assistant-local-run-busy",
        "data-cluster-v2-assistant-local-run-progress",
    ):
        assert marker in block, marker
    # One button, disabled while the stop is in flight, and only offered for a
    # job that holds this Mac's engine.
    assert 'x-show="assistantLocalRunStoppable()"' in block
    assert ':disabled="assistant.localRun.stopping"' in block
    assert "stopAssistantLocalRun()" in block
    # The load spinner follows the job's own state, not just the promise.
    assert 'x-show="assistantLocalRunBusy()"' in block
    # The endpoint + chat link stay inside the ready-gated container.
    assert 'x-show="assistantLocalEndpointUrl()"' in block
    # Still no invented progress and still no input in the card block.
    assert "%" not in block
    assert "tok/s" not in block
    assert "<input" not in block

    # The stop route lives in the API map, built from the job id.
    api_map = javascript.split("const ASSISTANT_API = {", 1)[1].split("};", 1)[0]
    assert "runLocalStop" in api_map
    assert "/run-local/${encodeURIComponent(jobId)}/stop" in api_map

    stop = javascript.split("async stopAssistantLocalRun(", 1)[1].split(
        "retireAssistantLocalRun() {", 1
    )[0]
    # Single-flight: one guard, one POST, one re-read, no stream.
    assert "assistantLocalRunStoppable()" in stop
    assert stop.count("method: 'POST'") == 1
    assert stop.count("runLocalStop") == 1
    assert "runLocalStatus" in stop
    assert "retireAssistantLocalRun()" in stop
    assert "refusal.action" in stop
    assert "EventSource" not in stop
    # The browser asks only the assistant route; it never unloads a deployment
    # or dials another Mac itself.
    assert "/admin/api/cluster/deployments" not in stop

    # Strict ready gating: both the state and the job's own endpoint answer.
    gating = javascript.split("assistantLocalEndpointUrl() {", 1)[1].split(
        "},", 1
    )[0]
    assert "job.state !== 'ready'" in gating
    assert "job.endpoint?.ready !== true" in gating

    # The poll loop retires on a stop instead of resurrecting the job.
    poll = javascript.split("async pollAssistantLocalRun(", 1)[1].split(
        "applyAssistantLocalRun(snapshot) {", 1
    )[0]
    assert "job.stopping || job.stopped" in poll
    apply = javascript.split("applyAssistantLocalRun(snapshot) {", 1)[1].split(
        "// Stop/Unload for", 1
    )[0]
    assert "if (job.stopped) return;" in apply


def test_assistant_local_tune_posts_once_and_tracks_the_winner():
    """Tune on this Mac: one POST for a ready local run, then poll the winner."""

    result = _run_wizard(
        _assistant_flow(
            """
(async () => {
  // A ready local run is what tune-local needs; set it directly.
  component.assistant.localRun.state = 'ready';
  component.assistant.localRun.jobId = 'local-1';
  const polls = [
    { tune_job_id: 'lt-1', state: 'running', total: 3, done: 1,
      message: 'measuring mtp_off (1 of 3)',
      rows: [{ config: 'mtp_off', feasible: true, decode_tok_per_s: 22.0 }],
      best: null, sampler: null, preset_key: '', error: null },
    { tune_job_id: 'lt-1', state: 'completed', total: 3, done: 3, message: 'done',
      rows: [
        { config: 'mtp_off', feasible: true, decode_tok_per_s: 22.0 },
        { config: 'mtp_on_depth_3', feasible: true, decode_tok_per_s: 31.6 },
      ],
      best: { config: 'mtp_on_depth_3', mtp: true, mtp_depth: 3 },
      sampler: { sampler: 'balanced' }, preset_key: 'abc123',
      mtp_applicable: true, error: null },
  ];
  global.setTimeout = (fn) => { if (typeof fn === 'function') fn(); return 0; };
  const calls = [];
  component.apiFetch = async (url, options = {}) => {
    calls.push({
      url,
      method: (options && options.method) || 'GET',
      body: options && options.body ? JSON.parse(options.body) : null,
    });
    if (url.endsWith('/assistant/tune-local')) {
      return { tune_job_id: 'lt-1', state: 'queued', total: 3, mtp_applicable: true };
    }
    if (url.endsWith('/assistant/tune-local/lt-1')) return polls.shift();
    throw new Error('unexpected URL ' + url);
  };
  const tunable = component.assistantLocalTunable();
  const returned = await component.runAssistantTuneLocal();
  process.stdout.write(JSON.stringify({
    tunable, calls, returned,
    state: component.assistant.localTune.state,
    rows: component.assistantLocalTuneRows(),
    best: component.assistantLocalTuneBestLabel(),
    preset: component.assistant.localTune.presetKey,
    busy: component.assistant.localTune.busy,
  }));
})();
"""
        )
    )

    assert result["tunable"] is True
    posts = [c for c in result["calls"] if c["method"] == "POST"]
    assert len(posts) == 1
    assert posts[0]["url"] == "/admin/api/cluster/assistant/tune-local"
    assert posts[0]["body"] == {"job_id": "local-1"}
    gets = [c["url"] for c in result["calls"] if c["method"] == "GET"]
    assert gets == ["/admin/api/cluster/assistant/tune-local/lt-1"] * 2
    assert result["state"] == "completed"
    assert [row["config"] for row in result["rows"]] == ["mtp_off", "mtp_on_depth_3"]
    assert result["best"] == "MTP on, depth 3"
    assert result["preset"] == "abc123"
    assert result["busy"] is False


def test_assistant_local_tune_markup_and_wiring_are_declared():
    javascript = (ROOT / "omlx_web/static/js/cluster_v2.js").read_text()
    template = (ROOT / "omlx_web/templates/dashboard/_cluster_v2.html").read_text()

    assert "tuneLocal: '/admin/api/cluster/assistant/tune-local'" in javascript
    assert (
        "/admin/api/cluster/assistant/tune-local/${encodeURIComponent(tuneJobId)}"
        in javascript
    )
    for need in (
        "runAssistantTuneLocal(",
        "pollAssistantTuneLocal(",
        "assistantLocalTunable(",
        "assistantLocalTuneBestLabel(",
    ):
        assert need in javascript, need
    for marker in (
        "data-cluster-v2-assistant-tune-local",
        "data-cluster-v2-assistant-tune-local-button",
        "data-cluster-v2-assistant-tune-local-rows",
        "data-cluster-v2-assistant-tune-local-best",
    ):
        assert marker in template, marker


# --- Assistant UI spec 2026-11-16 (U1-U6) ------------------------------------
#
# The screens are a render-mode on the frozen state machine. These tests drive
# the shipped methods with the same fake seams as the rest of this file: no
# real Mac, model or endpoint appears here (AGENTS: live evidence lives in
# PHASE-2B-RUN.md).


def test_assistant_screen_is_exactly_one_per_backend_state():
    result = _run_wizard(
        """
(() => {
  const out = {};
  const set = (state, extra = {}) => {
    Object.assign(component.assistant, {
      state, plan: null, stopped: false, busy: false,
    }, extra);
    component.assistantChooseOpen = false;
    return component.assistantScreen();
  };
  out.idle = set('idle');
  out.measuring = set('measuring');
  out.recommending = set('recommending');
  out.measuringModel = set('measuring_model');
  out.planningWait = set('planning');
  out.planningPlan = set('planning', {plan: {feasible: true}});
  out.applying = set('applying');
  out.starting = set('starting');
  out.loading = set('loading');
  out.ready = set('ready');
  out.stopped = set('ready', {stopped: true});
  out.failed = set('failed');
  component.assistant.plan = {feasible: true};
  component.assistantChooseOpen = true;
  out.choose = component.assistantScreen();
  process.stdout.write(JSON.stringify(out));
})();
"""
    )

    assert result == {
        "idle": "start",
        "measuring": "check",
        "recommending": "check",
        "measuringModel": "check",
        "planningWait": "check",
        "planningPlan": "plan",
        "applying": "launch",
        "starting": "launch",
        "loading": "launch",
        "ready": "ready",
        "stopped": "stopped",
        "failed": "failed",
        "choose": "choose",
    }


def test_choose_re_measures_only_a_changed_model_and_replans_on_context():
    result = _run_wizard(
        _assistant_flow(
            """
(async () => {
  await reachPlan();
  component.modelOptions = [
    { id: 'a', model_path: '/models/a', display_name: 'A', model_context_length: 32768 },
    { id: 'b', model_path: '/models/b', display_name: 'B', model_context_length: 131072 },
  ];
  component.selectedModelPath = component.assistant.selectedModel;

  // Unchanged confirm: the screen closes and nothing is posted.
  component.assistantOpenChoose();
  const opened = {
    screen: component.assistantScreen(),
    draft: component.assistantChooseDraft.model_path,
  };
  let mark = calls.length;
  await component.assistantConfirmChoose();
  const unchanged = {
    screen: component.assistantScreen(),
    posted: calls.slice(mark).map((call) => call.url),
  };

  // Context only: the hardware and model measurements are reused, plan runs.
  component.assistantOpenChoose();
  component.assistantChooseContext(16384);
  mark = calls.length;
  await component.assistantConfirmChoose();
  const contextOnly = {
    posted: calls.slice(mark).map((call) => call.url),
    context: component.assistant.selectedContextTokens,
    model: component.assistant.selectedModel,
    state: component.assistantState(),
  };

  // Model change: measured model-bound again before plan.
  component.assistantOpenChoose();
  component.assistantChooseModel('/models/b');
  mark = calls.length;
  await component.assistantConfirmChoose();
  const modelChange = {
    posted: calls.slice(mark).map((call) => call.url),
    body: calls.slice(mark).map((call) => call.body),
    model: component.assistant.selectedModel,
    state: component.assistantState(),
  };
  process.stdout.write(JSON.stringify({ opened, unchanged, contextOnly, modelChange }));
})();
"""
        )
    )

    assert result["opened"]["screen"] == "choose"
    assert result["unchanged"] == {"screen": "plan", "posted": []}
    assert result["contextOnly"]["posted"] == ["/admin/api/cluster/assistant/plan"]
    assert result["contextOnly"]["context"] == 16384
    assert result["contextOnly"]["state"] == "planning"
    assert result["modelChange"]["posted"] == ["/admin/api/cluster/assistant/measure"]
    assert result["modelChange"]["body"][0]["model_path"] == "/models/b"
    assert result["modelChange"]["body"][0]["mtp"] is False
    assert result["modelChange"]["model"] == "/models/b"
    assert result["modelChange"]["state"] == "measuring_model"


def test_changed_inputs_are_stale_and_never_applied_against_the_old_plan():
    result = _run_wizard(
        _assistant_flow(
            """
(async () => {
  await reachPlan();
  const fresh = component.assistantInputsStale();
  component.assistant.selectedContextTokens = 16384;
  const stale = component.assistantInputsStale();
  const staleRow = component.assistantRowStale('planning');
  const freshRow = component.assistantRowStale('checking');
  const mark = calls.length;
  await component.runAssistantApply();
  const posted = calls.slice(mark).map((call) => call.url);
  const after = {
    state: component.assistantState(),
    stale: component.assistantInputsStale(),
  };
  process.stdout.write(JSON.stringify({
    fresh, stale, staleRow, freshRow, posted, after,
    applyState: component.assistant.applyState,
  }));
})();
"""
        )
    )

    assert result["fresh"] is False
    assert result["stale"] is True
    assert result["staleRow"] is True
    assert result["freshRow"] is False
    # Start setup re-planned first; apply was never posted against the old plan.
    assert result["posted"] == ["/admin/api/cluster/assistant/plan"]
    assert result["after"]["stale"] is False
    assert result["applyState"] == ""


def test_back_is_refused_while_a_step_is_running_and_returns_to_the_plan():
    result = _run_wizard(
        _assistant_flow(
            """
(async () => {
  await reachPlan();
  const planState = component.assistantState();
  component.assistant.state = 'starting';
  component.assistant.busy = true;
  const refused = component.assistantBackAllowed();
  component.assistantBackToPlan();
  const kept = {
    state: component.assistantState(),
    screen: component.assistantScreen(),
  };
  component.assistant.busy = false;
  component.assistant.state = 'failed';
  component.assistant.error = { code: 'plan', message: 'x', action: 'plan' };
  const allowed = component.assistantBackAllowed();
  component.assistantBackToPlan();
  const back = {
    state: component.assistantState(),
    screen: component.assistantScreen(),
    error: component.assistant.error,
  };
  process.stdout.write(JSON.stringify({ planState, refused, kept, allowed, back }));
})();
"""
        )
    )

    assert result["planState"] == "planning"
    assert result["refused"] is False
    assert result["kept"] == {"state": "starting", "screen": "launch"}
    assert result["allowed"] is True
    assert result["back"] == {"state": "planning", "screen": "plan", "error": None}


def test_a_failed_start_retries_in_place_against_the_applied_deployment():
    result = _run_wizard(
        _assistant_flow(
            """
(async () => {
  await reachPlan();
  component.assistant.applyOperationId = 'op-apply';
  component.assistant.applyState = 'applied';
  component.assistant.operationId = 'op-apply';
  component.assistant.error = {
    code: 'coordinator_unreachable', message: 'no answer', action: 'start',
  };
  component.assistant.state = 'failed';
  const step = component.assistantFailureStep();
  const label = component.assistantRetryLabel();
  const mark = calls.length;
  await component.retryAssistantStep();
  const posted = calls.slice(mark);
  process.stdout.write(JSON.stringify({
    step, label, state: component.assistantState(),
    url: posted.length ? posted[0].url : '',
    operationId: posted.length ? posted[0].body.operation_id : '',
  }));
})();
"""
        )
    )

    assert result["step"] == "start"
    assert result["url"] == "/admin/api/cluster/assistant/start"
    assert result["operationId"] == "op-apply"
    assert result["state"] == "starting"


def test_assistant_entry_loads_the_catalogue_before_the_entry_measurement():
    """Assistant mode never visits the manual plan step, so the catalogue has to
    be loaded here: candidates and the MTP head come from it."""

    result = _run_wizard(
        _assistant_flow(
            """
(async () => {
  const inner = component.apiFetch;
  component.apiFetch = async (url, options = {}) => {
    if (url.endsWith('/admin/api/cluster/models')) {
      calls.push({ url, method: 'POST', body: null });
      return {
        models: [{
          id: 'qwen', model_path: '/models/qwen3-32b',
          display_name: 'Qwen3 32B', model_context_length: 131072,
          mtp_compatible: true,
        }],
      };
    }
    return inner(url, options);
  };
  component.modelOptions = [];
  await component.startAssistant();
  process.stdout.write(JSON.stringify({
    models: component.modelOptions.length,
    mtp: component.assistantMtpFor('/models/qwen3-32b'),
    state: component.assistantState(),
    urls: calls.map((call) => call.url),
  }));
})();
"""
        )
    )

    assert result["models"] == 1
    assert result["mtp"] is True
    assert result["state"] == "measuring"
    assert result["urls"].index("/admin/api/cluster/models") < result["urls"].index(
        "/admin/api/cluster/assistant/measure"
    )


def test_pairing_from_a_failure_switches_to_manual_setup():
    result = _run_wizard(
        _WIZARD_TWO_MACS
        + "\n"
        + """
(() => {
  component.devicesPayload.discovered = [{
    node_id: 'node-c', friendly_name: 'Node C', paired: false,
    state: 'discovered', caps: { ram_gb: 64 }, addrs: [],
  }];
  component.setClusterMode('assistant');
  const visibleBefore = component.manualSectionsVisible();
  component.assistant.error = { code: 'node_not_paired', message: 'x', action: 'pair_mac' };
  component.assistant.state = 'failed';
  component.openAssistantPairing();
  process.stdout.write(JSON.stringify({
    visibleBefore,
    mode: component.clusterMode,
    visibleAfter: component.manualSectionsVisible(),
    target: component.pairing.target,
    state: component.assistantState(),
  }));
})();
"""
    )

    assert result["visibleBefore"] is False
    assert result["mode"] == "manual"
    assert result["visibleAfter"] is True
    assert result["target"]["node_id"] == "node-c"
    assert result["state"] == "idle"


def test_diagnostics_keeps_the_failure_message_and_unfolds_the_payload():
    result = _run_wizard(
        """
(() => {
  component.assistant.state = 'failed';
  component.assistant.error = {
    code: 'insufficient_memory', message: 'Mac Studio (.13) has no room',
    action: 'diagnostics',
  };
  component.openAssistantDiagnostics();
  process.stdout.write(JSON.stringify({
    state: component.assistantState(),
    message: component.assistant.error && component.assistant.error.message,
    open: component.assistantFailureOpen,
    title: component.assistantFailureTitle(),
  }));
})();
"""
    )

    assert result["state"] == "failed"
    assert result["message"] == "Mac Studio (.13) has no room"
    assert result["open"] is True
    assert result["title"] == "The model does not fit in memory"


def test_the_failed_screen_does_not_duplicate_the_primary_action():
    result = _run_wizard(
        """
(() => {
  const cases = {};
  const set = (action, applyState) => {
    component.assistant.applyState = applyState;
    component.assistant.plan = { feasible: true };
    component.assistant.error = { code: 'x', message: 'x', action };
    component.assistant.state = 'failed';
    return {
      step: component.assistantFailureStep(),
      duplicate: component.assistantRetryDuplicatesAction(),
      label: component.assistantRetryLabel(),
    };
  };
  cases.plan = set('plan', '');
  cases.measure = set('measure', '');
  cases.apply = set('apply', '');
  cases.start = set('start', 'applied');
  cases.retryCoordinator = set('retry_coordinator', 'applied');
  cases.wired = set('raise_wired_limit', '');
  cases.freeDisk = set('free_disk_space', '');
  process.stdout.write(JSON.stringify(cases));
})();
"""
    )

    assert result["plan"]["duplicate"] is True
    assert result["measure"]["duplicate"] is True
    assert result["apply"]["duplicate"] is True
    assert result["start"]["duplicate"] is True
    assert result["retryCoordinator"]["duplicate"] is True
    # The one-action steps that are *not* the primary keep a retry link.
    assert result["wired"]["duplicate"] is False
    assert result["wired"]["label"] == "Plan again"
    assert result["freeDisk"]["duplicate"] is False


def test_a_model_change_through_choose_reads_stale_until_the_new_plan():
    result = _run_wizard(
        _assistant_flow(
            """
(async () => {
  await reachPlan();
  component.modelOptions = [
    { id: 'b', model_path: '/models/b', display_name: 'B', model_context_length: 131072 },
  ];
  const before = component.assistantInputsStale();
  component.assistantOpenChoose();
  component.assistantChooseModel('/models/b');
  await component.assistantConfirmChoose();
  const during = {
    state: component.assistantState(),
    stale: component.assistantInputsStale(),
    stalePlanRow: component.assistantRowStale('planning'),
    staleCheckRow: component.assistantRowStale('checking'),
  };
  await component.pollAssistantMeasurement();   // model-bound ready → plan
  const after = {
    state: component.assistantState(),
    stale: component.assistantInputsStale(),
    model: component.assistant.inputs.model_path,
  };
  process.stdout.write(JSON.stringify({ before, during, after }));
})();
"""
        )
    )

    assert result["before"] is False
    assert result["during"]["state"] == "measuring_model"
    assert result["during"]["stale"] is True
    assert result["during"]["stalePlanRow"] is True
    assert result["during"]["staleCheckRow"] is False
    assert result["after"] == {
        "state": "planning",
        "stale": False,
        "model": "/models/b",
    }


def test_failed_branch_affordances_stay_inside_the_failure_screen():
    result = _run_wizard(
        _assistant_flow(
            """
(async () => {
  component.assistant.state = 'failed';
  component.assistant.busy = false;
  component.assistant.error = {
    code: 'insufficient_disk', message: 'no room', action: 'free_disk_space',
  };
  component.assistant.measurementNodes = [
    { node_id: 'node-b', model_paths: ['/models/old'] },
  ];
  component.assistantActionHandler('free_disk_space');
  const free = {
    open: component.assistant.freeSpaceOpen,
    error: component.assistant.error && component.assistant.error.action,
    state: component.assistantState(),
    screen: component.assistantScreen(),
  };

  component.assistant.state = 'failed';
  component.assistant.error = {
    code: 'topology_incomplete', message: 'no line', action: 'plan',
  };
  component.assistant.topologyFallback = {
    rank_order: ['node-a', 'node-b'], context_tokens: 32768,
  };
  component.assistant.missingLinks = [['node-b', 'node-c']];
  component.assistant.measurementId = 'm1';
  component.assistant.selectedModel = '/models/qwen3-32b';
  component.assistant.selectedContextTokens = 32768;
  const mark = calls.length;
  await component.confirmAssistantTwoMacFallback();
  const topology = {
    posted: calls.slice(mark).map((call) => call.url),
    body: calls.slice(mark).map((call) => call.body),
    state: component.assistantState(),
  };
  process.stdout.write(JSON.stringify({ free, topology }));
})();
"""
        )
    )

    assert result["free"] == {
        "open": True,
        "error": "free_disk_space",
        "state": "failed",
        "screen": "failed",
    }
    assert result["topology"]["posted"] == ["/admin/api/cluster/assistant/plan"]
    assert result["topology"]["body"][0]["confirm_two_mac_fallback"] is True
    assert result["topology"]["state"] == "planning"


ASSISTANT_FAILURE_ACTIONS = (
    "measure",
    "plan",
    "apply",
    "start",
    "status",
    "add_mac",
    "pair_mac",
    "copy_model",
    "configure_endpoint",
    "retry_coordinator",
    "diagnostics",
    "adjust_model_or_context",
    "choose_smaller_model",
    "choose_pipeline_model",
    "check_model_path",
    "free_disk_space",
    "raise_wired_limit",
)


def test_every_error_action_renders_on_the_failed_screen_with_one_label():
    """U3/A7: every action of the frozen vocabulary (plus the recommendation's
    near misses and R5) keeps the failure on the card with a real label."""

    result = _run_wizard(
        """
(() => {
  const actions = %s;
  const out = {};
  for (const action of actions) {
    component.assistant.state = 'failed';
    component.assistant.busy = false;
    component.assistant.plan = { feasible: true };
    component.assistant.applyState = '';
    component.assistant.error = { code: 'x', message: 'a plain cause', action };
    out[action] = {
      screen: component.assistantScreen(),
      label: component.assistantActionLabel(action),
      title: component.assistantFailureTitle(),
      message: component.assistantErrorMessage(),
      duplicate: component.assistantRetryDuplicatesAction(),
      step: component.assistantFailureStep(),
    };
  }
  process.stdout.write(JSON.stringify(out));
})();
"""
        % json.dumps(list(ASSISTANT_FAILURE_ACTIONS))
    )

    assert set(result) == set(ASSISTANT_FAILURE_ACTIONS)
    for action, row in result.items():
        assert row["screen"] == "failed", action
        # A label always comes from the catalogue, never the raw key.
        assert row["label"], action
        assert not row["label"].startswith("cluster.assistant."), action
        assert row["title"], action
        assert row["message"] == "a plain cause", action
        assert row["step"] in ("measure", "plan", "apply", "start"), action


def test_plan_warnings_never_colour_a_progress_row():
    """A7: warnings stay notes on the plan; no row becomes `failed` because of
    one, and the stale marking is amber — never red."""

    result = _run_wizard(
        _assistant_flow(
            """
(async () => {
  await reachPlan();
  component.assistant.warnings = [
    { code: 'tier_raised', message: 'tier raised for headroom' },
  ];
  component.assistant.plan.warnings = [
    { code: 'macs_not_needed', node_id: 'node-b', message: 'Mac not needed' },
  ];
  const rows = component.assistantProgressRows();
  process.stdout.write(JSON.stringify({
    state: component.assistantState(),
    screen: component.assistantScreen(),
    states: rows.map((row) => row.state),
    failedRows: rows.filter((row) => row.state === 'failed').length,
    warnings: component.assistantPlanWarnings().length,
    stale: rows.filter((row) => component.assistantRowStale(row.key)).length,
  }));
})();
"""
        )
    )

    assert result["state"] == "planning"
    assert result["screen"] == "plan"
    assert result["warnings"] == 1
    assert result["failedRows"] == 0
    assert "failed" not in result["states"]
    # Nothing is stale here, so no row claims it either.
    assert result["stale"] == 0
