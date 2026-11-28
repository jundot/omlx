# SPDX-License-Identifier: Apache-2.0
"""Drive the cluster wizard against the backend contracts it depends on."""

# The fixture templates use %-formatting on purpose: their JS bodies contain
# braces, so .format()/f-strings would need escaping that makes them unreadable.
# ruff: noqa: UP031

import asyncio
import json

import pytest
from omlx.cluster import routes
from tests.test_cluster_pairing import _loopback_pair
from tests.ui.test_cluster_wizard import _WIZARD_TWO_MACS, _fixtures, _run_wizard

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
component.assistant.state = 'planning';
component.assistant.plan = {plan_signature: 'a'.repeat(16)};
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
const failureFixture = %s;
const calls = [];
const measured = {};
let sequence = 0;
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
  if (url.includes('/assistant/models/')) {
    return {node_id: 'node-b', model_id: 'old', deleted: true};
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
    # Phase C3 (the roadmap, recorded in the first UI slice notes
    # the notes): the entry measurement is hardware-only (no model_path), and the
    # recommended model is measured model-bound before plan; plan then uses
    # that measurement and exactly that model.
    assert result["measureModels"] == [
        None,
        "/models/qwen3-32b-4bit",
    ]
    assert result["planBody"]["measurement_id"] == "m2"
    assert result["planBody"]["model_path"] == "/models/qwen3-32b-4bit"
    assert result["planBody"]["context_tokens"] == 32768


def test_assistant_retry_coordinator_retries_the_elected_coordinator_only():
    result = _run_wizard(
        _assistant_flow(
            """
(async () => {
  await reachPlan();
  component.assistant.error = {
    code: 'coordinator_unresolved',
    message: 'the elected coordinator did not answer',
    action: 'retry_coordinator',
  };
  component.assistant.state = 'failed';
  const failed = {
    state: component.assistantState(),
    action: component.assistant.error && component.assistant.error.action,
    coordinator: component.assistantCoordinatorName(),
    note: component.assistantCoordinatorInstruction(),
    label: component.assistantActionLabel('retry_coordinator'),
  };
  const counts = () => ({
    measure: calls.filter((c) => c.url.endsWith('/assistant/measure')).length,
    plan: calls.filter((c) => c.url.endsWith('/assistant/plan')).length,
  });
  const before = counts();
  await component.assistantActionHandler('retry_coordinator');
  const after = counts();
  const plans = calls.filter((c) => c.url.endsWith('/assistant/plan'));
  process.stdout.write(JSON.stringify({
    failed, before, after,
    planInputs: plans[plans.length - 1].body,
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
    # Retry re-posts the same frozen plan: no new measurement, and the
    # coordinator never becomes this Mac.
    assert result["after"]["measure"] == result["before"]["measure"]
    assert result["after"]["plan"] == result["before"]["plan"] + 1
    assert result["planInputs"]["measurement_id"] == "m2"
    assert result["selfNode"] == "node-a"


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

    assert result["state"] == "planning"
    assert result["offending"] == []
    assert all("/assistant/" in url for url in result["urls"])


def test_assistant_plan_inputs_are_frozen_from_the_recommendation():
    """The plan request is the component's one frozen input object."""

    result = _run_wizard(
        _assistant_flow(
            """
(async () => {
  await reachPlan();
  const body = calls.find((c) => c.url.endsWith('/assistant/plan')).body;
  component.toggleAssistantNodeSelection();
  await component.runAssistantPlan();
  const replans = calls.filter((c) => c.url.endsWith('/assistant/plan'));
  process.stdout.write(JSON.stringify({
    body,
    state: component.assistantState(),
    signature: component.assistant.plan && component.assistant.plan.plan_signature,
    inputs: component.assistant.inputs,
    nodeSelection: replans[replans.length - 1].body.node_selection,
  }));
})();
"""
        )
    )

    body = result["body"]
    for key in (
        "measurement_id",
        "model_path",
        "context_tokens",
        "rank0_extra_bytes",
        "mtp",
        "execution_profile",
        "microbatch_size",
        "node_selection",
        "confirm_two_mac_fallback",
    ):
        assert key in body, key
    assert body["measurement_id"] == "m2"
    assert body["model_path"] == "/models/qwen3-32b-4bit"
    assert body["context_tokens"] == 32768
    assert body["rank0_extra_bytes"] == 0
    assert body["mtp"] is False
    assert body["execution_profile"] == "balanced"
    assert body["microbatch_size"] is None
    assert body["node_selection"] == "minimal"
    assert body["confirm_two_mac_fallback"] is False
    # The plan rests in `planning` with the backend's own signature, and the
    # frozen inputs are what the card keeps.
    assert result["state"] == "planning"
    assert result["signature"] == "3f9a1c7d20b45e68"
    assert result["inputs"]["measurement_id"] == "m2"
    assert result["inputs"]["context_tokens"] == 32768
    # The advanced override re-plans with the other selection.
    assert result["nodeSelection"] == "all"


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
