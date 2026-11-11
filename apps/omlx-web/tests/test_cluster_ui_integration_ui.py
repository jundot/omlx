# SPDX-License-Identifier: Apache-2.0
"""Drive the cluster wizard against the backend contracts it depends on."""

import asyncio

import pytest

from omlx.cluster import routes
from tests.test_cluster_pairing import _loopback_pair
from tests.ui.test_cluster_v2_wizard import _WIZARD_TWO_MACS, _run_wizard


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
