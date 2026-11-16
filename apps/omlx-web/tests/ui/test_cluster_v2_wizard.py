# SPDX-License-Identifier: Apache-2.0
"""Offline smoke tests for the cluster v2 wizard (Module C).

The wizard is a client-side Alpine component, so "rendering each state with
fixture JSON" is verified offline in two halves:

1. The Jinja side: dashboard.html renders with every state's markup present,
   each guarded by its ``wizardState() === '<state>'`` expression.
2. The data side: tests/ui/fixtures/cluster_v2/*.json pin the endpoint
   payloads the wizard consumes — one fixture per state transition — and are
   checked against the PeerRecord / plan / deployment contracts from
   ops/notes/omlx_cluster_v2_spec.md.

Nothing here starts a server, opens a socket, or touches mlx.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from omlx_web import routes as webui

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/ui/fixtures/cluster_v2"

TEMPLATE = "omlx_web/templates/dashboard/_cluster_v2.html"
JAVASCRIPT = "omlx_web/static/js/cluster_v2.js"
DASHBOARD = "omlx_web/templates/dashboard.html"

WIZARD_STATES = (
    "empty",
    "discovering",
    "device_card",
    "pairing",
    "checks",
    "plan",
    "active",
    "error",
)

PEER_RECORD_FIELDS = {
    "node_id",
    "friendly_name",
    "version",
    "cluster_name",
    "caps",
    "addrs",
    "http_port",
    "paired",
    "last_seen",
    "link",
    "state",
}

# The complete network surface the wizard is allowed to use (spec Module C:
# Module A/B endpoints + the pre-existing planner/activate API only).
ALLOWED_ENDPOINTS = {
    "/api/cluster/pair/join/cleanup",
    "/api/cluster/pair/invite",


    "/admin/api/cluster/forget",
    "/api/cluster/devices",
    "/api/cluster/node_id",
    "/api/cluster/discovery/health",  # Module C stub, pending Module A impl
    "/api/cluster/pair/approve",
    "/api/cluster/pair/deny",
    "/api/cluster/pair/join",
    "/api/cluster/pair/join/cancel",
    "/api/cluster/devices/manual",
    "/admin/api/cluster/models",
    "/admin/api/cluster/catalogue",
    "/admin/api/cluster/peer-probe",
    "/admin/api/cluster/autoconfigure",
    "/admin/api/cluster/node-roles",
    "/admin/api/cluster/node-budgets",
    "/admin/api/cluster/stage",
    "/admin/api/cluster/runtime",
    "/admin/api/cluster/deployments",
    "/admin/api/cluster/replan",
    "/admin/api/cluster/diagnostics",
    "/admin/api/cluster/join-keys",
    "/admin/api/cluster/join-status",
    "/admin/api/cluster/cuda-fabric/verify",
    "/admin/api/cluster/rdma-links",
    "/admin/api/cluster/rdma-links/verify",
    # Cluster Assistant (SCREEN-3-MINIMAL.md §2): the entire browser surface
    # for Screen 3. Nothing else in the tab may reach these.
    "/admin/api/cluster/assistant/measure",
    "/admin/api/cluster/assistant/status",
    "/admin/api/cluster/assistant/recommend",
    "/admin/api/cluster/assistant/plan",
    "/admin/api/cluster/assistant/apply",
    "/admin/api/cluster/assistant/start",
    # The automatic constellation search (synchronous, minutes-long).
    "/admin/api/cluster/assistant/search",
    "/admin/api/cluster/assistant/operations",
    # Screen 6: the stop relay to the elected coordinator's own unload route.
    "/admin/api/cluster/assistant/stop",
    # Reload resume: the last persisted deployment (read-only).
    "/admin/api/cluster/assistant/session",
    # The redacted support bundle (read-only).
    "/admin/api/cluster/assistant/diagnostics",
    # The read-only local network preflight.
    "/admin/api/cluster/assistant/network-preflight",
    # The in-app switch for the post-ready measurement.
    "/admin/api/cluster/assistant/performance-probe",
    # The free-space relay (PHASE-2B-RUN.md §9): the only assistant route that
    # changes another Mac's disk.
    "/admin/api/cluster/assistant/models",
    # Tuning: the quick sampler sweep and the minutes-long MTP depth sweep.
    "/admin/api/cluster/assistant/tune",
    "/admin/api/cluster/assistant/tune-redeploy",
    "/admin/api/cluster/assistant/tune-local",
    # Run on this Mac: single-node inference on the origin Mac, as an async job.
    "/admin/api/cluster/assistant/run-local",
}


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text()


def _fixtures():
    return {path.name: json.loads(path.read_text()) for path in FIXTURES.glob("*.json")}


def test_dashboard_renders_every_wizard_state():
    rendered = webui.templates.get_template("dashboard.html").render()

    assert "data-cluster-v2-wizard" in rendered
    for state in WIZARD_STATES:
        assert f'data-cluster-v2-state="{state}"' in rendered, state


def test_each_state_section_is_guarded_by_its_state_machine_value():
    template = _read(TEMPLATE)

    for state in WIZARD_STATES:
        assert f'data-cluster-v2-state="{state}"' in template, state
    # Each guarded state marker sits on an element whose x-show pins exactly
    # the state it names.
    for match in re.finditer(
        r'x-show="wizardState\(\) === \'(\w+)\'"[^>]*data-cluster-v2-state="(\w+)"',
        template,
    ):
        assert match.group(1) == match.group(2)
    guarded = {
        m.group(2)
        for m in re.finditer(
            r'x-show="wizardState\(\) === \'(\w+)\'"[^>]*data-cluster-v2-state="(\w+)"',
            template,
        )
    }
    # device_card shares its section guard with later states, so it is checked
    # separately; every other state must be guarded 1:1.
    assert guarded >= {"empty", "discovering", "pairing", "checks", "plan", "active", "error"}
    assert "['device_card', 'pairing', 'checks', 'plan'].includes(wizardState())" in template


def test_fixtures_cover_every_state_and_honor_the_peer_contract():
    fixtures = _fixtures()
    assert fixtures, "no fixtures found"

    covered = {fixture.get("_state") for fixture in fixtures.values()}
    assert covered >= {
        "empty",
        "discovering",
        "device_card",
        "pairing",
        "checks",
        "plan",
        "active",
    }

    for name, fixture in fixtures.items():
        if "paired" not in fixture:
            continue
        assert set(fixture) >= {"paired", "discovered", "self"}, name
        devices = list(fixture["paired"]) + list(fixture["discovered"])
        if fixture["self"]:
            devices.append(fixture["self"])
        for device in devices:
            missing = PEER_RECORD_FIELDS - set(device)
            assert not missing, f"{name}: {device.get('node_id')}: {missing}"
            caps = device["caps"]
            assert {"chip", "ram_gb", "backends", "thunderbolt", "jaccl"} <= set(caps), name
            for addr in device["addrs"]:
                assert {"ip", "if_type"} <= set(addr), name


def test_discovered_fixture_proves_the_two_mac_cap_is_gone():
    discovered = _fixtures()["devices_discovered.json"]

    assert len(discovered["discovered"]) == 3

    javascript = _read(JAVASCRIPT)
    template = _read(TEMPLATE)
    assert "length === 2" not in javascript
    assert "planNodes().slice(0, 2" not in javascript
    assert "max 2" not in javascript.lower()
    assert 'x-for="device in allDevices()"' in template


def test_wizard_consumes_only_contract_endpoints():
    javascript = _read(JAVASCRIPT)

    found = set(re.findall(r"'(/(?:admin/)?api/cluster[^']*)'", javascript))
    unexpected = found - ALLOWED_ENDPOINTS
    assert not unexpected, f"wizard calls undeclared endpoints: {unexpected}"

    # The binding contract endpoints must all be present.
    for endpoint in (
        "/api/cluster/devices",
        "/api/cluster/pair/approve",
        "/api/cluster/pair/deny",
        "/api/cluster/pair/join",
        "/admin/api/cluster/autoconfigure",
        "/admin/api/cluster/runtime",
        "/admin/api/cluster/deployments",
    ):
        assert endpoint in javascript

    # DELETE /api/cluster/devices/{node_id} (unpair).
    assert re.search(r"`/api/cluster/devices/\$\{encodeURIComponent\(nodeId\)\}`", javascript)
    # GET /admin/api/cluster/stage/{job_id} (auto-staging progress poll).
    assert re.search(r"`/admin/api/cluster/stage/\$\{encodeURIComponent\(id\)\}`", javascript)


def test_polling_is_one_hertz_and_visibility_gated():
    javascript = _read(JAVASCRIPT)

    assert "CLUSTER_V2_POLL_MS = 1000" in javascript
    assert "document.hidden" in javascript
    assert "this.mainTab === 'cluster'" in javascript
    assert "clusterLegacyView" not in javascript
    assert "setInterval(() => this.tick(), CLUSTER_V2_POLL_MS)" in javascript


def test_errors_are_toasts_and_banners_never_modals():
    template = _read(TEMPLATE)
    javascript = _read(JAVASCRIPT)

    assert "data-cluster-v2-toasts" in template
    assert "data-cluster-v2-error" in template
    # No modal overlay primitives anywhere in the v2 partial.
    assert "fixed inset-0" not in template
    assert "x-model" not in template or "showModal" not in template
    assert "notify(type, message" in javascript


def test_version_mismatch_banner_is_actionable_and_keeps_the_device():
    template = _read(TEMPLATE)
    javascript = _read(JAVASCRIPT)

    assert "data-cluster-v2-version-mismatch" in template
    assert "cluster.v2.version_mismatch.title" in template
    assert "brew upgrade omlx" in template
    assert "versionMismatches()" in javascript
    # The banner compares peer vs self versions and names both.
    assert "device.version !== self.version" in javascript
    # The device list is never filtered on version — mismatch is a banner, not
    # a hidden card.
    all_devices_body = javascript.split("allDevices() {", 1)[1].split(
        "activeDeployment()", 1
    )[0]
    assert "version" not in all_devices_body


def test_multicast_self_test_stub_degrades_gracefully():
    javascript = _read(JAVASCRIPT)
    template = _read(TEMPLATE)

    assert "/api/cluster/discovery/health" in javascript
    assert "discoveryHealthUnsupported" in javascript
    assert "error?.status === 404" in javascript
    assert "cluster.v2.discovering.local_network" in template
    # The fixture pins the stub contract for whoever implements it.
    health = _fixtures()["discovery_health_ok.json"]
    assert "multicast_rx_within_5s" in health


def test_pairing_flow_uses_module_b_endpoints():
    template = _read(TEMPLATE)
    javascript = _read(JAVASCRIPT)

    assert "data-cluster-v2-pairing" in template
    assert "data-cluster-v2-pair-code" in template
    assert "data-cluster-v2-approve" in template
    assert "data-cluster-v2-deny" in template
    assert "submitPairApproval" in javascript
    assert "submitPairDenial" in javascript
    # Six-digit code, validated client-side before any request leaves.
    assert re.search(r"\^\\d\{6\}\$", javascript)
    # Pending join requests surface from the devices snapshot.
    assert "awaiting_approval" in javascript
    fixture = _fixtures()["pair_approve_request.json"]
    assert set(fixture) >= {"node_id", "code"}


def test_checks_rows_cover_the_six_spec_checks():
    javascript = _read(JAVASCRIPT)
    template = _read(TEMPLATE)

    for key in ("ssh", "model", "version", "rdma", "benchmark", "multicast"):
        assert f"key: '{key}'" in javascript, key
    assert "data-cluster-v2-check-row" in template
    assert "runBenchmark()" in javascript
    assert "checksBlockingPass()" in javascript


def test_plan_state_renders_the_signed_layer_split_bar():
    template = _read(TEMPLATE)
    javascript = _read(JAVASCRIPT)
    plan = _fixtures()["plan_response.json"]

    assert "data-cluster-v2-split-bar" in template
    assert "assignment.layer_count / Math.max(planTotalLayers(), 1)" in template
    assert "placement_signature" in javascript
    assert "approved_placement" in javascript
    assert len(plan["assignments"]) == 3  # N-node split, not two
    assert plan["placement_signature"]
    for assignment in plan["assignments"]:
        assert {
            "rank",
            "node_id",
            "start_layer",
            "end_layer",
            "layer_count",
        } <= set(assignment)


def test_configured_deployment_panel_lists_devices_and_deactivates():
    template = _read(TEMPLATE)
    javascript = _read(JAVASCRIPT)
    deployments = _fixtures()["deployments_active.json"]

    assert "data-cluster-v2-active" in template
    assert "data-cluster-v2-active-device" in template
    assert "data-cluster-v2-active-fabric" in template
    assert "data-cluster-v2-runtime-state" in template
    assert "data-cluster-v2-deactivate" in template
    assert "data-cluster-v2-model-lifecycle" in template
    assert "data-cluster-v2-change-model" in template
    assert "data-cluster-v2-unload-weights" in template
    assert "data-cluster-v2-load-weights" in template
    assert "data-cluster-v2-change-model-confirm" in template
    assert "configuredDeployment()" in template
    assert "deploymentStatus().label" in template
    assert "deactivateDeployment" in javascript
    assert "changeClusterModel" in javascript
    assert "unloadDeploymentWeights" in javascript
    assert "loadDeploymentWeights" in javascript
    assert "cluster.v2.link.tailscale_control" in javascript
    assert "cluster.v2.deploy.fabric_jaccl" in javascript
    assert "cluster.v2.active.fabric_note" in template
    assert re.search(
        r"`/admin/api/cluster/deployments/\$\{encodeURIComponent\(id\)\}`",
        javascript,
    )
    assert deployments["deployments"][0]["deployment_id"]
    # This is the real ClusterDeployment field, not the stale fixture-only
    # alias that made the live page render "this model".
    assert deployments["deployments"][0]["model"].endswith("minimax-m3")
    assert "model_path" not in deployments["deployments"][0]


def test_change_model_reuses_onboarding_with_current_serving_settings():
    result = _run_wizard(
        _WIZARD_TWO_MACS
        + """
global.setTimeout = () => 0;
const deployment = {
  deployment_id: 'pool-a', model: '/models/old',
  target_context_tokens: 131072,
  execution: {
    profile: 'throughput', prompt_cache_ssd: true,
    prompt_cache_ssd_max_bytes: 20 * (1024 ** 3),
  },
  assignments: [
    { node_id: 'node-a', role: 'workstation' },
    { node_id: 'node-b', role: 'headless' },
  ],
};
component.deploymentsPayload = [deployment];
component.deploymentsLoaded = true;
component.runtimePayload = { jobs: [{
  deployment_id: 'pool-a', rank: 0, phase: 'ready', live: true,
  ownership: 'loaded',
}], launchers: [] };
component.runtimeLoaded = true;
component.selectedModelPath = '/models/old';
component.planStrategy = 'tensor';
component.modelOptions = [{ model_path: '/models/old', id: 'old' }];
const calls = [];
component.apiFetch = async (url, options = {}) => {
  calls.push({ url, method: options.method || 'GET' });
  if (url.endsWith('/deployments')) return { deployments: [] };
  if (url.endsWith('/runtime')) return { jobs: [], launchers: [] };
  if (url.endsWith('/models')) return { models: [] };
  if (url.endsWith('/node-roles')) return { roles: [] };
  return { ok: true };
};
(async () => {
  component.beginModelChange(deployment);
  await component.changeClusterModel(deployment);
  process.stdout.write(JSON.stringify({
    stage: component.stage,
    deployments: component.deploymentsPayload,
    selectedModelPath: component.selectedModelPath,
    executionProfile: component.executionProfile,
    promptCacheSsd: component.promptCacheSsd,
    promptCacheSsdMaxGiB: component.promptCacheSsdMaxGiB,
    targetContextTokens: component.targetContextTokens,
    nodeRoles: component.nodeRoles,
    planStrategy: component.planStrategy,
    calls,
  }));
})();
"""
    )

    assert result["stage"] == "plan"
    assert result["deployments"] == []
    assert result["selectedModelPath"] == ""
    assert result["executionProfile"] == "throughput"
    assert result["promptCacheSsd"] is True
    assert result["promptCacheSsdMaxGiB"] == 20
    assert result["targetContextTokens"] == 131072
    assert result["nodeRoles"] == {
        "node-a": "workstation",
        "node-b": "headless",
    }
    assert result["planStrategy"] == "auto"
    assert {
        "url": "/admin/api/cluster/deployments/pool-a",
        "method": "DELETE",
    } in result["calls"]


def test_active_card_load_and_unload_keep_the_signed_deployment():
    result = _run_wizard("""
global.setTimeout = () => 0;
const deployment = { deployment_id: 'pool-a', model: '/models/m' };
component.deploymentsPayload = [deployment];
component.deploymentsLoaded = true;
component.runtimeLoaded = true;
component.runtimePayload = { jobs: [{
  deployment_id: 'pool-a', rank: 0, phase: 'ready', live: true,
  ownership: 'loaded',
}], launchers: [] };
const calls = [];
component.apiFetch = async (url, options = {}) => {
  calls.push({ url, method: options.method || 'GET' });
  if (url.endsWith('/deployments')) return { deployments: [deployment] };
  if (url.endsWith('/runtime')) return { jobs: [], launchers: [] };
  return { ok: true };
};
(async () => {
  await component.unloadDeploymentWeights(deployment);
  const armed = component.confirmUnloadFor;
  await component.unloadDeploymentWeights(deployment);
  const deploymentAfterUnload = component.configuredDeployment();
  await component.loadDeploymentWeights(deployment);
  process.stdout.write(JSON.stringify({
    armed,
    deploymentAfterUnload: deploymentAfterUnload && deploymentAfterUnload.deployment_id,
    lifecycleBusy: component.clusterLifecycleBusy,
    calls,
  }));
})();
""")

    assert result["armed"] == "pool-a"
    assert result["deploymentAfterUnload"] == "pool-a"
    assert result["lifecycleBusy"] is False
    assert {
        "url": "/admin/api/cluster/deployments/pool-a/unload",
        "method": "POST",
    } in result["calls"]
    assert {
        "url": "/admin/api/cluster/deployments/pool-a/load",
        "method": "POST",
    } in result["calls"]


def test_runtime_residency_separates_configuration_from_observed_workers():
    fixtures = _fixtures()
    deployment = fixtures["deployments_active.json"]["deployments"]
    snapshots = {
        name: fixtures[name]
        for name in (
            "runtime_detached.json",
            "runtime_loading.json",
            "runtime_ready.json",
            "runtime_failed.json",
        )
    }
    result = _run_wizard(f"""
component.deploymentsPayload = {json.dumps(deployment)};
component.deploymentsLoaded = true;
const savedDeployment = component.deploymentsPayload[0];

const snapshots = {json.dumps(snapshots)};
const samples = {{}};
for (const [name, snapshot] of Object.entries(snapshots)) {{
  component.runtimePayload = snapshot;
  component.runtimeLoaded = true;
  component.runtimeError = '';
  const configured = component.configuredDeployment();
  const active = component.activeDeployment();
  const visible = configured || savedDeployment;
  const status = component.deploymentStatus(visible);
  samples[name] = {{
    runtimeState: component.deploymentRuntimeState(visible),
    wizardState: component.wizardState(),
    configured: configured && configured.deployment_id,
    active: active && active.deployment_id,
    modelName: component.displayModelName(visible),
    label: status.label,
    eyebrow: status.eyebrow,
    detail: status.detail,
    tone: status.tone,
  }};
}}

// A clean unload removes the marker entirely; it has the same observed state
// as a detached ready marker left behind by a reboot.
component.runtimePayload = {{ jobs: [], launchers: [], warnings: [] }};
component.runtimeLoaded = true;
samples.empty = {{
  runtimeState: component.deploymentRuntimeState(savedDeployment),
  active: component.activeDeployment(),
  label: component.deploymentStatus(savedDeployment).label,
}};
// A launcher record is process-manager metadata, not proof that every rank is
// alive or that the engine pool still owns the weights.
component.runtimePayload = {{
  jobs: [],
  launchers: [{{
    deployment_id: savedDeployment.deployment_id,
    phase: 'ready',
    returncode: null,
  }}],
}};
samples.launcherOnly = {{
  runtimeState: component.deploymentRuntimeState(savedDeployment),
  active: component.activeDeployment(),
  label: component.deploymentStatus(savedDeployment).label,
}};
process.stdout.write(JSON.stringify(samples));
""")

    detached = result["runtime_detached.json"]
    assert detached["runtimeState"] == "configured"
    assert detached["wizardState"] == "active"  # management remains accessible
    assert detached["configured"] is not None
    assert detached["active"] is None, "a detached marker is not residency"
    assert detached["modelName"] == "minimax-m3"
    assert detached["label"] == "Not loaded"
    assert "no model weights are resident" in detached["detail"]
    assert "amber" in detached["tone"]

    assert result["empty"] == {
        "runtimeState": "configured",
        "active": None,
        "label": "Not loaded",
    }
    assert result["launcherOnly"] == result["empty"]

    loading = result["runtime_loading.json"]
    assert loading["runtimeState"] == "loading"
    assert loading["active"] is None
    assert loading["label"] == "Loading"
    assert "blue" in loading["tone"]

    ready = result["runtime_ready.json"]
    assert ready["runtimeState"] == "ready"
    assert ready["active"] == ready["configured"]
    assert ready["label"] == "Ready"
    assert ready["eyebrow"] == "Cluster running"
    assert "green" in ready["tone"]

    failed = result["runtime_failed.json"]
    assert failed["runtimeState"] == "failed"
    assert failed["active"] is None
    assert failed["label"] == "Failed"
    assert failed["eyebrow"] == "Cluster needs attention"
    assert failed["detail"] == "rank 1 stopped heartbeating"
    assert "red" in failed["tone"]


def test_runtime_poll_failure_revokes_a_previous_ready_snapshot():
    fixtures = _fixtures()
    result = _run_wizard(
        f"""
component.deploymentsPayload = {json.dumps(fixtures['deployments_active.json']['deployments'])};
component.runtimePayload = {json.dumps(fixtures['runtime_ready.json'])};
component.runtimeLoaded = true;
const before = component.deploymentRuntimeState();
component.apiFetch = async () => {{ throw new Error('runtime endpoint down'); }};
(async () => {{
  await component.refreshRuntime();
  process.stdout.write(JSON.stringify({{
    before,
    after: component.deploymentRuntimeState(),
    active: component.activeDeployment(),
    loaded: component.runtimeLoaded,
    payload: component.runtimePayload,
    label: component.deploymentStatus().label,
    detail: component.deploymentStatus().detail,
  }}));
}})();
"""
    )

    assert result["before"] == "ready"
    assert result["after"] == "unknown"
    assert result["active"] is None
    assert result["loaded"] is False
    assert result["payload"] is None
    assert result["label"] == "Checking"
    assert "runtime endpoint down" in result["detail"]


def test_active_card_prefers_loaded_runtime_owner_over_stale_registry_order():
    result = _run_wizard(
        """
component.deploymentsPayload = [
  { deployment_id: 'ds4-cold', model: '/models/ds4', serving_mode: 'sharded' },
  { deployment_id: 'qwen-phase', model: '/models/qwen',
    serving_mode: 'disaggregated', prefill_rank: 1, decode_rank: 0 },
];
const fallback = component.configuredDeployment().deployment_id;
component.runtimePayload = { jobs: [{
  deployment_id: 'qwen-phase', rank: 0, live: true,
  phase: 'ready', ownership: 'loaded', metrics: {},
}], launchers: [] };
component.runtimeLoaded = true;
process.stdout.write(JSON.stringify({
  fallback,
  selected: component.configuredDeployment().deployment_id,
  mode: component.configuredDeployment().serving_mode,
  state: component.deploymentRuntimeState(),
}));
"""
    )

    assert result == {
        "fallback": "ds4-cold",
        "selected": "qwen-phase",
        "mode": "disaggregated",
        "state": "ready",
    }


def test_activation_progress_tracks_rank_stages_until_readiness():
    result = _run_wizard(
        """
component.deploymentsPayload = [{ deployment_id: 'pool-a', model: '/models/qwen' }];
component.deploymentsLoaded = true;
component.runtimeLoaded = true;
component.runtimePayload = { jobs: [{
  deployment_id: 'pool-a', rank: 0, world_size: 2,
  phase: 'loading', load_stage: 'materializing_layers',
  live: true, ownership: 'loading',
}], launchers: [] };
const loading = {
  visible: component.activationProgressVisible(),
  percent: component.activationProgressPercent(),
  label: component.activationProgressLabel(),
  detail: component.activationProgressDetail(),
  steps: component.wizardSteps().map((step) => step.state),
};
component.runtimePayload.jobs[0] = {
  ...component.runtimePayload.jobs[0], phase: 'ready', load_stage: 'ready',
  ownership: 'loaded', live: true,
};
const ready = {
  visible: component.activationProgressVisible(),
  steps: component.wizardSteps().map((step) => step.state),
};
process.stdout.write(JSON.stringify({ loading, ready }));
"""
    )

    assert result["loading"] == {
        "visible": True,
        "percent": 60,
        "label": "Materializing model layers",
        "detail": "0 of 2 ranks ready · readiness canary runs last",
        "steps": ["done", "done", "done", "done", "active"],
    }
    assert result["ready"] == {
        "visible": False,
        "steps": ["done", "done", "done", "done", "done"],
    }

    template = _read(TEMPLATE)
    assert "data-cluster-v2-activation-progress" in template
    assert "data-cluster-v2-activation-progress-bar" in template
    assert '<i data-lucide="check" class="w-3 h-3"></i>' not in template


def test_prefill_accelerator_reopens_for_a_mac_in_use_and_ends_at_its_checks():
    result = _run_wizard(
        """
(async () => {
component.devicesPayload = { paired: [{ node_id: 'm5', friendly_name: 'M5max' }] };
component.devicesLoaded = true;
component.deploymentsLoaded = true;
// An old record says this Mac helps M5max; its model using M5max wins.
component.helperRole = { active: false, helps: { node_id: 'm5', name: 'M5max' } };
let check;
component.apiFetch = (url) => url === '/admin/api/models'
  ? Promise.resolve({ models: [{ settings: { prefill_helper_node_id: 'm5' } }] })
  : new Promise((resolve) => { check = resolve; });
await component.restoreAccelerator();
const reopened = [component.wizardState(), component.accelerator.nodeId];
const running = component.wizardSteps().map((step) => step.state);
check({ rows: [{ key: 'ssh', status: 'pass' }, { key: 'disk', status: 'warn' }] });
await new Promise((resolve) => setTimeout(resolve));
const ready = component.wizardSteps().map((step) => step.state);
// Change setup leaves the screen until the page loads again.
component.choosePurpose();
await component.restoreAccelerator();
process.stdout.write(JSON.stringify({
  reopened, running, ready, changed: component.wizardState(),
}));
})();
"""
    )

    assert result == {
        "reopened": ["accelerator", "m5"],
        "running": ["done", "done", "active"],
        "ready": ["done"] * 3,
        "changed": "purpose",
    }


def test_a_paired_helper_daemon_reopens_before_any_model_uses_it():
    result = _run_wizard(
        """
(async () => {
const reopen = async (...paired) => {
  const page = clusterV2Wizard();
  page.devicesPayload = { paired };
  page.devicesLoaded = true;
  page.deploymentsLoaded = true;
  page.apiFetch = (url) => url === '/admin/api/models'
    ? Promise.resolve({ models: [] })
    : new Promise(() => {});
  await page.restoreAccelerator();
  return [page.wizardState(), page.accelerator?.nodeId || null];
};
const agent = { control: 'agent' };
const helper = await reopen(
  { node_id: 'old', caps: agent, state: 'dead' }, { node_id: 'gpu', caps: agent },
);
const mac = await reopen({ node_id: 'm5', friendly_name: 'M5max' });
process.stdout.write(JSON.stringify({ helper, mac }));
})();
"""
    )

    # The helper that answers comes first. A paired Mac may also split a
    # model, so it stays on the device cards.
    assert result == {"helper": ["accelerator", "gpu"], "mac": ["device_card", None]}


def test_a_helper_mac_shows_whom_it_helps_until_setup_changes():
    result = _run_wizard(
        """
component.devicesPayload = { paired: [{ node_id: 'm3', friendly_name: 'HJKmac' }] };
component.helperRole = { active: false, helps: { node_id: 'm3', name: 'HJKmac' } };
const idle = [component.wizardState(), component.helperRoleText()];
component.helperRole = {
  active: true, host: 'HJKmac', layers: 28, model: '/m/MiniMax',
  helps: { node_id: 'm3', name: 'HJKmac' },
};
const busy = component.helperRoleText();
const steps = component.wizardSteps().map((step) => step.state);
component.choosePurpose(true);
const afterPairing = component.wizardState();
component.choosePurpose();
const changed = component.wizardState();
component.startAccelerator({ node_id: 'm3' });
process.stdout.write(JSON.stringify({
  idle, busy, steps, afterPairing, changed, picked: component.wizardState(),
}));
"""
    )

    assert result["idle"][0] == "helper"
    assert result["idle"][1].startswith("It starts when HJKmac loads")
    assert result["busy"] == "Helping HJKmac with long prompts: 28 layers of MiniMax."
    assert result["steps"] == ["done"] * 3
    # The chooser that opens after pairing yields; Change setup does not.
    assert result["afterPairing"] == "helper"
    assert result["changed"] == "purpose"
    # Picking Prefill Accelerator for the Mac it helps shows the helper side.
    assert result["picked"] == "helper"


def test_a_helper_daemon_pairs_by_its_log_code_and_never_joins_a_split():
    result = _run_wizard(
        """
(async () => {
const helper = {
  node_id: 'gpu', friendly_name: 'gpu-box', version: '0.1.0', paired: true, addrs: [],
  caps: { chip: '2x NVIDIA RTX A6000', ram_gb: 96, control: 'agent' },
};
component.devicesPayload = {
  self: { node_id: 'node-a', version: '0.9.0', caps: { ram_gb: 256 }, addrs: [] },
  paired: [
    { node_id: 'node-b', version: '0.9.0', caps: { ram_gb: 128 }, addrs: [], paired: true },
    helper,
  ],
  discovered: [],
};
const split = {
  nodes: component.planNodes().map((node) => node.node_id),
  hosts: component.inventoryHosts().map((host) => host.node_id),
  mismatches: component.versionMismatches().length,
};
component.notify = () => {};
component.formatSizeBytes = (bytes) => (bytes / 1024 ** 3).toFixed(1) + ' GB';
component.apiFetch = async (url) => url.startsWith('/admin/api/prefill-helper/check')
  ? { name: 'gpu-box', rows: [
      { key: 'control', status: 'pass', detail: '192.168.1.230:8851' },
      { key: 'version', status: 'pass', detail: '0.1.0', families: ['qwen4_exp'], fix: '' },
      { key: 'memory', status: 'pass', bytes: 101786320896, gpus: 2 },
    ] }
  : {};
component.pairing = { target: helper, code: '754901', busy: false, error: '' };
const stepOne = component.pairingStepOne();
await component.submitPairApproval(helper);
await new Promise((resolve) => setTimeout(resolve));
process.stdout.write(JSON.stringify({
  split, stepOne, state: component.wizardState(),
  rows: component.acceleratorRows().map((row) => [row.label, row.detail]),
}));
})();
"""
    )

    # The helper takes no layers, and its own version is not a mismatch.
    assert result["split"] == {
        "nodes": ["node-a", "node-b"],
        "hosts": ["node-a", "node-b"],
        "mismatches": 0,
    }
    assert "in its log" in result["stepOne"]
    # Approval goes to the accelerator checks without the purpose choice.
    assert result["state"] == "accelerator"
    assert result["rows"] == [
        ["Control channel", "Reached 192.168.1.230:8851"],
        ["Helper version", "0.1.0 works with this oMLX version"],
        ["GPU memory for layers", "94.8 GB free (GPUs: 2)"],
    ]


def test_retry_loading_ownership_supersedes_retained_failure_marker():
    result = _run_wizard(
        """
component.deploymentsPayload = [{ deployment_id: 'pool-a', model: '/models/qwen' }];
component.deploymentsLoaded = true;
component.runtimeLoaded = true;
component.runtimePayload = { jobs: [{
  deployment_id: 'pool-a', rank: 0, world_size: 2,
  phase: 'failed', load_stage: 'loading_weights',
  live: false, ownership: 'loading', error: 'failure from the previous attempt',
}], launchers: [] };
const retry = {
  state: component.deploymentRuntimeState(),
  label: component.deploymentStatus().label,
  progress: component.activationProgressVisible(),
  progressLabel: component.activationProgressLabel(),
  steps: component.wizardSteps().map((step) => step.state),
};
component.runtimePayload.jobs.push({
  deployment_id: 'pool-a', rank: 0, world_size: 2,
  phase: 'ready', load_stage: 'ready',
  live: true, ownership: 'loaded',
});
// Reconciliation assigns deployment ownership to every retained marker. A
// current live ready rank must likewise outrank the old diagnostic failure.
component.runtimePayload.jobs[0].ownership = 'loaded';
const ready = {
  state: component.deploymentRuntimeState(),
  label: component.deploymentStatus().label,
  progress: component.activationProgressVisible(),
};
process.stdout.write(JSON.stringify({ retry, ready }));
"""
    )

    assert result["retry"] == {
        "state": "loading",
        "label": "Loading",
        "progress": True,
        "progressLabel": "Loading model weights",
        "steps": ["done", "done", "done", "done", "active"],
    }
    assert result["ready"] == {
        "state": "ready",
        "label": "Ready",
        "progress": False,
    }


def test_active_cluster_can_preview_a_signed_three_mac_membership_plan():
    result = _run_wizard(
        """
component.devicesPayload = {
  self: { node_id: 'node-a', friendly_name: 'Studio', paired: true,
    caps: { ram_gb: 256, chip: 'M3 Ultra', jaccl: true },
    addrs: [{ ip: '10.0.0.1', if_type: 'thunderbolt' }] },
  paired: [
    { node_id: 'node-b', friendly_name: 'MacBook', paired: true,
      ssh_target: '10.0.0.2', caps: { ram_gb: 128, chip: 'M5 Max', jaccl: true },
      addrs: [{ ip: '10.0.0.2', if_type: 'thunderbolt' }] },
    { node_id: 'node-c', friendly_name: 'Mini', paired: true,
      ssh_target: '10.0.0.3', caps: { ram_gb: 64, chip: 'M4 Pro', jaccl: true },
      addrs: [{ ip: '10.0.0.3', if_type: 'thunderbolt' }] },
  ],
  discovered: [],
};
component.devicesLoaded = true;
component.deploymentsPayload = [{
  deployment_id: 'pool-a', model: '/models/qwen', backend: 'jaccl',
  serving_mode: 'sharded', tensor_parallel_size: 2,
  target_context_tokens: 32768, mtp_enabled: false,
  execution: { profile: 'balanced', auto_tune: true,
    sampling_rank_only: true, async_overlap: true, cache_affinity: true,
    prompt_cache_ssd: true, prompt_cache_ssd_max_bytes: 21474836480,
    max_kv_size: 32768, ring_connections_per_ip: 1 },
  assignments: [
    { node_id: 'node-a', rank: 0, role: 'workstation' },
    { node_id: 'node-b', rank: 1, role: 'headless' },
  ],
  hosts: [], performance_profiles: [],
}];
component.deploymentsLoaded = true;
component.runtimeLoaded = true;
component.runtimePayload = { jobs: [{ deployment_id: 'pool-a', rank: 0,
  phase: 'ready', live: true, ownership: 'loaded', metrics: {} }], launchers: [] };
const calls = [];
component.apiFetch = async (url, options = {}) => {
  const body = options.body ? JSON.parse(options.body) : null;
  calls.push({ url, body });
  if (url.includes('peer-probe')) {
    return { status: { runtime: { python_executable: '/opt/omlx/python' } } };
  }
  if (url.includes('node-budgets')) {
    return { nodes: [
      { node_id: 'node-a', capacity_bytes: 250000, reserve_bytes: 10000 },
      { node_id: 'node-b', capacity_bytes: 120000, reserve_bytes: 10000 },
      { node_id: 'node-c', capacity_bytes: 60000, reserve_bytes: 10000 },
    ] };
  }
  if (url.includes('autoconfigure')) {
    return {
      ready_to_activate: true, backend: 'jaccl', summary: 'TP3 over RDMA',
      plan: { assignments: body.nodes.map((node, rank) => ({
        node_id: node.node_id, rank, start_layer: 0, end_layer: 64,
        planned_weight_bytes: 1000, tensor_parallel_size: 3,
        tensor_parallel_rank: rank,
      })) },
      activation: { deployment_id: body.deployment_id, model_path: body.model_path,
        nodes: body.nodes, hosts: body.hosts, approved_placement: 'signed-plan' },
    };
  }
  return {};
};
(async () => {
  component.membershipPanelOpen = true;
  await component.previewMembershipExpansion();
  const request = calls.find((call) => call.url.includes('autoconfigure')).body;
  process.stdout.write(JSON.stringify({
    candidates: component.membershipCandidates().map((device) => device.node_id),
    request,
    rows: component.membershipPlanAssignments().map((row) => row.node_id),
    ready: component.membershipProposal.ready_to_activate,
    error: component.membershipError,
  }));
})();
"""
    )

    assert result["candidates"] == ["node-c"]
    assert result["request"]["deployment_id"] == "pool-a"
    assert result["request"]["model_path"] == "/models/qwen"
    assert result["request"]["strategy"] == "auto"
    assert [node["node_id"] for node in result["request"]["nodes"]] == [
        "node-a",
        "node-b",
        "node-c",
    ]
    assert [host["node_id"] for host in result["request"]["hosts"]] == [
        "node-a",
        "node-b",
        "node-c",
    ]
    assert result["rows"] == ["node-a", "node-b", "node-c"]
    assert result["ready"] is True
    assert result["error"] == ""

    template = _read(TEMPLATE)
    assert "data-cluster-v2-add-mac" in template
    assert "data-cluster-v2-membership-preview" in template
    assert "data-cluster-v2-membership-plan" in template
    assert "data-cluster-v2-membership-apply" in template
    assert "data-cluster-v2-plan-add-mac" in template
    assert "data-cluster-v2-plan-add-mac-panel" in template


def test_plan_add_mac_reopens_pairing_without_losing_model_selection():
    result = _run_wizard(
        """
component.devicesPayload = {
  self: { node_id: 'node-a', paired: true, caps: { ram_gb: 64 }, addrs: [] },
  paired: [{ node_id: 'node-b', paired: true, caps: { ram_gb: 64 }, addrs: [] }],
  discovered: [],
};
component.devicesLoaded = true;
component.deploymentsLoaded = true;
component.runtimeLoaded = true;
component.runtimePayload = { jobs: [], launchers: [] };
component.stage = 'plan';
component.selectedModelPath = '/models/qwen';
component.executionProfile = 'throughput';
component.toggleMembershipPanel();
const opened = {
  wizard: component.wizardState(),
  stage: component.stage,
  model: component.selectedModelPath,
  profile: component.executionProfile,
  open: component.membershipPanelOpen,
};
component.toggleMembershipPanel();
const closed = {
  wizard: component.wizardState(),
  stage: component.stage,
  model: component.selectedModelPath,
  profile: component.executionProfile,
  open: component.membershipPanelOpen,
};
process.stdout.write(JSON.stringify({ opened, closed }));
"""
    )

    assert result["opened"] == {
        "wizard": "device_card",
        "stage": None,
        "model": "/models/qwen",
        "profile": "throughput",
        "open": True,
    }
    assert result["closed"] == {
        "wizard": "plan",
        "stage": "plan",
        "model": "/models/qwen",
        "profile": "throughput",
        "open": False,
    }


def test_cold_saved_setups_keep_management_without_claiming_residency():
    result = _run_wizard("""
component.devicesPayload = {
  self: { node_id: 'node-a', friendly_name: 'Node A', caps: {}, addrs: [] },
  paired: [{ node_id: 'node-b', friendly_name: 'Node B', caps: {}, addrs: [], paired: true }],
  discovered: [],
};
component.devicesLoaded = true;
component.deploymentsPayload = [
  { deployment_id: 'ds4-cold', model: '/models/ds4' },
  { deployment_id: 'qwen-cold', model: '/models/qwen', serving_mode: 'disaggregated' },
];
component.deploymentsLoaded = true;
component.runtimePayload = { jobs: [], launchers: [] };
component.runtimeLoaded = true;
component.apiFetch = async () => ({});
component.ensureColdModelSelection();
process.stdout.write(JSON.stringify({
  configured: component.configuredDeployment(),
  stage: component.stage,
  wizard: component.wizardState(),
}));
""")

    assert result["configured"]["deployment_id"] == "ds4-cold"
    assert result["stage"] is None
    assert result["wizard"] == "active"

    template = _read(TEMPLATE)
    assert 'x-for="deployment in deploymentsPayload"' not in template
    assert template.count("data-cluster-v2-model-lifecycle") == 1
    assert template.count("data-cluster-v2-load-weights") == 1
    assert template.count("data-cluster-v2-deactivate") == 1


def test_active_panel_preserves_per_request_prefill_and_decode_rates():
    result = _run_wizard(
        """
component.deploymentsPayload = [{ deployment_id: 'pool-a', model: '/models/m' }];
component.runtimePayload = { jobs: [
  // Rank zero is the coordinator and owns end-to-end request telemetry.
  { deployment_id: 'pool-a', rank: 0, live: true, metrics: {
    active_requests: 2,
    active_request_metrics_truncated: 0,
    active_request_metrics: [
      {
        request_id: 41, status: 'running', prompt_tokens: 4000,
        cached_tokens: 1000, completion_tokens: 0, prefill_tps: 812.4,
        decode_tps: 0, prefill_progress: {
          active: true, processed: 2000, total: 3000,
          average_speed: 812.4,
        },
      },
      {
        request_id: 42, status: 'running', prompt_tokens: 512,
        cached_tokens: 0, completion_tokens: 11, prefill_tps: 905.2,
        decode_tps: 44.25, prefill_progress: {
          active: false, processed: 512, total: 512,
          average_speed: 905.2,
        },
      },
    ],
    last_request: { request_id: 42, status: 'running' },
  } },
  // A peer marker may carry the same requests, but must not duplicate rows.
  { deployment_id: 'pool-a', rank: 1, live: true, metrics: {
    active_requests: 1,
    active_request_metrics: [{ request_id: 999, status: 'running' }],
  } },
] };
const rows = component.deploymentRequestMetrics();
const active = {
  ids: rows.map((row) => row.request_id),
  phases: rows.map((row) => component.requestPhaseLabel(row)),
  prefill: rows.map((row) => component.requestPrefillRate(row)),
  decode: rows.map((row) => component.requestDecodeRate(row)),
  count: component.deploymentRequestCountLabel(),
};
const metrics = component.runtimePayload.jobs[0].metrics;
metrics.active_requests = 0;
metrics.active_request_metrics = [];
metrics.last_request = {
  request_id: 42, status: 'completed', prompt_tokens: 512,
  cached_tokens: 0, completion_tokens: 64, prefill_tps: 905.2,
  decode_tps: 44.25,
};
const completed = component.deploymentRequestMetrics()[0];
metrics.last_request = {
  request_id: 43, status: 'completed', prompt_tokens: 512,
  cached_tokens: 511, completion_tokens: 64, prefill_tps: 1.2,
  decode_tps: 44.25,
};
const cached = component.deploymentRequestMetrics()[0];
process.stdout.write(JSON.stringify({
  active,
  completed: {
    id: completed.request_id,
    history: completed._history,
    phase: component.requestPhaseLabel(completed),
    count: component.deploymentRequestCountLabel(),
  },
  cached: {
    prefill: component.requestPrefillRate(cached),
    detail: component.requestPrefillDetail(cached),
  },
}));
"""
    )

    assert result["active"] == {
        "ids": [41, 42],
        "phases": ["prefill", "decode"],
        "prefill": ["812 tok/s", "905 tok/s"],
        "decode": ["—", "44.3 tok/s"],
        "count": "2 active",
    }
    assert result["completed"] == {
        "id": 42,
        "history": True,
        "phase": "complete",
        "count": "Last completed request",
    }
    assert result["cached"] == {
        "prefill": "Cache hit",
        "detail": "511 reused · 1 new",
    }

    template = _read(TEMPLATE)
    assert "data-cluster-v2-request-speeds" in template
    assert "data-cluster-v2-request-speed-row" in template
    assert "data-cluster-v2-request-prefill-rate" in template
    assert "data-cluster-v2-request-decode-rate" in template


def test_serving_profile_drives_the_server_owned_signed_plan():
    result = _run_wizard(_WIZARD_TWO_MACS + """
component.setExecutionProfile('throughput');
component.modelOptions = [{ model_path: '/models/m', id: 'm' }];
component.selectedModelPath = '/models/m';
component.targetContextTokens = 262144;
let posted = null;
component.apiFetch = async (url, options) => {
  if (url.endsWith('/node-budgets')) return {nodes: component.planNodes()};
  if (url.endsWith('/autoconfigure')) {
    posted = JSON.parse(options.body);
    return {
      plan: { assignments: [], placement_signature: 'a'.repeat(16) },
      activation: {
        execution_profile: posted.execution_profile,
        approved_placement: 'a'.repeat(16),
      },
    };
  }
  return {};
};
(async () => {
  await component.runPlan();
  process.stdout.write(JSON.stringify({
    selected: component.executionProfile,
    posted: posted.execution_profile,
    context: posted.target_context_tokens,
    activation: component.activationRequestBody().execution_profile,
    presets: component.executionProfileOptions().map((option) => ({
      key: option.key,
      limits: option.limits,
    })),
  }));
})();
""")

    assert result["selected"] == "throughput"
    assert result["posted"] == "throughput"
    assert result["context"] == 262144
    assert result["activation"] == "throughput"
    assert result["presets"] == [
        {"key": "interactive", "limits": "4 decode · 2 prompt · batch 2"},
        {"key": "balanced", "limits": "8 decode · 4 prompt · batch 4"},
        {"key": "throughput", "limits": "16 decode · 8 prompt · batch 8"},
    ]

    template = _read(TEMPLATE)
    assert "data-cluster-v2-serving-profile" in template
    assert "data-cluster-v2-serving-profile-option" in template
    assert "data-cluster-v2-context-reservation" in template
    assert "cluster.v2.plan.context_aria" in template
    assert "cluster.v2.plan.batching_note" in template


def test_active_serving_status_uses_resolved_runtime_limits_and_batch_evidence():
    result = _run_wizard(
        """
component.deploymentsPayload = [{
  deployment_id: 'pool-a', model: '/models/m',
  execution: {
    profile: 'balanced', decode_concurrency: 8, prompt_concurrency: 4,
    pipeline_microbatch_size: 4, tuning_reason: 'configured fallback',
  },
}];
component.runtimePayload = { jobs: [{
  deployment_id: 'pool-a', rank: 0, live: true,
  optimizations: { coalesced_batching: {
    enabled: true, active: true, reason: 'model cache is mergeable',
  } },
  metrics: {
    execution: {
      profile: 'throughput', decode_concurrency: 12,
      prompt_concurrency: 6, pipeline_microbatch_size: 8,
      tuning_reason: 'throughput profile auto-tuned for headroom',
    },
    pipeline: {
      microbatch_target: 8,
      last_batch: { coalesced_batch_size: 5 },
    },
  },
}] };
const execution = component.deploymentExecution();
const batch = component.deploymentBatchStatus();
process.stdout.write(JSON.stringify({
  profile: component.deploymentExecutionProfileLabel(),
  decode: execution.decode_concurrency,
  prompt: execution.prompt_concurrency,
  batch,
}));
"""
    )

    assert result["profile"] == "Throughput"
    assert result["decode"] == 12
    assert result["prompt"] == 6
    assert result["batch"]["label"] == "Batched 5"
    assert result["batch"]["target"] == 8
    assert "5 of 8" in result["batch"]["detail"]

    template = _read(TEMPLATE)
    assert "data-cluster-v2-serving-status" in template
    assert "data-cluster-v2-batching-state" in template
    assert "data-cluster-v2-decode-concurrency" in template
    assert "data-cluster-v2-prompt-concurrency" in template


def test_active_profile_change_uses_preview_signature_for_replan_apply():
    result = _run_wizard(
        """
global.setTimeout = () => 0;
component.deploymentsPayload = [{
  deployment_id: 'pool-a', model: '/models/m', target_context_tokens: 131072,
  mtp_enabled: false,
  execution: {
    profile: 'balanced', auto_tune: true, sampling_rank_only: true,
    async_overlap: true, cache_affinity: true, prompt_cache_ssd: true,
    max_kv_size: 131072, ring_connections_per_ip: 4,
    decode_concurrency: 8, prompt_concurrency: 4,
    pipeline_microbatch_size: 4,
  },
}];
const replanBodies = [];
component.apiFetch = async (url, options) => {
  const body = options && options.body ? JSON.parse(options.body) : null;
  if (url.endsWith('/replan')) {
    replanBodies.push(body);
    if (!body.approved_placement) {
      return { mode: 'preview', plan: { placement_signature: 'f'.repeat(16) } };
    }
    return { mode: 'applied', ok: true };
  }
  if (url.endsWith('/deployments')) return { deployments: component.deploymentsPayload };
  if (url.endsWith('/runtime')) return { jobs: [], launchers: [] };
  return {};
};
(async () => {
  await component.previewExecutionProfile('throughput');
  const pending = component.executionReplan && component.executionReplan.profile;
  await component.applyExecutionProfileReplan();
  process.stdout.write(JSON.stringify({
    pending,
    bodies: replanBodies,
    selected: component.executionProfile,
    cleared: component.executionReplan,
  }));
})();
"""
    )

    assert result["pending"] == "throughput"
    assert len(result["bodies"]) == 2
    preview, apply = result["bodies"]
    assert preview["deployment_id"] == "pool-a"
    assert preview["execution_profile"] == "throughput"
    assert preview["max_kv_size"] == 131072
    assert preview["prompt_cache_ssd"] is True
    assert "approved_placement" not in preview
    assert apply == preview | {"approved_placement": "f" * 16}
    assert "nodes" not in apply and "hosts" not in apply
    assert result["selected"] == "throughput"
    assert result["cleared"] is None

    template = _read(TEMPLATE)
    assert "data-cluster-v2-serving-replan" in template
    assert "data-cluster-v2-serving-replan-confirm" in template
    assert "data-cluster-v2-serving-replan-apply" in template


def test_dark_tensor_controls_use_explicit_high_contrast_palette():
    template = _read(TEMPLATE)
    stylesheet = _read("omlx_web/static/css/dashboard.css")

    assert (
        ":data-selected=\"planStrategy === option.key ? 'true' : 'false'\"" in template
    )
    assert "cluster-v2-tensor-segment" in template
    assert ':data-tensor-tone="index % 5"' in template
    assert (
        ".cluster-v2-tensor-segment--2 { background: rgb(var(--palette-zinc-600)); }"
        in stylesheet
    )
    assert '[data-theme="dark"] [data-cluster-v2-active] .text-neutral-600' in stylesheet
    assert "color: rgb(var(--palette-zinc-300)) !important" in stylesheet

    tensor_bar = template.split("data-cluster-v2-split-bar-tensor", 1)[1].split(
        "</template>", 1
    )[0]
    assert "bg-neutral-400 text-white" not in tensor_bar


def test_one_hertz_tick_polls_runtime_ownership_and_not_just_deployments():
    javascript = _read(JAVASCRIPT)
    tick = javascript.split("async tick() {", 1)[1].split("},", 1)[0]

    assert "this.refreshRuntime()" in tick
    assert "runtime: '/admin/api/cluster/runtime'" in javascript
    # A registry record keeps the management panel mounted, while only the
    # runtime selector may return an actually active deployment.
    state = javascript.split("wizardState() {", 1)[1].split("wizardSteps()", 1)[0]
    assert "this.configuredDeployment()" in state
    active = javascript.split("activeDeployment() {", 1)[1].split("},", 1)[0]
    assert "deploymentRuntimeState" in active
    assert "=== 'ready'" in active


def test_cluster_v2_is_the_sole_dashboard_flow():
    dashboard = _read(DASHBOARD)
    template = _read(TEMPLATE)

    assert '{% include "dashboard/_cluster.html" %}' not in dashboard
    assert dashboard.count('{% include "dashboard/_cluster_v2.html" %}') == 1
    assert "clusterLegacyView" not in dashboard + template
    assert "Advanced (legacy)" not in dashboard + template
    # cluster_v2.js loads before dashboard.js so the factory exists when
    # Alpine initializes x-data="clusterV2Wizard()".
    scripts = dashboard.index("js/cluster_v2.js"), dashboard.index("js/dashboard.js")
    assert scripts[0] < scripts[1]


@pytest.mark.parametrize("fixture_name", sorted(p.name for p in FIXTURES.glob("*.json")))
def test_fixtures_are_valid_json_with_state_annotations(fixture_name):
    payload = json.loads((FIXTURES / fixture_name).read_text())

    assert payload, fixture_name
    if fixture_name.startswith("devices_"):
        assert "_state" in payload or "_variant" in payload


# --- Joiner side: this Mac shows the code, the other Mac approves ---------------


def test_joiner_panel_renders_the_six_digit_code():
    template = _read(TEMPLATE)
    javascript = _read(JAVASCRIPT)

    assert "data-cluster-v2-joining" in template
    assert "data-cluster-v2-join-code" in template
    assert "data-cluster-v2-join-countdown" in template
    assert "data-cluster-v2-join-cancel" in template
    assert "joinActive()" in javascript
    assert "joinCountdownLabel()" in javascript
    # An active local join lands in the pairing wizard state, alongside (never
    # instead of) a pending approval arriving from the other Mac.
    state_body = javascript.split("wizardState() {", 1)[1].split("wizardSteps()", 1)[0]
    assert "this.joinActive()" in state_body
    assert "this.pendingApprovals().length" in state_body

    fixture = _fixtures()["pair_join_state.json"]
    assert fixture["state"] == "awaiting_approval"
    assert re.fullmatch(r"\d{6}", fixture["code"])
    assert fixture["coordinator_addr"]
    assert fixture["seconds_remaining"] > 0
    assert fixture["error"] is None


def test_incoming_join_opens_the_code_form_without_a_reload():
    result = _run_wizard(
        """
component.devicesPayload = {
  self: { node_id: 'node-a', friendly_name: 'Node A', caps: { ram_gb: 64 }, addrs: [] },
  paired: [{ node_id: 'node-b', friendly_name: 'Node B', caps: { ram_gb: 64 }, addrs: [], paired: true }],
  discovered: [{
    node_id: 'node-c', friendly_name: 'Node C', caps: { ram_gb: 64 }, addrs: [],
    paired: false, state: 'awaiting_approval',
  }],
};
component.devicesLoaded = true;
component.deploymentsLoaded = true;
component.runtimeLoaded = true;
component.runtimePayload = { jobs: [], launchers: [] };
component.deploymentsPayload = [];
// The wizard has already advanced past device_card; the incoming request
// must still open the pairing section, otherwise only the label changes.
component.stage = 'checks';
const checks = component.wizardState();
component.stage = 'purpose';
const purpose = component.wizardState();
component.stage = null;
const idle = component.wizardState();
// A configured deployment keeps its management panel; the membership panel
// owns the approval there.
component.deploymentsPayload = [{ deployment_id: 'pool-a', model: '/models/m' }];
component.stage = 'checks';
const active = component.wizardState();
process.stdout.write(JSON.stringify({ checks, purpose, idle, active }));
"""
    )

    assert result == {
        "checks": "pairing",
        "purpose": "pairing",
        "idle": "pairing",
        "active": "active",
    }


def test_pending_join_wins_over_stage_shortcuts_in_wizard_state():
    javascript = _read(JAVASCRIPT)
    state = javascript.split("wizardState() {", 1)[1].split("wizardSteps()", 1)[0]

    pairing = state.index("this.pendingApprovals().length")
    # checks/purpose/plan are shortcuts derived from local progress; an
    # incoming request must not depend on a reload to become visible.
    assert pairing < state.index("this.stage === 'checks'")
    assert pairing < state.index("this.stage === 'purpose'")
    assert pairing < state.index("this.choosingModel && this.stage === 'plan'")
    # The active panel keeps its membership approval flow mounted.
    assert state.index("this.configuredDeployment()") < pairing
    assert "this.joinActive()" in state
    assert "this.pairing.target" in state


def test_joiner_poll_drives_approval_and_survives_reloads():
    javascript = _read(JAVASCRIPT)

    # The 1 Hz tick polls the server-owned join snapshot, so a page reload
    # mid-join restores the panel and the coordinator's approval completes
    # by itself — no background thread on either side.
    tick_body = javascript.split("async tick() {", 1)[1].split("},", 1)[0]
    assert "this.refreshJoinState()" in tick_body
    assert "refreshJoinState" in javascript

    approved = javascript.split("snapshot.state === 'approved'", 1)[1]
    assert "joined" in approved
    assert "this.refreshDevices()" in approved
    denied = javascript.split("snapshot.state === 'denied'", 1)[1]
    assert "cluster.v2.toast.join_denied" in denied
    # Denied is terminal server-side; the UI clears it via the cancel endpoint.
    assert "/api/cluster/pair/join/cancel" in javascript


def test_show_code_instead_posts_the_best_reachable_address():
    template = _read(TEMPLATE)
    javascript = _read(JAVASCRIPT)

    assert "data-cluster-v2-show-code" in template
    assert "beginJoinAsJoiner" in javascript
    assert "bestDeviceAddr" in javascript
    # Manual/Thunderbolt/ethernet/tailscale IPv4 win; a bare link-local fe80::
    # (no scope zone) is never dialed.
    assert "'manual', 'tb', 'thunderbolt', 'ethernet', 'tailscale'" in javascript
    assert "fe80:" in javascript
    assert "device.http_port || 8000" in javascript
    # The join POST carries exactly the coordinator address.
    assert re.search(
        r"JSON\.stringify\(\{ coordinator_addr: coordinatorAddr \}\)", javascript
    )


def test_loopback_only_macs_explain_the_bind_instead_of_offering_pairing():
    template = _read(TEMPLATE)
    javascript = _read(JAVASCRIPT)

    assert "device?.caps?.loopback_only" in javascript
    # This Mac: a banner, because peers can never reach it.
    assert 'x-show="deviceLoopbackOnly(selfDevice())"' in template
    assert "cluster.v2.loopback.self_notice" in template
    # Other Macs: every pairing entry point is disabled with the reason shown.
    assert ':disabled="deviceLoopbackOnly(device)"' in template
    assert ':disabled="join.busy || deviceLoopbackOnly(device)"' in template
    assert ':disabled="pairing.busy || deviceLoopbackOnly(device)"' in template
    assert template.count("cluster.v2.loopback.peer_notice") == 2


def test_expired_join_offers_start_again_not_a_dead_end():
    template = _read(TEMPLATE)
    javascript = _read(JAVASCRIPT)

    assert "data-cluster-v2-join-restart" in template
    assert "restartJoin" in javascript
    # Restart reuses the remembered coordinator address with a fresh code.
    assert "this.join.coordinator_addr" in javascript


def test_add_by_ip_validates_then_offers_the_join_flow():
    template = _read(TEMPLATE)
    javascript = _read(JAVASCRIPT)

    assert "data-cluster-v2-manual-add" in template
    assert "data-cluster-v2-manual-input" in template
    assert "data-cluster-v2-manual-submit" in template
    assert "submitManualPeer" in javascript
    assert "/api/cluster/devices/manual" in javascript
    # Client-side validation before any request leaves: IPv4 + optional port.
    assert re.search(r"\^\(\?<host>\\d\{1,3\}\(\?:\\\.\\d\{1,3\}\)\{3\}", javascript)
    assert "match.groups.host" in javascript
    assert "port >= 1 && port <= 65535" in javascript
    # A verified address flows straight into the joiner code panel (one click).
    assert "beginJoinAddr(`${ip}:${port}`" in javascript


# --- Plan step: per-node roles, usable budgets, actionable fit failures -------


def _run_wizard(body: str) -> dict:
    """Execute the shipped wizard component under Node with stubbed I/O."""

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to execute the wizard component")
    # The component resolves its copy through window.t(...); give the sandbox
    # the real English catalog so runtime assertions still read the shipped
    # strings. A missing key falls through as the key itself, which makes a
    # forgotten catalog entry visible rather than silently blank.
    catalog = json.loads(
        (ROOT / "omlx_web/i18n/en.json").read_text(encoding="utf-8")
    )
    script = f"""
{_read(JAVASCRIPT)}
const __catalog = {json.dumps(catalog, ensure_ascii=False)};
global.window = {{ t: (key) => (key in __catalog ? __catalog[key] : key) }};
const component = clusterV2Wizard();
{body}
"""
    result = subprocess.run(
        [node, "-e", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


_WIZARD_TWO_MACS = """
const roles = JSON.parse(
  require('fs').readFileSync(
    %s,
    'utf8',
  ),
).roles;
component.devicesPayload = {
  self: { node_id: 'node-a', friendly_name: 'Node A', caps: { ram_gb: 256 }, addrs: [] },
  paired: [
    { node_id: 'node-b', friendly_name: 'Node B', caps: { ram_gb: 128 }, addrs: [], paired: true },
  ],
  discovered: [],
};
component.roleOptions = roles;
""" % json.dumps(str(FIXTURES / "node_roles.json"))


def test_cuda_guide_pairs_windows_with_a_mac_address_the_pc_can_reach():
    result = _run_wizard(
        """
component.devicesPayload = { self: { http_port: 8000, addrs: [
  { ip: '10.0.1.1' }, { ip: 'fe80::1%en0' }, { ip: '192.168.1.30' },
] } };
component.cudaGuideOs = 'windows';
const command = () => component.cudaGuideSteps().at(-1).command;
global.window.location = { hostname: '127.0.0.1', port: '8000' };
const local = command();
global.window.location = { hostname: '192.168.2.1', port: '8000' };
const lan = command();
process.stdout.write(JSON.stringify({ local, lan }));
""",
    )
    # A bridge or Thunderbolt address usually ends in .1.
    assert "--pair 192.168.1.30:8000" in result["local"]
    # The address the page was opened at reaches this Mac.
    assert "--pair 192.168.2.1:8000" in result["lan"]


def test_plan_step_has_a_per_node_role_picker_with_defaults_unchanged():
    template = _read(TEMPLATE)
    javascript = _read(JAVASCRIPT)

    assert "data-cluster-v2-node-roles" in template
    assert "data-cluster-v2-role-picker" in template
    assert "data-cluster-v2-role-workstation" in template
    assert "data-cluster-v2-role-headless" in template
    assert "data-cluster-v2-usable-budget" in template
    # Default stays as it always was: this Mac workstation, peers headless.
    assert "isSelf ? 'workstation' : 'headless'" in javascript
    # Changing a role re-runs the plan (on click — no silent flips).
    setter = javascript.split("setNodeRole(device, role) {", 1)[1]
    assert "this.runPlan()" in setter
    # planNodes() reads the picker state instead of hard-coding roles.
    planner = javascript.split("planNodes() {", 1)[1].split("},", 1)[0]
    assert "this.nodeRole(self.node_id, true)" in planner
    assert "this.nodeRole(peer.node_id, false)" in planner
    assert "role: 'workstation'" not in planner
    assert "role: 'headless'" not in planner


def test_role_reserve_math_comes_from_the_server_with_a_synced_mirror():
    javascript = _read(JAVASCRIPT)

    assert "'/admin/api/cluster/node-roles'" in javascript
    assert "reserve_fraction" in javascript
    # The offline mirror names the module it mirrors so they cannot drift
    # apart silently.
    mirror = javascript.split("CLUSTER_V2_ROLE_FALLBACK", 2)[0]
    assert "node_role.py" in mirror

    # The fixture pins the endpoint contract against node_role.py itself.
    from omlx.cluster.node_role import ROLES

    fixture = _fixtures()["node_roles.json"]
    assert fixture["default"] == "headless"
    by_key = {role["key"]: role for role in fixture["roles"]}
    for key, role in ROLES.items():
        assert by_key[key]["label"] == role.label
        assert by_key[key]["reserve_bytes"] == role.reserve_bytes
        assert by_key[key]["reserve_fraction"] == role.reserve_fraction


def test_fit_failure_banner_is_actionable_and_never_silent():
    template = _read(TEMPLATE)
    javascript = _read(JAVASCRIPT)

    assert "data-cluster-v2-fit-banner" in template
    assert "data-cluster-v2-fit-switch-headless" in template
    assert "cluster.v2.plan.switch_headless" in template
    assert "parseFitFailure" in javascript
    assert r"at least (?<bytes>\d+) additional bytes" in javascript
    assert "match.groups.bytes" in javascript
    assert "canFixWithHeadless" in javascript
    # The all-headless flip exists only as the click handler — runPlan parses
    # the failure but never mutates nodeRoles itself.
    runner = javascript.split("async runPlan() {", 1)[1].split("},", 1)[0]
    assert "nodeRoles" not in runner.replace("this.nodeRoles =", "")
    assert "this.nodeRoles[" not in runner


def test_role_picker_defaults_reserve_math_and_replan():
    result = _run_wizard(
        _WIZARD_TWO_MACS
        + """
const gib = 1024 ** 3;
let replans = 0;
component.runPlan = async () => { replans += 1; };
component.selectedModelPath = '/models/m';

const defaults = component.planNodes().map((node) => node.role);
component.setNodeRole(component.pairedDevices()[0], 'workstation');
const afterPick = component.planNodes().map((node) => node.role);

process.stdout.write(JSON.stringify({
  defaults,
  afterPick,
  replans,
  wsReserve256: component.reserveBytesFor('workstation', 256 * gib),
  hlReserve128: component.reserveBytesFor('headless', 128 * gib),
  wsReserve64: component.reserveBytesFor('workstation', 64 * gib),
  usableSelf: component.usableGbLabel(component.allDevices()[0]),
}));
""",
    )

    gib = 1024**3
    assert result["defaults"] == ["workstation", "headless"]
    assert result["afterPick"] == ["workstation", "workstation"]
    assert result["replans"] == 1, "a role change re-runs the plan"
    # node_role.py reserve_for(): workstation max(32 GiB, 50%), headless 10%.
    assert result["wsReserve256"] == 128 * gib
    assert result["hlReserve128"] == int(128 * gib * 0.1)
    assert result["wsReserve64"] == 32 * gib, "the 32 GiB floor binds"
    assert result["usableSelf"] == "128 GB usable as Workstation"


def test_fit_failure_parses_the_shortfall_and_flips_only_on_click():
    result = _run_wizard(
        _WIZARD_TWO_MACS
        + """
const gib = 1024 ** 3;
// A 274 GiB model across a 256 + 128 GiB pair: the workstation reserve on
// this Mac is exactly what makes the plan fail.
const failure = component.parseFitFailure(
  'model does not fit the supplied per-node budgets (at least 33294121120 additional bytes required)',
);
const otherError = component.parseFitFailure('ssh: connect timeout');
const noFigure = component.parseFitFailure(
  'model does not fit the supplied per-node budgets',
);
const hugeGap = component.parseFitFailure(
  'model does not fit the supplied per-node budgets (at least 999999999999 additional bytes required)',
);
const rolesBefore = { ...component.nodeRoles };
const gainBytes = component.headlessGainBytes();
component.planFitFailure = failure;
const label = component.fitShortfallLabel();

let replans = 0;
component.runPlan = async () => { replans += 1; };
component.selectedModelPath = '/models/m';
component.switchAllToHeadless().then(() => {
  process.stdout.write(JSON.stringify({
    shortfall: failure.shortfallBytes,
    canFix: failure.canFixWithHeadless,
    gainBytes,
    otherError: otherError === null,
    noFigure: noFigure === null,
    hugeCanFix: hugeGap.canFixWithHeadless,
    rolesBefore,
    rolesAfter: component.planNodes().map((node) => node.role),
    replans,
    label,
  }));
});
""",
    )

    assert result["shortfall"] == 33294121120
    assert result["canFix"] is True
    assert result["gainBytes"] >= result["shortfall"]
    assert result["otherError"] is True
    assert result["noFigure"] is True
    assert result["hugeCanFix"] is False, "an unclosable gap offers no button"
    assert result["rolesBefore"] == {}, "parsing the error never flips roles"
    assert result["rolesAfter"] == ["headless", "headless"]
    assert result["replans"] == 1, "the click re-runs the plan"
    assert result["label"] == "31.0 GiB"


# --- Execution strategy picker, catalogue recommendation, naming fixes -------


def test_strategy_picker_renders_between_models_and_roles():
    template = _read(TEMPLATE)

    assert "data-cluster-v2-strategy-picker" in template
    # Same segmented control as the role picker.
    picker = template.split("data-cluster-v2-strategy-picker", 1)[1].split(
        "data-cluster-v2-node-roles", 1
    )[0]
    assert "segmented__item--active" in picker
    assert ':data-cluster-v2-strategy="option.key"' in picker
    # Green "Recommended" pill, exactly one at a time.
    assert "data-cluster-v2-strategy-recommended" in picker
    assert (
        "bg-green-50 border-green-200 text-green-700" in picker
    )
    assert "recommendedStrategy() === option.key" in picker
    # Disabled options explain themselves.
    assert ":disabled=\"option.disabled\"" in picker
    assert ":title=\"option.disabledReason\"" in picker


def test_persistent_prompt_cache_defaults_on_and_can_be_disabled():
    template = _read(TEMPLATE)
    javascript = _read(JAVASCRIPT)

    assert "data-cluster-v2-prompt-cache-ssd" in template
    assert "data-cluster-v2-prompt-cache-ssd-toggle" in template
    assert "data-cluster-v2-prompt-cache-ssd-cap" in template
    assert "data-cluster-v2-active-cache-mode" in template
    assert 'x-model="promptCacheSsd"' in template
    assert '@change="runPlan()"' in template
    assert "promptCacheSsd: true" in javascript
    assert "promptCacheSsdMaxGiB: 20" in javascript
    assert "prompt_cache_ssd: this.promptCacheSsd" in javascript
    assert "prompt_cache_ssd_max_bytes" in javascript
    assert "cluster.v2.plan.prompt_reuse_blurb" in template

    result = _run_wizard(
        _WIZARD_TWO_MACS + """
const posted = [];
component.apiFetch = async (url, options) => {
  if (url.endsWith('/node-budgets')) return {nodes: component.planNodes()};
  const body = options && options.body ? JSON.parse(options.body) : null;
  if (url.endsWith('/autoconfigure')) {
    posted.push(body.prompt_cache_ssd);
    const signature = body.prompt_cache_ssd ? 'd'.repeat(16) : 'c'.repeat(16);
    return {
      plan: { assignments: [], placement_signature: signature },
      activation: { approved_placement: signature },
    };
  }
  return {};
};
component.modelOptions = [{ model_path: '/models/m', id: 'm' }];
component.selectedModelPath = '/models/m';

(async () => {
  const defaultValue = component.promptCacheSsd;
  await component.runPlan();
  component.promptCacheSsd = false;
  await component.runPlan();
  process.stdout.write(JSON.stringify({
    defaultValue,
    posted,
    enabledLabel: component.deploymentCacheLabel({
      execution: { prompt_cache_ssd: true },
    }),
    disabledLabel: component.deploymentCacheLabel({
      execution: { prompt_cache_ssd: false },
    }),
  }));
})();
""",
    )

    assert result["defaultValue"] is True
    assert result["posted"] == [True, False]
    assert "persistent SSD snapshots" in result["enabledLabel"]
    assert "SSD snapshots off" in result["disabledLabel"]


def test_every_strategy_uses_server_autoconfigure_and_its_tp_choice():
    result = _run_wizard(
        _WIZARD_TWO_MACS + _WIZARD_TIMER_STUBS + """
const bodies = [];
function proposalFor(strategy) {
  const tp = strategy === 'pipeline' ? 1 : 2;
  const signature = (strategy === 'tensor' ? 'a' : strategy === 'auto' ? 'b' : 'c').repeat(16);
  const nodes = [
    { node_id: 'node-a', capacity_bytes: 256 * 1024 ** 3,
      performance: { node_id: 'node-a', rank: 0, marker: strategy + '-a' } },
    { node_id: 'node-b', capacity_bytes: 128 * 1024 ** 3,
      performance: { node_id: 'node-b', rank: 1, marker: strategy + '-b' } },
  ];
  const hosts = [
    { node_id: 'node-a', ssh: '127.0.0.1', ips: ['10.0.0.1'], rdma: [] },
    { node_id: 'node-b', ssh: 'worker', ips: ['10.0.0.2'], rdma: [] },
  ];
  return {
    ready_to_activate: true,
    backend: strategy === 'pipeline' ? 'ring' : 'jaccl',
    performance_probe: { ok: true, status: 'applied_before_staging' },
    plan: { assignments: [], tensor_parallel_size: tp, placement_signature: signature },
    activation: {
      model_path: '/models/m', backend: strategy === 'pipeline' ? 'ring' : 'jaccl',
      nodes, hosts, tensor_parallel_size: tp, approved_placement: signature,
      server_strategy_marker: strategy,
    },
  };
}
component.apiFetch = async (url, options) => {
  if (url.endsWith('/node-budgets')) return {nodes: component.planNodes()};
  const body = options && options.body ? JSON.parse(options.body) : null;
  bodies.push({
    url,
    body,
  });
  if (url.endsWith('/autoconfigure')) return proposalFor(body.strategy);
  return {};
};
component.modelOptions = [{ model_path: '/models/m', id: 'm' }];
component.selectedModelPath = '/models/m';

(async () => {
  component.planStrategy = 'tensor';
  await component.runPlan();
  await component.activatePlan();
  component.planStrategy = 'auto';
  await component.runPlan();
  component.planStrategy = 'pipeline';
  await component.runPlan();
  process.stdout.write(JSON.stringify({
    bodies: bodies.map((entry) => ({
      url: entry.url,
      strategy: entry.body ? entry.body.strategy : null,
      tp: entry.body ? entry.body.tensor_parallel_size : null,
      marker: entry.body ? entry.body.server_strategy_marker : null,
    })),
    stepHint: component.wizardSteps()[3].hint,
  }));
})();
""",
    )

    proposals = [b for b in result["bodies"] if b["url"].endswith("/autoconfigure")]
    deploys = [
        b
        for b in result["bodies"]
        if b["url"].endswith("/deployments") and b["marker"] is not None
    ]
    assert [item["strategy"] for item in proposals] == ["tensor", "auto", "pipeline"]
    assert all(
        item.get("tp") is None for item in proposals
    ), "the browser must not choose a TP degree"
    assert len(deploys) == 1
    assert deploys[0]["url"] == "/admin/api/cluster/deployments"
    assert deploys[0].get("strategy") is None
    assert deploys[0]["tp"] == 2
    assert deploys[0]["marker"] == "tensor"
    # Step-4 hint is strategy-aware only for tensor.
    assert result["stepHint"] == "Layers per Mac"


def test_strategy_picker_disables_tensor_on_a_single_mac():
    result = _run_wizard(
        """
component.devicesPayload = {
  self: { node_id: 'node-a', friendly_name: 'Solo', caps: { ram_gb: 128 }, addrs: [] },
  paired: [],
  discovered: [],
};
const options = component.strategyOptions();
component.setPlanStrategy('tensor');
process.stdout.write(JSON.stringify({
  keys: options.map((option) => option.key),
  tensor: options.find((option) => option.key === 'tensor'),
  afterPick: component.planStrategy,
  recommended: component.recommendedStrategy(),
}));
""",
    )

    assert result["keys"] == ["auto", "tensor", "pipeline"]
    assert result["tensor"]["disabled"] is True
    assert result["tensor"]["disabledReason"] == "Tensor parallelism needs 2+ Macs"
    assert result["afterPick"] == "auto", "a disabled option cannot be picked"
    # No catalogue call ever fired on a one-Mac setup → no badge, no errors.
    assert result["recommended"] == ""


def test_catalogue_drives_the_recommendation_badge_and_capability_locks():
    result = _run_wizard(
        _WIZARD_TWO_MACS
        + """
let catalogueCalls = 0;
component.apiFetch = async (url) => {
  if (url.endsWith('/catalogue')) {
    catalogueCalls += 1;
    return {
      models: [{
        model_path: '/models/m',
        strategy: 'tensor',
        tensor_parallel_size: 2,
        supports_pipeline: false,
        supports_tensor_parallel: true,
      }],
    };
  }
  return {};
};
component.modelOptions = [{ model_path: '/models/m', id: 'm' }];
component.selectedModelPath = '/models/m';
component.planStrategy = 'pipeline';

(async () => {
  await component.loadCatalogue();
  const options = component.strategyOptions();
  process.stdout.write(JSON.stringify({
    catalogueCalls,
    recommended: component.recommendedStrategy(),
    pipeline: options.find((option) => option.key === 'pipeline'),
    afterNormalize: component.planStrategy,
  }));
})();
""",
    )

    assert result["catalogueCalls"] == 1
    assert result["recommended"] == "tensor"
    assert result["pipeline"]["disabled"] is True
    assert "pipeline" in result["pipeline"]["disabledReason"].lower()
    # The current pick was invalidated by the catalogue — fell back to auto.
    assert result["afterNormalize"] == "auto"


def test_catalogue_failure_falls_back_to_the_fast_transport_heuristic():
    result = _run_wizard(
        """
component.devicesPayload = {
  self: { node_id: 'node-a', caps: { ram_gb: 128, jaccl: true }, addrs: [] },
  paired: [
    { node_id: 'node-b', caps: { ram_gb: 128, jaccl: true }, addrs: [], paired: true },
  ],
  discovered: [],
};
component.apiFetch = async (url) => {
  if (url.endsWith('/catalogue')) throw new Error('catalogue down');
  return {};
};
component.modelOptions = [{ model_path: '/models/m', id: 'm' }];
component.selectedModelPath = '/models/m';

const slow = clusterV2Wizard();
slow.devicesPayload = {
  self: { node_id: 'node-c', caps: { ram_gb: 128 }, addrs: [] },
  paired: [
    { node_id: 'node-d', caps: { ram_gb: 128 }, addrs: [], paired: true },
  ],
  discovered: [],
};
slow.apiFetch = async () => { throw new Error('catalogue down'); };
slow.modelOptions = [{ model_path: '/models/m', id: 'm' }];
slow.selectedModelPath = '/models/m';

(async () => {
  await component.loadCatalogue();
  await slow.loadCatalogue();
  process.stdout.write(JSON.stringify({
    fastRecommended: component.recommendedStrategy(),
    fastFailed: component.catalogueFailed,
    slowRecommended: slow.recommendedStrategy(),
  }));
})();
""",
    )

    assert result["fastFailed"] is True
    # Every member on jaccl → tensor; any member off it → auto.
    assert result["fastRecommended"] == "tensor"
    assert result["slowRecommended"] == "auto"


def test_calibration_requires_a_real_model_and_performance_probe_success():
    result = _run_wizard(
        _WIZARD_TWO_MACS + """
const proposal = JSON.parse(
  require('fs').readFileSync(
    %s,
    'utf8',
  ),
);
const bodies = [];
let probeOk = false;
component.apiFetch = async (url, options) => {
  if (url.endsWith('/node-budgets')) return {nodes: component.planNodes()};
  if (url.endsWith('/autoconfigure')) {
    bodies.push(JSON.parse(options.body));
    return {
      ...proposal,
      performance_probe: probeOk
        ? { ok: true, status: 'applied_before_staging' }
        : { ok: false, status: 'memory_fallback', reason: 'probe unavailable' },
    };
  }
  return { deployments: [] };
};

(async () => {
  // The checks step precedes model selection: it must not fake calibration
  // with a placeholder model size.
  await component.runBenchmark();
  const withoutModel = { calls: bodies.length, ...component.checks.benchmark };
  component.modelOptions = [{
    model_path: '/models/m', id: 'm', model_source: 'worker.example',
    locations: [{
      ssh: 'worker.example', python_executable: '/peer/venv/bin/python',
    }],
  }];
  component.selectedModelPath = '/models/m';
  await component.runBenchmark();
  const failedProbe = { ...component.checks.benchmark };
  probeOk = true;
  await component.runBenchmark();
  const passedProbe = { ...component.checks.benchmark };
  process.stdout.write(JSON.stringify({
    withoutModel, failedProbe, passedProbe, bodies,
  }));
})();
""" % json.dumps(str(FIXTURES / "autoconfigure_proposal.json")),
    )

    assert result["withoutModel"]["calls"] == 0
    assert result["withoutModel"]["ok"] is False
    assert "Choose a downloaded model" in result["withoutModel"]["error"]
    assert len(result["bodies"]) == 2
    for body in result["bodies"]:
        assert body["model_path"] == "/models/m"
        assert "model_size_bytes" not in body
        assert body["model_source"] == "worker.example"
        assert body["model_source_python"] == "/peer/venv/bin/python"
        assert body["measure_performance"] is True
        assert body["strategy"] == "auto"
    assert result["failedProbe"]["ok"] is False
    assert result["failedProbe"]["error"] == "probe unavailable"
    assert result["passedProbe"]["ok"] is True


def test_display_model_name_strips_hashes_and_unwraps_hub_dirs():
    result = _run_wizard(
        """
const fortyHex = 'a'.repeat(40);
const names = {
  displayName: component.displayModelName({
    display_name: 'Qwen/Qwen3-32B', model_path: '/some/opaque/path',
  }),
  snapshotHash: component.displayModelName({
    model_path: '/cache/hub/models--mlx-community--Llama-3-8B/snapshots/' + fortyHex,
  }),
  plainTail: component.displayModelName({ model_path: '/models/llama-3-8b' }),
  empty: component.displayModelName({ model_path: '' }),
  shortDelegates: component.shortModelName({ display_name: 'org/name' }),
};
process.stdout.write(JSON.stringify(names));
""",
    )

    assert result["displayName"] == "Qwen/Qwen3-32B"
    assert result["snapshotHash"] == "mlx-community/Llama-3-8B"
    assert result["plainTail"] == "llama-3-8b"
    assert result["empty"] == "this model"
    assert result["shortDelegates"] == "org/name"

    javascript = _read(JAVASCRIPT)
    # filteredModels searches the display name too.
    filtered = javascript.split("filteredModels() {", 1)[1].split("},", 1)[0]
    assert "display_name" in filtered
    # The configured-deployment title no longer re-implements name cleanup.
    template = _read(TEMPLATE)
    assert "displayModelName(configuredDeployment())" in template
    assert "model_path.split('/').filter(Boolean).pop()" not in template


def test_model_presence_counts_only_macs_that_run_the_split():
    result = _run_wizard(
        """
component.devicesPayload = {
  self: { node_id: 'node-a', caps: { ram_gb: 128 }, addrs: [] },
  paired: [
    { node_id: 'node-b', caps: { ram_gb: 64 }, addrs: [], paired: true },
  ],
  discovered: [
    // Unpaired and merely nearby — it never receives layers and must not
    // inflate the denominator.
    { node_id: 'node-c', caps: { ram_gb: 32 }, addrs: [], paired: false, state: 'discovered' },
  ],
};
const everywhere = {
  model_path: '/models/m',
  locations: [{ node_id: 'node-a' }, { node_id: 'node-b' }],
};
const partial = {
  model_path: '/models/m',
  locations: [{ node_id: 'node-a' }],
};
process.stdout.write(JSON.stringify({
  allDevices: component.allDevices().length,
  full: component.modelPresenceLabel(everywhere),
  partial: component.modelPresenceLabel(partial),
}));
""",
    )

    assert result["allDevices"] == 3, "the unpaired Mac still renders as a card"
    assert result["full"] == "on every Mac"
    assert result["partial"] == "on 1 of 2 Macs — copied at activation"


def test_beacon_row_is_an_amber_warning_not_a_red_failure():
    result = _run_wizard(
        """
component.checks.started = true;
component.discoveryHealth = { multicast_rx_within_5s: false };
const row = component.checkRows().find((item) => item.key === 'multicast');
process.stdout.write(JSON.stringify({
  status: row.status,
  label: row.label,
  blocking: component.checksBlockingPass() === false,
}));
""",
    )

    assert result["status"] == "warn"
    assert result["label"] == "Local network permission (discovery only)"

    javascript = _read(JAVASCRIPT)
    template = _read(TEMPLATE)
    # The beacon row's old dead 'unknown' branch and stale stub comments are
    # gone; runtime ownership legitimately has a separate unknown state.
    check_rows = javascript.split("checkRows() {", 1)[1].split(
        "checksBlockingPass()", 1
    )[0]
    assert "'unknown'" not in check_rows
    assert "STUB" not in javascript
    assert "row.status === 'warn'" in template
    assert "triangle-alert" in template
    assert "text-amber-800 bg-amber-50 border border-amber-200" in template


def test_peer_runtime_warning_is_amber_and_does_not_block_planning():
    result = _run_wizard(
        _WIZARD_TWO_MACS
        + """
component.checks.started = true;
component.checks.probes = {
  'node-b': {
    ok: true,
    result: { runtime_warnings: ['Remote runtime uses a fallback interpreter.'] },
  },
};
const row = component.checkRows().find((item) => item.key === 'ssh');
process.stdout.write(JSON.stringify({
  status: row.status,
  detail: row.detail,
  blockingPass: component.checksBlockingPass(),
}));
""",
    )

    assert result == {
        "status": "warn",
        "detail": "Remote runtime uses a fallback interpreter.",
        "blockingPass": True,
    }


def test_split_bar_has_a_tensor_variant_and_width_transitions():
    template = _read(TEMPLATE)
    javascript = _read(JAVASCRIPT)

    assert "data-cluster-v2-split-bar-tensor" in template
    assert "planIsTensor()" in template
    assert "tensorShareLabel()" in template
    assert template.count("transition-[width] duration-700 ease-out") >= 2
    # The contiguous-range bar is untouched for pipeline plans.
    assert "assignment.layer_count / Math.max(planTotalLayers(), 1)" in template
    # planIsTensor keys off the plan payload, not the picker.
    body = javascript.split("planIsTensor() {", 1)[1].split("},", 1)[0]
    assert "tensor_parallel_size" in body


# --- Auto-staging: /stage as phase 1 of activation -----------------------------
#
# The model picker's "copied at activation" promise is literal: when /models
# shows the selected model is not on every plan Mac, activatePlan POSTs
# /admin/api/cluster/stage with {activation, parallel}, polls the job at 1 Hz
# on a dedicated timer (tick() is visibility-gated; activation is not), and
# the identical activation body goes to /deployments when the job completes.

# Timer stubs: the real setInterval would keep the Node event loop alive
# forever and the real setTimeout would slow every notify() by 6 s. Polling
# is driven by calling pollStagingJob() directly.
_WIZARD_TIMER_STUBS = """
global.setInterval = () => 1;
global.clearInterval = () => {};
global.setTimeout = () => 0;
"""

_WIZARD_SIGNED_PROPOSAL = """
const signedProposal = JSON.parse(
  require('fs').readFileSync(
    %s,
    'utf8',
  ),
);
component.planProposal = signedProposal;
component.plan = signedProposal.plan;
""" % json.dumps(str(FIXTURES / "autoconfigure_proposal.json"))

# Two Macs, model complete on node-a only (estimated_size proxy: a holder is
# a location whose size matches the largest location).
_WIZARD_PARTIAL_MODEL = """
component.modelOptions = [{
  model_path: '/models/m', id: 'm', model_source: '127.0.0.1',
  locations: [
    { node_id: 'node-a', ssh: '127.0.0.1', model_path: '/models/m', estimated_size: 1000 },
  ],
}];
component.selectedModelPath = '/models/m';
""" + _WIZARD_SIGNED_PROPOSAL


def _stage_node(node_id, rank, status, files_total=0, files_completed=0,
                bytes_total=0, bytes_completed=0, error=""):
    return {
        "node_id": node_id,
        "rank": rank,
        "status": status,
        "files_total": files_total,
        "files_completed": files_completed,
        "bytes_total": bytes_total,
        "bytes_completed": bytes_completed,
        "files": {},
        "error": error,
    }


def test_staging_runs_before_activation_with_identical_bodies():
    result = _run_wizard(
        _WIZARD_TWO_MACS
        + _WIZARD_TIMER_STUBS
        + _WIZARD_PARTIAL_MODEL
        + """
const calls = [];
const jobId = 'a'.repeat(24);
component.apiFetch = async (url, options) => {
  const body = options && options.body ? JSON.parse(options.body) : null;
  calls.push({ url, body });
  if (url.endsWith('/stage')) {
    return { job_id: jobId, status: 'running', ready: false, error: '', nodes: {
      'node-a': %s,
      'node-b': %s,
    } };
  }
  if (url.endsWith('/stage/' + jobId)) {
    return { job_id: jobId, status: 'completed', ready: true, error: '', nodes: {
      'node-a': %s,
      'node-b': %s,
    } };
  }
  return { deployments: [] };
};

(async () => {
  const missing = component.nodesMissingModel().map((node) => node.node_id);
  await component.activatePlan();
  const midFlight = {
    label: component.stagingButtonLabel(),
    timer: component.stagingTimer !== null,
    busy: component.activateBusy,
    overall: component.stagingOverallLabel(),
  };
  await component.pollStagingJob();
  process.stdout.write(JSON.stringify({
    missing,
    midFlight,
    urls: calls.map((entry) => entry.url),
    stageBody: (calls.find((entry) => entry.url.endsWith('/stage')) || {}).body,
    deployBody: (calls.find(
      (entry) => entry.url.endsWith('/deployments') && entry.body,
    ) || {}).body,
    after: {
      stage: component.stage,
      plan: component.plan,
      planProposal: component.planProposal,
      stagingJob: component.stagingJob,
      stagingActivation: component.stagingActivation,
      timer: component.stagingTimer,
      busy: component.activateBusy,
    },
  }));
})();
"""
        % (
            json.dumps(_stage_node("node-a", 0, "ready")),
            json.dumps(_stage_node("node-b", 1, "copying", 10, 3, 100, 30)),
            json.dumps(_stage_node("node-a", 0, "ready")),
            json.dumps(
                _stage_node("node-b", 1, "ready", 10, 10, 100, 100)
            ),
        ),
    )

    assert result["missing"] == ["node-b"]
    assert result["midFlight"]["label"] == "Copying model to 1 Mac…"
    assert result["midFlight"]["timer"] is True
    assert result["midFlight"]["busy"] is True
    assert result["midFlight"]["overall"] == (
        "Copying the model to your Macs — 1 of 2 ready"
    )
    job_id = "a" * 24
    assert result["urls"] == [
        "/admin/api/cluster/stage",
        f"/admin/api/cluster/stage/{job_id}",
        "/admin/api/cluster/deployments",
        "/admin/api/cluster/deployments",  # refreshDeployments() GET
        "/admin/api/cluster/runtime",  # prove eager activation is resident
    ]
    # The staging activation payload is byte-identical to the activation body
    # (one shared builder), plus parallel as an int.
    assert result["stageBody"]["activation"] == result["deployBody"]
    expected_activation = _fixtures()["autoconfigure_proposal.json"]["activation"]
    assert result["deployBody"] == expected_activation
    assert result["deployBody"]["backend"] == "ring"
    assert result["deployBody"]["tensor_parallel_size"] == 2
    assert [host["node_id"] for host in result["deployBody"]["hosts"]] == [
        "node-a",
        "node-b",
    ]
    assert [node["performance"]["rank"] for node in result["deployBody"]["nodes"]] == [
        0,
        1,
    ]
    assert result["stageBody"]["parallel"] == 4
    assert result["deployBody"]["approved_placement"] == "b" * 16
    # Phase 2 finished: wizard back to its post-activation state.
    assert result["after"] == {
        "stage": None,
        "plan": None,
        "planProposal": None,
        "stagingJob": None,
        "stagingActivation": None,
        "timer": None,
        "busy": False,
    }


def test_staging_is_skipped_when_every_mac_has_the_model():
    result = _run_wizard(
        _WIZARD_TWO_MACS
        + _WIZARD_TIMER_STUBS
        + _WIZARD_SIGNED_PROPOSAL
        + """
component.modelOptions = [{
  model_path: '/models/m', id: 'm', model_source: '127.0.0.1',
  locations: [
    { node_id: 'node-a', ssh: '127.0.0.1', model_path: '/models/m', estimated_size: 1000 },
    { node_id: 'node-b', ssh: 'b.local', model_path: '/models/m', estimated_size: 1000 },
  ],
}];
component.selectedModelPath = '/models/m';
const calls = [];
component.apiFetch = async (url, options) => {
  calls.push({
    url,
    body: options && options.body ? JSON.parse(options.body) : null,
  });
  return { deployments: [] };
};

(async () => {
  await component.activatePlan();
  process.stdout.write(JSON.stringify({
    needsStaging: component.needsStaging(),
    calls,
  }));
})();
""",
    )

    assert result["needsStaging"] is False
    posts = [c for c in result["calls"] if c["body"] is not None]
    assert [c["url"] for c in posts] == ["/admin/api/cluster/deployments"]
    assert all("/stage" not in c["url"] for c in result["calls"])


def test_stage_409_replans_like_activation_409():
    result = _run_wizard(
        _WIZARD_TWO_MACS + _WIZARD_TIMER_STUBS + _WIZARD_PARTIAL_MODEL + """
const calls = [];
component.apiFetch = async (url, options) => {
  calls.push(url);
  if (url.endsWith('/node-budgets')) return {nodes: component.planNodes()};
  if (url.endsWith('/stage')) {
    const error = new Error(
      'The staging request no longer matches the approved plan.',
    );
    error.status = 409;
    throw error;
  }
  if (url.endsWith('/autoconfigure')) {
    const refreshed = JSON.parse(JSON.stringify(signedProposal));
    refreshed.plan.placement_signature = 'c'.repeat(16);
    refreshed.activation.approved_placement = 'c'.repeat(16);
    return refreshed;
  }
  return {};
};

(async () => {
  await component.activatePlan();
  process.stdout.write(JSON.stringify({
    calls,
    signature: component.plan && component.plan.placement_signature,
    stagingError: component.stagingError,
    stagingJob: component.stagingJob,
    busy: component.activateBusy,
    toasts: component.toasts.map((toast) => toast.type + ':' + toast.message),
  }));
})();
""",
    )

    # Signature drift on /stage takes activation's 409 path verbatim: one
    # warning toast, then a fresh signed plan.
    assert result["calls"] == [
        "/admin/api/cluster/stage",
        "/admin/api/cluster/node-budgets",
        "/admin/api/cluster/autoconfigure",
    ]
    assert result["signature"] == "c" * 16
    assert result["stagingError"] == ""
    assert result["stagingJob"] is None
    assert result["busy"] is False
    assert result["toasts"] == [
        "warning:The plan changed since you reviewed it — rebuilding it now."
    ]


def test_stage_poll_404_is_a_retryable_lost_job():
    result = _run_wizard(
        _WIZARD_TWO_MACS
        + _WIZARD_TIMER_STUBS
        + _WIZARD_PARTIAL_MODEL
        + """
const jobId = 'a'.repeat(24);
let stagePosts = 0;
component.apiFetch = async (url, options) => {
  if (url.endsWith('/stage')) {
    stagePosts += 1;
    return { job_id: jobId, status: 'running', ready: false, error: '', nodes: {} };
  }
  if (url.includes('/stage/')) {
    // Coordinator restarted mid-copy: the in-memory job is gone.
    const error = new Error('staging job not found');
    error.status = 404;
    throw error;
  }
  return { deployments: [] };
};

(async () => {
  await component.activatePlan();
  await component.pollStagingJob();
  const lost = {
    error: component.stagingError,
    busy: component.activateBusy,
    timer: component.stagingTimer,
  };
  // Retry = press Activate again; a fresh job is POSTed and the server's
  // size-verified skip turns it into a resume.
  await component.activatePlan();
  process.stdout.write(JSON.stringify({ lost, stagePosts }));
})();
""",
    )

    assert result["lost"] == {
        "error": "The coordinator restarted mid-copy — press Activate to resume.",
        "busy": False,
        "timer": None,
    }
    assert result["stagePosts"] == 2


def test_stage_job_failure_keeps_per_node_errors_and_retries():
    result = _run_wizard(
        _WIZARD_TWO_MACS
        + _WIZARD_TIMER_STUBS
        + _WIZARD_PARTIAL_MODEL
        + """
const jobId = 'a'.repeat(24);
let stagePosts = 0;
component.apiFetch = async (url, options) => {
  if (url.endsWith('/stage')) {
    stagePosts += 1;
    return { job_id: jobId, status: 'running', ready: false, error: '', nodes: {} };
  }
  if (url.includes('/stage/')) {
    return { job_id: jobId, status: 'failed', ready: false,
      error: 'Model staging failed on node-b',
      nodes: {
        'node-a': %s,
        'node-b': %s,
      } };
  }
  return { deployments: [] };
};

(async () => {
  await component.activatePlan();
  await component.pollStagingJob();
  const failed = {
    error: component.stagingError,
    nodeError: component.stagingJob.nodes['node-b'].error,
    busy: component.activateBusy,
    timer: component.stagingTimer,
    toasts: component.toasts.map((toast) => toast.type),
  };
  await component.activatePlan();
  process.stdout.write(JSON.stringify({ failed, stagePosts }));
})();
"""
        % (
            json.dumps(_stage_node("node-a", 0, "ready")),
            json.dumps(
                _stage_node(
                    "node-b", 1, "failed", 10, 4,
                    error="Failed to copy: model-00002-of-00076.safetensors",
                )
            ),
        ),
    )

    assert result["failed"]["error"] == "Model staging failed on node-b"
    # The per-node error stays in the snapshot so the row can show it inline.
    assert (
        result["failed"]["nodeError"]
        == "Failed to copy: model-00002-of-00076.safetensors"
    )
    assert result["failed"]["busy"] is False
    assert result["failed"]["timer"] is None
    assert result["failed"]["toasts"] == ["error"]
    assert result["stagePosts"] == 2


def test_staging_progress_rows_render_per_node_file_counts():
    template = _read(TEMPLATE)

    assert "data-cluster-v2-staging" in template
    assert "data-cluster-v2-staging-node" in template
    assert "data-cluster-v2-staging-error" in template
    assert "data-cluster-v2-staging-dismiss" in template
    # Per-node thin bar over file counts — never a global byte %.
    assert "node.files_completed || 0" in template
    assert "Math.max(node.files_total || 0, 1)" in template
    # The bar is hidden for already-staged nodes (files_total === 0).
    assert 'x-show="(node.files_total || 0) > 0"' in template
    # The activate button narrates the phase.
    assert "stagingButtonLabel()" in template

    result = _run_wizard(
        _WIZARD_TWO_MACS
        + """
component.stagingJob = { job_id: 'a'.repeat(24), status: 'running', nodes: {
  'node-a': %s,
  'node-b': %s,
  'node-c': %s,
  'node-d': %s,
  'node-e': %s,
} };
process.stdout.write(JSON.stringify({
  labels: component.stagingNodes().map((node) => [
    node.node_id, component.stagingNodeLabel(node),
  ]),
  overall: component.stagingOverallLabel(),
}));
"""
        % (
            # Complete model already local: ready with nothing copied.
            json.dumps(_stage_node("node-a", 0, "ready")),
            json.dumps(_stage_node("node-b", 1, "copying", 10, 3, 100, 30)),
            json.dumps(_stage_node("node-c", 2, "queued")),
            json.dumps(
                _stage_node(
                    "node-d", 3, "failed", 4, 1,
                    error="Failed to copy: model-00002-of-00076.safetensors",
                )
            ),
            # The sub-second blip: copying with no files tallied yet.
            json.dumps(_stage_node("node-e", 4, "copying")),
        ),
    )

    assert result["labels"] == [
        ["node-a", "already has it"],
        ["node-b", "copying 3 of 10 files"],
        ["node-c", "waiting"],
        ["node-d", "Failed to copy: model-00002-of-00076.safetensors"],
        ["node-e", "checking what is already there…"],
    ]
    assert result["overall"] == "Copying the model to your Macs — 1 of 5 ready"


def test_old_server_without_stage_degrades_to_direct_activation():
    result = _run_wizard(
        _WIZARD_TWO_MACS
        + _WIZARD_TIMER_STUBS
        + _WIZARD_PARTIAL_MODEL
        + """
const calls = [];
component.apiFetch = async (url, options) => {
  const body = options && options.body ? JSON.parse(options.body) : null;
  calls.push({ url, body });
  if (url.endsWith('/stage')) {
    const error = new Error('Not Found');
    error.status = 404;
    throw error;
  }
  return { deployments: [] };
};

(async () => {
  await component.activatePlan();
  process.stdout.write(JSON.stringify({
    urls: calls.map((entry) => entry.url),
    deployed: calls.some(
      (entry) => entry.url.endsWith('/deployments') && entry.body,
    ),
    stagingError: component.stagingError,
    busy: component.activateBusy,
    toasts: component.toasts.map((toast) => toast.type),
  }));
})();
""",
    )

    # A 404 on the /stage POST means the server predates staging: warn once,
    # then run the pre-staging direct activation rather than bricking.
    assert result["deployed"] is True
    assert result["urls"] == [
        "/admin/api/cluster/stage",
        "/admin/api/cluster/deployments",
        "/admin/api/cluster/deployments",  # refreshDeployments() GET
        "/admin/api/cluster/runtime",  # fail closed if residency is absent
    ]
    assert result["stagingError"] == ""
    assert result["busy"] is False
    assert result["toasts"] == ["warning", "success"]


@pytest.mark.parametrize("delayed_action", ["poll", "begin"])
def test_cancel_ignores_older_join_response(delayed_action):
    result = _run_wizard("""
let resolveOld;
const notices = [];
component.notify = (kind, message) => notices.push({kind, message});
component.apiFetch = (url) => url.endsWith('/cancel')
  ? Promise.resolve({state: 'idle', cleanup_pending: true})
  : new Promise(resolve => { resolveOld = resolve; });
(async () => {
  const pending = DELAYED_ACTION === 'poll'
    ? component.refreshJoinState()
    : component.beginJoinAddr('peer:8000', 'Peer');
  await component.cancelJoin();
  resolveOld({state: 'awaiting_approval', code: '123456'});
  await pending;
  process.stdout.write(JSON.stringify({join: component.join, notices}));
})().catch(error => { console.error(error); process.exit(1); });
""".replace("DELAYED_ACTION", json.dumps(delayed_action)))
    assert result["join"]["state"] == "idle"
    assert result["join"]["busy"] is False
    assert result["join"]["cleanup_pending"] is True
    assert result["notices"][0]["kind"] == "warning"
    assert "on this Mac" in result["notices"][0]["message"]


def test_cleanup_status_does_not_hide_new_join_controls():
    result = _run_wizard("""
component.join.cleanup_pending = true;
component.devicesPayload = {
  self: {node_id: 'self'}, paired: [],
  discovered: [{node_id: 'other', state: 'discovered'}],
};
process.stdout.write(JSON.stringify({state: component.wizardState(), active: component.joinActive()}));
""")
    assert result == {"state": "device_card", "active": False}
    template = _read(TEMPLATE)
    assert "data-cluster-v2-join-cleanup" in template
    assert 'x-show="join.cleanup_pending"' in template


@pytest.mark.parametrize(
    "deployment,expected",
    [
        ({}, True),
        ({"execution": {}}, True),
        ({"execution": {"prompt_cache_ssd": False}}, False),
        ({"prompt_cache_ssd": False}, False),
        ({"execution": {"prompt_cache_ssd": False}, "prompt_cache_ssd": True}, False),
        ({"execution": {"prompt_cache_ssd": True}, "prompt_cache_ssd": False}, True),
    ],
)
def test_ssd_hydration_defaults_on_but_preserves_explicit_choice(deployment, expected):
    result = _run_wizard(
        "component.hydratePlannerFromDeployment(" + json.dumps(deployment) + ");"
        "console.log(JSON.stringify({enabled: component.promptCacheSsd}));"
    )
    assert result["enabled"] is expected


@pytest.mark.parametrize("login", ["", "remote_user@"])
@pytest.mark.parametrize("verified", [True, False])
def test_wizard_selects_verified_address_without_ssh_user(login, verified):
    original = login + "192.0.2.1"
    expected = login + ("192.0.2.2" if verified else "192.0.2.1")
    result = _run_wizard(
        """
const peer = {
    node_id: 'peer', paired: true, ssh_target: ORIGINAL,
    addrs: [{ip: '192.0.2.1'}, {ip: '192.0.2.2'}, {ip: '192.0.2.3'}],
    address_health: {
        '192.0.2.1': {state: 'stale'},
        '192.0.2.2': {state: STATE},
        '192.0.2.3': {state: STATE},
    },
};
component.devicesPayload = {self: null, paired: [peer], discovered: []};
component.planProposal = {activation: {hosts: [{ssh: ORIGINAL}]}};
let probed;
component.apiFetch = async (url, options) => {
    probed = JSON.parse(options.body).ssh;
    return {};
};
(async () => {
    await component.probePeer(peer);
    console.log(JSON.stringify({target: component.sshTargetFor(peer),
        host: component.deploymentHosts()[0].ssh, probed,
        original: component.planProposal.activation.hosts[0].ssh}));
})();
""".replace("ORIGINAL", json.dumps(original)).replace(
            "STATE", json.dumps("verified" if verified else "stale")
        )
    )
    assert result == {
        "target": expected,
        "host": expected,
        "probed": expected,
        "original": original,
    }


def test_explicit_ssh_user_follows_peer_addresses_and_deployment_hosts():
    result = _run_wizard("""
const peer = {node_id: 'worker', paired: true, ssh_user: 'remote_user',
              addrs: [{ip: '192.0.2.10'}]};
component.devicesPayload = {paired: [peer], discovered: [], self: null};
const initial = component.sshTargetFor(peer);
peer.addrs = [{ip: '192.0.2.20'}];
const moved = component.deploymentHosts()[0].ssh;
peer.ssh_target = 'enrolled@worker.example';
const override = component.sshTargetFor(peer);
delete peer.ssh_user;
const fallback = component.sshTargetFor(peer);
console.log(JSON.stringify({initial, moved, override, fallback}));
""")
    assert result == {
        "initial": "remote_user@192.0.2.10",
        "moved": "remote_user@192.0.2.20",
        "override": "remote_user@worker.example",
        "fallback": "enrolled@worker.example",
    }


def test_save_ssh_user_invalidates_old_plan_and_probes():
    result = _run_wizard("""
(async () => {
const peer = {node_id: 'worker', paired: true};
component.sshUserDrafts.worker = ' remote_user ';
component.plan = {old: true}; component.planProposal = {old: true};
component.checks.probes = {worker: {ok: true}};
component.checks.started = true;
let sent;
component.apiFetch = async (url, options) => {
    sent = {url, body: JSON.parse(options.body)};
    return {ssh_user: 'remote_user'};
};
component.refreshDevices = async () => {};
component.notify = () => {};
await component.saveSSHUser(peer);
console.log(JSON.stringify({sent, user: peer.ssh_user, plan: component.plan,
    proposal: component.planProposal, probes: component.checks.probes,
    started: component.checks.started}));
})().catch(error => {console.error(error); process.exit(1);});
""")
    assert result["sent"] == {
        "url": "/api/cluster/devices/worker/ssh-user",
        "body": {"ssh_user": "remote_user"},
    }
    assert result["user"] == "remote_user"
    assert result["plan"] is None and result["proposal"] is None
    assert result["probes"] == {} and result["started"] is False


def test_ssh_repair_form_only_belongs_to_failed_check():
    template = _read(TEMPLATE)
    form = template.index("data-cluster-v2-ssh-user")
    checks = template.index("data-cluster-v2-checks")
    assert form > checks
    assert "row.key === 'ssh' && row.status === 'fail'" in template[checks:form]
    assert "checks.probes[peer.node_id]?.ok === false" in template[checks:form]


def test_ssh_repair_input_allows_dotted_accounts():
    template = _read(TEMPLATE)
    assert 'pattern="[A-Za-z_][A-Za-z0-9_.\\-]{0,63}"' in template


def test_discovered_membership_card_offers_pairing_with_a_persistent_form():
    template = _read(TEMPLATE)
    start = template.index(":key=\"'membership-found-' + device.node_id\"")
    end = template.index("</template>", start)
    assert '@click="beginMembershipPairing(device)"' in template[start:end]
    form = template.index("data-cluster-v2-membership-pairing-form")
    assert form > end
    assert 'x-show="pairing.target"' in template[end:form]
    assert '@keydown.enter="submitPairApproval(pairing.target)"' in template[form:]


def test_membership_pairing_waits_for_the_selected_macs_request():
    result = _run_wizard("""
        (async () => {
            const calls = [];
            component.apiFetch = async (url) => { calls.push(url); return {}; };
            component.devicesPayload = {paired: [], discovered: [{node_id: 'new', state: 'discovered'}]};
            component.beginMembershipPairing({node_id: 'new'});
            component.pairing.code = '123456';
            await component.submitPairApproval(component.pairing.target);
            const missing = {calls: calls.length, error: component.pairing.error};
            component.devicesPayload.discovered = [{node_id: 'other', state: 'awaiting_approval'}];
            await component.submitPairApproval(component.pairing.target);
            console.log(JSON.stringify({missing, calls, open: component.membershipPanelOpen, target: component.pairing.target.node_id}));
        })();
    """)
    # Only the invite to that Mac; nothing is approved before its request.
    assert result["missing"]["calls"] == 1
    assert result["missing"]["error"]
    assert result["calls"] == ["/api/cluster/pair/invite"]
    assert result["open"] is True
    assert result["target"] == "new"


def test_membership_pairing_keeps_the_active_model_until_explicit_replan():
    result = _run_wizard("""
        (async () => {
            const calls = [], probes = [];
            const deployment = {deployment_id: 'existing', model: 'current-model', assignments: [{node_id: 'self'}, {node_id: 'old'}]};
            component.configuredDeployment = () => deployment;
            component.devicesPayload = {paired: [{node_id: 'old', paired: true}], discovered: [{node_id: 'new', state: 'discovered'}]};
            component.executionProfile = 'custom';
            component.selectedModelPath = 'current-model';
            component.beginMembershipPairing({node_id: 'new'});
            component.pairing.code = '123456';
            component.devicesPayload.discovered = [{node_id: 'new', state: 'awaiting_approval'}];
            component.apiFetch = async (url, options) => { calls.push({url, body: JSON.parse(options.body)}); return {}; };
            component.notify = () => {};
            component.refreshDevices = async () => { component.devicesPayload = {paired: [{node_id: 'old', paired: true}, {node_id: 'new', paired: true}], discovered: []}; };
            component.probePeer = async (peer) => probes.push(peer.node_id);
            component.startChecks = () => { throw Error('must not leave active cluster'); };
            component.runPlan = async () => { throw Error('must not replan automatically'); };
            await component.submitPairApproval(component.pairing.target);
            console.log(JSON.stringify({calls, probes, candidates: component.membershipCandidates().map(x => x.node_id), target: component.pairing.target, profile: component.executionProfile, model: component.selectedModelPath, deployment}));
        })();
    """)
    assert result["calls"] == [
        {
            "url": "/api/cluster/pair/approve",
            "body": {"node_id": "new", "code": "123456"},
        }
    ]
    assert result["probes"] == ["new"]
    assert result["candidates"] == ["new"]
    assert result["target"] is None
    assert result["profile"] == "custom"
    assert result["model"] == "current-model"
    assert result["deployment"]["assignments"] == [
        {"node_id": "self"},
        {"node_id": "old"},
    ]


def test_membership_pairing_failure_can_be_retried_or_cancelled():
    result = _run_wizard("""
        (async () => {
            component.devicesPayload = {paired: [], discovered: [{node_id: 'new', state: 'awaiting_approval'}]};
            component.beginMembershipPairing({node_id: 'new'});
            component.pairing.code = '123456';
            component.apiFetch = async () => { throw Object.assign(Error('Incorrect code'), {status: 403}); };
            await component.submitPairApproval(component.pairing.target);
            const failed = {target: component.pairing.target.node_id, code: component.pairing.code, error: component.pairing.error, busy: component.pairing.busy};
            component.cancelPairing();
            console.log(JSON.stringify({failed, target: component.pairing.target}));
        })();
    """)
    assert result["failed"] == {
        "target": "new",
        "code": "123456",
        "error": "Incorrect code",
        "busy": False,
    }
    assert result["target"] is None


def test_forget_cleanup_requires_confirmation_and_ignores_old_poll():
    result = _run_wizard("""
        (async () => {
            const calls = [];
            let resolvePoll;
            component.join = {...component.join, state: 'approved', cleanup_pending: true};
            component.joinApprovedNotified = true;
            component.notify = () => {};
            component.apiFetch = (url, options) => {
                if (!options) return new Promise(resolve => { resolvePoll = resolve; });
                calls.push({url, method: options.method});
                return Promise.resolve({state: 'approved', cleanup_pending: false});
            };
            const poll = component.refreshJoinState();
            await component.forgetJoinCleanup();
            const before = calls.length;
            await component.forgetJoinCleanup();
            resolvePoll({state: 'approved', cleanup_pending: true});
            await poll;
            console.log(JSON.stringify({before, calls, pending: component.join.cleanup_pending, state: component.join.state, busy: component.join.busy}));
        })();
    """)
    assert result["before"] == 0
    assert result["calls"] == [
        {"url": "/api/cluster/pair/join/cleanup", "method": "DELETE"}
    ]
    assert result["pending"] is False
    assert result["state"] == "approved"
    assert result["busy"] is False


def test_forget_cleanup_failure_keeps_banner():
    result = _run_wizard("""
        (async () => {
            const notices = [];
            component.join.cleanup_pending = true;
            component.notify = (kind, text) => notices.push({kind, text});
            component.apiFetch = async () => { throw Error('Storage failed'); };
            await component.forgetJoinCleanup();
            await component.forgetJoinCleanup();
            console.log(JSON.stringify({pending: component.join.cleanup_pending, busy: component.join.busy, notices}));
        })();
    """)
    assert result["pending"] is True
    assert result["busy"] is False
    assert result["notices"] == [{"kind": "error", "text": "Storage failed"}]


def test_forget_local_requires_its_own_confirmation_and_reports_remote_uncertainty():
    result = _run_wizard("""
(async () => {
    const requests = [], notices = [];
    component.apiFetch = async (url, options) => { requests.push({url, method: options.method}); return {}; };
    component.notify = (level, text) => notices.push(text);
    component.refreshDeployments = async () => {};
    component.refreshRuntime = async () => {};
    const deployment = {deployment_id: 'offline'};
    await component.deactivateDeployment(deployment);
    await component.deactivateDeployment(deployment, true);
    const beforeConfirm = requests.length;
    await component.deactivateDeployment(deployment, true);
    console.log(JSON.stringify({beforeConfirm, requests, notices, busy: component.clusterLifecycleBusy}));
})();
""")
    assert result["beforeConfirm"] == 0
    assert result["requests"] == [
        {
            "url": "/admin/api/cluster/deployments/offline?local_only=true",
            "method": "DELETE",
        }
    ]
    assert "Remote shutdown was not verified" in result["notices"][0]
    assert result["busy"] is False


def test_forget_member_and_entire_cluster_have_separate_confirmations():
    result = _run_wizard("""
        (async () => {
            const requests = [];
            component.apiFetch = async (url) => { requests.push(url); return {}; };
            component.notify = () => {};
            component.refreshDevices = async () => {};
            component.refreshDeployments = async () => {};
            component.refreshRuntime = async () => {};
            await component.forgetCluster({node_id: 'peer/a'});
            await component.forgetCluster({node_id: 'peer/b'});
            await component.forgetCluster();
            const beforeConfirm = requests.length;
            await component.forgetCluster();
            await component.forgetCluster({node_id: 'peer/a'});
            await component.forgetCluster({node_id: 'peer/a'});
            console.log(JSON.stringify({beforeConfirm, requests, busy: component.clusterLifecycleBusy}));
        })();
    """)
    assert result["beforeConfirm"] == 0
    assert result["requests"] == [
        "/admin/api/cluster/forget",
        "/admin/api/cluster/forget?node_id=peer%2Fa",
    ]
    assert result["busy"] is False
    template = _read(TEMPLATE)
    assert 'data-cluster-v2-forget-device' in template
    assert 'data-cluster-v2-forget-all' in template


@pytest.mark.parametrize(
    "device, expected",
    [
        pytest.param(
            {"ssh_target": "192.0.2.1", "ssh_user": "paired_user"},
            "paired_user@192.0.2.2",
            id="pairing-account",
        ),
        pytest.param(
            {"ssh_target": "old_user@192.0.2.1", "ssh_user": "saved_user"},
            "saved_user@192.0.2.2",
            id="saved-account-overrides-enrollment",
        ),
        pytest.param(
            {"ssh_target": "enrolled_user@192.0.2.1"},
            "enrolled_user@192.0.2.2",
            id="enrolled-account",
        ),
        pytest.param({}, "192.0.2.2", id="no-enrolled-target-or-account"),
        pytest.param(
            {"ssh_target": "192.0.2.1", "address_health": {}},
            "192.0.2.1",
            id="no-verified-address-keeps-enrollment",
        ),
        pytest.param(
            {"ssh_target": "192.0.2.1", "address_health": None},
            "192.0.2.1",
            id="legacy-peer-without-health",
        ),
        pytest.param(
            {
                "ssh_target": "old_user@192.0.2.1",
                "ssh_user": "saved_user",
                "address_health": {},
            },
            "saved_user@192.0.2.1",
            id="fallback-keeps-account-override",
        ),
        pytest.param(
            {"address_health": {}},
            "192.0.2.1",
            id="legacy-address-fallback",
        ),
        pytest.param(
            {"ssh_target": "192.0.2.1", "addrs": []},
            "192.0.2.1",
            id="health-address-must-be-advertised",
        ),
        pytest.param(
            {"ssh_target": "192.0.2.1", "addrs": None},
            "192.0.2.1",
            id="missing-address-list",
        ),
        pytest.param(
            {
                "address_health": {
                    "192.0.2.1": {"state": "verified"},
                    "192.0.2.2": {"state": "verified"},
                }
            },
            "192.0.2.1",
            id="first-verified-address-is-stable",
        ),
        pytest.param(
            {
                "addrs": [None, {}, {"ip": "fe80::1"}, {"ip": "192.0.2.2"}],
                "address_health": {
                    "fe80::1": {"state": "verified"},
                    "192.0.2.2": {"state": "verified"},
                },
            },
            "192.0.2.2",
            id="ignore-unscoped-link-local-and-malformed-entries",
        ),
        pytest.param(
            {
                "address_health": {
                    "192.0.2.1": {"state": "unverified"},
                    "192.0.2.2": {"state": "stale"},
                    "192.0.2.3": {"state": "verified"},
                }
            },
            "192.0.2.3",
            id="skip-unverified-and-stale-addresses",
        ),
        pytest.param(
            {
                "addrs": [{"ip": "2001:db8::1"}, {"ip": "192.0.2.2"}],
                "address_health": {
                    "2001:db8::1": {"state": "verified"},
                    "192.0.2.2": {"state": "verified"},
                },
            },
            "192.0.2.2",
            id="verified-ipv4-matches-ssh-policy",
        ),
        pytest.param(
            {
                "ssh_target": "192.0.2.1",
                "addrs": [{"ip": "2001:db8::1"}],
                "address_health": {"2001:db8::1": {"state": "verified"}},
            },
            "192.0.2.1",
            id="ipv6-only-health-keeps-enrollment",
        ),
    ],
)
def test_wizard_verified_address_selection_preserves_login_and_fallback(
    device, expected
):
    peer = {
        "node_id": "peer",
        "paired": True,
        "addrs": [{"ip": "192.0.2.1"}, {"ip": "192.0.2.2"}, {"ip": "192.0.2.3"}],
        "address_health": {
            "192.0.2.1": {"state": "stale"},
            "192.0.2.2": {"state": "verified"},
            "192.0.2.3": {"state": "verified"},
        },
        **device,
    }
    result = _run_wizard(
        "console.log(JSON.stringify(component.sshTargetFor(PEER)));".replace(
            "PEER", json.dumps(peer)
        )
    )
    assert result == expected


# --- Cluster Assistant (minimal Screen-3 slice) -------------------------------
#
# One card inside the plan step, driven by the assistant admin API. The states
# are pinned by data-assistant-state; the wizard's 8-state machine is untouched.

ASSISTANT_STATES = (
    "idle",
    "measuring",
    "recommending",
    "measuring_model",
    "planning",
    "applying",
    "starting",
    "loading",
    "ready",
    "failed",
)

ASSISTANT_ENDPOINTS = {
    "/admin/api/cluster/assistant/measure",
    "/admin/api/cluster/assistant/status",
    "/admin/api/cluster/assistant/recommend",
    "/admin/api/cluster/assistant/plan",
    "/admin/api/cluster/assistant/apply",
    "/admin/api/cluster/assistant/start",
    # The automatic constellation search (synchronous, minutes-long).
    "/admin/api/cluster/assistant/search",
    "/admin/api/cluster/assistant/operations",
    # Screen 6: the stop relay to the elected coordinator's own unload route.
    "/admin/api/cluster/assistant/stop",
    # Reload resume: the last persisted deployment (read-only).
    "/admin/api/cluster/assistant/session",
    # The redacted support bundle (read-only).
    "/admin/api/cluster/assistant/diagnostics",
    # The read-only local network preflight.
    "/admin/api/cluster/assistant/network-preflight",
    # The in-app switch for the post-ready measurement.
    "/admin/api/cluster/assistant/performance-probe",
    # The free-space relay (PHASE-2B-RUN.md §9): the only assistant route that
    # changes another Mac's disk.
    "/admin/api/cluster/assistant/models",
    # Tuning: the quick sampler sweep and the minutes-long MTP depth sweep.
    "/admin/api/cluster/assistant/tune",
    "/admin/api/cluster/assistant/tune-redeploy",
    "/admin/api/cluster/assistant/tune-local",
    # Run on this Mac: single-node inference on the origin Mac, as an async job.
    "/admin/api/cluster/assistant/run-local",
}

ASSISTANT_ACTIONS = (
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
)


def _assistant_card() -> str:
    template = _read(TEMPLATE)
    start = template.index("data-cluster-v2-assistant>")
    end = template.index("<!-- /Assistant -->", start)
    return template[start:end]


def test_assistant_states_are_guarded_and_separate_from_the_wizard():
    template = _read(TEMPLATE)
    javascript = _read(JAVASCRIPT)

    for state in ASSISTANT_STATES:
        assert f'data-assistant-state="{state}"' in template, state
        assert re.search(
            rf'x-show="assistantState\(\) === \'{state}\'"[^>]*'
            rf'data-assistant-state="{state}"',
            template,
        ), state

    assert "assistantState() {" in javascript
    # A UI state is never invented that the backend cannot report, and the
    # existing 8-state contract is not rewritten: no shared identifier.
    assert set(ASSISTANT_STATES).isdisjoint(WIZARD_STATES)
    wizard_body = javascript.split("wizardState() {", 1)[1].split(
        "wizardSteps()", 1
    )[0]
    assert "assistant" not in wizard_body


def test_assistant_endpoints_are_only_in_the_api_map_and_allowlisted():
    javascript = _read(JAVASCRIPT)
    template = _read(TEMPLATE)

    found = set(
        re.findall(r"/admin/api/cluster/assistant(?:/[a-z_-]+)*", javascript)
    )
    assert found == ASSISTANT_ENDPOINTS

    api_map = javascript.split("const ASSISTANT_API = {", 1)[1].split(
        "};", 1
    )[0]
    declared = set(
        re.findall(r"/admin/api/cluster/assistant(?:/[a-z_-]+)*", api_map)
    )
    assert declared == ASSISTANT_ENDPOINTS
    # The template never names an endpoint; the component owns the network.
    assert "/admin/api/cluster/assistant" not in template
    assert ASSISTANT_ENDPOINTS <= ALLOWED_ENDPOINTS


def test_assistant_strings_exist_in_the_english_catalog():
    catalog = json.loads(
        (ROOT / "omlx_web/i18n/en.json").read_text(encoding="utf-8")
    )
    text = _read(TEMPLATE) + "\n" + _read(JAVASCRIPT)
    keys = {
        key
        for key in re.findall(r"cluster\.assistant\.[A-Za-z0-9_.]+", text)
        # The action label is a template literal built from the action name;
        # the concrete keys are asserted below instead.
        if not key.endswith(".")
    }
    assert keys
    assert not sorted(key for key in keys if key not in catalog)
    for action in ASSISTANT_ACTIONS:
        assert f"cluster.assistant.action.{action}" in catalog
    assert "cluster.assistant.action.unknown" in catalog
    # English only: no translated copy is added. The repo's own
    # normalize_i18n.py (`make web`) mirrors the key set into every locale with
    # the English value as fallback, so the key sets stay identical and every
    # cluster.assistant.* value is exactly the English one.
    for path in sorted((ROOT / "omlx_web/i18n").glob("*.json")):
        locale = json.loads(path.read_text(encoding="utf-8"))
        assert set(locale) == set(catalog), path.name
        for key, value in locale.items():
            if key.startswith("cluster.assistant."):
                assert value == catalog[key], (path.name, key)


def test_assistant_primary_path_has_no_ip_ssh_rank_layer_or_model_path_field():
    card = _assistant_card()

    # The primary path is buttons and rendered values only. The one exception
    # is the rank-0 pin (frozen decision, 2026-11-15): a single Mac picker for
    # preferred_coordinator — never IP/SSH/rank/layer/model-path inputs.
    assert "<input" not in card
    assert card.count("<select") == 1
    assert "data-cluster-v2-assistant-coordinator" in card
    assert "setAssistantCoordinator" in card
    assert "x-model" not in card
    assert "data-cluster-v2-assistant-entry" in card
    assert "data-cluster-v2-assistant-start-setup" in card
    # The plan-time inputs are the component's frozen object, not inputs.
    assert "runAssistantApply()" in card
    assert "assistantModelLabel()" in card
    assert "assistantContextLabel()" in card


def test_assistant_failure_renders_exactly_one_action_from_error_action():
    card = _assistant_card()

    assert "data-cluster-v2-assistant-error" in card
    assert card.count("data-cluster-v2-assistant-action") == 1
    assert "assistantActionHandler(assistant.error?.action)" in card
    assert "assistantActionLabel(assistant.error?.action)" in card
    # retry_coordinator shows the coordinator instruction next to the action.
    assert "assistant.error?.action === 'retry_coordinator'" in card
    assert "assistantCoordinatorInstruction()" in card


def test_assistant_can_recover_from_a_recommendation_refusal():
    """A refusal whose action the card did not know was a one-way door.

    The recommendation answers with its own near-miss actions
    (``choose_smaller_model``, ``choose_pipeline_model``, ``check_model_path``,
    ``free_disk_space``). The card only knew the twelve frozen ones, so it
    rendered "Continue" and the handler fell through to a toast: after changing
    the model or what a Mac contributes there was no way back to the entry
    button without reloading the page.
    """

    javascript = _read(JAVASCRIPT)
    label_block = javascript.split("assistantActionLabel(action) {", 1)[1].split(
        "assistantActionHandler(action) {", 1
    )[0]
    for action in (
        "choose_smaller_model",
        "choose_pipeline_model",
        "check_model_path",
        "free_disk_space",
    ):
        assert f"'{action}'" in label_block
        assert f"cluster.assistant.action.{action}" not in label_block

    handler_block = javascript.split("assistantActionHandler(action) {", 1)[1].split(
        "resumeAssistantPolling() {", 1
    )[0]
    for action in (
        "choose_smaller_model",
        "choose_pipeline_model",
        "check_model_path",
        "free_disk_space",
    ):
        assert f"case '{action}':" in handler_block
    assert "reopenAssistantPicker()" in handler_block

    catalog = json.loads(_read("omlx_web/i18n/en.json"))
    for action in (
        "choose_smaller_model",
        "choose_pipeline_model",
        "check_model_path",
        "free_disk_space",
    ):
        assert catalog[f"cluster.assistant.action.{action}"]


def test_the_two_mac_fallback_is_explicit_and_signed():
    """An incomplete line offers 2 Macs only through a deliberate click."""

    javascript = _read(JAVASCRIPT)
    template = _read(TEMPLATE)

    assert "confirmAssistantTwoMacFallback()" in javascript
    assert "function confirmAssistantTwoMacFallback" not in javascript
    assert "confirm_two_mac_fallback" in javascript
    inputs = javascript.split("assistantPlanInputs() {", 1)[1].split(
        "async runAssistantPlan", 1
    )[0]
    assert "confirm_two_mac_fallback" in inputs
    # The fallback is re-planned, never applied straight from the refusal body.
    confirm = javascript.split("confirmAssistantTwoMacFallback() {", 1)[1].split(
        "async runAssistantApply", 1
    )[0]
    assert "runAssistantPlan" in confirm
    assert "topologyFallback" in confirm
    assert "data-cluster-v2-assistant-two-mac-fallback" in template
    assert "topology_incomplete" in template
    assert "assistant.missingLinks" in template


def test_the_network_preflight_shows_commands_and_executes_nothing():
    javascript = _read(JAVASCRIPT)
    template = _read(TEMPLATE)

    assert "runAssistantNetworkPreflight()" in javascript
    assert "copyAssistantNetworkCommands()" in javascript
    assert "data-cluster-v2-assistant-network-preflight" in template
    assert "data-cluster-v2-assistant-network-commands" in template
    assert "Nothing is executed" in template
    block = javascript.split("runAssistantNetworkPreflight() {", 1)[1].split(
        "copyAssistantNetworkCommands", 1
    )[0]
    for forbidden in ("exec(", "spawn(", "child_process", "osascript"):
        assert forbidden not in block
    # The response is only stored and rendered; there is no apply call.
    assert "ASSISTANT_API.networkPreflight" in block
    assert "assistant.networkPreflight = plan" in block


def test_copy_diagnostics_is_one_block_without_secrets():
    javascript = _read(JAVASCRIPT)
    template = _read(TEMPLATE)

    block = javascript.split("assistantDiagnosticsText() {", 1)[1].split(
        "copyAssistantDiagnostics() {", 1
    )[0]
    for field in (
        "measurement:",
        "plan:",
        "coordinator:",
        "rank_order:",
        "excluded:",
        "table_offload:",
        "admission:",
        "observed:",
        "warnings:",
    ):
        assert field in block
    for forbidden in ("api_key", "password", "authorization", "ssh"):
        assert forbidden not in block.lower()
    assert "data-cluster-v2-assistant-copy-diagnostics" in template
    # The backend bundle is the source; the local block is the offline fallback.
    assert "ASSISTANT_API.diagnostics" in javascript


def test_copy_model_action_runs_the_assistant_flow():
    """`copy_model` is not a toast any more: apply stages from a peer copy."""

    javascript = _read(JAVASCRIPT)
    handler_block = javascript.split("assistantActionHandler(action) {", 1)[1].split(
        "resumeAssistantPolling() {", 1
    )[0]
    copy_block = handler_block.split("case 'copy_model':", 1)[1].split(
        "default:", 1
    )[0]
    assert "startAssistantForModel" in copy_block
    assert "assistantRecommendedCandidate" in copy_block
    # The message-only fallback is gone: a copy is the real apply/stage flow.
    assert "action.other" not in copy_block
    assert "startAssistantForModel(modelPath) {" in javascript


def test_assistant_plans_lightning_mtp_from_the_model_catalogue():
    """MTP is a plan input, so the assistant must not hardcode it off.

    The draft head occupies rank-0 space: planning ``mtp: false`` while the
    model has an MTP head leaves the cluster decoding without speculative
    tokens (the coordinator logs "no room for the draft head"), and planning
    ``true`` for a model without a head would reserve space for weights that do
    not exist. The catalogue's ``mtp_compatible`` is the fact either way.
    """

    javascript = _read(JAVASCRIPT)
    block = javascript.split("// Assistant — the minimal Screen-3 slice", 1)[1]

    assert "mtp: false" not in block
    # The model-bound measurement, the frozen plan inputs and the single-node
    # run-local request ask the question; the hardware-only entry measurement has
    # no model to ask about (Phase C3: measure hardware, recommend, measure the
    # chosen model).
    assert block.count("mtp: this.assistantMtpFor(modelPath)") == 3

    helper = javascript.split("assistantMtpFor(modelPath) {", 1)[1].split("},", 1)[0]
    assert "mtp_compatible" in helper
    assert "this.selectedModelPath" in helper


def test_assistant_endpoint_is_unreachable_until_ready():
    card = _assistant_card()
    javascript = _read(JAVASCRIPT)

    assert 'x-show="assistantState() === \'ready\'"' in card
    assert 'x-show="assistantReady()"' in card
    assert "assistantReady() ? assistant.endpoint.url : null" in card
    assert "data-cluster-v2-assistant-open-chat" in card
    assert "data-cluster-v2-assistant-connect-agent" in card

    ready = javascript.split("assistantReady() {", 1)[1].split("},", 1)[0]
    assert "state === 'ready'" in ready
    assert "endpoint?.ready === true" in ready
    assert "!this.assistant.stopped" in ready


# --- Cluster Assistant: the automatic constellation search --------------------
#
# Search is an action on the existing card (plan stage), not a new wizard or
# assistant state. The route is synchronous and minutes-long, so the UI is
# single-flight + generation-guarded and only renders measured rates.


def test_assistant_search_affordance_and_api_map_entry_exist():
    javascript = _read(JAVASCRIPT)
    card = _assistant_card()

    # The route is declared in the assistant API map (the allowlist source) and
    # in both contract allowlists.
    assert "search: '/admin/api/cluster/assistant/search'" in javascript
    assert "/admin/api/cluster/assistant/search" in ASSISTANT_ENDPOINTS
    assert "/admin/api/cluster/assistant/search" in ALLOWED_ENDPOINTS

    # The affordance lives inside the assistant card, gated to the plan stage.
    assert "data-cluster-v2-assistant-search" in card
    assert "data-cluster-v2-assistant-search-button" in card
    assert "runAssistantSearch()" in card
    assert "assistantSearchAvailable()" in card
    assert "assistantState() === 'planning' && assistantSearchAvailable()" in card
    # Priority + max-count controls (buttons, keeping the card input-free).
    assert "data-cluster-v2-assistant-search-priority" in card
    assert "data-cluster-v2-assistant-search-max-count" in card
    assert "setAssistantSearchPriority(option.key)" in card
    assert "setAssistantSearchMaxCount(count)" in card
    # Result, busy and error surfaces.
    assert "data-cluster-v2-assistant-search-busy" in card
    assert "data-cluster-v2-assistant-search-best" in card
    assert "data-cluster-v2-assistant-search-results" in card
    assert "data-cluster-v2-assistant-search-errors" in card
    assert "data-cluster-v2-assistant-search-error" in card
    # The card's primary path stays input-free; the rank-0 pin's Mac picker
    # (preferred_coordinator, frozen decision 2026-11-15) is the only select.
    assert card.count("<select") == 1
    assert "data-cluster-v2-assistant-coordinator" in card
    assert "<input" not in card
    assert "x-model" not in card


def test_assistant_search_strings_exist_and_are_mirrored():
    catalog = json.loads(
        (ROOT / "omlx_web/i18n/en.json").read_text(encoding="utf-8")
    )
    keys = [
        "cluster.assistant.search.title",
        "cluster.assistant.search.blurb",
        "cluster.assistant.search.button",
        "cluster.assistant.search.searching",
        "cluster.assistant.search.priority_label",
        "cluster.assistant.search.max_count_label",
        "cluster.assistant.search.best_title",
        "cluster.assistant.search.no_best",
        "cluster.assistant.search.results_title",
        "cluster.assistant.search.table.constellation",
        "cluster.assistant.search.table.macs",
        "cluster.assistant.search.table.measured",
        "cluster.assistant.search.table.feasible",
        "cluster.assistant.search.yes",
        "cluster.assistant.search.no",
        "cluster.assistant.search.not_measured",
        "cluster.assistant.search.errors_title",
        "cluster.assistant.search.failed",
    ]
    priorities = [
        f"cluster.assistant.search.priority.{p}"
        for p in ("balanced", "speed", "ttft", "quality", "context")
    ]
    for key in keys + priorities:
        assert catalog.get(key), key
    # Mirrored into every locale with the English fallback (no translation).
    for path in sorted((ROOT / "omlx_web/i18n").glob("*.json")):
        locale = json.loads(path.read_text(encoding="utf-8"))
        assert set(locale) == set(catalog), path.name
        for key in keys + priorities:
            assert locale.get(key) == catalog[key], (path.name, key)


def test_assistant_search_is_single_flight_generation_guarded_and_measured_only():
    javascript = _read(JAVASCRIPT)
    block = _assistant_method("runAssistantSearch")
    # Busy guard prevents a double POST of the minutes-long request.
    assert "this.assistant.search.busy" in block
    assert "assistantRunning()" in block
    # Generation guard mirrors the polls; a stale answer is dropped.
    assert "this.assistant.generation += 1;" in block
    assert "const generation = this.assistant.generation;" in block
    assert "generation !== this.assistant.generation" in block
    # The search is a background job: POST returns a handle, then the UI polls
    # searchStatus(jobId) until the job completes (generation-guarded).
    assert "ASSISTANT_API.search" in block
    assert "ASSISTANT_API.searchStatus" in block
    assert "job_id" in block
    # The request carries the measurement and the chosen options.
    assert "measurement_id: this.assistant.measurementId" in block
    assert "priority: this.assistant.search.priority" in block
    assert "max_count: this.assistant.search.maxCount" in block

    # Rates are rendered only for a measured constellation (never invented).
    measured = _assistant_method("assistantSearchMeasuredLabel")
    assert "row.measured !== true" in measured

    # No endpoint is offered from a search result: the endpoint stays gated on
    # assistantReady(), and search never transitions the state machine.
    card = _assistant_card()
    search_block = card.split("data-cluster-v2-assistant-search>", 1)[1].split(
        'data-assistant-state="applying"', 1
    )[0]
    assert "assistant.endpoint" not in search_block
    assert "assistantReady()" not in search_block
    assert "assistant.state =" not in block


# --- Screen 5: the handoff (SCREEN-5-HANDOFF.md) ------------------------------
#
# The ready surface opens the *existing* /chat with the cluster's model, and
# Connect agent shows the base URL, the serving model ID and where the key is.
# Both links are built from endpoint.url plus the catalogue's id; nothing is
# derived and no secret is rendered.

CHAT_TEMPLATE = "omlx_web/templates/chat.html"


def _assistant_method(name: str) -> str:
    javascript = _read(JAVASCRIPT)
    for marker in (f"        {name}(", f"        async {name}("):
        if marker in javascript:
            start = javascript.index(marker)
            break
    else:
        raise AssertionError(f"no method {name}")
    depth = 0
    index = start
    while index < len(javascript):
        if javascript[index] == "{":
            depth += 1
        elif javascript[index] == "}":
            depth -= 1
            if depth == 0:
                return javascript[start : index + 1]
        index += 1
    raise AssertionError(f"unterminated method {name}")


def test_assistant_handoff_links_come_from_the_endpoint_and_the_serving_id():
    card = _assistant_card()
    javascript = _read(JAVASCRIPT)

    # One URL each, built by the component, never typed into the markup.
    assert "data-cluster-v2-assistant-open-chat" in card
    assert "data-cluster-v2-assistant-connect-agent" in card
    assert "data-cluster-v2-assistant-agent-base-url" in card
    assert "assistantAgentBaseUrl()" in card
    assert "assistantAgentModelId()" in card
    assert "assistantChatUrl()" in javascript

    chat = _assistant_method("assistantChatUrl")
    assert "assistantReady()" in chat
    assert "assistantServingModelId()" in chat
    assert "encodeURIComponent" in chat
    # An unresolved id links without the parameter; it never guesses one.
    assert "return modelId" in chat

    base = _assistant_method("assistantChatBase")
    assert "endpoint?.url" in base
    assert "assistantCoordinatorIsLocal() ? '/chat' : `${url}/chat`" in base

    serving = _assistant_method("assistantServingModelId")
    assert "model_path === path" in serving
    assert "model?.id" in serving
    # The planner path is never presented as a serving id.
    assert "selectedModelId" not in serving

    agent = _assistant_method("assistantAgentBaseUrl")
    assert "`${url}/v1`" in agent

    open_chat = _assistant_method("openAssistantChat")
    assert "assistantChatUrl()" in open_chat
    assert "'/chat'" not in open_chat


def test_assistant_handoff_is_ready_gated_and_carries_no_secret():
    card = _assistant_card()
    ready = card.index('x-show="assistantState() === \'ready\'"')
    endpoint = card.index('x-show="assistantReady()"')
    chat = card.index("data-cluster-v2-assistant-open-chat")
    agent = card.index("data-cluster-v2-assistant-connect-agent")
    assert ready < endpoint < chat < agent

    # The endpoint stays null until the operation proves it, and no key, token
    # or storage access is rendered in the card.
    assert "assistantReady() ? assistant.endpoint.url : null" in card
    for forbidden in ("api_key", "Authorization", "Bearer", "localStorage"):
        assert forbidden not in card, forbidden


def test_the_chat_page_preselection_is_the_only_new_behaviour():
    template = _read(CHAT_TEMPLATE)

    method = template.split("applyRequestedModelFromUrl() {", 1)[1].split(
        "\n            },", 1
    )[0]
    assert "requestedModelApplied" in method
    assert "isModelAvailableOnServer(requested)" in method
    assert "resolveGatewayModelId(requested)" in method
    # An unknown value falls through to the normal default selection.
    assert "if (!requested || !this.isModelAvailableOnServer(requested)) return;" in method

    reader = template.split("requestedModelFromUrl() {", 1)[1].split(
        "\n            },", 1
    )[0]
    assert "URLSearchParams(window.location.search" in reader
    assert "params.get('model')" in reader
    # Reading the handoff parameter never writes history, fetches, redirects or
    # carries a credential.
    for forbidden in (
        "history.",
        "pushState",
        "replaceState",
        "fetch(",
        "assign(",
        "apiKey",
        "endpoint",
    ):
        assert forbidden not in reader, forbidden
        assert forbidden not in method, forbidden

    # It runs once the model list is non-empty, after the normal default.
    load_models = template.split("async loadModels() {", 1)[1]
    load_models = load_models.split("\n            },\n", 1)[0]
    assert "reconcileCurrentModelAfterLoad(defaultModel);" in load_models
    assert "this.applyRequestedModelFromUrl();" in load_models


def test_assistant_component_never_stages_deploys_or_launches():
    javascript = _read(JAVASCRIPT)
    start = javascript.index("// Assistant — the minimal Screen-3 slice")
    end = javascript.index("// Feedback — toasts, never modals", start)
    block = javascript[start:end]

    for forbidden in (
        "CLUSTER_V2_API",
        "/deployments",
        "/stage",
        "/replan",
        "/autoconfigure",
        "ssh-key",
        "postActivation",
        "activatePlan",
        "run_remote_python",
    ):
        assert forbidden not in block, forbidden
    assert "ASSISTANT_API" in block
    # One poll loop per job, generation-guarded and single-flight.
    assert block.count("setInterval(") == 2
    assert "measurePollBusy" in block
    assert "operationPollBusy" in block
    assert "generation !== this.assistant.generation" in block


def test_assistant_fixtures_pin_the_backend_payload_shapes():
    fixtures = _fixtures()

    running = fixtures["assistant_measure_running.json"]
    ready = fixtures["assistant_measure_ready.json"]
    status_fields = {
        "measurement_id",
        "state",
        "created_at",
        "updated_at",
        "measured_at",
        "model",
        "node_count",
        "nodes",
        "warnings",
        "error",
    }
    for payload in (running, ready):
        assert status_fields <= set(payload)
    assert running["state"] == "running"
    assert ready["state"] == "ready"
    assert ready["node_count"] == 2

    recommendation = fixtures["assistant_recommendation.json"]
    assert {
        "selected_model",
        "selected_context_tokens",
        "candidates",
        "hardware_only",
        "feasible",
        "target_context_tokens",
    } <= set(recommendation)
    # Chapter 3 still requires a model identity: model-less is not available.
    assert recommendation["hardware_only"] is False
    assert recommendation["selected_context_tokens"] > 0
    assert {
        "model_id",
        "model_path",
        "label",
        "feasible",
        "nodes_required",
        "max_context_tokens",
        "measured_speed",
    } <= set(recommendation["candidates"][0])

    plan = fixtures["assistant_plan.json"]
    assert {
        "plan_signature",
        "coordinator_node",
        "rank_order",
        "stages",
        "predicted",
        "observed",
        "endpoint",
    } <= set(plan)
    assert plan["observed"] is None
    assert plan["predicted"]["measured"] is False
    assert plan["predicted"]["critical_stage_seconds"] is None
    assert plan["endpoint"]["ready"] is False
    assert plan["coordinator_node"] in plan["rank_order"]

    operation_fields = {
        "operation_id",
        "kind",
        "state",
        "measurement_id",
        "plan_signature",
        "coordinator_node",
        "progress",
        "endpoint",
        "deployment_id",
        "stopped",
        "observed",
        "warnings",
        "error",
    }
    for name in (
        "assistant_operation_applying.json",
        "assistant_operation_loading.json",
        "assistant_operation_ready.json",
        # Screen 4: the row that is current has a payload per phase, plus the
        # unknown phase and the free-space failure the card must survive.
        "assistant_operation_starting.json",
        "assistant_operation_unknown_phase.json",
        "assistant_error_free_disk_space.json",
    ):
        assert operation_fields <= set(fixtures[name])
    assert fixtures["assistant_operation_applying.json"]["state"] == "applying"
    assert fixtures["assistant_operation_applying.json"]["endpoint"] is None
    loading = fixtures["assistant_operation_loading.json"]
    assert loading["state"] == "loading"
    assert loading["endpoint"]["ready"] is False
    ready_operation = fixtures["assistant_operation_ready.json"]
    assert ready_operation["state"] == "ready"
    assert ready_operation["endpoint"]["ready"] is True
    assert ready_operation["observed"] is None
    # Screen 5: a measured operation carries the one rate block, and nothing
    # else in the payload does.
    measured = fixtures["assistant_operation_ready_observed.json"]
    assert measured["observed"]["decode_tok_per_s"] > 0
    assert measured["observed"]["measured_at"]
    # Screen 6: a ready operation knows its deployment; a stopped one says so
    # while its state stays ready.
    stopped = fixtures["assistant_operation_ready_stopped.json"]
    assert stopped["state"] == "ready"
    assert stopped["stopped"] is True
    assert stopped["deployment_id"]

    retry = fixtures["assistant_error_retry_coordinator.json"]
    assert retry["state"] == "failed"
    assert retry["error"]["action"] == "retry_coordinator"
    assert retry["endpoint"]["ready"] is False

    pair = fixtures["assistant_error_pair_mac.json"]
    assert {"code", "message", "action"} <= set(pair)
    assert pair["action"] == "pair_mac"


def test_a_small_mac_is_forced_to_headless_and_says_so():
    """A <64 GB Mac cannot hold a workstation reserve.

    Live, a 32 GB Mac asking to be a workstation admitted 1.0 GiB and every
    model was refused. The assistant sends it Headless and the card explains it;
    the backend enforces the same rule from measured capacity.
    """

    javascript = _read(JAVASCRIPT)
    card = _assistant_card()

    assert "ASSISTANT_SMALL_MAC_RAM_GB = 64" in javascript
    roles = javascript.split("assistantRoles(nodes) {", 1)[1].split("},", 1)[0]
    assert "ASSISTANT_SMALL_MAC_RAM_GB" in roles
    assert "'headless'" in roles

    notes = javascript.split("assistantSmallMacNotes() {", 1)[1].split("},", 1)[0]
    assert "cluster.assistant.small_mac_note" in notes
    assert "data-cluster-v2-assistant-small-mac" in card
    assert "assistantSmallMacNotes()" in card


# --- Cluster Assistant: Screen 4 (the progress surface) -----------------------
#
# SCREEN-4-PROGRESS.md §2 maps eight labels onto six observable rows. These tests
# pin that mapping, the two combined pairs, the unknown-phase fallback, the
# failure row and the absence of any invented progress. The behaviour of the
# rows under a running flow is covered under Node in
# tests/test_cluster_ui_integration_ui.py.

SCREEN_4_ROWS = (
    "checking",
    "planning",
    "preparing",
    "starting",
    "loading",
    "ready",
)


def _row_table() -> str:
    return _read(JAVASCRIPT).split("const ASSISTANT_PROGRESS_ROWS = [", 1)[1].split(
        "];", 1
    )[0]


def test_screen_4_rows_are_six_labels_in_the_corrected_order():
    javascript = _read(JAVASCRIPT)
    card = _assistant_card()
    table = _row_table()

    keys = re.findall(r"key: '(\w+)'", table)
    assert tuple(keys) == SCREEN_4_ROWS
    # Planning precedes staging: the plan is what staging stages.
    assert keys.index("planning") < keys.index("preparing")

    # Every row is labelled, and the two pairs the backend cannot split are one
    # row each — never a separate "Copying" or "Verifying" row.
    catalog = json.loads(_read("omlx_web/i18n/en.json"))
    labels = re.findall(r"label: '(cluster\.assistant\.progress\.\w+)'", table)
    assert len(labels) == len(SCREEN_4_ROWS)
    for key in labels:
        assert catalog[key]
    assert not {"copying", "verifying", "staged", "applied"} & set(keys)
    assert "copying the model" in catalog["cluster.assistant.progress.preparing"]
    assert "verifying the endpoint" in catalog["cluster.assistant.progress.loading"]

    # The card renders the rows; the component owns the mapping.
    assert "data-cluster-v2-assistant-progress" in card
    assert "assistantProgressRows()" in card
    assert "ASSISTANT_PROGRESS_ROWS" in javascript


def test_screen_4_each_row_is_driven_by_one_backend_value():
    """Exactly one predicate per row, and no timer may ever advance one."""

    javascript = _read(JAVASCRIPT)
    reported = javascript.split("assistantReportedRow() {", 1)[1].split(
        "assistantProgressRow() {", 1
    )[0]

    # 1–2 are the assistant's own states, which the poll sets from the job.
    assert "state === 'measuring' || state === 'measuring_model'" in reported
    assert "'checking'" in reported
    assert "state === 'recommending' || state === 'planning'" in reported
    assert "'planning'" in reported
    # 3–5 are the operation's kind and state, never a guess.
    assert "state === 'applying'" in reported
    assert "'preparing'" in reported
    assert "state === 'starting'" in reported
    assert "'starting'" in reported
    assert "state === 'loading'" in reported
    assert "'loading'" in reported
    # 6 needs both the ready operation and a proven endpoint.
    assert "state === 'ready'" in reported
    assert "this.assistantReady() ? 'ready' : 'loading'" in reported

    for forbidden in ("setTimeout", "setInterval", "Date.now", "performance.now"):
        assert forbidden not in reported
        assert forbidden not in javascript.split(
            "assistantProgressRows() {", 1
        )[1].split("assistantRowMessage(rowKey) {", 1)[0]


def test_screen_4_ready_row_is_unreachable_without_a_ready_endpoint():
    card = _assistant_card()

    # The endpoint block is still gated on assistantReady(), and the Ready row
    # itself is only current when that same predicate is true.
    assert 'x-show="assistantReady()"' in card
    assert "assistantReady() ? assistant.endpoint.url : null" in card
    javascript = _read(JAVASCRIPT)
    ready = javascript.split("assistantReady() {", 1)[1].split("},", 1)[0]
    assert "state === 'ready'" in ready
    assert "endpoint?.ready === true" in ready
    assert "!this.assistant.stopped" in ready


def test_screen_4_unknown_phase_keeps_the_previous_row_and_shows_the_message():
    javascript = _read(JAVASCRIPT)
    fixture = _fixtures()["assistant_operation_unknown_phase.json"]

    # The fixture is a real operation payload whose phase this build cannot map.
    assert fixture["progress"]["phase"] == "future_phase"
    assert fixture["state"] == "loading"
    operation_fields = {
        "operation_id",
        "kind",
        "state",
        "progress",
        "endpoint",
        "error",
    }
    assert operation_fields <= set(fixture)

    known = javascript.split("const ASSISTANT_KNOWN_PHASES = [", 1)[1].split(
        "];", 1
    )[0]
    assert "staging" in known and "loading" in known
    assert "future_phase" not in known
    message = javascript.split("assistantRowMessage(rowKey) {", 1)[1].split(
        "},", 1
    )[0]
    # An unknown phase narrates no row, so the row already shown stays and its
    # raw message is rendered underneath. A known phase of another row never
    # labels this one.
    assert "ASSISTANT_KNOWN_PHASES.includes(phase)" in message
    assert "progress.message" in message
    # Nothing blanks the row list when the phase is unknown.
    rows = javascript.split("assistantProgressRows() {", 1)[1].split(
        "},", 1
    )[0]
    assert "const current = this.assistantProgressRow();" in rows


def test_screen_4_failure_keeps_the_last_phase_and_one_action():
    javascript = _read(JAVASCRIPT)
    card = _assistant_card()
    fixture = _fixtures()["assistant_error_free_disk_space.json"]

    fail = javascript.split("failAssistant(code, message, action) {", 1)[1].split(
        "failAssistantFrom(", 1
    )[0]
    # The row the backend last named is captured before the state becomes
    # "failed", so a failure can say where it stopped.
    assert "this.assistant.failedRow =" in fail
    assert "this.assistantReportedRow() || this.assistant.failedRow" in fail
    assert fail.index("failedRow") < fail.index("state = 'failed'")
    assert "'failed'" in javascript.split("assistantProgressRows() {", 1)[1].split(
        "},", 1
    )[0]

    assert fixture["state"] == "failed"
    assert fixture["error"]["action"] == "free_disk_space"
    assert fixture["progress"]["phase"] == "staging"
    # The failure row keeps the staging row it stopped in.
    assert fixture["progress"]["message"]

    # Still exactly one action button, labelled from error.action.
    assert card.count("data-cluster-v2-assistant-action") == 1
    assert "assistantActionHandler(assistant.error?.action)" in card


def test_screen_4_never_renders_a_rate_a_percentage_or_an_eta():
    # HTML comments are not rendered; the visible markup is what must stay free
    # of invented progress.
    card = re.sub(r"<!--.*?-->", "", _assistant_card(), flags=re.S)

    for forbidden in (
        "%",
        "tok/s",
        "token rate",
        "ETA",
        "percent",
        "bytes copied",
        "predicted.measured",
    ):
        assert forbidden not in card, forbidden
    # A rate may only come from a measured `observed` block, so the ready card
    # renders one line through this helper and the template carries no number.
    assert "assistantObservedLabel()" in card
    assert "cluster.assistant.observed.title" in card
    # The only count is the one the backend reports, as "N of M Macs".
    assert "cluster.assistant.progress_nodes" in _read(JAVASCRIPT)
    assert "assistantProgressCount()" in _read(JAVASCRIPT)


# --- Cluster Assistant: the free-space affordance ----------------------------
#
# PHASE-2B-RUN.md §9 (decision A): an insufficient_disk refusal keeps the one
# action free_disk_space, and the card may then delete an unused model on the Mac
# that is short — two-step confirm, never automatic, never the model the cluster
# is about to run, and only a Mac the user picked.


def test_free_space_panel_lists_only_the_mac_the_user_picked():
    javascript = _read(JAVASCRIPT)
    card = _assistant_card()

    assert "data-cluster-v2-assistant-free-space" in card
    assert "data-cluster-v2-assistant-free-space-node" in card
    assert "data-cluster-v2-assistant-free-space-model" in card
    assert "data-cluster-v2-assistant-free-space-delete" in card
    assert "assistant.error?.action === 'free_disk_space'" in card
    assert "assistant.freeSpaceOpen" in card

    # The node list comes from the measurement; the model list is empty until a
    # Mac is picked, so no other Mac's models are ever shown.
    nodes = javascript.split("assistantFreeSpaceNodes() {", 1)[1].split("},", 1)[0]
    assert "this.assistant.measurementNodes" in nodes
    models = javascript.split("assistantFreeSpaceModels() {", 1)[1].split(
        "async deleteAssistantModel(", 1
    )[0]
    assert "this.assistant.freeSpaceNode" in models
    assert "model.size_bytes" in javascript
    open_panel = javascript.split("openAssistantFreeSpace() {", 1)[1].split(
        "},", 1
    )[0]
    # Nothing is preselected: the Mac the user did not pick is never listed.
    assert "this.assistant.freeSpaceNode = '';" in open_panel


def test_free_space_never_offers_the_model_the_cluster_will_run():
    javascript = _read(JAVASCRIPT)
    models = javascript.split("assistantFreeSpaceModels() {", 1)[1].split(
        "async deleteAssistantModel(", 1
    )[0]

    assert "this.assistant.selectedModel" in models
    assert "this.assistant.selectedModelId" in models
    assert "this.selectedModelPath" in models
    assert "!planned.has(model.model_id)" in models
    assert "!planned.has(model.model_path)" in models


def test_free_space_delete_is_a_two_step_confirm_then_a_remeasure():
    javascript = _read(JAVASCRIPT)
    card = _assistant_card()
    delete = javascript.split("async deleteAssistantModel(model) {", 1)[1].split(
        "// -----------------------------------------------------------------\n        // Ready-only",
        1,
    )[0]

    # First click arms the row; only the second click sends the request.
    assert "if (this.assistant.freeSpaceConfirm !== model.model_id)" in delete
    assert delete.index("freeSpaceConfirm = model.model_id") < delete.index(
        "await this.apiFetch("
    )
    assert "cluster.assistant.free_space.delete_confirm" in javascript
    assert "assistantDeleteLabel(model)" in card
    # The request is the ninth route, and it always names the measurement.
    assert "ASSISTANT_API.models(nodeId, model.model_id)" in delete
    assert "method: 'DELETE'" in delete
    assert "measurement_id=" in delete
    # After a delete the assistant measures again: the freed space is that Mac's
    # number, not the card's assumption.
    assert "await this.runAssistantMeasure();" in delete
    assert "cluster.assistant.free_space.deleted" in delete
    assert "data-cluster-v2-assistant-free-space-error" in card


def test_free_space_action_opens_the_panel_and_never_deletes_from_the_browser():
    javascript = _read(JAVASCRIPT)
    handler = javascript.split("assistantActionHandler(action) {", 1)[1].split(
        "resumeAssistantPolling() {", 1
    )[0]

    assert "case 'free_disk_space':" in handler
    assert "return this.openAssistantFreeSpace();" in handler

    block = javascript.split("// Assistant — the minimal Screen-3 slice", 1)[1]
    for forbidden in ("run_remote_python", "/admin/api/hf/models"):
        assert forbidden not in block, forbidden
    # The browser deletes through the assistant route, never a Mac's own admin
    # route or an SSH call.
    assert "ASSISTANT_API.models(" in block


# --- Cluster Assistant: Screen 6 (state + stop/restart) ----------------------
#
# STOP is one assistant route that relays to the elected coordinator's own
# unload route; Start again re-applies the frozen plan inputs. No re-planning,
# no model/context edit, no Mac changes.

SCREEN_6_STRINGS = (
    "cluster.assistant.stop",
    "cluster.assistant.stop.done",
    "cluster.assistant.stop.failed",
    "cluster.assistant.stopped",
    "cluster.assistant.start_again",
    "cluster.assistant.details.deployment",
)


def test_screen_6_stop_and_start_again_are_ready_gated():
    card = _assistant_card()

    assert "data-cluster-v2-assistant-stop" in card
    assert "data-cluster-v2-assistant-stopped" in card
    assert "data-cluster-v2-assistant-start-again" in card
    assert "stopAssistantCluster()" in card
    assert "startAssistantAgain()" in card

    ready = card.index('x-show="assistantState() === \'ready\'"')
    stop = card.index("data-cluster-v2-assistant-stop")
    assert ready < stop
    # The stopped block is hidden until the backend (or a successful stop)
    # reports `stopped`; Start again lives only inside it.
    assert 'x-show="assistant.stopped"' in card
    stopped = card.index("data-cluster-v2-assistant-stopped")
    again = card.index("data-cluster-v2-assistant-start-again")
    assert stopped < again


def test_screen_6_stop_relays_and_never_unloads_from_the_browser():
    javascript = _read(JAVASCRIPT)

    assert "stop: '/admin/api/cluster/assistant/stop'" in javascript
    block = _assistant_method("stopAssistantCluster")
    assert "ASSISTANT_API.stop" in block
    assert "operation_id" in block
    for forbidden in ("/deployments", "/unload", "fetch(", "ssh", "CLUSTER_V2_API"):
        assert forbidden not in block, forbidden


def test_screen_6_start_again_reuses_the_frozen_plan_inputs():
    block = _assistant_method("startAssistantAgain")

    assert "this.assistant.inputs" in block
    assert "this.assistant.plan?.plan_signature" in block
    assert "this.runAssistantApply()" in block
    # No re-plan, no model/context edit, no Mac changes.
    assert "runAssistantPlan" not in block
    assert "selectedModelPath" not in block
    assert "targetContextTokens" not in block


def test_screen_6_a_stopped_ready_operation_does_not_regress_to_loading():
    reported = _assistant_method("assistantReportedRow")

    assert "if (this.assistant.stopped) return 'ready';" in reported


def test_screen_6_strings_carry_the_english_copy():
    catalog = json.loads(
        (ROOT / "omlx_web/i18n/en.json").read_text(encoding="utf-8")
    )
    for key in SCREEN_6_STRINGS:
        assert catalog.get(key), key


# --- Cluster Assistant: review slice 1 (template safety + flow guards) -------
#
# REVIEW-2026-11-13.md U1/U3/U4/U5 and the small-Mac note: one character-class
# bug shipped because the static tests only grep substrings, and three flow
# guards were missing. These tests execute the shipped component/template.


def test_assistant_template_binding_expressions_all_parse():
    """Every Alpine binding in the template must be compilable JavaScript.

    U1: a misplaced quote in one x-text threw on every dashboard load and
    silently killed the network-preflight warnings. Substring tests cannot
    catch that class; compiling every bound expression can.
    """

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to parse the template expressions")
    template = _read(TEMPLATE)
    bindings = []
    pattern = (
        r'(?:^|\s)(x-[a-z][a-z0-9.:-]*|:[a-z][a-z0-9.:-]*|@[a-z][a-z0-9.:-]*)="([^"]*)"'
    )
    for match in re.finditer(pattern, template):
        name, value = match.group(1), match.group(2)
        # x-for is Alpine iteration syntax, not a JavaScript expression.
        if name.startswith("x-for"):
            continue
        # x-transition:* values are CSS class lists, not expressions.
        if name.startswith("x-transition"):
            continue
        value = (
            value.replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", '"')
            .replace("&#39;", "'")
            .replace("&amp;", "&")
        )
        bindings.append({"attr": name, "code": value})
    script = """
const bindings = %s;
const broken = [];
for (const {attr, code} of bindings) {
  // Alpine evaluates value bindings as expressions and on/init handlers as
  // statement bodies; compile each the way Alpine will.
  const isStatement = attr.startsWith('@') || attr === 'x-init';
  const body = isStatement ? code : 'return (' + code + ');';
  try { new Function(body); } catch (e) { broken.push({attr, code, error: String(e)}); }
}
process.stdout.write(JSON.stringify({checked: bindings.length, broken}));
""" % json.dumps(bindings)
    result = subprocess.run(
        [node, "-e", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["checked"] > 100
    assert payload["broken"] == []


def test_assistant_session_endpoint_is_hidden_when_stopped():
    """U5: a stopped session never offers its dead endpoint link (Screen 6)."""

    result = _run_wizard(
        """
component.assistant.session = {
  state: 'ready', stopped: false, endpoint: {ready: true, url: 'http://172.16.1.11:8099'},
};
const running = component.assistantSessionEndpoint();
component.assistant.session.endpoint.ready = false;
const notReady = component.assistantSessionEndpoint();
component.assistant.session.endpoint.ready = true;
component.assistant.session.stopped = true;
const stopped = component.assistantSessionEndpoint();
component.assistant.session = {
  state: 'applied', stopped: false, endpoint: {ready: true, url: 'http://172.16.1.11:8099'},
};
const applied = component.assistantSessionEndpoint();
process.stdout.write(JSON.stringify({running, notReady, stopped, applied}));
"""
    )
    assert result == {
        "running": "http://172.16.1.11:8099",
        "notReady": "",
        "stopped": "",
        "applied": "",
    }


def test_assistant_small_mac_notes_render_for_a_32_gb_mac():
    """The headless explanation uses the module constant, not this.constructor."""

    result = _run_wizard(
        _WIZARD_TWO_MACS
        + """
component.devicesPayload.paired[0].caps.ram_gb = 32;
const notes = component.assistantSmallMacNotes();
process.stdout.write(JSON.stringify({count: notes.length, note: notes[0] || ''}));
"""
    )
    assert result["count"] == 1
    assert "Node B" in result["note"]
    assert "32" in result["note"]


def test_assistant_use_all_macs_is_guarded_and_generation_safe():
    """U3: the override re-plans only at rest, and bumps the generation."""

    result = _run_wizard(
        """
const planCalls = [];
component.runAssistantPlan = async (gen) => { planCalls.push(gen); };
component.assistant.measurementId = 'm1';
component.assistant.state = 'planning';
const genBefore = component.assistant.generation;
component.toggleAssistantNodeSelection();
const afterFirst = {
  selection: component.assistant.nodeSelection,
  calls: planCalls.length,
  gen: component.assistant.generation,
};
component.assistant.state = 'ready';
component.toggleAssistantNodeSelection();
component.assistant.state = 'planning';
component.assistant.busy = true;
component.toggleAssistantNodeSelection();
component.assistant.busy = false;
component.assistant.stopped = true;
component.toggleAssistantNodeSelection();
component.assistant.stopped = false;
component.assistant.measurementId = '';
component.toggleAssistantNodeSelection();
process.stdout.write(JSON.stringify({
  afterFirst, genBefore,
  finalSelection: component.assistant.nodeSelection,
  totalCalls: planCalls.length,
}));
"""
    )
    assert result["afterFirst"] == {
        "selection": "all",
        "calls": 1,
        "gen": result["genBefore"] + 1,
    }
    # ready, busy, stopped and unmeasured all refuse the override.
    assert result["finalSelection"] == "all"
    assert result["totalCalls"] == 1


def test_assistant_candidates_are_capped_at_32_with_deterministic_priority():
    """U4: a catalogue beyond the backend cap is narrowed, never refused."""

    result = _run_wizard(
        """
const models = [];
for (let i = 0; i < 40; i++) {
  models.push({
    model_path: '/models/m' + i, id: 'm' + i, display_name: 'M' + i,
    estimated_size: (40 - i) * 1000, model_context_length: 8192,
  });
}
models.push({
  model_path: '/models/chosen', id: 'chosen', display_name: 'Chosen',
  estimated_size: 999999, model_context_length: 8192,
});
component.modelOptions = models;
component.selectedModelPath = '/models/chosen';
component.assistant.measurementNodes = [
  {node_id: 'node-b', model_paths: ['/models/m39']},
];
const candidates = component.assistantCandidates();
component.modelOptions = models.slice(0, 32);
component.assistant.measurementNodes = [];
const exact = component.assistantCandidates().length;
process.stdout.write(JSON.stringify({
  count: candidates.length,
  first: candidates[0].model_path,
  second: candidates[1].model_path,
  third: candidates[2].model_path,
  exact,
}));
"""
    )
    assert result["count"] == 32
    # chosen model first, then the peer-held model, then the smallest rest.
    assert result["first"] == "/models/chosen"
    assert result["second"] == "/models/m39"
    assert result["third"] == "/models/m38"
    assert result["exact"] == 32


def test_assistant_action_handler_ignores_double_clicks_while_busy():
    """A second click while busy must not post the same request again."""

    result = _run_wizard(
        """
let fetches = 0;
component.apiFetch = async () => { fetches++; return {measurement_id: 'm1'}; };
component.assistant.busy = true;
component.assistantActionHandler('measure');
component.assistantActionHandler('plan');
const whileBusy = fetches;
component.assistant.busy = false;
let starts = 0;
component.startAssistant = async () => { starts++; };
component.assistantActionHandler('measure');
process.stdout.write(JSON.stringify({whileBusy, starts}));
"""
    )
    assert result == {"whileBusy": 0, "starts": 1}


def test_assistant_card_is_mounted_independently_of_the_wizard_state():
    """B3: the card lives in its own always-mounted section, not in plan."""

    template = _read(TEMPLATE)
    assert "data-cluster-v2-assistant-section" in template
    assert re.search(
        r'<section x-show="assistantSurfaceVisible\(\)"[^>]*'
        r'data-cluster-v2-assistant-section>',
        template,
    )
    # The plan section no longer contains the assistant card.
    plan_start = template.index("data-cluster-v2-plan>")
    plan_end = template.index("</section>", plan_start)
    assert "data-cluster-v2-assistant>" not in template[plan_start:plan_end]
    # The assistant section is mounted before the plan section.
    assert template.index("data-cluster-v2-assistant-section>") < template.index(
        "data-cluster-v2-plan>"
    )


def test_assistant_card_announces_progress_and_failures_and_hides_spinners():
    """L5: the progress rows are a polite live region, the failure banner is an
    alert, and the decorative spinner SVGs are hidden from assistive tech (the
    adjacent text already names the state)."""

    card = _assistant_card()

    progress = re.search(r"<div[^>]*data-cluster-v2-assistant-progress>", card)
    assert progress, "progress container missing"
    assert 'aria-live="polite"' in progress.group(0)

    banner = re.search(r"<div[^>]*data-cluster-v2-assistant-error>", card)
    assert banner, "failure banner missing"
    assert 'role="alert"' in banner.group(0)

    spinners = [
        tag for tag in re.findall(r"<svg\b[^>]*>", card) if "animate-spin" in tag
    ]
    # One per Screen-4 current row plus the seven busy state sentences.
    assert len(spinners) >= 8
    for tag in spinners:
        assert 'aria-hidden="true"' in tag, tag


# --- Assistant UI spec 2026-11-16 (U1-U6) ------------------------------------
#
# The assistant card is a render-mode on the frozen state machine: exactly one
# `data-assistant-screen` div per state, a mode split that keeps today's wizard
# byte-identical behind `Manual setup`, failure screens that own the card, and a
# client-side back/stale model. No backend state is added.

ASSISTANT_SCREENS = (
    "start",
    "check",
    "plan",
    "launch",
    "ready",
    "stopped",
    "failed",
    "choose",
)


def test_assistant_renders_one_screen_per_backend_state():
    template = _read(TEMPLATE)
    javascript = _read(JAVASCRIPT)

    table = re.findall(r"const ASSISTANT_SCREENS = \[(.*?)\];", javascript, re.S)
    assert table, "ASSISTANT_SCREENS missing"
    assert tuple(re.findall(r"'(\w+)'", table[0])) == ASSISTANT_SCREENS

    for name in ASSISTANT_SCREENS:
        assert template.count(f'data-assistant-screen="{name}"') == 1, name
        assert re.search(
            rf"x-show=\"assistantScreen\(\) === '{name}'\"[^>]*"
            rf'data-assistant-screen="{name}"',
            template,
        ), name

    # One function decides the screen; it adds no state and calls no backend.
    body = javascript.split("assistantScreen() {", 1)[1].split(
        "assistantEyebrow() {", 1
    )[0]
    for screen in ASSISTANT_SCREENS:
        assert f"'{screen}'" in body, screen
    assert "apiFetch" not in body
    assert "assistant.state =" not in body


def test_the_assistant_is_the_default_flow_and_manual_setup_is_one_click():
    template = _read(TEMPLATE)
    javascript = _read(JAVASCRIPT)

    assert "clusterMode: 'assistant'" in javascript
    assert "const CLUSTER_MODE_KEY = 'clusterV2.mode';" in javascript
    # The Node component tests construct the factory without localStorage; an
    # unguarded read would break every one of them.
    assert "typeof localStorage !== 'undefined'" in javascript
    assert "setClusterMode(mode) {" in javascript

    for marker in (
        "data-cluster-v2-mode",
        "data-cluster-v2-mode-assistant",
        "data-cluster-v2-mode-manual",
        "data-cluster-v2-assistant-manual-link",
        "data-cluster-v2-manual-progress",
        "data-cluster-v2-manual-sections>",
        "data-cluster-v2-manual-plan>",
    ):
        assert marker in template, marker

    # The assistant section is the first thing after the header; the wizard
    # tiles and both manual regions follow it.
    assistant = template.index("data-cluster-v2-assistant-section>")
    for manual in (
        "data-cluster-v2-manual-progress>",
        "data-cluster-v2-manual-sections>",
        "data-cluster-v2-manual-plan>",
    ):
        assert assistant < template.index(manual), manual

    # Manual content is shown in Manual mode, and in Assistant mode only when
    # the assistant cannot render yet (a fresh Mac still has to pair/install).
    assert "clusterMode === 'manual' || !this.assistantSurfaceVisible()" in javascript

    # Manual mode keeps today's markup: every wizard state section still exists.
    for state in (
        "empty",
        "discovering",
        "device_card",
        "pairing",
        "purpose",
        "helper",
        "accelerator",
        "checks",
        "plan",
        "active",
    ):
        assert template.count(f'data-cluster-v2-state="{state}"') >= 1, state


def test_the_choose_screen_re_measures_only_a_changed_model():
    javascript = _read(JAVASCRIPT)

    choose = _assistant_card().split('data-assistant-screen="choose"', 1)[1]
    assert "<input" not in choose
    assert "x-model" not in choose
    assert "<select" not in choose
    for marker in (
        "data-cluster-v2-assistant-choose-candidate",
        "data-cluster-v2-assistant-choose-context",
        "data-cluster-v2-assistant-choose-context-option",
        "data-cluster-v2-assistant-choose-back",
        "data-cluster-v2-assistant-choose-continue",
    ):
        assert marker in choose, marker

    body = javascript.split("async assistantConfirmChoose() {", 1)[1].split(
        "assistantInputsStale() {", 1
    )[0]
    assert body.count("runAssistantModelMeasure(") == 1
    assert "assistant.selectedModel = draft.model_path" in body
    assert "runAssistantPlan(" in body
    assert "apiFetch" not in body
    # The draft is the only thing the buttons change until Confirm.
    assert "assistantChooseDraft.model_path = modelPath" in javascript
    assert "assistantChooseDraft.context_tokens = value" in javascript


def test_a_failure_owns_the_card_with_one_action_and_a_way_back():
    card = _assistant_card()
    failed = card.split('data-assistant-screen="failed"', 1)[1].split(
        'data-assistant-screen="choose"', 1
    )[0]

    assert "data-cluster-v2-assistant-error" in failed
    assert failed.count("data-cluster-v2-assistant-action") == 1
    assert "assistantActionHandler(assistant.error?.action)" in failed
    assert "data-cluster-v2-assistant-retry" in failed
    assert "data-cluster-v2-assistant-back" in failed
    assert "data-cluster-v2-assistant-failure-details" in failed
    assert "data-cluster-v2-assistant-topology" in failed
    assert "data-cluster-v2-assistant-free-space" in failed
    # No other screen's primary action lives inside the failure.
    assert "data-cluster-v2-assistant-start-setup" not in failed
    assert "data-cluster-v2-assistant-open-chat" not in failed

    # Warnings are amber notes on the plan, never an error (A7).
    warnings = re.search(
        r"<div[^>]*data-cluster-v2-assistant-plan-warnings", card
    )
    assert warnings, "plan warnings missing"
    assert "amber" in warnings.group(0)


def test_back_reaches_applied_and_retry_reenters_the_failed_step():
    template = _read(TEMPLATE)
    javascript = _read(JAVASCRIPT)

    back = javascript.split("assistantBackAllowed() {", 1)[1].split(
        "assistantCanChoose() {", 1
    )[0]
    assert "!this.assistantRunning()" in back
    assert "this.assistant.state !== 'ready'" in back
    assert "!this.assistant.stopped" in back

    retry = javascript.split("async retryAssistantStep() {", 1)[1].split(
        "assistantFailureDetails() {", 1
    )[0]
    assert "runAssistantStart(generation)" in retry
    assert "runAssistantApply()" in retry
    assert "runAssistantPlan(generation)" in retry
    assert "startAssistant()" in retry
    # An applied deployment is retried in place, never re-applied.
    step = javascript.split("assistantFailureStep() {", 1)[1].split(
        "assistantRetryLabel() {", 1
    )[0]
    assert "applyState === 'applied'" in step
    assert "return 'start'" in step

    # Start setup re-plans a stale plan instead of applying it.
    assert (
        "assistantInputsStale() ? runAssistantPlan() : runAssistantApply()"
        in template
    )

    stale = javascript.split("assistantInputsStale() {", 1)[1].split(
        "assistantRowStale(rowKey) {", 1
    )[0]
    assert "this.assistant.inputs" in stale
    assert "selectedModel" in stale
    assert "selectedContextTokens" in stale
    assert "preferred_coordinator" in stale

    # Going back from a failure returns to the plan review, not to idle.
    back_to_plan = javascript.split("assistantBackToPlan() {", 1)[1].split(
        "assistantOpenChoose() {", 1
    )[0]
    assert "this.assistant.error = null" in back_to_plan
    assert "this.assistant.state = 'planning'" in back_to_plan


def test_assistant_screen_copy_exists_in_the_english_catalog():
    catalog = json.loads(_read("omlx_web/i18n/en.json"))
    for key in (
        "cluster.assistant.mode.assistant",
        "cluster.assistant.mode.manual",
        "cluster.assistant.mode.manual_link",
        "cluster.assistant.screen.check.title",
        "cluster.assistant.screen.plan.title",
        "cluster.assistant.screen.launch.title",
        "cluster.assistant.screen.ready.title",
        "cluster.assistant.screen.stopped.title",
        "cluster.assistant.screen.failed.title",
        "cluster.assistant.screen.choose.title",
        "cluster.assistant.screen.choose.continue",
        "cluster.assistant.back.plan",
        "cluster.assistant.stale.banner",
        "cluster.assistant.stale.step",
        "cluster.assistant.retry.plan",
        "cluster.assistant.retry.start",
        "cluster.assistant.stepper.checking",
        "cluster.assistant.stepper.ready",
        "cluster.assistant.macs.combined",
        "cluster.assistant.macs.measured_ago",
        "cluster.assistant.performance.title",
        "cluster.assistant.failure.topology",
        "cluster.assistant.failure.wired",
        "cluster.assistant.show_details",
    ):
        assert catalog.get(key), key
