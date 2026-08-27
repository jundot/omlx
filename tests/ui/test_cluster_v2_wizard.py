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

from omlx.admin import routes as admin_routes

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/ui/fixtures/cluster_v2"

TEMPLATE = "omlx/admin/templates/dashboard/_cluster_v2.html"
JAVASCRIPT = "omlx/admin/static/js/cluster_v2.js"
DASHBOARD = "omlx/admin/templates/dashboard.html"

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
    "/admin/api/cluster/tp-layout-qualifications",
}


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text()


def _fixtures():
    return {path.name: json.loads(path.read_text()) for path in FIXTURES.glob("*.json")}


def test_dashboard_renders_every_wizard_state():
    rendered = admin_routes.templates.get_template("dashboard.html").render()

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
    assert "slice(0, 2" not in javascript
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
    assert "this.clusterLegacyView" in javascript
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
    assert "Version mismatch across your Macs" in template
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
    assert "Local Network" in template
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
    assert "Tailscale control" in javascript
    assert "Inference: JACCL over Thunderbolt RDMA" in javascript
    assert "device links below are control/discovery routes" in template
    assert "planIsHybrid()" in javascript
    assert "Hybrid TP×pipeline" in javascript
    assert "pipeline stages" in javascript
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
    result = _run_wizard(
        """
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
"""
    )

    assert result["armed"] == "pool-a"
    assert result["deploymentAfterUnload"] is None
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
    result = _run_wizard(
        f"""
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
"""
    )

    detached = result["runtime_detached.json"]
    assert detached["runtimeState"] == "configured"
    assert detached["wizardState"] != "active"
    assert detached["configured"] is None
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


def test_cold_saved_setups_open_one_model_picker_instead_of_a_fake_active_card():
    result = _run_wizard(
        """
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
"""
    )

    assert result == {"configured": None, "stage": "plan", "wizard": "plan"}

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
        "phases": ["Prefill", "Decode"],
        "prefill": ["812 tok/s", "905 tok/s"],
        "decode": ["—", "44.3 tok/s"],
        "count": "2 active",
    }
    assert result["completed"] == {
        "id": 42,
        "history": True,
        "phase": "Complete",
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
    result = _run_wizard(
        _WIZARD_TWO_MACS
        + """
component.setExecutionProfile('throughput');
component.modelOptions = [{ model_path: '/models/m', id: 'm' }];
component.selectedModelPath = '/models/m';
component.targetContextTokens = 262144;
let posted = null;
component.apiFetch = async (url, options) => {
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
"""
    )

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
    assert 'aria-label="Distributed context reservation"' in template
    assert "There is no separate batching switch" in template


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
  serving_mode: 'disaggregated', prefill_rank: 1, decode_rank: 0,
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
    assert preview["serving_mode"] == "disaggregated"
    assert preview["prefill_rank"] == 1
    assert preview["decode_rank"] == 0
    assert "approved_placement" not in preview
    assert apply == preview | {"approved_placement": "f" * 16}
    assert "nodes" not in apply and "hosts" not in apply
    assert result["selected"] == "throughput"
    assert result["cleared"] is None

    template = _read(TEMPLATE)
    assert "data-cluster-v2-serving-replan" in template
    assert "data-cluster-v2-serving-replan-confirm" in template
    assert "data-cluster-v2-serving-replan-apply" in template


def test_phase_runtime_helpers_show_roles_and_rdma_handoff():
    result = _run_wizard(
        """
component.deploymentsPayload = [{
  deployment_id: 'phase-a', serving_mode: 'disaggregated',
  prefill_rank: 1, decode_rank: 0,
  assignments: [
    { node_id: 'm3', rank: 0 },
    { node_id: 'm5', rank: 1 },
  ],
}];
component.runtimePayload = { jobs: [{
  deployment_id: 'phase-a', rank: 0, live: true,
  metrics: { phase_split: {
    handoffs_completed: 4, last_handoff_bytes: 2147483648,
    last_handoff_arrays: 128, last_handoff_seconds: 0.25,
    last_handoff_bytes_per_second: 8589934592, queue_depth: 2,
  } },
}] };
process.stdout.write(JSON.stringify({
  rate: component.phaseHandoffRateLabel(),
  detail: component.phaseHandoffDetail(),
  prefill: component.activeDevicePhaseRole({ node_id: 'm5' }),
  decode: component.activeDevicePhaseRole({ node_id: 'm3' }),
}));
"""
    )

    assert result["rate"] == "8.59 GB/s"
    assert result["detail"] == "2.00 GiB · 250 ms · 128 tensors"
    assert result["prefill"] == "Prefill · full replica"
    assert result["decode"] == "Decode + API · full replica"
    assert "data-cluster-v2-phase-runtime" in _read(TEMPLATE)


def test_dark_tensor_controls_use_explicit_high_contrast_palette():
    template = _read(TEMPLATE)
    stylesheet = _read("omlx/admin/static/css/dashboard.css")

    assert (
        ":data-selected=\"planStrategy === option.key ? 'true' : 'false'\"" in template
    )
    assert "cluster-v2-tensor-segment" in template
    assert ':data-tensor-tone="index % 5"' in template
    assert "[data-cluster-v2-strategy-picker]" in stylesheet
    assert 'button[data-selected="true"]' in stylesheet
    assert "[data-cluster-v2-strategy-recommended]" in stylesheet
    assert "color: #f8fafc !important" in stylesheet
    assert ".cluster-v2-tensor-segment--2 { background: #52525b; }" in stylesheet
    assert '[data-theme="dark"] [data-cluster-v2-active] .text-neutral-600' in stylesheet
    assert "color: #d4d4d8 !important" in stylesheet

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


def test_legacy_view_toggle_keeps_v1_reachable_exactly_once():
    dashboard = _read(DASHBOARD)
    template = _read(TEMPLATE)

    assert dashboard.count('{% include "dashboard/_cluster.html" %}') == 1
    assert dashboard.count('{% include "dashboard/_cluster_v2.html" %}') == 1
    assert 'x-data="{ clusterLegacyView: false }"' in dashboard
    assert "clusterLegacyView = true" in template
    assert "clusterLegacyView = false" in dashboard
    assert "Advanced (legacy)" in template
    assert "data-cluster-legacy-view" in dashboard
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
    assert "denied the join request" in denied
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
    assert re.search(r"\^\(\\d\{1,3\}\(\?:\\\.\\d\{1,3\}\)\{3\}", javascript)
    assert "port >= 1 && port <= 65535" in javascript
    # A verified address flows straight into the joiner code panel (one click).
    assert "beginJoinAddr(`${ip}:${port}`" in javascript


# --- Plan step: per-node roles, usable budgets, actionable fit failures -------


def _run_wizard(body: str) -> dict:
    """Execute the shipped wizard component under Node with stubbed I/O."""

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to execute the wizard component")
    script = f"""
{_read(JAVASCRIPT)}
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
    assert "Switch all to Headless and retry" in template
    assert "parseFitFailure" in javascript
    assert r"at least (\d+) additional bytes" in javascript
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
    # Same segmented-control pattern as the role picker (neutral-900 active).
    picker = template.split("data-cluster-v2-strategy-picker", 1)[1].split(
        "data-cluster-v2-node-roles", 1
    )[0]
    assert "bg-neutral-900 text-white" in picker
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


def test_persistent_prompt_cache_is_visible_opt_in_and_replans():
    template = _read(TEMPLATE)
    javascript = _read(JAVASCRIPT)

    assert "data-cluster-v2-prompt-cache-ssd" in template
    assert "data-cluster-v2-prompt-cache-ssd-toggle" in template
    assert "data-cluster-v2-prompt-cache-ssd-cap" in template
    assert "data-cluster-v2-active-cache-mode" in template
    assert 'x-model="promptCacheSsd"' in template
    assert '@change="runPlan()"' in template
    assert "promptCacheSsd: false" in javascript
    assert "promptCacheSsdMaxGiB: 20" in javascript
    assert "prompt_cache_ssd: this.promptCacheSsd" in javascript
    assert "prompt_cache_ssd_max_bytes" in javascript
    assert "Writes run in the background" in template
    assert "512 MiB pending limit" in template

    result = _run_wizard(
        _WIZARD_TWO_MACS
        + """
const posted = [];
component.apiFetch = async (url, options) => {
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
  component.promptCacheSsd = true;
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

    assert result["defaultValue"] is False
    assert result["posted"] == [False, True]
    assert "persistent SSD snapshots" in result["enabledLabel"]
    assert "SSD snapshots off" in result["disabledLabel"]


def test_every_strategy_uses_server_autoconfigure_and_its_tp_choice():
    result = _run_wizard(
        _WIZARD_TWO_MACS
        + """
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

    proposals = [
        b for b in result["bodies"] if b["url"].endswith("/autoconfigure")
    ]
    deploys = [
        b
        for b in result["bodies"]
        if b["url"].endswith("/deployments") and b["marker"] is not None
    ]
    assert [item["strategy"] for item in proposals] == ["tensor", "auto", "pipeline"]
    assert all(item.get("tp") is None for item in proposals), (
        "the browser must not choose a TP degree"
    )
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

    assert result["keys"] == ["auto", "tensor", "pipeline", "disaggregated"]
    assert result["tensor"]["disabled"] is True
    assert result["tensor"]["disabledReason"] == "Tensor parallelism needs 2+ Macs"
    assert result["afterPick"] == "auto", "a disabled option cannot be picked"
    # No catalogue call ever fired on a one-Mac setup → no badge, no errors.
    assert result["recommended"] == ""


def test_phase_split_picker_posts_signed_role_ownership_and_renders_flow():
    result = _run_wizard(
        _WIZARD_TWO_MACS
        + """
const bodies = [];
component.apiFetch = async (url, options) => {
  const body = options?.body ? JSON.parse(options.body) : null;
  bodies.push({ url, body });
  if (url.endsWith('/autoconfigure')) {
    const prefill = Number(body.prefill_rank);
    const decode = Number(body.decode_rank);
    return {
      serving_mode: 'disaggregated',
      performance_probe: { ok: false, status: 'phase_probe_required' },
      plan: {
        serving_mode: 'disaggregated', prefill_rank: prefill, decode_rank: decode,
        assignments: [
          { rank: 0, node_id: 'node-a', start_layer: 0, end_layer: 48, layer_count: 48 },
          { rank: 1, node_id: 'node-b', start_layer: 0, end_layer: 48, layer_count: 48 },
        ],
        tensor_parallel_size: 1, pipeline_stages: 1,
        placement_signature: 'f'.repeat(16),
      },
      activation: {
        serving_mode: 'disaggregated', prefill_rank: prefill, decode_rank: decode,
        approved_placement: 'f'.repeat(16),
      },
    };
  }
  return {};
};
component.modelOptions = [{ model_path: '/models/qwen', id: 'qwen' }];
component.selectedModelPath = '/models/qwen';

(async () => {
  component.setPlanStrategy('disaggregated');
  await new Promise((resolve) => setTimeout(resolve, 0));
  const initial = bodies.filter((entry) => entry.url.endsWith('/autoconfigure')).at(-1).body;
  component.setPhasePrefillRank(0);
  await new Promise((resolve) => setTimeout(resolve, 0));
  process.stdout.write(JSON.stringify({
    initial,
    body: bodies.filter((entry) => entry.url.endsWith('/autoconfigure')).at(-1)?.body,
    isPhase: component.planIsDisaggregated(),
    totalLayers: component.planTotalLayers(),
    roles: component.planAssignments().map((row) => component.phasePlanRole(row)),
    recommended: component.phaseRecommendedPrefillRank(),
    benchmarkOkay: component.checks.benchmark?.ok,
  }));
})();
""",
    )

    assert result["body"]["strategy"] == "disaggregated"
    assert result["initial"]["prefill_rank"] == 1
    assert result["initial"]["decode_rank"] == 0
    assert result["body"]["prefill_rank"] == 0
    assert result["body"]["decode_rank"] == 1
    assert result["body"]["measure_performance"] is False
    assert result["isPhase"] is True
    assert result["totalLayers"] == 48
    assert result["roles"] == ["Prefill", "Decode"]
    assert result["recommended"] == 1
    assert result["benchmarkOkay"] is True

    template = _read(TEMPLATE)
    assert "data-cluster-v2-phase-role-picker" in template
    assert "data-cluster-v2-split-bar-disaggregated" in template
    assert "Full-replica Beta" in template
    assert "bg-neutral-900 text-white" in template


def test_phase_split_is_disabled_when_a_full_replica_exceeds_smallest_mac():
    result = _run_wizard(
        """
component.devicesPayload = {
  self: { node_id: 'node-a', friendly_name: 'M3 Ultra',
    caps: { ram_gb: 256, chip: 'M3 Ultra' }, addrs: [] },
  paired: [{ node_id: 'node-b', friendly_name: 'M5 Max', paired: true,
    caps: { ram_gb: 128, chip: 'M5 Max' }, addrs: [] }],
  discovered: [],
};
component.modelOptions = [{ model_path: '/models/huge', id: 'huge' }];
component.selectedModelPath = '/models/huge';
component.catalogueModels = [{
  model_path: '/models/huge', weight_bytes: 200 * (1024 ** 3),
  fits: true, supports_pipeline: true, supports_tensor_parallel: true,
}];
const option = component.strategyOptions().find((item) => item.key === 'disaggregated');
process.stdout.write(JSON.stringify({
  disabled: option.disabled,
  reason: option.disabledReason,
  summary: component.phaseFitSummary(),
}));
"""
    )

    assert result["disabled"] is True
    assert result["reason"] == "The complete model does not fit on both Macs."
    assert "200.0 GiB" in result["summary"]
    assert "smallest Mac" in result["summary"]


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
        _WIZARD_TWO_MACS
        + """
const proposal = JSON.parse(
  require('fs').readFileSync(
    %s,
    'utf8',
  ),
);
const bodies = [];
let probeOk = false;
component.apiFetch = async (url, options) => {
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


def test_split_bar_has_a_tensor_variant_and_width_transitions():
    template = _read(TEMPLATE)
    javascript = _read(JAVASCRIPT)

    assert "data-cluster-v2-split-bar-tensor" in template
    assert "planIsTensor()" in template
    assert "tensorShareLabel(assignment)" in template
    assert "tensorSharePercent(assignment)" in template
    assert "data-cluster-v2-tp-qualification" in template
    assert "Performance-qualified" in javascript
    assert "Experimental" in javascript
    assert "Equal" in javascript
    assert "calibration required" in javascript
    assert "parity or performance failed" in javascript
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
        _WIZARD_TWO_MACS
        + _WIZARD_TIMER_STUBS
        + _WIZARD_PARTIAL_MODEL
        + """
const calls = [];
component.apiFetch = async (url, options) => {
  calls.push(url);
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
