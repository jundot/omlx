# SPDX-License-Identifier: Apache-2.0
"""Behavioral contracts for the dashboard's one-click cluster path."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_JS = ROOT / "omlx" / "admin" / "static" / "js" / "dashboard.js"


def _method_source(name: str) -> str:
    """Lift one complete Alpine method out of the shipped dashboard source."""

    source = DASHBOARD_JS.read_text()
    match = re.search(rf"^[ \t]*(?:async\s+)?{re.escape(name)}\(", source, re.M)
    assert match is not None, f"dashboard.js has no {name}() method"
    start = match.start()
    cursor = source.index("(", start)
    depth = 0
    for index in range(cursor, len(source)):
        if source[index] == "(":
            depth += 1
        elif source[index] == ")":
            depth -= 1
            if depth == 0:
                cursor = index
                break
    body_start = source.index("{", cursor)
    depth = 0
    for index in range(body_start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"{name} has unbalanced braces in dashboard.js")


def _run_one_click(proposal: dict, *, link_status: dict | None = None) -> dict:
    """Execute the shipped orchestration with deterministic browser seams."""

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to execute the dashboard orchestration")
    methods = ",\n".join(
        _method_source(name)
        for name in (
            "loadClusterRuntime",
            "clusterActivationProgressFromRuntime",
            "startClusterActivationProgress",
            "stopClusterActivationProgress",
            "clusterWorkerPeers",
            "clusterClusterHostsPayload",
            "autoconfigureCluster",
            "prepareClusterLink",
            "prepareCudaSupernodesForActivation",
            "startCluster",
            "activateClusterProposal",
        )
    )
    script = f"""
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const calls = [];
let deploymentLoads = 0;
global.window = {{
  location: {{ href: '' }},
  localStorage: {{ setItem: () => {{}} }},
}};
global.setTimeout = () => 0;
global.clearTimeout = () => {{}};
global.setInterval = () => 1;
global.clearInterval = () => {{}};
global.fetch = async (url, options = {{}}) => {{
  calls.push({{ url, body: options.body ? JSON.parse(options.body) : null }});
  if (url.endsWith('/runtime')) {{
    return {{
      status: 200,
      ok: true,
      json: async () => ({{ jobs: [], warnings: [], launchers: [] }}),
    }};
  }}
  if (url.endsWith('/autoconfigure')) {{
    return {{ status: 200, ok: true, json: async () => input.proposal }};
  }}
  if (url.endsWith('/link-setup')) {{
    return {{
      status: 200,
      ok: true,
      json: async () => ({{
        state: 'rdma_ready',
        ready: true,
        setup_available: false,
      }}),
    }};
  }}
  if (url.endsWith('/deployments')) {{
    return {{
      status: 200,
      ok: true,
      json: async () => ({{
        ok: true,
        plan: {{ placement_signature: 'launched' }},
        plan_changes: {{ changed: false }},
      }}),
    }};
  }}
  throw new Error(`unexpected fetch ${{url}}`);
}};
const component = {{
  {methods},
  clusterNodePayloads() {{ return input.nodes; }},
  adoptClusterFabric(fabric) {{ this.adoptedFabric = fabric; }},
  invalidateClusterPlan() {{ this.clusterPlan = null; }},
  async loadClusterDeployments() {{ deploymentLoads += 1; }},
  async clusterResponseError(_response, fallback) {{ return fallback; }},
  clusterAutoconfigureLoading: false,
  clusterAutoconfigureError: '',
  clusterAutoconfigure: null,
  clusterActivationLoading: false,
  clusterLinkSetupLoading: false,
  clusterActivationProgress: '',
  _clusterActivationProgressTimer: null,
  clusterStatus: {{ runtime_jobs: {{ jobs: [] }} }},
  clusterActivationResult: null,
  clusterPlanChanges: null,
  clusterError: '',
  clusterPeerSsh: 'studio.local',
  clusterSelectedPeers: [{{ ssh: 'studio.local', name: 'Mac Studio' }}],
  clusterPlanNodes: input.nodes,
  clusterLocalIp: '10.0.0.1',
  clusterPeerIp: '10.0.0.2',
  clusterFabric: {{ hosts: [
    {{ host: '127.0.0.1', ips: ['10.0.0.1'], rdma: [null, 'rdma0'] }},
    {{ host: 'studio.local', ips: ['10.0.0.2'], rdma: ['rdma0', null] }},
  ] }},
  clusterFabricError: '',
  async loadClusterFabric() {{}},
  async loadClusterLinkStatus() {{
    return this.clusterLinkStatus || input.linkStatus;
  }},
  clusterPlanMode: 'model',
  clusterPlanModelPath: '/models/qwen',
  clusterPlanTensorParallelSize: 1,
  clusterExecutionProfile: 'balanced',
  clusterAutoconfigurePrefer: 'speed',
  clusterStrategy: 'auto',
  clusterAutoTune: true,
  clusterSamplingRankOnly: false,
  clusterAsyncOverlap: true,
  clusterCacheAffinity: true,
  clusterMaxKvSize: '',
  clusterRingConnectionsPerIp: 2,
  clusterLastGoodConfig: null,
}};
(async () => {{
  await component.startCluster();
  process.stdout.write(JSON.stringify({{
    calls,
    deploymentLoads,
    autoconfigureError: component.clusterAutoconfigureError,
    activationResult: component.clusterActivationResult,
    clusterError: component.clusterError,
  }}));
}})().catch(error => {{
  console.error(error);
  process.exit(1);
}});
"""
    result = subprocess.run(
        [node, "-e", script],
        input=json.dumps(
            {
                "proposal": proposal,
                "linkStatus": link_status
                or {
                    "state": "rdma_ready",
                    "ready": True,
                    "setup_available": False,
                },
                "nodes": [
                    {"node_id": "local", "capacity_bytes": 64 * 1024**3},
                    {"node_id": "studio", "capacity_bytes": 128 * 1024**3},
                ],
            }
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _run_dashboard_helpers(method_names: tuple[str, ...], body: str) -> dict:
    """Execute pure dashboard helpers against a small Alpine-shaped object."""

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to execute dashboard helpers")
    methods = ",\n".join(_method_source(name) for name in method_names)
    script = f"""
const component = {{
  {methods},
  models: [],
  clusterCatalogue: null,
  clusterModelSearch: '',
  clusterPlanModelPath: '',
  clusterDeployments: [],
  clusterPeerSsh: '',
}};
{body}
"""
    result = subprocess.run(
        [node, "-e", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_cluster_quick_status_distinguishes_runtime_mismatch_from_loading():
    result = _run_dashboard_helpers(
        ("clusterQuickStatus",),
        """
component.clusterSelectedModel = () => null;
component.clusterPrimaryDeployment = () => null;
component.clusterLiveJobs = () => [];
component.clusterAutoconfigureError = '';
component.clusterError = '';
component.clusterDeploymentsError = '';
component.clusterDeactivatingId = '';
component.clusterActivationLoading = false;
component.clusterAutoconfigureLoading = false;
component.clusterLinkSetupLoading = false;
component.clusterPeerSsh = 'studio.local';
component.clusterFabricLoading = false;
component.clusterLinkStatusLoading = false;
component.clusterCatalogueLoading = false;
const states = {};

component.clusterPeerProbeLoading = false;
component.clusterPeerProbe = null;
states.firstProbe = component.clusterQuickStatus();

component.clusterPeerProbeLoading = true;
component.clusterPeerProbe = { runtime_compatible: true };
states.reprobe = component.clusterQuickStatus();

component.clusterPeerProbeLoading = false;
component.clusterPeerProbe = {
  runtime_compatible: false,
  runtime_mismatches: [
    'omlx local=0.6.0 remote=0.5.4',
    'mlx local=0.32.0 remote=0.31.0',
  ],
};
states.mismatch = component.clusterQuickStatus();

component.clusterPeerProbe = {
  runtime_compatible: true,
  runtime_warnings: ['python minor differs: local=3.11 remote=3.12'],
};
states.warning = component.clusterQuickStatus();

process.stdout.write(JSON.stringify(states));
""",
    )

    assert result["firstProbe"]["key"] == "checking"
    assert result["firstProbe"]["busy"] is True
    assert result["reprobe"]["key"] == "checking"
    assert result["reprobe"]["busy"] is True
    assert result["mismatch"] == {
        "key": "runtime-mismatch",
        "label": "Worker runtime mismatch",
        "detail": (
            "omlx local=0.6.0 remote=0.5.4 · "
            "mlx local=0.32.0 remote=0.31.0"
        ),
        "tone": "red",
        "busy": False,
    }
    assert result["warning"]["key"] == "model"
    assert result["warning"]["tone"] == "amber"


def test_manual_memory_allowance_survives_automatic_budget_refresh():
    result = _run_dashboard_helpers(
        (
            "saveClusterMemoryAllowances",
            "clusterManualMemoryAllowanceGiB",
            "clusterSetMemoryAllowance",
            "measureClusterBudgets",
        ),
        """
const writes = [];
global.window = {
  localStorage: {
    setItem: (key, value) => writes.push([key, JSON.parse(value)]),
    removeItem: key => writes.push([key, null]),
  },
};
global.fetch = async () => ({
  ok: true,
  json: async () => ({ nodes: [{
    node_id: 'MacBook-Pro',
    capacity_bytes: 108 * (1024 ** 3),
    reserve_bytes: 66 * (1024 ** 3),
    summary: '42 GiB automatic',
  }] }),
});
component.clusterMemoryAllowancesGiB = {};
component.clusterMemoryLimitsManual = false;
component.clusterWeightTargetsGiB = {};
component.clusterPlanNodes = [{
  node_id: 'MacBook-Pro',
  capacity_gib: 108,
  reserve_gib: 66,
  role: 'workstation',
}];
component.clusterBudgetsLoading = false;
component.clusterBudgetsError = '';
component.clusterBudgetHostsPayload = () => [{ ssh: '127.0.0.1' }];
component.clusterErrorMessage = detail => String(detail || 'Could not measure the Macs.');
component.invalidateClusterPlan = () => {};
component.clusterSetMemoryAllowance({
  nodeId: 'MacBook-Pro',
  budget: component.clusterPlanNodes[0],
  capacityGiB: 108,
  minGiB: 4,
}, 80);
(async () => {
  await component.measureClusterBudgets();
  process.stdout.write(JSON.stringify({
    reserve: component.clusterPlanNodes[0].reserve_gib,
    automaticReserve: component.clusterPlanNodes[0].automatic_reserve_gib,
    allowed: component.clusterMemoryAllowancesGiB['MacBook-Pro'],
    persisted: writes.at(-1),
  }));
})().catch(error => {
  console.error(error);
  process.exit(1);
});
""",
    )

    assert result == {
        "reserve": 28,
        "automaticReserve": 66,
        "allowed": 80,
        "persisted": [
            "omlx.cluster.memoryAllowances",
            {"version": 1, "allowances_gib": {"MacBook-Pro": 80}},
        ],
    }


def test_manual_memory_allowance_is_marked_in_every_planner_payload():
    result = _run_dashboard_helpers(
        ("clusterManualMemoryAllowanceGiB", "clusterNodePayloads"),
        """
component.clusterMemoryAllowancesGiB = { 'MacBook-Pro': 90 };
component.clusterPlanNodes = [
  { node_id: 'MacBook-Pro', capacity_gib: 108, reserve_gib: 18, role: 'workstation' },
  { node_id: 'Studio', capacity_gib: 256, reserve_gib: 16, role: 'headless' },
];
component.clusterWeightTargetsGiB = {};
component.clusterSplitGiB = null;
process.stdout.write(JSON.stringify(component.clusterNodePayloads()));
""",
    )

    assert result[0]["manual_memory_limit"] is True
    assert result[1]["manual_memory_limit"] is False
    assert result[0]["reserve_bytes"] == 18 * 1024**3


def test_memory_measurement_hosts_do_not_require_collective_addresses():
    result = _run_dashboard_helpers(
        ("clusterBudgetHostsPayload",),
        """
component.clusterPlanNodes = [
  { node_id: 'local' },
  { node_id: 'studio' },
];
component.clusterWorkerPeers = () => [{ ssh: 'studio.local' }];
process.stdout.write(JSON.stringify(component.clusterBudgetHostsPayload()));
""",
    )

    assert result == [
        {"node_id": "local", "ssh": "127.0.0.1"},
        {"node_id": "studio", "ssh": "studio.local"},
    ]


def test_unmeasured_peer_memory_is_unknown_not_a_56_gib_placeholder():
    result = _run_dashboard_helpers(
        ("clusterNodeId", "syncClusterNodesFromPeers"),
        """
const gib = 1024 ** 3;
component.clusterStatus = { node: {
  hostname: 'local',
  admission_ceiling_bytes: 498 * gib,
} };
component.clusterPlanNodes = [
  { key: 1, node_id: 'this-mac', capacity_gib: 128, reserve_gib: 8 },
  { key: 2, node_id: 'peer-mac', capacity_gib: 256, reserve_gib: 8 },
];
component.clusterDeployments = [];
component.clusterPeerProbes = {};
component.clusterWorkerPeers = () => [{ ssh: 'studio.local', name: 'studio' }];
component.clusterFriendlyMacName = value => value;
component.normalizeClusterTensorParallelSize = () => {};
component.invalidateClusterPlan = () => {};
component._clusterNodeKey = 2;
component.syncClusterNodesFromPeers();
process.stdout.write(JSON.stringify(component.clusterPlanNodes));
""",
    )

    assert result[0]["capacity_bytes"] == 498 * 1024**3
    assert result[1]["node_id"] == "studio"
    assert result[1]["capacity_gib"] == 0
    assert result[1]["capacity_bytes"] == 0
    assert result[1]["reserve_gib"] == 0


def test_local_role_defaults_to_workstation_and_preserves_headless_selection():
    result = _run_dashboard_helpers(
        ("clusterNodeId", "syncClusterNodesFromPeers"),
        """
component.clusterStatus = { node: {
  hostname: 'local',
  admission_ceiling_bytes: 1000,
} };
component.clusterPlanNodes = [
  { key: 1, node_id: 'local', ssh: '127.0.0.1' },
];
component.clusterDeployments = [];
component.clusterPeerProbes = {};
component.clusterWorkerPeers = () => [];
component.normalizeClusterTensorParallelSize = () => {};
component.invalidateClusterPlan = () => {};
component._clusterNodeKey = 1;
component.syncClusterNodesFromPeers();
const defaultRole = component.clusterPlanNodes[0].role;
component.clusterPlanNodes[0].role = 'headless';
component.syncClusterNodesFromPeers();
process.stdout.write(JSON.stringify({
  defaultRole,
  selectedRole: component.clusterPlanNodes[0].role,
}));
""",
    )

    assert result == {
        "defaultRole": "workstation",
        "selectedRole": "headless",
    }


def test_same_named_peer_does_not_inherit_another_macs_capacity():
    result = _run_dashboard_helpers(
        ("clusterNodeId", "syncClusterNodesFromPeers"),
        """
const gib = 1024 ** 3;
component.clusterStatus = { node: {
  hostname: 'local',
  admission_ceiling_bytes: 120 * gib,
} };
component.clusterPlanNodes = [
  { key: 1, node_id: 'local', ssh: '127.0.0.1', capacity_bytes: 120 * gib },
  { key: 2, node_id: 'Mac Studio', ssh: 'old.local',
    capacity_gib: 498, capacity_bytes: 498 * gib, reserve_gib: 50 },
];
component.clusterDeployments = [];
component.clusterPeerProbes = {};
component.clusterWorkerPeers = () => [{ ssh: 'new.local', name: 'Mac Studio' }];
component.clusterFriendlyMacName = value => value;
component.normalizeClusterTensorParallelSize = () => {};
component.invalidateClusterPlan = () => {};
component._clusterNodeKey = 2;
component.syncClusterNodesFromPeers();
process.stdout.write(JSON.stringify(component.clusterPlanNodes[1]));
""",
    )

    assert result["ssh"] == "new.local"
    assert result["capacity_gib"] == 0
    assert result["capacity_bytes"] == 0
    assert result["reserve_gib"] == 0


def test_default_macos_names_use_valid_network_node_ids():
    result = _run_dashboard_helpers(
        ("clusterNodeId", "syncClusterNodesFromPeers"),
        """
component.clusterStatus = { node: {
  hostname: 'Coordinator’s Mac Studio',
  admission_ceiling_bytes: 1000,
} };
component.clusterDeployments = [];
component.clusterPlanNodes = [
  { key: 1, node_id: 'this-mac' },
  { key: 2, node_id: 'peer-mac' },
];
component.clusterPeerProbes = {
  'Deepaks-Mac-mini.local': {
    status: { node: { hostname: 'Deepak’s Mac mini' } },
  },
  'Sarahs-MacBook-Pro.local': {
    status: { node: { hostname: 'Sarah’s MacBook Pro' } },
  },
  'Alexs-Mac-Studio.local': {
    status: { node: { hostname: 'Alex’s Mac Studio' } },
  },
};
component.clusterWorkerPeers = () => [
  { ssh: 'Deepaks-Mac-mini.local', name: 'Deepak’s Mac mini' },
  { ssh: 'Sarahs-MacBook-Pro.local', name: 'Sarah’s MacBook Pro' },
  { ssh: 'Alexs-Mac-Studio.local', name: 'Alex’s Mac Studio' },
];
component.normalizeClusterTensorParallelSize = () => {};
component.invalidateClusterPlan = () => {};
component._clusterNodeKey = 2;
component.syncClusterNodesFromPeers();
process.stdout.write(JSON.stringify(
  component.clusterPlanNodes.map(node => node.node_id)
));
""",
    )

    assert result == [
        "this-mac",
        "Deepaks-Mac-mini.local",
        "Sarahs-MacBook-Pro.local",
        "Alexs-Mac-Studio.local",
    ]
    assert all(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", node_id)
        for node_id in result
    )


def test_sync_preserves_valid_deployment_node_ids():
    result = _run_dashboard_helpers(
        ("clusterNodeId", "syncClusterNodesFromPeers"),
        """
component.clusterStatus = { node: {
  hostname: 'Coordinator’s Mac Studio',
  admission_ceiling_bytes: 1000,
} };
component.clusterDeployments = [{ hosts: [
  { node_id: 'coordinator', ssh: '127.0.0.1' },
  { node_id: 'mac-studio-01', ssh: 'Alexs-Mac-Studio.local' },
] }];
component.clusterPlanNodes = [
  { key: 1, node_id: 'this-mac' },
  { key: 2, node_id: 'peer-mac' },
];
component.clusterPeerProbes = {};
component.clusterWorkerPeers = () => [{
  ssh: 'Alexs-Mac-Studio.local',
  name: 'Alex’s Mac Studio',
}];
component.normalizeClusterTensorParallelSize = () => {};
component.invalidateClusterPlan = () => {};
component._clusterNodeKey = 2;
component.syncClusterNodesFromPeers();
process.stdout.write(JSON.stringify(
  component.clusterPlanNodes.map(node => node.node_id)
));
""",
    )

    assert result == ["coordinator", "mac-studio-01"]


def test_unknown_middle_peer_does_not_shift_memory_to_another_card():
    result = _run_dashboard_helpers(
        ("clusterMemoryNodes", "clusterMemoryAllowanceNodes"),
        """
component.clusterPlanNodes = [
  { node_id: 'local', capacity_gib: 120, reserve_gib: 10 },
  { node_id: 'unknown', capacity_gib: 0, reserve_gib: 0 },
  { node_id: 'large', capacity_gib: 498, reserve_gib: 50 },
];
component.clusterQuickNodes = () => [
  { name: 'Local' },
  { name: 'Unknown peer' },
  { name: 'Large peer' },
];
process.stdout.write(JSON.stringify({
  memory: component.clusterMemoryNodes().map(node => ({
    name: node.name,
    capacity: node.capacityGiB,
  })),
  allowances: component.clusterMemoryAllowanceNodes().map(node => node.name),
}));
""",
    )

    assert result == {
        "memory": [
            {"name": "Local", "capacity": 120},
            {"name": "Unknown peer", "capacity": 0},
            {"name": "Large peer", "capacity": 498},
        ],
        "allowances": ["Local", "Large peer"],
    }


def test_peer_retry_keeps_the_existing_error_mounted_until_ssh_answers():
    result = _run_dashboard_helpers(
        ("probeClusterPeer", "_escalateClusterProbeBackoff"),
        """
let errorDuringFetch = null;
global.window = { location: { href: '' } };
global.fetch = async () => {
  errorDuringFetch = component.clusterError;
  return { status: 503, ok: false };
};
component.clusterPeerSsh = 'studio.local';
component.clusterPeerProbeLoading = false;
component.clusterPeerProbe = null;
component.clusterLocalIp = '';
component.clusterError = 'The other Mac rejected the login';
component.clusterConnectionError = 'The other Mac rejected the login';
component.clusterResponseError = async () => 'The other Mac rejected the login';
(async () => {
  await component.probeClusterPeer();
  process.stdout.write(JSON.stringify({
    errorDuringFetch,
    finalError: component.clusterError,
    finalConnectionError: component.clusterConnectionError,
  }));
})().catch(error => {
  console.error(error);
  process.exit(1);
});
""",
    )

    assert result == {
        "errorDuringFetch": "The other Mac rejected the login",
        "finalError": "The other Mac rejected the login",
        "finalConnectionError": "The other Mac rejected the login",
    }


def test_successful_peer_retry_clears_connection_error_and_guidance():
    result = _run_dashboard_helpers(
        (
            "clusterNodeId",
            "probeClusterPeer",
            "clusterDisplayedError",
            "resetClusterProbeBackoff",
        ),
        """
global.window = { location: { href: '' } };
global.fetch = async () => ({
  status: 200,
  ok: true,
  json: async () => ({
    status: { node: { hostname: 'Alex’s Mac Studio' } },
  }),
});
component.clusterPeerSsh = 'studio.local';
component.clusterPeerProbeLoading = false;
component.clusterPeerProbe = null;
component.clusterPeerProbes = {};
component.clusterDeployments = [];
component.clusterPlanNodes = [
  { node_id: 'local' },
  { node_id: 'peer-mac' },
];
component.clusterLocalIp = '';
component.clusterConnectionError = 'The other Mac rejected the oMLX key';
component.clusterError = 'The other Mac rejected the oMLX key';
component.clusterGuidance = { title: 'Pair the Macs' };
component.dismissClusterGuidance = () => { component.clusterGuidance = null; };
component.saveClusterKnownNodes = () => {};
component.$nextTick = async () => {};
component.invalidateClusterPlan = () => {};
component.loadClusterFabric = async () => {};
(async () => {
  await component.probeClusterPeer();
  process.stdout.write(JSON.stringify({
    displayed: component.clusterDisplayedError(),
    genericError: component.clusterError,
    connectionError: component.clusterConnectionError,
    guidance: component.clusterGuidance,
    probed: Boolean(component.clusterPeerProbe),
    nodeId: component.clusterPlanNodes[1].node_id,
  }));
})().catch(error => {
  console.error(error);
  process.exit(1);
});
""",
    )

    assert result == {
        "displayed": "",
        "genericError": "",
        "connectionError": "",
        "guidance": None,
        "probed": True,
        "nodeId": "studio.local",
    }


def test_status_polling_preserves_peer_connection_guidance():
    result = _run_dashboard_helpers(
        ("loadClusterStatus",),
        """
global.window = { location: { href: '' } };
global.fetch = async () => ({
  status: 200,
  ok: true,
  json: async () => ({ node: { hostname: 'local' } }),
});
component.clusterLoading = false;
component.clusterError = 'The other Mac rejected the oMLX key';
component.clusterConnectionError = 'The other Mac rejected the oMLX key';
component.clusterGuidance = { title: 'Pair the Macs' };
component.clusterRouteTo = '';
component._clusterDefaultsApplied = true;
component.loadRememberedClusterConfig = () => {};
component.dismissClusterGuidance = () => { component.clusterGuidance = null; };
(async () => {
  await component.loadClusterStatus();
  process.stdout.write(JSON.stringify({
    genericError: component.clusterError,
    connectionError: component.clusterConnectionError,
    guidance: component.clusterGuidance,
    statusHost: component.clusterStatus.node.hostname,
  }));
})().catch(error => {
  console.error(error);
  process.exit(1);
});
""",
    )

    assert result == {
        "genericError": "",
        "connectionError": "The other Mac rejected the oMLX key",
        "guidance": {"title": "Pair the Macs"},
        "statusHost": "local",
    }


def test_background_replanning_does_not_unmount_a_peer_connection_error():
    result = _run_dashboard_helpers(
        ("clusterDisplayedError", "invalidateClusterPlan"),
        """
component.clusterConnectionError = 'The other Mac rejected the oMLX key';
component.clusterError = 'The other Mac rejected the oMLX key';
component.clusterPlan = { ready: true };
component.clusterPlanError = 'old plan';
component.clusterAutoconfigureError = 'old setup';
component.clusterActivationResult = { ok: false };
component.clusterPlanChanges = { changed: true };
component._clusterPlanSignature = 'old';
component._clusterPlanRevision = 1;
const before = component.clusterDisplayedError();
component.invalidateClusterPlan();
process.stdout.write(JSON.stringify({
  before,
  after: component.clusterDisplayedError(),
  genericError: component.clusterError,
  connectionError: component.clusterConnectionError,
}));
""",
    )

    assert result == {
        "before": "The other Mac rejected the oMLX key",
        "after": "The other Mac rejected the oMLX key",
        "genericError": "",
        "connectionError": "The other Mac rejected the oMLX key",
    }


def test_changing_the_peer_clears_its_connection_error_and_guidance():
    result = _run_dashboard_helpers(
        ("clusterDisplayedError", "invalidateClusterPeer"),
        """
component.clusterPeerProbe = { ok: false };
component.clusterConnectionError = 'The old Mac rejected the oMLX key';
component.clusterError = 'The old Mac rejected the oMLX key';
component.clusterGuidance = { title: 'Pair the old Mac' };
component.dismissClusterGuidance = () => { component.clusterGuidance = null; };
component.invalidateClusterPlan = () => {};
component.invalidateClusterPeer(false);
process.stdout.write(JSON.stringify({
  displayed: component.clusterDisplayedError(),
  genericError: component.clusterError,
  connectionError: component.clusterConnectionError,
  guidance: component.clusterGuidance,
}));
""",
    )

    assert result == {
        "displayed": "",
        "genericError": "",
        "connectionError": "",
        "guidance": None,
    }


def test_validation_errors_are_rendered_as_messages_not_object_strings():
    result = _run_dashboard_helpers(
        ("clusterErrorMessage",),
        """
process.stdout.write(JSON.stringify(component.clusterErrorMessage([
  { loc: ['body', 'hosts', 0, 'ips'], msg: 'List should have at least 1 item' },
  { loc: ['body', 'hosts', 1, 'ips'], msg: 'List should have at least 1 item' },
])));
""",
    )

    assert result == (
        "hosts.0.ips: List should have at least 1 item, "
        "hosts.1.ips: List should have at least 1 item"
    )


def test_cached_peer_measurement_cannot_make_the_catalogue_ready():
    result = _run_dashboard_helpers(
        ("clusterCatalogueInputsReady",),
        """
component.clusterWorkerPeers = () => [{ ssh: 'studio.local' }];
component.clusterPeerProbeLoading = false;
component.clusterBudgetsLoading = false;
component.clusterFabricLoading = false;
component.clusterPlanNodes = [
  { node_id: 'local', capacity_bytes: 120 },
  { node_id: 'studio', capacity_bytes: 498 },
];
component.clusterPeerProbes = {
  'studio.local': { cached: true, status: { node: { hostname: 'studio' } } },
};
const cached = component.clusterCatalogueInputsReady();
component.clusterPeerProbes['studio.local'].cached = false;
const live = component.clusterCatalogueInputsReady();
process.stdout.write(JSON.stringify({ cached, live }));
""",
    )

    assert result == {"cached": False, "live": True}


def test_manual_link_addresses_override_discovery_and_drop_the_rdma_matrix():
    result = _run_dashboard_helpers(
        ("clusterClusterHostsPayload",),
        """
component.clusterPlanNodes = [
  { node_id: 'local' },
  { node_id: 'studio' },
];
component.clusterWorkerPeers = () => [{ ssh: 'studio.local' }];
component.clusterLocalIp = '10.77.0.1';
component.clusterPeerIp = '10.77.0.2';
component.clusterIpsOverridden = true;
component.clusterFabric = { hosts: [
  { host: '127.0.0.1', ips: ['10.0.1.1'], rdma: [null, 'rdma_en4'] },
  { host: 'studio.local', ips: ['10.0.1.2'], rdma: ['rdma_en5', null] },
] };
process.stdout.write(JSON.stringify({
  hosts: component.clusterClusterHostsPayload(),
}));
""",
    )

    assert [host["ips"] for host in result["hosts"]] == [
        ["10.77.0.1"],
        ["10.77.0.2"],
    ]
    assert [host["rdma"] for host in result["hosts"]] == [[], []]


def test_manual_link_addresses_disable_transport_rediscovery():
    result = _run_dashboard_helpers(
        ("autoconfigureCluster",),
        """
let posted = null;
global.window = {
  location: { href: '' },
  localStorage: { setItem: () => {} },
};
global.fetch = async (_url, options) => {
  posted = JSON.parse(options.body);
  return {
    status: 200,
    ok: true,
    json: async () => ({
      ready_to_activate: false,
      fabric: null,
      tensor_parallel_size: 1,
      plan: null,
    }),
  };
};
Object.assign(component, {
  _clusterPlanRevision: 0,
  clusterAutoconfigureLoading: false,
  clusterAutoconfigureError: '',
  clusterAutoconfigure: null,
  clusterIpsOverridden: true,
  clusterExecutionProfile: 'balanced',
  clusterAutoconfigurePrefer: 'speed',
  clusterStrategy: 'auto',
  clusterAutoTune: false,
  clusterSamplingRankOnly: true,
  clusterAsyncOverlap: true,
  clusterCacheAffinity: true,
  clusterMaxKvSize: '',
  clusterTargetContextTokens: 8192,
  clusterRingConnectionsPerIp: 2,
  clusterPlanMode: 'model',
  clusterPlanModelPath: '/models/qwen',
  clusterPlanModelSource: '127.0.0.1',
  clusterNodePayloads: () => [
    { node_id: 'local', capacity_bytes: 64 * (1024 ** 3) },
    { node_id: 'studio', capacity_bytes: 64 * (1024 ** 3) },
  ],
  clusterClusterHostsPayload: () => [
    { node_id: 'local', ssh: '127.0.0.1', ips: ['10.77.0.1'], rdma: [] },
    { node_id: 'studio', ssh: 'studio.local', ips: ['10.77.0.2'], rdma: [] },
  ],
  adoptClusterFabric: () => {},
  invalidateClusterPlan: () => {},
});
(async () => {
  await component.autoconfigureCluster();
  process.stdout.write(JSON.stringify(posted));
})().catch(error => {
  console.error(error);
  process.exit(1);
});
""",
    )

    assert result["detect_transports"] is False
    assert [host["ips"] for host in result["hosts"]] == [
        ["10.77.0.1"],
        ["10.77.0.2"],
    ]


def test_stale_catalogue_result_is_discarded_and_retried_with_measured_budgets():
    result = _run_dashboard_helpers(
        (
            "clusterCatalogueModels",
            "clusterPythonExecutableForSsh",
            "clusterCatalogueInputsReady",
            "clusterCatalogueRequestKey",
            "loadClusterCatalogue",
        ),
        """
let releaseFirst = null;
const requests = [];
global.window = { location: { href: '' } };
global.fetch = async (_url, options) => {
  requests.push(JSON.parse(options.body));
  if (requests.length === 1) {
    return await new Promise(resolve => {
      releaseFirst = () => resolve({
        status: 200,
        ok: true,
        json: async () => ({ models: [{ model_path: '/deepseek', fits: false }] }),
      });
    });
  }
  return {
    status: 200,
    ok: true,
    json: async () => ({ models: [{ model_path: '/deepseek', fits: true }] }),
  };
};
component.clusterWorkerPeers = () => [{ ssh: 'studio.local' }];
component.clusterPeerProbes = {
  'studio.local': { status: { node: { hostname: 'studio' } } },
};
component.clusterPeerProbeLoading = false;
component.clusterBudgetsLoading = false;
component.clusterFabricLoading = false;
component.clusterPlanNodes = [
  { node_id: 'local', capacity_bytes: 120 },
  { node_id: 'studio', capacity_bytes: 64 },
];
component.clusterNodePayloads = () => component.clusterPlanNodes.map(node => ({
  node_id: node.node_id,
  capacity_bytes: node.capacity_bytes,
}));
component.clusterAllModels = () => [{
  id: 'deepseek',
  model_path: '/deepseek',
  model_source: '127.0.0.1',
}];
component.clusterModelDisplayName = model => model.id;
component.clusterModelInventory = { models: [] };
component.loadClusterModelInventory = async () => {};
component.clusterCatalogueLoading = false;
component.clusterCatalogueError = '';
component.clusterExecutionProfile = 'balanced';
component.clusterSelectedModel = () => null;
const pending = component.loadClusterCatalogue();
(async () => {
  while (!releaseFirst) await new Promise(resolve => setImmediate(resolve));
  component.clusterPlanNodes[1].capacity_bytes = 220;
  releaseFirst();
  await pending;
  process.stdout.write(JSON.stringify({
    requestCapacities: requests.map(request =>
      request.nodes.map(node => node.capacity_bytes)
    ),
    fit: component.clusterCatalogue.models[0].fits,
  }));
})().catch(error => {
  console.error(error);
  process.exit(1);
});
""",
    )

    assert result == {
        "requestCapacities": [[120, 64], [120, 220]],
        "fit": True,
    }


def test_catalogue_waits_until_the_peer_probe_finishes():
    result = _run_dashboard_helpers(
        (
            "clusterCatalogueModels",
            "clusterCatalogueInputsReady",
            "clusterCatalogueRequestKey",
            "loadClusterCatalogue",
        ),
        """
let calls = 0;
global.fetch = async () => { calls += 1; throw new Error('must not fetch'); };
component.clusterWorkerPeers = () => [{ ssh: 'studio.local' }];
component.clusterPeerProbes = {};
component.clusterPeerProbeLoading = true;
component.clusterBudgetsLoading = false;
component.clusterFabricLoading = false;
component.clusterPlanNodes = [
  { node_id: 'local', capacity_bytes: 120 },
  { node_id: 'studio', capacity_bytes: 64 },
];
component.clusterNodePayloads = () => component.clusterPlanNodes;
component.clusterAllModels = () => [];
component.clusterCatalogueLoading = false;
component.clusterCatalogue = { models: [{ fits: false }] };
component.clusterCatalogueError = 'old refusal';
(async () => {
  await component.loadClusterCatalogue();
  process.stdout.write(JSON.stringify({
    calls,
    catalogue: component.clusterCatalogue,
    error: component.clusterCatalogueError,
  }));
})().catch(error => {
  console.error(error);
  process.exit(1);
});
""",
    )

    assert result == {"calls": 0, "catalogue": None, "error": ""}


def test_catalogue_does_not_treat_a_display_placeholder_as_measured_memory():
    result = _run_dashboard_helpers(
        (
            "clusterCatalogueModels",
            "clusterCatalogueInputsReady",
            "clusterCatalogueRequestKey",
            "loadClusterCatalogue",
        ),
        """
let calls = 0;
global.fetch = async () => { calls += 1; throw new Error('must not fetch'); };
component.clusterWorkerPeers = () => [{ ssh: 'studio.local' }];
component.clusterPeerProbes = {
  'studio.local': { status: { node: { hostname: 'studio' } } },
};
component.clusterPeerProbeLoading = false;
component.clusterBudgetsLoading = false;
component.clusterFabricLoading = false;
component.clusterPlanNodes = [
  { node_id: 'local', capacity_bytes: 120, capacity_gib: 120 },
  { node_id: 'studio', capacity_gib: 64 },
];
component.clusterNodePayloads = () => component.clusterPlanNodes;
component.clusterAllModels = () => [];
component.clusterCatalogueLoading = false;
component.clusterCatalogue = { models: [{ fits: false }] };
(async () => {
  await component.loadClusterCatalogue();
  process.stdout.write(JSON.stringify({ calls, catalogue: component.clusterCatalogue }));
})().catch(error => {
  console.error(error);
  process.exit(1);
});
""",
    )

    assert result == {"calls": 0, "catalogue": None}


def test_model_context_is_clamped_to_what_the_cluster_can_serve():
    result = _run_dashboard_helpers(
        (
            "clusterCatalogueFit",
            "clusterTokens",
            "clusterModelContextLabel",
            "clusterModelTargetContext",
        ),
        """
component.clusterCatalogue = { models: [{
  model_path: '/glm',
  fits: true,
  max_context_tokens: 131072,
}] };
const model = { model_path: '/glm', model_context_length: 1048576 };
process.stdout.write(JSON.stringify({
  target: component.clusterModelTargetContext(model, 1048576),
  label: component.clusterModelContextLabel(model),
}));
""",
    )

    assert result == {"target": 131072, "label": "Up to 128k context"}


def test_context_picker_offers_256k_and_shows_its_real_binary_label():
    result = _run_dashboard_helpers(
        (
            "clusterContextMaximumTokens",
            "clusterContextOptions",
            "clusterTokens",
        ),
        """
component.clusterSelectedModel = () => ({
  model_path: '/minimax',
  model_context_length: 1048576,
});
component.clusterCatalogueFit = () => ({
  fits: true,
  declared_context_tokens: 1048576,
  max_context_tokens: 524288,
});
component.clusterTargetContextTokens = 262144;
process.stdout.write(JSON.stringify({
  options: component.clusterContextOptions(),
  label: component.clusterTokens(262144),
}));
""",
    )

    assert 262144 in result["options"]
    assert result["label"] == "256k"


def test_automatic_context_tracks_the_highest_safe_model_and_memory_limit():
    result = _run_dashboard_helpers(
        ("clusterContextMaximumTokens", "clusterSetAutomaticContext"),
        """
let saved = null;
global.window = {
  localStorage: {
    setItem: (_key, value) => { saved = JSON.parse(value); },
  },
};
component.clusterTargetContextTokens = 131072;
component.clusterContextMode = 'manual';
component.clusterMaxKvSize = '4096';
component.clusterPlanModelSource = 'studio';
component.clusterSelectedModel = () => ({
  model_path: '/minimax',
  model_source: 'studio',
  model_context_length: 524288,
});
component.clusterCatalogueFit = () => ({
  fits: true,
  declared_context_tokens: 524288,
  max_context_tokens: 262144,
});
component.invalidateClusterPlan = () => {};
component.previewClusterWeightBalance = async () => {};
(async () => {
  await component.clusterSetAutomaticContext();
  process.stdout.write(JSON.stringify({
    target: component.clusterTargetContextTokens,
    mode: component.clusterContextMode,
    saved,
  }));
})().catch(error => {
  console.error(error);
  process.exit(1);
});
""",
    )

    assert result["target"] == 262144
    assert result["mode"] == "auto"
    assert result["saved"]["context_mode"] == "auto"


def test_remote_model_selection_carries_its_linux_python_path():
    result = _run_dashboard_helpers(
        ("clusterPythonExecutableForSsh", "selectClusterModel"),
        """
let saved = null;
global.window = {
  localStorage: {
    setItem: (_key, value) => { saved = JSON.parse(value); },
  },
};
Object.assign(component, {
  clusterPlanNodes: [{
    ssh: 'cuda-worker-1',
    python_executable: '/opt/omlx/bin/python',
  }],
  clusterPeerProbes: {},
  clusterCatalogueFit: () => null,
  clusterTensorParallelOptions: () => [1],
  clusterModelTargetContext: (_model, value) => value,
  clusterShowModelPicker: true,
  clusterWeightTargetsGiB: {},
  invalidateClusterPlan: () => {},
  previewClusterWeightBalance: () => {},
});
component.selectClusterModel({
  model_path: '/models/glm',
  model_source: 'cuda-worker-1',
  model_context_length: 131072,
});
process.stdout.write(JSON.stringify({
  sourcePython: component.clusterPlanModelSourcePython,
  savedPython: saved.model_source_python,
}));
""",
    )

    assert result == {
        "sourcePython": "/opt/omlx/bin/python",
        "savedPython": "/opt/omlx/bin/python",
    }


def test_automatic_context_does_not_flash_an_unverified_native_ceiling():
    result = _run_dashboard_helpers(
        ("clusterContextMaximumTokens",),
        """
component.clusterTargetContextTokens = 262144;
component.clusterSelectedModel = () => ({
  model_path: '/large',
  model_context_length: 1048576,
});
component.clusterCatalogueFit = () => null;
process.stdout.write(JSON.stringify({
  pending: component.clusterContextMaximumTokens(),
}));
""",
    )

    assert result["pending"] == 262144


def test_context_choice_replans_and_is_remembered_with_the_model():
    result = _run_dashboard_helpers(
        ("clusterSetTargetContext", "invalidateClusterPlan"),
        """
let saved = null;
global.window = {
  localStorage: {
    setItem: (_key, value) => { saved = JSON.parse(value); },
  },
};
component.clusterTargetContextTokens = 8192;
component.clusterMaxKvSize = '4096';
component.clusterPlanModelSource = 'studio';
component.clusterAutoconfigureError =
  'model weights and KV cache for 131072 tokens do not fit';
component.clusterError = 'old activation failed';
component.clusterSelectedModel = () => ({
  model_path: '/minimax',
  model_source: 'studio',
});
component.clusterModelTargetContext = (_model, requested) => requested;
component.previewClusterWeightBalance = async () => {};
(async () => {
  await component.clusterSetTargetContext(262144);
  process.stdout.write(JSON.stringify({
    target: component.clusterTargetContextTokens,
    legacyOverride: component.clusterMaxKvSize,
    autoconfigureError: component.clusterAutoconfigureError,
    clusterError: component.clusterError,
    saved,
  }));
})().catch(error => {
  console.error(error);
  process.exit(1);
});
""",
    )

    assert result["target"] == 262144
    assert result["legacyOverride"] == ""
    assert result["autoconfigureError"] == ""
    assert result["clusterError"] == ""
    assert result["saved"]["target_context_tokens"] == 262144
    assert result["saved"]["model_path"] == "/minimax"
    assert result["saved"]["context_mode"] == "manual"


def test_late_autoconfigure_failure_cannot_restore_an_invalidated_error():
    result = _run_dashboard_helpers(
        ("autoconfigureCluster",),
        """
let completeRequest = null;
global.window = {
  location: { href: '' },
  localStorage: { setItem: () => {} },
};
global.fetch = async () => new Promise(resolve => {
  completeRequest = resolve;
});
Object.assign(component, {
  _clusterPlanRevision: 7,
  clusterAutoconfigure: null,
  clusterAutoconfigureLoading: false,
  clusterAutoconfigureError: '',
  clusterExecutionProfile: 'balanced',
  clusterAutoconfigurePrefer: 'speed',
  clusterStrategy: 'auto',
  clusterAutoTune: false,
  clusterSamplingRankOnly: true,
  clusterAsyncOverlap: true,
  clusterCacheAffinity: true,
  clusterMaxKvSize: '',
  clusterTargetContextTokens: 131072,
  clusterRingConnectionsPerIp: 1,
  clusterPlanMode: 'model',
  clusterPlanModelPath: '/glm',
  clusterPlanModelSource: 'studio',
  clusterNodePayloads: () => [{ node_id: 'local' }, { node_id: 'studio' }],
  clusterClusterHostsPayload: () => [],
  clusterResponseError: async () =>
    'model weights and KV cache for 131072 tokens do not fit',
});
(async () => {
  const pending = component.autoconfigureCluster();
  while (!completeRequest) await Promise.resolve();
  component._clusterPlanRevision += 1;
  component.clusterTargetContextTokens = 8192;
  completeRequest({ status: 400, ok: false });
  const proposal = await pending;
  process.stdout.write(JSON.stringify({
    proposal,
    error: component.clusterAutoconfigureError,
    target: component.clusterTargetContextTokens,
  }));
})().catch(error => {
  console.error(error);
  process.exit(1);
});
""",
    )

    assert result == {"proposal": None, "error": "", "target": 8192}


def test_settling_a_memory_slider_rechecks_fit_before_replanning():
    result = _run_dashboard_helpers(
        ("clusterMemoryAllowanceChanged",),
        """
const calls = [];
component.clusterCatalogue = { models: [{ model_path: '/old' }] };
component.loadClusterCatalogue = async () => calls.push('catalogue');
component.previewClusterWeightBalance = async () => calls.push('plan');
(async () => {
  await component.clusterMemoryAllowanceChanged();
  process.stdout.write(JSON.stringify({
    calls,
    catalogue: component.clusterCatalogue,
  }));
})().catch(error => {
  console.error(error);
  process.exit(1);
});
""",
    )

    assert result == {"calls": ["catalogue", "plan"], "catalogue": None}


def test_reset_memory_allowance_restores_measured_automatic_default():
    result = _run_dashboard_helpers(
        (
            "saveClusterMemoryAllowances",
            "clusterResetMemoryAllowances",
        ),
        """
const removals = [];
global.window = {
  localStorage: {
    setItem: () => {},
    removeItem: key => removals.push(key),
  },
};
const budget = { node_id: 'MacBook-Pro', capacity_gib: 108, reserve_gib: 28 };
component.clusterMemoryAllowancesGiB = { 'MacBook-Pro': 80 };
component.clusterMemoryLimitsManual = true;
component.clusterWeightTargetsGiB = {};
component.clusterMemoryAllowanceNodes = () => [{
  nodeId: 'MacBook-Pro',
  budget,
  automaticReserveGiB: 66,
}];
component.invalidateClusterPlan = () => {};
component.clusterMemoryAllowanceChanged = () => {};
component.clusterResetMemoryAllowances();
process.stdout.write(JSON.stringify({
  reserve: budget.reserve_gib,
  allowances: component.clusterMemoryAllowancesGiB,
  manual: component.clusterMemoryLimitsManual,
  removals,
}));
""",
    )

    assert result == {
        "reserve": 66,
        "allowances": {},
        "manual": False,
        "removals": ["omlx.cluster.memoryAllowances"],
    }


def test_known_nodes_render_immediately_without_caching_launch_approval():
    result = _run_dashboard_helpers(
        ("loadClusterKnownNodes",),
        """
global.window = {
  localStorage: {
    getItem: key => key === 'omlx.cluster.knownNodes'
      ? JSON.stringify({
          version: 1,
          peers: [{
            ssh: 'studio',
            name: 'Mac Studio',
            service: 'oMLX Distributed',
            transport: 'rdma',
            rdma_available: true,
          }],
          selected_ssh: ['studio'],
          hardware: {
            studio: {
              hostname: 'Studio',
              chip_name: 'Apple M3 Ultra',
              physical_memory_bytes: 256 * (1024 ** 3),
              recommended_working_set_bytes: 223 * (1024 ** 3),
            },
          },
        })
      : null,
  },
};
component._clusterKnownNodesHydrated = false;
component._clusterKnownNodesNeedsSync = false;
component.clusterDiscoveredPeers = null;
component.clusterSelectedPeers = [];
component.clusterPeerProbes = {};
component.loadClusterKnownNodes();
process.stdout.write(JSON.stringify({
  peer: component.clusterDiscoveredPeers[0],
  selected: component.clusterSelectedPeers.map(peer => peer.ssh),
  peerSsh: component.clusterPeerSsh,
  hardware: component.clusterPeerProbes.studio.status.node,
  runtimeCompatible: component.clusterPeerProbes.studio.runtime_compatible ?? null,
  needsSync: component._clusterKnownNodesNeedsSync,
}));
""",
    )

    assert result == {
        "peer": {
            "ssh": "studio",
            "name": "Mac Studio",
            "service": "oMLX Distributed",
            "transport": "rdma",
            "rdma_available": True,
            "cached": True,
        },
        "selected": ["studio"],
        "peerSsh": "studio",
        "hardware": {
            "hostname": "Studio",
            "chip_name": "Apple M3 Ultra",
            "physical_memory_bytes": 256 * 1024**3,
            "recommended_working_set_bytes": 223 * 1024**3,
            "admission_ceiling_bytes": 0,
        },
        "runtimeCompatible": None,
        "needsSync": True,
    }


def test_connectx_cuda_pair_is_one_logical_unit_with_two_physical_members():
    result = _run_dashboard_helpers(
        (
            "clusterFriendlyMacName",
            "clusterWorkerPeers",
            "clusterQuickNodes",
            "clusterLogicalNodes",
            "clusterCudaFabricVerificationKey",
            "clusterNodeHardwareLabel",
            "clusterDeviceCountLabel",
            "clusterMemoryNodes",
            "clusterCombinedUsableMemoryGiB",
            "clusterCombinedPhysicalMemoryGiB",
            "clusterMemoryGiBLabel",
            "clusterCombinedMemoryLabel",
        ),
        """
const gib = 1024 ** 3;
component.clusterStatus = { node: {
  hostname: 'Mac Studio', accelerator: 'metal', accelerator_vendor: 'apple',
  chip_name: 'Apple M3 Ultra', physical_memory_bytes: 256 * gib,
  recommended_working_set_bytes: 220 * gib,
} };
component.clusterSelectedPeers = [
  { ssh: 'spark-a', name: 'Spark A' },
  { ssh: 'spark-b', name: 'Spark B' },
];
component.clusterPeerProbes = {
  'spark-a': { runtime_compatible: true, status: { node: {
    hostname: 'spark-a', chip_name: 'NVIDIA GB10', accelerator: 'cuda',
    accelerator_vendor: 'nvidia', fabric_kind: 'connectx-7',
    physical_memory_bytes: 128 * gib, recommended_working_set_bytes: 115 * gib,
  } } },
  'spark-b': { runtime_compatible: true, status: { node: {
    hostname: 'spark-b', chip_name: 'NVIDIA GB10', accelerator: 'cuda',
    accelerator_vendor: 'nvidia', fabric_kind: 'connectx-7',
    physical_memory_bytes: 128 * gib, recommended_working_set_bytes: 115 * gib,
  } } },
};
component.clusterPlanNodes = [
  { node_id: 'Mac Studio', capacity_gib: 220, reserve_gib: 0 },
  { node_id: 'spark-a', capacity_gib: 115, reserve_gib: 0 },
  { node_id: 'spark-b', capacity_gib: 115, reserve_gib: 0 },
];
const physical = component.clusterQuickNodes();
const logical = component.clusterLogicalNodes();
const supernode = logical.find(node => node.memberCount === 2);
process.stdout.write(JSON.stringify({
  physicalCount: physical.length,
  logicalCount: logical.length,
  memberCount: supernode.memberCount,
  ranks: supernode.ranks,
  label: component.clusterNodeHardwareLabel(supernode),
  countLabel: component.clusterDeviceCountLabel(),
  memoryLabel: component.clusterCombinedMemoryLabel(),
}));
""",
    )

    assert result == {
        "physicalCount": 3,
        "logicalCount": 2,
        "memberCount": 2,
        "ranks": [1, 2],
        "label": "2× NVIDIA CUDA pair · GB10 · 256 GB pooled",
        "countLabel": "2 visual groups · 3 execution ranks",
        "memoryLabel": "450 GiB model-usable of 512 GiB installed",
    }


def test_dashboard_verifies_connectx_and_persists_group_in_plan_payload():
    result = _run_dashboard_helpers(
        (
            "clusterCudaFabricVerificationKey",
            "clusterNeuralFabricJob",
            "clusterCanVerifyCudaSupernode",
            "verifyCudaSupernode",
        ),
        """
const calls = [];
global.window = { location: { href: '' } };
global.fetch = async (url, options) => {
  calls.push({ url, body: JSON.parse(options.body) });
  return {
    status: 200,
    ok: true,
    json: async () => ({
      ok: true,
      verified: true,
      group_id: 'connectx-live-proof',
      payload_bytes_per_second: 3 * (1024 ** 3),
    }),
  };
};
component.clusterResponseError = async (_response, fallback) => fallback;
component.invalidateClusterPlan = () => { component.invalidated = true; };
component.clusterActivationLoading = false;
component.clusterStatus = { runtime_jobs: { jobs: [] } };
component.clusterCudaFabricVerificationLoading = '';
component.clusterCudaFabricVerificationError = '';
component.clusterCudaFabricVerifications = {};
component.clusterPlanNodes = [
  { node_id: 'spark-a', ssh: 'spark-a', accelerator: 'cuda' },
  { node_id: 'spark-b', ssh: 'spark-b', accelerator: 'cuda' },
];
const node = {
  memberCount: 2,
  accelerator: 'cuda',
  members: [
    { name: 'Spark A', ssh: 'spark-a', online: true },
    { name: 'Spark B', ssh: 'spark-b', online: true },
  ],
};
(async () => {
  const verified = await component.verifyCudaSupernode(node);
  process.stdout.write(JSON.stringify({
    verified,
    calls,
    planNodes: component.clusterPlanNodes,
    invalidated: component.invalidated,
    error: component.clusterCudaFabricVerificationError,
  }));
})().catch(error => { console.error(error); process.exit(1); });
""",
    )

    assert result["verified"] is True
    assert result["calls"] == [
        {
            "url": "/admin/api/cluster/cuda-fabric/verify",
            "body": {
                "hosts": [
                    {"node_id": "Spark A", "ssh": "spark-a"},
                    {"node_id": "Spark B", "ssh": "spark-b"},
                ]
            },
        }
    ]
    assert all(node["fabric_verified"] for node in result["planNodes"])
    assert {
        node["fabric_group_id"] for node in result["planNodes"]
    } == {"connectx-live-proof"}
    assert result["invalidated"] is True
    assert result["error"] == ""


def test_activation_reverifies_a_cached_connectx_pair():
    result = _run_dashboard_helpers(
        ("ensureCudaSupernodesVerified",),
        """
let calls = 0;
component.clusterLogicalNodes = () => [{
  memberCount: 2,
  accelerator: 'cuda',
  fabricVerified: true,
}];
component.verifyCudaSupernode = async () => { calls += 1; return true; };
(async () => {
  const verified = await component.ensureCudaSupernodesVerified();
  process.stdout.write(JSON.stringify({ verified, calls }));
})().catch(error => { console.error(error); process.exit(1); });
""",
    )

    assert result == {"verified": True, "calls": 1}


def test_activation_guard_rebuilds_a_plan_changed_by_cuda_reverification():
    result = _run_dashboard_helpers(
        ("prepareCudaSupernodesForActivation",),
        """
let planBuilds = 0;
component._clusterPlanRevision = 4;
component.ensureCudaSupernodesVerified = async () => {
  component._clusterPlanRevision = 5;
  return true;
};
component.runClusterPlan = async () => { planBuilds += 1; };
component.clusterActivationReady = () => true;
(async () => {
  const prepared = await component.prepareCudaSupernodesForActivation({
    rebuildPlan: true,
  });
  process.stdout.write(JSON.stringify({ prepared, planBuilds }));
})().catch(error => { console.error(error); process.exit(1); });
""",
    )

    assert result == {"prepared": True, "planBuilds": 1}


def test_manual_activation_stops_when_cuda_reverification_fails():
    result = _run_dashboard_helpers(
        ("activateClusterDeployment",),
        """
let guardCalls = 0;
let deploymentCalls = 0;
global.fetch = async () => {
  deploymentCalls += 1;
  return { status: 200, ok: true, json: async () => ({}) };
};
component.clusterActivationReady = () => true;
component.prepareCudaSupernodesForActivation = async () => {
  guardCalls += 1;
  return false;
};
(async () => {
  await component.activateClusterDeployment();
  process.stdout.write(JSON.stringify({ guardCalls, deploymentCalls }));
})().catch(error => { console.error(error); process.exit(1); });
""",
    )

    assert result == {"guardCalls": 1, "deploymentCalls": 0}


def test_model_picker_ranks_real_fit_and_uses_friendly_names():
    result = _run_dashboard_helpers(
        (
            "clusterAllModels",
            "clusterModelCandidates",
            "clusterModelHostsLabel",
            "clusterModelOptions",
            "clusterRecommendedModels",
            "clusterModelGroupLabel",
            "clusterModelDisplayName",
            "clusterModelOwner",
            "clusterModelFailureLabel",
            "clusterModelBadge",
            "clusterCatalogueFit",
        ),
        """
component.models = [
  { id: 'owner/small', model_path: '/small', model_type: 'llm',
    estimated_size: 10, is_default: true },
  { id: 'lab/big', model_path: '/big', model_type: 'llm',
    estimated_size: 90 },
  { id: 'huge', model_path: '/huge', model_type: 'llm',
    estimated_size: 400, is_favorite: true },
];
component.clusterCatalogue = { models: [
  { model_path: '/big', fits: true, nodes_required: 2 },
  { model_path: '/small', fits: true, nodes_required: 1 },
  { model_path: '/huge', fits: false, nodes_required: 0 },
] };
const options = component.clusterModelOptions();
process.stdout.write(JSON.stringify({
  order: options.map(model => model.model_path),
  recommended: component.clusterRecommendedModels().map(model => model.model_path),
  firstGroup: component.clusterModelGroupLabel(0),
  allGroup: component.clusterModelGroupLabel(2),
  bestBadge: component.clusterModelBadge(options[0]),
  tooLargeBadge: component.clusterModelBadge(options[2]),
  displayName: component.clusterModelDisplayName(options[0]),
  owner: component.clusterModelOwner(options[0]),
}));
""",
    )

    assert result == {
        "order": ["/big", "/small", "/huge"],
        "recommended": ["/big", "/small"],
        "firstGroup": "Recommended for this pool",
        "allGroup": "All other models",
        "bestBadge": "Best for this pool",
        "tooLargeBadge": "Does not fit",
        "displayName": "big",
        "owner": "lab",
    }


def test_primary_action_speaks_in_user_actions_not_cluster_roles():
    result = _run_dashboard_helpers(
        (
            "clusterAllModels",
            "clusterPrimaryActionLabel",
            "clusterPrimaryActionDisabled",
            "clusterPrimaryDeployment",
            "clusterSelectedModel",
            "clusterCatalogueFit",
            "clusterModelDisplayName",
            "clusterLiveJobs",
            "clusterWorkerPeers",
            "clusterQuickNodes",
            "clusterFriendlyMacName",
            "clusterPeerDisplayName",
        ),
        """
component.models = [
  { id: 'lab/Qwen3.6-27B-q3', model_path: '/qwen', model_type: 'llm' },
];
component.clusterPlanModelPath = '/qwen';
component.clusterPeerSsh = 'studio.local';
component.clusterSelectedPeers = [{ ssh: 'studio.local', name: 'Mac Studio' }];
component.clusterDiscoveredPeers = component.clusterSelectedPeers;
component.clusterStatus = { node: { hostname: 'MacBook Pro' }, runtime_jobs: { jobs: [] } };
component.clusterDeactivatingId = '';
component.clusterActivationLoading = false;
component.clusterLinkSetupLoading = false;
component.clusterAutoconfigureLoading = false;
const start = component.clusterPrimaryActionLabel();
component.clusterCatalogue = {
  models: [{ model_path: '/qwen', fits: false, failure_kind: 'single_node_only' }],
};
const rejected = component.clusterPrimaryActionLabel();
const rejectedDisabled = component.clusterPrimaryActionDisabled();
component.clusterCatalogue = null;
component.clusterError = 'The approved plan changed';
const retry = component.clusterPrimaryActionLabel();
component.clusterError = '';
component.clusterDeployments = [
  { deployment_id: 'qwen-pair', model: '/qwen' },
];
component.clusterStatus.runtime_jobs.jobs = [{ live: true }];
const stop = component.clusterPrimaryActionLabel();
component.clusterDeployments = [];
component.clusterStatus.runtime_jobs.jobs = [];
component.clusterPeerSsh = '';
component.clusterSelectedPeers = [];
const find = component.clusterPrimaryActionLabel();
process.stdout.write(JSON.stringify({ start, rejected, rejectedDisabled, retry, stop, find }));
""",
    )

    assert result == {
        "start": "Start Qwen3.6-27B-q3 on 2 devices",
        "rejected": "Choose a cluster-compatible model",
        "rejectedDisabled": True,
        "retry": "Retry Qwen3.6-27B-q3 setup",
        "stop": "Stop",
        "find": "Find workers",
    }


def test_three_mac_cluster_builds_three_rank_hosts_and_valid_tp_choices():
    result = _run_dashboard_helpers(
        (
            "clusterWorkerPeers",
            "clusterClusterHostsPayload",
            "clusterTensorParallelOptions",
        ),
        """
component.clusterSelectedPeers = [
  { ssh: 'studio.local', name: 'Mac Studio' },
  { ssh: 'mini.local', name: 'Mac mini' },
];
component.clusterPlanNodes = [
  { node_id: 'MacBook Pro' },
  { node_id: 'Mac Studio' },
  { node_id: 'Mac mini' },
];
component.clusterLocalIp = '10.0.0.1';
component.clusterPeerIp = '10.0.0.2';
component.clusterFabric = { hosts: [
  { host: '127.0.0.1', ips: ['10.0.0.1'], rdma: [null, 'rdma0', 'rdma1'] },
  { host: 'studio.local', ips: ['10.0.0.2'], rdma: ['rdma0', null, 'rdma2'] },
  { host: 'mini.local', ips: ['10.0.0.3'], rdma: ['rdma1', 'rdma2', null] },
] };
process.stdout.write(JSON.stringify({
  hosts: component.clusterClusterHostsPayload(),
  tp: component.clusterTensorParallelOptions(),
}));
""",
    )

    assert [host["ssh"] for host in result["hosts"]] == [
        "127.0.0.1",
        "studio.local",
        "mini.local",
    ]
    assert [host["node_id"] for host in result["hosts"]] == [
        "MacBook Pro",
        "Mac Studio",
        "Mac mini",
    ]
    assert result["tp"] == [1, 3]


def test_three_mac_weight_slider_rebalances_every_other_rank():
    result = _run_dashboard_helpers(
        (
            "clusterWeightPlan",
            "clusterWeightNodes",
            "clusterWeightTargetGiB",
            "clusterWeightMinimumGiB",
            "clusterWeightTotalGiB",
            "clusterWeightBounds",
            "clusterSetWeightTarget",
            "clusterNodePayloads",
        ),
        """
component.clusterPlanNodes = [
  { node_id: 'MacBook Pro', capacity_gib: 80, reserve_gib: 0, role: 'workstation' },
  { node_id: 'Mac Studio', capacity_gib: 100, reserve_gib: 0, role: 'headless' },
  { node_id: 'Mac mini', capacity_gib: 60, reserve_gib: 0, role: 'headless' },
];
component.clusterPlan = {
  tensor_parallel_size: 1,
  model: {
    fixed_weight_bytes: 0,
    layer_weight_bytes: Array(12).fill(10 * 1024 ** 3),
  },
  assignments: [
    { node_id: 'MacBook Pro', rank: 0, layer_weight_bytes: 20 * 1024 ** 3,
      fixed_weight_bytes: 0 },
    { node_id: 'Mac Studio', rank: 1, layer_weight_bytes: 70 * 1024 ** 3,
      fixed_weight_bytes: 0 },
    { node_id: 'Mac mini', rank: 2, layer_weight_bytes: 30 * 1024 ** 3,
      fixed_weight_bytes: 0 },
  ],
};
component.clusterAutoconfigure = null;
component.clusterWeightTargetsGiB = {};
component.clusterSplitGiB = null;
component.clusterSetWeightTarget(component.clusterWeightNodes()[0], 30);
process.stdout.write(JSON.stringify({
  targets: component.clusterWeightTargetsGiB,
  payloads: component.clusterNodePayloads(),
}));
""",
    )

    assert result["targets"] == {
        "MacBook Pro": 30,
        "Mac Studio": 62.5,
        "Mac mini": 27.5,
    }
    assert [
        round(node["target_weight_bytes"] / 1024**3, 1)
        for node in result["payloads"]
    ] == [30, 62.5, 27.5]


def test_weight_slider_is_hidden_for_an_unsplittable_architecture():
    result = _run_dashboard_helpers(
        ("clusterWeightPlan", "clusterSplitAvailable"),
        """
component.clusterPlan = {
  tensor_parallel_size: 1,
  assignments: [{ rank: 0 }, { rank: 1 }],
};
component.clusterAutoconfigure = null;
component.clusterSelectedModel = () => ({ model_path: '/qwen' });
component.clusterCatalogueFit = () => ({ fits: false, splittable: false,
  failure_kind: 'single_node_only' });
process.stdout.write(JSON.stringify({ available: component.clusterSplitAvailable() }));
""",
    )

    assert result == {"available": False}


def test_first_run_adopts_omlx_peers_before_transport_has_been_measured():
    source = _method_source("initializeClusterSetup")

    assert "peer.service === 'oMLX Distributed'" in source
    assert "omlxPeers.length" in source
    assert "automaticPeers" in source
    assert "if (this.clusterWorkerPeers().length" in source
    assert source.count("await this.measureClusterBudgets()") == 1
    assert "await this.previewClusterWeightBalance()" in source


def test_initialization_previews_by_default_but_polling_skips_it():
    result = _run_dashboard_helpers(
        ("initializeClusterSetup", "refreshClusterExperience"),
        """
const calls = [];
Object.assign(component, {
  clusterStatus: null,
  clusterDiscoveredPeers: [],
  clusterPeerProbe: null,
  clusterModelInventory: {},
  clusterModelInventoryLoading: false,
  clusterCatalogueLoading: false,
  _clusterKnownNodesNeedsSync: false,
  _clusterDiscoveryRefreshCounter: 4,
  clusterWorkerPeers: () => [],
  clusterProbeBackoffActive: () => false,
  clusterRecommendedModels: () => [],
  clusterModelCandidates: () => [],
  loadClusterPeerHardware: async () => {},
  normalizeClusterTensorParallelSize: () => {},
  previewClusterWeightBalance: async () => calls.push('preview'),
  loadClusterRuntime: async () => calls.push('runtime'),
  loadClusterIncidents: async () => calls.push('incidents'),
  discoverClusterPeers: async () => calls.push('discover'),
  loadClusterJoinStatus: async () => calls.push('join'),
});
(async () => {
  await component.initializeClusterSetup();
  await component.refreshClusterExperience();
  process.stdout.write(JSON.stringify({ calls }));
})().catch(error => {
  console.error(error);
  process.exit(1);
});
""",
    )

    assert result == {
        "calls": ["preview", "runtime", "incidents", "discover", "join"]
    }


def test_initialization_resyncs_live_capabilities_before_measuring_budgets():
    source = _method_source("initializeClusterSetup")

    hardware = source.index("await this.loadClusterPeerHardware()")
    resync = source.index("this.syncClusterNodesFromPeers()", hardware)
    budgets = source.index("await this.measureClusterBudgets()", resync)

    assert hardware < resync < budgets


def test_cached_worker_hardware_is_reprobed_for_runtime_paths():
    result = _run_dashboard_helpers(
        ("loadClusterPeerHardware",),
        """
const calls = [];
global.fetch = async (_url, options) => {
  const request = JSON.parse(options.body);
  calls.push(request.ssh);
  return {
    ok: true,
    json: async () => ({
      status: {
        node: { hostname: request.ssh, accelerator: 'cuda' },
        runtime: { python_executable: '/opt/omlx/bin/python' },
      },
    }),
  };
};
Object.assign(component, {
  clusterPeerProbe: null,
  clusterPeerSsh: '',
  clusterLocalIp: '',
  clusterWorkerPeers: () => [
    { ssh: 'spark-a.local' },
    { ssh: 'spark-b.local' },
  ],
  clusterPeerProbes: {
    'spark-a.local': { cached: true, status: { node: { hostname: 'old-a' } } },
    'spark-b.local': { cached: true, status: { node: { hostname: 'old-b' } } },
  },
  saveClusterKnownNodes: () => {},
});
(async () => {
  await component.loadClusterPeerHardware();
  process.stdout.write(JSON.stringify({
    calls,
    python: component.clusterPeerProbes['spark-b.local']
      .status.runtime.python_executable,
  }));
})().catch(error => {
  console.error(error);
  process.exit(1);
});
""",
    )

    assert result == {
        "calls": ["spark-a.local", "spark-b.local"],
        "python": "/opt/omlx/bin/python",
    }


def test_automatic_weight_preview_cannot_start_rank_processes():
    source = _method_source("previewClusterWeightBalance")

    assert "await this.runClusterPlan()" in source
    assert "startCluster" not in source
    assert "activateCluster" not in source


def test_measured_budgets_invalidate_optimistic_catalogue_results():
    source = _method_source("measureClusterBudgets")

    assert "this.clusterCatalogue = null" in source
    assert "this.invalidateClusterPlan()" in source


def test_activation_progress_is_runtime_driven_not_timer_simulated():
    source = _method_source("activateClusterProposal")

    assert "startClusterActivationProgress" in source
    assert "stopClusterActivationProgress" in source
    assert "progressSteps" not in source
    assert "setTimeout" not in source


def test_one_click_posts_the_server_activation_payload_unchanged():
    activation = {
        "model_path": "/models/qwen",
        "backend": "jaccl",
        "nodes": [{"node_id": "local"}, {"node_id": "studio"}],
        "hosts": [{"ssh": "127.0.0.1"}, {"ssh": "studio.local"}],
        "preflight": True,
        "tensor_parallel_size": 2,
        "auto_tune": True,
        "marker": "server-owned-payload",
    }
    result = _run_one_click(
        {
            "ready_to_activate": True,
            "activation": activation,
            "fabric": None,
            "tensor_parallel_size": 2,
            "summary": "2-way tensor parallel",
        }
    )

    operational = [
        call for call in result["calls"]
        if not call["url"].endswith("/runtime")
    ]
    assert [call["url"] for call in operational] == [
        "/admin/api/cluster/autoconfigure",
        "/admin/api/cluster/deployments",
    ]
    assert operational[1]["body"] == activation
    assert any(call["url"].endswith("/runtime") for call in result["calls"])
    assert result["deploymentLoads"] == 1
    assert result["activationResult"]["ok"] is True
    assert result["clusterError"] == ""


def test_one_click_never_posts_a_blocked_proposal():
    result = _run_one_click(
        {
            "ready_to_activate": False,
            "activation": {"must_not": "launch"},
            "fabric": None,
            "tensor_parallel_size": 1,
            "summary": "blocked",
            "preflight": "Studio is missing mlx_vlm.",
            "preflight_issues": [
                {
                    "node_id": "studio",
                    "kind": "import_missing",
                    "detail": "No module named mlx_vlm",
                    "commands": ["pip install mlx-vlm"],
                }
            ],
        }
    )

    assert [call["url"] for call in result["calls"]] == [
        "/admin/api/cluster/autoconfigure"
    ]
    assert result["deploymentLoads"] == 0
    assert result["activationResult"] is None
    assert result["autoconfigureError"] == "Studio is missing mlx_vlm."


def test_one_click_does_not_stage_or_launch_without_a_verified_fabric():
    result = _run_one_click(
        {
            "ready_to_activate": False,
            "fabric_ready": False,
            "fabric_blocker": "worker 1 cannot reach worker 2",
            "staging": {"ready": False},
            "activation": {"must_not": "launch"},
            "fabric": {"ok": False},
            "tensor_parallel_size": 1,
            "summary": "blocked fabric",
        }
    )

    assert [call["url"] for call in result["calls"]] == [
        "/admin/api/cluster/autoconfigure"
    ]
    assert result["deploymentLoads"] == 0
    assert result["activationResult"] is None
    assert result["autoconfigureError"] == "worker 1 cannot reach worker 2"


def test_one_click_prepares_thunderbolt_then_continues_to_activation():
    result = _run_one_click(
        {
            "ready_to_activate": True,
            "activation": {
                "model_path": "/models/qwen",
                "backend": "jaccl",
                "nodes": [{"node_id": "local"}, {"node_id": "studio"}],
                "hosts": [{"ssh": "127.0.0.1"}, {"ssh": "studio.local"}],
            },
            "fabric": None,
            "tensor_parallel_size": 2,
            "summary": "ready",
        },
        link_status={
            "state": "rdma_needs_setup",
            "ready": False,
            "setup_available": True,
        },
    )

    operational = [
        call for call in result["calls"]
        if not call["url"].endswith("/runtime")
    ]
    assert [call["url"] for call in operational] == [
        "/admin/api/cluster/link-setup",
        "/admin/api/cluster/autoconfigure",
        "/admin/api/cluster/deployments",
    ]
    assert operational[0]["body"] == {
        "hosts": ["127.0.0.1", "studio.local"]
    }
    assert result["activationResult"]["ok"] is True


def test_probe_failures_back_off_instead_of_retrying_every_ten_seconds():
    """SSH retries against an unpaired Mac escalate 10s -> 60s and hold there.

    A success or a completed pairing clears the hold; the automatic refresh
    loop consults ``clusterProbeBackoffActive`` before firing another probe.
    """

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to execute the dashboard orchestration")
    methods = ",\n".join(
        _method_source(name)
        for name in (
            "clusterProbeBackoffActive",
            "resetClusterProbeBackoff",
            "_escalateClusterProbeBackoff",
            "probeClusterPeer",
        )
    )
    script = f"""
let nowMs = 1_000_000;
Date.now = () => nowMs;
global.window = {{ location: {{ href: '' }} }};
let probeOk = false;
global.fetch = async () => probeOk
  ? {{ status: 200, ok: true, json: async () => ({{ status: {{ node: {{}} }} }}) }}
  : {{ status: 503, ok: false }};
const component = {{
  {methods},
  clusterPeerSsh: 'studio.local',
  clusterLocalIp: '',
  clusterPeerProbeLoading: false,
  clusterPeerProbes: {{}},
  clusterPlanNodes: [],
  clusterError: '',
  clusterConnectionError: '',
  async clusterResponseError() {{ return 'Permission denied (publickey).'; }},
  dismissClusterGuidance() {{}},
  saveClusterKnownNodes() {{}},
  async $nextTick() {{}},
  invalidateClusterPlan() {{}},
  async loadClusterFabric() {{}},
}};
(async () => {{
  const holds = [];
  for (let attempt = 0; attempt < 5; attempt += 1) {{
    await component.probeClusterPeer();
    holds.push(component._clusterProbeHoldUntilMs - nowMs);
    nowMs = component._clusterProbeHoldUntilMs + 1;
  }}
  const activeBeforeExpiry = (() => {{
    nowMs = component._clusterProbeHoldUntilMs - 1;
    return component.clusterProbeBackoffActive();
  }})();
  const activeAfterExpiry = (() => {{
    nowMs = component._clusterProbeHoldUntilMs + 1;
    return component.clusterProbeBackoffActive();
  }})();
  probeOk = true;
  await component.probeClusterPeer();
  console.log(JSON.stringify({{
    holds,
    activeBeforeExpiry,
    activeAfterExpiry,
    failuresAfterSuccess: component._clusterProbeFailureCount,
    activeAfterSuccess: component.clusterProbeBackoffActive(),
  }}));
}})();
"""
    completed = subprocess.run(
        [node, "-e", script], capture_output=True, text=True, timeout=30
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)

    assert result["holds"] == [10000, 20000, 40000, 60000, 60000]
    assert result["activeBeforeExpiry"] is True
    assert result["activeAfterExpiry"] is False
    assert result["failuresAfterSuccess"] == 0
    assert result["activeAfterSuccess"] is False


def test_the_automatic_refresh_paths_respect_the_probe_hold():
    """The 10s loop and budget measurement consult the hold; clicks reset it."""

    source = DASHBOARD_JS.read_text()

    assert source.count("this.clusterProbeBackoffActive()") >= 3
    select = _method_source("selectClusterDiscoveredPeer")
    assert "resetClusterProbeBackoff()" in select
    exchange = _method_source("exchangeKeysWithPeer")
    assert "resetClusterProbeBackoff()" in exchange
