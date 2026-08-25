# SPDX-License-Identifier: Apache-2.0
"""Static contracts for the native distributed-cluster dashboard view."""

import json
from pathlib import Path

from omlx.admin import routes as admin_routes

ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text()


def test_dashboard_renders_cluster_partial():
    rendered = admin_routes.templates.get_template("dashboard.html").render()

    assert "data-cluster-view" in rendered
    assert "data-cluster-node-card" in rendered
    assert "data-cluster-planner" in rendered
    assert "data-cluster-peer" in rendered
    assert "data-cluster-activation" in rendered
    assert "data-cluster-live-summary" in rendered
    assert "data-cluster-runtime" not in rendered
    assert "Start Cluster" in rendered
    assert "Activate manual plan" in rendered
    assert '@click="startCluster()"' in rendered
    assert "Cluster live" in rendered
    assert "Copy API details" in rendered
    assert "Live cluster measurements" in rendered


def test_cluster_navigation_exists_for_desktop_and_mobile():
    navbar = _read("omlx/admin/templates/dashboard/_navbar.html")
    dashboard = _read("omlx/admin/templates/dashboard.html")

    assert dashboard.count('{% include "dashboard/_cluster.html" %}') == 1
    assert navbar.count("setMainTab('cluster')") == 2
    assert navbar.count("mainTab === 'cluster'") == 2
    assert navbar.count("navbar.tab.cluster") == 2
    assert navbar.count(
        'x-show="globalSettings.server.distributed_inference_active"'
    ) == 2


def test_distributed_inference_is_an_advanced_restart_scoped_opt_in():
    settings = _read("omlx/admin/templates/dashboard/_settings.html")
    javascript = _read("omlx/admin/static/js/dashboard.js")

    distributed_section = settings.index("<!-- Distributed inference subsection -->")
    performance_section = settings.index("<!-- Performance subsection -->")

    assert distributed_section < performance_section
    assert "settings.advanced.distributed_inference" in settings
    assert "settings.advanced.distributed_inference_enabled" in settings
    assert "settings.advanced.distributed_inference_hint" in settings
    assert "distributed_inference_enabled: false" in javascript
    assert "distributed_inference_active: false" in javascript
    assert "if (tab === 'cluster' && !this.globalSettings.server.distributed_inference_active) return" in javascript


def test_cluster_dashboard_uses_authenticated_cluster_apis():
    javascript = _read("omlx/admin/static/js/dashboard.js")

    assert "'cluster'" in javascript.split("DASHBOARD_MAIN_TABS", 1)[1].split(";", 1)[0]
    assert "/admin/api/cluster/status" in javascript
    assert "/admin/api/cluster/worker-smoke" in javascript
    assert "/admin/api/cluster/collective-smoke" in javascript
    assert "/admin/api/cluster/pipeline-smoke" in javascript
    assert "/admin/api/cluster/peer-probe" in javascript
    assert "/admin/api/cluster/deployments" in javascript
    assert "/admin/api/cluster/runtime" in javascript
    assert "/admin/api/cluster/diagnostics" in javascript
    assert "/admin/api/cluster/discover" in javascript
    assert "/admin/api/cluster/plan" in javascript
    assert "async runClusterPlan()" in javascript
    assert "async startCluster()" in javascript
    assert "async activateClusterProposal(activation)" in javascript
    assert "async loadClusterStatus()" in javascript
    assert "async downloadClusterDiagnostics()" in javascript
    assert "async activateClusterDeployment()" in javascript
    assert "clusterRuntimeAssignments(job)" in javascript
    assert "formatClusterRate(rate)" in javascript
    assert "formatClusterBandwidth(bytesPerSecond)" in javascript
    assert "formatClusterLatency(seconds)" in javascript
    assert "clusterRuntimePhaseLabel(job)" in javascript
    assert "clusterExecutionProfile" in javascript
    assert "Preparing embeddings and shared weights" in javascript
    assert "Loading model layers" in javascript
    assert "Sharding model layers" in javascript
    assert "job.layers_loaded" in javascript
    assert "job.layers_total" in javascript


def test_cuda_workers_join_from_a_gui_generated_one_time_command():
    cluster = _read("omlx/admin/templates/dashboard/_cluster.html")
    javascript = _read("omlx/admin/static/js/dashboard.js")

    assert "data-cluster-cuda-enrollment" in cluster
    assert "data-cluster-generate-cuda-join" in cluster
    assert "data-cluster-cuda-join-command" in cluster
    assert "data-cluster-joined-cuda-nodes" in cluster
    assert "Add a CUDA worker" in cluster
    assert "Generate a fresh command for the second CUDA worker" in cluster
    assert "Server host is set to 0.0.0.0 in Settings" in javascript
    assert "async generateClusterCudaJoinCommand()" in javascript
    assert "async loadClusterJoinStatus()" in javascript
    assert "async revokeClusterJoinCommand()" in javascript
    assert "/admin/api/cluster/join-keys" in javascript
    assert "/admin/api/cluster/join-status" in javascript
    assert "ttl_seconds: 1800" in javascript
    assert "service: 'oMLX CUDA Worker'" in javascript
    assert "selected.set(peer.ssh, peer)" in javascript
    # The command contains the one-time credential and therefore stays only
    # in current page memory. Persisted known-node hints contain identities,
    # hardware, and SSH targets, never the enrollment command.
    assert "localStorage.setItem('omlx.cluster.join" not in javascript
    assert "localStorage.getItem('omlx.cluster.join" not in javascript


def test_cluster_dashboard_renders_advanced_activation_controls():
    cluster = _read("omlx/admin/templates/dashboard/_cluster.html")

    assert "clusterShowAdvanced = !clusterShowAdvanced" in cluster
    for model in (
        "clusterAutoconfigurePrefer",
        "clusterAutoTune",
        "clusterSamplingRankOnly",
        "clusterAsyncOverlap",
        "clusterCacheAffinity",
        "clusterRingConnectionsPerIp",
    ):
        assert f'x-model="{model}"' in cluster or f'x-model.number="{model}"' in cluster
    assert "x-show=\"clusterBackend === 'ring'\"" in cluster


def test_cluster_dashboard_names_roles_and_uses_detected_topology():
    cluster = _read("omlx/admin/templates/dashboard/_cluster.html")
    javascript = _read("omlx/admin/static/js/dashboard.js")

    assert "Coordinator · rank 0" in cluster
    assert "Worker · rank " in cluster
    assert "clusterPeerDisplayName()" in cluster
    assert "clusterTopologySummary()" in cluster
    assert "Physical peer detected" not in cluster
    assert "Ports detected" not in cluster
    assert "async initializeClusterSetup({ preview = true } = {})" in javascript
    assert "this.loadClusterKnownNodes();" in javascript
    assert "omlx.cluster.knownNodes" in javascript
    assert "Cached nodes are display hints only" in javascript
    assert "clusterSelectedPeers = preferred" in javascript
    assert "fastPeers.length === 1" not in javascript
    transports = javascript.split("async loadClusterTransports()", 1)[1].split(
        "clusterTransportLabel(", 1
    )[0]
    assert "this.clusterWorkerPeers().map(peer => peer.ssh)" in transports
    peer_health = javascript.split("async loadClusterPeerHealth()", 1)[1].split(
        "async stageClusterModel(", 1
    )[0]
    link_status = javascript.split("async loadClusterLinkStatus()", 1)[1].split(
        "async prepareClusterLink()", 1
    )[0]
    link_setup = javascript.split("async prepareClusterLink()", 1)[1].split(
        "clusterModelCandidates()", 1
    )[0]
    assert "this.clusterWorkerPeers().map(peer => peer.ssh)" in peer_health
    assert "this.clusterWorkerPeers().map(peer => peer.ssh)" in link_status
    assert "this.clusterTransports?.transports || []" in link_setup
    assert "seen.has(key)" in link_setup
    assert "if (!pairs.length && hosts.length === 2)" in link_setup
    node_payload = javascript.split("clusterNodePayloads(", 1)[1].split(
        "async runClusterPlan()", 1
    )[0]
    assert "memory_guard_tier:" in node_payload
    assert "this.globalSettings?.memory?.memory_guard_tier || 'balanced'" in node_payload
    assert "const useDiscoveredFabric = !this.clusterIpsOverridden" in javascript
    assert "if (this.clusterIpsOverridden) return 'ring'" in javascript
    assert "this.clusterFabric?.backend === 'jaccl'" in javascript
    assert "hosts.length > 0 && !this.clusterIpsOverridden" in javascript
    assert "TCP ring · manual addresses" in javascript
    assert 'clusterFabric && !clusterIpsOverridden' in cluster
    assert "clusterCatalogueInputsReady()" in javascript
    assert "requestKey !== this.clusterCatalogueRequestKey()" in javascript


def test_cluster_dashboard_leads_with_one_click_setup():
    cluster = _read("omlx/admin/templates/dashboard/_cluster.html")
    javascript = _read("omlx/admin/static/js/dashboard.js")
    quick_start = cluster.split("data-cluster-quick-start", 1)[1].split(
        "data-cluster-advanced-toggle", 1
    )[0]

    assert "data-cluster-quick-start" in cluster
    assert "data-cluster-topology" in cluster
    assert "data-cluster-neural-fabric" in cluster
    assert "Your compute pool" in cluster
    assert "clusterPairTitle()" in cluster
    assert "Change model" in cluster
    assert "data-cluster-model-picker" in cluster
    assert "data-cluster-primary-action" in cluster
    assert "data-cluster-live-summary" in cluster
    assert "data-cluster-live-stop" in cluster
    assert "clusterLiveModelId()" in cluster
    assert "clusterPublicEndpoint()" in cluster
    assert "data-cluster-advanced-toggle" in cluster
    assert "Diagnostics" in cluster
    assert "Download diagnostic report" in cluster
    assert "runClusterPrimaryAction()" in cluster
    assert cluster.count('@click="startCluster()"') >= 1
    assert "clusterShowModelPicker: false" in javascript
    assert "clusterShowSetupDetails: false" in javascript
    assert "this.clusterShowModelPicker = false" in javascript
    assert "refreshClusterExperience()" in javascript

    # The default path is generated from every detected Mac.
    assert "clusterDeviceCountLabel()" in quick_start
    assert "clusterNodeRankLabel(node)" in quick_start
    assert "Coordinator · ${rankLabel}" in javascript
    assert "Worker · ${rankLabel}" in javascript
    assert "Multi-node preview" not in quick_start
    assert "Refresh node" not in quick_start

    # The topology grows from two to many Macs around a measured ring.
    assert "clusterNeuralFabricNodes()" in quick_start
    assert "clusterNeuralFabricRingPath()" in quick_start
    assert "clusterNeuralFabricEdges()" in javascript
    assert 'style="max-height: 24rem;"' in quick_start

    for technical_panel in (
        "data-cluster-node-card",
        "data-cluster-peer",
        "data-cluster-planner",
    ):
        panel = cluster.split(technical_panel, 1)[0].rsplit("<div", 1)[1]
        assert 'x-show="clusterShowSetupDetails"' in panel

    memory_heading = cluster.index("Memory each accelerator gives")
    memory_panel_prefix = cluster[max(0, memory_heading - 500) : memory_heading]
    assert 'x-show="clusterShowSetupDetails"' in memory_panel_prefix
    assert (
        'x-if="clusterShowSetupDetails && clusterWeightPlan() && clusterSplitAvailable()"'
        in cluster
    )


def test_cluster_model_search_keeps_the_icon_clear_of_the_text():
    cluster = _read("omlx/admin/templates/dashboard/_cluster.html")

    assert cluster.count('placeholder="Search models across the pool"') == 2
    assert cluster.count(
        "pointer-events-none absolute left-3 top-1/2 -translate-y-1/2"
    ) == 2
    assert "pl-10 pr-3" not in cluster


def test_cluster_model_picker_uses_omlx_models_not_repository_directories():
    cluster = _read("omlx/admin/templates/dashboard/_cluster.html")
    javascript = _read("omlx/admin/static/js/dashboard.js")

    assert "Choose a downloaded model" in cluster
    assert 'x-for="model in clusterModelOptions()"' in cluster
    assert "clusterCatalogueFit(model.model_path)" in cluster
    assert "Recommended for this pool" in javascript
    assert "Best for this pool" in javascript
    assert "Uses ${fit.nodes_required} devices" in javascript
    assert "Mac Studio only" not in javascript
    assert "clusterModelFailureLabel(fit)" in javascript
    assert "single_node_only" in javascript
    assert "Does not fit" in javascript
    assert "Too large" not in javascript
    assert "clusterModelDisplayName(model)" in cluster
    assert "clusterModelOwner(model)" in cluster
    assert "/admin/api/cluster/models" in javascript
    assert "clusterModelInventoryHosts()" in javascript
    assert "clusterModelHostsLabel(model)" in cluster
    assert "this.clusterFriendlyMacName(location.node_id)" in javascript
    assert "(?:omlx\\s+on\\s+)?(?:mac\\s+)?studio" in javascript
    assert "Search models across the pool" in cluster
    assert "model_source: model.model_source" in javascript
    assert "models," in javascript
    assert "model_dir: dir" not in javascript
    assert cluster.count(
        "(model.model_source || '127.0.0.1') + ':' + model.model_path"
    ) >= 2
    assert "'transport-' + index" in cluster


def test_dashboard_does_not_reference_an_unbundled_alpine_plugin():
    for template in (
        "omlx/admin/templates/dashboard/_settings.html",
        "omlx/admin/templates/dashboard/_bench.html",
        "omlx/admin/templates/dashboard/_bench_accuracy.html",
    ):
        assert "x-collapse" not in _read(template)


def test_cluster_quick_start_shows_truthful_combined_memory():
    cluster = _read("omlx/admin/templates/dashboard/_cluster.html")
    javascript = _read("omlx/admin/static/js/dashboard.js")

    assert "data-cluster-memory-allowances" in cluster
    assert "Pooled accelerator memory" in cluster
    assert "Installed memory is shown separately" in cluster
    assert "Splitting stays automatic" in cluster
    assert "clusterCombinedUsableMemoryGiB()" in javascript
    assert "clusterCombinedPhysicalMemoryGiB()" in javascript
    assert "clusterCombinedMemoryLabel()" in cluster


def test_cluster_model_setup_shows_context_and_per_node_kv_cost():
    cluster = _read("omlx/admin/templates/dashboard/_cluster.html")
    javascript = _read("omlx/admin/static/js/dashboard.js")
    css = _read("omlx/admin/static/css/dashboard.css")

    assert "data-cluster-context-planner" in cluster
    assert "Context window" in cluster
    assert "clusterContextOptions()" in cluster
    assert "clusterContextMemoryNodes()" in cluster
    assert "Weights" in cluster
    assert "KV cache" in cluster
    assert "clusterSetAutomaticContext()" in cluster
    assert "'Auto · ' + clusterTokens(" in cluster
    assert "async clusterSetTargetContext(tokens)" in javascript
    assert "async clusterSetAutomaticContext()" in javascript
    assert javascript.count("target_context_tokens: Number(") >= 4
    context_css = css.split(".cluster-context {", 1)[1].split(
        ".cluster-fabric__metrics", 1
    )[0]
    assert "font-size: 0.5rem" not in context_css
    assert "font-size: 0.52rem" not in context_css
    assert "font-size: 0.55rem" not in context_css


def test_cluster_quick_start_has_per_mac_memory_allowances():
    cluster = _read("omlx/admin/templates/dashboard/_cluster.html")
    javascript = _read("omlx/admin/static/js/dashboard.js")

    quick_start = cluster.split("data-cluster-quick-start", 1)[1].split(
        "data-cluster-advanced-toggle", 1
    )[0]
    assert "data-cluster-memory-allowances" in quick_start
    assert "clusterMemoryAllowanceNodes()" in quick_start
    assert "clusterSetMemoryAllowance(" in quick_start
    assert "clusterResetMemoryAllowances()" in quick_start
    assert "Model weight balance" not in quick_start
    assert "target weights" not in quick_start.lower()
    assert "node.budget.reserve_gib" in javascript
    assert "this.clusterWeightTargetsGiB = {}" in javascript
    assert "clusterMemoryAllowancesGiB: {}" in javascript
    assert "omlx.cluster.memoryAllowances" in javascript
    assert "this.clusterManualMemoryAllowanceGiB(measured.node_id)" in javascript
    assert "node.automatic_reserve_gib = automaticReserveGiB" in javascript


def test_cluster_nodes_show_dynamic_hardware_identity_for_every_peer():
    cluster = _read("omlx/admin/templates/dashboard/_cluster.html")
    javascript = _read("omlx/admin/static/js/dashboard.js")

    assert "clusterPeerProbes: {}" in javascript
    assert "async loadClusterPeerHardware()" in javascript
    assert "await this.loadClusterPeerHardware()" in javascript
    assert "result.bootstrap_required" in javascript
    # #2680: the banner must repeat what the probe measured, not assert
    # "not installed" over a runtime it was merely unable to check.
    assert "result.runtime_mismatches" in javascript
    assert "its oMLX worker runtime is not installed yet" in javascript
    assert "probe?.ssh_reachable" in javascript
    assert "this.clusterPeerProbes?.[ssh]" in javascript
    assert "hardware.chip_name" in javascript
    assert "hardware.physical_memory_bytes" in javascript
    assert "clusterNodeHardwareLabel(node)" in javascript
    assert "replace(/^Apple\\s+/i, '')" in javascript
    assert cluster.count("clusterNodeHardwareLabel(node)") >= 3
    assert "M5 Max" not in cluster
    assert "128 GB" not in cluster


def test_cluster_dashboard_groups_cuda_workers_and_shows_pooled_memory():
    cluster = _read("omlx/admin/templates/dashboard/_cluster.html")
    javascript = _read("omlx/admin/static/js/dashboard.js")
    stylesheet = _read("omlx/admin/static/css/dashboard.css")

    assert "Pooled accelerator memory" in cluster
    assert "model-usable of" in javascript
    assert "clusterLogicalNodes()" in javascript
    assert "connectx-7-auto-pair" in javascript
    assert "Verified CUDA pair" in javascript
    assert "ConnectX-7 verified" in cluster
    assert "direct link not verified" in cluster
    assert "Verify ConnectX" in cluster
    assert "/admin/api/cluster/cuda-fabric/verify" in javascript
    assert "<title>NVIDIA CUDA</title>" in cluster
    assert "cluster-fabric-node--cuda" in stylesheet
    assert "cluster-fabric-node--supernode" in stylesheet
    assert "Model shard balance" in cluster
    assert "clusterSetWeightTarget(" in cluster


def test_cluster_neural_fabric_uses_real_runtime_measurements():
    cluster = _read("omlx/admin/templates/dashboard/_cluster.html")
    javascript = _read("omlx/admin/static/js/dashboard.js")
    stylesheet = _read("omlx/admin/static/css/dashboard.css")

    assert "Neural fabric" in cluster
    assert "Ring latency" in javascript
    assert "collective_latency_seconds" in javascript
    assert "collective_bandwidth_bytes_per_second" in javascript
    assert "requestActive && aggregateDecode > 0" not in javascript
    assert "lastRequest?.prefill_progress" in javascript
    assert "prefillProgress?.processed" in javascript
    assert "prefillProgress?.speed" in javascript
    assert "prefillProgress?.average_speed" in javascript
    assert "avg`" in javascript
    assert "now`" in javascript
    assert "prefillProgress?.eta" in javascript
    assert "request?.prefill_progress?.active" in javascript
    assert "Starts after prefill" in javascript
    assert "completionTokens > 1" in javascript
    assert "lastRequest?.decode_tps" in javascript
    assert "lastRequest?.prefill_tps" in javascript
    assert "cluster-fabric-metric__progress" in cluster
    assert ".cluster-fabric-metric__progress" in stylesheet
    assert "Collective throughput" in javascript
    assert "1 MiB all-reduce · startup probe · slowest rank" in javascript
    assert "Negotiated speed · not measured throughput" in javascript
    assert "Measured when the cluster starts" in javascript
    assert "Latest startup probe · slowest hop" in javascript
    assert "clusterNeuralFabricLinkCapacityGbps()" in javascript
    fabric_job = javascript.split("clusterNeuralFabricJob()", 1)[1].split(
        "clusterNeuralFabricMode()", 1
    )[0]
    assert "jobs.find(job => job.live) || null" in fabric_job
    assert "clusterNeuralFabricFiring()" in cluster
    assert ".cluster-fabric--firing" in stylesheet
    assert "@media (prefers-reduced-motion: reduce)" in stylesheet


def test_tensor_parallel_controls_are_derived_from_detected_node_count():
    cluster = _read("omlx/admin/templates/dashboard/_cluster.html")
    javascript = _read("omlx/admin/static/js/dashboard.js")

    assert 'x-for="size in clusterTensorParallelOptions()"' in cluster
    assert "clusterTensorParallelOptions()" in javascript
    assert "nodes > 1 ? [1, nodes] : [1]" in javascript
    assert '<option value="4">' not in cluster


def test_pairing_failure_exposes_omlx_and_terminal_recovery_paths():
    cluster = _read("omlx/admin/templates/dashboard/_cluster.html")
    javascript = _read("omlx/admin/static/js/dashboard.js")

    assert "Open SSH setup in oMLX" in cluster
    assert "Or run this in Terminal on this Mac" in cluster
    assert "Don't have the oMLX SSH key yet?" in cluster
    assert "Terminal SSH can work while oMLX cannot" in cluster
    assert "Step 1 of 3" in cluster
    assert "Step 2 of 3" in cluster
    assert "Step 3 of 3" in cluster
    assert "Repeat in the other direction" in cluster
    assert "copyClusterPairingSecret()" in cluster
    assert "t('cluster.pairing.shared_secret')" in cluster
    assert "t('cluster.pairing.shared_secret_hint')" in cluster
    assert "data-cluster-ssh-setup" in cluster
    assert "openClusterPairingSetup()" in javascript
    assert "Regenerating this key disconnects every paired worker" in javascript
    assert "?overwrite=true" in javascript
    assert "Regenerating disconnects every paired worker" in cluster


def test_every_dashboard_locale_names_cluster_tab():
    locale_dir = ROOT / "omlx/admin/i18n"
    required = {
        "navbar.tab.cluster",
        "settings.advanced.distributed_inference",
        "settings.advanced.distributed_inference_enabled",
        "settings.advanced.distributed_inference_hint",
        "cluster.pairing.shared_secret",
        "cluster.pairing.shared_secret_hint",
        "cluster.pairing.shared_secret_placeholder",
        "cluster.pairing.generate",
        "cluster.pairing.copy",
    }

    for locale_path in locale_dir.glob("*.json"):
        locale = json.loads(locale_path.read_text())
        missing = {key for key in required if not locale.get(key)}
        assert not missing, f"{locale_path.name}: missing {sorted(missing)}"
