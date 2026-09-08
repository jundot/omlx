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
    "/admin/api/cluster/models",
    "/admin/api/cluster/catalogue",
    "/admin/api/cluster/peer-probe",
    "/admin/api/cluster/autoconfigure",
    "/admin/api/cluster/plan",
    "/admin/api/cluster/deployments",
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
        "/admin/api/cluster/plan",
        "/admin/api/cluster/deployments",
    ):
        assert endpoint in javascript

    # DELETE /api/cluster/devices/{node_id} (unpair).
    assert re.search(r"`/api/cluster/devices/\$\{encodeURIComponent\(nodeId\)\}`", javascript)


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


def test_active_state_lists_devices_and_deactivates():
    template = _read(TEMPLATE)
    javascript = _read(JAVASCRIPT)
    deployments = _fixtures()["deployments_active.json"]

    assert "data-cluster-v2-active" in template
    assert "data-cluster-v2-active-device" in template
    assert "data-cluster-v2-deactivate" in template
    assert "deactivateDeployment" in javascript
    assert re.search(
        r"`/admin/api/cluster/deployments/\$\{encodeURIComponent\(id\)\}`",
        javascript,
    )
    assert deployments["deployments"][0]["deployment_id"]


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
