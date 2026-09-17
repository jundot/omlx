# SPDX-License-Identifier: Apache-2.0
"""Offline tests for the v2 discovery/identity endpoints."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.cluster import discovery_routes
from omlx.cluster.discovery import (
    DiscoveryConfig,
    DiscoveryService,
    PeerCaps,
    PeerRecord,
    configure_discovery_service,
)
from omlx.cluster.identity import (
    NodeIdentity,
    configure_node_identity,
    reset_configured_identity,
)
from omlx.cluster.registry import (
    DeviceRegistry,
    configure_device_registry,
    reset_configured_device_registry,
)


@pytest.fixture(autouse=True)
def _configured_stores(tmp_path):
    reset_configured_identity()
    reset_configured_device_registry()
    configure_discovery_service(None)
    discovery_routes.probe_rate_limiter._buckets.clear()
    identity = configure_node_identity(tmp_path)
    registry = configure_device_registry(tmp_path / "devices.json")
    # Bypass admin auth for the inventory endpoint in these unit tests;
    # admin wiring is exercised by the existing admin auth test suite.
    app = FastAPI()

    async def _allow():
        return True

    app.dependency_overrides[discovery_routes.require_admin] = _allow
    app.include_router(discovery_routes.discovery_router)
    client = TestClient(app)
    yield identity, registry, client
    reset_configured_identity()
    reset_configured_device_registry()
    configure_discovery_service(None)


def test_node_id_probe_is_public_and_returns_identity(_configured_stores):
    identity, _, client = _configured_stores

    response = client.get("/api/cluster/node_id")

    assert response.status_code == 200
    payload = response.json()
    assert payload["node_id"] == identity.node_id
    assert payload["version"]
    assert payload["cluster_name"] == "omlx"  # default with no service
    # The probe must not leak capabilities or the device inventory.
    assert "caps" not in payload
    assert "devices" not in payload


def test_node_id_probe_is_rate_limited(_configured_stores):
    _, _, client = _configured_stores
    limiter = discovery_routes.probe_rate_limiter
    original = (limiter._rate, limiter._burst)
    limiter._rate, limiter._burst = 0.0, 3
    try:
        codes = [client.get("/api/cluster/node_id").status_code for _ in range(5)]
    finally:
        limiter._rate, limiter._burst = original
    assert codes[:3] == [200, 200, 200]
    assert codes[3:] == [429, 429]


def test_devices_returns_inventory_with_multicast_signal(_configured_stores):
    identity, registry, client = _configured_stores
    registry.mark_paired("peer-1", friendly_name="studio-b", paired_at=10.0)

    response = client.get("/api/cluster/devices")

    assert response.status_code == 200
    payload = response.json()
    assert payload["self"]["node_id"] == identity.node_id
    assert payload["self"]["paired"] is True
    assert payload["paired"][0]["node_id"] == "peer-1"
    assert payload["discovered"] == []
    # No discovery service running: multicast_ok must be False so the UI
    # shows the Local Network permission guidance instead of silent zeros.
    assert payload["multicast_ok"] is False
    assert payload["mdns_available"] is False


def test_devices_paired_rows_carry_paired_flag(_configured_stores):
    """Wizard consumers key on device.paired (the per-Mac role list filters
    on it, and the device card's Pair/Paired state derives from it), so
    rows in the paired list must carry the flag explicitly -- list
    membership alone is not visible to those consumers."""
    _, registry, client = _configured_stores
    registry.mark_paired("peer-1", friendly_name="studio-b", paired_at=10.0)

    payload = client.get("/api/cluster/devices").json()

    assert payload["paired"][0]["paired"] is True


def test_devices_reflects_live_discovery_service(_configured_stores, tmp_path):
    identity, registry, client = _configured_stores

    service = DiscoveryService(
        identity,
        registry,
        DiscoveryConfig(cluster_name="home", http_port=8000),
        prober=lambda ip, port, timeout: None,
        interface_lister=lambda: [],
        zeroconf_module=None,
    )
    configure_discovery_service(service)
    peer = PeerRecord(node_id="peer-9", friendly_name="nine", http_port=8000)
    peer.last_seen = service._clock()
    service._peers["peer-9"] = peer
    service._last_hello_at = service._clock()

    payload = client.get("/api/cluster/devices").json()

    assert payload["self"]["cluster_name"] == "home"
    assert payload["self"]["http_port"] == 8000
    assert payload["multicast_ok"] is True
    assert any(d["node_id"] == "peer-9" for d in payload["discovered"])
    # Registry merge happened through the service path? No — direct insertion
    # here; the discovered list is sourced from service.peers().
    configure_discovery_service(None)


def test_devices_excludes_paired_peers_from_discovered(_configured_stores, tmp_path):
    identity, registry, client = _configured_stores
    registry.mark_paired("peer-1", friendly_name="studio-b")

    service = DiscoveryService(
        identity,
        registry,
        DiscoveryConfig(),
        prober=lambda ip, port, timeout: None,
        interface_lister=lambda: [],
        zeroconf_module=None,
    )
    configure_discovery_service(service)
    peer = PeerRecord(node_id="peer-1", friendly_name="studio-b")
    peer.paired = True
    service._peers["peer-1"] = peer

    payload = client.get("/api/cluster/devices").json()

    assert payload["discovered"] == []
    assert payload["paired"][0]["node_id"] == "peer-1"
    configure_discovery_service(None)


def test_discovery_health_without_service(_configured_stores):
    _, _, client = _configured_stores

    payload = client.get("/api/cluster/discovery/health").json()

    assert payload == {
        "multicast_rx_within_5s": False,
        "last_multicast_rx_at": None,
        "mdns_active": False,
        "transport": "disabled",
    }


def test_discovery_health_reflects_live_service(_configured_stores):
    identity, registry, client = _configured_stores
    service = DiscoveryService(
        identity,
        registry,
        DiscoveryConfig(cluster_name="home", http_port=8000),
        prober=lambda ip, port, timeout: None,
        interface_lister=lambda: [],
        zeroconf_module=None,
    )
    configure_discovery_service(service)
    service._last_hello_at = service._clock()
    service._last_hello_wall = 1782000002.5

    payload = client.get("/api/cluster/discovery/health").json()

    assert payload["multicast_rx_within_5s"] is True
    assert payload["last_multicast_rx_at"] == 1782000002.5
    assert payload["mdns_active"] is False  # zeroconf disabled in tests
    assert payload["transport"] == "multicast"
    configure_discovery_service(None)


def test_devices_merges_pending_pairing_requests(_configured_stores, monkeypatch):
    identity, registry, client = _configured_stores

    class _FakePairingManager:
        def pending_requests(self):
            return [
                {
                    "node_id": "peer-pending",
                    "friendly_name": "studio-2",
                    "caps": {"chip": "M3"},
                    "addrs": ["192.168.1.11"],
                    "http_port": 8000,
                    "state": "awaiting_approval",
                    "created_at": 100.0,
                    "expires_at": 700.0,
                    "attempts": 0,
                    "locked": False,
                    "locked_until": None,
                }
            ]

    from omlx.cluster import pairing

    monkeypatch.setattr(pairing, "get_pairing_manager", lambda: _FakePairingManager())

    payload = client.get("/api/cluster/devices").json()

    pending = [d for d in payload["discovered"] if d["node_id"] == "peer-pending"]
    assert len(pending) == 1
    assert pending[0]["state"] == "awaiting_approval"
    assert pending[0]["friendly_name"] == "studio-2"


def test_devices_paired_rows_carry_enrolled_ssh_target(_configured_stores, monkeypatch):
    from types import SimpleNamespace

    from omlx.cluster import enrollment

    _, registry, client = _configured_stores
    registry.mark_paired("peer-1", friendly_name="studio-b")
    enrolled = SimpleNamespace(node_id="peer-1", ssh="omlx@studio-b.local")
    fake_store = SimpleNamespace(list_nodes=lambda: (enrolled,))
    monkeypatch.setattr(enrollment, "get_cluster_enrollment", lambda: fake_store)

    payload = client.get("/api/cluster/devices").json()

    assert payload["paired"][0]["ssh_target"] == "omlx@studio-b.local"


def test_cluster_name_persisted_config(tmp_path, monkeypatch):
    from omlx.cluster.discovery import default_cluster_config_path, load_cluster_name

    assert load_cluster_name(tmp_path) == "omlx"  # no file yet
    path = default_cluster_config_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"cluster_name": "studio-lan"}')
    assert load_cluster_name(tmp_path) == "studio-lan"
    # Malformed JSON falls back to the default instead of raising.
    path.write_text("{not json")
    assert load_cluster_name(tmp_path) == "omlx"


def test_devices_requires_admin_auth(tmp_path):
    # Without the dependency override, an unauthenticated call is rejected.
    reset_configured_identity()
    configure_node_identity(tmp_path)
    app = FastAPI()
    app.include_router(discovery_routes.discovery_router)
    client = TestClient(app, raise_server_exceptions=False)
    try:
        response = client.get("/api/cluster/devices")
        assert response.status_code in {401, 302, 307}
    finally:
        reset_configured_identity()


def test_node_id_probe_503_without_identity():
    reset_configured_identity()
    discovery_routes.probe_rate_limiter._buckets.clear()
    app = FastAPI()
    app.include_router(discovery_routes.discovery_router)
    client = TestClient(app)
    try:
        response = client.get("/api/cluster/node_id")
        assert response.status_code == 503
    finally:
        pass  # fixture resets on next test


# -- manual peer add + health detail ------------------------------------------


def _live_service(registry, prober=None):
    service = DiscoveryService(
        NodeIdentity(node_id="self-node", friendly_name="self", created_at=1.0),
        registry,
        DiscoveryConfig(cluster_name="omlx", http_port=8000),
        prober=prober or (lambda ip, port, timeout: None),
        interface_lister=lambda: [],
        zeroconf_module=None,
    )
    configure_discovery_service(service)
    return service


def test_manual_peer_add_verifies_and_returns_peer(_configured_stores):
    _, registry, client = _configured_stores

    def prober(ip, port, timeout):
        if ip == "10.0.0.2":
            return {"node_id": "tb-peer", "friendly_name": "m5-max"}
        return None

    service = _live_service(registry, prober)

    response = client.post("/api/cluster/devices/manual", json={"ip": "10.0.0.2"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["verified"] is True
    assert payload["peer"]["node_id"] == "tb-peer"
    assert payload["peer"]["friendly_name"] == "m5-max"
    assert payload["peer"]["addrs"] == [{"ip": "10.0.0.2", "if_type": "manual"}]
    service.stop()


def test_manual_peer_add_unverified_when_address_silent(_configured_stores):
    _, registry, client = _configured_stores
    service = _live_service(registry)  # prober answers None for everything

    response = client.post(
        "/api/cluster/devices/manual", json={"ip": "10.0.0.99", "port": 8000}
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["verified"] is False
    assert payload["peer"] is None
    service.stop()


def test_manual_peer_add_rejects_bad_input(_configured_stores):
    _, registry, client = _configured_stores
    _live_service(registry)

    assert (
        client.post("/api/cluster/devices/manual", json={"ip": "not-an-ip"}).status_code
        == 400
    )
    assert (
        client.post(
            "/api/cluster/devices/manual",
            json={"ip": "10.0.0.2", "port": 70000},
        ).status_code
        == 400
    )
    assert client.post("/api/cluster/devices/manual", json={}).status_code == 400


def test_manual_peer_add_503_when_discovery_disabled(_configured_stores):
    _, _, client = _configured_stores  # fixture leaves service unset

    response = client.post("/api/cluster/devices/manual", json={"ip": "10.0.0.2"})

    assert response.status_code == 503


def test_health_detail_reports_loop_state(_configured_stores):
    _, registry, client = _configured_stores
    _live_service(registry)

    response = client.get("/api/cluster/discovery/health/detail")

    assert response.status_code == 200
    payload = response.json()
    assert payload["multicast_loop_alive"] is False  # never started
    assert payload["maintenance_loop_alive"] is False
    assert payload["socket_open"] is False
    assert payload["joined_interfaces"] == []
    assert payload["peers"] == 0


def test_health_detail_503_when_discovery_disabled(_configured_stores):
    _, _, client = _configured_stores

    assert client.get("/api/cluster/discovery/health/detail").status_code == 503


def test_devices_self_row_lists_local_addresses(_configured_stores):
    _, _, client = _configured_stores

    response = client.get("/api/cluster/devices")

    assert response.status_code == 200
    addrs = response.json()["self"]["addrs"]
    assert isinstance(addrs, list)
    for entry in addrs:
        assert set(entry) == {"ip", "if_type"}


# -- paired suppression + paired-row enrichment ---------------------------------


def test_devices_suppresses_paired_node_with_stale_dead_peer(
    _configured_stores,
):
    identity, registry, client = _configured_stores
    registry.mark_paired("peer-1", friendly_name="studio-b")

    service = DiscoveryService(
        identity,
        registry,
        DiscoveryConfig(),
        prober=lambda ip, port, timeout: None,
        interface_lister=lambda: [],
        zeroconf_module=None,
    )
    configure_discovery_service(service)
    # Stale in-memory record: the peer went silent before pairing completed,
    # so nothing ever flipped PeerRecord.paired on this dead row.
    service._peers["peer-1"] = PeerRecord(
        node_id="peer-1", friendly_name="studio-b", state="dead"
    )

    payload = client.get("/api/cluster/devices").json()

    assert payload["discovered"] == []
    assert [row["node_id"] for row in payload["paired"]] == ["peer-1"]
    configure_discovery_service(None)


def test_devices_pending_request_wins_over_stale_discovered_record(
    _configured_stores, monkeypatch
):
    identity, registry, client = _configured_stores

    service = DiscoveryService(
        identity,
        registry,
        DiscoveryConfig(),
        prober=lambda ip, port, timeout: None,
        interface_lister=lambda: [],
        zeroconf_module=None,
    )
    configure_discovery_service(service)
    service._peers["peer-pending"] = PeerRecord(
        node_id="peer-pending", friendly_name="stale-name", state="dead"
    )

    class _FakePairingManager:
        def pending_requests(self):
            return [
                {
                    "node_id": "peer-pending",
                    "friendly_name": "studio-2",
                    "caps": {"chip": "M3"},
                    "addrs": ["192.168.1.11"],
                    "http_port": 8000,
                    "state": "awaiting_approval",
                    "created_at": 100.0,
                    "expires_at": 700.0,
                    "attempts": 0,
                    "locked": False,
                    "locked_until": None,
                }
            ]

    from omlx.cluster import pairing

    monkeypatch.setattr(pairing, "get_pairing_manager", lambda: _FakePairingManager())

    payload = client.get("/api/cluster/devices").json()

    pending = [d for d in payload["discovered"] if d["node_id"] == "peer-pending"]
    assert len(pending) == 1
    assert pending[0]["state"] == "awaiting_approval"
    assert pending[0]["friendly_name"] == "studio-2"
    configure_discovery_service(None)


def test_devices_enriches_paired_row_from_discovery_record(_configured_stores):
    identity, registry, client = _configured_stores
    # Paired before caps were exchanged: empty caps persisted on disk.
    registry.mark_paired(
        "peer-1",
        friendly_name="studio-b",
        caps={},
        addrs=["192.168.1.20"],
        paired_at=10.0,
    )

    service = DiscoveryService(
        identity,
        registry,
        DiscoveryConfig(),
        prober=lambda ip, port, timeout: None,
        interface_lister=lambda: [],
        zeroconf_module=None,
    )
    configure_discovery_service(service)
    peer = PeerRecord(
        node_id="peer-1",
        friendly_name="studio-b",
        version="1.2.3",
        caps=PeerCaps(
            chip="M3 Max",
            ram_gb=96.0,
            backends=["jaccl"],
            thunderbolt=True,
            jaccl=True,
        ),
        addrs=[
            {"ip": "192.168.1.20", "if_type": "lan"},
            {"ip": "192.168.1.21", "if_type": "tb"},
        ],
        http_port=8000,
        paired=True,
        last_seen=1234.5,
        state="dead",
    )
    peer.link = "tb"
    service._peers["peer-1"] = peer

    payload = client.get("/api/cluster/devices").json()

    assert payload["discovered"] == []  # suppressed, but still enriches
    row = payload["paired"][0]
    assert row["caps"] == {
        "chip": "M3 Max",
        "ram_gb": 96.0,
        "backends": ["jaccl"],
        "thunderbolt": True,
        "jaccl": True,
    }
    assert row["version"] == "1.2.3"
    assert row["link"] == "tb"
    assert row["state"] == "dead"
    assert row["last_seen"] == 1234.5
    assert row["http_port"] == 8000
    # Normalized addrs: last_addrs entries come first with the "paired"
    # marker; the discovered record upgrades if_type where the IP matches
    # and appends addresses the pairing flow never saw.
    assert row["addrs"] == [
        {"ip": "192.168.1.20", "if_type": "lan"},
        {"ip": "192.168.1.21", "if_type": "tb"},
    ]
    configure_discovery_service(None)


def test_devices_paired_row_normalizes_last_addrs_and_keeps_ssh_target(
    _configured_stores, monkeypatch
):
    from types import SimpleNamespace

    from omlx.cluster import enrollment

    _, registry, client = _configured_stores
    registry.mark_paired(
        "peer-1",
        friendly_name="studio-b",
        caps={"chip": "M4", "ram_gb": 64},
        addrs=["192.168.1.20", "192.168.1.21"],
    )
    enrolled = SimpleNamespace(node_id="peer-1", ssh="omlx@studio-b.local")
    fake_store = SimpleNamespace(list_nodes=lambda: (enrolled,))
    monkeypatch.setattr(enrollment, "get_cluster_enrollment", lambda: fake_store)

    payload = client.get("/api/cluster/devices").json()

    row = payload["paired"][0]
    # No discovery record: caps stay as persisted and nothing is invented.
    assert row["caps"] == {"chip": "M4", "ram_gb": 64}
    assert row["addrs"] == [
        {"ip": "192.168.1.20", "if_type": "paired"},
        {"ip": "192.168.1.21", "if_type": "paired"},
    ]
    assert "version" not in row
    assert row["state"] == "suspect"
    assert row["last_seen"] is None
    assert row["http_port"] == 0
    # The enrollment seam still applies on top of the normalization.
    assert row["ssh_target"] == "omlx@studio-b.local"


def test_manual_paired_endpoint_persists_and_rehydrates_on_reboot(
    _configured_stores,
):
    identity, registry, client = _configured_stores
    registry.mark_paired("tb-peer", friendly_name="m5-max", paired_at=1.0)

    def prober(ip, port, timeout):
        if (ip, port) == ("10.0.0.2", 9123):
            return {
                "node_id": "tb-peer",
                "friendly_name": "m5-max",
                "version": "0.6.1",
                "cluster_name": "omlx",
            }
        return None

    service = _live_service(registry, prober)
    response = client.post(
        "/api/cluster/devices/manual",
        json={"ip": "10.0.0.2", "port": 9123},
    )

    assert response.status_code == 200
    assert response.json()["verified"] is True
    persisted = DeviceRegistry(registry.path)
    record = persisted.get("tb-peer")
    assert record["last_addrs"] == ["10.0.0.2"]
    assert record["http_port"] == 9123

    row = client.get("/api/cluster/devices").json()["paired"][0]
    assert row["state"] == "discovered"
    assert row["last_seen"] is not None
    assert row["http_port"] == 9123

    rebooted = DiscoveryService(
        identity,
        persisted,
        DiscoveryConfig(cluster_name="omlx", http_port=8000),
        prober=prober,
        interface_lister=lambda: [],
        zeroconf_module=None,
    )
    assert ("10.0.0.2", 9123) in rebooted._candidates
    assert rebooted._candidates[("10.0.0.2", 9123)]["node_id"] == "tb-peer"
    service.stop()
