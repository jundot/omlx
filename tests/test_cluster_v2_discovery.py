# SPDX-License-Identifier: Apache-2.0
"""Offline tests for the v2 discovery service.

No real multicast, mDNS, or HTTP happens here: sockets, the probe function,
the clock, interface listing, and tailscale status are all injected.
"""

import errno
import io
import json
import logging
import socket
import struct
import time

import pytest

from omlx.cluster.discovery import (
    MULTICAST_GROUP,
    MULTICAST_PORT,
    _TX_FAIL_RESET_ROUNDS,
    DiscoveryConfig,
    DiscoveryService,
    PeerCaps,
    PeerRecord,
    _classify_link,
    _http_probe_node_id,
    _system_proxy_probe_node_id,
    _tailscale_executable,
    cluster_hash_u64,
    decode_hello,
    decode_wassup,
    encode_hello,
    encode_wassup,
)
from omlx.cluster.identity import NodeIdentity
from omlx.cluster.registry import DeviceRegistry


class FakeHTTPStream:
    def __init__(self, response: bytes) -> None:
        self.response = response
        self.sent = b""

    def sendall(self, payload: bytes) -> None:
        self.sent += payload

    def makefile(self, *_args, **_kwargs):
        return io.BytesIO(self.response)


class FakeSystemProxy:
    def __init__(self, response: bytes) -> None:
        self.stream = FakeHTTPStream(response)
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeSocket:
    """Duck-typed stand-in for the UDP multicast socket."""

    def __init__(self) -> None:
        self.sent: list[tuple[bytes, tuple]] = []
        self.inbox: list[tuple[bytes, tuple]] = []
        self.closed = False
        self.joined: list[bytes] = []
        self.opts: list[tuple] = []

    def setsockopt(self, *args):
        self.opts.append(args)
        if len(args) == 3 and args[1] == socket.IPV6_JOIN_GROUP:
            self.joined.append(args[2])

    def bind(self, addr):
        pass

    def settimeout(self, value):
        pass

    def sendto(self, data, addr):
        self.sent.append((bytes(data), tuple(addr)))

    def recvfrom(self, size):
        if not self.inbox:
            raise socket.timeout()
        return self.inbox.pop(0)

    def close(self):
        self.closed = True


def _identity(node_id: str, name: str = "node") -> NodeIdentity:
    return NodeIdentity(node_id=node_id, friendly_name=name, created_at=1.0)


def _service(
    node_id: str = "aaaa-node",
    *,
    clock: FakeClock | None = None,
    prober=None,
    socket_factory=None,
    registry=None,
    tailscale_status=None,
    zeroconf_module=None,
    interface_lister=None,
    config: DiscoveryConfig | None = None,
):
    clock = clock or FakeClock()
    cfg = config or DiscoveryConfig(cluster_name="omlx", http_port=8000)
    service = DiscoveryService(
        _identity(node_id),
        registry if registry is not None else DeviceRegistry("/nonexistent/dir/devices.json"),
        cfg,
        socket_factory=socket_factory,
        prober=prober or (lambda ip, port, timeout: None),
        clock=clock,
        interface_lister=interface_lister or (lambda: ["en0"]),
        tailscale_status=tailscale_status,
        zeroconf_module=zeroconf_module,
    )
    return service, clock


# -- codec --------------------------------------------------------------------


def test_hello_codec_roundtrip():
    payload = encode_hello(0xDEADBEEF, cluster_hash_u64("omlx"))
    assert decode_hello(payload) == (0xDEADBEEF, cluster_hash_u64("omlx"))


def test_hello_codec_rejects_garbage():
    assert decode_hello(b"") is None
    assert decode_hello(b"OMLX") is None
    assert decode_hello(b"NOPE" + b"\x00" * 16) is None
    assert decode_hello(encode_hello(1, 2) + b"extra") is None


def test_wassup_codec_roundtrip():
    payload = encode_wassup(7, "node-1", 8000)
    assert decode_wassup(payload) == {
        "nonce": 7,
        "node_id": "node-1",
        "http_port": 8000,
    }


def test_wassup_codec_rejects_garbage():
    assert decode_wassup(b"") is None
    assert decode_wassup(b"OMLXW{not json") is None
    assert decode_wassup(b"OMLXW" + json.dumps({"nonce": -1}).encode()) is None
    assert (
        decode_wassup(
            b"OMLXW"
            + json.dumps({"nonce": 1, "node_id": "x", "http_port": 0}).encode()
        )
        is None
    )


def test_cluster_hash_is_blake2s_prefix():
    import hashlib

    expected = int.from_bytes(
        hashlib.blake2s(b"omlx").digest()[:8], "big"
    )
    assert cluster_hash_u64("omlx") == expected
    assert cluster_hash_u64("omlx") != cluster_hash_u64("other")


# -- HELLO handling -----------------------------------------------------------


def test_hello_from_same_cluster_sets_multicast_ok_and_replies_wassup():
    sock = FakeSocket()
    service, clock = _service(socket_factory=lambda: sock)
    assert service.multicast_ok is False

    service._handle_hello(123, service._cluster_hash, ("fe80::99", 53413), sock)

    assert service.multicast_ok is True
    assert len(sock.sent) == 1
    reply = decode_wassup(sock.sent[0][1] and sock.sent[0][0])
    assert reply == {
        "nonce": 123,
        "node_id": "aaaa-node",
        "http_port": 8000,
    }
    assert sock.sent[0][1] == ("fe80::99", 53413)


def test_multicast_ok_expires_after_window():
    service, clock = _service()
    service._handle_hello(1, service._cluster_hash, ("fe80::99", 53413), None)
    assert service.multicast_ok is True
    clock.advance(4.9)
    assert service.multicast_ok is True
    clock.advance(0.2)
    assert service.multicast_ok is False


def test_hello_with_foreign_cluster_hash_is_ignored_silently():
    sock = FakeSocket()
    service, _ = _service(socket_factory=lambda: sock)

    service._handle_hello(1, cluster_hash_u64("someone-else"), ("fe80::99", 53413), sock)

    assert sock.sent == []
    assert service.peers() == []
    assert service.multicast_ok is False


def test_own_nonce_echo_is_dropped():
    sock = FakeSocket()
    service, _ = _service(socket_factory=lambda: sock)
    service._nonces.append(555)  # simulate a HELLO we sent

    service._handle_hello(555, service._cluster_hash, ("fe80::1", 53413), sock)

    assert sock.sent == []
    assert service.multicast_ok is False


# -- WASSUP handling ----------------------------------------------------------


def _announce_nonce(service) -> int:
    service._nonces.append(42)
    return 42


def test_wassup_creates_peer_and_candidate():
    service, _ = _service("aaaa-node")  # lower id than peer
    _announce_nonce(service)

    service._handle_wassup(
        {"nonce": 42, "node_id": "bbbb-node", "http_port": 8000},
        ("fe80::99", 53413),
    )

    peers = service.peers()
    assert [p.node_id for p in peers] == ["bbbb-node"]
    assert peers[0].http_port == 8000
    assert peers[0].addrs == [{"ip": "fe80::99", "if_type": "unknown"}]
    assert ("fe80::99", 8000) in service._candidates


def test_wassup_with_unknown_nonce_is_ignored():
    service, _ = _service()
    service._handle_wassup(
        {"nonce": 999, "node_id": "bbbb-node", "http_port": 8000},
        ("fe80::99", 53413),
    )
    assert service.peers() == []


def test_higher_node_id_probes_immediately_lower_defers():
    calls: list[tuple[str, int]] = []

    def prober(ip, port, timeout):
        calls.append((ip, port))
        return {"node_id": "aaaa-node", "version": "0.6.1", "cluster_name": "omlx"}

    # We are zzzz-node: higher than aaaa-node → we initiate contact.
    service, _ = _service("zzzz-node", prober=prober)
    _announce_nonce(service)
    service._handle_wassup(
        {"nonce": 42, "node_id": "aaaa-node", "http_port": 8000},
        ("fe80::99", 53413),
    )
    assert calls == [("fe80::99", 8000)]

    # We are aaaa-node: lower → no immediate probe.
    calls.clear()
    service2, _ = _service("aaaa-node", prober=prober)
    _announce_nonce(service2)
    service2._handle_wassup(
        {"nonce": 42, "node_id": "zzzz-node", "http_port": 8000},
        ("fe80::99", 53413),
    )
    assert calls == []


def test_wassup_dedupes_repeated_announcements():
    service, _ = _service("aaaa-node")
    _announce_nonce(service)
    for _ in range(3):
        service._handle_wassup(
            {"nonce": 42, "node_id": "bbbb-node", "http_port": 8000},
            ("fe80::99", 53413),
        )
    peers = service.peers()
    assert len(peers) == 1
    assert peers[0].addrs == [{"ip": "fe80::99", "if_type": "unknown"}]


# -- verification probe --------------------------------------------------------


def test_successful_probe_fills_peer_details_and_link():
    service, _ = _service("aaaa-node", prober=lambda ip, port, timeout: {
        "node_id": "bbbb-node",
        "version": "0.6.1",
        "cluster_name": "omlx",
    })
    service.add_manual("10.0.0.5", 8000)
    service.probe_now()

    peer = service.peers()[0]
    assert peer.node_id == "bbbb-node"
    assert peer.version == "0.6.1"
    assert peer.state == "discovered"
    assert peer.link in {"tb", "ethernet", "wifi"}  # RTT-classified


def test_probe_node_id_mismatch_drops_address():
    service, _ = _service("zzzz-node", prober=lambda ip, port, timeout: {
        "node_id": "impostor",
        "version": "0.6.1",
        "cluster_name": "omlx",
    })
    _announce_nonce(service)
    service._handle_wassup(
        {"nonce": 42, "node_id": "bbbb-node", "http_port": 8000},
        ("fe80::99", 53413),
    )

    # The higher-id immediate probe already ran and found a mismatch.
    peer = service.peers()[0]
    assert peer.addrs == []
    assert ("fe80::99", 8000) not in service._candidates


def test_failed_probe_keeps_candidate_unverified_for_retry():
    clock = FakeClock()
    service, _ = _service(clock=clock)  # prober default: None (unreachable)
    service.add_manual("10.0.0.5", 8000)
    service.probe_now()
    assert service.peers() == []
    assert ("10.0.0.5", 8000) in service._candidates
    assert service._candidates[("10.0.0.5", 8000)]["verified"] is False


def test_http_probe_falls_back_to_system_python_carrier(monkeypatch):
    from omlx.cluster import discovery

    monkeypatch.setattr(
        discovery.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("macOS denied direct subnet")
        ),
    )
    expected = {
        "node_id": "peer-node",
        "version": "0.6.4.dev1",
        "cluster_name": "omlx",
    }
    calls = []
    monkeypatch.setattr(
        discovery,
        "_system_proxy_probe_node_id",
        lambda ip, port, timeout: calls.append((ip, port, timeout)) or expected,
    )

    assert _http_probe_node_id("10.0.0.1", 8000, 3.0) == expected
    assert calls == [("10.0.0.1", 8000, 3.0)]


def test_tailscale_probe_failure_does_not_spawn_direct_subnet_proxy(monkeypatch):
    from omlx.cluster import discovery

    monkeypatch.setattr(
        discovery.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("connection refused")
        ),
    )
    monkeypatch.setattr(
        discovery,
        "_system_proxy_probe_node_id",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("Tailscale must stay on its direct userspace route")
        ),
    )

    assert _http_probe_node_id("100.64.0.2", 8000, 3.0) is None


def test_system_python_carrier_parses_bounded_node_probe(monkeypatch):
    from omlx.cluster import system_socket_proxy

    body = json.dumps(
        {
            "node_id": "peer-node",
            "version": "0.6.4.dev1",
            "cluster_name": "omlx",
        }
    ).encode()
    response = (
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
        + body
    )
    proxy = FakeSystemProxy(response)
    monkeypatch.setattr(
        system_socket_proxy,
        "should_proxy_control_socket",
        lambda _host: True,
    )
    monkeypatch.setattr(
        system_socket_proxy,
        "open_system_tcp_proxy",
        lambda host, port, *, timeout: proxy,
    )

    payload = _system_proxy_probe_node_id("10.0.0.1", 8000, 3.0)

    assert payload is not None and payload["node_id"] == "peer-node"
    assert proxy.stream.sent.startswith(b"GET /api/cluster/node_id HTTP/1.1\r\n")
    assert proxy.closed is True


def test_paired_manual_address_and_port_rehydrate_after_reboot(tmp_path):
    path = tmp_path / "devices.json"
    registry = DeviceRegistry(path)
    registry.mark_paired(
        "bbbb-node",
        friendly_name="studio-b",
        addrs=["10.0.0.5", "fe80::1", "not-an-ip"],
        http_port=9123,
        paired_at=1.0,
    )
    restored = DeviceRegistry(path)
    calls = []

    service, _ = _service(
        "aaaa-node",
        registry=restored,
        prober=lambda ip, port, timeout: calls.append((ip, port))
        or {
            "node_id": "bbbb-node",
            "friendly_name": "studio-b",
            "version": "0.6.1",
            "cluster_name": "omlx",
        },
    )

    assert ("10.0.0.5", 9123) in service._candidates
    assert service._candidates[("10.0.0.5", 9123)]["node_id"] == "bbbb-node"
    assert service._candidates[("10.0.0.5", 9123)]["if_type"] == "paired"
    assert all(ip != "not-an-ip" for ip, _ in service._candidates)
    assert all(ip != "fe80::1" for ip, _ in service._candidates)

    service.probe_now()

    assert calls == [("10.0.0.5", 9123)]
    [peer] = service.peers()
    assert peer.node_id == "bbbb-node"
    assert peer.http_port == 9123
    assert peer.paired is True
    assert peer.state == "discovered"


def test_legacy_paired_address_rehydrates_on_the_configured_default_port(tmp_path):
    registry = DeviceRegistry(tmp_path / "devices.json")
    registry.mark_paired(
        "bbbb-node",
        friendly_name="studio-b",
        addrs=["10.0.0.8"],
        paired_at=1.0,
    )

    service, _ = _service(
        "aaaa-node",
        registry=DeviceRegistry(registry.path),
        config=DiscoveryConfig(cluster_name="omlx", http_port=8765),
    )

    assert ("10.0.0.8", 8765) in service._candidates
    assert service._candidates[("10.0.0.8", 8765)]["node_id"] == "bbbb-node"


def test_probe_sweep_respects_interval():
    calls: list[tuple[str, int]] = []
    clock = FakeClock()
    service, _ = _service(
        clock=clock,
        prober=lambda ip, port, timeout: calls.append((ip, port)) or None,
    )
    service.add_manual("10.0.0.5", 8000)
    service.probe_now()
    service.probe_now()  # too soon: deduped by probe interval
    assert calls == [("10.0.0.5", 8000)]
    clock.advance(10.0)
    service.probe_now()
    assert calls == [("10.0.0.5", 8000)] * 2


def test_verified_candidate_uses_heartbeat_cadence():
    calls: list[tuple[str, int]] = []
    clock = FakeClock()
    service, _ = _service(
        clock=clock,
        prober=lambda ip, port, timeout: calls.append((ip, port))
        or {
            "node_id": "peer-node",
            "version": "0.6.4.dev1",
            "cluster_name": "omlx",
        },
    )
    service.add_manual("10.0.0.5", 8000)
    service.probe_now()
    clock.advance(service.config.heartbeat_interval)
    service.probe_now()

    assert calls == [("10.0.0.5", 8000)] * 2
    assert service.peers()[0].state == "discovered"


def test_explicit_candidate_probe_bypasses_periodic_dedupe():
    calls: list[tuple[str, int]] = []
    service, _ = _service(
        prober=lambda ip, port, timeout: calls.append((ip, port)) or None,
    )
    service.add_manual("10.0.0.5", 8000)

    service.probe_candidate_now("10.0.0.5", 8000)
    service.probe_candidate_now("10.0.0.5", 8000)

    assert calls == [("10.0.0.5", 8000)] * 2
    [state] = service.health()["candidate_states"]
    assert state["last_transport"] == "custom"
    assert state["last_error"] == "probe returned no identity"


# -- liveness -------------------------------------------------------------------


def _verified_peer(service, node_id="bbbb-node"):
    _announce_nonce(service)
    service._handle_wassup(
        {"nonce": 42, "node_id": node_id, "http_port": 8000},
        ("fe80::99", 53413),
    )
    return service.peers()[0]


def test_liveness_transitions_discovered_suspect_dead():
    clock = FakeClock()
    service, _ = _service("aaaa-node", clock=clock)
    events: list[tuple[str, str]] = []
    service.on_change(lambda peer: events.append((peer.node_id, peer.state)))

    peer = _verified_peer(service)
    assert peer.state == "discovered"
    assert ("bbbb-node", "discovered") in events  # new-peer event

    clock.advance(6.0)
    service.tick_liveness()
    assert peer.state == "suspect"

    clock.advance(24.0)
    service.tick_liveness()
    assert peer.state == "dead"

    clock.advance(3600)
    service.tick_liveness()
    assert peer.state == "dead"

    states = [state for _, state in events]
    assert states == ["discovered", "suspect", "dead"]


def test_fresh_sign_of_life_revives_dead_peer():
    clock = FakeClock()
    service, _ = _service("aaaa-node", clock=clock)
    peer = _verified_peer(service)
    clock.advance(60.0)
    service.tick_liveness()
    assert peer.state == "dead"

    service._nonces.append(43)
    service._handle_wassup(
        {"nonce": 43, "node_id": "bbbb-node", "http_port": 8000},
        ("fe80::99", 53413),
    )
    assert peer.state == "discovered"


def test_on_change_callback_failure_does_not_kill_service():
    service, _ = _service("aaaa-node")
    service.on_change(lambda peer: 1 / 0)
    peer = _verified_peer(service)  # must not raise
    assert peer.state == "discovered"


# -- registry integration ------------------------------------------------------


def test_discovered_peer_merges_into_registry_unpaired(tmp_path):
    registry = DeviceRegistry(tmp_path / "devices.json")
    service, _ = _service("aaaa-node", registry=registry, prober=lambda *a: {
        "node_id": "bbbb-node",
        "version": "0.6.1",
        "cluster_name": "omlx",
    })
    _verified_peer(service)

    assert registry.discovered()[0]["node_id"] == "bbbb-node"
    assert registry.paired() == []
    assert service.peers()[0].paired is False

    registry.mark_paired("bbbb-node", friendly_name="peer")
    service._merge_registry(service.peers()[0])
    assert service.peers()[0].paired is True


# -- tailscale ------------------------------------------------------------------


def test_tailscale_peers_become_candidates():
    status = {
        "Peer": {
            "k1": {"TailscaleIPs": ["100.64.0.2", "fe80::1"]},
            "k2": {"TailscaleIPs": ["100.64.0.3"]},
        }
    }
    service, _ = _service(tailscale_status=lambda: status)
    service._tailscale_sweep()

    assert ("100.64.0.2", 8000) in service._candidates
    assert ("100.64.0.3", 8000) in service._candidates
    assert service._candidates[("100.64.0.2", 8000)]["if_type"] == "tailscale"
    # Non-tailscale addresses are not picked up from tailscale status.
    assert all(":" not in ip for ip, _ in service._candidates)


def test_tailscale_sweep_ignores_offline_peers():
    service, _ = _service(
        tailscale_status=lambda: {
            "Peer": {
                "online": {
                    "Online": True,
                    "TailscaleIPs": ["100.64.0.2"],
                },
                "offline": {
                    "Online": False,
                    "TailscaleIPs": ["100.64.0.3"],
                },
            }
        }
    )

    service._tailscale_sweep()

    assert ("100.64.0.2", 8000) in service._candidates
    assert ("100.64.0.3", 8000) not in service._candidates


def test_tailscale_absent_is_a_noop():
    service, _ = _service(tailscale_status=lambda: None)
    service._tailscale_sweep()
    assert service._candidates == {}


def test_macos_tailscale_app_binary_is_a_cli_fallback(
    tmp_path, monkeypatch
):
    executable = tmp_path / "Tailscale"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    monkeypatch.setattr("omlx.cluster.discovery.shutil.which", lambda _name: None)
    monkeypatch.setattr("omlx.cluster.discovery.sys.platform", "darwin")
    monkeypatch.setenv("OMLX_TAILSCALE_CLI", str(executable))

    assert _tailscale_executable() == str(executable)


# -- link classification --------------------------------------------------------


def test_link_classification():
    assert _classify_link("100.64.0.2", "unknown", None) == "tailscale"
    assert _classify_link("10.0.0.1", "tailscale", 5.0) == "tailscale"
    assert _classify_link("10.0.0.1", "unknown", None) == "unknown"
    assert _classify_link("fe80::1", "unknown", 0.0005) == "tb"
    assert _classify_link("10.0.0.1", "unknown", 0.005) == "ethernet"
    assert _classify_link("10.0.0.1", "unknown", 0.050) == "wifi"


# -- mDNS -----------------------------------------------------------------------


def test_zeroconf_absence_disables_only_mdns():
    service, _ = _service(zeroconf_module=None)
    assert service.mdns_available is False
    # start()/stop() still work with mDNS unavailable (multicast thread is
    # given a fake socket so nothing real happens).
    sock = FakeSocket()
    service._socket_factory = lambda: sock
    service.start()
    service.stop()
    assert sock.closed


def test_mdns_service_creates_peer_from_txt():
    service, _ = _service("aaaa-node")

    class FakeInfo:
        port = 8000
        properties = {
            b"id": b"bbbb-node",
            b"name": b"studio-b",
            b"ver": b"0.6.1",
            b"cl": b"omlx",
            b"caps": json.dumps({"chip": "M3 Ultra", "ram_gb": 96}).encode(),
        }

        def parsed_addresses(self):
            return ["10.0.0.7"]

    service._handle_mdns_service(FakeInfo())

    peer = service.peers()[0]
    assert peer.node_id == "bbbb-node"
    assert peer.friendly_name == "studio-b"
    assert peer.caps.chip == "M3 Ultra"
    assert peer.addrs == [{"ip": "10.0.0.7", "if_type": "mdns"}]
    assert ("10.0.0.7", 8000) in service._candidates


def test_mdns_foreign_cluster_and_self_are_ignored():
    service, _ = _service("aaaa-node")

    class FakeInfo:
        port = 8000
        properties = {b"id": b"bbbb-node", b"cl": b"other-cluster"}

        def parsed_addresses(self):
            return ["10.0.0.7"]

    service._handle_mdns_service(FakeInfo())
    assert service.peers() == []

    class SelfInfo:
        port = 8000
        properties = {b"id": b"aaaa-node", b"cl": b"omlx"}

        def parsed_addresses(self):
            return ["10.0.0.9"]

    service._handle_mdns_service(SelfInfo())
    assert service.peers() == []


def test_mdns_handler_exception_is_contained():
    service, _ = _service("aaaa-node")

    class BadInfo:
        port = 8000

        @property
        def properties(self):
            raise RuntimeError("boom")

    service._handle_mdns_service(BadInfo())  # must not raise
    assert service.peers() == []


# -- interface joins ------------------------------------------------------------


def test_interface_sync_skips_unsupported_interfaces():
    sock = FakeSocket()
    service, _ = _service(
        socket_factory=lambda: sock,
        # en0 joins fine; gif0 raises EAFNOSUPPORT like loopback-class ifaces
        interface_lister=lambda: ["en0", "gif0"],
    )

    real_setsockopt = sock.setsockopt

    def setsockopt(*args):
        if len(args) == 3 and args[1] == socket.IPV6_JOIN_GROUP:
            ifindex = struct.unpack("@I", args[2][16:])[0]
            if socket.if_indextoname(ifindex) == "gif0":
                raise OSError(47, "Address family not supported")
        real_setsockopt(*args)

    sock.setsockopt = setsockopt
    service._sync_interfaces(sock)

    assert "en0" in service._joined
    assert "gif0" not in service._joined


# -- peer record ----------------------------------------------------------------


def test_peer_record_to_dict_shape():
    record = PeerRecord(
        node_id="n1",
        friendly_name="studio",
        version="0.6.1",
        cluster_name="omlx",
        caps=PeerCaps(chip="M3 Ultra", ram_gb=96.0, backends=["jaccl"], thunderbolt=True, jaccl=True),
        addrs=[{"ip": "fe80::1", "if_type": "mdns"}],
        http_port=8000,
        paired=True,
        last_seen=123.0,
        link="tb",
        state="discovered",
    )
    payload = record.to_dict()
    assert payload["caps"]["jaccl"] is True
    assert payload["link"] == "tb"
    assert payload["state"] == "discovered"
    json.dumps(payload)  # must be JSON-serializable for the API


def test_mark_paired_flips_in_memory_peer_flag():
    service, _ = _service()
    service._peers["peer-1"] = PeerRecord(node_id="peer-1")

    service.mark_paired("peer-1")

    assert service._peers["peer-1"].paired is True
    # Unknown node_ids are a no-op; the flag is advisory, not authoritative.
    service.mark_paired("unknown-node")


# -- announced caps provider (pairing payload seam) ------------------------------


def test_announced_caps_reflects_configured_service():
    from omlx.cluster.discovery import announced_caps, configure_discovery_service

    caps = PeerCaps(
        chip="M3 Max", ram_gb=96.0, backends=["jaccl"], thunderbolt=True, jaccl=True
    )
    service, _ = _service(config=DiscoveryConfig(caps=caps))
    configure_discovery_service(service)
    try:
        assert announced_caps() == caps.to_dict()
    finally:
        configure_discovery_service(None)


def test_announced_caps_falls_back_to_local_snapshot_without_service(monkeypatch):
    from omlx.cluster import discovery

    discovery.configure_discovery_service(None)
    monkeypatch.setattr(
        discovery, "local_caps", lambda: PeerCaps(chip="M-test", ram_gb=32.0)
    )
    assert discovery.announced_caps() == PeerCaps(
        chip="M-test", ram_gb=32.0
    ).to_dict()


def test_announced_caps_degrades_to_empty_dict_on_failure(monkeypatch):
    from omlx.cluster import discovery

    class _BrokenCaps:
        def to_dict(self):
            raise RuntimeError("boom")

    service, _ = _service()
    service.config.caps = _BrokenCaps()
    discovery.configure_discovery_service(service)

    def _boom():
        raise OSError("no sysctl")

    monkeypatch.setattr(discovery, "local_caps", _boom)
    try:
        assert discovery.announced_caps() == {}
    finally:
        discovery.configure_discovery_service(None)


# -- mDNS announce path with a fake zeroconf module -----------------------------


class _FakeZeroconfModule:
    class ServiceInfo:
        def __init__(self, type_, name, addresses, port, properties):
            self.type = type_
            self.name = name
            self.addresses = addresses
            self.port = port
            self.properties = properties

    class ServiceBrowser:
        def __init__(self, zc, type_, listener):
            self.cancelled = False

        def cancel(self):
            self.cancelled = True

    class Zeroconf:
        def __init__(self):
            self.registered = []
            self.closed = False

        def register_service(self, info):
            self.registered.append(info)

        def unregister_service(self, info):
            self.registered.remove(info)

        def close(self):
            self.closed = True


def test_mdns_announce_publishes_spec_txt_record():
    service, _ = _service("aaaa-node", zeroconf_module=_FakeZeroconfModule)
    service.identity.friendly_name = "studio-a"

    service._start_mdns()

    info = service._zc_instance.registered[0]
    assert info.type == "_omlx._tcp.local."
    assert info.name.startswith("studio-a._omlx._tcp.local.")
    assert info.port == 8000
    assert info.properties["id"] == "aaaa-node"
    assert info.properties["name"] == "studio-a"
    assert info.properties["cl"] == "omlx"
    assert info.properties["ver"]
    caps = json.loads(info.properties["caps"])
    assert set(caps) == {"chip", "ram_gb", "backends", "thunderbolt", "jaccl"}
    service.stop()


def test_mdns_start_failure_disables_mdns_without_killing_service():
    class BoomZeroconf(_FakeZeroconfModule):
        class Zeroconf:
            def __init__(self):
                raise RuntimeError("Local Network permission denied")

    service, _ = _service("aaaa-node", zeroconf_module=BoomZeroconf)
    service._socket_factory = FakeSocket
    service.start()  # must not raise
    service.stop()


# -- resilience: scoped sends, stale joins, supervision, self-heal ------------


def test_send_hello_uses_scoped_4tuple_per_interface():
    sock = FakeSocket()
    service, _ = _service(socket_factory=lambda: sock)
    service._joined = {"en5": 20, "en1": 26}

    service._send_hello(sock)

    targets = [addr for _, addr in sock.sent]
    assert (MULTICAST_GROUP, MULTICAST_PORT, 0, 20) in targets
    assert (MULTICAST_GROUP, MULTICAST_PORT, 0, 26) in targets
    # The shared socket's IPV6_MULTICAST_IF must not be mutated per round;
    # the scope id in the destination carries the egress interface instead.
    assert not any(
        len(opt) == 3 and opt[1] == socket.IPV6_MULTICAST_IF
        for opt in sock.opts
    )


def test_send_hello_without_joins_uses_default_route_2tuple():
    sock = FakeSocket()
    service, _ = _service(socket_factory=lambda: sock)

    service._send_hello(sock)

    assert sock.sent[0][1] == (MULTICAST_GROUP, MULTICAST_PORT)


def test_wassup_reply_preserves_link_local_scope_id():
    sock = FakeSocket()
    service, _ = _service(socket_factory=lambda: sock)

    service._handle_hello(
        42, service._cluster_hash, ("fe80::99", 53413, 0, 20), sock
    )

    assert len(sock.sent) == 1
    assert sock.sent[0][1] == ("fe80::99", 53413, 0, 20)


def test_sync_interfaces_rejoins_after_renumber(monkeypatch):
    sock = FakeSocket()
    service, _ = _service(interface_lister=lambda: ["en5"])
    state = {"idx": 20}
    monkeypatch.setattr(
        socket, "if_nametoindex", lambda name: state["idx"]
    )

    service._sync_interfaces(sock)
    assert service._joined == {"en5": 20}
    assert len(sock.joined) == 1

    state["idx"] = 21  # Thunderbolt hotplug renumbered the interface
    service._sync_interfaces(sock)
    assert service._joined == {"en5": 21}
    assert len(sock.joined) == 2  # re-joined under the new index


def test_sync_interfaces_drops_vanished_interfaces(monkeypatch):
    sock = FakeSocket()
    names = ["en5", "en1"]
    service, _ = _service(interface_lister=lambda: names)
    index_of = {"en5": 20, "en1": 26}
    monkeypatch.setattr(
        socket,
        "if_nametoindex",
        lambda name: index_of.get(name) or (_ for _ in ()).throw(OSError()),
    )

    service._sync_interfaces(sock)
    assert set(service._joined) == {"en5", "en1"}

    names.remove("en5")  # cable pulled / bridge torn down
    service._sync_interfaces(sock)
    assert set(service._joined) == {"en1"}


class _FailingSocket(FakeSocket):
    """Every send fails the way macOS reports a dead interface."""

    def sendto(self, data, addr):
        raise OSError(errno.EHOSTUNREACH, "No route to host")


def test_send_failures_are_rate_limited(caplog):
    sock = _FailingSocket()
    service, clock = _service(socket_factory=lambda: sock)
    service._joined = {"en0": 10}

    with caplog.at_level(logging.DEBUG, logger="omlx.cluster.discovery"):
        service._send_hello(sock)
        clock.advance(5)
        service._send_hello(sock)  # inside the 60s window: silent
        assert (
            sum("HELLO send on if" in r.getMessage() for r in caplog.records)
            == 1
        )
        clock.advance(61)
        service._send_hello(sock)
        assert (
            sum("HELLO send on if" in r.getMessage() for r in caplog.records)
            == 2
        )


def test_consecutive_failed_rounds_request_socket_reset():
    sock = _FailingSocket()
    service, clock = _service(socket_factory=lambda: sock)
    service._joined = {"en0": 10}

    for _ in range(_TX_FAIL_RESET_ROUNDS):
        service._send_hello(sock)
        clock.advance(1)

    assert service._needs_socket_reset is True
    assert service._consecutive_tx_fail_rounds >= _TX_FAIL_RESET_ROUNDS
    assert service._last_tx_error


def test_successful_send_clears_fail_state():
    sock = FakeSocket()
    service, _ = _service(socket_factory=lambda: sock)
    service._joined = {"en0": 10}
    service._consecutive_tx_fail_rounds = 5
    service._last_tx_error = "if 10: boom"

    service._send_hello(sock)

    assert service._consecutive_tx_fail_rounds == 0
    assert service._last_tx_error is None
    assert service._last_tx_ok_wall is not None


def test_multicast_loop_rebuilds_wedged_socket():
    socks = [FakeSocket(), FakeSocket()]
    service, clock = _service(socket_factory=lambda: socks.pop(0))
    service._needs_socket_reset = True
    service.config.iface_poll_interval = 0.0
    service.config.hello_interval = 0.0

    service.start()
    try:
        deadline = time.time() + 5
        while time.time() < deadline:
            if (
                not service._needs_socket_reset
                and service._socket is not None
                and service._joined
            ):
                break
            time.sleep(0.02)
        assert service._needs_socket_reset is False
        assert service._socket is not None
        assert service._joined  # re-joined after the rebuild
        health = service.health()
        assert health["multicast_loop_alive"] is True
        assert health["socket_open"] is True
    finally:
        service.stop()


def test_spawned_thread_restarts_after_crash():
    service, _ = _service()
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")
        service._stop.set()

    service._spawn(flaky, "test-flaky")
    deadline = time.time() + 8
    while len(calls) < 2 and time.time() < deadline:
        time.sleep(0.05)
    service.stop()

    assert len(calls) == 2  # crashed once, supervised restart ran it again


def test_health_reports_dead_loops_before_start():
    service, _ = _service()

    health = service.health()

    assert health["multicast_loop_alive"] is False
    assert health["maintenance_loop_alive"] is False
    assert health["socket_open"] is False
    assert health["joined_interfaces"] == []
    assert health["consecutive_tx_fail_rounds"] == 0


def test_blocked_suspicion_flag_lifecycle():
    sock = FakeSocket()
    service, _ = _service(socket_factory=lambda: sock)
    service._joined = {"en0": 10}
    service._local_network_blocked_suspected = True
    service._consecutive_socket_resets = 6

    service._send_hello(sock)  # a successful round ends the suspicion

    assert service._local_network_blocked_suspected is False
    assert service._consecutive_socket_resets == 0

    service._local_network_blocked_suspected = True
    service._handle_hello(7, service._cluster_hash, ("fe80::99", 53413), sock)
    assert service._local_network_blocked_suspected is False


def test_rdma_fabric_caps_detects_jaccl_when_enabled_with_devices():
    from omlx.cluster import discovery

    calls = []

    def runner(args, **kwargs):
        calls.append(args[0])

        class Result:
            returncode = 0
            stdout = (
                "enabled\n"
                if args[0].endswith("rdma_ctl")
                # Real ibv_devices output: two-row header, indented devices.
                else "    device          \t   node GUID\n"
                "    ------          \t----------------\n"
                "    rdma_en6        \t715a4d3d9c48ac05\n"
                "    rdma_en1        \t705a4d3d9c48ac05\n"
            )

        return Result()

    caps = discovery.PeerCaps()
    discovery._rdma_fabric_caps(caps, runner=runner)
    if discovery.sys.platform == "darwin":
        assert caps.thunderbolt is True
        assert caps.jaccl is True


def test_rdma_fabric_caps_stays_false_when_disabled_or_no_devices():
    from omlx.cluster import discovery

    def disabled(args, **kwargs):
        class Result:
            returncode = 0
            stdout = "disabled\n" if args[0].endswith("rdma_ctl") else "rdma_en6\n"

        return Result()

    caps = discovery.PeerCaps()
    discovery._rdma_fabric_caps(caps, runner=disabled)
    if discovery.sys.platform == "darwin":
        assert caps.thunderbolt is True  # fabric exists; RDMA is just off
        assert caps.jaccl is False

    def failing(args, **kwargs):
        class Result:
            returncode = 1
            stdout = ""

        return Result()

    caps = discovery.PeerCaps()
    discovery._rdma_fabric_caps(caps, runner=failing)
    assert caps.thunderbolt is False
    assert caps.jaccl is False
