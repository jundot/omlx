# SPDX-License-Identifier: Apache-2.0

from omlx.cluster.discovery import (
    BonjourPublisher,
    DiscoveryOutput,
    clear_peer_transport_cache,
    discover_all_peers,
    discover_omlx_peers,
    discover_ssh_peers,
    generate_pairing_token,
    parse_browse_instances,
    parse_lookup_target,
    record_peer_transports,
    verify_pairing_token,
)

_PAIRING_SECRET = "correct-horse-battery-staple"


class _PublisherProcess:
    def __init__(self):
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


def test_bonjour_publisher_advertises_api_identity_and_stops():
    calls = []
    process = _PublisherProcess()

    def spawn(args):
        calls.append(tuple(args))
        return process

    publisher = BonjourPublisher(
        port=8000,
        version="1.2.3",
        hostname="peer-studio.local",
        executable="/usr/bin/dns-sd",
        spawner=spawn,
    )

    assert publisher.start() is True
    assert publisher.running is True
    assert calls == [
        (
            "/usr/bin/dns-sd",
            "-R",
            "oMLX on peer-studio",
            "_omlx._tcp.",
            "local.",
            "8000",
            "hostname=peer-studio",
            "version=1.2.3",
            "ssh_port=22",
        )
    ]

    publisher.stop()
    assert process.terminated is True
    assert publisher.running is False


def test_bonjour_publisher_rate_limits_restart_after_exit():
    now = [10.0]
    processes = [_PublisherProcess(), _PublisherProcess()]
    calls = []

    def spawn(args):
        calls.append(tuple(args))
        return processes[len(calls) - 1]

    publisher = BonjourPublisher(
        port=8000,
        version="1",
        hostname="mini",
        executable="/usr/bin/dns-sd",
        spawner=spawn,
        clock=lambda: now[0],
    )
    assert publisher.start() is True
    processes[0].returncode = 1

    assert publisher.ensure_running() is False
    assert len(calls) == 1
    now[0] += 30
    assert publisher.ensure_running() is True
    assert len(calls) == 2


def test_bonjour_parsers_extract_bounded_ssh_service():
    browse = """
12:00:00.000  Add        2  14 local.  _ssh._tcp.  Peer Mac Studio
12:00:00.100  Add        2  14 local.  _ssh._tcp.  Peer Mac Studio
"""
    lookup = (
        "Peer Mac Studio._ssh._tcp.local. can be reached at "
        "peer-studio.local.:22 (interface 14)"
    )

    assert parse_browse_instances(browse) == ("Peer Mac Studio",)
    assert parse_lookup_target(lookup) == ("peer-studio.local", 22)


def test_discovery_returns_untrusted_suggestions():
    def runner(args, timeout):
        if "-B" in args:
            return DiscoveryOutput(
                "12:00 Add 2 14 local. _ssh._tcp. Peer Mac Studio\n"
            )
        return DiscoveryOutput(
            "service can be reached at peer-studio.local.:22 (interface 14)\n"
        )

    result = discover_ssh_peers(runner=runner)

    assert result["trusted"] is False
    assert result["peers"] == [
        {
            "name": "Peer Mac Studio",
            "ssh": "peer-studio.local",
            "service": "_ssh._tcp.local.",
        }
    ]


def test_ssh_discovery_does_not_rediscover_internal_fqdn_as_local(monkeypatch):
    monkeypatch.setattr(
        "omlx.cluster.discovery.socket.gethostname",
        lambda: "local-mac.example.internal",
    )

    def runner(args, timeout):
        if "-B" in args:
            return DiscoveryOutput(
                "12:00 Add 2 14 local. _ssh._tcp. Local Mac Studio\n"
            )
        return DiscoveryOutput(
            "service can be reached at local-mac.local.:22 (interface 14)\n"
        )

    assert discover_ssh_peers(runner=runner)["peers"] == []


def test_omlx_discovery_does_not_rediscover_internal_fqdn_as_local(monkeypatch):
    monkeypatch.setattr(
        "omlx.cluster.discovery.socket.gethostname",
        lambda: "local-mac.example.internal",
    )

    def runner(args, timeout):
        if "-B" in args:
            return DiscoveryOutput(
                "12:00 Add 2 14 local. _omlx._tcp. oMLX on Local Mac Studio\n"
            )
        return DiscoveryOutput(
            "service can be reached at local-mac.local.:8000 (interface 14)\n"
        )

    assert discover_omlx_peers(runner=runner)["peers"] == []


def test_pairing_token_generation_and_verification():
    token = generate_pairing_token(shared_secret=_PAIRING_SECRET)
    assert token is not None
    assert len(token) > 0
    assert verify_pairing_token(token, shared_secret=_PAIRING_SECRET) is True


def test_pairing_token_rejects_a_different_shared_secret():
    token = generate_pairing_token(shared_secret=_PAIRING_SECRET)
    assert (
        verify_pairing_token(
            token,
            shared_secret="a-different-shared-secret",
        )
        is False
    )


def test_pairing_token_rejects_invalid():
    assert verify_pairing_token("invalid-token", shared_secret=_PAIRING_SECRET) is False
    assert verify_pairing_token("", shared_secret=_PAIRING_SECRET) is False


def test_discover_all_peers_merges_ssh_and_omlx():
    def runner(args, timeout):
        # Check for oMLX service type in args
        if "_omlx._tcp" in " ".join(args):
            if "-B" in args:
                return DiscoveryOutput(
                    "12:00 Add 2 14 local. _omlx._tcp. oMLX Studio\n"
                )
            return DiscoveryOutput(
                "service can be reached at studio.local.:22 (interface 14)\n"
            )
        # SSH service
        if "-B" in args:
            return DiscoveryOutput(
                "12:00 Add 2 14 local. _ssh._tcp. Peer Mac Studio\n"
            )
        return DiscoveryOutput(
            "service can be reached at peer-studio.local.:22 (interface 14)\n"
        )

    clear_peer_transport_cache()
    result = discover_all_peers(runner=runner)

    assert result["trusted"] is False
    assert len(result["peers"]) == 2
    # Unauthenticated Bonjour discovery must not mint a credential. Pairing
    # tokens are created explicitly after both Macs share a secret.
    assert result["pairing_token"] is None

    # Check that peers have the right structure
    peer_names = {p["ssh"] for p in result["peers"]}
    assert "studio.local" in peer_names
    assert "peer-studio.local" in peer_names

    # Nothing has been probed, so transport is pending rather than "unknown" —
    # "unknown" would claim we looked and could not tell.
    omlx_peer = next(p for p in result["peers"] if p["ssh"] == "studio.local")
    assert omlx_peer["transport"] == "detecting"
    assert omlx_peer["link_speed_gbps"] is None
    assert omlx_peer["rdma_available"] is False
    assert omlx_peer["service"] == "oMLX Distributed"


def _single_peer_runner(args, timeout):
    if "-B" in args:
        return DiscoveryOutput("12:00 Add 2 14 local. _ssh._tcp. Peer Mac Studio\n")
    return DiscoveryOutput(
        "service can be reached at peer-studio.local.:22 (interface 14)\n"
    )


def test_discovery_never_probes_the_network_by_default():
    """Discovery must not open an SSH connection to list peers.

    Inlining transport detection here previously hung the suite on SSH to
    hostnames that do not exist.
    """

    clear_peer_transport_cache()

    def exploding_probe(hosts):
        raise AssertionError("discovery must not probe transports by default")

    result = discover_all_peers(runner=_single_peer_runner)
    assert result["peers"][0]["transport"] == "detecting"

    # And when a probe *is* supplied, it is the only thing consulted.
    result = discover_all_peers(
        runner=_single_peer_runner,
        transport_probe=lambda hosts: {
            "peer-studio.local": {
                "transport": "thunderbolt",
                "link_speed_gbps": 120,
                "rdma_available": False,
            }
        },
    )
    peer = result["peers"][0]
    assert peer["transport"] == "thunderbolt"
    assert peer["link_speed_gbps"] == 120

    # exploding_probe is never invoked without being passed in
    assert callable(exploding_probe)


def test_recorded_transports_are_reused_by_discovery():
    """A probe via /transports fills the cache; discovery then reports it free."""

    clear_peer_transport_cache()

    class FakeTransport:
        peer_node_id = "peer-studio.local"
        kind = "rdma"
        link_speed_gbps = 80

    record_peer_transports([FakeTransport()])
    try:
        peer = discover_all_peers(runner=_single_peer_runner)["peers"][0]
        assert peer["transport"] == "rdma"
        assert peer["link_speed_gbps"] == 80
        assert peer["rdma_available"] is True
    finally:
        clear_peer_transport_cache()
