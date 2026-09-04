# SPDX-License-Identifier: Apache-2.0
"""Each link state must be distinguishable and carry the right fix.

"RDMA is not working" is useless. There are four different reasons it might not
be, each with a different remedy, and only one of them is something the user can
be told to click. These tests pin the distinctions.
"""

import ipaddress
import subprocess

import pytest

from omlx.cluster.transport import (
    HostInterfaces,
    InterfaceAddress,
    LinkStatus,
    TransportInfo,
    assess_link,
    classify_link,
    configure_link,
    detect_transports,
    parse_interface_addresses,
    parse_linux_ip_addresses,
    parse_thunderbolt_interfaces,
    resolve_link_addresses,
    select_backend,
    shared_link_addresses,
    verify_link_reachability,
)

HOSTS = ("127.0.0.1", "Studio.local")


def _classify(**overrides):
    base = dict(
        rdma_devices={h: ["rdma_en6"] for h in HOSTS},
        active_ports={"127.0.0.1": "en6", "Studio.local": "en5"},
        port_ips={"127.0.0.1": "10.0.1.1", "Studio.local": "10.0.1.2"},
        thunderbolt=True,
        link_speed_gbps=120,
        tb_version="TB5",
    )
    base.update(overrides)
    return classify_link(**base)


def test_everything_working_says_so_plainly():
    status = _classify()
    assert status.state == "rdma_ready"
    assert status.title == "RDMA enabled and linked"
    assert status.ready is True
    assert status.backend == "jaccl"
    assert status.commands == (), "nothing to fix, so offer no commands"
    assert status.link_label == "TB5 at 120 Gb/s"


def test_linux_ip_fallback_reads_active_ipv4_interfaces():
    addresses = parse_linux_ip_addresses(
        "2: enp1s0f0np0    inet 192.168.100.1/30 brd 192.168.100.3 scope global\n"
        "1: lo    inet 127.0.0.1/8 scope host\n"
    )

    assert addresses == (
        InterfaceAddress("enp1s0f0np0", "192.168.100.1", 30),
    )


def test_ports_without_an_ip_are_prepared_by_start_cluster():
    """The live state is a GUI action, not a terminal handoff."""

    status = _classify(port_ips={"127.0.0.1": None, "Studio.local": None})

    assert status.state == "rdma_needs_setup"
    assert status.ready is False
    assert status.setup_available is True
    assert status.commands == ()
    assert "Start Cluster" in status.detail
    assert "administrator" in status.detail
    assert status.doc_url


def test_rdma_never_enabled_points_at_recovery_not_sudo():
    """rdma_ctl runs in Recovery; no amount of sudo helps, so do not suggest it."""

    status = _classify(rdma_devices={"127.0.0.1": ["rdma_en6"], "Studio.local": []})

    assert status.state == "rdma_not_enabled"
    assert "Studio.local" in status.detail
    assert any("rdma_ctl enable" in c for c in status.commands)
    assert not any("sudo ifconfig" in c for c in status.commands)


def test_thunderbolt_without_an_active_port_mentions_tb4():
    status = _classify(active_ports={"127.0.0.1": None, "Studio.local": None})

    assert status.state == "thunderbolt"
    assert status.ready is False
    assert "Thunderbolt 4" in status.detail
    assert status.commands == (), "a cable or hardware issue has no command fix"


def test_lan_only_is_a_valid_state_not_an_error():
    status = _classify(thunderbolt=False, tb_version=None, link_speed_gbps=None)

    assert status.state == "ethernet"
    assert status.ready is True, "Ethernet works; it is just slower"
    assert status.backend == "ring"
    assert "slower" in status.detail
    assert status.commands == ()


def test_rdma_transport_selects_jaccl_without_duplicate_thunderbolt_record():
    transports = (
        TransportInfo(
            kind="rdma",
            interface="rdma",
            peer_node_id="Studio.local",
            source_node_id="127.0.0.1",
        ),
    )

    assert select_backend(transports) == "jaccl"


def test_no_peers_is_not_reported_as_a_failure():
    status = assess_link([])
    assert status.state == "unknown"
    assert status.commands == ()


def test_failed_peer_probe_is_not_mislabeled_as_rdma_disabled(monkeypatch):
    def rejected(_host):
        raise RuntimeError("SSH permission denied")

    monkeypatch.setattr("omlx.cluster.transport._rdma_devices", rejected)

    status = assess_link(HOSTS)

    assert status.state == "unknown"
    assert status.ready is False
    assert status.title == "Could not verify the peer's RDMA state"
    assert "SSH permission denied" in status.detail
    assert status.commands == ()


def test_routable_rdma_fabric_overrides_stale_port_down_state(monkeypatch):
    """The address path is stronger evidence than stale ibv PORT_DOWN."""

    monkeypatch.setattr(
        "omlx.cluster.transport._rdma_devices",
        lambda host: ["rdma_en6"] if host == HOSTS[0] else ["rdma_en5"],
    )
    monkeypatch.setattr(
        "omlx.cluster.transport._active_rdma_port",
        lambda _host: None,
    )
    interfaces = {
        HOSTS[0]: _host(
            HOSTS[0],
            [("en6", "10.0.1.1", 24)],
            rdma=("en6",),
            thunderbolt=("en6",),
        ),
        HOSTS[1]: _host(
            HOSTS[1],
            [("en5", "10.0.1.2", 24)],
            rdma=("en5",),
            thunderbolt=("en5",),
        ),
    }

    status = assess_link(HOSTS, probe=interfaces.__getitem__)

    assert status.state == "rdma_ready"
    assert status.backend == "jaccl"
    assert status.ready is True
    assert "en6" in status.detail
    assert "en5" in status.detail


def test_link_status_does_not_call_unreachable_rdma_addresses_ready(monkeypatch):
    monkeypatch.setattr(
        "omlx.cluster.transport._rdma_devices",
        lambda host: ["rdma_en6"] if host == HOSTS[0] else ["rdma_en5"],
    )
    monkeypatch.setattr(
        "omlx.cluster.transport._active_rdma_port",
        lambda host: "en6" if host == HOSTS[0] else "en5",
    )
    monkeypatch.setattr(
        "omlx.cluster.transport._interface_ip",
        lambda host, _interface: "10.0.1.1" if host == HOSTS[0] else "10.0.1.2",
    )
    interfaces = {
        HOSTS[0]: _host(HOSTS[0], [("en6", "10.0.1.1", 30)], rdma=("en6",)),
        HOSTS[1]: _host(HOSTS[1], [("en5", "10.0.1.2", 30)], rdma=("en5",)),
    }

    status = assess_link(
        HOSTS,
        probe=interfaces.__getitem__,
        verify=lambda _link: (False, "the peer did not answer"),
    )

    assert status.state == "thunderbolt"
    assert status.ready is False
    assert status.backend == "ring"
    assert "did not answer" in status.detail


def test_link_status_reports_verified_ethernet_when_rdma_does_not_answer(
    monkeypatch,
):
    monkeypatch.setattr(
        "omlx.cluster.transport._rdma_devices",
        lambda host: ["rdma_en6"] if host == HOSTS[0] else ["rdma_en5"],
    )
    monkeypatch.setattr(
        "omlx.cluster.transport._active_rdma_port",
        lambda host: "en6" if host == HOSTS[0] else "en5",
    )
    monkeypatch.setattr(
        "omlx.cluster.transport._interface_ip",
        lambda host, _interface: "10.0.1.1" if host == HOSTS[0] else "10.0.1.2",
    )
    interfaces = {
        HOSTS[0]: _host(
            HOSTS[0],
            [("en6", "10.0.1.1", 30), ("en0", "192.168.4.21", 24)],
            rdma=("en6",),
        ),
        HOSTS[1]: _host(
            HOSTS[1],
            [("en5", "10.0.1.2", 30), ("en0", "192.168.4.22", 24)],
            rdma=("en5",),
        ),
    }

    status = assess_link(
        HOSTS,
        probe=interfaces.__getitem__,
        verify=lambda link: (link.kind == "ethernet", "RDMA did not answer"),
    )

    assert status.state == "ethernet"
    assert status.ready is True
    assert status.backend == "ring"
    assert "192.168.4.21" in status.detail
    assert "verified path" in status.detail


def test_link_status_does_not_call_an_unverified_network_route_ready(monkeypatch):
    monkeypatch.setattr("omlx.cluster.transport._rdma_devices", lambda _host: [])
    monkeypatch.setattr(
        "omlx.cluster.transport._active_rdma_port", lambda _host: None
    )
    interfaces = {
        HOSTS[0]: _host(HOSTS[0], [("en0", "192.168.4.21", 24)]),
        HOSTS[1]: _host(HOSTS[1], [("en0", "192.168.4.22", 24)]),
    }

    status = assess_link(
        HOSTS,
        probe=interfaces.__getitem__,
        verify=lambda _link: (False, "the peer did not answer"),
    )

    assert status.state == "unknown"
    assert status.ready is False
    assert status.backend == "ring"
    assert "unverified address" in status.detail


def test_gui_setup_addresses_only_the_missing_endpoint(monkeypatch):
    """A retry is idempotent and does not prompt again on the ready Mac."""

    states = iter(
        [
            LinkStatus(
                state="rdma_needs_setup",
                title="setup",
                detail="setup",
                backend="jaccl",
                ready=False,
                setup_available=True,
            ),
            LinkStatus(
                state="rdma_ready",
                title="ready",
                detail="ready",
                backend="jaccl",
                ready=True,
            ),
        ]
    )
    monkeypatch.setattr("omlx.cluster.transport.assess_link", lambda hosts: next(states))
    monkeypatch.setattr(
        "omlx.cluster.transport._active_rdma_port",
        lambda host: "en6" if host == "127.0.0.1" else "en5",
    )
    monkeypatch.setattr(
        "omlx.cluster.transport._interface_ip",
        lambda host, interface: None if host == "127.0.0.1" else "10.0.1.2",
    )
    configured = []
    monkeypatch.setattr(
        "omlx.cluster.transport._authorized_ifconfig",
        lambda host, interface, address: configured.append((host, interface, address)),
    )

    status = configure_link(HOSTS)

    assert status.ready is True
    assert configured == [("127.0.0.1", "en6", "10.0.1.1")]


def test_gui_setup_uses_native_authorization_on_both_macs(monkeypatch):
    states = iter(
        [
            LinkStatus(
                state="rdma_needs_setup",
                title="setup",
                detail="setup",
                backend="jaccl",
                ready=False,
                setup_available=True,
            ),
            LinkStatus(
                state="rdma_ready",
                title="ready",
                detail="ready",
                backend="jaccl",
                ready=True,
            ),
        ]
    )
    monkeypatch.setattr("omlx.cluster.transport.assess_link", lambda hosts: next(states))
    monkeypatch.setattr(
        "omlx.cluster.transport._active_rdma_port",
        lambda host: "en6" if host == "127.0.0.1" else "en5",
    )
    monkeypatch.setattr(
        "omlx.cluster.transport._interface_ip", lambda host, interface: None
    )
    configured = []
    monkeypatch.setattr(
        "omlx.cluster.transport._authorized_ifconfig",
        lambda host, interface, address: configured.append((host, interface, address)),
    )

    configure_link(HOSTS)

    assert configured == [
        ("127.0.0.1", "en6", "10.0.1.1"),
        ("Studio.local", "en5", "10.0.1.2"),
    ]


def test_gui_setup_can_configure_a_worker_to_worker_pair(monkeypatch):
    hosts = ("mini.local", "studio.local")
    states = iter(
        [
            LinkStatus(
                state="rdma_needs_setup",
                title="setup",
                detail="setup",
                backend="jaccl",
                ready=False,
                setup_available=True,
            ),
            LinkStatus(
                state="rdma_ready",
                title="ready",
                detail="ready",
                backend="jaccl",
                ready=True,
            ),
        ]
    )
    monkeypatch.setattr("omlx.cluster.transport.assess_link", lambda pair: next(states))
    monkeypatch.setattr(
        "omlx.cluster.transport._active_rdma_port",
        lambda host: "en5",
    )
    monkeypatch.setattr(
        "omlx.cluster.transport._interface_ip",
        lambda host, interface: None,
    )
    configured = []
    monkeypatch.setattr(
        "omlx.cluster.transport._authorized_ifconfig",
        lambda host, interface, address: configured.append((host, address)),
    )

    assert configure_link(hosts).ready is True
    assert configured == [
        ("mini.local", "10.0.1.1"),
        ("studio.local", "10.0.1.2"),
    ]


def test_remote_authorization_is_launched_in_the_peers_gui_session(monkeypatch):
    """SSH's audit session cannot host SecurityAgent; LaunchServices can."""

    from omlx.cluster import transport

    calls = []

    def run(command, **kwargs):
        calls.append(command)
        remote_command = command[-1]
        returncode = (
            0
            if "/bin/test" not in remote_command or ".success" in remote_command
            else 1
        )
        return subprocess.CompletedProcess(command, returncode, "", "")

    monkeypatch.setattr(transport.subprocess, "run", run)

    transport._remote_gui_authorize(
        "Studio.local",
        "/sbin/ifconfig en5 inet 10.0.1.2 netmask 255.255.255.0 up",
    )

    remote_commands = [item[-1] for item in calls]
    compile_command = next(item for item in remote_commands if "osacompile" in item)
    assert "administrator privileges" in compile_command
    assert "/sbin/ifconfig en5 inet 10.0.1.2" in compile_command
    assert any("/usr/bin/open" in item and " -W " in item for item in remote_commands)
    assert any("/bin/rm" in item for item in remote_commands)
    assert not any(
        item.startswith("/usr/bin/osascript") for item in remote_commands
    ), "a remote osascript would put SecurityAgent back in the SSH session"


def test_remote_authorization_cancel_is_distinct_from_bad_credentials(monkeypatch):
    from omlx.cluster import transport

    def run(command, **kwargs):
        remote_command = command[-1]
        if "/bin/test" in remote_command:
            returncode = 0 if ".cancelled" in remote_command else 1
        else:
            returncode = 0
        return subprocess.CompletedProcess(command, returncode, "", "")

    monkeypatch.setattr(transport.subprocess, "run", run)

    with pytest.raises(transport.LinkAuthorizationCancelledError):
        transport._remote_gui_authorize("Studio.local", "/usr/bin/true")


def test_rdma_detection_does_not_invent_a_three_mac_full_mesh(monkeypatch):
    class FakeConfig:
        class Host:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        @staticmethod
        def extract_connectivity(hosts, verbose=False):
            return hosts, {}

        @staticmethod
        def make_connectivity_matrix(hosts, reverse):
            return [
                [False, True, False],
                [True, False, True],
                [False, True, False],
            ]

    monkeypatch.setattr(
        "omlx.cluster.transport._import_mlx_config",
        lambda: FakeConfig,
    )
    monkeypatch.setattr(
        "omlx.cluster.transport._extract_tb_link_speed",
        lambda host: 120,
    )
    monkeypatch.setattr(
        "omlx.cluster.transport._rdma_available",
        lambda hosts, ssh_prefix="": True,
    )

    transports = detect_transports(["a.local", "b.local", "c.local"])
    rdma_edges = {
        (item.source_node_id, item.peer_node_id)
        for item in transports
        if item.kind == "rdma"
    }

    assert rdma_edges == {
        ("a.local", "b.local"),
        ("b.local", "a.local"),
        ("b.local", "c.local"),
        ("c.local", "b.local"),
    }
    assert ("a.local", "c.local") not in rdma_edges


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"port_ips": {h: None for h in HOSTS}},
        {"rdma_devices": {h: [] for h in HOSTS}},
        {"active_ports": {h: None for h in HOSTS}},
        {"thunderbolt": False},
    ],
)
def test_every_state_is_serialisable_and_self_explaining(overrides):
    payload = _classify(**overrides).to_dict()
    assert payload["title"] and payload["detail"]
    assert payload["backend"] in {"ring", "jaccl", "jaccl-ring"}
    assert isinstance(payload["commands"], list)
    # A state that is not ready must either give commands or explain why not.
    if not payload["ready"]:
        assert payload["commands"] or payload["detail"]


# ---------------------------------------------------------------------------
# Link addressing
#
# A launch died with "[ring] Couldn't bind socket (error: 49)" because the
# hostfile named 10.0.1.1 on en6 and macOS had renumbered that Thunderbolt port
# to en4. Every test below asks what the host reports now.
# ---------------------------------------------------------------------------

# The laptop after the renumbering: the Thunderbolt port is en4, en6 is down and
# still carrying the address that used to be the fast path.
_LAPTOP_IFCONFIG = """\
lo0: flags=8049<UP,LOOPBACK,RUNNING,MULTICAST> mtu 16384
    inet 127.0.0.1 netmask 0xff000000
en0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
    ether fc:b2:14:9a:cc:e0
    inet 192.168.4.21 netmask 0xffffff00 broadcast 192.168.4.255
en4: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
    inet 10.0.1.1 netmask 0xffffff00 broadcast 10.0.1.255
en6: flags=8822<BROADCAST,SMART,SIMPLEX,MULTICAST> mtu 1500
    inet 10.0.1.9 netmask 0xffffff00 broadcast 10.0.1.255
en5: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
    inet 169.254.138.14 netmask 0xffff0000 broadcast 169.254.255.255
bridge100: flags=8a63<UP,BROADCAST,SMART,RUNNING,ALLMULTI,SIMPLEX,MULTICAST> mtu 1500
    inet 192.168.105.1 netmask 0xffffff00 broadcast 192.168.105.255
"""

_LAPTOP_PORTS = """\
Hardware Port: Wi-Fi
Device: en0
Ethernet Address: fc:b2:14:9a:cc:e0

Hardware Port: USB 10/100/1000 LAN
Device: en8
Ethernet Address: b0:4f:13:ec:8b:86

Hardware Port: Thunderbolt 1
Device: en4
Ethernet Address: 36:e3:f0:2a:46:80

Hardware Port: Thunderbolt 2
Device: en6
Ethernet Address: 36:e3:f0:2a:46:84
"""


def _host(name, addresses, *, rdma=(), thunderbolt=()):
    return HostInterfaces(
        host=name,
        addresses=tuple(InterfaceAddress(*entry) for entry in addresses),
        rdma_interfaces=frozenset(rdma),
        thunderbolt_interfaces=frozenset(thunderbolt),
    )


def _laptop():
    return HostInterfaces(
        host="127.0.0.1",
        addresses=parse_interface_addresses(_LAPTOP_IFCONFIG),
        rdma_interfaces=frozenset({"en4"}),
        thunderbolt_interfaces=parse_thunderbolt_interfaces(_LAPTOP_PORTS),
    )


def _studio():
    return _host(
        "Studio.local",
        [("en0", "192.168.4.22", 24), ("en5", "10.0.1.2", 24)],
        rdma={"en5"},
        thunderbolt={"en5"},
    )


def test_an_address_on_a_down_interface_is_not_offered():
    """en6 still lists 10.0.1.9; binding it fails the same way a stale one does."""

    parsed = parse_interface_addresses(_LAPTOP_IFCONFIG)
    addresses = {(entry.interface, entry.address) for entry in parsed}

    assert ("en4", "10.0.1.1") in addresses
    assert ("en6", "10.0.1.9") not in addresses


def test_loopback_and_self_assigned_addresses_are_never_a_shared_subnet():
    """Every Mac has 127.0.0.1, and 169.254 means DHCP failed on both ends."""

    addresses = {a.address for a in parse_interface_addresses(_LAPTOP_IFCONFIG)}

    assert "127.0.0.1" not in addresses
    assert not any(address.startswith("169.254.") for address in addresses)


def test_a_hex_flag_word_does_not_push_an_address_onto_the_interface_above_it():
    """ifconfig prints flags in hex: bridge100's 8a63 read as no header at all,
    and its address was attributed to the interface listed before it."""

    parsed = parse_interface_addresses(_LAPTOP_IFCONFIG)
    owner = {entry.address: entry.interface for entry in parsed}

    assert owner["192.168.105.1"] == "bridge100"


def test_hex_netmasks_become_the_subnet_a_peer_can_be_matched_against():
    by_interface = {a.interface: a for a in parse_interface_addresses(_LAPTOP_IFCONFIG)}

    assert by_interface["en4"].prefix_length == 24
    assert by_interface["en4"].network == ipaddress.ip_network("10.0.1.0/24")


def test_only_thunderbolt_hardware_ports_are_called_thunderbolt():
    assert parse_thunderbolt_interfaces(_LAPTOP_PORTS) == frozenset({"en4", "en6"})


def test_the_fast_link_wins_over_the_lan_both_hosts_are_also_on():
    link = shared_link_addresses(_laptop(), _studio())

    assert link.kind == "rdma"
    assert link.source.address == "10.0.1.1"
    assert link.peer.address == "10.0.1.2"


def test_a_renumbered_port_resolves_to_the_name_and_address_the_host_has_now():
    """The launch failure: the hostfile said en6, the address had moved to en4."""

    link = shared_link_addresses(_laptop(), _studio())

    assert (link.source.interface, link.source.address) == ("en4", "10.0.1.1")


def test_without_a_fast_link_any_common_subnet_is_still_used():
    laptop = _host("127.0.0.1", [("en0", "192.168.4.21", 24)])
    studio = _host("Studio.local", [("en0", "192.168.4.22", 24)])

    link = shared_link_addresses(laptop, studio)

    assert link.ok
    assert link.kind == "ethernet"
    assert (link.source.address, link.peer.address) == ("192.168.4.21", "192.168.4.22")


def test_a_point_to_point_subnet_beats_the_building_wide_one():
    laptop = _host("a", [("en0", "10.0.0.5", 8), ("en1", "10.0.1.1", 30)])
    studio = _host("b", [("en0", "10.0.0.6", 8), ("en1", "10.0.1.2", 30)])

    link = shared_link_addresses(laptop, studio)

    assert (link.source.address, link.peer.address) == ("10.0.1.1", "10.0.1.2")


def test_two_hosts_holding_the_same_address_is_a_collision_not_a_link():
    """Every Mac's VM bridge picks 192.168.105.1; that is not a cable between them."""

    bridge = [("bridge100", "192.168.105.1", 24)]
    link = shared_link_addresses(_host("a", bridge), _host("b", bridge))

    assert not link.ok
    assert "share no subnet" in link.reason


def test_no_shared_subnet_reports_what_each_host_actually_has():
    laptop = _host("127.0.0.1", [("en4", "10.0.1.1", 24)])
    studio = _host("Studio.local", [("en5", "192.168.9.4", 24)])

    link = shared_link_addresses(laptop, studio)

    assert not link.ok
    assert "10.0.1.1" in link.reason
    assert "192.168.9.4" in link.reason


def test_a_host_with_nothing_routable_is_named_rather_than_blamed_on_the_pair():
    link = shared_link_addresses(
        _host("Studio.local", []), _host("127.0.0.1", [("en0", "192.168.4.21", 24)])
    )

    assert not link.ok
    assert link.reason.startswith("Studio.local")


def test_resolving_a_link_reads_both_hosts_instead_of_a_written_address():
    probed = []
    hosts = {"127.0.0.1": _laptop(), "Studio.local": _studio()}

    def probe(host):
        probed.append(host)
        return hosts[host]

    link = resolve_link_addresses("127.0.0.1", "Studio.local", probe=probe)

    assert probed == ["127.0.0.1", "Studio.local"]
    assert link.source.address == "10.0.1.1"


def test_resolving_a_link_can_reject_an_unreachable_shared_subnet():
    hosts = {"127.0.0.1": _laptop(), "Studio.local": _studio()}

    link = resolve_link_addresses(
        "127.0.0.1",
        "Studio.local",
        probe=hosts.__getitem__,
        verify=lambda _link: (False, "the peer did not answer on that address"),
    )

    assert not link.ok
    assert link.kind == "rdma"
    assert "did not answer" in link.reason


def test_resolving_a_link_falls_back_to_a_verified_ethernet_path():
    hosts = {
        "127.0.0.1": _host(
            "127.0.0.1",
            [("en4", "10.0.1.1", 30), ("en0", "192.168.4.21", 24)],
            rdma=("en4",),
            thunderbolt=("en4",),
        ),
        "Studio.local": _host(
            "Studio.local",
            [("en5", "10.0.1.2", 30), ("en0", "192.168.4.22", 24)],
            rdma=("en5",),
            thunderbolt=("en5",),
        ),
    }
    checked = []

    def verify(link):
        checked.append(link.kind)
        return (link.kind == "ethernet", f"{link.kind} did not answer")

    link = resolve_link_addresses(
        "127.0.0.1",
        "Studio.local",
        probe=hosts.__getitem__,
        verify=verify,
    )

    assert checked == ["rdma", "ethernet"]
    assert link.ok
    assert link.kind == "ethernet"
    assert link.source.address == "192.168.4.21"
    assert link.peer.address == "192.168.4.22"


def test_link_verification_checks_route_and_ping_from_both_macs():
    link = shared_link_addresses(_laptop(), _studio())
    calls = []

    def runner(host, command):
        calls.append((host, tuple(command)))
        if command[0] == "/sbin/route":
            interface = "en4" if host == "127.0.0.1" else "en5"
            return subprocess.CompletedProcess(command, 0, f"interface: {interface}\n", "")
        return subprocess.CompletedProcess(command, 0, "one packet received\n", "")

    verified, reason = verify_link_reachability(link, runner=runner)

    assert verified is True
    assert reason == link.reason
    assert [command[0] for _, command in calls] == [
        "/sbin/route",
        "/sbin/ping",
        "/sbin/route",
        "/sbin/ping",
    ]


def test_link_verification_rejects_a_route_on_the_wrong_interface():
    link = shared_link_addresses(_laptop(), _studio())

    def runner(_host, command):
        return subprocess.CompletedProcess(command, 0, "interface: en0\n", "")

    verified, reason = verify_link_reachability(link, runner=runner)

    assert verified is False
    assert "uses en0" in reason
    assert "en4" in reason


def test_the_detected_link_speed_is_carried_into_the_explanation():
    hosts = {"127.0.0.1": _laptop(), "Studio.local": _studio()}
    transports = (
        TransportInfo(
            kind="thunderbolt",
            interface="Thunderbolt 1",
            peer_node_id="Studio.local",
            source_node_id="127.0.0.1",
            link_speed_gbps=120,
            tb_version="TB5",
        ),
    )

    link = resolve_link_addresses(
        "127.0.0.1", "Studio.local", transports=transports, probe=hosts.__getitem__
    )

    assert link.link_speed_gbps == 120
    assert "120 Gb/s" in link.reason
    assert link.to_dict()["source"]["address"] == "10.0.1.1"
