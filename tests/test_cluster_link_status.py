# SPDX-License-Identifier: Apache-2.0
"""Link detection, addressing and reachability between cluster Macs."""

import ipaddress
import shlex
import subprocess
from dataclasses import replace

import pytest

from omlx.cluster.transport import (
    HostInterfaces,
    InterfaceAddress,
    TransportInfo,
    _run_link_command,
    detect_transports,
    link_warnings,
    parse_interface_addresses,
    parse_link_local_addresses,
    parse_media_speeds,
    parse_self_assigned_interfaces,
    parse_thunderbolt_interfaces,
    parse_wifi_interfaces,
    resolve_link_addresses,
    select_backend,
    shared_link_addresses,
    verify_link_reachability,
)


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


@pytest.mark.parametrize("probe_fails, kinds", [(True, set()), (False, {"ethernet"})])
def test_a_failed_probe_reports_no_link_instead_of_ethernet(
    monkeypatch, probe_fails, kinds
):
    class FakeConfig:
        class Host:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        @staticmethod
        def extract_connectivity(hosts, verbose=False):
            if probe_fails:
                raise ValueError("ssh returned no output")
            return hosts, {}

        @staticmethod
        def make_connectivity_matrix(hosts, reverse):
            return []

    monkeypatch.setattr(
        "omlx.cluster.transport._import_mlx_config",
        lambda: FakeConfig,
    )
    monkeypatch.setattr(
        "omlx.cluster.transport._rdma_available",
        lambda hosts, ssh_prefix="": False,
    )

    transports = detect_transports(["a.local", "b.local"])

    assert {item.kind for item in transports} == kinds


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


def test_a_thunderbolt_cable_on_its_self_assigned_address_is_named_with_the_fix():
    """A Thunderbolt Bridge left on DHCP gets 169.254, which nothing routes."""

    ports = _LAPTOP_PORTS + "\nHardware Port: Thunderbolt Bridge\nDevice: bridge0\n"
    ifconfig = (
        _LAPTOP_IFCONFIG
        + "bridge0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST>\n"
        + "    inet 169.254.10.1 netmask 0xffff0000 broadcast 169.254.255.255\n"
    )
    laptop = HostInterfaces(
        host="127.0.0.1",
        addresses=parse_interface_addresses(ifconfig),
        thunderbolt_interfaces=parse_thunderbolt_interfaces(ports),
        wifi_interfaces=parse_wifi_interfaces(ports),
        self_assigned_interfaces=parse_self_assigned_interfaces(ifconfig),
    )
    assert laptop.self_assigned_interfaces == {"en5", "bridge0"}
    assert laptop.wifi_interfaces == {"en0"}
    studio = HostInterfaces(
        host="Studio.local",
        addresses=(InterfaceAddress("en0", "192.168.4.22", 24),),
        thunderbolt_interfaces=frozenset({"bridge0"}),
        wifi_interfaces=frozenset({"en0"}),
        self_assigned_interfaces=frozenset({"bridge0"}),
    )

    (warning,) = link_warnings(laptop, studio, shared_link_addresses(laptop, studio))
    assert "127.0.0.1 (bridge0) and Studio.local (bridge0)" in warning
    assert "self-assigned 169.254" in warning
    assert "uses Wi-Fi instead" in warning
    assert "manual IPv4 address on both Macs" in warning
    # Wi-Fi alone is still worth saying; a Thunderbolt link needs nothing.
    plain = replace(studio, self_assigned_interfaces=frozenset())
    (warning,) = link_warnings(laptop, plain, shared_link_addresses(laptop, plain))
    assert warning.startswith("127.0.0.1 and Studio.local reach each other over Wi-Fi")
    thunderbolt = shared_link_addresses(_laptop(), _studio())
    assert link_warnings(_laptop(), _studio(), thunderbolt) == ()


def test_an_address_on_a_down_interface_is_not_offered():
    """en6 still lists 10.0.1.9; binding it fails the same way a stale one does."""

    parsed = parse_interface_addresses(_LAPTOP_IFCONFIG)
    addresses = {(entry.interface, entry.address) for entry in parsed}

    assert ("en4", "10.0.1.1") in addresses
    assert ("en6", "10.0.1.9") not in addresses


def test_link_local_addresses_and_media_speeds_are_read_per_up_interface():
    ifconfig = (
        "bridge0: flags=8a63<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500\n"
        "\tinet6 fe80::34dc:83ff:fef6:3f80%bridge0 prefixlen 64 scopeid 0x1a \n"
        "\tinet 169.254.120.99 netmask 0xffff0000 broadcast 169.254.255.255\n"
        "\tmedia: autoselect\n"
        "en0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500\n"
        "\tinet6 fd34::1 prefixlen 64 autoconf secured \n"
        "\tinet6 fe80::4a6:44e2:6646:9035%en0 prefixlen 64 secured scopeid 0xb \n"
        "\tmedia: autoselect (10Gbase-T <full-duplex>)\n"
        "en5: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500\n"
        "\tmedia: 100baseTX <full-duplex>\n"
        "en7: flags=8822<BROADCAST,SMART,SIMPLEX,MULTICAST> mtu 1500\n"
        "\tinet6 fe80::9%en7 prefixlen 64 scopeid 0x9 \n"
        "\tmedia: autoselect (2500Base-T <full-duplex>)\n"
    )

    assert parse_link_local_addresses(ifconfig) == {
        "bridge0": "fe80::34dc:83ff:fef6:3f80",
        "en0": "fe80::4a6:44e2:6646:9035",
    }
    assert parse_media_speeds(ifconfig) == {"en0": 10000, "en5": 100}


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


def test_remote_link_probe_reaches_the_peer_shell_intact(monkeypatch):
    seen = []

    def fake_run(argv, **_kwargs):
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr("omlx.cluster.transport.subprocess.run", fake_run)
    command = ("python3", "-c", "import socket\ns=(1, 2)", "10.0.0.1")

    _run_link_command("Studio.local", command)

    assert seen[0][0] == "ssh"
    assert shlex.split(seen[0][-1]) == list(command)


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
