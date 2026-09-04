# SPDX-License-Identifier: Apache-2.0
"""VPN pre-warning heuristics: detection, parsers, hostile-prefix enrichment.

Fixtures model the real incident: the peer ran Cloudflare WARP full-tunnel
under an org policy applying to all interfaces, and the Thunderbolt fabric
was silently swallowed until the link was moved into WARP's 172.16.0.0/12
exclusion. Detection here only warns and steers selection — no configuration
read may promote the transport ladder.
"""

from __future__ import annotations

import ipaddress
import subprocess

from omlx.cluster.transport import HostInterfaces, InterfaceAddress
from omlx.cluster.vpn import (
    VPNProfile,
    detect_vpn,
    exclusion_instruction,
    full_tunnel_warning,
    hostile_networks,
    parse_route_table,
    parse_warp_settings,
)

# Shape of `netstat -rn` on a macOS 15 Mac running Cloudflare WARP in
# full-tunnel mode (captured 2026-08 on the incident cluster's peer): the
# 0/1 + 128/1 pair outranks the physical default, and RFC1918 blocks are
# routed through the tunnel's utun.
_WARP_FULL_TUNNEL_NETSTAT = """\
Routing tables

Internet:
Destination        Gateway            Flags               Netif Expire
default            192.168.4.1        UGScg                 en0
default            link#22            UCSIg               utun4
0/1                utun4              USc                 utun4
10/8               utun4              USc                 utun4
100.64/10          utun4              USc                 utun4
127                127.0.0.1          UCS                   lo0
127.0.0.1          127.0.0.1          UH                    lo0
128.0/1            utun4              USc                 utun4
169.254            link#12            UCS                   en0      !
192.168.4          link#12            UCS                   en0      !
192.168.4.1/32     link#12            UCS                   en0      !
192.168.4.21       8c:aa:ee:0:11:22   UHLWIir               en0   1187

Internet6:
Destination                             Gateway                         Flags               Netif Expire
default                                 fe80::%utun0                    UGcIg               utun0
"""

# `warp-cli settings` (warp-cli 2025.x) with a managed split-tunnel Exclude
# policy — the 172.16.0.0/12 line is the range the incident's fix used.
# 10.0.0.0/8 is deliberately NOT here: the org policy did not exclude it,
# which is exactly why the original 10.0.1.x fabric assignment got tunneled
# (see _WARP_FULL_TUNNEL_NETSTAT's "10/8 utun4" route below) instead of
# reaching the peer directly — an excluded 10.0.0.0/8 would have made the
# incident impossible to reproduce, and the two fixtures would contradict
# each other (a range can't be both tunneled and excluded from the tunnel).
_WARP_SETTINGS_EXCLUDE = """\
Always On: true
Switch Locked: true
Mode: WarpWithDnsOverHttps
Disabled for Wifi: false
Disabled for Ethernet: false
Fallback domains: intranet, internal, private, localdomain
Exclude mode, with hosts/ips:
  100.64.0.0/10
  169.254.0.0/16
  172.16.0.0/12
  192.168.0.0/16
  broker.example.com
  224.0.0.0/24
Daemon Teams Auth: true
"""


def _interfaces(host, entries):
    return HostInterfaces(
        host=host,
        addresses=tuple(InterfaceAddress(*entry) for entry in entries),
    )


class _Runner:
    """A LinkCommandRunner returning canned output per command binary."""

    def __init__(self, outputs):
        # outputs: {binary basename: (returncode, stdout)}
        self.outputs = outputs
        self.calls = []

    def __call__(self, host, command):
        self.calls.append((host, tuple(command)))
        name = command[0].rsplit("/", 1)[-1]
        returncode, stdout = self.outputs.get(name, (1, ""))
        return subprocess.CompletedProcess(list(command), returncode, stdout, "")


def _warp_runner():
    return _Runner(
        {
            "netstat": (0, _WARP_FULL_TUNNEL_NETSTAT),
            "ls": (0, "Cloudflare WARP.app\nSafari.app\nUtilities\n"),
            "warp-cli": (0, _WARP_SETTINGS_EXCLUDE),
        }
    )


def test_warp_full_tunnel_fixture_is_detected():
    profile = detect_vpn(
        "peer.local",
        runner=_warp_runner(),
        interfaces=_interfaces(
            "peer.local",
            [("en0", "192.168.4.22", 24), ("utun4", "172.16.0.2", 32)],
        ),
    )

    assert profile.present
    assert profile.full_tunnel
    assert profile.client == "warp"
    assert "utun4" in profile.utun_interfaces
    assert "172.16.0.0/12" in profile.exclusions


def test_split_tunnel_exclusions_are_parsed_and_hostnames_skipped():
    exclusions = parse_warp_settings(_WARP_SETTINGS_EXCLUDE)

    assert "172.16.0.0/12" in exclusions
    assert "192.168.0.0/16" in exclusions
    # 10.0.0.0/8 is deliberately NOT excluded in this fixture -- see its
    # definition above: it's the range the incident's netstat capture shows
    # actually tunneled, and a fixture claiming it as both tunneled and
    # excluded would be internally contradictory.
    assert "10.0.0.0/8" not in exclusions
    assert all("/" in entry for entry in exclusions)
    assert not any("example.com" in entry for entry in exclusions)


def test_include_mode_yields_no_readable_exclusions():
    output = "Include mode, with hosts/ips:\n  10.10.0.0/16\n"

    assert parse_warp_settings(output) == ()


def test_garbage_output_degrades_to_unknown_without_raising():
    runner = _Runner(
        {
            "netstat": (0, "%% not a routing table %%\n<garbage>"),
            "ls": (0, "\x00binary junk"),
            "warp-cli": (0, "segfault-ish output ///"),
        }
    )

    profile = detect_vpn(
        "peer.local",
        runner=runner,
        interfaces=_interfaces("peer.local", [("utun4", "172.16.0.2", 32)]),
    )

    assert profile.present
    assert profile.client == "unknown"
    assert profile.full_tunnel is False
    assert profile.exclusions == ()


def test_every_command_failing_still_reports_the_probed_utun():
    runner = _Runner({})  # every command returns rc 1

    profile = detect_vpn(
        "peer.local",
        runner=runner,
        interfaces=_interfaces("peer.local", [("utun4", "172.16.0.2", 32)]),
    )

    assert profile.present
    assert profile.client == "unknown"
    assert profile.exclusions == ()


def test_a_clean_host_is_reported_clean_without_remote_reads():
    runner = _Runner({})

    profile = detect_vpn(
        "peer.local",
        runner=runner,
        interfaces=_interfaces("peer.local", [("en0", "192.168.4.22", 24)]),
    )

    assert profile == VPNProfile()
    assert runner.calls == []


def test_route_table_parses_macos_truncated_destinations():
    routes = dict(parse_route_table(_WARP_FULL_TUNNEL_NETSTAT))

    assert routes[ipaddress.ip_network("10.0.0.0/8")] == "utun4"
    assert routes[ipaddress.ip_network("100.64.0.0/10")] == "utun4"
    assert routes[ipaddress.ip_network("127.0.0.0/8")] == "lo0"
    assert routes[ipaddress.ip_network("192.168.4.0/24")] == "en0"
    assert routes[ipaddress.ip_network("0.0.0.0/1")] == "utun4"
    # The Internet6 section is ignored entirely.
    assert all(net.version == 4 for net in routes)


def test_the_half_default_pair_via_utun_means_full_tunnel():
    # No literal default route through the tunnel — only the 0/1 + 128/1
    # pair VPN clients install to outrank the physical default.
    runner = _Runner(
        {
            "netstat": (
                0,
                "Internet:\n"
                "Destination        Gateway            Flags               Netif Expire\n"
                "default            192.168.4.1        UGScg                 en0\n"
                "0/1                utun4              USc                 utun4\n"
                "128.0/1            utun4              USc                 utun4\n",
            ),
            "ls": (0, "Cloudflare WARP.app\n"),
            "warp-cli": (0, _WARP_SETTINGS_EXCLUDE),
        }
    )

    profile = detect_vpn(
        "peer.local",
        runner=runner,
        interfaces=_interfaces("peer.local", [("utun4", "172.16.0.2", 32)]),
    )

    assert profile.full_tunnel


def test_a_split_tunnel_client_is_present_but_not_full_tunnel():
    runner = _Runner(
        {
            "netstat": (
                0,
                "Internet:\n"
                "Destination        Gateway            Flags               Netif Expire\n"
                "default            192.168.4.1        UGScg                 en0\n"
                "100.64/10          utun6              USc                 utun6\n",
            ),
            "ls": (0, "Tailscale.app\n"),
        }
    )

    profile = detect_vpn(
        "peer.local",
        runner=runner,
        interfaces=_interfaces("peer.local", [("utun6", "100.101.1.2", 32)]),
    )

    assert profile.present
    assert profile.client == "tailscale"
    assert not profile.full_tunnel


def test_utun_routed_10_slash_8_marks_the_whole_slash_8_hostile():
    runner = _Runner({"netstat": (0, _WARP_FULL_TUNNEL_NETSTAT)})
    interfaces = {
        "peer.local": _interfaces(
            "peer.local",
            [("en0", "192.168.4.22", 24), ("utun4", "172.16.0.2", 32)],
        )
    }

    hostile = hostile_networks(
        ["peer.local"], runner=runner, interfaces=interfaces
    )

    assert ipaddress.ip_network("10.0.0.0/8") in hostile
    # Catch-alls mark full_tunnel, they do not veto every candidate subnet.
    assert ipaddress.ip_network("0.0.0.0/1") not in hostile
    assert ipaddress.ip_network("128.0.0.0/1") not in hostile
    # Interface subnets are still counted, as _occupied_networks did.
    assert ipaddress.ip_network("192.168.4.0/24") in hostile


def test_hostile_networks_tolerates_probe_and_route_failures():
    def failing_probe(host):
        raise RuntimeError("ssh down")

    hostile = hostile_networks(
        ["peer.local"], probe=failing_probe, runner=_Runner({})
    )

    assert hostile == ()


def test_profile_serializes_the_response_shape():
    profile = VPNProfile(
        present=True,
        client="warp",
        full_tunnel=True,
        utun_interfaces=("utun4",),
        exclusions=("172.16.0.0/12",),
    )

    assert profile.to_dict() == {
        "present": True,
        "client": "warp",
        "full_tunnel": True,
        "exclusions": ["172.16.0.0/12"],
    }
    assert profile.exclusion_networks == (
        ipaddress.ip_network("172.16.0.0/12"),
    )


def test_warning_and_instruction_copy_are_single_sourced():
    warning = full_tunnel_warning("aphoenix-mbp.local", "warp")

    assert "is on a corporate VPN that captures all traffic" in warning
    assert "Cloudflare WARP" in warning
    assert "pick link addresses the VPN ignores" in warning
    assert "verify the link end-to-end before use" in warning
    assert "172.16.99.0/24" in exclusion_instruction("warp", "172.16.99.0/24")
    assert "172.16.99.0/24" in exclusion_instruction("", "172.16.99.0/24")
