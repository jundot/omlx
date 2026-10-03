# SPDX-License-Identifier: Apache-2.0
"""Detect a hungry full-tunnel VPN before the fabric ever hits it.

The real incident: the peer ran Cloudflare WARP under an org policy that
applies to all interfaces, and the tunnel silently swallowed the Thunderbolt
fabric until the link was manually re-addressed into WARP's 172.16.0.0/12
split-tunnel exclusion. This module makes that discovery automatic — utun
interfaces, catch-all routes and client signatures become (a) a plain-language
pre-warning and (b) readable exclusion ranges that steer subnet selection.

Nothing read here ever promotes the transport ladder. A configuration read can
be MDM-locked or lie; the bidirectional bound-connect probes in
``transport.verify_link_reachability`` remain the only promotion authority.
Every read below is failure-tolerant: an absent client binary, a locked
``warp-cli`` or garbage output degrades to ``client="unknown"`` and empty
exclusions, never to an exception.
"""

from __future__ import annotations

import ipaddress
import re
import subprocess
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass

from .transport import (
    HostInterfaces,
    LinkCommandRunner,
    _run_link_command,
    probe_host_interfaces,
)

_KNOWN_CLIENTS = ("warp", "tailscale", "globalprotect", "anyconnect")

# App-bundle signatures, matched against one bounded ``ls /Applications``.
# ``ls`` takes no path with spaces here because a remote invocation goes
# through the peer's shell unquoted (see transport._run_link_command).
_APP_SIGNATURES: tuple[tuple[str, str], ...] = (
    ("Cloudflare WARP.app", "warp"),
    ("Tailscale.app", "tailscale"),
    ("GlobalProtect.app", "globalprotect"),
    ("Cisco Secure Client.app", "anyconnect"),
    ("Cisco AnyConnect Secure Mobility Client.app", "anyconnect"),
    # AnyConnect/Secure Client historically installs under /Applications/Cisco.
    ("Cisco", "anyconnect"),
)

# Tailscale addresses every node inside the CGNAT range; an interface address
# there is a client signature even when the app bundle is not readable.
_TAILSCALE_RANGE = ipaddress.ip_network("100.64.0.0/10")

# Per-client copy-paste exclusion instructions (design C.4). Kept beside the
# detection so the Doctor, the dashboard and the incident copy all cite one
# source. ``{cidr}`` is the fabric range to exclude.
EXCLUSION_INSTRUCTIONS: dict[str, str] = {
    "warp": (
        "In Cloudflare WARP: Settings → Preferences → Advanced → Split "
        "Tunnels → add {cidr} to the Exclude list (or ask IT to add it to "
        "the managed Split Tunnel policy). CLI: warp-cli tunnel ip add {cidr}"
    ),
    "tailscale": (
        "Tailscale only claims 100.64.0.0/10; if it still captures the link, "
        "disable 'Use Tailscale subnets' or ask IT to exclude {cidr} from "
        "advertised routes."
    ),
    "globalprotect": (
        "GlobalProtect split tunnels are set by IT: ask them to add {cidr} "
        "to the split-tunnel exclude list of your portal's agent config."
    ),
    "anyconnect": (
        "AnyConnect split tunnels are set by IT: ask them to add {cidr} to "
        "the split-exclude network list of your group policy."
    ),
    "unknown": (
        "Ask IT to exclude {cidr} from the VPN tunnel so the Macs can talk "
        "over the Thunderbolt cable directly."
    ),
}


def exclusion_instruction(client: str, cidr: str) -> str:
    """The copy-paste fix for keeping ``cidr`` out of ``client``'s tunnel."""

    template = EXCLUSION_INSTRUCTIONS.get(client) or EXCLUSION_INSTRUCTIONS[
        "unknown"
    ]
    return template.format(cidr=cidr)


def full_tunnel_warning(host: str, client: str = "") -> str:
    """The design C.5 pre-warning, parameterized on host and client name."""

    names = {
        "warp": "Cloudflare WARP",
        "tailscale": "Tailscale",
        "globalprotect": "GlobalProtect",
        "anyconnect": "Cisco AnyConnect",
    }
    named = f" ({names[client]})" if client in names else ""
    return (
        f"{host} is on a corporate VPN that captures all traffic{named}. "
        "oMLX will pick link addresses the VPN ignores and verify the link "
        "end-to-end before use."
    )


@dataclass(frozen=True)
class VPNProfile:
    """What one host's VPN posture looks like from its own configuration.

    ``client`` is one of ``"warp" | "tailscale" | "globalprotect" |
    "anyconnect" | "unknown"`` when a VPN is present, and ``""`` when none is.
    ``full_tunnel`` means a default route — or the 0.0.0.0/1 + 128.0.0.0/1
    pair VPN clients use to outrank it — points at a utun. ``exclusions`` are
    readable split-tunnel exclusion CIDRs, possibly empty even when a VPN is
    present (locked or unreadable client config).
    """

    present: bool = False
    client: str = ""
    full_tunnel: bool = False
    utun_interfaces: tuple[str, ...] = ()
    exclusions: tuple[str, ...] = ()

    @property
    def exclusion_networks(self) -> tuple[ipaddress.IPv4Network, ...]:
        networks = []
        for cidr in self.exclusions:
            with suppress(ValueError):
                network = ipaddress.ip_network(cidr, strict=False)
                if network.version == 4:
                    networks.append(network)
        return tuple(networks)

    def to_dict(self) -> dict[str, object]:
        return {
            "present": self.present,
            "client": self.client,
            "full_tunnel": self.full_tunnel,
            "exclusions": list(self.exclusions),
        }


# macOS ``netstat -rn`` interface names: en0, utun4, lo0, bridge100 — a word
# of letters then digits. Numbers (the Expire column) never match.
_NETIF = re.compile(r"^[a-z][a-z0-9]*[0-9]$")

# Routes whose destination is the default or a /1 half are tunnel catch-alls:
# they mean "everything", not "this range is claimed", so they mark
# ``full_tunnel`` rather than poisoning every candidate subnet as hostile.
_CATCH_ALL_PREFIX_LENGTH = 2


def _destination_network(token: str) -> ipaddress.IPv4Network | None:
    """One ``netstat -rn`` destination as a network, or None if unreadable.

    macOS truncates trailing zero octets ("10/8", "100.64/10", "127",
    "192.168.1") and spells the default route "default".
    """

    token = token.strip()
    if not token or ":" in token or "%" in token:
        return None
    if token == "default":
        return ipaddress.ip_network("0.0.0.0/0")
    address, _, prefix = token.partition("/")
    octets = address.split(".")
    if not 1 <= len(octets) <= 4 or not all(
        part.isdigit() for part in octets
    ):
        return None
    if not prefix:
        # Truncated destinations are classful-style: each present octet is 8
        # bits of prefix; a full 4-octet destination is a host route.
        prefix = str(8 * len(octets)) if len(octets) < 4 else "32"
    elif not prefix.isdigit():
        return None
    padded = ".".join(octets + ["0"] * (4 - len(octets)))
    with suppress(ValueError):
        return ipaddress.ip_network(f"{padded}/{prefix}", strict=False)
    return None


def parse_route_table(
    netstat_output: str,
) -> tuple[tuple[ipaddress.IPv4Network, str], ...]:
    """(destination, interface) for every readable IPv4 route.

    Pure and defensive: unreadable lines are skipped, the Internet6 section is
    ignored, and garbage input yields an empty tuple rather than an error.
    """

    routes: list[tuple[ipaddress.IPv4Network, str]] = []
    in_ipv4 = True
    for line in netstat_output.splitlines():
        stripped = line.strip()
        if stripped.rstrip(":").lower() == "internet":
            in_ipv4 = True
            continue
        if stripped.rstrip(":").lower() == "internet6":
            in_ipv4 = False
            continue
        if not in_ipv4 or not stripped:
            continue
        fields = stripped.split()
        if len(fields) < 3:
            continue
        destination = _destination_network(fields[0])
        if destination is None:
            continue
        netif = next(
            (field for field in reversed(fields[1:]) if _NETIF.match(field)),
            "",
        )
        if netif:
            routes.append((destination, netif))
    return tuple(routes)


def parse_warp_settings(output: str) -> tuple[str, ...]:
    """Split-tunnel exclusion CIDRs from ``warp-cli settings`` output.

    The exclusion block ("Exclude mode, with hosts/ips:", warp-cli 2023–2025)
    lists one entry per indented line; hostnames are skipped, only IPv4 CIDRs
    are kept. "Include mode" means the listed ranges are what IS tunneled, so
    no exclusions are readable and the tuple is empty. Garbage in, empty out.
    """

    exclusions: list[str] = []
    in_exclude_block = False
    for line in output.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if "include mode" in lowered:
            return ()
        if "exclude mode" in lowered:
            in_exclude_block = True
            continue
        if not in_exclude_block:
            continue
        if not line[:1].isspace():
            # A new top-level "Key: value" section ends the block.
            if stripped:
                break
            continue
        token = stripped.split()[0] if stripped else ""
        if "/" not in token:
            continue
        with suppress(ValueError):
            network = ipaddress.ip_network(token, strict=False)
            if network.version == 4:
                exclusions.append(str(network))
    return tuple(dict.fromkeys(exclusions))


def _utun_interfaces(interfaces: HostInterfaces | None) -> set[str]:
    if interfaces is None:
        return set()
    return {
        entry.interface
        for entry in interfaces.addresses
        if entry.interface.startswith("utun")
    }


def _routes(
    host: str, run: LinkCommandRunner
) -> tuple[tuple[ipaddress.IPv4Network, str], ...]:
    result = run(host, ("/usr/sbin/netstat", "-rn"))
    if result.returncode != 0:
        return ()
    return parse_route_table(result.stdout or "")


def _full_tunnel(
    routes: Iterable[tuple[ipaddress.IPv4Network, str]],
) -> bool:
    utun_destinations = {
        str(destination)
        for destination, netif in routes
        if netif.startswith("utun")
    }
    if "0.0.0.0/0" in utun_destinations:
        return True
    return {"0.0.0.0/1", "128.0.0.0/1"} <= utun_destinations


def _client_signature(
    host: str, run: LinkCommandRunner, interfaces: HostInterfaces | None
) -> str:
    """The best-effort client name, or "" when no signature matches."""

    listing = run(host, ("/bin/ls", "/Applications"))
    if listing.returncode == 0:
        names = {line.strip() for line in (listing.stdout or "").splitlines()}
        for bundle, client in _APP_SIGNATURES:
            if bundle in names:
                return client
    if interfaces is not None and any(
        entry.interface.startswith("utun")
        and ipaddress.ip_address(entry.address) in _TAILSCALE_RANGE
        for entry in interfaces.addresses
    ):
        return "tailscale"
    return ""


def _warp_exclusions(host: str, run: LinkCommandRunner) -> tuple[str, ...]:
    """Readable WARP split-tunnel exclusions, or () when locked/absent."""

    for binary in ("/usr/local/bin/warp-cli", "warp-cli"):
        result = run(host, (binary, "settings"))
        if result.returncode == 0:
            return parse_warp_settings(result.stdout or "")
    return ()


def detect_vpn(
    host: str,
    *,
    runner: LinkCommandRunner | None = None,
    interfaces: HostInterfaces | None = None,
) -> VPNProfile:
    """One host's VPN posture, read heuristically and failure-tolerantly.

    Heuristics in order: utun interfaces from the live addressing (pass
    ``interfaces`` when a fresh ``probe_host_interfaces`` reading is already in
    hand); default//1+/1 routes via ``netstat -rn``; client app signatures,
    then a readable WARP exclusion list. A host with no IPv4-addressed utun
    and no utun routes is reported clean without further remote reads.

    This never promotes the transport ladder — it only warns and enriches
    candidate selection. Any read failure degrades to ``client="unknown"``
    with empty exclusions rather than raising.
    """

    run = runner or _run_link_command
    if interfaces is None:
        with suppress(RuntimeError, OSError, subprocess.SubprocessError):
            interfaces = probe_host_interfaces(host)
    utuns = _utun_interfaces(interfaces)
    if interfaces is not None and not utuns:
        # A clean addressing read is decisive enough to skip the remote route
        # and signature reads: every VPN client here addresses its utun.
        return VPNProfile()

    try:
        routes = _routes(host, run)
        utuns |= {
            netif for _, netif in routes if netif.startswith("utun")
        }
        if not utuns:
            return VPNProfile()
        client = _client_signature(host, run, interfaces)
        exclusions: tuple[str, ...] = ()
        if client in ("warp", ""):
            exclusions = _warp_exclusions(host, run)
            if exclusions and not client:
                client = "warp"
        return VPNProfile(
            present=True,
            client=client or "unknown",
            full_tunnel=_full_tunnel(routes),
            utun_interfaces=tuple(sorted(utuns)),
            exclusions=exclusions,
        )
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
        return VPNProfile(
            present=True,
            client="unknown",
            utun_interfaces=tuple(sorted(utuns)),
        )


def hostile_networks(
    hosts: Iterable[str],
    *,
    probe: Callable[[str], HostInterfaces] | None = None,
    runner: LinkCommandRunner | None = None,
    interfaces: Mapping[str, HostInterfaces] | None = None,
) -> tuple[ipaddress.IPv4Network, ...]:
    """Interface subnets ∪ utun-routed prefixes across the given hosts.

    The superset of ``transport._occupied_networks``: a route sending
    10.0.0.0/8 through a utun claims the whole /8 for the tunnel even though
    no interface carries an address there — the exact shape of the WARP
    incident. Tunnel catch-alls (default, the /1 pair) are full-tunnel
    evidence, not claimed ranges, and are deliberately not counted: counting
    them would veto every candidate and the empirical probes are what decide
    whether a chosen link actually carries traffic. Probe failures are
    non-fatal, matching ``_occupied_networks``.
    """

    probe = probe or probe_host_interfaces
    run = runner or _run_link_command
    nets: set[ipaddress.IPv4Network] = set()
    for host in hosts:
        host_interfaces = (interfaces or {}).get(host)
        if host_interfaces is None:
            with suppress(RuntimeError, OSError, subprocess.SubprocessError):
                host_interfaces = probe(host)
        if host_interfaces is not None:
            for entry in host_interfaces.addresses:
                with suppress(ValueError):
                    nets.add(entry.network)
        for destination, netif in _routes(host, run):
            if (
                netif.startswith("utun")
                and destination.prefixlen >= _CATCH_ALL_PREFIX_LENGTH
            ):
                nets.add(destination)
    return tuple(nets)
