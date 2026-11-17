# SPDX-License-Identifier: Apache-2.0
"""Read-only network preflight for the local Mac.

The P2 decision is a preflight that **reads** the local interfaces and shows
the exact commands a person would run, with an explicit confirmation before any
change. This module only reads: it never calls ``ifconfig`` with a mutating
action, never opens an administrator dialog and never touches another Mac.

The parsing itself is oMLX's existing ``transport.probe_host_interfaces``; this
module turns that evidence into a small, copyable plan. A bridge plan is a
template, not a promise: macOS does not run STP on its bridges by default, so a
ring is never proposed silently.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from ..transport import HostInterfaces, probe_host_interfaces

# The default bridge name used by macOS networking; the warning says to replace
# it if the name is already in use.
DEFAULT_BRIDGE = "bridge0"

WARNING_READ_FAILED = "interface_read_failed"
WARNING_NO_THUNDERBOLT = "no_thunderbolt_interface"
WARNING_SELF_ASSIGNED = "self_assigned_thunderbolt"
WARNING_RING = "ring_needs_stp"
WARNING_EXISTING_BRIDGE = "existing_bridge"
WARNING_MULTIPLE_PORTS = "multiple_thunderbolt_ports"


def build_network_preflight(
    hosts: Sequence[str] = ("127.0.0.1",),
    *,
    probe: Callable[[str], HostInterfaces] = probe_host_interfaces,
) -> dict[str, Any]:
    """Read each host's interfaces and return commands; never execute them."""

    views: list[dict[str, Any]] = []
    commands: list[str] = []
    warnings: list[str] = []

    for host in hosts:
        try:
            facts = probe(host)
        except Exception as exc:  # a failed read is evidence, not a crash
            warnings.append(f"{WARNING_READ_FAILED}: {host}: {exc}")
            continue

        # ``parse_thunderbolt_interfaces`` includes the Thunderbolt Bridge
        # (``bridge0``) itself. A bridge is not a member of itself; keep the
        # existing bridges separate and never add one to another.
        bridges = sorted(
            item for item in facts.thunderbolt_interfaces if item.startswith("bridge")
        )
        thunderbolt = sorted(
            item
            for item in facts.thunderbolt_interfaces
            if not item.startswith("bridge")
        )
        rdma = sorted(facts.rdma_interfaces)
        views.append(
            {
                "host": host,
                "bridges": bridges,
                "thunderbolt_interfaces": thunderbolt,
                "rdma_interfaces": rdma,
                "addresses": [
                    {
                        "interface": item.interface,
                        "address": item.address,
                        "prefix_length": item.prefix_length,
                    }
                    for item in facts.addresses
                ],
                "self_assigned_interfaces": sorted(
                    facts.self_assigned_interfaces
                ),
                "link_local_addresses": dict(facts.link_local_addresses),
                "media_speeds": dict(facts.media_speeds),
            }
        )

        if not thunderbolt and not bridges:
            warnings.append(
                f"{WARNING_NO_THUNDERBOLT}: {host} has no Thunderbolt interface "
                "in networksetup; use the existing Thunderbolt Bridge or cable "
                "one before planning a cluster"
            )
            continue

        stranded = sorted(facts.self_assigned_interfaces & facts.thunderbolt_interfaces)
        if stranded:
            warnings.append(
                f"{WARNING_SELF_ASSIGNED}: {host} Thunderbolt "
                f"{', '.join(stranded)} only has a self-assigned 169.254 "
                "address; give it a manual IPv4 address on the cluster subnet"
            )

        # One bridge per host is the line shape. If macOS already created the
        # Thunderbolt Bridge, do not create a second one and never add the
        # bridge to itself; only tell the user which ports are candidates.
        commands.append(f"# on {host}")
        if bridges:
            warnings.append(
                f"{WARNING_EXISTING_BRIDGE}: {host} already has "
                f"{', '.join(bridges)}; the preflight will not modify it"
            )
            commands.append(
                f"# {bridges[0]} already exists; do not create a second bridge"
            )
            if thunderbolt:
                commands.append(
                    "# candidate Thunderbolt ports: " + ", ".join(thunderbolt)
                )
                commands.append(
                    f"# add only the port connected to the cluster peer, e.g. "
                    f"sudo ifconfig {bridges[0]} addm {thunderbolt[0]}"
                )
            commands.append(
                "# review the bridge members and address in System Settings > Network"
            )
            continue

        if len(thunderbolt) > 1:
            warnings.append(
                f"{WARNING_MULTIPLE_PORTS}: {host} has Thunderbolt ports "
                f"{', '.join(thunderbolt)}; add only the port connected to the "
                "cluster peer, never all of them"
            )
        commands.append(f"sudo ifconfig {DEFAULT_BRIDGE} create")
        if len(thunderbolt) == 1:
            commands.append(f"sudo ifconfig {DEFAULT_BRIDGE} addm {thunderbolt[0]}")
        else:
            commands.append(
                "# choose one port: "
                + " | ".join(
                    f"sudo ifconfig {DEFAULT_BRIDGE} addm {interface}"
                    for interface in thunderbolt
                )
            )
        commands.append(f"sudo ifconfig {DEFAULT_BRIDGE} up")
        commands.append(
            f"# assign one unique address per Mac, e.g. "
            f"sudo ifconfig {DEFAULT_BRIDGE} 10.10.0.1 netmask 255.255.255.0"
        )

    if len(hosts) > 2:
        warnings.append(
            f"{WARNING_RING}: {len(hosts)} Macs in a ring are a layer-2 loop; "
            "macOS does not run STP on bridges by default. Run the cluster as a "
            "line, or configure STP explicitly on every bridge member"
        )

    if not commands:
        commands.append(
            "# no automatic bridge plan was produced; review the warnings above"
        )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "hosts": views,
        "commands": commands,
        "warnings": warnings,
        "confirmation_required": True,
        # This module never executes; the field is deliberately explicit.
        "executed": False,
    }
