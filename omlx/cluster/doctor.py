# SPDX-License-Identifier: Apache-2.0
"""The Fabric Doctor: the ladder checks, run in order, stopping at the first red.

One guided flow turns the diagnosis knowledge spread across ``transport``,
``vpn``, and ``collective`` into a single ordered report. Each check is pure
over injected probe results (the ``assess_link`` injection pattern), so every
branch is testable without two Macs — and the default probes are only ever
exercised by an explicit, user-initiated Doctor run.

Design notes that shipped with this file:

- Check order is the readiness-ladder order (``readiness.LADDER_ORDER``):
  link/address sanity, subnet collision, routes + bound connect, the JACCL
  collective probe, then RDMA staleness + admin port. The Doctor stops at the
  first red rung; later checks are reported ``skipped`` because they would
  only report consequences of the first failure.
- The collective probe hard-refuses to run while a deployment is registered
  for the target hosts: probing live collective ports would perturb a serving
  cluster. The refusal is a named ``skipped`` finding, never a silent pass.
- Exit-code diagnosis: JACCL failures carrying ``error: 60`` (ETIMEDOUT) are
  a firewall/VPN silently dropping the fabric path; ``error: 61``
  (ECONNREFUSED) is a launch-order condition where retrying is correct — the
  table below converts failure #6's silent exponential backoff into one named
  finding.
"""

from __future__ import annotations

import ipaddress
import json
import re
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .collective import (
    CollectiveSmokeError,
    LauncherRunner,
    _find_loopback_port_span,
    _parse_collective_records,
    _run_launcher,
)
from .transport import (
    HostInterfaces,
    LinkCommandRunner,
    SharedLink,
    _run_link_command,
    probe_host_interfaces,
    shared_link_addresses,
    verify_link_reachability,
)
from .vpn import VPNProfile, detect_vpn, hostile_networks

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}

# The peer-side worker runtime strings launch.py reports; a missing runtime
# degrades the collective probe to "skipped", it does not fail the fabric.
_RUNTIME_MISSING = "oMLX worker runtime is not installed"

# errno → (name, plain-language diagnosis, remedy or None). Scanned out of the
# launcher's stderr as ``error: <n>``; anything unlisted gets the generic copy
# plus the raw detail so nothing is swallowed.
ERRNO_DIAGNOSES: dict[int, tuple[str, str, str | None]] = {
    60: (
        "ETIMEDOUT",
        "the connection is being silently dropped — a firewall or VPN on the "
        "fabric path",
        "Click Start Cluster again to move the link to a subnet the VPN "
        "ignores, or add a VPN split-tunnel exclusion for the fabric range.",
    ),
    61: (
        "ECONNREFUSED",
        "the peer's worker isn't listening yet — a launch-order problem, not "
        "a network problem; retrying is correct",
        None,
    ),
}

_ERRNO_PATTERN = re.compile(r"error:\s*(\d+)")


class FabricProbeRefusedError(RuntimeError):
    """The collective probe refused to run beside an active deployment."""


@dataclass(frozen=True)
class DoctorFinding:
    """One named result row: check · state · evidence · diagnosis · fix."""

    check_id: str  # "link_presence" | "address_sanity" | "subnet_collision"
    # | "route_pinning" | "bound_connect" | "jaccl_probe"
    # | "rdma_staleness" | "admin_port"
    # "warn" is for a finding that is real but not a fabric fault -- admin_port
    # is the one user today (a heuristic, best-effort read whose failure only
    # means planning falls back to a slower path): it must not stop the
    # ladder, gate DoctorReport.ok, or become the verdict's first_red (#2878
    # review).
    state: str  # "pass" | "fail" | "warn" | "skipped"
    evidence: str
    diagnosis: str = ""
    remedy: str = ""
    fix_action: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "state": self.state,
            "evidence": self.evidence,
            "diagnosis": self.diagnosis,
            "remedy": self.remedy,
            "fix_action": dict(self.fix_action) if self.fix_action else None,
        }


@dataclass(frozen=True)
class DoctorCheck:
    """One ladder rung the Doctor exercises, and the finding rows it owns."""

    name: str
    title: str
    finding_ids: tuple[str, ...]


# Ladder order. A red rung stops the run; the remaining checks' finding rows
# are reported ``skipped`` rather than guessed at.
DOCTOR_CHECKS: tuple[DoctorCheck, ...] = (
    DoctorCheck(
        name="link_address_sanity",
        title="Thunderbolt link and address sanity",
        finding_ids=("link_presence", "address_sanity"),
    ),
    DoctorCheck(
        name="subnet_collision",
        title="Fabric subnet collision",
        finding_ids=("subnet_collision",),
    ),
    DoctorCheck(
        name="reachability",
        title="Route pinning and bound TCP",
        finding_ids=("route_pinning", "bound_connect"),
    ),
    DoctorCheck(
        name="jaccl_probe",
        title="Two-rank JACCL collective probe",
        finding_ids=("jaccl_probe",),
    ),
    DoctorCheck(
        name="staleness_admin",
        title="RDMA address staleness and admin port",
        finding_ids=("rdma_staleness", "admin_port"),
    ),
)

_SKIPPED_DIAGNOSIS = (
    "Later checks would only report consequences of the first failure."
)
_SKIPPED_REMEDY = "Fix the first failing check, then run the Doctor again."


@dataclass(frozen=True)
class DoctorReport:
    """The full ordered run: every finding row plus one human verdict line."""

    hosts: tuple[str, ...]
    findings: tuple[DoctorFinding, ...]
    verdict: str
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def ok(self) -> bool:
        return not any(finding.state == "fail" for finding in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hosts": list(self.hosts),
            "findings": [finding.to_dict() for finding in self.findings],
            "verdict": self.verdict,
            "ok": self.ok,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


def _registered_deployments() -> tuple[Any, ...]:
    """Active deployments, fail-soft: an unconfigured registry means none."""

    from .registry import get_cluster_registry

    try:
        return tuple(get_cluster_registry().list())
    except RuntimeError:
        return ()


def _local_admin_port_health() -> tuple[bool, str]:
    """Is this node's own admin API answering on its configured port?

    A heuristic health read only (C5 threads the peer's advertised port); any
    inability to determine the port degrades to a pass with honest evidence.
    """

    try:
        from ..settings import get_settings

        port = int(get_settings().server.port)
    except Exception:  # noqa: BLE001 - settings absent in worker-only installs
        return True, "admin port not checked — server settings unavailable"
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            return True, f"the admin API is answering on port {port}"
    except OSError:
        return False, (
            f"nothing is answering on admin port {port} — Start will still "
            "work but planning will be slower"
        )


@dataclass(frozen=True)
class DoctorProbes:
    """Everything the Doctor reads, injectable so every check is pure.

    The defaults reach real hosts over the paired SSH identity; tests inject
    fakes for all of them (the ``assess_link`` injection pattern).
    """

    interfaces: Callable[[str], HostInterfaces] = probe_host_interfaces
    vpn: Callable[..., VPNProfile] = detect_vpn
    hostile: Callable[..., tuple[ipaddress.IPv4Network, ...]] = hostile_networks
    shared_link: Callable[..., SharedLink] = shared_link_addresses
    verify: Callable[[SharedLink], tuple[bool, str]] = verify_link_reachability
    collective: Callable[..., dict[str, Any]] = None  # type: ignore[assignment]
    deployments: Callable[[], tuple[Any, ...]] = _registered_deployments
    admin_port: Callable[[], tuple[bool, str]] = _local_admin_port_health

    def __post_init__(self) -> None:
        if self.collective is None:
            object.__setattr__(self, "collective", run_fabric_collective_probe)


def _normalized_host(host: str) -> str:
    normalized = host.strip().lower()
    return "127.0.0.1" if normalized in _LOCAL_HOSTS else normalized


def _active_deployment_for(
    hosts: Sequence[str], deployments: Iterable[Any]
) -> Any | None:
    """The first registered deployment sharing a host with this Doctor run.

    Rank zero of every deployment is the coordinator itself (``127.0.0.1``),
    so a Doctor run that includes the local Mac refuses beside *any* active
    deployment — which is the safe reading: the probe and the collective
    would contend for the same fabric.
    """

    targets = {_normalized_host(host) for host in hosts}
    for deployment in deployments:
        deployed = {
            _normalized_host(str(getattr(host, "ssh", host)))
            for host in getattr(deployment, "hosts", ())
        }
        if targets & deployed:
            return deployment
    return None


def _validate_rdma_matrix(
    rdma_matrix: Sequence[Sequence[Any]],
) -> None:
    """Mirror the JACCL matrix contract ``ClusterDeployment`` validates."""

    if len(rdma_matrix) != 2:
        raise ValueError("the fabric probe needs one RDMA matrix row per host")
    for rank, row in enumerate(rdma_matrix):
        if len(row) != 2:
            raise ValueError("JACCL requires a full RDMA connectivity matrix")
        if row[rank] is not None:
            raise ValueError("JACCL RDMA matrix diagonal must be null")
        for index, path in enumerate(row):
            if index == rank:
                continue
            if path is None:
                raise ValueError("JACCL RDMA matrix is missing a peer path")


# mlx._distributed_utils.launch's ring backend assigns one port per rank,
# globally sequential across every host in hostfile order, not one shared
# port re-used per host: rank 0 (the coordinator, hosts[0]) binds
# starting_port, rank 1 (the peer, hosts[1]) binds starting_port + 1. A span
# sized for a single rank under-covers a two-rank launch, and checking only
# the coordinator's own loopback never proves anything about the port the
# peer actually binds (#2878 review).
_PORT_CHECK_SCRIPT = (
    "import socket,sys\n"
    "s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)\n"
    "s.bind(('0.0.0.0',int(sys.argv[1])))\n"
    "s.close()"
)
_PORT_SPAN_ATTEMPTS = 8


def _remote_port_free(
    host: str,
    port: int,
    *,
    runner: LinkCommandRunner,
) -> bool:
    """Best-effort bind check for one port on a remote host.

    A TOCTOU race remains between this check and the real launch -- the same
    tradeoff ``choose_fabric_subnet`` already accepts for the fabric range
    itself: a port verified free moments ago is still better odds than one
    never checked at all.
    """

    result = runner(
        host, ("/usr/bin/python3", "-c", _PORT_CHECK_SCRIPT, str(port))
    )
    return result.returncode == 0


def _reserve_port_span(
    hosts: Sequence[str],
    *,
    port_check_runner: LinkCommandRunner,
) -> int:
    """Find a starting port whose per-rank span is free on every host.

    hosts[i] binds starting_port + i (see the ring-backend note above).
    Local ports are reserved with a real bind-and-release the same way
    ``_find_loopback_port_span`` always has; remote ports are checked
    best-effort over the same SSH identity every other Doctor probe uses.
    """

    last_conflict = ""
    for _ in range(_PORT_SPAN_ATTEMPTS):
        starting_port = _find_loopback_port_span(len(hosts))
        conflict = next(
            (
                host
                for offset, host in enumerate(hosts)
                if host not in _LOCAL_HOSTS
                and not _remote_port_free(
                    host, starting_port + offset, runner=port_check_runner
                )
            ),
            None,
        )
        if conflict is None:
            return starting_port
        last_conflict = conflict
    raise CollectiveSmokeError(
        "could not find a port span free on both the coordinator and "
        f"{last_conflict or 'the peer'} after "
        f"{_PORT_SPAN_ATTEMPTS} attempts"
    )


def run_fabric_collective_probe(
    hosts: Sequence[str],
    addresses: Sequence[str],
    rdma_matrix: Sequence[Sequence[Any]],
    timeout: float = 10.0,
    *,
    runner: LauncherRunner = _run_launcher,
    port_check_runner: LinkCommandRunner = _run_link_command,
    deployments: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """A minimal two-rank JACCL handshake across the fabric addresses.

    Launched the way ``run_local_collective_smoke`` launches loopback ranks,
    but with a hostfile naming the two hosts, their fabric IPs, and the same
    RDMA-matrix shape the real deployment path validates. Refuses outright —
    before any process is spawned — while a deployment is registered for
    these hosts, because a probe on live collective ports would perturb the
    serving collective.
    """

    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    if len(hosts) != 2:
        raise ValueError("the fabric probe runs across exactly two hosts")
    if len(addresses) != 2:
        raise ValueError("the fabric probe needs one address per host")
    for address in addresses:
        ipaddress.IPv4Address(address)
    _validate_rdma_matrix(rdma_matrix)

    active = _active_deployment_for(
        hosts,
        _registered_deployments() if deployments is None else deployments,
    )
    if active is not None:
        deployment_id = getattr(active, "deployment_id", "unknown")
        model = getattr(active, "model", "")
        described = f"{deployment_id} ({model})" if model else deployment_id
        raise FabricProbeRefusedError(
            f"deployment {described} is active on these hosts — a fabric "
            "probe on live collective ports would perturb the serving "
            "collective. Stop the deployment first."
        )

    starting_port = _reserve_port_span(hosts, port_check_runner=port_check_runner)
    hostfile_payload = {
        "backend": "jaccl",
        "envs": ["MLX_METAL_FAST_SYNCH=1"],
        "hosts": [
            {
                "ssh": str(host),
                "ips": [str(address)],
                "rdma": [
                    list(path) if isinstance(path, tuple) else path
                    for path in row
                ],
            }
            for host, address, row in zip(hosts, addresses, rdma_matrix)
        ],
    }
    launcher = (
        "from mlx._distributed_utils.launch import main; raise SystemExit(main() or 0)"
    )
    started_at = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="omlx-fabric-doctor-") as temporary:
        hostfile = Path(temporary) / "hostfile.json"
        hostfile.write_text(
            json.dumps(hostfile_payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        argv = [
            sys.executable,
            "-c",
            launcher,
            "--hostfile",
            str(hostfile),
            "--starting-port",
            str(starting_port),
            "--",
            sys.executable,
            "-m",
            "omlx.cluster.collective_worker",
        ]
        try:
            completed = runner(argv, timeout=timeout)
        except (OSError, subprocess.SubprocessError) as exc:
            raise CollectiveSmokeError(
                f"could not launch the fabric probe: {exc}"
            ) from exc

    detail = completed.stderr.strip()
    if completed.returncode != 0:
        suffix = f": {detail}" if detail else ""
        raise CollectiveSmokeError(
            f"fabric probe launcher exited with code {completed.returncode}{suffix}"
        )
    records = {
        record["rank"]: record
        for record in _parse_collective_records(completed.stdout)
        if isinstance(record.get("rank"), int)
    }
    if set(records) != {0, 1}:
        suffix = f": {detail}" if detail else ""
        raise CollectiveSmokeError(
            "fabric probe did not return one result from each rank" + suffix
        )
    for rank, record in records.items():
        if record.get("size") != 2 or record.get("sum") != 3:
            raise CollectiveSmokeError(
                f"rank {rank} returned an invalid fabric probe result: {record}"
            )
    elapsed = time.monotonic() - started_at
    bandwidth = None
    for record in records.values():
        with suppress(TypeError, ValueError):
            if record.get("bandwidth_gbps") is not None:
                bandwidth = float(record["bandwidth_gbps"])
    return {
        "ok": True,
        "backend": "jaccl",
        "rank_count": 2,
        "hosts": [str(host) for host in hosts],
        "addresses": [str(address) for address in addresses],
        "starting_port": starting_port,
        "elapsed_seconds": elapsed,
        "bandwidth_gbps": bandwidth,
    }


# --- pure check functions -------------------------------------------------


def _fabric_interfaces(interfaces: HostInterfaces) -> frozenset[str]:
    return interfaces.rdma_interfaces | interfaces.thunderbolt_interfaces


def _fabric_addresses(interfaces: HostInterfaces):
    fabric = _fabric_interfaces(interfaces)
    return tuple(
        entry for entry in interfaces.addresses if entry.interface in fabric
    )


def check_link_address_sanity(
    hosts: Sequence[str],
    interfaces: Mapping[str, HostInterfaces | None],
) -> list[DoctorFinding]:
    """Check 1: the Thunderbolt link exists and its addresses are usable."""

    findings: list[DoctorFinding] = []

    missing = [host for host in hosts if interfaces.get(host) is None]
    portless = [
        host
        for host in hosts
        if interfaces.get(host) is not None
        and not _fabric_interfaces(interfaces[host])  # type: ignore[index]
    ]
    if missing:
        findings.append(
            DoctorFinding(
                check_id="link_presence",
                state="fail",
                evidence=(
                    "could not read interface state on "
                    + ", ".join(sorted(missing))
                ),
                diagnosis=(
                    "the host did not answer the interface probe, so no link "
                    "evidence exists at all"
                ),
                remedy=(
                    "Check the Mac is awake and reachable over SSH, then run "
                    "the Doctor again."
                ),
            )
        )
    elif portless:
        findings.append(
            DoctorFinding(
                check_id="link_presence",
                state="fail",
                evidence=(
                    ", ".join(sorted(portless))
                    + " report no Thunderbolt or RDMA interfaces"
                ),
                diagnosis="no Thunderbolt peer link is visible on the host",
                remedy=(
                    "Connect a Thunderbolt 5 cable between the Macs and check "
                    "both are awake."
                ),
            )
        )
    else:
        described = []
        for host in hosts:
            probed = interfaces[host]
            assert probed is not None
            names = sorted(_fabric_interfaces(probed))
            described.append(f"{host}: {', '.join(names)}")
        findings.append(
            DoctorFinding(
                check_id="link_presence",
                state="pass",
                evidence="; ".join(described),
            )
        )

    if findings[-1].state == "fail":
        # No link at all — address sanity has nothing to inspect.
        findings.append(
            DoctorFinding(
                check_id="address_sanity",
                state="skipped",
                evidence="no fabric interfaces to inspect",
                diagnosis=_SKIPPED_DIAGNOSIS,
                remedy=_SKIPPED_REMEDY,
            )
        )
        return findings

    unaddressed: list[str] = []
    renumbered: list[str] = []
    for host in hosts:
        probed = interfaces[host]
        assert probed is not None
        # A 169.254 self-assigned address can never appear here to detect
        # directly: probe_host_interfaces already drops _UNROUTABLE_NETWORKS
        # entries (including 169.254.0.0/16) at the source, so a check
        # looking for one on probed.addresses can never fire (#2878 review).
        # What that failure actually looks like once filtered is an RDMA
        # device with NO address at all -- the same shape rdma_staleness
        # (check 5) detects, moved here so the ladder stops with the right
        # diagnosis immediately instead of limping through checks 2-4 first
        # on a link that was never really addressed. Split from interface
        # renumbering below (en6 → en4 after replug/reboot, where an
        # orphaned address still sits on the old Thunderbolt name) --
        # those two share the same "RDMA device unaddressed" trigger but
        # need different evidence and diagnosis text.
        if not probed.rdma_interfaces or any(
            entry.interface in probed.rdma_interfaces
            for entry in probed.addresses
        ):
            continue
        orphaned = [
            str(entry)
            for entry in probed.addresses
            if entry.interface in probed.thunderbolt_interfaces
        ]
        if orphaned:
            renumbered.append(f"{host}: {', '.join(orphaned)}")
        else:
            unaddressed.append(
                f"{host} has RDMA devices "
                f"({', '.join(sorted(probed.rdma_interfaces))}) but no "
                "fabric address"
            )

    if unaddressed:
        findings.append(
            DoctorFinding(
                check_id="address_sanity",
                state="fail",
                evidence="; ".join(unaddressed),
                diagnosis=(
                    "macOS never finished configuring the link, or a "
                    "169.254 self-assigned address was applied and could "
                    "not be used (this commonly appears after a reboot)"
                ),
                remedy="Click Start Cluster again — it re-addresses the link automatically.",
                fix_action={"kind": "readdress", "hosts": list(hosts)},
            )
        )
    elif renumbered:
        findings.append(
            DoctorFinding(
                check_id="address_sanity",
                state="fail",
                evidence=(
                    "addresses sit on a Thunderbolt interface that is no "
                    "longer the RDMA device: " + "; ".join(renumbered)
                ),
                diagnosis=(
                    "the Thunderbolt interface was renumbered (for example "
                    "en6 → en4 after a replug or reboot), leaving the fabric "
                    "address on the old interface name"
                ),
                remedy="Click Start Cluster again — it re-addresses the link automatically.",
                fix_action={"kind": "readdress", "hosts": list(hosts)},
            )
        )
    else:
        addressed = [
            f"{host}: "
            + (
                ", ".join(
                    str(entry)
                    for entry in _fabric_addresses(interfaces[host])  # type: ignore[arg-type]
                )
                or "no fabric addresses yet"
            )
            for host in hosts
        ]
        findings.append(
            DoctorFinding(
                check_id="address_sanity",
                state="pass",
                evidence="; ".join(addressed),
            )
        )
    return findings


def check_subnet_collision(
    hosts: Sequence[str],
    interfaces: Mapping[str, HostInterfaces | None],
    hostile: Sequence[ipaddress.IPv4Network],
    vpn_profiles: Mapping[str, VPNProfile],
) -> list[DoctorFinding]:
    """Check 2: nothing else the Macs carry or route claims the fabric range."""

    fabric_networks: set[ipaddress.IPv4Network] = set()
    interface_networks: set[ipaddress.IPv4Network] = set()
    for host in hosts:
        probed = interfaces.get(host)
        if probed is None:
            continue
        for entry in probed.addresses:
            with suppress(ValueError):
                interface_networks.add(entry.network)
        for entry in _fabric_addresses(probed):
            with suppress(ValueError):
                fabric_networks.add(entry.network)

    if not fabric_networks:
        return [
            DoctorFinding(
                check_id="subnet_collision",
                state="pass",
                evidence="no fabric addresses assigned yet — nothing to collide",
            )
        ]

    # The link's own subnets are not collisions with themselves. Excluded
    # when a hostile entry IS (or is narrower than) a fabric network, not
    # merely overlapping it (transport.configure_link's own_networks had
    # the same equality-vs-real-collision gap — #2875 review; plain
    # overlap over-excludes here too: a hostile entry broader than the
    # fabric subnet, like a utun routing all of 10/8 while the fabric
    # sits at 10.0.1.0/24, is a real collision risk this check exists to
    # catch, not something to wave away just because it happens to cover
    # the fabric's own range).
    collision_set = [
        net
        for net in hostile
        if not any(net == own or net.subnet_of(own) for own in fabric_networks)
    ]
    tunnel_routed = {
        net for net in collision_set if net not in interface_networks
    }
    clients = sorted(
        {
            profile.client
            for profile in vpn_profiles.values()
            if profile.present and profile.client not in ("", "unknown")
        }
    )
    utuns = sorted(
        {
            utun
            for profile in vpn_profiles.values()
            for utun in profile.utun_interfaces
        }
    )

    for fabric_net in sorted(fabric_networks, key=str):
        for other in collision_set:
            if not fabric_net.overlaps(other):
                continue
            if other in tunnel_routed:
                client = (clients[0].upper() if clients else "A VPN")
                via = f" through {', '.join(utuns)}" if utuns else ""
                evidence = f"{client} routes {other}{via}"
                diagnosis = (
                    f"the fabric subnet {fabric_net} is captured by a VPN "
                    "tunnel route — traffic addressed there leaves over the "
                    "tunnel instead of the Thunderbolt cable"
                )
            else:
                evidence = f"{other} is already in use on a LAN interface"
                diagnosis = (
                    f"the fabric subnet {fabric_net} overlaps a network one "
                    "of the Macs already uses, so routes are ambiguous"
                )
            return [
                DoctorFinding(
                    check_id="subnet_collision",
                    state="fail",
                    evidence=evidence,
                    diagnosis=diagnosis,
                    remedy=(
                        "Click Start Cluster again — it will pick a "
                        "different, collision-free subnet automatically."
                    ),
                    fix_action={"kind": "move_subnet", "hosts": list(hosts)},
                )
            ]

    return [
        DoctorFinding(
            check_id="subnet_collision",
            state="pass",
            evidence=(
                ", ".join(str(net) for net in sorted(fabric_networks, key=str))
                + " collides with nothing the Macs carry or route"
            ),
        )
    ]


def check_reachability(
    link: SharedLink | None,
    verify_result: tuple[bool, str] | None,
) -> list[DoctorFinding]:
    """Check 3: the route pins to the fabric interface and bound TCP answers."""

    if link is None or not link.ok:
        reason = getattr(link, "reason", "") or "no shared fabric link"
        return [
            DoctorFinding(
                check_id="route_pinning",
                state="fail",
                evidence=reason,
                diagnosis=(
                    "the two Macs share no verified fabric addressing, so "
                    "there is no route to check"
                ),
                remedy="Click Start Cluster again — it re-addresses the link automatically.",
                fix_action={"kind": "readdress"},
            ),
            DoctorFinding(
                check_id="bound_connect",
                state="skipped",
                evidence="no route to test a bound connection over",
                diagnosis=_SKIPPED_DIAGNOSIS,
                remedy=_SKIPPED_REMEDY,
            ),
        ]

    ok, detail = verify_result if verify_result is not None else (False, "")
    endpoints = ""
    if link.source is not None and link.peer is not None:
        endpoints = f"{link.source.address} ⇄ {link.peer.address}"
    if ok:
        return [
            DoctorFinding(
                check_id="route_pinning",
                state="pass",
                evidence=(
                    "both directions route over the fabric interface"
                    + (f" ({endpoints})" if endpoints else "")
                ),
            ),
            DoctorFinding(
                check_id="bound_connect",
                state="pass",
                evidence=(
                    "bound TCP connections succeeded in both directions"
                    + (f" ({endpoints})" if endpoints else "")
                ),
            ),
        ]

    route_failure = "cannot use" in detail or "no usable route" in detail
    if route_failure:
        return [
            DoctorFinding(
                check_id="route_pinning",
                state="fail",
                evidence=detail,
                diagnosis=(
                    "the operating system is not routing the peer's fabric "
                    "address over the Thunderbolt interface"
                ),
                remedy=(
                    "Click Start Cluster again — it re-addresses the link "
                    "automatically, clearing the stale route."
                ),
                fix_action={"kind": "readdress"},
            ),
            DoctorFinding(
                check_id="bound_connect",
                state="skipped",
                evidence="the route check failed first",
                diagnosis=_SKIPPED_DIAGNOSIS,
                remedy=_SKIPPED_REMEDY,
            ),
        ]

    firewall = "firewall or VPN" in detail
    return [
        DoctorFinding(
            check_id="route_pinning",
            state="pass",
            evidence=(
                "both directions route over the fabric interface"
                + (f" ({endpoints})" if endpoints else "")
            ),
        ),
        DoctorFinding(
            check_id="bound_connect",
            state="fail",
            evidence=detail,
            diagnosis=(
                "a firewall or VPN is dropping TCP on the fabric path while "
                "letting ping through"
                if firewall
                else "the peer did not answer on its fabric address"
            ),
            remedy=(
                "Allow TCP between the fabric addresses (firewall/VPN "
                "settings apply to all interfaces), or move the link to a "
                "subnet the VPN ignores."
                if firewall
                else "Check the peer Mac is awake, then re-run the Doctor."
            ),
            fix_action={"kind": "move_subnet"} if firewall else None,
        ),
    ]


def check_jaccl_probe(
    hosts: Sequence[str],
    link: SharedLink | None,
    interfaces: Mapping[str, HostInterfaces | None],
    *,
    collective: Callable[..., dict[str, Any]],
    deployments: tuple[Any, ...],
    timeout: float = 10.0,
) -> tuple[DoctorFinding, dict[str, Any] | None]:
    """Check 4: a real two-rank JACCL handshake, errno-mapped on failure.

    Returns the finding plus the probe result (for the verdict's bandwidth).
    The active-deployment refusal happens here, before ``collective`` is
    invoked, and again inside ``run_fabric_collective_probe`` — defense in
    depth for the one check that launches real processes.
    """

    active = _active_deployment_for(hosts, deployments)
    if active is not None:
        deployment_id = getattr(active, "deployment_id", "unknown")
        model = getattr(active, "model", "")
        described = f"{deployment_id} ({model})" if model else str(deployment_id)
        return (
            DoctorFinding(
                check_id="jaccl_probe",
                state="skipped",
                evidence=(
                    f"deployment {described} is active on these hosts — a "
                    "probe on live collective ports would perturb it"
                ),
                diagnosis=(
                    "the collective probe never runs beside a serving "
                    "deployment"
                ),
                remedy=(
                    "Stop the active deployment, then run the Doctor again."
                ),
            ),
            None,
        )

    if link is None or link.source is None or link.peer is None:
        return (
            DoctorFinding(
                check_id="jaccl_probe",
                state="skipped",
                evidence="no verified fabric addresses to probe over",
                diagnosis=_SKIPPED_DIAGNOSIS,
                remedy=_SKIPPED_REMEDY,
            ),
            None,
        )

    addresses = (link.source.address, link.peer.address)
    rdma_matrix = (
        (None, f"rdma_{link.source.interface}"),
        (f"rdma_{link.peer.interface}", None),
    )
    try:
        result = collective(
            tuple(hosts), addresses, rdma_matrix, timeout=timeout
        )
    except FabricProbeRefusedError as exc:
        return (
            DoctorFinding(
                check_id="jaccl_probe",
                state="skipped",
                evidence=str(exc),
                diagnosis=(
                    "the collective probe never runs beside a serving "
                    "deployment"
                ),
                remedy=(
                    "Stop the active deployment, then run the Doctor again."
                ),
            ),
            None,
        )
    except (CollectiveSmokeError, OSError, RuntimeError, ValueError) as exc:
        message = str(exc)
        if _RUNTIME_MISSING in message:
            return (
                DoctorFinding(
                    check_id="jaccl_probe",
                    state="skipped",
                    evidence=f"worker runtime missing: {message}",
                    diagnosis=(
                        "the peer has no oMLX worker runtime to answer the "
                        "probe — this is an install gap, not a fabric fault"
                    ),
                    remedy=(
                        "Install the oMLX worker runtime on the peer, then "
                        "run the Doctor again."
                    ),
                ),
                None,
            )
        matched = _ERRNO_PATTERN.search(message)
        if matched:
            errno_value = int(matched.group(1))
            known = ERRNO_DIAGNOSES.get(errno_value)
            if known is not None:
                name, diagnosis, remedy = known
                return (
                    DoctorFinding(
                        check_id="jaccl_probe",
                        state="fail",
                        evidence=f"the collective probe failed with error "
                        f"{errno_value} ({name})",
                        diagnosis=diagnosis,
                        remedy=remedy or "",
                        fix_action=(
                            {"kind": "move_subnet", "hosts": list(hosts)}
                            if errno_value == 60
                            else None
                        ),
                    ),
                    None,
                )
            return (
                DoctorFinding(
                    check_id="jaccl_probe",
                    state="fail",
                    evidence=(
                        f"the collective probe failed with error "
                        f"{errno_value}: {message}"
                    ),
                    diagnosis=(
                        "the two-rank collective handshake failed with an "
                        "error the Doctor does not recognize"
                    ),
                    remedy=(
                        "Download the diagnostics bundle and file the raw "
                        "detail with a support report."
                    ),
                ),
                None,
            )
        return (
            DoctorFinding(
                check_id="jaccl_probe",
                state="fail",
                evidence=message,
                diagnosis="the two-rank collective handshake did not complete",
                remedy=(
                    "Check both Macs are awake and the fabric checks above "
                    "are green, then run the Doctor again."
                ),
            ),
            None,
        )

    bandwidth = result.get("bandwidth_gbps")
    elapsed = result.get("elapsed_seconds")
    if bandwidth:
        evidence = f"two-rank JACCL handshake passed — {bandwidth:.0f} Gb/s"
    elif isinstance(elapsed, (int, float)):
        evidence = f"two-rank JACCL handshake passed in {elapsed:.2f}s"
    else:
        evidence = "two-rank JACCL handshake passed"
    return (
        DoctorFinding(check_id="jaccl_probe", state="pass", evidence=evidence),
        result,
    )


def check_staleness_admin(
    hosts: Sequence[str],
    interfaces: Mapping[str, HostInterfaces | None],
    admin_result: tuple[bool, str],
) -> list[DoctorFinding]:
    """Check 5: RDMA devices without addresses, and the admin port answers."""

    findings: list[DoctorFinding] = []
    lost: list[str] = []
    for host in hosts:
        probed = interfaces.get(host)
        if probed is None or not probed.rdma_interfaces:
            continue
        if not any(
            entry.interface in probed.rdma_interfaces
            for entry in probed.addresses
        ):
            lost.append(
                f"{host} has RDMA devices "
                f"({', '.join(sorted(probed.rdma_interfaces))}) but no fabric "
                "addresses"
            )
    if lost:
        findings.append(
            DoctorFinding(
                check_id="rdma_staleness",
                state="fail",
                evidence="; ".join(lost),
                diagnosis=(
                    "the link's addresses were lost (this happens after a "
                    "reboot when addressing was applied with ifconfig)"
                ),
                remedy="Click Start Cluster again — it re-addresses the link automatically.",
                fix_action={"kind": "readdress", "hosts": list(hosts)},
            )
        )
    else:
        findings.append(
            DoctorFinding(
                check_id="rdma_staleness",
                state="pass",
                evidence="every RDMA device carries a routable fabric address",
            )
        )

    ok, detail = admin_result
    findings.append(
        DoctorFinding(
            check_id="admin_port",
            # A heuristic best-effort read, not a fabric fault: "fail" here
            # would stop the ladder and gate the whole report on something
            # whose worst case is "planning is a bit slower" (#2878 review).
            state="pass" if ok else "warn",
            evidence=detail,
            diagnosis=(
                ""
                if ok
                else "the fast memory-probe target is not answering — "
                "planning falls back to the slower local computation"
            ),
            remedy=(
                ""
                if ok
                else "Check the oMLX server is running on the advertised "
                "admin port."
            ),
        )
    )
    return findings


def _skipped_findings(check: DoctorCheck, failed_check: str) -> list[DoctorFinding]:
    return [
        DoctorFinding(
            check_id=finding_id,
            state="skipped",
            evidence=f"skipped — {failed_check} failed first",
            diagnosis=_SKIPPED_DIAGNOSIS,
            remedy=_SKIPPED_REMEDY,
        )
        for finding_id in check.finding_ids
    ]


def _verdict(
    findings: Sequence[DoctorFinding],
    probe_result: dict[str, Any] | None,
) -> str:
    first_red = next(
        (finding for finding in findings if finding.state == "fail"), None
    )
    if first_red is not None:
        return (
            f"Fabric Doctor stopped at {first_red.check_id}: "
            f"{first_red.diagnosis or first_red.evidence}"
        )
    # A "warn" finding (admin_port today) is real but never the fabric
    # verdict's headline -- it did not stop the ladder and does not gate
    # DoctorReport.ok, so it rides along as a suffix on whatever verdict
    # the actually-decisive checks produced (#2878 review).
    warnings = [finding for finding in findings if finding.state == "warn"]
    warning_suffix = (
        " (" + "; ".join(finding.evidence for finding in warnings) + ")"
        if warnings
        else ""
    )
    probe = next(
        (finding for finding in findings if finding.check_id == "jaccl_probe"),
        None,
    )
    if probe is not None and probe.state == "skipped":
        return (
            f"No faults found; the collective probe was skipped: "
            f"{probe.evidence}{warning_suffix}"
        )
    if probe_result is not None:
        bandwidth = probe_result.get("bandwidth_gbps")
        if bandwidth:
            return (
                f"Fabric verified — {bandwidth:.0f} Gb/s measured across the "
                f"link.{warning_suffix}"
            )
        elapsed = probe_result.get("elapsed_seconds")
        if isinstance(elapsed, (int, float)):
            return (
                "Fabric verified — the two-rank collective handshake "
                f"completed in {elapsed:.2f}s.{warning_suffix}"
            )
    return f"Fabric verified — every check passed.{warning_suffix}"


def run_fabric_doctor(
    hosts: Sequence[str],
    *,
    probes: DoctorProbes | None = None,
    probe_timeout: float = 10.0,
) -> DoctorReport:
    """Run the ladder checks in order and stop at the first red rung.

    Every probe read is fail-soft: a host that cannot be read produces a
    named failing finding rather than an exception, so the report always
    completes and always carries a verdict.
    """

    if len(hosts) != 2:
        raise ValueError("the Fabric Doctor runs across exactly two hosts")
    hosts = tuple(str(host) for host in hosts)
    probes = probes or DoctorProbes()
    started_at = time.time()

    interfaces: dict[str, HostInterfaces | None] = {}
    for host in hosts:
        try:
            interfaces[host] = probes.interfaces(host)
        except (RuntimeError, OSError, subprocess.SubprocessError):
            interfaces[host] = None

    findings: list[DoctorFinding] = []
    failed_check: str | None = None
    probe_result: dict[str, Any] | None = None
    link: SharedLink | None = None

    for check in DOCTOR_CHECKS:
        if failed_check is not None:
            findings.extend(_skipped_findings(check, failed_check))
            continue

        if check.name == "link_address_sanity":
            produced = check_link_address_sanity(hosts, interfaces)
        elif check.name == "subnet_collision":
            try:
                hostile = tuple(probes.hostile(hosts, interfaces=interfaces))
            except (RuntimeError, OSError, subprocess.SubprocessError):
                hostile = ()
            vpn_profiles: dict[str, VPNProfile] = {}
            for host in hosts:
                try:
                    vpn_profiles[host] = probes.vpn(
                        host, interfaces=interfaces.get(host)
                    )
                except (RuntimeError, OSError, subprocess.SubprocessError):
                    vpn_profiles[host] = VPNProfile()
            produced = check_subnet_collision(
                hosts, interfaces, hostile, vpn_profiles
            )
        elif check.name == "reachability":
            source, peer = interfaces.get(hosts[0]), interfaces.get(hosts[1])
            if source is not None and peer is not None:
                link = probes.shared_link(source, peer)
            verify_result: tuple[bool, str] | None = None
            if link is not None and link.ok:
                try:
                    verify_result = probes.verify(link)
                except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
                    verify_result = (False, str(exc))
            produced = check_reachability(link, verify_result)
        elif check.name == "jaccl_probe":
            try:
                deployments = tuple(probes.deployments())
            except (RuntimeError, OSError):
                deployments = ()
            finding, probe_result = check_jaccl_probe(
                hosts,
                link,
                interfaces,
                collective=probes.collective,
                deployments=deployments,
                timeout=probe_timeout,
            )
            produced = [finding]
        else:
            try:
                admin_result = probes.admin_port()
            except (RuntimeError, OSError):
                admin_result = (True, "admin port not checked")
            produced = check_staleness_admin(hosts, interfaces, admin_result)

        findings.extend(produced)
        if any(finding.state == "fail" for finding in produced):
            failed_check = check.name

    return DoctorReport(
        hosts=hosts,
        findings=tuple(findings),
        verdict=_verdict(findings, probe_result),
        started_at=started_at,
        finished_at=time.time(),
    )
