# SPDX-License-Identifier: Apache-2.0
"""Transport detection for distributed cluster peers.

Wraps ``mlx._distributed_utils.config`` to discover available transports
(TB4, TB5, Ethernet, RDMA) without reimplementing the underlying probing.
"""

from __future__ import annotations

import ipaddress
import json
import re
import secrets
import shlex
import subprocess
import threading
from collections.abc import Callable, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import Any

from .ssh_policy import apply_cluster_ssh_policy, cluster_ssh_options

# Link speed thresholds for distinguishing TB4 from TB5
_TB4_MAX_SPEED_GTBS = 40  # TB4 is up to 40 Gb/s
_TB5_MIN_SPEED_GTBS = 80  # TB5 starts at 80 Gb/s
_MLX_CONFIG_RUN_LOCK = threading.Lock()


@dataclass(frozen=True)
class TransportInfo:
    """Information about a transport link to a peer."""

    kind: str  # "thunderbolt", "ethernet", "rdma"
    interface: str
    peer_node_id: str
    # Both endpoints, so rank placement can reason about the fabric as a graph.
    # Without a source, "there is a Thunderbolt link to node B" says nothing
    # about which node it starts from.
    source_node_id: str = ""
    link_speed_gbps: int | None = None
    rdma_device: str | None = None
    tb_version: str | None = None  # "TB4" or "TB5"


@dataclass(frozen=True)
class TransportMatrix:
    """Full transport matrix for a cluster."""

    transports: tuple[TransportInfo, ...]
    backend: str  # "ring", "jaccl", "jaccl-ring"


def _import_mlx_config() -> Any:
    """Import the MLX distributed config module, failing with an actionable error."""

    try:
        import mlx._distributed_utils.config as config

        return config
    except ImportError as exc:
        raise RuntimeError(
            "mlx._distributed_utils.config is unavailable. "
            "Ensure MLX is installed with distributed support."
        ) from exc


@contextmanager
def _mlx_config_ssh_policy(config: Any):
    """Make MLX's hard-coded ``ssh`` subprocess use the cluster policy."""

    original_run = getattr(config, "run", None)
    if original_run is None:
        # Small test doubles and future MLX versions may inject transport by a
        # different seam. Their own implementation remains authoritative.
        yield
        return

    with _MLX_CONFIG_RUN_LOCK:

        def policy_run(argv: list[str], *args: Any, **kwargs: Any):
            if argv and str(argv[0]).rsplit("/", 1)[-1] in {"ssh", "scp"}:
                argv = apply_cluster_ssh_policy(argv, connect_timeout=10)
            return original_run(argv, *args, **kwargs)

        config.run = policy_run
        try:
            yield
        finally:
            config.run = original_run


def _run_ssh(ssh_hostname: str, command: str) -> str:
    """Run a command over SSH and return stdout."""

    result = subprocess.run(
        [
            "ssh",
            *cluster_ssh_options(connect_timeout=10),
            ssh_hostname,
            command,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"SSH command failed on {ssh_hostname}: {command}\n"
            f"stderr: {result.stderr.strip()}"
        )
    return result.stdout


_SPEED = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(gb/s|gbps|mb/s|mbps)", re.IGNORECASE)


def _parse_thunderbolt_speed(speed_str: str) -> int | None:
    """Parse a Thunderbolt link speed string to Gb/s.

    system_profiler reports values like ``"Up to 120 Gb/s"``, so the number
    must be extracted rather than assumed to be the whole string.
    """

    if not speed_str:
        return None
    match = _SPEED.search(speed_str)
    if match is None:
        return None
    value = float(match.group(1))
    if match.group(2).lower().startswith("mb"):
        value /= 1000
    return int(value)


def _detect_tb_version(link_speed_gbps: int | None) -> str | None:
    """Determine TB4 vs TB5 from link speed."""

    if link_speed_gbps is None:
        return None
    if link_speed_gbps >= _TB5_MIN_SPEED_GTBS:
        return "TB5"
    if link_speed_gbps <= _TB4_MAX_SPEED_GTBS:
        return "TB4"
    return None


def _receptacle_speeds(node: Any) -> list[int]:
    """Collect link speeds from a system_profiler bus entry and its children.

    Verified against ``system_profiler SPThunderboltDataType -json`` on macOS:
    the speed lives in ``receptacle_<n>_tag.current_speed_key`` as a string
    like ``"Up to 120 Gb/s"``, not at the top level of the bus entry.
    """

    speeds: list[int] = []
    if not isinstance(node, dict):
        return speeds
    for key, value in node.items():
        if key.endswith("_tag") and isinstance(value, dict):
            parsed = _parse_thunderbolt_speed(str(value.get("current_speed_key", "")))
            if parsed:
                speeds.append(parsed)
    for item in node.get("_items", []):
        speeds.extend(_receptacle_speeds(item))
    return speeds


def _extract_tb_link_speed(ssh_hostname: str) -> int | None:
    """Extract the fastest Thunderbolt link speed reported by a host.

    Callers run inside ``detect_transports``' degradation handler, so failures
    are allowed to propagate rather than being swallowed a second time here.
    """

    output = _run_ssh(ssh_hostname, "system_profiler SPThunderboltDataType -json")
    data = json.loads(output)
    speeds: list[int] = []
    for bus in data.get("SPThunderboltDataType", []):
        speeds.extend(_receptacle_speeds(bus))
    return max(speeds) if speeds else None


_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
_FAST_KINDS = {"thunderbolt", "rdma"}


def _rdma_devices(ssh_hostname: str) -> list[str]:
    """RDMA devices on one host, queried locally when the host is this machine.

    MLX's own ``check_rdma`` shells out to ``ssh <host> ibv_devices`` for every
    host including ``127.0.0.1``. On a Mac without SSH-to-self configured that
    fails, and RDMA is reported as unavailable on hardware that has it — a
    false negative observed on a machine with rdma_en1/en2/en6 present.
    """

    if ssh_hostname in _LOCAL_HOSTS:
        command = ["ibv_devices"]
    else:
        command = [
            "ssh",
            *cluster_ssh_options(connect_timeout=10),
            ssh_hostname,
            "ibv_devices",
        ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"RDMA device probe failed for {ssh_hostname}: {exc}"
        ) from exc
    if getattr(result, "returncode", 0) != 0:
        detail = str(result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"RDMA device probe failed for {ssh_hostname}"
            + (f": {detail[-400:]}" if detail else "")
        )
    return [
        token
        for token in result.stdout.split()
        if token.startswith("rdma_")
    ]


def _rdma_available(hosts: list[str] | tuple[str, ...], *, ssh_prefix: str = "") -> bool:
    """True when every host reports at least one RDMA device."""

    return bool(hosts) and all(
        _rdma_devices(f"{ssh_prefix}{host}") for host in hosts
    )


def detect_transports(
    hosts: list[str] | tuple[str, ...],
    *,
    ssh_prefix: str = "",
) -> tuple[TransportInfo, ...]:
    """Detect available transports for a set of cluster hosts.

    Uses MLX's ``extract_connectivity`` and ``check_rdma`` to discover
    Thunderbolt topology and RDMA availability. Degrades gracefully to
    Ethernet when Thunderbolt is not available.

    Args:
        hosts: List of SSH hostnames for the cluster nodes.
        ssh_prefix: Optional prefix for SSH hostnames (e.g., "user@").

    Returns:
        Tuple of ``TransportInfo`` objects, one per peer link.
    """

    config = _import_mlx_config()

    # Build MLX Host objects
    mlx_hosts = [
        config.Host(
            rank=i,
            ssh_hostname=f"{ssh_prefix}{h}",
            ips=[],
            rdma=[],
        )
        for i, h in enumerate(hosts)
    ]

    transports: list[TransportInfo] = []
    physical_edges: set[tuple[int, int]] = set()

    # Try Thunderbolt connectivity
    try:
        with _mlx_config_ssh_policy(config):
            tb_hosts, uuid_reverse_index = config.extract_connectivity(
                mlx_hosts, verbose=False
            )
        connectivity = config.make_connectivity_matrix(tb_hosts, uuid_reverse_index)

        # Extract transport info from the connectivity matrix
        for i, _host in enumerate(hosts):
            for j, peer_host in enumerate(hosts):
                if i == j:
                    continue
                # Check if there's a TB connection between host i and host j
                if connectivity and i < len(connectivity):
                    row = connectivity[i]
                    if j < len(row) and row[j]:
                        # TB connection exists
                        physical_edges.add((i, j))
                        link_speed = _extract_tb_link_speed(f"{ssh_prefix}{hosts[i]}")
                        tb_version = _detect_tb_version(link_speed)
                        transports.append(
                            TransportInfo(
                                kind="thunderbolt",
                                interface=f"Thunderbolt {j}",
                                peer_node_id=peer_host,
                                source_node_id=hosts[i],
                                link_speed_gbps=link_speed,
                                tb_version=tb_version,
                            )
                        )
    except Exception:
        # Thunderbolt detection failed — degrade to Ethernet
        pass

    # Check RDMA
    try:
        rdma_available = _rdma_available(hosts, ssh_prefix=ssh_prefix)
        if rdma_available:
            rdma_edges = set(physical_edges)
            # MLX may be unable to report the Thunderbolt UUID topology even
            # after the link has routable RDMA addresses. Recover real edges
            # from shared RDMA subnets. Never turn "each Mac has an RDMA
            # device" into a fictional full mesh: on three Macs that routes
            # ranks over ports which are not physically connected.
            if not rdma_edges:
                interfaces = [
                    probe_host_interfaces(f"{ssh_prefix}{host}") for host in hosts
                ]
                for i, source in enumerate(interfaces):
                    for j, peer in enumerate(interfaces):
                        if i == j:
                            continue
                        if shared_link_addresses(source, peer).kind == "rdma":
                            rdma_edges.add((i, j))
            for i, j in sorted(rdma_edges):
                peer_host = hosts[j]
                transports.append(
                    TransportInfo(
                        kind="rdma",
                        interface="rdma",
                        peer_node_id=peer_host,
                        source_node_id=hosts[i],
                        rdma_device="rdma_en",
                    )
                )
    except Exception:
        # RDMA detection failed — skip
        pass

    # If no transports detected, assume Ethernet
    if not transports:
        for i, _host in enumerate(hosts):
            for j, peer_host in enumerate(hosts):
                if i == j:
                    continue
                transports.append(
                    TransportInfo(
                        kind="ethernet",
                        interface="ethernet",
                        peer_node_id=peer_host,
                        source_node_id=hosts[i],
                    )
                )

    return tuple(transports)


def select_backend(transports: tuple[TransportInfo, ...]) -> str:
    """Select the optimal backend based on detected transports.

    - RDMA-capable Thunderbolt mesh → jaccl
    - Thunderbolt ring → jaccl-ring
    - Ethernet only → ring
    """

    has_rdma = any(t.kind == "rdma" for t in transports)
    has_tb = any(t.kind == "thunderbolt" for t in transports)

    if has_rdma:
        return "jaccl"
    if has_tb:
        return "jaccl-ring"
    return "ring"


def detect_cluster_transports(
    hosts: list[str] | tuple[str, ...],
    *,
    ssh_prefix: str = "",
) -> TransportMatrix:
    """Detect transports and return a full transport matrix.

    Args:
        hosts: List of SSH hostnames for the cluster nodes.
        ssh_prefix: Optional prefix for SSH hostnames.

    Returns:
        ``TransportMatrix`` with transports and the backend they allow.
    """

    transports = detect_transports(hosts, ssh_prefix=ssh_prefix)
    backend = select_backend(transports)

    return TransportMatrix(
        transports=transports,
        backend=backend,
    )


# ---------------------------------------------------------------------------
# Link readiness
#
# "Is RDMA actually working?" is not answerable from device presence alone.
# There are four distinct states a user can be in, each with a different fix,
# and telling them apart is the difference between a useful page and a shrug:
#
#   rdma_ready        devices present, ports ACTIVE, ports routable
#   rdma_needs_setup  devices present and ACTIVE, but ports have no IP
#   rdma_not_enabled  no rdma_* devices at all (never enabled in Recovery)
#   thunderbolt       a TB link without usable RDMA (e.g. TB4, or unpaired)
#   ethernet          no Thunderbolt between these hosts
# ---------------------------------------------------------------------------

_RDMA_STATES = (
    "rdma_ready",
    "rdma_needs_setup",
    "rdma_not_enabled",
    "thunderbolt",
    "ethernet",
    "unknown",
)


@dataclass(frozen=True)
class LinkStatus:
    """What the fabric can actually do right now, and how to fix it."""

    state: str
    title: str
    detail: str
    backend: str
    ready: bool
    link_label: str = ""
    commands: tuple[str, ...] = ()
    doc_url: str = ""
    setup_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "title": self.title,
            "detail": self.detail,
            "backend": self.backend,
            "ready": self.ready,
            "link_label": self.link_label,
            "commands": list(self.commands),
            "doc_url": self.doc_url,
            "setup_available": self.setup_available,
        }


def classify_link(
    *,
    rdma_devices: dict[str, list[str]],
    active_ports: dict[str, str | None],
    port_ips: dict[str, str | None],
    thunderbolt: bool,
    link_speed_gbps: int | None = None,
    tb_version: str | None = None,
) -> LinkStatus:
    """Decide the link state from per-host probe results.

    Pure: every argument is data gathered elsewhere, so each branch is testable
    without two Macs.
    """

    label = ""
    if tb_version and link_speed_gbps:
        label = f"{tb_version} at {link_speed_gbps} Gb/s"
    elif tb_version:
        label = tb_version
    elif link_speed_gbps:
        label = f"{link_speed_gbps} Gb/s"

    if not thunderbolt:
        return LinkStatus(
            state="ethernet",
            title="Connected over the network",
            detail=(
                "No Thunderbolt link was found between these Macs, so the "
                "cluster will use TCP over your network. That works, but every "
                "layer's all-reduce crosses it, so expect tensor parallelism to "
                "be slower than a single Mac. Connect a Thunderbolt cable for "
                "the fast path."
            ),
            backend="ring",
            ready=True,
            link_label=label or "Ethernet / Wi-Fi",
        )

    hosts = list(rdma_devices)
    without_devices = [h for h in hosts if not rdma_devices.get(h)]
    if without_devices:
        return LinkStatus(
            state="rdma_not_enabled",
            title="RDMA is not enabled on every Mac",
            detail=(
                f"Thunderbolt is connected, but {', '.join(without_devices)} "
                f"reports no RDMA devices. RDMA is off by default and can only "
                f"be turned on from macOS Recovery — it cannot be enabled "
                f"remotely. Without it the cluster falls back to TCP over "
                f"Thunderbolt, which still works but is much slower."
            ),
            backend="jaccl-ring" if label else "ring",
            ready=False,
            link_label=label,
            commands=(
                "# On each Mac: shut down, hold the power button to enter "
                "Recovery,",
                "# then open Utilities -> Terminal and run:",
                "rdma_ctl enable",
            ),
            doc_url="https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt",
        )

    inactive = [h for h in hosts if not active_ports.get(h)]
    if inactive:
        return LinkStatus(
            state="thunderbolt",
            title="Thunderbolt connected, RDMA port not active",
            detail=(
                f"RDMA is enabled but no Thunderbolt port is reporting an "
                f"active RDMA link on {', '.join(inactive)}. Check the cable is "
                f"seated and that both Macs are awake, then detect the link "
                f"again. Thunderbolt 4 links can also show this — RDMA needs "
                f"Thunderbolt 5 on both ends."
            ),
            backend="jaccl-ring",
            ready=False,
            link_label=label,
        )

    unrouted = [h for h in hosts if not port_ips.get(h)]
    if unrouted:
        return LinkStatus(
            state="rdma_needs_setup",
            title="Thunderbolt is ready to configure",
            detail=(
                "RDMA is enabled and the Thunderbolt ports are active, but they "
                "have no IP address, so the RDMA queue pairs cannot be "
                "established. Start Cluster will ask macOS for administrator "
                "approval, configure the link, verify it, and continue."
            ),
            backend="jaccl",
            ready=False,
            link_label=label,
            doc_url="https://github.com/ml-explore/mlx/discussions/3481",
            setup_available=True,
        )

    return LinkStatus(
        state="rdma_ready",
        title="RDMA enabled and linked",
        detail=(
            "Every Mac has an active, routable RDMA port over Thunderbolt. The "
            "cluster will use the jaccl backend, which keeps tensor-parallel "
            "all-reduces off the TCP stack."
        ),
        backend="jaccl",
        ready=True,
        link_label=label,
    )


def _rdma_port_state(ssh_hostname: str, device: str) -> str | None:
    """PORT_ACTIVE / PORT_DOWN for one RDMA device, or None if unreadable."""

    command = ["ibv_devinfo", "-d", device]
    if ssh_hostname not in _LOCAL_HOSTS:
        command = [
            "ssh",
            *cluster_ssh_options(connect_timeout=10),
            ssh_hostname,
            *command,
        ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"RDMA port probe failed for {ssh_hostname}: {exc}"
        ) from exc
    if getattr(result, "returncode", 0) != 0:
        detail = str(result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"RDMA port probe failed for {ssh_hostname}"
            + (f": {detail[-400:]}" if detail else "")
        )
    for line in result.stdout.splitlines():
        if "state:" in line and "PORT_" in line:
            return "PORT_ACTIVE" if "PORT_ACTIVE" in line else "PORT_DOWN"
    return None


def _active_rdma_port(ssh_hostname: str) -> str | None:
    """The interface behind the one RDMA device that is actually linked."""

    for device in _rdma_devices(ssh_hostname):
        if _rdma_port_state(ssh_hostname, device) == "PORT_ACTIVE":
            return device.removeprefix("rdma_")
    return None


def _interface_ip(ssh_hostname: str, interface: str) -> str | None:
    """The IPv4 address on an interface, however it was assigned.

    Deliberately parses ``ifconfig`` rather than using ``ipconfig getifaddr``:
    the latter only reports addresses belonging to a configured *network
    service*, so an address set directly with ``ifconfig`` — which is exactly
    how the RDMA ports get one — reads back as absent. That made readiness
    detection report "needs setup" on a link that was already up.
    """

    command = ["ifconfig", interface]
    if ssh_hostname not in _LOCAL_HOSTS:
        command = [
            "ssh",
            *cluster_ssh_options(connect_timeout=10),
            ssh_hostname,
            *command,
        ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"RDMA address probe failed for {ssh_hostname}: {exc}"
        ) from exc
    if getattr(result, "returncode", 0) != 0:
        detail = str(result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"RDMA address probe failed for {ssh_hostname}"
            + (f": {detail[-400:]}" if detail else "")
        )
    for line in result.stdout.splitlines():
        fields = line.split()
        if fields and fields[0] == "inet" and len(fields) > 1:
            try:
                address = ipaddress.IPv4Address(fields[1])
            except ValueError:
                continue
            if any(address in network for network in _UNROUTABLE_NETWORKS):
                continue
            return str(address)
    return None


class LinkSetupError(RuntimeError):
    """The GUI could not prepare an otherwise usable RDMA link."""


class LinkAuthorizationCancelledError(LinkSetupError):
    """The user dismissed a macOS administrator authorization dialog."""


_INTERFACE_NAME = re.compile(r"^en\d{1,3}$")


def _remote_gui_authorize(host: str, shell_command: str) -> None:
    """Run one fixed privileged command in the peer's Aqua login session.

    An ``osascript`` process started directly by SSH belongs to the SSH audit
    session. macOS may create SecurityAgent, but it cannot attach the password
    window to the peer's desktop; the coordinator then sees ``-60007`` while
    the user sees no prompt on that Mac. LaunchServices is the supported bridge
    into the signed-in GUI session, so compile a tiny one-shot AppleScript app,
    open it there, and read back only fixed success/cancel/failure markers.
    """

    token = secrets.token_hex(12)
    stem = f"/tmp/com.omlx.link-authorization-{token}"
    app_path = f"{stem}.app"
    success_path = f"{stem}.success"
    cancelled_path = f"{stem}.cancelled"
    failure_path = f"{stem}.failed"

    def remote(argv: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                "ssh",
                *cluster_ssh_options(connect_timeout=10),
                host,
                shlex.join(argv),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )

    touch_success = shlex.join(["/usr/bin/touch", success_path])
    touch_cancelled = shlex.join(["/usr/bin/touch", cancelled_path])
    touch_failure = shlex.join(["/usr/bin/touch", failure_path])
    apple_script = "\n".join(
        [
            "try",
            f"do shell script {json.dumps(shell_command)} "
            "with administrator privileges",
            f"do shell script {json.dumps(touch_success)}",
            "on error errorMessage number errorNumber",
            "if errorNumber is -128 then",
            f"do shell script {json.dumps(touch_cancelled)}",
            "else",
            f"do shell script {json.dumps(touch_failure)}",
            "end if",
            "end try",
        ]
    )
    paths = [app_path, success_path, cancelled_path, failure_path]
    try:
        compiled = remote(
            ["/usr/bin/osacompile", "-o", app_path, "-e", apple_script]
        )
        if compiled.returncode:
            detail = (compiled.stderr or compiled.stdout or "").strip()
            raise LinkSetupError(
                f"Could not prepare macOS authorization on {host}"
                + (f": {detail[-400:]}" if detail else ".")
            )
        opened = remote(["/usr/bin/open", "-gj", "-W", app_path], timeout=180)
        if opened.returncode:
            detail = (opened.stderr or opened.stdout or "").strip()
            raise LinkSetupError(
                f"Could not show macOS authorization on {host}"
                + (f": {detail[-400:]}" if detail else ".")
            )
        if remote(["/bin/test", "-f", success_path]).returncode == 0:
            return
        if remote(["/bin/test", "-f", cancelled_path]).returncode == 0:
            raise LinkAuthorizationCancelledError(
                f"Administrator approval was cancelled on {host}."
            )
        raise LinkSetupError(
            f"Administrator approval failed on {host}. Unlock that Mac and "
            "enter the password for its signed-in administrator account."
        )
    finally:
        # Random, exact /tmp paths only. The app is a one-shot authorization
        # bridge and must not accumulate or be reusable after this request.
        with suppress(OSError, subprocess.SubprocessError):
            remote(["/bin/rm", "-rf", "--", *paths])


def _authorized_ifconfig(host: str, interface: str, address: str) -> None:
    """Assign one fixed link address through the native macOS auth dialog.

    The API deliberately accepts no command text. Interface and address are
    discovered server-side, validated here, and converted to one fixed
    ``ifconfig`` invocation. A remote peer opens the same macOS authorization
    dialog in its signed-in GUI session through the already-paired SSH link.
    """

    if not _INTERFACE_NAME.fullmatch(interface):
        raise LinkSetupError(f"Refusing an invalid interface name: {interface!r}")
    try:
        parsed_address = ipaddress.IPv4Address(address)
    except ValueError as exc:
        raise LinkSetupError(f"Refusing an invalid link address: {address!r}") from exc
    if host.startswith("-") or not host.strip():
        raise LinkSetupError(f"Refusing an invalid peer hostname: {host!r}")

    shell_command = shlex.join(
        [
            "/sbin/ifconfig",
            interface,
            "inet",
            str(parsed_address),
            "netmask",
            "255.255.255.0",
            "up",
        ]
    )
    apple_script = (
        f"do shell script {json.dumps(shell_command)} with administrator privileges"
    )
    osascript = ["/usr/bin/osascript", "-e", apple_script]
    if host not in _LOCAL_HOSTS:
        try:
            _remote_gui_authorize(host, shell_command)
            return
        except subprocess.TimeoutExpired as exc:
            raise LinkSetupError(
                f"{host} did not answer the macOS authorization request. "
                "Make sure it is awake and signed in, then try Start Cluster again."
            ) from exc
        except OSError as exc:
            raise LinkSetupError(
                f"Could not open macOS authorization on {host}: {exc}"
            ) from exc
    command = osascript

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
    except subprocess.TimeoutExpired as exc:
        raise LinkSetupError(
            f"{host} did not answer the macOS authorization request. "
            "Make sure it is awake and signed in, then try Start Cluster again."
        ) from exc
    except OSError as exc:
        raise LinkSetupError(f"Could not open macOS authorization on {host}: {exc}") from exc

    if result.returncode == 0:
        return
    error = (result.stderr or result.stdout or "").strip()
    if "-128" in error or "cancel" in error.lower():
        raise LinkAuthorizationCancelledError(
            f"Administrator approval was cancelled on {host}."
        )
    raise LinkSetupError(
        f"macOS could not configure the Thunderbolt link on {host}"
        + (f": {error[-400:]}" if error else ".")
    )


def configure_link(hosts: list[str] | tuple[str, ...]) -> LinkStatus:
    """Prepare an active RDMA link entirely behind the GUI's Start button."""

    if len(hosts) != 2:
        raise LinkSetupError(
            "Automatic link setup requires exactly one pair of Macs."
        )
    if any(
        not host.strip() or host.startswith("-") or any(char.isspace() for char in host)
        for host in hosts
    ):
        raise LinkSetupError("Automatic link setup received an invalid peer hostname.")

    initial = assess_link(hosts)
    if initial.ready:
        return initial
    if not initial.setup_available:
        raise LinkSetupError(initial.detail)

    try:
        active_ports = {host: _active_rdma_port(host) for host in hosts}
        if any(port is None for port in active_ports.values()):
            raise LinkSetupError(
                "The Thunderbolt RDMA port changed while it was being configured. "
                "Check the cable and try Start Cluster again."
            )
        current_ips = {
            host: _interface_ip(host, active_ports[host] or "")
            for host in hosts
        }
    except RuntimeError as exc:
        raise LinkSetupError(str(exc)) from exc

    # Reuse the subnet already present on either endpoint. Otherwise choose a
    # private point-to-point range. Rank order makes retries deterministic.
    existing = next((value for value in current_ips.values() if value), None)
    network = ipaddress.ip_network(f"{existing or '10.0.1.1'}/24", strict=False)
    for rank, host in enumerate(hosts):
        if current_ips[host]:
            continue
        target = str(network.network_address + rank + 1)
        _authorized_ifconfig(host, active_ports[host] or "", target)

    configured = assess_link(hosts)
    if not configured.ready:
        raise LinkSetupError(
            "macOS accepted the link setup, but the Thunderbolt addresses did "
            "not become routable. Check both Macs are awake and try again."
        )
    return configured


def assess_link(
    hosts: list[str] | tuple[str, ...],
    *,
    transports: tuple[TransportInfo, ...] = (),
    probe: Callable[[str], HostInterfaces] | None = None,
    verify: LinkVerifier | None = None,
) -> LinkStatus:
    """Probe every host and classify what the fabric can do right now."""

    if not hosts:
        return LinkStatus(
            state="unknown",
            title="No peers yet",
            detail="Add a peer Mac to check the link between them.",
            backend="ring",
            ready=False,
        )

    try:
        rdma_devices = {host: _rdma_devices(host) for host in hosts}
        active_ports = {host: _active_rdma_port(host) for host in hosts}
        port_ips = {
            host: _interface_ip(host, port) if port else None
            for host, port in active_ports.items()
        }
    except RuntimeError as exc:
        # An authentication/tooling failure is not evidence that RDMA is
        # disabled. Calling it that sent users into Recovery even though both
        # local diagnostics showed six healthy devices.
        return LinkStatus(
            state="unknown",
            title="Could not verify the peer's RDMA state",
            detail=(
                f"{exc}. Fix the SSH connection and detect the link again; "
                "oMLX has not changed the peer's RDMA configuration."
            ),
            backend="ring",
            ready=False,
        )
    thunderbolt = any(
        getattr(t, "kind", "") in _FAST_KINDS for t in transports
    ) or any(active_ports.values())
    speeds = [t.link_speed_gbps for t in transports if t.link_speed_gbps]
    versions = [t.tb_version for t in transports if t.tb_version]
    status = classify_link(
        rdma_devices=rdma_devices,
        active_ports=active_ports,
        port_ips=port_ips,
        thunderbolt=thunderbolt,
        link_speed_gbps=min(speeds) if speeds else None,
        tb_version=versions[0] if versions else None,
    )
    # ``ibv_devinfo`` is not a reliable cable oracle after a JACCL process has
    # torn down: macOS can report PORT_DOWN while both live interfaces still
    # carry routable addresses and are explicitly backed by rdma_* devices.
    # The two-ended address probe is stronger evidence because it identifies
    # the interface and RDMA backing on both Macs. Prefer it over that stale
    # control-plane state so the GUI does not say "Ethernet" beside a fabric
    # that activation will correctly launch as JACCL.
    if len(hosts) == 2:
        try:
            # An injected probe represents already-captured test/planning data
            # and stays pure unless its caller also injects a verifier. The
            # live product path proves reachability before saying "ready".
            link_verifier = verify
            if link_verifier is None and probe is None:
                link_verifier = verify_link_reachability
            shared = resolve_link_addresses(
                hosts[0],
                hosts[1],
                transports=transports,
                probe=probe or probe_host_interfaces,
                verify=link_verifier,
            )
        except (OSError, RuntimeError, ValueError):
            shared = None
        if shared is not None and shared.ok and shared.kind == "rdma":
            label = status.link_label if status.state != "ethernet" else ""
            if not label and shared.link_speed_gbps:
                label = f"{shared.link_speed_gbps} Gb/s"
            return LinkStatus(
                state="rdma_ready",
                title="Thunderbolt RDMA ready",
                detail=(
                    f"{shared.reason}. Both Macs have a live, routable RDMA "
                    "path; oMLX will use JACCL automatically."
                ),
                backend="jaccl",
                ready=True,
                link_label=label or "Thunderbolt RDMA",
            )
        if status.state == "rdma_ready" and shared is not None and shared.ok:
            return LinkStatus(
                state="ethernet",
                title="Using the verified TCP route",
                detail=(
                    "The Thunderbolt RDMA addresses did not form the usable "
                    f"route between these Macs. {shared.reason}. The cluster "
                    "will use the TCP ring on this verified path."
                ),
                backend="ring",
                ready=True,
                link_label="Ethernet / Wi-Fi",
            )
        if status.state == "rdma_ready" and (shared is None or not shared.ok):
            reason = (
                shared.reason
                if shared is not None
                else "The live address check could not be completed."
            )
            return LinkStatus(
                state="thunderbolt",
                title="Thunderbolt addresses are not reachable",
                detail=(
                    f"{reason} oMLX will not put these addresses in a hostfile. "
                    "Check the static addresses and routes on both Macs, or "
                    "use a verified network address for the TCP ring."
                ),
                backend="ring",
                ready=False,
                link_label=status.link_label or "Thunderbolt",
            )
        if status.ready and (shared is None or not shared.ok):
            reason = (
                shared.reason
                if shared is not None
                else "The live address check could not be completed."
            )
            return LinkStatus(
                state="unknown",
                title="No verified cluster route",
                detail=(
                    f"{reason} oMLX will not put an unverified address in the "
                    "cluster hostfile. Check the network addresses and routes "
                    "on both Macs."
                ),
                backend="ring",
                ready=False,
                link_label=status.link_label,
            )
    return status


# ---------------------------------------------------------------------------
# Link addressing
#
# A hostfile naming an address a host no longer has fails as
# "[ring] Couldn't bind socket (error: 49)" — EADDRNOTAVAIL — well after the
# launch looked healthy. macOS renumbers Thunderbolt interfaces across reboots
# and cable changes, and an address set with ``ifconfig`` goes with the old name
# (en6 becoming en4 took 10.0.1.1 with it), so an address is only true at the
# moment it is read off the host. Nothing below caches; that is the point.
# ---------------------------------------------------------------------------

# Two hosts can carry these at once without being able to reach each other:
# loopback is per-host, and a 169.254 address means DHCP failed, so unrelated
# Macs agree on the subnet and on nothing else.
_UNROUTABLE_NETWORKS = (
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
)

# Which shared link to prefer when hosts have several. RDMA over Thunderbolt
# beats plain Thunderbolt beats whatever else routes.
_LINK_KIND_RANK = {"rdma": 3, "thunderbolt": 2, "ethernet": 1}

# ifconfig prints the flag word in hex, so bridge100's "flags=8a63" is a header
# and vmenet0's "flags=8963" only looks decimal by accident.
_IFCONFIG_HEADER = re.compile(
    r"^(?P<name>[^\s:]+):\s+flags=[0-9a-fA-F]+<(?P<flags>[^>]*)>"
)
_HARDWARE_PORT = re.compile(r"^Hardware Port:\s*(?P<port>.+)$")
_HARDWARE_DEVICE = re.compile(r"^Device:\s*(?P<device>\S+)$")


@dataclass(frozen=True)
class InterfaceAddress:
    """One IPv4 address an interface carries right now."""

    interface: str
    address: str
    prefix_length: int

    @property
    def network(self) -> ipaddress.IPv4Network:
        return ipaddress.IPv4Network(
            f"{self.address}/{self.prefix_length}", strict=False
        )

    def __str__(self) -> str:
        return f"{self.interface} {self.address}/{self.prefix_length}"


@dataclass(frozen=True)
class HostInterfaces:
    """One host's addressing as it stood at the moment it was read."""

    host: str
    addresses: tuple[InterfaceAddress, ...] = ()
    rdma_interfaces: frozenset[str] = frozenset()
    thunderbolt_interfaces: frozenset[str] = frozenset()


@dataclass(frozen=True)
class LinkEndpoint:
    """Where one end of a link answers."""

    host: str
    interface: str
    address: str


@dataclass(frozen=True)
class SharedLink:
    """The addresses a pair of hosts can reach each other on, or why they cannot."""

    source: LinkEndpoint | None = None
    peer: LinkEndpoint | None = None
    kind: str = "none"
    reason: str = ""
    link_speed_gbps: int | None = None

    @property
    def ok(self) -> bool:
        return self.source is not None and self.peer is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "source": vars(self.source) if self.source else None,
            "peer": vars(self.peer) if self.peer else None,
            "kind": self.kind,
            "reason": self.reason,
            "link_speed_gbps": self.link_speed_gbps,
        }


def _prefix_length(token: str) -> int | None:
    """Prefix length from an ifconfig netmask, which macOS prints in hex."""

    try:
        value = (
            int(token, 16)
            if token.lower().startswith("0x")
            else int(ipaddress.IPv4Address(token))
        )
        # A non-contiguous mask raises here rather than yielding a bit count
        # that would silently widen the subnet.
        return ipaddress.ip_network(f"0.0.0.0/{ipaddress.IPv4Address(value)}").prefixlen
    except ValueError:
        return None


def _interface_address(interface: str, fields: list[str]) -> InterfaceAddress | None:
    """One ``inet`` line of ifconfig output, or None if it cannot carry a link."""

    try:
        address = ipaddress.IPv4Address(fields[1])
    except ValueError:
        return None
    if any(address in network for network in _UNROUTABLE_NETWORKS):
        return None
    prefix_length = 32
    if "netmask" in fields:
        index = fields.index("netmask") + 1
        parsed = _prefix_length(fields[index]) if index < len(fields) else None
        if parsed is None:
            return None
        prefix_length = parsed
    return InterfaceAddress(interface, str(address), prefix_length)


def parse_interface_addresses(output: str) -> tuple[InterfaceAddress, ...]:
    """Routable IPv4 addresses per interface, from ``ifconfig -a`` output.

    Addresses on a down interface are dropped: ifconfig keeps listing them, and
    a rank told to bind one gets the same EADDRNOTAVAIL as a stale address.
    """

    addresses: list[InterfaceAddress] = []
    interface = ""
    is_up = False
    for line in output.splitlines():
        if line[:1].strip():
            # Every interface block starts at column 0. A header we cannot read
            # must still end the previous block, or its addresses get attributed
            # to the interface listed above it.
            header = _IFCONFIG_HEADER.match(line)
            interface = header.group("name") if header else ""
            is_up = header is not None and "UP" in header.group("flags").split(",")
            continue
        fields = line.split()
        if not interface or not is_up or len(fields) < 2 or fields[0] != "inet":
            continue
        entry = _interface_address(interface, fields)
        if entry is not None:
            addresses.append(entry)
    return tuple(addresses)


def parse_thunderbolt_interfaces(output: str) -> frozenset[str]:
    """Interface names macOS calls Thunderbolt, from ``-listallhardwareports``.

    A shared subnet says nothing about which cable carries it. This is what
    separates the Thunderbolt link from the office LAN when both are routable.
    """

    interfaces: set[str] = set()
    port = ""
    for raw in output.splitlines():
        line = raw.strip()
        match = _HARDWARE_PORT.match(line)
        if match is not None:
            port = match.group("port").strip()
            continue
        match = _HARDWARE_DEVICE.match(line)
        if match is not None:
            if port.startswith("Thunderbolt"):
                interfaces.add(match.group("device"))
            port = ""
    return frozenset(interfaces)


def parse_linux_ip_addresses(output: str) -> tuple[InterfaceAddress, ...]:
    """Parse ``ip -o -4 address show up`` when Linux has no ifconfig."""

    addresses: list[InterfaceAddress] = []
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 4 or fields[2] != "inet":
            continue
        interface = fields[1].split("@", 1)[0]
        try:
            network = ipaddress.ip_interface(fields[3])
        except ValueError:
            continue
        if network.ip.is_loopback or not isinstance(network, ipaddress.IPv4Interface):
            continue
        addresses.append(
            InterfaceAddress(
                interface=interface,
                address=str(network.ip),
                prefix_length=network.network.prefixlen,
            )
        )
    return tuple(addresses)


def _read(ssh_hostname: str, command: list[str]) -> str:
    """Run one read-only command on a host, returning "" when it cannot run."""

    if ssh_hostname not in _LOCAL_HOSTS:
        command = [
            "ssh",
            *cluster_ssh_options(connect_timeout=10),
            ssh_hostname,
            *command,
        ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout


def probe_host_interfaces(ssh_hostname: str) -> HostInterfaces:
    """Read one host's live addressing, in full, in one pass.

    ``ifconfig -a`` costs the same round trip as a single interface, and asking
    for everything at once is what lets the caller find a link it did not
    already know the name of.
    """

    addresses = parse_interface_addresses(_read(ssh_hostname, ["ifconfig", "-a"]))
    if not addresses:
        addresses = parse_linux_ip_addresses(
            _read(ssh_hostname, ["ip", "-o", "-4", "address", "show", "up"])
        )
    return HostInterfaces(
        host=ssh_hostname,
        addresses=addresses,
        rdma_interfaces=frozenset(
            device.removeprefix("rdma_") for device in _rdma_devices(ssh_hostname)
        ),
        thunderbolt_interfaces=parse_thunderbolt_interfaces(
            _read(ssh_hostname, ["networksetup", "-listallhardwareports"])
        ),
    )


def _pair_kind(
    source: HostInterfaces,
    peer: HostInterfaces,
    source_address: InterfaceAddress,
    peer_address: InterfaceAddress,
) -> str:
    """What kind of link a candidate address pair actually rides on.

    Both ends must agree: one Thunderbolt port talking to an Ethernet port is
    two different cables that happen to share a subnet, not a fast link.
    """

    if (
        source_address.interface in source.rdma_interfaces
        and peer_address.interface in peer.rdma_interfaces
    ):
        return "rdma"
    if (
        source_address.interface in source.thunderbolt_interfaces
        and peer_address.interface in peer.thunderbolt_interfaces
    ):
        return "thunderbolt"
    return "ethernet"


_Candidate = tuple[str, InterfaceAddress, InterfaceAddress]
LinkVerifier = Callable[[SharedLink], tuple[bool, str]]


def _preference(candidate: _Candidate) -> tuple[int, int, str, str]:
    """Sort key putting the best address pair first.

    What the link is beats how specific the subnet is, which beats the interface
    names — the last only so the same fabric always yields the same answer.
    """

    kind, source_address, peer_address = candidate
    return (
        -_LINK_KIND_RANK[kind],
        -source_address.prefix_length,
        source_address.interface,
        peer_address.interface,
    )


def _candidate_links(
    source: HostInterfaces,
    peer: HostInterfaces,
    *,
    link_speed_gbps: int | None = None,
) -> tuple[SharedLink, ...]:
    """Every shared address pair, fastest first."""

    candidates: list[_Candidate] = []
    for source_address in source.addresses:
        for peer_address in peer.addresses:
            if source_address.network != peer_address.network:
                continue
            if source_address.address == peer_address.address:
                # Both ends holding one address is a coincidence of local
                # bridges — VM networks all pick 192.168.x.1 — not a link.
                continue
            kind = _pair_kind(source, peer, source_address, peer_address)
            candidates.append((kind, source_address, peer_address))

    links = []
    for kind, source_address, peer_address in sorted(candidates, key=_preference):
        label = (
            kind
            if link_speed_gbps is None
            else f"{kind} at {link_speed_gbps} Gb/s"
        )
        links.append(
            SharedLink(
                source=LinkEndpoint(
                    source.host, source_address.interface, source_address.address
                ),
                peer=LinkEndpoint(
                    peer.host, peer_address.interface, peer_address.address
                ),
                kind=kind,
                reason=(
                    f"{source.host} {source_address} and {peer.host} {peer_address} "
                    f"share {source_address.network} over {label}"
                ),
                link_speed_gbps=link_speed_gbps,
            )
        )
    return tuple(links)


def shared_link_addresses(
    source: HostInterfaces,
    peer: HostInterfaces,
    *,
    link_speed_gbps: int | None = None,
) -> SharedLink:
    """The best address each of two hosts should bind to reach the other.

    Pure over probed data, so every branch is testable without two Macs.
    Candidates are ranked by what the link is — RDMA over Thunderbolt, then
    Thunderbolt, then anything routable — and then by how specific the subnet
    is: a /30 between two Macs is a cable, a /16 is the building.
    """

    if not source.addresses:
        return SharedLink(reason=f"{source.host} has no routable IPv4 address")
    if not peer.addresses:
        return SharedLink(reason=f"{peer.host} has no routable IPv4 address")

    links = _candidate_links(source, peer, link_speed_gbps=link_speed_gbps)
    if not links:
        return SharedLink(
            reason=(
                f"{source.host} and {peer.host} share no subnet. "
                f"{source.host} has {', '.join(str(a) for a in source.addresses)}; "
                f"{peer.host} has {', '.join(str(a) for a in peer.addresses)}."
            )
        )
    return links[0]


def _pair_link_speed(
    transports: Sequence[TransportInfo],
    source_host: str,
    peer_host: str,
) -> int | None:
    """The fastest speed detection reported for the link between two hosts."""

    speeds = [
        transport.link_speed_gbps
        for transport in transports
        if transport.link_speed_gbps
        and {transport.source_node_id, transport.peer_node_id}
        == {source_host, peer_host}
    ]
    return max(speeds) if speeds else None


def resolve_link_addresses(
    source_host: str,
    peer_host: str,
    *,
    transports: Sequence[TransportInfo] = (),
    probe: Callable[[str], HostInterfaces] = probe_host_interfaces,
    verify: LinkVerifier | None = None,
) -> SharedLink:
    """Ask both hosts where they answer, rather than trusting a written address.

    ``probe`` is injectable so a caller holding fresh readings need not pay for
    a second round of SSH, but nothing here remembers an address between calls —
    remembering is what renumbering invalidates.
    """

    source = probe(source_host)
    peer = probe(peer_host)
    speed = _pair_link_speed(transports, source_host, peer_host)
    link = shared_link_addresses(source, peer, link_speed_gbps=speed)
    if verify is None or not link.ok:
        return link
    rejected = []
    for candidate in _candidate_links(source, peer, link_speed_gbps=speed):
        reachable, reason = verify(candidate)
        if reachable:
            return candidate
        rejected.append(reason)
    return SharedLink(
        kind=link.kind,
        reason="No verified shared path. " + " ".join(dict.fromkeys(rejected)),
        link_speed_gbps=link.link_speed_gbps,
    )


LinkCommandRunner = Callable[
    [str, Sequence[str]], subprocess.CompletedProcess[str]
]


def _run_link_command(
    ssh_hostname: str,
    command: Sequence[str],
) -> subprocess.CompletedProcess[str]:
    """Run one bounded link check locally or through the paired SSH identity."""

    argv = list(command)
    if ssh_hostname not in _LOCAL_HOSTS:
        argv = [
            "ssh",
            *cluster_ssh_options(connect_timeout=5),
            ssh_hostname,
            *argv,
        ]
    try:
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return subprocess.CompletedProcess(argv, 255, "", str(exc))


def _route_interface(output: str) -> str:
    """Interface selected by macOS ``route -n get`` output."""

    for line in output.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().lower() == "interface":
            return value.strip()
    return ""


def verify_link_reachability(
    link: SharedLink,
    *,
    runner: LinkCommandRunner | None = None,
) -> tuple[bool, str]:
    """Prove both endpoints route and answer over the selected interfaces.

    Sharing a subnet is only a candidate. Macs can retain stale addresses on
    old Thunderbolt interfaces, and two unreachable interfaces can therefore
    look like a perfect point-to-point link. The route must name the selected
    interface in both directions and one bounded ICMP probe must succeed from
    each endpoint before the address enters a hostfile.
    """

    source, peer = link.source, link.peer
    if source is None or peer is None:
        return False, link.reason or "the selected link has no endpoints"
    run = runner or _run_link_command
    directions = ((source, peer), (peer, source))
    for local, remote in directions:
        route = run(local.host, ("/sbin/route", "-n", "get", remote.address))
        selected = _route_interface(route.stdout) if route.returncode == 0 else ""
        if selected != local.interface:
            # Linux does not implement macOS's ``route -n get`` form. Binding
            # a TCP connection to the candidate source address proves both the
            # route and that the peer's SSH service answers on that exact path,
            # without needing a platform-specific interface command.
            script = (
                "import socket,sys\n"
                "s=socket.create_connection((sys.argv[2],22),timeout=3,"
                "source_address=(sys.argv[1],0))\n"
                "s.close()"
            )
            if route.returncode != 0:
                bound = run(
                    local.host,
                    ("python3", "-c", script, local.address, remote.address),
                )
                if bound.returncode == 0:
                    continue
            detail = (
                f"route uses {selected}"
                if selected
                else "the host reported no usable route"
            )
            return False, (
                f"{local.host} cannot use {local.interface} to reach "
                f"{remote.address}: {detail}."
            )
        ping = run(
            local.host,
            ("/sbin/ping", "-n", "-c", "1", "-W", "1000", remote.address),
        )
        if ping.returncode != 0:
            return False, (
                f"{local.host} routes {remote.address} over {local.interface}, "
                "but the peer did not answer on that address."
            )
    return True, link.reason
