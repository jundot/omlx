# SPDX-License-Identifier: Apache-2.0
"""Supervise an isolated multi-host MLX inference job."""

from __future__ import annotations

import hashlib
import importlib.metadata
import ipaddress
import json
import logging
import math
import os
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omlx.utils import hardware

from .deployment import ClusterDeployment, validate_ssh_target
from .liveness import _LOOPBACK_TARGETS, read_marker, read_remote_marker
from .models import CLUSTER_PROTOCOL_VERSION
from .performance import performance_profiles_from_records
from .ssh_policy import cluster_ssh_options
from .staging import home_relative_model_path, validate_staged_model

logger = logging.getLogger(__name__)

_EVENT_PREFIX = "OMLX_CLUSTER_EVENT:"
_LOG_LINE_LIMIT = 8192
_LOG_HISTORY = 200
_REMOTE_OUTPUT_LIMIT = 64 * 1024
_FABRIC_INTERFACE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
_DEFAULT_CONNECTX_MIN_BYTES_PER_SECOND = 2 * 1024**3


class DistributedLaunchError(RuntimeError):
    """Raised when a distributed job cannot become or remain ready."""


@dataclass(frozen=True)
class CudaFabricProbeHost:
    """One CUDA worker and its direct-link NCCL endpoint."""

    node_id: str
    ssh: str
    ips: tuple[str, ...]
    interfaces: tuple[str, ...]
    rdma_devices: tuple[str, ...]
    python_executable: str | None = None

    def __post_init__(self) -> None:
        if not self.node_id.strip() or len(self.node_id) > 128:
            raise ValueError("CUDA fabric member requires a node ID")
        object.__setattr__(self, "ssh", validate_ssh_target(self.ssh.strip()))
        if not self.ips or not (
            len(self.ips) == len(self.interfaces) == len(self.rdma_devices)
        ):
            raise ValueError(
                "CUDA fabric member requires matching IP, interface, and RDMA lists"
            )
        normalized_ips: list[str] = []
        for raw_ip in self.ips:
            try:
                normalized_ips.append(str(ipaddress.ip_address(raw_ip.strip())))
            except ValueError as exc:
                raise ValueError(
                    "CUDA fabric member requires valid IP addresses"
                ) from exc
        object.__setattr__(self, "ips", tuple(normalized_ips))
        interfaces = tuple(interface.strip() for interface in self.interfaces)
        if any(not _FABRIC_INTERFACE_RE.fullmatch(item) for item in interfaces):
            raise ValueError("CUDA fabric member has an invalid interface name")
        object.__setattr__(self, "interfaces", interfaces)
        rdma_devices = tuple(device.strip() for device in self.rdma_devices)
        if any(not _FABRIC_INTERFACE_RE.fullmatch(item) for item in rdma_devices):
            raise ValueError("CUDA fabric member has an invalid RDMA device name")
        object.__setattr__(self, "rdma_devices", rdma_devices)
        if self.python_executable is not None:
            object.__setattr__(
                self,
                "python_executable",
                _validate_python_executable(self.python_executable.strip()),
            )


def _process_group_alive(process_group: int) -> bool:
    """Return whether any process still belongs to a supervised launch group."""

    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # A group we can no longer signal is still a live group. Treating it
        # as gone would make Stop claim success while a rank remains resident.
        return True
    return True


def _wait_for_process_group_exit(process_group: int, timeout: float) -> bool:
    """Wait for every launcher/rank process, not only the launcher parent."""

    deadline = time.monotonic() + max(0.0, timeout)
    while _process_group_alive(process_group):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.05, remaining))
    return True


def _available_launch_ports(
    deployment: ClusterDeployment,
) -> tuple[int, int]:
    """Choose one private API port and a collision-free collective span."""

    collective_count = (
        sum(len(host.ips) for host in deployment.hosts)
        * deployment.execution.ring_connections_per_ip
        if deployment.backend == "ring"
        else 1
    )
    if not 1 <= collective_count <= 1024:
        raise ValueError("collective port span is out of range")

    for _ in range(128):
        listeners: list[socket.socket] = []
        try:
            first = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            first.bind(("127.0.0.1", 0))
            listeners.append(first)
            collective_port = int(first.getsockname()[1])
            if collective_port + collective_count - 1 > 65535:
                continue
            for port in range(
                collective_port + 1,
                collective_port + collective_count,
            ):
                listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                listener.bind(("127.0.0.1", port))
                listeners.append(listener)
            api_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            api_listener.bind(("127.0.0.1", 0))
            listeners.append(api_listener)
            api_port = int(api_listener.getsockname()[1])
            return api_port, collective_port
        except OSError:
            continue
        finally:
            for listener in listeners:
                listener.close()
    raise DistributedLaunchError(
        f"could not reserve {collective_count} contiguous collective ports"
    )


def _package_version(name: str) -> str:
    if name == "omlx":
        try:
            from omlx._version import __version__

            return __version__
        except ImportError:
            # omlx._version arrived in 0.1.2. A peer older than that must still
            # produce a legible "omlx local=X remote=Y" mismatch rather than
            # fail the probe outright, so fall through to metadata.
            pass
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _local_runtime_versions() -> dict[str, str]:
    """Versions as ``_PREFLIGHT_SCRIPT`` reads them on the peer: metadata."""

    return {
        "omlx": _package_version("omlx"),
        "mlx": _package_version("mlx"),
        "mlx-lm": _package_version("mlx-lm"),
    }


def _local_probe_versions() -> dict[str, str]:
    """Versions as ``probe.py`` reads them on the peer: module constants.

    The two remote paths report differently. ``_PREFLIGHT_SCRIPT`` calls
    ``importlib.metadata.version``, while ``probe.py`` reports
    ``mx.__version__`` / ``mlx_lm.__version__`` through these helpers. Comparing
    a probe result against metadata therefore repeated the #2705 bug for mlx and
    mlx-lm: an editable or nightly install whose dist-info has drifted from the
    module blocks two ranks running identical code (#2726).

    Each comparison now reads the local side the same way its own remote does.
    That also keeps the missing-value sentinel consistent — these return
    ``"Unknown"`` where ``_package_version`` returns ``"unknown"``, and the two
    spellings used to read as a mismatch between two ranks that both lacked mlx.
    """

    return {
        "omlx": _package_version("omlx"),
        "mlx": hardware.get_mlx_version(),
        "mlx-lm": hardware.get_mlx_lm_version(),
    }


def _python_minor(version: str) -> tuple[str, str] | None:
    """``"3.12.13"`` -> ``("3", "12")``; ``None`` when it is not a version."""

    parts = version.split(".")
    if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
        return None
    return (parts[0], parts[1])


def _interpreter_parity(
    local: str, remote: Any
) -> tuple[str | None, str | None]:
    """Compare the interpreter two ranks will actually run under (#2695).

    Returns ``(blocking, warning)``, at most one of which is set.

    Python ABI tags govern which wheel each rank can load locally; Python
    objects do not cross the MLX transport boundary.  A minor-version split is
    therefore reported but does not block ranks whose package and protocol
    versions otherwise match.  Missing, malformed, or different-major reports
    remain blocking because they do not establish a compatible runtime.
    """

    remote_text = remote.strip() if isinstance(remote, str) else ""
    if not remote_text:
        return (f"python local={local} remote=missing", None)
    local_minor = _python_minor(local)
    remote_minor = _python_minor(remote_text)
    if remote_minor is None or local_minor is None:
        return (f"python local={local} remote={remote_text}", None)
    if local_minor[0] != remote_minor[0]:
        return (f"python local={local} remote={remote_text}", None)
    if local_minor != remote_minor:
        return (None, f"python minor differs: local={local} remote={remote_text}")
    if local != remote_text:
        return (None, f"python patch differs: local={local} remote={remote_text}")
    return (None, None)


def _validate_python_executable(value: str) -> str:
    path = Path(value)
    if not path.is_absolute() or "\x00" in value or len(value.encode()) > 4096:
        raise ValueError("distributed Python executable must be an absolute path")
    return str(path)


def _rank_python_module_argv(
    python_by_rank: list[str | None],
    *,
    fallback: str,
    module: str,
) -> list[str]:
    """Build one command that selects each host's probed interpreter by rank."""

    default = _validate_python_executable(fallback)
    executables = [
        _validate_python_executable(value) if value else default
        for value in python_by_rank
    ]
    if len(set(executables)) == 1:
        return [executables[0], "-m", module]
    cases = " ".join(
        f"{rank}) omlx_python={shlex.quote(executable)};;"
        for rank, executable in enumerate(executables)
    )
    script = (
        f'case "${{MLX_RANK:-}}" in {cases} *) exit 64;; esac; '
        'exec "$omlx_python" "$@"'
    )
    return ["/bin/sh", "-c", script, "omlx-rank-python", "-m", module]


def build_mlx_launch_argv(
    deployment: ClusterDeployment,
    *,
    hostfile: Path,
    api_port: int,
    collective_port: int,
    python_executable: str = sys.executable,
    cwd: Path | None = None,
    state_dir: str = "~/.omlx/cluster/runtime",
) -> list[str]:
    """Build an argument vector without a user-controlled shell fragment.

    One argv, run unchanged on every host. ``mlx.launch`` ships this vector to
    each Mac verbatim, so anything appended here says the *same thing* to every
    rank. Per-node settings — the node role above all, which decides how full a
    Mac is allowed to get — cannot be expressed as a flag: there is no way to
    write "studio=headless, macbook=workstation" in a single command line.

    They ride ``--plan`` instead. The encoded plan holds one
    ``PipelineAssignment`` per rank and the worker indexes it by its own rank
    (``assignments[rank]``), which is the only channel in this launcher that
    can carry a different value to each machine. Add per-node values to
    ``PipelineAssignment``, not to this argv.
    """

    python_executable = _validate_python_executable(python_executable)
    if not hostfile.is_absolute():
        raise ValueError("hostfile path must be absolute")
    for label, port in (
        ("API port", api_port),
        ("collective port", collective_port),
    ):
        if not 1 <= port <= 65535:
            raise ValueError(f"{label} must be between 1 and 65535")
    if api_port == collective_port:
        raise ValueError("API and collective ports must be distinct")
    if cwd is not None and not cwd.is_absolute():
        raise ValueError("distributed working directory must be absolute")

    launcher = (
        "from mlx._distributed_utils.launch import main; raise SystemExit(main() or 0)"
    )
    argv = [
        python_executable,
        "-c",
        launcher,
        "--hostfile",
        str(hostfile),
        "--starting-port",
        str(collective_port),
    ]
    if (
        deployment.backend == "ring"
        and deployment.execution.ring_connections_per_ip > 1
    ):
        argv.extend(
            [
                "--connections-per-ip",
                str(deployment.execution.ring_connections_per_ip),
            ]
        )
    if cwd is not None:
        argv.extend(["--cwd", str(cwd)])
    argv.extend(
        [
            "--",
            *_rank_python_module_argv(
                [host.python_executable for host in deployment.hosts],
                fallback=python_executable,
                module="omlx.cluster.inference_worker",
            ),
            # One shared argv launches every rank; send the ~-form so each rank
            # resolves the model in its own home (worker expands it). A coord
            # absolute path outside ~ is returned unchanged.
            "--model",
            home_relative_model_path(deployment.model),
            "--backend",
            deployment.backend,
            "--port",
            str(api_port),
            "--deployment-id",
            deployment.deployment_id,
            "--plan-hash",
            deployment.plan_hash,
            "--plan",
            deployment.encode_worker_plan(),
            "--peer-hosts",
            ",".join(host.ssh for host in deployment.hosts),
            "--state-dir",
            state_dir,
            "--execution-profile",
            deployment.execution.profile,
            "--decode-concurrency",
            str(deployment.execution.decode_concurrency),
            "--prompt-concurrency",
            str(deployment.execution.prompt_concurrency),
            "--prefill-step-size",
            str(deployment.execution.prefill_step_size),
            "--prompt-cache-size",
            str(deployment.execution.prompt_cache_size),
            "--pipeline-microbatch-size",
            str(deployment.execution.pipeline_microbatch_size),
            "--ring-connections-per-ip",
            str(deployment.execution.ring_connections_per_ip),
            "--tuning-reason",
            deployment.execution.tuning_reason,
        ]
    )
    if deployment.execution.prompt_cache_bytes is not None:
        argv.extend(
            [
                "--prompt-cache-bytes",
                str(deployment.execution.prompt_cache_bytes),
            ]
        )
    if deployment.execution.max_kv_size is not None:
        argv.extend(["--max-kv-size", str(deployment.execution.max_kv_size)])
    if deployment.execution.cache_affinity:
        argv.append("--cache-affinity")
    if deployment.execution.prompt_cache_ssd:
        argv.append("--prompt-cache-ssd")
    if deployment.execution.auto_tune:
        argv.append("--auto-tune")
    if deployment.execution.sampling_rank_only:
        argv.append("--sampling-rank-only")
    if deployment.execution.async_overlap:
        argv.append("--async-overlap")
    if deployment.trust_remote_code:
        argv.append("--trust-remote-code")
    return argv


PerformanceProbeRunner = Callable[
    ...,
    subprocess.CompletedProcess[str],
]


def _run_performance_launcher(
    argv: list[str],
    *,
    timeout: float,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=2.0)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        detail = stderr.strip() or stdout.strip()
        suffix = f": {detail[-2000:]}" if detail else ""
        raise DistributedLaunchError(
            f"performance probe exceeded {timeout:.1f}s{suffix}"
        ) from exc
    return subprocess.CompletedProcess(
        args=argv,
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def run_cluster_performance_probe(
    deployment: ClusterDeployment,
    *,
    timeout: float = 60.0,
    python_executable: str = sys.executable,
    cwd: Path | None = None,
    runner: PerformanceProbeRunner = _run_performance_launcher,
) -> dict[str, Any]:
    """Benchmark every node and the selected transport before model loading."""

    if timeout <= 0:
        raise ValueError("performance probe timeout must be positive")
    python_executable = _validate_python_executable(python_executable)
    if cwd is not None and not cwd.is_absolute():
        raise ValueError("distributed working directory must be absolute")

    with tempfile.TemporaryDirectory(
        prefix="omlx-distributed-performance-"
    ) as temporary_name:
        temporary = Path(temporary_name)
        _install_cluster_ssh_wrapper(temporary)
        hostfile = temporary / "hostfile.json"
        hostfile.write_text(
            json.dumps(
                deployment.hostfile_dict(),
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        _, collective_port = _available_launch_ports(deployment)
        launcher = (
            "from mlx._distributed_utils.launch import main; "
            "raise SystemExit(main() or 0)"
        )
        argv = [
            python_executable,
            "-c",
            launcher,
            "--hostfile",
            str(hostfile),
            "--starting-port",
            str(collective_port),
        ]
        if (
            deployment.backend == "ring"
            and deployment.execution.ring_connections_per_ip > 1
        ):
            argv.extend(
                [
                    "--connections-per-ip",
                    str(deployment.execution.ring_connections_per_ip),
                ]
            )
        if cwd is not None:
            argv.extend(["--cwd", str(cwd)])
        argv.extend(
            [
                "--",
                *_rank_python_module_argv(
                    [host.python_executable for host in deployment.hosts],
                    fallback=python_executable,
                    module="omlx.cluster.performance_worker",
                ),
            ]
        )
        environment = os.environ.copy()
        environment["PATH"] = f"{temporary}{os.pathsep}{environment.get('PATH', '')}"
        environment["SSH_ASKPASS_REQUIRE"] = "never"
        environment["PYTHONUNBUFFERED"] = "1"
        try:
            completed = runner(argv, timeout=timeout, env=environment)
        except (OSError, subprocess.SubprocessError) as exc:
            raise DistributedLaunchError(
                f"could not launch performance probe: {exc}"
            ) from exc

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        suffix = f": {detail[-2000:]}" if detail else ""
        raise DistributedLaunchError(
            f"performance probe exited with code {completed.returncode}{suffix}"
        )
    records: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("type") == "performance_result":
            records.append(value)
    try:
        profiles = performance_profiles_from_records(
            records,
            node_ids=[host.node_id for host in deployment.hosts],
            backend=deployment.backend,
        )
    except ValueError as exc:
        raise DistributedLaunchError(str(exc)) from exc
    return {
        "ok": True,
        "backend": deployment.backend,
        "world_size": deployment.world_size,
        "connections_per_ip": (
            deployment.execution.ring_connections_per_ip
            if deployment.backend == "ring"
            else 1
        ),
        "profiles": [profile.to_dict() for profile in profiles],
    }


def run_cuda_fabric_probe(
    hosts: tuple[CudaFabricProbeHost, CudaFabricProbeHost],
    *,
    timeout: float = 90.0,
    payload_mib: int = 64,
    repeats: int = 3,
    minimum_bytes_per_second: float = _DEFAULT_CONNECTX_MIN_BYTES_PER_SECOND,
    python_executable: str = sys.executable,
    cwd: Path | None = None,
    runner: PerformanceProbeRunner = _run_performance_launcher,
) -> dict[str, Any]:
    """Launch an isolated two-worker NCCL test over explicit ConnectX IPs.

    This is intentionally separate from the heterogeneous outer Ring. A Ring
    subgroup still uses Ring and therefore cannot prove that NCCL or the direct
    ConnectX path is active. The dashboard calls this before it marks a CUDA
    pair verified.
    """

    if len(hosts) != 2:
        raise ValueError("CUDA fabric verification requires exactly two workers")
    if len({host.ssh for host in hosts}) != 2:
        raise ValueError("CUDA fabric workers must use distinct SSH targets")
    if set(hosts[0].ips) & set(hosts[1].ips):
        raise ValueError("CUDA fabric workers must use distinct direct-link IPs")
    if timeout <= 0:
        raise ValueError("CUDA fabric probe timeout must be positive")
    if not 1 <= payload_mib <= 1024:
        raise ValueError("CUDA fabric payload must be between 1 and 1024 MiB")
    if not 1 <= repeats <= 20:
        raise ValueError("CUDA fabric repeats must be between 1 and 20")
    if not math.isfinite(minimum_bytes_per_second) or minimum_bytes_per_second <= 0:
        raise ValueError("CUDA fabric throughput floor must be finite and positive")
    python_executable = _validate_python_executable(python_executable)
    if cwd is not None and not cwd.is_absolute():
        raise ValueError("distributed working directory must be absolute")

    with tempfile.TemporaryDirectory(prefix="omlx-cuda-fabric-") as temporary_name:
        temporary = Path(temporary_name)
        _install_cluster_ssh_wrapper(temporary)
        hostfile = temporary / "hostfile.json"
        hostfile.write_text(
            json.dumps(
                {
                    "backend": "nccl",
                    "envs": [],
                    "hosts": [
                        {"ssh": host.ssh, "ips": list(host.ips), "rdma": []}
                        for host in hosts
                    ],
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            nccl_port = int(listener.getsockname()[1])
        launcher = (
            "from mlx._distributed_utils.launch import main; "
            "raise SystemExit(main() or 0)"
        )
        argv = [
            python_executable,
            "-c",
            launcher,
            "--backend",
            "nccl",
            "--hostfile",
            str(hostfile),
            "--nccl-port",
            str(nccl_port),
        ]
        if cwd is not None:
            argv.extend(["--cwd", str(cwd)])
        argv.extend(
            [
                "--",
                *_rank_python_module_argv(
                    [host.python_executable for host in hosts],
                    fallback=python_executable,
                    module="omlx.cluster.nccl_fabric_worker",
                ),
                "--interfaces",
                json.dumps([list(host.interfaces) for host in hosts]),
                "--rdma-devices",
                json.dumps([list(host.rdma_devices) for host in hosts]),
                "--payload-mib",
                str(payload_mib),
                "--repeats",
                str(repeats),
            ]
        )
        environment = os.environ.copy()
        environment["PATH"] = f"{temporary}{os.pathsep}{environment.get('PATH', '')}"
        environment["SSH_ASKPASS_REQUIRE"] = "never"
        environment["PYTHONUNBUFFERED"] = "1"
        try:
            completed = runner(argv, timeout=timeout, env=environment)
        except (OSError, subprocess.SubprocessError) as exc:
            raise DistributedLaunchError(
                f"could not launch CUDA fabric probe: {exc}"
            ) from exc

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        suffix = f": {detail[-2000:]}" if detail else ""
        raise DistributedLaunchError(
            f"CUDA fabric probe exited with code {completed.returncode}{suffix}"
        )
    if len(completed.stdout.encode()) + len(completed.stderr.encode()) > _REMOTE_OUTPUT_LIMIT:
        raise DistributedLaunchError("CUDA fabric probe output exceeded the safe limit")
    records: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("type") == "nccl_fabric_result":
            records.append(value)
    ranks = {int(record.get("rank", -1)) for record in records}
    if ranks != {0, 1} or len(records) != 2:
        raise DistributedLaunchError(
            "CUDA fabric probe did not return one result from each worker"
        )
    observed = min(float(record.get("payload_bytes_per_second") or 0) for record in records)
    verified = observed >= minimum_bytes_per_second
    identity = "\0".join(sorted(host.ssh for host in hosts)).encode()
    group_id = f"connectx-{hashlib.sha256(identity).hexdigest()[:16]}"
    return {
        "ok": True,
        "verified": verified,
        "group_id": group_id,
        "transport": "nccl",
        "members": [
            {
                "node_id": host.node_id,
                "ssh": host.ssh,
                "ips": list(host.ips),
                "interfaces": list(host.interfaces),
                "rdma_devices": list(host.rdma_devices),
                "rank": index,
            }
            for index, host in enumerate(hosts)
        ],
        "payload_mib": payload_mib,
        "repeats": repeats,
        "payload_bytes_per_second": observed,
        "minimum_bytes_per_second": minimum_bytes_per_second,
        "reason": (
            "NCCL completed over the selected direct-link interfaces"
            if verified
            else "NCCL completed, but measured throughput did not clear the "
            "ConnectX verification floor"
        ),
        "records": sorted(records, key=lambda item: int(item["rank"])),
    }


SSHRunner = Callable[..., subprocess.CompletedProcess[str]]


def _openssh_executable() -> str:
    system_ssh = Path("/usr/bin/ssh")
    if system_ssh.is_file() and os.access(system_ssh, os.X_OK):
        return str(system_ssh)
    discovered = shutil.which("ssh")
    if discovered is None:
        raise DistributedLaunchError("OpenSSH client is unavailable")
    executable = Path(discovered).resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise DistributedLaunchError("OpenSSH client is unavailable")
    return str(executable)


def _cluster_ssh_argv(ssh_target: str, remote_command: str) -> list[str]:
    return [
        _openssh_executable(),
        # A control channel that goes idle while ranks talk over RDMA is the
        # one that gets dropped — and the remote rank dies of SIGHUP with it.
        # Both MiniMax runs that reached ready ended exactly this way.
        *cluster_ssh_options(connect_timeout=5, keepalive=True),
        validate_ssh_target(ssh_target),
        remote_command,
    ]


def _install_cluster_ssh_wrapper(directory: Path) -> Path:
    """Make MLX's internal SSH calls inherit the prompt-free trust policy."""

    executable = _openssh_executable()
    wrapper = directory / "ssh"
    options = shlex.join(cluster_ssh_options(connect_timeout=5, keepalive=True))
    wrapper.write_text(
        f'#!/bin/sh\nexec {shlex.quote(executable)} {options} "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o700)
    return wrapper


def _run_cluster_ssh(
    ssh_target: str,
    remote_command: str,
    *,
    timeout: float,
    runner: SSHRunner,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = runner(
            _cluster_ssh_argv(ssh_target, remote_command),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DistributedLaunchError(
            f"SSH connection failed for {ssh_target}: {exc}"
        ) from exc
    output_size = len(completed.stdout.encode()) + len(completed.stderr.encode())
    if output_size > _REMOTE_OUTPUT_LIMIT:
        raise DistributedLaunchError(
            f"SSH response from {ssh_target} exceeded {_REMOTE_OUTPUT_LIMIT} bytes"
        )
    return completed


def _raise_for_ssh_transport_failure(
    ssh_target: str,
    completed: subprocess.CompletedProcess[str],
) -> None:
    """Keep OpenSSH failures out of remote interpreter fallback."""

    if completed.returncode != 255:
        return
    detail = completed.stderr.strip() or "OpenSSH exited with status 255"
    raise DistributedLaunchError(f"SSH to {ssh_target} failed: {detail}")


def discover_remote_python_executable(
    ssh_target: str,
    *,
    preferred: str = sys.executable,
    timeout: float = 8.0,
    runner: SSHRunner = subprocess.run,
) -> str:
    """Find the peer interpreter that can import oMLX without user commands."""

    preferred = _validate_python_executable(preferred)
    candidates = (
        preferred,
        "~/.omlx/bin/omlx-cluster-python",
        "~/.omlx/bin/omlx-source-python",
        "/opt/omlx-cluster-worker/venv/bin/python",
        "/opt/omlx/bin/python",
        "/usr/local/bin/python3",
        "/usr/bin/python3",
        "~/omlx-distributed/.venv/bin/python",
        "python3",
    )
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        # The packaged macOS app and the worker shim are launchers which
        # assemble PYTHONPATH for an underlying interpreter.  Returning
        # ``sys.executable`` from a launcher loses that environment and makes
        # the very next probe fail with ModuleNotFoundError.  Preserve every
        # path-shaped candidate (absolute or ``~``) exactly as given; only a
        # bare command name (``python3``) needs ``sys.executable`` to become an
        # absolute path.
        script = (
            "import os,omlx; print(os.path.expanduser("
            f"{candidate!r}))"
            if candidate.startswith(("~", "/"))
            else "import sys,omlx; print(sys.executable)"
        )
        command = (
            f"{candidate} {shlex.join(['-c', script])}"
            if candidate.startswith("~")
            else shlex.join([candidate, "-c", script])
        )
        completed = _run_cluster_ssh(
            ssh_target,
            command,
            timeout=timeout,
            runner=runner,
        )
        _raise_for_ssh_transport_failure(ssh_target, completed)
        if completed.returncode != 0:
            continue
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            continue
        try:
            return _validate_python_executable(lines[-1])
        except ValueError:
            continue
    raise DistributedLaunchError(
        f"{ssh_target} has no Python interpreter that can import oMLX"
    )


def discover_remote_system_python(
    ssh_target: str,
    *,
    preferred: str = sys.executable,
    timeout: float = 8.0,
    runner: SSHRunner = subprocess.run,
) -> str:
    """Find a plain peer Python for read-only pre-install hardware inventory."""

    preferred = _validate_python_executable(preferred)
    script = "import sys; print(sys.executable)"
    candidates = (
        preferred,
        "/usr/local/bin/python3",
        "/usr/bin/python3",
        "python3",
    )
    for candidate in dict.fromkeys(candidates):
        completed = _run_cluster_ssh(
            ssh_target,
            shlex.join([candidate, "-c", script]),
            timeout=timeout,
            runner=runner,
        )
        _raise_for_ssh_transport_failure(ssh_target, completed)
        if completed.returncode != 0:
            continue
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            continue
        try:
            return _validate_python_executable(lines[-1])
        except ValueError:
            continue
    raise DistributedLaunchError(
        f"{ssh_target} has no Python interpreter for hardware discovery"
    )


_REMOTE_SYSTEM_PROBE = r"""
import json, os, platform, socket, subprocess, sys

def run(argv):
    try:
        value = subprocess.run(
            argv, capture_output=True, text=True, check=False, timeout=8
        )
        return value.returncode, value.stdout
    except Exception:
        return 127, ""

def memory_bytes():
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except Exception:
        return 0

# Look for an oMLX install rather than asserting there isn't one. This runs
# under a plain system interpreter, so importing omlx is not expected to work
# even when the app is present; anything found here means the runtime exists
# and merely could not be started, which is a different diagnosis from
# "not installed".
def note(message):
    # stderr only: stdout carries the JSON payload this probe is read for.
    sys.stderr.write("omlx-probe: " + message + "\n")

def worker_runtime_evidence():
    found = []
    for path in (
        os.path.expanduser("~/.omlx/bin/omlx"),
        os.path.expanduser("~/.omlx/bin/omlx-cluster-python"),
        "/Applications/oMLX.app",
        os.path.expanduser("~/Applications/oMLX.app"),
        "/opt/omlx-cluster-worker/venv/bin/python",
    ):
        try:
            if os.path.exists(path):
                found.append(path)
        except (OSError, ValueError) as exc:
            # An unreadable parent or a bad path is not evidence either way;
            # say so, because silence here reads as "absent".
            note("cannot test %s: %r" % (path, exc))
    try:
        import importlib.util
        if importlib.util.find_spec("omlx") is not None:
            found.append("import omlx")
    except (ImportError, AttributeError, TypeError, ValueError, OSError) as exc:
        # A broken or partially removed install raises here rather than
        # returning None, and that is worth seeing in the SSH stderr.
        note("cannot look up the omlx package: %r" % (exc,))
    return found

gpu_code, gpu_output = run([
    "nvidia-smi", "--query-gpu=name", "--format=csv,noheader"
])
cuda = gpu_code == 0 and bool(gpu_output.strip())
chip = gpu_output.splitlines()[0].strip() if cuda else (
    "Apple Silicon" if platform.system() == "Darwin" else platform.machine()
)
rdma_code, rdma_output = run(["rdma", "-j", "link", "show"])
try:
    raw_links = json.loads(rdma_output) if rdma_code == 0 else []
except Exception:
    raw_links = []
devices = []
addresses = {}
interfaces = {}
for item in raw_links if isinstance(raw_links, list) else []:
    if not isinstance(item, dict):
        continue
    device = str(item.get("ifname") or item.get("dev") or "").split("/", 1)[0]
    interface = str(item.get("netdev") or item.get("netdev_name") or "")
    if not device or not interface:
        continue
    devices.append(device)
    interfaces[device] = interface
    code, output = run(["ip", "-j", "address", "show", "dev", interface])
    try:
        rows = json.loads(output) if code == 0 else []
    except Exception:
        rows = []
    for row in rows if isinstance(rows, list) else []:
        for address in row.get("addr_info", []) if isinstance(row, dict) else []:
            value = str(address.get("local") or "") if isinstance(address, dict) else ""
            if value and address.get("scope") in (None, "global", "link"):
                addresses[device] = value
                break
        if device in addresses:
            break
physical = memory_bytes()
system = platform.system().lower() or "unknown"
accelerator = "cuda" if cuda else "metal" if system == "darwin" else "cpu"
payload = {
    "protocol_version": None,
    "node": {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "chip_name": chip,
        "physical_memory_bytes": physical,
        "recommended_working_set_bytes": int(physical * 0.90),
        "admission_ceiling_bytes": 0,
        "accelerator": accelerator,
        "accelerator_vendor": "nvidia" if cuda else "apple" if system == "darwin" else "unknown",
        "memory_kind": "unified" if accelerator in ("metal", "cuda") and platform.machine().lower() in ("arm64", "aarch64") else "system",
        "distributed_backends": [],
        "fabric_kind": "connectx-7" if cuda and addresses else None,
        "fabric_group_id": None,
        "fabric_verified": False,
        "worker_runtime_ready": False,
        "worker_runtime_evidence": worker_runtime_evidence(),
    },
    "runtime": {
        "omlx_version": "",
        "mlx_version": "",
        "mlx_lm_version": "",
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "macos_version": platform.mac_ver()[0] or "unknown",
        "os_name": system,
        "os_version": platform.release() or "unknown",
    },
    "transport": {
        "state": "enabled_no_peer" if devices else "unavailable",
        "rdma": {
            "control_status": "enabled" if devices else "unavailable",
            "enabled": bool(devices),
            "devices": devices,
            "addresses": addresses,
            "network_interfaces": interfaces,
        },
        "thunderbolt": {"ports": [], "peer_connected": False},
        "route": None,
    },
    # Filled in by probe_remote_system_host from the evidence above, so the
    # verdict has exactly one author.
    "warnings": [],
}
print(json.dumps(payload, sort_keys=True))
"""

_RUNTIME_MISSING = "oMLX worker runtime is not installed"
_RUNTIME_UNVERIFIED = "oMLX worker runtime could not be verified"


def probe_remote_system_host(
    ssh_target: str,
    *,
    preferred_python: str = sys.executable,
    timeout: float = 15.0,
    runner: SSHRunner = subprocess.run,
) -> dict[str, Any]:
    """Inventory an SSH host before the oMLX worker runtime is installed."""

    python_executable = discover_remote_system_python(
        ssh_target,
        preferred=preferred_python,
        timeout=min(timeout, 8.0),
        runner=runner,
    )
    completed = _run_cluster_ssh(
        ssh_target,
        shlex.join([python_executable, "-c", _REMOTE_SYSTEM_PROBE]),
        timeout=timeout,
        runner=runner,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise DistributedLaunchError(
            f"hardware discovery failed for {ssh_target}"
            + (f": {detail[:500]}" if detail else "")
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise DistributedLaunchError(
            f"{ssh_target} did not return hardware discovery JSON"
        ) from exc
    if not isinstance(payload, dict) or not all(
        isinstance(payload.get(key), dict) for key in ("node", "runtime", "transport")
    ):
        raise DistributedLaunchError(
            f"{ssh_target} returned incomplete hardware discovery"
        )
    # "Not installed" is a claim about the peer, so it has to be earned. The
    # inventory looks for an oMLX install with a plain interpreter; when it
    # finds one, all we actually know is that no interpreter here could load
    # the worker — say that instead (#2680).
    raw_evidence = payload["node"].get("worker_runtime_evidence")
    evidence = (
        [str(item) for item in raw_evidence] if isinstance(raw_evidence, list) else []
    )
    payload["warnings"] = [
        "oMLX is installed on this node but its worker runtime could not be run."
        if evidence
        else "oMLX worker runtime is not installed on this node."
    ]
    return {
        "ok": False,
        "ssh": validate_ssh_target(ssh_target),
        "ssh_reachable": True,
        "status": payload,
        "runtime_compatible": False,
        "runtime_mismatches": [
            _RUNTIME_UNVERIFIED if evidence else _RUNTIME_MISSING
        ],
        # Same keys as the healthy path, so a caller never has to know which
        # branch produced the result before reading it.
        "runtime_warnings": [],
        "worker_runtime_evidence": evidence,
        "bootstrap_required": True,
    }


def probe_remote_admission_ceiling(
    ssh_target: str,
    *,
    python_executable: str | None = None,
    timeout: float = 8.0,
    runner: SSHRunner = subprocess.run,
) -> int:
    """Read only the peer's live memory ceiling, without a hardware rescan.

    The full capability probe invokes ``system_profiler`` and transport tools;
    doing that merely to refresh a slider made every cluster-page load slow.
    This fixed, bounded command uses the peer's own oMLX memory guard and
    normally completes in one SSH round trip.

    ``python_executable`` is the interpreter a previous capability probe found
    on the peer.  It used to default to the coordinator's ``sys.executable``,
    which inside the packaged app is a bundled binary that exists on the peer
    but cannot import oMLX — so every dashboard poll raised, and the cluster
    page oscillated between ready and Needs Attention (#2680).  With no known
    interpreter, or when the known one has stopped working, discover the peer's
    own instead.
    """

    # Ask the peer's live server first: one HTTP round trip instead of a cold
    # `import omlx` (which imports MLX and can blow the SSH timeout). Peers
    # conventionally run the coordinator's port; 9000 is kept as a legacy
    # candidate for mixed setups. Hardcoding 9000 alone left the fast path dead
    # on every cluster serving on the (default) port 8000.
    try:
        from omlx.settings import get_settings

        local_port = int(get_settings().server.port)
    except Exception:
        local_port = 8000
    ports = tuple(dict.fromkeys((local_port, 9000)))
    script = (
        "import json,urllib.request\n"
        "ceiling=0\n"
        f"for port in {ports!r}:\n"
        "    try:\n"
        "        with urllib.request.urlopen("
        "'http://127.0.0.1:%d/health'%port,timeout=2) as response:\n"
        "            health=json.load(response)\n"
        "        ceiling=int(health.get('engine_pool',{}).get('final_ceiling',0))\n"
        "        if ceiling>0:\n"
        "            break\n"
        "    except Exception:\n"
        "        pass\n"
        "if ceiling<=0:\n"
        "    from omlx.cluster.memory_guard import ceiling_breakdown\n"
        "    ceiling=int(ceiling_breakdown().get('hard_limit',0))\n"
        "print(json.dumps({'admission_ceiling_bytes':ceiling}))"
    )

    def _read(executable: str) -> subprocess.CompletedProcess[str]:
        return _run_cluster_ssh(
            ssh_target,
            shlex.join([executable, "-c", script]),
            timeout=timeout,
            runner=runner,
        )

    attempted: str | None = None
    completed: subprocess.CompletedProcess[str] | None = None
    if python_executable is not None:
        attempted = _validate_python_executable(python_executable)
        completed = _read(attempted)
    if completed is None or completed.returncode != 0:
        detail = (
            (completed.stderr.strip() or completed.stdout.strip())
            if completed is not None
            else ""
        )
        failure = f"memory ceiling probe failed for {ssh_target}" + (
            f": {detail[:500]}" if detail else ""
        )
        try:
            discovered = discover_remote_python_executable(
                ssh_target,
                preferred=attempted or sys.executable,
                timeout=min(timeout, 8.0),
                runner=runner,
            )
        except DistributedLaunchError as exc:
            # Carry the discovery reason into the message. Dropping it left
            # the operator with a bare ModuleNotFoundError repeating every
            # poll and no indication that a search had even been attempted.
            raise DistributedLaunchError(
                f"{failure}; no interpreter that can import oMLX was found "
                f"on {ssh_target}: {exc}"
            ) from exc
        if discovered == attempted:
            raise DistributedLaunchError(
                f"{failure}; {discovered} is the only interpreter "
                f"{ssh_target} offers and it did not answer"
            )
        # A silent recovery is how the same stale interpreter got retried
        # hundreds of times before anyone noticed (#2680).
        logger.info(
            "Memory ceiling probe for %s fell back from %s to the discovered %s",
            ssh_target,
            attempted or "no known interpreter",
            discovered,
        )
        completed = _read(discovered)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise DistributedLaunchError(
                f"memory ceiling probe failed for {ssh_target}"
                + (f": {detail[:500]}" if detail else "")
            )
    try:
        payload = json.loads(completed.stdout)
        ceiling = int(payload.get("admission_ceiling_bytes") or 0)
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DistributedLaunchError(
            f"{ssh_target} returned an invalid memory ceiling"
        ) from exc
    if ceiling <= 0:
        raise DistributedLaunchError(
            f"{ssh_target} did not report an oMLX memory ceiling"
        )
    return ceiling


def probe_remote_host(
    ssh_target: str,
    *,
    route_to: str | None = None,
    python_executable: str = sys.executable,
    timeout: float = 25.0,
    runner: SSHRunner = subprocess.run,
) -> dict[str, Any]:
    """Read peer capability through host-key-verified, non-interactive SSH."""

    python_executable = _validate_python_executable(python_executable)
    effective_python = python_executable
    if timeout <= 0:
        raise ValueError("SSH probe timeout must be positive")
    command = [
        python_executable,
        "-m",
        "omlx.cli",
        "cluster",
        "status",
        "--json",
    ]
    if route_to is not None:
        command.extend(["--route-to", route_to])
    completed = _run_cluster_ssh(
        ssh_target,
        shlex.join(command),
        timeout=timeout,
        runner=runner,
    )
    if completed.returncode != 0:
        try:
            discovered = discover_remote_python_executable(
                ssh_target,
                preferred=python_executable,
                timeout=min(timeout, 8.0),
                runner=runner,
            )
        except DistributedLaunchError:
            return probe_remote_system_host(
                ssh_target,
                preferred_python=python_executable,
                timeout=min(timeout, 15.0),
                runner=runner,
            )
        if discovered != python_executable:
            effective_python = discovered
            command[0] = discovered
            completed = _run_cluster_ssh(
                ssh_target,
                shlex.join(command),
                timeout=timeout,
                runner=runner,
            )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        # An older packaged oMLX can import successfully while lacking the
        # cluster CLI/protocol entirely.  Keep that SSH-reachable hardware in
        # the GUI, but fail it closed for launches until its worker runtime is
        # upgraded.  A raw inventory is also more useful than presenting this
        # as a connection failure.
        try:
            fallback = probe_remote_system_host(
                ssh_target,
                preferred_python=python_executable,
                timeout=min(timeout, 15.0),
                runner=runner,
            )
        except DistributedLaunchError as fallback_exc:
            suffix = f": {detail[:500]}" if detail else ""
            raise DistributedLaunchError(
                f"peer capability probe failed for {ssh_target}{suffix}"
            ) from fallback_exc
        mismatch = "installed oMLX worker does not support the cluster protocol"
        fallback["runtime_mismatches"] = [mismatch]
        status = fallback.get("status")
        if isinstance(status, dict):
            warnings = status.get("warnings")
            if isinstance(warnings, list):
                status["warnings"] = [
                    "Update the installed oMLX worker to enable cluster execution."
                ]
        return fallback
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise DistributedLaunchError(
            f"{ssh_target} did not return cluster status JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise DistributedLaunchError(f"{ssh_target} returned invalid cluster status")
    if payload.get("protocol_version") != CLUSTER_PROTOCOL_VERSION:
        raise DistributedLaunchError(
            f"cluster protocol mismatch on {ssh_target}: "
            f"local={CLUSTER_PROTOCOL_VERSION} "
            f"remote={payload.get('protocol_version', 'missing')}"
        )
    runtime = payload.get("runtime")
    node = payload.get("node")
    transport = payload.get("transport")
    if not all(isinstance(value, dict) for value in (runtime, node, transport)):
        raise DistributedLaunchError(f"{ssh_target} returned incomplete cluster status")
    # A packaged-app launcher may intentionally report its underlying CPython
    # binary.  Reusing that binary directly loses the launcher's PYTHONPATH;
    # carry forward the exact executable that passed this probe instead.
    runtime["python_executable"] = effective_python
    # probe.py reports mlx/mlx-lm as module constants, so read ours the same
    # way rather than from dist-info (#2726).
    expected = _local_probe_versions()
    remote_versions = {
        "omlx": runtime.get("omlx_version"),
        "mlx": runtime.get("mlx_version"),
        "mlx-lm": runtime.get("mlx_lm_version"),
    }
    mismatches = [
        f"{name} local={expected[name]} remote={remote_versions[name] or 'missing'}"
        for name in expected
        if expected[name] != remote_versions[name]
    ]
    # The packages can match while the interpreter under them does not (#2695).
    blocking, warning = _interpreter_parity(
        platform.python_version(), runtime.get("python_version")
    )
    if blocking is not None:
        mismatches.append(blocking)
    warnings = [warning] if warning is not None else []
    if warnings:
        existing = payload.get("warnings")
        payload["warnings"] = (
            [*existing, *warnings] if isinstance(existing, list) else list(warnings)
        )
    return {
        "ok": not mismatches,
        "ssh": validate_ssh_target(ssh_target),
        "status": payload,
        "runtime_compatible": not mismatches,
        "runtime_mismatches": mismatches,
        "runtime_warnings": warnings,
    }


_PREFLIGHT_SCRIPT = (
    "import importlib.metadata as m,json,pathlib,platform,sys\n"
    "from omlx.cluster.memory_guard import ceiling_breakdown\n"
    "from omlx.cluster.models import CLUSTER_PROTOCOL_VERSION as p\n"
    "from omlx.cluster.staging import validate_staged_model as validate\n"
    "from omlx._torch_stub import install as install_torch_stub\n"
    "install_torch_stub()\n"
    "import mlx_lm.server\n"
    "import omlx.adapter.output_parser\n"
    # Use the coordinator's source of truth, but tolerate peers that predate
    # omlx._version so preflight can report a readable version mismatch.
    "def package_version(name):\n"
    "    if name == 'omlx':\n"
    "        try:\n"
    "            from omlx._version import __version__\n"
    "            return __version__\n"
    "        except ImportError:\n"
    "            pass\n"
    "    try:\n"
    "        return m.version(name)\n"
    "    except m.PackageNotFoundError:\n"
    "        return 'unknown'\n"
    "x=pathlib.Path(sys.argv[1]).expanduser()\n"
    "v={n:package_version(n) for n in ('omlx','mlx','mlx-lm')}\n"
    "v['cluster-protocol']=p\n"
    # Report the interpreter this rank will actually run under. Package and
    # protocol parity decide compatibility; Python minor differences are
    # surfaced as diagnostics because each rank loads its own matching wheel.
    "v['python']=platform.python_version()\n"
    "v['admission-ceiling-bytes']=int("
    "ceiling_breakdown(sys.argv[5]).get('hard_limit',0))\n"
    "v['model-exists']=x.is_dir()\n"
    "if v['model-exists']:\n"
    "    v.update(validate(x,int(sys.argv[2]),int(sys.argv[3]),"
    "rank=int(sys.argv[4])))\n"
    "print(json.dumps(v,sort_keys=True))"
)


def preflight_remote_hosts(
    deployment: ClusterDeployment,
    *,
    python_executable: str = sys.executable,
    timeout: float = 8.0,
    runner: SSHRunner = subprocess.run,
) -> list[dict[str, Any]]:
    """Require non-interactive, host-key-verified SSH and matching runtimes."""

    python_executable = _validate_python_executable(python_executable)
    if timeout <= 0:
        raise ValueError("SSH preflight timeout must be positive")
    expected = _local_runtime_versions()
    script = _PREFLIGHT_SCRIPT
    local_python_version = platform.python_version()
    results: list[dict[str, Any]] = []
    local_model_exists = Path(deployment.model).is_dir()
    assignments = sorted(deployment.assignments, key=lambda item: item.rank)
    if len(assignments) != len(deployment.hosts):
        raise DistributedLaunchError(
            "assignment count does not match the cluster host count"
        )
    local_validation = (
        validate_staged_model(
            deployment.model,
            assignments[0].start_layer,
            assignments[0].end_layer,
            rank=assignments[0].rank,
        )
        if local_model_exists
        else {}
    )
    local_identity = local_validation.get("model_identity")
    try:
        from .memory_guard import ceiling_breakdown

        local_admission_ceiling = int(
            ceiling_breakdown(
                assignments[0].memory_guard_tier
            ).get("hard_limit", 0)
        )
    except Exception as exc:
        raise DistributedLaunchError(
            f"could not measure the admission ceiling on "
            f"{deployment.hosts[0].node_id}: {exc}"
        ) from exc

    for rank, host in enumerate(deployment.hosts):
        assignment = assignments[rank]
        if rank == 0:
            if local_model_exists and not local_validation.get("stage_ready"):
                detail = _staged_model_failure(local_validation)
                raise DistributedLaunchError(
                    f"model stage is incomplete on {host.node_id}: {detail}"
                )
            results.append(
                {
                    "rank": rank,
                    "node_id": host.node_id,
                    "ssh": host.ssh,
                    "runtime": expected
                    | {"cluster-protocol": CLUSTER_PROTOCOL_VERSION},
                    "model_exists": local_model_exists,
                    "model_identity": local_identity,
                    "stage_ready": local_validation.get("stage_ready"),
                    "admission_ceiling_bytes": local_admission_ceiling,
                    "runtime_warnings": [],
                    "local": True,
                }
            )
            continue
        remote_python = host.python_executable or python_executable
        # Send the ~-form: deployment.model is the coordinator's absolute path,
        # which names nothing on a peer with a different macOS account. The
        # preflight script expanduser()s it in the peer's own home.
        remote_command = shlex.join(
            [
                remote_python,
                "-c",
                script,
                home_relative_model_path(deployment.model),
                str(assignment.start_layer),
                str(assignment.end_layer),
                str(assignment.rank),
                assignment.memory_guard_tier,
            ]
        )
        try:
            completed = _run_cluster_ssh(
                host.ssh,
                remote_command,
                timeout=timeout,
                runner=runner,
            )
        except DistributedLaunchError as exc:
            raise DistributedLaunchError(
                f"SSH preflight failed for {host.node_id}: {exc}"
            ) from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            suffix = f": {detail}" if detail else ""
            raise DistributedLaunchError(
                f"SSH preflight failed for {host.node_id}{suffix}"
            )
        try:
            versions = json.loads(completed.stdout.strip())
        except json.JSONDecodeError as exc:
            raise DistributedLaunchError(
                f"{host.node_id} did not return runtime metadata"
            ) from exc
        if not isinstance(versions, dict):
            raise DistributedLaunchError(
                f"{host.node_id} returned invalid runtime metadata"
            )
        mismatches = [
            f"{name} local={expected[name]} remote={versions.get(name, 'missing')}"
            for name in expected
            if expected[name] != versions.get(name)
        ]
        if versions.get("cluster-protocol") != CLUSTER_PROTOCOL_VERSION:
            mismatches.append(
                "cluster-protocol "
                f"local={CLUSTER_PROTOCOL_VERSION} "
                f"remote={versions.get('cluster-protocol', 'missing')}"
            )
        blocking, warning = _interpreter_parity(
            local_python_version, versions.get("python")
        )
        if blocking is not None:
            mismatches.append(blocking)
        runtime_warnings = [warning] if warning is not None else []
        if versions.get("model-exists") is not True:
            mismatches.append(
                f"model directory is missing on remote host: {deployment.model}"
            )
        if local_identity is not None and versions.get("model_identity") != local_identity:
            mismatches.append(
                "model identity differs from the coordinator "
                "(config, tokenizer, processor, or weight index)"
            )
        if mismatches:
            raise DistributedLaunchError(
                f"runtime mismatch on {host.node_id}: " + "; ".join(mismatches)
            )
        if local_model_exists and versions.get("stage_ready") is not True:
            raise DistributedLaunchError(
                f"model stage is incomplete on {host.node_id}: "
                f"{_staged_model_failure(versions)}"
            )
        results.append(
            {
                "rank": rank,
                "node_id": host.node_id,
                "ssh": host.ssh,
                "runtime": versions,
                "model_exists": True,
                "model_identity": versions.get("model_identity"),
                "stage_ready": versions.get("stage_ready"),
                "admission_ceiling_bytes": int(
                    versions.get("admission-ceiling-bytes") or 0
                ),
                "runtime_warnings": runtime_warnings,
                "local": False,
            }
        )
    _validate_deployment_admission(deployment, results)
    return results


def _validate_deployment_admission(
    deployment: ClusterDeployment,
    preflight: list[dict[str, Any]],
) -> None:
    """Refuse a stale/optimistic memory plan before any rank is launched.

    The planner and each rank use the same role/manual-limit arithmetic, but
    unified-memory pressure can change between opening the page and pressing
    Start. Measuring every Mac during preflight closes that final race. This
    also prevents installed RAM from being mistaken for an MLX working set.
    """

    from .memory_guard import stage_budget

    by_rank = {int(item.get("rank", -1)): item for item in preflight}
    failures: list[str] = []
    gib = 1024**3
    for assignment in sorted(deployment.assignments, key=lambda item: item.rank):
        status = by_rank.get(assignment.rank, {})
        ceiling = int(status.get("admission_ceiling_bytes") or 0)
        if ceiling <= 0:
            failures.append(
                f"rank {assignment.rank} ({assignment.node_id}) did not report "
                "its current oMLX memory ceiling"
            )
            continue
        safety = None
        if assignment.manual_memory_limit:
            safety = max(
                0.0,
                min(
                    1.0,
                    (
                        assignment.capacity_bytes - assignment.reserve_bytes
                    )
                    / assignment.capacity_bytes,
                ),
            )
        admissible = stage_budget(
            ceiling,
            role=assignment.role,
            safety=safety,
        )
        required = assignment.planned_weight_bytes
        if required > admissible:
            failures.append(
                f"rank {assignment.rank} ({assignment.node_id}) needs "
                f"{required / gib:.1f} GiB but can admit "
                f"{admissible / gib:.1f} GiB right now"
            )
    if failures:
        raise DistributedLaunchError(
            "Model does not fit the current per-node memory limits; no ranks "
            "were started. "
            + "; ".join(failures)
            + ". Rebuild the plan, lower the context window, allow more "
            "memory, close other apps, or choose a smaller model."
        )


def _staged_model_failure(status: dict[str, Any]) -> str:
    """Compact rank-local staging diagnostics for the activation error."""

    problems: list[str] = []
    missing = status.get("missing_files") or []
    corrupt = status.get("corrupt_files") or []
    layers = status.get("missing_layers") or []
    if missing:
        problems.append("missing " + ", ".join(str(name) for name in missing[:5]))
    if corrupt:
        problems.append("invalid " + ", ".join(str(name) for name in corrupt[:3]))
    if layers:
        rendered = ", ".join(str(layer) for layer in layers[:8])
        problems.append(f"missing layers {rendered}")
    return "; ".join(problems) or "rank-local validation did not pass"


@dataclass(frozen=True)
class DistributedJobStatus:
    deployment_id: str
    phase: str
    endpoint: str | None
    pid: int | None
    returncode: int | None
    world_size: int
    plan_hash: str
    stderr_tail: tuple[str, ...]
    failure_reason: str | None = None
    ranks: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "phase": self.phase,
            "endpoint": self.endpoint,
            "pid": self.pid,
            "returncode": self.returncode,
            "world_size": self.world_size,
            "plan_hash": self.plan_hash,
            "stderr_tail": list(self.stderr_tail),
            "failure_reason": self.failure_reason,
            "ranks": [dict(rank) for rank in self.ranks],
        }


class DistributedJobSupervisor:
    """Own the launcher process group and enforce readiness/teardown deadlines."""

    def __init__(
        self,
        deployment: ClusterDeployment,
        *,
        python_executable: str = sys.executable,
        cwd: Path | None = None,
        state_dir: str = "~/.omlx/cluster/runtime",
        load_timeout: float = 1800.0,
        stop_timeout: float = 10.0,
        preflight: bool = True,
    ) -> None:
        if load_timeout <= 0 or stop_timeout <= 0:
            raise ValueError("supervisor timeouts must be positive")
        self.deployment = deployment
        self.python_executable = _validate_python_executable(python_executable)
        self.cwd = cwd
        self.state_dir = state_dir
        self.load_timeout = load_timeout
        self.stop_timeout = stop_timeout
        self.preflight = preflight
        self.process: subprocess.Popen[str] | None = None
        self.port: int | None = None
        self.collective_port: int | None = None
        self.ready_event: dict[str, Any] | None = None
        self.rank_ready_events: dict[int, dict[str, Any]] = {}
        self.failure_event: dict[str, Any] | None = None
        self._phase = "stopped"
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._stdout = deque(maxlen=_LOG_HISTORY)
        self._stderr = deque(maxlen=_LOG_HISTORY)
        self._condition = threading.Condition()
        self._readers: list[threading.Thread] = []

    @property
    def endpoint(self) -> str | None:
        return f"http://127.0.0.1:{self.port}" if self.port is not None else None

    def start(self) -> dict[str, Any]:
        if self.process is not None:
            raise RuntimeError("distributed job is already started")
        self._phase = "preflight"
        if self.preflight:
            preflight_remote_hosts(
                self.deployment,
                python_executable=self.python_executable,
            )

        self._temporary = tempfile.TemporaryDirectory(prefix="omlx-distributed-launch-")
        hostfile = Path(self._temporary.name) / "hostfile.json"
        _install_cluster_ssh_wrapper(Path(self._temporary.name))
        hostfile.write_text(
            json.dumps(
                self.deployment.hostfile_dict(),
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        self.port, self.collective_port = _available_launch_ports(self.deployment)
        argv = build_mlx_launch_argv(
            self.deployment,
            hostfile=hostfile,
            api_port=self.port,
            collective_port=self.collective_port,
            python_executable=self.python_executable,
            cwd=self.cwd,
            state_dir=self.state_dir,
        )
        self._phase = "loading"
        try:
            environment = os.environ.copy()
            environment["PATH"] = (
                f"{self._temporary.name}{os.pathsep}{environment.get('PATH', '')}"
            )
            environment["SSH_ASKPASS_REQUIRE"] = "never"
            environment["PYTHONUNBUFFERED"] = "1"
            self.process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=environment,
                start_new_session=True,
            )
            self._start_readers()
            event = self._wait_for_ready()
            self._wait_for_listener()
            self._phase = "ready"
            self.ready_event = event
            return event
        except Exception:
            self._terminate()
            raise

    def _start_readers(self) -> None:
        process = self.process
        if process is None or process.stdout is None or process.stderr is None:
            raise DistributedLaunchError("launcher pipes are unavailable")
        for stream, destination, parse_events in (
            (process.stdout, self._stdout, True),
            (process.stderr, self._stderr, False),
        ):
            reader = threading.Thread(
                target=self._drain,
                args=(stream, destination, parse_events),
                daemon=True,
            )
            reader.start()
            self._readers.append(reader)

    def _drain(
        self,
        stream: Any,
        destination: deque[str],
        parse_events: bool,
    ) -> None:
        for raw_line in stream:
            line = raw_line.rstrip("\r\n")[:_LOG_LINE_LIMIT]
            with self._condition:
                destination.append(line)
                if parse_events and line.startswith(_EVENT_PREFIX):
                    try:
                        event = json.loads(line.removeprefix(_EVENT_PREFIX))
                    except json.JSONDecodeError:
                        event = None
                    if isinstance(event, dict):
                        if event.get("type") == "ready":
                            self.ready_event = event
                        elif event.get("type") == "rank_ready":
                            rank = event.get("rank")
                            if isinstance(rank, int) and not isinstance(rank, bool):
                                self.rank_ready_events[rank] = event
                        elif event.get("reason") or event.get("error"):
                            self.failure_event = event
                self._condition.notify_all()

    def _wait_for_ready(self) -> dict[str, Any]:
        deadline = time.monotonic() + self.load_timeout
        with self._condition:
            while (
                self.ready_event is None
                or len(self.rank_ready_events) < self.deployment.world_size
            ):
                if self.failure_event is not None:
                    raise DistributedLaunchError(self._failure_detail())
                process = self.process
                if process is None:
                    raise DistributedLaunchError("launcher disappeared")
                returncode = process.poll()
                if returncode is not None:
                    raise DistributedLaunchError(self._exit_detail(returncode))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"distributed model did not load within "
                        f"{self.load_timeout:.1f}s"
                    )
                self._condition.wait(timeout=min(remaining, 0.5))

        event = self.ready_event
        assert event is not None
        expected = {
            "deployment_id": self.deployment.deployment_id,
            "plan_hash": self.deployment.plan_hash,
            "world_size": self.deployment.world_size,
            "port": self.port,
        }
        for key, value in expected.items():
            if event.get(key) != value:
                raise DistributedLaunchError(f"worker ready event has unexpected {key}")
        ranks: list[dict[str, Any]] = []
        for rank, assignment in enumerate(self.deployment.assignments):
            rank_event = self.rank_ready_events.get(rank)
            if rank_event is None:
                raise DistributedLaunchError(
                    f"worker ready event is missing rank {rank}"
                )
            rank_expected = {
                "deployment_id": self.deployment.deployment_id,
                "plan_hash": self.deployment.plan_hash,
                "rank": rank,
                "node_id": assignment.node_id,
                "world_size": self.deployment.world_size,
            }
            for key, value in rank_expected.items():
                if rank_event.get(key) != value:
                    raise DistributedLaunchError(
                        f"rank {rank} ready event has unexpected {key}"
                    )
            ranks.append(dict(rank_event))
        return dict(event) | {"ranks": ranks}

    def _wait_for_listener(self) -> None:
        assert self.port is not None
        deadline = time.monotonic() + min(15.0, self.load_timeout)
        while time.monotonic() < deadline:
            process = self.process
            if process is None or process.poll() is not None:
                code = None if process is None else process.returncode
                raise DistributedLaunchError(self._exit_detail(code))
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.25):
                    return
            except OSError:
                time.sleep(0.05)
        raise TimeoutError("rank-zero inference endpoint did not start listening")

    def stop(self) -> None:
        self._terminate()

    def _terminate(self) -> None:
        process = self.process
        if process is not None:
            process_group = process.pid
            deadline = time.monotonic() + self.stop_timeout
            if _process_group_alive(process_group):
                with suppress(ProcessLookupError):
                    os.killpg(process_group, signal.SIGTERM)
            if process.poll() is None:
                with suppress(subprocess.TimeoutExpired):
                    process.wait(
                        timeout=max(0.0, deadline - time.monotonic())
                    )

            # mlx.launch can exit promptly while a local rank remains blocked
            # in a Metal/JACCL collective. Waiting only for the launcher left
            # that rank orphaned (PPID 1) and its unified-memory allocation
            # resident. Stop is complete only when the entire launch process
            # group has disappeared.
            remaining = max(0.0, deadline - time.monotonic())
            if not _wait_for_process_group_exit(process_group, remaining):
                with suppress(ProcessLookupError):
                    os.killpg(process_group, signal.SIGKILL)
                _wait_for_process_group_exit(process_group, 2.0)
            if process.poll() is None:
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=2.0)
        self._reap_remote_ranks()
        for reader in self._readers:
            reader.join(timeout=0.5)
        self._readers.clear()
        if process is not None:
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
        self.process = None
        self.port = None
        self.collective_port = None
        self.ready_event = None
        self.rank_ready_events.clear()
        self.failure_event = None
        self._phase = "stopped"
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def _reap_remote_ranks(self, *, runner: SSHRunner = subprocess.run) -> None:
        """Kill rank workers on peer hosts that the local group kill cannot reach.

        ``mlx.launch`` runs each rank through SSH in its own process group on
        its host. When local supervision stops, an uncooperative remote rank
        (for example, blocked in a Metal/JACCL collective) remains resident,
        pinning unified memory and breaking future admission.

        Each rank writes a marker at
        ``{state_dir}/{deployment_id}-rank-{rank}.json``. Read that marker
        over SSH, validate that the PID matches the expected deployment,
        plan, and rank, verify the process command line before signaling,
        and escalate SIGTERM -> wait -> SIGKILL -> confirmed dead.

        The marker file is deliberately left in place: it holds the rank's
        failure phase and error, which ``_runtime_failure_reason`` and the
        liveness views read after exactly this teardown. The script reports
        what it did as one JSON line so a failed reap is visible in the logs
        instead of passing as a clean stop.
        """
        if not self.deployment or not self.deployment.hosts:
            return
        for rank, host in enumerate(self.deployment.hosts):
            if host.ssh in _LOOPBACK_TARGETS:
                continue
            filename = f"{self.deployment.deployment_id}-rank-{rank}.json"
            script = (
                "import json,os,pathlib,signal,subprocess,sys,time\n"
                f"root=pathlib.Path({self.state_dir!r}).expanduser()\n"
                f"p=root / {filename!r}\n"
                f"dep_id={self.deployment.deployment_id!r}\n"
                f"p_hash={self.deployment.plan_hash!r}\n"
                f"rank_idx={rank!r}\n"
                "def out(action):\n"
                "  print(json.dumps({'rank':rank_idx,'action':action}))\n"
                "  sys.stdout.flush()\n"
                "if not p.is_file():\n"
                "  out('no-marker'); raise SystemExit(0)\n"
                "try:\n"
                "  d=json.loads(p.read_text())\n"
                "except Exception:\n"
                "  out('marker-unreadable'); raise SystemExit(0)\n"
                "if d.get('deployment_id') != dep_id or d.get('plan_hash') != p_hash or d.get('rank') != rank_idx:\n"
                "  out('identity-mismatch'); raise SystemExit(0)\n"
                "pid=d.get('pid')\n"
                "if not isinstance(pid,int) or isinstance(pid,bool) or pid<=0 or pid==os.getpid():\n"
                "  out('identity-mismatch'); raise SystemExit(0)\n"
                "def is_dead(target_pid):\n"
                "  try:\n"
                "    os.kill(target_pid,0)\n"
                "  except ProcessLookupError:\n"
                "    return True\n"
                "  except PermissionError:\n"
                "    pass\n"
                "  try:\n"
                "    res=subprocess.run(['ps','-p',str(target_pid),'-o','stat='],capture_output=True,text=True,check=False)\n"
                "    st=res.stdout.strip()\n"
                "    if not st or 'Z' in st:\n"
                "      return True\n"
                "  except Exception:\n"
                "    pass\n"
                "  return False\n"
                "if is_dead(pid):\n"
                "  out('already-dead'); raise SystemExit(0)\n"
                "try:\n"
                "  res=subprocess.run(['ps','-p',str(pid),'-o','args='],capture_output=True,text=True,check=False)\n"
                "  cmd=res.stdout.strip()\n"
                "  if not cmd:\n"
                "    res=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True,check=False)\n"
                "    cmd=res.stdout.strip()\n"
                "except Exception:\n"
                "  cmd=''\n"
                "if not ('omlx.cluster.inference_worker' in cmd and dep_id in cmd):\n"
                "  out('pid-reused'); raise SystemExit(0)\n"
                "try:\n"
                "  os.kill(pid,signal.SIGTERM)\n"
                "except ProcessLookupError:\n"
                "  out('already-dead'); raise SystemExit(0)\n"
                "deadline=time.monotonic()+3.0\n"
                "while time.monotonic()<deadline:\n"
                "  if is_dead(pid):\n"
                "    out('terminated'); raise SystemExit(0)\n"
                "  time.sleep(0.05)\n"
                "try:\n"
                "  os.kill(pid,signal.SIGKILL)\n"
                "except ProcessLookupError:\n"
                "  out('terminated'); raise SystemExit(0)\n"
                "kill_deadline=time.monotonic()+2.0\n"
                "while time.monotonic()<kill_deadline:\n"
                "  if is_dead(pid):\n"
                "    out('killed'); raise SystemExit(0)\n"
                "  time.sleep(0.05)\n"
                "out('kill-failed')\n"
            )
            try:
                completed = _run_cluster_ssh(
                    host.ssh,
                    f"python3 -c {shlex.quote(script)}",
                    timeout=8.0,
                    runner=runner,
                )
            except (DistributedLaunchError, OSError) as exc:
                # Best effort: an unreachable peer or disconnected link
                # cannot leave a resident rank behind. Still say so — a
                # reap that never ran must not read as a clean stop.
                logger.warning(
                    "remote rank reap unreachable for rank %d on %s: %s",
                    rank,
                    host.ssh,
                    exc,
                )
                continue
            action = ""
            for line in reversed(completed.stdout.strip().splitlines()):
                try:
                    payload = json.loads(line)
                except ValueError:
                    continue
                if isinstance(payload, dict) and payload.get("action"):
                    action = str(payload["action"])
                    break
            if action == "kill-failed":
                logger.warning(
                    "remote rank %d on %s survived SIGKILL; its memory stays "
                    "resident until cleaned up manually",
                    rank,
                    host.ssh,
                )
            elif action in ("terminated", "killed"):
                logger.info(
                    "reaped remote rank %d on %s (%s)", rank, host.ssh, action
                )
            elif action:
                logger.debug(
                    "remote rank reap for rank %d on %s: %s",
                    rank,
                    host.ssh,
                    action,
                )
            else:
                logger.warning(
                    "remote rank reap for rank %d on %s returned no report "
                    "(exit %s)",
                    rank,
                    host.ssh,
                    completed.returncode,
                )

    def _exit_detail(self, returncode: int | None) -> str:
        failure_reason = self._failure_reason() or self._runtime_failure_reason()
        if failure_reason:
            return (
                f"distributed launcher exited with code {returncode}: {failure_reason}"
            )
        lines = tuple(self._stderr)[-20:] or tuple(self._stdout)[-20:]
        detail = "\n".join(lines)
        suffix = f": {detail}" if detail else ""
        return f"distributed launcher exited with code {returncode}{suffix}"

    def _runtime_failure_reason(self) -> str | None:
        """Recover the rank exception MLX's SSH cleanup can otherwise hide.

        ``mlx.launch`` may exit successfully after a worker failure and then
        print a ``CalledProcessError`` from its cleanup thread because the
        remote PID has already disappeared. Every rank has already persisted
        the useful exception in its own bounded marker, so prefer that evidence
        when it belongs to this exact deployment and plan.
        """

        failures: list[str] = []
        for rank, host in enumerate(self.deployment.hosts):
            filename = f"{self.deployment.deployment_id}-rank-{rank}.json"
            if host.ssh in _LOOPBACK_TARGETS:
                marker = read_marker(
                    Path(self.state_dir).expanduser() / filename
                )
            else:
                remote_root = self.state_dir.rstrip("/") or "."
                marker, _, _, _ = read_remote_marker(
                    host.ssh,
                    f"{remote_root}/{filename}",
                )
            if not isinstance(marker, dict):
                continue
            if (
                marker.get("deployment_id") != self.deployment.deployment_id
                or marker.get("plan_hash") != self.deployment.plan_hash
                or marker.get("rank") != rank
                or marker.get("phase") not in {
                    "failed",
                    "peer_lost",
                    "launcher_lost",
                }
            ):
                continue
            error = marker.get("error")
            if isinstance(error, str) and error.strip():
                failures.append(
                    f"rank {rank} ({host.node_id}): {error.strip()}"
                )
        return "; ".join(failures)[:_LOG_LINE_LIMIT] or None

    def _failure_reason(self) -> str | None:
        event = self.failure_event
        if not event:
            return None
        value = event.get("reason") or event.get("error")
        return str(value)[:_LOG_LINE_LIMIT] if value else None

    def _failure_detail(self) -> str:
        reason = self._failure_reason()
        event_type = str((self.failure_event or {}).get("type") or "worker failure")
        return (
            f"distributed worker reported {event_type}: {reason or 'unknown failure'}"
        )

    def status(self) -> DistributedJobStatus:
        process = self.process
        return DistributedJobStatus(
            deployment_id=self.deployment.deployment_id,
            phase=self._phase,
            endpoint=self.endpoint,
            pid=process.pid if process is not None else None,
            returncode=process.poll() if process is not None else None,
            world_size=self.deployment.world_size,
            plan_hash=self.deployment.plan_hash,
            stderr_tail=tuple(self._stderr)[-20:],
            failure_reason=self._failure_reason(),
            ranks=tuple(
                dict(self.rank_ready_events[rank])
                for rank in sorted(self.rank_ready_events)
            ),
        )

    def __enter__(self) -> DistributedJobSupervisor:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()
