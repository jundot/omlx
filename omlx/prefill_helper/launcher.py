# SPDX-License-Identifier: Apache-2.0
"""Find, provision and start a prefill helper on a paired Mac over SSH.

The helper runs from the oMLX installed on that Mac (the interpreter shim its
server writes), with the cluster pairing key. It lives as long as this Mac's
SSH session: the worker exits when its stdin closes, so a crashed or
restarted server never leaves layers loaded on the other Mac.
"""

from __future__ import annotations

import getpass
import ipaddress
import json
import queue
import re
import secrets
import shlex
import subprocess
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..cluster.registry import get_device_registry
from ..cluster.ssh_policy import cluster_ssh_options
from ..cluster.staging import scp_push, sidecar_files
from ..cluster.transport import probe_host_interfaces, shared_link_addresses
from .adapter import StageAdapter
from .worker import READY_PREFIX

DEFAULT_PYTHON = "~/.omlx/bin/omlx-cluster-python"
COPY_DIR = "~/.omlx/prefill-helper/models"
_READY_TIMEOUT_S = 900.0
_PARALLEL_COPIES = 4

Runner = Callable[..., subprocess.CompletedProcess]
# The interpreter path stays unquoted so "~" expands on the helper.
_PYTHON_PATH = re.compile(r"[~\w./-]+")


class LaunchError(RuntimeError):
    """The helper cannot start; the message is shown to the user."""


@dataclass(frozen=True)
class HelperTarget:
    node_id: str
    name: str
    ssh_target: str
    # The helper binds here and this Mac connects here (Thunderbolt first).
    address: str
    link: str
    python: str = DEFAULT_PYTHON

    def __post_init__(self) -> None:
        if not _PYTHON_PATH.fullmatch(self.python):
            raise ValueError(f"unsupported helper interpreter path: {self.python!r}")


def _ssh(target: str, command: str, *, runner: Runner = subprocess.run, timeout=30):
    return runner(
        ["ssh", *cluster_ssh_options(connect_timeout=5), target, command],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def resolve_target(node_id: str, *, runner: Runner = subprocess.run) -> HelperTarget:
    """The paired Mac's SSH target and the link address to use."""
    device = get_device_registry().get(node_id)
    if device is None:
        raise LaunchError("This Mac is not paired. Pair it in the Cluster tab.")
    user = device.get("ssh_user") or getpass.getuser()
    target = None
    for address in device.get("last_addrs", []):
        try:
            if ipaddress.ip_address(address).version != 4:
                continue
        except ValueError:
            continue
        candidate = f"{user}@{address}"
        try:
            if _ssh(candidate, "true", runner=runner, timeout=15).returncode == 0:
                target = candidate
                break
        except (OSError, subprocess.SubprocessError):
            continue
    if target is None:
        raise LaunchError(f"{device['friendly_name']} does not answer over SSH.")
    link = shared_link_addresses(
        probe_host_interfaces("localhost"), probe_host_interfaces(target)
    )
    if not link.ok:
        raise LaunchError(link.reason or "No network link to the helper Mac.")
    return HelperTarget(
        node_id=node_id,
        name=device["friendly_name"],
        ssh_target=target,
        address=link.peer.address,
        link=link.kind,
    )


_INFO = (
    "import json, omlx, mlx.core as mx; i = mx.device_info(); "
    "print(json.dumps({'version': omlx.__version__, "
    "'budget': i['max_recommended_working_set_size'], "
    "'memory': i['memory_size'], 'device': i.get('device_name', '')}))"
)


def remote_info(target: HelperTarget, *, runner: Runner = subprocess.run) -> dict:
    """The helper's oMLX version and GPU memory budget."""
    result = _ssh(
        target.ssh_target,
        f"{target.python} -c {shlex.quote(_INFO)}",
        runner=runner,
        timeout=60,
    )
    lines = result.stdout.strip().splitlines() if result.returncode == 0 else []
    try:
        return json.loads(lines[-1])
    except (IndexError, ValueError):
        raise LaunchError(f"oMLX is not installed on {target.name}.") from None


def required_files(
    model_path: str | Path, layers: int, adapter: StageAdapter
) -> dict[str, int]:
    """Files the helper needs for layers ``[0, layers)``, with their sizes."""
    root = Path(model_path).expanduser()
    names = sorted(adapter.weight_files(root, layers)) + list(sidecar_files(root))
    return {name: (root / name).stat().st_size for name in dict.fromkeys(names)}


# Runs on the helper with /usr/bin/python3 -I: finds a complete copy of the
# model in its model folders, else reports what the copy folder still lacks.
_PROBE = r"""
import json, os, shutil, sys
req = json.loads(sys.argv[1])
home = os.path.expanduser("~")
dirs = []
try:
    with open(os.path.join(home, ".omlx", "settings.json")) as f:
        model = json.load(f).get("model") or {}
    dirs = list(model.get("model_dirs") or []) or [model.get("model_dir")]
except Exception:
    pass
dirs = [os.path.expanduser(d) for d in dirs if d] or [os.path.join(home, ".omlx", "models")]
def complete(root):
    for name, size in req["files"].items():
        path = os.path.join(root, name)
        if not os.path.isfile(path) or os.path.getsize(path) != size:
            return False
    return True
found = None
for top in dirs:
    for root, subdirs, _ in os.walk(top):
        if root[len(top):].count(os.sep) >= 3:
            subdirs[:] = []
        if os.path.basename(root) == req["name"] and complete(root):
            found = root
            break
    if found:
        break
copy = os.path.join(os.path.expanduser(req["copy_dir"]), req["name"])
missing = [n for n, s in req["files"].items()
           if not (os.path.isfile(os.path.join(copy, n)) and os.path.getsize(os.path.join(copy, n)) == s)]
parent = copy
while not os.path.isdir(parent):
    parent = os.path.dirname(parent)
print(json.dumps({"found": found, "copy": copy, "missing": missing,
                  "free": shutil.disk_usage(parent).free}))
"""


def locate_files(
    target: HelperTarget,
    model_name: str,
    files: dict[str, int],
    *,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Where the helper already has the model, or what it is missing."""
    request = json.dumps({"name": model_name, "files": files, "copy_dir": COPY_DIR})
    command = f"/usr/bin/python3 -I -c {shlex.quote(_PROBE)} {shlex.quote(request)}"
    result = _ssh(target.ssh_target, command, runner=runner, timeout=120)
    if result.returncode != 0:
        raise LaunchError(f"Could not check the model files on {target.name}.")
    return json.loads(result.stdout.strip().splitlines()[-1])


def copy_files(
    target: HelperTarget,
    model_path: str | Path,
    destination: str,
    names: list[str],
    progress: Callable[[str], None] | None = None,
) -> None:
    """Copy *names* over the link address, a few files at a time."""
    source = str(Path(model_path).expanduser())

    def push(name: str) -> None:
        scp_push(
            destination_host=target.ssh_target,
            source_dir=source,
            destination_dir=destination,
            filename=name,
            transfer_address=target.address,
        )
        if progress is not None:
            progress(name)

    with ThreadPoolExecutor(max_workers=_PARALLEL_COPIES) as pool:
        for future in [pool.submit(push, name) for name in names]:
            future.result()


@dataclass
class HelperProcess:
    """A started helper: its SSH session, port and greeting."""

    process: subprocess.Popen
    port: int
    hello: dict[str, Any]
    token: bytes

    def stop(self) -> None:
        # Closing stdin ends the helper; terminate covers a stuck session.
        if self.process.poll() is None:
            try:
                if self.process.stdin is not None:
                    self.process.stdin.close()
                self.process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self.process.terminate()


def launch(
    target: HelperTarget,
    model_dir: str,
    model_type: str,
    layers: int,
    *,
    host_name: str,
    warm_tokens: int,
) -> HelperProcess:
    """Start the helper and wait until it has loaded and warmed its layers."""
    token = secrets.token_hex(32).encode()
    arguments = (
        "-m",
        "omlx.prefill_helper.worker",
        "--model",
        model_dir,
        "--model-type",
        model_type,
        "--layers",
        str(layers),
        "--bind",
        target.address,
        "--warm-tokens",
        str(warm_tokens),
        "--host-name",
        host_name,
    )
    remote = f"/usr/bin/caffeinate -i {target.python} " + " ".join(
        shlex.quote(part) for part in arguments
    )
    process = subprocess.Popen(
        [
            "ssh",
            *cluster_ssh_options(connect_timeout=10, keepalive=True),
            target.ssh_target,
            remote,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    process.stdin.write(token.decode() + "\n")
    process.stdin.flush()
    errors: list[str] = []
    lines: queue.Queue = queue.Queue()

    def read_stdout() -> None:
        for line in process.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=read_stdout, daemon=True).start()
    threading.Thread(target=lambda: errors.extend(process.stderr), daemon=True).start()
    deadline = time.monotonic() + _READY_TIMEOUT_S
    while True:
        try:
            line = lines.get(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Empty:
            break
        if line is None:
            break
        if line.startswith(READY_PREFIX):
            hello = json.loads(line[len(READY_PREFIX) :])
            return HelperProcess(process, int(hello["port"]), hello, token)
    process.terminate()
    detail = "".join(errors[-5:]).strip()
    raise LaunchError(
        f"The helper on {target.name} did not start"
        + (f": {detail}" if detail else ".")
    )
