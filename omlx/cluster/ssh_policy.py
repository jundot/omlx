# SPDX-License-Identifier: Apache-2.0
"""One non-interactive OpenSSH policy for every cluster subprocess, and one
retrying way to actually run it.

Cluster peers are commonly discovered through both Bonjour names and changing
Thunderbolt IP addresses.  OpenSSH's default ``ask`` policy turns a harmless
new alias for an already-known key into an interactive prompt, which a server
process cannot answer.  ``accept-new`` records a first-seen alias without a
prompt while still refusing a changed key for an alias already on record.

``CheckHostIP=no`` is explicit because a user's ssh_config may enable it.  The
host name (or literal address when that is what the operator selected) remains
the known_hosts identity; DHCP and Thunderbolt address movement must not add a
second implicit IP check behind it.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

_MANAGED_IDENTITY = "~/.ssh/omlx_cluster"


def cluster_ssh_options(
    *,
    connect_timeout: float | None = None,
    keepalive: bool = False,
) -> list[str]:
    """Return ``-o`` arguments that never read from an interactive terminal."""

    options = [
        "BatchMode=yes",
        "PasswordAuthentication=no",
        "KbdInteractiveAuthentication=no",
        # Automatic cluster fabric discovery currently emits IPv4 addresses.
        # Keep the SSH control channel on the same address family instead of
        # letting a .local name select an unrelated or unroutable AAAA record.
        # The hostname remains the known_hosts identity; only resolution is
        # constrained, so pairing survives address changes.
        "AddressFamily=inet",
        "StrictHostKeyChecking=accept-new",
        "CheckHostIP=no",
        # The pairing UI creates this dedicated identity. Naming it explicitly
        # makes the exchanged key usable without ssh-agent or ~/.ssh/config,
        # while OpenSSH can still fall back to an operator's existing keys.
        f"IdentityFile={_MANAGED_IDENTITY}",
        # Suppress the benign "permanently added" / "known by other names"
        # chatter. Changed-key failures are errors and remain visible.
        "LogLevel=ERROR",
    ]
    if connect_timeout is not None:
        if isinstance(connect_timeout, bool) or connect_timeout <= 0:
            raise ValueError("SSH connect timeout must be positive")
        options.append(f"ConnectTimeout={int(max(1, connect_timeout))}")
    if keepalive:
        options.extend(
            (
                "ServerAliveInterval=15",
                "ServerAliveCountMax=4",
                "TCPKeepAlive=yes",
            )
        )
    return [part for option in options for part in ("-o", option)]


def apply_cluster_ssh_policy(
    argv: Sequence[str],
    *,
    connect_timeout: float | None = None,
    keepalive: bool = False,
) -> list[str]:
    """Insert the shared policy into an ``ssh`` or ``scp`` command."""

    if not argv or Path(argv[0]).name not in {"ssh", "scp"}:
        raise ValueError("cluster SSH policy requires an ssh or scp command")
    return [
        argv[0],
        *cluster_ssh_options(
            connect_timeout=connect_timeout,
            keepalive=keepalive,
        ),
        *argv[1:],
    ]


def run_ssh_retrying(
    argv: Sequence[str],
    *,
    timeout: float,
    attempts: int = 3,
    delay: float = 0.5,
    runner: Callable[..., Any] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one subprocess, retrying a bounded few times on failure.

    mDNS resolution for a ``.local`` peer is intermittently flaky — one
    dropped multicast round trip should not be mistaken for a permanently
    unreachable host. Every caller across this module's users is a read-only
    remote query (interface/inventory/shard/layout/RDMA-device/heartbeat
    probes), so retrying is always safe — even a peer that genuinely has no
    marker yet just takes a little longer to be correctly reported that way,
    rather than being misread as unreachable. An exception raised by
    ``subprocess.run`` itself (the ``ssh``/``scp`` binary missing, say) is
    folded into a failed ``CompletedProcess`` the same as a nonzero exit, so
    callers only ever need to check ``returncode``.

    ``runner`` defaults to the real ``subprocess.run``, resolved fresh on
    every call rather than captured once as a default argument — a captured
    default would keep pointing at the original function object even after a
    test replaces ``subprocess.run``, silently reaching a real network call
    instead of the test double. Pass ``runner`` explicitly for a seam the
    liveness probes need to simulate a peer's SSH response.
    """

    run = runner if runner is not None else subprocess.run
    argv = list(argv)
    result = subprocess.CompletedProcess(argv, 255, "", "")
    for attempt in range(attempts):
        try:
            result = run(
                argv, capture_output=True, text=True, check=False, timeout=timeout
            )
        except (OSError, subprocess.SubprocessError) as exc:
            result = subprocess.CompletedProcess(argv, 255, "", str(exc))
        if result.returncode == 0:
            return result
        if attempt < attempts - 1:
            time.sleep(delay)
    return result
