# SPDX-License-Identifier: Apache-2.0
"""Verify one RDMA link live: byte-checked round trips and bulk transfers in both directions."""

from __future__ import annotations

import json
import queue
import shlex
import subprocess
import threading
import time
from collections.abc import Callable
from typing import Any

from ..launch import _cluster_ssh_argv
from . import layout
from .daemon import DaemonStatus
from .links import NodeAddress, RdmaLink
from .mailbox import ClientMailbox, MailboxError
from .probe_run import FULL, QUICK, ProbeSettings, probe_link
from .probe_wire import ProbeError
from .verification import (
    DriverIdentity,
    LinkVerification,
    ProbeMeasurements,
    link_identity,
)
from .words import WordOps


class ProbeService:
    """The probe service running on the link's listening host over SSH."""

    def __init__(self, process: subprocess.Popen) -> None:
        self._process = process
        self._lines: queue.Queue[str | None] = queue.Queue()
        # The last plain line the service printed, usually why it failed.
        self._said = ""
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        stream = self._process.stdout
        if stream is not None:
            for line in stream:
                self._lines.put(line.strip())
        self._lines.put(None)

    def _next_json(self, timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                line = self._lines.get(timeout=max(0.01, deadline - time.monotonic()))
            except queue.Empty as exc:
                raise ProbeError(
                    f"the probe service said nothing for {timeout_s:.0f} s"
                ) from exc
            if line is None:
                raise ProbeError(
                    f"the probe service exited early: {self._said or 'no output'}"
                )
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except ValueError:
                    pass
            if line:
                self._said = line[-200:]

    def ready(self, timeout_s: float = 30.0) -> dict[str, Any]:
        """Wait until the service has attached to its mailbox."""
        announced = self._next_json(timeout_s)
        if not announced.get("ready"):
            raise ProbeError(
                f"the probe service could not start: {announced.get('error', 'unknown error')}"
            )
        return announced

    def finish(self, timeout_s: float = 30.0) -> dict[str, Any]:
        """The service's closing summary after the probe sent END."""
        summary = self._next_json(timeout_s)
        self._process.wait(timeout=timeout_s)
        return summary

    def close(self) -> None:
        # Closing stdin tells the remote service to let go of the link even if SSH lingers.
        stdin = getattr(self._process, "stdin", None)
        if stdin is not None:
            stdin.close()
        if self._process.poll() is None:
            self._process.kill()
            self._process.wait(timeout=10)


def start_probe_service(
    node: NodeAddress,
    link: str,
    socket_path: str,
    *,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> ProbeService:
    """Start the probe service for `link` on `node` over the cluster's SSH policy."""
    if not node.python_executable:
        raise ProbeError(f"{node.node_id} has no recorded worker Python")
    command = [
        node.python_executable,
        "-m",
        "omlx.cluster.rdma.link_probe_service",
        "--name",
        layout.valid_link_name(link),
        "--socket",
        socket_path,
    ]
    argv = _cluster_ssh_argv(node.ssh, shlex.join(command))
    # stdin stays open for the probe's lifetime; the remote service lets go of the link when it closes.
    process = popen(
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return ProbeService(process)


def _run_client(
    node: NodeAddress,
    arguments: list[str],
    timeout_s: float,
    run: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Run the worker-side probe client over the cluster's SSH policy and return its JSON line."""
    if not node.python_executable:
        raise ProbeError(f"{node.node_id} has no recorded worker Python")
    command = [
        node.python_executable,
        "-m",
        "omlx.cluster.rdma.link_probe_client",
        *arguments,
    ]
    done = run(
        _cluster_ssh_argv(node.ssh, shlex.join(command)),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    for line in reversed((done.stdout or "").splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    said = ((done.stderr or "") + (done.stdout or "")).strip().splitlines()
    last = said[-1][-200:] if said else f"exit code {done.returncode}"
    raise ProbeError(f"the probe client on {node.node_id} gave no result: {last}")


def read_remote_status(
    node: NodeAddress, *, run: Callable[..., Any] = subprocess.run
) -> DaemonStatus:
    """The connect daemon's status on a worker that is the client end of a link."""
    where = f"{node.node_id}:mcdma-rpcd"
    try:
        return DaemonStatus.from_dict(_run_client(node, ["status"], 30.0, run))
    except (
        ProbeError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        return DaemonStatus(
            where, False, f"could not read mcdma-rpcd on {node.node_id}: {exc}"
        )


def probe_remote_link(
    node: NodeAddress,
    link: str,
    settings: ProbeSettings,
    *,
    run: Callable[..., Any] = subprocess.run,
) -> ProbeMeasurements:
    """Run the probe on `node`, the client end of `link`; raises ProbeError on any wrong byte."""
    arguments = [
        "probe",
        "--name",
        layout.valid_link_name(link),
        "--warmup",
        str(settings.warmup),
        "--round-trips",
        str(settings.round_trips),
        "--bulk-bytes",
        str(settings.bulk_bytes),
        "--repeats",
        str(settings.repeats),
        "--call-timeout",
        str(settings.call_timeout_s),
    ]
    calls = settings.warmup + settings.round_trips + 2 * settings.repeats + 1
    result = _run_client(node, arguments, 60.0 + calls * settings.call_timeout_s, run)
    if not result.get("ok"):
        raise ProbeError(str(result.get("error") or "the probe client failed"))
    return ProbeMeasurements.from_dict(result["measurements"])


def _probed(
    link: RdmaLink,
    node: NodeAddress,
    identity: dict[str, str],
    run_probe: Callable[[], ProbeMeasurements],
    start_service: Callable[[NodeAddress, str, str], ProbeService],
    clock: Callable[[], float],
    missing: str = "",
) -> LinkVerification:
    """Start the probe service on `node`, run `run_probe` against it, and keep the evidence."""

    def outcome(
        reason: str, measurements: ProbeMeasurements | None = None
    ) -> LinkVerification:
        return LinkVerification(
            link.name, node.node_id, not reason, reason, clock(), identity, measurements
        )

    if not link.usable:
        return outcome(link.reason or "mcdma-rpcd reports this link down")
    if link.peer_node_id != node.node_id:
        return outcome(
            f"link {link.name} reaches {link.peer_node_id}, not {node.node_id}"
        )
    if missing:
        return outcome(missing)
    service = None
    try:
        service = start_service(node, link.name, layout.service_socket_path(link.name))
        service.ready()
        measurements = run_probe()
        summary = service.finish()
        if summary.get("ended") != "end":
            return outcome(
                f"the probe service ended with {summary.get('ended', 'no summary')}"
            )
        return outcome("", measurements)
    except (
        ProbeError,
        MailboxError,
        OSError,
        KeyError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        return outcome(str(exc))
    finally:
        if service is not None:
            service.close()


def verify_link(
    link: RdmaLink,
    node: NodeAddress,
    *,
    status: DaemonStatus,
    driver: DriverIdentity | None,
    ops: WordOps | None,
    ops_reason: str = "",
    settings: ProbeSettings = FULL,
    attach: Callable[[str, WordOps], ClientMailbox] = ClientMailbox.attach,
    start_service: Callable[
        [NodeAddress, str, str], ProbeService
    ] = start_probe_service,
    probe: Callable[..., ProbeMeasurements] = probe_link,
    clock: Callable[[], float] = time.time,
) -> LinkVerification:
    """Probe `link` from this coordinator to `node` and return the evidence, verified or not."""

    def run_probe() -> ProbeMeasurements:
        mailbox = attach(link.name, ops)
        try:
            if not mailbox.connected:
                raise ProbeError("mcdma-rpcd reports this link down")
            return probe(mailbox, settings)
        finally:
            mailbox.close()

    return _probed(
        link,
        node,
        link_identity(link, status, driver),
        run_probe,
        start_service,
        clock,
        "" if ops is not None else ops_reason or "libmcdma-rpc is not installed",
    )


def verify_remote_link(
    link: RdmaLink,
    client: NodeAddress,
    node: NodeAddress,
    *,
    status: DaemonStatus,
    settings: ProbeSettings = QUICK,
    start_service: Callable[
        [NodeAddress, str, str], ProbeService
    ] = start_probe_service,
    probe: Callable[..., ProbeMeasurements] = probe_remote_link,
    clock: Callable[[], float] = time.time,
) -> LinkVerification:
    """Probe `link` from worker `client`, its connect end, to worker `node` over SSH."""
    identity = {**link_identity(link, status, None), "client_node_id": client.node_id}
    return _probed(
        link,
        node,
        identity,
        lambda: probe(client, link.name, settings),
        start_service,
        clock,
    )
