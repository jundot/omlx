# SPDX-License-Identifier: Apache-2.0
"""Verify one RDMA link live: byte-checked round trips and bulk transfers in both directions."""

from __future__ import annotations

import json
import queue
import shlex
import subprocess
import threading
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..launch import _cluster_ssh_argv
from . import layout, probe_wire
from .daemon import DaemonStatus
from .links import NodeAddress, RdmaLink
from .mailbox import ClientMailbox, MailboxError
from .probe_wire import ProbeError, ProbeRequest
from .verification import (
    DriverIdentity,
    LinkVerification,
    ProbeMeasurements,
    link_identity,
)
from .words import WordOps


@dataclass(frozen=True)
class ProbeSettings:
    """How hard one probe works the link."""

    warmup: int = 20
    round_trips: int = 200
    bulk_bytes: int = 64 << 20
    repeats: int = 3
    call_timeout_s: float = 10.0


FULL = ProbeSettings()
# Before every launch: enough to prove bytes move both ways without delaying the load.
QUICK = ProbeSettings(warmup=5, round_trips=50, bulk_bytes=8 << 20, repeats=1)


def _call(
    mailbox: ClientMailbox, parts: tuple[Any, ...], timeout_s: float
) -> memoryview:
    seq = mailbox.stage(parts)
    reply = mailbox.wait(seq, timeout_s)
    if reply is None:
        state = "up" if mailbox.connected else "down"
        raise ProbeError(
            f"no reply within {timeout_s:.0f} s (daemon reports the link {state})"
        )
    return reply


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def probe_link(
    mailbox: ClientMailbox,
    settings: ProbeSettings = FULL,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> ProbeMeasurements:
    """Run the probe through an attached client mailbox; raises ProbeError on any wrong byte."""
    echo = probe_wire.pack_request(ProbeRequest(probe_wire.ECHO))
    body = probe_wire.pattern(7, 64).tobytes()
    samples = []
    for index in range(settings.warmup + settings.round_trips):
        began = clock()
        reply = _call(mailbox, (echo, body), settings.call_timeout_s)
        elapsed = clock() - began
        if bytes(reply) != body[::-1]:
            raise ProbeError("echo reply did not match the request")
        if index >= settings.warmup:
            samples.append(elapsed * 1e6)
    to_peer = min(
        settings.bulk_bytes, mailbox.sizes.max_request - probe_wire.HEADER_BYTES
    )
    from_peer = min(settings.bulk_bytes, mailbox.sizes.max_reply)
    sink = probe_wire.pack_request(ProbeRequest(probe_wire.SINK))
    sent_seconds = 0.0
    for repeat in range(settings.repeats):
        payload = probe_wire.pattern(1000 + repeat, to_peer)
        began = clock()
        reply = _call(mailbox, (sink, payload), settings.call_timeout_s)
        sent_seconds += clock() - began
        if len(reply) < probe_wire.SINK_REPLY.size:
            raise ProbeError("the peer's answer to a bulk transfer was short")
        crc, length = probe_wire.SINK_REPLY.unpack_from(reply, 0)
        if length != to_peer or crc != zlib.crc32(payload):
            raise ProbeError("the peer received different bytes than were sent")
    received_seconds = 0.0
    for repeat in range(settings.repeats):
        seed = 2000 + repeat
        source = probe_wire.pack_request(
            ProbeRequest(probe_wire.SOURCE, seed=seed, reply_bytes=from_peer)
        )
        began = clock()
        reply = _call(mailbox, (source,), settings.call_timeout_s)
        received_seconds += clock() - began
        if len(reply) != from_peer or not np.array_equal(
            np.frombuffer(reply, dtype=np.uint8), probe_wire.pattern(seed, from_peer)
        ):
            raise ProbeError("bytes from the peer did not match the expected pattern")
    if (
        bytes(
            _call(
                mailbox,
                (probe_wire.pack_request(ProbeRequest(probe_wire.END)),),
                settings.call_timeout_s,
            )
        )
        != probe_wire.BYE
    ):
        raise ProbeError("the probe service did not acknowledge the end of the probe")
    bits_out = 8 * to_peer * settings.repeats
    bits_in = 8 * from_peer * settings.repeats
    return ProbeMeasurements(
        round_trips=settings.round_trips,
        latency_p50_us=round(_percentile(samples, 0.50), 2),
        latency_p99_us=round(_percentile(samples, 0.99), 2),
        to_peer_bytes=to_peer * settings.repeats,
        to_peer_gbit_s=round(bits_out / max(sent_seconds, 1e-9) / 1e9, 2),
        from_peer_bytes=from_peer * settings.repeats,
        from_peer_gbit_s=round(bits_in / max(received_seconds, 1e-9) / 1e9, 2),
    )


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
    """Probe `link` end to end and return the evidence, verified or not."""
    identity = link_identity(link, status, driver)

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
    if ops is None:
        return outcome(ops_reason or "libmcdma-rpc is not installed")
    service = None
    try:
        service = start_service(node, link.name, layout.service_socket_path(link.name))
        service.ready()
        mailbox = attach(link.name, ops)
        try:
            if not mailbox.connected:
                return outcome("mcdma-rpcd reports this link down")
            measurements = probe(mailbox, settings)
        finally:
            mailbox.close()
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
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        return outcome(str(exc))
    finally:
        if service is not None:
            service.close()
