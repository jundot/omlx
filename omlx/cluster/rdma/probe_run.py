# SPDX-License-Identifier: Apache-2.0
"""Run the byte-checked link probe through an attached client mailbox; safe to import on a worker."""

from __future__ import annotations

import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import probe_wire
from .mailbox import ClientMailbox
from .probe_wire import ProbeError, ProbeRequest
from .verification import ProbeMeasurements


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
