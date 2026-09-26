# SPDX-License-Identifier: Apache-2.0
"""Request format and byte patterns shared by the link probe and its service."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

MAGIC = b"OMPB"
VERSION = 1
HEADER_BYTES = 32
# Reply with the request body reversed: a content-checked round trip.
ECHO = 1
# Reply with the CRC-32 and length of the request body: checks bytes sent to the peer.
SINK = 2
# Reply with `reply_bytes` of the seeded pattern: checks bytes sent from the peer.
SOURCE = 3
# Reply BYE and stop serving.
END = 4
_KINDS = (ECHO, SINK, SOURCE, END)
_LAYOUT = struct.Struct("<4sHHQQ")
SINK_REPLY = struct.Struct("<IQ")
BYE = b"BYE"


class ProbeError(RuntimeError):
    """A probe exchange failed or returned the wrong bytes."""


@dataclass(frozen=True)
class ProbeRequest:
    """The header at the start of every probe request payload."""

    kind: int
    seed: int = 0
    reply_bytes: int = 0


def pack_request(request: ProbeRequest) -> bytes:
    """Serialize a request header into HEADER_BYTES bytes."""
    packed = _LAYOUT.pack(
        MAGIC, VERSION, request.kind, request.seed, request.reply_bytes
    )
    return packed.ljust(HEADER_BYTES, b"\0")


def unpack_request(payload: memoryview | bytes) -> ProbeRequest:
    """Parse a request header, or raise ProbeError."""
    if len(payload) < HEADER_BYTES:
        raise ProbeError("probe request is shorter than its header")
    magic, version, kind, seed, reply_bytes = _LAYOUT.unpack_from(payload, 0)
    if magic != MAGIC or version != VERSION or kind not in _KINDS:
        raise ProbeError("payload is not an oMLX link probe request")
    return ProbeRequest(kind=kind, seed=seed, reply_bytes=reply_bytes)


def pattern(seed: int, nbytes: int) -> np.ndarray:
    """A deterministic, seed-dependent byte pattern that catches misplaced or stale bytes."""
    words = (nbytes + 3) // 4
    values = np.arange(words, dtype=np.uint32) * np.uint32(2654435761) + np.uint32(
        seed & 0xFFFFFFFF
    )
    return values.view(np.uint8)[:nbytes]
