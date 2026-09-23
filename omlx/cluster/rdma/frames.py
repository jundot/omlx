# SPDX-License-Identifier: Apache-2.0
"""Fixed 128-byte headers that frame stage activations inside mailbox payloads."""

from __future__ import annotations

import struct
from dataclasses import dataclass

MAGIC = b"OMSL"
VERSION = 1
HEADER_BYTES = 128
MAX_DIMS = 8
REQUEST = 1
FRAME = 2
# magic, version, kind, message, frame, frames, offset, total, nbytes, dtype, ndim, pad, shape
_LAYOUT = struct.Struct("<4sHHQIIQQQ12sB3x8Q")
_KINDS = (REQUEST, FRAME)


class FrameError(ValueError):
    """A header is malformed or does not match what the receiver expects."""


@dataclass(frozen=True)
class Frame:
    """One header: a receiver's request for a frame, or a sender's frame of a message."""

    kind: int
    message: int
    frame: int
    frames: int = 0
    offset: int = 0
    total: int = 0
    nbytes: int = 0
    dtype: str = ""
    shape: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise FrameError(f"unknown frame kind {self.kind}")
        if len(self.shape) > MAX_DIMS or any(dim < 0 for dim in self.shape):
            raise FrameError(f"frame shape {self.shape} is not supported")
        if len(self.dtype.encode()) > 12:
            raise FrameError(f"dtype name {self.dtype!r} is too long")
        if self.kind == FRAME and (
            self.frames < 1
            or self.frame >= self.frames
            or self.offset + self.nbytes > self.total
        ):
            raise FrameError("frame bounds are inconsistent")


def request(message: int, frame: int) -> Frame:
    """A receiver's request for frame `frame` of message `message`."""
    return Frame(kind=REQUEST, message=message, frame=frame)


def pack(frame: Frame) -> bytes:
    """Serialize a header into exactly HEADER_BYTES bytes."""
    shape = tuple(frame.shape) + (0,) * (MAX_DIMS - len(frame.shape))
    packed = _LAYOUT.pack(
        MAGIC,
        VERSION,
        frame.kind,
        frame.message,
        frame.frame,
        frame.frames,
        frame.offset,
        frame.total,
        frame.nbytes,
        frame.dtype.encode(),
        len(frame.shape),
        *shape,
    )
    return packed.ljust(HEADER_BYTES, b"\0")


def unpack(payload: memoryview | bytes) -> Frame:
    """Parse the header at the start of `payload`, or raise FrameError."""
    if len(payload) < HEADER_BYTES:
        raise FrameError("payload is shorter than a frame header")
    fields = _LAYOUT.unpack_from(payload, 0)
    magic, version, kind, message, frame, frames, offset, total, nbytes, dtype, ndim = (
        fields[:11]
    )
    if magic != MAGIC or version != VERSION:
        raise FrameError("payload does not start with an oMLX stage frame header")
    if ndim > MAX_DIMS:
        raise FrameError(f"frame header claims {ndim} dimensions")
    if kind == FRAME and len(payload) < HEADER_BYTES + nbytes:
        raise FrameError("frame payload is shorter than its header says")
    return Frame(
        kind=kind,
        message=message,
        frame=frame,
        frames=frames,
        offset=offset,
        total=total,
        nbytes=nbytes,
        dtype=dtype.rstrip(b"\0").decode(),
        shape=tuple(fields[11 : 11 + ndim]),
    )


def plan(total: int, capacity: int) -> tuple[tuple[int, int], ...]:
    """Split `total` bytes into (offset, nbytes) frames of at most `capacity` bytes."""
    if capacity < 1:
        raise FrameError("mailbox has no room for frame payload")
    if total == 0:
        return ((0, 0),)
    return tuple(
        (offset, min(capacity, total - offset)) for offset in range(0, total, capacity)
    )
