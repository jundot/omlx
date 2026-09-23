# SPDX-License-Identifier: Apache-2.0
"""Shared-memory mailbox layout of MCDMA's link daemon, mcdma-rpcd protocol 1."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Each half starts with one control page; payloads follow it.
CTRL = 4096
# Daemons that predate the size fields write zero, meaning 4 MiB halves.
DEFAULT_HALF = 4 << 20
# Request half, both ends: the staged (client) or landed (service) request word.
REQUEST_WORD = 0
# Request half, client end: 1 while the daemon's link to the peer is up.
CONNECTED_FLAG = 64
# Request half, client end: bumped each time the daemon's link comes up; daemons that do not count leave 0.
GENERATION_WORD = 72
# Request half, both ends: request and reply half sizes as two u64 values.
SIZES = 256
# Reply half, client end: the service's reply has landed.
DONE_WORD = 64
# Reply half, service end: a reply is staged for the daemon to send.
STAGED_WORD = 128
# Connect-side mailboxes are POSIX shared memory objects with this prefix.
CLIENT_PREFIX = "mcdma-rpc."
# Listen-side mailboxes are files in this directory with the same prefix.
SERVICE_DIR = "/dev/shm"
# macOS caps POSIX shared memory names at 31 bytes including the slash.
_NAME = re.compile(r"^[A-Za-z0-9_-]{1,20}$")
_SEQ_MASK = 0xFFFFFFFF


def valid_link_name(name: object) -> str:
    """Return a link name that fits both mailbox naming schemes, or raise."""
    if not isinstance(name, str) or _NAME.fullmatch(name) is None:
        raise ValueError(f"invalid RDMA link name: {name!r}")
    return name


def client_shm_name(name: str) -> str:
    """POSIX shared memory name of a connect-side mailbox, without the slash."""
    return CLIENT_PREFIX + valid_link_name(name)


def service_mailbox_path(name: str) -> str:
    """File path of a listen-side mailbox."""
    return f"{SERVICE_DIR}/{CLIENT_PREFIX}{valid_link_name(name)}"


def service_socket_path(name: str) -> str:
    """Conventional control socket of the listen-side daemon serving `name`."""
    return f"/tmp/mcdma-rpcd.{valid_link_name(name)}.sock"


def pack_word(seq: int, length: int) -> int:
    """A mailbox word: sequence in the high 32 bits, byte length in the low 32."""
    return (seq & _SEQ_MASK) << 32 | (length & _SEQ_MASK)


def word_seq(word: int) -> int:
    """Sequence number of a mailbox word; zero means empty."""
    return (word >> 32) & _SEQ_MASK


def word_length(word: int) -> int:
    """Payload length carried by a mailbox word."""
    return word & _SEQ_MASK


def next_seq(seq: int) -> int:
    """The sequence after `seq`, wrapping past zero, which marks an empty word."""
    return seq % _SEQ_MASK + 1


@dataclass(frozen=True)
class HalfSizes:
    """Byte sizes of a mailbox's request and reply halves."""

    request: int
    reply: int

    def __post_init__(self) -> None:
        for size in (self.request, self.reply):
            if size <= CTRL or size % CTRL:
                raise ValueError(f"mailbox half size {size} is not whole pages")

    @property
    def max_request(self) -> int:
        return self.request - CTRL

    @property
    def max_reply(self) -> int:
        return self.reply - CTRL

    @property
    def total(self) -> int:
        return self.request + self.reply


def read_sizes(buffer: memoryview) -> HalfSizes:
    """Read the half sizes the daemon published in the request control page."""
    request = int.from_bytes(buffer[SIZES : SIZES + 8], "little") or DEFAULT_HALF
    reply = int.from_bytes(buffer[SIZES + 8 : SIZES + 16], "little") or DEFAULT_HALF
    sizes = HalfSizes(request=request, reply=reply)
    if sizes.total > len(buffer):
        raise ValueError("mailbox is smaller than the halves it announces")
    return sizes
