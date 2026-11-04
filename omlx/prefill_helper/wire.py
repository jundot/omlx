# SPDX-License-Identifier: Apache-2.0
"""Framed messages between this Mac and a prefill helper.

A frame is a little-endian u32 header length, a JSON header and then the raw
bytes of each array the header lists. Both ends prove they hold the session
token before anything else crosses the socket.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import socket
import struct
from typing import Any

import mlx.core as mx
import numpy as np

PROTOCOL = 1
# Chunks a helper may run ahead of the host's acknowledgements.
WINDOW = 2
_MAX_HEADER = 1 << 24
_NONCE = 32


class ProtocolError(RuntimeError):
    pass


def tune(sock: socket.socket) -> None:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 << 20)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    # A pulled cable or a sleeping Mac sends no FIN; notice it in ~25 s
    # instead of the system's two hours.
    for name, value in (
        ("TCP_KEEPALIVE", 10),
        ("TCP_KEEPINTVL", 5),
        ("TCP_KEEPCNT", 3),
    ):
        option = getattr(socket, name, None)
        if option is not None:
            sock.setsockopt(socket.IPPROTO_TCP, option, value)


def to_host(array: mx.array) -> tuple[np.ndarray, str]:
    """A contiguous numpy view of an MLX array and its dtype name."""
    dtype = str(array.dtype).rsplit(".", 1)[-1]
    if array.dtype == mx.bfloat16:
        # numpy has no bfloat16; the bits travel as uint16.
        array = array.view(mx.uint16)
    array = mx.contiguous(array)
    mx.eval(array)
    return np.ascontiguousarray(np.asarray(array)), dtype


def from_host(data: np.ndarray, dtype: str) -> mx.array:
    array = mx.array(data)
    if dtype == "bfloat16":
        array = array.view(mx.bfloat16)
        # A pending view runs on this thread's stream; another thread cannot.
        mx.eval(array)
    return array


def _recv_exact(sock: socket.socket, view: memoryview) -> None:
    got = 0
    while got < len(view):
        n = sock.recv_into(view[got:], len(view) - got)
        if not n:
            raise ConnectionError("peer closed the connection")
        got += n


def send(
    sock: socket.socket,
    meta: dict[str, Any],
    arrays: dict[str, tuple[np.ndarray, str]] | None = None,
) -> None:
    # Contiguous before the header goes out, so a frame is never cut short.
    arrays = {
        name: (np.ascontiguousarray(data), dtype)
        for name, (data, dtype) in (arrays or {}).items()
    }
    specs = [
        {"name": name, "dtype": dtype, "np": data.dtype.str, "shape": list(data.shape)}
        for name, (data, dtype) in arrays.items()
    ]
    header = json.dumps({"meta": meta, "arrays": specs}).encode()
    sock.sendall(struct.pack("<I", len(header)) + header)
    for data, _ in arrays.values():
        if data.nbytes:
            sock.sendall(memoryview(data).cast("B"))


def recv(
    sock: socket.socket,
) -> tuple[dict[str, Any], dict[str, tuple[np.ndarray, str]]]:
    size = bytearray(4)
    _recv_exact(sock, memoryview(size))
    (length,) = struct.unpack("<I", size)
    if length > _MAX_HEADER:
        raise ProtocolError("frame header too large")
    header = bytearray(length)
    _recv_exact(sock, memoryview(header))
    parsed = json.loads(header)
    arrays = {}
    for spec in parsed["arrays"]:
        data = np.empty(spec["shape"], np.dtype(spec["np"]))
        if data.nbytes:
            _recv_exact(sock, memoryview(data).cast("B"))
        arrays[spec["name"]] = (data, spec["dtype"])
    return parsed["meta"], arrays


def _digest(token: bytes, label: bytes, nonce: bytes) -> bytes:
    return hmac.new(token, label + nonce, hashlib.sha256).digest()


def _exchange(sock: socket.socket, payload: bytes) -> bytes:
    sock.sendall(payload)
    reply = bytearray(len(payload))
    _recv_exact(sock, memoryview(reply))
    return bytes(reply)


def authenticate(sock: socket.socket, token: bytes, *, server: bool) -> None:
    """Mutual challenge-response on a fresh connection.

    Each side sends a nonce and answers the other's with an HMAC that names
    its role, so a reflected answer does not verify.
    """
    mine = os.urandom(_NONCE)
    theirs = _exchange(sock, mine)
    me, peer = (b"helper", b"host") if server else (b"host", b"helper")
    answer = _exchange(sock, _digest(token, me, theirs))
    if not hmac.compare_digest(answer, _digest(token, peer, mine)):
        raise ProtocolError("prefill helper authentication failed")
