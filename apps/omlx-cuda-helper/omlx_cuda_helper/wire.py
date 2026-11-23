# SPDX-License-Identifier: Apache-2.0
"""Framed messages shared by the control and the prefill data channels.

A frame is a little-endian u32 header length, a JSON header and then the raw
bytes of each array the header lists (the oMLX prefill-helper framing). Both
ends prove they hold the same key before any frame is trusted.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import socket
import struct
from typing import Any

import numpy as np

_MAX_HEADER = 1 << 24
_NONCE = 32
# An array as it crosses the link: (data, MLX dtype name).
HostArray = tuple[np.ndarray, str]


class ProtocolError(RuntimeError):
    pass


def control_key(cluster_key: bytes) -> bytes:
    """Key of the control channel, derived from the pairing key."""
    return hmac.new(cluster_key, b"omlx-prefill-control-v1", hashlib.sha256).digest()


def data_token(cluster_key: bytes, session: bytes) -> bytes:
    """Token of one prefill session; both ends derive it from the session id."""
    digest = hmac.new(
        cluster_key, b"omlx-prefill-data-v1\0" + session, hashlib.sha256
    ).hexdigest()
    return digest.encode()


def tune(sock: socket.socket) -> None:
    """Low latency and a dead-peer timeout of about 25 s."""
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    # Linux names the idle time TCP_KEEPIDLE, macOS TCP_KEEPALIVE.
    idle = getattr(socket, "TCP_KEEPIDLE", None) or getattr(
        socket, "TCP_KEEPALIVE", None
    )
    for option, value in (
        (idle, 10),
        (getattr(socket, "TCP_KEEPINTVL", None), 5),
        (getattr(socket, "TCP_KEEPCNT", None), 3),
    ):
        if option is not None:
            sock.setsockopt(socket.IPPROTO_TCP, option, value)


def recv_exact(sock: socket.socket, view: memoryview) -> None:
    got = 0
    while got < len(view):
        n = sock.recv_into(view[got:], len(view) - got)
        if not n:
            raise ConnectionError("peer closed the connection")
        got += n


def send(
    sock: socket.socket,
    meta: dict[str, Any],
    arrays: dict[str, HostArray] | None = None,
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
    sock: socket.socket, *, max_header: int = _MAX_HEADER, max_bytes: int | None = None
) -> tuple[dict[str, Any], dict[str, HostArray]]:
    """One frame. ``max_bytes`` bounds the array payload of an untrusted peer."""
    size = bytearray(4)
    recv_exact(sock, memoryview(size))
    (length,) = struct.unpack("<I", size)
    if length > max_header:
        raise ProtocolError("frame header too large")
    header = bytearray(length)
    recv_exact(sock, memoryview(header))
    try:
        parsed = json.loads(header)
        meta, specs = parsed["meta"], parsed["arrays"]
        shapes = [
            (s["name"], s["dtype"], np.dtype(s["np"]), tuple(s["shape"])) for s in specs
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise ProtocolError("malformed frame header") from error
    if not isinstance(meta, dict):
        raise ProtocolError("malformed frame header")
    if max_bytes is not None:
        total = sum(
            dtype.itemsize * int(np.prod(shape)) for _, _, dtype, shape in shapes
        )
        if total > max_bytes:
            raise ProtocolError("frame payload too large")
    arrays: dict[str, HostArray] = {}
    for name, mlx_dtype, dtype, shape in shapes:
        data = np.empty(shape, dtype)
        if data.nbytes:
            recv_exact(sock, memoryview(data).cast("B"))
        arrays[name] = (data, mlx_dtype)
    return meta, arrays


def _digest(key: bytes, label: bytes, nonce: bytes) -> bytes:
    return hmac.new(key, label + nonce, hashlib.sha256).digest()


def _exchange(sock: socket.socket, payload: bytes) -> bytes:
    sock.sendall(payload)
    reply = bytearray(len(payload))
    recv_exact(sock, memoryview(reply))
    return bytes(reply)


def authenticate(sock: socket.socket, key: bytes, *, server: bool) -> None:
    """Mutual challenge-response on a fresh connection.

    Each side sends a nonce and answers the other's with an HMAC that names
    its role, so a reflected answer does not verify.
    """
    mine = os.urandom(_NONCE)
    theirs = _exchange(sock, mine)
    me, peer = (b"helper", b"host") if server else (b"host", b"helper")
    answer = _exchange(sock, _digest(key, me, theirs))
    if not hmac.compare_digest(answer, _digest(key, peer, mine)):
        raise ProtocolError("authentication failed")
