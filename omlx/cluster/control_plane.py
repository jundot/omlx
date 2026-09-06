# SPDX-License-Identifier: Apache-2.0
"""Small reliable rank-control channel kept separate from tensor collectives."""

from __future__ import annotations

import hashlib
import hmac
import pickle
import secrets
import socket
import struct
import time
import zlib
from contextlib import AbstractContextManager, suppress
from typing import Any

from .system_socket_proxy import (
    SystemSocketProxy,
    open_system_tcp_proxy,
    should_proxy_control_socket,
)

_HANDSHAKE_MAGIC = b"OC2H"
_HANDSHAKE_CHALLENGE_MAGIC = b"OC2C"
_HANDSHAKE_ACK_MAGIC = b"OC2A"
_MESSAGE_MAGIC = b"OC2M"
_VERSION = 1
_HANDSHAKE_CHALLENGE = struct.Struct("!4sI32s")
_HANDSHAKE = struct.Struct("!4sII32s")
_HANDSHAKE_ACK = struct.Struct("!4sI32s")
_HEADER_PREFIX = struct.Struct("!4sIIII")
_HEADER = struct.Struct("!4sIIII32s")
_MAX_OBJECT_BYTES = 256 * 1024 * 1024
_WORKER_AUTH_DOMAIN = b"omlx-rank-control-worker-v1"
_COORDINATOR_AUTH_DOMAIN = b"omlx-rank-control-coordinator-v1"
_MESSAGE_AUTH_DOMAIN = b"omlx-rank-control-message-v1"


def _recv_exact(stream: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        part = stream.recv(size - len(chunks))
        if not part:
            raise ConnectionError("rank-control peer closed its socket")
        chunks.extend(part)
    return bytes(chunks)


class RankControlPlane(AbstractContextManager["RankControlPlane"]):
    """Rank-zero object broadcast over TCP, with strict ordering/integrity.

    JACCL remains the high-bandwidth tensor transport. Request metadata and
    cancellation lists are single-producer control messages, and routing them
    through tiny RDMA reductions caused corrupt headers and lost completions.
    One persistent socket per worker avoids per-token connection setup while
    keeping control failures explicit and bounded.
    """

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        host: str,
        port: int,
        token: str,
        connect_timeout: float = 120.0,
        io_timeout: float = 120.0,
    ) -> None:
        if not 0 <= rank < world_size or world_size < 2:
            raise ValueError("rank-control identity is invalid")
        if not 1 <= int(port) <= 65535:
            raise ValueError("rank-control port is invalid")
        encoded_token = token.encode("ascii", "strict")
        if not encoded_token or len(encoded_token) > 64:
            raise ValueError("rank-control token must be 1..64 ASCII bytes")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.host = str(host)
        self.port = int(port)
        self._token = encoded_token.ljust(64, b"\0")
        self._connect_timeout = float(connect_timeout)
        self._io_timeout = float(io_timeout)
        self._listener: socket.socket | None = None
        self._peers: dict[int, socket.socket] = {}
        self._stream: socket.socket | None = None
        self._stream_proxy: SystemSocketProxy | None = None
        self._sequence = 0

    def __enter__(self) -> RankControlPlane:
        try:
            if self.rank == 0:
                self._accept_workers()
            else:
                self._connect_to_coordinator()
        except BaseException:
            self.close()
            raise
        return self

    def _configure(self, stream: socket.socket) -> None:
        stream.settimeout(self._io_timeout)
        stream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        stream.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

    def _handshake_tag(self, domain: bytes, challenge: bytes, rank: int) -> bytes:
        identity = struct.pack("!II", _VERSION, int(rank))
        return hmac.new(
            self._token,
            domain + challenge + identity,
            hashlib.sha256,
        ).digest()

    def _accept_workers(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.host, self.port))
        listener.listen(self.world_size - 1)
        listener.settimeout(min(1.0, self._connect_timeout))
        self._listener = listener
        deadline = time.monotonic() + self._connect_timeout
        while len(self._peers) < self.world_size - 1:
            if time.monotonic() >= deadline:
                raise TimeoutError("rank-control workers did not connect")
            try:
                stream, _address = listener.accept()
            except TimeoutError:
                continue
            remaining = max(0.1, deadline - time.monotonic())
            stream.settimeout(min(5.0, remaining))
            try:
                challenge = secrets.token_bytes(32)
                stream.sendall(
                    _HANDSHAKE_CHALLENGE.pack(
                        _HANDSHAKE_CHALLENGE_MAGIC,
                        _VERSION,
                        challenge,
                    )
                )
                magic, version, rank, observed_tag = _HANDSHAKE.unpack(
                    _recv_exact(stream, _HANDSHAKE.size)
                )
                expected_tag = self._handshake_tag(
                    _WORKER_AUTH_DOMAIN,
                    challenge,
                    rank,
                )
                if (
                    magic != _HANDSHAKE_MAGIC
                    or version != _VERSION
                    or not 0 < rank < self.world_size
                    or rank in self._peers
                    or not hmac.compare_digest(observed_tag, expected_tag)
                ):
                    stream.close()
                    continue
                self._configure(stream)
                stream.sendall(
                    _HANDSHAKE_ACK.pack(
                        _HANDSHAKE_ACK_MAGIC,
                        _VERSION,
                        self._handshake_tag(
                            _COORDINATOR_AUTH_DOMAIN,
                            challenge,
                            rank,
                        ),
                    )
                )
                self._peers[rank] = stream
            except (OSError, TimeoutError, ConnectionError, struct.error):
                stream.close()
                continue

    def _authenticate_worker_stream(self, stream: socket.socket) -> None:
        challenge_magic, challenge_version, challenge = _HANDSHAKE_CHALLENGE.unpack(
            _recv_exact(stream, _HANDSHAKE_CHALLENGE.size)
        )
        if (
            challenge_magic != _HANDSHAKE_CHALLENGE_MAGIC
            or challenge_version != _VERSION
        ):
            raise RuntimeError("rank-control challenge is invalid")
        stream.sendall(
            _HANDSHAKE.pack(
                _HANDSHAKE_MAGIC,
                _VERSION,
                self.rank,
                self._handshake_tag(
                    _WORKER_AUTH_DOMAIN,
                    challenge,
                    self.rank,
                ),
            )
        )
        ack_magic, ack_version, ack_tag = _HANDSHAKE_ACK.unpack(
            _recv_exact(stream, _HANDSHAKE_ACK.size)
        )
        expected_ack = self._handshake_tag(
            _COORDINATOR_AUTH_DOMAIN,
            challenge,
            self.rank,
        )
        if (
            ack_magic != _HANDSHAKE_ACK_MAGIC
            or ack_version != _VERSION
            or not hmac.compare_digest(ack_tag, expected_ack)
        ):
            raise RuntimeError("rank-control handshake was not acknowledged")

    def _connect_to_coordinator(self) -> None:
        if should_proxy_control_socket(self.host):
            proxy = open_system_tcp_proxy(
                self.host,
                self.port,
                timeout=self._connect_timeout,
            )
            stream = proxy.stream
            try:
                self._configure(stream)
                self._authenticate_worker_stream(stream)
            except BaseException:
                proxy.close()
                raise
            self._stream_proxy = proxy
            self._stream = stream
            return
        deadline = time.monotonic() + self._connect_timeout
        last_error: OSError | None = None
        while time.monotonic() < deadline:
            stream = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                stream.settimeout(min(1.0, self._connect_timeout))
                stream.connect((self.host, self.port))
                self._configure(stream)
                self._authenticate_worker_stream(stream)
                self._stream = stream
                return
            except RuntimeError:
                stream.close()
                raise
            except OSError as exc:
                last_error = exc
                stream.close()
                time.sleep(0.05)
        raise TimeoutError(f"rank-control coordinator was unreachable: {last_error}")

    def broadcast_object(self, obj: Any) -> Any:
        """Broadcast one rank-zero-owned Python object in strict sequence."""

        self._sequence += 1
        if self.rank == 0:
            payload = pickle.dumps(obj) if obj is not None else b""
            if len(payload) > _MAX_OBJECT_BYTES:
                raise RuntimeError("rank-control object exceeds 256 MiB")
            checksum = zlib.crc32(payload)
            prefix = _HEADER_PREFIX.pack(
                _MESSAGE_MAGIC,
                _VERSION,
                self._sequence,
                len(payload),
                checksum,
            )
            tag = hmac.new(
                self._token,
                _MESSAGE_AUTH_DOMAIN + prefix + payload,
                hashlib.sha256,
            ).digest()
            header = _HEADER.pack(
                _MESSAGE_MAGIC,
                _VERSION,
                self._sequence,
                len(payload),
                checksum,
                tag,
            )
            packet = header + payload
            for rank in range(1, self.world_size):
                self._peers[rank].sendall(packet)
            return obj

        stream = self._stream
        if stream is None:
            raise RuntimeError("rank-control worker is not connected")
        magic, version, sequence, size, checksum, observed_tag = _HEADER.unpack(
            _recv_exact(stream, _HEADER.size)
        )
        if magic != _MESSAGE_MAGIC or version != _VERSION:
            raise RuntimeError("rank-control message header is invalid")
        if sequence != self._sequence:
            raise RuntimeError(
                f"rank-control sequence diverged: expected {self._sequence}, "
                f"received {sequence}"
            )
        if size > _MAX_OBJECT_BYTES:
            raise RuntimeError("rank-control object has an invalid size")
        payload = _recv_exact(stream, size) if size else b""
        if zlib.crc32(payload) != checksum:
            raise RuntimeError("rank-control object failed CRC32")
        prefix = _HEADER_PREFIX.pack(magic, version, sequence, size, checksum)
        expected_tag = hmac.new(
            self._token,
            _MESSAGE_AUTH_DOMAIN + prefix + payload,
            hashlib.sha256,
        ).digest()
        if not hmac.compare_digest(observed_tag, expected_tag):
            raise RuntimeError("rank-control object failed authentication")
        return pickle.loads(payload) if payload else None

    def close(self) -> None:
        for stream in self._peers.values():
            with suppress(OSError):
                stream.close()
        self._peers.clear()
        if self._stream is not None:
            with suppress(OSError):
                self._stream.close()
            self._stream = None
        if self._stream_proxy is not None:
            self._stream_proxy.close()
            self._stream_proxy = None
        if self._listener is not None:
            with suppress(OSError):
                self._listener.close()
            self._listener = None

    def __exit__(self, *_exc: Any) -> None:
        self.close()


__all__ = ["RankControlPlane"]
