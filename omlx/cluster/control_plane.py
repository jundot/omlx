# SPDX-License-Identifier: Apache-2.0
"""Small reliable rank-control channel kept separate from tensor collectives."""

from __future__ import annotations

import pickle
import os
import socket
import struct
import threading
import time
import zlib
from contextlib import AbstractContextManager
from typing import Any

from .system_socket_proxy import (
    SystemSocketProxy,
    open_system_tcp_proxy,
    should_proxy_control_socket,
)

_HANDSHAKE_MAGIC = b"OC2H"
_MESSAGE_MAGIC = b"OC2M"
_OWNED_BYTES_MAGIC = b"OC2B"
_BARRIER_MAGIC = b"OC2R"
_VERSION = 1
_HANDSHAKE = struct.Struct("!4sII64s")
_HEADER = struct.Struct("!4sIIII")
_OWNED_BYTES_HEADER = struct.Struct("!4sIIIII")
_BARRIER_PACKET = struct.Struct("!4sIII")
_MAX_OBJECT_BYTES = 256 * 1024 * 1024

_ACTIVE_LOCK = threading.Lock()
_ACTIVE_CONTROL_PLANE: "RankControlPlane | None" = None


def _trace_control(rank: int, sequence: int, operation: str, detail: str = "") -> None:
    if os.environ.get("OMLX_CLUSTER_TRACE_COLLECTIVES", "0").strip().lower() not in {
        "1",
        "true",
        "on",
        "yes",
    }:
        return
    print(
        "OMLX_CONTROL_TRACE:"
        + f"rank={rank} sequence={sequence} operation={operation} detail={detail}",
        flush=True,
    )


def _recv_exact(stream: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        part = stream.recv(size - len(chunks))
        if not part:
            raise ConnectionError("rank-control peer closed its socket")
        chunks.extend(part)
    return bytes(chunks)


def active_rank_control_plane() -> "RankControlPlane | None":
    """The process's connected rank-control channel, if serving a cluster.

    Distributed rank workers host exactly one model server per process.  The
    generation thread is created after the control channel connects, but it is
    a different Python thread, so a ``ContextVar`` or thread-local cannot carry
    the channel into model code.  A process-scoped reference matches the worker
    lifetime while keeping ordinary/single-node model imports independent.
    """

    with _ACTIVE_LOCK:
        return _ACTIVE_CONTROL_PLANE


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
        # Request/cancellation control and model-owned decisions normally run
        # on the same generation thread.  Serialize defensively so an
        # accidental second caller cannot interleave packet headers or consume
        # the shared sequence twice.
        self._operation_lock = threading.RLock()

    def __enter__(self) -> "RankControlPlane":
        global _ACTIVE_CONTROL_PLANE
        if self.rank == 0:
            self._accept_workers()
        else:
            self._connect_to_coordinator()
        with _ACTIVE_LOCK:
            _ACTIVE_CONTROL_PLANE = self
        return self

    def _configure(self, stream: socket.socket) -> None:
        stream.settimeout(self._io_timeout)
        stream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        stream.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

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
            self._configure(stream)
            try:
                magic, version, rank, token = _HANDSHAKE.unpack(
                    _recv_exact(stream, _HANDSHAKE.size)
                )
                if (
                    magic != _HANDSHAKE_MAGIC
                    or version != _VERSION
                    or not 0 < rank < self.world_size
                    or rank in self._peers
                    or token != self._token
                ):
                    raise RuntimeError("rank-control handshake is invalid")
                self._peers[rank] = stream
            except Exception:
                stream.close()
                raise

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
                stream.sendall(
                    _HANDSHAKE.pack(
                        _HANDSHAKE_MAGIC,
                        _VERSION,
                        self.rank,
                        self._token,
                    )
                )
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
                stream.sendall(
                    _HANDSHAKE.pack(
                        _HANDSHAKE_MAGIC,
                        _VERSION,
                        self.rank,
                        self._token,
                    )
                )
                self._stream = stream
                return
            except OSError as exc:
                last_error = exc
                stream.close()
                time.sleep(0.05)
        raise TimeoutError(
            f"rank-control coordinator was unreachable: {last_error}"
        )

    def broadcast_object(self, obj: Any) -> Any:
        """Broadcast one rank-zero-owned Python object in strict sequence."""

        with self._operation_lock:
            self._sequence += 1
            if self.rank == 0:
                payload = pickle.dumps(obj) if obj is not None else b""
                if len(payload) > _MAX_OBJECT_BYTES:
                    raise RuntimeError("rank-control object exceeds 256 MiB")
                header = _HEADER.pack(
                    _MESSAGE_MAGIC,
                    _VERSION,
                    self._sequence,
                    len(payload),
                    zlib.crc32(payload),
                )
                packet = header + payload
                _trace_control(
                    self.rank,
                    self._sequence,
                    "broadcast-send",
                    f"type={type(obj).__name__} bytes={len(payload)}",
                )
                for rank in range(1, self.world_size):
                    self._peers[rank].sendall(packet)
                return obj

            stream = self._stream
            if stream is None:
                raise RuntimeError("rank-control worker is not connected")
            magic, version, sequence, size, checksum = _HEADER.unpack(
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
            result = pickle.loads(payload) if payload else None
            _trace_control(
                self.rank,
                self._sequence,
                "broadcast-recv",
                f"type={type(result).__name__} bytes={len(payload)}",
            )
            return result

    def _owned_bytes_packet(
        self,
        stream: socket.socket,
        *,
        sequence: int,
        source_rank: int,
        expected_size: int,
    ) -> tuple[bytes, bytes]:
        header = _recv_exact(stream, _OWNED_BYTES_HEADER.size)
        magic, version, received_sequence, source, size, checksum = (
            _OWNED_BYTES_HEADER.unpack(header)
        )
        if magic != _OWNED_BYTES_MAGIC or version != _VERSION:
            raise RuntimeError("rank-control owned-bytes header is invalid")
        if received_sequence != sequence:
            raise RuntimeError(
                f"rank-control sequence diverged: expected {sequence}, "
                f"received {received_sequence}"
            )
        if source != source_rank:
            raise RuntimeError(
                f"rank-control owned-bytes source diverged: expected {source_rank}, "
                f"received {source}"
            )
        if size != expected_size or size > _MAX_OBJECT_BYTES:
            raise RuntimeError(
                f"rank-control owned-bytes size diverged: expected {expected_size}, "
                f"received {size}"
            )
        payload = _recv_exact(stream, size) if size else b""
        if zlib.crc32(payload) != checksum:
            raise RuntimeError("rank-control owned bytes failed CRC32")
        return header + payload, payload

    def broadcast_owned_bytes(
        self,
        payload: bytes | None,
        *,
        source_rank: int,
        expected_size: int,
    ) -> bytes:
        """Broadcast fixed-size bytes owned by any rank in strict sequence.

        Rank zero is the TCP hub.  A nonzero owner sends one framed packet to
        rank zero, which validates and forwards the exact packet to every other
        worker.  The source rank never receives its own payload back.  This is
        the fixed-schedule primitive needed by model decisions such as DS4's
        decode top-k indices: all ranks call it exactly once, the expected size
        is known before I/O, and no tensor reduction or pickle is involved.
        """

        if not 0 <= int(source_rank) < self.world_size:
            raise ValueError("rank-control owned-bytes source is invalid")
        if not 0 <= int(expected_size) <= _MAX_OBJECT_BYTES:
            raise ValueError("rank-control owned-bytes size is invalid")
        if payload is not None and not isinstance(payload, bytes):
            raise TypeError("rank-control owned payload must be bytes")
        if self.rank == source_rank:
            if payload is None or len(payload) != expected_size:
                raise RuntimeError(
                    "rank-control owned source produced an invalid payload"
                )
        elif payload is not None:
            raise RuntimeError("rank-control non-source supplied owned bytes")

        with self._operation_lock:
            self._sequence += 1
            sequence = self._sequence
            _trace_control(self.rank, sequence, "owned-bytes-enter")
            if self.rank == source_rank:
                header = _OWNED_BYTES_HEADER.pack(
                    _OWNED_BYTES_MAGIC,
                    _VERSION,
                    sequence,
                    source_rank,
                    expected_size,
                    zlib.crc32(payload),
                )
                packet = header + payload
            else:
                packet = b""

            if self.rank == 0:
                if source_rank == 0:
                    owned = payload
                else:
                    source = self._peers.get(source_rank)
                    if source is None:
                        raise RuntimeError("rank-control owned source is not connected")
                    packet, owned = self._owned_bytes_packet(
                        source,
                        sequence=sequence,
                        source_rank=source_rank,
                        expected_size=expected_size,
                    )
                for rank in range(1, self.world_size):
                    if rank != source_rank:
                        self._peers[rank].sendall(packet)
                return owned

            stream = self._stream
            if stream is None:
                raise RuntimeError("rank-control worker is not connected")
            if self.rank == source_rank:
                stream.sendall(packet)
                return payload
            _packet, owned = self._owned_bytes_packet(
                stream,
                sequence=sequence,
                source_rank=source_rank,
                expected_size=expected_size,
            )
            return owned

    def barrier(self) -> None:
        """Wait until every rank reaches one ordered control boundary.

        A rank-zero broadcast is not a barrier: ``sendall`` may return after
        copying into the kernel socket buffer while a worker is still finishing
        its prior Metal graph. Prompt-to-decode and terminal batch transitions
        need a two-way rendezvous so no rank can construct a differently shaped
        tensor collective early.
        """

        with self._operation_lock:
            self._sequence += 1
            sequence = self._sequence
            _trace_control(self.rank, sequence, "barrier-enter")
            if self.rank == 0:
                for rank in range(1, self.world_size):
                    stream = self._peers.get(rank)
                    if stream is None:
                        raise RuntimeError("rank-control barrier peer is not connected")
                    magic, version, received_sequence, received_rank = (
                        _BARRIER_PACKET.unpack(_recv_exact(stream, _BARRIER_PACKET.size))
                    )
                    if (
                        magic != _BARRIER_MAGIC
                        or version != _VERSION
                        or received_sequence != sequence
                        or received_rank != rank
                    ):
                        raise RuntimeError(
                            "rank-control barrier arrival is invalid: "
                            f"expected magic={_BARRIER_MAGIC!r} version={_VERSION} "
                            f"sequence={sequence} rank={rank}; received "
                            f"magic={magic!r} version={version} "
                            f"sequence={received_sequence} rank={received_rank}"
                        )
                release = _BARRIER_PACKET.pack(
                    _BARRIER_MAGIC,
                    _VERSION,
                    sequence,
                    0,
                )
                for rank in range(1, self.world_size):
                    self._peers[rank].sendall(release)
                _trace_control(self.rank, sequence, "barrier-release")
                return

            stream = self._stream
            if stream is None:
                raise RuntimeError("rank-control worker is not connected")
            stream.sendall(
                _BARRIER_PACKET.pack(
                    _BARRIER_MAGIC,
                    _VERSION,
                    sequence,
                    self.rank,
                )
            )
            magic, version, received_sequence, coordinator = _BARRIER_PACKET.unpack(
                _recv_exact(stream, _BARRIER_PACKET.size)
            )
            if (
                magic != _BARRIER_MAGIC
                or version != _VERSION
                or received_sequence != sequence
                or coordinator != 0
            ):
                raise RuntimeError(
                    "rank-control barrier release is invalid: "
                    f"expected magic={_BARRIER_MAGIC!r} version={_VERSION} "
                    f"sequence={sequence} coordinator=0; received "
                    f"magic={magic!r} version={version} "
                    f"sequence={received_sequence} coordinator={coordinator}"
                )
            _trace_control(self.rank, sequence, "barrier-exit")

    def close(self) -> None:
        global _ACTIVE_CONTROL_PLANE
        with _ACTIVE_LOCK:
            if _ACTIVE_CONTROL_PLANE is self:
                _ACTIVE_CONTROL_PLANE = None
        for stream in self._peers.values():
            try:
                stream.close()
            except OSError:
                pass
        self._peers.clear()
        if self._stream is not None:
            try:
                self._stream.close()
            except OSError:
                pass
            self._stream = None
        if self._stream_proxy is not None:
            self._stream_proxy.close()
            self._stream_proxy = None
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
            self._listener = None

    def __exit__(self, *_exc: Any) -> None:
        self.close()


__all__ = ["RankControlPlane", "active_rank_control_plane"]
