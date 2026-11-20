# SPDX-License-Identifier: Apache-2.0
"""Reliable JACCL metadata exchange for distributed oMLX workers.

JACCL uses a tiny TCP all-gather only while it creates its RDMA queue pairs.
The native helper is normally sufficient, but a direct Thunderbolt interface
can occasionally leave its client bootstrap sockets in ``SYN_SENT`` even
though ordinary TCP on the same interface succeeds.  MLX exposes a supported
``all_gather_factory`` specifically for replacing that bootstrap path.  This
module supplies a bounded, ordered implementation without changing the RDMA
data path used after initialization.
"""

from __future__ import annotations

import os
import selectors
import socket
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from .coordinator_route import coordinator_addresses, listen_addresses

_RANK_BYTES = struct.Struct("!I")
_DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS = 20.0
_DEFAULT_CONNECT_RETRY_SECONDS = 0.05
_MAX_METADATA_BYTES = 8 * 1024 * 1024
_SYSTEM_PYTHON = "/usr/bin/python3"

# The serving interpreter can be denied macOS Local Network access even when
# Apple's system Python is allowed to use the same direct Thunderbolt IP.  This
# tiny, standard-library-only helper owns only the one-time JACCL metadata
# socket.  The parent still owns every RDMA queue pair and collective.
_SIDECAR_PROGRAM = r"""
import select, socket, struct, sys, time

RANK = struct.Struct("!I")

def recv_exact(stream, count, allow_eof=False):
    parts = []
    remaining = count
    while remaining:
        part = stream.read(remaining)
        if not part:
            if allow_eof and not parts:
                return None
            raise RuntimeError("side-channel stream closed")
        parts.append(part)
        remaining -= len(part)
    return b"".join(parts)

def sock_recv_exact(peer, count):
    parts = []
    remaining = count
    while remaining:
        part = peer.recv(remaining)
        if not part:
            raise RuntimeError("side-channel peer disconnected")
        parts.append(part)
        remaining -= len(part)
    return b"".join(parts)

def connect(host, port, timeout):
    deadline = time.monotonic() + timeout
    error = None
    while time.monotonic() < deadline:
        peer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            peer.settimeout(min(1.0, max(0.05, deadline - time.monotonic())))
            peer.connect((host, port))
            peer.settimeout(timeout)
            return peer
        except OSError as exc:
            error = exc
            peer.close()
            time.sleep(0.05)
    raise RuntimeError("side-channel could not reach coordinator: %s" % error)

def accept(listeners, timeout):
    for listener in select.select(listeners, [], [], timeout)[0]:
        try:
            return listener.accept()[0]
        except BlockingIOError:
            pass
    return None

def bootstrap(hosts, port, rank, size, timeout):
    if rank:
        peer = connect(hosts[0], port, timeout)
        peer.sendall(RANK.pack(rank))
        return peer
    listeners = []
    try:
        for host in hosts:
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listeners.append(listener)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((host, port))
            listener.listen(size)
            listener.setblocking(False)
        peers = [None] * size
        deadline = time.monotonic() + timeout
        while any(peer is None for peer in peers[1:]):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("side-channel timed out waiting for ranks")
            peer = accept(listeners, remaining)
            if peer is None:
                continue
            try:
                peer.settimeout(timeout)
                peer_rank = RANK.unpack(sock_recv_exact(peer, RANK.size))[0]
                if not 1 <= peer_rank < size or peers[peer_rank] is not None:
                    raise RuntimeError("side-channel received invalid rank")
            except BaseException:
                peer.close()
                raise
            peers[peer_rank] = peer
        return peers
    finally:
        for listener in listeners:
            listener.close()

def main():
    hosts, port, rank, size, timeout = (
        sys.argv[1].split(","), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
    )
    peers = bootstrap(hosts, port, rank, size, timeout)
    source, sink = sys.stdin.buffer, sys.stdout.buffer
    while True:
        header = recv_exact(source, RANK.size, allow_eof=True)
        if header is None:
            return
        count = RANK.unpack(header)[0]
        payload = recv_exact(source, count)
        if rank == 0:
            gathered = [payload]
            for peer in peers[1:]:
                gathered.append(sock_recv_exact(peer, count))
            result = b"".join(gathered)
            for peer in peers[1:]:
                peer.sendall(result)
        else:
            peers.sendall(payload)
            result = sock_recv_exact(peers, size * count)
        sink.write(result)
        sink.flush()

if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print("oMLX JACCL side-channel helper failed: %s" % exc, file=sys.stderr, flush=True)
        raise
"""


def _coordinator_endpoint() -> tuple[str, int]:
    raw = os.environ.get("MLX_JACCL_COORDINATOR", "").strip()
    host, separator, port_text = raw.rpartition(":")
    if not separator or not host or not port_text.isdecimal():
        raise RuntimeError(
            "JACCL side channel requires MLX_JACCL_COORDINATOR as IPv4:port"
        )
    port = int(port_text)
    if not 1 <= port <= 65535:
        raise RuntimeError("JACCL side channel coordinator port is invalid")
    try:
        socket.inet_aton(host)
    except OSError as exc:
        raise RuntimeError(
            "JACCL side channel currently requires an IPv4 coordinator"
        ) from exc
    return host, port


def _coordinator_route(rank: int, size: int) -> tuple[tuple[str, ...], int]:
    """Where rank zero listens, or the one address another rank connects to.

    ``MLX_JACCL_COORDINATOR`` is the same for every rank. Macs cabled as a mesh
    reach rank zero on a different address each, and rank zero then listens on
    all of them.
    """

    host, port = _coordinator_endpoint()
    addresses = coordinator_addresses(size, host)
    if rank == 0:
        return listen_addresses(addresses), port
    return (addresses[rank],), port


def _accept(listeners: Sequence[socket.socket], timeout: float) -> socket.socket | None:
    """The next connection on any of ``listeners``, if one arrives in time."""

    # selectors rather than select(): a rank that holds many open files can
    # hand these sockets descriptors select() cannot address.
    with selectors.DefaultSelector() as selector:
        for listener in listeners:
            selector.register(listener, selectors.EVENT_READ)
        for key, _events in selector.select(timeout):
            try:
                return key.fileobj.accept()[0]
            except BlockingIOError:
                continue
    return None


def _recv_exact(sock: socket.socket, n_bytes: int) -> bytes:
    chunks: list[bytes] = []
    remaining = n_bytes
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise RuntimeError("JACCL side channel peer disconnected")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_exact_stream(stream: object, n_bytes: int) -> bytes:
    """Read a fixed frame from a local sidecar pipe."""

    chunks: list[bytes] = []
    remaining = n_bytes
    read = stream.read
    while remaining:
        chunk = read(remaining)
        if not chunk:
            raise RuntimeError("JACCL side-channel helper closed its pipe")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_all(sock: socket.socket, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = sock.send(view)
        if written <= 0:
            raise RuntimeError("JACCL side channel peer rejected metadata")
        view = view[written:]


class _SocketAllGather:
    """One rank's persistent, serialized bootstrap TCP connection set."""

    def __init__(self, rank: int, size: int) -> None:
        if not 0 <= rank < size or size < 2:
            raise RuntimeError("JACCL side channel received invalid rank metadata")
        self.rank = rank
        self.size = size
        self.timeout = _DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS
        self._lock = threading.Lock()
        self._peers: list[socket.socket] | socket.socket
        hosts, port = _coordinator_route(rank, size)
        if rank == 0:
            self._peers = self._accept_peers(hosts, port)
        else:
            self._peers = self._connect(hosts[0], port)

    def _accept_peers(self, hosts: Sequence[str], port: int) -> list[socket.socket]:
        servers: list[socket.socket] = []
        try:
            for host in hosts:
                server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                servers.append(server)
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind((host, port))
                server.listen(self.size)
                server.setblocking(False)
            peers: list[socket.socket | None] = [None] * self.size
            deadline = time.monotonic() + self.timeout
            while any(peer is None for peer in peers[1:]):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        "JACCL side channel timed out waiting for every rank"
                    )
                peer = _accept(servers, remaining)
                if peer is None:
                    continue
                try:
                    peer.settimeout(self.timeout)
                    peer_rank = _RANK_BYTES.unpack(_recv_exact(peer, _RANK_BYTES.size))[
                        0
                    ]
                    if not 1 <= peer_rank < self.size or peers[peer_rank] is not None:
                        raise RuntimeError(
                            "JACCL side channel received an invalid peer rank"
                        )
                except BaseException:
                    peer.close()
                    raise
                peers[peer_rank] = peer
            return [peer for peer in peers[1:] if peer is not None]
        except BaseException:
            for peer in locals().get("peers", []):
                if peer is not None:
                    peer.close()
            raise
        finally:
            for server in servers:
                server.close()

    def _connect(self, host: str, port: int) -> socket.socket:
        deadline = time.monotonic() + self.timeout
        last_error: OSError | None = None
        while time.monotonic() < deadline:
            peer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                peer.settimeout(min(1.0, max(0.05, deadline - time.monotonic())))
                peer.connect((host, port))
                peer.settimeout(self.timeout)
                _send_all(peer, _RANK_BYTES.pack(self.rank))
                return peer
            except OSError as exc:
                last_error = exc
                peer.close()
                time.sleep(_DEFAULT_CONNECT_RETRY_SECONDS)
        detail = f": {last_error}" if last_error else ""
        raise RuntimeError(f"JACCL side channel could not reach coordinator{detail}")

    def __call__(self, src: bytes, n_bytes: int) -> bytes:
        if n_bytes < 0 or len(src) != n_bytes:
            raise RuntimeError("JACCL side channel received malformed metadata")
        with self._lock:
            if self.rank == 0:
                assert isinstance(self._peers, list)
                gathered = [src]
                for peer in self._peers:
                    gathered.append(_recv_exact(peer, n_bytes))
                result = b"".join(gathered)
                for peer in self._peers:
                    _send_all(peer, result)
                return result
            assert isinstance(self._peers, socket.socket)
            _send_all(self._peers, src)
            return _recv_exact(self._peers, self.size * n_bytes)


def _sidecar_python() -> str | None:
    """Return Apple's system Python when it is usable as a network carrier."""

    candidate = Path(_SYSTEM_PYTHON)
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return None


def _side_channel_transport() -> str:
    # The helper is off the hot path and prevents a macOS Local Network
    # entitlement mismatch from making an otherwise verified TB fabric
    # unusable. Non-macOS hosts retain the direct standard-library path.
    return "sidecar" if sys.platform == "darwin" and _sidecar_python() else "direct"


class _SidecarAllGather:
    """A local-pipe adapter whose bootstrap sockets live in system Python."""

    def __init__(self, rank: int, size: int) -> None:
        if not 0 <= rank < size or size < 2:
            raise RuntimeError("JACCL side channel received invalid rank metadata")
        hosts, port = _coordinator_route(rank, size)
        self.rank = rank
        self.size = size
        self.timeout = _DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS
        self._lock = threading.Lock()
        executable = _sidecar_python()
        if executable is None:  # defensive: transport selection already checks
            raise RuntimeError("JACCL side-channel helper Python is unavailable")
        # -I: the bundled app exports PYTHONHOME for its own interpreter, and
        # system Python cannot start under that home.
        self._process = subprocess.Popen(
            [
                executable,
                "-I",
                "-u",
                "-c",
                _SIDECAR_PROGRAM,
                ",".join(hosts),
                str(port),
                str(rank),
                str(size),
                str(self.timeout),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

    def _failure(self) -> RuntimeError:
        status = self._process.poll()
        detail = ""
        if status is not None and self._process.stderr is not None:
            detail = self._process.stderr.read().decode("utf-8", "replace").strip()
        suffix = f": {detail[-1000:]}" if detail else ""
        return RuntimeError(
            f"JACCL side-channel helper stopped (status={status}){suffix}"
        )

    def __call__(self, src: bytes, n_bytes: int) -> bytes:
        if n_bytes < 0 or n_bytes > _MAX_METADATA_BYTES or len(src) != n_bytes:
            raise RuntimeError("JACCL side channel received malformed metadata")
        if self._process.stdin is None or self._process.stdout is None:
            raise self._failure()
        with self._lock:
            try:
                self._process.stdin.write(_RANK_BYTES.pack(n_bytes) + src)
                self._process.stdin.flush()
                result = _read_exact_stream(self._process.stdout, self.size * n_bytes)
            except (BrokenPipeError, OSError, RuntimeError) as exc:
                raise self._failure() from exc
        return result

    def close(self) -> None:
        """Close the local control pipe; used by isolated verification tests."""

        if self._process.stdin is not None and not self._process.stdin.closed:
            self._process.stdin.close()
        try:
            self._process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            self._process.wait(timeout=1.0)


def jaccl_all_gather_factory(rank: int, size: int) -> Callable[[bytes, int], bytes]:
    """Return JACCL's supported metadata all-gather callable for one rank."""

    if _side_channel_transport() == "sidecar":
        return _SidecarAllGather(rank, size)
    return _SocketAllGather(rank, size)


def _mlx_supports_all_gather_factory(distributed: object) -> bool:
    """Detect the *loaded* nanobind API, not an optionally newer stub file.

    Development installs can briefly pair an updated ``distributed.pyi`` with
    an older ``core`` extension.  Retrying through a TypeError would be safe,
    but inspecting the native function documentation first keeps a failed
    JACCL load from looking like a fabric failure.
    """

    init = distributed.init
    return "all_gather_factory" in (getattr(init, "__doc__", "") or "")


def init_cluster_group(
    mx: object,
    *,
    backend: str | None = None,
    strict: bool = False,
) -> object:
    """Initialize a distributed group with the resilient JACCL bootstrap.

    Ring execution remains byte-for-byte on MLX's normal path.  For a
    JACCL worker, only connection metadata travels through this small TCP
    control channel; collectives immediately thereafter remain RDMA/JACCL.
    """

    distributed = mx.distributed
    selected = backend
    if selected is None and os.environ.get("MLX_JACCL_COORDINATOR"):
        selected = "jaccl"
    if selected == "jaccl" and _mlx_supports_all_gather_factory(distributed):
        return distributed.init(
            backend="jaccl",
            strict=strict,
            all_gather_factory=jaccl_all_gather_factory,
        )
    if selected is None:
        return distributed.init(strict=strict)
    return distributed.init(backend=selected, strict=strict)
