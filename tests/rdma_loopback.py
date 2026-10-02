# SPDX-License-Identifier: Apache-2.0
"""A Python stand-in for mcdma-rpcd's direct mode, so RDMA tests run without hardware."""

from __future__ import annotations

import _posixshmem
import ctypes
import mmap
import os
import secrets
import select
import shutil
import socket
import tempfile
import threading
import time
from contextlib import suppress
from multiprocessing import resource_tracker, shared_memory

from omlx.cluster.rdma import layout


class PythonWordOps:
    """WordOps for tests: plain ctypes loads and stores, without the helper's memory ordering."""

    def wait(self, address: int, seq: int, *, equal: bool, timeout_s: float) -> int:
        cell = ctypes.c_uint64.from_address(address)
        deadline = time.monotonic() + timeout_s
        while True:
            value = int(cell.value)
            current = layout.word_seq(value)
            if (current == seq) if equal else (current != 0 and current != seq):
                return value
            if time.monotonic() > deadline:
                return 0
            time.sleep(0.0001)

    def store(self, address: int, value: int) -> None:
        ctypes.c_uint64.from_address(address).value = value


def _set(buffer: memoryview, offset: int, value: int) -> None:
    buffer[offset : offset + 8] = int(value).to_bytes(8, "little")


def _get(buffer: memoryview, offset: int) -> int:
    return int.from_bytes(buffer[offset : offset + 8], "little")


class LoopbackLink:
    """One link: a client shared memory object and a service file, joined by a forwarding thread."""

    def __init__(
        self, *, request_bytes: int = 1 << 20, reply_bytes: int = 1 << 20
    ) -> None:
        self.name = "t" + secrets.token_hex(4)
        self.request = request_bytes
        self.reply = reply_bytes
        size = request_bytes + reply_bytes
        self._shm = shared_memory.SharedMemory(
            name=layout.client_shm_name(self.name), create=True, size=size
        )
        with suppress(Exception):
            resource_tracker.unregister(
                f"/{layout.client_shm_name(self.name)}", "shared_memory"
            )
        self.directory = tempfile.mkdtemp(prefix="rdma-", dir="/tmp")
        self.mailbox_path = os.path.join(self.directory, "service.box")
        self.socket_path = os.path.join(self.directory, "service.sock")
        with open(self.mailbox_path, "wb") as stream:
            stream.truncate(size)
        descriptor = os.open(self.mailbox_path, os.O_RDWR)
        try:
            self._service_map = mmap.mmap(descriptor, size)
        finally:
            os.close(descriptor)
        self.client = self._shm.buf
        self.service = memoryview(self._service_map)
        for view in (self.client, self.service):
            _set(view, layout.SIZES, request_bytes)
            _set(view, layout.SIZES + 8, reply_bytes)
        _set(self.client, layout.CONNECTED_FLAG, 1)
        _set(self.client, layout.GENERATION_WORD, 1)
        self._flap = False
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(self.socket_path)
        self._listener.listen(1)
        self._listener.settimeout(0.05)
        self.service_connection: socket.socket | None = None
        self.mode_lines: list[bytes] = []
        # What STATUS on the service socket reports for this link, as the listen daemon would.
        self.link_up = True
        self.corrupt_next_reply = False
        # While set, staged replies wait in the mailbox as if the daemon had not reached them yet.
        self.hold_replies = False
        # Replies carried from the service to the client, so tests can prove traffic used the link.
        self.replies = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        last_request = layout.word_seq(_get(self.client, layout.REQUEST_WORD))
        last_staged = layout.word_seq(
            _get(self.service, self.request + layout.STAGED_WORD)
        )
        while not self._stop.is_set():
            if select.select([self._listener], [], [], 0)[0]:
                self._register(self._listener.accept()[0])
            if self._flap:
                # A reconnect drops the staged request, clears the landed word and ends the service's registration.
                self._flap = False
                if self.service_connection is not None:
                    self.service_connection.sendall(b"BYE\n")
                    self.service_connection.close()
                    self.service_connection = None
                last_request = layout.word_seq(_get(self.client, layout.REQUEST_WORD))
                _set(self.service, layout.REQUEST_WORD, 0)
                _set(
                    self.client,
                    layout.GENERATION_WORD,
                    _get(self.client, layout.GENERATION_WORD) + 1,
                )
            word = _get(self.client, layout.REQUEST_WORD)
            if layout.word_seq(word) and layout.word_seq(word) != last_request:
                last_request = layout.word_seq(word)
                length = layout.word_length(word)
                self.service[layout.CTRL : layout.CTRL + length] = self.client[
                    layout.CTRL : layout.CTRL + length
                ]
                _set(self.service, layout.REQUEST_WORD, word)
            staged = _get(self.service, self.request + layout.STAGED_WORD)
            if (
                not self.hold_replies
                and layout.word_seq(staged)
                and layout.word_seq(staged) != last_staged
            ):
                last_staged = layout.word_seq(staged)
                length = layout.word_length(staged)
                _set(self.service, self.request + layout.READY_WORD, staged)
                start = self.request + layout.CTRL
                self.client[start : start + length] = self.service[
                    start : start + length
                ]
                if self.corrupt_next_reply and length:
                    self.client[start] ^= 0xFF
                    self.corrupt_next_reply = False
                _set(self.client, self.request + layout.DONE_WORD, staged)
                self.replies += 1
            time.sleep(0.00005)

    def _register(self, connection: socket.socket) -> None:
        connection.settimeout(1.0)
        line = connection.recv(64)
        if line == b"STATUS\n":
            state = "up" if self.link_up else "down"
            connection.sendall(
                f"VERSION mcdma-rpcd 1 test\nPEER {self.name} {state} calls 0 failures 0 MiB 0\nEND\n".encode()
            )
            connection.close()
            return
        self.mode_lines.append(line)
        if self.service_connection is None and line == b"MODE poll\n":
            connection.sendall(b"OK\n")
            self.service_connection = connection
        else:
            connection.sendall(b"ERR busy\n")
            connection.close()

    def say_bye(self) -> None:
        """End the service's registration the way a daemon does on SHUTDOWN."""
        if self.service_connection is not None:
            self.service_connection.sendall(b"BYE\n")

    def wait_service(self, timeout_s: float = 60.0) -> None:
        """Block until a service registers, as the stage-link vote guarantees in a deployment."""
        deadline = time.monotonic() + timeout_s
        while self.service_connection is None:
            if time.monotonic() > deadline:
                raise TimeoutError("no service registered")
            time.sleep(0.01)

    def wait_landed(self, seq: int, timeout_s: float = 2.0) -> None:
        """Block until request `seq` has landed in the service mailbox."""
        deadline = time.monotonic() + timeout_s
        while layout.word_seq(_get(self.service, layout.REQUEST_WORD)) != seq:
            if time.monotonic() > deadline:
                raise TimeoutError(f"request {seq} never landed")
            time.sleep(0.001)

    def flap(self) -> None:
        """Reset the link faster than a waiter polls, losing whatever request was in flight."""
        self._flap = True

    def drop(self) -> None:
        """Report the link down to the client and detach the service."""
        _set(self.client, layout.CONNECTED_FLAG, 0)
        if self.service_connection is not None:
            self.service_connection.close()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        if self.service_connection is not None:
            self.service_connection.close()
        self._listener.close()
        self.service.release()
        self._service_map.close()
        with suppress(BufferError):
            self._shm.close()
        # Unlink directly: tracking was dropped at creation, as for a daemon's object.
        _posixshmem.shm_unlink(f"/{layout.client_shm_name(self.name)}")
        shutil.rmtree(self.directory, ignore_errors=True)
