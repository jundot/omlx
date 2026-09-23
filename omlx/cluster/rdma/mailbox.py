# SPDX-License-Identifier: Apache-2.0
"""Attach to one mcdma-rpcd mailbox as its client or its service; never opens a verbs context."""

from __future__ import annotations

import mmap
import os
import secrets
import select
import socket
from collections.abc import Iterable
from contextlib import suppress
from typing import Any

import numpy as np

from . import layout
from .words import WordOps


class MailboxError(RuntimeError):
    """A mailbox could not be attached or used."""


def _read_line(control: socket.socket, limit: int = 256) -> bytes:
    """One answer line from the daemon, read byte by byte so nothing after it is consumed."""
    received = b""
    while not received.endswith(b"\n") and len(received) < limit:
        chunk = control.recv(1)
        if not chunk:
            break
        received += chunk
    return received.strip()


class _Mailbox:
    """Bounds-checked access to one attached mailbox buffer."""

    def __init__(self, buffer: memoryview, ops: WordOps) -> None:
        self._buffer = buffer
        self._bytes = np.frombuffer(buffer, dtype=np.uint8)
        self._base = int(self._bytes.ctypes.data)
        self._ops = ops
        try:
            self.sizes = layout.read_sizes(buffer)
        except ValueError as exc:
            raise MailboxError(str(exc)) from exc

    def _load(self, offset: int) -> int:
        return int.from_bytes(self._buffer[offset : offset + 8], "little")

    def _store(self, offset: int, value: int) -> None:
        self._ops.store(self._base + offset, value)

    def _wait(self, offset: int, seq: int, *, equal: bool, timeout_s: float) -> int:
        return self._ops.wait(
            self._base + offset, seq, equal=equal, timeout_s=timeout_s
        )

    def _write(self, start: int, limit: int, parts: Iterable[Any]) -> int:
        offset = start
        for part in parts:
            view = memoryview(part).cast("B")
            end = offset + view.nbytes
            if end > limit:
                raise MailboxError("payload is larger than the mailbox half")
            self._buffer[offset:end] = view
            offset = end
        return offset - start

    def _release(self) -> None:
        self._bytes = None
        # A caller may still hold a payload view; the mapping then goes with the process.
        with suppress(BufferError):
            self._buffer.release()


class ClientMailbox(_Mailbox):
    """The connecting end: stage a request, then wait for the reply that answers it."""

    def __init__(self, name: str, shared: Any, ops: WordOps) -> None:
        self.name = name
        self._shared = shared
        super().__init__(shared.buf, ops)
        # A random start keeps a restarted client from reusing a served sequence.
        self._seq = secrets.randbits(31) | 1
        self._staged_generation = self.generation

    @classmethod
    def attach(cls, name: str, ops: WordOps) -> ClientMailbox:
        """Attach to the daemon-owned shared memory object of link `name`."""
        from multiprocessing import resource_tracker, shared_memory

        shm_name = layout.client_shm_name(name)
        try:
            try:
                shared = shared_memory.SharedMemory(
                    name=shm_name, create=False, track=False
                )
                tracked = False
            except TypeError:
                # Before Python 3.13 every attach is tracked and unlinked at exit.
                shared = shared_memory.SharedMemory(name=shm_name, create=False)
                tracked = True
        except FileNotFoundError as exc:
            raise MailboxError(
                f"no mailbox /{shm_name}: is mcdma-rpcd running?"
            ) from exc
        except OSError as exc:
            raise MailboxError(f"cannot open mailbox /{shm_name}: {exc}") from exc
        if tracked:
            with suppress(Exception):
                resource_tracker.unregister(f"/{shm_name}", "shared_memory")
        return cls(name, shared, ops)

    @property
    def connected(self) -> bool:
        """Whether the daemon reports its link to the peer as up."""
        return self._load(layout.CONNECTED_FLAG) == 1

    @property
    def generation(self) -> int:
        """How many times the daemon's link has come up."""
        return self._load(layout.GENERATION_WORD)

    @property
    def reconnected(self) -> bool:
        """Whether the link came up again after the last request was staged, which loses that request."""
        return self.generation != self._staged_generation

    def stage(self, parts: Iterable[Any]) -> int:
        """Copy a request into the request half and publish it; returns its sequence."""
        length = self._write(layout.CTRL, self.sizes.request, parts)
        self._seq = layout.next_seq(self._seq)
        # Read before publishing: a reconnect in between then fails the wait rather than hiding a lost request.
        self._staged_generation = self.generation
        self._store(layout.REQUEST_WORD, layout.pack_word(self._seq, length))
        return self._seq

    def wait(self, seq: int, timeout_s: float) -> memoryview | None:
        """The reply payload of request `seq`, or None if it has not landed yet."""
        done = self.sizes.request + layout.DONE_WORD
        found = self._wait(done, seq, equal=True, timeout_s=timeout_s)
        if not found:
            return None
        start = self.sizes.request + layout.CTRL
        length = layout.word_length(found)
        if length > self.sizes.max_reply:
            raise MailboxError("reply length exceeds the reply half")
        return self._buffer[start : start + length]

    def close(self) -> None:
        """Detach; the daemon keeps the shared memory object."""
        self._release()
        with suppress(BufferError):
            self._shared.close()


class ServiceMailbox(_Mailbox):
    """The listening end: take requests landed by the peer and stage replies to them."""

    def __init__(
        self, name: str, mapping: mmap.mmap, control: socket.socket, ops: WordOps
    ) -> None:
        self.name = name
        self._mapping = mapping
        self._control = control
        super().__init__(memoryview(mapping), ops)
        # Whatever request is already there was meant for a service that is gone.
        self._last = layout.word_seq(self._load(layout.REQUEST_WORD))

    @classmethod
    def attach(
        cls,
        name: str,
        socket_path: str,
        ops: WordOps,
        *,
        mailbox_path: str | None = None,
    ) -> ServiceMailbox:
        """Map the daemon's mailbox file and register with it as a polling service."""
        path = mailbox_path or layout.service_mailbox_path(name)
        try:
            descriptor = os.open(path, os.O_RDWR)
        except OSError as exc:
            raise MailboxError(f"cannot open mailbox {path}: {exc}") from exc
        try:
            mapping = mmap.mmap(descriptor, os.fstat(descriptor).st_size)
        except (OSError, ValueError) as exc:
            raise MailboxError(f"cannot map mailbox {path}: {exc}") from exc
        finally:
            os.close(descriptor)
        control = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            control.settimeout(5.0)
            control.connect(socket_path)
            control.sendall(b"MODE poll\n")
            answer = _read_line(control)
            control.settimeout(None)
        except OSError as exc:
            control.close()
            mapping.close()
            raise MailboxError(
                f"cannot register with mcdma-rpcd at {socket_path}: {exc}"
            ) from exc
        if answer != b"OK":
            control.close()
            mapping.close()
            refusal = answer.decode(errors="replace") or "no answer"
            raise MailboxError(
                f"mcdma-rpcd at {socket_path} refused the service: {refusal}"
            )
        return cls(name, mapping, control, ops)

    @property
    def alive(self) -> bool:
        """Whether the daemon still holds the registration; it sends nothing until it ends it."""
        try:
            readable, _, _ = select.select([self._control], [], [], 0)
        except (OSError, ValueError):
            return False
        return not readable

    def next_request(self, timeout_s: float) -> tuple[int, memoryview] | None:
        """The next landed request as (sequence, payload), or None after `timeout_s`."""
        found = self._wait(
            layout.REQUEST_WORD, self._last, equal=False, timeout_s=timeout_s
        )
        if not found:
            return None
        length = layout.word_length(found)
        if length > self.sizes.max_request:
            raise MailboxError("request length exceeds the request half")
        self._last = layout.word_seq(found)
        return self._last, self._buffer[layout.CTRL : layout.CTRL + length]

    def reply(self, seq: int, parts: Iterable[Any]) -> None:
        """Copy a reply into the reply half and hand it to the daemon."""
        start = self.sizes.request + layout.CTRL
        length = self._write(start, self.sizes.total, parts)
        self._store(
            self.sizes.request + layout.STAGED_WORD, layout.pack_word(seq, length)
        )

    def close(self) -> None:
        """Detach from the daemon; it keeps the mailbox file."""
        self._release()
        self._control.close()
        with suppress(BufferError):
            self._mapping.close()
