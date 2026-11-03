# SPDX-License-Identifier: Apache-2.0
"""This Mac's side of a prefill helper connection.

A receiver thread fills numpy buffers; MLX arrays are made on the caller's
thread when a chunk is taken. Taking a chunk acknowledges it, which lets the
helper compute the next one. While any job is open, a tiny GPU kernel ticks
on a separate stream: the scheduler waits for the helper between chunks, and
an idle GPU stalls the next command buffer for seconds.
"""

from __future__ import annotations

import contextlib
import itertools
import queue
import socket
import threading
import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from . import wire

_KEEP_WARM_SECONDS = 0.5


class HelperError(RuntimeError):
    """The helper failed or went away; the request prefills here instead."""


@dataclass
class Chunk:
    index: int
    start: int
    size: int
    seconds: float
    boundary: dict[str, mx.array]
    state: dict[str, mx.array]


class HelperJob:
    def __init__(self, client: HelperClient, job_id: str, chunks: list[int]):
        self.client, self.job_id, self.chunks = client, job_id, chunks
        self._frames: queue.Queue = queue.Queue()
        self.finished = False

    def ready(self) -> bool:
        """True when the next chunk (or an error) is waiting."""
        return not self._frames.empty()

    def take(self, timeout: float | None = None) -> Chunk:
        """The next chunk as MLX arrays. Raises HelperError on failure."""
        try:
            meta, arrays = self._frames.get(timeout=timeout)
        except queue.Empty as error:
            raise HelperError("prefill helper did not answer in time") from error
        if meta.get("op") != "chunk":
            self._finish()
            raise HelperError(meta.get("error") or "prefill helper stopped")
        self.client._send({"op": "ack", "job_id": self.job_id, "index": meta["index"]})
        boundary, state = {}, {}
        for name, value in arrays.items():
            target = boundary if name.startswith("b:") else state
            target[name[2:]] = wire.from_host(*value)
        if meta["index"] == len(self.chunks) - 1:
            self._finish()
        return Chunk(
            meta["index"], meta["start"], meta["size"], meta["seconds"], boundary, state
        )

    def cancel(self) -> None:
        if not self.finished:
            self.client._send({"op": "cancel", "job_id": self.job_id})
            self._finish()

    def _deliver(self, meta: dict, arrays: dict) -> None:
        self._frames.put((meta, arrays))

    def _finish(self) -> None:
        if not self.finished:
            self.finished = True
            self.client._job_closed(self)


class HelperClient:
    """One authenticated connection to a running helper."""

    def __init__(self, host: str, port: int, token: bytes, *, timeout: float = 10.0):
        sock = socket.create_connection((host, port), timeout=timeout)
        try:
            wire.tune(sock)
            wire.authenticate(sock, token, server=False)
            meta, _ = wire.recv(sock)
            if meta.get("op") != "hello" or meta.get("protocol") != wire.PROTOCOL:
                raise wire.ProtocolError("unexpected prefill helper greeting")
        except BaseException:
            sock.close()
            raise
        sock.settimeout(None)
        self.sock, self.info = sock, meta
        self._jobs: dict[str, HelperJob] = {}
        self._ids = itertools.count()
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._open = threading.Event()
        self.alive = True
        threading.Thread(target=self._receive, daemon=True).start()
        threading.Thread(target=self._keep_warm, daemon=True).start()

    def submit(
        self,
        tokens: np.ndarray,
        chunks: list[int],
        prefix: dict[str, mx.array] | None = None,
    ) -> HelperJob:
        if not self.alive:
            raise HelperError("prefill helper connection is closed")
        job = HelperJob(self, f"job-{next(self._ids)}", list(chunks))
        with self._lock:
            self._jobs[job.job_id] = job
            self._open.set()
        arrays = {"tokens": (np.asarray(tokens, np.int32).reshape(1, -1), "int32")}
        for name, value in (prefix or {}).items():
            arrays[f"p:{name}"] = wire.to_host(value)
        self._send({"op": "job", "job_id": job.job_id, "chunks": job.chunks}, arrays)
        return job

    def close(self) -> None:
        if self.alive:
            with contextlib.suppress(HelperError):
                self._send({"op": "bye"})
        self._fail("prefill helper connection closed")
        with contextlib.suppress(OSError):
            self.sock.close()

    def _send(self, meta: dict, arrays: dict | None = None) -> None:
        try:
            with self._send_lock:
                wire.send(self.sock, meta, arrays)
        except OSError as error:
            self._fail(f"prefill helper connection lost: {error}")
            raise HelperError("prefill helper connection lost") from error

    def _receive(self) -> None:
        try:
            while True:
                meta, arrays = wire.recv(self.sock)
                with self._lock:
                    job = self._jobs.get(meta.get("job_id", ""))
                if job is not None:
                    job._deliver(meta, arrays)
        except (OSError, ValueError, wire.ProtocolError) as error:
            self._fail(f"prefill helper connection lost: {error}")

    def _fail(self, reason: str) -> None:
        self.alive = False
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            job._deliver({"op": "error", "error": reason}, {})

    def _job_closed(self, job: HelperJob) -> None:
        with self._lock:
            self._jobs.pop(job.job_id, None)
            if not self._jobs:
                self._open.clear()

    def _keep_warm(self) -> None:
        stream = mx.new_stream(mx.gpu)
        while self.alive:
            if not self._open.wait(_KEEP_WARM_SECONDS):
                continue
            with mx.stream(stream):
                mx.eval(mx.zeros((1,), dtype=mx.float32) + 1)
            time.sleep(_KEEP_WARM_SECONDS)

    def job_count(self) -> int:
        with self._lock:
            return len(self._jobs)
