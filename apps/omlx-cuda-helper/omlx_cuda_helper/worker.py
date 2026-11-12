# SPDX-License-Identifier: Apache-2.0
"""One GPU stage of a prefill session: layers ``[first, last)`` on one GPU.

A later stage is the host of the stage before it, so to the oMLX host the last
stage is one helper that holds ``[0, last)``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import signal
import socket
import sys
import threading
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import DATA_PROTOCOL, READY_PREFIX, __version__, families, tensors, wire

logger = logging.getLogger("omlx_cuda_helper.worker")

# Chunks a stage may run ahead of the acknowledgements of the next one.
WINDOW = 2
_UPSTREAM_TIMEOUT_S = 120.0
# Shorter than the 10 s a host waits to connect: a silent peer that came
# first must not use up that wait.
_AUTH_TIMEOUT_S = 3.0

Arrays = dict[str, wire.HostArray]


class StageError(RuntimeError):
    pass


class _OutOfMemoryError(StageError):
    """The next chunk would take the stage past its memory limit."""


def _layer_of(name: str) -> int:
    """State and prefix arrays are named ``"{layer}.{slot}"``."""
    return int(name.split(".", 1)[0])


class Upstream:
    """This stage as the host of the stage before it."""

    def __init__(self, sock: socket.socket, token: bytes):
        wire.authenticate(sock, token, server=False)
        hello, _ = wire.recv(sock)
        if hello.get("op") != "hello" or hello.get("protocol") != DATA_PROTOCOL:
            raise wire.ProtocolError("unexpected greeting from the stage before")
        self.sock, self.info = sock, hello
        self._frames: dict[str, queue.Queue] = {}
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        threading.Thread(
            target=self._receive, name="stage-upstream", daemon=True
        ).start()

    def _receive(self) -> None:
        try:
            while True:
                meta, arrays = wire.recv(self.sock)
                with self._lock:
                    frames = self._frames.get(meta.get("job_id", ""))
                if frames is not None:
                    frames.put((meta, arrays))
        except (OSError, ValueError, wire.ProtocolError):
            pass
        with self._lock:
            for frames in self._frames.values():
                frames.put(({"op": "error", "error": "the stage before stopped"}, {}))

    def send(self, meta: dict[str, Any], arrays: Arrays | None = None) -> None:
        try:
            with self._send_lock:
                wire.send(self.sock, meta, arrays)
        except OSError as error:
            raise StageError("the stage before stopped") from error

    def submit(self, meta: dict[str, Any], arrays: Arrays) -> None:
        with self._lock:
            self._frames[meta["job_id"]] = queue.Queue()
        self.send(meta, arrays)

    def take(self, job_id: str, index: int) -> tuple[dict[str, Any], Arrays] | None:
        """The chunk *index* of a job, or None once ``cancel`` ended the job.

        Taking a chunk lets the stage before go on.
        """
        try:
            meta, arrays = self._frames[job_id].get(timeout=_UPSTREAM_TIMEOUT_S)
        except queue.Empty as error:
            raise StageError("the stage before did not answer in time") from error
        if meta.get("op") == "cancel":
            return None
        if meta.get("op") != "chunk" or meta.get("index") != index:
            raise StageError(str(meta.get("error") or "the stage before stopped"))
        self.send({"op": "ack", "job_id": job_id, "index": index})
        return meta, arrays

    def cancel(self, job_id: str) -> None:
        """End a job on the stage before and wake a ``take`` that waits for it."""
        with self._lock:
            frames = self._frames.get(job_id)
        if frames is not None:
            frames.put(({"op": "cancel"}, {}))
        self.send({"op": "cancel", "job_id": job_id})

    def forget(self, job_id: str) -> None:
        with self._lock:
            self._frames.pop(job_id, None)


@dataclass
class _Job:
    job_id: str
    tokens: np.ndarray
    chunks: list[int]
    # Layers of the whole helper this job uses, counted from layer 0.
    layers: int
    prefix: Arrays
    feed: bool = False
    capture: dict | None = None
    # The host follows the layers each chunk reports, so the job may drop some.
    fewer: bool = False
    inputs: dict[int, Arrays] = field(default_factory=dict)
    acked: int = -1
    cancelled: bool = False


@dataclass
class _Use:
    """Memory of a chunk: bytes it keeps per token and own layer, and the
    bytes it holds only while it runs."""

    kept: float
    passing: int


@dataclass
class _Session:
    sock: socket.socket
    jobs: deque = field(default_factory=deque)
    by_id: dict = field(default_factory=dict)
    outbox: queue.Queue = field(default_factory=queue.Queue)
    changed: threading.Condition = field(default_factory=threading.Condition)
    closed: bool = False


class StageServer:
    """Serves prompt chunks for the layers of one stage and those before it."""

    def __init__(
        self,
        stage: Any,
        token: bytes,
        info: dict[str, Any],
        upstream: Upstream | None = None,
        memory_limit: int = 0,
        working: int = 0,
    ):
        """``working`` is what the warm chunk held while it ran."""
        self.stage, self.token, self.info, self.upstream = stage, token, info, upstream
        # A chunk that would take the stage past this drops layers first; a
        # job with no layer left to drop hands the prompt back to the host.
        self.memory_limit = memory_limit
        # Layers held at start. Dropped ones load again while no job waits.
        self.top = stage.last
        # Bytes each dropped layer freed, the last dropped at the end.
        self._dropped: list[int] = []
        # What the first chunk of a job is expected to take.
        self._opening = _Use(0.0, working)
        self._session: _Session | None = None

    def notify_stopped(self) -> None:
        """Tell the host this helper now serves another model, so it stays off."""
        session = self._session
        if session is None or session.closed:
            return
        sent = threading.Event()
        session.outbox.put(({"op": "stopped", "replaced": True}, None))
        session.outbox.put(("flush", sent))
        sent.wait(2.0)

    def serve(self, listener: socket.socket) -> None:
        while True:
            try:
                sock, _ = listener.accept()
            except OSError:
                return
            try:
                if sock.family != socket.AF_UNIX:
                    wire.tune(sock)
                # Sessions are served one at a time: a peer that sends
                # nothing must not hold the accept loop.
                sock.settimeout(_AUTH_TIMEOUT_S)
                wire.authenticate(sock, self.token, server=True)
                sock.settimeout(None)
            except (OSError, wire.ProtocolError):
                sock.close()
                continue
            self._serve_session(_Session(sock))

    def _serve_session(self, session: _Session) -> None:
        self._session = session
        reader = threading.Thread(target=self._read, args=(session,), daemon=True)
        sender = threading.Thread(target=self._send, args=(session,), daemon=True)
        reader.start()
        sender.start()
        session.outbox.put(({"op": "hello", **self.info}, None))
        try:
            while True:
                job = self._next_job(session)
                if job is None:
                    break
                self._run_job(session, job)
        finally:
            session.outbox.put(None)
            sender.join(timeout=30)
            session.sock.close()

    def _read(self, session: _Session) -> None:
        try:
            while True:
                meta, arrays = wire.recv(session.sock)
                op = meta.get("op")
                with session.changed:
                    if op == "job":
                        job = self._accept_job(meta, arrays)
                        session.jobs.append(job)
                        session.by_id[job.job_id] = job
                    elif op == "bye":
                        break
                    else:
                        job = session.by_id.get(meta.get("job_id", ""))
                        if job is not None:
                            self._update_job(job, op, meta, arrays)
                    session.changed.notify_all()
        except (OSError, ValueError, KeyError, StageError, wire.ProtocolError):
            pass
        finally:
            with session.changed:
                session.closed = True
                session.changed.notify_all()

    def _accept_job(self, meta: dict[str, Any], arrays: Arrays) -> _Job:
        prefix = {n[2:]: v for n, v in arrays.items() if n.startswith("p:")}
        job = _Job(
            meta["job_id"],
            arrays["tokens"][0].reshape(-1),
            [int(n) for n in meta["chunks"]],
            min(int(meta.get("layers", self.top)), self.top),
            prefix,
            bool(meta.get("feed")),
            meta.get("capture"),
            bool(meta.get("fewer_layers")),
        )
        if self.upstream is not None:
            first = self.stage.first
            forward = {"tokens": arrays["tokens"]}
            forward.update(
                {f"p:{n}": v for n, v in prefix.items() if _layer_of(n) < first}
            )
            up = {
                "op": "job",
                "job_id": job.job_id,
                "chunks": job.chunks,
                "layers": min(job.layers, first),
            }
            if job.feed:
                up["feed"] = True
            if job.fewer:
                up["fewer_layers"] = True
            if job.capture:
                below = [i for i in job.capture["layers"] if i < first]
                if below:
                    up["capture"] = {"layers": below, "from": job.capture["from"]}
            self.upstream.submit(up, forward)
        return job

    def _update_job(
        self, job: _Job, op: str, meta: dict[str, Any], arrays: Arrays
    ) -> None:
        if op == "inputs":
            job.inputs[int(meta["index"])] = arrays
            if self.upstream is not None:
                # Host-fed rows belong to whichever stage runs their layer.
                self.upstream.send(meta, arrays)
        elif op == "ack":
            job.acked = max(job.acked, int(meta["index"]))
        elif op == "cancel":
            job.cancelled = True
            if self.upstream is not None:
                self.upstream.cancel(job.job_id)

    def _send(self, session: _Session) -> None:
        while True:
            item = session.outbox.get()
            if item is None:
                return
            if item[0] == "flush":
                item[1].set()
                continue
            try:
                wire.send(session.sock, *item)
            except OSError:
                with session.changed:
                    session.closed = True
                    session.changed.notify_all()
                return

    def _next_job(self, session: _Session) -> _Job | None:
        while True:
            with session.changed:
                if session.jobs or session.closed:
                    return session.jobs.popleft() if session.jobs else None
                if not self._room_to_restore():
                    session.changed.wait()
                    continue
            # One layer at a time, so a new job waits for at most one.
            self._restore()

    def _room_to_restore(self) -> bool:
        return bool(self._dropped) and (
            self.stage.memory_bytes() + self._dropped[-1] + self._opening.passing
            <= self.memory_limit
        )

    def _restore(self) -> None:
        self.stage.add_layer()
        self._dropped.pop()
        logger.info("layer %d is back in GPU memory", self.stage.last - 1)

    def _drop(self) -> None:
        held = self.stage.memory_bytes()
        self.stage.drop_layer()
        self._dropped.append(max(0, held - self.stage.memory_bytes()))
        logger.info("layer %d left GPU memory for a long prompt", self.stage.last)

    def _fit(self, job: _Job, state: Any, own: int, size: int, use: _Use) -> int:
        """The own layers the next chunk can run within the memory limit.

        Layers the job does not use go first, then the job's last ones; the
        host runs those from this chunk on. The first stage keeps one layer.
        """
        stage = self.stage
        while True:
            need = stage.memory_bytes()
            if own:
                need += int(own * size * use.kept) + use.passing
            if need <= self.memory_limit:
                return own
            if stage.last > stage.first + own:
                self._drop()
            elif job.fewer and own > (0 if stage.first else 1):
                own -= 1
                stage.trim(state, own)
                self._drop()
            else:
                raise _OutOfMemoryError(
                    "the helper has no GPU memory left for this prompt"
                )

    def _run_job(self, session: _Session, job: _Job) -> None:
        try:
            self._run_chunks(session, job)
        except Exception as error:  # reported to the host, which prefills locally
            meta = {"op": "error", "job_id": job.job_id, "error": repr(error)}
            if isinstance(error, _OutOfMemoryError):
                logger.warning("job %s: %s", job.job_id, error)
                meta["out_of_memory"] = True
            else:
                logger.exception("job %s failed", job.job_id)
            session.outbox.put((meta, None))
        finally:
            session.by_id.pop(job.job_id, None)
            if self.upstream is not None:
                # A job that ended early leaves the stage before waiting for
                # its acknowledgements. A finished job ignores the cancel.
                with suppress(StageError):
                    self.upstream.cancel(job.job_id)
                self.upstream.forget(job.job_id)

    def _run_chunks(self, session: _Session, job: _Job) -> None:
        stage, upstream = self.stage, self.upstream
        if not job.fewer:
            # This host runs the layers it asked for on every chunk.
            while stage.last < job.layers:
                self._restore()
        own = max(0, min(job.layers, stage.last) - stage.first)
        state = stage.new_job(own) if own else None
        # A later stage may read state of the layers before it: its optional
        # load_upstream gets the job's "p:" arrays, then each chunk's "s:" arrays.
        earlier = getattr(stage, "load_upstream", None) if own and stage.first else None
        if own:
            end = stage.first + own
            mine = {
                n: v for n, v in job.prefix.items() if stage.first <= _layer_of(n) < end
            }
            if mine:
                stage.load_prefix(state, mine)
        if earlier is not None:
            below = {
                f"p:{n}": v for n, v in job.prefix.items() if _layer_of(n) < stage.first
            }
            if below:
                earlier(state, below)
        capture = None
        if own and job.capture:
            wanted = [i for i in job.capture["layers"] if i >= stage.first]
            capture = (
                {"layers": wanted, "from": job.capture["from"]} if wanted else None
            )
        start, use, rise, ahead = 0, self._opening, 0, 0
        for index, size in enumerate(job.chunks):
            with session.changed:
                while (
                    index - job.acked > WINDOW or (job.feed and index not in job.inputs)
                ) and not (job.cancelled or session.closed):
                    session.changed.wait()
                if job.cancelled or session.closed:
                    return
                inputs = job.inputs.pop(index, {})
            arrays: Arrays = {}
            boundary: Arrays | None = None
            before = 0.0
            if upstream is not None:
                taken = upstream.take(job.job_id, index)
                if taken is None:
                    return
                meta, received = taken
                before = float(meta["seconds"])
                ahead = int(meta.get("layers", stage.first))
                if own and ahead < stage.first:
                    # The stage before dropped layers; these cannot follow them.
                    own = 0
                    stage.trim(state, 0)
                boundary = {}
                for name, value in received.items():
                    if name.startswith("b:") and not name.startswith("b:cap."):
                        boundary[name[2:]] = value
                    else:
                        # State and captured layers of the stages before.
                        arrays[name] = value
                if earlier is not None and own:
                    earlier(state, {n: v for n, v in arrays.items() if n[:2] == "s:"})
            if self.memory_limit:
                # A chunk's passing bytes can grow with the prompt: count the
                # last rise once more.
                expected = _Use(use.kept, use.passing + rise)
                own = self._fit(job, state, own, size, expected)
            began = time.perf_counter()
            if own:
                held = stage.memory_bytes()
                stage.peak_bytes()
                following = job.chunks[index + 1] if index + 1 < len(job.chunks) else 0
                boundary, mine = stage.run(
                    state,
                    start=start,
                    tokens=job.tokens[start : start + size],
                    boundary=boundary,
                    inputs=inputs,
                    following=following,
                    capture=capture,
                )
                arrays.update({f"s:{n}": v for n, v in mine.items()})
                kept = stage.memory_bytes()
                passing = max(0, stage.peak_bytes() - kept)
                rise = max(0, passing - use.passing) if index else 0
                use = _Use(max(0, kept - held) / (own * size), passing)
                if index == 0:
                    self._opening = use
            elapsed = time.perf_counter() - began
            arrays.update({f"b:{n}": v for n, v in (boundary or {}).items()})
            reply = {
                "op": "chunk",
                "job_id": job.job_id,
                "index": index,
                "start": start,
                "size": size,
                # The stages overlap in time, so the slower one paces.
                "seconds": max(before, elapsed),
            }
            if job.fewer:
                reply["layers"] = stage.first + own if own else ahead
            session.outbox.put((reply, arrays))
            start += size
        session.outbox.put(({"op": "done", "job_id": job.job_id}, None))


def _listen(spec: str) -> tuple[socket.socket, int]:
    kind, _, rest = spec.partition(":")
    if kind == "unix":
        if os.path.exists(rest):
            os.unlink(rest)
        listener = socket.socket(socket.AF_UNIX)
        listener.bind(rest)
        os.chmod(rest, 0o600)
        listener.listen(1)
        return listener, 0
    host, _, port = rest.rpartition(":")
    if ":" not in host:
        listener = socket.create_server((host, int(port)))
        return listener, listener.getsockname()[1]
    # A link-local address binds only with its interface index, which
    # getaddrinfo reads from the "%interface" suffix.
    family, _, _, _, address = socket.getaddrinfo(
        host, int(port), socket.AF_INET6, socket.SOCK_STREAM, 0, socket.AI_NUMERICHOST
    )[0]
    listener = socket.socket(family, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(address)
    listener.listen()
    return listener, listener.getsockname()[1]


def _connect_unix(path: str, timeout: float) -> socket.socket:
    deadline = time.monotonic() + timeout
    while True:
        sock = socket.socket(socket.AF_UNIX)
        try:
            sock.connect(path)
            return sock
        except OSError:
            sock.close()
            if time.monotonic() > deadline:
                raise
            time.sleep(0.2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="omlx_cuda_helper.worker")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--last", type=int, required=True)
    parser.add_argument("--listen", required=True, help="tcp:HOST:PORT or unix:PATH")
    parser.add_argument("--upstream", help="unix:PATH of the stage before")
    parser.add_argument("--warm-tokens", type=int, default=8192)
    parser.add_argument(
        "--memory-limit",
        type=int,
        default=0,
        help="bytes a stage may hold before it drops layers or hands a prompt back",
    )
    parser.add_argument("--weights-on-gpu", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format=f"stage[{args.first}:{args.last}] %(levelname)s %(message)s",
    )
    # Tokens arrive on stdin so they never show up in ps.
    token = sys.stdin.readline().strip().encode()
    upstream_token = sys.stdin.readline().strip().encode() if args.upstream else b""
    if not token:
        print("stage worker: no session token on stdin", file=sys.stderr)
        return 2

    tensors.check_backend()
    tensors.limit_memory(args.memory_limit)
    tensors.weights_on_gpu = args.weights_on_gpu
    stage = families.stage_module(args.model_type).load_stage(
        args.model, args.first, args.last
    )
    held = stage.memory_bytes()
    stage.peak_bytes()
    warm_seconds = stage.warm(args.warm_tokens)
    working = max(0, stage.peak_bytes() - held)
    listener, port = _listen(args.listen)
    upstream = None
    warm_before, active_before = 0.0, 0.0
    if args.upstream:
        path = args.upstream.partition(":")[2]
        upstream = Upstream(_connect_unix(path, 900.0), upstream_token)
        warm_before = float(upstream.info.get("warm_seconds") or 0.0)
        active_before = float(upstream.info.get("active_gib") or 0.0)
    hello = {
        "protocol": DATA_PROTOCOL,
        "version": __version__,
        "contract": families.FAMILIES[args.model_type].contract,
        "model_type": args.model_type,
        "layers": args.last,
        "warm_tokens": args.warm_tokens,
        "warm_seconds": max(warm_seconds, warm_before),
        "active_gib": stage.memory_bytes() / 2**30 + active_before,
        "device": stage.device,
    }
    server = StageServer(stage, token, hello, upstream, args.memory_limit, working)

    def stop(*_: Any) -> None:
        os._exit(0)

    def replaced(*_: Any) -> None:
        # The supervisor starts a session for another host or model.
        server.notify_stopped()
        stop()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGUSR1, replaced)

    def watch_stdin() -> None:
        # The supervisor holds stdin open; when it goes, so does this stage.
        for _ in sys.stdin:
            pass
        stop()

    threading.Thread(target=watch_stdin, daemon=True).start()
    print(READY_PREFIX + json.dumps({**hello, "port": port}), flush=True)
    server.serve(listener)
    return 0


if __name__ == "__main__":
    sys.exit(main())
