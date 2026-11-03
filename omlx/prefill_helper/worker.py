# SPDX-License-Identifier: Apache-2.0
"""The helper process: runs layers ``[0, k)`` of prompt chunks for one host.

Started over SSH by the host (``python -m omlx.prefill_helper.worker``). It
loads its layers, warms the kernels at the chunk size, prints a ready line and
then serves one authenticated connection at a time. Jobs run in arrival order.
Each job streams one frame per chunk and stays at most ``WINDOW`` frames ahead
of the host's acknowledgements, so a slow host never piles chunks in memory.
A job that takes host inputs (``feed``) also waits for each chunk's inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import socket
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from .. import __version__
from . import wire
from .adapter import Capture
from .adapters import adapter_for
from .lending import active_lending, active_marker_path

WINDOW = wire.WINDOW
_KEEP_WARM_SECONDS = 0.5
READY_PREFIX = "OMLX_PREFILL_HELPER_READY "


@dataclass
class _Job:
    job_id: str
    tokens: np.ndarray
    chunks: list[int]
    layers: int
    prefix: dict[str, mx.array] | None = None
    feed: bool = False
    # Block drafter layers to return: {"layers": [...], "from": job row}.
    capture: dict | None = None
    inputs: dict[int, dict] = field(default_factory=dict)
    acked: int = -1
    cancelled: bool = False


@dataclass
class _Session:
    """One host connection: jobs queued by the reader, frames for the sender."""

    sock: socket.socket
    jobs: deque = field(default_factory=deque)
    by_id: dict = field(default_factory=dict)
    outbox: queue.Queue = field(default_factory=queue.Queue)
    changed: threading.Condition = field(default_factory=threading.Condition)
    closed: bool = False


class HelperServer:
    """Serves prompt chunks for layers ``[0, k)`` of one loaded model."""

    def __init__(
        self,
        lm: Any,
        adapter: Any,
        k: int,
        token: bytes,
        info: dict,
        memory_limit: int = 0,
    ):
        self.lm, self.adapter, self.k, self.token = lm, adapter, k, token
        # A job whose caches outgrow this hands the prompt back to the host.
        self.memory_limit = memory_limit
        self.info = info
        self.state = "idle"
        self._session: _Session | None = None

    def notify_stopped(self, replaced: bool = False, timeout: float = 2.0) -> None:
        """Tell the host this helper was stopped here or replaced by another
        model's, so it stays off."""
        session = self._session
        if session is None or session.closed:
            return
        sent = threading.Event()
        session.outbox.put(({"op": "stopped", "replaced": replaced}, None))
        session.outbox.put(("flush", sent))
        sent.wait(timeout)

    def serve(self, listener: socket.socket, *, once: bool = False) -> None:
        while True:
            try:
                sock, _ = listener.accept()
            except OSError:
                return
            try:
                wire.tune(sock)
                wire.authenticate(sock, self.token, server=True)
            except (OSError, wire.ProtocolError):
                sock.close()
                continue
            self._serve_session(_Session(sock))
            if once:
                return

    def _serve_session(self, session: _Session) -> None:
        self._session = session
        stop_warm = _keep_gpu_warm()
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
            stop_warm()
            session.outbox.put(None)
            sender.join(timeout=30)
            session.sock.close()
            self.state = "idle"

    def _read(self, session: _Session) -> None:
        try:
            while True:
                meta, arrays = wire.recv(session.sock)
                op = meta.get("op")
                with session.changed:
                    if op == "job":
                        prefix = {
                            name[2:]: wire.from_host(*value)
                            for name, value in arrays.items()
                            if name.startswith("p:")
                        }
                        job = _Job(
                            meta["job_id"],
                            arrays["tokens"][0].reshape(-1),
                            [int(n) for n in meta["chunks"]],
                            min(int(meta.get("layers", self.k)), self.k),
                            prefix or None,
                            bool(meta.get("feed")),
                            meta.get("capture"),
                        )
                        session.jobs.append(job)
                        session.by_id[job.job_id] = job
                    elif op == "inputs":
                        job = session.by_id.get(meta["job_id"])
                        if job is not None:
                            job.inputs[int(meta["index"])] = arrays
                    elif op == "ack":
                        job = session.by_id.get(meta["job_id"])
                        if job is not None:
                            job.acked = max(job.acked, int(meta["index"]))
                    elif op == "cancel":
                        job = session.by_id.get(meta["job_id"])
                        if job is not None:
                            job.cancelled = True
                    elif op == "bye":
                        break
                    session.changed.notify_all()
        except (OSError, ValueError, wire.ProtocolError):
            pass
        finally:
            with session.changed:
                session.closed = True
                session.changed.notify_all()

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
        with session.changed:
            while not session.jobs and not session.closed:
                session.changed.wait()
            if not session.jobs:
                return None
            return session.jobs.popleft()

    def _run_job(self, session: _Session, job: _Job) -> None:
        self.state = "working"
        cache = None
        try:
            with self.adapter.use_first(self.lm, job.layers):
                cache = self.lm.make_cache()
                self._run_chunks(session, job, cache)
        except Exception as error:  # reported to the host, which prefills locally
            session.outbox.put(
                ({"op": "error", "job_id": job.job_id, "error": repr(error)}, None)
            )
        finally:
            session.by_id.pop(job.job_id, None)
            del cache
            mx.clear_cache()
            self.state = "idle"

    def _run_chunks(self, session: _Session, job: _Job, cache: list) -> None:
        if job.prefix:
            self.adapter.load_prefix(cache, job.prefix)
        start = 0
        for index, size in enumerate(job.chunks):
            with session.changed:
                while (
                    index - job.acked > WINDOW or (job.feed and index not in job.inputs)
                ) and not (job.cancelled or session.closed):
                    session.changed.wait()
                if job.cancelled or session.closed:
                    return
                # None is the warm-up call; a real chunk always has a dict.
                inputs = job.inputs.pop(index, {})
            tokens = mx.array(job.tokens[start : start + size])[None]
            following = job.chunks[index + 1] if index + 1 < len(job.chunks) else 0
            capture = None
            if job.capture and start + size > job.capture["from"]:
                capture = Capture(
                    job.capture["layers"], max(0, job.capture["from"] - start)
                )
            began = time.perf_counter()
            boundary, state = self.adapter.run_first(
                self.lm, cache, tokens, following, inputs, capture
            )
            if capture is not None:
                boundary.update({f"cap.{i}": v for i, v in capture.out.items()})
            elapsed = time.perf_counter() - began
            if self.memory_limit and mx.get_active_memory() > self.memory_limit:
                raise RuntimeError(
                    "the helper Mac has no GPU memory left for this prompt"
                )
            arrays = {f"b:{n}": wire.to_host(v) for n, v in boundary.items()}
            arrays.update({f"s:{n}": wire.to_host(v) for n, v in state.items()})
            session.outbox.put(
                (
                    {
                        "op": "chunk",
                        "job_id": job.job_id,
                        "index": index,
                        "start": start,
                        "size": size,
                        "seconds": elapsed,
                    },
                    arrays,
                )
            )
            start += size
        session.outbox.put(({"op": "done", "job_id": job.job_id}, None))


def _keep_gpu_warm():
    """Tick a tiny kernel on a separate stream so the GPU never parks between
    chunks (an idle GPU stalls the next command buffer for seconds)."""
    stop = threading.Event()

    def tick() -> None:
        stream = mx.new_stream(mx.gpu)
        while not stop.wait(_KEEP_WARM_SECONDS):
            with mx.stream(stream):
                mx.eval(mx.zeros((1,), dtype=mx.float32) + 1)

    threading.Thread(target=tick, name="omlx-prefill-helper-warm", daemon=True).start()
    return stop.set


def _write_marker(info: dict) -> Path:
    path = active_marker_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({**info, "pid": os.getpid(), "since": time.time()}))
    tmp.replace(path)
    return path


def _clear_marker(path: Path) -> None:
    try:
        data = json.loads(path.read_text())
        if data.get("pid") == os.getpid():
            path.unlink()
    except (OSError, ValueError):
        pass


def _replace_running_helper(timeout: float = 30.0) -> None:
    """One helper per Mac: stop the one running here and wait until its
    memory is free."""
    previous = active_lending()
    if previous is None:
        return
    pid = int(previous["pid"])
    os.kill(pid, signal.SIGUSR1)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="omlx.prefill_helper.worker")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--bind", required=True)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--warm-tokens", type=int, default=8192)
    parser.add_argument("--host-name", default="")
    parser.add_argument("--a8-min-tokens", type=int)
    args = parser.parse_args(argv)
    # The session token arrives on stdin so it never shows up in ps.
    token = sys.stdin.readline().strip().encode()
    if not token:
        print("prefill helper: no session token on stdin", file=sys.stderr)
        return 2

    adapter = adapter_for(args.model_type)
    if adapter is None:
        print(
            f"prefill helper: unsupported model type {args.model_type}", file=sys.stderr
        )
        return 2
    _replace_running_helper()
    info = mx.device_info()
    mx.set_cache_limit(int(info["memory_size"]))
    mx.set_wired_limit(int(info["max_recommended_working_set_size"]))
    lm = adapter.load_first(args.model, args.layers, args.a8_min_tokens)

    began = time.perf_counter()
    warm = np.arange(args.warm_tokens, dtype=np.int32) % 1000 + 3
    adapter.run_first(lm, lm.make_cache(), mx.array(warm)[None], 0)
    warm_seconds = time.perf_counter() - began
    mx.clear_cache()

    # getaddrinfo keeps the interface scope of a link-local address.
    family, _, _, _, address = socket.getaddrinfo(
        args.bind, args.port, type=socket.SOCK_STREAM
    )[0]
    listener = socket.socket(family)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(address)
    listener.listen(1)
    port = listener.getsockname()[1]
    hello = {
        "protocol": wire.PROTOCOL,
        "version": __version__,
        "model_type": args.model_type,
        "layers": args.layers,
        "warm_tokens": args.warm_tokens,
        "warm_seconds": warm_seconds,
        "active_gib": mx.get_active_memory() / 2**30,
        "device": info.get("device_name", ""),
    }
    marker = _write_marker(
        {
            "host": args.host_name,
            "model": args.model,
            "layers": args.layers,
            "port": port,
            "active_gib": hello["active_gib"],
        }
    )

    server = HelperServer(
        lm,
        adapter,
        args.layers,
        token,
        hello,
        memory_limit=int(info["max_recommended_working_set_size"] * 0.95),
    )

    def stop(*_: Any) -> None:
        _clear_marker(marker)
        os._exit(0)

    def stopped_here(*_: Any) -> None:
        # SIGTERM comes from this Mac's oMLX (the user pressed Stop).
        server.notify_stopped()
        stop()

    def replaced(*_: Any) -> None:
        # SIGUSR1 comes from a helper starting here for another model.
        server.notify_stopped(replaced=True)
        stop()

    signal.signal(signal.SIGTERM, stopped_here)
    signal.signal(signal.SIGUSR1, replaced)
    signal.signal(signal.SIGHUP, stop)

    def watch_stdin() -> None:
        # The host's SSH session holds stdin open; when it ends, so does this.
        for _ in sys.stdin:
            pass
        stop()

    threading.Thread(target=watch_stdin, daemon=True).start()
    print(READY_PREFIX + json.dumps({**hello, "port": port}), flush=True)
    try:
        server.serve(listener)
    finally:
        _clear_marker(marker)
    return 0


if __name__ == "__main__":
    sys.exit(main())
