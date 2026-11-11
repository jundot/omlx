# SPDX-License-Identifier: Apache-2.0
"""Runs the GPU stages of at most one prefill session.

MLX uses one CUDA device per process, so each GPU gets its own stage worker;
the last stage is the endpoint the oMLX host connects to.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import READY_PREFIX, families, gpus

logger = logging.getLogger("omlx_cuda_helper.supervisor")

# Activations of one chunk and allocator headroom on each GPU.
RESERVE_BYTES = 6 << 30
_READY_TIMEOUT_S = 900.0


def plan_stages(
    shared: int,
    per_layer: list[int],
    layers: int,
    free: list[int],
    first_stage_min: int = 1,
) -> list[tuple[int, int, int, int]]:
    """Stages as (gpu position, first layer, last layer, bytes).

    Two GPUs split the layers as evenly as memory allows, since the stages
    overlap in time. The first stage also holds the tensors outside the layers.
    """
    sizes = per_layer[:layers]
    if len(sizes) < layers:
        raise RuntimeError(f"the checkpoint has only {len(sizes)} layers")

    def cost(first: int, last: int) -> int:
        return sum(sizes[first:last]) + (shared if first == 0 else 0)

    if len(free) >= 2 and layers > first_stage_min:
        cuts = sorted(range(first_stage_min, layers), key=lambda a: abs(a - layers / 2))
        a, b = sorted(sorted(range(len(free)), key=lambda i: -free[i])[:2])
        for cut in cuts:
            head, tail = cost(0, cut), cost(cut, layers)
            # Either GPU may take the first stage, whichever has the room.
            for first, second in ((a, b), (b, a)):
                if (
                    head + RESERVE_BYTES <= free[first]
                    and tail + RESERVE_BYTES <= free[second]
                ):
                    return [(first, 0, cut, head), (second, cut, layers, tail)]
    whole = cost(0, layers)
    for position in sorted(range(len(free)), key=lambda i: -free[i]):
        if whole + RESERVE_BYTES <= free[position]:
            return [(position, 0, layers, whole)]
    raise RuntimeError(f"{layers} layers do not fit in the free GPU memory")


@dataclass
class _Session:
    owner: object
    host: dict[str, str]
    model_dir: str
    model_type: str
    layers: int
    stages: list[tuple[int, int, int, int]]
    procs: list[subprocess.Popen] = field(default_factory=list)
    state: str = "loading"


class Supervisor:
    def __init__(
        self,
        gpu_indices: list[int] | None,
        run_dir: Path,
        *,
        data_port: int = 0,
        worker: list[str] | None = None,
    ):
        self.gpu_indices, self.run_dir = gpu_indices, run_dir
        self.data_port = data_port
        # The command that runs one stage; arguments are appended.
        self.worker = worker or [sys.executable, "-m", "omlx_cuda_helper.worker"]
        self._lock = threading.Lock()
        # Sessions load one after another: a start that takes over waits for
        # a loading session, so the host of that one hears why it ends.
        self._starting = threading.Lock()
        self._session: _Session | None = None

    def model_dir(self) -> str:
        """Real path of the running session's model folder, or ""."""
        session = self._session
        return os.path.realpath(session.model_dir) if session else ""

    def status(self) -> dict[str, Any] | None:
        session = self._session
        if session is None:
            return None
        count = len(gpus.gpus(self.gpu_indices)) or 1 + max(
            s[0] for s in session.stages
        )
        held = [0] * count
        for position, _, _, size in session.stages:
            if position < count:
                held[position] = size
        return {
            "host": session.host.get("name"),
            "host_node_id": session.host.get("node_id"),
            "model": Path(session.model_dir).name,
            "model_type": session.model_type,
            "layers": session.layers,
            "stages": [stage[2] for stage in session.stages],
            "gpu_bytes": held,
            "state": session.state,
        }

    def start(
        self,
        owner: object,
        *,
        model_dir: str,
        model_type: str,
        layers: int,
        warm_tokens: int,
        host: dict[str, str],
        token: bytes,
        bind: str,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> dict[str, Any]:
        """Load and warm the stages; returns the endpoint port and greeting.

        The load ends early on ``stop`` or when ``cancelled`` turns true.
        """
        with self._starting:
            with self._lock:
                self._stop(replaced=True)
                planner = families.planner(model_type)
                shared, per_layer = planner.plan(model_dir)
                found = gpus.gpus(self.gpu_indices)
                if not found:
                    raise RuntimeError("no NVIDIA GPU is available")
                free = [gpu["free"] for gpu in found]
                first_min = getattr(planner, "FIRST_STAGE_MIN", 1)
                stages = None
                # The host sizes the layers from the file sizes; what a stage
                # really loads can be larger, so hold as many as fit.
                for count in range(layers, 0, -1):
                    try:
                        stages = plan_stages(shared, per_layer, count, free, first_min)
                    except RuntimeError:
                        continue
                    if count < layers:
                        logger.warning(
                            "%s of the %s requested layers fit in GPU memory",
                            count,
                            layers,
                        )
                    layers = count
                    break
                if stages is None:
                    raise RuntimeError("not even one layer fits in the free GPU memory")
                session = _Session(owner, host, model_dir, model_type, layers, stages)
                self._session = session
                try:
                    readers = self._launch(session, found, warm_tokens, token, bind)
                except BaseException:
                    self._stop()
                    raise
            # The lock is free while the stages load, so the load can be stopped.
            deadline = time.monotonic() + _READY_TIMEOUT_S
            hello: dict[str, Any] | None = None
            for reader in readers:
                hello = reader.wait(deadline, cancelled)
                if hello is None:
                    break
            with self._lock:
                if self._session is not session:
                    raise RuntimeError("the session was stopped while it loaded")
                if hello is None:
                    self._stop()
                    raise RuntimeError("a GPU stage did not start; see the helper log")
                session.state = "ready"
                threading.Thread(
                    target=self._watch,
                    args=(session,),
                    name="omlx-helper-watch",
                    daemon=True,
                ).start()
                logger.info(
                    "session ready for %s: %s layers %s on %s",
                    host.get("name") or host.get("node_id"),
                    Path(model_dir).name,
                    [(first, last) for _, first, last, _ in stages],
                    [found[position]["name"] for position, *_ in stages],
                )
                return {"port": int(hello["port"]), "hello": hello}

    def _launch(
        self,
        session: _Session,
        found: list[dict[str, Any]],
        warm_tokens: int,
        token: bytes,
        bind: str,
    ) -> list[_ReadyLine]:
        """Start one worker per stage; each reader waits for its ready line."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        previous_socket, previous_token = "", b""
        readers = []
        for number, (position, first, last, _) in enumerate(session.stages):
            endpoint = number == len(session.stages) - 1
            own_token = token if endpoint else secrets.token_hex(32).encode()
            socket_path = str(self.run_dir / f"stage{number}.sock")
            listen = (
                f"tcp:{bind}:{self.data_port}" if endpoint else f"unix:{socket_path}"
            )
            command = [
                *self.worker,
                "--model",
                session.model_dir,
                "--model-type",
                session.model_type,
                "--first",
                str(first),
                "--last",
                str(last),
                "--listen",
                listen,
                "--warm-tokens",
                str(warm_tokens),
            ]
            if previous_socket:
                command += ["--upstream", f"unix:{previous_socket}"]
            env = {
                **os.environ,
                # nvidia-smi numbers the GPUs by PCI bus. The default CUDA
                # order (fastest first) differs on a machine with mixed GPUs.
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": str(found[position]["index"]),
                # MLX names a compiled kernel by its source only, so a cache
                # shared across MLX builds hands one the kernels of another.
                "MLX_PTX_CACHE_DIR": str(self._kernel_cache()),
                # FP32 products pick sparse-attention blocks; TF32 would
                # change which blocks win.
                "MLX_ENABLE_TF32": "0",
            }
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                env=env,
                text=True,
            )
            session.procs.append(proc)
            proc.stdin.write(own_token.decode() + "\n")
            if previous_socket:
                proc.stdin.write(previous_token.decode() + "\n")
            proc.stdin.flush()
            readers.append(_ReadyLine(proc))
            previous_socket, previous_token = socket_path, own_token
        return readers

    def _kernel_cache(self) -> Path:
        try:
            build = importlib.metadata.version("mlx")
        except importlib.metadata.PackageNotFoundError:
            build = "unknown"
        return self.run_dir.parent / "cache" / f"mlx-ptx-{build}"

    def _watch(self, session: _Session) -> None:
        while self._session is session:
            if any(proc.poll() is not None for proc in session.procs):
                with self._lock:
                    if self._session is session:
                        logger.warning("a stage worker exited; the session ends")
                        self._stop()
                return
            time.sleep(1.0)

    def stop(self, owner: object | None = None) -> None:
        """End the session, if *owner* started it (or always with None)."""
        with self._lock:
            session = self._session
            if session is not None and (owner is None or session.owner is owner):
                self._stop()

    def _stop(self, replaced: bool = False) -> None:
        session, self._session = self._session, None
        if session is None:
            return
        procs = [proc for proc in session.procs if proc.poll() is None]
        if replaced and procs and session.state == "ready":
            # The endpoint tells its host why it goes, so the host stays off.
            procs[-1].send_signal(signal.SIGUSR1)
            procs = procs[:-1]
        for proc in procs:
            proc.terminate()
        deadline = time.monotonic() + 10
        for proc in session.procs:
            try:
                proc.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        logger.info("session ended: GPU memory is free")


class _ReadyLine:
    """Reads a worker's stdout until its ready line."""

    def __init__(self, proc: subprocess.Popen):
        self.proc = proc
        self._hello: dict[str, Any] | None = None
        self._done = threading.Event()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self.proc.stdout:
            if line.startswith(READY_PREFIX) and self._hello is None:
                self._hello = json.loads(line[len(READY_PREFIX) :])
                self._done.set()
        self._done.set()

    def wait(
        self, deadline: float, cancelled: Callable[[], bool]
    ) -> dict[str, Any] | None:
        """The ready line, or None: the worker ended, the deadline passed or
        ``cancelled`` turned true."""
        while not self._done.wait(0.2):
            if cancelled() or time.monotonic() > deadline:
                break
        return self._hello
