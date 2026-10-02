# SPDX-License-Identifier: Apache-2.0
"""Lifecycle supervision for isolated oMLX cluster workers."""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import WORKER_PROTOCOL_VERSION


@dataclass(frozen=True)
class JacclLaunchConfig:
    """Environment contract used by MLX's JACCL launcher."""

    rank: int
    coordinator: str
    ibv_devices_file: Path
    ring: bool = False

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise ValueError("rank must be non-negative")
        if not self.coordinator or ":" not in self.coordinator:
            raise ValueError("coordinator must be in host:port form")
        if not str(self.ibv_devices_file):
            raise ValueError("ibv_devices_file is required")

    def environment(self) -> dict[str, str]:
        result = {
            "MLX_RANK": str(self.rank),
            "MLX_JACCL_COORDINATOR": self.coordinator,
            "MLX_IBV_DEVICES": str(self.ibv_devices_file),
        }
        if self.ring:
            result["MLX_JACCL_RING"] = "1"
        return result


class WorkerProtocolError(RuntimeError):
    """Raised when a worker violates the control protocol."""


class WorkerSupervisor:
    """Start one local worker and enforce ready/ping/shutdown deadlines."""

    def __init__(
        self,
        *,
        rank: int = 0,
        plan_hash: str = "smoke",
        timeout: float = 5.0,
        jaccl: JacclLaunchConfig | None = None,
    ) -> None:
        if rank < 0:
            raise ValueError("rank must be non-negative")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if jaccl is not None and jaccl.rank != rank:
            raise ValueError("worker rank and JACCL rank must match")
        self.rank = rank
        self.plan_hash = plan_hash
        self.timeout = timeout
        self.jaccl = jaccl
        self.process: subprocess.Popen[str] | None = None
        self.ready_event: dict[str, Any] | None = None

    def start(self) -> dict[str, Any]:
        if self.process is not None:
            raise RuntimeError("worker is already started")

        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        if self.jaccl is not None:
            environment.update(self.jaccl.environment())

        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "omlx.cluster.worker",
                "--stdio",
                "--rank",
                str(self.rank),
                "--plan-hash",
                self.plan_hash,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=environment,
        )
        try:
            event = self._read_event()
            self._expect(event, "ready")
            if event.get("protocol_version") != WORKER_PROTOCOL_VERSION:
                raise WorkerProtocolError("worker protocol version mismatch")
            if event.get("rank") != self.rank:
                raise WorkerProtocolError("worker rank mismatch")
            if event.get("plan_hash") != self.plan_hash:
                raise WorkerProtocolError("worker plan hash mismatch")
            self.ready_event = event
            return event
        except Exception:
            self._terminate()
            raise

    def ping(self, nonce: Any = None) -> dict[str, Any]:
        self._send({"command": "ping", "nonce": nonce})
        event = self._read_event()
        self._expect(event, "pong")
        if event.get("nonce") != nonce:
            raise WorkerProtocolError("worker pong nonce mismatch")
        return event

    def stop(self) -> dict[str, Any] | None:
        process = self.process
        if process is None:
            return None

        event: dict[str, Any] | None = None
        try:
            if process.poll() is None:
                self._send({"command": "shutdown"})
                event = self._read_event()
                self._expect(event, "stopped")
                process.wait(timeout=self.timeout)
        except Exception:
            self._terminate()
            raise
        finally:
            self._close_pipes()
            self.process = None
        return event

    def _send(self, payload: dict[str, Any]) -> None:
        process = self.process
        if process is None or process.poll() is not None or process.stdin is None:
            raise WorkerProtocolError("worker is not running")
        process.stdin.write(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        )
        process.stdin.flush()

    def _read_event(self) -> dict[str, Any]:
        process = self.process
        if process is None or process.stdout is None:
            raise WorkerProtocolError("worker is not started")

        selector = selectors.DefaultSelector()
        try:
            selector.register(process.stdout, selectors.EVENT_READ)
            events = selector.select(self.timeout)
        finally:
            selector.close()

        if not events:
            if process.poll() is not None:
                raise WorkerProtocolError(self._exit_detail(process))
            raise TimeoutError(f"worker did not respond within {self.timeout:.2f}s")

        line = process.stdout.readline()
        if not line:
            raise WorkerProtocolError(self._exit_detail(process))
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise WorkerProtocolError("worker emitted invalid JSON") from exc
        if not isinstance(event, dict):
            raise WorkerProtocolError("worker event must be an object")
        return event

    @staticmethod
    def _expect(event: dict[str, Any], event_type: str) -> None:
        if event.get("type") != event_type:
            raise WorkerProtocolError(
                f"expected worker event {event_type!r}, got {event.get('type')!r}"
            )

    @staticmethod
    def _exit_detail(process: subprocess.Popen[str]) -> str:
        stderr = ""
        if process.stderr is not None and process.poll() is not None:
            stderr = process.stderr.read().strip()
        suffix = f": {stderr}" if stderr else ""
        return f"worker exited with code {process.poll()}{suffix}"

    def _terminate(self) -> None:
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=min(self.timeout, 2.0))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        self._close_pipes()
        self.process = None

    def _close_pipes(self) -> None:
        process = self.process
        if process is None:
            return
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()

    def __enter__(self) -> WorkerSupervisor:
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.process is not None:
            if exc is None:
                self.stop()
            else:
                self._terminate()


def run_worker_smoke(*, timeout: float = 5.0) -> dict[str, Any]:
    """Run a real child-process ready/ping/shutdown round trip."""

    started_at = time.monotonic()
    supervisor = WorkerSupervisor(timeout=timeout)
    try:
        ready = supervisor.start()
        pong = supervisor.ping(nonce="omlx-cluster-smoke")
        stopped = supervisor.stop()
    finally:
        if supervisor.process is not None:
            supervisor._terminate()

    return {
        "ok": True,
        "protocol_version": WORKER_PROTOCOL_VERSION,
        "elapsed_seconds": time.monotonic() - started_at,
        "worker_pid": ready["pid"],
        "ready": ready,
        "pong": pong,
        "stopped": stopped,
    }
