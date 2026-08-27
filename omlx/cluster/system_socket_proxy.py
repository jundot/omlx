# SPDX-License-Identifier: Apache-2.0
"""Carry one local TCP stream through macOS's system Python when required.

Some macOS installations deny a Homebrew Python process outbound access to a
direct Thunderbolt subnet (``ENETUNREACH``) while Apple's system Python can use
the exact same address.  Distributed ranks still need a reliable, ordered TCP
control channel for cancellation, cache agreement, and request steering.  A
small system-Python child bridges a local loopback socket to that direct path;
the parent process keeps its normal socket API and no model data crosses it.
"""

from __future__ import annotations

import ipaddress
import os
from pathlib import Path
import select
import socket
import subprocess
import sys
import time
from dataclasses import dataclass


_DEFAULT_TIMEOUT_SECONDS = 20.0
_PROXY_PROGRAM = r'''
import socket, sys, threading, time

def connect(host, port, timeout):
    deadline = time.monotonic() + timeout
    error = None
    while time.monotonic() < deadline:
        remote = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            remote.settimeout(min(1.0, max(0.05, deadline - time.monotonic())))
            remote.connect((host, port))
            remote.settimeout(None)
            return remote
        except OSError as exc:
            error = exc
            remote.close()
            time.sleep(0.05)
    raise RuntimeError("could not reach control coordinator: %s" % error)

def copy(source, target):
    try:
        while True:
            payload = source.recv(65536)
            if not payload:
                return
            target.sendall(payload)
    finally:
        try:
            target.shutdown(socket.SHUT_WR)
        except OSError:
            pass

def main():
    host, port, timeout = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        print("PORT %d" % listener.getsockname()[1], flush=True)
        remote = connect(host, port, timeout)
        print("READY", flush=True)
        local, _ = listener.accept()
    finally:
        listener.close()
    thread = threading.Thread(target=copy, args=(remote, local), daemon=True)
    thread.start()
    copy(local, remote)
    thread.join(timeout=1.0)
    local.close()
    remote.close()

if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print("oMLX control socket helper failed: %s" % exc, file=sys.stderr, flush=True)
        raise
'''


def _system_python() -> str | None:
    candidate = Path(
        os.environ.get("OMLX_CLUSTER_CONTROL_PROXY_PYTHON", "/usr/bin/python3")
    )
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return None


def should_proxy_control_socket(host: str) -> bool:
    """Whether a non-loopback control socket should use the macOS carrier."""

    mode = os.environ.get("OMLX_CLUSTER_CONTROL_TRANSPORT", "auto").strip().lower()
    if mode not in {"", "auto", "direct", "system-proxy"}:
        raise RuntimeError(
            "OMLX_CLUSTER_CONTROL_TRANSPORT must be auto, direct, or system-proxy"
        )
    if mode == "direct":
        return False
    if mode == "system-proxy":
        if _system_python() is None:
            raise RuntimeError("system Python control proxy is unavailable")
        return True
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host.strip().lower() in {"localhost", "localhost.local"}
    return sys.platform == "darwin" and not loopback and _system_python() is not None


@dataclass
class SystemSocketProxy:
    """The parent-visible loopback stream and its helper process."""

    stream: socket.socket
    process: subprocess.Popen[bytes]

    def close(self) -> None:
        try:
            self.stream.close()
        except OSError:
            pass
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=1.0)


def _helper_error(process: subprocess.Popen[bytes]) -> str:
    status = process.poll()
    detail = b""
    if status is not None and process.stderr is not None:
        detail = process.stderr.read()
    text = detail.decode("utf-8", "replace").strip()
    suffix = f": {text[-1000:]}" if text else ""
    return f"control socket helper stopped (status={status}){suffix}"


def _read_line(
    process: subprocess.Popen[bytes],
    *,
    deadline: float,
) -> bytes:
    if process.stdout is None:
        raise RuntimeError("control socket helper has no stdout")
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("control socket helper did not become ready")
        ready, _write, _error = select.select([process.stdout], [], [], remaining)
        if ready:
            line = process.stdout.readline()
            if line:
                return line.rstrip(b"\r\n")
            raise RuntimeError(_helper_error(process))
        if process.poll() is not None:
            raise RuntimeError(_helper_error(process))


def open_system_tcp_proxy(
    host: str,
    port: int,
    *,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
) -> SystemSocketProxy:
    """Open a local stream bridged to ``host:port`` by system Python.

    The helper does not report ready until it established the direct remote
    connection, so a rank-control handshake cannot be buffered against a
    dead proxy and mistaken for a connected worker.
    """

    if not 1 <= int(port) <= 65535 or timeout <= 0:
        raise ValueError("control socket proxy endpoint is invalid")
    executable = _system_python()
    if executable is None:
        raise RuntimeError("system Python control proxy is unavailable")
    process = subprocess.Popen(
        [executable, "-u", "-c", _PROXY_PROGRAM, host, str(port), str(timeout)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    stream: socket.socket | None = None
    try:
        deadline = time.monotonic() + timeout
        port_line = _read_line(process, deadline=deadline)
        label, separator, value = port_line.partition(b" ")
        if label != b"PORT" or not separator or not value.isdigit():
            raise RuntimeError("control socket helper returned an invalid port")
        local_port = int(value)
        stream = socket.create_connection(
            ("127.0.0.1", local_port),
            timeout=max(0.05, deadline - time.monotonic()),
        )
        if _read_line(process, deadline=deadline) != b"READY":
            raise RuntimeError("control socket helper did not reach its coordinator")
        return SystemSocketProxy(stream=stream, process=process)
    except BaseException:
        if stream is not None:
            stream.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        raise


__all__ = ["SystemSocketProxy", "open_system_tcp_proxy", "should_proxy_control_socket"]
