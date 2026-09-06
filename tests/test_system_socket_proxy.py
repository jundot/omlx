# SPDX-License-Identifier: Apache-2.0
"""Tests for the macOS system-Python carrier used by discovery probes."""

from __future__ import annotations

import socket
import threading

import pytest

from omlx.cluster.system_socket_proxy import (
    open_system_tcp_proxy,
    should_proxy_control_socket,
)


def test_system_proxy_bridges_a_loopback_stream():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = int(listener.getsockname()[1])
    result: dict[str, bytes] = {}

    def server() -> None:
        stream, _address = listener.accept()
        try:
            result["request"] = stream.recv(4)
            stream.sendall(b"pong")
        finally:
            stream.close()
            listener.close()

    thread = threading.Thread(target=server)
    thread.start()
    proxy = open_system_tcp_proxy("127.0.0.1", port, timeout=3)
    try:
        proxy.stream.sendall(b"ping")
        assert proxy.stream.recv(4) == b"pong"
    finally:
        proxy.close()
    thread.join(3)

    assert not thread.is_alive()
    assert result == {"request": b"ping"}


def test_control_proxy_auto_skips_loopback(monkeypatch):
    monkeypatch.delenv("OMLX_CLUSTER_CONTROL_TRANSPORT", raising=False)
    assert should_proxy_control_socket("127.0.0.1") is False
    assert should_proxy_control_socket("localhost") is False


def test_control_proxy_direct_override(monkeypatch):
    monkeypatch.setenv("OMLX_CLUSTER_CONTROL_TRANSPORT", "direct")
    assert should_proxy_control_socket("10.0.0.1") is False


def test_system_proxy_reaches_ipv6_loopback():
    if not socket.has_ipv6:
        pytest.skip("IPv6 is unavailable")
    listener = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    try:
        listener.bind(("::1", 0))
    except OSError:
        listener.close()
        pytest.skip("IPv6 loopback is unavailable")
    listener.listen(1)
    port = int(listener.getsockname()[1])

    def server() -> None:
        stream, _address = listener.accept()
        try:
            stream.sendall(b"ok")
        finally:
            stream.close()
            listener.close()

    thread = threading.Thread(target=server)
    thread.start()
    proxy = open_system_tcp_proxy("::1", port, timeout=3)
    try:
        assert proxy.stream.recv(2) == b"ok"
    finally:
        proxy.close()
    thread.join(3)
    assert not thread.is_alive()
