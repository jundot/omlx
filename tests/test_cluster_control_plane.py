# SPDX-License-Identifier: Apache-2.0
"""Reliable TCP control plane kept independent of MLX/JACCL collectives."""

import errno
import pickle
import socket
import struct
import sys
import threading
import zlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from omlx.cluster import control_plane as control_module
from omlx.cluster import system_socket_proxy as proxy_module
from omlx.cluster.control_plane import RankControlPlane


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
        stream.bind(("127.0.0.1", 0))
        return int(stream.getsockname()[1])


def test_rank_control_plane_broadcasts_objects_in_sequence():
    port = _free_port()
    token = "a" * 64
    expected = [None, ("request", {"max_tokens": 7}), [], [3, 9]]
    received = []
    failures = []

    def coordinator():
        try:
            with RankControlPlane(
                rank=0,
                world_size=2,
                host="127.0.0.1",
                port=port,
                token=token,
                connect_timeout=5,
                io_timeout=5,
            ) as control:
                for value in expected:
                    assert control.broadcast_object(value) is value
        except Exception as exc:  # pragma: no cover - relayed to main thread
            failures.append(exc)

    thread = threading.Thread(target=coordinator)
    thread.start()
    try:
        with RankControlPlane(
            rank=1,
            world_size=2,
            host="127.0.0.1",
            port=port,
            token=token,
            connect_timeout=5,
            io_timeout=5,
        ) as control:
            for _ in expected:
                received.append(control.broadcast_object(None))
    finally:
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert failures == []
    assert received == expected


def test_rank_control_plane_supports_barrier_and_nonzero_owned_bytes():
    port = _free_port()
    token = "c" * 64
    worker_owned = b"worker-one-cache-plan"
    coordinator_owned = b"rank-zero-follow-up"
    received = {}
    failures = []

    def participant(rank):
        try:
            with RankControlPlane(
                rank=rank,
                world_size=3,
                host="127.0.0.1",
                port=port,
                token=token,
                connect_timeout=5,
                io_timeout=5,
            ) as control:
                obj = control.broadcast_object(
                    {"kind": "request"} if rank == 0 else None
                )
                control.barrier()
                from_worker = control.broadcast_owned_bytes(
                    worker_owned if rank == 1 else None,
                    source_rank=1,
                    expected_size=len(worker_owned),
                )
                from_coordinator = control.broadcast_owned_bytes(
                    coordinator_owned if rank == 0 else None,
                    source_rank=0,
                    expected_size=len(coordinator_owned),
                )
                received[rank] = (obj, from_worker, from_coordinator)
        except Exception as exc:  # pragma: no cover - relayed below
            failures.append(exc)

    threads = [threading.Thread(target=participant, args=(rank,)) for rank in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=8)

    assert all(not thread.is_alive() for thread in threads)
    assert failures == []
    assert received == {
        rank: ({"kind": "request"}, worker_owned, coordinator_owned)
        for rank in range(3)
    }


def test_rank_control_plane_rejects_invalid_identity():
    try:
        RankControlPlane(
            rank=2,
            world_size=2,
            host="127.0.0.1",
            port=12345,
            token="x",
        )
    except ValueError as exc:
        assert "identity" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid rank identity was accepted")


def test_invalid_handshake_is_dropped_without_blocking_a_valid_rank():
    port = _free_port()
    token = "b" * 64
    failures = []

    def coordinator():
        try:
            with RankControlPlane(
                rank=0,
                world_size=2,
                host="127.0.0.1",
                port=port,
                token=token,
                connect_timeout=3,
                io_timeout=3,
            ) as control:
                control.broadcast_object({"ready": True})
        except Exception as exc:  # pragma: no cover - relayed below
            failures.append(exc)

    thread = threading.Thread(target=coordinator)
    thread.start()
    for _attempt in range(100):
        rogue = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            rogue.connect(("127.0.0.1", port))
            break
        except ConnectionRefusedError:
            rogue.close()
            threading.Event().wait(0.01)
    else:  # pragma: no cover - diagnostics for a wedged test host
        pytest.fail("coordinator listener did not start")
    rogue.sendall(b"x" * struct.calcsize("!4sII64s"))
    rogue.close()

    with RankControlPlane(
        rank=1,
        world_size=2,
        host="127.0.0.1",
        port=port,
        token=token,
        connect_timeout=3,
        io_timeout=3,
    ) as control:
        assert control.broadcast_object(None) == {"ready": True}
    thread.join(3)

    assert not thread.is_alive()
    assert failures == []


def test_worker_requires_a_valid_coordinator_acknowledgement():
    port = _free_port()
    ready = threading.Event()

    def fake_coordinator():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", port))
            listener.listen(1)
            ready.set()
            stream, _ = listener.accept()
            with stream:
                challenge = b"q" * 32
                stream.sendall(struct.pack("!4sI32s", b"OC2C", 1, challenge))
                handshake = stream.recv(struct.calcsize("!4sII32s"))
                assert b"c" * 64 not in handshake
                stream.sendall(struct.pack("!4sI32s", b"NOPE", 1, b"\0" * 32))

    thread = threading.Thread(target=fake_coordinator)
    thread.start()
    assert ready.wait(2)
    with (
        pytest.raises(RuntimeError, match="not acknowledged"),
        RankControlPlane(
            rank=1,
            world_size=2,
            host="127.0.0.1",
            port=port,
            token="c" * 64,
            connect_timeout=2,
            io_timeout=2,
        ),
    ):
        pass
    thread.join(2)
    assert not thread.is_alive()


def test_worker_authenticates_payload_before_unpickling():
    sender, receiver = socket.socketpair()
    control = RankControlPlane(
        rank=1,
        world_size=2,
        host="127.0.0.1",
        port=12345,
        token="d" * 64,
    )
    control._stream = receiver
    payload = pickle.dumps({"unsafe": "payload"})
    sender.sendall(
        struct.pack(
            "!4sIIII32s",
            b"OC2M",
            1,
            1,
            len(payload),
            zlib.crc32(payload),
            b"\0" * 32,
        )
        + payload
    )
    try:
        with pytest.raises(RuntimeError, match="authentication"):
            control.broadcast_object(None)
    finally:
        sender.close()
        control.close()


@pytest.mark.parametrize("error_number", [errno.EACCES, errno.EPERM, errno.ETIMEDOUT])
def test_auto_transport_falls_back_before_coordinator_deadline(
    monkeypatch, error_number
):
    port = _free_port()
    monkeypatch.setenv("OMLX_CLUSTER_CONTROL_TRANSPORT", "auto")
    monkeypatch.setenv("OMLX_CLUSTER_CONTROL_PROXY_PYTHON", sys.executable)
    # Exercise the non-loopback macOS policy using a real local coordinator.
    monkeypatch.setattr(proxy_module, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(
        control_module,
        "should_proxy_control_socket",
        lambda host: proxy_module.should_proxy_control_socket("10.0.0.1"),
    )
    original_connect = socket.socket.connect
    worker_thread = threading.current_thread()
    attempts = []

    def connect(stream, address):
        if threading.current_thread() is worker_thread and address[1] == port:
            attempts.append(address)
            if error_number == errno.ETIMEDOUT:
                threading.Event().wait(stream.gettimeout())
            raise OSError(error_number, "Injected direct connection failure")
        return original_connect(stream, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    failures = []

    def coordinator():
        try:
            with RankControlPlane(
                rank=0,
                world_size=2,
                host="127.0.0.1",
                port=port,
                token="test",
                connect_timeout=2,
                io_timeout=2,
            ) as control:
                control.broadcast_object({"proxy": "authenticated"})
        except Exception as exc:
            failures.append(exc)

    thread = threading.Thread(target=coordinator)
    thread.start()
    try:
        with RankControlPlane(
            rank=1,
            world_size=2,
            host="127.0.0.1",
            port=port,
            token="test",
            connect_timeout=2,
            io_timeout=3,
        ) as control:
            assert control.broadcast_object(None) == {"proxy": "authenticated"}
            assert control._stream.gettimeout() == 3
            proxy = control._stream_proxy
            assert proxy is not None
    finally:
        thread.join(3)
    assert not thread.is_alive()
    assert failures == []
    assert proxy.process.poll() is not None
    if error_number in (errno.EACCES, errno.EPERM):
        assert len(attempts) == 1


@pytest.mark.parametrize("mode", ["auto", "", "direct", "system-proxy", "invalid"])
def test_control_transport_overrides_and_validation(monkeypatch, mode):
    monkeypatch.setenv("OMLX_CLUSTER_CONTROL_TRANSPORT", mode)
    monkeypatch.setenv("OMLX_CLUSTER_CONTROL_PROXY_PYTHON", sys.executable)
    monkeypatch.setattr(proxy_module, "sys", SimpleNamespace(platform="darwin"))
    control = RankControlPlane(
        rank=1,
        world_size=2,
        host="10.0.0.1",
        port=12345,
        token="test",
    )
    direct = Mock()
    proxy = Mock()
    monkeypatch.setattr(control, "_connect_direct", direct)
    monkeypatch.setattr(control, "_connect_via_proxy", proxy)
    if mode == "invalid":
        with pytest.raises(RuntimeError, match="must be auto"):
            control._connect_to_coordinator()
        direct.assert_not_called()
        proxy.assert_not_called()
    else:
        control._connect_to_coordinator()
        assert direct.call_count == (mode != "system-proxy")
        assert proxy.call_count == (mode == "system-proxy")


@pytest.mark.parametrize("mode", ["auto", "", "direct"])
def test_transport_fallback_preserves_overall_connection_budget(monkeypatch, mode):
    monkeypatch.setenv("OMLX_CLUSTER_CONTROL_TRANSPORT", mode)
    monkeypatch.setenv("OMLX_CLUSTER_CONTROL_PROXY_PYTHON", sys.executable)
    monkeypatch.setattr(proxy_module, "sys", SimpleNamespace(platform="darwin"))
    now = [100.0]
    monkeypatch.setattr(control_module.time, "monotonic", lambda: now[0])
    control = RankControlPlane(
        rank=1,
        world_size=2,
        host="10.0.0.1",
        port=12345,
        token="test",
        connect_timeout=120,
    )

    def fail_direct(*, deadline, allow_proxy):
        assert deadline == 220.0
        assert allow_proxy == (mode != "direct")
        now[0] = 101.0
        raise TimeoutError("Direct connection timed out")

    proxy = Mock()
    monkeypatch.setattr(control, "_connect_direct", fail_direct)
    monkeypatch.setattr(control, "_connect_via_proxy", proxy)
    if mode == "direct":
        with pytest.raises(TimeoutError):
            control._connect_to_coordinator()
        proxy.assert_not_called()
    else:
        control._connect_to_coordinator()
        proxy.assert_called_once_with(deadline=220.0)


def test_auto_transport_does_not_retry_invalid_authentication_via_proxy(monkeypatch):
    monkeypatch.setattr(
        control_module, "should_proxy_control_socket", lambda host: True
    )
    monkeypatch.setenv("OMLX_CLUSTER_CONTROL_TRANSPORT", "auto")
    control = RankControlPlane(
        rank=1,
        world_size=2,
        host="10.0.0.1",
        port=12345,
        token="test",
    )
    monkeypatch.setattr(
        control,
        "_connect_direct",
        Mock(side_effect=RuntimeError("Invalid acknowledgement")),
    )
    proxy = Mock()
    monkeypatch.setattr(control, "_connect_via_proxy", proxy)
    with pytest.raises(RuntimeError, match="Invalid acknowledgement"):
        control._connect_to_coordinator()
    proxy.assert_not_called()


@pytest.mark.parametrize("listener_delay", [6.0, 120.0])
def test_auto_transport_waits_for_late_listener_without_switching_proxy(
    monkeypatch, listener_delay
):
    monkeypatch.setenv("OMLX_CLUSTER_CONTROL_TRANSPORT", "auto")
    monkeypatch.setattr(
        control_module, "should_proxy_control_socket", lambda host: True
    )
    now = [100.0]
    monkeypatch.setattr(control_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(control_module.time, "sleep", lambda delay: None)
    control = RankControlPlane(
        rank=1,
        world_size=2,
        host="10.0.0.1",
        port=12345,
        token="test",
        connect_timeout=120,
    )
    first = Mock()
    second = Mock()

    def refused(address):
        now[0] += listener_delay
        raise ConnectionRefusedError("Coordinator has not started listening")

    first.connect.side_effect = refused
    monkeypatch.setattr(
        control_module.socket, "socket", Mock(side_effect=[first, second])
    )
    monkeypatch.setattr(control, "_authenticate_worker_stream", Mock())
    proxy = Mock()
    monkeypatch.setattr(control, "_connect_via_proxy", proxy)
    try:
        if listener_delay == 120:
            with pytest.raises(TimeoutError):
                control._connect_to_coordinator()
        else:
            control._connect_to_coordinator()
            assert control._stream is second
        first.close.assert_called_once()
        proxy.assert_not_called()
    finally:
        control.close()
