# SPDX-License-Identifier: Apache-2.0
"""Reliable TCP control plane kept independent of MLX/JACCL collectives."""

import pickle
import socket
import struct
import threading
import zlib

import pytest

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
    rogue = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    for _attempt in range(100):
        try:
            rogue.connect(("127.0.0.1", port))
            break
        except ConnectionRefusedError:
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
