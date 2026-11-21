# SPDX-License-Identifier: Apache-2.0
"""Tests for the bounded Python control plane used to bootstrap JACCL."""

from __future__ import annotations

import contextlib
import functools
import socket
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from omlx.cluster import inference_worker, jaccl_side_channel
from omlx.cluster.control_plane import RankControlPlane
from omlx.cluster.jaccl_side_channel import (
    _coordinator_endpoint,
    init_cluster_group,
    jaccl_all_gather_factory,
)


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def test_socket_side_channel_orders_ranks_and_reuses_connections(monkeypatch):
    port = _free_loopback_port()
    monkeypatch.setenv("MLX_JACCL_COORDINATOR", f"127.0.0.1:{port}")
    monkeypatch.setattr(jaccl_side_channel, "_DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS", 3.0)
    monkeypatch.setattr(jaccl_side_channel, "_side_channel_transport", lambda: "direct")
    results: dict[str, bytes] = {}
    errors: list[BaseException] = []
    server_ready = threading.Event()

    def rank_zero() -> None:
        try:
            gather = jaccl_all_gather_factory(0, 2)
            server_ready.set()
            results["first_server"] = gather(b"aa", 2)
            results["second_server"] = gather(b"cc", 2)
        except BaseException as exc:  # propagate thread failures to pytest
            errors.append(exc)
            server_ready.set()

    thread = threading.Thread(target=rank_zero)
    thread.start()
    gather = jaccl_all_gather_factory(1, 2)
    assert server_ready.wait(3)
    results["first_client"] = gather(b"bb", 2)
    results["second_client"] = gather(b"dd", 2)
    thread.join(3)

    assert not thread.is_alive()
    assert not errors
    assert results == {
        "first_server": b"aabb",
        "first_client": b"aabb",
        "second_server": b"ccdd",
        "second_client": b"ccdd",
    }


def test_sidecar_orders_ranks_without_using_parent_network(monkeypatch, tmp_path):
    port = _free_loopback_port()
    # oMLX.app exports PYTHONHOME for its bundled interpreter.
    monkeypatch.setenv("PYTHONHOME", str(tmp_path))
    monkeypatch.setenv("MLX_JACCL_COORDINATOR", f"127.0.0.1:{port}")
    monkeypatch.setattr(jaccl_side_channel, "_DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS", 3.0)
    monkeypatch.setattr(
        jaccl_side_channel, "_side_channel_transport", lambda: "sidecar"
    )
    results: dict[str, bytes] = {}
    errors: list[BaseException] = []

    first = jaccl_all_gather_factory(0, 2)
    second = jaccl_all_gather_factory(1, 2)

    def rank_zero() -> None:
        try:
            results["server"] = first(b"aa", 2)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=rank_zero)
    thread.start()
    results["client"] = second(b"bb", 2)
    thread.join(3)
    first.close()
    second.close()

    assert not thread.is_alive()
    assert not errors
    assert results == {"server": b"aabb", "client": b"aabb"}


@pytest.fixture
def second_local_ipv4() -> str:
    """A local IPv4 address besides 127.0.0.1, for rank zero's second listener.

    Linux answers on all of 127.0.0.0/8. macOS answers on 127.0.0.1 alone, so
    the address of its default-route interface stands in.
    """

    candidates = ["127.0.0.2"]
    with (
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe,
        contextlib.suppress(OSError),
    ):
        # Connecting a datagram socket picks a route and sends nothing.
        probe.connect(("192.0.2.1", 9))
        candidates.append(probe.getsockname()[0])
    for candidate in candidates:
        with (
            socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener,
            contextlib.suppress(OSError),
        ):
            listener.bind((candidate, 0))
            return candidate
    pytest.skip("this host has no local IPv4 address besides 127.0.0.1")


def _mesh(monkeypatch, rank_two_address: str, *, timeout: float) -> SimpleNamespace:
    """Announce rank zero where no rank reaches it, and give each rank a route.

    mlx.launch names rank zero by one address for every rank; 192.0.2.1
    (TEST-NET-1) stands in for it. Rank one is sent to 127.0.0.1 and rank two
    to another local address, so rank zero has to accept on both. Returns
    the rank-control arguments every rank is launched with.
    """

    monkeypatch.setenv("MLX_JACCL_COORDINATOR", f"192.0.2.1:{_free_loopback_port()}")
    monkeypatch.setenv(
        "OMLX_COORDINATOR_ADDRESSES", f"192.0.2.1,127.0.0.1,{rank_two_address}"
    )
    monkeypatch.setattr(
        jaccl_side_channel, "_DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS", timeout
    )
    monkeypatch.setattr(
        inference_worker,
        "RankControlPlane",
        functools.partial(
            RankControlPlane, connect_timeout=timeout, io_timeout=timeout
        ),
    )
    return SimpleNamespace(
        control_host="192.0.2.1",
        control_port=_free_loopback_port(),
        control_token="a" * 64,
    )


@pytest.mark.parametrize("transport", ["direct", "sidecar"])
def test_every_rank_bootstraps_on_its_own_route_to_rank_zero(
    monkeypatch, second_local_ipv4, transport
):
    if transport == "direct" and sys.platform == "darwin":
        # A rank can connect between rank zero's bind() and listen(), and macOS
        # drops that SYN instead of refusing it (about 1 run in 600).
        pytest.skip("macOS bootstraps through the sidecar")
    args = _mesh(monkeypatch, second_local_ipv4, timeout=5.0)
    monkeypatch.setattr(
        jaccl_side_channel, "_side_channel_transport", lambda: transport
    )
    results: dict[int, tuple[bytes, object]] = {}
    errors: list[BaseException] = []

    def rank(index: int) -> None:
        # Both bootstrap channels, in the order run_worker opens them.
        try:
            gather = jaccl_all_gather_factory(index, 3)
            gathered = gather(str(index).encode(), 1)
            with inference_worker._rank_control_plane(
                args, rank=index, world_size=3
            ) as control:
                sent = "go" if index == 0 else None
                results[index] = (gathered, control.broadcast_object(sent))
            if transport == "sidecar":
                gather.close()
        except BaseException as exc:  # propagate thread failures to pytest
            errors.append(exc)

    threads = [
        threading.Thread(target=rank, args=(index,), daemon=True) for index in range(3)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(15)

    assert errors == []
    assert results == {index: (b"012", "go") for index in range(3)}


@pytest.mark.parametrize("channel", ["direct", "sidecar", "rank-control"])
def test_rank_zero_on_two_addresses_still_gives_up_at_its_deadline(
    monkeypatch, second_local_ipv4, channel
):
    args = _mesh(monkeypatch, second_local_ipv4, timeout=0.5)
    if channel != "rank-control":
        monkeypatch.setattr(
            jaccl_side_channel, "_side_channel_transport", lambda: channel
        )
    waited: list[float] = []

    def rank_zero() -> None:
        # Neither rank one nor rank two ever arrives.
        started = time.monotonic()
        with contextlib.suppress(RuntimeError, TimeoutError):
            if channel == "rank-control":
                with inference_worker._rank_control_plane(args, rank=0, world_size=3):
                    pass
            else:
                jaccl_all_gather_factory(0, 3)(b"0", 1)
        waited.append(time.monotonic() - started)

    thread = threading.Thread(target=rank_zero, daemon=True)
    thread.start()
    thread.join(10)

    assert not thread.is_alive(), "rank zero kept waiting past its deadline"
    assert 0.5 <= waited[0] < 5


@pytest.mark.parametrize("value", ["", "host", "127.0.0.1:0", "[::1]:8000"])
def test_side_channel_requires_a_valid_ipv4_coordinator(monkeypatch, value):
    monkeypatch.setenv("MLX_JACCL_COORDINATOR", value)
    with pytest.raises(RuntimeError, match="coordinator|IPv4"):
        _coordinator_endpoint()


def test_init_cluster_group_only_injects_factory_for_jaccl(monkeypatch):
    calls: list[dict[str, object]] = []

    class Distributed:
        def init(self, **kwargs):
            calls.append(kwargs)
            return "group"

    Distributed.init.__doc__ = "init(..., *, all_gather_factory: Callable = None)"

    class MX:
        distributed = Distributed()

    monkeypatch.setenv("MLX_JACCL_COORDINATOR", "10.0.0.1:8000")
    assert init_cluster_group(MX(), backend="jaccl", strict=True) == "group"
    assert calls[-1]["backend"] == "jaccl"
    assert calls[-1]["strict"] is True
    assert calls[-1]["all_gather_factory"] is jaccl_all_gather_factory

    assert init_cluster_group(MX(), backend="ring", strict=False) == "group"
    assert calls[-1] == {"backend": "ring", "strict": False}


def test_init_cluster_group_does_not_trust_a_newer_stub_than_native():
    calls: list[dict[str, object]] = []

    class Distributed:
        def init(self, **kwargs):
            calls.append(kwargs)
            return "group"

    class MX:
        distributed = Distributed()

    assert init_cluster_group(MX(), backend="jaccl", strict=True) == "group"
    assert calls == [{"backend": "jaccl", "strict": True}]
