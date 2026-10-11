# SPDX-License-Identifier: Apache-2.0
"""Stage activations cross the RDMA edge intact, and every rank agrees before any edge is used."""

from __future__ import annotations

import multiprocessing
import threading
import time

import mlx.core as mx
import pytest
from rdma_loopback import LoopbackLink, PythonWordOps

from omlx.cluster.rdma import frames
from omlx.cluster.rdma.mailbox import ClientMailbox, ServiceMailbox
from omlx.cluster.rdma.stage_plan import StageLink
from omlx.cluster.rdma.stage_transport import (
    LinkDownError,
    StageReceiver,
    StageSender,
    install_stage_links,
)


@pytest.fixture
def link():
    # Small halves force multi-frame messages.
    loop = LoopbackLink(request_bytes=64 * 1024, reply_bytes=64 * 1024)
    yield loop
    loop.close()


def _stage(link):
    return StageLink(1, 0, link.name, link.socket_path)


def _messages():
    mx.random.seed(3)
    return [
        (mx.random.normal((2, 3, 4096)) * 3).astype(mx.bfloat16),
        mx.random.normal((1, 64, 1024)),
        mx.arange(10, dtype=mx.int32).reshape(2, 5)[:, ::2],
    ]


def _sender_main(name, socket_path, mailbox_path, which):
    # The sending rank is its own process, as in a deployment; MLX streams are per thread.
    service = ServiceMailbox.attach(
        name, socket_path, PythonWordOps(), mailbox_path=mailbox_path
    )
    try:
        sender = StageSender(
            mx, service, StageLink(1, 0, name, socket_path), timeout_s=20
        )
        messages = (
            _messages()
            if which == "all"
            else [mx.ones((4, 4))]
            if which == "ones"
            else _messages()[:1]
        )
        for message in messages:
            assert sender.send(message) is message
    finally:
        service.close()


def _start_sender(link, which):
    process = multiprocessing.get_context("spawn").Process(
        target=_sender_main,
        args=(link.name, link.socket_path, link.mailbox_path, which),
        daemon=True,
    )
    process.start()
    # A deployment's vote guarantees the service end attached before the first request.
    link.wait_service()
    return process


def test_messages_arrive_bit_exact_across_single_and_multi_frame_sends(link):
    process = _start_sender(link, "all")
    client = ClientMailbox.attach(link.name, PythonWordOps())
    receiver = StageReceiver(mx, client, _stage(link), timeout_s=60)
    try:
        for message in _messages():
            received = receiver.recv_like(mx.zeros(message.shape, dtype=message.dtype))
            assert received.dtype == message.dtype and received.shape == message.shape
            assert mx.array_equal(received, message).item()
    finally:
        process.join(timeout=30)
        client.close()
    assert process.exitcode == 0


def test_a_template_that_disagrees_with_the_sender_is_refused(link):
    process = _start_sender(link, "ones")
    client = ClientMailbox.attach(link.name, PythonWordOps())
    receiver = StageReceiver(mx, client, _stage(link), timeout_s=60)
    try:
        with pytest.raises(frames.FrameError, match="stage frame mismatch"):
            receiver.recv_like(mx.zeros((4, 5)))
    finally:
        process.join(timeout=30)
        client.close()


def test_a_link_that_drops_while_waiting_raises_instead_of_hanging(link):
    client = ClientMailbox.attach(link.name, PythonWordOps())
    receiver = StageReceiver(mx, client, _stage(link), timeout_s=30)
    threading.Timer(0.2, link.drop).start()
    try:
        with pytest.raises(LinkDownError, match="went down while waiting for rank 1"):
            receiver.recv_like(mx.zeros((8,)))
    finally:
        client.close()


def test_a_link_that_reconnects_under_a_request_raises_instead_of_waiting(link):
    client = ClientMailbox.attach(link.name, PythonWordOps())
    receiver = StageReceiver(mx, client, _stage(link), timeout_s=5)
    threading.Timer(0.2, link.flap).start()
    began = time.monotonic()
    try:
        with pytest.raises(LinkDownError, match="reconnected while waiting for rank 1"):
            receiver.recv_like(mx.zeros((8,)))
    finally:
        client.close()
    assert time.monotonic() - began < 3


def test_a_link_that_resets_under_a_waiting_sender_raises_instead_of_waiting(link):
    service = ServiceMailbox.attach(
        link.name, link.socket_path, PythonWordOps(), mailbox_path=link.mailbox_path
    )
    sender = StageSender(mx, service, _stage(link), timeout_s=5)
    threading.Timer(0.2, link.flap).start()
    began = time.monotonic()
    try:
        with pytest.raises(LinkDownError, match="dropped the service end"):
            sender.send(mx.ones((4, 4)))
    finally:
        service.close()
    assert time.monotonic() - began < 3


def _fake_gather(monkeypatch, peers):
    def all_gather(value, group=None):
        votes = [*value.tolist(), *peers]
        return mx.array(votes, dtype=mx.int32)

    monkeypatch.setattr(mx.distributed, "all_gather", all_gather)


def test_rank_zero_routes_its_incoming_edge_over_rdma_when_both_ends_agree(
    link, monkeypatch
):
    _fake_gather(monkeypatch, [0, 1])
    original_send, original_recv = mx.distributed.send, mx.distributed.recv_like
    process = _start_sender(link, "first")
    message = _messages()[0]
    try:
        with install_stage_links(
            mx,
            None,
            (_stage(link),),
            rank=0,
            ops_loader=lambda: (PythonWordOps(), ""),
            timeout_s=60,
        ) as state:
            assert state["active"] and state["edges"][0]["active"]
            received = mx.distributed.recv_like(
                mx.zeros(message.shape, dtype=message.dtype), 1
            )
            assert mx.array_equal(received, message).item()
    finally:
        process.join(timeout=30)
    assert process.exitcode == 0
    assert (
        mx.distributed.send is original_send
        and mx.distributed.recv_like is original_recv
    )


def test_an_edge_the_peer_could_not_attach_stays_on_the_ring(link, monkeypatch):
    _fake_gather(monkeypatch, [0, 0])
    ring = []
    monkeypatch.setattr(
        mx.distributed,
        "recv_like",
        lambda value, src, **kwargs: ring.append(src) or value,
    )
    with install_stage_links(
        mx, None, (_stage(link),), rank=0, ops_loader=lambda: (PythonWordOps(), "")
    ) as state:
        edge = state["edges"][0]
        assert not state["active"] and not edge["active"]
        assert edge["reason"] == "rank 1 could not attach its end"
        mx.distributed.recv_like(mx.zeros((2,)), 1)
    assert ring == [1]


def test_a_rank_that_cannot_load_the_helper_still_votes(link, monkeypatch):
    calls = []

    def all_gather(value, group=None):
        calls.append(value.tolist())
        return mx.array([*value.tolist(), 1, 1], dtype=mx.int32)

    monkeypatch.setattr(mx.distributed, "all_gather", all_gather)
    with install_stage_links(
        mx,
        None,
        (_stage(link),),
        rank=0,
        ops_loader=lambda: (None, "libmcdma-rpc is not installed"),
    ) as state:
        assert calls == [[0, 0]]
        assert state["edges"][0]["reason"] == "libmcdma-rpc is not installed"
        assert not state["active"]


def test_a_sending_rank_whose_daemon_reports_the_link_down_votes_no(link, monkeypatch):
    link.link_up = False
    # Rank 0 attached its receiving end; this rank's own vote follows it.
    monkeypatch.setattr(
        mx.distributed,
        "all_gather",
        lambda value, group=None: mx.array([1, 0, *value.tolist()], dtype=mx.int32),
    )
    with install_stage_links(
        mx,
        None,
        (_stage(link),),
        rank=1,
        ops_loader=lambda: (PythonWordOps(), ""),
        attach_service=lambda name, sock, ops: ServiceMailbox.attach(
            name, sock, ops, mailbox_path=link.mailbox_path
        ),
    ) as state:
        (edge,) = state["edges"]
    assert not edge["active"]
    assert edge["reason"] == "mcdma-rpcd reports the link down"


def test_a_deployment_without_stage_links_changes_nothing(monkeypatch):
    def never(*args, **kwargs):
        raise AssertionError("no vote without stage links")

    monkeypatch.setattr(mx.distributed, "all_gather", never)
    original = mx.distributed.send
    with install_stage_links(mx, None, (), rank=0) as state:
        assert state == {
            "enabled": False,
            "active": False,
            "reason": "no verified RDMA stage link",
            "edges": [],
        }
        assert mx.distributed.send is original
