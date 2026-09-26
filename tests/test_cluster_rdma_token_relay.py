# SPDX-License-Identifier: Apache-2.0
"""Sampled tokens take the RDMA relay only when every edge is live and rank zero samples alone."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from rdma_loopback import LoopbackLink, PythonWordOps

from omlx.cluster.rdma.stage_plan import StageLink
from omlx.cluster.rdma.stage_transport import install_stage_links
from omlx.cluster.rdma.token_relay import TokenRelay


@pytest.fixture
def link():
    loop = LoopbackLink(request_bytes=64 * 1024, reply_bytes=64 * 1024)
    yield loop
    loop.close()


def _vote(monkeypatch, flags):
    monkeypatch.setattr(
        mx.distributed,
        "all_gather",
        lambda value, group=None: mx.array([*value.tolist(), *flags], dtype=mx.int32),
    )


def _install(link, rank=0):
    return install_stage_links(
        mx,
        None,
        (StageLink(1, 0, link.name, link.socket_path),),
        rank=rank,
        ops_loader=lambda: (PythonWordOps(), ""),
    )


def test_a_fully_live_pipeline_relays_tokens_once_sampling_is_rank_zero_only(
    link, monkeypatch
):
    _vote(monkeypatch, [0, 1])
    with _install(link) as state:
        assert state.relay.ready
        assert state.report["token_relay"] == {
            "active": False,
            "reason": "waiting for the rank-zero sampling decision",
        }
        assert state.relay.activate(True)
        assert state.report["token_relay"]["active"] is True


def test_tokens_stay_on_the_ring_while_every_rank_samples(link, monkeypatch):
    _vote(monkeypatch, [0, 1])
    with _install(link) as state:
        assert not state.relay.activate(False)
        assert state.report["token_relay"] == {
            "active": False,
            "reason": "rank-zero sampling is off, so tokens stay on the ring",
        }


def test_an_edge_that_is_not_live_keeps_the_tokens_on_the_ring(link, monkeypatch):
    _vote(monkeypatch, [0, 0])
    with _install(link) as state:
        assert not state.relay.ready
        assert not state.relay.activate(True)
        assert state.report["token_relay"]["reason"] == "an RDMA stage edge is not live"


def test_an_edge_without_a_link_keeps_the_tokens_on_the_ring(link, monkeypatch):
    # Three ranks, but only edge 1 -> 0 has a link.
    _vote(monkeypatch, [0, 1, 1, 1])
    with _install(link) as state:
        assert state.report["active"]
        assert not state.relay.activate(True)
        assert (
            state.report["token_relay"]["reason"]
            == "1 of 2 stage edges have no RDMA link"
        )


class _Receiver:
    def __init__(self):
        self.prepost = True
        self.sent = []

    def send_tokens(self, values):
        self.sent.append(np.asarray(values).tolist())


class _Sender:
    def __init__(self, values):
        self.values = np.array(values, dtype=np.uint32)
        self.counts = []

    def take_tokens(self, count):
        self.counts.append(count)
        return self.values


def _relay(receiver, sender):
    relay = TokenRelay(receiver, sender, ready=True, reason="", report={})
    assert relay.activate(True)
    return relay


def test_rank_zero_sends_its_own_samples_and_stops_pre_posting():
    receiver = _Receiver()
    relay = _relay(receiver, None)
    sampled = mx.array([5, 6], dtype=mx.uint32)
    assert relay.broadcast(mx, sampled, 2) is sampled
    assert receiver.sent == [[5, 6]]
    assert receiver.prepost is False


def test_a_middle_rank_passes_the_tokens_on():
    receiver, sender = _Receiver(), _Sender([8, 9])
    relay = _relay(receiver, sender)
    result = relay.broadcast(mx, mx.zeros((2,), dtype=mx.uint32), 2)
    assert sender.counts == [2]
    assert receiver.sent == [[8, 9]]
    assert result.dtype == mx.uint32 and result.tolist() == [8, 9]


def test_the_first_stage_only_takes_the_tokens():
    sender = _Sender([3])
    relay = _relay(None, sender)
    assert relay.broadcast(mx, mx.zeros((1,), dtype=mx.uint32), 1).tolist() == [3]
