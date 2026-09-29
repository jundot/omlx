# SPDX-License-Identifier: Apache-2.0
"""Rejected memory hits must not hide usable rank-local SSD snapshots."""

import struct
from types import SimpleNamespace

import mlx.core as mx
import mlx_lm.server as server
import pytest
from mlx_lm.models.cache import KVCache

from omlx.cluster.prompt_snapshot_cache import SSDPromptSnapshotStore
from omlx.cluster.telemetry import install_server_telemetry

MODEL = "synthetic-model"
TOKENS = list(range(10))


def kv(offset):
    cache = KVCache()
    cache.update_and_fetch(mx.ones((1, 1, offset, 2)), mx.ones((1, 1, offset, 2)))
    return [cache]


class Marker:
    def update(self, *args, **kwargs):
        pass


class Agreement:
    def __init__(self, peer_plans):
        self.peer_plans = iter(peer_plans)
        self.local_plans = []

    def broadcast_owned_bytes(self, payload, *, source_rank, expected_size):
        assert expected_size == 24
        if source_rank == 0:
            self.local_plans.append(struct.unpack("!QQQ", payload))
            return payload
        return struct.pack("!QQQ", *next(self.peer_plans))


def setup(monkeypatch, tmp_path, memory, rest, peers, *, ssd=True, votes=2):
    if ssd:
        store = SSDPromptSnapshotStore(tmp_path, step=4, persistent=True)
        assert store.put(MODEL, TOKENS[:4], kv(4))
        assert store.put(MODEL, TOKENS[:8], kv(8))
    monkeypatch.setattr(
        mx.distributed, "init", lambda: SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    )
    collective_calls = []

    def all_sum(value):
        collective_calls.append(value.tolist())
        return value * votes

    monkeypatch.setattr(mx.distributed, "all_sum", all_sum)
    monkeypatch.setattr(
        server.LRUPromptCache, "fetch_nearest_cache", lambda *args: (memory, rest)
    )
    agreement = Agreement(peers)
    context = install_server_telemetry(
        Marker(),
        heartbeat_interval=0,
        control_plane=agreement,
        ssd_cache_dir=str(tmp_path) if ssd else None,
        ssd_cache_persistent=True,
        prefill_step_size=4,
    )
    return context, agreement, collective_calls


@pytest.mark.parametrize("local_hit", [True, False])
def test_memory_disagreement_retries_shared_ssd(monkeypatch, tmp_path, local_hit):
    memory, rest = (kv(4), TOKENS[4:]) if local_hit else (None, TOKENS)
    peer = (0, 10, 0) if local_hit else (4, 6, 0)
    ctx, agreement, calls = setup(
        monkeypatch, tmp_path, memory, rest, [peer, (8, 2, 0)]
    )
    with ctx as telemetry:
        hit, remaining = server.LRUPromptCache().fetch_nearest_cache(MODEL, TOKENS)
        assert hit[0].offset == 8
        assert remaining == TOKENS[8:]
        assert telemetry.snapshot()["cache"]["tokens_reused"] == 8
    assert len(agreement.local_plans) == 2
    assert calls == [[1, 1]]


def test_invalid_memory_offset_retries_ssd(monkeypatch, tmp_path):
    ctx, _, calls = setup(
        monkeypatch, tmp_path, kv(9), TOKENS[8:], [(8, 2, 1), (8, 2, 0)]
    )
    with ctx:
        hit, rest = server.LRUPromptCache().fetch_nearest_cache(MODEL, TOKENS)
        assert hit[0].offset == 8
        assert rest == TOKENS[8:]
    assert calls == [[1, 1]]


def test_valid_shared_memory_hit_skips_ssd(monkeypatch, tmp_path):
    ctx, _, calls = setup(monkeypatch, tmp_path, kv(8), TOKENS[8:], [(8, 2, 0)])
    with ctx:
        hit, rest = server.LRUPromptCache().fetch_nearest_cache(MODEL, TOKENS)
        assert hit[0].offset == 8
        assert rest == TOKENS[8:]
    assert calls == []


@pytest.mark.parametrize("failure", ["local_load", "peer_load", "no_common_boundary"])
def test_ssd_failure_returns_synchronized_full_prefill(monkeypatch, tmp_path, failure):
    peers = [(8, 2, 1)]
    if failure != "no_common_boundary":
        peers.append((0, 10, 0) if failure == "peer_load" else (8, 2, 0))
    ctx, agreement, _ = setup(
        monkeypatch,
        tmp_path,
        kv(9),
        TOKENS[8:],
        peers,
        votes=1 if failure == "no_common_boundary" else 2,
    )
    if failure == "local_load":
        monkeypatch.setattr(SSDPromptSnapshotStore, "load", lambda *args: None)
    with ctx as telemetry:
        hit, rest = server.LRUPromptCache().fetch_nearest_cache(MODEL, TOKENS)
        assert hit is None
        assert rest == TOKENS
        assert telemetry.snapshot()["cache"]["tokens_reused"] == 0
    assert len(agreement.local_plans) == (1 if failure == "no_common_boundary" else 2)


def test_disabled_ssd_keeps_safe_memory_rejection(monkeypatch, tmp_path):
    ctx, _, calls = setup(
        monkeypatch, tmp_path, kv(9), TOKENS[8:], [(8, 2, 1)], ssd=False
    )
    with ctx:
        hit, rest = server.LRUPromptCache().fetch_nearest_cache(MODEL, TOKENS)
        assert hit is None and rest == TOKENS
    assert calls == []
