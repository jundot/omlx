# SPDX-License-Identifier: Apache-2.0
"""P6 tests: the expert-streaming gate signal + MTP entry policy.

``streaming_gate_state()`` aggregates live bounded caches over a weakref
registry; ``_mtp_streaming_gate()`` maps the env mode + that signal to the
MTP entry decision (auto = cheap warmup, off = never enter while streaming
is live, on = untouched). These tests exercise the host-side logic only —
no MLX / GPU.
"""

import gc

import pytest
from streaming_fixtures import closer

from omlx.patches.expert_streaming.streaming_switch import (
    ExpertLRUCache,
    streaming_gate_state,
)
from omlx.patches.mlx_lm_mtp import batch_generator as bg


@pytest.fixture
def live_cache(closer):
    return closer(
        ExpertLRUCache(budget_bytes=1 << 20, per_expert_bytes=1024, num_layers=2)
    )


def test_gate_state_none_without_live_cache():
    assert streaming_gate_state() is None


def test_gate_state_aggregates_live_cache(live_cache):
    live_cache.stats.decode_hits = 7
    live_cache.stats.decode_misses = 3
    sig = streaming_gate_state()
    assert sig is not None and sig["active"] is True
    assert sig["capacity"] == live_cache.capacity
    assert sig["decode_hit_rate"] == pytest.approx(0.7)


def test_gate_state_ignores_closed_cache(live_cache):
    live_cache.close()
    assert streaming_gate_state() is None


def test_gate_state_ignores_unbounded_cache(closer):
    cache = closer(
        ExpertLRUCache(budget_bytes=0, per_expert_bytes=1024, num_layers=2)
    )
    assert cache.capacity == 0
    assert streaming_gate_state() is None


def test_gate_state_dead_cache_does_not_linger():
    # The admission worker holds only a weakref to its cache, so a dead
    # engine's cache is collected and the WeakSet drops it immediately.
    cache = ExpertLRUCache(budget_bytes=1 << 20, per_expert_bytes=1024, num_layers=2)
    del cache
    gc.collect()
    assert streaming_gate_state() is None


def test_mtp_gate_modes(monkeypatch, live_cache):
    monkeypatch.setattr(bg, "_MTP_STREAM_GATE", "auto")
    mode, sig = bg._mtp_streaming_gate()
    assert mode == "auto" and sig is not None

    monkeypatch.setattr(bg, "_MTP_STREAM_GATE", "off")
    mode, sig = bg._mtp_streaming_gate()
    assert mode == "off" and sig is not None

    monkeypatch.setattr(bg, "_MTP_STREAM_GATE", "on")
    mode, sig = bg._mtp_streaming_gate()
    assert mode == "on" and sig is not None  # signal present but unconstraining


def test_mtp_gate_no_streaming_never_constrains(monkeypatch):
    for mode in ("auto", "off", "on", "bogus"):
        monkeypatch.setattr(bg, "_MTP_STREAM_GATE", mode)
        got, sig = bg._mtp_streaming_gate()
        assert got == "on" and sig is None


def _controller():
    """DepthController primed for the cheap-entry regime: one shallow
    speculation pick plus a full warmup baseline, exit streak at the
    module default."""
    c = bg._DepthController(3)
    c.cur = 1
    c._warmup = [0, 0, 0]
    c.EXIT_STREAK = bg._MTP_STREAM_EXIT_STREAK
    return c


def test_cheap_entry_parks_losing_speculation_fast():
    # The auto-gate's controller mutation: depth-1 once + 3 baseline
    # samples, then the short exit streak parks losing speculation — the
    # behavior that replaces the full depth sweep under SSD streaming.
    c = _controller()

    picks = []
    cycles = 0
    # Streaming regime: verify positions pay real expert reads (~2x cycle
    # cost) and drafts churn — every speculative depth loses to the plain
    # step once the warmup baseline is measured.
    ms = {0: 400.0, 1: 800.0, 2: 1200.0, 3: 1600.0}
    while not c.should_exit() and cycles < 40:
        d = c.cur
        picks.append(d)
        c.observe(d, 0, ms[d])
        cycles += 1
    assert c.should_exit()
    assert cycles <= 4 + bg._MTP_STREAM_EXIT_STREAK
    assert picks[0] == 1  # one shallow speculation sample, not the deep sweep
    assert all(d == 0 for d in picks[1:])  # then parked at the plain step


def test_cheap_entry_keeps_winning_speculation():
    c = _controller()
    # RAM-served verify: depth-1 cycles are barely costlier than plain steps
    # and acceptance is high — the gate must not park a winning MTP.
    for _ in range(40):
        d = c.cur
        c.observe(d, d, {0: 400.0, 1: 405.0, 2: 410.0, 3: 415.0}[d])
    assert not c.should_exit()
