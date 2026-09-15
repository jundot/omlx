# SPDX-License-Identifier: Apache-2.0
"""Warm/pin hook phase threading (audit): seq_len is the authoritative
decode/prefill signal — routed-row count alone conflates a short prefill
with a wide decode.

Also covers the PinController latch fix: a pin pass that cannot run
(empty regime / unknown expert width) must not latch ``pinned`` — it
would suppress every later attempt.
"""
from types import SimpleNamespace

from omlx.patches.expert_streaming.warmer import (
    PageCacheWarmer,
    PinController,
    PrefillHotnessRecorder,
    _is_decode_call,
)


def test_is_decode_call_seq_len_authoritative():
    # Real seq_len wins over the routed-row bound both ways.
    assert _is_decode_call(64, seq_len=1) is True   # wide decode
    assert _is_decode_call(8, seq_len=4) is False   # 4-token prefill
    assert _is_decode_call(200, seq_len=200) is False
    # Unknown seq_len: routed-row fallback (legacy/test wiring).
    assert _is_decode_call(64) is True
    assert _is_decode_call(65) is False


def _lin(layer=0):
    return SimpleNamespace(
        backing=None,
        stacked_weight_key=None,
        stacked_scales_key=None,
        stacked_biases_key=None,
    )


def test_warmer_gates_on_seq_len_not_rows():
    w = PageCacheWarmer({0: [_lin()], 1: [_lin()]})
    # seq_len=4 with only 8 routed rows: prefill — must not record or
    # fire readahead for the next layer.
    w.on_layer_plan(0, [1, 2, 3], 8, seq_len=4)
    assert w.last_uniq[0] == []
    w.on_layer_plan(0, [1, 2, 3], 8, seq_len=1)
    assert w.last_uniq[0] == [1, 2, 3]


def test_pin_controller_regime_by_seq_len():
    p = PinController({}, backing=None, observe_calls=1, num_experts=8)
    # 8 routed rows / seq_len=1 -> decode regime.
    p.on_layer_plan(0, [1, 2], 8, counts=None, seq_len=1)
    assert p.regimes["decode"][0].get(1) == 1
    assert not p.regimes["prefill"]
    # Same row count at seq_len=3 -> prefill regime (the old rows-only
    # gate filed this under decode).
    p.on_layer_plan(0, [3, 4], 8, counts=None, seq_len=3)
    assert p.regimes["prefill"][0].get(3) == 1
    assert p.regimes["decode"][0].get(3) is None


def test_pin_latch_not_set_when_nothing_to_pin():
    """An empty regime or unknown expert width must not latch pinned."""
    from collections import Counter

    p = PinController({}, backing=None, observe_calls=1, num_experts=8)
    # No frequencies recorded -> the pin pass has nothing to do; pinned
    # stays False so a later observed window can still fire it.
    p._pin_all(sync=True)
    assert p.pinned is False
    # Unknown expert width with a real regime also refuses to latch.
    p2 = PinController({}, backing=None, observe_calls=1, num_experts=8)
    p2.regimes["decode"][0] = Counter({1: 5})
    p2.per_expert_bytes = 0
    p2._pin_all(sync=True)
    assert p2.pinned is False


def test_recorder_seed_gates_on_seq_len():
    rec = PrefillHotnessRecorder({0: [_lin()]}, backing=None)
    # Prefill-shaped call accumulates; decode-shaped fires the seed.
    rec.on_layer_plan(0, [1, 2], 200, seq_len=25)
    assert rec.saw_prefill and rec.freq[0].get(1) == 1
    rec.maybe_seed(0, 8, seq_len=1)
    assert rec.seeded
    # A second recorder: decode-shaped calls never accumulate freq.
    rec2 = PrefillHotnessRecorder({0: [_lin()]}, backing=None)
    rec2.on_layer_plan(0, [1], 8, seq_len=1)
    assert not rec2.saw_prefill
    # ...and a multi-token call does not fire a pending seed early.
    rec3 = PrefillHotnessRecorder({0: [_lin()]}, backing=None)
    rec3.saw_prefill = True
    rec3.maybe_seed(0, 8, seq_len=4)  # rows say decode, seq_len says not
    assert not rec3.seeded


