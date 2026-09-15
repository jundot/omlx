# SPDX-License-Identifier: Apache-2.0
"""SpeculationState lifecycle: closed guards, staged-future cancellation,
non-consuming probes (stage_pending), and shared-future drop semantics.
"""
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from omlx.patches.expert_streaming import speculation as spec_mod
from omlx.patches.expert_streaming.speculation import SpeculationState


def _linear(layer=0, key_prefix="w"):
    return SimpleNamespace(
        layer_idx=layer,
        bundle_key=lambda e: (layer, int(e), key_prefix),
    )


def test_stage_pending_is_non_consuming():
    spec = SpeculationState()
    if not spec_mod._STAGED_ENV:
        pytest.skip("staged prefetch disabled by env")
    lin = _linear()
    fut: Future = Future()  # never completes
    n = spec.stage_register(fut, [1, 2], lin)
    assert n == 2
    key = lin.bundle_key(1)
    # Probe reports coverage without consuming or blocking.
    assert spec.stage_pending(key) is True
    assert spec.stage_pending(lin.bundle_key(99)) is False
    # Still registered afterwards — nothing was popped.
    assert key in spec.staged_futs


def test_stage_resolve_completed_row_consumes():
    spec = SpeculationState()
    if not spec_mod._STAGED_ENV:
        pytest.skip("staged prefetch disabled by env")
    lin = _linear()
    key = lin.bundle_key(7)
    spec.staged[key] = ("w7", "s7", None)
    assert spec.stage_pending(key) is True
    row = spec.stage_resolve(key)
    assert row == ("w7", "s7", None)
    assert spec.stage_pending(key) is False


def test_close_cancels_staged_futures():
    spec = SpeculationState()
    if not spec_mod._STAGED_ENV:
        pytest.skip("staged prefetch disabled by env")
    lin = _linear()
    pending: Future = Future()
    spec.stage_register(pending, [1, 2, 3], lin)
    spec.close()
    assert pending.cancelled() or pending.done()
    assert spec.staged_futs == {}
    assert spec.staged == {}
    # Idempotent.
    spec.close()


def test_closed_state_guards_mutators():
    spec = SpeculationState()
    spec.record_prev(0, [1, 2], positions=2)
    spec.register_linears(0, [_linear()])
    spec.close()
    clock = spec._advise_clock
    # Mutations after close are no-ops.
    spec.record_prev(0, [3, 4], positions=2)
    spec.register_linears(1, [_linear(1)])
    spec.bump("advised")
    spec.note_trans_extras(0, [9])
    assert spec._advise_clock == clock
    assert spec.prev_uniq_by_layer == {}
    assert 1 not in spec.linears_by_layer
    assert spec.stats.get("advised", 0) == 0
    assert spec._trans_pending == {}


def test_closed_state_serves_nothing():
    spec = SpeculationState()
    if not spec_mod._STAGED_ENV:
        pytest.skip("staged prefetch disabled by env")
    lin = _linear()
    key = lin.bundle_key(5)
    spec.staged[key] = ("w", "s", None)
    fut: Future = Future()
    spec.close()
    assert spec.stage_pending(key) is False
    assert spec.stage_resolve(key) is None
    assert spec.stage_room() is False
    assert spec.stage_register(fut, [1], lin) == 0
    assert not fut.cancelled()  # never registered -> never cancelled


def test_stage_drop_cancels_only_unreferenced_future():
    spec = SpeculationState()
    if not spec_mod._STAGED_ENV:
        pytest.skip("staged prefetch disabled by env")
    lin = _linear()
    shared: Future = Future()
    spec.stage_register(shared, [1, 2], lin)
    # Drop one key of a two-key registration: the shared future must
    # survive for the sibling's join.
    spec.stage_drop(lin.bundle_key(1))
    assert not shared.cancelled()
    assert spec.stage_pending(lin.bundle_key(2)) is True
    # Dropping the last referencing key cancels the read.
    spec.stage_drop(lin.bundle_key(2))
    assert shared.cancelled()


def test_close_mid_join_serves_only_requested_key():
    spec = SpeculationState()
    if not spec_mod._STAGED_ENV:
        pytest.skip("staged prefetch disabled by env")
    lin = _linear()

    class ClosingFuture:
        """result() closes the state, simulating close() landing between
        the staged_futs lookup and the post-join fan-out."""

        def done(self):
            return False

        def result(self, timeout=None):
            spec.close()
            return [("w1", "s1", None), ("w2", "s2", None)]

        def cancel(self):
            return False

    spec.stage_register(ClosingFuture(), [1, 2], lin)
    # The join found the future before close(); after close it must serve
    # this key's row without fanning rows back into the cleared dicts.
    assert spec.stage_resolve(lin.bundle_key(1)) == ("w1", "s1", None)
    assert spec.staged == {}
    assert spec.staged_futs == {}
    # A fully-post-close resolve serves nothing (entries are gone).
    assert spec.stage_resolve(lin.bundle_key(2)) is None
