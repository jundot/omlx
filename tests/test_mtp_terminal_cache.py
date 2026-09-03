# SPDX-License-Identifier: Apache-2.0
"""MTP terminal-cache ownership and exact resident handoff tests."""

from types import SimpleNamespace

import mlx.core as mx

from omlx.patches.mlx_lm_mtp import batch_generator as bg


class _Response(SimpleNamespace):
    pass


class _Batch(SimpleNamespace):
    Response = _Response

    def extract_cache(self, _index):
        return self.prompt_cache

    def filter(self, keep):
        if not keep:
            self.uids = []


class _OffsetCache:
    def __init__(self, offset):
        self.offset = offset
        self.rollback_state = None
        self._mtp_draft_stash = None
        self._mtp_undo = None
        self._undo = None


def _batch(cache, *, uid=9):
    return _Batch(
        model=SimpleNamespace(model_type="qwen3_5"),
        uids=[uid],
        tokens=[[101]],
        _num_tokens=[0],
        max_tokens=[1],
        state_machines=[SimpleNamespace(match=lambda *_: (None, None, None))],
        _matcher_states=[None],
        _omlx_mtp_state=SimpleNamespace(uid=uid),
        prompt_cache=cache,
    )


def test_unproved_mtp_terminal_never_replays_full_history(monkeypatch):
    calls = []

    def reconcile(batch, state):
        calls.append((batch, state))
        raise AssertionError("completed requests must not replay their prompt")

    monkeypatch.setattr(bg, "_reconcile_mtp_to_standard", reconcile)
    batch = _batch(["reconciled-cache"])
    response = bg._emit_response(
        batch,
        token_id=7,
        logprobs_1d=mx.zeros((8,), dtype=mx.float32),
    )[0]

    assert calls == []
    assert response.prompt_cache is None
    assert response.all_tokens is None
    assert not hasattr(response, "_omlx_mtp_standard_terminal_exact")


def test_exact_qwen35_terminal_skips_full_replay(monkeypatch):
    def unexpected_reconcile(*_args):
        raise AssertionError("exact target cache must not replay the prompt")

    monkeypatch.setattr(bg, "_reconcile_mtp_to_standard", unexpected_reconcile)
    cache = [_OffsetCache(2)]
    batch = _batch(cache, uid=10)
    response = bg._emit_response(
        batch,
        token_id=7,
        logprobs_1d=mx.zeros((8,), dtype=mx.float32),
    )[0]

    assert response.prompt_cache is cache
    assert response._omlx_mtp_standard_terminal_exact is True
