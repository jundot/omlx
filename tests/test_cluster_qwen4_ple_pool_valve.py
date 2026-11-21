# SPDX-License-Identifier: Apache-2.0
"""Fake-driven checks for the B3 PLE prefill pool valve
(``_release_ple_prefill_pool`` in the vendored qwen4_exp language model).

The valve is the measured fix for the table-stage growth: release the Metal
buffer pool at every PLE-streaming prefill chunk boundary — the measured
intervention cleared at EVERY boundary and won on both peak and wall clock.
A pool-vs-live pressure gate was tried and suppressed the valve (the KV fill
grows with the pool), so the valve is unconditional within its containment.
These tests pin that containment with monkeypatched counters. No model, no
real prefill.
"""

from __future__ import annotations

import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

compat.apply_mlx_vlm_qwen4_exp_compat_patch()

from mlx_vlm.models.qwen4_exp import language  # noqa: E402

GIB = 1024**3


@pytest.fixture()
def valve(monkeypatch):
    calls = []
    monkeypatch.setattr(language.mx, "clear_cache", lambda: calls.append(1))
    state = {"mode": "mmap"}
    monkeypatch.setattr(language, "_PLE_RUNTIME_MODE", state["mode"])

    def run(tokens, batch=1, mode=None):
        if mode is not None:
            monkeypatch.setattr(language, "_PLE_RUNTIME_MODE", mode)
        calls.clear()
        inputs = language.mx.zeros((batch, tokens), dtype=language.mx.int32)
        language._release_ple_prefill_pool(inputs)
        return len(calls)

    return run


def test_ple_prefill_chunk_clears(valve):
    assert valve(1024) == 1


def test_clears_even_when_the_pool_is_below_live(valve, monkeypatch):
    # The regression that killed the pressure gate: after a real chunk the
    # pool (0.72 GiB) is below the live non-weight arrays (1.88 GiB) because
    # the KV fill grows with the pool. The measured intervention clears at
    # every boundary regardless — the valve must too.
    monkeypatch.setattr(language.mx, "get_cache_memory", lambda: int(0.72 * GIB))
    monkeypatch.setattr(language.mx, "get_active_memory", lambda: int(36.29 * GIB))
    assert valve(1024) == 1


def test_resident_mode_never_clears(valve):
    assert valve(1024, mode="resident") == 0


def test_decode_step_never_clears(valve):
    assert valve(1) == 0


def test_mtp_verify_window_never_clears(valve):
    assert valve(16) == 0


def test_batched_prefill_is_out_of_scope(valve):
    assert valve(1024, batch=2) == 0


def test_the_64_token_boundary(valve):
    assert valve(64) == 1
    assert valve(63) == 0


def test_none_inputs_are_safe(valve):
    language._release_ple_prefill_pool(None)  # must not raise