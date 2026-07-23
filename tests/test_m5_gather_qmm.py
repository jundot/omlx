# SPDX-License-Identifier: Apache-2.0
"""Tests for the M5 sorted gather_qmm reroute (issue #2267)."""

from __future__ import annotations

import mlx.core as mx
import pytest

import omlx.patches.m5_gather_qmm as patch_mod
from omlx.patches.m5_gather_qmm import apply_m5_gather_qmm_workaround


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    """Start each test unwrapped and restore the session state after.

    Restoration pins everything to the raw builtin captured at setup —
    monkeypatched stand-ins (e.g. call spies on ``_original_gather_qmm``)
    must never leak into ``mx.gather_qmm`` for later test files. The
    reinstall bypasses ``apply`` so a kill-switch env var set by the
    test cannot leave the session unwrapped.
    """
    monkeypatch.delenv("OMLX_M5_GATHER_QMM_FIX", raising=False)
    was_installed = getattr(mx.gather_qmm, "_omlx_m5_reroute", False)
    raw = patch_mod._original_gather_qmm if was_installed else mx.gather_qmm
    saved_defective = patch_mod._defective
    if was_installed:
        mx.gather_qmm = raw
    yield
    mx.gather_qmm = raw
    patch_mod._original_gather_qmm = raw
    patch_mod._defective = saved_defective
    if was_installed:
        mx.gather_qmm = patch_mod._gather_qmm_rerouted


def test_apply_idempotent():
    assert apply_m5_gather_qmm_workaround()
    assert getattr(mx.gather_qmm, "_omlx_m5_reroute", False)
    assert not apply_m5_gather_qmm_workaround()


def test_env_kill_switch(monkeypatch):
    monkeypatch.setenv("OMLX_M5_GATHER_QMM_FIX", "0")
    assert not apply_m5_gather_qmm_workaround()
    assert not getattr(mx.gather_qmm, "_omlx_m5_reroute", False)


def _call(x_shape, **kwargs):
    x = mx.zeros(x_shape, dtype=mx.bfloat16)
    return patch_mod._needs_reroute(x, (), kwargs)


def test_needs_reroute_conditions():
    rows = mx.zeros((80,), dtype=mx.uint32)
    big = mx.zeros((32769,), dtype=mx.uint32)

    # K % 64 != 0 on the sorted rhs path triggers.
    assert _call((80, 1, 96), rhs_indices=rows, sorted_indices=True)
    # Aligned K with a small row count stays on the fast path.
    assert not _call((80, 1, 128), rhs_indices=rows, sorted_indices=True)
    # ml-explore/mlx#3856: row counts above 32768 trigger even aligned.
    assert _call((32769, 1, 128), rhs_indices=big, sorted_indices=True)
    # Unsorted calls never reroute.
    assert not _call((80, 1, 96), rhs_indices=rows, sorted_indices=False)
    assert not _call((80, 1, 96), rhs_indices=rows)
    # lhs-gather and non-transposed calls never select the rhs kernel.
    assert not _call(
        (80, 1, 96), lhs_indices=rows, rhs_indices=rows, sorted_indices=True
    )
    assert not _call(
        (80, 1, 96), rhs_indices=rows, transpose=False, sorted_indices=True
    )
    assert not _call((80, 1, 96), sorted_indices=True)


def test_wrapper_drops_sorted_flag_only_when_defective(monkeypatch):
    captured = {}

    def spy(x, w, *args, **kwargs):
        captured.update(kwargs)
        return mx.zeros((1,))

    assert apply_m5_gather_qmm_workaround()
    monkeypatch.setattr(patch_mod, "_original_gather_qmm", spy)

    x = mx.zeros((80, 1, 96), dtype=mx.bfloat16)
    idx = mx.zeros((80,), dtype=mx.uint32)

    monkeypatch.setattr(patch_mod, "_defective", True)
    mx.gather_qmm(x, x, x, rhs_indices=idx, sorted_indices=True)
    assert captured["sorted_indices"] is False

    monkeypatch.setattr(patch_mod, "_defective", False)
    mx.gather_qmm(x, x, x, rhs_indices=idx, sorted_indices=True)
    assert captured["sorted_indices"] is True


def _kernel_defective_here() -> bool:
    if not mx.metal.is_available():
        return False
    raw = mx.gather_qmm
    if getattr(raw, "_omlx_m5_reroute", False):
        raw = patch_mod._original_gather_qmm
    saved_orig, saved_flag = patch_mod._original_gather_qmm, patch_mod._defective
    patch_mod._original_gather_qmm = raw
    patch_mod._defective = None
    try:
        return patch_mod._sorted_gather_qmm_defective()
    finally:
        patch_mod._original_gather_qmm = saved_orig
        patch_mod._defective = saved_flag


@pytest.mark.skipif(
    not _kernel_defective_here(),
    reason="sorted gather_qmm NAX kernel is healthy on this machine",
)
def test_reroute_restores_correct_output_on_defective_hardware():
    """On affected hardware the patched call matches the fp32 reference."""
    assert apply_m5_gather_qmm_workaround()

    n, e, out_dim, k = 80, 8, 64, 96
    keys = mx.random.split(mx.random.key(1), 3)
    w = mx.random.normal((e, out_dim, k), key=keys[0]).astype(mx.bfloat16)
    wq, scales, biases = mx.quantize(w, group_size=32, bits=4)
    x = (mx.random.normal((n, 1, k), key=keys[1]) * 0.5).astype(mx.bfloat16)
    idx = mx.sort(mx.random.randint(0, e, (n,), key=keys[2]).astype(mx.uint32))
    wd = mx.dequantize(wq, scales, biases, group_size=32, bits=4)
    ref = x.astype(mx.float32) @ wd[idx].swapaxes(-1, -2).astype(mx.float32)

    out = mx.gather_qmm(
        x,
        wq,
        scales,
        biases,
        rhs_indices=idx,
        transpose=True,
        group_size=32,
        bits=4,
        sorted_indices=True,
    )
    err = mx.abs(out.astype(mx.float32) - ref).max().item()
    assert err < 0.2, f"still corrupt through the reroute: max err {err}"
