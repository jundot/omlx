# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the fused Qwen3.5/3.6 GDN verify prework kernel.

The fused kernel must be BIT-exact to the composed chain (conv-state concat
+ depthwise conv1d + SiLU + split + ones-weight RMS norms + scalar scales +
next conv-state slice) at every verify width it claims (S in 3..9).
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches import qwen35_gdn_prework as prework_mod
from omlx.patches.qwen35_gdn_prework import gdn_prework_fused

HK, HV, DK, DV = 16, 48, 128, 128
C = 2 * HK * DK + HV * DV
KEY_DIM = HK * DK


def _composed(qkv, conv_state, conv1d):
    B, S, _ = qkv.shape
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    new_state = mx.contiguous(conv_input[:, -3:, :])
    co = nn.silu(conv1d(conv_input))
    q, k, v = mx.split(co, [KEY_DIM, 2 * KEY_DIM], -1)
    q = q.reshape(B, S, HK, DK)
    k = k.reshape(B, S, HK, DK)
    v = v.reshape(B, S, HV, DV)
    inv = DK**-0.5
    q = (inv**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv * mx.fast.rms_norm(k, None, 1e-6)
    return q, k, v, new_state


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("seq", [3, 4, 5, 7, 9])
def test_fused_prework_bit_exact(seq):
    mx.random.seed(11)
    conv_w = (mx.random.normal((C, 4, 1)) * 0.2).astype(mx.bfloat16)
    conv1d = nn.Conv1d(C, C, kernel_size=4, groups=C, bias=False)
    conv1d.weight = conv_w
    qkv = (mx.random.normal((1, seq, C)) * 0.5).astype(mx.bfloat16)
    state = (mx.random.normal((1, 3, C)) * 0.5).astype(mx.bfloat16)
    inv = DK**-0.5
    q_scale = mx.array(inv * inv, dtype=mx.bfloat16)
    k_scale = mx.array(inv, dtype=mx.bfloat16)

    ref = _composed(qkv, state, conv1d)
    got = gdn_prework_fused(qkv, state, conv_w, q_scale, k_scale, HK, HV, DK, DV)
    for name, r, g in zip(("q", "k", "v", "conv_state"), ref, got):
        assert r.shape == g.shape, name
        assert bool((r == g).all().item()), f"{name} not bit-exact at S={seq}"


class _FakeCache:
    """Minimal cache[0]/cache[1]/advance duck-type for patched_call."""

    def __init__(self, conv_state, recurrent_state=None):
        self._store = {0: conv_state, 1: recurrent_state}
        self.lengths = None
        self.advance_calls = 0

    def __getitem__(self, i):
        return self._store[i]

    def __setitem__(self, i, v):
        self._store[i] = v

    def advance(self, n):
        self.advance_calls += 1


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_patched_call_restores_conv_state_and_skips_advance_on_failure(monkeypatch):
    """E3 regression: an exception raised after cache[0] is set to the fused
    kernel's post-update state (but before the delta update finishes) must
    not leave that clobbered state behind for the stock fallback to see, and
    must not have advanced the cache offset yet (which would double-advance
    once the fallback runs its own single advance)."""
    q35 = pytest.importorskip("mlx_vlm.models.qwen3_5.language")
    cls = q35.Qwen3_5GatedDeltaNet

    original_conv_state = mx.full((1, 3, 4), 1.0, dtype=mx.bfloat16)
    new_conv_state = mx.full((1, 3, 4), 2.0, dtype=mx.bfloat16)

    orig_call_calls = []

    def fake_orig_call(self, inputs, mask=None, cache=None, gdn_sink=None,
                        target_verify=False):
        orig_call_calls.append(cache[0])
        return "stock-result"

    def fake_target_verify_linears(linears, inputs, flag):
        z = mx.zeros((1, inputs.shape[1], HK, DV))
        return mx.zeros((1, inputs.shape[1], 1)), z, None, None

    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(prework_mod, "_PATCHED", False)
    monkeypatch.setattr(prework_mod, "gdn_prework_fused",
                         lambda *a, **kw: (None, None, None, new_conv_state))
    monkeypatch.setattr(cls, "__call__", fake_orig_call, raising=False)
    monkeypatch.setattr(cls, "_omlx_gdn_prework_patched", False, raising=False)
    monkeypatch.setattr(q35, "_target_verify_linears", fake_target_verify_linears)
    monkeypatch.setattr(q35, "_gated_delta_update_verify_decode", _raise)

    assert prework_mod.apply_qwen35_gdn_prework_patch() is True
    patched_call = cls.__call__
    assert patched_call is not fake_orig_call  # actually installed the wrapper

    fake_self = SimpleNamespace(
        in_proj_qkv=None, in_proj_z=None, in_proj_b=None, in_proj_a=None,
        conv1d=SimpleNamespace(weight=mx.zeros((1,), dtype=mx.bfloat16), bias=None),
        head_k_dim=DK, head_v_dim=DV, num_k_heads=HK, num_v_heads=HV,
        A_log=None, dt_bias=None, training=False, conv_kernel_size=4,
    )
    cache = _FakeCache(original_conv_state)
    inputs = mx.zeros((1, 4, 1), dtype=mx.bfloat16)

    result = patched_call(fake_self, inputs, mask=None, cache=cache,
                           gdn_sink=[], target_verify=True)

    assert result == "stock-result"  # fell back to orig_call
    assert cache.advance_calls == 0  # never reached the (now-deferred) advance
    assert len(orig_call_calls) == 1
    assert bool((orig_call_calls[0] == original_conv_state).all().item()), (
        "orig_call must see the pre-mutation conv state, not the fused "
        "kernel's clobbered post-update state"
    )


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_patched_call_restores_state_and_discards_sink_on_late_failure(monkeypatch):
    """A failure AFTER the fused arm has appended its gdn_sink entry (e.g.
    in norm/out_proj) falls back to orig_call, and the stock path appends
    its own sink entry for the same layer call. The fused entry must be
    discarded, otherwise the sink holds two entries for one layer and every
    later layer's rollback capture is shifted by one."""
    q35 = pytest.importorskip("mlx_vlm.models.qwen3_5.language")
    cls = q35.Qwen3_5GatedDeltaNet

    original_conv_state = mx.full((1, 3, 4), 1.0, dtype=mx.bfloat16)
    new_conv_state = mx.full((1, 3, 4), 2.0, dtype=mx.bfloat16)
    original_recurrent_state = mx.full((1, 1), 3.0, dtype=mx.float32)
    new_recurrent_state = mx.full((1, 1), 4.0, dtype=mx.float32)
    stock_recurrent_states = []

    def fake_orig_call(self, inputs, mask=None, cache=None, gdn_sink=None,
                        target_verify=False):
        # Mirror the stock path: it appends its own rollback entry when a
        # sink is present.
        stock_recurrent_states.append(cache[1])
        gdn_sink.append("stock-entry")
        return "stock-result"

    def fake_target_verify_linears(linears, inputs, flag):
        z = mx.zeros((1, inputs.shape[1], HK, DV))
        # Last dim matches conv_state so the conv_input concatenate works.
        return mx.zeros((1, inputs.shape[1], 4), dtype=mx.bfloat16), z, None, None

    def fake_delta_update(*args, **kwargs):
        out = mx.zeros((1, 4, HV, DV), dtype=mx.bfloat16)
        return out, new_recurrent_state, None

    def _raise(*args, **kwargs):
        raise RuntimeError("boom in out_proj")

    monkeypatch.setattr(prework_mod, "_PATCHED", False)
    monkeypatch.setattr(prework_mod, "gdn_prework_fused",
                         lambda *a, **kw: (None, None, None, new_conv_state))
    monkeypatch.setattr(cls, "__call__", fake_orig_call, raising=False)
    monkeypatch.setattr(cls, "_omlx_gdn_prework_patched", False, raising=False)
    monkeypatch.setattr(q35, "_target_verify_linears", fake_target_verify_linears)
    monkeypatch.setattr(q35, "_gated_delta_update_verify_decode", fake_delta_update)
    monkeypatch.setattr(q35, "_target_verify_linear", _raise)

    assert prework_mod.apply_qwen35_gdn_prework_patch() is True
    patched_call = cls.__call__
    assert patched_call is not fake_orig_call

    fake_self = SimpleNamespace(
        in_proj_qkv=None, in_proj_z=None, in_proj_b=None, in_proj_a=None,
        conv1d=SimpleNamespace(weight=mx.zeros((1,), dtype=mx.bfloat16), bias=None),
        head_k_dim=DK, head_v_dim=DV, num_k_heads=HK, num_v_heads=HV,
        A_log=None, dt_bias=None, training=False, conv_kernel_size=4,
        norm=lambda out, z: out, out_proj=None,
    )
    cache = _FakeCache(original_conv_state, original_recurrent_state)
    inputs = mx.zeros((1, 4, 1), dtype=mx.bfloat16)
    sink = []

    result = patched_call(fake_self, inputs, mask=None, cache=cache,
                           gdn_sink=sink, target_verify=True)

    assert result == "stock-result"
    assert sink == ["stock-entry"], (
        f"sink must hold exactly the stock entry, got {len(sink)} entries: "
        "the fused arm's entry was not discarded on fallback"
    )
    assert len(stock_recurrent_states) == 1
    assert bool(
        (stock_recurrent_states[0] == original_recurrent_state).all().item()
    ), "stock fallback must see the pre-call recurrent state"
