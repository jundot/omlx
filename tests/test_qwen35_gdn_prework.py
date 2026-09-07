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
from omlx.patches.qwen35_gdn_prework import (
    gdn_prework_fused,
    qwen4_decode_norm_gate_fused,
    qwen4_decode_prework_fused,
)

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


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_qwen4_decode_prework_is_bit_exact_including_fp32_gate():
    from mlx_vlm.models.qwen3_5.gated_delta import _compute_g_beta

    mx.random.seed(29)
    conv_w = (mx.random.normal((C, 4, 1)) * 0.2).astype(mx.bfloat16)
    conv1d = nn.Conv1d(C, C, kernel_size=4, groups=C, bias=False)
    conv1d.weight = conv_w
    qkv = (mx.random.normal((1, 1, C)) * 0.5).astype(mx.bfloat16)
    state = (mx.random.normal((1, 3, C)) * 0.5).astype(mx.bfloat16)
    a = (mx.random.normal((1, 1, HV)) * 0.2).astype(mx.bfloat16)
    b = (mx.random.normal((1, 1, HV)) * 0.2).astype(mx.bfloat16)
    A_log = (mx.random.normal((HV,)) * 0.2).astype(mx.bfloat16)
    dt_bias = (mx.random.normal((HV,)) * 0.2).astype(mx.bfloat16)
    inv = DK**-0.5
    q_scale = mx.array(inv * inv, dtype=mx.bfloat16)
    k_scale = mx.array(inv, dtype=mx.bfloat16)

    q, k, v, next_state = _composed(qkv, state, conv1d)
    g, beta = _compute_g_beta(A_log, a, b, dt_bias)
    reference = (q, k, v, next_state, g, beta)
    actual = qwen4_decode_prework_fused(
        qkv,
        state,
        conv_w,
        q_scale,
        k_scale,
        b,
        a,
        A_log,
        dt_bias,
        HK,
        HV,
        DK,
        DV,
    )
    mx.eval(*reference, *actual)
    for name, expected, observed in zip(
        ("q", "k", "v", "conv_state", "g", "beta"),
        reference,
        actual,
    ):
        assert expected.dtype == observed.dtype, name
        assert mx.array_equal(expected, observed).item(), name


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_qwen4_decode_norm_gate_is_bit_exact():
    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
    )

    apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpRMSNormGated

    mx.random.seed(31)
    y = (mx.random.normal((1, 1, HV, DV)) * 0.25).astype(mx.bfloat16)
    z = (mx.random.normal((1, 1, HV, DV)) * 0.25).astype(mx.bfloat16)
    norm = Qwen4ExpRMSNormGated(DV, eps=1e-6, activation="sigmoid")
    norm.weight = (1 + mx.random.normal((DV,)) * 0.1).astype(mx.bfloat16)

    expected = norm(y, z).reshape(1, 1, HV * DV)
    observed = qwen4_decode_norm_gate_fused(
        y,
        z,
        norm.weight,
        hv=HV,
        dv=DV,
        eps=norm.eps,
    )
    mx.eval(expected, observed)
    assert mx.array_equal(expected, observed).item()


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


def test_qwen4_decode_dynamic_gate_is_strictly_b1_t1_nonverify(monkeypatch):
    monkeypatch.setattr(prework_mod, "_qwen4_decode_static_eligible", lambda _: True)
    module = object()
    inputs = mx.zeros((1, 1, 2560), dtype=mx.bfloat16)
    cache = _FakeCache(
        mx.zeros((1, 3, C), dtype=mx.bfloat16),
        mx.zeros((1, HV, DV, DK), dtype=mx.float32),
    )
    cache.left_padding = None

    def eligible(**changes):
        args = {
            "module": module,
            "inputs": inputs,
            "mask": None,
            "cache": cache,
            "gdn_sink": None,
            "target_verify": False,
        }
        args.update(changes)
        return prework_mod._qwen4_decode_dynamic_eligible(**args)

    assert eligible()
    assert not eligible(inputs=mx.zeros((2, 1, 2560), dtype=mx.bfloat16))
    assert not eligible(inputs=mx.zeros((1, 2, 2560), dtype=mx.bfloat16))
    assert not eligible(inputs=mx.zeros((1, 1, 2560), dtype=mx.float16))
    assert not eligible(mask="causal")
    assert not eligible(gdn_sink=[])
    assert not eligible(target_verify=True)

    cache.lengths = mx.array([1])
    assert not eligible()
    cache.lengths = None
    cache.left_padding = mx.array([0])
    assert not eligible()
    cache.left_padding = None
    cache[1] = mx.zeros((1, HV, DV, DK), dtype=mx.bfloat16)
    assert not eligible()


def _fake_quantized_linear(input_dims, output_dims, bits, group_size):
    linear = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
    nn.Module.__init__(linear)
    linear.bits = bits
    linear.group_size = group_size
    linear.mode = "affine"
    linear.weight = mx.zeros(
        (output_dims, input_dims * bits // 32),
        dtype=mx.uint32,
    )
    linear.scales = mx.zeros(
        (output_dims, input_dims // group_size),
        dtype=mx.bfloat16,
    )
    linear.biases = mx.zeros_like(linear.scales)
    return linear


@pytest.mark.parametrize(
    "signatures",
    [
        ((6, 64), (6, 64), (6, 64), (6, 64)),  # physical layer 0
        ((4, 64), (5, 128), (5, 128), (5, 128)),  # physical layer 1
        ((5, 64), (6, 64), (6, 64), (6, 64)),  # physical layer 29
    ],
)
def test_qwen4_decode_static_gate_accepts_canonical_oqe_allocations(signatures):
    module_type = type("Qwen4ExpGatedDeltaNet", (), {})
    module_type.__module__ = "mlx_vlm.models.qwen4_exp.language"
    module = module_type()
    module.training = False
    module.num_k_heads = HK
    module.num_v_heads = HV
    module.head_k_dim = DK
    module.head_v_dim = DV
    module.conv_kernel_size = 4
    module.conv1d = SimpleNamespace(
        weight=mx.zeros((C, 4, 1), dtype=mx.bfloat16),
        bias=None,
    )
    module.norm = SimpleNamespace(
        activation="sigmoid",
        weight=mx.ones((DV,), dtype=mx.bfloat16),
    )
    module.A_log = mx.zeros((HV,), dtype=mx.bfloat16)
    module.dt_bias = mx.zeros((HV,), dtype=mx.bfloat16)
    rows = (C, HV * DV, HV, HV)
    projections = [
        _fake_quantized_linear(2560, output, bits, group)
        for output, (bits, group) in zip(rows, signatures)
    ]
    (
        module.in_proj_qkv,
        module.in_proj_z,
        module.in_proj_b,
        module.in_proj_a,
    ) = projections
    module.out_proj = _fake_quantized_linear(6144, 2560, 5, 128)

    assert prework_mod._qwen4_decode_static_eligible(module)
    module.in_proj_z.group_size = 64 if module.in_proj_z.group_size == 128 else 128
    assert not prework_mod._qwen4_decode_static_eligible(module)


def test_qwen4_decode_route_commits_both_states_and_advances_once(monkeypatch):
    q35 = pytest.importorskip("mlx_vlm.models.qwen3_5.language")
    cls = q35.Qwen3_5GatedDeltaNet
    old_conv = mx.zeros((1, 3, C), dtype=mx.bfloat16)
    old_recurrent = mx.zeros((1, HV, DV, DK), dtype=mx.float32)
    next_conv = mx.ones_like(old_conv)
    next_recurrent = mx.ones_like(old_recurrent)
    fused = mx.ones((1, 1, 2560), dtype=mx.bfloat16)

    def stock(*args, **kwargs):
        raise AssertionError("eligible Qwen4 decode unexpectedly fell back")

    monkeypatch.setattr(prework_mod, "_PATCHED", False)
    monkeypatch.setattr(prework_mod, "_QWEN4_DECODE_ENGAGED_LOGGED", False)
    monkeypatch.setattr(cls, "__call__", stock, raising=False)
    monkeypatch.setattr(cls, "_omlx_gdn_prework_patched", False, raising=False)
    monkeypatch.setattr(
        prework_mod,
        "_qwen4_decode_dynamic_eligible",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        q35,
        "_target_verify_linears",
        lambda *args, **kwargs: (
            mx.zeros((1, 1, C), dtype=mx.bfloat16),
            mx.zeros((1, 1, HV * DV), dtype=mx.bfloat16),
            mx.zeros((1, 1, HV), dtype=mx.bfloat16),
            mx.zeros((1, 1, HV), dtype=mx.bfloat16),
        ),
    )
    monkeypatch.setattr(
        prework_mod,
        "qwen4_decode_prework_fused",
        lambda *args, **kwargs: (None, None, None, next_conv, None, None),
    )
    monkeypatch.setattr(
        prework_mod,
        "_qwen4_decode_recurrence",
        lambda *args, **kwargs: (None, next_recurrent),
    )
    monkeypatch.setattr(
        prework_mod,
        "qwen4_decode_norm_gate_fused",
        lambda *args, **kwargs: fused,
    )
    monkeypatch.setattr(q35, "_target_verify_linear", lambda *args: fused)

    assert prework_mod.apply_qwen35_gdn_prework_patch()
    module = SimpleNamespace(
        in_proj_qkv=None,
        in_proj_z=None,
        in_proj_b=None,
        in_proj_a=None,
        conv1d=SimpleNamespace(weight=None),
        head_k_dim=DK,
        head_v_dim=DV,
        num_k_heads=HK,
        num_v_heads=HV,
        A_log=None,
        dt_bias=None,
        norm=SimpleNamespace(weight=None, eps=1e-6),
        out_proj=None,
    )
    cache = _FakeCache(old_conv, old_recurrent)
    result = cls.__call__(
        module,
        mx.zeros((1, 1, 2560), dtype=mx.bfloat16),
        cache=cache,
    )
    assert result is fused
    assert cache[0] is next_conv
    assert cache[1] is next_recurrent
    assert cache.advance_calls == 1


def test_qwen4_decode_route_restores_states_before_stock_fallback(monkeypatch):
    q35 = pytest.importorskip("mlx_vlm.models.qwen3_5.language")
    cls = q35.Qwen3_5GatedDeltaNet
    old_conv = mx.zeros((1, 3, C), dtype=mx.bfloat16)
    old_recurrent = mx.zeros((1, HV, DV, DK), dtype=mx.float32)
    seen = []

    def stock(self, inputs, mask=None, cache=None, gdn_sink=None,
              target_verify=False):
        seen.append((cache[0], cache[1], cache.advance_calls))
        return "stock"

    monkeypatch.setattr(prework_mod, "_PATCHED", False)
    monkeypatch.setattr(cls, "__call__", stock, raising=False)
    monkeypatch.setattr(cls, "_omlx_gdn_prework_patched", False, raising=False)
    monkeypatch.setattr(
        prework_mod,
        "_qwen4_decode_dynamic_eligible",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        q35,
        "_target_verify_linears",
        lambda *args, **kwargs: (None, None, None, None),
    )
    monkeypatch.setattr(
        prework_mod,
        "qwen4_decode_prework_fused",
        lambda *args, **kwargs: (
            None,
            None,
            None,
            mx.ones_like(old_conv),
            None,
            None,
        ),
    )
    monkeypatch.setattr(
        prework_mod,
        "_qwen4_decode_recurrence",
        lambda *args, **kwargs: (None, mx.ones_like(old_recurrent)),
    )
    monkeypatch.setattr(
        prework_mod,
        "qwen4_decode_norm_gate_fused",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("late")),
    )

    assert prework_mod.apply_qwen35_gdn_prework_patch()
    module = SimpleNamespace(
        in_proj_qkv=None,
        in_proj_z=None,
        in_proj_b=None,
        in_proj_a=None,
        conv1d=SimpleNamespace(weight=None),
        head_k_dim=DK,
        head_v_dim=DV,
        num_k_heads=HK,
        num_v_heads=HV,
        A_log=None,
        dt_bias=None,
        norm=SimpleNamespace(weight=None, eps=1e-6),
        out_proj=None,
    )
    cache = _FakeCache(old_conv, old_recurrent)
    result = cls.__call__(
        module,
        mx.zeros((1, 1, 2560), dtype=mx.bfloat16),
        cache=cache,
    )
    assert result == "stock"
    assert len(seen) == 1
    assert seen[0][0] is old_conv
    assert seen[0][1] is old_recurrent
    assert seen[0][2] == 0


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
