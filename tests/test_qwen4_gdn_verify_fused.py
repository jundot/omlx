"""Row-batched Qwen4 GDN verify kernels vs the composed stock ops."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches import qwen35_gdn_prework as prework_mod

HK, HV, DK, DV, KS = 16, 48, 128, 128, 4
C = 2 * HK * DK + HV * DV  # 10240


def _stock_prework(qkv, conv_state, conv_w, b, a, A_log, dt_bias):
    from mlx_vlm.models.qwen3_5.gated_delta import _compute_g_beta

    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    next_state = mx.contiguous(conv_input[:, -(KS - 1) :, :])
    conv = nn.Conv1d(C, C, kernel_size=KS, groups=C, padding=0, bias=False)
    conv.weight = conv_w
    conv_out = nn.silu(conv(conv_input))
    q, k, v = mx.split(conv_out, [HK * DK, 2 * HK * DK], -1)
    T = qkv.shape[1]
    q = q.reshape(1, T, HK, DK)
    k = k.reshape(1, T, HK, DK)
    v = v.reshape(1, T, HV, DV)
    inv = DK**-0.5
    q = (inv * inv) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv * mx.fast.rms_norm(k, None, 1e-6)
    g, beta = _compute_g_beta(A_log, a, b, dt_bias)
    return q, k, v, next_state, g, beta


def _inputs(T, seed=0):
    ks = mx.random.split(mx.random.key(seed), 7)
    qkv = (mx.random.normal((1, T, C), key=ks[0]) * 0.5).astype(mx.bfloat16)
    conv_state = (mx.random.normal((1, KS - 1, C), key=ks[1]) * 0.5).astype(
        mx.bfloat16
    )
    conv_w = (mx.random.normal((C, KS, 1), key=ks[2]) * 0.3).astype(mx.bfloat16)
    b = mx.random.normal((1, T, HV), key=ks[3]).astype(mx.bfloat16)
    a = mx.random.normal((1, T, HV), key=ks[4]).astype(mx.bfloat16)
    A_log = (mx.random.normal((HV,), key=ks[5]) * 0.2).astype(mx.bfloat16)
    dt_bias = (mx.random.normal((HV,), key=ks[6]) * 0.2).astype(mx.bfloat16)
    return qkv, conv_state, conv_w, b, a, A_log, dt_bias


@pytest.mark.parametrize("T", [1, 2, 3, 4, 8])
def test_verify_prework_matches_stock(T):
    qkv, conv_state, conv_w, b, a, A_log, dt_bias = _inputs(T)
    inv = DK**-0.5
    q_scale = mx.array(inv * inv, dtype=mx.bfloat16)
    k_scale = mx.array(inv, dtype=mx.bfloat16)
    got = prework_mod.qwen4_verify_prework_fused(
        qkv, conv_state, conv_w, q_scale, k_scale, b, a, A_log, dt_bias, HK, HV, DK, DV
    )
    exp = _stock_prework(qkv, conv_state, conv_w, b, a, A_log, dt_bias)
    mx.eval(*got, *exp)
    names = ["q", "k", "v", "conv_state", "g", "beta"]
    for name, x, y in zip(names, got, exp):
        assert x.shape == y.shape, name
        assert x.dtype == y.dtype, name
        assert mx.array_equal(x, y).item(), name


@pytest.mark.parametrize("T", [1, 2, 4, 8])
def test_verify_norm_gate_matches_module(T):
    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
    )

    apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpRMSNormGated

    ks = mx.random.split(mx.random.key(1), 3)
    y = mx.random.normal((1, T, HV, DV), key=ks[0]).astype(mx.bfloat16)
    z = mx.random.normal((1, T, HV, DV), key=ks[1]).astype(mx.bfloat16)
    norm = Qwen4ExpRMSNormGated(DV, eps=1e-6, activation="sigmoid")
    norm.weight = (mx.random.normal((DV,), key=ks[2]) * 0.1).astype(mx.bfloat16)
    exp = norm(y, z).reshape(1, T, HV * DV)
    got = prework_mod.qwen4_verify_norm_gate_fused(
        y, z, norm.weight, hv=HV, dv=DV, eps=1e-6
    )
    mx.eval(exp, got)
    assert got.shape == exp.shape
    assert mx.array_equal(exp, got).item()


def test_verify_rows_supported_is_one_to_eight():
    assert prework_mod.qwen4_verify_rows_supported(1)
    assert prework_mod.qwen4_verify_rows_supported(2)
    assert prework_mod.qwen4_verify_rows_supported(8)
    assert not prework_mod.qwen4_verify_rows_supported(0)
    assert not prework_mod.qwen4_verify_rows_supported(9)


def test_verify_recurrence_with_states_matches_public_helper():
    """Precomputed g/beta through the states kernel == gated_delta_update_with_states."""
    from mlx_vlm.models.qwen3_5.gated_delta import (
        _compute_g_beta,
        gated_delta_update_with_states,
    )

    T = 4
    qkv, conv_state, conv_w, b, a, A_log, dt_bias = _inputs(T, seed=3)
    q, k, v, _, g, beta = _stock_prework(qkv, conv_state, conv_w, b, a, A_log, dt_bias)
    state = (mx.random.normal((1, HV, DV, DK), key=mx.random.key(9)) * 0.01).astype(
        mx.float32
    )
    exp = gated_delta_update_with_states(
        q, k, v, a, b, A_log, dt_bias, state, None, use_kernel=True
    )
    got = prework_mod._qwen4_verify_recurrence_with_states(q, k, v, g, beta, state)
    mx.eval(*exp, *got)
    assert len(got) == 3
    for x, y in zip(got, exp):
        assert x.shape == y.shape
        assert mx.array_equal(x, y).item()


class _FakeCache:
    def __init__(self, conv_state, recurrent_state=None):
        self._store = {0: conv_state, 1: recurrent_state}
        self.lengths = None
        self.left_padding = None
        self.advance_calls = []

    def __getitem__(self, i):
        return self._store[i]

    def __setitem__(self, i, v):
        self._store[i] = v

    def advance(self, n):
        self.advance_calls.append(n)


def _fake_module():
    from types import SimpleNamespace

    return SimpleNamespace(
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
        conv_kernel_size=KS,
        training=False,
    )


def _arm_route(monkeypatch, q35, S, prework):
    monkeypatch.setattr(prework_mod, "_PATCHED", False)
    monkeypatch.setattr(prework_mod, "_QWEN4_VERIFY_ENGAGED_LOGGED", False)
    monkeypatch.setattr(q35.Qwen3_5GatedDeltaNet, "_omlx_gdn_prework_patched", False, raising=False)
    monkeypatch.setattr(prework_mod, "_qwen4_decode_dynamic_eligible", lambda *a, **k: False)
    monkeypatch.setattr(prework_mod, "_qwen4_verify_dynamic_eligible", lambda *a, **k: True)
    monkeypatch.setattr(
        q35,
        "_target_verify_linears",
        lambda *a, **k: (
            mx.zeros((1, S, C), dtype=mx.bfloat16),
            mx.zeros((1, S, HV * DV), dtype=mx.bfloat16),
            mx.zeros((1, S, HV), dtype=mx.bfloat16),
            mx.zeros((1, S, HV), dtype=mx.bfloat16),
        ),
    )
    monkeypatch.setattr(prework_mod, "qwen4_verify_prework_fused", prework)


@pytest.mark.parametrize("S", [2, 4])
def test_verify_route_commits_states_sink_and_advances_once(monkeypatch, S):
    q35 = pytest.importorskip("mlx_vlm.models.qwen3_5.language")
    cls = q35.Qwen3_5GatedDeltaNet
    old_conv = mx.zeros((1, 3, C), dtype=mx.bfloat16)
    old_recurrent = mx.zeros((1, HV, DV, DK), dtype=mx.float32)
    next_conv = mx.ones_like(old_conv)
    next_recurrent = mx.ones_like(old_recurrent)
    fused = mx.ones((1, S, 2560), dtype=mx.bfloat16)

    def stock(*args, **kwargs):
        raise AssertionError("eligible Qwen4 verify unexpectedly fell back")

    monkeypatch.setattr(cls, "__call__", stock, raising=False)
    _arm_route(
        monkeypatch, q35, S,
        lambda *a, **k: (None, None, None, next_conv, None, None),
    )
    monkeypatch.setattr(
        prework_mod,
        "_qwen4_verify_recurrence_with_states",
        lambda *a, **k: (None, next_recurrent, "states"),
    )
    monkeypatch.setattr(prework_mod, "qwen4_verify_norm_gate_fused", lambda *a, **k: fused)
    monkeypatch.setattr(q35, "_target_verify_linear", lambda *a: fused)
    assert prework_mod.apply_qwen35_gdn_prework_patch()

    cache = _FakeCache(old_conv, old_recurrent)
    sink = []
    result = cls.__call__(
        _fake_module(), mx.zeros((1, S, 2560), dtype=mx.bfloat16), cache=cache, gdn_sink=sink
    )
    assert result is fused
    assert cache[0] is next_conv
    assert cache[1] is next_recurrent
    assert cache.advance_calls == [S]
    assert len(sink) == 1 and len(sink[0]) == 12
    assert sink[0][7] is old_recurrent and sink[0][11] == "states"


def test_verify_route_restores_states_and_sink_before_stock_fallback(monkeypatch):
    q35 = pytest.importorskip("mlx_vlm.models.qwen3_5.language")
    cls = q35.Qwen3_5GatedDeltaNet
    S = 2
    old_conv = mx.zeros((1, 3, C), dtype=mx.bfloat16)
    old_recurrent = mx.zeros((1, HV, DV, DK), dtype=mx.float32)
    calls = []

    def stock(self, inputs, mask=None, cache=None, gdn_sink=None, target_verify=False):
        calls.append((cache[0], cache[1], len(gdn_sink)))
        return "stock"

    def boom(*a, **k):
        raise RuntimeError("kernel failed")

    monkeypatch.setattr(cls, "__call__", stock, raising=False)
    _arm_route(monkeypatch, q35, S, boom)
    assert prework_mod.apply_qwen35_gdn_prework_patch()

    cache = _FakeCache(old_conv, old_recurrent)
    sink = ["earlier-layer"]
    module = _fake_module()
    x = mx.zeros((1, S, 2560), dtype=mx.bfloat16)
    assert cls.__call__(module, x, cache=cache, gdn_sink=sink) == "stock"
    assert cls.__call__(module, x, cache=cache, gdn_sink=sink) == "stock"
    assert calls == [(old_conv, old_recurrent, 1), (old_conv, old_recurrent, 1)]
    assert cache.advance_calls == []


def test_verify_beta_matches_prebuilt_sigmoid_at_fast_math_outlier():
    """mx.sigmoid is a prebuilt precise-exp op; the JIT fast exp disagrees at b=-6.84375."""
    T = 2
    qkv, conv_state, conv_w, b, a, A_log, dt_bias = _inputs(T, seed=5)
    b = mx.full((1, T, HV), -6.84375, dtype=mx.bfloat16)
    inv = DK**-0.5
    got = prework_mod.qwen4_verify_prework_fused(
        qkv, conv_state, conv_w, mx.array(inv * inv, dtype=mx.bfloat16),
        mx.array(inv, dtype=mx.bfloat16), b, a, A_log, dt_bias, HK, HV, DK, DV,
    )
    beta = got[5]
    exp = mx.sigmoid(b)
    mx.eval(beta, exp)
    assert mx.array_equal(beta, exp).item()
