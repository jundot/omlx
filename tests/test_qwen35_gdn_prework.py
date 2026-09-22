# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the fused Qwen3.5/3.6 GDN verify prework kernel.

The fused kernel must be BIT-exact to the composed chain (conv-state concat
+ depthwise conv1d + SiLU + split + ones-weight RMS norms + scalar scales +
next conv-state slice) at every verify width it claims (S in 3..9).
"""

from __future__ import annotations

from types import ModuleType, SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_vlm.models.qwen3_5 import language
from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward
from mlx_vlm.speculative.cache_state import start_speculative_cache
from mlx_vlm.speculative.ops import linear as linear_ops


@pytest.fixture(autouse=True)
def restore_hooks(monkeypatch):
    cls = language.Qwen3_5GatedDeltaNet
    monkeypatch.setattr(cls, "__call__", cls.__call__)
    verifier = Qwen3_5BatchInvariantForward
    monkeypatch.setattr(verifier, "_gated_delta", verifier._gated_delta)


from omlx.patches import qwen35_gdn_prework as prework_mod
from omlx.patches.qwen35_gdn_prework import (
    gdn_prework_fused,
    qwen4_decode_norm_gate_fused,
    qwen4_decode_prework_fused,
)
from omlx.patches.qwen35_q4_mlp import _VLMQuantizedPrefillLinear

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
@pytest.mark.parametrize("seq", [2, 3, 4, 5, 7, 9])
@pytest.mark.parametrize("batch", [1, 2, 4])
def test_fused_prework_bit_exact(seq, batch):
    mx.random.seed(11)
    conv_w = (mx.random.normal((C, 4, 1)) * 0.2).astype(mx.bfloat16)
    conv1d = nn.Conv1d(C, C, kernel_size=4, groups=C, bias=False)
    conv1d.weight = conv_w
    qkv = (mx.random.normal((batch, seq, C)) * 0.5).astype(mx.bfloat16)
    state = (mx.random.normal((batch, 3, C)) * 0.5).astype(mx.bfloat16)
    inv = DK**-0.5
    q_scale = mx.array(inv * inv, dtype=mx.bfloat16)
    k_scale = mx.array(inv, dtype=mx.bfloat16)

    ref = _composed(qkv, state, conv1d)
    got = gdn_prework_fused(qkv, state, conv_w, q_scale, k_scale, HK, HV, DK, DV)
    for name, r, g in zip(("q", "k", "v", "conv_state"), ref, got):
        assert r.shape == g.shape, name
        assert bool((r == g).all().item()), f"{name} not bit-exact at S={seq}"


def _composed_l2(qkv, conv_state, conv1d):
    """Stock Qwen4 chain: same prework, Qwen4 L2 q/k normalization."""
    batch, seq, _ = qkv.shape
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    new_state = mx.contiguous(conv_input[:, -3:, :])
    co = nn.silu(conv1d(conv_input))
    q, k, v = mx.split(co, [KEY_DIM, 2 * KEY_DIM], -1)
    q = q.reshape(batch, seq, HK, DK)
    k = k.reshape(batch, seq, HK, DK)
    v = v.reshape(batch, seq, HV, DV)
    q = q * mx.rsqrt(mx.sum(mx.square(q), axis=-1, keepdims=True) + 1e-6)
    k = k * mx.rsqrt(mx.sum(mx.square(k), axis=-1, keepdims=True) + 1e-6)
    return q * (DK**-0.5), k, v, new_state


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("seq", [2, 3, 4, 5, 7, 9])
@pytest.mark.parametrize("batch", [1, 2, 4])
def test_fused_prework_l2_bit_exact(seq, batch):
    mx.random.seed(41)
    conv_w = (mx.random.normal((C, 4, 1)) * 0.2).astype(mx.bfloat16)
    conv1d = nn.Conv1d(C, C, kernel_size=4, groups=C, bias=False)
    conv1d.weight = conv_w
    qkv = (mx.random.normal((batch, seq, C)) * 0.5).astype(mx.bfloat16)
    state = (mx.random.normal((batch, 3, C)) * 0.5).astype(mx.bfloat16)
    inv = DK**-0.5
    q_scale = mx.array(inv, dtype=mx.bfloat16)
    k_scale = mx.array(1.0, dtype=mx.bfloat16)

    ref = _composed_l2(qkv, state, conv1d)
    got = gdn_prework_fused(
        qkv, state, conv_w, q_scale, k_scale, HK, HV, DK, DV, l2=True
    )
    for name, r, g in zip(("q", "k", "v", "conv_state"), ref, got):
        assert r.shape == g.shape, name
        assert bool((r == g).all().item()), f"{name} not bit-exact at S={seq}"


def test_verify_gate_routes_qwen4_l2_norm(monkeypatch):
    q4 = pytest.importorskip("mlx_vlm.models.qwen4_exp.language")
    from mlx_vlm.models.cache import ArraysCache
    from mlx_vlm.models.qwen3_5 import language as q35
    from mlx_vlm.speculative.cache_state import start_speculative_cache

    monkeypatch.setattr(prework_mod, "_PATCHED", False)
    assert prework_mod.apply_qwen35_gdn_prework_patch()
    # Mirror the patch's runtime resolution: compat vendor wins when
    # installed, upstream mlx-vlm otherwise.
    ver_cls = getattr(q4, "_Qwen4Verifier", None) or getattr(
        q4, "Qwen4ExpBatchInvariantForward", None
    )
    assert ver_cls is not None

    args = SimpleNamespace(
        hidden_size=64,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
    )
    mx.random.seed(77)
    module = q35.Qwen3_5GatedDeltaNet(args)
    # The gate pins the Qwen4 L2 site by layer identity; graft it onto the
    # q35 test module (same shape, same _normalize_qk function object).
    module.__class__ = type(
        "Q4GatedDeltaNet",
        (q35.Qwen3_5GatedDeltaNet,),
        {"_normalize_qk": q4.Qwen4ExpGatedDeltaNet._normalize_qk},
    )
    module.set_dtype(mx.bfloat16)
    module.eval()
    inputs = mx.random.normal((2, 4, 64)).astype(mx.bfloat16)

    seen = []
    kernel = prework_mod.gdn_prework_fused

    def record(*call_args, **call_kwargs):
        seen.append(call_kwargs.get("l2"))
        return kernel(*call_args, **call_kwargs)

    monkeypatch.setattr(prework_mod, "gdn_prework_fused", record)

    cache = ArraysCache(size=2)
    cache[0] = mx.random.normal((2, 3, module.conv_dim)).astype(mx.bfloat16)
    cache[1] = mx.random.normal((2, 4, 128, 128)) * 0.01
    transaction = start_speculative_cache([cache], 4)
    ver_cls()._gated_delta(module, inputs, None, cache)
    mx.eval(cache.state)
    assert seen == [True]
    transaction.abort()

    seen.clear()
    cache2 = ArraysCache(size=2)
    cache2[0] = mx.random.normal((2, 3, module.conv_dim)).astype(mx.bfloat16)
    cache2[1] = mx.random.normal((2, 4, 128, 128)) * 0.01
    transaction2 = start_speculative_cache([cache2], 4)
    Qwen3_5BatchInvariantForward()._gated_delta(module, inputs, None, cache2)
    mx.eval(cache2.state)
    assert seen == [False]
    transaction2.abort()


def test_prework_patch_does_not_import_qwen4_exp(monkeypatch):
    """The patch runs at every VLM start. Importing qwen4_exp there would
    pin the upstream module before the compat vendor registers its own, and
    a later Qwen4 load would fail on the vendor-only runtime symbols.
    """
    import sys

    for name in [n for n in sys.modules if n.startswith("mlx_vlm.models.qwen4_exp")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(prework_mod, "_PATCHED", False)
    assert prework_mod.apply_qwen35_gdn_prework_patch()
    assert not [n for n in sys.modules if n.startswith("mlx_vlm.models.qwen4_exp")]


@pytest.mark.parametrize("vendor_registered_first", [True, False])
def test_verify_gate_resolves_compat_vendor_verifier(
    monkeypatch, vendor_registered_first
):
    """Regression: the compat vendor inserts its qwen4_exp module at
    __path__[0], whose verifier is ``_Qwen4Verifier`` (no
    ``Qwen4ExpBatchInvariantForward``). The gate must resolve it and
    engage the L2 variant, pinned to the layer's ``_normalize_qk`` site,
    whether the vendor registered before the patch (Qwen4 loaded first) or
    after it (another VLM started first).
    """
    import sys

    from mlx_vlm.models.cache import ArraysCache

    pytest.importorskip("mlx_vlm.models.qwen4_exp.language")
    if not vendor_registered_first:
        monkeypatch.setattr(prework_mod, "_PATCHED", False)
        assert prework_mod.apply_qwen35_gdn_prework_patch()

    class VendorGDN(language.Qwen3_5GatedDeltaNet):
        @staticmethod
        def _normalize_qk(q, k):
            scale = q.shape[-1] ** -0.5
            q = q * mx.rsqrt(mx.sum(mx.square(q), axis=-1, keepdims=True) + 1e-6)
            k = k * mx.rsqrt(mx.sum(mx.square(k), axis=-1, keepdims=True) + 1e-6)
            return q * scale, k

    class VendorVerifier(Qwen3_5BatchInvariantForward):
        @staticmethod
        def _normalize_gated_delta_qk(layer, q, k):
            return layer._normalize_qk(q, k)

    fake = ModuleType("mlx_vlm.models.qwen4_exp.language")
    fake._Qwen4Verifier = VendorVerifier
    fake.Qwen4ExpGatedDeltaNet = VendorGDN
    monkeypatch.setitem(sys.modules, "mlx_vlm.models.qwen4_exp.language", fake)
    import mlx_vlm.models.qwen4_exp as q4_pkg

    monkeypatch.setattr(q4_pkg, "language", fake, raising=False)

    if vendor_registered_first:
        monkeypatch.setattr(prework_mod, "_PATCHED", False)
        assert prework_mod.apply_qwen35_gdn_prework_patch()

    args = SimpleNamespace(
        hidden_size=64,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
    )
    mx.random.seed(79)
    module = VendorGDN(args)
    module.set_dtype(mx.bfloat16)
    module.eval()
    inputs = mx.random.normal((2, 4, 64)).astype(mx.bfloat16)

    seen = []
    kernel = prework_mod.gdn_prework_fused

    def record(*call_args, **call_kwargs):
        seen.append(call_kwargs.get("l2"))
        return kernel(*call_args, **call_kwargs)

    monkeypatch.setattr(prework_mod, "gdn_prework_fused", record)

    cache = ArraysCache(size=2)
    cache[0] = mx.random.normal((2, 3, module.conv_dim)).astype(mx.bfloat16)
    cache[1] = mx.random.normal((2, 4, 128, 128)) * 0.01
    transaction = start_speculative_cache([cache], 4)
    VendorVerifier()._gated_delta(module, inputs, None, cache)
    mx.eval(cache.state)
    assert seen == [True]
    transaction.abort()


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


def _canonical_qwen4_decode_module(signatures):
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
    return module


@pytest.mark.parametrize(
    "signatures",
    [
        ((6, 64), (6, 64), (6, 64), (6, 64)),  # physical layer 0
        ((4, 64), (5, 128), (5, 128), (5, 128)),  # physical layer 1
        ((5, 64), (6, 64), (6, 64), (6, 64)),  # physical layer 29
    ],
)
def test_qwen4_decode_static_gate_accepts_canonical_oqe_allocations(signatures):
    module = _canonical_qwen4_decode_module(signatures)

    assert prework_mod._qwen4_decode_static_eligible(module)
    module.in_proj_z.group_size = 64 if module.in_proj_z.group_size == 128 else 128
    assert not prework_mod._qwen4_decode_static_eligible(module)


def test_qwen4_decode_static_gate_survives_prefill_linear_reclass():
    """The VLM engine reclasses projections for q4 prefill routing (#3755)."""
    module = _canonical_qwen4_decode_module(((6, 64), (6, 64), (6, 64), (6, 64)))
    for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
        getattr(module, name).__class__ = _VLMQuantizedPrefillLinear

    assert prework_mod._qwen4_decode_static_eligible(module)


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
        linear_ops,
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
        out_proj=lambda x: fused,
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


def test_qwen4_decode_route_does_not_commit_states_on_failure(monkeypatch):
    q35 = pytest.importorskip("mlx_vlm.models.qwen3_5.language")
    cls = q35.Qwen3_5GatedDeltaNet
    old_conv = mx.zeros((1, 3, C), dtype=mx.bfloat16)
    old_recurrent = mx.zeros((1, HV, DV, DK), dtype=mx.float32)
    seen = []

    def stock(self, inputs, mask=None, cache=None, gdn_sink=None, target_verify=False):
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
        linear_ops,
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
    with pytest.raises(RuntimeError, match="late"):
        cls.__call__(module, mx.zeros((1, 1, 2560), dtype=mx.bfloat16), cache=cache)
    assert not seen
    assert cache[0] is old_conv and cache[1] is old_recurrent
    assert cache.advance_calls == 0


@pytest.mark.parametrize("batch", [2, 4])
@pytest.mark.parametrize("seq", [2, 3])
def test_batched_verify_preserves_output_and_all_rollback_states(
    monkeypatch, batch, seq
):
    import copy

    from mlx.utils import tree_flatten
    from mlx_vlm.models.cache import ArraysCache
    from mlx_vlm.models.qwen3_5 import language as q35

    args = SimpleNamespace(
        hidden_size=64,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
    )
    mx.random.seed(193)
    module = q35.Qwen3_5GatedDeltaNet(args)
    module.set_dtype(mx.bfloat16)
    module.eval()
    inputs = mx.random.normal((batch, seq, 64)).astype(mx.bfloat16)
    cache = ArraysCache(size=2)
    cache[0] = mx.random.normal((batch, 3, module.conv_dim)).astype(mx.bfloat16)
    cache[1] = mx.random.normal((batch, 4, 128, 128)) * 0.01
    reference_cache = copy.deepcopy(cache)
    verifier = Qwen3_5BatchInvariantForward()
    reference_transaction = start_speculative_cache([reference_cache], seq)
    reference = verifier._gated_delta(module, inputs, None, reference_cache)
    mx.eval(reference, reference_cache.state)

    monkeypatch.setattr(prework_mod, "_PATCHED", False)
    assert prework_mod.apply_qwen35_gdn_prework_patch()
    calls = []
    kernel = prework_mod.gdn_prework_fused

    def record(*args, **kwargs):
        calls.append(args[0].shape)
        return kernel(*args, **kwargs)

    monkeypatch.setattr(prework_mod, "gdn_prework_fused", record)
    transaction = start_speculative_cache([cache], seq)
    actual = verifier._gated_delta(module, inputs, None, cache)
    mx.eval(actual, cache.state)
    assert calls == [(batch, seq, module.conv_dim)]
    assert mx.array_equal(actual, reference).item()
    for retained in ([1] * batch, [seq] * batch, [1 + i % seq for i in range(batch)]):
        actual_cache, actual_tx = copy.deepcopy((cache, transaction))
        expected_cache, expected_tx = copy.deepcopy(
            (reference_cache, reference_transaction)
        )
        actual_tx.commit(retained)
        expected_tx.commit(retained)
        for (_, a), (_, b) in zip(
            tree_flatten(actual_cache.state), tree_flatten(expected_cache.state)
        ):
            assert mx.array_equal(a, b).item()
    transaction.abort()
    reference_transaction.abort()
    assert all(
        mx.array_equal(a, b).item() for a, b in zip(cache.state, reference_cache.state)
    )
