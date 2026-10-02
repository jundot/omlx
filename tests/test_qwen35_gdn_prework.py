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
from mlx_lm.models.gated_delta import normalize_qk
from mlx_vlm.models.qwen3_5 import language
from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward
from mlx_vlm.speculative.cache_state import start_speculative_cache
from mlx_vlm.speculative.ops import linear as linear_ops


from omlx.patches import qwen35_gdn_prework as prework_mod


@pytest.fixture(autouse=True)
def restore_hooks(monkeypatch):
    cls = language.Qwen3_5GatedDeltaNet
    monkeypatch.setattr(cls, "__call__", cls.__call__)
    monkeypatch.setattr(cls, "_normalize_qk", cls._normalize_qk)
    verifier = Qwen3_5BatchInvariantForward
    monkeypatch.setattr(verifier, "_gated_delta", verifier._gated_delta)
    monkeypatch.setattr(
        verifier,
        "_normalize_gated_delta_qk",
        verifier.__dict__["_normalize_gated_delta_qk"],
    )
    prework_mod.apply_qwen35_vlm_qk_norm_patch()


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
    q, k = normalize_qk(q, k, inv_scale=DK**-0.5, eps=1e-6)
    return q, k, v, new_state


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("seq", [2, 3, 4, 5, 7, 9])
@pytest.mark.parametrize("batch", [1, 2, 4])
def test_fused_prework_bit_exact(seq, batch, dtype):
    mx.random.seed(11)
    conv_w = (mx.random.normal((C, 4, 1)) * 0.2).astype(dtype)
    conv1d = nn.Conv1d(C, C, kernel_size=4, groups=C, bias=False)
    conv1d.weight = conv_w
    qkv = (mx.random.normal((batch, seq, C)) * 0.5).astype(dtype)
    state = (mx.random.normal((batch, 3, C)) * 0.5).astype(dtype)
    inv = DK**-0.5
    q_scale = mx.array(inv * inv, dtype=dtype)
    k_scale = mx.array(inv, dtype=dtype)

    ref = _composed(qkv, state, conv1d)
    got = gdn_prework_fused(qkv, state, conv_w, q_scale, k_scale, HK, HV, DK, DV)
    for name, r, g in zip(("q", "k", "v", "conv_state"), ref, got):
        assert r.shape == g.shape, name
        assert bool((r == g).all().item()), f"{name} not bit-exact at S={seq}"


def test_vlm_qk_norm_patch_matches_mlx_lm_normalize_qk():
    """mlx-vlm added the l2norm eps to mean(x^2); tiny k rows expose it."""
    mx.random.seed(3)
    q = mx.random.normal((1, 2, HK, DK)).astype(mx.bfloat16)
    k = (mx.random.normal((1, 2, HK, DK)) * 1e-3).astype(mx.bfloat16)
    expected = normalize_qk(q, k, inv_scale=DK**-0.5, eps=1e-6)

    layer = language.Qwen3_5GatedDeltaNet.__new__(language.Qwen3_5GatedDeltaNet)
    for actual in (
        layer._normalize_qk(q, k),
        Qwen3_5BatchInvariantForward._normalize_gated_delta_qk(layer, q, k),
    ):
        assert all(mx.array_equal(a, e).item() for a, e in zip(actual, expected))


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
    # A fast bf16 exp in the sigmoid gives this beta one ulp off MLX's on M3.
    b = mx.concatenate(
        [mx.full((1, 1, 1), -6.84375, dtype=mx.bfloat16), b[..., 1:]], axis=-1
    )
    A_log = (mx.random.normal((HV,)) * 0.2).astype(mx.bfloat16)
    dt_bias = (mx.random.normal((HV,)) * 0.2).astype(mx.bfloat16)
    q_scale = mx.array(DK**-0.5, dtype=mx.bfloat16)

    q, k, v, next_state = _composed_l2(qkv, state, conv1d)
    g, beta = _compute_g_beta(A_log, a, b, dt_bias)
    reference = (q, k, v, next_state, g, beta)
    actual = qwen4_decode_prework_fused(
        qkv,
        state,
        conv_w,
        q_scale,
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
@pytest.mark.parametrize("seq", [1, 2, 3, 4])
def test_qwen4_verify_prework_rows_equal_serial_decode_steps(seq):
    """Lightning MTP verify rows must reproduce the fused decode step per token."""
    mx.random.seed(53 + seq)
    conv_w = (mx.random.normal((C, 4, 1)) * 0.2).astype(mx.bfloat16)
    qkv = (mx.random.normal((1, seq, C)) * 0.5).astype(mx.bfloat16)
    state = (mx.random.normal((1, 3, C)) * 0.5).astype(mx.bfloat16)
    a = (mx.random.normal((1, 1, HV)) * 0.2).astype(mx.bfloat16)
    b = (mx.random.normal((1, 1, HV)) * 0.2).astype(mx.bfloat16)
    A_log = (mx.random.normal((HV,)) * 0.2).astype(mx.bfloat16)
    dt_bias = (mx.random.normal((HV,)) * 0.2).astype(mx.bfloat16)
    q_scale = mx.array(DK**-0.5, dtype=mx.bfloat16)

    verify = gdn_prework_fused(
        qkv, state, conv_w, q_scale, mx.array(1.0, dtype=mx.bfloat16), HK, HV, DK, DV, l2=True
    )
    serial_state = state
    for row in range(seq):
        q, k, v, serial_state, _, _ = qwen4_decode_prework_fused(
            qkv[:, row : row + 1], serial_state, conv_w, q_scale, b, a, A_log, dt_bias,
            HK, HV, DK, DV,
        )
        for name, step, window in zip(("q", "k", "v"), (q, k, v), verify[:3]):
            assert mx.array_equal(step[:, 0], window[:, row]).item(), f"{name} row {row}"
    assert mx.array_equal(serial_state, verify[3]).item()


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


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("seed", [3, 17, 41])
def test_qwen4_decode_step_kernel_equals_its_three_launches(dtype, seed):
    """Prework + recurrence + norm-gate in one launch, over two chained steps.

    The FP32 recurrent state is compared bit for bit; the FP32 instantiation
    also exposes the unrounded recurrence output and gate products that the
    BF16 output rounds away.
    """
    mx.random.seed(seed)
    conv_w = (mx.random.normal((C, 4, 1)) * 0.3).astype(dtype)
    q_scale = mx.array(DK**-0.5, dtype=dtype)
    A_log = (mx.random.normal((HV,)) * 0.5).astype(dtype)
    dt_bias = (mx.random.normal((HV,)) * 0.5).astype(dtype)
    norm_w = (1 + mx.random.normal((DV,)) * 0.1).astype(dtype)
    eps = mx.array(1e-6, dtype=mx.float32)
    conv_ref = conv_new = (mx.random.normal((1, 3, C)) * 0.5).astype(dtype)
    state_ref = state_new = mx.random.normal((1, HV, DV, DK)) * 0.1
    for _ in range(2):
        projected = (mx.random.normal((1, 1, C + HV * DV + 2 * HV)) * 0.8).astype(dtype)
        qkv, z, b, a = mx.split(projected, [C, C + HV * DV, C + HV * DV + HV], axis=-1)
        q, k, v, conv_ref, g, beta = qwen4_decode_prework_fused(
            qkv, conv_ref, conv_w, q_scale, b, a, A_log, dt_bias, HK, HV, DK, DV
        )
        y, state_ref = prework_mod._qwen4_decode_recurrence(q, k, v, g, beta, state_ref)
        gated_ref = prework_mod._qwen4_norm_gate(y, z, norm_w, eps, HV, DV)
        conv_new, state_new, gated_new = prework_mod.qwen4_decode_step_fused(
            qkv, z, b, a, conv_new, conv_w, q_scale, A_log, dt_bias, state_new, norm_w,
            eps, HK, HV, DK, DV,
        )
        for name, expected, observed in (
            ("conv_state", conv_ref, conv_new),
            ("state", state_ref, state_new),
            ("gated", gated_ref, gated_new),
        ):
            assert expected.dtype == observed.dtype, name
            assert expected.shape == observed.shape, name
            assert mx.array_equal(_bits(expected), _bits(observed)).item(), name


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_qwen4_prefill_route_is_bit_exact_across_chunks(monkeypatch):
    from mlx.utils import tree_map
    from mlx_vlm.models.cache import ArraysCache

    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
    )

    apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpGatedDeltaNet

    cls = language.Qwen3_5GatedDeltaNet
    monkeypatch.setattr(prework_mod, "_PATCHED", False)
    monkeypatch.setattr(cls, "_omlx_gdn_prework_patched", False, raising=False)
    assert prework_mod.apply_qwen35_gdn_prework_patch()

    mx.random.seed(37)
    config = SimpleNamespace(
        hidden_size=2560,
        linear_num_value_heads=HV,
        linear_num_key_heads=HK,
        linear_key_head_dim=DK,
        linear_value_head_dim=DV,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
        output_gate_type="sigmoid",
        hidden_act="silu",
    )
    module = Qwen4ExpGatedDeltaNet(config)
    module.update(
        tree_map(lambda p: (p * 0.2).astype(mx.bfloat16), module.parameters())
    )
    module.eval()
    chunks = [
        (mx.random.normal((1, rows, 2560)) * 0.5).astype(mx.bfloat16)
        for rows in (80, 67)
    ]

    def run(fused):
        monkeypatch.setattr(prework_mod, "_QWEN4_PREFILL_ENABLED", fused)
        cache = ArraysCache(size=2)
        outputs = [module(x, cache=cache) for x in chunks]
        mx.eval(outputs, cache[0], cache[1])
        return outputs, cache

    monkeypatch.setattr(prework_mod, "_QWEN4_PREFILL_ENGAGED_LOGGED", False)
    stock_out, stock_cache = run(False)
    assert not prework_mod._QWEN4_PREFILL_ENGAGED_LOGGED
    fused_out, fused_cache = run(True)
    assert prework_mod._QWEN4_PREFILL_ENGAGED_LOGGED

    for expected, observed in zip(stock_out, fused_out):
        assert mx.array_equal(expected, observed).item()
    for i in (0, 1):
        assert fused_cache[i].dtype == stock_cache[i].dtype
        assert mx.array_equal(stock_cache[i], fused_cache[i]).item()


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


@pytest.mark.parametrize(
    "signatures,out_proj",
    [
        # Community Qwen3.8-Flash-Next opt8: every GDN projection is 8-bit/g64.
        (((8, 64), (8, 64), (8, 64), (8, 64)), (8, 64)),
        # 27B Qwen3.5-lineage exports: 5-bit/g64 projections, 4-bit/g64 out_proj.
        (((5, 64), (5, 64), (5, 64), (5, 64)), (4, 64)),
        # Mixed per-tensor allocations from a sensitivity search.
        (((4, 64), (6, 64), (3, 64), (2, 128)), (4, 128)),
    ],
)
def test_qwen4_decode_static_gate_community_allocations_are_opt_in(
    signatures, out_proj
):
    module = _canonical_qwen4_decode_module(signatures)
    module.out_proj = _fake_quantized_linear(6144, 2560, *out_proj)

    assert not prework_mod._qwen4_decode_static_eligible(module)

    module._omlx_qwen4_wide_projections = True
    assert prework_mod._qwen4_decode_static_eligible(module)

    # still fail-closed on the canonical-layout checks, not just the recipe
    module.in_proj_z.group_size = 64 if module.in_proj_z.group_size == 128 else 128
    assert not prework_mod._qwen4_decode_static_eligible(module)


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("mode", "mxfp4"),
        ("group_size", 96),  # not a group size the quantizer implements
        ("group_size", 16),  # nvfp4-only: mx.quantize rejects it
        ("group_size", 256),  # ditto: affine tops out at 128
        ("bits", 7),  # not an affine width we have been shown
    ],
)
def test_qwen4_decode_static_gate_fails_closed_on_opt_in(attribute, value):
    module = _canonical_qwen4_decode_module(((8, 64), (8, 64), (8, 64), (8, 64)))
    module.out_proj = _fake_quantized_linear(6144, 2560, 8, 64)
    module._omlx_qwen4_wide_projections = True
    assert prework_mod._qwen4_decode_static_eligible(module)

    setattr(module.in_proj_a, attribute, value)
    assert not prework_mod._qwen4_decode_static_eligible(module)


def test_qwen4_decode_wide_projections_are_bit_exact_either_way():
    from mlx_vlm.speculative.ops.linear import (
        _decode_quantized_linears_fused,
        _target_verify_linears,
    )

    hidden = 2560
    rows = (C, HV * DV, HV, HV)
    recipes = [
        ((8, 64), (8, 64), (8, 64), (8, 64)),  # community opt8
        ((5, 64), (5, 64), (5, 64), (5, 64)),  # 27B Qwen3.5-lineage
        ((6, 128), (6, 128), (6, 128), (6, 128)),
        ((2, 32), (2, 32), (2, 32), (2, 32)),  # smallest admitted affine pair
        ((4, 64), (6, 64), (3, 64), (2, 128)),  # mixed: no concat, must fall back
    ]
    mx.random.seed(7)
    inputs = (mx.random.normal((1, 1, hidden)) * 0.1).astype(mx.bfloat16)
    for signatures in recipes:
        linears = []
        for output, (bits, group_size) in zip(rows, signatures):
            weight = (mx.random.normal((output, hidden)) * 0.05).astype(mx.bfloat16)
            packed, scales, biases = mx.quantize(
                weight, group_size=group_size, bits=bits, mode="affine"
            )
            linear = nn.QuantizedLinear(
                hidden, output, bias=False, group_size=group_size, bits=bits
            )
            linear.weight, linear.scales, linear.biases = packed, scales, biases
            linears.append(linear)
        separate = tuple(linear(inputs) for linear in linears)
        fused = _target_verify_linears(tuple(linears), inputs)
        mx.eval(*separate, *fused)
        for expected, observed in zip(separate, fused):
            assert mx.array_equal(expected, observed).item(), signatures
        # the mixed allocation must genuinely take the fallback, so a future
        # helper that silently stops concatenating cannot pass this vacuously
        concat_applies = (
            _decode_quantized_linears_fused(tuple(linears), inputs) is not None
        )
        homogeneous = len({(b, g) for b, g in signatures}) == 1
        assert concat_applies == homogeneous, signatures


def test_qwen4_decode_wide_allow_list_matches_the_quantizer():
    from omlx import oq

    assert prework_mod._ALLOWED_GROUPS == frozenset(oq._AFFINE_GROUP_SIZES)
    for bits in sorted(prework_mod._ALLOWED_BITS):
        mx.quantize(mx.zeros((64, 2560)), group_size=64, bits=bits, mode="affine")
    for group in sorted(prework_mod._ALLOWED_GROUPS):
        mx.quantize(mx.zeros((64, 2560)), group_size=group, bits=8, mode="affine")


@pytest.mark.parametrize("hidden_size", [5120, 4096, 2048])
def test_qwen4_decode_wide_opt_in_stays_within_the_2560_family(hidden_size):
    module = _canonical_qwen4_decode_module(((8, 64), (8, 64), (8, 64), (8, 64)))
    module.out_proj = _fake_quantized_linear(6144, 2560, 8, 64)
    module._omlx_qwen4_wide_projections = True
    assert prework_mod._qwen4_decode_static_eligible(module)

    def widen(linear):
        rows = linear.weight.shape[0]
        return _fake_quantized_linear(hidden_size, rows, linear.bits, linear.group_size)

    for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
        wider = _canonical_qwen4_decode_module(((8, 64), (8, 64), (8, 64), (8, 64)))
        wider._omlx_qwen4_wide_projections = True
        wider.out_proj = _fake_quantized_linear(6144, 2560, 8, 64)
        setattr(wider, name, widen(getattr(wider, name)))
        assert not prework_mod._qwen4_decode_static_eligible(wider), name

    # ...and a wider out_proj row count (hidden_size instead of 2560) also fails.
    wider = _canonical_qwen4_decode_module(((8, 64), (8, 64), (8, 64), (8, 64)))
    wider._omlx_qwen4_wide_projections = True
    wider.out_proj = _fake_quantized_linear(6144, hidden_size, 8, 64)
    assert not prework_mod._qwen4_decode_static_eligible(wider)


def test_qwen4_decode_static_gate_fails_closed_on_noncanonical_bias():
    module = _canonical_qwen4_decode_module(((8, 64), (8, 64), (8, 64), (8, 64)))
    module.out_proj = _fake_quantized_linear(6144, 2560, 8, 64)
    module._omlx_qwen4_wide_projections = True
    assert prework_mod._qwen4_decode_static_eligible(module)

    module.out_proj.biases = mx.zeros_like(module.out_proj.biases).astype(mx.float32)
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


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("batch,retained", [(1, [3]), (1, [8]), (3, [1, 8, 5])])
def test_fused_verify_replays_committed_rows_in_the_next_block(
    monkeypatch, batch, retained, dtype
):
    """The fused verify stores no per-row states: a commit leaves a lazy replay
    that the next block applies in its own launch. Outputs and committed states
    stay bit-exact to the stock recording path across two blocks. fp16 allows
    one ulp: on M1/M2 MLX's softplus rounds tiny values differently."""
    import copy

    from mlx_vlm.models.cache import ArraysCache
    from mlx_vlm.models.qwen3_5 import language as q35

    from omlx.patches import qwen35_gdn_verify_fused as fused_mod

    args = SimpleNamespace(
        hidden_size=64,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
    )
    seq = 8
    mx.random.seed(71)
    module = q35.Qwen3_5GatedDeltaNet(args)
    module.set_dtype(dtype)
    module.eval()
    blocks = [mx.random.normal((batch, seq, 64)).astype(dtype) for _ in range(2)]
    cache = ArraysCache(size=2)
    cache[0] = mx.random.normal((batch, 3, module.conv_dim)).astype(dtype)
    cache[1] = mx.random.normal((batch, 4, 128, 128)) * 0.01
    reference_cache = copy.deepcopy(cache)
    verifier = Qwen3_5BatchInvariantForward()

    def run(target, inputs):
        transaction = start_speculative_cache([target], seq)
        out = verifier._gated_delta(module, inputs, None, target)
        mx.eval(out)
        return out, transaction

    expected = []
    for inputs in blocks:
        out, transaction = run(reference_cache, inputs)
        transaction.commit(retained)
        expected.append(out)

    monkeypatch.setattr(prework_mod, "_PATCHED", False)
    assert prework_mod.apply_qwen35_gdn_prework_patch()
    replays = []
    kernel = fused_mod._kernel

    def record(main, replay):
        replays.append((main, replay))
        return kernel(main, replay)

    monkeypatch.setattr(fused_mod, "_kernel", record)
    def same(actual, reference):
        if dtype == mx.bfloat16:
            return mx.array_equal(actual, reference).item()
        return mx.allclose(actual, reference, rtol=2e-3, atol=1e-6).item()

    for index, inputs in enumerate(blocks):
        out, transaction = run(cache, inputs)
        assert same(out, expected[index])
        transaction.commit(retained)
    # The second block folded the first block's commit into its own launch.
    assert (True, True) in replays
    for actual, reference in zip(cache.state, reference_cache.state):
        assert same(actual, reference)


def test_qwen4_decode_setting_is_captured_per_model(monkeypatch):
    from omlx.scheduler import SchedulerConfig

    module = _canonical_qwen4_decode_module(((8, 64),) * 4)
    module.out_proj = _fake_quantized_linear(6144, 2560, 8, 64)
    model = SimpleNamespace(modules=lambda: [module])
    config = SchedulerConfig(qwen4_gdn_decode_wide_proj=True)
    monkeypatch.setenv("OMLX_QWEN4_GDN_DECODE_WIDE_PROJ", "1")
    assert not prework_mod._qwen4_decode_static_eligible(module)

    prework_mod.configure_qwen4_decode(
        model, wide_projections=config.qwen4_gdn_decode_wide_proj
    )
    config.qwen4_gdn_decode_wide_proj = False
    assert prework_mod._qwen4_decode_static_eligible(module)

    reloaded = _canonical_qwen4_decode_module(((8, 64),) * 4)
    reloaded.out_proj = _fake_quantized_linear(6144, 2560, 8, 64)
    prework_mod.configure_qwen4_decode(
        SimpleNamespace(modules=lambda: [reloaded]),
        wide_projections=config.qwen4_gdn_decode_wide_proj,
    )
    assert not prework_mod._qwen4_decode_static_eligible(reloaded)
    assert prework_mod._qwen4_decode_static_eligible(module)


def _bits(array):
    return array.view(mx.uint32 if array.dtype == mx.float32 else mx.uint16)


def _random_projection(input_dims, output_dims, bits, group_size):
    weight = (mx.random.normal((output_dims, input_dims)) * 0.05).astype(mx.bfloat16)
    linear = nn.QuantizedLinear(
        input_dims, output_dims, bias=False, group_size=group_size, bits=bits
    )
    linear.weight, linear.scales, linear.biases = mx.quantize(
        weight, group_size=group_size, bits=bits, mode="affine"
    )
    return linear


def _real_qwen4_decode_module(signatures, seed):
    from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import (
        Qwen4ExpGatedDeltaNet,
        Qwen4ExpRMSNormGated,
    )

    mx.random.seed(seed)
    module = Qwen4ExpGatedDeltaNet.__new__(Qwen4ExpGatedDeltaNet)
    nn.Module.__init__(module)
    module.num_k_heads, module.num_v_heads = HK, HV
    module.head_k_dim, module.head_v_dim = DK, DV
    module.conv_kernel_size = 4
    module.conv1d = nn.Conv1d(C, C, 4, groups=C, bias=False)
    module.conv1d.weight = (mx.random.normal((C, 4, 1)) * 0.3).astype(mx.bfloat16)
    module.norm = Qwen4ExpRMSNormGated(DV, eps=1e-6, activation="sigmoid")
    module.norm.weight = (1 + mx.random.normal((DV,)) * 0.1).astype(mx.bfloat16)
    module.A_log = (mx.random.normal((HV,)) * 0.5).astype(mx.bfloat16)
    module.dt_bias = (mx.random.normal((HV,)) * 0.5).astype(mx.bfloat16)
    for name, rows, (bits, group) in zip(
        ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"),
        (C, HV * DV, HV, HV),
        signatures,
    ):
        setattr(module, name, _random_projection(2560, rows, bits, group))
    module.out_proj = _random_projection(HV * DV, 2560, 5, 128)
    module.eval()
    mx.eval(module.parameters())
    return module


def _decode_steps(module, inputs, conv_state, recurrent_state):
    from mlx_vlm.models.cache import ArraysCache

    cache = ArraysCache(size=2)
    cache[0], cache[1] = conv_state, recurrent_state
    outputs = []
    for x in inputs:
        outputs.append(module(x, cache=cache))
        mx.eval(outputs[-1], cache[0], cache[1])
    return outputs, (cache[0], cache[1])


def _assert_planned_decode_matches_per_call_path(monkeypatch, module, steps, seed):
    mx.random.seed(seed)
    inputs = [
        (mx.random.normal((1, 1, 2560)) * 0.5).astype(mx.bfloat16) for _ in range(steps)
    ]
    conv_state = (mx.random.normal((1, 3, C)) * 0.5).astype(mx.bfloat16)
    recurrent_state = mx.random.normal((1, HV, DV, DK)) * 0.01
    monkeypatch.setattr(prework_mod, "_QWEN4_DECODE_PLAN_ENABLED", False)
    expected, expected_state = _decode_steps(
        module, inputs, conv_state, recurrent_state
    )
    monkeypatch.setattr(prework_mod, "_QWEN4_DECODE_PLAN_ENABLED", True)
    actual, actual_state = _decode_steps(module, inputs, conv_state, recurrent_state)
    for want, got in zip(
        (*expected, *expected_state), (*actual, *actual_state), strict=True
    ):
        assert want.dtype == got.dtype and want.shape == got.shape
        assert mx.array_equal(_bits(want), _bits(got)).item()


@pytest.fixture
def patched_decode(monkeypatch):
    def stock(*args, **kwargs):
        raise AssertionError("eligible Qwen4 decode unexpectedly fell back")

    monkeypatch.setattr(prework_mod, "_PATCHED", False)
    monkeypatch.setattr(language.Qwen3_5GatedDeltaNet, "__call__", stock)
    assert prework_mod.apply_qwen35_gdn_prework_patch()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize(
    "signatures",
    [
        ((6, 64), (6, 64), (6, 64), (6, 64)),  # one fused in-projection launch
        ((4, 64), (5, 128), (5, 128), (5, 128)),  # four projection launches
    ],
)
@pytest.mark.parametrize("seed", [3, 11])
@pytest.mark.parametrize(
    "step_fused, qmv", [(True, True), (True, False), (False, True), (False, False)]
)
def test_qwen4_planned_decode_is_bit_identical_to_per_call_path(
    monkeypatch, patched_decode, signatures, seed, step_fused, qmv
):
    from omlx.patches.row_exact_qmv import OneRowQmv

    monkeypatch.setattr(prework_mod, "_QWEN4_DECODE_STEP_FUSED", step_fused)
    monkeypatch.setattr(prework_mod, "_QWEN4_DECODE_QMV", qmv)
    launches = []
    step = prework_mod.qwen4_decode_step_fused
    projection = OneRowQmv.__call__

    def counted_step(*args):
        launches.append("step")
        return step(*args)

    def counted_projection(self, x):
        launches.append("qmv")
        return projection(self, x)

    monkeypatch.setattr(prework_mod, "qwen4_decode_step_fused", counted_step)
    monkeypatch.setattr(OneRowQmv, "__call__", counted_projection)
    module = _real_qwen4_decode_module(signatures, seed)
    _assert_planned_decode_matches_per_call_path(monkeypatch, module, 3, seed)
    # The planned steps ran the new launches: the step kernel, the out-projection
    # and, when the four projections share one allocation, the in-projection.
    projections = (1 + (len(set(signatures)) == 1)) if qmv else 0
    assert launches.count("step") == (3 if step_fused else 0)
    assert launches.count("qmv") == 3 * projections


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_qwen4_decode_plan_follows_replaced_weights_and_modules(
    monkeypatch, patched_decode
):
    module = _real_qwen4_decode_module(((6, 64),) * 4, 5)
    _assert_planned_decode_matches_per_call_path(monkeypatch, module, 1, 5)
    # A projection with another allocation turns the fused in-projection into four.
    module.in_proj_z = _random_projection(2560, HV * DV, 5, 128)
    _assert_planned_decode_matches_per_call_path(monkeypatch, module, 1, 6)
    # New tensors on the same modules.
    module.in_proj_qkv.weight = mx.random.randint(
        0, 2**32 - 1, module.in_proj_qkv.weight.shape, dtype=mx.uint32
    )
    module.conv1d.weight = (mx.random.normal((C, 4, 1)) * 0.3).astype(mx.bfloat16)
    module.A_log = (mx.random.normal((HV,)) * 0.5).astype(mx.bfloat16)
    module.norm.weight = (1 + mx.random.normal((DV,)) * 0.1).astype(mx.bfloat16)
    module.out_proj = _random_projection(HV * DV, 2560, 5, 128)
    _assert_planned_decode_matches_per_call_path(monkeypatch, module, 1, 7)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_qwen4_decode_plan_turns_ineligible_on_replacement(monkeypatch):
    fallbacks = []

    def stock(self, inputs, mask=None, cache=None):
        fallbacks.append(inputs.shape)
        return inputs

    monkeypatch.setattr(prework_mod, "_PATCHED", False)
    monkeypatch.setattr(language.Qwen3_5GatedDeltaNet, "__call__", stock)
    assert prework_mod.apply_qwen35_gdn_prework_patch()
    module = _real_qwen4_decode_module(((6, 64),) * 4, 9)
    _decode_steps(
        module,
        [mx.zeros((1, 1, 2560), dtype=mx.bfloat16)],
        mx.zeros((1, 3, C), dtype=mx.bfloat16),
        mx.zeros((1, HV, DV, DK)),
    )
    assert not fallbacks
    # 8-bit in-projections are outside the shipped allocation allow-list.
    module.in_proj_qkv = _random_projection(2560, C, 8, 64)
    _decode_steps(
        module,
        [mx.zeros((1, 1, 2560), dtype=mx.bfloat16)],
        mx.zeros((1, 3, C), dtype=mx.bfloat16),
        mx.zeros((1, HV, DV, DK)),
    )
    assert fallbacks == [(1, 1, 2560)]
    # ...until the model opts in to wide projections.
    prework_mod.configure_qwen4_decode(
        SimpleNamespace(modules=lambda: [module]), wide_projections=True
    )
    _decode_steps(
        module,
        [mx.zeros((1, 1, 2560), dtype=mx.bfloat16)],
        mx.zeros((1, 3, C), dtype=mx.bfloat16),
        mx.zeros((1, HV, DV, DK)),
    )
    assert fallbacks == [(1, 1, 2560)]


# --- Fused Qwen4 speculative verify ------------------------------------------

P = C + HV * DV + 2 * HV  # stacked in-projection row [qkv | z | b | a]


@pytest.fixture
def qwen4_verify(monkeypatch):
    """Patched decode and verify entries, the row-exact projection routing, and
    a count of fused verify launches."""
    from omlx.patches import qwen35_verify_qmm

    monkeypatch.setattr(prework_mod, "_PATCHED", False)
    assert prework_mod.apply_qwen35_gdn_prework_patch()
    qwen35_verify_qmm.apply_verify_qmm_patch()
    launches = []
    step = prework_mod.qwen4_verify_step_fused

    def counted(*args):
        launches.append(args[0].shape[1])
        return step(*args)

    monkeypatch.setattr(prework_mod, "qwen4_verify_step_fused", counted)
    yield launches
    qwen35_verify_qmm.set_verify_qmm_armed(False)


def _verify_block(module, inputs, conv_state, recurrent_state, *, row_exact=True):
    from mlx_vlm.models.cache import ArraysCache
    from mlx_vlm.models.qwen4_exp import language as q4

    from omlx.patches import qwen35_verify_qmm

    cache = ArraysCache(size=2)
    cache[0], cache[1] = conv_state, recurrent_state
    transaction = start_speculative_cache([cache], inputs.shape[1])
    qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=row_exact)
    try:
        # The compat vendor's verifier, or upstream mlx-vlm's when a test
        # earlier in the session imported that first (as the patch resolves).
        verifier = (getattr(q4, "_Qwen4Verifier", None) or q4.Qwen4ExpBatchInvariantForward)()
        out = verifier._gated_delta(module, inputs, None, cache)
    finally:
        qwen35_verify_qmm.set_verify_qmm_armed(False)
    mx.eval(out, cache.state)
    return out, cache, transaction


def _same(want, got):
    return (
        want.dtype == got.dtype
        and want.shape == got.shape
        and mx.array_equal(_bits(want), _bits(got)).item()
    )


def _verify_inputs(rows, seed):
    mx.random.seed(seed)
    return (
        (mx.random.normal((1, rows, 2560)) * 0.5).astype(mx.bfloat16),
        (mx.random.normal((1, 3, C)) * 0.5).astype(mx.bfloat16),
        mx.random.normal((1, HV, DV, DK)) * 0.05,
    )


# On the paravirtual GPU of hosted macOS runners the per-op verify reference
# drifts from serial decode in later rows; the fused verify still equals serial
# decode there (test_qwen4_fused_verify_rows_equal_serial_decode_steps).
_PARAVIRTUAL_GPU = mx.metal.is_available() and not str(
    mx.device_info().get("architecture", "")
).startswith("applegpu")


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.skipif(_PARAVIRTUAL_GPU, reason="per-op reference is not row-exact here")
@pytest.mark.parametrize("signatures", [((6, 64),) * 4, ((8, 64),) * 4])
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 9])
@pytest.mark.parametrize("seed", [3, 11])
def test_qwen4_fused_verify_equals_per_op_verify_and_rollback(
    monkeypatch, qwen4_verify, signatures, rows, seed
):
    """Outputs, next states, rollback records and every accepted prefix."""
    import copy

    from mlx.utils import tree_flatten

    module = _real_qwen4_decode_module(signatures, seed)
    inputs, conv_state, recurrent_state = _verify_inputs(rows, seed + rows)
    monkeypatch.setattr(prework_mod, "_QWEN4_VERIFY_FUSED", False)
    want, want_cache, want_tx = _verify_block(module, inputs, conv_state, recurrent_state)
    assert qwen4_verify == []
    monkeypatch.setattr(prework_mod, "_QWEN4_VERIFY_FUSED", True)
    got, got_cache, got_tx = _verify_block(module, inputs, conv_state, recurrent_state)
    assert qwen4_verify == [rows]

    assert _same(want, got)
    for index in (0, 1):
        assert _same(want_cache[index], got_cache[index])
    want_records = want_cache._speculation["records"]
    got_records = got_cache._speculation["records"]
    (kind, want_window, width), got_window = want_records[0], got_records[0]
    assert got_window[0] == kind == "window" and got_window[2] == width == 3
    assert _same(want_window, got_window[1])
    (kind, want_history, want_final), got_states = want_records[1], got_records[1]
    assert got_states[0] == kind == "states"
    if rows == 1:
        assert want_history is None and got_states[1] is None
    else:
        assert _same(want_history, got_states[1])
    assert _same(want_final, got_states[2])
    for keep in range(rows + 1):
        want_kept, want_keep_tx = copy.deepcopy((want_cache, want_tx))
        got_kept, got_keep_tx = copy.deepcopy((got_cache, got_tx))
        want_keep_tx.commit([keep])
        got_keep_tx.commit([keep])
        for (_, a), (_, b) in zip(
            tree_flatten(want_kept.state), tree_flatten(got_kept.state), strict=True
        ):
            assert _same(a, b), keep
    want_tx.abort()
    got_tx.abort()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("rows", [2, 3, 4])
@pytest.mark.parametrize("seed", [5, 23])
def test_qwen4_fused_verify_rows_equal_serial_decode_steps(
    monkeypatch, qwen4_verify, rows, seed
):
    """Row t, the state after it and the conv window equal the t-th planned
    one-token decode step; a partial accept restores that step's states."""
    import copy

    from mlx_vlm.models.cache import ArraysCache

    decode_steps = []
    decode_step = prework_mod.qwen4_decode_step_fused

    def counted(*args):
        decode_steps.append(1)
        return decode_step(*args)

    monkeypatch.setattr(prework_mod, "qwen4_decode_step_fused", counted)
    module = _real_qwen4_decode_module(((6, 64),) * 4, seed)
    inputs, conv_state, recurrent_state = _verify_inputs(rows, seed)
    serial = ArraysCache(size=2)
    serial[0], serial[1] = conv_state, recurrent_state
    outputs, convs, states = [], [conv_state], [recurrent_state]
    for t in range(rows):
        outputs.append(module(inputs[:, t : t + 1], cache=serial))
        mx.eval(outputs[-1], serial[0], serial[1])
        convs.append(serial[0])
        states.append(serial[1])
    assert len(decode_steps) == rows

    got, cache, transaction = _verify_block(module, inputs, conv_state, recurrent_state)
    assert qwen4_verify == [rows]
    for t in range(rows):
        assert _same(outputs[t][:, 0], got[:, t]), t
    window = cache._speculation["records"][0][1]
    _, history, final = cache._speculation["records"][1]
    for t in range(rows + 1):
        assert _same(convs[t], window[:, t : t + 3]), t
    for t in range(rows - 1):
        assert _same(states[t + 1], history[:, t]), t
    assert _same(states[rows], final)
    assert _same(convs[rows], cache[0]) and _same(states[rows], cache[1])
    for keep in range(rows + 1):
        kept, keep_tx = copy.deepcopy((cache, transaction))
        keep_tx.commit([keep])
        assert _same(convs[keep], kept[0]) and _same(states[keep], kept[1]), keep
    transaction.abort()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("rows", [1, 2, 4, 9])
@pytest.mark.parametrize("seed", [7, 29])
def test_qwen4_verify_step_kernel_equals_chained_decode_step_kernels(dtype, rows, seed):
    """The FP32 instantiation carries the unrounded conv, q/k/v, recurrence
    output and gate products that BF16 rounds away."""
    mx.random.seed(seed)
    conv_w = (mx.random.normal((C, 4, 1)) * 0.3).astype(dtype)
    q_scale = mx.array(DK**-0.5, dtype=dtype)
    A_log = (mx.random.normal((HV,)) * 0.5).astype(dtype)
    dt_bias = (mx.random.normal((HV,)) * 0.5).astype(dtype)
    norm_w = (1 + mx.random.normal((DV,)) * 0.1).astype(dtype)
    eps = mx.array(1e-6, dtype=mx.float32)
    conv_state = (mx.random.normal((1, 3, C)) * 0.5).astype(dtype)
    state = mx.random.normal((1, HV, DV, DK)) * 0.1
    proj = (mx.random.normal((1, rows, P)) * 0.8).astype(dtype)

    conv_out, window, history, final, out = prework_mod.qwen4_verify_step_fused(
        proj, conv_state, conv_w, q_scale, A_log, dt_bias, state, norm_w, eps, HK, HV, DK, DV
    )
    assert (history is None) == (rows == 1)
    assert _same(mx.concatenate([conv_state, proj[..., :C]], axis=1), window)
    conv, recurrent = conv_state, state
    for t in range(rows):
        qkv, z, b, a = mx.split(proj[:, t : t + 1], [C, C + HV * DV, C + HV * DV + HV], axis=-1)
        conv, recurrent, gated = prework_mod.qwen4_decode_step_fused(
            qkv, z, b, a, conv, conv_w, q_scale, A_log, dt_bias, recurrent, norm_w, eps,
            HK, HV, DK, DV,
        )
        assert _same(gated[:, 0], out[:, t]), t
        if t < rows - 1:
            assert _same(recurrent, history[:, t]), t
    assert _same(conv, conv_out) and _same(recurrent, final)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_qwen4_fused_verify_multi_row_blocks_need_row_exact_arming(qwen4_verify):
    """Unarmed multi-row verify projections run other kernels, so their rows
    keep the per-op path; a one-row block takes one-row arithmetic either way."""
    module = _real_qwen4_decode_module(((6, 64),) * 4, 13)
    for rows in (3, 1):
        inputs, conv_state, recurrent_state = _verify_inputs(rows, 13)
        _, _, transaction = _verify_block(
            module, inputs, conv_state, recurrent_state, row_exact=False
        )
        transaction.abort()
    assert qwen4_verify == [1]
