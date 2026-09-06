# SPDX-License-Identifier: Apache-2.0
"""Fused hyper-connection kernels: parity with the canonical path, eligibility, kill switch."""
from __future__ import annotations

import importlib

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat


@pytest.fixture(autouse=True)
def _vendored_qwen4():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()

HC, HIDDEN, LOWRANK = 4, 2560, 320
WIDTH = HC * HIDDEN


def _module(bits: int, use_combine: bool = True, hidden: int = HIDDEN):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpGatedResidual, Qwen4ExpRMSNorm

    width = HC * hidden
    module = Qwen4ExpGatedResidual.__new__(Qwen4ExpGatedResidual)
    nn.Module.__init__(module)
    module.hc_count, module.hidden_size, module.hc_lowrank = HC, hidden, LOWRANK
    module.hc_norm = Qwen4ExpRMSNorm(width, group_size=hidden, eps=1e-6)
    module.hc_norm.weight = (mx.random.normal((width,)) * 0.05).astype(mx.bfloat16)
    module.input_mix_weight_down = nn.QuantizedLinear(width, LOWRANK, bias=False, group_size=64, bits=bits)
    module.input_mix_weight_up = nn.QuantizedLinear(LOWRANK, width, bias=False, group_size=64, bits=bits)
    if use_combine:
        module.block_inject_weight = nn.QuantizedLinear(width, HC, bias=False, group_size=64, bits=bits)
    for name in ("input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"):
        projection = getattr(module, name, None)
        if projection is not None:
            # Checkpoint-like statistics: positive scales, small biases. Random-sign scales drive the
            # up-projection gate into saturation where any rounding difference flips whole elements.
            projection.scales = (mx.abs(mx.random.normal(projection.scales.shape)) * 0.01 + 0.002).astype(mx.bfloat16)
            projection.biases = (mx.random.normal(projection.biases.shape) * 0.005).astype(mx.bfloat16)
    mx.eval(module.parameters())
    return module


def _reference_fp32(module, x):
    def dequant(q):
        return mx.dequantize(q.weight, q.scales, q.biases, group_size=q.group_size, bits=q.bits).astype(mx.float32)

    normed = module.hc_norm(x).astype(mx.float32)
    mix = nn.silu((normed @ dequant(module.input_mix_weight_down).T) / HC)
    gate = mx.sigmoid(mix @ dequant(module.input_mix_weight_up).T)
    hidden = module.hidden_size
    mixed = mx.mean(gate.reshape(*gate.shape[:-1], HC, hidden) * normed.reshape(*normed.shape[:-1], HC, hidden), axis=-2)
    if "block_inject_weight" not in module:
        return mixed, None
    return mixed, 2 * mx.sigmoid((normed @ dequant(module.block_inject_weight).T) / HC)


def _ulps(a, b):
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    ulp = float(mx.abs(b).max().item()) * 2.0**-7
    diff = mx.abs(a - b) / ulp
    return float(diff.max().item()), float(diff.mean().item())


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
@pytest.mark.parametrize("rows", [1, 4, 16])
@pytest.mark.parametrize("use_combine", [True, False])
def test_fused_matches_canonical_path(bits, rows, use_combine):
    mx.random.seed(20260905 + bits * 100 + rows)
    _assert_fused_matches_canonical(_module(bits, use_combine), rows, use_combine)


# Sizes the checkpoint never has, chosen so every kernel sees a partial final block:
#   768  -> down tail 256 for 4/5-bit;                          norm and inject loops exact
#   1152 -> down tail 128 (all bits), inject tail 128 (4/5-bit), norm tail 128
#   1344 -> down tail 320 (4/5) / 64 (6/8), inject tail 64,       norm tail 64
#   512, 1536 -> odd multiples of 512, no tails anywhere
@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
@pytest.mark.parametrize("hidden", [512, 768, 1152, 1344, 1536])
def test_fused_matches_canonical_path_at_other_hidden_sizes(hidden, bits):
    mx.random.seed(20260906 + hidden + bits)
    _assert_fused_matches_canonical(_module(bits, True, hidden=hidden), 16, True)


def _assert_fused_matches_canonical(module, rows, use_combine):
    from mlx_vlm.models.qwen4_exp import hc_fused

    hidden = module.hidden_size
    x = mx.random.normal((1, rows, HC * hidden)).astype(mx.bfloat16)
    mx.eval(x)
    assert hc_fused.compatible(module, x)
    fused = hc_fused.fused_forward(module, x)
    assert fused is not None
    canonical = module._forward(x)
    mx.eval(fused, canonical)
    ref_mixed, ref_inject = _reference_fp32(module, x)
    if use_combine:
        fused_mixed, passthrough, fused_inject = fused
        canon_mixed, _, canon_inject = canonical
        assert passthrough is x
        assert fused_inject.shape == canon_inject.shape == (1, rows, HC)
        assert _ulps(fused_inject, canon_inject)[0] <= 4
        assert _ulps(fused_inject, ref_inject)[0] <= 4
    else:
        fused_mixed, canon_mixed = fused, canonical
    assert fused_mixed.shape == canon_mixed.shape == (1, rows, hidden)
    assert fused_mixed.dtype == mx.bfloat16
    max_vs_canon, mean_vs_canon = _ulps(fused_mixed, canon_mixed)
    assert max_vs_canon <= 16 and mean_vs_canon <= 0.5
    # Both paths round differently; judge each against fp32. The fused path keeps fp32 through
    # the epilogues, so it must stay at least as close to fp32 as the canonical path (with slack).
    max_fused, mean_fused = _ulps(fused_mixed, ref_mixed)
    max_canon, mean_canon = _ulps(canon_mixed, ref_mixed)
    assert max_fused <= max(2 * max_canon, 6)
    assert mean_fused <= mean_canon * 1.5 + 0.05


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("hidden", [HIDDEN, 768, 1152, 1344])
def test_fused_norm_is_bit_identical_to_rms_norm(hidden):
    from mlx_vlm.models.qwen4_exp import hc_fused

    mx.random.seed(7)
    module = _module(4, hidden=hidden)
    width = HC * hidden
    x = (mx.random.normal((1, 4, width)) * 3).astype(mx.bfloat16)
    flat = x.reshape(4, width)
    normed = hc_fused._kernel("omlx_qwen4_hc_fused_norm", ["x", "w", "eps"], ["xn"], hc_fused._N_SOURCE)(
        inputs=[flat, module.hc_norm.weight, hc_fused._eps_array(module)],
        template=[("T", mx.bfloat16), ("K", width), ("H", hidden)],
        grid=(256, HC, 4),
        threadgroup=(256, 1, 1),
        output_shapes=[(4, width)],
        output_dtypes=[mx.bfloat16],
    )[0]
    expected = module.hc_norm(x).reshape(4, width)
    mx.eval(normed, expected)
    assert mx.array_equal(normed.view(mx.uint16), expected.view(mx.uint16)).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_gated_residual_call_routes_through_fused_path(monkeypatch):
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    x = mx.random.normal((1, 2, WIDTH)).astype(mx.bfloat16)
    calls = []
    original = hc_fused.fused_forward
    monkeypatch.setattr(hc_fused, "fused_forward", lambda m, h: calls.append(h.shape) or original(m, h))
    out = module(x)
    mx.eval(out)
    assert calls == [(1, 2, WIDTH)]


@pytest.mark.parametrize("hidden,bits", [(800, 4), (1056, 5)])
def test_compatible_rejects_hidden_not_multiple_of_64(hidden, bits):
    # 64 is the quantisation group and the up kernel's grid unit; nothing below it is handled.
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(bits, True, hidden=hidden)
    x = mx.random.normal((1, 4, HC * hidden)).astype(mx.bfloat16)
    assert not hc_fused.compatible(module, x)


def test_ineligible_model_is_logged_once(monkeypatch, caplog):
    from mlx_vlm.models.qwen4_exp import hc_fused

    monkeypatch.setattr(hc_fused, "_INELIGIBLE_LOGGED", False)
    module = _module(4, True, hidden=800)
    x = mx.random.normal((1, 4, HC * 800)).astype(mx.bfloat16)
    with caplog.at_level("INFO", logger=hc_fused.logger.name):
        assert not hc_fused.compatible(module, x)
        assert not hc_fused.compatible(module, x)
        # Prefill-sized inputs are expected to skip the fused path and must not log.
        monkeypatch.setattr(hc_fused, "_INELIGIBLE_LOGGED", False)
        assert not hc_fused.compatible(_module(4), mx.random.normal((1, 64, WIDTH)).astype(mx.bfloat16))
    messages = [r.getMessage() for r in caplog.records if "fused hyper-connection kernels not used" in r.getMessage()]
    assert len(messages) == 1
    assert "hidden_size=800" in messages[0]


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_fused_path_takes_precedence_over_exact_hybrid_projection(monkeypatch):
    # With MTP off, load_weights flags modules with upstream's exact hybrid projection and compiles
    # their single-token _forward. The fused path must still win for decode rows (+12..14% serial).
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    module._omlx_exact_hybrid_projection = True
    module._compiled_forward = lambda h: pytest.fail("compiled single-token path must not run")
    x = mx.random.normal((1, 1, WIDTH)).astype(mx.bfloat16)
    assert hc_fused.compatible(module, x)
    calls = []
    original = hc_fused.fused_forward
    monkeypatch.setattr(hc_fused, "fused_forward", lambda m, h: calls.append(h.shape) or original(m, h))
    out = module(x)
    mx.eval(out)
    assert calls == [(1, 1, WIDTH)]


def test_compatible_fails_closed():
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    ok = mx.random.normal((1, 4, WIDTH)).astype(mx.bfloat16)
    if mx.metal.is_available():
        assert hc_fused.compatible(module, ok)
    assert not hc_fused.compatible(module, mx.random.normal((1, 17, WIDTH)).astype(mx.bfloat16))
    assert not hc_fused.compatible(module, mx.random.normal((2, 9, WIDTH)).astype(mx.bfloat16))
    assert not hc_fused.compatible(module, mx.random.normal((1, 4, WIDTH)).astype(mx.float16))
    assert not hc_fused.compatible(module, mx.random.normal((4, WIDTH)).astype(mx.bfloat16))
    module.input_inject_weight = nn.Linear(WIDTH, LOWRANK + HC, bias=False)
    assert not hc_fused.compatible(module, ok)
    del module.input_inject_weight
    module.input_mix_weight_down = nn.Linear(WIDTH, LOWRANK, bias=False)
    assert not hc_fused.compatible(module, ok)


def test_kill_switch_disables_fused_path(monkeypatch):
    monkeypatch.setenv("OMLX_QWEN4_HC_FUSED", "0")
    from mlx_vlm.models.qwen4_exp import hc_fused

    reloaded = importlib.reload(hc_fused)
    try:
        assert not reloaded.enabled()
        assert not reloaded.compatible(_module(4), mx.random.normal((1, 4, WIDTH)).astype(mx.bfloat16))
    finally:
        monkeypatch.delenv("OMLX_QWEN4_HC_FUSED")
        importlib.reload(hc_fused)
