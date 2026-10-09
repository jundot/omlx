# SPDX-License-Identifier: Apache-2.0
"""Fused hyper-connection kernels: parity with the canonical path, eligibility, kill switch."""

from __future__ import annotations

import dataclasses
import importlib
from unittest.mock import Mock

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from omlx.custom_kernels.nax import is_nax_available
from omlx.patches import mlx_vlm_qwen4_exp_compat as compat
from tests.test_mlx_vlm_qwen4_exp_compat import _tiny_config


@pytest.fixture(autouse=True)
def _vendored_qwen4():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()


HC, HIDDEN, LOWRANK = 4, 2560, 320
WIDTH = HC * HIDDEN


def _module(bits: int, use_combine: bool = True, hidden: int = HIDDEN, group_size: int = 64):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpGatedResidual, Qwen4ExpRMSNorm

    width = HC * hidden
    module = Qwen4ExpGatedResidual.__new__(Qwen4ExpGatedResidual)
    nn.Module.__init__(module)
    module.hc_count, module.hidden_size, module.hc_lowrank = HC, hidden, LOWRANK
    module.hc_norm = Qwen4ExpRMSNorm(width, group_size=hidden, eps=1e-6)
    module.hc_norm.weight = (mx.random.normal((width,)) * 0.05).astype(mx.bfloat16)
    module.input_mix_weight_down = nn.QuantizedLinear(
        width, LOWRANK, bias=False, group_size=group_size, bits=bits
    )
    module.input_mix_weight_up = nn.QuantizedLinear(
        LOWRANK, width, bias=False, group_size=group_size, bits=bits
    )
    if use_combine:
        module.block_inject_weight = nn.QuantizedLinear(
            width, HC, bias=False, group_size=group_size, bits=bits
        )
    for name in ("input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"):
        projection = getattr(module, name, None)
        if projection is not None:
            # Checkpoint-like statistics: positive scales, small biases. Random-sign scales drive the
            # up-projection gate into saturation where any rounding difference flips whole elements.
            projection.scales = (
                mx.abs(mx.random.normal(projection.scales.shape)) * 0.01 + 0.002
            ).astype(mx.bfloat16)
            projection.biases = (
                mx.random.normal(projection.biases.shape) * 0.005
            ).astype(mx.bfloat16)
    mx.eval(module.parameters())
    return module


def _reference_fp32(module, x):
    def dequant(q):
        return mx.dequantize(
            q.weight, q.scales, q.biases, group_size=q.group_size, bits=q.bits
        ).astype(mx.float32)

    normed = module.hc_norm(x).astype(mx.float32)
    mix = nn.silu((normed @ dequant(module.input_mix_weight_down).T) / HC)
    gate = mx.sigmoid(mix @ dequant(module.input_mix_weight_up).T)
    hidden = module.hidden_size
    mixed = mx.mean(
        gate.reshape(*gate.shape[:-1], HC, hidden)
        * normed.reshape(*normed.shape[:-1], HC, hidden),
        axis=-2,
    )
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
@pytest.mark.parametrize("group_size", [32, 64])
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
@pytest.mark.parametrize("rows", [1, 4, 16])
@pytest.mark.parametrize("use_combine", [True, False])
def test_fused_matches_canonical_path(bits, rows, use_combine, group_size):
    mx.random.seed(20260905 + bits * 100 + rows)
    _assert_fused_matches_canonical(
        _module(bits, use_combine, group_size=group_size), rows, use_combine
    )


# Sizes the checkpoint never has, chosen so every kernel sees a partial final block:
#   768  -> down tail 256 for 4/5-bit;                          norm and inject loops exact
#   1152 -> down tail 128 (all bits), inject tail 128 (4/5-bit), norm tail 128
#   1344 -> down tail 320 (4/5) / 64 (6/8), inject tail 64,       norm tail 64
@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("group_size", [32, 64])
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
@pytest.mark.parametrize("hidden", [768, 1152, 1344])
def test_fused_matches_canonical_path_at_other_hidden_sizes(hidden, bits, group_size):
    mx.random.seed(20260906 + hidden + bits)
    module = _module(bits, True, hidden=hidden, group_size=group_size)
    _assert_fused_matches_canonical(module, 16, True)


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
    normed = hc_fused._kernel(
        "omlx_qwen4_hc_fused_norm", ["x", "w", "eps"], ["xn"], hc_fused._N_SOURCE
    )(
        inputs=[flat, module.hc_norm.weight, hc_fused._eps_array(module)],
        template=[("T", mx.bfloat16), ("K", width), ("H", hidden)],
        grid=(256, HC, 4),
        threadgroup=(256, 1, 1),
        output_shapes=[(4, width)],
        output_dtypes=[mx.bfloat16],
    )[
        0
    ]
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
    monkeypatch.setattr(
        hc_fused, "fused_forward", lambda m, h: calls.append(h.shape) or original(m, h)
    )
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
        assert not hc_fused.compatible(
            _module(4), mx.random.normal((1, 64, WIDTH)).astype(mx.bfloat16)
        )
    messages = [
        r.getMessage()
        for r in caplog.records
        if "fused hyper-connection kernels not used" in r.getMessage()
    ]
    assert len(messages) == 1
    assert "hidden_size=800" in messages[0]


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_fused_path_takes_precedence_over_exact_hybrid_projection(monkeypatch):
    # Fused dispatch takes precedence over the compiled hybrid decode path.
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    module._omlx_exact_hybrid_projection = True
    module._compiled_forward = lambda h: pytest.fail(
        "compiled single-token path must not run"
    )
    x = mx.random.normal((1, 1, WIDTH)).astype(mx.bfloat16)
    assert hc_fused.compatible(module, x)
    calls = []
    original = hc_fused.fused_forward
    monkeypatch.setattr(
        hc_fused, "fused_forward", lambda m, h: calls.append(h.shape) or original(m, h)
    )
    out = module(x)
    mx.eval(out)
    assert calls == [(1, 1, WIDTH)]


def test_compatible_fails_closed():
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    ok = mx.random.normal((1, 4, WIDTH)).astype(mx.bfloat16)
    if mx.metal.is_available():
        assert hc_fused.compatible(module, ok)
    assert not hc_fused.compatible(
        module, mx.random.normal((1, 17, WIDTH)).astype(mx.bfloat16)
    )
    assert not hc_fused.compatible(
        module, mx.random.normal((2, 9, WIDTH)).astype(mx.bfloat16)
    )
    assert not hc_fused.compatible(
        module, mx.random.normal((1, 4, WIDTH)).astype(mx.float16)
    )
    assert not hc_fused.compatible(
        module, mx.random.normal((4, WIDTH)).astype(mx.bfloat16)
    )
    module.input_inject_weight = nn.Linear(WIDTH, LOWRANK + HC, bias=False)
    assert not hc_fused.compatible(module, ok)
    del module.input_inject_weight
    module.input_mix_weight_down = nn.Linear(WIDTH, LOWRANK, bias=False)
    assert not hc_fused.compatible(module, ok)
    # The layout verdict is cached per module; replacing a weight tensor must re-check it.
    module = _module(4)
    if mx.metal.is_available():
        assert hc_fused.compatible(module, ok)
    module.input_mix_weight_up.scales = module.input_mix_weight_up.scales.astype(
        mx.float16
    )
    assert not hc_fused.compatible(module, ok)


def test_kill_switch_disables_fused_path(monkeypatch):
    monkeypatch.setenv("OMLX_QWEN4_HC_FUSED", "0")
    from mlx_vlm.models.qwen4_exp import hc_fused

    reloaded = importlib.reload(hc_fused)
    try:
        assert not reloaded.enabled()
        assert not reloaded.write_enabled()
        assert not reloaded.compatible(
            _module(4), mx.random.normal((1, 4, WIDTH)).astype(mx.bfloat16)
        )
    finally:
        monkeypatch.delenv("OMLX_QWEN4_HC_FUSED")
        importlib.reload(hc_fused)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize(
    "two_launch,source_name",
    [
        (False, "_N_SOURCE"),
        (False, "_D_SOURCE"),
        (False, "_U_SOURCE"),
        (True, "_NDN_SOURCE"),
        (True, "_U2_SOURCE"),
    ],
)
@pytest.mark.parametrize("use_combine", [False, True])
def test_lazy_compilation_failure_returns_canonical_output(
    monkeypatch, caplog, two_launch, source_name, use_combine
):
    from mlx_vlm.models.qwen4_exp import hc_fused

    monkeypatch.setattr(hc_fused, "_DECODE_V2", two_launch)
    monkeypatch.setattr(hc_fused, "_KERNELS", {})
    monkeypatch.setattr(hc_fused, "_VALIDATED", set())
    monkeypatch.setattr(hc_fused, "_FAILURE_LOGGED", False)
    monkeypatch.setattr(
        hc_fused,
        source_name,
        getattr(hc_fused, source_name) + "\nintentional_compile_error;\n",
    )
    module = _module(4, use_combine, hidden=64)
    x = mx.ones((1, 1, HC * 64), dtype=mx.bfloat16)
    expected = module._forward(x)
    mx.eval(expected)
    with caplog.at_level("WARNING", logger=hc_fused.logger.name):
        actual = module(x)
        mx.eval(actual)
        repeated = module(x)
        mx.eval(repeated)
    assert not hc_fused._VALIDATED
    assert hc_fused.enabled()
    if not use_combine:
        actual, expected, repeated = [actual], [expected], [repeated]
    for value, reference, again in zip(actual, expected, repeated):
        assert mx.array_equal(value, reference).item()
        assert mx.array_equal(again, reference).item()
    messages = [r for r in caplog.records if "failed closed" in r.getMessage()]
    assert len(messages) == 1
    assert "intentional_compile_error" in messages[0].getMessage()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_specializations_validate_once_and_keep_warm_calls_lazy(monkeypatch):
    from mlx_vlm.models.qwen4_exp import hc_fused

    monkeypatch.setattr(hc_fused, "_VALIDATED", set())
    # Cover each varying template input, including mixed projection bit widths.
    for hidden, rows, down_bits, up_bits, inject_bits in [
        (64, 1, 4, 4, 4),
        (64, 4, 4, 4, 4),
        (128, 4, 4, 4, 4),
        (128, 4, 5, 4, 4),
        (128, 4, 5, 6, 4),
        (128, 4, 5, 6, 8),
        (128, 4, 5, 6, None),
    ]:
        module = _module(down_bits, inject_bits is not None, hidden=hidden)
        module.input_mix_weight_up = _module(up_bits, hidden=hidden).input_mix_weight_up
        if inject_bits is not None:
            module.block_inject_weight = _module(
                inject_bits, hidden=hidden
            ).block_inject_weight
        hc_fused._eps_array(module)
        x = mx.ones((1, rows, HC * hidden), dtype=mx.bfloat16)
        mx.eval(x)
        with monkeypatch.context() as patch:
            evaluate = Mock(wraps=mx.eval)
            patch.setattr(mx, "eval", evaluate)
            first = module(x)
            second = module(x)
            assert evaluate.call_count == 1
        mx.eval(first, second)
    assert len(hc_fused._VALIDATED) == 7


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("group_size", [32, 64])
@pytest.mark.parametrize("bits", [4, 5])
@pytest.mark.parametrize("rows", [17, 2048])
@pytest.mark.parametrize("use_combine", [True, False])
def test_prefill_path_matches_canonical(bits, rows, use_combine, group_size, monkeypatch):
    """Prefill fuses the stream norm and, with inject weights, the mix/inject tail."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(bits, use_combine, group_size=group_size)
    x = (mx.random.normal((1, rows, WIDTH)) * 2).astype(mx.bfloat16)
    mx.eval(x)
    assert not hc_fused.compatible(module, x)
    assert hc_fused.prefill_compatible(module, x)
    tail = Mock(wraps=hc_fused._tail)
    monkeypatch.setattr(hc_fused, "_tail", tail)
    out = hc_fused.prefill_forward(module, x)
    assert out is not None
    assert tail.called is not use_combine
    canon = module._forward(x)
    if use_combine:
        mixed, passthrough, inject = out
        canon_mixed, _, canon_inject = canon
        assert passthrough is x
        assert inject.shape == canon_inject.shape == (1, rows, HC)
        assert _ulps(inject, canon_inject)[0] <= 4
    else:
        mixed, canon_mixed = out, canon
    assert mixed.shape == canon_mixed.shape == (1, rows, HIDDEN)
    assert mixed.dtype == mx.bfloat16
    max_ulps, mean_ulps = _ulps(mixed, canon_mixed)
    assert max_ulps <= 16 and mean_ulps <= 0.5


# The prefill tail/inject arithmetic as originally shipped: one row per threadgroup, the
# normed row read from device memory for both the mix and the inject dot. Kernel changes
# must keep these bits (prefix caches and accuracy baselines depend on them).
_REFERENCE_TAIL_INJECT = r"""
    const uint row = threadgroup_position_in_grid.z;
    const uint t = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const device T* up_r = up + (size_t)row * K;
    const device T* xn_r = xn + (size_t)row * K;
    for (int h = int(t); h < H; h += 256) {
        float acc = 0.0f;
        for (int s = 0; s < HC; ++s) {
            const int n = s * H + h;
            const T g = T(1.0f / (1.0f + metal::exp(-float(up_r[n]))));
            const T p = T(float(g) * float(xn_r[n]));
            acc = s == 0 ? float(p) : float(T(acc + float(p)));
        }
        mixed[(size_t)row * H + h] = T(acc * (1.0f / float(HC)));
    }
    constexpr int PF = hc_pack_factor<BITS_I>();
    constexpr int BP = hc_bytes_per_pack<BITS_I>();
    constexpr int ROW_BYTES = K * BP / PF;
    constexpr int GROUPS = K / GS_I;
    constexpr int PER = K / 256;
    float res[HC] = {0.0f};
    float xv[PF];
    const int e0 = int(t) * PER;
    for (int e = e0; e < e0 + PER; e += PF) {
        const float sum = hc_load_vector<T, PF, BITS_I>(xn_r + e, xv);
        const int g = e / GS_I;
        for (int r = 0; r < HC; ++r) {
            res[r] += hc_qdot<PF, BITS_I>(
                (const device uint8_t*)inject_w + r * ROW_BYTES + e * BP / PF,
                xv, float(inject_s[r * GROUPS + g]), float(inject_b[r * GROUPS + g]),
                sum);
        }
    }
    threadgroup float part[HC][8];
    for (int r = 0; r < HC; ++r) {
        const float v = simd_sum(res[r]);
        if (lane == 0) part[r][sg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < HC) {
        float v = 0.0f;
        for (int i = 0; i < 8; ++i) v += part[t][i];
        const float q = float(T(float(T(v)) / float(HC)));
        const float gate = float(T(1.0f / (1.0f + metal::exp(-q))));
        inj[(size_t)row * HC + t] = T(2.0f * gate);
    }
"""


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("group_size", [32, 64])
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
@pytest.mark.parametrize("rows,seed", [(17, 0), (300, 1), (300, 2)])
def test_prefill_tail_inject_keeps_reference_bits(bits, rows, seed, group_size):
    from mlx_vlm.models.qwen4_exp import hc_fused

    mx.random.seed(900 + seed * 10 + bits + group_size)
    module = _module(bits, group_size=group_size)
    inject = module.block_inject_weight
    up = (mx.random.normal((rows, WIDTH)) * 3).astype(mx.bfloat16)
    normed = (mx.random.normal((rows, WIDTH)) * 2).astype(mx.bfloat16)

    def run(name, source, header):
        return hc_fused._kernel(
            name,
            ["up", "xn", "inject_w", "inject_s", "inject_b"],
            ["mixed", "inj"],
            source,
            header=header,
        )(
            inputs=[up, normed, inject.weight, inject.scales, inject.biases],
            template=[
                ("T", mx.bfloat16),
                ("BITS_I", bits),
                ("GS_I", group_size),
                ("K", WIDTH),
                ("H", HIDDEN),
                ("HC", HC),
            ],
            grid=(256, 1, rows),
            threadgroup=(256, 1, 1),
            output_shapes=[(rows, HIDDEN), (rows, HC)],
            output_dtypes=[mx.bfloat16, mx.bfloat16],
        )

    expected = run(
        "test_reference_tail_inject", _REFERENCE_TAIL_INJECT, hc_fused._HEADER
    )
    actual = run(
        "omlx_qwen4_hc_prefill_tail_inject",
        hc_fused._TI_SOURCE,
        hc_fused._HEADER + hc_fused._TG_HEADER,
    )
    mx.eval(expected, actual)
    for observed, reference in zip(actual, expected):
        assert mx.array_equal(observed.view(mx.uint16), reference.view(mx.uint16)).item()


def test_prefill_activation_rounds_like_eager_ops():
    from mlx_vlm.models.qwen4_exp import hc_fused

    mx.random.seed(5)
    special = mx.array(
        [0.0, -0.0, 1e-40, -1e-40, 3e38, -3e38, float("inf"), -float("inf"), float("nan")]
    )
    for scale in (1e-38, 1e-3, 1.0, 40.0, 1e5):
        y = mx.concatenate([mx.random.normal((4096,)) * scale, special]).astype(
            mx.bfloat16
        ).reshape(1, -1, 1)
        expected = nn.silu(y / HC)
        actual = hc_fused._act(HC)(y)
        mx.eval(expected, actual)
        assert mx.array_equal(actual.view(mx.uint16), expected.view(mx.uint16)).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("group_size", [32, 64])
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
@pytest.mark.parametrize(
    "shape", [(1, 1), (1, 2), (1, 3), (1, 4), (2, 2), (1, 16), (1, 17), (1, 2048)]
)
@pytest.mark.parametrize("use_combine", [True, False])
def test_pending_write_matches_eager_write(bits, shape, use_combine, group_size):
    """The write-norm kernels store the eager residual bit for bit and normalize the same bits."""
    from mlx_vlm.models.qwen4_exp import hc_fused, language

    mx.random.seed(11 + 31 * bits + 7 * shape[0] + shape[1] + group_size)
    module = _module(bits, use_combine, group_size=group_size)
    hyper = (mx.random.normal((*shape, WIDTH)) * 2).astype(mx.bfloat16)
    branch = mx.random.normal((*shape, HIDDEN)).astype(mx.bfloat16)
    gate = (2 * mx.sigmoid(mx.random.normal((*shape, HC)))).astype(mx.bfloat16)
    written = language._hc_write(hyper, branch, gate)
    decode = shape[0] * shape[1] <= hc_fused.MAX_ROWS
    forward = hc_fused.fused_forward if decode else hc_fused.prefill_forward
    expected = forward(module, written)
    actual = forward(module, hyper, (branch, gate))
    expected = (
        (expected[0], written, expected[2]) if use_combine else (expected, written)
    )
    assert actual is not None and len(actual) == len(expected)
    mx.eval(actual, expected)
    for observed, reference in zip(actual, expected):
        assert observed.shape == reference.shape
        assert mx.array_equal(
            observed.view(mx.uint16), reference.view(mx.uint16)
        ).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("rows", [2, 64])
def test_promoting_pending_write_falls_back_to_eager_write(rows):
    # An FP32 branch promotes the eager write; the BF16 kernels must not take it.
    from mlx_vlm.models.qwen4_exp import language

    mx.random.seed(rows)
    module = _module(6)
    hyper = mx.random.normal((1, rows, WIDTH)).astype(mx.bfloat16)
    branch = mx.random.normal((1, rows, HIDDEN))
    gate = (2 * mx.sigmoid(mx.random.normal((1, rows, HC)))).astype(mx.bfloat16)
    expected = module(language._hc_write(hyper, branch, gate))
    actual = module(hyper, write=(branch, gate))
    mx.eval(actual, expected)
    assert actual[1].dtype == mx.float32
    for observed, reference in zip(actual, expected):
        assert mx.array_equal(observed, reference).item()


def test_prefill_path_not_offered_for_fused_rows():
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    x = mx.zeros((1, hc_fused.MAX_ROWS, WIDTH), dtype=mx.bfloat16)
    assert not hc_fused.prefill_compatible(module, x)


def test_prefill_path_kill_switch(monkeypatch):
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    x = mx.zeros((1, 64, WIDTH), dtype=mx.bfloat16)
    assert hc_fused.prefill_compatible(module, x)
    monkeypatch.setattr(hc_fused, "_DISABLED", True)
    assert not hc_fused.prefill_compatible(module, x)


def test_module_routes_prefill_rows_through_the_prefill_path(monkeypatch):
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    x = mx.zeros((1, 64, WIDTH), dtype=mx.bfloat16)
    calls = []
    monkeypatch.setattr(hc_fused, "prefill_forward", lambda m, h: calls.append(h.shape) or "prefill")
    assert module(x) == "prefill"
    assert calls == [(1, 64, WIDTH)]
    assert module(x, target_verify=True) != "prefill"


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("path", ["prefill", "decode"])
def test_transient_failure_preserves_other_models_and_recovers(monkeypatch, path):
    from mlx_vlm.models.qwen4_exp import hc_fused

    failed = _module(4, hidden=64)
    other = _module(8, hidden=64)
    decode = mx.ones((1, 4, HC * 64), dtype=mx.bfloat16)
    inputs = mx.ones((1, 32 if path == "prefill" else 4, HC * 64), dtype=mx.bfloat16)
    expected = failed._forward(inputs)
    mx.eval(expected)
    hook = "_tail" if path == "prefill" else "_kernel_norm_down"
    with monkeypatch.context() as fault:
        fault.setattr(
            hc_fused, hook, Mock(side_effect=RuntimeError("transient failure"))
        )
        actual = failed(inputs)
        mx.eval(actual)
    for value, reference in zip(actual, expected):
        assert mx.array_equal(value, reference).item()

    assert hc_fused.compatible(other, decode)
    assert hc_fused.compatible(failed, decode)
    if path == "prefill":
        assert hc_fused.prefill_compatible(failed, inputs)
    monkeypatch.setattr(
        failed, "_forward", Mock(side_effect=AssertionError("Unexpected fallback"))
    )
    monkeypatch.setattr(
        other, "_forward", Mock(side_effect=AssertionError("Unexpected fallback"))
    )
    mx.eval(failed(inputs))
    mx.eval(other(decode, target_verify=True))


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("group_size", [32, 64])
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
@pytest.mark.parametrize("batch,length", [(1, 3), (2, 2), (2, 8)])
def test_fused_rows_match_independent_singletons(bits, batch, length, group_size):
    from mlx_vlm.models.qwen4_exp import hc_fused

    mx.random.seed(3770 + bits + group_size)
    module = _module(bits, group_size=group_size)
    x = mx.random.normal((batch, length, WIDTH)).astype(mx.bfloat16)
    actual, _, actual_injection = hc_fused.fused_forward(module, x)
    singletons = [
        hc_fused.fused_forward(module, row.reshape(1, 1, WIDTH))
        for row in x.reshape(-1, WIDTH)
    ]
    expected = mx.concatenate([row[0] for row in singletons], axis=1).reshape(
        batch, length, HIDDEN
    )
    expected_injection = mx.concatenate([row[2] for row in singletons], axis=1).reshape(
        batch, length, HC
    )
    mx.eval(actual, expected, actual_injection, expected_injection)
    assert mx.array_equal(actual, expected).item()
    assert mx.array_equal(actual_injection, expected_injection).item()

needs_metal = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
needs_nax = pytest.mark.skipif(
    not (mx.metal.is_available() and is_nax_available()),
    reason="requires Metal tensor units (M5)",
)


# Small quantized Qwen4ExpModel for model-level prefill checks.


def _deferred_model():
    """Three decoder layers (the middle, linear one carries PLE) whose hyper-connections
    take the fused prefill kernels: hc_count 4, 64-aligned widths, 4-bit
    group-64 projections."""
    from mlx_vlm.models.qwen4_exp.language import LanguageModel

    config = _tiny_config()
    config.text_config = dataclasses.replace(
        config.text_config,
        hidden_size=512,
        hc_count=4,
        hc_lowrank=64,
        ple_embed_dim=512,
        num_hidden_layers=3,
        layer_types=["linear_attention", "linear_attention", "full_attention"],
        ple_layer_ids=[2],
    )
    mx.random.seed(5)
    model = LanguageModel(config.text_config, config)
    nn.quantize(
        model,
        group_size=64,
        bits=4,
        class_predicate=lambda path, module: isinstance(module, nn.Linear)
        and "hyper_connection" in path,
    )
    # Checkpoint-like quantization statistics.
    for _, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            module.scales = (
                mx.abs(mx.random.normal(module.scales.shape)) * 0.01 + 0.002
            ).astype(mx.bfloat16)
            module.biases = (mx.random.normal(module.biases.shape) * 0.005).astype(
                mx.bfloat16
            )
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model, config


def _logits(model, inputs):
    out = model(inputs)
    out = getattr(out, "logits", out)
    mx.eval(out)
    return out


# Tensor-unit (NAX) prefill: bitwise parity with the MLX matmul path.


def _nax_module(bits_down: int, bits_up: int, bits_inject: int | None, seed: int):
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpGatedResidual, Qwen4ExpRMSNorm

    mx.random.seed(seed)
    module = Qwen4ExpGatedResidual.__new__(Qwen4ExpGatedResidual)
    nn.Module.__init__(module)
    module.hc_count, module.hidden_size, module.hc_lowrank = HC, HIDDEN, LOWRANK
    module.hc_norm = Qwen4ExpRMSNorm(WIDTH, group_size=HIDDEN, eps=1e-6)
    module.hc_norm.weight = (mx.random.normal((WIDTH,)) * 0.05).astype(mx.bfloat16)
    module.input_mix_weight_down = nn.QuantizedLinear(
        WIDTH, LOWRANK, bias=False, group_size=64, bits=bits_down
    )
    module.input_mix_weight_up = nn.QuantizedLinear(
        LOWRANK, WIDTH, bias=False, group_size=64, bits=bits_up
    )
    if bits_inject is not None:
        module.block_inject_weight = nn.QuantizedLinear(
            WIDTH, HC, bias=False, group_size=64, bits=bits_inject
        )
    for name in ("input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"):
        projection = getattr(module, name, None)
        if projection is not None:
            # Checkpoint-like statistics.
            projection.scales = (
                mx.abs(mx.random.normal(projection.scales.shape)) * 0.01 + 0.002
            ).astype(mx.bfloat16)
            projection.biases = (
                mx.random.normal(projection.biases.shape) * 0.005
            ).astype(mx.bfloat16)
    mx.eval(module.parameters())
    return module


def _inputs(rows: int, write: bool):
    hyper = (mx.random.normal((1, rows, WIDTH)) * 2).astype(mx.bfloat16)
    pending = None
    if write:
        branch = mx.random.normal((1, rows, HIDDEN)).astype(mx.bfloat16)
        gate = (2 * mx.sigmoid(mx.random.normal((1, rows, HC)))).astype(mx.bfloat16)
        pending = (branch, gate)
    mx.eval(hyper, pending)
    return hyper, pending


def _bits(a: mx.array) -> mx.array:
    return a.view(mx.uint16) if a.dtype == mx.bfloat16 else a


def _mlx_path(monkeypatch, module, hyper, write):
    from mlx_vlm.models.qwen4_exp import hc_fused

    with monkeypatch.context() as patch:
        patch.setattr(hc_fused, "_NAX_DISABLED", True)
        out = hc_fused.prefill_forward(module, hyper, write)
        mx.eval(out)
    return out


def _spies(monkeypatch):
    from mlx_vlm.models.qwen4_exp import hc_prefill_nax

    spies = {}
    for name in ("norm_inject", "down_silu", "up_tail"):
        spies[name] = Mock(wraps=getattr(hc_prefill_nax, name))
        monkeypatch.setattr(hc_prefill_nax, name, spies[name])
    return spies


@needs_nax
@pytest.mark.parametrize("bits", [(4, 4, 4), (5, 5, 5), (6, 6, 6), (8, 8, 8), (4, 6, 5)])
@pytest.mark.parametrize("rows", [65, 3265, 4133])
@pytest.mark.parametrize("write", [False, True])
def test_prefill_is_bit_identical_to_mlx_path(monkeypatch, bits, rows, write):
    """Mixed input, written stream and inject weights match the six-dispatch path
    bit for bit; the down projection runs on the NAX kernel from 3265 rows on."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _nax_module(*bits, seed=rows + 7 * bits[0])
    hyper, write_args = _inputs(rows, write)
    reference = _mlx_path(monkeypatch, module, hyper, write_args)
    spies = _spies(monkeypatch)
    out = hc_fused.prefill_forward(module, hyper, write_args)
    mx.eval(out)
    assert spies["up_tail"].call_count == 1
    assert spies["down_silu"].call_count == (1 if rows >= 3265 else 0)
    mixed, passthrough, injection = out
    assert mixed.shape == (1, rows, HIDDEN) and injection.shape == (1, rows, HC)
    if not write:
        assert passthrough is hyper
    for observed, expected in zip(out, reference):
        assert observed.dtype == expected.dtype == mx.bfloat16
        assert mx.array_equal(_bits(observed), _bits(expected)).item()


@needs_nax
def test_batched_prefill_is_bit_identical(monkeypatch):
    """[batch, seq] rows flatten like MLX's non-batched quantized matmul."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _nax_module(4, 5, 6, seed=9)
    hyper = (mx.random.normal((2, 2048, WIDTH)) * 2).astype(mx.bfloat16)
    branch = mx.random.normal((2, 2048, HIDDEN)).astype(mx.bfloat16)
    gate = (2 * mx.sigmoid(mx.random.normal((2, 2048, HC)))).astype(mx.bfloat16)
    reference = _mlx_path(monkeypatch, module, hyper, (branch, gate))
    spies = _spies(monkeypatch)
    out = hc_fused.prefill_forward(module, hyper, (branch, gate))
    mx.eval(out)
    assert spies["down_silu"].call_count == 1
    assert out[0].shape == (2, 2048, HIDDEN) and out[2].shape == (2, 2048, HC)
    for observed, expected in zip(out, reference):
        assert mx.array_equal(_bits(observed), _bits(expected)).item()


def test_plain_qmm_nax_mirrors_mlx_dispatch():
    from mlx_vlm.models.qwen4_exp.hc_prefill_nax import plain_qmm_nax

    # Up projection (N = 10240): single 64-row tiles use MLX's split K.
    assert not plain_qmm_nax(64, WIDTH)
    assert plain_qmm_nax(65, WIDTH)
    # Down projection (N = 320): release wheels split K below 801 rows and
    # M5 source builds split on the tensor-unit path below 3265 rows.
    assert not plain_qmm_nax(800, LOWRANK)
    assert not plain_qmm_nax(3264, LOWRANK)
    assert plain_qmm_nax(3265, LOWRANK)
    assert plain_qmm_nax(8191, LOWRANK)


@needs_nax
def test_few_rows_and_missing_inject_keep_mlx_path(monkeypatch):
    from mlx_vlm.models.qwen4_exp import hc_fused

    spies = _spies(monkeypatch)
    module = _nax_module(4, 4, 4, seed=1)
    hyper, _ = _inputs(64, False)
    mx.eval(hc_fused.prefill_forward(module, hyper))
    no_inject = _nax_module(4, 4, None, seed=2)
    hyper, _ = _inputs(128, False)
    mx.eval(hc_fused.prefill_forward(no_inject, hyper))
    assert not spies["norm_inject"].called


@needs_nax
def test_nax_prefill_kill_switch(monkeypatch):
    from mlx_vlm.models.qwen4_exp import hc_fused

    spies = _spies(monkeypatch)
    monkeypatch.setattr(hc_fused, "_NAX_DISABLED", True)
    module = _nax_module(4, 4, 4, seed=4)
    hyper, _ = _inputs(128, False)
    mx.eval(hc_fused.prefill_forward(module, hyper))
    assert not spies["norm_inject"].called


@needs_nax
def test_group_size_32_keeps_mlx_prefill_path(monkeypatch):
    """The tensor-unit tile loop steps K one 64-wide group at a time."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    spies = _spies(monkeypatch)
    mx.random.seed(6)
    module = _module(4, group_size=32)
    hyper, _ = _inputs(128, False)
    assert hc_fused.prefill_compatible(module, hyper)
    assert not hc_fused._nax_prefill_ok(module, 128, WIDTH, module.block_inject_weight)
    mx.eval(hc_fused.prefill_forward(module, hyper))
    assert not spies["norm_inject"].called


@needs_nax
def test_kernel_failure_falls_back_to_mlx_path(monkeypatch):
    from mlx_vlm.models.qwen4_exp import hc_fused, hc_prefill_nax

    module = _nax_module(4, 4, 4, seed=5)
    hyper, write = _inputs(128, True)
    reference = _mlx_path(monkeypatch, module, hyper, write)
    monkeypatch.setattr(hc_fused, "_NAX_BROKEN", False)
    monkeypatch.setattr(
        hc_prefill_nax, "up_tail", Mock(side_effect=RuntimeError("compile"))
    )
    out = hc_fused.prefill_forward(module, hyper, write)
    mx.eval(out)
    for observed, expected in zip(out, reference):
        assert mx.array_equal(_bits(observed), _bits(expected)).item()
    # Later calls do not retry the broken kernels.
    assert hc_fused._NAX_BROKEN
    assert not hc_fused._nax_prefill_ok(module, 128, WIDTH, module.block_inject_weight)


@needs_nax
def test_first_call_probe_keeps_mlx_path_on_mismatch(monkeypatch):
    """A specialization whose first result differs from the MLX path in any
    bit (e.g. an MLX with different quantized-matmul arithmetic) is replaced
    by the MLX result and the NAX kernels are not used again."""
    from mlx_vlm.models.qwen4_exp import hc_fused, hc_prefill_nax

    module = _nax_module(4, 4, 4, seed=6)
    hyper, write = _inputs(128, True)
    reference = _mlx_path(monkeypatch, module, hyper, write)
    real = hc_prefill_nax.up_tail

    def off_by_one_ulp(*args, **kwargs):
        mixed, inj = real(*args, **kwargs)
        return (mixed.view(mx.uint16) ^ 1).view(mx.bfloat16), inj

    monkeypatch.setattr(hc_fused, "_VALIDATED", set())
    monkeypatch.setattr(hc_fused, "_NAX_BROKEN", False)
    monkeypatch.setattr(hc_prefill_nax, "up_tail", off_by_one_ulp)
    out = hc_fused.prefill_forward(module, hyper, write)
    mx.eval(out)
    for observed, expected in zip(out, reference):
        assert mx.array_equal(_bits(observed), _bits(expected)).item()
    assert hc_fused._NAX_BROKEN


@needs_nax
def test_first_call_probe_accepts_bitwise_results(monkeypatch):
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _nax_module(5, 6, 8, seed=7)
    hyper, write = _inputs(3300, True)
    monkeypatch.setattr(hc_fused, "_VALIDATED", set())
    monkeypatch.setattr(hc_fused, "_NAX_BROKEN", False)
    mx.eval(hc_fused.prefill_forward(module, hyper, write))
    assert not hc_fused._NAX_BROKEN
    assert any(sig[0] == "prefill_nax" for sig in hc_fused._VALIDATED)


@needs_nax
def test_model_prefill_logits_are_bit_identical(monkeypatch):
    """A small quantized model's prefill logits with the NAX kernels in every
    hyper-connection (and deferred writes) match the MLX path bit for bit."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    model, config = _deferred_model()
    inputs = mx.random.randint(0, config.text_config.vocab_size, (1, 520))
    with monkeypatch.context() as patch:
        patch.setattr(hc_fused, "_NAX_DISABLED", True)
        reference = _logits(model, inputs)
    spies = _spies(monkeypatch)
    observed = _logits(model, inputs)
    # 3 layers x 2 hyper-connections; the final mixer has no inject weights.
    assert spies["up_tail"].call_count == 6
    view = mx.uint32 if observed.dtype == mx.float32 else mx.uint16
    assert mx.array_equal(observed.view(view), reference.view(view)).item()

def _decode_both_ways(hc_fused, monkeypatch, module, hyper, write):
    """fused_forward outputs from the three-launch and the two-launch decode kernels.

    The two-launch kernels run at every decode row count here, including the rows
    that normally keep the three-launch kernels for speed.
    """
    monkeypatch.setattr(hc_fused, "_V2_MAX_ROWS", hc_fused.MAX_ROWS)
    outputs = []
    for two_launch in (False, True):
        monkeypatch.setattr(hc_fused, "_DECODE_V2", two_launch)
        out = hc_fused.fused_forward(module, hyper, write)
        assert out is not None
        outputs.append(out if isinstance(out, tuple) else (out,))
    mx.eval(outputs)
    return outputs


def _assert_same_bits(outputs):
    three_launch, two_launch = outputs
    assert len(two_launch) == len(three_launch)
    for new, old in zip(two_launch, three_launch):
        assert new.shape == old.shape and new.dtype == old.dtype
        assert mx.array_equal(new.view(mx.uint16), old.view(mx.uint16)).item()


def _two_launch_fits(hc_fused, monkeypatch, module, rows):
    monkeypatch.setattr(hc_fused, "_DECODE_V2", True)
    monkeypatch.setattr(hc_fused, "_V2_MAX_ROWS", hc_fused.MAX_ROWS)
    return hc_fused._decode_v2(
        rows,
        HC,
        module.hidden_size,
        LOWRANK,
        module.input_mix_weight_up.bits,
        "block_inject_weight" in module,
    )


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
@pytest.mark.parametrize("use_combine", [True, False])
def test_two_launch_decode_keeps_three_launch_bits(bits, use_combine, monkeypatch):
    """Mixed, written residual and injection match the three-launch kernels bit for bit at
    every decode row count, with and without a pending write (checkpoint shapes)."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    rows_list = range(1, hc_fused.MAX_ROWS + 1) if bits == 6 else (1, 2, 3, 5, 16)
    for seed in (0, 1, 2) if bits == 6 else (0,):
        mx.random.seed(4017 + 97 * seed + bits)
        module = _module(bits, use_combine)
        for rows in rows_list:
            assert _two_launch_fits(hc_fused, monkeypatch, module, rows)
            hyper = (mx.random.normal((1, rows, WIDTH)) * 2).astype(mx.bfloat16)
            branch = mx.random.normal((1, rows, HIDDEN)).astype(mx.bfloat16)
            gate = (2 * mx.sigmoid(mx.random.normal((1, rows, HC)))).astype(mx.bfloat16)
            for write in (None, (branch, gate)):
                _assert_same_bits(
                    _decode_both_ways(hc_fused, monkeypatch, module, hyper, write)
                )


# 64: every down block is a partial tail; 768/1152/1344: partial final down, inject
# and norm blocks (see test_fused_matches_canonical_path_at_other_hidden_sizes).
@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
@pytest.mark.parametrize("hidden", [64, 1152, 1344])
def test_two_launch_decode_keeps_bits_with_partial_blocks(hidden, bits, monkeypatch):
    from mlx_vlm.models.qwen4_exp import hc_fused

    mx.random.seed(4018 + hidden + bits)
    module = _module(bits, True, hidden=hidden)
    for rows in (1, 3, 16):
        assert _two_launch_fits(hc_fused, monkeypatch, module, rows)
        hyper = (mx.random.normal((1, rows, HC * hidden)) * 2).astype(mx.bfloat16)
        branch = mx.random.normal((1, rows, hidden)).astype(mx.bfloat16)
        gate = (2 * mx.sigmoid(mx.random.normal((1, rows, HC)))).astype(mx.bfloat16)
        for write in (None, (branch, gate)):
            _assert_same_bits(_decode_both_ways(hc_fused, monkeypatch, module, hyper, write))


# Norm/down threadgroup (g, ks) stores strip g of stream ks, so a stream needs no
# more 256-element strips than there are row groups: with two down rows per
# simdgroup (rows >= 3) that is 320 / 16 + 1 = 21, so 5376 is the widest hidden
# size and 5440 must keep the three-launch kernels.
@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_two_launch_decode_strip_boundary(monkeypatch):
    from mlx_vlm.models.qwen4_exp import hc_fused

    mx.random.seed(4019)
    widest = _module(6, True, hidden=5376)
    assert _two_launch_fits(hc_fused, monkeypatch, widest, 3)
    hyper = (mx.random.normal((1, 3, HC * 5376)) * 2).astype(mx.bfloat16)
    _assert_same_bits(_decode_both_ways(hc_fused, monkeypatch, widest, hyper, None))
    too_wide = _module(6, True, hidden=5440)
    assert not _two_launch_fits(hc_fused, monkeypatch, too_wide, 3)
    assert _two_launch_fits(hc_fused, monkeypatch, too_wide, 1)
    hyper = (mx.random.normal((1, 3, HC * 5440)) * 2).astype(mx.bfloat16)
    _assert_same_bits(_decode_both_ways(hc_fused, monkeypatch, too_wide, hyper, None))


def _probe(source: str, epilogue: str, replacement: str) -> str:
    """A kernel source whose epilogue stores the FP32 value it would have rounded."""
    assert epilogue in source
    return source.replace(epilogue, replacement)


_D_ACT_EPILOGUE = """            v = v / float(HC);
            act[(size_t)row * R + int(tg) * 8 + int(sg) * 4 + r]
                = T(v / (1.0f + metal::exp(-v)));
"""
_D_INJ_EPILOGUE = """            v = v / float(HC);
            inj[(size_t)row * HC + r] = T(2.0f / (1.0f + metal::exp(-v)));
"""
_U_EPILOGUE = """    float gate = 1.0f / (1.0f + metal::exp(-acc));
    float v = gate * float(xn[(size_t)r * K + n]);
    v += simd_shuffle_down(v, 1);
    v += simd_shuffle_down(v, 2);
    if (s == 0) mixed[(size_t)r * H + h] = T(v / float(HC));
"""
_U2_EPILOGUE = """    float gate = 1.0f / (1.0f + metal::exp(-acc));
    float v = gate * xnv;
    v += simd_shuffle_down(v, 1);
    v += simd_shuffle_down(v, 2);
    if (s == 0) mixed[(size_t)r * H + h] = T(v / float(HC));
    if (inj_thread) inj[(size_t)r * HC + int(t)] = injv;
"""
_U2_ACT_COMBINE = """        v = v / float(HC);
        acts[i] = T(v / (1.0f + metal::exp(-v)));
"""
_U2_INJ_COMBINE = """        v = v / float(HC);
        injv = T(2.0f / (1.0f + metal::exp(-v)));
"""
_N_INV = "    const float inv = metal::rsqrt(tot / float(H) + eps[0]);\n"
_NDN_INV = "    const float inv = metal::rsqrt(tot / float(H) + e);\n"


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("group_size", [32, 64])
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
@pytest.mark.parametrize("rows", [1, 3, 16])
def test_two_launch_decode_keeps_three_launch_fp32_sums(bits, rows, group_size, monkeypatch):
    """The BF16 outputs absorb most one-ulp FP32 changes (reversing the up sum order
    changed no output bit over 128 rows), so compare the FP32 values each path rounds:
    the stream sums of squares, the down and inject slice sums
    (part[0] + part[1] + part[2] + part[3]) and the up accumulators before the gate."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    monkeypatch.setattr(hc_fused, "_V2_MAX_ROWS", hc_fused.MAX_ROWS)
    mx.random.seed(4020 + 10 * bits + rows + group_size)
    module = _module(bits, True, group_size=group_size)
    down, up = module.input_mix_weight_down, module.input_mix_weight_up
    inject = module.block_inject_weight
    flat = (mx.random.normal((rows, WIDTH)) * 2).astype(mx.bfloat16)
    norm_inputs = [flat, module.hc_norm.weight, hc_fused._eps_array(module)]

    _, normed, parts = hc_fused._kernel_norm_down(
        module, flat, None, rows, HC, HIDDEN, LOWRANK, mx.bfloat16, down, inject
    )
    rps = hc_fused._down_rps(rows)
    new_tots = mx.fast.metal_kernel(
        name="test_hc_norm_down_sums",
        input_names=["x", "w", "eps", "down_w", "down_s", "down_b", "inject_w", "inject_s", "inject_b"],
        output_names=["xn", "parts", "tots"],
        header=hc_fused._V2_HEADER,
        source=_probe(
            hc_fused._NDN_SOURCE,
            _NDN_INV,
            _NDN_INV + "    if (t == 0 && g == 0) tots[(size_t)row * HC + s] = tot;\n",
        ),
    )(
        inputs=[*norm_inputs, down.weight, down.scales, down.biases, inject.weight, inject.scales, inject.biases],
        template=[
            ("T", mx.bfloat16),
            ("BITS_D", bits),
            ("BITS_I", bits),
            ("GS_D", group_size),
            ("GS_I", group_size),
            ("K", WIDTH),
            ("H", HIDDEN),
            ("R", LOWRANK),
            ("HC", HC),
            ("INJ", 1),
            ("RPS", rps),
        ],
        grid=(256, HC * hc_fused._down_row_groups(LOWRANK, True, rps), rows),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, WIDTH), (rows, HC, LOWRANK + HC), (rows, HC)],
        output_dtypes=[mx.bfloat16, mx.float32, mx.float32],
    )[2]
    old_normed, old_tots = mx.fast.metal_kernel(
        name="test_hc_norm_sums",
        input_names=["x", "w", "eps"],
        output_names=["xn", "tots"],
        source=_probe(
            hc_fused._N_SOURCE,
            _N_INV,
            _N_INV + "    if (t == 0) tots[(size_t)row * 4 + s] = tot;\n",
        ),
    )(
        inputs=norm_inputs,
        template=[("T", mx.bfloat16), ("K", WIDTH), ("H", HIDDEN)],
        grid=(256, HC, rows),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, WIDTH), (rows, HC)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )
    mx.eval(normed, old_normed, new_tots, old_tots)
    assert mx.array_equal(new_tots.view(mx.uint32), old_tots.view(mx.uint32)).item()
    assert mx.array_equal(normed.view(mx.uint16), old_normed.view(mx.uint16)).item()
    down_inputs = [
        old_normed,
        down.weight,
        down.scales,
        down.biases,
        inject.weight,
        inject.scales,
        inject.biases,
    ]
    down_names = ["xn", "down_w", "down_s", "down_b", "inject_w", "inject_s", "inject_b"]
    down_launch = dict(
        template=[
            ("T", mx.bfloat16),
            ("BITS_D", bits),
            ("BITS_I", bits),
            ("GS_D", group_size),
            ("GS_I", group_size),
            ("K", WIDTH),
            ("R", LOWRANK),
            ("HC", HC),
            ("INJ", 1),
        ],
        grid=(32, 8 * (LOWRANK // 8 + 1), rows),
        threadgroup=(32, 8, 1),
    )
    probe = _probe(
        _probe(
            hc_fused._D_SOURCE,
            _D_ACT_EPILOGUE,
            "            act[(size_t)row * R + int(tg) * 8 + int(sg) * 4 + r] = v;\n",
        ),
        _D_INJ_EPILOGUE,
        "            inj[(size_t)row * HC + r] = v;\n",
    )
    down_sums, inject_sums = mx.fast.metal_kernel(
        name="test_hc_down_sums",
        input_names=down_names,
        output_names=["act", "inj"],
        header=hc_fused._HEADER,
        source=probe,
    )(
        inputs=down_inputs,
        output_shapes=[(rows, LOWRANK), (rows, HC)],
        output_dtypes=[mx.float32, mx.float32],
        **down_launch,
    )
    old_act, _ = mx.fast.metal_kernel(
        name="test_hc_down",
        input_names=down_names,
        output_names=["act", "inj"],
        header=hc_fused._HEADER,
        source=hc_fused._D_SOURCE,
    )(
        inputs=down_inputs,
        output_shapes=[(rows, LOWRANK), (rows, HC)],
        output_dtypes=[mx.bfloat16, mx.bfloat16],
        **down_launch,
    )

    up_template = [
        ("T", mx.bfloat16),
        ("BITS_U", bits),
        ("GS_U", group_size),
        ("K", WIDTH),
        ("R", LOWRANK),
        ("HC", HC),
        ("H", HIDDEN),
    ]
    store_acc = "    mixed[(size_t)r * K + n] = acc;\n"
    old_accs = mx.fast.metal_kernel(
        name="test_hc_up_sums",
        input_names=["xn", "act", "up_w", "up_s", "up_b"],
        output_names=["mixed"],
        header=hc_fused._HEADER,
        source=_probe(hc_fused._U_SOURCE, _U_EPILOGUE, store_acc),
    )(
        inputs=[old_normed, old_act, up.weight, up.scales, up.biases],
        template=up_template,
        grid=(256, HIDDEN // 64, rows),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, WIDTH)],
        output_dtypes=[mx.float32],
    )[0]
    chunks = hc_fused._up_chunks(bits, LOWRANK)
    threads = chunks * (hc_fused._V2_UP_THREADS // chunks)
    outputs = hc_fused._V2_UP_OUTPUTS
    up2_probe = _probe(
        _probe(
            _probe(hc_fused._U2_SOURCE, _U2_EPILOGUE, store_acc),
            _U2_ACT_COMBINE,
            "        if (threadgroup_position_in_grid.y == 0) sums[(size_t)r * PR + i] = v;\n"
            + _U2_ACT_COMBINE,
        ),
        _U2_INJ_COMBINE,
        "        sums[(size_t)r * PR + i] = v;\n" + _U2_INJ_COMBINE,
    )
    new_accs, _, new_sums = mx.fast.metal_kernel(
        name="test_hc_up2_sums",
        input_names=["xn", "parts", "up_w", "up_s", "up_b"],
        output_names=["mixed", "inj", "sums"],
        header=hc_fused._V2_HEADER,
        source=up2_probe,
    )(
        inputs=[normed, parts, up.weight, up.scales, up.biases],
        template=[*up_template, ("INJ", 1), ("NO", outputs), ("TGS", threads)],
        grid=(threads, WIDTH // outputs, rows),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows, WIDTH), (rows, HC), (rows, LOWRANK + HC)],
        output_dtypes=[mx.float32, mx.bfloat16, mx.float32],
    )

    mx.eval(new_sums, down_sums, inject_sums, old_accs, new_accs)
    for new, old in (
        (new_sums[:, :LOWRANK], down_sums),
        (new_sums[:, LOWRANK:], inject_sums),
        (new_accs, old_accs),
    ):
        assert mx.array_equal(new.view(mx.uint32), old.view(mx.uint32)).item()


# Deferred residual writes: a real-shape stack must match eager writes bit for bit.
# Each layer's tail write is carried into the next hyper-connection norm (and the
# final mixer); the reference runs with OMLX_QWEN4_HC_FUSED_WRITE off.


def _stack(seed: int):
    """Four layers (DeltaNet, DeltaNet+PLE, sparse attention, DeltaNet) at the checkpoint's HC shapes."""
    from mlx_vlm.models import qwen4_exp
    from mlx_vlm.models.qwen4_exp import language

    text = qwen4_exp.TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=2560,
        num_hidden_layers=4,
        num_attention_heads=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        num_experts=16,
        num_experts_per_tok=10,
        shared_expert_intermediate_size=64,
        moe_intermediate_size=64,
        rms_norm_eps=1e-6,
        vocab_size=256,
        num_key_value_heads=2,
        max_position_embeddings=65536,
        hc_count=4,
        hc_lowrank=320,
        head_dim=64,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
        ],
        ple_layer_ids=[2],
        ple_embed_dim=64,
        ple_conv_kernel_size=3,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=64,
        indexer_budget=64,
        indexer_compress_ratio=4,
        eos_token_id=1,
        rope_parameters={
            "rope_type": "default",
            "mrope_section": [16, 8, 8],
            "rope_theta": 10_000,
            "partial_rotary_factor": 1.0,
        },
    )
    vision = qwen4_exp.VisionConfig(
        model_type="qwen4_exp",
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        out_hidden_size=32,
        num_heads=4,
        patch_size=14,
        in_channels=3,
        spatial_merge_size=2,
        temporal_patch_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )
    config = qwen4_exp.ModelConfig(
        text_config=text,
        vision_config=vision,
        model_type="qwen4_exp",
        image_token_id=250,
        video_token_id=251,
        vision_start_token_id=252,
        vision_end_token_id=253,
        vocab_size=256,
    )
    mx.random.seed(seed)
    model = language.LanguageModel(text, config)
    head = language.Qwen4ExpMTPModule(text)
    # Checkpoint layout: 6-bit layer HC with one 8-bit layer, 5-bit final mixer.
    for module, layout in (
        (
            model,
            [
                ("model.layers.3.", 8),
                ("model.layers.", 6),
                ("model.hyper_connection_mixer", 5),
            ],
        ),
        (head, [("", 6)]),
    ):
        module.set_dtype(mx.bfloat16)
        for name, sub in module.named_modules():
            if name.endswith("hc_norm"):
                sub.weight = (mx.random.normal(sub.weight.shape) * 0.05).astype(
                    mx.bfloat16
                )
        for prefix, bits in layout:
            nn.quantize(
                module,
                group_size=64,
                bits=bits,
                class_predicate=lambda path, m, prefix=prefix: isinstance(m, nn.Linear)
                and path.startswith(prefix)
                and "hyper_connection" in path,
            )
        mx.eval(module.parameters())
    return model, head


def _run(monkeypatch, model, head, seed, deferred, script):
    from mlx_vlm.models.qwen4_exp import hc_fused, language

    monkeypatch.setattr(hc_fused, "_WRITE_DISABLED", not deferred)
    eager_writes = []
    real_write = language._hc_write
    monkeypatch.setattr(
        language,
        "_hc_write",
        lambda *args: eager_writes.append(args[0].shape) or real_write(*args),
    )
    rng = np.random.default_rng(seed)
    outputs = script(
        model,
        head,
        lambda batch, length: mx.array(
            rng.integers(2, 256, (batch, length)), dtype=mx.int32
        ),
    )
    mx.eval(outputs)
    return outputs, eager_writes


def _assert_identical(monkeypatch, seed, script):
    model, head = _stack(seed)
    # Separate contexts so the second run does not wrap the first run's spy.
    with monkeypatch.context() as patch:
        expected, eager = _run(patch, model, head, seed, False, script)
    with monkeypatch.context() as patch:
        actual, deferred = _run(patch, model, head, seed, True, script)
    # Only the PLE layer still materializes its incoming write.
    assert len(deferred) < len(eager)
    assert len(actual) == len(expected)
    for index, (observed, reference) in enumerate(zip(actual, expected)):
        assert observed.dtype == reference.dtype == mx.bfloat16, index
        assert observed.shape == reference.shape, index
        assert mx.array_equal(
            observed.view(mx.uint16), reference.view(mx.uint16)
        ).item(), f"output {index} differs"


def _decode_and_verify(model, head, ids):
    outputs = []
    cache = model.make_cache()
    outputs.append(model(ids(1, 17), cache=cache).logits)
    for rows in (1, 2, 3, 4):
        outputs.append(model(ids(1, rows), cache=cache).logits)
    for rows in (1, 2, 3, 4):
        # Lightning MTP verify: target-verify rows plus the pre-mixer residual
        # (a one-row window is the decode step itself, with no transaction).
        out = model(ids(1, rows), cache=cache, return_hidden=True)
        hidden = out.hidden_states[0]
        if out.gdn_states is not None:
            out.gdn_states.commit([rows])
        mixed, residual = head(hidden, ids(1, rows), model.model.embed_tokens)
        outputs += [out.logits, hidden, mixed, residual]
    sink = []
    outputs.append(
        model.model(ids(1, 3), cache=cache, hidden_sink=sink, capture_layer_ids=[0, 2])
    )
    outputs += sink
    batch = model.make_cache()
    outputs.append(model(ids(2, 9), cache=batch).logits)
    for rows in (1, 2):
        outputs.append(model(ids(2, rows), cache=batch).logits)
    return outputs


@needs_metal
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_decode_and_verify_rows_match_eager_writes(monkeypatch, seed):
    _assert_identical(monkeypatch, seed, _decode_and_verify)


@needs_metal
@pytest.mark.parametrize("chunk", [17, 257, 2048])
def test_prefill_chunks_match_eager_writes(monkeypatch, chunk):
    def prefill(model, head, ids):
        cache = model.make_cache()
        first = model(ids(1, chunk), cache=cache).logits
        second = model(ids(1, chunk), cache=cache).logits
        return [first, second, model(ids(1, 1), cache=cache).logits]

    _assert_identical(monkeypatch, 40 + chunk, prefill)
