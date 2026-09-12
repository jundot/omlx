# SPDX-License-Identifier: Apache-2.0
"""Fused hyper-connection kernels: parity with the canonical path, eligibility, kill switch."""

from __future__ import annotations

import importlib
from unittest.mock import Mock

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat


@pytest.fixture(autouse=True)
def _vendored_qwen4():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()


HC, HIDDEN, LOWRANK = 4, 2560, 320
WIDTH = HC * HIDDEN


def _module(
    bits: int,
    use_combine: bool = True,
    hidden: int = HIDDEN,
    dense_inject: bool = False,
):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpGatedResidual, Qwen4ExpRMSNorm

    width = HC * hidden
    module = Qwen4ExpGatedResidual.__new__(Qwen4ExpGatedResidual)
    nn.Module.__init__(module)
    module.hc_count, module.hidden_size, module.hc_lowrank = HC, hidden, LOWRANK
    module.hc_norm = Qwen4ExpRMSNorm(width, group_size=hidden, eps=1e-6)
    module.hc_norm.weight = (mx.random.normal((width,)) * 0.05).astype(mx.bfloat16)
    module.input_mix_weight_down = nn.QuantizedLinear(
        width, LOWRANK, bias=False, group_size=64, bits=bits
    )
    module.input_mix_weight_up = nn.QuantizedLinear(
        LOWRANK, width, bias=False, group_size=64, bits=bits
    )
    if use_combine:
        if dense_inject:
            module.block_inject_weight = nn.Linear(width, HC, bias=False)
            module.block_inject_weight.weight = (
                module.block_inject_weight.weight.astype(mx.bfloat16)
            )
        else:
            module.block_inject_weight = nn.QuantizedLinear(
                width, HC, bias=False, group_size=64, bits=bits
            )
    for name in ("input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"):
        projection = getattr(module, name, None)
        if isinstance(projection, nn.QuantizedLinear):
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
        if type(q) is nn.Linear:
            return q.weight.astype(mx.float32)
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
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
@pytest.mark.parametrize("rows", [1, 4, 16])
@pytest.mark.parametrize("use_combine", [True, False])
def test_fused_matches_canonical_path(bits, rows, use_combine):
    mx.random.seed(20260905 + bits * 100 + rows)
    _assert_fused_matches_canonical(_module(bits, use_combine), rows, use_combine)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
@pytest.mark.parametrize("rows", [1, 2, 4, 6, 16])
def test_bf16_injection_matches_canonical(bits, rows):
    mx.random.seed(20260912 + bits * 100 + rows)
    _assert_fused_matches_canonical(_module(bits, dense_inject=True), rows, True)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("hidden", [64, 768, 1152, 1344])
def test_bf16_injection_handles_other_hidden_sizes(hidden):
    mx.random.seed(hidden + 123)
    _assert_fused_matches_canonical(
        _module(8, hidden=hidden, dense_inject=True), 4, True
    )


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_bf16_injection_serial_and_verify_rows_agree():
    from mlx_vlm.models.qwen4_exp import hc_fused

    mx.random.seed(20260912)
    module = _module(8, dense_inject=True)
    x = mx.random.normal((1, 6, WIDTH)).astype(mx.bfloat16)
    batched = module(x, target_verify=True)
    serial = [module(x[:, i : i + 1], target_verify=False) for i in range(6)]
    canonical = module._forward(x, target_verify=True)
    mx.eval(batched, serial, canonical)
    assert hc_fused.compatible(module, x)
    for component in (0, 2):
        joined = mx.concatenate([out[component] for out in serial], axis=1)
        assert mx.array_equal(batched[component], joined).item()
    assert _ulps(batched[2], canonical[2])[0] <= 4


@pytest.mark.parametrize(
    "dtype,bias,shape",
    [
        (mx.float32, False, (HC, WIDTH)),
        (mx.float16, False, (HC, WIDTH)),
        (mx.bfloat16, True, (HC, WIDTH)),
        (mx.bfloat16, False, (HC + 1, WIDTH)),
        (mx.bfloat16, False, (HC, WIDTH - 1)),
    ],
)
def test_bf16_injection_rejects_unsupported_layout(dtype, bias, shape):
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(8, dense_inject=True)
    module.block_inject_weight = nn.Linear(shape[1], shape[0], bias=bias)
    module.block_inject_weight.weight = module.block_inject_weight.weight.astype(dtype)
    assert not hc_fused.compatible(module, mx.zeros((1, 1, WIDTH), dtype=mx.bfloat16))


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


def test_kill_switch_disables_fused_path(monkeypatch):
    monkeypatch.setenv("OMLX_QWEN4_HC_FUSED", "0")
    from mlx_vlm.models.qwen4_exp import hc_fused

    reloaded = importlib.reload(hc_fused)
    try:
        assert not reloaded.enabled()
        assert not reloaded.compatible(
            _module(4), mx.random.normal((1, 4, WIDTH)).astype(mx.bfloat16)
        )
    finally:
        monkeypatch.delenv("OMLX_QWEN4_HC_FUSED")
        importlib.reload(hc_fused)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("source_name", ["_N_SOURCE", "_D_SOURCE", "_U_SOURCE"])
@pytest.mark.parametrize("use_combine", [False, True])
def test_lazy_compilation_failure_returns_canonical_output(
    monkeypatch, caplog, source_name, use_combine
):
    from mlx_vlm.models.qwen4_exp import hc_fused

    monkeypatch.setattr(hc_fused, "_KERNELS", {})
    monkeypatch.setattr(hc_fused, "_VALIDATED", set())
    monkeypatch.setattr(hc_fused, "_RUNTIME_FAILED", False)
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
    assert hc_fused._RUNTIME_FAILED
    assert not hc_fused._VALIDATED
    assert not hc_fused.enabled()
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
        (128, 4, 5, 6, 16),
        (128, 4, 5, 6, None),
    ]:
        module = _module(down_bits, inject_bits is not None, hidden=hidden)
        module.input_mix_weight_up = _module(up_bits, hidden=hidden).input_mix_weight_up
        if inject_bits is not None:
            module.block_inject_weight = _module(
                8 if inject_bits == 16 else inject_bits,
                hidden=hidden,
                dense_inject=inject_bits == 16,
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
    assert len(hc_fused._VALIDATED) == 8


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("bits", [4, 5])
@pytest.mark.parametrize("rows", [17, 2048])
@pytest.mark.parametrize("use_combine", [True, False])
def test_prefill_path_matches_canonical(bits, rows, use_combine, monkeypatch):
    """Prefill retains canonical normalization while compiling the stream mean."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(bits, use_combine)
    x = (mx.random.normal((1, rows, WIDTH)) * 2).astype(mx.bfloat16)
    mx.eval(x)
    assert not hc_fused.compatible(module, x)
    assert hc_fused.prefill_compatible(module, x)
    kernel_norm = Mock(side_effect=AssertionError("prefill must use canonical norm"))
    monkeypatch.setattr(hc_fused, "_kernel_norm", kernel_norm)
    out = hc_fused.prefill_forward(module, x)
    kernel_norm.assert_not_called()
    assert out is not None
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


def test_prefill_path_not_offered_for_fused_rows():
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    x = mx.zeros((1, hc_fused.MAX_ROWS, WIDTH), dtype=mx.bfloat16)
    assert not hc_fused.prefill_compatible(module, x)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("rows", [1, 2, 4, 6])
@pytest.mark.parametrize("use_combine", [False, True])
def test_m4_compiled_hc_matches_eager_and_reads_replaced_weights(
    monkeypatch, rows, use_combine
):
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(8, use_combine, dense_inject=use_combine)
    x = mx.random.normal((1, rows, WIDTH)).astype(mx.bfloat16)
    first = None
    for replace_weights in (False, True):
        if replace_weights:
            # The synthetic Q8 weights can saturate the sigmoid. Zeroing the
            # up projection moves every gate to 0.5 and must affect the output.
            up = module.input_mix_weight_up
            up.scales = mx.zeros_like(up.scales)
            up.biases = mx.zeros_like(up.biases)
        monkeypatch.setattr(hc_fused, "_m4_max", lambda: False)
        eager = hc_fused.fused_forward(module, x)
        monkeypatch.setattr(hc_fused, "_m4_max", lambda: True)
        compiled = hc_fused.fused_forward(module, x)
        mx.eval(eager, compiled)
        a = eager if use_combine else [eager]
        b = compiled if use_combine else [compiled]
        assert all(mx.array_equal(v, ref).item() for v, ref in zip(a, b))
        if first is None:
            first = b[0]
        else:
            assert not mx.array_equal(first, b[0]).item()


@pytest.mark.parametrize(
    "m4,rows,down_bits,up_bits,inject_bits,expected",
    [
        (True, 4, 8, 8, 16, True),
        (True, 4, 8, 8, None, True),
        (False, 4, 8, 8, 16, False),
        (True, 7, 8, 8, 16, False),
        (True, 4, 5, 8, 16, False),
        (True, 4, 8, 6, 16, False),
        (True, 4, 8, 8, 8, False),
    ],
)
def test_hc_compilation_stays_within_measured_m4_layout(
    monkeypatch, m4, rows, down_bits, up_bits, inject_bits, expected
):
    from mlx_vlm.models.qwen4_exp import hc_fused

    monkeypatch.setattr(hc_fused, "_m4_max", lambda: m4)
    _, signature = hc_fused._fused_plan(
        mx.bfloat16, 4, 2560, 320, rows, down_bits, up_bits, inject_bits
    )
    assert signature[-1] is expected


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
    monkeypatch.setattr(
        hc_fused, "prefill_forward", lambda m, h: calls.append(h.shape) or "prefill"
    )
    assert module(x) == "prefill"
    assert calls == [(1, 64, WIDTH)]
    assert module(x, target_verify=True) != "prefill"
