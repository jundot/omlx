"""Grouped small-row quantized linears: bit-exact per row, one dispatch per signature."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches import qwen35_grouped_linears as grouped

K = 2560


def _qlinear(rows, bits, group_size, seed):
    lin = nn.Linear(K, rows, bias=False)
    lin.weight = (mx.random.normal((rows, K), key=mx.random.key(seed)) * 0.05).astype(
        mx.bfloat16
    )
    return nn.QuantizedLinear.from_linear(lin, group_size=group_size, bits=bits)


@pytest.mark.parametrize("T", [2, 3, 4, 8])
@pytest.mark.parametrize(
    "sigs",
    [
        [(10240, 4, 64), (6144, 5, 128), (48, 5, 128), (48, 5, 128)],
        [(12288, 8, 32), (512, 8, 32), (512, 8, 32)],
        [(640, 8, 128), (640, 8, 128)],
        [(4096, 6, 64), (512, 6, 64), (512, 6, 64), (48, 6, 64)],
    ],
)
def test_grouped_matches_separate(T, sigs):
    linears = [_qlinear(r, b, g, i) for i, (r, b, g) in enumerate(sigs)]
    x = mx.random.normal((1, T, K), key=mx.random.key(99)).astype(mx.bfloat16)
    got = grouped.grouped_quantized_linears(linears, x)
    exp = tuple(lin(x) for lin in linears)
    mx.eval(*got, *exp)
    assert len(got) == len(exp)
    for a, b in zip(got, exp):
        assert a.shape == b.shape
        assert mx.array_equal(a, b).item()


def test_one_dispatch_per_signature(monkeypatch):
    linears = [
        _qlinear(10240, 4, 64, 0),
        _qlinear(6144, 5, 128, 1),
        _qlinear(48, 5, 128, 2),
        _qlinear(48, 5, 128, 3),
    ]
    x = mx.random.normal((1, 4, K)).astype(mx.bfloat16)
    grouped.grouped_quantized_linears(linears, x)  # build the group cache first
    calls = []
    orig = mx.quantized_matmul

    def counting(*a, **kw):
        calls.append(kw.get("bits"))
        return orig(*a, **kw)

    monkeypatch.setattr(mx, "quantized_matmul", counting)
    out = grouped.grouped_quantized_linears(linears, x)
    mx.eval(*out)
    assert sorted(calls) == [4, 5]


def test_group_cache_is_not_a_module_parameter():
    linears = [_qlinear(64, 4, 64, 0), _qlinear(64, 4, 64, 1)]
    x = mx.zeros((1, 2, K), dtype=mx.bfloat16)
    grouped.grouped_quantized_linears(linears, x)
    from mlx.utils import tree_flatten

    names = [name for name, _ in tree_flatten(linears[0].parameters())]
    assert not any(grouped._CACHE_ATTR in name for name in names)


def test_declines_batch_wide_rows_and_non_quantized():
    linears = [_qlinear(64, 4, 64, 0), _qlinear(64, 4, 64, 1)]
    assert grouped.grouped_quantized_linears(linears, mx.zeros((2, 1, K), dtype=mx.bfloat16)) is None
    assert grouped.grouped_quantized_linears(linears, mx.zeros((1, 1, K), dtype=mx.bfloat16)) is None
    assert grouped.grouped_quantized_linears(linears, mx.zeros((1, 9, K), dtype=mx.bfloat16)) is None
    mixed = [linears[0], nn.Linear(K, 64, bias=False)]
    assert grouped.grouped_quantized_linears(mixed, mx.zeros((1, 2, K), dtype=mx.bfloat16)) is None


def test_routing_mismatch_falls_back_to_members(monkeypatch):
    from omlx.patches import qwen35_verify_qmm as vk

    linears = [_qlinear(12288, 8, 32, 0), _qlinear(512, 8, 32, 1), _qlinear(512, 8, 32, 2)]
    x = mx.random.normal((1, 4, K)).astype(mx.bfloat16)
    monkeypatch.setattr(vk, "_is_armed", lambda: True)
    monkeypatch.setattr(vk, "_MIN_ROUTE_N", 13000)  # group N=13312 routes, members do not
    grouped.grouped_quantized_linears(linears, x)
    calls = []
    orig = mx.quantized_matmul

    def counting(*a, **kw):
        calls.append(a[1].shape[0])
        return orig(*a, **kw)

    monkeypatch.setattr(mx, "quantized_matmul", counting)
    out = grouped.grouped_quantized_linears(linears, x)
    mx.eval(*out)
    assert calls == [12288, 512, 512]


def test_patch_rebinds_vendored_qwen4_module(monkeypatch):
    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
    )

    apply_mlx_vlm_qwen4_exp_compat_patch()
    import mlx_vlm.models.qwen3_5.language as q35
    import mlx_vlm.models.qwen4_exp.language as q4

    monkeypatch.setattr(grouped, "_PATCHED", False)
    assert grouped.apply_qwen35_grouped_linears_patch()
    assert q4._target_verify_linears is q35._target_verify_linears
    assert q35._target_verify_linears.__name__ == "patched"
