# SPDX-License-Identifier: Apache-2.0
"""Row-exact verify projections: every row equals a one-row decode call.

Lightning MTP greedy output matches MTP-off output only if each verify row's
quantized projection has the serial step's bits; stock multi-row kernels
(qmv_wide, qmm) do not.
"""

from __future__ import annotations

import inspect

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches import qwen35_verify_qmm, row_exact_qmv

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


@pytest.fixture(autouse=True)
def disarm():
    row_exact_qmv._UNROLLED_VERDICTS.clear()
    yield
    qwen35_verify_qmm.set_verify_qmm_armed(False)
    row_exact_qmv._UNROLLED_VERDICTS.clear()


def _linear(k, n, bits, group_size, seed):
    mx.random.seed(seed)
    linear = nn.QuantizedLinear(k, n, bias=False, group_size=group_size, bits=bits)
    weight = (mx.random.normal((n, k)) * 0.05).astype(mx.bfloat16)
    linear.weight, linear.scales, linear.biases = mx.quantize(
        weight, group_size=group_size, bits=bits
    )
    return linear


def _serial(linear, x):
    return mx.concatenate([linear(x[:, r : r + 1]) for r in range(x.shape[1])], axis=1)


def _bit_equal(a, b):
    return mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item()


# (K, N, bits, group_size): Qwen4 oQ5e projections (qmv_fast), the non-fast
# qmv arm (K off the qmv_fast block), partial output tiles (N % 8) and N < 8,
# other widths, and K = 256 (mod 512) at 6/8 bits, where MLX's qmv_fast block
# is 256 wide (512 at 4/5 bits).
SHAPES = [
    (2560, 10240, 6, 64),
    (6144, 2560, 5, 128),
    (2560, 248320, 8, 64),
    (640, 2560, 8, 128),
    (2560, 1, 8, 64),
    (2560, 12, 6, 64),
    (320, 1024, 6, 64),
    (2560, 1024, 4, 64),
    (2560, 1024, 5, 32),
    (768, 1024, 8, 64),
    (1280, 1024, 6, 64),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("rows", [3, 8, 16])
def test_rows_equal_one_row_quantized_matmul(shape, rows):
    k, n, bits, group_size = shape
    linear = _linear(k, n, bits, group_size, seed=k + n + rows)
    x = mx.random.normal((1, rows, k)).astype(mx.bfloat16)
    assert _bit_equal(row_exact_qmv.quantized_linear(linear, x), _serial(linear, x))


@pytest.mark.parametrize(
    "k, sizes, bits, group_size",
    [
        (2560, [10240, 6144, 48, 48], 6, 64),  # DeltaNet in_proj qkv/z/b/a
        (2560, [12288, 512, 512], 6, 64),  # attention q/k/v
        (2560, [640, 640], 8, 128),  # shared expert gate/up
        (640, [2560, 1, 12], 8, 128),  # non-fast K with partial tiles
    ],
)
@pytest.mark.parametrize("rows", [2, 4])
def test_grouped_rows_equal_one_row_quantized_matmul(k, sizes, bits, group_size, rows):
    linears = [_linear(k, n, bits, group_size, seed=i + k) for i, n in enumerate(sizes)]
    x = mx.random.normal((1, rows, k)).astype(mx.bfloat16)
    outputs = row_exact_qmv.quantized_linears(linears, x)
    assert len(outputs) == len(linears)
    for linear, output in zip(linears, outputs):
        assert _bit_equal(output, _serial(linear, x))


def test_row_exact_arming_routes_multi_row_quantized_linear():
    qwen35_verify_qmm.apply_verify_qmm_patch()
    linear = _linear(2560, 6144, 6, 64, seed=3)
    x = mx.random.normal((1, 4, 2560)).astype(mx.bfloat16)
    reference = _serial(linear, x)
    qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=True)
    routed = linear(x)
    qwen35_verify_qmm.set_verify_qmm_armed(False)
    assert _bit_equal(routed, reference)


def _quantized(k, n, bits, group_size, dtype, seed):
    mx.random.seed(seed)
    weight = (mx.random.normal((n, k)) * 0.05).astype(dtype)
    return mx.quantize(weight, group_size=group_size, bits=bits)


# FP32 rows expose the unrounded accumulations that a BF16 output hides.
@pytest.mark.parametrize(
    "k, n, bits, group_size",
    [
        (2560, 16480, 6, 64),  # DeltaNet stacked qkv/z/b/a in-projection
        (6144, 2560, 5, 128),  # DeltaNet out-projection
        (2560, 1024, 4, 64),
        (1024, 520, 8, 32),
        (768, 1024, 8, 64),  # 8-bit qmv_fast block is 256 wide
    ],
)
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("rps", [1, 2, 4])
def test_one_row_qmv_equals_quantized_matmul(k, n, bits, group_size, dtype, rps):
    for seed in (5, 23):
        weight, scales, biases = _quantized(k, n, bits, group_size, dtype, seed + rps)
        x = mx.random.normal((1, 1, k)).astype(dtype)
        launch = row_exact_qmv.one_row_qmv(
            weight, scales, biases, bits, group_size, "affine", dtype, rps
        )
        expected = mx.quantized_matmul(
            x, weight, scales, biases, transpose=True, group_size=group_size, bits=bits
        )
        observed = launch(x)
        assert observed.shape == expected.shape and observed.dtype == dtype
        view = mx.uint32 if dtype == mx.float32 else mx.uint16
        assert mx.array_equal(observed.view(view), expected.view(view)).item()


@pytest.mark.parametrize(
    "k, n, rps",
    [
        (640, 2560, 1),  # stock runs qmv, not qmv_fast (K off the 256 block)
        (2560, 12, 1),  # stock runs qmv (N % 8)
        (2560, 40, 8),  # the 16-column tile does not divide N
    ],
)
def test_one_row_qmv_declines_shapes_stock_does_not_run_fast(k, n, rps):
    weight, scales, biases = _quantized(k, n, 8, 64, mx.bfloat16, 1)
    assert (
        row_exact_qmv.one_row_qmv(weight, scales, biases, 8, 64, "affine", mx.bfloat16, rps)
        is None
    )


# Every (columns per simdgroup, rows per threadgroup) tile gives each row the
# one-row bits, so retuning the geometry cannot change a verify row.
@pytest.mark.parametrize(
    "k, n, bits, group_size",
    [
        (2560, 16480, 6, 64),  # DeltaNet stacked qkv/z/b/a in-projection
        (6144, 2560, 5, 128),  # DeltaNet out-projection
        (2560, 1024, 8, 64),
    ],
)
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("rows", [3, 4, 9])
def test_rows_qmv_every_geometry_equals_one_row_quantized_matmul(
    k, n, bits, group_size, dtype, rows
):
    weight, scales, biases = _quantized(k, n, bits, group_size, dtype, k + rows)
    x = mx.random.normal((1, rows, k)).astype(dtype)
    expected = mx.concatenate(
        [
            mx.quantized_matmul(
                x[:, r : r + 1], weight, scales, biases, transpose=True,
                group_size=group_size, bits=bits,
            )
            for r in range(rows)
        ],
        axis=1,
    )
    view = mx.uint32 if dtype == mx.float32 else mx.uint16
    for rps in (1, 2, 4):
        for per_group in (d for d in (1, 2, 3, 4) if rows % d == 0):
            launch = row_exact_qmv.rows_qmv(
                weight, scales, biases, bits, group_size, "affine", dtype,
                lambda _, g=(rps, per_group): g,
            )
            observed = launch(x)
            assert observed.shape == expected.shape and observed.dtype == dtype
            assert mx.array_equal(observed.view(view), expected.view(view)).item(), (
                rps,
                per_group,
            )


# The unrolled tile keeps the rolled tile's per-accumulator order, so every
# tile inside its register envelope gives each row the one-row bits (at the
# DeltaNet verify shapes and bit widths, and past the 4-row blocks).
@pytest.mark.parametrize(
    "k, n, bits, group_size",
    [
        (2560, 16480, 6, 64),  # DeltaNet stacked in-projection (oQ5e)
        (2560, 16480, 8, 64),  # its 8-bit layer
        (6144, 2560, 5, 128),  # DeltaNet out-projection
        (2560, 1024, 4, 64),
    ],
)
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("rows", [2, 3, 4, 5, 6, 7, 8])
def test_unrolled_rows_qmv_tiles_equal_one_row_quantized_matmul(
    k, n, bits, group_size, dtype, rows
):
    weight, scales, biases = _quantized(k, n, bits, group_size, dtype, k + n + rows)
    x = mx.random.normal((1, rows, k)).astype(dtype)
    expected = mx.concatenate(
        [
            mx.quantized_matmul(
                x[:, r : r + 1], weight, scales, biases, transpose=True,
                group_size=group_size, bits=bits,
            )
            for r in range(rows)
        ],
        axis=1,
    )
    view = mx.uint32 if dtype == mx.float32 else mx.uint16
    tiles = [
        (rps, per_group)
        for rps in (1, 2, 4, 8)
        for per_group in range(1, rows + 1)
        if rows % per_group == 0 and row_exact_qmv.unrolled_tile_ok(bits, rps, per_group)
    ]
    for tile in tiles:
        launch = row_exact_qmv.rows_qmv(
            weight, scales, biases, bits, group_size, "affine", dtype,
            lambda _, g=tile: g, unrolled=True,
        )
        observed = launch(x)
        assert observed.shape == expected.shape and observed.dtype == dtype
        assert mx.array_equal(observed.view(view), expected.view(view)).item(), tile


# Tiles past the envelope compiled to wrong bits (5-bit 8x4, 4-bit 2x8).
@pytest.mark.parametrize("k, n, bits, group_size, tile", [
    (6144, 2560, 5, 128, (8, 4)),
    (2560, 1024, 4, 64, (2, 8)),
])
def test_unrolled_rows_qmv_refuses_tiles_past_the_register_envelope(
    k, n, bits, group_size, tile
):
    weight, scales, biases = _quantized(k, n, bits, group_size, mx.bfloat16, 3)
    launch = row_exact_qmv.rows_qmv(
        weight, scales, biases, bits, group_size, "affine", mx.bfloat16,
        lambda _: tile, unrolled=True,
    )
    with pytest.raises(ValueError):
        launch(mx.zeros((1, 8, k), dtype=mx.bfloat16))


def _miscompiled(factory):
    """``factory`` whose unrolled kernels return every output off by one, as a
    GPU whose compiler gets the unrolled tile wrong would."""
    signature = inspect.signature(factory)

    def make(*args, **kwargs):
        kernel = factory(*args, **kwargs)
        if not signature.bind(*args, **kwargs).arguments.get("unrolled", False):
            return kernel
        return lambda **kw: [y + 1 for y in kernel(**kw)]

    return make


# A GPU that computes an unrolled tile wrong (an M3 Ultra did at 4-bit gs64)
# must still get one-row bits: the first launch of each shape is checked and
# falls back to the rolled tile.
def test_rows_qmv_falls_back_when_this_gpu_gets_the_unrolled_tile_wrong(monkeypatch):
    k, n, bits, group_size, rows = 2560, 1024, 4, 64, 4
    weight, scales, biases = _quantized(k, n, bits, group_size, mx.bfloat16, 5)
    x = mx.random.normal((1, rows, k)).astype(mx.bfloat16)
    expected = mx.concatenate(
        [
            mx.quantized_matmul(
                x[:, r : r + 1], weight, scales, biases, transpose=True,
                group_size=group_size, bits=bits,
            )
            for r in range(rows)
        ],
        axis=1,
    )
    monkeypatch.setattr(row_exact_qmv, "_kernel", _miscompiled(row_exact_qmv._kernel))
    launch = row_exact_qmv.rows_qmv(
        weight, scales, biases, bits, group_size, "affine", mx.bfloat16,
        lambda _: (2, 2), unrolled=True,
    )
    assert _bit_equal(launch(x), expected)
    assert list(row_exact_qmv._UNROLLED_VERDICTS.values()) == [False]


def test_grouped_tiles_fall_back_when_this_gpu_gets_the_unrolled_tile_wrong(monkeypatch):
    linears = [_linear(2560, n, 4, 64, seed) for seed, n in enumerate((1024, 512))]
    x = mx.random.normal((1, 4, 2560)).astype(mx.bfloat16)
    monkeypatch.setattr(row_exact_qmv, "_group_kernel", _miscompiled(row_exact_qmv._group_kernel))
    outputs = row_exact_qmv.quantized_linears_tiled(linears, x, 2, 4, True)
    for linear, observed in zip(linears, outputs):
        assert _bit_equal(observed, _serial(linear, x))
    assert list(row_exact_qmv._UNROLLED_VERDICTS.values()) == [False]
