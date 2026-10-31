# SPDX-License-Identifier: Apache-2.0
"""The M5 small-row HC layout keeps FP32 controls and explicit fallbacks."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.deepseek_v41 import language


def _config():
    return SimpleNamespace(hc_mult=4, norm_eps=1e-6, hc_eps=1e-6, hc_sinkhorn_iters=20)


@pytest.mark.parametrize("rows", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("strided", [False, True])
def test_hc_transposed_controls_are_bit_exact(rows, strided):
    if not language._hc_transposed_projection_available():
        pytest.skip("bitwise layout equivalence is qualified on M5 Metal")
    mx.random.seed(1710 + rows)
    if strided:
        x = mx.random.normal((1, rows, 4, 10240))[..., ::2].astype(mx.bfloat16)
        # Preserve a genuinely strided input after the dtype conversion.
        x = mx.concatenate([x, x], axis=-1)[..., ::2]
        fn = mx.random.normal((24, 40960))[:, ::2] * 0.03
        fn = mx.concatenate([fn, fn], axis=-1)[:, ::2]
    else:
        x = mx.random.normal((1, rows, 4, 5120)).astype(mx.bfloat16)
        fn = mx.random.normal((24, 20480)) * 0.03
    scale = mx.array([0.2, 0.3, 0.4])
    base = mx.random.normal((24,)) * 0.1
    args = (x, fn, scale, base, 4, 1e-6, 1e-6, 20)
    expected = language._hc_mixes(*args)
    actual = language._hc_mixes_transposed(*args)
    repeated = language._hc_mixes_transposed(*args)
    for got, want, again in zip(actual, expected, repeated):
        assert got.dtype == mx.float32
        np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(got, again)


@pytest.mark.parametrize("value", [0, 1, -1])
def test_hc_transposed_zero_and_constant_controls(value):
    if not language._hc_transposed_projection_available():
        pytest.skip("bitwise layout equivalence is qualified on M5 Metal")
    x = mx.full((1, 4, 4, 5120), value, mx.bfloat16)
    fn = mx.full((24, 20480), 0.01)
    args = (x, fn, mx.ones((3,)), mx.zeros((24,)), 4, 1e-6, 1e-6, 20)
    for got, want in zip(
        language._hc_mixes_transposed(*args), language._hc_mixes(*args)
    ):
        np.testing.assert_array_equal(got, want)
        assert bool(mx.all(mx.isfinite(got)).item())


def test_hc_transposed_cancellation_preserves_projection_bits():
    if not language._hc_transposed_projection_available():
        pytest.skip("bitwise layout equivalence is qualified on M5 Metal")
    mx.random.seed(1719)
    # Mix very different magnitudes and cancellation across the reduction
    # dimension. Check the projection itself before sigmoid can hide an ULP.
    values = mx.array([2**20, 1, -(2**20), -1], mx.bfloat16)
    x = mx.tile(values, 20480 // 4).reshape(1, 1, 4, 5120)
    x = mx.concatenate([x, -x, x * 0.5], axis=1)
    fn = mx.random.normal((24, 20480)) * 0.03
    flat = x.flatten(-2).astype(mx.float32)
    np.testing.assert_array_equal(
        mx.swapaxes(fn @ mx.swapaxes(flat, -1, -2), -1, -2), flat @ fn.T
    )
    args = (x, fn, mx.ones((3,)), mx.zeros((24,)), 4, 1e-6, 1e-6, 20)
    for got, want in zip(
        language._hc_mixes_transposed(*args), language._hc_mixes(*args)
    ):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize(
    "shape,dtype,fn_shape,fn_dtype,streams,device,available,expected",
    [
        (
            (1, 2, 4, 5120),
            mx.bfloat16,
            (24, 20480),
            mx.float32,
            4,
            mx.gpu,
            True,
            "layout",
        ),
        (
            (1, 8, 4, 5120),
            mx.bfloat16,
            (24, 20480),
            mx.float32,
            4,
            mx.gpu,
            True,
            "layout",
        ),
        (
            (1, 1, 4, 5120),
            mx.bfloat16,
            (24, 20480),
            mx.float32,
            4,
            mx.gpu,
            True,
            "stock",
        ),
        (
            (1, 9, 4, 5120),
            mx.bfloat16,
            (24, 20480),
            mx.float32,
            4,
            mx.gpu,
            True,
            "stock",
        ),
        (
            (2, 2, 4, 5120),
            mx.bfloat16,
            (24, 20480),
            mx.float32,
            4,
            mx.gpu,
            True,
            "stock",
        ),
        (
            (1, 2, 2, 5120),
            mx.bfloat16,
            (24, 10240),
            mx.float32,
            2,
            mx.gpu,
            True,
            "stock",
        ),
        (
            (1, 2, 4, 5120),
            mx.bfloat16,
            (24, 20480),
            mx.float32,
            2,
            mx.gpu,
            True,
            "stock",
        ),
        ((2, 4, 5120), mx.bfloat16, (24, 20480), mx.float32, 4, mx.gpu, True, "stock"),
        ((1, 2, 4, 64), mx.bfloat16, (24, 256), mx.float32, 4, mx.gpu, True, "stock"),
        (
            (1, 2, 4, 5120),
            mx.float16,
            (24, 20480),
            mx.float32,
            4,
            mx.gpu,
            True,
            "stock",
        ),
        (
            (1, 2, 4, 5120),
            mx.float32,
            (24, 20480),
            mx.float32,
            4,
            mx.gpu,
            True,
            "stock",
        ),
        (
            (1, 2, 4, 5120),
            mx.bfloat16,
            (24, 20480),
            mx.float16,
            4,
            mx.gpu,
            True,
            "stock",
        ),
        (
            (1, 2, 4, 5120),
            mx.bfloat16,
            (25, 20480),
            mx.float32,
            4,
            mx.gpu,
            True,
            "stock",
        ),
        (
            (1, 2, 4, 5120),
            mx.bfloat16,
            (24, 20480),
            mx.float32,
            4,
            mx.cpu,
            True,
            "stock",
        ),
        (
            (1, 2, 4, 5120),
            mx.bfloat16,
            (24, 20480),
            mx.float32,
            4,
            mx.gpu,
            False,
            "stock",
        ),
    ],
)
def test_hc_layout_dispatch_contract(
    monkeypatch, shape, dtype, fn_shape, fn_dtype, streams, device, available, expected
):
    # Metadata-only inputs ensure this checks dispatch rather than a tolerance
    # that could accidentally hide the use of the wrong implementation.
    x = SimpleNamespace(ndim=len(shape), shape=shape, dtype=dtype)
    fn = SimpleNamespace(shape=fn_shape, dtype=fn_dtype)
    monkeypatch.setattr(
        language, "_hc_transposed_projection_available", lambda: available
    )
    monkeypatch.setattr(mx, "default_device", lambda: device)
    calls = []

    def record(which):
        def run(*args):
            calls.append(which)
            return which

        return run

    monkeypatch.setattr(language, "_hc_mixes", record("stock"))
    monkeypatch.setattr(language, "_hc_mixes_transposed", record("layout"))
    config = _config()
    config.hc_mult = streams
    assert language.hc_mixes(x, fn, None, None, config) == expected
    assert calls == [expected]


@pytest.mark.parametrize("architecture", ["applegpu_g17s", "applegpu_g15d", "", None])
def test_hc_layout_hardware_scope(monkeypatch, architecture):
    available = language._hc_transposed_projection_available
    available.cache_clear()
    try:
        monkeypatch.setattr(mx, "device_info", lambda: {"architecture": architecture})
        assert available() is (architecture == "applegpu_g17s")
    finally:
        available.cache_clear()


def test_hc_layout_device_query_failure_uses_stock(monkeypatch):
    available = language._hc_transposed_projection_available
    available.cache_clear()
    try:

        def fail():
            raise RuntimeError("Metal device query unavailable")

        monkeypatch.setattr(mx, "device_info", fail)
        assert available() is False
    finally:
        available.cache_clear()


def test_hc_layout_preserves_prefill_fused_projection(monkeypatch):
    x = SimpleNamespace(ndim=4, shape=(1, 256, 4, 5120), dtype=mx.bfloat16)
    fn = SimpleNamespace(shape=(24, 20480), dtype=mx.float32)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(language, "_hc_transposed_projection_available", lambda: True)
    calls = []

    def fused(*args):
        calls.append("fused")
        return "projection"

    def controls(mixes, *args):
        assert mixes == "projection"
        return "controls"

    def unexpected(*args):
        pytest.fail("prefill must retain the existing fused projection")

    monkeypatch.setattr(language, "fused_hc_projection", fused)
    monkeypatch.setattr(language, "_hc_mix_weights", controls)
    monkeypatch.setattr(language, "_hc_mixes_transposed", unexpected)
    monkeypatch.setattr(language, "_hc_mixes", unexpected)
    assert language.hc_mixes(x, fn, None, None, _config()) == "controls"
    assert calls == ["fused"]
