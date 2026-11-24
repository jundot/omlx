# SPDX-License-Identifier: Apache-2.0
"""Compare short sliding verification against the installed MLX SDPA path."""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from omlx.patches.hy_v4 import layers, verify_dispatch  # noqa: E402


@pytest.mark.parametrize("rows", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("full_window", [False, True])
def test_sliding_values_match_stock_with_mask_and_sinks(monkeypatch, rows, full_window):
    if not mx.metal.is_available() or not verify_dispatch._supported_gpu():
        pytest.skip("requires the validated MLX/Metal backend")

    with mx.stream(mx.gpu):
        mx.random.seed(2049 + rows)
        keys = 512 if full_window else 511 + rows

        def normal(shape):
            return (mx.random.normal(shape) * 0.33).astype(mx.bfloat16)

        q = normal((1, 64, rows, 512))
        q_pe = normal((1, 64, rows, 64))
        latent = normal((1, 1, keys, 512))
        k_pe = normal((1, 1, keys, 64))
        sinks = normal((64,))
        columns = mx.arange(keys)[None, :]
        row_end = mx.arange(keys - rows, keys)[:, None]
        mask = (columns >= (row_end - 511)) & (columns != 7) & (columns != 21)
        attention = SimpleNamespace(
            kind="sliding", embed_q=lambda value: value, unembed_out=lambda value: value,
            scale=256**-0.5,
        )
        inputs = attention, q, q_pe, latent, k_pe, None, mask[None, None], sinks
        monkeypatch.setattr(verify_dispatch, "eligible", lambda _: False)
        expected = layers.Attention._verify(*inputs)

        called = []
        original = layers._sliding_shared_values

        def shared(*args):
            called.append(True)
            return original(*args)

        monkeypatch.setattr(layers, "_sliding_shared_values", shared)
        monkeypatch.setattr(verify_dispatch, "eligible", lambda _: True)
        actual = layers.Attention._verify(*inputs)

        assert called == [True]
        assert mx.array_equal(expected.view(mx.uint16), actual.view(mx.uint16)).item()
