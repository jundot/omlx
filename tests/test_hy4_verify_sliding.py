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
            kind="sliding",
            embed_q=lambda value: value,
            unembed_out=lambda value: value,
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


@pytest.mark.parametrize("cursor", [139, 512])
@pytest.mark.parametrize("rows,accepted", [(2, 1), (4, 1), (6, 4)])
def test_sliding_append_and_undo_use_the_stock_layout(cursor, rows, accepted):
    if not mx.metal.is_available() or not verify_dispatch._supported_gpu():
        pytest.skip("requires the validated MLX/Metal backend")

    from mlx_lm.models.cache import RotatingKVCache

    from omlx.cache._rotating_subclass import PrefillReadyRotatingKVCache
    from omlx.patches.hy_v4.verify_layers import _append, _sliding_layout
    from omlx.patches.mlx_lm_mtp import cache_rollback

    cache_rollback.apply()
    window = PrefillReadyRotatingKVCache(512)
    baseline = RotatingKVCache(512)
    mx.random.seed(813 + rows + cursor)

    def normal(count, dims):
        return mx.random.normal((1, 1, count, dims)).astype(mx.bfloat16)

    keys, values = normal(512, 512), normal(512, 64)
    new_keys, new_values = normal(rows, 512), normal(rows, 64)
    for item in (window, baseline):
        item.keys, item.values, item.offset, item._idx = keys, values, 4096, cursor
    assert _sliding_layout(window) == (4096, cursor, 512)

    def compare():
        assert (window.offset, window._idx) == (baseline.offset, baseline._idx)
        for a, b in zip((window.keys, window.values), (baseline.keys, baseline.values)):
            assert mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item()

    start = mx.array([(cursor + 1) % 512], dtype=mx.uint32)
    baseline.update_and_fetch(new_keys, new_values)
    cache_rollback.set_undo_armed(True)
    try:
        cache_rollback.stage_functional_rotating_update(
            window, new_keys, new_values, decode_consistent=False
        )
    finally:
        cache_rollback.set_undo_armed(False)
    window.keys, window.values = _append(keys, values, new_keys, new_values, start)
    window.offset += rows
    window._idx = window.keys.shape[2]
    compare()

    assert window.trim(rows - accepted) == rows - accepted
    baseline.keys, baseline.values, baseline.offset, baseline._idx = (
        keys,
        values,
        4096,
        cursor,
    )
    baseline.update_and_fetch(new_keys[:, :, :accepted], new_values[:, :, :accepted])
    compare()
