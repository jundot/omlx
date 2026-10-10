# SPDX-License-Identifier: Apache-2.0
"""Routing order must stay exact when inversion stops using a second sort."""

import mlx.core as mx
import pytest
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

from omlx.patches.moe_routes import sort_routes


@pytest.mark.parametrize("tokens", [1, 8, 127, 511, 512, 513, 1024, 4097])
@pytest.mark.parametrize("routing", ["random", "ties", "skewed"])
def test_routes_match_stock_and_restore_original_rows(tokens, routing):
    x = mx.arange(tokens * 16, dtype=mx.float32).reshape(1, tokens, 1, 1, 16)
    if routing == "ties":
        indices = mx.full((1, tokens, 8), 17, dtype=mx.uint32)
    else:
        experts = 32 if routing == "skewed" else 256
        indices = mx.random.randint(
            0, experts, (1, tokens, 8), key=mx.random.key(tokens)
        )
    reference_x, reference_idx, reference_inv = _gather_sort(x, indices)
    x_tok, row_map, idx, inverse = sort_routes(x, indices)
    assert mx.array_equal(x_tok[row_map], reference_x).item()
    assert mx.array_equal(idx, reference_idx).item()
    assert mx.array_equal(inverse, reference_inv).item()
    restored = _scatter_unsort(x_tok[row_map], inverse, indices.shape)
    expected = mx.broadcast_to(x.reshape(1, tokens, 1, 1, 16), (1, tokens, 8, 1, 16))
    assert mx.array_equal(restored, expected).item()


@pytest.mark.parametrize("shape", [(1, 513, 8), (2, 256, 8), (2, 257, 8)])
def test_compiled_routes_match_stock_for_batches_and_tails(shape):
    x = mx.random.normal((*shape[:2], 1, 1, 16), key=mx.random.key(17))
    indices = mx.random.randint(0, 256, shape, key=mx.random.key(18))
    reference_x, reference_idx, reference_inv = _gather_sort(x, indices)
    x_tok, row_map, idx, inverse = mx.compile(sort_routes)(x, indices)
    assert mx.array_equal(x_tok[row_map], reference_x).item()
    assert mx.array_equal(idx, reference_idx).item()
    assert mx.array_equal(inverse, reference_inv).item()
