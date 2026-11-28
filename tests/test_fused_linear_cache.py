# SPDX-License-Identifier: Apache-2.0
"""The one-row fused in-projection keeps its projections' weights once."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches import qwen35_gdn_prework as prework


def _linears(seed=0):
    mx.random.seed(seed)
    out = []
    for rows in (96, 64, 8, 8):
        lin = nn.Linear(128, rows, bias=False)
        q = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=8)
        out.append(q)
    mx.eval([q.parameters() for q in out])
    return out


def test_cache_rebinds_projections_to_views_of_one_buffer():
    linears = _linears()
    x = mx.random.normal((1, 1, 128))
    ref = [lin(x) for lin in linears]
    mx.eval(ref)
    cached = prework.fused_linear_cache(linears)
    _, weights, scales, biases, split_indices = cached
    assert split_indices == [96, 160, 168]
    assert weights.shape[0] == 176
    # Each projection now reads its rows of the shared buffer and gives the same output.
    for lin, r in zip(linears, ref):
        assert mx.array_equal(lin(x), r)
    # A second call reuses the cache instead of concatenating again.
    assert prework.fused_linear_cache(linears) is cached


def test_mlx_vlm_one_row_fused_decode_uses_the_shared_cache():
    pytest.importorskip("mlx_vlm")
    assert prework._install_fused_linear_cache()
    from mlx_vlm.speculative.ops import linear as vlm_linear

    linears = _linears(1)
    x = mx.random.normal((1, 1, 128))
    ref = [lin(x) for lin in linears]
    mx.eval(ref)
    out = vlm_linear._decode_quantized_linears_fused(linears, x)
    assert out is not None
    for o, r in zip(out, ref):
        assert mx.allclose(o, r, atol=1e-4).item()
    assert linears[0]._fused_decode_linears[1].shape[0] == 176
    # One-row only: a two-row input keeps the four projections.
    assert (
        vlm_linear._decode_quantized_linears_fused(linears, mx.zeros((1, 2, 128)))
        is None
    )


@pytest.mark.skipif(not mx.metal.is_available(), reason="measures Metal memory")
def test_cache_frees_the_original_weights():
    linears = _linears(2)
    mx.eval([lin.weight for lin in linears])
    mx.clear_cache()
    before = mx.get_active_memory()
    prework.fused_linear_cache(linears)
    mx.clear_cache()
    after = mx.get_active_memory()
    # The concatenated copy replaces the originals instead of adding to them.
    assert after - before < sum(lin.weight.nbytes for lin in linears) // 2
