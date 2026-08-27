"""Exact-shape guards for the DS4 B1 native router selector."""

import mlx.core as mx
import pytest

from omlx.custom_kernels.glm_moe_dsa import fast

SYMBOL = "ds4_router_topk_indices"


def test_router_topk_symbol_is_registered():
    assert SYMBOL in fast.NATIVE_SYMBOLS
    assert callable(fast.ds4_router_topk_indices)


@pytest.mark.skipif(
    not fast.has_symbol(SYMBOL),
    reason="extension predates the DS4 router selector",
)
@pytest.mark.parametrize(
    "scores",
    (
        mx.random.normal((8, 256)).astype(mx.float32),
        mx.zeros((2, 256), dtype=mx.float32),
        mx.concatenate([mx.ones((2, 128)), mx.zeros((2, 128))], -1).astype(
            mx.float32
        ),
    ),
)
def test_router_topk_matches_argpartition_order(scores):
    expected = mx.argpartition(-scores, kth=5, axis=-1)[..., :6]
    actual = fast.ds4_router_topk_indices(mx.contiguous(scores))
    mx.eval(expected, actual)
    assert bool(mx.array_equal(expected, actual).item())


@pytest.mark.skipif(
    not fast.has_symbol(SYMBOL),
    reason="extension predates the DS4 router selector",
)
def test_router_topk_rejects_non_contract_shapes():
    with pytest.raises(ValueError, match="FP32.*256"):
        fast.ds4_router_topk_indices(mx.zeros((1, 255), dtype=mx.float32))
