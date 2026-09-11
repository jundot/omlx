import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.deepseek_v41.routing import combine_sorted_experts as combine


@pytest.mark.parametrize(
    "batch,length,width", [(1, 32, 64), (2, 33, 513), (1, 256, 5120)]
)
def test_reference_repeat_and_layout(batch, length, width):
    mx.random.seed(147)
    n = batch * length * 6
    inverse = mx.argsort(mx.random.uniform(shape=(n,)))
    # Noncontiguous storage also exercises the Metal input-copy contract.
    routed = mx.random.normal((n, 1, width * 2)).astype(mx.bfloat16)[..., ::2]
    shared = mx.random.normal((batch, length, width)).astype(mx.bfloat16)
    expected = (
        routed[inverse].reshape(batch, length, 6, width).astype(mx.float32).sum(-2)
        + shared.astype(mx.float32)
    ).astype(mx.bfloat16)
    actual = combine(routed, inverse, shared)
    assert actual is not None and actual.dtype == mx.bfloat16
    np.testing.assert_array_equal(
        actual.astype(mx.float32), expected.astype(mx.float32)
    )
    np.testing.assert_array_equal(
        actual.astype(mx.float32), combine(routed, inverse, shared).astype(mx.float32)
    )


def test_cancellation_and_shared_rounding():
    routed = mx.array([256.0, 1.0, -256.0, 0.125, -0.125, 2.0], mx.bfloat16).reshape(
        6, 1, 1
    )
    shared = mx.array([[[0.25]]], mx.bfloat16)
    actual = combine(routed, mx.arange(6, dtype=mx.uint32), shared)
    assert actual.item() == 3.25


def test_unsupported_layouts():
    x = mx.zeros((6, 1, 4), mx.bfloat16)
    order = mx.arange(6, dtype=mx.uint32)
    s = mx.zeros((1, 1, 4), mx.bfloat16)
    assert combine(x.astype(mx.float32), order, s) is None
    assert combine(x, order.astype(mx.int32), s) is None
    assert combine(x[:3], order[:3], s) is None
    assert combine(x, order, s.astype(mx.float32)) is None
    assert combine(x[:0], order[:0], s[:, :0]) is None


def test_moe_dispatch_preserves_sorted_reference(monkeypatch):
    from mlx.utils import tree_flatten
    from test_deepseek_v41 import tiny

    import omlx.patches.deepseek_v41.language as language

    mx.random.seed(417)
    layer = language.MoE(tiny(n_routed_experts=8, n_activated_experts=6))
    layer.load_weights(
        [
            (name, (mx.random.normal(weight.shape) * 0.1).astype(mx.bfloat16))
            for name, weight in tree_flatten(layer.parameters())
        ]
    )
    x = mx.random.normal((2, 33, 32)).astype(mx.bfloat16)
    calls = []

    def observed(*args):
        result = combine(*args)
        calls.append(result is not None)
        return result

    monkeypatch.setattr(language, "combine_sorted_experts", observed)
    actual = layer(x, None)
    mx.eval(actual)
    assert calls == [True]
    monkeypatch.setattr(language, "combine_sorted_experts", lambda *args: None)
    expected = layer(x, None)
    np.testing.assert_array_equal(
        actual.astype(mx.float32), expected.astype(mx.float32)
    )
