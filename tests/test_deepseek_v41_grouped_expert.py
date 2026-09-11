"""Grouped expert execution preserves quantization and stream ownership."""

import mlx.core as mx
import numpy as np
import pytest

from omlx.custom_kernels.glm_moe_dsa import fast
from omlx.patches.deepseek_v41.activation import quantize_swiglu_activation
from omlx.patches.deepseek_v41.config import ModelConfig
from omlx.patches.deepseek_v41.language import Expert
from omlx.patches.deepseek_v41.quantization import QuantizedProjection

pytestmark = pytest.mark.skipif(
    not fast.has_symbol("deepseek_v41_grouped_expert"),
    reason="Grouped expert extension is not built",
)


def make_graph():
    x = mx.ones((1, 1, 64), mx.bfloat16)
    w, s = mx.quantize(
        mx.ones((2, 64, 64), mx.bfloat16) * 0.03125, group_size=32, bits=4, mode="mxfp4"
    )
    ids = mx.array([0, 1], mx.uint32)

    def q(value, left):
        return mx.gather_qmm(
            value,
            w,
            s,
            lhs_indices=left,
            rhs_indices=ids,
            group_size=32,
            bits=4,
            mode="mxfp4",
        )

    gate = q(x, mx.zeros((2,), mx.uint32))
    up = q(x, mx.zeros((2,), mx.uint32))
    y = quantize_swiglu_activation(gate, up, mx.array([0.3, 0.7]), mx.bfloat16, 10)
    return gate, up, y, q(y, ids)


def test_grouped_expert_async_streams_and_temporary_lifetime():
    streams = [mx.new_stream(mx.gpu), mx.new_stream(mx.gpu)]
    outputs = []
    for i in range(12):
        with mx.stream(streams[i % 2]):
            graph = make_graph()
            grouped = fast.deepseek_v41_grouped_expert(*graph)
            mx.async_eval(grouped)
            outputs.append((grouped, graph[-1]))
        mx.clear_cache()
    for actual, expected in outputs:
        np.testing.assert_array_equal(
            actual.astype(mx.float32), expected.astype(mx.float32)
        )


def test_grouped_expert_rejects_evaluated_graph():
    graph = make_graph()
    mx.eval(*graph)
    with pytest.raises(ValueError, match="unevaluated"):
        fast.deepseek_v41_grouped_expert(*graph)


@pytest.mark.parametrize("length", [1, 2, 4, 6, 8])
@pytest.mark.parametrize(
    "mode,bits",
    [
        ("mxfp4", 4),
        ("affine", 2),
        ("affine", 3),
        ("affine", 4),
        ("affine", 6),
        ("affine", 8),
    ],
)
def test_expert_dispatch_matches_unfused_and_repeats(monkeypatch, length, mode, bits):
    mx.random.seed(847)
    expert = Expert(ModelConfig(dim=64, moe_inter_dim=64, n_routed_experts=4), True)
    for name in ("w1", "w3", "w2"):
        packed = mx.quantize(
            mx.random.normal((4, 64, 64)).astype(mx.bfloat16) * 0.05,
            group_size=32 if mode == "mxfp4" else 64,
            bits=bits,
            mode=mode,
        )
        setattr(
            expert,
            name,
            QuantizedProjection(
                packed[0],
                packed[1],
                bits,
                mode,
                biases=packed[2] if mode == "affine" else None,
                group_size=32 if mode == "mxfp4" else 64,
            ),
        )
    x = mx.random.normal((1, length, 1, 1, 64)).astype(mx.bfloat16)
    ids = (mx.arange(length * 2, dtype=mx.uint32) % 4).reshape(1, length, 2)
    weights = mx.broadcast_to(mx.array([[[0.3, 0.7]]]), (1, length, 2))
    actual = expert(x, ids, weights)
    repeated = expert(x, ids, weights)
    with monkeypatch.context() as patch:
        patch.setattr(fast, "has_symbol", lambda _: False)
        expected = expert(x, ids, weights)
    mx.eval(actual, repeated, expected)
    np.testing.assert_array_equal(
        actual.astype(mx.float32), expected.astype(mx.float32)
    )
    np.testing.assert_array_equal(
        actual.astype(mx.float32), repeated.astype(mx.float32)
    )


def test_grouped_expert_rejects_mixed_streams():
    first, second = mx.new_stream(mx.gpu), mx.new_stream(mx.gpu)
    with mx.stream(first):
        gate, up, _, old_down = make_graph()
    with mx.stream(second):
        activation = quantize_swiglu_activation(
            gate, up, mx.array([0.3, 0.7]), mx.bfloat16, 10
        )
        w, s = mx.quantize(
            mx.ones((2, 64, 64), mx.bfloat16) * 0.03125,
            group_size=32,
            bits=4,
            mode="mxfp4",
        )
        ids = mx.array([0, 1], mx.uint32)
        down = mx.gather_qmm(
            activation,
            w,
            s,
            lhs_indices=ids,
            rhs_indices=ids,
            group_size=32,
            bits=4,
            mode="mxfp4",
        )
    with pytest.raises(ValueError, match="share a GPU stream"):
        fast.deepseek_v41_grouped_expert(gate, up, activation, down)
