"""Routing preserves row-local activation quantization and router inputs."""

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.deepseek_v41 import language
from omlx.patches.deepseek_v41.config import ModelConfig
from omlx.patches.deepseek_v41.quantization import QuantizedProjection


@pytest.mark.parametrize("length", [1, 32])
@pytest.mark.parametrize(
    "routed,shared", [(True, True), (True, False), (False, True), (False, False)]
)
def test_moe_quantization_reuse_matches_per_expert_reference(length, routed, shared):
    mx.random.seed(412)
    config = ModelConfig(
        dim=64, moe_inter_dim=64, n_routed_experts=4, n_activated_experts=2
    )
    moe = language.MoE(config)
    moe.gate.weight = mx.random.normal((4, 64))
    for expert, switched, quantizes in [
        (moe.experts, True, routed),
        (moe.shared_experts, False, shared),
    ]:
        for name in ("w1", "w3", "w2"):
            weight = (
                mx.random.normal((4, 64, 64) if switched else (64, 64)).astype(
                    mx.bfloat16
                )
                * 0.05
            )
            packed, scales, biases = mx.quantize(weight, bits=4, group_size=32)
            setattr(
                expert,
                name,
                QuantizedProjection(
                    packed, scales, 4, "affine", biases=biases, quantize_input=quantizes
                ),
            )
    x = mx.random.normal((1, length, 64)).astype(mx.bfloat16)
    # Route raw inputs and project each selected expert without prequantization.
    indices, weights = moe.gate(x, None)
    selected = moe.experts(x[..., None, None, :], indices, weights).squeeze(-2)
    expected = (
        selected.astype(mx.float32).sum(-2) + moe.shared_experts(x).astype(mx.float32)
    ).astype(x.dtype)
    actual = moe(x, None)
    repeated = moe(x, None)
    mx.eval(expected, actual, repeated)
    np.testing.assert_allclose(
        actual.astype(mx.float32), expected.astype(mx.float32), rtol=0.01, atol=0.002
    )
    np.testing.assert_array_equal(
        actual.astype(mx.float32), repeated.astype(mx.float32)
    )


def test_expert_rejects_prequantized_input_for_unquantized_projections():
    expert = language.Expert(ModelConfig(dim=64, moe_inter_dim=64))
    with pytest.raises(ValueError, match="matching quantized projections"):
        expert(mx.zeros((1, 1, 64)), input_quantized=True)
