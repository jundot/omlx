"""All oQ affine widths use the V4.1 expert execution optimizations."""

import mlx.core as mx
import numpy as np
import pytest

from omlx.custom_kernels.glm_moe_dsa import fast
from omlx.patches.deepseek_v41.config import ModelConfig
from omlx.patches.deepseek_v41.language import Expert
from omlx.patches.deepseek_v41.quantization import QuantizedProjection


def expert_for(bits):
    expert = Expert(ModelConfig(dim=64, moe_inter_dim=64, n_routed_experts=4), True)
    widths = (bits,) * 3 if isinstance(bits, int) else bits
    for name, width in zip(("w1", "w3", "w2"), widths):
        weight, scales, biases = mx.quantize(
            (mx.random.normal((4, 64, 64)) * 0.1).astype(mx.bfloat16),
            group_size=64,
            bits=width,
            mode="affine",
        )
        setattr(
            expert,
            name,
            QuantizedProjection(
                weight,
                scales,
                width,
                "affine",
                biases,
                group_size=64,
            ),
        )
    return expert


@pytest.mark.parametrize("bits", [2, 3, 4, 6, 8, (2, 3, 6)])
@pytest.mark.skipif(
    not fast.has_symbol("deepseek_v41_grouped_expert"),
    reason="Requires the native extension",
)
def test_affine_decode_grouped_matches_stock(monkeypatch, bits):
    mx.random.seed(948)
    expert = expert_for(bits)
    x = mx.random.normal((1, 1, 1, 1, 64)).astype(mx.bfloat16)
    ids = mx.array([[[0, 2]]], mx.uint32)
    weights = mx.array([[[0.3, 0.7]]])
    calls = []
    original = fast.deepseek_v41_grouped_expert

    def counted(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(fast, "deepseek_v41_grouped_expert", counted)
    actual = expert(x, ids, weights)
    repeated = expert(x, ids, weights)
    assert len(calls) == 2
    with monkeypatch.context() as patch:
        patch.setattr(fast, "has_symbol", lambda _: False)
        expected = expert(x, ids, weights)
    np.testing.assert_array_equal(
        actual.astype(mx.float32), expected.astype(mx.float32)
    )
    np.testing.assert_array_equal(
        actual.astype(mx.float32), repeated.astype(mx.float32)
    )


@pytest.mark.parametrize("bits", [2, 3, 4, 6, 8, (2, 3, 6)])
@pytest.mark.skipif(
    not fast.has_symbol("deepseek_affine_gather_qmm_blocks"),
    reason="Requires the native extension",
)
def test_affine_prefill_native_blocks_and_pair(monkeypatch, bits):
    mx.random.seed(714)
    expert = expert_for(bits)
    x = mx.random.normal((1024, 1, 64)).astype(mx.bfloat16)
    ids = mx.repeat(mx.arange(4, dtype=mx.uint32), 256)
    weights = mx.random.uniform(shape=(1024,))
    calls = []
    original = fast.deepseek_affine_gather_qmm_pair_concat_blocks

    def counted(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(fast, "deepseek_affine_gather_qmm_pair_concat_blocks", counted)
    actual = expert(x, ids, weights, sorted_indices=True)
    assert calls == ([True] if isinstance(bits, int) else [])
    with monkeypatch.context() as patch:
        patch.setattr(fast, "has_symbol", lambda _: False)
        expected = expert(x, ids, weights, sorted_indices=True)
    np.testing.assert_allclose(
        actual.astype(mx.float32), expected.astype(mx.float32), rtol=0.03, atol=0.0002
    )
    # Exercise each packed bit layout directly, independently of nonlinearities.
    projection = expert.w1
    reference = mx.gather_qmm(
        x,
        projection.weight,
        projection.scales,
        projection.biases,
        rhs_indices=ids,
        group_size=64,
        bits=projection.bits,
        mode="affine",
        sorted_indices=True,
    )
    native = projection.project_quantized(x, ids, True)
    np.testing.assert_allclose(
        native.astype(mx.float32), reference.astype(mx.float32), rtol=0.03, atol=0.02
    )
