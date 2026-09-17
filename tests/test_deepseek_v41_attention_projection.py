"""Attention input sharing preserves each projection's quantization setting."""

import mlx.core as mx
import numpy as np
import pytest
from test_deepseek_v41 import tiny

from omlx.patches.deepseek_v41.cache import DeepseekV41Cache
from omlx.patches.deepseek_v41.language import Attention
from omlx.patches.deepseek_v41.quantization import QuantizedProjection


@pytest.mark.parametrize(
    "quantize_q,quantize_kv",
    [(True, True), (True, False), (False, True), (False, False)],
)
@pytest.mark.parametrize("length", [1, 33])
@pytest.mark.parametrize(
    "mode,bits", [("mxfp8", 8)] + [("affine", b) for b in (2, 3, 4, 6, 8)]
)
def test_attention_shared_inputs_match_individual_calls(
    monkeypatch, quantize_q, quantize_kv, length, mode, bits
):
    mx.random.seed(964)
    attention = Attention(tiny(), 0)
    for name, quantizes in [("wq_a", quantize_q), ("wkv", quantize_kv)]:
        packed, scales, *biases = mx.quantize(
            mx.random.normal((32, 32)).astype(mx.bfloat16) * 0.05,
            mode=mode,
            bits=bits,
            group_size=32,
        )
        setattr(
            attention,
            name,
            QuantizedProjection(
                packed,
                scales,
                bits,
                mode,
                biases[0] if biases else None,
                quantize_input=quantizes,
            ),
        )
    for name in ("wq_b", "wo_a", "wo_b"):
        projection = getattr(attention, name)
        projection.weight = projection.weight.astype(mx.bfloat16)
    x = mx.random.normal((1, length, 32)).astype(mx.bfloat16)
    cache = DeepseekV41Cache()
    actual = attention(x, cache, {}, 0)
    with monkeypatch.context() as patch:
        patch.setattr(
            Attention,
            "_input_projections",
            lambda self, value: (self.wq_a(value), self.wkv(value)),
        )
        expected_cache = DeepseekV41Cache()
        expected = attention(x, expected_cache, {}, 0)
    np.testing.assert_array_equal(
        actual.astype(mx.float32), expected.astype(mx.float32)
    )
    np.testing.assert_array_equal(cache[1], expected_cache[1])
