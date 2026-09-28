# SPDX-License-Identifier: Apache-2.0
"""Native decode attention must support its largest dispatched threadgroup."""

import mlx.core as mx
import pytest

from omlx.custom_kernels.decode_fast import fast


@pytest.mark.skipif(not fast.NATIVE_AVAILABLE, reason="native extension not built")
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "head_dim,value_dim", [(64, 64), (96, 96), (128, 128), (192, 128), (256, 256)]
)
@pytest.mark.parametrize("key_length", [512, 2048])
@pytest.mark.parametrize("query_length", [1, 4])
def test_native_launch_size(dtype, head_dim, value_dim, key_length, query_length):
    mx.random.seed(0)
    q = mx.random.normal((1, 8, query_length, head_dim)).astype(dtype)
    k = mx.random.normal((1, 1, key_length, head_dim)).astype(dtype)
    v = mx.random.normal((1, 1, key_length, value_dim)).astype(dtype)
    scale = head_dim**-0.5
    # Eight query heads and four query rows reach the split kernel's supported
    # maximum of 32 SIMD groups, while the single-pass kernel always uses 32.
    causal = query_length > 1
    assert fast._ext.sdpa_decode_supported(q, k, v)
    out = fast._ext.sdpa_decode(q, k, v, scale, causal)
    expected = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask="causal" if causal else None
    )
    mx.eval(out, expected)
    tolerance = 1e-5 if dtype == mx.float32 else 5e-3
    assert mx.allclose(out, expected, atol=tolerance, rtol=tolerance).item()
