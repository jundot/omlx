# SPDX-License-Identifier: Apache-2.0
"""Match short sparse Hy4 verification with the stock gather path."""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from omlx.patches.hy_v4 import verify_dispatch, verify_sparse  # noqa: E402
from omlx.patches.hy_v4.layers import Attention  # noqa: E402


@pytest.fixture
def fast_gpu():
    if not mx.metal.is_available() or not verify_dispatch._supported_gpu():
        pytest.skip("short Hy4 verify kernels require the validated MLX/Metal backend")
    with mx.stream(mx.gpu):
        yield


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_shared_sparse_verify_matches_stock_with_padding_and_future_keys(
    fast_gpu, dtype
):
    mx.random.seed(314)
    rows, heads, total, width = 4, 32, 2300, 2048

    def normal(shape):
        return (mx.random.normal(shape) * 0.12).astype(dtype)

    latent = normal((1, 1, total, 512))
    q_latent = normal((1, heads, rows, 512))
    q_pe = normal((1, heads, rows, 64))
    k_pe = normal((1, 1, total, 64))
    sinks = normal((heads,))
    topk = mx.random.randint(0, total + 7, shape=(1, 1, rows, width))
    mask = (mx.arange(total) >= 11)[None, None, None, :]
    scale = 192**-0.5
    stub = SimpleNamespace(
        kind="sparse", embed_q=lambda x: x, unembed_out=lambda x: x, scale=scale
    )
    expected = Attention._verify(stub, q_latent, q_pe, latent, k_pe, topk, mask, sinks)
    actual = verify_sparse.attend(
        q_latent, q_pe, latent, k_pe, topk, mask, sinks, scale
    )
    assert mx.array_equal(expected.view(mx.uint16), actual.view(mx.uint16)).item()
