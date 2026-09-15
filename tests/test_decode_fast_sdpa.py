# SPDX-License-Identifier: Apache-2.0
"""Decode SDPA (decode_fast) matches mx.fast.scaled_dot_product_attention."""

import pytest
import mlx.core as mx

fast = pytest.importorskip("omlx.custom_kernels.decode_fast.fast")

@pytest.mark.skipif(
    not fast.NATIVE_AVAILABLE, reason="native extension not built"
)
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16, mx.float16])
@pytest.mark.parametrize(
    "B,H,Hkv,qL,kL,D",
    [
        (1, 8, 1, 1, 512, 128),
        (1, 8, 1, 1, 4096, 128),
        (1, 8, 1, 4, 2048, 128),  # causal, gqa*qL = 32 (limit)
        (1, 4, 4, 1, 1024, 64),   # MHA
        (2, 8, 2, 1, 1500, 96),   # odd kL, head 96
        (1, 8, 1, 1, 777, 128),   # odd kL 1-pass
        (1, 16, 2, 1, 16384, 128),
    ],
)
def test_matches_mx_fast(dtype, B, H, Hkv, qL, kL, D):
    mx.random.seed(0)
    q = mx.random.normal((B, H, qL, D)).astype(dtype)
    k = mx.random.normal((B, Hkv, kL, D)).astype(dtype)
    v = mx.random.normal((B, Hkv, kL, D)).astype(dtype)
    scale = 1.0 / (D ** 0.5)
    causal = qL > 1
    if not fast._ext.sdpa_decode_supported(q, k, v):
        # Device reports the selected pipeline occupancy below the fixed
        # threadgroup size; the wrapper would fail closed to portable SDPA
        # (parity contract lives in test_head_dim_256_never_silently_corrupts).
        pytest.skip("device kernel occupancy below fixed group size")
    out = fast._ext.sdpa_decode(q, k, v, scale, causal)
    if causal:
        mask = mx.triu(mx.full((qL, kL), float("-inf")), k=kL - qL + 1)
        mask = mask.astype(dtype)
        ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    else:
        ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    mx.eval(out, ref)
    tol = 1e-5 if dtype == mx.float32 else 5e-3
    assert mx.allclose(out, ref, atol=tol, rtol=tol).item()


# jundot/omlx#3660: Flash-Next QSA geometry (D=V=256, gqa 12). The heavy
# D=256/D=192 one-pass instantiations exceed the fixed 1024-thread group on
# some GPUs (832 on air64_v28); Metal then drops the dispatch silently and
# leaves the output unwritten. The support gate must fail closed so the
# wrapper falls back to portable SDPA; where the kernel is resident it must
# match directly. Either way parity is required, so this test is red on any
# device whose dispatch silently corrupts these shapes.
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16, mx.float16])
@pytest.mark.parametrize("H,Hkv,kL,D,V", [
    (24, 2, 512, 256, 256),   # one-pass zone on s/d-class arch chars
    (24, 2, 800, 256, 256),   # odd kL, one-pass zone
    (24, 2, 2051, 256, 256),  # production gathered QSA widths
    (8, 2, 512, 192, 128),    # qk 192 / value 128 instantiation
])
def test_head_dim_256_never_silently_corrupts(dtype, H, Hkv, kL, D, V):
    mx.random.seed(0)
    q = mx.random.normal((1, H, 1, D)).astype(dtype)
    k = mx.random.normal((1, Hkv, kL, D)).astype(dtype)
    v = mx.random.normal((1, Hkv, kL, V)).astype(dtype)
    scale = 1.0 / (D ** 0.5)
    out = fast.sdpa_decode(q, k, v, scale)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    mx.eval(out, ref)
    tol = 1e-5 if dtype == mx.float32 else 5e-3
    assert mx.allclose(out, ref, atol=tol, rtol=tol).item()


def test_wrapper_falls_back_for_long_query():
    q = mx.random.normal((1, 4, 16, 64))  # qL=16 > 8: not decode mode
    k = mx.random.normal((1, 4, 64, 64))
    v = mx.random.normal((1, 4, 64, 64))
    out = fast.sdpa_decode(q, k, v, 0.125)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125)
    mx.eval(out, ref)
    assert mx.allclose(out, ref, atol=1e-5, rtol=1e-5).item()
