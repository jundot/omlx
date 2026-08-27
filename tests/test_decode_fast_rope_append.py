# SPDX-License-Identifier: Apache-2.0
"""Fused RoPE+KV append (decode_fast) matches composed rope + slice update."""

import pytest
import mlx.core as mx

fast = pytest.importorskip("omlx.custom_kernels.decode_fast.fast")

pytestmark = pytest.mark.skipif(
    not fast.NATIVE_AVAILABLE, reason="native extension not built"
)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16, mx.float16])
@pytest.mark.parametrize("traditional", [False, True])
@pytest.mark.parametrize(
    "B,Hkv,kL,D,dims,offset",
    [
        (1, 1, 512, 128, 128, 100),
        (1, 2, 1024, 128, 64, 255),
        (2, 4, 768, 64, 64, 0),
        (1, 1, 2048, 128, 128, 1023),
    ],
)
def test_matches_composed(dtype, traditional, B, Hkv, kL, D, dims, offset):
    mx.random.seed(0)
    keys = mx.random.normal((B, Hkv, 1, D)).astype(dtype)
    vals = mx.random.normal((B, Hkv, 1, D)).astype(dtype)
    kc = mx.random.normal((B, Hkv, kL, D)).astype(dtype)
    vc = mx.random.normal((B, Hkv, kL, D)).astype(dtype)
    mx.eval(keys, vals, kc, vc)
    assert fast._ext.rope_kv_append_supported(keys, vals, kc, vc, offset, dims)
    kc_n, vc_n = fast._ext.rope_kv_append(
        keys, vals, kc, vc, offset, dims, traditional, 10000.0, 1.0
    )
    k_rot = mx.fast.rope(
        keys, dims, traditional=traditional, base=10000.0, scale=1.0, offset=offset
    )
    kc_ref = kc + mx.zeros_like(kc)
    vc_ref = vc + mx.zeros_like(vc)
    kc_ref[:, :, offset : offset + 1, :] = k_rot
    vc_ref[:, :, offset : offset + 1, :] = vals
    mx.eval(kc_n, vc_n, kc_ref, vc_ref)
    tol = 1e-5 if dtype == mx.float32 else 2e-3
    assert mx.allclose(kc_n, kc_ref, atol=tol, rtol=tol).item()
    assert mx.allclose(vc_n, vc_ref, atol=tol, rtol=tol).item()


def test_non_donatable_cache_leaves_original_untouched():
    keys = mx.random.normal((1, 1, 1, 128))
    vals = mx.random.normal((1, 1, 1, 128))
    kc = mx.random.normal((1, 1, 512, 128))
    vc = mx.random.normal((1, 1, 512, 128))
    mx.eval(keys, vals, kc, vc)
    kc_snap = kc + mx.zeros_like(kc)
    vc_snap = vc + mx.zeros_like(vc)
    mx.eval(kc_snap, vc_snap)
    kc_out, vc_out = fast._ext.rope_kv_append(
        keys, vals, kc, vc, 100, 128, False, 10000.0, 1.0
    )
    mx.eval(kc_out, vc_out)
    assert mx.allclose(kc, kc_snap, atol=0).item()
    assert mx.allclose(vc, vc_snap, atol=0).item()
