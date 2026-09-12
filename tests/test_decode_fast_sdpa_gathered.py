# SPDX-License-Identifier: Apache-2.0
"""Gathered decode SDPA (decode_fast): K/V rows read through per-query index sets."""

import mlx.core as mx
import pytest

fast = pytest.importorskip("omlx.custom_kernels.decode_fast.fast")

pytestmark = pytest.mark.skipif(
    not fast.NATIVE_AVAILABLE, reason="native extension not built"
)


def _reference(q, k, v, indices, scale):
    """Gather the selected rows per query row and run MLX SDPA with a validity mask."""
    B, H, qL, D = q.shape
    outs = []
    for b in range(B):
        rows = []
        for r in range(qL):
            idx = indices[b, r]
            valid = idx >= 0
            safe = mx.where(valid, idx, 0)
            ks = mx.take(k[b], safe, axis=1)[None]
            vs = mx.take(v[b], safe, axis=1)[None]
            out = mx.fast.scaled_dot_product_attention(
                q[b : b + 1, :, r : r + 1], ks, vs, scale=scale, mask=valid[None, None, None]
            )
            rows.append(out[:, :, 0])
        outs.append(mx.stack(rows, axis=2))
    return mx.concatenate(outs, axis=0)


def _inputs(dtype, B, H, Hkv, qL, kL, D, S, holes):
    mx.random.seed(0)
    q = mx.random.normal((B, H, qL, D)).astype(dtype)
    k = mx.random.normal((B, Hkv, kL, D)).astype(dtype)
    v = mx.random.normal((B, Hkv, kL, D)).astype(dtype)
    indices = mx.sort(mx.random.randint(0, kL, (B, qL, S)), axis=-1).astype(mx.int32)
    if holes:
        # Invalidate a ragged tail per query row, like QSA's incomplete last block.
        keep = mx.arange(S)[None, None] < (S - 1 - mx.arange(qL)[None, :, None])
        indices = mx.where(keep, indices, -1)
    return q, k, v, indices


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16, mx.float16])
@pytest.mark.parametrize(
    "B,H,Hkv,qL,kL,D,S,holes",
    [
        (1, 24, 2, 1, 4096, 256, 2051, False),   # Qwen4 QSA decode: top-2048 + 3 tail rows
        (1, 24, 2, 4, 65536, 256, 2051, True),   # MTP verify width, ragged tails, gqa*qL > 32
        (1, 24, 2, 8, 8192, 256, 2051, True),
        (1, 8, 1, 1, 512, 128, 100, False),      # S not a multiple of the key tile
        (2, 8, 2, 3, 1500, 128, 257, True),      # batch > 1
        (1, 4, 4, 1, 1024, 64, 33, False),       # MHA, small head
    ],
)
def test_gathered_matches_reference(dtype, B, H, Hkv, qL, kL, D, S, holes):
    q, k, v, indices = _inputs(dtype, B, H, Hkv, qL, kL, D, S, holes)
    scale = D**-0.5
    assert fast._ext.sdpa_decode_gathered_supported(q, k, v, indices)
    out = fast._ext.sdpa_decode_gathered(q, k, v, indices, scale)
    ref = _reference(q, k, v, indices, scale)
    mx.eval(out, ref)
    assert out.shape == (B, H, qL, D)
    tol = 1e-5 if dtype == mx.float32 else 5e-3
    assert mx.allclose(out, ref, atol=tol, rtol=tol).item()


def test_gathered_reads_the_cache_in_place():
    """A strided K/V view of a larger cache must be read without a copy: rows past the
    view's length are never touched, so out-of-range rows in the backing store do not matter."""
    q, k, v, indices = _inputs(mx.bfloat16, 1, 24, 2, 4, 2048, 256, 300, True)
    backing_k = mx.concatenate([k, mx.full(k.shape, float("nan"), dtype=k.dtype)], axis=2)
    backing_v = mx.concatenate([v, mx.full(v.shape, float("nan"), dtype=v.dtype)], axis=2)
    kv_view = backing_k[:, :, :2048], backing_v[:, :, :2048]
    mx.eval(*kv_view)
    assert fast._ext.sdpa_decode_gathered_supported(q, *kv_view, indices)
    out = fast._ext.sdpa_decode_gathered(q, *kv_view, indices, 1 / 16)
    ref = _reference(q, k, v, indices, 1 / 16)
    mx.eval(out, ref)
    assert mx.allclose(out, ref, atol=5e-3, rtol=5e-3).item()


def test_gathered_supported_rejects_wrong_inputs():
    q, k, v, indices = _inputs(mx.bfloat16, 1, 8, 1, 1, 512, 128, 100, False)
    ok = fast._ext.sdpa_decode_gathered_supported
    assert ok(q, k, v, indices)
    assert not ok(q, k, v, indices.astype(mx.uint32))                # index dtype
    assert not ok(q, k, v, indices[0])                               # rank
    assert not ok(q, k.astype(mx.float32), v, indices)               # dtype mismatch
    assert not ok(q, k, v, mx.zeros((1, 2, 100), dtype=mx.int32))    # qL mismatch
    assert not ok(mx.zeros((1, 8, 1, 32), dtype=mx.bfloat16), k, v, indices)  # head dim


def test_wrapper_accepts_strided_queries_and_indices():
    """Slices of the verify window arrive non-contiguous; the wrapper must not hand them to the kernel raw."""
    q, k, v, indices = _inputs(mx.bfloat16, 1, 24, 2, 8, 4096, 256, 512, True)
    q_view, idx_view = q[:, :, ::2], indices[:, ::2, ::2]
    out = fast.sdpa_decode_gathered(q_view, k, v, idx_view, 1 / 16)
    ref = _reference(mx.contiguous(q_view), k, v, mx.contiguous(idx_view), 1 / 16)
    mx.eval(out, ref)
    assert mx.allclose(out, ref, atol=5e-3, rtol=5e-3).item()


def test_wrapper_falls_back_when_unsupported():
    q, k, v, indices = _inputs(mx.float32, 1, 4, 4, 1, 256, 32, 20, True)  # D=32: no kernel
    out = fast.sdpa_decode_gathered(q, k, v, indices, 32**-0.5)
    ref = _reference(q, k, v, indices, 32**-0.5)
    mx.eval(out, ref)
    assert mx.allclose(out, ref, atol=1e-5, rtol=1e-5).item()
