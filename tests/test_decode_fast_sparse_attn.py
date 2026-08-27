# SPDX-License-Identifier: Apache-2.0
"""Fused sparse decode attention (decode_fast) vs the composed reference.

The reference mirrors
``omlx.patches.deepseek_v4.deepseek_v4_model._dspark_sparse_exact_attention``
(four rowwise GEMMs + logsumexp/logaddexp/exp glue) without importing the
patched model module. Also verifies the DSpark batch-invariance contract:
per-row results for a B>1 verify batch must be bitwise-identical to the
B=1 decode call on the same row.
"""

import pytest
import mlx.core as mx

fast = pytest.importorskip("omlx.custom_kernels.decode_fast.fast")

pytestmark = pytest.mark.skipif(
    not fast.NATIVE_AVAILABLE, reason="native extension not built"
)


def reference(q_scaled, local_kv, pooled, sinks, dtype=None):
    """Composed sparse exact attention (mirrors _dspark_sparse_exact_attention).

    Each batch row is computed independently — plain batched ``@`` in MLX is
    not batch-consistent for these shapes (the reason omlx uses rowwise
    GEMMs in the first place), so the reference loops rows explicitly.
    """
    dt = dtype or q_scaled.dtype
    outs = []
    for b in range(q_scaled.shape[0]):
        q_b = q_scaled[b : b + 1].astype(dt)
        lk_b = local_kv[b : b + 1].astype(dt)
        pk_b = pooled[b : b + 1].astype(dt)
        ls = (q_b.transpose(0, 2, 1, 3) @ lk_b[:, 0].swapaxes(-1, -2)).transpose(0, 2, 1, 3)
        ps = (q_b.transpose(0, 2, 1, 3) @ pk_b[:, 0].swapaxes(-1, -2)).transpose(0, 2, 1, 3)
        nrm = mx.logaddexp(
            mx.logsumexp(ls, -1, keepdims=True),
            mx.logsumexp(ps, -1, keepdims=True),
        )
        if dt in (mx.bfloat16, mx.float16):
            nrm = mx.logaddexp(nrm, sinks[None, :, None, None])  # fp32 promotion
        else:
            nrm = mx.logaddexp(nrm, sinks[None, :, None, None].astype(dt))
        lw = mx.exp(ls - nrm)
        pw = mx.exp(ps - nrm)
        lo = (lw.transpose(0, 2, 1, 3) @ lk_b[:, 0]).transpose(0, 2, 1, 3)
        po = (pw.transpose(0, 2, 1, 3) @ pk_b[:, 0]).transpose(0, 2, 1, 3)
        outs.append(lo + po)
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize(
    "B,H,W,P",
    [
        (1, 64, 128, 512),   # V4 decode, full topk
        (1, 64, 128, 179),   # 23K ctx ratio-128 layer
        (1, 64, 29, 33),     # short context, partial window
        (1, 64, 1, 1),       # first token
        (5, 64, 128, 512),   # DSpark verify batch
        (2, 64, 100, 300),   # odd sizes
    ],
)
def test_matches_reference(dtype, B, H, W, P):
    mx.random.seed(0)
    D = 512
    q = (mx.random.normal((B, H, 1, D)) * 0.5).astype(dtype)
    lk = (mx.random.normal((B, 1, W, D)) * 0.3).astype(dtype)
    pk = (mx.random.normal((B, 1, P, D)) * 0.3).astype(dtype)
    sinks = mx.random.normal((H,)).astype(mx.float32)
    mx.eval(q, lk, pk, sinks)

    out = fast.sparse_attn_decode(q, lk, pk, sinks)
    assert out is not None
    ref = reference(q, lk, pk, sinks)
    truth = reference(q, lk, pk, sinks, dtype=mx.float32).astype(mx.float32)
    mx.eval(out, ref, truth)

    assert out.shape == ref.shape
    assert out.dtype == dtype
    # Gate on accuracy vs the fp32-computed chain: the kernel's error must be
    # comparable to the reference bf16 chain's own intrinsic rounding error.
    err_kernel = float(mx.abs(out.astype(mx.float32) - truth).max())
    err_ref = float(mx.abs(ref.astype(mx.float32) - truth).max())
    assert err_kernel <= max(1e-3, 1.5 * err_ref), (
        f"kernel error {err_kernel} vs reference-chain error {err_ref}"
    )


def test_batch_row_invariance():
    """Verify-batch rows must be bitwise-identical to single-row decode."""
    mx.random.seed(1)
    B, H, W, P, D = 5, 64, 128, 512, 512
    q = (mx.random.normal((B, H, 1, D)) * 0.5).astype(mx.bfloat16)
    lk = (mx.random.normal((B, 1, W, D)) * 0.3).astype(mx.bfloat16)
    pk = (mx.random.normal((B, 1, P, D)) * 0.3).astype(mx.bfloat16)
    sinks = mx.random.normal((H,)).astype(mx.float32)
    mx.eval(q, lk, pk, sinks)

    batch_out = fast.sparse_attn_decode(q, lk, pk, sinks)
    mx.eval(batch_out)
    for b in range(B):
        single = fast.sparse_attn_decode(
            q[b : b + 1], lk[b : b + 1], pk[b : b + 1], sinks
        )
        mx.eval(single)
        same = mx.array_equal(batch_out[b : b + 1], single)
        assert bool(same), f"row {b} differs between batched and single calls"


def test_strided_cache_views():
    """Prefix slices of a larger cache buffer (RotatingKVCache layout)."""
    mx.random.seed(2)
    B, H, D = 1, 64, 512
    cap, W, P = 4096, 128, 512
    big_kv = (mx.random.normal((B, 1, cap, D)) * 0.3).astype(mx.bfloat16)
    big_pool = (mx.random.normal((B, 1, 6000, D)) * 0.3).astype(mx.bfloat16)
    q = (mx.random.normal((B, H, 1, D)) * 0.5).astype(mx.bfloat16)
    sinks = mx.random.normal((H,)).astype(mx.float32)
    mx.eval(big_kv, big_pool, q, sinks)

    lk = big_kv[:, :, :W]
    pk = big_pool[:, :, :P]
    out = fast.sparse_attn_decode(q, lk, pk, sinks)
    assert out is not None
    ref = reference(q, lk, pk, sinks)
    truth = reference(q, lk, pk, sinks, dtype=mx.float32).astype(mx.float32)
    mx.eval(out, ref, truth)
    err_kernel = float(mx.abs(out.astype(mx.float32) - truth).max())
    err_ref = float(mx.abs(ref.astype(mx.float32) - truth).max())
    assert err_kernel <= max(1e-3, 1.5 * err_ref)


def test_unsupported_returns_none():
    q = mx.zeros((1, 64, 1, 256), dtype=mx.bfloat16)  # wrong head dim
    lk = mx.zeros((1, 1, 128, 256), dtype=mx.bfloat16)
    pk = mx.zeros((1, 1, 64, 256), dtype=mx.bfloat16)
    sinks = mx.zeros((64,), dtype=mx.float32)
    assert fast.sparse_attn_decode(q, lk, pk, sinks) is None

    q = mx.zeros((1, 64, 1, 512), dtype=mx.bfloat16)
    lk = mx.zeros((1, 1, 900, 512), dtype=mx.bfloat16)
    pk = mx.zeros((1, 1, 300, 512), dtype=mx.bfloat16)  # W+P > 1024
    assert fast.sparse_attn_decode(q, lk, pk, sinks) is None
