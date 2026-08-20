"""Bit-exactness tests for the v25 MMA DSA indexer score kernel.

dsa_indexer_scores_mma (zero-per-head-barrier from-scratch simdgroup GEMM,
~1.37x over Steel on M2 Ultra) must be BIT-IDENTICAL to the Steel
dsa_indexer_scores for every configuration it serves: bf16, H=64, D=128,
weights [B, L, H], non-causal, mask_ratio 0 or the fused pooled-ratio mask —
across tile-aligned AND unaligned M/N (the boundary-kernel path) and
chunked-prefill mask offsets.
"""

import mlx.core as mx
import pytest

from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast

pytestmark = pytest.mark.skipif(
    not (
        glm_fast.is_native_available()
        and glm_fast._EXT_MASK_FOLD
        and glm_fast._EXT_MMA_SCORE
        and glm_fast.has_symbol("dsa_indexer_scores")
    ),
    reason="glm_moe_dsa native extension with the MMA score kernel not built",
)


def _inputs(M, N, seed=42):
    mx.random.seed(seed)
    q = mx.random.uniform(-0.5, 0.5, (1, 64, M, 128)).astype(mx.bfloat16)
    k = mx.random.uniform(-0.5, 0.5, (1, 1, N, 128)).astype(mx.bfloat16)
    w = mx.random.uniform(-0.5, 0.5, (1, M, 64)).astype(mx.bfloat16)
    mx.eval(q, k, w)
    return q, k, w


def _bit_equal(a, b):
    mx.eval(a, b)
    return bool(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)))


@pytest.mark.parametrize(
    "M,N,mask_ratio,mask_q_offset",
    [
        # aligned (interior kernel only)
        (128, 512, 4, 0),
        (256, 1024, 4, 0),
        (64, 64, 4, 0),
        (512, 4096, 4, 4096),
        # unaligned M and/or N (boundary kernel active) — production N is
        # NOT tile-aligned (observed live: N=11999)
        (895, 1999, 4, 4096),
        (947, 1007, 4, 0),
        (512, 1999, 4, 2048),
        (64, 65, 4, 0),
        # mask modes
        (256, 1024, 0, 0),
        (256, 1024, 1, 0),
    ],
)
def test_mma_scores_bit_exact_vs_steel(M, N, mask_ratio, mask_q_offset):
    q, k, w = _inputs(M, N)
    ref = glm_fast.dsa_indexer_scores(
        q,
        k,
        w,
        causal=False,
        mask_ratio=mask_ratio,
        mask_q_offset=mask_q_offset,
    )
    got = glm_fast.dsa_indexer_scores_mma(
        q, k, w, mask_ratio=mask_ratio, mask_q_offset=mask_q_offset
    )
    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert _bit_equal(ref, got)


def test_mma_scores_second_seed():
    q, k, w = _inputs(256, 1024, seed=7)
    ref = glm_fast.dsa_indexer_scores(
        q, k, w, causal=False, mask_ratio=4, mask_q_offset=0
    )
    got = glm_fast.dsa_indexer_scores_mma(q, k, w, mask_ratio=4, mask_q_offset=0)
    assert _bit_equal(ref, got)


def test_mma_scores_batched():
    # B > 1 exercises the per-batch base-pointer arithmetic (tgpig.z), which
    # the B=1 matrix above never touches.
    mx.random.seed(13)
    q = mx.random.uniform(-0.5, 0.5, (3, 64, 895, 128)).astype(mx.bfloat16)
    k = mx.random.uniform(-0.5, 0.5, (3, 1, 1999, 128)).astype(mx.bfloat16)
    w = mx.random.uniform(-0.5, 0.5, (3, 895, 64)).astype(mx.bfloat16)
    mx.eval(q, k, w)
    ref = glm_fast.dsa_indexer_scores(
        q, k, w, causal=False, mask_ratio=4, mask_q_offset=4096
    )
    got = glm_fast.dsa_indexer_scores_mma(
        q, k, w, mask_ratio=4, mask_q_offset=4096
    )
    assert _bit_equal(ref, got)


def test_mma_scores_rejects_unsupported_configs():
    # fp16 (kernel is bf16-only)
    q, k, w = _inputs(128, 512)
    with pytest.raises(Exception):
        glm_fast.dsa_indexer_scores_mma(
            q.astype(mx.float16), k.astype(mx.float16), w.astype(mx.float16)
        )
    # H != 64 (the GLM caller's H=32 must never land here)
    mx.random.seed(0)
    q32 = mx.random.uniform(-0.5, 0.5, (1, 32, 128, 128)).astype(mx.bfloat16)
    w32 = mx.random.uniform(-0.5, 0.5, (1, 128, 32)).astype(mx.bfloat16)
    with pytest.raises(Exception):
        glm_fast.dsa_indexer_scores_mma(q32, k, w32)
    # weights rank 4 (LH layout only)
    with pytest.raises(Exception):
        glm_fast.dsa_indexer_scores_mma(q, k, w[..., None])


def test_mma_topk_selection_matches_steel():
    # end-of-pipeline check: identical scores must give identical indices
    q, k, w = _inputs(512, 4096)
    ref = glm_fast.dsa_indexer_scores(
        q, k, w, causal=False, mask_ratio=4, mask_q_offset=4096
    )
    got = glm_fast.dsa_indexer_scores_mma(
        q, k, w, mask_ratio=4, mask_q_offset=4096
    )
    idx_ref = glm_fast.dsa_topk_indices(ref, 512)
    idx_got = glm_fast.dsa_topk_indices(got, 512)
    mx.eval(idx_ref, idx_got)
    assert bool(mx.array_equal(idx_ref, idx_got))
