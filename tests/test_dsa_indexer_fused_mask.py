"""Regression tests for the fused pooled-ratio mask in dsa_indexer_scores.

The kernel epilogue can apply the PoolingCache causal mask directly
(mask_ratio/mask_q_offset), replacing a separate mx.where pass. The fused
path must be BIT-IDENTICAL to the unfused reference (mask_ratio=0 followed
by mx.where with finfo.min), including -0.0/+0.0 behavior, and top-k
indices must match exactly.
"""

import mlx.core as mx
import pytest

from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast

pytestmark = pytest.mark.skipif(
    not (
        glm_fast.is_native_available()
        and glm_fast._EXT_MASK_FOLD
        and glm_fast.has_symbol("dsa_indexer_scores")
        and glm_fast.has_symbol("dsa_topk_indices")
    ),
    reason="fold-aware glm_moe_dsa native extension not built",
)


def _reference_mask(L, P, ratio, q_offset):
    """PoolingCache.make_mask semantics: visible iff col < (q_offset+row+1)//ratio."""
    rows = mx.arange(L)[:, None]
    cols = mx.arange(P)[None, :]
    return cols < ((q_offset + rows + 1) // ratio)


def _bit_equal(a, b):
    mx.eval(a, b)
    return bool(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)))


@pytest.mark.parametrize(
    "L,P,dtype",
    [
        (1, 513, mx.bfloat16),
        (63, 575, mx.bfloat16),
        (65, 577, mx.bfloat16),
        (127, 639, mx.bfloat16),
        (65, 577, mx.float16),
    ],
)
def test_unaligned_tail_matches_zero_padded_reference(L, P, dtype):
    """Partial M/N tiles must match the old aligned kernel domain exactly."""
    mx.random.seed(19)
    H, D = 64, 128
    q = mx.random.normal((1, H, L, D)).astype(dtype)
    pooled = mx.random.normal((1, 1, P, D)).astype(dtype)
    weights = mx.random.normal((1, L, H)).astype(dtype)

    actual = glm_fast.dsa_indexer_scores(q, pooled, weights, causal=False)

    padded_l = ((L + 63) // 64) * 64
    padded_p = ((P + 63) // 64) * 64
    padded_q = mx.pad(q, ((0, 0), (0, 0), (0, padded_l - L), (0, 0)))
    padded_pool = mx.pad(
        pooled,
        ((0, 0), (0, 0), (0, padded_p - P), (0, 0)),
    )
    padded_weights = mx.pad(weights, ((0, 0), (0, padded_l - L), (0, 0)))
    reference = glm_fast.dsa_indexer_scores(
        padded_q,
        padded_pool,
        padded_weights,
        causal=False,
    )[:, :, :L, :P]

    assert actual.shape == (1, 1, L, P)
    assert _bit_equal(actual, reference)

    indices = glm_fast.dsa_topk_indices(actual, 512, bucketed=False)
    mx.eval(indices)
    assert indices.shape == (1, 1, L, 512)
    assert bool(mx.all(indices < P).item())


@pytest.mark.parametrize(
    "L,P,ratio,q_offset,dtype",
    [
        (128, 1088, 4, 256, mx.bfloat16),   # offset > 0
        (64, 2048, 4, 0, mx.bfloat16),      # offset 0
        (128, 2560, 128, 1024, mx.bfloat16),  # ratio-128 style
        (128, 1088, 4, 256, mx.float16),    # fp16
        (65, 577, 4, 256, mx.bfloat16),     # partial M/N tiles
    ],
)
def test_fused_mask_bit_identical(L, P, ratio, q_offset, dtype):
    mx.random.seed(42)
    H, D = 64, 128
    q = mx.random.normal((1, H, L, D)).astype(dtype)
    pooled = mx.random.normal((1, P, D)).astype(dtype)
    weights = mx.random.normal((1, L, H)).astype(dtype)  # [B, L, H] convention

    mask = _reference_mask(L, P, ratio, q_offset)

    # Reference: unfused scores + mx.where pass (the old call-site flow).
    ref = glm_fast.dsa_indexer_scores(q, pooled[:, None], weights, causal=False)
    ref = mx.where(mask[None, None], ref, mx.finfo(ref.dtype).min)

    # Fused: kernel epilogue applies the same mask with the same sentinel.
    fused = glm_fast.dsa_indexer_scores(
        q,
        pooled[:, None],
        weights,
        causal=False,
        mask_ratio=ratio,
        mask_q_offset=q_offset,
    )

    assert fused.shape == ref.shape
    assert _bit_equal(fused, ref), "fused mask output differs bitwise from reference"

    k = min(512, P)
    idx_ref = glm_fast.dsa_topk_indices(ref, k, bucketed=False)
    idx_fused = glm_fast.dsa_topk_indices(fused, k, bucketed=False)
    mx.eval(idx_ref, idx_fused)
    assert bool(mx.array_equal(idx_ref, idx_fused)), "top-k indices differ"


def test_mask_ratio_zero_matches_unmasked():
    mx.random.seed(7)
    H, D, L, P = 64, 128, 64, 512
    q = mx.random.normal((1, H, L, D)).astype(mx.bfloat16)
    pooled = mx.random.normal((1, P, D)).astype(mx.bfloat16)
    weights = mx.random.normal((1, L, H)).astype(mx.bfloat16)  # [B, L, H]

    plain = glm_fast.dsa_indexer_scores(q, pooled[:, None], weights, causal=False)
    zero_ratio = glm_fast.dsa_indexer_scores(
        q, pooled[:, None], weights, causal=False, mask_ratio=0, mask_q_offset=0
    )
    assert _bit_equal(plain, zero_ratio)
