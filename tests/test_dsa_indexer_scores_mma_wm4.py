"""Lossless gates for the default-off DS4 WM4xWN1 MMA partition."""

import mlx.core as mx
import pytest

from omlx.custom_kernels.glm_moe_dsa import fast

pytestmark = pytest.mark.skipif(
    not (
        fast.is_native_available()
        and fast._EXT_MASK_FOLD
        and fast._EXT_MMA_SCORE
        and fast._EXT_MMA_WM4
        and fast.has_symbol("dsa_indexer_scores")
        and fast.has_symbol("dsa_topk_indices")
    ),
    reason="glm_moe_dsa extension with the WM4xWN1 MMA candidate not built",
)


def _inputs(query_tokens: int, pooled_tokens: int, seed: int):
    mx.random.seed(seed)
    q = mx.random.uniform(-0.5, 0.5, (1, 64, query_tokens, 128)).astype(mx.bfloat16)
    k = mx.random.uniform(-0.5, 0.5, (1, 1, pooled_tokens, 128)).astype(mx.bfloat16)
    w = mx.random.uniform(-0.5, 0.5, (1, query_tokens, 64)).astype(mx.bfloat16)
    mx.eval(q, k, w)
    return q, k, w


@pytest.mark.parametrize(
    "query_tokens,pooled_tokens,mask_offset,seed",
    [
        (64, 512, 0, 3),
        (512, 4096, 3072, 7),
        # Both output axes exercise the separately compiled boundary path.
        (513, 4097, 3072, 11),
    ],
)
def test_wm4_score_sheet_is_bit_exact(query_tokens, pooled_tokens, mask_offset, seed):
    q, k, w = _inputs(query_tokens, pooled_tokens, seed)
    steel = fast.dsa_indexer_scores(
        q,
        k,
        w,
        causal=False,
        mask_ratio=4,
        mask_q_offset=mask_offset,
        use_nax=False,
    )
    baseline = fast.dsa_indexer_scores_mma(
        q, k, w, mask_ratio=4, mask_q_offset=mask_offset
    )
    candidate = fast.dsa_indexer_scores_mma(
        q,
        k,
        w,
        mask_ratio=4,
        mask_q_offset=mask_offset,
        use_wm4_wn1=True,
    )
    mx.eval(steel, baseline, candidate)
    assert bool(mx.array_equal(steel.view(mx.uint16), candidate.view(mx.uint16)).item())
    assert bool(
        mx.array_equal(baseline.view(mx.uint16), candidate.view(mx.uint16)).item()
    )


def test_wm4_deterministic_topk_matches_production():
    q, k, w = _inputs(512, 7500, 17)
    baseline = fast.dsa_indexer_scores_mma(
        q, k, w, mask_ratio=4, mask_q_offset=7500 * 4 - 1024
    )
    candidate = fast.dsa_indexer_scores_mma(
        q,
        k,
        w,
        mask_ratio=4,
        mask_q_offset=7500 * 4 - 1024,
        use_wm4_wn1=True,
    )
    baseline_indices = fast.dsa_topk_indices(baseline, 512, bucketed=False)
    candidate_indices = fast.dsa_topk_indices(candidate, 512, bucketed=False)
    mx.eval(baseline_indices, candidate_indices)
    assert bool(mx.array_equal(baseline_indices, candidate_indices).item())
