"""Lossless-contract tests for the optional M5/NAX DS4 score tile.

The NAX route is intentionally narrower than ``dsa_indexer_scores``: exact
DS4-Flash BF16 H64/D128 ratio-4 prefill only. Its score sheet must retain the
Steel output ABI, exact BF16 pooled-mask sentinel, and deterministic top-k
membership/order. The score-bit test is the promotion gate and remains xfailed
while the first M5 sweep has a one-ULP miss; production dispatch is default-off.
These tests skip on pre-NAX machines and source-only builds.
"""

import mlx.core as mx
import pytest

from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast
from omlx.custom_kernels.nax import is_nax_available


pytestmark = pytest.mark.skipif(
    not (
        is_nax_available()
        and glm_fast.is_native_available()
        and glm_fast._EXT_MASK_FOLD
        and glm_fast._EXT_NAX_SCORE
        and glm_fast.dsa_indexer_nax_kernels_built()
        and glm_fast.has_symbol("dsa_topk_indices")
    ),
    reason="M5/NAX glm_moe_dsa indexer-score metallib not available",
)


def _inputs(M: int, N: int, seed: int = 42):
    mx.random.seed(seed)
    q = mx.random.uniform(-0.5, 0.5, (1, 64, M, 128)).astype(mx.bfloat16)
    k = mx.random.uniform(-0.5, 0.5, (1, 1, N, 128)).astype(mx.bfloat16)
    w = mx.random.uniform(-0.5, 0.5, (1, M, 64)).astype(mx.bfloat16)
    mx.eval(q, k, w)
    return q, k, w


def _bits(a: mx.array) -> mx.array:
    return a.view(mx.uint16)


@pytest.mark.parametrize(
    "M,N,q_offset",
    [
        (16, 513, 2048),       # minimum routed query tile; partial N tile
        (31, 769, 3072),       # partial M and N tiles
        (64, 1024, 4096),      # aligned production-sized tile grid
        (127, 1537, 6144),     # multi-tile unaligned chunk and context
    ],
)
@pytest.mark.xfail(
    reason="M5 candidate has a measured one-BF16-ULP score drift; default off",
    strict=False,
)
def test_nax_scores_are_bit_exact_vs_steel(M, N, q_offset):
    q, k, w = _inputs(M, N)
    steel = glm_fast.dsa_indexer_scores(
        q,
        k,
        w,
        causal=False,
        mask_ratio=4,
        mask_q_offset=q_offset,
        use_nax=False,
    )
    nax = glm_fast.dsa_indexer_scores(
        q,
        k,
        w,
        causal=False,
        mask_ratio=4,
        mask_q_offset=q_offset,
        use_nax=True,
    )
    mx.eval(steel, nax)

    assert nax.shape == (1, 1, M, N)
    assert nax.dtype == mx.bfloat16
    assert bool(mx.array_equal(_bits(nax), _bits(steel)))
    assert glm_fast.dsa_indexer_nax_runtime_active()


def test_nax_mask_sentinel_and_topk_match_steel_exactly():
    M, N, q_offset = 33, 769, 2048
    q, k, w = _inputs(M, N, seed=7)
    steel = glm_fast.dsa_indexer_scores(
        q,
        k,
        w,
        causal=False,
        mask_ratio=4,
        mask_q_offset=q_offset,
        use_nax=False,
    )
    nax = glm_fast.dsa_indexer_scores(
        q,
        k,
        w,
        causal=False,
        mask_ratio=4,
        mask_q_offset=q_offset,
        use_nax=True,
    )
    rows = mx.arange(M)[:, None]
    cols = mx.arange(N)[None, :]
    masked = cols >= ((q_offset + rows + 1) // 4)
    sentinel = mx.full((M, N), mx.finfo(mx.bfloat16).min, dtype=mx.bfloat16)
    nax2 = nax[0, 0]
    mx.eval(nax2, masked, sentinel)

    sentinel_exact = mx.where(masked, _bits(nax2) == _bits(sentinel), True)
    assert bool(mx.all(sentinel_exact).item())

    idx_steel = glm_fast.dsa_topk_indices(steel, 512, bucketed=False)
    idx_nax = glm_fast.dsa_topk_indices(nax, 512, bucketed=False)
    mx.eval(idx_steel, idx_nax)
    assert bool(mx.array_equal(idx_nax, idx_steel))


def test_use_nax_hint_preserves_steel_for_out_of_domain_shapes():
    # M < 16 is outside the amortized prefill domain. The C++ primitive must
    # silently retain Steel even when a caller supplies the preference.
    q, k, w = _inputs(15, 513, seed=11)
    steel = glm_fast.dsa_indexer_scores(
        q, k, w, causal=False, mask_ratio=4, mask_q_offset=2048
    )
    hinted = glm_fast.dsa_indexer_scores(
        q,
        k,
        w,
        causal=False,
        mask_ratio=4,
        mask_q_offset=2048,
        use_nax=True,
    )
    mx.eval(steel, hinted)
    assert bool(mx.array_equal(_bits(hinted), _bits(steel)))
