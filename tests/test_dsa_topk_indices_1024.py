# SPDX-License-Identifier: Apache-2.0
"""topk=1024 native top-k must match the stable-sort reference.

DeepSeek-V4-Pro ships ``index_topk=1024``; the kernel previously had only
512/2048 instantiations, so Pro's long-context prefill silently dropped to
the MLX fallback (~4x worse indexer slope). The kernel template is generic
over TOPK — these tests pin the new instantiation to the deterministic
tie-break contract the call site depends on.
"""

import pytest

mx = pytest.importorskip("mlx.core")

try:
    from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast
except Exception:  # pragma: no cover - depends on local native build
    glm_fast = None

# The V4 model module resolves ``from .base import ...`` only once the patch
# has registered it as ``mlx_lm.models.deepseek_v4`` — it cannot be imported
# by its file path.
try:
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()
    from mlx_lm.models.deepseek_v4 import _stable_topk_indices
except Exception:  # pragma: no cover - depends on mlx_lm availability
    _stable_topk_indices = None

pytestmark = pytest.mark.skipif(
    glm_fast is None
    or not glm_fast.is_native_available()
    or not glm_fast.has_symbol("dsa_topk_indices")
    or _stable_topk_indices is None,
    reason="GLM MoE DSA native extension or mlx_lm is unavailable",
)


def _native_sorted(scores, k):
    indices = glm_fast.dsa_topk_indices(scores, k, bucketed=False)
    return mx.sort(indices, axis=-1)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("pooled", [1088, 4096])
def test_topk_1024_matches_stable_reference(dtype, pooled):
    mx.random.seed(3)
    # ReLU-style scores: many exact zero ties, the case the deterministic
    # temporal tie-break exists for.
    scores = mx.maximum(
        mx.random.normal((1, 1, 4, pooled)), 0.0
    ).astype(dtype)
    got = _native_sorted(scores, 1024)
    want = _stable_topk_indices(scores, 1024)
    mx.eval(got, want)
    assert got.shape == (1, 1, 4, 1024)
    assert bool(mx.array_equal(got.astype(mx.uint32), want.astype(mx.uint32)))


def test_topk_1024_exact_pool_size_is_identity():
    mx.random.seed(5)
    scores = mx.random.normal((1, 1, 2, 1024)).astype(mx.float16)
    got = _native_sorted(scores, 1024)
    mx.eval(got)
    want = mx.broadcast_to(mx.arange(1024, dtype=got.dtype), got.shape)
    assert bool(mx.array_equal(got, want))


def test_topk_512_and_2048_unchanged():
    mx.random.seed(7)
    scores = mx.maximum(mx.random.normal((1, 1, 2, 4096)), 0.0).astype(
        mx.float16
    )
    for k in (512, 2048):
        got = _native_sorted(scores, k)
        want = _stable_topk_indices(scores, k)
        mx.eval(got, want)
        assert bool(
            mx.array_equal(got.astype(mx.uint32), want.astype(mx.uint32))
        )


def test_dispatch_gate_accepts_1024():
    from omlx.patches.deepseek_v4.indexer_dispatch import (
        native_indexer_shape_eligible,
    )

    for topk, want in ((512, True), (1024, True), (2048, True), (768, False)):
        assert (
            native_indexer_shape_eligible(
                query_tokens=64,
                pooled_tokens=topk + 64,
                n_heads=64,
                head_dim=128,
                index_topk=topk,
                dtype_supported=True,
            )
            is want
        )
