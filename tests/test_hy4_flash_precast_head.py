"""The optional FP32 head does not alter a single reference logit."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.patches.hy_v4_flash.runner import prepare_fp32_head


def model(*, tied=False, fp32=True, dtype=mx.bfloat16):
    mx.random.seed(2026)
    return SimpleNamespace(
        args=SimpleNamespace(tie_word_embeddings=tied, enable_lm_head_fp32=fp32),
        lm_head=SimpleNamespace(weight=mx.random.normal((511, 64)).astype(dtype)),
    )


def test_identical_fp32_products_and_incremental_allocation():
    m = model()
    original = m.lm_head.weight
    for n in (1, 5):
        x = mx.random.normal((1, n, 64)).astype(mx.bfloat16)
        expected = x.astype(mx.float32) @ original.astype(mx.float32).T
        mx.eval(expected)
        assert prepare_fp32_head(m) == (original.nbytes if n == 1 else 0)
        assert m.lm_head.weight.dtype == mx.float32
        assert mx.array_equal(
            expected, x.astype(mx.float32) @ m.lm_head.weight.T
        ).item()
    assert original.dtype == mx.bfloat16


@pytest.mark.parametrize(
    "tied,fp32,dtype",
    [(True, True, mx.bfloat16), (False, False, mx.bfloat16), (False, True, mx.float16)],
)
def test_incompatible_head_fails_without_mutation(tied, fp32, dtype):
    m = model(tied=tied, fp32=fp32, dtype=dtype)
    before = m.lm_head.weight
    with pytest.raises(ValueError):
        prepare_fp32_head(m)
    assert m.lm_head.weight is before
