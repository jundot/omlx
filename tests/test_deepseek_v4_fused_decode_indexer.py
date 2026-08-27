# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4 fused decode indexer (dsa_decode_scores) parity tests.

The DSpark verify/decode indexer score pipeline routes through the fused
``dsa_decode_scores`` Metal scan (64-head instantiation, fp32 score output,
fp32 head weights) instead of casting the pooled context to fp32 and
materializing a (rows, 64, P) score sheet. These tests pin:

* score parity vs the fp32 reference reduction (relu(q . k) * scale,
  weighted by projected_weights * n_heads**-0.5, summed over heads),
* exact top-k index-set parity through the unchanged downstream selection
  stage (``dspark_fp32_topk_indices`` / ``_stable_topk_indices``),
* the 6-row DSpark verify batch with +/-1 pooled-length mismatch: the padded
  tail must stay excluded from top-k,
* the deliberate exclusion of single-row calls (M=1 decode / depth-1 verify),
  where the fused scan measures slower than the reference GEMV,
* the env opt-out (OMLX_DSV4_FUSED_DECODE_INDEXER=0, read at import) and the
  runtime fallback to the fp32 reference path on kernel error.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest


@pytest.fixture(scope="module")
def dsv4():
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()
    import mlx_lm.models.deepseek_v4 as module

    return module


@pytest.fixture(scope="module")
def glm_fast():
    from omlx.custom_kernels.glm_moe_dsa import fast

    if not (
        fast.is_native_available() and fast.has_symbol("dsa_decode_scores")
    ):
        pytest.skip("glm_moe_dsa native extension with dsa_decode_scores not built")
    return fast


@pytest.fixture(autouse=True)
def _reset_fused_state(dsv4):
    """Each test starts with the fused path enabled and not runtime-failed."""
    old_env = dsv4._DEEPSEEK_V4_FUSED_DECODE_INDEXER_ENV_DISABLED
    old_failed = dsv4._DEEPSEEK_V4_FUSED_DECODE_INDEXER_FAILED
    dsv4._DEEPSEEK_V4_FUSED_DECODE_INDEXER_ENV_DISABLED = False
    dsv4._DEEPSEEK_V4_FUSED_DECODE_INDEXER_FAILED = False
    yield
    dsv4._DEEPSEEK_V4_FUSED_DECODE_INDEXER_ENV_DISABLED = old_env
    dsv4._DEEPSEEK_V4_FUSED_DECODE_INDEXER_FAILED = old_failed


def _reference_scores(pooled, q, w, n_heads, scale):
    """The fp32 reference reduction used by the pre-fused decode path."""
    qf = q[:, :, 0].astype(mx.float32)  # (B, H, D)
    kf = pooled[:, 0].swapaxes(-1, -2).astype(mx.float32)  # (B, D, P)
    scores = mx.maximum(qf @ kf, 0) * scale  # (B, H, P)
    weights = w.astype(mx.float32) * (n_heads**-0.5)  # (B, H)
    return (scores * weights[..., None]).sum(axis=1)  # (B, P)


def _make_indexer(dsv4, **overrides):
    config = dsv4.ModelArgs(
        hidden_size=16,
        q_lora_rank=16,
        qk_rope_head_dim=2,
        num_hidden_layers=1,
        compress_ratios=[4],
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return dsv4.Indexer(config, compress_ratio=4)


@pytest.mark.parametrize("pooled_length", [2048, 4096, 8192])
def test_fused_decode_scores_match_fp32_reference(dsv4, glm_fast, pooled_length):
    mx.random.seed(1000 + pooled_length)
    heads, head_dim, topk = 64, 128, 512
    scale = head_dim**-0.5

    q = mx.random.normal((1, heads, 1, head_dim), dtype=mx.bfloat16)
    pooled = mx.random.normal((1, 1, pooled_length, head_dim), dtype=mx.bfloat16)
    w = mx.random.normal((1, heads), dtype=mx.float32)

    fused = glm_fast.dsa_decode_scores(
        q, pooled, w * ((heads**-0.5) * scale), fp32_scores=True
    )
    reference = _reference_scores(pooled, q, w, heads, scale)
    mx.eval(fused, reference)

    fused_np = np.array(fused[0, 0, 0])
    ref_np = np.array(reference[0])
    # fp32 weights in both paths: only the dot-product accumulation order
    # differs, so agreement is at fp32 rounding scale even at the cutoff.
    assert np.abs(fused_np - ref_np).max() < 1e-4

    # The downstream top-k stage is shared, so identical scores must give
    # identical index sets.
    fused_topk = glm_fast.dspark_fp32_topk_indices(
        mx.contiguous(fused.reshape(1, pooled_length)), topk
    )
    ref_topk = glm_fast.dspark_fp32_topk_indices(
        mx.contiguous(reference), topk
    )
    mx.eval(fused_topk, ref_topk)
    assert set(np.array(fused_topk[0]).tolist()) == set(
        np.array(ref_topk[0]).tolist()
    )


def test_fused_decode_scores_long_context_parity(dsv4, glm_fast):
    """~100K pooled tokens: scores stay at fp32 rounding scale, top-k equal."""
    mx.random.seed(4242)
    heads, head_dim, topk = 64, 128, 512
    scale = head_dim**-0.5
    pooled_length = 100_352

    q = mx.random.normal((1, heads, 1, head_dim), dtype=mx.bfloat16)
    pooled = mx.random.normal((1, 1, pooled_length, head_dim), dtype=mx.bfloat16)
    w = mx.random.normal((1, heads), dtype=mx.float32)

    fused = glm_fast.dsa_decode_scores(
        q, pooled, w * ((heads**-0.5) * scale), fp32_scores=True
    )
    reference = _reference_scores(pooled, q, w, heads, scale)
    mx.eval(fused, reference)

    fused_np = np.array(fused[0, 0, 0])
    ref_np = np.array(reference[0])
    max_abs = np.abs(fused_np - ref_np).max()
    assert max_abs < 1e-4, f"score drift at 100K: {max_abs}"

    fused_topk = glm_fast.dspark_fp32_topk_indices(
        mx.contiguous(fused.reshape(1, pooled_length)), topk
    )
    ref_topk = glm_fast.dspark_fp32_topk_indices(
        mx.contiguous(reference), topk
    )
    mx.eval(fused_topk, ref_topk)
    assert set(np.array(fused_topk[0]).tolist()) == set(
        np.array(ref_topk[0]).tolist()
    )


def test_fused_decode_scores_tie_band_matches_stable_topk(dsv4, glm_fast):
    """ReLU'd indexer scores carry exact-zero ties at the cutoff. The native
    fp32 top-k resolves cutoff ties lowest-index-first, bit-identical to
    ``_stable_topk_indices`` — that contract is what the fused route relies
    on, so pin it directly."""
    mx.random.seed(31)
    size, topk = 4096, 512
    scores = np.zeros(size, dtype=np.float32)
    scores[:1600] = np.random.default_rng(3).random(1600).astype(np.float32)
    # Force a wide exact-tie band straddling the cutoff.
    scores[800:2000] = np.float32(0.5)
    scores_mx = mx.array(scores)[None]

    native = glm_fast.dspark_fp32_topk_indices(scores_mx, topk)
    stable = dsv4._stable_topk_indices(scores_mx, topk)
    mx.eval(native, stable)
    assert set(np.array(native[0]).tolist()) == set(
        np.array(stable[0]).tolist()
    )


def test_batch_indexer_rows_fused_matches_fp32_six_rows(dsv4, glm_fast):
    """The 6-row DSpark verify batch with +/-1 pooled-length mismatch."""
    mx.random.seed(911)
    lengths = (2048, 2049, 2049, 8192, 8193, 8193)
    indexer = SimpleNamespace(index_topk=512, n_heads=64, scale=128**-0.5)
    pooled_rows = [
        mx.random.normal((1, length, 128), dtype=mx.bfloat16)
        for length in lengths
    ]
    projected_q = mx.random.normal((1, 64, 6, 128), dtype=mx.bfloat16)
    projected_weights = mx.random.normal((1, 6, 64), dtype=mx.bfloat16)

    fused = dsv4._batch_indexer_rows(
        indexer, pooled_rows, projected_q, projected_weights
    )
    mx.eval(*fused)
    assert not dsv4._DEEPSEEK_V4_FUSED_DECODE_INDEXER_FAILED

    dsv4._DEEPSEEK_V4_FUSED_DECODE_INDEXER_ENV_DISABLED = True
    reference = dsv4._batch_indexer_rows(
        indexer, pooled_rows, projected_q, projected_weights
    )
    mx.eval(*reference)
    dsv4._DEEPSEEK_V4_FUSED_DECODE_INDEXER_ENV_DISABLED = False

    assert len(fused) == len(reference) == 6
    for idx, (candidate, expected) in enumerate(zip(fused, reference)):
        # Same scores to accumulation-order rounding through the same top-k
        # stage select the same set in the same sorted order.
        assert mx.array_equal(candidate, expected).item(), f"row {idx}"
        # The +/-1 padded tail stays excluded from the selection.
        assert int(mx.max(candidate).item()) < lengths[idx]
        assert candidate.dtype == mx.uint32


def test_batch_indexer_rows_fused_equal_length_group(dsv4, glm_fast):
    """Equal-length rows take the grouped branch; fused must match there too."""
    mx.random.seed(912)
    indexer = SimpleNamespace(index_topk=512, n_heads=64, scale=128**-0.5)
    pooled_rows = [
        mx.random.normal((1, 4096, 128), dtype=mx.bfloat16) for _ in range(6)
    ]
    projected_q = mx.random.normal((1, 64, 6, 128), dtype=mx.bfloat16)
    projected_weights = mx.random.normal((1, 6, 64), dtype=mx.bfloat16)

    fused = dsv4._batch_indexer_rows(
        indexer, pooled_rows, projected_q, projected_weights
    )
    mx.eval(*fused)
    dsv4._DEEPSEEK_V4_FUSED_DECODE_INDEXER_ENV_DISABLED = True
    reference = dsv4._batch_indexer_rows(
        indexer, pooled_rows, projected_q, projected_weights
    )
    mx.eval(*reference)
    dsv4._DEEPSEEK_V4_FUSED_DECODE_INDEXER_ENV_DISABLED = False

    assert all(
        mx.array_equal(candidate, expected).item()
        for candidate, expected in zip(fused, reference)
    )


def test_batch_indexer_rows_fused_runtime_fallback(dsv4, glm_fast, monkeypatch):
    """A kernel error mid-batch falls back to the fp32 path and latches off."""
    mx.random.seed(913)
    indexer = SimpleNamespace(index_topk=512, n_heads=64, scale=128**-0.5)
    pooled_rows = [
        mx.random.normal((1, 2049, 128), dtype=mx.bfloat16),
        mx.random.normal((1, 2050, 128), dtype=mx.bfloat16),
    ]
    projected_q = mx.random.normal((1, 64, 2, 128), dtype=mx.bfloat16)
    projected_weights = mx.random.normal((1, 2, 64), dtype=mx.bfloat16)

    def boom(*args, **kwargs):
        raise RuntimeError("injected kernel failure")

    monkeypatch.setattr(glm_fast, "dsa_decode_scores", boom)
    fused = dsv4._batch_indexer_rows(
        indexer, pooled_rows, projected_q, projected_weights
    )
    mx.eval(*fused)
    assert dsv4._DEEPSEEK_V4_FUSED_DECODE_INDEXER_FAILED

    dsv4._DEEPSEEK_V4_FUSED_DECODE_INDEXER_ENV_DISABLED = True
    dsv4._DEEPSEEK_V4_FUSED_DECODE_INDEXER_FAILED = False
    reference = dsv4._batch_indexer_rows(
        indexer, pooled_rows, projected_q, projected_weights
    )
    mx.eval(*reference)
    # The fallback result is bit-identical to the plain fp32 path.
    assert all(
        mx.array_equal(candidate, expected).item()
        for candidate, expected in zip(fused, reference)
    )


def test_fused_decode_indexer_env_opt_out(dsv4):
    """The import-time env flag (OMLX_DSV4_FUSED_DECODE_INDEXER=0) keeps the
    helper off entirely: no kernel dispatch is attempted and the fp32 path
    produces the row."""
    mx.random.seed(914)
    indexer = SimpleNamespace(index_topk=512, n_heads=64, scale=128**-0.5)
    pooled_rows = [mx.random.normal((1, 2048, 128), dtype=mx.bfloat16)]
    projected_q = mx.random.normal((1, 64, 1, 128), dtype=mx.bfloat16)
    projected_weights = mx.random.normal((1, 1, 64), dtype=mx.bfloat16)

    dsv4._DEEPSEEK_V4_FUSED_DECODE_INDEXER_ENV_DISABLED = True
    assert (
        dsv4._dspark_fused_indexer_scores(
            indexer, pooled_rows, projected_q, projected_weights
        )
        is None
    )
    result = dsv4._batch_indexer_rows(
        indexer, pooled_rows, projected_q, projected_weights
    )
    mx.eval(*result)
    assert result[0] is not None
    assert result[0].shape == (1, 1, 512)


def test_m1_decode_stays_on_reference_path(dsv4, glm_fast, monkeypatch):
    """M=1 decode deliberately does NOT route through the fused scan: the
    single-row dispatch measures slower than the fp32 GEMV reference at long
    context (documented on _dspark_fused_indexer_scores). Pin that contract
    so a future kernel revision re-evaluates it deliberately."""
    mx.random.seed(915)
    pooled_length = 4096
    indexer = _make_indexer(dsv4)
    pooled = mx.random.normal((1, pooled_length, 128), dtype=mx.bfloat16)
    monkeypatch.setattr(
        dsv4.Compressor,
        "__call__",
        lambda self, x, pool_cache, offset: pooled,
    )

    x = mx.random.normal((1, 1, 16), dtype=mx.bfloat16)
    projected_q = mx.random.normal((1, 64, 1, 128), dtype=mx.bfloat16)
    projected_weights = mx.random.normal((1, 1, 64), dtype=mx.bfloat16)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("fused helper called for M=1 decode")

    monkeypatch.setattr(dsv4, "_dspark_fused_indexer_scores", fail_if_called)
    result = indexer(
        x,
        q_residual=x,
        position_rope=None,
        pool_cache=None,
        offset=0,
        projected_q=projected_q,
        projected_weights=projected_weights,
    )
    mx.eval(result)
    assert result.shape == (1, 1, 512)

    # The reference reduction must produce exactly these indices.
    reference = _reference_scores(
        pooled[:, None], projected_q, projected_weights[0], 64, 128**-0.5
    )
    expected = dsv4._stable_topk_indices(reference[None], 512)
    mx.eval(expected)
    assert mx.array_equal(result, expected).item()


def test_single_verify_row_stays_fp32(dsv4, glm_fast):
    """A one-row verify batch (depth-1 DSpark block) declines the fused scan
    and still returns the reference-path indices."""
    mx.random.seed(916)
    indexer = SimpleNamespace(index_topk=512, n_heads=64, scale=128**-0.5)
    pooled_rows = [mx.random.normal((1, 4096, 128), dtype=mx.bfloat16)]
    projected_q = mx.random.normal((1, 64, 1, 128), dtype=mx.bfloat16)
    projected_weights = mx.random.normal((1, 1, 64), dtype=mx.bfloat16)

    assert (
        dsv4._dspark_fused_indexer_scores(
            indexer, pooled_rows, projected_q, projected_weights
        )
        is None
    )
    result = dsv4._batch_indexer_rows(
        indexer, pooled_rows, projected_q, projected_weights
    )
    mx.eval(*result)
    assert not dsv4._DEEPSEEK_V4_FUSED_DECODE_INDEXER_FAILED

    reference = _reference_scores(
        pooled_rows[0][:, None], projected_q, projected_weights[0], 64, 128**-0.5
    )
    expected = dsv4._stable_topk_indices(reference, 512)
    mx.eval(expected)
    assert mx.array_equal(result[0][0], expected).item()


def test_fused_decode_scores_rejects_fp16_weight_mismatch(glm_fast):
    """fp32 weights are accepted only for the 64-head fp32-score route."""
    q = mx.zeros((1, 32, 1, 128), dtype=mx.bfloat16)
    k = mx.zeros((1, 1, 2048, 128), dtype=mx.bfloat16)
    w = mx.zeros((1, 32), dtype=mx.float32)
    with pytest.raises(Exception):
        mx.eval(glm_fast.dsa_decode_scores(q, k, w, fp32_scores=True))


def test_fused_decode_scores_h32_wsame_regression(dsv4, glm_fast):
    """GLM's historical 32-head same-dtype route is unchanged."""
    mx.random.seed(917)
    heads, head_dim, pooled_length = 32, 128, 4096
    q = mx.random.normal((1, heads, 1, head_dim), dtype=mx.bfloat16)
    pooled = mx.random.normal((1, 1, pooled_length, head_dim), dtype=mx.bfloat16)
    w = mx.random.normal((1, heads), dtype=mx.bfloat16)

    scores = glm_fast.dsa_decode_scores(q, pooled, w)
    mx.eval(scores)
    assert scores.shape == (1, 1, 1, pooled_length)
    assert scores.dtype == mx.bfloat16

    # GLM folds its weight scale into w before the call; the kernel computes
    # sum_h relu(q . k) * w_h with fp32 accumulation.
    qf = q[:, :, 0].astype(mx.float32)
    kf = pooled[:, 0].swapaxes(-1, -2).astype(mx.float32)
    reference = (mx.maximum(qf @ kf, 0) * w.astype(mx.float32)[..., None]).sum(
        axis=1
    )
    mx.eval(reference)
    ref_np = np.array(reference[0])
    got_np = np.array(scores[0, 0, 0].astype(mx.float32))
    # bf16 score output: relative agreement at bf16 precision.
    close = np.isclose(got_np, ref_np, rtol=2e-2, atol=1e-2)
    assert close.mean() > 0.99
