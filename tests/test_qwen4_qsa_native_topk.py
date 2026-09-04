"""Lossless native Qwen4 QSA FP32 top-512 regression tests."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from omlx.custom_kernels.glm_moe_dsa import fast

TOPK = 512


def _native_available() -> bool:
    return fast.is_native_available() and fast.has_symbol("qwen4_qsa_topk_indices")


def _reference_indices(scores: mx.array) -> mx.array:
    return mx.argpartition(scores, kth=-TOPK, axis=-1)[..., -TOPK:]


def _sorted_sets(indices: mx.array) -> mx.array:
    return mx.sort(indices.astype(mx.int32), axis=-1)


def test_qwen4_qsa_topk_symbol_is_part_of_the_extension_abi():
    assert "qwen4_qsa_topk_indices" in fast.NATIVE_SYMBOLS


@pytest.mark.skipif(
    not _native_available(),
    reason="native Qwen4 QSA top-k ABI is unavailable",
)
@pytest.mark.parametrize("blocks", [12_500, 25_000])
def test_qwen4_qsa_topk_matches_argpartition_at_50k_100k_equivalent(blocks):
    """50K/100K tokens at Qwen4's four-token compression ratio."""
    rng = np.random.default_rng(41 + blocks)
    scores = rng.standard_normal((1, 4, blocks), dtype=np.float32)
    actual = fast.qwen4_qsa_topk_indices(mx.array(scores))
    expected = _reference_indices(mx.array(scores))
    mx.eval(actual, expected)

    assert actual.shape == (1, 4, TOPK)
    assert actual.dtype == mx.uint32
    assert mx.array_equal(_sorted_sets(actual), _sorted_sets(expected)).item()


@pytest.mark.skipif(
    not _native_available(),
    reason="native Qwen4 QSA top-k ABI is unavailable",
)
def test_qwen4_qsa_topk_cutoff_ties_match_argpartition_membership():
    """MLX keeps the highest indices when a tie crosses the top-k cutoff."""
    scores = np.full((1, 1, 2048), -4.0, dtype=np.float32)
    tie_indices = np.arange(200, 1600, dtype=np.int32)
    strict_indices = np.concatenate(
        (
            np.arange(0, 73, dtype=np.int32),
            np.arange(1800, 1887, dtype=np.int32),
        )
    )
    scores[0, 0, tie_indices] = np.float32(0.5)
    scores[0, 0, strict_indices] = np.float32(1.0)

    actual = fast.qwen4_qsa_topk_indices(mx.array(scores))
    reference = _reference_indices(mx.array(scores))
    mx.eval(actual, reference)

    tie_budget = TOPK - strict_indices.size
    expected = np.sort(np.concatenate((strict_indices, tie_indices[-tie_budget:])))
    actual_set = np.sort(np.asarray(actual, dtype=np.uint32)[0, 0])
    reference_set = np.sort(np.asarray(reference, dtype=np.uint32)[0, 0])
    np.testing.assert_array_equal(actual_set, expected)
    np.testing.assert_array_equal(actual_set, reference_set)


@pytest.mark.skipif(
    not _native_available(),
    reason="native Qwen4 QSA top-k ABI is unavailable",
)
def test_qwen4_qsa_topk_treats_signed_zero_as_one_numeric_tie():
    scores = np.empty((1, 1, 1024), dtype=np.float32)
    scores[..., ::2] = np.float32(-0.0)
    scores[..., 1::2] = np.float32(0.0)
    actual = fast.qwen4_qsa_topk_indices(mx.array(scores))
    reference = _reference_indices(mx.array(scores))
    mx.eval(actual, reference)
    assert mx.array_equal(_sorted_sets(actual), _sorted_sets(reference)).item()


@pytest.mark.skipif(
    not _native_available(),
    reason="native Qwen4 QSA top-k ABI is unavailable",
)
@pytest.mark.parametrize("sentinel", [float("-inf"), np.finfo(np.float32).min])
def test_qwen4_qsa_topk_matches_causal_sentinel_rows(sentinel):
    """Sentinels sort below valid ties and keep MLX's tie membership."""
    blocks = 2048
    valid_counts = (257, 513, 617, 1025, 1537)
    scores = np.full((1, len(valid_counts), blocks), sentinel, dtype=np.float32)
    # Exact zero bands exercise QSA's ReLU ties at the same time as the mask.
    for row, valid in enumerate(valid_counts):
        scores[0, row, :valid] = np.float32(0.0)

    actual = fast.qwen4_qsa_topk_indices(mx.array(scores))
    reference = _reference_indices(mx.array(scores))
    mx.eval(actual, reference)

    assert mx.array_equal(_sorted_sets(actual), _sorted_sets(reference)).item()
    actual_np = np.asarray(actual, dtype=np.uint32)[0]
    for row, valid in enumerate(valid_counts):
        if valid >= TOPK:
            assert np.all(actual_np[row] < valid)
            expected = np.arange(valid - TOPK, valid, dtype=np.uint32)
        else:
            # The ranked branch is discarded by QSA's canonical short-row
            # selection, but the native primitive itself still matches MLX:
            # every valid zero plus the highest-index sentinel ties.
            expected = np.concatenate(
                (
                    np.arange(valid, dtype=np.uint32),
                    np.arange(blocks - (TOPK - valid), blocks, dtype=np.uint32),
                )
            )
        np.testing.assert_array_equal(np.sort(actual_np[row]), expected)


@pytest.mark.skipif(
    not _native_available(),
    reason="native Qwen4 QSA top-k ABI is unavailable",
)
@pytest.mark.parametrize(
    ("scores", "topk"),
    [
        (mx.zeros((2, 1, 512), dtype=mx.float32), 512),
        (mx.zeros((1, 512), dtype=mx.float32), 512),
        (mx.zeros((1, 1, 512), dtype=mx.float16), 512),
        (mx.zeros((1, 1, 511), dtype=mx.float32), 512),
        (mx.zeros((1, 1, 512), dtype=mx.float32), 256),
    ],
)
def test_qwen4_qsa_topk_fails_closed_outside_fixed_contract(scores, topk):
    with pytest.raises(ValueError, match="qwen4_qsa_topk_indices"):
        fast.qwen4_qsa_topk_indices(scores, topk)
