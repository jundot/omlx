"""Qwen4 QSA native H=4/D=128 indexer-score regression tests."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import math

import mlx.core as mx
import pytest

from omlx.custom_kernels.glm_moe_dsa import fast
from omlx.patches import mlx_vlm_qwen4_exp_compat as compat


compat.apply_mlx_vlm_qwen4_exp_compat_patch()
qsa_fast = importlib.import_module("mlx_vlm.models.qwen4_exp.qsa_fast")


def _native_available() -> bool:
    return fast.is_native_available() and fast.has_symbol(
        "qwen4_qsa_indexer_scores"
    )


def _reference_scores(q, k, *, mask_ratio=4, mask_q_offset=2048):
    # Native q is [B,H,M,D]; the portable QSA expression uses [B,M,H,D].
    q_bmhd = q.transpose(0, 2, 1, 3)
    scores = q_bmhd.astype(mx.float32) @ k.astype(mx.float32).swapaxes(-1, -2)
    scores = mx.sum(mx.maximum(scores, 0), axis=-2) / math.sqrt(q.shape[-1])
    valid = mx.arange(k.shape[2])[None, None, :] < (
        mask_q_offset + mx.arange(q.shape[2])[None, :, None] + 1
    ) // mask_ratio
    return mx.where(valid, scores, mx.finfo(mx.float32).min)


def _topk_sets(scores, topk):
    indices = mx.argpartition(scores, kth=-topk, axis=-1)[..., -topk:]
    return mx.sort(indices.astype(mx.int32), axis=-1)


def test_qwen4_qsa_symbol_is_part_of_the_extension_abi():
    assert "qwen4_qsa_indexer_scores" in fast.NATIVE_SYMBOLS


def test_qwen4_qsa_production_geometry_routes_to_native_abi(monkeypatch):
    q = mx.zeros((1, 7, 4, 128), dtype=mx.bfloat16)
    k = mx.zeros((1, 19, 128), dtype=mx.bfloat16)
    seen = []

    def scores(queries, pooled_keys, **kwargs):
        seen.append((queries.shape, pooled_keys.shape, kwargs))
        return mx.zeros((1, 7, 19), dtype=mx.float32)

    monkeypatch.setattr(fast, "is_native_available", lambda: True)
    monkeypatch.setattr(fast, "has_symbol", lambda name: True)
    monkeypatch.setattr(fast, "qwen4_qsa_indexer_scores", scores)
    monkeypatch.setattr(qsa_fast, "_NATIVE_QSA_SCORE_DISABLED", False)
    monkeypatch.setattr(qsa_fast, "_NATIVE_QSA_SCORE_PROVEN", False)

    actual = qsa_fast._native_indexer_scores(
        q,
        k,
        head_dim=128,
        compress_ratio=4,
        mask_q_offset=4096,
    )
    assert actual is not None
    assert seen == [
        (
            (1, 4, 7, 128),
            (1, 1, 19, 128),
            {"mask_ratio": 4, "mask_q_offset": 4096},
        )
    ]
    assert qsa_fast._NATIVE_QSA_SCORE_PROVEN is True


def test_qwen4_native_dispatch_rejection_latches_to_portable(monkeypatch):
    q = mx.zeros((1, 3, 4, 128), dtype=mx.bfloat16)
    k = mx.zeros((1, 16, 128), dtype=mx.bfloat16)
    calls = 0

    def reject(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("stale native extension")

    monkeypatch.setattr(fast, "is_native_available", lambda: True)
    monkeypatch.setattr(fast, "has_symbol", lambda name: True)
    monkeypatch.setattr(fast, "qwen4_qsa_indexer_scores", reject)
    monkeypatch.setattr(qsa_fast, "_NATIVE_QSA_SCORE_DISABLED", False)
    monkeypatch.setattr(qsa_fast, "_NATIVE_QSA_SCORE_PROVEN", False)

    kwargs = dict(
        head_dim=128,
        compress_ratio=4,
        mask_q_offset=2048,
    )
    assert qsa_fast._native_indexer_scores(q, k, **kwargs) is None
    assert qsa_fast._native_indexer_scores(q, k, **kwargs) is None
    assert calls == 1


def test_gathered_qsa_native_score_hook_matches_portable_path(monkeypatch):
    mx.random.seed(19)
    queries = mx.random.normal((1, 4, 20, 8)).astype(mx.bfloat16)
    keys = mx.random.normal((1, 2, 20, 8)).astype(mx.bfloat16)
    values = mx.random.normal((1, 2, 20, 8)).astype(mx.bfloat16)
    index_queries = mx.random.normal((1, 20, 4, 128)).astype(mx.bfloat16)
    index_keys = mx.random.normal((1, 20, 128)).astype(mx.bfloat16)
    positions = mx.arange(20, dtype=mx.int32)[None]
    seen_offsets = []

    def fused_scores(q, k, **kwargs):
        offset = kwargs["mask_q_offset"]
        seen_offsets.append(offset)
        scores = qsa_fast._portable_indexer_scores(q, k, 128)
        valid = mx.arange(k.shape[1])[None, None, :] < (
            offset + mx.arange(q.shape[1])[None, :, None] + 1
        ) // 4
        return mx.where(valid, scores, mx.finfo(mx.float32).min)

    kwargs = dict(
        num_query_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        indexer_head_dim=128,
        compress_ratio=4,
        token_budget=8,
        index_key_norm=lambda x: x,
        apply_index_rope=lambda x, position_ids: x,
        query_chunk=7,
    )
    monkeypatch.setattr(qsa_fast, "_native_indexer_scores", fused_scores)
    actual = qsa_fast.contiguous_causal_gathered_qsa(
        queries,
        keys,
        values,
        index_queries,
        index_keys,
        positions,
        **kwargs,
    )
    monkeypatch.setattr(qsa_fast, "_native_indexer_scores", lambda *a, **k: None)
    expected = qsa_fast.contiguous_causal_gathered_qsa(
        queries,
        keys,
        values,
        index_queries,
        index_keys,
        positions,
        **kwargs,
    )
    mx.eval(actual, expected)

    assert seen_offsets == [0, 7, 14]
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.skipif(
    not _native_available(),
    reason="native Qwen4 QSA indexer-score ABI is unavailable",
)
def test_qwen4_qsa_native_scores_and_topk_match_fp32_reference(dtype):
    mx.random.seed(1729)
    q = (mx.random.normal((1, 4, 67, 128)) * 0.25).astype(dtype)
    k = (mx.random.normal((1, 1, 521, 128)) * 0.25).astype(dtype)
    mask_q_offset = 2048

    actual = fast.qwen4_qsa_indexer_scores(
        q,
        k,
        mask_ratio=4,
        mask_q_offset=mask_q_offset,
    )
    expected = _reference_scores(
        q,
        k,
        mask_ratio=4,
        mask_q_offset=mask_q_offset,
    )
    mx.eval(actual, expected)

    assert actual.shape == (1, 67, 521)
    assert actual.dtype == mx.float32
    assert mx.array_equal(actual, expected).item()
    assert mx.array_equal(_topk_sets(actual, 64), _topk_sets(expected, 64)).item()


@pytest.mark.skipif(
    not _native_available(),
    reason="native Qwen4 QSA indexer-score ABI is unavailable",
)
def test_qwen4_qsa_native_fuses_exact_float32_pooled_causal_sentinel():
    q = mx.ones((1, 4, 5, 128), dtype=mx.bfloat16)
    k = mx.ones((1, 1, 11, 128), dtype=mx.bfloat16)
    offset = 8
    actual = fast.qwen4_qsa_indexer_scores(
        q,
        k,
        mask_ratio=4,
        mask_q_offset=offset,
    )
    mx.eval(actual)

    valid = mx.arange(11)[None, None, :] < (
        offset + mx.arange(5)[None, :, None] + 1
    ) // 4
    invalid_values = mx.where(valid, mx.finfo(mx.float32).min, actual)
    assert mx.array_equal(
        invalid_values,
        mx.full(actual.shape, mx.finfo(mx.float32).min),
    ).item()


@pytest.mark.skipif(
    not _native_available(),
    reason="native Qwen4 QSA indexer-score ABI is unavailable",
)
def test_qwen4_qsa_native_abi_rejects_nonproduction_geometry():
    with pytest.raises(ValueError, match="expected q"):
        fast.qwen4_qsa_indexer_scores(
            mx.zeros((1, 3, 8, 128), dtype=mx.bfloat16),
            mx.zeros((1, 1, 16, 128), dtype=mx.bfloat16),
            mask_q_offset=64,
        )
