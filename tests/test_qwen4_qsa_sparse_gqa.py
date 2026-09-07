# SPDX-License-Identifier: Apache-2.0
"""Narrow native Qwen4 QSA main-attention regression tests."""

from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.custom_kernels.glm_moe_dsa import fast
from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

compat.apply_mlx_vlm_qwen4_exp_compat_patch()
from mlx_vlm.models.qwen4_exp import qsa_fast  # noqa: E402


def _native_available() -> bool:
    return fast.is_native_available() and fast.has_symbol(
        "qwen4_qsa_sparse_gqa_attention"
    )


def test_qwen4_sparse_gqa_symbol_is_part_of_extension_abi():
    assert "qwen4_qsa_sparse_gqa_attention" in fast.NATIVE_SYMBOLS


def test_qwen4_sparse_gqa_route_forwards_compact_blocks_and_transposes(monkeypatch):
    queries = mx.zeros((1, 24, 3, 256), dtype=mx.bfloat16)
    keys = mx.zeros((1, 2, 20, 256), dtype=mx.bfloat16)
    values = mx.zeros_like(keys)
    blocks = mx.broadcast_to(
        mx.arange(512, dtype=mx.int32)[None, None],
        (1, 3, 512),
    )
    calls = []

    monkeypatch.setattr(fast, "is_native_available", lambda: True)
    monkeypatch.setattr(fast, "has_symbol", lambda name: True)

    def native(
        q,
        k,
        v,
        selected,
        scale,
        q_offset,
        *,
        key_tile=128,
        dimension_tile=32,
        stream=None,
    ):
        del k, v, stream
        mx.eval(selected)
        calls.append(
            (selected.shape, selected.dtype, scale, q_offset, key_tile, dimension_tile)
        )
        return mx.zeros(q.shape, dtype=q.dtype)

    monkeypatch.setattr(fast, "qwen4_qsa_sparse_gqa_attention", native)
    monkeypatch.setattr(qsa_fast, "_NATIVE_QSA_MAIN_DISABLED", False)
    monkeypatch.setattr(qsa_fast, "_NATIVE_QSA_MAIN_PROVEN", False)

    output = qsa_fast._native_sparse_gqa_attention(
        queries,
        keys,
        values,
        blocks,
        q_offset=10,
    )
    assert output is not None
    mx.eval(output)
    assert output.shape == (1, 3, 24, 256)
    assert calls == [
        (
            (1, 1, 3, 512),
            mx.uint32,
            256**-0.5,
            10,
            64,
            64,
        )
    ]


def test_qwen4_sparse_gqa_route_fails_closed_outside_production_geometry(
    monkeypatch,
):
    monkeypatch.setattr(qsa_fast, "_NATIVE_QSA_MAIN_DISABLED", False)
    bad_queries = mx.zeros((1, 4, 2, 256), dtype=mx.bfloat16)
    keys = mx.zeros((1, 2, 4, 256), dtype=mx.bfloat16)
    blocks = mx.zeros((1, 2, 3), dtype=mx.int32)
    assert (
        qsa_fast._native_sparse_gqa_attention(
            bad_queries,
            keys,
            keys,
            blocks,
            q_offset=2,
        )
        is None
    )


def test_qwen4_prefill_restores_chronological_selected_order(monkeypatch):
    mx.random.seed(81)
    total = 10
    queries = mx.random.normal((1, 24, total, 256)).astype(mx.float16)
    keys = mx.random.normal((1, 2, total, 256)).astype(mx.float16)
    values = mx.random.normal((1, 2, total, 256)).astype(mx.float16)
    index_queries = mx.random.normal((1, total, 2, 8)).astype(mx.float16)
    index_keys = mx.random.normal((1, total, 8)).astype(mx.float16)
    positions = mx.arange(total, dtype=mx.int32)[None]
    captured = []

    monkeypatch.setattr(qsa_fast, "_native_indexer_scores", lambda *a, **k: None)

    def reverse_topk(scores, topk):
        del scores
        return mx.broadcast_to(
            mx.arange(topk - 1, -1, -1, dtype=mx.int32)[None, None],
            (1, total, topk),
        )

    monkeypatch.setattr(qsa_fast, "_native_topk_indices", reverse_topk)

    def capture(q, k, v, selected, *, q_offset):
        del k, v, q_offset
        captured.append(selected)
        return mx.zeros((1, q.shape[2], 24, 256), dtype=q.dtype)

    monkeypatch.setattr(qsa_fast, "_native_sparse_gqa_attention", capture)
    output = qsa_fast.contiguous_causal_gathered_qsa(
        queries,
        keys,
        values,
        index_queries,
        index_keys,
        positions,
        num_query_heads=24,
        num_key_value_heads=2,
        head_dim=256,
        indexer_head_dim=8,
        compress_ratio=2,
        token_budget=8,
        index_key_norm=lambda x: x,
        apply_index_rope=lambda x, p: x,
        query_chunk=total,
    )
    mx.eval(output, *captured)
    assert output.shape == (1, total, 24, 256)
    selected = captured[0]
    assert selected[0, -1].tolist() == [0, 1, 2, 3]


@pytest.mark.skipif(not _native_available(), reason="native Qwen4 GQA not built")
@pytest.mark.parametrize(
    ("key_tile", "dimension_tile"),
    [(64, 64), (128, 32), (256, 32)],
)
def test_qwen4_sparse_gqa_native_matches_fp32_gather_reference(
    key_tile,
    dimension_tile,
):
    mx.random.seed(121)
    query_tokens = 17
    key_tokens = 2111
    selected_blocks = 512
    q_offset = key_tokens - query_tokens
    queries = mx.random.normal((1, 24, query_tokens, 256)).astype(mx.bfloat16)
    keys = mx.random.normal((1, 2, key_tokens, 256)).astype(mx.bfloat16)
    values = mx.random.normal((1, 2, key_tokens, 256)).astype(mx.bfloat16)
    starts = mx.arange(query_tokens, dtype=mx.int32) + q_offset
    blocks = mx.stack(
        [
            mx.arange(
                (int(end) + 1) // 4 - selected_blocks,
                (int(end) + 1) // 4,
                dtype=mx.int32,
            )
            for end in starts
        ],
        axis=0,
    )[None]

    native = fast.qwen4_qsa_sparse_gqa_attention(
        queries,
        keys,
        values,
        blocks[:, None].astype(mx.uint32),
        256**-0.5,
        q_offset,
        key_tile=key_tile,
        dimension_tile=dimension_tile,
    )
    complete = (starts + 1) // 4
    expanded = (
        blocks[..., None] * 4 + mx.arange(4, dtype=mx.int32)
    ).reshape(1, query_tokens, 2048)
    tail = complete[None, :, None] * 4 + mx.arange(3, dtype=mx.int32)
    tail_valid = tail <= starts[None, :, None]
    selected = mx.concatenate((expanded, tail), axis=-1)
    selected_valid = mx.concatenate(
        (mx.ones(expanded.shape, dtype=mx.bool_), tail_valid), axis=-1
    )
    key_rows = keys.transpose(0, 2, 1, 3)
    value_rows = values.transpose(0, 2, 1, 3)
    safe = mx.where(selected_valid, selected, 0)
    gathered_k = qsa_fast._batch_gather_tokens(key_rows, safe).transpose(
        0, 1, 3, 2, 4
    )
    gathered_v = qsa_fast._batch_gather_tokens(value_rows, safe).transpose(
        0, 1, 3, 2, 4
    )
    grouped_q = queries.transpose(0, 2, 1, 3).reshape(
        1, query_tokens, 2, 12, 256
    )
    scores = (
        grouped_q.astype(mx.float32)
        @ gathered_k.astype(mx.float32).swapaxes(-1, -2)
    ) / (256**0.5)
    scores = mx.where(
        selected_valid[:, :, None, None],
        scores,
        mx.finfo(scores.dtype).min,
    )
    probs = mx.softmax(scores, axis=-1).astype(queries.dtype)
    reference = (probs @ gathered_v).reshape(1, query_tokens, 24, 256)
    native_rows = native.transpose(0, 2, 1, 3)
    mx.eval(native_rows, reference)
    max_error = mx.max(mx.abs(native_rows.astype(mx.float32) - reference.astype(mx.float32)))
    assert float(max_error.item()) <= 5e-3


@pytest.mark.skipif(not _native_available(), reason="native Qwen4 GQA not built")
def test_qwen4_sparse_gqa_native_masks_future_blocks_in_first_chunk():
    """Canonical 0..511 placeholders must not expose future first-chunk K/V."""

    mx.random.seed(313)
    query_tokens = 33
    key_tokens = 4096
    queries = mx.random.normal((1, 24, query_tokens, 256)).astype(mx.bfloat16)
    keys = mx.random.normal((1, 2, key_tokens, 256)).astype(mx.bfloat16)
    values = mx.random.normal((1, 2, key_tokens, 256)).astype(mx.bfloat16)
    blocks = mx.broadcast_to(
        mx.arange(512, dtype=mx.uint32)[None, None],
        (1, query_tokens, 512),
    )

    native = fast.qwen4_qsa_sparse_gqa_attention(
        queries,
        keys,
        values,
        blocks[:, None],
        256**-0.5,
        0,
        key_tile=64,
        dimension_tile=64,
    ).transpose(0, 2, 1, 3)

    visible = mx.arange(1, query_tokens + 1, dtype=mx.int32)[None]
    complete = visible // 4
    block_valid = mx.arange(512)[None, None, :] < complete[..., None]
    expanded = (
        blocks.astype(mx.int32)[..., None] * 4
        + mx.arange(4, dtype=mx.int32)
    ).reshape(1, query_tokens, 2048)
    expanded_valid = mx.broadcast_to(
        block_valid[..., None], (1, query_tokens, 512, 4)
    ).reshape(1, query_tokens, 2048)
    tail = complete[..., None] * 4 + mx.arange(3, dtype=mx.int32)
    tail_valid = tail < visible[..., None]
    selected = mx.concatenate((expanded, tail), axis=-1)
    selected_valid = mx.concatenate((expanded_valid, tail_valid), axis=-1)
    safe = mx.where(selected_valid, selected, 0)

    gathered_k = qsa_fast._batch_gather_tokens(
        keys.transpose(0, 2, 1, 3), safe
    ).transpose(0, 1, 3, 2, 4)
    gathered_v = qsa_fast._batch_gather_tokens(
        values.transpose(0, 2, 1, 3), safe
    ).transpose(0, 1, 3, 2, 4)
    grouped_q = queries.transpose(0, 2, 1, 3).reshape(
        1, query_tokens, 2, 12, 256
    )
    scores = (
        grouped_q.astype(mx.float32)
        @ gathered_k.astype(mx.float32).swapaxes(-1, -2)
    ) / (256**0.5)
    scores = mx.where(
        selected_valid[:, :, None, None],
        scores,
        mx.finfo(scores.dtype).min,
    )
    reference = (
        mx.softmax(scores, axis=-1).astype(queries.dtype) @ gathered_v
    ).reshape(1, query_tokens, 24, 256)
    mx.eval(native, reference)

    error = mx.abs(native.astype(mx.float32) - reference.astype(mx.float32))
    # The zero-prefix row has exactly one visible value and must therefore be
    # bit-identical; this is the strongest future-leak sentinel. Later tiny
    # rows differ by at most one BF16 output ULP because native online softmax
    # keeps probabilities in FP32 while the portable oracle casts them first.
    assert mx.array_equal(native[:, :1], reference[:, :1]).item()
    assert float(mx.max(error).item()) <= 2e-2
