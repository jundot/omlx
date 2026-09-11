# SPDX-License-Identifier: Apache-2.0
"""Exercise prefill tile selection through the gathered QSA entry point."""

from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.custom_kernels.glm_moe_dsa import fast
from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

compat.apply_mlx_vlm_qwen4_exp_compat_patch()
from mlx_vlm.models.qwen4_exp import qsa_fast  # noqa: E402


@pytest.fixture
def native_calls(monkeypatch):
    """Record dispatch geometry without allocating gathered K/V per query."""
    calls = []
    monkeypatch.setattr(fast, "is_native_available", lambda: True)
    monkeypatch.setattr(fast, "has_symbol", lambda name: True)
    for stage in ("MAIN", "SCORE", "TOPK"):
        monkeypatch.setattr(qsa_fast, f"_NATIVE_QSA_{stage}_PROVEN", True)
        monkeypatch.setattr(qsa_fast, f"_NATIVE_QSA_{stage}_DISABLED", False)

    def scores(q, pooled, **kwargs):
        return mx.zeros((1, q.shape[1], pooled.shape[1]), dtype=mx.float32)

    def topk(scores, k):
        return mx.broadcast_to(mx.arange(k), (*scores.shape[:-1], k))

    def attention(q, k, v, selected, *, q_offset):
        calls.append((q.shape[2], q_offset))
        return q.transpose(0, 2, 1, 3)

    monkeypatch.setattr(qsa_fast, "_native_indexer_scores", scores)
    monkeypatch.setattr(qsa_fast, "_native_topk_indices", topk)
    monkeypatch.setattr(qsa_fast, "_native_sparse_gqa_attention", attention)
    return calls


def _run(*, rows=2049, tokens=4096, dtype=mx.bfloat16, **overrides):
    kwargs = dict(
        num_query_heads=24,
        num_key_value_heads=2,
        head_dim=256,
        indexer_head_dim=128,
        compress_ratio=4,
        token_budget=2048,
        index_key_norm=lambda x: x,
        apply_index_rope=lambda x, p: x,
    )
    kwargs.update(overrides)
    queries = mx.zeros((1, 24, rows, 256), dtype=dtype)
    keys = mx.zeros((1, 2, tokens, 256), dtype=dtype)
    output = qsa_fast.contiguous_causal_gathered_qsa(
        queries,
        keys,
        keys,
        mx.zeros((1, rows, 4, 128), dtype=dtype),
        mx.zeros((1, tokens, 128), dtype=dtype),
        mx.arange(tokens)[None],
        **kwargs,
    )
    mx.eval(output)
    assert output.shape == (1, rows, 24, 256)


def test_warm_native_prefill_uses_wider_tiles_and_exact_tail(native_calls):
    _run()
    assert native_calls == [(1024, 2047), (1024, 3071), (1, 4095)]


@pytest.mark.parametrize("rows", [2, 6, 1023])
def test_decode_verify_and_short_prefill_keep_existing_tiles(native_calls, rows):
    _run(rows=rows)
    assert max(width for width, _ in native_calls) <= 256


@pytest.mark.parametrize("stage", ["MAIN", "SCORE", "TOPK"])
@pytest.mark.parametrize("state", ["unproven", "disabled"])
def test_unproven_or_disabled_kernel_keeps_existing_tiles(
    monkeypatch, native_calls, stage, state
):
    suffix, value = ("PROVEN", False) if state == "unproven" else ("DISABLED", True)
    monkeypatch.setattr(qsa_fast, f"_NATIVE_QSA_{stage}_{suffix}", value)
    _run()
    assert max(width for width, _ in native_calls) <= 256


@pytest.mark.parametrize("missing", ["extension", "symbol"])
def test_missing_native_abi_keeps_portable_tile_size(
    monkeypatch, native_calls, missing
):
    if missing == "extension":
        monkeypatch.setattr(fast, "is_native_available", lambda: False)
    else:
        monkeypatch.setattr(fast, "has_symbol", lambda name: False)
    _run()
    assert max(width for width, _ in native_calls) == 32


def test_explicit_query_chunk_is_respected(native_calls):
    _run(query_chunk=37)
    assert max(width for width, _ in native_calls) == 37
    assert sum(width for width, _ in native_calls) == 2049


@pytest.mark.parametrize(
    "kwargs", [{"dtype": mx.float32}, {"compress_ratio": 2}, {"token_budget": 1024}]
)
def test_other_dtypes_and_sparse_geometry_keep_existing_tiles(native_calls, kwargs):
    _run(**kwargs)
    assert max(width for width, _ in native_calls) <= 256


@pytest.mark.parametrize(
    "tokens,expected", [(65536, 1024), (262144, 512), (524288, 256), (524292, 128)]
)
def test_score_sheet_cap_shrinks_tiles_at_long_context(native_calls, tokens, expected):
    # Broadcast K/V are lazy and the fake attention never evaluates them.
    _run(rows=1024, tokens=tokens)
    assert max(width for width, _ in native_calls) == expected
    assert expected * (tokens // 4) * 4 <= 128 * 1024 * 1024


@pytest.mark.parametrize("stage", ["score", "topk", "main"])
def test_native_minimum_rows_override_keeps_existing_tiles(
    monkeypatch, native_calls, stage
):
    monkeypatch.setattr(qsa_fast, f"_native_{stage}_min_rows", lambda: 2048)
    _run()
    assert max(width for width, _ in native_calls) == 256


def test_native_rejection_retries_before_a_wide_portable_gather(
    monkeypatch, native_calls
):
    monkeypatch.setattr(qsa_fast, "_native_sparse_gqa_attention", lambda *a, **k: None)

    class ReachedPortableGatherError(Exception):
        pass

    widths = []

    def gather(kv, indices):
        widths.append(indices.shape[1])
        raise ReachedPortableGatherError

    monkeypatch.setattr(qsa_fast, "_gather_kv_rows", gather)
    with pytest.raises(ReachedPortableGatherError):
        _run()
    assert widths == [32]


_NATIVE = fast.is_native_available() and all(
    fast.has_symbol(name)
    for name in (
        "qwen4_qsa_indexer_scores",
        "qwen4_qsa_topk_indices",
        "qwen4_qsa_sparse_gqa_attention",
    )
)


@pytest.mark.skipif(not _NATIVE, reason="native Qwen4 QSA kernels not built")
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("tokens", [4096, 65536])
def test_native_wide_prefill_preserves_attention_bytes(monkeypatch, dtype, tokens):
    # Exercise actual score, top-k and attention kernels, including an uneven
    # final tile, against the existing explicit 256-query schedule.
    for stage in ("MAIN", "SCORE", "TOPK"):
        monkeypatch.setattr(qsa_fast, f"_NATIVE_QSA_{stage}_PROVEN", False)
        monkeypatch.setattr(qsa_fast, f"_NATIVE_QSA_{stage}_DISABLED", False)
    mx.random.seed(81)
    rows = 1025
    inputs = (
        (mx.random.normal((1, 24, rows, 256)) * 0.1).astype(dtype),
        (mx.random.normal((1, 2, tokens, 256)) * 0.1).astype(dtype),
        (mx.random.normal((1, 2, tokens, 256)) * 0.1).astype(dtype),
        (mx.random.normal((1, rows, 4, 128)) * 0.1).astype(dtype),
        (mx.random.normal((1, tokens, 128)) * 0.1).astype(dtype),
        mx.arange(tokens)[None],
    )
    mx.eval(inputs)
    kwargs = dict(
        num_query_heads=24,
        num_key_value_heads=2,
        head_dim=256,
        indexer_head_dim=128,
        compress_ratio=4,
        token_budget=2048,
        index_key_norm=lambda x: x,
        apply_index_rope=lambda x, p: x,
    )
    expected = qsa_fast.contiguous_causal_gathered_qsa(
        *inputs, query_chunk=256, **kwargs
    )
    mx.eval(expected)
    assert all(
        getattr(qsa_fast, f"_NATIVE_QSA_{stage}_PROVEN")
        for stage in ("MAIN", "SCORE", "TOPK")
    )
    widths = []
    original = qsa_fast._native_sparse_gqa_attention

    def record(q, *args, **kw):
        widths.append(q.shape[2])
        return original(q, *args, **kw)

    monkeypatch.setattr(qsa_fast, "_native_sparse_gqa_attention", record)
    actual = qsa_fast.contiguous_causal_gathered_qsa(*inputs, **kwargs)
    mx.eval(actual)
    assert widths == [1024, 1]
    assert mx.array_equal(expected.view(mx.uint8), actual.view(mx.uint8)).item()
