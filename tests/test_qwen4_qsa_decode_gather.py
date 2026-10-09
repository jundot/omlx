# SPDX-License-Identifier: Apache-2.0
"""Batch-one Qwen4 QSA decode gather regression tests."""

from __future__ import annotations

import importlib
import math
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from omlx.memory_monitor import qwen4_gathered_min_query_tokens
from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

compat.apply_mlx_vlm_qwen4_exp_compat_patch()
language = importlib.import_module("mlx_vlm.models.qwen4_exp.language")
qsa_fast = importlib.import_module("mlx_vlm.models.qwen4_exp.qsa_fast")


@pytest.fixture(autouse=True)
def _vendored_qwen4(monkeypatch):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    monkeypatch.setattr(qsa_fast, "_DECODE_SELECT_FAILED", {})


def _tiny_text_config():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp import TextConfig

    return TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=3,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=16,
        moe_intermediate_size=16,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=2,
        max_position_embeddings=128,
        hc_count=2,
        hc_lowrank=8,
        head_dim=8,
        layer_types=["linear_attention", "qwen_sparse_attention"],
        ple_layer_ids=[],
        ple_embed_dim=32,
        ple_conv_kernel_size=3,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=2,
        eos_token_id=1,
        rope_parameters={
            "rope_type": "default",
            "mrope_section": [2, 1, 1],
            "rope_theta": 10_000,
            "partial_rotary_factor": 1.0,
        },
    )


def test_qwen4_decode_gathers_budget_and_tail_and_matches_official(monkeypatch):
    monkeypatch.setenv("OMLX_QWEN4_GATHERED_MIN_QUERY", "2")
    config = _tiny_text_config()
    import mlx_vlm.models.qwen4_exp.language as language
    import mlx_vlm.models.qwen4_exp.qsa_fast as qsa_fast

    attention = language.Qwen4ExpAttention(config)
    mx.eval(attention.parameters())
    fast_cache = language.QSAKVCache()
    reference_cache = language.QSAKVCache()

    mx.random.seed(19)
    prefix = mx.random.normal((1, 10, config.hidden_size))
    decode = mx.random.normal((1, 1, config.hidden_size))
    fast_prefix = attention(prefix, mask="causal", cache=fast_cache)
    reference_prefix = attention(prefix, mask="causal", cache=reference_cache)
    mx.eval(fast_prefix, reference_prefix)

    gathered_lengths = []
    original_sdpa = qsa_fast._decode_qsa_sdpa

    def tracked_sdpa(queries, keys, values, scale):
        gathered_lengths.append(int(keys.shape[2]))
        return original_sdpa(queries, keys, values, scale)

    monkeypatch.setattr(qsa_fast, "_decode_qsa_sdpa", tracked_sdpa)
    actual = attention(decode, cache=fast_cache)

    monkeypatch.setattr(
        language.Qwen4ExpAttention,
        "_gathered_text_decode_eligible",
        lambda *args, **kwargs: False,
    )
    expected = attention(decode, cache=reference_cache)
    mx.eval(actual, expected)

    # key_len=11, budget=8, incomplete causal tail=1.
    assert gathered_lengths == [9]
    assert mx.allclose(actual, expected, rtol=2e-5, atol=2e-5).item()
    assert mx.array_equal(
        mx.argmax(actual, axis=-1),
        mx.argmax(expected, axis=-1),
    ).item()
    assert fast_cache.offset == reference_cache.offset == 11
    assert fast_cache._omlx_last_prefill_gathered is True
    assert reference_cache._omlx_last_prefill_gathered is True
    for fast_value, reference_value in zip(
        fast_cache.state,
        reference_cache.state,
    ):
        assert mx.array_equal(fast_value, reference_value).item()


def test_qwen4_language_wrapper_routes_2d_text_positions_to_gather(monkeypatch):
    # The fixture prefill is 10 rows; lower the gathered width gate for it.
    monkeypatch.setenv("OMLX_QWEN4_GATHERED_MIN_QUERY", "2")
    config = _tiny_text_config()
    import mlx_vlm.models.qwen4_exp.language as language

    root_config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=58,
    )
    model = language.LanguageModel(config, root_config)
    mx.eval(model.parameters())
    fast_cache = model.make_cache()
    reference_cache = model.make_cache()
    calls = []

    original_prefill = language.Qwen4ExpAttention._gathered_text_prefill
    original_decode = language.Qwen4ExpAttention._gathered_text_decode
    original_prefill_eligible = (
        language.Qwen4ExpAttention._gathered_text_prefill_eligible
    )

    def tracked_prefill(self, x, cache, position_ids=None):
        calls.append(("prefill", position_ids.ndim, position_ids.shape))
        return original_prefill(self, x, cache, position_ids)

    def tracked_decode(self, x, cache, position_ids=None):
        calls.append(("decode", position_ids.ndim, position_ids.shape))
        return original_decode(self, x, cache, position_ids)

    monkeypatch.setattr(
        language.Qwen4ExpAttention,
        "_gathered_text_prefill",
        tracked_prefill,
    )
    monkeypatch.setattr(
        language.Qwen4ExpAttention,
        "_gathered_text_decode",
        tracked_decode,
    )

    prefix = mx.arange(2, 12, dtype=mx.int32)[None]
    fast_prefix = model(prefix, cache=fast_cache)

    # Replay the same wrapper-owned text sequence through the official path.
    model._position_ids = None
    model._rope_deltas = None
    monkeypatch.setattr(
        language.Qwen4ExpAttention,
        "_gathered_text_prefill_eligible",
        lambda *args, **kwargs: False,
    )
    reference_prefix = model(prefix, cache=reference_cache)
    monkeypatch.setattr(
        language.Qwen4ExpAttention,
        "_gathered_text_prefill_eligible",
        original_prefill_eligible,
    )

    decode_token = mx.array([[12]], dtype=mx.int32)
    position_ids = model._position_ids
    rope_deltas = model._rope_deltas
    actual = model(decode_token, cache=fast_cache)
    model._position_ids = position_ids
    model._rope_deltas = rope_deltas
    monkeypatch.setattr(
        language.Qwen4ExpAttention,
        "_gathered_text_decode_eligible",
        lambda *args, **kwargs: False,
    )
    expected = model(decode_token, cache=reference_cache)
    mx.eval(fast_prefix.logits, reference_prefix.logits, actual.logits, expected.logits)

    assert calls == [
        ("prefill", 2, (1, 10)),
        ("decode", 2, (1, 1)),
    ]
    assert mx.allclose(
        fast_prefix.logits,
        reference_prefix.logits,
        rtol=2e-4,
        atol=2e-4,
    ).item()
    assert mx.allclose(actual.logits, expected.logits, rtol=2e-4, atol=2e-4).item()
    assert mx.array_equal(
        mx.argmax(actual.logits[:, -1], axis=-1),
        mx.argmax(expected.logits[:, -1], axis=-1),
    ).item()


def test_qwen4_decode_keeps_official_path_until_complete_block_crossover(
    monkeypatch,
):
    config = _tiny_text_config()
    import mlx_vlm.models.qwen4_exp.language as language

    attention = language.Qwen4ExpAttention(config)
    cache = language.QSAKVCache()
    prefix = mx.random.normal((1, 8, config.hidden_size))
    attention(prefix, mask="causal", cache=cache)

    def must_not_gather(*args, **kwargs):
        raise AssertionError("decode at the QSA block budget must stay official")

    monkeypatch.setattr(
        language,
        "contiguous_causal_gathered_qsa_decode",
        must_not_gather,
    )
    output = attention(mx.random.normal((1, 1, config.hidden_size)), cache=cache)
    mx.eval(output)

    # Nine visible rows still contain only four complete two-token blocks.
    assert cache.offset == 9
    assert output.shape == (1, 1, config.hidden_size)


def test_qwen4_decode_gather_eligibility_fails_closed_for_general_paths():
    config = _tiny_text_config()
    import mlx_vlm.models.qwen4_exp.language as language

    attention = language.Qwen4ExpAttention(config)
    cache = language.QSAKVCache()
    prefix = mx.random.normal((1, 10, config.hidden_size))
    mx.eval(attention(prefix, mask="causal", cache=cache))
    token = mx.random.normal((1, 1, config.hidden_size))

    assert attention._gathered_text_decode_eligible(
        token, None, cache, None, None, False
    )
    assert not attention._gathered_text_decode_eligible(
        token, "left_padded_decode", cache, None, None, False
    )
    assert attention._gathered_text_decode_eligible(
        token, None, cache, mx.array([[10]], dtype=mx.int32), None, False
    )
    assert not attention._gathered_text_decode_eligible(
        token,
        None,
        cache,
        mx.array([[[10]], [[10]], [[10]]], dtype=mx.int32),
        None,
        False,
    )
    assert not attention._gathered_text_decode_eligible(
        token, None, cache, None, None, True
    )
    assert not attention._gathered_text_decode_eligible(
        mx.broadcast_to(token, (2, 1, config.hidden_size)),
        None,
        cache,
        None,
        None,
        False,
    )

    incomplete = language.QSAKVCache()
    incomplete.offset = cache.offset
    assert not attention._gathered_text_decode_eligible(
        token, None, incomplete, None, None, False
    )


@pytest.mark.parametrize("key_tokens", [4097, 32769])
def test_qwen4_decode_gather_stays_budget_bounded_at_long_cache(
    monkeypatch,
    key_tokens,
):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    import mlx_vlm.models.qwen4_exp.qsa_fast as qsa_fast

    mx.random.seed(23)
    queries = mx.random.normal((1, 4, 1, 8)).astype(mx.float32)
    keys = mx.random.normal((1, 2, key_tokens, 8)).astype(mx.float32)
    values = mx.random.normal((1, 2, key_tokens, 8)).astype(mx.float32)
    index_queries = mx.random.normal((1, 1, 2, 8)).astype(mx.float32)
    pooled = mx.random.normal((1, key_tokens // 2, 8)).astype(mx.float32)

    gathered_lengths = []
    original_sdpa = qsa_fast._decode_qsa_sdpa

    def tracked_sdpa(q, k, v, scale):
        gathered_lengths.append(int(k.shape[2]))
        return original_sdpa(q, k, v, scale)

    monkeypatch.setattr(qsa_fast, "_decode_qsa_sdpa", tracked_sdpa)
    output = qsa_fast.contiguous_causal_gathered_qsa_decode(
        queries,
        keys,
        values,
        index_queries,
        pooled,
        num_query_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        indexer_head_dim=8,
        compress_ratio=2,
        token_budget=8,
    )
    mx.eval(output)

    assert gathered_lengths == [9]
    assert output.shape == (1, 1, 4, 8)


def test_qwen4_gathered_prefill_is_exact_across_uneven_query_chunks():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp import qsa_fast

    mx.random.seed(29)
    key_tokens = 18
    query_tokens = 9
    queries = mx.random.normal((1, 4, query_tokens, 8)).astype(mx.bfloat16)
    keys = mx.random.normal((1, 2, key_tokens, 8)).astype(mx.bfloat16)
    values = mx.random.normal((1, 2, key_tokens, 8)).astype(mx.bfloat16)
    index_queries = mx.random.normal((1, query_tokens, 2, 8)).astype(mx.bfloat16)
    index_keys = mx.random.normal((1, key_tokens, 8)).astype(mx.bfloat16)
    index_positions = mx.arange(key_tokens, dtype=mx.int32)[None]
    pooled_index_keys = mx.random.normal((1, key_tokens // 2, 8)).astype(mx.bfloat16)

    def gathered(query_chunk):
        return qsa_fast.contiguous_causal_gathered_qsa(
            queries,
            keys,
            values,
            index_queries,
            index_keys,
            index_positions,
            num_query_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            indexer_head_dim=8,
            compress_ratio=2,
            token_budget=8,
            index_key_norm=lambda value: value,
            apply_index_rope=lambda value, _: value,
            pooled_index_keys=pooled_index_keys,
            query_chunk=query_chunk,
        )

    reference = gathered(2)
    actual = gathered(5)
    mx.eval(reference, actual)

    assert reference.shape == actual.shape == (1, query_tokens, 4, 8)
    assert mx.array_equal(actual, reference).item()


def test_qwen4_decode_sdpa_fails_closed_when_native_shape_is_rejected(monkeypatch):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    import mlx_vlm.models.qwen4_exp.qsa_fast as qsa_fast

    from omlx.custom_kernels.decode_fast import fast

    q = mx.random.normal((1, 4, 1, 8))
    k = mx.random.normal((1, 2, 9, 8))
    v = mx.random.normal((1, 2, 9, 8))
    monkeypatch.setattr(fast, "NATIVE_AVAILABLE", True)
    monkeypatch.setattr(
        fast,
        "_ext",
        SimpleNamespace(sdpa_decode_supported=lambda *args: False),
    )

    def must_not_run(*args, **kwargs):
        raise AssertionError("rejected decode_fast shape must use MLX SDPA")

    monkeypatch.setattr(fast, "sdpa_decode", must_not_run)
    actual = qsa_fast._decode_qsa_sdpa(q, k, v, 8**-0.5)
    expected = mx.fast.scaled_dot_product_attention(q, k, v, scale=8**-0.5)
    mx.eval(actual, expected)

    assert mx.array_equal(actual, expected).item()


def test_qwen4_decode_sdpa_uses_native_only_after_capability_accepts(monkeypatch):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    import mlx_vlm.models.qwen4_exp.qsa_fast as qsa_fast

    from omlx.custom_kernels.decode_fast import fast

    q = mx.random.normal((1, 24, 1, 256)).astype(mx.bfloat16)
    k = mx.random.normal((1, 2, 2051, 256)).astype(mx.bfloat16)
    v = mx.random.normal((1, 2, 2051, 256)).astype(mx.bfloat16)
    calls = []

    def supported(queries, keys, values):
        calls.append((queries.shape, keys.shape, values.shape, "probe"))
        return True

    def native(queries, keys, values, scale, causal=False):
        calls.append((scale, causal, "native"))
        return mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
        )

    monkeypatch.setattr(fast, "NATIVE_AVAILABLE", True)
    monkeypatch.setattr(
        fast,
        "_ext",
        SimpleNamespace(sdpa_decode_supported=supported),
    )
    monkeypatch.setattr(fast, "sdpa_decode", native)
    actual = qsa_fast._decode_qsa_sdpa(q, k, v, 256**-0.5)
    expected = mx.fast.scaled_dot_product_attention(q, k, v, scale=256**-0.5)
    mx.eval(actual, expected)

    assert calls == [
        (q.shape, k.shape, v.shape, "probe"),
        (256**-0.5, False, "native"),
    ]
    assert mx.array_equal(actual, expected).item()


def _crossover_cache(config, attention, length=12, seed=23):
    """Prefill ``length`` tokens so completed blocks exceed the QSA budget."""
    mx.random.seed(seed)
    cache = _language().QSAKVCache()
    prefix = mx.random.normal((1, length, config.hidden_size))
    mx.eval(attention(prefix, mask="causal", cache=cache))
    return cache


def _language():
    import mlx_vlm.models.qwen4_exp.language as language

    return language


# Stored-layout row gather and Lightning MTP verification share the same QSA
# gather path as the decode tests above.


def _reference(kv, indices):
    rows = kv.transpose(0, 2, 1, 3)
    batch, tokens = rows.shape[:2]
    trailing = rows.shape[2:]
    offsets = mx.arange(batch, dtype=mx.int32).reshape(
        (batch,) + (1,) * (indices.ndim - 1)
    ) * tokens
    flat = (indices.astype(mx.int32) + offsets).reshape(-1)
    gathered = rows.reshape(batch * tokens, *trailing)[flat].reshape(
        *indices.shape, *trailing
    )
    axes = (0, 2, 1, 3) if indices.ndim == 2 else (0, 1, 3, 2, 4)
    return mx.contiguous(gathered.transpose(*axes))


def _dispatch_row_gather(monkeypatch, per_query, tokens):
    from mlx_vlm.models.qwen4_exp import qsa_fast

    picked = []
    for name in ("_gather_kv_rows_stored", "_gather_kv_rows_token_major"):
        monkeypatch.setattr(
            qsa_fast,
            name,
            lambda kv, idx, _name=name: picked.append(_name),
        )
    kv = mx.zeros((1, 2, tokens, 16), dtype=mx.bfloat16)
    indices = mx.zeros((1, per_query, 2051), dtype=mx.int32)
    qsa_fast._gather_kv_rows(kv, indices)
    return picked


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize(
    "form",
    ["_gather_kv_rows", "_gather_kv_rows_stored", "_gather_kv_rows_token_major"],
)
def test_gather_kv_rows_matches_token_major_gather(batch, form):
    from mlx_vlm.models.qwen4_exp import qsa_fast

    mx.random.seed(3)
    kv = mx.random.normal((batch, 2, 300, 16)).astype(mx.bfloat16)
    indices = mx.sort(
        mx.random.randint(0, 300, (batch, 37)).astype(mx.int32), axis=-1
    )
    out = getattr(qsa_fast, form)(kv, indices)
    mx.eval(out)
    assert out.shape == (batch, 2, 37, 16)
    assert mx.array_equal(
        out.view(mx.uint16), _reference(kv, indices).view(mx.uint16)
    ).item()


@pytest.mark.parametrize("per_query", [1, 4, 16])
@pytest.mark.parametrize("tokens", [4096, 65536, 206848])
def test_gather_kv_rows_decode_and_verify_widths_use_stored_layout(
    monkeypatch, per_query, tokens
):
    assert _dispatch_row_gather(monkeypatch, per_query, tokens) == [
        "_gather_kv_rows_stored"
    ]


@pytest.mark.parametrize("tokens", [4096, 16384, 65536])
def test_gather_kv_rows_prefill_width_copies_token_major_below_threshold(
    monkeypatch, tokens
):
    assert _dispatch_row_gather(monkeypatch, 64, tokens) == [
        "_gather_kv_rows_token_major"
    ]


def test_gather_kv_rows_prefill_width_uses_stored_layout_at_long_context(monkeypatch):
    assert _dispatch_row_gather(monkeypatch, 64, 131072) == [
        "_gather_kv_rows_stored"
    ]


def test_gather_kv_rows_rank_three_forms_agree():
    from mlx_vlm.models.qwen4_exp import qsa_fast

    mx.random.seed(5)
    kv = mx.random.normal((2, 2, 500, 16)).astype(mx.bfloat16)
    indices = mx.sort(
        mx.random.randint(0, 500, (2, 8, 21)).astype(mx.int32), axis=-1
    )
    stored = qsa_fast._gather_kv_rows_stored(kv, indices)
    token_major = qsa_fast._gather_kv_rows_token_major(kv, indices)
    mx.eval(stored, token_major)
    assert stored.shape == token_major.shape == (2, 8, 2, 21, 16)
    assert mx.array_equal(
        stored.view(mx.uint16), token_major.view(mx.uint16)
    ).item()
    assert mx.array_equal(
        stored.view(mx.uint16), _reference(kv, indices).view(mx.uint16)
    ).item()


def _layer_and_prefix(seed: int = 19, prefix_tokens: int = 10):
    import mlx_vlm.models.qwen4_exp.language as language

    config = _tiny_text_config()
    attention = language.Qwen4ExpAttention(config)
    mx.eval(attention.parameters())
    fast_cache = language.QSAKVCache()
    reference_cache = language.QSAKVCache()
    mx.random.seed(seed)
    prefix = mx.random.normal((1, prefix_tokens, config.hidden_size))
    mx.eval(
        attention(prefix, mask="causal", cache=fast_cache),
        attention(prefix, mask="causal", cache=reference_cache),
    )
    return config, attention, fast_cache, reference_cache


@pytest.mark.parametrize("rows", [2, 4])
def test_qwen4_verify_rows_gather_selected_blocks_and_match_official(
    monkeypatch, rows
):
    import mlx_vlm.models.qwen4_exp.language as language

    config, attention, fast_cache, reference_cache = _layer_and_prefix()
    verify = mx.random.normal((1, rows, config.hidden_size))

    gathered_query_tokens = []
    original = language.contiguous_causal_gathered_qsa

    def tracked(queries, *args, **kwargs):
        gathered_query_tokens.append(int(queries.shape[2]))
        return original(queries, *args, **kwargs)

    monkeypatch.setattr(language, "contiguous_causal_gathered_qsa", tracked)
    actual = attention(verify, mask="causal", cache=fast_cache, target_verify=True)

    monkeypatch.setattr(
        language.Qwen4ExpAttention,
        "_gathered_text_verify_eligible",
        lambda *a, **k: False,
        raising=False,
    )
    expected = attention(
        verify, mask="causal", cache=reference_cache, target_verify=True
    )
    mx.eval(actual, expected)

    assert gathered_query_tokens == [rows]
    assert mx.allclose(actual, expected, rtol=2e-5, atol=2e-5).item()
    assert mx.array_equal(
        mx.argmax(actual, axis=-1), mx.argmax(expected, axis=-1)
    ).item()
    assert fast_cache.offset == reference_cache.offset == 10 + rows
    for fast_value, reference_value in zip(fast_cache.state, reference_cache.state):
        assert mx.array_equal(fast_value, reference_value).item()


def test_qwen4_verify_rows_survive_rollback_like_official(monkeypatch):
    import mlx_vlm.models.qwen4_exp.language as language

    config, attention, fast_cache, reference_cache = _layer_and_prefix(seed=23)
    first = mx.random.normal((1, 4, config.hidden_size))
    mx.eval(attention(first, mask="causal", cache=fast_cache, target_verify=True))
    monkeypatch.setattr(
        language.Qwen4ExpAttention,
        "_gathered_text_verify_eligible",
        lambda *a, **k: False,
        raising=False,
    )
    mx.eval(
        attention(first, mask="causal", cache=reference_cache, target_verify=True)
    )
    monkeypatch.undo()
    for cache in (fast_cache, reference_cache):
        cache.trim(3)
    assert fast_cache.offset == reference_cache.offset == 11

    second = mx.random.normal((1, 4, config.hidden_size))
    actual = attention(second, mask="causal", cache=fast_cache, target_verify=True)
    monkeypatch.setattr(
        language.Qwen4ExpAttention,
        "_gathered_text_verify_eligible",
        lambda *a, **k: False,
        raising=False,
    )
    expected = attention(
        second, mask="causal", cache=reference_cache, target_verify=True
    )
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, rtol=2e-5, atol=2e-5).item()
    for fast_value, reference_value in zip(fast_cache.state, reference_cache.state):
        assert mx.array_equal(fast_value, reference_value).item()


def test_qwen4_verify_gather_kill_switch_keeps_official_path(monkeypatch):
    import mlx_vlm.models.qwen4_exp.language as language

    config, attention, fast_cache, _ = _layer_and_prefix()
    calls = []
    original = language.contiguous_causal_gathered_qsa
    monkeypatch.setattr(
        language,
        "contiguous_causal_gathered_qsa",
        lambda *a, **k: calls.append(1) or original(*a, **k),
    )
    monkeypatch.setattr(language, "_GATHERED_VERIFY_DISABLED", True)
    mx.eval(
        attention(
            mx.random.normal((1, 4, config.hidden_size)),
            mask="causal",
            cache=fast_cache,
            target_verify=True,
        )
    )
    assert calls == []


def test_qwen4_verify_gather_requires_rank_two_positions():
    config, attention, fast_cache, _ = _layer_and_prefix()
    rows = 4
    verify = mx.random.normal((1, rows, config.hidden_size))
    text = mx.arange(fast_cache.offset, fast_cache.offset + rows)[None, :]
    planes = mx.broadcast_to(text[None, :, :], (3, 1, rows))

    def eligible(positions):
        return attention._gathered_text_verify_eligible(
            verify, "causal", fast_cache, positions, None, True
        )

    assert fast_cache.offset + rows > attention.indexer.token_budget
    assert eligible(text) is True
    assert eligible(None) is True
    assert eligible(planes) is False


def test_qwen4_gathered_prefill_requires_minimum_query_width(monkeypatch):
    """Narrow multi-row windows (MTP passes) stay on the cheaper official path."""
    config = _tiny_text_config()
    language = _language()
    attention = language.Qwen4ExpAttention(config)
    mx.eval(attention.parameters())
    cache = _crossover_cache(config, attention)
    narrow = mx.random.normal((1, 4, config.hidden_size))
    wide = mx.random.normal((1, 16, config.hidden_size))

    monkeypatch.delenv("OMLX_QWEN4_GATHERED_MIN_QUERY", raising=False)
    assert qwen4_gathered_min_query_tokens() == 16
    assert not attention._gathered_text_prefill_eligible(
        narrow, "causal", cache, None, None, False
    )
    assert attention._gathered_text_prefill_eligible(
        wide, "causal", cache, None, None, False
    )

    monkeypatch.setenv("OMLX_QWEN4_GATHERED_MIN_QUERY", "2")
    assert attention._gathered_text_prefill_eligible(
        narrow, "causal", cache, None, None, False
    )
    monkeypatch.setenv("OMLX_QWEN4_GATHERED_MIN_QUERY", "garbage")
    assert qwen4_gathered_min_query_tokens() == 16


def test_qwen4_trim_keeps_pooled_index_prefix_exact():
    """trim() clamps the pooled frontier instead of re-pooling every block."""
    config = _tiny_text_config()
    language = _language()
    attention = language.Qwen4ExpAttention(config)
    mx.eval(attention.parameters())
    ratio = config.indexer_compress_ratio
    indexer = attention.indexer

    cache = _crossover_cache(config, attention, length=14, seed=51)
    pooled_before = cache.pooled_indexer_keys(
        ratio, indexer.k_layernorm, indexer._apply_rope, cache_tag=indexer
    )
    mx.eval(pooled_before)
    assert cache._pooled_index_offset == 14 // ratio

    # Speculative window of 3 rows, then reject two of them (MTP rollback).
    window = mx.random.normal((1, 3, config.hidden_size))
    mx.eval(attention(window, mask="causal", cache=cache, target_verify=True))
    assert cache.trim(2) == 2
    assert cache.offset == 15
    # Pooled blocks below the new complete count survive the trim.
    assert cache._pooled_index_keys is not None
    assert cache._pooled_index_offset == 15 // ratio

    incremental = cache.pooled_indexer_keys(
        ratio, indexer.k_layernorm, indexer._apply_rope, cache_tag=indexer
    )
    cache._invalidate_pooled_indexer()
    full = cache.pooled_indexer_keys(
        ratio, indexer.k_layernorm, indexer._apply_rope, cache_tag=indexer
    )
    mx.eval(incremental, full)
    assert incremental.shape == full.shape == (1, 15 // ratio, config.indexer_head_dim)
    assert mx.array_equal(incremental, full).item()

    # A trim that crosses a completed block boundary drops that block too.
    assert cache.trim(3) == 3
    assert cache._pooled_index_offset == 12 // ratio
    again = cache.pooled_indexer_keys(
        ratio, indexer.k_layernorm, indexer._apply_rope, cache_tag=indexer
    )
    cache._invalidate_pooled_indexer()
    again_full = cache.pooled_indexer_keys(
        ratio, indexer.k_layernorm, indexer._apply_rope, cache_tag=indexer
    )
    mx.eval(again, again_full)
    assert mx.array_equal(again, again_full).item()


def test_qwen4_trim_then_decode_matches_official_after_partial_invalidation(
    monkeypatch,
):
    """Verify -> rollback -> decode stays exact with the retained pooled prefix."""
    config = _tiny_text_config()
    language = _language()
    monkeypatch.setenv("OMLX_QWEN4_GATHERED_MIN_QUERY", "2")
    attention = language.Qwen4ExpAttention(config)
    mx.eval(attention.parameters())
    fast_cache = _crossover_cache(config, attention, length=12, seed=61)
    reference_cache = _crossover_cache(config, attention, length=12, seed=61)

    window = mx.random.normal((1, 4, config.hidden_size))
    mx.eval(
        attention(window, mask="causal", cache=fast_cache, target_verify=True),
        attention(window, mask="causal", cache=reference_cache, target_verify=True),
    )
    assert fast_cache.trim(2) == reference_cache.trim(2) == 2
    # Reference: force a full re-pool as the previous behaviour did.
    reference_cache._invalidate_pooled_indexer()

    token = mx.random.normal((1, 1, config.hidden_size))
    actual = attention(token, mask=None, cache=fast_cache)
    expected = attention(token, mask=None, cache=reference_cache)
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected).item()


def test_qwen4_batched_decode_does_not_read_gpu_scalars(monkeypatch):
    config = _tiny_text_config()
    language = _language()
    attention = language.Qwen4ExpAttention(config)
    cache = language.BatchQSAKVCache([0, 0])
    hidden = mx.random.normal((2, 1, config.hidden_size))
    mx.eval(attention.parameters(), hidden)

    def unexpected_scalar_read(*args, **kwargs):
        raise AssertionError("Batched attention must not read GPU scalars")

    with monkeypatch.context() as patch:
        patch.setattr(mx.array, "item", unexpected_scalar_read)
        for _ in range(3):
            positions = mx.broadcast_to(cache.offset[None, :, None], (3, 2, 1))
            mx.eval(attention(hidden, cache=cache, position_ids=positions))

    assert cache.offset.tolist() == [3, 3]


def test_qwen4_prefill_memory_marker_tracks_query_width():
    config = _tiny_text_config()
    language = _language()
    attention = language.Qwen4ExpAttention(config)
    cache = _crossover_cache(config, attention)

    mx.eval(attention(mx.zeros((1, 4, config.hidden_size)), cache=cache))
    assert cache._omlx_last_prefill_gathered is False
    mx.eval(attention(mx.zeros((1, 16, config.hidden_size)), cache=cache))
    assert cache._omlx_last_prefill_gathered is True


# One-launch decode selection and selected-keys SDPA against MLX. The selection
# kernel must give the same mask as the argpartition path: same FP32 scores, the
# highest block indices on ties, NaN above +inf. The masked SDPA visits only the
# selected keys but keeps MLX's two-pass partitions, order and arithmetic.


TOPK = 512
RATIO = 4
HEAD_DIM = 128


def _official_mask(head_scores: mx.array, key_len: int) -> mx.array:
    """``Qwen4ExpQSAIndexer.from_projected`` after the matmul, seq_len 1, aligned."""
    batch, seq_len = 1, 1
    past_len = key_len - 1
    max_complete_blocks = key_len // RATIO
    complete_key_len = max_complete_blocks * RATIO
    scores = mx.sum(mx.maximum(head_scores, 0), axis=1)
    scores = scores / math.sqrt(HEAD_DIM)
    query_ends = past_len + mx.arange(seq_len) + 1
    complete_counts = query_ends // RATIO
    valid_blocks = (
        mx.arange(max_complete_blocks)[None, None, :] < complete_counts[None, :, None]
    )
    scores = mx.where(valid_blocks, scores, -mx.inf)
    selected_blocks = mx.argpartition(scores, kth=-TOPK, axis=-1)[..., -TOPK:]
    block_hits = mx.put_along_axis(
        mx.zeros((batch, seq_len, max_complete_blocks), dtype=mx.bool_),
        selected_blocks,
        mx.array(True),
        axis=-1,
    )
    selected_tokens = mx.repeat(block_hits, RATIO, axis=-1)
    if complete_key_len < key_len:
        selected_tokens = mx.concatenate(
            [
                selected_tokens,
                mx.zeros((batch, seq_len, key_len - complete_key_len), dtype=mx.bool_),
            ],
            axis=-1,
        )
    token_indices = mx.arange(key_len)
    tail_starts = complete_counts * RATIO
    tail = (token_indices[None, None, :] >= tail_starts[None, :, None]) & (
        token_indices[None, None, :] < query_ends[None, :, None]
    )
    causal = token_indices[None, None, :] < query_ends[None, :, None]
    use_sparse = complete_counts > TOPK
    selected_tokens = mx.where(use_sparse[None, :, None], selected_tokens | tail, causal)
    return selected_tokens[:, None]


def _head_scores(blocks: int, kind: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "normal":
        return rng.standard_normal((4, blocks)).astype(np.float32)
    if kind == "magnitudes":
        # Sums whose value depends on the order the four heads are added.
        return (
            rng.standard_normal((4, blocks)) * np.exp2(rng.integers(-30, 30, (4, blocks)))
        ).astype(np.float32)
    if kind == "cutoff_ties":
        # 160 strict winners, then a tie that straddles the top-512 cutoff.
        scores = np.full((4, blocks), -1.0, dtype=np.float32)
        order = rng.permutation(blocks)
        tied = min(1400, blocks - 160)
        scores[:, order[:tied]] = 0.5
        scores[:, order[tied : tied + 160]] = 2.0
        return scores
    if kind == "specials":
        scores = rng.standard_normal((4, blocks)).astype(np.float32)
        for value, count in ((np.nan, 150), (np.inf, 150), (-np.inf, 150), (-0.0, 600)):
            scores[rng.integers(0, 4, count), rng.integers(0, blocks, count)] = value
        return scores
    raise ValueError(kind)


def _require_selection(result):
    """The selection kernel's result; skip where the GPU caps the pipeline
    below its 1024-thread threadgroup (larger PER on some GPUs)."""
    if result is None and any(
        "threads per threadgroup" in error for error in qsa_fast._DECODE_SELECT_FAILED.values()
    ):
        pytest.skip("decode selection threadgroup exceeds this GPU's pipeline limit")
    assert result is not None
    return result


@pytest.mark.parametrize(
    "key_len",
    [
        2052,  # 513 blocks: one above the budget
        2055,  # three-token tail
        24003,  # 24K decode
        32768,  # 8192 blocks: largest bank held 8 per thread
        32773,  # 8193 blocks, one-token tail: 16 per thread
    ],
)
@pytest.mark.parametrize("kind", ["normal", "magnitudes", "cutoff_ties", "specials"])
def test_decode_mask_matches_the_official_indexer_ops(key_len, kind):
    blocks = key_len // RATIO
    head_scores = mx.array(_head_scores(blocks, kind, key_len).reshape(1, 4, 1, blocks))

    actual = qsa_fast.decode_block_selection_mask(
        head_scores,
        head_dim=HEAD_DIM,
        key_tokens=key_len,
        compress_ratio=RATIO,
        block_topk=TOPK,
    )
    expected = _official_mask(head_scores, key_len)

    _require_selection(actual)
    assert actual.shape == expected.shape == (1, 1, 1, key_len)
    assert actual.dtype == mx.bool_
    assert mx.array_equal(actual, expected).item()


def test_cutoff_tie_keeps_the_highest_block_indices():
    blocks, key_len = 2048, 8192
    scores = np.full((4, 1, blocks), -1.0, dtype=np.float32)
    strict = np.arange(0, 100)
    tied = np.arange(300, 1500)
    scores[0, 0, strict] = 3.0
    scores[0, 0, tied] = 1.0
    mask = qsa_fast.decode_block_selection_mask(
        mx.array(scores[None]),
        head_dim=HEAD_DIM,
        key_tokens=key_len,
        compress_ratio=RATIO,
        block_topk=TOPK,
    )
    selected = np.flatnonzero(np.asarray(_require_selection(mask)).reshape(-1)[::RATIO])
    expected = np.concatenate((strict, tied[-(TOPK - strict.size) :]))
    np.testing.assert_array_equal(selected, expected)


def test_pipeline_limit_keeps_the_kernel_for_smaller_banks(monkeypatch):
    """A PER the GPU cannot launch at 1024 threads falls back alone; smaller
    block banks keep the kernel."""

    def select(key_len):
        blocks = key_len // RATIO
        scores = mx.array(_head_scores(blocks, "normal", key_len).reshape(1, 4, 1, blocks))
        kwargs = dict(head_dim=HEAD_DIM, key_tokens=key_len, compress_ratio=RATIO, block_topk=TOPK)
        return qsa_fast.decode_block_selection_mask(scores, **kwargs), scores

    _require_selection(select(2052)[0])
    kernel = qsa_fast._DECODE_SELECT_KERNELS["mask"]
    launches = []

    def limited(*args, template, **kwargs):
        launches.append(dict(template)["PER"])
        if dict(template)["PER"] > 8:
            raise ValueError("maximum allowed threads per threadgroup (832)")
        return kernel(*args, template=template, **kwargs)

    monkeypatch.setitem(qsa_fast._DECODE_SELECT_KERNELS, "mask", limited)
    assert select(32773)[0] is None
    assert select(32773)[0] is None
    actual, scores = select(2055)
    assert actual is not None
    assert mx.array_equal(actual, _official_mask(scores, 2055)).item()
    assert launches == [16, 8]


def test_block_scores_round_like_maximum_sum_divide():
    """The kernel's score is bit-identical to MLX's three ops, NaN payloads included."""
    kernel = mx.fast.metal_kernel(
        name="test_qwen4_qsa_decode_block_score",
        input_names=["head_scores", "divisor"],
        output_names=["out"],
        header=qsa_fast._DECODE_SELECT_HEADER,
        source="""
            const uint n = uint(head_scores_shape[head_scores_ndim - 1]);
            const uint e = thread_position_in_grid.x;
            if (e < n) {
                out[e] = qsa_decode_block_score(head_scores, n, e, H, divisor[0]);
            }
        """,
    )
    blocks = 6000
    for kind in ("magnitudes", "specials"):
        head_scores = mx.array(_head_scores(blocks, kind, 7).reshape(1, 4, 1, blocks))
        expected = mx.sum(mx.maximum(head_scores, 0), axis=1) / math.sqrt(HEAD_DIM)
        actual = kernel(
            inputs=[head_scores, mx.array([math.sqrt(HEAD_DIM)], dtype=mx.float32)],
            template=[("H", 4)],
            grid=(blocks, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(1, 1, blocks)],
            output_dtypes=[mx.float32],
        )[0]
        assert mx.array_equal(actual.view(mx.uint32), expected.view(mx.uint32)).item()


def _attention_and_caches(key_len: int):
    from mlx_vlm.models.qwen4_exp import TextConfig

    config = TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=256,
        num_hidden_layers=1,
        num_attention_heads=24,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=3,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=16,
        moe_intermediate_size=16,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=2,
        max_position_embeddings=65536,
        head_dim=256,
        layer_types=["full_attention"],
        ple_layer_ids=[],
        ple_embed_dim=32,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=HEAD_DIM,
        indexer_budget=TOPK * RATIO,
        indexer_compress_ratio=RATIO,
        eos_token_id=1,
        rope_parameters={
            "rope_type": "default",
            "mrope_section": [3, 3, 2],
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
        },
    )
    mx.random.seed(3)
    attention = language.Qwen4ExpAttention(config)
    attention.set_dtype(mx.bfloat16)
    mx.eval(attention.parameters())
    keys = mx.random.normal((1, 2, key_len, 256)).astype(mx.bfloat16)
    values = mx.random.normal((1, 2, key_len, 256)).astype(mx.bfloat16)
    index_keys = mx.random.normal((1, key_len, HEAD_DIM)).astype(mx.bfloat16)
    positions = mx.arange(key_len, dtype=mx.int32)[None]
    caches = []
    for _ in range(2):
        cache = language.QSAKVCache()
        cache.update_and_fetch(keys, values)
        cache.update_indexer(index_keys, positions)
        caches.append(cache)
    mx.eval([c.state for c in caches])
    return config, attention, caches


def _set_kernels(monkeypatch, enabled: bool):
    monkeypatch.setattr(qsa_fast, "_DECODE_SELECT_DISABLED", not enabled)
    monkeypatch.setattr(qsa_fast, "_DECODE_SDPA_DISABLED", not enabled)


@pytest.mark.parametrize("key_len", [3001, 16390])
def test_decode_steps_match_the_ops_path_bit_for_bit(monkeypatch, key_len):
    """Rank-three (served below the gathered crossover) decode through the
    module: outputs and every cache array equal, across block completions."""
    config, attention, (fast_cache, ops_cache) = _attention_and_caches(key_len)
    ran = {"mask": 0, "sdpa": 0}

    def recording(name, function):
        def wrapper(*args, **kwargs):
            result = function(*args, **kwargs)
            ran[name] += result is not None
            return result

        return wrapper

    monkeypatch.setattr(
        language,
        "decode_block_selection_mask",
        recording("mask", qsa_fast.decode_block_selection_mask),
    )
    monkeypatch.setattr(
        language, "masked_decode_sdpa", recording("sdpa", qsa_fast.masked_decode_sdpa)
    )
    for step in range(6):
        x = mx.random.normal((1, 1, config.hidden_size)).astype(mx.bfloat16)
        offset = fast_cache.offset
        positions = mx.broadcast_to(mx.array([[offset]], dtype=mx.int32)[None], (3, 1, 1))
        _set_kernels(monkeypatch, True)
        fast = attention(x, mask=None, cache=fast_cache, position_ids=positions)
        _set_kernels(monkeypatch, False)
        ops = attention(x, mask=None, cache=ops_cache, position_ids=positions)
        mx.eval(fast, ops)
        assert mx.array_equal(fast.view(mx.uint16), ops.view(mx.uint16)).item(), step
    assert ran["mask"] == 6
    if qsa_fast._gpu_class() == "d":
        assert ran["sdpa"] == 6
    for fast_state, ops_state in zip(fast_cache.state, ops_cache.state):
        assert mx.array_equal(fast_state, ops_state).item()
    pooled = fast_cache._pooled_index_offset
    assert pooled == ops_cache._pooled_index_offset == (key_len + 6) // RATIO
    assert mx.array_equal(
        fast_cache._pooled_index_keys[:, :pooled], ops_cache._pooled_index_keys[:, :pooled]
    ).item()


@pytest.mark.skipif(
    qsa_fast._gpu_class() != "d",
    reason="MLX's two-pass partition count is transcribed for 'd'-class GPUs only",
)
@pytest.mark.parametrize(
    "key_len",
    [
        2052,
        16383,  # last 128-partition length
        16384,  # first 512-partition length
        65536,  # first 1024-partition length
    ],
)
@pytest.mark.parametrize("kind", ["qsa", "sparse_head", "all"])
def test_masked_decode_sdpa_matches_mlx_bit_for_bit(key_len, kind):
    rng = np.random.default_rng(key_len)
    capacity = key_len + 777
    # Cache-shaped views: a prefix of a larger buffer, as update_and_fetch returns.
    keys = mx.random.normal((1, 2, capacity, 256), key=mx.random.key(1)).astype(mx.bfloat16)
    values = mx.random.normal((1, 2, capacity, 256), key=mx.random.key(2)).astype(mx.bfloat16)
    keys, values = keys[:, :, :key_len], values[:, :, :key_len]
    queries = (4 * mx.random.normal((1, 24, 1, 256), key=mx.random.key(3))).astype(mx.bfloat16)
    selected = np.zeros(key_len, dtype=bool)
    if kind == "qsa":
        blocks = key_len // RATIO
        for block in rng.choice(blocks, TOPK, replace=False):
            selected[block * RATIO : (block + 1) * RATIO] = True
        selected[blocks * RATIO :] = True
    elif kind == "sparse_head":
        selected[:37] = True  # most partitions see no key at all
    else:
        selected[:] = True
    mask = mx.array(selected).reshape(1, 1, 1, key_len)

    actual = qsa_fast.masked_decode_sdpa(queries, keys, values, mask, 256**-0.5)
    expected = mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=256**-0.5, mask=mask
    )

    assert actual is not None
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert mx.array_equal(actual.view(mx.uint16), expected.view(mx.uint16)).item()


def _official_gathered_tokens(head_scores: mx.array, key_len: int) -> mx.array:
    """``contiguous_causal_gathered_qsa_decode``'s argpartition path, verbatim."""
    blocks = key_len // RATIO
    scores = mx.sum(mx.maximum(head_scores, 0), axis=-2) / math.sqrt(HEAD_DIM)
    selected = mx.argpartition(scores, kth=-TOPK, axis=-1)[..., -TOPK:].astype(mx.int32)
    selected = mx.sort(selected, axis=-1)
    tokens = (selected[..., None] * RATIO + mx.arange(RATIO, dtype=mx.int32)).reshape(
        1, TOPK * RATIO
    )
    if blocks * RATIO < key_len:
        tail = mx.arange(blocks * RATIO, key_len, dtype=mx.int32)[None]
        tokens = mx.concatenate((tokens, tail), axis=-1)
    return tokens


@pytest.mark.parametrize("key_len", [2052, 32770, 65539])
@pytest.mark.parametrize("kind", ["normal", "cutoff_ties", "specials"])
def test_gathered_decode_tokens_match_the_argpartition_path(key_len, kind):
    blocks = key_len // RATIO
    head_scores = mx.array(_head_scores(blocks, kind, key_len).reshape(1, 1, 4, blocks))

    actual = qsa_fast.decode_block_selection_tokens(
        head_scores,
        head_dim=HEAD_DIM,
        key_tokens=key_len,
        compress_ratio=RATIO,
        block_topk=TOPK,
    )
    expected = _official_gathered_tokens(head_scores, key_len)

    _require_selection(actual)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert mx.array_equal(actual, expected).item()
