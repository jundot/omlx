# SPDX-License-Identifier: Apache-2.0
"""Batch-one Qwen4 QSA decode gather regression tests."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat


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

    monkeypatch.setattr(language, "contiguous_causal_gathered_qsa_decode", must_not_gather)
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
    assert language._gathered_min_query_tokens() == 16
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
    assert language._gathered_min_query_tokens() == 16


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
