# SPDX-License-Identifier: Apache-2.0
"""Fix 2: TurboQuant KV quantization for Qwen4-Exp QSA caches.

Covers the new QSATurboQuantKVCache / BatchQSATurboQuantKVCache classes, the
scheduler eligibility + conversion wiring, the dense-store serialization
handler, and -- most importantly -- the MTP multi-row verify forward, which the
naive ``to_quantized`` approach crashed on (non-subscriptable
``_QuantizedStateProxy`` in the per-row slice loop).
"""
# Class objects are bound to their CamelCase names via the _classes() factory.
# ruff: noqa: N806

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat


def _classes():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import (
        BatchQSATurboQuantKVCache,
        QSAKVCache,
        QSATurboQuantKVCache,
    )

    import omlx.scheduler  # noqa: F401 - installs merge monkeypatch + patches

    return QSAKVCache, QSATurboQuantKVCache, BatchQSATurboQuantKVCache


def _populated_qsa(tokens: int, *, seed: int = 0, heads: int = 2, dim: int = 256):
    QSAKVCache, _, _ = _classes()
    mx.random.seed(seed)
    cache = QSAKVCache()
    cache.update_and_fetch(
        mx.random.normal((1, heads, tokens, dim)).astype(mx.bfloat16),
        mx.random.normal((1, heads, tokens, dim)).astype(mx.bfloat16),
    )
    cache.update_indexer(
        mx.random.normal((1, tokens, 128)).astype(mx.bfloat16),
        mx.arange(tokens, dtype=mx.int32).reshape(1, tokens),
    )
    return cache


# --------------------------------------------------------------------------- #
# Cache-class behavior
# --------------------------------------------------------------------------- #


def test_from_cache_keeps_indexer_full_precision():
    _, QSATQ, _ = _classes()
    src = _populated_qsa(40, seed=1)
    tq = QSATQ.from_cache(src, bits=4.0)

    # Indexer keys/positions must be byte-identical (never quantized).
    assert mx.array_equal(tq.index_keys, src.index_keys).item()
    assert mx.array_equal(tq.index_position_ids, src.index_position_ids).item()
    assert tq.offset == src.offset

    # K/V round-trips within 4-bit MSE tolerance and is much smaller.
    dk, _ = tq.dequantize()
    ok = src.state[0]
    k_rel = (
        mx.mean(mx.abs(dk.astype(mx.float32) - ok.astype(mx.float32)))
        / mx.mean(mx.abs(ok.astype(mx.float32)))
    ).item()
    assert k_rel < 0.20  # random-normal worst case; real KV quantizes better
    assert tq.nbytes < src.nbytes


def test_merge_produces_qsa_batch_not_base():
    """merge() must return the QSA batch (carrying index), not the base TQ
    batch installed by the scheduler monkeypatch (which drops update_indexer)."""
    _, QSATQ, BatchQSATQ = _classes()
    from omlx.turboquant_kv import BatchTurboQuantKVCache

    tq = QSATQ.from_cache(_populated_qsa(30, seed=2), bits=4.0)
    merged = QSATQ.merge([tq])
    assert isinstance(merged, BatchQSATQ)
    assert type(merged) is not BatchTurboQuantKVCache
    assert merged.index_keys is not None
    assert hasattr(merged, "update_indexer")


def test_merge_carries_and_aligns_indexer():
    _, QSATQ, _ = _classes()
    a = QSATQ.from_cache(_populated_qsa(40, seed=3), bits=4.0)
    b = QSATQ.from_cache(_populated_qsa(24, seed=4), bits=4.0)
    c = QSATQ.from_cache(_populated_qsa(31, seed=5), bits=4.0)
    ik_a, ik_b, ik_c = a.index_keys, b.index_keys, c.index_keys

    batch = QSATQ.merge([a, b, c])
    target = max(40, 24, 31)
    assert batch.index_offset == target
    assert batch.index_keys.shape == (3, target, 128)

    # Each row is left-padded to `target`; the tail must equal the source keys.
    assert mx.array_equal(batch.index_keys[0:1, target - 40 :], ik_a).item()
    assert mx.array_equal(batch.index_keys[1:2, target - 24 :], ik_b).item()
    assert mx.array_equal(batch.index_keys[2:3, target - 31 :], ik_c).item()


def test_batch_extract_row_exact():
    _, QSATQ, _ = _classes()
    a = QSATQ.from_cache(_populated_qsa(40, seed=6), bits=4.0)
    b = QSATQ.from_cache(_populated_qsa(24, seed=7), bits=4.0)
    ik_a = a.index_keys
    batch = QSATQ.merge([a, b])

    extracted = batch.extract(0)
    assert type(extracted).__name__ == "QSATurboQuantKVCache"
    assert extracted.index_keys.shape[1] == 40
    assert mx.array_equal(extracted.index_keys, ik_a).item()


def test_batch_filter_drops_row_and_index():
    _, QSATQ, _ = _classes()
    a = QSATQ.from_cache(_populated_qsa(40, seed=8), bits=4.0)
    b = QSATQ.from_cache(_populated_qsa(40, seed=9), bits=4.0)
    ik_b = b.index_keys
    batch = QSATQ.merge([a, b])
    batch.filter(mx.array([1]))
    assert batch.index_keys.shape[0] == 1
    # No shared left padding here (equal lengths) so the row is intact.
    assert mx.array_equal(batch.index_keys[:, -40:], ik_b).item()


# --------------------------------------------------------------------------- #
# Serialization handler (dense store)
# --------------------------------------------------------------------------- #


def test_handler_dense_roundtrip():
    _, QSATQ, _ = _classes()
    from omlx.cache.type_registry import CacheTypeRegistry

    src = _populated_qsa(20, seed=10)
    tq = QSATQ.from_cache(src, bits=4.0)
    handler = CacheTypeRegistry.get_handler_by_class_name("QSATurboQuantKVCache")
    assert handler.supports_block_slicing is True

    ser = handler.serialize_state(tq)
    assert ser[0].dtype == mx.bfloat16  # dense keys
    assert mx.array_equal(ser[2], src.index_keys).item()  # index verbatim

    restored = handler.deserialize_state(ser)
    assert type(restored).__name__ == "QSAKVCache"  # dense; scheduler reconverts
    assert mx.array_equal(restored.index_keys, src.index_keys).item()


# --------------------------------------------------------------------------- #
# MTP multi-row verify -- the path the naive approach crashed on
# --------------------------------------------------------------------------- #


def _tiny_attention():
    """A single Qwen4ExpAttention layer at head_dim=256 (production width)."""
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp import TextConfig
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpAttention

    cfg = TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=512,
        num_hidden_layers=1,
        num_attention_heads=8,
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
        max_position_embeddings=4096,
        hc_count=2,
        hc_lowrank=8,
        head_dim=256,
        layer_types=["full_attention"],
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
        indexer_head_dim=128,
        indexer_budget=16,
        indexer_compress_ratio=4,
        eos_token_id=1,
        rope_parameters={
            "rope_type": "default",
            "mrope_section": [2, 1, 1],
            "rope_theta": 10_000,
            "partial_rotary_factor": 1.0,
        },
    )
    attn = Qwen4ExpAttention(cfg)
    mx.eval(attn.parameters())
    return attn, cfg


def _apply_patches():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from omlx.patches.turboquant_attention import apply_turboquant_attention_patch

    apply_turboquant_attention_patch()


def test_mtp_verify_multirow_b1_no_crash_and_finite():
    """L=4 verify over a quantized QSA cache must not hit the
    non-subscriptable-proxy crash (the naive-approach failure) and must produce
    finite, sanely-scaled output.

    The crash is the deliverable: without the inherited TurboQuant verify patch
    the qwen3_5 per-row loop slices ``keys[:, :, :prefix+i+1, :]`` on a
    ``_QuantizedStateProxy`` (no ``__getitem__``) and raises. Reaching this line
    at all requires target_verify AND L>1. We assert no exception, no NaN/Inf,
    correct shape, and that the result tracks a full-precision reference within
    a generous bound -- an exact match is *not* expected: the TQ path quantizes
    the appended verify K/V, and target_verify routes projections through
    approximate verify kernels. Seeded for determinism under any ordering."""
    _apply_patches()
    QSAKVCache, QSATQ, _ = _classes()
    attn, _ = _tiny_attention()

    prefill_len = 64  # > indexer_budget so the sparse mask engages
    mx.random.seed(20260829)
    x_pre = mx.random.normal((1, prefill_len, 512))
    x_ver = mx.random.normal((1, 4, 512))
    mx.eval(x_pre, x_ver)

    pos_pre = mx.arange(prefill_len, dtype=mx.int32).reshape(1, prefill_len)
    warm = QSAKVCache()
    mx.eval(attn(x_pre, mask="causal", cache=warm, position_ids=pos_pre))
    tq = QSATQ.from_cache(warm, bits=4.0)

    # Full-precision reference (exact bf16 K/V) for a sanity bound only.
    ref = QSAKVCache()
    ref.update_and_fetch(*warm.state[:2])
    ref.index_keys = warm.index_keys
    ref.index_position_ids = warm.index_position_ids

    pos_ver = mx.arange(prefill_len, prefill_len + 4, dtype=mx.int32).reshape(1, 4)
    out_tq = attn(
        x_ver, mask="causal", cache=tq, position_ids=pos_ver, target_verify=True
    )
    out_ref = attn(
        x_ver, mask="causal", cache=ref, position_ids=pos_ver, target_verify=True
    )
    mx.eval(out_tq, out_ref)

    assert out_tq.shape == (1, 4, 512)  # did NOT crash on the proxy slice
    assert not mx.any(mx.isnan(out_tq)).item()
    assert mx.all(mx.isfinite(out_tq)).item()
    # Tracks the full-precision reference (catches gross plumbing errors);
    # loose because 4-bit codec + approximate verify kernels legitimately differ.
    max_diff = mx.max(mx.abs(out_tq - out_ref)).item()
    assert max_diff < 0.5, f"verify vs fp reference max abs diff {max_diff}"


def test_mtp_verify_multirow_b2_no_crash():
    """Batched (B=2) verify over the quantized batch cache must not hit the
    non-subscriptable-proxy crash. The QSA sparse indexer uses a scalar
    ``past_len`` and so is inactive under batched decode (context <= budget);
    the batched TQ *K/V* verify path -- the thing that crashed before -- is what
    this exercises through the patch's dequantize-per-row branch."""
    _apply_patches()
    QSAKVCache, QSATQ, _ = _classes()
    attn, _ = _tiny_attention()

    ctx = 12  # < indexer_budget (16) so the sparse mask stays inactive
    a = QSATQ.from_cache(_warm_layer(attn, ctx, seed=11), bits=4.0)
    b = QSATQ.from_cache(_warm_layer(attn, ctx, seed=12), bits=4.0)
    batch = QSATQ.merge([a, b])  # BatchQSATurboQuantKVCache

    mx.random.seed(4242)
    x_ver = mx.random.normal((2, 4, 512))
    pos_ver = mx.broadcast_to(
        mx.arange(ctx, ctx + 4, dtype=mx.int32).reshape(1, 4), (2, 4)
    )
    out = attn(
        x_ver, mask="causal", cache=batch, position_ids=pos_ver, target_verify=True
    )
    mx.eval(out)
    assert out.shape[0] == 2
    assert not mx.any(mx.isnan(out)).item()


def _warm_layer(attn, tokens: int, *, seed: int):
    QSAKVCache, _, _ = _classes()
    mx.random.seed(seed)
    warm = QSAKVCache()
    _ = attn(
        mx.random.normal((1, tokens, 512)),
        mask="causal",
        cache=warm,
        position_ids=None,
    )
    mx.eval(_)
    return warm


def test_verify_patch_recognizes_subclasses():
    """Protects the 'subclass => free MTP support' invariant."""
    _apply_patches()
    _, QSATQ, BatchQSATQ = _classes()
    from mlx_vlm.turboquant import TurboQuantKVCache

    from omlx.turboquant_kv import BatchTurboQuantKVCache

    assert isinstance(QSATQ(bits=4.0), (TurboQuantKVCache, BatchTurboQuantKVCache))
    assert isinstance(
        BatchQSATQ([0], bits=4.0), (TurboQuantKVCache, BatchTurboQuantKVCache)
    )


# --------------------------------------------------------------------------- #
# Scheduler eligibility + conversion (the "looks fixed, isn't" trap)
# --------------------------------------------------------------------------- #


def _scheduler_stub():
    """A minimal object carrying the conversion methods, unbound."""
    from omlx.scheduler import Scheduler

    stub = SimpleNamespace(
        _turboquant_kv_bits=4,
        _turboquant_skip_last=False,
        _model_uses_mla=lambda: False,
        _model_uses_attention_sinks=lambda: False,
    )
    stub._turboquant_eligible = Scheduler._turboquant_eligible.__get__(stub)
    stub._apply_turboquant_kv_convert = Scheduler._apply_turboquant_kv_convert.__get__(
        stub
    )
    stub._apply_turboquant_kv_empty = Scheduler._apply_turboquant_kv_empty.__get__(stub)
    return stub


def test_turboquant_eligible_accepts_qsa_hybrid():
    QSAKVCache, _, _ = _classes()
    # The VENDORED qwen4_exp ArraysCache (independent _BaseCache subclass) --
    # NOT mlx_lm's. The 36 GDN layers use this; using mlx_lm's here would mask
    # the eligibility gap that kept conversion from firing in production.
    from mlx_vlm.models.qwen4_exp.cache import ArraysCache

    stub = _scheduler_stub()
    cache_list = [QSAKVCache(), ArraysCache(size=4), QSAKVCache()]
    assert stub._turboquant_eligible(cache_list) is True


def test_conversion_actually_converts_qsa_layers():
    """The trap: eligibility flips but the isinstance(KVCache) branch skips QSA,
    leaving converted == 0. Assert real conversion happens."""
    QSAKVCache, QSATQ, _ = _classes()
    # The VENDORED qwen4_exp ArraysCache (independent _BaseCache subclass) --
    # NOT mlx_lm's. The 36 GDN layers use this; using mlx_lm's here would mask
    # the eligibility gap that kept conversion from firing in production.
    from mlx_vlm.models.qwen4_exp.cache import ArraysCache

    stub = _scheduler_stub()
    src = _populated_qsa(30, seed=13)
    prompt_cache = [src, ArraysCache(size=4), _populated_qsa(30, seed=14)]

    stub._apply_turboquant_kv_convert(prompt_cache)
    converted = [type(c).__name__ for c in prompt_cache]
    assert converted == ["QSATurboQuantKVCache", "ArraysCache", "QSATurboQuantKVCache"]
    # And the indexer survived conversion at full precision.
    assert mx.array_equal(prompt_cache[0].index_keys, src.index_keys).item()


def test_conversion_empty_sets_qsa_turboquant():
    QSAKVCache, QSATQ, _ = _classes()
    # The VENDORED qwen4_exp ArraysCache (independent _BaseCache subclass) --
    # NOT mlx_lm's. The 36 GDN layers use this; using mlx_lm's here would mask
    # the eligibility gap that kept conversion from firing in production.
    from mlx_vlm.models.qwen4_exp.cache import ArraysCache

    stub = _scheduler_stub()
    prompt_cache = [QSAKVCache(), ArraysCache(size=4)]
    stub._apply_turboquant_kv_empty(prompt_cache)
    assert type(prompt_cache[0]).__name__ == "QSATurboQuantKVCache"
    assert type(prompt_cache[1]).__name__ == "ArraysCache"


# --------------------------------------------------------------------------- #
# Estimator split-width formula (design doc §5 numbers)
# --------------------------------------------------------------------------- #


def test_estimator_bf16_and_quantized_widths():
    from omlx.memory_monitor import estimate_qwen4_exp_kv_bytes_per_token

    QSAKVCache, _, _ = _classes()
    config = SimpleNamespace(
        model_type="qwen4_exp",
        num_key_value_heads=2,
        head_dim=256,
        indexer_head_dim=128,
    )
    cache_list = [QSAKVCache() for _ in range(12)]

    # bf16 (pre-Fix-2 / phase-A default): 12 x (2*2*256*2 + 128*2 + 24) = 27,936
    bf16 = estimate_qwen4_exp_kv_bytes_per_token(config, cache_list, 2.0)
    assert bf16 == pytest.approx(27_936.0)

    # 4-bit TurboQuant K/V (phase B): width = 0.5 + 2/256 = 0.5078125
    # 12 x (2*2*256*0.5078125 + 128*2 + 24) = 12 x 800 = 9,600
    tq_w = 4.0 / 8.0 + 2.0 / 256.0
    quant = estimate_qwen4_exp_kv_bytes_per_token(
        config, cache_list, 2.0, kv_dtype_size=tq_w
    )
    assert quant == pytest.approx(9_600.0)

    # Indexer stays full precision: the drop is exactly the K/V term.
    kv_bf16 = 12 * 2 * 2 * 256 * 2.0
    kv_quant = 12 * 2 * 2 * 256 * tq_w
    assert bf16 - quant == pytest.approx(kv_bf16 - kv_quant)


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
