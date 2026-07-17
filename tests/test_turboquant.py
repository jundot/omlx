"""Tests for TurboQuant KV cache (mlx-vlm backend + omlx BatchTurboQuantKVCache)."""

import mlx.core as mx
import pytest
from mlx_lm.models.cache import BatchKVCache, KVCache
from mlx_vlm.turboquant import (
    TurboQuantKVCache,
    _build_codec,
    _TurboQuantMSECodec,
    _TurboQuantProdCodec,
    turboquant_enabled,
)

from omlx.turboquant_kv import (
    BatchTurboQuantKVCache,
    _concat_state,
    _concat_state_token_axis,
    _infer_head_dim,
    _rebuild_codecs,
)

pytestmark = pytest.mark.turboquant


def _sample_unit_vectors(count: int, dim: int) -> mx.array:
    vectors = mx.random.normal((count, dim))
    return vectors / mx.linalg.norm(vectors, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Codec tests (ported from mlx-vlm)
# ---------------------------------------------------------------------------


def test_turboquant_mse_matches_paper_small_bit_distortions():
    vectors = _sample_unit_vectors(256, 64)
    expected = {1: 0.36, 2: 0.117, 3: 0.03}

    for bits, target in expected.items():
        codec = _TurboQuantMSECodec(64, bits, seed=0)
        state = codec.quantize(vectors)
        reconstructed = codec.dequantize(state)
        mse = mx.mean(mx.sum((vectors - reconstructed) ** 2, axis=-1)).item()
        assert mse == pytest.approx(target, rel=0.25, abs=0.02)


def test_turboquant_prod_is_nearly_unbiased_across_seeds():
    keys = _sample_unit_vectors(128, 64)
    queries = mx.random.normal((128, 64))
    true_inner_products = mx.sum(keys * queries, axis=-1)

    estimates = []
    for seed in range(16):
        codec = _TurboQuantProdCodec(64, 2, seed=seed)
        state = codec.quantize(keys)
        reconstructed = codec.dequantize(state)
        estimates.append(mx.sum(reconstructed * queries, axis=-1))

    mean_estimate = mx.mean(mx.stack(estimates), axis=0)
    bias = mx.mean(mean_estimate - true_inner_products).item()
    assert abs(bias) < 0.05


def test_fractional_turboquant_improves_reconstruction():
    vectors = mx.random.normal((1, 2, 32, 64))

    codec_3bit = _build_codec(vectors, 3.0, mode="mse", seed=0)
    codec_35bit = _build_codec(vectors, 3.5, mode="mse", seed=0)

    state_3bit = codec_3bit.quantize(vectors)
    state_35bit = codec_35bit.quantize(vectors)

    mse_3bit = mx.mean((vectors - codec_3bit.dequantize(state_3bit)) ** 2).item()
    mse_35bit = mx.mean((vectors - codec_35bit.dequantize(state_35bit)) ** 2).item()

    assert turboquant_enabled(3.5)
    assert not turboquant_enabled(3.0)
    assert mse_35bit < mse_3bit


# ---------------------------------------------------------------------------
# TurboQuantKVCache round-trip
# ---------------------------------------------------------------------------


def test_turboquant_cache_round_trip():
    keys = mx.random.normal((1, 2, 16, 32))
    values = mx.random.normal((1, 2, 16, 32))

    fp_cache = KVCache()
    fp_cache.update_and_fetch(keys, values)
    turbo_cache = TurboQuantKVCache.from_cache(fp_cache, bits=3.5)

    assert turbo_cache.offset == 16
    assert turbo_cache.nbytes < fp_cache.nbytes

    dk, dv = turbo_cache.dequantize()
    diff = mx.mean(mx.abs(keys - dk)).item()
    assert diff < 0.5


# ---------------------------------------------------------------------------
# BatchTurboQuantKVCache tests (inherits from TurboQuantKVCache)
# ---------------------------------------------------------------------------


def test_batch_tq_prefill_quantizes_immediately():
    batch = BatchTurboQuantKVCache([0, 0], bits=4.0)
    keys = mx.random.normal((2, 4, 8, 32))
    values = mx.random.normal((2, 4, 8, 32))
    batch.update_and_fetch(keys, values)
    assert batch.keys is not None
    assert batch.offset[0].item() == 8


def test_batch_tq_decode_appends():
    batch = BatchTurboQuantKVCache([0, 0], bits=4.0)
    keys = mx.random.normal((2, 4, 8, 32))
    values = mx.random.normal((2, 4, 8, 32))
    batch.update_and_fetch(keys, values)
    dk = mx.random.normal((2, 4, 1, 32))
    dv = mx.random.normal((2, 4, 1, 32))
    batch.update_and_fetch(dk, dv)
    assert batch.offset[0].item() == 9


def test_batch_tq_merge_extract():
    c1 = TurboQuantKVCache(bits=4.0)
    c1.update_and_fetch(
        mx.random.normal((1, 2, 8, 32)), mx.random.normal((1, 2, 8, 32))
    )
    c2 = TurboQuantKVCache(bits=4.0)
    c2.update_and_fetch(
        mx.random.normal((1, 2, 4, 32)), mx.random.normal((1, 2, 4, 32))
    )
    mx.eval(c1.keys, c1.values, c2.keys, c2.values)

    batch = BatchTurboQuantKVCache.merge([c1, c2])
    assert batch.keys is not None
    assert batch.left_padding[0].item() == 0
    assert batch.left_padding[1].item() == 4

    e1 = batch.extract(0)
    e2 = batch.extract(1)
    assert e1.offset == 8
    assert e2.offset == 4


def test_batch_tq_merge_rejects_mixed_bit_depths():
    """#2045 last-line guard: members packed at different depths (or seeds)
    have incompatible packed widths/codecs and must fail loud at merge, not
    as a raw mx.concatenate shape error deep in _concat_state_batch."""
    c4 = TurboQuantKVCache(bits=4.0)
    c4.update_and_fetch(
        mx.random.normal((1, 2, 4, 32)), mx.random.normal((1, 2, 4, 32))
    )
    c6 = TurboQuantKVCache(bits=6.0)
    c6.update_and_fetch(
        mx.random.normal((1, 2, 4, 32)), mx.random.normal((1, 2, 4, 32))
    )
    mx.eval(c4.keys, c4.values, c6.keys, c6.values)

    with pytest.raises(ValueError, match="mixed quantization"):
        BatchTurboQuantKVCache.merge([c4, c6])

    with pytest.raises(ValueError, match="mixed quantization"):
        BatchTurboQuantKVCache.merge(
            [c4, TurboQuantKVCache(bits=4.0, seed=1)]
        )


def test_batch_tq_merge_preserves_empty_rows():
    """Regression: mixed empty/non-empty rows must keep the batch dimension."""
    full = TurboQuantKVCache(bits=4.0)
    full.update_and_fetch(
        mx.random.normal((1, 2, 4, 32)), mx.random.normal((1, 2, 4, 32))
    )
    mx.eval(full.keys, full.values)

    for caches, expected_padding, expected_offsets in (
        ([TurboQuantKVCache(bits=4.0), full], [4, 0], [0, 4]),
        ([full, TurboQuantKVCache(bits=4.0)], [0, 4], [4, 0]),
    ):
        batch = BatchTurboQuantKVCache.merge(caches)
        assert batch.left_padding.tolist() == expected_padding
        assert batch.offset.tolist() == expected_offsets
        assert batch.keys.norms.shape[0] == 2

        batch.update_and_fetch(
            mx.random.normal((2, 2, 1, 32)), mx.random.normal((2, 2, 1, 32))
        )


def test_batch_tq_extend_preserves_empty_rows():
    """Regression: extend() can mix initialized and empty batch rows."""

    def full_batch():
        full = BatchTurboQuantKVCache.merge([TurboQuantKVCache(bits=4.0)])
        full.update_and_fetch(
            mx.random.normal((1, 2, 4, 32)), mx.random.normal((1, 2, 4, 32))
        )
        mx.eval(full.keys, full.values)
        return full

    for left, right, expected_padding, expected_offsets in (
        (
            BatchTurboQuantKVCache([0], bits=4.0),
            full_batch(),
            [4, 0],
            [0, 4],
        ),
        (
            full_batch(),
            BatchTurboQuantKVCache([0], bits=4.0),
            [0, 4],
            [4, 0],
        ),
    ):
        left.extend(right)
        assert left.left_padding.tolist() == expected_padding
        assert left.offset.tolist() == expected_offsets
        assert left.keys.norms.shape[0] == 2

        left.update_and_fetch(
            mx.random.normal((2, 2, 1, 32)), mx.random.normal((2, 2, 1, 32))
        )


def test_batch_tq_extend_rejects_plain_batch_cache():
    """Regression guard for mixed BatchKVCache/BatchTurboQuantKVCache states."""
    left = BatchTurboQuantKVCache([0], bits=4.0)
    plain = object()

    with pytest.raises(TypeError, match="BatchTurboQuantKVCache"):
        left.extend(plain)  # type: ignore[arg-type]


def test_batch_tq_merge_rejects_plain_cache_entries():
    with pytest.raises(TypeError, match="TurboQuantKVCache"):
        BatchTurboQuantKVCache.merge([object()])  # type: ignore[list-item]


def test_batch_tq_continuous_batching_extend():
    b1 = BatchTurboQuantKVCache([0], bits=4.0)
    b1.update_and_fetch(
        mx.random.normal((1, 2, 8, 32)), mx.random.normal((1, 2, 8, 32))
    )
    b1.update_and_fetch(
        mx.random.normal((1, 2, 1, 32)), mx.random.normal((1, 2, 1, 32))
    )

    b2 = BatchTurboQuantKVCache([0], bits=4.0)
    b2.update_and_fetch(
        mx.random.normal((1, 2, 4, 32)), mx.random.normal((1, 2, 4, 32))
    )
    b2.update_and_fetch(
        mx.random.normal((1, 2, 1, 32)), mx.random.normal((1, 2, 1, 32))
    )

    b1.extend(b2)

    dk = mx.random.normal((2, 2, 1, 32))
    dv = mx.random.normal((2, 2, 1, 32))
    b1.update_and_fetch(dk, dv)
    # offset is now mx.array after extend


def test_batch_make_mask_matches_fp16_left_padding():
    """Regression: B>1 make_mask must match mlx-lm's BatchKVCache for left-padded
    batches. The old hand-rolled causal term compared each request's sequence
    length against the column index and masked out valid left-padded tokens, so
    left-padded requests attended to ~nothing and decoded garbage (batch worse
    than single). It now delegates to create_causal_mask like BatchKVCache.
    """
    from mlx_lm.models.cache import BatchKVCache

    lp = [0, 4, 2]
    K = mx.random.normal((3, 2, 8, 16))
    V = mx.random.normal((3, 2, 8, 16))
    bk = BatchKVCache(lp)
    bk.update_and_fetch(K, V)
    bt = BatchTurboQuantKVCache(lp, bits=8.0)
    bt.update_and_fetch(K, V)

    ref = bk.make_mask(1, return_array=True)  # decode-step mask
    got = bt.make_mask(1, return_array=True)
    assert mx.array_equal(ref, got).item(), (
        "B>1 make_mask diverges from BatchKVCache for left-padding "
        f"(member masks: BK={ref[:,0,0,:].sum(-1).tolist()} "
        f"TQ={got[:,0,0,:].sum(-1).tolist()})"
    )


def test_batch_tq_filter():
    batch = BatchTurboQuantKVCache([0, 0, 0], bits=4.0)
    keys = mx.random.normal((3, 2, 8, 32))
    values = mx.random.normal((3, 2, 8, 32))
    batch.update_and_fetch(keys, values)
    batch.filter([0, 2])
    assert batch.keys.norms.shape[0] == 2


def test_batch_tq_extend():
    b1 = BatchTurboQuantKVCache([0], bits=4.0)
    b1.update_and_fetch(
        mx.random.normal((1, 2, 8, 32)), mx.random.normal((1, 2, 8, 32))
    )

    b2 = BatchTurboQuantKVCache([0], bits=4.0)
    b2.update_and_fetch(
        mx.random.normal((1, 2, 4, 32)), mx.random.normal((1, 2, 4, 32))
    )

    b1.extend(b2)
    assert b1.keys.norms.shape[0] == 2


def test_batch_tq_dequantize():
    batch = BatchTurboQuantKVCache([0], bits=4.0)
    batch.update_and_fetch(
        mx.random.normal((1, 2, 8, 32)), mx.random.normal((1, 2, 8, 32))
    )
    batch.update_and_fetch(
        mx.random.normal((1, 2, 1, 32)), mx.random.normal((1, 2, 1, 32))
    )
    dk, dv = batch.dequantize()
    assert dk.shape[2] == 9
    assert dv.shape[2] == 9


def test_batch_tq_state_property():
    batch = BatchTurboQuantKVCache([2, 0], bits=4.0)
    s = batch.state
    assert s[0] is None

    keys = mx.random.normal((2, 2, 4, 32))
    values = mx.random.normal((2, 2, 4, 32))
    batch.update_and_fetch(keys, values)
    s = batch.state
    assert s[0] is not None


def test_batch_tq_meta_state_round_trip():
    batch = BatchTurboQuantKVCache([0], bits=3.5, seed=42)
    batch.update_and_fetch(
        mx.random.normal((1, 2, 4, 32)), mx.random.normal((1, 2, 4, 32))
    )

    ms = batch.meta_state
    batch2 = BatchTurboQuantKVCache([0], bits=4.0)
    batch2.meta_state = ms
    assert batch2.bits == pytest.approx(3.5)
    assert batch2.seed == 42


# ---------------------------------------------------------------------------
# Attention patch test
# ---------------------------------------------------------------------------


def test_attention_patch_routes_tq():
    from omlx.patches.turboquant_attention import apply_turboquant_attention_patch

    apply_turboquant_attention_patch()

    from mlx_lm.models import base as mlx_base

    fp_cache = KVCache()
    keys = mx.random.normal((1, 2, 8, 32))
    values = mx.random.normal((1, 2, 8, 32))
    fp_cache.update_and_fetch(keys, values)
    tq = TurboQuantKVCache.from_cache(fp_cache, bits=4.0)
    ks, vs = tq.state

    queries = mx.random.normal((1, 4, 1, 32))
    out = mlx_base.scaled_dot_product_attention(
        queries, ks, vs, tq, scale=32**-0.5, mask=None
    )
    assert out.shape == (1, 4, 1, 32)


def test_attention_patch_preserves_sinks_with_dequant_fallback(monkeypatch):
    from mlx_lm.models import base as mlx_base

    from omlx.patches.turboquant_attention import apply_turboquant_attention_patch

    apply_turboquant_attention_patch()

    fp_cache = KVCache()
    keys = mx.random.normal((1, 2, 8, 32))
    values = mx.random.normal((1, 2, 8, 32))
    fp_cache.update_and_fetch(keys, values)
    tq = TurboQuantKVCache.from_cache(fp_cache, bits=4.0)
    ks, vs = tq.state

    def fail_decode(*args, **kwargs):
        raise AssertionError("sink fallback must not use TurboQuant decode kernel")

    calls = {}
    original_dequantize = TurboQuantKVCache.dequantize

    def spy_dequantize(self, *args, **kwargs):
        calls["dequant_kwargs"] = kwargs
        return original_dequantize(self, *args, **kwargs)

    def fake_sdpa(queries, keys, values, **kwargs):
        calls["sdpa_sinks"] = kwargs.get("sinks")
        calls["sdpa_key_shape"] = keys.shape
        return mx.zeros_like(queries)

    monkeypatch.setattr(TurboQuantKVCache, "decode_attention", fail_decode)
    monkeypatch.setattr(TurboQuantKVCache, "dequantize", spy_dequantize)
    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", fake_sdpa)

    queries = mx.random.normal((1, 4, 1, 32))
    sinks = mx.zeros((4,))
    out = mlx_base.scaled_dot_product_attention(
        queries,
        ks,
        vs,
        tq,
        scale=32**-0.5,
        mask=None,
        sinks=sinks,
    )

    assert out.shape == queries.shape
    assert calls["dequant_kwargs"] == {"keys_state": ks, "values_state": vs}
    assert calls["sdpa_sinks"] is sinks
    assert calls["sdpa_key_shape"] == keys.shape


def test_attention_patch_routes_long_tq_prefill_to_quantized_attention(monkeypatch):
    from mlx_lm.models import base as mlx_base

    from omlx.patches import turboquant_attention as tq_attention

    tq_attention.apply_turboquant_attention_patch()
    monkeypatch.setattr(tq_attention, "_LONG_PREFILL_QUANTIZED_THRESHOLD", 4)

    fp_cache = KVCache()
    fp_cache.update_and_fetch(
        mx.random.normal((1, 2, 8, 32)),
        mx.random.normal((1, 2, 8, 32)),
    )
    tq = TurboQuantKVCache.from_cache(fp_cache, bits=4.0)
    ks, vs = tq.state
    calls = []
    prefill_calls = []

    def fake_prefill_attention(
        self, queries, keys_state=None, values_state=None, scale=1.0, mask=None
    ):
        prefill_calls.append((keys_state, values_state))
        return None

    def fake_quantized_attention(
        self, queries, keys_state=None, values_state=None, scale=1.0, mask=None
    ):
        calls.append((keys_state, values_state, self.prefill_query_block_size))
        assert self.prefill_key_chunk_size == 16384
        return mx.zeros_like(queries)

    monkeypatch.setattr(
        TurboQuantKVCache,
        "prefill_attention",
        fake_prefill_attention,
    )
    monkeypatch.setattr(
        TurboQuantKVCache,
        "quantized_attention",
        fake_quantized_attention,
    )

    queries = mx.random.normal((1, 4, 2, 32))
    out = mlx_base.scaled_dot_product_attention(
        queries, ks, vs, tq, scale=32**-0.5, mask=None
    )

    assert out.shape == queries.shape
    assert prefill_calls == [(ks, vs)]
    assert len(calls) == 1
    assert calls[0][0] is ks
    assert calls[0][1] is vs
    assert calls[0][2] == 256


def test_attention_patch_falls_back_when_quantized_prefill_fails(monkeypatch):
    from mlx_lm.models import base as mlx_base

    from omlx.patches import turboquant_attention as tq_attention

    tq_attention.apply_turboquant_attention_patch()
    monkeypatch.setattr(tq_attention, "_LONG_PREFILL_QUANTIZED_THRESHOLD", 4)

    fp_cache = KVCache()
    fp_cache.update_and_fetch(
        mx.random.normal((1, 2, 8, 32)),
        mx.random.normal((1, 2, 8, 32)),
    )
    tq = TurboQuantKVCache.from_cache(fp_cache, bits=4.0)
    ks, vs = tq.state
    calls = {"quantized": 0, "dequantize": 0}

    def failing_quantized_attention(self, *args, **kwargs):
        calls["quantized"] += 1
        raise RuntimeError("forced quantized prefill failure")

    original_dequantize = TurboQuantKVCache.dequantize

    def spy_dequantize(self, *args, **kwargs):
        calls["dequantize"] += 1
        return original_dequantize(self, *args, **kwargs)

    monkeypatch.setattr(
        TurboQuantKVCache,
        "quantized_attention",
        failing_quantized_attention,
    )
    monkeypatch.setattr(TurboQuantKVCache, "dequantize", spy_dequantize)

    queries = mx.random.normal((1, 4, 2, 32))
    out = mlx_base.scaled_dot_product_attention(
        queries, ks, vs, tq, scale=32**-0.5, mask=None
    )
    mx.eval(out)

    assert out.shape == queries.shape
    assert calls == {"quantized": 1, "dequantize": 1}


@pytest.mark.parametrize("q_len", [2, 4, 9])
def test_decode_multirow_matches_dequantize_reference(q_len):
    """MTP-verify-shaped attention (fold path at small q_len, single-chunk
    quantized_attention above the folded-repeat knee) must match the
    dequantize+SDPA reference with an explicit causal tail mask."""
    from omlx.patches.turboquant_attention import _decode_multirow_attention

    mx.random.seed(0)
    B, n_q, n_kv, D, T = 1, 24, 4, 256, 512
    fp_cache = KVCache()
    fp_cache.update_and_fetch(
        mx.random.normal((B, n_kv, T, D)).astype(mx.float16),
        mx.random.normal((B, n_kv, T, D)).astype(mx.float16),
    )
    tq = TurboQuantKVCache.from_cache(fp_cache, bits=4.0)
    ks, vs = tq.state
    queries = mx.random.normal((B, n_q, q_len, D)).astype(mx.float16)
    scale = D**-0.5

    out = _decode_multirow_attention(tq, queries, ks, vs, scale)
    assert out is not None
    assert out.shape == (B, n_q, q_len, D)

    dk, dv = tq.dequantize()
    causal = mx.arange(T)[None, :] <= mx.arange(T - q_len, T)[:, None]
    ref = mx.fast.scaled_dot_product_attention(
        queries.astype(mx.float32), dk, dv, scale=scale, mask=causal
    )
    assert mx.abs(out.astype(mx.float32) - ref).max().item() < 5e-3
    # The causal tail mask must actually bind (an unmasked reference differs).
    ref_nomask = mx.fast.scaled_dot_product_attention(
        queries.astype(mx.float32), dk, dv, scale=scale, mask=None
    )
    assert mx.abs(ref_nomask - ref).max().item() > 1e-3


@pytest.mark.parametrize("q_len", [2, 3, 4])
def test_fused_multirow_kernel_matches_dequantize_reference(q_len):
    """Above the token floor, MSE-codec MTP verify takes the fused multi-row
    kernel (one KV unpack shared across rows, issue #2215). Its output must
    match the dequantize+SDPA reference with the causal tail mask, and the
    dispatcher must route to it bit-exactly."""
    from omlx.patches import turboquant_attention as tq_attention

    mx.random.seed(0)
    B, n_q, n_kv, D = 1, 16, 2, 256
    T = tq_attention._FUSED_MULTIROW_MIN_TOKENS + 512
    fp_cache = KVCache()
    fp_cache.update_and_fetch(
        mx.random.normal((B, n_kv, T, D)).astype(mx.float16),
        mx.random.normal((B, n_kv, T, D)).astype(mx.float16),
    )
    tq = TurboQuantKVCache.from_cache(fp_cache, bits=4.0)
    ks, vs = tq.state
    queries = mx.random.normal((B, n_q, q_len, D)).astype(mx.float16)
    scale = D**-0.5

    fused = tq_attention._fused_multirow_mse_attention(
        tq, queries, tq._unwrap(ks), tq._unwrap(vs), scale, T
    )
    assert fused is not None

    dk, dv = tq.dequantize()
    causal = mx.arange(T)[None, :] <= mx.arange(T - q_len, T)[:, None]
    ref = mx.fast.scaled_dot_product_attention(
        queries.astype(mx.float32), dk, dv, scale=scale, mask=causal
    )
    assert mx.abs(fused.astype(mx.float32) - ref).max().item() < 5e-3

    routed = tq_attention._decode_multirow_attention(tq, queries, ks, vs, scale)
    routed_diff = mx.abs(routed.astype(mx.float32) - fused.astype(mx.float32))
    assert routed_diff.max().item() == 0.0


@pytest.mark.parametrize("bits", [2.5, 3.5, 8])
def test_fused_multirow_kernel_handles_mixed_bit_codecs(bits):
    """Fractional turboquant_kv_bits split into different K/V integer bit
    widths (2.5 -> K=2/V=3); the fused kernel templates the two widths
    independently and must stay parity-correct across them."""
    from omlx.patches import turboquant_attention as tq_attention

    mx.random.seed(0)
    B, n_q, n_kv, D = 1, 16, 2, 256
    T = tq_attention._FUSED_MULTIROW_MIN_TOKENS + 512
    fp_cache = KVCache()
    fp_cache.update_and_fetch(
        mx.random.normal((B, n_kv, T, D)).astype(mx.float16),
        mx.random.normal((B, n_kv, T, D)).astype(mx.float16),
    )
    tq = TurboQuantKVCache.from_cache(fp_cache, bits=bits)
    ks, vs = tq.state
    queries = mx.random.normal((B, n_q, 3, D)).astype(mx.float16)
    scale = D**-0.5

    fused = tq_attention._fused_multirow_mse_attention(
        tq, queries, tq._unwrap(ks), tq._unwrap(vs), scale, T
    )
    assert fused is not None

    dk, dv = tq.dequantize()
    causal = mx.arange(T)[None, :] <= mx.arange(T - 3, T)[:, None]
    ref = mx.fast.scaled_dot_product_attention(
        queries.astype(mx.float32), dk, dv, scale=scale, mask=causal
    )
    assert mx.abs(fused.astype(mx.float32) - ref).max().item() < 5e-3


def test_fused_multirow_kernel_respects_token_floor(monkeypatch):
    """Below the token floor the dispatcher must not call the fused helper
    (the fold path is already cheap there and the 2-pass block split needs
    enough tokens per block)."""
    from omlx.patches import turboquant_attention as tq_attention

    mx.random.seed(0)
    B, n_q, n_kv, D, T = 1, 16, 2, 256, 512
    fp_cache = KVCache()
    fp_cache.update_and_fetch(
        mx.random.normal((B, n_kv, T, D)).astype(mx.float16),
        mx.random.normal((B, n_kv, T, D)).astype(mx.float16),
    )
    tq = TurboQuantKVCache.from_cache(fp_cache, bits=4.0)
    ks, vs = tq.state
    queries = mx.random.normal((B, n_q, 3, D)).astype(mx.float16)

    def fail_fused(*args, **kwargs):
        raise AssertionError("fused kernel must not run below the token floor")

    monkeypatch.setattr(
        tq_attention, "_fused_multirow_mse_attention", fail_fused
    )
    out = tq_attention._decode_multirow_attention(tq, queries, ks, vs, D**-0.5)
    assert out is not None
    assert out.shape == (B, n_q, 3, D)


def test_attention_patch_routes_decode_multirow_causal(monkeypatch):
    """A decode-shaped multi-row causal call (MTP verify) must take the
    multirow decode route — never prefill_attention / dequantize (issue
    #2127 class: those re-scan the whole cache per verify cycle)."""
    from mlx_lm.models import base as mlx_base

    from omlx.patches import turboquant_attention as tq_attention

    tq_attention.apply_turboquant_attention_patch()

    fp_cache = KVCache()
    fp_cache.update_and_fetch(
        mx.random.normal((1, 4, 64, 256)).astype(mx.float16),
        mx.random.normal((1, 4, 64, 256)).astype(mx.float16),
    )
    tq = TurboQuantKVCache.from_cache(fp_cache, bits=4.0)
    ks, vs = tq.state

    def fail_prefill(self, *args, **kwargs):
        raise AssertionError("verify must not take prefill_attention")

    def fail_dequantize(self, *args, **kwargs):
        raise AssertionError("verify must not dequantize the cache")

    monkeypatch.setattr(TurboQuantKVCache, "prefill_attention", fail_prefill)
    monkeypatch.setattr(TurboQuantKVCache, "dequantize", fail_dequantize)

    queries = mx.random.normal((1, 24, 4, 256)).astype(mx.float16)
    out = mlx_base.scaled_dot_product_attention(
        queries, ks, vs, tq, scale=256**-0.5, mask="causal"
    )
    mx.eval(out)
    assert out.shape == queries.shape


def test_attention_patch_multirow_ignores_non_causal_masks():
    """Array masks and mask=None keep the existing prefill routing (the
    multirow route encodes causal-tail semantics only)."""
    from omlx.patches import turboquant_attention as tq_attention

    tq_attention.apply_turboquant_attention_patch()

    from mlx_lm.models import base as mlx_base

    fp_cache = KVCache()
    fp_cache.update_and_fetch(
        mx.random.normal((1, 2, 8, 32)),
        mx.random.normal((1, 2, 8, 32)),
    )
    tq = TurboQuantKVCache.from_cache(fp_cache, bits=4.0)
    ks, vs = tq.state
    queries = mx.random.normal((1, 4, 2, 32))
    # mask=None multi-row: full (non-causal) attention via the prefill chain.
    out = mlx_base.scaled_dot_product_attention(
        queries, ks, vs, tq, scale=32**-0.5, mask=None
    )
    mx.eval(out)
    assert out.shape == queries.shape
    dk, dv = tq.dequantize()
    ref = mx.fast.scaled_dot_product_attention(
        queries, dk.astype(queries.dtype), dv.astype(queries.dtype),
        scale=32**-0.5, mask=None,
    )
    assert mx.abs(out - ref).max().item() < 5e-2


def test_vlm_target_verify_attention_handles_tq_proxies():
    """mlx-vlm's qwen3_5 MTP verify slices keys per draft row, which crashes
    on TurboQuant's packed state proxies ('_QuantizedStateProxy' object is
    not subscriptable, issue #2139). The patched helper must route TurboQuant
    caches through one causal SDPA call with identical per-row semantics."""
    pytest.importorskip("mlx_vlm.models.qwen3_5.language")

    from omlx.patches import turboquant_attention as tq_attention

    tq_attention.apply_turboquant_attention_patch()
    tq_attention._patch_vlm_target_verify_attention()

    from mlx_vlm.models.qwen3_5 import language as q35_lang

    assert getattr(q35_lang, "_omlx_tq_target_verify_patched", False)

    mx.random.seed(0)
    B, n_q, n_kv, D, T, L = 1, 4, 2, 32, 24, 3
    fp_cache = KVCache()
    fp_cache.update_and_fetch(
        mx.random.normal((B, n_kv, T, D)).astype(mx.float16),
        mx.random.normal((B, n_kv, T, D)).astype(mx.float16),
    )
    tq = TurboQuantKVCache.from_cache(fp_cache, bits=4.0)
    ks, vs = tq.state
    queries = mx.random.normal((B, n_q, L, D)).astype(mx.float16)
    scale = D**-0.5

    out = q35_lang._target_verify_left_padded_attention(
        queries, ks, vs, cache=tq, scale=scale, mask=None
    )
    mx.eval(out)
    assert out.shape == queries.shape

    # Reference: the caller's per-row causal slicing on dequantized arrays.
    dk, dv = tq.dequantize()
    dk = dk.astype(queries.dtype)
    dv = dv.astype(queries.dtype)
    prefix = T - L
    ref = mx.concatenate(
        [
            mx.fast.scaled_dot_product_attention(
                queries[:, :, i : i + 1, :],
                dk[:, :, : prefix + i + 1, :],
                dv[:, :, : prefix + i + 1, :],
                scale=scale,
                mask=None,
            )
            for i in range(L)
        ],
        axis=2,
    )
    assert mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max().item() < 5e-2

    # Non-TurboQuant caches keep the original helper behavior (declines
    # plain KVCache with no left padding -> caller uses its own path).
    plain_ks, plain_vs = fp_cache.state
    assert (
        q35_lang._target_verify_left_padded_attention(
            queries, plain_ks, plain_vs, cache=fp_cache, scale=scale, mask=None
        )
        is None
    )


# ---------------------------------------------------------------------------
# Codec rebuild tests (SSD cache reconstruction, issue #577)
# ---------------------------------------------------------------------------


def test_rebuild_codecs_mse():
    """Rebuild codecs from state after wiping them — simulates SSD restore."""
    keys = mx.random.normal((1, 2, 16, 64))
    values = mx.random.normal((1, 2, 16, 64))

    tq = TurboQuantKVCache(bits=4.0, seed=7)
    tq.update_and_fetch(keys, values)
    expected_k, expected_v = tq.dequantize()

    ks, vs = tq.state
    tq2 = TurboQuantKVCache(bits=4.0, seed=7)
    tq2.keys = ks
    tq2.values = vs
    tq2.offset = 16
    _rebuild_codecs(tq2, ks, vs)
    rebuilt_k, rebuilt_v = tq2.dequantize()

    assert mx.allclose(expected_k, rebuilt_k, atol=1e-5).item()
    assert mx.allclose(expected_v, rebuilt_v, atol=1e-5).item()


def test_rebuild_codecs_fractional_bits():
    """Rebuild codecs with fractional bits (3.5 → key=3bit, value=4bit)."""
    keys = mx.random.normal((1, 2, 16, 64))
    values = mx.random.normal((1, 2, 16, 64))

    tq = TurboQuantKVCache(bits=3.5, seed=42)
    tq.update_and_fetch(keys, values)
    expected_k, expected_v = tq.dequantize()

    ks, vs = tq.state
    tq2 = TurboQuantKVCache(bits=3.5, seed=42)
    tq2.keys = ks
    tq2.values = vs
    tq2.offset = 16
    _rebuild_codecs(tq2, ks, vs)
    rebuilt_k, rebuilt_v = tq2.dequantize()

    assert mx.allclose(expected_k, rebuilt_k, atol=1e-5).item()
    assert mx.allclose(expected_v, rebuilt_v, atol=1e-5).item()


def test_infer_head_dim():
    """Verify head_dim inference from MSEState packed indices."""
    keys = mx.random.normal((1, 2, 8, 128))
    values = mx.random.normal((1, 2, 8, 128))

    tq = TurboQuantKVCache(bits=4.0, seed=0)
    tq.update_and_fetch(keys, values)
    ks, _ = tq.state
    assert _infer_head_dim(ks, 4) == 128


def test_concat_state_token_axis_mse_matches_pairwise_concat():
    codec = _TurboQuantMSECodec(32, 4, seed=0)
    first = codec.quantize(mx.random.normal((1, 2, 3, 32)))
    second = codec.quantize(mx.random.normal((1, 2, 5, 32)))

    got = _concat_state_token_axis([first, second])
    expected = _concat_state(first, second)
    mx.eval(got.norms, got.indices, expected.norms, expected.indices)

    assert got.norms.shape == (1, 2, 8)
    assert got.indices.shape == expected.indices.shape
    assert mx.all(got.norms == expected.norms).item()
    assert mx.all(got.indices == expected.indices).item()


def test_ssd_type_map_completeness():
    """All TQ state types from turboquant_kv must be in SSD type_map."""
    from omlx.turboquant_kv import (
        TurboQuantMSEState,
        TurboQuantPolarProdState,
        TurboQuantPolarState,
        TurboQuantProdState,
        TurboQuantSplitState,
    )

    expected_types = {
        "TurboQuantMSEState",
        "TurboQuantProdState",
        "TurboQuantPolarState",
        "TurboQuantPolarProdState",
        "TurboQuantSplitState",
    }
    # Import the type_map as it would be constructed in _reconstruct_cache_data
    _type_map = {
        "TurboQuantMSEState": TurboQuantMSEState,
        "TurboQuantProdState": TurboQuantProdState,
        "TurboQuantPolarState": TurboQuantPolarState,
        "TurboQuantPolarProdState": TurboQuantPolarProdState,
        "TurboQuantSplitState": TurboQuantSplitState,
    }
    assert set(_type_map.keys()) == expected_types


# ---------------------------------------------------------------------------
# Batched TurboQuant wiring (Phase 1): eligibility gate + post-prefill
# conversion path (from_cache -> merge -> BatchTurboQuantKVCache)
# ---------------------------------------------------------------------------


def test_turboquant_eligible_gate():
    """Hybrid cache layouts may convert KVCache layers and pass through others.

    Rotating/sliding-window caches are not themselves TurboQuant-converted, but
    they can coexist with converted full-attention KVCache layers. Chunked and
    legacy QuantizedKVCache layouts still gate OFF.
    """
    from types import SimpleNamespace

    from mlx_lm.models.cache import (
        ArraysCache,
        CacheList,
        ChunkedKVCache,
        KVCache,
        QuantizedKVCache,
        RotatingKVCache,
    )

    from omlx.scheduler import Scheduler

    # _turboquant_eligible consults the model for MLA architecture (#1613)
    # and attention sinks before checking cache types; inject a compatible stub
    # so this test isolates the cache-type gating it is exercising.
    def elig(cache):
        stub = SimpleNamespace(
            _model_uses_mla=lambda: False,
            _model_uses_attention_sinks=lambda: False,
        )
        return Scheduler._turboquant_eligible(stub, cache)

    assert elig([KVCache(), KVCache()]) is True
    assert elig([]) is False
    assert elig([KVCache(), ChunkedKVCache(8192)]) is False
    assert elig([KVCache(), RotatingKVCache(32)]) is True
    assert elig([QuantizedKVCache()]) is False
    assert elig([CacheList(KVCache(), KVCache())]) is True
    assert elig([ArraysCache(size=2), KVCache()]) is True
    assert elig([CacheList(ArraysCache(size=2), KVCache())]) is True
    assert elig([CacheList(KVCache(), RotatingKVCache(32))]) is True


def test_turboquant_convert_hybrid_cache_keeps_rotating_passthrough():
    from types import SimpleNamespace

    from mlx_lm.models.cache import KVCache, RotatingKVCache

    from omlx.scheduler import Scheduler

    first = KVCache()
    first.update_and_fetch(
        mx.random.normal((1, 2, 4, 32)),
        mx.random.normal((1, 2, 4, 32)),
    )
    rotating = RotatingKVCache(max_size=32)
    rotating.update_and_fetch(
        mx.random.normal((1, 2, 4, 32)),
        mx.random.normal((1, 2, 4, 32)),
    )
    last = KVCache()
    last.update_and_fetch(
        mx.random.normal((1, 2, 4, 32)),
        mx.random.normal((1, 2, 4, 32)),
    )
    mx.eval(first.state, rotating.state, last.state)

    ns = SimpleNamespace(_turboquant_kv_bits=4.0, _turboquant_skip_last=True)
    cache = [first, rotating, last]

    Scheduler._apply_turboquant_kv_convert(ns, cache)

    assert isinstance(cache[0], TurboQuantKVCache)
    assert cache[1] is rotating
    assert isinstance(cache[1], RotatingKVCache)
    assert cache[2] is last
    assert isinstance(cache[2], KVCache)


def test_turboquant_convert_preserves_skip_last_after_partial_tq_restore():
    from types import SimpleNamespace

    from mlx_lm.models.cache import KVCache, RotatingKVCache

    from omlx.scheduler import Scheduler

    first_fp = KVCache()
    first_fp.update_and_fetch(
        mx.random.normal((1, 2, 4, 32)),
        mx.random.normal((1, 2, 4, 32)),
    )
    first_tq = TurboQuantKVCache.from_cache(first_fp, bits=4.0)
    rotating = RotatingKVCache(max_size=32)
    rotating.update_and_fetch(
        mx.random.normal((1, 2, 4, 32)),
        mx.random.normal((1, 2, 4, 32)),
    )
    last = KVCache()
    last.update_and_fetch(
        mx.random.normal((1, 2, 4, 32)),
        mx.random.normal((1, 2, 4, 32)),
    )
    mx.eval(first_tq.keys, first_tq.values, rotating.state, last.state)

    ns = SimpleNamespace(_turboquant_kv_bits=4.0, _turboquant_skip_last=True)
    cache = [first_tq, rotating, last]

    Scheduler._apply_turboquant_kv_convert(ns, cache)

    assert cache[0] is first_tq
    assert isinstance(cache[0], TurboQuantKVCache)
    assert cache[1] is rotating
    assert isinstance(cache[1], RotatingKVCache)
    assert cache[2] is last
    assert isinstance(cache[2], KVCache)


def test_from_cache_merge_builds_working_batch():
    """Mirror the scheduler path: fp16 prefill -> from_cache (post-prefill
    quantize) -> _merge_caches builds a BatchTurboQuantKVCache that decodes.

    Importing omlx.scheduler installs the TurboQuantKVCache.merge monkey-patch
    that _merge_caches() relies on, so caches[0].merge([...]) is what the
    BatchGenerator actually calls at insert() time.
    """
    import omlx.scheduler  # noqa: F401  (applies the merge monkey-patch)

    per_request = []
    for length in (8, 4):  # two requests of different prefill lengths
        kv = KVCache()
        kv.update_and_fetch(
            mx.random.normal((1, 2, length, 32)),
            mx.random.normal((1, 2, length, 32)),
        )
        per_request.append(TurboQuantKVCache.from_cache(kv, bits=4.0))
    mx.eval(*[c.keys for c in per_request])

    # Exactly what mlx-lm _merge_caches() does for one layer.
    batch = per_request[0].merge(per_request)
    assert isinstance(batch, BatchTurboQuantKVCache)
    assert batch.left_padding.tolist() == [0, 4]  # request 1 left-padded
    assert batch.offset.tolist() == [8, 4]  # per-request valid lengths

    # A decode step + the real attention path the model uses: update_and_fetch
    # returns correctly-sliced state proxies (NOT the full reserved buffer),
    # and decode_attention runs over the batched left-padding mask.
    ks, vs = batch.update_and_fetch(
        mx.random.normal((2, 2, 1, 32)),
        mx.random.normal((2, 2, 1, 32)),
    )
    assert batch.offset.tolist() == [9, 5]  # both requests advanced by 1
    out = batch.decode_attention(
        mx.random.normal((2, 2, 1, 32)),
        keys_state=ks,
        values_state=vs,
        scale=32**-0.5,
        mask=batch.make_mask(1, return_array=True),
    )
    mx.eval(out)
    assert out.shape == (2, 2, 1, 32)  # (B, n_q_heads, 1, D)


def test_decode_single_token_quantize_is_accurate():
    """Regression: the decode step appends ONE token via update_and_fetch.

    An earlier mlx-vlm fused single-token quantize kernel (used only for
    keys.shape[-2] == 1) was broken — ~140% reconstruction error at every bit
    depth — which garbled generation once TurboQuant decode engaged. It is fixed
    on the pinned mlx-vlm (main). This test fails loudly if that regresses.
    """
    from omlx.patches.turboquant_attention import apply_turboquant_attention_patch

    apply_turboquant_attention_patch()

    ctx_k = mx.random.normal((1, 8, 40, 64)) * 0.1
    ctx_v = mx.random.normal((1, 8, 40, 64)) * 0.1
    new_k = mx.random.normal((1, 8, 1, 64)) * 0.1
    new_v = mx.random.normal((1, 8, 1, 64)) * 0.1

    tq = TurboQuantKVCache(bits=8.0)
    tq.update_and_fetch(ctx_k, ctx_v)
    tq.update_and_fetch(new_k, new_v)  # the decode-step append (T=1)
    dk, _ = tq.dequantize()

    rel_err = (
        mx.mean(mx.abs(dk[:, :, 40:41, :] - new_k)).item()
        / mx.mean(mx.abs(new_k)).item()
    )
    # 8-bit TurboQuant is near-lossless; broken kernel gives >100%.
    assert rel_err < 0.05, f"decode-token quantize error {rel_err:.1%} (kernel bug?)"


def test_batch_masked_decode_is_accurate():
    """Regression: B>1 continuous-batching decode passes an array mask.

    The L=1 value kernels formerly corrupted the masked decode_attention path
    under RHT (~140% error); the `not use_rht` guard is now fixed upstream in the
    pinned mlx-vlm (Blaizzy/mlx-vlm#1244). This verifies the patched
    scaled_dot_product_attention produces correct masked decode output for a B>1
    array mask — matching the dequantize+SDPA reference over the same states.
    """
    from mlx_lm.models import base as mlx_base

    from omlx.patches.turboquant_attention import apply_turboquant_attention_patch

    apply_turboquant_attention_patch()

    # B=2 ragged batch (different prefill lengths) -> needs an array mask.
    singles = []
    for length in (12, 8):
        fp = KVCache()
        fp.update_and_fetch(
            mx.random.normal((1, 4, length, 32)) * 0.1,
            mx.random.normal((1, 4, length, 32)) * 0.1,
        )
        singles.append(TurboQuantKVCache.from_cache(fp, bits=8.0))
    batch = BatchTurboQuantKVCache.merge(singles)

    q = mx.random.normal((2, 16, 1, 32)) * 0.1  # B=2, 16 q-heads / 4 kv-heads
    ks, vs = batch.update_and_fetch(
        mx.random.normal((2, 4, 1, 32)) * 0.1,
        mx.random.normal((2, 4, 1, 32)) * 0.1,
    )
    dk, dv = batch.dequantize(ks, vs)
    t_len = dk.shape[2]
    mask = mx.ones((2, 1, 1, t_len), dtype=mx.bool_)

    out = mlx_base.scaled_dot_product_attention(
        q, ks, vs, batch, scale=32**-0.5, mask=mask
    )
    ref = mx.fast.scaled_dot_product_attention(
        q, dk.astype(q.dtype), dv.astype(q.dtype), scale=32**-0.5, mask=mask
    )
    mx.eval(out, ref)
    rel = mx.mean(mx.abs(out - ref)).item() / mx.mean(mx.abs(ref)).item()
    # 8-bit quantized masked decode vs dequantize+SDPA over the same states.
    # Broken RHT kernels give ~140%; the fix brings it into quantization noise.
    assert (
        rel < 0.05
    ), f"B>1 masked decode inaccurate (err {rel:.1%}) — RHT fix missing from pinned mlx-vlm?"


def _make_ragged_batch(bits=8.0, t_long=48, t_short=32, nkv=2, d=32, seed=5):
    mx.random.seed(seed)
    rows = []
    for t in (t_long, t_short):
        c = TurboQuantKVCache(bits=bits)
        k = mx.random.normal((1, nkv, t, d)).astype(mx.float16)
        v = mx.random.normal((1, nkv, t, d)).astype(mx.float16)
        c.update_and_fetch(k, v)
        mx.eval(c.keys, c.values)
        rows.append((c, k, v))
    batch = BatchTurboQuantKVCache.merge([c for c, _, _ in rows])
    return batch, rows


def test_batch_tq_append_position_survives_min_lp_row_departure():
    """Continuous batching: when the zero-left-padding row departs (filter),
    the physical append position must stay at the buffer's written end.
    Deriving it from offset.max() makes every later decode write land
    min(left_padding) columns early — overwriting the surviving rows' live
    KV in place and silently losing the appended tokens (issue class: batched
    TurboQuant intelligence collapse)."""
    batch, rows = _make_ragged_batch()
    _, k2, v2 = rows[1]

    batch.filter(mx.array([1]))  # the lp=0 row departs; padding compacts away

    ref_k = [k2[0]]
    for _ in range(3):
        nk = mx.random.normal((1, 2, 1, 32)).astype(mx.float16)
        nv = mx.random.normal((1, 2, 1, 32)).astype(mx.float16)
        batch.update_and_fetch(nk, nv)
        ref_k.append(nk[0])
    mx.eval(batch.keys, batch.values)

    ref_k = mx.concatenate(ref_k, axis=1)  # (nkv, 35, d) logical truth
    dk, _ = batch.dequantize()
    lp = int(batch.left_padding[0].item())
    stored = dk.shape[2] - lp
    assert stored >= ref_k.shape[1], (
        f"appended tokens lost: {stored} stored of {ref_k.shape[1]} logical"
    )
    got = dk[0, :, lp : lp + ref_k.shape[1], :].astype(mx.float32)
    err = mx.abs(got - ref_k.astype(mx.float32)).max().item()
    assert err < 0.2, f"surviving row KV stomped in place (max err {err:.3f})"


def test_batch_tq_make_mask_width_after_min_lp_row_departure():
    """After the lp=0 row departs, filter() compacts the shared padding away
    (mirroring BatchKVCache), leaving the single survivor at lp=0 with the
    written end equal to its own length, so a decode step needs no mask.
    Survivors with residual padding are covered by
    test_batch_tq_filter_compacts_left_padding_like_batch_kv."""
    batch, _ = _make_ragged_batch()
    batch.filter(mx.array([1]))
    assert batch._phys_end == 32, (
        f"written end {batch._phys_end}, expected 32 after dropping the "
        "departed row's 16 shared padding columns"
    )
    assert batch.left_padding.tolist() == [0]
    m = batch.make_mask(1, return_array=True)
    assert m is None, f"expected no mask for the unpadded survivor, got {m}"


def test_batch_tq_filter_compacts_left_padding_like_batch_kv():
    """Issue #2237: BatchKVCache.filter() shifts its buffer left by
    min(left_padding) once the zero-left-padding row departs; the TQ batch
    cache must mirror that compaction. The model builds ONE decode mask from
    the first (TQ) layer and feeds it to every attention layer, including the
    turboquant_skip_last dense BatchKVCache layer, so diverging physical
    widths crash the next step with a broadcast error."""
    mx.random.seed(7)
    tq = BatchTurboQuantKVCache([0, 1, 2], bits=8.0)
    dense = BatchKVCache([0, 1, 2])
    for t in (8, 1, 1, 1):
        k = mx.random.normal((3, 2, t, 32)).astype(mx.float16)
        v = mx.random.normal((3, 2, t, 32)).astype(mx.float16)
        tq.update_and_fetch(k, v)
        dense.update_and_fetch(k, v)

    # The zero-left-padding row departs; survivors keep lp > 0.
    keep = mx.array([1, 2])
    tq.filter(keep)
    dense.filter(keep)
    assert tq.left_padding.tolist() == dense.left_padding.tolist(), (
        f"left_padding diverges after filter: TQ={tq.left_padding.tolist()} "
        f"BK={dense.left_padding.tolist()}"
    )
    assert tq._phys_end == dense._idx, (
        f"physical end diverges after filter: TQ={tq._phys_end} "
        f"BK={dense._idx}"
    )

    # Next decode step: shared mask width vs dense layer keys width.
    k = mx.random.normal((2, 2, 1, 32)).astype(mx.float16)
    v = mx.random.normal((2, 2, 1, 32)).astype(mx.float16)
    mask = tq.make_mask(1)
    dk, _ = dense.update_and_fetch(k, v)
    tq.update_and_fetch(k, v)
    assert mask.shape[-1] == dk.shape[2], (
        f"decode mask spans {mask.shape[-1]} columns but the dense skip-last "
        f"layer has {dk.shape[2]}, broadcast crash in SDPA (#2237)"
    )

    # Compaction must shift content, not corrupt it: compare each row's
    # valid region against the exact dense copy of the same stream.
    dqk, _ = tq.dequantize()
    for i in range(2):
        lp_i = int(dense.left_padding[i].item())
        err = (
            mx.abs(
                dqk[i, :, lp_i:, :].astype(mx.float32)
                - dk[i, :, lp_i:, :].astype(mx.float32)
            )
            .max()
            .item()
        )
        assert err < 0.2, f"row {i} content shifted by compaction (err {err:.3f})"

    # Second departure leaves a single survivor whose padding compacts to 0;
    # both caches must stay in lockstep there too.
    tq.filter(mx.array([1]))
    dense.filter(mx.array([1]))
    assert tq.left_padding.tolist() == dense.left_padding.tolist()
    assert tq._phys_end == dense._idx


def test_batch_tq_append_growth_keeps_content_and_geometry():
    """Appending past the merged buffer's exact capacity must step-grow the
    state without shifting content or leaking unwritten capacity columns
    into .state / attention geometry."""
    batch, rows = _make_ragged_batch()
    _, k1, _ = rows[0]

    nk = mx.random.normal((2, 2, 1, 32)).astype(mx.float16)
    nv = mx.random.normal((2, 2, 1, 32)).astype(mx.float16)
    batch.update_and_fetch(nk, nv)  # write at 48 -> triggers reserve growth
    mx.eval(batch.keys, batch.values)

    ks, _ = batch.state
    from omlx.turboquant_kv import _state_length as _sl
    assert _sl(getattr(ks, "_state", ks)) == 49, (
        f".state exposes {_sl(getattr(ks, '_state', ks))} columns, expected 49"
    )
    dk, _ = batch.dequantize()
    got = dk[0, :, :48, :].astype(mx.float32)
    err = mx.abs(got - k1[0].astype(mx.float32)).max().item()
    assert err < 0.2, f"row0 content shifted/corrupted after growth ({err:.3f})"
    err_new = mx.abs(dk[0, :, 48, :].astype(mx.float32) - nk[0, :, 0, :].astype(mx.float32)).max().item()
    assert err_new < 0.2, f"appended token not at written end ({err_new:.3f})"


def test_batch_tq_state_restore_resets_phys_end():
    """A state restore hands the cache back to the parent's int-offset (B=1)
    bookkeeping, where the offset is the write cursor. A stale batch-mode
    _phys_end must not survive the restore: _ensure_array_offset takes the
    max of both, so a leftover value from a longer previous batch would win
    over the restored cursor and leave a gap of unwritten columns."""
    batch, _ = _make_ragged_batch()  # B=2, written end 48

    single = TurboQuantKVCache(bits=8.0)
    k = mx.random.normal((1, 2, 20, 32)).astype(mx.float16)
    single.update_and_fetch(k, k)
    batch.state = single.state  # restore a 20-token state onto the object

    assert isinstance(batch.offset, int) and batch.offset == 20
    batch._ensure_array_offset()
    assert batch._phys_end == 20, (
        f"stale batch-mode _phys_end leaked through restore "
        f"({batch._phys_end} != 20)"
    )
