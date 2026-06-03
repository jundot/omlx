"""Tests for TurboQuant KV cache (mlx-vlm backend + omlx BatchTurboQuantKVCache)."""

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from mlx_vlm.turboquant import (
    TurboQuantKVCache,
    _TurboQuantMSECodec,
    _TurboQuantProdCodec,
    _build_codec,
    turboquant_enabled,
)

from omlx.turboquant_kv import BatchTurboQuantKVCache, _rebuild_codecs, _infer_head_dim

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
    c1.update_and_fetch(mx.random.normal((1, 2, 8, 32)), mx.random.normal((1, 2, 8, 32)))
    c2 = TurboQuantKVCache(bits=4.0)
    c2.update_and_fetch(mx.random.normal((1, 2, 4, 32)), mx.random.normal((1, 2, 4, 32)))
    mx.eval(c1.keys, c1.values, c2.keys, c2.values)

    batch = BatchTurboQuantKVCache.merge([c1, c2])
    assert batch.keys is not None
    assert batch.left_padding[0].item() == 0
    assert batch.left_padding[1].item() == 4

    e1 = batch.extract(0)
    e2 = batch.extract(1)
    assert e1.offset == 8
    assert e2.offset == 4


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


def test_batch_tq_continuous_batching_extend():
    b1 = BatchTurboQuantKVCache([0], bits=4.0)
    b1.update_and_fetch(mx.random.normal((1, 2, 8, 32)), mx.random.normal((1, 2, 8, 32)))
    b1.update_and_fetch(mx.random.normal((1, 2, 1, 32)), mx.random.normal((1, 2, 1, 32)))

    b2 = BatchTurboQuantKVCache([0], bits=4.0)
    b2.update_and_fetch(mx.random.normal((1, 2, 4, 32)), mx.random.normal((1, 2, 4, 32)))
    b2.update_and_fetch(mx.random.normal((1, 2, 1, 32)), mx.random.normal((1, 2, 1, 32)))

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

    ref = bk.make_mask(1, return_array=True)        # decode-step mask
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
    b1.update_and_fetch(mx.random.normal((1, 2, 8, 32)), mx.random.normal((1, 2, 8, 32)))

    b2 = BatchTurboQuantKVCache([0], bits=4.0)
    b2.update_and_fetch(mx.random.normal((1, 2, 4, 32)), mx.random.normal((1, 2, 4, 32)))

    b1.extend(b2)
    assert b1.keys.norms.shape[0] == 2


def test_batch_tq_dequantize():
    batch = BatchTurboQuantKVCache([0], bits=4.0)
    batch.update_and_fetch(mx.random.normal((1, 2, 8, 32)), mx.random.normal((1, 2, 8, 32)))
    batch.update_and_fetch(mx.random.normal((1, 2, 1, 32)), mx.random.normal((1, 2, 1, 32)))
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
    batch.update_and_fetch(mx.random.normal((1, 2, 4, 32)), mx.random.normal((1, 2, 4, 32)))

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


def test_ssd_type_map_completeness():
    """All TQ state types from turboquant_kv must be in SSD type_map."""
    from omlx.turboquant_kv import (
        TurboQuantMSEState,
        TurboQuantProdState,
        TurboQuantPolarState,
        TurboQuantPolarProdState,
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
    """Only dense KVCache (and CacheList of KVCache) is batch-convertible.

    Chunked/rotating/quantized caches must gate OFF so chunked-attention
    models (Llama-4) and sliding-window models stay fp16 instead of crashing
    in _merge_caches() — the #771 SIGABRT class.
    """
    from types import SimpleNamespace

    from mlx_lm.models.cache import (
        CacheList,
        ChunkedKVCache,
        KVCache,
        QuantizedKVCache,
        RotatingKVCache,
    )

    from omlx.scheduler import Scheduler

    # _turboquant_eligible consults the model for MLA architecture (#1613)
    # before checking cache types; inject a non-MLA stub so this test isolates
    # the cache-type gating it is exercising.
    def elig(cache):
        stub = SimpleNamespace(_model_uses_mla=lambda: False)
        return Scheduler._turboquant_eligible(stub, cache)

    assert elig([KVCache(), KVCache()]) is True
    assert elig([]) is False
    assert elig([KVCache(), ChunkedKVCache(8192)]) is False
    assert elig([KVCache(), RotatingKVCache(32)]) is False
    assert elig([QuantizedKVCache()]) is False
    assert elig([CacheList(KVCache(), KVCache())]) is True
    assert elig([CacheList(KVCache(), RotatingKVCache(32))]) is False


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
    assert batch.left_padding.tolist() == [0, 4]   # request 1 left-padded
    assert batch.offset.tolist() == [8, 4]         # per-request valid lengths

    # A decode step + the real attention path the model uses: update_and_fetch
    # returns correctly-sliced state proxies (NOT the full reserved buffer),
    # and decode_attention runs over the batched left-padding mask.
    ks, vs = batch.update_and_fetch(
        mx.random.normal((2, 2, 1, 32)),
        mx.random.normal((2, 2, 1, 32)),
    )
    assert batch.offset.tolist() == [9, 5]         # both requests advanced by 1
    out = batch.decode_attention(
        mx.random.normal((2, 2, 1, 32)),
        keys_state=ks,
        values_state=vs,
        scale=32**-0.5,
        mask=batch.make_mask(1, return_array=True),
    )
    mx.eval(out)
    assert out.shape == (2, 2, 1, 32)              # (B, n_q_heads, 1, D)


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

    out = mlx_base.scaled_dot_product_attention(q, ks, vs, batch, scale=32**-0.5, mask=mask)
    ref = mx.fast.scaled_dot_product_attention(
        q, dk.astype(q.dtype), dv.astype(q.dtype), scale=32**-0.5, mask=mask
    )
    mx.eval(out, ref)
    rel = mx.mean(mx.abs(out - ref)).item() / mx.mean(mx.abs(ref)).item()
    # 8-bit quantized masked decode vs dequantize+SDPA over the same states.
    # Broken RHT kernels give ~140%; the fix brings it into quantization noise.
    assert rel < 0.05, f"B>1 masked decode inaccurate (err {rel:.1%}) — RHT fix missing from pinned mlx-vlm?"
