"""Affine4 numerics, native dispatch, and packed cache lifecycle."""

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import KVCache
from mlx_vlm.turboquant import TurboQuantKVCache, TurboQuantMSEState

import omlx.affine4 as affine4
from omlx.affine4 import Affine4Codec, Affine4KVCache, BatchAffine4KVCache


def random(shape, dtype=mx.float16, seed=0):
    return mx.random.normal(shape, key=mx.random.key(seed)).astype(dtype)


def close(actual, expected, atol=2e-3, rtol=3e-3):
    mx.eval(actual, expected)
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        atol=atol,
        rtol=rtol,
    )


def dense(cache, queries, mask=None, sinks=None, states=None):
    keys, values = cache.dequantize(*(states or (None, None)))
    return mx.fast.scaled_dot_product_attention(
        queries.astype(mx.float32),
        keys,
        values,
        scale=queries.shape[-1] ** -0.5,
        mask=mask,
        sinks=None if sinks is None else sinks.astype(mx.float32),
    ).astype(queries.dtype)


@pytest.mark.parametrize("dim", [1, 3, 7, 32, 63, 64, 80, 96, 128, 256, 512])
def test_codec_orthogonal_and_quantization(dim):
    codec = Affine4Codec(dim, seed=17)
    vectors = random((2, 3, 5, dim), mx.float32)
    if codec.rotation is not None:
        rotation = np.array(codec.rotation)
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(dim), atol=5e-7)
    close(codec._rotate_inverse(codec._rotate_forward(vectors)), vectors, 8e-3, 3e-3)
    state = codec.quantize(vectors)
    assert isinstance(state, TurboQuantMSEState)
    assert state.indices.shape == (2, 3, 5, (dim + 7) // 8)
    assert state.indices.dtype == mx.uint32
    assert state.norms.dtype == mx.float32
    error = mx.mean(mx.square(codec.dequantize(state) - vectors)).item()
    assert error < 0.03
    other = Affine4Codec(dim, seed=17).quantize(vectors)
    assert mx.array_equal(state.indices, other.indices).item()


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("magnitude", [0, 1e-8, 1000, 1e6])
def test_finite_scale_range(dtype, magnitude):
    if dtype == mx.float16 and magnitude > 65504:
        pytest.skip("Magnitude exceeds the input dtype")
    vectors = mx.full((1, 2, 5, 64), magnitude, dtype=dtype)
    cache = Affine4KVCache(seed=4)
    cache.update_and_fetch(vectors, vectors)
    decoded, _ = cache.dequantize()
    assert mx.all(mx.isfinite(decoded)).item()
    if mx.max(mx.abs(vectors)).item():
        assert mx.max(mx.abs(decoded)).item() > magnitude / 2
    else:
        assert mx.all(decoded == 0).item()


@pytest.mark.parametrize("bits", [3, 3.5, 4.00001, 8, float("nan")])
def test_reject_non_four_bits(bits):
    with pytest.raises(ValueError, match="exactly 4"):
        Affine4KVCache(bits)
    with pytest.raises(ValueError, match="exactly 4"):
        BatchAffine4KVCache([0], bits)


@pytest.mark.parametrize("dim", [0, -1, 2.5])
def test_reject_invalid_dimension(dim):
    with pytest.raises(ValueError):
        Affine4Codec(dim)


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 1, 32), (2, 3, 7, 64), (1, 2, 9, 128), (2, 1, 3, 256), (1, 1, 2, 512)],
)
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_actual_fused_quantize_matches_portable(shape, dtype):
    if not mx.metal.is_available():
        pytest.skip("Fused quantization requires Metal")
    cache = Affine4KVCache(seed=13)
    keys, values = random(shape, dtype), random(shape, dtype, 1)
    cache._ensure_codecs(keys, values)
    ks, vs = cache._try_fused_kv_quantize(keys, values)
    assert ks is not None, "Eligible fused quantization unexpectedly fell back"
    for codec, source, state in (
        (cache.key_codec, keys, ks),
        (cache.value_codec, values, vs),
    ):
        reference = codec.quantize(source)
        close(state.norms, reference.norms, 1e-6, 2e-6)
        close(codec.dequantize(state), codec.dequantize(reference), 2e-5, 2e-5)


@pytest.mark.parametrize(
    "shape", [(1, 2, 3, 63, 79), (2, 2, 4, 80, 96), (2, 3, 17, 64, 64)]
)
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize(
    "mask_kind",
    ["none", "causal", "head", "broadcast", "false", "infinity", "additive"],
)
def test_portable_attention_masks_sinks_and_dimensions(
    monkeypatch, shape, dtype, mask_kind
):
    monkeypatch.setattr(affine4, "_m5_mpp_available", lambda: False)
    batch, heads, rows, dk, dv = shape
    tokens = 23
    cache = Affine4KVCache()
    states = cache.update_and_fetch(
        random((batch, heads, tokens, dk), dtype),
        random((batch, heads, tokens, dv), dtype, 1),
    )
    q = random((batch, heads * 3, rows, dk), dtype, 2)
    mask = None
    if mask_kind == "causal":
        mask = "causal"
    elif mask_kind == "head":
        mask = random((batch, heads * 3, rows, tokens)) > 0
    elif mask_kind == "broadcast":
        mask = mx.arange(tokens)[None, None, None, :] >= 8
    elif mask_kind == "false":
        mask = mx.zeros((rows, tokens), mx.bool_)
    elif mask_kind == "infinity":
        mask = mx.full((rows, tokens), -float("inf"))
    elif mask_kind == "additive":
        mask = random((rows, tokens), mx.float32) * 0.3
    sinks = random((heads * 3,), dtype, 8)
    for sink in (None, sinks):
        actual = cache.attention(q, *states, scale=dk**-0.5, mask=mask, sinks=sink)
        assert actual.dtype == dtype
        close(
            actual,
            dense(cache, q, mask, sink),
            atol=8e-3 if dtype == mx.bfloat16 else 3e-3,
        )


@pytest.mark.parametrize(
    "batch,heads,repeats,rows,dim,tokens",
    [
        (1, 1, 1, 1, 32, 257),
        (1, 2, 3, 1, 64, 260),
        (2, 3, 4, 2, 96, 257),
        (2, 2, 2, 4, 128, 513),
        (1, 2, 8, 4, 256, 1025),
        (1, 1, 4, 1, 512, 259),
    ],
)
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_actual_native_matches_portable(
    monkeypatch, batch, heads, repeats, rows, dim, tokens, dtype
):
    if not affine4._m5_mpp_available():
        pytest.skip("Signed-int4 attention requires M5")
    cache = Affine4KVCache()
    cache.update_and_fetch(
        random((batch, heads, tokens, dim), dtype),
        random((batch, heads, tokens, dim), dtype, 1),
    )
    q = random((batch, heads * repeats, rows, dim), dtype, 2)
    mask = "causal" if rows > 1 else None
    native = affine4._native_attention(cache, q, *cache.state, dim**-0.5, mask)
    assert native is not None, "Eligible native attention unexpectedly fell back"
    monkeypatch.setattr(affine4, "_m5_mpp_available", lambda: False)
    portable = cache.attention(q, scale=dim**-0.5, mask=mask)
    close(native, portable, atol=6e-3 if dtype == mx.bfloat16 else 2e-3)


def test_native_large_values():
    if not affine4._m5_mpp_available():
        pytest.skip("Signed-int4 attention requires M5")
    cache = Affine4KVCache()
    cache.update_and_fetch(
        mx.zeros((1, 2, 257, 64), mx.bfloat16),
        mx.full((1, 2, 257, 64), 1e6, mx.bfloat16),
    )
    q = mx.zeros((1, 8, 1, 64), mx.bfloat16)
    output = affine4._native_attention(cache, q, *cache.state, 0.125, None)
    assert output is not None
    close(output, dense(cache, q), atol=8192, rtol=0.01)


@pytest.mark.parametrize("magnitude", [60000, 1e6])
def test_native_large_queries(magnitude):
    if not affine4._m5_mpp_available():
        pytest.skip("Signed-int4 attention requires M5")
    cache = Affine4KVCache()
    cache.update_and_fetch(random((1, 2, 257, 64)), random((1, 2, 257, 64), seed=1))
    q = mx.full((1, 8, 1, 64), magnitude, mx.bfloat16)
    output = affine4._native_attention(cache, q, *cache.state, 0.125, None)
    assert output is not None
    close(output, dense(cache, q), atol=0.02)


@pytest.mark.parametrize("rows", [1, 4])
def test_native_arbitrary_boolean_mask(rows):
    if not affine4._m5_mpp_available():
        pytest.skip("Signed-int4 attention requires M5")
    cache = Affine4KVCache()
    cache.update_and_fetch(random((2, 2, 257, 64)), random((2, 2, 257, 64), seed=1))
    q = random((2, 6, rows, 64), seed=2)
    mask = random((2, 6, rows, 257), seed=3) > 0
    mask[:, 0] = False
    output = affine4._native_attention(cache, q, *cache.state, 0.125, mask)
    assert output is not None
    close(output, dense(cache, q, mask))


def test_supplied_strided_states():
    if not affine4._m5_mpp_available():
        pytest.skip("Signed-int4 attention requires M5")
    cache = Affine4KVCache()
    cache.update_and_fetch(random((1, 2, 520, 64)), random((1, 2, 520, 64), seed=1))
    states = tuple(
        TurboQuantMSEState(s.norms[:, :, ::2], s.indices[:, :, ::2, :])
        for s in cache.state
    )
    q = random((1, 8, 1, 64), seed=2)
    close(cache.attention(q, *states, scale=0.125), dense(cache, q, states=states))


def test_supplied_packed_word_stride():
    cache = Affine4KVCache()
    cache.update_and_fetch(random((1, 2, 257, 64)), random((1, 2, 257, 64), seed=1))
    states = []
    for state in cache.state:
        indices = mx.stack((state.indices, mx.zeros_like(state.indices)), axis=-1)
        indices = indices.reshape(*state.indices.shape[:-1], -1)[..., ::2]
        states.append(TurboQuantMSEState(state.norms, indices))
    q = random((1, 8, 1, 64), seed=2)
    close(cache.attention(q, *states, scale=0.125), dense(cache, q, states=states))
    restored = Affine4KVCache.from_state(tuple(states), cache.meta_state)
    close(restored.attention(q, scale=0.125), dense(cache, q, states=states))


def test_snapshot_metadata_roundtrip(tmp_path):
    cache = Affine4KVCache(seed=23)
    cache.update_and_fetch(random((1, 2, 13, 63)), random((1, 2, 13, 79), seed=1))
    ks, vs = cache.state
    path = str(tmp_path / "cache.safetensors")
    mx.save_safetensors(
        path,
        dict(ks=ks.norms, ki=ks.indices, vs=vs.norms, vi=vs.indices),
        {"meta": ",".join(cache.meta_state)},
    )
    arrays, metadata = mx.load(path, return_metadata=True)
    restored = Affine4KVCache()
    restored.state = (
        TurboQuantMSEState(arrays["ks"], arrays["ki"]),
        TurboQuantMSEState(arrays["vs"], arrays["vi"]),
    )
    restored.meta_state = tuple(metadata["meta"].split(","))
    restored.rebuild_codecs(*restored.state)
    assert restored.meta_state == ("13", "4.0", "23", "affine4", "63", "79")
    q = random((1, 6, 1, 63))
    close(restored.attention(q), cache.attention(q), 0, 0)
    restored.update_and_fetch(random((1, 2, 1, 63)), random((1, 2, 1, 79)))
    assert restored.offset == 14


def test_missing_and_invalid_metadata():
    cache = Affine4KVCache()
    cache.update_and_fetch(random((1, 2, 3, 63)), random((1, 2, 3, 79)))
    restored = Affine4KVCache()
    with pytest.raises(ValueError, match="missing"):
        restored.rebuild_codecs(*cache.state)
    for meta in [
        ("3", "4", "0"),
        ("3", "4", "0", "turboquant", "63", "79"),
        ("3", "4", "0", "affine4", "0", "0"),
    ]:
        with pytest.raises(ValueError):
            restored.meta_state = meta
    restored.meta_state = ("3", "4", "0", "affine4", "65", "79")
    with pytest.raises(ValueError, match="metadata"):
        restored.rebuild_codecs(*cache.state)


def test_from_cache_trim_and_batch_lifecycle():
    singles = []
    for length in (3, 7, 0):
        raw = KVCache()
        if length:
            raw.update_and_fetch(random((1, 2, length, 63)), random((1, 2, length, 79)))
        singles.append(Affine4KVCache.from_cache(raw, seed=11))
    clone = Affine4KVCache.from_cache(singles[1], seed=11)
    assert clone.trim(2) == 2
    assert singles[1].offset == 7
    batch = singles[0].merge(singles)
    assert type(batch) is BatchAffine4KVCache
    for i, single in enumerate(singles):
        extracted = batch.extract(i)
        assert type(extracted) is Affine4KVCache
        assert extracted.offset == single.offset
        if single.offset:
            close(extracted.dequantize()[0], single.dequantize()[0], 0, 0)
    batch.filter(mx.array([0, 2]))
    batch.update_and_fetch(random((2, 2, 1, 63)), random((2, 2, 1, 79)))
    assert batch.extract(0).offset == 4
    batch.extend(BatchAffine4KVCache.merge([clone]))
    assert batch.extract(2).offset == 5


@pytest.mark.parametrize("rows", [1, 3])
def test_batch_native_standard_mask_and_padding(monkeypatch, rows):
    if not affine4._m5_mpp_available():
        pytest.skip("Signed-int4 attention requires M5")
    cache = BatchAffine4KVCache([37, 0])
    cache.update_and_fetch(random((2, 2, 260, 64)), random((2, 2, 260, 64), seed=1))
    mask = cache.make_mask(rows)
    assert mask is None or isinstance(mask, mx.array) or mask == "causal"
    cache.update_and_fetch(random((2, 2, rows, 64)), random((2, 2, rows, 64), seed=3))
    q = random((2, 6, rows, 64), seed=2)
    native = affine4._native_attention(cache, q, *cache.state, 0.125, mask)
    assert native is not None
    monkeypatch.setattr(affine4, "_m5_mpp_available", lambda: False)
    close(native, cache.attention(q, scale=0.125, mask=mask))
    arbitrary = mask & (mx.arange(260 + rows) % 3 != 0)
    close(cache.attention(q, scale=0.125, mask=arbitrary), dense(cache, q, arbitrary))


def test_scheme_merge_guard():
    with pytest.raises(ValueError, match="config"):
        Affine4KVCache.merge([Affine4KVCache(), TurboQuantKVCache(bits=4)])


def test_matching_packed_width_does_not_allow_different_dimensions():
    caches = [Affine4KVCache(), Affine4KVCache()]
    for cache, dim in zip(caches, (63, 64)):
        cache.update_and_fetch(random((1, 2, 3, dim)), random((1, 2, 3, 79)))
    with pytest.raises(ValueError, match="dimensions"):
        Affine4KVCache.merge(caches)
    batches = [BatchAffine4KVCache.merge([cache]) for cache in caches]
    with pytest.raises(ValueError, match="dimensions"):
        batches[0].extend(batches[1])


def test_batch_state_and_metadata_restore():
    cache = BatchAffine4KVCache([3, 0], seed=21)
    cache.update_and_fetch(random((2, 2, 9, 63)), random((2, 2, 9, 79)))
    restored = BatchAffine4KVCache([3, 0])
    restored.state = cache.state
    restored.meta_state = cache.meta_state
    restored.rebuild_codecs(*restored.state)
    q = random((2, 6, 1, 63))
    close(restored.attention(q), cache.attention(q), 0, 0)
    restored.update_and_fetch(random((2, 2, 1, 63)), random((2, 2, 1, 79)))
    assert restored.extract(0).offset == 7
    assert restored.extract(1).offset == 10


@pytest.mark.parametrize("which", ["attention", "quantize"])
def test_first_native_deferred_failure_recovers(monkeypatch, which):
    if not affine4._m5_mpp_available():
        pytest.skip("Native lazy failures require Metal")
    cache = Affine4KVCache()
    keys, values = random((1, 2, 257, 64)), random((1, 2, 257, 64), seed=1)
    cache.update_and_fetch(keys, values)
    if which == "attention":
        monkeypatch.setattr(affine4, "_NATIVE_LAUNCHABLE", {})
        kernel = mx.fast.metal_kernel(
            name="affine4_bad_attention",
            input_names=["x"],
            output_names=["y"],
            source="undefined_symbol;",
        )

        def deferred(*args):
            def launch(**kwargs):
                return tuple(
                    kernel(
                        inputs=[keys],
                        output_shapes=[shape],
                        output_dtypes=[mx.float32],
                        grid=(1, 1, 1),
                        threadgroup=(1, 1, 1),
                    )[0]
                    for shape in kwargs["output_shapes"]
                )

            return launch

        monkeypatch.setattr(affine4, "_mpp_attention_kernel", deferred)
        q = random((1, 8, 1, 64), seed=2)
        close(cache.attention(q, scale=0.125), dense(cache, q))
        assert False in affine4._NATIVE_LAUNCHABLE.values()
    else:
        monkeypatch.setattr(affine4, "_FUSED_QUANTIZE_LAUNCHABLE", {})
        kernel = mx.fast.metal_kernel(
            name="affine4_bad_quantize",
            input_names=["x"],
            output_names=["y"],
            source="undefined_symbol;",
        )

        def deferred(dim):
            def launch(**kwargs):
                return tuple(
                    kernel(
                        inputs=[keys],
                        output_shapes=[shape],
                        output_dtypes=[dtype],
                        grid=(1, 1, 1),
                        threadgroup=(1, 1, 1),
                    )[0]
                    for shape, dtype in zip(
                        kwargs["output_shapes"], kwargs["output_dtypes"]
                    )
                )

            return launch

        monkeypatch.setattr(affine4, "_fused_quantize_kernel", deferred)
        cache.update_and_fetch(keys[:, :, :1], values[:, :, :1])
        assert cache.offset == 258
        assert mx.all(mx.isfinite(cache.dequantize()[0])).item()
        assert False in affine4._FUSED_QUANTIZE_LAUNCHABLE.values()


def test_warm_execution_errors_propagate_at_evaluation(monkeypatch):
    """Warm native calls stay asynchronous after successful JIT validation."""
    if not affine4._m5_mpp_available():
        pytest.skip("Native lazy failures require Metal")
    cache = Affine4KVCache()
    keys = random((1, 2, 257, 64))
    cache.update_and_fetch(keys, keys)
    q = random((1, 8, 1, 64), seed=2)
    mx.eval(cache.attention(q, scale=0.125))
    kernel = mx.fast.metal_kernel(
        name="affine4_bad_warm",
        input_names=["x"],
        output_names=["y"],
        source="undefined_symbol;",
    )

    def deferred(*args):
        def launch(**kwargs):
            return tuple(
                kernel(
                    inputs=[keys],
                    output_shapes=[shape],
                    output_dtypes=[mx.float32],
                    grid=(1, 1, 1),
                    threadgroup=(1, 1, 1),
                )[0]
                for shape in kwargs["output_shapes"]
            )

        return launch

    monkeypatch.setattr(affine4, "_mpp_attention_kernel", deferred)
    output = cache.attention(q, scale=0.125)
    with pytest.raises(RuntimeError, match="metal library"):
        mx.eval(output)


@pytest.mark.parametrize("dim", [1, 7, 63, 80, 128, 257])
@pytest.mark.parametrize("shape", [(), (3,), (2, 3, 9)])
def test_fused_rotated_dequantize_arbitrary_dimensions(dim, shape):
    if not mx.metal.is_available():
        pytest.skip("Fused unpack requires Metal")
    codec = Affine4Codec(dim)
    state = codec.quantize(random((*shape, dim), mx.float32))
    reference = codec._codes(state) * state.norms[..., None]
    close(codec.dequantize_rotated(state), reference, 0, 0)
    assert affine4._DEQUANTIZE_LAUNCHABLE[(dim, max(1, len(shape)), mx.float32)] is True


def test_fused_rotated_dequantize_strides():
    codec = Affine4Codec(79)
    state = codec.quantize(random((2, 3, 19, 79)))
    indices = mx.stack((state.indices, mx.zeros_like(state.indices)), axis=-1)
    indices = indices.reshape(2, 3, 19, -1)[..., ::2]
    state = TurboQuantMSEState(state.norms[:, ::2, ::3], indices[:, ::2, ::3])
    close(
        codec.dequantize_rotated(state),
        codec._codes(state) * state.norms[..., None],
        0,
        0,
    )
    reversed_state = TurboQuantMSEState(
        state.norms[::-1, :, ::-1], state.indices[::-1, :, ::-1, ::-1]
    )
    close(
        codec.dequantize_rotated(reversed_state),
        codec._codes(reversed_state) * reversed_state.norms[..., None],
        0,
        0,
    )


def test_dequantize_first_deferred_failure_recovers(monkeypatch):
    if not mx.metal.is_available():
        pytest.skip("Deferred Metal failure requires Metal")
    codec = Affine4Codec(63)
    state = codec.quantize(random((1, 2, 7, 63)))
    monkeypatch.setattr(affine4, "_DEQUANTIZE_LAUNCHABLE", {})
    kernel = mx.fast.metal_kernel(
        name="affine4_bad_unpack",
        input_names=["packed", "scales"],
        output_names=["output"],
        source="undefined_symbol;",
    )
    monkeypatch.setattr(affine4, "_dequantize_rotated_kernel", lambda: kernel)
    close(
        codec.dequantize_rotated(state),
        codec._codes(state) * state.norms[..., None],
        0,
        0,
    )
    assert affine4._DEQUANTIZE_LAUNCHABLE[(63, 3, mx.float32)] is False


@pytest.mark.parametrize("device_context", ["default", "stream"])
def test_explicit_cpu_uses_portable_operations(monkeypatch, device_context):
    from contextlib import nullcontext

    def no_metal(*args, **kwargs):
        pytest.fail("CPU execution invoked a Metal kernel")

    monkeypatch.setattr(mx.fast, "metal_kernel", no_metal)
    monkeypatch.setattr(affine4, "_fused_quantize_kernel", no_metal)
    monkeypatch.setattr(affine4, "_dequantize_rotated_kernel", no_metal)
    monkeypatch.setattr(affine4, "_mpp_attention_kernel", no_metal)
    previous = mx.default_device()
    if device_context == "default":
        mx.set_default_device(mx.cpu)
    try:
        with mx.stream(mx.cpu) if device_context == "stream" else nullcontext():
            cache = Affine4KVCache()
            cache.update_and_fetch(random((1, 2, 257, 64)), random((1, 2, 257, 79)))
            queries = random((1, 6, 1, 64))
            close(cache.attention(queries, scale=0.125), dense(cache, queries))
            assert mx.all(mx.isfinite(cache.dequantize()[1])).item()
    finally:
        mx.set_default_device(previous)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize(
    "mask_kind",
    ["none", "causal", "head", "broadcast", "tokens", "false", "infinity", "additive"],
)
@pytest.mark.parametrize("with_sinks", [False, True])
def test_bounded_prefill_preserves_masks_and_sinks(
    monkeypatch, batched, mask_kind, with_sinks
):
    cache = BatchAffine4KVCache([3, 0]) if batched else Affine4KVCache()
    cache.update_and_fetch(random((2, 2, 23, 63)), random((2, 2, 23, 79)))
    q = random((2, 6, 7, 63), seed=2)
    masks = {
        "none": None,
        "causal": "causal",
        "head": random((2, 6, 7, 23), seed=3) > 0,
        "broadcast": mx.arange(23)[None, None, None, :] >= 8,
        "tokens": mx.arange(23) >= 8,
        "false": mx.zeros((7, 23), mx.bool_),
        "infinity": mx.full((7, 23), -float("inf")),
        "additive": random((2, 6, 7, 23), mx.float32, seed=4),
    }
    mask = masks[mask_kind]
    sinks = random((6,), mx.float32, seed=5) if with_sinks else None
    expected = cache.attention(q, scale=63**-0.5, mask=mask, sinks=sinks)
    mx.eval(expected)
    monkeypatch.setattr(affine4, "_MAX_SCORE_ELEMENTS", 2 * 6 * 23 * 3)
    original = mx.fast.scaled_dot_product_attention
    query_lengths = []

    def record(queries, *args, **kwargs):
        query_lengths.append(queries.shape[-2])
        return original(queries, *args, **kwargs)

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", record)
    close(cache.attention(q, scale=63**-0.5, mask=mask, sinks=sinks), expected, 3e-3)
    assert query_lengths == [3, 3, 1]


def test_bounded_prefill_peak_memory():
    if not mx.metal.is_available():
        pytest.skip("Peak allocation tracking requires Metal")
    cache = Affine4KVCache()
    cache.update_and_fetch(
        random((1, 4, 32768, 256)), random((1, 4, 32768, 256), seed=1)
    )
    queries = random((1, 24, 256, 256), seed=2)
    mx.eval(queries, cache.state)
    mx.synchronize()
    mx.clear_cache()
    baseline = mx.get_active_memory()
    mx.reset_peak_memory()
    output = cache.attention(queries, scale=256**-0.5, mask="causal")
    mx.eval(output)
    temporary = mx.get_peak_memory() - baseline
    assert temporary < 600 * 1024**2, f"Prefill allocated {temporary / 1024**2:.1f} MiB"
    assert mx.all(mx.isfinite(output)).item()
