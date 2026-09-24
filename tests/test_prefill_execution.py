"""Real MLX cache contracts for persistent cold-text prefill groups."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.cache import (
    BatchKVCache,
    KVCache,
    RotatingKVCache,
    make_prompt_cache,
)

from omlx.prefill import mlx_adapter
from omlx.prefill.execution import BatchedPrefillGroup


class RecordingModel:
    def __init__(self, model):
        self.inner = model
        self.calls = []
        self.streams = []
        self.skip_lm_head_calls = []

    def make_cache(self):
        return make_prompt_cache(self.inner)

    def __call__(self, inputs, cache, skip_lm_head=False):
        self.calls.append(tuple(inputs.shape))
        self.streams.append(mx.default_stream(mx.default_device()))
        self.skip_lm_head_calls.append(skip_lm_head)
        if skip_lm_head:
            return self.inner.model(inputs, cache=cache)
        return self.inner(inputs, cache=cache)


@pytest.fixture
def engine_stream():
    stream = mx.new_thread_local_stream(mx.default_device())
    # Cleanup must not invoke stream assertions installed by individual tests.
    synchronize = mx.synchronize
    yield stream
    synchronize(stream)


@pytest.fixture
def tiny_model(engine_stream, request):
    model_type = getattr(request, "param", "llama")
    model_module = importlib.import_module(f"mlx_lm.models.{model_type}")
    with mx.stream(engine_stream):
        model = model_module.Model(
            model_module.ModelArgs.from_dict(
                {
                    "model_type": model_type,
                    "hidden_size": 32,
                    "num_hidden_layers": 2,
                    "intermediate_size": 64,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "rms_norm_eps": 1e-5,
                    "vocab_size": 64,
                    "head_dim": 8,
                    "max_position_embeddings": 2048,
                    "rope_theta": 10000,
                    "tie_word_embeddings": True,
                }
            )
        )
        model.eval()
        mx.eval(model.parameters())
    return RecordingModel(model)


@pytest.fixture
def group_factory(tiny_model, engine_stream):
    groups = []

    def create(rows, **kwargs):
        group = BatchedPrefillGroup(tiny_model, rows, engine_stream, **kwargs)
        groups.append(group)
        return group

    yield create
    for group in groups:
        group.close()


def assert_row_matches_single(model, tokens, extracted, engine_stream):
    with mx.stream(engine_stream):
        baseline = make_prompt_cache(model.inner)
        model.inner(mx.array([tokens], dtype=mx.int32), cache=baseline)
        mx.eval([layer_cache.state for layer_cache in baseline])
        for actual, expected in zip(extracted, baseline):
            assert type(actual) is KVCache
            assert actual.offset == expected.offset == len(tokens)
            assert mx.allclose(
                actual.keys_and_values()[0], expected.keys_and_values()[0], atol=2e-5
            ).item()
            assert mx.allclose(
                actual.keys_and_values()[1], expected.keys_and_values()[1], atol=2e-5
            ).item()
        kickoff = mx.array([[37]], dtype=mx.int32)
        expected_logits = model.inner(kickoff, cache=baseline)
        actual_logits = model.inner(kickoff, cache=extracted)
        mx.eval(expected_logits, actual_logits)
        assert mx.allclose(actual_logits, expected_logits, atol=2e-5).item()


@pytest.mark.parametrize("tiny_model", ["llama", "qwen2", "qwen3"], indirect=True)
def test_different_lengths_keep_batch_cache_across_steps(
    tiny_model, engine_stream, group_factory, monkeypatch
):
    rows = [
        ("short", [1, 3, 5, 7]),
        ("medium", [2, 4, 6, 8, 10, 12]),
        ("long", [9, 8, 7, 6, 5, 4, 3, 2]),
    ]
    group = group_factory(rows)
    owned_cache = group.cache

    def unexpected_merge(*args, **kwargs):
        pytest.fail("An existing prefill group must not remerge its cache")

    monkeypatch.setattr(BatchKVCache, "merge", unexpected_merge)
    first = group.step(2)
    assert first.request_ids == ("short", "medium", "long")
    assert first.tokens_per_request == 2
    assert first.processed_tokens == 6
    assert first.completed_request_ids == ()
    assert first.elapsed_s >= 0
    assert group.cache == owned_cache
    assert group.tokens_processed == 2
    assert group.remaining_tokens == {"short": 2, "medium": 4, "long": 6}
    assert group.cache_nbytes == sum(layer.nbytes for layer in group.cache) > 0

    second = group.step(2)
    assert second.completed_request_ids == ("short",)
    assert group.cache == owned_cache
    assert group.remaining_min == 0
    with pytest.raises(ValueError, match="shortest"):
        group.step(1)

    completed = group.extract("short")
    assert group.request_ids == ("short", "medium", "long")
    group.remove(("short",))
    assert_row_matches_single(tiny_model, rows[0][1], completed, engine_stream)
    assert group.request_ids == ("medium", "long")
    assert group.cache == owned_cache
    assert group.step(2).completed_request_ids == ("medium",)
    completed = group.extract("medium")
    group.remove(("medium",))
    assert_row_matches_single(tiny_model, rows[1][1], completed, engine_stream)

    assert group.batch_size == 1
    assert group.step(2).completed_request_ids == ("long",)
    completed = group.extract("long")
    group.remove(("long",))
    assert_row_matches_single(tiny_model, rows[2][1], completed, engine_stream)
    assert tiny_model.calls == [(3, 2), (3, 2), (2, 2), (1, 2)]
    assert group.batch_size == 0
    assert group.cache_nbytes == 0
    assert not group.valid


def test_partial_row_extraction_and_middle_cancellation_preserve_survivors(
    tiny_model, engine_stream, group_factory
):
    rows = [
        ("first", [1, 2, 3, 4, 5, 6]),
        ("cancelled", [7, 8, 9, 10, 11, 12]),
        ("last", [13, 14, 15, 16, 17, 18]),
    ]
    group = group_factory(rows)
    group.step(2)
    partial = group.extract("cancelled")
    group.remove(("cancelled",))
    assert_row_matches_single(tiny_model, rows[1][1][:2], partial, engine_stream)
    assert group.request_ids == ("first", "last")
    assert group.remaining_tokens == {"first": 4, "last": 4}
    group.step(4)
    for request_id, token_ids in (rows[0], rows[2]):
        assert_row_matches_single(
            tiny_model, token_ids, group.extract(request_id), engine_stream
        )


def test_extraction_does_not_change_or_remove_owned_cache(
    tiny_model, engine_stream, group_factory
):
    rows = [("first", [1, 2, 3, 4]), ("second", [5, 6, 7, 8])]
    group = group_factory(rows)
    group.step(2)
    partial = group.extract("first")
    assert_row_matches_single(tiny_model, rows[0][1][:2], partial, engine_stream)
    assert group.request_ids == ("first", "second")
    assert group.tokens_processed == 2
    group.step(2)
    assert_row_matches_single(
        tiny_model, rows[0][1], group.extract("first"), engine_stream
    )


def test_remove_before_first_forward_and_extract_empty_cache(group_factory):
    group = group_factory([("first", [1, 2]), ("cancelled", [3, 4])])
    extracted = group.extract("cancelled")
    assert all(layer_cache.keys is None for layer_cache in extracted)
    assert all(layer_cache.offset == 0 for layer_cache in extracted)
    group.remove(("cancelled",))
    assert group.request_ids == ("first",)
    assert group.step(2).completed_request_ids == ("first",)


def test_engine_stream_owns_forward_and_cache_transforms(
    tiny_model, engine_stream, group_factory, monkeypatch
):
    observed = []
    with mx.stream(engine_stream):
        bound_stream = mx.default_stream(mx.default_device())

    def record_stream(operation):
        observed.append(operation)
        assert mx.default_stream(mx.default_device()) == bound_stream

    original_make_cache = mlx_adapter.make_prompt_cache
    original_construct = mlx_adapter._UnpaddedPrefillKVCache.__init__
    original_extract = mlx_adapter._UnpaddedPrefillKVCache.extract
    original_filter = mlx_adapter._UnpaddedPrefillKVCache.filter
    original_eval = mx.eval
    original_synchronize = mx.synchronize

    def checked_make_cache(*args, **kwargs):
        record_stream("make_cache")
        return original_make_cache(*args, **kwargs)

    def checked_construct(*args, **kwargs):
        record_stream("construct")
        return original_construct(*args, **kwargs)

    def checked_extract(*args, **kwargs):
        record_stream("extract")
        return original_extract(*args, **kwargs)

    def checked_filter(*args, **kwargs):
        record_stream("filter")
        return original_filter(*args, **kwargs)

    def checked_eval(*args, **kwargs):
        record_stream("eval")
        return original_eval(*args, **kwargs)

    def checked_synchronize(stream):
        record_stream("synchronize")
        assert stream is engine_stream
        return original_synchronize(stream)

    monkeypatch.setattr(mlx_adapter, "make_prompt_cache", checked_make_cache)
    monkeypatch.setattr(
        mlx_adapter._UnpaddedPrefillKVCache, "__init__", checked_construct
    )
    monkeypatch.setattr(mlx_adapter._UnpaddedPrefillKVCache, "extract", checked_extract)
    monkeypatch.setattr(mlx_adapter._UnpaddedPrefillKVCache, "filter", checked_filter)
    monkeypatch.setattr(mx, "eval", checked_eval)
    monkeypatch.setattr(mx, "synchronize", checked_synchronize)

    group = group_factory([("first", [1, 2, 3]), ("second", [4, 5, 6])])
    group.step(2)
    group.extract("first")
    group.remove(("first",))
    group.close()

    assert tiny_model.streams == [bound_stream]
    assert set(observed) == {
        "make_cache",
        "construct",
        "extract",
        "filter",
        "eval",
        "synchronize",
    }


def test_unpadded_causal_mask_survives_filtering_and_uses_no_host_reads(
    engine_stream, group_factory, monkeypatch
):
    group = group_factory(
        [("first", [1, 2, 3, 4]), ("middle", [5, 6, 7, 8]), ("last", [9, 10, 11, 12])]
    )
    group.step(2)
    group.remove(("middle",))
    with mx.stream(engine_stream):
        native = BatchKVCache([0, 0])
        assert isinstance(native.make_mask(2), mx.array)
        for layer_cache in group.cache:
            assert layer_cache.left_padding.tolist() == [0, 0]
            assert layer_cache.offset.tolist() == [2, 2]

    def unexpected_array_work(*args, **kwargs):
        pytest.fail("The unpadded causal-mask fast path must not inspect GPU arrays")

    with monkeypatch.context() as patcher:
        patcher.setattr(BatchKVCache, "make_mask", unexpected_array_work)
        patcher.setattr(mx.array, "item", unexpected_array_work)
        patcher.setattr(mx.array, "tolist", unexpected_array_work)
        patcher.setattr(mx, "eval", unexpected_array_work)
        patcher.setattr(mx, "synchronize", unexpected_array_work)
        patcher.setattr(mx, "arange", unexpected_array_work)
        for layer_cache in group.cache:
            assert layer_cache.make_mask(2) == "causal"
            assert layer_cache.make_mask(2, window_size=None) == "causal"
            assert layer_cache.make_mask(1) is None

    extracted = group.extract("first")
    assert all(type(layer_cache) is KVCache for layer_cache in extracted)
    group.remove(("first",))
    assert group.cache[0].make_mask(2) == "causal"


@pytest.mark.parametrize("token_count", [1, 255, 256, 257])
@pytest.mark.parametrize("trim_count", [0, 1])
def test_unpadded_cache_transforms_match_native_without_host_reads(
    engine_stream, monkeypatch, token_count, trim_count
):
    with mx.stream(engine_stream):
        fast = mlx_adapter._UnpaddedPrefillKVCache(3)
        native = BatchKVCache([0, 0, 0])
        keys = mx.arange(3 * 2 * token_count * 4).reshape(3, 2, token_count, 4)
        keys = keys.astype(mx.float32)
        values = keys + 0.5
        for layer_cache in (fast, native):
            layer_cache.update_and_fetch(keys, values)
            assert layer_cache.trim(trim_count) == trim_count
        expected_row = native.extract(2)
        native.filter([2, 0])
        native_extract = BatchKVCache.extract
        native_filter = BatchKVCache.filter

        def unexpected_host_read(*args, **kwargs):
            pytest.fail("Unpadded cache transforms must not inspect GPU scalars")

        with monkeypatch.context() as patcher:
            patcher.setattr(mx.array, "item", unexpected_host_read)
            patcher.setattr(mx.array, "tolist", unexpected_host_read)
            patcher.setattr(BatchKVCache, "extract", unexpected_host_read)
            patcher.setattr(BatchKVCache, "filter", unexpected_host_read)
            actual_row = mlx_adapter.extract_row([fast], 2, stream=engine_stream)[0]
            mlx_adapter.filter_rows([fast], [2, 0], stream=engine_stream)

        assert BatchKVCache.extract is native_extract
        assert BatchKVCache.filter is native_filter
        assert type(actual_row) is KVCache
        assert actual_row.offset == expected_row.offset == token_count - trim_count
        assert fast.size() == native.size() == actual_row.offset
        assert fast.keys.shape == native.keys.shape
        assert fast.nbytes == native.nbytes
        for actual, expected in zip(fast.state, native.state):
            assert mx.array_equal(actual, expected).item()
        for actual, expected in zip(actual_row.state, expected_row.state):
            assert mx.array_equal(actual, expected).item()

        fast.trim(token_count)
        replacement = mx.full((2, 2, 2, 4), -1, dtype=mx.float32)
        fast.update_and_fetch(replacement, replacement)
        mlx_adapter.evaluate_cache([fast], stream=engine_stream)
        for actual, expected in zip(actual_row.state, expected_row.state):
            assert mx.array_equal(actual, expected).item()


def test_native_padded_cache_still_trims_padding(engine_stream):
    with mx.stream(engine_stream):
        native = BatchKVCache([2, 1, 0])
        values = mx.arange(3 * 2 * 4 * 8).reshape(3, 2, 4, 8).astype(mx.float32)
        native.update_and_fetch(values, values)
        expected = native.extract(0)
        native.filter([0, 1])
        assert native.size() == 3
        assert native.left_padding.tolist() == [1, 0]
        assert native.offset.tolist() == [2, 3]
        actual = native.extract(0)
        assert actual.offset == expected.offset == 2
        for actual_state, expected_state in zip(actual.state, expected.state):
            assert mx.array_equal(actual_state, expected_state).item()


@pytest.mark.parametrize("token_count", [1, 3])
@pytest.mark.parametrize("processed_tokens", [0, 2])
@pytest.mark.parametrize(
    "mask_kwargs",
    [
        {"return_array": True},
        {"window_size": 0},
        {"window_size": 1},
        {"window_size": 16},
        {"return_array": True, "window_size": 3},
    ],
)
def test_explicit_mask_and_window_semantics_match_native_batch_cache(
    group_factory, engine_stream, token_count, processed_tokens, mask_kwargs
):
    group = group_factory([("first", [1, 2, 3, 4]), ("second", [5, 6, 7, 8])])
    if processed_tokens:
        group.step(processed_tokens)
    layer_cache = group.cache[0]
    with mx.stream(engine_stream):
        actual = layer_cache.make_mask(token_count, **mask_kwargs)
        expected = BatchKVCache.make_mask(layer_cache, token_count, **mask_kwargs)
        mx.eval(actual, expected)
        assert isinstance(actual, mx.array)
        expected_shape = (2, 1, token_count, processed_tokens + token_count)
        assert actual.shape == expected.shape == expected_shape
        assert mx.array_equal(actual, expected).item()


def test_extra_mask_parameters_are_not_silently_ignored(group_factory, engine_stream):
    group = group_factory([("first", [1, 2, 3, 4]), ("second", [5, 6, 7, 8])])
    with mx.stream(engine_stream):
        layer_cache = group.cache[0]
        padding = mx.array([0, 1])
        actual = layer_cache.make_mask(3, right_padding=padding)
        expected = BatchKVCache.make_mask(layer_cache, 3, right_padding=padding)
        assert mx.array_equal(actual, expected).item()
        with pytest.raises(TypeError):
            layer_cache.make_mask(3, unsupported=None)


@pytest.mark.parametrize(
    "padding_kwargs",
    [
        {"left_padding": [0, 0]},
        {"left_padding": [0, 1]},
        {"right_padding": [0, 0]},
        {"right_padding": [1, 0]},
    ],
)
def test_unpadded_cache_rejects_padding_before_any_mutation(
    group_factory, padding_kwargs
):
    group = group_factory([("first", [1, 2]), ("second", [3, 4])])
    for layer_cache in group.cache:
        with pytest.raises(ValueError, match="padding"):
            layer_cache.prepare(**padding_kwargs)
        layer_cache.prepare(lengths=[2, 2])
        layer_cache.finalize()
        assert layer_cache.offset.tolist() == [0, 0]
        assert layer_cache.left_padding.tolist() == [0, 0]
        assert layer_cache.make_mask(2) == "causal"
    group.step(2)


def test_unpadded_cache_rejects_warm_restore_merge_and_extension(
    group_factory, engine_stream
):
    group = group_factory([("first", [1, 2, 3, 4]), ("second", [5, 6, 7, 8])])
    group.step(2)
    layer_cache = group.cache[0]
    original_keys = layer_cache.keys
    with mx.stream(engine_stream):
        padded = BatchKVCache([0, 1])
        values = mx.ones((2, 2, 2, 8))
        padded.update_and_fetch(values, values)
        mx.eval(padded.state)
        with pytest.raises(ValueError, match="extension"):
            layer_cache.extend(padded)
        with pytest.raises(ValueError, match="restoration"):
            layer_cache.state = padded.state
        with pytest.raises(ValueError, match="restoration"):
            type(layer_cache).from_state(padded.state)
        with pytest.raises(ValueError, match="cold cache construction"):
            type(layer_cache).merge([group.extract("first")[0], KVCache()])
        assert layer_cache.keys is original_keys
        assert layer_cache.offset.tolist() == [2, 2]
        assert layer_cache.left_padding.tolist() == [0, 0]
        assert layer_cache.make_mask(2) == "causal"


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5, [0, 1]])
def test_unpadded_cache_requires_positive_integer_width(batch_size, engine_stream):
    with mx.stream(engine_stream):
        with pytest.raises(ValueError, match="positive batch size"):
            mlx_adapter._UnpaddedPrefillKVCache(batch_size)
        with pytest.raises(ValueError, match="positive batch size"):
            mlx_adapter.create_cold_batch_cache(
                SimpleNamespace(make_cache=lambda: [KVCache()]),
                batch_size,
                stream=engine_stream,
            )


def test_skip_lm_head_is_explicit_and_preserves_cache(
    tiny_model, engine_stream, group_factory
):
    rows = [("first", [1, 3]), ("second", [2, 4])]
    group = group_factory(rows, skip_lm_head=True)
    group.step(2)
    assert tiny_model.skip_lm_head_calls == [True]
    assert_row_matches_single(
        tiny_model, rows[0][1], group.extract("first"), engine_stream
    )


def test_forward_failure_invalidates_partially_updated_group(
    tiny_model, engine_stream, monkeypatch
):
    def fail_after_first_layer(recording_model, inputs, cache, skip_lm_head=False):
        partial = mx.ones((inputs.shape[0], 2, inputs.shape[1], 8))
        cache[0].update_and_fetch(partial, partial)
        mx.async_eval(cache[0].keys)
        raise RuntimeError("simulated second-layer failure")

    monkeypatch.setattr(RecordingModel, "__call__", fail_after_first_layer)
    group = BatchedPrefillGroup(
        tiny_model, [("first", [1, 2]), ("second", [3, 4])], engine_stream
    )
    try:
        with pytest.raises(RuntimeError, match="second-layer failure"):
            group.step(2)
        assert not group.valid
        assert group.tokens_processed == 0
        assert group.request_ids == ("first", "second")
        with pytest.raises(RuntimeError, match="invalidated"):
            group.step(1)
        with pytest.raises(RuntimeError, match="invalidated"):
            group.extract("first")
        with pytest.raises(RuntimeError, match="invalidated"):
            group.remove(("first",))
    finally:
        group.close()
    assert group.cache == ()
    assert group.request_ids == ()


def test_failed_filter_preserves_row_bookkeeping_and_invalidates_group(
    group_factory, monkeypatch
):
    group = group_factory([("first", [1, 2]), ("second", [3, 4])])
    group.step(1)

    def failed_filter(layer_cache, indices):
        raise RuntimeError("simulated filter failure")

    monkeypatch.setattr(mlx_adapter._UnpaddedPrefillKVCache, "filter", failed_filter)
    with pytest.raises(RuntimeError, match="filter failure"):
        group.remove(("first",))
    assert group.request_ids == ("first", "second")
    assert not group.valid


@pytest.mark.parametrize("error_type", [RuntimeError, MemoryError])
def test_cache_evaluation_failure_invalidates_completed_forward(
    group_factory, monkeypatch, error_type
):
    group = group_factory([("first", [1, 2]), ("second", [3, 4])])
    original_eval = mx.eval

    def failed_eval(*arrays):
        original_eval(*arrays)
        raise error_type("simulated cache evaluation failure")

    with monkeypatch.context() as patcher:
        patcher.setattr(mx, "eval", failed_eval)
        with pytest.raises(error_type, match="cache evaluation failure"):
            group.step(1)

    assert not group.valid
    assert group.tokens_processed == 0
    assert all(layer_cache.offset.tolist() == [1, 1] for layer_cache in group.cache)
    with pytest.raises(RuntimeError, match="invalidated"):
        group.extract("first")
    group.close()
    assert group.cache == ()
    assert group.request_ids == ()


def test_late_filter_failure_never_exposes_partially_filtered_rows(
    group_factory, monkeypatch
):
    group = group_factory([("first", [1, 2]), ("cancelled", [3, 4]), ("last", [5, 6])])
    group.step(1)
    original_filter = mlx_adapter._UnpaddedPrefillKVCache.filter
    filtered_layers = []

    def failed_filter(layer_cache, indices):
        if filtered_layers:
            raise MemoryError("simulated second-layer filter failure")
        original_filter(layer_cache, indices)
        mx.eval(layer_cache.state)
        filtered_layers.append(layer_cache)

    monkeypatch.setattr(mlx_adapter._UnpaddedPrefillKVCache, "filter", failed_filter)
    with pytest.raises(MemoryError, match="second-layer filter failure"):
        group.remove(("cancelled",))

    assert group.cache[0].keys.shape[0] == 2
    assert group.cache[1].keys.shape[0] == 3
    assert group.request_ids == ("first", "cancelled", "last")
    assert not group.valid
    with pytest.raises(RuntimeError, match="invalidated"):
        group.extract("last")
    with pytest.raises(RuntimeError, match="invalidated"):
        group.step(1)
    group.close()
    assert group.cache == ()
    assert group.request_ids == ()


def test_failed_extraction_preserves_group_for_retry(group_factory, monkeypatch):
    group = group_factory([("first", [1, 2]), ("second", [3, 4])])
    group.step(1)

    def failed_extract(layer_cache, row_index):
        raise MemoryError("simulated handoff allocation failure")

    with monkeypatch.context() as patcher:
        patcher.setattr(mlx_adapter._UnpaddedPrefillKVCache, "extract", failed_extract)
        with pytest.raises(MemoryError, match="handoff allocation"):
            group.extract("first")
    assert group.valid
    assert group.request_ids == ("first", "second")
    assert group.extract("first")[0].offset == 1


@pytest.mark.parametrize("chunk_tokens", [0, -1, 3, True, 1.5])
def test_invalid_chunk_does_not_change_group(group_factory, chunk_tokens):
    group = group_factory([("first", [1, 2]), ("second", [3, 4])])
    with pytest.raises(ValueError):
        group.step(chunk_tokens)
    assert group.valid
    assert group.tokens_processed == 0
    assert group.cache_nbytes == 0


@pytest.mark.parametrize(
    "rows",
    [[], [("only", [1])], [("same", [1]), ("same", [2])], [("empty", []), ("ok", [1])]],
)
def test_invalid_rows_fail_before_cache_creation(tiny_model, engine_stream, rows):
    with pytest.raises(ValueError):
        BatchedPrefillGroup(tiny_model, rows, engine_stream)


def test_unknown_row_does_not_change_group(group_factory):
    group = group_factory([("first", [1, 2]), ("second", [3, 4])])
    with pytest.raises(KeyError):
        group.extract("missing")
    with pytest.raises(KeyError):
        group.remove(("first", "missing"))
    group.remove(())
    assert group.request_ids == ("first", "second")
    assert group.valid


@pytest.mark.parametrize("cache_kind", ["rotating", "subclass", "warm", "empty"])
def test_rejects_unsupported_model_caches(engine_stream, cache_kind):
    class DerivedKVCache(KVCache):
        pass

    def make_cache():
        if cache_kind == "empty":
            return []
        if cache_kind == "rotating":
            return [RotatingKVCache(max_size=16)]
        if cache_kind == "subclass":
            return [DerivedKVCache()]
        cache = KVCache()
        cache.update_and_fetch(mx.ones((1, 1, 1, 8)), mx.ones((1, 1, 1, 8)))
        return [cache]

    with pytest.raises(ValueError, match="cache|KVCache"):
        BatchedPrefillGroup(
            SimpleNamespace(make_cache=make_cache),
            [("first", [1, 2]), ("second", [3, 4])],
            engine_stream,
        )


def test_requires_explicit_stream(tiny_model):
    with pytest.raises(ValueError, match="explicit engine stream"):
        BatchedPrefillGroup(tiny_model, [("first", [1]), ("second", [2])], None)
