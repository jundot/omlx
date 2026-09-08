"""The OCR decode ring must survive oMLX cache boundaries unchanged."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.cache.type_registry import CacheTypeRegistry
from omlx.models.vlm import VLMModelAdapter
from omlx.scheduler import Scheduler, SchedulerConfig, _patched_merge_caches


def adapter():
    from omlx.patches.mlx_vlm_unlimited_ocr_compat import (
        apply_mlx_vlm_unlimited_ocr_compat_patch,
    )

    apply_mlx_vlm_unlimited_ocr_compat_patch()
    from mlx_vlm.models.unlimited_ocr.language import RingSlidingKVCache

    return VLMModelAdapter(
        SimpleNamespace(
            config=SimpleNamespace(model_type="unlimited-ocr"),
            language_model=SimpleNamespace(make_cache=lambda: [RingSlidingKVCache(2)]),
        )
    )


def append(cache, values):
    tensor = mx.array(values, dtype=mx.float32).reshape(1, 1, -1, 1)
    return cache.update_and_fetch(tensor, tensor)[0].reshape(-1).tolist()


def make_filled_cache():
    cache = adapter().make_cache()[0]
    append(cache, [0, 1, 2, 3])
    for token in range(4, 12):
        append(cache, [token])
    return cache


def test_singleton_merge_keeps_native_ring():
    cache = make_filled_cache()
    merged = _patched_merge_caches([[cache]])[0]
    assert append(merged, [12]) == [0, 1, 2, 3, 12, 11]
    assert merged.offset == 13


def test_unsafe_multirow_merge_is_rejected():
    with pytest.raises(ValueError, match="serial"):
        _patched_merge_caches([[make_filled_cache()], [make_filled_cache()]])


@pytest.mark.parametrize(
    "name, expected",
    [
        ("RingSlidingKVCache", True),
        ("OMLXRingSlidingKVCache", True),
        ("KVCache", False),
        ("RotatingKVCache", False),
        ("UnregisteredRingSlidingKVCache", False),
    ],
)
def test_ring_family_requires_registered_ring_type(name, expected):
    assert CacheTypeRegistry.is_ring_family(name) is expected


def test_ring_family_does_not_imply_adapted_prefill_interface():
    native = adapter()._language_model.make_cache()[0]
    append(native, [0, 1])
    assert CacheTypeRegistry.is_ring_family(type(native).__name__)
    Scheduler._set_ring_prefill_end([native], 3)
    assert native.offset == 2 and native.prefill_length is None


def test_ring_prefix_serialization_does_not_export_overwritten_slots():
    cache = make_filled_cache()
    handler = CacheTypeRegistry.get_handler_for_object(cache)
    state = handler.serialize_state(cache)
    assert state[0].reshape(-1).tolist() == [0, 1, 2, 3]
    assert state[1].shape == (1, 1, 4, 1)


def test_restored_partial_prefix_reestablishes_ring_boundary():
    cache = make_filled_cache()
    handler = CacheTypeRegistry.get_handler_for_object(cache)
    state = {"keys": cache.keys[:, :, :2], "values": cache.values[:, :, :2]}
    restored = handler.reconstruct_cache(state, handler.serialize_meta_state(cache))
    append(restored, [2, 3])
    for token in range(4, 12):
        result = append(restored, [token])
    assert result == [0, 1, 2, 3, 10, 11]
    assert restored.offset == 12


@pytest.mark.parametrize(
    "meta",
    [
        None,
        (),
        ("0", "-1", "4", "0"),
        ("2", "4", "99", "0"),
        ("2", "4", "4", "0", "extra"),
    ],
)
def test_bad_ring_metadata_rejected(meta):
    cache = make_filled_cache()
    handler = CacheTypeRegistry.get_handler_for_object(cache)
    state = {"keys": cache.keys[:, :, :4], "values": cache.values[:, :, :4]}
    with pytest.raises(ValueError):
        handler.reconstruct_cache(state, meta)


def test_scheduler_serializes_only_ring_model(mock_model, mock_tokenizer):
    scheduler = Scheduler(mock_model, mock_tokenizer, SchedulerConfig(max_num_seqs=8))
    assert scheduler._effective_max_num_seqs() == 8
    scheduler.model = adapter()
    assert scheduler._effective_max_num_seqs() == 1


def test_scheduler_exports_only_intact_prefix(mock_model, mock_tokenizer):
    scheduler = Scheduler(mock_model, mock_tokenizer)
    payload, config = scheduler._extract_cache_states([make_filled_cache()])
    assert payload[0]["state"][0].reshape(-1).tolist() == [0, 1, 2, 3]
    handler = CacheTypeRegistry.get_handler_by_class_name(
        config.layer_configs[0].class_name
    )
    restored = handler.reconstruct_cache(
        {"keys": payload[0]["state"][0], "values": payload[0]["state"][1]},
        payload[0]["meta_state"],
    )
    for token in range(4, 12):
        values = append(restored, [token])
    assert values == [0, 1, 2, 3, 10, 11]


def test_ring_is_not_turboquant_convertible(mock_model, mock_tokenizer):
    scheduler = Scheduler(mock_model, mock_tokenizer)
    assert not scheduler._turboquant_eligible([make_filled_cache()])


def test_turboquant_refuses_cache_when_registry_is_unavailable(
    monkeypatch, mock_model, mock_tokenizer
):
    from mlx_lm.models.cache import KVCache

    import omlx.scheduler as scheduler_module

    scheduler = Scheduler(mock_model, mock_tokenizer)
    monkeypatch.setattr(scheduler_module, "CacheTypeRegistry", None)
    assert not scheduler._turboquant_eligible([KVCache()])


def test_stale_plain_kv_prefix_is_incompatible():
    from mlx_lm.models.cache import KVCache

    model = adapter()
    assert not model.is_prefix_cache_compatible([KVCache()])
    assert model.is_prefix_cache_compatible(model.make_cache())


def test_singleton_protocol_and_trim():
    cache = adapter().make_cache()[0]
    append(cache, [0, 1, 2, 3])
    assert cache.to_batch([0]) is cache
    assert cache.extract(0) is cache
    cache.filter([0])
    assert cache.is_trimmable()
    assert cache.trim(1) == 1
    assert cache.offset == 3
    append(cache, [3])
    assert not cache.is_trimmable()
    with pytest.raises(ValueError, match="trim"):
        cache.trim(1)
    with pytest.raises(ValueError, match="serial"):
        cache.to_batch([1])
    with pytest.raises(ValueError, match="serial"):
        cache.extend(cache)
    cache.filter([])
    assert cache.offset == 0 and cache.keys is None


@pytest.mark.parametrize("block_size", [2, 3])
def test_ssd_store_and_restore_only_prompt_blocks(
    tmp_path, mock_model, mock_tokenizer, caplog, block_size
):
    from omlx.cache.paged_cache import PagedCacheManager
    from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
    from omlx.cache.prefix_cache import BlockAwarePrefixCache

    ssd = PagedSSDCacheManager(
        cache_dir=tmp_path,
        max_size_bytes=1024**2,
        expected_model_name="ocr",
        expected_num_layers=1,
        expected_block_size=block_size,
    )
    paged = PagedCacheManager(
        block_size=block_size, max_blocks=32, initial_blocks=32, model_name="ocr"
    )
    paged.set_paged_ssd_cache_manager(ssd)
    prefix = BlockAwarePrefixCache(
        model=SimpleNamespace(layers=[None]),
        paged_cache_manager=paged,
        paged_ssd_cache_manager=ssd,
    )
    scheduler = Scheduler(mock_model, mock_tokenizer)
    payload, config = scheduler._extract_cache_states([make_filled_cache()])
    try:
        table = prefix.store_cache(
            "cold", list(range(12)), payload, model_cache_config=config
        )
        assert table is not None
        assert table.num_tokens == 4 // block_size * block_size
        assert "Rejecting block" not in caplog.text
        restored = prefix.reconstruct_cache(table)[0]
        if table.num_tokens < 4:
            # Append a multi-token suffix so it remains prefill, not decode.
            append(restored, list(range(table.num_tokens, 5)))
            expected_prefix = list(range(5))
        else:
            expected_prefix = list(range(4))
        for token in range(len(expected_prefix), 12):
            values = append(restored, [token])
        assert values[: len(expected_prefix)] == expected_prefix
        assert sorted(values[len(expected_prefix) :]) == [10, 11]

        # The first blocks retain the first capture's metadata after dedup.
        extended = adapter().make_cache()[0]
        append(extended, list(range(8)))
        payload, config = scheduler._extract_cache_states([extended])
        table = prefix.store_cache(
            "extended", list(range(8)), payload, model_cache_config=config
        )
        restored = prefix.reconstruct_cache(table)
        assert restored is not None
        assert restored[0].offset == 8 // block_size * block_size
    finally:
        ssd.close()


def test_store_refuses_missing_ring_layer_before_allocating_blocks(
    tmp_path, mock_model, mock_tokenizer
):
    scheduler = Scheduler(mock_model, mock_tokenizer)
    payload, config = scheduler._extract_cache_states(
        [make_filled_cache(), make_filled_cache()]
    )
    prefix, paged, ssd = make_ssd_stack(tmp_path, 2)
    try:
        # The model declares two ring layers, but only the first was captured.
        table = prefix.store_cache(
            "incomplete", list(range(12)), payload[:1], model_cache_config=config
        )
        assert table is None
        assert paged.get_block_table("incomplete") is None
        assert "incomplete" not in prefix._request_tables
        assert all(b.ref_count == 0 for b in paged.blocks if not b.is_null)
        assert all(b.block_hash is None for b in paged.blocks if not b.is_null)
    finally:
        ssd.close()


@pytest.mark.parametrize("with_config", [False, True])
@pytest.mark.parametrize("layer_idx", [0, 1])
@pytest.mark.parametrize(
    "malformed_payload",
    [
        pytest.param(lambda k, v: {}, id="missing-state"),
        pytest.param(lambda k, v: {"state": None}, id="none-state"),
        pytest.param(lambda k, v: {"state": ()}, id="empty-state"),
        pytest.param(lambda k, v: {"state": 7}, id="scalar-state"),
        pytest.param(lambda k, v: {"state": (k,)}, id="missing-value"),
        pytest.param(lambda k, v: {"state": (k, v, v)}, id="extra-element"),
        pytest.param(lambda k, v: {"state": (None, v)}, id="none-key"),
        pytest.param(lambda k, v: {"state": (k, None)}, id="none-value"),
        pytest.param(lambda k, v: {"state": (7, v)}, id="non-tensor-key"),
        pytest.param(lambda k, v: {"state": (k, 7)}, id="non-tensor-value"),
        pytest.param(lambda k, v: {"state": (k[0, 0], v)}, id="rank-two-key"),
        pytest.param(lambda k, v: {"state": (k, v[0])}, id="rank-three-value"),
        pytest.param(lambda k, v: {"state": (k[None], v)}, id="rank-five-key"),
        pytest.param(lambda k, v: {"state": (k, v[:, :, :2])}, id="short-value"),
        pytest.param(
            lambda k, v: {"state": (k, mx.concatenate([v, v], axis=1))},
            id="head-count-mismatch",
        ),
        pytest.param(
            lambda k, v: {
                "state": (mx.concatenate([k, k]), mx.concatenate([v, v]))
            },
            id="multi-row",
        ),
        pytest.param(
            lambda k, v: {"state": (k[:, :, :0], v[:, :, :0])},
            id="empty-prefix",
        ),
    ],
)
def test_store_refuses_malformed_ring_state_before_allocating_blocks(
    tmp_path, mock_model, mock_tokenizer, malformed_payload, layer_idx, with_config
):
    scheduler = Scheduler(mock_model, mock_tokenizer)
    payload, config = scheduler._extract_cache_states(
        [make_filled_cache(), make_filled_cache()]
    )
    keys, values = payload[layer_idx].pop("state")
    payload[layer_idx].update(malformed_payload(keys, values))
    prefix, paged, ssd = make_ssd_stack(tmp_path, 2, num_layers=2)
    try:
        table = prefix.store_cache(
            "malformed",
            list(range(12)),
            payload,
            model_cache_config=config if with_config else None,
        )
        assert table is None
        assert paged.get_block_table("malformed") is None
        assert "malformed" not in prefix._request_tables
        assert all(b.ref_count == 0 for b in paged.blocks if not b.is_null)
        assert all(b.block_hash is None for b in paged.blocks if not b.is_null)
    finally:
        ssd.close()


@pytest.mark.parametrize("with_config", [False, True])
@pytest.mark.parametrize("state_type", [tuple, list])
def test_store_accepts_ring_kv_with_distinct_feature_dimensions(
    tmp_path, mock_model, mock_tokenizer, with_config, state_type
):
    scheduler = Scheduler(mock_model, mock_tokenizer)
    payload, config = scheduler._extract_cache_states([make_filled_cache()])
    keys = mx.arange(8, dtype=mx.float32).reshape(1, 1, 4, 2)
    values = mx.arange(12, dtype=mx.float32).reshape(1, 1, 4, 3)
    payload[0]["state"] = state_type((keys, values))
    prefix, paged, ssd = make_ssd_stack(tmp_path, 2)
    try:
        table = prefix.store_cache(
            "valid",
            list(range(12)),
            payload,
            model_cache_config=config if with_config else None,
        )
        assert table is not None and table.num_tokens == 4
        restored = prefix.reconstruct_cache(table)
        assert restored is not None and restored[0].offset == 4
        restored_keys, restored_values = restored[0].state
        assert mx.array_equal(restored_keys, keys).item()
        assert mx.array_equal(restored_values, values).item()
    finally:
        ssd.close()


def test_restored_unequal_width_ring_matches_native_decode_after_wrap(
    tmp_path, mock_model, mock_tokenizer
):
    # Native KVCache allocates K and V widths separately. Verify the restored
    # ring preserves that contract through decode and actual MLX attention.
    native = adapter()._language_model.make_cache()[0]
    keys = mx.broadcast_to(mx.arange(4)[None, None, :, None], (1, 1, 4, 192))
    values = mx.broadcast_to(
        mx.arange(100, 104)[None, None, :, None], (1, 1, 4, 128)
    )
    native.update_and_fetch(keys.astype(mx.float32), values.astype(mx.float32))
    scheduler = Scheduler(mock_model, mock_tokenizer)
    payload, config = scheduler._extract_cache_states([native])
    prefix, paged, ssd = make_ssd_stack(tmp_path, 2)
    try:
        table = prefix.store_cache(
            "unequal-decode", list(range(4)), payload, model_cache_config=config
        )
        assert table is not None and table.num_tokens == 4
        restored_layers = prefix.reconstruct_cache(table)
        assert restored_layers is not None
        restored = restored_layers[0]
        assert restored is not native
        query = mx.ones((1, 1, 1, 192))
        for token in range(4, 10):
            keys = mx.full((1, 1, 1, 192), float(token))
            values = mx.full((1, 1, 1, 128), float(100 + token))
            expected_keys, expected_values = native.update_and_fetch(keys, values)
            actual_keys, actual_values = restored.update_and_fetch(keys, values)
            assert mx.array_equal(actual_keys, expected_keys).item()
            assert mx.array_equal(actual_values, expected_values).item()
            expected = mx.fast.scaled_dot_product_attention(
                query, expected_keys, expected_values, scale=192**-0.5
            )
            actual = mx.fast.scaled_dot_product_attention(
                query, actual_keys, actual_values, scale=192**-0.5
            )
            assert actual.shape == (1, 1, 1, 128)
            assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item()
        # Four prompt tokens stay permanent; the two-slot ring wrapped twice.
        assert restored.offset == 10 and restored.prefill_length == 4
        assert actual_keys[0, 0, :, 0].tolist() == [0, 1, 2, 3, 8, 9]
        assert actual_values[0, 0, :, 0].tolist() == [100, 101, 102, 103, 108, 109]
    finally:
        ssd.close()


def test_scheduler_discards_legacy_ocr_prefix(mock_model, mock_tokenizer):
    from unittest.mock import MagicMock

    from mlx_lm.models.cache import KVCache

    from omlx.cache.paged_cache import BlockTable
    from omlx.request import Request, SamplingParams

    scheduler = Scheduler(mock_model, mock_tokenizer)
    scheduler.model = adapter()
    scheduler.block_aware_cache = MagicMock()
    scheduler.paged_cache_manager = MagicMock()
    table = BlockTable(request_id="legacy", block_ids=[1], num_tokens=2)
    scheduler.block_aware_cache.fetch_cache.return_value = (table, [12, 13])
    scheduler.block_aware_cache.reconstruct_cache.return_value = [KVCache()]
    request = Request(
        request_id="legacy",
        prompt=[10, 11, 12, 13],
        sampling_params=SamplingParams(max_tokens=4),
    )
    scheduler.add_request(request)
    scheduler._prepare_prefix_cache_for_request(request)
    assert request.cached_tokens == 0 and request.prompt_cache is None
    assert request.remaining_tokens == [10, 11, 12, 13]
    scheduler.paged_cache_manager.delete_block_table.assert_called_once_with("legacy")


def make_ssd_stack(directory, block_size, num_layers=1):
    from omlx.cache.paged_cache import PagedCacheManager
    from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
    from omlx.cache.prefix_cache import BlockAwarePrefixCache

    ssd = PagedSSDCacheManager(
        cache_dir=directory,
        max_size_bytes=1024**2,
        hot_cache_max_bytes=0,
        expected_model_name="ocr",
        expected_num_layers=num_layers,
        expected_block_size=block_size,
    )
    paged = PagedCacheManager(
        block_size=block_size, max_blocks=32, initial_blocks=32, model_name="ocr"
    )
    paged.set_paged_ssd_cache_manager(ssd)
    prefix = BlockAwarePrefixCache(
        model=SimpleNamespace(layers=[None] * num_layers),
        paged_cache_manager=paged,
        paged_ssd_cache_manager=ssd,
    )
    return prefix, paged, ssd


class TinyRingLanguageModel:
    """Weight-free model exercising real scheduler prefill and EOS completion."""

    layers = [None]

    def __init__(self):
        self.seen_tokens = []

    def make_cache(self):
        from mlx_vlm.models.unlimited_ocr.language import RingSlidingKVCache

        return [RingSlidingKVCache(2)]

    def __call__(self, input_ids, cache=None, **kwargs):
        self.seen_tokens.extend(input_ids.reshape(-1).tolist())
        tensor = input_ids.astype(mx.float32)[:, None, :, None]
        for layer in cache or []:
            layer.update_and_fetch(tensor, tensor)
        # The tokenizer fixture's EOS is 2. Return real logits, not a mocked
        # scheduler/BatchGenerator response, so fallback must actually run.
        return mx.broadcast_to(mx.array([0.0, 0.0, 10.0, 0.0]), (*input_ids.shape, 4))


class RetentionLogitsModel(TinyRingLanguageModel):
    """Make an early ring overwrite observable in generated token IDs."""

    def __init__(self):
        self.boundaries = []

    def make_cache(self):
        from mlx_vlm.models.unlimited_ocr.language import RingSlidingKVCache

        return [RingSlidingKVCache(128)]

    def __call__(self, input_ids, cache=None, **kwargs):
        tensor = input_ids.astype(mx.float32)[:, None, :, None]
        keys, _ = cache[0].update_and_fetch(tensor, tensor)
        self.boundaries.append(cache[0].prefill_length)
        token = 3 + int(mx.sum(keys).item()) % 20
        logits = mx.where(mx.arange(32) == token, 10.0, -10.0)
        return mx.broadcast_to(logits, (*input_ids.shape, 32))


@pytest.mark.parametrize(
    "prompt_length,stored_length,step_size,chunked",
    [
        (8, 8, 2048, False),  # exact hit trims to N-1
        (9, 8, 2048, False),  # one uncached token is kickoff, not prefill
        (8, 6, 2048, False),  # two uncached tokens split into prefill + kickoff
        (8, 0, 2, False),  # synchronous prefill ends in a single-token chunk
        (8, 0, 2, True),  # the same tail across scheduler step() calls
        (8, 4, 2, True),  # chunked continuation from a partial prefix
        (8, 0, 1, True),  # every prefill chunk is a single token
    ],
)
def test_scheduler_prefill_boundary_matches_cold_past_ring_window(
    tmp_path,
    mock_model,
    mock_tokenizer,
    prompt_length,
    stored_length,
    step_size,
    chunked,
):
    import time

    from omlx.request import Request, SamplingParams

    prompt = list(range(10, 10 + prompt_length))

    def run(label, stored, step, use_chunks):
        model = adapter()
        language = RetentionLogitsModel()
        model._language_model = language
        scheduler = Scheduler(
            mock_model,
            mock_tokenizer,
            SchedulerConfig(prefill_step_size=step, chunked_prefill=use_chunks),
        )
        scheduler.model = model
        prefix, paged, ssd = make_ssd_stack(tmp_path / label, 2)
        scheduler.block_aware_cache = prefix
        scheduler.paged_cache_manager = paged
        try:
            if stored:
                cache = model.make_cache()[0]
                append(cache, prompt[:stored])
                payload, config = scheduler._extract_cache_states([cache])
                table = prefix.store_cache(
                    "seed", prompt[:stored], payload, model_cache_config=config
                )
                assert table is not None
                paged.delete_block_table("seed")
            request = Request(
                label, prompt, SamplingParams(max_tokens=132, temperature=0)
            )
            scheduler.add_request(request)
            scheduler._prepare_prefix_cache_for_request(request)
            assert request.cached_tokens == (
                stored - 1 if stored == prompt_length else stored
            )
            deadline = time.monotonic() + 10
            while not request.is_finished() and time.monotonic() < deadline:
                scheduler.step()
                # Chunked prefill yields to the scheduler's fairness timer.
                if not request.is_finished():
                    time.sleep(0.001)
            state = scheduler._prefill_states.get(request.request_id)
            assert request.is_finished(), (
                f"{label}: scheduler did not finish within 10s; "
                f"status={request.status.name}, "
                f"generated={len(request.output_token_ids)}/132, "
                f"cached_tokens={request.cached_tokens}, "
                f"chunked={use_chunks}, step_size={step}, "
                f"prefill_processed={state.tokens_processed if state else 'inactive'}, "
                f"prefill_remaining={state.tokens_remaining.size if state else 'inactive'}"
            )
            assert len(request.output_token_ids) == 132
            assert request.get_finish_reason() == "length"
            assert all(b.ref_count == 0 for b in paged.blocks if not b.is_null)
            return request.output_token_ids, language.boundaries
        finally:
            if scheduler.batch_generator is not None:
                scheduler.batch_generator.close()
            ssd.close()

    cold, _ = run("cold-reference", 0, 2048, False)
    actual, boundaries = run("candidate", stored_length, step_size, chunked)
    # Preserve this serving engine's N-1 kickoff, not native full-prompt parity.
    assert {b for b in boundaries if b is not None} == {prompt_length - 1}
    assert actual == cold


def test_explicit_prefill_end_survives_singletons_and_resume():
    cache = adapter().make_cache()[0]
    cache.set_prefill_end(4)
    append(cache, [0, 1])
    # Resume after a scheduler yield, still before the same absolute boundary.
    cache.set_prefill_end(4)
    append(cache, [2])
    assert cache.prefill_length is None
    assert cache.is_trimmable()
    append(cache, [3])
    assert cache.prefill_length is None
    for token in range(4, 12):
        values = append(cache, [token])
    assert cache.prefill_length == 4
    assert values == [0, 1, 2, 3, 10, 11]


def test_prefill_end_rejects_crossing_or_reopening_decode():
    cache = adapter().make_cache()[0]
    with pytest.raises(ValueError):
        cache.set_prefill_end(-1)
    cache.set_prefill_end(2)
    with pytest.raises(ValueError):
        append(cache, [0, 1, 2])
    assert cache.offset == 0
    append(cache, [0, 1])
    with pytest.raises(ValueError):
        cache.set_prefill_end(1)
    append(cache, [2])
    with pytest.raises(ValueError):
        cache.set_prefill_end(5)


def test_empty_filter_discards_request_prefill_end():
    cache = adapter().make_cache()[0]
    cache.set_prefill_end(100)
    append(cache, [0, 1])
    cache.filter([])
    append(cache, [10, 11])
    for token in range(12, 18):
        values = append(cache, [token])
    assert cache.prefill_length == 2
    assert values == [10, 11, 16, 17]


def test_guard_refuses_native_ring_until_registration(
    monkeypatch, mock_model, mock_tokenizer
):
    model = adapter()
    native_model = model._language_model
    with monkeypatch.context() as unregistered:
        for name in ("RingSlidingKVCache", "OMLXRingSlidingKVCache"):
            unregistered.delitem(CacheTypeRegistry._class_name_map, name)
        before = Scheduler(mock_model, mock_tokenizer)
        before.model = native_model
        assert before._model_has_unreconstructible_cache()
    after = Scheduler(mock_model, mock_tokenizer)
    after.model = model
    assert not after._model_has_unreconstructible_cache()


@pytest.mark.parametrize("block_size", [2, 3])
@pytest.mark.parametrize("legacy_layout", ["native-decode", "plain-kv"])
def test_persisted_legacy_entries_release_real_references_after_restart(
    tmp_path, mock_model, mock_tokenizer, block_size, legacy_layout, monkeypatch
):
    from mlx_lm.models.cache import KVCache

    from omlx.cache.hybrid_cache import ModelCacheConfig
    from omlx.request import Request, SamplingParams

    # Capture the old writer's raw state/meta contract, before the ring handler
    # exported prompt-only tensors. No production image/hash salt is involved.
    native = adapter()._language_model.make_cache()[0]
    append(native, [0, 1, 2, 3])
    for token in range(4, 12):
        append(native, [token])
    if legacy_layout == "plain-kv":
        old = KVCache()
        old.state = native.state
    else:
        old = native
        assert old.meta_state == ("2", "4", "12", "0")
    payload = [
        {
            "state": old.state,
            "meta_state": old.meta_state,
            "class_name": type(old).__name__,
            "cache_type": "KVCache",
        }
    ]
    prefix, paged, ssd = make_ssd_stack(tmp_path, block_size)
    try:
        with monkeypatch.context() as unregistered:
            for name in ("RingSlidingKVCache", "OMLXRingSlidingKVCache"):
                unregistered.delitem(CacheTypeRegistry._class_name_map, name)
            config = ModelCacheConfig.from_cache_list([old])
            table = prefix.store_cache(
                "old-writer", list(range(12)), payload, model_cache_config=config
            )
        assert table is not None and table.num_tokens == 6
    finally:
        ssd.close()
    assert list(tmp_path.rglob("*.safetensors")), "fixture must reach disk"

    prefix, paged, ssd = make_ssd_stack(tmp_path, block_size)
    scheduler = Scheduler(mock_model, mock_tokenizer)
    scheduler.model = adapter()
    language_model = TinyRingLanguageModel()
    scheduler.model._language_model = language_model
    scheduler.block_aware_cache = prefix
    scheduler.paged_cache_manager = paged
    try:
        # Prove the unchanged identity actually hits the old persisted chain.
        table, remaining = prefix.fetch_cache("probe", list(range(8)))
        assert table is not None and table.num_tokens == 6
        assert remaining == [6, 7]
        if legacy_layout == "native-decode":
            # Exercise handler metadata refusal, not just the adapter's
            # different-class check for a plain-KV reconstruction.
            assert prefix.reconstruct_cache(table) is None
        else:
            assert type(prefix.reconstruct_cache(table)[0]) is KVCache
        old_block_ids = tuple(table.block_ids)
        paged.delete_block_table("probe")
        for attempt in range(3):
            request = Request(
                request_id=f"legacy-{attempt}",
                prompt=list(range(8)),
                sampling_params=SamplingParams(max_tokens=4),
            )
            scheduler.add_request(request)
            scheduler._prepare_prefix_cache_for_request(request)
            if attempt == 0:
                assert request.cached_tokens == 0
                assert request.prompt_cache is None and request.block_table is None
                assert request.remaining_tokens == list(range(8))
                assert paged.get_block_table(request.request_id) is None
                assert all(paged.blocks[i].ref_count == 0 for i in old_block_ids)
                assert all(b.ref_count == 0 for b in paged.blocks if not b.is_null)
                assert paged.free_block_queue.num_free_blocks == 31
                expected_prefill = list(range(8))
            else:
                # Completion rewrites the rejected entries with safe prompt
                # captures. Subsequent requests must reuse those, not keep
                # missing forever or inherit the old full-KV attention.
                assert request.cached_tokens == 6
                assert scheduler.model.is_prefix_cache_compatible(request.prompt_cache)
                expected_prefill = [6, 7]
            language_model.seen_tokens.clear()
            for _ in range(10):
                scheduler.step()
                if request.is_finished():
                    break
            assert request.is_finished()
            assert request.get_finish_reason() == "stop"
            assert (
                language_model.seen_tokens[: len(expected_prefill)] == expected_prefill
            )
            assert all(
                token == 2
                for token in language_model.seen_tokens[len(expected_prefill) :]
            )
            assert all(b.ref_count == 0 for b in paged.blocks if not b.is_null)
    finally:
        ssd.close()


@pytest.mark.parametrize("block_size", [2, 3])
def test_guarded_reuse_after_restart_preserves_ring(
    tmp_path, mock_model, mock_tokenizer, block_size
):
    from omlx.request import Request, SamplingParams

    scheduler = Scheduler(mock_model, mock_tokenizer)
    cache = adapter().make_cache()[0]
    append(cache, list(range(8)))
    payload, config = scheduler._extract_cache_states([cache])
    prefix, _, ssd = make_ssd_stack(tmp_path, block_size)
    try:
        table = prefix.store_cache(
            "cold", list(range(8)), payload, model_cache_config=config
        )
        assert table is not None
    finally:
        ssd.close()
    prefix, paged, ssd = make_ssd_stack(tmp_path, block_size)
    scheduler.model = adapter()
    scheduler.block_aware_cache = prefix
    scheduler.paged_cache_manager = paged
    try:
        request = Request("warm", list(range(10)), SamplingParams(max_tokens=4))
        scheduler.add_request(request)
        scheduler._prepare_prefix_cache_for_request(request)
        assert request.cached_tokens == (8 if block_size == 2 else 6)
        restored = request.prompt_cache[0]
        append(restored, request.remaining_tokens)
        for token in range(10, 16):
            values = append(restored, [token])
        assert values == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15]
        paged.delete_block_table(request.request_id)
        assert all(b.ref_count == 0 for b in paged.blocks if not b.is_null)
    finally:
        ssd.close()


@pytest.mark.parametrize("block_size", [2, 3])
def test_deduplicated_legacy_prefill_cannot_hide_decoded_tail(
    tmp_path, monkeypatch, block_size
):
    from omlx.cache.hybrid_cache import ModelCacheConfig

    native = adapter()._language_model.make_cache()[0]
    append(native, [0, 1, 2, 3])
    prefix, _, ssd = make_ssd_stack(tmp_path, block_size)
    try:
        with monkeypatch.context() as unregistered:
            unregistered.delitem(
                CacheTypeRegistry._class_name_map, "RingSlidingKVCache"
            )
            config = ModelCacheConfig.from_cache_list([native])
            first = [
                {
                    "state": native.state,
                    "meta_state": native.meta_state,
                    "class_name": "RingSlidingKVCache",
                    "cache_type": "KVCache",
                }
            ]
            table = prefix.store_cache(
                "old-prefill", list(range(4)), first, model_cache_config=config
            )
            assert table is not None
            for token in range(4, 12):
                append(native, [token])
            decoded = [
                {
                    "state": native.state,
                    "meta_state": native.meta_state,
                    "class_name": "RingSlidingKVCache",
                    "cache_type": "KVCache",
                }
            ]
            table = prefix.store_cache(
                "old-decode", list(range(12)), decoded, model_cache_config=config
            )
            assert table is not None and table.num_tokens == 6
    finally:
        ssd.close()
    prefix, paged, ssd = make_ssd_stack(tmp_path, block_size)
    try:
        table, _ = prefix.fetch_cache("warm-mixed", list(range(8)))
        assert table is not None and table.num_tokens == 6
        # The first block's metadata is valid prefill; a later block contains
        # overwritten decode slots [10, 11] falsely labeled as prompt [4, 5].
        assert prefix.reconstruct_cache(table) is None
        paged.delete_block_table("warm-mixed")
        assert all(b.ref_count == 0 for b in paged.blocks if not b.is_null)
    finally:
        ssd.close()


def test_ring_block_tensor_length_must_match_token_count(
    tmp_path, mock_model, mock_tokenizer
):
    scheduler = Scheduler(mock_model, mock_tokenizer)
    cache = adapter().make_cache()[0]
    append(cache, [0, 1, 2, 3])
    payload, config = scheduler._extract_cache_states([cache])
    prefix, paged, ssd = make_ssd_stack(tmp_path, 2)
    try:
        table = prefix.store_cache(
            "cold", [0, 1, 2, 3], payload, model_cache_config=config
        )
        tail_hash = paged.blocks[table.block_ids[-1]].block_hash
    finally:
        ssd.close()
    prefix, _, ssd = make_ssd_stack(tmp_path, 2)
    try:
        short = mx.array([2.0]).reshape(1, 1, 1, 1)
        assert ssd.save_block(
            tail_hash,
            [(short, short)],
            token_count=2,
            model_name="ocr",
            layer_cache_types=["OMLXRingSlidingKVCache"],
            layer_meta_states=[("2", "-1", "4", "0")],
            replace_existing=True,
        )
    finally:
        ssd.close()
    prefix, paged, ssd = make_ssd_stack(tmp_path, 2)
    try:
        table, _ = prefix.fetch_cache("warm-short", [0, 1, 2, 3, 4, 5])
        assert table is not None and table.num_tokens == 4
        assert prefix.reconstruct_cache(table) is None
        paged.delete_block_table("warm-short")
        assert all(b.ref_count == 0 for b in paged.blocks if not b.is_null)
    finally:
        ssd.close()


@pytest.mark.parametrize(
    "corruption, diagnostic",
    [
        ("metadata", "Missing Unlimited-OCR metadata for layer 1 in block 1"),
        ("layer", "Missing Unlimited-OCR layer 1 in block 1"),
        ("allocation", "is no longer allocated"),
    ],
)
def test_ring_reconstruction_missing_entries_report_context_and_release_refs(
    tmp_path, mock_model, mock_tokenizer, monkeypatch, caplog, corruption, diagnostic
):
    from omlx.request import Request, SamplingParams

    scheduler = Scheduler(mock_model, mock_tokenizer)
    caches = [make_filled_cache(), make_filled_cache()]
    payload, config = scheduler._extract_cache_states(caches)
    prefix, paged, ssd = make_ssd_stack(tmp_path, 2, num_layers=2)
    try:
        table = prefix.store_cache(
            "seed", list(range(4)), payload, model_cache_config=config
        )
        assert table is not None and table.num_tokens == 4
        tail_hash = paged.blocks[table.block_ids[-1]].block_hash
        paged.delete_block_table("seed")
        load = ssd.load_block_with_metadata

        def load_incomplete_tail(block_hash, **kwargs):
            data, metadata = load(block_hash, **kwargs)
            if block_hash == tail_hash:
                if corruption == "metadata":
                    metadata = dict(metadata)
                    metadata["layer_meta_states"] = metadata["layer_meta_states"][:1]
                elif corruption == "layer":
                    data = data[:1]
                else:
                    # Simulate a missing allocation on the validation lookup
                    # only. Keep the real owned blocks available for cleanup.
                    class MissingValidationLookup(dict):
                        missing = True

                        def get(self, key, default=None):
                            block = super().get(key, default)
                            if (
                                block is not None
                                and block.block_hash == tail_hash
                                and self.missing
                            ):
                                self.missing = False
                                return None
                            return block

                    monkeypatch.setattr(
                        paged,
                        "allocated_blocks",
                        MissingValidationLookup(paged.allocated_blocks),
                    )
            return data, metadata

        monkeypatch.setattr(ssd, "load_block_with_metadata", load_incomplete_tail)
        scheduler.block_aware_cache = prefix
        scheduler.paged_cache_manager = paged
        request = Request("incomplete-tail", list(range(6)), SamplingParams(max_tokens=4))
        scheduler.add_request(request)
        scheduler._prepare_prefix_cache_for_request(request)
        assert request.cached_tokens == 0
        assert request.prompt_cache is None and request.block_table is None
        assert request.remaining_tokens == list(range(6))
        assert paged.get_block_table(request.request_id) is None
        assert all(b.ref_count == 0 for b in paged.blocks if not b.is_null)
        assert diagnostic in caplog.text
    finally:
        ssd.close()
