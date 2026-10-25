# SPDX-License-Identifier: Apache-2.0
"""Production cache/batch lifecycle with an opt-in shared QSA family route."""

import copy
import importlib

import mlx.core as mx
import pytest
from test_qwen4_qsa_decode_gather import _tiny_text_config

from omlx.cache.type_handlers import Qwen4QSAKVCacheHandler
from omlx.cache.type_registry import CacheTypeRegistry
from omlx.patches import mlx_vlm_qwen4_exp_compat as compat
from omlx.scheduler import _patched_extend_cache, _patched_merge_caches

compat.apply_mlx_vlm_qwen4_exp_compat_patch()
language = importlib.import_module("mlx_vlm.models.qwen4_exp.language")
shared = importlib.import_module("mlx_vlm.models.qwen4_exp.shared_suffix")


def _warm(attention, length=16, *, shared_serving=True):
    cache = language.QSAKVCache(shared_serving=shared_serving)
    mx.eval(attention(mx.ones((1, length, 32)), mask="causal", cache=cache))
    return cache


def _equal_rows(batch, refs):
    for i, reference in enumerate(refs):
        actual = batch.extract(i)
        if isinstance(actual, shared.SharedSuffixQSAKVCache):
            actual = actual.materialize()
        for got, want in zip(actual.state, reference.state):
            assert mx.array_equal(got, want).item()


def test_default_off_and_selection_is_frozen_at_cache_creation():
    old = language.QSAKVCache()
    current = language.QSAKVCache(shared_serving=True)
    assert _patched_merge_caches([[old]])[0] is old
    assert isinstance(
        _patched_merge_caches([[current]])[0], shared.SharedBatchQSAKVCache
    )


def test_model_setting_marks_new_main_and_mtp_caches_for_shared_serving():
    from test_mlx_lm_mtp_patch import _model

    model = _model("qwen4")
    ordinary = [c for c in model.make_cache() if isinstance(c, language.QSAKVCache)]
    assert ordinary and all(not c._shared_serving for c in ordinary)

    model._language_model.configure_qsa_shared_prefix(True)
    main = [c for c in model.make_cache() if isinstance(c, language.QSAKVCache)]
    mtp = [c for c in model.make_mtp_cache() if isinstance(c, language.QSAKVCache)]
    assert main and all(c._shared_serving for c in main)
    assert mtp and all(c._shared_serving for c in mtp)


@pytest.mark.parametrize("length", [1, 3, 4])
@pytest.mark.parametrize("verify", [False, True])
def test_restored_prefix_batch_join_filter_clone_and_vector_rollback(
    monkeypatch, length, verify
):
    attention = language.Qwen4ExpAttention(_tiny_text_config())
    source = _warm(attention)
    handler = Qwen4QSAKVCacheHandler()
    saved = handler.serialize_state(source)
    restored = [handler.deserialize_state(saved), handler.deserialize_state(saved)]
    assert all(not hasattr(row, "_shared_loaded_owner") for row in restored)
    for row in restored:
        row.set_shared_serving(True)
    batch = _patched_merge_caches([[row] for row in restored])[0]
    reference = language.BatchQSAKVCache.merge(
        [handler.deserialize_state(saved) for _ in restored]
    )
    hidden = mx.random.normal((2, length, 32))
    actual = attention(hidden, cache=batch, target_verify=verify)
    expected = attention(hidden, mask="causal", cache=reference, target_verify=verify)
    mx.eval(actual, expected, batch.state)
    assert mx.allclose(actual, expected, atol=2e-5, rtol=2e-5).item()
    assert batch.rows[0].base is batch.rows[1].base
    assert batch.metrics["prefix_reuses"] == 1
    assert batch.metrics["materializations"] == 0
    before = copy.deepcopy(batch)
    batch.prepare(right_padding=[length, 0])
    batch.finalize()
    reference.prepare(right_padding=[length, 0])
    reference.finalize()
    _equal_rows(batch, [reference.extract(i) for i in range(2)])
    assert before.offset.tolist() == [16 + length] * 2
    # Cancellation/filter leaves the surviving row and immutable prefix intact.
    survivor_base = batch.rows[1].base
    batch.filter([1])
    reference.filter([1])
    assert batch.rows[0].base is survivor_base
    donor = _warm(attention, 18)
    joined = _patched_extend_cache([batch], _patched_merge_caches([[donor]]))[0]
    reference.extend(language.BatchQSAKVCache.merge([donor]))
    second = mx.random.normal((2, 1, 32))
    actual = attention(second, cache=joined)
    expected = attention(second, cache=reference)
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=2e-5, rtol=2e-5).item()
    _equal_rows(joined, [reference.extract(i) for i in range(2)])
    assert joined.metrics["materializations"] == 0
    joined.filter([])
    assert joined.offset.size == 0 and joined.nbytes == 0


def test_persistence_materializes_only_at_registered_handler_boundary(monkeypatch):
    attention = language.Qwen4ExpAttention(_tiny_text_config())
    source = _warm(attention)
    batch = language.QSAKVCache.merge([source, source])
    mx.eval(attention(mx.ones((2, 1, 32)), cache=batch), batch.state)
    assert batch.metrics["materializations"] == 0
    handler = CacheTypeRegistry.get_handler_by_class_name(type(batch).__name__)
    saved = handler.serialize_state(batch)
    assert batch.metrics["materializations"] == 1
    restored = handler.deserialize_state(saved)
    assert type(restored) is language.BatchQSAKVCache
    extracted = batch.extract(0)
    single_handler = CacheTypeRegistry.get_handler_by_class_name(
        type(extracted).__name__
    )
    single = single_handler.deserialize_state(single_handler.serialize_state(extracted))
    assert type(single) is language.QSAKVCache
    assert not hasattr(single, "_shared_loaded_owner")
    for got, want in zip(single.state, restored.extract(0).state):
        assert mx.array_equal(got, want).item()


def test_model_weight_revision_and_prefix_scope_are_enforced(monkeypatch):
    attention = language.Qwen4ExpAttention(_tiny_text_config())
    source = _warm(attention)
    a, b = shared._clone_plain(source), shared._clone_plain(source)
    a._shared_cache_scope, b._shared_cache_scope = "key-a", "key-b"
    batch = language.QSAKVCache.merge([a, b])
    mx.eval(attention(mx.ones((2, 1, 32)), cache=batch))
    assert batch.rows[0].base is not batch.rows[1].base
    with pytest.raises(ValueError, match="revision"):
        other = language.Qwen4ExpAttention(_tiny_text_config())
        other(mx.ones((2, 1, 32)), cache=batch)
    attention.q_proj.weight = mx.array(attention.q_proj.weight)
    with pytest.raises(ValueError, match="revision"):
        attention(mx.ones((2, 1, 32)), cache=batch)


def test_unsupported_positions_materialize_reference_without_losing_history(
    monkeypatch,
):
    attention = language.Qwen4ExpAttention(_tiny_text_config())
    source = _warm(attention)
    batch = language.QSAKVCache.merge([source])
    mx.eval(attention(mx.ones((1, 1, 32)), cache=batch))
    reference = batch.materialize()
    positions = mx.array([[30]])
    hidden = mx.random.normal((1, 1, 32))
    actual = attention(hidden, cache=batch, position_ids=positions)
    expected = attention(hidden, cache=reference, position_ids=positions)
    mx.eval(actual, expected)
    assert batch.shared_prefix_metrics()["fallback"] is True
    assert mx.allclose(actual, expected, atol=2e-5, rtol=2e-5).item()


def test_failed_vector_rollback_does_not_modify_any_row(monkeypatch):
    attention = language.Qwen4ExpAttention(_tiny_text_config())
    source = _warm(attention)
    batch = language.QSAKVCache.merge([source, source])
    mx.eval(attention(mx.ones((2, 1, 32)), cache=batch))
    batch.prepare(right_padding=[1, 2])
    with pytest.raises(ValueError, match="immutable"):
        batch.finalize()
    assert batch.offset.tolist() == [17, 17]


def test_batch_generator_decode_late_join_cancel_and_warm_restore(monkeypatch):
    from mlx_lm.generate import BatchGenerator
    from test_mlx_lm_mtp_patch import _model

    from omlx.patches import mlx_lm_mtp

    previous = mlx_lm_mtp.is_mtp_active()
    try:
        mlx_lm_mtp.set_mtp_active(False)
        mx.random.seed(389)
        model = _model("qwen4")
        model._language_model._omlx_mtp_decode_enabled = False
        prompts = [
            list(range(3, 21)),
            list(range(3, 21)) + [22, 23],
            list(range(3, 21)),
        ]

        def run(enabled):
            model._language_model.configure_qsa_shared_prefix(enabled)
            generator = BatchGenerator(
                model,
                sampler=lambda lp: mx.argmax(lp, -1),
                prefill_batch_size=3,
                max_tokens=24,
                stop_tokens=[],
            )
            try:
                ids = generator.insert(prompts[:2], max_tokens=[24, 24])
                output = {uid: [] for uid in ids}
                counters = []
                joined = None
                for step in range(80):
                    if step == 3:
                        joined = generator.insert([prompts[2]], max_tokens=[24])[0]
                        output[joined] = []
                    if step == 6:
                        generator.remove([ids[1]])
                    _, responses = generator.next()
                    for layer in generator._generation_batch.prompt_cache:
                        if isinstance(layer, shared.SharedBatchQSAKVCache):
                            counters.append(layer.shared_prefix_metrics())
                    for response in responses:
                        output[response.uid].append(response.token)
                    if (
                        len(output[ids[0]]) >= 24
                        and joined is not None
                        and len(output[joined]) >= 24
                    ):
                        break
                return output, counters
            finally:
                generator.close()

        expected, _ = run(False)
        actual, counters = run(True)
        assert actual == expected
        assert any(counter["shared_attention_calls"] > 0 for counter in counters)
        assert all(counter["materializations"] == 0 for counter in counters)
    finally:
        mlx_lm_mtp.set_mtp_active(previous)


def test_longer_identical_lineage_keeps_only_its_private_tail(monkeypatch):
    attention = language.Qwen4ExpAttention(_tiny_text_config())
    small = _warm(attention, 16)
    large = shared._clone_plain(small)
    mx.eval(attention(mx.ones((1, 4, 32)), mask="causal", cache=large))
    batch = language.QSAKVCache.merge([large, small])
    mx.eval(attention(mx.ones((2, 1, 32)), cache=batch))
    assert batch.rows[0].base is batch.rows[1].base
    assert batch.rows[0].base.length == 16
    assert [row.suffix.offset for row in batch.rows] == [5, 1]


def test_chunked_right_padding_never_freezes_provisional_tail(monkeypatch):
    attention = language.Qwen4ExpAttention(_tiny_text_config())
    batch = language.QSAKVCache.merge(
        [
            language.QSAKVCache(shared_serving=True),
            language.QSAKVCache(shared_serving=True),
        ]
    )
    reference = language.BatchQSAKVCache.merge(
        [language.QSAKVCache(), language.QSAKVCache()]
    )
    batch.prepare(right_padding=[5, 0])
    reference.prepare(right_padding=[5, 0])
    hidden = mx.random.normal((2, 12, 32))
    for start in range(0, 12, 4):
        mx.eval(attention(hidden[:, start : start + 4], cache=batch))
        mx.eval(attention(hidden[:, start : start + 4], mask="causal", cache=reference))
    assert batch.metrics["shared_rows"] == 0
    batch.finalize()
    reference.finalize()
    _equal_rows(batch, [reference.extract(i) for i in range(2)])
    mx.eval(attention(mx.ones((2, 1, 32)), cache=batch))
    assert batch.offset.tolist() == [8, 13]
    assert batch.metrics["shared_rows"] == 2


@pytest.mark.parametrize("native_mrope", [False, True])
def test_generator_warm_prefix_restore_selects_shared_route(monkeypatch, native_mrope):
    from mlx_lm.generate import BatchGenerator
    from test_mlx_lm_mtp_patch import _model

    from omlx.patches import mlx_lm_mtp

    previous = mlx_lm_mtp.is_mtp_active()
    try:
        mlx_lm_mtp.set_mtp_active(False)
        mx.random.seed(189)
        if native_mrope:
            from types import SimpleNamespace

            from test_mlx_vlm_qwen4_exp_compat import (
                _make_bound_qwen4_language_model,
                _tiny_config,
            )

            from omlx.models.vlm import VLMModelAdapter

            config = _tiny_config()
            config.text_config.head_dim = 32
            config.text_config.indexer_head_dim = 32
            config.text_config.rope_parameters["partial_rotary_factor"] = 0.25
            native_language, owner = _make_bound_qwen4_language_model(config)
            model = VLMModelAdapter(
                SimpleNamespace(language_model=native_language, config=config)
            )
            model._mtp_owner = owner
        else:
            model = _model("qwen4")
        model._language_model._omlx_mtp_decode_enabled = False
        prefix = list(range(3, 19))
        cache = model.make_cache()
        if native_mrope:
            native_positions = mx.broadcast_to(
                mx.arange(len(prefix))[None, None], (3, 1, len(prefix))
            )
            mx.eval(
                model(mx.array([prefix]), cache=cache, position_ids=native_positions)
            )
        else:
            mx.eval(model(mx.array([prefix]), cache=cache))
        serialized = []
        for layer in cache:
            handler = CacheTypeRegistry.get_handler_by_class_name(type(layer).__name__)
            serialized.append(
                (
                    handler,
                    handler.serialize_state(layer),
                    handler.serialize_meta_state(layer),
                    layer,
                )
            )

        def run(enabled):
            model._language_model.configure_qsa_shared_prefix(enabled)
            # QSA uses the real persisted block layout; unchanged recurrent
            # layers retain their runtime class because the baseline generic
            # ArraysCache handler reconstructs mlx-lm rather than mlx-vlm.
            caches = [
                [
                    (
                        handler.deserialize_state(state, meta)
                        if type(original) is language.QSAKVCache
                        else copy.deepcopy(original)
                    )
                    for handler, state, meta, original in serialized
                ]
                for _ in range(2)
            ]
            for row in caches:
                for layer in row:
                    select_shared = getattr(layer, "set_shared_serving", None)
                    if callable(select_shared):
                        select_shared(enabled)
            assert all(
                not hasattr(layer, "_shared_loaded_owner")
                for row in caches
                for layer in row
            )
            generator = BatchGenerator(
                model,
                sampler=lambda lp: mx.argmax(lp, -1),
                prefill_batch_size=2,
                max_tokens=16,
                stop_tokens=[],
            )
            output, metrics = {}, []
            try:
                ids = generator.insert(
                    [[19, 20], [19, 20]],
                    max_tokens=[16, 16],
                    caches=caches,
                    all_tokens=[prefix, prefix],
                )
                output = {uid: [] for uid in ids}
                for _ in range(50):
                    _, responses = generator.next()
                    for layer in generator._generation_batch.prompt_cache:
                        if isinstance(layer, shared.SharedBatchQSAKVCache):
                            metrics.append(layer.shared_prefix_metrics())
                    for response in responses:
                        output[response.uid].append(response.token)
                    if all(len(tokens) == 16 for tokens in output.values()):
                        break
                return output, metrics
            finally:
                generator.close()

        expected, _ = run(False)
        actual, metrics = run(True)
        assert actual == expected
        assert any(
            m["prefix_reuses"] > 0 and m["shared_attention_calls"] > 0 for m in metrics
        )
        assert all(m["materializations"] == 0 for m in metrics)
        if native_mrope:
            # Native Qwen4 layers consume rank-three positions directly.
            assert all(not m["fallback"] for m in metrics)
    finally:
        mlx_lm_mtp.set_mtp_active(previous)


@pytest.mark.parametrize("late_join", [False, True])
def test_native_mtp_verifier_keeps_shared_cache_parity(monkeypatch, late_join):
    from test_mlx_lm_mtp_patch import _model, generate

    from omlx.patches import mlx_lm_mtp

    previous = mlx_lm_mtp.is_mtp_active()
    try:
        mlx_lm_mtp.set_mtp_active(True)
        mx.random.seed(181)
        model = _model("qwen4")
        prompts = [list(range(3, 21)), list(range(3, 21)) + [22, 23]]
        expected, _ = generate(model, prompts, [32, 28], late_join=late_join)
        model._language_model.configure_qsa_shared_prefix(True)
        calls = []
        original = shared.SharedBatchQSAKVCache._batched_attention

        def counted(self, attention, hidden, **kwargs):
            calls.append((hidden.shape[1], kwargs.get("target_verify")))
            return original(self, attention, hidden, **kwargs)

        monkeypatch.setattr(shared.SharedBatchQSAKVCache, "_batched_attention", counted)
        actual, _ = generate(model, prompts, [32, 28], late_join=late_join)
        assert actual == expected
        assert calls
        if not late_join:
            assert any(verify for _, verify in calls)
            assert any(width > 1 for width, _ in calls)
        # The existing Qwen4 policy can choose ordinary decode after a join;
        # comparison uses that same policy and still requires storage use.
    finally:
        mlx_lm_mtp.set_mtp_active(previous)


def test_production_request_scope_handoff_and_owner_error_are_atomic(monkeypatch):
    from types import SimpleNamespace

    from omlx.scheduler import _bind_request_cache_scope

    attention = language.Qwen4ExpAttention(_tiny_text_config())
    source = _warm(attention)
    a, b = shared._clone_plain(source), shared._clone_plain(source)
    _bind_request_cache_scope(
        [a],
        SimpleNamespace(vlm_extra_keys_for_cache=("salt-a",)),
        shared_prefix_enabled=True,
    )
    _bind_request_cache_scope(
        [b],
        SimpleNamespace(vlm_extra_keys_for_cache=("salt-b",)),
        shared_prefix_enabled=True,
    )
    batch = language.QSAKVCache.merge([a, b])
    mx.eval(attention(mx.ones((2, 1, 32)), cache=batch))
    assert batch.rows[0].base is not batch.rows[1].base
    other = language.Qwen4ExpAttention(_tiny_text_config())
    wrong_cache = _warm(other)
    mixed = shared.SharedBatchQSAKVCache([batch.rows[0].clone(), wrong_cache])
    with pytest.raises(ValueError, match="revision"):
        attention(mx.ones((2, 1, 32)), cache=mixed)
    assert mixed.offset.tolist() == [17, 16]


@pytest.mark.parametrize("length", [1, 3, 4])
def test_native_text_mrope_and_rotary_embeddings_share_exactly(monkeypatch, length):
    attention = language.Qwen4ExpAttention(_tiny_text_config())
    prefix = mx.random.normal((1, 16, 32))
    source = language.QSAKVCache(shared_serving=True)
    positions = mx.broadcast_to(mx.arange(16)[None, None], (3, 1, 16))
    mx.eval(attention(prefix, cache=source, mask="causal", position_ids=positions))
    batch = shared.SharedBatchQSAKVCache.merge([source, source])
    reference = language.BatchQSAKVCache.merge([source, source])
    hidden = mx.random.normal((2, length, 32))
    positions = mx.broadcast_to(mx.arange(16, 16 + length)[None, None], (3, 2, length))
    rotary = attention.rotary_emb(hidden, positions)
    actual = attention(
        hidden,
        cache=batch,
        mask="causal",
        position_ids=positions,
        position_embeddings=rotary,
    )
    expected = attention(
        hidden,
        cache=reference,
        mask="causal",
        position_ids=positions,
        position_embeddings=rotary,
    )
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=2e-5, rtol=2e-5).item()
    assert batch.rows[0].base is batch.rows[1].base
    assert batch.metrics["rotary_comparisons"] == 1
    assert not batch.shared_prefix_metrics()["fallback"]
    _equal_rows(batch, [reference.extract(i) for i in range(2)])
    # Custom rotary values cannot reinterpret an existing shared prefix.
    bad = (rotary[0] + 0.25, rotary[1])
    positions = positions + length
    attention(
        hidden,
        cache=batch,
        mask="causal",
        position_ids=positions,
        position_embeddings=bad,
    )
    assert batch.shared_prefix_metrics()["fallback"]
    with pytest.raises(ValueError, match="revision"):
        language.Qwen4ExpAttention(_tiny_text_config())(hidden, cache=batch)


def test_fallback_extract_rejoin_preserves_scope_and_loaded_owner(monkeypatch):
    attention = language.Qwen4ExpAttention(_tiny_text_config())
    source = _warm(attention)
    rows = [shared._clone_plain(source) for _ in range(2)]
    for row, scope in zip(rows, ["a", "b"]):
        row.bind_cache_scope(scope)
    batch = shared.SharedBatchQSAKVCache.merge(rows)
    mx.eval(attention(mx.ones((2, 1, 32)), cache=batch))
    batch.fallback()
    batch.filter([1, 0])
    donor = shared.SharedBatchQSAKVCache.merge([batch.extract(i) for i in range(2)])
    clone = copy.deepcopy(batch)
    batch.extend(donor)
    assert [batch.extract(i)._shared_cache_scope for i in range(4)] == [
        "b",
        "a",
        "b",
        "a",
    ]
    assert clone.extract(0)._shared_cache_scope == "b"
    mx.eval(attention(mx.ones((2, 1, 32)), cache=donor))
    assert donor.rows[0].base is not donor.rows[1].base
    rejoined = shared.SharedBatchQSAKVCache.merge([clone.extract(0)])
    attention.q_proj.weight = mx.array(attention.q_proj.weight)
    with pytest.raises(ValueError, match="revision"):
        attention(mx.ones((1, 1, 32)), cache=rejoined)


def test_fallback_preserves_pending_padding_and_rejects_unfinished_join(monkeypatch):
    attention = language.Qwen4ExpAttention(_tiny_text_config())
    source = _warm(attention)
    batch = shared.SharedBatchQSAKVCache.merge([source, source])
    batch.prepare(right_padding=[2, 0])
    positions = mx.array([[90, 91], [90, 91]])
    mx.eval(
        attention(
            mx.ones((2, 2, 32)), cache=batch, position_ids=positions, mask="causal"
        )
    )
    assert batch.shared_prefix_metrics()["fallback"]
    assert batch.offset.tolist() == [18, 18]
    donor = shared.SharedBatchQSAKVCache.merge([source])
    before = batch.offset.tolist()
    with pytest.raises(ValueError, match="unfinished"):
        donor.extend(batch)
    assert donor.offset.tolist() == [16] and batch.offset.tolist() == before
    batch.finalize()
    assert batch.offset.tolist() == [16, 18]
    donor.prepare(right_padding=[1])
    with pytest.raises(ValueError, match="unfinished"):
        donor.extend(batch)
    assert donor.offset.tolist() == [16]


@pytest.mark.parametrize("activate", [False, True])
def test_cloned_banks_are_independent_after_trim_and_reappend(monkeypatch, activate):
    attention = language.Qwen4ExpAttention(_tiny_text_config())
    source = _warm(attention)
    batch = shared.SharedBatchQSAKVCache.merge([source])
    hidden = mx.random.normal((1, 4, 32))
    if activate:
        mx.eval(attention(hidden, cache=batch, mask="causal"))
    else:
        # A provisional chunk prevents freezing but still writes plain banks.
        batch.prepare(right_padding=[0])
        mx.eval(attention(hidden, cache=batch, mask="causal"))
        batch.finalize()
    donor = copy.deepcopy(batch)
    saved = (
        tuple(mx.array(value) for value in donor.extract(0).materialize().state)
        if activate
        else tuple(mx.array(value) for value in donor.extract(0).state)
    )
    batch.trim(2)
    mx.eval(attention(mx.random.normal((1, 2, 32)), cache=batch, mask="causal"))
    live = donor.extract(0).materialize() if activate else donor.extract(0)
    assert all(mx.array_equal(a, b).item() for a, b in zip(live.state, saved))
    untouched = shared.SharedBatchQSAKVCache.merge([source])
    mx.eval(attention(hidden, cache=untouched, mask="causal"))
    actual = attention(mx.ones((1, 1, 32)), cache=donor)
    expected = attention(mx.ones((1, 1, 32)), cache=untouched)
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected).item()


def test_fallback_before_first_attention_still_checks_loaded_owner(monkeypatch):
    attention = language.Qwen4ExpAttention(_tiny_text_config())
    batch = shared.SharedBatchQSAKVCache.merge([_warm(attention)])
    batch.fallback()
    with pytest.raises(ValueError, match="revision"):
        language.Qwen4ExpAttention(_tiny_text_config())(
            mx.ones((1, 1, 32)), cache=batch
        )
    assert batch.offset.tolist() == [16]
    attention.q_proj.weight = mx.array(attention.q_proj.weight)
    with pytest.raises(ValueError, match="revision"):
        attention(mx.ones((1, 1, 32)), cache=batch)


def test_fallback_pending_filter_padding_validation_and_merge_are_atomic(monkeypatch):
    attention = language.Qwen4ExpAttention(_tiny_text_config())
    source = _warm(attention)
    batch = shared.SharedBatchQSAKVCache.merge([source, source])
    batch.fallback()
    for invalid in [[-1, 0], [1]]:
        with pytest.raises(ValueError, match="padding"):
            batch.prepare(right_padding=invalid)
        assert batch._pending_right_padding is None
    batch.prepare(right_padding=[2, 0])
    with pytest.raises(ValueError, match="unfinished"):
        shared.SharedBatchQSAKVCache.merge([batch])
    batch.filter([1])
    batch.finalize()
    assert batch.offset.tolist() == [16]
    with pytest.raises(ValueError, match="history"):
        batch.trim(17)
    with pytest.raises(ValueError, match="left padding"):
        batch.prepare(left_padding=[0])
    assert batch.offset.tolist() == [16]
    batch.prepare(right_padding=[17])
    with pytest.raises(ValueError, match="history"):
        batch.finalize()
    assert batch.offset.tolist() == [16]
    batch.filter([])
    assert batch.nbytes == 0 and batch.offset.size == 0
