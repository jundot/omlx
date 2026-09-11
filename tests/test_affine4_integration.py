# SPDX-License-Identifier: Apache-2.0
"""Affine4 routing through model attention, batching, and scheduler admission."""

import importlib
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache, RotatingKVCache

from omlx.affine4 import (
    Affine4KVCache,
    Affine8KVCache,
    BatchAffine4KVCache,
    BatchAffine8KVCache,
)
from omlx.patches.turboquant_attention import apply_turboquant_attention_patch
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, _is_turboquant_kv_family_cache
from omlx.turboquant_kv import BatchTurboQuantKVCache, TurboQuantKVCache


@pytest.fixture
def scheduler(mock_model, mock_tokenizer):
    scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
    scheduler._turboquant_kv_bits = 4.0
    scheduler._turboquant_kv_scheme = "affine4"
    return scheduler


@pytest.mark.parametrize("populated", [False, True])
def test_scheduler_preserves_hybrid_layout_and_last_layer(scheduler, populated):
    caches = [KVCache(), ArraysCache(size=2), RotatingKVCache(max_size=32), KVCache()]
    if populated:
        for cache in (caches[0], caches[-1]):
            cache.update_and_fetch(mx.ones((1, 2, 7, 64)), mx.ones((1, 2, 7, 64)))
    untouched = caches[1:]
    assert scheduler._turboquant_eligible(caches)
    convert = (
        scheduler._apply_turboquant_kv_convert
        if populated
        else scheduler._apply_turboquant_kv_empty
    )
    convert(caches)
    assert type(caches[0]) is Affine4KVCache
    assert caches[1:] == untouched
    assert caches[0].offset == (7 if populated else 0)
    assert _is_turboquant_kv_family_cache(caches[0])
    assert scheduler._turboquant_eligible(caches)


def test_scheduler_signature_matches_selected_format(scheduler):
    scheduler.model = SimpleNamespace(make_cache=lambda: [KVCache(), KVCache()])
    types, bits, _ = scheduler._infer_live_layer_cache_types()
    assert types == ["Affine4KVCache", "KVCache"]
    assert bits == 4
    scheduler._turboquant_kv_scheme = "affine8"
    scheduler._turboquant_kv_bits = 8
    types, bits, _ = scheduler._infer_live_layer_cache_types()
    assert types == ["Affine8KVCache", "KVCache"]
    assert bits == 8
    scheduler._turboquant_kv_scheme = "turboquant"
    types, bits, _ = scheduler._infer_live_layer_cache_types()
    assert types == ["TurboQuantKVCache", "KVCache"]


def test_scheduler_keeps_nonstandard_cache_guards(scheduler):
    scheduler.model = SimpleNamespace(args=SimpleNamespace(kv_lora_rank=128))
    assert not scheduler._turboquant_eligible([KVCache()])
    scheduler._mla_model = False
    assert not scheduler._turboquant_eligible(
        [CacheList(KVCache(), ArraysCache(size=2))]
    )


def test_affine4_allows_sink_aware_fallback(scheduler):
    scheduler.model = SimpleNamespace(modules=lambda: [{"sinks": mx.zeros((4,))}])
    assert scheduler._turboquant_eligible([KVCache()])
    scheduler._turboquant_kv_scheme = "turboquant"
    assert not scheduler._turboquant_eligible([KVCache()])


@pytest.mark.parametrize("module", ["mlx_lm.models.base", "mlx_vlm.models.base"])
def test_attention_dispatch_bypasses_turboquant_codebook_kernels(module):
    apply_turboquant_attention_patch()
    dispatch = importlib.import_module(module).scaled_dot_product_attention
    cache = Affine4KVCache()
    q = mx.ones((1, 4, 3, 64))
    keys, values = cache.update_and_fetch(
        mx.ones((1, 2, 8, 64)), mx.ones((1, 2, 8, 64))
    )
    sinks = mx.zeros((4,))
    with patch.object(cache, "attention", return_value=q) as attention:
        assert dispatch(q, keys, values, cache, 0.125, "causal", sinks) is q
    attention.assert_called_once_with(
        q, keys_state=keys, values_state=values, scale=0.125, mask="causal", sinks=sinks
    )


def test_batch_rejects_same_width_different_codecs():
    affine4 = Affine4KVCache()
    affine8 = Affine8KVCache()
    turbo4 = TurboQuantKVCache(bits=4)
    turbo8 = TurboQuantKVCache(bits=8)
    for cache in (affine4, affine8, turbo4, turbo8):
        cache.update_and_fetch(mx.ones((1, 2, 8, 64)), mx.ones((1, 2, 8, 64)))
    for batch_class, caches in (
        (BatchAffine4KVCache, (affine4, turbo4)),
        (BatchAffine4KVCache, (affine4, affine8)),
        (BatchAffine8KVCache, (affine8, turbo8)),
        (BatchAffine8KVCache, (affine8, affine4)),
        (BatchTurboQuantKVCache, (turbo4, affine4)),
        (BatchTurboQuantKVCache, (turbo8, affine8)),
    ):
        with pytest.raises(ValueError, match="mixed quantization"):
            batch_class.merge(list(caches))
    a4_batch = BatchAffine4KVCache.merge([affine4])
    a8_batch = BatchAffine8KVCache.merge([affine8])
    with pytest.raises(ValueError, match="mixed quantization"):
        a4_batch.extend(a8_batch)
    with pytest.raises(ValueError, match="mixed quantization"):
        a8_batch.extend(a4_batch)


@pytest.mark.parametrize(
    "cache_class,batch_class",
    [
        (Affine4KVCache, BatchAffine4KVCache),
        (Affine8KVCache, BatchAffine8KVCache),
    ],
)
def test_singleton_cache_converts_when_another_request_joins(
    cache_class, batch_class
):
    generate = importlib.import_module("mlx_lm.generate")
    caches = [cache_class(), cache_class()]
    for cache in caches:
        cache.update_and_fetch(mx.ones((1, 2, 8, 64)), mx.ones((1, 2, 8, 64)))
    single = generate._merge_caches([[caches[0]]])
    assert single[0] is caches[0]
    joined = generate._extend_cache(single, [caches[1]])
    assert type(joined[0]) is batch_class
    assert joined[0].left_padding.shape == (2,)


def test_format_change_requires_engine_reload():
    from omlx.engine_pool import EnginePool
    from omlx.model_settings import ModelSettings

    pool = EnginePool.__new__(EnginePool)
    pool._entries = {}
    turbo = ModelSettings(turboquant_kv_enabled=True)
    affine4 = ModelSettings(
        turboquant_kv_enabled=True, turboquant_kv_scheme="affine4"
    )
    affine8 = ModelSettings(
        turboquant_kv_enabled=True,
        turboquant_kv_scheme="affine8",
        turboquant_kv_bits=8,
    )
    signatures = {
        pool._engine_runtime_signature("model", settings)
        for settings in (turbo, affine4, affine8)
    }
    assert len(signatures) == 3


@pytest.mark.parametrize("architecture", ["llama", "qwen2"])
@pytest.mark.parametrize(
    "scheme,bits,cache_class,batch_class",
    [
        ("affine4", 4, Affine4KVCache, BatchAffine4KVCache),
        ("affine8", 8, Affine8KVCache, BatchAffine8KVCache),
    ],
)
def test_real_model_prefill_convert_and_batched_decode(
    architecture, scheduler, scheme, bits, cache_class, batch_class
):
    scheduler._turboquant_kv_scheme = scheme
    scheduler._turboquant_kv_bits = bits
    module = importlib.import_module(f"mlx_lm.models.{architecture}")
    model = module.Model(
        module.ModelArgs(
            model_type=architecture,
            hidden_size=128,
            num_hidden_layers=2,
            intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=64,
        )
    )
    model.set_dtype(mx.float16)
    apply_turboquant_attention_patch()
    requests = []
    for length in (260, 256):
        caches = [KVCache(), KVCache()]
        mx.eval(model(mx.zeros((1, length), dtype=mx.int32), cache=caches))
        scheduler._apply_turboquant_kv_convert(caches)
        requests.append(caches)
    generate = importlib.import_module("mlx_lm.generate")
    batch = generate._merge_caches(requests)
    assert type(batch[0]) is batch_class
    logits = model(mx.ones((2, 1), dtype=mx.int32), cache=batch)
    mx.eval(logits)
    assert logits.shape == (2, 1, 64)
    assert bool(mx.all(mx.isfinite(logits)))
    assert type(batch[0].extract(1)) is cache_class
    assert batch[0].extract(1).offset == 257
    for cache in batch:
        cache.filter([1])
    logits = model(mx.ones((1, 3), dtype=mx.int32), cache=batch)
    mx.eval(logits)
    assert bool(mx.all(mx.isfinite(logits)))
    assert batch[0].extract(0).offset == 260


@pytest.mark.parametrize("architecture", ["llama", "qwen2", "qwen3_5"])
@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize(
    "scheme,bits,cache_class",
    [
        ("affine4", 4, Affine4KVCache),
        ("affine8", 8, Affine8KVCache),
    ],
)
def test_incremental_prefill_matches_explicit_cache_updates(
    architecture, chunked, scheme, bits, cache_class, scheduler, monkeypatch
):
    from mlx_lm.models.cache import make_prompt_cache

    module = importlib.import_module(f"mlx_lm.models.{architecture}")
    args = dict(
        model_type=architecture,
        hidden_size=256,
        num_hidden_layers=4,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=64,
    )
    if architecture == "qwen3_5":
        args.update(
            full_attention_interval=2,
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
        )
        model = module.TextModel(module.TextModelArgs(**args))
    else:
        model = module.Model(module.ModelArgs(**args))
    model.set_dtype(mx.bfloat16)
    scheduler.model = model
    scheduler._turboquant_kv_scheme = scheme
    scheduler._turboquant_kv_bits = bits
    scheduler.config.prefill_step_size = 128
    scheduler.config.paged_cache_block_size = 0
    apply_turboquant_attention_patch()
    prompt = [i % 64 for i in range(513)]
    request = Request("incremental", prompt=prompt, sampling_params=SamplingParams())
    reference = make_prompt_cache(model)
    scheduler._prepare_affine_prefill_cache(reference)
    for start in range(0, 512, 128):
        mx.eval(model(mx.array([prompt[start : start + 128]]), cache=reference))
    expected = model(mx.array([prompt[-1:]]), cache=reference)
    mx.eval(expected)
    updates = []
    original = cache_class.update_and_fetch

    def record(self, keys, values):
        updates.append((self.offset, keys.shape[-2]))
        return original(self, keys, values)

    monkeypatch.setattr(cache_class, "update_and_fetch", record)
    if chunked:
        state = scheduler._begin_prefill(request, prompt, None)
        while not scheduler._step_prefill_chunk(state):
            pass
        caches, last = state.cache, state.last_token
    else:
        caches, last = scheduler._do_external_prefill(request, prompt, None)
    packed = [cache for cache in caches if type(cache) is cache_class]
    assert packed
    assert all(cache.offset == 512 for cache in packed)
    assert {offset for offset, _ in updates} == {0, 128, 256, 384}
    assert all(count == 128 for _, count in updates)
    if architecture == "qwen3_5":
        assert [type(cache) for cache in caches] == [
            ArraysCache,
            cache_class,
            ArraysCache,
            KVCache,
        ]
    actual = model(mx.array([last]), cache=caches)
    mx.eval(actual)
    assert mx.all(mx.isfinite(actual)).item()
    assert mx.array_equal(actual, expected).item()
