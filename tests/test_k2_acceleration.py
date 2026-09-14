# SPDX-License-Identifier: Apache-2.0
"""K2 compilation, prefill residency, and decode boundary checks."""

import gc
import os
from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.cache import make_prompt_cache

from omlx.patches.k2_horizon.ane_prefill import PrefillMLP, enable_ane_prefill
from omlx.patches.k2_horizon.compiled import install_compiled_blocks, make_regions
from omlx.patches.k2_horizon.k2_horizon_model import Model, ModelArgs
from omlx.patches.k2_horizon.uno_adapter import ConditionalLoRALinear, TARGETS
from omlx.patches.k2_horizon.uno_decode import UnoDecoder
from test_k2_horizon import small_config


def make_model():
    mx.random.seed(211)
    model = Model(ModelArgs.from_dict(small_config(head_dim=128)))
    model.set_dtype(mx.bfloat16)
    for layer in model.layers:
        for scope, names in TARGETS.items():
            owner = getattr(layer, scope)
            for name in names:
                linear = getattr(owner, name)
                n, k = linear.weight.shape
                a = (mx.random.normal((4, k)) * 0.02).astype(mx.bfloat16)
                b = (mx.random.normal((n, 4)) * 0.02).astype(mx.bfloat16)
                setattr(owner, name, ConditionalLoRALinear(linear, a, b, 2.0))
    model._uno_adapter_loaded = True
    mx.eval(model.parameters())
    return model


def close(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    assert mx.all(mx.isfinite(b)).item()
    assert (
        mx.sqrt(mx.mean((a - b) ** 2) / mx.maximum(mx.mean(a * a), 1e-30)).item() < 0.02
    )


@pytest.mark.parametrize("conditional", [False, True])
def test_compiled_cache_rollback_and_path_alternation(conditional):
    model = make_model()
    cache = make_prompt_cache(model)
    model(mx.array([[2, 3, 5, 7]]), cache=cache)
    mx.eval([c.state for c in cache])
    clone = lambda: [type(c).from_state(c.state, c.meta_state) for c in cache]
    ids = mx.array([[11, 13, 17]])
    mask = mx.array([[0.0, 1.0, 1.0]]) if conditional else None
    rc = clone()
    expected = model(ids, cache=rc, lora_mask=mask)
    mx.eval(expected)
    weights = model.parameters()
    install_compiled_blocks(model)
    ac = clone()
    actual = model(ids, cache=ac, lora_mask=mask)
    mx.eval(actual)
    close(expected, actual)
    for r, c in zip(rc, ac):
        assert r.offset == c.offset == 7
        for a, b in zip(r.state, c.state):
            close(a, b)
        c.trim(3)
    assert mx.array_equal(actual, model(ids, cache=ac, lora_mask=mask)).item()
    assert mx.all(mx.isfinite(model(ids, cache=clone()))).item()
    assert weights.keys() == model.parameters().keys()


def test_compiled_default_rope_dynamic_long_offsets():
    model = make_model()
    layer = model.layers[0]
    attn = layer.self_attn
    regions = make_regions(layer)
    x = mx.random.normal((1, 3, 64)).astype(mx.bfloat16)
    q = (
        attn.q_proj(layer.input_layernorm(x))
        .reshape(1, 3, 4, 128)
        .transpose(0, 2, 1, 3)
    )
    previous = None
    for offset in (0, 2048, 4096, 8192, 8193):
        actual, _, _ = regions["pre", False](x, mx.array(offset, dtype=mx.int32))
        close(attn.rope(q, offset=offset), actual)
        if previous is not None:
            assert not mx.array_equal(actual, previous).item()
        previous = actual


def test_compiled_rejects_unsupported_rope_and_cache():
    model = make_model()
    model.args.rope_head_dim = 64
    with pytest.raises(ValueError, match="full-head"):
        install_compiled_blocks(model)
    model.args.rope_head_dim = 128
    install_compiled_blocks(model)
    cache = make_prompt_cache(model)
    model(mx.array([[2, 3]]), cache=cache)
    mx.eval([c.state for c in cache])
    cache[0].trim(1)
    with pytest.raises(ValueError, match="aligned"):
        model(mx.array([[5]]), cache=cache)
    model.model_type = "unsupported"
    with pytest.raises(ValueError, match="K2 model"):
        enable_ane_prefill(model)


@pytest.mark.parametrize("offset", [0, 2])
@pytest.mark.parametrize("chunked", [False, True])
def test_scheduler_prefill_excludes_seed_drafts_and_verification(
    mock_tokenizer, offset, chunked
):
    from omlx.request import Request, SamplingParams
    from omlx.scheduler import Scheduler, SchedulerConfig

    model = make_model()
    cache = make_prompt_cache(model)
    prompt = [2, 3, 5, 7, 11, 13]
    if offset:
        model(mx.array([prompt[:offset]]), cache=cache)
        mx.eval([c.state for c in cache])
    calls = []

    def prefill(ids, *, cache):
        calls.append(ids.tolist()[0])
        return model(ids, cache=cache)

    model._omlx_prefill = prefill
    scheduler = Scheduler(
        model,
        mock_tokenizer,
        SchedulerConfig(prefill_step_size=2, paged_cache_block_size=0),
    )
    request = Request(
        request_id="prefill", prompt=prompt, sampling_params=SamplingParams()
    )
    request.prompt_token_ids, request.num_prompt_tokens = prompt, len(prompt)
    request.cached_tokens = offset
    scheduler.requests[request.request_id] = request
    try:
        if chunked:
            state = scheduler._begin_prefill(request, prompt[offset:], cache)
            while not scheduler._step_prefill_chunk(state):
                pass
            cache, last = state.cache, state.last_token
        else:
            cache, last = scheduler._do_external_prefill(
                request, prompt[offset:], cache
            )
        assert sum(calls, []) == prompt[offset:-1]
        assert last == prompt[-1:]
        assert all(c.offset == len(prompt) - 1 for c in cache)
        calls.clear()
        decoder = UnoDecoder(model, eos_token_ids=[], temperature=0)
        cycle = decoder.cycle(last[0], cache=cache, frontier=len(prompt), max_tokens=12)
        assert cycle.tokens and calls == []
        assert all(c.offset == len(prompt) + len(cycle.tokens) - 1 for c in cache)
    finally:
        scheduler.shutdown()


def test_compiled_prefill_is_explicit_and_skips_final_mlp():
    model = make_model()
    install_compiled_blocks(model)
    mlp = model.layers[0].mlp
    spy = Mock(side_effect=mlp)
    model.model._prefill_mlps = (spy,)
    ids = mx.array([[2, 3, 5]])
    cache = make_prompt_cache(model)
    model.model(ids, cache=cache, prefill=True)
    mx.eval([c.state for c in cache])
    assert spy.call_count == 1
    model(ids, cache=cache)
    model(ids, cache=cache, lora_mask=mx.array([[0.0, 1.0, 1.0]]))
    mx.eval([c.state for c in cache])
    assert spy.call_count == 1
    with pytest.raises(ValueError, match="conditional"):
        model.model(ids, lora_mask=mx.ones((1, 3)), prefill=True)


@pytest.mark.skipif(os.environ.get("OMLX_TEST_K2_ANE") != "1", reason="requires ANE")
@pytest.mark.parametrize("rows", [7, 32])
@pytest.mark.parametrize("bits", [None, 4, 8])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_native_prefill_owns_weights_outputs_and_keeps_gpu_decode(rows, bits, dtype):
    model = make_model()
    model.set_dtype(dtype)
    ref = model.layers[0].mlp
    if bits is not None:
        for name in ("gate_proj", "up_proj", "down_proj"):
            wrapped = getattr(ref, name)
            wrapped.linear = nn.QuantizedLinear.from_linear(
                wrapped.linear, group_size=64, bits=bits
            )
    weights = [
        getattr(ref, n).linear.weight for n in ("gate_proj", "up_proj", "down_proj")
    ]
    split = PrefillMLP(ref, cut=64, width=32)
    x = mx.random.normal((1, rows, 64)).astype(dtype)
    mask = (mx.arange(rows)[None] % 2).astype(mx.float32)
    target, draft = ref(x), ref(x, lora_mask=mask)
    before = split(x)
    mx.eval(target, draft, before)
    close(target, before)
    gc.collect()
    mx.clear_cache()
    for _ in range(3):
        actual = split(x * mx.array(1, dtype=x.dtype))
        assert mx.array_equal(before, actual).item()
    changed = split(x * mx.array(2, dtype=x.dtype))
    mx.eval(changed)
    assert not mx.array_equal(before, changed).item()
    assert mx.array_equal(before, split(x)).item()
    assert all(
        a is getattr(ref, n).linear.weight
        for a, n in zip(weights, ("gate_proj", "up_proj", "down_proj"))
    )
    assert mx.array_equal(target, ref(x)).item()
    assert mx.array_equal(draft, ref(x, lora_mask=mask)).item()


@pytest.mark.skipif(os.environ.get("OMLX_TEST_K2_ANE") != "1", reason="requires ANE")
def test_mova_prefill_preserves_routes_and_gpu_decode():
    from omlx.custom_kernels.qwen35_prefill import fast

    mx.random.seed(51)
    config = small_config(
        num_hidden_layers=3,
        mlp_only_layers=[0],
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        moe_intermediate_size=128,
        mova_num_experts=4,
        mova_num_experts_per_tok=2,
        attention_gate_func="softplus",
        head_dim=128,
    )
    model = Model(ModelArgs.from_dict(config))
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    ids = mx.array([[i % 100 for i in range(39)]])
    reference_cache = make_prompt_cache(model)
    model(ids, cache=reference_cache)
    mx.eval([c.state for c in reference_cache])
    expected = model(mx.array([[43]]), cache=reference_cache)
    mx.eval(expected)
    routers = [
        (l.mlp.gate.weight, l.self_attn.v_router.weight) for l in model.layers[1:]
    ]
    prefill = enable_ane_prefill(model, fraction=0.5, width=32)
    cache = make_prompt_cache(model)
    fast.qwen35_ane_profile_set_enabled(True)
    fast.qwen35_ane_profile_reset()
    try:
        with mx.stream(mx.new_stream(mx.gpu)):
            prefill(ids, cache=cache)
        ops = fast.qwen35_ane_profile_snapshot()["mlp"]["operations"]
        assert ops == 2
        assert all(c.offset == 39 for c in cache)
        actual = model(mx.array([[43]]), cache=cache)
        mx.eval(actual)
        close(expected, actual)
        assert fast.qwen35_ane_profile_snapshot()["mlp"]["operations"] == ops
        assert all(c.offset == 40 for c in cache)
        for (a, b), layer in zip(routers, model.layers[1:]):
            assert a is layer.mlp.gate.weight and b is layer.self_attn.v_router.weight
    finally:
        fast.qwen35_ane_profile_set_enabled(False)
    program = model.layers[0].mlp._omlx_ane_prefill
    finish = program.finish
    program.finish = Mock(side_effect=RuntimeError("injected failure"))
    with pytest.raises(RuntimeError, match="injected"):
        prefill(ids, cache=make_prompt_cache(model))
    assert not program.active
    program.finish = finish
    assert mx.all(
        mx.isfinite(model(mx.array([[43]]), cache=make_prompt_cache(model)))
    ).item()


@pytest.mark.skipif(os.environ.get("OMLX_TEST_K2_ANE") != "1", reason="requires ANE")
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("rows,tiles", [(30, 0), (31, 1), (32, 1), (63, 2)])
def test_ane_near_full_tail_preserves_cache_and_decode(compiled, rows, tiles):
    from omlx.custom_kernels.qwen35_prefill import fast

    model = make_model()
    ids = mx.array([[i % 100 for i in range(rows)]])
    reference_cache = make_prompt_cache(model)
    model(ids, cache=reference_cache)
    mx.eval([c.state for c in reference_cache])
    expected = model(mx.array([[43]]), cache=reference_cache)
    mx.eval(expected)
    if compiled:
        install_compiled_blocks(model)
    prefill = enable_ane_prefill(model, fraction=0.5, width=32)
    cache = make_prompt_cache(model)
    fast.qwen35_ane_profile_set_enabled(True)
    fast.qwen35_ane_profile_reset()
    try:
        prefill(ids, cache=cache)
        ops = fast.qwen35_ane_profile_snapshot()["mlp"]["operations"]
        assert ops == tiles * (len(model.layers) - 1)
        assert all(c.offset == rows for c in cache)
        actual = model(mx.array([[43]]), cache=cache)
        mx.eval(actual)
        close(expected, actual)
        assert all(c.offset == rows + 1 for c in cache)
        for actual_cache, ref_cache in zip(cache, reference_cache):
            for a, b in zip(actual_cache.state, ref_cache.state):
                close(b, a)
            actual_cache.trim(1)
        repeated = model(mx.array([[43]]), cache=cache)
        assert mx.array_equal(actual, repeated).item()
        assert fast.qwen35_ane_profile_snapshot()["mlp"]["operations"] == ops
        assert all(not l.mlp._omlx_ane_prefill.active for l in model.layers[:-1])
    finally:
        fast.qwen35_ane_profile_set_enabled(False)


def test_family_partitions_use_checkpoint_dimensions():
    from omlx.patches.k2_horizon.ane_prefill import (
        partition_channels,
        prefill_memory_reservation,
    )

    assert partition_channels(12288, 1 / 3) == 4096
    assert partition_channels(26624, 1 / 3) == 8832
    assert partition_channels(768, 1) == 768
    assert prefill_memory_reservation(
        dict(hidden_size=5120, intermediate_size=26624, num_hidden_layers=64)
    ) > prefill_memory_reservation(
        dict(hidden_size=4096, intermediate_size=12288, num_hidden_layers=36)
    )


@pytest.mark.skipif(os.getenv("OMLX_TEST_K2_ANE") != "1", reason="requires local ANE")
@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("prefix", [0, 16, 48])
def test_mova_scheduler_prefill_and_restored_cache(mock_tokenizer, chunked, prefix):
    from omlx.custom_kernels.qwen35_prefill import fast
    from omlx.request import Request, SamplingParams
    from omlx.scheduler import Scheduler, SchedulerConfig

    model = Model(
        ModelArgs.from_dict(
            small_config(
                num_hidden_layers=3,
                mlp_only_layers=[0],
                num_experts=4,
                num_experts_per_tok=2,
                num_shared_experts=1,
                moe_intermediate_size=128,
                mova_num_experts=4,
                mova_num_experts_per_tok=2,
                attention_gate_func="softplus",
                head_dim=128,
            )
        )
    )
    model.set_dtype(mx.bfloat16)
    enable_ane_prefill(model, fraction=0.5, width=32)
    scheduler = Scheduler(
        model=model,
        tokenizer=mock_tokenizer,
        config=SchedulerConfig(
            prefill_step_size=32,
            chunked_prefill=chunked,
            paged_cache_block_size=0,
        ),
    )
    prompt = list(range(65))
    request = Request(
        request_id="mova", prompt=prompt, sampling_params=SamplingParams()
    )
    request.prompt_token_ids, request.num_prompt_tokens = prompt, len(prompt)
    request.cached_tokens = prefix
    cache = None
    if prefix:
        cache = make_prompt_cache(model)
        model(mx.array([prompt[:prefix]]), cache=cache)
        mx.eval([c.state for c in cache])
        cache = [type(c).from_state(c.state, c.meta_state) for c in cache]
    scheduler.requests[request.request_id] = request
    fast.qwen35_ane_profile_set_enabled(True)
    fast.qwen35_ane_profile_reset()
    try:
        if chunked:
            state = scheduler._begin_prefill(request, prompt[prefix:], cache)
            while not scheduler._step_prefill_chunk(state):
                pass
            cache, last = state.cache, state.last_token
        else:
            cache, last = scheduler._do_external_prefill(
                request, prompt[prefix:], cache
            )
        assert last == prompt[-1:]
        assert all(c.offset == 64 for c in cache)
        ops = fast.qwen35_ane_profile_snapshot()["mlp"]["operations"]
        assert ops == 2 * ((64 - prefix) // 32)
        logits = model(mx.array([last]), cache=cache)
        assert mx.all(mx.isfinite(logits)).item()
        assert all(c.offset == 65 for c in cache)
        assert fast.qwen35_ane_profile_snapshot()["mlp"]["operations"] == ops
    finally:
        fast.qwen35_ane_profile_set_enabled(False)


def test_ane_prefix_namespace_separates_compiled_gpu_math(monkeypatch):
    from omlx.patches.k2_horizon import ane_prefill

    monkeypatch.setattr(ane_prefill, "PrefillMLP", lambda *a, **k: SimpleNamespace())
    model = make_model()
    enable_ane_prefill(model, fraction=0.5, width=32)
    ordinary = model._omlx_k2_ane_signature
    install_compiled_blocks(model)
    enable_ane_prefill(model, fraction=0.5, width=32)
    assert model._omlx_k2_ane_signature != ordinary


@pytest.mark.skipif(os.getenv("OMLX_TEST_K2_ANE") != "1", reason="requires local ANE")
@pytest.mark.parametrize("prefix", [0, 32])
@pytest.mark.parametrize("compiled", [False, True])
def test_ane_prefill_preserves_eight_decode_rows_and_cache_after_removal(
    mock_tokenizer, monkeypatch, prefix, compiled
):
    from mlx_lm.generate import BatchGenerator, GenerationBatch
    from omlx.custom_kernels.qwen35_prefill import fast
    from omlx.request import Request, SamplingParams
    from omlx.scheduler import Scheduler, SchedulerConfig

    model = make_model()
    if compiled:
        from omlx.patches.k2_horizon.compiled import install_compiled_blocks

        install_compiled_blocks(model)
    enable_ane_prefill(model, fraction=0.5, width=32)
    scheduler = Scheduler(
        model=model,
        tokenizer=mock_tokenizer,
        config=SchedulerConfig(prefill_step_size=32, paged_cache_block_size=0),
    )
    prompts = [[(i + j) % 100 for j in range(40)] for i in range(8)]
    peak = 0
    step = GenerationBatch._step

    def observe(batch):
        nonlocal peak
        peak = max(peak, len(batch))
        return step(batch)

    monkeypatch.setattr(GenerationBatch, "_step", observe)
    batch = BatchGenerator(
        model,
        max_tokens=16,
        completion_batch_size=8,
        prefill_batch_size=1,
        sampler=lambda x: mx.argmax(x, axis=-1),
        stream=scheduler._stream,
    )
    scheduler.batch_generator = batch
    received = {i: [] for i in range(8)}
    completed = set()

    def insert(index):
        prompt = prompts[index]
        cache = make_prompt_cache(model)
        if prefix:
            model._omlx_prefill(mx.array([prompt[:prefix]]), cache=cache)
            cache = [type(c).from_state(c.state, c.meta_state) for c in cache]
        request = Request(
            request_id=str(index), prompt=prompt, sampling_params=SamplingParams()
        )
        request.prompt_token_ids, request.num_prompt_tokens = prompt, len(prompt)
        request.cached_tokens = prefix
        before = fast.qwen35_ane_profile_snapshot()["mlp"]["operations"]
        cache, last = scheduler._do_external_prefill(request, prompt[prefix:], cache)
        after = fast.qwen35_ane_profile_snapshot()["mlp"]["operations"]
        assert after - before == (0 if prefix else model._omlx_k2_ane_prefill_count)
        return batch.insert([last], caches=[cache], all_tokens=[prompt[:-1]])[0]

    fast.qwen35_ane_profile_set_enabled(True)
    fast.qwen35_ane_profile_reset()
    try:
        with mx.stream(scheduler._stream):
            assert insert(0) == 0
            joined = removed = False
            for _ in range(50):
                before = fast.qwen35_ane_profile_snapshot()["mlp"]["operations"]
                responses = batch.next_generated()
                assert fast.qwen35_ane_profile_snapshot()["mlp"]["operations"] == before
                for response in responses:
                    received[response.uid].append(response.token)
                    if response.prompt_cache is not None:
                        tokens = prompts[response.uid] + received[response.uid]
                        assert all(
                            c.offset == len(tokens) for c in response.prompt_cache
                        )
                        actual = model(mx.array([[19]]), cache=response.prompt_cache)[
                            :, -1
                        ]
                        expected = model(mx.array([tokens + [19]]))[:, -1]
                        close(expected, actual)
                        completed.add(response.uid)
                if not joined:
                    for i in range(1, 8):
                        assert insert(i) == i
                    joined = True
                if peak == 8 and not removed:
                    # Cancelling one active row must leave the other seven caches usable.
                    batch.remove([7])
                    removed = True
                if len(completed) == 7:
                    break
            assert peak == 8 and removed and completed == set(range(7))
    finally:
        fast.qwen35_ane_profile_set_enabled(False)
        scheduler.shutdown()


@pytest.mark.skipif(os.getenv("OMLX_TEST_K2_ANE") != "1", reason="requires local ANE")
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_planar_transfer_preserves_outputs_from_lazy_inputs_on_multiple_streams(
    asynchronous, reverse
):
    from omlx.custom_kernels.qwen35_prefill import fast

    model = make_model()
    model.set_dtype(mx.float16)
    ref = model.layers[0].mlp
    split = PrefillMLP(ref, cut=model.args.intermediate_size, width=32)
    expected, actual = [], []
    for i in range(8):
        with mx.stream(mx.new_stream(mx.gpu)):
            x = mx.random.normal((32, 64)).astype(mx.float16) + i / 8
            planar = mx.contiguous(
                mx.concatenate([x, mx.ones((32, 1), dtype=mx.float16)], axis=-1).T
            )
            expected.append(ref(x[None])[0].T)
            actual.append(fast._ext.ane_planar(planar, split.program))
    # The graph must retain the program and each output across surface reuse.
    del split, ref, model
    gc.collect()
    if reverse:
        expected.reverse()
        actual.reverse()
    if asynchronous:
        mx.async_eval(expected, actual)
        gc.collect()
    mx.eval(expected, actual)
    for reference, result in zip(expected, actual):
        close(reference, result)
