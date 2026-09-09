# SPDX-License-Identifier: Apache-2.0
"""K2 compilation, prefill residency, and decode boundary checks."""

import gc
import os
from unittest.mock import Mock

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.cache import make_prompt_cache
from test_k2_horizon import small_config

from omlx.patches.k2_horizon.ane_prefill import PrefillMLP, enable_ane_prefill
from omlx.patches.k2_horizon.k2_horizon_model import Model, ModelArgs


def make_model():
    mx.random.seed(211)
    model = Model(ModelArgs.from_dict(small_config(head_dim=128)))
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model


def close(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    assert mx.all(mx.isfinite(b)).item()
    assert (
        mx.sqrt(mx.mean((a - b) ** 2) / mx.maximum(mx.mean(a * a), 1e-30)).item() < 0.02
    )


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
            setattr(
                ref,
                name,
                nn.QuantizedLinear.from_linear(
                    getattr(ref, name), group_size=64, bits=bits
                ),
            )
    weights = [getattr(ref, n).weight for n in ("gate_proj", "up_proj", "down_proj")]
    split = PrefillMLP(ref, cut=64, width=32)
    x = mx.random.normal((1, rows, 64)).astype(dtype)
    target = ref(x)
    before = split(x)
    mx.eval(target, before)
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
        a is getattr(ref, n).weight
        for a, n in zip(weights, ("gate_proj", "up_proj", "down_proj"))
    )
    assert mx.array_equal(target, ref(x)).item()


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
        (layer.mlp.gate.weight, layer.self_attn.v_router.weight)
        for layer in model.layers[1:]
    ]
    prefill = enable_ane_prefill(model, fraction=0.5, width=32)
    cache = make_prompt_cache(model)
    fast.qwen35_ane_profile_set_enabled(True)
    fast.qwen35_ane_profile_reset()
    try:
        with mx.stream(mx.new_stream(mx.gpu)):
            prefill(ids, cache=cache)
        ops = fast.qwen35_ane_profile_snapshot()["mlp"]["operations"]
        assert ops == 4
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
@pytest.mark.parametrize("prefix", [0, 16])
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
        assert ops == 4
        logits = model(mx.array([last]), cache=cache)
        assert mx.all(mx.isfinite(logits)).item()
        assert all(c.offset == 65 for c in cache)
        assert fast.qwen35_ane_profile_snapshot()["mlp"]["operations"] == ops
    finally:
        fast.qwen35_ane_profile_set_enabled(False)
