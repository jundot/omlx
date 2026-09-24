"""HyV3 uses plain KV ownership with a separately verified expert workspace."""

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache, make_prompt_cache
from prefill_helpers import make_request
from prefill_helpers import model_factory as model_factory

from omlx.patches.qwen35_moe_gate_up import apply_qwen35_moe_gate_up_fusion
from omlx.prefill.capabilities import inspect_prefill_model
from omlx.prefill.execution import BatchedPrefillGroup
from omlx.prefill.policy import prefill_eligibility


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_hy_capability_and_chunked_cache_parity(model_factory, quantized, fused):
    harness = model_factory("hy_v3", quantized=quantized)
    model = harness.model
    if fused:
        assert apply_qwen35_moe_gate_up_fusion(model) == 3
    contract, reason = inspect_prefill_model(model)
    assert reason == "supported"
    assert contract.geometry.experts.experts_per_token == 2
    assert contract.cache_types == (KVCache,) * 4
    assert prefill_eligibility(
        model, make_request("cold", 8), make_prompt_cache(model)
    ).eligible
    rows = [("first", [1, 2, 3, 4, 5]), ("second", [11, 12, 13, 14, 15])]
    group = BatchedPrefillGroup(model, rows, harness.stream)
    try:
        group.step(2)
        group.step(3)
        with mx.stream(harness.stream):
            for rid, tokens in rows:
                actual, expected = group.extract(rid), make_prompt_cache(model)
                model(mx.array([tokens]), cache=expected)
                mx.eval([layer.state for layer in actual + expected])
                for a, b in zip(actual, expected):
                    assert type(a) is KVCache and a.offset == b.offset == 5
                    for x, y in zip(a.keys_and_values(), b.keys_and_values()):
                        assert mx.allclose(
                            x, y, atol=harness.tolerance, rtol=harness.tolerance
                        ).item()
                x = model(mx.array([[37]]), cache=actual)
                y = model(mx.array([[37]]), cache=expected)
                assert mx.allclose(
                    x, y, atol=harness.tolerance, rtol=harness.tolerance
                ).item()
    finally:
        group.close()


def test_hy_unknown_expert_wrapper_falls_back(model_factory):
    harness = model_factory("hy_v3")
    harness.model.layers[1].mlp.switch_mlp = object()
    contract, reason = inspect_prefill_model(harness.model)
    assert contract is None and reason == "custom_execution"


def test_hy_existing_prefix_is_retained(model_factory):
    harness = model_factory("hy_v3")
    cache = make_prompt_cache(harness.model)
    with mx.stream(harness.stream):
        harness.model(mx.array([[1, 2]]), cache=cache)
        mx.eval([c.state for c in cache])
    keys = [c.keys for c in cache]
    result = prefill_eligibility(harness.model, make_request("warm", 8), cache)
    assert not result.eligible and result.reason == "prefix_cache"
    assert all(c.keys is k and c.offset == 2 for c, k in zip(cache, keys))
