# SPDX-License-Identifier: Apache-2.0
"""CPU guards for same-input KDA fused projection references."""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_vlm.models.cache import ArraysCache

from tests import test_glm5_split_k_attention as source
from tests.glm5_split_k_reference import (
    FusedProjectionReference,
    MLXBits,
    Phase,
    ReferenceRejectedError,
)


def captured():
    owner = object()
    phase = Phase(1, "verify", 8, (2, 3))
    ref = FusedProjectionReference(MLXBits(mx), [(owner, 2)], [phase])
    tokens = mx.arange(16).reshape(2, 8).astype(mx.uint32)
    inputs = mx.arange(80).reshape(2, 8, 5).astype(mx.bfloat16)
    outputs = [inputs + i for i in range(len(ref.PARTS))]
    ref.begin(phase, tokens)
    with ref.scope(2):
        assert ref.project_fused(owner, inputs, lambda *_: outputs) is outputs
    ref.seal()
    return ref, owner, phase, tokens, inputs, outputs


def test_fused_batch_objects_preserved_and_reference_never_recomputes():
    ref, owner, phase, tokens, inputs, outputs = captured()
    for row, prefix in enumerate(phase.prefixes):
        ref.begin_single(phase, row, tokens[row : row + 1, :prefix])
        with ref.scope(2):
            result = ref.project_fused(
                owner,
                inputs[row : row + 1, :prefix],
                lambda *_: pytest.fail("reference GEMM ran"),
            )
        for actual, expected in zip(result, outputs):
            assert mx.array_equal(
                actual.view(mx.uint16), expected[row : row + 1, :prefix].view(mx.uint16)
            ).item()
        ref.finish_single()
    ref.finish()
    ref.complete()
    assert ref.captures == 1 and ref.consumptions == 2
    assert ref.records[0]["actual_batch_object_preserved"]
    assert all(r["uint16_input_identical"] for r in ref.records[1:])


def test_actual_kda_fused_projection_call_and_return_shape():
    host = source._host()
    owner = host.model.layers[2].self_attn
    phase = Phase(1, "verify", 8, (2, 3))
    ref = FusedProjectionReference(MLXBits(mx), [(owner, 2)], [phase])
    tokens = mx.arange(16).reshape(2, 8).astype(mx.uint32)
    value = mx.ones((2, 8, owner.hidden_size), mx.bfloat16)
    observed = {}

    def original(target, inputs):
        observed["actual"] = target._fused_in_proj(inputs)
        return observed["actual"]

    ref.begin(phase, tokens)
    with ref.scope(2):
        assert ref.project_fused(owner, value, original) is observed["actual"]
    ref.seal()
    for row, prefix in enumerate(phase.prefixes):
        ref.begin_single(phase, row, tokens[row : row + 1, :prefix])
        with ref.scope(2):
            returned = ref.project_fused(
                owner,
                value[row : row + 1, :prefix],
                lambda *_: pytest.fail("single GEMM recomputed"),
            )
        assert len(returned) == len(observed["actual"])
        assert all(p.shape[:2] == (1, prefix) for p in returned)
        ref.finish_single()
    ref.finish()
    ref.complete()


@pytest.mark.parametrize("case", ["bits", "tail", "dtype", "layer", "no_scope"])
def test_fused_input_guard_rejects_before_consuming(case):
    ref, owner, phase, tokens, inputs, _ = captured()
    ref.begin_single(phase, 0, tokens[:1, :2])
    value = inputs[:1, :2]
    if case == "bits":
        value = (value.view(mx.uint16) ^ mx.ones(value.shape, mx.uint16)).view(
            mx.bfloat16
        )
    elif case == "tail":
        value = inputs[:1]
    elif case == "dtype":
        value = value.astype(mx.float32)
    if case == "no_scope":
        with pytest.raises(ReferenceRejectedError):
            ref.project_fused(owner, value, lambda *_: None)
    else:
        with (
            pytest.raises(ReferenceRejectedError),
            ref.scope(3 if case == "layer" else 2),
        ):
            ref.project_fused(owner, value, lambda *_: None)
    assert ref.consumptions == 0 and len(ref.slots) == 2


@pytest.mark.parametrize("case", ["cycle", "row", "tail", "token_bits", "reuse"])
def test_fused_phase_prefix_and_reuse_guards(case):
    ref, owner, phase, tokens, inputs, _ = captured()
    if case == "reuse":
        ref.begin_single(phase, 0, tokens[:1, :2])
        with ref.scope(2):
            ref.project_fused(owner, inputs[:1, :2], lambda *_: None)
        ref.finish_single()
    if case == "cycle":
        phase = Phase(0, "verify", 8, (2, 3))
    row = 1 if case == "row" else 0
    value = tokens[:1] if case == "tail" else tokens[:1, :2]
    if case == "token_bits":
        value = value ^ mx.ones(value.shape, mx.uint32)
    with pytest.raises(ReferenceRejectedError):
        ref.begin_single(phase, row, value)


def test_actual_history_injection_and_exact_coordinate(monkeypatch):
    caches = [ArraysCache(2) for _ in range(3)]
    for cache in caches:
        cache[0] = mx.ones((2, 3, 384), mx.bfloat16)
        cache[1] = mx.zeros((2, 4, 32, 32), mx.float32)
    rows = [[cache.extract(row) for cache in caches] for row in range(2)]
    trace = SimpleNamespace(state_checks=[])
    monkeypatch.setattr(source.language, "_glm40_test_trace", trace, raising=False)
    mutation = source._inject_conv_history_fault(caches)
    assert mutation["changed_word_count"] == 1
    with pytest.raises(source.ContinuityMismatchError, match="cache state"):
        source._assert_cache_rows(caches, rows, 1, "verify", overlap=False)
    failure = trace.state_checks[-1]
    assert (failure["layer"], failure["row"], failure["field"]) == (2, 1, "0")
    assert failure["first_coordinate"] == [0, 2, 0]
    assert failure["actual_bits"] ^ failure["reference_bits"] == 1
