# SPDX-License-Identifier: Apache-2.0
"""CPU checks for accepted-prefix projection reference ownership."""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from tests.glm5_split_k_reference import (
    Phase,
    ProjectionReference,
    ReferenceRejectedError,
)
from tests.glm5_split_k_trace import KernelTrace


def test_kernel_trace_preserves_inputs_and_returned_objects(monkeypatch, tmp_path):
    phase = Phase(0, "verify", 2, (1, 2))
    owner = SimpleNamespace(active=phase, mode="batch", layer=3, row=None)
    outputs = [
        np.arange(4 * 16, dtype=np.float32).reshape(4, 1, 16),
        np.arange(4 * 16, dtype=np.float32).reshape(4, 1, 16),
        np.arange(4 * 16 * 128, dtype=np.float32).reshape(4, 1, 16, 128),
    ]
    seen = {}

    def original_factory(**definition):
        seen["definition"] = definition

        def kernel(**arguments):
            seen["arguments"] = arguments
            return outputs

        return kernel

    mx = SimpleNamespace(
        fast=SimpleNamespace(metal_kernel=original_factory),
        bfloat16=object(),
        float16=np.float16,
        float32=np.float32,
        uint16=np.uint16,
        int32=np.int32,
        uint32=np.uint32,
        bool_=np.bool_,
        uint8=np.uint8,
        array=lambda value: np.array(value, copy=True),
    )
    kernels = SimpleNamespace(_kernels=None, _b_kernels=None)
    trace = KernelTrace(mx, owner)
    trace.install(monkeypatch, kernels)
    inputs = [
        np.arange(2 * 16 * 2 * 128, dtype=np.float32).reshape(2, 16, 2, 128),
        np.arange(2 * 8 * 128, dtype=np.float32).reshape(2, 1, 8, 128),
        np.arange(4, dtype=np.int32).reshape(2, 2, 1),
        np.array([2, 8, 1, 0, 0, 1, 8 * 128], np.int32),
        np.array([0.1], np.float32),
        np.array([[1, 3, 2], [1, 4, 1]], np.int32),
        np.zeros((0,), np.bool_),
    ]
    names = ["Q", "KV", "win", "dims", "scale_in", "rows", "attn_mask"]
    wrapped = mx.fast.metal_kernel(
        name="glm53_sparse_mla_decode_blocks_batched",
        input_names=names,
        output_names=["pm", "pl", "po"],
        source="read Q",
        header="header",
    )
    assert (
        wrapped(
            inputs=inputs,
            template=[("T", np.float32), ("H", 16), ("D", 128)],
            grid=(128, 4, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[v.shape for v in outputs],
            output_dtypes=[np.float32] * 3,
        )
        is outputs
    )
    assert seen["arguments"]["inputs"] is inputs
    assert seen["definition"]["source"] == "read Q"
    expected_kv = [inputs[1][0:1, :, 2:6].copy(), inputs[1][1:2, :, 1:7].copy()]
    expected_q = [inputs[0][0:1, :, :1].copy(), inputs[0][1:2, :, :2].copy()]
    expected_po = [outputs[2][0:1].copy(), outputs[2][2:4].copy()]
    inputs[0][...] = inputs[1][...] = 999
    inputs[2][...] = 999
    for value in outputs:
        value[...] = 999
    rows = trace.materialize()[0]
    assert (rows["B"], rows["S"]) == (2, 2)
    assert rows["physical_KV_batch_stride_from_dims"] == 8 * 128
    assert [r["accepted_prefix"] for r in rows["logical_rows"]] == [1, 2]
    assert [r["selected_windows"] for r in rows["logical_rows"]] == [[[0]], [[2], [3]]]
    for r in range(2):
        logical = rows["logical_rows"][r]
        for name, value in [("q", expected_q[r]), ("logical_kv", expected_kv[r])]:
            assert (
                logical[name]["raw_sha256"]
                == hashlib.sha256(value.view(np.uint32).tobytes()).hexdigest()
            )
        assert (
            logical["outputs"][2]["raw_sha256"]
            == hashlib.sha256(expected_po[r].view(np.uint32).tobytes()).hexdigest()
        )
    assert rows["call_non_tensor_kwargs"]["output_shapes"] == [
        [4, 1, 16],
        [4, 1, 16],
        [4, 1, 16, 128],
    ]
    assert rows["constructor_kwargs"]["output_names"] == ["pm", "pl", "po"]
    monkeypatch.setenv("GLM40_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "trace fixture (call)")
    trace.finish()
    assert len(list(tmp_path.glob("*.json"))) == 1
    trace.finish()
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_trace_error_preserves_original_failure(monkeypatch, tmp_path):
    trace = KernelTrace(None, None)
    monkeypatch.setenv("GLM40_TRACE_DIR", str(tmp_path))

    def broken():
        raise OSError("trace fixture")

    monkeypatch.setattr(trace, "materialize", broken)
    original = AssertionError("original numeric failure")
    trace.finish(original)
    assert "OSError" in original.__notes__[0]
    with pytest.raises(OSError, match="trace fixture"):
        trace.finish()


class Bits:
    def bf16(self, value):
        return value.dtype == np.uint16

    def tokens(self, value):
        return value.dtype == np.uint32

    def same_bits(self, left, right):
        return np.array_equal(left, right)

    same_tokens = same_bits

    def snapshot_bf16(self, value):
        return value.copy()

    snapshot_tokens = snapshot_bf16


def captured():
    owner = object()
    phase = Phase(1, "verify", 8, (2, 3))
    reference = ProjectionReference(Bits(), [(owner, 7)], [phase])
    tokens = np.arange(16, dtype=np.uint32).reshape(2, 8)
    inputs = np.arange(80, dtype=np.uint16).reshape(2, 8, 5)
    output = inputs + np.uint16(5)
    reference.begin(phase, tokens)
    with reference.scope(7):
        assert reference.project(owner, inputs, lambda *_: output) is output
    reference.seal()
    return reference, owner, phase, tokens, inputs, output


def consume(reference, owner, phase, tokens, inputs, row):
    prefix = phase.prefixes[row]
    reference.begin_single(phase, row, tokens[row : row + 1, :prefix])
    with reference.scope(7):
        result = reference.project(
            owner,
            inputs[row : row + 1, :prefix],
            lambda *_: pytest.fail("reference recomputed the projection"),
        )
    reference.finish_single()
    return result


def test_projection_reference_preserves_batch_and_only_consumes_accepted_rows():
    reference, owner, phase, tokens, inputs, output = captured()
    assert sorted(slot[0].shape[1] for slot in reference.slots.values()) == [2, 3]
    for row in range(2):
        result = consume(reference, owner, phase, tokens, inputs, row)
        np.testing.assert_array_equal(
            result, output[row : row + 1, : phase.prefixes[row]]
        )
    reference.finish()
    reference.complete()
    assert reference.captures == 1 and reference.consumptions == 2


def test_capture_is_not_mutated_by_its_original_arrays():
    reference, _, _, tokens, inputs, output = captured()
    token_snapshot = reference.batch_tokens.copy()
    source, projected = (a.copy() for a in reference.slots[7, 0])
    tokens[:] = 0
    inputs[:] = 0
    output[:] = 0
    np.testing.assert_array_equal(reference.batch_tokens, token_snapshot)
    np.testing.assert_array_equal(reference.slots[7, 0][0], source)
    np.testing.assert_array_equal(reference.slots[7, 0][1], projected)


@pytest.mark.parametrize(
    "mistake", ["cycle", "operation", "row", "tokens", "token_dtype", "tail"]
)
def test_reference_rejects_context_or_token_mismatch(mistake):
    reference, _, phase, tokens, _, _ = captured()
    row, value = 0, tokens[:1, :2].copy()
    if mistake == "cycle":
        phase = Phase(0, "verify", 8, (2, 3))
    elif mistake == "operation":
        phase = Phase(1, "decode0", 8, (2, 3))
    elif mistake == "row":
        row = 1
    elif mistake == "tokens":
        value[0, 0] ^= np.uint32(1)
    elif mistake == "token_dtype":
        value = value.astype(np.int32)
    else:
        value = tokens[:1]
    with pytest.raises(ReferenceRejectedError):
        reference.begin_single(phase, row, value)
    assert reference.mode == "sealed" and len(reference.slots) == 2


@pytest.mark.parametrize("mistake", ["bits", "dtype", "tail", "layer", "no_scope"])
def test_reference_rejects_projection_mismatch_without_consuming(mistake):
    reference, owner, phase, tokens, inputs, _ = captured()
    reference.begin_single(phase, 0, tokens[:1, :2])
    value = inputs[:1, :2].copy()
    if mistake == "bits":
        value[0, 0, 0] ^= np.uint16(1)
    elif mistake == "dtype":
        value = value.astype(np.int32)
    elif mistake == "tail":
        value = inputs[:1]
    if mistake == "no_scope":
        with pytest.raises(ReferenceRejectedError):
            reference.project(owner, value, lambda *_: None)
    elif mistake == "layer":
        with pytest.raises(ReferenceRejectedError), reference.scope(3):
            reference.project(owner, value, lambda *_: None)
    else:
        with reference.scope(7), pytest.raises(ReferenceRejectedError):
            reference.project(owner, value, lambda *_: None)
    assert len(reference.slots) == 2 and reference.consumptions == 0


def test_reference_rejects_row_capture_reuse():
    reference, owner, phase, tokens, inputs, _ = captured()
    consume(reference, owner, phase, tokens, inputs, 0)
    with pytest.raises(ReferenceRejectedError):
        reference.begin_single(phase, 0, tokens[:1, :2])
    consume(reference, owner, phase, tokens, inputs, 1)
    reference.finish()
    with pytest.raises(ReferenceRejectedError):
        reference.begin(phase, tokens)


def test_reference_rejects_incomplete_phase_and_layer_capture():
    reference, _, _, _, _, _ = captured()
    with pytest.raises(ReferenceRejectedError):
        reference.finish()
    with pytest.raises(ReferenceRejectedError):
        reference.complete()
    owner = object()
    phase = Phase(0, "verify", 8, (1, 2))
    reference = ProjectionReference(Bits(), [(owner, 3)], [phase])
    reference.begin(phase, np.zeros((2, 8), np.uint32))
    with pytest.raises(ReferenceRejectedError):
        reference.seal()


@pytest.mark.parametrize("prefixes", [(0, 1), (1, 9), (True, 1)])
def test_reference_rejects_invalid_prefix(prefixes):
    owner = object()
    phase = Phase(0, "verify", 8, prefixes)
    reference = ProjectionReference(Bits(), [(owner, 3)], [phase])
    with pytest.raises(ReferenceRejectedError):
        reference.begin(phase, np.zeros((2, 8), np.uint32))
