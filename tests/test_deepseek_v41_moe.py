"""Routing preserves row-local activation quantization and router inputs."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from omlx.custom_kernels.quant_matmul import fast as quant_fast
from omlx.patches.deepseek_v41 import a8, language
from omlx.patches.deepseek_v41.config import ModelConfig
from omlx.patches.deepseek_v41.quantization import QuantizedProjection


@pytest.mark.parametrize("length", [1, 32])
@pytest.mark.parametrize(
    "routed,shared", [(True, True), (True, False), (False, True), (False, False)]
)
def test_moe_quantization_reuse_matches_per_expert_reference(length, routed, shared):
    mx.random.seed(412)
    config = ModelConfig(
        dim=64, moe_inter_dim=64, n_routed_experts=4, n_activated_experts=2
    )
    moe = language.MoE(config)
    moe.gate.weight = mx.random.normal((4, 64))
    for expert, switched, quantizes in [
        (moe.experts, True, routed),
        (moe.shared_experts, False, shared),
    ]:
        for name in ("w1", "w3", "w2"):
            weight = (
                mx.random.normal((4, 64, 64) if switched else (64, 64)).astype(
                    mx.bfloat16
                )
                * 0.05
            )
            packed, scales, biases = mx.quantize(weight, bits=4, group_size=32)
            setattr(
                expert,
                name,
                QuantizedProjection(
                    packed, scales, 4, "affine", biases=biases, quantize_input=quantizes
                ),
            )
    x = mx.random.normal((1, length, 64)).astype(mx.bfloat16)
    # Route raw inputs and project each selected expert without prequantization.
    indices, weights = moe.gate(x, None)
    selected = moe.experts(x[..., None, None, :], indices, weights).squeeze(-2)
    expected = (
        selected.astype(mx.float32).sum(-2) + moe.shared_experts(x).astype(mx.float32)
    ).astype(x.dtype)
    actual = moe(x, None)
    repeated = moe(x, None)
    mx.eval(expected, actual, repeated)
    np.testing.assert_allclose(
        actual.astype(mx.float32), expected.astype(mx.float32), rtol=0.01, atol=0.002
    )
    np.testing.assert_array_equal(
        actual.astype(mx.float32), repeated.astype(mx.float32)
    )


def _a8_moe(a8_prefill, min_tokens=128, mx_formats=False):
    """A MoE laid out like oQ3e: routed experts affine Q3 GS64 (the A8 gather)
    and an affine Q8 shared expert (the dense A8 GEMM once tagged). With
    ``mx_formats`` it is laid out like oQ4e: MXFP4 routed experts and an MXFP8
    shared expert, both quantizing their input to FP8 on the A16 path."""
    mx.random.seed(7)
    config = ModelConfig(
        dim=256,
        moe_inter_dim=128,
        n_routed_experts=8,
        n_activated_experts=2,
        a8_prefill=a8_prefill,
        a8_prefill_min_tokens=min_tokens,
    )
    moe = language.MoE(config)
    moe.gate.weight = mx.random.normal((8, 256))
    for expert, switched in [(moe.experts, True), (moe.shared_experts, False)]:
        for name, shape in (("w1", (128, 256)), ("w3", (128, 256)), ("w2", (256, 128))):
            weight = mx.random.normal(((8,) if switched else ()) + shape) * 0.05
            if mx_formats:
                bits, mode = (4, "mxfp4") if switched else (8, "mxfp8")
                packed, scales = mx.quantize(
                    weight.astype(mx.bfloat16), bits=bits, group_size=32, mode=mode
                )
                setattr(expert, name, QuantizedProjection(packed, scales, bits, mode))
                continue
            bits = 3 if switched else 8
            packed, scales, biases = mx.quantize(
                weight.astype(mx.bfloat16), bits=bits, group_size=64
            )
            setattr(
                expert,
                name,
                QuantizedProjection(
                    packed, scales, bits, "affine", biases=biases, group_size=64
                ),
            )
    if a8_prefill:
        a8.tag_dense_a8(SimpleNamespace(layers=[moe]), min_tokens)
    return moe


def test_moe_a8_declined_by_the_kernels_keeps_the_a16_result(monkeypatch):
    monkeypatch.setattr(language, "routed_experts_a8", lambda *args: None)
    x = mx.random.normal((1, 256, 256), key=mx.random.key(3)).astype(mx.bfloat16)
    monkeypatch.setattr(language, "shared_expert_a8", lambda *args: None)
    expected, actual = _a8_moe(False)(x, None), _a8_moe(True)(x, None)
    np.testing.assert_array_equal(
        expected.astype(mx.float32), actual.astype(mx.float32)
    )


def test_tag_dense_a8_tags_mxfp8_projections():
    moe = _a8_moe(True, mx_formats=True)
    for p in (moe.shared_experts.w1, moe.shared_experts.w3, moe.shared_experts.w2):
        assert getattr(p, a8._PLAN) == (a8._MXFP8, 128)


@pytest.mark.skipif(
    not quant_fast.oq_a8_available(), reason="needs INT8 tensor-unit kernels"
)
@pytest.mark.parametrize("mx_formats", [False, True])
def test_moe_a8_prefill_runs_the_a8_gather_near_a16(mx_formats, monkeypatch):
    calls = []
    original = language.routed_experts_a8

    def spy(*args):
        out = original(*args)
        calls.append(out is not None)
        return out

    monkeypatch.setattr(language, "routed_experts_a8", spy)
    x = mx.random.normal((1, 256, 256), key=mx.random.key(3)).astype(mx.bfloat16)
    moe = _a8_moe(True, mx_formats=mx_formats)
    assert a8.shared_expert_a8(moe.shared_experts, x) is not None
    expected = _a8_moe(False, mx_formats=mx_formats)(x, None)
    actual = moe(x, None)
    assert calls == [True]
    expected, actual = expected.astype(mx.float32), actual.astype(mx.float32)
    # The MX A16 path quantizes activations to FP8 (about 4% off float), so
    # its distance from A8 is mostly that error.
    bound = 0.08 if mx_formats else 0.05
    assert float(mx.linalg.norm(actual - expected) / mx.linalg.norm(expected)) < bound


@pytest.mark.skipif(
    not quant_fast.oq_a8_available(), reason="needs INT8 tensor-unit kernels"
)
@pytest.mark.parametrize("mx_formats", [False, True])
def test_moe_a8_keeps_a16_below_min_tokens(mx_formats):
    x = mx.random.normal((1, 256, 256), key=mx.random.key(3)).astype(mx.bfloat16)
    expected = _a8_moe(False, mx_formats=mx_formats)(x, None)
    actual = _a8_moe(True, min_tokens=257, mx_formats=mx_formats)(x, None)
    np.testing.assert_array_equal(
        expected.astype(mx.float32), actual.astype(mx.float32)
    )


@pytest.mark.skipif(
    not quant_fast.oq_a8_available(), reason="needs INT8 tensor-unit kernels"
)
def test_shared_expert_a8_takes_an_mxfp8_down_projection():
    """Affine gate/up with the down projection kept at source MXFP8."""
    moe = _a8_moe(True)
    weight = mx.random.normal((256, 128), key=mx.random.key(9)) * 0.05
    packed, scales = mx.quantize(
        weight.astype(mx.bfloat16), bits=8, group_size=32, mode="mxfp8"
    )
    moe.shared_experts.w2 = QuantizedProjection(packed, scales, 8, "mxfp8")
    a8.tag_dense_a8(SimpleNamespace(layers=[moe]), 128)
    x = mx.random.normal((1, 256, 256), key=mx.random.key(3)).astype(mx.bfloat16)
    actual = a8.shared_expert_a8(moe.shared_experts, x)
    assert actual is not None
    for name in ("w1", "w3", "w2"):
        object.__setattr__(getattr(moe.shared_experts, name), a8._PLAN, None)
    expected = moe.shared_experts(x).astype(mx.float32)
    actual = actual.astype(mx.float32)
    assert float(mx.linalg.norm(actual - expected) / mx.linalg.norm(expected)) < 0.08


@pytest.mark.skipif(
    not quant_fast.oq_a8_available(), reason="needs INT8 tensor-unit kernels"
)
def test_dense_a8_pair_runs_two_mxfp8_projections():
    """The attention wq_a/wkv pair of an oQ4e layer."""
    projections = []
    for seed, rows in ((10, 128), (11, 64)):
        weight = mx.random.normal((rows, 256), key=mx.random.key(seed)) * 0.05
        packed, scales = mx.quantize(
            weight.astype(mx.bfloat16), bits=8, group_size=32, mode="mxfp8"
        )
        projections.append(QuantizedProjection(packed, scales, 8, "mxfp8"))
    a8.tag_dense_a8(SimpleNamespace(layers=projections), 128)
    x = mx.random.normal((1, 256, 256), key=mx.random.key(3)).astype(mx.bfloat16)
    pair = a8.dense_a8_pair(*projections, x)
    assert pair is not None
    for p, actual in zip(projections, pair):
        object.__setattr__(p, a8._PLAN, None)
        expected = p(x).astype(mx.float32)
        error = mx.linalg.norm(actual.astype(mx.float32) - expected)
        assert float(error / mx.linalg.norm(expected)) < 0.05


def test_a8_skips_offloaded_experts_and_mismatched_metadata():
    """Offloaded experts have no resident weights, and fp32 scales are not
    what the A8 kernels read: both stay on the A16 path."""
    x = mx.zeros((1, 256, 64), mx.bfloat16)
    flat = mx.zeros((512,), mx.int32)
    weights = mx.ones((1, 256, 2))
    offloaded = SimpleNamespace(slots=None)
    assert (
        a8.routed_experts_a8(offloaded, x, flat, mx.argsort(flat), weights, 128) is None
    )
    packed, scales, biases = mx.quantize(
        mx.zeros((128, 64), mx.bfloat16), bits=8, group_size=64
    )
    bf16 = QuantizedProjection(
        packed, scales, 8, "affine", biases=biases, group_size=64
    )
    fp32 = QuantizedProjection(
        packed,
        scales.astype(mx.float32),
        8,
        "affine",
        biases=biases.astype(mx.float32),
        group_size=64,
    )
    assert a8._affine_gs64(bf16, (8,), 2)
    assert not a8._affine_gs64(fp32, (8,), 2)


def test_expert_rejects_prequantized_input_for_unquantized_projections():
    expert = language.Expert(ModelConfig(dim=64, moe_inter_dim=64))
    with pytest.raises(ValueError, match="matching quantized projections"):
        expert(mx.zeros((1, 1, 64)), input_quantized=True)
