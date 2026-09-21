# SPDX-License-Identifier: Apache-2.0
"""Community mlx_lm affine DeepSeek V4.1 sources: loading and expert offload.

These checkpoints declare their format in a top-level ``quantization`` dict
instead of an ``omlx_deepseek_v41`` spec, are packed per expert with a U32
weight plus float scales and biases, and leave a mixed set of projections
dense. ``write_affine_checkpoint`` reproduces that layout.
"""

import json

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from test_deepseek_v41 import write_affine_checkpoint


@pytest.mark.parametrize("index_order", ["sorted", "insertion"])
def test_source_affine_checkpoint_loads_whatever_the_index_order(tmp_path, index_order):
    """The loader must not depend on the order the index lists tensors in.

    `mlx_lm.utils.save_model` writes the weight map sorted, which puts every
    `<module>.biases` ahead of its `<module>.weight`; a loader that only skips
    metadata after reading its weight raises
    "Unexpected target tensor shape: language_model.embed.biases" on exactly
    the checkpoints this module exists for.
    """
    from omlx.patches.deepseek_v41.loading import load
    from omlx.patches.deepseek_v41.quantization import QuantizedProjection

    source, _ = write_affine_checkpoint(tmp_path, vision=False, index_order=index_order)
    model, _ = load(source)
    try:
        packed = model.language_model.layers[0].attn.wq_a
        assert isinstance(packed, QuantizedProjection)
        assert (packed.bits, packed.mode, packed.group_size) == (2, "affine", 64)
    finally:
        model.close()


def test_sorted_index_puts_metadata_before_its_weight(tmp_path):
    """Keep the fixture on the realistic order the test above depends on."""
    source, _ = write_affine_checkpoint(tmp_path, vision=False)
    keys = list(
        json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    )
    assert keys == sorted(keys)
    assert keys.index("head.biases") < keys.index("head.weight")


def test_source_affine_checkpoint_loads_packed_and_declared_dense(tmp_path):
    """Packed projections become QuantizedProjection, declared-dense ones do not."""
    from omlx.patches.deepseek_v41.loading import load
    from omlx.patches.deepseek_v41.quantization import QuantizedProjection

    source, _ = write_affine_checkpoint(tmp_path, vision=False)
    model, _ = load(source)
    try:
        layer = model.language_model.layers[0]
        packed = layer.attn.wq_a
        assert isinstance(packed, QuantizedProjection)
        assert (packed.bits, packed.mode, packed.group_size) == (2, "affine", 64)
        assert packed.weight.dtype == mx.uint32
        assert packed.scales.dtype == mx.bfloat16
        assert packed.biases.dtype == mx.bfloat16
        # A third-party conversion quantizes weights only.
        assert packed.quantize_input is False
        # The router keeps its own dense bias, so it must stay unpacked.
        assert not isinstance(layer.ffn.gate, QuantizedProjection)
        assert hasattr(layer.ffn.gate, "bias")
        assert type(layer.attn.wo_a).__name__ == "Linear"
    finally:
        model.close()


def test_source_affine_biased_projection_stays_dense_with_its_bias(tmp_path):
    """A packed projection that carries a dense bias has to materialize."""
    from omlx.patches.deepseek_v41.loading import load
    from omlx.patches.deepseek_v41.quantization import QuantizedProjection

    source, _ = write_affine_checkpoint(tmp_path, vision=True)
    tensors = dict(mx.load(str(source / "model.safetensors")))
    model, _ = load(source)
    try:
        attn = model.vision.blocks[0].attn
        # The checkpoint packs wqkv and keeps wqkv.bias dense beside it.
        assert not isinstance(attn.wqkv, QuantizedProjection)
        # The bias survives the module materializing densely.
        np.testing.assert_array_equal(
            np.asarray(attn.wqkv.bias.astype(mx.float32)),
            np.asarray(tensors["vision.blocks.0.attn.wqkv.bias"].astype(mx.float32)),
        )
        expected = mx.dequantize(
            tensors["vision.blocks.0.attn.wqkv.weight"],
            tensors["vision.blocks.0.attn.wqkv.scales"],
            tensors["vision.blocks.0.attn.wqkv.biases"],
            group_size=64,
            bits=2,
            mode="affine",
        ).astype(mx.bfloat16)
        np.testing.assert_array_equal(
            np.asarray(attn.wqkv.weight.astype(mx.float32)),
            np.asarray(expected.astype(mx.float32)),
        )
    finally:
        model.close()


def test_source_affine_forced_dense_projection_dequantizes_exactly(tmp_path):
    """head/embed have to stay dense, so they must dequantize exactly."""
    from omlx.patches.deepseek_v41.loading import load

    source, _ = write_affine_checkpoint(tmp_path, vision=False)
    tensors = dict(mx.load(str(source / "model.safetensors")))
    model, _ = load(source)
    try:
        expected = mx.dequantize(
            tensors["head.weight"],
            tensors["head.scales"],
            tensors["head.biases"],
            group_size=64,
            bits=2,
            mode="affine",
        ).astype(mx.bfloat16)
        got = model.language_model.head.weight
        assert got.dtype == mx.bfloat16
        np.testing.assert_array_equal(
            np.asarray(got.astype(mx.float32)),
            np.asarray(expected.astype(mx.float32)),
        )
    finally:
        model.close()


def test_source_affine_experts_are_offload_eligible(tmp_path):
    """The offload plan accepts affine source experts and their metadata."""
    from omlx.patches.deepseek_v41.config import ModelConfig
    from omlx.patches.deepseek_v41.moe_offload import (
        ExpertOffloadPlan,
        estimate_expert_savings,
    )
    from omlx.patches.moe_offload_compat import moe_offload_compatibility

    source, _ = write_affine_checkpoint(tmp_path, vision=False)
    assert moe_offload_compatibility(source) == (True, "")
    assert estimate_expert_savings(source, 0.5) > 0

    raw = json.loads((source / "config.json").read_text())
    mapping = json.loads((source / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    plan = ExpertOffloadPlan(source, raw, mapping, ModelConfig.from_dict(raw), 0.5)
    try:
        # Every expert projection contributes weight, scales and biases.
        assert plan.full_bytes == plan.expert_bytes * plan.count * len(plan.layers)
        assert plan.layers["language_model.layers.0.ffn.experts"]["w1"] == {
            "bits": 2,
            "group_size": 64,
            "mode": "affine",
            "quantize_input": False,
        }
    finally:
        plan.close()


def test_source_affine_convert_round_trip_matches_direct_load(tmp_path):
    """convert() publishes the same affine spec the loader reads directly."""
    from omlx.patches.deepseek_v41.convert import convert
    from omlx.patches.deepseek_v41.loading import load

    source, _ = write_affine_checkpoint(tmp_path, vision=False)
    target = tmp_path / "converted"
    convert(source, target)

    specs = json.loads((target / "config.json").read_text())["omlx_deepseek_v41"][
        "quantized_modules"
    ]
    for name in (
        "language_model.layers.0.attn.wq_a",
        "language_model.layers.0.ffn.experts.w1",
    ):
        assert specs[name] == {
            "bits": 2,
            "group_size": 64,
            "mode": "affine",
            "quantize_input": False,
        }

    direct, _ = load(source)
    converted, _ = load(target)
    try:
        direct_params = dict(tree_flatten(direct.parameters()))
        converted_params = dict(tree_flatten(converted.parameters()))
        assert direct_params.keys() == converted_params.keys()
        for name, value in direct_params.items():
            other = converted_params[name]
            assert value.shape == other.shape, name
            assert value.dtype == other.dtype, name
            np.testing.assert_array_equal(
                np.asarray(value.astype(mx.float32)),
                np.asarray(other.astype(mx.float32)),
                err_msg=name,
            )
    finally:
        direct.close()
        converted.close()


def test_source_quantization_spec_reads_declarations():
    """The declared format decides bits/group/mode; unreadable ones raise."""
    from omlx.patches.deepseek_v41.convert import source_quantization_spec

    base = {"bits": 2, "group_size": 64, "mode": "affine"}
    resolved = {
        "bits": 2,
        "group_size": 64,
        "mode": "affine",
        "quantize_input": False,
    }
    assert source_quantization_spec({}, "layers.0.attn.wq_a") is None
    assert source_quantization_spec({"quantization": base}, "any") == resolved
    assert source_quantization_spec({"quantization_config": base}, "any") == resolved
    # A per-module override wins, under either key namespace.
    for key in ("layers.0.attn.wq_a", "language_model.layers.0.attn.wq_a"):
        config = {"quantization": {**base, key: {"bits": 4}}}
        assert source_quantization_spec(config, "layers.0.attn.wq_a")["bits"] == 4
    assert (
        source_quantization_spec(
            {"quantization": {**base, "model.layers.0.attn.wq_a": {"bits": 8}}},
            "layers.0.attn.wq_a",
        )["bits"]
        == 8
    )
    # A declared-dense module resolves to no format at all.
    declared_dense = {"quantization": {**base, "layers.0.ffn.gate": False}}
    assert source_quantization_spec(declared_dense, "layers.0.ffn.gate") is None
    # Declarations the loader cannot read fail loudly instead of defaulting.
    for unreadable in (
        {"quantization": {"quant_method": "fp8"}},
        {"quantization": {"bits": 2, "mode": "affine"}},
        {"quantization": {**base, "mode": "mxfp4"}},
        {"quantization": {**base, "bits": True}},
    ):
        with pytest.raises(ValueError):
            source_quantization_spec(unreadable, "any")
    # A section that is not a dict is not a declaration at all.
    assert source_quantization_spec({"quantization": "mlx"}, "any") is None


def test_source_engram_table_reports_affine_metadata():
    """Affine Engram tables carry their format and stay on one shard."""
    from omlx.patches.deepseek_v41.convert import source_engram_tables

    config = {"quantization": {"bits": 2, "group_size": 64, "mode": "affine"}}
    mapping = {
        "layers.1.engram.embed.weight": "a.safetensors",
        "layers.1.engram.embed.scales": "a.safetensors",
        "layers.1.engram.embed.biases": "a.safetensors",
    }
    table = source_engram_tables(mapping, config)[
        "language_model.layers.1.engram.embed"
    ]
    assert table["bias_key"] == "layers.1.engram.embed.biases"
    assert table["scale_key"] == "layers.1.engram.embed.scales"
    assert (table["bits"], table["group_size"]) == (2, 64)

    split = dict(mapping, **{"layers.1.engram.embed.biases": "b.safetensors"})
    with pytest.raises(ValueError):
        source_engram_tables(split, config)
    with pytest.raises(ValueError):
        source_engram_tables(mapping, {})

    plain = {"layers.1.engram.embed.weight": "a.safetensors"}
    assert (
        source_engram_tables(plain, {})["language_model.layers.1.engram.embed"]["bits"]
        is None
    )


@pytest.mark.parametrize("bits", [2, 3, 4, 6, 8])
def test_affine_dequantize_matches_quantized_matmul(bits):
    """force_dense relies on mx.dequantize agreeing with the packed matmul.

    They agree to quantization rounding rather than bit for bit: the fused
    kernel and the dequantize-then-multiply path differ by up to ~1e-3 on
    bf16 inputs, and the exact difference is machine-dependent, so this pins
    a tolerance rather than equality.
    """
    mx.random.seed(11)
    weight = (mx.random.normal((128, 256)) * 0.05).astype(mx.bfloat16)
    x = (mx.random.normal((1, 4, 256)) * 0.05).astype(mx.bfloat16)
    packed, scales, biases = mx.quantize(
        weight, group_size=64, bits=bits, mode="affine"
    )
    quantized = mx.quantized_matmul(
        x,
        packed,
        scales,
        biases,
        transpose=True,
        group_size=64,
        bits=bits,
        mode="affine",
    )
    dense = (
        x
        @ mx.dequantize(
            packed, scales, biases, group_size=64, bits=bits, mode="affine"
        ).T
    )
    mx.eval(quantized, dense)
    got = np.asarray(quantized.astype(mx.float32))
    want = np.asarray(dense.astype(mx.float32))
    np.testing.assert_allclose(got, want, rtol=1e-2, atol=2e-3)
