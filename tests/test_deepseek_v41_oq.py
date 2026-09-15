"""Independent Engram budgets and bounded affine storage round trips."""

import json
import math

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.deepseek_v41.oq import quantize_engram, source_budget
from omlx.patches.deepseek_v41.storage import DiskEngramEmbedding, EngramPrefetch


def test_shards_pack_small_groups_without_splitting_quantized_projection(tmp_path):
    from omlx.patches.deepseek_v41.sharding import ShardWriter

    writer = ShardWriter(tmp_path, max_shard_bytes=160)
    groups = [
        {"a.weight": mx.arange(20, dtype=mx.float32)},
        {
            "b.weight": mx.arange(12, dtype=mx.uint32),
            "b.scales": mx.ones((4,), mx.float32),
            "b.biases": mx.zeros((4,), mx.float32),
        },
        {"c.weight": mx.arange(20, dtype=mx.float32)},
    ]
    for group in groups:
        writer.add(group)
    mapping = writer.finish()
    assert len(set(mapping.values())) == 2
    assert (
        mapping["a.weight"]
        == mapping["b.weight"]
        == mapping["b.scales"]
        == mapping["b.biases"]
    )
    assert len(list(tmp_path.glob("*.safetensors"))) == 2
    for group in groups:
        for name, value in group.items():
            np.testing.assert_array_equal(
                mx.load(str(tmp_path / mapping[name]))[name], value
            )


def test_shard_writer_keeps_oversized_projection_intact(tmp_path):
    from omlx.patches.deepseek_v41.sharding import ShardWriter

    writer = ShardWriter(tmp_path, max_shard_bytes=16)
    writer.add({"big.weight": mx.ones((16,)), "big.scales": mx.ones((2,))})
    writer.add({"small.weight": mx.ones((1,))})
    mapping = writer.finish()
    assert mapping["big.weight"] == mapping["big.scales"]
    assert mapping["big.weight"] != mapping["small.weight"]


@pytest.mark.parametrize("switched", [False, True])
def test_affine_projection_respects_bias_and_group_size(switched):
    from omlx.patches.deepseek_v41.quantization import (
        QuantizedProjection,
        quantize_activation,
    )

    mx.random.seed(41)
    weights = mx.random.normal((3, 16, 64) if switched else (16, 64)).astype(
        mx.bfloat16
    )
    packed, scales, biases = mx.quantize(weights, bits=4, group_size=64)
    projection = QuantizedProjection(
        packed, scales, bits=4, mode="affine", biases=biases, group_size=64
    )
    x = mx.random.normal((2, 1, 64) if switched else (2, 64)).astype(mx.bfloat16)
    restored = mx.dequantize(packed, scales, biases, bits=4, group_size=64)
    if switched:
        indices = mx.array([2, 0])
        actual = projection(x, indices)
        expected = mx.matmul(quantize_activation(x), restored[indices].swapaxes(-1, -2))
    else:
        actual = projection(x)
        expected = quantize_activation(x) @ restored.T
    np.testing.assert_allclose(
        actual.astype(mx.float32), expected.astype(mx.float32), atol=0.125, rtol=0.02
    )


def test_engram_size_does_not_change_remaining_budget(tmp_path):
    reports = []
    for rows in (4, 400):
        path = tmp_path / str(rows)
        path.mkdir()
        weights = {
            "layers.0.attn.wq_a.weight": mx.ones((32, 32), mx.bfloat16),
            "layers.1.engram.embed.weight": mx.ones((rows, 32), mx.bfloat16),
        }
        mx.save_safetensors(str(path / "model.safetensors"), weights)
        reports.append(
            source_budget(
                path, {}, {k: "model.safetensors" for k in weights}, preserve_mtp=False
            )
        )
    assert reports[0]["remaining_weights"] == reports[1]["remaining_weights"]
    assert reports[0]["remaining_weights"]["effective_bpw"] == 16
    assert (
        reports[1]["engram"]["tensor_bytes"]
        == reports[0]["engram"]["tensor_bytes"] * 100
    )
    assert reports[1]["engram"]["effective_bpw"] == 5


@pytest.mark.parametrize("resident", [False, True])
@pytest.mark.parametrize("bits", [2, 3, 4, 6, 8])
def test_chunked_engram_affine_matches_mlx_with_prefetch(tmp_path, resident, bits):
    mx.random.seed(341)
    weight = mx.random.normal((17, 256)).astype(mx.bfloat16)
    mx.save_safetensors(str(tmp_path / "source.safetensors"), {"embed.weight": weight})
    result = quantize_engram(
        tmp_path,
        tmp_path / "engram.safetensors",
        {"weight_file": "source.safetensors", "weight_key": "embed.weight"},
        rows_per_chunk=3,
        bits=bits,
    )
    loaded = mx.load(str(tmp_path / "engram.safetensors"))
    expected = mx.quantize(weight, bits=bits, group_size=32)
    for key, value in zip(("weight", "scales", "biases"), expected):
        np.testing.assert_array_equal(
            loaded[key].view(mx.uint32 if key == "weight" else mx.uint16),
            value.view(mx.uint32 if key == "weight" else mx.uint16),
        )
    embed = DiskEngramEmbedding(
        tmp_path / result["weight_file"],
        result["weight_key"],
        result["scale_key"],
        bias_key=result["bias_key"],
        bits=bits,
        group_size=32,
    )
    prefetch = EngramPrefetch()
    try:
        if resident:
            embed.make_resident()
        ids = mx.array([[0, 16, 7, 7]])
        with prefetch.forward():
            prefetch.submit(embed, ids)
            actual = embed(ids)
        np.testing.assert_array_equal(
            actual.astype(mx.float32),
            mx.dequantize(*expected, bits=bits, group_size=32)[ids].astype(mx.float32),
        )
        assert embed.selected_bytes(4) == 4 * (256 * bits // 8 + 16 + 16)
    finally:
        prefetch.close()
        embed.close()


def test_engram_does_not_hide_remaining_budget_failure(tmp_path):
    from omlx.patches.deepseek_v41.oq import quantize

    source = tmp_path / "source"
    source.mkdir()
    weights = {
        "layers.0.attn.wq_a.weight": mx.ones((32, 32), mx.bfloat16),
        "layers.1.engram.embed.weight": mx.ones((400, 32), mx.bfloat16),
    }
    mx.save_safetensors(str(source / "model.safetensors"), weights)
    (source / "config.json").write_text(json.dumps({"model_type": "deepseek_v41"}))
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in weights}})
    )
    # The combined average is below 6 bpw, but the separate trunk is 16 bpw.
    with pytest.raises(ValueError, match="exceeds"):
        quantize(source, tmp_path / "output", target_bpw=6, hard_cap_bpw=6)
    assert not (tmp_path / "output").exists()


def test_oq_entrypoint_preserves_dspark_and_separate_report(tmp_path):
    from mlx.utils import tree_flatten
    from test_deepseek_v41 import write_checkpoint

    from omlx.oq import quantize_oq_streaming
    from omlx.patches.deepseek_v41.loading import load

    source, original = write_checkpoint(
        tmp_path,
        preserve_mtp=True,
        n_mtp_layers=3,
        dspark_block_size=3,
        dspark_noise_token_id=2,
        dspark_target_layer_ids=(2, 3, 4),
        dspark_n_routed_experts=2,
        dspark_n_activated_experts=1,
        dspark_markov_rank=32,
        compress_ratios=(0, 2, 2, 1, 1, 0, 0, 0),
    )
    output = tmp_path / "tiny-oQ4e-mtp"
    # The tiny fixture is FP32; use an explicit budget to test the lossless path.
    quantize_oq_streaming(
        str(source),
        str(output),
        4,
        enhanced=True,
        preserve_mtp=True,
        target_bpw=32,
        hard_cap_bpw=32,
    )
    model, _ = load(output)
    try:
        expected = dict(tree_flatten(original.parameters()))
        for name, value in tree_flatten(model.parameters()):
            np.testing.assert_array_equal(value, expected[name])
        assert len(model.language_model.mtp) == 3
        report = json.loads((output / "quantization_report.json").read_text())
        assert report["budget_scope"] == "remaining_weights_excluding_engram"
        assert report["preserved_source_draft_tensors"] > 0
        assert report["imatrix_applied_modules"] == []
    finally:
        model.close()


def test_unified_index_engram_and_mixed_shard_offload(tmp_path, monkeypatch):
    from test_deepseek_v41 import write_checkpoint

    from omlx.patches.deepseek_v41.loading import load
    from omlx.patches.deepseek_v41.oq import quantize
    from omlx.patches.deepseek_v41.residency import deepseek_v41_residency_estimate

    source, _ = write_checkpoint(
        tmp_path,
        vision=False,
        engram_layer_ids=(1, 3),
        engram_num_embeddings=(72, 204),
        engram_vocab_size=5,
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_head_dim=32,
        engram_compressed_vocab_size=64,
    )
    output = tmp_path / "quantized"
    quantize(source, output, enhanced=True, target_bpw=32, hard_cap_bpw=32)
    assert not (output / "engram").exists()
    config = json.loads((output / "config.json").read_text())
    index = json.loads((output / "model.safetensors.index.json").read_text())
    mapping = index["weight_map"]
    tables = config["omlx_deepseek_v41"]["engram_tables"]
    table_keys = {
        table[key]
        for table in tables.values()
        for key in ("weight_key", "scale_key", "bias_key")
    }
    assert table_keys <= set(mapping)
    assert all("/" not in name for name in mapping.values())
    before = deepseek_v41_residency_estimate(output)
    report = config["omlx_deepseek_v41"]["quantization_report"]
    assert before.engram_bytes == report["engram"]["tensor_bytes"]
    assert (
        before.mmap_bytes
        == int(report["remaining_weights"]["tensor_bytes"] * 1.05) + 32 * 1024**2
    )

    # Put one lookup table and normal weights in the same small fixture shard.
    first = next(iter(tables.values()))
    normal_file = next(
        filename for key, filename in mapping.items() if key not in table_keys
    )
    values = {
        **mx.load(str(output / normal_file)),
        **mx.load(str(output / first["weight_file"])),
    }
    mx.save_safetensors(str(output / "mixed.safetensors"), values)
    for key in values:
        mapping[key] = "mixed.safetensors"
    first["weight_file"] = first["scale_file"] = "mixed.safetensors"
    (output / "model.safetensors.index.json").write_text(json.dumps(index))
    (output / "config.json").write_text(json.dumps(config))
    actual_load = mx.load

    def guarded_load(path, *args, **kwargs):
        assert not str(path).endswith(
            "mixed.safetensors"
        ), "Must not load offloaded table through mx.load"
        return actual_load(path, *args, **kwargs)

    monkeypatch.setattr(mx, "load", guarded_load)
    disk, _ = load(output, engram_ssd_offload=True)
    resident, _ = load(output, engram_ssd_offload=False)
    try:
        ids = mx.array([[3, 4, 5]])
        np.testing.assert_allclose(
            disk(ids).astype(mx.float32), resident(ids).astype(mx.float32), atol=1e-6
        )
        assert deepseek_v41_residency_estimate(output) == before
    finally:
        disk.close()
        resident.close()


@pytest.mark.parametrize("bits", [2, 3, 4, 6, 8])
@pytest.mark.parametrize("rank", [2, 3])
def test_requantize_projection_uses_weighted_oq_arithmetic(bits, rank):
    from omlx.oq import _quantize_chunked
    from omlx.patches.deepseek_v41.oq import requantize_projection

    with mx.stream(mx.cpu):
        shape = (16, 64) if rank == 2 else (2, 16, 64)
        weight = mx.sin(mx.arange(math.prod(shape)).reshape(shape)).astype(mx.bfloat16)
        importance = mx.linspace(0.1, 2.0, 64)
        if rank == 3:
            importance = mx.stack([importance, importance[::-1]])
        values, spec = requantize_projection(
            {"p.weight": weight}, "p", None, bits=bits, importance=importance
        )
        reference = _quantize_chunked(weight, 64, bits, "affine", importance)
        for suffix, expected in zip(("weight", "scales", "biases"), reference):
            assert mx.array_equal(values["p." + suffix], expected).item()
        assert spec == dict(
            bits=bits, group_size=64, mode="affine", quantize_input=False
        )


@pytest.mark.parametrize(
    "source_bits,mode", [(4, "mxfp4"), (8, "mxfp8"), (6, "affine")]
)
@pytest.mark.parametrize("quantize_input", [False, True])
def test_requantize_packed_source_preserves_activation_policy(
    source_bits, mode, quantize_input
):
    from omlx.patches.deepseek_v41.oq import requantize_projection

    with mx.stream(mx.cpu):
        dense = mx.cos(mx.arange(2048).reshape(2, 16, 64)).astype(mx.bfloat16)
        packed = mx.quantize(dense, bits=source_bits, group_size=32, mode=mode)
        values = {"p." + k: v for k, v in zip(("weight", "scales", "biases"), packed)}
        source_spec = dict(
            bits=source_bits, group_size=32, mode=mode, quantize_input=quantize_input
        )
        actual, spec = requantize_projection(values, "p", source_spec, bits=3)
        restored = mx.dequantize(*packed, bits=source_bits, group_size=32, mode=mode)
        expected, _ = requantize_projection({"p.weight": restored}, "p", None, bits=3)
        for key in expected:
            assert mx.array_equal(actual[key], expected[key]).item()
        assert spec["quantize_input"] is quantize_input


@pytest.mark.parametrize("enhanced", [False, True])
def test_official_oq3_streams_affine_weights_and_unified_engram(
    tmp_path, monkeypatch, enhanced
):
    from test_deepseek_v41 import write_checkpoint

    from omlx import oq
    from omlx.patches.deepseek_v41 import oq as converter
    from omlx.patches.deepseek_v41.loading import load

    with mx.stream(mx.cpu):
        source, original = write_checkpoint(
            tmp_path,
            vision=False,
            dim=64,
            engram_layer_ids=(1, 3),
            engram_num_embeddings=(72, 204),
            engram_vocab_size=5,
            engram_max_ngram_size=4,
            engram_n_heads=2,
            engram_head_dim=32,
            engram_compressed_vocab_size=64,
        )
        name = "language_model.layers.0.attn.wq_a"
        weight = original.language_model.layers[0].attn.wq_a.weight
        received = {}

        def prepare(source, output, config, mapping, budget, **kwargs):
            received.update(kwargs)
            count = weight.size
            new_bytes = (
                budget["remaining_weights"]["tensor_bytes"]
                - weight.nbytes
                + count * 7 // 16
            )
            stats = oq.OQImatrixData(
                {
                    name: oq.OQImatrixEntry(
                        np.linspace(1, 2, 64).astype(np.float32), np.array([1])
                    )
                },
                {},
                "fixture",
            )
            return (
                {name: dict(bits=3, group_size=64, mode="affine")},
                stats,
                dict(
                    tensor_bytes=new_bytes,
                    effective_bpw=8
                    * new_bytes
                    / budget["remaining_weights"]["logical_parameters"],
                ),
            )

        monkeypatch.setattr(converter, "prepare_affine_conversion", prepare)
        output = tmp_path / "oq3"
        oq.quantize_oq_streaming(
            str(source),
            str(output),
            3,
            enhanced=enhanced,
            target_bpw=32,
            hard_cap_bpw=32,
            imatrix_num_samples=17,
            imatrix_seq_length=64,
            imatrix_strict=True,
            imatrix_reuse_cache=False,
            imatrix_cache_path="selected-cache",
            sensitivity_model_path="selected-proxy",
        )
        assert received["imatrix_num_samples"] == 17
        assert received["imatrix_seq_length"] == 64
        assert received["imatrix_strict"] is True
        assert received["imatrix_reuse_cache"] is False
        assert received["imatrix_cache_path"] == "selected-cache"
        assert received["sensitivity_model_path"] == "selected-proxy"
        config = json.loads((output / "config.json").read_text())
        spec = config["omlx_deepseek_v41"]
        assert spec["quantized_modules"][name]["bits"] == 3
        assert spec["quantized_modules"][name]["quantize_input"] is False
        assert all(table["bits"] == 3 for table in spec["engram_tables"].values())
        assert spec["engram_in_index"]
        assert not (output / "conversion.inprogress.json").exists()
        assert not (output / "engram").exists()
        report = json.loads((output / "quantization_report.json").read_text())
        assert report["imatrix_applied_modules"] == ([name] if enhanced else [])
        loaded, _ = load(output, engram_ssd_offload=True)
        try:
            projection = loaded.language_model.layers[0].attn.wq_a
            actual = projection(mx.ones((1, 64), mx.bfloat16))
            assert mx.all(mx.isfinite(actual)).item()
        finally:
            loaded.close()


def test_ui_source_filter_excludes_converted_v41():
    from omlx.oq import validate_quantizable

    source = {
        "model_type": "deepseek_v41",
        "quantization_config": {"quant_method": "fp8", "expert_dtype": "fp4"},
    }
    assert validate_quantizable(source)
    assert not validate_quantizable(
        {"model_type": "deepseek_v41", "omlx_deepseek_v41": {"version": 1}}
    )


@pytest.mark.parametrize("level", [3, 4])
def test_v41_estimate_uses_independent_engram_budget(tmp_path, level):
    from omlx.oq import estimate_bpw_and_size

    source = tmp_path / "source"
    source.mkdir()
    # Estimation reads headers only; no valid inference architecture is needed.
    with mx.stream(mx.cpu):
        weights = {
            "layers.0.attn.wq_a.weight": mx.ones((64, 64), mx.bfloat16),
            "layers.1.engram.embed.weight": mx.ones((128, 64), mx.bfloat16),
        }
        mx.save_safetensors(str(source / "model.safetensors"), weights)
    (source / "config.json").write_text(json.dumps({"model_type": "deepseek_v41"}))
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in weights}})
    )
    result = estimate_bpw_and_size(str(source), level)
    assert result["engram_size_bytes"] == 8192 * level // 8 + 8192 // 32 * 4
    assert result["remaining_size_bytes"] == (
        math.ceil(4096 * 3.7 / 8) if level == 3 else 8192
    )
    assert (
        result["output_size_bytes"]
        == result["engram_size_bytes"] + result["remaining_size_bytes"]
    )
    assert result["memory_streaming_bytes"] > 0


def test_official_oq3_reuses_signed_imatrix_and_applies_real_plan(
    tmp_path, monkeypatch
):
    from test_deepseek_v41 import write_checkpoint

    from omlx import oq
    from omlx.patches.deepseek_v41 import loading

    with mx.stream(mx.cpu):
        source, original = write_checkpoint(tmp_path, vision=False, dim=64)
        config = json.loads((source / "config.json").read_text())
        name = "language_model.layers.0.attn.wq_a"
        cache_path = tmp_path / "imatrix.npz"
        metadata = oq._source_imatrix_signature(
            source,
            config,
            num_samples=17,
            seq_length=64,
            calib_dataset=oq._OQE_CALIB_DATASET,
        )
        metadata["collection"] = {"uncalibrated_policy": "preserve_source_precision"}
        oq._save_oqe_imatrix(
            cache_path,
            {
                name: oq.OQImatrixEntry(
                    np.linspace(1, 2, 64).astype(np.float32), np.array([17])
                )
            },
            metadata,
        )

        def no_model_load(*args, **kwargs):
            raise AssertionError(
                "Signed cache and sensitivity override must avoid calibration load"
            )

        monkeypatch.setattr(loading, "load", no_model_load)
        output = tmp_path / "converted"
        oq.quantize_oq_streaming(
            str(source),
            str(output),
            3,
            enhanced=True,
            target_bpw=32,
            hard_cap_bpw=32,
            imatrix_num_samples=17,
            imatrix_seq_length=64,
            imatrix_cache_path=str(cache_path),
            sensitivity_map_override={0: 0.25},
        )
        report = json.loads((output / "quantization_report.json").read_text())
        assert report["calibration"]["imatrix_cache_reused"] is True
        assert report["calibration"]["sensitivity"] == {"0": 0.25}
        assert report["imatrix_applied_modules"] == [name]
        assert (
            report["remaining_weights"]["tensor_bytes"]
            == report["calibration"]["tensor_bytes"]
        )
        saved = json.loads((output / "config.json").read_text())["omlx_deepseek_v41"]
        assert saved["quantized_modules"][name]["bits"] in (3, 4, 6, 8)


def test_expert_requantization_cancellation_does_not_process_remaining_experts():
    from omlx.patches.deepseek_v41.oq import requantize_projection

    with mx.stream(mx.cpu):
        weight = mx.ones((3, 16, 64), mx.bfloat16)
        values = {"p.weight": weight}
        calls = []

        def cancel(done, total):
            calls.append((done, total))
            raise RuntimeError("cancelled")

        with pytest.raises(RuntimeError, match="cancelled"):
            requantize_projection(values, "p", None, bits=3, progress=cancel)
        assert calls == [(16, 48)]
        assert list(values) == ["p.weight"]
        assert values["p.weight"] is weight


def test_unobserved_expert_uses_affine_three_bits_without_importance():
    from omlx import oq
    from omlx.patches.deepseek_v41.oq import requantize_projection

    with mx.stream(mx.cpu):
        weight = mx.sin(mx.arange(2048).reshape(2, 16, 64)).astype(mx.bfloat16)
        entry = oq.OQImatrixEntry(
            np.stack([np.linspace(0.1, 2, 64), np.zeros(64)]).astype(np.float32),
            np.array([10, 0]),
        )
        data = oq.OQImatrixData({"p": entry}, {}, "fixture")
        importance = oq._lookup_imatrix_importance(
            data, "p.weight", weight.shape, strict=True, report=None
        )
        assert mx.array_equal(importance[1], mx.ones((64,))).item()
        result, spec = requantize_projection(
            {"p.weight": weight}, "p", None, bits=3, importance=importance
        )
        plain = mx.quantize(weight[1], bits=3, group_size=64, mode="affine")
        for suffix, expected in zip(("weight", "scales", "biases"), plain):
            assert mx.array_equal(result["p." + suffix][1], expected).item()
        observed = oq._quantize_chunked(weight[0], 64, 3, "affine", importance[0])
        for suffix, expected in zip(("weight", "scales", "biases"), observed):
            assert mx.array_equal(result["p." + suffix][0], expected).item()
        assert spec["bits"] == 3 and spec["mode"] == "affine"
