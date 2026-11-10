# SPDX-License-Identifier: Apache-2.0
"""Cover the load-time bfloat16 -> float16 activation recast."""

import json
import struct

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from omlx.cache.paged_ssd_cache import numerics_revision_for_model
from omlx.utils.model_loading import (
    cast_bf16_params_to_fp16,
    checkpoint_has_bf16_leaves,
)


def _toy_quantized_model() -> nn.Module:
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.RMSNorm(128),
        nn.Linear(128, 32),
    )
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=64, bits=4)
    mx.eval(model.parameters())
    return model


def test_recast_moves_bf16_leaves_to_fp16_and_keeps_u32_payloads():
    model = _toy_quantized_model()
    x = mx.random.normal((4, 64)).astype(mx.bfloat16)
    before = model(x)
    mx.eval(before)

    quant_before = {
        key: value
        for key, value in tree_flatten(model.parameters())
        if value.dtype == mx.uint32
    }
    assert quant_before, "toy model should carry quantized weights"

    recast = cast_bf16_params_to_fp16(model)
    assert recast > 0

    dtypes = {value.dtype for _, value in tree_flatten(model.parameters())}
    assert mx.bfloat16 not in dtypes
    assert mx.float16 in dtypes

    for key, value in tree_flatten(model.parameters()):
        if key in quant_before:
            assert value.dtype == mx.uint32
            assert mx.array_equal(value, quant_before[key])

    after = model(x.astype(mx.float16))
    mx.eval(after)
    assert after.dtype == mx.float16
    # Same math, different rounding: bf16 has 8 mantissa bits, fp16 has 11.
    assert mx.allclose(after.astype(mx.float32), before.astype(mx.float32), atol=5e-2)


def test_recast_is_a_noop_when_nothing_is_bf16():
    model = nn.Linear(8, 8)
    model.set_dtype(mx.float16)
    mx.eval(model.parameters())
    assert cast_bf16_params_to_fp16(model) == 0
    assert numerics_revision_for_model(model) is None


def test_recast_tags_cache_numerics():
    model = _toy_quantized_model()
    assert numerics_revision_for_model(model) is None
    cast_bf16_params_to_fp16(model)
    assert numerics_revision_for_model(model) == "fp16-recast"


def test_recast_peak_stays_within_one_chunk():
    model = nn.Sequential(*[nn.Linear(512, 512, bias=False) for _ in range(32)])
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()
    before = mx.get_active_memory()

    cast_bf16_params_to_fp16(model)
    mx.synchronize()

    # A second copy of every leaf would add 32 x 512 KiB; one chunk adds 8.
    tensor_bytes = 512 * 512 * 2
    assert mx.get_peak_memory() - before <= 9 * tensor_bytes


def test_cast_keeps_bf16_when_a_value_exceeds_the_float16_range():
    model = _toy_quantized_model()
    norm = model.layers[1]
    norm.weight = mx.full(norm.weight.shape, 70000.0).astype(mx.bfloat16)
    mx.eval(norm.weight)
    before = {key: value.dtype for key, value in tree_flatten(model.parameters())}

    with pytest.raises(ValueError, match="float16 range"):
        cast_bf16_params_to_fp16(model)

    after = {key: value.dtype for key, value in tree_flatten(model.parameters())}
    assert after == before
    assert numerics_revision_for_model(model) is None


def test_checkpoint_has_bf16_leaves_reads_safetensors_headers(tmp_path):
    def _write(path, dtype):
        header = json.dumps(
            {"w": {"dtype": dtype, "shape": [2], "data_offsets": [0, 4]}}
        ).encode()
        path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * 4)

    bf16_dir = tmp_path / "bf16"
    bf16_dir.mkdir()
    _write(bf16_dir / "model-00001.safetensors", "F16")
    _write(bf16_dir / "model-00002.safetensors", "BF16")
    assert checkpoint_has_bf16_leaves(str(bf16_dir)) is True

    fp16_dir = tmp_path / "fp16"
    fp16_dir.mkdir()
    _write(fp16_dir / "model-00001.safetensors", "F16")
    assert checkpoint_has_bf16_leaves(str(fp16_dir)) is False

    assert checkpoint_has_bf16_leaves("") is False
    assert checkpoint_has_bf16_leaves(str(tmp_path / "missing")) is False
