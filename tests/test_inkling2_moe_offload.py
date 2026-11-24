# SPDX-License-Identifier: Apache-2.0
"""Mixed-bit Inkling 2 expert offload and route-order parity."""

import json

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches.inkling2.model import Inkling2Experts, TextArgs
from omlx.patches.moe_expert_offload import (
    apply_moe_expert_offload,
    estimate_offload_admission_bytes,
    materialize_offload_state,
    moe_offload_caches,
)
from omlx.patches.moe_offload_compat import moe_offload_compatibility
from omlx.patches.qwen35_verify_qmm import set_verify_qmm_armed

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

_PREFIX = "model.layers.0.mlp.experts"


@pytest.fixture
def checkpoint(tmp_path):
    args = TextArgs(
        num_hidden_layers=1,
        n_routed_experts=32,
        num_experts_per_tok=8,
        hidden_size=256,
        intermediate_size=256,
    )
    w13 = {"bits": 2, "group_size": 64, "mode": "affine"}
    w2 = {**w13, "bits": 3}
    config = {
        "model_type": "inkling2_loop",
        "text_config": vars(args),
        "quantization": {**w13, f"{_PREFIX}.w2": w2},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.random.seed(20)
    experts = Inkling2Experts(args, w13, w2)
    tensors = {}
    for name in ("w13", "w2"):
        proj = getattr(experts, name)
        shape = (*proj.weight.shape[:2], args.hidden_size)
        proj.weight, proj.scales, proj.biases = mx.quantize(
            (mx.random.normal(shape) * 0.1).astype(mx.bfloat16),
            bits=proj.bits,
            group_size=proj.group_size,
        )
        for field in ("weight", "scales", "biases"):
            tensors[f"{_PREFIX}.{name}.{field}"] = proj[field]
    shard = tmp_path / "model.safetensors"
    mx.save_safetensors(str(shard), tensors)
    mlp = nn.Module()
    mlp.experts = experts
    layer = nn.Module()
    layer.mlp = mlp
    model = nn.Module()
    model.layers = [layer]
    return model, experts, shard, tensors


@pytest.mark.parametrize("rows,row_exact", [(1, False), (32, False), (9, True)])
def test_inkling_expert_offload_routes(checkpoint, rows, row_exact):
    model, resident, shard, tensors = checkpoint
    path = shard.parent
    full_size = shard.stat().st_size
    expert_bytes = sum(array.nbytes for array in tensors.values())
    assert moe_offload_compatibility(path) == (True, "")
    assert estimate_offload_admission_bytes(path, full_size, 0.125) == (
        full_size - expert_bytes * 24 // 32
    )
    assert apply_moe_expert_offload(model, path, 0.125) == 1
    assert materialize_offload_state(model) == 1
    (cache,) = moe_offload_caches(model)
    assert cache.capacity == 8
    assert not model.layers[0].mlp.experts.parameters()

    x = mx.random.normal((rows, 256)).astype(mx.bfloat16)
    indices = mx.arange(rows * 8).reshape(rows, 8) % 32
    weights = mx.random.uniform(shape=indices.shape)
    set_verify_qmm_armed(row_exact, row_exact=row_exact)
    try:
        expected, _ = resident.unreduced(x, indices, weights)
        actual, actual_weights = model.layers[0].mlp.experts.unreduced(
            x, indices, weights
        )
        assert mx.array_equal(actual, expected).item()
        assert actual_weights is weights
    finally:
        set_verify_qmm_armed(False)
    assert cache.misses == min(32, rows * 8)


def test_incomplete_inkling_experts_fail_before_wrapping(checkpoint):
    model, resident, shard, tensors = checkpoint
    name = f"{_PREFIX}.w2.biases"
    del tensors[name]
    mx.save_safetensors(str(shard), tensors)
    full_size = shard.stat().st_size
    assert moe_offload_compatibility(shard.parent) == (
        False,
        f"Checkpoint is missing Inkling 2 expert tensor: {name}",
    )
    assert estimate_offload_admission_bytes(shard.parent, full_size, 0.125) == full_size
    with pytest.raises(
        ValueError, match=r"missing Inkling 2 expert tensor: .*w2.biases"
    ):
        apply_moe_expert_offload(model, shard.parent, 0.125)
    assert model.layers[0].mlp.experts is resident
