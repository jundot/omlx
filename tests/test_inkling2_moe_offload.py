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


def _repack_module():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "scripts" / "repack_inkling2_experts.py"
    spec = importlib.util.spec_from_file_location("repack_inkling2_experts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_per_expert(shard, tensors, drop=()):
    """The offload repack's layout: each expert's six tensors back to back."""
    import numpy as np

    drop = set(drop)
    repack = _repack_module()
    entries = []
    for e in range(32):
        for proj, field in repack._EXPERT_FIELDS:
            name = f"{_PREFIX}.{e}.{proj}.{field}"
            if name in drop:
                continue
            a = tensors[f"{_PREFIX}.{proj}.{field}"][e]
            st = repack._MX_TO_ST[a.dtype]
            data = np.array(a.view(mx.uint16) if st == "BF16" else a)
            entries.append((name, st, list(a.shape), data))
    shard.unlink()
    repack._write_shard(shard, entries)


def test_per_expert_readahead_merges_one_run_per_miss(checkpoint, monkeypatch):
    import omlx.patches.moe_expert_offload as moe

    model, resident, shard, tensors = checkpoint
    _write_per_expert(shard, tensors)
    assert moe_offload_compatibility(shard.parent) == (True, "")
    full_size = shard.stat().st_size
    expert_bytes = sum(array.nbytes for array in tensors.values())
    assert estimate_offload_admission_bytes(shard.parent, full_size, 0.125) == (
        full_size - expert_bytes * 24 // 32
    )
    assert apply_moe_expert_offload(model, shard.parent, 0.125) == 1
    assert materialize_offload_state(model) == 1
    (cache,) = moe_offload_caches(model)
    assert cache.direct_reads and cache.overlap_grace_s == 0.0

    # Over-capacity calls take the read-ahead path; the rest read into slots.
    reads = []
    into, read = moe._preadv_into, moe.CheckpointExpertStore.read
    monkeypatch.setattr(
        moe,
        "_preadv_into",
        lambda plan, rows: reads.append(plan.nbytes) or into(plan, rows),
    )
    monkeypatch.setattr(
        moe.CheckpointExpertStore,
        "read",
        staticmethod(lambda plan: reads.append(plan.nbytes) or read(plan)),
    )
    x = mx.random.normal((4, 256)).astype(mx.bfloat16)
    indices = mx.arange(32).reshape(4, 8)
    weights = mx.random.uniform(shape=indices.shape)
    expected, _ = resident.unreduced(x, indices, weights)
    actual, _ = model.layers[0].mlp.experts.unreduced(x, indices, weights)
    assert mx.array_equal(actual, expected).item()
    assert cache.misses == 32
    assert reads == [cache.expert_bytes] * 32


def test_per_expert_direct_decode_reads_one_run_per_miss(checkpoint, monkeypatch):
    import omlx.patches.moe_expert_offload as moe

    model, resident, shard, tensors = checkpoint
    _write_per_expert(shard, tensors)
    assert apply_moe_expert_offload(model, shard.parent, 0.125) == 1
    materialize_offload_state(model)
    (cache,) = moe_offload_caches(model)

    reads = []
    into = moe._preadv_into
    monkeypatch.setattr(
        moe,
        "_preadv_into",
        lambda plan, rows: reads.append(plan.nbytes) or into(plan, rows),
    )
    x = mx.random.normal((1, 256)).astype(mx.bfloat16)
    indices = mx.arange(8).reshape(1, 8)
    weights = mx.random.uniform(shape=indices.shape)
    expected, _ = resident.unreduced(x, indices, weights)
    actual, _ = model.layers[0].mlp.experts.unreduced(x, indices, weights)
    assert mx.array_equal(actual, expected).item()
    assert cache.misses == 8
    assert reads == [cache.expert_bytes] * 8


def test_per_expert_repack_keeps_expert_tensors_contiguous(checkpoint):
    import struct

    _, _, shard, tensors = checkpoint
    _write_per_expert(shard, tensors)
    repack = _repack_module()
    with open(shard, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    for e in (0, 17, 31):
        specs = []
        for proj, field in repack._EXPERT_FIELDS:
            spec = header[f"{_PREFIX}.{e}.{proj}.{field}"]
            specs.append(spec["data_offsets"])
        for (_, end), (next_start, _) in zip(specs, specs[1:]):
            assert end == next_start


def test_incomplete_per_expert_layout_fails_before_wrapping(checkpoint):
    model, resident, shard, tensors = checkpoint
    missing = f"{_PREFIX}.31.w2.biases"
    _write_per_expert(shard, tensors, drop={missing})
    full_size = shard.stat().st_size
    assert moe_offload_compatibility(shard.parent) == (
        False,
        f"Checkpoint is missing Inkling 2 expert tensor: {missing}",
    )
    assert estimate_offload_admission_bytes(shard.parent, full_size, 0.125) == full_size
    with pytest.raises(
        ValueError, match=r"missing Inkling 2 expert tensor: .*31.w2.biases"
    ):
        apply_moe_expert_offload(model, shard.parent, 0.125)
    assert model.layers[0].mlp.experts is resident


def test_stack_per_expert_restores_stacked_tables(checkpoint):
    from omlx.patches.inkling2.loader import stack_per_expert

    _, _, shard, tensors = checkpoint
    _write_per_expert(shard, tensors)
    stacked = stack_per_expert(mx.load(str(shard)), 32)
    assert sorted(stacked) == sorted(tensors)
    for name, value in tensors.items():
        assert mx.array_equal(stacked[name], value).item()
    loaded = mx.load(str(shard))
    loaded.pop(f"{_PREFIX}.31.w2.biases")
    with pytest.raises(ValueError, match="missing experts"):
        stack_per_expert(loaded, 32)


def test_sanitize_weights_restacks_per_expert_checkpoint(checkpoint):
    from types import SimpleNamespace

    from omlx.patches.inkling2.loader import sanitize_weights

    _, _, shard, tensors = checkpoint
    _write_per_expert(shard, tensors)
    model = SimpleNamespace(
        args=SimpleNamespace(text=SimpleNamespace(n_routed_experts=32))
    )
    stacked, formats = sanitize_weights(model, mx.load(str(shard)))
    assert formats == {}
    assert sorted(stacked) == sorted(tensors)
    for name, value in tensors.items():
        assert mx.array_equal(stacked[name], value).item()


@pytest.mark.parametrize("per_expert", [False, True])
def test_direct_reads_wait_for_queued_gathers(checkpoint, per_expert):
    """Slot rows are written in place, so a gather queued before a miss must
    still see the expert that miss evicts."""
    model, resident, shard, tensors = checkpoint
    if per_expert:
        _write_per_expert(shard, tensors)
    assert apply_moe_expert_offload(model, shard.parent, 0.125) == 1
    materialize_offload_state(model)
    experts = model.layers[0].mlp.experts
    x = mx.random.normal((1, 256)).astype(mx.bfloat16)
    weights = mx.ones((1, 8))
    outs, refs = [], []
    for start in range(0, 32, 8):  # each call evicts the previous call's experts
        inds = mx.arange(start, start + 8).reshape(1, 8)
        outs.append(experts.unreduced(x, inds, weights)[0])
        refs.append(resident.unreduced(x, inds, weights)[0])
    mx.eval(outs, refs)
    for out, ref in zip(outs, refs):
        assert mx.array_equal(out, ref).item()


def test_direct_read_failure_frees_claimed_slots(checkpoint, monkeypatch):
    import omlx.patches.moe_expert_offload as moe

    model, resident, shard, tensors = checkpoint
    _write_per_expert(shard, tensors)
    assert apply_moe_expert_offload(model, shard.parent, 0.125) == 1
    materialize_offload_state(model)
    (cache,) = moe_offload_caches(model)
    cache.ensure(mx.arange(8))
    assert sorted(cache.slot_of) == list(range(8))

    def fail(plan, rows):
        raise OSError("injected read failure")

    with monkeypatch.context() as patch:
        patch.setattr(moe, "_preadv_into", fail)
        with pytest.raises(OSError, match="injected read failure"):
            cache.ensure(mx.arange(8, 12))
    # The four evicted experts are gone; nothing half-read is resident.
    assert len(cache.slot_of) == 4 and len(cache.free) == 4
    assert not set(cache.slot_of) & set(range(8, 12))
    for e, slot in cache.slot_of.items():
        assert cache.map[e].item() == slot and cache.slot_expert[slot] == e
    x = mx.random.normal((1, 256)).astype(mx.bfloat16)
    inds = mx.arange(8, 16).reshape(1, 8)
    weights = mx.ones((1, 8))
    expected = resident.unreduced(x, inds, weights)[0]
    actual = model.layers[0].mlp.experts.unreduced(x, inds, weights)[0]
    assert mx.array_equal(actual, expected).item()
