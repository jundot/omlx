# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.switch_layers import SwiGLU

from omlx.expert_streaming.execution import CacheSnapshot, SpeculativeExecution
from omlx.expert_streaming.integrate import install_expert_streaming
from omlx.expert_streaming.manifest import (
    load_soft_reap_manifest,
    validate_soft_reap_manifest_data,
)
from omlx.expert_streaming.pool import StreamingSwitchGLU
from omlx.expert_streaming.residency import estimate_expert_streaming_residency
from omlx.expert_streaming.routing import resident_preferred_topk
from omlx.expert_streaming.safetensors import (
    ExpertReader,
    SafetensorExpertIndex,
)


def _checkpoint(path: Path, *, experts: int = 6, layers: int = 1):
    tensors = {}
    full = {}
    for layer in range(layers):
        for projection, output, inputs in (
            ("gate_proj", 32, 64),
            ("up_proj", 32, 64),
            ("down_proj", 64, 32),
        ):
            dense = mx.random.normal((experts, output, inputs)).astype(mx.float16)
            weight, scales, biases = mx.quantize(
                dense, group_size=32, bits=4, mode="affine"
            )
            prefix = f"language_model.model.layers.{layer}.mlp.switch_mlp.{projection}"
            tensors[f"{prefix}.weight"] = weight
            tensors[f"{prefix}.scales"] = scales
            tensors[f"{prefix}.biases"] = biases
            full[(layer, projection)] = (weight, scales, biases)
    shard = "model-00001-of-00001.safetensors"
    mx.save_safetensors(str(path / shard), tensors, metadata={"format": "mlx"})
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: shard for key in tensors}})
    )
    return full


def _reference(full, x, indices):
    expanded = mx.expand_dims(x, (-2, -3))

    def projection(name, value):
        weight, scales, biases = full[(0, name)]
        return mx.gather_qmm(
            value,
            weight,
            scales,
            biases,
            rhs_indices=indices,
            transpose=True,
            group_size=32,
            bits=4,
            mode="affine",
        )

    up = projection("up_proj", expanded)
    gate = projection("gate_proj", expanded)
    return projection("down_proj", SwiGLU()(up, gate)).squeeze(-2)


def test_official_reap_layer_mapping_shape(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"0": [4, 1, 3], "1": [0, 2, 5]}))
    manifest = load_soft_reap_manifest(path, num_layers=2, num_experts=6)
    assert manifest.layers == {0: (1, 3, 4), 1: (0, 2, 5)}
    assert manifest.pinned_count_range == (3, 3)


def test_manifest_rejects_duplicates_and_missing_layers():
    with pytest.raises(ValueError, match="duplicate"):
        validate_soft_reap_manifest_data({"0": [1, 1]}, num_layers=1, num_experts=2)
    with pytest.raises(ValueError, match="missing layers"):
        validate_soft_reap_manifest_data({"0": [1]}, num_layers=2, num_experts=2)


def test_cache_only_residency_requires_no_manifest(tmp_path):
    _checkpoint(tmp_path)
    estimate = estimate_expert_streaming_residency(
        tmp_path,
        None,
        cache_experts=2,
        num_layers=1,
        num_experts=6,
        top_k=2,
        streaming_mode="cache_only",
    )

    assert estimate.pinned_bytes == 0
    assert estimate.cache_slots_per_layer == 2


def test_cache_only_install_requires_no_manifest(tmp_path):
    _checkpoint(tmp_path)

    class Model:
        def __init__(self):
            projection = SimpleNamespace(
                weight=mx.zeros((1, 1)),
                scales=mx.zeros((1, 1)),
                biases=mx.zeros((1, 1)),
                group_size=32,
                bits=4,
                mode="affine",
            )
            switch = SimpleNamespace(
                gate_proj=projection,
                up_proj=projection,
                down_proj=projection,
                activation=SwiGLU(),
            )
            mlp = SimpleNamespace(num_experts=6, top_k=2, switch_mlp=switch)
            self.layers = [SimpleNamespace(mlp=mlp)]

        def __call__(self, inputs, cache=None):
            del cache
            return inputs

    model = Model()
    runtime = install_expert_streaming(
        model,
        tmp_path,
        None,
        cache_experts=2,
        streaming_mode="cache_only",
    )
    try:
        assert runtime.streaming_mode == "cache_only"
        assert runtime.manifest.pinned_count_range == (0, 0)
        assert runtime.pools[0].pinned_count == 0
    finally:
        runtime.close()


def test_resident_substitution_uses_relative_router_threshold():
    gates = mx.array([[0.40, 0.35, 0.20, 0.05]], dtype=mx.float32)
    resident = mx.array([False, True, True, False])

    below, _ = resident_preferred_topk(gates, resident, top_k=1, threshold_percent=10)
    above, scores = resident_preferred_topk(
        gates, resident, top_k=1, threshold_percent=15
    )
    resident_only, _ = resident_preferred_topk(
        gates, resident, top_k=2, threshold_percent=100
    )
    mx.eval(below, above, scores, resident_only)

    assert int(below.item()) == 0
    assert int(above.item()) == 1
    assert float(scores.item()) == pytest.approx(1.0)
    assert set(resident_only.tolist()[0]) == {1, 2}


def test_reader_returns_exact_selected_rows(tmp_path):
    full = _checkpoint(tmp_path)
    index = SafetensorExpertIndex(tmp_path)
    reader = ExpertReader(index)
    try:
        rows = reader.read_rows(index.layer(0)[("gate_proj", "weight")], [5, 0, 3])
        expected = full[(0, "gate_proj")][0][[5, 0, 3]]
        mx.eval(rows, expected)
        assert bool(mx.all(rows == expected).item())
    finally:
        reader.close()


def test_index_ignores_mtp_layer_zero_experts(tmp_path):
    _checkpoint(tmp_path)
    index_path = tmp_path / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    shard = next(iter(weight_map.values()))
    for projection in ("gate_proj", "up_proj", "down_proj"):
        for part in ("weight", "scales", "biases"):
            weight_map[f"mtp.layers.0.mlp.switch_mlp.{projection}.{part}"] = shard
    index_path.write_text(json.dumps({"weight_map": weight_map}))

    locations = SafetensorExpertIndex(tmp_path).layer(0)

    assert all(not location.name.startswith("mtp.") for location in locations.values())


@pytest.mark.parametrize("expert_major", [False, True])
def test_streaming_switch_glu_is_bit_exact(tmp_path, expert_major):
    full = _checkpoint(tmp_path)
    index = SafetensorExpertIndex(tmp_path)
    reader = ExpertReader(index)
    try:
        pool = StreamingSwitchGLU(
            layer=0,
            num_experts=6,
            top_k=2,
            pinned_experts=(0,),
            cache_slots=2 if expert_major else 5,
            locations=index.layer(0),
            projection_metadata={
                name: {"group_size": 32, "bits": 4, "mode": "affine"}
                for name in ("gate_proj", "up_proj", "down_proj")
            },
            activation=SwiGLU(),
            reader=reader,
        )
        indices = (
            mx.array([[[0, 1], [2, 3], [4, 5]]], dtype=mx.int32)
            if expert_major
            else mx.array([[[0, 1], [2, 3]]], dtype=mx.int32)
        )
        x = mx.random.normal((*indices.shape[:-1], 64)).astype(mx.float16)
        output = pool(x, indices)
        expected = _reference(full, x, indices)
        mx.eval(output, expected)
        assert bool(mx.all(output == expected).item())
        assert pool.stats.expert_major_calls == int(expert_major)
    finally:
        reader.close()


def test_cache_only_switch_glu_is_bit_exact(tmp_path):
    full = _checkpoint(tmp_path)
    index = SafetensorExpertIndex(tmp_path)
    reader = ExpertReader(index)
    try:
        pool = StreamingSwitchGLU(
            layer=0,
            num_experts=6,
            top_k=2,
            pinned_experts=(),
            cache_slots=2,
            locations=index.layer(0),
            projection_metadata={
                name: {"group_size": 32, "bits": 4, "mode": "affine"}
                for name in ("gate_proj", "up_proj", "down_proj")
            },
            activation=SwiGLU(),
            reader=reader,
        )
        indices = mx.array([[[2, 5]]], dtype=mx.int32)
        x = mx.random.normal((1, 1, 64)).astype(mx.float16)
        output = pool(x, indices)
        expected = _reference(full, x, indices)
        mx.eval(output, expected)

        assert pool.pinned_count == 0
        assert bool(mx.all(output == expected).item())
    finally:
        reader.close()


def test_route_larger_than_hot_cache_chunks_without_evicting_itself(tmp_path):
    full = _checkpoint(tmp_path)
    index = SafetensorExpertIndex(tmp_path)
    reader = ExpertReader(index)
    try:
        pool = StreamingSwitchGLU(
            layer=0,
            num_experts=6,
            top_k=2,
            pinned_experts=(),
            cache_slots=2,
            locations=index.layer(0),
            projection_metadata={
                name: {"group_size": 32, "bits": 4, "mode": "affine"}
                for name in ("gate_proj", "up_proj", "down_proj")
            },
            activation=SwiGLU(),
            reader=reader,
        )
        # Prime two experts so the next route has only two misses, but needs
        # four unpinned experts in total. The old miss-count check evicted an
        # expert protected by this same route and then raised KeyError.
        mx.eval(pool.ensure(mx.array([[0, 1]], dtype=mx.int32)))
        indices = mx.array([[[0, 1], [2, 3]]], dtype=mx.int32)
        x = mx.random.normal((1, 2, 64)).astype(mx.float16)

        output = pool(x, indices)
        expected = _reference(full, x, indices)
        mx.eval(output, expected)

        assert bool(mx.all(output == expected).item())
        assert pool.stats.expert_major_calls == 1
    finally:
        reader.close()


def test_direct_projection_helper_chunks_oversized_route(tmp_path):
    full = _checkpoint(tmp_path)
    index = SafetensorExpertIndex(tmp_path)
    reader = ExpertReader(index)
    try:
        pool = StreamingSwitchGLU(
            layer=0,
            num_experts=6,
            top_k=2,
            pinned_experts=(),
            cache_slots=2,
            locations=index.layer(0),
            projection_metadata={
                name: {"group_size": 32, "bits": 4, "mode": "affine"}
                for name in ("gate_proj", "up_proj", "down_proj")
            },
            activation=SwiGLU(),
            reader=reader,
        )
        mx.eval(pool.ensure(mx.array([[0, 1]], dtype=mx.int32)))
        indices = mx.array([[[0, 1], [2, 3]]], dtype=mx.int32)
        x = mx.random.normal((1, 2, 64)).astype(mx.float16)
        flat_indices = indices.reshape(2, 2)
        flat_x = mx.expand_dims(x.reshape(2, 64), (-2, -3))

        up = pool.up_proj(flat_x, flat_indices, sorted_indices=False)
        gate = pool.gate_proj(flat_x, flat_indices, sorted_indices=False)
        output = pool.down_proj(
            pool.activation(up, gate), flat_indices, sorted_indices=False
        ).squeeze(-2).reshape(1, 2, 2, -1)
        expected = _reference(full, x, indices)
        mx.eval(output, expected)

        assert bool(mx.all(output == expected).item())
        assert pool.stats.expert_major_calls == 3
    finally:
        reader.close()


def test_route_frequency_cache_retains_frequent_expert(tmp_path):
    _checkpoint(tmp_path)
    index = SafetensorExpertIndex(tmp_path)
    reader = ExpertReader(index)
    try:
        pool = StreamingSwitchGLU(
            layer=0,
            num_experts=6,
            top_k=1,
            pinned_experts=(),
            cache_slots=2,
            locations=index.layer(0),
            projection_metadata={
                name: {"group_size": 32, "bits": 4, "mode": "affine"}
                for name in ("gate_proj", "up_proj", "down_proj")
            },
            activation=SwiGLU(),
            reader=reader,
            cache_policy="route_frequency",
        )
        for _ in range(5):
            mx.eval(pool.ensure(mx.array([0], dtype=mx.int32)))
        mx.eval(pool.ensure(mx.array([1], dtype=mx.int32)))
        mx.eval(pool.ensure(mx.array([2], dtype=mx.int32)))

        resident = pool.resident_mask.tolist()
        assert resident[0] is True
        assert resident[1] is False
        assert resident[2] is True
    finally:
        reader.close()


def test_cache_snapshot_restores_nested_cache_metadata():
    class Cache:
        def __init__(self):
            self.offset = 7
            self.cache = [mx.array([1]), mx.array([2])]

    cache = Cache()
    original = cache.cache[0]
    caches = [cache]
    snapshot = CacheSnapshot(caches)
    cache.offset = 8
    cache.cache[0] = mx.array([99])
    caches[0] = object()

    snapshot.restore()

    assert cache.offset == 7
    assert cache.cache[0] is original
    assert caches == [cache]


def test_speculative_miss_rolls_back_promotes_and_retries_exactly(tmp_path):
    full = _checkpoint(tmp_path)
    index = SafetensorExpertIndex(tmp_path)
    reader = ExpertReader(index)
    pool = StreamingSwitchGLU(
        layer=0,
        num_experts=6,
        top_k=2,
        pinned_experts=(0,),
        cache_slots=2,
        locations=index.layer(0),
        projection_metadata={
            name: {"group_size": 32, "bits": 4, "mode": "affine"}
            for name in ("gate_proj", "up_proj", "down_proj")
        },
        activation=SwiGLU(),
        reader=reader,
    )

    class Cache:
        offset = 0

    class Model:
        def __init__(self):
            self.indices = mx.array([[[0, 0]]], dtype=mx.int32)
            self.x = mx.random.normal((1, 1, 64)).astype(mx.float16)

        def __call__(self, input_ids, cache=None):
            del input_ids
            if cache is not None:
                cache.offset += 1
            return pool(self.x, self.indices)

    class Runtime:
        pools = [pool]

    model = Model()
    cache = Cache()
    execution = SpeculativeExecution(Runtime(), policy="speculative")
    execution.attach(model)
    try:
        # The first pass is intentionally checked so a one-token prompt cannot
        # trigger a cascade of speculative cold misses.
        mx.eval(model(mx.array([[1]]), cache=cache))
        assert cache.offset == 1

        model.indices = mx.array([[[0, 5]]], dtype=mx.int32)
        result = model(mx.array([[2]]), cache=cache)
        expected = _reference(full, model.x, model.indices)
        mx.eval(result, expected)

        assert bool(mx.all(result == expected).item())
        assert cache.offset == 2
        assert execution.stats.speculative_passes == 2
        assert execution.stats.speculative_retries == 1
        assert execution.stats.speculative_hits == 1
        assert execution.stats.speculative_fallbacks == 0
        assert pool.stats.cold_loads == 1
    finally:
        execution.close()
        reader.close()
