# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.switch_layers import SwiGLU, _gather_sort, _scatter_unsort

from omlx.expert_streaming.adapters import discover_moe_layers
from omlx.expert_streaming.integrate import install_expert_streaming
from omlx.expert_streaming.manifest import (
    load_soft_reap_manifest,
    validate_soft_reap_manifest_data,
)
from omlx.expert_streaming.pool import StreamingSwitchGLU
from omlx.expert_streaming.residency import estimate_expert_streaming_residency
from omlx.expert_streaming.safetensors import (
    ExpertReader,
    SafetensorExpertIndex,
    TensorLocation,
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


def _reference(full, x, indices, *, sort_routes=False):
    expanded = mx.expand_dims(x, (-2, -3))
    routed_indices = indices
    inv_order = None
    if sort_routes:
        expanded, routed_indices, inv_order = _gather_sort(expanded, indices)

    def projection(name, value):
        weight, scales, biases = full[(0, name)]
        return mx.gather_qmm(
            value,
            weight,
            scales,
            biases,
            rhs_indices=routed_indices,
            transpose=True,
            group_size=32,
            bits=4,
            mode="affine",
            sorted_indices=sort_routes,
        )

    up = projection("up_proj", expanded)
    gate = projection("gate_proj", expanded)
    output = projection("down_proj", SwiGLU()(up, gate))
    if inv_order is not None:
        output = _scatter_unsort(output, inv_order, indices.shape)
    return output.squeeze(-2)


def _quantized_projection(dense, *, group_size, bits, mode):
    weight, scales, *biases = mx.quantize(
        dense, group_size=group_size, bits=bits, mode=mode
    )
    return SimpleNamespace(
        weight=weight,
        scales=scales,
        biases=biases[0] if biases else None,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )


def _streamable_test_model(*, experts: int = 6, top_k: int = 2):
    projections = {}
    for name, output, inputs in (
        ("gate_proj", 32, 64),
        ("up_proj", 32, 64),
        ("down_proj", 64, 32),
    ):
        projections[name] = _quantized_projection(
            mx.zeros((experts, output, inputs), dtype=mx.float16),
            group_size=32,
            bits=4,
            mode="affine",
        )
    switch = SimpleNamespace(
        **projections,
        activation=SwiGLU(),
    )
    mlp = SimpleNamespace(
        num_experts=experts,
        top_k=top_k,
        switch_mlp=switch,
    )

    class Model:
        def __init__(self):
            self.layers = [SimpleNamespace(mlp=mlp)]

        def __call__(self, inputs, cache=None):
            del cache
            return inputs

    return Model()


def _family_checkpoint(
    path: Path,
    *,
    container: str,
    fragmented: bool = False,
    fused_gate_up: bool = False,
):
    experts = 4
    projections = {}
    for name, output, inputs in (
        ("gate_proj", 32, 64),
        ("up_proj", 32, 64),
        ("down_proj", 64, 32),
    ):
        dense = mx.random.normal((experts, output, inputs)).astype(mx.float16)
        projections[name] = _quantized_projection(
            dense, group_size=32, bits=4, mode="mxfp4"
        )
    tensors = {}
    if fused_gate_up:
        gate = projections["gate_proj"]
        up = projections["up_proj"]
        fused = SimpleNamespace(
            weight=mx.concatenate([gate.weight, up.weight], axis=1),
            scales=mx.concatenate([gate.scales, up.scales], axis=1),
            biases=None,
            group_size=32,
            bits=4,
            mode="mxfp4",
        )
        modules = {"gate_up_proj": fused, "down_proj": projections["down_proj"]}
    else:
        modules = projections
    if fragmented:
        aliases = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}
        for projection, module in projections.items():
            for expert in range(experts):
                prefix = (
                    f"model.layers.1.{container}.experts.{expert}.{aliases[projection]}"
                )
                tensors[f"{prefix}.weight"] = module.weight[expert]
                tensors[f"{prefix}.scale"] = module.scales[expert]
    else:
        for projection, module in modules.items():
            prefix = f"model.layers.1.{container}.switch_mlp.{projection}"
            tensors[f"{prefix}.weight"] = module.weight
            tensors[f"{prefix}.scales"] = module.scales
    shard = "model-00001-of-00001.safetensors"
    mx.save_safetensors(str(path / shard), tensors, metadata={"format": "mlx"})
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: shard for key in tensors}})
    )
    return projections, modules


def _family_reference(projections, x, indices):
    expanded = mx.expand_dims(x, (-2, -3))

    def project(name, value):
        module = projections[name]
        return mx.gather_qmm(
            value,
            module.weight,
            module.scales,
            module.biases,
            rhs_indices=indices,
            transpose=True,
            group_size=module.group_size,
            bits=module.bits,
            mode=module.mode,
        )

    up = project("up_proj", expanded)
    gate = project("gate_proj", expanded)
    return project("down_proj", SwiGLU()(up, gate)).squeeze(-2)


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


def test_qwen4_exp_residency_excludes_forced_mmap_ple(tmp_path):
    _checkpoint(tmp_path)
    ple = {
        "language_model.model.layers.0.ple.ngram_embedding.weight": mx.zeros(
            (4096, 64), dtype=mx.float16
        )
    }
    ple_shard = "model-00002-of-00002.safetensors"
    mx.save_safetensors(str(tmp_path / ple_shard), ple, metadata={"format": "mlx"})
    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"].update({key: ple_shard for key in ple})
    index_path.write_text(json.dumps(index))
    ple_bytes = 4096 * 64 * 2

    def estimate(model_type):
        (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
        return estimate_expert_streaming_residency(
            tmp_path,
            None,
            cache_experts=2,
            scratch_experts=0,
            num_layers=1,
            num_experts=6,
            top_k=2,
            streaming_mode="cache_only",
        )

    plain = estimate("qwen3_5_moe")
    qwen4 = estimate("qwen4_exp")

    # Streaming forces the Qwen4-Exp PLE onto mmap, so it must not be charged
    # as resident; any other architecture keeps the full fixed trunk.
    assert plain.mmap_bytes == 0
    assert qwen4.mmap_bytes == ple_bytes
    assert plain.fixed_bytes - qwen4.fixed_bytes == ple_bytes
    assert plain.resident_bytes - qwen4.resident_bytes == pytest.approx(
        ple_bytes * 1.05, abs=1
    )


def test_cold_reader_coalesces_small_gaps_without_affecting_direct_preload(tmp_path):
    _checkpoint(tmp_path)
    index = SafetensorExpertIndex(tmp_path)
    cold = ExpertReader(index)
    direct = ExpertReader(index)
    try:
        cold_rows = cold.read_many(index.layer(0), [0, 2], use_file_cache=True)
        direct_rows = direct.read_many(index.layer(0), [0, 2], use_file_cache=False)
        mx.eval(*cold_rows.values(), *direct_rows.values())
        assert all(
            bool(mx.all(cold_rows[key] == direct_rows[key]).item())
            for key in cold_rows
        )
        assert cold.read_operations < direct.read_operations
        assert cold.bytes_read > direct.bytes_read
    finally:
        cold.close()
        direct.close()


def test_async_reader_materializes_mlx_only_on_inference_thread(
    tmp_path, monkeypatch
):
    _checkpoint(tmp_path)
    index = SafetensorExpertIndex(tmp_path)
    reader = ExpertReader(index)
    inference_thread = threading.get_ident()
    original_array = mx.array

    def inference_thread_array(*args, **kwargs):
        assert threading.get_ident() == inference_thread
        return original_array(*args, **kwargs)

    monkeypatch.setattr(mx, "array", inference_thread_array)
    try:
        locations = index.layer(0)
        cpu_components = reader.read_many_async(
            locations, [0, 2], use_file_cache=True
        ).result()
        assert all(type(value).__module__ == "numpy" for value in cpu_components.values())

        arrays = reader.materialize_many(locations, cpu_components)
        expected = reader.read_many(locations, [0, 2], use_file_cache=True)
        mx.eval(*arrays.values(), *expected.values())
        assert all(
            bool(mx.all(arrays[key] == expected[key]).item()) for key in arrays
        )
    finally:
        reader.close()


def test_reader_converts_checkpoint_bf16_to_runtime_fp16_metadata():
    source = mx.array([[1.5, -2.25]], dtype=mx.bfloat16)
    raw = np.asarray(source.view(mx.uint16))
    location = TensorLocation(
        name="scales",
        path=Path("unused.safetensors"),
        dtype="F16",
        shape=(1, 2),
        data_start=0,
        row_bytes=4,
        storage_dtype="BF16",
        storage_shape=(1, 2),
        storage_row_bytes=4,
    )
    reader = ExpertReader.__new__(ExpertReader)

    converted = reader.materialize_many({"scales": location}, {"scales": raw})[
        "scales"
    ]
    mx.eval(converted)

    assert converted.dtype == mx.float16
    assert mx.array_equal(converted, source.astype(mx.float16)).item()
    assert reader.fast_resource_compatible({"scales": location})
    assert not reader.fast_direct_compatible({"scales": location})


def test_cache_only_install_requires_no_manifest(tmp_path):
    _checkpoint(tmp_path)
    model = _streamable_test_model()
    original_switch = model.layers[0].mlp.switch_mlp
    original_call = type(model).__call__
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
        assert runtime.hotlist_preloaded == 0
        assert runtime.optimistic_preloaded == 2
        assert sum(runtime.pools[0].resident_mask.tolist()) == 2
        assert not any(runtime.pools[0]._route_hotness)
    finally:
        runtime.close()
    assert model.layers[0].mlp.switch_mlp is original_switch
    assert not hasattr(model, "_omlx_expert_streaming_runtime")
    assert type(model).__call__ is original_call


def test_native_demand_patches_synchronize_until_last_runtime_closes(tmp_path):
    from omlx.custom_kernels.fast_resource_loading import available

    if not available():
        pytest.skip("Fast Resource Loading extension unavailable")
    original = mx.synchronize
    _checkpoint(tmp_path)
    kwargs = dict(
        cache_experts=2,
        streaming_mode="cache_only",
        fast_resource_loading=True,
        direct_io=True,
        native_demand=True,
    )
    first = install_expert_streaming(_streamable_test_model(), tmp_path, None, **kwargs)
    try:
        assert mx.synchronize is not original
        assert mx.synchronize.__wrapped__ is original
        second = install_expert_streaming(
            _streamable_test_model(), tmp_path, None, **kwargs
        )
        try:
            # The GIL-releasing variant must still be a working barrier.
            value = mx.random.normal((64, 64)) @ mx.random.normal((64, 64))
            mx.async_eval(value)
            mx.synchronize()
            mx.synchronize(mx.default_stream(mx.default_device()))
            mx.synchronize(mx.default_device())
            mx.synchronize(mx.new_thread_local_stream(mx.default_device()))
            with pytest.raises(TypeError):
                mx.synchronize("not a stream")
            assert value.shape == (64, 64)
        finally:
            second.close()
        assert mx.synchronize is not original
    finally:
        first.close()
    assert mx.synchronize is original

    # Without native demand the module is left untouched.
    plain = install_expert_streaming(
        _streamable_test_model(),
        tmp_path,
        None,
        cache_experts=2,
        streaming_mode="cache_only",
    )
    try:
        assert mx.synchronize is original
    finally:
        plain.close()


def test_execution_releases_pool_after_prefill_passes_only(tmp_path, monkeypatch):
    from omlx.expert_streaming import execution as execution_module

    _checkpoint(tmp_path)
    model = _streamable_test_model()
    runtime = install_expert_streaming(
        model, tmp_path, None, cache_experts=2, streaming_mode="cache_only"
    )
    calls: list[str] = []
    monkeypatch.setattr(execution_module.mx, "clear_cache", lambda: calls.append("clear"))
    try:
        prefill = mx.zeros((1, 8), dtype=mx.uint32)
        decode = mx.zeros((1, 1), dtype=mx.uint32)
        runtime.execution.execute_call(model, lambda: None, (prefill,), {})
        assert calls == ["clear"], "prefill pass must return the pool"
        runtime.execution.execute_call(model, lambda: None, (decode,), {})
        assert calls == ["clear"], "decode pass must keep warm buffers"
    finally:
        runtime.close()


def test_native_demand_route_maps_are_updated_in_place(tmp_path):
    from omlx.custom_kernels.fast_resource_loading import available

    if not available():
        pytest.skip("Fast Resource Loading extension unavailable")
    _checkpoint(tmp_path)
    runtime = install_expert_streaming(
        _streamable_test_model(),
        tmp_path,
        None,
        cache_experts=2,
        streaming_mode="cache_only",
        fast_resource_loading=True,
        direct_io=True,
        native_demand=True,
    )
    try:
        pool = runtime.pools[0]
        slot_map, resident = pool._slot_map, pool._resident_mask
        before = set(pool._expert_to_slot)
        missing = [e for e in range(pool.num_experts) if e not in before][:1]
        pool._ensure_values(tuple(missing), destinations_idle=True)
        # Same array objects, new contents: a pipelined graph holding these
        # arrays sees the update when its remap kernel runs.
        assert pool._slot_map is slot_map and pool._resident_mask is resident
        assert slot_map.tolist() == pool._slot_map_np.tolist()
        assert resident.tolist() == pool._resident_mask_np.tolist()
        assert resident.tolist()[missing[0]] is True
    finally:
        runtime.close()


def test_fast_resource_loading_writes_exact_preallocated_bank_rows(tmp_path):
    from omlx.custom_kernels.fast_resource_loading import available

    if not available():
        pytest.skip("Fast Resource Loading native extension is not built")
    full = _checkpoint(tmp_path)
    model = _streamable_test_model()
    runtime = install_expert_streaming(
        model,
        tmp_path,
        None,
        cache_experts=2,
        scratch_experts=2,
        streaming_mode="cache_only",
        fast_resource_loading=True,
    )
    try:
        indices = mx.array([[[4, 5]]], dtype=mx.int32)
        x = mx.random.normal((1, 1, 64)).astype(mx.float16)
        output = runtime.pools[0](x, indices)
        expected = _reference(full, x, indices)
        mx.eval(output, expected)
        assert bool(mx.all(output == expected).item())
        stats = runtime.stats()
        assert stats["fast_resource_loading"] is True
        assert stats["frl_loads"] >= 2
        assert stats["frl_bytes_read"] > 0
        assert stats["frl_read_operations"] > 0
        assert stats["frl_copy_seconds"] >= 0
    finally:
        runtime.close()


def test_fast_resource_loading_can_write_exact_idle_bank_rows_directly(tmp_path):
    from omlx.custom_kernels.fast_resource_loading import available

    if not available():
        pytest.skip("Fast Resource Loading native extension is not built")
    full = _checkpoint(tmp_path)
    runtime = install_expert_streaming(
        _streamable_test_model(),
        tmp_path,
        None,
        cache_experts=2,
        scratch_experts=2,
        streaming_mode="cache_only",
        fast_resource_loading=True,
        direct_io=True,
    )
    try:
        pool = runtime.pools[0]
        pool._ensure_values((4, 5), destinations_idle=True)
        indices = mx.array([[[4, 5]]], dtype=mx.int32)
        x = mx.random.normal((1, 1, 64)).astype(mx.float16)
        output = pool(x, indices)
        expected = _reference(full, x, indices)
        mx.eval(output, expected)
        assert bool(mx.all(output == expected).item())
        stats = runtime.stats()
        assert stats["frl_direct_loads"] == 1
        assert stats["frl_direct_bytes"] > 0
    finally:
        runtime.close()


def test_learned_hotlist_warm_starts_evictable_cache(tmp_path):
    model_path = tmp_path / "model"
    profile_dir = tmp_path / "profiles"
    model_path.mkdir()
    full = _checkpoint(model_path)

    first = install_expert_streaming(
        _streamable_test_model(),
        model_path,
        None,
        cache_experts=2,
        streaming_mode="cache_only",
        hotlist_profile_dir=profile_dir,
    )
    try:
        for _ in range(5):
            mx.eval(first.pools[0].ensure(mx.array([4], dtype=mx.int32)))
        for _ in range(2):
            mx.eval(first.pools[0].ensure(mx.array([1], dtype=mx.int32)))
        mx.eval(first.pools[0].ensure(mx.array([3], dtype=mx.int32)))
    finally:
        first.close()

    profiles = list(profile_dir.glob("*.json"))
    assert len(profiles) == 1
    second = install_expert_streaming(
        _streamable_test_model(),
        model_path,
        None,
        cache_experts=2,
        streaming_mode="cache_only",
        hotlist_profile_dir=profile_dir,
    )
    try:
        resident = second.pools[0].resident_mask.tolist()
        assert second.hotlist_preloaded == 2
        assert second.optimistic_preloaded == 0
        assert second.pools[0].pinned_count == 0
        assert resident[4] is True
        assert resident[1] is True
        assert second.pools[0].stats.warm_start_loads == 2
        assert second.pools[0].stats.route_lookups == 0
        bytes_after_preload = second.reader.bytes_read
        indices = mx.array([[[4, 1]]], dtype=mx.int32)
        x = mx.random.normal((1, 1, 64)).astype(mx.float16)
        output = second.pools[0](x, indices)
        expected = _reference(full, x, indices)
        mx.eval(output, expected)
        assert bool(mx.all(output == expected).item())
        assert second.reader.bytes_read == bytes_after_preload
        assert second.pools[0].stats.cache_hits == 2
        stats = second.stats()
        assert stats["ssd_io_seconds"] >= 0
        assert stats["ssd_decode_seconds"] >= 0
        assert stats["bank_bind_seconds"] >= 0
        assert stats["bank_materialize_seconds"] >= 0
        assert stats["qmm_calls"] == 2
        assert stats["fused_gate_up"] is True
        assert stats["execution_bank_slots"] == 6
        assert stats["execution_banks_per_layer"] == 1
        assert stats["scratch_slots_per_layer"] == 4
        assert stats["resident_experts"] == 2
        assert stats["resident_capacity"] == 2
    finally:
        second.close()


def test_invalid_hotlist_profile_is_ignored(tmp_path):
    model_path = tmp_path / "model"
    profile_dir = tmp_path / "profiles"
    model_path.mkdir()
    _checkpoint(model_path)
    first = install_expert_streaming(
        _streamable_test_model(),
        model_path,
        None,
        cache_experts=2,
        streaming_mode="cache_only",
        hotlist_profile_dir=profile_dir,
    )
    profile_path = first.hotlist_profile_path
    first.close()
    assert profile_path is not None
    profile_path.write_text("not json")

    second = install_expert_streaming(
        _streamable_test_model(),
        model_path,
        None,
        cache_experts=2,
        streaming_mode="cache_only",
        hotlist_profile_dir=profile_dir,
    )
    try:
        assert second.hotlist_preloaded == 0
        assert second.optimistic_preloaded == 2
        assert second.hotlist_profile_error.startswith("load:")
        assert second.pools[0].resident_mask.tolist() == [
            True,
            True,
            False,
            False,
            False,
            False,
        ]
        assert not any(second.pools[0]._route_hotness)
    finally:
        second.close()


@pytest.mark.parametrize(
    ("fragmented", "fused_gate_up", "fast_resource_loading"),
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, False, True),
        (False, True, True),
    ],
)
def test_cross_family_ffn_mxfp4_streaming_is_exact(
    tmp_path, fragmented, fused_gate_up, fast_resource_loading
):
    if fast_resource_loading:
        from omlx.custom_kernels.fast_resource_loading import available

        if not available():
            pytest.skip("Fast Resource Loading native extension is not built")
    projections, modules = _family_checkpoint(
        tmp_path,
        container="ffn",
        fragmented=fragmented,
        fused_gate_up=fused_gate_up,
    )
    switch = SimpleNamespace(activation=SwiGLU(), **modules)
    moe = SimpleNamespace(
        switch_mlp=switch,
        gate=SimpleNamespace(top_k=2, num_experts=4),
        config=SimpleNamespace(n_routed_experts=4, num_experts_per_tok=2),
    )

    class Model:
        def __init__(self):
            self.layers = [SimpleNamespace(ffn=object()), SimpleNamespace(ffn=moe)]

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
        fast_resource_loading=fast_resource_loading,
    )
    try:
        assert [pool.layer for pool in runtime.pools] == [1]
        assert runtime.manifest.layers == {1: ()}
        pool = runtime.pools[0]
        assert all(
            "arrays" not in metadata for metadata in pool.projection_metadata.values()
        )
        indices = mx.array([[[0, 3]]], dtype=mx.int32)
        x = mx.random.normal((1, 1, 64)).astype(mx.float16)
        output = pool(x, indices)
        expected = _family_reference(projections, x, indices)
        mx.eval(output, expected)
        assert bool(mx.all(output == expected).item())

        scores = mx.array([[[0.25, 0.75]]], dtype=mx.float32)
        combined = pool(x, indices, scores=scores, weighted_sum=True)
        expected_combined = (expected * scores[..., None].astype(expected.dtype)).sum(
            -2
        )
        mx.eval(combined, expected_combined)
        assert bool(mx.all(combined == expected_combined).item())
    finally:
        runtime.close()


def test_adapter_discovers_common_moe_geometry_and_skips_dense_layers():
    projection = SimpleNamespace(
        weight=mx.zeros((8, 4, 2), dtype=mx.uint32),
        scales=mx.zeros((8, 4, 1), dtype=mx.float16),
        biases=mx.zeros((8, 4, 1), dtype=mx.float16),
    )
    switch = SimpleNamespace(
        gate_proj=projection,
        up_proj=projection,
        down_proj=projection,
    )
    moe = SimpleNamespace(
        switch_mlp=switch,
        num_experts_per_tok=3,
        router=SimpleNamespace(num_experts=8),
    )
    model = SimpleNamespace(
        layers=[SimpleNamespace(mlp=object()), SimpleNamespace(mlp=moe)]
    )

    targets = discover_moe_layers(model)

    assert len(targets) == 1
    assert targets[0].layer_id == 1
    assert targets[0].container_name == "mlp"
    assert targets[0].num_experts == 8
    assert targets[0].top_k == 3


def test_adapter_discovers_qwen38_flash_next_vlm_backbone():
    projection = SimpleNamespace(
        weight=mx.zeros((512, 4, 2), dtype=mx.uint32),
        scales=mx.zeros((512, 4, 1), dtype=mx.float16),
        biases=mx.zeros((512, 4, 1), dtype=mx.float16),
    )
    switch = SimpleNamespace(
        gate_proj=projection,
        up_proj=projection,
        down_proj=projection,
    )
    moe = SimpleNamespace(
        switch_mlp=switch,
        num_experts=512,
        top_k=10,
        gate=object(),
    )
    model = SimpleNamespace(
        language_model=SimpleNamespace(
            model=SimpleNamespace(layers=[SimpleNamespace(mlp=moe)])
        )
    )

    targets = discover_moe_layers(model)

    assert len(targets) == 1
    assert targets[0].layer_id == 0
    assert targets[0].container_name == "mlp"
    assert targets[0].router_name == "gate"
    assert targets[0].num_experts == 512
    assert targets[0].top_k == 10


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


def test_qwen38_index_ignores_separate_mtp_expert_layer(tmp_path):
    _checkpoint(tmp_path)
    index_path = tmp_path / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    shard = next(iter(weight_map.values()))
    for projection in ("gate_proj", "up_proj", "down_proj"):
        for part in ("weight", "scales", "biases"):
            weight_map[
                f"language_model.mtp.layers.0.mlp.switch_mlp.{projection}.{part}"
            ] = shard
    index_path.write_text(json.dumps({"weight_map": weight_map}))

    locations = SafetensorExpertIndex(tmp_path).layer(0)

    assert all(".mtp." not in location.name for location in locations.values())


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


def test_resident_prefill_sorts_physical_slots_and_remains_bit_exact(tmp_path):
    full = _checkpoint(tmp_path)
    index = SafetensorExpertIndex(tmp_path)
    reader = ExpertReader(index)
    try:
        pool = StreamingSwitchGLU(
            layer=0,
            num_experts=6,
            top_k=2,
            pinned_experts=(5, 2, 0, 4, 1, 3),
            cache_slots=0,
            locations=index.layer(0),
            projection_metadata={
                name: {"group_size": 32, "bits": 4, "mode": "affine"}
                for name in ("gate_proj", "up_proj", "down_proj")
            },
            activation=SwiGLU(),
            reader=reader,
        )
        route = [value % 6 for value in range(64)]
        indices = mx.array(route, dtype=mx.int32).reshape(1, 32, 2)
        x = mx.random.normal((1, 32, 64)).astype(mx.float16)

        output = pool(x, indices)
        expected = _reference(full, x, indices, sort_routes=True)
        mx.eval(output, expected)

        assert bool(mx.all(output == expected).item())
        assert pool.stats.sorted_prefill_groups == 1
        assert pool.stats.sorted_prefill_routes == 64
        assert pool.stats.sorted_qmm_calls == 2
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


def test_scratch_bank_executes_cold_prefill_without_polluting_hot_cache(tmp_path):
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
            scratch_slots=2,
            locations=index.layer(0),
            projection_metadata={
                name: {"group_size": 32, "bits": 4, "mode": "affine"}
                for name in ("gate_proj", "up_proj", "down_proj")
            },
            activation=SwiGLU(),
            reader=reader,
        )
        pool.preload_hotlist([(0, 2), (1, 1)])
        before = tuple(pool._dynamic_lru)
        indices = mx.array([[[0, 1], [2, 3], [4, 5]]], dtype=mx.int32)
        x = mx.random.normal((1, 3, 64)).astype(mx.float16)
        output = pool(x, indices)
        expected = _reference(full, x, indices)
        mx.eval(output, expected)
        assert bool(mx.all(output == expected).item())
        assert tuple(pool._dynamic_lru) == before
        assert pool.stats.scratch_loads == 4
        assert pool.stats.evictions == 0
        assert pool.stats.cache_hits == 2
        assert pool.stats.cache_misses == 4
        assert pool.bank_size == 4
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
        output = (
            pool.down_proj(
                pool.activation(up, gate), flat_indices, sorted_indices=False
            )
            .squeeze(-2)
            .reshape(1, 2, 2, -1)
        )
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


def test_streaming_affine_pool_matches_f16_execution_path(tmp_path):
    full = _checkpoint(tmp_path)
    index = SafetensorExpertIndex(tmp_path)
    reader = ExpertReader(index)
    try:
        pool = StreamingSwitchGLU(
            layer=0,
            num_experts=6,
            top_k=2,
            pinned_experts=(0,),
            cache_slots=5,
            locations=index.layer(0),
            projection_metadata={
                name: {"group_size": 32, "bits": 4, "mode": "affine"}
                for name in ("gate_proj", "up_proj", "down_proj")
            },
            activation=SwiGLU(),
            reader=reader,
        )
        pool.preload_hotlist([])
        pool.set_execution_mode("resident")
        indices = mx.array([[[0, 5], [2, 3]]], dtype=mx.int32)
        x = mx.random.normal((1, 2, 64)).astype(mx.bfloat16)

        output = pool(x, indices)
        expected = _reference(full, x.astype(mx.float16), indices).astype(mx.bfloat16)
        mx.eval(output, expected)

        assert pool._uses_f16_affine_moe(x)
        assert output.dtype == mx.bfloat16
        assert bool(mx.all(output == expected).item())
    finally:
        reader.close()


def test_elastic_decode_cache_reuses_and_returns_scratch_rows(tmp_path):
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
            scratch_slots=2,
            locations=index.layer(0),
            projection_metadata={
                name: {"group_size": 32, "bits": 4, "mode": "affine"}
                for name in ("gate_proj", "up_proj", "down_proj")
            },
            activation=SwiGLU(),
            reader=reader,
            cache_policy="route_frequency",
        )
        pool.enable_elastic_decode_cache()
        mx.eval(pool.ensure(mx.array([0, 1], dtype=mx.int32)))

        assert pool.set_elastic_decode_cache_active(True)
        assert pool.cache_slots == 4
        assert pool.scratch_slots == 0
        mx.eval(pool.ensure(mx.array([2, 3], dtype=mx.int32)))
        assert len(pool._expert_to_slot) == 4

        assert pool.set_elastic_decode_cache_active(False)
        assert pool.cache_slots == 2
        assert pool.scratch_slots == 2
        assert all(slot < 2 for slot in pool._expert_to_slot.values())
        assert pool.stats.elastic_decode_cache_activations == 1
        assert pool.stats.elastic_decode_cache_demotions == 1
    finally:
        reader.close()
