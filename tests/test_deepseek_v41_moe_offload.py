"""V4.1 expert residency preserves routing, projection arithmetic and loading."""

import json
import weakref
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np
import pytest
from test_deepseek_v41 import write_checkpoint

from omlx.patches.deepseek_v41.convert import convert
from omlx.patches.deepseek_v41.loading import load
from omlx.patches.deepseek_v41.moe_offload import OffloadedExpert


@pytest.mark.parametrize("converted", [False, True])
@pytest.mark.parametrize("engram", [False, True])
def test_checkpoint_offload_matches_prefill_decode_and_closes(
    tmp_path, converted, engram
):
    kwargs = dict(n_routed_experts=8, n_activated_experts=2)
    if engram:
        kwargs.update(
            engram_layer_ids=(1, 3),
            engram_num_embeddings=(72, 204),
            engram_vocab_size=5,
            engram_max_ngram_size=4,
            engram_n_heads=2,
            engram_head_dim=32,
            engram_compressed_vocab_size=64,
        )
    source, _ = write_checkpoint(tmp_path, vision=False, **kwargs)
    if converted:
        target = tmp_path / "converted"
        convert(source, target)
        source = target
    resident, _ = load(source, engram_ssd_offload=engram)
    with ThreadPoolExecutor(max_workers=1) as executor:
        disk, _ = executor.submit(
            load,
            source,
            engram_ssd_offload=engram,
            moe_expert_offload_resident_fraction=0.25,
        ).result()
    plan = disk._moe_offload_plan
    try:
        assert plan.capacity == 2 and plan.count == 8
        assert plan.resident_bytes * 4 == plan.full_bytes
        assert isinstance(disk.language_model.layers[0].ffn.experts, OffloadedExpert)
        caches = [m.language_model.make_cache() for m in (resident, disk)]
        # Include the caller's sorted expert path and several eviction rounds.
        for ids in (mx.array([[3, 4] * 17]), mx.array([[5]]), mx.array([[6, 7]])):
            out = [m(ids, cache=c) for m, c in zip((resident, disk), caches)]
            mx.eval(out)
            np.testing.assert_allclose(out[0], out[1], rtol=2e-4, atol=2e-5)
        assert disk.language_model.layers[0].ffn.experts.slots.misses >= 2
    finally:
        resident.close()
        disk.close()
    assert plan._closed and not plan._fds
    disk.close()


def test_missing_expert_is_rejected_before_loading(tmp_path):
    source, _ = write_checkpoint(tmp_path, vision=False)
    path = source / "model.safetensors.index.json"
    mapping = json.loads(path.read_text())
    del mapping["weight_map"]["layers.0.ffn.experts.3.w2.weight"]
    path.write_text(json.dumps(mapping))
    with pytest.raises(KeyError):
        load(source, moe_expert_offload_resident_fraction=0.25)


@pytest.mark.parametrize("source_format", ["affine", "mxfp4", "mxfp8"])
@pytest.mark.parametrize("sorted_routes", [False, True])
def test_quantized_expert_eviction_preserves_arithmetic(
    tmp_path, source_format, sorted_routes
):
    from mlx.utils import tree_flatten

    from omlx.patches.deepseek_v41.config import ModelConfig
    from omlx.patches.deepseek_v41.language import Expert
    from omlx.patches.deepseek_v41.moe_offload import ExpertOffloadPlan
    from omlx.patches.deepseek_v41.quantization import QuantizedProjection

    mx.random.seed(412)
    config = ModelConfig(
        dim=64, moe_inter_dim=64, n_layers=1, n_routed_experts=8, n_activated_experts=2
    )
    reference = Expert(config, True)
    specs = {}
    for proj in ("w1", "w3", "w2"):
        mode = source_format
        bits = 8 if mode == "mxfp8" or (mode == "affine" and proj == "w2") else 4
        arrays = mx.quantize(
            mx.random.normal((8, 64, 64)).astype(mx.bfloat16) * 0.05,
            bits=bits,
            group_size=32,
            mode=mode,
        )
        params = dict(weight=arrays[0], scales=arrays[1], bits=bits, mode=mode)
        if len(arrays) == 3:
            params["biases"] = arrays[2]
        setattr(reference, proj, QuantizedProjection(**params))
        specs[f"language_model.layers.0.ffn.experts.{proj}"] = dict(
            bits=bits, mode=mode
        )
    prefix = "language_model.layers.0.ffn.experts"
    tensors = {prefix + "." + k: v for k, v in tree_flatten(reference.parameters())}
    mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors)
    mapping = {k: "model.safetensors" for k in tensors}
    raw = {"omlx_deepseek_v41": {"version": 1, "quantized_modules": specs}}
    plan = ExpertOffloadPlan(tmp_path, raw, mapping, config, 0.25)
    disk = OffloadedExpert(Expert(config, True), plan, prefix)
    try:
        for step in range(4):
            if sorted_routes:
                idx = mx.array([0, 1, 2, 3, 4, 5, 6, 7])
                x = mx.random.normal((8, 1, 64)).astype(mx.bfloat16)
            else:
                idx = (mx.arange(6).reshape(1, 3, 2) + step * 2) % 8
                x = mx.random.normal((1, 3, 1, 1, 64)).astype(mx.bfloat16)
            weights = mx.ones(idx.shape) / 2
            expected = reference(x, idx, weights, sorted_indices=sorted_routes)
            actual = disk(x, idx, weights, sorted_indices=sorted_routes)
            mx.eval(expected, actual)
            np.testing.assert_allclose(
                actual.astype(mx.float32),
                expected.astype(mx.float32),
                rtol=0.01,
                atol=0.002,
            )
        assert disk.slots.misses > plan.capacity
    finally:
        plan.close()


@pytest.mark.parametrize("key", ["mtp_enabled", "vlm_mtp_enabled", "dflash_enabled"])
def test_speculative_offload_conflict(key):
    from omlx.model_settings import ModelSettings

    with pytest.raises(ValueError, match="MoE expert offload cannot"):
        ModelSettings(moe_expert_offload_enabled=True, **{key: True})


def test_converted_draft_weights_are_not_loaded_with_offload(tmp_path):
    from mlx.utils import tree_flatten

    source, _ = write_checkpoint(
        tmp_path,
        vision=False,
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
    target = tmp_path / "converted"
    convert(source, target, preserve_mtp=True)
    model, _ = load(target, moe_expert_offload_resident_fraction=0.5)
    try:
        assert not hasattr(model.language_model, "mtp")
        assert not model.config.preserve_mtp
        assert model._moe_offload_plan.draft_bytes > 0
        assert not any(".mtp." in name for name, _ in tree_flatten(model.parameters()))
        mx.eval(model(mx.array([[3, 4, 5]])))
    finally:
        model.close()


@pytest.mark.parametrize("fraction", [0, -0.25, 1.01, float("nan")])
def test_offload_api_rejects_invalid_fraction(fraction):
    from types import SimpleNamespace

    from fastapi import HTTPException

    from omlx.admin.routes import _validate_model_settings

    with pytest.raises(HTTPException) as error:
        _validate_model_settings(
            SimpleNamespace(config_model_type="deepseek_v41"),
            {"moe_expert_offload_resident_fraction": fraction},
        )
    assert error.value.status_code == 400


def test_loader_never_reads_nonresident_stacked_experts(tmp_path, monkeypatch):
    from omlx.patches.deepseek_v41 import loading
    from omlx.patches.deepseek_v41.moe_offload import ExpertOffloadPlan
    from omlx.patches.deepseek_v41.storage import TensorFile

    source, _ = write_checkpoint(tmp_path, vision=False, n_routed_experts=8)
    target = tmp_path / "converted"
    convert(source, target)
    seen = []
    original_read = ExpertOffloadPlan.read

    def read(slab):
        assert slab.expert is not None, "Whole expert tensor materialized"
        seen.append(slab.expert)
        return original_read(slab)

    monkeypatch.setattr(ExpertOffloadPlan, "read", staticmethod(read))
    original_gather = TensorFile.read

    def gather(self, key, rows=None, **kwargs):
        assert ".ffn.experts." not in key, "Expert slab read through the mmap"
        return original_gather(self, key, rows, **kwargs)

    monkeypatch.setattr(TensorFile, "read", gather)

    def no_whole_shards(*args, **kwargs):
        raise AssertionError("Shared shard would materialize nonresident experts")

    monkeypatch.setattr(loading, "_load_shard", no_whole_shards)
    model, _ = load(target, moe_expert_offload_resident_fraction=0.125)
    try:
        assert model._moe_offload_plan.capacity == 2  # Routing top-k floor.
        assert seen and set(seen) == {0}  # Only a one-expert shape/dtype sample.
        mx.eval(model(mx.array([[3, 4, 5]])))
    finally:
        model.close()


@pytest.mark.parametrize("bits", [4, 8])
def test_original_quantized_expert_reads_repack_only_selected_experts(tmp_path, bits):
    from test_deepseek_v41 import raw_safetensors

    from omlx.patches.deepseek_v41.config import ModelConfig
    from omlx.patches.deepseek_v41.language import Expert
    from omlx.patches.deepseek_v41.moe_offload import ExpertOffloadPlan
    from omlx.patches.deepseek_v41.quantization import QuantizedProjection

    mx.random.seed(441)
    config = ModelConfig(
        dim=64, moe_inter_dim=64, n_layers=1, n_routed_experts=8, n_activated_experts=2
    )
    reference = Expert(config, True)
    tensors = {}
    for proj in ("w1", "w3", "w2"):
        w, scales = mx.quantize(
            mx.random.normal((8, 64, 64)), bits=bits, group_size=32, mode=f"mxfp{bits}"
        )
        if bits == 8:
            scales = mx.repeat(scales[:, ::32], 32, axis=1)
        setattr(reference, proj, QuantizedProjection(w, scales, bits, f"mxfp{bits}"))
        for e in range(8):
            name = f"layers.0.ffn.experts.{e}.{proj}"
            tensors[name + ".weight"] = (
                np.asarray(w[e]).view(np.uint8),
                "I8" if bits == 4 else "F8_E4M3",
            )
            tensors[name + ".scale"] = (
                np.asarray(scales[e])[:: 32 if bits == 8 else 1],
                "F8_E8M0",
            )
    raw_safetensors(tmp_path / "model.safetensors", tensors)
    plan = ExpertOffloadPlan(
        tmp_path, {}, {k: "model.safetensors" for k in tensors}, config, 0.25
    )
    disk = OffloadedExpert(
        Expert(config, True), plan, "language_model.layers.0.ffn.experts"
    )
    try:
        for ids in ([[0, 1]], [[2, 3]], [[0, 7]], [[6, 5]]):
            x = mx.random.normal((1, 1, 1, 64)).astype(mx.bfloat16)
            idx = mx.array(ids)
            scores = mx.ones(idx.shape) / 2
            a, b = reference(x, idx, scores), disk(x, idx, scores)
            mx.eval(a, b)
            np.testing.assert_array_equal(a.astype(mx.float32), b.astype(mx.float32))
        assert disk.slots.misses > 2
    finally:
        plan.close()


def test_engram_and_expert_estimates_compose(tmp_path, monkeypatch):
    from test_engine_pool import _make_pool

    from omlx.engine_pool import EngineEntry
    from omlx.model_settings import ModelSettings
    from omlx.patches.deepseek_v41.moe_offload import estimate_expert_savings
    from omlx.patches.deepseek_v41.residency import deepseek_v41_residency_estimate

    source, _ = write_checkpoint(
        tmp_path,
        vision=False,
        n_routed_experts=8,
        engram_layer_ids=(1, 3),
        engram_num_embeddings=(72, 204),
        engram_vocab_size=5,
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_head_dim=32,
        engram_compressed_vocab_size=64,
    )
    target = tmp_path / "converted"
    convert(source, target)
    base = deepseek_v41_residency_estimate(target)
    pool = _make_pool(ceiling=1024**3)
    entry = EngineEntry(
        model_id="v41",
        model_path=str(target),
        model_type="vlm",
        engine_type="vlm",
        config_model_type="deepseek_v41",
        estimated_size=base.resident_bytes,
    )
    settings = ModelSettings(
        moe_expert_offload_enabled=True,
        moe_expert_offload_resident_fraction=0.25,
        deepseek_v41_engram_ssd_offload=True,
    )
    saved = int(estimate_expert_savings(target, 0.25) * 1.05)
    expected = base.mmap_bytes - saved
    assert pool._entry_runtime_resident_size(entry, settings) == expected
    assert (
        pool._entry_runtime_resident_size(
            entry, settings, include_ane_reservation=False
        )
        == expected
    )
    monkeypatch.setenv("OMLX_MOE_EXPERT_OFFLOAD", "0")
    assert pool._entry_runtime_resident_size(entry, settings) == base.mmap_bytes


def _synthetic_affine_experts(tmp_path, fraction, seed=412):
    """A one-layer stacked affine checkpoint with a resident reference Expert."""
    from mlx.utils import tree_flatten

    from omlx.patches.deepseek_v41.config import ModelConfig
    from omlx.patches.deepseek_v41.language import Expert
    from omlx.patches.deepseek_v41.moe_offload import ExpertOffloadPlan
    from omlx.patches.deepseek_v41.quantization import QuantizedProjection

    mx.random.seed(seed)
    config = ModelConfig(
        dim=64, moe_inter_dim=64, n_layers=1, n_routed_experts=8, n_activated_experts=2
    )
    reference = Expert(config, True)
    specs = {}
    for proj in ("w1", "w3", "w2"):
        arrays = mx.quantize(
            mx.random.normal((8, 64, 64)).astype(mx.bfloat16) * 0.05,
            bits=4,
            group_size=32,
            mode="affine",
        )
        setattr(
            reference,
            proj,
            QuantizedProjection(arrays[0], arrays[1], 4, "affine", biases=arrays[2]),
        )
        specs[f"language_model.layers.0.ffn.experts.{proj}"] = dict(
            bits=4, mode="affine"
        )
    prefix = "language_model.layers.0.ffn.experts"
    tensors = {prefix + "." + k: v for k, v in tree_flatten(reference.parameters())}
    mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors)
    mapping = {k: "model.safetensors" for k in tensors}
    raw = {"omlx_deepseek_v41": {"version": 1, "quantized_modules": specs}}
    plan = ExpertOffloadPlan(tmp_path, raw, mapping, config, fraction)
    return reference, OffloadedExpert(Expert(config, True), plan, prefix), plan


def _lru_reference(sequence, capacity):
    """The serial residency policy: hits protect, misses evict the LRU expert."""
    slot_of, free = {}, list(range(capacity))
    for ids in sequence:
        needed = list(dict.fromkeys(ids))
        for e in needed:
            if e in slot_of:
                slot_of[e] = slot_of.pop(e)
        protected = set(needed)
        for e in needed:
            if e not in slot_of:
                slot = (
                    free.pop()
                    if free
                    else slot_of.pop(next(k for k in slot_of if k not in protected))
                )
                slot_of[e] = slot
    return list(slot_of.items())


@pytest.mark.parametrize("inflight", [1, 1 << 30])
def test_parallel_reads_install_in_serial_lru_order(tmp_path, monkeypatch, inflight):
    from omlx.patches.deepseek_v41 import moe_offload

    monkeypatch.setattr(moe_offload, "INFLIGHT_BYTES", inflight)
    _, disk, plan = _synthetic_affine_experts(tmp_path, 0.375)
    try:
        assert plan.capacity == 3
        sequence = [[0, 1, 2], [3, 1], [4, 5, 5], [1, 4], [6, 7, 0], [2]]
        for ids in sequence:
            slots = disk.slots.ensure_ids(ids)
            assert slots == [disk.slots.slot_of[e] for e in ids]
        assert list(disk.slots.slot_of.items()) == _lru_reference(sequence, 3)
        misses = sum(len(dict.fromkeys(ids)) for ids in sequence) - disk.slots.hits
        assert disk.slots.misses == misses == 11
        assert disk.slots.fetched_bytes == misses * plan.expert_bytes
        assert plan.expert_bytes * plan.count == plan.full_bytes
    finally:
        plan.close()
    assert not plan._fds
    with pytest.raises(RuntimeError, match="closed"):
        disk(mx.zeros((1, 1, 64), mx.bfloat16), mx.array([0]), sorted_indices=True)


def test_failed_read_leaves_the_cache_intact(tmp_path, monkeypatch):
    from omlx.patches.deepseek_v41.moe_offload import ExpertOffloadPlan

    _, disk, plan = _synthetic_affine_experts(tmp_path, 0.375)
    original = ExpertOffloadPlan.read
    broken = {"expert": None}

    def read(slab):
        if slab.expert == broken["expert"] and slab.key.endswith("w3.scales"):
            raise OSError("simulated EIO")
        return original(slab)

    monkeypatch.setattr(ExpertOffloadPlan, "read", staticmethod(read))
    try:
        assert disk.slots.ensure_ids([0, 1, 2]) == [2, 1, 0]
        broken["expert"] = 4
        with pytest.raises(OSError, match="EIO"):
            disk.slots.ensure_ids([3, 4])
        # The failing expert and everything queued behind it left no trace:
        # the slot count is intact and only the completed install counts.
        assert len(disk.slots.free) + len(disk.slots.slot_of) == plan.capacity
        assert 4 not in disk.slots.slot_of and 3 in disk.slots.slot_of
        assert disk.slots.misses == 4
        broken["expert"] = None
        assert disk.slots.ensure_ids([4, 5, 6]) == [
            disk.slots.slot_of[e] for e in (4, 5, 6)
        ]
        assert disk.slots.misses == 7
    finally:
        plan.close()
    plan.close()  # Idempotent, and never re-closes a released descriptor.
    assert not plan._fds


def test_sorted_routes_chunk_on_expert_boundaries(tmp_path, monkeypatch):
    from omlx.patches.deepseek_v41.language import Expert

    reference, disk, plan = _synthetic_affine_experts(tmp_path, 0.25)
    calls = []
    original = Expert.__call__

    def counted(self, x, indices=None, *args, **kwargs):
        if self is disk.slots.expert and kwargs.get("sorted_indices"):
            calls.append(indices.size)
        return original(self, x, indices, *args, **kwargs)

    monkeypatch.setattr(Expert, "__call__", counted)
    try:
        ids = sorted([0, 0, 0, 1, 2, 2, 3, 4, 4, 4, 4, 5, 6, 7, 7])
        idx = mx.array(ids)
        x = mx.random.normal((len(ids), 1, 64)).astype(mx.bfloat16)
        weights = mx.ones(idx.shape) / 2
        expected = reference(x, idx, weights, sorted_indices=True)
        actual = disk(x, idx, weights, sorted_indices=True)
        mx.eval(expected, actual)
        np.testing.assert_allclose(
            actual.astype(mx.float32),
            expected.astype(mx.float32),
            rtol=0.01,
            atol=0.002,
        )
        # Eight experts at two resident slots: every route of two experts per
        # chunk, never a chunk cut inside an expert's run.
        assert calls == [4, 3, 5, 3]
        assert disk.slots.misses == 8 and disk.slots.hits == 0
        # A second sweep evicts the last resident pair before reaching it:
        # sorted routes over more experts than slots thrash by construction.
        disk(x, idx, weights, sorted_indices=True)
        assert disk.slots.misses == 16 and disk.slots.hits == 0
    finally:
        plan.close()


@pytest.mark.parametrize("engram", [False, True])
def test_admission_and_fit_match_the_engine_pool(tmp_path, engram):
    from test_engine_pool import _make_pool

    from omlx.engine_pool import EngineEntry
    from omlx.model_discovery import estimate_model_size
    from omlx.model_settings import ModelSettings
    from omlx.patches.deepseek_v41.moe_offload import (
        admission_bytes,
        fit_resident_fraction,
    )

    kwargs = dict(n_routed_experts=8, n_activated_experts=2)
    if engram:
        kwargs.update(
            engram_layer_ids=(1, 3),
            engram_num_embeddings=(72, 204),
            engram_vocab_size=5,
            engram_max_ngram_size=4,
            engram_n_heads=2,
            engram_head_dim=32,
            engram_compressed_vocab_size=64,
        )
    source, _ = write_checkpoint(tmp_path, vision=False, **kwargs)
    target = tmp_path / "converted"
    convert(source, target)
    pool = _make_pool(ceiling=1024**3)
    entry = EngineEntry(
        model_id="v41",
        model_path=str(target),
        model_type="vlm",
        engine_type="vlm",
        config_model_type="deepseek_v41",
        estimated_size=estimate_model_size(target),
    )
    assert fit_resident_fraction(target, 1 << 60, engram_ssd_offload=engram) == 1.0
    assert fit_resident_fraction(target, 0, engram_ssd_offload=engram) is None
    for capacity in range(2, 9):
        fraction = capacity / 8
        settings = ModelSettings(
            moe_expert_offload_enabled=True,
            moe_expert_offload_resident_fraction=fraction,
            deepseek_v41_engram_ssd_offload=engram,
        )
        expected = pool._entry_runtime_resident_size(entry, settings)
        assert expected > 0
        assert admission_bytes(target, fraction, engram_ssd_offload=engram) == expected
        assert fit_resident_fraction(target, expected, engram_ssd_offload=engram) == (
            fraction
        )
        assert fit_resident_fraction(
            target, expected - 1, engram_ssd_offload=engram
        ) == ((capacity - 1) / 8 if capacity > 2 else None)


@pytest.mark.parametrize("window_experts", [1, 2])
def test_consumed_read_buffers_are_released_within_window(
    tmp_path, monkeypatch, window_experts
):
    from omlx.patches.deepseek_v41 import moe_offload

    _, disk, plan = _synthetic_affine_experts(tmp_path, 1.0)
    budget = window_experts * plan.expert_bytes
    monkeypatch.setattr(moe_offload, "INFLIGHT_BYTES", budget)
    refs, samples = [], []
    original_decode = plan.decode

    class TrackedBuffer(bytearray):
        pass

    def allocate(size):
        buffer = TrackedBuffer(size)
        refs.append((weakref.ref(buffer), size))
        return buffer

    def decode(slabs, raws):
        samples.append(sum(size for ref, size in refs if ref() is not None))
        return original_decode(slabs, raws)

    monkeypatch.setattr(moe_offload, "bytearray", allocate, raising=False)
    monkeypatch.setattr(plan, "decode", decode)
    try:
        disk.slots.ensure_ids(list(range(plan.count)))
        assert disk.slots.misses == plan.count
        assert max(samples) <= budget
        assert all(ref() is None for ref, _ in refs)
    finally:
        plan.close()
