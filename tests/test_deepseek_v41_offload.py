# SPDX-License-Identifier: Apache-2.0
"""Resident/SSD equivalence and bounded Engram prefetch lifecycle."""

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event, current_thread

import mlx.core as mx
import numpy as np
import pytest
from test_deepseek_v41 import raw_safetensors, write_checkpoint

from omlx.patches.deepseek_v41 import loading, storage
from omlx.patches.deepseek_v41.convert import convert
from omlx.patches.deepseek_v41.loading import load
from omlx.patches.deepseek_v41.residency import deepseek_v41_residency_estimate
from omlx.patches.deepseek_v41.storage import DiskEngramEmbedding, EngramPrefetch


def table(tmp_path):
    path = tmp_path / "rows.safetensors"
    raw = np.arange(16 * 64, dtype=np.uint8).reshape(16, 64) % 120
    scales = np.arange(32, dtype=np.uint8).reshape(16, 2) + 110
    raw_safetensors(path, {"w": (raw, "F8_E4M3"), "s": (scales, "F8_E8M0")})
    return DiskEngramEmbedding(path, "w", "s")


def test_packed_resident_and_prefetched_rows_match(tmp_path, monkeypatch):
    disk = table(tmp_path)
    resident = table(tmp_path)
    resident.make_resident()
    prefetch = EngramPrefetch()
    try:
        assert resident._weights._mapping is None
        assert resident._resident["w"][0].nbytes == 16 * 64
        ids = np.array([[5, 1, 5, 0]])
        expected = np.asarray(resident(mx.array(ids)).astype(mx.float32))
        original = disk._read_rows
        threads = []

        def read(rows):
            threads.append(current_thread().name)
            return original(rows)

        monkeypatch.setattr(disk, "_read_rows", read)
        with prefetch.forward():
            prefetch.submit(disk, ids)
            actual = disk(mx.array(ids))
        np.testing.assert_array_equal(actual.astype(mx.float32), expected)
        assert len(threads) == 1 and threads[0].startswith("v41-engram")
        assert disk._prefetched is None and prefetch._pending is None
        with prefetch.forward():
            prefetch.submit(disk, ids)
            # A mismatched request must never consume stale row data.
            actual = disk(mx.array([[2, 3]]))
        np.testing.assert_array_equal(
            actual.astype(mx.float32), resident(mx.array([[2, 3]])).astype(mx.float32)
        )
        monkeypatch.setattr(storage, "PREFETCH_BYTES", 1)
        prefetch.submit(disk, ids)
        assert prefetch._pending is None
        prefetch.submit(resident, ids)
        assert prefetch._pending is None
    finally:
        prefetch.close()
        disk.close()
        resident.close()


def test_prefetch_drains_before_unmap(tmp_path, monkeypatch):
    disk = table(tmp_path)
    prefetch = EngramPrefetch()
    entered, release, closing = Event(), Event(), Event()
    original = disk._read_rows

    def read(rows):
        entered.set()
        assert release.wait(5)
        assert disk._weights._mapping is not None
        return original(rows)

    monkeypatch.setattr(disk, "_read_rows", read)
    prefetch.submit(disk, np.array([1]))
    assert entered.wait(5)

    def close():
        closing.set()
        prefetch.close()
        disk.close()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(close)
        try:
            assert closing.wait(5)
            assert not future.done()
        finally:
            release.set()
        future.result(timeout=5)
    assert disk._weights._mapping is None
    assert prefetch._pending is None
    prefetch.close()
    disk.close()


def test_prefetch_forward_error_releases_pending_rows(tmp_path):
    disk = table(tmp_path)
    prefetch = EngramPrefetch()
    try:
        with pytest.raises(ValueError, match="cancelled"), prefetch.forward():
            prefetch.submit(disk, np.array([1, 2]))
            raise ValueError("cancelled")
        assert prefetch._pending is None and disk._prefetched is None
        # Subsequent requests remain usable.
        with prefetch.forward():
            prefetch.submit(disk, np.array([3]))
            mx.eval(disk(mx.array([3])))
    finally:
        prefetch.close()
        disk.close()


@pytest.fixture
def converted(tmp_path):
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
    target = tmp_path / "converted"
    convert(source, target)
    return target


def test_converted_modes_match_chunked_prefill_decode_and_cleanup(converted):
    resident, _ = load(converted, engram_ssd_offload=False)
    disk, _ = load(converted, engram_ssd_offload=True)
    try:
        caches = [model.language_model.make_cache() for model in (resident, disk)]
        for ids in ([[3, 4, 5]], [[6, 7]], [[8]], [[9]], [[10]]):
            outputs = [
                model(mx.array(ids), cache=cache)
                for model, cache in zip((resident, disk), caches)
            ]
            mx.eval(outputs)
            np.testing.assert_array_equal(outputs[0], outputs[1])
            assert disk.language_model._engram_prefetch._pending is None
            assert resident.language_model._engram_prefetch._pending is None
            assert all(
                layer.engram.embed._resident is not None
                for layer in resident.language_model.layers
                if "engram" in layer
            )
        from omlx.models.vlm import VLMModelAdapter

        VLMModelAdapter(disk).release_resources()
        for layer in disk.language_model.layers:
            if "engram" in layer:
                assert layer.engram.embed._weights._mapping is None
        assert disk.language_model._engram_prefetch._closed
    finally:
        resident.close()
        disk.close()


def test_residency_counts_indexed_tensors_not_linked_shard_bytes(converted):
    estimate = deepseek_v41_residency_estimate(converted)
    spec = json.loads((converted / "config.json").read_text())["omlx_deepseek_v41"]
    expected = (72 + 204) * 32 * 4  # Fixture uses float32 embeddings.
    assert estimate.engram_bytes == expected
    assert estimate.supported
    assert estimate.resident_bytes > expected
    # Files copied from the source contain unrelated tensors too.
    assert (
        sum(
            (converted / t["weight_file"]).stat().st_size
            for t in spec["engram_tables"].values()
        )
        > expected
    )
    # Tiny fixtures cannot save enough RAM to cover the bounded I/O allowance.
    assert not estimate.force_ssd_offload(estimate.resident_bytes - 1)


@pytest.mark.parametrize(
    "ceiling,requested,expected,forced",
    [
        (0, False, False, False),
        (1000, False, False, False),
        (500, False, True, True),
        (300, False, False, False),
        (1000, True, True, False),
    ],
)
def test_memory_decision_and_reload_signature(
    monkeypatch, ceiling, requested, expected, forced
):
    from test_engine_pool import _make_pool

    from omlx.engine_pool import EngineEntry
    from omlx.model_settings import ModelSettings
    from omlx.patches.deepseek_v41.residency import EngramResidencyEstimate

    estimate = EngramResidencyEstimate(True, 1000, 400, 600)
    monkeypatch.setattr(
        "omlx.patches.deepseek_v41.residency.deepseek_v41_residency_estimate",
        lambda _: estimate,
    )
    pool = _make_pool(ceiling=ceiling)
    entry = EngineEntry(
        model_id="v41",
        model_path="/fixture",
        model_type="vlm",
        engine_type="vlm",
        config_model_type="deepseek_v41",
        estimated_size=10,
    )
    pool._entries["v41"] = entry
    settings = ModelSettings(deepseek_v41_engram_ssd_offload=requested)
    enabled, actual_forced, _ = pool._deepseek_v41_engram_offload_status(
        entry, settings
    )
    assert (enabled, actual_forced) == (expected, forced)
    assert pool._entry_runtime_resident_size(entry, settings) == (
        400 if expected else 1000
    )
    effective = pool._effective_deepseek_v41_model_settings(entry, settings)
    assert effective.deepseek_v41_engram_ssd_offload == expected
    assert settings.deepseek_v41_engram_ssd_offload == requested
    if forced:
        assert pool._effective_deepseek_v41_model_settings(
            entry, None
        ).deepseek_v41_engram_ssd_offload
    if ceiling == 1000:
        before = pool._engine_runtime_signature("v41", settings)
        settings.deepseek_v41_engram_ssd_offload = not requested
        assert before != pool._engine_runtime_signature("v41", settings)


def test_setting_roundtrip_excludes_shared_profile(tmp_path):
    from omlx.model_settings import ModelSettingsManager

    manager = ModelSettingsManager(tmp_path)
    settings = manager.get_settings("v41")
    settings.deepseek_v41_engram_ssd_offload = True
    manager.set_settings("v41", settings)
    restored = ModelSettingsManager(tmp_path).get_settings("v41")
    assert restored.deepseek_v41_engram_ssd_offload
    manager.save_profile("v41", "sample", "Sample", None, restored.to_dict())
    profile = manager.list_profiles("v41")[0]
    assert "deepseek_v41_engram_ssd_offload" not in profile["settings"]


def test_loader_failure_closes_all_tables(converted, monkeypatch):
    tables = []
    original = DiskEngramEmbedding.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        tables.append(self)

    def fail(*args, **kwargs):
        raise ValueError("tokenizer failure")

    monkeypatch.setattr(DiskEngramEmbedding, "__init__", init)
    monkeypatch.setattr(
        "omlx.patches.deepseek_v41.loading.PreTrainedTokenizerFast.from_pretrained",
        fail,
    )
    with pytest.raises(ValueError, match="tokenizer failure"):
        load(converted, engram_ssd_offload=True)
    assert len(tables) == 2
    assert all(t._closed and t._weights._mapping is None for t in tables)


def test_model_close_unmaps_even_after_prefetch_failure(converted, monkeypatch):
    model, _ = load(converted, engram_ssd_offload=True)
    prefetch = model.language_model._engram_prefetch
    embed = model.language_model.layers[1].engram.embed
    entered = Event()

    def fail(rows):
        entered.set()
        raise OSError("read failed")

    monkeypatch.setattr(embed, "_read_rows", fail)
    prefetch.submit(embed, np.array([1]))
    assert entered.wait(5)
    with pytest.raises(OSError, match="read failed"):
        model.close()
    for layer in model.language_model.layers:
        if "engram" in layer:
            assert layer.engram.embed._weights._mapping is None
    model.close()


@pytest.mark.parametrize("offload", [False, True])
def test_original_checkpoint_load_matches_export_without_writing(
    converted, offload, monkeypatch
):
    source = converted.parent / "source"
    before = {
        str(p): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in source.rglob("*")
        if p.is_file()
    }
    exported, _ = load(converted, engram_ssd_offload=offload)

    def no_export(*args, **kwargs):
        raise AssertionError("Direct loading must not export tensors")

    monkeypatch.setattr(mx, "save_safetensors", no_export)
    original, _ = load(source, engram_ssd_offload=offload)
    try:
        caches = [m.language_model.make_cache() for m in (exported, original)]
        for ids in ([[3, 4, 5]], [[6]], [[7]]):
            actual = [
                m(mx.array(ids), cache=c) for m, c in zip((exported, original), caches)
            ]
            mx.eval(actual)
            np.testing.assert_array_equal(*actual)
        assert deepseek_v41_residency_estimate(
            source
        ) == deepseek_v41_residency_estimate(converted)
        after = {
            str(p): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in source.rglob("*")
            if p.is_file()
        }
        assert before == after
    finally:
        exported.close()
        original.close()


@pytest.mark.parametrize("bits", [4, 8])
def test_direct_quantized_experts_and_dense_projection_match_export(converted, bits):
    source = converted.parent / "source"
    filename = source / "model.safetensors"
    data = mx.load(str(filename))
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    packed = {}
    for key in list(data):
        if not (
            (key.startswith("layers.0.ffn.experts.") and key.endswith(".w1.weight"))
            or key == "layers.0.attn.wo_a.weight"
        ):
            continue
        shape = data.pop(key).shape
        width = shape[1] // 2 if bits == 4 and ".experts." in key else shape[1]
        fp4 = width != shape[1]
        raw = np.full((shape[0], width), 0x22, dtype=np.int8 if fp4 else np.uint8)
        scale = np.full(
            (shape[0] if fp4 else (shape[0] + 31) // 32, shape[1] // 32),
            124,
            dtype=np.uint8,
        )
        scale_key = key.removesuffix(".weight") + ".scale"
        packed[key] = (raw, "I8" if fp4 else "F8_E4M3")
        packed[scale_key] = (scale, "F8_E8M0")
        index["weight_map"][key] = "quant.safetensors"
        index["weight_map"][scale_key] = "quant.safetensors"
    mx.eval(data)
    mx.save_safetensors(str(filename), data)
    raw_safetensors(source / "quant.safetensors", packed)
    index_path.write_text(json.dumps(index))
    target = converted.parent / f"quantized-{bits}"
    convert(source, target)
    direct, _ = load(source, engram_ssd_offload=True)
    exported, _ = load(target, engram_ssd_offload=True)
    try:
        assert direct.language_model.layers[0].ffn.experts.w1.bits == bits
        assert deepseek_v41_residency_estimate(
            source
        ) == deepseek_v41_residency_estimate(target)
        outputs = [m(mx.array([[3, 4, 5]])) for m in (direct, exported)]
        mx.eval(outputs)
        assert bool(mx.all(mx.isfinite(outputs[0])).item())
        np.testing.assert_array_equal(*outputs)
    finally:
        direct.close()
        exported.close()


def test_source_iterator_releases_mappings_before_yield(converted, monkeypatch):
    from omlx.patches.deepseek_v41 import convert as converter

    source = converted.parent / "source"
    config = json.loads((source / "config.json").read_text())
    mapping = json.loads((source / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    readers = []
    original = converter.TensorFile

    def open_reader(path):
        reader = original(path)
        readers.append(reader)
        return reader

    monkeypatch.setattr(converter, "TensorFile", open_reader)
    stream = converter.iter_source_weights(source, config, mapping)
    try:
        for values, _ in stream:
            assert readers and all(reader._mapping is None for reader in readers)
            mx.eval(values)
    finally:
        stream.close()


def test_residency_tracks_stored_head_precision(converted):
    before = deepseek_v41_residency_estimate(converted)
    mapping = json.loads((converted / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    name = "language_model.head.weight"
    filename = converted / mapping[name]
    weights = mx.load(str(filename))
    assert weights[name].dtype == mx.float32
    weights[name] = weights[name].astype(mx.bfloat16)
    mx.eval(weights)
    mx.save_safetensors(str(filename), weights)
    after = deepseek_v41_residency_estimate(converted)
    assert before.resident_bytes > after.resident_bytes
    assert before.mmap_bytes > after.mmap_bytes
    assert before.engram_bytes == after.engram_bytes


@pytest.mark.parametrize("short_read", [False, True])
def test_resident_reads_are_bounded_and_preserve_bytes(
    tmp_path, monkeypatch, short_read
):
    disk = table(tmp_path)
    reader = disk._weights
    expected, dtype = reader.read("w", np.arange(16))
    source = reader._file
    requests = []

    class TrackedFile:
        def fileno(self):
            return source.fileno()

        def seek(self, offset):
            return source.seek(offset)

        def readinto(self, buffer):
            requests.append(len(buffer))
            return source.readinto(buffer[:7] if short_read else buffer)

        def close(self):
            source.close()

    reader._file = TrackedFile()
    monkeypatch.setattr(storage, "RESIDENT_READ_BYTES", 31)
    try:
        actual, actual_dtype = reader.read("w")
        assert actual_dtype == dtype
        np.testing.assert_array_equal(actual, expected)
        assert len(requests) > 1 and max(requests) <= 31
        assert actual.flags.owndata
    finally:
        disk.close()


@pytest.mark.parametrize("failed_key", ["w", "s"])
def test_resident_read_rejects_unexpected_eof(tmp_path, monkeypatch, failed_key):
    disk = table(tmp_path)
    reader = disk._weights
    source = reader._file
    original = source.readinto
    fail_at = reader._start + reader.header[failed_key]["data_offsets"][0]

    def readinto(buffer):
        return 0 if source.tell() >= fail_at else original(buffer)

    monkeypatch.setattr(source, "readinto", readinto)
    toggles = []
    original_fcntl = storage.fcntl.fcntl

    def tracked_fcntl(fd, command, value):
        toggles.append(value)
        return original_fcntl(fd, command, value)

    monkeypatch.setattr(storage.fcntl, "fcntl", tracked_fcntl)
    try:
        with pytest.raises(ValueError, match="Truncated tensor data"):
            disk.make_resident()
        assert disk._resident is None
        if hasattr(storage.fcntl, "F_NOCACHE"):
            assert toggles and toggles == [1, 0] * (len(toggles) // 2)
    finally:
        disk.close()


def test_empty_resident_tensor(tmp_path):
    path = tmp_path / "empty.safetensors"
    raw_safetensors(path, {"w": (np.empty((0, 32), np.uint8), "U8")})
    reader = storage.TensorFile(path)
    try:
        result, dtype = reader.read("w")
        assert result.shape == (0, 32) and dtype == "U8"
    finally:
        reader.close()


@pytest.mark.parametrize("fail", [False, True])
def test_resident_load_restores_wired_limit(converted, monkeypatch, fail):
    def fail_tokenizer(*args, **kwargs):
        raise ValueError("tokenizer failure")

    if fail:
        monkeypatch.setattr(
            loading.PreTrainedTokenizerFast, "from_pretrained", fail_tokenizer
        )
    previous = mx.set_wired_limit(0)
    model = None
    try:
        if fail:
            with pytest.raises(ValueError, match="tokenizer failure"):
                load(converted, engram_ssd_offload=False)
        else:
            model, _ = load(converted, engram_ssd_offload=False)
        assert mx.set_wired_limit(0) == 0
    finally:
        if model is not None:
            model.close()
        mx.synchronize()
        mx.set_wired_limit(previous)


def test_resident_storage_counts_packed_bytes_and_releases_on_close(tmp_path):
    import gc

    path = tmp_path / "packed.safetensors"
    weight = np.arange(8192 * 32, dtype=np.uint32).reshape(8192, 32)
    scale = np.full((8192, 8), 0x3F80, dtype=np.uint16)
    raw_safetensors(path, {"w": (weight, "U32"), "s": (scale, "BF16")})
    disk = DiskEngramEmbedding(path, "w", "s")
    mx.synchronize()
    gc.collect()
    before = mx.get_active_memory()
    try:
        disk.make_resident()
        assert mx.get_active_memory() >= before + weight.nbytes + scale.nbytes
        assert not disk._resident["w"][0].flags.owndata
        np.testing.assert_array_equal(disk._resident["w"][0], weight)
        np.testing.assert_array_equal(disk._resident["s"][0], scale)
        assert disk._resident["s"][1] == "BF16"
        assert disk._weights._mapping is None
    finally:
        disk.close()
    mx.synchronize()
    gc.collect()
    assert mx.get_active_memory() <= before


def test_resident_engram_follows_backbone_load(converted, monkeypatch):
    events = []
    original_weights = loading.Model.load_weights
    original_resident = DiskEngramEmbedding.make_resident

    def weights(self, *args, **kwargs):
        result = original_weights(self, *args, **kwargs)
        events.append("weights")
        return result

    def resident(self):
        events.append("resident")
        return original_resident(self)

    monkeypatch.setattr(loading.Model, "load_weights", weights)
    monkeypatch.setattr(DiskEngramEmbedding, "make_resident", resident)
    model, _ = load(converted, engram_ssd_offload=False)
    try:
        assert "weights" in events and "resident" in events
        first_resident = events.index("resident")
        assert all(event == "weights" for event in events[:first_resident])
        assert all(event == "resident" for event in events[first_resident:])
        logits = model.language_model(mx.array([[3, 4, 5]]))
        assert mx.all(mx.isfinite(logits)).item()
    finally:
        model.close()


def test_shard_reader_preserves_packed_bytes_after_close(tmp_path, monkeypatch):
    path = tmp_path / "packed.safetensors"
    expected = {
        "weight": mx.arange(128, dtype=mx.uint32).reshape(4, 32),
        "scales": mx.array([[0.5, 1.0], [2.0, 4.0]], dtype=mx.bfloat16),
    }
    mx.save_safetensors(str(path), expected)
    original = mx.load
    readers = []

    def tracked_load(reader):
        readers.append(reader)
        assert not reader.closed
        return original(reader)

    monkeypatch.setattr(loading.mx, "load", tracked_load)
    actual = loading._load_shard(path)
    assert len(readers) == 1 and readers[0].closed
    for name, value in expected.items():
        assert actual[name].dtype == value.dtype
        assert actual[name].shape == value.shape
        assert mx.array_equal(actual[name], value).item()


def test_shard_reader_closes_on_invalid_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "invalid.safetensors"
    path.write_bytes(b"not a checkpoint")
    original = mx.load
    readers = []

    def tracked_load(reader):
        readers.append(reader)
        return original(reader)

    monkeypatch.setattr(loading.mx, "load", tracked_load)
    with pytest.raises((ValueError, RuntimeError)):
        loading._load_shard(path)
    assert len(readers) == 1 and readers[0].closed


def page_table(tmp_path):
    path = tmp_path / "pages.safetensors"
    raw = np.arange(4096 * 64, dtype=np.uint8).reshape(4096, 64)
    raw_safetensors(path, {"w": (raw, "U8")})
    return storage.TensorFile(path), raw


def test_parallel_pages_preserve_order_short_reads_and_warm_hits(tmp_path, monkeypatch):
    reader, raw = page_table(tmp_path)
    original = storage.os.pread
    calls = []

    def short_read(fd, count, offset):
        calls.append((count, offset))
        return original(fd, min(count, 4096), offset)

    monkeypatch.setattr(storage.os, "pread", short_read)
    ids = np.tile(np.array([4095, 0, 255, 256, 1023, 256]), 32)
    try:
        actual, dtype = reader.read("w", ids)
        assert dtype == "U8" and calls
        np.testing.assert_array_equal(actual, raw[ids])
        count = len(calls)
        repeated, _ = reader.read("w", ids)
        assert len(calls) == count
        np.testing.assert_array_equal(repeated, actual)
    finally:
        reader.close()
    # A returned row copy cannot retain or depend on the closed file mapping.
    np.testing.assert_array_equal(actual, raw[ids])
    assert reader._seen_pages is None


def test_decode_rows_skip_page_prefetch(tmp_path, monkeypatch):
    reader, raw = page_table(tmp_path)

    def unexpected(*args):
        raise AssertionError("Decode rows must retain the direct mmap path")

    monkeypatch.setattr(storage.os, "pread", unexpected)
    try:
        ids = np.arange(24)
        actual, _ = reader.read("w", ids)
        np.testing.assert_array_equal(actual, raw[ids])
        assert reader._seen_pages is None
    finally:
        reader.close()


def test_page_read_failure_drains_workers_before_close(tmp_path, monkeypatch):
    reader, _ = page_table(tmp_path)
    original = storage.os.pread
    entered, release, closing = Event(), Event(), Event()
    monkeypatch.setattr(storage, "PAGE_IO_WORKERS", 2)

    def blocked_read(fd, count, offset):
        if offset == 0:
            raise OSError("page read failed")
        entered.set()
        assert release.wait(5)
        assert not reader._file.closed
        return original(fd, count, offset)

    def close():
        closing.set()
        reader.close()

    monkeypatch.setattr(storage.os, "pread", blocked_read)
    with ThreadPoolExecutor(max_workers=2) as executor:
        reading = executor.submit(reader.read, "w", np.arange(4096))
        try:
            assert entered.wait(5)
            closed = executor.submit(close)
            assert closing.wait(5)
            assert not closed.done()
        finally:
            release.set()
        with pytest.raises(OSError, match="page read failed"):
            reading.result(timeout=5)
        closed.result(timeout=5)
    assert reader._file.closed and reader._mapping is None


def test_slow_warm_gather_rearms_page_prefetch(tmp_path, monkeypatch):
    reader, _ = page_table(tmp_path)
    ids = np.arange(256)
    try:
        reader.read("w", ids)
        assert reader._seen_pages is not None
        ticks = iter([0.0, 1.0])
        monkeypatch.setattr(storage.time, "perf_counter", lambda: next(ticks))
        monkeypatch.setattr(storage.time, "monotonic", lambda: 100.0)
        reader.read("w", ids)
        assert reader._seen_pages is None
        reader.read("w", ids)
        assert reader._seen_pages is not None
    finally:
        reader.close()
