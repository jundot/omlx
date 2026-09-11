# SPDX-License-Identifier: Apache-2.0
"""Sidecar lifecycle, failures, and races against the real SSD writer."""

import json
import queue
import stat
import threading
from pathlib import Path

import mlx.core as mx
import pytest
from tokenizers import Tokenizer, models

from omlx.cache.inspection import BlockInspection, InspectionRenderer
from omlx.cache.inspection_files import cleanup_orphans, sidecar_paths
from omlx.cache.paged_cache import compute_block_hash
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager


@pytest.fixture
def renderer():
    return InspectionRenderer(
        Tokenizer(models.WordLevel({"hello": 0, "world": 1})), "test"
    )


def make_manager(tmp_path, renderer=None, **kwargs):
    return PagedSSDCacheManager(
        tmp_path / "cache", 100 * 1024**2, inspection_renderer=renderer, **kwargs
    )


def save(manager, tokens=(0, 1), parent=None, start=0):
    block_hash = compute_block_hash(parent, list(tokens), model_name="test")
    arrays = [(mx.zeros((1, 1, len(tokens), 4)), mx.ones((1, 1, len(tokens), 4)))]
    assert manager.save_block(
        block_hash,
        arrays,
        len(tokens),
        "test",
        layer_cache_types=["KVCache"],
        inspection=BlockInspection(
            tuple(tokens), start, parent.hex() if parent else None
        ),
    )
    return block_hash, manager._get_file_path(block_hash)


def drain(manager):
    # FIFO barrier on the one SSD writer, without sleeps or model work.
    done = threading.Event()
    original = manager._write_inspection
    from omlx.cache.paged_ssd_cache import PagedSSDBlockMetadata, _InspectionWrite

    metadata = PagedSSDBlockMetadata(b"barrier", Path("/nonexistent"), 0, 0, 0, 0, 0)

    def barrier(meta, inspection):
        if meta is metadata:
            done.set()
            return
        return original(meta, inspection)

    manager._write_inspection = barrier
    try:
        manager._write_queue.put(
            _InspectionWrite(metadata, BlockInspection((), 0, None)), timeout=5
        )
        assert done.wait(5)
    finally:
        manager._write_inspection = original


def test_enabled_saves_all_three_and_accounts_for_sidecars(tmp_path, renderer):
    manager = make_manager(tmp_path, renderer)
    try:
        block_hash, path = save(manager)
        drain(manager)
        paths = (path, *sidecar_paths(path))
        assert all(p.is_file() for p in paths)
        assert manager._index.total_size == sum(p.stat().st_size for p in paths)
        assert json.loads(paths[1].read_text())["token_ids"] == [0, 1]
        assert "hello world" in paths[2].read_text()
        assert stat.S_IMODE(paths[1].stat().st_mode) == 0o600
        assert manager.load_block(block_hash) is not None
        assert manager.get_stats_dict()["inspection_writes"] == 1
    finally:
        manager.close()


def test_disabled_does_not_create_sidecars(tmp_path):
    manager = make_manager(tmp_path)
    _, path = save(manager)
    manager.close()
    assert path.exists()
    assert not any(p.exists() for p in sidecar_paths(path))


@pytest.mark.parametrize("write_through", [False, True])
def test_hot_cache_follows_tensor_persistence(tmp_path, renderer, write_through):
    manager = make_manager(
        tmp_path,
        renderer,
        hot_cache_max_bytes=1024**2,
        hot_cache_write_through=write_through,
    )
    _, path = save(manager)
    if not write_through:
        assert not path.exists()
        assert not any(p.exists() for p in sidecar_paths(path))
    else:
        drain(manager)
        assert all(p.exists() for p in (path, *sidecar_paths(path)))
    manager.close()
    assert all(p.exists() for p in (path, *sidecar_paths(path)))


def test_hot_only_never_writes(tmp_path, renderer):
    manager = make_manager(
        tmp_path, renderer, hot_cache_only=True, hot_cache_max_bytes=1024**2
    )
    save(manager)
    manager.close()
    assert not (tmp_path / "cache").exists()


def test_existing_block_backfills_without_rewriting_tensors(tmp_path, renderer):
    manager = make_manager(tmp_path)
    block_hash, path = save(manager)
    manager.close()
    before = path.stat().st_mtime_ns
    manager = make_manager(tmp_path, renderer)
    try:
        manager.backfill_inspection(block_hash, BlockInspection((0, 1), 0, None))
        drain(manager)
        assert path.stat().st_mtime_ns == before
        assert all(p.exists() for p in sidecar_paths(path))
        sidecar_times = [p.stat().st_mtime_ns for p in sidecar_paths(path)]
        manager.backfill_inspection(block_hash, BlockInspection((0, 1), 0, None))
        drain(manager)
        assert [p.stat().st_mtime_ns for p in sidecar_paths(path)] == sidecar_times
        assert manager.get_stats_dict()["inspection_writes"] == 1
    finally:
        manager.close()


@pytest.mark.parametrize("suffix", [".tokens", ".txt"])
def test_sidecar_failure_preserves_kv_and_partial_size(
    tmp_path, renderer, monkeypatch, suffix
):
    import omlx.cache.paged_ssd_cache as ssd

    original = ssd.atomic_write

    def fail(path, content):
        if path.suffix == suffix:
            raise OSError("simulated write failure")
        original(path, content)

    monkeypatch.setattr(ssd, "atomic_write", fail)
    manager = make_manager(tmp_path, renderer)
    try:
        block_hash, path = save(manager)
        drain(manager)
        assert manager.load_block(block_hash) is not None
        assert manager.get_stats_dict()["inspection_errors"] == 1
        assert not manager._index.get(block_hash).inspection_complete
        assert manager._index.total_size == sum(
            p.stat().st_size for p in (path, *sidecar_paths(path)) if p.exists()
        )
        assert manager.delete_block(block_hash)
        assert not any(p.exists() for p in (path, *sidecar_paths(path)))
    finally:
        manager.close()


@pytest.mark.parametrize("operation", ["delete", "clear", "evict"])
def test_eviction_during_render_cannot_resurrect_files(tmp_path, renderer, operation):
    entered, release = threading.Event(), threading.Event()
    original = renderer.render

    def blocked(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)

    renderer.render = blocked
    manager = make_manager(tmp_path, renderer)
    try:
        block_hash, path = save(manager)
        assert entered.wait(5)
        if operation == "delete":
            assert manager.delete_block(block_hash)
        elif operation == "clear":
            manager.clear()
        else:
            manager._enforce_size_limit_for_new_block(manager.max_size)
        release.set()
        drain(manager)
        assert not any(p.exists() for p in (path, *sidecar_paths(path)))
    finally:
        release.set()
        manager.close()


def test_restart_counts_sidecars_even_when_disabled_and_clear_removes_them(
    tmp_path, renderer
):
    manager = make_manager(tmp_path, renderer)
    _, path = save(manager)
    manager.close()
    manager = make_manager(tmp_path)
    assert manager._index.total_size == sum(
        p.stat().st_size for p in (path, *sidecar_paths(path))
    )
    manager.clear()
    manager.close()
    assert not any(p.exists() for p in (path, *sidecar_paths(path)))


def test_orphan_cleanup_preserves_unrelated_files(tmp_path):
    directory = tmp_path / "a"
    directory.mkdir()
    stem = "a" * 64
    orphan = directory / f"{stem}.tokens"
    temporary = directory / f".{stem}.random.inspection-tmp"
    unrelated = directory / "notes.txt"
    for path in (orphan, temporary, unrelated):
        path.write_text("test")
    cleanup_orphans(directory)
    assert not orphan.exists() and not temporary.exists()
    assert unrelated.exists()


def test_full_backfill_queue_does_not_block_or_rewrite(tmp_path, renderer, monkeypatch):
    manager = make_manager(tmp_path)
    block_hash, path = save(manager)
    manager.close()
    manager = make_manager(tmp_path, renderer)
    try:

        def full(*args):
            raise queue.Full

        monkeypatch.setattr(manager._write_queue, "put_nowait", full)
        manager.backfill_inspection(block_hash, BlockInspection((0, 1), 0, None))
        assert not manager._inspection_pending
        assert path.exists()
        assert manager.get_stats_dict()["inspection_backfill_drops"] == 1
    finally:
        monkeypatch.undo()
        manager.close()


def test_clear_discards_unflushed_hot_entries(tmp_path, renderer):
    manager = make_manager(tmp_path, renderer, hot_cache_max_bytes=1024**2)
    _, path = save(manager)
    manager.clear()
    manager.close()
    assert not any(p.exists() for p in (path, *sidecar_paths(path)))


def test_corrupt_sidecars_do_not_affect_kv_restore(tmp_path, renderer):
    manager = make_manager(tmp_path, renderer)
    block_hash, path = save(manager)
    manager.close()
    path.with_suffix(".tokens").write_bytes(b"broken json")
    path.with_suffix(".txt").write_bytes(b"\xff")
    manager = make_manager(tmp_path, renderer)
    try:
        assert manager.load_block(block_hash) is not None
    finally:
        manager.close()


def test_failed_unlink_remains_tracked_and_can_be_retried(
    tmp_path, renderer, monkeypatch
):
    manager = make_manager(tmp_path, renderer)
    block_hash, path = save(manager)
    drain(manager)
    original = Path.unlink

    def fail(self, *args, **kwargs):
        if self == path.with_suffix(".txt"):
            raise PermissionError("test")
        return original(self, *args, **kwargs)

    try:
        with monkeypatch.context() as context:
            context.setattr(Path, "unlink", fail)
            assert not manager.delete_block(block_hash)
            assert manager._index.contains(block_hash)
        assert manager.delete_block(block_hash)
        assert not any(p.exists() for p in (path, *sidecar_paths(path)))
    finally:
        manager.close()


def test_inline_fallback_writes_sidecars(tmp_path, renderer, monkeypatch):
    manager = make_manager(tmp_path, renderer)
    original = manager._write_queue.put

    def full(*args, **kwargs):
        raise queue.Full

    try:
        monkeypatch.setattr(manager._write_queue, "put", full)
        _, path = save(manager)
        assert all(p.exists() for p in (path, *sidecar_paths(path)))
        assert manager.get_stats_dict()["ssd_inline_write_fallbacks"] == 1
    finally:
        monkeypatch.setattr(manager._write_queue, "put", original)
        manager.close()


def test_replaced_pending_write_cannot_clear_new_incarnation(tmp_path, renderer):
    entered, release = threading.Event(), threading.Event()
    original = renderer.render
    calls = 0

    def blocked(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(5)
        return original(*args)

    renderer.render = blocked
    manager = make_manager(tmp_path, renderer)
    try:
        block_hash, path = save(manager)
        assert entered.wait(5)
        assert manager.delete_block(block_hash)
        save(manager)  # Same hash, new indexed metadata and queued payload.
        newest = manager._index.get(block_hash)
        release.set()
        drain(manager)
        assert manager._index.get(block_hash) is newest
        assert newest.inspection_complete
        assert manager.get_stats_dict()["inspection_writes"] == 1
        assert all(p.exists() for p in (path, *sidecar_paths(path)))
        assert manager.load_block(block_hash) is not None
    finally:
        release.set()
        manager.close()


def test_backfill_enforces_size_limit(tmp_path, renderer):
    manager = make_manager(tmp_path)
    block_hash, path = save(manager)
    manager.close()
    manager = make_manager(tmp_path, renderer)
    try:
        manager._max_size = path.stat().st_size + 1
        manager.backfill_inspection(block_hash, BlockInspection((0, 1), 0, None))
        drain(manager)
        assert manager._index.total_size <= manager.max_size
        assert not any(p.exists() for p in (path, *sidecar_paths(path)))
    finally:
        manager.close()
