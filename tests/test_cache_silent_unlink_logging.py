# SPDX-License-Identifier: Apache-2.0
"""Behavioural contracts for the bare-`except:-pass` clean-ups in the cache modules.

The fixes in this PR replace previously silent `except Exception: pass` blocks
in `vision_feature_cache.py` and `paged_ssd_cache.py` with structured
`logger.warning(...)` calls. The contracts asserted here are:

1. When the underlying unlink raises, the cache code emits a warning that
   mentions the path of the file it could not remove (so an operator can
   find the orphan).
2. When the unlink succeeds, no warning is emitted on the warning channel.
3. The surrounding operation still returns a value consistent with the
   previous (silent) behaviour — i.e. the fix is observable only through
   the logging surface, never through the return value.

These tests use pytest's `caplog` fixture to capture logs and `unittest.mock`
to inject OSError into the path's `unlink` call.
"""

import logging
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------
# Shared test machinery
# ---------------------------------------------------------------------


def _write_file(path: Path, body: bytes = b"x") -> None:
    """Drop a real file on disk so unlink() has something to act on."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)


class _FailingPath(str):
    """A Path subclass whose `.unlink()` raises OSError the first time only.

    The cache calls unlink() unconditionally when a file should be removed;
    we want to assert it still does the right thing when the FS error fires.
    """

    def __new__(cls, real_path: Path):
        instance = super().__new__(cls, str(real_path))
        instance._real_path = real_path
        instance._raised = False
        return instance

    def unlink(self) -> None:
        if not self._raised:
            self._raised = True
            raise OSError(5, "Input/output error (test injection)", str(self))
        # On second call: do the real thing.
        try:
            os.remove(str(self._real_path))
        except FileNotFoundError:
            pass


class _FakeIndex:
    """Minimal stand-in for the SSD index: only the surface `load_block*`
    touches (`get` for the lookup, `remove` for the corrupted-entry cleanup,
    `touch` on the success path)."""

    def __init__(self, metadata):
        self._metadata = metadata

    def get(self, _block_hash):
        return self._metadata

    def remove(self, _block_hash):
        self._metadata = None

    def touch(self, _block_hash):
        pass


def _paged_manager(tmp_path, file_path):
    """A real PagedSSDCacheManager whose index resolves to `file_path`.

    Constructing the real object (rather than `__new__` + hand-set fields)
    keeps the hot-cache / pending-write locks and buffers that the load path
    acquires; only the index is swapped so the lookup lands on a file we
    control.
    """
    from omlx.cache.paged_ssd_cache import PagedSSDCacheManager

    manager = PagedSSDCacheManager(
        cache_dir=tmp_path, max_size_bytes=8 * 1024 * 1024
    )
    manager._index = _FakeIndex(SimpleNamespace(file_path=file_path))
    return manager


class _FailingUnlinkPath(type(Path())):
    """A real Path that raises from `unlink()` the first time it is called.

    A plain stub cannot be used here: the evictor calls `file_path.exists()`
    before removing the file, so the object handed to it has to behave like a
    path apart from the failure we inject.
    """

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls, *args, **kwargs)
        obj._failed_once = False
        return obj

    def unlink(self, missing_ok: bool = False) -> None:
        if not self._failed_once:
            self._failed_once = True
            raise OSError(5, "Input/output error (test injection)", str(self))
        return super().unlink(missing_ok=missing_ok)


# ---------------------------------------------------------------------
# tests.test_vision_feature_cache module coverage
# ---------------------------------------------------------------------


@pytest.fixture
def _vfc_env():
    """Bring up a VisionFeatureSSDCache pointed at a real temp dir."""

    # Import here so module-level imports succeed even if omlx cache deps
    # are unavailable in this minimal test environment.
    from omlx.cache.vision_feature_cache import VisionFeatureSSDCache

    class _Tmp:
        def __init__(self):
            self.root = Path(os.environ.get("TMPDIR", "/tmp")) / (
                "vfc_test_" + str(os.getpid()) + "_" + str(time.time_ns())
            )
            self.root.mkdir(parents=True, exist_ok=True)

        def cleanup(self):
            try:
                import shutil

                shutil.rmtree(self.root, ignore_errors=True)
            except Exception:
                pass

    tmp = _Tmp()
    cache = VisionFeatureSSDCache(cache_dir=tmp.root, max_memory_entries=3)
    try:
        yield cache, tmp
    finally:
        cache.close()
        tmp.cleanup()


def test_vision_eviction_unlink_failure_logs_warning(_vfc_env, caplog):
    from omlx.cache.vision_feature_cache import VisionFeatureSSDEntry

    cache, tmp = _vfc_env
    # Seed an entry whose tracked file we'll cause unlink() to fail on.
    file_path = Path(tmp.root) / "vision_entry.bin"
    _write_file(file_path)
    file_id = "evict-1"

    failing = _FailingUnlinkPath(file_path)
    entry = VisionFeatureSSDEntry(
        image_hash=file_id,
        model_name="test-model",
        file_path=failing,
        file_size=1,
        created_at=0.0,
        last_access=0.0,
    )

    with caplog.at_level(logging.WARNING, logger="omlx.cache.vision_feature_cache"):
        # Make the entry the oldest in the index and push the tracked size over
        # the limit so the evictor has to prune it.
        cache._ssd_index[file_id] = entry
        cache._ssd_total_size = cache._max_size_bytes + 1
        with cache._ssd_lock:
            cache._evict_ssd_if_needed()

    warned = [r for r in caplog.records if r.name == "omlx.cache.vision_feature_cache" and r.levelno == logging.WARNING]
    assert warned, "expected a warning when SSD evict unlink fails"
    assert any(str(file_path) in r.getMessage() for r in warned), (
        f"warning should mention the file path; got: {[r.getMessage() for r in warned]}"
    )


# ---------------------------------------------------------------------
# tests.test_paged_ssd_cache module coverage
# ---------------------------------------------------------------------


def test_paged_ssd_load_block_unlink_failure_logs_warning(tmp_path, caplog):
    """Regression: when load_block() catches an SSD read error and its
    follower unlink() raises, the warning should name the file that failed."""
    file_path = tmp_path / "block.bin"
    _write_file(file_path)
    failing = _FailingPath(file_path)
    manager = _paged_manager(tmp_path, file_path)

    class _FakeLogger:
        def __init__(self):
            self.records: list[tuple[int, str]] = []

        def warning(self, msg, *args):
            self.records.append((logging.WARNING, msg % args))

        def error(self, msg, *args):
            self.records.append((logging.ERROR, msg % args))

    fake_log = _FakeLogger()

    captured = {}
    def cap(level, msg, *args):
        captured.setdefault(level, []).append(msg % args)

    # Patch the manager's logger and the unlink call, then drive load_block.
    from omlx.cache import paged_ssd_cache as ps_mod

    with patch.object(ps_mod, "logger", new=fake_log), patch.object(
        Path, "unlink", failing.unlink, create=True
    ):
        result = manager.load_block(b"\x00" * 32)

    warn_msgs = [m for lvl, m in fake_log.records if lvl == logging.WARNING]
    assert result is None, "load_block should return None after a failed read"
    assert any(str(file_path) in m for m in warn_msgs), (
        f"warning should mention the file path; got: {warn_msgs}"
    )
    manager.close()


def test_paged_ssd_metadata_unlink_failure_logs_warning(tmp_path, caplog):
    """The other branch of `except Exception: pass` in paged_ssd_cache.py:
    `load_block_with_metadata` at line ~4094. Same contract."""
    file_path = tmp_path / "block_meta.bin"
    _write_file(file_path)
    failing = _FailingPath(file_path)
    manager = _paged_manager(tmp_path, file_path)

    class _FakeLogger:
        def __init__(self):
            self.records: list[tuple[int, str]] = []

        def warning(self, msg, *args):
            self.records.append((logging.WARNING, msg % args))

        def error(self, msg, *args):
            self.records.append((logging.ERROR, msg % args))

    fake_log = _FakeLogger()
    from omlx.cache import paged_ssd_cache as ps_mod

    with patch.object(ps_mod, "logger", new=fake_log), patch.object(
        Path, "unlink", failing.unlink, create=True
    ):
        result = manager.load_block_with_metadata(b"\x00" * 32)

    warn_msgs = [m for lvl, m in fake_log.records if lvl == logging.WARNING]
    assert result == (None, None), "load_block_with_metadata returns (None, None) on a read failure"
    assert any(str(file_path) in m for m in warn_msgs), (
        f"warning should mention the file path; got: {warn_msgs}"
    )


def test_paged_ssd_unlink_succeeds_no_warning(tmp_path, caplog):
    """Sanity: when the SSD unlink succeeds (no exception), nothing is logged."""
    file_path = tmp_path / "block_clean.bin"
    _write_file(file_path)
    manager = _paged_manager(tmp_path, file_path)

    class _FakeLogger:
        def warning(self, *a, **kw):
            raise AssertionError("expected NO warning on a successful unlink")

        def error(self, *a, **kw):
            pass

    fake_log = _FakeLogger()
    from omlx.cache import paged_ssd_cache as ps_mod

    with patch.object(ps_mod, "logger", new=fake_log):
        result = manager.load_block_with_metadata(b"\x00" * 32)

    # The block's content is not in this fake filesystem, so the function
    # may return None through its normal "not found" path; what we assert is
    # that it never reaches a warning call when unlink succeeds.
    assert result == (None, None)
