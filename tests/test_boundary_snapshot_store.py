# SPDX-License-Identifier: Apache-2.0
"""Tests for BoundarySnapshotSSDStore and _BoundarySnapshotProvider."""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

# MLX may not be available in CI — tests skip gracefully.
try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")

from omlx.cache.boundary_snapshot_store import BoundarySnapshotSSDStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_extracted(num_layers: int = 4) -> List[Dict[str, Any]]:
    """Create a list of extracted cache state dicts (mimics _extract_cache_states output).

    Layers 0 and 2 are KVCache placeholders (empty state).
    Layers 1 and 3 are ArraysCache with real tensors.
    """
    result = []
    for i in range(num_layers):
        if i % 2 == 0:
            # KVCache placeholder (skipped sliceable layer)
            result.append({
                "state": (),
                "meta_state": (),
                "class_name": "KVCache",
                "cache_type": "KVCache",
            })
        else:
            # ArraysCache with small tensors (conv_state + recurrent_state)
            conv_state = mx.ones((1, 3, 16), dtype=mx.float16)
            recurrent_state = mx.ones((1, 4, 8, 12), dtype=mx.bfloat16)
            result.append({
                "state": (conv_state, recurrent_state),
                "meta_state": (),
                "class_name": "ArraysCache",
                "cache_type": "ArraysCache",
            })
    return result


def _mock_extract_cache_states(snapshot_cache):
    """Mock for Scheduler._extract_cache_states."""
    return _make_extracted(), None


# ---------------------------------------------------------------------------
# BoundarySnapshotSSDStore tests
# ---------------------------------------------------------------------------


class TestBoundarySnapshotSSDStore:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.base_dir = tmp_path / "ssd_cache"
        self.base_dir.mkdir()
        self.store = BoundarySnapshotSSDStore(base_dir=self.base_dir)
        yield
        self.store.shutdown()

    def test_save_and_load_roundtrip(self):
        """Save a snapshot and load it back — tensors should match."""
        ok = self.store.save(
            "req-1", 1024, [MagicMock()], _mock_extract_cache_states
        )
        assert ok

        loaded = self.store.load("req-1", 1024)
        assert loaded is not None
        assert len(loaded) == 4

        # KVCache placeholder layers
        assert loaded[0]["state"] == ()
        assert loaded[0]["class_name"] == "KVCache"

        # ArraysCache layers — tensors should have correct shapes
        assert loaded[1]["class_name"] == "ArraysCache"
        state = loaded[1]["state"]
        assert len(state) == 2
        assert state[0].shape == (1, 3, 16)
        assert state[1].shape == (1, 4, 8, 12)

    def test_has_returns_true_after_save(self):
        self.store.save("req-1", 2048, [MagicMock()], _mock_extract_cache_states)
        assert self.store.has("req-1", 2048)
        assert not self.store.has("req-1", 4096)
        assert not self.store.has("req-2", 2048)

    def test_load_nonexistent_returns_none(self):
        assert self.store.load("req-1", 999) is None

    def test_cleanup_request_removes_files(self):
        self.store.save("req-1", 1024, [MagicMock()], _mock_extract_cache_states)
        self.store.save("req-1", 2048, [MagicMock()], _mock_extract_cache_states)
        self.store.save("req-2", 1024, [MagicMock()], _mock_extract_cache_states)

        self.store.cleanup_request("req-1")

        assert not self.store.has("req-1", 1024)
        assert not self.store.has("req-1", 2048)
        # req-2 unaffected
        assert self.store.has("req-2", 1024)

    def test_cleanup_all(self):
        self.store.save("req-1", 1024, [MagicMock()], _mock_extract_cache_states)
        self.store.save("req-2", 2048, [MagicMock()], _mock_extract_cache_states)

        self.store.cleanup_all()

        assert not self.store.has("req-1", 1024)
        assert not self.store.has("req-2", 2048)
        # Directory still exists (recreated).
        assert (self.base_dir / "_boundary_snapshots").exists()

    def test_load_from_disk_after_pending_writes_cleared(self):
        """After background writer completes, load should read from disk."""
        import time

        self.store.save("req-1", 1024, [MagicMock()], _mock_extract_cache_states)

        # Wait for background writer to complete.
        time.sleep(0.5)

        # Force clear pending writes to simulate post-write state.
        with self.store._pending_lock:
            self.store._pending_writes.clear()

        # Should load from disk.
        loaded = self.store.load("req-1", 1024)
        assert loaded is not None
        assert len(loaded) == 4
        assert loaded[1]["class_name"] == "ArraysCache"

    def test_multiple_snapshots_per_request(self):
        """Multiple token boundaries for the same request."""
        for tc in [1024, 2048, 3072, 4096]:
            ok = self.store.save(
                "req-1", tc, [MagicMock()], _mock_extract_cache_states
            )
            assert ok

        for tc in [1024, 2048, 3072, 4096]:
            loaded = self.store.load("req-1", tc)
            assert loaded is not None

    def test_save_returns_false_without_mlx(self):
        """Graceful failure when extract function returns empty."""
        def failing_extract(cache):
            return [], None

        ok = self.store.save("req-1", 1024, [MagicMock()], failing_extract)
        assert not ok

    def test_bfloat16_roundtrip(self):
        """Ensure bfloat16 tensors survive serialization."""
        def bf16_extract(cache):
            return [{
                "state": (
                    mx.ones((2, 3), dtype=mx.bfloat16),
                    mx.zeros((2, 3), dtype=mx.bfloat16),
                ),
                "meta_state": (1, 2, 3),
                "class_name": "ArraysCache",
                "cache_type": "ArraysCache",
            }], None

        self.store.save("req-bf", 1024, [MagicMock()], bf16_extract)
        loaded = self.store.load("req-bf", 1024)
        assert loaded is not None
        assert loaded[0]["state"][0].dtype == mx.bfloat16
        assert loaded[0]["meta_state"] == (1, 2, 3)

    def test_startup_cleans_orphaned_files(self):
        """Constructor should remove orphaned files from previous crashes."""
        # Create some orphaned files.
        orphan_dir = self.base_dir / "_boundary_snapshots" / "orphan-req"
        orphan_dir.mkdir(parents=True)
        (orphan_dir / "1024.safetensors").write_text("garbage")

        # Re-create store — should clean up.
        self.store.shutdown()
        store2 = BoundarySnapshotSSDStore(base_dir=self.base_dir)
        assert not orphan_dir.exists()
        store2.shutdown()

    def test_cleanup_request_skips_queued_writes(self):
        """Writer thread should skip items for a cleaned-up request."""
        import time

        self.store.save("req-1", 1024, [MagicMock()], _mock_extract_cache_states)
        self.store.save("req-1", 2048, [MagicMock()], _mock_extract_cache_states)

        # Cleanup before writer thread processes items.
        self.store.cleanup_request("req-1")

        # Wait for writer to process remaining queue items.
        time.sleep(1.0)

        # No files should have been written for req-1.
        req_dir = self.base_dir / "_boundary_snapshots" / "req-1"
        assert not req_dir.exists()

    def test_cleanup_all_drains_queue(self):
        """cleanup_all() should leave the snapshot directory empty no
        matter where the writer thread was in its processing cycle.

        Previously this test slept 1.0 s as a guess at the writer's
        finish time and was flaky ~20% of the time: the writer could
        ``os.rename`` a temp file into its final path *after* cleanup_all
        had rmtree'd the directory, leaving an orphaned file.
        cleanup_all now holds the writer-busy lock until any in-flight
        item is done, so no sleep is required.
        """
        self.store.save("req-1", 1024, [MagicMock()], _mock_extract_cache_states)
        self.store.save("req-2", 2048, [MagicMock()], _mock_extract_cache_states)

        # cleanup_all() must synchronize with the writer.
        self.store.cleanup_all()

        # Snapshot directory should be clean (recreated but empty).
        snapshot_dir = self.base_dir / "_boundary_snapshots"
        assert snapshot_dir.exists()
        children = list(snapshot_dir.iterdir())
        assert len(children) == 0

    def test_cleanup_all_blocks_until_writer_finishes_pinned_item(self):
        """Deterministic regression for the writer-vs-cleanup race.

        Pins the writer mid-item with a slow ``_write_safetensors_no_mx``
        replacement, fires ``cleanup_all()`` from the test thread, and
        asserts that:
          1. cleanup_all does not return before the writer finishes its
             pinned item (would-be-orphaned rename), AND
          2. the snapshot directory ends up empty.

        Without the ``_writer_busy`` lock this would fail deterministically
        rather than flakily — the writer's ``os.rename`` lands after the
        rmtree and an orphan survives.
        """
        import threading
        import time
        from unittest.mock import patch

        writer_in_item = threading.Event()
        release_writer = threading.Event()
        original_write = None

        def slow_write(*args, **kwargs):
            writer_in_item.set()
            # Hold the writer here so cleanup_all is forced to wait on
            # _writer_busy. 1 s is plenty for the test thread to call
            # cleanup_all and start blocking.
            release_writer.wait(timeout=5.0)
            return original_write(*args, **kwargs)

        from omlx.cache import boundary_snapshot_store as mod

        original_write = mod._write_safetensors_no_mx

        with patch.object(mod, "_write_safetensors_no_mx", side_effect=slow_write):
            self.store.save("req-pinned", 1024, [MagicMock()], _mock_extract_cache_states)

            # Wait until the writer has picked up the item and is inside
            # the slow_write hook.
            assert writer_in_item.wait(timeout=5.0), "writer never started"

            # Kick off cleanup_all from a background thread so we can
            # observe that it does not complete while the writer is pinned.
            cleanup_done = threading.Event()

            def _do_cleanup():
                self.store.cleanup_all()
                cleanup_done.set()

            t = threading.Thread(target=_do_cleanup, name="cleanup-all-test")
            t.start()

            # cleanup_all must NOT return while the writer holds _writer_busy.
            assert not cleanup_done.wait(timeout=0.5), (
                "cleanup_all returned while writer was mid-item — "
                "_writer_busy lock is not being honored"
            )

            # Release the writer; cleanup_all should then complete.
            release_writer.set()
            assert cleanup_done.wait(timeout=10.0), "cleanup_all hung"
            t.join(timeout=5.0)

        # Give the writer one more tick to fully exit _process_write_item
        # before asserting on the directory.
        time.sleep(0.1)
        snapshot_dir = self.base_dir / "_boundary_snapshots"
        assert snapshot_dir.exists()
        assert list(snapshot_dir.iterdir()) == []

    def test_cleanup_request_blocks_until_writer_finishes_pinned_item(self):
        """Symmetric regression to cleanup_all: cleanup_request must also
        wait on the writer's in-flight item before rmtree, otherwise the
        writer's late ``os.rename`` lands under the just-cleaned dir.
        """
        import threading
        import time
        from unittest.mock import patch

        writer_in_item = threading.Event()
        release_writer = threading.Event()
        original_write = None

        def slow_write(*args, **kwargs):
            writer_in_item.set()
            release_writer.wait(timeout=5.0)
            return original_write(*args, **kwargs)

        from omlx.cache import boundary_snapshot_store as mod

        original_write = mod._write_safetensors_no_mx

        with patch.object(mod, "_write_safetensors_no_mx", side_effect=slow_write):
            self.store.save("req-cleanup", 2048, [MagicMock()], _mock_extract_cache_states)

            assert writer_in_item.wait(timeout=5.0), "writer never started"

            cleanup_done = threading.Event()

            def _do_cleanup():
                self.store.cleanup_request("req-cleanup")
                cleanup_done.set()

            t = threading.Thread(target=_do_cleanup, name="cleanup-req-test")
            t.start()

            assert not cleanup_done.wait(timeout=0.5), (
                "cleanup_request returned while writer was mid-item — "
                "_writer_busy lock is not being honored"
            )

            release_writer.set()
            assert cleanup_done.wait(timeout=10.0), "cleanup_request hung"
            t.join(timeout=5.0)

        # After cleanup_request the per-request directory must be gone.
        time.sleep(0.1)
        req_dir = self.base_dir / "_boundary_snapshots" / "req-cleanup"
        assert not req_dir.exists()

    def test_cleanup_request_keeps_counter_on_timeout(self):
        """When ``cleanup_request`` cannot acquire ``_writer_busy`` within
        ``_CLEANUP_REQUEST_TIMEOUT_S``, it must NOT pop
        ``_cancelled_requests[request_id]``: the counter is the rescue
        path the docstring promises for the late-rename window. The
        previous code popped unconditionally and silently defeated the
        rescue. Regression for the real bug found in review.
        """
        import threading
        import time
        from unittest.mock import patch

        # Pin the writer mid-item so the cleanup_request acquire times out.
        writer_in_item = threading.Event()
        release_writer = threading.Event()
        original_write = None

        def slow_write(*args, **kwargs):
            writer_in_item.set()
            release_writer.wait(timeout=10.0)
            return original_write(*args, **kwargs)

        from omlx.cache import boundary_snapshot_store as mod

        original_write = mod._write_safetensors_no_mx

        # Tighten the timeout for the test so the test runs fast.
        with patch.object(
            type(self.store), "_CLEANUP_REQUEST_TIMEOUT_S", 0.1
        ), patch.object(mod, "_write_safetensors_no_mx", side_effect=slow_write):
            self.store.save(
                "req-timeout-rescue",
                2048,
                [MagicMock()],
                _mock_extract_cache_states,
            )
            assert writer_in_item.wait(timeout=5.0), "writer never started"

            # cleanup_request returns once the 0.1s timeout fires — writer
            # is still pinned. The counter MUST remain so _is_cancelled
            # can still catch the late rename.
            self.store.cleanup_request("req-timeout-rescue")

            with self.store._cancelled_lock:
                assert (
                    "req-timeout-rescue" in self.store._cancelled_requests
                ), (
                    "counter dropped on timeout — late-rename rescue "
                    "would be defeated"
                )

            # Let the writer finish; rescue then drops the counter via
            # _is_cancelled → _dec_cancelled.
            release_writer.set()
            time.sleep(0.5)

    def test_cleanup_request_timeout_drains_counter_on_writer_early_return(
        self,
    ):
        """Regression: when ``cleanup_request`` times out while
        ``_cancelled_requests[rid]`` is non-zero, items that the writer
        later dequeues but whose pending entry was already cleared by
        cleanup must still decrement the counter on the early-return
        path. Without that decrement the rid stays in
        ``_cancelled_requests`` for the process lifetime and every
        future write under that rid is silently discarded by the
        ``_is_cancelled`` gates.
        """
        import threading
        import time
        from unittest.mock import patch

        writer_in_item = threading.Event()
        release_writer = threading.Event()
        original_write = None

        def slow_write(*args, **kwargs):
            writer_in_item.set()
            release_writer.wait(timeout=10.0)
            return original_write(*args, **kwargs)

        from omlx.cache import boundary_snapshot_store as mod

        original_write = mod._write_safetensors_no_mx

        with patch.object(
            type(self.store), "_CLEANUP_REQUEST_TIMEOUT_S", 0.1
        ), patch.object(
            mod, "_write_safetensors_no_mx", side_effect=slow_write
        ):
            # Two items for the same rid: A pins the writer; B sits in
            # the queue behind A.
            self.store.save(
                "req-drain", 2048, [MagicMock()],
                _mock_extract_cache_states,
            )
            self.store.save(
                "req-drain", 4096, [MagicMock()],
                _mock_extract_cache_states,
            )
            assert writer_in_item.wait(timeout=5.0), (
                "writer never started item A"
            )

            # cleanup_request snapshots both pending items, sets
            # counter=2, then times out (writer still pinned on A).
            self.store.cleanup_request("req-drain")
            with self.store._cancelled_lock:
                assert (
                    self.store._cancelled_requests.get("req-drain") == 2
                ), (
                    "cleanup_request did not record both pending items "
                    "before timing out"
                )

            # Releasing A lets the writer finish: post-rename
            # _is_cancelled fires → 2→1. The queue then advances to B
            # whose pending entry was already cleared by cleanup;
            # writer's early-return path MUST decrement 1→0 and pop.
            release_writer.set()

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                with self.store._cancelled_lock:
                    if "req-drain" not in self.store._cancelled_requests:
                        break
                time.sleep(0.02)
            else:
                with self.store._cancelled_lock:
                    state = dict(self.store._cancelled_requests)
                raise AssertionError(
                    "_cancelled_requests still pins 'req-drain' after "
                    f"both items processed: {state}"
                )

    def test_cleanup_request_no_pending_does_not_pin_counter_on_timeout(self):
        """Regression: ``cleanup_request("X")`` for an rid with NO
        pending items must NOT bump ``_cancelled_requests[X] = 0``.

        Previously the unconditional bump would write ``X: 0``, then on
        the acquired path pop it. On the timeout fallback the pop never
        ran and the ``X: 0`` entry lingered for the process lifetime —
        every subsequent ``save()`` under that rid (or any later reuse
        of the same string) was silently discarded by the writer's
        ``_is_cancelled`` gates, which check key membership not
        value > 0.
        """
        import threading
        import time
        from unittest.mock import patch

        # Pin the writer with an unrelated save so cleanup_request's
        # _writer_busy.acquire times out without any item for our rid.
        writer_in_item = threading.Event()
        release_writer = threading.Event()
        original_write = None

        def slow_write(*args, **kwargs):
            writer_in_item.set()
            release_writer.wait(timeout=10.0)
            return original_write(*args, **kwargs)

        from omlx.cache import boundary_snapshot_store as mod

        original_write = mod._write_safetensors_no_mx

        with patch.object(
            type(self.store), "_CLEANUP_REQUEST_TIMEOUT_S", 0.1
        ), patch.object(
            mod, "_write_safetensors_no_mx", side_effect=slow_write
        ):
            self.store.save(
                "req-blocker", 2048, [MagicMock()],
                _mock_extract_cache_states,
            )
            assert writer_in_item.wait(timeout=5.0), (
                "writer never started blocker item"
            )

            # cleanup_request for an rid that was NEVER saved. count==0.
            # _writer_busy is held by the blocker → acquire times out.
            self.store.cleanup_request("never-saved-rid")

            with self.store._cancelled_lock:
                assert (
                    "never-saved-rid" not in self.store._cancelled_requests
                ), (
                    "cleanup_request bumped _cancelled_requests for an "
                    "rid with no pending items — the stale 0-counter "
                    "would silently kill every future save under that rid"
                )

            # Verify the bug's downstream consequence directly:
            # a save() under the same rid must succeed, not be discarded
            # by the writer's _is_cancelled gates.
            release_writer.set()
            time.sleep(0.2)  # let blocker drain
            ok = self.store.save(
                "never-saved-rid", 4096, [MagicMock()],
                _mock_extract_cache_states,
            )
            assert ok, "save() failed"
            # Wait for the writer to finish.
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if self.store.has("never-saved-rid", 4096):
                    break
                time.sleep(0.02)
            file_path = (
                self.base_dir / "_boundary_snapshots" / "never-saved-rid"
                / "4096.safetensors"
            )
            # Either the file is on disk OR still buffered in pending —
            # but it must not have been silently discarded.
            with self.store._pending_lock:
                still_pending = (
                    "never-saved-rid", 4096
                ) in self.store._pending_writes
            assert file_path.exists() or still_pending, (
                "save() under rid was silently discarded — stale "
                "_cancelled_requests entry defeated the new write"
            )

    def test_save_queue_full_rolls_back_pending_and_registry(self):
        """When the writer queue is saturated, ``save()`` must roll back
        its pending_writes / file_registry entries and return False.
        Otherwise a later ``cleanup_request`` for the same rid would
        count this orphan entry into ``_cancelled_requests`` while no
        queue item ever exists to decrement it — the rid would stay
        pinned in the cancelled set and every subsequent save under
        that rid would be silently discarded by the ``_is_cancelled``
        gates.
        """
        from unittest.mock import patch
        import queue as _queue

        def _full(*args, **kwargs):
            raise _queue.Full

        with patch.object(
            self.store._write_queue, "put_nowait", side_effect=_full
        ):
            ok = self.store.save(
                "req-qfull", 2048, [MagicMock()],
                _mock_extract_cache_states,
            )

        assert ok is False, (
            "save() must return False when the queue is full so the "
            "caller knows the write was dropped"
        )
        with self.store._pending_lock:
            assert ("req-qfull", 2048) not in self.store._pending_writes
        with self.store._registry_lock:
            assert "req-qfull" not in self.store._file_registry

        # cleanup_request on the same rid must NOT pin the counter.
        self.store.cleanup_request("req-qfull")
        with self.store._cancelled_lock:
            assert "req-qfull" not in self.store._cancelled_requests

    def test_shutdown_cleanup_true_runs_cleanup_before_setting_flag(self):
        """``shutdown(cleanup=True)`` must run ``cleanup_all()`` BEFORE
        flipping ``_shutdown`` so the writer still reacquires
        ``_writer_busy`` per item during the cleanup. Otherwise the
        cleanup degrades to an in-memory-only clear (see the
        post-shutdown branch in ``cleanup_all``).
        """
        # Save a block first so cleanup_all has something to drain.
        self.store.save("req-shutdown", 1024, [MagicMock()], _mock_extract_cache_states)

        # Use a small custom store so we can shut down without affecting
        # other tests in this class.
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            store2 = BoundarySnapshotSSDStore(Path(td))
            try:
                store2.save(
                    "req-x", 256, [MagicMock()], _mock_extract_cache_states
                )
                snapshot_dir = store2._snapshot_dir
                assert snapshot_dir.exists()
                store2.shutdown(cleanup=True)
                # After cleanup_all+shutdown the per-request dir is empty
                # of leftover request subdirs (the cleanup itself rmtrees
                # then mkdirs the parent).
                assert snapshot_dir.exists()
                leftover = list(snapshot_dir.iterdir())
                assert leftover == [], (
                    f"shutdown(cleanup=True) left files behind: {leftover}"
                )
            finally:
                if store2._writer_thread.is_alive():
                    store2.shutdown()

    def test_cancelled_requests_dict_is_thread_safe(self):
        """Concurrent cleanup_request + writer should not race on
        _cancelled_requests. Without locking, the counter underflows or
        cancellation can be silently lost.
        """
        import threading

        # Fire many concurrent cleanup_request calls against requests
        # that don't have any pending items — exercises the lock acquire
        # / set / clear paths without needing real file I/O.
        errors: list[Exception] = []

        def cancel_loop(rid_prefix: str):
            try:
                for i in range(200):
                    self.store.cleanup_request(f"{rid_prefix}-{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=cancel_loop, args=(f"t{tid}",))
            for tid in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)
        assert not errors, errors
        # The dict must not be in a corrupt state — clear() and len()
        # both succeed.
        with self.store._cancelled_lock:
            assert len(self.store._cancelled_requests) >= 0


    def test_concurrent_save_cleanup_request_cleanup_all_no_orphans(self):
        """Stress: concurrent save() + cleanup_request() + cleanup_all().

        Regression target: the late-rename window where the writer pulled
        an item from the queue but had not yet entered the busy-lock
        critical section while cleanup ran would leave an orphaned file
        under the recreated snapshot directory. The _process_write_item
        pending-writes membership check closes that window.

        Test asserts: after all activity quiesces, every file on disk
        also has a corresponding entry in _file_registry — i.e. no
        orphans.
        """
        import threading
        import time as _time

        stop = threading.Event()
        errors: list[Exception] = []

        def saver(rid_prefix: str):
            try:
                tc = 0
                while not stop.is_set():
                    tc += 1
                    self.store.save(
                        f"{rid_prefix}-{tc % 7}",
                        tc * 1024,
                        [MagicMock()],
                        _mock_extract_cache_states,
                    )
            except Exception as e:
                errors.append(e)

        def cleaner(rid_prefix: str):
            try:
                tc = 0
                while not stop.is_set():
                    tc += 1
                    self.store.cleanup_request(f"{rid_prefix}-{tc % 7}")
                    _time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def all_cleaner():
            try:
                while not stop.is_set():
                    _time.sleep(0.05)
                    self.store.cleanup_all()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=saver, args=("a",)),
            threading.Thread(target=saver, args=("b",)),
            threading.Thread(target=cleaner, args=("a",)),
            threading.Thread(target=cleaner, args=("b",)),
            threading.Thread(target=all_cleaner),
        ]
        for t in threads:
            t.start()
        _time.sleep(1.5)
        stop.set()
        for t in threads:
            t.join(timeout=10.0)
        assert not errors, errors

        # Let writer drain.
        _time.sleep(0.5)

        # Orphan check: every .safetensors on disk must have a matching
        # registry entry. The reverse direction is fine to drift (the
        # registry may have entries the writer hasn't materialised yet).
        snap_root = self.base_dir / "_boundary_snapshots"
        on_disk = list(snap_root.rglob("*.safetensors"))
        registered_paths: set[Path] = set()
        with self.store._registry_lock:
            for tc_to_path in self.store._file_registry.values():
                registered_paths.update(tc_to_path.values())

        orphans = [p for p in on_disk if p not in registered_paths]
        # Allow a small tolerance for in-flight temp files only — those
        # have "_tmp" in the stem and are not real orphans.
        real_orphans = [p for p in orphans if "_tmp" not in p.stem]
        assert not real_orphans, (
            f"Found {len(real_orphans)} orphaned files: "
            f"{real_orphans[:5]}"
        )


# ---------------------------------------------------------------------------
# _BoundarySnapshotProvider tests
# ---------------------------------------------------------------------------


class TestBoundarySnapshotProvider:
    def test_provider_loads_from_store(self, tmp_path):
        """Provider should load snapshots from SSD store on __getitem__."""
        from omlx.scheduler import _BoundarySnapshotProvider

        base_dir = tmp_path / "ssd"
        base_dir.mkdir()
        store = BoundarySnapshotSSDStore(base_dir=base_dir)

        # Save a snapshot.
        store.save("req-1", 1024, [MagicMock()], _mock_extract_cache_states)

        # Create provider with None markers (SSD offloaded).
        snapshots = {1024: None, 2048: None}
        provider = _BoundarySnapshotProvider(
            store=store,
            request_id="req-1",
            valid_tcs=[1024],
            in_memory_snapshots=snapshots,
            extract_fn=_mock_extract_cache_states,
        )

        assert bool(provider)
        assert 1024 in provider
        assert 2048 not in provider

        loaded = provider[1024]
        assert loaded is not None
        assert len(loaded) == 4

        store.shutdown()

    def test_provider_falls_back_to_in_memory(self):
        """Provider should extract from in-memory snapshots when value is not None."""
        from omlx.scheduler import _BoundarySnapshotProvider

        mock_cache = MagicMock()
        snapshots = {1024: mock_cache}  # Not None = in-memory

        provider = _BoundarySnapshotProvider(
            store=None,
            request_id="req-1",
            valid_tcs=[1024],
            in_memory_snapshots=snapshots,
            extract_fn=_mock_extract_cache_states,
        )

        loaded = provider[1024]
        assert loaded is not None
        assert len(loaded) == 4

    def test_provider_empty(self):
        """Empty provider should be falsy."""
        from omlx.scheduler import _BoundarySnapshotProvider

        provider = _BoundarySnapshotProvider(
            store=None,
            request_id="req-1",
            valid_tcs=[],
            in_memory_snapshots={},
            extract_fn=_mock_extract_cache_states,
        )

        assert not bool(provider)
        assert 1024 not in provider
