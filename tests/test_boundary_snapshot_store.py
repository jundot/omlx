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
