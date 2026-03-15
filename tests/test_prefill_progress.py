# SPDX-License-Identifier: Apache-2.0
"""Tests for PrefillProgressTracker."""

import threading

import pytest

from omlx.prefill_progress import PrefillProgressTracker


class TestPrefillProgressTracker:
    def setup_method(self):
        self.tracker = PrefillProgressTracker()

    def test_update_and_get_progress(self):
        self.tracker.update("req-1", 2048, 8192, "llama-3b")
        result = self.tracker.get_model_progress("llama-3b")
        assert len(result) == 1
        assert result[0]["request_id"] == "req-1"
        assert result[0]["processed"] == 2048
        assert result[0]["total"] == 8192

    def test_auto_remove_on_complete(self):
        self.tracker.update("req-1", 2048, 8192, "llama-3b")
        assert len(self.tracker.get_model_progress("llama-3b")) == 1

        # Processed reaches total -> auto remove
        self.tracker.update("req-1", 8192, 8192, "llama-3b")
        assert len(self.tracker.get_model_progress("llama-3b")) == 0

    def test_auto_remove_on_exceed(self):
        self.tracker.update("req-1", 9000, 8192, "llama-3b")
        assert len(self.tracker.get_model_progress("llama-3b")) == 0

    def test_explicit_remove(self):
        self.tracker.update("req-1", 2048, 8192, "llama-3b")
        self.tracker.remove("req-1")
        assert len(self.tracker.get_model_progress("llama-3b")) == 0

    def test_remove_nonexistent(self):
        # Should not raise
        self.tracker.remove("nonexistent")

    def test_multiple_requests_same_model(self):
        self.tracker.update("req-1", 2048, 8192, "llama-3b")
        self.tracker.update("req-2", 1024, 4096, "llama-3b")
        result = self.tracker.get_model_progress("llama-3b")
        assert len(result) == 2
        ids = {r["request_id"] for r in result}
        assert ids == {"req-1", "req-2"}

    def test_multiple_models(self):
        self.tracker.update("req-1", 2048, 8192, "llama-3b")
        self.tracker.update("req-2", 1024, 4096, "qwen-7b")
        assert len(self.tracker.get_model_progress("llama-3b")) == 1
        assert len(self.tracker.get_model_progress("qwen-7b")) == 1
        assert len(self.tracker.get_model_progress("other")) == 0

    def test_update_overwrites_previous(self):
        self.tracker.update("req-1", 2048, 8192, "llama-3b")
        self.tracker.update("req-1", 4096, 8192, "llama-3b")
        result = self.tracker.get_model_progress("llama-3b")
        assert len(result) == 1
        assert result[0]["processed"] == 4096

    def test_clear(self):
        self.tracker.update("req-1", 2048, 8192, "llama-3b")
        self.tracker.update("req-2", 1024, 4096, "qwen-7b")
        self.tracker.clear()
        assert len(self.tracker.get_model_progress("llama-3b")) == 0
        assert len(self.tracker.get_model_progress("qwen-7b")) == 0

    def test_thread_safety(self):
        """Concurrent updates from multiple threads should not corrupt state."""
        errors = []

        def updater(model_id, start):
            try:
                for i in range(100):
                    rid = f"req-{model_id}-{start + i}"
                    self.tracker.update(rid, i * 100, 10000, model_id)
                    if i % 2 == 0:
                        self.tracker.remove(rid)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=updater, args=(f"model-{t}", t * 100))
            for t in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
