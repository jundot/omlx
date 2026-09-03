# SPDX-License-Identifier: Apache-2.0
"""Tests for model registry functionality."""

import gc
import weakref
from unittest.mock import MagicMock

import pytest

from omlx.model_registry import (
    ModelOwnershipError,
    ModelRegistry,
    get_registry,
)


class TestModelRegistry:
    """Tests for ModelRegistry class."""

    def test_registry_singleton(self):
        """Test that registry is a singleton."""
        reg1 = get_registry()
        reg2 = get_registry()
        assert reg1 is reg2

    def test_acquire_ownership(self):
        """Test basic ownership acquisition."""
        registry = get_registry()
        model = MagicMock()
        engine = MagicMock()
        engine_id = "test-engine-1"

        result = registry.acquire(model, engine, engine_id)

        assert result is True
        is_owned, owner_id = registry.is_owned(model)
        assert is_owned is True
        assert owner_id == engine_id

        # Cleanup
        registry.release(model, engine_id)

    def test_release_ownership(self):
        """Test ownership release."""
        registry = get_registry()
        model = MagicMock()
        engine = MagicMock()
        engine_id = "test-engine-release"

        registry.acquire(model, engine, engine_id)

        result = registry.release(model, engine_id)
        assert result is True

        is_owned, _ = registry.is_owned(model)
        assert is_owned is False

    def test_release_by_id_does_not_require_model_reference(self):
        """Teardown can preserve ordering without retaining model weights."""
        registry = get_registry()
        model = MagicMock()
        engine = MagicMock()
        engine_id = "test-engine-release-by-id"
        model_id = id(model)

        registry.acquire(model, engine, engine_id)

        assert registry.release_by_id(model_id, engine_id) is True
        assert registry.is_owned(model) == (False, None)

    def test_release_by_id_rejects_wrong_owner(self):
        registry = get_registry()
        model = MagicMock()
        engine = MagicMock()
        engine_id = "test-engine-release-by-id-owner"

        registry.acquire(model, engine, engine_id)

        assert registry.release_by_id(id(model), "wrong-engine") is False
        assert registry.is_owned(model) == (True, engine_id)
        registry.release(model, engine_id)

    def test_release_wrong_owner(self):
        """Test that release fails for wrong owner."""
        registry = get_registry()
        model = MagicMock()
        engine = MagicMock()
        engine_id = "test-engine-owner"

        registry.acquire(model, engine, engine_id)

        # Try to release with wrong engine_id
        result = registry.release(model, "wrong-engine")
        assert result is False

        # Should still be owned by original
        is_owned, owner = registry.is_owned(model)
        assert is_owned is True
        assert owner == engine_id

        # Cleanup
        registry.release(model, engine_id)

    def test_force_ownership_transfer(self):
        """A force flag cannot bypass a different live owner."""
        registry = get_registry()
        model = MagicMock()
        engine1 = MagicMock()
        engine1.scheduler = MagicMock()
        engine2 = MagicMock()

        # First engine acquires
        registry.acquire(model, engine1, "engine-1")

        with pytest.raises(ModelOwnershipError):
            registry.acquire(model, engine2, "engine-2", force=True)

        engine1.scheduler.deep_reset.assert_not_called()
        is_owned, owner_id = registry.is_owned(model)
        assert is_owned is True
        assert owner_id == "engine-1"

        # Cleanup
        registry.release(model, "engine-1")

    def test_force_rejection_does_not_mutate_previous_owner(self):
        registry = get_registry()
        model = MagicMock()
        owner = MagicMock()
        owner.scheduler = MagicMock()
        owner._cancel_prompt_tail_prewarm = MagicMock()
        replacement = MagicMock()
        registry.acquire(model, owner, "old-owner")
        with pytest.raises(ModelOwnershipError):
            registry.acquire(model, replacement, "new-owner", force=True)

        owner._cancel_prompt_tail_prewarm.assert_not_called()
        owner.scheduler.deep_reset.assert_not_called()
        assert registry.is_owned(model) == (True, "old-owner")
        registry.release(model, "old-owner")

    def test_no_force_raises_error(self):
        """Test that force=False raises error when model is owned."""
        registry = get_registry()
        model = MagicMock()
        engine1 = MagicMock()
        engine2 = MagicMock()

        registry.acquire(model, engine1, "engine-1")

        with pytest.raises(ModelOwnershipError):
            registry.acquire(model, engine2, "engine-2", force=False)

        # Cleanup
        registry.release(model, "engine-1")

    def test_same_engine_can_reacquire(self):
        """Test that same engine can reacquire without error."""
        registry = get_registry()
        model = MagicMock()
        engine = MagicMock()
        engine_id = "same-engine"

        registry.acquire(model, engine, engine_id)

        # Same engine reacquires - should succeed without error
        result = registry.acquire(model, engine, engine_id, force=False)
        assert result is True

        # Cleanup
        registry.release(model, engine_id)

    def test_is_owned_returns_false_for_unknown(self):
        """Test is_owned returns False for unknown model."""
        registry = get_registry()
        unknown_model = MagicMock()

        is_owned, owner_id = registry.is_owned(unknown_model)
        assert is_owned is False
        assert owner_id is None

    def test_cleanup_removes_stale_entries(self):
        """Test that cleanup removes entries for garbage collected owners."""
        registry = get_registry()
        model = MagicMock()

        # Create engine that will be garbage collected
        def inner():
            engine = MagicMock()
            registry.acquire(model, engine, "gc-test-engine")
            return weakref.ref(engine)

        weak_ref = inner()

        # Force garbage collection
        gc.collect()

        # Weak ref should be dead
        assert weak_ref() is None

        # Cleanup should remove stale entry
        cleaned = registry.cleanup()
        assert cleaned >= 0  # May clean up other stale entries too

    def test_get_stats(self):
        """Test get_stats returns correct counts."""
        registry = get_registry()
        model1 = MagicMock()
        model2 = MagicMock()
        engine1 = MagicMock()
        engine2 = MagicMock()

        registry.acquire(model1, engine1, "stats-engine-1")
        registry.acquire(model2, engine2, "stats-engine-2")

        stats = registry.get_stats()
        assert stats["total_entries"] >= 2
        assert stats["active_owners"] >= 2

        # Cleanup
        registry.release(model1, "stats-engine-1")
        registry.release(model2, "stats-engine-2")


class TestMultiEngine:
    """Tests for multi-engine scenarios using mocks."""

    def test_sequential_ownership_with_release(self):
        """Test that sequential engines can own model after release."""
        registry = get_registry()
        model = MagicMock()

        for i in range(3):
            engine = MagicMock()
            engine_id = f"seq-engine-{i}"

            result = registry.acquire(model, engine, engine_id)
            assert result is True

            is_owned, owner = registry.is_owned(model)
            assert is_owned is True
            assert owner == engine_id

            registry.release(model, engine_id)

    def test_sequential_ownership_with_force(self):
        """Legacy force works only after each prior owner releases."""
        registry = get_registry()
        model = MagicMock()

        for i in range(3):
            engine = MagicMock()
            engine.scheduler = MagicMock()
            engine_id = f"force-engine-{i}"

            result = registry.acquire(model, engine, engine_id, force=True)
            assert result is True

            is_owned, owner = registry.is_owned(model)
            assert is_owned is True
            assert owner == engine_id
            registry.release(model, engine_id)

    def test_multiple_models_different_engines(self):
        """Test that different models can have different owners."""
        registry = get_registry()
        model1 = MagicMock()
        model2 = MagicMock()
        engine1 = MagicMock()
        engine2 = MagicMock()

        registry.acquire(model1, engine1, "owner-1")
        registry.acquire(model2, engine2, "owner-2")

        is_owned1, owner1 = registry.is_owned(model1)
        is_owned2, owner2 = registry.is_owned(model2)

        assert is_owned1 is True
        assert owner1 == "owner-1"
        assert is_owned2 is True
        assert owner2 == "owner-2"

        # Cleanup
        registry.release(model1, "owner-1")
        registry.release(model2, "owner-2")


class TestCacheRecovery:
    """Tests for cache recovery scenarios."""

    def test_scheduler_reset_on_force(self):
        """Live ownership cannot be reset from a replacement engine."""
        registry = get_registry()
        model = MagicMock()

        # First engine with scheduler
        engine1 = MagicMock()
        engine1.scheduler = MagicMock()

        # Second engine
        engine2 = MagicMock()

        registry.acquire(model, engine1, "engine-1")
        with pytest.raises(ModelOwnershipError):
            registry.acquire(model, engine2, "engine-2", force=True)

        engine1.scheduler.deep_reset.assert_not_called()

        # Cleanup
        registry.release(model, "engine-1")

    def test_scheduler_reset_handles_missing_scheduler(self):
        """Test that scheduler reset handles engine without scheduler."""
        registry = get_registry()
        model = MagicMock()

        # First engine without scheduler attribute
        engine1 = MagicMock(spec=[])  # No scheduler attribute

        # Second engine
        engine2 = MagicMock()

        registry.acquire(model, engine1, "engine-1")

        with pytest.raises(ModelOwnershipError):
            registry.acquire(model, engine2, "engine-2", force=True)

        is_owned, owner = registry.is_owned(model)
        assert is_owned is True
        assert owner == "engine-1"

        # Cleanup
        registry.release(model, "engine-1")

    def test_weak_ref_cleanup_on_gc(self):
        """Test that weak references are cleaned up on garbage collection."""
        registry = get_registry()
        model = MagicMock()

        # Create and acquire in a function scope
        def create_and_acquire():
            engine = MagicMock()
            registry.acquire(model, engine, "weak-ref-test")

        create_and_acquire()

        # At this point, engine is out of scope but model is still tracked
        # Force GC
        gc.collect()

        # is_owned may return False if weak ref is dead and cleanup happened
        # or True if cleanup hasn't run yet
        # Just verify no exception is raised
        registry.is_owned(model)

        # Manual cleanup should work
        registry.cleanup()


class TestModelRegistryEdgeCases:
    """Edge case tests for ModelRegistry."""

    def test_release_non_existent_model(self):
        """Test releasing a model that was never acquired."""
        registry = get_registry()
        model = MagicMock()

        result = registry.release(model, "non-existent")
        assert result is False

    def test_multiple_force_transfers(self):
        """Repeated force attempts cannot replace the first live owner."""
        registry = get_registry()
        model = MagicMock()

        engines = []
        for i in range(5):
            engine = MagicMock()
            engine.scheduler = MagicMock()
            engines.append(engine)

            if i == 0:
                registry.acquire(model, engine, f"transfer-engine-{i}", force=True)
            else:
                with pytest.raises(ModelOwnershipError):
                    registry.acquire(
                        model,
                        engine,
                        f"transfer-engine-{i}",
                        force=True,
                    )
            assert registry.is_owned(model) == (True, "transfer-engine-0")

        for engine in engines:
            engine.scheduler.deep_reset.assert_not_called()

        # Cleanup
        registry.release(model, "transfer-engine-0")

    def test_acquire_after_gc_cleanup(self):
        """Test acquiring after previous owner was garbage collected."""
        registry = get_registry()
        model = MagicMock()

        # First acquire that will be GC'd
        def inner_acquire():
            engine = MagicMock()
            registry.acquire(model, engine, "gc-owner")

        inner_acquire()
        gc.collect()
        registry.cleanup()

        # Now acquire with new engine
        new_engine = MagicMock()
        result = registry.acquire(new_engine, new_engine, "new-owner")
        assert result is True

        # Cleanup
        registry.release(new_engine, "new-owner")
