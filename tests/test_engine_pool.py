# SPDX-License-Identifier: Apache-2.0
"""Tests for EnginePool functionality."""

import asyncio
import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.engine_pool import EngineEntry, EnginePool
from omlx.exceptions import (
    InsufficientMemoryError,
    ModelBusyError,
    ModelLoadingError,
    ModelNotFoundError,
    ModelTooLargeError,
)
from omlx.scheduler import PrefillEvictionRequest


def _make_pool(ceiling: int | None = None, **kwargs) -> EnginePool:
    """Helper: create an EnginePool with a stubbed final-ceiling callback.

    Tests historically constructed EnginePool with `max_model_memory=X`. The
    pre-load admission ceiling now comes from the ProcessMemoryEnforcer via
    a callback, so tests inject a fake callback that returns the desired
    ceiling. `ceiling=None` (or 0) disables the limit (callback returns 0).
    """
    pool = EnginePool(**kwargs)
    if ceiling is None or ceiling <= 0:
        pool._get_final_ceiling = lambda: 0
    else:
        pool._get_final_ceiling = lambda c=int(ceiling): c
    return pool


@pytest.fixture
def mock_model_dir(tmp_path):
    """Create a mock model directory with multiple models."""
    # Create model-a (1GB)
    model_a = tmp_path / "model-a"
    model_a.mkdir()
    (model_a / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (model_a / "model.safetensors").write_bytes(b"0" * (1024 * 1024 * 1024))  # 1GB

    # Create model-b (2GB)
    model_b = tmp_path / "model-b"
    model_b.mkdir()
    (model_b / "config.json").write_text(json.dumps({"model_type": "qwen"}))
    (model_b / "model.safetensors").write_bytes(b"0" * (2 * 1024 * 1024 * 1024))  # 2GB

    # Create model-c (500MB MLLM)
    model_c = tmp_path / "model-c"
    model_c.mkdir()
    (model_c / "config.json").write_text(json.dumps({"vision_config": {}}))
    (model_c / "model.safetensors").write_bytes(b"0" * (512 * 1024 * 1024))  # 500MB

    return tmp_path


@pytest.fixture
def small_mock_model_dir(tmp_path):
    """Create a mock model directory with small models for fast tests."""
    # Create model-a (1KB)
    model_a = tmp_path / "model-a"
    model_a.mkdir()
    (model_a / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (model_a / "model.safetensors").write_bytes(b"0" * 1024)

    # Create model-b (2KB)
    model_b = tmp_path / "model-b"
    model_b.mkdir()
    (model_b / "config.json").write_text(json.dumps({"model_type": "qwen"}))
    (model_b / "model.safetensors").write_bytes(b"0" * 2048)

    return tmp_path


class TestEnginePoolInit:
    """Tests for EnginePool initialization."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        pool = _make_pool(ceiling=32 * 1024**3)
        assert pool._current_ceiling() == 32 * 1024**3
        assert pool.current_model_memory == 0
        assert pool.model_count == 0
        assert pool.loaded_model_count == 0

    def test_init_disabled_memory(self):
        """Test initialization with disabled (None) memory limit."""
        pool = _make_pool(ceiling=None)
        assert pool._current_ceiling() == 0
        assert pool.current_model_memory == 0

    def test_model_too_large_skipped_when_disabled(self, small_mock_model_dir):
        """Test that ModelTooLargeError is NOT raised when memory is disabled."""
        pool = _make_pool(ceiling=None)
        pool.discover_models(str(small_mock_model_dir))
        # With None (disabled), the size check should be skipped entirely
        # Model loading will fail for other reasons (mock), but ModelTooLargeError
        # should not be raised
        entry = pool.get_entry("model-a")
        assert entry is not None
        # Verify the entry has a nonzero estimated size
        assert entry.estimated_size > 0

    def test_discover_models(self, small_mock_model_dir):
        """Test model discovery."""
        pool = _make_pool(ceiling=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        assert pool.model_count == 2
        assert "model-a" in pool.get_model_ids()
        assert "model-b" in pool.get_model_ids()

    def test_discover_models_with_pinned(self, small_mock_model_dir):
        """Test model discovery with pinned models."""
        pool = _make_pool(ceiling=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir), pinned_models=["model-a"])

        entry = pool.get_entry("model-a")
        assert entry is not None
        assert entry.is_pinned is True

        entry_b = pool.get_entry("model-b")
        assert entry_b is not None
        assert entry_b.is_pinned is False


class TestExposedProfileModelResolution:
    """Tests for exposed profile model IDs that share a physical engine."""

    def _manager_with_exposed_profile(self, tmp_path):
        from omlx.model_settings import ModelSettingsManager

        manager = ModelSettingsManager(tmp_path)
        manager.save_profile(
            model_id="model-b",
            name="thinking",
            display_name="Thinking",
            description=None,
            settings={"enable_thinking": True},
            expose_as_model=True,
        )
        return manager

    def test_resolve_model_id_maps_exposed_profile_to_source_model(
        self, small_mock_model_dir, tmp_path
    ):
        """Exposed profile model IDs resolve to their base model for loading."""
        pool = _make_pool(ceiling=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))
        manager = self._manager_with_exposed_profile(tmp_path)

        resolved = pool.resolve_model_id("model-b:thinking", manager)

        assert resolved == "model-b"

    def test_resolve_model_id_maps_provider_prefixed_exposed_profile_to_source(
        self, small_mock_model_dir, tmp_path
    ):
        """Provider prefixes do not prevent exposed profile resolution."""
        pool = _make_pool(ceiling=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))
        manager = self._manager_with_exposed_profile(tmp_path)

        resolved = pool.resolve_model_id("omlx/model-b:thinking", manager)

        assert resolved == "model-b"


class TestDiscoverModelsMerge:
    """Tests for discover_models merge behavior (issue #89)."""

    def test_rediscover_preserves_loaded_engine(self, small_mock_model_dir):
        """Test that re-discovery preserves loaded engine state."""
        pool = _make_pool(ceiling=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        # Simulate loaded model-a
        entry_a = pool.get_entry("model-a")
        mock_engine = MagicMock()
        entry_a.engine = mock_engine
        entry_a.last_access = 42.0
        pool._current_model_memory = entry_a.estimated_size

        original_size = entry_a.estimated_size

        # Re-discover (simulates model deletion or download completion)
        pool.discover_models(str(small_mock_model_dir))

        # Loaded model should be preserved
        entry_a_after = pool.get_entry("model-a")
        assert entry_a_after is entry_a  # Same object
        assert entry_a_after.engine is mock_engine
        assert entry_a_after.last_access == 42.0
        assert entry_a_after.estimated_size == original_size
        assert pool._current_model_memory == original_size
        assert pool.model_count == 2

    def test_rediscover_removes_stale_unloaded(self, small_mock_model_dir):
        """Test that unloaded models missing from disk are removed."""
        pool = _make_pool(ceiling=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))
        assert "model-a" in pool.get_model_ids()
        assert "model-b" in pool.get_model_ids()

        # Delete model-b from disk
        import shutil

        shutil.rmtree(small_mock_model_dir / "model-b")

        # Re-discover
        pool.discover_models(str(small_mock_model_dir))

        assert "model-a" in pool.get_model_ids()
        assert "model-b" not in pool.get_model_ids()
        assert pool.model_count == 1

    def test_rediscover_keeps_loaded_model_missing_from_disk(
        self, small_mock_model_dir
    ):
        """Test that loaded models are kept even if missing from disk."""
        pool = _make_pool(ceiling=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        # Simulate loaded model-b
        entry_b = pool.get_entry("model-b")
        entry_b.engine = MagicMock()
        entry_b.last_access = 99.0

        # Delete model-b from disk
        import shutil

        shutil.rmtree(small_mock_model_dir / "model-b")

        # Re-discover
        pool.discover_models(str(small_mock_model_dir))

        # model-b should still be in entries (loaded in memory)
        assert "model-b" in pool.get_model_ids()
        assert pool.get_entry("model-b").engine is not None
        assert pool.get_entry("model-b").last_access == 99.0

    def test_rediscover_updates_pinned_flag_on_loaded(self, small_mock_model_dir):
        """Test that pinned flag is updated on loaded models during re-discovery."""
        pool = _make_pool(ceiling=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        # Simulate loaded model-a (not pinned)
        entry_a = pool.get_entry("model-a")
        entry_a.engine = MagicMock()
        assert entry_a.is_pinned is False

        # Re-discover with model-a pinned
        pool.discover_models(str(small_mock_model_dir), pinned_models=["model-a"])

        # Pinned flag should be updated, engine preserved
        assert pool.get_entry("model-a").is_pinned is True
        assert pool.get_entry("model-a").engine is not None

        # Re-discover without pinning
        pool.discover_models(str(small_mock_model_dir), pinned_models=[])
        assert pool.get_entry("model-a").is_pinned is False
        assert pool.get_entry("model-a").engine is not None


class TestEnginePoolErrors:
    """Tests for EnginePool error handling."""

    def test_model_not_found_error(self, small_mock_model_dir):
        """Test ModelNotFoundError when model doesn't exist."""
        pool = _make_pool(ceiling=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        with pytest.raises(ModelNotFoundError) as exc_info:
            asyncio.run(pool.get_engine("nonexistent"))

        assert exc_info.value.model_id == "nonexistent"
        assert "model-a" in exc_info.value.available_models

    def test_model_too_large_error(self, small_mock_model_dir):
        """Test ModelTooLargeError when model exceeds memory limit."""
        # Set very small memory limit
        pool = _make_pool(ceiling=100)  # 100 bytes
        pool.discover_models(str(small_mock_model_dir))

        with pytest.raises(ModelTooLargeError) as exc_info:
            asyncio.run(pool.get_engine("model-a"))

        assert exc_info.value.model_id == "model-a"
        assert exc_info.value.ceiling == 100


class TestEnginePoolStatus:
    """Tests for EnginePool status reporting."""

    def test_get_status(self, small_mock_model_dir):
        """Test get_status returns correct information."""
        pool = _make_pool(ceiling=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir), pinned_models=["model-a"])

        status = pool.get_status()

        assert status["final_ceiling"] == 10 * 1024**3
        assert status["current_model_memory"] == 0
        assert status["model_count"] == 2
        assert status["loaded_count"] == 0
        assert len(status["models"]) == 2

        # Check model details
        model_ids = {m["id"] for m in status["models"]}
        assert model_ids == {"model-a", "model-b"}

        # Check pinned status
        model_a_status = next(m for m in status["models"] if m["id"] == "model-a")
        assert model_a_status["pinned"] is True
        assert model_a_status["loaded"] is False

    def test_get_model_ids(self, small_mock_model_dir):
        """Test get_model_ids returns all model IDs."""
        pool = _make_pool(ceiling=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        ids = pool.get_model_ids()
        assert set(ids) == {"model-a", "model-b"}

    def test_get_loaded_model_ids_empty(self, small_mock_model_dir):
        """Test get_loaded_model_ids when no models loaded."""
        pool = _make_pool(ceiling=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        assert pool.get_loaded_model_ids() == []


class TestEngineEntry:
    """Tests for EngineEntry dataclass."""

    def test_entry_defaults(self):
        """Test EngineEntry default values."""
        entry = EngineEntry(
            model_id="test",
            model_path="/path/to/model",
            model_type="llm",
            engine_type="batched",
            estimated_size=1000,
        )

        assert entry.engine is None
        assert entry.last_access == 0.0
        assert entry.is_loading is False
        assert entry.is_pinned is False

    def test_entry_with_values(self):
        """Test EngineEntry with custom values."""
        entry = EngineEntry(
            model_id="test",
            model_path="/path/to/model",
            model_type="embedding",
            engine_type="embedding",
            estimated_size=2000,
            is_pinned=True,
        )

        assert entry.model_type == "embedding"
        assert entry.engine_type == "embedding"
        assert entry.is_pinned is True


class TestApplySettingsOverrides:
    """Tests for apply_settings_overrides method."""

    def test_override_changes_model_type(self, small_mock_model_dir):
        """Test that model_type_override changes entry model_type and engine_type."""
        pool = _make_pool(ceiling=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        # Verify auto-detected types
        assert pool.get_entry("model-a").model_type == "llm"
        assert pool.get_entry("model-a").engine_type == "batched"

        # Mock settings manager
        from omlx.model_settings import ModelSettings

        settings_manager = MagicMock()
        settings_manager.get_settings.side_effect = lambda mid: (
            ModelSettings(model_type_override="vlm")
            if mid == "model-a"
            else ModelSettings()
        )

        pool.apply_settings_overrides(settings_manager)

        # model-a should be overridden to vlm
        assert pool.get_entry("model-a").model_type == "vlm"
        assert pool.get_entry("model-a").engine_type == "vlm"

        # model-b should remain unchanged (no override)
        assert pool.get_entry("model-b").model_type == "llm"
        assert pool.get_entry("model-b").engine_type == "batched"

    def test_no_override_leaves_entry_unchanged(self, small_mock_model_dir):
        """Test that None override doesn't change entry types."""
        pool = _make_pool(ceiling=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        from omlx.model_settings import ModelSettings

        settings_manager = MagicMock()
        settings_manager.get_settings.return_value = ModelSettings()

        pool.apply_settings_overrides(settings_manager)

        assert pool.get_entry("model-a").model_type == "llm"
        assert pool.get_entry("model-a").engine_type == "batched"


class TestVLMFallback:
    """Tests for VLM-to-LLM fallback during engine loading."""

    @pytest.mark.asyncio
    async def test_vlm_fallback_to_llm_on_start_failure(self, small_mock_model_dir):
        """Test that VLM loading failure falls back to LLM BatchedEngine."""
        pool = _make_pool(ceiling=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        # Force model-a to be VLM type
        entry = pool.get_entry("model-a")
        entry.model_type = "vlm"
        entry.engine_type = "vlm"

        # VLM engine that fails on start
        mock_vlm_engine = MagicMock()
        mock_vlm_engine.start = AsyncMock(
            side_effect=Exception("Missing vision_tower parameters")
        )
        mock_vlm_engine.stop = AsyncMock()

        # Batched engine that succeeds
        mock_batched_engine = MagicMock()
        mock_batched_engine.start = AsyncMock()

        with (
            patch("omlx.engine_pool.VLMBatchedEngine", return_value=mock_vlm_engine),
            patch("omlx.engine_pool.BatchedEngine", return_value=mock_batched_engine),
        ):
            await pool._load_engine("model-a")

        # Should have fallen back to LLM
        assert entry.model_type == "llm"
        assert entry.engine_type == "batched"
        assert entry.engine is mock_batched_engine

    @pytest.mark.asyncio
    async def test_non_vlm_failure_still_raises(self, small_mock_model_dir):
        """Test that non-VLM engine failures propagate normally."""
        pool = _make_pool(ceiling=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        entry = pool.get_entry("model-a")
        assert entry.engine_type == "batched"  # LLM, not VLM

        mock_engine = MagicMock()
        mock_engine.start = AsyncMock(side_effect=Exception("Load failed"))

        with (
            patch("omlx.engine_pool.BatchedEngine", return_value=mock_engine),
            pytest.raises(Exception, match="Load failed"),
        ):
            await pool._load_engine("model-a")

    @pytest.mark.asyncio
    async def test_force_lm_fallback_to_vlm_on_start_failure(
        self, small_mock_model_dir
    ):
        """Test that force_lm failure for VLM model falls back to VLMBatchedEngine."""
        pool = _make_pool(ceiling=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        # Force model-a to be VLM type
        entry = pool.get_entry("model-a")
        entry.model_type = "vlm"
        entry.engine_type = "vlm"

        # BatchedEngine (force_lm) fails on start
        mock_batched_engine = MagicMock()
        mock_batched_engine.start = AsyncMock(
            side_effect=TypeError(
                "ModelArgs.__init__() missing 1 required positional argument: "
                "'tie_word_embeddings'"
            )
        )
        mock_batched_engine.stop = AsyncMock()

        # VLMBatchedEngine succeeds
        mock_vlm_engine = MagicMock()
        mock_vlm_engine.start = AsyncMock()

        with (
            patch("omlx.engine_pool.BatchedEngine", return_value=mock_batched_engine),
            patch("omlx.engine_pool.VLMBatchedEngine", return_value=mock_vlm_engine),
        ):
            await pool._load_engine("model-a", force_lm=True)

        # Should have fallen back to VLM
        assert entry.model_type == "vlm"
        assert entry.engine_type == "vlm"
        assert entry.engine is mock_vlm_engine

    @pytest.mark.asyncio
    async def test_force_lm_no_fallback_for_non_vlm(self, small_mock_model_dir):
        """Test that force_lm failure for non-VLM model still raises."""
        pool = _make_pool(ceiling=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        entry = pool.get_entry("model-a")
        assert entry.engine_type == "batched"  # LLM, not VLM

        mock_engine = MagicMock()
        mock_engine.start = AsyncMock(side_effect=Exception("Load failed"))

        with (
            patch("omlx.engine_pool.BatchedEngine", return_value=mock_engine),
            pytest.raises(Exception, match="Load failed"),
        ):
            await pool._load_engine("model-a", force_lm=True)

    @pytest.mark.asyncio
    async def test_vlm_fallback_to_llm_both_fail_surfaces_both_errors(
        self, small_mock_model_dir
    ):
        """When VLM start fails AND the LLM fallback also fails, the raised
        RuntimeError should embed both messages and chain ``__cause__`` to the
        original VLM error (PR #1283)."""
        pool = _make_pool(ceiling=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        entry = pool.get_entry("model-a")
        entry.model_type = "vlm"
        entry.engine_type = "vlm"

        mock_vlm_engine = MagicMock()
        mock_vlm_engine.start = AsyncMock(
            side_effect=Exception("Missing vision_tower parameters")
        )
        mock_vlm_engine.stop = AsyncMock()

        mock_batched_engine = MagicMock()
        mock_batched_engine.start = AsyncMock(
            side_effect=Exception("Model type lfm2_vl not supported")
        )

        with (
            patch("omlx.engine_pool.VLMBatchedEngine", return_value=mock_vlm_engine),
            patch("omlx.engine_pool.BatchedEngine", return_value=mock_batched_engine),
            pytest.raises(RuntimeError) as excinfo,
        ):
            await pool._load_engine("model-a")

        msg = str(excinfo.value)
        assert "VLM load failed" in msg
        assert "Missing vision_tower parameters" in msg
        assert "LLM fallback also failed" in msg
        assert "Model type lfm2_vl not supported" in msg
        # __cause__ chain preserves the original VLM error
        assert excinfo.value.__cause__ is not None
        assert "Missing vision_tower parameters" in str(excinfo.value.__cause__)

    @pytest.mark.asyncio
    async def test_force_lm_fallback_to_vlm_both_fail_surfaces_both_errors(
        self, small_mock_model_dir
    ):
        """force_lm path: LM start fails AND VLM fallback also fails. Both
        error messages should land in the raised RuntimeError, with the LM
        error as ``__cause__`` (PR #1283)."""
        pool = _make_pool(ceiling=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        entry = pool.get_entry("model-a")
        entry.model_type = "vlm"
        entry.engine_type = "vlm"

        mock_batched_engine = MagicMock()
        mock_batched_engine.start = AsyncMock(
            side_effect=TypeError(
                "ModelArgs.__init__() missing 1 required positional argument: "
                "'tie_word_embeddings'"
            )
        )
        mock_batched_engine.stop = AsyncMock()

        mock_vlm_engine = MagicMock()
        mock_vlm_engine.start = AsyncMock(
            side_effect=Exception("vision encoder weights missing")
        )

        with (
            patch("omlx.engine_pool.BatchedEngine", return_value=mock_batched_engine),
            patch("omlx.engine_pool.VLMBatchedEngine", return_value=mock_vlm_engine),
            pytest.raises(RuntimeError) as excinfo,
        ):
            await pool._load_engine("model-a", force_lm=True)

        msg = str(excinfo.value)
        assert "LM load failed" in msg
        assert "force_lm=True" in msg
        assert "tie_word_embeddings" in msg
        assert "VLM fallback also failed" in msg
        assert "vision encoder weights missing" in msg
        assert isinstance(excinfo.value.__cause__, TypeError)


class TestEnginePoolLRU:
    """Tests for LRU eviction logic."""

    @pytest.fixture
    def pool_with_entries(self, small_mock_model_dir):
        """Create pool with mock entries for LRU testing."""
        pool = _make_pool(ceiling=5000)  # 5KB limit
        pool.discover_models(str(small_mock_model_dir))
        return pool

    def test_find_lru_victim_no_loaded(self, pool_with_entries):
        """Test finding LRU victim when no models loaded."""
        victim = pool_with_entries._find_lru_victim()
        assert victim is None

    def test_find_lru_victim_all_pinned(self, pool_with_entries):
        """Test finding LRU victim when all loaded models are pinned."""
        # Mark both as pinned
        pool_with_entries._entries["model-a"].is_pinned = True
        pool_with_entries._entries["model-b"].is_pinned = True

        # Simulate loaded state
        pool_with_entries._entries["model-a"].engine = MagicMock()
        pool_with_entries._entries["model-a"].last_access = 100

        victim = pool_with_entries._find_lru_victim()
        assert victim is None

    def test_find_lru_victim_oldest_first(self, pool_with_entries):
        """Test that oldest (lowest last_access) is selected."""
        # Simulate loaded state with different access times
        mock_a = MagicMock()
        mock_a.has_active_requests.return_value = False
        pool_with_entries._entries["model-a"].engine = mock_a
        pool_with_entries._entries["model-a"].last_access = 100  # Older

        mock_b = MagicMock()
        mock_b.has_active_requests.return_value = False
        pool_with_entries._entries["model-b"].engine = mock_b
        pool_with_entries._entries["model-b"].last_access = 200  # Newer

        victim = pool_with_entries._find_lru_victim()
        assert victim == "model-a"

    def test_pinned_model_skipped_for_eviction(self, pool_with_entries):
        """Test that pinned models are skipped during eviction."""
        # model-a is pinned and older
        pool_with_entries._entries["model-a"].is_pinned = True
        mock_a = MagicMock()
        mock_a.has_active_requests.return_value = False
        pool_with_entries._entries["model-a"].engine = mock_a
        pool_with_entries._entries["model-a"].last_access = 50

        # model-b is not pinned and newer
        mock_b = MagicMock()
        mock_b.has_active_requests.return_value = False
        pool_with_entries._entries["model-b"].engine = mock_b
        pool_with_entries._entries["model-b"].last_access = 200

        victim = pool_with_entries._find_lru_victim()
        # model-a is skipped (pinned), model-b is selected
        assert victim == "model-b"

    def test_find_lru_victim_skips_active_requests(self, pool_with_entries):
        """Test that models with active requests are skipped during eviction."""
        # model-a has active requests
        mock_engine_a = MagicMock()
        mock_engine_a.has_active_requests.return_value = True
        pool_with_entries._entries["model-a"].engine = mock_engine_a
        pool_with_entries._entries["model-a"].last_access = 50  # Older

        # model-b has no active requests
        mock_engine_b = MagicMock()
        mock_engine_b.has_active_requests.return_value = False
        pool_with_entries._entries["model-b"].engine = mock_engine_b
        pool_with_entries._entries["model-b"].last_access = 200  # Newer

        victim = pool_with_entries._find_lru_victim()
        # model-a skipped (active requests), model-b selected
        assert victim == "model-b"

    def test_find_lru_victim_all_active(self, pool_with_entries):
        """Test that None is returned when all models have active requests."""
        for mid in ("model-a", "model-b"):
            mock_engine = MagicMock()
            mock_engine.has_active_requests.return_value = True
            pool_with_entries._entries[mid].engine = mock_engine
            pool_with_entries._entries[mid].last_access = 100

        victim = pool_with_entries._find_lru_victim()
        assert victim is None

    def test_find_lru_victim_no_has_active_requests(self, pool_with_entries):
        """Test graceful handling when engine lacks has_active_requests."""
        mock_engine = MagicMock(spec=[])  # No has_active_requests
        pool_with_entries._entries["model-a"].engine = mock_engine
        pool_with_entries._entries["model-a"].last_access = 100

        victim = pool_with_entries._find_lru_victim()
        assert victim == "model-a"


class TestEnginePoolDFlashIsolation:
    """Tests for DFlash process-global runtime isolation."""

    class DFlashEngine:
        def __init__(self, *, active: bool = False):
            self.active = active

        def has_active_requests(self):
            return self.active

    class OtherEngine:
        def has_active_requests(self):
            return False

    @staticmethod
    def _entry(model_id: str, engine) -> EngineEntry:
        return EngineEntry(
            model_id=model_id,
            model_path=f"/models/{model_id}",
            model_type="llm",
            engine_type="batched",
            estimated_size=1024,
            engine=engine,
        )

    @pytest.mark.asyncio
    async def test_unload_other_dflash_engines_unloads_idle_dflash_only(self):
        pool = EnginePool()
        pool._entries["old-dflash"] = self._entry("old-dflash", self.DFlashEngine())
        pool._entries["other"] = self._entry("other", self.OtherEngine())
        pool._entries["new-dflash"] = self._entry("new-dflash", None)

        unloaded = []

        async def fake_unload(model_id):
            unloaded.append(model_id)
            pool._entries[model_id].engine = None

        pool._unload_engine = fake_unload

        await pool._unload_other_dflash_engines("new-dflash")

        assert unloaded == ["old-dflash"]
        assert pool._entries["other"].engine is not None

    @pytest.mark.asyncio
    async def test_unload_other_dflash_engines_blocks_active_dflash(self):
        pool = EnginePool()
        pool._entries["active-dflash"] = self._entry(
            "active-dflash", self.DFlashEngine(active=True)
        )
        pool._entries["new-dflash"] = self._entry("new-dflash", None)
        pool._unload_engine = AsyncMock()

        with pytest.raises(RuntimeError, match="active-dflash"):
            await pool._unload_other_dflash_engines("new-dflash")

        pool._unload_engine.assert_not_awaited()


class TestEnginePoolAsync:
    """Async tests for EnginePool (mocked)."""

    @pytest.fixture
    def pool_with_mock_engines(self, small_mock_model_dir):
        """Create pool with mocked engine loading."""
        pool = _make_pool(ceiling=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))
        return pool

    @pytest.mark.asyncio
    async def test_get_engine_loads_model(self, pool_with_mock_engines):
        """Test that get_engine loads the model."""
        pool = pool_with_mock_engines

        # Mock the engine loading
        mock_engine = MagicMock()
        mock_engine.start = AsyncMock()
        mock_engine.stop = AsyncMock()

        with patch("omlx.engine_pool.BatchedEngine", return_value=mock_engine):
            engine = await pool.get_engine("model-a")

        assert engine == mock_engine
        mock_engine.start.assert_called_once()
        assert pool.loaded_model_count == 1
        assert pool.current_model_memory > 0

    @pytest.mark.asyncio
    async def test_runtime_settings_signature_reload(self, pool_with_mock_engines):
        """A profile runtime variant with engine fields reloads the base engine."""
        from omlx.model_settings import ModelSettings

        pool = pool_with_mock_engines
        pool._settings_manager = MagicMock()
        pool._settings_manager.get_settings.return_value = ModelSettings(
            mtp_enabled=False
        )

        base_engine = MagicMock()
        base_engine.start = AsyncMock()
        base_engine.stop = AsyncMock()
        profile_engine = MagicMock()
        profile_engine.start = AsyncMock()
        profile_engine.stop = AsyncMock()

        with patch(
            "omlx.engine_pool.BatchedEngine",
            side_effect=[base_engine, profile_engine],
        ):
            first = await pool.get_engine("model-a")
            second = await pool.get_engine(
                "model-a",
                runtime_settings=ModelSettings(mtp_enabled=True),
            )

        assert first is base_engine
        assert second is profile_engine
        base_engine.stop.assert_awaited_once()
        assert pool.get_entry("model-a").runtime_settings_signature == (
            pool._engine_runtime_signature(
                "model-a",
                ModelSettings(mtp_enabled=True),
            )
        )

    @pytest.mark.asyncio
    async def test_runtime_settings_reload_rejected_while_leased(
        self, pool_with_mock_engines
    ):
        """A profile variant switch must not unload an engine held by a request."""
        from omlx.model_settings import ModelSettings

        pool = pool_with_mock_engines
        pool._settings_manager = MagicMock()
        pool._settings_manager.get_settings.return_value = ModelSettings(
            mtp_enabled=False
        )

        base_engine = MagicMock()
        base_engine.start = AsyncMock()
        base_engine.stop = AsyncMock()

        with patch("omlx.engine_pool.BatchedEngine", return_value=base_engine):
            first = await pool.get_engine("model-a")
            pool.get_entry("model-a").in_use = 1
            with pytest.raises(ModelBusyError, match="runtime settings variant"):
                await pool.get_engine(
                    "model-a",
                    runtime_settings=ModelSettings(mtp_enabled=True),
                )

        assert first is base_engine
        assert pool.get_entry("model-a").engine is base_engine
        base_engine.stop.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_runtime_sampling_only_profile_reuses_loaded_engine(
        self, pool_with_mock_engines
    ):
        from omlx.model_settings import ModelSettings

        pool = pool_with_mock_engines
        pool._settings_manager = MagicMock()
        pool._settings_manager.get_settings.return_value = ModelSettings(
            mtp_enabled=False
        )

        base_engine = MagicMock()
        base_engine.start = AsyncMock()
        base_engine.stop = AsyncMock()

        with patch("omlx.engine_pool.BatchedEngine", return_value=base_engine):
            first = await pool.get_engine("model-a")
            second = await pool.get_engine(
                "model-a",
                runtime_settings=ModelSettings(temperature=0.9, mtp_enabled=False),
            )

        assert first is base_engine
        assert second is base_engine
        base_engine.stop.assert_not_awaited()

    def test_runtime_signature_ignores_request_only_profile_fields(
        self, pool_with_mock_engines
    ):
        from omlx.model_settings import ModelSettings

        pool = pool_with_mock_engines

        pure = ModelSettings(
            enable_thinking=False,
            dflash_enabled=False,
            dflash_draft_model="/stale/draft",
            vlm_mtp_enabled=False,
            vlm_mtp_draft_model="/stale/assistant",
        )
        pure_think = ModelSettings(
            enable_thinking=True,
            dflash_enabled=False,
            dflash_draft_model=None,
            vlm_mtp_enabled=False,
            vlm_mtp_draft_model=None,
        )
        dflash = ModelSettings(
            dflash_enabled=True,
            dflash_draft_model="/draft",
        )
        vlm_mtp = ModelSettings(
            vlm_mtp_enabled=True,
            vlm_mtp_draft_model="/assistant",
        )

        pure_signature = pool._engine_runtime_signature("model-a", pure)
        pure_think_signature = pool._engine_runtime_signature("model-a", pure_think)

        assert pure_signature == pure_think_signature
        assert pool._engine_runtime_signature("model-a", dflash) != pure_signature
        assert pool._engine_runtime_signature("model-a", vlm_mtp) != pure_signature

    @pytest.mark.asyncio
    async def test_base_request_reloads_after_profile_variant(
        self, pool_with_mock_engines
    ):
        from omlx.model_settings import ModelSettings

        pool = pool_with_mock_engines
        pool._settings_manager = MagicMock()
        pool._settings_manager.get_settings.return_value = ModelSettings(
            mtp_enabled=False
        )

        profile_engine = MagicMock()
        profile_engine.start = AsyncMock()
        profile_engine.stop = AsyncMock()
        base_engine = MagicMock()
        base_engine.start = AsyncMock()
        base_engine.stop = AsyncMock()

        with patch(
            "omlx.engine_pool.BatchedEngine",
            side_effect=[profile_engine, base_engine],
        ):
            first = await pool.get_engine(
                "model-a",
                runtime_settings=ModelSettings(mtp_enabled=True),
            )
            second = await pool.get_engine("model-a")

        assert first is profile_engine
        assert second is base_engine
        profile_engine.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_embedding_engine_receives_scheduler_config(self, tmp_path):
        """Embedding chunk sizing should come from the shared scheduler config."""
        from omlx.scheduler import SchedulerConfig

        model_path = tmp_path / "embed-model"
        model_path.mkdir()
        scheduler_config = SchedulerConfig(
            completion_batch_size=6,
            embedding_batch_size=4,
        )
        pool = _make_pool(
            ceiling=10 * 1024**3,
            scheduler_config=scheduler_config,
        )
        pool._entries["embed-model"] = EngineEntry(
            model_id="embed-model",
            model_path=str(model_path),
            model_type="embedding",
            engine_type="embedding",
            estimated_size=1024,
        )

        mock_engine = MagicMock()
        mock_engine.start = AsyncMock()

        with patch(
            "omlx.engine_pool.EmbeddingEngine",
            return_value=mock_engine,
        ) as MockEmbeddingEngine:
            engine = await pool.get_engine("embed-model")

        assert engine is mock_engine
        MockEmbeddingEngine.assert_called_once_with(
            model_name=str(model_path),
            trust_remote_code=False,
            scheduler_config=scheduler_config,
        )

    @pytest.mark.asyncio
    async def test_embedding_engine_receives_fallback_scheduler_config(self, tmp_path):
        """A bare EnginePool should pass its fallback scheduler config consistently."""
        model_path = tmp_path / "embed-model"
        model_path.mkdir()
        pool = _make_pool(ceiling=10 * 1024**3)
        pool._entries["embed-model"] = EngineEntry(
            model_id="embed-model",
            model_path=str(model_path),
            model_type="embedding",
            engine_type="embedding",
            estimated_size=1024,
        )

        mock_engine = MagicMock()
        mock_engine.start = AsyncMock()

        with patch(
            "omlx.engine_pool.EmbeddingEngine",
            return_value=mock_engine,
        ) as MockEmbeddingEngine:
            engine = await pool.get_engine("embed-model")

        assert engine is mock_engine
        MockEmbeddingEngine.assert_called_once_with(
            model_name=str(model_path),
            trust_remote_code=False,
            scheduler_config=pool._scheduler_config,
        )

    @pytest.mark.asyncio
    async def test_apply_embedding_batch_size_updates_loaded_embedding_engines(self):
        """Runtime setting changes should update pool config and loaded embedding engines."""
        from omlx.engine.embedding import EmbeddingEngine
        from omlx.scheduler import SchedulerConfig

        engine = EmbeddingEngine("embed-model", batch_size=8)
        pool = _make_pool(
            ceiling=10 * 1024**3,
            scheduler_config=SchedulerConfig(embedding_batch_size=8),
        )
        pool._entries["embed-model"] = EngineEntry(
            model_id="embed-model",
            model_path="/tmp/embed-model",
            model_type="embedding",
            engine_type="embedding",
            estimated_size=1024,
            engine=engine,
        )

        await pool.apply_embedding_batch_size(5)

        assert pool._scheduler_config.embedding_batch_size == 5
        assert engine.get_stats()["batch_size"] == 5

    @pytest.mark.asyncio
    async def test_get_engine_returns_cached(self, pool_with_mock_engines):
        """Test that get_engine returns cached engine on second call."""
        pool = pool_with_mock_engines

        mock_engine = MagicMock()
        mock_engine.start = AsyncMock()

        with patch("omlx.engine_pool.BatchedEngine", return_value=mock_engine):
            engine1 = await pool.get_engine("model-a")
            engine2 = await pool.get_engine("model-a")

        assert engine1 is engine2
        # start() should only be called once
        assert mock_engine.start.call_count == 1

    @pytest.mark.asyncio
    async def test_unload_engine(self, pool_with_mock_engines):
        """Test unloading an engine."""
        pool = pool_with_mock_engines

        mock_engine = MagicMock()
        mock_engine.start = AsyncMock()
        mock_engine.stop = AsyncMock()

        with patch("omlx.engine_pool.BatchedEngine", return_value=mock_engine):
            await pool.get_engine("model-a")
            initial_memory = pool.current_model_memory

            await pool._unload_engine("model-a")

        mock_engine.stop.assert_called_once()
        assert pool.current_model_memory < initial_memory
        assert pool._entries["model-a"].engine is None

    @pytest.mark.asyncio
    async def test_shutdown_unloads_all(self, pool_with_mock_engines):
        """Test that shutdown unloads all engines."""
        pool = pool_with_mock_engines

        mock_engine_a = MagicMock()
        mock_engine_a.start = AsyncMock()
        mock_engine_a.stop = AsyncMock()

        mock_engine_b = MagicMock()
        mock_engine_b.start = AsyncMock()
        mock_engine_b.stop = AsyncMock()

        engines = [mock_engine_a, mock_engine_b]
        engine_idx = [0]

        def create_engine(*args, **kwargs):
            engine = engines[engine_idx[0]]
            engine_idx[0] += 1
            return engine

        with patch("omlx.engine_pool.BatchedEngine", side_effect=create_engine):
            await pool.get_engine("model-a")
            await pool.get_engine("model-b")

            await pool.shutdown()

        mock_engine_a.stop.assert_called_once()
        mock_engine_b.stop.assert_called_once()
        assert pool.loaded_model_count == 0


class TestEnginePoolEviction:
    """Tests for memory-based eviction."""

    @pytest.fixture
    def tight_memory_pool(self, small_mock_model_dir, monkeypatch):
        """Create pool with tight memory limit.

        Admission compares `phys_footprint + model_size` against the
        ceiling. For these byte-sized synthetic ceilings we proxy
        phys_footprint to the pool's tracked model weight sum so loading
        a second model triggers eviction the same way the original
        max_model_memory logic did.
        """
        # Model-a is ~1KB (1024 bytes + overhead ~1.1KB)
        # Model-b is ~2KB (2048 bytes + overhead ~2.1KB)
        # Set limit to allow each model individually but not both together
        pool = _make_pool(ceiling=2500)  # Allows each but not both
        pool.discover_models(str(small_mock_model_dir))
        monkeypatch.setattr(
            "omlx.engine_pool.get_phys_footprint",
            lambda: pool._current_model_memory,
        )
        monkeypatch.setattr("omlx.engine_pool.mx.get_active_memory", lambda: 0)
        return pool

    @pytest.mark.asyncio
    async def test_eviction_before_load(self, tight_memory_pool):
        """Test that eviction happens before loading new model."""
        pool = tight_memory_pool

        mock_engine_a = MagicMock()
        mock_engine_a.start = AsyncMock()
        mock_engine_a.stop = AsyncMock()
        mock_engine_a.has_active_requests.return_value = False

        mock_engine_b = MagicMock()
        mock_engine_b.start = AsyncMock()
        mock_engine_b.has_active_requests.return_value = False

        call_count = [0]

        def create_engine(*args, **kwargs):
            call_count[0] += 1
            if "model-a" in str(kwargs.get("model_name", args[0] if args else "")):
                return mock_engine_a
            return mock_engine_b

        with patch("omlx.engine_pool.BatchedEngine", side_effect=create_engine):
            # Load model-a first
            await pool.get_engine("model-a")
            assert pool.loaded_model_count == 1

            # Load model-b - should evict model-a first
            await pool.get_engine("model-b")

        # model-a should have been unloaded
        mock_engine_a.stop.assert_called_once()
        assert pool._entries["model-a"].engine is None
        assert pool._entries["model-b"].engine is not None

    @pytest.fixture
    def undercounting_memory_pool(self, small_mock_model_dir, monkeypatch):
        """Pool where live Metal memory under-reports the committed footprint.

        The #1623 condition: after a model settles/idles, both
        `mx.get_active_memory()` and `get_phys_footprint()` can read well below
        the model's true resident size, while the tracked accumulator
        (`_current_model_memory`) still reflects it. Unlike `tight_memory_pool`,
        this fixture does NOT proxy phys_footprint to the accumulator — it leaves
        live memory at 0 so admission has to consult the accumulator itself.
        """
        pool = _make_pool(ceiling=2500)  # each model fits alone, not both
        pool.discover_models(str(small_mock_model_dir))
        monkeypatch.setattr("omlx.engine_pool.get_phys_footprint", lambda: 0)
        monkeypatch.setattr("omlx.engine_pool.mx.get_active_memory", lambda: 0)
        return pool

    @pytest.mark.asyncio
    async def test_eviction_when_live_memory_undercounts(
        self, undercounting_memory_pool
    ):
        """#1623: a second model must still evict the first when live memory
        under-reports but the tracked accumulator shows the pair over-commits.

        Without consulting `_current_model_memory`, admission sees
        `current = max(0, 0) = 0`, projects `0 + model-b` under the ceiling, and
        loads model-b alongside model-a — over-committing past the ceiling.
        """
        pool = undercounting_memory_pool

        mock_engine_a = MagicMock()
        mock_engine_a.start = AsyncMock()
        mock_engine_a.stop = AsyncMock()
        mock_engine_a.has_active_requests.return_value = False

        mock_engine_b = MagicMock()
        mock_engine_b.start = AsyncMock()
        mock_engine_b.has_active_requests.return_value = False

        def create_engine(*args, **kwargs):
            if "model-a" in str(kwargs.get("model_name", args[0] if args else "")):
                return mock_engine_a
            return mock_engine_b

        with patch("omlx.engine_pool.BatchedEngine", side_effect=create_engine):
            await pool.get_engine("model-a")
            assert pool.loaded_model_count == 1

            # Load model-b: the pair exceeds the ceiling, so model-a must be
            # evicted first even though live memory reads 0.
            await pool.get_engine("model-b")

        mock_engine_a.stop.assert_called_once()
        assert pool._entries["model-a"].engine is None
        assert pool._entries["model-b"].engine is not None
        assert pool.loaded_model_count == 1

    @pytest.mark.asyncio
    async def test_insufficient_memory_all_pinned(self, tight_memory_pool):
        """Test InsufficientMemoryError when all models are pinned."""
        pool = tight_memory_pool

        # Pin model-a
        pool._entries["model-a"].is_pinned = True

        mock_engine = MagicMock()
        mock_engine.start = AsyncMock()

        with patch("omlx.engine_pool.BatchedEngine", return_value=mock_engine):
            # Load pinned model-a
            await pool.get_engine("model-a")

            # Try to load model-b - should fail (can't evict pinned model-a)
            with pytest.raises(InsufficientMemoryError):
                await pool.get_engine("model-b")


class TestEnginePoolPrefillEviction:
    """Tests for request-time idle LRU eviction before prefill throttling."""

    @staticmethod
    def _entry(model_id: str, size: int, *, active: bool = False) -> EngineEntry:
        engine = MagicMock()
        engine.has_active_requests.return_value = active
        engine.scheduler = None
        engine._engine = None
        return EngineEntry(
            model_id=model_id,
            model_path=f"/models/{model_id}",
            model_type="llm",
            engine_type="batched",
            estimated_size=size,
            engine=engine,
            last_access=0.0,
        )

    @pytest.mark.asyncio
    async def test_prefill_eviction_evicts_idle_lru_until_target(self):
        gb = 1024**3
        pool = _make_pool(ceiling=0)
        pool._entries = {
            "idle-a": self._entry("idle-a", 20 * gb),
            "idle-b": self._entry("idle-b", 15 * gb),
            "target": self._entry("target", 25 * gb),
        }
        pool._entries["idle-a"].last_access = 1.0
        pool._entries["idle-b"].last_access = 2.0
        pool._entries["target"].last_access = 3.0
        pool._current_model_memory = 60 * gb
        unloaded = []

        async def fake_unload(model_id):
            unloaded.append(model_id)
            entry = pool._entries[model_id]
            entry.engine = None
            pool._current_model_memory -= entry.estimated_size

        pool._unload_engine = fake_unload
        req = PrefillEvictionRequest(
            request_id="req-1",
            model_id="target",
            current_bytes=60 * gb,
            target_cap_bytes=40 * gb,
            predicted_transient_bytes=10 * gb,
            requested_tokens=2048,
            reason="adaptive_prefill_throttle",
        )

        with (
            patch("omlx.engine_pool.mx.get_active_memory", return_value=0),
            patch("omlx.engine_pool.get_phys_footprint", return_value=0),
        ):
            evicted = await pool._evict_idle_lru_for_prefill("target", req)

        assert evicted is True
        assert unloaded == ["idle-a", "idle-b"]
        assert pool._entries["target"].engine is not None

    @pytest.mark.asyncio
    async def test_prefill_eviction_skips_active_pinned_loading_and_current(self):
        gb = 1024**3
        pool = _make_pool(ceiling=0)
        pool._entries = {
            "active": self._entry("active", 20 * gb, active=True),
            "pinned": self._entry("pinned", 20 * gb),
            "loading": self._entry("loading", 20 * gb),
            "target": self._entry("target", 25 * gb),
        }
        pool._entries["pinned"].is_pinned = True
        pool._entries["loading"].is_loading = True
        pool._current_model_memory = 85 * gb
        pool._unload_engine = AsyncMock()
        req = PrefillEvictionRequest(
            request_id="req-1",
            model_id="target",
            current_bytes=85 * gb,
            target_cap_bytes=40 * gb,
            predicted_transient_bytes=10 * gb,
            requested_tokens=2048,
            reason="adaptive_prefill_throttle",
        )

        with (
            patch("omlx.engine_pool.mx.get_active_memory", return_value=0),
            patch("omlx.engine_pool.get_phys_footprint", return_value=0),
        ):
            evicted = await pool._evict_idle_lru_for_prefill("target", req)

        assert evicted is False
        pool._unload_engine.assert_not_awaited()


class TestEnginePoolStatus:
    """Tests for get_status is_loading field."""

    def test_get_status_includes_is_loading(self, small_mock_model_dir):
        """Test get_status includes is_loading field."""
        pool = _make_pool(ceiling=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        status = pool.get_status()
        for model in status["models"]:
            assert "is_loading" in model
            assert model["is_loading"] is False


class TestEnginePoolTTL:
    """Tests for TTL expiration checking."""

    @pytest.fixture
    def pool_with_loaded_model(self, small_mock_model_dir):
        """Create pool with a mock-loaded model."""
        pool = _make_pool(ceiling=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        # Mock-load model-a
        entry = pool._entries["model-a"]
        entry.engine = MagicMock(spec=[])
        entry.engine.stop = AsyncMock()
        entry.engine.has_active_requests = MagicMock(return_value=False)
        entry.last_access = 100.0  # Old access time
        pool._current_model_memory = entry.estimated_size
        return pool

    @pytest.mark.asyncio
    async def test_ttl_expires_idle_model(self, pool_with_loaded_model):
        """Test that TTL unloads idle model."""
        pool = pool_with_loaded_model
        settings_manager = MagicMock()
        settings = MagicMock()
        settings.ttl_seconds = 60
        settings_manager.get_settings.return_value = settings

        with patch("time.time", return_value=200.0):  # 100s idle > 60s TTL
            expired = await pool.check_ttl_expirations(settings_manager)

        assert "model-a" in expired
        assert pool._entries["model-a"].engine is None

    @pytest.mark.asyncio
    async def test_ttl_skips_model_within_ttl(self, pool_with_loaded_model):
        """Test that TTL does not unload model within TTL."""
        pool = pool_with_loaded_model
        settings_manager = MagicMock()
        settings = MagicMock()
        settings.ttl_seconds = 300
        settings_manager.get_settings.return_value = settings

        with patch("time.time", return_value=200.0):  # 100s idle < 300s TTL
            expired = await pool.check_ttl_expirations(settings_manager)

        assert expired == []
        assert pool._entries["model-a"].engine is not None

    @pytest.mark.asyncio
    async def test_ttl_skips_no_ttl_model(self, pool_with_loaded_model):
        """Test that models without TTL are not unloaded."""
        pool = pool_with_loaded_model
        settings_manager = MagicMock()
        settings = MagicMock()
        settings.ttl_seconds = None
        settings_manager.get_settings.return_value = settings

        with patch("time.time", return_value=99999.0):
            expired = await pool.check_ttl_expirations(settings_manager)

        assert expired == []
        assert pool._entries["model-a"].engine is not None

    @pytest.mark.asyncio
    async def test_ttl_skips_pinned_model(self, pool_with_loaded_model):
        """Test that pinned models are not unloaded by TTL."""
        pool = pool_with_loaded_model
        pool._entries["model-a"].is_pinned = True

        settings_manager = MagicMock()
        settings = MagicMock()
        settings.ttl_seconds = 60
        settings_manager.get_settings.return_value = settings

        with patch("time.time", return_value=200.0):
            expired = await pool.check_ttl_expirations(settings_manager)

        assert expired == []
        assert pool._entries["model-a"].engine is not None

    @pytest.mark.asyncio
    async def test_ttl_skips_model_with_active_requests(self, pool_with_loaded_model):
        """Test that TTL does not unload model with active requests."""
        pool = pool_with_loaded_model

        # Mock an engine that reports active requests
        mock_engine = MagicMock()
        mock_engine.has_active_requests.return_value = True

        pool._entries["model-a"].engine = mock_engine

        settings_manager = MagicMock()
        settings = MagicMock()
        settings.ttl_seconds = 60
        settings_manager.get_settings.return_value = settings

        with patch("time.time", return_value=200.0):
            expired = await pool.check_ttl_expirations(settings_manager)

        assert expired == []
        # last_access should be refreshed
        assert pool._entries["model-a"].last_access == 200.0

    @pytest.mark.asyncio
    async def test_ttl_skips_vlm_with_active_requests(self, pool_with_loaded_model):
        """Test that TTL does not unload VLM engine with active requests."""
        pool = pool_with_loaded_model

        mock_engine = MagicMock()
        mock_engine.has_active_requests.return_value = True

        pool._entries["model-a"].engine = mock_engine

        settings_manager = MagicMock()
        settings = MagicMock()
        settings.ttl_seconds = 60
        settings_manager.get_settings.return_value = settings

        with patch("time.time", return_value=200.0):
            expired = await pool.check_ttl_expirations(settings_manager)

        assert expired == []
        assert pool._entries["model-a"].last_access == 200.0

    @pytest.mark.asyncio
    async def test_ttl_skips_non_streaming_with_active_requests(
        self, pool_with_loaded_model
    ):
        """Test that TTL does not unload non-streaming engine with active requests."""
        pool = pool_with_loaded_model

        mock_engine = MagicMock()
        mock_engine.has_active_requests.return_value = True

        pool._entries["model-a"].engine = mock_engine

        settings_manager = MagicMock()
        settings = MagicMock()
        settings.ttl_seconds = 60
        settings_manager.get_settings.return_value = settings

        with patch("time.time", return_value=200.0):
            expired = await pool.check_ttl_expirations(settings_manager)

        assert expired == []
        assert pool._entries["model-a"].last_access == 200.0

    @pytest.mark.asyncio
    async def test_ttl_falls_back_to_global_idle_timeout(self, pool_with_loaded_model):
        """Per-model TTL None falls back to global idle timeout."""
        pool = pool_with_loaded_model
        settings_manager = MagicMock()
        settings = MagicMock()
        settings.ttl_seconds = None
        settings_manager.get_settings.return_value = settings

        with patch("time.time", return_value=200.0):  # 100s idle > 60s global
            expired = await pool.check_ttl_expirations(
                settings_manager, global_idle_timeout_seconds=60
            )

        assert "model-a" in expired
        assert pool._entries["model-a"].engine is None

    @pytest.mark.asyncio
    async def test_ttl_global_disabled_when_none(self, pool_with_loaded_model):
        """Per-model None + global None keeps model loaded."""
        pool = pool_with_loaded_model
        settings_manager = MagicMock()
        settings = MagicMock()
        settings.ttl_seconds = None
        settings_manager.get_settings.return_value = settings

        with patch("time.time", return_value=99999.0):
            expired = await pool.check_ttl_expirations(
                settings_manager, global_idle_timeout_seconds=None
            )

        assert expired == []
        assert pool._entries["model-a"].engine is not None

    @pytest.mark.asyncio
    async def test_per_model_ttl_overrides_global(self, pool_with_loaded_model):
        """Per-model TTL wins over global idle timeout."""
        pool = pool_with_loaded_model
        settings_manager = MagicMock()
        settings = MagicMock()
        settings.ttl_seconds = 300  # per-model wider than global
        settings_manager.get_settings.return_value = settings

        with patch("time.time", return_value=200.0):  # 100s idle < 300s per-model
            expired = await pool.check_ttl_expirations(
                settings_manager, global_idle_timeout_seconds=60
            )

        assert expired == []
        assert pool._entries["model-a"].engine is not None


class TestHasActiveRequests:
    """Tests for has_active_requests() on engine types."""

    def test_base_non_streaming_engine_active_count(self):
        """Test BaseNonStreamingEngine active request tracking."""
        from omlx.engine.base import BaseNonStreamingEngine

        class DummyEngine(BaseNonStreamingEngine):
            @property
            def model_name(self):
                return "dummy"

            async def start(self):
                pass

            async def stop(self):
                pass

            def get_stats(self):
                return {}

        engine = DummyEngine()
        assert engine.has_active_requests() is False

        with engine._active_lock:
            engine._active_count += 1
        assert engine.has_active_requests() is True

        with engine._active_lock:
            engine._active_count -= 1
        assert engine.has_active_requests() is False

    def test_batched_engine_has_active_requests(self):
        """Test BatchedEngine.has_active_requests() via _output_collectors."""
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._engine = None
        assert engine.has_active_requests() is False

        # Simulate engine with active collectors
        mock_engine_core = MagicMock()
        mock_inner = MagicMock()
        mock_inner._output_collectors = {"req1": MagicMock()}
        mock_engine_core.engine = mock_inner
        engine._engine = mock_engine_core
        assert engine.has_active_requests() is True

        # Empty collectors
        mock_inner._output_collectors = {}
        assert engine.has_active_requests() is False

    def test_vlm_engine_has_active_requests(self):
        """Test VLMBatchedEngine.has_active_requests() via _output_collectors."""
        from omlx.engine.vlm import VLMBatchedEngine

        engine = VLMBatchedEngine.__new__(VLMBatchedEngine)
        engine._engine = None
        assert engine.has_active_requests() is False

        mock_engine_core = MagicMock()
        mock_inner = MagicMock()
        mock_inner._output_collectors = {"req1": MagicMock()}
        mock_engine_core.engine = mock_inner
        engine._engine = mock_engine_core
        assert engine.has_active_requests() is True


class TestResolveModelId:
    """Tests for resolve_model_id method."""

    def test_direct_match(self, small_mock_model_dir):
        """Test direct model_id match returns immediately."""
        pool = _make_pool(ceiling=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        result = pool.resolve_model_id("model-a", settings_manager=None)
        assert result == "model-a"

    def test_alias_match(self, small_mock_model_dir):
        """Test alias resolution returns real model_id."""
        pool = _make_pool(ceiling=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        settings_manager = MagicMock()
        settings_manager.get_exposed_profile_source_model_id.return_value = None
        from omlx.model_settings import ModelSettings

        settings_manager.get_all_settings.return_value = {
            "model-a": ModelSettings(model_alias="gpt-4"),
            "model-b": ModelSettings(),
        }

        result = pool.resolve_model_id("gpt-4", settings_manager)
        assert result == "model-a"

    def test_no_match_returns_original(self, small_mock_model_dir):
        """Test unresolved name returns original string."""
        pool = _make_pool(ceiling=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        settings_manager = MagicMock()
        settings_manager.get_exposed_profile_source_model_id.return_value = None
        from omlx.model_settings import ModelSettings

        settings_manager.get_all_settings.return_value = {
            "model-a": ModelSettings(),
        }

        result = pool.resolve_model_id("nonexistent", settings_manager)
        assert result == "nonexistent"

    def test_alias_match_no_settings_manager(self, small_mock_model_dir):
        """Test with None settings_manager falls back to direct match only."""
        pool = _make_pool(ceiling=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        result = pool.resolve_model_id("some-alias", settings_manager=None)
        assert result == "some-alias"

    def test_provider_prefix_alias_match(self, small_mock_model_dir):
        """Test alias resolution with provider prefix (e.g. omlx/alias)."""
        pool = _make_pool(ceiling=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        settings_manager = MagicMock()
        settings_manager.get_exposed_profile_source_model_id.return_value = None
        from omlx.model_settings import ModelSettings

        settings_manager.get_all_settings.return_value = {
            "model-a": ModelSettings(model_alias="gpt-4"),
            "model-b": ModelSettings(),
        }

        result = pool.resolve_model_id("omlx/gpt-4", settings_manager)
        assert result == "model-a"

    def test_provider_prefix_direct_match(self, small_mock_model_dir):
        """Test direct match with provider prefix (e.g. provider/model-a)."""
        pool = _make_pool(ceiling=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        result = pool.resolve_model_id("provider/model-a", settings_manager=None)
        assert result == "model-a"

    def test_provider_prefix_no_match(self, small_mock_model_dir):
        """Test prefix strip still returns original when no match found."""
        pool = _make_pool(ceiling=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        settings_manager = MagicMock()
        settings_manager.get_exposed_profile_source_model_id.return_value = None
        from omlx.model_settings import ModelSettings

        settings_manager.get_all_settings.return_value = {
            "model-a": ModelSettings(),
        }

        result = pool.resolve_model_id("omlx/nonexistent", settings_manager)
        assert result == "omlx/nonexistent"

    def test_case_insensitive_match(self, small_mock_model_dir):
        """Test case-insensitive fallback when exact match fails."""
        pool = _make_pool(ceiling=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        result = pool.resolve_model_id("MODEL-A", settings_manager=None)
        assert result == "model-a"

    def test_case_insensitive_with_provider_prefix(self, small_mock_model_dir):
        """Test case-insensitive match after stripping provider prefix."""
        pool = _make_pool(ceiling=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        result = pool.resolve_model_id("omlx/MODEL-B", settings_manager=None)
        assert result == "model-b"

    def test_exact_match_preferred_over_case_insensitive(self, small_mock_model_dir):
        """Test exact match takes priority over case-insensitive."""
        pool = _make_pool(ceiling=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        # Exact match should be returned directly
        result = pool.resolve_model_id("model-a", settings_manager=None)
        assert result == "model-a"


class TestMemorySettleBarrier:
    """Tests for memory settle barrier in _unload_engine()."""

    @pytest.fixture
    def pool_with_loaded_model(self, small_mock_model_dir):
        """Create pool with a mock-loaded model for settle barrier testing.

        Sets estimated_size to 5GB. With scaled tolerance
        (max(2GB, 5% of 5GB) = max(2GB, 0.25GB) = 2GB), the barrier
        requires at least 3GB freed.
        """
        pool = _make_pool(ceiling=100 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        entry = pool._entries["model-a"]
        entry.estimated_size = 5 * 1024**3  # 5GB (> 2GB tolerance)
        mock_engine = MagicMock()
        mock_engine.stop = AsyncMock()
        mock_engine.has_active_requests = MagicMock(return_value=False)
        entry.engine = mock_engine
        entry.last_access = 100.0
        pool._current_model_memory = entry.estimated_size
        return pool

    @pytest.mark.asyncio
    async def test_settle_succeeds_first_round(self, pool_with_loaded_model):
        """Test that settle barrier passes on first round when memory is freed."""
        pool = pool_with_loaded_model
        est_size = pool._entries["model-a"].estimated_size  # 5GB
        initial_memory = pool._current_model_memory

        # Pre-unload: 10GB active. After GC: drops to 5GB (5GB freed >= 3GB needed).
        active_memory_values = [10 * 1024**3, 5 * 1024**3]
        call_idx = [0]

        def mock_get_active():
            val = active_memory_values[min(call_idx[0], len(active_memory_values) - 1)]
            call_idx[0] += 1
            return val

        with (
            patch("omlx.engine_pool.mx") as mock_mx,
            patch("omlx.engine_pool.get_mlx_executor", return_value=None),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_mx.get_active_memory = mock_get_active
            mock_mx.synchronize = MagicMock()
            mock_mx.clear_cache = MagicMock()

            await pool._unload_engine("model-a")

        assert pool._entries["model-a"].engine is None
        assert pool._current_model_memory == initial_memory - est_size

    @pytest.mark.asyncio
    async def test_settle_takes_multiple_rounds(self, pool_with_loaded_model):
        """Test settle barrier succeeds after multiple rounds of GC."""
        pool = pool_with_loaded_model
        # est_size = 5GB, tolerance = 2GB, so need >= 3GB freed
        # Pre-unload: 10GB. Need active <= 7GB to settle.
        # Round 1: 9GB (freed=1GB < 3GB), Round 2: 8GB (freed=2GB < 3GB),
        # Round 3: 5GB (freed=5GB >= 3GB) → settled
        active_memory_values = [10 * 1024**3, 9 * 1024**3, 8 * 1024**3, 5 * 1024**3]
        call_idx = [0]

        def mock_get_active():
            val = active_memory_values[min(call_idx[0], len(active_memory_values) - 1)]
            call_idx[0] += 1
            return val

        sleep_calls = []

        async def mock_sleep(duration):
            sleep_calls.append(duration)

        with (
            patch("omlx.engine_pool.mx") as mock_mx,
            patch("omlx.engine_pool.get_mlx_executor", return_value=None),
            patch("asyncio.sleep", side_effect=mock_sleep),
        ):
            mock_mx.get_active_memory = mock_get_active
            mock_mx.synchronize = MagicMock()
            mock_mx.clear_cache = MagicMock()

            await pool._unload_engine("model-a")

        # Should have slept at least once (0.5s between rounds)
        assert any(d == 0.5 for d in sleep_calls)
        assert pool._entries["model-a"].engine is None
        assert pool._current_model_memory == 0

    @pytest.mark.asyncio
    async def test_settle_timeout_triggers_emergency(self, pool_with_loaded_model):
        """Test emergency reclaim is triggered when settle barrier times out."""
        pool = pool_with_loaded_model
        # est_size = 5GB, tolerance = 2GB, need >= 3GB freed
        # Memory stays at 10GB during all 10 settle rounds (0GB freed < 3GB)
        # After emergency reclaim: drops to safe level
        settle_calls = [0]

        def mock_get_active():
            settle_calls[0] += 1
            # 1 pre-unload + 10 settle rounds (each calls once) = 11
            # After emergency: return safe level
            if settle_calls[0] <= 11:
                return 10 * 1024**3
            return 0

        sleep_calls = []

        async def mock_sleep(duration):
            sleep_calls.append(duration)

        with (
            patch("omlx.engine_pool.mx") as mock_mx,
            patch("omlx.engine_pool.get_mlx_executor", return_value=None),
            patch("asyncio.sleep", side_effect=mock_sleep),
        ):
            mock_mx.get_active_memory = mock_get_active
            mock_mx.synchronize = MagicMock()
            mock_mx.clear_cache = MagicMock()

            await pool._unload_engine("model-a")

        # Emergency reclaim uses 1.0s sleeps (3 rounds)
        assert sleep_calls.count(1.0) == 3
        assert pool._entries["model-a"].engine is None

    @pytest.mark.asyncio
    async def test_emergency_reclaim_failure_logs_error(self, pool_with_loaded_model):
        """Test error is logged when emergency reclaim fails to free enough memory."""
        pool = pool_with_loaded_model

        # Memory never drops — stays at 10GB throughout (well above 5GB threshold)
        with (
            patch("omlx.engine_pool.mx") as mock_mx,
            patch("omlx.engine_pool.get_mlx_executor", return_value=None),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_mx.get_active_memory = MagicMock(return_value=10 * 1024**3)
            mock_mx.synchronize = MagicMock()
            mock_mx.clear_cache = MagicMock()

            with patch("omlx.engine_pool.logger") as mock_logger:
                await pool._unload_engine("model-a")

            # Should have logged an error about emergency reclaim failure
            error_calls = [str(c) for c in mock_logger.error.call_args_list]
            assert any("Emergency reclaim failed" in s for s in error_calls)

    @pytest.mark.asyncio
    async def test_memory_counter_decremented_after_barrier(
        self, pool_with_loaded_model
    ):
        """Regression test: _current_model_memory must not be decremented
        before the settle barrier completes."""
        pool = pool_with_loaded_model
        est_size = pool._entries["model-a"].estimated_size  # 5GB
        original_memory = pool._current_model_memory

        memory_during_settle = []

        def mock_get_active():
            # Record the pool's memory counter state during settle polling
            memory_during_settle.append(pool._current_model_memory)
            # Return high value for 3 rounds, then settle
            # Need >= 3GB freed (5GB - 2GB tolerance)
            if len(memory_during_settle) <= 3:
                return 10 * 1024**3  # 0GB freed
            return 5 * 1024**3  # 5GB freed >= 3GB needed

        with (
            patch("omlx.engine_pool.mx") as mock_mx,
            patch("omlx.engine_pool.get_mlx_executor", return_value=None),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_mx.get_active_memory = mock_get_active
            mock_mx.synchronize = MagicMock()
            mock_mx.clear_cache = MagicMock()

            await pool._unload_engine("model-a")

        # During settle barrier, memory counter should NOT have been decremented
        for mem in memory_during_settle[:-1]:
            assert mem == original_memory, (
                f"Memory counter was {mem} during settle, expected {original_memory}. "
                "Counter must not be decremented before barrier completes."
            )

        # After barrier, it should be decremented
        assert pool._current_model_memory == original_memory - est_size

    @pytest.mark.asyncio
    async def test_settle_large_model_proportional_tolerance(
        self, small_mock_model_dir
    ):
        """Test that settle tolerance scales with model size for large models.

        For a 60GB model, 5% = 3GB > 2GB floor, so tolerance = 3GB.
        min_expected_freed = 60GB - 3GB = 57GB.
        """
        pool = _make_pool(ceiling=200 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        entry = pool._entries["model-a"]
        entry.estimated_size = 60 * 1024**3  # 60GB
        mock_engine = MagicMock()
        mock_engine.stop = AsyncMock()
        mock_engine.has_active_requests = MagicMock(return_value=False)
        entry.engine = mock_engine
        entry.last_access = 100.0
        pool._current_model_memory = entry.estimated_size

        # Freed = 80 - 23 = 57GB. With proportional tolerance (3GB) this
        # settles, but would fail with the old fixed 2GB tolerance (needed 58GB).
        active_memory_values = [80 * 1024**3, 23 * 1024**3]
        call_idx = [0]

        def mock_get_active():
            val = active_memory_values[min(call_idx[0], len(active_memory_values) - 1)]
            call_idx[0] += 1
            return val

        with (
            patch("omlx.engine_pool.mx") as mock_mx,
            patch("omlx.engine_pool.get_mlx_executor", return_value=None),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_mx.get_active_memory = mock_get_active
            mock_mx.synchronize = MagicMock()
            mock_mx.clear_cache = MagicMock()

            await pool._unload_engine("model-a")

        assert pool._entries["model-a"].engine is None
        assert pool._current_model_memory == 0

    @pytest.mark.asyncio
    async def test_settle_small_model_uses_floor_tolerance(self, small_mock_model_dir):
        """Test that 2GB floor tolerance applies for small models.

        For a 1GB model, 5% = 0.05GB << 2GB, so tolerance = 2GB floor.
        min_expected_freed = max(0, 1GB - 2GB) = 0, settle is trivially true.
        """
        pool = _make_pool(ceiling=100 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        entry = pool._entries["model-a"]
        entry.estimated_size = 1 * 1024**3  # 1GB
        mock_engine = MagicMock()
        mock_engine.stop = AsyncMock()
        mock_engine.has_active_requests = MagicMock(return_value=False)
        entry.engine = mock_engine
        entry.last_access = 100.0
        pool._current_model_memory = entry.estimated_size

        # Even 0 freed should settle (min_expected_freed = 0)
        active_memory_values = [10 * 1024**3, 10 * 1024**3]
        call_idx = [0]

        def mock_get_active():
            val = active_memory_values[min(call_idx[0], len(active_memory_values) - 1)]
            call_idx[0] += 1
            return val

        with (
            patch("omlx.engine_pool.mx") as mock_mx,
            patch("omlx.engine_pool.get_mlx_executor", return_value=None),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_mx.get_active_memory = mock_get_active
            mock_mx.synchronize = MagicMock()
            mock_mx.clear_cache = MagicMock()

            await pool._unload_engine("model-a")

        assert pool._entries["model-a"].engine is None
        assert pool._current_model_memory == 0

    @pytest.mark.asyncio
    async def test_settle_bails_out_under_concurrent_activity(
        self, pool_with_loaded_model, caplog
    ):
        """1774 regression: with another engine serving, the global freed
        delta is unmeasurable (it can read negative as the other engine
        allocates). The barrier must bail after one sample instead of burning
        10 settle rounds + emergency reclaim — ~8s of gc/synchronize/
        clear_cache serialized against live decode, under the pool lock.
        """
        pool = pool_with_loaded_model

        # Second entry actively serving.
        other_engine = MagicMock()
        other_engine.has_active_requests = MagicMock(return_value=True)
        pool._entries["model-b"].engine = other_engine

        # Global gauge RISES during settle (concurrent prefill/KV growth),
        # so freed = pre_unload - active_now is negative every round.
        call_idx = [0]

        def rising_gauge():
            val = (10 + call_idx[0]) * 1024**3
            call_idx[0] += 1
            return val

        sleep_calls: list[float] = []

        async def record_sleep(duration, *args, **kwargs):
            sleep_calls.append(duration)

        with (
            patch("omlx.engine_pool.mx") as mock_mx,
            patch("omlx.engine_pool.get_mlx_executor", return_value=None),
            patch("asyncio.sleep", side_effect=record_sleep),
            caplog.at_level(logging.DEBUG, logger="omlx.engine_pool"),
        ):
            mock_mx.get_active_memory = rising_gauge
            mock_mx.synchronize = MagicMock()
            mock_mx.clear_cache = MagicMock()

            await pool._unload_engine("model-a")

        assert "indeterminate under concurrent activity" in caplog.text
        # No settle-round burn, no timeout warning, no emergency reclaim.
        assert sleep_calls.count(0.5) == 0
        assert "Settle barrier timed out" not in caplog.text
        assert "Emergency reclaim" not in caplog.text
        # Only the initial pre-barrier release cycle touched the executor.
        assert mock_mx.synchronize.call_count == 1
        assert mock_mx.clear_cache.call_count == 1
        # The unload itself still completes and is accounted.
        assert pool._entries["model-a"].engine is None
        assert pool._current_model_memory == 0

    @pytest.mark.asyncio
    async def test_settle_still_waits_when_pool_otherwise_idle(
        self, pool_with_loaded_model, caplog
    ):
        """Idle-pool behavior is unchanged (#768 protection): with no other
        entry serving, an unsatisfied barrier still burns its settle rounds
        and escalates to emergency reclaim.
        """
        pool = pool_with_loaded_model

        call_idx = [0]

        def rising_gauge():
            val = (10 + call_idx[0]) * 1024**3
            call_idx[0] += 1
            return val

        with (
            patch("omlx.engine_pool.mx") as mock_mx,
            patch("omlx.engine_pool.get_mlx_executor", return_value=None),
            patch("asyncio.sleep", new_callable=AsyncMock),
            caplog.at_level(logging.DEBUG, logger="omlx.engine_pool"),
        ):
            mock_mx.get_active_memory = rising_gauge
            mock_mx.synchronize = MagicMock()
            mock_mx.clear_cache = MagicMock()

            await pool._unload_engine("model-a")

        assert "indeterminate under concurrent activity" not in caplog.text
        assert "Settle barrier timed out" in caplog.text
        # Full barrier behavior preserved: 1 initial release cycle + 10 settle
        # rounds + 3 emergency-reclaim rounds on the executor.
        assert mock_mx.synchronize.call_count == 14
        assert mock_mx.clear_cache.call_count == 14

    def test_other_entries_serving_in_use_lease_counts(self, pool_with_loaded_model):
        """The in-use lease (acquired but not yet active) also marks the pool
        as serving — eviction paths already treat it as activity (#1667).
        """
        pool = pool_with_loaded_model
        entry_b = pool._entries["model-b"]

        assert pool._other_entries_serving("model-a") is False

        entry_b.engine = MagicMock()
        entry_b.engine.has_active_requests = MagicMock(return_value=False)
        assert pool._other_entries_serving("model-a") is False

        entry_b.in_use = 1
        assert pool._other_entries_serving("model-a") is True


class TestEnginePoolInUseLease:
    """Tests for the acquire-vs-use in-use lease (#1667).

    The memory enforcer must never evict an engine that a request has
    acquired but not yet registered activity on. The lease is taken under the
    pool lock at acquire time and skipped by every eviction path, in addition
    to the existing has_active_requests() gate.
    """

    @staticmethod
    def _loaded_entry(model_id: str, last_access: float = 0.0) -> EngineEntry:
        engine = MagicMock()
        engine.has_active_requests.return_value = False  # looks idle
        engine.stop = AsyncMock()
        # Mirror TestEnginePoolPrefillEviction: no scheduler → prefill-idle.
        engine.scheduler = None
        engine._engine = None
        return EngineEntry(
            model_id=model_id,
            model_path=f"/models/{model_id}",
            model_type="embedding",
            engine_type="embedding",
            estimated_size=1024,
            engine=engine,
            last_access=last_access,
        )

    def test_engine_entry_in_use_default_zero(self):
        """EngineEntry exposes an in_use lease counter defaulting to 0."""
        entry = EngineEntry(
            model_id="x",
            model_path="/p",
            model_type="embedding",
            engine_type="embedding",
            estimated_size=1,
        )
        assert entry.in_use == 0

    def test_find_lru_victim_skips_in_use_engine(self):
        """An idle-looking engine with in_use > 0 must not be an LRU victim."""
        pool = _make_pool(ceiling=0)
        pool._entries = {
            "leased": self._loaded_entry("leased", last_access=1.0),  # oldest
            "free": self._loaded_entry("free", last_access=2.0),
        }
        pool._entries["leased"].in_use = 1

        victim = pool._find_lru_victim()
        # Without the lease the older "leased" would be picked; lease forces "free".
        assert victim == "free"

    def test_find_lru_victim_none_when_only_in_use(self):
        """If the only loaded engine is leased, there is no LRU victim."""
        pool = _make_pool(ceiling=0)
        pool._entries = {"leased": self._loaded_entry("leased", last_access=1.0)}
        pool._entries["leased"].in_use = 2

        assert pool._find_lru_victim() is None

    def test_is_idle_for_prefill_eviction_skips_in_use(self):
        """Prefill-pressure eviction must skip a leased (in_use > 0) engine."""
        pool = _make_pool(ceiling=0)
        entry = self._loaded_entry("leased")
        # Idle-looking → would be evictable, but the lease blocks it.
        assert pool._is_idle_for_prefill_eviction(entry) is True
        entry.in_use = 1
        assert pool._is_idle_for_prefill_eviction(entry) is False

    @pytest.mark.asyncio
    async def test_check_ttl_expirations_skips_in_use(self):
        """TTL eviction must not unload a leased engine even when idle past TTL."""
        pool = _make_pool(ceiling=0)
        entry = self._loaded_entry("leased", last_access=100.0)
        entry.in_use = 1
        pool._entries = {"leased": entry}

        settings_manager = MagicMock()
        settings = MagicMock()
        settings.ttl_seconds = 60
        settings_manager.get_settings.return_value = settings

        with patch("time.time", return_value=200.0):  # 100s idle > 60s TTL
            expired = await pool.check_ttl_expirations(settings_manager)

        assert expired == []
        assert pool._entries["leased"].engine is not None
        # in_use counts as activity → last_access refreshed
        assert pool._entries["leased"].last_access == 200.0

    @pytest.mark.asyncio
    async def test_release_engine_floors_at_zero(self):
        """release_engine decrements under lock and never goes below 0."""
        pool = _make_pool(ceiling=0)
        entry = self._loaded_entry("leased")
        entry.in_use = 1
        pool._entries = {"leased": entry}

        await pool.release_engine("leased")
        assert entry.in_use == 0
        # Extra release is a no-op (floor at 0), not a negative count.
        await pool.release_engine("leased")
        assert entry.in_use == 0
        # Unknown model id is a harmless no-op.
        await pool.release_engine("nope")

    @pytest.mark.asyncio
    async def test_release_engine_unloads_pending_after_lease_drains(self):
        """Pending hard-pressure unload runs as soon as the lease drains."""
        pool = _make_pool(ceiling=0)
        entry = self._loaded_entry("leased")
        entry.in_use = 1
        entry.pending_unload_reason = "hard memory pressure"
        entry.abort_requested = True
        pool._entries = {"leased": entry}
        pool._unload_engine = AsyncMock()

        await pool.release_engine("leased")

        assert entry.in_use == 0
        assert entry.pending_unload_reason is None
        assert entry.abort_requested is False
        pool._unload_engine.assert_awaited_once_with("leased")

    @pytest.mark.asyncio
    async def test_release_engine_keeps_pending_while_scheduler_active(self):
        """A drained lease is not enough if scheduler requests are still active."""
        pool = _make_pool(ceiling=0)
        entry = self._loaded_entry("leased")
        entry.in_use = 1
        entry.engine.has_active_requests.return_value = True
        entry.pending_unload_reason = "hard memory pressure"
        entry.abort_requested = True
        pool._entries = {"leased": entry}
        pool._unload_engine = AsyncMock()

        await pool.release_engine("leased")

        assert entry.in_use == 0
        assert entry.pending_unload_reason == "hard memory pressure"
        assert entry.abort_requested is True
        pool._unload_engine.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_acquire_leases_then_releases_on_success(self):
        """acquire() leases on enter and releases in finally on normal exit."""
        pool = _make_pool(ceiling=0)
        entry = self._loaded_entry("model-a")
        pool._entries = {"model-a": entry}

        async with pool.acquire("model-a") as engine:
            assert engine is entry.engine
            assert entry.in_use == 1  # leased while in use
        assert entry.in_use == 0  # released on exit

    @pytest.mark.asyncio
    async def test_acquire_releases_lease_on_exception(self):
        """acquire() releases the lease even if the body raises."""
        pool = _make_pool(ceiling=0)
        entry = self._loaded_entry("model-a")
        pool._entries = {"model-a": entry}

        with pytest.raises(RuntimeError, match="boom"):
            async with pool.acquire("model-a"):
                assert entry.in_use == 1
                raise RuntimeError("boom")
        assert entry.in_use == 0  # no leaked lease


class TestResetActivityTracking:
    """Tests for BaseNonStreamingEngine._reset_activity_tracking (#1595)."""

    @staticmethod
    def _engine():
        from omlx.engine.base import BaseNonStreamingEngine

        class DummyEngine(BaseNonStreamingEngine):
            @property
            def model_name(self):
                return "dummy"

            async def start(self):
                pass

            async def stop(self):
                pass

            def get_stats(self):
                return {}

        return DummyEngine()

    def test_reset_clears_leaked_counter_and_activities(self):
        """Reset zeros a leaked _active_count and clears _activities.

        Mirrors the immediate-abort eviction case where stop() runs without
        the per-request completion callbacks, leaving a phantom busy count.
        """
        engine = self._engine()
        with engine._active_lock:
            engine._active_count = 2
            engine._activities = {"a": {}, "b": {}}
        assert engine.has_active_requests() is True  # phantom busy

        engine._reset_activity_tracking()

        assert engine._active_count == 0
        assert engine._activities == {}
        assert engine.has_active_requests() is False

    @pytest.mark.asyncio
    async def test_unload_engine_calls_reset_on_teardown(self):
        """EnginePool._unload_engine resets activity tracking after stop()."""
        pool = _make_pool(ceiling=0)
        engine = self._engine()
        with engine._active_lock:
            engine._active_count = 1  # leaked phantom busy
        entry = EngineEntry(
            model_id="model-a",
            model_path="/models/model-a",
            model_type="embedding",
            engine_type="embedding",
            estimated_size=1024,
            engine=engine,
            last_access=0.0,
        )
        pool._entries = {"model-a": entry}
        pool._current_model_memory = entry.estimated_size

        with (
            patch("omlx.engine_pool.mx") as mock_mx,
            patch("omlx.engine_pool.get_mlx_executor", return_value=None),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_mx.get_active_memory = MagicMock(return_value=0)
            mock_mx.synchronize = MagicMock()
            mock_mx.clear_cache = MagicMock()

            await pool._unload_engine("model-a")

        # Teardown reset the leaked counter (getattr-guarded reset called).
        assert engine._active_count == 0
        assert engine.has_active_requests() is False
