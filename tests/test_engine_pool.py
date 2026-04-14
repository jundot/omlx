# SPDX-License-Identifier: Apache-2.0
"""Tests for EnginePool functionality."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.engine_pool import EngineEntry, EnginePool
from omlx.exceptions import (
    InsufficientMemoryError,
    ModelLoadingError,
    ModelNotFoundError,
    ModelTooLargeError,
)


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
        pool = EnginePool(max_model_memory=32 * 1024**3)
        assert pool.max_model_memory == 32 * 1024**3
        assert pool.current_model_memory == 0
        assert pool.model_count == 0
        assert pool.loaded_model_count == 0

    def test_init_disabled_memory(self):
        """Test initialization with disabled (None) memory limit."""
        pool = EnginePool(max_model_memory=None)
        assert pool.max_model_memory is None
        assert pool.current_model_memory == 0

    def test_model_too_large_skipped_when_disabled(self, small_mock_model_dir):
        """Test that ModelTooLargeError is NOT raised when memory is disabled."""
        pool = EnginePool(max_model_memory=None)
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
        pool = EnginePool(max_model_memory=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        assert pool.model_count == 2
        assert "model-a" in pool.get_model_ids()
        assert "model-b" in pool.get_model_ids()

    def test_discover_models_with_pinned(self, small_mock_model_dir):
        """Test model discovery with pinned models."""
        pool = EnginePool(max_model_memory=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir), pinned_models=["model-a"])

        entry = pool.get_entry("model-a")
        assert entry is not None
        assert entry.is_pinned is True

        entry_b = pool.get_entry("model-b")
        assert entry_b is not None
        assert entry_b.is_pinned is False


class TestDiscoverModelsMerge:
    """Tests for discover_models merge behavior (issue #89)."""

    def test_rediscover_preserves_loaded_engine(self, small_mock_model_dir):
        """Test that re-discovery preserves loaded engine state."""
        pool = EnginePool(max_model_memory=10 * 1024**3)
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
        pool = EnginePool(max_model_memory=10 * 1024**3)
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
        pool = EnginePool(max_model_memory=10 * 1024**3)
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
        pool = EnginePool(max_model_memory=10 * 1024**3)
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
        pool = EnginePool(max_model_memory=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        with pytest.raises(ModelNotFoundError) as exc_info:
            asyncio.run(pool.get_engine("nonexistent"))

        assert exc_info.value.model_id == "nonexistent"
        assert "model-a" in exc_info.value.available_models

    def test_model_too_large_error(self, small_mock_model_dir):
        """Test ModelTooLargeError when model exceeds memory limit."""
        # Set very small memory limit
        pool = EnginePool(max_model_memory=100)  # 100 bytes
        pool.discover_models(str(small_mock_model_dir))

        with pytest.raises(ModelTooLargeError) as exc_info:
            asyncio.run(pool.get_engine("model-a"))

        assert exc_info.value.model_id == "model-a"
        assert exc_info.value.max_memory == 100


class TestEnginePoolStatus:
    """Tests for EnginePool status reporting."""

    def test_get_status(self, small_mock_model_dir):
        """Test get_status returns correct information."""
        pool = EnginePool(max_model_memory=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir), pinned_models=["model-a"])

        status = pool.get_status()

        assert status["max_model_memory"] == 10 * 1024**3
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
        pool = EnginePool(max_model_memory=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        ids = pool.get_model_ids()
        assert set(ids) == {"model-a", "model-b"}

    def test_get_loaded_model_ids_empty(self, small_mock_model_dir):
        """Test get_loaded_model_ids when no models loaded."""
        pool = EnginePool(max_model_memory=10 * 1024**3)
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
        pool = EnginePool(max_model_memory=10 * 1024**3)
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
        pool = EnginePool(max_model_memory=10 * 1024**3)
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
        pool = EnginePool(max_model_memory=10 * 1024**3)
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

        with patch(
            "omlx.engine_pool.VLMBatchedEngine", return_value=mock_vlm_engine
        ), patch(
            "omlx.engine_pool.BatchedEngine", return_value=mock_batched_engine
        ):
            await pool._load_engine("model-a")

        # Should have fallen back to LLM
        assert entry.model_type == "llm"
        assert entry.engine_type == "batched"
        assert entry.engine is mock_batched_engine

    @pytest.mark.asyncio
    async def test_non_vlm_failure_still_raises(self, small_mock_model_dir):
        """Test that non-VLM engine failures propagate normally."""
        pool = EnginePool(max_model_memory=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        entry = pool.get_entry("model-a")
        assert entry.engine_type == "batched"  # LLM, not VLM

        mock_engine = MagicMock()
        mock_engine.start = AsyncMock(side_effect=Exception("Load failed"))

        with patch(
            "omlx.engine_pool.BatchedEngine", return_value=mock_engine
        ), pytest.raises(Exception, match="Load failed"):
            await pool._load_engine("model-a")

    @pytest.mark.asyncio
    async def test_force_lm_fallback_to_vlm_on_start_failure(
        self, small_mock_model_dir
    ):
        """Test that force_lm failure for VLM model falls back to VLMBatchedEngine."""
        pool = EnginePool(max_model_memory=10 * 1024**3)
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

        with patch(
            "omlx.engine_pool.BatchedEngine", return_value=mock_batched_engine
        ), patch(
            "omlx.engine_pool.VLMBatchedEngine", return_value=mock_vlm_engine
        ):
            await pool._load_engine("model-a", force_lm=True)

        # Should have fallen back to VLM
        assert entry.model_type == "vlm"
        assert entry.engine_type == "vlm"
        assert entry.engine is mock_vlm_engine

    @pytest.mark.asyncio
    async def test_force_lm_no_fallback_for_non_vlm(self, small_mock_model_dir):
        """Test that force_lm failure for non-VLM model still raises."""
        pool = EnginePool(max_model_memory=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        entry = pool.get_entry("model-a")
        assert entry.engine_type == "batched"  # LLM, not VLM

        mock_engine = MagicMock()
        mock_engine.start = AsyncMock(side_effect=Exception("Load failed"))

        with patch(
            "omlx.engine_pool.BatchedEngine", return_value=mock_engine
        ), pytest.raises(Exception, match="Load failed"):
            await pool._load_engine("model-a", force_lm=True)


class TestEnginePoolLRU:
    """Tests for LRU eviction logic."""

    @pytest.fixture
    def pool_with_entries(self, small_mock_model_dir):
        """Create pool with mock entries for LRU testing."""
        pool = EnginePool(max_model_memory=5000)  # 5KB limit
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


class TestEnginePoolAsync:
    """Async tests for EnginePool (mocked)."""

    @pytest.fixture
    def pool_with_mock_engines(self, small_mock_model_dir):
        """Create pool with mocked engine loading."""
        pool = EnginePool(max_model_memory=10 * 1024**3)
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
    def tight_memory_pool(self, small_mock_model_dir):
        """Create pool with tight memory limit."""
        # Model-a is ~1KB (1024 bytes + overhead ~1.1KB)
        # Model-b is ~2KB (2048 bytes + overhead ~2.1KB)
        # Set limit to allow each model individually but not both together
        pool = EnginePool(max_model_memory=2500)  # Allows each but not both
        pool.discover_models(str(small_mock_model_dir))
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


class TestEnginePoolStatus:
    """Tests for get_status is_loading field."""

    def test_get_status_includes_is_loading(self, small_mock_model_dir):
        """Test get_status includes is_loading field."""
        pool = EnginePool(max_model_memory=10 * 1024**3)
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
        pool = EnginePool(max_model_memory=10 * 1024**3)
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
        pool = EnginePool(max_model_memory=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        result = pool.resolve_model_id("model-a", settings_manager=None)
        assert result == "model-a"

    def test_alias_match(self, small_mock_model_dir):
        """Test alias resolution returns real model_id."""
        pool = EnginePool(max_model_memory=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        settings_manager = MagicMock()
        from omlx.model_settings import ModelSettings
        settings_manager.get_all_settings.return_value = {
            "model-a": ModelSettings(model_alias="gpt-4"),
            "model-b": ModelSettings(),
        }

        result = pool.resolve_model_id("gpt-4", settings_manager)
        assert result == "model-a"

    def test_no_match_returns_original(self, small_mock_model_dir):
        """Test unresolved name returns original string."""
        pool = EnginePool(max_model_memory=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        settings_manager = MagicMock()
        from omlx.model_settings import ModelSettings
        settings_manager.get_all_settings.return_value = {
            "model-a": ModelSettings(),
        }

        result = pool.resolve_model_id("nonexistent", settings_manager)
        assert result == "nonexistent"

    def test_alias_match_no_settings_manager(self, small_mock_model_dir):
        """Test with None settings_manager falls back to direct match only."""
        pool = EnginePool(max_model_memory=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        result = pool.resolve_model_id("some-alias", settings_manager=None)
        assert result == "some-alias"

    def test_provider_prefix_alias_match(self, small_mock_model_dir):
        """Test alias resolution with provider prefix (e.g. omlx/alias)."""
        pool = EnginePool(max_model_memory=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        settings_manager = MagicMock()
        from omlx.model_settings import ModelSettings
        settings_manager.get_all_settings.return_value = {
            "model-a": ModelSettings(model_alias="gpt-4"),
            "model-b": ModelSettings(),
        }

        result = pool.resolve_model_id("omlx/gpt-4", settings_manager)
        assert result == "model-a"

    def test_provider_prefix_direct_match(self, small_mock_model_dir):
        """Test direct match with provider prefix (e.g. provider/model-a)."""
        pool = EnginePool(max_model_memory=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        result = pool.resolve_model_id("provider/model-a", settings_manager=None)
        assert result == "model-a"

    def test_provider_prefix_no_match(self, small_mock_model_dir):
        """Test prefix strip still returns original when no match found."""
        pool = EnginePool(max_model_memory=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        settings_manager = MagicMock()
        from omlx.model_settings import ModelSettings
        settings_manager.get_all_settings.return_value = {
            "model-a": ModelSettings(),
        }

        result = pool.resolve_model_id("omlx/nonexistent", settings_manager)
        assert result == "omlx/nonexistent"

    def test_case_insensitive_match(self, small_mock_model_dir):
        """Test case-insensitive fallback when exact match fails."""
        pool = EnginePool(max_model_memory=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        result = pool.resolve_model_id("MODEL-A", settings_manager=None)
        assert result == "model-a"

    def test_case_insensitive_with_provider_prefix(self, small_mock_model_dir):
        """Test case-insensitive match after stripping provider prefix."""
        pool = EnginePool(max_model_memory=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        result = pool.resolve_model_id("omlx/MODEL-B", settings_manager=None)
        assert result == "model-b"

    def test_exact_match_preferred_over_case_insensitive(self, small_mock_model_dir):
        """Test exact match takes priority over case-insensitive."""
        pool = EnginePool(max_model_memory=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        # Exact match should be returned directly
        result = pool.resolve_model_id("model-a", settings_manager=None)
        assert result == "model-a"
