# SPDX-License-Identifier: Apache-2.0
"""Tests for VisionFeatureSSDCache (memory LRU + SSD persistence)."""

import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

from omlx.cache.vision_feature_cache import (
    VisionFeatureSSDCache,
    VisionFeatureSSDEntry,
    _composite_hash,
    _composite_key,
)


@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Provide a temporary directory for SSD cache tests."""
    return tmp_path / "vision_cache"


@pytest.fixture
def memory_only_cache():
    """Create a memory-only cache (no SSD)."""
    cache = VisionFeatureSSDCache(cache_dir=None, max_memory_entries=3)
    yield cache
    cache.close()


@pytest.fixture
def ssd_cache(tmp_cache_dir):
    """Create a cache with SSD persistence."""
    cache = VisionFeatureSSDCache(
        cache_dir=tmp_cache_dir,
        max_size_bytes=10 * 1024 * 1024,  # 10MB for testing
        max_memory_entries=3,
    )
    yield cache
    cache.close()


class TestCompositeKey:
    def test_composite_key_format(self):
        key = _composite_key("model-a", "hash123")
        assert key == "model-a:hash123"

    def test_composite_hash_deterministic(self):
        h1 = _composite_hash("model", "abc")
        h2 = _composite_hash("model", "abc")
        assert h1 == h2

    def test_composite_hash_differs_for_different_models(self):
        h1 = _composite_hash("model-a", "same_hash")
        h2 = _composite_hash("model-b", "same_hash")
        assert h1 != h2


class TestMemoryCache:
    def test_put_get(self, memory_only_cache):
        features = mx.ones((4, 8))
        memory_only_cache.put("img_hash", "model_a", features)
        result = memory_only_cache.get("img_hash", "model_a")
        assert result is not None
        assert mx.array_equal(result, features)

    def test_miss_returns_none(self, memory_only_cache):
        result = memory_only_cache.get("nonexistent", "model")
        assert result is None

    def test_lru_eviction(self, memory_only_cache):
        # max_memory_entries=3, insert 4 → first should be evicted
        for i in range(4):
            memory_only_cache.put(f"img_{i}", "model", mx.ones((2, 2)) * i)

        # img_0 should be evicted
        assert memory_only_cache.get("img_0", "model") is None
        # img_1, img_2, img_3 should remain
        assert memory_only_cache.get("img_1", "model") is not None
        assert memory_only_cache.get("img_2", "model") is not None
        assert memory_only_cache.get("img_3", "model") is not None

    def test_lru_access_refreshes(self, memory_only_cache):
        # Insert 3 items
        for i in range(3):
            memory_only_cache.put(f"img_{i}", "model", mx.ones((2, 2)) * i)

        # Access img_0 to refresh it
        memory_only_cache.get("img_0", "model")

        # Insert 1 more → img_1 should be evicted (oldest non-accessed)
        memory_only_cache.put("img_3", "model", mx.ones((2, 2)) * 3)

        assert memory_only_cache.get("img_0", "model") is not None  # refreshed
        assert memory_only_cache.get("img_1", "model") is None  # evicted
        assert memory_only_cache.get("img_2", "model") is not None
        assert memory_only_cache.get("img_3", "model") is not None

    def test_composite_key_isolation(self, memory_only_cache):
        features_a = mx.ones((2, 2)) * 1
        features_b = mx.ones((2, 2)) * 2
        memory_only_cache.put("same_hash", "model_a", features_a)
        memory_only_cache.put("same_hash", "model_b", features_b)

        result_a = memory_only_cache.get("same_hash", "model_a")
        result_b = memory_only_cache.get("same_hash", "model_b")
        assert mx.array_equal(result_a, features_a)
        assert mx.array_equal(result_b, features_b)

    def test_overwrite_same_key(self, memory_only_cache):
        memory_only_cache.put("img", "model", mx.ones((2, 2)))
        memory_only_cache.put("img", "model", mx.zeros((2, 2)))
        result = memory_only_cache.get("img", "model")
        assert mx.array_equal(result, mx.zeros((2, 2)))

    def test_stats_tracking(self, memory_only_cache):
        memory_only_cache.put("img", "model", mx.ones((2, 2)))
        memory_only_cache.get("img", "model")  # hit
        memory_only_cache.get("missing", "model")  # miss

        stats = memory_only_cache.stats
        assert stats["saves"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1


class TestSSDCache:
    def test_ssd_write_and_load(self, ssd_cache):
        features = mx.random.normal((10, 16))
        mx.eval(features)
        ssd_cache.put("img_hash", "model_a", features)

        # Wait for background writer
        time.sleep(0.5)

        # Clear memory cache to force SSD read
        with ssd_cache._memory_lock:
            ssd_cache._memory_cache.clear()

        result = ssd_cache.get("img_hash", "model_a")
        assert result is not None
        assert mx.allclose(result, features, atol=1e-5)

    def test_ssd_file_exists(self, ssd_cache, tmp_cache_dir):
        features = mx.ones((4, 8))
        mx.eval(features)
        ssd_cache.put("img_hash", "model_a", features)

        time.sleep(0.5)

        # Check safetensors file exists
        safetensors_files = list(tmp_cache_dir.rglob("*.safetensors"))
        assert len(safetensors_files) == 1

    def test_ssd_startup_scan(self, tmp_cache_dir):
        # Phase 1: create cache and store features
        cache1 = VisionFeatureSSDCache(
            cache_dir=tmp_cache_dir, max_memory_entries=3
        )
        features = mx.ones((4, 8))
        mx.eval(features)
        cache1.put("img_hash", "model_a", features)
        time.sleep(0.5)
        cache1.close()

        # Phase 2: create new cache instance — should scan existing files
        cache2 = VisionFeatureSSDCache(
            cache_dir=tmp_cache_dir, max_memory_entries=3
        )

        # Memory cache is empty, but SSD index should have the entry
        result = cache2.get("img_hash", "model_a")
        assert result is not None
        assert mx.allclose(result, features, atol=1e-5)
        cache2.close()

    def test_ssd_eviction(self, tmp_cache_dir):
        # Very small max_size to trigger eviction
        cache = VisionFeatureSSDCache(
            cache_dir=tmp_cache_dir,
            max_size_bytes=100,  # 100 bytes — any real tensor will exceed this
            max_memory_entries=10,
        )

        # Store multiple features that exceed max_size
        for i in range(3):
            f = mx.ones((4, 8)) * i
            mx.eval(f)
            cache.put(f"img_{i}", "model", f)

        time.sleep(0.5)

        # SSD index should have evicted older entries
        assert cache._ssd_total_size <= 100 or len(cache._ssd_index) <= 1
        cache.close()

    def test_corrupted_file_recovery(self, ssd_cache, tmp_cache_dir):
        features = mx.ones((4, 8))
        mx.eval(features)
        ssd_cache.put("img_hash", "model_a", features)
        time.sleep(0.5)

        # Clear memory cache
        with ssd_cache._memory_lock:
            ssd_cache._memory_cache.clear()

        # Corrupt the file
        safetensors_files = list(tmp_cache_dir.rglob("*.safetensors"))
        assert len(safetensors_files) == 1
        with open(safetensors_files[0], "wb") as f:
            f.write(b"corrupted data")

        # Should return None and remove from index
        result = ssd_cache.get("img_hash", "model_a")
        assert result is None

    def test_close_flushes_writes(self, tmp_cache_dir):
        cache = VisionFeatureSSDCache(
            cache_dir=tmp_cache_dir, max_memory_entries=3
        )
        features = mx.ones((4, 8))
        mx.eval(features)
        cache.put("img_hash", "model_a", features)

        # Close immediately — should flush pending writes
        cache.close()

        # Verify file was written
        safetensors_files = list(tmp_cache_dir.rglob("*.safetensors"))
        assert len(safetensors_files) == 1

    def test_memory_only_mode_no_ssd(self, memory_only_cache):
        features = mx.ones((4, 8))
        memory_only_cache.put("img", "model", features)
        result = memory_only_cache.get("img", "model")
        assert result is not None
        assert mx.array_equal(result, features)

        # No SSD directory should exist
        assert memory_only_cache._cache_dir is None


class TestMultiTensorFeatures:
    def test_multi_tensor_put_get_memory(self, memory_only_cache):
        features = [mx.ones((2, 4)), mx.ones((3, 4)) * 2]
        memory_only_cache.put("multi_img", "model", features)
        result = memory_only_cache.get("multi_img", "model")
        assert isinstance(result, list)
        assert len(result) == 2
        assert mx.array_equal(result[0], features[0])
        assert mx.array_equal(result[1], features[1])

    def test_multi_tensor_ssd_roundtrip(self, ssd_cache):
        features = [mx.ones((2, 4)), mx.ones((3, 4)) * 2]
        for f in features:
            mx.eval(f)
        ssd_cache.put("multi_img", "model", features)
        time.sleep(0.5)

        # Clear memory to force SSD load
        with ssd_cache._memory_lock:
            ssd_cache._memory_cache.clear()

        result = ssd_cache.get("multi_img", "model")
        assert isinstance(result, list)
        assert len(result) == 2
        assert mx.allclose(result[0], features[0], atol=1e-5)
        assert mx.allclose(result[1], features[1], atol=1e-5)


class TestVLMEngineIntegration:
    """Integration tests for vision cache in VLMBatchedEngine using mocks."""

    def test_compute_vision_features_encode_image(self):
        """Model with encode_image should use it directly."""
        from omlx.engine.vlm import VLMBatchedEngine

        engine = VLMBatchedEngine.__new__(VLMBatchedEngine)
        engine._vlm_model = MagicMock()
        engine._vlm_model.config.model_type = "gemma4"

        expected = mx.ones((10, 16))
        engine._vlm_model.encode_image.return_value = expected

        result = engine._compute_vision_features(
            mx.zeros((1, 3, 224, 224)), {}
        )
        assert result is expected
        engine._vlm_model.encode_image.assert_called_once()

    def test_compute_vision_features_qwen_style(self):
        """Qwen-style model should call vision_tower(pv, grid_thw) directly."""
        from omlx.engine.vlm import VLMBatchedEngine

        engine = VLMBatchedEngine.__new__(VLMBatchedEngine)
        engine._vlm_model = MagicMock(spec=[
            "vision_tower", "config",
        ])
        engine._vlm_model.config.model_type = "qwen3_5_moe"

        expected = mx.ones((10, 16))
        engine._vlm_model.vision_tower.return_value = (expected, None)
        engine._vlm_model.vision_tower.patch_embed.proj.weight.dtype = mx.float16

        pixel_values = mx.zeros((1, 3, 224, 224))
        grid_thw = mx.array([[1, 14, 14]])

        result = engine._compute_vision_features(
            pixel_values, {"image_grid_thw": grid_thw}
        )
        assert result is expected
        engine._vlm_model.vision_tower.assert_called_once()

    def test_compute_vision_features_unsupported(self):
        """Unsupported model should return None."""
        from omlx.engine.vlm import VLMBatchedEngine

        engine = VLMBatchedEngine.__new__(VLMBatchedEngine)
        engine._vlm_model = MagicMock(spec=["config"])
        engine._vlm_model.config.model_type = "deepseekocr_2"

        result = engine._compute_vision_features(mx.zeros((1, 3, 224, 224)), {})
        assert result is None

    def test_compute_vision_features_qwen_no_grid_thw(self):
        """Qwen model without grid_thw in extras should return None."""
        from omlx.engine.vlm import VLMBatchedEngine

        engine = VLMBatchedEngine.__new__(VLMBatchedEngine)
        engine._vlm_model = MagicMock(spec=["vision_tower", "config"])
        engine._vlm_model.config.model_type = "qwen2_vl"

        result = engine._compute_vision_features(
            mx.zeros((1, 3, 224, 224)), {}
        )
        assert result is None

    def test_compute_vision_features_llava_style(self):
        """LLaVA model should use vision_tower → select → projector."""
        from omlx.engine.vlm import VLMBatchedEngine

        engine = VLMBatchedEngine.__new__(VLMBatchedEngine)
        engine._vlm_model = MagicMock(spec=[
            "vision_tower", "multi_modal_projector",
            "vision_feature_layer", "vision_feature_select_strategy",
            "config",
        ])
        engine._vlm_model.config.model_type = "llava"
        engine._vlm_model.vision_feature_layer = -2
        engine._vlm_model.vision_feature_select_strategy = "default"

        # vision_tower returns (_, _, hidden_states)
        hidden_state = mx.ones((1, 257, 1024))  # 256 patches + 1 CLS
        engine._vlm_model.vision_tower.return_value = (None, None, [
            mx.zeros((1, 257, 1024)),  # layer -3
            hidden_state,              # layer -2 (selected)
            mx.zeros((1, 257, 1024)),  # layer -1
        ])
        projected = mx.ones((1, 256, 4096))
        engine._vlm_model.multi_modal_projector.return_value = projected

        pixel_values = mx.zeros((1, 3, 336, 336))
        result = engine._compute_vision_features(pixel_values, {})

        assert result is projected
        engine._vlm_model.vision_tower.assert_called_once()
        engine._vlm_model.multi_modal_projector.assert_called_once()
