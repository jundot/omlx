# SPDX-License-Identifier: Apache-2.0
"""Tests for models/vlm.py — VLMModelAdapter for BatchGenerator compatibility."""

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

# Mock mlx before importing the module
import sys


# Create mock mlx modules
class MockMXArray:
    """Minimal mock for mx.array."""

    def __init__(self, shape=None, data=None):
        self._shape = shape or (1, 10, 128)
        self._data = data

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    def __getitem__(self, key):
        return MockMXArray(self._shape)


class TestVLMModelAdapter:
    """Tests for VLMModelAdapter."""

    def _make_mock_vlm_model(self):
        """Create a mock VLM model with language_model."""
        vlm_model = MagicMock()
        language_model = MagicMock()

        # Set up language_model properties
        language_model.model = MagicMock()
        language_model.model.layers = [MagicMock() for _ in range(4)]
        language_model.args = MagicMock()

        vlm_model.language_model = language_model
        vlm_model.config = MagicMock()
        vlm_model.config.model_type = "qwen3_5_moe"

        return vlm_model

    def test_init(self):
        """Test initialization stores vlm_model reference."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_mock_vlm_model()
        adapter = VLMModelAdapter(vlm)

        assert adapter._vlm_model is vlm
        assert adapter._language_model is vlm.language_model
        assert adapter._pending_embeds is None
        assert adapter._embed_offset == 0

    def test_layers_property(self):
        """Test layers property delegates to language_model.model.layers."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_mock_vlm_model()
        adapter = VLMModelAdapter(vlm)

        assert adapter.layers is vlm.language_model.model.layers
        assert len(adapter.layers) == 4

    def test_config_property(self):
        """Test config property returns vlm_model config."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_mock_vlm_model()
        adapter = VLMModelAdapter(vlm)

        assert adapter.config is vlm.config

    def test_model_type_property(self):
        """Test model_type property returns config.model_type."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_mock_vlm_model()
        adapter = VLMModelAdapter(vlm)

        assert adapter.model_type == "qwen3_5_moe"

    def test_args_property(self):
        """Test args property delegates to language_model."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_mock_vlm_model()
        adapter = VLMModelAdapter(vlm)

        assert adapter.args is vlm.language_model.args

    def test_make_cache_delegates(self):
        """Test make_cache delegates to language_model."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_mock_vlm_model()
        vlm.language_model.make_cache.return_value = [MagicMock()]
        adapter = VLMModelAdapter(vlm)

        cache = adapter.make_cache()
        vlm.language_model.make_cache.assert_called_once()
        assert cache is vlm.language_model.make_cache.return_value

    def test_set_pending_embeddings(self):
        """Test set_pending_embeddings stores state."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_mock_vlm_model()
        adapter = VLMModelAdapter(vlm)

        embeds = MockMXArray(shape=(1, 20, 128))
        kwargs = {"position_ids": MockMXArray()}

        adapter.set_pending_embeddings(embeds, kwargs)

        assert adapter._pending_embeds is embeds
        assert adapter._pending_kwargs == kwargs
        assert adapter._embed_offset == 0
        assert adapter.has_pending_embeddings is True

    def test_clear_pending_embeddings(self):
        """Test clear_pending_embeddings resets state."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_mock_vlm_model()
        adapter = VLMModelAdapter(vlm)

        embeds = MockMXArray(shape=(1, 20, 128))
        adapter.set_pending_embeddings(embeds)

        adapter.clear_pending_embeddings()

        assert adapter._pending_embeds is None
        assert adapter._pending_kwargs == {}
        assert adapter._embed_offset == 0
        assert adapter.has_pending_embeddings is False

    def test_forward_without_embeddings(self):
        """Test forward pass without pending embeddings delegates to language_model."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_mock_vlm_model()
        adapter = VLMModelAdapter(vlm)

        input_ids = MockMXArray(shape=(1, 10))
        cache = [MagicMock()]
        expected = MagicMock()
        vlm.language_model.__call__ = MagicMock(return_value=expected)

        result = adapter(input_ids, cache=cache)
        # Cache is wrapped with _IntOffsetCacheProxy, so check args manually
        vlm.language_model.assert_called_once()
        call_args = vlm.language_model.call_args
        assert call_args[0][0] is input_ids
        assert len(call_args[1]["cache"]) == 1

    def test_forward_with_embeddings(self):
        """Test forward pass with pending embeddings injects inputs_embeds."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_mock_vlm_model()
        adapter = VLMModelAdapter(vlm)

        # Set up pending embeddings (batch=1, seq=20, hidden=128)
        embeds = MockMXArray(shape=(1, 20, 128))
        adapter.set_pending_embeddings(embeds)

        # Call with chunk of 10 tokens
        input_ids = MockMXArray(shape=(1, 10))
        cache = [MagicMock()]
        adapter(input_ids, cache=cache)

        # Should call language_model with inputs_embeds chunk
        call_args = vlm.language_model.call_args
        assert "inputs_embeds" in call_args.kwargs or len(call_args.args) > 1
        assert adapter._embed_offset == 10

    def test_embedding_offset_tracks_chunks(self):
        """Test that embed_offset correctly tracks through chunked prefill."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_mock_vlm_model()
        adapter = VLMModelAdapter(vlm)

        embeds = MockMXArray(shape=(1, 30, 128))
        adapter.set_pending_embeddings(embeds)

        # Chunk 1: 10 tokens
        adapter(MockMXArray(shape=(1, 10)), cache=[MagicMock()])
        assert adapter._embed_offset == 10
        assert adapter.has_pending_embeddings is True

        # Chunk 2: 10 tokens
        adapter(MockMXArray(shape=(1, 10)), cache=[MagicMock()])
        assert adapter._embed_offset == 20
        assert adapter.has_pending_embeddings is True

        # Chunk 3: 10 tokens (final, should clear)
        adapter(MockMXArray(shape=(1, 10)), cache=[MagicMock()])
        # After consuming all embeddings, should be cleared
        assert adapter._pending_embeds is None

    def test_get_input_embeddings_delegates(self):
        """Test get_input_embeddings delegates to vlm_model."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_mock_vlm_model()
        expected = MagicMock()
        vlm.get_input_embeddings.return_value = expected
        adapter = VLMModelAdapter(vlm)

        input_ids = MockMXArray()
        pixel_values = MockMXArray()
        result = adapter.get_input_embeddings(input_ids, pixel_values)

        vlm.get_input_embeddings.assert_called_once_with(input_ids, pixel_values)
        assert result is expected


class TestVLMModelAdapterModelProperty:
    """Tests for VLMModelAdapter.model property (for nested access)."""

    def test_model_property(self):
        """Test .model returns language_model.model for BatchGenerator compatibility."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = MagicMock()
        vlm.language_model.model = MagicMock()
        vlm.language_model.model.layers = [MagicMock()]
        adapter = VLMModelAdapter(vlm)

        # BatchGenerator accesses model.layers
        assert adapter.layers is vlm.language_model.model.layers
