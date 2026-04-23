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
        vlm.language_model.assert_called_once()
        call_args = vlm.language_model.call_args
        assert call_args[0][0] is input_ids
        assert call_args[1]["cache"] is cache

    def test_forward_text_only_uses_language_model_directly(self):
        """Text-only decode passes cache directly to language_model."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_mock_vlm_model()
        adapter = VLMModelAdapter(vlm)

        input_ids = MockMXArray(shape=(1, 10))
        cache = [MagicMock()]
        vlm.language_model.__call__ = MagicMock(return_value=MagicMock())

        adapter(input_ids, cache=cache)

        vlm.language_model.assert_called_once()
        call_args = vlm.language_model.call_args
        assert call_args[1]["cache"] is cache

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


    def test_forward_with_inputs_embeds_kwarg(self):
        """Test batched VLM path: inputs_embeds kwarg passed to language_model."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_mock_vlm_model()
        adapter = VLMModelAdapter(vlm)

        input_ids = MockMXArray(shape=(2, 10))
        cache = [MagicMock()]
        embeds = MockMXArray(shape=(2, 10, 128))
        extra = {"position_ids": MockMXArray(shape=(2, 10))}

        adapter(input_ids, cache=cache, inputs_embeds=embeds, vlm_extra_kwargs=extra)

        # Should call language_model with inputs_embeds and extra kwargs
        call_args = vlm.language_model.call_args
        assert call_args.kwargs.get("inputs_embeds") is embeds
        assert call_args.kwargs.get("position_ids") is extra["position_ids"]
        # _pending_embeds should NOT be set (batched path doesn't use it)
        assert adapter._pending_embeds is None

    def test_inputs_embeds_kwarg_takes_priority_over_pending(self):
        """Test that inputs_embeds kwarg takes priority over _pending_embeds."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_mock_vlm_model()
        adapter = VLMModelAdapter(vlm)

        # Set pending embeddings (legacy path)
        pending = MockMXArray(shape=(1, 20, 128))
        adapter.set_pending_embeddings(pending)

        # Call with explicit inputs_embeds kwarg (batched path)
        batched = MockMXArray(shape=(2, 10, 128))
        input_ids = MockMXArray(shape=(2, 10))
        adapter(input_ids, cache=[MagicMock()], inputs_embeds=batched)

        # Batched path should be used, not legacy path
        call_args = vlm.language_model.call_args
        assert call_args.kwargs.get("inputs_embeds") is batched


class TestMRoPEDetection:
    """Tests for mRoPE detection and per-request position tracking."""

    def test_detect_mrope_via_rope_scaling(self):
        """Detect mRoPE via text_config.rope_scaling.mrope_section (Qwen3-VL)."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = MagicMock(spec=[])
        vlm.config = MagicMock(spec=[])
        vlm.config.text_config = MagicMock(spec=[])
        vlm.config.text_config.rope_scaling = {
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20],
            "rope_type": "default",
        }
        vlm.config.text_config.rope_parameters = None
        assert VLMModelAdapter._detect_mrope(vlm) is True

    def test_detect_mrope_via_rope_parameters(self):
        """Detect mRoPE via text_config.rope_parameters.mrope_section (Qwen3.5)."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = MagicMock(spec=[])
        vlm.config = MagicMock(spec=[])
        vlm.config.text_config = MagicMock(spec=[])
        vlm.config.text_config.rope_scaling = None
        vlm.config.text_config.rope_parameters = {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "rope_theta": 10000000,
        }
        assert VLMModelAdapter._detect_mrope(vlm) is True

    def test_detect_mrope_false_for_standard_rope(self):
        """Standard RoPE (no mrope_section) should return False."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = MagicMock(spec=[])
        vlm.config = MagicMock(spec=[])
        vlm.config.text_config = MagicMock(spec=[])
        vlm.config.text_config.rope_scaling = None
        vlm.config.text_config.rope_parameters = {
            "full_attention": {"rope_theta": 1000000.0},
            "sliding_attention": {"rope_theta": 10000.0},
        }
        assert VLMModelAdapter._detect_mrope(vlm) is False

    def test_detect_mrope_false_for_no_config(self):
        """No config attribute should return False."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = MagicMock(spec=[])
        assert VLMModelAdapter._detect_mrope(vlm) is False


class TestPerRequestMRoPEDecode:
    """Tests for per-request mRoPE position_ids computation during decode."""

    def _make_mrope_vlm_model(self):
        """Create a mock VLM model with mRoPE config."""
        vlm = MagicMock()
        vlm.language_model = MagicMock()
        vlm.language_model.model = MagicMock()
        vlm.language_model.model.layers = [MagicMock() for _ in range(4)]
        vlm.language_model.args = MagicMock()
        vlm.config = MagicMock(spec=[])
        vlm.config.text_config = MagicMock(spec=[])
        vlm.config.text_config.rope_scaling = {
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20],
        }
        vlm.config.text_config.rope_parameters = None
        vlm.config.model_type = "qwen3_vl_moe"
        return vlm

    def test_mrope_decode_uses_language_model_with_position_ids(self):
        """mRoPE decode with batch_rope_deltas should use language_model with position_ids."""
        import mlx.core as mx
        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)
        assert adapter._uses_mrope is True

        adapter.set_batch_rope_deltas(mx.array([10.0, 0.0]))

        input_ids = mx.zeros((2, 1), dtype=mx.int32)
        cache_layer = MagicMock()
        cache_layer.offset = mx.array([50, 30])
        cache = [cache_layer]

        adapter(input_ids, cache=cache)

        vlm.language_model.assert_called_once()
        call_kwargs = vlm.language_model.call_args[1]
        assert "position_ids" in call_kwargs
        assert call_kwargs["cache"][0] is cache_layer

    def test_mrope_always_uses_language_model(self):
        """mRoPE model always uses vlm language_model with position_ids."""
        import mlx.core as mx
        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)

        cache_layer = MagicMock()
        cache_layer.offset = mx.array([50])

        input_ids = mx.zeros((1, 1), dtype=mx.int32)
        adapter(input_ids, cache=[cache_layer])

        vlm.language_model.assert_called_once()

    def test_position_ids_shape_and_values(self):
        """Verify position_ids = (3, batch, seq) with correct offset+delta values."""
        import mlx.core as mx
        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)

        # Request 0: VLM (offset=100, delta=-50) → position=50
        # Request 1: text-only (offset=80, delta=0) → position=80
        adapter.set_batch_rope_deltas(mx.array([-50.0, 0.0]))

        input_ids = mx.zeros((2, 1), dtype=mx.int32)
        cache_layer = MagicMock()
        cache_layer.offset = mx.array([100, 80])
        cache = [cache_layer]

        adapter(input_ids, cache=cache)

        call_kwargs = vlm.language_model.call_args[1]
        pos_ids = call_kwargs["position_ids"]
        # Shape: (3, 2, 1) — 3 mRoPE dimensions, 2 requests, 1 token
        assert pos_ids.shape == (3, 2, 1)
        # All 3 dimensions should have same values for text-only decode
        # Request 0: 100 + (-50) = 50
        # Request 1: 80 + 0 = 80
        assert pos_ids[0, 0, 0].item() == 50.0
        assert pos_ids[0, 1, 0].item() == 80.0

    def test_get_last_rope_deltas(self):
        """get_last_rope_deltas extracts value from language model."""
        import mlx.core as mx
        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)

        vlm.language_model._rope_deltas = mx.array(-42.0)
        assert adapter.get_last_rope_deltas() == -42.0

        vlm.language_model._rope_deltas = None
        assert adapter.get_last_rope_deltas() == 0.0


class TestLogitsExtraction:
    """Tests for LanguageModelOutput.logits extraction."""

    def _make_mock_vlm_model(self):
        """Create a mock VLM model with language_model."""
        vlm = MagicMock()
        vlm.language_model = MagicMock()
        vlm.language_model.model = MagicMock()
        vlm.language_model.model.layers = [MagicMock() for _ in range(4)]
        vlm.language_model.args = MagicMock()
        vlm.config = MagicMock()
        vlm.config.model_type = "test"
        return vlm

    def test_logits_extraction_from_language_model_output(self):
        """Test that LanguageModelOutput.logits is extracted for BatchGenerator."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_mock_vlm_model()
        adapter = VLMModelAdapter(vlm)

        # Simulate LanguageModelOutput with .logits attribute
        lm_output = MagicMock()
        lm_output.logits = MockMXArray(shape=(2, 10, 32000))
        vlm.language_model.return_value = lm_output

        result = adapter(MockMXArray(shape=(2, 10)), cache=[MagicMock()])
        assert result is lm_output.logits


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



