# SPDX-License-Identifier: Apache-2.0
"""Tests for models/vlm.py — VLMModelAdapter for BatchGenerator compatibility."""

import pytest

from unittest.mock import MagicMock


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

    def test_release_resources_drops_model_references(self):
        """release_resources drops raw VLM/language model and pending arrays."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_mock_vlm_model()
        adapter = VLMModelAdapter(vlm)
        adapter._pending_embeds = MockMXArray()
        adapter._pending_kwargs = {"position_ids": MockMXArray()}
        adapter._uid_rope_deltas[1] = 2.0
        adapter._batch_rope_deltas = MockMXArray()

        adapter.release_resources()

        assert adapter._vlm_model is None
        assert adapter._language_model is None
        assert adapter._pending_embeds is None
        assert adapter._pending_kwargs == {}
        assert adapter._uid_rope_deltas == {}
        assert adapter._batch_rope_deltas is None

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

        adapter(input_ids, cache=cache)
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

    def test_detect_mrope_true_for_minimax_m3_vl(self):
        """MiniMax M3 uses per-row decode positions even without mrope_section."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = MagicMock(spec=[])
        vlm.config = MagicMock(spec=[])
        vlm.config.model_type = "minimax_m3_vl"
        assert VLMModelAdapter._detect_mrope(vlm) is True
        assert VLMModelAdapter._detect_minimax_m3(vlm) is True


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

    def _make_minimax_m3_vlm_model(self):
        """Create a mock MiniMax M3 VLM model."""
        vlm = self._make_mrope_vlm_model()
        vlm.config.model_type = "minimax_m3_vl"
        vlm.config.text_config.model_type = "minimax_m3_vl"
        vlm.config.text_config.rope_scaling = None
        vlm.config.text_config.rope_parameters = None
        return vlm

    def _make_qwen4_mrope_vlm_model(self):
        """Create the exact root/text model types shipped by Flash Next."""
        vlm = self._make_mrope_vlm_model()
        vlm.config.model_type = "qwen4_exp"
        vlm.config.text_config.model_type = "qwen4_exp_text"
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

    def test_mrope_decode_scalar_cache_offset_uses_position_ids(self):
        """Singleton KVCache offset should not rely on stale language-model state."""
        import mlx.core as mx

        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)
        adapter.set_batch_rope_deltas(mx.array([0.0]))

        input_ids = mx.zeros((1, 1), dtype=mx.int32)
        cache_layer = MagicMock()
        cache_layer.offset = 16384
        cache = [cache_layer]

        adapter(input_ids, cache=cache)

        call_kwargs = vlm.language_model.call_args[1]
        pos_ids = call_kwargs["position_ids"]
        assert pos_ids.shape == (3, 1, 1)
        assert pos_ids[0, 0, 0].item() == 16384.0

    def test_qwen4_b1_text_prefill_uses_canonical_rank_two_positions(self):
        """Three broadcast-identical text planes stay in QSA's proven shape."""
        import mlx.core as mx

        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_qwen4_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)
        assert adapter.model_type == "qwen4_exp"

        adapter.set_text_prefill_rope_delta(0.0)
        cache_layer = MagicMock()
        cache_layer.offset = 16384
        adapter(mx.zeros((1, 4), dtype=mx.int32), cache=[cache_layer])

        position_ids = vlm.language_model.call_args.kwargs["position_ids"]
        assert position_ids.shape == (1, 4)
        assert position_ids.tolist() == [[16384, 16385, 16386, 16387]]

    def test_qwen4_text_prefill_proof_is_one_shot_after_failed_call(self):
        """An exception cannot leak the text-only proof into the next call."""
        import mlx.core as mx
        import pytest

        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_qwen4_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)
        cache_layer = MagicMock()
        cache_layer.offset = 64
        vlm.language_model.side_effect = [RuntimeError("cancelled"), MagicMock()]

        adapter.set_text_prefill_rope_delta(0.0)
        with pytest.raises(RuntimeError, match="cancelled"):
            adapter(mx.zeros((1, 2), dtype=mx.int32), cache=[cache_layer])

        # No rebind at all: the old delta array is still present, but the
        # stronger text-only capability must have been consumed by the failed
        # call and therefore cannot affect this later generic request.
        adapter(mx.zeros((1, 2), dtype=mx.int32), cache=[cache_layer])
        position_ids = vlm.language_model.call_args.kwargs["position_ids"]
        assert position_ids.shape == (3, 1, 2)

    def test_qwen4_text_prefill_b2_remains_rank_three(self):
        """The text proof is not widened to an unqualified batched QSA path."""
        import mlx.core as mx

        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_qwen4_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)
        adapter.set_text_prefill_rope_delta(0.0)
        # A synthetic second delta demonstrates that the adapter refuses to
        # reinterpret the one-row proof when the model call is batched.
        adapter._batch_rope_deltas = mx.array([0.0, 0.0])
        cache_layer = MagicMock()
        cache_layer.offset = mx.array([128, 96])
        adapter(mx.zeros((2, 2), dtype=mx.int32), cache=[cache_layer])

        position_ids = vlm.language_model.call_args.kwargs["position_ids"]
        assert position_ids.shape == (3, 2, 2)

    def test_qwen4_media_positions_remain_divergent_rank_three(self):
        """True mRoPE media planes bypass text canonicalization unchanged."""
        import mlx.core as mx

        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_qwen4_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)
        divergent = mx.array(
            [
                [[10, 11, 12]],
                [[10, 10, 11]],
                [[7, 8, 8]],
            ],
            dtype=mx.int32,
        )
        adapter(
            mx.zeros((1, 3), dtype=mx.int32),
            cache=[MagicMock()],
            inputs_embeds=mx.zeros((1, 3, 8)),
            vlm_extra_kwargs={"position_ids": divergent},
        )

        position_ids = vlm.language_model.call_args.kwargs["position_ids"]
        assert position_ids.shape == (3, 1, 3)
        assert mx.array_equal(position_ids, divergent).item()

    def test_non_minimax_mrope_mismatched_delta_size_keeps_existing_path(self):
        """Non-MiniMax mRoPE models keep prior no-position_ids mismatch behavior."""
        import mlx.core as mx

        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)
        assert adapter._uses_minimax_m3_positions is False

        adapter.set_batch_rope_deltas(mx.array([10.0, 0.0]))

        input_ids = mx.zeros((3, 1), dtype=mx.int32)
        cache_layer = MagicMock()
        cache_layer.offset = mx.array([50, 30, 20])
        cache = [cache_layer]

        adapter(input_ids, cache=cache)

        call_kwargs = vlm.language_model.call_args[1]
        assert "position_ids" not in call_kwargs

    def test_minimax_m3_decode_uses_2d_position_ids(self):
        """MiniMax M3 expects position_ids = (batch, seq), not Qwen-style rank 3."""
        import mlx.core as mx

        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_minimax_m3_vlm_model()
        adapter = VLMModelAdapter(vlm)
        assert adapter._uses_mrope is True
        assert adapter._uses_minimax_m3_positions is True

        adapter.set_batch_rope_deltas(mx.array([-50.0, 0.0]))

        input_ids = mx.zeros((2, 2), dtype=mx.int32)
        cache_layer = MagicMock()
        cache_layer.offset = mx.array([100, 80])
        cache = [cache_layer]

        adapter(input_ids, cache=cache)

        call_kwargs = vlm.language_model.call_args[1]
        pos_ids = call_kwargs["position_ids"]
        assert pos_ids.shape == (2, 2)
        assert pos_ids[0, 0].item() == 50.0
        assert pos_ids[0, 1].item() == 51.0
        assert pos_ids[1, 0].item() == 80.0
        assert pos_ids[1, 1].item() == 81.0

    def test_mrope_multi_token_window_advances_positions(self):
        """Regression: each row of an mRoPE window must advance from its start.

        Multi-token windows (speculative-decode verify) previously broadcast each
        row's start offset across the whole window, so every position in the
        window was rope-rotated at the first position. That silently corrupted the
        keys the verify wrote back into the cache. The consuming attention builds
        its own positions as arange(offset, offset + L) when none are supplied;
        the positions we pass must match that.
        """
        import mlx.core as mx

        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)
        assert adapter._uses_minimax_m3_positions is False

        adapter.set_batch_rope_deltas(mx.array([-50.0, 0.0]))

        input_ids = mx.zeros((2, 2), dtype=mx.int32)
        cache_layer = MagicMock()
        cache_layer.offset = mx.array([100, 80])
        cache = [cache_layer]

        adapter(input_ids, cache=cache)

        call_kwargs = vlm.language_model.call_args[1]
        pos_ids = call_kwargs["position_ids"]
        assert pos_ids.shape == (3, 2, 2)
        for section in range(3):
            assert pos_ids[section, 0, 0].item() == 50.0
            assert pos_ids[section, 0, 1].item() == 51.0
            assert pos_ids[section, 1, 0].item() == 80.0
            assert pos_ids[section, 1, 1].item() == 81.0

    def test_get_last_rope_deltas(self):
        """get_last_rope_deltas extracts value from language model."""
        import mlx.core as mx

        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)

        vlm.language_model._rope_deltas = mx.array(-42.0)
        assert adapter.get_last_rope_deltas() == -42.0

        vlm.language_model._rope_deltas = mx.array([[-42.0], [-7.0]])
        assert adapter.get_last_rope_deltas() == -42.0

        vlm.language_model._rope_deltas = None
        assert adapter.get_last_rope_deltas() == 0.0

    def test_mrope_scalar_offset_fallback_initializes_position_state(self):
        """Regression #2387: MiniCPM-o text-only prefill with scalar cache offsets.

        MiniCPM-o detects as mRoPE (mlx-vlm injects mrope_section into its
        text config) but its SigLIP VisionConfig has no spatial_merge_size,
        so the borrowed qwen3_vl LanguageModel crashes in get_rope_index()
        unless position state is initialized first (#241). The mRoPE branch
        fallback for scalar cache offsets must call _set_position_state.
        """
        import mlx.core as mx

        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)
        assert adapter._uses_mrope is True

        input_ids = mx.zeros((1, 16), dtype=mx.int32)
        cache_layer = MagicMock(spec=["offset"])
        cache_layer.offset = 0
        cache = [cache_layer]

        adapter(input_ids, cache=cache)

        vlm._set_position_state.assert_called_once_with(input_ids)
        call_kwargs = vlm.language_model.call_args[1]
        assert "position_ids" not in call_kwargs

    def test_mrope_delta_fallback_initializes_position_state(self):
        """Same as above for the batch-deltas branch with unusable offsets."""
        import mlx.core as mx

        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)
        adapter.set_batch_rope_deltas(mx.array([0.0]))

        input_ids = mx.zeros((1, 16), dtype=mx.int32)
        cache_layer = MagicMock(spec=[])
        cache = [cache_layer]

        adapter(input_ids, cache=cache)

        vlm._set_position_state.assert_called_once_with(input_ids)
        call_kwargs = vlm.language_model.call_args[1]
        assert "position_ids" not in call_kwargs


    def test_qwen4_text_request_steps_use_rank_two_positions(self, monkeypatch):
        """A scheduler-proven text request keeps (1, T) positions through decode and MTP verify steps."""
        import omlx.models.vlm as vlm_module

        monkeypatch.setattr(vlm_module, "_STEP_TEXT_POSITIONS_MIN_CONTEXT", 0)
        import mlx.core as mx

        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_qwen4_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)
        adapter.mark_text_positions(7)
        cache_layer = MagicMock()
        cache_layer.offset = 64

        adapter.set_step_rope_deltas(mx.array([0.0]), uids=[7])
        adapter(mx.zeros((1, 1), dtype=mx.int32), cache=[cache_layer])
        assert vlm.language_model.call_args.kwargs["position_ids"].tolist() == [[64]]

        adapter.set_step_rope_deltas(mx.array([0.0]), uids=[7])
        adapter(mx.zeros((1, 4), dtype=mx.int32), cache=[cache_layer])
        position_ids = vlm.language_model.call_args.kwargs["position_ids"]
        assert position_ids.shape == (1, 4)
        assert position_ids.tolist() == [[64, 65, 66, 67]]

    def test_qwen4_step_positions_stay_rank_three_without_text_proof(self, monkeypatch):
        """Unproven requests and batched steps keep the fail-closed (3, B, T) form."""
        import omlx.models.vlm as vlm_module

        monkeypatch.setattr(vlm_module, "_STEP_TEXT_POSITIONS_MIN_CONTEXT", 0)
        import mlx.core as mx

        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_qwen4_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)
        adapter.mark_text_positions(7)
        cache_layer = MagicMock()
        cache_layer.offset = 64

        adapter.set_step_rope_deltas(mx.array([0.0]), uids=[8])  # never proven
        adapter(mx.zeros((1, 4), dtype=mx.int32), cache=[cache_layer])
        assert vlm.language_model.call_args.kwargs["position_ids"].shape == (3, 1, 4)

        adapter.set_step_rope_deltas(mx.array([0.0, 0.0]), uids=[7, 9])  # batched
        cache_layer.offset = mx.array([64, 32])
        adapter(mx.zeros((2, 1), dtype=mx.int32), cache=[cache_layer])
        assert vlm.language_model.call_args.kwargs["position_ids"].shape == (3, 2, 1)

        # A step-bound proof covers every adapter call of that step (an MTP step
        # runs a decode forward and then the verify forward) and is cleared by
        # the next bind, unlike the one-shot prefill proof.
        cache_layer.offset = 64
        adapter.set_step_rope_deltas(mx.array([0.0]), uids=[7])
        adapter(mx.zeros((1, 1), dtype=mx.int32), cache=[cache_layer])
        adapter(mx.zeros((1, 4), dtype=mx.int32), cache=[cache_layer])
        assert vlm.language_model.call_args.kwargs["position_ids"].shape == (1, 4)
        adapter.set_batch_rope_deltas(mx.array([0.0]))
        adapter(mx.zeros((1, 1), dtype=mx.int32), cache=[cache_layer])
        assert vlm.language_model.call_args.kwargs["position_ids"].shape == (3, 1, 1)
        adapter.set_step_rope_deltas(mx.array([0.0]), uids=[7])
        adapter.set_text_prefill_rope_delta(0.0)
        adapter(mx.zeros((1, 2), dtype=mx.int32), cache=[cache_layer])
        adapter(mx.zeros((1, 2), dtype=mx.int32), cache=[cache_layer])
        assert vlm.language_model.call_args.kwargs["position_ids"].shape == (3, 1, 2)

    def test_qwen4_step_text_positions_kill_switch(self, monkeypatch):
        """OMLX_QWEN4_STEP_TEXT_POSITIONS=0 keeps every step on the rank-three form."""
        import mlx.core as mx

        import omlx.models.vlm as vlm_module
        from omlx.models.vlm import VLMModelAdapter

        monkeypatch.setattr(vlm_module, "_STEP_TEXT_POSITIONS_DISABLED", True)
        vlm = self._make_qwen4_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)
        adapter.mark_text_positions(7)
        cache_layer = MagicMock()
        cache_layer.offset = 64
        adapter.set_step_rope_deltas(mx.array([0.0]), uids=[7])
        adapter(mx.zeros((1, 4), dtype=mx.int32), cache=[cache_layer])
        assert vlm.language_model.call_args.kwargs["position_ids"].shape == (3, 1, 4)

    def test_qwen4_step_text_positions_engage_only_above_min_context(self, monkeypatch):
        """Backbone rows keep the generic form below the context threshold (gathered arms are
        null-to-negative there) and switch to (1, T) above it."""
        import mlx.core as mx

        import omlx.models.vlm as vlm_module
        from omlx.models.vlm import VLMModelAdapter

        monkeypatch.setattr(vlm_module, "_STEP_TEXT_POSITIONS_MIN_CONTEXT", 65536)
        vlm = self._make_qwen4_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)
        adapter.mark_text_positions(7)
        cache_layer = MagicMock()

        cache_layer.offset = 41_000
        adapter.set_step_rope_deltas(mx.array([0.0]), uids=[7])
        adapter(mx.zeros((1, 4), dtype=mx.int32), cache=[cache_layer])
        assert vlm.language_model.call_args.kwargs["position_ids"].shape == (3, 1, 4)

        cache_layer.offset = 82_000
        adapter.set_step_rope_deltas(mx.array([0.0]), uids=[7])
        adapter(mx.zeros((1, 4), dtype=mx.int32), cache=[cache_layer])
        position_ids = vlm.language_model.call_args.kwargs["position_ids"]
        assert position_ids.shape == (1, 4)
        assert position_ids.tolist() == [[82_000, 82_001, 82_002, 82_003]]

        # The scheduler-proven prefill positions are not subject to the threshold.
        cache_layer.offset = 1_000
        adapter.set_text_prefill_rope_delta(0.0)
        adapter(mx.zeros((1, 4), dtype=mx.int32), cache=[cache_layer])
        assert vlm.language_model.call_args.kwargs["position_ids"].shape == (1, 4)

    def test_qwen4_unregister_clears_text_positions_proof(self):
        import mlx.core as mx

        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_qwen4_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)
        adapter.mark_text_positions(7)
        adapter.unregister_rope_delta(7)
        cache_layer = MagicMock()
        cache_layer.offset = 64
        adapter.set_step_rope_deltas(mx.array([0.0]), uids=[7])
        adapter(mx.zeros((1, 1), dtype=mx.int32), cache=[cache_layer])
        assert vlm.language_model.call_args.kwargs["position_ids"].shape == (3, 1, 1)


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

    def test_return_hidden_preserves_language_model_output(self):
        """MTP backbone calls must keep hidden_states/gdn_states intact."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = self._make_mock_vlm_model()
        adapter = VLMModelAdapter(vlm)

        lm_output = MagicMock()
        lm_output.logits = MockMXArray(shape=(2, 10, 32000))
        lm_output.hidden_states = [MockMXArray(shape=(2, 10, 128))]
        lm_output.gdn_states = [{"state": "mock"}]
        vlm.language_model.return_value = lm_output

        result = adapter(
            MockMXArray(shape=(2, 10)),
            cache=[MagicMock()],
            return_hidden=True,
        )
        assert result is lm_output


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


def test_adapter_forwards_prefetch_ple_to_the_language_model():
    from unittest.mock import MagicMock

    from omlx.models.vlm import VLMModelAdapter

    vlm = MagicMock()
    vlm.config.model_type = "qwen4_exp"
    adapter = VLMModelAdapter(vlm)
    next_ids, current_ids = object(), object()
    adapter.prefetch_ple(next_ids, current_ids)
    vlm.language_model.prefetch_ple.assert_called_once_with(next_ids, current_ids)
    plain = MagicMock(spec=[])
    plain.language_model = MagicMock(spec=[])
    plain.config = MagicMock()
    plain.config.model_type = "qwen3_5_moe"
    VLMModelAdapter(plain).prefetch_ple(next_ids, current_ids)  # no hook: no error


def test_ssd_cache_restore_binds_nested_vlm_caches_and_preserves_quantization():
    from types import SimpleNamespace

    import mlx.core as mx
    from mlx_lm.models import cache as lm_cache
    from mlx_vlm.models import cache as vlm_cache

    from omlx.models.vlm import VLMModelAdapter
    from omlx.turboquant_kv import TurboQuantKVCache

    keys = mx.arange(16 * 64).reshape(1, 1, 16, 64).astype(mx.float16) / 1024
    kv = lm_cache.KVCache()
    kv.update_and_fetch(keys, keys * 0.5)
    recurrent = lm_cache.ArraysCache(4)
    recurrent[0] = mx.ones((1, 3, 8))
    recurrent[1] = mx.ones((1, 2, 4, 4))
    recurrent[2] = mx.zeros((1, 2, 8))
    recurrent[3] = mx.array([[3, 4]])
    rotating = lm_cache.RotatingKVCache(max_size=8)
    rotating.update_and_fetch(keys, keys)
    chunked = lm_cache.ChunkedKVCache(chunk_size=8)
    chunked.update_and_fetch(keys, keys)
    chunked.maybe_trim_front()
    quantized = TurboQuantKVCache.from_cache(kv, bits=4)
    source = [
        lm_cache.CacheList(kv, lm_cache.CacheList(recurrent, rotating)),
        quantized,
        chunked,
    ]

    def make_cache():
        return [
            vlm_cache.CacheList(
                vlm_cache.KVCache(),
                vlm_cache.CacheList(
                    vlm_cache.ArraysCache(4), vlm_cache.RotatingKVCache(max_size=8)
                ),
            ),
            vlm_cache.KVCache(),
            vlm_cache.ChunkedKVCache(chunk_size=8),
        ]

    adapter = VLMModelAdapter(
        SimpleNamespace(language_model=SimpleNamespace(make_cache=make_cache))
    )
    restored = adapter.restore_cache(source)
    assert type(restored[0]) is vlm_cache.CacheList
    assert type(restored[0][0]) is vlm_cache.KVCache
    assert type(restored[0][1][0]) is vlm_cache.ArraysCache
    assert type(restored[0][1][1]) is vlm_cache.RotatingKVCache
    assert restored[0][0].offset == kv.offset
    assert restored[0][1][1].meta_state == tuple(
        map(str, (rotating.keep, rotating.max_size, rotating.offset, rotating._idx))
    )
    for old, new in zip(recurrent.cache, restored[0][1][0].cache):
        assert mx.array_equal(old, new)
    assert restored[1] is quantized
    assert type(restored[2]) is vlm_cache.ChunkedKVCache
    assert restored[2].keys is chunked.keys
    assert restored[2].values is chunked.values
    assert restored[2].offset == chunked.offset
    assert restored[2].start_position == chunked.start_position
    assert restored[2].chunk_size == chunked.chunk_size
    restored[0][1][0].update_window(3, mx.array([[3, 4, 5]]), 2)
    assert restored[0][1][0][3].tolist() == [[4, 5]]


def test_restored_rotating_cache_uses_vlm_speculative_buffer():
    from types import SimpleNamespace

    import mlx.core as mx
    from mlx_vlm.models import cache as vlm_cache
    from mlx_vlm.speculative.mtp import _buffer_mtp_target_cache

    from omlx.cache.type_handlers import RotatingKVCacheHandler
    from omlx.models.vlm import VLMModelAdapter

    keys = mx.arange(6 * 8).reshape(1, 1, 6, 8).astype(mx.float16)
    source = RotatingKVCacheHandler().reconstruct_cache(
        {"keys": keys, "values": keys / 2}, (0, 8, 64, 6)
    )
    adapter = VLMModelAdapter(
        SimpleNamespace(
            language_model=SimpleNamespace(
                make_cache=lambda: [vlm_cache.RotatingKVCache(max_size=8)]
            )
        )
    )
    restored = adapter.restore_cache([source])
    assert isinstance(restored[0], vlm_cache.RotatingKVCache)
    assert restored[0].size() == 6
    assert restored[0].offset == 64
    _buffer_mtp_target_cache(
        restored, SimpleNamespace(config=SimpleNamespace(block_size=4)), None
    )
    buffered = restored[0]
    assert isinstance(buffered, vlm_cache.BufferedRotatingKVCache)
    assert buffered.start_position == 58
    assert buffered._idx == 6
    assert mx.array_equal(buffered.state[0], keys)
    buffered.update_and_fetch(mx.ones((1, 1, 4, 8)), mx.ones((1, 1, 4, 8)))
    buffered.trim(3)
    assert buffered.offset == 65
    assert buffered._idx == 7
    assert mx.array_equal(buffered.state[0][..., :6, :], keys)


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("boundary_delta", [-1, 0, 1])
def test_deepseek_v4_pooling_boundary_restore(ratio, boundary_delta, tmp_path):
    from types import SimpleNamespace

    import mlx.core as mx
    from mlx_vlm.models import cache as vlm_cache

    from omlx.cache.type_handlers import CacheListHandler
    from omlx.models.vlm import VLMModelAdapter
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()
    count = ratio + boundary_delta
    remainder = count % ratio
    pool = vlm_cache.PoolingCache(ratio)
    pool.state = (
        mx.arange(remainder * 8).reshape(1, remainder, 8) if remainder else None,
        mx.zeros((1, remainder, 8)) if remainder else None,
        mx.ones((1, count // ratio, 8)),
    )
    kv = vlm_cache.RotatingKVCache(max_size=8)
    keys = mx.ones((1, 1, count, 8))
    kv.update_and_fetch(keys, keys)
    original = vlm_cache.CacheList(kv, vlm_cache.CacheList(pool))
    handler = CacheListHandler()
    state = handler.extract_state(original)
    meta = handler.serialize_meta_state(original)
    # Persist the actual tensors before reconstructing the nested SSD state.
    tensors = {}

    def save(value):
        if isinstance(value, mx.array):
            key = str(len(tensors))
            tensors[key] = value
            return key
        if isinstance(value, (tuple, list)):
            return [save(v) for v in value]
        return value

    layout = save(state["sub_states"])
    path = str(tmp_path / "boundary.safetensors")
    mx.save_safetensors(path, tensors)
    loaded = mx.load(path)

    def load(value):
        if isinstance(value, str):
            return loaded[value]
        if isinstance(value, list):
            return tuple(load(v) for v in value)
        return value

    state["sub_states"] = load(layout)
    source = handler.reconstruct_cache(state, meta)
    adapter = VLMModelAdapter(
        SimpleNamespace(
            language_model=SimpleNamespace(
                make_cache=lambda: [
                    vlm_cache.CacheList(
                        vlm_cache.RotatingKVCache(max_size=8),
                        vlm_cache.CacheList(vlm_cache.PoolingCache(ratio)),
                    )
                ]
            )
        )
    )
    restored = adapter.restore_cache([source])[0]
    result = restored[1][0]
    assert type(result) is vlm_cache.PoolingCache
    assert result.ratio == ratio
    assert result.remainder == remainder
    assert restored[0].offset == count
    for before, after in zip(pool.state, result.state, strict=True):
        if before is None:
            assert after is None
        else:
            assert mx.array_equal(before, after)
    continuation = mx.ones((1, ratio + 1, 8))
    expected = pool.accumulate_windows(continuation, continuation, count)
    actual = result.accumulate_windows(continuation, continuation, count)
    for before, after in zip(expected, actual, strict=True):
        if before is None:
            assert after is None
        else:
            assert mx.array_equal(before, after)


def test_vlm_pooling_restore_rejects_text_overlap_state():
    from types import SimpleNamespace

    import mlx.core as mx
    from mlx_vlm.models.cache import PoolingCache

    from omlx.models.vlm import VLMModelAdapter
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch
    from omlx.patches.deepseek_v4.cache_handlers import PoolingCacheHandler

    apply_deepseek_v4_patch()
    handler = PoolingCacheHandler()
    upstream = PoolingCache(4)
    assert handler.extract_state(upstream)["prev_win_kv"] is None
    source = handler.deserialize_state(
        (None, None, None, mx.ones((1, 1, 4, 8)), None), 4
    )
    adapter = VLMModelAdapter(
        SimpleNamespace(
            language_model=SimpleNamespace(make_cache=lambda: [PoolingCache(4)])
        )
    )
    with pytest.raises(ValueError, match="text pooling overlap"):
        adapter.restore_cache([source])


@pytest.mark.parametrize(
    "tokens, expected", [([1, 2, 3], 0), ([1, 16, 17, 2], 3), ([16, 2, 16, 17, 3], 4)]
)
def test_deepseek_v4_image_prefix_covers_all_images(tokens, expected):
    from types import SimpleNamespace

    from omlx.models.vlm import VLMModelAdapter

    config = SimpleNamespace(
        model_type="deepseek_v4", vision_n_layers=32, vocab_size=16
    )
    adapter = VLMModelAdapter(
        SimpleNamespace(config=config, language_model=SimpleNamespace(config=config))
    )
    assert adapter.minimum_prefill_prefix(tokens) == expected
    config.vision_n_layers = 0
    assert adapter.minimum_prefill_prefix(tokens) == 0
