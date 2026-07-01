# SPDX-License-Identifier: Apache-2.0
"""Tests for embedding functionality."""

import asyncio
import base64
import json
import math
import numpy as np
import struct
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.api.embedding_models import (
    EmbeddingData,
    EmbeddingInputItem,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)
from omlx.api.embedding_utils import (
    count_tokens,
    encode_embedding_base64,
    normalize_embedding_items,
    normalize_input,
    truncate_embedding,
)
from omlx.engine.embedding import EmbeddingEngine
from omlx.model_discovery import detect_model_type
from omlx.models.embedding import EmbeddingOutput


class TestEmbeddingModels:
    """Tests for embedding API Pydantic models."""

    def test_embedding_request_single_input(self):
        """Test EmbeddingRequest with single text input."""
        request = EmbeddingRequest(
            input="Hello, world!",
            model="all-MiniLM-L6-v2",
        )
        assert request.input == "Hello, world!"
        assert request.model == "all-MiniLM-L6-v2"
        assert request.encoding_format == "float"
        assert request.dimensions is None

    def test_embedding_request_list_input(self):
        """Test EmbeddingRequest with list of texts."""
        request = EmbeddingRequest(
            input=["Hello", "World"],
            model="all-MiniLM-L6-v2",
            encoding_format="base64",
            dimensions=256,
        )
        assert request.input == ["Hello", "World"]
        assert request.encoding_format == "base64"
        assert request.dimensions == 256

    def test_embedding_request_max_length_and_truncation(self):
        """Test optional embedding token length controls."""
        request = EmbeddingRequest(
            input="Hello",
            model="all-MiniLM-L6-v2",
            max_length=4096,
            truncation=False,
        )

        assert request.max_length == 4096
        assert request.truncation is False

    def test_embedding_request_rejects_invalid_max_length(self):
        """Test max_length must be a positive integer."""
        with pytest.raises(ValueError, match="greater than 0"):
            EmbeddingRequest(input="Hello", model="all-MiniLM-L6-v2", max_length=0)

    def test_embedding_request_items_input(self):
        """Test EmbeddingRequest with structured items."""
        request = EmbeddingRequest(
            items=[
                EmbeddingInputItem(text="hello"),
                EmbeddingInputItem(image="https://example.com/image.jpg"),
            ],
            model="test-model",
        )
        assert request.input is None
        assert len(request.items) == 2

    def test_embedding_request_rejects_both_input_and_items(self):
        """Test EmbeddingRequest rejects mixed input sources."""
        with pytest.raises(ValueError, match="cannot be provided together"):
            EmbeddingRequest(
                input="hello",
                items=[EmbeddingInputItem(text="world")],
                model="test-model",
            )

    def test_embedding_input_item_requires_text_or_image(self):
        """Test EmbeddingInputItem rejects empty payloads."""
        with pytest.raises(ValueError, match="text or image"):
            EmbeddingInputItem()

    def test_embedding_input_item_allows_empty_string_text(self):
        """Test EmbeddingInputItem preserves empty-string text items."""
        item = EmbeddingInputItem(text="")
        assert item.text == ""
        assert item.image is None

    def test_embedding_data(self):
        """Test EmbeddingData model."""
        data = EmbeddingData(
            index=0,
            embedding=[0.1, 0.2, 0.3],
        )
        assert data.object == "embedding"
        assert data.index == 0
        assert data.embedding == [0.1, 0.2, 0.3]

    def test_embedding_data_base64(self):
        """Test EmbeddingData with base64 embedding."""
        data = EmbeddingData(
            index=1,
            embedding="AAAAAAAAAIA/AAAAQAAAAEA=",
        )
        assert data.embedding == "AAAAAAAAAIA/AAAAQAAAAEA="

    def test_embedding_usage(self):
        """Test EmbeddingUsage model."""
        usage = EmbeddingUsage(
            prompt_tokens=10,
            total_tokens=10,
        )
        assert usage.prompt_tokens == 10
        assert usage.total_tokens == 10

    def test_embedding_response(self):
        """Test EmbeddingResponse model."""
        response = EmbeddingResponse(
            data=[
                EmbeddingData(index=0, embedding=[0.1, 0.2]),
                EmbeddingData(index=1, embedding=[0.3, 0.4]),
            ],
            model="all-MiniLM-L6-v2",
            usage=EmbeddingUsage(prompt_tokens=5, total_tokens=5),
        )
        assert response.object == "list"
        assert len(response.data) == 2
        assert response.model == "all-MiniLM-L6-v2"


class TestEmbeddingUtils:
    """Tests for embedding utility functions."""

    def test_encode_embedding_base64(self):
        """Test base64 encoding of embeddings."""
        embedding = [0.0, 1.0, 2.0, 3.0]
        encoded = encode_embedding_base64(embedding)

        # Decode and verify
        decoded = base64.b64decode(encoded)
        values = struct.unpack(f"<{len(embedding)}f", decoded)
        assert list(values) == embedding

    def test_encode_embedding_base64_empty(self):
        """Test base64 encoding of empty embedding."""
        encoded = encode_embedding_base64([])
        assert encoded == ""

    def test_truncate_embedding_shorter_than_dimensions(self):
        """Test truncation when embedding is shorter than target."""
        embedding = [0.1, 0.2, 0.3]
        result = truncate_embedding(embedding, 5)
        assert result == embedding

    def test_truncate_embedding_exact_dimensions(self):
        """Test truncation when embedding equals target dimensions."""
        embedding = [0.1, 0.2, 0.3]
        result = truncate_embedding(embedding, 3)
        assert result == embedding

    def test_truncate_embedding_with_renormalization(self):
        """Test truncation with proper renormalization."""
        # Create a unit vector [0.6, 0.8, 0.0] (norm = 1.0)
        embedding = [0.6, 0.8, 0.0]

        # Truncate to 2 dimensions
        result = truncate_embedding(embedding, 2)

        # Should be [0.6, 0.8] renormalized
        # Original truncated: [0.6, 0.8], norm = sqrt(0.36 + 0.64) = 1.0
        # So no change needed
        assert len(result) == 2
        assert abs(result[0] - 0.6) < 1e-6
        assert abs(result[1] - 0.8) < 1e-6

        # Verify it's still unit length
        norm = math.sqrt(sum(x * x for x in result))
        assert abs(norm - 1.0) < 1e-6

    def test_truncate_embedding_renormalization_needed(self):
        """Test truncation when renormalization changes the values."""
        # Create a vector [1, 1, 1] / sqrt(3) = [0.577, 0.577, 0.577]
        original_norm = math.sqrt(3)
        embedding = [1.0 / original_norm] * 3

        # Truncate to 2 dimensions
        result = truncate_embedding(embedding, 2)

        # The truncated vector [0.577, 0.577] has norm = sqrt(2) * 0.577 = 0.816
        # After renormalization, should be [1/sqrt(2), 1/sqrt(2)]
        expected = 1.0 / math.sqrt(2)
        assert len(result) == 2
        assert abs(result[0] - expected) < 1e-6
        assert abs(result[1] - expected) < 1e-6

    def test_truncate_embedding_zero_vector(self):
        """Test truncation of zero vector."""
        embedding = [0.0, 0.0, 0.0]
        result = truncate_embedding(embedding, 2)
        assert result == [0.0, 0.0]

    def test_normalize_input_string(self):
        """Test normalizing string input to list."""
        result = normalize_input("Hello")
        assert result == ["Hello"]

    def test_normalize_input_list(self):
        """Test normalizing list input."""
        result = normalize_input(["Hello", "World"])
        assert result == ["Hello", "World"]

    def test_normalize_embedding_items(self):
        """Test structured embedding items normalization."""
        result = normalize_embedding_items(
            [
                EmbeddingInputItem(text="hello"),
                EmbeddingInputItem(image="https://example.com/image.jpg"),
                EmbeddingInputItem(
                    text="hello",
                    image="https://example.com/image.jpg",
                ),
            ]
        )
        assert result == [
            {"text": "hello"},
            {"image": "https://example.com/image.jpg"},
            {
                "text": "hello",
                "image": "https://example.com/image.jpg",
            },
        ]

    def test_count_tokens_with_encode(self):
        """Test token counting with tokenizer that has encode method."""
        mock_processor = MagicMock()
        mock_processor.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens

        count = count_tokens(mock_processor, ["Hello", "World"])
        assert count == 10  # 5 tokens * 2 texts

    def test_count_tokens_with_nested_tokenizer(self):
        """Test token counting with processor that has nested tokenizer."""
        mock_processor = MagicMock(spec=[])  # No encode method
        mock_processor.tokenizer = MagicMock()
        mock_processor.tokenizer.encode.return_value = [1, 2, 3]  # 3 tokens

        count = count_tokens(mock_processor, ["Test"])
        assert count == 3

    def test_count_tokens_fallback(self):
        """Test token counting fallback for unknown processor type."""
        mock_processor = MagicMock(spec=[])  # No encode or tokenizer

        count = count_tokens(mock_processor, ["Hello world test"])
        # Fallback: 3 words + 2 special tokens = 5
        assert count == 5


class TestModelDiscoveryEmbedding:
    """Tests for embedding model detection."""

    def test_detect_bert_model(self, tmp_path):
        """Test detection of BERT embedding model."""
        config = {
            "model_type": "bert",
            "architectures": ["BertModel"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "embedding"

    def test_detect_xlm_roberta_model(self, tmp_path):
        """Test detection of XLM-RoBERTa embedding model."""
        config = {
            "model_type": "xlm-roberta",
            "architectures": ["XLMRobertaModel"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "embedding"

    def test_detect_modernbert_model(self, tmp_path):
        """Test detection of ModernBERT embedding model."""
        config = {
            "model_type": "modernbert",
            "architectures": ["ModernBertModel"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "embedding"

    def test_detect_siglip_model(self, tmp_path):
        """Test detection of SigLIP vision-language embedding model."""
        config = {
            "model_type": "siglip",
            "architectures": ["SiglipModel"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "embedding"

    def test_detect_qwen3_embedding_model(self, tmp_path):
        """Test detection of Qwen3 embedding model."""
        config = {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForTextEmbedding"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "embedding"

    def test_detect_embedding_by_architecture_only(self, tmp_path):
        """Test detection by architecture when model_type is unknown."""
        config = {
            "model_type": "custom-bert",
            "architectures": ["BertModel"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "embedding"

    def test_llm_not_detected_as_embedding(self, tmp_path):
        """Test that LLM models are not detected as embedding."""
        config = {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "llm"

    def test_qwen_llm_not_detected_as_embedding(self, tmp_path):
        """Test that Qwen LLM is not detected as embedding model."""
        config = {
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "llm"

    def test_detect_reranker_model(self, tmp_path):
        """Test detection of reranker model."""
        config = {
            "model_type": "modernbert",
            "architectures": ["ModernBertForSequenceClassification"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "reranker"

    def test_detect_xlm_roberta_reranker(self, tmp_path):
        """Test detection of XLM-RoBERTa reranker model."""
        config = {
            "model_type": "xlm-roberta",
            "architectures": ["XLMRobertaForSequenceClassification"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "reranker"

    def test_no_config_defaults_to_llm(self, tmp_path):
        """Test that missing config.json defaults to LLM."""
        assert detect_model_type(tmp_path) == "llm"


class TestExtractEmbeddingsArray:
    """Tests for _extract_embeddings_array method."""

    def test_extract_text_embeds(self):
        """Test extraction from text_embeds field."""
        import mlx.core as mx
        from omlx.models.embedding import MLXEmbeddingModel

        model = MLXEmbeddingModel("test-model")
        outputs = MagicMock(spec=[])
        outputs.text_embeds = mx.array([[0.1, 0.2]])
        outputs.pooler_output = None
        outputs.last_hidden_state = None

        result = model._extract_embeddings_array(outputs)
        assert result is outputs.text_embeds

    def test_extract_pooler_output(self):
        """Test extraction from pooler_output when text_embeds is absent."""
        import mlx.core as mx
        from omlx.models.embedding import MLXEmbeddingModel

        model = MLXEmbeddingModel("test-model")
        outputs = MagicMock(spec=[])
        outputs.pooler_output = mx.array([[0.3, 0.4]])
        outputs.last_hidden_state = None

        result = model._extract_embeddings_array(outputs)
        assert result is outputs.pooler_output

    def test_extract_last_hidden_state_mean_pool(self):
        """Test mean pooling fallback from last_hidden_state."""
        import mlx.core as mx
        from omlx.models.embedding import MLXEmbeddingModel

        model = MLXEmbeddingModel("test-model")
        outputs = MagicMock(spec=[])
        outputs.last_hidden_state = mx.ones((1, 4, 3))

        result = model._extract_embeddings_array(outputs)
        mx.eval(result)
        assert result.shape == (1, 3)

    def test_extract_raises_when_no_fields(self):
        """Test ValueError when no embedding fields are present."""
        from omlx.models.embedding import MLXEmbeddingModel

        model = MLXEmbeddingModel("test-model")
        outputs = MagicMock(spec=[])

        with pytest.raises(ValueError, match="expected embedding fields"):
            model._extract_embeddings_array(outputs)

    def test_extract_text_embeds_3d_mean_pool(self):
        """Per-token text_embeds (e.g. ModernBERT MaskedLM) should be mean pooled."""
        import mlx.core as mx
        from omlx.models.embedding import MLXEmbeddingModel

        model = MLXEmbeddingModel("test-model")
        outputs = MagicMock(spec=[])
        # Shape (batch=1, seq_len=2, hidden=2). Mean over axis=1 → [[0.2, 0.3]].
        outputs.text_embeds = mx.array([[[0.1, 0.2], [0.3, 0.4]]])
        outputs.pooler_output = None
        outputs.last_hidden_state = None

        result = model._extract_embeddings_array(outputs)
        mx.eval(result)
        assert result.shape == (1, 2)
        assert result.tolist()[0] == pytest.approx([0.2, 0.3])

    def test_extract_pooler_output_3d_mean_pool(self):
        """Per-token pooler_output should also be mean pooled to 2D."""
        import mlx.core as mx
        from omlx.models.embedding import MLXEmbeddingModel

        model = MLXEmbeddingModel("test-model")
        outputs = MagicMock(spec=[])
        outputs.pooler_output = mx.ones((2, 4, 3))
        outputs.last_hidden_state = None

        result = model._extract_embeddings_array(outputs)
        mx.eval(result)
        assert result.shape == (2, 3)


class TestEmbeddingCompileFallback:
    """Tests for embedding compile path fallback behavior."""

    def test_compiled_path_fallback_on_failure(self):
        """Test that embed() falls back to eager when compiled path raises."""
        import mlx.core as mx
        from omlx.models.embedding import MLXEmbeddingModel

        class StandardTokenizer:
            def encode(self, text, add_special_tokens=True):
                del text, add_special_tokens
                return [1, 2, 3]

        class StandardProcessor:
            def __init__(self):
                self._tokenizer = StandardTokenizer()

        model = MLXEmbeddingModel("test-model")
        model._loaded = True
        model._is_compiled = True
        model._compiled_embed = MagicMock(side_effect=RuntimeError("compile fail"))
        model.model = MagicMock()
        model.processor = StandardProcessor()

        # Mock generate to return outputs with text_embeds
        mock_outputs = MagicMock()
        mock_outputs.text_embeds = mx.array([[0.1, 0.2, 0.3]])
        mock_outputs.pooler_output = None
        mock_outputs.last_hidden_state = None

        with patch("mlx_embeddings.generate", return_value=mock_outputs):
            with patch("mlx_embeddings.utils.prepare_inputs"):
                result = model.embed(["test"])

        assert len(result.embeddings) == 1
        assert result.embeddings[0] == pytest.approx([0.1, 0.2, 0.3], abs=1e-5)

    def test_is_compiled_false_uses_eager_path(self):
        """Test that embed() uses eager path when _is_compiled is False."""
        import mlx.core as mx
        from omlx.models.embedding import MLXEmbeddingModel

        model = MLXEmbeddingModel("test-model")
        model._loaded = True
        model._is_compiled = False
        model._compiled_embed = None
        model.model = MagicMock()
        model.processor = MagicMock(spec=[])

        mock_outputs = MagicMock(spec=[])
        mock_outputs.text_embeds = mx.array([[0.5, 0.6]])
        mock_outputs.pooler_output = None
        mock_outputs.last_hidden_state = None

        with patch("mlx_embeddings.generate", return_value=mock_outputs):
            result = model.embed(["test"])

        assert len(result.embeddings) == 1

    def test_default_max_length_uses_model_config(self):
        """Omitted max_length should use model context metadata, not 512."""
        import mlx.core as mx
        from omlx.models.embedding import MLXEmbeddingModel

        model = MLXEmbeddingModel("test-model")
        model._loaded = True
        model._is_compiled = False
        model._compiled_embed = None
        model.model = SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=40960)
        )
        model.processor = SimpleNamespace()

        mock_outputs = MagicMock(spec=[])
        mock_outputs.text_embeds = mx.array([[0.5, 0.6]])
        mock_outputs.pooler_output = None
        mock_outputs.last_hidden_state = None

        with patch("mlx_embeddings.generate", return_value=mock_outputs) as generate:
            model.embed(["test"])

        assert generate.call_args.kwargs["max_length"] == 40960

    def test_default_max_length_uses_tokenizer_config_fallback(self):
        """Tokenizer model_max_length is used when model config lacks a limit."""
        import mlx.core as mx
        from omlx.models.embedding import MLXEmbeddingModel

        model = MLXEmbeddingModel("test-model")
        model._loaded = True
        model._is_compiled = False
        model._compiled_embed = None
        model.model = SimpleNamespace(config=SimpleNamespace())
        model.processor = SimpleNamespace(model_max_length=8192)

        mock_outputs = MagicMock(spec=[])
        mock_outputs.text_embeds = mx.array([[0.5, 0.6]])
        mock_outputs.pooler_output = None
        mock_outputs.last_hidden_state = None

        with patch("mlx_embeddings.generate", return_value=mock_outputs) as generate:
            model.embed(["test"])

        assert generate.call_args.kwargs["max_length"] == 8192

    def test_unknown_default_max_length_falls_back_to_512(self):
        """Keep a conservative final fallback when no metadata exists."""
        import mlx.core as mx
        from omlx.models.embedding import MLXEmbeddingModel

        model = MLXEmbeddingModel("test-model")
        model._loaded = True
        model._is_compiled = False
        model._compiled_embed = None
        model.model = SimpleNamespace(config=SimpleNamespace())
        model.processor = SimpleNamespace()

        mock_outputs = MagicMock(spec=[])
        mock_outputs.text_embeds = mx.array([[0.5, 0.6]])
        mock_outputs.pooler_output = None
        mock_outputs.last_hidden_state = None

        with patch("mlx_embeddings.generate", return_value=mock_outputs) as generate:
            model.embed(["test"])

        assert generate.call_args.kwargs["max_length"] == 512

    def test_explicit_max_length_is_respected(self):
        """Explicit max_length should override metadata."""
        import mlx.core as mx
        from omlx.models.embedding import MLXEmbeddingModel

        model = MLXEmbeddingModel("test-model")
        model._loaded = True
        model._is_compiled = False
        model._compiled_embed = None
        model.model = SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=40960)
        )
        model.processor = SimpleNamespace()

        mock_outputs = MagicMock(spec=[])
        mock_outputs.text_embeds = mx.array([[0.5, 0.6]])
        mock_outputs.pooler_output = None
        mock_outputs.last_hidden_state = None

        with patch("mlx_embeddings.generate", return_value=mock_outputs) as generate:
            model.embed(["test"], max_length=1024)

        assert generate.call_args.kwargs["max_length"] == 1024

    def test_custom_processor_compiled_path_uses_prepare_embedding_inputs(self):
        """Custom embedding processors should use their own prepare API."""
        import mlx.core as mx
        from omlx.models.embedding import MLXEmbeddingModel

        model = MLXEmbeddingModel("test-model")
        model._loaded = True
        model._is_compiled = True
        model._compiled_embed = MagicMock(return_value=mx.array([[0.1, 0.2]]))
        model.model = MagicMock()

        processor = MagicMock(spec=[])
        processor.prepare_embedding_inputs = MagicMock(
            return_value={
                "input_ids": mx.array([[1, 2, 3]]),
                "attention_mask": mx.array([[1, 1, 1]]),
            }
        )
        model.processor = processor

        with patch("mlx_embeddings.generate") as mock_generate:
            result = model.embed(["hello world"])

        processor.prepare_embedding_inputs.assert_called_once_with(
            [{"text": "hello world"}], return_tensors="mlx"
        )
        mock_generate.assert_not_called()
        assert result.embeddings[0] == pytest.approx([0.1, 0.2], abs=1e-5)

    def test_custom_processor_eager_path_bypasses_generate(self):
        """Custom embedding processors should bypass mlx_embeddings.generate()."""
        import mlx.core as mx
        from omlx.models.embedding import MLXEmbeddingModel

        model = MLXEmbeddingModel("test-model")
        model._loaded = True
        model._is_compiled = False
        model._compiled_embed = None

        mock_outputs = MagicMock(spec=[])
        mock_outputs.text_embeds = mx.array([[0.3, 0.4, 0.5]])
        mock_outputs.pooler_output = None
        mock_outputs.last_hidden_state = None
        model.model = MagicMock(return_value=mock_outputs)

        processor = MagicMock(spec=[])
        processor.prepare_embedding_inputs = MagicMock(
            return_value={
                "input_ids": mx.array([[4, 5, 6]]),
                "attention_mask": mx.array([[1, 1, 1]]),
            }
        )
        model.processor = processor

        with patch("mlx_embeddings.generate") as mock_generate:
            result = model.embed(["hello world"])

        processor.prepare_embedding_inputs.assert_called_once_with(
            [{"text": "hello world"}], return_tensors="mlx"
        )
        mock_generate.assert_not_called()
        model.model.assert_called_once()
        assert result.embeddings[0] == pytest.approx([0.3, 0.4, 0.5], abs=1e-5)

    def test_custom_processor_eager_path_remaps_input_ids_for_inputs_signature(self):
        """Models that accept `inputs` instead of `input_ids` should still work."""
        import mlx.core as mx
        from omlx.models.embedding import MLXEmbeddingModel

        class InputsOnlyModel:
            def __call__(self, inputs, attention_mask=None):
                assert inputs.tolist() == [[4, 5, 6]]
                assert attention_mask.tolist() == [[1, 1, 1]]

                outputs = MagicMock(spec=[])
                outputs.text_embeds = mx.array([[0.7, 0.8, 0.9]])
                outputs.pooler_output = None
                outputs.last_hidden_state = None
                return outputs

        model = MLXEmbeddingModel("test-model")
        model._loaded = True
        model._is_compiled = False
        model._compiled_embed = None
        model.model = InputsOnlyModel()
        model._detect_input_key_remapping()

        processor = MagicMock(spec=[])
        processor.prepare_embedding_inputs = MagicMock(
            return_value={
                "input_ids": mx.array([[4, 5, 6]]),
                "attention_mask": mx.array([[1, 1, 1]]),
            }
        )
        model.processor = processor

        with patch("mlx_embeddings.generate") as mock_generate:
            result = model.embed(["hello world"])

        mock_generate.assert_not_called()
        assert result.embeddings[0] == pytest.approx([0.7, 0.8, 0.9], abs=1e-5)

    def test_custom_processor_receives_image_items_unchanged(self):
        """Custom processors should receive raw image strings unchanged."""
        import mlx.core as mx
        from omlx.models.embedding import MLXEmbeddingModel

        model = MLXEmbeddingModel("test-model")
        model._loaded = True
        model._is_compiled = False
        model._compiled_embed = None

        mock_outputs = MagicMock(spec=[])
        mock_outputs.text_embeds = mx.array([[0.3, 0.4, 0.5]])
        mock_outputs.pooler_output = None
        mock_outputs.last_hidden_state = None
        model.model = MagicMock(return_value=mock_outputs)

        processor = MagicMock(spec=[])
        processor.prepare_embedding_inputs = MagicMock(
            return_value={
                "input_ids": mx.array([[4, 5, 6]]),
                "attention_mask": mx.array([[1, 1, 1]]),
            }
        )
        model.processor = processor

        inputs = [
            {"text": "hello"},
            {"image": "https://example.com/image.jpg"},
            {
                "text": "hello",
                "image": "https://example.com/image.jpg",
            },
        ]
        result = model.embed(inputs)

        processor.prepare_embedding_inputs.assert_called_once_with(
            inputs, return_tensors="mlx"
        )
        assert result.embeddings[0] == pytest.approx([0.3, 0.4, 0.5], abs=1e-5)

    def test_custom_processor_counts_image_only_tokens_from_prepared_inputs(self):
        """Image-only custom processor inputs should contribute to usage stats."""
        import mlx.core as mx
        from omlx.models.embedding import MLXEmbeddingModel

        model = MLXEmbeddingModel("test-model")
        model._loaded = True
        model._is_compiled = False
        model._compiled_embed = None

        mock_outputs = MagicMock(spec=[])
        mock_outputs.text_embeds = mx.array([[0.3, 0.4, 0.5]])
        mock_outputs.pooler_output = None
        mock_outputs.last_hidden_state = None
        model.model = MagicMock(return_value=mock_outputs)

        processor = MagicMock(spec=[])
        processor.prepare_embedding_inputs = MagicMock(
            return_value={
                "input_ids": mx.array([[11, 12, 13, 14]]),
                "attention_mask": mx.array([[1, 1, 1, 1]]),
            }
        )
        model.processor = processor

        result = model.embed([{"image": "https://example.com/image.jpg"}])

        assert result.total_tokens == 4

    def test_standard_processor_rejects_image_inputs(self):
        """Standard text embedding processors should reject image items."""
        from omlx.models.embedding import MLXEmbeddingModel

        model = MLXEmbeddingModel("test-model")
        model._loaded = True
        model._is_compiled = False
        model._compiled_embed = None
        model.model = MagicMock()
        model.processor = MagicMock()

        with pytest.raises(ValueError, match="does not support image inputs"):
            model.embed([{"image": "https://example.com/image.jpg"}])

    def test_try_compile_respects_disable_env(self, monkeypatch):
        """OMLX_EMBEDDING_COMPILE=0 should skip mx.compile for root-cause probes."""
        from omlx.models.embedding import MLXEmbeddingModel

        monkeypatch.setenv("OMLX_EMBEDDING_COMPILE", "0")
        model = MLXEmbeddingModel("test-model")
        model.model = MagicMock()

        with patch("omlx.models.embedding.mx") as mock_mx:
            result = model._try_compile()

        assert result is False
        assert model._compiled_embed is None
        mock_mx.compile.assert_not_called()

    def test_close_releases_compiled_model_and_processor_resources(self):
        """close() should drop wrapper references before clearing MLX caches."""
        from omlx.models.embedding import MLXEmbeddingModel

        model = MLXEmbeddingModel("test-model")
        model.model = MagicMock()
        model.processor = MagicMock()
        model._loaded = True
        model._hidden_size = 384
        model._using_native = False
        model._is_compiled = True
        model._compiled_embed = MagicMock()
        model._remap_input_ids_to_inputs = True

        with patch("omlx.models.embedding.gc.collect") as collect, \
             patch("omlx.models.embedding.mx") as mock_mx, \
             patch(
                 "omlx.models.embedding.clear_thread_compile_cache"
             ) as clear_compile_cache:
            model.close()

        assert model.model is None
        assert model.processor is None
        assert model._compiled_embed is None
        assert model._loaded is False
        assert model._hidden_size is None
        assert model._using_native is False
        assert model._is_compiled is False
        assert model._remap_input_ids_to_inputs is False
        mock_mx.synchronize.assert_called_once()
        mock_mx.clear_cache.assert_called_once()
        clear_compile_cache.assert_called_once()
        assert collect.call_count == 2


class TestEmbeddingEngine:
    """Tests for EmbeddingEngine."""

    def test_engine_lifecycle(self):
        """Test engine start and stop lifecycle."""
        import asyncio
        from omlx.engine.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model")

        # Mock the MLXEmbeddingModel
        with patch("omlx.engine.embedding.MLXEmbeddingModel") as MockModel:
            mock_model = MagicMock()
            mock_model.hidden_size = 384
            MockModel.return_value = mock_model

            asyncio.run(engine.start())

            MockModel.assert_called_once_with("test-model", trust_remote_code=False)
            mock_model.load.assert_called_once()

            asyncio.run(engine.stop())
            mock_model.close.assert_called_once()
            assert engine._model is None

    def test_engine_embed(self):
        """Test embedding generation through engine."""
        import asyncio
        from omlx.engine.embedding import EmbeddingEngine
        from omlx.models.embedding import EmbeddingOutput

        engine = EmbeddingEngine("test-model")

        with patch("omlx.engine.embedding.MLXEmbeddingModel") as MockModel:
            mock_model = MagicMock()
            mock_model.embed.return_value = EmbeddingOutput(
                embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                total_tokens=10,
                dimensions=3,
            )
            MockModel.return_value = mock_model

            asyncio.run(engine.start())
            result = asyncio.run(engine.embed(["Hello", "World"]))

            assert len(result.embeddings) == 2
            assert result.total_tokens == 10
            assert result.dimensions == 3

    def test_engine_not_started_raises_error(self):
        """Test that embed raises error if engine not started."""
        import asyncio
        from omlx.engine.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model")

        with pytest.raises(RuntimeError, match="Engine not started"):
            asyncio.run(engine.embed(["Hello"]))

    def test_engine_get_stats(self):
        """Test engine statistics."""
        from omlx.engine.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model")

        stats = engine.get_stats()
        assert stats["model_name"] == "test-model"
        assert stats["loaded"] is False

    def test_engine_uses_scheduler_embedding_batch_size(self):
        """Embedding chunk size should follow shared scheduler config."""
        from omlx.engine.embedding import EmbeddingEngine
        from omlx.scheduler import SchedulerConfig

        engine = EmbeddingEngine(
            "test-model",
            scheduler_config=SchedulerConfig(
                completion_batch_size=6,
                embedding_batch_size=4,
            ),
        )

        assert engine.get_stats()["batch_size"] == 4

    def test_engine_ignores_scheduler_completion_batch_size(self):
        """Completion batching should not affect embedding forward chunks."""
        from omlx.engine.embedding import EmbeddingEngine
        from omlx.scheduler import SchedulerConfig

        engine = EmbeddingEngine(
            "test-model",
            scheduler_config=SchedulerConfig(completion_batch_size=6),
        )

        assert engine.get_stats()["batch_size"] == 32

    def test_engine_preserves_positional_batch_size_argument(self):
        """Keep EmbeddingEngine(model, trust_remote_code, batch_size) working."""
        from omlx.engine.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model", False, 3)

        assert engine.get_stats()["batch_size"] == 3

    def test_engine_get_model_info_not_loaded(self):
        """Test get_model_info when model is not loaded."""
        from omlx.engine.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model")

        info = engine.get_model_info()
        assert info["loaded"] is False
        assert info["model_name"] == "test-model"

    def test_engine_repr(self):
        """Test engine string representation."""
        from omlx.engine.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model")

        repr_str = repr(engine)
        assert "test-model" in repr_str
        assert "stopped" in repr_str

    def test_engine_properties(self):
        """Test engine property accessors."""
        import asyncio
        from omlx.engine.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model")

        # Not loaded
        assert engine.processor is None
        assert engine.hidden_size is None

        # After loading
        with patch("omlx.engine.embedding.MLXEmbeddingModel") as MockModel:
            mock_model = MagicMock()
            mock_model.processor = MagicMock()
            mock_model.hidden_size = 384
            MockModel.return_value = mock_model

            asyncio.run(engine.start())

            assert engine.processor is mock_model.processor
            assert engine.hidden_size == 384

    def test_engine_clears_metal_cache_after_embed(self):
        """Metal cache should be cleared after every embed request (#684)."""
        engine = EmbeddingEngine("test-model")

        with patch("omlx.engine.embedding.MLXEmbeddingModel") as MockModel, \
             patch("omlx.engine.embedding.mx") as mock_mx:
            mock_model = MagicMock()
            mock_model.embed.return_value = EmbeddingOutput(
                embeddings=[[0.1, 0.2]],
                total_tokens=5,
                dimensions=2,
            )
            MockModel.return_value = mock_model

            asyncio.run(engine.start())
            asyncio.run(engine.embed(["Hello"]))

            mock_mx.synchronize.assert_called_once()
            mock_mx.clear_cache.assert_called_once()

    def test_engine_clears_metal_cache_per_concurrent_request(self):
        """Cache clear must fire per request even under concurrency (#684 regression).

        The earlier fix gated the clear on `_active_count == 0`, which never
        triggered under steady concurrent RAG indexing loads. This asserts
        every request clears, not just the last one.
        """
        engine = EmbeddingEngine("test-model")
        concurrency = 4

        with patch("omlx.engine.embedding.MLXEmbeddingModel") as MockModel, \
             patch("omlx.engine.embedding.mx") as mock_mx:
            mock_model = MagicMock()
            mock_model.embed.return_value = EmbeddingOutput(
                embeddings=[[0.1, 0.2]],
                total_tokens=5,
                dimensions=2,
            )
            MockModel.return_value = mock_model

            async def run_concurrent():
                await engine.start()
                await asyncio.gather(
                    *(engine.embed([f"text-{i}"]) for i in range(concurrency))
                )

            asyncio.run(run_concurrent())

            assert mock_mx.synchronize.call_count == concurrency
            assert mock_mx.clear_cache.call_count == concurrency

    def test_engine_chunks_large_embedding_requests_and_clears_each_chunk(self):
        """Large embedding requests should not hold the whole batch in MLX memory."""
        engine = EmbeddingEngine("test-model", batch_size=2)

        def embed_side_effect(inputs, **kwargs):
            return EmbeddingOutput(
                embeddings=[[float(text.rsplit("-", 1)[-1])] for text in inputs],
                total_tokens=len(inputs),
                dimensions=1,
            )

        with patch("omlx.engine.embedding.MLXEmbeddingModel") as MockModel, \
             patch("omlx.engine.embedding.mx") as mock_mx:
            mock_model = MagicMock()
            mock_model.embed.side_effect = embed_side_effect
            MockModel.return_value = mock_model

            asyncio.run(engine.start())
            result = asyncio.run(
                engine.embed([f"text-{i}" for i in range(5)])
            )

            assert result.embeddings == [[0.0], [1.0], [2.0], [3.0], [4.0]]
            assert result.total_tokens == 5
            assert result.dimensions == 1
            assert [
                call.kwargs["inputs"] for call in mock_model.embed.call_args_list
            ] == [
                ["text-0", "text-1"],
                ["text-2", "text-3"],
                ["text-4"],
            ]
            assert mock_mx.synchronize.call_count == 3
            assert mock_mx.clear_cache.call_count == 3

    def test_engine_snapshots_batch_size_per_request(self):
        """Live batch-size updates must not skip or duplicate active request inputs."""
        engine = EmbeddingEngine("test-model", batch_size=2)
        observed_batches = []

        def embed_side_effect(inputs, **kwargs):
            observed_batches.append(list(inputs))
            if len(observed_batches) == 1:
                engine._batch_size = 1
            return EmbeddingOutput(
                embeddings=[[float(text.rsplit("-", 1)[-1])] for text in inputs],
                total_tokens=len(inputs),
                dimensions=1,
            )

        with patch("omlx.engine.embedding.MLXEmbeddingModel") as MockModel, \
             patch("omlx.engine.embedding.mx"):
            mock_model = MagicMock()
            mock_model.embed.side_effect = embed_side_effect
            MockModel.return_value = mock_model

            asyncio.run(engine.start())
            result = asyncio.run(
                engine.embed([f"text-{i}" for i in range(5)])
            )

            assert result.embeddings == [[0.0], [1.0], [2.0], [3.0], [4.0]]
            assert observed_batches == [
                ["text-0", "text-1"],
                ["text-2", "text-3"],
                ["text-4"],
            ]

    def test_concurrent_large_embedding_requests_interleave_between_chunks(self):
        """One large embedding request should not monopolize the MLX executor."""
        engine = EmbeddingEngine("test-model", batch_size=2)
        observed_chunks = []
        observed_lock = threading.Lock()

        def embed_side_effect(inputs, **kwargs):
            time.sleep(0.01)
            with observed_lock:
                observed_chunks.append(tuple(inputs))
            return EmbeddingOutput(
                embeddings=[[float(text.rsplit("-", 1)[-1])] for text in inputs],
                total_tokens=len(inputs),
                dimensions=1,
            )

        with patch("omlx.engine.embedding.MLXEmbeddingModel") as MockModel, \
             patch("omlx.engine.embedding.mx"):
            mock_model = MagicMock()
            mock_model.embed.side_effect = embed_side_effect
            MockModel.return_value = mock_model

            async def run_concurrent():
                await engine.start()
                return await asyncio.gather(
                    engine.embed([f"a-{i}" for i in range(4)]),
                    engine.embed([f"b-{i}" for i in range(4)]),
                )

            first, second = asyncio.run(run_concurrent())

            assert [row[0] for row in first.embeddings] == [0.0, 1.0, 2.0, 3.0]
            assert [row[0] for row in second.embeddings] == [0.0, 1.0, 2.0, 3.0]
            assert observed_chunks == [
                ("a-0", "a-1"),
                ("b-0", "b-1"),
                ("a-2", "a-3"),
                ("b-2", "b-3"),
            ]


class TestEmbeddingModelsPydantic:
    """Additional Pydantic model tests."""

    def test_embedding_request_defaults(self):
        """Test EmbeddingRequest default values."""
        request = EmbeddingRequest(input="test", model="model-name")

        assert request.encoding_format == "float"
        assert request.dimensions is None
        assert request.max_length is None
        assert request.truncation is True

    def test_embedding_data_defaults(self):
        """Test EmbeddingData default values."""
        data = EmbeddingData(index=0, embedding=[0.1])

        assert data.object == "embedding"

    def test_embedding_response_defaults(self):
        """Test EmbeddingResponse default values."""
        response = EmbeddingResponse(
            data=[],
            model="test",
            usage=EmbeddingUsage(prompt_tokens=0, total_tokens=0)
        )

        assert response.object == "list"

    def test_embedding_request_validation(self):
        """Test EmbeddingRequest validation."""
        # Valid with string input
        request = EmbeddingRequest(input="test", model="model")
        assert request.input == "test"

        # Valid with list input
        request = EmbeddingRequest(input=["a", "b"], model="model")
        assert request.input == ["a", "b"]

        request = EmbeddingRequest(
            items=[EmbeddingInputItem(text="test")], model="model"
        )
        assert request.items[0].text == "test"

    def test_embedding_data_accepts_string_embedding(self):
        """Test EmbeddingData accepts string (base64) embedding."""
        data = EmbeddingData(index=0, embedding="base64string")
        assert data.embedding == "base64string"


@pytest.mark.slow
class TestEmbeddingIntegration:
    """Integration tests requiring actual model loading.

    These tests are marked as slow and require mlx-embeddings to be installed.
    """

    def test_real_embedding_generation(self):
        """Test embedding generation with a real model.

        This test requires a small embedding model to be available.
        Skip if mlx-embeddings is not installed.
        """
        import asyncio
        pytest.importorskip("mlx_embeddings")

        from omlx.engine.embedding import EmbeddingEngine

        # Use a small model for testing
        # This model should be available or downloaded from HuggingFace
        model_name = "mlx-community/all-MiniLM-L6-v2-4bit"

        try:
            engine = EmbeddingEngine(model_name)
            asyncio.run(engine.start())

            result = asyncio.run(engine.embed(["Hello, world!", "How are you?"]))

            # Verify structure
            assert len(result.embeddings) == 2
            assert result.dimensions == 384  # MiniLM-L6-v2 has 384 dims
            assert result.total_tokens > 0

            # Verify embedding values are reasonable (normalized)
            for emb in result.embeddings:
                norm = math.sqrt(sum(x * x for x in emb))
                assert abs(norm - 1.0) < 0.01  # Should be approximately unit length

            asyncio.run(engine.stop())

        except Exception as e:
            pytest.skip(f"Could not load model: {e}")


class TestNativeEmbeddingLoading:
    """Tests for native embedding model loading (without mlx-embeddings)."""

    class MockNativeTokenizer:
        """Minimal tokenizer used only by native-loading tests."""

        def __init__(self, vocab_size: int = 30522):
            self.vocab_size = max(vocab_size, 16)

        def encode(self, text: str, add_special_tokens: bool = True):
            tokens = [abs(hash(token)) % (self.vocab_size - 3) + 3 for token in text.split()]
            if add_special_tokens:
                return [101, *tokens, 102]
            return tokens

        def __call__(
            self,
            texts,
            *,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        ):
            del truncation, return_tensors
            encoded = [self.encode(text, add_special_tokens=True)[:max_length] for text in texts]
            target_len = max(len(ids) for ids in encoded) if padding and encoded else 0
            input_ids = []
            attention_mask = []
            for ids in encoded:
                pad_len = max(target_len - len(ids), 0)
                input_ids.append(ids + [0] * pad_len)
                attention_mask.append([1] * len(ids) + [0] * pad_len)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _write_full_native_checkpoint(self, tmp_path, config):
        """Write a complete native checkpoint for a small embedding model."""
        from mlx.utils import tree_flatten
        from omlx.models.xlm_roberta import Model, ModelArgs
        from safetensors.numpy import save_file

        model_config = ModelArgs(**config)
        model = Model(model_config)
        weights = {name: np.array(value) for name, value in tree_flatten(model.parameters())}
        save_file(weights, str(tmp_path / "model.safetensors"))

    def test_load_native_bert_model(self, tmp_path):
        """Test native loading of BERT embedding model."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from safetensors.numpy import save_file

        # Create minimal BERT model structure
        config = {
            "model_type": "bert",
            "architectures": ["BertModel"],
            "hidden_size": 384,
            "num_hidden_layers": 6,
            "vocab_size": 30522,
            "num_attention_heads": 12,
            "intermediate_size": 1536,
            "max_position_embeddings": 512,
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "pad_token_id": 0,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))

        vocab_size = 30522
        save_file(
            {"embeddings.word_embeddings.weight": np.zeros((1, 1), dtype=np.float32)},
            str(tmp_path / "model.safetensors"),
        )

        from omlx.models.embedding import MLXEmbeddingModel

        model = MLXEmbeddingModel(str(tmp_path))
        tokenizer = self.MockNativeTokenizer(vocab_size=vocab_size)
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ) as mock_from_pretrained, patch(
            "omlx.models.embedding.MLXEmbeddingModel._validate_native_weights",
            return_value=None,
        ) as mock_validate_weights, patch(
            "omlx.models.xlm_roberta.Model.load_weights",
            return_value=None,
        ) as mock_load_weights:
            result = model._load_native()

        assert result is True
        assert model._loaded is True
        assert model._using_native is True
        mock_from_pretrained.assert_called()
        mock_validate_weights.assert_called_once()
        assert mock_load_weights.call_args.kwargs["strict"] is False

    def test_load_native_xlm_roberta_model(self, tmp_path):
        """Test native loading of XLMRoBERTa embedding model."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from safetensors.numpy import save_file

        config = {
            "model_type": "xlm-roberta",
            "architectures": ["XLMRobertaModel"],
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "vocab_size": 250002,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "max_position_embeddings": 514,
            "attention_probs_dropout_prob": 0.1,
            "hidden_dropout_prob": 0.1,
            "pad_token_id": 1,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))

        vocab_size = 250002
        save_file(
            {"embeddings.word_embeddings.weight": np.zeros((1, 1), dtype=np.float32)},
            str(tmp_path / "model.safetensors"),
        )

        from omlx.models.embedding import MLXEmbeddingModel

        model = MLXEmbeddingModel(str(tmp_path))
        tokenizer = self.MockNativeTokenizer(vocab_size=vocab_size)
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ) as mock_from_pretrained, patch(
            "omlx.models.embedding.MLXEmbeddingModel._validate_native_weights",
            return_value=None,
        ) as mock_validate_weights, patch(
            "omlx.models.xlm_roberta.Model.load_weights",
            return_value=None,
        ) as mock_load_weights:
            result = model._load_native()

        assert result is True
        assert model._loaded is True
        assert model._using_native is True
        mock_from_pretrained.assert_called()
        mock_validate_weights.assert_called_once()
        assert mock_load_weights.call_args.kwargs["strict"] is False

    def test_load_native_supports_bfloat16_safetensors(self, tmp_path):
        """Native embedding load must not route bf16 safetensors through NumPy."""
        import mlx.core as mx

        config = {
            "model_type": "xlm-roberta",
            "architectures": ["XLMRobertaModel"],
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "vocab_size": 16,
            "num_attention_heads": 1,
            "intermediate_size": 8,
            "max_position_embeddings": 8,
            "attention_probs_dropout_prob": 0.0,
            "hidden_dropout_prob": 0.0,
            "pad_token_id": 1,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        mx.save_safetensors(
            str(tmp_path / "model.safetensors"),
            {
                "embeddings.word_embeddings.weight": mx.ones(
                    (16, 4), dtype=mx.bfloat16
                )
            },
        )

        from omlx.models.embedding import MLXEmbeddingModel

        model = MLXEmbeddingModel(str(tmp_path))
        tokenizer = self.MockNativeTokenizer(vocab_size=config["vocab_size"])
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ), patch(
            "omlx.models.embedding.MLXEmbeddingModel._validate_native_weights",
            return_value=None,
        ) as mock_validate_weights, patch(
            "omlx.models.xlm_roberta.Model.load_weights",
            return_value=None,
        ) as mock_load_weights:
            result = model._load_native()

        assert result is True
        mock_validate_weights.assert_called_once()
        loaded_weights = dict(mock_load_weights.call_args.args[0])
        assert loaded_weights["embeddings.word_embeddings.weight"].dtype == mx.bfloat16

    def test_load_native_rejects_missing_required_weights(self, tmp_path):
        """Native loading must fail when core transformer weights are missing."""
        from safetensors.numpy import save_file

        config = {
            "model_type": "bert",
            "architectures": ["BertModel"],
            "hidden_size": 384,
            "num_hidden_layers": 2,
            "vocab_size": 30522,
            "num_attention_heads": 12,
            "intermediate_size": 1536,
            "max_position_embeddings": 512,
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "pad_token_id": 0,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))

        save_file(
            {"embeddings.word_embeddings.weight": np.random.randn(30522, 384).astype(np.float32)},
            str(tmp_path / "model.safetensors"),
        )

        from omlx.models.embedding import MLXEmbeddingModel

        model = MLXEmbeddingModel(str(tmp_path))
        tokenizer = self.MockNativeTokenizer(vocab_size=30522)
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ):
            result = model._load_native()

        assert result is False
        assert model._loaded is False

    def test_load_native_rejects_shape_mismatches(self, tmp_path):
        """Native loading must fail when a required weight shape is incompatible."""
        from safetensors.numpy import save_file

        config = {
            "model_type": "bert",
            "architectures": ["BertModel"],
            "hidden_size": 384,
            "num_hidden_layers": 2,
            "vocab_size": 30522,
            "num_attention_heads": 12,
            "intermediate_size": 1536,
            "max_position_embeddings": 512,
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "pad_token_id": 0,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))

        self._write_full_native_checkpoint(tmp_path, config)

        import mlx.core as mx
        from safetensors import safe_open

        weights = {}
        with safe_open(tmp_path / "model.safetensors", framework="mlx") as f:
            for key in f.keys():
                weights[key] = np.array(f.get_tensor(key))

        weights["embeddings.word_embeddings.weight"] = np.random.randn(30523, 384).astype(
            np.float32
        )
        save_file(weights, str(tmp_path / "model.safetensors"))

        from omlx.models.embedding import MLXEmbeddingModel

        model = MLXEmbeddingModel(str(tmp_path))
        tokenizer = self.MockNativeTokenizer(vocab_size=30522)
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ):
            result = model._load_native()

        assert result is False
        assert model._loaded is False

    def test_load_native_falls_back_for_unknown_arch(self, tmp_path):
        """Test that native loading returns False for unsupported architectures."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        # Create config with unknown embedding architecture
        config = {
            "model_type": "custom-embedding",
            "architectures": ["CustomEmbeddingModel"],
            "hidden_size": 512,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))

        from omlx.models.embedding import MLXEmbeddingModel

        model = MLXEmbeddingModel(str(tmp_path))
        result = model._load_native()
        assert result is False
        assert model._loaded is False

    def test_embed_produces_normalized_vectors(self, tmp_path):
        """Test that embed produces L2-normalized embedding vectors."""
        import sys, math
        sys.path.insert(0, str(Path(__file__).parent.parent))

        config = {
            "model_type": "bert",
            "architectures": ["BertModel"],
            "hidden_size": 128,
            "num_hidden_layers": 2,
            "vocab_size": 1000,
            "num_attention_heads": 4,
            "intermediate_size": 512,
            "max_position_embeddings": 512,
            "attention_probs_dropout_prob": 0.0,
            "hidden_dropout_prob": 0.0,
            "pad_token_id": 0,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        vocab_size = config["vocab_size"]

        self._write_full_native_checkpoint(tmp_path, config)

        from omlx.models.embedding import MLXEmbeddingModel

        model = MLXEmbeddingModel(str(tmp_path))
        tokenizer = self.MockNativeTokenizer(vocab_size=vocab_size)
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ):
            model.load()
            output = model.embed(["hello world"])

        # Check normalization
        emb = output.embeddings[0]
        norm = math.sqrt(sum(x * x for x in emb))
        assert abs(norm - 1.0) < 0.01, f"Embedding not normalized: norm={norm}"


class TestGetEmbeddingMaxLength:
    """The server helper that resolves the per-request embedding token cap."""

    def test_request_override_wins(self):
        from omlx import server

        with patch.object(server, "get_max_context_window", return_value=32768):
            assert server.get_embedding_max_length("m", 4096) == 4096

    def test_uses_configured_context_window(self):
        from omlx import server

        with patch.object(server, "get_max_context_window", return_value=32768):
            assert server.get_embedding_max_length("m", None) == 32768

    def test_returns_none_without_window_so_model_resolves(self):
        # No request override and no configured window: defer to the model's
        # own context-length resolution instead of a hard 512 cap (#1687).
        from omlx import server

        with patch.object(server, "get_max_context_window", return_value=None):
            assert server.get_embedding_max_length("m", None) is None


class TestNativeQwen2Embedding:
    """Native Qwen2-decoder embedding adapter (jina-code / gte-Qwen2; #686)."""

    # Tiny Qwen2 config exercising grouped-query attention (4 heads / 2 kv).
    _CONFIG = {
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 128,
        "vocab_size": 128,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
    }

    class MockQwen2Tokenizer:
        """Right-padding tokenizer mirroring the native-path encode contract."""

        def __init__(self, vocab_size: int):
            self.vocab_size = vocab_size

        def encode(self, text: str, add_special_tokens: bool = True):
            return [abs(hash(token)) % self.vocab_size for token in text.split()] or [1]

        def __call__(
            self,
            texts,
            *,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        ):
            del truncation, return_tensors
            encoded = [self.encode(t)[:max_length] for t in texts]
            target = max((len(ids) for ids in encoded), default=0) if padding else 0
            input_ids, attention_mask = [], []
            for ids in encoded:
                pad = max(target - len(ids), 0)
                input_ids.append(ids + [0] * pad)
                attention_mask.append([1] * len(ids) + [0] * pad)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _write_full_qwen2_checkpoint(self, tmp_path, config):
        """Write a complete native Qwen2 checkpoint from the adapter's own params."""
        from mlx.utils import tree_flatten
        from omlx.models.qwen2_embedding import Model, ModelArgs
        from safetensors.numpy import save_file

        model = Model(ModelArgs(**config))
        weights = {
            name: np.array(value) for name, value in tree_flatten(model.parameters())
        }
        save_file(weights, str(tmp_path / "model.safetensors"))

    def _load(self, tmp_path):
        from omlx.models.embedding import MLXEmbeddingModel

        (tmp_path / "config.json").write_text(json.dumps(self._CONFIG))
        self._write_full_qwen2_checkpoint(tmp_path, self._CONFIG)

        model = MLXEmbeddingModel(str(tmp_path))
        tokenizer = self.MockQwen2Tokenizer(vocab_size=self._CONFIG["vocab_size"])
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ):
            model.load()
        return model

    def test_load_native_qwen2_takes_native_path(self, tmp_path):
        """Qwen2ForCausalLM routes through the native adapter, not mlx-embeddings."""
        model = self._load(tmp_path)
        assert model._using_native is True
        assert model._hidden_size == self._CONFIG["hidden_size"]
        # The adapter, not the qwen3/mlx-embeddings fallback.
        from omlx.models.qwen2_embedding import Model as Qwen2EmbeddingModel

        assert isinstance(model.model, Qwen2EmbeddingModel)

    def test_qwen2_embed_shape_and_normalized(self, tmp_path):
        """embed() returns one L2-normalized vector per input at the model dim."""
        model = self._load(tmp_path)
        output = model.embed(["def add(a, b): return a + b", "how to sort a list"])

        assert len(output.embeddings) == 2
        for emb in output.embeddings:
            assert len(emb) == self._CONFIG["hidden_size"]
            norm = math.sqrt(sum(x * x for x in emb))
            assert abs(norm - 1.0) < 1e-3, f"not L2-normalized: norm={norm}"

    def test_qwen2_last_token_pool_is_mask_aware(self, tmp_path):
        """Left- vs right-padding the same sequence yields the same vector.

        A causal decoder with RoPE encodes only relative positions, so the
        final real-token state is padding-side invariant *iff* the pool indexes
        the last non-pad token via the attention mask. A hardcoded ``[:, -1]``
        would read a pad position under right padding and diverge.
        """
        import mlx.core as mx
        from omlx.models.qwen2_embedding import Model, ModelArgs

        mx.random.seed(0)
        model = Model(ModelArgs(**self._CONFIG))
        mx.eval(model.parameters())

        right_ids = mx.array([[5, 9, 7, 0, 0]])
        right_mask = mx.array([[1, 1, 1, 0, 0]])
        left_ids = mx.array([[0, 0, 5, 9, 7]])
        left_mask = mx.array([[0, 0, 1, 1, 1]])

        right = np.array(model(right_ids, right_mask).text_embeds[0].tolist())
        left = np.array(model(left_ids, left_mask).text_embeds[0].tolist())

        # Mask-aware pooling agrees to float32 noise (~1e-4); a hardcoded
        # ``[:, -1]`` pool would read the trailing pad token under right padding
        # and diverge by O(0.1+). 1e-3 sits cleanly between the two regimes.
        assert np.max(np.abs(right - left)) < 1e-3, (
            "last-token pool is not mask-aware: left/right padding diverged"
        )

    def test_qwen2_is_causal_flag_controls_attention(self, tmp_path):
        """is_causal=False makes attention bidirectional (gte-Qwen2 family).

        Under causal attention an earlier token cannot attend to a later one, so
        perturbing the last token leaves earlier hidden states unchanged; under
        bidirectional attention it changes them. This pins the config gate that
        distinguishes jina-code (causal) from gte-Qwen2 (``is_causal: false``).
        """
        import mlx.core as mx
        from omlx.models.qwen2_embedding import Model, ModelArgs

        base = mx.array([[5, 9, 7, 3]])
        perturbed = mx.array([[5, 9, 7, 8]])  # differ only in the LAST token
        mask = mx.array([[1, 1, 1, 1]])

        def first_token_drift(is_causal):
            mx.random.seed(0)
            model = Model(ModelArgs(**{**self._CONFIG, "is_causal": is_causal}))
            mx.eval(model.parameters())
            a = np.array(model.model(base, mask).tolist())[0, 0]
            b = np.array(model.model(perturbed, mask).tolist())[0, 0]
            return float(np.max(np.abs(a - b)))

        assert first_token_drift(is_causal=True) < 1e-6, "causal leaked future token"
        assert first_token_drift(is_causal=False) > 1e-3, "bidirectional did not attend forward"
