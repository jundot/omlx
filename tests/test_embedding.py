# SPDX-License-Identifier: Apache-2.0
"""Tests for embedding functionality."""

import base64
import json
import math
import struct
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.api.embedding_models import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)
from omlx.api.embedding_utils import (
    count_tokens,
    encode_embedding_base64,
    normalize_input,
    truncate_embedding,
)
from omlx.model_discovery import detect_model_type


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


class TestEmbeddingCompileFallback:
    """Tests for embedding compile path fallback behavior."""

    def test_compiled_path_fallback_on_failure(self):
        """Test that embed() falls back to eager when compiled path raises."""
        import mlx.core as mx
        from omlx.models.embedding import MLXEmbeddingModel

        model = MLXEmbeddingModel("test-model")
        model._loaded = True
        model._is_compiled = True
        model._compiled_embed = MagicMock(side_effect=RuntimeError("compile fail"))
        model.model = MagicMock()
        model.processor = MagicMock(spec=[])
        model.processor._tokenizer = MagicMock()

        # Mock generate to return outputs with text_embeds
        mock_outputs = MagicMock(spec=[])
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

            MockModel.assert_called_once_with("test-model")
            mock_model.load.assert_called_once()

            asyncio.run(engine.stop())

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


class TestEmbeddingModelsPydantic:
    """Additional Pydantic model tests."""

    def test_embedding_request_defaults(self):
        """Test EmbeddingRequest default values."""
        request = EmbeddingRequest(input="test", model="model-name")

        assert request.encoding_format == "float"
        assert request.dimensions is None

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

    def test_load_native_bert_model(self, tmp_path):
        """Test native loading of BERT embedding model."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

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

        # Create minimal safetensors file
        import struct
        import numpy as np
        from safetensors.numpy import save_file

        # Create minimal embeddings weight
        vocab_size, hidden_size = 30522, 384
        embeddings = np.random.randn(vocab_size, hidden_size).astype(np.float32)
        save_file({"embeddings.word_embeddings.weight": embeddings}, str(tmp_path / "model.safetensors"))

        from omlx.models.embedding import MLXEmbeddingModel

        model = MLXEmbeddingModel(str(tmp_path))
        result = model._load_native()
        assert result is True
        assert model._loaded is True
        assert model._using_native is True

    def test_load_native_xlm_roberta_model(self, tmp_path):
        """Test native loading of XLMRoBERTa embedding model."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

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

        import numpy as np
        from safetensors.numpy import save_file

        vocab_size, hidden_size = 250002, 768
        embeddings = np.random.randn(vocab_size, hidden_size).astype(np.float32)
        save_file({"embeddings.word_embeddings.weight": embeddings}, str(tmp_path / "model.safetensors"))

        from omlx.models.embedding import MLXEmbeddingModel

        model = MLXEmbeddingModel(str(tmp_path))
        result = model._load_native()
        assert result is True
        assert model._loaded is True
        assert model._using_native is True

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

        import numpy as np
        from safetensors.numpy import save_file

        # Create small model weights
        vocab_size, hidden_size = 1000, 128
        weights = {
            "embeddings.word_embeddings.weight": np.random.randn(vocab_size, hidden_size).astype(np.float32) * 0.02,
            "embeddings.position_embeddings.weight": np.random.randn(512, hidden_size).astype(np.float32) * 0.02,
            "embeddings.token_type_embeddings.weight": np.zeros((1, hidden_size), dtype=np.float32),
            "embeddings.LayerNorm.weight": np.ones(hidden_size, dtype=np.float32),
            "embeddings.LayerNorm.bias": np.zeros(hidden_size, dtype=np.float32),
        }
        save_file(weights, str(tmp_path / "model.safetensors"))

        from omlx.models.embedding import MLXEmbeddingModel

        model = MLXEmbeddingModel(str(tmp_path))
        model.load()
        output = model.embed(["hello world"])

        # Check normalization
        emb = output.embeddings[0]
        norm = math.sqrt(sum(x * x for x in emb))
        assert abs(norm - 1.0) < 0.01, f"Embedding not normalized: norm={norm}"
