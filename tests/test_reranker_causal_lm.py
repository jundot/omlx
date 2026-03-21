# SPDX-License-Identifier: Apache-2.0
"""Tests for CausalLM-based reranker support."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from omlx.models.reranker import MLXRerankerModel, RerankOutput


class TestCausalLMReranker:
    """Tests for CausalLM reranker (e.g., Qwen3-Reranker) functionality."""

    def _make_model_dir(self, tmp_path, name="Qwen3-Reranker-0.6B"):
        """Create a mock model directory with CausalLM reranker config."""
        model_dir = tmp_path / name
        model_dir.mkdir()
        config = {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
        }
        (model_dir / "config.json").write_text(json.dumps(config))
        return model_dir

    def test_validate_architecture_accepts_causal_lm_reranker(self, tmp_path):
        """CausalLM architecture is accepted when directory name contains 'reranker'."""
        model_dir = self._make_model_dir(tmp_path, "Qwen3-Reranker-0.6B")
        model = MLXRerankerModel(str(model_dir))
        # Should not raise
        model._validate_architecture()

    def test_validate_architecture_rejects_plain_causal_lm(self, tmp_path):
        """CausalLM architecture is rejected when directory name lacks reranker hint."""
        model_dir = self._make_model_dir(tmp_path, "Qwen3-0.6B")
        model = MLXRerankerModel(str(model_dir))
        with pytest.raises(ValueError, match="does not contain"):
            model._validate_architecture()

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_rerank_causal_lm_scoring(self, tmp_path):
        """Test _rerank_causal_lm produces correct scores from mocked logits."""
        model_dir = self._make_model_dir(tmp_path)

        model = MLXRerankerModel(str(model_dir))
        model._is_causal_lm = True
        model._loaded = True
        model._token_true_id = 9693  # "yes"
        model._token_false_id = 2152  # "no"
        model._prefix_tokens = [1, 2, 3]
        model._suffix_tokens = [4, 5]

        # Mock tokenizer: return simple token IDs for each document
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": [[10, 11, 12], [20, 21, 22]],
        }
        model.processor = mock_tokenizer

        # Mock model forward pass: return logits where "yes" > "no" for doc 0,
        # and "no" > "yes" for doc 1

        call_count = [0]

        def mock_forward(input_ids):
            vocab_size = 10000
            seq_len = input_ids.shape[1]
            logits = mx.zeros((1, seq_len, vocab_size))
            # Set logits at last position
            last_pos = np.zeros(vocab_size)
            if call_count[0] == 0:
                # Doc 0: yes=5.0, no=0.0 → high relevance
                last_pos[9693] = 5.0
                last_pos[2152] = 0.0
            else:
                # Doc 1: yes=0.0, no=5.0 → low relevance
                last_pos[9693] = 0.0
                last_pos[2152] = 5.0
            call_count[0] += 1
            # Construct logits with the last position set
            logits_np = np.zeros((1, seq_len, vocab_size), dtype=np.float32)
            logits_np[0, -1, :] = last_pos
            return mx.array(logits_np)

        model.model = MagicMock(side_effect=mock_forward)

        result = model._rerank_causal_lm("test query", ["relevant doc", "irrelevant doc"])

        assert isinstance(result, RerankOutput)
        assert len(result.scores) == 2
        # Doc 0 should have high score (yes >> no)
        assert result.scores[0] > 0.9
        # Doc 1 should have low score (no >> yes)
        assert result.scores[1] < 0.1
        # Sorted indices: doc 0 first
        assert result.indices == [0, 1]
        assert result.total_tokens > 0

    def test_rerank_causal_lm_empty_documents(self, tmp_path):
        """Test rerank with empty document list returns empty result."""
        model_dir = self._make_model_dir(tmp_path)
        model = MLXRerankerModel(str(model_dir))
        model._is_causal_lm = True
        model._loaded = True

        result = model.rerank("test query", [])
        assert result.scores == []
        assert result.indices == []
        assert result.total_tokens == 0

    def test_rerank_dispatches_to_causal_lm(self, tmp_path):
        """Test that rerank() dispatches to _rerank_causal_lm when _is_causal_lm is True."""
        model_dir = self._make_model_dir(tmp_path)
        model = MLXRerankerModel(str(model_dir))
        model._is_causal_lm = True
        model._loaded = True

        mock_result = RerankOutput(scores=[0.9], indices=[0], total_tokens=10)
        with patch.object(model, "_rerank_causal_lm", return_value=mock_result) as mock_method:
            result = model.rerank("query", ["doc"])
            mock_method.assert_called_once()
            assert result.scores == [0.9]

    def test_max_length_default_for_causal_lm(self, tmp_path):
        """Test that CausalLM reranker uses 8192 as effective max_length by default."""
        model_dir = self._make_model_dir(tmp_path)
        model = MLXRerankerModel(str(model_dir))
        model._is_causal_lm = True
        model._loaded = True

        mock_result = RerankOutput(scores=[0.5], indices=[0], total_tokens=10)
        with patch.object(model, "_rerank_causal_lm", return_value=mock_result) as mock_method:
            model.rerank("query", ["doc"])
            # max_length=None should use default 8192 for CausalLM
            args, _ = mock_method.call_args
            assert args[2] == 8192  # query, documents, max_length

    def test_max_length_explicit_override(self, tmp_path):
        """Test that explicit max_length is respected even for CausalLM."""
        model_dir = self._make_model_dir(tmp_path)
        model = MLXRerankerModel(str(model_dir))
        model._is_causal_lm = True
        model._loaded = True

        mock_result = RerankOutput(scores=[0.5], indices=[0], total_tokens=10)
        with patch.object(model, "_rerank_causal_lm", return_value=mock_result) as mock_method:
            model.rerank("query", ["doc"], max_length=1024)
            args, _ = mock_method.call_args
            assert args[2] == 1024

    def test_max_length_512_explicit_respected_for_causal_lm(self, tmp_path):
        """Test that explicitly passing max_length=512 is respected (not overridden)."""
        model_dir = self._make_model_dir(tmp_path)
        model = MLXRerankerModel(str(model_dir))
        model._is_causal_lm = True
        model._loaded = True

        mock_result = RerankOutput(scores=[0.5], indices=[0], total_tokens=10)
        with patch.object(model, "_rerank_causal_lm", return_value=mock_result) as mock_method:
            model.rerank("query", ["doc"], max_length=512)
            args, _ = mock_method.call_args
            assert args[2] == 512


class TestRerankerCompileFallback:
    """Tests for reranker compiled path fallback behavior."""

    def _make_model_dir(self, tmp_path, name="bge-reranker-v2-m3"):
        """Create a mock model directory with SequenceClassification config."""
        model_dir = tmp_path / name
        model_dir.mkdir()
        config = {
            "model_type": "modernbert",
            "architectures": ["ModernBertForSequenceClassification"],
        }
        (model_dir / "config.json").write_text(json.dumps(config))
        return model_dir

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_compiled_path_fallback_on_failure(self, tmp_path):
        """Test that _rerank_seq_classification falls back to eager on compile failure."""
        model_dir = self._make_model_dir(tmp_path)
        model = MLXRerankerModel(str(model_dir))
        model._loaded = True
        model._is_causal_lm = False
        model._is_compiled = True
        model._compiled_seq_logits = MagicMock(
            side_effect=RuntimeError("compile fail")
        )

        # Mock processor
        mock_processor = MagicMock()
        mock_processor.return_value = {
            "input_ids": [[1, 2, 3, 4]],
            "attention_mask": [[1, 1, 1, 1]],
        }
        model.processor = mock_processor

        # Mock model to return pooler_output
        mock_outputs = MagicMock(spec=[])
        mock_outputs.pooler_output = mx.array([[0.85]])
        model.model = MagicMock(return_value=mock_outputs)

        result = model._rerank_seq_classification("query", ["doc"])

        assert len(result.scores) == 1
        # Compiled path failed, eager path should have been used
        model.model.assert_called_once()

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_eager_path_when_not_compiled(self, tmp_path):
        """Test that _rerank_seq_classification uses eager path when not compiled."""
        model_dir = self._make_model_dir(tmp_path)
        model = MLXRerankerModel(str(model_dir))
        model._loaded = True
        model._is_causal_lm = False
        model._is_compiled = False
        model._compiled_seq_logits = None

        mock_processor = MagicMock()
        mock_processor.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }
        model.processor = mock_processor

        mock_outputs = MagicMock(spec=[])
        mock_outputs.pooler_output = mx.array([[0.7]])
        model.model = MagicMock(return_value=mock_outputs)

        result = model._rerank_seq_classification("query", ["doc"])

        assert len(result.scores) == 1
        model.model.assert_called_once()

    def test_try_compile_skips_causal_lm(self, tmp_path):
        """Test that _try_compile returns False for causal-lm rerankers."""
        model_dir = tmp_path / "Qwen3-Reranker-0.6B"
        model_dir.mkdir()
        config = {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
        }
        (model_dir / "config.json").write_text(json.dumps(config))

        model = MLXRerankerModel(str(model_dir))
        model._is_causal_lm = True
        model.model = MagicMock()

        result = model._try_compile()

        assert result is False
        assert model._compiled_seq_logits is None
