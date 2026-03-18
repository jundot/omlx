# SPDX-License-Identifier: Apache-2.0
"""Tests for embedding/reranker engine mx.compile integration."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest


class TestTryCompile:
    """Tests for _try_compile in model wrappers."""

    def test_embedding_try_compile_success(self):
        """_try_compile should return True and set _compiled_embed on success."""
        from omlx.models.embedding import MLXEmbeddingModel

        model = MLXEmbeddingModel("test-model")
        model.model = MagicMock()

        with patch("omlx.models.embedding.mx") as mock_mx:
            mock_compiled_fn = MagicMock(return_value=MagicMock())
            mock_mx.compile.return_value = mock_compiled_fn
            mock_mx.zeros.return_value = MagicMock()
            mock_mx.int32 = "int32"
            result = model._try_compile()

        assert result is True
        assert model._compiled_embed is mock_compiled_fn

    def test_embedding_try_compile_failure(self):
        """_try_compile should return False and clear _compiled_embed on failure."""
        from omlx.models.embedding import MLXEmbeddingModel

        model = MLXEmbeddingModel("test-model")
        model.model = MagicMock()

        with patch("omlx.models.embedding.mx") as mock_mx:
            mock_mx.compile.side_effect = RuntimeError("compile failed")
            result = model._try_compile()

        assert result is False
        assert model._compiled_embed is None


class TestEmbeddingEngineStartStop:
    """Tests for EmbeddingEngine start/stop lifecycle."""

    def test_engine_starts_without_keepalive(self):
        """Engine should start without any background keepalive task."""
        from omlx.engine.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model")

        with patch("omlx.engine.embedding.MLXEmbeddingModel") as MockModel:
            mock_model = MagicMock()
            mock_model._is_compiled = False
            mock_model.hidden_size = 384
            MockModel.return_value = mock_model

            asyncio.run(engine.start())

        assert not hasattr(engine, "_keepalive_task")


class TestRerankerEngineStartStop:
    """Tests for RerankerEngine start/stop lifecycle."""

    def test_engine_starts_without_keepalive(self):
        """Engine should start without any background keepalive task."""
        from omlx.engine.reranker import RerankerEngine

        engine = RerankerEngine("test-model")

        with patch("omlx.engine.reranker.MLXRerankerModel") as MockModel:
            mock_model = MagicMock()
            mock_model._is_compiled = False
            MockModel.return_value = mock_model

            asyncio.run(engine.start())

        assert not hasattr(engine, "_keepalive_task")
