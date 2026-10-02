# SPDX-License-Identifier: Apache-2.0
"""Tests for embedding/reranker engine mx.compile integration."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest


class _MaskBranchingModel:
    """Forward with a Python `if` on a mask-dependent lazy comparison.

    Mirrors mlx-embeddings qwen3's last_token_pool: with attention_mask=None
    the default mask is built from the (static) shape, so it is a tracing
    constant and legal to eval; with a traced attention_mask input the same
    `if` forces an eval during tracing and mx.compile raises (issue #2447).
    """

    def __call__(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = mx.ones(input_ids.shape, dtype=mx.int32)
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            pooled = input_ids[:, -1:]
        else:
            pooled = input_ids[:, :1]
        return SimpleNamespace(text_embeds=pooled.astype(mx.float32))


class _MaskFreeModel:
    """Forward with no data-dependent Python branching — compiles cleanly."""

    def __call__(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = mx.ones(input_ids.shape, dtype=mx.int32)
        summed = (input_ids * attention_mask).sum(axis=1, keepdims=True)
        return SimpleNamespace(text_embeds=summed.astype(mx.float32))


class TestTryCompileMaskProbe:
    """The compile probe must include a traced attention_mask (issue #2447).

    Real requests always carry one (prepare_inputs emits it), so a mask-less
    probe can pass at load while every real request falls back to eager.
    These tests run real mx.compile — no mocks — so they fail if the probe
    stops representing the real request path.
    """

    def test_mask_branching_model_falls_back_at_load(self, monkeypatch):
        from omlx.models.embedding import MLXEmbeddingModel

        # An exported OMLX_EMBEDDING_COMPILE=0 would make _try_compile return
        # False before ever calling mx.compile — a vacuously passing test.
        monkeypatch.delenv("OMLX_EMBEDDING_COMPILE", raising=False)
        model = MLXEmbeddingModel("test-model")
        model.model = _MaskBranchingModel()

        assert model._try_compile() is False
        assert model._compiled_embed is None

    def test_mask_free_model_still_compiles(self, monkeypatch):
        from omlx.models.embedding import MLXEmbeddingModel

        monkeypatch.delenv("OMLX_EMBEDDING_COMPILE", raising=False)
        model = MLXEmbeddingModel("test-model")
        model.model = _MaskFreeModel()

        assert model._try_compile() is True
        assert model._compiled_embed is not None


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
