# SPDX-License-Identifier: Apache-2.0
"""Tests for embedding/reranker engine keepalive and mx.compile integration."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest


class TestCompiledForwardWrapper:
    """Tests for _CompiledForward wrapper class."""

    def test_compiled_forward_delegates_attributes(self):
        """Wrapper should delegate attribute access to the inner module."""
        from omlx.models.embedding import _CompiledForward

        mock_module = MagicMock()
        mock_module.config = MagicMock(hidden_size=384)
        mock_module.__call__ = MagicMock(return_value="output")

        with patch("omlx.models.embedding.mx") as mock_mx:
            mock_mx.compile.return_value = MagicMock(return_value="output")
            wrapper = _CompiledForward(mock_module)

        assert wrapper.config is mock_module.config
        assert wrapper.config.hidden_size == 384

    def test_compiled_forward_call_uses_compiled(self):
        """Wrapper __call__ should use the compiled function."""
        from omlx.models.embedding import _CompiledForward

        mock_module = MagicMock()
        mock_compiled = MagicMock(return_value="compiled_output")

        with patch("omlx.models.embedding.mx") as mock_mx:
            mock_mx.compile.return_value = mock_compiled
            wrapper = _CompiledForward(mock_module)

        result = wrapper("input1", key="val")
        mock_compiled.assert_called_once_with("input1", key="val")
        assert result == "compiled_output"


class TestTryCompile:
    """Tests for _try_compile in model wrappers."""

    def test_try_compile_success(self):
        """Should set _is_compiled=True on successful compilation."""
        from omlx.models.embedding import MLXEmbeddingModel, _CompiledForward

        model = MLXEmbeddingModel("test-model")
        mock_raw_model = MagicMock()
        mock_raw_model.__call__ = MagicMock(return_value=MagicMock())
        model.model = mock_raw_model

        with patch("omlx.models.embedding.mx") as mock_mx:
            mock_mx.compile.return_value = MagicMock(return_value=MagicMock())
            mock_mx.zeros.return_value = MagicMock()
            mock_mx.int32 = "int32"
            result = model._try_compile()

        assert result is True
        assert isinstance(model.model, _CompiledForward)

    def test_try_compile_failure_reverts(self):
        """Should revert to original model on compilation failure."""
        from omlx.models.embedding import MLXEmbeddingModel

        model = MLXEmbeddingModel("test-model")
        original_model = MagicMock()
        model.model = original_model

        with patch("omlx.models.embedding.mx") as mock_mx:
            mock_mx.compile.side_effect = RuntimeError("compile failed")
            result = model._try_compile()

        assert result is False
        assert model.model is original_model


class TestEmbeddingEngineKeepalive:
    """Tests for keepalive integration in EmbeddingEngine."""

    def test_compiled_model_no_keepalive(self):
        """Keepalive should NOT start if mx.compile succeeded."""
        from omlx.engine.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model")

        with patch("omlx.engine.embedding.MLXEmbeddingModel") as MockModel:
            mock_model = MagicMock()
            mock_model._is_compiled = True
            mock_model.hidden_size = 384
            MockModel.return_value = mock_model

            asyncio.run(engine.start())

        assert engine._keepalive_task is None

    def test_uncompiled_model_starts_keepalive(self):
        """Keepalive SHOULD start if mx.compile failed."""
        from omlx.engine.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model")

        with patch("omlx.engine.embedding.MLXEmbeddingModel") as MockModel:
            mock_model = MagicMock()
            mock_model._is_compiled = False
            mock_model.hidden_size = 384
            MockModel.return_value = mock_model

            async def _run():
                await engine.start()
                assert engine._keepalive_task is not None
                await engine.stop()

            asyncio.run(_run())

    def test_keepalive_stops_on_engine_stop(self):
        """Keepalive task should be cancelled on engine stop."""
        from omlx.engine.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model")

        with patch("omlx.engine.embedding.MLXEmbeddingModel") as MockModel:
            mock_model = MagicMock()
            mock_model._is_compiled = False
            MockModel.return_value = mock_model

            async def _run():
                await engine.start()
                assert engine._keepalive_task is not None
                await engine.stop()
                assert engine._keepalive_task is None

            asyncio.run(_run())

    def test_active_requests_tracking(self):
        """_active_requests should be incremented during embed calls."""
        from omlx.engine.embedding import EmbeddingEngine
        from omlx.models.embedding import EmbeddingOutput

        engine = EmbeddingEngine("test-model")

        with patch("omlx.engine.embedding.MLXEmbeddingModel") as MockModel:
            mock_model = MagicMock()
            mock_model._is_compiled = True
            mock_model.embed.return_value = EmbeddingOutput(
                embeddings=[[0.1]], total_tokens=1, dimensions=1,
            )
            MockModel.return_value = mock_model

            async def _run():
                await engine.start()
                assert engine._active_requests == 0
                await engine.embed(["test"])
                assert engine._active_requests == 0  # Back to 0 after completion

            asyncio.run(_run())


class TestRerankerEngineKeepalive:
    """Tests for keepalive integration in RerankerEngine."""

    def test_compiled_model_no_keepalive(self):
        """Keepalive should NOT start if mx.compile succeeded."""
        from omlx.engine.reranker import RerankerEngine

        engine = RerankerEngine("test-model")

        with patch("omlx.engine.reranker.MLXRerankerModel") as MockModel:
            mock_model = MagicMock()
            mock_model._is_compiled = True
            MockModel.return_value = mock_model

            asyncio.run(engine.start())

        assert engine._keepalive_task is None

    def test_uncompiled_model_starts_keepalive(self):
        """Keepalive SHOULD start if mx.compile failed."""
        from omlx.engine.reranker import RerankerEngine

        engine = RerankerEngine("test-model")

        with patch("omlx.engine.reranker.MLXRerankerModel") as MockModel:
            mock_model = MagicMock()
            mock_model._is_compiled = False
            MockModel.return_value = mock_model

            async def _run():
                await engine.start()
                assert engine._keepalive_task is not None
                await engine.stop()

            asyncio.run(_run())

    def test_keepalive_stops_on_engine_stop(self):
        """Keepalive task should be cancelled on engine stop."""
        from omlx.engine.reranker import RerankerEngine

        engine = RerankerEngine("test-model")

        with patch("omlx.engine.reranker.MLXRerankerModel") as MockModel:
            mock_model = MagicMock()
            mock_model._is_compiled = False
            MockModel.return_value = mock_model

            async def _run():
                await engine.start()
                assert engine._keepalive_task is not None
                await engine.stop()
                assert engine._keepalive_task is None

            asyncio.run(_run())
