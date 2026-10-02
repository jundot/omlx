# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-token (late-interaction / MaxSim) embedding output."""

import asyncio
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.api.embedding_models import (
    TokenEmbeddingRequest,
)
from omlx.api.embedding_utils import find_non_finite_token_embeddings
from omlx.engine.embedding import EmbeddingEngine
from omlx.models.embedding import MLXEmbeddingModel, TokenEmbeddingOutput


class _StubModel:
    """A forward pass that returns each token's own id (plus derived columns).

    Deterministic and id-derived, so a token's vector identifies the token and
    the tests can assert exactly which positions were returned.
    """

    def __init__(self, hidden: int = 1):
        self.hidden = hidden
        self.batch_shapes = []

    def __call__(self, **kwargs):
        input_ids = kwargs["input_ids"]
        batch, width = input_ids.shape
        self.batch_shapes.append((batch, width))
        values = input_ids.astype(mx.float32)
        expanded = mx.expand_dims(values, -1)
        if self.hidden > 1:
            expanded = mx.concatenate(
                [expanded + offset for offset in range(self.hidden)], axis=-1
            )
        return SimpleNamespace(last_hidden_state=expanded)


def _stub_backed_model(hidden: int = 1):
    model = MLXEmbeddingModel("stub-token-embedder")
    model._loaded = True
    model.model = _StubModel(hidden)
    return model


class TestTokenEmbeddingRequest:
    """The per-token request contract."""

    def test_accepts_explicit_ids(self):
        request = TokenEmbeddingRequest(model="m", input_ids=[[1, 2, 3]])
        assert request.input_ids == [[1, 2, 3]]

    def test_rejects_empty_batch(self):
        with pytest.raises(Exception):
            TokenEmbeddingRequest(model="m", input_ids=[])

    def test_rejects_empty_sequence(self):
        with pytest.raises(Exception):
            TokenEmbeddingRequest(model="m", input_ids=[[1], []])

    def test_rejects_negative_id(self):
        with pytest.raises(Exception):
            TokenEmbeddingRequest(model="m", input_ids=[[-1]])


class TestNonFiniteTokenEmbeddings:
    """The NaN/Inf guard for nested token output."""

    def test_flags_the_offending_sequence(self):
        bad = [[[0.0, float("nan")], [1.0, 1.0]], [[2.0, 2.0]]]
        assert find_non_finite_token_embeddings(bad) == [0]

    def test_clean_input_passes(self):
        assert find_non_finite_token_embeddings([[[1.0], [2.0]]]) == []


class TestEmbedTokenIds:
    """`MLXEmbeddingModel.embed_token_ids`."""

    def test_returns_ragged_rows(self):
        model = _stub_backed_model(hidden=3)
        out = model.embed_token_ids([[1, 2, 3], [4, 5]])

        assert isinstance(out, TokenEmbeddingOutput)
        assert [len(row) for row in out.token_embeddings] == [3, 2]
        assert all(len(vec) == 3 for row in out.token_embeddings for vec in row)
        assert out.total_tokens == 5
        assert out.dimensions == 3

    def test_pads_the_batch_then_slices_each_row(self):
        model = _stub_backed_model()
        out = model.embed_token_ids([[7, 8, 9], [3]])

        # One padded forward pass, both rows sliced back to their own length.
        assert model.model.batch_shapes == [(2, 3)]
        assert [len(row) for row in out.token_embeddings] == [3, 1]
        assert [vec[0] for vec in out.token_embeddings[0]] == [7.0, 8.0, 9.0]
        assert [vec[0] for vec in out.token_embeddings[1]] == [3.0]

    def test_values_are_the_raw_final_hidden_state(self):
        # No pooling and no L2 normalization: the caller owns both.
        model = _stub_backed_model(hidden=2)
        out = model.embed_token_ids([[1, 2]])
        assert out.token_embeddings == [[[1.0, 2.0], [2.0, 3.0]]]

    def test_rejects_an_empty_sequence(self):
        model = _stub_backed_model()
        with pytest.raises(ValueError):
            model.embed_token_ids([[1, 2], []])

    def test_rejects_a_model_without_token_features(self):
        model = _stub_backed_model()
        model.model = lambda **kwargs: SimpleNamespace(text_embeds=mx.zeros((1, 4)))
        with pytest.raises(ValueError):
            model.embed_token_ids([[1, 2]])


class _StubEngineModel:
    """Engine-level stub: only `embed_token_ids` is exercised."""

    def __init__(self, model):
        self._inner = model

    def embed_token_ids(self, ids_batch):
        return self._inner.embed_token_ids(ids_batch)


class TestEngineEmbedTokenIds:
    """`EmbeddingEngine.embed_token_ids` keeps the caller's order."""

    def test_restores_request_order_after_length_grouping(self):
        engine = EmbeddingEngine("stub-token-embedder", batch_size=16)
        engine._model = _StubEngineModel(_stub_backed_model())

        out = asyncio.run(engine.embed_token_ids([[1, 2, 3], [4]]))

        # Length grouping scores [4] first; the response must still be request order.
        assert [vec[0] for vec in out.token_embeddings[0]] == [1.0, 2.0, 3.0]
        assert [vec[0] for vec in out.token_embeddings[1]] == [4.0]
        assert out.total_tokens == 4

    def test_empty_batch_is_empty(self):
        engine = EmbeddingEngine("stub-token-embedder")
        engine._model = _StubEngineModel(_stub_backed_model())
        out = asyncio.run(engine.embed_token_ids([]))
        assert out.token_embeddings == []
        assert out.total_tokens == 0

    def test_requires_a_started_engine(self):
        engine = EmbeddingEngine("stub-token-embedder")
        with pytest.raises(RuntimeError):
            asyncio.run(engine.embed_token_ids([[1]]))
