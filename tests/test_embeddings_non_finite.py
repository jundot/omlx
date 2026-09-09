"""/v1/embeddings must not answer 200 with null-filled vectors (#3507)."""

from __future__ import annotations

import math
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from omlx.api.embedding_utils import find_non_finite_embeddings
from omlx.models.embedding import EmbeddingOutput
from omlx.server import ServerState, app


def test_find_non_finite_embeddings_reports_bad_items_only():
    embeddings = [
        [0.1, 0.2],
        [math.nan, 0.3],
        [0.4, math.inf],
        [-math.inf, 0.0],
        [0.5, 0.6],
    ]
    assert find_non_finite_embeddings(embeddings) == [1, 2, 3]
    assert find_non_finite_embeddings([[0.1], [0.2]]) == []
    assert find_non_finite_embeddings([]) == []


def test_find_non_finite_embeddings_accepts_array_like_rows():
    class _Row:
        def __init__(self, values):
            self._values = values

        def __iter__(self):
            return iter(self._values)

    assert find_non_finite_embeddings([_Row([1.0, 2.0]), _Row([math.nan])]) == [1]


def _post_embeddings(embeddings: list[list[float]]):
    engine = MagicMock()
    engine.embed = AsyncMock(
        return_value=EmbeddingOutput(
            embeddings=embeddings,
            total_tokens=len(embeddings) * 2,
            dimensions=len(embeddings[0]) if embeddings else 0,
        )
    )

    @asynccontextmanager
    async def _acquire(_model):
        yield engine

    state = ServerState()
    with (
        patch("omlx.server._server_state", state),
        patch("omlx.server.get_embedding_engine", AsyncMock(return_value=engine)),
        patch("omlx.server.acquire_embedding_engine", _acquire),
        patch("omlx.server.get_embedding_max_length", return_value=512),
        patch("omlx.server.resolve_model_id", return_value="emb"),
        patch("omlx.server.get_server_metrics", return_value=MagicMock()),
    ):
        client = TestClient(app, raise_server_exceptions=False)
        return client.post(
            "/v1/embeddings",
            json={"model": "emb", "input": ["short", "a much longer input"]},
        )


def test_embeddings_endpoint_rejects_non_finite_vectors():
    response = _post_embeddings([[0.1, 0.2], [math.nan, math.nan]])

    assert response.status_code == 500
    body = response.json()
    # API routes answer with the OpenAI-style error envelope.
    message = body["error"]["message"] if "error" in body else body["detail"]
    assert "non-finite" in message
    assert "[1]" in message


def test_embeddings_endpoint_returns_finite_vectors():
    response = _post_embeddings([[0.1, 0.2], [0.3, 0.4]])

    assert response.status_code == 200
    data = response.json()["data"]
    assert [item["embedding"] for item in data] == [[0.1, 0.2], [0.3, 0.4]]
