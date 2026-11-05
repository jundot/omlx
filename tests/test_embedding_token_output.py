# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-token (late-interaction / MaxSim) embedding output."""

import asyncio
from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import mlx.core as mx
import pytest
from fastapi.testclient import TestClient

from omlx.api.embedding_models import (
    MAX_TOKEN_IDS_PER_REQUEST,
    EmbeddingUsage,
    TokenEmbeddingData,
    TokenEmbeddingRequest,
    TokenEmbeddingResponse,
)
from omlx.engine.embedding import EmbeddingEngine
from omlx.models.embedding import MLXEmbeddingModel, TokenEmbeddingOutput
from omlx.server import ServerState, app


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


class _NanStubModel(_StubModel):
    """Like ``_StubModel`` but emits NaN wherever ``bad_token`` appears."""

    def __init__(self, bad_token: int, hidden: int = 1):
        super().__init__(hidden)
        self.bad_token = bad_token

    def __call__(self, **kwargs):
        output = super().__call__(**kwargs)
        bad = mx.expand_dims(
            mx.equal(kwargs["input_ids"], self.bad_token), -1
        )
        corrupt = mx.full(output.last_hidden_state.shape, float("nan"))
        return SimpleNamespace(
            last_hidden_state=mx.where(bad, corrupt, output.last_hidden_state)
        )


class _MaskAwareModel:
    """A forward that actually applies ``attention_mask``.

    Every position's value is its own id plus the mean of the real (unmasked)
    token ids in its row. If padding or the mask is mishandled the mean shifts
    and a padded batch disagrees with the single-sequence forward.
    """

    def __call__(self, input_ids, attention_mask=None):
        ids = input_ids.astype(mx.float32)
        mask = attention_mask.astype(mx.float32)
        denom = mx.maximum(mx.sum(mask, axis=-1, keepdims=True), 1.0)
        mean = mx.sum(ids * mask, axis=-1, keepdims=True) / denom
        return SimpleNamespace(last_hidden_state=mx.expand_dims(ids + mean, -1))


def _stub_backed_model(
    hidden: int = 1,
    vocab_size: int | None = None,
    max_position_embeddings: int | None = None,
    stub=None,
):
    model = MLXEmbeddingModel("stub-token-embedder")
    model._loaded = True
    model.model = stub if stub is not None else _StubModel(hidden)
    if vocab_size is not None or max_position_embeddings is not None:
        cfg = {}
        if vocab_size is not None:
            cfg["vocab_size"] = vocab_size
        if max_position_embeddings is not None:
            cfg["max_position_embeddings"] = max_position_embeddings
        model.model.config = SimpleNamespace(**cfg)
    return model


class TestTokenEmbeddingRequest:
    """The per-token request contract."""

    def test_accepts_explicit_ids(self):
        request = TokenEmbeddingRequest(model="m", input_ids=[[1, 2, 3]])
        assert request.input_ids == [[1, 2, 3]]

    def test_accepts_max_length(self):
        request = TokenEmbeddingRequest(
            model="m", input_ids=[[1, 2, 3]], max_length=128
        )
        assert request.max_length == 128

    def test_rejects_empty_batch(self):
        with pytest.raises(ValueError):
            TokenEmbeddingRequest(model="m", input_ids=[])

    def test_rejects_empty_sequence(self):
        with pytest.raises(ValueError):
            TokenEmbeddingRequest(model="m", input_ids=[[1], []])

    def test_rejects_negative_id(self):
        with pytest.raises(ValueError):
            TokenEmbeddingRequest(model="m", input_ids=[[-1]])

    def test_accepts_a_request_at_the_token_cap(self):
        request = TokenEmbeddingRequest(
            model="m", input_ids=[[1] * MAX_TOKEN_IDS_PER_REQUEST]
        )
        assert len(request.input_ids[0]) == MAX_TOKEN_IDS_PER_REQUEST

    def test_rejects_a_request_over_the_token_cap(self):
        with pytest.raises(ValueError, match="per-request limit"):
            TokenEmbeddingRequest(
                model="m", input_ids=[[1] * (MAX_TOKEN_IDS_PER_REQUEST + 1)]
            )

    def test_rejects_a_ragged_batch_over_the_token_cap(self):
        # The cap counts the whole request, not one sequence.
        per_sequence = MAX_TOKEN_IDS_PER_REQUEST // 2 + 1
        with pytest.raises(ValueError, match="per-request limit"):
            TokenEmbeddingRequest(
                model="m", input_ids=[[1] * per_sequence] * 2
            )


class TestTokenEmbeddingResponseContract:
    """The response reuses the pooled usage envelope and drops token_count."""

    def test_usage_is_the_shared_embedding_usage(self):
        response = TokenEmbeddingResponse(
            data=[TokenEmbeddingData(index=0, embedding=[[0.5, 1.5]])],
            model="m",
            usage=EmbeddingUsage(prompt_tokens=4, total_tokens=4),
        )
        payload = response.model_dump()
        assert payload["usage"] == {"prompt_tokens": 4, "total_tokens": 4}

    def test_data_item_has_no_token_count(self):
        item = TokenEmbeddingData(index=0, embedding=[[1.0], [2.0]])
        assert "token_count" not in item.model_dump()
        assert item.embedding == [[1.0], [2.0]]


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
        assert out.non_finite_items == []

    def test_pads_the_batch_then_slices_each_row(self):
        model = _stub_backed_model()
        out = model.embed_token_ids([[7, 8, 9], [3]])

        # One padded forward pass, both rows sliced back to their own length.
        assert model.model.batch_shapes == [(2, 3)]
        assert [len(row) for row in out.token_embeddings] == [3, 1]
        assert [vec[0] for vec in out.token_embeddings[0]] == [7.0, 8.0, 9.0]
        assert [vec[0] for vec in out.token_embeddings[1]] == [3.0]

    def test_mask_aware_padding_matches_single_sequence_forwards(self):
        # The stub uses attention_mask, so a padding leak changes the values.
        model = _stub_backed_model(stub=_MaskAwareModel())
        batched = model.embed_token_ids([[5, 6, 7], [8]])
        single0 = model.embed_token_ids([[5, 6, 7]]).token_embeddings[0]
        single1 = model.embed_token_ids([[8]]).token_embeddings[0]

        for got, want in zip(batched.token_embeddings[0], single0, strict=True):
            assert got == pytest.approx(want, abs=1e-6)
        for got, want in zip(batched.token_embeddings[1], single1, strict=True):
            assert got == pytest.approx(want, abs=1e-6)

    def test_values_are_the_raw_final_hidden_state(self):
        # No pooling and no L2 normalization: the caller owns both.
        model = _stub_backed_model(hidden=2)
        out = model.embed_token_ids([[1, 2]])
        assert out.token_embeddings == [[[1.0, 2.0], [2.0, 3.0]]]

    def test_splits_forwards_on_the_encoder_token_budget(self, monkeypatch):
        # 3 padded tokens * 3 sequences would be 9; a budget of 6 must split it.
        monkeypatch.setattr(
            "omlx.models.embedding.ENCODER_BATCH_TOKEN_BUDGET", 6
        )
        model = _stub_backed_model()
        out = model.embed_token_ids([[1, 2, 3], [4, 5], [6]])

        assert all(batch * width <= 6 for batch, width in model.model.batch_shapes)
        assert len(model.model.batch_shapes) > 1
        assert [vec[0] for row in out.token_embeddings for vec in row] == [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        ]
        assert out.total_tokens == 6

    def test_non_finite_rows_are_reported_without_vectors(self):
        model = _stub_backed_model(stub=_NanStubModel(bad_token=2))
        out = model.embed_token_ids([[1, 2, 3], [4, 5]])

        assert out.non_finite_items == [0]
        assert out.token_embeddings == []
        assert out.total_tokens == 5

    def test_nan_at_padding_positions_is_not_reported(self):
        # Token 0 is the pad id: a NaN there is not a returned token vector.
        model = _stub_backed_model(stub=_NanStubModel(bad_token=0))
        out = model.embed_token_ids([[1, 2, 3], [4]])

        assert out.non_finite_items == []
        assert [len(row) for row in out.token_embeddings] == [3, 1]

    def test_rejects_an_empty_sequence(self):
        model = _stub_backed_model()
        with pytest.raises(ValueError):
            model.embed_token_ids([[1, 2], []])

    def test_rejects_a_model_without_token_features(self):
        model = _stub_backed_model()
        model.model = lambda **kwargs: SimpleNamespace(text_embeds=mx.zeros((1, 4)))
        with pytest.raises(ValueError):
            model.embed_token_ids([[1, 2]])

    def test_rejects_token_id_past_vocab_size(self):
        model = _stub_backed_model(vocab_size=100)
        with pytest.raises(ValueError, match="exceeds model vocabulary size of 100"):
            model.embed_token_ids([[10, 100]])

    def test_rejects_token_id_at_2_to_the_31(self):
        model = _stub_backed_model()
        with pytest.raises(ValueError, match="exceeds 32-bit integer range"):
            model.embed_token_ids([[2**31]])

    def test_rejects_sequence_exceeding_max_position_embeddings(self):
        model = _stub_backed_model(max_position_embeddings=8192)
        with pytest.raises(ValueError, match="exceeds maximum context length of 8192"):
            model.embed_token_ids([[1] * 9000])

    def test_rejects_sequence_exceeding_explicit_max_length(self):
        model = _stub_backed_model(max_position_embeddings=8192)
        with pytest.raises(ValueError, match="exceeds maximum context length of 10"):
            model.embed_token_ids([[1] * 12], max_length=10)

    def test_explicit_max_length_cannot_exceed_the_model_bound(self):
        # A caller value above the position table must not lift the bound: an
        # out-of-range table row reads zeros instead of raising.
        model = _stub_backed_model(max_position_embeddings=512)
        with pytest.raises(ValueError, match="exceeds maximum context length of 512"):
            model.embed_token_ids([[1] * 600], max_length=100_000)


class _StubEngineModel:
    """Engine-level stub: only `embed_token_ids` is exercised."""

    def __init__(self, model):
        self._inner = model

    def embed_token_ids(self, ids_batch, max_length=None):
        return self._inner.embed_token_ids(ids_batch, max_length=max_length)


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

    def test_non_finite_items_are_mapped_to_request_indices(self):
        # The short (bad) sequence is grouped first; the engine must report the
        # caller's index, not the length-grouped position.
        engine = EmbeddingEngine("stub-token-embedder", batch_size=16)
        engine._model = _StubEngineModel(
            _stub_backed_model(stub=_NanStubModel(bad_token=20))
        )

        out = asyncio.run(engine.embed_token_ids([[10, 11, 12], [20]]))

        assert out.non_finite_items == [1]
        assert out.token_embeddings == []

    def test_empty_batch_is_empty(self):
        engine = EmbeddingEngine("stub-token-embedder")
        engine._model = _StubEngineModel(_stub_backed_model())
        out = asyncio.run(engine.embed_token_ids([]))
        assert out.token_embeddings == []
        assert out.total_tokens == 0
        assert out.non_finite_items == []

    def test_requires_a_started_engine(self):
        engine = EmbeddingEngine("stub-token-embedder")
        with pytest.raises(RuntimeError):
            asyncio.run(engine.embed_token_ids([[1]]))

    def test_reuses_fairness_gate_and_clears_cache_conditionally(self, monkeypatch):
        engine = EmbeddingEngine("stub-token-embedder", batch_size=2)
        engine._model = _StubEngineModel(_stub_backed_model())

        wait_turn_calls = []
        settle_calls = []
        clear_cache_calls = []

        orig_wait = engine._fairness.wait_turn
        orig_settle = engine._fairness.settle

        def mock_wait_turn():
            wait_turn_calls.append(True)
            return orig_wait()

        def mock_settle(duration, count, contended):
            settle_calls.append((count, contended))
            return orig_settle(duration, count, contended)

        monkeypatch.setattr(engine._fairness, "wait_turn", mock_wait_turn)
        monkeypatch.setattr(engine._fairness, "settle", mock_settle)
        monkeypatch.setattr(
            "omlx.engine.embedding.mx.clear_cache",
            lambda: clear_cache_calls.append(True),
        )

        # By default should_clear_cache is False, so clear_cache must NOT be called
        monkeypatch.setattr(engine._fairness, "should_clear_cache", lambda: False)
        asyncio.run(engine.embed_token_ids([[1], [2], [3]]))
        assert len(wait_turn_calls) == 2  # 3 items with batch_size=2 -> 2 chunks
        assert len(settle_calls) == 2
        assert len(clear_cache_calls) == 0

        # When should_clear_cache is True, clear_cache is called
        monkeypatch.setattr(engine._fairness, "should_clear_cache", lambda: True)
        asyncio.run(engine.embed_token_ids([[1], [2]]))
        assert len(clear_cache_calls) == 1

    def test_fairness_chunk_cap_limits_batch_size(self, monkeypatch):
        engine = EmbeddingEngine("stub-token-embedder", batch_size=16)
        engine._model = _StubEngineModel(_stub_backed_model())

        # chunk_cap limits batch to 1
        monkeypatch.setattr(engine._fairness, "chunk_cap", lambda: 1)
        batches_seen = []
        orig_embed = engine._model.embed_token_ids

        def mock_embed(batch, max_length=None):
            batches_seen.append(len(batch))
            return orig_embed(batch, max_length=max_length)

        monkeypatch.setattr(engine._model, "embed_token_ids", mock_embed)
        asyncio.run(engine.embed_token_ids([[1], [2], [3]]))
        assert batches_seen == [1, 1, 1]


class TestServerTokenEmbeddingsEndpoint:
    """`/v1/embeddings/tokens` model-bound and contract behaviour."""

    @staticmethod
    @asynccontextmanager
    async def _noop_acquire(engine):
        yield engine

    @classmethod
    @contextmanager
    def _client(cls, engine):
        state = ServerState()
        with (
            patch("omlx.server._JSON_KEEPALIVE_GRACE_S", 2),
            patch("omlx.server._server_state", state),
            patch("omlx.server.get_embedding_engine", AsyncMock(return_value=engine)),
            patch(
                "omlx.server.acquire_embedding_engine",
                lambda _model: cls._noop_acquire(engine),
            ),
            patch("omlx.server.get_max_context_window", return_value=None),
            patch("omlx.server.resolve_model_id", return_value="test-stub-embedder"),
            patch("omlx.server.get_server_metrics", return_value=MagicMock()),
        ):
            yield TestClient(app)

    @pytest.fixture
    def test_client(self):
        engine = EmbeddingEngine("test-stub-embedder")
        engine._model = _stub_backed_model(vocab_size=1000, max_position_embeddings=512)
        with self._client(engine) as client:
            yield client

    def test_rejects_token_id_past_vocab_size_with_400(self, test_client):
        res = test_client.post(
            "/v1/embeddings/tokens",
            json={"model": "test-stub-embedder", "input_ids": [[5, 1000]]},
        )
        assert res.status_code == 400
        assert "vocabulary size" in res.text

    def test_rejects_token_id_at_2_to_the_31_with_400(self, test_client):
        res = test_client.post(
            "/v1/embeddings/tokens",
            json={"model": "test-stub-embedder", "input_ids": [[2**31]]},
        )
        assert res.status_code == 400
        assert "32-bit integer" in res.text or "vocabulary size" in res.text

    def test_rejects_sequence_exceeding_max_position_embeddings_with_400(
        self, test_client
    ):
        res = test_client.post(
            "/v1/embeddings/tokens",
            json={"model": "test-stub-embedder", "input_ids": [[1] * 600]},
        )
        assert res.status_code == 400
        assert "context length" in res.text

    def test_explicit_max_length_cannot_exceed_the_model_bound(self, test_client):
        res = test_client.post(
            "/v1/embeddings/tokens",
            json={
                "model": "test-stub-embedder",
                "input_ids": [[1] * 600],
                "max_length": 100_000,
            },
        )
        assert res.status_code == 400
        assert "context length of 512" in res.text

    def test_rejects_a_request_over_the_token_cap_with_422(self, test_client):
        res = test_client.post(
            "/v1/embeddings/tokens",
            json={
                "model": "test-stub-embedder",
                "input_ids": [[1] * (MAX_TOKEN_IDS_PER_REQUEST + 1)],
            },
        )
        assert res.status_code == 422
        assert "per-request limit" in res.text

    def test_returns_200_with_shared_usage_and_no_token_count(self, test_client):
        res = test_client.post(
            "/v1/embeddings/tokens",
            json={"model": "test-stub-embedder", "input_ids": [[1, 2, 3]]},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["object"] == "list"
        assert body["model"] == "test-stub-embedder"
        assert body["usage"] == {"prompt_tokens": 3, "total_tokens": 3}
        assert len(body["data"]) == 1
        assert "token_count" not in body["data"][0]
        assert len(body["data"][0]["embedding"]) == 3

    def test_non_finite_output_is_a_500_listing_the_item(self):
        engine = EmbeddingEngine("test-stub-embedder")
        engine._model = _stub_backed_model(
            vocab_size=1000,
            max_position_embeddings=512,
            stub=_NanStubModel(bad_token=2),
        )
        with self._client(engine) as client:
            res = client.post(
                "/v1/embeddings/tokens",
                json={"model": "test-stub-embedder", "input_ids": [[1, 2, 3]]},
            )
        assert res.status_code == 500
        assert "non-finite" in res.text
        assert "item(s) [0]" in res.text
