# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx/api/rerank_models.py — the Pydantic schemas served at
/v1/rerank. Pins down Cohere/Jina compatibility: required fields,
multimodal query/document shapes, defaults, and the auto-generated
``id`` prefix that downstream clients filter on.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omlx.api.rerank_models import (
    RerankRequest,
    RerankResponse,
    RerankResult,
    RerankUsage,
)


class TestRerankRequest:
    def test_minimal_text_request(self):
        req = RerankRequest(
            model="qwen3-reranker",
            query="best wireless headphones",
            documents=["Sony WH-1000XM5", "Bose QC45"],
        )
        assert req.model == "qwen3-reranker"
        assert req.query == "best wireless headphones"
        assert req.documents == ["Sony WH-1000XM5", "Bose QC45"]

    def test_defaults(self):
        req = RerankRequest(model="m", query="q", documents=["d"])
        assert req.top_n is None
        assert req.return_documents is True  # Cohere-compat default
        assert req.max_chunks_per_doc is None

    def test_dict_query_for_multimodal(self):
        image = (
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
            "x8AAwMCAO+/p9sAAAAASUVORK5CYII="
        )
        req = RerankRequest(
            model="qwen3-vl-reranker",
            query={"text": "a red car", "image": image},
            documents=["doc1"],
        )
        assert req.query == {"text": "a red car", "image": image}

    def test_dict_documents_for_multimodal(self):
        req = RerankRequest(
            model="qwen3-vl-reranker",
            query="cars",
            documents=[
                {"text": "ferrari", "image": "data:image/png;base64,AAA"},
                {"text": "porsche"},
            ],
        )
        assert isinstance(req.documents[0], dict)
        assert req.documents[0]["image"].startswith("data:image/png")

    def test_top_n_accepts_int(self):
        req = RerankRequest(model="m", query="q", documents=["a", "b", "c"], top_n=2)
        assert req.top_n == 2

    def test_missing_model_rejected(self):
        with pytest.raises(ValidationError):
            RerankRequest(query="q", documents=["d"])  # type: ignore[call-arg]

    def test_missing_query_rejected(self):
        with pytest.raises(ValidationError):
            RerankRequest(model="m", documents=["d"])  # type: ignore[call-arg]

    def test_missing_documents_rejected(self):
        with pytest.raises(ValidationError):
            RerankRequest(model="m", query="q")  # type: ignore[call-arg]

    def test_return_documents_false_round_trips(self):
        req = RerankRequest(
            model="m", query="q", documents=["d"], return_documents=False
        )
        restored = RerankRequest.model_validate(req.model_dump())
        assert restored.return_documents is False


class TestRerankResult:
    def test_minimal_result_with_no_document(self):
        r = RerankResult(index=3, relevance_score=0.91)
        assert r.index == 3
        assert r.relevance_score == 0.91
        assert r.document is None  # return_documents=False path

    def test_result_with_text_document(self):
        r = RerankResult(
            index=0, relevance_score=0.5, document={"text": "Sony WH-1000XM5"}
        )
        assert r.document == {"text": "Sony WH-1000XM5"}

    def test_result_preserves_multimodal_document(self):
        r = RerankResult(
            index=1,
            relevance_score=0.3,
            document={"text": "ferrari", "image": "data:image/png;base64,AAA"},
        )
        assert "image" in r.document
        assert r.document["image"].startswith("data:image/png")

    def test_missing_index_rejected(self):
        with pytest.raises(ValidationError):
            RerankResult(relevance_score=0.5)  # type: ignore[call-arg]

    def test_missing_score_rejected(self):
        with pytest.raises(ValidationError):
            RerankResult(index=0)  # type: ignore[call-arg]


class TestRerankUsage:
    def test_required_field(self):
        u = RerankUsage(total_tokens=42)
        assert u.total_tokens == 42

    def test_missing_total_tokens_rejected(self):
        with pytest.raises(ValidationError):
            RerankUsage()  # type: ignore[call-arg]


class TestRerankResponse:
    def test_minimal_response(self):
        resp = RerankResponse(
            results=[RerankResult(index=0, relevance_score=0.9)],
            model="qwen3-reranker",
        )
        assert resp.model == "qwen3-reranker"
        assert len(resp.results) == 1
        assert resp.usage is None  # optional

    def test_auto_id_has_rerank_prefix(self):
        """Cohere clients filter telemetry on the ``rerank-`` prefix."""
        resp = RerankResponse(results=[], model="m")
        assert resp.id.startswith("rerank-")
        # 8 hex chars after the prefix
        assert len(resp.id) == len("rerank-") + 8

    def test_two_responses_get_distinct_ids(self):
        a = RerankResponse(results=[], model="m")
        b = RerankResponse(results=[], model="m")
        assert a.id != b.id

    def test_explicit_id_is_preserved(self):
        resp = RerankResponse(id="rerank-custom123", results=[], model="m")
        assert resp.id == "rerank-custom123"

    def test_usage_attached(self):
        resp = RerankResponse(
            results=[],
            model="m",
            usage=RerankUsage(total_tokens=128),
        )
        assert resp.usage is not None
        assert resp.usage.total_tokens == 128

    def test_missing_results_rejected(self):
        with pytest.raises(ValidationError):
            RerankResponse(model="m")  # type: ignore[call-arg]

    def test_missing_model_rejected(self):
        with pytest.raises(ValidationError):
            RerankResponse(results=[])  # type: ignore[call-arg]

    def test_round_trip_via_json(self):
        original = RerankResponse(
            results=[
                RerankResult(index=2, relevance_score=0.95, document={"text": "a"}),
                RerankResult(index=0, relevance_score=0.40, document={"text": "b"}),
            ],
            model="qwen3-reranker",
            usage=RerankUsage(total_tokens=64),
        )
        restored = RerankResponse.model_validate_json(original.model_dump_json())
        assert restored.model == original.model
        assert restored.id == original.id
        assert len(restored.results) == 2
        assert restored.results[0].relevance_score == 0.95
        assert restored.usage.total_tokens == 64
