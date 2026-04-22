# SPDX-License-Identifier: Apache-2.0
"""Tests for multimodal (Qwen3-VL) reranker support."""

import json
from unittest.mock import MagicMock, patch

import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from omlx.models.reranker import (
    MLXRerankerModel,
    RerankOutput,
    _coerce_item_to_text,
)


class TestCoerceItemToText:
    def test_str_passthrough(self):
        assert _coerce_item_to_text("hello") == "hello"

    def test_dict_text_extract(self):
        assert _coerce_item_to_text({"text": "hi"}) == "hi"

    def test_dict_image_only_returns_empty(self):
        # Text-only paths should not crash when only image is present; they
        # just get an empty string.
        assert _coerce_item_to_text({"image": "https://x/y.jpg"}) == ""

    def test_dict_text_and_image_takes_text(self):
        assert _coerce_item_to_text({"text": "t", "image": "i"}) == "t"

    def test_non_str_non_dict_stringifies(self):
        assert _coerce_item_to_text(42) == "42"


class TestVLRerankerValidation:
    def _make_model_dir(self, tmp_path, name):
        d = tmp_path / name
        d.mkdir()
        config = {
            "model_type": "qwen3_vl",
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "vision_config": {"hidden_size": 1024},
        }
        (d / "config.json").write_text(json.dumps(config))
        return d

    def test_validate_accepts_vl_reranker_with_dir_hint(self, tmp_path):
        d = self._make_model_dir(tmp_path, "Qwen3-VL-Reranker-2B")
        model = MLXRerankerModel(str(d))
        model._validate_architecture()

    def test_validate_rejects_vl_without_dir_hint(self, tmp_path):
        d = self._make_model_dir(tmp_path, "Qwen3-VL-2B")
        model = MLXRerankerModel(str(d))
        with pytest.raises(ValueError, match="does not contain"):
            model._validate_architecture()


class TestVLItemBuilder:
    def test_str_becomes_text_dict(self, tmp_path):
        model = MLXRerankerModel(str(tmp_path))
        assert model._build_vl_item("hello") == {"text": "hello"}

    def test_dict_text_only(self, tmp_path):
        model = MLXRerankerModel(str(tmp_path))
        assert model._build_vl_item({"text": "hi"}) == {"text": "hi"}

    def test_dict_image_loads_via_load_image(self, tmp_path):
        model = MLXRerankerModel(str(tmp_path))
        fake_img = object()
        with patch(
            "omlx.models.reranker.load_image", return_value=fake_img
        ) as mock_load:
            result = model._build_vl_item({"image": "https://x/y.jpg"})
        mock_load.assert_called_once_with("https://x/y.jpg")
        assert result == {"image": fake_img}

    def test_dict_text_and_image(self, tmp_path):
        model = MLXRerankerModel(str(tmp_path))
        fake_img = object()
        with patch(
            "omlx.models.reranker.load_image", return_value=fake_img
        ):
            result = model._build_vl_item({"text": "t", "image": "i"})
        assert result == {"text": "t", "image": fake_img}

    def test_empty_dict_raises(self, tmp_path):
        model = MLXRerankerModel(str(tmp_path))
        with pytest.raises(ValueError, match="at least 'text' or 'image'"):
            model._build_vl_item({})


class TestVLRerankScoring:
    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_rerank_vl_wraps_process_output(self, tmp_path):
        """_rerank_vl sorts model.process() scores into RerankOutput."""
        model = MLXRerankerModel(str(tmp_path))
        model._is_vl_reranker = True
        model._loaded = True

        # Mock mlx-embeddings model: process() returns mx.array([0.2, 0.9, 0.5])
        mock_model = MagicMock()
        mock_model.process.return_value = mx.array([0.2, 0.9, 0.5])
        model.model = mock_model
        model.processor = MagicMock()

        output = model._rerank_vl(
            query="cat",
            documents=["doc a", "doc b", "doc c"],
            max_length=8192,
        )

        assert isinstance(output, RerankOutput)
        assert output.scores == pytest.approx([0.2, 0.9, 0.5], rel=1e-5)
        assert output.indices == [1, 2, 0]  # sorted descending
        assert output.total_tokens == 0

        # process() called with the expected input dict shape
        call_args = mock_model.process.call_args
        inputs = call_args[0][0]
        assert "instruction" in inputs
        assert inputs["query"] == {"text": "cat"}
        assert inputs["documents"] == [
            {"text": "doc a"},
            {"text": "doc b"},
            {"text": "doc c"},
        ]
        assert call_args[1]["processor"] is model.processor

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_rerank_vl_with_image_documents(self, tmp_path):
        """_rerank_vl threads image dicts through _build_vl_item."""
        model = MLXRerankerModel(str(tmp_path))
        model._is_vl_reranker = True
        model._loaded = True

        mock_model = MagicMock()
        mock_model.process.return_value = mx.array([0.7, 0.3])
        model.model = mock_model
        model.processor = MagicMock()

        fake_img = object()
        with patch(
            "omlx.models.reranker.load_image", return_value=fake_img
        ):
            output = model._rerank_vl(
                query={"text": "a dog"},
                documents=[
                    {"text": "desc"},
                    {"image": "https://x/dog.jpg"},
                ],
                max_length=8192,
            )

        assert output.indices == [0, 1]
        inputs = mock_model.process.call_args[0][0]
        assert inputs["documents"][0] == {"text": "desc"}
        assert inputs["documents"][1] == {"image": fake_img}


class TestRerankDispatchCoerce:
    """Regression: text-only reranker paths still receive strings even when
    callers pass dict inputs (backwards compat for /v1/rerank dict docs)."""

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_causal_lm_path_receives_strings_from_dict_inputs(self, tmp_path):
        model = MLXRerankerModel(str(tmp_path))
        model._is_causal_lm = True
        model._loaded = True

        captured = {}

        def fake_causal_lm(query, docs, max_length):
            captured["query"] = query
            captured["docs"] = docs
            return RerankOutput(scores=[0.5, 0.5], indices=[0, 1], total_tokens=0)

        model._rerank_causal_lm = fake_causal_lm

        model.rerank(
            query={"text": "q", "image": "ignored"},
            documents=[{"text": "a"}, "b"],
        )

        assert captured["query"] == "q"
        assert captured["docs"] == ["a", "b"]
