# SPDX-License-Identifier: Apache-2.0
"""Tests for multimodal (Qwen3-VL) reranker support."""

import json
from unittest.mock import MagicMock, patch

import pytest
from mlx_vlm.server.reranking import RerankItem

from omlx.exceptions import InvalidRequestError
from omlx.models.reranker import (
    MLXRerankerModel,
    RerankOutput,
    _coerce_item_to_text,
)

IMAGE_DATA_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
    "x8AAwMCAO+/p9sAAAAASUVORK5CYII="
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

    def test_str_becomes_text_item(self, tmp_path):
        model = MLXRerankerModel(str(tmp_path))
        assert model._build_vl_item("hello") == RerankItem(text="hello")

    def test_dict_text_only(self, tmp_path):
        model = MLXRerankerModel(str(tmp_path))
        assert model._build_vl_item({"text": "hi"}) == RerankItem(text="hi")

    def test_dict_image_loads_via_load_image(self, tmp_path):
        model = MLXRerankerModel(str(tmp_path))
        fake_img = object()
        with patch(
            "omlx.models.reranker.load_image", return_value=fake_img
        ) as mock_load:
            result = model._build_vl_item({"image": IMAGE_DATA_URI})
        mock_load.assert_called_once_with(IMAGE_DATA_URI, field="image")
        assert result == RerankItem(image=fake_img)

    def test_dict_text_and_image(self, tmp_path):
        model = MLXRerankerModel(str(tmp_path))
        fake_img = object()
        with patch(
            "omlx.models.reranker.load_image", return_value=fake_img
        ) as mock_load:
            result = model._build_vl_item({"text": "t", "image": IMAGE_DATA_URI})
        mock_load.assert_called_once_with(IMAGE_DATA_URI, field="image")
        assert result == RerankItem(text="t", image=fake_img)

    def test_dict_image_rejects_url(self, tmp_path):
        model = MLXRerankerModel(str(tmp_path))
        with pytest.raises(InvalidRequestError):
            model._build_vl_item({"image": "https://x/y.jpg"})

    def test_empty_dict_raises(self, tmp_path):
        model = MLXRerankerModel(str(tmp_path))
        with pytest.raises(ValueError, match="at least 'text' or 'image'"):
            model._build_vl_item({})


class TestVLRerankScoring:

    @pytest.mark.parametrize("trust_remote_code", [False, True])
    def test_loader_preserves_remote_code_setting(self, tmp_path, trust_remote_code):
        tmp_path = tmp_path / "qwen3-vl-reranker"
        tmp_path.mkdir()
        config = {
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "model_type": "qwen3_vl",
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        processor = MagicMock()
        with (
            patch(
                "omlx.models.reranker.load_reranker",
                return_value=(object(), processor),
            ) as load,
            patch("omlx.models.reranker.ensure_chat_template"),
        ):
            model = MLXRerankerModel(str(tmp_path), trust_remote_code=trust_remote_code)
            model.load()
        load.assert_called_once_with(tmp_path, trust_remote_code=trust_remote_code)
        assert model._vl_config == config

    def test_untrusted_custom_architecture_is_rejected_before_loading(self, tmp_path):
        tmp_path = tmp_path / "qwen3-vl-reranker"
        tmp_path.mkdir()
        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["Qwen3VLForConditionalGeneration"],
                    "model_type": "qwen3_vl",
                    "model_file": "custom.py",
                }
            )
        )
        with (
            patch("omlx.models.reranker.load_reranker") as load,
            pytest.raises(ValueError, match="Enable Trust Remote Code"),
        ):
            MLXRerankerModel(str(tmp_path)).load()
        load.assert_not_called()

    @staticmethod
    def _vl_model(tmp_path):
        model = MLXRerankerModel(str(tmp_path))
        model._is_vl_reranker = True
        model._loaded = True
        model.model = MagicMock()
        model.processor = MagicMock()
        model._vl_config = {"model_type": "qwen3_vl"}
        return model

    def test_rerank_vl_sorts_mlx_vlm_scores(self, tmp_path):
        model = self._vl_model(tmp_path)
        with patch(
            "omlx.models.reranker.score_documents",
            return_value=([0.2, 0.9, 0.5], 42),
        ) as score:
            output = model._rerank_vl(
                query="cat", documents=["doc a", "doc b", "doc c"]
            )

        assert isinstance(output, RerankOutput)
        assert output.scores == pytest.approx([0.2, 0.9, 0.5], rel=1e-5)
        assert output.indices == [1, 2, 0]
        assert output.total_tokens == 42
        args = score.call_args[0]
        assert args[:3] == (model.model, model.processor, model._vl_config)
        assert args[3] == RerankItem(text="cat")
        assert args[4] == [
            RerankItem(text="doc a"),
            RerankItem(text="doc b"),
            RerankItem(text="doc c"),
        ]

    def test_rerank_vl_with_image_documents(self, tmp_path):
        model = self._vl_model(tmp_path)
        fake_img = object()
        with (
            patch("omlx.models.reranker.load_image", return_value=fake_img),
            patch(
                "omlx.models.reranker.score_documents", return_value=([0.7, 0.3], 9)
            ) as score,
        ):
            output = model._rerank_vl(
                query={"text": "a dog"},
                documents=[{"text": "desc"}, {"image": IMAGE_DATA_URI}],
            )

        assert output.indices == [0, 1]
        assert score.call_args[0][4] == [
            RerankItem(text="desc"),
            RerankItem(image=fake_img),
        ]

    @pytest.mark.parametrize(
        ("instruction", "expected"),
        [
            (None, MLXRerankerModel._CAUSAL_LM_DEFAULT_INSTRUCTION),
            ("", MLXRerankerModel._CAUSAL_LM_DEFAULT_INSTRUCTION),
            ("Find diagrams", "Find diagrams"),
        ],
    )
    def test_rerank_forwards_instruction(self, tmp_path, instruction, expected):
        model = self._vl_model(tmp_path)
        with patch(
            "omlx.models.reranker.score_documents", return_value=([0.5], 3)
        ) as score:
            model.rerank("cat", ["doc"], instruction=instruction)

        assert score.call_args[0][5] == expected


class TestRerankDispatchCoerce:
    """Regression: text-only reranker paths still receive strings even when
    callers pass dict inputs (backwards compat for /v1/rerank dict docs)."""

    def test_causal_lm_path_receives_strings_from_dict_inputs(self, tmp_path):
        model = MLXRerankerModel(str(tmp_path))
        model._is_causal_lm = True
        model._loaded = True

        captured = {}

        def fake_causal_lm(query, docs, max_length, instruction=None):
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
