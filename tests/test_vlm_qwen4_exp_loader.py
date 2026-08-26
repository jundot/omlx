# SPDX-License-Identifier: Apache-2.0
"""Tests for the Qwen4-Exp text-only mlx-vlm load path."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("mlx.core")

from omlx.engine import vlm as vlm_module
from omlx.engine.vlm import VLMBatchedEngine, _load_qwen4_exp_text_model
from omlx.exceptions import InvalidRequestError


class _FakeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 2
    eos_token_ids = None
    pad_token = None


class _FakeDetokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


class _FakeStoppingCriteria:
    def __init__(self, eos_token_ids, tokenizer):
        self.eos_token_ids = eos_token_ids
        self.tokenizer = tokenizer


def test_qwen4_exp_loader_never_constructs_auto_processor(monkeypatch, tmp_path):
    import mlx_vlm.tokenizer_utils as tokenizer_utils
    import mlx_vlm.utils as vlm_utils
    import transformers

    model = SimpleNamespace(config=SimpleNamespace(eos_token_id=[7]))
    tokenizer = _FakeTokenizer()
    load_model = MagicMock(return_value=model)
    load_processor = MagicMock(side_effect=AssertionError("must not be called"))

    monkeypatch.setattr(vlm_utils, "get_model_path", lambda model_name: tmp_path)
    monkeypatch.setattr(vlm_utils, "load_model", load_model)
    monkeypatch.setattr(vlm_utils, "load_processor", load_processor)
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    monkeypatch.setattr(
        tokenizer_utils,
        "load_tokenizer",
        lambda *args, **kwargs: _FakeDetokenizer,
    )
    monkeypatch.setattr(vlm_utils, "StoppingCriteria", _FakeStoppingCriteria)

    loaded_model, loaded_processor = _load_qwen4_exp_text_model("qwen4")

    assert loaded_model is model
    assert loaded_processor is tokenizer
    load_processor.assert_not_called()
    load_model.assert_called_once_with(
        tmp_path,
        lazy=False,
        strict=True,
        trust_remote_code=False,
    )
    assert tokenizer.pad_token == "<eos>"
    assert isinstance(tokenizer.detokenizer, _FakeDetokenizer)
    assert isinstance(tokenizer.stopping_criteria, _FakeStoppingCriteria)
    assert tokenizer.stopping_criteria.eos_token_ids == [7]


@pytest.mark.parametrize(
    ("images", "audio"),
    [([object()], None), ([], [("samples", 16000)])],
)
def test_qwen4_exp_text_runtime_rejects_media(images, audio):
    engine = VLMBatchedEngine("qwen4")
    engine._vlm_model = SimpleNamespace(
        config=SimpleNamespace(model_type=vlm_module.QWEN4_EXP_MODEL_TYPE)
    )

    with pytest.raises(InvalidRequestError, match="text-only"):
        engine._prepare_vision_inputs(
            [{"role": "user", "content": "hello"}],
            images=images,
            audio=audio,
        )
