# SPDX-License-Identifier: Apache-2.0
"""Tests for the Cohere2 MoE mlx-vlm text-only load path."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")

from omlx.engine import vlm as vlm_module
from omlx.engine.vlm import (
    VLMBatchedEngine,
    _load_cohere2_moe_text_model,
)
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


def test_cohere2_moe_loader_uses_upstream_processor(monkeypatch, tmp_path):
    import mlx_vlm.utils as vlm_utils

    model = SimpleNamespace(config=SimpleNamespace(eos_token_id=[2]))
    processor = object()

    monkeypatch.setattr(vlm_utils, "get_model_path", lambda model_name: tmp_path)
    monkeypatch.setattr(vlm_utils, "load_model", lambda *a, **k: model)
    monkeypatch.setattr(vlm_utils, "load_processor", lambda *a, **k: processor)

    loaded_model, loaded_processor = _load_cohere2_moe_text_model("cohere")

    assert loaded_model is model
    assert loaded_processor is processor


def test_cohere2_moe_loader_falls_back_to_tokenizer(monkeypatch, tmp_path):
    import mlx_vlm.tokenizer_utils as tokenizer_utils
    import mlx_vlm.utils as vlm_utils
    import transformers

    model = SimpleNamespace(config=SimpleNamespace(eos_token_id=[7]))
    tokenizer = _FakeTokenizer()

    monkeypatch.setattr(vlm_utils, "get_model_path", lambda model_name: tmp_path)
    monkeypatch.setattr(vlm_utils, "load_model", lambda *a, **k: model)

    def fail_processor(*args, **kwargs):
        raise ValueError("no processor")

    monkeypatch.setattr(vlm_utils, "load_processor", fail_processor)
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *a, **k: tokenizer,
    )
    monkeypatch.setattr(
        tokenizer_utils,
        "load_tokenizer",
        lambda *a, **k: _FakeDetokenizer,
    )
    monkeypatch.setattr(vlm_utils, "StoppingCriteria", _FakeStoppingCriteria)

    loaded_model, loaded_processor = _load_cohere2_moe_text_model("cohere")

    assert loaded_model is model
    assert loaded_processor is tokenizer
    assert tokenizer.pad_token == "<eos>"
    assert isinstance(tokenizer.detokenizer, _FakeDetokenizer)
    assert isinstance(tokenizer.stopping_criteria, _FakeStoppingCriteria)
    assert tokenizer.stopping_criteria.eos_token_ids == [7]


def test_cohere2_moe_rejects_image_input():
    engine = VLMBatchedEngine("cohere")
    engine._vlm_model = SimpleNamespace(
        config=SimpleNamespace(model_type=vlm_module.COHERE2_MOE_MODEL_TYPE)
    )

    with pytest.raises(InvalidRequestError, match="text-only"):
        engine._prepare_vision_inputs(
            [{"role": "user", "content": "describe"}],
            images=[object()],
        )


def test_cohere2_moe_rejects_audio_input():
    engine = VLMBatchedEngine("cohere")
    engine._vlm_model = SimpleNamespace(
        config=SimpleNamespace(model_type=vlm_module.COHERE2_MOE_MODEL_TYPE)
    )

    with pytest.raises(InvalidRequestError, match="text-only"):
        engine._prepare_vision_inputs(
            [{"role": "user", "content": "transcribe"}],
            images=[],
            audio=[("samples", 16000)],
        )
