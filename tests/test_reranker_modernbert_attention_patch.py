# SPDX-License-Identifier: Apache-2.0
"""Regression: a ModernBERT reranker must get the finite attention-mask patch.

The stock mlx-vlm mask uses -1e9, which overflows to -inf in fp16 and makes
fully padded (short) queries produce NaN. omlx already patches this for
embeddings (issue #3507); the reranker's sequence-classification branch must do
the same.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_vlm.models.modernbert.config import ModelConfig
from mlx_vlm.models.modernbert.modernbert import SequenceClassificationModel

import omlx.models.reranker as reranker_module
from omlx.models.reranker import MLXRerankerModel


def test_modernbert_reranker_applies_finite_attention_patch(tmp_path, monkeypatch):
    loaded = SimpleNamespace()
    patched: list = []
    monkeypatch.setattr(
        reranker_module, "patch_modernbert_attention", lambda m: patched.append(m)
    )
    monkeypatch.setattr(reranker_module, "load_encoder_model", lambda *a, **k: loaded)
    monkeypatch.setattr(reranker_module, "load_processor", lambda *a, **k: object())

    model_dir = tmp_path / "modernbert-reranker"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["ModernBertForSequenceClassification"],
                "model_type": "modernbert",
            }
        )
    )

    MLXRerankerModel(str(model_dir)).load()

    assert patched == [loaded], "the reranker load path must patch ModernBERT attention"


@pytest.mark.parametrize("trust_remote_code", [False, True])
@pytest.mark.parametrize("num_labels", [1, 2])
@pytest.mark.parametrize("compiled", [False, True])
def test_modernbert_checkpoint_preserves_labels_activation_and_trust(
    tmp_path, trust_remote_code, num_labels, compiled
):
    config = {
        "architectures": ["ModernBertForSequenceClassification"],
        "model_type": "modernbert",
        "vocab_size": 32,
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "id2label": {str(i): f"LABEL_{i}" for i in range(num_labels)},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    reference = SequenceClassificationModel(
        ModelConfig.from_dict({**config, "num_labels": num_labels})
    )
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(tree_flatten(reference.parameters()))
    )
    processor = object()
    model = MLXRerankerModel(str(tmp_path), trust_remote_code=trust_remote_code)
    with patch.object(
        reranker_module, "load_processor", return_value=processor
    ) as load:
        model.load()
    load.assert_called_once_with(
        tmp_path, add_detokenizer=False, trust_remote_code=trust_remote_code
    )
    assert model.num_labels == num_labels
    assert model._is_compiled
    model._is_compiled = compiled
    ids = mx.array([[1, 2, 3], [4, 5, 0]])
    mask = mx.array([[1, 1, 1], [1, 1, 0]])
    logits = reference(ids, attention_mask=mask).logits
    expected = mx.sigmoid(logits) if num_labels == 1 else mx.softmax(logits, axis=-1)
    np.testing.assert_allclose(
        np.array(model._seq_classification_logits(ids, mask)),
        np.array(expected),
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("flags", "id2label", "is_regression"),
    [
        ({"is_regression": True}, {"0": "LABEL_0", "1": "LABEL_1"}, True),
        (
            {"pipeline_config": {"binary_sigmoid": True}},
            {"0": "LABEL_0", "1": "LABEL_1"},
            False,
        ),
        ({}, None, False),
    ],
)
@pytest.mark.parametrize("compiled", [False, True])
def test_modernbert_single_output_flags_and_regression_scores(
    tmp_path, flags, id2label, is_regression, compiled
):
    config = {
        "architectures": ["ModernBertForSequenceClassification"],
        "model_type": "modernbert",
        "vocab_size": 32,
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "id2label": id2label,
        **flags,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    reference = SequenceClassificationModel(
        ModelConfig.from_dict({**config, "num_labels": 1})
    )
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(tree_flatten(reference.parameters()))
    )
    with patch.object(reranker_module, "load_processor", return_value=object()):
        model = MLXRerankerModel(str(tmp_path))
        model.load()

    assert model.num_labels == 1
    assert model._is_regression is is_regression
    model._is_compiled = compiled
    ids = mx.array([[1, 2, 3]])
    mask = mx.ones_like(ids)
    logits = reference(ids, attention_mask=mask).logits
    expected = logits if is_regression else mx.sigmoid(logits)
    np.testing.assert_allclose(
        np.array(model._seq_classification_logits(ids, mask)),
        np.array(expected),
        atol=1e-6,
    )


def test_modernbert_custom_model_code_is_rejected_before_loading(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["ModernBertForSequenceClassification"],
                "model_type": "modernbert",
                "model_file": "custom.py",
            }
        )
    )
    with (
        patch.object(reranker_module, "load_encoder_model") as load_model,
        patch.object(reranker_module, "load_processor") as load_processor,
        pytest.raises(ValueError, match="Enable Trust Remote Code"),
    ):
        MLXRerankerModel(str(tmp_path)).load()
    load_model.assert_not_called()
    load_processor.assert_not_called()
