# SPDX-License-Identifier: Apache-2.0
"""Strict-loading regressions for legacy encoder quantization metadata."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_vlm.models.modernbert.config import ModelConfig
from mlx_vlm.models.modernbert.modernbert import Model, SequenceClassificationModel

from omlx.models.embedding import MLXEmbeddingModel
from omlx.models.reranker import MLXRerankerModel
from omlx.utils.model_loading import normalize_encoder_quantization_config


@pytest.mark.parametrize(
    ("method", "group_size", "bits", "mode"),
    [
        ("compressed-tensors", 32, 4, "affine"),
        ("mxfp4", 32, 4, "mxfp4"),
        ("nvfp4", 16, 4, "nvfp4"),
        ("mxfp8", 32, 8, "mxfp8"),
    ],
)
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("reranker", [False, True])
def test_legacy_quantization_metadata_loads_packed_checkpoint(
    tmp_path, method, group_size, bits, mode, nested, reranker
):
    config = {
        "architectures": [
            "ModernBertForSequenceClassification" if reranker else "ModernBertModel"
        ],
        "model_type": "modernbert",
        "vocab_size": 64,
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "intermediate_size": 128,
        "num_attention_heads": 2,
    }
    model_class = SequenceClassificationModel if reranker else Model
    reference = model_class(ModelConfig.from_dict(config))
    nn.quantize(reference, group_size=group_size, bits=bits, mode=mode)
    reference.eval()
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(tree_flatten(reference.parameters()))
    )
    metadata = {"quantization_config": {"quant_method": method}}
    config.update({"text_config": metadata} if nested else metadata)
    (tmp_path / "config.json").write_text(json.dumps(config))

    wrapper = (
        MLXRerankerModel(str(tmp_path))
        if reranker
        else MLXEmbeddingModel(str(tmp_path))
    )
    module = "reranker" if reranker else "embedding"
    with (
        patch(
            f"omlx.models.{module}.load_processor",
            return_value=SimpleNamespace(tokenizer=object()),
        ),
        patch.object(wrapper, "_try_compile", return_value=False),
    ):
        wrapper.load()

    ids = mx.array([[1, 2, 3]])
    mask = mx.ones_like(ids)
    expected = reference(ids, attention_mask=mask)
    actual = wrapper.model(ids, attention_mask=mask)
    field = "logits" if reranker else "text_embeds"
    np.testing.assert_allclose(
        np.array(getattr(actual, field)),
        np.array(getattr(expected, field)),
        atol=1e-5,
    )


@pytest.mark.parametrize("quantization", [None, {"group_size": 64, "bits": 8}])
def test_existing_quantization_entry_is_authoritative(quantization):
    config = {
        "quantization": quantization,
        "quantization_config": {"quant_method": "mxfp4"},
    }

    normalize_encoder_quantization_config(config)

    assert config["quantization"] is quantization
    assert config["quantization_config"] == {"quant_method": "mxfp4"}


def test_top_level_metadata_takes_precedence_over_text_config():
    config = {
        "quantization_config": {"quant_method": "nvfp4"},
        "text_config": {"quantization_config": {"quant_method": "mxfp4"}},
    }

    normalize_encoder_quantization_config(config)

    assert config["quantization"] == {"group_size": 16, "bits": 4, "mode": "nvfp4"}
