# SPDX-License-Identifier: Apache-2.0
"""Regression: a ModernBERT reranker must get the finite attention-mask patch.

The stock mlx-embeddings mask uses -1e9, which overflows to -inf in fp16 and
makes fully padded (short) queries produce NaN. omlx already patches this for
embeddings (issue #3507); the reranker's mlx-embeddings branch must do the same.
"""

import json
from types import SimpleNamespace

import omlx.models.reranker as reranker_module
from omlx.models.reranker import MLXRerankerModel


def test_modernbert_reranker_applies_finite_attention_patch(tmp_path, monkeypatch):
    loaded = SimpleNamespace(
        model=SimpleNamespace(),
        config=SimpleNamespace(num_labels=1),
    )
    patched: list = []
    monkeypatch.setattr(
        reranker_module, "patch_modernbert_attention", lambda m: patched.append(m)
    )
    monkeypatch.setattr("mlx_embeddings.load", lambda *a, **k: (loaded, object()))

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
