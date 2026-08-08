# SPDX-License-Identifier: Apache-2.0
"""Opt-in parity test for the original Jina reranker v3 scoring path.

Run with a local official checkpoint:

    OMLX_JINA_V3_MODEL_PATH=/absolute/path/to/jina-reranker-v3-mlx \
        pytest tests/integration/test_jina_v3_real_model.py -m slow -q
"""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

import mlx.core as mx
import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        sys.platform != "darwin" or platform.machine() != "arm64",
        reason="Jina v3 MLX integration requires macOS on Apple Silicon.",
    ),
]

_ENV_VAR = "OMLX_JINA_V3_MODEL_PATH"
_QUERY = "What are the health benefits of green tea?"
_DOCUMENTS = [
    "Green tea contains catechin antioxidants that may reduce inflammation.",
    "Drinking green tea may improve alertness because it contains caffeine and L-theanine.",
    "Some studies associate green tea consumption with improved cardiovascular markers.",
    "Green tea may modestly increase energy expenditure and fat oxidation.",
    "Tea is prepared by steeping leaves in hot water.",
    "Black tea is oxidized more extensively than green tea.",
    "Coffee contains caffeine and antioxidants.",
    "Regular exercise improves cardiovascular health and mood.",
    "A balanced diet includes fruits, vegetables, protein, and whole grains.",
    "Green tea can taste bitter when brewed too hot.",
    "Catechins may help protect cells from oxidative stress.",
    "Evidence for weight-loss effects of green tea is mixed and generally modest.",
    "Basketball is played by two teams.",
    "Supply chain problems can raise coffee prices.",
    "Green tea is traditionally consumed in many Asian countries.",
]


def _model_path_from_environment() -> Path:
    configured = os.environ.get(_ENV_VAR)
    if not configured:
        pytest.skip(f"Set {_ENV_VAR} to run this Jina v3 real-model test.")

    model_path = Path(configured).expanduser()
    config_path = model_path / "config.json"
    if not config_path.is_file():
        pytest.skip(f"Jina v3 config.json not found at {config_path}")

    config = json.loads(config_path.read_text())
    assert "JinaForRanking" in (config.get("architectures") or [])
    assert (
        "layer_types" not in config
    ), f"{_ENV_VAR} must point to the original Jina v3 checkpoint."
    for required_file in ("rerank.py", "projector.safetensors"):
        if not (model_path / required_file).is_file():
            pytest.skip(f"Jina v3 reference file missing: {required_file}")
    return model_path


def _reference_scores(model_path: Path) -> list[float]:
    spec = importlib.util.spec_from_file_location(
        "_jina_v3_reference_rerank", model_path / "rerank.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = module.MLXReranker(
        str(model_path),
        projector_path=str(model_path / "projector.safetensors"),
    )
    results = reference.rerank(_QUERY, _DOCUMENTS)
    scores_by_index = {item["index"]: item["relevance_score"] for item in results}
    del reference
    gc.collect()
    mx.clear_cache()
    return [scores_by_index[index] for index in range(len(_DOCUMENTS))]


def test_jina_v3_matches_reference_scores_and_ranking():
    """v3 must retain its single-token prompt and independent scoring path."""
    model_path = _model_path_from_environment()
    expected_scores = _reference_scores(model_path)

    from omlx.models.reranker import MLXRerankerModel

    model = MLXRerankerModel(str(model_path))
    model.load()
    result = model.rerank(_QUERY, _DOCUMENTS, max_length=8192)

    assert model._is_jina_v35 is False
    assert result.scores == pytest.approx(expected_scores, abs=1e-6)
    expected_order = sorted(
        range(len(expected_scores)), key=expected_scores.__getitem__, reverse=True
    )
    assert result.indices == expected_order
    model.close()
