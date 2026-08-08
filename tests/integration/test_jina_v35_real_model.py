# SPDX-License-Identifier: Apache-2.0
"""Opt-in real-model parity test for Jina reranker v3.5 (dual matching +
block fusion), following up on jundot/omlx#2422 and PR #2449.

Compares oMLX's MLXRerankerModel against Jina's own reference rerank.py
(shipped inside the model repo) on the same query/documents, using the real
downloaded checkpoint. Never downloads anything itself.

Run explicitly after downloading jinaai/jina-reranker-v3.5-mlx:

    OMLX_JINA_V35_MODEL_PATH=/absolute/path/to/jina-reranker-v3.5-mlx \
        uv run pytest tests/integration/test_jina_v35_real_model.py -m slow -s -q

The ``slow`` marker excludes this test from default pytest runs and
repository CI. The environment variable prevents accidental use of an
arbitrary local model.

The reference implementation (rerank.py/modeling.py) is loaded dynamically
from inside the model directory itself, not vendored into oMLX - it ships
alongside the weights and isn't oMLX's code to own or maintain.

Two scenarios, not one, because block fusion is a weighted average over
whatever chunking happened - a fair comparison needs matching chunking on
both sides:

- single-chunk: generous max_length, everything fits in one chunk on both
  implementations. Isolates dual matching + the patched backbone + the
  projector from the fusion path (fusion is a no-op with one chunk).
- forced multi-chunk: a small, matching max_length on both sides forces each
  document into its own chunk on both implementations (empirically measured
  under the real tokenizer: 1 doc = 180 tokens, 2 docs = 205 - max_length=190
  fits exactly 1), genuinely exercising block fusion against real model
  weights on both sides.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        sys.platform != "darwin" or platform.machine() != "arm64",
        reason="Jina v3.5 MLX integration requires macOS on Apple Silicon.",
    ),
]

_ENV_VAR = "OMLX_JINA_V35_MODEL_PATH"

_QUERY = "What are the health benefits of green tea?"
_DOCUMENTS = [
    "Green tea contains antioxidants called catechins that may help reduce inflammation.",
    "Studies show that drinking green tea regularly can improve brain function.",
    "Basketball is one of the most popular sports in the United States.",
]

# Empirically measured under the real tokenizer: 1 doc = 180 tokens, 2 docs =
# 205. Fits exactly 1 doc per chunk for this document set, never 2.
_FORCED_CHUNK_MAX_LENGTH = 190


def _model_path_from_environment() -> Path:
    configured = os.environ.get(_ENV_VAR)
    if not configured:
        pytest.skip(f"Set {_ENV_VAR} to run this Jina v3.5 real-model test.")

    model_path = Path(configured).expanduser()
    config_path = model_path / "config.json"
    if not config_path.is_file():
        pytest.skip(f"Jina v3.5 config.json not found at {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = config.get("architectures") or []
    assert "JinaForRanking" in architectures, (
        f"{_ENV_VAR} must point to a JinaForRanking checkpoint, "
        f"not architectures={architectures!r}."
    )

    if (
        not (model_path / "rerank.py").is_file()
        or not (model_path / "modeling.py").is_file()
    ):
        pytest.skip(
            f"Reference rerank.py/modeling.py not found alongside {model_path} "
            "- needed to compare against oMLX's implementation."
        )
    return model_path


def _load_reference_reranker(model_path: Path, max_length: int):
    """Dynamically load Jina's own reference MLXReranker from inside the
    checkpoint directory. rerank.py does `import modeling as _modeling`, a
    bare (non-package) import, so the checkpoint directory must be on
    sys.path while it executes."""
    module_name = "_jina_v35_reference_rerank"
    spec = importlib.util.spec_from_file_location(
        module_name, str(model_path / "rerank.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(model_path))
    try:
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module.MLXReranker(str(model_path), max_length=max_length)
    finally:
        sys.path.remove(str(model_path))


def _assert_parity(omlx_scores: list[float], reference_results: list[dict]):
    """Align by original document index, then assert ranking and score
    parity between oMLX and the reference implementation."""
    reference_by_index = {r["index"]: r["relevance_score"] for r in reference_results}
    assert set(reference_by_index) == set(range(len(omlx_scores))), (
        "Reference did not return a score for every document: "
        f"got indices {sorted(reference_by_index)}, expected "
        f"{list(range(len(omlx_scores)))}."
    )

    omlx_ranking = sorted(
        range(len(omlx_scores)), key=lambda i: omlx_scores[i], reverse=True
    )
    reference_ranking = sorted(
        reference_by_index, key=lambda i: reference_by_index[i], reverse=True
    )
    assert (
        omlx_ranking == reference_ranking
    ), f"Ranking mismatch: oMLX={omlx_ranking}, reference={reference_ranking}"

    for idx in range(len(omlx_scores)):
        assert omlx_scores[idx] == pytest.approx(reference_by_index[idx], abs=1e-3), (
            f"Score mismatch for doc {idx}: oMLX={omlx_scores[idx]}, "
            f"reference={reference_by_index[idx]}"
        )


def test_jina_v35_matches_reference_single_chunk():
    """Baseline parity: everything fits in one chunk on both sides."""
    model_path = _model_path_from_environment()

    from omlx.models.reranker import MLXRerankerModel

    model = MLXRerankerModel(str(model_path))
    model.load()
    assert model._is_jina_v35 is True
    omlx_result = model.rerank(_QUERY, _DOCUMENTS, max_length=8192)

    reference = _load_reference_reranker(model_path, max_length=131072)
    reference_results = reference.rerank(_QUERY, _DOCUMENTS)

    _assert_parity(omlx_result.scores, reference_results)
    model.close()


def test_jina_v35_matches_reference_forced_multi_chunk():
    """Forced multi-chunk parity: each document in its own chunk on both
    sides, exercising the real block-fusion path against real weights."""
    model_path = _model_path_from_environment()

    from omlx.models.reranker import MLXRerankerModel

    model = MLXRerankerModel(str(model_path))
    model.load()
    assert model._is_jina_v35 is True
    omlx_result = model.rerank(_QUERY, _DOCUMENTS, max_length=_FORCED_CHUNK_MAX_LENGTH)

    reference = _load_reference_reranker(
        model_path, max_length=_FORCED_CHUNK_MAX_LENGTH
    )
    reference_results = reference.rerank(_QUERY, _DOCUMENTS)

    _assert_parity(omlx_result.scores, reference_results)
    model.close()
