# SPDX-License-Identifier: Apache-2.0
"""
d1 decision models on the LFM2-VL backbone.

Covers the ``lfm2_vl`` + ``D1Model`` detection (decision_kind / detect_model_type),
the wrapper-class selection in the decision engine, and the D1Lfm2Model wrapper
driving mlx-vlm's native ``models.d1`` readout.
"""

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from omlx.engine.decision import _MODEL_CLASSES, DecisionEngine  # noqa: E402
from omlx.model_discovery import decision_kind, detect_model_type  # noqa: E402
from omlx.models.d1 import D1Model  # noqa: E402
from omlx.models.d1_lfm2 import D1Lfm2Model  # noqa: E402
from omlx.models.decision import (  # noqa: E402
    DecisionContextLengthError,
    DecisionRequestError,
)


def _lfm2_d1_config() -> dict:
    return {
        "architectures": ["D1Model"],
        "auto_map": {"AutoModel": "modeling_d1.D1Model"},
        "model_type": "lfm2_vl",
        "text_config": {"max_position_embeddings": 32768},
    }


def _write_model_dir(tmp_path, config: dict, name: str = "checkpoint"):
    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(config))
    return model_dir


# --------------------------------------------------------------------------
# decision_kind / detect_model_type


def test_decision_kind_lfm2_d1_via_architecture(tmp_path):
    path = _write_model_dir(tmp_path, _lfm2_d1_config())
    assert decision_kind(path) == "d1_lfm2"


def test_decision_kind_lfm2_d1_via_auto_map(tmp_path):
    config = _lfm2_d1_config()
    config["architectures"] = ["Lfm2VlForConditionalGeneration"]
    path = _write_model_dir(tmp_path, config)
    assert decision_kind(path) == "d1_lfm2"


def test_decision_kind_lfm3_d1_unchanged(tmp_path):
    for model_type in ("lfm3_vl", "lfm3_omni"):
        config = _lfm2_d1_config()
        config["model_type"] = model_type
        path = _write_model_dir(tmp_path, config, name=model_type)
        assert decision_kind(path) == "d1"


def test_decision_kind_plain_lfm2_vl_is_not_a_decision(tmp_path):
    config = _lfm2_d1_config()
    config["architectures"] = ["Lfm2VlForConditionalGeneration"]
    config["auto_map"] = {}
    path = _write_model_dir(tmp_path, config)
    assert decision_kind(path) is None


def test_detect_model_type_lfm2_d1_is_decision(tmp_path):
    path = _write_model_dir(tmp_path, _lfm2_d1_config())
    assert detect_model_type(path) == "decision"


# --------------------------------------------------------------------------
# wrapper-class selection


def test_decision_classes_route_each_kind():
    assert _MODEL_CLASSES["d1"] is D1Model
    assert _MODEL_CLASSES["d1_lfm2"] is D1Lfm2Model


def test_engine_start_selects_the_lfm2_wrapper(tmp_path, monkeypatch):
    path = _write_model_dir(tmp_path, _lfm2_d1_config())
    monkeypatch.setattr(D1Lfm2Model, "load", lambda self: None)
    engine = DecisionEngine(str(path))
    asyncio.run(engine.start())
    assert engine.kind == "d1_lfm2"
    assert isinstance(engine._model, D1Lfm2Model)


# --------------------------------------------------------------------------
# D1Lfm2Model wrapper


class _WordTokenizer:
    """One token per distinct string, so every option code is a single token."""

    bos_token = "<s>"

    def __init__(self):
        self._ids: dict[str, int] = {}

    def encode(self, text, add_special_tokens=False):
        if text not in self._ids:
            self._ids[text] = 5 + len(self._ids)
        return [self._ids[text]]


@pytest.fixture(scope="module")
def tiny_lfm2():
    from mlx_vlm.models.d1 import Model as MlxD1Model
    from mlx_vlm.models.d1.config import ModelConfig, TextConfig, VisionConfig

    config = ModelConfig(
        projector_hidden_size=32,
        downsample_factor=2,
        text_config=TextConfig(
            model_type="lfm2",
            hidden_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=64,
            block_dim=32,
            conv_dim=32,
            vocab_size=512,
            layer_types=["conv", "full_attention", "conv", "full_attention"],
            max_position_embeddings=4096,
        ),
        vision_config=VisionConfig(
            model_type="lfm2_vl",
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=32,
            patch_size=16,
            num_patches=4,
        ),
    )
    mx.random.seed(0)
    model = MlxD1Model(config)
    mx.eval(model.parameters())
    model.eval()
    wrapper = D1Lfm2Model("unused")
    wrapper.model = model
    wrapper.tokenizer = _WordTokenizer()
    wrapper.processor = SimpleNamespace(tokenizer=wrapper.tokenizer)
    wrapper.limit = 4096
    return wrapper


def _request(**questions):
    return {
        "state": "def add(a, b):\n    return a + b",
        "questions": questions,
    }


def _drain(steps):
    counts = []
    try:
        while True:
            counts.append(next(steps))
    except StopIteration as stop:
        return counts, stop.value


def test_wrapper_reads_every_question_type(tiny_lfm2):
    plan = tiny_lfm2.encode(
        _request(
            noul={"type": "noul", "instructions": "Is the code correct?"},
            pick={
                "type": "choice",
                "instructions": "Which?",
                "criteria": {"a": "first", "b": "second", "c": "third"},
            },
            rate={
                "type": "score",
                "instructions": "Rate it",
                "criteria": ["bad", "ok", "good"],
            },
        )
    )
    counts, result = _drain(tiny_lfm2.run(plan, lambda: 8))
    assert sum(counts) == result["input_tokens"]

    answers = result["answers"]
    assert set(answers) == {"noul", "pick", "rate"}
    assert answers["noul"]["type"] == "noul"
    assert 0.0 <= answers["noul"]["noul"] <= 1.0
    assert answers["pick"]["type"] == "choice"
    assert answers["pick"]["choice"] in {"a", "b", "c"}
    assert set(answers["pick"]["probabilities"]) == {"a", "b", "c"}
    # The readout softmaxes the option scores; round4 leaves ~1e-4 of slack.
    assert abs(sum(answers["pick"]["probabilities"].values()) - 1.0) < 5e-4
    assert answers["rate"]["type"] == "score"
    assert set(answers["rate"]["legend"]) == {"0", "1", "2"}
    assert 0.0 <= answers["rate"]["score"] <= 2.0


def test_wrapper_answers_come_from_the_native_readout(tiny_lfm2):
    from mlx_vlm.models.d1.prompt import as_question, readout

    request = _request(pick={"type": "choice", "instructions": "Which?",
                             "criteria": {"a": "first", "b": "second"}})
    plan = tiny_lfm2.encode(request)
    logprobs = tiny_lfm2.model.answer_logprobs(
        plan.trunk, plan.branches, plan.vision, tiny_lfm2.limit
    )
    expected = readout(
        tiny_lfm2.tokenizer, as_question(request["questions"]["pick"]), logprobs[0]
    )
    _, result = _drain(tiny_lfm2.run(plan, lambda: 8))
    got = result["answers"]["pick"]["probabilities"]
    assert got == {name: round(p, 4) for name, p in zip(("a", "b"), expected)}


def test_wrapper_rejects_videos(tiny_lfm2):
    request = _request(q={"type": "noul", "instructions": "?"})
    request["videos"] = ["data:video/mp4;base64,AA=="]
    with pytest.raises(DecisionRequestError, match="videos"):
        tiny_lfm2.encode(request)


def test_wrapper_reports_malformed_questions(tiny_lfm2):
    request = _request(q={"type": "choice", "instructions": "?", "criteria": {"a": "A"}})
    with pytest.raises(DecisionRequestError, match="at least two"):
        tiny_lfm2.encode(request)


def test_wrapper_enforces_the_context_limit(tiny_lfm2, monkeypatch):
    request = _request(
        q={"type": "noul", "instructions": "?"},
        r={"type": "noul", "instructions": "?"},
    )
    monkeypatch.setattr(tiny_lfm2, "limit", 1)
    with pytest.raises(DecisionContextLengthError, match="the limit is 1"):
        tiny_lfm2.encode(request)


def test_wrapper_has_vision_and_plan_shape(tiny_lfm2):
    assert tiny_lfm2.has_vision is True
    plan = tiny_lfm2.encode(_request(q={"type": "noul", "instructions": "?"}))
    assert plan.questions.keys() == {"q"}
    assert isinstance(plan.trunk, list) and plan.vision == {}
    # One question is a single pass: mlx-vlm folds the suffix into the trunk.
    assert plan.branches == []
    hidden = np.array(tiny_lfm2.model.language_model.model.embed_tokens(mx.array([plan.trunk])))
    assert hidden.shape[0] == 1


# --------------------------------------------------------------------------
# Calibration probe (needs the real checkpoint, hence slow)


_DIFF = """diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1,3 +1,4 @@
 def read(p):
-    return open(p).read()
+    f = open(p)
+    return f.read()
"""


def _d1_candidates():
    """Candidate d1 directories on this machine, in probe order."""
    import os
    from pathlib import Path

    candidates = []
    if os.environ.get("OMLX_D1_MODEL_DIR"):
        candidates.append(Path(os.environ["OMLX_D1_MODEL_DIR"]))
    models = Path.home() / ".omlx" / "models"
    if models.is_dir():
        candidates.extend(sorted(p for p in models.iterdir() if p.is_dir()))
    return [p for p in candidates if (p / "config.json").is_file()]


def _real_d1_dir():
    """A real d1 checkpoint on disk of the family this probe needs, if any."""
    import os
    from pathlib import Path

    for path in _d1_candidates():
        if decision_kind(path) == "d1_lfm2":
            return path
    return None


@pytest.mark.slow
def test_calibration_probe_handle_leak_is_safe():
    """A recorded probe, not a property: needs the real checkpoint.

    The diff drops the file object without closing it. Two independent runs of
    the real checkpoint call it `safe` (0.719/0.281 and 0.742/0.258 against
    `bug`), while a strict reading of the same diff calls the missing close a
    bug. That is what makes it a useful calibration point: a change that moves
    the verdict here moves it on a borderline case, not on an obvious one. It
    is not an assertion about the model, so a checkpoint that answers `bug` is
    a finding, not a test failure.

    Needs the LFM2-VL family specifically: d1 also ships on an LFM3 backbone,
    which ``decision_kind()`` reports as ``d1`` and this probe declines, so on
    such an install the probe always skips. Point ``OMLX_D1_MODEL_DIR`` at an
    lfm2_vl d1 checkpoint to actually run it.
    """
    path = _real_d1_dir()
    if path is None:
        other = sorted({k for k in map(decision_kind, _d1_candidates()) if k})
        pytest.skip(
            "needs a d1_lfm2 (LFM2-VL) d1 checkpoint; set OMLX_D1_MODEL_DIR to one "
            + (f"(found other families on disk: {other})" if other
               else "(no d1 checkpoint on disk)")
        )

    engine = DecisionEngine(str(path))
    asyncio.run(engine.start())
    try:
        plan = asyncio.run(engine.encode({
            "state": _DIFF,
            "questions": {"verdict": {
                "type": "choice",
                "instructions": "does this patch introduce a bug",
                "criteria": {"safe": "no bug", "bug": "introduces a bug"},
            }},
        }))
        payload = asyncio.run(engine.systemone(plan))
    finally:
        asyncio.run(engine.stop())

    assert payload["input_tokens"] > 0
    verdict = payload["answers"]["verdict"]
    assert verdict["type"] == "choice"
    assert set(verdict["probabilities"]) == {"safe", "bug"}
    assert verdict["choice"] == "safe"
