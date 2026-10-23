# SPDX-License-Identifier: Apache-2.0
"""Opt-in Mammoth-1 / Elephant-1 validation; never downloads weights or code.

OMLX_BASELABS_MODEL_PATH points to a local MLX conversion. For independent
parity, OMLX_BASELABS_REFERENCE_PATH may point to decision_mlx.py from
0x36/elephant-1-9B-MLX-4bit at 16de0e500eea4e44fc7db79928c03c53e324bcb0.
That port of the published OneJev runtime runs on the same quantized weights;
this comparison does not measure quantization error against the bf16 model.
"""

import gc
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.slow


def _requests():
    yield {
        "state": {
            "ticket": "I was charged twice this month. Please refund the duplicate charge."
        },
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "billing": "Payments and refunds",
                    "technical": "Software bugs",
                    "other": None,
                },
            },
            "refund": {
                "type": "noul",
                "instructions": "Is the customer requesting a refund?",
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this?",
                "criteria": ["Can wait", "Today", "Blocking the customer now"],
            },
        },
    }
    yield {
        "state": {
            "message": "请退还重复扣款。",
            "amount": 42,
            "evidence": [True, None, "literal evidence"],
        },
        "questions": {
            "correlation-metadata-only": {
                "type": "choice",
                "instructions": {"task": "Route this support message"},
                "criteria": {
                    "refund": {"means": "Wants money back"},
                    "delivery": ["Shipment", "Tracking"],
                },
            },
            "single": {"type": "choice", "criteria": {"only": None}},
            "default": {
                "type": "noul",
                "criteria": {
                    "true": "A duplicate payment occurred.",
                    "false": "No duplicate payment occurred.",
                },
            },
            "levels": {"type": "score", "criteria": [f"Level {i}" for i in range(10)]},
        },
    }
    yield {
        "state": "The requested destination is destination 254.",
        "questions": {
            "wide": {
                "type": "choice",
                "instructions": "Choose the requested destination.",
                "criteria": {f"destination {i}": None for i in range(255)},
            }
        },
    }


@pytest.fixture(scope="module")
def native_model():
    path = os.environ.get("OMLX_BASELABS_MODEL_PATH")
    if not path:
        pytest.skip("set OMLX_BASELABS_MODEL_PATH to an existing local MLX checkpoint")
    from omlx.models.baselabs import BaseLabsModel

    model = BaseLabsModel(str(Path(path).resolve()))
    model.load()
    try:
        yield model
    finally:
        import mlx.core as mx

        model.close()
        gc.collect()
        mx.clear_cache()


def _drain(steps):
    while True:
        try:
            next(steps)
        except StopIteration as stop:
            return stop.value


def test_real_decisions_are_constrained_and_repeatable(native_model):
    from omlx.model_discovery import decision_kind

    assert decision_kind(Path(native_model.model_path)) == "baselabs"
    for request in _requests():
        plan = native_model.encode(request)
        actual = _drain(native_model.run(plan, lambda: 128))
        assert set(actual["answers"]) == set(request["questions"])
        reordered = native_model.encode(
            {**request, "questions": dict(reversed(list(request["questions"].items())))}
        )
        repeated = _drain(native_model.run(reordered, lambda: 4096))
        for key, answer in actual["answers"].items():
            assert answer["type"] == request["questions"][key]["type"]
            if answer["type"] == "noul":
                assert 0 <= answer["noul"] <= 1
                assert answer["noul"] == pytest.approx(
                    repeated["answers"][key]["noul"], abs=2e-4
                )
            else:
                probabilities = list(answer["probabilities"].values())
                assert all(0 <= value <= 1 for value in probabilities)
                assert sum(probabilities) == pytest.approx(
                    1, abs=len(probabilities) * 5e-5
                )
                assert probabilities == pytest.approx(
                    list(repeated["answers"][key]["probabilities"].values()), abs=2e-4
                )


def test_real_decisions_match_independent_mlx_reference(native_model):
    path = os.environ.get("OMLX_BASELABS_REFERENCE_PATH")
    if not path:
        pytest.skip(
            "set OMLX_BASELABS_REFERENCE_PATH to the independently downloaded MLX port"
        )
    spec = importlib.util.spec_from_file_location("_baselabs_reference", path)
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    # Share only the weights/tokenizer, avoiding a second 27B allocation.
    oracle = reference.DecisionEngine.__new__(reference.DecisionEngine)
    oracle.model_path = native_model.model_path
    oracle.model = native_model.backbone.model
    oracle.text_model = native_model.backbone._text_model
    oracle.tokenizer = native_model.backbone.tokenizer
    oracle.style = reference.DEFAULT_STYLE
    oracle.template_kwargs = {"enable_thinking": False}
    oracle.slot_ids = oracle._verify_slots()
    oracle.slot_head = oracle._slot_head_fp32()
    oracle.calibration = reference.Calibration()
    oracle.max_branch_tokens = 16_384
    oracle.max_request_tokens = 65_536
    oracle.info = SimpleNamespace(fork_mode="sequential")
    largest = 0.0
    questions = 0
    for body in _requests():
        request = reference.SystemOneRequest.model_validate(body)
        rendered = [
            reference.render_question(key, question)
            for key, question in request.questions.items()
        ]
        prefix, suffixes = oracle._prefix_and_suffix_ids(request.state, rendered)
        plan = native_model.encode(body)
        assert np.array_equal(prefix, plan.prefix_ids)
        assert all(np.array_equal(a, b) for a, b in zip(suffixes, plan.suffix_ids))
        expected, _ = oracle.decide(request)
        actual = _drain(native_model.run(plan, lambda: 128))
        assert actual["input_tokens"] == expected.usage.input_tokens
        for key, answer in actual["answers"].items():
            target = expected.answers[key].model_dump()
            if answer["type"] == "noul":
                values, reference_values = [answer["noul"]], [target["noul"]]
            else:
                values = list(answer["probabilities"].values())
                reference_values = list(target["probabilities"].values())
                assert answer["confidence"] == pytest.approx(
                    target["confidence"], abs=2e-4
                )
                if answer["type"] == "choice":
                    assert answer["choice"] == target["choice"]
                else:
                    assert answer["score"] == pytest.approx(target["score"], abs=2e-4)
            largest = max(
                largest, float(np.max(np.abs(np.asarray(values) - reference_values)))
            )
            assert values == pytest.approx(reference_values, abs=2e-4)
            questions += 1
    print(f"matched {questions} questions; max probability difference {largest:.6f}")
