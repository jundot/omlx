# SPDX-License-Identifier: Apache-2.0
"""
LiquidAI d1 decision models on the LFM2-VL backbone.

Older d1 checkpoints (and the code-review conversions) declare ``D1Model`` on
an ``lfm2_vl`` config, which mlx-vlm serves from ``mlx_vlm.models.d1``: the
state is read as a trunk, every question continues it as a branch, and the
option log-probabilities are read off the next-token logits at the answer
slot. This wrapper drives that package and keeps the family's answer shape, so
``/v1/systemone`` gets the model's own readout instead of the letter
log-probability probes of the chat path.

The LFM3 d1 wrapper (:mod:`omlx.models.d1`) is untouched: it reads a whole set
of questions in one forward pass. This backbone reads one branch per question,
so a request is one forward pass and no set attention.
"""

from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

from mlx_vlm.models.d1.prompt import as_question, readout
from mlx_vlm.utils import load as vlm_load

from ..utils.model_loading import materialize_lazy_state, maybe_apply_pre_load_patches
from .decision import (
    ChunkSize,
    DecisionContextLengthError,
    DecisionRequestError,
    decode_images,
    round4,
)

# mlx-vlm's own default when the checkpoint declares no context limit.
DEFAULT_TOKEN_BUDGET = 65536


def _answer(question: dict, probabilities: list) -> dict:
    """One question's answer, in the shape the other d1 wrappers answer.

    mlx-vlm's ``d1.Model`` builds its own answer dict; this keeps the family
    keys (``noul`` / ``choice`` / ``score``) that ``/v1/systemone`` emits.
    """
    if question["type"] == "noul":
        return {"type": "noul", "noul": round4(probabilities[0])}
    best = max(range(len(probabilities)), key=probabilities.__getitem__)
    if question["type"] == "choice":
        names = list(question["criteria"])
        return {
            "type": "choice",
            "choice": names[best],
            "confidence": round4(probabilities[best]),
            "probabilities": {
                name: round4(p)
                for name, p in zip(names, probabilities, strict=True)
            },
        }
    return {
        "type": "score",
        "score": round4(sum(i * p for i, p in enumerate(probabilities))),
        "confidence": round4(probabilities[best]),
        "probabilities": {str(i): round4(p) for i, p in enumerate(probabilities)},
        "legend": {str(i): text for i, text in enumerate(question["criteria"])},
    }


@dataclass
class D1Lfm2Plan:
    """A validated request: its state and mlx-vlm's trunk and branch token ids."""

    questions: dict
    state: Any
    trunk: list
    branches: list
    vision: dict


class D1Lfm2Model:
    """A d1 checkpoint on the LFM2-VL backbone (mlx-vlm's ``d1`` package)."""

    has_vision = True

    def __init__(self, model_path: str, trust_remote_code: bool = False):
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.model: Any = None
        self.processor: Any = None
        self.tokenizer: Any = None
        self.limit: int | None = None

    def load(self) -> None:
        maybe_apply_pre_load_patches(self.model_path, for_vlm=True)
        model, processor = vlm_load(
            self.model_path, trust_remote_code=self.trust_remote_code
        )
        materialize_lazy_state(model)
        self.model = model
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        text_config = getattr(model.config, "text_config", None)
        self.limit = getattr(text_config, "max_position_embeddings", None)

    def close(self) -> None:
        self.model = None
        self.processor = None
        self.tokenizer = None

    def encode(self, request: dict, truncate: bool = True) -> D1Lfm2Plan:
        """Validate the questions and build the trunk and branch token ids.

        d1 never truncates; ``truncate`` is accepted for API symmetry.
        """
        if request.get("videos"):
            raise DecisionRequestError("videos: d1 models read images only")
        try:
            questions = {
                name: as_question(question)
                for name, question in request["questions"].items()
            }
        except ValueError as exc:
            raise DecisionRequestError(str(exc)) from exc
        state = request["state"]
        images = decode_images(request.get("images"))
        trunk, branches, vision = self.model.decision_inputs(
            self.processor, state, list(questions.values()), images
        )
        total = len(trunk) + sum(len(branch) for branch in branches)
        if self.limit is not None and total > self.limit:
            raise DecisionContextLengthError(
                f"the request needs {total} tokens; the limit is {self.limit}"
            )
        return D1Lfm2Plan(questions, state, trunk, branches, vision)

    def run(
        self, plan: D1Lfm2Plan, chunk_size: ChunkSize
    ) -> Generator[int, None, dict]:
        """Read every question's options off one forward pass, then build answers.

        mlx-vlm chunks the trunk itself, so one step here is the whole forward
        pass; ``chunk_size`` is accepted for symmetry with the layer-stepped
        wrappers.
        """
        budget = self.limit or DEFAULT_TOKEN_BUDGET
        logprobs = self.model.answer_logprobs(
            plan.trunk, plan.branches, plan.vision, budget
        )
        read = len(plan.trunk) + sum(len(branch) for branch in plan.branches)
        yield read
        probabilities = [
            readout(self.tokenizer, question, row)
            for question, row in zip(plan.questions.values(), logprobs, strict=True)
        ]
        answers = {
            name: _answer(question, probs)
            for (name, question), probs in zip(
                plan.questions.items(), probabilities, strict=True
            )
        }
        return {"answers": answers, "input_tokens": read}
