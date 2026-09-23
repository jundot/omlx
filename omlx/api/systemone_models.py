# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for the Jev-compatible System One endpoint.

Wire shape is the TypeSafe Jev OpenAPI 0.2.0 subset that ``razorback16/openjev``
implements (Apache-2.0 — the question/answer shapes are reproduced so a client
written against their SDK reads the same JSON here):

    POST /v1/systemone
        {"model": …, "state": …, "questions": {"urgent": {"type": "noul", …}}}

    {"model": …, "answers": {"urgent": {"type": "noul", "noul": 0.987}},
     "usage": {"input_tokens": 214, "output_tokens": 0}}

Two deliberate properties of these models:

* ``extra="forbid"`` everywhere. A question with a misspelled ``criteria`` key
  would otherwise compile into a schema whose labels are silently the defaults,
  and the answer would look like a model opinion rather than a client typo.
* the question union is **discriminated** on ``type``, so FastAPI's 422 names
  the question and the field rather than reporting four failed branches.

``oQ``/omlx extensions (``images``, ``steps``, ``samples``, ``think``,
``sequential``, ``queue``, ``seed``) are additive and all default to plain Jev
behaviour: a request that omits them is a plain Jev request.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ChoiceAnswer",
    "ChoiceQuestion",
    "NoulAnswer",
    "NoulQuestion",
    "ScoreAnswer",
    "ScoreQuestion",
    "SystemOneAnswer",
    "SystemOneQuestion",
    "SystemOneRequest",
    "SystemOneResponse",
    "SystemOneUsage",
]

#: Upstream's per-choice option ceiling, mirrored in ``schema.MAX_LABEL_IDS``.
#: A choice wider than this is a schema the caller must see, not a truncation.
MAX_CRITERIA_OPTIONS = 128

#: Answer-template levels. ``schema.INDEXED_ABOVE`` switches the template past
#: ten questions; the cap here keeps one request from turning into thousands of
#: canvas forwards.
MAX_QUESTIONS = 64

#: ``structured_read`` accepts 1-32 draws per slot and rejects anything else.
MAX_SAMPLES = 32


class _Strict(BaseModel):
    """Base: unknown keys are a 422, not a shrug."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------
# Request
# --------------------------------------------------------------------------


class NoulCriteria(_Strict):
    """What counts as yes and what counts as no. Both sides optional."""

    true: str = ""
    false: str = ""


class NoulQuestion(_Strict):
    """A yes/no read: one number, the mass on ``true``."""

    type: Literal["noul"]
    instructions: str | dict[str, Any] | list[Any] | None = None
    criteria: NoulCriteria | None = None


class ChoiceQuestion(_Strict):
    """A labelled choice. Keys are the labels the client gets back."""

    type: Literal["choice"]
    instructions: str | dict[str, Any] | list[Any] | None = None
    criteria: dict[str, str] = Field(min_length=2, max_length=MAX_CRITERIA_OPTIONS)


class ScoreQuestion(_Strict):
    """An ordinal scale. Order is meaning: index 0 is the first level."""

    type: Literal["score"]
    instructions: str | dict[str, Any] | list[Any] | None = None
    criteria: list[str] = Field(min_length=2, max_length=10)


SystemOneQuestion = Annotated[
    NoulQuestion | ChoiceQuestion | ScoreQuestion,
    Field(discriminator="type"),
]


class SystemOneRequest(_Strict):
    """One structured read over a state, answering a fixed set of questions."""

    model: str
    """omlx model id, or a configured alias resolving to a block-diffusion model."""

    state: str | dict[str, Any] | list[Any]
    """The situation being judged. Objects and arrays are serialized as JSON."""

    questions: dict[str, SystemOneQuestion] = Field(
        min_length=1, max_length=MAX_QUESTIONS
    )
    """Questions keyed by the id the client wants back. Order is the model's order."""

    # -- extensions; absent means plain Jev ------------------------------------

    images: list[str] | None = None
    """Base64 data URIs, placed ahead of the state. Not served yet (P4c)."""

    steps: int = Field(default=1, ge=1, le=8)
    """Denoise steps per read. Only ``1`` is served yet; the rest is P4b."""

    samples: int = Field(default=1, ge=1, le=MAX_SAMPLES)
    """Independent noise draws per slot, averaged. One prefill serves them all."""

    think: int = Field(default=0, ge=0, le=4096)
    """Tokens of thought before the read. Not served yet (P4d)."""

    sequential: bool = False
    """Chunk reads conditioned on earlier answers. Not served yet (P4d)."""

    queue: bool = True
    """``False`` answers 529 immediately when the diffusion lane is busy, instead
    of waiting behind the request already on it."""

    seed: int | None = None
    """Pin the noise draws. Absent means a fresh seed per request; the value used
    comes back in the ``x-systemone-seed`` response header, so a measurement stays
    reproducible after the fact."""


# --------------------------------------------------------------------------
# Response
# --------------------------------------------------------------------------


class NoulAnswer(BaseModel):
    """``noul`` is the renormalized mass on the ``true`` label."""

    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(BaseModel):
    """Argmax label plus the whole distribution over the criteria keys."""

    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    """Expected level, weighted by the distribution, plus the legend as sent."""

    type: Literal["score"] = "score"
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float


SystemOneAnswer = Annotated[
    NoulAnswer | ChoiceAnswer | ScoreAnswer,
    Field(discriminator="type"),
]


class SystemOneUsage(BaseModel):
    """Prompt tokens the model was conditioned on. Nothing is generated."""

    input_tokens: int
    output_tokens: int = 0
    """0 unless ``think`` generated text — a read answers without generating."""


class SystemOneResponse(BaseModel):
    """Answers keyed exactly as the request keyed its questions."""

    model: str
    answers: dict[str, Any]
    usage: SystemOneUsage
