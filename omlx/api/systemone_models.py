# SPDX-License-Identifier: Apache-2.0
"""
Pydantic models for the TypeSafe System One API.

These models define the request schema for:
- /v1/systemone endpoint (decision models such as Clef, OpenJev, Decision 3.0
  and LiquidAI d1, and chat models)

Each decision model applies its own rules to ``instructions`` and
``criteria``, and serves its own set of ``type``s; this schema only checks the
shared shape and lets the model refuse what it cannot read out.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class SystemOneQuestion(BaseModel):
    """One typed question about the state."""

    type: Literal["noul", "choice", "score", "box", "boxes", "number", "text", "date"]
    """noul: yes/no probability. choice: one option key, or any of them with
    ``multiple``. score: ordered level. box: one object in the request's
    picture. boxes: every instance of something in it. number: an estimate over
    a ``range``. text and date read a value out of the state.

    Which of these a model answers is its own decision: every model answers
    noul, choice and score, and the rest are refused with what is missing."""

    instructions: Any = None
    """The question text, or any JSON value."""

    criteria: dict[str, Any] | list[Any] | None = None
    """choice: option -> description or null. score: ordered level
    descriptions. noul: optional {"true": ..., "false": ...} descriptions."""

    multiple: bool | None = None
    """choice: answer with any number of options, each scored yes/no."""

    max: int | None = None
    """boxes: instances to read at most. A checkpoint caps it (16 for d1-2B);
    without it the model reads as many blocks as it was trained with."""

    range: list[float] | None = None
    """number: the lowest and the highest estimate."""

    step: float | None = None
    """number: how far apart two estimates are."""

    unit: str | None = None
    """number: the unit the estimate is in."""


class SystemOneRequest(BaseModel):
    """Request for ``POST /v1/systemone``."""

    model: str
    """ID of the model to use."""

    state: Any = None
    """The content to evaluate: a string, an object or an array. A d1 model also
    answers from a picture alone, where there is no state and the key can be
    left out; every other model reads its questions after a state, and says so."""

    questions: dict[str, SystemOneQuestion] = Field(..., min_length=1)
    """Named questions, answered independently of their order."""

    images: list[str] | None = None
    """Base64 image data URIs. An oMLX extension to the System One API."""

    videos: list[str] | None = None
    """Base64 video data URIs, for models that read video (Decision 3.0)."""

    truncate: bool = True
    """Clef only: cut the state to fit the context instead of returning 413."""
