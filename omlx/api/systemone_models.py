# SPDX-License-Identifier: Apache-2.0
"""
Pydantic models for the TypeSafe System One API.

These models define the request schema for:
- /v1/systemone endpoint (decision models such as Clef, OpenJev and Decision 3.0,
  and chat models)

Each decision model applies its own rules to ``instructions`` and
``criteria``; this schema only checks the shared shape.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class SystemOneQuestion(BaseModel):
    """One typed question about the state."""

    type: Literal["noul", "choice", "score", "locate"]
    """noul: yes/no probability. choice: one option key. score: ordered level.

    locate: verbatim spans of the state (Perplexity Decider only); the question
    key is the default label and ``instructions`` replaces it.
    """

    instructions: Any = None
    """The question text, or any JSON value."""

    criteria: dict[str, Any] | list[Any] | None = None
    """choice: option -> description or null. score: ordered level
    descriptions. noul: optional {"true": ..., "false": ...} descriptions."""

    cutoff: float | None = None
    """locate: span probability floor in (0, 1]; the checkpoint default applies
    when null. Raise it to drop weak spans (a 4-bit quant drifts at the release
    floor)."""

    @field_validator("cutoff")
    @classmethod
    def _check_cutoff(cls, value: float | None) -> float | None:
        if value is not None and not 0 < value <= 1:
            raise ValueError("cutoff must be in (0, 1]")
        return value


class SystemOneRequest(BaseModel):
    """Request for ``POST /v1/systemone``."""

    model: str
    """ID of the model to use."""

    state: Any = Field(...)
    """The content to evaluate: a string, an object or an array."""

    questions: dict[str, SystemOneQuestion] = Field(..., min_length=1)
    """Named questions, answered independently of their order."""

    images: list[str] | None = None
    """Base64 image data URIs. An oMLX extension to the System One API."""

    videos: list[str] | None = None
    """Base64 video data URIs, for models that read video (Decision 3.0)."""

    mode: Literal["separate", "isolated"] = "separate"
    """Decider models only: one sequence per question (``separate``) or one
    sequence for the whole request (``isolated``, cheaper on long states)."""

    truncate: bool = True
    """Clef only: cut the state to fit the context instead of returning 413."""

    @field_validator("state")
    @classmethod
    def _state_present(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("state must not be null")
        return value
