# SPDX-License-Identifier: Apache-2.0
"""
Pydantic models for the TypeSafe System One API.

These models define the request schema for:
- /v1/systemone endpoint (decision models such as Clef and OpenJev)

Each decision model applies its own rules to ``instructions`` and
``criteria``; this schema only checks the shared shape.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class SystemOneQuestion(BaseModel):
    """One typed question about the state."""

    type: Literal["noul", "choice", "score"]
    """noul: yes/no probability. choice: one option key. score: ordered level."""

    instructions: Any = None
    """The question text, or any JSON value."""

    criteria: dict[str, Any] | list[Any] | None = None
    """choice: option -> description or null. score: ordered level
    descriptions. noul: optional {"true": ..., "false": ...} descriptions."""


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

    truncate: bool = True
    """Clef only: cut the state to fit the context instead of returning 413."""

    @field_validator("state")
    @classmethod
    def _state_present(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("state must not be null")
        return value
