# SPDX-License-Identifier: Apache-2.0
"""
Pydantic models for the System One API.

System One models answer typed questions about a piece of content with
calibrated probabilities instead of generated text. These models define the
request and response schemas for:
- /v1/systemone endpoint

Field names, types and constraints mirror TypeSafe's published OpenAPI 0.2.0
(https://api.typesafe.ai/openapi.json), so the TypeSafe SDKs work against
oMLX by pointing their base URL at it.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

JSONContent = str | dict[str, Any] | list[Any]
"""Text, or structured JSON the model reads as text."""

Described = JSONContent | None
"""An optional instruction or description; ``None`` means none was given."""


class NoulCriteria(BaseModel):
    """What counts as a yes and as a no for a noul question."""

    true: Described = None
    false: Described = None


class NoulQuestion(BaseModel):
    """A yes/no question, or a statement to judge true or false."""

    type: Literal["noul"]
    instructions: Described = None
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseModel):
    """Select one option; ``criteria`` maps option names to descriptions."""

    type: Literal["choice"]
    instructions: Described = None
    criteria: dict[str, Described]


class ScoreQuestion(BaseModel):
    """Rate against ordered levels; a level's index is its score."""

    type: Literal["score"]
    instructions: Described = None
    criteria: list[JSONContent] = Field(min_length=1)


Question = Annotated[
    NoulQuestion | ChoiceQuestion | ScoreQuestion,
    Field(discriminator="type"),
]


class SystemOneRequest(BaseModel):
    """Request for POST /v1/systemone."""

    state: JSONContent
    """The content every question refers to."""

    model: str
    """ID or alias of a model that supports System One reads."""

    questions: dict[str, Question] = Field(min_length=1)
    """Questions keyed by names the caller chooses; answers use the same keys.
    The names are never shown to the model."""


class NoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float
    """Probability of yes, from 0 to 1."""


class ChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    """The option with the highest probability."""
    confidence: float
    """How concentrated ``probabilities`` is, from 0 (uniform) to 1."""
    probabilities: dict[str, float]
    """Probability of each option, keyed by option name."""


class ScoreAnswer(BaseModel):
    type: Literal["score"]
    score: float
    """Expected level: the probability-weighted mean of the level indices."""
    confidence: float
    """How concentrated ``probabilities`` is, from 0 (uniform) to 1."""
    legend: dict[str, JSONContent]
    """Each level index (as a string) mapped back to its description."""
    probabilities: dict[str, float]
    """Probability of each level, keyed like ``legend``."""


Answer = Annotated[
    NoulAnswer | ChoiceAnswer | ScoreAnswer,
    Field(discriminator="type"),
]


class SystemOneUsage(BaseModel):
    input_tokens: int
    """Prompt tokens the model prefilled to answer the request."""
    output_tokens: int
    """Tokens generated; reads generate none."""


class SystemOneResponse(BaseModel):
    """Response for POST /v1/systemone."""

    model: str
    """The model that answered, which may differ from a requested alias."""
    answers: dict[str, Answer]
    """One answer per question, keyed and ordered like ``questions``."""
    usage: SystemOneUsage
