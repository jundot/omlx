# SPDX-License-Identifier: Apache-2.0
"""
Pydantic models for the OpenAI Decisions API.

These models define the request schema for:
- /v1/decisions endpoint (any chat model, and decision models)
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StrictBool, field_validator, model_validator

MAX_QUESTIONS = 200
MAX_CHOICES = 255
MAX_SCORE_LEVELS = 10


def choice_key(value: str | bool) -> str:
    """Text form of a choice value; booleans use their JSON spelling."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


class DecisionInputText(BaseModel):
    type: Literal["input_text"]
    text: str


class DecisionInputImage(BaseModel):
    type: Literal["input_image"]
    image_url: str
    """Base64 image data URL."""

    @field_validator("image_url")
    @classmethod
    def _data_url(cls, value: str) -> str:
        if not value.startswith("data:image/"):
            raise ValueError("image_url must be a base64 data URL (data:image/...)")
        return value


DecisionContentPart = Annotated[
    DecisionInputText | DecisionInputImage, Field(discriminator="type")
]


class DecisionInputMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str | list[DecisionContentPart]


class DecisionChoiceOption(BaseModel):
    value: StrictBool | str
    description: str | None = None


class DecisionScoreLevel(BaseModel):
    label: str
    description: str | None = None


class _QuestionBase(BaseModel):
    name: str = Field(..., min_length=1)
    instructions: str = Field(..., min_length=1)


class PredicateQuestion(_QuestionBase):
    """Probability that a statement about the input is true."""

    type: Literal["predicate"]


class ChoiceQuestion(_QuestionBase):
    """One value out of unordered options."""

    type: Literal["choice"]
    choices: list[DecisionChoiceOption] = Field(
        ..., min_length=2, max_length=MAX_CHOICES
    )

    @field_validator("choices", mode="before")
    @classmethod
    def _plain_values(cls, value):
        if isinstance(value, list):
            return [{"value": v} if isinstance(v, (str, bool)) else v for v in value]
        return value

    @field_validator("choices")
    @classmethod
    def _unique_values(cls, value: list[DecisionChoiceOption]):
        # true and "true" render the same, so they count as duplicates.
        if len({choice_key(c.value) for c in value}) != len(value):
            raise ValueError("choice values must be unique")
        return value


class ScoreQuestion(_QuestionBase):
    """A level on an ordered scale, indexed from 0."""

    type: Literal["score"]
    levels: list[DecisionScoreLevel] = Field(
        ..., min_length=2, max_length=MAX_SCORE_LEVELS
    )

    @field_validator("levels", mode="before")
    @classmethod
    def _plain_labels(cls, value):
        if isinstance(value, list):
            return [{"label": v} if isinstance(v, str) else v for v in value]
        return value


DecisionQuestion = Annotated[
    PredicateQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")
]


class DecisionRequest(BaseModel):
    """Request for ``POST /v1/decisions``."""

    model: str
    input: str | list[DecisionInputMessage]
    questions: list[DecisionQuestion] = Field(
        ..., min_length=1, max_length=MAX_QUESTIONS
    )

    @field_validator("input")
    @classmethod
    def _input_present(cls, value):
        if isinstance(value, list) and not value:
            raise ValueError("input must not be empty")
        return value

    @model_validator(mode="after")
    def _unique_names(self):
        names = [q.name for q in self.questions]
        if len(set(names)) != len(names):
            raise ValueError("question names must be unique")
        return self
