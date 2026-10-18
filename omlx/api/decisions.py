# SPDX-License-Identifier: Apache-2.0
"""
OpenAI Decisions API on top of any chat model.

Each readout renders one question after the input as a single user turn,
lists the options under letters, and continues an assistant reply that starts
with ``Answer:``. Continuing the reply makes the chat template emit its own
answer header (a closed think block, a Harmony final channel), so the next
token is the answer letter on templates that would otherwise open reasoning
first. A logits processor reads the next-token log-probabilities of the
option letters while one token is generated, and the letters are renormalized
into the answer probabilities. When the letters hold too little of the
next-token mass the model is not answering, and the question is a refusal.
Every readout starts with the same input, so the prefix cache serves it after
the first readout.

Decision models (Clef, OpenJev) answer through their own heads instead: the
request is translated to a System One request and back.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np
from pydantic import ValidationError

from ..models.decision import round4
from ..models.openjev import (
    choice_confidence,
    compose_groups,
    description_text,
    score_confidence,
)
from .decisions_models import (
    ChoiceQuestion,
    DecisionChoiceOption,
    DecisionInputImage,
    DecisionInputMessage,
    DecisionInputText,
    DecisionRequest,
    DecisionScoreLevel,
    PredicateQuestion,
    ScoreQuestion,
    choice_key,
)
from .systemone_models import SystemOneRequest

LETTERS = [chr(code) for code in range(ord("A"), ord("Z") + 1)]
PREDICATE_OPTIONS = [
    ("Yes", "The statement is true."),
    ("No", "The statement is false."),
]
SCORE_SUFFIX = " Rate along the ordered levels below (lowest first)."
ANSWER_PREFIX = "Answer:"
READOUT_TEMPLATE_KWARGS = {"enable_thinking": False}
# Minimum next-token probability the option letters must hold together for
# the readout to count as an answer.
MIN_LETTER_MASS = 0.5
READOUT_CONCURRENCY = 8


class DecisionsRequestError(ValueError):
    """The request cannot be answered by the selected model."""


class OptionLogitsCapture:
    """Logits processor that records option-letter log-probabilities.

    Reads the first position it sees and returns the logits unchanged.
    """

    def __init__(self, letter_ids: list[list[int]]):
        self._owners = [i for i, ids in enumerate(letter_ids) for _ in ids]
        self._ids = mx.array([t for ids in letter_ids for t in ids])
        self._count = len(letter_ids)
        self.logprobs: list[float] | None = None

    def __call__(self, tokens: Any, logits: mx.array) -> mx.array:
        if self.logprobs is None:
            row = logits.reshape(-1, logits.shape[-1])[-1].astype(mx.float32)
            picked = (row - mx.logsumexp(row))[self._ids].tolist()
            merged = np.full(self._count, -np.inf)
            for owner, value in zip(self._owners, picked, strict=True):
                merged[owner] = np.logaddexp(merged[owner], value)
            self.logprobs = merged.tolist()
        return logits


def letter_token_ids(tokenizer: Any) -> list[list[int]]:
    """Single-token spellings of each letter, bare and after a space."""
    result = []
    seen: set[int] = set()
    for letter in LETTERS:
        ids = []
        for text in (letter, " " + letter):
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if len(encoded) == 1 and encoded[0] not in ids:
                ids.append(int(encoded[0]))
        if not ids or seen.intersection(ids):
            raise DecisionsRequestError(
                "the model's tokenizer has no distinct single token for the "
                f"option letter {letter!r}"
            )
        seen.update(ids)
        result.append(ids)
    return result


def input_parts(request: DecisionRequest) -> list[dict]:
    """Content parts of the input in chat-completions form."""
    if isinstance(request.input, str):
        return [{"type": "text", "text": request.input}]
    parts = []
    for message in request.input:
        content = message.content
        if isinstance(content, str):
            parts.append({"type": "text", "text": content})
            continue
        for part in content:
            if isinstance(part, DecisionInputImage):
                parts.append(
                    {"type": "image_url", "image_url": {"url": part.image_url}}
                )
            else:
                parts.append({"type": "text", "text": part.text})
    return parts


def question_options(question: Any) -> tuple[str, list[tuple[str, str]]]:
    if isinstance(question, ChoiceQuestion):
        return question.instructions, [
            (choice_key(c.value), c.description or "") for c in question.choices
        ]
    if isinstance(question, ScoreQuestion):
        return question.instructions + SCORE_SUFFIX, [
            (level.label, level.description or "") for level in question.levels
        ]
    return question.instructions, PREDICATE_OPTIONS


def render_question(instructions: str, options: list[tuple[str, str]]) -> str:
    lines = "\n".join(
        f"{LETTERS[i]}. {key}: {desc}" if desc else f"{LETTERS[i]}. {key}"
        for i, (key, desc) in enumerate(options)
    )
    return (
        f"Question: {instructions}\nOptions:\n{lines}"
        "\n\nReply with the letter of the best option only."
    )


def build_messages(
    parts: list[dict], question_text: str, multimodal: bool
) -> list[dict]:
    if multimodal:
        content: Any = [*parts, {"type": "text", "text": "\n\n" + question_text}]
    else:
        texts = [p["text"] for p in parts]
        content = "\n\n".join([*texts, question_text])
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": ANSWER_PREFIX},
    ]


def option_groups(count: int) -> list[range]:
    """Split options into near-equal readouts that each fit in the letters."""
    if count <= len(LETTERS):
        return [range(count)]
    groups = -(-count // len(LETTERS))
    size = -(-count // groups)
    return [range(start, min(count, start + size)) for start in range(0, count, size)]


@dataclass
class _Readout:
    probs: list[float] | None
    prompt_tokens: int
    cached_tokens: int


async def _readout(
    chat: Callable,
    parts: list[dict],
    multimodal: bool,
    letter_ids: list[list[int]],
    instructions: str,
    options: list[tuple[str, str]],
    require_answer: bool = True,
) -> _Readout:
    """Option probabilities, or None when the model gives no letter.

    ``require_answer`` applies the letter-mass check. A group readout of a
    large choice question skips it: the model may rightly reject every
    option in a group, and the readout over the winners still applies it.
    """
    capture = OptionLogitsCapture(letter_ids[: len(options)])
    output = await chat(
        build_messages(parts, render_question(instructions, options), multimodal),
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        chat_template_kwargs=dict(READOUT_TEMPLATE_KWARGS),
        is_partial=True,
        logits_observer=capture,
    )
    if capture.logprobs is None:
        raise DecisionsRequestError(
            "the model's engine does not expose next-token logits"
        )
    tokens = output.prompt_tokens, output.cached_tokens
    scores = np.asarray(capture.logprobs, dtype=np.float64)
    if not np.isfinite(scores).any():
        return _Readout(None, *tokens)
    top = scores[np.isfinite(scores)].max()
    weights = np.exp(scores - top)
    if require_answer and np.exp(top) * weights.sum() < MIN_LETTER_MASS:
        return _Readout(None, *tokens)
    return _Readout((weights / weights.sum()).tolist(), *tokens)


async def _readout_options(
    chat: Callable,
    parts: list[dict],
    multimodal: bool,
    letter_ids: list[list[int]],
    instructions: str,
    options: list[tuple[str, str]],
    require_answer: bool = True,
) -> tuple[list[float] | None, list[_Readout]]:
    """Compose group probabilities until the winners fit in one readout."""
    groups = option_groups(len(options))
    readouts = [
        await _readout(
            chat,
            parts,
            multimodal,
            letter_ids,
            instructions,
            [options[i] for i in group],
            require_answer=require_answer and len(groups) == 1,
        )
        for group in groups
    ]
    if any(r.probs is None for r in readouts):
        return None, readouts
    if len(readouts) == 1:
        probs = readouts[0].probs
    else:
        parts_probs = [r.probs for r in readouts]
        winners = [
            options[group[int(np.argmax(p))]]
            for group, p in zip(groups, parts_probs, strict=True)
        ]
        final_probs, final_readouts = await _readout_options(
            chat, parts, multimodal, letter_ids, instructions, winners, require_answer
        )
        readouts.extend(final_readouts)
        if final_probs is None:
            return None, readouts
        probs = compose_groups(parts_probs, final_probs)
    return probs, readouts


async def _answer_question(
    chat: Callable,
    parts: list[dict],
    multimodal: bool,
    letter_ids: list[list[int]],
    question: Any,
    require_answer: bool = True,
) -> tuple[dict, list[_Readout]]:
    instructions, options = question_options(question)
    probs, readouts = await _readout_options(
        chat, parts, multimodal, letter_ids, instructions, options, require_answer
    )
    answer = (
        {"type": "refusal", "name": question.name}
        if probs is None
        else build_answer(question, probs)
    )
    return answer, readouts


def build_answer(question: Any, probs: list[float]) -> dict:
    """Answer object for option probabilities in the question's order."""
    if isinstance(question, PredicateQuestion):
        return {
            "type": "predicate",
            "name": question.name,
            "probability": round4(probs[0]),
        }
    if isinstance(question, ChoiceQuestion):
        best = int(np.argmax(probs))
        return {
            "type": "choice",
            "name": question.name,
            "choice": question.choices[best].value,
            "probabilities": [
                {"value": c.value, "probability": round4(p)}
                for c, p in zip(question.choices, probs, strict=True)
            ],
            "confidence": round4(choice_confidence(probs)),
        }
    return {
        "type": "score",
        "name": question.name,
        "score": round4(sum(i * p for i, p in enumerate(probs))),
        "probabilities": [
            {"value": i, "label": level.label, "probability": round4(p)}
            for i, (level, p) in enumerate(zip(question.levels, probs, strict=True))
        ],
        "confidence": round4(score_confidence(probs)),
    }


async def decide(
    engine: Any,
    request: DecisionRequest,
    multimodal: bool,
    prepare: Callable[[list[dict]], dict] | None = None,
    require_answer: bool = True,
) -> dict:
    """Answer every question with a chat model; returns answers and tokens.

    ``prepare`` checks the messages of each readout before it runs and
    returns extra ``engine.chat`` kwargs. ``require_answer=False`` skips the
    letter-mass check.
    """
    parts = input_parts(request)
    if not multimodal and any(p["type"] == "image_url" for p in parts):
        raise DecisionsRequestError("this model does not accept image input")
    letter_ids = letter_token_ids(engine.tokenizer)

    async def chat(messages, **kwargs):
        if prepare is not None:
            kwargs.update(prepare(messages))
        return await engine.chat(messages=messages, **kwargs)

    async def answer(question):
        return await _answer_question(
            chat, parts, multimodal, letter_ids, question, require_answer
        )

    # The first readout fills the prefix cache for the shared input.
    first = await answer(request.questions[0])
    gate = asyncio.Semaphore(READOUT_CONCURRENCY)

    async def gated(question):
        async with gate:
            return await answer(question)

    tasks = [asyncio.create_task(gated(q)) for q in request.questions[1:]]
    try:
        rest = await asyncio.gather(*tasks)
    except BaseException:
        # A failed question fails the request: stop the other readouts (the
        # engine aborts a cancelled request) before the lease is released.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    results = [first, *rest]
    readouts = [r for _, question_readouts in results for r in question_readouts]
    return {
        "answers": [a for a, _ in results],
        "input_tokens": sum(r.prompt_tokens for r in readouts),
        "cached_tokens": sum(r.cached_tokens for r in readouts),
    }


def to_systemone(request: DecisionRequest) -> dict:
    """System One request for a decision model."""
    texts, images = [], []
    for part in input_parts(request):
        if part["type"] == "image_url":
            images.append(part["image_url"]["url"])
        else:
            texts.append(part["text"])
    questions = {}
    for question in request.questions:
        if isinstance(question, ChoiceQuestion):
            questions[question.name] = {
                "type": "choice",
                "instructions": question.instructions,
                "criteria": {
                    choice_key(c.value): c.description for c in question.choices
                },
            }
        elif isinstance(question, ScoreQuestion):
            questions[question.name] = {
                "type": "score",
                "instructions": question.instructions,
                "criteria": [
                    (
                        f"{level.label}: {level.description}"
                        if level.description
                        else level.label
                    )
                    for level in question.levels
                ],
            }
        else:
            questions[question.name] = {
                "type": "noul",
                "instructions": question.instructions,
                "criteria": None,
            }
    return {
        "model": request.model,
        "state": "\n\n".join(texts) if texts else "(see image)",
        "questions": questions,
        "images": images or None,
    }


def from_systemone(request: DecisionRequest, answers: dict) -> list[dict]:
    """Decisions answers from System One answers."""
    result = []
    for question in request.questions:
        answer = answers[question.name]
        if isinstance(question, PredicateQuestion):
            result.append(
                {
                    "type": "predicate",
                    "name": question.name,
                    "probability": answer["noul"],
                }
            )
        elif isinstance(question, ChoiceQuestion):
            probs = answer["probabilities"]
            values = {choice_key(c.value): c.value for c in question.choices}
            result.append(
                {
                    "type": "choice",
                    "name": question.name,
                    "choice": values[answer["choice"]],
                    "probabilities": [
                        {"value": c.value, "probability": probs[choice_key(c.value)]}
                        for c in question.choices
                    ],
                    "confidence": answer["confidence"],
                }
            )
        else:
            probs = answer["probabilities"]
            result.append(
                {
                    "type": "score",
                    "name": question.name,
                    "score": answer["score"],
                    "probabilities": [
                        {"value": i, "label": level.label, "probability": probs[str(i)]}
                        for i, level in enumerate(question.levels)
                    ],
                    "confidence": answer["confidence"],
                }
            )
    return result


def decisions_request(request: SystemOneRequest) -> DecisionRequest:
    """Decisions request for a System One request to a chat model."""
    if request.videos:
        raise DecisionsRequestError("this model does not take videos")
    # model_construct skips the Decisions API limits, which System One lacks.
    questions = []
    for name, question in request.questions.items():
        criteria = question.criteria
        instructions = description_text(question.instructions) or name
        if question.type == "score":
            if not isinstance(criteria, list) or not criteria:
                raise DecisionsRequestError(
                    "score criteria must be a non-empty array of levels"
                )
            levels = [DecisionScoreLevel(label=description_text(c)) for c in criteria]
            questions.append(
                ScoreQuestion.model_construct(
                    type="score", name=name, instructions=instructions, levels=levels
                )
            )
            continue
        if question.type == "choice":
            if not isinstance(criteria, dict) or not criteria:
                raise DecisionsRequestError(
                    "choice criteria must be a non-empty map of option to "
                    "description or null"
                )
            options = [(key, description_text(desc)) for key, desc in criteria.items()]
        else:
            if criteria is not None and not isinstance(criteria, dict):
                raise DecisionsRequestError("noul criteria must be an object")
            criteria = criteria or {}
            # Same prompt as a predicate; criteria replace the option texts.
            options = [
                (value, description_text(criteria.get(side)) or default)
                for (value, default), side in zip(
                    PREDICATE_OPTIONS, ("true", "false"), strict=True
                )
            ]
        choices = [DecisionChoiceOption(value=v, description=d) for v, d in options]
        questions.append(
            ChoiceQuestion.model_construct(
                type="choice", name=name, instructions=instructions, choices=choices
            )
        )
    text = description_text(request.state)
    content: Any = text
    if request.images:
        try:
            images = [
                DecisionInputImage(type="input_image", image_url=url)
                for url in request.images
            ]
        except ValidationError as e:
            raise DecisionsRequestError(
                "images must be base64 data URLs (data:image/...)"
            ) from e
        text_part = DecisionInputText(type="input_text", text=text)
        content = [DecisionInputMessage(content=[*images, text_part])]
    return DecisionRequest.model_construct(
        model=request.model, input=content, questions=questions
    )


def systemone_answers(request: SystemOneRequest, answers: list[dict]) -> dict:
    """System One answers from the answers to :func:`decisions_request`."""
    result = {}
    for (name, question), answer in zip(
        request.questions.items(), answers, strict=True
    ):
        if answer["type"] == "refusal":
            raise DecisionsRequestError(
                f"the model gives no finite option score for question {name!r}"
            )
        probs = [p["probability"] for p in answer["probabilities"]]
        if question.type == "noul":
            result[name] = {"type": "noul", "noul": probs[0]}
        elif question.type == "choice":
            result[name] = {
                "type": "choice",
                "choice": answer["choice"],
                "confidence": answer["confidence"],
                "probabilities": dict(zip(question.criteria, probs, strict=True)),
            }
        else:
            result[name] = {
                "type": "score",
                "score": answer["score"],
                "confidence": answer["confidence"],
                "legend": {str(i): level for i, level in enumerate(question.criteria)},
                "probabilities": {str(i): p for i, p in enumerate(probs)},
            }
    return result
