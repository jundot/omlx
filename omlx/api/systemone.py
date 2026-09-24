# SPDX-License-Identifier: Apache-2.0
"""
System One reads on DiffusionGemma.

A discrete diffusion model denoises a whole canvas of tokens per decoder
pass. Seed the canvas with an answer template, leave one noise token where
each question's label goes, and a single read-only pass gives every slot a
probability distribution over that question's labels. The distribution is
the answer: it cannot go off-schema, and its confidence comes from the
model's own probabilities rather than from anything the model writes.

The request and answer shapes are TypeSafe's System One API (see
systemone_models); where that contract is silent, the read follows djev
(mmastrac/djev, the structured-read server of vllm-project/vllm#57250,
Apache-2.0): its prompts, answer templates, canvas layout, noise draws and
usage counts. The contract wins where the two differ, so question names stay
out of the prompt, a choice may have up to 255 options, a question with one
possible answer is answered without a read, and confidence is TypeSafe's.
"""

import json
import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..exceptions import InvalidRequestError
from .systemone_models import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneUsage,
)

logger = logging.getLogger(__name__)

SUPPORTED_MODEL_TYPES = frozenset({"diffusion_gemma"})

# The empty thought block DiffusionGemma's chat template leaves for the model
# to write. Every canvas opens with it, so the answers follow a closed thought.
_EMPTY_THOUGHT = "<|channel>thought\n<channel|>"
_TURN_CLOSE = "<turn|>"

# A read canvas holds the answer template and the turn close, padded to a
# multiple of _CANVAS_STEP. The width is the serving choice validated on
# JevBench, not the checkpoint's generation canvas_length; question lists
# that do not fit are answered in several reads.
_MAX_CANVAS = 64
_CANVAS_STEP = 16

_MAX_CHOICES = 255
_MAX_SCORE_LEVELS = 10

# djev's "auto" noise policy: one draw per read, and when any slot's entropy
# over its top _TOP_K tokens exceeds _AUTO_THRESHOLD, _AUTO_MAX draws in all,
# averaged. The extra draws reuse the read's prefill; the model setting
# system_one_rereads_enabled turns them off.
_TOP_K = 20
_AUTO_THRESHOLD = 0.1
_AUTO_MAX = 4

# djev's default noise seed and the offsets that give every group and draw of
# a request its own noise, so the same request always reads the same noise.
_SEED = 42
_GROUP_SEED_STRIDE = 104729
_DRAW_SEED_STRIDE = 7919

_SYSTEM_PREAMBLE = (
    "Answer a fixed set of questions about the state the user provides. "
    "Each question lists its allowed answers; reply with exactly one label "
    "per question.\n"
)

ReadQuestion = NoulQuestion | ChoiceQuestion | ScoreQuestion


class _AnswerFormat(Enum):
    """How the answer template lays out the questions' labels."""

    LINES = (
        "\n",
        "{id}: ",
        'Reply with one line per question, in this order, formatted as "id: label".',
    )
    # Fewer tokens per question, so long lists need fewer reads.
    INDEXED = (
        " ",
        "{id}",
        "Reply on one line with each question's id immediately followed by "
        "its label, separated by single spaces.",
    )

    def __init__(self, join: str, lead: str, instruction: str):
        self.join = join
        self.lead = lead
        self.instruction = instruction


# Up to this many read questions use LINES; longer lists use INDEXED.
_LINES_MAX_QUESTIONS = 10


@dataclass(frozen=True)
class _Question:
    """A question the model reads, as the prompt presents it."""

    key: str
    id: str
    question: ReadQuestion
    labels: tuple[str, ...]
    options: tuple[tuple[str, str], ...]
    """``(name, description)`` for each label, in label order."""


@dataclass(frozen=True)
class _Read:
    """One read: a prompt, its canvas, and where each question's label goes."""

    questions: list[_Question]
    prompt_ids: list[int]
    canvas: list[int]
    slots: list[tuple[int, list[int]]]
    rows: int
    """Canvas tokens the answer occupies: the template and the turn close."""


def _text(value: Any) -> str:
    """Instructions, descriptions and state may be strings or JSON values."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False)


def _confidence(probs: list[float]) -> float:
    """TypeSafe's confidence: the top probability rescaled so that a uniform
    distribution scores 0 and a certain one scores 1, (K * max(p) - 1) / (K - 1).
    It reproduces every worked example in TypeSafe's documentation."""
    k = len(probs)
    return min(1.0, max(0.0, (k * max(probs) - 1.0) / (k - 1)))


def _answer(question: _Question, probs: list[float]) -> Answer:
    q = question.question
    if isinstance(q, NoulQuestion):
        return NoulAnswer(type="noul", noul=probs[0])
    if isinstance(q, ChoiceQuestion):
        names = [name for name, _ in question.options]
        return ChoiceAnswer(
            type="choice",
            choice=names[max(range(len(probs)), key=probs.__getitem__)],
            confidence=_confidence(probs),
            probabilities=dict(zip(names, probs)),
        )
    return ScoreAnswer(
        type="score",
        score=sum(level * p for level, p in enumerate(probs)),
        confidence=_confidence(probs),
        legend={str(level): value for level, value in enumerate(q.criteria)},
        probabilities={str(level): p for level, p in enumerate(probs)},
    )


def _forced_answer(q: ReadQuestion) -> Answer | None:
    """The answer to a question with one possible answer, which needs no read."""
    if isinstance(q, ChoiceQuestion) and len(q.criteria) == 1:
        (name,) = q.criteria
        return ChoiceAnswer(
            type="choice", choice=name, confidence=1.0, probabilities={name: 1.0}
        )
    if isinstance(q, ScoreQuestion) and len(q.criteria) == 1:
        return ScoreAnswer(
            type="score",
            score=0.0,
            confidence=1.0,
            legend={"0": q.criteria[0]},
            probabilities={"0": 1.0},
        )
    return None


def _slot_distribution(
    logprobs: dict[int, float], label_ids: list[int]
) -> tuple[list[float], float]:
    """One slot of one draw: the softmax over its labels, and the entropy of
    the returned tokens (the top ones and the labels), which decides whether
    a read needs more draws."""
    label_logprobs = [logprobs[i] for i in label_ids]
    top = max(label_logprobs)
    weights = [math.exp(lp - top) for lp in label_logprobs]
    total = sum(weights)
    entropy = -sum(math.exp(lp) * lp for lp in logprobs.values())
    return [w / total for w in weights], entropy


class _ReadBuilder:
    """Turns questions into prompts and canvases for one tokenizer."""

    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer
        self._head = self._encode(_EMPTY_THOUGHT)
        self._turn_close = tokenizer.convert_tokens_to_ids(_TURN_CLOSE)
        self._pad = tokenizer.pad_token_id
        self._choice_labels = self._single_token_labels()

    def _encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def _single_token_labels(self) -> list[str]:
        """Choice labels that stay one token after "q1: ", in a stable order."""
        base = self._encode("q1: A")
        candidates = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
        candidates += [chr(c) for c in range(ord("a"), ord("z") + 1)]
        upper = candidates[:26]
        candidates += [a + b for a in upper for b in upper]
        labels, seen = [], set()
        for label in candidates:
            ids = self._encode("q1: " + label)
            if len(ids) == len(base) and ids[:-1] == base[:-1] and ids[-1] not in seen:
                seen.add(ids[-1])
                labels.append(label)
                if len(labels) == _MAX_CHOICES:
                    break
        return labels

    def questions(
        self, questions: dict[str, ReadQuestion]
    ) -> tuple[list[_Question], dict[str, Answer]]:
        """Split questions into ones to read and ones answered outright."""
        reads: list[_Question] = []
        forced: dict[str, Answer] = {}
        for key, q in questions.items():
            field = f"questions.{key}.criteria"
            if isinstance(q, ChoiceQuestion):
                if not q.criteria:
                    raise InvalidRequestError(
                        f"Choice question must have at least one choice: {key}",
                        field=field,
                    )
                limit = min(_MAX_CHOICES, len(self._choice_labels))
                if len(q.criteria) > limit:
                    raise InvalidRequestError(
                        f"Too many choices. Must have at most {limit} choices.",
                        field=field,
                    )
            elif isinstance(q, ScoreQuestion) and len(q.criteria) > _MAX_SCORE_LEVELS:
                raise InvalidRequestError(
                    "Too many score levels. Must have at most "
                    f"{_MAX_SCORE_LEVELS} levels.",
                    field=field,
                )
            answer = _forced_answer(q)
            if answer is not None:
                forced[key] = answer
                continue
            if isinstance(q, NoulQuestion):
                criteria = q.criteria
                options = (
                    ("yes", _text(criteria.true if criteria else None)),
                    ("no", _text(criteria.false if criteria else None)),
                )
                labels = ("yes", "no")
            elif isinstance(q, ChoiceQuestion):
                options = tuple(
                    (name, _text(desc)) for name, desc in q.criteria.items()
                )
                labels = tuple(self._choice_labels[: len(options)])
            else:
                options = tuple(
                    (str(level), _text(desc)) for level, desc in enumerate(q.criteria)
                )
                labels = _score_labels(len(options))
            reads.append(
                _Question(
                    key=key,
                    id=f"q{len(reads) + 1}",
                    question=q,
                    labels=labels,
                    options=options,
                )
            )
        return reads, forced

    def reads(
        self, questions: list[_Question], state: str, max_prompt_tokens: int | None
    ) -> list[_Read]:
        """Plan the reads that answer ``questions``, in as few reads as fit."""
        fmt = (
            _AnswerFormat.LINES
            if len(questions) <= _LINES_MAX_QUESTIONS
            else _AnswerFormat.INDEXED
        )
        groups = self._groups(questions, fmt)
        plans = []
        for group in groups:
            system = _system_text(group, fmt)
            prompt_ids = self._prompt_ids(system, state)
            if max_prompt_tokens and len(prompt_ids) > max_prompt_tokens:
                raise InvalidRequestError(
                    f"Prompt too long: {len(prompt_ids)} tokens exceeds max "
                    f"context window of {max_prompt_tokens} tokens",
                    field="state",
                )
            canvas, slots = self._canvas(group, fmt)
            rows = canvas.index(self._turn_close) + 1
            plans.append(_Read(group, prompt_ids, canvas, slots, rows))
        return plans

    def _groups(
        self, questions: list[_Question], fmt: _AnswerFormat
    ) -> list[list[_Question]]:
        """Split questions, in order, into the fewest groups whose answer
        templates fit a canvas."""
        groups: list[list[_Question]] = []
        group: list[_Question] = []
        for q in questions:
            trial = group + [q]
            answer = _answer_text(trial, [0] * len(trial), fmt)
            if group and len(self._head) + len(self._encode(answer)) + 1 > _MAX_CANVAS:
                groups.append(group)
                group = [q]
            else:
                group = trial
        groups.append(group)
        return groups

    def _prompt_ids(self, system: str, state: str) -> list[int]:
        """The chat prompt, as token ids, ending where the model turn begins."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": state},
        ]
        ids = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        return [int(t) for t in ids]

    def _canvas(
        self, group: list[_Question], fmt: _AnswerFormat
    ) -> tuple[list[int], list[tuple[int, list[int]]]]:
        """The canvas for one read and each question's slot on it.

        Every label must change exactly one template token, at the same
        position for all of a question's labels; that position is the slot.
        """
        base_labels = [0] * len(group)
        template = self._head + self._encode(_answer_text(group, base_labels, fmt))
        slots = []
        for qi, q in enumerate(group):
            position = None
            label_ids = [0] * len(q.labels)
            for li in range(1, len(q.labels)):
                labels = list(base_labels)
                labels[qi] = li
                ids = self._head + self._encode(_answer_text(group, labels, fmt))
                diffs = [i for i, (a, b) in enumerate(zip(ids, template)) if a != b]
                if (
                    len(ids) != len(template)
                    or len(diffs) != 1
                    or (position is not None and diffs[0] != position)
                ):
                    raise InvalidRequestError(
                        f"Question {q.key!r}: labels do not share one template slot",
                        field=f"questions.{q.key}",
                    )
                position = diffs[0]
                label_ids[li] = ids[position]
            label_ids[0] = template[position]
            if len(set(label_ids)) != len(label_ids):
                raise InvalidRequestError(
                    f"Question {q.key!r}: two labels tokenize to the same id",
                    field=f"questions.{q.key}",
                )
            slots.append((position, label_ids))
        needed = len(template) + 1
        width = min(_MAX_CANVAS, -(-needed // _CANVAS_STEP) * _CANVAS_STEP)
        canvas = template + [self._turn_close] + [self._pad] * (width - needed)
        return canvas, slots


def _system_text(group: list[_Question], fmt: _AnswerFormat) -> str:
    """The question list of one read: only that read's questions, as djev
    prompts each chunk of a long list."""
    text = _SYSTEM_PREAMBLE
    for q in group:
        text += f"\nQuestion {q.id}: {_text(q.question.instructions)}\n"
        for (name, desc), label in zip(q.options, q.labels):
            if isinstance(q.question, ChoiceQuestion):
                text += (
                    f"  {label}: {name} ({desc})\n" if desc else f"  {label}: {name}\n"
                )
            elif isinstance(q.question, ScoreQuestion):
                text += f"  {label}: {desc}\n"
            else:
                text += f"  {label}: {desc}\n" if desc else f"  {label}\n"
    return text + "\n" + fmt.instruction


def _score_labels(levels: int) -> tuple[str, ...]:
    """Levels read as 1-9, or as letters past nine; answers index from 0."""
    if levels <= 9:
        return tuple(str(level + 1) for level in range(levels))
    return tuple(chr(ord("A") + level) for level in range(levels))


def _answer_text(group: list[_Question], labels: list[int], fmt: _AnswerFormat) -> str:
    return fmt.join.join(
        fmt.lead.format(id=q.id) + q.labels[label] for q, label in zip(group, labels)
    )


class SystemOnePlan:
    """A System One request, validated and tokenized, ready to read.

    Building the plan checks every question against the protocol's limits and
    tokenizes every prompt, so a request that cannot be answered is rejected
    before it waits for the model.
    """

    def __init__(
        self,
        tokenizer: Any,
        request: SystemOneRequest,
        *,
        max_prompt_tokens: int | None = None,
        rereads: bool = True,
    ):
        self._rereads = rereads
        builder = _ReadBuilder(tokenizer)
        questions, self._forced = builder.questions(request.questions)
        self._reads = (
            builder.reads(questions, _text(request.state), max_prompt_tokens)
            if questions
            else []
        )
        self._keys = list(request.questions)

    @property
    def usage(self) -> SystemOneUsage:
        """djev's counts: the longest read prompt as input, and the canvas
        tokens the answers occupy as output."""
        return SystemOneUsage(
            input_tokens=max((len(read.prompt_ids) for read in self._reads), default=0),
            output_tokens=sum(read.rows for read in self._reads),
        )

    @property
    def prefill_tokens(self) -> int:
        """Prompt tokens the model prefills: every read's prompt, once."""
        return sum(len(read.prompt_ids) for read in self._reads)

    async def answer(self, engine: Any) -> dict[str, Answer]:
        """Run the reads on a diffusion engine (see
        ``VLMBatchedEngine.diffusion_read_session``). Answers are keyed and
        ordered like the request's questions."""
        answers = dict(self._forced)
        for k, read in enumerate(self._reads):
            seed = _SEED + k * _GROUP_SEED_STRIDE
            async with engine.diffusion_read_session(read.prompt_ids) as session:
                draws = await self._draws(session, read, [seed])
                if (
                    self._rereads
                    and max(entropy for _, entropy in draws[0]) > _AUTO_THRESHOLD
                ):
                    more = [
                        seed + 1 + j * _DRAW_SEED_STRIDE for j in range(_AUTO_MAX - 1)
                    ]
                    draws += await self._draws(session, read, more)
            logger.debug(
                "System One read %d: %d questions, %d draws",
                k,
                len(read.questions),
                len(draws),
            )
            for qi, q in enumerate(read.questions):
                probs = [draw[qi][0] for draw in draws]
                mean = [
                    sum(p[li] for p in probs) / len(probs)
                    for li in range(len(q.labels))
                ]
                answers[q.key] = _answer(q, mean)
        return {key: answers[key] for key in self._keys}

    @staticmethod
    async def _draws(
        session: Any, read: _Read, seeds: list[int]
    ) -> list[list[tuple[list[float], float]]]:
        """Label probabilities and entropy per draw, per question."""
        results = await session.read(read.canvas, read.slots, seeds, _TOP_K)
        return [
            [
                _slot_distribution(logprobs, label_ids)
                for logprobs, (_, label_ids) in zip(result, read.slots)
            ]
            for result in results
        ]
