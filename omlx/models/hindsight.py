# SPDX-License-Identifier: Apache-2.0
"""
Hindsight decision models served through ``/v1/systemone``.

``hindsight-ai/hs-decide-1-flash`` is a 0.8B Qwen3.5 finetune that answers typed
questions in a single forward pass. The checkpoint ships its own input format in
``decision.json`` and its own serving code, and this module reproduces both:

* the request is rendered by hand into one chat-framed prompt — a user turn
  holding ``State:`` and one ``[Qn] (kind)`` block per question, then an
  assistant turn holding one ``Qn:`` answer line per question;
* each question is answered at the colon that ends its line, over the
  checkpoint's single-token answer codes: ``no``/``yes`` for a noul, the digits
  for a score, ``A``..``Z`` then the two-letter codes for a choice.

The readout is the LM head at the slot positions, maxed over each code's
spellings and softmaxed over the options. Nothing is generated, so the model is
served by :class:`~omlx.engine.decision.DecisionEngine` like the other decision
families; only the prompt layout and the readout are family-specific.
"""

from __future__ import annotations

import json
import string
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from .decision import (
    ChunkSize,
    DecisionBackbone,
    DecisionContextLengthError,
    DecisionRequestError,
    round4,
)

SETTINGS_FILE = "decision.json"
PROMPT_VERSION = 2
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
NOUL_CODES: tuple[str, str] = ("no", "yes")
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS, MAX_SCORE_LEVELS = 2, 10
MAX_SCORE_DIGITS = 10


class HindsightFormatError(ValueError):
    """``decision.json`` is not a format this family serves."""


# --------------------------------------------------------------------------
# Request text


def content_text(value: Any) -> str:
    """Text as is, structured content as compact JSON with non-ASCII kept."""
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def state_text(state: Any) -> str:
    """A text state as is, a structured one as JSON with non-ASCII kept."""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def spellings(code: str) -> tuple[str, ...]:
    """The forms of an answer code the checkpoint reads, bare and after a space."""
    if code in NOUL_CODES:
        return (code, code.capitalize(), " " + code, " " + code.capitalize())
    return (code, " " + code)


# --------------------------------------------------------------------------
# Tokenizer lookup


class TokenTable:
    """Per-tokenizer caches: single-token lookups and the two-letter code list."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self._single: dict[str, int | None] = {}
        self._pairs: tuple[str, ...] | None = None

    def single(self, text: str) -> int | None:
        """The one token ``text`` encodes to, or None when it takes several."""
        if text not in self._single:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            self._single[text] = int(ids[0]) if len(ids) == 1 else None
        return self._single[text]

    def pair_codes(self) -> tuple[str, ...]:
        """A to Z, then every two-letter uppercase code that is one token bare and spaced."""
        if self._pairs is None:
            codes = list(string.ascii_uppercase)
            for first in string.ascii_uppercase:
                for second in string.ascii_uppercase:
                    code = first + second
                    if (
                        self.single(code) is not None
                        and self.single(" " + code) is not None
                    ):
                        codes.append(code)
            if len(codes) < MAX_CHOICE_OPTIONS:
                raise ValueError(
                    f"only {len(codes)} single-token codes in this tokenizer"
                )
            self._pairs = tuple(codes[:MAX_CHOICE_OPTIONS])
        return self._pairs


def plain_ids(tokenizer: Any, text: str) -> list[int]:
    """``text`` encoded with none of the tokenizer's added tokens.

    A literal ``<|im_end|>`` in a memory is then spelled out as characters
    rather than turning into a control token.
    """
    added_vocab = getattr(tokenizer, "get_added_vocab", None)
    vocab: dict[str, int] = added_vocab() if added_vocab is not None else {}
    if not vocab or not any(token in text for token in vocab):
        return list(tokenizer.encode(text, add_special_tokens=False))
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise DecisionRequestError(
            "the tokenizer has added tokens but no backend tokenizer, so its text "
            "cannot be encoded without them"
        )
    normalized = backend.normalizer.normalize_str(text) if backend.normalizer else text
    pieces = (
        backend.pre_tokenizer.pre_tokenize_str(normalized)
        if backend.pre_tokenizer
        else [(normalized, (0, len(normalized)))]
    )
    return [token.id for piece, _ in pieces for token in backend.model.tokenize(piece)]


# --------------------------------------------------------------------------
# Questions


@dataclass(frozen=True)
class TypedQuestion:
    id: str
    kind: str
    instructions: str
    keys: list[str]
    descriptions: list[str]
    codes: list[str]


@dataclass(frozen=True)
class RenderedQuestion:
    id: str
    kind: str
    # Index into the prompt of the colon that ends this question's answer line.
    slot: int
    codes: list[str]
    # Per option, the token ids of its code's single-token spellings.
    groups: list[list[int]]


@dataclass(frozen=True)
class RenderedRequest:
    ids: list[int]
    questions: list[RenderedQuestion]

    def tokens(self) -> int:
        return len(self.ids)


def option_keys(question: dict) -> list[str]:
    """The answer's probability keys in canonical order."""
    kind = question.get("type")
    criteria = question.get("criteria")
    if kind == "noul":
        return ["false", "true"]
    if kind == "score":
        return [str(i) for i in range(len(criteria or []))]
    if not isinstance(criteria, dict):
        return []
    return [str(key) for key in criteria]


def codes_for(kind: str, count: int, table: TokenTable) -> list[str]:
    """The answer codes in ``option_keys`` order."""
    if kind == "noul":
        return list(NOUL_CODES)
    if kind == "score":
        return [str(i) for i in range(count)]
    return list(table.pair_codes()[:count])


def typed_question(name: str, question: dict, table: TokenTable) -> TypedQuestion:
    """One question with its options in canonical order; option counts checked here."""
    kind = question.get("type")
    if kind not in ("noul", "choice", "score"):
        raise DecisionRequestError(
            f"{name}: unknown question type {kind!r}; use noul, choice or score"
        )
    criteria = question.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, dict):
            raise DecisionRequestError(
                f"{name}: choice criteria must be a map of option to description or null"
            )
        if len(criteria) < 2:
            raise DecisionRequestError(f"{name}: a choice needs at least two options")
        if len(criteria) > MAX_CHOICE_OPTIONS:
            raise DecisionRequestError(
                f"{name}: {len(criteria)} options; at most {MAX_CHOICE_OPTIONS} "
                "options per choice"
            )
        keys = [str(key) for key in criteria]
        descriptions = [content_text(value) for value in criteria.values()]
    elif kind == "score":
        if not isinstance(criteria, list) or not (
            MIN_SCORE_LEVELS <= len(criteria) <= MAX_SCORE_LEVELS
        ):
            raise DecisionRequestError(
                f"{name}: a score takes {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} "
                "ordered levels"
            )
        keys = [str(i) for i in range(len(criteria))]
        descriptions = [content_text(level) for level in criteria]
    else:
        criteria = criteria or {}
        if not isinstance(criteria, dict):
            raise DecisionRequestError(f"{name}: noul criteria must be an object")
        keys = ["false", "true"]
        descriptions = [
            content_text(criteria.get("false")),
            content_text(criteria.get("true")),
        ]
    return TypedQuestion(
        id=name,
        kind=kind,
        instructions=content_text(question.get("instructions")),
        keys=keys,
        descriptions=descriptions,
        codes=codes_for(kind, len(keys), table),
    )


def alias_groups(question: TypedQuestion, table: TokenTable) -> list[list[int]]:
    """Per option, the token ids of its code's single-token spellings."""
    groups: list[list[int]] = []
    for code in question.codes:
        ids: list[int] = []
        for form in spellings(code):
            token = table.single(form)
            if token is not None and token not in ids:
                ids.append(token)
        if not ids:
            raise DecisionRequestError(
                f"answer code {code!r} is not a single token in this tokenizer"
            )
        groups.append(ids)
    return groups


def render_request(tokenizer: Any, request: dict, max_tokens: int) -> RenderedRequest:
    """Token ids for one request, with each question's slot and option groups.

    A request over the checkpoint's limit is refused, never cut.
    """
    table = TokenTable(tokenizer)
    questions = [
        typed_question(str(name), question, table)
        for name, question in (request.get("questions") or {}).items()
    ]
    if not questions:
        raise DecisionRequestError("the request has no questions")

    def enc(text: str) -> list[int]:
        return list(tokenizer.encode(text, add_special_tokens=False))

    def user(text: str) -> list[int]:
        return plain_ids(tokenizer, text)

    ids = enc(f"{IM_START}user\n") + enc("State:\n")
    ids += user(state_text(request.get("state")))
    ids += enc("\n\n") + enc("Questions:\n")
    for index, question in enumerate(questions, start=1):
        ids += user(f"[Q{index}] ({question.kind}) ")
        ids += user(question.instructions) + enc("\n")
        for code, key, description in zip(
            question.codes, question.keys, question.descriptions, strict=True
        ):
            ids += enc(f"  {code}")
            if question.kind == "noul":
                if description:
                    ids += enc(": ") + user(description)
            elif question.kind == "score":
                ids += enc(": ") + user(description)
            else:
                ids += enc(": ") + user(key)
                if description and description != key:
                    ids += enc(": ") + user(description)
            ids += enc("\n")
        ids += enc("\n")
    ids += enc(f"{IM_END}\n{IM_START}assistant\n")

    rendered: list[RenderedQuestion] = []
    for index, question in enumerate(questions, start=1):
        line = user(f"Q{index}") + enc(":")
        rendered.append(
            RenderedQuestion(
                id=question.id,
                kind=question.kind,
                slot=len(ids) + len(line) - 1,
                codes=question.codes,
                groups=alias_groups(question, table),
            )
        )
        ids += line + enc("\n")
    if len(ids) > max_tokens:
        raise DecisionContextLengthError(
            f"{len(ids)} tokens is over this model's context window of "
            f"{max_tokens} tokens"
        )
    return RenderedRequest(ids=ids, questions=rendered)


# --------------------------------------------------------------------------
# Answers


def softmax(logits: Sequence[float]) -> list[float]:
    top = max(logits)
    weights = [float(np.exp(value - top)) for value in logits]
    total = sum(weights)
    return [value / total for value in weights]


def choice_confidence(probabilities: Sequence[float]) -> float:
    """How far the top option stands above a uniform guess."""
    n = len(probabilities)
    return 1.0 if n == 1 else min(1.0, max(0.0, (n * max(probabilities) - 1) / (n - 1)))


def score_confidence(probabilities: Sequence[float]) -> float:
    """One minus the spread around the top level, relative to a uniform distribution."""
    n = len(probabilities)
    if n == 1:
        return 1.0
    top = max(range(n), key=lambda i: probabilities[i])
    spread = sum(p * abs(i - top) for i, p in enumerate(probabilities))
    uniform = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1 - spread / uniform)


def format_answer(question: dict, probabilities: Sequence[float]) -> dict:
    """A System One answer from probabilities in ``option_keys`` order."""
    keys = option_keys(question)
    if len(probabilities) != len(keys):
        raise ValueError(f"{len(probabilities)} probabilities for {len(keys)} options")
    kind = question.get("type")
    if kind == "noul":
        # Codes are (no, yes); the answer reports the probability of yes.
        return {"type": "noul", "noul": round4(probabilities[1])}
    if kind == "score":
        criteria = question.get("criteria") or []
        return {
            "type": "score",
            "score": round4(sum(i * p for i, p in enumerate(probabilities))),
            "confidence": round4(score_confidence(probabilities)),
            "legend": {str(i): level for i, level in enumerate(criteria)},
            "probabilities": {str(i): round4(p) for i, p in enumerate(probabilities)},
        }
    best = max(range(len(keys)), key=lambda i: probabilities[i])
    return {
        "type": "choice",
        "choice": keys[best],
        "confidence": round4(choice_confidence(probabilities)),
        "probabilities": dict(
            zip(keys, (round4(p) for p in probabilities), strict=True)
        ),
    }


def load_settings(directory: Path) -> int:
    """The checkpoint's request limit, refusing a format this family does not serve."""
    path = directory / SETTINGS_FILE
    try:
        settings = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise HindsightFormatError(f"{path} is not readable: {error}") from error
    typed = settings.get("typed") or {}
    if settings.get("prompt_version") != PROMPT_VERSION:
        raise HindsightFormatError(
            f"{path} declares prompt_version {settings.get('prompt_version')!r}; "
            f"this family serves {PROMPT_VERSION}"
        )
    if typed.get("plain_text") is not True or typed.get("codes") != "pairs":
        raise HindsightFormatError(
            f"{path} is not a format this family serves: {typed!r}"
        )
    return int(typed.get("max_request_tokens") or 65536)


# --------------------------------------------------------------------------
# Model


@dataclass
class HindsightPlan:
    request: dict
    rendered: RenderedRequest


class HindsightModel:
    """hs-decide-1-flash and its MLX conversions."""

    def __init__(self, model_path: str, trust_remote_code: bool = False):
        self.model_path = model_path
        self.backbone = DecisionBackbone(model_path, trust_remote_code)
        self._table: TokenTable | None = None
        self._classifier_ids: list[int] = []
        self._column: dict[int, int] = {}
        self._max_request_tokens = 65536

    @property
    def has_vision(self) -> bool:
        return False

    def load(self) -> None:
        self._max_request_tokens = load_settings(Path(self.model_path))
        self.backbone.load()
        tokenizer = self.backbone.tokenizer
        self._table = TokenTable(tokenizer)
        codes = list(NOUL_CODES) + [str(d) for d in range(MAX_SCORE_DIGITS)]
        codes += list(self._table.pair_codes())
        ids: list[int] = []
        for code in codes:
            for form in spellings(code):
                token = self._table.single(form)
                if token is not None and token not in ids:
                    ids.append(token)
        if not ids:
            raise HindsightFormatError("the tokenizer has no single-token answer codes")
        self._classifier_ids = ids
        self._column = {token: column for column, token in enumerate(ids)}
        # Every question's slot is a colon, so it must be one token.
        if self._table.single(":") is None:
            raise HindsightFormatError("':' is not a single token in this tokenizer")

    def close(self) -> None:
        self.backbone.close()

    def encode(self, request: dict, truncate: bool = True) -> HindsightPlan:
        """Validate and render the request; ``truncate`` is accepted for API symmetry."""
        if request.get("images") or request.get("videos"):
            raise DecisionRequestError("this model does not take images or videos")
        rendered = render_request(
            self.backbone.tokenizer, request, self._max_request_tokens
        )
        return HindsightPlan(request=request, rendered=rendered)

    def run(
        self, plan: HindsightPlan, chunk_size: ChunkSize
    ) -> Generator[int, None, dict]:
        """Run one prefill, read every question at its slot, then build the answers."""
        backbone = self.backbone
        ids = np.asarray(plan.rendered.ids, dtype=np.int32)
        embeds, positions = backbone.embed(ids)
        cache = backbone.make_cache()
        hidden = yield from backbone.prefill(
            ids, embeds, positions, cache, chunk_size, keep_all=True
        )

        slots = np.asarray([q.slot for q in plan.rendered.questions], dtype=np.int32)
        rows = backbone.output_rows(mx.array(self._classifier_ids)).astype(mx.float32)
        slot_hidden = hidden[mx.array(slots)].astype(mx.float32)
        logits = np.asarray((slot_hidden @ rows.T).tolist(), dtype=np.float64)

        questions = plan.request.get("questions") or {}
        answers = {}
        for question, row in zip(plan.rendered.questions, logits, strict=True):
            scores = [
                float(np.max(row[[self._column[token] for token in group]]))
                for group in question.groups
            ]
            answers[question.id] = format_answer(
                questions.get(question.id) or {}, softmax(scores)
            )
        return {"answers": answers, "input_tokens": int(len(ids))}
