# SPDX-License-Identifier: Apache-2.0
"""LiquidAI d1 decision model for ``/v1/systemone``.

d1 is an LFM2 or LFM2-VL checkpoint that reads the answer off the last-token
logits. It does not have Clef's joint head. The prompt and readout match the
checkpoint's ``prompt.py``: one forward per question, no generated tokens.
"""

from __future__ import annotations

import math
from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass
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

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
YES_FORMS = ("yes", "Yes", "YES")
NO_FORMS = ("no", "No", "NO")
_FALLBACK_POOL = (
    [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + [f"{i:02d}" for i in range(100)]
    + [chr(c) for c in range(ord("a"), ord("z") + 1)]
)


@dataclass
class D1Question:
    name: str
    kind: str
    instructions: str
    criteria: Any
    prompt: str
    groups: list[list[int]]
    labels: list[str]


@dataclass
class D1Plan:
    questions: list[D1Question]


def _single_ids(tokenize, texts: Sequence[str]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for text in texts:
        ids = tokenize(text)
        if len(ids) == 1 and ids[0] not in seen:
            out.append(ids[0])
            seen.add(ids[0])
    return out


def _option_codes(labels: Sequence[str]) -> list[str]:
    labs = [str(label).strip() for label in labels]
    if labs and all(len(label) == 1 and label.isalpha() for label in labs):
        return labs
    if len(labs) <= 26:
        return [chr(ord("A") + i) for i in range(len(labs))]
    return [f"{i:02d}" for i in range(len(labs))]


def _aliases(tokenize, labels: Sequence[str]) -> list[tuple[str, int]]:
    used: set[int] = set()
    out: list[tuple[str, int]] = []

    def take(raw: str) -> bool:
        ids = tokenize(raw)
        if len(ids) != 1 or ids[0] in used:
            return False
        out.append((raw, ids[0]))
        used.add(ids[0])
        return True

    for code in _option_codes(labels):
        if take(code):
            continue
        if not any(take(raw) for raw in _FALLBACK_POOL):
            raise DecisionRequestError(
                f"no single-token alias left for {len(labels)} options"
            )
    return out


def _state_block(state: Any) -> str:
    # json_only, the checkpoint default: a string is itself, anything else is JSON.
    if isinstance(state, str):
        return f"{state}\n\n"
    import json

    return json.dumps(state, ensure_ascii=False, indent=2) + "\n\n"


def render_prompt(tokenize, state: Any, question: Mapping[str, Any], bos: str) -> tuple[str, list[list[int]], list[str]]:
    """Return the prompt, readout token groups, and labels in option order."""
    kind = question.get("type", "choice")
    instructions = question.get("instructions") or ""
    criteria = question.get("criteria")
    if kind == "noul":
        body = f"{instructions}\n\nReply with yes or no only."
        yes, no = _single_ids(tokenize, YES_FORMS), _single_ids(tokenize, NO_FORMS)
        if not yes or not no:
            raise DecisionRequestError("tokenizer has no single-token yes/no")
        groups, labels = [yes, no], ["true", "false"]
    elif kind == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            raise DecisionRequestError("score criteria must be 2 to 10 levels")
        legend = "\n".join(f"{i} {name}" for i, name in enumerate(criteria))
        body = (
            f"{instructions}\n\n{legend}\n\n"
            f"Reply with a single digit 0-{len(criteria) - 1} only."
        )
        groups = [_single_ids(tokenize, [str(i)]) for i in range(len(criteria))]
        if any(not group for group in groups):
            raise DecisionRequestError("score levels need single-token digits")
        labels = [str(i) for i in range(len(criteria))]
    elif kind == "choice":
        if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 26:
            raise DecisionRequestError("choice criteria must have 2 to 26 options")
        labels = list(criteria)
        codes = _aliases(tokenize, labels)
        lines = "\n".join(
            f"{codes[i][0]} {criteria[label] or str(label).replace('_', ' ')}"
            for i, label in enumerate(labels)
        )
        body = f"{instructions}\n\nOptions:\n{lines}\n\nReply with the option code only."
        groups = []
        for code, token_id in codes:
            extra = [i for i in _single_ids(tokenize, [f" {code}"]) if i != token_id]
            groups.append([token_id, *extra])
    else:
        raise DecisionRequestError(f"unknown question type {kind!r}")

    state_body = "" if state is None else f"{_state_block(state)}\nQUESTION:\n"
    prompt = f"{bos}{IM_START}user\n{state_body}{body}{IM_END}\n{IM_START}assistant\n"
    return prompt, groups, labels


def _softmax(scores: list[float]) -> list[float]:
    peak = max(scores)
    exps = [math.exp(score - peak) for score in scores]
    total = sum(exps)
    return [item / total for item in exps]


class D1Model:
    """d1-3B: an LFM backbone whose answer is a softmax over option tokens."""

    def __init__(self, model_path: str, trust_remote_code: bool = False):
        self.model_path = model_path
        self.backbone = DecisionBackbone(model_path, trust_remote_code)
        self._bos = ""

    def load(self) -> None:
        self.backbone.load()
        bos = getattr(self.backbone.tokenizer, "bos_token", None)
        self._bos = bos if isinstance(bos, str) else ""

    def close(self) -> None:
        self.backbone.close()

    def encode(self, request: dict, truncate: bool = True) -> D1Plan:
        if request.get("images"):
            raise DecisionRequestError(
                "d1 image decisions are not supported yet; send text state only"
            )
        questions = request.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise DecisionRequestError("questions must be a non-empty object")
        limit = self.backbone.max_position_embeddings
        encoded: list[D1Question] = []
        for name, question in questions.items():
            if not isinstance(question, dict):
                raise DecisionRequestError(f"question {name!r} must be an object")
            prompt, groups, labels = render_prompt(
                self.backbone.tokenize, request.get("state"), question, self._bos
            )
            ids = self.backbone.tokenize(prompt)
            if limit and len(ids) > limit:
                if not truncate:
                    raise DecisionContextLengthError(
                        f"question {name!r} is {len(ids)} tokens, limit is {limit}"
                    )
                raise DecisionContextLengthError(
                    f"question {name!r} is {len(ids)} tokens and does not fit "
                    f"the {limit}-token context"
                )
            encoded.append(
                D1Question(
                    name=str(name),
                    kind=str(question.get("type", "choice")),
                    instructions=str(question.get("instructions") or ""),
                    criteria=question.get("criteria"),
                    prompt=prompt,
                    groups=groups,
                    labels=labels,
                )
            )
        return D1Plan(questions=encoded)

    def run(self, plan: D1Plan, chunk_size: ChunkSize) -> Generator[int, None, dict]:
        del chunk_size  # each d1 question is one short forward, not a long prefill
        answers = {}
        total = 0
        for question in plan.questions:
            ids = self.backbone.tokenize(question.prompt)
            logits = self._last_logits(ids)
            logz = (logits - mx.logsumexp(logits)).tolist()
            scores = [max(logz[token_id] for token_id in group) for group in question.groups]
            probabilities = _softmax(scores)
            mx.eval(logits)
            total += len(ids)
            yield len(ids)
            answers[question.name] = self._answer(question, probabilities)
        return {"answers": answers, "input_tokens": total}

    def _last_logits(self, ids: list[int]) -> mx.array:
        tokens = mx.array(ids)[None]
        embeds = self.backbone.model.get_input_embeddings(tokens, None).inputs_embeds
        output = self.backbone.model.language_model(tokens, inputs_embeds=embeds)
        row = output.logits if hasattr(output, "logits") else output
        mx.eval(row)
        return row[0, -1]

    def _answer(self, question: D1Question, probabilities: list[float]) -> dict:
        if question.kind == "noul":
            yes = probabilities[0]
            return {"type": "noul", "noul": round4(yes)}
        if question.kind == "score":
            expected = sum(i * p for i, p in enumerate(probabilities))
            return {
                "type": "score",
                "score": round4(expected),
                "probabilities": {
                    label: round4(p) for label, p in zip(question.labels, probabilities)
                },
            }
        winner = max(range(len(probabilities)), key=lambda i: probabilities[i])
        return {
            "type": "choice",
            "choice": question.labels[winner],
            "probabilities": {
                label: round4(p) for label, p in zip(question.labels, probabilities)
            },
            "confidence": round4(max(probabilities)),
        }
