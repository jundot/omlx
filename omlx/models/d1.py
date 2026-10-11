# SPDX-License-Identifier: Apache-2.0
"""
LiquidAI d1 decision model.

d1 is an LFM2.5-VL backbone post-trained for System One decisions. Each
question is rendered after the state as the release's ``prompt.py`` writes it
(the shared prefix ends with ``QUESTION:``; each question is a suffix up to the
assistant header), and the answer is a softmax over each option's best token
logit at the answer slot. d1 ships no calibration, so the probabilities are
returned as read.

Several questions over one state share the prefix: it is prefilled once and
each question continues from a copy of that cache, which is the release
runner's trunk-and-branches pass, one branch at a time. A single question is
one plain pass over the whole prompt, as the runner does.
"""

import copy
import json
import math
from collections.abc import Generator
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
SCORE_LEVELS = (2, 10)

# Option codes tried when a code is not one free token (release prompt.py).
_FALLBACK_POOL = (
    [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + [f"{i:02d}" for i in range(100)]
    + [chr(c) for c in range(ord("a"), ord("z") + 1)]
    + [f"#{i}" for i in range(200)]
    + [
        chr(a) + chr(b)
        for a in range(ord("A"), ord("Z") + 1)
        for b in range(ord("A"), ord("Z") + 1)
    ]
)


# Instructions, descriptions and levels are written with f-strings, as the
# release does, so non-string JSON values render as Python's str() of them.


class Lfm2DecisionBackbone(DecisionBackbone):
    """LFM2 / LFM2-VL text backbone for decision prefill.

    The text model takes ``input_embeddings`` and derives positions from the
    cache offset, so continuation needs no position ids. Images are not wired
    yet, so the backbone reports no vision even when the checkpoint has a
    vision tower.
    """

    def load(self) -> None:
        super().load()
        self.image_processor = None

    def embed(
        self,
        ids: np.ndarray,
        pixel_values: np.ndarray | None = None,
        image_grid_thw: np.ndarray | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        if pixel_values is not None:
            raise DecisionRequestError("d1 images are not supported yet")
        return self._text_model.embed_tokens(mx.array(ids)[None]), None

    def _forward(
        self,
        tokens: mx.array,
        embeds: mx.array,
        positions: mx.array | None,
        cache: list,
    ) -> mx.array:
        return self._text_model(tokens, cache=cache, input_embeddings=embeds)


def option_codes(labels: list[str]) -> list[str]:
    """Native letters when the labels already are letters, else A..Z, else 00.."""
    labs = [str(x).strip() for x in labels]
    if labs and all(len(k) == 1 and k.isalpha() for k in labs):
        return labs
    if len(labs) <= 26:
        return [chr(ord("A") + i) for i in range(len(labs))]
    return [f"{i:02d}" for i in range(len(labs))]


def state_block(state: Any) -> str:
    """The release's default ``json_only`` state rendering."""
    if isinstance(state, str):
        return f"{state}\n\n"
    return json.dumps(state, ensure_ascii=False, indent=2) + "\n\n"


def prefix_text(bos: str, state: Any) -> str:
    """Everything before the question, shared by all questions on one state."""
    return f"{bos}{IM_START}user\n{state_block(state)}\nQUESTION:\n"


@dataclass
class _Question:
    question_id: str
    kind: str
    names: list[str]
    legend: list[Any]
    suffix: str
    # One group of token ids per option; an option scores its best token.
    groups: list[list[int]]


@dataclass
class D1Plan:
    questions: dict
    items: list[_Question]
    prefix: str


class D1Model:
    """LiquidAI d1 and its MLX conversions."""

    def __init__(self, model_path: str, trust_remote_code: bool = False):
        self.model_path = model_path
        self.backbone = Lfm2DecisionBackbone(model_path, trust_remote_code)
        self._bos = ""
        self._aliases: dict[tuple[str, ...], list[tuple[str, int]]] = {}

    def load(self) -> None:
        self.backbone.load()
        bos = getattr(self.backbone.tokenizer, "bos_token", None)
        self._bos = bos if isinstance(bos, str) else ""
        self._aliases = {}

    def close(self) -> None:
        self.backbone.close()

    # ------------------------------------------------------------- tokens

    def _single_ids(self, texts: list[str]) -> list[int]:
        """The single-token encodings among ``texts``, without repeats."""
        out: list[int] = []
        for text in texts:
            ids = self.backbone.tokenize(text)
            if len(ids) == 1 and ids[0] not in out:
                out.append(ids[0])
        return out

    def _option_aliases(self, labels: list[str]) -> list[tuple[str, int]]:
        """A distinct single-token code per label: ``[(code, token_id)]``."""
        codes = tuple(option_codes(labels))
        cached = self._aliases.get(codes)
        if cached is not None:
            return cached
        used: set[int] = set()
        out: list[tuple[str, int]] = []

        def take(raw: str) -> bool:
            ids = self.backbone.tokenize(raw)
            if len(ids) != 1 or ids[0] in used:
                return False
            out.append((raw, ids[0]))
            used.add(ids[0])
            return True

        for code in codes:
            if take(code):
                continue
            if not any(take(raw) for raw in _FALLBACK_POOL):
                raise DecisionRequestError(
                    f"no single-token code left for {len(codes)} options"
                )
        self._aliases[codes] = out
        return out

    # ------------------------------------------------------------- encode

    def _question(self, question_id: str, question: dict) -> _Question:
        kind = question["type"]
        instructions = question.get("instructions")
        if instructions is None or instructions == "":
            raise DecisionRequestError(f"question {question_id!r} needs instructions")
        criteria = question.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, dict) or not criteria:
                raise DecisionRequestError(
                    "choice criteria must be a non-empty map of option to "
                    "description or null"
                )
            names = [str(k) for k in criteria]
            codes = self._option_aliases(names)
            lines = []
            for (code, _), name, desc in zip(codes, names, criteria.values()):
                text = desc or name.replace("_", " ")
                lines.append(f"{code} {text}")
            body = (
                f"{instructions}\n\nOptions:\n" + "\n".join(lines) + "\n\n"
                "Reply with the option code only."
            )
            groups = []
            for code, token in codes:
                extra = self._single_ids([f" {code}"])
                groups.append([token] + [i for i in extra if i != token])
            legend: list[Any] = []
        elif kind == "score":
            low, high = SCORE_LEVELS
            if not isinstance(criteria, list) or not low <= len(criteria) <= high:
                raise DecisionRequestError(
                    f"score criteria must be an ordered array of {low} to {high} levels"
                )
            names = [str(i) for i in range(len(criteria))]
            legend_text = "\n".join(f"{i} {level}" for i, level in enumerate(criteria))
            body = (
                f"{instructions}\n\n{legend_text}\n\n"
                f"Reply with a single digit 0-{len(criteria) - 1} only."
            )
            groups = [self._single_ids([str(i)]) for i in range(len(criteria))]
            if any(not g for g in groups):
                raise DecisionRequestError("score levels need single-token digits")
            legend = list(criteria)
        else:
            if criteria is not None and not isinstance(criteria, dict):
                raise DecisionRequestError("noul criteria must be an object")
            extra = ""
            if criteria:
                extra = f"\nYes: {criteria.get('true')}\nNo: {criteria.get('false')}"
            body = f"{instructions}{extra}\n\nReply with yes or no only."
            names = ["yes", "no"]
            groups = [
                self._single_ids(list(YES_FORMS)),
                self._single_ids(list(NO_FORMS)),
            ]
            if not all(groups):
                raise DecisionRequestError("the tokenizer has no single-token yes/no")
            legend = []
        suffix = f"{body}{IM_END}\n{IM_START}assistant\n"
        return _Question(str(question_id), kind, names, legend, suffix, groups)

    def encode(self, request: dict, truncate: bool = True) -> D1Plan:
        """Validate and render the questions.

        ``truncate`` is ignored: the release never cuts the state, so a prompt
        over the context limit is an error.
        """
        if request.get("images"):
            raise DecisionRequestError("d1 images are not supported yet")
        items = [
            self._question(str(qid), question)
            for qid, question in request["questions"].items()
        ]
        plan = D1Plan(
            questions=request["questions"],
            items=items,
            prefix=prefix_text(self._bos, request["state"]),
        )
        limit = self.backbone.max_position_embeddings
        if limit is not None:
            prefix_len = len(self.backbone.tokenize(plan.prefix))
            longest = max(len(self.backbone.tokenize(q.suffix)) for q in items)
            if prefix_len + longest > limit:
                raise DecisionContextLengthError(
                    f"the prompt needs {prefix_len + longest} tokens; the limit is "
                    f"{limit} (d1 does not truncate the state)"
                )
        return plan

    # ---------------------------------------------------------------- run

    def run(self, plan: D1Plan, chunk_size: ChunkSize) -> Generator[int, None, dict]:
        """Answer every question, yielding per prefill chunk."""
        backbone = self.backbone
        answers = {}
        if len(plan.items) == 1:
            item = plan.items[0]
            ids = np.asarray(
                backbone.tokenize(plan.prefix + item.suffix), dtype=np.int32
            )
            hidden = yield from self._prefill(ids, backbone.make_cache(), chunk_size)
            answers[item.question_id] = self._answer(item, self._probs(item, hidden))
            return {"answers": answers, "input_tokens": int(len(ids))}

        trunk = np.asarray(backbone.tokenize(plan.prefix), dtype=np.int32)
        trunk_cache = backbone.make_cache()
        yield from self._prefill(trunk, trunk_cache, chunk_size)
        input_tokens = len(trunk)
        for item in plan.items:
            branch = np.asarray(backbone.tokenize(item.suffix), dtype=np.int32)
            # Caches update in place; each question continues from its own copy.
            hidden = yield from self._prefill(
                branch, copy.deepcopy(trunk_cache), chunk_size
            )
            answers[item.question_id] = self._answer(item, self._probs(item, hidden))
            input_tokens += len(branch)
        return {"answers": answers, "input_tokens": int(input_tokens)}

    def _prefill(
        self, ids: np.ndarray, cache: list, chunk_size: ChunkSize
    ) -> Generator[int, None, mx.array]:
        embeds, positions = self.backbone.embed(ids)
        return (
            yield from self.backbone.prefill(
                ids, embeds, positions, cache, chunk_size, keep_all=False
            )
        )

    def _probs(self, item: _Question, hidden: mx.array) -> list[float]:
        """Softmax over each option's best token logit at the answer slot.

        Only the option tokens' output rows are computed; the release reads
        log-probabilities, whose shared normalizer cancels in this softmax.
        """
        flat = [i for group in item.groups for i in group]
        rows = self.backbone.output_rows(mx.array(flat)).astype(mx.float32)
        logits = (rows @ hidden.astype(mx.float32).reshape(-1)).tolist()
        scores, start = [], 0
        for group in item.groups:
            scores.append(max(logits[start : start + len(group)]))
            start += len(group)
        if not all(math.isfinite(s) for s in scores):
            raise RuntimeError("option scores are not finite")
        top = max(scores)
        weights = [math.exp(s - top) for s in scores]
        total = sum(weights)
        return [w / total for w in weights]

    def _answer(self, item: _Question, probs: list[float]) -> dict:
        """The release's answer fields: P(yes); a pick; an expected level."""
        if item.kind == "noul":
            return {"type": "noul", "noul": round4(probs[0])}
        best = int(np.argmax(probs))
        if item.kind == "choice":
            return {
                "type": "choice",
                "choice": item.names[best],
                "confidence": round4(probs[best]),
                "probabilities": {
                    name: round4(p) for name, p in zip(item.names, probs, strict=True)
                },
            }
        return {
            "type": "score",
            "score": round4(sum(i * p for i, p in enumerate(probs))),
            "confidence": round4(probs[best]),
            "probabilities": {str(i): round4(p) for i, p in enumerate(probs)},
            "legend": {str(i): level for i, level in enumerate(item.legend)},
        }
