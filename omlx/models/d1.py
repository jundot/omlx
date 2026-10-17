# SPDX-License-Identifier: Apache-2.0
"""
LiquidAI d1 decision model.

d1 is an LFM2.5-VL backbone post-trained for System One decisions. Each
question is rendered after the state as the release's ``prompt.py`` writes it
(the shared prefix ends with ``QUESTION:``; each question is a suffix up to the
assistant header), and the answer is a softmax over each option's best token
logit at the answer slot. d1 ships no calibration, so the probabilities are
returned as read.

Several questions over one state share the prefix, as in the release runner's
trunk-and-branches pass: the prefix is prefilled once, then the questions run
together as one right-padded batch that continues from the prefix cache. The
release packs its branches without padding under a tree mask; padding after a
row's last token cannot reach that token, so each row reads what it would
alone. A single question is one plain pass over the whole prompt, as the
runner does.

Prefix caches are kept between requests, so later requests on the same state
run only their questions. A single question on an uncached prefix keeps the
one plain pass unless the prefix is long enough to be worth keeping. The same
question can therefore read slightly differently with and without a cached
prefix (different batch shapes round differently in bfloat16), as the
release's single and multi-question passes do.
"""

import json
import math
from collections import OrderedDict
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from .decision import (
    ChunkSize,
    DecisionBackbone,
    DecisionContextLengthError,
    DecisionRequestError,
    round4,
)

# Prefix tokens repeated across one batch of questions: each row of a batch
# holds its own copy of the prefix's attention cache while it runs.
BRANCH_PREFIX_TOKENS = 32768
# Bytes of prefix caches kept between requests. The engine pool's memory
# estimate for the model does not include them.
PREFIX_CACHE_BYTES = 512 * 1024**2
# A single question on an uncached prefix splits it off, and keeps it, only from
# this many prefix tokens. The split costs one extra short pass (about 11 ms
# on d1-3B 4-bit, M5 Max) and a cache hit then saves the whole prefix pass.
SPLIT_PREFIX_TOKENS = 512

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

    @staticmethod
    def branch_caches(cache: list, rows: int) -> list:
        """Fresh caches that continue ``cache`` for ``rows`` rows.

        The prefix arrays are broadcast, not copied; the layers write new
        arrays into the new caches, so ``cache`` itself is never changed.
        """
        out = []
        for layer in cache:
            if hasattr(layer, "keys"):  # KVCache
                fresh = type(layer)()
                keys, values = layer.state
                fresh.state = (
                    mx.broadcast_to(keys, (rows, *keys.shape[1:])),
                    mx.broadcast_to(values, (rows, *values.shape[1:])),
                )
            else:  # ArraysCache: the conv layers' recent inputs
                fresh = type(layer)(len(layer.cache))
                fresh.cache = [
                    None if a is None else mx.broadcast_to(a, (rows, *a.shape[1:]))
                    for a in layer.cache
                ]
            out.append(fresh)
        return out

    def prefill_branches(self, cache: list, rows: list[np.ndarray]) -> mx.array:
        """Run ``rows`` as one right-padded batch after ``cache``.

        Returns each row's hidden state at its last token, ``[len(rows), D]``.
        """
        width = max(len(r) for r in rows)
        tokens = np.zeros((len(rows), width), dtype=np.int32)
        for i, r in enumerate(rows):
            tokens[i, : len(r)] = r
        tokens = mx.array(tokens)
        hidden = self._forward(
            tokens,
            self._text_model.embed_tokens(tokens),
            None,
            self.branch_caches(cache, len(rows)),
        )
        last = hidden[mx.arange(len(rows)), mx.array([len(r) - 1 for r in rows])]
        mx.eval(last)
        return last

    @staticmethod
    def snapshot_cache(cache: list) -> list:
        """A compact copy of ``cache`` for keeping.

        Every array is copied: the KV state is a view of a buffer with spare
        capacity, and each conv state is a slice of its layer's whole prefill
        buffer, which a kept slice would hold alive.
        """
        out = []
        for layer in cache:
            if hasattr(layer, "keys"):  # KVCache
                fresh = type(layer)()
                keys, values = layer.state
                fresh.state = (mx.contiguous(keys), mx.contiguous(values))
            else:
                fresh = type(layer)(len(layer.cache))
                fresh.cache = [
                    None if a is None else mx.contiguous(a) for a in layer.cache
                ]
            out.append(fresh)
        mx.eval([c.state for c in out])
        return out


def cache_nbytes(cache: list) -> int:
    return sum(a.nbytes for layer in cache for _, a in tree_flatten(layer.state))


class PrefixCache:
    """Least recently used prefix caches, keyed by the prefix token ids.

    Entries are only read: questions continue from fresh caches built over
    them (``Lfm2DecisionBackbone.branch_caches``).
    """

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.nbytes = 0
        self._entries: OrderedDict[bytes, tuple[list, int]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: bytes) -> list | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        return entry[0]

    def put(self, key: bytes, cache: list) -> None:
        size = cache_nbytes(cache)
        if size > self.max_bytes:
            return
        old = self._entries.pop(key, None)
        if old is not None:
            self.nbytes -= old[1]
        self._entries[key] = (cache, size)
        self.nbytes += size
        while self.nbytes > self.max_bytes:
            _, (_, evicted) = self._entries.popitem(last=False)
            self.nbytes -= evicted

    def clear(self) -> None:
        self._entries.clear()
        self.nbytes = 0


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
        self.prefixes = PrefixCache(PREFIX_CACHE_BYTES)

    def load(self) -> None:
        self.backbone.load()
        bos = getattr(self.backbone.tokenizer, "bos_token", None)
        self._bos = bos if isinstance(bos, str) else ""
        self._aliases = {}
        self.prefixes.clear()

    def close(self) -> None:
        self.prefixes.clear()
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
        if request.get("videos"):
            raise DecisionRequestError("this model does not take videos")
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
        """Answer every question, yielding per prefill chunk.

        ``cached_tokens`` counts the prefix tokens read from the prefix cache;
        ``input_tokens`` counts them too, as the release does.
        """
        backbone = self.backbone
        answers = {}
        trunk = self._prefix_ids(plan)
        key = trunk.tobytes()
        branches = [
            np.asarray(backbone.tokenize(item.suffix), dtype=np.int32)
            for item in plan.items
        ]
        trunk_cache = self.prefixes.get(key)
        one_pass = False
        if len(plan.items) == 1:
            # The release reads a single question in one pass over the whole
            # prompt. Its prefix is only split off when that leaves the token
            # ids unchanged: text can merge across the boundary (the prefix
            # ends with a newline).
            ids = np.asarray(
                backbone.tokenize(plan.prefix + plan.items[0].suffix), dtype=np.int32
            )
            splits = np.array_equal(ids, np.concatenate([trunk, branches[0]]))
            one_pass = not splits or (
                trunk_cache is None and len(trunk) < SPLIT_PREFIX_TOKENS
            )
        if one_pass:
            item = plan.items[0]
            hidden = yield from self._prefill(ids, backbone.make_cache(), chunk_size)
            answers[item.question_id] = self._answer(item, self._probs(item, hidden))
            return {
                "answers": answers,
                "input_tokens": int(len(ids)),
                "cached_tokens": 0,
            }

        cached_tokens = 0 if trunk_cache is None else len(trunk)
        if trunk_cache is None:
            fresh = backbone.make_cache()
            yield from self._prefill(trunk, fresh, chunk_size)
            trunk_cache = backbone.snapshot_cache(fresh)
            self.prefixes.put(key, trunk_cache)
        hidden = yield from self._branches(
            trunk_cache, len(trunk), branches, chunk_size
        )
        for item, h in zip(plan.items, hidden, strict=True):
            answers[item.question_id] = self._answer(item, self._probs(item, h))
        input_tokens = len(trunk) + sum(len(b) for b in branches)
        return {
            "answers": answers,
            "input_tokens": int(input_tokens),
            "cached_tokens": int(cached_tokens),
        }

    def _prefix_ids(self, plan: D1Plan) -> np.ndarray:
        return np.asarray(self.backbone.tokenize(plan.prefix), dtype=np.int32)

    def prefix_key(self, plan: D1Plan) -> bytes:
        """The prefix cache key of ``plan``: its prefix token ids."""
        return self._prefix_ids(plan).tobytes()

    def _branches(
        self,
        trunk_cache: list,
        trunk_len: int,
        branches: list[np.ndarray],
        chunk_size: ChunkSize,
    ) -> Generator[int, None, list[mx.array]]:
        """Each branch's last hidden state after the prefix, in request order.

        Branches are grouped shortest first, so a batch pads little, and a
        batch stays within one prefill chunk and ``BRANCH_PREFIX_TOKENS``. A
        branch that fills a chunk alone is prefilled chunk by chunk.
        """
        order = sorted(range(len(branches)), key=lambda i: len(branches[i]))
        hidden: list[mx.array | None] = [None] * len(branches)
        start = 0
        while start < len(order):
            limit = max(1, chunk_size())
            group = [order[start]]
            for i in order[start + 1 :]:
                rows = len(group) + 1
                if (
                    rows * len(branches[i]) > limit
                    or rows * trunk_len > BRANCH_PREFIX_TOKENS
                ):
                    break
                group.append(i)
            if len(group) == 1:
                i = group[0]
                hidden[i] = yield from self._prefill(
                    branches[i],
                    self.backbone.branch_caches(trunk_cache, 1),
                    chunk_size,
                )
            else:
                out = self.backbone.prefill_branches(
                    trunk_cache, [branches[i] for i in group]
                )
                for row, i in enumerate(group):
                    hidden[i] = out[row]
                yield sum(len(branches[i]) for i in group)
            start += len(group)
        return hidden

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
