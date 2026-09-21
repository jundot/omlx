# SPDX-License-Identifier: Apache-2.0
"""Questions → system text, answer template, and a slot map. Pure.

Ported from ``openjev/openjev/engine.py`` (razorback16/openjev, Apache-2.0),
which in turn adapts ``vllm-project/vllm#57250``. The token-diffing trick in
:func:`resolve_template` and the label-packing in :func:`choice_labels` are the
upstream mechanism and are reproduced rather than reinvented.

Nothing here imports ``mlx`` or a tokenizer module: a tokenizer is duck-typed
through :class:`TokenizerLike` and only ever asked for ``encode``. That keeps
quantisation, device and weight concerns out of the layer that decides *what
the model sees*.

Two deliberate departures from upstream, both documented at their site:

* :func:`canvas_width` is exact, not rounded up to a step and padded. Canvas
  rows attend bidirectionally, so a pad row is a real token that perturbs the
  label slots. See ``notes/openjev-systemone-on-omlx.md`` §11.
* :func:`chat_prompt_ids` does not append :data:`SCAFFOLD_TEXT` blindly.
  Whether the served chat template already emits the empty thought block varies
  between re-quants of the same base model, so the caller asserts the class it
  got. Notes §10.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .types import ReadGroup, ReadSlot

__all__ = [
    "SCAFFOLD_TEXT",
    "SchemaError",
    "TokenizerLike",
    "answer_text",
    "build_prompt_ids",
    "build_schema",
    "canvas_width",
    "choice_labels",
    "groups",
    "read_groups",
    "resolve_template",
    "scaffold_ids",
    "system_text",
    "template_emits_scaffold",
]

#: The empty thought block the chat template may or may not leave to the model.
#: Tokenizes to 4 tokens ``[100, 45518, 107, 101]`` on the diffusiongemma
#: tokenizer — verified, not assumed (notes §10).
SCAFFOLD_TEXT = "<|channel>thought\n<channel|>"

TURN_CLOSE = 106  # '<turn|>'

#: vLLM's ``logprob_token_ids`` cap per request. Kept because a wider choice
#: list is a schema problem the caller must see, not a silent truncation.
MAX_LABEL_IDS = 128

#: Answer template shapes: (join between questions, what precedes the label,
#: reply instruction). ``indexed`` costs fewer rows a question; past ten
#: questions the saved rows keep a schema in one read.
FORMATS: dict[str, tuple[str, str, str]] = {
    "lines": (
        "\n",
        "{id}: ",
        'Reply with one line per question, in this order, formatted as "id: label".',
    ),
    "indexed": (
        " ",
        "{id}",
        "Reply on one line with each question's id immediately followed by its "
        "label, separated by single spaces.",
    ),
}

#: Questions answered per format boundary, upstream's ``<= 10`` rule.
INDEXED_ABOVE = 10


class SchemaError(ValueError):
    """A request the model cannot answer as asked.

    ``loc`` mirrors the wire path so a route can surface it as a ``422``
    without re-deriving where the problem was.
    """

    def __init__(self, msg: str, loc: Sequence[Any] = ("body",)) -> None:
        super().__init__(msg)
        self.loc = list(loc)


class TokenizerLike(Protocol):
    """The only tokenizer surface this module needs."""

    def encode(
        self, text: str, *, add_special_tokens: bool = False
    ) -> Sequence[int]: ...


@dataclass(frozen=True)
class Schema:
    """Compiled questions: what the model sees, and how answers map back."""

    questions: tuple[dict[str, Any], ...]
    fmt: str

    def __len__(self) -> int:
        return len(self.questions)


def text_of(value: Any) -> str:
    """Descriptions and instructions may be strings, objects or arrays."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False)


def _enc(tok: TokenizerLike, text: str) -> list[int]:
    return [int(t) for t in tok.encode(text, add_special_tokens=False)]


def scaffold_ids(tok: TokenizerLike) -> tuple[int, ...]:
    """The scaffold as token ids, from the *live* tokenizer."""

    return tuple(_enc(tok, SCAFFOLD_TEXT))


def template_emits_scaffold(
    tok: TokenizerLike,
    messages: Sequence[Mapping[str, Any]],
    *,
    apply_chat_template: Callable[..., Any] | None = None,
) -> bool:
    """Whether the served chat template already emits the empty thought block.

    The answer differs between re-quants of the same base model: the affine
    8-bit diffusiongemma template emits ``<|channel>thought\\n<channel|>`` when
    thinking is off, the mxfp4 and mxfp8 templates do not (notes §10). Callers
    append :data:`SCAFFOLD_TEXT` only when this returns ``False``, so that a
    prompt never carries the thought block twice.

    ``apply_chat_template`` defaults to the tokenizer's own method, which lets
    a test pass a stub tokenizer that has no chat template at all.
    """

    apply = apply_chat_template or getattr(tok, "apply_chat_template", None)
    if apply is None:
        raise SchemaError(
            "tokenizer has no apply_chat_template; cannot determine scaffold class",
            ("body", "model"),
        )
    out = apply(
        list(messages),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = out["input_ids"] if hasattr(out, "keys") else out
    ids = [int(t) for t in ids]
    return _ends_with(ids, scaffold_ids(tok))


def _ends_with(ids: Sequence[int], tail: Sequence[int]) -> bool:
    n = len(tail)
    return n > 0 and len(ids) >= n and list(ids[-n:]) == list(tail)


def choice_labels(tok: TokenizerLike) -> tuple[str, ...]:
    """Choice labels that stay one token after ``"q1: "``, in a stable order.

    A label that tokenizes to two tokens cannot share a canvas slot with its
    siblings, so the probe — not the alphabet — decides which labels exist.
    Upstream's rule, reproduced verbatim: candidates are ``A-Z``, ``a-z``, then
    two-letter pairs, kept while the encoding of ``"q1: " + label`` is the same
    length as the base and differs only in the last token.
    """

    base = _enc(tok, "q1: A")
    cands = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    cands += [chr(c) for c in range(ord("a"), ord("z") + 1)]
    cands += [
        a + b
        for a in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for b in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ]

    out: list[str] = []
    seen: set[int] = set()
    for cand in cands:
        e = _enc(tok, f"q1: {cand}")
        if len(e) == len(base) and e[:-1] == base[:-1] and e[-1] not in seen:
            seen.add(e[-1])
            out.append(cand)
        if len(out) == MAX_LABEL_IDS:
            break
    return tuple(out)


def build_schema(questions: Mapping[str, Any], tok: TokenizerLike) -> Schema:
    """Compile a request's questions.

    Ids never reach the model: it sees ``q1, q2, …`` and answers map back by
    position.
    """

    if not questions:
        raise SchemaError("no questions", ("body", "questions"))

    labels_for_choices = choice_labels(tok)

    out: list[dict[str, Any]] = []
    for i, (qid, q) in enumerate(questions.items()):
        loc = ("body", "questions", qid, "criteria")
        kind = q["type"]
        if kind == "noul":
            crit = q.get("criteria") or {}
            choices = [
                ("yes", text_of(crit.get("true"))),
                ("no", text_of(crit.get("false"))),
            ]
            labels = ["yes", "no"]
        elif kind == "choice":
            crit = q["criteria"]
            if len(crit) < 2:
                raise SchemaError("a choice needs at least two options", loc)
            if len(crit) > len(labels_for_choices):
                raise SchemaError(
                    f"at most {len(labels_for_choices)} options per choice",
                    loc,
                )
            choices = [(name, text_of(desc)) for name, desc in crit.items()]
            labels = list(labels_for_choices[: len(crit)])
        elif kind == "score":
            crit = q["criteria"]
            if not 2 <= len(crit) <= 10:
                raise SchemaError("a score takes 2 to 10 levels", loc)
            choices = [(str(i), text_of(c)) for i, c in enumerate(crit)]
            labels = [str(i) for i in range(len(crit))]
        else:
            raise SchemaError(
                f"unknown question type {kind!r}",
                ("body", "questions", qid, "type"),
            )

        out.append(
            {
                "key": qid,
                "id": f"q{i + 1}",
                "type": kind,
                "instructions": text_of(q.get("instructions")),
                "choices": choices,
                "labels": labels,
                # score legends echo the criteria exactly as sent
                "legend": list(q["criteria"]) if kind == "score" else None,
            }
        )

    return Schema(tuple(out), "lines" if len(out) <= INDEXED_ABOVE else "indexed")


def system_text(
    questions: Sequence[Mapping[str, Any]],
    fmt: str,
    chunked: bool = False,
) -> str:
    """The system turn: every question with its allowed labels spelled out."""

    s = (
        "Answer a fixed set of questions about the state the user provides. "
        "Each question lists its allowed answers; reply with exactly one label "
        "per question.\n"
    )
    for q in questions:
        s += f"\nQuestion {q['id']}: {q['instructions'] or 'Answer about the state.'}\n"
        for (name, desc), label in zip(q["choices"], q["labels"]):
            if q["type"] == "noul":
                s += f"  {label}: {desc}\n" if desc else f"  {label}\n"
            elif q["type"] == "score":
                s += f"  {label}: {desc}\n"
            else:
                s += f"  {label}: {name} ({desc})\n" if desc else f"  {label}: {name}\n"
    s += "\n" + FORMATS[fmt][2]
    if chunked:
        s += (
            " A reply may cover only some of the questions; answer every line"
            " that is present."
        )
    return s


def answer_text(
    questions: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
    fmt: str,
) -> str:
    """The answer template as text, with the given label index per question."""

    join, lead, _ = FORMATS[fmt]
    return join.join(
        lead.format(id=q["id"]) + q["labels"][i] for q, i in zip(questions, labels)
    )


def resolve_template(
    tok: TokenizerLike,
    questions: Sequence[Mapping[str, Any]],
    fmt: str,
    *,
    scaffold: Sequence[int] | None = None,
    lead: str = "",
) -> tuple[tuple[int, ...], tuple[ReadSlot, ...]]:
    """Tokenize the answer template and find each question's slot.

    Every label of a question must change exactly one token, at the same
    position for all of them. A question whose labels tokenize to different
    lengths cannot be answered by a single canvas position, so this raises
    rather than letting the read answer a different string.

    ``scaffold`` is the token run the canvas starts with — the empty thought
    block for a plain read, and nothing when the prompt already closes the
    thought.
    """

    head = list(scaffold) if scaffold is not None else []
    base_labels = [0] * len(questions)
    base = head + _enc(tok, lead + answer_text(questions, base_labels, fmt))

    slots: list[ReadSlot] = []
    for qi, q in enumerate(questions):
        pos: int | None = None
        ids: list[int] = [0] * len(q["labels"])
        for li in range(1, len(q["labels"])):
            labels = list(base_labels)
            labels[qi] = li
            e = head + _enc(tok, lead + answer_text(questions, labels, fmt))
            diffs = [i for i in range(min(len(e), len(base))) if e[i] != base[i]]
            same_len = len(e) == len(base)
            if not same_len or len(diffs) != 1 or (pos is not None and diffs[0] != pos):
                raise SchemaError(
                    f"question {q['key']!r}: labels do not share one template slot",
                    ("body", "questions", q["key"]),
                )
            pos = diffs[0]
            ids[li] = e[pos]
        if pos is None:
            # A single-label question has nothing to diff; upstream requires >=2.
            raise SchemaError(
                f"question {q['key']!r} has no alternative label",
                ("body", "questions", q["key"]),
            )
        ids[0] = base[pos]
        slots.append(
            ReadSlot(
                key=q["key"],
                position=pos,
                label_ids=tuple(ids),
                label_keys=tuple(name for name, _ in q["choices"]),
            )
        )

    return tuple(base), tuple(slots)


def canvas_width(template_len: int, canvas_length: int) -> int:
    """Exact canvas width: the template, the turn-close, nothing else.

    Upstream rounds up to ``canvas_step`` and pads with ``PAD``. That is not
    portable here — canvas rows attend bidirectionally to every other row, so a
    pad row is a real token that perturbs the label slots. The cap still
    applies, because the model's block size is a hard limit.
    """

    width = template_len + 1
    if canvas_length and width > canvas_length:
        raise SchemaError(
            f"answer template is {template_len} tokens; the canvas holds "
            f"{canvas_length}",
            ("body", "questions"),
        )
    return width


def groups(
    tok: TokenizerLike,
    questions: Sequence[Mapping[str, Any]],
    fmt: str,
    *,
    scaffold: Sequence[int],
    canvas_length: int,
) -> list[list[dict[str, Any]]]:
    """Split questions, in order, into the fewest groups that fit the canvas."""

    out: list[list[dict[str, Any]]] = []
    group: list[dict[str, Any]] = []
    for q in questions:
        trial = group + [q]
        text = answer_text(trial, [0] * len(trial), fmt)
        need = len(scaffold) + len(_enc(tok, text)) + 1
        if canvas_length and need > canvas_length and group:
            out.append(group)
            group = [q]
        else:
            group = trial
    out.append(group)
    return out


def build_prompt_ids(
    tok: TokenizerLike,
    system: str,
    state: str,
    *,
    thinking: bool = False,
    scaffold: Sequence[int] | None = None,
) -> list[int]:
    """The chat prompt as token ids, ending after the model turn marker.

    Text states only. The scaffold is appended only when the served template did
    not already emit it, so the thought block appears exactly once whatever
    quant is loaded (notes §10).
    """

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": state},
    ]
    out = tok.apply_chat_template(  # type: ignore[attr-defined]
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    ids = out["input_ids"] if hasattr(out, "keys") else out
    ids = [int(t) for t in ids]
    tail = tuple(scaffold) if scaffold is not None else scaffold_ids(tok)
    if not thinking and not _ends_with(ids, tail):
        ids = ids + list(tail)
    return ids


def read_groups(
    tok: TokenizerLike,
    schema: Schema,
    *,
    scaffold: Sequence[int] | None = None,
    canvas_length: int = 0,
) -> list[ReadGroup]:
    """Compile a schema into the canvas groups the engine reads in one go."""

    head = tuple(scaffold) if scaffold is not None else scaffold_ids(tok)
    out: list[ReadGroup] = []
    for group in groups(
        tok,
        list(schema.questions),
        schema.fmt,
        scaffold=head,
        canvas_length=canvas_length,
    ):
        template, slots = resolve_template(tok, group, schema.fmt, scaffold=head)
        first, last = group[0]["id"], group[-1]["id"]
        out.append(
            ReadGroup(
                template_ids=template,
                slots=slots,
                label=f"{first}-{'last' if len(group) == 1 else last}",
                metadata={"format": schema.fmt, "n_questions": len(group)},
            )
        )
    return out
