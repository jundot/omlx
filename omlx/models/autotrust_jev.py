# SPDX-License-Identifier: Apache-2.0
#
# The prompt format, slot readout and calibration follow the System 1 path
# (``s1_pass``) of ``serve_decide.py`` in the autotrust JEV model
# repositories (https://huggingface.co/autotrust), Apache License 2.0.
"""
autotrust JEV decision models (JEV-27B, JEV-9B).

An autotrust JEV bundle is an unchanged Qwen3.5-family backbone plus a PEFT
LoRA adapter, a decision head and one temperature per question kind. Each
question becomes one prompt in the release's ``bare-v1`` format, without a
chat template or BOS token::

    [kind] {kind}
    [state] {state}
    [question] {question}
    [options]
    {one line per option}
    [decision]:

The head reads the final normed hidden state of the last prompt token into
the slots of the question's kind (two for noul, six for score, sixteen for
choice), the kind's temperature scales the logits, and a softmax gives the
probabilities. The adapter is applied unmerged with
``omlx.lora_adapters.apply_adapter``, so quantized backbones work.

Prompts of the same kind share their first line and the state, so that
token prefix is prefilled once and each readout continues from a copy of the
cache.

How System One requests map onto the release format:

- state: a string as it is; any other JSON value as JSON
  (``ensure_ascii=False``).
- question: ``instructions`` as text (non-strings as JSON); required.
- choice: one line per option, ``{label}) {key}: {description}`` or
  ``{label}) {key}``, newlines flattened, labels ``A``..``P`` (the head's
  choice slots). More than 16 options (up to 256) use the release's
  ``tournament`` strategy: equal groups of at most 16 options are asked one
  after the other, the winner of each group and the best runners-up (16 in
  all) are asked again in a final, and the group and final probabilities are
  combined the way the release combines them. Every pass reads only the
  head's slots.
- score with six levels: the native kind, options ``0``..``5``. Level
  descriptions, for the levels that say more than their index, follow the
  question as ``Levels: 0 = ...; 1 = ...; ....`` A ``0: `` prefix that repeats the
  index (``/v1/decisions`` levels) is dropped.
- score with any other number of levels (2 to 256): asked as a choice over
  the level texts in order; the answer keeps the score shape, with the
  expected level as the score.
- noul: options ``false``, ``true``; the answer is P(true). ``criteria``
  descriptions follow the question as ``True means: .... False means: ....``

The training data has no convention for rubric text (score questions name
their scale in the question, noul questions carry no criteria), so the
appended sentences are this module's choice. They stay on the question's
line, like the explanations that release questions do carry.
"""

import copy
import json
import math
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from ..lora_adapters import apply_adapter, compatibility_problem, read_adapter_info
from ..model_discovery import (
    AUTOTRUST_JEV_CALIBRATION,
    AUTOTRUST_JEV_HEAD_WEIGHTS,
    AUTOTRUST_JEV_JUDGE_CONFIG,
    autotrust_jev_adapter_subfolder,
)
from .decision import (
    ChunkSize,
    DecisionBackbone,
    DecisionContextLengthError,
    DecisionRequestError,
    round4,
)
from .openjev import choice_confidence, score_confidence

KINDS = ("noul", "score", "choice")
NOUL_LINES = ("false", "true")
SCORE_LEVELS = 6
MAX_CHOICE_OPTIONS = 256
# The line an option label appears in, as the release checks its labels.
LABEL_CONTEXT = "x\n{}) y"


@dataclass
class _Readout:
    question_id: str
    # The prompt's kind, which picks the head slots. A score that is not
    # six levels is asked as a choice.
    kind: str
    # The System One question type, which picks the answer shape.
    answer: str
    # Answer keys in option order: option keys for choice, levels for score.
    keys: list[str]
    # The prompts known before the run: the only one, or one per tournament
    # group. A tournament's final depends on the group results.
    passes: list[np.ndarray]
    # Tournament only: the option groups, and what the final prompt needs.
    groups: list[list[int]] | None = None
    question: str = ""
    options: list[str] = field(default_factory=list)


@dataclass
class AutoTrustJevPlan:
    questions: dict
    state: str
    readouts: list[_Readout]


def render_state(state: Any) -> str:
    """The state as text: strings as they are, other JSON values as JSON."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_prompt(kind: str, state: str, question: str, lines: list[str]) -> str:
    """One prompt in the release's ``bare-v1`` format."""
    return (
        f"[kind] {kind}\n[state] {state}\n[question] {question}\n[options]\n"
        + "\n".join(lines)
        + "\n[decision]:"
    )


def _question_text(question: dict) -> str:
    instructions = question.get("instructions")
    if instructions is None:
        raise DecisionRequestError("instructions is required")
    if isinstance(instructions, str):
        return instructions
    return json.dumps(instructions, ensure_ascii=False)


def _one_line(text: str) -> str:
    """``text`` on one line: its lines, trimmed, joined by single spaces."""
    return " ".join(part.strip() for part in text.splitlines() if part.strip())


def _description(value: Any) -> str:
    """A criteria value as one line of text; empty when there is none."""
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return _one_line(text).strip()


def _sentence(text: str) -> str:
    return text if text.endswith((".", "!", "?")) else text + "."


def _option_text(key: str, description: Any) -> str:
    """``key: description``, or ``key`` alone, on a single line."""
    text = _description(description)
    return _one_line(f"{key}: {text}" if text else key)


def _level_text(index: int, level: Any) -> str:
    """A score level's description, without a leading ``{index}: ``."""
    text = _description(level)
    return text.removeprefix(f"{index}: ") if text != str(index) else ""


def noul_question(question: str, criteria: dict | None) -> str:
    """The question, followed by what true and false mean when given."""
    parts = [question]
    for name in ("true", "false"):
        text = _description((criteria or {}).get(name))
        if text:
            parts.append(f"{name.capitalize()} means: {_sentence(text)}")
    return " ".join(parts)


def score_question(question: str, levels: list) -> str:
    """The question, followed by the six level descriptions when given."""
    texts = [_level_text(i, level) for i, level in enumerate(levels)]
    if not any(texts):
        return question
    listed = "; ".join(f"{i} = {text}" for i, text in enumerate(texts) if text)
    return f"{question} Levels: {_sentence(listed)}"


def tournament_groups(count: int, size: int) -> list[list[int]]:
    """Option indices in consecutive groups of at most ``size``, as equal in
    size as possible (the release's ``_groups``)."""
    groups = math.ceil(count / size)
    base, extra = divmod(count, groups)
    out, start = [], 0
    for group in range(groups):
        length = base + (1 if group < extra else 0)
        out.append(list(range(start, start + length)))
        start += length
    return out


def tournament_finalists(
    groups: list[list[int]], in_group: dict[int, float], size: int
) -> list[int]:
    """Each group's winner, then the best runners-up, up to ``size`` options,
    in option order. Ties go to the earlier option."""
    winners = [max(group, key=lambda o: (in_group[o], -o)) for group in groups]
    taken = set(winners)
    runners_up = sorted(
        (o for group in groups for o in group if o not in taken),
        key=lambda o: (-in_group[o], o),
    )
    return sorted(winners + runners_up[: max(0, size - len(winners))])


def tournament_probabilities(
    groups: list[list[int]], in_group: dict[int, float], final: dict[int, float]
) -> list[float]:
    """Combine group and final probabilities as the release does.

    A finalist gets its final probability; any other option shares its
    group's final mass in proportion to its group probability.
    """
    group_of = {o: g for g, group in enumerate(groups) for o in group}
    share = [0.0] * len(groups)
    cap = [0.0] * len(groups)
    for o, p in final.items():
        share[group_of[o]] += p
        cap[group_of[o]] += in_group[o]
    among = sum(a * b for a, b in zip(share, cap, strict=True))
    count = sum(len(group) for group in groups)
    probs = [
        final[o] * among if o in final else share[group_of[o]] * in_group[o]
        for o in range(count)
    ]
    total = sum(probs)
    return [p / total for p in probs]


class AutoTrustJevModel:
    """An autotrust JEV bundle: backbone, unmerged LoRA adapter and head."""

    def __init__(self, model_path: str, trust_remote_code: bool = False):
        self.model_path = model_path
        self.backbone = DecisionBackbone(model_path, trust_remote_code)
        self._ranges: dict[str, tuple[int, int]] = {}
        self._temperatures: dict[str, float] = {}
        self._choice_labels: list[str] = []
        self._head_weight: mx.array | None = None
        self._head_bias: mx.array | None = None

    @property
    def has_vision(self) -> bool:
        # The adapter and head were trained on text, even when the backbone
        # has a vision tower.
        return False

    def load(self) -> None:
        path = Path(self.model_path)
        subfolder = autotrust_jev_adapter_subfolder(path)
        if subfolder is None:
            raise ValueError(f"{path} is not an autotrust JEV bundle")
        judge = json.loads((path / AUTOTRUST_JEV_JUDGE_CONFIG).read_text())
        ranges, verbalizers, verbalizer_ids = _check_judge(judge)
        temperatures = _read_temperatures(path / AUTOTRUST_JEV_CALIBRATION)
        head = mx.load(str(path / AUTOTRUST_JEV_HEAD_WEIGHTS))
        weight, bias = head.get("proj.weight"), head.get("proj.bias")
        if weight is None or bias is None:
            raise ValueError(
                f"{AUTOTRUST_JEV_HEAD_WEIGHTS} needs proj.weight and proj.bias"
            )
        slots = len(verbalizer_ids)
        if weight.shape != (slots, judge["hidden_size"]) or bias.shape != (slots,):
            raise ValueError(
                f"head shapes {weight.shape} and {bias.shape} do not fit "
                f"{slots} slots and hidden size {judge['hidden_size']}"
            )
        # Checking the adapter first fails fast before the backbone loads.
        adapter_path = path / subfolder
        info = read_adapter_info(adapter_path)
        if info is None:
            raise ValueError(f"{adapter_path} is not a LoRA adapter folder")
        problem = compatibility_problem(
            info, json.loads((path / "config.json").read_text())
        )
        if problem:
            raise ValueError(f"the autotrust JEV adapter cannot be applied: {problem}")

        self.backbone.load()
        config = self.backbone.config.get("text_config") or self.backbone.config
        if config.get("hidden_size") != judge["hidden_size"]:
            raise ValueError(
                f"the backbone's hidden size {config.get('hidden_size')} is not "
                f"{judge['hidden_size']}: this bundle needs a different base model"
            )
        for text, token_id in zip(verbalizers, verbalizer_ids, strict=True):
            if self.backbone.tokenize(text) != [token_id]:
                raise ValueError(
                    f"the backbone's tokenizer does not encode {text!r} as "
                    f"{token_id}: this bundle needs a different base model"
                )
        lo, hi = ranges["choice"]
        for label, token_id in zip(verbalizers[lo:hi], verbalizer_ids[lo:hi]):
            if token_id not in self.backbone.tokenize(LABEL_CONTEXT.format(label)):
                raise ValueError(
                    f"the choice label {label!r} is not its own token in an "
                    "option line: this bundle needs a different base model"
                )
        apply_adapter(self.backbone._text_model, adapter_path)

        self._ranges = ranges
        self._temperatures = temperatures
        self._choice_labels = verbalizers[lo:hi]
        self._head_weight = weight.astype(mx.float32)
        self._head_bias = bias.astype(mx.float32)
        mx.eval(self._head_weight, self._head_bias)

    def close(self) -> None:
        self.backbone.close()
        self._head_weight = None
        self._head_bias = None

    def encode(self, request: dict, truncate: bool = True) -> AutoTrustJevPlan:
        """Validate the request and tokenize one prompt per question.

        The state is never cut; ``truncate`` is accepted for API symmetry and
        a prompt over the context limit is an error.
        """
        if request.get("images") or request.get("videos"):
            raise DecisionRequestError("autotrust JEV models read text only")
        state = render_state(request["state"])
        readouts = []
        for question_id, question in request["questions"].items():
            answer = question["type"]
            kind, keys, text, options = self._prompt_parts(question)
            readout = _Readout(str(question_id), kind, answer, keys, passes=[])
            slots = len(self._choice_labels)
            if kind != "choice":
                readout.passes = [self._prompt_ids(kind, state, text, options)]
            elif len(options) <= slots:
                lines = self._choice_lines(options)
                readout.passes = [self._prompt_ids(kind, state, text, lines)]
            else:
                readout.groups = tournament_groups(len(options), slots)
                readout.passes = [
                    self._prompt_ids(
                        kind, state, text, self._choice_lines([options[i] for i in g])
                    )
                    for g in readout.groups
                ]
                readout.question, readout.options = text, options
                # The final asks some 16 of the options; check the 16 whose
                # option lines take the most tokens.
                label = self._choice_labels[0]
                size = [len(self.backbone.tokenize(f"{label}) {o}")) for o in options]
                order = sorted(range(len(options)), key=lambda i: -size[i])
                worst = [options[i] for i in order[:slots]]
                self._prompt_ids(kind, state, text, self._choice_lines(worst))
            readouts.append(readout)
        return AutoTrustJevPlan(request["questions"], state, readouts)

    def _prompt_ids(
        self, kind: str, state: str, question: str, lines: list[str]
    ) -> np.ndarray:
        """Token ids of one prompt; an error if it is over the context limit."""
        prompt = render_prompt(kind, state, question, lines)
        ids = np.asarray(self.backbone.tokenize(prompt), dtype=np.int32)
        limit = self.backbone.max_position_embeddings
        if limit is not None and len(ids) > limit:
            raise DecisionContextLengthError(
                f"the prompt needs {len(ids)} tokens; the limit is {limit}"
            )
        return ids

    def _prompt_parts(self, question: dict) -> tuple[str, list[str], str, list[str]]:
        """Prompt kind, answer keys, question text, and the option texts for
        choice or the option lines for the other kinds."""
        text = _question_text(question)
        criteria = question.get("criteria")
        if question["type"] == "choice":
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise DecisionRequestError(
                    "choice criteria must be a map of at least two options to "
                    "a description or null"
                )
            keys = [str(key) for key in criteria]
            options = [_option_text(k, d) for k, d in zip(keys, criteria.values())]
            return "choice", keys, text, self._check_option_count(options)
        if question["type"] == "score":
            if not isinstance(criteria, list) or len(criteria) < 2:
                raise DecisionRequestError(
                    "score criteria must be an ordered array of at least two levels"
                )
            keys = [str(i) for i in range(len(criteria))]
            if len(criteria) == SCORE_LEVELS:
                return "score", keys, score_question(text, criteria), keys
            options = [_level_text(i, c) or str(i) for i, c in enumerate(criteria)]
            return "choice", keys, text, self._check_option_count(options)
        if criteria is not None and not isinstance(criteria, dict):
            raise DecisionRequestError("noul criteria must be an object")
        return "noul", list(NOUL_LINES), noul_question(text, criteria), list(NOUL_LINES)

    def _check_option_count(self, options: list[str]) -> list[str]:
        """At most 256 options, and no more than a tournament over the head's
        choice slots can finish: as many groups as slots, each at most that
        size."""
        slots = len(self._choice_labels)
        limit = min(MAX_CHOICE_OPTIONS, slots * slots)
        if len(options) > limit:
            raise DecisionRequestError(f"choice takes at most {limit} options")
        return options

    def _choice_lines(self, options: list[str]) -> list[str]:
        """Option lines of one pass: at most one per head slot."""
        return [
            f"{label}) {text}"
            for label, text in zip(self._choice_labels, options, strict=False)
        ]

    def run(
        self, plan: AutoTrustJevPlan, chunk_size: ChunkSize
    ) -> Generator[int, None, dict]:
        """Run every readout, yielding per prefill chunk, then build answers."""
        probabilities = {}
        input_tokens = 0
        for kind in KINDS:
            group = [r for r in plan.readouts if r.kind == kind]
            shared = yield from self._prefill_shared_prefix(
                [ids for r in group for ids in r.passes], chunk_size
            )
            for readout in group:
                if readout.groups is None:
                    ids = readout.passes[0]
                    probs = yield from self._readout(
                        ids, kind, len(readout.keys), shared, chunk_size
                    )
                    tokens = len(ids)
                else:
                    probs, tokens = yield from self._tournament(
                        plan, readout, shared, chunk_size
                    )
                probabilities[readout.question_id] = probs
                input_tokens += tokens
        answers = {
            r.question_id: self._answer(
                r, plan.questions[r.question_id], probabilities[r.question_id]
            )
            for r in plan.readouts
        }
        return {"answers": answers, "input_tokens": input_tokens}

    def _tournament(
        self,
        plan: AutoTrustJevPlan,
        readout: _Readout,
        shared: tuple | None,
        chunk_size: ChunkSize,
    ) -> Generator[int, None, tuple[list[float], int]]:
        """Probabilities over more options than the head has slots, and the
        prompt tokens used."""
        in_group: dict[int, float] = {}
        tokens = 0
        for group, ids in zip(readout.groups, readout.passes, strict=True):
            probs = yield from self._readout(
                ids, "choice", len(group), shared, chunk_size
            )
            in_group.update(zip(group, probs, strict=True))
            tokens += len(ids)
        slots = len(self._choice_labels)
        finalists = tournament_finalists(readout.groups, in_group, slots)
        lines = self._choice_lines([readout.options[o] for o in finalists])
        ids = self._prompt_ids("choice", plan.state, readout.question, lines)
        final = yield from self._readout(
            ids, "choice", len(finalists), shared, chunk_size
        )
        tokens += len(ids)
        combined = tournament_probabilities(
            readout.groups, in_group, dict(zip(finalists, final, strict=True))
        )
        return combined, tokens

    def _prefill_shared_prefix(
        self, readouts: list[np.ndarray], chunk_size: ChunkSize
    ) -> Generator[int, None, tuple | None]:
        """Prefill the longest token prefix common to the readouts of a kind."""
        if len(readouts) < 2:
            return None
        # Each readout keeps at least one token of its own to read out.
        length = min(len(ids) for ids in readouts) - 1
        reference = readouts[0]
        for ids in readouts[1:]:
            mismatch = np.nonzero(ids[:length] != reference[:length])[0]
            if mismatch.size:
                length = int(mismatch[0])
        if length <= 0:
            return None
        prefix = reference[:length]
        backbone = self.backbone
        embeds, positions = backbone.embed(prefix)
        cache = backbone.make_cache()
        yield from backbone.prefill(
            prefix, embeds, positions, cache, chunk_size, keep_all=False
        )
        last_position = None if positions is None else positions[:, :, -1]
        return prefix, cache, last_position

    def _readout(
        self,
        ids: np.ndarray,
        kind: str,
        count: int,
        shared: tuple | None,
        chunk_size: ChunkSize,
    ) -> Generator[int, None, list[float]]:
        """Probabilities over the first ``count`` slots of ``kind``."""
        backbone = self.backbone
        if shared is not None and np.array_equal(ids[: len(shared[0])], shared[0]):
            prefix, prefix_cache, last_position = shared
            # Caches update in place; each readout continues from its own copy.
            cache = copy.deepcopy(prefix_cache)
            tail = ids[len(prefix) :]
            embeds, positions = backbone.embed_continuation(tail, last_position)
        else:
            cache = backbone.make_cache()
            tail = ids
            embeds, positions = backbone.embed(ids)
        hidden = yield from backbone.prefill(
            tail, embeds, positions, cache, chunk_size, keep_all=False
        )
        logits = self._head_weight @ hidden.astype(mx.float32) + self._head_bias
        lo = self._ranges[kind][0]
        slots = np.asarray(logits[lo : lo + count].tolist(), dtype=np.float64)
        if not np.all(np.isfinite(slots)):
            raise RuntimeError("decision head logits are not finite")
        z = slots / self._temperatures[kind]
        weights = np.exp(z - z.max())
        return (weights / weights.sum()).tolist()

    def _answer(self, readout: _Readout, question: dict, probs: list[float]) -> dict:
        if readout.answer == "choice":
            best = int(np.argmax(probs))
            return {
                "type": "choice",
                "choice": readout.keys[best],
                "probabilities": {
                    key: round4(p) for key, p in zip(readout.keys, probs, strict=True)
                },
                "confidence": round4(choice_confidence(probs)),
            }
        if readout.answer == "score":
            levels = question["criteria"]
            return {
                "type": "score",
                "score": round4(sum(i * p for i, p in enumerate(probs))),
                "legend": {str(i): level for i, level in enumerate(levels)},
                "probabilities": {str(i): round4(p) for i, p in enumerate(probs)},
                "confidence": round4(score_confidence(probs)),
            }
        # Slot order is false, true.
        return {"type": "noul", "noul": round4(probs[1])}


def _check_judge(judge: dict) -> tuple[dict, list[str], list[int]]:
    """Slot ranges, verbalizers and their token ids of a judge config."""
    if judge.get("weights_mode", "unmerged") != "unmerged":
        raise ValueError(
            f"weights_mode {judge['weights_mode']!r} is not supported, only "
            "'unmerged'"
        )
    if judge.get("softcap") is not None:
        raise ValueError("a head softcap is not supported")
    if not isinstance(judge.get("hidden_size"), int):
        raise ValueError("the judge config needs an integer hidden_size")
    verbalizers = judge["slots"].get("verbalizers")
    verbalizer_ids = judge["verbalizer_ids"]
    if not isinstance(verbalizers, list) or len(verbalizers) != len(verbalizer_ids):
        raise ValueError("verbalizers and verbalizer_ids must have the same length")
    sizes = {"noul": len(NOUL_LINES), "score": SCORE_LEVELS, "choice": None}
    ranges = {}
    for kind, size in sizes.items():
        lo, hi = judge["slots"]["ranges"].get(kind, (0, 0))
        if not 0 <= lo < hi <= len(verbalizer_ids) or size not in (None, hi - lo):
            raise ValueError(f"the {kind} slot range {[lo, hi]} is not supported")
        ranges[kind] = (lo, hi)
    if ranges["choice"][1] - ranges["choice"][0] < 2:
        raise ValueError("the choice slot range needs at least two slots")
    return ranges, verbalizers, verbalizer_ids


def _read_temperatures(path: Path) -> dict[str, float]:
    """One temperature per kind; 1.0 when the bundle has no calibration."""
    per_kind = {}
    if path.is_file():
        per_kind = json.loads(path.read_text()).get("per_kind")
        if not isinstance(per_kind, dict):
            raise ValueError(f"{path.name} needs a per_kind map of temperatures")
    temperatures = {}
    for kind in KINDS:
        value = float(per_kind.get(kind, 1.0))
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"the {kind} temperature {value} is not positive")
        temperatures[kind] = value
    return temperatures
