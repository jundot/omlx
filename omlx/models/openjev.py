# SPDX-License-Identifier: Apache-2.0
"""
OpenJev decision model.

OpenJev is a fine-tuned Qwen3.5 chat model. Each readout renders one
question as a chat prompt and reads the option-letter logits at the first
output position. The prompt text, letters and calibration constants follow
the release helper and its published serving settings, because the model was
tuned and calibrated on them.

All readouts of a request share the chat header and the state, so the shared
token prefix is prefilled once and each readout continues from a copy of
that cache.
"""

import ast
import copy
import json
import math
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import numpy as np

from .decision import (
    ChunkSize,
    DecisionBackbone,
    DecisionContextLengthError,
    DecisionRequestError,
    decode_images,
    round4,
)

LETTERS = [chr(code) for code in range(ord("A"), ord("Z") + 1)] + [
    chr(code) for code in range(ord("a"), ord("z") + 1)
]
READOUT_TEMPERATURE = 0.85
NOUL_TEMPERATURE = 1.829074
NOUL_BIAS = 0.0
NOUL_CLIP = 1e-4
# Longer strings in state.screenshot / state.image are raw base64 images.
RAW_IMAGE_MIN_CHARS = 2000
HISTORY_KEYS = ("previous_actions", "history", "recent_actions")
SCORE_SUFFIX = " Rate along the ordered levels below (lowest first)."


@dataclass
class OpenJevState:
    text: str
    image: str | None = None
    # State fields without the image, read by the screenshot task layout.
    fields: dict = field(default_factory=dict)


def split_state(state: Any, images: list[str] | None) -> OpenJevState:
    """Render the state and pick its image, if any."""
    result = None
    if isinstance(state, dict):
        for key in ("screenshot", "image"):
            value = state.get(key)
            if isinstance(value, str) and (
                value.startswith("data:image") or len(value) > RAW_IMAGE_MIN_CHARS
            ):
                rest = {k: v for k, v in state.items() if k != key}
                result = OpenJevState(
                    text=(
                        json.dumps(rest, ensure_ascii=False)
                        if rest
                        else "(see screenshot)"
                    ),
                    image=(
                        value
                        if value.startswith("data:image")
                        else "data:image/png;base64," + value
                    ),
                    fields=rest,
                )
                break
        if result is None:
            result = OpenJevState(json.dumps(state, ensure_ascii=False), fields=state)
    else:
        result = OpenJevState(state if isinstance(state, str) else str(state))
    if images:
        if result.image is not None or len(images) > 1:
            raise DecisionRequestError("OpenJev takes at most one image per request")
        result.image = images[0]
    return result


def instruction_text(question: dict) -> str:
    instructions = question.get("instructions")
    if instructions is None:
        raise DecisionRequestError("instructions is required")
    if isinstance(instructions, str):
        return instructions
    # The release model was tuned on Python-literal text for structured values.
    return str(instructions)


def description_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def task_text(fields: dict, instructions: str) -> str | None:
    """Return the agent task for the screenshot layout, if the request has one."""
    if fields.get("task"):
        return str(fields["task"])
    if instructions.startswith("{"):
        for parse in (json.loads, ast.literal_eval):
            try:
                value = parse(instructions)
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                continue
            return value.get("goal") if isinstance(value, dict) else None
    return None


def render_prompt(
    state: OpenJevState, instructions: str, options: list[tuple[str, str]]
) -> str:
    """Return the user message text of one readout."""
    task = task_text(state.fields, instructions) if state.image else None
    if task:
        history = next(
            (state.fields[k] for k in HISTORY_KEYS if state.fields.get(k)), []
        )
        entries = history if isinstance(history, list) else [history]
        history_text = "\n".join(
            f"- {e if isinstance(e, str) else json.dumps(e, ensure_ascii=False)}"
            for e in entries
        )
        extra = {
            k: v for k, v in state.fields.items() if k not in ("task", *HISTORY_KEYS)
        }
        marks = "\n".join(
            f"[{LETTERS[i]}] {desc if desc else key}"
            for i, (key, desc) in enumerate(options)
        )
        shown = (
            "The screenshot shows the current page with candidate elements marked "
            "by red letters."
            if "task" in state.fields
            else "The screenshot shows the current page; candidate elements:"
        )
        head = f"Task: {task}\nPrevious actions:\n{history_text or '- (none)'}\n"
        if extra:
            head += f"Context: {json.dumps(extra, ensure_ascii=False)}\n"
        return (
            f"{head}\n{shown}\n{marks}\n\n{instructions} Answer with the letter only."
        )
    lines = "\n".join(
        f"[{LETTERS[i]}] {key}: {desc}" for i, (key, desc) in enumerate(options)
    )
    lead = "The screenshot shows the current screen.\n" if state.image else ""
    return (
        f"{lead}State:\n{state.text}\n\nQuestion: {instructions}\nOptions:\n{lines}"
        "\n\nAnswer with the letter of the best option only."
    )


def option_groups(count: int) -> list[range]:
    """Split options into near-equal readouts of at most one letter each."""
    if count <= len(LETTERS):
        return [range(count)]
    groups = -(-count // len(LETTERS))
    size = -(-count // groups)
    return [range(start, min(count, start + size)) for start in range(0, count, size)]


def compose_groups(parts: list[list[float]], final: list[float]) -> list[float]:
    """Merge per-group readouts through the readout over the group winners.

    Each winner's probability in the final readout anchors the mass of its
    group, so every option keeps a non-zero share and the result sums to 1.
    """
    raw = []
    for group, probs in enumerate(parts):
        winner = probs[int(np.argmax(probs))]
        raw.extend(final[group] * p / winner for p in probs)
    total = sum(raw)
    return [value / total for value in raw]


def choice_confidence(probs: list[float]) -> float:
    if len(probs) == 1:
        return 1.0
    uniform = 1.0 / len(probs)
    return max(0.0, (max(probs) - uniform) / (1.0 - uniform))


def score_confidence(probs: list[float]) -> float:
    count = len(probs)
    if count == 1:
        return 1.0
    mode = int(np.argmax(probs))
    spread = sum(p * abs(i - mode) for i, p in enumerate(probs))
    center = (count - 1) / 2
    uniform_spread = sum(abs(i - center) for i in range(count)) / count
    return max(0.0, 1.0 - spread / uniform_spread)


def noul_probability(p_yes: float) -> float:
    p_yes = min(max(p_yes, NOUL_CLIP), 1.0 - NOUL_CLIP)
    z = math.log(p_yes / (1.0 - p_yes)) / NOUL_TEMPERATURE + NOUL_BIAS
    return 1.0 / (1.0 + math.exp(-z))


@dataclass
class _Question:
    question_id: str
    kind: str
    instructions: str
    options: list[tuple[str, str]]
    groups: list[range]
    # Token ids of the first readout of each group.
    group_ids: list[np.ndarray]


@dataclass
class OpenJevPlan:
    questions: dict
    state: OpenJevState
    items: list[_Question]
    pixel_values: np.ndarray | None = None
    image_grid_thw: np.ndarray | None = None
    image_tokens: int = 0


class OpenJevModel:
    """OpenJev and its MLX conversions."""

    def __init__(self, model_path: str, trust_remote_code: bool = False):
        self.model_path = model_path
        self.backbone = DecisionBackbone(model_path, trust_remote_code)
        self._letter_ids: list[int] = []
        self._vision_end_id: int | None = None

    def load(self) -> None:
        self.backbone.load()
        tokenizer = self.backbone.tokenizer
        letter_ids = [self.backbone.tokenize(letter)[0] for letter in LETTERS]
        if len(set(letter_ids)) != len(letter_ids):
            raise ValueError("option letters do not map to distinct tokens")
        self._letter_ids = letter_ids
        self._vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")

    def close(self) -> None:
        self.backbone.close()

    def encode(self, request: dict, truncate: bool = True) -> OpenJevPlan:
        """Validate, preprocess the image and tokenize the first readouts.

        OpenJev never truncates; ``truncate`` is accepted for API symmetry.
        """
        state = split_state(request["state"], request.get("images"))
        plan = OpenJevPlan(questions=request["questions"], state=state, items=[])
        if state.image is not None:
            if not self.backbone.has_vision:
                raise DecisionRequestError("this model has no vision tower")
            (image,) = decode_images([state.image])
            plan.pixel_values, plan.image_grid_thw = self.backbone.preprocess_images(
                [image]
            )
            plan.image_tokens = self.backbone.image_token_count(plan.image_grid_thw[0])

        limit = self.backbone.max_position_embeddings
        for question_id, question in request["questions"].items():
            kind = question["type"]
            instructions = instruction_text(question)
            criteria = question.get("criteria")
            if kind == "choice":
                if not isinstance(criteria, dict) or not criteria:
                    raise DecisionRequestError(
                        "choice criteria must be a non-empty map of option to "
                        "description or null"
                    )
                options = [(str(k), description_text(v)) for k, v in criteria.items()]
            elif kind == "score":
                if not isinstance(criteria, list) or len(criteria) < 2:
                    raise DecisionRequestError(
                        "score criteria must be an ordered array of at least two "
                        "levels"
                    )
                instructions += SCORE_SUFFIX
                options = [
                    (str(i), description_text(v)) for i, v in enumerate(criteria)
                ]
            else:
                if criteria is not None and not isinstance(criteria, dict):
                    raise DecisionRequestError("noul criteria must be an object")
                criteria = criteria or {}
                options = [
                    (
                        "yes",
                        description_text(criteria.get("true"))
                        or "The statement is true.",
                    ),
                    (
                        "no",
                        description_text(criteria.get("false"))
                        or "The statement is false.",
                    ),
                ]
            groups = option_groups(len(options))
            group_ids = [
                self._prompt_ids(plan, instructions, [options[i] for i in group])
                for group in groups
            ]
            for ids in group_ids:
                if limit is not None and len(ids) > limit:
                    raise DecisionContextLengthError(
                        f"the prompt needs {len(ids)} tokens; the limit is {limit}"
                    )
            plan.items.append(
                _Question(
                    str(question_id), kind, instructions, options, groups, group_ids
                )
            )
        return plan

    def _prompt_ids(
        self, plan: OpenJevPlan, instructions: str, options: list[tuple[str, str]]
    ) -> np.ndarray:
        text = render_prompt(plan.state, instructions, options)
        content: Any = text
        if plan.state.image is not None:
            content = [{"type": "image"}, {"type": "text", "text": text}]
        rendered = self.backbone.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if plan.state.image is not None:
            rendered = rendered.replace(
                "<|image_pad|>", "<|image_pad|>" * plan.image_tokens, 1
            )
        return np.asarray(self.backbone.tokenize(rendered), dtype=np.int32)

    def run(
        self, plan: OpenJevPlan, chunk_size: ChunkSize
    ) -> Generator[int, None, dict]:
        """Run every readout, yielding per prefill chunk, then build answers."""
        first = [ids for item in plan.items for ids in item.group_ids]
        shared = yield from self._prefill_shared_prefix(plan, first, chunk_size)
        answers = {}
        input_tokens = 0
        for item in plan.items:
            parts = []
            for group, ids in zip(item.groups, item.group_ids, strict=True):
                probs = yield from self._readout(
                    plan, ids, len(group), shared, chunk_size
                )
                parts.append(probs)
                input_tokens += len(ids)
            if len(parts) == 1:
                probs = parts[0]
            else:
                winners = [
                    item.options[group[int(np.argmax(p))]]
                    for group, p in zip(item.groups, parts, strict=True)
                ]
                ids = self._prompt_ids(plan, item.instructions, winners)
                final = yield from self._readout(
                    plan, ids, len(winners), shared, chunk_size
                )
                input_tokens += len(ids)
                probs = compose_groups(parts, final)
            answers[item.question_id] = self._answer(item, plan.questions, probs)
        return {"answers": answers, "input_tokens": input_tokens}

    def _prefill_shared_prefix(
        self, plan: OpenJevPlan, readouts: list[np.ndarray], chunk_size: ChunkSize
    ) -> Generator[int, None, tuple | None]:
        """Prefill the longest token prefix common to every first readout."""
        if len(readouts) < 2:
            return None
        length = min(len(ids) for ids in readouts) - 1
        reference = readouts[0]
        for ids in readouts[1:]:
            mismatch = np.nonzero(ids[:length] != reference[:length])[0]
            if mismatch.size:
                length = int(mismatch[0])
        if plan.state.image is not None:
            # The whole image must sit inside the prefix.
            ends = np.nonzero(reference[:length] == self._vision_end_id)[0]
            if not ends.size:
                return None
        if length <= 0:
            return None
        prefix = reference[:length]
        backbone = self.backbone
        embeds, positions = backbone.embed(
            prefix, plan.pixel_values, plan.image_grid_thw
        )
        cache = backbone.make_cache()
        yield from backbone.prefill(
            prefix, embeds, positions, cache, chunk_size, keep_all=False
        )
        last_position = None if positions is None else positions[:, :, -1]
        return prefix, cache, last_position

    def _readout(
        self,
        plan: OpenJevPlan,
        ids: np.ndarray,
        count: int,
        shared: tuple | None,
        chunk_size: ChunkSize,
    ) -> Generator[int, None, list[float]]:
        """Return the probabilities of the first ``count`` option letters."""
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
            embeds, positions = backbone.embed(
                ids, plan.pixel_values, plan.image_grid_thw
            )
        hidden = yield from backbone.prefill(
            tail, embeds, positions, cache, chunk_size, keep_all=False
        )
        logits = backbone.logits(hidden).astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits)
        scores = np.asarray(
            logprobs[mx.array(self._letter_ids[:count])].tolist(), dtype=np.float64
        )
        if not np.all(np.isfinite(scores)):
            raise RuntimeError("option letter scores are not finite")
        z = scores / READOUT_TEMPERATURE
        weights = np.exp(z - z.max())
        return (weights / weights.sum()).tolist()

    def _answer(self, item: _Question, questions: dict, probs: list[float]) -> dict:
        if item.kind == "choice":
            best = int(np.argmax(probs))
            return {
                "type": "choice",
                "choice": item.options[best][0],
                "probabilities": {
                    key: round4(p)
                    for (key, _), p in zip(item.options, probs, strict=True)
                },
                "confidence": round4(choice_confidence(probs)),
            }
        if item.kind == "score":
            levels = questions[item.question_id]["criteria"]
            return {
                "type": "score",
                "score": round4(sum(i * p for i, p in enumerate(probs))),
                "legend": {str(i): level for i, level in enumerate(levels)},
                "probabilities": {str(i): round4(p) for i, p in enumerate(probs)},
                "confidence": round4(score_confidence(probs)),
            }
        return {"type": "noul", "noul": round4(noul_probability(probs[0]))}
