# SPDX-License-Identifier: Apache-2.0
#
# Prompt text and answer labels follow OmniJev/OneJev at 81ce62f (Apache-2.0)
# and the serve.patch published with baselabs/mammoth-1 and elephant-1 (MIT).
"""Native Mammoth-1 / Elephant-1 decisions using the published OneJev format.

The state is one bidirectional block and every question is a separate second
block. Questions share the state's cache, never each other's tokens. Scores
are the FP32 LM-head rows of the release's answer labels, without decoding.
"""

import bisect
import copy
import json
import math
import re
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from ..model_discovery import BASELABS_CONTEXT_LENGTH
from .decision import (
    ChunkSize,
    DecisionBackbone,
    DecisionContextLengthError,
    DecisionRequestError,
    decode_images,
    round4,
)
from .openjev import choice_confidence, score_confidence

SYSTEM_PROMPT = (
    "Apply the question to the state. Choose exactly one of the listed options. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_SPLIT_LABELS = {
    "BQ",
    "BZ",
    "CJ",
    "CQ",
    "CZ",
    "DQ",
    "DZ",
    "EJ",
    "EY",
    "FJ",
    "FQ",
    "FV",
    "FZ",
    "GJ",
    "GK",
    "GQ",
    "GZ",
    "HJ",
    "IY",
    "JF",
    "JG",
    "JH",
    "JL",
    "JN",
    "JQ",
}
SLOT_LABELS = (
    tuple(_LETTERS)
    + tuple(a + b for a in _LETTERS for b in _LETTERS if a + b not in _SPLIT_LABELS)[
        :230
    ]
)
MAX_BRANCH_TOKENS = BASELABS_CONTEXT_LENGTH
MAX_REQUEST_TOKENS = 65_536
_IMAGE_PLACEHOLDER = re.compile(r"<image:(\d+)>")


def render_entry(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, (dict, list)):
        raise DecisionRequestError(
            "instructions and criteria entries must be text, objects, arrays or null"
        )
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise DecisionRequestError(
            "instructions and criteria must contain finite JSON values"
        ) from error


@dataclass
class _Question:
    question_id: str
    kind: str
    labels: list[str]
    levels: list | None
    suffix: str


def normalize_question(question_id: str, question: dict) -> _Question:
    kind = question.get("type")
    criteria = question.get("criteria")
    instructions = render_entry(question.get("instructions"))
    levels = None
    if kind == "noul":
        if criteria is not None and not isinstance(criteria, dict):
            raise DecisionRequestError("noul criteria must be an object or null")
        criteria = criteria or {}
        labels = names = ["yes", "no"]
        descriptions = [
            (
                "the statement is true / the answer is yes"
                if criteria.get("true") is None
                else render_entry(criteria["true"])
            ),
            (
                "the statement is false / the answer is no"
                if criteria.get("false") is None
                else render_entry(criteria["false"])
            ),
        ]
        instructions = (
            instructions
            or "Is the statement true, or is the answer to the question yes?"
        )
    elif kind == "choice":
        if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255:
            raise DecisionRequestError("choice criteria must contain 1 to 255 options")
        labels = names = list(criteria)
        descriptions = [render_entry(value) for value in criteria.values()]
        instructions = instructions or "Which option applies to the state?"
    elif kind == "score":
        if not isinstance(criteria, list) or not 1 <= len(criteria) <= 10:
            raise DecisionRequestError(
                "score criteria must contain 1 to 10 ordered levels"
            )
        levels = criteria
        labels = [str(i) for i in range(len(criteria))]
        names = [f"level {i}" for i in range(len(criteria))]
        descriptions = [render_entry(value) for value in criteria]
        instructions = instructions or "Which level describes the state?"
    else:
        raise DecisionRequestError("question type must be noul, choice or score")
    lines = [
        f"{letter}. {name}" + (f": {description}" if description else "")
        for letter, name, description in zip(SLOT_LABELS, names, descriptions)
    ]
    header = "Rate the state:" if kind == "score" else "Question:"
    answer = "letter" if len(labels) <= 26 else "label"
    suffix = (
        f"{header} {instructions}\n\nOptions:\n"
        + "\n".join(lines)
        + f"\n\nAnswer with one {answer}: {', '.join(SLOT_LABELS[:len(labels)])}."
    )
    return _Question(question_id, kind, labels, levels, suffix)


def format_answer(question: _Question, probabilities: list[float]) -> dict:
    if question.kind == "noul":
        return {"type": "noul", "noul": round4(probabilities[0])}
    answer = {
        "type": question.kind,
        "probabilities": {
            key: round4(p) for key, p in zip(question.labels, probabilities)
        },
    }
    if question.kind == "choice":
        answer.update(
            choice=question.labels[int(np.argmax(probabilities))],
            confidence=round4(choice_confidence(probabilities)),
        )
    else:
        answer.update(
            score=round4(sum(i * p for i, p in enumerate(probabilities))),
            legend={str(i): value for i, value in enumerate(question.levels)},
            confidence=round4(score_confidence(probabilities)),
        )
    return answer


@dataclass
class BaseLabsPlan:
    questions: dict
    items: list[_Question]
    prefix_ids: np.ndarray
    suffix_ids: list[np.ndarray]
    pixel_values: np.ndarray | None = None
    image_grid_thw: np.ndarray | None = None


class BaseLabsModel:
    """Mammoth-1, Elephant-1 and conversions retaining their decision metadata."""

    def __init__(self, model_path: str, trust_remote_code: bool = False):
        self.model_path = model_path
        self.backbone = DecisionBackbone(model_path, trust_remote_code)
        self._slot_weights: mx.array | None = None
        self._temperatures: dict[str, float] = {}

    def load(self) -> None:
        path = Path(self.model_path)
        config = json.loads((path / "config.json").read_text())
        overrides = config.get("modeling_overrides") or {}
        readout = json.loads((path / "readout.json").read_text())
        if overrides.get("attention") != {
            "full_attention": "block_bidirectional",
            "linear_attention": "causal",
        }:
            raise ValueError(
                "Base Labs decisions require block-bidirectional full attention and causal linear attention"
            )
        for settings in (overrides, readout):
            if settings.get("readout") != "slot_lm" or settings.get("n_slots") != 256:
                raise ValueError(
                    "Base Labs decisions require the released slot_lm readout with 256 slots"
                )
        if readout.get("attention", "noncausal") != "noncausal":
            raise ValueError(
                "readout.json conflicts with the block-bidirectional attention config"
            )
        calibration = path / "calibration.json"
        if calibration.is_file():
            temperatures = json.loads(calibration.read_text()).get("temperatures", {})
            self._temperatures = {
                key: float(value) for key, value in temperatures.items()
            }
            if any(
                not math.isfinite(value) or value <= 0
                for value in self._temperatures.values()
            ):
                raise ValueError(
                    "decision calibration temperatures must be finite and positive"
                )
        self.backbone.load()
        try:
            ids = []
            for label in SLOT_LABELS:
                encoded = self.backbone.tokenize(label)
                if (
                    len(encoded) != 1
                    or self.backbone.tokenizer.decode(encoded) != label
                ):
                    raise ValueError(
                        f"answer label {label!r} must be a single round-trip token"
                    )
                ids.append(encoded[0])
            if len(set(ids)) != len(ids):
                raise ValueError("answer label tokens must be distinct")
            self._slot_weights = self.backbone.output_rows(mx.array(ids)).astype(
                mx.float32
            )
            mx.eval(self._slot_weights)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._slot_weights = None
        self._temperatures.clear()
        self.backbone.close()

    def _messages(self, block: str, suffix: str, image_count: int) -> list[dict]:
        content: str | list = block + suffix
        if image_count:
            parts, seen, start = [], [], 0
            for match in _IMAGE_PLACEHOLDER.finditer(block):
                parts.append({"type": "text", "text": block[start : match.start()]})
                parts.append({"type": "image"})
                seen.append(int(match[1]))
                start = match.end()
            if seen != list(range(1, image_count + 1)):
                raise DecisionRequestError(
                    "reference each image once, in order, as <image:1>, <image:2>, ... in state"
                )
            parts.append({"type": "text", "text": block[start:] + suffix})
            content = parts
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def encode(self, request: dict, truncate: bool = True) -> BaseLabsPlan:
        """Use the release prompt and block boundary; never truncate evidence."""
        if request.get("videos"):
            raise DecisionRequestError("Mammoth-1 and Elephant-1 do not take videos")
        urls = request.get("images") or []
        if urls and not self.backbone.has_vision:
            raise DecisionRequestError(
                "this text-only MLX conversion has no vision tower"
            )
        images = decode_images(urls)
        state = request["state"]
        try:
            text = (
                state
                if isinstance(state, str)
                else json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False)
            )
        except (TypeError, ValueError) as error:
            raise DecisionRequestError(
                "state must contain finite JSON values"
            ) from error
        block = f"<state>\n{text}\n</state>\n\n"
        items = [
            normalize_question(key, question)
            for key, question in request["questions"].items()
        ]
        if images and any(_IMAGE_PLACEHOLDER.search(item.suffix) for item in items):
            raise DecisionRequestError(
                "image placeholders belong in state, not questions"
            )
        prefix_ids, suffix_ids = None, []
        tokenizer = self.backbone.tokenizer
        for item in items:
            prompt = tokenizer.apply_chat_template(
                self._messages(block, item.suffix, len(images)),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            encoded = tokenizer(
                prompt, add_special_tokens=False, return_offsets_mapping=True
            )
            # Offset-based boundaries match the patched OneJev readout layout.
            boundary = bisect.bisect_right(
                [end for _, end in encoded["offset_mapping"]],
                prompt.rindex(item.suffix),
            )
            ids = encoded["input_ids"]
            prefix, suffix = ids[:boundary], ids[boundary:]
            if prefix_ids is None:
                prefix_ids = prefix
            if prefix != prefix_ids or not prefix or not suffix:
                raise DecisionRequestError(
                    "the chat template does not preserve the state/question block boundary"
                )
            suffix_ids.append(np.asarray(suffix, dtype=np.int32))
        pixels = grid = None
        if images:
            pixels, grid = self.backbone.preprocess_images(images)
            image_id = self.backbone.config["image_token_id"]
            counts = iter(self.backbone.image_token_count(entry) for entry in grid)
            expanded = []
            try:
                for token in prefix_ids:
                    expanded.extend(
                        [token] * next(counts) if token == image_id else [token]
                    )
            except StopIteration as error:
                raise DecisionRequestError(
                    "image placeholders do not match the image inputs"
                ) from error
            if next(counts, None) is not None:
                raise DecisionRequestError(
                    "image placeholders do not match the image inputs"
                )
            prefix_ids = expanded
        limit = min(
            MAX_BRANCH_TOKENS,
            self.backbone.max_position_embeddings or MAX_BRANCH_TOKENS,
        )
        longest = len(prefix_ids) + max(map(len, suffix_ids))
        total = len(prefix_ids) + sum(map(len, suffix_ids))
        if longest > limit or total > MAX_REQUEST_TOKENS:
            raise DecisionContextLengthError(
                f"decision request exceeds the {limit}-token branch or {MAX_REQUEST_TOKENS}-token request limit"
            )
        return BaseLabsPlan(
            request["questions"],
            items,
            np.asarray(prefix_ids, dtype=np.int32),
            suffix_ids,
            pixels,
            grid,
        )

    def _prefill_block(
        self,
        embeds: mx.array,
        positions: mx.array | None,
        cache: list,
        chunk_size: ChunkSize,
    ) -> Generator[int, None, mx.array]:
        """Read one bidirectional block, attending to its cached predecessors.

        A block cannot be split across tokens: every full-attention query must
        see its whole block. Yield between groups of layers instead. Gated
        DeltaNet retains its causal recurrence and its own cache state.
        """
        backbone = self.backbone
        layers = backbone._text_model.layers
        depth, length = len(layers), embeds.shape[1]
        positional = {}
        if backbone.has_vision:
            rotary = next(
                layer.self_attn.rotary_emb for layer in layers if not layer.is_linear
            )
            positional = {
                "position_ids": positions,
                "position_embeddings": (
                    None if rotary.fused_apply else rotary(embeds, positions)
                ),
            }
        hidden = embeds
        done = 0
        while done < depth:
            end = min(depth, done + max(1, chunk_size() * depth // length))
            for index in range(done, end):
                hidden = layers[index](
                    hidden, mask=None, cache=cache[index], **positional
                )
            mx.eval(hidden, [entry.state for entry in cache[done:end]])
            yield length * end // depth - length * done // depth
            done = end
        return backbone._text_model.norm(hidden[0, -1])

    def run(
        self, plan: BaseLabsPlan, chunk_size: ChunkSize
    ) -> Generator[int, None, dict]:
        if self._slot_weights is None:
            raise RuntimeError("decision readout is not loaded")
        backbone = self.backbone
        embeds, positions = backbone.embed(
            plan.prefix_ids, plan.pixel_values, plan.image_grid_thw
        )
        prefix_cache = backbone.make_cache()
        yield from self._prefill_block(embeds, positions, prefix_cache, chunk_size)
        last_position = None if positions is None else positions[:, :, -1]
        answers = {}
        for item, suffix in zip(plan.items, plan.suffix_ids):
            cache = copy.deepcopy(prefix_cache)
            embeds, positions = backbone.embed_continuation(suffix, last_position)
            hidden = yield from self._prefill_block(
                embeds, positions, cache, chunk_size
            )
            logits = (
                self._slot_weights[: len(item.labels)] @ hidden.astype(mx.float32)
            ).tolist()
            temperature = self._temperatures.get(
                f"{item.kind}:{len(item.labels)}",
                self._temperatures.get(item.kind, 1.0),
            )
            scaled = np.asarray(logits, dtype=np.float64) / temperature
            probabilities = np.exp(scaled - scaled.max())
            probabilities /= probabilities.sum()
            answers[item.question_id] = format_answer(item, probabilities.tolist())
            del cache
        return {
            "answers": answers,
            "input_tokens": len(plan.prefix_ids) + sum(map(len, plan.suffix_ids)),
        }
