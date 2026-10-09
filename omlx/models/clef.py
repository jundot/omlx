# SPDX-License-Identifier: Apache-2.0
"""
Cloudflare Clef decision model.

Clef reads one prompt that holds the state and every question with its
options, then scores all options at once with a joint schema head stored in
``joint_head.safetensors``. The head pools the backbone's hidden states over
the question and option spans. The prompt text, token layout and answer
format follow the Clef release because the head was trained on them.
"""

import json
import math
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..model_discovery import CLEF_HEAD_CONFIG, CLEF_HEAD_WEIGHTS
from .decision import (
    ChunkSize,
    DecisionBackbone,
    DecisionContextLengthError,
    DecisionRequestError,
    decode_images,
    round4,
)

SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
QUESTION_TYPE_IDS = {"noul": 0, "choice": 1, "score": 2}
NOUL_DEFAULT_CRITERIA = {
    "true": "The proposition is true or the answer is yes.",
    "false": "The proposition is false or the answer is no.",
}
MAX_LENGTH = 16384


def render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def question_options(question: dict) -> list[tuple[str, Any]]:
    """Return ``(option_id, description)`` pairs in prompt order."""
    criteria = question.get("criteria")
    kind = question["type"]
    if kind == "noul":
        if criteria is not None and not isinstance(criteria, dict):
            raise DecisionRequestError("noul criteria must be an object")
        merged = {**NOUL_DEFAULT_CRITERIA, **(criteria or {})}
        return [(key, merged[key]) for key in ("true", "false")]
    if kind == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise DecisionRequestError("choice criteria must be a non-empty object")
        return sorted((str(key), value) for key, value in criteria.items())
    if not isinstance(criteria, list) or not criteria:
        raise DecisionRequestError("score criteria must be a non-empty array")
    return [(str(index), value) for index, value in enumerate(criteria)]


@dataclass
class ClefPlan:
    """Token ids and question/option spans for one request."""

    input_ids: np.ndarray
    questions: dict
    question_ids: list[str]
    type_ids: list[int]
    question_spans: list[tuple[int, int]]
    option_spans: list[list[tuple[int, int]]]
    option_ids: list[list[str]]
    pixel_values: np.ndarray | None = None
    image_grid_thw: np.ndarray | None = None


def encode_record(
    tokenize,
    request: dict,
    image_tokens: list[int] | None = None,
    max_length: int = MAX_LENGTH,
    truncate: bool = True,
) -> tuple[list[int], list]:
    """Build the prompt token ids and the question/option spans.

    Each text piece is tokenized on its own, as in training. The state is
    cut to fit ``max_length`` unless ``truncate`` is False.
    """
    schema = tokenize("\n\nSCHEMA FIELDS:\n")
    fields = []
    for number, (question_id, question) in enumerate(request["questions"].items(), 1):
        schema += tokenize(
            f"\nFIELD {number}\nID: {question_id}\nTYPE: {question['type']}"
            "\nINSTRUCTION: "
        )
        question_start = len(schema)
        schema += tokenize(render(question.get("instructions") or str(question_id)))
        question_span = (question_start, len(schema))
        schema += tokenize("\nALLOWED OPTIONS:\n")
        option_spans, option_ids = [], []
        for option_number, (option_id, description) in enumerate(
            question_options(question), 1
        ):
            schema += tokenize(f"OPTION {option_number}: ")
            option_start = len(schema)
            semantics = {"option_id": option_id}
            if description is not None:
                semantics["description"] = description
            schema += tokenize(render(semantics))
            option_spans.append((option_start, len(schema)))
            option_ids.append(option_id)
            schema += tokenize("\n")
        schema += tokenize("END FIELD\n")
        fields.append((str(question_id), question_span, option_spans, option_ids))

    prefix = tokenize(
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
    )
    prefix += image_tokens or []
    suffix = tokenize(
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        "JOINT SCHEMA DECISIONS:"
    )
    state = tokenize(render(request["state"]))
    fixed = len(prefix) + len(schema) + len(suffix)
    if fixed > max_length:
        raise DecisionContextLengthError(
            f"the schema needs {fixed} tokens before the state; the limit is "
            f"{max_length}"
        )
    if not truncate and fixed + len(state) > max_length:
        raise DecisionContextLengthError(
            f"the request needs {fixed + len(state)} tokens; the limit is "
            f"{max_length}"
        )
    state = state[: max_length - fixed]
    offset = len(prefix) + len(state)
    shifted = [
        (
            question_id,
            (span[0] + offset, span[1] + offset),
            [(s + offset, e + offset) for s, e in option_spans],
            option_ids,
        )
        for question_id, span, option_spans, option_ids in fields
    ]
    return prefix + state + schema + suffix, shifted


class _PackedAttention(nn.Module):
    """``torch.nn.MultiheadAttention`` with its packed input projection."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.in_proj_weight = mx.zeros((3 * width, width))
        self.in_proj_bias = mx.zeros((3 * width,))
        self.out_proj = nn.Linear(width, width)

    def _split_heads(self, x: mx.array) -> mx.array:
        batch, length, width = x.shape
        x = x.reshape(batch, length, self.heads, width // self.heads)
        return x.transpose(0, 2, 1, 3)

    def __call__(self, queries: mx.array, keys: mx.array, values: mx.array):
        weights = mx.split(self.in_proj_weight, 3)
        biases = mx.split(self.in_proj_bias, 3)
        q, k, v = (
            self._split_heads(x @ w.T + b)
            for x, w, b in zip((queries, keys, values), weights, biases)
        )
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5)
        batch, _, length, _ = out.shape
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(batch, length, -1))


def _mlp(width: int, hidden: int, out: int) -> list[nn.Module]:
    # Same indices as the torch Sequential(Linear, GELU, Dropout, Linear) keys.
    return [nn.Linear(width, hidden), nn.GELU(), nn.Identity(), nn.Linear(hidden, out)]


def _run(layers: list[nn.Module], x: mx.array) -> mx.array:
    for layer in layers:
        x = layer(x)
    return x


class _EvidenceRoutingLayer(nn.Module):
    def __init__(self, width: int, heads: int, feedforward: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = _PackedAttention(width, heads)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = _mlp(width, feedforward, width)

    def __call__(self, queries: mx.array, memory: mx.array) -> mx.array:
        memory = self.memory_norm(memory)
        queries = queries + self.attention(self.query_norm(queries), memory, memory)
        return queries + _run(self.feedforward, self.feedforward_norm(queries))


class _DecoderLayer(nn.Module):
    """``torch.nn.TransformerDecoderLayer(norm_first=True, activation="gelu")``."""

    def __init__(self, width: int, heads: int, feedforward: int):
        super().__init__()
        self.self_attn = _PackedAttention(width, heads)
        self.multihead_attn = _PackedAttention(width, heads)
        self.linear1 = nn.Linear(width, feedforward)
        self.linear2 = nn.Linear(feedforward, width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.norm3 = nn.LayerNorm(width)

    def __call__(self, x: mx.array, memory: mx.array) -> mx.array:
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h)
        x = x + self.multihead_attn(self.norm2(x), memory, memory)
        return x + self.linear2(nn.gelu(self.linear1(self.norm3(x))))


def _unit(x: mx.array, eps: float = 1e-12) -> mx.array:
    return x / mx.maximum(mx.linalg.norm(x, axis=-1, keepdims=True), eps)


class JointSchemaHead(nn.Module):
    """Scores every option of every question from backbone hidden states."""

    def __init__(
        self,
        hidden_size: int,
        width: int,
        routing_layers: int,
        layers: int,
        heads: int,
        feedforward: int,
    ):
        super().__init__()
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.memory_projection = nn.Linear(hidden_size, width, bias=False)
        self.question_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_question_projection = nn.Linear(hidden_size, width, bias=False)
        self.global_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_context_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_lexical_projection = nn.Linear(hidden_size, width, bias=False)
        self.type_embedding = nn.Embedding(len(QUESTION_TYPE_IDS), width)
        self.evidence_layers = [
            _EvidenceRoutingLayer(width, heads, feedforward)
            for _ in range(routing_layers)
        ]
        self.option_summary_norm = nn.LayerNorm(width)
        self.layers = [_DecoderLayer(width, heads, feedforward) for _ in range(layers)]
        self.field_norm = nn.LayerNorm(width)
        self.option_norm = nn.LayerNorm(width)
        self.residual_scorer = _mlp(4 * width, width, 1)
        self.prior_logit_scale = mx.zeros(())
        self.joint_logit_scale = mx.zeros(())
        self.residual_gate = mx.zeros(())

    def __call__(
        self, hidden: mx.array, lexical: list[mx.array], plan: ClefPlan
    ) -> list[mx.array]:
        """Return the option logits of each question.

        ``hidden`` is ``(L, hidden_size)``. ``lexical`` holds, per question,
        the mean output-embedding row of each option span. Spans are pooled
        one at a time and each question is scored on its own, the same order
        of bf16 operations as the release implementation.
        """
        h = self.hidden_norm(hidden)
        memory = self.memory_projection(h)[None]
        global_vector = h[-1]
        questions = mx.stack([h[s:e].mean(0) for s, e in plan.question_spans])

        queries = [
            self.option_context_projection(mx.stack([h[s:e].mean(0) for s, e in spans]))
            + self.option_lexical_projection(rows)
            + self.option_question_projection(questions[i])[None]
            for i, (spans, rows) in enumerate(zip(plan.option_spans, lexical))
        ]
        routed = mx.concatenate(queries, axis=0)[None]
        for layer in self.evidence_layers:
            routed = layer(routed, memory)
        bounds = np.cumsum([len(spans) for spans in plan.option_spans])[:-1]
        options = mx.split(routed[0], bounds.tolist()) if len(bounds) else [routed[0]]

        base = self.question_projection(questions)
        scale = math.sqrt(routed.shape[-1])
        summaries = []
        for field, opts in zip(base, options):
            weights = mx.softmax((opts @ field) / scale, axis=0)
            summaries.append((weights[:, None] * opts).sum(0))
        fields = (
            base
            + self.option_summary_norm(mx.stack(summaries))
            + self.global_projection(global_vector)[None]
            + self.type_embedding(mx.array(plan.type_ids))
        )[None]
        for layer in self.layers:
            fields = layer(fields, memory)
        fields = self.field_norm(fields[0])

        cap = math.log(100.0)
        prior_scale = mx.exp(mx.minimum(self.prior_logit_scale, cap))
        joint_scale = mx.exp(mx.minimum(self.joint_logit_scale, cap))
        gate = mx.sigmoid(self.residual_gate)
        logits = []
        for i, (field, rows, opts) in enumerate(zip(fields, lexical, options)):
            anchor = _unit(questions[i] + global_vector)
            prior = prior_scale * (_unit(rows) @ anchor)
            opts = self.option_norm(opts)
            field = mx.broadcast_to(field[None], opts.shape)
            cosine = (field * opts).sum(-1) / mx.maximum(
                mx.linalg.norm(field, axis=-1) * mx.linalg.norm(opts, axis=-1), 1e-8
            )
            features = mx.concatenate(
                [field, opts, field * opts, mx.abs(field - opts)], axis=-1
            )
            residual = _run(self.residual_scorer, features)[:, 0]
            logits.append(prior + gate * (joint_scale * cosine + residual))
        return logits


def format_answer(question: dict, probabilities: dict[str, float]) -> dict:
    kind = question["type"]
    if kind == "noul":
        return {"type": "noul", "noul": round4(probabilities["true"])}
    if kind == "choice":
        keys = [str(key) for key in question["criteria"]]
        choice = max(keys, key=probabilities.__getitem__)
        return {
            "type": "choice",
            "choice": choice,
            "confidence": round4(probabilities[choice]),
            "probabilities": {key: round4(probabilities[key]) for key in keys},
        }
    levels = [str(index) for index in range(len(question["criteria"]))]
    return {
        "type": "score",
        "score": round4(
            sum(index * probabilities[level] for index, level in enumerate(levels))
        ),
        "confidence": round4(max(probabilities[level] for level in levels)),
        "legend": dict(zip(levels, question["criteria"], strict=True)),
        "probabilities": {level: round4(probabilities[level]) for level in levels},
    }


class ClefModel:
    """Clef or Clef-Flash: a Qwen3.5 backbone plus the joint schema head."""

    def __init__(self, model_path: str, trust_remote_code: bool = False):
        self.model_path = model_path
        self.backbone = DecisionBackbone(model_path, trust_remote_code)
        self.head: JointSchemaHead | None = None

    def load(self) -> None:
        self.backbone.load()
        path = Path(self.model_path)
        config = json.loads((path / CLEF_HEAD_CONFIG).read_text())
        head = JointSchemaHead(**config)
        head.load_weights(list(mx.load(str(path / CLEF_HEAD_WEIGHTS)).items()))
        head.set_dtype(mx.bfloat16)
        head.eval()
        mx.eval(head.parameters())
        self.head = head

    def close(self) -> None:
        self.backbone.close()
        self.head = None

    def encode(self, request: dict, truncate: bool = True) -> ClefPlan:
        """Tokenize and preprocess on the CPU; raises the request errors."""
        backbone = self.backbone
        images = decode_images(request.get("images"))
        pixel_values = image_grid_thw = None
        image_tokens: list[int] = []
        if images:
            if not backbone.has_vision:
                raise DecisionRequestError("this model has no vision tower")
            pixel_values, image_grid_thw = backbone.preprocess_images(images)
            text = "".join(
                "<|vision_start|>"
                + "<|image_pad|>" * backbone.image_token_count(grid)
                + "<|vision_end|>"
                for grid in image_grid_thw
            )
            image_tokens = backbone.tokenize(text + "\n")

        ids, fields = encode_record(
            backbone.tokenize, request, image_tokens, truncate=truncate
        )
        return ClefPlan(
            input_ids=np.asarray(ids, dtype=np.int32),
            questions=request["questions"],
            question_ids=[question_id for question_id, _, _, _ in fields],
            type_ids=[
                QUESTION_TYPE_IDS[q["type"]] for q in request["questions"].values()
            ],
            question_spans=[span for _, span, _, _ in fields],
            option_spans=[spans for _, _, spans, _ in fields],
            option_ids=[option_ids for _, _, _, option_ids in fields],
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

    def run(self, plan: ClefPlan, chunk_size: ChunkSize) -> Generator[int, None, dict]:
        """Prefill the prompt, yielding per chunk, then score every option."""
        backbone = self.backbone
        embeds, positions = backbone.embed(
            plan.input_ids, plan.pixel_values, plan.image_grid_thw
        )
        hidden = yield from backbone.prefill(
            plan.input_ids,
            embeds,
            positions,
            backbone.make_cache(),
            chunk_size,
            keep_all=True,
        )
        tokens = mx.array(plan.input_ids)
        lexical = [
            mx.stack([backbone.output_rows(tokens[s:e]).mean(0) for s, e in spans])
            for spans in plan.option_spans
        ]
        logits = self.head(hidden, lexical, plan)
        answers = {}
        for question_id, option_ids, question_logits in zip(
            plan.question_ids, plan.option_ids, logits, strict=True
        ):
            probabilities = mx.softmax(question_logits.astype(mx.float32)).tolist()
            answers[question_id] = format_answer(
                plan.questions[question_id],
                dict(zip(option_ids, probabilities, strict=True)),
            )
        return {"answers": answers, "input_tokens": len(plan.input_ids)}
