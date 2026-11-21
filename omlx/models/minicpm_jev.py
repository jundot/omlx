# SPDX-License-Identifier: Apache-2.0
"""
MiniCPM-OCR-Jev decision model for oMLX ``/v1/systemone``.

Scores binary OCR region verification inputs consisting of:
  image 1: original document crop
  state:   candidate OCR text
  image 2: rendered candidate OCR image

Uses the exact upstream training prompt (asserted via MD5), white-canvas
aspect-ratio padding past 199:1, Qwen3.5-VL causal backbone forward with
chunked ``ForwardFairnessGate`` yielding, and the 2-class ``score`` head
over the final-normed last prompt token (expected token ID 271).

Install unmarked original or converted checkpoints with this local oMLX
adapter identity in a ``decision_config.json`` beside ``config.json``::

    {"format_id": "minicpm-ocr-jev-v1",
     "prompt_template_md5": "9aab68e8ba8ab4da17ba9752bddb199c",
     "images_per_prompt": 2}

Prompt source: OpenBMB/MiniCPM-OCR, scripts/minicpm-ocr-jev/infer_vllm.py
(revision 74c8876056c536ee55b204cc842112c515764512).

Only add this marker after verifying the checkpoint uses the official binary
OCR head and exact prompt above. This metadata declares the adapter contract;
it does not modify model weights or upstream config. The streaming oQ writer
preserves ``decision_config.json``. With other conversion tools, copy the
identity sidecar explicitly and verify discovery again.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np
from PIL import Image

from ..model_discovery import (
    MINICPM_JEV_IMAGES_PER_PROMPT as IMAGES_PER_PROMPT,
)
from ..model_discovery import (
    MINICPM_JEV_PROMPT_TEMPLATE_MD5 as PROMPT_TEMPLATE_MD5,
)
from .decision import (
    ChunkSize,
    DecisionBackbone,
    DecisionContextLengthError,
    DecisionRequestError,
    decode_images,
    round4,
)

IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
IMAGE_TOKEN = "<|image_pad|>"
VIDEO_TOKEN = "<|video_pad|>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"

INSTRUCTION = (
    "第一张图是原图，第二张图是上述 OCR 文本渲染得到的图片。\n"
    "请判断该 OCR 文本及其对应渲染图是否与原图内容一致。\n"
    "这是一个二分类任务：一致标为 1，不一致标为 0。"
)
ASSISTANT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
EXPECTED_LAST_TOKEN = 271
MAX_ASPECT_RATIO = 199.0

_POSITIVE_CHOICE_KEYS = frozenset(
    {
        "1",
        "yes",
        "true",
        "match",
        "consistent",
        "correct",
        "faithful",
        "valid",
        "accurate",
        "same",
        "pass",
    }
)
_NEGATIVE_CHOICE_KEYS = frozenset(
    {
        "0",
        "no",
        "false",
        "mismatch",
        "inconsistent",
        "incorrect",
        "unfaithful",
        "invalid",
        "inaccurate",
        "different",
        "fail",
    }
)


def build_prompt(text: str) -> str:
    """Assemble the exact raw MiniCPM-OCR-Jev classifier prompt string."""
    return (
        f"<|im_start|>user\n"
        f"{IMAGE_PLACEHOLDER}\n"
        f"OCR文本：\n{text}\n"
        f"{IMAGE_PLACEHOLDER}\n"
        f"{INSTRUCTION}"
        f"{ASSISTANT_SUFFIX}"
    )


def _verify_prompt_fingerprint() -> None:
    digest = hashlib.md5(build_prompt("{text}").encode("utf-8")).hexdigest()
    if digest != PROMPT_TEMPLATE_MD5:
        raise RuntimeError(
            f"MiniCPM-OCR-Jev prompt MD5 mismatch: expected {PROMPT_TEMPLATE_MD5}, got {digest}"
        )


def pad_aspect_ratio(
    image: Image.Image, max_ratio: float = MAX_ASPECT_RATIO
) -> Image.Image:
    """Pad extreme-aspect-ratio crops with white rather than distorting scale."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    longest = max(width, height)
    shortest = max(1, min(width, height))
    if longest / shortest <= max_ratio:
        return rgb
    if width >= height:
        new_w = width
        new_h = max(height, int(width / max_ratio + 0.999999))
    else:
        new_h = height
        new_w = max(width, int(height / max_ratio + 0.999999))
    canvas = Image.new("RGB", (new_w, new_h), "white")
    canvas.paste(rgb, ((new_w - width) // 2, (new_h - height) // 2))
    return canvas


def extract_ocr_text(state: Any) -> str:
    """Extract candidate OCR text from ``request['state']`` without altering edge whitespace."""
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        for key in ("ocr_text", "candidate_text", "text"):
            val = state.get(key)
            if isinstance(val, str):
                return val
        raise DecisionRequestError(
            "MiniCPM-OCR-Jev state dict must contain 'ocr_text', 'candidate_text', or 'text'"
        )
    raise DecisionRequestError("MiniCPM-OCR-Jev state must be candidate OCR text")


def _classify_binary_choice(criteria: Any) -> tuple[str, str, list[str]]:
    """Map binary choice keys to ``(positive_key, negative_key, ordered_keys)``."""
    if not isinstance(criteria, dict) or len(criteria) != 2:
        raise DecisionRequestError(
            "MiniCPM-OCR-Jev choice questions require exactly two binary options"
        )
    keys = [str(k) for k in criteria]
    lower = [k.strip().lower() for k in keys]
    if lower[0] in _POSITIVE_CHOICE_KEYS and lower[1] in _NEGATIVE_CHOICE_KEYS:
        return keys[0], keys[1], keys
    if lower[1] in _POSITIVE_CHOICE_KEYS and lower[0] in _NEGATIVE_CHOICE_KEYS:
        return keys[1], keys[0], keys
    raise DecisionRequestError(
        "MiniCPM-OCR-Jev binary choice requires recognizable match/mismatch option keys "
        f"(got {keys!r})"
    )


@dataclass
class _JevQuestion:
    name: str
    qtype: str
    ordered_keys: list[str]
    positive_key: str | None = None
    negative_key: str | None = None


@dataclass
class MiniCPMJevPlan:
    """Validated MiniCPM-OCR-Jev request prepared off the GPU executor."""

    questions: dict
    items: list[_JevQuestion]
    ids: np.ndarray
    pixel_values: np.ndarray
    image_grids: np.ndarray


class MiniCPMJevModel:
    """MiniCPM-OCR-Jev sequence-classification OCR verifier for ``DecisionEngine``."""

    def __init__(self, model_path: str, trust_remote_code: bool = False):
        self.model_path = model_path
        self.backbone = DecisionBackbone(model_path, trust_remote_code)
        self._image_token_id: int = 0
        self._video_token_id: int = 0
        self._vision_start_token_id: int = 0
        self._vision_end_token_id: int = 0

    @property
    def has_vision(self) -> bool:
        return self.backbone.has_vision

    def load(self) -> None:
        _verify_prompt_fingerprint()
        self.backbone.load()
        if not self.backbone.has_vision:
            raise ValueError("MiniCPM-OCR-Jev requires its vision tower")
        if not hasattr(self.backbone.model, "score"):
            raise ValueError("MiniCPM-OCR-Jev model is missing its binary 'score' head")
        tokenizer = self.backbone.tokenizer
        tokens = (IMAGE_TOKEN, VIDEO_TOKEN, VISION_START_TOKEN, VISION_END_TOKEN)
        vocab = tokenizer.get_vocab()
        if any(token not in vocab for token in tokens):
            raise ValueError(
                "MiniCPM-OCR-Jev tokenizer is missing reserved vision tokens"
            )
        token_ids = [tokenizer.convert_tokens_to_ids(token) for token in tokens]
        if any(not isinstance(token_id, int) for token_id in token_ids) or len(
            set(token_ids)
        ) != len(tokens):
            raise ValueError(
                "MiniCPM-OCR-Jev tokenizer has invalid reserved vision token IDs"
            )
        (
            self._image_token_id,
            self._video_token_id,
            self._vision_start_token_id,
            self._vision_end_token_id,
        ) = token_ids
        try:
            self._validate_prompt_tokens(self.backbone.tokenize(build_prompt("")))
        except DecisionRequestError as exc:
            raise ValueError(
                f"MiniCPM-OCR-Jev tokenizer/prompt mismatch: {exc}"
            ) from exc

    def close(self) -> None:
        self.backbone.close()

    def _validate_prompt_tokens(self, token_ids: list[int]) -> None:
        if not token_ids or token_ids[-1] != EXPECTED_LAST_TOKEN:
            raise DecisionRequestError(
                f"MiniCPM-OCR-Jev prompt expected trailing token {EXPECTED_LAST_TOKEN}, "
                f"got {token_ids[-1] if token_ids else None}"
            )
        if token_ids.count(self._image_token_id) != IMAGES_PER_PROMPT:
            raise DecisionRequestError(
                "Candidate OCR text must yield exactly 2 image placeholder tokens"
            )
        if token_ids.count(self._video_token_id) or any(
            token_ids.count(token_id) != IMAGES_PER_PROMPT
            for token_id in (self._vision_start_token_id, self._vision_end_token_id)
        ):
            raise DecisionRequestError(
                "Candidate OCR text cannot contain reserved video or vision boundary tokens"
            )

    def encode(self, request: dict, truncate: bool = True) -> MiniCPMJevPlan:
        """Validate ternary OCR input, pad extreme aspect ratios, and tokenize prompt."""
        if request.get("videos"):
            raise DecisionRequestError("MiniCPM-OCR-Jev does not take videos")
        if "_text_parts" in request and len(request["_text_parts"]) != 1:
            raise DecisionRequestError(
                "MiniCPM-OCR-Jev requires exactly one input_text part on /v1/decisions; "
                "an empty string is allowed"
            )

        images_raw = request.get("images") or []
        if len(images_raw) != IMAGES_PER_PROMPT:
            raise DecisionRequestError(
                "MiniCPM-OCR-Jev requires exactly 2 images: [original, render]"
            )
        decoded = decode_images(images_raw)
        padded = [pad_aspect_ratio(img) for img in decoded]

        ocr_text = extract_ocr_text(request["state"])
        raw_prompt = build_prompt(ocr_text)
        if raw_prompt.count(IMAGE_PLACEHOLDER) != IMAGES_PER_PROMPT:
            raise DecisionRequestError(
                "Candidate OCR text cannot contain image placeholder tokens"
            )

        items: list[_JevQuestion] = []
        for qname, q in request["questions"].items():
            qtype = q["type"]
            criteria = q.get("criteria")
            if qtype == "noul":
                if criteria is not None and not isinstance(criteria, dict):
                    raise DecisionRequestError("noul criteria must be an object")
                items.append(
                    _JevQuestion(name=str(qname), qtype="noul", ordered_keys=[])
                )
            elif qtype == "score":
                if not isinstance(criteria, list) or len(criteria) != 2:
                    raise DecisionRequestError(
                        "MiniCPM-OCR-Jev score questions require exactly 2 levels"
                    )
                items.append(
                    _JevQuestion(
                        name=str(qname), qtype="score", ordered_keys=["0", "1"]
                    )
                )
            elif qtype == "choice":
                pos_key, neg_key, ordered = _classify_binary_choice(criteria)
                items.append(
                    _JevQuestion(
                        name=str(qname),
                        qtype="choice",
                        ordered_keys=ordered,
                        positive_key=pos_key,
                        negative_key=neg_key,
                    )
                )
            else:
                raise DecisionRequestError(f"unsupported question type {qtype!r}")

        backbone = self.backbone
        token_ids = backbone.tokenize(raw_prompt)
        self._validate_prompt_tokens(token_ids)
        try:
            pixel_values, image_grids = backbone.preprocess_images(padded)
        except ValueError as exc:
            raise DecisionRequestError(f"images: {exc}") from exc

        merge = backbone.image_processor.merge_size
        counts = [int(np.prod(grid)) // merge**2 for grid in image_grids]
        counts_iter = iter(counts)
        expanded: list[int] = []
        for tok in token_ids:
            if tok == self._image_token_id:
                expanded.extend([tok] * next(counts_iter))
            else:
                expanded.append(tok)

        limit = backbone.max_position_embeddings
        if limit is not None and len(expanded) > limit:
            raise DecisionContextLengthError(
                f"prompt requires {len(expanded)} tokens; limit is {limit}"
            )

        return MiniCPMJevPlan(
            questions=request["questions"],
            items=items,
            ids=np.asarray(expanded, dtype=np.int32),
            pixel_values=pixel_values,
            image_grids=image_grids,
        )

    def run(
        self, plan: MiniCPMJevPlan, chunk_size: ChunkSize
    ) -> Generator[int, None, dict]:
        """Run vision embeddings and chunked causal forward pass, then score head."""
        backbone = self.backbone
        embeds, positions = backbone.embed(
            plan.ids, plan.pixel_values, plan.image_grids
        )
        mx.eval(embeds, positions)
        yield 0

        hidden = yield from backbone.encode(embeds, positions, chunk_size, causal=True)
        logits = backbone.model.score(hidden[None])[0]
        mx.eval(logits)
        logits_f32 = logits.astype(mx.float32)
        if not bool(mx.all(mx.isfinite(logits_f32)).item()):
            raise RuntimeError("MiniCPM-OCR-Jev produced non-finite score logits")
        probs = mx.softmax(logits_f32).tolist()
        p_no, p_yes = float(probs[0]), float(probs[1])
        if not (math.isfinite(p_no) and math.isfinite(p_yes)):
            raise RuntimeError("MiniCPM-OCR-Jev produced non-finite probabilities")

        conf = round4(abs(p_yes - p_no))
        answers: dict[str, dict[str, Any]] = {}
        for item in plan.items:
            if item.qtype == "noul":
                answers[item.name] = {
                    "type": "noul",
                    "noul": round4(p_yes),
                }
            elif item.qtype == "score":
                answers[item.name] = {
                    "type": "score",
                    "score": round4(p_yes),
                    "legend": {
                        str(index): level
                        for index, level in enumerate(
                            plan.questions[item.name]["criteria"]
                        )
                    },
                    "confidence": conf,
                    "probabilities": {"0": round4(p_no), "1": round4(p_yes)},
                }
            else:
                pmap_raw = {item.positive_key: p_yes, item.negative_key: p_no}
                pmap = {k: round4(pmap_raw[k]) for k in item.ordered_keys}
                best = item.positive_key if p_yes >= p_no else item.negative_key
                answers[item.name] = {
                    "type": "choice",
                    "choice": best,
                    "confidence": conf,
                    "probabilities": pmap,
                }
        return {"answers": answers, "input_tokens": int(len(plan.ids))}
