# SPDX-License-Identifier: Apache-2.0
"""
LiquidAI d1 decision models on LFM3-VL and LFM3-Omni backbones.

mlx-vlm ships the d1 prompt, its set attention and the readout. d1 reads the
state and a group of questions in one forward pass: on full-attention layers
the state and each question read each other both ways. This wrapper runs that
pass in layer steps, as Decision 3.0 does, so other models get the GPU between
steps of a long state.
"""

import copy
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_vlm.models.d1_lfm3_vl.d1 import SystemOne, pack_set, set_masks
from mlx_vlm.models.d1_lfm3_vl.media import media_list, spoken
from mlx_vlm.models.d1_lfm3_vl.prompt import (
    prefix_text,
    question_header,
    state_text,
    suffix_layout,
)
from mlx_vlm.utils import load as vlm_load

from ..utils.model_loading import materialize_lazy_state, maybe_apply_pre_load_patches
from .decision import (
    ChunkSize,
    DecisionContextLengthError,
    DecisionRequestError,
    decode_images,
    round4,
)


def round_answer(answer: dict) -> dict:
    """Round an answer's probabilities to four places, as the other decision models do."""
    out = {}
    for key, value in answer.items():
        if key == "probabilities":
            value = {name: round4(p) for name, p in value.items()}
        elif isinstance(value, float):
            value = round4(value)
        out[key] = value
    return out


@dataclass
class D1Plan:
    """A validated request: its state, typed questions and decoded images."""

    questions: dict
    state: Any
    items: dict
    images: list


class D1Model:
    """A d1 checkpoint: mlx-vlm's d1 model and its System One engine."""

    has_vision = True

    def __init__(self, model_path: str, trust_remote_code: bool = False):
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.model: Any = None
        self.limit: int | None = None
        self._engine: SystemOne | None = None
        # encode() runs off the executor, so it uses its own tokenizer copy.
        self._requests: SystemOne | None = None

    def load(self) -> None:
        maybe_apply_pre_load_patches(self.model_path, for_vlm=True)
        model, processor = vlm_load(
            self.model_path, trust_remote_code=self.trust_remote_code
        )
        materialize_lazy_state(model)
        model.attach_processor(processor)
        self.model = model
        self._engine = model.engine
        self._requests = SystemOne(
            model, copy.deepcopy(processor), self._engine.calibration
        )
        self.limit = getattr(model.config.text_config, "max_position_embeddings", None)

    def close(self) -> None:
        self.model = None
        self._engine = None
        self._requests = None

    def encode(self, request: dict, truncate: bool = True) -> D1Plan:
        """Validate the questions and decode images on the CPU.

        d1 never truncates; ``truncate`` is accepted for API symmetry.
        """
        if request.get("videos"):
            raise DecisionRequestError("videos: d1 models read images only")
        requests = self._requests
        try:
            items = requests.named(request["questions"])
        except ValueError as exc:
            raise DecisionRequestError(str(exc)) from exc
        for name, question in items.items():
            if spoken(question) is not None:
                raise DecisionRequestError(
                    f"questions.{name}: instructions must be text, not media"
                )
        state = request["state"]
        state_tokens = len(
            requests.tokenizer.encode(
                state_text(state, requests.state_style), add_special_tokens=False
            )
        )
        if self.limit is not None and state_tokens > self.limit:
            raise DecisionContextLengthError(
                f"the state needs {state_tokens} tokens; the limit is {self.limit}"
            )
        return D1Plan(
            request["questions"], state, items, decode_images(request.get("images"))
        )

    def run(self, plan: D1Plan, chunk_size: ChunkSize) -> Generator[int, None, dict]:
        """Read each group of questions with the state in layer steps, then read out."""
        engine, model = self._engine, self.model
        tokenizer = engine.tokenizer
        markup, parts = (
            engine.media.encode(media_list(plan.images)) if plan.images else ("", [])
        )
        media = [model.d1_features(part) for part in parts]
        if media:
            mx.eval(media)
            yield 0
        prefix = prefix_text(
            tokenizer, plan.state, engine.bos, engine.state_style, engine.system, markup
        )
        header = question_header(tokenizer, plan.state)
        items = list(plan.items.values())
        layouts = [
            suffix_layout(tokenizer, question, engine.lead) for question in items
        ]
        probabilities: list = []
        input_tokens = 0
        for start in range(0, len(items), engine.set_size):
            group = slice(start, start + engine.set_size)
            suffixes = [text for text, _ in layouts[group]]
            ids, state_length, question, ends = pack_set(
                tokenizer,
                prefix,
                suffixes,
                header,
                [offsets for _, offsets in layouts[group]],
                tokenizer.encode(prefix + suffixes[0], add_special_tokens=False),
            )
            if self.limit is not None and len(ids) > self.limit:
                raise DecisionContextLengthError(
                    f"the request needs {len(ids)} tokens; the limit is {self.limit}"
                )
            hidden = yield from self._hidden(
                ids, state_length, question, media, chunk_size
            )
            probabilities += engine._read(items[group], ends, hidden, None)
            input_tokens += len(ids)
        # The expected level, as the other /v1/systemone models answer a score.
        answers = {
            name: round_answer(engine.answer(plan.state, item, probs, "mean"))
            for (name, item), probs in zip(
                plan.items.items(), probabilities, strict=True
            )
        }
        return {"answers": answers, "input_tokens": input_tokens}

    def _hidden(
        self,
        ids: list[int],
        state_length: int,
        question: list[int],
        media: list,
        chunk_size: ChunkSize,
    ) -> Generator[int, None, mx.array]:
        """``D1.d1_hidden`` from mlx-vlm, yielding after each layer step.

        One step runs the layers that ``chunk_size()`` tokens of prefill would
        cost and yields that token count.
        """
        backbone = self.model.language_model.model
        tokens = np.asarray(ids)
        hidden = backbone.embed_tokens(mx.array(tokens))
        for token in dict.fromkeys(token for token, _ in media):
            slots = np.flatnonzero(tokens == token)
            features = mx.concatenate([f for t, f in media if t == token])
            if len(slots) != features.shape[0]:
                raise ValueError(
                    f"placeholders {token} ({len(slots)}) and features "
                    f"({features.shape[0]}) do not match"
                )
            hidden[mx.array(slots)] = features.astype(hidden.dtype)
        length = len(ids)
        masks = set_masks(
            length, state_length, question, self.model.config.text_config.sliding_window
        )
        hidden = hidden[None]
        layers = backbone.layers
        depth = len(layers)
        done = 0
        while done < depth:
            end = min(depth, done + max(1, chunk_size() * depth // length))
            for layer in layers[done:end]:
                hidden = layer(hidden, masks[layer.layer_type], None)
            mx.eval(hidden)
            yield length * end // depth - length * done // depth
            done = end
        return backbone.embedding_norm(hidden)[0]
