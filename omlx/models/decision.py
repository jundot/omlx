# SPDX-License-Identifier: Apache-2.0
"""
Shared runtime for decision models served through ``/v1/systemone``.

A decision model reads a prompt with a Qwen3.5 backbone and scores the
options of typed questions without generating text. This module loads the
backbone (mlx-vlm when the checkpoint has a vision tower, mlx-lm otherwise)
and runs causal prefill in chunks, so the engine can give the GPU to other
models between chunks.
"""

import json
import logging
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm.utils import load as vlm_load

from ..model_discovery import _vision_weight_bytes
from ..utils.image import load_image
from ..utils.model_loading import (
    lm_load_compat,
    materialize_lazy_state,
    maybe_apply_pre_load_patches,
)

logger = logging.getLogger(__name__)

ChunkSize = Callable[[], int]


class DecisionRequestError(ValueError):
    """The request breaks a rule of the model's question format."""


class DecisionContextLengthError(ValueError):
    """The prompt does not fit the model's context limit."""


def round4(value: float) -> float:
    return round(float(value), 4)


def decode_images(urls: list[str] | None) -> list:
    return [load_image(url, field="images") for url in urls or []]


class DecisionBackbone:
    """Qwen3.5 backbone that returns final normed hidden states."""

    def __init__(self, model_path: str, trust_remote_code: bool = False):
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.model: Any = None
        self.tokenizer: Any = None
        self.image_processor: Any = None
        self.config: dict = {}
        self._text_model: Any = None
        self._lm_head: Any = None

    @property
    def has_vision(self) -> bool:
        return self.image_processor is not None

    @property
    def max_position_embeddings(self) -> int | None:
        text_config = self.config.get("text_config") or self.config
        value = text_config.get("max_position_embeddings")
        return int(value) if value else None

    def load(self) -> None:
        path = Path(self.model_path)
        self.config = json.loads((path / "config.json").read_text())
        if _vision_weight_bytes(path) > 0:
            # Also installs the Qwen3.5 GatedDeltaNet q/k norm fix for mlx-vlm.
            maybe_apply_pre_load_patches(self.model_path, for_vlm=True)
            model, processor = vlm_load(
                self.model_path, trust_remote_code=self.trust_remote_code
            )
            tokenizer = processor.tokenizer
            self.image_processor = processor.image_processor
        else:
            maybe_apply_pre_load_patches(self.model_path)
            model, wrapper = lm_load_compat(
                self.model_path, trust_remote_code=self.trust_remote_code
            )
            tokenizer = wrapper._tokenizer
        materialize_lazy_state(model)
        self.model = model
        self.tokenizer = tokenizer
        language_model = model.language_model
        self._text_model = language_model.model
        self._lm_head = getattr(language_model, "lm_head", None)
        if self._lm_head is None:
            self._lm_head = self._text_model.embed_tokens

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self._text_model = None
        self._lm_head = None

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def preprocess_images(self, images: list) -> tuple[np.ndarray, np.ndarray]:
        """Return pixel values and ``(t, h, w)`` patch grids for PIL images."""
        out = self.image_processor(images=images)
        return np.asarray(out["pixel_values"]), np.asarray(out["image_grid_thw"])

    def image_token_count(self, grid: np.ndarray) -> int:
        return int(np.prod(grid)) // self.image_processor.merge_size**2

    def make_cache(self) -> list:
        return self.model.language_model.make_cache()

    def embed(
        self,
        ids: np.ndarray,
        pixel_values: np.ndarray | None = None,
        image_grid_thw: np.ndarray | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        """Return input embeddings and M-RoPE positions for a full prompt.

        Positions are None on the text-only backbone, which derives them from
        the cache offset.
        """
        tokens = mx.array(ids)[None]
        if not self.has_vision:
            return self._text_model.embed_tokens(tokens), None
        features = self.model.get_input_embeddings(
            tokens,
            pixel_values=None if pixel_values is None else mx.array(pixel_values),
            image_grid_thw=(
                None if image_grid_thw is None else mx.array(image_grid_thw)
            ),
        )
        positions = features.position_ids
        if positions.ndim == 2:
            positions = mx.broadcast_to(positions[None], (3, *positions.shape))
        return features.inputs_embeds, positions

    def embed_continuation(
        self, ids: np.ndarray, last_position: mx.array | None
    ) -> tuple[mx.array, mx.array | None]:
        """Embed text that follows a prefilled prefix ending at ``last_position``."""
        embeds = self._text_model.embed_tokens(mx.array(ids)[None])
        if last_position is None:
            return embeds, None
        steps = mx.arange(1, len(ids) + 1, dtype=last_position.dtype)
        positions = last_position[:, :, None] + steps[None, None, :]
        return embeds, positions

    def prefill(
        self,
        ids: np.ndarray,
        embeds: mx.array,
        positions: mx.array | None,
        cache: list,
        chunk_size: ChunkSize,
        keep_all: bool,
    ) -> Generator[int, None, mx.array]:
        """Run causal prefill chunk by chunk and yield the tokens done per chunk.

        Returns the hidden states of every token when ``keep_all`` is set,
        otherwise only the last token's hidden state.
        """
        tokens = mx.array(ids)[None]
        total = len(ids)
        kept = []
        start = 0
        while start < total:
            end = min(total, start + max(1, chunk_size()))
            hidden = self._forward(
                tokens[:, start:end],
                embeds[:, start:end],
                None if positions is None else positions[..., start:end],
                cache,
            )
            kept.append(hidden[0] if keep_all else hidden[0, -1:])
            mx.eval(kept[-1], [c.state for c in cache])
            if not keep_all:
                kept = kept[-1:]
            yield end - start
            start = end
        return mx.concatenate(kept, axis=0) if keep_all else kept[-1][0]

    def _forward(
        self,
        tokens: mx.array,
        embeds: mx.array,
        positions: mx.array | None,
        cache: list,
    ) -> mx.array:
        # Call the inner text model: the outer LanguageModel keeps request
        # state between calls and is wrapped by the MTP runtime patch.
        if self.has_vision:
            return self._text_model(
                tokens, inputs_embeds=embeds, cache=cache, position_ids=positions
            )
        hidden = self._text_model(tokens, cache=cache, input_embeddings=embeds)
        # The oMLX MTP patch makes the mlx-lm text model return pre-norm hidden
        # states and leaves the final norm to TextModel. It patches the class,
        # so it can be installed after this model was loaded.
        if getattr(type(self._text_model).__call__, "_omlx_mtp_call_marker", False):
            hidden = self._text_model.norm(hidden)
        return hidden

    def logits(self, hidden: mx.array) -> mx.array:
        if isinstance(self._lm_head, nn.Embedding | nn.QuantizedEmbedding):
            return self._lm_head.as_linear(hidden)
        return self._lm_head(hidden)

    def output_rows(self, ids: mx.array) -> mx.array:
        """Rows of the output projection for ``ids``, dequantized if needed."""
        head = self._lm_head
        if isinstance(head, nn.QuantizedLinear | nn.QuantizedEmbedding):
            biases = head.get("biases")
            return mx.dequantize(
                head.weight[ids],
                head.scales[ids],
                None if biases is None else biases[ids],
                group_size=head.group_size,
                bits=head.bits,
                mode=getattr(head, "mode", "affine"),
            )
        return head.weight[ids]
