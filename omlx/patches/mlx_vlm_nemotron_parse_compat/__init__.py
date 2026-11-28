# SPDX-License-Identifier: Apache-2.0
"""Give Nemotron 3.5 Parse prompts a placeholder per visual-prefix embedding.

mlx-vlm's ``nemotron_parse_v3_5`` model prepends the vision prefix to the text
embeddings, so ``get_input_embeddings`` returns more rows than the prompt has
tokens. The generic VLM engine assumes every image feature has a placeholder in
the prompt: it reports the prompt length as ``prompt_tokens``, keys the prefix
cache on prompt tokens, and matches cached vision features against the image
token count. With one image and a six-token prompt the two disagree, and the
request degenerates (#5008).

The hooks make the lengths agree:

- the processor prepends one ``<pad>`` placeholder per visual-prefix embedding,
  so the prompt carries the image features;
- ``get_input_embeddings`` drops those placeholders again and puts the vision
  prefix in their place;
- the config exposes ``image_token_id`` so the engine can find them.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

_APPLIED = False


def apply_mlx_vlm_nemotron_parse_compat_patch() -> bool:
    """Install the placeholder hooks on mlx-vlm's nemotron_parse_v3_5."""
    global _APPLIED
    if _APPLIED:
        return False

    try:
        from mlx_vlm.models.nemotron_parse_v3_5 import config as np_config
        from mlx_vlm.models.nemotron_parse_v3_5 import (
            nemotron_parse_v3_5 as np_model,
        )
        from mlx_vlm.models.nemotron_parse_v3_5 import (
            processing_nemotron_parse_v3_5 as np_processing,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("mlx-vlm nemotron_parse_v3_5 not importable: %s", exc)
        return False

    # <pad> is the family's placeholder token; the checkpoint's tokenizer maps it
    # to the same id, and the engine reads ``image_token_id`` off this config.
    placeholder_id = int(np_config.ModelConfig.pad_token_id)
    np_config.ModelConfig.image_token_id = placeholder_id
    _patch_get_input_embeddings(np_model)
    _patch_processor_placeholders(np_processing, placeholder_id)

    _APPLIED = True
    logger.info("Nemotron 3.5 Parse mlx-vlm compatibility patch applied")
    return True


def is_applied() -> bool:
    return _APPLIED


def _patch_get_input_embeddings(np_model: Any) -> None:
    model_cls = np_model.Model
    if getattr(model_cls.get_input_embeddings, "_omlx_np_compat", False):
        return

    features_cls = np_model.InputEmbeddingsFeatures

    def get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        if input_ids is None or pixel_values is None:
            raise ValueError("Nemotron 3.5 requires prompt input_ids and pixel_values")
        vision = kwargs.get("cached_image_features")
        if vision is None:
            vision = self.vision_tower(
                pixel_values, kwargs.get("image_sizes"), kwargs.get("content_sizes")
            )
        # Drop the leading placeholders the processor reserved for the prefix.
        count = vision.shape[1]
        placeholder = self.config.pad_token_id
        if input_ids.shape[1] > count and bool(
            mx.all(input_ids[:, :count] == placeholder)
        ):
            input_ids = input_ids[:, count:]
        return features_cls(
            inputs_embeds=mx.concatenate(
                [
                    vision.astype(self.language_model.embed_tokens.weight.dtype),
                    self.language_model.embed(input_ids),
                ],
                axis=1,
            )
        )

    get_input_embeddings._omlx_np_compat = True
    model_cls.get_input_embeddings = get_input_embeddings


def _patch_processor_placeholders(np_processing: Any, placeholder_id: int) -> None:
    processor_cls = np_processing.Processor
    original = processor_cls.__call__
    if getattr(original, "_omlx_np_compat", False):
        return

    def patched_call(self, *args, **kwargs):
        data = original(self, *args, **kwargs)
        images = kwargs.get("images", args[0] if args else None)
        text = kwargs.get("text", args[1] if len(args) > 1 else None)
        if (
            images is None
            or text is None
            or "image_sizes" not in data
            or "input_ids" not in data
        ):
            return data

        # One placeholder per visual-prefix embedding: patch rows x patch
        # columns from the neck, plus the one summary token.
        height, width = (int(v) for v in data["image_sizes"][0])
        count = (height // 16) * (width // 64) + 1
        ids = np.asarray(data["input_ids"])
        data["input_ids"] = np.concatenate(
            [
                np.full((ids.shape[0], count), placeholder_id, ids.dtype),
                ids,
            ],
            axis=1,
        )
        mask = data.get("attention_mask")
        if mask is not None:
            mask = np.asarray(mask)
            data["attention_mask"] = np.concatenate(
                [np.ones((mask.shape[0], count), mask.dtype), mask], axis=1
            )
        return data

    patched_call._omlx_np_compat = True
    processor_cls.__call__ = patched_call


__all__ = ["apply_mlx_vlm_nemotron_parse_compat_patch", "is_applied"]
