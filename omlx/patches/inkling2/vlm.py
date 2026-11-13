# SPDX-License-Identifier: Apache-2.0
"""Serve Inkling 2 on the VLM engine.

The model is the mlx-lm module (``omlx.patches.inkling2.model``) with its
vision tower. ``Model`` adds the mlx-vlm interface the VLM engine calls:
``language_model``, ``get_input_embeddings`` and ``encode_image`` for the
pixel blocks and ``image_grid_thw`` of the Inkling 2 processor.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx_vlm.prompt_utils as prompt_utils
from mlx_lm.utils import load_model
from mlx_vlm.models.base import InputEmbeddingsFeatures, install_auto_processor_patch
from mlx_vlm.utils import get_model_path, load_processor

from . import apply_inkling2_patch
from . import model as inkling2
from .processing import Inkling2Processor

_MODEL_TYPE = "inkling2"
_UNSUPPORTED_INPUTS = ("pixel_values_videos", "input_values", "input_features")
_REGISTERED = False


def _register_mlx_vlm() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    install_auto_processor_patch(_MODEL_TYPE, Inkling2Processor)
    # Without a MODEL_CONFIG entry apply_chat_template drops the image parts.
    model_config = getattr(prompt_utils, "MODEL_CONFIG", None)
    message_format = getattr(prompt_utils, "MessageFormat", None)
    if isinstance(model_config, dict) and message_format is not None:
        model_config.setdefault(_MODEL_TYPE, message_format.LIST_WITH_IMAGE_FIRST)
    _REGISTERED = True


class _LanguageModel:
    """The text forward of ``Model`` with the mlx-vlm call signature.

    It is not a module, so the model tree holds the text layers once. A block
    drafter attached to this object is passed to the forward explicitly.
    """

    def __init__(self, model: Model):
        self._model = model

    def __call__(self, inputs, cache=None, inputs_embeds=None, **kwargs):
        drafter = self.__dict__.get("_omlx_drafter")
        return self._model.forward(
            inputs, cache, inputs_embeds, drafter=drafter, **kwargs
        )

    @property
    def args(self):
        return self._model.args.text

    def __getattr__(self, name):
        # model, layers, make_cache and the speculative rollback hooks.
        return getattr(self._model, name)


class Model(inkling2.Model):
    @property
    def language_model(self) -> _LanguageModel:
        return _LanguageModel(self)

    def encode_image(self, pixel_values, image_grid_thw=None, **kwargs):
        return super().encode_image(pixel_values, image_grid_thw)

    def get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        for name in _UNSUPPORTED_INPUTS:
            if kwargs.get(name) is not None:
                raise ValueError(f"Inkling 2 does not support {name} in oMLX.")
        embeds = self.embed_inputs(
            input_ids,
            pixel_values,
            kwargs.get("image_grid_thw"),
            features=kwargs.get("cached_image_features"),
        )
        return InputEmbeddingsFeatures(inputs_embeds=embeds)


def load(model_path: str | Path, *, lazy: bool = False):
    """Load an Inkling 2 checkpoint with mlx-lm and its processor."""
    apply_inkling2_patch()
    _register_mlx_vlm()
    path = get_model_path(str(model_path))
    model, config = load_model(
        path,
        lazy=lazy,
        get_model_classes=lambda config: (Model, inkling2.ModelArgs),
    )
    model.config = SimpleNamespace(
        model_type=model.model_type,
        image_token_id=model.args.image_token_id,
        eos_token_id=config.get("eos_token_id"),
        vocab_size=model.args.text.vocab_size,
        text_config=model.args.text,
        vision_config=model.args.vision,
    )
    processor = load_processor(path, True, eos_token_ids=model.config.eos_token_id)
    return model, processor
