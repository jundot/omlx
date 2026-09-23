# SPDX-License-Identifier: Apache-2.0
"""MiMo V2.6 image-sidecar loader and mlx-vlm adapter."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .audio import MiMoAudioBridge, MiMoAudioProcessor
from .vision import VisionConfig, VisionModel

VISION_SIDECAR = Path("omnimodal/vision_encoder.safetensors")
AUDIO_SIDECAR = Path("omnimodal/audio_encoder.safetensors")
AUDIO_TOKENIZER = Path("omnimodal/audio_tokenizer")
OMNIMODAL_CONFIG = Path("omnimodal/config.json")


def has_vision_sidecar(model_path: str | Path) -> bool:
    root = Path(model_path)
    return (root / VISION_SIDECAR).is_file() and (root / OMNIMODAL_CONFIG).is_file()


class MiMoLanguageAdapter(nn.Module):
    """Add an input-embedding path to mlx-lm's MiMo V2 Flash model."""

    def __init__(self, target: nn.Module):
        super().__init__()
        self.target = target
        self._accepts_input_embeddings = (
            "input_embeddings" in inspect.signature(target.__call__).parameters
        )

    @property
    def model(self):
        return self.target.model

    @property
    def args(self):
        return self.target.args

    @property
    def layers(self):
        return self.target.layers

    @property
    def _omlx_mtp_decode_enabled(self):
        return getattr(self.target, "_omlx_mtp_decode_enabled", False)

    @property
    def _omlx_mtp_chain(self):
        return getattr(self.target, "_omlx_mtp_chain", False)

    @property
    def _omlx_mtp_depth(self):
        return getattr(self.target, "_omlx_mtp_depth", 1)

    @property
    def _omlx_mtp_head_clone(self):
        return getattr(self.target, "_omlx_mtp_head_clone", False)

    @property
    def _omlx_mtp_head_prenorm(self):
        return getattr(self.target, "_omlx_mtp_head_prenorm", False)

    def make_cache(self):
        return self.target.make_cache()

    def __call__(
        self,
        input_ids: mx.array,
        cache=None,
        inputs_embeds: mx.array | None = None,
        **kwargs,
    ):
        return_hidden = bool(kwargs.pop("return_hidden", False))
        kwargs.pop("logits_keep", None)
        n_confirmed = int(kwargs.pop("n_confirmed", 0) or 0)
        target_kwargs = {"cache": cache}
        if return_hidden:
            target_kwargs["return_hidden"] = True
        if n_confirmed:
            target_kwargs["n_confirmed"] = n_confirmed
        if inputs_embeds is None:
            return self.target(input_ids, **target_kwargs)
        if self._accepts_input_embeddings:
            # The vendored ``mimo_v2`` model already exposes this seam.
            return self.target(
                input_ids,
                input_embeddings=inputs_embeds,
                **target_kwargs,
            )

        # mlx-lm's released ``mimo_v2_flash`` model does not expose an
        # embedding argument. Run the same language forward from the supplied
        # image-merged embeddings instead of looking them up again.
        from mlx_lm.models.base import create_attention_mask

        language_model = self.target.model
        if not all(
            hasattr(language_model, name)
            for name in ("layers", "norm", "ga_idx", "swa_idx", "sliding_window_size")
        ):
            raise TypeError(
                f"{type(self.target).__name__} cannot accept MiMo image embeddings"
            )
        if cache is None:
            cache = [None] * len(language_model.layers)
        full_mask = create_attention_mask(inputs_embeds, cache[language_model.ga_idx])
        sliding_mask = create_attention_mask(
            inputs_embeds,
            cache[language_model.swa_idx],
            window_size=language_model.sliding_window_size,
        )
        hidden_states = inputs_embeds
        for layer, layer_cache in zip(language_model.layers, cache):
            mask = sliding_mask if layer.is_sliding_window else full_mask
            hidden_states = layer(hidden_states, mask, cache=layer_cache)
        hidden = hidden_states
        hidden_states = language_model.norm(hidden_states)
        if getattr(self.target.args, "tie_word_embeddings", False):
            logits = language_model.embed_tokens.as_linear(hidden_states)
        else:
            logits = self.target.lm_head(hidden_states)
        if return_hidden:
            return logits, hidden
        return logits

    def get_mtp_module(self):
        getter = getattr(self.target, "get_mtp_module", None)
        return getter() if callable(getter) else getattr(self.target, "mtp", None)

    def make_mtp_cache(self):
        method = getattr(self.target, "make_mtp_cache", None)
        return method() if callable(method) else []

    def mtp_begin_cycle(self, *args, **kwargs):
        method = getattr(self.target, "mtp_begin_cycle", None)
        if callable(method):
            return method(*args, **kwargs)

    def mtp_forward(self, *args, **kwargs):
        return self.target.mtp_forward(*args, **kwargs)

    def rollback_speculative_cache(self, *args, **kwargs):
        return self.target.rollback_speculative_cache(*args, **kwargs)


class MiMoOmnimodalModel(nn.Module):
    """Compose mlx-lm's text model with MiMo's vision and audio towers."""

    def __init__(
        self,
        text_model: nn.Module,
        vision_model: VisionModel,
        audio_bridge: MiMoAudioBridge | None,
        config: dict,
    ):
        super().__init__()
        self.language_model = MiMoLanguageAdapter(text_model)
        self.vision_tower = vision_model
        self.visual = vision_model
        self.audio_bridge = audio_bridge
        self.config = SimpleNamespace(**config)

    @property
    def layers(self):
        return self.language_model.layers

    @staticmethod
    def _merge_image_features(
        input_ids: mx.array,
        inputs_embeds: mx.array,
        image_features: mx.array,
        image_token_id: int,
    ) -> mx.array:
        image_positions = input_ids == image_token_id
        expected = int(mx.sum(image_positions).item())
        if expected != image_features.shape[0]:
            raise ValueError(
                "MiMo image placeholder count does not match vision features: "
                f"{expected} placeholders, {image_features.shape[0]} features"
            )
        outputs = []
        offset = 0
        for batch_index in range(input_ids.shape[0]):
            mask = image_positions[batch_index]
            count = int(mx.sum(mask).item())
            if count == 0:
                outputs.append(inputs_embeds[batch_index])
                continue
            features = image_features[offset : offset + count]
            indices = mx.where(mask, mx.cumsum(mask.astype(mx.int32)) - 1, 0)
            replacements = features[indices]
            outputs.append(
                mx.where(mask[:, None], replacements, inputs_embeds[batch_index])
            )
            offset += count
        return mx.stack(outputs, axis=0)

    def get_input_embeddings(
        self,
        input_ids: mx.array,
        pixel_values: mx.array | None = None,
        **kwargs,
    ):
        from mlx_vlm.models.base import InputEmbeddingsFeatures

        inputs_embeds = self.language_model.target.model.embed_tokens(input_ids)
        if pixel_values is not None:
            image_grid_thw = kwargs.get("image_grid_thw")
            if image_grid_thw is None:
                raise ValueError("MiMo image requests require image_grid_thw")
            image_features = kwargs.get("cached_image_features")
            if image_features is None:
                image_features = self.vision_tower(pixel_values, image_grid_thw)
            inputs_embeds = self._merge_image_features(
                input_ids,
                inputs_embeds,
                image_features,
                int(self.config.image_token_id),
            )

        audio_codes = kwargs.get("audio_codes")
        if audio_codes is not None:
            if self.audio_bridge is None:
                raise ValueError("MiMo audio sidecar is not loaded")
            audio_features = self.audio_bridge(audio_codes)
            inputs_embeds = self._merge_image_features(
                input_ids,
                inputs_embeds,
                audio_features,
                int(self.config.audio_token_id),
            )
        return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)


def _load_image_only_processor(
    model_path: str | Path,
    eos_token_ids: int | list[int] | None,
):
    """Load Qwen's PIL image processor without requiring PyTorch for video."""
    from mlx_vlm.tokenizer_utils import load_tokenizer
    from mlx_vlm.utils import StoppingCriteria
    from transformers import (
        AutoTokenizer,
        Qwen2VLImageProcessorPil,
        Qwen2VLProcessor,
    )
    from transformers.video_processing_utils import BaseVideoProcessor

    class _UnusedVideoProcessor(BaseVideoProcessor):
        def __call__(self, *args, **kwargs):
            del args, kwargs
            raise ValueError("MiMo video processing is not available yet")

    class _ImageOnlyQwen2VLProcessor(Qwen2VLProcessor):
        def check_argument_for_proper_class(self, argument_name, argument):
            if argument_name == "video_processor":
                return BaseVideoProcessor
            return super().check_argument_for_proper_class(argument_name, argument)

    root = Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(root))
    processor = _ImageOnlyQwen2VLProcessor(
        image_processor=Qwen2VLImageProcessorPil.from_pretrained(str(root)),
        tokenizer=tokenizer,
        video_processor=_UnusedVideoProcessor(),
        chat_template=tokenizer.chat_template,
    )
    detokenizer_class = load_tokenizer(root, return_tokenizer=False)
    processor.detokenizer = detokenizer_class(tokenizer)
    tokenizer.stopping_criteria = StoppingCriteria(
        eos_token_ids,
        tokenizer,
        additional_eos_token_ids=getattr(processor, "additional_eos_token_ids", ()),
    )
    return processor


def load(
    model_path: str | Path,
    *,
    model_settings: Any | None = None,
    trust_remote_code: bool = False,
):
    """Load the text checkpoint, vision sidecar, and Qwen-compatible processor."""
    root = Path(model_path)
    if not has_vision_sidecar(root):
        raise FileNotFoundError(
            f"MiMo vision sidecar not found below {root / 'omnimodal'}"
        )

    from mlx_vlm.utils import load_processor

    from ...utils.model_loading import load_text_model

    root_config = json.loads((root / "config.json").read_text())
    sidecar_config = json.loads((root / OMNIMODAL_CONFIG).read_text())
    vision_values = sidecar_config.get("vision_config")
    if not isinstance(vision_values, dict):
        raise ValueError("MiMo omnimodal config has no vision_config")

    text_model, tokenizer = load_text_model(
        str(root),
        model_settings=model_settings,
    )
    vision_model = VisionModel(VisionConfig.from_dict(vision_values))
    vision_weights = mx.load(str(root / VISION_SIDECAR))
    if not isinstance(vision_weights, dict):
        raise ValueError("MiMo vision sidecar must contain named tensors")
    vision_weights = vision_model.sanitize(vision_weights)
    vision_model.load_weights(list(vision_weights.items()), strict=True)
    mx.eval(vision_model.parameters())
    vision_model.eval()

    audio_bridge = None
    if (root / AUDIO_SIDECAR).is_file():
        audio_bridge = MiMoAudioBridge.load(root / AUDIO_SIDECAR)

    eos_ids = root_config.get("eos_token_id")
    try:
        processor = load_processor(
            root,
            eos_token_ids=eos_ids,
            trust_remote_code=trust_remote_code,
        )
    except ImportError as exc:
        # Transformers 5 loads Qwen's video processor even for image-only use.
        # Its default video backend requires PyTorch/torchvision, which oMLX
        # intentionally does not depend on. Stage 1 uses the equivalent PIL
        # image processor and a disabled video slot instead.
        if "VideoProcessor" not in str(exc):
            raise
        processor = _load_image_only_processor(root, eos_ids)
    # AutoProcessor owns its tokenizer; keep the already-loaded tokenizer only
    # as a fallback for processor implementations that expose none.
    if not hasattr(processor, "tokenizer"):
        processor.tokenizer = tokenizer
    if audio_bridge is not None:
        tokenizer_root = root / AUDIO_TOKENIZER
        if not (tokenizer_root / "model.safetensors").is_file():
            raise FileNotFoundError(
                f"MiMo audio tokenizer not found below {tokenizer_root}"
            )
        processor = MiMoAudioProcessor(processor, tokenizer_root)

    model = MiMoOmnimodalModel(text_model, vision_model, audio_bridge, root_config)
    model.eval()
    return model, processor
