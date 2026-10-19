# SPDX-License-Identifier: Apache-2.0
"""Load Qwen3.5 checkpoints saved from transformers' base ``Qwen3_5Model``.

Decision 3.0 (vllm-sr d3) checkpoints differ from the
``Qwen3_5ForConditionalGeneration`` layout that mlx-vlm reads:

- Weights are named ``language_model.*`` and ``visual.*``.
- The base model has no LM head, also when ``tie_word_embeddings`` is false.
- ``text_config.layer_types`` may not follow ``full_attention_interval``
  (d3-edge: ``L L F L L F L L F F F F``).

Each hook keeps checkpoints in the usual layout unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

BASE_ARCHITECTURE = "Qwen3_5Model"
_LANGUAGE_PREFIX = "language_model."
# Names that already follow mlx-vlm's layout under ``language_model.``.
_SANITIZED_LANGUAGE_CHILDREN = ("model.", "lm_head", "mtp.")

_APPLIED = False


def base_layout_key(key: str) -> str:
    """Map a base-model weight name to the ``ForConditionalGeneration`` name."""
    if key.startswith("visual."):
        return "model." + key
    if key.startswith(_LANGUAGE_PREFIX) and not key[len(_LANGUAGE_PREFIX) :].startswith(
        _SANITIZED_LANGUAGE_CHILDREN
    ):
        return "model." + key
    return key


def needs_base_compat(config: dict) -> bool:
    """True for a base-model checkpoint or a non-periodic layer schedule."""
    if config.get("architectures") == [BASE_ARCHITECTURE]:
        return True
    text_config = config.get("text_config") or {}
    layer_types = text_config.get("layer_types")
    interval = text_config.get("full_attention_interval", 4)
    if not layer_types or not interval:
        return False
    periodic = [
        "linear_attention" if (i + 1) % interval else "full_attention"
        for i in range(len(layer_types))
    ]
    return list(layer_types) != periodic


def apply_mlx_vlm_qwen35_base_compat_patch() -> bool:
    """Install the hooks on mlx-vlm's qwen3_5 model; True on first install."""
    global _APPLIED
    if _APPLIED:
        return False
    try:
        from mlx_vlm.models.qwen3_5 import config as q35_config
        from mlx_vlm.models.qwen3_5 import language as q35_language
        from mlx_vlm.models.qwen3_5 import qwen3_5 as q35_model
    except Exception as exc:  # noqa: BLE001
        logger.debug("mlx-vlm qwen3_5 not importable: %s", exc)
        return False

    _patch_sanitize_key(q35_model)
    _patch_configs(q35_config)
    _patch_layers(q35_language)
    _APPLIED = True
    return True


def _patch_sanitize_key(module: Any) -> None:
    original = module.sanitize_key

    def sanitize_key(key):
        return original(base_layout_key(key))

    module.sanitize_key = sanitize_key


def _with_layer_types(text_config: Any, params: Any) -> Any:
    layer_types = params.get("layer_types") if isinstance(params, dict) else None
    if layer_types and not isinstance(text_config, dict):
        text_config.layer_types = list(layer_types)
    return text_config


def _patch_configs(module: Any) -> None:
    text_cls = module.TextConfig
    model_cls = module.ModelConfig
    text_from_dict = text_cls.from_dict.__func__
    model_from_dict = model_cls.from_dict.__func__

    # mlx-vlm drops config keys that are not dataclass fields, and
    # ``load_model`` rebuilds ``text_config`` through ``TextConfig.from_dict``.
    def text_config_from_dict(cls, params):
        return _with_layer_types(text_from_dict(cls, params), params)

    def model_config_from_dict(cls, params):
        config = model_from_dict(cls, params)
        _with_layer_types(config.text_config, params.get("text_config"))
        if params.get("architectures") == [BASE_ARCHITECTURE]:
            config.headless = True
        return config

    text_cls.from_dict = classmethod(text_config_from_dict)
    model_cls.from_dict = classmethod(model_config_from_dict)


def _patch_layers(language: Any) -> None:
    layer_cls = language.Qwen3_5DecoderLayer
    model_cls = language.Qwen3_5Model
    lm_cls = language.LanguageModel
    layer_init = layer_cls.__init__
    model_init = model_cls.__init__
    lm_init = lm_cls.__init__

    def decoder_layer_init(self, args, layer_idx):
        layer_init(self, args, layer_idx)
        layer_types = getattr(args, "layer_types", None)
        if not layer_types:
            return
        is_linear = layer_types[layer_idx] == "linear_attention"
        if is_linear == self.is_linear:
            return
        if is_linear:
            del self.self_attn
            self.linear_attn = language.Qwen3_5GatedDeltaNet(args)
        else:
            del self.linear_attn
            self.self_attn = language.Qwen3_5Attention(args)
        self.is_linear = is_linear

    def text_model_init(self, args):
        model_init(self, args)
        if not getattr(args, "layer_types", None):
            return
        # The mask schedule reads one cache of each layer kind.
        kinds = [layer.is_linear for layer in self.layers]
        if False in kinds:
            self.fa_idx = kinds.index(False)
        if True in kinds:
            self.ssm_idx = kinds.index(True)

    def language_model_init(self, args, config=None):
        lm_init(self, args, config)
        if getattr(config, "headless", False) and "lm_head" in self:
            del self.lm_head

    layer_cls.__init__ = decoder_layer_init
    model_cls.__init__ = text_model_init
    lm_cls.__init__ = language_model_init
